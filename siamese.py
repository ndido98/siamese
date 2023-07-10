import numpy as np
import cv2 as cv
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import tensorflow as tf
from deepface import DeepFace


WEIGHTS_URL = "https://miatbiolab.csr.unibo.it/wp-content/uploads/2023/siamese-997d73a5.ckpt"


class Siamese(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.smad = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )
        self.arcface = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )
        self.select = torch.nn.Linear(4, 2)

    def forward(self, smad: torch.Tensor, arcface: torch.Tensor) -> torch.Tensor:
        smad_doc, smad_live = smad[0, ...], smad[1, ...]
        arcface_doc, arcface_live = arcface[0, ...], arcface[1, ...]
        smad_cat = torch.cat((smad_doc, smad_live), dim=0).unsqueeze(0)
        smad_decision = self.smad(smad_cat)
        smad_decision = torch.nn.functional.softmax(smad_decision, dim=1)
        arcface_cat = (arcface_doc - arcface_live).unsqueeze(0)
        arcface_decision = self.arcface(arcface_cat)
        arcface_decision = torch.nn.functional.softmax(arcface_decision, dim=1)
        decision = torch.cat((smad_decision, arcface_decision), dim=1)
        decision = self.select(decision)
        return decision[:, 1] - decision[:, 0]


def _crop_face(image_rgb: np.ndarray, mtcnn: MTCNN) -> np.ndarray:
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected.")
    biggest_box = np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
    box = boxes[biggest_box].astype(int)
    x1, y1, x2, y2 = (
        max(0, box[0]),
        max(0, box[1]),
        min(image_rgb.shape[1], box[2]),
        min(image_rgb.shape[0], box[3]),
    )
    cropped = image_rgb[y1:y2, x1:x2]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError("No face detected.")
    return cropped


def _preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    # Resize
    new_size = (299, 299)
    old_size = image_rgb.shape[:2]
    scale_factor = min(n / o for n, o in zip(new_size, old_size))
    rescaled = cv.resize(image_rgb, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    if rescaled.shape[0] == 0 or rescaled.shape[1] == 0:
        raise ValueError("Rescaling failed.")
    top_bottom, left_right = tuple(d - s for d, s in zip(new_size, rescaled.shape[:2]))
    top = top_bottom // 2
    bottom = top_bottom - top
    left = left_right // 2
    right = left_right - left
    resized = cv.copyMakeBorder(rescaled, top, bottom, left, right, cv.BORDER_CONSTANT, (0, 0, 0))
    # To float
    resized = resized.astype(np.float32) / 255.0
    # Normalize
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (resized - mean) / std
    # To tensor
    chw = torch.from_numpy(normalized.transpose((2, 0, 1)))
    if chw.ndim != 3:
        raise ValueError(f"Invalid image ndim: expected 3, got {chw.ndim}.")
    return chw


def _pytorch_to_tf2_device(device: torch.device) -> str:
    if device.type == "cpu":
        return "/cpu:0"
    elif device.type == "cuda":
        return f"/gpu:{device.index}"
    else:
        raise ValueError(f"Invalid device type: expected 'cpu' or 'cuda', got '{device.type}'.")


def _get_arcface_features(images: list[np.ndarray], device: str | torch.device = "cpu") -> torch.Tensor:
    tf_device = _pytorch_to_tf2_device(device)
    feats = []
    target_size = (112, 112, 3)
    with tf.device(tf_device):
        for image in images:
            # HACK: DeepFace does not pad the images anymore before passing them to the model
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            factor = min(target_size[i] / image.shape[i] for i in range(2))
            dsize = (int(image.shape[1] * factor), int(image.shape[0] * factor))
            image = cv.resize(image, dsize)
            diff_0, diff_1 = target_size[0] - image.shape[0], target_size[1] - image.shape[1]
            image = np.pad(image, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant")
            if image.shape != target_size:
                image = cv.resize(image, target_size[:2])
            image = image.astype(np.float32) / 255.0
            feat = DeepFace.represent(
                image,
                model_name="ArcFace",
                detector_backend="skip",
            )[0]["embedding"]
            feats.append(feat)
    return torch.tensor(feats, dtype=torch.float32, device=device)


def _get_couple_prediction(
    document_bgr: np.ndarray,
    live_bgr: np.ndarray,
    device: torch.device,
    mtcnn: MTCNN,
    smad_extractor: torch.nn.Module,
    siamese: torch.nn.Module,
) -> float:
    # Convert both images from BGR to RGB
    document_rgb = cv.cvtColor(document_bgr, cv.COLOR_BGR2RGB)
    live_rgb = cv.cvtColor(live_bgr, cv.COLOR_BGR2RGB)
    # Detect faces
    document_face = _crop_face(document_rgb, mtcnn)
    live_face = _crop_face(live_rgb, mtcnn)
    # Extract SMAD features
    with torch.no_grad():
        bchw = torch.stack([
            _preprocess_image(document_face),
            _preprocess_image(live_face),
        ])
        device_img = bchw.to(device)
        smad_features: torch.Tensor = smad_extractor(device_img)
        if smad_features.shape != (2, 512):
            raise ValueError(f"Invalid SMAD features shape: expected (2, 512), got {smad_features.shape}.")
    arcface_features = _get_arcface_features([document_face, live_face], device)
    if arcface_features.shape != (2, 512):
        raise ValueError(f"Invalid ArcFace features shape: expected (2, 512), got {arcface_features.shape}.")
    # Compute the final score
    with torch.no_grad():
        logits = siamese(smad_features, arcface_features)
        score = torch.sigmoid(logits).cpu().item()
    return score


def get_prediction(
    document_bgr: np.ndarray | list[np.ndarray],
    live_bgr: np.ndarray | list[np.ndarray],
    device: str | torch.device = "cpu",
) -> float | list[float]:
    """
    Get the prediction score(s) for the given document and live image(s).
    If two lists of images of equal length are passed as input, the output will be a list of corresponding scores.

    :param document_bgr: The document image(s) in BGR format.
    :param live_bgr: The live image(s) in BGR format.
    :param device: The device to use for the prediction. Can be either a string representing the device or a torch.device object.
    :return: The prediction score(s).
    """

    if isinstance(document_bgr, list) and isinstance(live_bgr, list) and len(live_bgr) != len(document_bgr):
        raise ValueError(f"Invalid number of images: expected {len(document_bgr)}, got {len(live_bgr)}.")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(document_bgr, np.ndarray):
        document_bgr = [document_bgr]
    if isinstance(live_bgr, np.ndarray):
        live_bgr = [live_bgr]
    # Download the Siamese weights
    state_dict = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", check_hash=True)
    # Load the Siamese network
    siamese = Siamese().eval()
    siamese.load_state_dict(state_dict["siamese"])
    siamese = siamese.to(device)
    # Load the SMAD extractor
    feature_extractor = InceptionResnetV1(
        pretrained=None,
        classify=True,
        num_classes=1,
        dropout_prob=0.6,
    ).eval()
    feature_extractor.logits = torch.nn.Identity()
    feature_extractor.load_state_dict(state_dict["smad"])
    feature_extractor = feature_extractor.to(device)
    # Load the MTCNN face detector
    mtcnn = MTCNN(select_largest=True, device=device)
    # Compute the prediction(s)
    scores = [
        _get_couple_prediction(doc, live, device, mtcnn, feature_extractor, siamese)
        for doc, live in zip(document_bgr, live_bgr)
    ]
    return scores if len(scores) > 1 else scores[0]
