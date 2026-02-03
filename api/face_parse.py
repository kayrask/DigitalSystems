import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

try:
    from api.models.bisenet import BiSeNet
except Exception:  # pragma: no cover
    BiSeNet = None

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Face parsing labels (standard)
# 1: skin, 2: left brow, 3: right brow, 4: left eye, 5: right eye
# 10: nose, 11: upper lip, 12: lower lip, 13: hair
SKIN_LABELS = {1, 10}  # skin + nose

_tf = T.Compose(
    [
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FaceParser:
    def __init__(self, ckpt_path: str | None):
        self.model = None
        if not ckpt_path or not os.path.exists(ckpt_path) or BiSeNet is None:
            return
        try:
            self.model = BiSeNet(n_classes=19)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            self.model.to(DEVICE).eval()
        except Exception:
            self.model = None

    def skin_mask(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        if self.model is None:
            return np.ones((h, w), dtype=np.uint8) * 255

        x = _tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(x)[0]
            parsing = out.argmax(1).cpu().numpy()[0]

        mask = np.isin(parsing, list(SKIN_LABELS)).astype(np.uint8) * 255
        return mask


def apply_skin_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(img).copy()
    neutral = np.full_like(arr, 128)
    m = mask[..., None] > 0
    out = np.where(m, arr, neutral)
    return Image.fromarray(out)
