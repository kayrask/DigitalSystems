import os
import traceback
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

try:
    from api.models.bisenet import BiSeNet
except Exception:  # pragma: no cover
    print("[FaceParser] FAILED to import BiSeNet class:")
    traceback.print_exc()
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
        if not ckpt_path:
            print("[FaceParser] WARNING: no ckpt_path provided — skin masking is DISABLED "
                  "(skin_mask() will return an all-skin mask, i.e. no constraint at all).")
            return
        if BiSeNet is None:
            print("[FaceParser] WARNING: BiSeNet class failed to import — skin masking is DISABLED. "
                  "See import error printed at startup.")
            return
        if not os.path.exists(ckpt_path):
            print(f"[FaceParser] WARNING: checkpoint not found at '{ckpt_path}' — skin masking is DISABLED "
                  "(skin_mask() will return an all-skin mask, i.e. no constraint at all).")
            return
        try:
            self.model = BiSeNet(n_classes=19)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            self.model.to(DEVICE).eval()
        except Exception:
            self.model = None
            print(f"[FaceParser] ERROR: failed to load BiSeNet checkpoint '{ckpt_path}' — "
                  "skin masking is DISABLED and skin_mask() will silently return an all-skin mask "
                  "(no constraint at all) until this is fixed. Full traceback:")
            traceback.print_exc()

    def skin_mask(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        if self.model is None:
            return np.ones((h, w), dtype=np.uint8) * 255

        x = _tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(x)[0]
            parsing = out.argmax(1).cpu().numpy()[0]

        # model outputs at 512x512; resize mask back to the original face size
        mask_small = np.isin(parsing, list(SKIN_LABELS)).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_small, mode="L").resize((w, h), Image.NEAREST)
        return np.asarray(mask_img, dtype=np.uint8)


def apply_skin_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(img).copy()
    neutral = np.full_like(arr, 128)
    m = mask[..., None] > 0
    out = np.where(m, arr, neutral)
    return Image.fromarray(out)
