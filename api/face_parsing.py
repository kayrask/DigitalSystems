import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

# Classes we want to KEEP (skin only)
SKIN_LABELS = {1}  # skin = 1 in CelebAMask-HQ


class FaceParser:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        try:
            self.model = torch.hub.load(
                "zllrunning/face-parsing.PyTorch",
                "BiSeNet",
                pretrained=True,
                trust_repo=True,
                force_reload=True,
            ).to(device).eval()
        except Exception:
            self.model = None

        self.tf = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def skin_mask(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        if self.model is None:
            return np.ones((h, w), dtype=np.uint8) * 255
        x = self.tf(img).unsqueeze(0).to(self.device)
        out = self.model(x)[0]
        parsing = out.argmax(0).cpu().numpy()

        mask = np.isin(parsing, list(SKIN_LABELS)).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask


def apply_skin_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(img).copy()
    neutral = np.full_like(arr, 128)
    m = mask[..., None] > 0
    out = np.where(m, arr, neutral)
    return Image.fromarray(out)
