from __future__ import annotations

from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO


class YoloSkinDetector:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        
        # Debug: verify we loaded the right model
        print(f"[YOLO] Loaded weights: {weights_path}")
        print(f"[YOLO] Model classes: {self.model.names}")
        print(f"[YOLO] Number of classes: {len(self.model.names)}")

    def predict(
        self,
        pil_img: Image.Image,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 512,
        max_det: int = 100,
    ) -> List[Dict[str, Any]]:
        res = self.model.predict(
            pil_img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )[0]

        boxes: List[Dict[str, Any]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return boxes

        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cls = int(b.cls[0].cpu().numpy())
            score = float(b.conf[0].cpu().numpy())
            label = self.model.names.get(cls, f"class_{cls}")
            boxes.append({
                "cls": cls,
                "label": label,
                "conf": score,
                "box": [x1, y1, x2, y2],
            })
        return boxes
