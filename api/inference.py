# api/inference.py
import torch
import json
import os
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]

class SkinModel:
    def __init__(self, model_path, thresholds_path=None, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # Model definition (same as training)
        self.model = self._build_model(num_classes=len(LABELS))
        ckpt = torch.load(model_path, map_location=self.device)

        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            raise ValueError("Checkpoint must contain 'model' or 'state_dict'.")

        self.model.to(self.device).eval()

        # Load thresholds
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {lbl: 0.5 for lbl in LABELS}

        # Optional: tone-aware thresholds (safe fallback)
        self.thresholds_by_tone = None
        self.tone_conf_min = 1.0

        tone_thr_path = os.path.join("models", "per_class_thresholds_by_tone.json")
        if os.path.exists(tone_thr_path):
            try:
                with open(tone_thr_path, "r") as f:
                    j = json.load(f)
                self.thresholds_by_tone = j.get("thresholds", None)
                self.tone_conf_min = float(j.get("tone_conf_min", 1.0))
            except Exception as e:
                print("Failed to load tone thresholds:", e)
                self.thresholds_by_tone = None
                self.tone_conf_min = 1.0

        # transforms (same as eval)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _build_model(self, num_classes):
        import torchvision.models as models
        m = models.resnet18(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    def predict(self, image: Image.Image, tone_group: str | None = None, tone_conf: float | None = None):
        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Pick thresholds: tone-aware if available + confident, else default
        thr_map = self.thresholds
        if (
            self.thresholds_by_tone
            and tone_group
            and tone_conf is not None
            and tone_conf >= self.tone_conf_min
            and tone_group in self.thresholds_by_tone
        ):
            thr_map = self.thresholds_by_tone[tone_group]
        elif self.thresholds_by_tone and "default" in self.thresholds_by_tone:
            thr_map = self.thresholds_by_tone["default"]

        result = {}
        for i, label in enumerate(LABELS):
            result[label] = {
                "probability": float(probs[i]),
                "prediction": int(probs[i] >= float(thr_map.get(label, 0.5)))
            }

        return result

class SkinTypeModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes = ckpt.get(
            "classes",
            ["combination", "dry", "normal", "oily", "sensitive"],
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features, len(self.classes)
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, pil_image: Image.Image):
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        return {
            "skin_type": self.classes[idx],
            "confidence": float(probs[idx]),
            "probs": {
                self.classes[i]: float(probs[i])
                for i in range(len(self.classes))
            },
        }

class SkinToneModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes = ckpt.get("classes", ["Black", "Brown", "White"])

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, pil_image: Image.Image):
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        return {
            "group": self.classes[idx],
            "confidence": float(probs[idx]),
            "probs": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
        }
