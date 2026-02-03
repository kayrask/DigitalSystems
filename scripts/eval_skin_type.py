import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import numpy as np


MODEL_PATH = "models/skin_type_resnet18.pt"
DATA_DIR = "data/skin_type_dataset/test"
BATCH_SIZE = 32
IMG_SIZE = 224


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_ds = ImageFolder(DATA_DIR, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("Classes:", classes)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
