#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SkinDefectsDataset(Dataset):
    """
    Expects CSV with columns: filepath,label,subject_id,split
    """
    def __init__(self, csv_path, split="train", transform=None, label_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

        # build label map once if not provided
        if label_to_idx is None:
            classes = sorted(self.df["label"].unique())
            self.label_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.label_to_idx = label_to_idx

        self.classes = [c for c, _ in sorted(self.label_to_idx.items(), key=lambda x: x[1])]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = self.label_to_idx[row["label"]]
        return img, y
