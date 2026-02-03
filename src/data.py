import os
from typing import List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SkinDataset(Dataset):
    def __init__(self, metadata_csv: str, images_root: str, classes: List[str]=None, split: str='train'):
        self.df = pd.read_csv(metadata_csv)
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split']==split].reset_index(drop=True)
        self.images_root = images_root

        # derive classes if not provided
        if classes is None:
            labels = set()
            for x in self.df['labels']:
                for t in str(x).split(','):
                    t = t.strip()
                    if t:
                        labels.add(t)
            self.classes = sorted(labels)
        else:
            self.classes = classes

        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        if split == 'train':
            self.tfm = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ColorJitter(0.25,0.25,0.25,0.1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=12, translate=(0.05,0.05), scale=(0.9,1.1)),
                transforms.ToTensor()
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row['image'])
        img = Image.open(img_path).convert('RGB')
        y = torch.zeros(len(self.classes), dtype=torch.float32)
        for t in str(row['labels']).split(','):
            t = t.strip()
            if t in self.class_to_idx:
                y[self.class_to_idx[t]] = 1.0
        x = self.tfm(img)
        return x, y
