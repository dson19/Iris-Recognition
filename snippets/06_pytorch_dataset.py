"""
SNIPPET 06 — PyTorch Dataset
Hai loại Dataset dùng trong project:
  A. IrisSegDataset  — cặp (ảnh, mask) cho U-Net
  B. NormalizedIrisDataset — ảnh normalized theo folder=class cho ArcFace
"""

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── A. Dataset cho Segmentation (U-Net) ──────────────────────────────────

class IrisSegDataset(Dataset):
    def __init__(self, pairs, transform=None):
        # pairs: [(img_path, mask_path), ...]
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)  # (H, W)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (H, W) values 0/1/2

        if self.transform:
            # Albumentations augment cả image lẫn mask cùng lúc
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]   # tensor (1, H, W)
            mask = aug["mask"]    # tensor (H, W)

        return img, mask.long()  # long() vì CrossEntropyLoss cần int64


# ── B. Dataset cho Classification (ArcFace) ──────────────────────────────

class NormalizedIrisDataset(Dataset):
    """
    Cấu trúc thư mục:
      root/
        001_L/  ← class identity
          img1.png
          img2.png
        001_R/
          ...
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []  # [(img_path, class_idx), ...]

        class_dirs        = sorted(d for d in Path(root).iterdir() if d.is_dir())
        self.classes      = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# ── Transform + DataLoader ────────────────────────────────────────────────

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

dataset = NormalizedIrisDataset("datasets/train", transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Duyệt qua loader trong training loop:
for imgs, labels in loader:
    # imgs  : (B, 1, H, W) float tensor
    # labels: (B,) int tensor
    pass
