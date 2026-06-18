"""
BÀI 06 — PyTorch Dataset
Mục tiêu: Viết Dataset class để load dữ liệu vào training loop.

Kiến thức cần: torch.utils.data.Dataset, __len__, __getitem__
Đáp án tham khảo: snippets/06_pytorch_dataset.py
"""

import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Phần A: Dataset cho ArcFace ───────────────────────────────────────────
# Mỗi subfolder trong root = 1 class identity

class NormalizedIrisDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []  # sẽ chứa [(img_path, class_idx), ...]

        # Bước 1: Liệt kê tất cả subfolder (mỗi folder = 1 class)
        # TODO: class_dirs = list các thư mục con trong root, sort theo tên
        class_dirs = ...

        # Bước 2: Tạo mapping tên class → index số
        # TODO: self.classes = list tên các folder
        self.classes = ...
        # TODO: self.class_to_idx = dict {tên: index}
        self.class_to_idx = ...

        # Bước 3: Thu thập tất cả ảnh kèm class index
        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            # TODO: duyệt qua tất cả file .png trong cls_dir
            # append (img_path, idx) vào self.samples
            for img_path in ...:
                self.samples.append(...)

    def __len__(self):
        # TODO: trả về số lượng samples
        return ...

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # TODO: đọc ảnh grayscale bằng cv2.imread
        img = ...

        # TODO: áp transform nếu có
        if self.transform:
            img = ...

        return img, label


# ── Phần B: Transform ──────────────────────────────────────────────────────

# TODO: tạo transform với Normalize(mean=(0.5,), std=(0.5,)) và ToTensorV2()
transform = A.Compose([
    ...,
    ...,
])


# ── Phần C: DataLoader ────────────────────────────────────────────────────

dataset = NormalizedIrisDataset("datasets/train", transform=transform)

# TODO: tạo DataLoader với batch_size=32, shuffle=True, num_workers=4
loader = DataLoader(...)

# Kiểm tra
for imgs, labels in loader:
    print(f"imgs shape : {imgs.shape}")   # (32, 1, 64, 512)
    print(f"labels shape: {labels.shape}") # (32,)
    break
