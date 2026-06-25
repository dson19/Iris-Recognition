"""
Train iris segmentation model using segmentation_models_pytorch.
Architecture: U-Net with ResNet34 encoder pretrained on ImageNet.

Dataset structure expected:
  datasets/preprocessed/<subject>/<eye>/<img>.jpg    ← images
  datasets/segmentation_masks/<subject>/<eye>/<img>_mask.png ← masks

Usage:
  pip install segmentation-models-pytorch albumentations
  python scripts/train_unet.py
"""

import os
import random
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

SEED = 42
NUM_CLASSES = 3  # 0=background, 1=iris, 2=pupil

# Masks are stored on disk at 0/122/255 (easy to view), but the model needs
# contiguous class ids 0/1/2 for CrossEntropyLoss.
MASK_VALUE_TO_CLASS = {0: 0, 122: 1, 255: 2}


def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for pixel_val, class_id in MASK_VALUE_TO_CLASS.items():
        out[mask == pixel_val] = class_id
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IrisSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], transform=None):
        self.pairs = pairs      # [(img_path, mask_path), ...]
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Masks stored at 0/122/255 for easy viewing → remap to class ids 0/1/2
        mask = remap_mask(mask)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.long()


def find_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Match preprocessed images to their labeled masks."""
    pairs = []
    for mask_path in sorted(mask_dir.rglob("*_mask.png")):
        # mask filename: S1001L01_mask.png → image: S1001L01.jpg
        img_stem = mask_path.stem.replace("_mask", "")
        rel = mask_path.relative_to(mask_dir).parent
        img_path = img_dir / rel / (img_stem + ".jpg")
        if img_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"  [WARN] No image for mask: {mask_path.name}")
    return pairs


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def get_train_transform():
    return A.Compose([
        # NOTE: không dùng RandomRotate90 — nó hoán đổi H↔W trên ảnh không vuông
        # (280×320 → 320×280) khiến DataLoader không stack được batch.
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Returns mean Dice per class (excluding background)."""
    scores = []
    for cls in range(1, num_classes):  # skip background
        pred_c = (preds == cls).float()
        tgt_c = (targets == cls).float()
        intersection = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        scores.append(dice.item())
    return scores  # [iris_dice, pupil_dice]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="  train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_dice_iris, all_dice_pupil = [], []

    for imgs, masks in tqdm(loader, desc="  val  ", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        for pred, tgt in zip(preds, masks):
            d = dice_score_per_class(pred, tgt, NUM_CLASSES)
            all_dice_iris.append(d[0])
            all_dice_pupil.append(d[1])

    return (
        total_loss / len(loader),
        np.mean(all_dice_iris),
        np.mean(all_dice_pupil),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    pairs = find_pairs(args.img_dir, args.mask_dir)
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found.\n"
                           f"  img_dir : {args.img_dir}\n"
                           f"  mask_dir: {args.mask_dir}")
    print(f"Total pairs: {len(pairs)}")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_ds = IrisSegDataset(train_pairs, get_train_transform())
    val_ds   = IrisSegDataset(val_pairs,   get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=1,
        classes=NUM_CLASSES,
    ).to(device)
    print(f"Model: U-Net + {args.encoder} (pretrained ImageNet)")

    # --- Training ---
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, dice_iris, dice_pupil = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        mean_dice = (dice_iris + dice_pupil) / 2
        improved = mean_dice > best_dice

        if improved:
            best_dice = mean_dice
            torch.save(model.state_dict(), args.output_dir / "unet_best.pth")

        log = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "dice_iris": round(dice_iris, 4), "dice_pupil": round(dice_pupil, 4),
        }
        history.append(log)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"dice_iris={dice_iris:.4f}  dice_pupil={dice_pupil:.4f}"
              + ("  ← best" if improved else ""))

    # Save training history
    with open(args.output_dir / "unet_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest mean Dice: {best_dice:.4f}")
    print(f"Checkpoint: {args.output_dir / 'unet_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    type=Path, default=Path("datasets/preprocessed"))
    parser.add_argument("--mask_dir",   type=Path, default=Path("datasets/segmentation_masks"))
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--encoder",    type=str,  default="resnet34")
    parser.add_argument("--epochs",     type=int,  default=40)
    parser.add_argument("--batch_size", type=int,  default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val_split",  type=float, default=0.15,
                        help="Fraction of pairs used for validation")
    args = parser.parse_args()
    main(args)
