"""
Train ResNet50 + ArcFace for iris identity embedding.

Input : normalized iris images (64×512 grayscale) from datasets/train/
Output: 256-D L2-normalized embedding vectors

Usage:
  python scripts/train_arcface.py
"""

import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

SEED = 42
EMBED_DIM = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NormalizedIrisDataset(Dataset):
    """Loads normalized 64×512 iris images from train/<class_name>/*.png"""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []   # [(img_path, class_idx)]
        self.classes = []

        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, idx))

        print(f"Dataset: {len(self.classes)} classes, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def get_train_transform():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IrisEmbedder(nn.Module):
    """ResNet50 backbone → 256-D L2-normalized embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Adapt first conv: grayscale (1ch) input — average ImageNet weights across RGB
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: B×2048×1×1
        self.embed = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x: B×1×H×W
        # ResNet expects 3-channel; we patched conv1 to accept 1-channel
        feat = self.backbone(x).flatten(1)   # B×2048
        emb  = self.bn(self.embed(feat))     # B×256
        return F.normalize(emb, p=2, dim=1)  # L2 normalize


# ---------------------------------------------------------------------------
# ArcFace Loss
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM,
                 margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Normalize class weights
        w = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity: B × num_classes
        cos_theta = embeddings @ w.T
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # Add angular margin to the ground-truth class
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        theta_m = theta + one_hot * self.margin
        logits = torch.cos(theta_m) * self.scale

        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, head, loader, optimizer, device):
    model.train()
    head.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        emb = model(imgs)
        loss = head(emb, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Accuracy proxy: nearest class weight
        with torch.no_grad():
            w = F.normalize(head.weight, p=2, dim=1)
            preds = (emb @ w.T).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    correct, total = 0, 0

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        emb = model(imgs)
        w = F.normalize(head.weight, p=2, dim=1)
        preds = (emb @ w.T).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    # Hai dataset object RIÊNG: train có augment, val không.
    # (Không dùng random_split vì 2 Subset của nó dùng CHUNG 1 dataset gốc →
    #  gán val.transform sẽ vô tình tắt augment của cả train.)
    train_ds = NormalizedIrisDataset(args.train_dir, get_train_transform())
    val_base = NormalizedIrisDataset(args.train_dir, get_val_transform())
    num_classes = len(train_ds.classes)

    # 10% of training data for validation proxy — chia theo index, cùng seed
    n_val = max(1, int(len(train_ds) * 0.10))
    perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(SEED)).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset   = torch.utils.data.Subset(val_base, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = IrisEmbedder(embed_dim=EMBED_DIM).to(device)
    head  = ArcFaceLoss(num_classes=num_classes, embed_dim=EMBED_DIM,
                        margin=args.margin, scale=args.scale).to(device)
    print(f"Model: ResNet50 + ArcFace | classes={num_classes} | embed={EMBED_DIM}D")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    epochs_no_improve = 0   # đếm số epoch liên tiếp val_acc không cải thiện
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, head, train_loader, optimizer, device)
        val_acc = evaluate(model, head, val_loader, device)
        scheduler.step()

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output_dir / "arcface_best.pth")
        else:
            epochs_no_improve += 1

        log = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)}
        history.append(log)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
              + ("  ← best" if improved else f"  (no improve {epochs_no_improve}/{args.patience})"))

        # Early stopping: val_acc không cải thiện sau `patience` epoch → dừng
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stop: val_acc không tăng trong {args.patience} epoch "
                  f"(best={best_acc:.4f}). Dừng ở epoch {epoch}.")
            break

    # Save metadata (class list needed for gallery building)
    meta = {"classes": train_ds.classes, "embed_dim": EMBED_DIM}
    with open(args.output_dir / "arcface_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.output_dir / "arcface_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Checkpoint : {args.output_dir / 'arcface_best.pth'}")
    print(f"Meta       : {args.output_dir / 'arcface_meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  type=Path,  default=Path("datasets/train"))
    parser.add_argument("--output_dir", type=Path,  default=Path("models"))
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--margin",     type=float, default=0.5)
    parser.add_argument("--scale",      type=float, default=64.0)
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Early stop nếu val_acc không cải thiện sau N epoch")
    args = parser.parse_args()
    main(args)
