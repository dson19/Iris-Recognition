"""
SNIPPET 07 — Training Loop
Vòng lặp train/eval chuẩn PyTorch dùng trong cả U-Net và ArcFace.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


# ── Train một epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()       # reset gradient từ batch trước
        output = model(imgs)        # forward pass
        loss   = loss_fn(output, labels)
        loss.backward()             # tính gradient
        optimizer.step()            # cập nhật weights

        total_loss += loss.item()

    return total_loss / len(loader)  # average loss per batch


# ── Evaluate ──────────────────────────────────────────────────────────────

@torch.no_grad()  # tắt gradient computation để tiết kiệm memory
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="  val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)
        loss   = loss_fn(output, labels)
        total_loss += loss.item()

        preds   = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Main training loop ────────────────────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = ...   # model của bạn
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# CosineAnnealingLR: giảm LR theo dạng cosine từ lr_max → 0 trong T_max epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_metric = 0.0

for epoch in range(1, 51):
    train_loss              = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc       = evaluate(model, val_loader, loss_fn, device)
    scheduler.step()

    # Lưu checkpoint khi metric cải thiện
    if val_acc > best_metric:
        best_metric = val_acc
        torch.save(model.state_dict(), "models/best.pth")
        tag = "  ← best"
    else:
        tag = ""

    print(f"Epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}{tag}")
