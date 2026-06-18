"""
BÀI 07 — Training Loop
Mục tiêu: Viết hoàn chỉnh vòng lặp train + eval + save checkpoint.

Kiến thức cần: model.train(), optimizer, loss.backward(), scheduler
Đáp án tham khảo: snippets/07_training_loop.py
"""

import torch
import torch.nn as nn
from tqdm import tqdm


# ── Phần A: Train một epoch ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    # TODO: set model sang chế độ train
    ...

    total_loss = 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        # TODO: chuyển imgs và labels lên device
        imgs, labels = ...

        # TODO: reset gradient (luôn làm đầu tiên trong mỗi batch)
        ...

        # TODO: forward pass → lấy output
        output = ...

        # TODO: tính loss
        loss = ...

        # TODO: backward pass (tính gradient)
        ...

        # TODO: cập nhật weights
        ...

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Phần B: Evaluate ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    # TODO: set model sang chế độ eval
    ...

    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="  val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # TODO: forward pass
        output = ...

        # TODO: tính loss và cộng vào total_loss
        ...

        # TODO: lấy class dự đoán (argmax theo dim=1)
        preds = ...

        # TODO: đếm số dự đoán đúng
        correct += ...
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Phần C: Main training loop ────────────────────────────────────────────

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = ...
loss_fn = nn.CrossEntropyLoss()

# TODO: tạo AdamW optimizer với lr=1e-4, weight_decay=1e-4
optimizer = ...

# TODO: tạo CosineAnnealingLR scheduler với T_max=50
scheduler = ...

best_val_acc = 0.0

for epoch in range(1, 51):
    # TODO: gọi train_one_epoch
    train_loss = ...

    # TODO: gọi evaluate
    val_loss, val_acc = ...

    # TODO: bước scheduler
    ...

    # TODO: nếu val_acc tốt hơn best, lưu model.state_dict()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # torch.save(...)
        ...
        tag = "  ← best"
    else:
        tag = ""

    print(f"Epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}{tag}")
