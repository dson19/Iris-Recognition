"""
BÀI 05 — U-Net Model + Loss + Dice Score
Mục tiêu: Khởi tạo U-Net, định nghĩa loss function và metric.

Kiến thức cần: segmentation_models_pytorch, nn.Module, CrossEntropyLoss
Đáp án tham khảo: snippets/05_unet_model.py
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

NUM_CLASSES = 3  # 0=background, 1=iris, 2=pupil


# ── Phần A: Khởi tạo U-Net ────────────────────────────────────────────────
# encoder_name="resnet34", encoder_weights="imagenet"
# in_channels=1 (grayscale), classes=NUM_CLASSES
# TODO:
model = smp.Unet(
    encoder_name=...,
    encoder_weights=...,
    in_channels=...,
    classes=...,
)

# Test forward pass
dummy  = torch.randn(2, 1, 320, 280)
output = model(dummy)
print(output.shape)  # phải là: torch.Size([2, 3, 320, 280])


# ── Phần B: Combined Loss ─────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: khởi tạo CrossEntropyLoss
        self.ce   = ...
        # TODO: khởi tạo DiceLoss với mode="multiclass"
        self.dice = ...

    def forward(self, logits, targets):
        # logits : (B, C, H, W)
        # targets: (B, H, W) integer class index
        # TODO: trả về tổng của ce loss và dice loss
        return ...


# ── Phần C: Dice Score per Class ──────────────────────────────────────────

def dice_score_per_class(preds, targets, num_classes=NUM_CLASSES):
    """
    preds, targets: (H, W) integer tensors
    Trả về list [dice_iris, dice_pupil]
    """
    scores = []
    for cls in range(1, num_classes):  # bắt đầu từ 1 để bỏ background

        # TODO: tạo binary tensor cho class cls trong preds và targets
        pred_c = ...   # (preds == cls).float()
        tgt_c  = ...

        # TODO: tính intersection (tích element-wise rồi sum)
        intersection = ...

        # TODO: tính denominator (tổng pred_c + tổng tgt_c)
        denom = ...

        # TODO: tính dice = (2 * intersection + epsilon) / (denom + epsilon)
        dice = ...

        scores.append(dice.item())

    return scores  # [iris_dice, pupil_dice]
