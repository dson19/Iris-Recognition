"""
SNIPPET 05 — U-Net Segmentation Model
Dùng thư viện segmentation_models_pytorch (smp) để tạo U-Net
với encoder ResNet34 pretrained trên ImageNet.

Tại sao dùng pretrained encoder?
  - Cần ít ảnh label hơn (~100 thay vì ~500+)
  - Convergence nhanh hơn
  - Feature extractor đã hiểu edge/texture từ ImageNet
"""

import torch
import segmentation_models_pytorch as smp

NUM_CLASSES = 3  # 0=background, 1=iris, 2=pupil

model = smp.Unet(
    encoder_name="resnet34",      # backbone: resnet18/34/50, efficientnet-b0, ...
    encoder_weights="imagenet",   # pretrained weights
    in_channels=1,                # grayscale input (1 channel, không phải RGB)
    classes=NUM_CLASSES,          # số class output
)

# Thử forward pass
dummy = torch.randn(2, 1, 320, 280)  # batch=2, 1ch, 320×280
output = model(dummy)
print(output.shape)  # → (2, 3, 320, 280): logits cho 3 class


# ── Loss ──────────────────────────────────────────────────────────────────
import torch.nn as nn

class CombinedLoss(nn.Module):
    """
    BCE (CrossEntropy) + Dice Loss.
    - CrossEntropy: học phân loại từng pixel
    - DiceLoss: tối ưu overlap, tốt cho class mất cân bằng (pupil rất nhỏ)
    """
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)

    def forward(self, logits, targets):
        # logits : (B, C, H, W) — raw model output
        # targets: (B, H, W)   — integer class index per pixel
        return self.ce(logits, targets) + self.dice(logits, targets)


# ── Dice Score ────────────────────────────────────────────────────────────
def dice_score_per_class(preds, targets, num_classes=NUM_CLASSES):
    """
    preds, targets: (H, W) integer tensors
    Trả về [dice_iris, dice_pupil] (bỏ class 0 = background)
    """
    scores = []
    for cls in range(1, num_classes):       # skip background
        pred_c = (preds == cls).float()
        tgt_c  = (targets == cls).float()
        intersection = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        scores.append(((2 * intersection + 1e-6) / (denom + 1e-6)).item())
    return scores  # [iris_dice, pupil_dice]
