"""
BÀI 08 — ResNet50 Embedder
Mục tiêu: Modify ResNet50 pretrained để nhận grayscale và output 256D embedding.

Kiến thức cần: nn.Module, torchvision.models, nn.Linear, F.normalize
Đáp án tham khảo: snippets/08_resnet50_embedder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBED_DIM = 256


class IrisEmbedder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

        # Bước 1: Load ResNet50 pretrained
        # TODO:
        backbone = models.resnet50(weights=...)

        # Bước 2: Thay conv1 từ 3-channel sang 1-channel
        # Lấy trung bình weight RGB → grayscale để giữ pretrained knowledge
        old_conv = backbone.conv1  # weight shape: (64, 3, 7, 7)

        # TODO: tạo Conv2d mới với in_channels=1, các tham số còn lại giống old_conv
        new_conv = nn.Conv2d(...)

        # TODO: copy weight: mean(dim=1, keepdim=True) để từ (64,3,7,7) → (64,1,7,7)
        new_conv.weight.data = ...

        backbone.conv1 = new_conv

        # Bước 3: Bỏ avgpool + fc, chỉ giữ feature extractor
        # list(backbone.children()) = [conv1, bn1, relu, maxpool, layer1..4, avgpool, fc]
        # TODO: lấy tất cả trừ 2 layer cuối (avgpool và fc)
        self.backbone = nn.Sequential(...)  # output shape: (B, 2048, 1, 1)

        # Bước 4: Projection head
        # TODO: Linear layer: 2048 → embed_dim
        self.embed = ...
        # TODO: BatchNorm1d cho embed_dim
        self.bn    = ...

    def forward(self, x):  # x: (B, 1, H, W)
        # TODO: chạy qua backbone, flatten từ (B, 2048, 1, 1) → (B, 2048)
        feat = ...

        # TODO: chạy qua embed và bn → (B, embed_dim)
        emb  = ...

        # TODO: L2 normalize (p=2, dim=1)
        return ...


# ── Test ──────────────────────────────────────────────────────────────────
model  = IrisEmbedder()
dummy  = torch.randn(4, 1, 64, 512)
output = model(dummy)
print(output.shape)         # phải là: torch.Size([4, 256])
print(output.norm(dim=1))   # phải là: tensor([1., 1., 1., 1.])
