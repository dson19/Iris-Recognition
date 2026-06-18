"""
SNIPPET 08 — ResNet50 Iris Embedder
ResNet50 pretrained ImageNet được adapt để:
  1. Nhận grayscale input (1 channel thay vì 3)
  2. Output 256-D L2-normalized embedding vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBED_DIM = 256

class IrisEmbedder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

        # Load ResNet50 pretrained trên ImageNet (3-channel RGB)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ── Adapt conv1: RGB (3ch) → grayscale (1ch) ──────────────────────
        # Thay vì khởi tạo lại ngẫu nhiên, lấy trung bình weight của 3 channel RGB
        # → giữ được prior knowledge từ ImageNet
        old_conv = backbone.conv1  # weight shape: (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        # mean(dim=1): average 3 channels → 1 channel, shape: (64, 1, 7, 7)
        backbone.conv1 = new_conv

        # Bỏ avgpool + fc (classification head), chỉ giữ feature extractor
        # children(): [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # output shape: (B, 2048, 1, 1)

        # Projection head: 2048 → embed_dim
        self.embed = nn.Linear(2048, embed_dim)
        self.bn    = nn.BatchNorm1d(embed_dim)

    def forward(self, x):  # x: (B, 1, H, W)
        feat = self.backbone(x).flatten(1)    # (B, 2048)
        emb  = self.bn(self.embed(feat))      # (B, embed_dim)
        return F.normalize(emb, p=2, dim=1)  # L2 normalize → cosine sim = dot product


# ── Test ──────────────────────────────────────────────────────────────────
model  = IrisEmbedder()
dummy  = torch.randn(4, 1, 64, 512)  # batch=4, grayscale, 64×512
output = model(dummy)
print(output.shape)         # → (4, 256)
print(output.norm(dim=1))   # → tensor([1., 1., 1., 1.])  đã L2-norm
