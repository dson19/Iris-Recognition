"""
BÀI 09 — ArcFace Loss
Mục tiêu: Implement ArcFace từ công thức toán học.

Kiến thức cần: nn.Parameter, F.normalize, torch.acos, torch.cos, scatter_
Đáp án tham khảo: snippets/09_arcface_loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embed_dim=256, margin=0.5, scale=64.0):
        super().__init__()
        self.scale  = scale
        self.margin = margin

        # TODO: khởi tạo weight matrix (num_classes, embed_dim) bằng nn.Parameter
        # Dùng torch.empty rồi init bằng nn.init.xavier_uniform_
        self.weight = ...
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # embeddings: (B, embed_dim) — đã L2-norm
        # labels    : (B,) — class index

        # Bước 1: Normalize class weights (cần để dot product = cosine sim)
        # TODO:
        w = F.normalize(...)  # (num_classes, embed_dim)

        # Bước 2: Cosine similarity = embeddings @ w.T
        # TODO: clamp về range (-1+ε, 1-ε) để tránh nan trong acos
        cos_theta = ...  # (B, num_classes)

        # Bước 3: Chuyển sang góc theta = arccos(cos_theta)
        # TODO:
        theta = ...

        # Bước 4: Tạo one-hot tensor đánh dấu ground-truth class
        # TODO: zeros_like(cos_theta) rồi scatter_ value=1.0 vào vị trí labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Bước 5: Cộng margin vào đúng ground-truth class
        # theta_margin = theta + one_hot * margin
        # TODO:
        theta_margin = ...

        # Bước 6: Tính logit = cos(theta_margin) * scale
        # TODO:
        logits = ...

        # Bước 7: Cross entropy loss
        # TODO:
        return F.cross_entropy(...)


# ── Test ──────────────────────────────────────────────────────────────────
B, D, C = 8, 256, 200   # batch, embed_dim, num_classes

embeddings = F.normalize(torch.randn(B, D), p=2, dim=1)
labels     = torch.randint(0, C, (B,))

head = ArcFaceLoss(num_classes=C, embed_dim=D)
loss = head(embeddings, labels)
print(f"Loss: {loss.item():.4f}")  # phải là số dương hữu hạn
