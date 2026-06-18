"""
SNIPPET 09 — ArcFace Loss
Metric learning loss: thêm angular margin m vào góc theta giữa embedding
và class weight để tăng inter-class separation và intra-class compactness.

Ý tưởng:
  Softmax thông thường: logit = cos(θ) × scale
  ArcFace             : logit = cos(θ + m) × scale  (chỉ với ground-truth class)

Càng tăng margin m → model càng bị ép học embedding "chắc chắn" hơn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embed_dim=256, margin=0.5, scale=64.0):
        super().__init__()
        self.scale  = scale   # s: nhân để logit không quá nhỏ sau cosine
        self.margin = margin  # m: góc margin tính bằng radian (~28.6°)

        # Class weight matrix: (num_classes, embed_dim)
        # Mỗi hàng = "prototype" embedding của một class
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # embeddings: (B, embed_dim) — đã L2-norm
        # labels    : (B,) — class index

        # Normalize class weights (cần thiết để dot product = cosine similarity)
        w = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embed_dim)

        # Cosine similarity giữa mỗi embedding và mỗi class prototype
        cos_theta = (embeddings @ w.T).clamp(-1 + 1e-7, 1 - 1e-7)  # (B, num_classes)

        # Chuyển sang góc theta để cộng margin
        theta = torch.acos(cos_theta)  # (B, num_classes)

        # one_hot: đánh dấu đúng class của mỗi sample
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.unsqueeze(1), 1.0)

        # Chỉ cộng margin vào ground-truth class
        theta_margin = theta + one_hot * self.margin

        # Chuyển về logit (scale để gradient không quá nhỏ)
        logits = torch.cos(theta_margin) * self.scale  # (B, num_classes)

        return F.cross_entropy(logits, labels)


# ── Dùng trong training ───────────────────────────────────────────────────
# model = IrisEmbedder()
# head  = ArcFaceLoss(num_classes=200, embed_dim=256, margin=0.5, scale=64)
#
# emb  = model(imgs)           # (B, 256) L2-normalized
# loss = head(emb, labels)     # ArcFace loss
# loss.backward()
