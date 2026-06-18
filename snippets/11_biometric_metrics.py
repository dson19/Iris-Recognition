"""
SNIPPET 11 — Biometric Evaluation Metrics
Các metrics chuẩn trong hệ thống nhận dạng sinh trắc học:
  - Rank-1 Accuracy
  - FAR (False Acceptance Rate)
  - FRR (False Rejection Rate)
  - EER (Equal Error Rate)
  - ROC Curve + AUC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ── Input ─────────────────────────────────────────────────────────────────
# probe_labels : ["001_L", "001_L", "002_R", ...]  — nhãn thật của probe
# top1_labels  : ["001_L", "003_R", "002_R", ...]  — nhãn dự đoán top-1
# top1_scores  : [0.92,    0.61,    0.88,   ...]   — cosine similarity score

probe_labels = [...]
top1_labels  = [...]
top1_scores  = np.array([...])

# ── Rank-1 Accuracy ───────────────────────────────────────────────────────
# Tỉ lệ probe được định danh đúng ở top-1
rank1_correct = sum(pl == tl for pl, tl in zip(probe_labels, top1_labels))
rank1_acc     = rank1_correct / len(probe_labels)
print(f"Rank-1: {rank1_acc*100:.2f}%")

# ── ROC Curve ─────────────────────────────────────────────────────────────
# y_true: 1 = genuine (probe khớp top-1), 0 = impostor (không khớp)
y_true = np.array([1 if pl == tl else 0
                   for pl, tl in zip(probe_labels, top1_labels)])

fpr, tpr, thresholds = roc_curve(y_true, top1_scores)
# fpr = FAR (False Acceptance Rate)
# tpr = TAR = 1 - FRR
# fnr = FRR (False Rejection Rate)
fnr     = 1 - tpr
roc_auc = auc(fpr, tpr)

# ── EER — Equal Error Rate ────────────────────────────────────────────────
# Điểm trên ROC mà FAR = FRR
# Tìm index có |FAR - FRR| nhỏ nhất
eer_idx   = np.abs(fpr - fnr).argmin()
eer       = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
eer_thr   = float(thresholds[eer_idx])
far_at_eer = float(fpr[eer_idx])
frr_at_eer = float(fnr[eer_idx])

print(f"EER         : {eer*100:.2f}%")
print(f"FAR @ EER   : {far_at_eer*100:.2f}%")
print(f"FRR @ EER   : {frr_at_eer*100:.2f}%")
print(f"Threshold   : {eer_thr:.4f}")
print(f"AUC         : {roc_auc:.4f}")

# ── ROC Plot ──────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.scatter([far_at_eer], [tpr[eer_idx]], color="red", zorder=5,
            label=f"EER = {eer*100:.2f}%  (thr={eer_thr:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("FAR (False Acceptance Rate)")
plt.ylabel("TAR (True Acceptance Rate = 1 - FRR)")
plt.title("ROC Curve — Iris Recognition 1:N")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
