"""
BÀI 11 — Biometric Metrics: Rank-1, EER, ROC
Mục tiêu: Tính các metrics đánh giá hệ thống nhận dạng sinh trắc học.

Kiến thức cần: numpy operations, sklearn.metrics.roc_curve
Đáp án tham khảo: snippets/11_biometric_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Dữ liệu mẫu (trong thực tế lấy từ evaluate.py)
np.random.seed(42)
N = 200
probe_labels = [f"{i % 49:03d}_L" for i in range(N)]
top1_labels  = probe_labels.copy()
# Giả lập 10% lỗi
for i in range(0, N, 10):
    top1_labels[i] = "999_X"

top1_scores = np.clip(np.random.randn(N) * 0.1 + 0.85, 0, 1).astype(np.float32)
top1_scores[::10] = np.clip(np.random.randn(N // 10) * 0.1 + 0.5, 0, 1)


# ── Phần A: Rank-1 Accuracy ───────────────────────────────────────────────

# TODO: đếm số probe được định danh đúng (probe_label == top1_label)
rank1_correct = ...

# TODO: tính tỉ lệ
rank1_acc = ...

print(f"Rank-1: {rank1_acc*100:.2f}%")


# ── Phần B: Chuẩn bị y_true cho ROC ──────────────────────────────────────

# y_true: 1 = genuine (đúng), 0 = impostor (sai)
# TODO: tạo numpy array y_true từ probe_labels và top1_labels
y_true = np.array([...])


# ── Phần C: ROC Curve ────────────────────────────────────────────────────

# TODO: dùng roc_curve(y_true, top1_scores) → fpr, tpr, thresholds
fpr, tpr, thresholds = ...

# fnr = 1 - tpr
# TODO:
fnr     = ...
roc_auc = auc(fpr, tpr)


# ── Phần D: EER ──────────────────────────────────────────────────────────

# EER = điểm có |FAR - FRR| nhỏ nhất
# TODO: tìm eer_idx bằng np.abs(...).argmin()
eer_idx = ...

# TODO: eer = trung bình của fpr[eer_idx] và fnr[eer_idx]
eer     = ...
eer_thr = float(thresholds[eer_idx])

print(f"EER      : {eer*100:.2f}%")
print(f"Threshold: {eer_thr:.4f}")
print(f"AUC      : {roc_auc:.4f}")


# ── Phần E: Vẽ ROC Curve ─────────────────────────────────────────────────

# TODO: vẽ đường ROC (fpr vs tpr) với label AUC
plt.figure(figsize=(7, 6))
plt.plot(...)

# TODO: đánh dấu điểm EER màu đỏ
plt.scatter(...)

# TODO: thêm đường baseline (diagonal) màu đen nét đứt
plt.plot([0, 1], [0, 1], ...)

plt.xlabel("FAR"); plt.ylabel("TAR")
plt.title("ROC Curve — Iris Recognition 1:N")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("roc_curve.png", dpi=150)
print("Saved roc_curve.png")
