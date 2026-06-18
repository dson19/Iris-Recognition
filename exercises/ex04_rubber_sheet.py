"""
BÀI 04 — Rubber Sheet Model
Mục tiêu: Ánh xạ iris ring (hình tròn) → ảnh chữ nhật 64×512.

Đây là bài khó nhất về numpy. Tập trung hiểu broadcasting.
Kiến thức cần: np.linspace, np.newaxis, broadcasting, cv2.remap
Đáp án tham khảo: snippets/04_rubber_sheet.py
"""

import cv2
import numpy as np

NORM_H = 64
NORM_W = 512


def rubber_sheet(img, mask, pupil, iris):
    px, py, pr = pupil  # tâm và bán kính pupil
    ix, iy, ir = iris   # tâm và bán kính iris

    # ── Bước 1: Tạo lưới góc theta và bán kính r ──────────────────────────
    # theta: NORM_W điểm từ 0 → 2π (không lấy endpoint)
    # TODO:
    theta = np.linspace(...)   # shape: (NORM_W,)

    # r_vals: NORM_H điểm từ 0 → 1
    # TODO:
    r_vals = np.linspace(...)  # shape: (NORM_H,)

    # ── Bước 2: Broadcast thành lưới 2D ───────────────────────────────────
    # theta_grid: shape (1, NORM_W) — dùng np.newaxis
    # r_grid    : shape (NORM_H, 1)
    # TODO:
    theta_grid = ...  # theta[np.newaxis, :]
    r_grid     = ...  # r_vals[:, np.newaxis]

    # ── Bước 3: Tọa độ biên pupil và iris tại từng góc theta ──────────────
    # Công thức điểm trên đường tròn: x = cx + r * cos(theta)
    # TODO:
    pupil_x = ...  # px + pr * cos(theta_grid)
    pupil_y = ...
    iris_x  = ...  # ix + ir * cos(theta_grid)
    iris_y  = ...

    # ── Bước 4: Nội suy tuyến tính từ pupil (r=0) đến iris (r=1) ──────────
    # Công thức: sample = (1 - r) * pupil + r * iris
    # TODO:
    sample_x = (...).astype(np.float32)  # shape: (NORM_H, NORM_W)
    sample_y = (...).astype(np.float32)

    # ── Bước 5: Remap ảnh và mask ─────────────────────────────────────────
    # img  → dùng INTER_LINEAR  (nội suy song tuyến tính cho ảnh)
    # mask → dùng INTER_NEAREST (giữ nguyên giá trị class, không nội suy)
    # TODO:
    normalized = cv2.remap(img,  ..., ..., ..., borderMode=cv2.BORDER_REFLECT)
    norm_mask  = cv2.remap(mask, ..., ..., ..., borderMode=cv2.BORDER_REFLECT)

    # ── Bước 6: Tạo noise mask ────────────────────────────────────────────
    # Noise = 1 ở vùng background (class 0), 0 ở vùng iris hợp lệ
    # TODO:
    noise = ...  # (norm_mask == 0).astype(np.uint8)

    return normalized, noise  # cả hai shape: (64, 512)
