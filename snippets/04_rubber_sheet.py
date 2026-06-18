"""
SNIPPET 04 — Rubber Sheet Model (Iris Normalization)
Ánh xạ iris ring (hình tròn) → ảnh chữ nhật 64×512.

Ý tưởng:
  - Trục ngang (W=512): góc theta từ 0 → 2π
  - Trục dọc  (H=64) : bán kính r từ 0 (pupil) → 1 (iris boundary)
  - Mỗi pixel (r, θ) trong ảnh normalized tương ứng điểm nội suy giữa
    pupil boundary và iris boundary tại góc θ.
"""

import cv2
import numpy as np

NORM_H = 64
NORM_W = 512

def rubber_sheet(img, mask, pupil, iris):
    px, py, pr = pupil
    ix, iy, ir = iris

    # Tạo lưới (NORM_H, NORM_W) cho theta và r
    theta  = np.linspace(0, 2 * np.pi, NORM_W, endpoint=False)  # (W,)
    r_vals = np.linspace(0, 1, NORM_H)                           # (H,)

    theta_grid = theta[np.newaxis, :]   # (1, W) — broadcast theo H
    r_grid     = r_vals[:, np.newaxis]  # (H, 1) — broadcast theo W

    # Tọa độ biên pupil và iris tại từng góc theta
    pupil_x = px + pr * np.cos(theta_grid)  # (1, W)
    pupil_y = py + pr * np.sin(theta_grid)
    iris_x  = ix + ir * np.cos(theta_grid)
    iris_y  = iy + ir * np.sin(theta_grid)

    # Nội suy tuyến tính từ pupil (r=0) đến iris (r=1)
    # → tọa độ pixel gốc cần sample tại mỗi ô (r, theta)
    sample_x = ((1 - r_grid) * pupil_x + r_grid * iris_x).astype(np.float32)
    sample_y = ((1 - r_grid) * pupil_y + r_grid * iris_y).astype(np.float32)

    # cv2.remap: với mỗi (row, col) trong output, lấy pixel tại (sample_x[row,col], sample_y[row,col])
    normalized = cv2.remap(img,  sample_x, sample_y, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
    norm_mask  = cv2.remap(mask, sample_x, sample_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # Noise mask: đánh dấu vùng bị lông mi che (class 0 = background)
    noise = (norm_mask == 0).astype(np.uint8)  # 1=che, 0=hợp lệ

    return normalized, noise  # cả hai shape: (64, 512)
