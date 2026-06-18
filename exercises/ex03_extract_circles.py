"""
BÀI 03 — Extract Circle Parameters từ Mask
Mục tiêu: Từ mask 3-class của U-Net, lấy ra (cx, cy, r) của pupil và iris.

Kiến thức cần: numpy boolean indexing, cv2.findContours, cv2.minEnclosingCircle
Đáp án tham khảo: snippets/03_extract_circles_from_mask.py
"""

import cv2
import numpy as np

mask = cv2.imread("predicted_mask.png", cv2.IMREAD_GRAYSCALE)
# mask có giá trị: 0=background, 1=iris, 2=pupil


def extract_circle(mask, class_id):
    """
    Tách pixels của class_id → tìm contour lớn nhất → trả về (cx, cy, r).
    Trả về None nếu không tìm được contour.
    """
    # Bước 1: Tạo binary mask chỉ giữ pixels của class_id
    # TODO: binary là array uint8, giá trị 1 nơi mask==class_id, 0 còn lại
    binary = ...

    # Bước 2: Tìm contours trong binary mask
    # TODO: dùng cv2.findContours với RETR_EXTERNAL và CHAIN_APPROX_SIMPLE
    contours, _ = ...

    # Bước 3: Kiểm tra có contour không
    # TODO: nếu contours rỗng, return None
    ...

    # Bước 4: Lấy contour lớn nhất theo diện tích
    # TODO: dùng max() với key=cv2.contourArea
    largest = ...

    # Bước 5: Tính vòng tròn bao nhỏ nhất
    # TODO: dùng cv2.minEnclosingCircle, kết quả là ((cx, cy), r)
    (cx, cy), r = ...

    # Bước 6: Trả về tuple int
    return (int(cx), int(cy), int(r))


# ── Test ──────────────────────────────────────────────────────────────────
pupil = extract_circle(mask, class_id=2)
iris  = extract_circle(mask, class_id=1)

if pupil and iris:
    print(f"Pupil: cx={pupil[0]}, cy={pupil[1]}, r={pupil[2]}")
    print(f"Iris : cx={iris[0]},  cy={iris[1]},  r={iris[2]}")
else:
    print("Không tìm được circle!")
