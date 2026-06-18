"""
SNIPPET 03 — Extract Circle Parameters từ U-Net Mask
Sau khi U-Net predict mask 3-class, dùng contour để lấy tâm + bán kính
của pupil (class 2) và iris (class 1).
"""

import cv2
import numpy as np

# mask: H×W numpy array, giá trị 0=background, 1=iris, 2=pupil
mask = cv2.imread("predicted_mask.png", cv2.IMREAD_GRAYSCALE)

def extract_circle(mask, class_id):
    """Tách class → tìm contour lớn nhất → minEnclosingCircle."""
    binary = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    return (int(cx), int(cy), int(r))  # (tâm_x, tâm_y, bán_kính)

pupil = extract_circle(mask, class_id=2)  # (px, py, pr)
iris  = extract_circle(mask, class_id=1)  # (ix, iy, ir)

print(f"Pupil: cx={pupil[0]}, cy={pupil[1]}, r={pupil[2]}")
print(f"Iris : cx={iris[0]},  cy={iris[1]},  r={iris[2]}")
