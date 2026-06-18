"""
SNIPPET 01 — Image Preprocessing
Chuyển ảnh sang grayscale và tăng contrast bằng CLAHE.
"""

import cv2

# CLAHE = Contrast Limited Adaptive Histogram Equalization
# Làm rõ texture iris theo từng vùng cục bộ thay vì toàn ảnh
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

img  = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
enhanced = clahe.apply(gray)

cv2.imwrite("enhanced.jpg", enhanced)
