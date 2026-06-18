"""
SNIPPET 02 — Hough Circle Transform
Tự động phát hiện vòng tròn pupil và iris trong ảnh NIR.
Dùng để tạo rough mask làm điểm khởi đầu cho Labelme.
"""

import cv2
import numpy as np

img  = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Làm mờ nhẹ trước khi detect để giảm nhiễu
blurred = cv2.medianBlur(gray, 5)

# HoughCircles trả về array shape (1, N, 3): [cx, cy, radius]
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,           # inverse ratio of resolution
    minDist=30,     # khoảng cách tối thiểu giữa 2 tâm
    param1=50,      # Canny upper threshold
    param2=30,      # accumulator threshold (nhỏ hơn = detect nhiều hơn)
    minRadius=20,
    maxRadius=100,
)

if circles is not None:
    circles = np.round(circles[0]).astype(int)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)   # vẽ vòng tròn
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 3)   # vẽ tâm
