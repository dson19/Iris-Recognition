"""
BÀI 02 — Hough Circle Transform
Mục tiêu: Phát hiện vòng tròn pupil/iris trong ảnh NIR.

Kiến thức cần: cv2.medianBlur, cv2.HoughCircles, vẽ circle
Đáp án tham khảo: snippets/02_hough_circles.py
"""

import cv2
import numpy as np

img  = cv2.imread("datasets/preprocessed/001/L/S1001L01.jpg")
# TODO: convert sang grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ── Bước 1: Làm mờ để giảm nhiễu trước khi detect ────────────────────────
# Dùng medianBlur với kernel size 5
# TODO:
blurred = cv2.medianBlur(gray, 5)


# ── Bước 2: Phát hiện circles bằng HoughCircles ───────────────────────────
# method=cv2.HOUGH_GRADIENT, dp=1, minDist=30
# param1=50, param2=30, minRadius=20, maxRadius=100
# TODO:
circles = cv2.HoughCircles(
    blurred,
    method = cv2.HOUGH_GRADIENT,
    dp = 1,
    minDist = 30,
    param1=50,
    param2=30,
    minRadius=20,
    maxRadius=100,
)


# ── Bước 3: Vẽ kết quả lên ảnh ────────────────────────────────────────────
# circles : shape (1, N, 3), bao gồm 1: lớp bọc ngoài, N là số vòng phát hiện, 3: mỗi vòng gồm cx, cy, r (radius)
if circles is not None:
    # TODO: convert circles sang int (dùng np.round rồi .astype(int))
    circles = np.round(circles[0]).astype(int) 

    for cx, cy, r in circles:
        # TODO: vẽ vòng tròn màu xanh lá, độ dày 2
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
        # TODO: vẽ tâm màu đỏ, độ dày 3, radius=2
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 3)

cv2.imwrite("hough_result.jpg", img)
print("Saved hough_result.jpg")
