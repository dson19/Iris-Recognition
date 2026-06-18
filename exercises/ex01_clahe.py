"""
BÀI 01 — CLAHE Preprocessing
Mục tiêu: Đọc ảnh, convert sang grayscale, áp CLAHE.

Kiến thức cần: cv2.imread, cv2.cvtColor, cv2.createCLAHE
Đáp án tham khảo: snippets/01_clahe_preprocess.py
"""

import cv2

# ── Bước 1: Tạo object CLAHE ───────────────────────────────────────────────
# Tham số: clipLimit=2.0, tileGridSize=(8, 8)
# TODO: tạo clahe object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── Bước 2: Đọc ảnh và convert sang grayscale ─────────────────────────────
img_path = "datasets/preprocessed/001/L/S1001L01.jpg"

# TODO: đọc ảnh bằng cv2.imread
img = cv2.imread(img_path)

# TODO: convert RGB → grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Bước 3: Áp CLAHE ──────────────────────────────────────────────────────
# TODO: áp clahe lên ảnh grayscale
enhanced = clahe.apply(gray)


# ── Bước 4: Lưu kết quả ───────────────────────────────────────────────────
# TODO: lưu ảnh enhanced ra file "enhanced.jpg"
cv2.imwrite("enhanced.jpg", enhanced)
...

print("Done. Ảnh đã được lưu.")
