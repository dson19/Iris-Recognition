"""
BÀI 10 — FAISS: Build Gallery + Search
Mục tiêu: Tạo FAISS index và thực hiện 1:N identification.

Kiến thức cần: faiss.IndexFlatIP, index.add, index.search
Đáp án tham khảo: snippets/10_faiss_gallery.py
"""

import json
import numpy as np
import faiss

EMBED_DIM = 256
THRESHOLD = 0.7


# ── Phần A: Build Gallery ─────────────────────────────────────────────────

# Giả sử đã có embeddings và labels
embeddings = [np.random.randn(EMBED_DIM).astype(np.float32) for _ in range(98)]
labels     = [f"{i//2:03d}_L" for i in range(98)]

# Bước 1: Stack thành matrix (N, 256) float32
# TODO:
vectors = ...

# Bước 2: Tạo FAISS IndexFlatIP
# TODO:
index = faiss.IndexFlatIP(...)

# Bước 3: Thêm vectors vào index
# TODO:
...

print(f"Gallery: {index.ntotal} vectors")  # phải là 98

# Bước 4: Lưu index và labels
# TODO: faiss.write_index(index, path)
...
# TODO: json.dump labels ra file
...


# ── Phần B: 1:N Search ────────────────────────────────────────────────────

# Load lại
index          = faiss.read_index("outputs/faiss_gallery.index")
gallery_labels = json.load(open("outputs/gallery_labels.json"))

# probe: 1 vector query shape (1, 256)
probe = np.random.randn(1, EMBED_DIM).astype(np.float32)

# Bước 5: Search top-1
# TODO: index.search trả về (scores, indices) shape (1, k)
scores, indices = index.search(...)

# Bước 6: Lấy top-1 score và index
# TODO:
top1_score = ...
top1_idx   = ...

# Bước 7: Quyết định Known/Unknown
# TODO: nếu score >= THRESHOLD → identity, ngược lại → "Unknown"
if ...:
    print(f"Identified: {gallery_labels[top1_idx]}  score={top1_score:.4f}")
else:
    print(f"Unknown User  score={top1_score:.4f}")
