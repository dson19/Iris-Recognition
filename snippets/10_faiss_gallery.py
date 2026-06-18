"""
SNIPPET 10 — FAISS Gallery: Build + Search
FAISS = Facebook AI Similarity Search
Dùng để tìm kiếm vector gần nhất trong gallery (1:N identification).

IndexFlatIP: exact search bằng Inner Product.
  Sau khi L2-normalize embedding, Inner Product = Cosine Similarity.
  → Score cao hơn = giống hơn (range: -1 đến 1)
"""

import json
import numpy as np
import faiss

EMBED_DIM = 256
THRESHOLD = 0.7   # cosine similarity threshold để quyết định Known/Unknown

# ── BUILD GALLERY ─────────────────────────────────────────────────────────

# embeddings: list of 256-D numpy arrays (đã L2-normalize)
# labels    : list of class names tương ứng
embeddings = [...]  # np.ndarray (256,) mỗi phần tử
labels     = [...]  # str, e.g. "001_L", "001_L", "002_R", ...

vectors = np.stack(embeddings).astype(np.float32)  # (N, 256)

index = faiss.IndexFlatIP(EMBED_DIM)  # IP = Inner Product
index.add(vectors)                    # thêm tất cả gallery vectors
print(f"Gallery: {index.ntotal} vectors")

# Lưu index và labels để dùng sau
faiss.write_index(index, "outputs/faiss_gallery.index")
with open("outputs/gallery_labels.json", "w") as f:
    json.dump(labels, f)


# ── SEARCH (1:N Identification) ───────────────────────────────────────────

index         = faiss.read_index("outputs/faiss_gallery.index")
gallery_labels = json.load(open("outputs/gallery_labels.json"))

# probe: (1, 256) float32, L2-normalized
probe  = np.array([...], dtype=np.float32).reshape(1, -1)
scores, indices = index.search(probe, k=1)  # tìm top-1

top1_score = scores[0, 0]    # cosine similarity
top1_idx   = indices[0, 0]   # vị trí trong gallery

if top1_score >= THRESHOLD:
    identity = gallery_labels[top1_idx]
    print(f"Identified: {identity}  (score={top1_score:.4f})")
else:
    print(f"Unknown User  (score={top1_score:.4f} < threshold={THRESHOLD})")
