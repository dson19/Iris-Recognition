"""
Nhận dạng 1:N cho MỘT ảnh mắt bất kỳ — nối toàn bộ pipeline:
  ảnh gốc → CLAHE → U-Net mask → Rubber Sheet (64×512) → embedder → FAISS search

Usage:
  python scripts/identify.py --image datasets/raw/casia_interval/050/L/S1050L03.jpg
  python scripts/identify.py --image path/to/eye.jpg --topk 5
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import faiss

from preprocess import apply_clahe
from normalize_iris import load_unet, predict_mask, extract_circles, rubber_sheet
from train_arcface import IrisEmbedder, EMBED_DIM
from build_gallery import TRANSFORM


@torch.no_grad()
def identify(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load models + gallery DB ---
    unet = load_unet(args.unet, "resnet34", device)
    embedder = IrisEmbedder(embed_dim=EMBED_DIM)
    embedder.load_state_dict(torch.load(args.embedder, map_location=device))
    embedder.eval().to(device)
    index = faiss.read_index(str(args.index))
    labels = json.load(open(args.labels))
    print(f"Gallery: {index.ntotal} vectors, {len(set(labels))} identities\n")

    # --- B1: đọc ảnh + CLAHE (giống preprocess.py) ---
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = apply_clahe(gray)

    # --- B2: U-Net dự đoán mask → fit vòng tròn ---
    mask = predict_mask(unet, gray, device)
    pupil, iris = extract_circles(mask)
    if pupil is None or iris is None:
        print("❌ Không tìm được vòng pupil/iris — ảnh có thể bị nhắm/mờ/không phải mắt.")
        return

    # --- B3: Rubber Sheet → iris phẳng 64×512 ---
    normalized, _noise = rubber_sheet(gray, mask, pupil, iris)

    # --- B4: embedder → vector 256-D ---
    tensor = TRANSFORM(image=normalized)["image"].unsqueeze(0).to(device)
    vec = embedder(tensor).cpu().numpy().astype(np.float32)  # 1×256

    # --- B5: FAISS tìm top-k gần nhất ---
    scores, indices = index.search(vec, k=args.topk)
    print(f"Ảnh: {args.image}")
    print(f"{'='*42}")
    for rank, (idx, sc) in enumerate(zip(indices[0], scores[0]), 1):
        marker = "  ← dự đoán" if rank == 1 else ""
        print(f"  Top-{rank}: {labels[idx]:10s}  cosine={sc:.4f}{marker}")
    print(f"{'='*42}")

    best_score = scores[0][0]
    if best_score < args.threshold:
        print(f"⚠️  cosine {best_score:.4f} < ngưỡng {args.threshold} → có thể là NGƯỜI LẠ (không có trong gallery)")
    else:
        print(f"✅ Danh tính: {labels[indices[0][0]]}  (cosine={best_score:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nhận dạng 1:N cho một ảnh mắt")
    parser.add_argument("--image",    type=Path, required=True, help="Đường dẫn ảnh mắt (jpg/png)")
    parser.add_argument("--topk",     type=int,  default=5, help="Số ứng viên gần nhất hiển thị")
    parser.add_argument("--threshold", type=float, default=0.3, help="Ngưỡng cosine để coi là người lạ")
    parser.add_argument("--unet",     type=Path, default=Path("models/unet_best.pth"))
    parser.add_argument("--embedder", type=Path, default=Path("models/arcface_best.pth"))
    parser.add_argument("--index",    type=Path, default=Path("outputs/faiss_gallery.index"))
    parser.add_argument("--labels",   type=Path, default=Path("outputs/gallery_labels.json"))
    args = parser.parse_args()
    identify(args)
