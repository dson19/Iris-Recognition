"""
Build FAISS gallery index from normalized iris images.

For each image in datasets/gallery/<class_name>/*.png:
  - Extract 256-D embedding with trained ResNet50
  - Add to FAISS IndexFlatIP (inner product = cosine sim after L2 norm)

Outputs:
  outputs/faiss_gallery.index
  outputs/gallery_labels.json   ← maps FAISS vector index → subject class name

Usage:
  python scripts/build_gallery.py
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from train_arcface import IrisEmbedder, EMBED_DIM


TRANSFORM = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


@torch.no_grad()
def extract_embedding(model, img_path: Path, device: torch.device) -> np.ndarray | None:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    tensor = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)  # 1×1×H×W
    emb = model(tensor).squeeze(0).cpu().numpy()                    # (256,)
    return emb


def build_gallery(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = IrisEmbedder(embed_dim=EMBED_DIM)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)
    print(f"Loaded embedder from {args.checkpoint}")

    # Collect gallery images
    img_paths = sorted(args.gallery_dir.rglob("*.png"))
    if not img_paths:
        raise RuntimeError(f"No PNG images found in {args.gallery_dir}")
    print(f"Gallery images: {len(img_paths)}")

    # Extract embeddings
    embeddings = []
    labels = []
    failed = 0

    for img_path in tqdm(img_paths, desc="Extracting"):
        class_name = img_path.parent.name   # e.g. "201_L"
        emb = extract_embedding(model, img_path, device)
        if emb is None:
            print(f"  [WARN] Cannot read: {img_path.name}")
            failed += 1
            continue
        embeddings.append(emb)
        labels.append(class_name)

    if not embeddings:
        raise RuntimeError("No embeddings extracted.")

    vectors = np.stack(embeddings, axis=0).astype(np.float32)   # N×256
    print(f"Extracted {len(vectors)} embeddings ({failed} failed)")

    # Build FAISS index (Inner Product — equivalent to cosine sim for L2-normed vecs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    print(f"FAISS index: {index.ntotal} vectors")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path  = args.output_dir / "faiss_gallery.index"
    labels_path = args.output_dir / "gallery_labels.json"

    faiss.write_index(index, str(index_path))
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSaved index  : {index_path}")
    print(f"Saved labels : {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery_dir", type=Path,  default=Path("datasets/normalized_gallery"),
                        help="Normalized gallery images (subdir of datasets/gallery after normalize_iris.py)")
    parser.add_argument("--checkpoint",  type=Path,  default=Path("models/arcface_best.pth"))
    parser.add_argument("--output_dir",  type=Path,  default=Path("outputs"))
    args = parser.parse_args()
    build_gallery(args)
