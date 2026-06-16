"""
Evaluate 1:N iris identification system.

Metrics:
  - Rank-1 Accuracy
  - FAR  (False Acceptance Rate)
  - FRR  (False Rejection Rate)
  - EER  (Equal Error Rate)
  - ROC curve + AUC

Usage:
  python scripts/evaluate.py
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    tensor = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)
    return model(tensor).squeeze(0).cpu().numpy()


def find_eer(fpr: np.ndarray, fnr: np.ndarray) -> tuple[float, float]:
    """Find EER = point where FAR ≈ FRR."""
    diff = np.abs(fpr - fnr)
    idx = diff.argmin()
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = idx  # index proxy; actual threshold from roc_curve
    return float(eer), int(idx)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    model = IrisEmbedder(embed_dim=EMBED_DIM)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)

    # --- Load FAISS index + labels ---
    index = faiss.read_index(str(args.index_path))
    with open(args.labels_path) as f:
        gallery_labels = json.load(f)   # list[str], index → class_name
    print(f"Gallery: {index.ntotal} vectors, {len(set(gallery_labels))} unique classes")

    # --- Extract probe embeddings ---
    probe_paths = sorted(args.probe_dir.rglob("*.png"))
    if not probe_paths:
        raise RuntimeError(f"No probe images found in {args.probe_dir}")
    print(f"Probe images: {len(probe_paths)}")

    probe_embeddings, probe_labels = [], []
    for img_path in tqdm(probe_paths, desc="Extracting probes"):
        emb = extract_embedding(model, img_path, device)
        if emb is None:
            continue
        probe_embeddings.append(emb)
        probe_labels.append(img_path.parent.name)   # e.g. "201_L"

    probes = np.stack(probe_embeddings).astype(np.float32)  # N×256

    # --- 1:N Search ---
    scores, indices = index.search(probes, k=1)   # N×1
    top1_scores  = scores[:, 0]
    top1_indices = indices[:, 0]
    top1_labels  = [gallery_labels[i] for i in top1_indices]

    # --- Rank-1 Accuracy ---
    rank1_correct = sum(
        pl == tl for pl, tl in zip(probe_labels, top1_labels)
    )
    rank1_acc = rank1_correct / len(probe_labels)

    # --- Build score pairs for ROC ---
    # genuine: probe and top-1 match (same class)
    # impostor: probe and top-1 don't match
    y_true = np.array([1 if pl == tl else 0
                       for pl, tl in zip(probe_labels, top1_labels)])

    fpr, tpr, thresholds = roc_curve(y_true, top1_scores)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)

    # EER
    diff = np.abs(fpr - fnr)
    eer_idx = diff.argmin()
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    # FAR / FRR at EER threshold
    far = float(fpr[eer_idx])
    frr = float(fnr[eer_idx])

    # --- Print results ---
    print(f"\n{'='*45}")
    print(f"  Rank-1 Accuracy : {rank1_acc*100:.2f}%  ({rank1_correct}/{len(probe_labels)})")
    print(f"  EER             : {eer*100:.2f}%")
    print(f"  FAR @ EER       : {far*100:.2f}%")
    print(f"  FRR @ EER       : {frr*100:.2f}%")
    print(f"  AUC             : {roc_auc:.4f}")
    print(f"  EER Threshold   : {eer_threshold:.4f}")
    print(f"{'='*45}")

    # --- Save results ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "rank1_accuracy": round(rank1_acc, 4),
        "eer": round(eer, 4),
        "far_at_eer": round(far, 4),
        "frr_at_eer": round(frr, 4),
        "auc": round(roc_auc, 4),
        "eer_threshold": round(eer_threshold, 4),
        "num_probes": len(probe_labels),
        "gallery_size": index.ntotal,
    }
    with open(args.output_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- ROC curve plot ---
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.scatter([far], [tpr[eer_idx]], color="red", zorder=5,
                label=f"EER = {eer*100:.2f}% (thr={eer_threshold:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("TAR (True Acceptance Rate)")
    plt.title("ROC Curve — Iris Recognition 1:N")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = args.output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    print(f"\nROC curve : {roc_path}")
    print(f"Report    : {args.output_dir / 'eval_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir",   type=Path, default=Path("datasets/normalized_probe"),
                        help="Normalized probe images")
    parser.add_argument("--index_path",  type=Path, default=Path("outputs/faiss_gallery.index"))
    parser.add_argument("--labels_path", type=Path, default=Path("outputs/gallery_labels.json"))
    parser.add_argument("--checkpoint",  type=Path, default=Path("models/arcface_best.pth"))
    parser.add_argument("--output_dir",  type=Path, default=Path("outputs"))
    args = parser.parse_args()
    evaluate(args)
