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

    probes = np.stack(probe_embeddings).astype(np.float32)  # Np×256
    probe_labels = np.array(probe_labels)

    # Lấy lại toàn bộ vector gallery từ FAISS index
    gallery_vecs = index.reconstruct_n(0, index.ntotal).astype(np.float32)  # Ng×256
    gallery_labels = np.array(gallery_labels)

    # --- Ma trận cosine: mọi probe × mọi gallery (vector đã L2-norm) ---
    sims = probes @ gallery_vecs.T   # Np×Ng

    # --- Rank-1: với mỗi probe, lấy gallery GẦN NHẤT ---
    best = sims.argmax(axis=1)
    pred_labels = gallery_labels[best]
    rank1_correct = int((pred_labels == probe_labels).sum())
    rank1_acc = rank1_correct / len(probe_labels)

    # --- Verification: DÙNG TOÀN BỘ cặp (genuine vs impostor) ---
    # match[i, j] = True nếu probe i và gallery j cùng danh tính
    match = (probe_labels[:, None] == gallery_labels[None, :])   # Np×Ng
    genuine_scores  = sims[match]      # cặp CÙNG người
    impostor_scores = sims[~match]     # cặp KHÁC người
    print(f"Cặp genuine : {len(genuine_scores)}  |  cặp impostor: {len(impostor_scores)}")

    y_true  = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)

    # EER: điểm FAR ≈ FRR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
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
    print(f"  Genuine  cosine : mean={genuine_scores.mean():.3f}  min={genuine_scores.min():.3f}")
    print(f"  Impostor cosine : mean={impostor_scores.mean():.3f}  max={impostor_scores.max():.3f}")
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
        "gallery_size": int(index.ntotal),
        "num_genuine_pairs": int(len(genuine_scores)),
        "num_impostor_pairs": int(len(impostor_scores)),
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
    plt.title("ROC Curve — Iris Recognition (verification, all pairs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = args.output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)

    # --- Histogram genuine vs impostor ---
    plt.figure(figsize=(7, 4))
    plt.hist(impostor_scores, bins=60, alpha=0.6, label="Impostor (khác người)", color="tab:red", density=True)
    plt.hist(genuine_scores,  bins=60, alpha=0.6, label="Genuine (cùng người)", color="tab:green", density=True)
    plt.axvline(eer_threshold, color="black", linestyle="--", label=f"EER thr = {eer_threshold:.3f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Mật độ")
    plt.title("Phân phối điểm: Genuine vs Impostor")
    plt.legend()
    plt.tight_layout()
    hist_path = args.output_dir / "score_distribution.png"
    plt.savefig(hist_path, dpi=130)

    print(f"\nROC curve    : {roc_path}")
    print(f"Score hist   : {hist_path}")
    print(f"Report       : {args.output_dir / 'eval_report.json'}")


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
