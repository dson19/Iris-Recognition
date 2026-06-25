"""
Subject-disjoint split for CASIA-Iris Interval dataset.

Chạy SAU normalize_iris.py — chia thẳng ảnh normalized (64×512 PNG) thành các tập,
nên không cần normalize lại từng split.

Split strategy:
  - Train  : first 200 subjects  → datasets/train/<subject_id>_<eye>/
  - Gallery: last 49 subjects, 2 images per subject/eye → datasets/normalized_gallery/
  - Probe  : last 49 subjects, remaining images         → datasets/normalized_probe/

Tên thư mục đầu ra khớp với default của train_arcface / build_gallery / evaluate.
Treats each (subject, eye) pair as one identity class.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


TRAIN_SUBJECTS = 200
GALLERY_IMGS_PER_EYE = 2
SEED = 42


def collect_subjects(normalized_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Returns {subject_id: {eye: [img_paths]}} sorted by filename."""
    subjects: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for img_path in normalized_dir.rglob("*.png"):
        # Directory structure: <subject_id>/<eye>/<filename>.png
        eye_dir = img_path.parent
        subject_dir = eye_dir.parent
        eye = eye_dir.name        # "L" or "R"
        subject_id = subject_dir.name  # "001", "002", ...
        subjects[subject_id][eye].append(img_path)

    # Sort images by filename for determinism
    for subject_id in subjects:
        for eye in subjects[subject_id]:
            subjects[subject_id][eye].sort()

    return subjects


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_dataset(normalized_dir: Path, output_dir: Path):
    subjects = collect_subjects(normalized_dir)
    subject_ids = sorted(subjects.keys())
    total = len(subject_ids)
    print(f"Total subjects found: {total}")

    random.seed(SEED)
    random.shuffle(subject_ids)

    train_ids = subject_ids[:TRAIN_SUBJECTS]
    eval_ids  = subject_ids[TRAIN_SUBJECTS:]
    print(f"Train: {len(train_ids)} subjects | Eval: {len(eval_ids)} subjects")

    train_dir   = output_dir / "train"
    gallery_dir = output_dir / "normalized_gallery"
    probe_dir   = output_dir / "normalized_probe"

    # --- Train set ---
    print("\nCopying train set...")
    for subject_id in tqdm(train_ids, desc="Train"):
        for eye, imgs in subjects[subject_id].items():
            class_name = f"{subject_id}_{eye}"
            for img in imgs:
                copy_file(img, train_dir / class_name / img.name)

    # --- Gallery + Probe ---
    print("\nCopying gallery and probe sets...")
    gallery_count = 0
    probe_count = 0
    for subject_id in tqdm(eval_ids, desc="Eval"):
        for eye, imgs in subjects[subject_id].items():
            class_name = f"{subject_id}_{eye}"
            gallery_imgs = imgs[:GALLERY_IMGS_PER_EYE]
            probe_imgs   = imgs[GALLERY_IMGS_PER_EYE:]

            for img in gallery_imgs:
                copy_file(img, gallery_dir / class_name / img.name)
                gallery_count += 1

            for img in probe_imgs:
                copy_file(img, probe_dir / class_name / img.name)
                probe_count += 1

            if len(probe_imgs) == 0:
                print(f"  [WARN] {subject_id}/{eye} has <= {GALLERY_IMGS_PER_EYE} images, no probe samples.")

    print(f"\nDone.")
    print(f"  Train  : {sum(len(imgs) for sid in train_ids for imgs in subjects[sid].values())} images → {train_dir}")
    print(f"  Gallery: {gallery_count} images → {gallery_dir}")
    print(f"  Probe  : {probe_count} images → {probe_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split normalized CASIA dataset into train/gallery/probe")
    parser.add_argument(
        "--normalized_dir",
        type=Path,
        default=Path("datasets/normalized"),
        help="Path to normalized iris images (output of normalize_iris.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets"),
        help="Root output directory (creates train/, normalized_gallery/, normalized_probe/ inside)",
    )
    args = parser.parse_args()

    split_dataset(args.normalized_dir, args.output_dir)
