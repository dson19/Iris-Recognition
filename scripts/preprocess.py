"""
Preprocessing pipeline for CASIA-Iris Interval dataset.
- Convert to grayscale + apply CLAHE
- Save preprocessed images to datasets/preprocessed/
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def preprocess_image(img_path: Path) -> cv2.typing.MatLike:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = apply_clahe(gray)
    return enhanced


def preprocess_dataset(raw_dir: Path, output_dir: Path):
    image_paths = sorted(raw_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images in {raw_dir}")

    skipped = 0
    for img_path in tqdm(image_paths, desc="Preprocessing"):
        # Preserve relative folder structure: subject/eye/filename
        rel_path = img_path.relative_to(raw_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            enhanced = preprocess_image(img_path)
            cv2.imwrite(str(out_path), enhanced)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            skipped += 1

    print(f"Done. {len(image_paths) - skipped} saved, {skipped} skipped.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CASIA iris images")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("datasets/raw/casia_interval"),
        help="Path to raw CASIA images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/preprocessed"),
        help="Output directory for preprocessed images",
    )
    args = parser.parse_args()

    preprocess_dataset(args.raw_dir, args.output_dir)
