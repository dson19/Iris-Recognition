"""
Generate rough 3-class segmentation masks using Hough Circle Transform.
Use these as starting points for manual correction in Labelme.

Output mask values:
  0 = Background
  1 = Iris ring
  2 = Pupil
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm

random.seed(42)


def detect_circles(gray: np.ndarray):
    """Returns (pupil_circle, iris_circle) each as (cx, cy, r) or None."""
    h, w = gray.shape
    cx_img, cy_img = w // 2, h // 2

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)

    # --- Pupil: search only in center 60% of image ---
    margin_x, margin_y = int(w * 0.20), int(h * 0.20)
    roi = blurred[margin_y:h - margin_y, margin_x:w - margin_x]
    rh, rw = roi.shape

    pupil_raw = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=60, param2=18,
        minRadius=int(min(h, w) * 0.06),
        maxRadius=int(min(h, w) * 0.20),
    )
    if pupil_raw is None:
        return None, None

    # Pick pupil closest to ROI center
    best = min(pupil_raw[0], key=lambda c: np.hypot(c[0] - rw // 2, c[1] - rh // 2))
    px = int(best[0]) + margin_x
    py = int(best[1]) + margin_y
    pr = int(best[2])

    # --- Iris: constrained by pupil ---
    max_iris_r = int(min(h, w) * 0.42)
    iris_raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=18,
        minRadius=int(pr * 1.8),
        maxRadius=max_iris_r,
    )
    if iris_raw is None:
        return (px, py, pr), None

    def _score_iris(c):
        ix, iy, ir = int(c[0]), int(c[1]), int(c[2])
        dist = np.hypot(ix - px, iy - py)
        ratio = ir / pr
        # Circle must be mostly inside image (center > 50% radius from each edge)
        inside = (ix - ir * 0.5 > 0 and ix + ir * 0.5 < w and
                  iy - ir * 0.5 > 0 and iy + ir * 0.5 < h)
        if not inside or not (1.8 <= ratio <= 3.5):
            return None
        return (dist, ix, iy, ir)

    candidates = [s for c in iris_raw[0] if (s := _score_iris(c)) is not None and s[0] < 35]
    if not candidates:
        # Fallback: relax distance constraint
        candidates = [s for c in iris_raw[0] if (s := _score_iris(c)) is not None]

    if not candidates:
        return (px, py, pr), None

    candidates.sort()
    _, ix, iy, ir = candidates[0]
    return (px, py, pr), (ix, iy, ir)


def make_mask(h: int, w: int, pupil_circle, iris_circle) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if iris_circle is not None:
        cv2.circle(mask, (iris_circle[0], iris_circle[1]), iris_circle[2], 1, -1)
    if pupil_circle is not None:
        cv2.circle(mask, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], 2, -1)
    return mask


def generate_masks(input_dir: Path, output_dir: Path, sample: int = 0):
    image_paths = sorted(input_dir.rglob("*.jpg"))
    if sample > 0:
        image_paths = random.sample(image_paths, min(sample, len(image_paths)))

    print(f"Processing {len(image_paths)} images...")
    failed = 0

    for img_path in tqdm(image_paths):
        rel = img_path.relative_to(input_dir)
        out_dir = output_dir / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed += 1
            continue

        pupil_circle, iris_circle = detect_circles(gray)

        if iris_circle is None:
            print(f"  [WARN] No iris detected: {img_path.name}")
            failed += 1
            mask = np.zeros(gray.shape, dtype=np.uint8)
        else:
            mask = make_mask(gray.shape[0], gray.shape[1], pupil_circle, iris_circle)

        # Raw mask (0/1/2) for programmatic use
        cv2.imwrite(str(out_dir / (rel.stem + "_mask.png")), mask)

        # Color-coded overlay for visual QC
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(vis)
        overlay[mask == 1] = [0, 200, 0]   # iris = green
        overlay[mask == 2] = [0, 0, 200]   # pupil = red
        qc = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(out_dir / (rel.stem + "_qc.png")), qc)

    print(f"\nDone. {len(image_paths) - failed} OK, {failed} failed.")
    print(f"Output: {output_dir}")
    print("\nNext: correct masks in Labelme, then run convert_labelme_masks.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rough iris masks with Hough Circles")
    parser.add_argument("--input_dir", type=Path, default=Path("datasets/preprocessed"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/rough_masks"))
    parser.add_argument(
        "--sample", type=int, default=150,
        help="Number of images to sample for labeling (0 = all)",
    )
    args = parser.parse_args()
    generate_masks(args.input_dir, args.output_dir, args.sample)
