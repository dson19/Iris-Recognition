"""
Convert Labelme polygon annotations (JSON) → 3-class PNG masks.

Label names expected in Labelme:
  "iris"   → pixel value 1
  "pupil"  → pixel value 2
  background → pixel value 0

Usage:
  python scripts/convert_labelme_masks.py \
      --json_dir datasets/labelme_annotations \
      --output_dir datasets/segmentation_masks \
      --verify
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

LABEL_MAP = {"iris": 122, "pupil": 255}

# Masks are stored on disk at 0/122/255 (easy to view), but downstream
# (verify, U-Net training) reasons in contiguous class ids 0/1/2.
MASK_VALUE_TO_CLASS = {0: 0, 122: 1, 255: 2}


def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for pixel_val, class_id in MASK_VALUE_TO_CLASS.items():
        out[mask == pixel_val] = class_id
    return out


def json_to_mask(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw iris first (class 1), then pupil on top (class 2)
    shapes = sorted(
        data.get("shapes", []),
        key=lambda s: LABEL_MAP.get(s["label"].lower(), 0),
    )
    for shape in shapes:
        class_id = LABEL_MAP.get(shape["label"].lower(), 0)
        if class_id == 0:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], class_id)

    return mask


def convert_all(json_dir: Path, output_dir: Path):
    json_files = sorted(json_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} annotations")
    failed = 0

    for jf in tqdm(json_files, desc="Converting"):
        rel = jf.relative_to(json_dir)
        out_path = output_dir / rel.parent / (jf.stem + "_mask.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mask = json_to_mask(jf)
            cv2.imwrite(str(out_path), mask)
        except Exception as e:
            print(f"  [ERROR] {jf.name}: {e}")
            failed += 1

    print(f"\nDone. {len(json_files) - failed} converted, {failed} failed.")
    print(f"Output: {output_dir}")


def verify_masks(mask_dir: Path, n: int = 5):
    masks = sorted(mask_dir.rglob("*_mask.png"))[:n]
    print(f"\n--- Sanity check (first {len(masks)} masks) ---")
    for mp in masks:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        m = remap_mask(m)   # 0/122/255 trên đĩa → class ids 0/1/2
        total = m.size
        iris_pct = 100 * (m == 1).sum() / total
        pupil_pct = 100 * (m == 2).sum() / total
        ok = "OK" if iris_pct > 1 and pupil_pct > 0.5 else "WARN"
        print(f"  [{ok}] {mp.name}: bg={100-(iris_pct+pupil_pct):.1f}%  "
              f"iris={iris_pct:.1f}%  pupil={pupil_pct:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=Path, default=Path("datasets/preprocessed"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/segmentation_masks"))
    parser.add_argument("--verify", action="store_true", help="Print class distribution after conversion")
    args = parser.parse_args()

    convert_all(args.json_dir, args.output_dir)
    if args.verify:
        verify_masks(args.output_dir)
