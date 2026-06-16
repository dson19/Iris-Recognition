"""
Iris normalization pipeline:
  1. Load preprocessed image + U-Net predicted mask
  2. Extract pupil/iris circle parameters from mask
  3. Apply Rubber Sheet Model → normalized image (64×512)
  4. Generate noise mask (eyelash/occlusion regions)

Usage:
  # Predict masks first (if not already done):
  python scripts/normalize_iris.py --predict_masks

  # Then normalize:
  python scripts/normalize_iris.py
"""

import cv2
import numpy as np
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp

NORM_H = 64
NORM_W = 512
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Mask prediction with trained U-Net
# ---------------------------------------------------------------------------

def load_unet(checkpoint: Path, encoder: str, device: torch.device):
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_mask(model, gray: np.ndarray, device: torch.device) -> np.ndarray:
    """Returns predicted 3-class mask (H×W, uint8)."""
    img = gray.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # 1×1×H×W
    logits = model(tensor)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Circle extraction from mask
# ---------------------------------------------------------------------------

def extract_circles(mask: np.ndarray):
    """
    Extract (cx, cy, r) for pupil (class 2) and iris (class 1) from mask.
    Returns (pupil_circle, iris_circle), each (cx, cy, r) or None.
    """
    results = {}
    for cls_id, name in [(2, "pupil"), (1, "iris")]:
        binary = (mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            results[name] = None
            continue
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest)
        results[name] = (int(cx), int(cy), int(r))
    return results.get("pupil"), results.get("iris")


# ---------------------------------------------------------------------------
# Rubber Sheet Model
# ---------------------------------------------------------------------------

def rubber_sheet(img: np.ndarray, mask: np.ndarray, pupil: tuple, iris: tuple):
    """
    Maps iris ring from polar to rectangular coordinates.

    Returns:
      normalized (NORM_H × NORM_W, uint8)
      noise_mask (NORM_H × NORM_W, uint8) — 1 = occluded/noise, 0 = valid
    """
    px, py, pr = pupil
    ix, iy, ir = iris

    theta = np.linspace(0, 2 * np.pi, NORM_W, endpoint=False)
    r_vals = np.linspace(0, 1, NORM_H)

    # Broadcast: (NORM_H, NORM_W)
    theta_grid = theta[np.newaxis, :]           # 1 × NORM_W
    r_grid     = r_vals[:, np.newaxis]          # NORM_H × 1

    # Pupil and iris boundary points at each angle
    pupil_x = px + pr * np.cos(theta_grid)
    pupil_y = py + pr * np.sin(theta_grid)
    iris_x  = ix + ir * np.cos(theta_grid)
    iris_y  = iy + ir * np.sin(theta_grid)

    # Interpolated sample coordinates
    sample_x = (1 - r_grid) * pupil_x + r_grid * iris_x   # NORM_H × NORM_W
    sample_y = (1 - r_grid) * pupil_y + r_grid * iris_y

    # Remap image and mask
    map_x = sample_x.astype(np.float32)
    map_y = sample_y.astype(np.float32)

    normalized = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
    norm_mask  = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # Noise mask: 1 where normalized mask is background (class 0), 0 = valid iris
    noise = (norm_mask == 0).astype(np.uint8)

    return normalized, noise


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load U-Net
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"U-Net checkpoint not found: {args.checkpoint}\n"
                                f"Run train_unet.py first.")
    model = load_unet(args.checkpoint, args.encoder, device)
    print(f"Loaded U-Net from {args.checkpoint}")

    image_paths = sorted(args.img_dir.rglob("*.jpg"))
    print(f"Processing {len(image_paths)} images...")

    args.norm_dir.mkdir(parents=True, exist_ok=True)
    args.noise_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for img_path in tqdm(image_paths):
        rel = img_path.relative_to(args.img_dir)
        norm_path  = args.norm_dir  / rel.parent / (rel.stem + ".png")
        noise_path = args.noise_dir / rel.parent / (rel.stem + "_noise.png")
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        noise_path.parent.mkdir(parents=True, exist_ok=True)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed += 1
            continue

        mask = predict_mask(model, gray, device)
        pupil, iris = extract_circles(mask)

        if pupil is None or iris is None:
            print(f"  [WARN] Circle extraction failed: {img_path.name}")
            failed += 1
            continue

        try:
            normalized, noise = rubber_sheet(gray, mask, pupil, iris)
        except Exception as e:
            print(f"  [WARN] Rubber sheet failed {img_path.name}: {e}")
            failed += 1
            continue

        cv2.imwrite(str(norm_path),  normalized)
        cv2.imwrite(str(noise_path), noise * 255)   # save as 0/255 for easy viewing

    print(f"\nDone. {len(image_paths) - failed} OK, {failed} failed.")
    print(f"Normalized : {args.norm_dir}")
    print(f"Noise masks: {args.noise_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rubber Sheet normalization for CASIA iris images")
    parser.add_argument("--img_dir",    type=Path, default=Path("datasets/preprocessed"))
    parser.add_argument("--norm_dir",   type=Path, default=Path("datasets/normalized"))
    parser.add_argument("--noise_dir",  type=Path, default=Path("datasets/noise_masks"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/unet_best.pth"))
    parser.add_argument("--encoder",    type=str,  default="resnet34")
    args = parser.parse_args()
    process_dataset(args)
