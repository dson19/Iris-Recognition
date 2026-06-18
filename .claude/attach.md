# Mapping: Exercise → Script (theo thứ tự pipeline)

| Bước | Script | Exercise | Tên bài |
|---|---|---|---|
| Bước 1 | `scripts/preprocess.py` | `ex01_clahe.py` | CLAHE Preprocessing |
| Bước 3 | `scripts/generate_rough_masks.py` | `ex02_hough.py` | Hough Circle Transform |
| Bước 6 | `scripts/train_unet.py` | `ex05_unet.py` | U-Net Model + Loss + Dice |
| Bước 6 | `scripts/train_unet.py` | `ex06_dataset.py` | PyTorch Dataset |
| Bước 6 | `scripts/train_unet.py` | `ex07_training_loop.py` | Training Loop |
| Bước 7 | `scripts/normalize_iris.py` | `ex03_extract_circles.py` | Extract Circle Parameters từ Mask |
| Bước 7 | `scripts/normalize_iris.py` | `ex04_rubber_sheet.py` | Rubber Sheet Model |
| Bước 8 | `scripts/train_arcface.py` | `ex06_dataset.py` | PyTorch Dataset |
| Bước 8 | `scripts/train_arcface.py` | `ex07_training_loop.py` | Training Loop |
| Bước 8 | `scripts/train_arcface.py` | `ex08_embedder.py` | ResNet50 Embedder |
| Bước 8 | `scripts/train_arcface.py` | `ex09_arcface.py` | ArcFace Loss |
| Bước 9 | `scripts/build_gallery.py` | `ex08_embedder.py` | ResNet50 Embedder |
| Bước 9 | `scripts/build_gallery.py` | `ex10_faiss.py` | FAISS Build Gallery + Search |
| Bước 10 | `scripts/evaluate.py` | `ex10_faiss.py` | FAISS Build Gallery + Search |
| Bước 10 | `scripts/evaluate.py` | `ex11_metrics.py` | Biometric Metrics: Rank-1, EER, ROC |
