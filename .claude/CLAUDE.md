# CLAUDE.md — Iris Recognition System (1:N)

> File gốc của dự án (gộp phần vận hành + kiến trúc). Root `CLAUDE.md` chỉ `@import` file này.
> Nguồn tham khảo: xem [ref.md](ref.md).

---

## 1. Tổng quan

Hệ thống nhận diện mống mắt 1:N theo pipeline:

```
Input Image → Iris Segmentation (U-Net) → Circle Extraction → Rubber Sheet Normalization
→ Noise Mask → Feature Extraction (ResNet50 + ArcFace) → Embedding 256D → FAISS 1:N Matching → Identity / Unknown
```

Mục tiêu: **Rank-1 > 95%, EER < 3%, Dice iris > 0.90.**

---

## 2. Scripts và thứ tự chạy

| Bước | Script | Mô tả |
|---|---|---|
| 1 | `scripts/preprocess.py` | Grayscale + CLAHE, giữ 320×280 |
| 2 | `scripts/split_dataset.py` | Subject-disjoint split (seed=42, không chạy lại) |
| 3 | `scripts/generate_rough_masks.py` | Hough Circle → rough mask để label Labelme |
| 4 | Labelme (tay) | Label ~100 ảnh, lưu JSON vào `datasets/labelme_annotations/` |
| 5 | `scripts/convert_labelme_masks.py` | JSON → PNG mask 3-class |
| 6 | `scripts/train_unet.py` | Train U-Net (encoder=resnet34, epochs=50) |
| 7 | `scripts/normalize_iris.py` | Rubber Sheet → 64×512 + noise mask |
| 8 | `scripts/train_arcface.py` | Train ResNet50 + ArcFace (epochs=60) |
| 9 | `scripts/build_gallery.py` | Extract embeddings → FAISS IndexFlatIP |
| 10 | `scripts/evaluate.py` | Rank-1, FAR, FRR, EER, ROC |

---

## 3. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Deep Learning | PyTorch |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Segmentation | U-Net (segmentation-models-pytorch, encoder=resnet34) |
| Feature Extraction | ResNet50 pretrained ImageNet |
| Metric Learning | ArcFace (margin=0.5, scale=64) |
| Retrieval | FAISS IndexFlatIP |
| Embedding | 256D, L2-normalized |

---

## 4. Dataset

- **CASIA-Iris Interval**: 249 subjects, ~2655 ảnh, ảnh NIR 320×280.
- **Split**: 200 subjects train (ArcFace) / 49 subjects eval (gallery + probe).
- **Gallery**: 49 subjects × 2 ảnh đầu mỗi mắt.
- **Probe**: 49 subjects × ảnh còn lại.
- **Quan trọng**: split dùng `random.seed(42)` — không chạy lại `split_dataset.py` sau khi đã train.

---

## 5. Cấu trúc dự án

### 5.1. Vấn đề của cấu trúc hiện tại

`scripts/` vừa là **entry point** vừa là **thư viện** → import chéo lẫn nhau:

```python
from train_arcface import IrisEmbedder, EMBED_DIM   # evaluate.py, build_gallery.py
from generate_rough_masks import detect_circles      # evaluate.py
```

Hệ quả: không tái sử dụng được logic mà không kéo theo `argparse`/`__main__`/side effects,
không test được, hyperparams hardcode rải rác. Cấu trúc chuẩn tách hẳn **logic (src/)** khỏi **entry point (scripts/)**.

### 5.2. Cấu trúc chuẩn (src-layout)

```
iris-recognition/
├── pyproject.toml              # metadata + deps + pip install -e .  (thay requirements.txt)
├── README.md
├── .gitignore
├── Dockerfile                  # đóng gói inference
│
├── configs/                    # ⭐ tách config khỏi code (Hydra/YAML)
│   ├── preprocess.yaml
│   ├── unet.yaml               # encoder, epochs, lr, batch_size...
│   ├── arcface.yaml            # margin=0.5, scale=64, embed_dim=256...
│   └── paths.yaml              # đường dẫn datasets/models/outputs
│
├── src/iris/                   # ⭐ THƯ VIỆN importable — toàn bộ logic ở đây
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocess.py       # CLAHE, grayscale (logic thuần, không argparse)
│   │   ├── split.py            # subject-disjoint, seed=42
│   │   ├── labelme.py          # json_to_mask
│   │   └── dataset.py          # torch Dataset
│   ├── segmentation/
│   │   ├── model.py            # U-Net factory (smp)
│   │   ├── circles.py          # detect_circles, extract from mask
│   │   └── normalize.py        # rubber sheet + noise mask
│   ├── recognition/
│   │   ├── embedder.py         # ⭐ IrisEmbedder sống ở ĐÂY (1 nguồn duy nhất)
│   │   ├── arcface.py          # ArcFace head/loss
│   │   └── gallery.py          # FAISS build/search
│   ├── metrics.py              # EER, FAR/FRR, Rank-1, ROC
│   └── utils/                  # seed, logging, io
│
├── scripts/                    # ⭐ entry point MỎNG — chỉ parse args + gọi src/
│   ├── 01_preprocess.py        # from iris.data.preprocess import run
│   ├── 02_split.py
│   ├── ...
│   └── 10_evaluate.py
│
├── api/                        # serving 1:N (nếu cần production thật)
│   └── main.py                 # FastAPI: POST ảnh → identity/unknown
│
├── tests/                      # ⭐ pytest
│   ├── test_normalize.py       # rubber sheet shape = 64×512
│   ├── test_metrics.py         # EER trên data giả
│   └── test_split.py           # subject-disjoint không leak
│
├── notebooks/                  # chỉ visualize, import từ src/
├── data/   models/   outputs/  # artifacts (gitignored, dùng DVC)
└── docs/                       # exercises/ + snippets/ chuyển vào đây
```

### 5.3. Nguyên tắc tách code

| Vấn đề hiện tại | Cách chuẩn giải quyết |
|---|---|
| Script import chéo nhau | Logic vào `src/iris/`, `pip install -e .` → `from iris.recognition.embedder import IrisEmbedder` ở mọi nơi |
| Hyperparams hardcode trong argparse | Gom vào `configs/*.yaml`, **log lại config mỗi lần train** để reproduce |
| Không test được | `src/` import được → `pytest` cho normalize / metrics / split |
| `seed=42` dễ bị phá | `tests/test_split.py` canh subject-disjoint, không lo chạy nhầm |
| `exercises/` + `snippets/` trùng vai | Cả hai là tài liệu học → gom vào `docs/` |

**Quy tắc vàng:** `scripts/` KHÔNG chứa logic — chỉ đọc config, gọi hàm trong `src/`, ghi output.
Xóa toàn bộ `scripts/` mà pipeline vẫn chạy được qua `import iris` thì cấu trúc đã đúng.

---

## 6. Quy ước code

- Grayscale input: 1 channel, không convert sang RGB trừ khi cần pretrained weights.
- Tất cả embedding phải L2-normalize trước khi lưu vào FAISS.
- Noise mask: binary 64×512, giá trị 1 = vùng bị che (eyelash/background), 0 = hợp lệ.
- Class segmentation: 0=background, 1=iris, 2=pupil.
- Split dùng `random.seed(42)` — **không chạy lại** sau khi đã train.

---

## 7. Lưu trữ file Markdown

Mọi file Markdown (`.md`) được tạo trong project — plan, notes, links, walkthrough, v.v. — phải lưu vào `.claude/`, không phải thư mục gốc.

Ví dụ đúng: `.claude/link.md`, `.claude/notes.md`

Thư mục `.claude/` bị ignore trong `.gitignore` và không được commit lên GitHub.
