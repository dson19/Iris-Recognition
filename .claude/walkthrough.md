# Walkthrough — Iris Recognition System (1:N)

## Trạng thái hiện tại

| Bước | Script | Trạng thái |
|---|---|---|
| 1. Preprocess ảnh | `preprocess.py` | ✅ Done |
| 2. Split dataset | `split_dataset.py` | ✅ Done |
| 3. Generate rough masks | `generate_rough_masks.py` | ✅ Done |
| 4. Label thủ công | Labelme | ⏳ Cần làm tay |
| 5. Convert annotations | `convert_labelme_masks.py` | ✅ Done |
| 6. Train U-Net | `train_unet.py` | ✅ Done |
| 7. Normalize iris | `normalize_iris.py` | ✅ Done |
| 8. Train ArcFace | `train_arcface.py` | ✅ Done |
| 9. Build gallery | `build_gallery.py` | ✅ Done |
| 10. Evaluate | `evaluate.py` | ✅ Done |

---

## Bước 1 — Preprocess ảnh

**Script:** `scripts/preprocess.py`

**Công dụng:** Chuẩn hóa toàn bộ ảnh đầu vào về cùng định dạng trước khi đưa vào pipeline. Grayscale loại bỏ thông tin màu không cần thiết (ảnh NIR vốn không mang màu). CLAHE tăng local contrast giúp U-Net dễ học biên iris/pupil hơn, đặc biệt với ảnh tối hoặc thiếu sáng đồng đều.

Đọc toàn bộ ảnh gốc trong `datasets/raw/casia_interval/`, convert sang grayscale rồi áp dụng CLAHE để tăng local contrast. Giữ nguyên kích thước gốc 320×280.

```bash
python scripts/preprocess.py \
    --raw_dir datasets/raw/casia_interval \
    --output_dir datasets/preprocessed
```

Output: `datasets/preprocessed/<subject>/<eye>/<img>.jpg`

---

## Bước 2 — Split dataset

**Script:** `scripts/split_dataset.py`

**Công dụng:** Đảm bảo tính công bằng của quá trình đánh giá. Nếu model thấy ảnh của một người lúc train rồi lại gặp họ lúc test, kết quả đánh giá sẽ bị inflate (data leakage). Subject-disjoint split đảm bảo model chưa từng thấy danh tính nào trong eval set, mô phỏng đúng điều kiện thực tế khi nhận diện người mới.

Tách 249 subjects theo nguyên tắc **subject-disjoint** (train và eval không trùng subject) để tránh data leakage.

- **Train**: 200 subjects đầu → dùng để train ArcFace
- **Gallery**: 49 subjects còn lại, mỗi subject lấy 2 ảnh đầu mỗi mắt → dùng build FAISS index
- **Probe**: 49 subjects còn lại, các ảnh còn lại → dùng để evaluate

> ⚠️ Dùng `random.seed(42)` nên split là deterministic. **Không chạy lại** sau khi đã train.

```bash
python scripts/split_dataset.py \
    --preprocessed_dir datasets/preprocessed \
    --output_dir datasets
```

Output:
- `datasets/train/<subject_id>_<eye>/` (mỗi folder = 1 class identity)
- `datasets/gallery/<subject_id>_<eye>/`
- `datasets/probe/<subject_id>_<eye>/`

---

## Bước 3 — Generate rough masks

**Script:** `scripts/generate_rough_masks.py`

**Công dụng:** Giảm thời gian label thủ công. Thay vì vẽ polygon iris từ đầu cho mỗi ảnh, bước này dùng Hough Circle để tự động tạo vòng tròn xấp xỉ. Người label chỉ cần chỉnh sửa nơi Hough sai thay vì vẽ lại từ đầu — giúp tiết kiệm ~50–70% thời gian labeling.

Dùng Hough Circle Transform để tự động phát hiện vòng tròn pupil và iris, tạo rough mask 3 class làm điểm khởi đầu cho Labelme.

Cải tiến so với Hough cơ bản:
- Áp CLAHE trước khi detect để tăng contrast
- Tìm pupil chỉ trong vùng trung tâm 60% ảnh (tránh nhầm với IR reflection ở góc)
- Ưu tiên pupil gần nhất với tâm ROI
- Filter iris theo constraint: center cách pupil < 35px, ratio radius 1.8–3.5x, circle phải nằm trong ảnh

```bash
python scripts/generate_rough_masks.py \
    --input_dir datasets/preprocessed \
    --output_dir datasets/rough_masks \
    --sample 200
```

Mỗi ảnh sinh ra 2 file:
- `*_mask.png` — raw mask (giá trị pixel 0/1/2)
- `*_qc.png` — overlay màu (xanh=iris, đỏ=pupil) để kiểm tra bằng mắt

---

## Bước 4 — Label thủ công bằng Labelme

**⏳ Bước này cần làm tay — không thể tự động hóa.**

**Công dụng:** Tạo ground-truth segmentation mask chính xác để train U-Net. CASIA-Iris Interval không cung cấp mask sẵn, nên phải tự label. Đây là bước quyết định chất lượng của toàn bộ segmentation pipeline — U-Net chỉ học được biên iris/pupil chính xác nếu ground-truth chính xác.

Mở Labelme, load ảnh từ `datasets/preprocessed/`, dùng rough mask ở `datasets/rough_masks/` làm tham chiếu để vẽ polygon chính xác hơn.

Label **~100 ảnh** là đủ (nhờ pretrained encoder U-Net cần ít data hơn train từ đầu).

```bash
pip install labelme
labelme datasets/preprocessed/
```

Tên label trong Labelme:
- `iris` → class 1
- `pupil` → class 2

Lưu JSON output vào `datasets/labelme_annotations/` giữ nguyên cấu trúc thư mục.

---

## Bước 5 — Convert Labelme annotations → PNG masks

**Script:** `scripts/convert_labelme_masks.py`

**Công dụng:** Chuyển đổi định dạng annotation từ JSON (Labelme lưu tọa độ polygon) sang PNG mask (ma trận pixel integer) để PyTorch DataLoader có thể đọc trực tiếp khi train U-Net. Thứ tự vẽ (iris trước, pupil đè sau) quan trọng vì pupil nằm bên trong iris — cần đảm bảo không bị che sai class.

Convert polygon JSON của Labelme thành PNG mask 3 class (0=background, 1=iris, 2=pupil). Vẽ iris trước, pupil đè lên sau.

```bash
python scripts/convert_labelme_masks.py \
    --json_dir datasets/preprocessed
    --output_dir datasets/segmentation_masks \
    --verify
```

`--verify` sẽ in phân bố pixel của 5 mask đầu để kiểm tra nhanh.

---

## Bước 6 — Train U-Net Segmentation

**Script:** `scripts/train_unet.py`

**Công dụng:** Train model phân đoạn iris tự động. Thay vì dùng Hough Circle (chỉ phát hiện hình tròn lý tưởng, dễ sai với iris bị lông mi che hoặc ánh sáng không đồng đều), U-Net học được hình dạng iris thực tế và phân biệt 3 vùng (background, iris, pupil) chính xác hơn nhiều. Đây là nền tảng cho mọi bước phía sau.

**Architecture:** U-Net với encoder ResNet34 pretrained trên ImageNet (từ thư viện `segmentation_models_pytorch`). Không train từ đầu vì:
- Cần ít ảnh label hơn (~100 thay vì ~200)
- Train nhanh hơn
- Performance tốt hơn

**Loss:** CrossEntropy + DiceLoss (kết hợp để xử lý class imbalance — pupil chiếm pixel rất ít)

**Input:** grayscale 320×280 (1 channel)
**Output:** mask 3 class 320×280

```bash
pip install segmentation-models-pytorch albumentations
python scripts/train_unet.py \
    --img_dir datasets/preprocessed \
    --mask_dir datasets/segmentation_masks \
    --output_dir models \
    --encoder resnet34 \
    --epochs 50 \
    --batch_size 8
```

Output:
- `models/unet_best.pth` — checkpoint tốt nhất theo mean Dice
- `models/unet_history.json` — loss và Dice theo từng epoch

KPI mục tiêu: Dice iris > 0.90, Dice pupil > 0.88

---

## Bước 7 — Normalize Iris (Rubber Sheet)

**Script:** `scripts/normalize_iris.py`

**Công dụng:** Đưa iris về không gian biểu diễn chuẩn để feature extraction có thể so sánh được giữa các ảnh. Iris trong ảnh gốc có kích thước và vị trí khác nhau (người gần/xa camera, mắt to/nhỏ). Rubber Sheet Model ánh xạ polar iris ring về ảnh chữ nhật 64×512 — bất kỳ iris nào cũng có cùng kích thước và orientation chuẩn, giúp ResNet50 tập trung học texture thay vì hình dạng. Noise mask đi kèm để đánh dấu vùng lông mi che — sẽ bị loại ở bước matching.

Với mỗi ảnh trong `datasets/preprocessed/`:
1. Dùng U-Net trained ở bước 6 để predict mask
2. Extract tâm + bán kính pupil và iris từ mask bằng `cv2.findContours` + `cv2.minEnclosingCircle`
3. Áp dụng **Rubber Sheet Model** — ánh xạ polar từ iris ring về ảnh chữ nhật 64×512
4. Tạo **noise mask** — đánh dấu vùng bị lông mi che (class 0 trong normalized mask)

```bash
python scripts/normalize_iris.py \
    --img_dir datasets/preprocessed \
    --norm_dir datasets/normalized \
    --noise_dir datasets/noise_masks \
    --checkpoint models/unet_best.pth
```

Output:
- `datasets/normalized/<subject>/<eye>/<img>.png` — ảnh 64×512
- `datasets/noise_masks/<subject>/<eye>/<img>_noise.png` — binary mask 64×512

---

## Bước 8 — Train ResNet50 + ArcFace

**Script:** `scripts/train_arcface.py`

**Công dụng:** Học embedding vector phân biệt danh tính. ResNet50 extract feature từ ảnh normalized, ArcFace loss ép các embedding của cùng một người lại gần nhau và đẩy các embedding của người khác ra xa trong không gian 256D. Kết quả là mỗi người có một "chữ ký" embedding riêng — hai ảnh của cùng người sẽ có cosine similarity cao, hai ảnh của người khác sẽ thấp, ngay cả khi model chưa từng thấy người đó lúc train.

Train feature extractor trên normalized images từ `datasets/train/`.

**Architecture:**
- ResNet50 pretrained ImageNet, conv1 thích nghi 1 channel (grayscale) bằng cách average weight RGB → 1 channel
- Embedding layer: 2048 → 256D + BatchNorm + L2 normalize
- ArcFace head: margin=0.5, scale=64

Mỗi folder trong `datasets/train/` là 1 class identity (e.g., `001_L`, `001_R`).

```bash
python scripts/train_arcface.py \
    --train_dir datasets/train \
    --output_dir models \
    --epochs 60 \
    --batch_size 32
```

Output:
- `models/arcface_best.pth` — checkpoint tốt nhất theo val accuracy
- `models/arcface_meta.json` — danh sách class names
- `models/arcface_history.json` — training log

KPI mục tiêu: Rank-1 > 95%, EER < 3%

---

## Bước 9 — Build FAISS Gallery

**Script:** `scripts/build_gallery.py`

**Công dụng:** Xây dựng cơ sở dữ liệu danh tính (Gallery) để phục vụ tìm kiếm 1:N. FAISS IndexFlatIP lưu toàn bộ embedding của 49 eval subjects (2 ảnh/mắt) và cho phép tìm embedding giống nhất với probe chỉ trong một phép inner product — nhanh và chính xác tuyệt đối với dataset nhỏ. Đây là bước "đăng ký người dùng" trong hệ thống thực tế.

Extract embedding cho từng ảnh gallery, nạp vào FAISS IndexFlatIP (exact cosine similarity search).

```bash
python scripts/build_gallery.py \
    --gallery_dir datasets/normalized/gallery \
    --checkpoint models/arcface_best.pth \
    --output_dir outputs
```

> Lưu ý: `--gallery_dir` trỏ đến thư mục gallery **đã normalize** (sau bước 7), không phải gallery gốc.

Output:
- `outputs/faiss_gallery.index`
- `outputs/gallery_labels.json` — map FAISS vector index → class name

---

## Bước 10 — Evaluate

**Script:** `scripts/evaluate.py`

**Công dụng:** Đánh giá toàn diện hiệu năng hệ thống theo chuẩn biometric. Chạy toàn bộ probe set (ảnh chưa từng có trong gallery) qua pipeline, so sánh với gallery và tính Rank-1, FAR, FRR, EER. ROC curve giúp chọn threshold tối ưu tùy use case (ưu tiên bảo mật → threshold cao, ưu tiên UX → threshold tại EER). Đây là bằng chứng định lượng để đánh giá hệ thống trong báo cáo.

Chạy toàn bộ probe set qua embedder → search FAISS → tính metrics.

```bash
python scripts/evaluate.py \
    --probe_dir datasets/normalized/probe \
    --index_path outputs/faiss_gallery.index \
    --labels_path outputs/gallery_labels.json \
    --checkpoint models/arcface_best.pth \
    --output_dir outputs
```

Output:
- `outputs/eval_report.json` — Rank-1, EER, FAR, FRR, AUC, threshold
- `outputs/roc_curve.png` — ROC curve plot

---

## Cấu trúc thư mục đầy đủ

```
GR1/
├── datasets/
│   ├── raw/casia_interval/          # ảnh gốc (249 subjects, 2639 ảnh)
│   ├── preprocessed/                # grayscale + CLAHE
│   ├── rough_masks/                 # Hough mask để tham chiếu Labelme
│   ├── labelme_annotations/         # JSON từ Labelme (tự label)
│   ├── segmentation_masks/          # PNG mask 3 class (sau convert)
│   ├── normalized/                  # ảnh 64×512 sau Rubber Sheet
│   ├── noise_masks/                 # binary noise mask 64×512
│   ├── train/                       # 200 subjects × ảnh normalized
│   ├── gallery/                     # 49 subjects × 2 ảnh/mắt
│   └── probe/                       # 49 subjects × ảnh còn lại
├── models/
│   ├── unet_best.pth
│   ├── unet_history.json
│   ├── arcface_best.pth
│   ├── arcface_meta.json
│   └── arcface_history.json
├── outputs/
│   ├── faiss_gallery.index
│   ├── gallery_labels.json
│   ├── eval_report.json
│   └── roc_curve.png
├── scripts/
│   ├── preprocess.py
│   ├── split_dataset.py
│   ├── generate_rough_masks.py
│   ├── convert_labelme_masks.py
│   ├── train_unet.py
│   ├── normalize_iris.py
│   ├── train_arcface.py
│   ├── build_gallery.py
│   └── evaluate.py
└── walkthrough.md
```

---

## Dependencies

```bash
pip install opencv-python-headless numpy tqdm torch torchvision \
            segmentation-models-pytorch albumentations \
            faiss-cpu scikit-learn matplotlib labelme
```

Nếu có GPU:
```bash
pip install faiss-gpu
```
