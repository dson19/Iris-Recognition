
# KẾ HOẠCH ĐỒ ÁN TỐT NGHIỆP
# HỆ THỐNG XÁC THỰC MỐNG MẮT 1:N (IRIS RECOGNITION)

---

# 1. Mục tiêu Đồ án

Xây dựng hệ thống nhận diện mống mắt theo mô hình 1:N Identification có khả năng:

- Nhận diện người dùng từ cơ sở dữ liệu Gallery.
- Hỗ trợ Unknown User Detection.
- Hoạt động với pipeline Deep Learning hoàn chỉnh.
- Có đánh giá biometric metrics: FAR, FRR, EER, Rank-1 Accuracy.
- Có demo thực tế.

---

# 2. Scope Đồ án

## 2.1 Các phần BẮT BUỘC phải implement

### Core Pipeline

```
Input Image
     ↓
Iris Segmentation (U-Net)
     ↓
Circle Parameter Extraction
     ↓
Iris Normalization (Rubber Sheet)
     ↓
Noise Mask Generation
     ↓
Feature Extraction (ResNet50 + ArcFace)
     ↓
Embedding Vector (256D, L2-normalized)
     ↓
1:N Matching (FAISS)
     ↓
Identity / Unknown
```

---

## 2.2 Các phần OPTIONAL (nếu còn thời gian)

### FastAPI Demo

- Endpoint `POST /enroll`: đăng ký người dùng mới vào Gallery.
- Endpoint `POST /identify`: truy vấn và trả kết quả định danh.

### PAD (Anti-Spoofing)

- Binary classifier phân biệt: Real / Fake.

---

# 3. Công nghệ Đề xuất

| Thành phần | Công nghệ |
|---|---|
| Deep Learning Framework | PyTorch |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Segmentation | U-Net |
| Feature Extraction | ResNet50 |
| Metric Learning | ArcFace |
| Retrieval | FAISS |
| API (optional) | FastAPI |

---

# 4. Dataset

## 4.1 Dataset chính — CASIA-Iris Interval

Lý do chọn:
- Phổ biến nhất trong nghiên cứu iris.
- Ảnh NIR chất lượng cao.
- Dễ benchmark với paper khác.
- Phù hợp scope thesis sinh viên.

Link: https://biometrics.idealtest.org/

Thông tin dataset:
- ~2,655 ảnh từ 249 subjects (mỗi subject: 2 mắt × nhiều ảnh).
- Kích thước gốc: 320×280.

---

## 4.2 Dataset phụ (nếu cần cross-dataset evaluation)

### IITD Iris Database

Link: https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm

---

# 5. Phân chia Dataset

## 5.1 Nguyên tắc Subject-Disjoint Split

**Quan trọng**: Phải đảm bảo các subject trong tập train và tập test **hoàn toàn không trùng nhau** để tránh data leakage.

```
Toàn bộ CASIA (249 subjects)
         ↓
   Subject-Disjoint Split
         ↓
┌─────────────────┬─────────────────┐
│  Train Set      │  Eval Set       │
│  ~200 subjects  │  ~49 subjects   │
│  (ArcFace train)│  (benchmark)    │
└─────────────────┴─────────────────┘
```

---

## 5.2 Train Set (~200 subjects)

Dùng để train ResNet50 + ArcFace.

- Mỗi subject tương ứng một class identity.
- Dùng toàn bộ ảnh của subject đó (không giới hạn 2 ảnh).
- Áp dụng augmentation mạnh để tăng diversity.

---

## 5.3 Eval Set (~49 subjects, không dùng khi train)

Chia tiếp thành:

### Gallery Set

- Mỗi subject: lấy **2 ảnh đầu tiên** làm ảnh đăng ký.
- Dùng để build FAISS index.

### Probe Set

- Mỗi subject: các ảnh còn lại dùng để truy vấn.
- Dùng để đánh giá Rank-1, FAR, FRR, EER.

---

# 6. Giai đoạn 1 — Chuẩn bị Dữ liệu

## 6.1 Tổ chức thư mục

```
dataset/
│
├── raw/                    # ảnh gốc CASIA
├── segmentation_masks/     # ground-truth masks (labeled)
├── normalized/             # ảnh đã Rubber Sheet (64×512)
├── noise_masks/            # noise mask tương ứng
├── train/                  # split cho ArcFace training
│   ├── subject_001/
│   ├── subject_002/
│   └── ...
├── gallery/                # eval: ảnh đăng ký
└── probe/                  # eval: ảnh truy vấn
```

---

## 6.2 Tiền xử lý ảnh

### Kích thước

Giữ nguyên kích thước gốc **320×280**, không resize về 320×240 vì:
- Tránh biến dạng hình học iris.
- Ảnh hưởng đến bước normalization và circle fitting.

Có thể scale lên 640×560 để tăng độ phân giải nếu cần.

### Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### CLAHE

Tăng local contrast, làm rõ iris texture:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

---

## 6.3 Data Augmentation

### Cho U-Net Segmentation Training

Augmentation mạnh hơn để bù cho số lượng label ít (100–200 ảnh):

| Augmentation | Mục tiêu |
|---|---|
| RandomRotate (±15°) | robustness góc xoay |
| GaussianBlur | robustness blur |
| RandomBrightnessContrast | robustness sáng/tối |
| ElasticTransform (nhẹ) | tăng diversity hình dạng |
| HorizontalFlip | đối xứng trái/phải |
| GridDistortion (nhẹ) | variation biên iris |

Không augment quá mạnh (crop sâu, distortion lớn) vì sẽ phá iris texture.

### Cho ResNet50 + ArcFace Training

| Augmentation | Mục tiêu |
|---|---|
| RandomRotate (±10°) | chống nghiêng đầu |
| GaussianBlur | chống blur |
| RandomBrightnessContrast | chống thay đổi sáng |
| ShiftScaleRotate | robustness vị trí |

---

# 7. Giai đoạn 2 — Iris Segmentation

## 7.1 Mục tiêu

Phân đoạn ảnh iris thành 3 class:

| Class | Ý nghĩa |
|---|---|
| 0 — Background | nền và lông mi |
| 1 — Iris | vùng iris ring |
| 2 — Pupil | vùng đồng tử |

---

## 7.2 Model — U-Net

Lý do chọn:
- Kiến trúc encoder-decoder hiệu quả với dataset nhỏ.
- Skip connections giữ detail biên tốt.
- Nhẹ, dễ train, phù hợp thesis.

---

## 7.3 Input / Output

```
Input : 320×280 grayscale image (1 channel)
Output: 320×280 multi-class mask (3 classes)
```

---

## 7.4 Loss Function

Kết hợp BCE Loss + Dice Loss:

```python
loss = BCE_loss + Dice_loss
```

Mục tiêu:
- BCE: học phân loại pixel.
- Dice: tối ưu overlap, đặc biệt tốt cho class mất cân bằng (pupil nhỏ).

---

## 7.5 Metrics

| Metric | Ý nghĩa |
|---|---|
| Dice Score (per class) | segmentation quality |
| Mean IoU | overlap trung bình |

---

## 7.6 Ground-truth Generation

### Vấn đề

CASIA-Iris Interval không có sẵn multi-class segmentation mask.

### Giải pháp — Semi-auto labeling

```
Hough Circle Transform
        ↓
Generate Rough Circular Mask (pupil + iris)
        ↓
Manual Correction bằng Labelme
        ↓
Export JSON → Convert thành PNG mask
        ↓
Training Dataset
```

### Số lượng label đề xuất

- ~100–200 ảnh đủ cho thesis.
- Kết hợp augmentation mạnh để tăng effective dataset size.
- Nếu kết quả Dice thấp: tăng thêm label hoặc dùng pre-trained weights từ dataset segmentation khác.

---

## 7.7 Xử lý Eyelash Occlusion

Lông mi được xử lý ngay trong U-Net bằng cách label class 0 (Background) cho vùng lông mi. Sau khi có mask:

- Vùng eyelash → class 0 → loại ra khi Normalization.
- Vùng bị che → đánh dấu trong Noise Mask (xem Section 8.5).

Không dùng SAM hoặc Mask R-CNN vì quá nặng và khó train trong scope thesis.

---

# 8. Giai đoạn 3 — Iris Normalization

## 8.1 Mục tiêu

Đưa iris ring (hình tròn) về ảnh chữ nhật cố định kích thước **64×512** để feature extraction dễ dàng hơn.

---

## 8.2 Circle Parameter Extraction

U-Net trả về mask, nhưng Rubber Sheet Model cần tâm và bán kính của pupil và iris.

Pipeline:

```
Mask (từ U-Net)
        ↓
Tách class Pupil (class 2) và Iris (class 1)
        ↓
cv2.findContours trên từng class
        ↓
cv2.minEnclosingCircle → (xp, yp, rp) và (xi, yi, ri)
```

Kết quả:
- `(xp, yp, rp)` — tâm và bán kính pupil.
- `(xi, yi, ri)` — tâm và bán kính iris.

---

## 8.3 Rubber Sheet Model

Ánh xạ polar từ iris ring về ảnh chữ nhật:

```
x(r, θ) = (1 - r) * xp(θ) + r * xi(θ)
y(r, θ) = (1 - r) * yp(θ) + r * yi(θ)
```

Trong đó:
- `r` ∈ [0, 1]: trục hướng kính (từ pupil ra iris boundary).
- `θ` ∈ [0, 2π]: trục góc.

Output: ảnh normalized **64×512** (height × width).

---

## 8.4 Noise Mask Generation

Sau khi normalize, cần tạo noise mask để đánh dấu các vùng không hợp lệ:

```
Normalized Image (64×512)
        ↓
Áp dụng cùng phép ánh xạ Rubber Sheet lên U-Net mask
        ↓
Vùng class 0 (background/eyelash) trong normalized mask → noise = 1
Vùng class 1 (iris) → noise = 0
        ↓
Noise Mask (64×512, binary)
```

Noise mask được lưu song song với normalized image và dùng để loại vùng bị che khi matching.

---

## 8.5 Rotation Handling

### Vấn đề

Người dùng nghiêng đầu → iris texture bị shift theo trục θ trong ảnh normalized.

### Giải pháp 1 — Data Augmentation (training)

Dùng RandomRotate trong augmentation để model học robustness với góc xoay.

### Giải pháp 2 — Multi-crop Inference (inference)

Tại bước inference:
1. Normalize ảnh với offset θ = 0, ±5°, ±10°, ±15°.
2. Extract embedding cho từng crop.
3. So sánh với gallery, lấy similarity score cao nhất.

Đây là cách chuẩn hơn so với circular shift trên embedding vector.

---

# 9. Giai đoạn 4 — Feature Extraction

## 9.1 Mục tiêu

Từ normalized iris image (64×512), sinh embedding vector 256D đặc trưng cho danh tính.

---

## 9.2 Model — ResNet50 + ArcFace

Lý do chọn ResNet50:
- Kiến trúc ổn định, nhiều paper benchmark.
- Pretrained ImageNet → transfer learning tốt.
- Dễ implement và tune.
- Phù hợp scope thesis.

Lý do không dùng ViT:
- Cần nhiều data hơn để train hiệu quả.
- Dễ mất local texture của iris.
- Hyperparameter tuning phức tạp hơn.

---

## 9.3 Pretrained Weights Strategy

```
ResNet50 pretrained trên ImageNet
        ↓
Thay đổi input layer: 1 channel (grayscale)
  - Option A: chuyển grayscale → 3 channel (repeat)
  - Option B: khởi tạo lại conv1 với 1 channel, giữ weights layer sau
        ↓
Thêm ArcFace classification head
        ↓
Fine-tune toàn bộ network trên CASIA Train Set
```

---

## 9.4 Training Phase

Mỗi subject trong Train Set tương ứng một class identity:

```python
# ArcFace head
class ArcFaceHead(nn.Module):
    # in_features: 256 (embedding dim)
    # out_features: số subjects trong train set (~200)
    ...
```

Loss: ArcFace Loss với `margin=0.5`, `scale=64`.

---

## 9.5 Inference Phase

Sau khi train xong:
- Bỏ ArcFace classification head.
- Chỉ giữ ResNet50 backbone → output 256D embedding.
- L2 normalize embedding trước khi lưu vào FAISS.

---

## 9.6 Embedding

| Thuộc tính | Giá trị |
|---|---|
| Kích thước | 256D |
| Normalization | L2 |
| Similarity | Cosine (tương đương Inner Product sau L2 norm) |

---

# 10. Giai đoạn 5 — Matching 1:N

## 10.1 Pipeline

```
Probe Image
      ↓
[Full Pipeline: Segment → Normalize → Extract]
      ↓
Probe Embedding (256D, L2-normalized)
      ↓
FAISS IndexFlatIP.search(probe, k=1)
      ↓
Top-1 similarity score + subject_id
      ↓
score > threshold ?
   ├── YES → Identity: subject_id
   └── NO  → Unknown User
```

---

## 10.2 Similarity

Cosine similarity (sau L2 normalization tương đương inner product):

```
cos(A, B) = dot(A, B) / (||A|| * ||B||)
           = dot(A, B)   [vì A, B đã L2-normalized]
```

Range: [-1, 1], cao hơn → giống hơn.

---

## 10.3 Retrieval — FAISS IndexFlatIP

Lý do chọn:
- Exact search, không mất precision.
- Đủ nhanh cho CASIA (~49 subjects × 2 ảnh = ~98 vectors trong gallery).
- Đơn giản, không cần tuning.

Không cần IVF / HNSW / PQ vì dataset nhỏ.

---

## 10.4 Noise Mask trong Matching

Khi so sánh probe với gallery entry, vùng bị che bởi eyelash cần được loại ra:

```python
# Tính effective similarity chỉ trên vùng hợp lệ
valid = (noise_mask_probe == 0) & (noise_mask_gallery == 0)
similarity = cosine_similarity(
    probe_embedding[valid], gallery_embedding[valid]
)
```

Lưu ý: Cách này áp dụng được nếu dùng IrisCode-style matching trực tiếp trên normalized image. Với deep embedding, cần xem xét integrate noise mask vào feature extraction (ví dụ masking trước khi pooling).

---

## 10.5 Unknown User Detection

```python
top1_score, top1_id = faiss_index.search(probe, k=1)

if top1_score >= THRESHOLD:
    return f"Identity: {top1_id}"
else:
    return "Unknown User"
```

Threshold được chọn dựa trên ROC curve (xem Section 11.5).

---

# 11. Giai đoạn 6 — Evaluation

## 11.1 Rank-1 Accuracy

Tỉ lệ probe được định danh đúng subject ở vị trí top-1:

```
Rank-1 = (số probe đúng top-1) / (tổng số probe)
```

---

## 11.2 FAR — False Acceptance Rate

Tỉ lệ probe của người lạ bị nhận nhầm là người đã đăng ký:

```
FAR = FP / (FP + TN)
```

---

## 11.3 FRR — False Rejection Rate

Tỉ lệ probe đúng người bị từ chối (score < threshold):

```
FRR = FN / (FN + TP)
```

---

## 11.4 EER — Equal Error Rate

Điểm mà FAR = FRR. Threshold tại EER thường được dùng làm operating point mặc định.

---

## 11.5 ROC Curve và Threshold Selection

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, scores)
# Vẽ ROC
# Tìm EER: điểm gần nhất với FAR = FRR
# Chọn threshold theo use case:
#   - Ưu tiên security: chọn threshold cao (giảm FAR)
#   - Ưu tiên UX: chọn threshold tại EER
```

---

# 12. Giai đoạn 7 — Demo Hệ thống (Optional)

## 12.1 FastAPI

```python
POST /enroll
    Input : iris image + subject_id
    Action: chạy pipeline → lưu embedding vào FAISS gallery
    Output: { "status": "enrolled", "subject_id": "..." }

POST /identify
    Input : iris image
    Action: chạy pipeline → search FAISS
    Output: { "identity": "subject_id" | "unknown", "score": 0.92 }
```

---

## 12.2 Demo Flow

```
Upload iris image
       ↓
Segmentation → Normalization → Noise Mask → Embedding
       ↓
FAISS Search
       ↓
Return { identity, score }
```

---

# 13. Timeline Đề xuất

| Tuần | Công việc | Deliverable |
|---|---|---|
| 1 | Dataset download + tổ chức thư mục + preprocessing pipeline | Script tiền xử lý chạy được |
| 2 | Semi-auto labeling ground-truth mask (100–200 ảnh) | Labelme JSON + PNG masks |
| 3 | Train U-Net segmentation | Model checkpoint |
| 4 | Evaluation segmentation (Dice, IoU) + debug | Dice Score report |
| 5 | Circle extraction + Rubber Sheet normalization + Noise Mask | Normalized images + noise masks |
| 6 | Subject-disjoint split + chuẩn bị train set ArcFace | Dataset splits |
| 7 | Train ResNet50 + ArcFace | Model checkpoint |
| 8 | Build FAISS gallery + Matching pipeline | End-to-end pipeline chạy được |
| 9 | Evaluation (Rank-1, FAR, FRR, EER, ROC) | Evaluation report |
| 10 | FastAPI demo (optional) / Buffer / Fix | Demo hoặc polish |
| 11 | Viết báo cáo + làm slide | Final report + slide |

---

# 14. KPI Mục tiêu

| Metric | Target (tốt) | Acceptable (đủ pass) |
|---|---|---|
| Dice Score (Iris class) | > 0.95 | > 0.90 |
| Dice Score (Pupil class) | > 0.92 | > 0.88 |
| Rank-1 Accuracy | > 95% | > 90% |
| EER | < 3% | < 5% |
| Inference Latency | < 2s/ảnh | < 5s/ảnh |

---

# 15. Hướng Phát triển Tương lai

Chỉ cần phân tích trong báo cáo, không cần implement:

- **Vision Transformer (ViT)**: thay ResNet50, cần nhiều data hơn.
- **PAD (Presentation Attack Detection)**: phát hiện ảnh giả.
- **Domain Adaptation**: chuyển model từ NIR sang visible light camera.
- **Cross-sensor Recognition**: generalize qua nhiều loại camera.
- **Privacy-preserving Biometrics**: cancelable biometrics, template protection.
- **Federated Learning**: train phân tán, bảo vệ dữ liệu.
- **Multi-modal Biometrics**: kết hợp iris + face + fingerprint.

---

# 16. Kết luận

Đồ án tập trung vào:
- Biometric pipeline hoàn chỉnh từ raw image đến identity.
- Deep learning (U-Net segmentation + ResNet50 + ArcFace metric learning).
- 1:N identification với Unknown User Detection.
- Evaluation thực tế theo chuẩn biometric (FAR, FRR, EER, Rank-1).

Điểm khác biệt so với các thesis thông thường:
- Subject-disjoint split đúng chuẩn, tránh data leakage.
- Noise mask xử lý eyelash occlusion đúng cách.
- Phân tách rõ training set (ArcFace) và eval set (gallery/probe).

Mục tiêu chính: hệ thống hoạt động ổn định, có metric rõ ràng, có khả năng mở rộng nghiên cứu sau này.
