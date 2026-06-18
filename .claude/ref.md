# ref.md — Nguồn tham khảo cấu trúc & MLOps chuẩn production

Nguồn cho cấu trúc trong [CLAUDE.md](CLAUDE.md). Tất cả đều thật, có thể trích dẫn trong báo cáo.

---

## 1. Template để clone trực tiếp

| Template | Link | Ghi chú |
|---|---|---|
| **Cookiecutter Data Science** (DrivenData) | `github.com/drivendata/cookiecutter-data-science` | Template DS/ML được trích dẫn nhiều nhất — gốc của src-layout |
| **Lightning-Hydra-Template** (ashleve) | `github.com/ashleve/lightning-hydra-template` | Sát dự án nhất: PyTorch Lightning + Hydra + configs |
| **pytorch-template** (victoresque) | `github.com/victoresque/pytorch-template` | PyTorch thuần, nhẹ, dễ đọc cho người mới |

---

## 2. Công cụ chuẩn production

| Công cụ | Link | Dùng để |
|---|---|---|
| **Hydra** | `hydra.cc` | Quản lý config YAML (thay argparse) |
| **DVC** | `dvc.org` | Version control cho `datasets/` và `models/` |
| **MLflow** | `mlflow.org` | Track experiment + model registry |
| **Weights & Biases** | `wandb.ai` | Track experiment (alternative MLflow) |
| **PyTorch Lightning** | `lightning.ai` | Tách training loop khỏi boilerplate |

---

## 3. Đọc về nguyên lý (citable)

| Nguồn | Link | Nội dung |
|---|---|---|
| **"Made With ML"** — Goku Mohandas | `madewithml.com` | Khóa MLOps miễn phí: code → test → deploy |
| **Chip Huyen — *Designing ML Systems*** (O'Reilly) | + course **CS329S** (Stanford) | Chuẩn thiết kế hệ thống ML |
| **Google — "Rules of Machine Learning"** | `developers.google.com/machine-learning/guides/rules-of-ml` | 43 quy tắc thực chiến (Martin Zinkevich) |
| **PyPA — "src layout vs flat layout"** | `packaging.python.org` | Lý do kỹ thuật của `src/` |
| **The Twelve-Factor App** | `12factor.net` | Nguyên tắc tách config/code/deploy |
