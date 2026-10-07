# Next-Best-Product Recommendation (Topic C1)

Dự án nghiên cứu và benchmark các họ mô hình Next-Best-Product Recommendation cho bài toán nghỉ dưỡng (Vinpearl); ngữ cảnh di chuyển / ẩm thực (GSM) dự kiến cho Q1/2027.

**Phạm vi hiện tại (khóa 2026-09-18):** chỉ domain du lịch, một dataset chính là **Expedia Hotel Recommendations** (Trivago 2019 là phương án dự phòng, chưa làm). **Phương pháp chính (chốt 2026-09-30):** SMLP4Rec làm bộ chấm điểm lõi, kết hợp **hybrid có trọng số** cho mọi người dùng: `điểm = w_m·log p_SMLP4Rec + w_p·log p_prior(điểm đến) + w_s·sameDest`, trọng số theo nhóm số lần đặt trước đó, chỉnh trên tập validation; người dùng mới (chưa có lịch sử) chỉ dùng prior điểm đến. Query token và ngữ cảnh theo từng lần đặt bên trong mô hình là phương án dự phòng. AdaGIN là mô hình so sánh có ngữ cảnh, LightGBM là baseline dạng bảng, quy tắc “cụm đã đặt tại cùng điểm đến trước, rồi prior” là mốc không học để vượt. Domain food, ride và cross-sell chưa nằm trong phạm vi. Chi tiết: `CLAUDE.md`, tiến độ và quyết định: `PROGRESS.md`.

**Trạng thái:** tuần 3 đã chạy thử pipeline SMLP4Rec trên Expedia (3 epoch, chỉ dùng lịch sử đặt phòng); kết quả ở `reports/summary/week3_implementation/`. Các thử nghiệm luật top-5 và hybrid (fusion với prior điểm đến, sameDest, người dùng mới) đã gộp vào `master` và chạy lại trong `notebooks/hospitality/smlp4rec/`; ghi chép trong `PROGRESS.md`. Đây là kiểm tra pipeline, chưa phải kết quả benchmark. Kế hoạch 10 ngày dựng lại pipeline dạng module: `docs/plans/plan_oct_01_14.md`.

## Cấu trúc thư mục (Directory Structure)

### Quy ước: mỗi thư mục một nghĩa

| Thư mục | Chứa gì | Không chứa |
|---|---|---|
| `docs/` | Tài liệu tham chiếu không gắn tuần: data card, rubric, checklist, protocol, đề bài; `plans/` cho kế hoạch theo ngày, `explainers/` cho giải thích kỹ thuật | Kết quả chạy, báo cáo tuần |
| `reports/summary/<tuần>/` | **Chỉ báo cáo và slide** giao cho người đọc: docx, pdf, slide html, báo cáo md | json, csv, log, hình sinh tự động, **README.md (không bao giờ thêm)** |
| `results/<tuần>/` | **Output tính toán của script** (được commit): json, csv, log, bảng sinh tự động; `results/figures/{en,vi}` cho hình sinh tự động | Báo cáo viết tay, checkpoint |
| `experiments/` | Output lần chạy huấn luyện (config, metrics, log, checkpoint) | Code (gitignored) |
| `scripts/` | Script phân tích, chạy thử, sinh báo cáo/slide | Output |
| `notebooks/` | EDA theo domain; train / test / đánh giá trong `hospitality/smlp4rec/` | |
| `src/nbp/` | Code tái sử dụng (data, prior, baseline, model, hybrid, eval) có test | |

Tên thư mục tuần dùng chung cho `reports/summary/` và `results/`: `week1_dataset_selection`, `week2_methodology`, `week3_implementation`; tuần sau theo mẫu `weekN_<chủ-đề>`.

```text
next-best-product-recommendation/
├── CLAUDE.md                  # Quy chuẩn dự án, phạm vi, hướng dẫn cho agent (local, gitignore)
├── PROGRESS.md                # Nhật ký quyết định / kết quả theo phase (local, gitignore)
├── README.md
├── pyproject.toml
├── configs/                   # data.yaml (nbp), smlprec_expedia.yaml (RecBole, legacy)
├── data/                      # gitignore: raw/ (chỉ đọc), interim/, processed/
├── docs/
│   ├── data_cards/            # 11 data card
│   ├── dataset_rubric.md  dataset_scores.md  eda_checklist.md  problem_statement_vi.md
│   ├── plans/                 # plan_oct_01_14.md (kế hoạch 10 ngày)
│   └── explainers/            # smlprec_explain.md
├── docs_vi/                   # bản tiếng Việt của docs/ (local, gitignore)
├── notebooks/
│   ├── <domain>/<dataset>_eda.ipynb
│   └── hospitality/smlp4rec/  # 01 train+test, 02 luật top-5, 03 prior, 04 hybrid sameDest, 05 tách theo thói quen
├── reports/summary/           # chỉ báo cáo + slide
│   ├── week1_dataset_selection/   # eda_summary.md, week1_review_additions.md, week1_report_v3.html
│   ├── week2_methodology/         # methodology_selection_report.docx, methodology_slides.html/.pdf
│   └── week3_implementation/      # week3_report_slides_v2.html/.pdf, week3_implementation_report.docx
├── results/                   # output của script (json, csv, log, hình)
│   ├── figures/{en,vi}/
│   ├── week1_dataset_selection/
│   ├── week2_methodology/
│   └── week3_implementation/
├── scripts/                   # expedia_*.py (EDA / phân tích), run_smlprec_*.py, week*_*.py (sinh báo cáo, slide)
├── src/
│   ├── models/smlprec.py      # LEGACY (RecBole), tham chiếu cho bước port, xóa ở Ngày 10
│   └── nbp/                   # package mới: paths, config, seed, data, priors, baselines, models, hybrid, eval
├── tests/                     # pytest
├── experiments/               # output lần chạy (gitignore)
└── sample_data/
```

## Hướng dẫn cài đặt

1. **Khởi tạo môi trường ảo:**
   ```bash
   python -m venv .venv
   ```

2. **Kích hoạt môi trường ảo:**
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc:**
   ```bash
   pip install -e .
   # Cài đặt thêm các gói cho môi trường phát triển (Jupyter, pytest, ruff):
   pip install -e ".[dev]"
   ```

### Môi trường riêng cho SMLP4Rec (RecBole)

> Ghi chú: mục này mô tả cách chạy hiện tại (RecBole) và sẽ được thay theo `docs/plans/plan_oct_01_14.md`: bỏ RecBole, dùng một môi trường `.venv` có torch, mã dạng module trong `src/`. Cập nhật lại sau khi Ngày 5 của kế hoạch qua bước kiểm tra.

`.venv` ở trên dùng pandas 3 / numpy 2 cho EDA và scripts. RecBole 1.0.1 cần numpy 1.23 và pandas 1.5, nên chạy SMLP4Rec trong venv Python 3.11 riêng, đặt ở đường dẫn **ngắn** (đường dẫn dài làm Windows không nạp được DLL của torch):

```bash
py -3.11 -m venv C:/Users/<user>/.venvs/smlp4rec
# cài torch (CPU), numpy==1.23.5, pandas==1.5.3, scipy==1.10.1, PyYAML, tqdm, rồi:
pip install --no-deps -e <đường-dẫn-tới-bản-clone-MLP4Rec>
```

RecBole 1.0.1 cần hai bản vá nhỏ: `weights_only=False` cho `torch.load` (PyTorch ≥ 2.6) và tắt anomaly detection do `sine.py` bật khi import. Chạy:

```bash
python scripts/expedia_to_recbole.py                     # trong .venv của dự án
python scripts/expedia_to_recbole.py --with-destination  # bản có srch_destination_id (notebook 03)
```

Huấn luyện, test và tính metric (Recall@K, NDCG@K) nằm trong các notebook `notebooks/hospitality/smlp4rec/` (kernel `smlp4rec`, đăng ký bằng `python -m ipykernel install --user --name smlp4rec` trong venv smlp4rec):

1. `01_train_test_smlp4rec.ipynb`: train 3 epoch bằng vòng lặp tường minh, test, heuristic, so sánh với `results/week3_implementation/smlprec_expedia_run.json`.
2. `02_top5_rerank_rules.ipynb`: luật past/novel trên top-5 (quota cố định, cap theo L), dùng trọng số từng epoch của notebook 01.
3. `03_destination_prior_hybrid.ipynb`: late fusion với prior điểm đến, hybrid + sameDest, người dùng mới, bootstrap.

Mỗi notebook ghi kết quả vào `experiments/<ngày>_expedia_smlp4rec_week3-repro/` (gitignored) và so từng con số với JSON của bản chạy bằng script trong `reports/summary/week3_implementation/`.
