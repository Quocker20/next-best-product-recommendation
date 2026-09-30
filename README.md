# Next-Best-Product Recommendation (Topic C1)

Dự án nghiên cứu và benchmark các họ mô hình Next-Best-Product Recommendation cho bài toán nghỉ dưỡng (Vinpearl); ngữ cảnh di chuyển / ẩm thực (GSM) dự kiến cho Q1/2027.

**Phạm vi hiện tại (khóa 2026-09-18):** chỉ domain du lịch, một dataset chính là **Expedia Hotel Recommendations** (Trivago 2019 là phương án dự phòng, chưa làm). **Phương pháp chính (chốt 2026-09-30):** SMLP4Rec làm bộ chấm điểm lõi, kết hợp **hybrid có trọng số** cho mọi người dùng: `điểm = w_m·log p_SMLP4Rec + w_p·log p_prior(điểm đến) + w_s·sameDest`, trọng số theo nhóm số lần đặt trước đó, chỉnh trên tập validation; người dùng mới (chưa có lịch sử) chỉ dùng prior điểm đến. Query token và ngữ cảnh theo từng lần đặt bên trong mô hình là phương án dự phòng. AdaGIN là mô hình so sánh có ngữ cảnh, LightGBM là baseline dạng bảng, quy tắc “cụm đã đặt tại cùng điểm đến trước, rồi prior” là mốc không học để vượt. Domain food, ride và cross-sell chưa nằm trong phạm vi. Chi tiết: `CLAUDE.md`, tiến độ và quyết định: `PROGRESS.md`.

**Trạng thái:** tuần 3 đã chạy thử pipeline SMLP4Rec trên Expedia (3 epoch, chỉ dùng lịch sử đặt phòng); kết quả ở `reports/summary/week3/`. Các thử nghiệm hybrid (fusion với prior điểm đến, người dùng mới, MAP@5) nằm ở các nhánh local `exp/*` và được ghi trong `PROGRESS.md`. Đây là kiểm tra pipeline, chưa phải kết quả benchmark. Kế hoạch 10 ngày dựng lại pipeline dạng module: `docs/plan_oct_01_14.md`.

## Cấu trúc thư mục (Directory Structure)

```text
next-best-product-recommendation/
├── CLAUDE.md                  # Quy chuẩn dự án, phạm vi nghiên cứu & hướng dẫn phát triển
├── README.md                  # Tài liệu tổng quan về repository
├── PROGRESS.md                # Nhật ký từng phase: quyết định, kết quả, việc còn mở (chỉ local, gitignore)
├── pyproject.toml             # Cấu hình dự án và khai báo dependencies
├── configs/                   # Cấu hình chạy mô hình (smlprec_expedia.yaml)
├── data/                      # Dữ liệu dự án (được gitignore, không commit)
│   ├── interim/               # Bảng trung gian (vd. recbole/expedia/expedia.inter, log, checkpoint)
│   └── raw/                   # Dữ liệu thô theo 4 nhóm domain (11 dataset)
│       ├── food/              # akeed, instacart, yelp
│       ├── hospitality/       # airbnb_new_user, expedia, hotel_booking_demand, trivago_2019
│       ├── prototype/         # movielens
│       └── ride/              # citibike, nyc_tlc, porto_taxi
├── docs/                      # Tài liệu kỹ thuật & thiết kế (Tiếng Anh)
│   ├── data_cards/            # Data card cho 11 dataset (kết quả EDA, schema, RQ fit)
│   ├── dataset_rubric.md      # Khung đánh giá Rubric (5 cổng sàng lọc + 10 tiêu chí)
│   ├── dataset_scores.md      # Bảng chấm điểm chi tiết 11 dataset ứng viên
│   ├── eda_checklist.md       # Checklist quy trình EDA chuẩn cho dataset
│   ├── plan_oct_01_14.md      # Kế hoạch 10 ngày (1–14/10): dựng lại pipeline dạng module + hybrid
│   └── problem_statement_vi.md # Đề bài gốc (tiếng Việt)
├── docs_vi/                   # Phiên bản Tiếng Việt song song của docs/
│   ├── data_cards/            # Data card tiếng Việt cho các dataset
│   ├── dataset_rubric.md      # Khung đánh giá Rubric tiếng Việt
│   ├── dataset_scores.md      # Bảng chấm điểm 11 dataset tiếng Việt
│   └── eda_checklist.md       # Checklist EDA tiếng Việt
├── notebooks/                 # Notebook phân tích, EDA theo từng domain
│   ├── food/                  # Notebooks EDA cho nhóm Food Delivery
│   ├── hospitality/           # Notebooks EDA cho nhóm Hospitality
│   ├── prototype/             # Notebooks thử nghiệm mẫu (MovieLens)
│   └── ride/                  # Notebooks EDA cho nhóm Ride / Mobility
├── reports/                   # Báo cáo tiến độ & tài liệu trình chiếu
│   ├── slides/                # Slide báo cáo theo tuần (week1, week2, week3), HTML + PDF
│   └── summary/               # Kết quả theo tuần: week1_dataset_selection, week2_methodology, week3
├── sample_data/               # Dữ liệu mẫu dung lượng nhỏ phục vụ kiểm thử pipeline
├── scripts/                   # Script tiện ích, tiền xử lý & tự động hóa
│   ├── download.py            # Script tự động tải các bộ dữ liệu
│   ├── dataset_scoring.py     # Script tính điểm rubric tự động
│   ├── profile_expedia.py     # Script profiling dữ liệu chi tiết Expedia
│   ├── expedia_*.py           # EDA, tín hiệu ngữ cảnh, lát cắt lịch sử (nguồn các mốc tuần 2)
│   ├── expedia_to_recbole.py  # Xuất đặt phòng Expedia sang định dạng .inter của RecBole
│   ├── run_smlprec_expedia.py # Chạy huấn luyện + đánh giá SMLP4Rec trên Expedia (tuần 3)
│   ├── recommend_smlprec_expedia.py # Gợi ý cụm tiếp theo cho một người dùng từ checkpoint đã lưu
│   └── week*_*_slides.py      # Sinh slide từng tuần từ file kết quả JSON
└── src/                       # Mã nguồn cốt lõi (reusable core modules)
    └── models/smlprec.py      # SMLP4Rec chép từ fork, chỉnh tối thiểu (đánh dấu # ADAPTED)
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

> Ghi chú: mục này mô tả cách chạy hiện tại (RecBole) và sẽ được thay theo `docs/plan_oct_01_14.md`: bỏ RecBole, dùng một môi trường `.venv` có torch, mã dạng module trong `src/`. Cập nhật lại sau khi Ngày 5 của kế hoạch qua bước kiểm tra.

`.venv` ở trên dùng pandas 3 / numpy 2 cho EDA và scripts. RecBole 1.0.1 cần numpy 1.23 và pandas 1.5, nên chạy SMLP4Rec trong venv Python 3.11 riêng, đặt ở đường dẫn **ngắn** (đường dẫn dài làm Windows không nạp được DLL của torch):

```bash
py -3.11 -m venv C:/Users/<user>/.venvs/smlp4rec
# cài torch (CPU), numpy==1.23.5, pandas==1.5.3, scipy==1.10.1, PyYAML, tqdm, rồi:
pip install --no-deps -e <đường-dẫn-tới-bản-clone-MLP4Rec>
```

RecBole 1.0.1 cần hai bản vá nhỏ: `weights_only=False` cho `torch.load` (PyTorch ≥ 2.6) và tắt anomaly detection do `sine.py` bật khi import. Chạy:

```bash
python scripts/expedia_to_recbole.py                  # trong .venv của dự án
python scripts/run_smlprec_expedia.py                 # trong venv smlp4rec
```

Kết quả ghi vào `reports/summary/week3/`.
