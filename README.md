# Next-Best-Product Recommendation (Topic C1)

Dự án nghiên cứu và benchmark các họ mô hình Next-Best-Product Recommendation cho bài toán nghỉ dưỡng (Vinpearl) và di chuyển / ẩm thực (GSM).

## Cấu trúc thư mục (Directory Structure)

```text
next-best-product-recommendation/
├── CLAUDE.md                  # Quy chuẩn dự án, phạm vi nghiên cứu & hướng dẫn phát triển
├── README.md                  # Tài liệu tổng quan về repository
├── pyproject.toml             # Cấu hình dự án và khai báo dependencies
├── data/                      # Dữ liệu dự án (được gitignore, không commit)
│   └── raw/                   # Dữ liệu thô theo 4 nhóm domain (11 dataset)
│       ├── food/              # akeed, instacart, yelp
│       ├── hospitality/       # airbnb_new_user, expedia, hotel_booking_demand, trivago_2019
│       ├── prototype/         # movielens
│       └── ride/              # citibike, nyc_tlc, porto_taxi
├── docs/                      # Tài liệu kỹ thuật & thiết kế (Tiếng Anh)
│   ├── data_cards/            # Data card cho 11 dataset (kết quả EDA, schema, RQ fit)
│   ├── dataset_rubric.md      # Khung đánh giá Rubric (5 cổng sàng lọc + 10 tiêu chí)
│   ├── dataset_scores.md      # Bảng chấm điểm chi tiết 11 dataset ứng viên
│   └── eda_checklist.md       # Checklist quy trình EDA chuẩn cho dataset
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
│   ├── slides/                # Slide thuyết trình báo cáo (e.g. week1-report.html)
│   └── summary/               # Tóm tắt kết quả EDA & profiling chuyên sâu
├── sample_data/               # Dữ liệu mẫu dung lượng nhỏ phục vụ kiểm thử pipeline
├── scripts/                   # Script tiện ích, tiền xử lý & tự động hóa
│   ├── download.py            # Script tự động tải các bộ dữ liệu
│   ├── dataset_scoring.py     # Script tính điểm rubric tự động
│   └── profile_expedia.py     # Script profiling dữ liệu chi tiết Expedia
└── src/                       # Mã nguồn cốt lõi (reusable core modules)
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
