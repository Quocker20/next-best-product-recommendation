# new-repository

Dự án phân tích dữ liệu và phát triển ứng dụng.

## Cấu trúc thư mục (Directory Structure)

```text
next-best-product-recommendation/
├── data/
│   └── raw/               # Dữ liệu thô theo domain
│       ├── food/          # akeed, yelp
│       ├── hospitality/   # airbnb_new_user, expedia, trivago_2019
│       ├── prototype/     # movielens
│       └── ride/          # nyc_tlc, porto_taxi
├── docs/                  # Tài liệu hướng dẫn, thiết kế
│   ├── data_cards/        # Data card cho từng dataset (kết quả EDA)
│   └── eda_checklist.md   # Checklist EDA lần đầu cho một dataset
├── notebooks/              # Notebook phân tích, EDA, thử nghiệm mô hình
├── sample_data/            # Tập dữ liệu mẫu/ví dụ phục vụ kiểm thử
├── scripts/                 # Script tiện ích, xử lý dữ liệu
├── src/                      # Mã nguồn cốt lõi (core code) của dự án
├── pyproject.toml           # Cấu hình dự án và khai báo dependencies
└── README.md                 # Tài liệu tổng quan về repository
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
