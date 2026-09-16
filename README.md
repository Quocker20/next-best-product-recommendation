# new-repository

Dự án phân tích dữ liệu và phát triển ứng dụng.

## Cấu trúc thư mục (Directory Structure)

```text
new-repository/
├── docs/             # Lưu trữ các tài liệu hướng dẫn, thiết kế
├── notebooks/        # Lưu trữ các notebook phân tích, EDA (Exploratory Data Analysis)
├── src/              # Mã nguồn cốt lõi (core code) của dự án
├── sample_data/      # Lưu trữ các tập dữ liệu mẫu/ví dụ phục vụ kiểm thử
├── pyproject.toml    # Cấu hình dự án và khai báo dependencies
└── README.md         # Tài liệu tổng quan về repository
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
