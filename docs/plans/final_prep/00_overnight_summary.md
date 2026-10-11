# Chạy qua đêm 2026-10-11: chuẩn bị slide báo cáo final (đọc file này trước)

Chưa commit gì và không train lại model nào. Tất cả file nằm sẵn trên đĩa để bạn xem lại.

## File cần kiểm tra, theo thứ tự
1. `results/week5_final/final_tables.md`: kết quả tổng hợp (model × all / cold / warm, bootstrap CI, slice, chỉ số beyond-accuracy). Sinh bởi `scripts/final_results_summary.py` (`.venv` của project, khoảng 100 giây); script phụ xuất top-k là `scripts/final_export_topk_smlp4rec.py` (venv smlp4rec). Bản tóm tắt dễ đọc: `01_results_summary.md`.
2. `05_verification_report.md`: kiểm tra độc lập bằng `scripts/verify_final_results.py` (43 PASS, 0 FAIL, 1 NOTE: `reports/benchmark_results.csv` chưa có các run early-stopping). Đã chạy lại tối 2026-10-11, kết quả y hệt; `pytest` 48 passed, 1 skipped.
3. `reports/summary/week5_final/final_slides_v1.html` (+ `.pdf`): slide bản v1, 22 slide (20 chính + 2 phụ lục). Phím: mũi tên / space để chuyển slide, `n` để hiện speaker notes. Builder là `scripts/final_slides.py`; hình ở `results/figures/final/en/`.
3b. **Bản tiếng Việt (2026-10-11)**: `reports/summary/week5_final/final_slides_v1.html` (+ `.pdf`, 22 slide, cả speaker notes và nhãn hình đã dịch; hình ở `results/figures/final/vi/`) và demo `results/week5_final/demo_storyboard_vi.html` (slide 19 của bản VI trỏ tới file này). Số liệu giữ nguyên dấu chấm thập phân; `scripts/check_deck_numbers.py` xác nhận 527 số trên 22 slide của bản VI và bản EN trùng khớp (0 khác biệt). Build lại: `.venv/Scripts/python.exe scripts/final_slides.py --lang vi`; `python scripts/demo_storyboard_build.py --lang vi`; kiểm số: `.venv/Scripts/python.exe scripts/check_deck_numbers.py`; PDF: `msedge --headless=new --no-pdf-header-footer --print-to-pdf=reports/summary/week5_final/final_slides_v1.pdf <url file html>`. Bản dịch là bản nháp đầu, cần bạn đọc lại thuật ngữ (ví dụ "Giả định" cho what-if).
4. `results/week5_final/demo_storyboard.html`: demo offline với 4 booking thật trong tập test, từng bước: code gốc (01e) → + query token → + sameDest, baseline lấy từ notebook 07b (rank đã kiểm 8/8 so với rank 07b đã lưu). Phím: mũi tên / space, W = what-if, 1-4 = chọn booking, S = tóm tắt. Demo live trên checkpoint early-stopping: `scripts/demo_prototypes/serve_es.py`.
5. `02_report_toc.md` (outline slide nằm ở mục 2; mục lục này chỉ dành cho slide, không làm docx), `03_slide_style_guide.md`, `04_demo_options.md`.
6. Skill đã tải về nhưng chưa cài và chưa chạy code nào: `skills_staging/` (antislop-decks MIT, frontend-design Apache-2.0, impeccable Apache-2.0 bản một phần). Trước khi cài impeccable, hãy xoá block `allowed-tools` của nó vì block này cho phép sẵn việc chạy launcher của chính nó.

## Các quyết định cần bạn chọn
1. Logistic regression: slide dùng bản notebook 07b (train trên all event, theo quy tắc của bạn). Bản notebook 07 mạnh hơn: mức tăng NDCG@5 warm là +0.111 (so với 07b) nhưng chỉ +0.089 (so với 07). Điều này đã được ghi chú ở slide 12 và phụ lục A1.
2. Slice mùa vụ: check-in trong tập test khoảng 52 % là tháng 12 và 21 % là tháng 1, nên slice này không thể cho thấy hiệu ứng mùa vụ. Các hướng: rolling-origin split, gộp valid + test, hoặc nêu thẳng đây là giới hạn. Hiện slide nêu giới hạn.
3. Các run early-stopping chỉ có seed 2022. CLAUDE.md yêu cầu ít nhất 3 seed; mỗi seed thêm mất khoảng 3 giờ.
4. `reports/benchmark_results.csv` đã cũ: chưa có hàng early-stopping, và logistic regression trong đó là bản notebook 07.
5. Hiệu ứng của việc train với first bookings mới chỉ đo trên các run 3 epoch; slide đã nói rõ điều này.
6. Cách đặt nhãn: các bảng ghi "warm rows only (L>=1)", còn doc và slide ghi "trained on warm users only". Cần chọn một.
7. CLAUDE.md §7 chưa liệt kê `results/figures/final/` và `results/week5_final/`.
8. Speaker notes bản EN vẫn bằng tiếng Anh; bản VI (`final_slides_v1.html`) đã có notes tiếng Việt.

## Vấn đề nhỏ đã biết
- `scripts/demo_ui.py` còn 9 lỗi ruff (`noqa` thừa, một biến không dùng, một `except` quá rộng). Các script mới đều sạch lint.
- `smlprec_query_cold_L0_cold_train_es.json` ghi "checkpoint of notebook 01c" trong trường `run` nhưng thực ra trỏ tới checkpoint early-stopping của 01d.
- Trong `seeds_summary.json`, các key `qt_minus_logreg` lặp lại giá trị của `qt`. Có vẻ là lỗi trong script tổng hợp; slide không dùng các key này.
- `.claude/launch.json` được tạo để test trên trình duyệt (static server).
