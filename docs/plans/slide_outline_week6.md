# Dàn ý slide báo cáo (week-6: chia theo ngày, đánh giá autoregressive)

Mỗi slide có ba phần: tiêu đề dạng khẳng định, nội dung chính, nguồn số liệu. Mọi số lấy từ `results/week6_retrain/` (`final_tables.md`, `eda_split.json`, `bootstrap_ci.csv`, `slices_*.csv`, `samedest_diagnostic.json`, `split_counts.json`). Slide đánh dấu **[thiếu]** cần thêm phân tích trước khi viết.

## Phần A. Bài toán và dữ liệu (EDA)

Nguồn chung: `results/week6_retrain/eda_split.json`, hình `results/figures/split/vi/`. Không trình bày quá trình làm sạch; số liệu là của tập booking đã sạch.

1. **Tiêu đề**: Next-best-product recommendation cho du lịch (Vinpearl × GSM, đề tài C1). Tên, người hướng dẫn, ngày.
2. **Đặc thù gợi ý du lịch và RQ1**: ít booking mỗi người, lựa chọn phụ thuộc chuyến đi (điểm đến, ngày, số người), mùa vụ mạnh. RQ1: mô hình chuỗi có xét ngữ cảnh có hơn CF truyền thống trên dữ liệu thưa, có mùa vụ và phụ thuộc ngữ cảnh chuyến đi không?
3. **Expedia: 2.952.675 booking, 813.985 user, 100 cluster, 2013-01 → 2014-12**: mỗi dòng = một lần tìm kiếm dẫn tới booking (điểm đến, ngày check-in/out, số người lớn/trẻ em/phòng, gói, kênh, thiết bị); `hotel_cluster` = mục tiêu, là proxy cho hạng phòng / gói; thông tin khách sạn chỉ là nhãn. Giới hạn: một dataset, offline, không có giá.
4. **Lượng booking tăng gấp đôi trong 2014, user mới đều đặn mỗi tháng**: 2013 khoảng 68–94 nghìn booking/tháng; từ 2014-03 (135.235) tăng lên ~185–199 nghìn ở valid/test; user mới 24.531–43.992/tháng trong 2014. Hình `eda_s1_volume`. (S1 monthly)
5. **Dữ liệu rất thưa: 39,34 % user chỉ đặt một lần**: trung vị 2 booking/user, phân vị 90 = 8, phân vị 99 = 26; 7,16 % user có ≥ 10 booking; mật độ user × cluster 2,90 %. Hình `eda_s2_users` (trái). (S2)
6. **Nhịp quay lại: hoặc rất nhanh hoặc cách nhiều tuần**: khoảng cách giữa hai booking liên tiếp (test) trung vị 23,1 ngày, phân vị 75 % = 94,7 ngày; 13,79 % booking cách booking trước ≤ 1 giờ, 20,34 % ≤ 1 ngày (giả thuyết: nhiều phòng / nhiều chặng cho cùng một chuyến). (S2 days_between_bookings)
7. **100 cluster phân bố khá đều, không có món "bán chạy" áp đảo**: top-1 4,04 %, top-10 22,86 %, entropy chuẩn hóa 0,9587 (train); 50 cluster phổ biến nhất của train phủ 75,75 % target test, nên gợi ý theo độ phổ biến không đủ. Hình `eda_s3_target` (trái). (S3)
8. **Điểm đến tập trung và quyết định cluster**: 36.933 điểm đến, top-100 chiếm 35,26 % booking, top-1000 chiếm 72,20 %; biết điểm đến, độ bất định của cluster giảm từ 6,37 xuống 3,95 bit, cluster phổ biến nhất tại một điểm đến chiếm trung bình 20,38 %. Hình `eda_s6_destinations` (trái). (S6)
9. **Ngữ cảnh chuyến đi đa dạng**: train có couple 42,43 %, solo 28,29 %, family 21,25 %, group 7,85 %; gói 14,62 %, mobile 9,43 %; trung vị đặt trước 16 ngày, ở 2 đêm. Hình `eda_s4_context` (trái, giữa). (S4 composition)
10. **Mùa vụ mạnh và cơ cấu khách thay đổi theo thời gian**: check-in tập trung vào mùa hè (tháng 8 chiếm 14,38 %, tháng 7 12,60 % trong train; tháng 1 chỉ 4,80 %); tỉ lệ family giảm từ 30,61 % (2013-06) xuống 10,10 % (2014-09), tỉ lệ gói giảm từ 21,72 % (2014-01) xuống 9,41 % (2014-12). Hình `eda_s5_season`, `eda_s1b_trip_mix`. (S5, S1 monthly)
11. **Khách có lặp lại, nhất là tại điểm đến cũ**: ở test, 24,79 % booking có cluster đã đặt trước đó, 23,66 % có điểm đến đã đặt trước đó; khi điểm đến đã có trong lịch sử, 51,78 % đặt lại đúng cluster cũ. Tỉ lệ cluster lặp tăng theo lịch sử trong cửa sổ: 14,03 % (h = 0) → 63,07 % (h ≥ 6). Hình `eda_s7_repeats`. (S7)
12. **Lần tìm kiếm hiện tại mang nhiều thông tin hơn lịch sử**: lượng thông tin về cluster (bit, đo trên test): điểm đến 2,017; quốc gia user 0,196; site 0,124; số đêm 0,061; lead time 0,053; gói 0,036; các trường còn lại < 0,03. Dẫn sang hướng context-aware. Hình `eda_s8_information`. (S8)

## Phần B. Thiết kế đánh giá

13. **Mô hình hóa bài toán**: đầu vào = lịch sử booking + lần tìm kiếm hiện tại; đầu ra = xếp hạng 100 cluster; full ranking, không lấy mẫu negative.
14. **Chia theo ngày booking**: train < 2014-09-01 ≤ valid < 2014-11-01 ≤ test < 2015-01-01; 73,85 / 13,32 / 12,83 %. Hình `eda_s1_volume`.
15. **Warm / cold và user mới mỗi tập**: bảng warm/cold (train 69,51 / 30,49; valid 80,29 / 19,71; test 81,07 / 18,93); 39,12 % user test chưa từng có trong train (32,24 % mới hoàn toàn). Nguồn `eda_split.json` S1.
16. **Giao thức autoregressive (chính) và teacher-forced (phụ)**: trong cửa sổ test, cluster thật không vào lịch sử, dùng top-1 dự đoán thay; 41,28 % dòng test có booking trước đó trong cùng cửa sổ nên bị ảnh hưởng. Sơ đồ minh họa một user có 3 booking test.
17. **Test lệch khỏi train chủ yếu theo mùa**: JSD tháng check-in 0,43 so với JSD cluster 0,00687; nhóm gia đình 21,25 % (train) → 12,76 % (test), một phần do xu hướng giảm từ 2013 (slide 10). Hình `eda_s5_season`, `eda_s4_context`.
18. **Chỉ số và quy trình**: NDCG@K (chính), Recall@K, K = 5/10/20; 3 seed, mean ± std; early stopping trên NDCG@10 valid AR; bootstrap CI theo user. Có thể chuyển vào phụ lục.

## Phần C. Mô hình

19. **Hai baseline**: ItemKNN (chỉ lịch sử) và độ phổ biến toàn cục (cùng một ranking theo tỷ lệ booking của cluster ở train, không lịch sử, không ngữ cảnh).
20. **SMLP4Rec gốc chỉ đọc lịch sử**: kiến trúc MLP cho chuỗi; bỏ qua lần tìm kiếm hiện tại.
21. **Query embedding**: đưa 14 trường tìm kiếm vào mô hình (lý do: slide 12).
22. **sameDest**: cộng điểm cho cluster đã đặt tại điểm đến đang tìm (lý do: slide 11, 51,78 %).

## Phần D. Kết quả (test, AR, toàn bộ dòng, NDCG@10)

23. **Bảng chính**: độ phổ biến toàn cục 0,1225; ItemKNN 0,1825; plain 0,1840; + query embedding 0,4403; + sameDest (chính) 0,4478 (seed báo cáo 2024).
24. **Bậc thang đóng góp**: + sameDest hơn + query embedding +0,0074 [+0,0070, +0,0079]; mô hình chính hơn độ phổ biến toàn cục +0,3253 [+0,3237, +0,3268]. Biểu đồ thác nước.
25. **Warm và cold**: cold 0,4138 so với độ phổ biến toàn cục 0,1114 (sameDest không tác động trên cold); warm 0,4557 so với 0,1250.
26. **Theo h**: dưới AR, mô hình chính giữ ổn định 0,4462–0,4537 theo h; dưới TF tăng 0,4537 → 0,5721; chênh lệch cho thấy TF phóng đại giá trị lịch sử. Nguồn: `slices_h.csv`.
27. **AR so với TF**: mô hình chính mất 0,0312 NDCG@10 khi chuyển sang AR (0,4790 → 0,4478), sameDest giảm từ +0,0198 xuống +0,0074; độ phổ biến toàn cục không đổi (không dùng lịch sử). Nguồn: bảng AR − TF.
28. **Theo mùa check-in**: mô hình chính hơn độ phổ biến toàn cục ở mọi tháng check-in (ví dụ tháng 12: 0,4478 so với 0,1349; tháng 11: 0,4702 so với 0,1411). Nguồn: `slices_checkin_month.csv`.
29. **[thiếu] Theo ngữ cảnh chuyến đi** (solo / couple / family / group, `is_package`) và **điểm đến đã biết / mới**: chưa tính lại trên split mới (week-5 có, split cũ).
30. **[thiếu] RQ4 đa dạng / độ phủ**: intra-list diversity, popularity bias, độ phủ theo điểm đến / mùa; chưa tính lại trên split mới. Nếu bỏ, ghi rõ trong phần giới hạn.

## Phần E. Kết luận

31. **Trả lời RQ1**: mô hình context-aware hơn CF truyền thống rõ rệt; phần lớn lợi ích đến từ lần tìm kiếm hiện tại, lịch sử thêm một bước nhỏ nhưng có ý nghĩa thống kê.
32. **Giới hạn và hướng tiếp theo**: cluster là proxy, chỉ một dataset, offline, chỉ hai baseline đơn giản; seed 2022 của query embedding dừng ở giới hạn 10 epoch; rủi ro sameDest tự xác nhận dưới AR (12,10 % dòng, `samedest_diagnostic.json`).
33. **Demo** (tùy chọn): trang demo trực tiếp; cần trỏ lại checkpoint week-6 trước khi dùng.

## Phụ lục
- A1. 14 trường của query embedding.
- A2. Công thức Recall@K / NDCG@K.
- A3. Công thức sameDest và điểm cuối cùng.
- A4. Cấu hình mô hình và lưới tinh chỉnh.
- A5. Bảng đầy đủ: all / warm / cold × AR / TF × K = 5, 10, 20.
- A6. Kích thước slice (S9) và các slice quá nhỏ.
