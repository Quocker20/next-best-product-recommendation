# EDA tập booking sạch sau khi chia theo ngày (Expedia)

Dữ liệu: Expedia Hotel Recommendations, 2.952.675 booking sau làm sạch (đã gộp burst repeat), 813.985 user, 100 `hotel_cluster`. Chia theo ngày booking: train < 2014-09-01 ≤ valid < 2014-11-01 ≤ test < 2015-01-01 (`configs/split.yaml`).
Mọi số dưới đây do `scripts/expedia_eda_split.py` tính và ghi ở `results/week6_retrain/eda_split.json`; hình ở `results/figures/split/{en,vi}/eda_s1..s9.png`. Thuật ngữ: **L** = số booking trước đó của user (mọi cửa sổ); **h** = số booking trước đó của user *trong cùng cửa sổ* (phần mà giao thức autoregressive thay bằng dự đoán). **Warm** = L ≥ 1, **cold** = L = 0 (booking đầu tiên của user).

Giới hạn cần nêu cùng EDA: `hotel_cluster` là proxy của hạng phòng / gói; chỉ một dataset; offline.

## 1. Quy mô và tỉ lệ warm / cold theo từng tập (S1) — hình `eda_s1_volume`

| Tập | Khoảng ngày | Booking | Tỉ lệ | User | Warm (L ≥ 1) | Cold (L = 0) | % warm | % cold |
|---|---|---|---|---|---|---|---|---|
| train | 2013-01-07 → 2014-08-31 | 2.180.488 | 73,85 % | 664.727 | 1.515.761 | 664.727 | 69,51 % | 30,49 % |
| valid | 2014-09-01 → 2014-10-31 | 393.399 | 13,32 % | 226.064 | 315.858 | 77.541 | 80,29 % | 19,71 % |
| test | 2014-11-01 → 2014-12-31 | 378.788 | 12,83 % | 222.428 | 307.071 | 71.717 | 81,07 % | 18,93 % |

- Cold ở train cao hơn hẳn (30,49 %) vì mỗi user có đúng một booking đầu trong train; ở valid/test lịch sử đã tích lũy nên warm chiếm ~80 %.
- Số booking mỗi tháng tăng mạnh từ 2014-03 (135.235) và ổn định ~185–199 nghìn/tháng ở valid/test; train gồm cả giai đoạn 2013 thấp (68–94 nghìn/tháng). Số user mới mỗi tháng trong 2014 nằm trong khoảng 24.531–43.992 (valid/test: 35.636–39.480).

## 2. User có mặt ở test nhưng chưa từng có mặt ở train (S1)

| Tập | User | User không có booking nào trong train | Tỉ lệ | Booking của user đã có trong train |
|---|---|---|---|---|
| valid | 226.064 | 77.541 | 34,30 % | 71,30 % |
| test | 222.428 | 87.008 | **39,12 %** | 64,16 % |

- **39,12 % user ở test chưa từng xuất hiện trong train** (87.008 / 222.428); 60,88 % đã có mặt trong train.
- Trong 87.008 user đó, 71.717 (32,24 % user test) là user mới hoàn toàn (booking đầu tiên nằm trong test, đúng bằng số dòng cold test); 15.291 user còn lại xuất hiện lần đầu ở valid.
- Do đó "cold dòng" (18,93 % booking test) và "user chưa có trong train" (39,12 % user test) là hai khác biệt: cold tính theo booking, tỉ lệ kia tính theo user. Mô hình không dùng embedding user nên không bị ảnh hưởng trực tiếp, nhưng lịch sử trong train của các user này bằng 0.

## 3. User và độ dài lịch sử (S2) — hình `eda_s2_users`

- 39,34 % user chỉ có 1 booking; trung vị 2, phân vị 90 = 8, 99 = 26; 7,16 % user có ≥ 10 booking; mật độ user × cluster 2,90 %.
- Độ dài L trung bình: train 4,458; valid 7,288; test 7,571. Phần dòng có L > 20 (bị cắt ở 20 mục): train 4,37 %, valid 9,38 %, test 9,78 %.
- Phân bố h (lịch sử trong cùng cửa sổ): ở test 58,72 % dòng có h = 0, 19,79 % h = 1, 8,71 % h = 2, 8,74 % h = 3–5, 4,04 % h ≥ 6; tức 41,28 % dòng test có ít nhất một booking trước đó trong cửa sổ test (valid: 42,54 %). Đây là phần mà giao thức autoregressive thay đổi so với teacher-forced.
- Khoảng cách giữa hai booking liên tiếp của một user (ngày, test): trung vị 23,1; phân vị 25 % = 2,4; 13,79 % booking cách booking trước ≤ 1 giờ, 20,34 % ≤ 1 ngày.

## 4. Mục tiêu: phân bố cluster và độ lệch giữa các tập (S3) — hình `eda_s3_target`

- Phân bố cluster khá phẳng: train top-1 4,04 %, top-5 13,62 %, top-10 22,86 %, entropy chuẩn hóa 0,9587, Gini 0,3385. Test tập trung hơn một chút: top-10 24,10 %, Gini 0,3669.
- JSD (bit) so với train: valid 0,00374; test 0,00687; hạng cluster theo train tương quan Spearman 0,9742 (valid) và 0,9576 (test). Độ lệch có thật nhưng nhỏ.
- Độ lệch tăng dần theo thời gian: JSD theo tháng so với train đi từ 0,00359 (2014-09) đến 0,00977 (2014-12), cao nhất ở tháng cuối của test.
- 10 cluster đứng đầu theo train phủ 22,23 % target valid và 24,10 % target test; 50 cluster đứng đầu phủ 75,42 % và 75,75 %.
- Cluster đổi thị phần nhiều nhất giữa train và test: 89 (0,65 % → 0,06 %), 15 (1,22 % → 1,80 %), 36 (1,27 % → 1,70 %), 12 (0,45 % → 0,86 %).

## 5. Ngữ cảnh chuyến đi: thành phần và dịch chuyển (S4) — hình `eda_s4_context`

| Tập | solo | couple | family | group | gói (`is_package`) | mobile |
|---|---|---|---|---|---|---|
| train | 28,29 % | 42,43 % | 21,25 % | 7,85 % | 14,62 % | 9,43 % |
| valid | 33,14 % | 47,92 % | 10,38 % | 8,43 % | 11,68 % | 9,99 % |
| test | 28,56 % | 49,73 % | 12,76 % | 8,84 % | 10,15 % | 12,89 % |

- Nhóm gia đình giảm từ 21,25 % (train) xuống 12,76 % (test). Đây là xu hướng giảm dài hạn, không chỉ do mùa: theo tháng booking, family giảm từ 30,61 % (2013-06) xuống 10,10 % (2014-09); tỉ lệ gói giảm từ 21,72 % (2014-01) xuống 9,41 % (2014-12) (hình `eda_s1b_trip_mix`). `is_package` giảm từ 14,62 % xuống 10,15 %, mobile tăng từ 9,43 % lên 12,89 %.
- Trung vị lead time 16 → 15 → 14 ngày, trung vị số đêm 2 ở cả ba tập.
- Trong test, cold khác warm: cold có 55,25 % couple, 15,31 % family, solo 20,01 % (warm: 48,44 %, 12,17 %, 30,56 %); gói 18,93 % (cold) so với 8,09 % (warm); mobile 16,18 % so với 12,13 %.
- JSD train–test theo trường (bit): `ci_month` 0,430 (áp đảo, xem mục 6), `children` 0,0135, `site` 0,0061, `channel` 0,0051, `lead` 0,0040; các trường còn lại < 0,004.

## 6. Mùa vụ (S5) — hình `eda_s5_season`

- Valid và test nằm trọn ở mùa thu–đông nên check-in month rất lệch so với train: JSD train–valid 0,40972; train–test 0,43. Check-in tháng 12 chiếm 42,59 % và tháng 11 chiếm 23,18 % dòng test; tháng 1 năm sau chiếm 17,02 %. 34,37 % dòng test có check-in sang năm 2015.
- Mọi tháng check-in của test đều xuất hiện trong train (100 %), nhưng chỉ ở năm 2013 hoặc trước 2014-09 với khối lượng nhỏ hơn nhiều (train: tháng 12 chỉ 4,61 %).
- Cluster theo mùa: với cùng tháng check-in, JSD phân bố cluster giữa train và test là 0,018 (tháng 12), 0,026 (tháng 11), 0,010 (tháng 1) nhưng 0,03–0,04 cho các tháng hè (số mẫu test nhỏ). Hàm ý: mùa vụ là nguồn dịch chuyển chính của bài toán, cần báo cáo slice theo `checkin_month`.
- Cluster test so với đúng mùa năm trước (tháng 11–12/2013): JSD 0,00745, không nhỏ hơn so với train toàn bộ (0,00687), tức "cùng mùa năm trước" không giải thích được độ lệch; phần lớn độ lệch đến từ xu hướng 2014.

## 7. Destination (S6) — hình `eda_s6_destinations`

- 36.933 destination; test dùng 17.125 destination (valid 18.177, train 33.260). Top-10 chiếm 11,44 % booking, top-100 35,26 %, top-1000 72,20 %.
- Destination ở test chưa từng có trong train: 0,71 % dòng (2.706 dòng); valid 0,62 %. Dưới 5 booking train: 2,17 % (test); dưới 20: 5,59 %. Tức hầu như mọi truy vấn test có destination đã thấy khi train.
- Destination mang rất nhiều thông tin: entropy cluster trung bình theo destination (≥ 20 booking train, 7.202 destination) là 3,95 bit so với 6,37 bit của phân bố biên; cluster phổ biến nhất theo destination chiếm trung bình 20,38 %.
- Dòng test theo destination: 89.610 dòng có destination đã nằm trong lịch sử user, 217.461 dòng warm có destination mới với user, 71.717 dòng cold.

## 8. Lặp lại trong lịch sử (S7) — hình `eda_s7_repeats`

Dòng test (toàn bộ, kể cả cold): cluster đã có trong lịch sử đầy đủ 24,79 %, destination 23,66 %, cặp (destination, cluster) 12,25 %. Tách theo nguồn: trong lịch sử *trước cửa sổ test* lần lượt 17,77 %, 15,31 %, 6,90 %; còn lại 7,01 %, 8,35 %, 5,35 % chỉ có được nhờ booking trong chính cửa sổ test — phần này bị mất hoặc thay bằng dự đoán dưới giao thức autoregressive.
- Khi destination đã có trong lịch sử, 51,78 % dòng test đặt lại đúng cluster đã đặt ở đó: cơ sở của sameDest.
- Theo h (lịch sử trong cửa sổ, test): tỉ lệ cluster đã có trong lịch sử đầy đủ tăng từ 14,03 % (h = 0) lên 31,85 % (h = 1), 40,18 % (h = 2), 48,00 % (h = 3–5), 63,07 % (h ≥ 6). Cặp (destination, cluster): 5,60 %, 17,41 %, 21,72 %, 25,48 %, 34,55 %.

## 9. Trường truy vấn nào mang thông tin (S8) — hình `eda_s8_information`

Lượng thông tin giữ lại (bit, cluster | trường, ước lượng trên train, đo trên test; entropy biên 6,37 bit): `dest` 2,017; `country` 0,196; `site` 0,124; `posa` 0,112; `stay` 0,061; `lead` 0,053; `dest_type` 0,044; `package` 0,036; `channel` 0,023; `adults` 0,011; `children` 0,009; `rooms` 0,001; `mobile` 0,001. `ci_month` âm (−0,022 trên test, −0,007 trên valid): phân bố check-in tháng của test lệch khỏi train nên bảng ước lượng từ train gây hại trực tiếp (hệ quả của mục 6). Ứng viên `user_region` 0,177 nhưng `user_city` âm (−0,071); cả hai không dùng làm đầu vào.

## 10. Kích thước slice cho RQ1 (S9) — hình `eda_s9_slices`

Test (378.788 dòng): theo h: 222.428 / 74.957 / 32.989 / 33.093 / 15.321 (h = 0, 1, 2, 3–5, 6+); theo L: 71.717 (0), 56.722 (1), 41.237 (2), 73.904 (3–5), 55.566 (6–10), 42.591 (11–20), 37.051 (21+); theo party: solo 108.182, couple 188.368, family 48.345, group 33.487, unknown 406; gói 38.432 / không gói 340.356; theo check-in month: tháng 12 là 161.329, tháng 11 là 87.795, tháng 1 là 64.464, các tháng 4–10 mỗi tháng dưới 9.000 (tháng 10 chỉ 1.114).
Hàm ý: slice `unknown` party (406) và tháng check-in 9–10 (2.127 và 1.114 dòng) quá nhỏ cho CI hẹp; slice mùa nên gộp (Nov–Dec, Jan–Mar, còn lại).

## Điểm cần đưa vào báo cáo

1. Warm / cold: test 81,07 % / 18,93 %; 39,12 % user test chưa từng có trong train (32,24 % là user mới hoàn toàn).
2. 41,28 % dòng test có booking trước đó trong cùng cửa sổ test, nên giao thức autoregressive ảnh hưởng tới ~41 % dòng.
3. Dịch chuyển chủ yếu theo mùa (JSD check-in month 0,43) và thành phần nhóm khách (family 21,25 % → 12,76 %); phân bố cluster chỉ lệch nhẹ (JSD 0,00687).
4. Destination là tín hiệu mạnh nhất (2,017 bit) và gần như luôn đã thấy trong train (99,29 % dòng test).
5. Số liệu mô tả, không phải kết luận nhân quả; mọi slice dùng cùng split và cùng hàng với đánh giá mô hình.
