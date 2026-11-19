# experiment.md: next-best-destination 3 tháng tới

Tổng hợp các quyết định đã chốt (đến 2026-11-20). Quy tắc làm việc cho agent ở `OPC.md`.

## 1. Bài toán

- Tại mốc snapshot t, với mỗi khách warm, xếp hạng 18 destination và đưa **1 top 3** cho 3 tháng tới.
- Nhãn: tập destination khách **đặt** (`transaction_date`) trong (t, t+3 tháng].
- Gợi ý offline: không có context search (không amount, số người, ngày đi của lần đặt sắp tới). Input chỉ là quá khứ đến t và lịch.
- Ngoài phạm vi hiện tại: khách cold (hybrid, popularity), rerank chống popularity bias, sameDest, propensity đi/không đi (chỉ xét khách có ≥ 1 destination trong cửa sổ), feature lifetime / feature store.

## 2. Dữ liệu

Mỗi dòng là một giao dịch.

| Cột | Ý nghĩa | Ghi chú |
|---|---|---|
| `customer_id` | string (uuid) | |
| `transaction_id` | string | không phải key: có id nhiều dòng (share bill, đi chung, khác khách) |
| `destination` | khu du lịch, **target**, 18 giá trị | mỗi destination thuộc đúng 1 tỉnh |
| `total_amount` | giá trị giao dịch | |
| `transaction_date` | ngày đặt | mốc thời gian chính |
| `updated_date` | ngày check-out | không có check-in |
| `channel` | ONLINE / OFFLINE | |
| `operator_name` | dịch vụ cụ thể (vd sân golf tại A) | tập đóng, đuôi dài; mỗi operator thuộc đúng 1 destination |
| `province` | tỉnh của destination | hàm của destination |
| `region` | Bắc / Trung / Nam | hàm của destination |
| `city_v_cid` | tỉnh của khách | rất nhiều null |
| `trip_id` | gán bằng rule | đếm riêng mỗi khách từ 1; gom giao dịch có `updated_date` liên tiếp cách nhau ≤ 7 ngày (đã xác nhận) |

## 3. EDA (sau dedup, đã bỏ test/OOT; số làm tròn)

**Thời gian**
- Số giao dịch theo năm ổn định. Theo tháng có mùa vụ rõ: đỉnh 6-7-8, giảm mạnh 10-11-12, tăng đều đến tháng 3. Theo `transaction_date` cùng dạng, dịch trái nhẹ. Lead time đặt → check-out khoảng 1 tháng (trung bình / trung vị, ước lượng).
- Gap `updated_date` giữa 2 trip liên tiếp: ≤ 90 ngày khoảng 1/3, ≤ 180 ngày khoảng 1/2, < 1 năm khoảng 70%.

**Cold start**
- Khách chỉ xuất hiện ở OOT: 13%. Mỗi tháng thêm vài nghìn đến khoảng chục nghìn khách mới.

| Số giao dịch / khách | % khách | % giao dịch |
|---|---|---|
| 1 | 58% | 36% |
| 2 | 15% | 20% |
| 3-5 | 10% | 23% |
| 6+ | 3% | 20% |

min 1, median 1, P90 3, P99 11, max 375.

**Repeat** (trip thứ 2 trở đi; trip nhiều destination lấy destination có amount lớn nhất làm đại diện)
- Destination mới: 45%. Lặp destination của trip gần nhất: 43%. Destination cũ hơn: khoảng 12%.
- Tỷ lệ lặp destination gần nhất giảm đều khi gap tăng.

**Popularity**
- Destination: top 1 37%, top 2 30%, top 3 10%, top 4 6%; 14 destination còn lại mỗi cái ≤ 3,3%.
- Operator: top chủ yếu thuộc destination top, đuôi rất dài.
- Gap càng dài, trip càng dồn vào top 4 destination.

## 4. Snapshot và chia tập

- Feature của snapshot = mọi giao dịch có `transaction_date` từ đầu tập (2020) đến hết tháng snapshot. Label = 3 tháng kế tiếp.

| Snapshot (hết tháng) | Vai trò | Label |
|---|---|---|
| 2025-01 | Train + OOS | 02-04/2025 |
| 2025-04 | Train + OOS | 05-07/2025 |
| 2025-07 | Train + OOS | 08-10/2025 |
| 2025-10 | Train + OOS | 11/2025-01/2026 |
| 2026-01 | Train + OOS | 02-04/2026 |
| 2026-04 | OOT | 05-07/2026 |
| 2026-05 | OOT | 06-08/2026 |
| 2026-06 | OOT | 07-09/2026 |
| 2026-07 | OOT | 08-10/2026 |
| 2026-09 | Infer | (10-12/2026) |

- Không leak giữa train và OOT: label train kết thúc 30/04/2026, label OOT bắt đầu 01/05/2026.
- OOS = validation cùng thời gian, chia theo `customer_id`. OOT = test theo thời gian, báo cáo theo từng snapshot (label các snapshot OOT chồng nhau).
- Một dòng = (khách, snapshot). Gộp mọi snapshot train thành một bảng, train một lần.
- Warm: có ít nhất 1 trip trước t là được train.
- Không loại năm nào.

## 5. Dựng dòng tại mốc t

- Lọc giao dịch `transaction_date` ≤ t, gom theo `trip_id`. Trip đang dở ở mốc cắt chỉ lấy phần đã đặt.
- Bản ghi trip: destination đại diện (amount lớn nhất), danh sách operator kèm tỷ trọng amount, tổng amount, channel, quy mô (số giao dịch), tháng.
- Chuỗi: L trip gần nhất (L khoảng 8-10), **đệm trái**.
- Giao dịch đặt ≤ t nhưng check-out > t: nằm trong feature (cờ "có trip chưa đi").
- Label: tập destination có `transaction_date` trong (t, t+3 tháng], **bỏ** giao dịch thuộc trip đã có giao dịch đặt ≤ t (đặt thêm trong cùng trip). Mỗi giao dịch là một nhãn; nhiều giao dịch cùng destination đếm theo số lần cho soft target, đếm một lần cho metric.

## 6. Model: SMLP4Rec mở trục feature + anchor

Tensor: `(B, F, L+1, d)`. L vị trí trip + 1 vị trí anchor ở cuối.

**Field theo trip**

| # | Field | Mã hoá |
|---|---|---|
| 0 | destination | embedding 18 + padding, **dùng chung với embedding output** |
| 1 | operator | EmbeddingBag gộp theo tỷ trọng amount; emb(operator) + emb(destination của nó); operator hiếm → OOV theo từng destination |
| 2 | thương hiệu operator | nếu tách được từ tên (Vinpearl, Sun World...), dùng chung giữa destination |
| 3 | loại dịch vụ | nếu gán được (resort, golf, khu vui chơi, tour...) |
| 4 | amount của trip | log, bucket |
| 5+ | channel, quy mô, tháng của trip | thêm dần theo ablation |

Province / region: không làm field (là hàm của destination); nếu dùng thì cộng vào embedding destination.

**Anchor** (vị trí cuối; không phải query search)
- Field 0 = mask học được + tổng embedding:
  - mùa của cửa sổ = emb(tháng t+1) + emb(tháng t+2) + emb(tháng t+3) (12 embedding tháng; 4 cửa sổ train đã phủ đủ 12 tháng, nên snapshot tháng chưa thấy như 9 vẫn ghép được);
  - số ngày từ lần đặt gần nhất (bucket);
  - cờ có trip chưa đi tại t;
  - số trip đã có (bucket);
  - tỉnh khách (`unknown` cho null);
  - channel của lần đặt gần nhất.
- Field ≥ 1 tại vị trí anchor: mask học riêng cho từng field.
- Không đưa năm.

**Forward**: dựng tensor → `n_layers` (≥ 2) lần khối `SMLP([F, L+1, d])` upstream → đọc field 0 tại vị trí anchor → LayerNorm → nhân embedding destination → 19 cột (cột 0 padding = `-inf`).

**Loss**: soft-target CE trên 18 cột, target = phân phối số giao dịch theo destination trong cửa sổ.

**Cách làm**: lớp mới `src/nbp/models/smlprec_trip.py` kế thừa `SMLPREC`, không dùng `FeatureSeqEmbLayer` (nó tra feature tĩnh theo item, còn feature ở đây đổi theo trip).

**Kích thước**: L 8-10, d 32-64, n_layers 2-3, dropout 0 + weight decay.

## 7. Đánh giá

- Mỗi (khách, snapshot) đúng 1 top 3, một lần gọi model tại t (không gọi 3 lần theo tháng rồi gộp: điểm 3 lần gọi không so được nếu không có trọng số tỷ lệ đặt theo tháng).
- Metric: `user_set_metrics(k=3)`: recall = |top3 ∩ truth| / |truth|, NDCG chuẩn hoá theo min(3, |truth|). Báo thêm Recall@1.
- Monitor early stopping: NDCG@3 trên OOS. `valid_nll` = soft CE trên OOS.
- Sai số / bootstrap gộp theo `customer_id`. Chọn cấu hình bằng one-SE rule như dự án cũ.
- Nên tách metric theo nhóm: label lặp destination gần nhất / destination mới; số trip lịch sử (1, 2, 3-5, 6+).

## 8. Baseline và ablation

**Baseline bắt buộc**
- Popularity toàn cục (train).
- Popularity theo mùa của cửa sổ (theo tháng snapshot / 3 tháng label).
- Repeat: tỷ trọng destination trong lịch sử khách, trọng số theo độ gần.
- ItemKNN.
- GBDT trên cùng feature anchor + tóm tắt lịch sử (khách ít trip nên GBDT có thể ngang hoặc hơn).

**Ablation** (mỗi bước thêm một thứ)
1. Destination only (F = 1), đệm trái.
2. + anchor.
3. + operator.
4. + thương hiệu / loại dịch vụ.
5. + amount, channel, quy mô, tháng của trip.

Có thể kiểm giá trị operator trước khi train: so P(dest kế | dest trước) với P(dest kế | operator trước, làm trơn về dest trước) bằng log-loss / Recall@3 trên phần giữ lại.

## 9. Khả thi đã đo (CPU, dữ liệu ngẫu nhiên, 4 luồng)

| Cấu hình (B = 1024, 2 layer, dropout 0) | Tham số | Dòng/giây |
|---|---|---|
| L=8, d=64, 1000 operator | 100k | 1.615 |
| L=8, d=32 | 45k | 3.829 |
| L=12, d=64, 3000 operator | 229k | 1.063 |
| L=20, d=128 | 508k | 510 |

- Khoảng 10 phút / 1 triệu dòng / epoch ở L=8, d=64. Số dòng xấp xỉ 3 × số trip có lịch sử khi snapshot hàng tháng; với 5 snapshot quý thì ít hơn.
- Dropout 0.1: chậm khoảng 5,6 lần (`aten::bernoulli_`).
- `n_layers = 1`: gradient từ field ≥ 1 ở vị trí lịch sử = 0. Bắt buộc ≥ 2.

## 10. Lưu ý đúng đắn

- Không có padding mask: vị trí đệm ra hằng số qua bias LayerNorm của khối SMLP (chấp nhận được, tín hiệu độ dài ngầm).
- Loss nhỏ nhất là entropy của target, không phải 0: theo dõi NDCG@3, không theo loss tuyệt đối.
- Dòng của cùng khách ở các snapshot gần nhau tương quan: số dòng hiệu dụng nhỏ hơn số dòng.
- Code SMLP4Rec trong repo khớp code upstream của tác giả (2 chỗ `# ADAPTED`). Chưa đối chiếu được với bài báo (ACM chặn tải). Hành vi upstream giữ nguyên: một khối SMLP dùng chung cho mọi layer, không residual, `expansion_factor` = 3.

## 11. Câu hỏi còn mở

1. Số operator thực tế và phân bố mỗi destination (ngưỡng OOV).
2. Tách được thương hiệu / loại dịch vụ từ `operator_name` không.
3. Thêm snapshot train hàng tháng hoặc trước 2025 không.
4. Infer 2026-09 trong khi OOT có label đến 10/2026: dữ liệu kết thúc khi nào, có retrain trên cả OOT trước khi infer không.
5. Tỷ lệ giao dịch label bị loại vì thuộc trip đã có booking trước t.
