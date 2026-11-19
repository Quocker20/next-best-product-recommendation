# OPC.md

Hướng dẫn cho coding agent làm bài **next-best-destination 3 tháng tới** bằng SMLP4Rec trên repo này. Đọc file này trước, rồi đọc `experiment.md` (toàn bộ quyết định, số liệu EDA, thiết kế model). Khi hai file mâu thuẫn với code cũ, hai file này thắng.

## Bối cảnh

- Repo hiện chứa dự án cũ: SMLP4Rec trên Expedia hotel clusters (100 item, một booking một dòng, next-item, có query search). Code cũ là **khuôn để tái dùng**, không phải bài toán hiện tại.
- Bài mới: dữ liệu giao dịch du lịch, 18 destination (tập đóng), không có context search. Tại mỗi mốc snapshot t, xếp hạng 18 destination cho mỗi khách, lấy top 3 cho 3 tháng tới.
- Phạm vi hiện tại: khách warm (có ít nhất 1 trip trước t). Cold start, rerank chống popularity bias, sameDest, propensity đi/không đi: chưa làm.
- Feature: chỉ các cột có trong tập giao dịch. Chưa dùng feature lifetime hay feature store.

## Quy tắc bắt buộc (không được vi phạm)

1. **Mốc thời gian theo `transaction_date`** (ngày đặt), không theo `updated_date`. Feature = giao dịch có `transaction_date` ≤ cuối tháng snapshot. Label = giao dịch có `transaction_date` trong 3 tháng kế tiếp.
2. **Không đứng giữa trip.** `trip_id` gom các giao dịch có `updated_date` liên tiếp cách nhau ≤ 7 ngày. Giao dịch đặt sau t mà thuộc trip đã có giao dịch đặt ≤ t thì **không** tính là label. Feature của trip chỉ tính từ các giao dịch có `transaction_date` ≤ t.
3. **Không leak:** không feature nào được dùng thông tin có `transaction_date` > t. Vocab, biên bucket chỉ lấy từ dòng train. Không đưa `customer_id`, năm, hay `trip_id` (số thứ tự) vào model.
4. **`transaction_id` không phải key** (có id nhiều dòng, nhiều khách). Không join hay dedup chỉ theo nó.
5. **Không sửa `src/nbp/models/smlprec.py`.** File do `scripts/fetch_smlp4rec.py` sinh ra (upstream không có license, file bị gitignore). Model mới là lớp con của `SMLPREC`, giữ nguyên khối `SMLP` upstream, giống cách `src/nbp/models/smlprec_query.py` làm.
6. **`n_layers ≥ 2`.** Đã đo: với 1 layer, gradient từ field 1 trở đi ở các vị trí lịch sử bằng 0 (khối SMLP không trộn chéo trục trong một lớp), model không thấy operator, amount...
7. **Đệm trái** chuỗi trip, trip gần nhất luôn đứng sát vị trí anchor ở cuối. Không dùng đệm phải kiểu RecBole.
8. **Mùa của cửa sổ** mã hoá bằng tổng embedding 3 tháng trong cửa sổ (t+1, t+2, t+3), không bằng tháng của t (snapshot train chỉ có tháng 1, 4, 7, 10).
9. **Không gộp province/region thành field riêng**: chúng là hàm của destination. Nếu dùng thì cộng vào embedding destination.
10. **OOS chia theo `customer_id`** (hash cố định), không theo dòng.
11. Mỗi (khách, snapshot) đúng **1 top 3**. Metric: `user_set_metrics(k=3)` của `src/nbp/eval/user_level.py`.

## Môi trường

- Windows, CPU. Hai venv như README: `.venv` (pandas, numpy mới, torch CPU) cho dữ liệu và baseline; venv RecBole (RecBole 1.0.1, numpy 1.23) cho model SMLP4Rec. Module dùng ở cả hai venv chỉ được import numpy.
- Chạy `python scripts/fetch_smlp4rec.py` một lần để có `src/nbp/models/smlprec.py`.
- Import torch trước pandas (lỗi `c10.dll` trên Windows).
- Dropout trên máy này rất chậm (`aten::bernoulli_` chiếm khoảng 68% thời gian, chậm khoảng 5,6 lần). Mặc định dropout 0 + weight decay.
- Python script và test: `.venv/Scripts/python.exe -m pytest tests`.

## Bản đồ code cũ (đọc theo luồng)

| Tầng | File cũ | Dùng thế nào ở bài mới |
|---|---|---|
| Nạp + làm sạch | `scripts/build_bookings.py`, `src/nbp/data/load.py`, `src/nbp/data/clean.py` | khuôn; viết mới cho giao dịch |
| Feature query | `scripts/build_query_codes.py`, `src/nbp/data/query_features.py` | giữ ý `build_vocab` (chỉ train) và `encode` (0 = OOV); viết mới cho anchor và field trip |
| Chia tập + chuỗi | `scripts/build_split.py`, `src/nbp/data/split.py`, `configs/split.yaml` | thay bằng bộ dựng snapshot |
| Model | `src/nbp/models/smlprec_query.py` | khuôn cho lớp model mới |
| Nối model–train–eval | `src/nbp/models/scorers.py` | thêm `build_trip`, `trip_loss`, `trip_scorer` |
| Train | `src/nbp/models/train.py` (`fit`), `src/nbp/models/early_stop.py` | dùng nguyên, `fit` không phụ thuộc model |
| Eval | `src/nbp/eval/runner.py`, `src/nbp/eval/autoregressive.py` | AR/TF không cần nữa; viết eval mới |
| Metric | `src/nbp/eval/user_level.py`, `src/nbp/eval/bootstrap.py`, `src/nbp/hybrid/fusion.py` (`cluster_se`, `one_se_pick`) | dùng nguyên, gom theo `customer_id` |
| Baseline | `src/nbp/baselines/popularity.py`, `itemknn.py` | thêm popularity theo tháng |
| Notebook | `notebooks/smlp4rec/02_query_samedest.ipynb` | khuôn cho notebook mới |

Không dùng cho bài mới: `src/nbp/eval/autoregressive.py`, `src/nbp/hybrid/same_dest.py`, `samedest_*` trong `scorers.py`, `src/nbp/eval/two_step.py`, các script Expedia.

## Thứ tự làm việc

1. **Nạp + làm sạch giao dịch**: ép kiểu ngày, dedup, map destination → 1..18 (0 = padding), map operator → id + bảng operator → destination, kiểm `transaction_date ≤ updated_date`, kiểm mỗi operator thuộc đúng 1 destination.
2. **Bộ dựng snapshot** (`src/nbp/data/snapshot.py`, numpy only): với mỗi t lọc `transaction_date ≤ t`, gom trip, tạo bản ghi trip, lấy L trip gần nhất (đệm trái), tạo anchor, tạo label set, chỉ giữ khách có ≥ 1 trip và label không rỗng.
3. **Config snapshot** (thay `configs/split.yaml`): danh sách mốc t và vai trò train / OOS / OOT / infer (xem `experiment.md`). Script `scripts/build_snapshots.py` ghi `rows.npz` mới và chạy các kiểm tra chống leak.
4. **Mã hoá feature + vocab** (chỉ từ train).
5. **Model** `src/nbp/models/smlprec_trip.py`: lớp con `SMLPREC`, tensor `(B, F, L+1, d)`.
6. **Loss + scorer** trong `scorers.py`: soft-target CE trên 18 cột, che cột padding 0 bằng `-inf`.
7. **Config model + train**: config mới cho model; `configs/train.yaml` đổi `monitor: ndcg@3`.
8. **Eval mới**: chấm thẳng mọi dòng (không AR), monitor = `user_set_metrics(k=3)` trên OOS, `valid_nll` = soft CE; OOT báo cáo theo từng snapshot.
9. **Train** bằng `fit`, hiệu chỉnh `min_delta` như notebook 00 trên monitor mới.
10. **Notebook** mới theo khung notebook 02 + baseline.

## Test

- Bước 2–3 phải có test trên dữ liệu giả nhỏ, tối thiểu:
  - trip vắt qua mốc t: phần đặt trước t vào feature, phần đặt sau không vào label;
  - giao dịch đặt ≤ t, check-out > t: có trong feature (cờ "trip chưa đi"), không có trong label;
  - không feature nào chứa `transaction_date` > t;
  - label = tập destination trong (t, t+3 tháng];
  - đệm trái, trip gần nhất ở vị trí L-1;
  - OOS chia theo khách, không khách nào vừa train vừa OOS.
- Model: test shape, test gradient tới field ≥ 1 ở vị trí lịch sử khác 0 với `n_layers = 2`.
- Giữ test cũ chạy được; không xoá test cũ khi chưa được yêu cầu.

## Quy ước

- Code, docstring, commit message: tiếng Anh. Tài liệu trao đổi: tiếng Việt được.
- Không tự commit hay push khi chưa được yêu cầu.
- Mọi con số báo cáo phải sinh từ code (file JSON trong `results/`), không gõ tay.
- Thay đổi quyết định trong `experiment.md` chỉ khi người dùng chốt.
