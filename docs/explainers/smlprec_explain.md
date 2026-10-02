# Giải thích `src/models/smlprec.py` (SMLP4Rec)

File này là bản sao của SMLP4Rec (Gao et al., TOIS 2024), lấy từ fork `github.com/Quocker20/SMLP4Rec`, thuộc nhóm code legacy (RecBole-era) — không chỉnh sửa logic gốc, chỉ có 2 điểm "ADAPTED" cho pipeline Expedia tuần 3:

1. Độ dài chuỗi (`seq_len`) lấy từ `config['MAX_ITEM_LIST_LENGTH']` thay vì hard-code 50.
2. Cho phép `selected_features` rỗng (Expedia hotel cluster không có thuộc tính item) — khi đó bỏ qua lớp feature embedding, mixer chỉ chạy trên embedding của item.

Các vấn đề giữ nguyên từ bản gốc: dùng chung 1 khối SMLP cho mọi layer (shared weights), không có residual connection quanh khối, và còn vài class không dùng tới trong forward chính (`TokenMixing`, `PatchMerging`, `FeedForward`).

---

## 1. Luồng tổng thể (overall flow)

```
item_seq (batch, seq_len)
    │
    ▼
item_embedding (nn.Embedding)          # tra bảng embedding cho từng item trong chuỗi
    │
    ▼
unsqueeze thêm 1 chiều "field"          # (batch, seq_len, 1, hidden_size)
    │
    ▼ (nếu có feature field)
ghép thêm feature_embedding (sparse/dense) theo chiều field
    │
    ▼
transpose -> (batch, field, seq_len, hidden_size)
    │
    ▼
lặp n_layers lần qua CÙNG MỘT SMLP block (self.layers[0])
    │
    ▼
transpose lại, lấy ra field đầu tiên (mixer_outputs[0])
    │
    ▼
gather_indexes theo item_seq_len - 1     # lấy vector ở vị trí "item cuối cùng thật" của mỗi chuỗi
    │
    ▼
LayerNorm
    │
    ▼
seq_output (batch, hidden_size)          # vector biểu diễn chuỗi hành vi của user
```

`seq_output` này được dùng ở 3 nơi:
- `calculate_loss`: tính loss khi train (BPR hoặc CE).
- `predict`: tính điểm cho 1 item cụ thể (dùng khi đánh giá theo kiểu sampled).
- `full_sort_predict`: tính điểm cho TOÀN BỘ item trong catalog (full ranking — đúng với protocol §8 của dự án, không dùng sampled-negative).

---

## 2. Giải thích từng class / hàm

### `sMLPBlock(nn.Module)`
Khối trộn thông tin theo 3 chiều của tensor 4D `(batch, channel=c, height=h, width=w)`:
- `proj_h`: Linear áp dụng theo chiều "h" (sau khi permute để h nằm ở trục cuối).
- `proj_w`: Linear áp dụng trực tiếp theo chiều "w".
- `x_id`: giữ nguyên input (giống nhánh identity).
- `fuse`: ghép 3 nhánh trên theo chiều channel (`3*c`) rồi Linear về lại `c` chiều.

Đây là ý tưởng theo kiểu "Sparse-MLP" / "S²-MLP": trộn thông tin theo chiều cao, chiều rộng, và giữ identity, sau đó fuse lại bằng 1 Linear.

**Lưu ý:** class này được định nghĩa nhưng KHÔNG được dùng trong luồng forward chính của `SMLPREC` (model chính dùng class `SMLP` bên dưới, khác với `sMLPBlock`). Đây là 1 trong các "unused classes" được nhắc ở docstring đầu file.

### `TokenMixing(nn.Module)`
Kết hợp:
- 1 lớp `depthwise conv2d` (`dwconv`, kernel 3x3, groups = số channel) + residual.
- 1 `sMLPBlock` + residual.
- Có `BatchNorm2d` trước mỗi nhánh.

Cũng là class KHÔNG được gọi trong `SMLPREC.forward` — còn sót lại từ kiến trúc gốc, không dùng trong model thực tế đang chạy.

### `PatchMerging(nn.Module)`
Giảm kích thước không gian bằng 1 `Conv2d` stride 2, đồng thời tăng gấp đôi số channel (`in_shape_0 -> 2*in_shape_0`). Thường dùng trong kiến trúc dạng Swin-Transformer để downsample theo tầng. Cũng KHÔNG được dùng trong forward chính.

### `FeedForward(dim, expansion_factor=2, dropout=0., dense=nn.Linear)`
Hàm factory trả về 1 MLP 2 lớp kiểu Transformer feed-forward: `Linear -> GELU -> Dropout -> Linear -> Dropout`, mở rộng chiều lên `dim * expansion_factor` rồi co lại. Cũng không được gọi ở đâu trong file — hàm tiện ích còn sót lại.

### `SMLP(nn.Module)` — khối mixer THỰC SỰ được dùng
Nhận `in_shape = [field, seq_len, hidden_size]` (tương ứng 3 chiều: field/channel, chiều dài chuỗi, chiều ẩn). Có 3 nhánh Linear riêng biệt áp dụng lần lượt theo từng chiều:
- `self.c`: trộn theo chiều "field" (`in_shape[0]`).
- `self.h`: trộn theo chiều "seq_len" (`in_shape[1]`).
- `self.w`: trộn theo chiều "hidden_size" (`in_shape[2]`).

Mỗi nhánh là 1 MLP 2 lớp (`Linear -> GELU -> Dropout -> Linear -> Dropout`, không bias).

`forward`:
1. `LayerNorm` input.
2. Transpose để đưa đúng chiều cần trộn về trục cuối, chạy Linear tương ứng, rồi transpose lại — làm với cả 3 nhánh `c`, `h`, `w`.
3. Cộng 3 kết quả lại (`x0 + x1 + x2`) — đây chính là bước "mixing" kiểu MLP-Mixer áp dụng đồng thời trên 3 chiều.

Đây là khối chính, được tái sử dụng (shared weights) qua toàn bộ `n_layers` vòng lặp trong `SMLPREC.forward`.

### `SMLPREC(SequentialRecommender)` — model chính

#### `__init__(self, config, dataset, seq_len=None)`
- Đọc tham số từ `config`: `n_layers`, `n_heads` (không thấy dùng trực tiếp trong forward — leftover), `hidden_size`, `hidden_dropout_prob`, `hidden_act` (không dùng), `layer_norm_eps`, `selected_features`, `pooling_mode`, `device`, `initializer_range`, `loss_type`.
- `seq_len = seq_len or config['MAX_ITEM_LIST_LENGTH']` — điểm ADAPTED số 1.
- `self.selected_features = config['selected_features'] or []` — điểm ADAPTED số 2: cho phép rỗng.
- `self.num_feature_field = len(self.selected_features)`.
- `self.item_embedding`: bảng embedding item, `padding_idx=0` (index 0 dùng để pad chuỗi ngắn).
- `self.feature_embed_layer`: chỉ khởi tạo (`FeatureSeqEmbLayer` của RecBole) nếu có ít nhất 1 feature field; nếu không thì giữ `None` — khi Expedia không có thuộc tính item cho hotel cluster, layer này bị bỏ qua hoàn toàn.
- `self.layers`: `ModuleList` nhưng chỉ chứa ĐÚNG 1 phần tử — 1 khối `SMLP` duy nhất với shape `[num_feature_field+1, seq_len, hidden_size]`. Đây là nguồn gốc của "known issue: 1 khối SMLP dùng chung cho mọi layer" — vòng lặp `n_layers` trong forward gọi lại `self.layers[0]` nhiều lần thay vì có `n_layers` khối riêng biệt.
- `self.LayerNorm`: chuẩn hoá output cuối.
- Chọn loss: `BPRLoss` (RecBole) nếu `loss_type == 'BPR'`, hoặc `nn.CrossEntropyLoss` nếu `'CE'`; báo lỗi `NotImplementedError` nếu giá trị khác.
- `self.apply(self._init_weights)`: khởi tạo trọng số.

#### `_init_weights(self, module)`
Khởi tạo chuẩn kiểu BERT: `Linear`/`Embedding` dùng normal(0, `initializer_range`); `LayerNorm` thì bias = 0, weight = 1; bias của `Linear` (nếu có) = 0.

#### `forward(self, item_seq, item_seq_len)`
Mô tả chi tiết đã nêu ở phần "Luồng tổng thể" bên trên. Điểm quan trọng:
- Nếu `feature_embed_layer` tồn tại, lấy `sparse_embedding`/`dense_embedding` cho field `'item'`, ghép cùng chiều với `item_emb` (dim=2, tức chiều "field").
- Nếu không có feature layer (trường hợp Expedia), `item_emb` chỉ có 1 field duy nhất (chính item embedding).
- Vòng `for x in range(self.n_layers): mixer_outputs = self.layers[0](mixer_outputs)` — áp dụng lặp lại cùng 1 khối SMLP `n_layers` lần (shared-weight recurrence, không phải stack các layer độc lập).
- `gather_indexes(mixer_outputs[0], item_seq_len - 1)`: hàm có sẵn của `SequentialRecommender` (RecBole), lấy ra vector tại vị trí tương ứng với item cuối cùng thật trong mỗi chuỗi (bỏ qua phần padding).

#### `calculate_loss(self, interaction)`
- Lấy `item_seq`, `item_seq_len`, chạy `forward` ra `seq_output`.
- Lấy `pos_items` (item đúng/positive).
- Nếu BPR: lấy thêm `neg_items` (item âm), tính điểm dot-product giữa `seq_output` và embedding của positive/negative, đưa vào `BPRLoss`.
- Nếu CE: tính logits bằng `seq_output @ toàn bộ item_embedding.T` (dạng full-softmax trên tất cả item), rồi `CrossEntropyLoss` so với `pos_items`. Cách này phù hợp hướng "full ranking over whole catalog" theo protocol của dự án (§8 CLAUDE.md).

#### `predict(self, interaction)`
Dùng khi cần điểm số cho 1 item cụ thể mỗi sample (không phải full-sort): dot-product giữa `seq_output` và embedding của `test_item`.

#### `full_sort_predict(self, interaction)`
Tính điểm cho TOÀN BỘ item trong catalog cùng lúc: `seq_output @ item_embedding.weight.T` → ma trận điểm (batch, n_items). Đây là hàm được dùng để đánh giá theo đúng protocol "full ranking" (Recall@K, NDCG@K, MRR trên toàn bộ catalog, không sample negative) mà dự án yêu cầu.

---

## 3. Tóm tắt các điểm cần lưu ý khi port sang `src/nbp/`

- Chỉ `SMLP` + `SMLPREC` là phần thực sự chạy; `sMLPBlock`, `TokenMixing`, `PatchMerging`, `FeedForward` là code chết (dead code) trong luồng hiện tại — không cần port trừ khi muốn mở rộng kiến trúc.
- "1 khối SMLP dùng chung cho mọi layer" là hành vi cố ý giữ nguyên từ bản gốc (không phải bug phát sinh do adapt) — khi viết lại bằng plain PyTorch (kế hoạch Day 5, §6 CLAUDE.md), cần quyết định rõ: giữ nguyên shared-weight recurrence (đúng bản gốc) hay đổi thành stack các layer riêng (thay đổi kiến trúc, cần ghi chú là research adaptation).
- Việc cho phép `selected_features = []` (bỏ feature layer) là đúng hướng cho Expedia (item = hotel cluster, không có text/feature), nhưng cần nhớ: paper gốc SMLP4Rec có dùng thêm context/feature field — phiên bản chạy trên Expedia sẽ là baseline/core yếu hơn bản đầy đủ của paper; phần context (party size, season, package...) sẽ được đưa vào qua late-fusion/hybrid riêng (đã làm ở nhánh `exp/late-fusion-destination-prior`), không phải qua `feature_embed_layer` này.
