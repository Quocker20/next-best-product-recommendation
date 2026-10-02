"""SMLP4Rec (Gao et al., TOIS 2024) copied from github.com/Quocker20/SMLP4Rec (smlprec.py).

Research adaptation, not the authors' release. Minimal changes for the week-3 Expedia
pipeline run, each marked "# ADAPTED":
- sequence length comes from config['MAX_ITEM_LIST_LENGTH'] instead of a hard-coded 50;
- an empty `selected_features` list is allowed (Expedia hotel clusters have no item
  attributes): the item-feature layer is skipped and the mixer runs on item embeddings only.

File này là BẢN SAO từ smlprec.py, thêm comment tiếng Việt giải thích từng dòng / từng khối
code, VÀ ĐÃ XÓA toàn bộ dead code (class/hàm/biến/import không được dùng trong luồng chạy
thực tế) so với bản gốc. Logic của phần còn lại (SMLP + SMLPREC) giữ nguyên 100%.

Đã xóa (dead code, không ảnh hưởng kết quả vì không được gọi/dùng ở đâu):
- class sMLPBlock, TokenMixing, PatchMerging, hàm FeedForward (không được gọi trong forward chính)
- import TransformerEncoder, VanillaAttention, partial (chỉ phục vụ các class/biến đã xóa)
- self.n_heads, self.hidden_act (đọc từ config nhưng không dùng ở đâu)
- chan_first, chan_last, self.layerSize (gán giá trị nhưng không dùng ở đâu)

Known issue GIỮ NGUYÊN (không phải dead code, là hành vi thật của model): 1 khối SMLP
được tái sử dụng (shared weights) cho tất cả n_layers vòng lặp, không có residual
connection quanh khối đó.
"""
import torch  # thư viện PyTorch chính
from torch import nn  # module chứa các lớp mạng nơ-ron (Linear, Embedding, ...)
from recbole.model.abstract_recommender import SequentialRecommender  # lớp cơ sở của RecBole cho model gợi ý theo chuỗi (sequential recommender)
from recbole.model.layers import FeatureSeqEmbLayer  # lớp embedding feature của item, dùng khi có selected_features
from recbole.model.loss import BPRLoss  # hàm loss BPR (Bayesian Personalized Ranking) có sẵn của RecBole


class SMLP(nn.Module):
    # Khối mixer CHÍNH, được dùng trong SMLPREC.forward bên dưới.
    # in_shape = [field, seq_len, hidden_size]: 3 chiều cần trộn thông tin.
    def __init__(self, in_shape, expansion_factor = 2, dropout=0.):
        super().__init__()
        # Nhánh "c": trộn theo chiều field (in_shape[0]) - MLP 2 lớp, không bias
        self.c = nn.Sequential(nn.Linear(in_shape[0], expansion_factor * in_shape[0], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[0], in_shape[0], bias=False),
                               nn.Dropout(dropout),)
        # Nhánh "h": trộn theo chiều seq_len (in_shape[1]) - MLP 2 lớp, không bias
        self.h = nn.Sequential(nn.Linear(in_shape[1], expansion_factor * in_shape[1], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[1], in_shape[1], bias=False),
                               nn.Dropout(dropout),)
        # Nhánh "w": trộn theo chiều hidden_size (in_shape[2]) - MLP 2 lớp, không bias
        self.w = nn.Sequential(nn.Linear(in_shape[2], expansion_factor * in_shape[2], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[2], in_shape[2], bias=False),
                               nn.Dropout(dropout),)
        self.norm2 = nn.LayerNorm(in_shape[2])  # LayerNorm áp dụng trước khi vào 3 nhánh trộn
    def forward(self, x):
        xn = self.norm2(x)  # chuẩn hóa input trước
        # Nhánh c: hoán vị để đưa chiều field (trục 1) về trục cuối (trục 3), áp self.c, rồi hoán vị lại
        x0 = (self.c(xn.transpose(1,3).contiguous())).transpose(3,1).contiguous()
        # Nhánh h: hoán vị để đưa chiều seq_len (trục 2) về trục cuối (trục 3), áp self.h, rồi hoán vị lại
        x1 = (self.h(xn.transpose(2,3).contiguous())).transpose(3,2).contiguous()
        # Nhánh w: hidden_size đã ở trục cuối sẵn, áp self.w trực tiếp, không cần hoán vị
        x2 = self.w(xn)
        y = x0 + x1 + x2  # cộng 3 nhánh lại - đây là bước "mixing" trên cả 3 chiều cùng lúc
        return y

class SMLPREC(SequentialRecommender):
    # Model chính: kế thừa SequentialRecommender của RecBole (có sẵn các hàm/thuộc tính
    # như self.n_items, self.ITEM_SEQ, self.gather_indexes, ...).
    def __init__(self, config, dataset, seq_len = None):
        super(SMLPREC, self).__init__(config, dataset)
        seq_len = seq_len or config['MAX_ITEM_LIST_LENGTH']  # ADAPTED  # độ dài chuỗi: ưu tiên tham số truyền vào, nếu không có thì lấy từ config (thay vì hard-code 50 như bản gốc)

        self.n_layers = config['n_layers']  # số lần lặp lại khối SMLP trong forward
        self.hidden_size = config['hidden_size']  # chiều ẩn (embedding dimension)
        self.hidden_dropout_prob = config['hidden_dropout_prob']  # tỉ lệ dropout dùng trong SMLP
        self.layer_norm_eps = config['layer_norm_eps']  # epsilon cho LayerNorm cuối cùng
        self.selected_features = config['selected_features'] or []  # ADAPTED  # danh sách feature field của item; cho phép rỗng (Expedia hotel cluster không có thuộc tính item)
        self.pooling_mode = config['pooling_mode']  # cách pooling cho feature embedding (vd mean/sum) khi có nhiều giá trị
        self.device = config['device']  # thiết bị chạy model (cpu/cuda)
        expansion_factor = 3  # hệ số mở rộng dùng khi tạo khối SMLP bên dưới
        self.num_feature_field = len(self.selected_features)  # ADAPTED  # số lượng feature field của item (0 nếu Expedia)

        self.initializer_range = config['initializer_range']  # độ lệch chuẩn dùng khi khởi tạo trọng số
        self.loss_type = config['loss_type']  # loại loss: 'BPR' hoặc 'CE'

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)  # bảng embedding cho item; index 0 dành cho padding (chuỗi ngắn hơn max length)

        self.feature_embed_layer = None  # ADAPTED: no item features -> no feature layer  # mặc định không có lớp feature; chỉ khởi tạo nếu có feature field
        if self.num_feature_field > 0:
            self.feature_embed_layer = FeatureSeqEmbLayer(
                dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
            )  # lớp embedding cho các feature field của item (sparse + dense), có sẵn trong RecBole

        self.layers = nn.ModuleList([])  # danh sách các khối mixer (nhưng thực tế chỉ thêm DUY NHẤT 1 phần tử bên dưới)
        dim_new = self.num_feature_field+1  # số "field" = số feature field + 1 (field của chính item embedding)
        self.layers.append(SMLP([dim_new, seq_len, self.hidden_size], expansion_factor, self.hidden_dropout_prob))  # tạo DUY NHẤT 1 khối SMLP, sẽ được tái sử dụng (dùng chung trọng số) qua n_layers vòng lặp trong forward
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)  # LayerNorm áp dụng lên output cuối cùng của model

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()  # loss BPR: dùng khi train theo cặp (positive, negative)
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()  # loss CE: dùng khi train kiểu full-softmax trên toàn bộ item
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")  # chỉ hỗ trợ 2 loại loss này

        self.apply(self._init_weights)  # áp dụng hàm khởi tạo trọng số cho toàn bộ module con

    def _init_weights(self, module):
        """ Initialize the weights """
        # Khởi tạo trọng số kiểu BERT: Linear/Embedding dùng phân phối chuẩn (normal),
        # LayerNorm thì bias=0 và weight=1, bias của Linear (nếu có) = 0.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)  # khởi tạo weight theo N(0, initializer_range^2)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # bias của LayerNorm = 0
            module.weight.data.fill_(1.0)  # weight (scale) của LayerNorm = 1
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # bias của Linear = 0 (nếu Linear có bias)

    def forward(self, item_seq, item_seq_len):
        # item_seq: (batch, seq_len) - chuỗi các item_id đã tương tác; item_seq_len: (batch,) - độ dài thực của mỗi chuỗi (chưa tính padding)
        item_emb = self.item_embedding(item_seq)  # tra embedding cho từng item trong chuỗi -> (batch, seq_len, hidden_size)
        item_emb = torch.unsqueeze(item_emb,2)  # thêm 1 chiều "field" ở vị trí 2 -> (batch, seq_len, 1, hidden_size)
        if self.feature_embed_layer is not None:  # ADAPTED  # chỉ chạy nhánh này nếu CÓ feature field (Expedia thì không có nên bỏ qua toàn bộ khối if này)
            sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)  # lấy embedding sparse/dense cho các feature field của item trong chuỗi
            sparse_embedding = sparse_embedding['item']  # chỉ lấy phần feature của "item" (không phải của "user")
            dense_embedding = dense_embedding['item']

            if sparse_embedding is not None:
                feature_embeddings = sparse_embedding  # nếu có sparse feature thì dùng trước
            if dense_embedding is not None:
                if sparse_embedding is not None:
                    feature_embeddings = torch.cat((sparse_embedding,dense_embedding),2)  # nếu có cả 2 loại thì ghép lại theo chiều field (dim=2)
                else:
                    feature_embeddings = dense_embedding  # nếu chỉ có dense thì dùng dense

            item_emb = torch.cat((item_emb,feature_embeddings),2)  # ghép thêm các feature field vào sau field của item embedding (dim=2)

        mixer_outputs = item_emb.transpose(1,2).contiguous()  # hoán đổi trục seq_len và trục field -> (batch, field, seq_len, hidden_size), dùng làm input cho SMLP

        for _ in range(self.n_layers):
            mixer_outputs = self.layers[0](mixer_outputs)  # LƯU Ý: luôn gọi lại self.layers[0] (CÙNG 1 khối SMLP) - "known issue": shared weights qua mọi layer, không phải n_layers khối độc lập

        mixer_outputs = mixer_outputs.transpose(1, 0).contiguous()  # hoán đổi trục batch và trục field -> (field, batch, seq_len, hidden_size)
        output = self.gather_indexes(mixer_outputs[0], item_seq_len - 1)  # lấy field đầu tiên (field của item), rồi lấy vector tại vị trí "item cuối cùng thật" (item_seq_len-1) của mỗi chuỗi - hàm có sẵn của SequentialRecommender
        output = self.LayerNorm(output)  # chuẩn hóa vector đại diện cuối cùng
        return output  # seq_output: (batch, hidden_size) - vector biểu diễn hành vi chuỗi của user

    def calculate_loss(self, interaction):
        # Hàm tính loss khi train, interaction là 1 batch dữ liệu (dict-like của RecBole)
        item_seq = interaction[self.ITEM_SEQ]  # chuỗi item đầu vào
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # độ dài thật của từng chuỗi
        seq_output = self.forward(item_seq, item_seq_len)  # chạy forward để lấy vector đại diện chuỗi
        pos_items = interaction[self.POS_ITEM_ID]  # item đúng (positive/ground-truth) cần dự đoán tiếp theo
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]  # item âm (negative) được sample sẵn từ RecBole
            pos_items_emb = self.item_embedding(pos_items)  # embedding của item positive
            neg_items_emb = self.item_embedding(neg_items)  # embedding của item negative
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # điểm positive = dot product giữa seq_output và embedding item đúng
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # điểm negative = dot product giữa seq_output và embedding item sai
            loss = self.loss_fct(pos_score, neg_score)  # BPR loss: khuyến khích pos_score > neg_score
            return loss
        else:
            test_item_emb = self.item_embedding.weight  # toàn bộ bảng embedding item -> dùng làm "weight" cho phép dự đoán full-softmax

            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # tính điểm cho TẤT CẢ item cùng lúc: (batch, hidden_size) x (hidden_size, n_items) -> (batch, n_items)
            loss = self.loss_fct(logits, pos_items)  # CrossEntropyLoss giữa logits và item đúng (full-softmax train)
            return loss

    def predict(self, interaction):
        # Dự đoán điểm cho 1 item CỤ THỂ mỗi sample (không phải full-sort) - dùng khi eval kiểu sampled
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]  # item cần tính điểm (được chỉ định sẵn trong interaction)
        seq_output = self.forward(item_seq, item_seq_len)  # vector đại diện chuỗi
        test_item_emb = self.item_embedding(test_item)  # embedding của item cần tính điểm
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # điểm = dot product giữa seq_output và embedding item đó
        return scores

    def full_sort_predict(self, interaction):
        # Dự đoán điểm cho TOÀN BỘ item trong catalog cùng lúc - dùng cho full ranking
        # (đúng protocol của dự án: Recall@K/NDCG@K/MRR trên toàn bộ catalog, không sample negative)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)  # vector đại diện chuỗi
        test_items_emb = self.item_embedding.weight  # toàn bộ bảng embedding item
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # (batch, hidden_size) x (hidden_size, n_items) -> (batch, n_items): điểm cho từng item
        return scores
