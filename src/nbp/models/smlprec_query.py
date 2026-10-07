"""SMLP4Rec with a query token: the current search enters the model as one extra position.

History = the user's previous clusters (positions 0..S-1, RecBole right-padding, exactly as the
history-only model). The query token sits at the fixed last position S: a learned [MASK] vector
plus the sum of one embedding per query field (destination, dates, party, package, channel...).
The hidden state at that position scores the item catalogue. The legacy mixer block and item
embedding come from `src/models/smlprec.py`, which stays untouched.

Research adaptation of SMLP4Rec (Gao et al., TOIS 2024); the paper's per-position context is not
included here.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.smlprec import SMLPREC

ROW_ID = "row_id"


class SMLPRECQuery(SMLPREC):
    """History-only SMLPREC + query token.

    Args:
        config: RecBole config (needs MAX_ITEM_LIST_LENGTH = S).
        dataset: RecBole dataset (item count only).
        query_idx: (N, F) integer tensor of embedding indices, row i = booking with row_id i.
        field_sizes: F embedding table sizes (vocab size + 1 for the out-of-vocab index 0).
    """

    def __init__(self, config, dataset, query_idx: torch.Tensor, field_sizes: list[int]):
        super().__init__(config, dataset, seq_len=config["MAX_ITEM_LIST_LENGTH"] + 1)
        self.query_emb = nn.ModuleList([nn.Embedding(n, self.hidden_size) for n in field_sizes])
        self.query_mask = nn.Parameter(torch.zeros(self.hidden_size))
        self.query_emb.apply(self._init_weights)
        nn.init.normal_(self.query_mask, std=self.initializer_range)
        self.register_buffer("query_idx", query_idx.to(torch.int32), persistent=False)

    def query_vector(self, q_idx: torch.Tensor) -> torch.Tensor:
        """(B, F) embedding indices -> (B, H) query vector."""
        q = self.query_mask.expand(q_idx.shape[0], -1)
        for j, emb in enumerate(self.query_emb):
            q = q + emb(q_idx[:, j].long())
        return q

    def forward(self, item_seq, q_idx):  # noqa: D102 (history (B, S) ids, query (B, F) indices)
        h = self.item_embedding(item_seq)  # (B, S, H)
        x = torch.cat([h, self.query_vector(q_idx).unsqueeze(1)], dim=1)  # (B, S+1, H)
        x = x.unsqueeze(2).transpose(1, 2).contiguous()  # (B, 1, S+1, H)
        for _ in range(self.n_layers):
            x = self.layers[0](x)
        out = x.transpose(1, 0).contiguous()[0][:, -1]  # query position
        return self.LayerNorm(out)

    def _rows(self, interaction) -> torch.Tensor:
        return self.query_idx[interaction[ROW_ID].long()].long()

    def calculate_loss(self, interaction):
        out = self.forward(interaction[self.ITEM_SEQ], self._rows(interaction))
        logits = out @ self.item_embedding.weight.T
        return self.loss_fct(logits, interaction[self.POS_ITEM_ID])

    def scores(self, item_seq, q_idx) -> torch.Tensor:
        """(B, n_items) scores; column 0 is the padding id (callers set it to -inf)."""
        return self.forward(item_seq, q_idx) @ self.item_embedding.weight.T

    def full_sort_predict(self, interaction):
        return self.scores(interaction[self.ITEM_SEQ], self._rows(interaction))
