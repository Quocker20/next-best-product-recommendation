# SMLP4Rec explained (`src/nbp/models/smlprec.py`)

`smlprec.py` is the SMLP4Rec model class (Gao et al., TOIS 2024) from the authors' repository (`github.com/Applied-Machine-Learning-Lab/SMLP4Rec`), built on RecBole. The upstream repository has no license, so the file is not stored in git: `scripts/fetch_smlp4rec.py` downloads it and writes the generated, gitignored `src/nbp/models/smlprec.py`. The original logic is untouched except for two changes marked `# ADAPTED`:

1. The sequence length (`seq_len`) is read from `config['MAX_ITEM_LIST_LENGTH']` instead of being hard-coded to 50.
2. `selected_features` may be empty (a hotel cluster has no item attributes). The feature-embedding layer is then skipped and the mixer runs on item embeddings only.

Behaviours inherited from the original and kept on purpose: one SMLP block shared by every layer (shared weights), no residual connection around the block, and a few classes that the main forward pass never uses (`TokenMixing`, `PatchMerging`, `FeedForward`, `sMLPBlock`).

The project's own extension, the query embedding, lives in `src/nbp/models/smlprec_query.py` (original code).

---

## 1. Overall flow

```
item_seq (batch, seq_len)
    |
    v
item_embedding (nn.Embedding)          # one embedding per item of the sequence
    |
    v
unsqueeze a "field" dimension           # (batch, seq_len, 1, hidden_size)
    |
    v (if there are feature fields)
concatenate feature_embedding (sparse / dense) along the field dimension
    |
    v
transpose -> (batch, field, seq_len, hidden_size)
    |
    v
n_layers passes through the SAME SMLP block (self.layers[0])
    |
    v
transpose back, take the first field (mixer_outputs[0])
    |
    v
gather_indexes at item_seq_len - 1      # the vector of the last real item of each sequence
    |
    v
LayerNorm
    |
    v
seq_output (batch, hidden_size)         # representation of the user's behaviour sequence
```

`seq_output` is used in three places:
- `calculate_loss`: the training loss (BPR or cross-entropy);
- `predict`: the score of one given item;
- `full_sort_predict`: the score of every item of the catalogue (full ranking, no sampled negatives).

---

## 2. Classes and functions

### `sMLPBlock(nn.Module)`
Mixes information along the three axes of a 4D tensor `(batch, channel=c, height=h, width=w)`:
- `proj_h`: a Linear applied along "h" (after a permute that moves h to the last axis);
- `proj_w`: a Linear applied directly along "w";
- `x_id`: the input kept as is (identity branch);
- `fuse`: concatenates the three branches along the channel axis (`3*c`) and projects back to `c` with a Linear.

This follows the Sparse-MLP / S2-MLP idea. **Not used** in the main forward pass of `SMLPREC`, which uses the `SMLP` class below.

### `TokenMixing(nn.Module)`
A depthwise `conv2d` (`dwconv`, 3x3 kernel, groups = channels) with a residual, plus an `sMLPBlock` with a residual, each preceded by `BatchNorm2d`. **Not called** by `SMLPREC.forward`; left over from the original architecture.

### `PatchMerging(nn.Module)`
Halves the spatial size with a stride-2 `Conv2d` and doubles the channels (`in_shape_0 -> 2*in_shape_0`), as in Swin-style hierarchical downsampling. **Not used.**

### `FeedForward(dim, expansion_factor=2, dropout=0., dense=nn.Linear)`
Factory for a Transformer-style two-layer MLP: `Linear -> GELU -> Dropout -> Linear -> Dropout`, expanding to `dim * expansion_factor` and back. **Not called** anywhere in the file.

### `SMLP(nn.Module)`: the mixer block that is actually used
Takes `in_shape = [field, seq_len, hidden_size]`. Three separate Linear branches act on one axis each:
- `self.c`: mixes along the field axis (`in_shape[0]`);
- `self.h`: mixes along the sequence axis (`in_shape[1]`);
- `self.w`: mixes along the hidden axis (`in_shape[2]`).

Each branch is a two-layer MLP (`Linear -> GELU -> Dropout -> Linear -> Dropout`, no bias).

`forward`:
1. `LayerNorm` of the input.
2. For each of the three branches: transpose the axis to mix to the last position, apply the Linear, transpose back.
3. Sum the three results (`x0 + x1 + x2`): an MLP-Mixer style mixing over the three axes at once.

The same block is reused (shared weights) for all `n_layers` iterations in `SMLPREC.forward`.

### `SMLPREC(SequentialRecommender)`: the model

#### `__init__(self, config, dataset, seq_len=None)`
- Reads `n_layers`, `n_heads` (not used in forward), `hidden_size`, `hidden_dropout_prob`, `hidden_act` (not used), `layer_norm_eps`, `selected_features`, `pooling_mode`, `device`, `initializer_range`, `loss_type` from `config`.
- `seq_len = seq_len or config['MAX_ITEM_LIST_LENGTH']` (ADAPTED change 1).
- `self.selected_features = config['selected_features'] or []` (ADAPTED change 2).
- `self.num_feature_field = len(self.selected_features)`.
- `self.item_embedding`: item embedding table with `padding_idx=0` (index 0 pads short sequences).
- `self.feature_embed_layer`: RecBole's `FeatureSeqEmbLayer`, created only when there is at least one feature field; otherwise `None`.
- `self.layers`: a `ModuleList` holding exactly ONE `SMLP` block of shape `[num_feature_field+1, seq_len, hidden_size]`. This is the source of the shared-weights behaviour: the `n_layers` loop in `forward` calls `self.layers[0]` repeatedly instead of stacking `n_layers` independent blocks.
- `self.LayerNorm`: normalises the final output.
- Loss: RecBole's `BPRLoss` if `loss_type == 'BPR'`, `nn.CrossEntropyLoss` if `'CE'`, `NotImplementedError` otherwise.
- `self.apply(self._init_weights)`: weight initialisation.

#### `_init_weights(self, module)`
BERT-style initialisation: `Linear` and `Embedding` weights ~ normal(0, `initializer_range`); `LayerNorm` bias 0 and weight 1; `Linear` bias 0.

#### `forward(self, item_seq, item_seq_len)`
Described in the flow above. Key points:
- If `feature_embed_layer` exists, the `sparse_embedding` / `dense_embedding` of the `'item'` field are concatenated with `item_emb` along the field dimension (dim=2).
- Without a feature layer (the Expedia case), `item_emb` has a single field: the item embedding itself.
- `for x in range(self.n_layers): mixer_outputs = self.layers[0](mixer_outputs)` applies the same SMLP block `n_layers` times (shared-weight recurrence, not a stack of independent layers).
- `gather_indexes(mixer_outputs[0], item_seq_len - 1)` is a `SequentialRecommender` helper that takes the vector of the last real item of each sequence and skips the padding.

#### `calculate_loss(self, interaction)`
- Runs `forward` to get `seq_output` and reads `pos_items`.
- BPR: also reads `neg_items`, takes the dot products of `seq_output` with the positive and negative item embeddings and feeds them to `BPRLoss`.
- CE: logits = `seq_output @ item_embedding.T` over all items (full softmax), then `CrossEntropyLoss` against `pos_items`. This matches the full-ranking evaluation used here.

#### `predict(self, interaction)`
Score of one item per sample: dot product of `seq_output` and the embedding of `test_item`.

#### `full_sort_predict(self, interaction)`
Scores every item of the catalogue at once: `seq_output @ item_embedding.weight.T`, a `(batch, n_items)` matrix. This is the function behind the full-ranking Recall@K and NDCG@K (no sampled negatives).

---

## 3. Points to keep in mind

- Only `SMLP` and `SMLPREC` run; `sMLPBlock`, `TokenMixing`, `PatchMerging` and `FeedForward` are dead code in the current forward pass.
- "One SMLP block shared by every layer" is kept from the original code on purpose (it is not a bug introduced by the adaptation). Any rewrite has to choose between keeping the shared-weight recurrence (faithful to the original) and stacking independent layers (an architectural change that must be documented as such).
- Allowing `selected_features = []` fits Expedia (the item is a hotel cluster with no attributes), but the original paper also uses context and feature fields, so this history-only version ("plain" in the results) is a weaker core than the full paper model. Search context (party size, season, package, ...) enters through the project's own query embedding (`src/nbp/models/smlprec_query.py`) and the sameDest fusion (`src/nbp/hybrid/`), not through `feature_embed_layer`.
