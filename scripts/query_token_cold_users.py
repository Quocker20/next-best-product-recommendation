"""Score the SMLP4Rec + query-token checkpoint (notebook 01c) on L = 0 rows: first bookings, empty history.

The 01c model never saw an empty history in training (targets are bookings 2..n of each user), so this
is the missing cold-start measurement. L = 0 rows are built exactly as in notebook 06: the first booking
of every user whose first booking falls in the valid / test window. The model gets an all-padding
history plus the query token of that booking.

Compared on the same rows: global popularity, destination prior only (notebook 03 tables, same cut),
query token alone, query token + prior (w chosen in 01c), query token with a shuffled query.

Usage (RecBole venv): C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/query_token_cold_users.py
Output: results/week4_rebuild/smlprec_query_cold_L0.json
"""

import functools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch  # before pandas (Windows c10.dll issue)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
torch.load = functools.partial(torch.load, weights_only=False)

import pandas as pd  # noqa: E402
from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

from nbp.data.query_features import QUERY_FIELDS, build_vocab, encode  # noqa: E402
from nbp.eval.bootstrap import paired_bootstrap  # noqa: E402
from nbp.eval.metrics import summarize, target_rank  # noqa: E402
from nbp.models.smlprec_query import ROW_ID, SMLPRECQuery  # noqa: E402
from src.models.smlprec import SMLPREC  # noqa: E402

logging.disable(logging.INFO)

CONFIG = ROOT / "configs" / "smlprec_expedia.yaml"
WORK = ROOT / "data" / "interim" / "recbole"
CODES = ROOT / "data" / "interim" / "expedia_query_codes.npz"
RUN_DIR = max((ROOT / "experiments").glob("*_expedia_smlp4rec_query-1c"))
QRUN = json.loads((ROOT / "results" / "week4_rebuild" / "smlprec_query_run.json").read_text(encoding="utf-8"))
SU = json.loads((ROOT / "results" / "week3_implementation" / "smlprec_expedia_seen_unseen_users.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "week4_rebuild" / "smlprec_query_cold_L0.json"
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
KS = (5, 10, 20)
N_CLUSTERS, M_SMOOTH, EPS, TIME_MARGIN = 100, 5, 1e-6, 128
W_FUSION = QRUN["prior_fusion_check"]["chosen_w_on_valid_recall@5"]

config = Config(
    model=SMLPREC,
    dataset="expedia_rowid",
    config_file_list=[str(CONFIG)],
    config_dict={
        "data_path": str(WORK),
        "checkpoint_dir": str(RUN_DIR),
        "load_col": {"inter": ["user_id", "item_id", "timestamp", "row_id"]},
    },
)
init_seed(config["seed"], config["reproducibility"])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
TIME, UID = config["TIME_FIELD"], config["USER_ID_FIELD"]
tok2id = dataset.field2token_id[dataset.iid_field]
uid_of_token = dataset.field2token_id[dataset.uid_field]
cluster_ids = np.array([tok2id[str(k)] for k in range(N_CLUSTERS)])  # model column of each raw cluster
n_items = dataset.item_num

z = np.load(CODES)
codes, fields = z["codes"], list(z["fields"])
assert fields == list(QUERY_FIELDS)
train_rows = train_data.dataset.inter_feat[ROW_ID].long().numpy()
vocab = build_vocab(codes, train_rows)
Q = encode(codes, vocab)
field_sizes = [len(v) + 1 for v in vocab]
ck = torch.load(RUN_DIR / "best.pth")
assert ck["field_sizes"] == field_sizes, "vocab differs from the checkpoint"
Qt = torch.from_numpy(Q)
model = SMLPRECQuery(config, train_data.dataset, Qt, field_sizes)
model.load_state_dict(ck["state_dict"])
model.eval()

# ---- L = 0 rows, as notebook 06
t_train_max = float(train_data.dataset.inter_feat[TIME].max())
t_valid_max = float(valid_data.dataset.inter_feat[TIME].max())
cut = t_train_max - TIME_MARGIN
train_users = np.unique(train_data.dataset.inter_feat[UID].numpy())
inter = pd.read_csv(WORK / "expedia_rowid" / "expedia_rowid.inter", sep="\t").sort_values(
    ["user_id:token", "timestamp:float"], kind="stable"
)
first = inter.drop_duplicates("user_id:token", keep="first")
windows = {
    "valid": (first["timestamp:float"] >= t_train_max) & (first["timestamp:float"] < t_valid_max),
    "test": first["timestamp:float"] >= t_valid_max,
}
DEST = fields.index("dest")
cold = {}
for sp, mask in windows.items():
    f = first[mask]
    row = f["row_id:float"].astype(int).values
    cold[sp] = {
        "row": row,
        "y_model": np.array([tok2id[str(int(k))] for k in f["item_id:token"].astype(int)]),
        "dest": codes[row, DEST].astype(np.int64),
        "user": np.array([uid_of_token[str(u)] for u in f["user_id:token"]]),
    }
    assert not np.isin(cold[sp]["user"], train_users).any()
EXPECT = SU["presence"]["test"]["unseen_events_L=0"]
assert len(cold["test"]["row"]) == EXPECT, (len(cold["test"]["row"]), EXPECT)
print({sp: len(c["row"]) for sp, c in cold.items()}, "L = 0 rows; matches notebook 06")


# ---- prior tables (notebook 03 / 01c)
def build_prior_tables(cut_seconds: float) -> dict:
    cols = ["date_time", "is_booking", "srch_destination_id", "hotel_market", "hotel_cluster"]
    dtypes = {c: "int32" for c in cols if c != "date_time"}
    parts = []
    for ch in pd.read_csv(RAW, usecols=cols, chunksize=2_000_000, dtype=dtypes):
        ch = ch[ch.is_booking == 1]
        ts = (pd.to_datetime(ch["date_time"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        parts.append(ch.loc[ts.values < cut_seconds, ["srch_destination_id", "hotel_market", "hotel_cluster"]])
    b = pd.concat(parts, ignore_index=True)

    def table(key):
        t = b.groupby([key, "hotel_cluster"]).size().unstack(fill_value=0)
        return t.reindex(columns=range(N_CLUSTERS), fill_value=0)

    dest, mkt = table("srch_destination_id"), table("hotel_market")
    dm = b.groupby(["srch_destination_id", "hotel_market"]).size().reset_index(name="n")
    modal = dm.sort_values("n", ascending=False).drop_duplicates("srch_destination_id")
    glob = np.bincount(b["hotel_cluster"].values, minlength=N_CLUSTERS).astype(np.float64)
    return {
        "n_bookings": len(b),
        "dest": {int(d): r.astype(np.float64) for d, r in zip(dest.index, dest.values)},
        "market": {int(m): r.astype(np.float64) for m, r in zip(mkt.index, mkt.values)},
        "dest2mkt": dict(zip(modal["srch_destination_id"].astype(int), modal["hotel_market"].astype(int))),
        "global": glob / glob.sum(),
    }


tables = build_prior_tables(cut)
print("prior bookings", tables["n_bookings"], "destinations", len(tables["dest"]))


def prior_vector(d: int) -> np.ndarray:
    row = tables["dest"].get(d)
    if row is None:
        return tables["global"]
    pm = tables["market"][tables["dest2mkt"][d]]
    pm = pm / pm.sum()
    return (row + M_SMOOTH * pm) / (row.sum() + M_SMOOTH)


def prior_logp(dests: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(dests, return_inverse=True)
    tab = np.stack([np.log(prior_vector(int(d)) + EPS) for d in uniq]).astype(np.float32)
    out = np.zeros((len(dests), n_items), np.float32)
    out[:, cluster_ids] = tab[inv]
    return out


@torch.no_grad()
def scores(q_idx: torch.Tensor, batch: int = 8192) -> np.ndarray:
    """(n, n_items) query-token scores for an EMPTY history (all padding) and the given query indices."""
    out = []
    seq = torch.zeros((1, config["MAX_ITEM_LIST_LENGTH"]), dtype=torch.long)
    for i in range(0, len(q_idx), batch):
        qb = q_idx[i : i + batch].long()
        s = model.scores(seq.expand(len(qb), -1), qb)
        s[:, 0] = float("-inf")
        out.append(s.numpy())
    return np.concatenate(out)


def rank(sc: np.ndarray, y: np.ndarray) -> np.ndarray:
    sc = sc.copy()
    sc[:, 0] = -np.inf
    return target_rank(sc, y)


res, ranks = {}, {}
for sp in ("valid", "test"):
    c = cold[sp]
    y, q = c["y_model"], Qt[c["row"]]
    ml = torch.log_softmax(torch.from_numpy(scores(q)), dim=1).numpy()
    pr = prior_logp(c["dest"])
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    ml_shuf = torch.log_softmax(torch.from_numpy(scores(q[perm])), dim=1).numpy()
    pop = np.bincount(train_data.dataset.inter_feat[config["ITEM_ID_FIELD"]].numpy(), minlength=n_items).astype(np.float64)
    r = {
        "global popularity": rank(np.tile(pop, (len(y), 1)), y),
        "destination prior only": rank(pr, y),
        "query token alone (empty history)": rank(ml, y),
        f"query token + prior (w={W_FUSION})": rank(ml + W_FUSION * pr, y),
        "query token, query shuffled": rank(ml_shuf, y),
    }
    ranks[sp] = r
    res[sp] = {k: summarize(v, ks=KS) for k, v in r.items()}
    print(sp, {k: round(v["ndcg@5"], 4) for k, v in res[sp].items()})

# sanity: prior-only must reproduce notebook 06 hybrid on L = 0 (prior only there)
ref = SU["results"]["test"]["unseen, L=0 (first booking)"]["hybrid (SMLP4Rec + prior + sameDest)"]
got = res["test"]["destination prior only"]
print("nb06 hybrid (prior only) L=0 test ndcg@5", ref["ndcg@5"], "| here", got["ndcg@5"])

oov = Q[cold["test"]["row"], DEST] == 0
tr = ranks["test"]
slices = {}
for name, m in (("destination in query-token vocab", ~oov), ("destination out of vocab", oov)):
    slices[name] = {k: summarize(v[m], ks=(5,)) for k, v in tr.items()}
boot = {
    "query token alone minus prior only": paired_bootstrap(tr["query token alone (empty history)"], tr["destination prior only"], k=5),
    "query token + prior minus prior only": paired_bootstrap(tr[f"query token + prior (w={W_FUSION})"], tr["destination prior only"], k=5),
}
warm_masked = QRUN["diagnostics_test"]["history masked"]
result = {
    "run": "SMLP4Rec + query token on L = 0 rows (first bookings, empty history), checkpoint of notebook 01c, no retraining",
    "checkpoint": str(RUN_DIR / "best.pth"),
    "definition": "L = 0 rows as notebook 06: first booking of each user whose first booking is in the valid / test window; history = all padding",
    "n_rows": {sp: len(c["row"]) for sp, c in cold.items()},
    "w_fusion_chosen_in_01c": W_FUSION,
    "results": res,
    "reference_notebook_06_L0_hybrid_prior_only_test": ref,
    "reference_warm_rows_history_masked_test_01c": warm_masked,
    "test_slices_ndcg5_recall5": slices,
    "test_share_destination_oov": float(oov.mean()),
    "bootstrap_test_95ci": boot,
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", OUT)
