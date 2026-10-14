"""User-level test evaluation: one fixed top-20 per user from two prediction steps, three rerankers.

Protocol in `nbp.eval.two_step` (steps, probabilities, rerankers) and
`nbp.eval.user_level.fixed_top_user_metrics` (set-based Recall@k / NDCG@k, mean over users).
No retraining: the SMLP4Rec checkpoints of the week-6 retrain (3 seeds) are reloaded, the baselines
are refitted exactly as notebook 03 (deterministic). Two venvs, as in notebook 05:

    .venv/Scripts/python.exe scripts/eval_user_top20.py baselines   # implicit (ItemKNN), popularity
    ~/.venvs/smlp4rec/Scripts/python.exe scripts/eval_user_top20.py smlp   # RecBole (SMLP4Rec)
    .venv/Scripts/python.exe scripts/eval_user_top20.py report      # merge the parts, write tables

Writes:
- `results/week6_retrain/user_top20.json`: metrics per model x seed x method x user set;
- `results/week6_retrain/user_top20_tables.md`: the tables (reported seed only);
- `data/interim/week6/user_top20.npz`: both step lists and the reranked lists per user.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch  # torch before pandas-importing modules (Windows c10.dll)
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from nbp.data.query_features import build_vocab, encode  # noqa: E402
from nbp.data.split import TRAIN  # noqa: E402
from nbp.eval import runner as R  # noqa: E402
from nbp.eval.two_step import RERANKERS, score_two_steps, user_plan  # noqa: E402
from nbp.eval.user_level import fixed_top_user_metrics  # noqa: E402
from nbp.paths import CONFIGS, EXPERIMENTS, INTERIM, PROCESSED  # noqa: E402

KS = (5, 10, 20)
SPLIT = yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))
TRAIN_CFG = yaml.safe_load((CONFIGS / "train.yaml").read_text(encoding="utf-8"))
MODEL_CFG = yaml.safe_load((CONFIGS / "smlprec_expedia.yaml").read_text(encoding="utf-8"))
SEEDS = list(SPLIT["seeds"])
BATCH = int(TRAIN_CFG["eval_batch"])
RES = ROOT / "results" / "week6_retrain"
OUT_NPZ = INTERIM / "week6"
LABEL = {"plain": "SMLP4Rec (plain)", "query": "+ query embedding", "main": "+ query embedding + sameDest",
         "itemknn": "ItemKNN", "popularity": "Global popularity"}


def res(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def ckpt(p: str) -> Path:
    p = Path(p)
    return p / "best.pth" if p.exists() else EXPERIMENTS / Path(str(p).replace("\\", "/")).name / "best.pth"


def load(model, path: Path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model.eval()


PART = sys.argv[1] if len(sys.argv) > 1 else ""
assert PART in ("baselines", "smlp", "report"), "usage: eval_user_top20.py baselines|smlp|report"

rows = R.load_rows(PROCESSED / "expedia_date_split" / "rows.npz")
codes = rows["codes"]
N_ITEMS = int(rows["y"].max()) + 1
v = R.window_views(rows)["test"]
it = R.window_index(rows)["test"]
plan = user_plan(v, rows["user"][it], rows["y"][it].astype(np.int64))
warm = rows["L"][it][plan.first] >= 1
USER_SETS = {"all": np.ones(len(plan.user), bool), "warm": warm, "cold": ~warm}
n_true = np.array([len(t) for t in plan.truths])
info = {"users": int(len(plan.user)), "warm_users": int(warm.sum()), "cold_users": int((~warm).sum()),
        "users_with_one_test_booking": int((plan.second < 0).sum()),
        "mean_distinct_clusters_per_user": float(n_true.mean()),
        "share_users_with_more_than_20_distinct": float((n_true > 20).mean())}
print(info, flush=True)

tr = np.flatnonzero(rows["window"] == TRAIN)
y_tr = rows["y"][tr].astype(np.int64)
vocab = build_vocab(codes, rows["event"][tr])  # train window only, as in notebook 02
Q = torch.from_numpy(encode(codes, vocab))
field_sizes = [len(x) + 1 for x in vocab]


def baseline_scorers():
    """(model key, seed or None, ScoreFn, probability kind)."""
    from nbp.baselines.itemknn import fit_itemknn, itemknn_scorer, user_item_matrix
    from nbp.baselines.popularity import fit_popularity, popularity_scorer

    pop = fit_popularity(y_tr, N_ITEMS)
    yield "popularity", None, popularity_scorer(pop), "sum"
    first = tr[rows["L"][tr] == 0]
    pop_first = np.bincount(rows["y"][first], minlength=N_ITEMS) / len(first)
    sim = fit_itemknn(user_item_matrix(rows["user"][tr], y_tr, N_ITEMS), res("itemknn")["k_chosen"])
    yield "itemknn", None, itemknn_scorer(sim, pop, pop_first, TRAIN_CFG["itemknn_tie_eps"]), "sum"


def smlp_scorers():
    from nbp.models import scorers as S

    with torch.no_grad():
        for seed in SEEDS:
            m = res(f"plain_s{seed}")
            cfg = S.model_config(MODEL_CFG, {"seed": seed, "hidden_dropout_prob": m["hidden_dropout_prob"]})
            yield "plain", seed, S.plain_scorer(load(S.build_plain(cfg, N_ITEMS), ckpt(m["run_dir"]))), "softmax"
            mq, mm = res(f"query_s{seed}"), res(f"main_s{seed}")
            cfg = S.model_config(MODEL_CFG, {"seed": seed, "hidden_dropout_prob": mq["hidden_dropout_prob"]})
            q_sf = S.query_scorer(load(S.build_query(cfg, N_ITEMS, Q, field_sizes), ckpt(mq["run_dir"])))
            yield "query", seed, q_sf, "softmax"
            sd = mm["samedest"]
            yield "main", seed, S.samedest_scorer(q_sf, sd["alpha"], sd["w"], sd["recency"]), "softmax"


PART_JSON = INTERIM / "week6" / "user_top20_{}.json"
PART_NPZ = INTERIM / "week6" / "user_top20_{}.npz"
results = {}  # model -> seed -> method -> user set -> metrics
store = {}
for key, seed, sf, kind in {"baselines": baseline_scorers, "smlp": smlp_scorers}.get(PART, lambda: [])():
    t0 = time.time()
    t = score_two_steps(v, plan, sf, kind, BATCH)
    sk = "-" if seed is None else str(seed)
    for meth, fn in RERANKERS.items():
        top = fn(t, N_ITEMS)
        m = fixed_top_user_metrics(top, plan.truths, KS)
        per = m.pop("per_user")
        results.setdefault(key, {}).setdefault(sk, {})[meth] = {
            us: {f"{x}@{k}": float(per[f"{x}@{k}"][mask].mean()) for x in ("recall", "ndcg") for k in KS}
            | {"users": int(mask.sum())}
            for us, mask in USER_SETS.items()
        }
        store[f"{key}_{sk}_{meth}"] = (top - 1).astype(np.int8)  # model id -> hotel_cluster
    store[f"{key}_{sk}_step1"] = (t.ids1 - 1).astype(np.int8)
    store[f"{key}_{sk}_step2"] = (t.ids2 - 1).astype(np.int8)
    store[f"{key}_{sk}_p1"] = t.p1.astype(np.float32)
    store[f"{key}_{sk}_p2"] = t.p2.astype(np.float32)
    a = results[key][sk]
    print(f"{key} {sk}: " + " | ".join(f"{mt} ndcg@10 {a[mt]['all']['ndcg@10']:.4f}" for mt in RERANKERS)
          + f" ({time.time() - t0:.0f}s)", flush=True)

if PART != "report":
    if PART == "baselines":  # popularity ignores the history, so every reranker returns the same list
        pk = [store[f"popularity_-_{mt}"] for mt in RERANKERS]
        assert all(np.array_equal(pk[0], x) for x in pk), "popularity rerankers differ"
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(PART_NPZ).format(PART), user=plan.user, **store)
    Path(str(PART_JSON).format(PART)).write_text(json.dumps(results), encoding="utf-8")
    print("wrote part", PART)
    sys.exit(0)

for part in ("baselines", "smlp"):
    results |= json.loads(Path(str(PART_JSON).format(part)).read_text(encoding="utf-8"))
    z = np.load(str(PART_NPZ).format(part))
    assert np.array_equal(z["user"], plan.user)
    store |= {k: z[k] for k in z.files if k != "user"}
REPORT_SEED = str(SPLIT["report_seed"])  # one fixed seed for every reported number (configs/split.yaml)
NL = chr(10)
CHOSEN = "m2_merge"  # reranker chosen 2026-10-14 (docs/plans/plan_oct_14_user_level_top20.md)
results = {k: {sd: x for sd, x in v.items() if sd in ("-", REPORT_SEED)} for k, v in results.items()}
cols = [f"{x}@{k}" for x in ("ndcg", "recall") for k in KS]


def val(key, meth, us, metric):
    return results[key][REPORT_SEED if REPORT_SEED in results[key] else "-"][meth][us][metric]


# selection record: the three rerankers side by side (all users, seed 2024); the reported tables use CHOSEN only
sel = [f"# Reranker selection record, seed {REPORT_SEED}, all {info['users']:,} test users", "",
       f"Chosen: `{CHOSEN}`. The other two rerankers are kept here only as the comparison behind the choice.", ""]
for meth in RERANKERS:
    sel += [f"## {meth}", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    sel += [f"| {LABEL[k]} | " + " | ".join(f"{val(k, meth, 'all', c):.4f}" for c in cols) + " |" for k in LABEL]
    sel.append("")
(RES / "user_top20_reranker_choice.md").write_text(NL.join(sel), encoding="utf-8")

# everything else: the chosen reranker only
results = {k: {sd: {CHOSEN: x[CHOSEN]} for sd, x in v.items()} for k, v in results.items()}
store = {k: x for k, x in store.items() if "_2022_" not in k and "_2023_" not in k
         and not k.endswith(("m1_first", "m3_sqrt_alpha"))}
np.savez_compressed(OUT_NPZ / "user_top20.npz", user=plan.user, **store)
(RES / "user_top20.json").write_text(json.dumps(
    {"protocol": "docs/eval_protocol.md (user-level fixed top-20)", "seed": REPORT_SEED, "reranker": CHOSEN, "ks": list(KS), "info": info,
     "results": {k: v[REPORT_SEED if REPORT_SEED in v else "-"][CHOSEN] for k, v in results.items()}}, indent=1), encoding="utf-8")

lines = [f"# User-level test metrics: fixed top-20 per user (two steps + rerank `{CHOSEN}`), seed {REPORT_SEED}", "",
         "Expedia, full test window (2014-11-01 to 2014-12-31), mean over test users (set-based Recall@k / NDCG@k, `docs/eval_protocol.md`). "
         f"SMLP4Rec rows: seed {REPORT_SEED}; ItemKNN and popularity are deterministic.", "",
         "Users: " + ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v:.3f}" for k, v in info.items()), ""]
for us in USER_SETS:
    lines += [f"## {us} users", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {LABEL[k]} | " + " | ".join(f"{val(k, CHOSEN, us, c):.4f}" for c in cols) + " |" for k in LABEL]
    lines.append("")
(RES / "user_top20_tables.md").write_text(NL.join(lines), encoding="utf-8")
print("wrote", RES / "user_top20.json", RES / "user_top20_tables.md")
