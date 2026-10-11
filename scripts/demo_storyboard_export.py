"""Precompute the step-by-step demo storyboard (four curated test bookings) as JSON.

For each curated test booking (picks of docs/plans/final_prep/04_demo_options.md §3, B replaced, see PICKS; drawn again
here with the same row mode and seed as the demo's /api/seed) the script scores all 100 clusters
with three model steps and the two baselines, and writes the top 5 plus the rank of the booked
cluster for each:

  step 1  original SMLP4Rec, history only: notebook 01e checkpoint
          (experiments/2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es/best.pth, best epoch 2;
          trained on the same rows as the final model: warm train targets + first bookings)
  step 2  + query token: final model, notebook 01d with early stopping (best epoch 8), model only
  step 3  + sameDest: the same scores plus the sameDest term, weights tuned on valid
          (`samedest_hybrid` in results/week4_rebuild/smlprec_query_cold_train_es_run.json)
  baselines ItemKNN and logistic regression from notebook 07b (trained on the same rows as the
          final model, as in the final table), exported to data/interim/demo_baselines_07b.npz by
          `scripts/export_baselines_for_demo.py --source 07b`; ties go to the more popular
          cluster and an empty history uses first-booking popularity, as in 07b

One what-if per booking (one search field or the history changed) is scored the same way.
A summary block copies the test Recall@5 of the five models from
results/week4_rebuild/baselines_vs_final_es.json. Check-in dates and booking dates for display
are read from data/interim/expedia_bookings_no_tgt.csv (same row order as the RecBole export;
checked row by row against user, cluster and destination). Nothing is trained and data/ is only read.

Output: results/week5_final/demo_storyboard.json (everything is also printed).

Usage (smlp4rec venv; about 1 to 2 min):
    .venv/Scripts/python.exe scripts/export_baselines_for_demo.py --source 07b   # once
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/demo_storyboard_export.py
    .venv/Scripts/python.exe scripts/export_baselines_for_demo.py --no-export         --check results/week5_final/demo_storyboard.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch  # before pandas (Windows c10.dll load order, see demo_ui.py)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_ui as du
import pandas as pd

from src.models.smlprec import SMLPREC

ROOT = du.ROOT
ES_FOLDER = "2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es"
ES_JSON = "smlprec_query_cold_train_es_run.json"
PLAIN_FOLDER = "2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es"
PLAIN_JSON = ROOT / "results" / "week4_rebuild" / "smlprec_plain_cold_train_es_run.json"
FINAL_JSON = ROOT / "results" / "week4_rebuild" / "baselines_vs_final_es.json"
BOOKINGS_CSV = ROOT / "data" / "interim" / "expedia_bookings_no_tgt.csv"
OUT = ROOT / "results" / "week5_final" / "demo_storyboard.json"
BASELINES_07B = ROOT / "data" / "interim" / "demo_baselines_07b.npz"  # export_baselines_for_demo.py
TOP_K = 5
SANITY_N = 2000  # warm test rows (seeded) to check that the plain checkpoint loads correctly

du.RUNS = {2022: (ES_FOLDER, ES_JSON)}  # early-stopping checkpoint only (as serve_es.py)

# Curated picks; `expect` = user and booked cluster found earlier, re-checked here.
# A, C, D are the picks of 04_demo_options.md §3. B replaces the 04 pick (warm seed 0): that
# user booked the same cluster at the same destination on the same day just before, so it shows a
# near-duplicate repeat rather than the method. The new B came from a scan of warm seeds 0..2999
# (destination in the history but not the last booking, query-token rank >= 2, sameDest rank 1).
PICKS = [
    {
        "key": "A",
        "role": "A new customer: first booking, no history",
        "mode": "cold",
        "seed": 960,
        "expect": {"user_id": 380617, "truth": 23},
        "whatif": {
            "label": "Same search by a family (2 adults, 2 children)",
            "fields": {"adults": 2, "children": 2, "rooms": 1},
        },
    },
    {
        "key": "B",
        "role": "A returning customer, back at the destination of an earlier trip",
        "mode": "warm",
        "seed": 385,
        "expect": {"user_id": 831510, "truth": 56},
        "whatif": {"label": "Same search, earlier bookings removed", "no_history": True},
    },
    {
        "key": "C",
        "role": "A family with three children, check-in in December",
        "mode": "warm",
        "seed": 10,
        "expect": {"user_id": 801932, "truth": 56},
        "whatif": {
            "label": "Same search by one adult travelling alone",
            "fields": {"adults": 1, "children": 0, "rooms": 1},
        },
    },
    {
        "key": "D",
        "role": "Where our model fails: the customer repeats an older choice",
        "mode": "warm",
        "seed": 2055,
        "expect": {"user_id": 692648, "truth": 37},
        "whatif": None,
    },
]

LR_KEY = "Logistic regression"
KNN_KEY = "ItemKNN"


class PlainRunner:
    """Original SMLP4Rec (history only), notebook 01e best checkpoint."""

    def __init__(self, data: du.Data) -> None:
        ck = torch.load(ROOT / "experiments" / PLAIN_FOLDER / "best.pth", weights_only=False)
        cfg = du.Cfg(ck["config"])
        cfg.update(
            USER_ID_FIELD="user_id",
            ITEM_ID_FIELD="item_id",
            LIST_SUFFIX="_list",
            ITEM_LIST_LENGTH_FIELD="item_length",
            NEG_PREFIX="neg_",
            n_heads=1,
            selected_features=[],
            pooling_mode="mean",
            device="cpu",
        )
        n_items = ck["state_dict"]["item_embedding.weight"].shape[0]
        ds = type("DS", (), {"num": staticmethod(lambda f: n_items)})()
        self.model = SMLPREC(cfg, ds)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.epoch = int(ck["epoch"])
        self.s_len = cfg["MAX_ITEM_LIST_LENGTH"]
        self.data = data

    @torch.no_grad()
    def scores(self, hist_events: np.ndarray) -> np.ndarray:
        """(100,) scores in raw cluster order; empty history = all-padding row, length 1 (as 01e)."""
        d = self.data
        ev = hist_events[-self.s_len :]
        seq = np.zeros((1, self.s_len), np.int64)
        seq[0, : len(ev)] = [d.cluster2id[int(d.item[e])] for e in ev]
        inter = {
            "item_id_list": torch.from_numpy(seq),
            "item_length": torch.tensor([max(len(ev), 1)]),
        }
        s = self.model.full_sort_predict(inter)
        s[:, 0] = float("-inf")
        return s.numpy()[0, d.cluster_ids]


def order_of(score: np.ndarray) -> np.ndarray:
    """Cluster ids sorted by descending score (stable on ties, as demo_ui)."""
    return np.argsort(-np.asarray(score, np.float64), kind="stable")


def rank_of(order: np.ndarray, c: int) -> int:
    return int(np.flatnonzero(order == c)[0]) + 1


def score_steps(
    plain: PlainRunner,
    runner: du.Runner,
    base: du.Baselines,
    hist: np.ndarray,
    fields: dict[str, int],
    truth: int,
) -> dict:
    """All steps for one history + search: top 5 per step, booked-cluster rank, movements."""
    d = runner.data
    o1 = order_of(plain.scores(hist))
    r = runner.score_all(hist, fields)
    o2 = order_of(r["p"][0])
    o3 = order_of(r["score"][0])
    sd = r["sd_p"][0]
    steps = {
        "plain": {
            "top": [{"cluster": int(c)} for c in o1[:TOP_K]],
            "truth_rank": rank_of(o1, truth),
        },
        "query": {
            "top": [
                {
                    "cluster": int(c),
                    "prev_rank": rank_of(o1, int(c)),
                    "p": round(float(r["p"][0, c]), 4),
                }
                for c in o2[:TOP_K]
            ],
            "truth_rank": rank_of(o2, truth),
        },
        "samedest": {
            "top": [
                {
                    "cluster": int(c),
                    "prev_rank": rank_of(o2, int(c)),
                    "samedest_share": round(float(sd[c]), 4),
                }
                for c in o3[:TOP_K]
            ],
            "truth_rank": rank_of(o3, truth),
            "known_destination": bool(r["known"]),
            "w_samedest": float(r["ws"]),
            "alpha": float(runner.alpha),
            "boosted_clusters": sorted(
                {
                    int(d.item[e])
                    for e in hist[-runner.s_len :]
                    if r["known"] and r["ws"] > 0 and int(d.codes[e, 0]) == fields["dest"]
                }
            ),
        },
    }
    b = base.recommend([int(d.item[e]) for e in hist], fields, truth)
    steps["baselines"] = {
        "lr": {"top": b[LR_KEY]["top"], "truth_rank": b[LR_KEY]["truth_rank"]},
        "knn": {
            "top": b[KNN_KEY]["top"],
            "truth_rank": b[KNN_KEY]["truth_rank"],
            "note": b[KNN_KEY]["note"],
        },
    }
    return steps


def read_rows(events: list[int]) -> pd.DataFrame:
    """Display columns of the given booking rows (chunked pass over the 600 MB CSV)."""
    want = set(events)
    cols = [
        "src_row",
        "user_id",
        "item_id",
        "timestamp",
        "ctx_dest_id",
        "ctx_ci",
        "ctx_co",
        "ctx_stay_nights",
        "ctx_adults",
        "ctx_children",
        "ctx_rooms",
        "ctx_is_package",
        "ctx_channel",
        "ctx_is_mobile",
        "ctx_party_type",
    ]
    parts, start = [], 0
    for chunk in pd.read_csv(BOOKINGS_CSV, usecols=cols, chunksize=500_000):
        chunk.index = np.arange(start, start + len(chunk))
        start += len(chunk)
        parts.append(chunk.loc[chunk.index.isin(want)])
    out = pd.concat(parts)
    assert start == len(du.STATE["data"].user), "CSV row count differs from the RecBole export"
    return out


def na(v) -> object:
    return None if pd.isna(v) else (int(v) if isinstance(v, (int, np.integer, float)) else str(v))


def search_of(row: pd.Series) -> dict:
    return {
        "dest": int(row["ctx_dest_id"]),
        "ci": na(row["ctx_ci"]),
        "co": na(row["ctx_co"]),
        "nights": na(row["ctx_stay_nights"]),
        "adults": na(row["ctx_adults"]),
        "children": na(row["ctx_children"]),
        "rooms": na(row["ctx_rooms"]),
        "package": int(row["ctx_is_package"]),
        "channel": int(row["ctx_channel"]),
        "mobile": int(row["ctx_is_mobile"]),
        "party_type": na(row["ctx_party_type"]),
    }


def fmt_steps(s: dict) -> str:
    def tops(x: list) -> str:
        return str([t["cluster"] if isinstance(t, dict) else t for t in x])

    return (
        f"plain {tops(s['plain']['top'])} rank {s['plain']['truth_rank']} | "
        f"+query {tops(s['query']['top'])} rank {s['query']['truth_rank']} | "
        f"+sameDest {tops(s['samedest']['top'])} rank {s['samedest']['truth_rank']} "
        f"(known dest {s['samedest']['known_destination']}, w {s['samedest']['w_samedest']}) | "
        f"LR rank {s['baselines']['lr']['truth_rank']} | "
        f"ItemKNN rank {s['baselines']['knn']['truth_rank']}"
    )


def sanity(plain: PlainRunner, runner: du.Runner, base: du.Baselines) -> dict:
    """Recall@5 of every step on SANITY_N seeded warm test rows (checks the checkpoint loading)."""
    d = runner.data
    rows = np.arange(runner.test_first, len(d.user))
    rows = rows[d.n_prior[rows] >= 1]
    pick = np.random.default_rng(0).choice(rows, SANITY_N, replace=False)
    hits = {"plain": 0, "query": 0, "samedest": 0, "lr": 0, "knn": 0}
    for e in pick:
        e = int(e)
        hist = d.history(int(d.user[e]), e)
        fields = {f: int(c) for f, c in zip(du.QUERY_FIELDS, d.codes[e])}
        s = score_steps(plain, runner, base, hist, fields, int(d.item[e]))
        for k in ("plain", "query", "samedest"):
            hits[k] += s[k]["truth_rank"] <= TOP_K
        hits["lr"] += s["baselines"]["lr"]["truth_rank"] <= TOP_K
        hits["knn"] += s["baselines"]["knn"]["truth_rank"] <= TOP_K
    return {
        "rows": f"{SANITY_N} warm test rows, numpy seed 0",
        **{k: round(v / SANITY_N, 4) for k, v in hits.items()},
    }


def summary_table() -> dict:
    """Test Recall@5 (warm, cold) of the five models, copied from the final comparison json."""
    fin = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    names = {
        "plain": "SMLP4Rec plain, no query token (01e, best epoch 2)",
        "query": "SMLP4Rec + query token (final, best epoch 8)",
        "samedest": "SMLP4Rec + query token + sameDest (final, best epoch 8)",
        "lr": "Logistic regression (C=1.0, pass 8, destination + context, no history)",
        "knn": "ItemKNN (cosine, K=50, history only; cold: first-booking popularity)",
    }
    out = {
        "source": str(FINAL_JSON.relative_to(ROOT)).replace("\\", "/"),
        "split": fin["split"],
        "seed": fin["seed"],
        "baselines_from": "notebook 07b",
        "models": {},
    }
    for k, name in names.items():
        m = fin["models"][name]
        out["models"][k] = {
            "name": name,
            "warm": {"n": m["warm"]["n"], "recall@5": m["warm"]["recall@5"]},
            "cold": {"n": m["cold (L=0)"]["n"], "recall@5": m["cold (L=0)"]["recall@5"]},
        }
    return out


def main() -> None:
    data = du.Data()
    du.STATE.update(data=data, runners={}, baselines=du.Baselines(BASELINES_07B))
    runner = du.get_runner(2022)
    plain = PlainRunner(data)
    base = du.STATE["baselines"]
    plain_meta = json.loads(PLAIN_JSON.read_text(encoding="utf-8"))
    es_meta = json.loads((ROOT / "results" / "week4_rebuild" / ES_JSON).read_text(encoding="utf-8"))
    assert plain.epoch == plain_meta["best_epoch_by_valid"]
    assert plain_meta["splits"]["test"]["first_target_event"] == runner.test_first
    print(
        f"plain checkpoint: {PLAIN_FOLDER}, best epoch {plain.epoch}; "
        f"final checkpoint: {ES_FOLDER}, best epoch {es_meta.get('best_epoch_by_valid')}; "
        f"sameDest alpha {runner.alpha}, weights {runner.ws}"
    )

    bookings, all_events = [], []
    for pk in PICKS:
        s = du.api_seed({"mode": pk["mode"], "seed": pk["seed"]})
        e, uid, truth = s["event"], s["user_id"], s["truth"]
        hist = data.history(uid, e)
        match = all(s[k] == v for k, v in pk["expect"].items())
        steps = score_steps(plain, runner, base, hist, s["fields"], truth)
        b = {
            "key": pk["key"],
            "role": pk["role"],
            "mode": pk["mode"],
            "seed": pk["seed"],
            "event": e,
            "user_id": uid,
            "truth": truth,
            "n_prior": s["n_prior"],
            "matches_expected": match,
            "fields": s["fields"],
            "history_events": [int(x) for x in hist[-runner.s_len :]],
            "steps": steps,
        }
        wi = pk["whatif"]
        if wi:
            w_hist = np.zeros(0, np.int64) if wi.get("no_history") else hist
            w_fields = {**s["fields"], **wi.get("fields", {})}
            b["whatif"] = {
                "label": wi["label"],
                "no_history": bool(wi.get("no_history")),
                "changed": wi.get("fields", {}),
                "steps": score_steps(plain, runner, base, w_hist, w_fields, truth),
            }
        bookings.append(b)
        all_events += [e, *b["history_events"]]
        print(
            f"\n[{pk['key']}] seed {pk['seed']} ({pk['mode']}) event {e} user {uid} "
            f"L={s['n_prior']} actual {truth}; matches expected: {match}"
        )
        print("   ", fmt_steps(steps))
        if wi:
            print(f"    what-if '{wi['label']}':", fmt_steps(b["whatif"]["steps"]))

    rows = read_rows(all_events)
    for b in bookings:
        r = rows.loc[b["event"]]
        assert int(r["user_id"]) == b["user_id"] and int(r["item_id"]) == b["truth"]
        assert int(r["ctx_dest_id"]) == b["fields"]["dest"]
        b["search"] = search_of(r)
        b["src_row"] = int(r["src_row"])
        b["booked_on"] = str(r["timestamp"])[:10]
        b["history"] = []
        for h in b["history_events"]:
            hr = rows.loc[h]
            assert int(hr["user_id"]) == b["user_id"] and int(hr["item_id"]) == int(data.item[h])
            assert int(hr["ctx_dest_id"]) == int(data.codes[h, 0])
            b["history"].append(
                {
                    "event": h,
                    "booked_on": str(hr["timestamp"])[:10],
                    "dest": int(hr["ctx_dest_id"]),
                    "cluster": int(hr["item_id"]),
                    "ci": na(hr["ctx_ci"]),
                    "co": na(hr["ctx_co"]),
                }
            )
        print(
            f"[{b['key']}] search {b['search']} booked on {b['booked_on']}; history: "
            + ", ".join(f"{h['booked_on']} d{h['dest']} c{h['cluster']}" for h in b["history"])
        )

    chk = sanity(plain, runner, base)
    summ = summary_table()
    print("\nsanity check, Recall@5 on a warm test sample:", chk)
    print(
        "full warm test Recall@5 (final table; baselines there are 07b):",
        {k: v["warm"]["recall@5"] for k, v in summ["models"].items()},
    )
    print(
        "cold test Recall@5 (final table):",
        {k: v["cold"]["recall@5"] for k, v in summ["models"].items()},
    )

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    out = {
        "dataset": "Expedia Hotel Recommendations (public), bookings, burst repeats collapsed",
        "split": "test (global temporal 80/10/10), seed 2022",
        "generated_by": "scripts/demo_storyboard_export.py",
        "git_commit": commit,
        "models": {
            "plain": f"SMLP4Rec, history only (notebook 01e, {PLAIN_FOLDER}, best epoch {plain.epoch}; "
            "trained on the same rows as the final model)",
            "query": f"SMLP4Rec + query token (notebook 01d, {ES_FOLDER}, best epoch "
            f"{es_meta.get('best_epoch_by_valid')}), model scores only",
            "samedest": "the same + sameDest (alpha and weights tuned on valid, "
            f"results/week4_rebuild/{ES_JSON})",
            "baselines": "ItemKNN and logistic regression from notebook 07b "
            "(data/interim/demo_baselines_07b.npz), the same models as the final table; "
            "real-booking ranks checked against data/interim/baselines_es_ranks_test.parquet",
        },
        "selection": "picked for illustration from seeded test draws "
        "(docs/plans/final_prep/04_demo_options.md §3); D is a deliberate miss",
        "bookings": bookings,
        "sanity_sample": chk,
        "summary": summ,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
