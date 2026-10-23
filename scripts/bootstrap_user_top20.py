"""Paired user bootstrap of the user-level test (fixed top-20 per user, reported seed).

Differences main - query embedding (the sameDest gain) and query embedding - plain, for Recall@K and
NDCG@K, K = 5, 10, 20, per reranker, on all / warm / cold users and on warm users split by whether the
destination searched at booking 1 is known to the user (an earlier booking at that destination in the
history) or new. Users are the independent units, so users are resampled (1000 draws, the same draw for
both models). Reads `data/interim/week6/user_top20.npz`; writes `results/user_top20_bootstrap.{json,md}`.
Runs in the project `.venv`.
"""

import json

import numpy as np
import yaml

from nbp.data.split import TEST
from nbp.eval import runner as R
from nbp.eval.two_step import user_plan
from nbp.eval.user_level import fixed_top_user_metrics
from nbp.paths import CONFIGS, INTERIM, PROCESSED, ROOT

KS = (5, 10, 20)
CHOSEN = "m2_merge"  # the reranker chosen on 2026-10-14; the others are not bootstrapped
RERANKERS = [CHOSEN]
N_BOOT = 1000
SPLIT = yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))
SEED = str(SPLIT["report_seed"])
RES = ROOT / "results"


def main() -> None:
    rows = R.load_rows(PROCESSED / "expedia_date_split" / "rows.npz")
    v = R.window_views(rows)["test"]
    it = R.window_index(rows)["test"]
    plan = user_plan(v, rows["user"][it], rows["y"][it].astype(np.int64))
    z = np.load(INTERIM / "week6" / "user_top20.npz")
    assert np.array_equal(z["user"], plan.user)

    first = plan.first
    warm = v.seq_len[first] >= 1
    slot = np.arange(v.seq.shape[1])[None, :] < v.seq_len[first][:, None]
    known = (slot & (v.h_dest[first] == v.dest[first][:, None])).any(1)
    SETS = {"all": np.ones(len(first), bool), "warm": warm, "cold": ~warm,
            "warm, destination known": warm & known, "warm, destination new": warm & ~known}
    print({k: int(m.sum()) for k, m in SETS.items()}, flush=True)

    # window / split sanity (data actually used)
    ts = np.load(PROCESSED / "expedia_date_split" / "rows.npz")["window"]
    print("test rows", int((ts == TEST).sum()), "users", len(first), flush=True)

    def per_user(key: str, meth: str) -> dict:
        sk = "-" if key in ("itemknn", "popularity") else SEED
        top = z[f"{key}_{sk}_{meth}"].astype(np.int64) + 1  # hotel_cluster -> model id
        return fixed_top_user_metrics(top, plan.truths, KS)["per_user"]

    def boot(d: np.ndarray, rng) -> tuple[float, float, float]:
        n = len(d)
        m = np.empty(N_BOOT)
        for b in range(N_BOOT):
            m[b] = d[rng.integers(0, n, n)].mean()
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


    PAIRS = {"sameDest gain (main - query embedding)": ("main", "query"),
             "query embedding gain (query embedding - plain)": ("query", "plain")}
    out = {}
    for meth in RERANKERS:
        P = {k: per_user(k, meth) for k in ("plain", "query", "main")}
        for pname, (a, b) in PAIRS.items():
            for us, mask in SETS.items():
                cell = {}
                for metric in [f"{x}@{k}" for x in ("ndcg", "recall") for k in KS]:
                    d = (P[a][metric] - P[b][metric])[mask]
                    diff, lo, hi = boot(d, np.random.default_rng(0))
                    cell[metric] = {"diff": diff, "ci95": [lo, hi]}
                out.setdefault(meth, {}).setdefault(pname, {})[us] = {"users": int(mask.sum()), **cell}
        print(meth, "done", flush=True)

    (RES / "user_top20_bootstrap.json").write_text(json.dumps(
        {"seed": SEED, "n_boot": N_BOOT, "unit": "user (resampled)", "ks": list(KS), "results": out}, indent=1), encoding="utf-8")

    cols = [f"{x}@{k}" for x in ("ndcg", "recall") for k in KS]
    lines = [f"# Paired user bootstrap, user-level top-20 test, seed {SEED}", "",
             f"Difference of the mean per-user metric with a 95 % percentile interval ({N_BOOT} draws, users resampled, same draw for both models). "
             "A difference is significant when its interval excludes 0.", ""]
    for meth in RERANKERS:
        for pname in PAIRS:
            lines += [f"## {pname}, {meth}", "", "| User set | users | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
            for us in SETS:
                c = out[meth][pname][us]
                lines.append(f"| {us} | {c['users']:,} | " + " | ".join(
                    f"{c[m]['diff']:+.4f} [{c[m]['ci95'][0]:+.4f}, {c[m]['ci95'][1]:+.4f}]" for m in cols) + " |")
            lines.append("")
    (RES / "user_top20_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RES / "user_top20_bootstrap.md")


if __name__ == "__main__":
    main()
