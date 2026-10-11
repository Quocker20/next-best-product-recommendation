"""Export the two fitted baselines as plain arrays so the demo code can load them in the SMLP4Rec
venv (numpy 1.23), where the joblib files do not unpickle.

Sources:
  --source 07   notebook 07 models (basic_baselines_models.joblib) -> data/interim/demo_baselines.npz
                (default; used by the live demo, scripts/demo_ui.py)
  --source 07b  notebook 07b models (baselines_es_models.joblib, trained on the same rows as the
                final model) -> data/interim/demo_baselines_07b.npz (used by the demo storyboard).
                07b's dump has no popularity vector, so both are recomputed here exactly as in 07b:
                `pop` = cluster counts over all training bookings (every booking up to the last
                train target; tie-break), `pop_first` = counts over the first bookings among them
                (ItemKNN fallback for an empty history).
  --check JSON  after exporting, compare the baseline ranks of the real bookings in the storyboard
                json (results/week5_final/demo_storyboard.json) with 07b's saved test ranks
                (data/interim/baselines_es_ranks_test.parquet), matched by src_row; asserts equality.

Run with the project .venv:
    .venv/Scripts/python.exe scripts/export_baselines_for_demo.py [--source 07b] [--check JSON]
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nbp.paths import ROOT

INTERIM = ROOT / "data" / "interim"
SOURCES = {
    "07": (INTERIM / "basic_baselines_models.joblib", INTERIM / "demo_baselines.npz"),
    "07b": (INTERIM / "baselines_es_models.joblib", INTERIM / "demo_baselines_07b.npz"),
}
BOOKINGS = INTERIM / "expedia_bookings.parquet"
FINAL_JSON = ROOT / "results" / "week4_rebuild" / "smlprec_query_cold_train_es_run.json"
RANKS_07B = INTERIM / "baselines_es_ranks_test.parquet"
N_CLUSTERS = 100


def popularity_07b() -> tuple[np.ndarray, np.ndarray]:
    """Training-set popularity (all bookings, first bookings) as in notebook 07b, cells 3 and 5."""
    d = pd.read_parquet(BOOKINGS, columns=["user_id", "item_id", "timestamp", "src_row"])
    d["ts"] = ((d["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(
        "int64"
    )
    d = d.sort_values(["ts", "user_id", "src_row"], kind="stable").reset_index(drop=True)
    d["pos"] = d.groupby("user_id").cumcount()
    last_train = json.loads(FINAL_JSON.read_text(encoding="utf-8"))["splits"]["train"][
        "last_target_event"
    ]
    train = d.iloc[: last_train + 1]
    item = train["item_id"].to_numpy().astype(np.int64)
    pop = np.bincount(item, minlength=N_CLUSTERS).astype(np.float64)
    pop_first = np.bincount(item[train["pos"].to_numpy() == 0], minlength=N_CLUSTERS).astype(
        np.float64
    )
    print(f"07b training rows {len(train):,} (first bookings {int(pop_first.sum()):,})")
    return pop, pop_first


def export(source: str) -> None:
    src, out = SOURCES[source]
    m = joblib.load(src)
    sim = m["knn_similarity"]
    sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
    enc, lr = m["lr_encoder"], m["lr"]
    cats = {f"cat{i}": np.asarray(c).astype(str) for i, c in enumerate(enc.categories_)}
    extra = {}
    if source == "07b":
        pop, pop_first = popularity_07b()
        extra = {"pop": pop, "pop_first": pop_first}
    else:
        extra = {"pop": np.asarray(m["pop"])}
    np.savez_compressed(
        out,
        # 07b keeps float64: near-ties (e.g. two self-similarities of 1 - 1e-13) decide ranks
        knn_similarity=sim.astype(np.float64 if source == "07b" else np.float32),
        coef=lr.coef_.astype(np.float32),
        intercept=lr.intercept_.astype(np.float32),
        classes=lr.classes_,
        n_cat=len(cats),
        **extra,
        **cats,
    )
    print("saved", out, "| similarity", sim.shape, "| coef", lr.coef_.shape, "| C", lr.C)


def check(path: Path) -> None:
    """Assert storyboard baseline ranks (real bookings) == 07b saved test ranks."""
    sb = json.loads(path.read_text(encoding="utf-8"))
    r = pd.read_parquet(RANKS_07B).set_index("src_row")
    knn_col = next(c for c in r.columns if c.startswith("ItemKNN"))
    lr_col = next(c for c in r.columns if c.startswith("Logistic"))
    for b in sb["bookings"]:
        row = r.loc[b["src_row"]]
        got = b["steps"]["baselines"]
        want = {"lr": int(row[lr_col]), "knn": int(row[knn_col])}
        print(
            f"[{b['key']}] src_row {b['src_row']} ({row['slice']}): LR {got['lr']['truth_rank']} vs "
            f"saved {want['lr']}; ItemKNN {got['knn']['truth_rank']} vs saved {want['knn']}"
        )
        assert got["lr"]["truth_rank"] == want["lr"] and got["knn"]["truth_rank"] == want["knn"]
    print("storyboard baseline ranks match the 07b saved test ranks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=sorted(SOURCES), default="07")
    ap.add_argument("--check", type=Path, help="storyboard json to verify against 07b ranks")
    ap.add_argument("--no-export", action="store_true", help="only run --check")
    a = ap.parse_args()
    if not a.no_export:
        export(a.source)
    if a.check:
        check(a.check)


if __name__ == "__main__":
    main()
