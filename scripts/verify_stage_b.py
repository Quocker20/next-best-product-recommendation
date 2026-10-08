"""Final consistency check of the stage-B results (collapsed data). Prints PASS / FAIL per check.

Reads only committed result files and data/interim/expedia_bookings.parquet; writes
results/week4_rebuild/verification.json. A failing check exits with status 1.

Checks
1. Split sizes agree across the plain run, the query runs (01c, 01d) and notebook 07, and equal
   rows - users of the collapsed table (80/10/10 of the next-booking targets).
2. 01d training rows: the extra first bookings are all before the first valid target, number
   exactly the first bookings of the train window, and share no user with the valid / test cold rows.
3. Cold row counts agree (cold JSONs, notebook 07) and equal the first bookings in the valid / test
   windows computed from the table.
4. benchmark_results.csv equals what `build_benchmark_results.py` produces from the JSON files
   (rebuilt in memory), and all_events n = warm n + cold n for every model.
5. Metrics are monotone in K (Recall and NDCG) and within [0, 1] in the CSV.
6. No forbidden raw column feeds the query fields (leakage audit) and no history-after-target.

Usage: .venv/Scripts/python.exe scripts/verify_stage_b.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from nbp.paths import INTERIM, ROOT

RES = ROOT / "results" / "week4_rebuild"
CSV = ROOT / "reports" / "benchmark_results.csv"
OUT = RES / "verification.json"
checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def main() -> None:
    d = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=["user_id", "ts_unix"])
    n_rows, n_users = len(d), d["user_id"].nunique()
    targets = n_rows - n_users
    n_va = int(0.1 * targets)
    n_tr = int(0.8 * targets)
    exp = {"train": n_tr, "valid": n_va, "test": targets - n_tr - n_va}

    plain, q1c, q1d = (
        load("smlprec_plain_run.json"),
        load("smlprec_query_run.json"),
        load("smlprec_query_cold_train_run.json"),
    )
    base = load("basic_baselines.json")
    for tag, run in (("plain", plain), ("01c", q1c), ("01d", q1d)):
        got = {k: v["targets"] for k, v in run["splits"].items()}
        check(f"split sizes {tag} = rows - users, 80/10/10", got == exp, f"{got}")
    bc = base["split"]["counts"]
    check("split sizes notebook 07", {k: bc[k] for k in exp} == exp, str({k: bc[k] for k in exp}))
    same_ranges = all(
        plain["splits"][s][k] == r["splits"][s][k]
        for r in (q1c, q1d)
        for s in exp
        for k in ("first_target_event", "last_target_event")
    )
    check("event ranges of the splits identical in plain, 01c and 01d", same_ranges)

    # 2/3: first bookings
    d = d.reset_index(drop=True)
    first_idx = d.groupby("user_id").head(1).index.to_numpy()
    last_train = q1d["splits"]["train"]["last_target_event"]
    first_valid, last_valid = (
        q1d["splits"]["valid"]["first_target_event"],
        q1d["splits"]["valid"]["last_target_event"],
    )
    cold_train = first_idx[first_idx < last_train]
    cold_valid = first_idx[(first_idx > last_train) & (first_idx < last_valid)]
    cold_test = first_idx[first_idx > last_valid]
    check(
        "01d cold training rows = first bookings before the last train target",
        len(cold_train) == q1d["cold_training_rows"],
        f"{len(cold_train)} vs {q1d['cold_training_rows']}",
    )
    check(
        "01d cold training rows end before the first valid target",
        cold_train.max() < first_valid,
        f"max {cold_train.max()} < {first_valid}",
    )
    users = d["user_id"].to_numpy()
    overlap = np.intersect1d(
        users[cold_train], np.concatenate([users[cold_valid], users[cold_test]])
    )
    check(
        "no user is both a cold training row and a cold valid / test row",
        len(overlap) == 0,
        f"{len(overlap)} users",
    )
    check(
        "cold test rows: notebook 07 = table = cold JSONs",
        bc["test_cold"]
        == len(cold_test)
        == load("smlprec_query_cold_L0.json")["n_rows"]["test"]
        == load("smlprec_query_cold_L0_cold_train.json")["n_rows"]["test"],
        f"{bc['test_cold']} / {len(cold_test)}",
    )
    check(
        "cold valid rows: notebook 07 = table = cold JSONs",
        bc["valid_cold"]
        == len(cold_valid)
        == load("smlprec_query_cold_L0.json")["n_rows"]["valid"],
        f"{bc['valid_cold']} / {len(cold_valid)}",
    )

    # 4: CSV equals a rebuild from the JSON files
    old = pd.read_csv(CSV)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_benchmark_results.py")],
        check=True,
        capture_output=True,
    )
    new = pd.read_csv(CSV)
    check(
        "benchmark_results.csv equals a rebuild from the JSON files",
        old.equals(new),
        f"{len(new)} rows",
    )
    for model, g in new.groupby("model"):
        by = g.set_index("row_set")["n"]
        if {"warm", "cold", "all_events"} <= set(by.index):
            check(
                f"all_events n = warm n + cold n ({model[:44]})",
                by["all_events"] == by["warm"] + by["cold"],
                f"{by['all_events']}",
            )

    # 5: metric sanity
    ok = True
    for suffix in ("recall", "ndcg"):
        a, b, c = new[f"{suffix}@5"], new[f"{suffix}@10"], new[f"{suffix}@20"]
        ok &= bool(((a <= b + 1e-9) & (b <= c + 1e-9) & (a >= 0) & (c <= 1)).all())
    check("metrics monotone in K and within [0, 1]", ok)

    # 6: leakage audit
    la = load("leakage_audit.json")
    check(
        "no forbidden raw column in the query fields",
        la["forbidden_raw_columns_in_query_fields"] == [],
    )
    check(
        "no history booking at or after its target",
        all(v["violations"] == 0 for v in la["history_before_target"].values()),
    )

    OUT.write_text(
        json.dumps({"all_passed": all(c["pass"] for c in checks), "checks": checks}, indent=2),
        encoding="utf-8",
    )
    failed = [c["check"] for c in checks if not c["pass"]]
    print(
        f"\n{len(checks) - len(failed)}/{len(checks)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
