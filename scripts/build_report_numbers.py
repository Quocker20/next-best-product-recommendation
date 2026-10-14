"""One sheet of report-ready numbers: the user-level test, one seed, one reranker (numpy only).

Reads `results/week6_retrain/user_top20.json` and `user_top20_bootstrap.json` (both already restricted to the
reported seed and the chosen reranker) and writes `results/week6_retrain/report_numbers.md`. Nothing is typed by
hand; every number is copied from the two json files.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "week6_retrain"
main = json.loads((RES / "user_top20.json").read_text(encoding="utf-8"))
boot = json.loads((RES / "user_top20_bootstrap.json").read_text(encoding="utf-8"))
assert main["seed"] == boot["seed"] and main["reranker"] in boot["results"]
LABEL = {"plain": "SMLP4Rec (plain)", "query": "+ query embedding", "main": "+ query embedding + sameDest",
         "itemknn": "ItemKNN", "popularity": "Global popularity"}
COLS = [f"{x}@{k}" for x in ("ndcg", "recall") for k in main["ks"]]
info, res = main["info"], main["results"]
b = boot["results"][main["reranker"]]

L = [f"# Report numbers: user-level test, seed {main['seed']}, reranker {main['reranker']}", "",
     "Single version of every number for the report. Expedia, full real data, date split (valid 2014-09-01, test 2014-11-01 to 2015-01-01). "
     "Each test user has one fixed top-20 from two prediction steps (`docs/eval_protocol.md`); Recall@K and NDCG@K are set-based per user, "
     "then averaged over users. SMLP4Rec rows use seed 2024 only; ItemKNN and popularity are deterministic. No seed averages, no ±.", "",
     f"Test users {info['users']:,} (warm {info['warm_users']:,}, cold {info['cold_users']:,}); "
     f"{info['users_with_one_test_booking']:,} ({info['users_with_one_test_booking'] / info['users']:.1%}) have one test booking; "
     f"{info['mean_distinct_clusters_per_user']:.2f} distinct clusters per user on average.", ""]
for us in ("all", "warm", "cold"):
    L += [f"## Test metrics, {us} users", "", "| Model | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    L += [f"| {LABEL[k]} | " + " | ".join(f"{res[k][us][c]:.4f}" for c in COLS) + " |" for k in LABEL]
    L.append("")
for name, title in (("sameDest gain (main - query embedding)", "sameDest gain (main minus + query embedding)"),
                    ("query embedding gain (query embedding - plain)", "Query embedding gain (+ query embedding minus plain)")):
    L += [f"## {title}, paired bootstrap, 95 % CI", "", "| User set | users | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
    for us, c in b[name].items():
        L.append(f"| {us} | {c['users']:,} | " + " | ".join(
            f"{c[m]['diff']:+.4f} [{c[m]['ci95'][0]:+.4f}, {c[m]['ci95'][1]:+.4f}]" for m in COLS) + " |")
    L.append("")
L += ["Bootstrap: 1000 draws, users resampled, same draw for both models. A difference is significant when its interval excludes 0.", "",
      "## Superseded (do not quote in the report)", "",
      "- Row-level test tables: `reported_tables.md`, `final_tables.md`, `final_benchmark.csv`, `protocol_gap.csv`, `bootstrap_ci.csv`, `results_slices.json`, `slices_*.csv`, `reports/benchmark_results.csv` (autoregressive / teacher-forced, per booking; still valid for the valid window and for history in notebooks 01-05).",
      "- `user_top20_reranker_choice.md`: the three rerankers side by side, the record behind the choice of m2_merge.",
      "- Mean ± std over 3 seeds (`final_tables.md`) and `plan_oct_12_checkpoints.md`: historical.",
      "- Slides `reports/summary/week6_retrain/slides_week6_vi.html` still show the row-level numbers."]
(RES / "report_numbers.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
print("wrote", RES / "report_numbers.md")
