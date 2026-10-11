"""Prototype: pick curated test bookings for the final-presentation demo (stdlib only).

Talks to a running demo server (scripts/demo_ui.py or scripts/demo_prototypes/serve_es.py) and
scans test-split bookings drawn with fixed seeds 0..N-1 through /api/seed, so every pick is
reproducible: re-enter the printed seed and row mode in the demo page to get the same booking.
Nothing is written; the output is printed.

Categories:
  cold      first booking (L = 0); main model has the actual cluster in its top 3, ItemKNN
            (popularity fallback) does not have it in its top 5
  known     warm, destination already in the user's history (sameDest active); main rank 1
  family12  children >= 1 and check-in month 12; main model top-5 hit
  miss      honest failure: main rank > 10 while ItemKNN ranks the actual cluster first
For each pick a what-if is run: party -> solo (1 adult, 0 children, 1 room), party -> family
(2 adults, 2 children), check-in month -> 12 and -> 7; the top 5 and its overlap with the
original top 5 are printed.

Usage (any Python 3.11, server already running):
    python scripts/demo_prototypes/find_curated_examples.py --port 8766 --n-warm 3000 --n-cold 1500
"""

from __future__ import annotations

import argparse
import json
import urllib.request

WHAT_IF = {
    "solo": {"adults": 1, "children": 0, "rooms": 1},
    "family": {"adults": 2, "children": 2, "rooms": 1},
    "month12": {"ci_month": 12},
    "month7": {"ci_month": 7},
}


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), method="POST"
    )
    out = json.loads(urllib.request.urlopen(req, timeout=60).read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def recommend(port: int, s: dict, fields: dict | None = None) -> dict:
    return post(
        port,
        "/api/recommend",
        {
            "user_id": s["user_id"],
            "event": s["event"],
            "truth": s["truth"],
            "model_seed": 2022,
            "fields": fields or s["fields"],
        },
    )


def top(r: dict) -> list[int]:
    return [t["cluster"] for t in r["top"]]


def describe(port: int, mode: str, s: dict, r: dict) -> None:
    f = s["fields"]
    print(
        f"  seed={s['seed']} rows={mode} user={s['user_id']} event={s['event']} L={s['n_prior']} "
        f"dest={f['dest']} month={f['ci_month']} adults={f['adults']} children={f['children']} "
        f"rooms={f['rooms']} package={f['package']} actual={s['truth']}"
    )
    b = r["baselines"]
    print(
        f"    main top5={top(r)} rank={r['truth']['rank']} known_dest={r['known_destination']} "
        f"w_sd={r['w_samedest']} | ItemKNN top5={b['ItemKNN']['top']} rank={b['ItemKNN']['truth_rank']} "
        f"| LR top5={b['Logistic regression']['top']} rank={b['Logistic regression']['truth_rank']}"
    )
    print(
        "    main rows (cluster, final, model_p, sameDest share): "
        + ", ".join(
            f"({t['cluster']}, {t['score']}, {t['model_p']}, {t['samedest_share']})"
            for t in r["top"]
        )
    )
    print(
        "    history clusters (oldest first): "
        + ", ".join(f"{h['cluster']}@d{h['dest']}" for h in r["history"])
    )
    base = set(top(r))
    for name, patch in WHAT_IF.items():
        w = recommend(port, s, {**f, **patch})
        print(
            f"    what-if {name:8s} top5={top(w)} overlap={len(base & set(top(w)))}/5 actual rank={w['truth']['rank']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--n-warm", type=int, default=3000)
    ap.add_argument("--n-cold", type=int, default=1500)
    ap.add_argument("--per-cat", type=int, default=3)
    a = ap.parse_args()
    found: dict[str, list] = {"cold": [], "known": [], "family12": [], "miss": []}
    stats = {"warm": 0, "cold": 0, "main_hit5": 0, "knn_hit5": 0, "lr_hit5": 0}
    for mode, n in (("cold", a.n_cold), ("warm", a.n_warm)):
        for seed in range(n):
            s = post(a.port, "/api/seed", {"mode": mode, "seed": seed})
            r = recommend(a.port, s)
            stats[mode] += 1
            rk, b = r["truth"]["rank"], r["baselines"]
            knn, lr = b["ItemKNN"]["truth_rank"], b["Logistic regression"]["truth_rank"]
            stats["main_hit5"] += rk <= 5
            stats["knn_hit5"] += knn <= 5
            stats["lr_hit5"] += lr <= 5
            f = s["fields"]
            if mode == "cold" and rk <= 3 and knn > 5:
                found["cold"].append((rk, -lr, seed, mode, s, r))
            if mode == "warm" and r["known_destination"] and rk == 1:
                found["known"].append((-r["top"][0]["samedest_share"], seed, mode, s, r))
            if f["children"] >= 1 and f["ci_month"] == 12 and rk <= 5:
                found["family12"].append((rk, seed, mode, s, r))
            if mode == "warm" and rk > 10 and knn == 1:
                found["miss"].append((-rk, seed, mode, s, r))
    n = stats["warm"] + stats["cold"]
    print(
        f"scanned {stats['cold']} cold + {stats['warm']} warm test bookings (seeds 0..N-1); "
        f"Recall@5 on this sample: main {stats['main_hit5'] / n:.3f}, ItemKNN {stats['knn_hit5'] / n:.3f}, "
        f"LR {stats['lr_hit5'] / n:.3f} (cold and warm pooled; not a reported metric)"
    )
    for cat, items in found.items():
        items.sort(key=lambda t: t[:-3])
        print(f"\n== {cat}: {len(items)} candidates, showing {min(a.per_cat, len(items))}")
        for t in items[: a.per_cat]:
            describe(a.port, t[-3], t[-2], t[-1])


if __name__ == "__main__":
    main()
