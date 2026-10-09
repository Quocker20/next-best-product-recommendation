"""EDA of the cleaned Expedia bookings (burst repeats collapsed). Plan: docs/eda_strategy_clean.md.

Inputs : data/interim/expedia_bookings.parquet (collapsed), expedia_bookings_flagged.parquet,
         expedia_query_codes_clean.npz, results/week4_rebuild/bookings_counts.json
Outputs: results/week4_rebuild/eda_clean.json (every number), results/figures/clean/{en,vi}/eda_e1..e8.png,
         the generated section of docs/data_cards/expedia.md (between the eda_clean markers)

Event index = row position of the collapsed table (sorted by time, user, source row). Splits follow
docs/eval_protocol.md: global temporal 80/10/10 over next-booking targets; cold rows = first bookings
inside the valid / test windows. History is the user's full earlier record (the cap of 20 used by the
models is ignored here; it affects 12 % of the warm test rows, see E2).

Usage: .venv/Scripts/python.exe scripts/expedia_eda_clean.py
"""

from __future__ import annotations

import json
import re
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbp.data.query_features import QUERY_FIELDS
from nbp.paths import INTERIM, ROOT

RES = ROOT / "results" / "week4_rebuild"
FIG = ROOT / "results" / "figures" / "clean"
CARD = ROOT / "docs" / "data_cards" / "expedia.md"
N_CLUSTERS = 100
SPLIT = (0.8, 0.1, 0.1)
ALPHA = 5.0  # smoothing of p(cluster | field value) toward the marginal (as the destination prior, m = 5)
MIN_COUNTS = (1, 2, 5, 10, 20)
BUCKETS = [("L=1", 1, 1), ("L=2-4", 2, 4), ("L=5-9", 5, 9), ("L>=10", 10, 10**9)]
LABELS = {  # key -> (en, vi)
    "e1_title": (
        "Cleaning impact: gap between a burst repeat and the booking it repeats",
        "Tác động làm sạch: khoảng cách giữa booking lặp (burst) và booking gốc",
    ),
    "gap_s": ("gap (seconds, log scale)", "khoảng cách (giây, thang log)"),
    "n_rows": ("rows", "số dòng"),
    "tiers": ("rows affected per cleaning rule", "số dòng theo từng quy tắc làm sạch"),
    "e2_hist": ("Bookings per user", "Số booking mỗi user"),
    "users": ("users (log)", "số user (log)"),
    "bookings": ("bookings per user (log)", "booking mỗi user (log)"),
    "e2_L": ("History length of warm test rows", "Độ dài lịch sử của dòng warm test"),
    "share": ("share of rows", "tỉ lệ dòng"),
    "e3_title": ("Target: cluster share, train vs test", "Mục tiêu: tỉ lệ cluster, train và test"),
    "rank": ("cluster rank by train share", "hạng cluster theo tỉ lệ train"),
    "train": ("train targets", "target train"),
    "test": ("test targets", "target test"),
    "e4_cum": (
        "Cumulative share of bookings by destination rank",
        "Tỉ lệ tích lũy booking theo hạng destination",
    ),
    "dest_rank": ("destination rank (log)", "hạng destination (log)"),
    "cum_share": ("cumulative share of bookings", "tỉ lệ booking tích lũy"),
    "e4_oov": (
        "Test targets whose destination is rare in train",
        "Target test có destination hiếm trong train",
    ),
    "min_count": (
        "train bookings of the destination below m",
        "số booking train của destination nhỏ hơn m",
    ),
    "e5_title": (
        "Repeats inside the history of the target's user (warm test rows)",
        "Lặp lại trong lịch sử của user (dòng warm test)",
    ),
    "cluster_in_hist": ("cluster already booked", "cluster đã từng đặt"),
    "dest_in_hist": ("destination already searched", "destination đã từng tìm"),
    "pair_in_hist": ("same cluster at the same destination", "cùng cluster tại cùng destination"),
    "e6_title": (
        "Held-out information about the cluster (bits, valid)",
        "Thông tin về cluster trên valid (bit)",
    ),
    "bits": ("information gain (bits, log scale)", "lượng thông tin (bit, thang log)"),
    "e7_party": ("Party type", "Loại nhóm khách"),
    "e7_month": ("Check-in month", "Tháng check-in"),
    "valid": ("valid targets", "target valid"),
    "e8_vol": ("Bookings and new users per month", "Booking và user mới theo tháng"),
    "events": ("bookings", "booking"),
    "new_users": ("new users (first booking)", "user mới (booking đầu)"),
    "e8_family": ("Family share of bookings per month", "Tỉ lệ nhóm gia đình theo tháng"),
    "family": ("family share", "tỉ lệ gia đình"),
}


def lab(key: str, lang: str) -> str:
    return LABELS[key][0 if lang == "en" else 1]


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two nonnegative vectors (normalised inside)."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())

    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 5)


def share_vec(values: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(values, minlength=size).astype(np.float64)


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    return round(float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())), 4)


def quantiles(x: np.ndarray, qs=(0.01, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    return {str(q): round(float(np.quantile(x, q)), 3) for q in qs}


def info_gain(values: np.ndarray, y: np.ndarray, train: np.ndarray, valid: np.ndarray) -> float:
    """Bits gained on valid by predicting the cluster from `values` instead of the train marginal."""
    p_y = share_vec(y[train], N_CLUSTERS)
    p_y = p_y / p_y.sum()
    ce_marg = float(-np.log2(p_y[y[valid]]).mean())
    vals, inv = np.unique(values, return_inverse=True)
    table = np.bincount(
        inv[train] * N_CLUSTERS + y[train], minlength=len(vals) * N_CLUSTERS
    ).reshape(len(vals), N_CLUSTERS)
    n_a = table.sum(1)
    cond = (table[inv[valid], y[valid]] + ALPHA * p_y[y[valid]]) / (n_a[inv[valid]] + ALPHA)
    return round(ce_marg - float(-np.log2(cond).mean()), 5)


def main() -> None:
    cols = [
        "user_id", "item_id", "timestamp", "ts_unix", "ctx_dest_id", "ctx_dest_type", "ctx_ci",
        "ctx_lead_days", "ctx_stay_nights", "ctx_checkin_month", "ctx_party_type", "ctx_is_package",
        "ctx_channel", "ctx_is_mobile", "ctx_user_region", "ctx_user_city",
    ]  # fmt: skip
    d = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=cols)
    n = len(d)
    uid = d["user_id"].to_numpy().astype("int64")
    y = d["item_id"].to_numpy().astype("int64")
    dest = d["ctx_dest_id"].to_numpy().astype("int64")
    sec = d["ts_unix"].to_numpy().astype("int64")
    ts = d["timestamp"]
    pos = d.groupby("user_id", sort=False).cumcount().to_numpy()
    res: dict = {"events": n}

    # ---- splits (event index = row position)
    tgt = np.flatnonzero(pos > 0)
    n_tr, n_va = int(SPLIT[0] * len(tgt)), int(SPLIT[1] * len(tgt))
    parts = {"train": tgt[:n_tr], "valid": tgt[n_tr : n_tr + n_va], "test": tgt[n_tr + n_va :]}
    first = np.flatnonzero(pos == 0)
    last_tr, last_va = parts["train"].max(), parts["valid"].max()
    parts["cold_valid"] = first[(first > last_tr) & (first < last_va)]
    parts["cold_test"] = first[first > last_va]
    parts["first_train"] = first[first < last_tr]
    res["splits"] = {
        k: {
            "rows": len(v),
            "first_date": str(ts.iloc[v.min()].date()),
            "last_date": str(ts.iloc[v.max()].date()),
        }
        for k, v in parts.items()
    }
    train, valid, test = parts["train"], parts["valid"], parts["test"]

    # ---- E1 cleaning impact
    counts = json.loads((RES / "bookings_counts.json").read_text(encoding="utf-8"))
    before, after = counts["flagged_stats_before_collapse"], counts["clean_stats"]
    res["E1"] = {
        "before_after": {
            k: {"before": before[k], "after": after[k]}
            for k in (
                "rows",
                "users",
                "unique_user_item_pairs",
                "single_booking_users",
                "single_booking_user_share",
            )
        },
        "party_type_counts": {
            "before": before["party_type_counts"],
            "after": after["party_type_counts"],
        },
        "rows_per_rule": {
            k: v
            for k, v in counts["cleaning_steps"].items()
            if k.startswith(("flag_", "duplicate"))
        },
    }
    fl = pd.read_parquet(
        INTERIM / "expedia_bookings_flagged.parquet",
        columns=["user_id", "timestamp", "src_row", "flag_burst_repeat"],
    ).sort_values(["user_id", "timestamp", "src_row"], kind="stable")
    gap = fl.groupby("user_id", sort=False)["timestamp"].diff().dt.total_seconds()
    burst_gap = gap[fl["flag_burst_repeat"]].to_numpy()
    edges = np.logspace(0, 7, 29)
    hist, _ = np.histogram(np.clip(burst_gap, 1, 1e7), bins=edges)
    res["E1"]["burst_gap_seconds"] = {
        "n": len(burst_gap),
        "quantiles": quantiles(burst_gap),
        "share_within_1h": round(float((burst_gap <= 3600).mean()), 4),
        "share_within_1d": round(float((burst_gap <= 86400).mean()), 4),
        "hist_edges": [round(float(e), 1) for e in edges],
        "hist_counts": [int(h) for h in hist],
    }
    del fl

    # ---- E2 users and histories
    per_user = d.groupby("user_id", sort=False).size().to_numpy()
    pairs = int(pd.Series(uid * 128 + y).nunique())
    gap_prev = pd.Series(sec).groupby(uid, sort=False).diff().to_numpy("float64")
    nonfirst = pos > 0
    L_test = pos[test]
    res["E2"] = {
        "users": len(per_user),
        "bookings_per_user_quantiles": quantiles(per_user, (0.5, 0.75, 0.9, 0.99, 0.999)),
        "share_single_booking": round(float((per_user == 1).mean()), 4),
        "share_5_plus": round(float((per_user >= 5).mean()), 4),
        "share_10_plus": round(float((per_user >= 10).mean()), 4),
        "user_cluster_density": round(pairs / (len(per_user) * N_CLUSTERS), 4),
        "days_between_bookings_quantiles": quantiles(gap_prev[nonfirst] / 86400),
        "share_next_booking_within_1h": round(float((gap_prev[nonfirst] <= 3600).mean()), 4),
        "share_next_booking_within_1d": round(float((gap_prev[nonfirst] <= 86400).mean()), 4),
        "warm_test_L_buckets": {
            b: round(float(((L_test >= lo) & (L_test <= hi)).mean()), 4) for b, lo, hi in BUCKETS
        },
        "warm_test_history_capped_at_20_share": round(float((L_test > 20).mean()), 4),
        "bookings_per_user_hist": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(np.minimum(per_user, 60), return_counts=True))
        },
    }

    # ---- E3 target
    c_all = share_vec(y, N_CLUSTERS)
    c_tr, c_te = share_vec(y[train], N_CLUSTERS), share_vec(y[test], N_CLUSTERS)
    p_all = c_all / c_all.sum()
    year = ts.dt.year.to_numpy()
    res["E3"] = {
        "top1_share": round(float(np.sort(p_all)[-1]), 4),
        "top5_share": round(float(np.sort(p_all)[-5:].sum()), 4),
        "top10_share": round(float(np.sort(p_all)[-10:].sum()), 4),
        "min_share": round(float(p_all.min()), 4),
        "entropy_normalised": round(float(-(p_all * np.log(p_all)).sum() / np.log(N_CLUSTERS)), 4),
        "gini": gini(c_all),
        "jsd_train_test_bits": jsd(c_tr, c_te),
        "jsd_2013_2014_bits": jsd(
            share_vec(y[year == 2013], N_CLUSTERS), share_vec(y[year == 2014], N_CLUSTERS)
        ),
    }
    rank_order = np.argsort(-c_tr)  # clusters by train share
    res["E3"]["train_share_by_train_rank"] = [
        round(float(v), 5) for v in (c_tr / c_tr.sum())[rank_order]
    ]
    res["E3"]["test_share_by_train_rank"] = [
        round(float(v), 5) for v in (c_te / c_te.sum())[rank_order]
    ]

    # ---- E4 destinations
    dc = np.bincount(dest)
    dc = dc[dc > 0]
    order = np.sort(dc)[::-1]
    cum = np.cumsum(order) / order.sum()
    train_dest = np.bincount(dest[train], minlength=dest.max() + 1)
    oov = {str(m): round(float((train_dest[dest[test]] < m).mean()), 4) for m in MIN_COUNTS}
    dest_seen = train_dest[dest[test]] >= 20
    tab = np.zeros((dest.max() + 1, N_CLUSTERS), np.int64)
    np.add.at(tab, (dest[train], y[train]), 1)
    big = np.flatnonzero(tab.sum(1) >= 20)
    pb = tab[big] / tab[big].sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pb > 0, pb * np.log2(pb), 0.0), axis=1)
    w = tab[big].sum(1) / tab[big].sum()
    res["E4"] = {
        "destinations": len(dc),
        "single_booking_destinations": int((dc == 1).sum()),
        "top10_share": round(float(order[:10].sum() / order.sum()), 4),
        "top100_share": round(float(order[:100].sum() / order.sum()), 4),
        "top1000_share": round(float(order[:1000].sum() / order.sum()), 4),
        "cum_share_at_rank": {
            str(r): round(float(cum[r - 1]), 4) for r in (1, 10, 100, 1000, 10000) if r <= len(cum)
        },
        "cum_curve": [
            round(float(c), 5)
            for c in cum[np.unique(np.logspace(0, np.log10(len(cum)), 80).astype(int)) - 1]
        ],
        "cum_curve_ranks": [
            int(r) for r in np.unique(np.logspace(0, np.log10(len(cum)), 80).astype(int))
        ],
        "test_targets_with_train_count_below": oov,
        "dest_with_ge20_train_bookings": len(big),
        "test_targets_with_ge20_train_bookings": round(float(dest_seen.mean()), 4),
        "mean_cluster_entropy_bits_given_destination_ge20": round(float((w * ent).sum()), 3),
        "cluster_entropy_bits_marginal": round(float(-(p_all * np.log2(p_all)).sum()), 3),
        "mean_top1_cluster_share_given_destination_ge20": round(float((w * pb.max(1)).sum()), 4),
    }

    # ---- E5 repeats in the history (full history)
    def seen_before(key: np.ndarray) -> np.ndarray:
        return pd.Series(key).groupby(key, sort=False).cumcount().to_numpy() > 0

    s_cl, s_de, s_pair = (
        seen_before(uid * 128 + y),
        seen_before(uid * 10**6 + dest),
        seen_before((uid * 10**6 + dest) * 128 + y),
    )
    res["E5"] = {}
    for name, idx in (("train", train), ("test", test)):
        r = {
            "all_warm": {
                "cluster_in_history": round(float(s_cl[idx].mean()), 4),
                "destination_in_history": round(float(s_de[idx].mean()), 4),
                "pair_in_history": round(float(s_pair[idx].mean()), 4),
                "rows": len(idx),
            }
        }
        for b, lo, hi in BUCKETS:
            m = (pos[idx] >= lo) & (pos[idx] <= hi)
            r[b] = {
                "rows": int(m.sum()),
                "cluster_in_history": round(float(s_cl[idx][m].mean()), 4),
                "destination_in_history": round(float(s_de[idx][m].mean()), 4),
                "pair_in_history": round(float(s_pair[idx][m].mean()), 4),
            }
        res["E5"][name] = r
    res["E5"]["test_pair_given_destination_in_history"] = round(
        float(s_pair[test][s_de[test]].mean()), 4
    )

    # ---- E6 information in the query fields
    z = np.load(INTERIM / "expedia_query_codes_clean.npz")
    codes, fields = z["codes"], list(z["fields"])
    assert fields == list(QUERY_FIELDS) and len(codes) == n
    gains = {
        f: info_gain(codes[:, j].astype("int64"), y, train, valid) for j, f in enumerate(fields)
    }
    cand = {
        "checkin_weekday": d["ctx_ci"].dt.dayofweek.to_numpy().astype("int64"),
        "user_region (candidate)": d["ctx_user_region"].to_numpy().astype("int64"),
        "user_city (candidate)": d["ctx_user_city"].to_numpy().astype("int64"),
    }
    gains.update({k: info_gain(v, y, train, valid) for k, v in cand.items()})
    res["E6"] = {
        "held_out_information_gain_bits_on_valid": dict(
            sorted(gains.items(), key=lambda kv: -kv[1])
        ),
        "cluster_marginal_entropy_bits": res["E4"]["cluster_entropy_bits_marginal"],
        "note": f"p(cluster | value) smoothed toward the train marginal with alpha {ALPHA}; fit on train targets, scored on valid targets",
    }

    # ---- E7 composition of the splits
    month_ci = d["ctx_checkin_month"].astype("float64").fillna(0).to_numpy().astype("int64")
    party = d["ctx_party_type"].astype(str).to_numpy()
    party_names = ["solo", "couple", "family", "group", "unknown"]
    comp: dict = {}
    for name, idx in (("train", train), ("valid", valid), ("test", test)):
        comp[name] = {
            "party": {p: round(float((party[idx] == p).mean()), 4) for p in party_names},
            "package_share": round(float(d["ctx_is_package"].to_numpy()[idx].mean()), 4),
            "mobile_share": round(float(d["ctx_is_mobile"].to_numpy()[idx].mean()), 4),
            "destination_in_history_share": round(float(s_de[idx].mean()), 4),
            "mean_history_length": round(float(pos[idx].mean()), 3),
            "checkin_month": {
                str(m): round(float((month_ci[idx] == m).mean()), 4) for m in range(1, 13)
            },
        }
    jfields = {}
    for j, f in enumerate(fields):
        if f in ("dest", "country"):
            continue
        size = int(codes[:, j].max()) + 1
        jfields[f] = jsd(
            share_vec(codes[train, j].astype("int64"), size),
            share_vec(codes[test, j].astype("int64"), size),
        )
    res["E7"] = {
        "composition": comp,
        "jsd_train_vs_test_bits_per_field": jfields,
        "cold_test_share_of_all_test_events": round(
            float(len(parts["cold_test"]) / (len(parts["cold_test"]) + len(test))), 4
        ),
        "cold_valid_share_of_all_valid_events": round(
            float(len(parts["cold_valid"]) / (len(parts["cold_valid"]) + len(valid))), 4
        ),
    }

    # ---- E8 time structure
    ym = ts.dt.to_period("M").astype(str).to_numpy()
    months = sorted(set(ym))
    new_users = pos == 0
    fam = d["ctx_party_type"].astype(str).to_numpy() == "family"
    res["E8"] = {
        "monthly": {
            m: {
                "bookings": int((ym == m).sum()),
                "new_users": int(((ym == m) & new_users).sum()),
                "active_users": len(np.unique(uid[ym == m])),
                "family_share": round(float(fam[ym == m].mean()), 4),
            }
            for m in months
        },
        "bookings_2013": int((year == 2013).sum()),
        "bookings_2014": int((year == 2014).sum()),
        "party_share_by_year": {
            str(yr): {p: round(float((party[year == yr] == p).mean()), 4) for p in party_names}
            for yr in (2013, 2014)
        },
        "lead_days_quantiles": quantiles(d["ctx_lead_days"].astype("float64").dropna().to_numpy()),
        "stay_nights_quantiles": quantiles(
            d["ctx_stay_nights"].astype("float64").dropna().to_numpy()
        ),
        "checkin_month_share_all": {
            str(m): round(float((month_ci == m).mean()), 4) for m in range(1, 13)
        },
    }

    (RES / "eda_clean.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    make_figures(res, months)
    write_card(res)
    print("wrote", RES / "eda_clean.json", "and", FIG)


def make_figures(res: dict, months: list[str]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for lang in ("en", "vi"):
        out = FIG / lang
        out.mkdir(parents=True, exist_ok=True)
        L = partial(lab, lang=lang)

        # E1
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        e = res["E1"]["burst_gap_seconds"]
        edges = np.array(e["hist_edges"])
        ax[0].bar(edges[:-1], e["hist_counts"], width=np.diff(edges), align="edge", color="#3b6ea5")
        ax[0].set_xscale("log")
        ax[0].set_xlabel(L("gap_s"))
        ax[0].set_ylabel(L("n_rows"))
        ax[0].set_title(L("e1_title"), fontsize=8)
        rules = res["E1"]["rows_per_rule"]
        names = list(rules)
        ax[1].barh(names, [rules[k] for k in names], color="#7a9e7e")
        ax[1].set_xscale("log")
        ax[1].set_xlabel(L("tiers"))
        fig.tight_layout()
        fig.savefig(out / "eda_e1_cleaning.png", dpi=140)
        plt.close(fig)

        # E2
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        h = res["E2"]["bookings_per_user_hist"]
        ax[0].loglog([int(k) for k in h], list(h.values()), "o", ms=3, color="#3b6ea5")
        ax[0].set_xlabel(L("bookings"))
        ax[0].set_ylabel(L("users"))
        ax[0].set_title(L("e2_hist"))
        b = res["E2"]["warm_test_L_buckets"]
        ax[1].bar(list(b), list(b.values()), color="#7a9e7e")
        ax[1].set_ylabel(L("share"))
        ax[1].set_title(L("e2_L"))
        fig.tight_layout()
        fig.savefig(out / "eda_e2_users.png", dpi=140)
        plt.close(fig)

        # E3 (needs cluster shares: recompute from the JSON-free source is avoided; use stored top shares only)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        s = res["E3"]
        ax.plot(range(1, 101), s["train_share_by_train_rank"], color="#3b6ea5", label=L("train"))
        ax.plot(range(1, 101), s["test_share_by_train_rank"], color="#c27c3f", label=L("test"))
        ax.set_xlabel(L("rank"))
        ax.set_ylabel(L("share"))
        ax.legend(fontsize=7)
        ax.set_title(
            f"{L('e3_title')}\nJSD train/test = {s['jsd_train_test_bits']} bits, 2013/2014 = {s['jsd_2013_2014_bits']} bits",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out / "eda_e3_target.png", dpi=140)
        plt.close(fig)

        # E4
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        e4 = res["E4"]
        ax[0].semilogx(e4["cum_curve_ranks"], e4["cum_curve"], color="#3b6ea5")
        ax[0].set_xlabel(L("dest_rank"))
        ax[0].set_ylabel(L("cum_share"))
        ax[0].set_title(L("e4_cum"), fontsize=8)
        o = e4["test_targets_with_train_count_below"]
        ax[1].bar(list(o), list(o.values()), color="#c27c3f")
        ax[1].set_xlabel(L("min_count"))
        ax[1].set_ylabel(L("share"))
        ax[1].set_title(L("e4_oov"), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "eda_e4_destinations.png", dpi=140)
        plt.close(fig)

        # E5
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bk = ["all_warm"] + [b[0] for b in BUCKETS]
        t = res["E5"]["test"]
        xs = np.arange(len(bk))
        for i, (key, color) in enumerate(
            (
                ("cluster_in_hist", "#3b6ea5"),
                ("dest_in_hist", "#7a9e7e"),
                ("pair_in_hist", "#c27c3f"),
            )
        ):
            src = {
                "cluster_in_hist": "cluster_in_history",
                "dest_in_hist": "destination_in_history",
                "pair_in_hist": "pair_in_history",
            }[key]
            ax.bar(
                xs + (i - 1) * 0.27, [t[b][src] for b in bk], width=0.27, label=L(key), color=color
            )
        ax.set_xticks(xs, bk)
        ax.set_ylabel(L("share"))
        ax.set_title(L("e5_title"), fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "eda_e5_repeats.png", dpi=140)
        plt.close(fig)

        # E6
        fig, ax = plt.subplots(figsize=(7, 4.2))
        g = res["E6"]["held_out_information_gain_bits_on_valid"]
        names = list(g)[::-1]
        ax.barh(
            names,
            [max(g[k], 1e-4) for k in names],
            color=["#3b6ea5" if g[k] > 0 else "#b0b0b0" for k in names],
        )
        for i, k in enumerate(names):
            if g[k] <= 0:  # no held-out gain: drawn at the axis floor and labelled
                ax.text(1.3e-4, i, f"{g[k]:.3f} (<= 0)", va="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(L("bits"))
        ax.set_title(L("e6_title"))
        fig.tight_layout()
        fig.savefig(out / "eda_e6_fields.png", dpi=140)
        plt.close(fig)

        # E7
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        comp = res["E7"]["composition"]
        party = list(comp["train"]["party"])
        xs = np.arange(len(party))
        for i, sp in enumerate(("train", "valid", "test")):
            ax[0].bar(
                xs + (i - 1) * 0.27,
                [comp[sp]["party"][p] for p in party],
                width=0.27,
                label=L(sp if sp != "valid" else "valid"),
            )
        ax[0].set_xticks(xs, party)
        ax[0].set_title(L("e7_party"))
        ax[0].legend(fontsize=7)
        for sp in ("train", "valid", "test"):
            ax[1].plot(
                range(1, 13),
                [comp[sp]["checkin_month"][str(m)] for m in range(1, 13)],
                marker="o",
                ms=3,
                label=L(sp),
            )
        ax[1].set_title(L("e7_month"))
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "eda_e7_split_composition.png", dpi=140)
        plt.close(fig)

        # E8
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        mo = res["E8"]["monthly"]
        ax[0].plot(months, [mo[m]["bookings"] for m in months], label=L("events"))
        ax[0].plot(months, [mo[m]["new_users"] for m in months], label=L("new_users"))
        ax[0].set_xticks(months[::3], months[::3], rotation=45)
        ax[0].set_title(L("e8_vol"))
        ax[0].legend(fontsize=7)
        ax[1].plot(months, [mo[m]["family_share"] for m in months], color="#c27c3f")
        ax[1].set_xticks(months[::3], months[::3], rotation=45)
        ax[1].set_title(L("e8_family"))
        ax[1].set_ylabel(L("family"))
        fig.tight_layout()
        fig.savefig(out / "eda_e8_time.png", dpi=140)
        plt.close(fig)


def write_card(res: dict) -> None:
    e1, e2, e3, e4, e5 = res["E1"], res["E2"], res["E3"], res["E4"], res["E5"]
    ba = e1["before_after"]
    sp = res["splits"]
    lines = [
        "<!-- eda_clean:start (generated by scripts/expedia_eda_clean.py; do not edit by hand) -->",
        "## Cleaned data (burst repeats collapsed, 2026-10-09)",
        "",
        "Source: `data/interim/expedia_bookings.parquet`; strategy `docs/data_cleaning_strategy.md`; every number below is in `results/week4_rebuild/eda_clean.json`. The sections above describe the raw file.",
        "",
        f"- Bookings {ba['rows']['after']:,} (raw after dedup {ba['rows']['before']:,}); users {ba['users']['after']:,}; unique user-cluster pairs {ba['unique_user_item_pairs']['after']:,}; single-booking users {ba['single_booking_users']['after']:,} ({ba['single_booking_user_share']['after']:.2%}, before {ba['single_booking_user_share']['before']:.2%}).",
        f"- Bookings per user: median {e2['bookings_per_user_quantiles']['0.5']}, 90th percentile {e2['bookings_per_user_quantiles']['0.9']}, 99th percentile {e2['bookings_per_user_quantiles']['0.99']}; 5+ bookings {e2['share_5_plus']:.2%}, 10+ {e2['share_10_plus']:.2%}; user x cluster density {e2['user_cluster_density']:.2%}.",
        f"- Burst repeats: {e1['burst_gap_seconds']['n']:,} rows, median gap {e1['burst_gap_seconds']['quantiles']['0.5']:.0f} s, {e1['burst_gap_seconds']['share_within_1h']:.1%} within one hour.",
        f"- Target: top-1 cluster {e3['top1_share']:.2%}, top-10 {e3['top10_share']:.2%}, normalised entropy {e3['entropy_normalised']}, Gini {e3['gini']}; JSD train vs test {e3['jsd_train_test_bits']} bits, 2013 vs 2014 {e3['jsd_2013_2014_bits']} bits.",
        f"- Destinations: {e4['destinations']:,} (single-booking {e4['single_booking_destinations']:,}); top-100 destinations hold {e4['top100_share']:.2%} of bookings; {e4['test_targets_with_train_count_below']['5']:.2%} of test targets have a destination with fewer than 5 train bookings.",
        f"- Splits (next-booking targets, 80/10/10): train {sp['train']['rows']:,} ({sp['train']['first_date']} to {sp['train']['last_date']}), valid {sp['valid']['rows']:,} (to {sp['valid']['last_date']}), test {sp['test']['rows']:,} (to {sp['test']['last_date']}); cold rows valid {sp['cold_valid']['rows']:,}, test {sp['cold_test']['rows']:,}.",
        f"- Repeats in the history (warm test rows): target cluster already booked {e5['test']['all_warm']['cluster_in_history']:.2%}, destination already searched {e5['test']['all_warm']['destination_in_history']:.2%}, same cluster at the same destination {e5['test']['all_warm']['pair_in_history']:.2%}.",
        "- Figures: `results/figures/clean/{en,vi}/eda_e1..e8`.",
        "<!-- eda_clean:end -->",
    ]
    text = CARD.read_text(encoding="utf-8")
    block = "\n".join(lines)
    if "<!-- eda_clean:start" in text:
        text = re.sub(
            r"<!-- eda_clean:start.*?<!-- eda_clean:end -->", lambda m: block, text, flags=re.DOTALL
        )
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    CARD.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
