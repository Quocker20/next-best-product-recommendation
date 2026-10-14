"""EDA of the cleaned Expedia bookings on the week-6 date split (train / valid / test windows).

Inputs : data/interim/expedia_bookings.parquet (cleaned, burst repeats collapsed, time-sorted),
         data/processed/expedia_date_split/rows.npz (window, L, h, codes; row i = booking i),
         configs/split.yaml
Outputs: results/week6_retrain/eda_split.json (every number),
         results/figures/split/{en,vi}/eda_s1..s9.png

Sections: S1 split composition and volume, S2 users and histories, S3 target (cluster) shift,
S4 trip context shift, S5 seasonality, S6 destinations, S7 repeats (pre-window vs in-window),
S8 information in the query fields, S9 slice sizes for RQ1.
A booking belongs to the window of its own time. L = earlier bookings of the user (any window),
h = earlier bookings of the user inside the same window (what autoregressive rollout replaces).

Usage: .venv/Scripts/python.exe scripts/expedia_eda_split.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from nbp.paths import CONFIGS, INTERIM, PROCESSED, ROOT

RES = ROOT / "results" / "week6_retrain"
FIG = ROOT / "results" / "figures" / "split"
N_CLUSTERS = 100
ALPHA = 5.0  # smoothing of p(cluster | value) toward the train marginal
WN = ["train", "valid", "test"]
MIN_COUNTS = (1, 2, 5, 10, 20)
L_BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 10**9)]  # fmt: skip
H_BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10**9)]
PARTY = ["solo", "couple", "family", "group", "unknown"]
COLORS = {"train": "#3b6ea5", "valid": "#d98e04", "test": "#c0392b"}


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two nonnegative count vectors."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())

    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 5)


def counts(values: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(values, minlength=size).astype(np.float64)


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    return round(float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())), 4)


def quant(x: np.ndarray, qs=(0.01, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    return {str(q): round(float(np.quantile(x, q)), 3) for q in qs}


def share(mask: np.ndarray) -> float:
    return round(float(mask.mean()), 4)


def bucket_shares(x: np.ndarray, buckets) -> dict:
    return {b: share((x >= lo) & (x <= hi)) for b, lo, hi in buckets}


def info_gain(values, y, fit, score) -> float:
    """Bits gained on `score` rows by predicting the cluster from `values` (fit on `fit` rows)."""
    p_y = counts(y[fit], N_CLUSTERS)
    p_y /= p_y.sum()
    ce_marg = float(-np.log2(p_y[y[score]]).mean())
    vals, inv = np.unique(values, return_inverse=True)
    table = np.bincount(inv[fit] * N_CLUSTERS + y[fit], minlength=len(vals) * N_CLUSTERS)
    table = table.reshape(len(vals), N_CLUSTERS)
    n_a = table.sum(1)
    cond = (table[inv[score], y[score]] + ALPHA * p_y[y[score]]) / (n_a[inv[score]] + ALPHA)
    return round(ce_marg - float(-np.log2(cond).mean()), 5)


def main() -> None:
    cfg = yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))
    cols = [
        "user_id", "item_id", "timestamp", "ts_unix", "ctx_dest_id", "ctx_dest_type", "ctx_ci",
        "ctx_lead_days", "ctx_stay_nights", "ctx_checkin_month", "ctx_adults", "ctx_children",
        "ctx_rooms", "ctx_party_type", "ctx_is_package", "ctx_channel", "ctx_is_mobile",
        "ctx_user_region", "ctx_user_city",
    ]  # fmt: skip
    d = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=cols)
    z = np.load(PROCESSED / "expedia_date_split" / "rows.npz")
    n = len(d)
    win, L, h = z["window"].astype(int), z["L"].astype("int64"), z["h"].astype("int64")
    codes, fields = z["codes"], [str(f) for f in z["fields"]]
    assert len(win) == n and len(codes) == n
    uid = z["user"].astype("int64")
    y = d["item_id"].to_numpy().astype("int64")
    dest = d["ctx_dest_id"].to_numpy().astype("int64")
    sec = d["ts_unix"].to_numpy().astype("int64")
    ts = d["timestamp"]
    party = d["ctx_party_type"].astype(str).to_numpy()
    pkg = d["ctx_is_package"].to_numpy().astype(int)
    mob = d["ctx_is_mobile"].to_numpy().astype(int)
    chan = d["ctx_channel"].to_numpy().astype("int64")
    cim = d["ctx_checkin_month"].astype("float64").fillna(0).to_numpy().astype("int64")
    lead = d["ctx_lead_days"].astype("float64").to_numpy()
    stay = d["ctx_stay_nights"].astype("float64").to_numpy()
    bm = ts.dt.month.to_numpy()
    ym = ts.dt.to_period("M").astype(str).to_numpy()
    months = sorted(set(ym))
    W = {k: np.flatnonzero(win == i) for i, k in enumerate(WN)}
    tr, va, te = W["train"], W["valid"], W["test"]
    res: dict = {"events": n, "dates": cfg}

    # ---------------- S1 split composition and volume
    first_seen = np.zeros(n, bool)
    first_seen[np.unique(uid, return_index=True)[1]] = True  # time-sorted: first index = first booking
    train_users = np.unique(uid[tr])
    train_valid_users = np.unique(uid[win <= 1])
    s1: dict = {}
    for k, idx in W.items():
        in_train = np.isin(uid[idx], train_users)
        s1[k] = {
            "bookings": len(idx),
            "share": round(len(idx) / n, 4),
            "users": len(np.unique(uid[idx])),
            "first_date": str(ts.iloc[idx.min()].date()),
            "last_date": str(ts.iloc[idx.max()].date()),
            "warm_L_ge_1": int((L[idx] >= 1).sum()),
            "cold_L_0": int((L[idx] == 0).sum()),
            "warm_share": share(L[idx] >= 1),
            "h_ge_1_share": share(h[idx] >= 1),
            "bookings_by_users_seen_in_train": share(in_train),
            "users_seen_in_train": round(
                float(np.isin(np.unique(uid[idx]), train_users).mean()), 4
            ),
            "cold_share": share(L[idx] == 0),
            "users_not_in_train": int((~np.isin(np.unique(uid[idx]), train_users)).sum()),
            "users_not_in_train_share": round(
                float((~np.isin(np.unique(uid[idx]), train_users)).mean()), 4
            ),
            "users_not_in_train_or_valid_share": round(
                float((~np.isin(np.unique(uid[idx]), train_valid_users)).mean()), 4
            ),
            "mean_bookings_per_user_in_window": round(len(idx) / len(np.unique(uid[idx])), 3),
        }
    res["S1"] = {
        "windows": s1,
        "monthly": {
            m: {
                "window": WN[int(win[ym == m][0])],
                "bookings": int((ym == m).sum()),
                "new_users": int(((ym == m) & first_seen).sum()),
                "active_users": len(np.unique(uid[ym == m])),
                "family_share": share(party[ym == m] == "family"),
                "package_share": share(pkg[ym == m] == 1),
            }
            for m in months
        },
    }

    # ---------------- S2 users and histories
    per_user = np.bincount(uid)
    per_user = per_user[per_user > 0]
    gap_prev = pd.Series(sec).groupby(uid, sort=False).diff().to_numpy("float64")
    res["S2"] = {
        "users": len(per_user),
        "bookings_per_user_quantiles": quant(per_user, (0.5, 0.75, 0.9, 0.99, 0.999)),
        "share_single_booking_users": share(per_user == 1),
        "share_users_5_plus": share(per_user >= 5),
        "share_users_10_plus": share(per_user >= 10),
        "bookings_per_user_hist": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(np.minimum(per_user, 60), return_counts=True))
        },
        "L_buckets": {k: bucket_shares(L[idx], L_BUCKETS) for k, idx in W.items()},
        "h_buckets": {k: bucket_shares(h[idx], H_BUCKETS) for k, idx in W.items()},
        "history_cap_20_share_of_rows": {k: share(L[idx] > 20) for k, idx in W.items()},
        "mean_L": {k: round(float(L[idx].mean()), 3) for k, idx in W.items()},
        "days_between_bookings": {
            k: quant(gap_prev[idx][L[idx] > 0] / 86400) for k, idx in W.items()
        },
        "share_gap_within_1h": {
            k: share(gap_prev[idx][L[idx] > 0] <= 3600) for k, idx in W.items()
        },
        "share_gap_within_1d": {
            k: share(gap_prev[idx][L[idx] > 0] <= 86400) for k, idx in W.items()
        },
        "user_cluster_density": round(
            float(pd.Series(uid * 128 + y).nunique() / (len(per_user) * N_CLUSTERS)), 4
        ),
    }

    # ---------------- S3 target (cluster) shift
    c = {k: counts(y[idx], N_CLUSTERS) for k, idx in W.items()}
    p = {k: v / v.sum() for k, v in c.items()}
    rank = np.argsort(-c["train"])
    train_rank = np.empty(N_CLUSTERS, int)
    train_rank[rank] = np.arange(N_CLUSTERS)

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return round(float(np.corrcoef(ra, rb)[0, 1]), 4)

    top_cover = {
        k: {
            str(m): share(train_rank[y[W[k]]] < m) for m in (1, 5, 10, 20, 50)
        }
        for k in ("valid", "test")
    }
    # same-season comparison: valid vs Sep-Oct 2013, test vs Nov-Dec 2013, train as reference
    s2013v = np.flatnonzero((ym >= "2013-09") & (ym <= "2013-10"))
    s2013t = np.flatnonzero((ym >= "2013-11") & (ym <= "2013-12"))
    res["S3"] = {
        "per_window": {
            k: {
                "top1_share": round(float(np.sort(p[k])[-1]), 4),
                "top5_share": round(float(np.sort(p[k])[-5:].sum()), 4),
                "top10_share": round(float(np.sort(p[k])[-10:].sum()), 4),
                "entropy_normalised": round(float(-(p[k] * np.log(p[k])).sum() / np.log(100)), 4),
                "gini": gini(c[k]),
                "min_share": round(float(p[k].min()), 5),
            }
            for k in WN
        },
        "jsd_bits": {
            "train_vs_valid": jsd(c["train"], c["valid"]),
            "train_vs_test": jsd(c["train"], c["test"]),
            "valid_vs_test": jsd(c["valid"], c["test"]),
            "valid_vs_sepoct2013": jsd(c["valid"], counts(y[s2013v], N_CLUSTERS)),
            "test_vs_novdec2013": jsd(c["test"], counts(y[s2013t], N_CLUSTERS)),
        },
        "spearman_cluster_share_train_vs": {
            "valid": spearman(c["train"], c["valid"]),
            "test": spearman(c["train"], c["test"]),
        },
        "targets_covered_by_top_m_train_clusters": top_cover,
        "monthly_jsd_vs_train_bits": {
            m: jsd(c["train"], counts(y[ym == m], N_CLUSTERS)) for m in months
        },
        "share_by_train_rank": {
            k: [round(float(v), 5) for v in p[k][rank]] for k in WN
        },
        "largest_share_changes_test_minus_train": [
            {
                "cluster": int(i),
                "train": round(float(p["train"][i]), 4),
                "test": round(float(p["test"][i]), 4),
            }
            for i in np.argsort(-np.abs(p["test"] - p["train"]))[:8]
        ],
    }

    # ---------------- S4 trip-context shift
    chan_top = [int(v) for v in np.argsort(-np.bincount(chan[tr]))[:5]]
    comp: dict = {}
    for k, idx in W.items():
        comp[k] = {
            "party": {pt: share(party[idx] == pt) for pt in PARTY},
            "package_share": share(pkg[idx] == 1),
            "mobile_share": share(mob[idx] == 1),
            "channel_top5_train": {str(ch): share(chan[idx] == ch) for ch in chan_top},
            "adults_ge3_share": share(d["ctx_adults"].to_numpy()[idx] >= 3),
            "children_ge1_share": share(d["ctx_children"].to_numpy()[idx] >= 1),
            "rooms_ge2_share": share(d["ctx_rooms"].to_numpy()[idx] >= 2),
            "lead_days_quantiles": quant(lead[idx][~np.isnan(lead[idx])]),
            "stay_nights_quantiles": quant(stay[idx][~np.isnan(stay[idx])]),
            "unknown_lead_share": share(np.isnan(lead[idx])),
        }
    jf: dict = {}
    for j, f in enumerate(fields):
        if f == "dest":
            continue
        size = int(codes[:, j].max()) + 1
        ct = counts(codes[tr, j].astype("int64"), size)
        jf[f] = {
            "train_vs_valid": jsd(ct, counts(codes[va, j].astype("int64"), size)),
            "train_vs_test": jsd(ct, counts(codes[te, j].astype("int64"), size)),
        }
    res["S4"] = {
        "composition": comp,
        "jsd_bits_per_field": jf,
        "test_warm_vs_cold": {
            name: {
                "party": {pt: share(party[idx][mask] == pt) for pt in PARTY},
                "package_share": share(pkg[idx][mask] == 1),
                "mobile_share": share(mob[idx][mask] == 1),
                "rows": int(mask.sum()),
            }
            for name, mask in (("warm", L[te] >= 1), ("cold", L[te] == 0))
            for idx in (te,)
        },
    }

    # ---------------- S5 seasonality
    res["S5"] = {
        "checkin_month_share": {
            k: {str(m): share(cim[idx] == m) for m in range(1, 13)} for k, idx in W.items()
        },
        "booking_month_share": {
            k: {str(m): share(bm[idx] == m) for m in range(1, 13)} for k, idx in W.items()
        },
        "jsd_checkin_month_bits": {
            "train_vs_valid": jsd(counts(cim[tr], 13), counts(cim[va], 13)),
            "train_vs_test": jsd(counts(cim[tr], 13), counts(cim[te], 13)),
        },
        "checkin_month_of_test_rows_seen_in_train_share": share(
            np.isin(cim[te], np.unique(cim[tr]))
        ),
        "checkin_in_next_calendar_year_share_test": share(
            (d["ctx_ci"].dt.year.to_numpy()[te] == 2015)
        ),
        "cluster_jsd_test_by_checkin_season_vs_train_same_checkin_month": {
            str(m): jsd(
                counts(y[tr][cim[tr] == m], N_CLUSTERS), counts(y[te][cim[te] == m], N_CLUSTERS)
            )
            for m in range(1, 13)
            if (cim[te] == m).sum() >= 1000 and (cim[tr] == m).sum() >= 1000
        },
        "cluster_jsd_test_vs_train_unconditional": jsd(c["train"], c["test"]),
        "test_rows_by_checkin_month": {
            str(m): int((cim[te] == m).sum()) for m in range(1, 13)
        },
    }

    # ---------------- S6 destinations
    ndest = int(dest.max()) + 1
    cnt_tr = np.bincount(dest[tr], minlength=ndest)
    dall = np.sort(np.bincount(dest)[np.bincount(dest) > 0])[::-1]
    cum = np.cumsum(dall) / dall.sum()
    tab = np.zeros((ndest, N_CLUSTERS), np.int64)
    np.add.at(tab, (dest[tr], y[tr]), 1)
    big = np.flatnonzero(tab.sum(1) >= 20)
    pb = tab[big] / tab[big].sum(1, keepdims=True)
    wgt = tab[big].sum(1) / tab[big].sum()
    ent = -np.where(pb > 0, pb * np.log2(np.where(pb > 0, pb, 1)), 0.0).sum(1)
    pa = p["train"]
    res["S6"] = {
        "destinations_all": len(dall),
        "destinations_per_window": {k: len(np.unique(dest[idx])) for k, idx in W.items()},
        "top10_share": round(float(dall[:10].sum() / dall.sum()), 4),
        "top100_share": round(float(dall[:100].sum() / dall.sum()), 4),
        "top1000_share": round(float(dall[:1000].sum() / dall.sum()), 4),
        "cum_curve_ranks": [
            int(r) for r in np.unique(np.logspace(0, np.log10(len(cum)), 80).astype(int))
        ],
        "cum_curve": [
            round(float(v), 5)
            for v in cum[np.unique(np.logspace(0, np.log10(len(cum)), 80).astype(int)) - 1]
        ],
        "targets_with_train_count_below": {
            k: {str(m): share(cnt_tr[dest[W[k]]] < m) for m in MIN_COUNTS}
            for k in ("valid", "test")
        },
        "targets_destination_never_in_train": {k: share(cnt_tr[dest[W[k]]] == 0) for k in ("valid", "test")},
        "dest_type_share": {
            k: {
                str(t): share(d["ctx_dest_type"].to_numpy()[idx] == t)
                for t in sorted(d["ctx_dest_type"].dropna().unique())[:6]
            }
            for k, idx in W.items()
        },
        "destinations_with_ge20_train_bookings": len(big),
        "mean_cluster_entropy_bits_given_destination_ge20": round(float((wgt * ent).sum()), 3),
        "cluster_entropy_bits_marginal_train": round(float(-(pa * np.log2(pa)).sum()), 3),
        "mean_top1_cluster_share_given_destination_ge20": round(float((wgt * pb.max(1)).sum()), 4),
    }

    # ---------------- S7 repeats: pre-window history vs in-window history
    idx_all = np.arange(n)

    def first_idx(key: np.ndarray) -> np.ndarray:
        """Event index of the first occurrence of each row's key."""
        return pd.Series(idx_all).groupby(key, sort=False).transform("min").to_numpy()

    keys = {
        "cluster": uid * 128 + y,
        "destination": uid * 10**6 + dest,
        "pair": (uid * 10**6 + dest) * 128 + y,
    }
    fi = {name: first_idx(kk) for name, kk in keys.items()}
    start_ev = {i: int(np.flatnonzero(win == i)[0]) for i in range(3)}
    s7: dict = {}
    for wi, k in ((0, "train"), (1, "valid"), (2, "test")):
        idx = W[k]
        pre = {name: fi[name][idx] < start_ev[wi] for name in keys}
        full = {name: fi[name][idx] < idx for name in keys}
        r = {
            "all": {
                name: {
                    "in_history_full": share(full[name]),
                    "in_pre_window_history": share(pre[name]),
                    "only_via_in_window": share(full[name] & ~pre[name]),
                }
                for name in keys
            }
        }
        for b, lo, hi in H_BUCKETS:
            m = (h[idx] >= lo) & (h[idx] <= hi)
            r[f"h={b}"] = {
                "rows": int(m.sum()),
                **{f"{name}_in_history_full": share(full[name][m]) for name in keys},
            }
        r["pair_given_destination_in_history_full"] = share(full["pair"][full["destination"]])
        s7[k] = r
    res["S7"] = s7

    # ---------------- S8 information in the query fields (fit on train)
    gains: dict = {}
    cand = {
        "checkin_weekday": d["ctx_ci"].dt.dayofweek.to_numpy().astype("int64"),
        "user_region (candidate)": d["ctx_user_region"].to_numpy().astype("int64"),
        "user_city (candidate)": d["ctx_user_city"].to_numpy().astype("int64"),
    }
    for name, vals in [(f, codes[:, j].astype("int64")) for j, f in enumerate(fields)] + list(
        cand.items()
    ):
        gains[name] = {
            "valid": info_gain(vals, y, tr, va),
            "test": info_gain(vals, y, tr, te),
        }
    res["S8"] = {
        "information_gain_bits": dict(sorted(gains.items(), key=lambda kv: -kv[1]["test"])),
        "cluster_marginal_entropy_bits_train": res["S6"]["cluster_entropy_bits_marginal_train"],
        "note": f"p(cluster | value) smoothed to the train marginal, alpha {ALPHA}; fit on train, scored on the window",
    }

    # ---------------- S9 slice sizes (test and valid)
    dest_in_hist = fi["destination"] < idx_all  # destination already in the user's full history
    s9: dict = {}
    for k in ("valid", "test"):
        idx = W[k]
        s9[k] = {
            "rows": len(idx),
            "by_h": {b: int(((h[idx] >= lo) & (h[idx] <= hi)).sum()) for b, lo, hi in H_BUCKETS},
            "by_L": {b: int(((L[idx] >= lo) & (L[idx] <= hi)).sum()) for b, lo, hi in L_BUCKETS},
            "by_party": {pt: int((party[idx] == pt).sum()) for pt in PARTY},
            "by_package": {"package": int((pkg[idx] == 1).sum()), "no_package": int((pkg[idx] == 0).sum())},
            "by_checkin_month": {str(m): int((cim[idx] == m).sum()) for m in range(1, 13)},
            "by_destination_known": {
                "dest_in_user_history": int(dest_in_hist[idx].sum()),
                "dest_new_for_user_warm": int((~dest_in_hist[idx] & (L[idx] >= 1)).sum()),
                "cold_no_history": int((L[idx] == 0).sum()),
            },
            "by_destination_seen_in_train": {
                "seen": int((cnt_tr[dest[idx]] >= 1).sum()),
                "unseen": int((cnt_tr[dest[idx]] == 0).sum()),
            },
        }
    res["S9"] = s9

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "eda_split.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    make_figures(res, months, comp_months=months)
    print("wrote", RES / "eda_split.json", "and", FIG)


LAB = {  # key -> (en, vi)
    "bookings": ("bookings per month", "booking mỗi tháng"),
    "new_users": ("new users", "user mới"),
    "L": ("History length L (earlier bookings of the user)", "Độ dài lịch sử L"),
    "h": ("In-window history h", "Lịch sử trong cùng cửa sổ h"),
    "share_rows": ("share of rows", "tỉ lệ dòng"),
    "users": ("users (log)", "số user (log)"),
    "per_user": ("bookings per user (log)", "booking mỗi user (log)"),
    "rank": ("cluster rank by train share", "hạng cluster theo tỉ lệ train"),
    "cl_share": ("share of bookings (log)", "tỉ lệ booking (log)"),
    "jsd_m": ("JSD of monthly cluster mix vs train (bits)", "JSD phân bố cluster theo tháng so với train (bit)"),
    "party": ("Party type", "Loại nhóm khách"),
    "feat": ("share of rows", "tỉ lệ dòng"),
    "ci_month": ("Check-in month", "Tháng check-in"),
    "dest_rank": ("destination rank (log)", "hạng destination (log)"),
    "cum": ("cumulative share of bookings", "tỉ lệ booking tích lũy"),
    "min_m": ("train bookings of the target's destination below m", "số booking train của destination nhỏ hơn m"),
    "rep": ("Share of rows whose item is already in the user's history", "Tỉ lệ dòng có mục đã nằm trong lịch sử"),
    "pre": ("in pre-window history", "trong lịch sử trước cửa sổ"),
    "inw": ("only through in-window bookings", "chỉ nhờ booking trong cửa sổ"),
    "bits": ("information gain (bits)", "lượng thông tin (bit)"),
    "rows": ("test rows", "số dòng test"),
}


def make_figures(res: dict, months: list[str], comp_months: list[str]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for lang in ("en", "vi"):
        T = (lambda k: LAB[k][0 if lang == "en" else 1])
        out = FIG / lang
        out.mkdir(parents=True, exist_ok=True)
        mo = res["S1"]["monthly"]
        x = np.arange(len(months))

        # S1 monthly volume with window shading
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(x, [mo[m]["bookings"] for m in months],
               color=[COLORS[mo[m]["window"]] for m in months], label=T("bookings"))
        ax.plot(x, [mo[m]["new_users"] for m in months], "k.-", lw=1, label=T("new_users"))
        ax.set_xticks(x[::2], [months[i] for i in range(0, len(months), 2)], rotation=60)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in WN]
                  + [ax.lines[0]], labels=WN + [T("new_users")], frameon=False)
        ax.set_ylabel(T("bookings"))
        fig.tight_layout(); fig.savefig(out / "eda_s1_volume.png", dpi=140); plt.close(fig)

        # S1b family and package share per booking month
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(x, [mo[m]["family_share"] for m in months], "o-", ms=3, color="#3b6ea5",
                label="family")
        ax.plot(x, [mo[m]["package_share"] for m in months], "s-", ms=3, color="#7a9e7e",
                label="is_package")
        for i, m in enumerate(months):
            if mo[m]["window"] != "train":
                ax.axvspan(i - 0.5, i + 0.5, color=COLORS[mo[m]["window"]], alpha=0.12, lw=0)
        ax.set_xticks(x[::2], [months[i] for i in range(0, len(months), 2)], rotation=60)
        ax.set_ylabel(T("share_rows")); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "eda_s1b_trip_mix.png", dpi=140); plt.close(fig)

        # S2 users and histories
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        hh = res["S2"]["bookings_per_user_hist"]
        ax[0].loglog([int(k) for k in hh], list(hh.values()), "o", ms=3, color="#3b6ea5")
        ax[0].set_xlabel(T("per_user")); ax[0].set_ylabel(T("users"))
        for a, key, ttl in ((ax[1], "L_buckets", "L"), (ax[2], "h_buckets", "h")):
            bk = list(res["S2"][key]["train"])
            w = 0.27
            for i, k in enumerate(WN):
                a.bar(np.arange(len(bk)) + (i - 1) * w, list(res["S2"][key][k].values()), w,
                      color=COLORS[k], label=k)
            a.set_xticks(np.arange(len(bk)), bk); a.set_title(T(ttl), fontsize=9)
            a.set_ylabel(T("share_rows"))
        ax[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "eda_s2_users.png", dpi=140); plt.close(fig)

        # S3 target shift
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for k in WN:
            ax[0].plot(res["S3"]["share_by_train_rank"][k], color=COLORS[k], label=k, lw=1.5)
        ax[0].set_yscale("log"); ax[0].set_xlabel(T("rank")); ax[0].set_ylabel(T("cl_share"))
        ax[0].legend(frameon=False)
        jm = res["S3"]["monthly_jsd_vs_train_bits"]
        ax[1].bar(x, [jm[m] for m in months], color=[COLORS[mo[m]["window"]] for m in months])
        ax[1].set_xticks(x[::2], [months[i] for i in range(0, len(months), 2)], rotation=60)
        ax[1].set_ylabel(T("jsd_m"))
        fig.tight_layout(); fig.savefig(out / "eda_s3_target.png", dpi=140); plt.close(fig)

        # S4 context composition
        cp = res["S4"]["composition"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        w = 0.27
        for i, k in enumerate(WN):
            ax[0].bar(np.arange(len(PARTY)) + (i - 1) * w, [cp[k]["party"][pt] for pt in PARTY], w,
                      color=COLORS[k], label=k)
        ax[0].set_xticks(np.arange(len(PARTY)), PARTY); ax[0].set_title(T("party"), fontsize=9)
        ax[0].legend(frameon=False); ax[0].set_ylabel(T("feat"))
        for i, k in enumerate(WN):
            ax[1].bar(np.arange(2) + (i - 1) * w, [cp[k]["package_share"], cp[k]["mobile_share"]], w,
                      color=COLORS[k])
        ax[1].set_xticks(np.arange(2), ["is_package", "is_mobile"])
        jf = res["S4"]["jsd_bits_per_field"]
        names = sorted(jf, key=lambda f: -jf[f]["train_vs_test"])
        ax[2].barh(names, [jf[f]["train_vs_test"] for f in names], color=COLORS["test"], label="train vs test")
        ax[2].barh(names, [jf[f]["train_vs_valid"] for f in names], color=COLORS["valid"], height=0.4, label="train vs valid")
        ax[2].set_xlabel("JSD (bits)"); ax[2].legend(frameon=False); ax[2].invert_yaxis()
        fig.tight_layout(); fig.savefig(out / "eda_s4_context.png", dpi=140); plt.close(fig)

        # S5 seasonality
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        for i, k in enumerate(WN):
            ax[0].bar(np.arange(1, 13) + (i - 1) * w, list(res["S5"]["checkin_month_share"][k].values()),
                      w, color=COLORS[k], label=k)
            ax[1].bar(np.arange(1, 13) + (i - 1) * w, list(res["S5"]["booking_month_share"][k].values()),
                      w, color=COLORS[k])
        ax[0].set_title(T("ci_month"), fontsize=9); ax[0].legend(frameon=False)
        ax[1].set_title("Booking month" if lang == "en" else "Tháng đặt (booking)", fontsize=9)
        for a in ax:
            a.set_xticks(range(1, 13)); a.set_ylabel(T("share_rows"))
        fig.tight_layout(); fig.savefig(out / "eda_s5_season.png", dpi=140); plt.close(fig)

        # S6 destinations
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].semilogx(res["S6"]["cum_curve_ranks"], res["S6"]["cum_curve"], color="#3b6ea5")
        ax[0].set_xlabel(T("dest_rank")); ax[0].set_ylabel(T("cum"))
        ob = res["S6"]["targets_with_train_count_below"]
        ms = list(ob["test"])
        for i, k in enumerate(("valid", "test")):
            ax[1].bar(np.arange(len(ms)) + (i - 0.5) * 0.4, [ob[k][m] for m in ms], 0.4,
                      color=COLORS[k], label=k)
        ax[1].set_xticks(np.arange(len(ms)), ms); ax[1].set_xlabel(T("min_m")); ax[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "eda_s6_destinations.png", dpi=140); plt.close(fig)

        # S7 repeats
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        for a, name in zip(ax, ("cluster", "destination", "pair")):
            pre = [res["S7"][k]["all"][name]["in_pre_window_history"] for k in WN]
            inw = [res["S7"][k]["all"][name]["only_via_in_window"] for k in WN]
            a.bar(WN, pre, color="#3b6ea5", label=T("pre"))
            a.bar(WN, inw, bottom=pre, color="#d98e04", label=T("inw"))
            a.set_title(name, fontsize=9)
        ax[0].set_ylabel(T("rep")); ax[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(out / "eda_s7_repeats.png", dpi=140); plt.close(fig)

        # S8 information gain
        g = res["S8"]["information_gain_bits"]
        names = list(g)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(names, [g[f]["test"] for f in names], color=COLORS["test"], label="test")
        ax.barh(names, [g[f]["valid"] for f in names], color=COLORS["valid"], height=0.4, label="valid")
        ax.set_xscale("log"); ax.invert_yaxis(); ax.set_xlabel(T("bits")); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "eda_s8_information.png", dpi=140); plt.close(fig)

        # S9 slice sizes
        s9 = res["S9"]["test"]
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
        for a, key in zip(ax, ("by_h", "by_party", "by_checkin_month", "by_destination_known")):
            a.bar(list(s9[key]), list(s9[key].values()), color=COLORS["test"])
            a.set_title(key, fontsize=9); a.tick_params(axis="x", rotation=45)
        ax[0].set_ylabel(T("rows"))
        fig.tight_layout(); fig.savefig(out / "eda_s9_slices.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
