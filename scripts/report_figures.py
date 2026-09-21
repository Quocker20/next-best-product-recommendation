"""Derive the ratios and shares quoted in the week-1 revision report.

Pure post-processing: reads the measured JSON outputs in reports/summary/ and
emits every derived figure (hourly shares, conversion by hour, peak/trough
ratios, concentration ratios, probe lifts) so no number in the written report
is computed by hand.

Input:  reports/summary/expedia_eda_detail.json, expedia_timeseries.json,
        expedia_context_signal.json, expedia_anomaly_audit.json,
        dataset_benchmark.json
Output: reports/summary/report_figures.json

Usage:
    .venv/Scripts/python.exe scripts/report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "reports" / "summary"


def load(name: str) -> dict:
    return json.loads((SUM / name).read_text(encoding="utf-8"))


def main() -> None:
    eda = load("expedia_eda_detail.json")
    ts = load("expedia_timeseries.json")
    ctx = load("expedia_context_signal.json")
    anom = load("expedia_anomaly_audit.json")
    bench = load("dataset_benchmark.json")

    bookings = eda["bookings"]
    clicks = eda["clicks"]
    hb = {int(k): v for k, v in eda["hour_of_day"]["bookings"].items()}
    hc = {int(k): v for k, v in eda["hour_of_day"]["clicks"].items()}
    hourly = {
        h: {
            "bookings": hb[h],
            "booking_share_of_day": round(hb[h] / bookings, 6),
            "clicks": hc[h],
            "click_share_of_day": round(hc[h] / clicks, 6),
            "conversion": round(hb[h] / (hb[h] + hc[h]), 6),
        }
        for h in sorted(hb)
    }
    peak_h = max(hourly, key=lambda h: hourly[h]["bookings"])
    trough_h = min(hourly, key=lambda h: hourly[h]["bookings"])
    peak_conv = max(hourly, key=lambda h: hourly[h]["conversion"])
    trough_conv = min(hourly, key=lambda h: hourly[h]["conversion"])
    peak_click_h = max(hourly, key=lambda h: hourly[h]["clicks"])

    ci = {int(k): v for k, v in eda["checkin_month_share"].items()}
    ci_by_party = {
        p: {int(m): s for m, s in d.items()}
        for p, d in eda["checkin_month_share_by_party"].items()
    }
    probe = ctx["popularity_probe"]
    base5 = probe["recall_at_5"]["global_popularity"]
    base1 = probe["recall_at_1"]["global_popularity"]

    pkg = eda["monthly_package_share"]
    pkg_first, pkg_last = list(pkg)[0], list(pkg)[-1]

    depth = eda["user_depth"]["histogram"]
    bench_by = {r["dataset"]: r for r in bench}

    out = {
        "generated_by": "scripts/report_figures.py",
        "inputs": [
            "expedia_eda_detail.json",
            "expedia_timeseries.json",
            "expedia_context_signal.json",
            "expedia_anomaly_audit.json",
            "dataset_benchmark.json",
        ],
        "hour_of_day": hourly,
        "hour_highlights": {
            "peak_booking_hour": peak_h,
            "peak_booking_share": hourly[peak_h]["booking_share_of_day"],
            "trough_booking_hour": trough_h,
            "trough_booking_share": hourly[trough_h]["booking_share_of_day"],
            "peak_over_trough_bookings": round(hb[peak_h] / hb[trough_h], 3),
            "peak_click_hour": peak_click_h,
            "best_conversion_hour": peak_conv,
            "best_conversion": hourly[peak_conv]["conversion"],
            "worst_conversion_hour": trough_conv,
            "worst_conversion": hourly[trough_conv]["conversion"],
        },
        "weekday_highlights": {
            "shares": ts["periodicity"]["weekday_share_of_bookings"],
            "weekend_share": ts["periodicity"]["weekend_share"],
            "peak_over_trough": ts["periodicity"]["peak_over_trough_ratio"],
        },
        "checkin_seasonality": {
            "shares": ci,
            "peak_month": max(ci, key=ci.get),
            "peak_share": ci[max(ci, key=ci.get)],
            "trough_month": min(ci, key=ci.get),
            "trough_share": ci[min(ci, key=ci.get)],
            "peak_over_trough": round(
                ci[max(ci, key=ci.get)] / ci[min(ci, key=ci.get)], 3
            ),
            "family_aug_share": ci_by_party["family"][8],
            "family_jul_share": ci_by_party["family"][7],
            "family_summer_share": round(
                ci_by_party["family"][7] + ci_by_party["family"][8], 6
            ),
            "solo_summer_share": round(
                ci_by_party["solo"][7] + ci_by_party["solo"][8], 6
            ),
            "couple_summer_share": round(
                ci_by_party["couple"][7] + ci_by_party["couple"][8], 6
            ),
            "family_vs_solo_summer_ratio": round(
                (ci_by_party["family"][7] + ci_by_party["family"][8])
                / (ci_by_party["solo"][7] + ci_by_party["solo"][8]),
                3,
            ),
        },
        "package_trend": {
            "first_month": pkg_first,
            "first_share": pkg[pkg_first],
            "last_month": pkg_last,
            "last_share": pkg[pkg_last],
            "relative_decline": round(pkg[pkg_last] / pkg[pkg_first] - 1, 6),
            "max_month": max(pkg, key=pkg.get),
            "max_share": pkg[max(pkg, key=pkg.get)],
        },
        "growth": {
            "bookings_2013": ts["yoy_total"]["bookings_2013"],
            "bookings_2014": ts["yoy_total"]["bookings_2014"],
            "growth": ts["yoy_total"]["growth"],
            "conversion_2013": ts["conversion_rate"]["2013_mean"],
            "conversion_2014": ts["conversion_rate"]["2014_mean"],
            "conversion_change_rel": round(
                ts["conversion_rate"]["2014_mean"] / ts["conversion_rate"]["2013_mean"]
                - 1,
                6,
            ),
        },
        "cold_start": {
            "users_1_booking_share": depth["1"]["share"],
            "users_1_2_bookings_share": round(
                depth["1"]["share"] + depth["2"]["share"], 6
            ),
            "users_ge5_share": round(
                depth["5-10"]["share"]
                + depth["11-20"]["share"]
                + depth["21+"]["share"],
                6,
            ),
            "top10pct_users_booking_share": eda["user_depth"][
                "top10pct_users_booking_share"
            ],
            "test_users_unseen_share": probe["test_users_unseen_in_train_share"],
            "repeat_last_available_share": probe["repeat_last_available_share"],
        },
        "concentration": {
            "cluster_top1": eda["cluster_popularity"]["share_sorted_desc"][0],
            "cluster_top10": eda["cluster_popularity"]["share_top10"],
            "cluster_top20": eda["cluster_popularity"]["share_top20"],
            "cluster_bottom50": eda["cluster_popularity"]["share_bottom50"],
            "cluster_top1_over_bottom1": round(
                eda["cluster_popularity"]["share_sorted_desc"][0]
                / eda["cluster_popularity"]["share_sorted_desc"][-1],
                2,
            ),
            "dest_top10": eda["destinations"]["share_top10"],
            "dest_top100": eda["destinations"]["share_top100"],
            "dest_top1000": eda["destinations"]["share_top1000"],
            "dest_single_booking": eda["destinations"]["n_dest_single_booking"],
            "dest_single_booking_share": round(
                eda["destinations"]["n_dest_single_booking"]
                / eda["destinations"]["n_destinations_bookings"],
                6,
            ),
        },
        "probe_lift": {
            "recall_at_5": probe["recall_at_5"],
            "recall_at_1": probe["recall_at_1"],
            "lift_vs_global_at_5": {
                k: round(v / base5, 3) for k, v in probe["recall_at_5"].items()
            },
            "lift_vs_global_at_1": {
                k: round(v / base1, 3) for k, v in probe["recall_at_1"].items()
            },
            "destination_x_month_vs_destination": round(
                probe["recall_at_5"]["popularity_by_destination_x_ci_month"]
                / probe["recall_at_5"]["popularity_by_srch_destination_id"]
                - 1,
                6,
            ),
        },
        "context_shift": {
            "cluster_js_by_month_mean": ctx["distribution_shift"]["checkin_month"][
                "js_mean"
            ],
            "cluster_js_by_party_mean": ctx["distribution_shift"]["party"]["js_mean"],
            "cluster_js_by_package_max": ctx["distribution_shift"]["is_package"][
                "js_max"
            ],
            "cluster_js_by_destination_mean": ctx["distribution_shift"][
                "srch_destination_top"
            ]["js_mean"],
            "destination_js_by_month_mean": round(
                sum(
                    x["js_divergence_bits"]
                    for x in ctx["destination_mix_shift_by_checkin_month"]
                )
                / len(ctx["destination_mix_shift_by_checkin_month"]),
                5,
            ),
            "destination_vs_cluster_month_ratio": round(
                (
                    sum(
                        x["js_divergence_bits"]
                        for x in ctx["destination_mix_shift_by_checkin_month"]
                    )
                    / len(ctx["destination_mix_shift_by_checkin_month"])
                )
                / ctx["distribution_shift"]["checkin_month"]["js_mean"],
                2,
            ),
        },
        "quality": {
            "click_share": round(clicks / eda["total_rows"], 6),
            "booking_share": round(bookings / eda["total_rows"], 6),
            "anomaly_checks_booking_vs_all": {
                k: {
                    "all_share": v["all_share"],
                    "booking_share": v["booking_share"],
                    "ratio_all_over_booking": (
                        round(v["all_share"] / v["booking_share"], 3)
                        if v["booking_share"] > 0
                        else None
                    ),
                }
                for k, v in anom["checks"].items()
            },
            "checkin_year_outliers_all": {
                y: c
                for y, c in anom["checkin_year_histogram"]["all"].items()
                if int(y) > 2016
            },
            "checkin_year_outliers_booking": {
                y: c
                for y, c in anom["checkin_year_histogram"]["booking"].items()
                if int(y) > 2016
            },
        },
        "benchmark_ratios": {
            "expedia_vs_trivago_span_days": (
                bench_by["Expedia Hotel Recommendations"]["span_days"]
                / bench_by["Trivago RecSys Challenge 2019"]["span_days"]
            ),
            "expedia_interactions_per_item": round(
                bench_by["Expedia Hotel Recommendations"]["interactions"]
                / bench_by["Expedia Hotel Recommendations"]["n_items"],
                1,
            ),
            "trivago_interactions_per_item": round(
                bench_by["Trivago RecSys Challenge 2019"]["interactions"]
                / bench_by["Trivago RecSys Challenge 2019"]["n_items"],
                2,
            ),
        },
    }

    (SUM / "report_figures.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
