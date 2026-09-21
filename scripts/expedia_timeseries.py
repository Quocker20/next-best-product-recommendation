"""Time-series decomposition and anomaly detection on the Expedia daily series.

Consumes reports/summary/expedia_daily_series.csv (written by
scripts/expedia_eda_detail.py) and produces the numbers the week-1 deck needs
for the "transaction volume over time / spikes / drops" charts:

  * weekly periodicity (weekday factors, autocorrelation)
  * trend + weekday adjusted expectation, robust residual z-score
  * ranked spikes and drops, each matched against a calendar of US public
    holidays computed programmatically (no hand-typed event labels)
  * monthly and year-over-year aggregates, conversion-rate series

Input:  reports/summary/expedia_daily_series.csv
Output: reports/summary/expedia_timeseries.json
        reports/summary/expedia_daily_anomalies.csv (per-day expectation + z)

Usage:
    .venv/Scripts/python.exe scripts/expedia_timeseries.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "summary"
SRC = OUT_DIR / "expedia_daily_series.csv"

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Z_THRESHOLD = 3.5


def us_holidays(years: list[int]) -> dict[str, str]:
    """Fixed + rule-based US public holidays for the covered years.

    Rules only (no hand-typed dates): fixed-date holidays, Thanksgiving as the
    4th Thursday of November, Black Friday / Cyber Monday relative to it,
    Memorial Day as the last Monday of May, Labor Day as the first Monday of
    September. Returns {ISO date: holiday name}.
    """
    out: dict[str, str] = {}
    for y in years:
        for name, (m, d) in {
            "New Year's Day": (1, 1),
            "Independence Day": (7, 4),
            "Christmas Eve": (12, 24),
            "Christmas Day": (12, 25),
            "New Year's Eve": (12, 31),
        }.items():
            out[f"{y}-{m:02d}-{d:02d}"] = name
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
        thanksgiving = nov[nov.dayofweek == 3][3]
        out[str(thanksgiving.date())] = "Thanksgiving"
        out[str((thanksgiving + pd.Timedelta(days=1)).date())] = "Black Friday"
        out[str((thanksgiving + pd.Timedelta(days=4)).date())] = "Cyber Monday"
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31", freq="D")
        out[str(may[may.dayofweek == 0][-1].date())] = "Memorial Day"
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30", freq="D")
        out[str(sep[sep.dayofweek == 0][0].date())] = "Labor Day"
    return out


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["date"]).set_index("date").sort_index()
    s = df["bookings"].astype(float)

    # ---- trend + weekday seasonality ----
    trend = s.rolling(7, center=True, min_periods=4).mean()
    ratio = s / trend
    weekday_factor = ratio.groupby(s.index.dayofweek).median()
    expected = trend * s.index.dayofweek.map(weekday_factor)
    resid = s / expected
    med, mad = float(resid.median()), float((resid - resid.median()).abs().median())
    scale = 1.4826 * mad
    z = (resid - med) / scale

    hol = us_holidays(sorted({int(y) for y in s.index.year.unique()}))
    out_df = pd.DataFrame(
        {
            "bookings": s,
            "clicks": df["clicks"],
            "booking_rate": df["booking_rate"],
            "weekday": [WEEKDAY_NAMES[i] for i in s.index.dayofweek],
            "trend_7d": trend.round(1),
            "expected": expected.round(1),
            "resid_ratio": resid.round(4),
            "robust_z": z.round(2),
            "is_anomaly": z.abs() > Z_THRESHOLD,
            "holiday": [hol.get(str(d.date()), "") for d in s.index],
        }
    )
    out_df.index.name = "date"

    def top(frame: pd.DataFrame, n: int = 12) -> list[dict]:
        return [
            {
                "date": str(idx.date()),
                "weekday": row["weekday"],
                "bookings": int(row["bookings"]),
                "expected": None
                if pd.isna(row["expected"])
                else float(row["expected"]),
                "pct_vs_expected": (
                    None
                    if pd.isna(row["resid_ratio"])
                    else round((row["resid_ratio"] - 1) * 100, 1)
                ),
                "robust_z": None
                if pd.isna(row["robust_z"])
                else float(row["robust_z"]),
                "holiday": row["holiday"] or None,
            }
            for idx, row in frame.head(n).iterrows()
        ]

    flagged = out_df[out_df["is_anomaly"]]
    spikes = top(
        flagged[flagged["robust_z"] > 0].sort_values("robust_z", ascending=False)
    )
    drops = top(flagged[flagged["robust_z"] < 0].sort_values("robust_z"))

    # ---- holiday effect (all flagged days vs holiday calendar) ----
    n_flagged = len(flagged)
    n_flagged_holiday = int((flagged["holiday"] != "").sum())

    # ---- weekly / monthly / yearly aggregates ----
    weekly = s.resample("W-SUN").sum()
    monthly = s.resample("MS").sum()
    by_weekday = s.groupby(s.index.dayofweek).sum()
    by_weekday_share = (by_weekday / by_weekday.sum()).round(6)

    y2013 = s[s.index.year == 2013]
    y2014 = s[s.index.year == 2014]
    m2013 = y2013.resample("MS").sum()
    m2014 = y2014.resample("MS").sum()
    yoy = {
        int(m): round(float(m2014.iloc[i] / v - 1), 6)
        for i, (m, v) in enumerate(zip(m2013.index.month, m2013.values))
        if i < len(m2014)
    }

    rate = df["booking_rate"].astype(float)

    result = {
        "source": str(SRC.relative_to(ROOT)),
        "generated_by": "scripts/expedia_timeseries.py",
        "days": len(s),
        "date_min": str(s.index.min().date()),
        "date_max": str(s.index.max().date()),
        "daily_bookings": {
            "mean": round(float(s.mean()), 1),
            "median": float(s.median()),
            "std": round(float(s.std()), 1),
            "min": int(s.min()),
            "min_date": str(s.idxmin().date()),
            "max": int(s.max()),
            "max_date": str(s.idxmax().date()),
            "cv": round(float(s.std() / s.mean()), 4),
        },
        "periodicity": {
            "autocorr_lag1": round(float(s.autocorr(1)), 4),
            "autocorr_lag7": round(float(s.autocorr(7)), 4),
            "autocorr_lag364": round(float(s.autocorr(364)), 4),
            "weekday_factor": {
                WEEKDAY_NAMES[int(k)]: round(float(v), 4)
                for k, v in weekday_factor.items()
            },
            "weekday_share_of_bookings": {
                WEEKDAY_NAMES[int(k)]: float(v) for k, v in by_weekday_share.items()
            },
            "weekend_share": round(float(by_weekday_share.loc[[5, 6]].sum()), 6),
            "peak_weekday": WEEKDAY_NAMES[int(by_weekday.idxmax())],
            "trough_weekday": WEEKDAY_NAMES[int(by_weekday.idxmin())],
            "peak_over_trough_ratio": round(
                float(by_weekday.max() / by_weekday.min()), 4
            ),
        },
        "anomalies": {
            "method": (
                "daily bookings / (7-day centred trend x weekday factor); "
                f"robust z = (residual - median) / (1.4826 x MAD); |z| > {Z_THRESHOLD}"
            ),
            "n_flagged": n_flagged,
            "flagged_share_of_days": round(n_flagged / len(s), 6),
            "n_flagged_on_us_holiday": n_flagged_holiday,
            "spikes": spikes,
            "drops": drops,
        },
        "weekly": {
            "n_weeks": len(weekly),
            "max_week_start": str((weekly.idxmax() - pd.Timedelta(days=6)).date()),
            "max_week_bookings": int(weekly.max()),
            "min_full_week_start": str(
                (weekly.iloc[1:-1].idxmin() - pd.Timedelta(days=6)).date()
            ),
            "min_full_week_bookings": int(weekly.iloc[1:-1].min()),
        },
        "monthly_bookings": {str(k.date())[:7]: int(v) for k, v in monthly.items()},
        "yoy_2014_vs_2013_by_month": yoy,
        "yoy_total": {
            "bookings_2013": int(y2013.sum()),
            "bookings_2014": int(y2014.sum()),
            "growth": round(float(y2014.sum() / y2013.sum() - 1), 6),
            "note": "2013 starts 2013-01-07, so January 2013 is 6 days short",
        },
        "conversion_rate": {
            "mean": round(float(rate.mean()), 6),
            "min": round(float(rate.min()), 6),
            "min_date": str(rate.idxmin().date()),
            "max": round(float(rate.max()), 6),
            "max_date": str(rate.idxmax().date()),
            "by_weekday": {
                WEEKDAY_NAMES[int(k)]: round(float(v), 6)
                for k, v in rate.groupby(rate.index.dayofweek).mean().items()
            },
            "2013_mean": round(float(rate[rate.index.year == 2013].mean()), 6),
            "2014_mean": round(float(rate[rate.index.year == 2014].mean()), 6),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "expedia_timeseries.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    out_df.to_csv(OUT_DIR / "expedia_daily_anomalies.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
