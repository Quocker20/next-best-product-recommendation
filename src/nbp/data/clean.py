"""Clean raw Expedia booking rows into the standard interaction schema.

Policy (decided 2026-10-05): rows are never dropped for context defects. The affected
context value is nulled and a `flag_*` column is set, so the evaluation event set is
unchanged. Only duplicate keys are dropped. Burst repeats are flagged, not removed; the
protocol (Day 3) decides whether to collapse them or report both ways.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DUP_KEY = ["user_id", "date_time", "srch_destination_id", "hotel_cluster"]
FLAG_COLS = [
    "flag_ci_before_search",
    "flag_co_before_ci",
    "flag_zero_adults",
    "flag_zero_rooms",
    "flag_burst_repeat",
]


def party_type(adults: pd.Series, children: pd.Series) -> pd.Series:
    """Trip party from raw counts: family (children>0) | solo (1) | couple (2) | group (3+) | unknown.

    Same rule as `scripts/expedia_context_signal.py`; applied before any nulling.
    """
    out = np.where(
        children > 0,
        "family",
        np.where(
            adults == 1, "solo", np.where(adults == 2, "couple", np.where(adults == 0, "unknown", "group"))
        ),
    )
    return pd.Series(out, index=adults.index, dtype="string")


def flag_burst_repeats(df: pd.DataFrame) -> pd.Series:
    """True where a booking repeats the same user's previous booking exactly.

    "Exactly" = same destination, check-in, check-out and cluster; previous = the user's
    preceding booking by (timestamp, file order). Input needs `user_id`, `date_time`,
    `srch_destination_id`, `srch_ci`, `srch_co`, `hotel_cluster`. Output aligned to `df.index`.
    """
    order = df.sort_values(["user_id", "date_time"], kind="stable").index
    s = df.loc[order]
    same_user = s["user_id"].eq(s["user_id"].shift())
    same_trip = (
        s["srch_destination_id"].eq(s["srch_destination_id"].shift())
        & s["srch_ci"].eq(s["srch_ci"].shift())
        & s["srch_co"].eq(s["srch_co"].shift())
        & s["hotel_cluster"].eq(s["hotel_cluster"].shift())
    )
    return (same_user & same_trip).reindex(df.index)


def clean_bookings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Raw bookings (`load.load_bookings` output) -> standard schema + step counts.

    Returns (clean, counts). `clean` is sorted by (timestamp, user_id, source row) and has
    `src_row` = row position in the raw bookings table. `counts` holds the exact numbers
    for the data card.
    """
    counts: dict = {"rows_in": len(raw)}
    df = raw.copy()
    df["src_row"] = np.arange(len(df), dtype="int64")

    dup = df.duplicated(DUP_KEY, keep="first")
    counts["duplicate_keys_dropped"] = int(dup.sum())
    df = df.loc[~dup].copy()

    df = df.sort_values(["date_time", "user_id", "src_row"], kind="stable").reset_index(drop=True)

    search_day = df["date_time"].dt.normalize()
    lead = (df["srch_ci"] - search_day).dt.days
    stay = (df["srch_co"] - df["srch_ci"]).dt.days
    counts["srch_ci_missing"] = int(df["srch_ci"].isna().sum())
    counts["srch_co_missing"] = int(df["srch_co"].isna().sum())
    counts["co_equals_ci"] = int((stay == 0).sum())

    out = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "item_id": df["hotel_cluster"].astype("int16"),
            "timestamp": df["date_time"],
            "event_type": "booking",
            "session_id": pd.array([pd.NA] * len(df), dtype="Int64"),
            "domain": "hospitality",
            "dataset": "expedia",
            "src_row": df["src_row"],
        }
    )

    f_ci = (lead < 0).fillna(False)
    f_co = (stay < 0).fillna(False)
    f_adults = df["srch_adults_cnt"] == 0
    f_rooms = df["srch_rm_cnt"] == 0

    out["ctx_dest_id"] = df["srch_destination_id"]
    out["ctx_dest_type"] = df["srch_destination_type_id"]
    out["ctx_ci"] = df["srch_ci"]
    out["ctx_co"] = df["srch_co"].mask(f_co)
    out["ctx_lead_days"] = lead.mask(f_ci).astype("Int16")
    out["ctx_stay_nights"] = stay.mask(f_co).astype("Int16")
    out["ctx_checkin_month"] = df["srch_ci"].dt.month.astype("Int8")
    out["ctx_adults"] = df["srch_adults_cnt"].mask(f_adults).astype("Int8")
    out["ctx_children"] = df["srch_children_cnt"].astype("Int8")
    out["ctx_rooms"] = df["srch_rm_cnt"].mask(f_rooms).astype("Int8")
    out["ctx_party_type"] = party_type(df["srch_adults_cnt"], df["srch_children_cnt"])
    out["ctx_is_package"] = df["is_package"].astype("int8")
    out["ctx_channel"] = df["channel"]
    out["ctx_is_mobile"] = df["is_mobile"].astype("int8")
    out["ctx_site_name"] = df["site_name"]
    out["ctx_posa_continent"] = df["posa_continent"]
    out["ctx_user_country"] = df["user_location_country"]
    out["ctx_user_region"] = df["user_location_region"]
    out["ctx_user_city"] = df["user_location_city"]
    # Describe the booked hotel: labels/analysis only, never model input.
    out["tgt_continent"] = df["hotel_continent"]
    out["tgt_country"] = df["hotel_country"]
    out["tgt_market"] = df["hotel_market"]
    out["tgt_orig_distance"] = df["orig_destination_distance"]
    out["aux_cnt"] = df["cnt"]

    out["flag_ci_before_search"] = f_ci
    out["flag_co_before_ci"] = f_co
    out["flag_zero_adults"] = f_adults
    out["flag_zero_rooms"] = f_rooms
    out["flag_burst_repeat"] = flag_burst_repeats(df)

    for c in FLAG_COLS:
        counts[c] = int(out[c].sum())
    counts["rows_out"] = len(out)
    return out, counts


def summarize(clean: pd.DataFrame) -> dict:
    """Exact data-card numbers computed from the cleaned table."""
    per_user = clean.groupby("user_id").size()
    ts = clean["timestamp"]
    return {
        "rows": len(clean),
        "users": int(per_user.size),
        "items": int(clean["item_id"].nunique()),
        "unique_user_item_pairs": int(clean[["user_id", "item_id"]].drop_duplicates().shape[0]),
        "single_booking_users": int((per_user == 1).sum()),
        "single_booking_user_share": float((per_user == 1).mean()),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
        "timestamp_dtype": str(ts.dtype),
        "ci_min": str(clean["ctx_ci"].min()),
        "ci_max": str(clean["ctx_ci"].max()),
        "party_type_counts": {k: int(v) for k, v in clean["ctx_party_type"].value_counts().items()},
        "burst_repeat_share": float(clean["flag_burst_repeat"].mean()),
    }
