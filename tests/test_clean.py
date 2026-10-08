import pandas as pd

from nbp.data.clean import (
    clean_bookings,
    collapse_bursts,
    flag_burst_repeats,
    flag_session_repeats,
    party_type,
    summarize,
)
from nbp.data.load import DATE_COLS, filter_bookings


def _row(**kw):
    base = {
        "date_time": "2014-03-01 10:00:00",
        "srch_ci": "2014-04-01",
        "srch_co": "2014-04-04",
        "orig_destination_distance": 100.0,
        "site_name": 2,
        "posa_continent": 3,
        "user_location_country": 66,
        "user_location_region": 1,
        "user_location_city": 1,
        "user_id": 1,
        "is_mobile": 0,
        "is_package": 0,
        "channel": 9,
        "srch_adults_cnt": 2,
        "srch_children_cnt": 0,
        "srch_rm_cnt": 1,
        "srch_destination_id": 10,
        "srch_destination_type_id": 1,
        "cnt": 1,
        "hotel_continent": 2,
        "hotel_country": 50,
        "hotel_market": 600,
        "hotel_cluster": 7,
        "is_booking": 1,
    }
    base.update(kw)
    return base


def _raw(rows):
    """Run hand-made raw rows through the same filter the loader applies."""
    return filter_bookings(pd.DataFrame([_row(**r) for r in rows]))


def test_filter_keeps_only_bookings_and_forces_ns():
    df = _raw([{}, {"is_booking": 0}, {"user_id": 2}])
    assert len(df) == 2
    for c in DATE_COLS:
        assert str(df[c].dtype) == "datetime64[ns]"
    assert "is_booking" not in df.columns


def test_duplicate_key_dropped_keep_first():
    raw = _raw([{}, {}, {"hotel_cluster": 8}])
    clean, counts = clean_bookings(raw)
    assert counts["duplicate_keys_dropped"] == 1
    assert counts["rows_out"] == 2
    assert sorted(clean["item_id"]) == [7, 8]


def test_defects_flagged_and_nulled_not_dropped():
    raw = _raw(
        [
            {"user_id": 1},  # clean
            {"user_id": 2, "srch_ci": "2014-02-01", "srch_co": "2014-02-03"},  # ci before search
            {"user_id": 3, "srch_ci": "2014-04-05", "srch_co": "2014-04-01"},  # co before ci
            {"user_id": 4, "srch_adults_cnt": 0},
            {"user_id": 5, "srch_rm_cnt": 0},
        ]
    )
    clean, counts = clean_bookings(raw)
    assert len(clean) == 5
    c = clean.set_index("user_id")
    assert c.loc[2, "flag_ci_before_search"] and pd.isna(c.loc[2, "ctx_lead_days"])
    assert c.loc[2, "ctx_stay_nights"] == 2
    assert c.loc[3, "flag_co_before_ci"]
    assert pd.isna(c.loc[3, "ctx_co"]) and pd.isna(c.loc[3, "ctx_stay_nights"])
    assert c.loc[4, "flag_zero_adults"] and pd.isna(c.loc[4, "ctx_adults"])
    assert c.loc[5, "flag_zero_rooms"] and pd.isna(c.loc[5, "ctx_rooms"])
    assert c.loc[1, "ctx_lead_days"] == 31 and c.loc[1, "ctx_stay_nights"] == 3
    assert not c.loc[1, ["flag_ci_before_search", "flag_co_before_ci"]].any()
    assert counts["flag_ci_before_search"] == 1
    assert counts["flag_co_before_ci"] == 1


def test_sorted_by_time_then_user_and_src_row_stable():
    raw = _raw(
        [
            {"user_id": 3, "date_time": "2014-03-02 00:00:00"},
            {"user_id": 2, "date_time": "2014-03-01 00:00:00"},
            {"user_id": 1, "date_time": "2014-03-01 00:00:00"},
        ]
    )
    clean, _ = clean_bookings(raw)
    assert list(clean["user_id"]) == [1, 2, 3]
    assert list(clean["src_row"]) == [2, 1, 0]
    assert str(clean["timestamp"].dtype) == "datetime64[ns]"


def test_burst_repeat_only_when_trip_identical_to_previous_booking():
    raw = _raw(
        [
            {"user_id": 1, "date_time": "2014-03-01 10:00:00"},
            {"user_id": 1, "date_time": "2014-03-01 10:05:00"},  # same trip -> burst
            {"user_id": 1, "date_time": "2014-03-02 10:00:00", "srch_co": "2014-04-05"},
            {"user_id": 2, "date_time": "2014-03-01 10:06:00"},  # other user, same trip
            {"user_id": 1, "date_time": "2014-03-03 10:00:00", "srch_co": "2014-04-05"},  # burst
        ]
    )
    flags = flag_burst_repeats(raw.assign(date_time=raw["date_time"]))
    assert list(flags) == [False, True, False, False, True]
    clean, counts = clean_bookings(raw)
    assert counts["flag_burst_repeat"] == 2
    assert clean["flag_burst_repeat"].sum() == 2


def test_party_type_rules():
    adults = pd.Series([1, 2, 3, 0, 0, 2])
    children = pd.Series([0, 0, 0, 0, 1, 1])
    assert list(party_type(adults, children)) == [
        "solo",
        "couple",
        "group",
        "unknown",
        "family",
        "family",
    ]


def test_summarize_counts():
    raw = _raw(
        [
            {"user_id": 1, "hotel_cluster": 7},
            {"user_id": 1, "hotel_cluster": 7, "date_time": "2014-03-02 10:00:00"},
            {"user_id": 1, "hotel_cluster": 8, "date_time": "2014-03-03 10:00:00"},
            {"user_id": 2, "hotel_cluster": 7},
        ]
    )
    clean, _ = clean_bookings(raw)
    s = summarize(clean)
    assert s["rows"] == 4 and s["users"] == 2 and s["items"] == 2
    assert s["unique_user_item_pairs"] == 3
    assert s["single_booking_users"] == 1 and s["single_booking_user_share"] == 0.5


def test_collapse_bursts_keeps_first_of_chain_and_is_idempotent():
    raw = _raw(
        [
            {"user_id": 1, "date_time": "2014-03-01 10:00:00"},
            {"user_id": 1, "date_time": "2014-03-01 10:05:00"},  # burst
            {"user_id": 1, "date_time": "2014-03-01 10:09:00"},  # burst (chain A, A, A)
            {"user_id": 1, "date_time": "2014-03-05 10:00:00", "hotel_cluster": 8},
            {"user_id": 2, "date_time": "2014-03-01 10:06:00"},
        ]
    )
    flagged, _ = clean_bookings(raw)
    out, counts = collapse_bursts(flagged)
    assert counts == {"rows_in": 5, "burst_dropped": 2, "rows_out": 3}
    first = out[out["user_id"] == 1].sort_values("timestamp")
    assert list(first["timestamp"].dt.minute) == [0, 0]  # earliest record kept
    assert list(first["item_id"]) == [7, 8]
    again, c2 = collapse_bursts(out)
    assert c2["burst_dropped"] == 0 and len(again) == len(out)


def test_session_repeat_window_and_burst_exclusion():
    raw = _raw(
        [
            {"user_id": 1, "date_time": "2014-03-01 10:00:00"},
            # same dest + cluster 30 min later, other dates -> session repeat (not burst)
            {"user_id": 1, "date_time": "2014-03-01 10:30:00", "srch_co": "2014-04-09"},
            # same dest + cluster 3 h after the previous one -> outside the window
            {"user_id": 1, "date_time": "2014-03-01 13:30:00", "srch_co": "2014-04-10"},
            # other cluster inside the window -> not a repeat
            {"user_id": 1, "date_time": "2014-03-01 13:40:00", "hotel_cluster": 9},
            # identical trip -> burst, so not counted as a session repeat
            {"user_id": 2, "date_time": "2014-03-01 10:00:00"},
            {"user_id": 2, "date_time": "2014-03-01 10:10:00"},
        ]
    )
    flags = flag_session_repeats(raw)
    assert list(flags) == [False, True, False, False, False, True]
    _, counts = clean_bookings(raw)
    assert counts["flag_session_repeat"] == 1  # the user-1 pair; user 2 is a burst
    assert counts["flag_burst_repeat"] == 1


def test_ts_unix_is_exact_int_seconds_and_heavy_user_flag():
    raw = _raw([{"user_id": 1}, {"user_id": 2, "date_time": "2014-03-01 10:00:01"}])
    clean, counts = clean_bookings(raw)
    assert str(clean["ts_unix"].dtype) == "int64"
    assert list(clean["ts_unix"]) == [1393668000, 1393668001]
    assert counts["flag_heavy_user"] == 0


def test_analysis_flags_do_not_change_rows_or_context():
    raw = _raw(
        [
            {"user_id": 1},
            {"user_id": 2, "srch_adults_cnt": 6, "srch_rm_cnt": 1},  # 6 guests in 1 room
            {"user_id": 3, "srch_ci": "2015-06-01", "srch_co": "2015-06-03"},  # lead > 365 days
            {"user_id": 4, "user_location_country": 0},
        ]
    )
    clean, counts = clean_bookings(raw)
    c = clean.set_index("user_id")
    assert len(clean) == 4
    assert c.loc[2, "flag_occupancy_odd"] and c.loc[2, "ctx_adults"] == 6
    assert c.loc[3, "flag_extreme_lead"] and c.loc[3, "ctx_lead_days"] > 365
    assert c.loc[4, "flag_unknown_geo"]
    assert not c.loc[1, ["flag_occupancy_odd", "flag_extreme_lead", "flag_unknown_geo"]].any()
    assert (
        counts["flag_occupancy_odd"],
        counts["flag_extreme_lead"],
        counts["flag_unknown_geo"],
    ) == (1, 1, 1)
