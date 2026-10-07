"""Query-time features of an Expedia booking row, for the SMLP4Rec query token.

Every field describes the search that led to the booking (known before booking). Hotel fields
(`hotel_*`), `orig_destination_distance`, `user_location_region/city` and `cnt` are never used
(see `FORBIDDEN_PREFIX`, `FORBIDDEN_EXACT`). Defects are handled as in `nbp.data.clean`: a negative lead time or stay
length, a missing date, or zero adults/rooms becomes the explicit "unknown" code, never a drop.

Pipeline: `build_query_codes` (raw rows -> small integer codes) -> `build_vocab` (from train
targets only) -> `encode` (code -> embedding index, 0 = out of vocabulary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbp.data.load import to_ns

LEAD_BINS = [-1, 0, 2, 6, 13, 29, 59, 89, 179, 364, 10_000]  # days between search and check-in
STAY_CLIP = 14  # nights; longer stays share one bucket
ADULTS_CLIP, CHILDREN_CLIP, ROOMS_CLIP = 9, 9, 8

# field name -> minimum train-target count for a value to get its own embedding row
QUERY_FIELDS: dict[str, int] = {
    "dest": 5,
    "dest_type": 1,
    "ci_month": 1,
    "lead": 1,
    "stay": 1,
    "adults": 1,
    "children": 1,
    "rooms": 1,
    "package": 1,
    "mobile": 1,
    "channel": 1,
    "site": 1,
    "posa": 1,
    "country": 5,
}
RAW_COLS = [
    "date_time",
    "srch_ci",
    "srch_co",
    "srch_destination_id",
    "srch_destination_type_id",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "is_package",
    "is_mobile",
    "channel",
    "site_name",
    "posa_continent",
    "user_location_country",
]
FORBIDDEN_PREFIX = ("hotel_",)
FORBIDDEN_EXACT = (
    "orig_destination_distance",
    "user_location_region",
    "user_location_city",
    "cnt",
    "is_booking",
)


def build_query_codes(raw: pd.DataFrame) -> np.ndarray:
    """Raw booking rows -> (n, len(QUERY_FIELDS)) int32 codes, columns in `QUERY_FIELDS` order.

    Input needs `RAW_COLS` (dates may be strings). Code 0 means unknown for `ci_month`, `lead`,
    `stay`, `adults` (zero adults) and `rooms` (zero rooms); other fields keep the raw value.
    """
    day = to_ns(raw["date_time"]).dt.normalize()
    ci, co = to_ns(raw["srch_ci"]), to_ns(raw["srch_co"])
    lead = (ci - day).dt.days
    stay = (co - ci).dt.days
    lead_bucket = pd.cut(lead.where(lead >= 0), LEAD_BINS, labels=False)  # NaN stays NaN
    cols = {
        "dest": raw["srch_destination_id"],
        "dest_type": raw["srch_destination_type_id"],
        "ci_month": ci.dt.month.fillna(0),
        "lead": lead_bucket.add(1).fillna(0),
        "stay": stay.where(stay >= 0).clip(0, STAY_CLIP).add(1).fillna(0),
        "adults": raw["srch_adults_cnt"].clip(0, ADULTS_CLIP),
        "children": raw["srch_children_cnt"].clip(0, CHILDREN_CLIP),
        "rooms": raw["srch_rm_cnt"].clip(0, ROOMS_CLIP),
        "package": raw["is_package"],
        "mobile": raw["is_mobile"],
        "channel": raw["channel"],
        "site": raw["site_name"],
        "posa": raw["posa_continent"],
        "country": raw["user_location_country"],
    }
    return np.column_stack([np.asarray(cols[f]).astype(np.int64) for f in QUERY_FIELDS]).astype(
        np.int32
    )


def build_vocab(codes: np.ndarray, train_rows: np.ndarray, min_counts: dict[str, int] | None = None):
    """Per-field sorted arrays of codes seen at least `min_count` times among the train targets.

    Args:
        codes: (n, F) from `build_query_codes` for all bookings.
        train_rows: row positions (into `codes`) of the train targets; nothing else is counted.
    Returns: list of F sorted int arrays; embedding index = position + 1, index 0 = out of vocab.
    """
    min_counts = min_counts or QUERY_FIELDS
    vocab = []
    for j, field in enumerate(QUERY_FIELDS):
        vals, counts = np.unique(codes[train_rows, j], return_counts=True)
        vocab.append(vals[counts >= min_counts[field]])
    return vocab


def encode(codes: np.ndarray, vocab: list[np.ndarray]) -> np.ndarray:
    """(n, F) codes -> (n, F) int32 embedding indices; values outside the vocab -> 0."""
    out = np.zeros(codes.shape, dtype=np.int32)
    for j, ids in enumerate(vocab):
        if len(ids) == 0:
            continue
        pos = np.searchsorted(ids, codes[:, j])
        hit = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == codes[:, j])
        out[:, j] = np.where(hit, pos + 1, 0)
    return out
