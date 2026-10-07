import numpy as np
import pandas as pd

from nbp.data.query_features import (
    FORBIDDEN_EXACT,
    FORBIDDEN_PREFIX,
    QUERY_FIELDS,
    RAW_COLS,
    build_query_codes,
    build_vocab,
    encode,
)

F = list(QUERY_FIELDS)


def _raw(**over) -> pd.DataFrame:
    base = {
        "date_time": ["2014-03-10 08:00:00"] * 4,
        "srch_ci": ["2014-03-10", "2014-03-20", "2014-03-05", None],
        "srch_co": ["2014-03-11", "2014-03-30", "2014-03-04", None],
        "srch_destination_id": [10, 11, 12, 13],
        "srch_destination_type_id": [1, 1, 5, 6],
        "srch_adults_cnt": [2, 0, 1, 20],
        "srch_children_cnt": [0, 1, 0, 3],
        "srch_rm_cnt": [1, 0, 1, 12],
        "is_package": [0, 1, 0, 1],
        "is_mobile": [0, 0, 1, 1],
        "channel": [9, 9, 2, 5],
        "site_name": [2, 2, 37, 2],
        "posa_continent": [3, 3, 1, 3],
        "user_location_country": [66, 66, 70, 66],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_codes_hand_computed():
    c = build_query_codes(_raw())
    row = lambda i: dict(zip(F, c[i].tolist()))  # noqa: E731
    # same-day check-in (lead 0 -> bucket 0 -> code 1), 1 night (stay 1 -> code 2), month 3
    assert row(0) | {} == {
        "dest": 10, "dest_type": 1, "ci_month": 3, "lead": 1, "stay": 2, "adults": 2,
        "children": 0, "rooms": 1, "package": 0, "mobile": 0, "channel": 9, "site": 2,
        "posa": 3, "country": 66,
    }
    # lead 10 days -> bin (6, 13] = index 3 -> code 4; stay 10 nights -> code 11; zero adults/rooms
    assert (row(1)["lead"], row(1)["stay"], row(1)["adults"], row(1)["rooms"]) == (4, 11, 0, 0)
    # check-in before the search day and check-out before check-in -> unknown codes
    assert (row(2)["lead"], row(2)["stay"], row(2)["ci_month"]) == (0, 0, 3)
    # missing dates -> unknown; counts clipped (adults 9, rooms 8)
    assert (row(3)["lead"], row(3)["stay"], row(3)["ci_month"]) == (0, 0, 0)
    assert (row(3)["adults"], row(3)["rooms"]) == (9, 8)


def test_no_forbidden_raw_column_is_used():
    bad = [c for c in RAW_COLS if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIX)]
    assert bad == []


def test_vocab_uses_train_rows_only_and_min_count():
    codes = np.zeros((6, len(F)), dtype=np.int32)
    codes[:, 0] = [7, 7, 7, 7, 7, 9]  # dest: 7 five times, 9 once
    codes[:, 13] = [1, 1, 2, 2, 2, 3]  # country
    train = np.array([0, 1, 2, 3, 4])  # row 5 (dest 9, country 3) is not a train target
    vocab = build_vocab(codes, train, {f: (5 if f == "dest" else 3 if f == "country" else 1) for f in F})
    assert vocab[0].tolist() == [7]  # dest 7 has 5 train targets
    assert vocab[13].tolist() == [2]  # country 2 has 3, country 1 has 2


def test_encode_maps_unseen_to_zero_and_known_to_rank_plus_one():
    vocab = [np.array([3, 8])] * len(F)
    codes = np.full((4, len(F)), 8, dtype=np.int32)
    codes[1] = 3
    codes[2] = 5  # between vocab values
    codes[3] = 99  # beyond the largest
    out = encode(codes, vocab)
    assert out[:, 0].tolist() == [2, 1, 0, 0]


def test_encode_empty_vocab():
    out = encode(np.array([[4] * len(F)], dtype=np.int32), [np.array([], dtype=np.int64)] * len(F))
    assert out.tolist() == [[0] * len(F)]
