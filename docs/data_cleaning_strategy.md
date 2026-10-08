# Expedia cleaning strategy (decided 2026-10-08)

Code: `src/nbp/data/clean.py` (`clean_bookings`, `collapse_bursts`), `scripts/expedia_build_bookings.py`, audit `scripts/expedia_data_audit.py`. Numbers: `results/week4_rebuild/bookings_counts.json` and `data_audit.json`.

## Principle
Drop a row only when it records the same event twice. A defect in a context field is nulled and flagged; a row that is unusual but real is kept and, if useful for slicing, flagged.

## Tiers

| Tier | Rule | Action | Rows |
|---|---|---|---|
| 1 duplicate key | same `user_id`, `date_time`, destination, cluster | drop, keep first | 8 |
| 2 burst repeat (user decision) | same user, destination, check-in, check-out and cluster as the user's previous booking | **collapse**: dropped from targets and histories, first record kept | 48,010 |
| 3 context defect | check-in before search date; check-out before check-in; zero adults; zero rooms | null the field, flag, keep row | 617 / 3 / 5,239 / 250 |
| 4 session repeat | same user, destination and cluster booked again within 3,600 s, dates differ | **flag only** (`flag_session_repeat`), sensitivity run | 76,046 |
| 5 heavy user | 50 or more bookings | **flag only** (`flag_heavy_user`), slicing only, never an input | 110,219 rows |
| 6 order | event time as exact int64 seconds (`ts_unix`); float32 unix time is only exact to about 128 s | new column | all |
| 7 unusual but real values | more than 4 guests per room; check-in more than 365 days after the search; unknown user country or point-of-sale continent (code 0) | **flag only** (`flag_occupancy_odd`, `flag_extreme_lead`, `flag_unknown_geo`); query features already clip or bucket them | 35,577 / 1,817 / 32,426 |

Kept as is: large leads (> 365 days) and long stays (> 30) are real; `query_features.py` already buckets and clips them. `orig_destination_distance` (33.8 % null), `hotel_*` and `cnt` stay `tgt_*` / `aux_*`, never inputs.

## Evidence (all from `data_audit.json` / `bookings_counts.json`)
- Burst repeats are 2.24 % of approximate test targets and 13 % of the rows where the target cluster was already booked at the query destination (sameDest ceiling). Median gap to the repeated booking is about 44 minutes.
- Further same-destination, same-cluster repeats within one hour (other dates) number 76,046. Together with bursts they are 124,056 rows. They are not collapsed: the dates differ, so it may be a deliberate change of trip. The flag lets us report the model with and without them.
- Users: maximum 100 bookings, no pile-up at the maximum, so no capping artefact; users with 50+ bookings hold 3.67 % of rows.
- Drift: the family share of bookings falls from 26.1 % (2013) to 15.0 % (2014) and monthly volume doubles in 2014. Season and time are confounded in the temporal split; state it with the season slices.
- `destinations.csv` covers 99.58 % of booking rows (latent features usable for rare destinations).
- 4 same-user pairs share an exact timestamp; the stable sort by source row decides their order.

## Checked and left alone
Code 0 in `channel` (11.4 % of rows) is a valid channel, not a missing value; 2,103 hotel markets almost always map to one country (29 do not), so `tgt_*` stays untouched; `site_name` determines `posa_continent`; each destination has one destination type; children outnumber adults in 1.36 % of rows (families with several children, kept); the data has no price column, so no price cleaning exists to do.

## Outputs
- `data/interim/expedia_bookings_flagged.parquet`: all rows after tier 1, all flags (the "with burst" sensitivity source).
- `data/interim/expedia_bookings.parquet`: tier 2 applied. Source of every export and notebook from now on.

## Effect of the collapse
Rows 3,000,685 to 2,952,675; users unchanged (813,985; the first record of each chain is kept); single-booking users 315,683 to 320,182. Next-booking targets fall from 2,186,700 to 2,138,690, so valid and test shrink to about 213,869 each (computed as rows minus users). The old numbers stay as a "with burst" sensitivity row.

## Open
- Whether session repeats (tier 4) should also be collapsed: user decision after the sensitivity run.
- `verify_basic_baselines.py` still checks the notebook 07 outputs against the old (uncollapsed) numbers; `scripts/verify_stage_b.py` is the current consistency check.
