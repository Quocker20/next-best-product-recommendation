# 04 Demo options for the final presentation

Written 2026-10-11. Scope: a short oral presentation with limited time. This file proposes demo
formats, ranks them, picks a combination, lists curated test bookings, and gives the change needed
to point the demo at the early-stopping checkpoint. Nothing here changes the evaluation protocol
or the data.

Statements every demo must carry (CLAUDE.md §2, §3): public Expedia data only; `hotel_cluster` is
an anonymous proxy for room category / package; offline evaluation only; no Vinpearl / GSM data
is used or implied.

## 0. Current state (checked 2026-10-11)

- `scripts/demo_ui.py` runs. Started in the smlp4rec venv, `/api/meta`, `/api/seed` and
  `/api/recommend` all answered; one recommendation (main model + two baselines) takes about
  0.01 to 0.06 s. It loads the 3-epoch checkpoints (`experiments/2026-10-09_..._query-1d-clean-cold*`),
  not the early-stopping run.
- The `ready:` line is printed without `flush`, so when the server runs in the background (or
  stdout is piped) nothing appears although the server is up. Start it with `python -u` or add
  `flush=True` (see §5).
- With only the early-stopping checkpoint loaded (prototype `scripts/demo_prototypes/serve_es.py`),
  the server printed `ready` about 4 s after launch and used roughly 750 MB RAM (task list).
- Baselines in the demo come from notebook 07 (`basic_baselines_models.joblib` exported to
  `data/interim/demo_baselines.npz`), not from notebook 07b (`baselines_es_models.joblib`, same
  training set as 01d). Baseline ranks in the demo therefore differ slightly from the final table
  in `results/week4_rebuild/baselines_vs_final_es.json`.
- Streamlit and Gradio are **not installed** in either venv (`pip list` in `.venv` and
  `C:/Users/quoca/.venvs/smlp4rec`). Both venvs have `pillow`; `.venv` has `matplotlib`.

Headline numbers a demo can quote (test, seed 2022, collapsed Expedia, source
`results/week4_rebuild/baselines_vs_final_es.json`):

| Row set | Main model Recall@5 | Logistic regression Recall@5 | ItemKNN Recall@5 |
|---|---|---|---|
| warm (n = 213,869) | 0.6104 | 0.4969 | 0.3036 |
| cold, L = 0 (n = 49,462) | 0.5252 | 0.4857 | 0.1318 |

Sanity check of the demo path: 1,500 cold + 3,000 warm test bookings sent through the API of the
early-stopping demo gave pooled Recall@5 0.577 (main), 0.511 (logistic regression), 0.246
(ItemKNN). This is a sample, not a reported metric; it is consistent with the table
(1,500 × 0.5252 + 3,000 × 0.6104) / 4,500 = 0.582, within about one binomial standard error
(0.007 at n = 4,500).
(Corrected 2026-10-11 by the verification pass: the warm main-model value was 0.5965, which is the
query token without sameDest; the main model is query token + sameDest, `final_benchmark.csv`.)

## 1. Ranked options

### Option 1 (rank 1): static storyboard of 3 to 4 curated real test bookings

**What the audience sees.** One HTML page (or 3 slides) per booking: a "search card" (destination
id, check-in month, party, package), the user's earlier bookings as a horizontal timeline of
cluster chips coloured by destination, then three top-5 columns side by side (main model,
logistic regression, ItemKNN) with the actual booked cluster highlighted. Under each booking one
"what-if" strip: the same search with one field changed (solo instead of family, history removed,
another destination) and the new top 5, changed positions marked. All numbers are precomputed.

**Claims supported.** RQ1: the context-aware model ranks the actual cluster well for a new user
where history-only CF cannot (cold case) and uses history plus the destination for returning
users (known-destination case). Shows trip context (family / December) matters. The honest miss
example supports the limitations section.

**Effort.** 3 to 5 h: extend `find_curated_examples.py` to dump JSON for the chosen seeds (written
under `results/<week>/`), a small stdlib builder in `scripts/` writing one HTML page under
`reports/summary/<week>/`. No new dependency.

**Risk on stage.** Very low: a local file in the browser or slides; no model, no server.

**Dependencies.** None beyond the existing demo server for the one-off precompute.

**Reuses.** `scripts/demo_ui.py` (all scoring via its API), `scripts/demo_prototypes/find_curated_examples.py`,
CSS tokens of `reports/summary/week4_rebuild/week4_clean_slides.html` for a consistent look.

**Talk track (about 75 s).**
> "Here are three real bookings from the test period, which the model never saw. First, a new
> customer: no history at all. ItemKNN can only fall back to popularity, and the cluster this person
> actually booked is ranked 75th. The logistic regression puts it at 26. Our model reads the search
> itself, destination, dates and party, and puts it first. Now watch what happens if the same
> search had been a family with two children: the top recommendation changes and the booked cluster
> drops to seventh. The model is reacting to trip context, not only to popularity.
> Second, a returning customer with eleven earlier bookings, searching a destination where they
> stayed before. The model ranks their actual choice first; remove the history and it falls to
> fourth. Third, where we fail: this customer went back to a cluster from an older trip, and our
> sameDest term pushed their last stay at this destination instead. Cluster ids are anonymous
> proxies for room or package type, and everything is offline on public Expedia data."

### Option 2 (rank 2): improved live stdlib demo ("demo v2")

**What the audience sees.** The existing page reorganised for a projector:
1. Preset buttons for the curated bookings (§3) next to "Random test booking".
2. Three top-5 columns side by side (main, logistic regression, ItemKNN); the actual cluster
   highlighted in each column, its rank under each column.
3. What-if buttons that change one thing and re-rank instantly: "solo", "family (2+2)",
   "check-in December", "check-in July", "remove history", "switch to a destination in the
   history". Each re-rank shows arrows (up / down / new) relative to the previous top 5.
4. sameDest highlight: in the main column, mark rows whose `samedest_share` > 0 and show the
   model-only order next to the final order (both are already returned: `model_p`,
   `samedest_share`, `w_samedest`).
5. History timeline: the user's last 20 bookings as chips coloured by destination, chips at the
   searched destination outlined (that is exactly what sameDest uses).
6. Large-font "presentation mode" (hide the 14-field form behind a toggle).

**Claims supported.** RQ1 (context and history both matter; side by side against both
baselines), plus a live demonstration that the query token reacts to party and season. RQ4 is not
well served by a single user; leave it to the result slides.

**Effort.** 6 to 10 h (front-end in the `PAGE` string, one extra endpoint not needed: what-ifs are
just repeated `/api/recommend` calls with edited fields; model-only order needs either all 100
`model_p` values returned or a re-sort of the top rows). Add a smoke test (start server, one call)
because the demo is not unit-tested (open item in PROGRESS.md).

**Risk on stage.** Medium. Failure points: wrong venv / kernel, missing `data/interim/` files on
the presenting laptop, port in use, projector resolution, sleeping laptop. Startup is about 4 s
with one checkpoint, so a restart is cheap. No network needed (127.0.0.1).

**Dependencies.** smlp4rec venv, `data/interim/recbole/expedia_clean_rowid/`, `data/interim/expedia_query_codes_clean.npz`,
`data/interim/demo_baselines.npz`, the early-stopping checkpoint folder. All gitignored, so the
demo only works on the machine that has them.

**Reuses.** Everything in `scripts/demo_ui.py`; `nbp.hybrid` sameDest code; run json weights.

**Talk track (about 60 s).**
> "This is the model running live on my laptop. I pick a returning customer from the test period:
> eleven earlier bookings, the coloured chips, and this search is for a destination they visited
> before, outlined here. Main model on the left, the two baselines on the right; the booked cluster
> is green, and only our model has it first. The marker shows the sameDest term: it adds the
> cluster this person booked last time at this destination. Now I press 'remove history': same
> search, new customer, and the booked cluster drops to fourth. And 'switch destination' to another
> place in their history: the top recommendation becomes what they booked there. Context moves
> the ranking; history moves it again."

### Option 3 (rank 3): recorded screen capture as fallback (MP4 / GIF)

**What the audience sees.** A silent 45 to 60 s recording of Option 2 (or of the storyboard
being clicked through), narrated live by the presenter.

**Claims supported.** Same as Option 2.

**Effort.** 1 to 2 h including two takes. Windows built-in tools only: Snipping Tool screen
recording or Xbox Game Bar (Win + Alt + R); no download. Optional GIF: frames from the MP4 with
Pillow (already installed) if a GIF is wanted inside the HTML slides; MP4 is simpler and smaller.

**Risk on stage.** Low. Embed it in the slide deck as a local file; test playback on the presenting
machine.

**Dependencies.** A working Option 2 once, before the talk.

**Reuses.** Option 2.

**Talk track.** Same as Option 2, spoken over the video. Practice with the video once so the
narration matches the clicks (a narrated recording reads as a demo; a recording with a foreign
voice-over does not).

### Option 4 (rank 4): interactive explainer of the query token

**What the audience sees.** A small page with the history as S boxes plus one extra box at the
end, the query token = a learned [MASK] vector plus one embedding per search field
(`src/nbp/models/smlprec_query.py`, `query_vector`). A stepper shows (1) embed history,
(2) build the query token from destination, dates, party, package, channel, (3) the MLP mixer mixes
across positions so the last position "reads" the history, (4) the last position scores all 100
clusters, (5) sameDest adds the earlier bookings at the searched destination (`w = 0.75` only for a
known destination; `alpha = 0.2`, tuned on valid, `smlprec_query_cold_train_es_run.json`).
Toggling a field in the query token can replay a precomputed example from Option 1 (no live model).

**Claims supported.** Methodology: why a session model with the search inside it can beat
CF and a context-only classifier (RQ1 mechanism). Good for a "how it works" slide.

**Effort.** 4 to 6 h (static HTML + inline SVG, stdlib build). Could also be one well-drawn static
slide in 1 to 2 h, which gives most of the value.

**Risk on stage.** Low (static).

**Dependencies.** None.

**Reuses.** The architecture slide content in `reports/summary/week4_rebuild/week4_clean_slides.html`
("Query token: 14 trường ...", "sameDest ..."), precomputed JSON of Option 1.

**Talk track (about 40 s).**
> "The original SMLP4Rec only sees the list of past clusters. We add one extra position at the end:
> the current search. Destination, dates, party size, package and channel each have an embedding,
> and their sum is that token. The mixer layers mix information across positions, so this last
> position combines the search with the history, and it scores all 100 clusters. On top, sameDest
> adds a bonus for what the customer booked before at the same destination. A new customer has an
> empty history, but the search token still works, which is why the model does not collapse to
> popularity for first bookings."

### Option 5 (rank 5, not recommended): resort-style mock booking page

**What the audience sees.** A generic hotel search form (destination, dates, adults, children,
package) that returns "recommended room / package types" as cluster cards.

**Why last.** The pull towards Vinpearl branding conflicts with CLAUDE.md §2 and with the rule that
nothing may impersonate a real brand: a page that looks like a Vinpearl booking flow invites the
reading that real Vinpearl data or a real integration exists. Clusters are anonymous, so the cards
would have to say "cluster 23" anyway, which makes the mock page look emptier than Option 2.
If the user still wants it: no Vinpearl name, logo, colours or photos; a permanent banner "Simulation
on the public Expedia Hotel Recommendations dataset. Cluster ids are anonymous proxies for room /
package type. No Vinpearl data."; generic title such as "Resort search (simulation)". Use it only as
a single static wireframe slide that says "how this could appear in a booking flow", not a live page.

**Claims supported.** Applicability / business framing only, no RQ evidence.

**Effort.** 4 to 6 h for a live page, 1 h for a wireframe slide.

**Risk on stage.** Low technically, high for framing (misreading as real data or a real product).

### Rejected: Streamlit / Gradio app

Neither is installed; adding one is a heavy dependency, which CLAUDE.md §9 says to ask about first.
It would give nicer widgets but nothing Option 2 cannot do in the existing stdlib server, and it
adds a second runtime that has to work with numpy 1.23 / pandas 1.5 / torch in the smlp4rec venv.
Not worth it for a short talk. Ask the user only if they want a hosted demo later.

## 2. Recommended combination

1. **Primary: Option 1 storyboard inside the slide flow** (3 bookings + 1 honest miss, about
   75 s). It cannot fail and carries the RQ1 message with real test bookings.
2. **One live moment: Option 2**, a single interaction (preset "returning customer", then
   "remove history" and "switch destination"), about 45 to 60 s, only if the time slot allows.
3. **Fallback: Option 3 recording** of exactly that interaction, embedded in the next slide.
   Switch rule: if the page is not responding within 10 to 15 s, or anything needs debugging, go to
   the video and narrate it. Never debug on stage.
4. **Optional: Option 4 as one static slide** before the demo, if the methodology slide does not
   already explain the query token.

Pre-talk checklist: start the server 5 minutes before the session (`python -u`); open the page
and press each preset once; disable sleep and notifications; projector resolution test; storyboard
HTML and MP4 copied locally (no network needed); a second copy on a USB stick.

Good-practice sources (web search 2026-10-11):
- Record a backup of each live demo, narrate it yourself, practise the switch, do not debug live:
  Greg Low, "How to avoid demo fails when presenting at tech conferences", https://www.red-gate.com/simple-talk/?p=111618
- Silent screencast narrated live; demo only what builds confidence:
  lornajane.net, "Live demo: risks and rewards", https://lornajane.net/posts/2017/live-demo-risks-and-rewards
- Pre-record demos; laptop, room and network may fail: DevConf speaker guide, https://www.devconf.info/us/speakerguide/
- Keep local copies of demos, videos and slides; test the AV setup 30 minutes early:
  HasGeek presenter guide, https://hasgeek.com/help/guide-for-speaking-at-events/sub/presenters-guide-for-speaking-at-meet-ups-and-conf-6tFvej5smE9Gs6GxRZKLkM
- Have a disaster-recovery plan for demos: https://oracle-base.com/blog/2014/01/02/public-speaking-tip-4-have-a-disaster-recovery-plan/
- Presentation tips from demo mistakes: https://haacked.com/archive/2011/04/18/presentation-tips.aspx/

No source found specifically on curating examples for ML research talks; the practice here
(fixed seeds, selection rules written down, one honest failure shown) follows from CLAUDE.md §5
and §9 (numbers computed by code, honest reporting).

## 3. Curated example bookings (computed)

### How they were computed

```
# 1. start the demo on the early-stopping checkpoint (prototype, does not edit demo_ui.py)
C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe -u scripts/demo_prototypes/serve_es.py --port 8766
# 2. scan test bookings drawn with seeds 0..N-1 through /api/seed and print candidates + what-ifs
python scripts/demo_prototypes/find_curated_examples.py --port 8766 --n-warm 3000 --n-cold 1500
```

Runtime about 3 min. Selection rules (in the script docstring): `cold` = L = 0, main rank ≤ 3,
ItemKNN rank > 5; `known` = warm, destination in history, main rank 1; `family12` = children ≥ 1,
check-in month 12, main top-5 hit; `miss` = warm, main rank > 10, ItemKNN rank 1. Candidates found:
464 cold, 360 known, 187 family12, 5 miss. Main model = early-stopping checkpoint (epoch 8) +
sameDest (`alpha` 0.2, `w` 0.75 for known destinations); baselines = notebook 07 export.
To reproduce a booking in the demo page: choose the row mode, type the seed, press
"Random test booking".

Caveat: these are picked for illustration (the main model is right in three of them by
construction). Say so on stage and show the miss; the representative numbers are the table in §0.

### Picks

| # | Role | Seed / rows | User, L | Search | Actual | Main rank | LR rank | ItemKNN rank |
|---|---|---|---|---|---|---|---|---|
| A | New customer (cold) | 960 / cold | 380617, L = 0 | dest 8254, month 2, 2 adults, 0 children, 1 room, no package | 23 | 1 | 26 | 75 (popularity fallback) |
| B | Returning, known destination | 0 / warm | 631266, L = 11 | dest 20845, month 2, 2 adults, 0 children | 31 | 1 | 3 | 10 |
| C | Family in December | 10 / warm | 801932, L = 8 | dest 8267, month 12, 2 adults, 3 children | 56 | 1 | 1 | 71 |
| C' | Family in December, sameDest reorders | 32 / warm | 705042, L = 6 | dest 8281, month 12, 2 adults, 2 children | 91 | 1 | 2 | 3 |
| D | Honest miss (sameDest overrides) | 2055 / warm | 692648, L = 2 | dest 12785, month 1, 2 adults | 37 | 37 | 47 | 1 |
| E | Spare cold, season effect | 1329 / cold | 104196, L = 0 | dest 8260, month 12, 2 adults | 77 | 1 | 8 | 32 |

What-ifs (main model top 5 before → after; computed by the same script and by one extra call for B):

- **A**: original top 5 `[23, 4, 50, 13, 39]`. "family (2+2)" → `[98, 4, 13, 19, 50]`, overlap 3/5,
  actual 23 drops to rank 7. "solo" keeps 23 first. Best live example of trip context.
- **B**: original `[31, 63, 76, 26, 61]`; cluster 31 has `model_p` 0.398 and sameDest share 1.0
  (final 0.977). "remove history" (blank user, same search) → `[76, 63, 26, 31, 61]`, actual rank 4.
  "switch destination" to 22076 (booked twice there, cluster 25) → 25 first; to 8254 (cluster 4
  there) → 4 first. Shows history + sameDest. Party / month what-ifs barely change it (overlap 4 to 5/5):
  for this user the history dominates, say so.
- **C**: original `[56, 41, 77, 23, 70]`. "solo" → `[23, 40, 50, 77, 56]`, overlap 3/5, actual 56
  drops to rank 5. "month 7" → overlap 4/5, 56 stays first.
- **C'**: model-only order of the top rows is 91 (0.151), 16 (0.134), 48 (0.077); sameDest (91 share
  0.588, 48 share 0.412: both booked before at dest 8281) moves 48 to second. Use it to show what
  sameDest does to an ordering rather than a single winner.
- **D**: history `37@d13559, 61@d12785`; the search is at 12785, so sameDest puts 61 first (final
  0.909) but the customer booked 37 again. ItemKNN, which ignores destination, gets it right. This is
  the honest limitation: the sameDest weight (0.75, tuned on valid) can override the model.
- **E**: "month 7" → `[49, 77, 48, 23, 19]`, actual 77 moves from 1 to 2: a small season effect.

Observation for the talk: most single-field what-ifs move the top 5 only a little (overlap 4 to
5/5 in most rows printed); the largest moves come from the party size on cold or no-sameDest users
(A, C) and from history / destination on returning users (B). Pick what-ifs from this list rather
than improvising on stage.

## 4. What the demo should say about each claim

| Claim | Shown by | Source of the number to quote |
|---|---|---|
| Main model beats both baselines on warm rows | Picks A to C + §0 table | `results/week4_rebuild/baselines_vs_final_es.json` |
| Cold users: main ≈ slightly above LR, far above ItemKNN | Pick A, E | same file, `cold (L=0)` |
| Context matters (party, season) | What-ifs A, C, E | printed by `find_curated_examples.py` |
| History / sameDest matters | B, C' | same |
| Limitation: sameDest can override | D | same |

## 5. Required change: point the demo at the early-stopping checkpoint

Minimal edit in `scripts/demo_ui.py` (the prototype `scripts/demo_prototypes/serve_es.py` does the
same by overriding `RUNS` at import, and was used for §3):

```python
RUNS = {  # model seed -> (experiment folder, run json)
    2022: ("2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es", "smlprec_query_cold_train_es_run.json"),
}
```

Checked: the early-stopping run json has the keys the `Runner` reads
(`samedest_hybrid.tuned_on_valid.A` with `alpha` 0.2 and `weights` `{0: [0, 0], 1: [0, 0.75]}`;
`splits.test.first_target_event` 2689344, the same split as the 3-epoch run), and the folder has
`best.pth` (epoch 8) and `vocab.npz`. The server started and served correctly with this setting.

Follow-ups in the same change:
1. Seeds 2023 / 2024 have no early-stopping run (PROGRESS.md open item). Either drop them (above)
   or keep them labelled "3-epoch" in the seed dropdown; do not mix them silently.
2. Page and docstring text: "SMLP4Rec + query token (01d, early stopping, best epoch 8) + sameDest".
3. `print(f"ready: ...", flush=True)` so the ready line shows when run in the background.
4. Optional, for consistency with the final table: export the 07b baselines instead of notebook 07.
   In `scripts/export_baselines_for_demo.py` set `SRC = .../baselines_es_models.joblib`. The 07b
   dump has `knn_similarity`, `lr`, `lr_encoder` but **no `pop` key**, which the demo uses for the
   ItemKNN cold fallback and tie-breaking. 07b's cold fallback is first-booking popularity of the
   training set, so add `pop_first` to the 07b dump (or compute it in the export script from the
   training rows, every booking up to event 2424705 with no prior booking) and save it as `pop`.
   Then rerun `find_curated_examples.py`, because baseline ranks in §3 will change.
5. Update the "Demo UI" entry in PROGRESS.md (checkpoint, baselines source).

## 6. Prototype files written for this note

- `scripts/demo_prototypes/serve_es.py`: launches `demo_ui` with the early-stopping checkpoint
  only (stdlib + the demo's own imports).
- `scripts/demo_prototypes/find_curated_examples.py`: stdlib client that scans the test split
  through the demo API and prints candidates and what-ifs (§3). Writes nothing.
