# Redesign Plan — Replace Rubric Scoring with Status-Enum Criteria

**Date**: 2026-09-21
**Status**: design draft only. No files deleted, no HTML edited yet — implementation is the next session.
**Target**: `reports/slides/week1_report_v2.html` (v1 stays frozen as the restore point).

---

## 1. Why the change, and what replaces the score

The rubric produced one number per dataset (Expedia 79, Trivago 77, Airbnb 46 …). Three problems:

- **False precision.** 79 vs 77 reads as a meaningful gap; it is two points on a 48-point weighted sum, well inside judgment noise. The tie-break was never really arithmetic.
- **Hidden reasoning.** The number absorbs ten criteria and buries the one fact that actually decided the choice (Expedia's two-year span).
- **Unfalsifiable.** A reader cannot check 79 without re-deriving weights. A reader *can* check "booking signal: yes/no".

Replacement principle: **verdict + evidence, never a sum.**

Each criterion gets a status enum *and* the measured fact that produced it. This is strictly more transparent than the score: the reader sees the judgment, the evidence behind it, and can disagree with one cell without the whole ranking collapsing. Nothing is added, multiplied, or ranked.

A second honesty gain: the matrix will openly show **Expedia losing four criteria** (task fit, item metadata, feasibility, license). The old score hid that. The decision rule below explains why it still wins.

---

## 2. Status enum vocabulary

Four states, reusing the existing tag CSS so the visual language does not change.

| Enum (VI) | Enum (EN) | CSS class | Meaning |
|---|---|---|---|
| **Phù hợp** | Suitable | `.tag-core` (green) | Criterion fully satisfied by measured evidence |
| **Cân nhắc** | Consider | `.tag-sec` (amber) | Usable with a stated caveat or workaround |
| **Không phù hợp** | Not suitable | `.tag-drop` (red) | Criterion not satisfied; blocking if in Group A (§4) |
| **Không áp dụng** | N/A | `.tag-na` (new, gray) | Field/concept does not exist in this dataset |

One new CSS class needed:

```css
.tag-na { background: var(--bg-card); color: var(--text-sub); border: 1px solid var(--border); }
```

Dataset-level verdict is a separate enum (replaces the old Core / Secondary / Drop tiers, which were score-derived):

**Đã chọn** (Selected) · **Dự phòng** (Backup) · **Ngoài phạm vi** (Out of scope) · **Loại** (Rejected) · **Chỉ thử pipeline** (Prototype only)

---

## 3. The criteria set

Derived from the old C1–C10, renamed so each is self-explanatory (no opaque codes, no weights). **C9 "cross-sell linkability" is dropped entirely** — RQ3 is out of scope per CLAUDE.md §1, so scoring it was noise. This also removes the need for the old C9 sensitivity test.

| # | Criterion (VI) | Question it answers | Evidence source |
|---|---|---|---|
| 1 | Khớp bài toán | Is the target a direct top-K next-item task? | data card + Step 0 mapping |
| 2 | Định danh người dùng | Is there a stable, real user ID? | `dataset_benchmark.json` |
| 3 | Chiều sâu lịch sử | Do users repeat enough to model a sequence? | `dataset_benchmark.json` |
| 4 | Độ dài thời gian & mùa vụ | Does the span cover a seasonal cycle? | `dataset_benchmark.json` |
| 5 | Ngữ cảnh chuyến đi | Are trip-context fields present and real? | `expedia_schema_table.md`, data cards |
| 6 | Chất lượng tín hiệu | Is the signal a purchase, or something weaker? | data cards |
| 7 | Metadata sản phẩm | Is there item text/attributes (for LLM rerank)? | `trivago_profile.json`, data cards |
| 8 | Chất lượng dữ liệu | How much of the file is defective? | `dataset_benchmark.json`, `expedia_anomaly_audit.json` |
| 9 | Tính khả thi | Does it run on the dev machine in the time budget? | file sizes, chunked-pass runtimes |
| 10 | Giấy phép | Does the license permit research use? | data cards |

---

## 4. The decision rule (explicit, non-arithmetic)

Stated on the decision slide so the reader can re-apply it by eye:

> **Bước 1 — Điều kiện bắt buộc.** Bộ dữ liệu phải đạt cả 5 yêu cầu nền (§5, Slide 1). Không đạt bất kỳ yêu cầu nào → loại, không xét tiếp.
>
> **Bước 2 — Tiêu chí chặn (Nhóm A).** Không được có trạng thái *Không phù hợp* ở: khớp bài toán, định danh người dùng, chiều sâu lịch sử, độ dài thời gian & mùa vụ, chất lượng tín hiệu, chất lượng dữ liệu.
>
> **Bước 3 — Tiêu chí hỗ trợ (Nhóm B).** Ngữ cảnh, metadata, khả thi, giấy phép: mô tả đánh đổi, **không dùng để loại**.
>
> **Bước 4 — Kết luận.** Chỉ Expedia không có *Không phù hợp* nào ở Nhóm A → **Đã chọn**. Trivago chỉ trượt đúng **một** tiêu chí chặn (độ dài thời gian) → **Dự phòng**, kích hoạt nếu RQ1 đổi sang bài toán session-based.

Why this is better than the tie-break it replaces: it names the *single* criterion separating Expedia from Trivago, and makes the contingency condition falsifiable — if the research question stops needing seasonality, the verdict flips by rule, not by re-scoring.

---

## 5. Level-1 screening: hard requirements (replaces the score table on Slide 1)

Five binary requirements, from the old gates but reframed so each is directly observable:

- **R1** Định danh actor thật (không phải proxy: tài xế, trạm xe, vùng)
- **R2** Item ID ổn định, gợi ý được
- **R3** Có mốc thời gian tuyệt đối
- **R4** Giấy phép cho phép dùng nghiên cứu
- **R5** Tải được & tái lập bằng script

| Bộ dữ liệu | R1 | R2 | R3 | R4 | R5 | Kết luận | Lý do chặn |
|---|---|---|---|---|---|---|---|
| Expedia | ✓ | ✓ | ✓ | ✓ | ✓ | Vào vòng so sánh | — |
| Trivago 2019 | ✓ | ✓ | ✓ | ✓ | ✓ | Vào vòng so sánh | — |
| Airbnb New User | ✓ | ✓ | ✓ | ✓ | ✓ | Vào vòng so sánh | — |
| Hotel Booking Demand | **✗** | ✓ | ✓ | ✓ | ✓ | **Loại** | Không có ID khách dưới mọi hình thức |
| Akeed | ✓ | ✓ | ✓ | ✓ | ✓ | Ngoài phạm vi (miền đồ ăn) | — |
| Instacart | ✓ | ✓ | **✗** | ✓ | ✓ | **Loại** | Không có ngày lịch tuyệt đối |
| Yelp | ✓ | ✓ | ✓ | ✓ | ✓ | Ngoài phạm vi (miền đồ ăn) | — |
| Porto Taxi | **✗** | ✓ | ✓ | ✓ | ✓ | **Loại** | `TAXI_ID` là tài xế, không phải hành khách |
| Citi Bike | **✗** | ✓ | ✓ | ✓ | ✓ | **Loại** | Không có ID người đi |
| NYC TLC | **✗** | ✓ | ✓ | ✓ | ✓ | **Loại** | Chỉ có vùng đón/trả |
| MovieLens | ✓ | ✓ | ✓ | ✓ | ✓ | Chỉ thử pipeline | Không có ngữ cảnh du lịch |

Story this tells: **4/11 bộ trượt ngay yêu cầu định danh actor; 1 bộ trượt yêu cầu thời gian.** Far more legible than an 11-row score column.

---

## 6. Level-2 criteria matrix (NEW slide — the core of the redesign)

Four travel candidates × 10 criteria. Each cell = enum badge + evidence (measured this session unless noted).

| Tiêu chí | Expedia | Trivago 2019 | Airbnb | Hotel Booking Demand |
|---|---|---|---|---|
| 1. Khớp bài toán | **Cân nhắc** — 100 cụm, top-K trực tiếp nhưng cụm ≠ hạng phòng (proxy) | **Phù hợp** — clickout từ impression list, đúng dạng top-K | **Không phù hợp** — 12 lớp quốc gia, 1 sự kiện/user | **Không phù hợp** — không có khoá cá nhân hoá |
| 2. Định danh người dùng | **Phù hợp** — 1,198,786 user; 813,985 có booking | **Cân nhắc** — 730,803 user nhưng 83.55% chỉ 1 session | **Cân nhắc** — có ID, đúng 1 sự kiện/user | **Không phù hợp** — không có ID khách |
| 3. Chiều sâu lịch sử | **Cân nhắc** — trung vị 2.0 booking; 61.22% user ≥2 | **Cân nhắc** — 56.43% user chỉ 1 clickout; chỉ dùng được ở mức session | **Không phù hợp** — 0% user có ≥2 | **Không áp dụng** — không có actor |
| 4. Độ dài thời gian & mùa vụ | **Phù hợp** — 723 ngày, 2 chu kỳ năm | **Không phù hợp** — 5 ngày (2018-11-01→06) | **Cân nhắc** — 1,641 ngày nhưng không có chuỗi/user | **Cân nhắc** — 1,063 ngày, không gắn được vào user |
| 5. Ngữ cảnh chuyến đi | **Phù hợp** — party/ngày/combo/điểm đến, 100% coverage trên booking | **Cân nhắc** — thiết bị, thành phố, giá; không có quy mô đoàn | **Không phù hợp** — không có ngữ cảnh chuyến đi | **Phù hợp** — party, ngày, giá, kênh, bữa ăn |
| 6. Chất lượng tín hiệu | **Phù hợp** — 3,000,693 booking thật, tách được click | **Cân nhắc** — clickout, không phải giao dịch mua | **Cân nhắc** — 58.35% user không đặt (NDF) | **Cân nhắc** — bản ghi đặt phòng, 37.04% huỷ |
| 7. Metadata sản phẩm | **Không phù hợp** — cụm ẩn danh, không có text | **Phù hợp** — 927,142 item, trung vị 15 tag, 157 thuộc tính | **Không phù hợp** — 12 quốc gia | **Cân nhắc** — loại phòng, bữa ăn, kênh |
| 8. Chất lượng dữ liệu | **Phù hợp** — 1.51% ô thiếu; 21/24 cột đầy đủ | **Cân nhắc** — 22.74% ô thiếu | **Cân nhắc** — 6.40%; `date_first_booking` thiếu 58.35% | **Không phù hợp** — 26.80% dòng trùng lặp hoàn toàn |
| 9. Tính khả thi | **Cân nhắc** — 4.07 GB, phải quét chunked | **Cân nhắc** — 2.10 GB, phải quét chunked | **Phù hợp** — 24.9 MB | **Phù hợp** — 16.9 MB |
| 10. Giấy phép | **Cân nhắc** — Kaggle rules, chỉ nghiên cứu | **Cân nhắc** — bản mirror, chưa xác minh | **Cân nhắc** — Kaggle rules | **Phù hợp** — CC BY 4.0 |

Reading of the matrix (goes on the slide as the insight strip): Expedia thắng **không phải vì tốt nhất ở mọi tiêu chí** — nó thua ở 4 tiêu chí — mà vì là ứng viên **duy nhất không có "Không phù hợp" nào trong 6 tiêu chí chặn**.

---

## 7. New slide sequence

| # | Slide | Origin | Action |
|---|---|---|---|
| 1 | Sàng lọc 11 bộ theo 5 điều kiện bắt buộc | old 1 | **Rebuild** — bỏ cột Điểm, bỏ pill Rubric, bỏ nút "Xem Phụ Lục A (Rubric)"; thay bằng bảng R1–R5 (§5) |
| 2 | Quyết định miền: Du lịch | old 2 | **Minor edit** — cột "Gọi xe" trích yêu cầu R1 đã trượt thay vì ngụ ý điểm số |
| 3 | Ma trận tiêu chí × 4 ứng viên du lịch | — | **NEW** (§6) |
| 4 | Quyết định & điều kiện kích hoạt dự phòng | — | **NEW** — quy tắc 4 bước (§4) + điều kiện đổi sang Trivago |
| 5 | Bằng chứng định lượng (6 bộ, số đo trực tiếp) | old 4 | **Keep**, chuyển vị trí; thêm hàng enum verdict trên đầu để nối với Slide 3 |
| 6 | Định nghĩa bài toán | old 3 | Keep |
| 7–8 | Schema I & II | old 5–6 | Keep |
| 9–12 | EDA trực quan ×4 | old 7–10 | Keep |
| 13 | Ngữ cảnh có giúp ích? | old 11 | Keep |
| 14 | Mô hình & giao thức đánh giá | old 12 | Keep |
| 15 | Giới hạn cần công khai | old 13 | Keep |
| **Phụ lục A** | Định nghĩa tiêu chí & thang trạng thái | old Appendix A (rubric) | **Replace** — legend 4 enum, 10 định nghĩa tiêu chí, Nhóm A/B, quy tắc quyết định, nguồn bằng chứng |
| **Phụ lục B** | Trivago — hồ sơ dự phòng chi tiết | old 14 | **Move to appendix** — bỏ "(Điểm: 77/100)", đổi thành nhãn **Dự phòng** |

Net: 15 main slides (was 14) + 2 appendices (was 1). The narrative becomes a funnel: **yêu cầu bắt buộc → miền → tiêu chí → quyết định → bằng chứng số**.

### JS changes this implies (`week1_report_v2.html`)

- `MAIN_SLIDES_COUNT` 14 → **15**
- `RUBRIC_SLIDE_INDEX` (single int) → **`APPENDIX_INDICES = [15, 16]`**; counter label logic must render "Phụ Lục A" / "Phụ Lục B" instead of one hardcoded string
- `pillJumpRubric` / `btnLinkToRubric` → rename to `pillJumpCriteria` / `btnLinkToCriteria`, repoint to Phụ lục A
- `btnBackToSlide1` stays; add the same button to Phụ lục B
- Footer `Trang NN / 14` → `/ 15` on all main slides
- Add `.tag-na` to the stylesheet (§2)

---

## 8. Files: delete, migrate, edit

### 8.1 Delete (6 files)

| File | Note |
|---|---|
| `docs/dataset_rubric.md` | method: gates, weights, formula |
| `docs/dataset_scores.md` | scoring sheet — **contains non-scoring content that must be migrated first (§8.2)** |
| `docs_vi/dataset_rubric.md` | VI mirror |
| `docs_vi/dataset_scores.md` | VI mirror — same migration caveat |
| `scripts/dataset_scoring.py` | `score_dataset()` + `quick_profile()`; verified: nothing imports either |
| `reports/summary/travel_rubric_without_c9.txt` | C9 sensitivity output — obsolete once C9 is dropped as a criterion |

### 8.2 Migrate before deleting — `dataset_scores.md` is not purely scoring

Three sections in it are genuinely useful and are **not** rubric arithmetic. Proposal: create **`docs/dataset_selection.md`** (+ VI mirror) as the single source the new slides cite, carrying:

1. **Step 0 interaction-definition table** — per-dataset `user_id` / `item_id` / what counts as an interaction / sequence unit. Load-bearing for Phase 2; nothing else documents it.
2. **Per-dataset context-field inventory** (`ctx_*` mapping) — load-bearing for the Phase-4 ablation.
3. **Weather-join feasibility test** (Trivago feasible / Akeed infeasible, tested 2026-09-17).
4. Plus the new content: the R1–R5 requirement table (§5), the criteria matrix (§6), the decision rule (§4).

Dropped on purpose: the scoring sheet, the per-score notes, the score-correction history, the RQ3 link-fields table (out of scope). The substantive findings inside those notes (Yelp `review_count` trap, Instacart no-calendar-date) already live in `reports/summary/eda_summary.md` §3 and stay there.

### 8.3 Edit (references to dead files)

| File | What to change |
|---|---|
| `CLAUDE.md` | §5 lines ~62/64/68/81 (rubric + scores refs, "scored after EDA"); §6 repo tree lines ~101/108/109; §7 line ~132 and §10 line ~167 (both cite `docs/dataset_scores.md` as a report target) — repoint to `docs/dataset_selection.md` |
| `README.md` | tree lines ~20/21/25/26/39 — drop rubric/scores/scoring-script entries, add `dataset_selection.md` |
| `reports/summary/eda_summary.md` | title "…and Scoring"; §1 methodology paragraph; §2 table **Score column**; line ~27 "Full scoring breakdown"; line ~85 score-correction note; file list ~89–90 |
| `docs/data_cards/airbnb_new_user.md` | line 40 "under the rubric … Scored C1=1" → restate as enum verdict |
| `docs/data_cards/akeed.md` | line 36 cross-ref to the weather note → repoint |
| `docs_vi/data_cards/{airbnb_new_user,akeed,movielens}.md` | same, VI |
| `reports/slides/week1_report_v2.html` | §7 above (next session) |

---

## 9. Open decisions for you

1. **When to delete.** The deck currently cites `docs/dataset_rubric.md` and `docs/dataset_scores.md` in two footers and has a whole rubric appendix. Deleting now leaves the committed deck pointing at dead files until the HTML lands. **Recommendation: delete + migrate + rewrite the HTML in one session**, so the repo is never inconsistent. Say the word if you'd rather I delete immediately anyway.
2. **`quick_profile()`.** It ships inside `dataset_scoring.py` but is a generic profiling helper, unrelated to scoring, and currently unused. Delete with the file, or keep it by moving it to `scripts/`-level utility? Recommendation: delete — the per-dataset profiling scripts written this session supersede it.
3. **Trivago slide placement.** Moved to Phụ lục B in this plan. Alternative: keep it as main slide 16. Recommendation: appendix, since Slide 4 now carries the switch trigger.
4. **Enum wording.** Currently Phù hợp / Cân nhắc / Không phù hợp / Không áp dụng. If the mentor prefers Đạt / Hạn chế / Không đạt, it is a find-replace — worth fixing before implementation, not after.
