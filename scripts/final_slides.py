"""Final slide deck v1 (HTML, 16:9) built from `results/week5_final/` and the week-4 run files.

Every number on a slide is read from a results file here; nothing is typed. Charts are inline SVG
in the style-guide palette (`docs/plans/final_prep/03_slide_style_guide.md`); each chart is also
written to `results/figures/final/en/<name>.svg`.

Titles-only outline (the argument, read in order):
 1. Search-aware next-booking recommendation on public Expedia data (title)
 2. A resort has to pick, at search time, which room or package to show
 3. We test whether a search-aware sequence model beats classical recommenders, and at what cost
 4. Expedia gives {bookings} bookings from {users} users, but {single} booked only once
 5. The searched destination says far more about the booked cluster than any other field
 6. Every model is scored on one temporal split, full ranking, warm and first-booking rows apart
 7. The original SMLP4Rec reads only past bookings and ignores the current search
 8. Our query token puts the 14 search fields into the model as one extra position
 9. sameDest adds what the guest booked before at the searched destination
10. Training with the {first} first bookings teaches the model the empty-history case
11. The query token carries most of the gain, sameDest adds a smaller, real step
12. The main model leads both baselines on all, cold and warm test rows
13. On first bookings only the search-aware models rank well
14. The lead over logistic regression grows with history, from {L=0 gap} to {L>=10 gap} NDCG@5
15. History pays where guests return: sameDest lifts known-destination NDCG@5 from {a} to {b}
16. The model order is the same for every party type and for package trips
17. Accuracy costs no global diversity, the trade-off is narrower lists per destination
18. RQ1 is answered for sparsity and trip context but not for season, RQ4 for bias and coverage
19. The demo replays real test bookings through the main model and both baselines
20. The current search is the main signal, history a smaller and real increment (+ limits)
A1. A stronger logistic regression narrows the warm gap but does not close it
A2. Shuffling the search destroys accuracy, masking the history costs much less

Output: `reports/summary/week5_final/final_slides_v1.html` (self-contained apart from the web
font, which falls back to Segoe UI / Arial). Keys: arrows / space / PageUp / PageDown, Home, End,
`n` toggles speaker notes. Print (or headless Edge `--print-to-pdf`) gives one 1280x720 page per
slide.

Limitations stated on the slides: `hotel_cluster` is a proxy for room category / package; one
dataset; offline only; no public in-stay data; only two simple baselines; the early-stopping runs
are one seed.

Language: `--lang en` (default) or `--lang vi`. Every visible English string is written once as
`T("english", "tiếng Việt")`; numbers come from the same f-string fields in both languages, so a
number can never differ between the two decks. Vietnamese output:
`reports/summary/week5_final/final_slides_v1_vi.html`, figures in `results/figures/final/vi/`.

Usage: .venv/Scripts/python.exe scripts/final_slides.py [--lang en|vi]
"""

from __future__ import annotations

import argparse
import html
import json
import re

import pandas as pd

from nbp.paths import ROOT

W5 = ROOT / "results" / "week5_final"
W4 = ROOT / "results" / "week4_rebuild"
LANG = "en"  # set by main(); read by T() and the path helpers below

# palette (03_slide_style_guide.md, section b)
BG, INK, MUTED, RULE = "#ffffff", "#20242a", "#5c6166", "#d9dadb"
MAIN, MAIN_DARK, CONTRAST, CONTEXT, ALERT = "#3b6ea5", "#2f5a88", "#c27c3f", "#a8a79f", "#a63d32"
FONT_EN = "'Source Sans 3', 'Segoe UI', Arial, sans-serif"
FONT_VI = "'Segoe UI', Arial, sans-serif"  # system stack: full Vietnamese diacritics offline
WORD_LIMIT = 40  # English words; Vietnamese is counted in syllables, hence the larger limit
WORD_LIMIT_VI = 62
TITLE_LIMIT, TITLE_LIMIT_VI = 17, 30
BANNED_WORDS = ("leverage", "seamless", "delve", "unlock", "robust", "elevate", "comprehensive")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def T(en, vi):
    """Return the string (or number, e.g. a label width) for the language being built."""
    return vi if LANG == "vi" else en


def font() -> str:
    return FONT_VI if LANG == "vi" else FONT_EN


def fig_dir():
    return ROOT / "results" / "figures" / "final" / LANG


def out_path():
    suffix = "_vi" if LANG == "vi" else ""
    return ROOT / "reports" / "summary" / "week5_final" / f"final_slides_v1{suffix}.html"


def jload(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_data() -> dict:
    bench = pd.read_csv(W5 / "final_benchmark.csv")
    hl = bench[bench["seed"].astype(str) == "2022"].set_index(["model_key", "row_set"])
    boot = pd.read_csv(W5 / "bootstrap_ci.csv")
    d = {
        "hl": hl,
        "boot": boot,
        "L": pd.read_csv(W5 / "slices_history_length.csv"),
        "known": pd.read_csv(W5 / "slices_destination_known.csv"),
        "party": pd.read_csv(W5 / "slices_party_type.csv"),
        "pkg": pd.read_csv(W5 / "slices_is_package.csv"),
        "ba": pd.read_csv(W5 / "beyond_accuracy_aggregate.csv"),
        "ba_dest": pd.read_csv(W5 / "beyond_accuracy_per_destination.csv"),
        "lr07": pd.read_csv(W5 / "sensitivity_logreg_nb07.csv"),
        "final": jload(W5 / "final_results.json"),
        "eda": jload(W4 / "eda_clean.json"),
        "counts": jload(W4 / "bookings_counts.json"),
        "run": jload(W4 / "smlprec_query_cold_train_es_run.json"),
        "base": jload(W4 / "baselines_es_run.json"),
        "plain_es": jload(W4 / "smlprec_plain_cold_train_es_run.json"),
        "seeds": jload(W4 / "seeds_summary.json"),
    }
    demo = W5 / "demo_storyboard.json"
    d["demo"] = jload(demo) if demo.exists() else None
    return d


def m(d: dict, key: str, rs: str, metric: str) -> float:
    return float(d["hl"].loc[(key, rs), metric])


def ci(d: dict, rs: str, a: str, b: str, metric: str) -> tuple[float, float, float]:
    bo = d["boot"]
    r = bo[(bo.row_set == rs) & (bo.a == a) & (bo.b == b) & (bo.metric == metric)].iloc[0]
    return float(r["diff"]), float(r["ci95_lo"]), float(r["ci95_hi"])


def f3(x: float) -> str:
    return f"{x:.3f}"


def sgn3(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def pct2(x: float) -> str:
    return f"{100 * x:.2f}%"


def num(n: int) -> str:
    return f"{int(n):,}"


def ci_txt(c: tuple[float, float, float]) -> str:
    return T(
        f"{sgn3(c[0])} (95% CI {sgn3(c[1])} to {sgn3(c[2])})",
        f"{sgn3(c[0])} (KTC 95% {sgn3(c[1])} đến {sgn3(c[2])})",
    )


# ----------------------------------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------------------------------
FIGURES: dict[str, str] = {}


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def txt(x, y, s, size=18, fill=INK, anchor="start", weight=400, italic=False) -> str:
    st = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}"{st}>{s}</text>'
    )


def rect(x, y, w, h, fill="none", stroke="none", sw=1.5, dash=False) -> str:
    da = ' stroke-dasharray="6 5"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{da}/>'
    )


def line(x1, y1, x2, y2, stroke=MUTED, sw=1.5, arrow=False, dash=False) -> str:
    mk = ' marker-end="url(#arr)"' if arrow else ""
    da = ' stroke-dasharray="5 5"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}"{mk}{da}/>'
    )


DEFS = (
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/>'
    "</marker></defs>"
)


def svg(name: str, w: int, h: int, inner: str, label: str) -> str:
    """Inline SVG (text in the page font); also saved as a standalone file under FIG."""
    # marker ids must be unique per SVG: a marker inside a hidden slide does not render elsewhere
    mid = f"arr-{name}"
    defs = DEFS.replace('id="arr"', f'id="{mid}"')
    inner = inner.replace("url(#arr)", f"url(#{mid})")
    body = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{esc(label)}">{defs}{inner}</svg>'
    )
    FIGURES[name] = body.replace("<svg ", f'<svg style="font-family:{font()};background:{BG}" ', 1)
    return body


def hbars(name, rows, w, label_w, vmax, fmt, label, bar_h=34, gap=20, top=8, val_w=110) -> str:
    """Horizontal bars, labelled directly. rows: dicts label, value, color, note (optional)."""
    span = w - label_w - val_w - 220
    out, y = [], top
    for r in rows:
        bw = max(2.0, span * r["value"] / vmax)
        out.append(txt(label_w - 16, y + bar_h * 0.68, r["label"], 19, r.get("lc", INK), "end"))
        out.append(rect(label_w, y, bw, bar_h, r["color"]))
        out.append(
            txt(
                label_w + bw + 10,
                y + bar_h * 0.68,
                fmt(r["value"]),
                19,
                r.get("vc", INK),
                weight=600,
            )
        )
        if r.get("note"):
            out.append(
                txt(label_w + bw + 10 + val_w * 0.62, y + bar_h * 0.68, r["note"], 17, MUTED)
            )
        y += bar_h + gap
    out.insert(0, line(label_w, top - 6, label_w, y - gap + 6, RULE, 1))
    return svg(name, w, int(y - gap + top + 4), "".join(out), label)


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def fig_users(d: dict) -> str:
    e2 = d["eda"]["E2"]
    s1, s5, s10 = e2["share_single_booking"], e2["share_5_plus"], e2["share_10_plus"]
    rows = [
        {"label": T("1 booking", "1 lượt đặt"), "value": s1, "color": INK},
        {"label": T("2 to 4 bookings", "2 đến 4 lượt đặt"), "value": 1 - s1 - s5, "color": CONTEXT},
        {"label": T("5 to 9 bookings", "5 đến 9 lượt đặt"), "value": s5 - s10, "color": CONTEXT},
        {"label": T("10 or more", "từ 10 lượt trở lên"), "value": s10, "color": CONTEXT},
    ]
    return hbars(
        "users_by_bookings",
        rows,
        760,
        T(200, 230),
        0.45,
        pct,
        T("Share of users by bookings", "Tỷ lệ người dùng theo số lượt đặt"),
        44,
        22,
    )


def fig_fields(d: dict) -> str:
    ig = d["eda"]["E6"]["held_out_information_gain_bits_on_valid"]
    names = {
        "dest": T("destination", "điểm đến"),
        "country": T("user country", "quốc gia người dùng"),
        "site": T("site", "site"),
        "posa": T("point-of-sale continent", "châu lục điểm bán"),
        "lead": T("days to check-in", "số ngày đến check-in"),
        "stay": T("nights", "số đêm"),
        "dest_type": T("destination type", "loại điểm đến"),
        "package": T("package", "gói"),
    }
    rows = [
        {"label": names[k], "value": ig[k], "color": INK if k == "dest" else CONTEXT} for k in names
    ]
    return hbars(
        "fields_information_gain",
        rows,
        900,
        T(260, 290),
        ig["dest"] * 1.05,
        lambda v: f"{v:.2f}",
        "bits",
        30,
        14,
    )


def fig_timeline(d: dict) -> str:
    sp = d["eda"]["splits"]
    w, h = 1152, 300
    segs = [  # name, x, width, warm, cold, dates
        ("train", 0, 700, sp["train"]["rows"], sp["first_train"]["rows"], sp["train"]),
        ("valid", 724, 200, sp["valid"]["rows"], sp["cold_valid"]["rows"], sp["valid"]),
        ("test", 948, 200, sp["test"]["rows"], sp["cold_test"]["rows"], sp["test"]),
    ]
    out = []
    for name, x, sw, warm, cold, dt in segs:
        col = INK if name == "test" else CONTEXT
        out.append(rect(x, 60, sw, 54, col))
        out.append(txt(x + 14, 95, name, 21, "#ffffff", weight=600))
        out.append(
            txt(
                x,
                40,
                T(
                    f"{dt['first_date']} to {dt['last_date']}",
                    f"{dt['first_date']} – {dt['last_date']}",
                ),
                17,
                MUTED,
            )
        )
        out.append(txt(x, 152, T(f"{num(warm)} warm targets", f"{num(warm)} mục tiêu warm"), 19))
        out.append(
            txt(
                x,
                180,
                T(f"{num(cold)} first bookings", f"{num(cold)} lượt đặt đầu tiên"),
                19,
                MUTED,
            )
        )
    out.append(
        txt(
            0,
            240,
            T(
                "warm = the guest has at least one earlier booking (L ≥ 1)",
                "warm = khách đã có ít nhất một lượt đặt trước đó (L ≥ 1)",
            ),
            17,
            MUTED,
        )
    )
    out.append(
        txt(
            0,
            266,
            T(
                "first booking = empty history (L = 0, scored separately)",
                "lượt đặt đầu tiên = lịch sử rỗng (L = 0, chấm riêng)",
            ),
            17,
            MUTED,
        )
    )
    return svg("split_timeline", w, h, "".join(out), T("Temporal split", "Chia theo thời gian"))


def fig_arch(d: dict, stage: int) -> str:
    """SMLP4Rec diagram in the layout of the paper figure: 1 = history only, 2 = + query token,
    3 = + sameDest. Input = cluster IDs only (no brand / category), 2 layers sharing one block."""
    run = d["run"]
    nf = len(run["query_fields"])
    tv = run["samedest_hybrid"]["tuned_on_valid"]["A"]
    w_new, w_known = tv["weights"]["1"]
    w, h = 1152, 440
    o: list[str] = []

    def vtxt(cx, cy, s, size=18, fill=INK, weight=400):
        return f'<g transform="rotate(-90 {cx} {cy})">{txt(cx, cy, s, size, fill, "middle", weight)}</g>'

    def box(x, y, bw, bh, fill="#ffffff", stroke=INK, sw=1.4, dash=False, rx=0):
        r = rect(x, y, bw, bh, fill, stroke, sw, dash)
        return r.replace("<rect ", f'<rect rx="{rx}" ', 1) if rx else r

    # ---- input: past bookings are cluster IDs only -----------------------------------------
    o.append(txt(30, 24, T("past bookings (up to 20)", "lượt đặt trước (tối đa 20)"), 16, MUTED))
    hist = [(44, T("booking 1", "lượt 1")), (106, T("booking 2", "lượt 2"))]
    hist.append((204, T("booking L", "lượt L")))
    for y, lab in hist:
        o.append(box(30, y, 150, 52, "#f2f2f1", MUTED, 1.2))
        o.append(txt(105, y + 24, lab, 18, INK, "middle"))
        o.append(txt(105, y + 43, "cluster ID", 16, MUTED, "middle"))
        o.append(line(180, y + 26, 203, y + 26, MUTED, 1.3, arrow=True))
    o.append(txt(105, 186, "⋮", 20, MUTED, "middle"))
    if stage == 1:
        o.append(box(30, 276, 150, 52, "none", CONTEXT, 1.5, dash=True))
        o.append(txt(105, 307, T("current search", "tìm kiếm hiện tại"), 17, MUTED, "middle"))
        o.append(
            txt(
                30,
                350,
                T("ignored by the original", "mô hình gốc bỏ qua"),
                16,
                MUTED,
                italic=True,
            )
        )
    else:
        o.append(box(30, 276, 150, 52, MAIN, MAIN, 1.4))
        o.append(txt(105, 308, "query token", 18, "#ffffff", "middle", 600))
        o.append(line(180, 302, 203, 302, MAIN, 1.8, arrow=True))
        o.append(
            txt(
                30,
                350,
                T(f"sum of {nf} field embeddings", f"tổng {nf} embedding trường"),
                16,
                MAIN_DARK,
                weight=600,
            )
        )
    # ---- embedding layer ---------------------------------------------------------------------
    o.append(box(205, 44, 34, 284, "#ececeb", MUTED, 1.2))
    o.append(vtxt(222, 186, T("Embedding layer", "Lớp embedding"), 18))
    # ---- main operation layer ----------------------------------------------------------------
    o.append(box(262, 20, 540, 400, "none", INK, 1.5))
    o.append(
        txt(
            532,
            52,
            T(
                "Main operation layer (× 2, one shared block)",
                "Lớp xử lý chính (× 2, dùng chung một khối)",
            ),
            19,
            INK,
            "middle",
            600,
        )
    )
    # block (sequence x channel x feature), last row = readout position
    ro = CONTEXT if stage == 1 else MAIN
    o.append(f'<polygon points="292,150 316,126 396,126 372,150" fill="#f2f2f1" stroke="{MUTED}" stroke-width="1.2"/>')  # fmt: skip
    o.append(f'<polygon points="372,150 396,126 396,206 372,230" fill="#e4e4e2" stroke="{MUTED}" stroke-width="1.2"/>')  # fmt: skip
    o.append(rect(292, 150, 80, 80, "#f2f2f1", MUTED, 1.2))
    o.append(rect(292, 210, 80, 20, ro, MUTED, 1.2))
    for gx in (312, 332, 352):
        o.append(line(gx, 150, gx, 230, MUTED, 1))
    for gy in (170, 190):
        o.append(line(292, gy, 372, gy, MUTED, 1))
    o.append(vtxt(281, 190, "sequence", 16, MUTED))
    o.append(txt(332, 252, "channel", 16, MUTED, "middle"))
    o.append(txt(372, 112, "feature", 16, MUTED, "middle"))
    o.append(line(400, 190, 426, 190, MUTED, 1.4, arrow=True))
    # layer norm, then three parallel mixing modules
    o.append(box(428, 110, 30, 160, "#ececeb", MUTED, 1.2))
    o.append(vtxt(445, 190, "Layer Norm", 17))
    o.append(line(458, 190, 484, 190, MUTED, 1.4))
    o.append(line(484, 109, 484, 281, MUTED, 1.4))
    mods = [
        (109, "Sequence mixing", T("across positions", "giữa các vị trí")),
        (190, "Feature mixing", T("1 feature only (cluster ID)", "chỉ 1 đặc trưng (cluster ID)")),
        (281, "Channel mixing", T("across hidden units", "giữa các đơn vị ẩn")),
    ]
    for my, name, sub in mods:
        o.append(line(484, my, 502, my, MUTED, 1.4, arrow=True))
        o.append(box(504, my - 29, 230, 58))
        o.append(txt(619, my - 3, name, 18, INK, "middle", 600))
        o.append(txt(619, my + 18, sub, 16, MUTED, "middle"))
    o.append(line(734, 109, 762, 109, MUTED, 1.4))
    o.append(line(762, 109, 762, 174, MUTED, 1.4, arrow=True))
    o.append(line(734, 281, 762, 281, MUTED, 1.4))
    o.append(line(762, 281, 762, 206, MUTED, 1.4, arrow=True))
    o.append(line(734, 190, 746, 190, MUTED, 1.4, arrow=True))
    o.append(f'<circle cx="762" cy="190" r="14" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>')
    o.append(line(750, 190, 774, 190, INK, 1.2))
    o.append(line(762, 178, 762, 202, INK, 1.2))
    o.append(line(776, 190, 820, 190, MUTED, 1.4, arrow=True))
    for dx in (478, 722):
        o.append(line(dx, 340, dx, 410, MUTED, 1.2, dash=True))
    o.append(txt(370, 392, T("Normalization", "Chuẩn hoá"), 16, MUTED, "middle"))
    o.append(
        txt(609, 392, T("Information extraction", "Trích xuất thông tin"), 16, MUTED, "middle")
    )
    o.append(txt(756, 392, T("Fusion", "Hợp nhất"), 16, MUTED, "middle"))
    # ---- output layer, scores ------------------------------------------------------------------
    o.append(box(822, 70, 34, 240, "#ececeb", MUTED, 1.2))
    o.append(vtxt(839, 190, T("Output layer", "Lớp đầu ra"), 18))
    o.append(line(858, 190, 890, 190, MAIN if stage > 1 else MUTED, 1.6, arrow=True))
    rd = (
        T("readout: last history position", "đọc tại: vị trí lịch sử cuối")
        if stage == 1
        else T("readout: query position", "đọc tại: vị trí query")
    )
    o.append(txt(892, 150, rd, 16, MUTED if stage == 1 else MAIN_DARK))
    o.append(box(892, 164, 150, 52))
    o.append(txt(967, 187, T("scores for", "điểm cho"), 17, INK, "middle"))
    o.append(txt(967, 207, T("100 clusters", "100 cluster"), 17, INK, "middle"))
    if stage < 3:
        o.append(line(1044, 190, 1066, 190, MUTED, 1.4, arrow=True))
        o.append(box(1068, 164, 84, 52))
        o.append(txt(1110, 196, "top 5", 18, INK, "middle", 600))
    else:
        # sameDest: late fusion with the guest's own earlier bookings at the searched destination
        o.append(line(967, 218, 967, 238, MUTED, 1.4, arrow=True))
        o.append(box(880, 240, 272, 54, "#ffffff", MAIN, 2))
        o.append(
            txt(1016, 262, "log p(model) + w · log p(sameDest)", 16, INK, "middle", italic=True)
        )
        o.append(txt(1016, 283, T("then top 5", "rồi lấy top 5"), 16, MUTED, "middle"))
        o.append(line(1016, 322, 1016, 296, MAIN, 1.6, arrow=True))
        o.append(box(852, 324, 300, 100, "#ffffff", MAIN, 2))
        o.append(txt(866, 348, "sameDest", 18, MAIN_DARK, weight=600))
        o.append(
            txt(866, 369, T("booked before at the searched", "cluster đã đặt trước tại điểm"), 16)
        )
        o.append(
            txt(
                866, 387, T("destination; recent count more", "đến đang tìm; lượt gần nặng hơn"), 16
            )
        )
        o.append(
            txt(
                866,
                409,
                T(
                    f"w = {w_known:g} if in history, else {w_new:g}",
                    f"w = {w_known:g} nếu đã đặt, không thì {w_new:g}",
                ),
                16,
                MUTED,
            )
        )
        o.append(f'<path d="M 12 70 V 434 H 1002 V 426" fill="none" stroke="{MAIN}" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#arr)"/>')  # fmt: skip
        for y, _ in hist:
            o.append(line(12, y + 26, 30, y + 26, MAIN, 1.6, dash=True))
    return svg(
        f"architecture_stage{stage}",
        w,
        h,
        "".join(o),
        T(f"Architecture stage {stage}", f"Kiến trúc, giai đoạn {stage}"),
    )


def fig_first_bookings(d: dict) -> str:
    b = d["base"]["training_set"]
    warm, first = b["warm_targets"], b["first_bookings"]
    s = d["seeds"]["per_variant"]
    c, dd = s["01c"]["cold/qt/ndcg@5"], s["01d"]["cold/qt/ndcg@5"]
    w, h = 1152, 340
    o = [txt(0, 22, T("training rows", "các dòng dùng để train"), 18, MUTED)]
    tot = warm + first
    span = 1000
    ww = span * warm / tot
    o.append(rect(0, 36, ww, 46, CONTEXT))
    o.append(rect(ww, 36, span * first / tot, 46, MAIN))
    o.append(
        txt(
            12,
            67,
            T(f"{num(warm)} warm targets", f"{num(warm)} mục tiêu warm"),
            19,
            "#ffffff",
            weight=600,
        )
    )
    o.append(
        txt(
            ww + 12,
            67,
            T(f"+ {num(first)} first bookings", f"+ {num(first)} lượt đặt đầu tiên"),
            19,
            "#ffffff",
            weight=600,
        )
    )
    # dot plot of the 3-epoch seed runs on cold test rows
    o.append(
        txt(
            0,
            150,
            T(
                "NDCG@5 on first-booking test rows, 3-epoch runs, 3 seeds",
                "NDCG@5 trên dòng test lượt đặt đầu tiên, run 3 epoch, 3 seed",
            ),
            18,
            MUTED,
        )
    )
    lo, hi = 0.33, 0.355
    x0, x1 = 360, 1000

    def xs(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    for i, (lab, e, col) in enumerate(
        (
            (T("trained on warm rows only", "train chỉ trên dòng warm"), c, CONTEXT),
            (T("+ first bookings", "+ lượt đặt đầu tiên"), dd, MAIN),
        )
    ):
        y = 200 + i * 56
        o.append(txt(x0 - 20, y + 6, lab, 19, INK, "end"))
        o.append(line(x0, y, x1, y, RULE, 1))
        for v in e["values"]:
            o.append(f'<circle cx="{xs(v):.1f}" cy="{y}" r="7" fill="{col}"/>')
        o.append(
            txt(
                xs(max(e["values"])) + 16,
                y + 6,
                T(f"mean {f3(e['mean'])}", f"trung bình {f3(e['mean'])}"),
                18,
                INK,
                weight=600,
            )
        )
    for t in (0.33, 0.34, 0.35):
        o.append(txt(xs(t), 318, f"{t:.2f}", 16, MUTED, "middle"))
    return svg(
        "first_bookings",
        w,
        h,
        "".join(o),
        T("Training rows and cold-row effect", "Dòng train và hiệu ứng trên dòng cold"),
    )


def fig_ladder(d: dict) -> str:
    rows = [
        {
            "label": T(
                "original SMLP4Rec, trained on warm users only",
                "SMLP4Rec gốc, train chỉ trên người dùng warm",
            ),
            "value": m(d, "plain_warm", "warm", "ndcg@5"),
            "color": CONTEXT,
            "lc": MUTED,
        },
        {
            "label": T(
                "original SMLP4Rec, trained on all events", "SMLP4Rec gốc, train trên mọi sự kiện"
            ),
            "value": m(d, "plain_all", "warm", "ndcg@5"),
            "color": CONTEXT,
        },
        {
            "label": "+ query token",
            "value": m(d, "query_token", "warm", "ndcg@5"),
            "color": CONTEXT,
        },
        {
            "label": T(
                "+ query token + sameDest (main)", "+ query token + sameDest (mô hình chính)"
            ),
            "value": m(d, "main", "warm", "ndcg@5"),
            "color": MAIN,
            "vc": MAIN_DARK,
            "lc": MAIN_DARK,
        },
    ]
    q = ci(d, "warm", "query_token", "plain_all", "ndcg@5")
    s = ci(d, "warm", "main", "query_token", "ndcg@5")
    rows[2]["note"] = f"{sgn3(q[0])}"
    rows[3]["note"] = f"{sgn3(s[0])}"
    return hbars(
        "ablation_ladder_warm", rows, 1152, T(470, 520), 0.5, f3, "NDCG@5 warm", 40, 22, val_w=90
    )


def fig_cold(d: dict) -> str:
    rows = [
        {
            "label": "SMLP4Rec + query token (+ sameDest)",
            "value": m(d, "main", "cold", "ndcg@5"),
            "color": MAIN,
            "vc": MAIN_DARK,
            "lc": MAIN_DARK,
        },
        {
            "label": "logistic regression",
            "value": m(d, "logreg", "cold", "ndcg@5"),
            "color": CONTRAST,
        },
        {
            "label": T("original SMLP4Rec (all events)", "SMLP4Rec gốc (mọi sự kiện)"),
            "value": m(d, "plain_all", "cold", "ndcg@5"),
            "color": CONTEXT,
            "note": T("one ranking for everyone", "một xếp hạng cho mọi người"),
        },
        {
            "label": "ItemKNN",
            "value": m(d, "itemknn", "cold", "ndcg@5"),
            "color": CONTEXT,
            "note": T("popularity fallback", "dự phòng theo độ phổ biến"),
        },
    ]
    return hbars("cold_rows_ndcg5", rows, 1152, 400, 0.4, f3, "NDCG@5 cold", 42, 24, val_w=90)


def fig_L(d: dict) -> str:
    L = d["L"][d["L"]["row_set"] == "all"]
    buckets = ["L=0 (cold)", "L=1", "L=2-4", "L=5-9", "L>=10"]
    xl = ["0", "1", "2–4", "5–9", "≥ 10"]
    series = [
        ("main", T("main model", "mô hình chính"), MAIN, 3),
        ("logreg", "logistic regression", CONTRAST, 2.5),
        ("plain_all", T("original SMLP4Rec", "SMLP4Rec gốc"), CONTEXT, 2),
        ("itemknn", "ItemKNN", CONTEXT, 2),
    ]
    w, h, x0, x1, y0, y1 = 1000, 420, 70, 760, 350, 20
    lo, hi = 0.0, 0.5

    def ys(v):
        return y0 - (y0 - y1) * (v - lo) / (hi - lo)

    def xs(i):
        return x0 + (x1 - x0) * i / (len(buckets) - 1)

    o = []
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        o.append(line(x0, ys(t), x1, ys(t), RULE, 1))
        o.append(txt(x0 - 12, ys(t) + 6, f"{t:.1f}", 16, MUTED, "end"))
    for i, lab in enumerate(xl):
        o.append(txt(xs(i), y0 + 30, lab, 17, MUTED, "middle"))
    o.append(
        txt(
            (x0 + x1) / 2,
            y0 + 58,
            T("earlier bookings of the guest (L)", "số lượt đặt trước đó của khách (L)"),
            16,
            MUTED,
            "middle",
        )
    )
    ends = []
    for key, name, col, sw in series:
        vals = [float(L[(L.bucket == b) & (L.model_key == key)]["ndcg@5"].iloc[0]) for b in buckets]
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="4.5" fill="{col}"/>')
        ends.append([ys(vals[-1]), name, col])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 22)
    for y, name, col in ends:
        o.append(txt(x1 + 14, y + 6, name, 18, MAIN_DARK if col == MAIN else INK))
    return svg(
        "history_length_ndcg5",
        w,
        h,
        "".join(o),
        T("NDCG@5 by history length", "NDCG@5 theo độ dài lịch sử"),
    )


def fig_known(d: dict) -> str:
    k = d["known"][d["known"]["row_set"] == "all"]

    def g(b, key):
        return float(k[(k.bucket == b) & (k.model_key == key)]["ndcg@5"].iloc[0])

    groups = [
        ("known destination", T("known destination", "điểm đến đã biết")),
        ("new destination", T("new destination", "điểm đến mới")),
    ]
    series = [("logreg", "logistic regression", CONTRAST), ("query_token", "query token", CONTEXT),
              ("main", "+ sameDest", MAIN)]  # fmt: skip
    w = 1000
    o, y = [], 10
    x0, span, vmax = T(230, 250), T(560, 540), 0.65
    for gi, (b, lab) in enumerate(groups):
        n = int(k[(k.bucket == b)]["n"].iloc[0])
        o.append(txt(0, y + 30, lab, 20, INK, weight=600))
        o.append(txt(0, y + 54, f"n = {num(n)}", 16, MUTED))
        for si, (key, name, col) in enumerate(series):
            v = g(b, key)
            yy = y + si * 40
            o.append(rect(x0, yy, span * v / vmax, 32, col))
            o.append(
                txt(
                    x0 + span * v / vmax + 10,
                    yy + 23,
                    f3(v),
                    18,
                    MAIN_DARK if key == "main" else INK,
                    weight=600,
                )
            )
            o.append(txt(x0 + span * v / vmax + 72, yy + 23, name, 17, MUTED))
        y += 3 * 40 + 34
    o.insert(0, line(x0, 4, x0, y - 30, RULE, 1))
    return svg(
        "known_vs_new_destination",
        w,
        int(y - 20),
        "".join(o),
        T("Known vs new destination", "Điểm đến đã biết và điểm đến mới"),
    )


def fig_context(d: dict) -> str:
    p = d["party"][d["party"]["row_set"] == "all"]
    k = d["pkg"][d["pkg"]["row_set"] == "all"]
    buckets = [(p, b) for b in ("solo", "couple", "family", "group")] + [
        (k, "package"),
        (k, "no package"),
    ]
    models = [("main", T("main model", "mô hình chính"), MAIN),
              ("logreg", "logistic regression", CONTRAST),
              ("itemknn", "ItemKNN", CONTEXT)]  # fmt: skip
    blab = {
        "solo": T("solo", "một mình"),
        "couple": T("couple", "cặp đôi"),
        "family": T("family", "gia đình"),
        "group": T("group", "nhóm"),
        "package": T("package", "có gói"),
        "no package": T("no package", "không gói"),
    }
    w, x0, x1 = 1100, 240, 900
    lo, hi = 0.1, 0.5

    def xs(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    o = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        o.append(line(xs(t), 30, xs(t), 30 + 46 * len(buckets), RULE, 1))
        o.append(txt(xs(t), 30 + 46 * len(buckets) + 24, f"{t:.1f}", 16, MUTED, "middle"))
    for i, (df, b) in enumerate(buckets):
        y = 52 + i * 46
        n = int(df[df.bucket == b]["n"].iloc[0])
        o.append(txt(x0 - 20, y + 6, blab[b], 19, INK, "end"))
        o.append(txt(x1 + 24, y + 6, f"n = {num(n)}", 16, MUTED))
        for key, _, col in models:
            v = float(df[(df.bucket == b) & (df.model_key == key)]["ndcg@5"].iloc[0])
            o.append(f'<circle cx="{xs(v):.1f}" cy="{y}" r="8" fill="{col}"/>')
    for j, (_, name, col) in enumerate(models):  # direct labels above the first row
        df, b = buckets[0]
        key = models[j][0]
        v = float(df[(df.bucket == b) & (df.model_key == key)]["ndcg@5"].iloc[0])
        o.append(txt(xs(v), 22, name, 17, MAIN_DARK if col == MAIN else INK, "middle"))
    return svg(
        "trip_context_ndcg5",
        w,
        30 + 46 * len(buckets) + 34,
        "".join(o),
        T("Trip context", "Ngữ cảnh chuyến đi"),
    )


def fig_rq4(d: dict) -> str:
    ba = d["ba"][d["ba"]["row_set"] == "all"].set_index("model_key")
    pts = [("main", T("main model", "mô hình chính"), MAIN), ("query_token", "query token", CONTEXT),
           ("logreg", "logistic regression", CONTRAST),
           ("plain_all", T("original SMLP4Rec", "SMLP4Rec gốc"), CONTEXT),
           ("itemknn", "ItemKNN", CONTEXT)]  # fmt: skip
    w, h, x0, x1, y0, y1 = 620, 410, 70, 590, 350, 44
    xlo, xhi, ylo, yhi = 0.1, 0.5, 0.3, 0.7

    def xs(v):
        return x0 + (x1 - x0) * (v - xlo) / (xhi - xlo)

    def ys(v):
        return y0 - (y0 - y1) * (v - ylo) / (yhi - ylo)

    o = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        o.append(line(x0, ys(t), x1, ys(t), RULE, 1))
        o.append(txt(x0 - 10, ys(t) + 6, f"{t:.1f}", 16, MUTED, "end"))
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        o.append(txt(xs(t), y0 + 26, f"{t:.1f}", 16, MUTED, "middle"))
    o.append(
        txt(x1, y0 + 52, T("NDCG@5, all test rows", "NDCG@5, mọi dòng test"), 16, MUTED, "end")
    )
    o.append(
        txt(
            x0 - 52,
            12,
            T(
                "exposure Gini of top-10 lists (lower = more even)",
                "Gini mức hiển thị của danh sách top-10 (thấp = đều hơn)",
            ),
            16,
            MUTED,
        )
    )
    offs = {"main": (12, 22), "query_token": (12, -10), "logreg": (12, 6),
            "plain_all": (12, 6), "itemknn": (-12, 22)}  # fmt: skip
    for key, name, col in pts:
        x, y = xs(m(d, key, "all", "ndcg@5")), ys(float(ba.loc[key, "exposure_gini"]))
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{col}"/>')
        dx, dy = offs[key]
        o.append(
            txt(
                x + dx,
                y + dy,
                name,
                17,
                MAIN_DARK if key == "main" else INK,
                "end" if dx < 0 else "start",
            )
        )
    return svg(
        "rq4_accuracy_vs_exposure",
        w,
        h,
        "".join(o),
        T("Accuracy vs exposure", "Độ chính xác và mức hiển thị"),
    )


def fig_dest(d: dict) -> str:
    bd = d["ba_dest"][d["ba_dest"]["row_set"] == "all"].set_index("model_key")
    booked = float(bd["mean_distinct_booked"].iloc[0])
    rows = [
        {"label": "ItemKNN", "value": float(bd.loc["itemknn", "mean_distinct_recommended@10"]), "color": CONTEXT},
        {"label": T("original SMLP4Rec", "SMLP4Rec gốc"), "value": float(bd.loc["plain_all", "mean_distinct_recommended@10"]), "color": CONTEXT},
        {"label": T("main model", "mô hình chính"), "value": float(bd.loc["main", "mean_distinct_recommended@10"]), "color": MAIN, "vc": MAIN_DARK, "lc": MAIN_DARK},
        {"label": "logistic regression", "value": float(bd.loc["logreg", "mean_distinct_recommended@10"]), "color": CONTRAST},
    ]  # fmt: skip
    body = hbars(
        "clusters_per_destination",
        rows,
        560,
        T(190, 200),
        80,
        lambda v: f"{v:.0f}",
        "clusters",
        34,
        16,
        val_w=60,
    )
    # booked reference line: drawn as a note under the bars (value from the same file)
    return body, booked


# ----------------------------------------------------------------------------------------------
# slides
# ----------------------------------------------------------------------------------------------
SLIDES: list[dict] = []


def slide(title, body, notes, src="", cls="", kind="content") -> None:
    SLIDES.append(
        {"title": title, "body": body, "notes": notes, "src": src, "cls": cls, "kind": kind}
    )


def table(head: list[str], rows: list[list[str]], hi_row: int | None = None, num_from=1) -> str:
    th = "".join(f'<th class="{"n" if i >= num_from else ""}">{h}</th>' for i, h in enumerate(head))
    trs = []
    for ri, r in enumerate(rows):
        cls = ' class="hi"' if ri == hi_row else ""
        tds = "".join(
            f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r)
        )
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f'<table class="t"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def build(d: dict) -> None:
    eda, counts, run = d["eda"], d["counts"], d["run"]
    rc = d["final"]["row_counts"]
    n_all, n_cold, n_warm = rc["all"], rc["cold"], rc["warm"]
    bookings = counts["collapse_steps"]["rows_out"]
    users = eda["E2"]["users"]
    single = eda["E2"]["share_single_booking"]
    first = d["base"]["training_set"]["first_bookings"]
    nf = len(run["query_fields"])
    BENCH = "results/week5_final/final_benchmark.csv"

    # 1 title
    slide(
        T(
            "Search-aware next-booking recommendation on public Expedia data",
            "Gợi ý lượt đặt tiếp theo có xét đến tìm kiếm hiện tại, trên dữ liệu Expedia công khai",
        ),
        T(
            '<p class="sub">Next-best-product recommendation for resort bookings, '
            "topic C1 (Vinpearl and GSM context)</p>"
            '<p class="sub2">Final presentation, draft v1, 11 October 2026</p>'
            '<p class="sub3">Public data only. No Vinpearl or GSM data is used.</p>',
            '<p class="sub">Gợi ý sản phẩm tiếp theo cho đặt phòng nghỉ dưỡng, '
            "đề tài C1 (bối cảnh Vinpearl và GSM)</p>"
            '<p class="sub2">Báo cáo cuối kỳ, bản nháp v1, ngày 11 tháng Mười năm 2026</p>'
            '<p class="sub3">Chỉ dùng dữ liệu công khai. Không dùng dữ liệu của Vinpearl hay GSM.</p>',
        ),
        T(
            "Open with the question, not the method: which room or package should a resort show while a "
            "guest is searching? We study it on the public Expedia Hotel Recommendations data because no "
            "resort data is available to the project. Everything shown is offline evaluation. The hotel "
            "cluster stands in for a room category or package; the ids are anonymous. Expect the question "
            "'is this Vinpearl data?': no, it is public Expedia data only.",
            "Mở đầu bằng câu hỏi, không phải phương pháp: khu nghỉ dưỡng nên hiển thị loại phòng hay "
            "gói nào khi khách đang tìm kiếm? Chúng tôi nghiên cứu trên dữ liệu công khai Expedia Hotel "
            "Recommendations vì dự án không có dữ liệu của khu nghỉ dưỡng. Mọi kết quả đều là đánh giá "
            "offline. hotel_cluster đại diện cho hạng phòng hoặc gói; các id đều ẩn danh. Câu hỏi có thể "
            "gặp: 'đây có phải dữ liệu Vinpearl không?': không, chỉ là dữ liệu công khai của Expedia.",
        ),
        kind="title",
    )

    # 2 problem
    prob = svg(
        "problem_schematic",
        1152,
        250,
        rect(0, 70, 300, 110, "#f2f2f1", MUTED, 1.2)
        + txt(20, 108, T("current search", "tìm kiếm hiện tại"), 20, INK, weight=600)
        + txt(20, 136, T("destination, dates, party,", "điểm đến, ngày, nhóm khách,"), 18, MUTED)
        + txt(20, 160, T("package, channel", "gói, kênh bán"), 18, MUTED)
        + rect(0, 0, 300, 50, "none", CONTEXT, 1.2, dash=True)
        + txt(
            20,
            32,
            T("a few past bookings, often none", "vài lượt đặt cũ, thường là không có"),
            18,
            MUTED,
        )
        + line(302, 125, 420, 125, MUTED, 1.5, arrow=True)
        + line(150, 52, 150, 66, MUTED, 1.2, arrow=True)
        + rect(424, 80, 260, 90, "#ffffff", INK, 1.5)
        + txt(554, 132, T("recommender", "hệ gợi ý"), 20, INK, "middle", 600)
        + line(686, 125, 800, 125, MUTED, 1.5, arrow=True)
        + "".join(
            rect(
                804,
                30 + i * 40,
                300,
                32,
                MAIN if i == 0 else "#ffffff",
                MAIN if i == 0 else RULE,
                1.2,
            )
            + txt(
                820,
                52 + i * 40,
                T(f"{i + 1}. room or package type", f"{i + 1}. loại phòng hoặc gói"),
                17,
                "#ffffff" if i == 0 else INK,
            )
            for i in range(5)
        ),
        T("Problem schematic", "Sơ đồ bài toán"),
    )
    slide(
        T(
            "A resort has to pick, at search time, which room or package to show",
            "Khu nghỉ dưỡng phải chọn, ngay lúc khách tìm kiếm, loại phòng hoặc gói nào để hiển thị",
        ),
        f'<div class="fig">{prob}</div>'
        + T(
            '<p class="lede">The output is a ranked top 5 out of 100 hotel clusters, '
            "a proxy for room category or package.</p>",
            '<p class="lede">Đầu ra là top 5 xếp hạng trong 100 hotel_cluster, '
            "đại diện (proxy) cho hạng phòng hoặc gói.</p>",
        ),
        T(
            "The task: given the current search and whatever the guest booked before, rank the 100 "
            "Expedia hotel clusters so that the one the guest books is near the top. For a resort, the "
            "cluster plays the role of a room category or package. Two things make it hard: most guests "
            "have very few past bookings, and the search itself (where, when, with whom) changes what "
            "they want. Likely question: why clusters and not hotels? Because Expedia only publishes "
            "anonymous cluster ids, which we state as a limitation.",
            "Bài toán: biết tìm kiếm hiện tại và những gì khách đã đặt trước đó, xếp hạng 100 hotel_cluster "
            "của Expedia sao cho cluster khách đặt nằm gần đầu danh sách. Với khu nghỉ dưỡng, cluster đóng "
            "vai trò hạng phòng hoặc gói. Có hai khó khăn: đa số khách có rất ít lượt đặt trước đó, và "
            "chính tìm kiếm (đi đâu, khi nào, với ai) làm thay đổi nhu cầu. Câu hỏi có thể gặp: sao "
            "dùng cluster mà không dùng khách sạn? Vì Expedia chỉ công bố id cluster ẩn danh, và chúng "
            "tôi nêu đây là một giới hạn.",
        ),
    )

    # 3 research questions
    slide(
        T(
            "We test whether a search-aware sequence model beats classical recommenders, and at what cost",
            "Chúng tôi kiểm tra liệu mô hình chuỗi có xét tìm kiếm có vượt các hệ gợi ý cổ điển không, "
            "và phải đánh đổi gì",
        ),
        T(
            '<div class="rq">'
            "<p><b>RQ1</b> Does a context-aware session model beat traditional collaborative filtering "
            "on sparse, seasonal, trip-dependent travel data?</p>"
            "<p><b>RQ4</b> What does the accuracy gain cost in diversity and coverage?</p>"
            '<p class="muted">RQ2 (food delivery) and RQ3 (cross-sell) are out of scope.</p></div>',
            '<div class="rq">'
            "<p><b>RQ1</b> Mô hình phiên có xét ngữ cảnh có vượt collaborative filtering truyền thống "
            "trên dữ liệu du lịch thưa, theo mùa, phụ thuộc chuyến đi không?</p>"
            "<p><b>RQ4</b> Mức tăng độ chính xác phải đánh đổi bao nhiêu về diversity và coverage?</p>"
            '<p class="muted">RQ2 (giao đồ ăn) và RQ3 (cross-sell) nằm ngoài phạm vi.</p></div>',
        ),
        T(
            "Two research questions remain active. RQ1 compares a context-aware sequence model with "
            "classical recommenders, and we break it down by sparsity, season and trip context. RQ4 asks "
            "whether higher accuracy comes with less diverse or more popularity-biased lists. RQ2 (food "
            "delivery) and RQ3 (cross-selling between Vinpearl and GSM) were dropped in week 4 because no "
            "public dataset links the two domains; one sentence is enough. Likely question: why only two "
            "RQs? The five-week budget went to one domain done properly.",
            "Còn hai câu hỏi nghiên cứu đang thực hiện. RQ1 so sánh mô hình chuỗi có xét ngữ cảnh với "
            "các hệ gợi ý cổ điển, và chúng tôi phân tích theo độ thưa, mùa vụ và ngữ cảnh chuyến đi. "
            "RQ4 hỏi liệu độ chính xác cao hơn có đi kèm danh sách kém đa dạng hơn hoặc lệch về mục phổ "
            "biến hơn không. RQ2 (giao đồ ăn) và RQ3 (cross-sell giữa Vinpearl và GSM) đã bị loại ở "
            "tuần 4 vì không có dataset công khai nào nối hai lĩnh vực; một câu là đủ. Câu hỏi có thể "
            "gặp: sao chỉ có hai RQ? Ngân sách năm tuần được dành để làm một lĩnh vực cho đến nơi đến chốn.",
        ),
    )

    # 4 sparsity
    slide(
        T(
            f"Expedia gives {num(bookings)} bookings from {num(users)} users, "
            f"but {pct(single)} booked only once",
            f"Expedia có {num(bookings)} lượt đặt phòng của {num(users)} người dùng, "
            f"nhưng {pct(single)} chỉ đặt một lần",
        ),
        f'<div class="row"><div class="fig">{fig_users(d)}</div>'
        + T(
            f'<p class="aside">Share of users by number of bookings, 2013 to 2014, '
            f"{eda['E1']['before_after']['rows']['after']:,} bookings after cleaning.</p></div>",
            f'<p class="aside">Tỷ lệ người dùng theo số lượt đặt, 2013 đến 2014, '
            f"{eda['E1']['before_after']['rows']['after']:,} lượt đặt sau khi làm sạch.</p></div>",
        ),
        T(
            "Bookings only; clicks are never treated as positives. After removing 8 duplicate keys and "
            f"collapsing {num(counts['collapse_steps']['burst_dropped'])} burst repeats we keep "
            f"{num(bookings)} bookings from {num(users)} users. {pct2(single)} of users booked once, so a "
            "model that needs a long history has little to work with. Source: "
            "results/week4_rebuild/eda_clean.json (E2) and bookings_counts.json. Likely question: why "
            "collapse burst repeats? They are the same booking repeated within minutes, which would "
            "inflate repeat-based models.",
            "Chỉ tính lượt đặt phòng; click không bao giờ được coi là mẫu dương. Sau khi bỏ 8 khóa "
            f"trùng và gộp {num(counts['collapse_steps']['burst_dropped'])} lượt lặp dồn dập (burst), "
            f"còn {num(bookings)} lượt đặt của {num(users)} người dùng. {pct2(single)} người dùng chỉ "
            "đặt một lần, nên mô hình cần lịch sử dài có rất ít thứ để dựa vào. Nguồn: "
            "results/week4_rebuild/eda_clean.json (E2) và bookings_counts.json. Câu hỏi có thể gặp: "
            "sao phải gộp burst? Đó là cùng một lượt đặt lặp lại trong vài phút, sẽ làm các mô hình "
            "dựa vào lượt lặp bị thổi phồng.",
        ),
        src=T(
            "Source: results/week4_rebuild/eda_clean.json (E2), bookings_counts.json. Expedia bookings, all users.",
            "Nguồn: results/week4_rebuild/eda_clean.json (E2), bookings_counts.json. Lượt đặt Expedia, mọi người dùng.",
        ),
    )

    # 5 fields
    ig = eda["E6"]["held_out_information_gain_bits_on_valid"]
    slide(
        T(
            "The searched destination says far more about the booked cluster than any other field",
            "Điểm đến đang tìm cho biết về cluster được đặt nhiều hơn hẳn mọi trường khác",
        ),
        f'<div class="fig">{fig_fields(d)}</div>'
        + T(
            '<p class="lede">Information gain about the cluster, in bits, measured on validation bookings.</p>',
            '<p class="lede">Information gain về cluster, tính bằng bit, đo trên các lượt đặt thuộc tập validation.</p>',
        ),
        T(
            f"Held-out information gain of each search field about the booked cluster: destination gives "
            f"{ig['dest']:.2f} bits, the next field (user country) {ig['country']:.2f}. This is why the "
            "current search, and above all the destination, has to enter the model. Hotel fields and "
            "orig_destination_distance describe the booked hotel and are never model input (label leak). "
            "Source: results/week4_rebuild/eda_clean.json (E6). Likely question: is this leakage? No, the "
            "destination is chosen by the guest before booking; the leakage audit is in leakage_audit.json.",
            f"Information gain trên tập held-out của từng trường tìm kiếm về cluster được đặt: điểm đến "
            f"cho {ig['dest']:.2f} bit, trường kế tiếp (quốc gia người dùng) {ig['country']:.2f}. Vì vậy "
            "tìm kiếm hiện tại, nhất là điểm đến, phải được đưa vào mô hình. Các trường hotel và "
            "orig_destination_distance mô tả khách sạn đã đặt và không bao giờ là đầu vào của mô hình "
            "(rò rỉ nhãn). Nguồn: results/week4_rebuild/eda_clean.json (E6). Câu hỏi có thể gặp: đây "
            "có phải rò rỉ không? Không, điểm đến do khách chọn trước khi đặt; kiểm tra rò rỉ nằm trong "
            "leakage_audit.json.",
        ),
        src=T(
            "Source: results/week4_rebuild/eda_clean.json (E6). Expedia, validation bookings.",
            "Nguồn: results/week4_rebuild/eda_clean.json (E6). Expedia, lượt đặt thuộc validation.",
        ),
    )

    # 6 protocol
    slide(
        T(
            "Every model is scored on one temporal split, full ranking, warm and first-booking rows apart",
            "Mọi mô hình được chấm trên cùng một lần chia theo thời gian, xếp hạng đầy đủ, "
            "dòng warm và lượt đặt đầu tiên tách riêng",
        ),
        f'<div class="fig">{fig_timeline(d)}</div>'
        + T(
            '<p class="lede">Full ranking over all 100 clusters. Recall and NDCG at 5, 10 and 20. '
            "Settings chosen on valid, test read once.</p>",
            '<p class="lede">Xếp hạng đầy đủ trên cả 100 cluster. Recall và NDCG tại 5, 10 và 20. '
            "Cấu hình chọn trên valid, test chỉ đọc một lần.</p>",
        ),
        T(
            f"Global temporal split 80/10/10 over next-booking targets. Test has {num(n_warm)} warm rows "
            f"and {num(n_cold)} first bookings, {num(n_all)} in total. Every model is scored on the same "
            "rows with the same code, over all 100 clusters, no sampled negatives. Early stopping on "
            "valid NDCG@10 (patience 2). Source: results/week4_rebuild/eda_clean.json (splits), "
            "results/week5_final/final_results.json (row_counts). Likely question: why not a random split? "
            "It would leak future bookings into training.",
            f"Chia theo thời gian toàn cục 80/10/10 trên các mục tiêu next-booking. Test có {num(n_warm)} "
            f"dòng warm và {num(n_cold)} lượt đặt đầu tiên, tổng {num(n_all)}. Mọi mô hình được chấm trên "
            "cùng các dòng bằng cùng một đoạn code, trên cả 100 cluster, không dùng negative lấy mẫu. "
            "Early stopping theo NDCG@10 trên valid (patience 2). Nguồn: "
            "results/week4_rebuild/eda_clean.json (splits), results/week5_final/final_results.json "
            "(row_counts). Câu hỏi có thể gặp: sao không chia ngẫu nhiên? Vì sẽ làm lộ các lượt đặt "
            "trong tương lai vào tập train.",
        ),
        src=T(
            "Source: results/week4_rebuild/eda_clean.json (splits); results/week5_final/final_results.json.",
            "Nguồn: results/week4_rebuild/eda_clean.json (splits); results/week5_final/final_results.json.",
        ),
    )

    # 7-9 architecture
    pl = d["plain_es"]
    slide(
        T(
            "The original SMLP4Rec reads only past bookings and ignores the current search",
            "SMLP4Rec gốc chỉ đọc các lượt đặt trước đó và bỏ qua tìm kiếm hiện tại",
        ),
        f'<div class="fig">{fig_arch(d, 1)}</div>',
        T(
            "SMLP4Rec (Gao et al., TOIS 2024) is an all-MLP sequence model. We copied the authors' code "
            "and made two marked changes: sequence length from the config and no item-feature list, "
            "because clusters have no attributes. Known issues of the original are kept (one block shared "
            f"across layers). The history-only model has {num(pl['n_parameters'])} parameters and scores "
            "the clusters from the last history position. It never sees the destination or the dates. "
            "Label: research adaptation, not the authors' release. Source: src/models/smlprec.py, "
            "smlprec_plain_cold_train_es_run.json.",
            "SMLP4Rec (Gao et al., TOIS 2024) là mô hình chuỗi toàn MLP. Chúng tôi sao chép code của tác "
            "giả và thực hiện hai thay đổi có đánh dấu: độ dài chuỗi lấy từ config và không có danh sách "
            "đặc trưng của item, vì cluster không có thuộc tính. Các vấn đề đã biết của bản gốc được "
            f"giữ nguyên (một khối dùng chung cho các lớp). Mô hình chỉ dùng lịch sử có {num(pl['n_parameters'])} "
            "tham số và chấm điểm các cluster từ vị trí lịch sử cuối cùng. Nó không bao giờ thấy điểm đến "
            "hay ngày tháng. Nhãn: bản điều chỉnh cho nghiên cứu, không phải bản phát hành của tác giả. "
            "Nguồn: src/models/smlprec.py, smlprec_plain_cold_train_es_run.json.",
        ),
        src=T(
            "Research adaptation of SMLP4Rec (Gao et al., TOIS 2024). Source: src/models/smlprec.py.",
            "Bản điều chỉnh cho nghiên cứu của SMLP4Rec (Gao et al., TOIS 2024). Nguồn: src/models/smlprec.py.",
        ),
    )
    slide(
        T(
            f"Our query token puts the {nf} search fields into the model as one extra position",
            f"query token của chúng tôi đưa {nf} trường tìm kiếm vào mô hình như một vị trí bổ sung",
        ),
        f'<div class="fig">{fig_arch(d, 2)}</div>',
        T(
            f"The query token is one extra position after the history: a learned mask vector plus the sum "
            f"of {nf} field embeddings (destination, destination type, check-in month, days to check-in, "
            "nights, adults, children, rooms, package, mobile, channel, site, point-of-sale continent, "
            "user country). The mixer lets this position read the history, and its hidden state scores "
            "the clusters. A first booking still has a query token, so the model works with an empty "
            "history. Vocabulary from training rows only. Source: src/nbp/models/smlprec_query.py, "
            "smlprec_query_cold_train_es_run.json (query_fields).",
            f"query token là một vị trí bổ sung sau lịch sử: một vector mask học được cộng với tổng của "
            f"{nf} embedding trường (điểm đến, loại điểm đến, tháng check-in, số ngày đến check-in, "
            "số đêm, người lớn, trẻ em, số phòng, gói, mobile, kênh, site, châu lục điểm bán, quốc gia "
            "người dùng). Mixer cho vị trí này đọc lịch sử, và trạng thái ẩn của nó chấm điểm các "
            "cluster. Lượt đặt đầu tiên vẫn có query token, nên mô hình chạy được với lịch sử rỗng. "
            "Từ vựng chỉ lấy từ các dòng train. Nguồn: src/nbp/models/smlprec_query.py, "
            "smlprec_query_cold_train_es_run.json (query_fields).",
        ),
        src=T(
            "Source: src/nbp/models/smlprec_query.py; results/week4_rebuild/smlprec_query_cold_train_es_run.json.",
            "Nguồn: src/nbp/models/smlprec_query.py; results/week4_rebuild/smlprec_query_cold_train_es_run.json.",
        ),
    )
    tv = run["samedest_hybrid"]["tuned_on_valid"]["A"]
    slide(
        T(
            "sameDest adds what the guest booked before at the searched destination",
            "sameDest bổ sung những gì khách đã đặt trước đây tại điểm đến đang tìm",
        ),
        f'<div class="fig">{fig_arch(d, 3)}</div>',
        T(
            "sameDest is a late-fusion term, not a network input. It looks for clusters the guest booked "
            "earlier at the destination being searched, weights recent ones more, and adds their log "
            f"share to the model's log probability. The smoothing alpha ({tv['alpha']:g}) and the weights "
            "were picked on valid with the one-standard-error rule: weight "
            f"{tv['weights']['1'][1]:g} when the destination is in the history, {tv['weights']['1'][0]:g} "
            "otherwise, so new destinations and first bookings are untouched. Source: "
            "src/nbp/hybrid/same_dest.py, smlprec_query_cold_train_es_run.json (samedest_hybrid).",
            "sameDest là một số hạng late-fusion, không phải đầu vào của mạng. Nó tìm các cluster khách "
            "đã đặt trước đó tại điểm đến đang tìm, cho lượt gần đây trọng số cao hơn, và cộng log của "
            f"tỷ lệ đó vào log xác suất của mô hình. Hệ số làm mượt alpha ({tv['alpha']:g}) và các trọng "
            "số được chọn trên valid bằng quy tắc một sai số chuẩn: trọng số "
            f"{tv['weights']['1'][1]:g} khi điểm đến có trong lịch sử, {tv['weights']['1'][0]:g} "
            "trong trường hợp còn lại, nên điểm đến mới và lượt đặt đầu tiên không bị ảnh hưởng. Nguồn: "
            "src/nbp/hybrid/same_dest.py, smlprec_query_cold_train_es_run.json (samedest_hybrid).",
        ),
        src=T(
            "Source: src/nbp/hybrid/; results/week4_rebuild/smlprec_query_cold_train_es_run.json (samedest_hybrid).",
            "Nguồn: src/nbp/hybrid/; results/week4_rebuild/smlprec_query_cold_train_es_run.json (samedest_hybrid).",
        ),
    )

    # 10 first bookings
    s = d["seeds"]["paired_01d_minus_01c"]["cold/qt/ndcg@5"]
    slide(
        T(
            f"Training with the {num(first)} first bookings teaches the model the empty-history case",
            f"Train thêm {num(first)} lượt đặt đầu tiên giúp mô hình học trường hợp lịch sử rỗng",
        ),
        f'<div class="fig">{fig_first_bookings(d)}</div>',
        T(
            f"The original recipe trains only on warm targets. We add the {num(first)} first bookings of "
            "the training window (empty history), so the model learns what to do with a new guest; the "
            "baselines and the plain model are retrained on the same rows. The effect is measured on the "
            f"3-epoch runs with three seeds: cold NDCG@5 rises by {f3(s['mean'])} on average, higher in "
            f"{s['seeds_where_01d_higher']} of 3 seeds. The early-stopping run has no warm-only twin, so "
            "this step is shown on the 3-epoch footing. Source: results/week4_rebuild/seeds_summary.json, "
            "baselines_es_run.json (training_set).",
            f"Công thức gốc chỉ train trên các mục tiêu warm. Chúng tôi thêm {num(first)} lượt đặt đầu "
            "tiên của giai đoạn train (lịch sử rỗng), để mô hình học cách xử lý khách mới; các baseline "
            "và mô hình gốc được train lại trên cùng các dòng. Hiệu ứng được đo trên các run 3 epoch với "
            f"ba seed: NDCG@5 trên dòng cold tăng trung bình {f3(s['mean'])}, cao hơn ở "
            f"{s['seeds_where_01d_higher']} trong 3 seed. Run early stopping không có bản đối chứng chỉ "
            "dùng warm, nên bước này được trình bày trên nền 3 epoch. Nguồn: "
            "results/week4_rebuild/seeds_summary.json, baselines_es_run.json (training_set).",
        ),
        src=T(
            "Source: results/week4_rebuild/baselines_es_run.json (training_set), seeds_summary.json "
            "(3-epoch runs, seeds 2022 to 2024). Expedia test, first-booking rows.",
            "Nguồn: results/week4_rebuild/baselines_es_run.json (training_set), seeds_summary.json "
            "(run 3 epoch, seed 2022 đến 2024). Expedia test, dòng lượt đặt đầu tiên.",
        ),
    )

    # 11 ladder
    q = ci(d, "warm", "query_token", "plain_all", "ndcg@5")
    sd = ci(d, "warm", "main", "query_token", "ndcg@5")
    pw = ci(d, "warm", "plain_all", "plain_warm", "ndcg@5")
    slide(
        T(
            "The query token carries most of the gain, sameDest adds a smaller, real step",
            "query token mang phần lớn mức tăng, sameDest thêm một bước nhỏ hơn nhưng có thật",
        ),
        f'<div class="fig">{fig_ladder(d)}</div>'
        + T(
            f'<p class="lede">NDCG@5 on warm test rows. Steps: query token {ci_txt(q)}, '
            f"sameDest {ci_txt(sd)}.</p>",
            f'<p class="lede">NDCG@5 trên dòng test warm. Các bước: query token {ci_txt(q)}, '
            f"sameDest {ci_txt(sd)}.</p>",
        ),
        T(
            f"Warm test rows, NDCG@5. The original SMLP4Rec scores {f3(m(d, 'plain_all', 'warm', 'ndcg@5'))} "
            f"when trained on all events and {f3(m(d, 'plain_warm', 'warm', 'ndcg@5'))} trained on warm "
            f"users only; the difference {ci_txt(pw)} is not significant. The query token adds {sgn3(q[0])}, "
            f"sameDest another {sgn3(sd[0])}. Paired bootstrap, 1000 resamples. Source: "
            "results/week5_final/final_benchmark.csv, bootstrap_ci.csv. Likely question: is sameDest just "
            "re-booking? Removing re-books within a day keeps a positive gain (slide 15).",
            f"Dòng test warm, NDCG@5. SMLP4Rec gốc đạt {f3(m(d, 'plain_all', 'warm', 'ndcg@5'))} "
            f"khi train trên mọi sự kiện và {f3(m(d, 'plain_warm', 'warm', 'ndcg@5'))} khi chỉ train trên "
            f"người dùng warm; chênh lệch {ci_txt(pw)} không có ý nghĩa thống kê. query token thêm "
            f"{sgn3(q[0])}, sameDest thêm {sgn3(sd[0])} nữa. Paired bootstrap, 1000 lần lấy mẫu lại. "
            "Nguồn: results/week5_final/final_benchmark.csv, bootstrap_ci.csv. Câu hỏi có thể gặp: "
            "sameDest có phải chỉ là đặt lại không? Loại các lượt đặt lại trong vòng một ngày vẫn còn "
            "mức tăng dương (slide 15).",
        ),
        src=T(
            f"Source: {BENCH}, bootstrap_ci.csv. Expedia test, warm rows, n = {num(n_warm)}, seed 2022.",
            f"Nguồn: {BENCH}, bootstrap_ci.csv. Expedia test, dòng warm, n = {num(n_warm)}, seed 2022.",
        ),
    )

    # 12 benchmark table
    rows_def = [
        ("itemknn", "ItemKNN", True),
        ("logreg", T("Logistic regression", "logistic regression"), False),
        (
            "plain_warm",
            T(
                "Original SMLP4Rec, trained on warm users only",
                "SMLP4Rec gốc, train chỉ trên người dùng warm",
            ),
            True,
        ),
        (
            "plain_all",
            T(
                "Original SMLP4Rec, trained on all events",
                "SMLP4Rec gốc, train trên mọi sự kiện",
            ),
            False,
        ),
        ("query_token", "+ query token", False),
        (
            "main",
            T("+ query token + sameDest (main)", "+ query token + sameDest (chính)"),
            False,
        ),
    ]
    trs = []
    for key, name, cold_fb in rows_def:
        cells = [name]
        for mt in ("recall@5", "ndcg@5"):
            for rs in ("all", "cold", "warm"):
                v = f3(m(d, key, rs, mt))
                if cold_fb and rs in ("cold", "all"):
                    v += "<sup>a</sup>"
                cells.append(v)
        trs.append(cells)
    lw = ci(d, "warm", "main", "logreg", "ndcg@5")
    lc = ci(d, "cold", "main", "logreg", "ndcg@5")
    kw = ci(d, "warm", "main", "itemknn", "ndcg@5")
    lr07 = d["lr07"].set_index("row_set")
    ALL = T("all", "tất cả")
    slide(
        T(
            "The main model leads both baselines on all, cold and warm test rows",
            "Mô hình chính dẫn trước cả hai baseline trên dòng test tất cả, cold và warm",
        ),
        table(
            [
                "",
                f"R@5 {ALL}",
                "R@5 cold",
                "R@5 warm",
                f"NDCG@5 {ALL}",
                "NDCG@5 cold",
                "NDCG@5 warm",
            ],
            trs,
            hi_row=5,
        )
        + T(
            f'<p class="fn"><sup>a</sup> Popularity fallback on first-booking rows, not a model score. '
            f"Main minus logistic regression, NDCG@5: warm {ci_txt(lw)}, cold {ci_txt(lc)}.</p>",
            f'<p class="fn"><sup>a</sup> Dự phòng theo độ phổ biến trên dòng lượt đặt đầu tiên, không '
            f"phải điểm của mô hình. Chính trừ logistic regression, NDCG@5: warm {ci_txt(lw)}, "
            f"cold {ci_txt(lc)}.</p>",
        ),
        T(
            f"All models except the warm-only original are trained on the same rows: warm targets plus "
            "first bookings. Baselines are from notebook 07b, the main model from notebook 01d with early "
            f"stopping (best epoch {run['best_epoch_by_valid']}). Warm NDCG@5 gap to logistic regression "
            f"{ci_txt(lw)}, to ItemKNN {ci_txt(kw)}. Footnote for questions: a logistic regression trained "
            f"in notebook 07 (C = 10) is stronger, warm gap {sgn3(float(lr07.loc['warm', 'main_minus_lr07_ndcg@5']))} "
            "NDCG@5 (appendix A1). Source: results/week5_final/final_benchmark.csv, bootstrap_ci.csv.",
            "Mọi mô hình trừ bản gốc chỉ-warm đều được train trên cùng các dòng: mục tiêu warm cộng "
            "lượt đặt đầu tiên. Baseline lấy từ notebook 07b, mô hình chính từ notebook 01d có early "
            f"stopping (best epoch {run['best_epoch_by_valid']}). Khoảng cách NDCG@5 warm so với logistic "
            f"regression {ci_txt(lw)}, so với ItemKNN {ci_txt(kw)}. Chú thích khi bị hỏi: logistic "
            f"regression train ở notebook 07 (C = 10) mạnh hơn, khoảng cách warm "
            f"{sgn3(float(lr07.loc['warm', 'main_minus_lr07_ndcg@5']))} NDCG@5 (phụ lục A1). "
            "Nguồn: results/week5_final/final_benchmark.csv, bootstrap_ci.csv.",
        ),
        src=T(
            f"Source: {BENCH}, bootstrap_ci.csv. Expedia test: all n = {num(n_all)}, cold n = "
            f"{num(n_cold)}, warm n = {num(n_warm)}. Seed 2022.",
            f"Nguồn: {BENCH}, bootstrap_ci.csv. Expedia test: tất cả n = {num(n_all)}, cold n = "
            f"{num(n_cold)}, warm n = {num(n_warm)}. Seed 2022.",
        ),
        cls="tbl",
    )

    # 13 cold
    slide(
        T(
            "On first bookings only the search-aware models rank well",
            "Trên lượt đặt đầu tiên, chỉ các mô hình có xét tìm kiếm mới xếp hạng tốt",
        ),
        f'<div class="fig">{fig_cold(d)}</div>'
        + T(
            f'<p class="lede">The main model leads logistic regression by {ci_txt(lc)} NDCG@5.</p>',
            f'<p class="lede">Mô hình chính dẫn trước logistic regression {ci_txt(lc)} NDCG@5.</p>',
        ),
        T(
            "Cold rows are first bookings: no history at all. On these rows the main model is the query "
            "token, because sameDest has nothing to work with. ItemKNN cannot score them and falls back "
            "to first-booking popularity; the plain SMLP4Rec scores them but returns one ranking for "
            f"everyone. Logistic regression reads the same search fields and comes second at "
            f"{f3(m(d, 'logreg', 'cold', 'ndcg@5'))}. Source: results/week5_final/final_benchmark.csv, "
            "bootstrap_ci.csv. Caveat: 'first booking' means first observed booking, the data starts in "
            "January 2013.",
            "Dòng cold là các lượt đặt đầu tiên: hoàn toàn không có lịch sử. Trên các dòng này mô hình "
            "chính chính là query token, vì sameDest không có gì để dùng. ItemKNN không chấm được và "
            "quay về độ phổ biến của lượt đặt đầu tiên; SMLP4Rec gốc chấm được nhưng trả về một xếp hạng "
            "cho mọi người. Logistic regression đọc cùng các trường tìm kiếm và đứng thứ hai với "
            f"{f3(m(d, 'logreg', 'cold', 'ndcg@5'))}. Nguồn: results/week5_final/final_benchmark.csv, "
            "bootstrap_ci.csv. Lưu ý: 'lượt đặt đầu tiên' nghĩa là lượt đặt đầu tiên quan sát được, dữ "
            "liệu bắt đầu từ tháng Một năm 2013.",
        ),
        src=T(
            f"Source: {BENCH}, bootstrap_ci.csv. Expedia test, first-booking rows, n = {num(n_cold)}.",
            f"Nguồn: {BENCH}, bootstrap_ci.csv. Expedia test, dòng lượt đặt đầu tiên, n = {num(n_cold)}.",
        ),
    )

    # 14 history length
    L = d["L"][d["L"]["row_set"] == "all"]

    def lv(b, k):
        return float(L[(L.bucket == b) & (L.model_key == k)]["ndcg@5"].iloc[0])

    g0 = lv("L=0 (cold)", "main") - lv("L=0 (cold)", "logreg")
    g10 = lv("L>=10", "main") - lv("L>=10", "logreg")
    slide(
        T(
            f"The lead over logistic regression grows with history, from {sgn3(g0)} to {sgn3(g10)} NDCG@5",
            f"Khoảng dẫn trước logistic regression tăng theo lịch sử, từ {sgn3(g0)} đến {sgn3(g10)} NDCG@5",
        ),
        f'<div class="fig">{fig_L(d)}</div>',
        T(
            "NDCG@5 by the number of earlier bookings. Logistic regression ignores history, so its line is "
            "flat; the history-only models start near popularity and climb; the main model is on top in "
            f"every bucket and the gap to logistic regression widens from {sgn3(g0)} (first bookings) to "
            f"{sgn3(g10)} (ten or more earlier bookings). This is the RQ1 sparsity answer. Source: "
            "results/week5_final/slices_history_length.csv; the CSV also gives row counts and a binomial "
            "half-width per bucket.",
            "NDCG@5 theo số lượt đặt trước đó. Logistic regression bỏ qua lịch sử nên đường của nó "
            "phẳng; các mô hình chỉ dùng lịch sử bắt đầu gần mức phổ biến rồi đi lên; mô hình chính đứng "
            f"đầu ở mọi nhóm và khoảng cách với logistic regression nới rộng từ {sgn3(g0)} (lượt đặt "
            f"đầu tiên) đến {sgn3(g10)} (từ mười lượt đặt trước trở lên). Đây là câu trả lời RQ1 về độ "
            "thưa. Nguồn: results/week5_final/slices_history_length.csv; file CSV cũng cho số dòng và "
            "nửa độ rộng nhị thức của từng nhóm.",
        ),
        src=T(
            "Source: results/week5_final/slices_history_length.csv. Expedia test, all rows, "
            f"n = {num(n_all)}.",
            "Nguồn: results/week5_final/slices_history_length.csv. Expedia test, mọi dòng, "
            f"n = {num(n_all)}.",
        ),
    )

    # 15 known vs new
    k = d["known"][d["known"]["row_set"] == "all"]

    def kv(b, key):
        return float(k[(k.bucket == b) & (k.model_key == key)]["ndcg@5"].iloc[0])

    bo = d["boot"]
    ex = bo[(bo.row_set == "warm excluding re-books within_1d") & (bo.metric == "ndcg@5")].iloc[0]
    slide(
        T(
            f"History pays where guests return: sameDest lifts known-destination NDCG@5 from "
            f"{f3(kv('known destination', 'query_token'))} to {f3(kv('known destination', 'main'))}",
            f"Lịch sử có giá trị ở nơi khách quay lại: sameDest nâng NDCG@5 tại điểm đến đã biết từ "
            f"{f3(kv('known destination', 'query_token'))} lên {f3(kv('known destination', 'main'))}",
        ),
        f'<div class="fig">{fig_known(d)}</div>'
        + T(
            f'<p class="lede">Without re-books within one day, the sameDest gain on warm rows is still '
            f"{sgn3(float(ex['diff']))} NDCG@5.</p>",
            f'<p class="lede">Loại các lượt đặt lại trong vòng một ngày, mức tăng của sameDest trên dòng '
            f"warm vẫn là {sgn3(float(ex['diff']))} NDCG@5.</p>",
        ),
        T(
            "Known destination means the searched destination appears in the guest's last 20 bookings, "
            "the same definition sameDest uses. There sameDest adds a lot; on new destinations its weight "
            "is zero, so the bars are identical. Logistic regression does not use history and stays flat. "
            f"Removing targets that re-book the same cluster at the same destination within a day "
            f"({int(ex['n']):,} rows remain) keeps a gain of {sgn3(float(ex['diff']))} (95% CI "
            f"{sgn3(float(ex['ci95_lo']))} to {sgn3(float(ex['ci95_hi']))}); this conditions on the label, "
            "so it is a lower bound. Source: slices_destination_known.csv, bootstrap_ci.csv.",
            "Điểm đến đã biết nghĩa là điểm đến đang tìm xuất hiện trong 20 lượt đặt gần nhất của khách, "
            "cùng định nghĩa mà sameDest dùng. Ở đó sameDest thêm rất nhiều; ở điểm đến mới trọng số của "
            "nó bằng không, nên các cột giống hệt nhau. Logistic regression không dùng lịch sử nên đứng "
            "phẳng. Loại các mục tiêu đặt lại cùng cluster tại cùng điểm đến trong vòng một ngày "
            f"(còn {int(ex['n']):,} dòng) vẫn giữ mức tăng {sgn3(float(ex['diff']))} (KTC 95% "
            f"{sgn3(float(ex['ci95_lo']))} đến {sgn3(float(ex['ci95_hi']))}); cách này điều kiện hóa "
            "theo nhãn nên chỉ là cận dưới. Nguồn: slices_destination_known.csv, bootstrap_ci.csv.",
        ),
        src=T(
            "Source: results/week5_final/slices_destination_known.csv, bootstrap_ci.csv. Expedia test, "
            "warm rows split by destination.",
            "Nguồn: results/week5_final/slices_destination_known.csv, bootstrap_ci.csv. Expedia test, "
            "dòng warm chia theo điểm đến.",
        ),
    )

    # 16 trip context
    comp = eda["E7"]["composition"]["test"]["checkin_month"]
    slide(
        T(
            "The model order is the same for every party type and for package trips",
            "Thứ tự các mô hình giống nhau ở mọi loại nhóm khách và ở chuyến đi có gói",
        ),
        f'<div class="fig">{fig_context(d)}</div>'
        + T(
            f'<p class="lede">Season is not tested: {pct(comp["12"])} of test check-ins fall in December '
            f"and {pct(comp['1'])} in January.</p>",
            f'<p class="lede">Mùa vụ chưa được kiểm tra: {pct(comp["12"])} check-in của tập test rơi vào '
            f"tháng Mười Hai và {pct(comp['1'])} vào tháng Một.</p>",
        ),
        T(
            "NDCG@5 by party type (solo, couple, family with children, group of three or more adults) and "
            "by package. The main model is first in every bucket, logistic regression second, ItemKNN "
            "last. The season slice is degenerate: the test bookings are made between mid-November and the "
            "end of December, so check-in months cluster in December and January. A seasonal answer would "
            "need a rolling-origin split or pooled valid and test rows, which is an open decision. Source: "
            "slices_party_type.csv, slices_is_package.csv, eda_clean.json (E7).",
            "NDCG@5 theo loại nhóm khách (một mình, cặp đôi, gia đình có trẻ em, nhóm từ ba người lớn trở "
            "lên) và theo gói. Mô hình chính đứng đầu ở mọi nhóm, logistic regression thứ hai, ItemKNN "
            "cuối. Lát cắt mùa vụ bị suy biến: các lượt đặt của tập test diễn ra từ giữa tháng Mười Một đến "
            "cuối tháng Mười Hai, nên tháng check-in dồn vào tháng Mười Hai và tháng Một. Muốn trả lời về mùa vụ cần "
            "chia rolling-origin hoặc gộp dòng valid và test, đây là một quyết định còn mở. Nguồn: "
            "slices_party_type.csv, slices_is_package.csv, eda_clean.json (E7).",
        ),
        src=T(
            "Source: results/week5_final/slices_party_type.csv, slices_is_package.csv; "
            "results/week4_rebuild/eda_clean.json (E7). Expedia test, all rows.",
            "Nguồn: results/week5_final/slices_party_type.csv, slices_is_package.csv; "
            "results/week4_rebuild/eda_clean.json (E7). Expedia test, mọi dòng.",
        ),
    )

    # 17 RQ4
    dest_svg, booked = fig_dest(d)
    ba = d["ba"][d["ba"]["row_set"] == "all"].set_index("model_key")
    slide(
        T(
            "Accuracy costs no global diversity, the trade-off is narrower lists per destination",
            "Độ chính xác không tốn diversity toàn cục, cái giá là danh sách hẹp hơn theo từng điểm đến",
        ),
        f'<div class="two"><div class="fig">{fig_rq4(d)}</div>'
        + T(
            f'<div><p class="cap">Distinct clusters recommended per destination (top 10)</p>{dest_svg}'
            f'<p class="cap">Guests actually book {booked:.0f} distinct clusters per destination.</p></div></div>',
            f'<div><p class="cap">Số cluster khác nhau được gợi ý cho mỗi điểm đến (top 10)</p>{dest_svg}'
            f'<p class="cap">Thực tế khách đặt {booked:.0f} cluster khác nhau cho mỗi điểm đến.</p></div></div>',
        ),
        T(
            "Left: the search-aware models are the most accurate and also spread exposure most evenly over "
            f"the 100 clusters (Gini {f3(float(ba.loc['main', 'exposure_gini']))} for the main model vs "
            f"{f3(float(ba.loc['plain_all', 'exposure_gini']))} for the original). Their average "
            f"popularity matches what guests book (ratio {float(ba.loc['main', 'ARP_ratio']):.2f}). Right: "
            "per destination they concentrate on fewer clusters than the history-only models, close to "
            "the number guests really book. Diversity uses co-booking similarity because clusters are "
            "anonymous. Source: beyond_accuracy_aggregate.csv, beyond_accuracy_per_destination.csv.",
            "Bên trái: các mô hình có xét tìm kiếm chính xác nhất và cũng phân bổ mức hiển thị đều nhất "
            f"trên 100 cluster (Gini {f3(float(ba.loc['main', 'exposure_gini']))} của mô hình chính so với "
            f"{f3(float(ba.loc['plain_all', 'exposure_gini']))} của bản gốc). Độ phổ biến trung bình của "
            f"chúng khớp với những gì khách đặt (tỷ lệ {float(ba.loc['main', 'ARP_ratio']):.2f}). Bên "
            "phải: theo từng điểm đến, chúng tập trung vào ít cluster hơn các mô hình chỉ dùng lịch sử, "
            "gần với số cluster khách thực sự đặt. Diversity được tính bằng độ tương đồng co-booking vì "
            "cluster ẩn danh. Nguồn: beyond_accuracy_aggregate.csv, beyond_accuracy_per_destination.csv.",
        ),
        src=T(
            "Source: results/week5_final/beyond_accuracy_aggregate.csv, beyond_accuracy_per_destination.csv "
            "(destinations with at least 20 rows). Expedia test, all rows.",
            "Nguồn: results/week5_final/beyond_accuracy_aggregate.csv, beyond_accuracy_per_destination.csv "
            "(điểm đến có ít nhất 20 dòng). Expedia test, mọi dòng.",
        ),
    )

    # 18 RQ map
    slide(
        T(
            "RQ1 is answered for sparsity and trip context but not for season, RQ4 for bias and coverage",
            "RQ1 đã trả lời về độ thưa và ngữ cảnh chuyến đi nhưng chưa về mùa vụ, RQ4 về bias và coverage",
        ),
        table(
            [
                T("Question", "Câu hỏi"),
                T("Evidence", "Bằng chứng"),
                T("Answer", "Câu trả lời"),
            ],
            [
                [
                    T("RQ1, sparse users", "RQ1, người dùng thưa"),
                    T("slides 13, 14", "slide 13, 14"),
                    T(
                        "yes, on first bookings and every history length",
                        "có, trên lượt đặt đầu tiên và mọi độ dài lịch sử",
                    ),
                ],
                [
                    T("RQ1, trip context", "RQ1, ngữ cảnh chuyến đi"),
                    T("slides 5, 16", "slide 5, 16"),
                    T("yes, the destination drives it", "có, điểm đến là yếu tố quyết định"),
                ],
                [
                    T("RQ1, season", "RQ1, mùa vụ"),
                    T("slide 16", "slide 16"),
                    T("not answerable on this split", "chưa trả lời được với cách chia này"),
                ],
                [
                    T("RQ4, diversity and bias", "RQ4, diversity và bias"),
                    T("slide 17", "slide 17"),
                    T(
                        "no global cost, narrower per destination",
                        "không tốn toàn cục, hẹp hơn theo điểm đến",
                    ),
                ],
                ["RQ2, RQ3", "", T("out of scope", "ngoài phạm vi")],
            ],
            num_from=9,
        ),
        T(
            "This is the map from research question to evidence. RQ1 sparsity: first bookings and the "
            "history-length slice. RQ1 trip context: the information-gain chart and the party / package "
            "slice; the field permutation in appendix A2 confirms the destination dominates. RQ1 season: "
            "the test window covers only six weeks of bookings, so we say it cannot be answered here. "
            "RQ4: popularity bias, exposure Gini and coverage per destination. RQ2 and RQ3 remain out of "
            "scope. Source: docs/plans/final_prep/01_results_summary.md section 9.",
            "Đây là bản đồ từ câu hỏi nghiên cứu đến bằng chứng. RQ1 về độ thưa: lượt đặt đầu tiên và "
            "lát cắt theo độ dài lịch sử. RQ1 về ngữ cảnh chuyến đi: biểu đồ information gain và lát "
            "cắt nhóm khách / gói; phép hoán vị trường ở phụ lục A2 xác nhận điểm đến chi phối. RQ1 về "
            "mùa vụ: cửa sổ test chỉ gồm sáu tuần đặt phòng, nên chúng tôi nói rõ là không trả lời được "
            "ở đây. RQ4: popularity bias, exposure Gini và coverage theo điểm đến. RQ2 và RQ3 vẫn "
            "ngoài phạm vi. Nguồn: docs/plans/final_prep/01_results_summary.md mục 9.",
        ),
        cls="tbl",
    )

    # 19 demo
    demo_body, demo_note, demo_src = demo_slide(d)
    slide(
        T(
            "The demo replays real test bookings through the main model and both baselines",
            "Bản demo phát lại các lượt đặt thật của tập test qua mô hình chính và cả hai baseline",
        ),
        demo_body,
        demo_note,
        src=demo_src,
    )

    # 20 conclusion
    slide(
        T(
            "The current search is the main signal, history a smaller and real increment",
            "Tìm kiếm hiện tại là tín hiệu chính, lịch sử là phần bổ sung nhỏ hơn nhưng có thật",
        ),
        T(
            '<ul class="concl">'
            f"<li>Search-aware SMLP4Rec leads logistic regression by {sgn3(lw[0])} and ItemKNN by "
            f"{sgn3(kw[0])} NDCG@5 on warm rows.</li>"
            "<li>First bookings are scored well, with no history needed.</li>"
            "<li>sameDest helps returning guests at known destinations.</li></ul>"
            '<p class="lim">Limits: cluster is a proxy for room or package, one dataset, offline only, '
            "no in-stay data, two simple baselines, one seed.</p>"
            '<p class="lim2">Next: two more seeds, a rolling-origin split for season.</p>',
            '<ul class="concl">'
            f"<li>SMLP4Rec có xét tìm kiếm dẫn trước logistic regression {sgn3(lw[0])} và ItemKNN "
            f"{sgn3(kw[0])} NDCG@5 trên dòng warm.</li>"
            "<li>Lượt đặt đầu tiên được chấm tốt, không cần lịch sử.</li>"
            "<li>sameDest giúp khách quay lại tại điểm đến đã biết.</li></ul>"
            '<p class="lim">Giới hạn: cluster chỉ là proxy cho phòng hoặc gói, một dataset, chỉ offline, '
            "không có dữ liệu trong thời gian lưu trú, hai baseline đơn giản, một seed.</p>"
            '<p class="lim2">Tiếp theo: thêm hai seed, chia rolling-origin để xét mùa vụ.</p>',
        ),
        T(
            "Three takeaways. The search inside the model is where most of the gain comes from. The same "
            "token lets the model handle first bookings. History, through the model and sameDest, adds a "
            "smaller gain concentrated on returning guests. Limitations: the cluster is an anonymous proxy "
            "for room category or package with no price; one dataset; offline evaluation only; no public "
            "in-stay service data; only two simple baselines; the early-stopping runs are one seed (seed "
            "spread on 3-epoch runs is about 0.002 NDCG@5). Next: more seeds, a rolling-origin split for "
            "season.",
            "Ba điểm rút ra. Đưa tìm kiếm vào trong mô hình là nơi phần lớn mức tăng đến từ đó. Cùng "
            "token này giúp mô hình xử lý lượt đặt đầu tiên. Lịch sử, qua mô hình và sameDest, thêm một "
            "mức tăng nhỏ hơn, tập trung ở khách quay lại. Giới hạn: cluster là proxy ẩn danh cho hạng "
            "phòng hoặc gói và không có giá; một dataset; chỉ đánh giá offline; không có dữ liệu dịch "
            "vụ trong thời gian lưu trú công khai; chỉ có hai baseline đơn giản; các run early stopping "
            "chỉ có một seed (độ lệch giữa các seed ở run 3 epoch khoảng 0.002 NDCG@5). Tiếp theo: "
            "thêm seed, chia rolling-origin để xét mùa vụ.",
        ),
        src=T(
            f"Source: {BENCH}, bootstrap_ci.csv. Expedia test, warm rows.",
            f"Nguồn: {BENCH}, bootstrap_ci.csv. Expedia test, dòng warm.",
        ),
    )

    # A1 logistic regression sensitivity
    rows07 = []
    for rs in ("all", "cold", "warm"):
        r = lr07.loc[rs]
        rows07.append(
            [
                ALL if rs == "all" else rs,
                f3(m(d, "logreg", rs, "ndcg@5")),
                f3(float(r["ndcg@5"])),
                f3(m(d, "main", rs, "ndcg@5")),
                sgn3(float(r["main_minus_lr07_ndcg@5"])),
            ]
        )
    slide(
        T(
            "Appendix A1. A stronger logistic regression narrows the warm gap but does not close it",
            "Phụ lục A1. Logistic regression mạnh hơn thu hẹp khoảng cách warm nhưng không xóa được",
        ),
        table(
            [
                T("Rows", "Dòng"),
                T("LR 07b (reported)", "LR 07b (báo cáo)"),
                T("LR notebook 07", "LR notebook 07"),
                T("Main model", "Mô hình chính"),
                T("Main minus LR 07", "Chính trừ LR 07"),
            ],
            rows07,
        )
        + T(
            '<p class="fn">NDCG@5. Notebook 07: C = 10, three SAGA passes, trained before the training '
            "cut. 07b: same rows as the main model, C = 1.0 picked on valid.</p>",
            '<p class="fn">NDCG@5. Notebook 07: C = 10, ba lượt SAGA, train trước mốc cắt tập train. '
            "07b: cùng các dòng với mô hình chính, C = 1.0 chọn trên valid.</p>",
        ),
        T(
            "The reported baseline is notebook 07b because it uses exactly the main model's training rows "
            "and early stopping. A logistic regression from notebook 07 (C = 10) scores higher on the same "
            "test rows. The main model still leads on all three row sets, with a smaller warm gap. Which "
            "run is the headline baseline is an open decision; the rule so far is 07b with this footnote. "
            "Source: results/week5_final/sensitivity_logreg_nb07.csv.",
            "Baseline được báo cáo là notebook 07b vì nó dùng đúng các dòng train và early stopping của "
            "mô hình chính. Logistic regression từ notebook 07 (C = 10) đạt điểm cao hơn trên cùng các "
            "dòng test. Mô hình chính vẫn dẫn trước trên cả ba tập dòng, với khoảng cách warm nhỏ hơn. "
            "Chọn lần chạy nào làm baseline chính là quyết định còn mở; quy tắc hiện tại là 07b kèm "
            "chú thích này. Nguồn: results/week5_final/sensitivity_logreg_nb07.csv.",
        ),
        src=T(
            "Source: results/week5_final/sensitivity_logreg_nb07.csv, final_benchmark.csv. Expedia test.",
            "Nguồn: results/week5_final/sensitivity_logreg_nb07.csv, final_benchmark.csv. Expedia test.",
        ),
        cls="tbl",
    )

    # A2 diagnostics
    dg = run["diagnostics_test"]
    fp = run["field_permutation_test"]
    top_drop = sorted(fp.items(), key=lambda kv_: -kv_[1]["ndcg@5_drop"])[:3]
    rows_dg = [
        [T("real search and history", "tìm kiếm và lịch sử thật"), f3(dg["real"]["ndcg@5"])],
        [T("history masked", "che lịch sử"), f3(dg["history masked"]["ndcg@5"])],
        [
            T("search shuffled across rows", "xáo trộn tìm kiếm giữa các dòng"),
            f3(dg["query shuffled"]["ndcg@5"]),
        ],
        [T("both", "cả hai"), f3(dg["both (shuffled + masked)"]["ndcg@5"])],
    ]
    slide(
        T(
            "Appendix A2. Shuffling the search destroys accuracy, masking the history costs much less",
            "Phụ lục A2. Xáo trộn tìm kiếm phá hủy độ chính xác, che lịch sử tốn ít hơn nhiều",
        ),
        table(
            [T("Query token model, input", "Mô hình query token, đầu vào"), "NDCG@5"],
            rows_dg,
        )
        + T(
            '<p class="fn">Largest drops when one field is permuted: ',
            '<p class="fn">Mức giảm lớn nhất khi hoán vị một trường: ',
        )
        + ", ".join(f"{k_} {sgn3(-v['ndcg@5_drop'])}" for k_, v in top_drop)
        + ".</p>",
        T(
            "Controls on the query-token model (no sameDest), warm test rows. Masking the history costs "
            "some accuracy; shuffling the search across rows drops it close to popularity, so the model "
            "really uses the search. Permuting one field at a time shows the destination carries almost "
            "all of it. Source: results/week4_rebuild/smlprec_query_cold_train_es_run.json "
            "(diagnostics_test, field_permutation_test).",
            "Các phép đối chứng trên mô hình query token (không có sameDest), dòng test warm. Che lịch "
            "sử làm giảm một phần độ chính xác; xáo trộn tìm kiếm giữa các dòng kéo nó xuống gần mức "
            "phổ biến, nên mô hình thực sự dùng tìm kiếm. Hoán vị từng trường một cho thấy điểm đến "
            "mang gần như toàn bộ tác dụng. Nguồn: results/week4_rebuild/smlprec_query_cold_train_es_run.json "
            "(diagnostics_test, field_permutation_test).",
        ),
        src=T(
            "Source: results/week4_rebuild/smlprec_query_cold_train_es_run.json. Expedia test, warm rows, "
            f"n = {num(dg['real']['n'])}.",
            "Nguồn: results/week4_rebuild/smlprec_query_cold_train_es_run.json. Expedia test, dòng warm, "
            f"n = {num(dg['real']['n'])}.",
        ),
        cls="tbl",
    )


def demo_slide(d: dict) -> tuple[str, str, str]:
    demo_file = "demo_storyboard_vi.html" if LANG == "vi" else "demo_storyboard.html"
    link = f'<a href="../../../results/week5_final/{demo_file}">results/week5_final/{demo_file}</a>'
    demo = d["demo"]
    try:
        b = next(x for x in demo["bookings"] if x["key"] == "A")
        sd, bl = b["steps"]["samedest"], b["steps"]["baselines"]
        s = b["search"]
        tops = [
            (T("main model", "mô hình chính"), [t["cluster"] for t in sd["top"]], sd["truth_rank"]),
            ("logistic regression", bl["lr"]["top"], bl["lr"]["truth_rank"]),
            ("ItemKNN", bl["knn"]["top"], bl["knn"]["truth_rank"]),
        ]
        rows = []
        for name, top, rank in tops:
            cells = [name]
            for c in top[:5]:
                cells.append(f'<span class="hit">{c}</span>' if c == b["truth"] else str(c))
            cells.append(str(rank))
            rows.append(cells)
        party = T(
            f"{s['adults']} adults, {s['children']} children",
            f"{s['adults']} người lớn, {s['children']} trẻ em",
        )
        if "07b" in str(demo["models"].get("baselines", "")):  # read, not assumed
            base_note = T(
                "The demo baselines are the 07b models of the table.",
                "Các baseline trong demo là các mô hình 07b của bảng.",
            )
            base_src = T(
                "baselines = notebook 07b, as in the tables",
                "baseline = notebook 07b, như trong các bảng",
            )
        else:
            base_note = T(
                "The demo baselines come from notebook 07, so ranks can differ from the table.",
                "Các baseline trong demo lấy từ notebook 07, nên hạng có thể khác với bảng.",
            )
            base_src = T(
                "demo baselines from notebook 07, not the 07b table",
                "baseline của demo lấy từ notebook 07, không phải bảng 07b",
            )
        body = (
            T(
                f'<p class="lede">Booking A, a first booking: destination {s["dest"]}, check-in '
                f"{s['ci']}, {party}. Booked cluster {b['truth']}.</p>",
                f'<p class="lede">Booking A, lượt đặt đầu tiên: điểm đến {s["dest"]}, check-in '
                f"{s['ci']}, {party}. Cluster được đặt: {b['truth']}.</p>",
            )
            + table(
                ["", "1", "2", "3", "4", "5", T("rank of booked", "hạng của cluster được đặt")],
                rows,
                hi_row=0,
            )
            + T(
                f'<p class="fn">Open the storyboard: {link}</p>',
                f'<p class="fn">Mở storyboard: {link}</p>',
            )
        )
        note = T(
            f"The demo is a static storyboard of {len(demo['bookings'])} real test bookings picked for "
            "illustration, one of them a deliberate miss. In booking A the guest has no history: the "
            f"main model ranks the booked cluster {sd['truth_rank']}, logistic regression "
            f"{bl['lr']['truth_rank']}, and ItemKNN can only fall back to popularity "
            f"({bl['knn']['truth_rank']}). Say clearly that the examples are selected; the representative "
            f"numbers are the tables. {base_note} Source: results/week5_final/demo_storyboard.json.",
            f"Bản demo là một storyboard tĩnh gồm {len(demo['bookings'])} lượt đặt thật của tập test được "
            "chọn để minh họa, trong đó có một trường hợp cố ý chọn lỗi. Ở booking A khách không có lịch "
            f"sử: mô hình chính xếp cluster được đặt ở hạng {sd['truth_rank']}, logistic regression ở "
            f"hạng {bl['lr']['truth_rank']}, còn ItemKNN chỉ có thể quay về độ phổ biến "
            f"({bl['knn']['truth_rank']}). Nói rõ là các ví dụ được chọn lọc; số liệu đại diện là các "
            f"bảng. {base_note} Nguồn: results/week5_final/demo_storyboard.json.",
        )
        src = T(
            "Source: results/week5_final/demo_storyboard.json (picked for illustration; "
            f"{base_src}). Expedia test booking.",
            "Nguồn: results/week5_final/demo_storyboard.json (chọn để minh họa; "
            f"{base_src}). Lượt đặt thuộc Expedia test.",
        )
    except (TypeError, KeyError, StopIteration):
        body = T(
            f'<p class="lede">Storyboard of curated test bookings: {link}</p>',
            f'<p class="lede">Storyboard các lượt đặt test được chọn lọc: {link}</p>',
        )
        note = T(
            "The storyboard JSON was not available when the deck was built.",
            "File JSON của storyboard chưa có khi dựng bộ slide.",
        )
        src = ""
    return body, note, src


# ----------------------------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------------------------
CSS = """
:root{--bg:#ffffff;--ink:#20242a;--muted:#5c6166;--rule:#d9dadb;--main:#3b6ea5;--main-dark:#2f5a88;
--contrast:#c27c3f;--context:#a8a79f;--alert:#a63d32}
*{box-sizing:border-box}
html,body{margin:0;background:#e9e9e7;font-family:FONT;color:var(--ink)}
#stage{position:absolute;width:1280px;height:720px;transform-origin:0 0}
.slide{display:none;position:absolute;inset:0;background:var(--bg);overflow:hidden;padding:44px 64px 0 64px}
.slide.on{display:block}
h1{font-size:34px;line-height:1.18;font-weight:600;margin:0 0 28px 0;max-width:1152px;letter-spacing:-0.005em}
.body{font-size:21px;line-height:1.4}
.fig{margin-top:4px}
.row{display:flex;gap:48px;align-items:flex-start}
.two{display:grid;grid-template-columns:620px 1fr;gap:40px;align-items:start}
.aside{max-width:300px;color:var(--muted);font-size:19px;margin-top:24px}
.lede{margin:20px 0 0 0;max-width:1000px}
.cap{font-size:18px;color:var(--muted);margin:0 0 12px 0}
.muted{color:var(--muted)}
.rq p{margin:0 0 26px 0;max-width:980px;font-size:24px}
.rq b{font-weight:700;margin-right:10px}
.src{position:absolute;left:64px;right:120px;bottom:22px;font-size:13px;color:var(--muted);margin:0}
.pg{position:absolute;right:64px;bottom:22px;font-size:13px;color:var(--muted)}
table.t{border-collapse:collapse;font-size:19px;font-variant-numeric:tabular-nums;margin-top:4px;
border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}
table.t th{font-weight:600;text-align:left;padding:10px 16px;border-bottom:1px solid var(--ink);vertical-align:bottom}
table.t td{padding:9px 16px;border-bottom:1px solid var(--rule)}
table.t tr:last-child td{border-bottom:none}
table.t .n{text-align:right}
table.t tr.hi td{color:var(--main-dark);font-weight:600}
.hit{color:#fff;background:var(--main);padding:1px 7px}
.fn{font-size:17px;color:var(--muted);margin-top:16px;max-width:1150px}
.fn a{color:var(--main-dark)}
.concl{margin:8px 0 0 0;padding-left:26px;font-size:24px;line-height:1.45}
.concl li{margin-bottom:14px}
.lim2{font-size:19px;color:var(--muted);margin-top:8px}
.lim{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);font-size:19px;color:var(--muted);max-width:1100px}
.title{padding-top:220px}
.title h1{font-size:46px;max-width:1000px;margin-bottom:28px}
.sub{font-size:24px;margin:0 0 10px 0}
.sub2{font-size:20px;color:var(--muted);margin:0 0 56px 0}
.sub3{font-size:18px;color:var(--muted);margin:0;padding-top:14px;border-top:1px solid var(--rule);display:inline-block}
aside.notes{display:none}
body.shownotes .slide.on aside.notes{display:block;position:absolute;left:0;right:0;bottom:0;max-height:46%;
overflow:auto;background:#fff;border-top:2px solid var(--ink);padding:16px 64px 20px 64px;font-size:18px;line-height:1.45}
::selection{background:#cfdcec}
@page{size:1280px 720px;margin:0}
@media print{html,body{background:#fff}#stage{position:static;transform:none!important}
.slide{display:block!important;position:relative;width:1280px;height:720px;page-break-after:always;break-after:page}
aside.notes{display:none!important}}
"""

# Vietnamese: longer headers must not wrap mid-phrase, ledes may use the full width
VI_CSS = "table.t th{white-space:nowrap}.lede{max-width:1152px}"

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
function fit(){const st=document.getElementById('stage');const s=Math.min(innerWidth/1280,innerHeight/720);
st.style.transform=`scale(${s})`;st.style.left=((innerWidth-1280*s)/2)+'px';st.style.top=((innerHeight-720*s)/2)+'px';}
function show(k){i=Math.max(0,Math.min(S.length-1,k));S.forEach((e,j)=>e.classList.toggle('on',j===i));
try{history.replaceState(null,'','#'+(i+1));}catch(e){}}
addEventListener('keydown',e=>{const k=e.key;
if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].includes(k)){show(i+1);e.preventDefault();}
else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(k)){show(i-1);e.preventDefault();}
else if(k==='Home')show(0);else if(k==='End')show(S.length-1);
else if(k==='n'||k==='N')document.body.classList.toggle('shownotes');});
addEventListener('resize',fit);fit();show((parseInt(location.hash.slice(1))||1)-1);
"""


def render() -> str:
    vi = LANG == "vi"
    css = CSS.replace("FONT", font()) + (VI_CSS if vi else "")
    parts = []
    n = len(SLIDES)
    for k, s in enumerate(SLIDES, 1):
        cls = "slide" + (" title" if s["kind"] == "title" else "")
        src = f'<p class="src">{s["src"]}</p>' if s["src"] else ""
        pg = f'<span class="pg">{k} / {n}</span>' if s["kind"] != "title" else ""
        parts.append(
            f'<section class="{cls}" data-n="{k}"><h1>{s["title"]}</h1>'
            f'<div class="body">{s["body"]}</div>{src}{pg}'
            f'<aside class="notes">{esc(s["notes"])}</aside></section>'
        )
    web_font = (
        ""  # Vietnamese deck: system fonts only (Segoe UI / Arial), no network
        if vi
        else '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;'
        '0,700;1,400&display=swap" rel="stylesheet">'
    )
    return (
        f'<!doctype html><html lang="{LANG}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{T('Final slides v1', 'Slide cuối kỳ v1 (tiếng Việt)')}</title>"
        f"{web_font}"
        f'<style>{css}</style></head><body><div id="stage">{"".join(parts)}</div>'
        f"<script>{JS}</script></body></html>"
    )


# ----------------------------------------------------------------------------------------------
# QA: words per slide and banned patterns
# ----------------------------------------------------------------------------------------------
def words(fragment: str) -> int:
    t = re.sub(r"<svg.*?</svg>", " ", fragment, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    # tokens that start with a letter (English words; Vietnamese syllables); table numbers are not prose
    return len(re.findall(r"[^\W\d_][\w.,%+−-]*", html.unescape(t)))


def qa(page: str) -> list[str]:
    issues = []
    limit = WORD_LIMIT_VI if LANG == "vi" else WORD_LIMIT
    tlimit = TITLE_LIMIT_VI if LANG == "vi" else TITLE_LIMIT
    unit = "syllables" if LANG == "vi" else "words"
    print(
        f"\n{unit} per slide (body tokens with a letter, limit {limit}; title limit {tlimit}; "
        "source, notes, chart text excluded):"
    )
    for k, s in enumerate(SLIDES, 1):
        wb = words(s["body"])
        wt = len(s["title"].split())
        wn = len(s["notes"].split())
        flag = " <- over" if wb > limit and s["kind"] != "title" else ""
        print(f"  {k:2d}  body {wb:3d}  title {wt:2d}  notes {wn:3d}{flag}")
        if wt > tlimit:
            issues.append(f"slide {k}: title has {wt} words")
        if flag:
            issues.append(f"slide {k}: body has {wb} words")
    low = page.lower()
    style = " ".join(re.findall(r"<style>(.*?)</style>", page, flags=re.DOTALL)).lower()
    checks = {
        "gradient": "gradient" in low,
        "box-shadow": "box-shadow" in style,
        "text-transform uppercase": "uppercase" in style,
        "border-radius > 4px": bool(re.search(r"border-radius:\s*([5-9]|\d{2,})px", style)),
        "rounded bars (rx=)": ' rx="' in page,
        "middle dot meta strings": "·" in re.sub(r"<svg.*?</svg>", "", page, flags=re.DOTALL),
        "emoji": bool(re.search("[\U0001f300-\U0001faff☀-➿]", page)),
        "monospace": "monospace" in style or "consolas" in style,
        "tailwind / indigo colors": bool(
            re.search(r"#(1e3a8a|2563eb|0f766e|dc2626|94a3b8|e2e8f0|6366f1|eef2ff)", low)
        ),
    }
    vis = re.sub(r"<aside.*?</aside>", "", page, flags=re.DOTALL).lower()
    for w in BANNED_WORDS:
        checks[f"word '{w}'"] = bool(re.search(rf"\b{w}", vis))
    print("banned patterns:", {k: v for k, v in checks.items() if v} or "none found")
    issues += [f"banned pattern: {k}" for k, v in checks.items() if v]
    return issues


def main() -> None:
    global LANG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    LANG = ap.parse_args().lang
    out, fig = out_path(), fig_dir()
    d = load_data()
    build(d)
    page = render()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    fig.mkdir(parents=True, exist_ok=True)
    for name, body in FIGURES.items():
        (fig / f"{name}.svg").write_text(body, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(SLIDES)} slides) and {len(FIGURES)} figures in "
          f"{fig.relative_to(ROOT)}")  # fmt: skip
    issues = qa(page)
    print("QA issues:", issues or "none")


if __name__ == "__main__":
    main()
