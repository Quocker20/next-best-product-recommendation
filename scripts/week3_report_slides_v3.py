"""Build the week-3 report deck v3 (Vietnamese, HTML) in the visual style of the week-2 slides.

v3 restructures v2 around the argument instead of the order of work: plain is weak -> output-side fixes fail ->
destination is the missing signal (baseline evidence) -> two ways to inject it (late fusion vs query token,
including query token + sameDest and the L = 0 users) -> consolidated comparison -> robustness -> conclusion;
every term is introduced before a later slide uses it; numeric results are charts where possible, full tables and the formulas
live in the appendix. Numbers are read from results/week3_implementation/*.json and
results/week4_rebuild/*.json exactly as in v2 (same loaders, same shorthand variables).

Usage: python scripts/week3_report_slides_v3.py
Output: reports/summary/week3_implementation/week3_report_slides_v3.html (PDF: print with headless Chrome)
"""
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W3 = ROOT / "results" / "week3_implementation"
OUT = ROOT / "reports" / "summary" / "week3_implementation" / "week3_report_slides_v3.html"
WEEK2 = (ROOT / "scripts" / "week2_methodology_slides.py").read_text(encoding="utf-8")


def load(name: str) -> dict:
    return json.loads((W3 / name).read_text(encoding="utf-8"))


RUN = load("smlprec_expedia_run.json")
MIX = load("smlprec_expedia_mixed_run.json")
CAP = load("smlprec_expedia_dynamic_cap_run.json")
LF = load("smlprec_expedia_late_fusion.json")
HY = load("smlprec_expedia_hybrid_samedest.json")
BS = load("smlprec_expedia_hybrid_behaviour_split.json")
SU = load("smlprec_expedia_seen_unseen_users.json")
W4 = ROOT / "results" / "week4_rebuild"
BL = json.loads((W4 / "basic_baselines.json").read_text(encoding="utf-8"))
BV = json.loads((W4 / "basic_baselines_verify.json").read_text(encoding="utf-8"))

# ---- style, JS and the architecture diagram come from the week-2 generator
CSS = re.search(r'\nCSS = """(.*?)"""', WEEK2, re.DOTALL).group(1)
JS = re.search(r'\nJS = """(.*?)"""', WEEK2, re.DOTALL).group(1)
NAVY, BLUE, TEAL, AMBER, RED, GRAY = (
    "#1e3a8a",
    "#2563eb",
    "#0f766e",
    "#b45309",
    "#dc2626",
    "#94a3b8",
)
_ns = {
    "NAVY": NAVY,
    "BLUE": BLUE,
    "TEAL": TEAL,
    "AMBER": AMBER,
    "RED": RED,
    "GRAY": GRAY,
}
_start = WEEK2.index("def architecture_svg")
_end = _start + 1 + re.search(r"\n(?=\S)", WEEK2[_start + 1 :]).start()
exec(WEEK2[_start:_end], _ns)  # noqa: S102 (reuse the week-2 diagram code)
architecture_svg = _ns["architecture_svg"]

CSS += """
.tbl.dense td{padding:11px 14px;font-size:15.5px}.tbl.dense th{padding:9px 14px;font-size:12px}
.tbl.dense tr.hl td{background:#f0fdfa;color:#134e4a;font-weight:600}
.tbl.dense td.r,.tbl.dense th.r{text-align:right}
.sm{font-size:12.5px;color:var(--sub);font-weight:500}
.kpis.three{grid-template-columns:repeat(3,1fr)}.kpis.two{grid-template-columns:repeat(2,1fr)}
.kpi>b.s{font-size:26px}
.flow.six{gap:8px}.flow.six .fbox{min-width:0;flex:1;padding:12px 8px;font-size:14.5px}.flow.six .farrow{font-size:20px}
.cmp2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.eq{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 20px;font-family:Consolas,monospace;font-size:17px;color:#1e3a8a;line-height:1.7}
.ex td,.ex th{padding:8px 12px}
.lg{white-space:nowrap}.legend{flex-wrap:wrap;gap:6px 16px}
"""


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def dec(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def pp(x: float, d: int = 2) -> str:
    return f"{x * 100:+.{d}f}".replace(".", ",") + " điểm"


def day(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%d/%m/%Y")


def table(head, rows, hl=(), cls="dense", right_from=1) -> str:
    th = "".join(
        f'<th class="{"r" if i >= right_from else ""}">{h}</th>' for i, h in enumerate(head)
    )
    trs = ""
    for ri, r in enumerate(rows):
        c = ' class="hl"' if ri in hl else ""
        trs += (
            f"<tr{c}>"
            + "".join(
                f'<td class="{"r" if i >= right_from else ""}">{x}</td>' for i, x in enumerate(r)
            )
            + "</tr>"
        )
    return f'<table class="tbl {cls}"><tr>{th}</tr>{trs}</table>'


def _lbl(x: float, y: float, text: str, cls: str = "cl") -> str:
    """SVG label; '|' splits it into lines."""
    t = "".join(
        f'<tspan x="{x:.1f}" dy="{0 if i == 0 else 14}">{p}</tspan>' for i, p in enumerate(text.split("|"))
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}">{t}</text>'


def grouped_bars(series, groups, ymax=0.7, w=640, h=300, as_pct=True, grid=(0.2, 0.4, 0.6), bottom=30) -> str:
    """series: [(label, values, color)], groups: category labels ('|' = line break), inline SVG."""
    top, left = 24, 6
    ph, gw = h - top - bottom, (w - left) / len(groups)
    bw = min(gw * 0.22, 46)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in grid:
        y = top + ph * (1 - g / ymax)
        out.append(f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(
            f'<text x="{left}" y="{y - 4:.1f}" font-size="11" fill="#94a3b8">{f"{round(g * 100)}%" if as_pct else dec(g, 1)}</text>'
        )
    for gi, g in enumerate(groups):
        span = len(series) * bw + 6 * (len(series) - 1)
        x0 = left + gi * gw + (gw - span) / 2
        for si, (_, vals, col) in enumerate(series):
            v = vals[gi]
            bh = ph * v / ymax
            x = x0 + si * (bw + 6)
            y = top + ph - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" class="cv">{pct(v, 1) if as_pct else dec(v, 3)}</text>')
        out.append(_lbl(x0 + span / 2, top + ph + 18, g))
    out.append("</svg>")
    legend = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def bar_chart(labels, values, colors, ymax, w=640, h=270, as_pct=False, legend=None) -> str:
    """One bar per label ('|' = line break); optional legend [(name, color)]."""
    top, bottom, left = 26, 46, 8
    ph, step = h - top - bottom, (w - left - 8) / len(values)
    bw = min(step * 0.6, 90)
    out = [
        f'<svg viewBox="0 0 {w} {h}" class="chart">',
        f'<line x1="{left}" y1="{top + ph}" x2="{w - 8}" y2="{top + ph}" stroke="#cbd5e1"/>',
    ]
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        x = left + i * step + (step - bw) / 2
        bh = ph * v / ymax
        out.append(f'<rect x="{x:.1f}" y="{top + ph - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>')
        out.append(
            f'<text x="{x + bw / 2:.1f}" y="{top + ph - bh - 6:.1f}" class="cv">{pct(v, 1) if as_pct else dec(v, 3)}</text>'
        )
        out.append(_lbl(x + bw / 2, top + ph + 18, lab))
    out.append("</svg>")
    lg = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, c in (legend or []))
    return "".join(out) + (f'<div class="legend">{lg}</div>' if legend else "")


def line_chart(xs, series, ymin, ymax, w=640, h=300, as_pct=True) -> str:
    """Categorical x axis. series: [(label, values, color)]."""
    left, right, top, bottom = 44, 14, 20, 34
    pw, ph = w - left - right, h - top - bottom
    step = pw / (len(xs) - 1)

    def X(i):
        return left + i * step

    def Y(v):
        return top + ph * (1 - (v - ymin) / (ymax - ymin))

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    n_ticks = 5
    for t in range(n_ticks + 1):
        v = ymin + (ymax - ymin) * t / n_ticks
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e2e8f0"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" font-size="11" fill="#94a3b8" text-anchor="end">{f"{round(v * 100)}%" if as_pct else dec(v, 2)}</text>'
        )
    for i, xv in enumerate(xs):
        out.append(f'<text x="{X(i):.1f}" y="{h - 10}" class="cl">{xv}</text>')
    for si, (label, vals, col) in enumerate(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}"/>')
        bi = max(range(len(vals)), key=lambda k: vals[k])
        out.append(
            f'<text x="{X(bi):.1f}" y="{Y(vals[bi]) + (-9 if si == 0 else 20):.1f}" class="cv">{pct(vals[bi], 1) if as_pct else dec(vals[bi], 3)}</text>'
        )
    out.append("</svg>")
    legend = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


slides = []


def slide(meta, title, body, sub=""):
    slides.append((meta, title, sub, body))


# ------------------------------------------------------------------ shorthand data
T, V = RUN["test"], RUN["valid"]
PE = RUN["per_epoch"]
POP = RUN["heuristics_same_test_rows"]["global_popularity"]
DS, CFG, SP = RUN["dataset_stats"], RUN["config"], RUN["splits"]
MIXT = {k: MIX["final_test_at_best_mixed_epoch"][k] for k in ("plain", "mixed")}
CAPT = {k: CAP["final_test_at_best_capped_epoch"][k] for k in ("plain", "capped")}
TB = RUN["top5_behavior"]
MX = RUN["target_mix"]
LFB = LF["best_global_w_on_valid"]
FKEY = f"fusion_global_w={LFB}"
# warm test rows (users with history): plain / prior alone / SMLP4Rec + prior from notebook 03
LV = LF["variants"]["test"]
LC = LF["combined_warm_plus_cold"]["test"]
K_PLAIN, K_PRIOR, K_FUSION = "plain_model (w=0)", "destination_prior_only", FKEY
WARM = {k: LV[k]["all"] for k in (K_PLAIN, K_PRIOR, K_FUSION)}
ALLEV = {
    K_PLAIN: LC["plain SMLP4Rec on warm + global popularity on cold"],
    K_PRIOR: LC["prior-only for everyone"],
    K_FUSION: LC["hybrid: fusion on warm + prior-only on cold"],
}
# hybrid with sameDest from notebook 04
K_HYBRID = "SMLP4Rec + prior + sameDest (this notebook)"
WARM[K_HYBRID] = HY["benchmark"]["test"]["warm"][K_HYBRID]
ALLEV[K_HYBRID] = HY["benchmark"]["test"]["all_events"][K_HYBRID]
HB = HY["bootstrap_test_95ci"]["hybrid minus SMLP4Rec + prior"]["warm"]
HW_CELLS = HY["cohorts"]["final_weights_per_cell"]  # "0" = destination new to the user, "1" = known
H_ALPHA = HY["selection"]["one_se_pick"][0]
H_SLICE = HY["test_slices_recall@5_ndcg@5"]
H_REF = next(k for k in HY["benchmark"]["test"]["warm"] if k != K_HYBRID)
EX12 = 0.7**3 + 0.7**0  # worked sameDest example: cluster 12 booked at the destination at ages 0 and 3
# behaviour split from notebook 05
BW, BA = BS["benchmark_test"]["warm"], BS["benchmark_test"]["all_events"]
BB = BS["bootstrap_test_95ci"]["best behaviour scheme minus reference"]["warm"]
B_CUT = BS["cut"]["chosen_on_train"]
B_BEST = BS["best_behaviour_scheme"]
B_CELLS = BS["final_weights_per_cell"]

# ================================================================== slides
# Flow: lý thuyết -> chạy plain -> rút ra gì -> thử nghiệm theo hướng nào -> kết luận.
# Only Recall@K and NDCG@K are reported (CLAUDE.md section 8, decision 2026-10-02); NDCG leads (user preference 2026-10-05).
FUS_TEST = LV[FKEY]
COLD_T = LF["cold_users_prior_only"]["test"]["prior_only"]
COLD_NDCG5 = COLD_T["ndcg@5"]
COLD_SHARE = LC["cold_share_of_events"]
TT = TB["test"]
RATIO = T["ndcg@5"] / POP["ndcg@5"]
mp, mm = MIXT["plain"], MIXT["mixed"]
cp, cc = CAPT["plain"], CAPT["capped"]
exch_fixed = (mp["old_target_hit_of_all_rows"] - mm["old_target_hit_of_all_rows"]) / (
    mm["new_target_hit_of_all_rows"] - mp["new_target_hit_of_all_rows"]
)
exch_cap = (cp["old_target_hit_of_all_rows"] - cc["old_target_hit_of_all_rows"]) / (
    cc["new_target_hit_of_all_rows"] - cp["new_target_hit_of_all_rows"]
)


def hits(d):
    return pct(d["old_target_hit_of_all_rows"]), pct(d["new_target_hit_of_all_rows"])


# ================================================================== definitions reused from the v2 analysis
wgrid = [str(w) for w in LF["w_grid"]]
def slice_pair(name: str, key: str) -> tuple:
    r, n = H_SLICE[name][key].split(" / ")
    return float(r), float(n)


def ci(d: dict, m: str) -> str:
    lo, hi = d[m]["ci95"]
    return f"{pp(d[m]['diff'])} [{dec(lo * 100, 2)}; {dec(hi * 100, 2)}]"


KD_REF, KD_HYB = slice_pair("known destination", H_REF), slice_pair("known destination", K_HYBRID)
ND_REF, ND_HYB = slice_pair("new destination", H_REF), slice_pair("new destination", K_HYBRID)
W_NEW, W_KNOWN = HW_CELLS["0"], HW_CELLS["1"]
PR = SU["presence"]
SR = SU["results"]["test"]
PL, HYK = "plain SMLP4Rec", "hybrid (SMLP4Rec + prior + sameDest)"
SUBS = [("seen users", "Đã thấy"), ("unseen users", "Chưa thấy (tất cả)"), ("unseen, L>=1", "Chưa thấy, L ≥ 1"), ("unseen, L=0 (first booking)", "Chưa thấy, L = 0*")]
su_gain = {s: SR[s][HYK]["ndcg@5"] - SR[s][PL]["ndcg@5"] for s, _ in SUBS}
BR = BL["results"]["test"]
KNN_K = next(k for k in BR if k.startswith("ItemKNN"))
LR_K = next(k for k in BR if k.startswith("Logistic"))
GLOB_K = "global popularity"
KM, LM = BL["models"][KNN_K], BL["models"][LR_K]
BSP = BL["split"]
CTRL = BL["verification"]["controls_test_warm"]
VER_MISMATCH = (
    sum(v["rank_mismatches"] for v in BV["logreg"].values() if isinstance(v, dict))
    + BV["itemknn"]["test_sample_rank_mismatches"]
    + len(BV["metrics_recomputed_from_saved_ranks"]["mismatches"])
)
BL_ALL = BSP["slice_sizes_vs_reference"]["test"]["all events"]
ROWS_BB = [
    ("Popularity toàn cục (không học)", BR[GLOB_K]),
    (f"ItemKNN (K = {KM['K_pick']}): chỉ lịch sử", BR[KNN_K]),
    ("SMLP4Rec plain: chỉ lịch sử", SU["results"]["test"]),
    (f"Hồi quy logistic (C = {dec(LM['C_pick'], 0)}): điểm đến + ngữ cảnh", BR[LR_K]),
    ("Hybrid: lịch sử + prior + sameDest", SU["results"]["test"]),
]
SL_BB = ["all events", "seen users", "unseen, L>=1", "unseen, L=0 (first booking)"]


def bb_cell(i: int, sl: str, m: str) -> float:
    d = ROWS_BB[i][1]
    if i == 2:
        return d[sl][PL][m]
    if i == 4:
        return d[sl][HYK][m]
    return d[sl][m]


bb_rows = [
    [ROWS_BB[i][0], dec(bb_cell(i, "all events", "ndcg@5"), 3), pct(bb_cell(i, "all events", "recall@5"), 1)]
    + [dec(bb_cell(i, sl, "ndcg@5"), 3) for sl in SL_BB[1:]]
    for i in range(len(ROWS_BB))
]
L0 = "unseen, L=0 (first booking)"
d_dest = bb_cell(3, "all events", "ndcg@5") - bb_cell(0, "all events", "ndcg@5")
d_hist = bb_cell(4, "all events", "ndcg@5") - bb_cell(3, "all events", "ndcg@5")
QT = json.loads((W4 / "smlprec_query_run.json").read_text(encoding="utf-8"))
QTT, QTP = QT["test"], QT["plain_rescored_test"]
QCFG, QCOV, Q_TIME = QT["config"], QT["query_coverage"], QT["timing_seconds"]
PF = QT["prior_fusion_check"]
PF_W = PF["chosen_w_on_valid_recall@5"]
PF_ALONE, PF_FUSED = "query token alone (w=0)", f"query token + prior (w={PF_W})"
PF_T, PF_B, PF_S = PF["test"], PF["bootstrap_fused_minus_query_token_test"], PF["test_slices_recall@5_ndcg@5"]
QD = QT["diagnostics_test"]
Q_PRIOR = WARM[K_PRIOR]
Q_SIM = WARM[K_FUSION]["ndcg@5"] - QTT["ndcg@5"]  # plain + prior minus query token (NDCG@5)
Q_GAP = WARM[K_HYBRID]["ndcg@5"] - QTT["ndcg@5"]  # hybrid minus query token (NDCG@5)
Q_FIELDS = [
    ["Điểm đến", "mã điểm đến (embedding riêng nếu có ≥ 5 lần ở train), loại điểm đến"],
    ["Lịch", "tháng check-in, số ngày từ lúc tìm đến check-in, số đêm"],
    ["Nhóm khách", "số người lớn, số trẻ em, số phòng"],
    ["Gói và kênh", "gói, thiết bị di động, kênh, site, châu lục điểm bán, quốc gia khách"],
]
Q_SERIES = [
    ("SMLP4Rec plain", [QTP["ndcg@5"], QTP["ndcg@10"]], GRAY),
    ("+ query token", [QTT["ndcg@5"], QTT["ndcg@10"]], BLUE),
    ("plain + prior", [WARM[K_FUSION]["ndcg@5"], WARM[K_FUSION]["ndcg@10"]], AMBER),
    ("hybrid (+ sameDest)", [WARM[K_HYBRID]["ndcg@5"], WARM[K_HYBRID]["ndcg@10"]], TEAL),
]
Q_ROWS = [
    ["SMLP4Rec plain: chỉ lịch sử", dec(QTP["ndcg@5"], 4), pct(QTP["recall@5"])],
    ["SMLP4Rec + query token", dec(QTT["ndcg@5"], 4), pct(QTT["recall@5"])],
    ["SMLP4Rec + prior (nb 03)", dec(WARM[K_FUSION]["ndcg@5"], 4), pct(WARM[K_FUSION]["recall@5"])],
    ["Hybrid + sameDest (nb 04)", dec(WARM[K_HYBRID]["ndcg@5"], 4), pct(WARM[K_HYBRID]["recall@5"])],
]
Q_BOOT = QT["bootstrap_query_minus_plain_test"]["all warm rows"]
Q_DIAG = [
    ["Thật (có lịch sử, có query)", QD["real"]],
    ["Xáo query giữa các dòng", QD["query shuffled"]],
    ["Che lịch sử, giữ query", QD["history masked"]],
    ["Vừa xáo vừa che", QD["both (shuffled + masked)"]],
]
Q_SL = [
    ("Phổ biến (từ 200 booking)", "destination 200+"),
    ("20–199 booking", "destination 20-199"),
    ("Hiếm (1–19 booking)", "destination 1-19 bookings in prior data"),
    ("Ngoài vocab query token", "destination out of query-token vocab"),
]
Q_SL_ROWS = [
    [
        lab,
        num(PF_S[f"{key} | query token alone"]["n"]),
        dec(PF_S[f"{key} | query token alone"]["ndcg@5"], 3),
        dec(PF_S[f"{key} | query token + prior"]["ndcg@5"], 3),
        dec(PF_S[f"{key} | prior only"]["ndcg@5"], 3),
    ]
    for lab, key in Q_SL
]
GRID = HY["selection"]["grid"]
CELL_VI = {
    "destination new to the user": "Điểm đến mới với người dùng",
    "destination already in history": "Điểm đến đã đặt trước đó",
    "new dest, rest": "Điểm đến mới · còn lại",
    "new dest, old-leaning": "Điểm đến mới · thiên cũ",
    "known dest, rest": "Đã đặt · còn lại",
    "known dest, old-leaning": "Đã đặt · thiên cũ",
}


def gl(vals) -> str:
    return " · ".join(dec(v, 2).rstrip("0").rstrip(",") if v % 1 else str(int(v)) for v in vals)


def ws_cell(label: str, w: float) -> str:
    """w_s is constant (no effect) when the query destination is new to the user."""
    return "— (hằng số)" if "new" in label.split(",")[0].lower() or label.startswith("destination new") else dec(w, 2)


b2_rows = [
    [CELL_VI[lab], dec(HW_CELLS[c][1], 2), dec(HW_CELLS[c][2], 2) if c == "1" else "— (hằng số)"]
    for c, lab in (("0", "destination new to the user"), ("1", "destination already in history"))
]
b3_rows = [
    [CELL_VI[lab], dec(w[0], 2), ws_cell(lab, w[1])] for lab, w in B_CELLS[B_BEST].items()
]
N_PAIRS = len(GRID["w_p"]) * len(GRID["w_s"])

LR_ALL = bb_cell(3, "all events", "ndcg@5")
KNN_ALL = bb_cell(1, "all events", "ndcg@5")
PLAIN_ALL = bb_cell(2, "all events", "ndcg@5")
HYB_ALL = bb_cell(4, "all events", "ndcg@5")
POP_ALL = bb_cell(0, "all events", "ndcg@5")
N_WARM = SP["test"]["targets"]


# ---- query token + sameDest (notebook 01c, section 5) and L = 0 rows
SD = QT["samedest_hybrid"]
SDW, SDA, SDB, SDT, SDS = (
    SD["benchmark_test"]["warm"],
    SD["benchmark_test"]["all_events"],
    SD["bootstrap_test_95ci"],
    SD["tuned_on_valid"],
    SD["test_slices_recall@5_ndcg@5"],
)
N_PL, N_PLP, N_OLD = "SMLP4Rec plain", "SMLP4Rec plain + prior (nb 03)", "Hybrid cũ: plain + prior + sameDest (nb 04)"
N_QT, N_QTP, N_A, N_B = "Query token", "Query token + prior (w = 0.25)", "Query token + sameDest", "Query token + prior + sameDest"


def sd_boot(n: str) -> dict:
    return SDB[f"{n} minus {N_OLD}"]


SD_KNOWN, SD_NEW = SDS["known destination"], SDS["new destination"]
SD_N_KNOWN, SD_N_NEW = SD_KNOWN["rows"], SD_NEW["rows"]


def sd_ndcg(sl: dict, name: str) -> float:
    return float(sl[name].split(" / ")[1])


CL = json.loads((W4 / "smlprec_query_cold_L0.json").read_text(encoding="utf-8"))
CLT, CLS = CL["results"]["test"], CL["test_slices_ndcg5_recall5"]
CL_POP, CL_PRI, CL_QT, CL_QTP = "global popularity", "destination prior only", "query token alone (empty history)", "query token + prior (w=0.25)"
CL_BOOT = CL["bootstrap_test_95ci"]
CL_IN, CL_OUT = "destination in query-token vocab", "destination out of vocab"

# ================================================================== slides (v3)
# Flow: lý thuyết + triển khai -> plain yếu -> sửa đầu ra không giúp -> điểm đến là tín hiệu thiếu (bằng chứng baseline)
# -> Cách 1 (prior + sameDest) -> Cách 2 (query token) -> query token + sameDest -> user mới (L = 0) -> gộp lại
# -> độ bền -> kết luận -> phụ lục.
slide_cover = (
    "cover",
    "",
    "",
    """<div class="cover">
  <div class="kicker">Tuần 3 · Triển khai &amp; thử nghiệm</div>
  <h1>SMLP4Rec trên Expedia:<br><span>điểm đến là tín hiệu quyết định, lịch sử thêm phần còn lại</span></h1>
  <p class="lead">Next-Best-Product Recommendation · Vinpearl × GSM · Bối cảnh (a): du lịch</p>
  <div class="cover-tags"><span>Plain còn yếu</span><span>Điểm đến là tín hiệu thiếu</span><span>Hai cách đưa vào</span><span>Query token + sameDest</span><span>Độ bền</span></div>
</div>""",
)
slides.append(slide_cover)

# ---- 1. lý thuyết + triển khai
slide(
    "1 · Lý thuyết và triển khai",
    "SMLP4Rec: đoán cụm khách sạn tiếp theo từ lịch sử booking",
    f"""
<div class="row">
  <div class="panel diagram-wrap">{architecture_svg()}</div>
  <div class="panel side steps">
    <div class="step"><span>1</span><p>Đầu vào: chuỗi các cụm đã đặt (và query token của lượt search).</p></div>
    <div class="step"><span>2</span><p>Ba MLP song song trộn theo <b>thời gian</b>, <b>context</b>, <b>embedding</b>, lặp L lần.</p></div>
    <div class="step"><span>3</span><p>Lấy hàng cuối, so với 100 cụm → điểm xếp hạng.</p></div>
  </div>
</div>""",
    sub="Chọn làm phương pháp chính ở tuần 2 (Gao et al., ACM TOIS 2024); tuần 3 chạy trên Expedia",
)

flow_steps = [
    ("in", "train.csv", "4 GB, đọc theo khối"),
    ("in", "Lọc booking", "giữ is_booking = 1"),
    ("prior", "expedia.inter", "user · cụm · giờ"),
    ("gate", "RecBole", "chuỗi + chia thời gian"),
    ("model", "SMLP4Rec", "CE trên 100 cụm"),
    ("out", "Đánh giá", "xếp hạng đầy đủ"),
]
flow_html = '<div class="farrow">→</div>'.join(
    f'<div class="fbox {k}">{a}<small>{b}</small></div>' for k, a, b in flow_steps
)
slide(
    "1 · Lý thuyết và triển khai",
    "Từ code gốc đến bản chạy với Expedia",
    f"""<div class="flow six">{flow_html}</div>
<div class="cmp2">
  <div class="panel"><div class="ph">Chỉnh sửa tối thiểu để chạy được</div>
  {
        table(
            ["Hạng mục", "Code gốc", "Bản cho Expedia"],
            [
                ["Dữ liệu", "MovieLens, item có feature", "Booking Expedia, 100 cụm không có feature"],
                ["Chuỗi", "Dài 50, feature bắt buộc", f"Dài {CFG['MAX_ITEM_LIST_LENGTH']}, cho phép rỗng"],
                ["Chia và chấm", "Leave-one-out, 100 mẫu", "Theo thời gian 80/10/10, xếp hạng đủ 100 cụm"],
                ["Cấu hình", "Theo paper, chậm trên CPU", "2 lớp, hidden 64, dropout 0"],
            ],
            right_from=9,
        )
    }</div>
  <div class="panel"><div class="ph">Chia theo thời gian, mục tiêu “đặt tiếp theo”</div>
  {
        table(
            ["Tập", "Số mục tiêu", "Từ ngày", "Đến ngày"],
            [
                [n, num(SP[k]["targets"]), day(SP[k]["first_target_unix"]), day(SP[k]["last_target_unix"])]
                for k, n in (("train", "Train"), ("valid", "Valid"), ("test", "Test"))
            ],
        )
    }
  <p class="sm">{num(DS["users"])} người dùng · {DS["items"]} cụm · {num(DS["sequence_targets"])} mục tiêu (mọi booking trừ lần đầu của mỗi người).</p></div>
</div>
<div class="note">RecBole 1.0.1 chạy trong venv Python 3.11 riêng, 2 bản vá nhỏ (đánh dấu <code>ADAPTED</code>). Giữ nguyên khối dùng chung cho mọi lớp, không residual.</div>""",
    sub="Bản chưa tối ưu: mục đích là có số liệu để so sánh",
)

# ---- 2. plain
slide(
    "2 · Plain: điểm yếu",
    "Plain chỉ dùng lịch sử: có tín hiệu nhưng điểm thấp",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@K, tập test, user có lịch sử, xếp hạng đủ 100 cụm</div>
    {
        grouped_bars(
            [
                ("Popularity toàn cục", [POP[f"ndcg@{k}"] for k in (5, 10, 20)], GRAY),
                (f"SMLP4Rec plain (epoch {RUN['best_epoch_by_valid']})", [T[f"ndcg@{k}"] for k in (5, 10, 20)], BLUE),
            ],
            ["NDCG@5", "NDCG@10", "NDCG@20"],
            as_pct=False,
            ymax=0.5,
            grid=(0.1, 0.2, 0.3, 0.4),
            h=250,
        )
    }</div>
  <div class="panel" style="width:430px"><div class="ph">Recall@5 của plain theo loại mục tiêu</div>
    {
        bar_chart(
            ["Đặt lại|cụm cũ", "Đặt|cụm mới", "Popularity|(cụm mới)"],
            [TT["old_target_in_top5_of_old_rows"], TT["new_target_in_top5_of_new_rows"], TT["global_popularity_top5_on_new_rows"]],
            [BLUE, AMBER, GRAY],
            1.0,
            w=400,
            h=250,
            as_pct=True,
        )
    }</div>
</div>
<div class="kpis two">
  <div class="kpi"><b>{pct(MX["test"]["new"], 0)}</b><span>lần đặt tiếp theo là <b>cụm mới</b>, lịch sử không đoán được</span></div>
  <div class="kpi"><b>{pct(TT["top1_is_last_booking"], 0)}</b><span>dòng có <b>top 1 = cụm đặt gần nhất</b>: plain chủ yếu gợi lại cụm cũ</span></div>
</div>""",
    sub=f"Gấp {dec(RATIO, 1)} lần popularity, nhưng chưa thấy điểm đến của lượt tìm nên yếu ở cụm mới. Thêm epoch gần như không giúp ({pp(PE[-1]['test']['ndcg@5'] - PE[0]['test']['ndcg@5'])} sau 3 epoch)",
)

# ---- 2b. sửa phía đầu ra không giúp (hướng A)
slide(
    "2 · Plain: điểm yếu",
    "Ép tỉ lệ cụm cũ / mới trong top 5 không giúp",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Hướng A: ép tỉ lệ cụm cũ / mới trong top 5, % dòng test trúng</div>
    {
        grouped_bars(
            [
                ("Cụm cũ trúng", [mp["old_target_hit_of_all_rows"], mm["old_target_hit_of_all_rows"], cc["old_target_hit_of_all_rows"]], BLUE),
                ("Cụm mới trúng", [mp["new_target_hit_of_all_rows"], mm["new_target_hit_of_all_rows"], cc["new_target_hit_of_all_rows"]], AMBER),
            ],
            [f"Plain|NDCG@5 {dec(mp['ndcg@5'], 3)}", f"2 cũ + 3 mới|NDCG@5 {dec(mm['ndcg@5'], 3)}", f"Giới hạn cũ theo L|NDCG@5 {dec(cc['ndcg@5'], 3)}"],
            ymax=0.35,
            grid=(0.1, 0.2, 0.3),
            w=640,
            h=270,
            bottom=46,
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Vì sao thử</div><p>Plain gợi lại cụm cũ, còn {pct(MX["test"]["new"], 0)} lần đặt tiếp theo là cụm mới. Thử chia lại 5 chỗ trong top 5 cho cụm mới.</p></div>
    <div class="th-card"><div class="lbl">Kết quả</div><p>Mỗi cụm mới trúng thêm làm mất khoảng {dec(exch_fixed, 0)} cụm cũ; NDCG@5 giảm ở cả hai cách.</p></div>
  </div>
</div>
<div class="note">Đổi cách chọn top 5 không thay được thông tin còn thiếu: để gợi đúng cụm mới, mô hình phải biết người dùng đang tìm ở đâu. Phần sau thử đúng điều này.</div>""",
    sub="Hai cách sửa ở đầu ra: cố định 2 cụm cũ + 3 cụm mới, hoặc giới hạn số cụm cũ theo độ dài lịch sử",
)

# ---- 3. điểm đến là tín hiệu thiếu
LADDER_L = ["Popularity|(không học)", "ItemKNN|(chỉ lịch sử)", "SMLP4Rec plain|(chỉ lịch sử)", "Hồi quy logistic|(điểm đến + ngữ cảnh)"]
slide(
    "3 · Điểm đến là tín hiệu thiếu",
    "Điểm đến và ngữ cảnh của lượt tìm là tín hiệu plain đang thiếu",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@5, tập test, mọi sự kiện ({num(BL_ALL["here"])} dòng)</div>
    {
        bar_chart(
            LADDER_L,
            [POP_ALL, KNN_ALL, PLAIN_ALL, LR_ALL],
            [GRAY, BLUE, BLUE, AMBER],
            0.5,
            w=700,
            h=270,
            legend=[("Không dùng thông tin", GRAY), ("Chỉ lịch sử", BLUE), ("Chỉ điểm đến + ngữ cảnh", AMBER)],
        )
    }</div>
  <div class="panel" style="width:400px"><div class="ph">Đối chứng xáo trộn, NDCG@5, user có lịch sử</div>
    {
        grouped_bars(
            [
                ("Thật", [CTRL["LR real"]["ndcg@5"], CTRL["ItemKNN real"]["ndcg@5"]], TEAL),
                ("Xáo trộn", [CTRL["LR shuffled query features"]["ndcg@5"], CTRL["ItemKNN shuffled histories"]["ndcg@5"]], GRAY),
            ],
            ["Hồi quy|logistic", "ItemKNN"],
            as_pct=False,
            ymax=0.5,
            grid=(0.1, 0.2, 0.3, 0.4),
            w=380,
            h=250,
            bottom=46,
        )
    }
    <p class="sm">Popularity cùng tập: {dec(CTRL["global popularity"]["ndcg@5"], 3)}. Xáo trộn đưa cả hai về mức popularity.</p></div>
</div>
<div class="note">Chỉ đọc điểm đến + ngữ cảnh (hồi quy logistic) hơn chỉ đọc lịch sử (SMLP4Rec plain) <b>+{dec(LR_ALL - PLAIN_ALL, 3)}</b> NDCG@5; xáo trộn đặc trưng đưa cả hai về mức popularity. Kiểm tra độc lập dựng lại phép chia, lịch sử, cosine và đặc trưng bằng code riêng: {VER_MISMATCH} sai khác về hạng, không thấy dấu hiệu rò rỉ. Chi tiết baseline ở phụ lục. Hai cách đưa điểm đến vào SMLP4Rec ở phần sau.</div>""",
    sub="Hai baseline có sẵn trong thư viện, mỗi cái bị chặn đúng một nguồn tín hiệu",
)

# ---- 4b. cách 1
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Cách 1: cộng prior điểm đến và sameDest sau SMLP4Rec",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@5 theo w: điểm = log p<sub>SMLP4Rec</sub> + w · log p<sub>prior</sub>(cụm | điểm đến); w = 0 là plain</div>
    {line_chart([w.replace(".", ",") for w in wgrid], [("Valid", [LF["sweep"]["valid"][w]["all"]["ndcg@5"] for w in wgrid], BLUE), ("Test", [LF["sweep"]["test"][w]["all"]["ndcg@5"] for w in wgrid], TEAL)], 0.20, 0.50, as_pct=False, w=620, h=250)}</div>
  <div class="panel" style="width:470px"><div class="ph">NDCG@5, test, user có lịch sử</div>
    {
        bar_chart(
            ["Plain", "Prior|một mình", "Plain|+ prior", "+ sameDest|(hybrid)"],
            [WARM[K_PLAIN]["ndcg@5"], WARM[K_PRIOR]["ndcg@5"], WARM[K_FUSION]["ndcg@5"], WARM[K_HYBRID]["ndcg@5"]],
            [GRAY, AMBER, BLUE, TEAL],
            0.6,
            w=440,
            h=270,
        )
    }</div>
</div>
<div class="note">Prior là đòn bẩy lớn nhất, chủ yếu ở cụm mới (NDCG@5 {dec(LV[K_PLAIN]["new_target_rows"]["ndcg@5"], 3)} → {dec(FUS_TEST["new_target_rows"]["ndcg@5"], 3)}). sameDest (điểm đến đã đặt trước đó, giảm dần theo tuổi) thêm {ci(HB, "ndcg@5")}, bootstrap 95%. Người dùng mới ({pct(COLD_SHARE, 0)} sự kiện test) chỉ dùng prior: NDCG@5 {dec(COLD_NDCG5, 3)}.</div>""",
    sub="Hybrid = SMLP4Rec + prior điểm đến + sameDest. Không huấn luyện lại; trọng số chọn trên valid, test chấm một lần",
)

# ---- 4b2. B3: tách trọng số theo thói quen (biến thể của Cách 1)
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Cách 1 mở rộng: tách trọng số theo thói quen không giúp",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Hướng B3: tách trọng số hybrid theo thói quen đặt lại, test</div>
    {
        grouped_bars(
            [
                ("Hybrid B2: 2 bộ trọng số", [BW["reference (notebook 04)"]["ndcg@5"], BW["reference (notebook 04)"]["ndcg@10"]], BLUE),
                (f"Thêm thói quen: {len(B_CELLS[B_BEST])} bộ", [BW["best behaviour scheme"]["ndcg@5"], BW["best behaviour scheme"]["ndcg@10"]], AMBER),
            ],
            ["NDCG@5", "NDCG@10"],
            as_pct=False,
            ymax=0.7,
            w=640,
            h=270,
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Thêm gì</div><p>Chia thêm theo mức thiên về cụm cũ của lịch sử: {len(B_CELLS[B_BEST])} bộ trọng số thay vì 2 (điểm đến mới / đã đặt).</p></div>
    <div class="th-card"><div class="lbl">NDCG@5 so với 2 bộ</div><p><b>{ci(BB, "ndcg@5")}</b><br>bootstrap 95%; quy tắc 1-SE vẫn chọn 2 bộ</p></div>
  </div>
</div>
<div class="note">Thói quen đặt lại đã nằm trong sameDest nên chia thêm không có gì để thêm. Giữ 2 bộ trọng số của Cách 1. Công thức và trọng số ở phụ lục.</div>""",
    sub="Thử mở rộng Cách 1: trọng số theo thói quen đặt lại",
)

# ---- 4c. cách 2: query token
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Cách 2: query token đưa lượt tìm vào trong mô hình",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Một vị trí cuối chuỗi cho lượt tìm hiện tại; hidden state tại đó chấm 100 cụm</div>
    {table(["Nhóm", "14 trường của lượt tìm, cộng thành một vector query"], Q_FIELDS, right_from=9)}
    <p class="sm">Lịch sử vẫn là tối đa 20 cụm đã đặt. {num(QT["n_parameters"])} tham số (chỉ lịch sử: {num(QT["n_parameters_history_only"])}). {QCFG["epochs"]} epoch, cùng phép chia và seed như plain. Không dùng mã khách sạn, khoảng cách, thành phố/vùng khách.</p></div>
  <div class="panel" style="width:520px"><div class="ph">NDCG@K, test, user có lịch sử ({num(QTT["n"])} dòng)</div>
    {grouped_bars(Q_SERIES, ["NDCG@5", "NDCG@10"], ymax=0.6, as_pct=False, w=480, h=250)}</div>
</div>
<div class="note">Chỉ thêm lượt tìm vào mô hình, NDCG@5 {dec(QTP["ndcg@5"], 3)} → {dec(QTT["ndcg@5"], 3)} ({ci(Q_BOOT, "ndcg@5")}, bootstrap 95%), ngang SMLP4Rec + prior (chênh {dec(abs(Q_SIM), 3)}); hybrid còn cao hơn {dec(Q_GAP, 3)}. Tăng ở mọi nhóm đã thấy / chưa thấy và mọi độ dài lịch sử.</div>""",
    sub="Một seed; kết hợp với sameDest và user mới (L = 0) ở hai slide sau",
)

# ---- 4d. chẩn đoán query token
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Query token mang thông tin giống prior điểm đến",
    f"""<div class="cmp2">
  <div class="panel"><div class="ph">Chẩn đoán cùng checkpoint, NDCG@5, test (không huấn luyện lại)</div>
    {
        bar_chart(
            ["Thật", "Xáo query|giữa các dòng", "Che lịch sử|giữ query", "Xáo và che"],
            [d["ndcg@5"] for _, d in Q_DIAG],
            [TEAL, RED, BLUE, GRAY],
            0.5,
            w=520,
            h=260,
        )
    }
    <p class="sm">Popularity: {dec(POP["ndcg@5"], 3)}. Xáo query đưa điểm về gần popularity: mô hình dựa vào query.</p></div>
  <div class="panel"><div class="ph">Các cách dùng điểm đến, NDCG@5, test</div>
    {
        bar_chart(
            ["Prior|một mình", "Query token|che lịch sử", "Query token|+ lịch sử", "Plain|+ prior"],
            [Q_PRIOR["ndcg@5"], QD["history masked"]["ndcg@5"], QTT["ndcg@5"], WARM[K_FUSION]["ndcg@5"]],
            [AMBER, BLUE, TEAL, GRAY],
            0.5,
            w=520,
            h=260,
        )
    }
    <p class="sm">Che lịch sử, chỉ còn query, vẫn đạt {dec(QD["history masked"]["ndcg@5"], 3)}, hơn plain đủ lịch sử ({dec(QTP["ndcg@5"], 3)}).</p></div>
</div>
<div class="note">Hợp lý: query token chứa điểm đến, nên mô hình tự học p(cụm | điểm đến) mà prior cung cấp bằng cách đếm. Slide sau kiểm tra trực tiếp: cộng thêm prior có tăng không.</div>""",
    sub="Giả thuyết: query token đã bao hàm prior",
)

slide(
    "4 · Hai cách đưa điểm đến vào",
    "Kiểm chứng: cộng prior vào query token tăng rất ít",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@5 theo độ phổ biến của điểm đến trong dữ liệu prior, test</div>
    {
        grouped_bars(
            [
                ("Query token một mình", [PF_S[f"{key} | query token alone"]["ndcg@5"] for _, key in Q_SL], BLUE),
                ("+ prior", [PF_S[f"{key} | query token + prior"]["ndcg@5"] for _, key in Q_SL], TEAL),
                ("Prior một mình", [PF_S[f"{key} | prior only"]["ndcg@5"] for _, key in Q_SL], GRAY),
            ],
            ["Phổ biến|(≥ 200 booking)", "20–199|booking", "Hiếm|(1–19 booking)", "Ngoài vocab|query token"],
            as_pct=False,
            ymax=0.6,
            w=700,
            h=270,
            bottom=46,
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Cộng prior (w = {dec(PF_W, 2)})</div><div class="val">{pp(PF_B["ndcg@5"]["diff"])}</div><p>NDCG@5 [{dec(PF_B["ndcg@5"]["ci95"][0] * 100, 2)}; {dec(PF_B["ndcg@5"]["ci95"][1] * 100, 2)}], bootstrap 95%; w chọn trên valid</p></div>
    <div class="th-card"><div class="lbl">sameDest, để so sánh</div><div class="val">{pp(HB["ndcg@5"]["diff"])}</div><p>NDCG@5 khi cộng vào SMLP4Rec + prior</p></div>
  </div>
</div>
<div class="note">Điểm đến phổ biến ({num(PF_S["destination 200+ | query token alone"]["n"])} dòng): prior không thêm ({dec(PF_S["destination 200+ | query token alone"]["ndcg@5"], 3)} → {dec(PF_S["destination 200+ | query token + prior"]["ndcg@5"], 3)}). Phần tăng nằm ở điểm đến hiếm hoặc ngoài vocab, nơi prior có backoff theo market. Khoảng cách {dec(Q_GAP, 3)} NDCG@5 còn lại với hybrid thuộc về sameDest, thứ query token chưa có.</div>""",
    sub="Giả thuyết đúng cho phần lớn dữ liệu",
)

# ---- 4e. query token + sameDest
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Query token + sameDest: bù phần query token còn thiếu",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@K, test, user có lịch sử ({num(SDW[N_QT]["n"])} dòng)</div>
    {
        grouped_bars(
            [
                ("Hybrid cũ (plain + prior + sameDest)", [SDW[N_OLD]["ndcg@5"], SDW[N_OLD]["ndcg@10"]], GRAY),
                ("Query token", [SDW[N_QT]["ndcg@5"], SDW[N_QT]["ndcg@10"]], BLUE),
                ("+ sameDest", [SDW[N_A]["ndcg@5"], SDW[N_A]["ndcg@10"]], AMBER),
                ("+ prior + sameDest", [SDW[N_B]["ndcg@5"], SDW[N_B]["ndcg@10"]], TEAL),
            ],
            ["NDCG@5", "NDCG@10"],
            as_pct=False,
            ymax=0.6,
            grid=(0.1, 0.2, 0.3, 0.4, 0.5),
            w=520,
            h=270,
        )
    }</div>
  <div class="panel grow"><div class="ph">NDCG@5 theo điểm đến của lượt tìm có trong lịch sử hay không</div>
    {
        grouped_bars(
            [
                ("Hybrid cũ", [sd_ndcg(SD_KNOWN, N_OLD), sd_ndcg(SD_NEW, N_OLD)], GRAY),
                ("Query token", [sd_ndcg(SD_KNOWN, N_QT), sd_ndcg(SD_NEW, N_QT)], BLUE),
                ("+ sameDest", [sd_ndcg(SD_KNOWN, N_A), sd_ndcg(SD_NEW, N_A)], AMBER),
                ("+ prior + sameDest", [sd_ndcg(SD_KNOWN, N_B), sd_ndcg(SD_NEW, N_B)], TEAL),
            ],
            [f"Đã đặt trước đó|({num(SD_N_KNOWN)} dòng)", f"Điểm đến mới|({num(SD_N_NEW)} dòng)"],
            as_pct=False,
            ymax=0.8,
            grid=(0.2, 0.4, 0.6),
            w=520,
            h=270,
            bottom=46,
        )
    }</div>
</div>
<div class="note">Cùng công thức, lưới, quy tắc 1-SE và 2 bộ trọng số (điểm đến mới / đã đặt) như Cách 1; chỉ thay SMLP4Rec plain bằng mô hình query token. Điểm đến đã đặt: sameDest đưa NDCG@5 {dec(sd_ndcg(SD_KNOWN, N_QT), 3)} → {dec(sd_ndcg(SD_KNOWN, N_A), 3)}, ngang hybrid cũ. Điểm đến mới: sameDest không tác dụng, cần prior ({dec(sd_ndcg(SD_NEW, N_QT), 3)} → {dec(sd_ndcg(SD_NEW, N_B), 3)}). So hybrid cũ, NDCG@5: query token + sameDest {ci(sd_boot(N_A)["warm"], "ndcg@5")}; query token + prior + sameDest {ci(sd_boot(N_B)["warm"], "ndcg@5")} (bootstrap 95%).</div>""",
    sub="Query token đã chứa điểm đến nhưng không có lịch sử điểm đến của người dùng; sameDest bù đúng phần đó",
)

# ---- 4f. L = 0
slide(
    "4 · Hai cách đưa điểm đến vào",
    "Người dùng mới (L = 0): prior nhỉnh hơn query token",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@5, test, lần đặt đầu của mỗi người ({num(CL["n_rows"]["test"])} dòng, lịch sử rỗng)</div>
    {
        bar_chart(
            ["Popularity", "Prior|một mình", "Query token|lịch sử rỗng", "Query token|+ prior"],
            [CLT[CL_POP]["ndcg@5"], CLT[CL_PRI]["ndcg@5"], CLT[CL_QT]["ndcg@5"], CLT[CL_QTP]["ndcg@5"]],
            [GRAY, AMBER, BLUE, TEAL],
            0.5,
            w=560,
            h=270,
        )
    }</div>
  <div class="panel" style="width:430px"><div class="ph">NDCG@5 theo điểm đến có trong vocab query token hay không</div>
    {
        grouped_bars(
            [
                ("Prior", [CLS[CL_IN][CL_PRI]["ndcg@5"], CLS[CL_OUT][CL_PRI]["ndcg@5"]], AMBER),
                ("Query token", [CLS[CL_IN][CL_QT]["ndcg@5"], CLS[CL_OUT][CL_QT]["ndcg@5"]], BLUE),
                ("+ prior", [CLS[CL_IN][CL_QTP]["ndcg@5"], CLS[CL_OUT][CL_QTP]["ndcg@5"]], TEAL),
            ],
            [f"Trong vocab|({num(CLS[CL_IN][CL_PRI]['n'])})", f"Ngoài vocab|({num(CLS[CL_OUT][CL_PRI]['n'])})"],
            as_pct=False,
            ymax=0.5,
            grid=(0.1, 0.2, 0.3, 0.4),
            w=400,
            h=225,
            bottom=46,
        )
    }</div>
</div>
<div class="note">Không có lịch sử, sameDest tự tắt. Query token một mình thấp hơn prior {ci(CL_BOOT["query token alone minus prior only"], "ndcg@5")} NDCG@5 (bootstrap 95%); cộng prior vẫn chưa vượt ({ci(CL_BOOT["query token + prior minus prior only"], "ndcg@5")}). Chênh lệch chủ yếu ở điểm đến ngoài vocab ({pct(CL["test_share_destination_oov"], 1)} dòng), nơi prior có backoff theo market. Giữ quy tắc cũ: người dùng mới chỉ dùng prior.</div>""",
    sub="Một seed; query token chạy trực tiếp với lịch sử toàn padding, cùng checkpoint, không huấn luyện lại",
)

# ---- 5a. gộp lại (sau khi cả hai cách đã được trình bày)
slide(
    "5 · Gộp lại và độ bền",
    "Gộp lại: điểm đến là bước nhảy lớn, sameDest là phần thêm",
    f"""<div class="kpis three">
  <div class="kpi"><b class="s">{dec(SDA[N_PL]["ndcg@5"], 3)} → {dec(SDA[N_QT]["ndcg@5"], 3)}</b><span>NDCG@5, mọi sự kiện: thêm lượt tìm vào SMLP4Rec (query token)</span></div>
  <div class="kpi"><b class="s">{dec(SDA[N_QT]["ndcg@5"], 3)} → {dec(SDA[N_A]["ndcg@5"], 3)}</b><span>thêm sameDest vào query token; cộng cả prior: {dec(SDA[N_B]["ndcg@5"], 3)}</span></div>
  <div class="kpi"><b class="s">{dec(SDA[N_OLD]["ndcg@5"], 3)} vs {dec(SDA[N_B]["ndcg@5"], 3)}</b><span>hybrid cũ (plain + prior + sameDest) vs query token + prior + sameDest</span></div>
</div>
<div class="panel"><div class="ph">NDCG@5, tập test, mọi sự kiện ({num(SDA[N_PL]["n"])} dòng: người dùng có lịch sử + lần đặt đầu)</div>
  {
        bar_chart(
            ["Plain", "Plain|+ prior", "Hybrid cũ|(+ sameDest)", "Query token", "Query token|+ prior", "Query token|+ sameDest", "Query token|+ prior|+ sameDest"],
            [SDA[n]["ndcg@5"] for n in (N_PL, N_PLP, N_OLD, N_QT, N_QTP, N_A, N_B)],
            [GRAY, GRAY, GRAY, BLUE, BLUE, BLUE, TEAL],
            0.5,
            w=900,
            h=165,
            legend=[("Nền SMLP4Rec plain", GRAY), ("Nền query token", BLUE), ("Cao nhất", TEAL)],
        )
    }</div>
<div class="note">Một seed, 3 epoch: plain + prior ({dec(SDA[N_PLP]["ndcg@5"], 3)}) và query token ({dec(SDA[N_QT]["ndcg@5"], 3)}) gần nhau; sameDest cộng thêm ở cả hai. Query token + prior + sameDest cao nhất, chỉ hơn hybrid cũ {pp(sd_boot(N_B)["all_events"]["ndcg@5"]["diff"])}. Người dùng mới (L = 0) dùng prior.</div>""",
    sub="Mọi biến thể trên cùng tập test; chi tiết từng cột ở phụ lục",
)

# ---- 5. độ bền
slide(
    "5 · Gộp lại và độ bền",
    "Hybrid hơn plain ở người dùng đã thấy và chưa thấy",
    f"""<div class="kpis three">
  <div class="kpi"><b class="s">{pct(PR["test"]["unseen_users_pct"] / 100, 1)}</b><span>người dùng test chưa có trong train ({num(PR["test"]["unseen_users"])} / {num(PR["test"]["users"])})</span></div>
  <div class="kpi"><b class="s">{num(PR["test"]["unseen_users_with_history_(L>=1)"])} + {num(PR["test"]["unseen_users_first_booking_only_(L=0)"])}</b><span>chưa thấy: đã có lịch sử (L ≥ 1) + chỉ có lần đặt đầu (L = 0)</span></div>
  <div class="kpi"><b class="s">{dec(PR["test"]["mean_history_len_seen"], 1)} vs {dec(PR["test"]["mean_history_len_unseen_L>=1"], 1)}</b><span>độ dài lịch sử trung bình: đã thấy vs chưa thấy (L ≥ 1)</span></div>
</div>
<div class="cmp2">
  <div class="panel"><div class="ph">NDCG@5, tập test</div>
    {grouped_bars([("SMLP4Rec plain", [SR[s][PL]["ndcg@5"] for s, _ in SUBS], BLUE), ("Hybrid", [SR[s][HYK]["ndcg@5"] for s, _ in SUBS], TEAL)], [n for _, n in SUBS], as_pct=False, ymax=0.6, w=520, h=185)}</div>
  <div class="panel"><div class="ph">Recall@5, tập test</div>
    {grouped_bars([("SMLP4Rec plain", [SR[s][PL]["recall@5"] for s, _ in SUBS], BLUE), ("Hybrid", [SR[s][HYK]["recall@5"] for s, _ in SUBS], TEAL)], [n for _, n in SUBS], w=520, h=185)}</div>
</div>
<div class="note">Plain gần như không phụ thuộc “đã thấy” (chưa thấy, L ≥ 1: {dec(SR["unseen, L>=1"][PL]["ndcg@5"], 3)} so với {dec(SR["seen users"][PL]["ndcg@5"], 3)}) vì chỉ đọc chuỗi booking: chênh lệch chính là độ dài lịch sử. * L = 0: plain dùng popularity, hybrid dùng prior điểm đến. Query token (một mình, test): đã thấy {dec(QT["slices_test"]["seen users | query token"]["ndcg@5"], 3)}, chưa thấy L ≥ 1 {dec(QT["slices_test"]["unseen users (L>=1) | query token"]["ndcg@5"], 3)}.</div>""",
    sub="Cùng checkpoint và trọng số, không huấn luyện lại; chấm riêng từng tập con",
)

# ---- 6. kết luận
slide(
    "6 · Kết luận",
    "Kết luận, giới hạn và bước tiếp",
    f"""<div class="grid3">
  <div class="card good"><h3>Rút ra cho RQ1</h3><p>Trên Expedia, tín hiệu chính là <b>điểm đến của lượt tìm</b>: hồi quy logistic chỉ đọc điểm đến + ngữ cảnh đạt NDCG@5 {dec(LR_ALL, 2)}, hơn SMLP4Rec plain ({dec(PLAIN_ALL, 2)}) và ItemKNN ({dec(KNN_ALL, 2)}). Đưa điểm đến vào SMLP4Rec bằng cộng điểm sau (plain + prior {dec(SDA[N_PLP]["ndcg@5"], 2)}) hay query token ({dec(SDA[N_QT]["ndcg@5"], 2)}) cho kết quả gần nhau; thêm sameDest lên {dec(SDA[N_OLD]["ndcg@5"], 2)} (plain) và {dec(SDA[N_B]["ndcg@5"], 2)} (query token + prior), mọi sự kiện test.</p></div>
  <div class="card"><h3>Giới hạn</h3><p>Một seed, 3 epoch, prior tĩnh. Chưa có MF, item2vec, GRU4Rec / SASRec, AdaGIN, LightGBM, LLM rerank. Chưa cắt theo mùa, mức hoạt động của user, chưa có metric beyond-accuracy (RQ4). Lịch sử chưa mang ngữ cảnh từng booking; người dùng mới vẫn dùng prior vì query token không vượt prior.</p></div>
  <div class="card"><h3>Bước tiếp</h3><p>Ngữ cảnh theo vị trí cho lịch sử, chạy lại trên chia theo sự kiện, ≥ 3 seed. Rồi các họ mô hình còn lại trên cùng phép chia để trả lời RQ1 đầy đủ, kèm RQ4.</p></div>
</div>
<div class="note">Quyết định: SMLP4Rec + prior điểm đến + sameDest, 2 bộ trọng số (điểm đến mới / đã đặt). Query token + prior + sameDest cao hơn một chút ({pp(sd_boot(N_B)["all_events"]["ndcg@5"]["diff"])} NDCG@5) và là ứng viên thay plain; người dùng mới dùng prior.</div>""",
)

# ---- phụ lục (nội dung v2, giữ nguyên số liệu)
slide(
    "Phụ lục",
    "Hướng B2: thêm sameDest, hybrid ba tín hiệu",
    f"""<div class="eq">điểm = log q<sub>SMLP4Rec</sub> + w<sub>p</sub> · log q<sub>prior</sub> + w<sub>s</sub> · log q<sub>sameDest</sub><br><span class="sm">trong đó mỗi q là xác suất p đã trộn với phân bố đều: log q = log((1 − α)·p + α/100)</span></div>
<p class="muted">sameDest(cụm) = tổng 0,7<sup>tuổi</sup> các lần người dùng đặt cụm đó <b>tại đúng điểm đến đang tìm</b> (tuổi 0 = lần gần nhất), chia cho tổng để thành phân bố. Ví dụ: cụm 12 đặt tại Cancún ở tuổi 0 và 3 → {
        dec(EX12, 3)
    }; đặt ở điểm đến khác → 0. Ba thành phần cùng một thang nên trọng số đọc được như mức tin cậy; chưa từng đặt tại điểm đến này → sameDest không tác động.</p>
{
        table(
            ["Cách (test)", "NDCG@5", "Recall@5", "NDCG@10", "Recall@10", "NDCG@5 gồm người dùng mới", "R@5 gồm người dùng mới"],
            [
                [
                    name,
                    dec(WARM[k]["ndcg@5"]),
                    pct(WARM[k]["recall@5"]),
                    dec(WARM[k]["ndcg@10"]),
                    pct(WARM[k]["recall@10"]),
                    dec(ALLEV[k]["ndcg@5"]),
                    pct(ALLEV[k]["recall@5"]),
                ]
                for name, k in (
                    ("1. Plain SMLP4Rec", K_PLAIN),
                    ("2. Prior điểm đến một mình", K_PRIOR),
                    ("3. SMLP4Rec + prior", K_FUSION),
                    ("<b>4. Hybrid: + sameDest</b>", K_HYBRID),
                )
            ],
            hl=(3,),
        )
    }
<div class="note">Rút ra: hybrid cao nhất ở mọi cột; so với SMLP4Rec + prior: NDCG@5 {ci(HB, "ndcg@5")}, Recall@5 {ci(HB, "recall@5")} (bootstrap 95%). Tăng ở điểm đến đã đặt (NDCG@5 {dec(KD_REF[1])} → {dec(KD_HYB[1])}; Recall@5 {pct(KD_REF[0], 1)} → {pct(KD_HYB[0], 1)}), điểm đến mới gần như giữ nguyên (NDCG@5 {dec(ND_REF[1])} → {dec(ND_HYB[1])}; Recall@5 {pct(ND_REF[0], 1)} → {pct(ND_HYB[0], 1)}). Trọng số và lưới tìm kiếm ở slide cuối.</div>""",
    sub="Giả thuyết: người dùng hay đặt lại cụm đã ở tại điểm đến này. Cột cuối: người dùng mới dùng prior",
)
slide(
    "Phụ lục",
    "Hướng B3: tách trọng số theo thói quen đặt lại",
    f"""<div class="kpis three">
  <div class="kpi"><b class="s">old_share</b><span>tỉ lệ booking trong lịch sử là <b>cụm đã đặt trước đó</b> (chỉ dùng lịch sử, không dùng mục tiêu)</span></div>
  <div class="kpi"><b class="s">{dec(B_CUT, 2)}</b><span>ngưỡng chọn trên train: <b>thiên cũ</b> nếu old_share ≥ ngưỡng, còn lại <b>thiên mới</b>; L = 1 là nhóm riêng</span></div>
  <div class="kpi"><b class="s">≥ 10%</b><span>mỗi bộ trọng số phải phủ ít nhất 10% dòng, không tạo nhóm nhỏ</span></div>
</div>
{
        table(
            ["Cách chia trọng số (test)", "NDCG@5", "Recall@5", "NDCG@10", "Recall@10", "NDCG@5 gồm người dùng mới", "R@5 gồm người dùng mới"],
            [
                [
                    name,
                    dec(BW[k]["ndcg@5"]),
                    pct(BW[k]["recall@5"]),
                    dec(BW[k]["ndcg@10"]),
                    pct(BW[k]["recall@10"]),
                    dec(BA[k]["ndcg@5"]),
                    pct(BA[k]["recall@5"]),
                ]
                for name, k in (
                    ("Hybrid B2: điểm đến mới / đã đặt (2 bộ)", "reference (notebook 04)"),
                    (f"Thêm thói quen: điểm đến × thiên cũ / còn lại ({len(B_CELLS[B_BEST])} bộ)", "best behaviour scheme"),
                )
            ],
            hl=(0,),
        )
    }
<div class="note">Rút ra: tách theo thói quen <b>không</b> cải thiện: NDCG@5 {ci(BB, "ndcg@5")}, Recall@5 {ci(BB, "recall@5")} (bootstrap 95%); quy tắc 1-SE trên valid vẫn chọn 2 bộ của B2. Thói quen đặt lại đã nằm sẵn trong sameDest, nên không cần trọng số riêng.</div>""",
    sub="Giả thuyết: người hay đặt lại cụm cũ cần trọng số khác người hay đặt cụm mới. Công thức B2 giữ nguyên, chỉ đổi cách chia nhóm",
)
slide(
    "Phụ lục",
    "Hai baseline cơ bản trên cùng phép chia 8/1/1",
    f"""{
        table(
            ["Baseline", "Thư viện", "Thấy", "Không thấy", "Chọn trên valid"],
            [
                [
                    "ItemKNN (cosine, Sarwar 2001)",
                    "implicit · CosineRecommender",
                    f"{CFG['MAX_ITEM_LIST_LENGTH']} booking gần nhất của người dùng",
                    "điểm đến, mọi ngữ cảnh",
                    f"K = {KM['K_pick']} trong {{{', '.join(str(k) for k in sorted(int(k) for k in KM['K_grid_valid']))}}}",
                ],
                [
                    "Hồi quy logistic đa lớp",
                    "scikit-learn · saga",
                    "điểm đến + ngữ cảnh lượt tìm (one-hot)",
                    "lịch sử, mọi trường của khách sạn đã đặt",
                    f"C = {dec(LM['C_pick'], 0)} trong {{{'; '.join(dec(c, 1).rstrip('0').rstrip(',') if c % 1 else dec(c, 0) for c in sorted(float(k) for k in LM['C_grid_valid']))}}}; {LM['epochs']} epoch",
                ],
            ],
            right_from=9,
        )
    }
<div class="kpis three">
  <div class="kpi"><b class="s">{num(BSP["counts"]["train"] + BSP["counts"]["valid"] + BSP["counts"]["test"])}</b><span>mục tiêu “đặt tiếp theo”, chia theo thời gian 80/10/10 như bản plain</span></div>
  <div class="kpi"><b class="s">{num(BL_ALL["here"])}</b><span>sự kiện test, bản plain có {num(BL_ALL["reference"])} (chênh {BL_ALL["diff"]})</span></div>
  <div class="kpi"><b class="s">{num(BL["verification"]["eval_rows_in_training_data"])}</b><span>dòng valid/test nằm trong dữ liệu huấn luyện của hai baseline</span></div>
</div>
<div class="note">Mục đích: kiểm tra điểm cao của hybrid đến từ dữ liệu chứ không từ rò rỉ hay lỗi chia tập, bằng hai mô hình có sẵn trong thư viện, không dùng code của hybrid. Dữ liệu huấn luyện của cả hai: booking trước mốc cắt (cùng mốc cắt của prior). ItemKNN không có epoch, tính một lần theo công thức. K và C chọn trên valid theo Recall@5 + NDCG@5; test chấm một lần.</div>""",
    sub="Một mô hình chỉ đọc lịch sử, một mô hình chỉ đọc điểm đến và ngữ cảnh; mỗi mô hình bị chặn đúng một nguồn tín hiệu",
)
slide(
    "Phụ lục",
    "Hai baseline so với plain và hybrid, tập test",
    f"""{
        table(
            ["Cách", "NDCG@5 (mọi sự kiện)", "Recall@5 (mọi sự kiện)", "NDCG@5 đã thấy", "NDCG@5 chưa thấy, L ≥ 1", "NDCG@5 L = 0*"],
            bb_rows,
            hl=(4,),
        )
    }
<div class="note">Rút ra: (1) ItemKNN, chỉ đọc lịch sử, nằm sát SMLP4Rec plain (NDCG@5 {dec(bb_cell(1, "all events", "ndcg@5"), 3)} so với {dec(bb_cell(2, "all events", "ndcg@5"), 3)}), nên plain không bị thổi phồng. (2) Hồi quy logistic, chỉ có điểm đến và ngữ cảnh, đạt {dec(bb_cell(3, "all events", "ndcg@5"), 3)}; ở lần đặt đầu (L = 0) chỉ kém hybrid {dec(bb_cell(4, L0, "ndcg@5") - bb_cell(3, L0, "ndcg@5"), 3)} điểm NDCG@5. (3) Hybrid cao hơn hồi quy logistic ở mọi cột. * L = 0: ItemKNN và plain dùng popularity toàn cục (cùng số). Số dòng mỗi tập con lệch bản plain tối đa 4 dòng.</div>""",
    sub="Cùng dòng test, cùng Recall@K và NDCG@K; hàng plain và hybrid lấy từ notebook 06",
)
slide(
    "Phụ lục",
    "Lưới trọng số của hai bản hybrid (B2 và B3)",
    f"""<div class="cmp2">
  <div class="panel"><div class="ph">B2: 2 bộ trọng số (điểm đến mới / đã đặt)</div>
    <p class="sm">α ∈ {{{gl(GRID["alpha"])}}} · w<sub>p</sub> ∈ {{{gl(GRID["w_p"])}}} · w<sub>s</sub> ∈ {{{gl(GRID["w_s"])}}} · w<sub>SMLP4Rec</sub> = {dec(GRID["w_SMLP4Rec"], 0)} → {len(GRID["alpha"]) * N_PAIRS} cấu hình, chấm trên valid.</p>
    {table(["Bộ trọng số (α = " + dec(H_ALPHA, 1) + ")", "w<sub>p</sub>", "w<sub>s</sub>"], b2_rows, hl=(1,))}
    <p class="sm">α = {dec(H_ALPHA, 1)} được chọn cùng lúc: {HY["selection"]["n_within_one_se"]} cấu hình nằm trong một sai số chuẩn của cấu hình tốt nhất.</p>
  </div>
  <div class="panel"><div class="ph">B3: {len(B_CELLS[B_BEST])} bộ trọng số (điểm đến × thói quen)</div>
    <p class="sm">Cùng lưới w<sub>p</sub> × w<sub>s</sub> ({N_PAIRS} cặp) cho từng bộ, α cố định {dec(H_ALPHA, 1)}, ngưỡng old_share {dec(B_CUT, 2)}.</p>
    {table(["Bộ trọng số (α = " + dec(H_ALPHA, 1) + ")", "w<sub>p</sub>", "w<sub>s</sub>"], b3_rows, hl=())}
    <p class="sm">Trọng số mỗi bộ chọn bằng cùng quy tắc; quy tắc 1-SE ở mức lược đồ vẫn giữ 2 bộ của B2.</p>
  </div>
</div>
<div class="note">Quy tắc chọn (chỉ dùng valid): điểm mỗi dòng = Recall@5 + NDCG@5; trong các cấu hình nằm trong một sai số chuẩn (gom theo người dùng) của cấu hình tốt nhất, lấy cấu hình đơn giản nhất = w lớn nhất nhỏ nhất, rồi tổng w nhỏ nhất, rồi α lớn nhất. w<sub>s</sub> “hằng số”: điểm đến chưa từng đặt thì sameDest không tác động xếp hạng.</div>""",
    sub="Không gian tìm kiếm, quy tắc chọn và trọng số cuối cùng",
)



slide(
    "Phụ lục · Query token + sameDest",
    "Các biến thể query token, test, so với hybrid cũ",
    f"""{
        table(
            ["Cách", "NDCG@5", "Recall@5", "NDCG@10", "Recall@10", "NDCG@5 mọi sự kiện", "R@5 mọi sự kiện"],
            [
                [n, dec(SDW[n]["ndcg@5"]), pct(SDW[n]["recall@5"]), dec(SDW[n]["ndcg@10"]), pct(SDW[n]["recall@10"]), dec(SDA[n]["ndcg@5"]), pct(SDA[n]["recall@5"])]
                for n in (N_PL, N_PLP, N_OLD, N_QT, N_QTP, N_A, N_B)
            ],
            hl=(6,),
        )
    }
<div class="note">Trọng số chọn trên valid (α = {dec(SDT["B"]["alpha"], 1)}). Query token + sameDest: w<sub>p</sub> = 0, w<sub>s</sub> = {dec(SDT["A"]["weights"]["1"][1], 2)} khi điểm đến đã đặt. Query token + prior + sameDest: điểm đến mới w<sub>p</sub> = {dec(SDT["B"]["weights"]["0"][0], 2)}; đã đặt w<sub>p</sub> = {dec(SDT["B"]["weights"]["1"][0], 2)}, w<sub>s</sub> = {dec(SDT["B"]["weights"]["1"][1], 2)} ({SDT["B"]["n_within_one_se"]} / {SDT["B"]["n_configs"]} cấu hình trong một sai số chuẩn). “Mọi sự kiện” thêm {num(CL["n_rows"]["test"])} dòng L = 0.</div>""",
    sub="Cùng công thức, lưới và quy tắc chọn như notebook 04; kết quả từ notebook 01c, mục 5",
)

def render() -> None:
    parts, total = [], len(slides)
    for n, (meta, title, sub, body) in enumerate(slides, 1):
        if meta == "cover":
            parts.append(
                f'<section class="slide">{body}<div class="ft"><span>Topic C1 · Next-Best-Product Recommendation</span><span>{n} / {total}</span></div></section>'
            )
            continue
        subh = f'<div class="sub">{sub}</div>' if sub else ""
        parts.append(
            f'<section class="slide"><div class="hd"><div><div class="meta">{meta}</div><h2>{title}</h2>{subh}</div>'
            f'<div class="pg">Tuần 3 · Triển khai</div></div><div class="body">{body}</div>'
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ results/week3_implementation và results/week4_rebuild</span><span>{n} / {total}</span></div></section>'
        )
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tuần 3: SMLP4Rec trên Expedia, điểm đến là tín hiệu quyết định</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="deck">{"".join(parts)}</div><script>{JS}</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    render()
