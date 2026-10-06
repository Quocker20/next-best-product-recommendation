"""Build the week-3 report deck (Vietnamese, HTML) in the visual style of the week-2 slides.

Sections: (1) design recap from week 2 (SMLP4Rec architecture + hybrid flow slides, taken as-is),
(2) from the raw SMLP4Rec code to the Expedia adaptation, (3) plain run results, (4) the
experiments and their findings, (5) conclusion. CSS, JS and the architecture diagram are read
from scripts/week2_methodology_slides.py so the style stays identical.

Experiment chain: plain -> top-5 rules (dropped) -> + destination prior -> + sameDest (hybrid)
-> behaviour split of the hybrid weights.

Every number on the slides is read from persisted outputs in results/week3_implementation/:
smlprec_expedia_run.json (plain run), smlprec_expedia_mixed_run.json (fixed quota),
smlprec_expedia_dynamic_cap_run.json (dynamic cap), smlprec_expedia_late_fusion.json (prior
fusion + cold users, notebook 03), smlprec_expedia_hybrid_samedest.json (hybrid with sameDest,
notebook 04), smlprec_expedia_hybrid_behaviour_split.json (behaviour split, notebook 05),
smlprec_expedia_seen_unseen_users.json (seen vs unseen users, notebook 06); and from
results/week4_rebuild/: basic_baselines.json (ItemKNN + logistic regression, notebook 07) and
basic_baselines_verify.json (independent check, scripts/verify_basic_baselines.py).

Usage: python scripts/week3_report_slides.py
Output: reports/summary/week3_implementation/week3_report_slides_v2.html (PDF: print with headless Chrome)
"""

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W3 = ROOT / "results" / "week3_implementation"
OUT = ROOT / "reports" / "summary" / "week3_implementation" / "week3_report_slides_v2.html"
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


def grouped_bars(series, groups, ymax=0.7, w=640, h=300, as_pct=True) -> str:
    """series: [(label, values, color)], groups: category labels (inline SVG)."""
    top, bottom, left = 24, 30, 6
    ph, gw = h - top - bottom, (w - left) / len(groups)
    bw = gw * 0.22
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in (0.2, 0.4, 0.6):
        y = top + ph * (1 - g / ymax)
        out.append(f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(
            f'<text x="{left}" y="{y - 4:.1f}" font-size="11" fill="#94a3b8">{f"{int(g * 100)}%" if as_pct else dec(g, 1)}</text>'
        )
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + gw * 0.14
        for si, (_, vals, col) in enumerate(series):
            v = vals[gi]
            bh = ph * v / ymax
            x = x0 + si * (bw + 6)
            y = top + ph - bh
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>'
            )
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" class="cv">{pct(v, 1) if as_pct else dec(v, 3)}</text>')
        out.append(
            f'<text x="{x0 + (len(series) * bw + 6 * (len(series) - 1)) / 2:.1f}" y="{h - 8}" class="cl">{g}</text>'
        )
    out.append("</svg>")
    legend = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


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
    for label, vals, col in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}"/>')
        bi = max(range(len(vals)), key=lambda k: vals[k])
        out.append(
            f'<text x="{X(bi):.1f}" y="{Y(vals[bi]) - 9:.1f}" class="cv">{pct(vals[bi], 1) if as_pct else dec(vals[bi], 3)}</text>'
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


slide_cover = (
    "cover",
    "",
    "",
    """<div class="cover">
  <div class="kicker">Tuần 3 · Triển khai &amp; thử nghiệm</div>
  <h1>Triển khai code gốc SMLP4Rec trên Expedia<br><span>và các thử nghiệm bổ sung</span></h1>
  <p class="lead">Next-Best-Product Recommendation · Vinpearl × GSM · Bối cảnh (a): du lịch</p>
  <div class="cover-tags"><span>Lý thuyết</span><span>Chạy plain</span><span>Rút ra</span><span>Thử nghiệm</span></div>
</div>""",
)
slides.append(slide_cover)

# ---- 1. lý thuyết
slide(
    "1 · Lý thuyết",
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
    sub="Chọn làm phương pháp chính ở tuần 2 (Gao et al., ACM TOIS 2024). Việc cần làm: chạy thử trên Expedia",
)

# ---- 2. triển khai
slide(
    "2 · Triển khai",
    "Từ code gốc đến bản chạy với Expedia",
    table(
        ["Hạng mục", "Code gốc", "Bản cho Expedia"],
        [
            [
                "Dữ liệu",
                "MovieLens, item có feature",
                "Booking Expedia: user, cụm (100), thời gian; cụm không có feature",
            ],
            ["Độ dài chuỗi", "Cố định 50", "Lấy từ cấu hình (= 20)"],
            ["Feature item", "Bắt buộc có", "Cho phép rỗng, bỏ lớp feature"],
            [
                "Chia và đánh giá",
                "Leave-one-out, 100 mẫu theo popularity",
                "Chia theo thời gian 80/10/10, xếp hạng đủ 100 cụm",
            ],
            ["Cấu hình", "Cấu hình paper, quá chậm trên CPU", "2 lớp, hidden 64, dropout 0"],
            [
                "Môi trường",
                "RecBole 1.0.1 (numpy 1.23, pandas 1.5)",
                "Venv Python 3.11 riêng + 2 bản vá nhỏ",
            ],
        ],
        right_from=9,
    )
    + '<div class="note">Giữ nguyên từ bản gốc: một khối dùng chung cho mọi lớp, không residual. Chỉ sửa 2 chỗ trong mã mô hình (đánh dấu <code>ADAPTED</code>).</div>',
    sub="Chỉnh sửa tối thiểu để chạy được, chưa tối ưu",
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
    "2 · Triển khai",
    "Pipeline và chia dữ liệu",
    f"""<div class="flow six">{flow_html}</div>"""
    + table(
        ["Tập", "Số mục tiêu", "Từ ngày", "Đến ngày"],
        [
            [
                n,
                num(SP[k]["targets"]),
                day(SP[k]["first_target_unix"]),
                day(SP[k]["last_target_unix"]),
            ]
            for k, n in (("train", "Train"), ("valid", "Valid"), ("test", "Test"))
        ],
    )
    + f"""<div class="note">{num(DS["users"])} người dùng · {DS["items"]} cụm · {num(DS["sequence_targets"])} mục tiêu “đặt tiếp theo” (mọi booking trừ lần đầu của mỗi người). Chia theo thời gian, xếp hạng đầy đủ.</div>""",
)

# ---- 3. chạy plain
slide(
    "3 · Chạy plain",
    "Chạy plain: chỉ dùng lịch sử booking",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">NDCG@K, tập test, xếp hạng đầy đủ 100 cụm</div>
    {
        grouped_bars(
            [
                ("Popularity toàn cục", [POP[f"ndcg@{k}"] for k in (5, 10, 20)], GRAY),
                (
                    f"SMLP4Rec plain (epoch {RUN['best_epoch_by_valid']})",
                    [T[f"ndcg@{k}"] for k in (5, 10, 20)],
                    BLUE,
                ),
            ],
            ["NDCG@5", "NDCG@10", "NDCG@20"],
            as_pct=False,
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">SMLP4Rec plain, test</div><div class="val">NDCG@5 {
        dec(T["ndcg@5"])
    }</div><p>Recall@5 {pct(T["recall@5"])} · NDCG@10 {dec(T["ndcg@10"])}</p></div>
    <div class="th-card"><div class="lbl">NDCG@5 qua 3 epoch</div><div class="val">{
        dec(PE[0]["test"]["ndcg@5"])
    } → {dec(PE[-1]["test"]["ndcg@5"])}</div><p>thêm epoch gần như không giúp ({
        pp(PE[-1]["test"]["ndcg@5"] - PE[0]["test"]["ndcg@5"])
    })</p></div>
  </div>
</div>
<div class="note">Rút ra: có tín hiệu (gấp {
        dec(RATIO, 1)
    } lần popularity) nhưng điểm thấp. Vì sao?</div>""",
    sub="Câu hỏi: chỉ lịch sử có đủ để đoán cụm tiếp theo không?",
)

slide(
    "3 · Chạy plain",
    "Vì sao plain thấp, và thử theo hướng nào",
    f"""<div class="kpis three">
  <div class="kpi"><b>{pct(MX["test"]["new"], 0)}</b><span>lần đặt tiếp theo là <b>cụm mới</b>, lịch sử không đoán được</span></div>
  <div class="kpi"><b>{pct(TT["top1_is_last_booking"], 0)}</b><span>dòng có <b>top 1 = cụm đặt gần nhất</b>: plain chủ yếu gợi lại</span></div>
  <div class="kpi"><b>{pct(TT["old_target_in_top5_of_old_rows"], 1)} vs {pct(TT["new_target_in_top5_of_new_rows"], 1)}</b><span>trúng khi đặt lại cụm cũ vs đặt cụm mới</span></div>
</div>
<div class="grid2">
  <div class="card good"><h3>Hướng A: chỉnh cách chọn top 5</h3><p>Ép một phần top 5 dành cho cụm mới.</p></div>
  <div class="card good"><h3>Hướng B: thêm thông tin điểm đến</h3><p>Prior điểm đến, rồi tín hiệu “đặt lại tại cùng điểm đến” (sameDest).</p></div>
</div>
<div class="note">Plain chưa thấy điểm đến của lần tìm kiếm, nên yếu ở nhóm cụm mới ({pct(TT["new_target_in_top5_of_new_rows"], 1)}, popularity top 5: {pct(TT["global_popularity_top5_on_new_rows"], 1)}).</div>""",
    sub="Rút ra từ top 5 của plain → hai hướng thử nghiệm",
)

# ---- 4. thử nghiệm
slide(
    "4 · Thử nghiệm",
    "Hướng A: ép tỉ lệ cụm cũ / cụm mới trong top 5",
    table(
        ["Cách", "NDCG@5", "Recall@5", "Cụm cũ trúng", "Cụm mới trúng"],
        [
            ["Plain: 5 điểm cao nhất", dec(mp["ndcg@5"]), pct(mp["recall@5"]), *hits(mp)],
            ["Cố định 2 cũ + 3 mới", dec(mm["ndcg@5"]), pct(mm["recall@5"]), *hits(mm)],
            [
                "Tối đa 1 cũ nếu L &lt; 5, 2 nếu L ≥ 5",
                dec(cc["ndcg@5"]),
                pct(cc["recall@5"]),
                *hits(cc),
            ],
        ],
        hl=(0,),
    )
    + f"""<div class="note">L = số booking trước đó. Rút ra: ép tỉ lệ làm <b>giảm</b> điểm; mỗi cụm mới trúng thêm làm mất khoảng {dec(exch_fixed, 0)} cụm cũ. Gợi cụm mới đang yếu, cần thêm thông tin → hướng B.</div>""",
    sub="Giả thuyết: top 5 thiên về cụm cũ, chia lại cho cụm mới sẽ tốt hơn? (% trúng tính trên toàn bộ dòng test)",
)

wgrid = [str(w) for w in LF["w_grid"]]
slide(
    "4 · Thử nghiệm",
    "Hướng B1: thêm prior điểm đến",
    f"""<div class="eq">điểm = log p<sub>SMLP4Rec</sub> + w · log p<sub>prior</sub>(cụm | điểm đến)</div>
<div class="row">
  <div class="panel grow"><div class="ph">NDCG@5 theo w (w = 0 là plain)</div>
    {line_chart([w.replace(".", ",") for w in wgrid], [("Valid", [LF["sweep"]["valid"][w]["all"]["ndcg@5"] for w in wgrid], BLUE), ("Test", [LF["sweep"]["test"][w]["all"]["ndcg@5"] for w in wgrid], TEAL)], 0.20, 0.50, as_pct=False)}</div>
  <div class="panel side2">
    {table(["Test", "NDCG@5", "Recall@5"], [[name, dec(WARM[k]["ndcg@5"]), pct(WARM[k]["recall@5"])] for name, k in (("Plain", K_PLAIN), ("Prior một mình", K_PRIOR), (f"SMLP4Rec + prior (w = {dec(LFB, 1)})", K_FUSION))], hl=(2,))}
  </div>
</div>
<div class="note">Rút ra: prior là đòn bẩy lớn nhất (NDCG@5 {dec(WARM[K_PLAIN]["ndcg@5"])} → {dec(WARM[K_FUSION]["ndcg@5"])}; {pp(WARM[K_FUSION]["recall@5"] - WARM[K_PLAIN]["recall@5"], 1)} Recall@5), chủ yếu ở cụm mới (NDCG@5 {dec(LV[K_PLAIN]["new_target_rows"]["ndcg@5"], 3)} → {dec(FUS_TEST["new_target_rows"]["ndcg@5"], 3)}; Recall@5 {pct(LV[K_PLAIN]["new_target_rows"]["recall@5"], 0)} → {pct(FUS_TEST["new_target_rows"]["recall@5"], 0)}). Người dùng mới ({pct(COLD_SHARE, 0)} sự kiện test) chỉ dùng prior: NDCG@5 {dec(COLD_NDCG5)}, Recall@5 {pct(COLD_T["recall@5"], 1)}.</div>""",
    sub="Giả thuyết: điểm đến đang tìm quyết định cụm được đặt. Không huấn luyện lại; w chọn trên valid",
)

def slice_pair(name: str, key: str) -> tuple:
    r, n = H_SLICE[name][key].split(" / ")
    return float(r), float(n)


def ci(d: dict, m: str) -> str:
    lo, hi = d[m]["ci95"]
    return f"{pp(d[m]['diff'])} [{dec(lo * 100, 2)}; {dec(hi * 100, 2)}]"


KD_REF, KD_HYB = slice_pair("known destination", H_REF), slice_pair("known destination", K_HYBRID)
ND_REF, ND_HYB = slice_pair("new destination", H_REF), slice_pair("new destination", K_HYBRID)
W_NEW, W_KNOWN = HW_CELLS["0"], HW_CELLS["1"]

slide(
    "4 · Thử nghiệm",
    "Hướng B2: thêm sameDest, hybrid ba tín hiệu",
    f"""<div class="eq">điểm = log q<sub>SMLP4Rec</sub> + w<sub>p</sub> · log q<sub>prior</sub> + w<sub>s</sub> · log q<sub>sameDest</sub> &nbsp;·&nbsp; log q = log((1 − α)·p + α/100)</div>
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
<div class="note">Rút ra: hybrid cao nhất ở mọi cột; so với SMLP4Rec + prior: NDCG@5 {ci(HB, "ndcg@5")}, Recall@5 {ci(HB, "recall@5")} (bootstrap 95%). Tăng ở điểm đến đã đặt (NDCG@5 {dec(KD_REF[1])} → {dec(KD_HYB[1])}; Recall@5 {pct(KD_REF[0], 1)} → {pct(KD_HYB[0], 1)}), điểm đến mới gần như giữ nguyên (NDCG@5 {dec(ND_REF[1])} → {dec(ND_HYB[1])}; Recall@5 {pct(ND_REF[0], 1)} → {pct(ND_HYB[0], 1)}). Trọng số chọn trên valid (quy tắc 1-SE), α = {dec(H_ALPHA, 1)}, 2 bộ: điểm đến mới w<sub>p</sub> {dec(W_NEW[1], 2)}; đã đặt w<sub>p</sub> {dec(W_KNOWN[1], 2)}, w<sub>s</sub> {dec(W_KNOWN[2], 2)}.</div>""",
    sub="Giả thuyết: người dùng hay đặt lại cụm đã ở tại điểm đến này. Cột cuối: người dùng mới dùng prior",
)

slide(
    "4 · Thử nghiệm",
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

# ---- 4b. người dùng đã thấy / chưa thấy trong train (notebook 06)
PR = SU["presence"]
SR = SU["results"]["test"]
PL, HYK = "plain SMLP4Rec", "hybrid (SMLP4Rec + prior + sameDest)"
slide(
    "4 · Thử nghiệm",
    "Đánh giá theo người dùng đã thấy và chưa thấy trong train",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Tỉ lệ người dùng và sự kiện theo việc có mặt trong train</div>
    {
        grouped_bars(
            [
                ("Đã có trong train", [PR[s][k] / 100 for s, k in (("valid", "seen_users_pct"), ("test", "seen_users_pct"), ("valid", "seen_events_pct"), ("test", "seen_events_pct"))], BLUE),
                ("Chưa có (cold-start)", [PR[s][k] / 100 for s, k in (("valid", "unseen_users_pct"), ("test", "unseen_users_pct"), ("valid", "unseen_events_pct"), ("test", "unseen_events_pct"))], AMBER),
            ],
            ["Người dùng · valid", "Người dùng · test", "Sự kiện · valid", "Sự kiện · test"],
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Test: người dùng chưa thấy</div><div class="val">{pct(PR["test"]["unseen_users_pct"] / 100, 1)}</div><p>{num(PR["test"]["unseen_users"])} / {num(PR["test"]["users"])} người dùng</p></div>
    <div class="th-card"><div class="lbl">Trong đó đã có lịch sử (L ≥ 1)</div><div class="val">{num(PR["test"]["unseen_users_with_history_(L>=1)"])}</div><p>còn lại {num(PR["test"]["unseen_users_first_booking_only_(L=0)"])} chỉ có lần đặt đầu (L = 0)</p></div>
    <div class="th-card"><div class="lbl">Độ dài lịch sử trung bình, test</div><div class="val">{dec(PR["test"]["mean_history_len_seen"], 1)} vs {dec(PR["test"]["mean_history_len_unseen_L>=1"], 1)}</div><p>đã thấy vs chưa thấy (L ≥ 1)</p></div>
  </div>
</div>
<div class="note">Đã thấy = có ít nhất một dòng trong tập train của RecBole. Chưa thấy gồm người có lịch sử nhưng toàn bộ booking nằm sau cửa sổ train, và người đặt lần đầu (L = 0). Số sự kiện test: {num(PR["test"]["events"])}.</div>""",
    sub="Chia tập valid và test thành hai tập con không giao nhau theo việc người dùng có mặt trong train",
)

SUBS = [("seen users", "Đã thấy"), ("unseen users", "Chưa thấy (tất cả)"), ("unseen, L>=1", "Chưa thấy, L ≥ 1"), ("unseen, L=0 (first booking)", "Chưa thấy, L = 0*")]
su_gain = {s: SR[s][HYK]["ndcg@5"] - SR[s][PL]["ndcg@5"] for s, _ in SUBS}
slide(
    "4 · Thử nghiệm",
    "Plain và hybrid trên người dùng đã thấy và chưa thấy",
    f"""<div class="cmp2">
  <div class="panel"><div class="ph">NDCG@5, tập test</div>
    {grouped_bars([("SMLP4Rec plain", [SR[s][PL]["ndcg@5"] for s, _ in SUBS], BLUE), ("Hybrid", [SR[s][HYK]["ndcg@5"] for s, _ in SUBS], TEAL)], [n for _, n in SUBS], as_pct=False, w=520)}</div>
  <div class="panel"><div class="ph">Recall@5, tập test</div>
    {grouped_bars([("SMLP4Rec plain", [SR[s][PL]["recall@5"] for s, _ in SUBS], BLUE), ("Hybrid", [SR[s][HYK]["recall@5"] for s, _ in SUBS], TEAL)], [n for _, n in SUBS], w=520)}</div>
</div>
<div class="note">Rút ra: hybrid cao hơn plain ở mọi tập con (NDCG@5 đã thấy {dec(SR["seen users"][PL]["ndcg@5"], 3)} → {dec(SR["seen users"][HYK]["ndcg@5"], 3)}; chưa thấy {dec(SR["unseen users"][PL]["ndcg@5"], 3)} → {dec(SR["unseen users"][HYK]["ndcg@5"], 3)}). Plain gần như không phụ thuộc việc “đã thấy”: người chưa thấy nhưng có lịch sử đạt NDCG@5 {dec(SR["unseen, L>=1"][PL]["ndcg@5"], 3)} so với {dec(SR["seen users"][PL]["ndcg@5"], 3)}, vì mô hình chỉ đọc chuỗi booking. Chênh lệch chính là độ dài lịch sử. * L = 0 (lần đặt đầu): plain không có đầu vào nên dùng popularity toàn cục; hybrid dùng prior điểm đến.</div>""",
    sub="Cùng checkpoint và trọng số của notebook 04, không huấn luyện lại; chấm riêng từng tập con",
)

# ---- 4c. hai baseline cơ bản (notebook 07)
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
slide(
    "4 · Thử nghiệm",
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
slide(
    "4 · Thử nghiệm",
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

d_dest = bb_cell(3, "all events", "ndcg@5") - bb_cell(0, "all events", "ndcg@5")
d_hist = bb_cell(4, "all events", "ndcg@5") - bb_cell(3, "all events", "ndcg@5")
slide(
    "4 · Thử nghiệm",
    "Điểm cao đến từ đâu: điểm đến trước, lịch sử sau",
    f"""<div class="kpis three">
  <div class="kpi"><b>{dec(bb_cell(0, "all events", "ndcg@5"), 3)}</b><span>popularity toàn cục: không dùng thông tin gì</span></div>
  <div class="kpi"><b>{dec(bb_cell(3, "all events", "ndcg@5"), 3)}</b><span>thêm điểm đến và ngữ cảnh (hồi quy logistic): <b>+{dec(d_dest, 3)}</b></span></div>
  <div class="kpi"><b>{dec(bb_cell(4, "all events", "ndcg@5"), 3)}</b><span>thêm lịch sử cùng điểm đến và SMLP4Rec (hybrid): <b>+{dec(d_hist, 3)}</b></span></div>
</div>
<div class="cmp2">
  <div class="panel"><div class="ph">Đối chứng, test, người dùng có lịch sử (NDCG@5)</div>
    {table(["Phép thử", "Thật", "Xáo trộn"], [
        ["Hồi quy logistic: xáo trộn đặc trưng lượt tìm", dec(CTRL["LR real"]["ndcg@5"], 3), dec(CTRL["LR shuffled query features"]["ndcg@5"], 3)],
        ["ItemKNN: xáo trộn lịch sử giữa các dòng", dec(CTRL["ItemKNN real"]["ndcg@5"], 3), dec(CTRL["ItemKNN shuffled histories"]["ndcg@5"], 3)],
    ])}
    <p class="sm">Mốc popularity toàn cục cùng tập: {dec(CTRL["global popularity"]["ndcg@5"], 3)}. Xáo trộn đưa cả hai về mức popularity, nên tín hiệu là thật.</p>
  </div>
  <div class="panel"><div class="ph">Kiểm tra độc lập (scripts/verify_basic_baselines.py)</div>
    <p class="sm">Dựng lại phép chia, lịch sử, độ tương tự cosine và đặc trưng bằng code riêng; chấm lại ItemKNN và hồi quy logistic; so với kết quả đã lưu: <b>{VER_MISMATCH}</b> sai khác về hạng.</p>
    <p class="sm">Hồi quy logistic không dùng trường nào của khách sạn đã đặt, thành phố hay vùng của khách ({len(BV["logreg"]["forbidden_features"])} đặc trưng vi phạm). Huấn luyện kết thúc trước mục tiêu valid đầu tiên {abs(BL["verification"]["training_max_ts_minus_first_valid_target_ts"])} giây.</p>
  </div>
</div>
<div class="note">Kết luận: phần lớn điểm của hybrid giải thích được bằng điểm đến của lượt tìm, thông tin có sẵn trước khi đặt; lịch sử thêm phần còn lại. Không thấy dấu hiệu rò rỉ hay lỗi chia tập. Giới hạn: một seed; C = {dec(LM["C_pick"], 0)} là giá trị lớn nhất trong lưới nên có thể còn tăng nhẹ; hồi quy logistic chỉ chạy {LM["epochs"]} epoch.</div>""",
    sub="Mỗi baseline chặn một nguồn tín hiệu; hybrid cộng cả hai",
)


# ---- 5. kết luận
slide(
    "5 · Kết luận",
    "Kết luận",
    f"""<div class="grid3">
  <div class="card good"><h3>Đã làm</h3><p>Chạy SMLP4Rec gốc trên Expedia (plain), rút ra điểm yếu, thử hai hướng: chỉnh top 5 và thêm thông tin điểm đến (prior, rồi sameDest); cuối cùng thử tách trọng số theo thói quen.</p></div>
  <div class="card good"><h3>Rút ra</h3><p>Đa số đặt cụm mới; điểm đến là tín hiệu chủ đạo. Ép tỉ lệ làm giảm điểm; prior và sameDest tăng mạnh: NDCG@5 {dec(WARM[K_PLAIN]["ndcg@5"], 2)} → {dec(WARM[K_HYBRID]["ndcg@5"], 2)}, Recall@5 {pct(WARM[K_PLAIN]["recall@5"], 0)} → {pct(WARM[K_HYBRID]["recall@5"], 0)}. Tách theo thói quen không thêm gì. Hồi quy logistic chỉ dùng điểm đến đạt NDCG@5 {dec(BL["results"]["test"][LR_K]["all events"]["ndcg@5"], 2)} (mọi sự kiện test), ItemKNN chỉ dùng lịch sử đạt {dec(BL["results"]["test"][KNN_K]["all events"]["ndcg@5"], 2)}: điểm cao chủ yếu do điểm đến; hai baseline không cho thấy dấu hiệu rò rỉ.</p></div>
  <div class="card"><h3>Còn lại</h3><p>Một seed, prior tĩnh; hai baseline cơ bản (ItemKNN, hồi quy logistic) đã chạy, chưa có MF, item2vec, AdaGIN, LightGBM. Tiếp: đưa điểm đến vào trong mô hình, chạy lại trên chia theo sự kiện (gồm người dùng mới).</p></div>
</div>
<div class="note">Quyết định: SMLP4Rec + prior điểm đến + sameDest, 2 bộ trọng số (điểm đến mới / đã đặt); không tách theo thói quen.</div>""",
)


# ---- phụ lục: lưới trọng số
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
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ results/week3_implementation/*.json</span><span>{n} / {total}</span></div></section>'
        )
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tuần 3: Triển khai SMLP4Rec trên Expedia và các thử nghiệm bổ sung</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="deck">{"".join(parts)}</div><script>{JS}</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    render()
