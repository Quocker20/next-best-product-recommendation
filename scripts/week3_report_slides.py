"""Build the week-3 report deck (Vietnamese, HTML) in the visual style of the week-2 slides.

Sections: (1) design recap from week 2 (SMLP4Rec architecture + hybrid flow slides, taken as-is),
(2) from the raw SMLP4Rec code to the Expedia adaptation, (3) plain run results, (4) the
experiments and their findings, (5) conclusion. CSS, JS and the architecture diagram are read
from scripts/week2_methodology_slides.py so the style stays identical.

Every number on the slides is read from persisted outputs in reports/summary/week3_implementation/:
smlprec_expedia_run.json (plain run), smlprec_expedia_mixed_run.json (fixed quota),
smlprec_expedia_dynamic_cap_run.json (dynamic cap), smlprec_expedia_late_fusion.json (prior
fusion + cold users), smlprec_expedia_hybrid_samedest.json (hybrid with sameDest + bootstrap).

Usage: python scripts/week3_report_slides.py
Output: reports/summary/week3_implementation/week3_report_slides.html (PDF: print with headless Chrome)
"""

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W3 = ROOT / "results" / "week3_implementation"
OUT = ROOT / "reports" / "summary" / "week3_implementation" / "week3_report_slides.html"
WEEK2 = (ROOT / "scripts" / "week2_methodology_slides.py").read_text(encoding="utf-8")


def load(name: str) -> dict:
    return json.loads((W3 / name).read_text(encoding="utf-8"))


RUN = load("smlprec_expedia_run.json")
MIX = load("smlprec_expedia_mixed_run.json")
CAP = load("smlprec_expedia_dynamic_cap_run.json")
LF = load("smlprec_expedia_late_fusion.json")
HY = load("smlprec_expedia_hybrid_samedest.json")

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


def grouped_bars(series, groups, ymax=0.7, w=640, h=300) -> str:
    """series: [(label, values, color)], groups: category labels (inline SVG)."""
    top, bottom, left = 24, 30, 6
    ph, gw = h - top - bottom, (w - left) / len(groups)
    bw = gw * 0.22
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in (0.2, 0.4, 0.6):
        y = top + ph * (1 - g / ymax)
        out.append(f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(
            f'<text x="{left}" y="{y - 4:.1f}" font-size="11" fill="#94a3b8">{int(g * 100)}%</text>'
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
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" class="cv">{pct(v, 1)}</text>')
        out.append(
            f'<text x="{x0 + (len(series) * bw + 6 * (len(series) - 1)) / 2:.1f}" y="{h - 8}" class="cl">{g}</text>'
        )
    out.append("</svg>")
    legend = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def line_chart(xs, series, ymin, ymax, w=640, h=300) -> str:
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
            f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" font-size="11" fill="#94a3b8" text-anchor="end">{round(v * 100)}%</text>'
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
            f'<text x="{X(bi):.1f}" y="{Y(vals[bi]) - 9:.1f}" class="cv">{pct(vals[bi], 1)}</text>'
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
HW = HY["warm"]["test"]
HA = HY["all_events"]["test"]
K_PLAIN, K_PRIOR, K_FUSION = (
    "1. plain SMLP4Rec",
    "2. regional prior only",
    "3. SMLP4Rec + prior (w_p)",
)
K_HYBRID = "4. hybrid: + sameDest (per bucket, by Recall@5)"
CW = HY["chosen_weights_on_valid"]["hybrid_per_bucket"]
LFB = LF["best_global_w_on_valid"]
EX_LIST = [
    0.7**2 + 1.0,
    0.7,
]  # worked sameDest example below: cluster 12 (ages 3,0 -> 0.343 + 1) and cluster 3 (age 1)
EX12 = 0.7**3 + 0.7**0
EX3 = 0.7**1

# ================================================================== slides
# Flow: lý thuyết -> chạy plain -> rút ra gì -> thử nghiệm theo hướng nào -> kết luận.
# Only Recall@K and NDCG@K are reported (CLAUDE.md section 8, decision 2026-10-02).
FKEY = f"fusion_global_w={LFB}"
FUS_TEST = LF["variants"]["test"][FKEY]
COLD_T = LF["cold_users_prior_only"]["test"]["prior_only"]
COLD_SHARE = HY["cold_rows"]["test"] / (HY["cold_rows"]["test"] + HY["rows"]["test"])
TT = TB["test"]
RATIO = T["recall@5"] / POP["recall@5"]
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
  <div class="panel grow"><div class="ph">Recall@K, tập test, xếp hạng đầy đủ 100 cụm</div>
    {
        grouped_bars(
            [
                ("Popularity toàn cục", [POP[f"recall@{k}"] for k in (5, 10, 20)], GRAY),
                (
                    f"SMLP4Rec plain (epoch {RUN['best_epoch_by_valid']})",
                    [T[f"recall@{k}"] for k in (5, 10, 20)],
                    BLUE,
                ),
            ],
            ["Recall@5", "Recall@10", "Recall@20"],
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">SMLP4Rec plain, test</div><div class="val">Recall@5 {
        pct(T["recall@5"])
    }</div><p>NDCG@10 {dec(T["ndcg@10"])}</p></div>
    <div class="th-card"><div class="lbl">Recall@5 qua 3 epoch</div><div class="val">{
        pct(PE[0]["test"]["recall@5"])
    } → {pct(PE[-1]["test"]["recall@5"])}</div><p>thêm epoch gần như không giúp ({
        pp(PE[-1]["test"]["recall@5"] - PE[0]["test"]["recall@5"])
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
        ["Cách", "Recall@5", "NDCG@5", "Cụm cũ trúng", "Cụm mới trúng"],
        [
            ["Plain: 5 điểm cao nhất", pct(mp["recall@5"]), dec(mp["ndcg@5"]), *hits(mp)],
            ["Cố định 2 cũ + 3 mới", pct(mm["recall@5"]), dec(mm["ndcg@5"]), *hits(mm)],
            [
                "Tối đa 1 cũ nếu L &lt; 5, 2 nếu L ≥ 5",
                pct(cc["recall@5"]),
                dec(cc["ndcg@5"]),
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
  <div class="panel grow"><div class="ph">Recall@5 theo w (w = 0 là plain)</div>
    {line_chart([w.replace(".", ",") for w in wgrid], [("Valid", [LF["sweep"]["valid"][w]["all"]["recall@5"] for w in wgrid], BLUE), ("Test", [LF["sweep"]["test"][w]["all"]["recall@5"] for w in wgrid], TEAL)], 0.30, 0.65)}</div>
  <div class="panel side2">
    {table(["Test", "Recall@5", "Recall@10", "NDCG@10"], [["Plain", pct(HW[K_PLAIN]["all"]["recall@5"]), pct(HW[K_PLAIN]["all"]["recall@10"]), dec(HW[K_PLAIN]["all"]["ndcg@10"])], ["Prior một mình", pct(HW[K_PRIOR]["all"]["recall@5"]), pct(HW[K_PRIOR]["all"]["recall@10"]), dec(HW[K_PRIOR]["all"]["ndcg@10"])], [f"SMLP4Rec + prior (w = {dec(LFB, 1)})", pct(HW[K_FUSION]["all"]["recall@5"]), pct(HW[K_FUSION]["all"]["recall@10"]), dec(HW[K_FUSION]["all"]["ndcg@10"])]], hl=(2,))}
  </div>
</div>
<div class="note">Rút ra: prior là đòn bẩy lớn nhất ({pp(HW[K_FUSION]["all"]["recall@5"] - HW[K_PLAIN]["all"]["recall@5"], 1)} Recall@5), chủ yếu ở cụm mới ({pct(HW[K_PLAIN]["new_target_rows"]["recall@5"], 0)} → {pct(FUS_TEST["new_target_rows"]["recall@5"], 0)}). Người dùng mới ({pct(COLD_SHARE, 0)} sự kiện test) chỉ dùng prior: Recall@5 {pct(COLD_T["recall@5"], 1)}.</div>""",
    sub="Giả thuyết: điểm đến đang tìm quyết định cụm được đặt. Không huấn luyện lại; w chọn trên valid",
)

slide(
    "4 · Thử nghiệm",
    "Hướng B2: thêm sameDest, hybrid ba tín hiệu",
    f"""<div class="eq">điểm = log p<sub>SMLP4Rec</sub> + w<sub>p</sub> · log p<sub>prior</sub> + w<sub>s</sub> · sameDest</div>
<p class="muted">sameDest(cụm) = tổng 0,7<sup>tuổi</sup> các lần người dùng đặt cụm đó <b>tại đúng điểm đến đang tìm</b> (tuổi 0 = lần gần nhất). Ví dụ: cụm 12 đặt tại Cancún ở tuổi 0 và 3 → {
        dec(EX12, 3)
    }; đặt ở điểm đến khác → 0.</p>
{
        table(
            ["Cách (test)", "Recall@5", "Recall@10", "NDCG@10", "Recall@5, gồm người dùng mới"],
            [
                [
                    name,
                    pct(HW[k]["all"]["recall@5"]),
                    pct(HW[k]["all"]["recall@10"]),
                    dec(HW[k]["all"]["ndcg@10"]),
                    pct(HA[k]["recall@5"]),
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
<div class="note">Rút ra: hybrid cao nhất ở mọi cột. Trọng số tốt nhất: w<sub>p</sub> {
        dec(min(v[0] for v in CW.values()), 1)
    }–{dec(max(v[0] for v in CW.values()), 1)}, w<sub>s</sub> = {
        dec(max(v[1] for v in CW.values()), 0)
    } (mép lưới).</div>""",
    sub="Giả thuyết: người dùng hay đặt lại cụm đã ở tại điểm đến này. Cột cuối: người dùng mới dùng prior",
)

# ---- 5. kết luận
slide(
    "5 · Kết luận",
    "Kết luận",
    f"""<div class="grid3">
  <div class="card good"><h3>Đã làm</h3><p>Chạy SMLP4Rec gốc trên Expedia (plain), rút ra điểm yếu, thử hai hướng: chỉnh top 5 và thêm thông tin điểm đến.</p></div>
  <div class="card good"><h3>Rút ra</h3><p>Đa số đặt cụm mới; điểm đến là tín hiệu chủ đạo. Ép tỉ lệ làm giảm điểm; prior và sameDest tăng mạnh: Recall@5 {pct(HW[K_PLAIN]["all"]["recall@5"], 0)} → {pct(HW[K_HYBRID]["all"]["recall@5"], 0)}.</p></div>
  <div class="card"><h3>Còn lại</h3><p>Một seed, prior tĩnh, w<sub>s</sub> ở mép lưới, chưa so với mô hình đếm lịch sử. Tiếp: đưa điểm đến vào trong mô hình, chạy lại trên chia theo sự kiện (gồm người dùng mới).</p></div>
</div>
<div class="note">Quyết định: SMLP4Rec + prior điểm đến + sameDest, trọng số theo nhóm số booking trước đó.</div>""",
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
