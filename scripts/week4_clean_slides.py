"""Build the week-4 deck (Vietnamese, HTML): last week's results, the cleaning pipeline, the rerun on the
cleaned data, its consequences, the final results; appendix with the prior formula, the sameDest formula and
the query-token feature engineering.

Every number is read from results/week4_rebuild/*.json (no number is typed here). Charts are inline SVG.
Style, JS and the architecture diagram are reused from the week-2 / week-3 decks.

Usage: python scripts/week4_clean_slides.py
Output: reports/summary/week4_rebuild/week4_clean_slides.html (PDF: print with headless Chrome)
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W4 = ROOT / "results" / "week4_rebuild"
OUT = ROOT / "reports" / "summary" / "week4_rebuild" / "week4_clean_slides.html"
WEEK2 = (ROOT / "scripts" / "week2_methodology_slides.py").read_text(encoding="utf-8")


def J(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- style, JS, diagram (reused)
CSS = re.search(r'\nCSS = """(.*?)"""', WEEK2, re.DOTALL).group(1)
JS = re.search(r'\nJS = """(.*?)"""', WEEK2, re.DOTALL).group(1)
NAVY, BLUE, TEAL, AMBER, RED, GRAY, LGRAY = "#1e3a8a", "#2563eb", "#0f766e", "#b45309", "#dc2626", "#94a3b8", "#cbd5e1"
_ns = {"NAVY": NAVY, "BLUE": BLUE, "TEAL": TEAL, "AMBER": AMBER, "RED": RED, "GRAY": GRAY}
_start = WEEK2.index("def architecture_svg")
_end = _start + 1 + re.search(r"\n(?=\S)", WEEK2[_start + 1 :]).start()
exec(WEEK2[_start:_end], _ns)  # noqa: S102 (the week-2 diagram, unchanged)
architecture_svg = _ns["architecture_svg"]

CSS += """
.lay{display:flex;gap:46px;flex:1;min-height:0;align-items:center}
.ch{flex:1;min-width:0}.tx{width:318px;display:flex;flex-direction:column;gap:20px;flex:none}
.stat{border-top:1.5px solid var(--line);padding-top:12px}.stat:first-child{border-top:none;padding-top:0}
.stat b{display:block;font-size:36px;font-weight:700;color:var(--navy);line-height:1.1}
.stat b.red{color:var(--red)}.stat b.teal{color:var(--teal)}.stat b.amber{color:var(--amber)}
.stat span{font-size:14.5px;color:var(--muted);line-height:1.45;display:block;margin-top:4px}
.take{font-size:17px;color:var(--navy);font-weight:600;border-left:4px solid var(--navy);padding:4px 0 4px 16px;line-height:1.45}
.ct{font-size:13.5px;font-weight:600;color:var(--muted);margin-bottom:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start}
.eq{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 20px;font-family:Consolas,monospace;font-size:17px;color:#1e3a8a;line-height:1.75}
.eq small{display:block;font-family:'Be Vietnam Pro',sans-serif;font-size:13px;color:var(--sub);margin-top:2px}
.tbl.dense td{padding:7px 12px;font-size:14.5px}.tbl.dense th{padding:7px 12px;font-size:11.5px}
.tbl.dense td.r,.tbl.dense th.r{text-align:right}
.tbl.dense tr.hl td{background:#f0fdfa;color:#134e4a;font-weight:600}
.sm{font-size:13px;color:var(--sub);line-height:1.5}
ul.clean.t{gap:11px;font-size:15.5px}
.col h3{font-size:19px;margin-bottom:12px;color:var(--navy)}
.legend{flex-wrap:wrap;gap:6px 16px}.lg{white-space:nowrap}
h2{font-size:30px;line-height:1.2}.pg{white-space:nowrap}
.stat b.s{font-size:27px}
.tbl.tight td{padding:4px 10px;font-size:13px}.tbl.tight th{padding:5px 10px;font-size:11px}.tbl.tight td:first-child{white-space:nowrap}
ul.clean.t.lg2{font-size:17.5px;gap:16px}
"""

# ---------------------------------------------------------------- formatting
def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def dec(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def sdec(x: float, d: int = 4) -> str:
    return f"{x:+.{d}f}".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


# ---------------------------------------------------------------- charts (inline SVG)
def hbar(labels, values, colors, vmax, w=700, row=48, label_w=230, fmt=dec, log=False, vmin=1.0, subs=None) -> str:
    """Horizontal bars. Bar length is linear in value, or in log10(value) when log=True."""
    h = row * len(values) + 10
    pw = w - label_w - 90

    def length(v: float) -> float:
        if log:
            return pw * (math.log10(max(v, vmin)) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        return pw * v / vmax

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        y = 5 + i * row
        bh = row * 0.58
        out.append(f'<text x="{label_w - 12}" y="{y + row * 0.5 + (0 if subs else 5):.1f}" font-size="14.5" fill="#334155" text-anchor="end">{lab}</text>')
        if subs:
            out.append(f'<text x="{label_w - 12}" y="{y + row * 0.5 + 15:.1f}" font-size="12" fill="#94a3b8" text-anchor="end">{subs[i]}</text>')
        out.append(f'<rect x="{label_w}" y="{y + (row - bh) / 2 - 4:.1f}" width="{max(length(v), 2):.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
        out.append(f'<text x="{label_w + length(v) + 8:.1f}" y="{y + row * 0.5 + 4:.1f}" font-size="14" font-weight="600" fill="#334155">{fmt(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def hbar_signed(labels, values, colors, vmin, vmax, w=760, row=46, label_w=230, fmt=lambda v: sdec(v, 4)) -> str:
    """Horizontal bars around zero (vmin < 0 < vmax)."""
    h = row * len(values) + 28
    pw = w - label_w - 30
    x0 = label_w + pw * (-vmin) / (vmax - vmin)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">', f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="6" y2="{h - 22}" stroke="#94a3b8"/>']
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        y = 8 + i * row
        bh = row * 0.55
        x = x0 if v >= 0 else x0 + pw * v / (vmax - vmin)
        wd = abs(pw * v / (vmax - vmin))
        out.append(f'<text x="{label_w - 12}" y="{y + row * 0.5:.1f}" font-size="14.5" fill="#334155" text-anchor="end">{lab}</text>')
        out.append(f'<rect x="{x:.1f}" y="{y + (row - bh) / 2 - 6:.1f}" width="{max(wd, 2):.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
        tx = x + wd + 8 if v >= 0 else x - 8
        out.append(f'<text x="{tx:.1f}" y="{y + row * 0.5:.1f}" font-size="14" font-weight="600" fill="#334155" text-anchor="{"start" if v >= 0 else "end"}">{fmt(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def grouped_bars(series, groups, ymax, w=760, h=330, grid=(0.1, 0.2, 0.3, 0.4), fmt=lambda v: dec(v, 3)) -> str:
    """Vertical grouped bars; series [(label, values, color)]."""
    top, left, bottom = 24, 6, 48
    ph, gw = h - top - bottom, (w - left) / len(groups)
    bw = min(gw * 0.26, 44)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in grid:
        y = top + ph * (1 - g / ymax)
        out.append(f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/><text x="{left}" y="{y - 4:.1f}" font-size="11" fill="#94a3b8">{dec(g, 1)}</text>')
    for gi, g in enumerate(groups):
        span = len(series) * bw + 6 * (len(series) - 1)
        x0 = left + gi * gw + (gw - span) / 2
        for si, (_, vals, col) in enumerate(series):
            v = vals[gi]
            bh = ph * v / ymax
            x, y = x0 + si * (bw + 6), top + ph - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" class="cv" style="font-size:11.5px">{fmt(v)}</text>')
        for li, part in enumerate(g.split("|")):
            out.append(f'<text x="{x0 + span / 2:.1f}" y="{top + ph + 18 + li * 15}" class="cl">{part}</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def hist_log(edges, counts, marks, w=760, h=330, color=BLUE) -> str:
    """Histogram on a log10 x axis. marks: [(x_seconds, label)] drawn as dashed vertical lines."""
    left, right, top, bottom = 46, 10, 26, 54
    pw, ph = w - left - right, h - top - bottom
    lo, hi = math.log10(edges[0]), math.log10(edges[-1])
    X = lambda v: left + pw * (math.log10(v) - lo) / (hi - lo)
    cmax = max(counts)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in range(0, 7001, 1000):
        if g > cmax * 1.05:
            break
        y = top + ph * (1 - g / (cmax * 1.08))
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/><text x="{left - 6}" y="{y + 4:.1f}" font-size="11" fill="#94a3b8" text-anchor="end">{num(g)}</text>')
    for i, c in enumerate(counts):
        x1, x2 = X(edges[i]), X(edges[i + 1])
        bh = ph * c / (cmax * 1.08)
        out.append(f'<rect x="{x1 + 1:.1f}" y="{top + ph - bh:.1f}" width="{x2 - x1 - 2:.1f}" height="{bh:.1f}" fill="{color}"/>')
    for xv, lab in marks:
        x = X(xv)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{top + ph}" stroke="{RED}" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{x + 5:.1f}" y="{top + 8}" font-size="12.5" font-weight="600" fill="{RED}">{lab}</text>')
    for xv, lab in ((60, "1 phút"), (3600, "1 giờ"), (86400, "1 ngày"), (604800, "1 tuần"), (2592000, "30 ngày")):
        out.append(f'<text x="{X(xv):.1f}" y="{h - 26}" class="cl">{lab}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{h - 5}" class="cl">khoảng cách tới booking được lặp lại (thang log)</text></svg>')
    return "".join(out)


def dotplot(rows, xmin, xmax, w=760, row=96, label_w=240, right_w=90) -> str:
    """rows: [(label, [(series_label, mean, std, [seed values], color)], paired_text)].  Whiskers = mean +- std."""
    h = row * len(rows) + 34
    pw = w - label_w - right_w
    X = lambda v: label_w + pw * (v - xmin) / (xmax - xmin)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    t = xmin
    while t <= xmax + 1e-9:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="4" y2="{h - 26}" stroke="#e2e8f0"/><text x="{X(t):.1f}" y="{h - 8}" class="cl">{dec(t, 2)}</text>')
        t += 0.03
    for ri, (lab, ser, paired) in enumerate(rows):
        y0 = 8 + ri * row
        out.append(f'<text x="{label_w - 14}" y="{y0 + row * 0.42:.1f}" font-size="14.5" fill="#334155" text-anchor="end">{lab}</text>')
        for si, (_, mean, std, vals, col) in enumerate(ser):
            y = y0 + 16 + si * 30
            out.append(f'<line x1="{X(mean - std):.1f}" x2="{X(mean + std):.1f}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="3"/>')
            out.append(f'<line x1="{X(mean):.1f}" x2="{X(mean):.1f}" y1="{y - 9}" y2="{y + 9}" stroke="{col}" stroke-width="3"/>')
            for v in vals:
                out.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5.5" fill="#fff" stroke="{col}" stroke-width="2.5"/>')
        out.append(f'<text x="{w - 6}" y="{y0 + row * 0.42:.1f}" font-size="15" font-weight="700" fill="{TEAL}" text-anchor="end">{paired}</text>')
    out.append("</svg>")
    return "".join(out)


def table(head, rows, hl=(), right_from=9, tight=False) -> str:
    th = "".join(f'<th class="{"r" if i >= right_from else ""}">{h}</th>' for i, h in enumerate(head))
    trs = "".join(
        f'<tr{" class=hl" if ri in hl else ""}>' + "".join(f'<td class="{"r" if i >= right_from else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>"
        for ri, r in enumerate(rows)
    )
    return f'<table class="tbl dense{" tight" if tight else ""}"><tr>{th}</tr>{trs}</table>'


def lay(chart: str, stats: list, title: str = "") -> str:
    st = "".join(f'<div class="stat"><b class="{c}">{b}</b><span>{s}</span></div>' for b, s, c in stats)
    ct = f'<div class="ct">{title}</div>' if title else ""
    return f'<div class="lay"><div class="ch">{ct}{chart}</div><div class="tx">{st}</div></div>'


def take(text: str) -> str:
    return f'<div class="take">{text}</div>'


# ---------------------------------------------------------------- data
OLD_Q, NEW_Q = J(W4 / "with_burst/smlprec_query_run.json"), J(W4 / "smlprec_query_run.json")
OLD_B = J(W4 / "with_burst/basic_baselines.json")["results"]["test"]
NEW_B = J(W4 / "basic_baselines.json")["results"]["test"]
EDA, AUD, CNT = J(W4 / "eda_clean.json"), J(W4 / "data_audit.json"), J(W4 / "bookings_counts.json")
SENS, SEEDS = J(W4 / "sensitivity.json"), J(W4 / "seeds_summary.json")
COLD = J(W4 / "smlprec_query_cold_L0.json")
RUN1D = J(W4 / "smlprec_query_cold_train_run.json")
KNN, LR = "ItemKNN (cosine, K=100, history only)", "Logistic regression (C=10.0, destination + context, no history)"
WARM_B, COLD_B = "warm (all L>=1 targets)", "unseen, L=0 (first booking)"


def bench(run: dict, name: str, scope: str = "warm") -> dict:
    b = run["samedest_hybrid"]["benchmark_test"][scope]
    return b[next(k for k in b if k.startswith(name))]


def nd(d: dict) -> float:
    return d["ndcg@5"]


def split_cell(s: str) -> tuple:
    a, b = s.split(" / ")
    return float(a), float(b)


MODELS = [  # label, old NDCG@5, new NDCG@5
    ("ItemKNN", nd(OLD_B[KNN][WARM_B]), nd(NEW_B[KNN][WARM_B])),
    ("Logistic|regression", nd(OLD_B[LR][WARM_B]), nd(NEW_B[LR][WARM_B])),
    ("SMLP4Rec|chỉ lịch sử", nd(bench(OLD_Q, "SMLP4Rec plain")), nd(bench(NEW_Q, "SMLP4Rec plain"))),
    ("+ query|token", nd(OLD_Q["samedest_hybrid"]["benchmark_test"]["warm"]["Query token"]), nd(NEW_Q["samedest_hybrid"]["benchmark_test"]["warm"]["Query token"])),
    ("+ sameDest|(mô hình chính)", nd(OLD_Q["samedest_hybrid"]["benchmark_test"]["warm"]["Query token + sameDest"]), nd(NEW_Q["samedest_hybrid"]["benchmark_test"]["warm"]["Query token + sameDest"])),
    ("Hybrid cũ|(+ prior)", nd(bench(OLD_Q, "Hybrid")), nd(bench(NEW_Q, "Hybrid"))),
]
N_OLD, N_NEW = bench(OLD_Q, "SMLP4Rec plain")["n"], bench(NEW_Q, "SMLP4Rec plain")["n"]
PRIOR_OLD = OLD_Q["prior_fusion_check"]["test"]["destination prior only"]
PRIOR_NEW = NEW_Q["prior_fusion_check"]["test"]["destination prior only"]

slides = []


def slide(meta, title, body, sub=""):
    slides.append((meta, title, sub, body))


# ================================================================== 1. cover
slides.append(
    (
        "cover",
        "",
        "",
        """<div class="cover">
  <div class="kicker">Tuần 4 · Làm sạch dữ liệu</div>
  <h1>Làm sạch Expedia và chạy lại SMLP4Rec:<br><span>điểm của các mô hình dùng lịch sử giảm, thứ hạng giữ nguyên</span></h1>
  <p class="lead">Next-Best-Product Recommendation · Vinpearl × GSM · Bối cảnh (a): du lịch</p>
</div>""",
    )
)

# ================================================================== 2. algorithm (as last week)
slide(
    "1 · Nhắc lại tuần trước",
    "SMLP4Rec: đoán cụm khách sạn tiếp theo từ lịch sử booking",
    f"""<div class="row">
  <div class="panel diagram-wrap">{architecture_svg()}</div>
  <div class="panel side steps">
    <div class="step"><span>1</span><p>Đầu vào: chuỗi các cụm đã đặt, cộng query token là lượt tìm hiện tại.</p></div>
    <div class="step"><span>2</span><p>Ba MLP song song trộn theo <b>thời gian</b>, <b>context</b>, <b>embedding</b>, lặp L lần.</p></div>
    <div class="step"><span>3</span><p>Lấy hàng cuối, so với 100 cụm để xếp hạng; sameDest cộng điểm cho cụm đã đặt tại đúng destination đang tìm.</p></div>
  </div>
</div>""",
    sub="Thuật toán không đổi so với tuần trước; tuần này chỉ đổi dữ liệu",
)

# ================================================================== 3. last week's results
wk = OLD_Q["samedest_hybrid"]["benchmark_test"]["warm"]
labels = ["ItemKNN", "Logistic regression", "SMLP4Rec chỉ lịch sử", "+ query token", "+ sameDest (mô hình chính)"]
vals = [MODELS[0][1], MODELS[1][1], MODELS[2][1], MODELS[3][1], MODELS[4][1]]
d_qt, d_sd = MODELS[3][1] - MODELS[2][1], MODELS[4][1] - MODELS[3][1]
slide(
    "1 · Nhắc lại tuần trước",
    f"Query token nâng NDCG@5 từ {dec(MODELS[2][1], 2)} lên {dec(MODELS[3][1], 2)}; sameDest thêm {dec(d_sd, 3)}",
    lay(
        hbar(labels, vals, [LGRAY, LGRAY, GRAY, BLUE, TEAL], 0.5, fmt=lambda v: dec(v, 4)),
        [
            (sdec(d_qt, 3), "NDCG@5 khi đưa lượt tìm hiện tại vào mô hình (query token so với chỉ lịch sử)", ""),
            (sdec(d_sd, 3), "NDCG@5 sameDest cộng thêm trên query token", "teal"),
            (num(N_OLD), "dòng warm trong tập test (user đã có lịch sử)", ""),
        ],
        "NDCG@5, dòng warm của tập test, dữ liệu thô, một seed",
    )
    + take("Kết quả tuần trước chạy trên dữ liệu chưa kiểm tra lỗi; tuần này kiểm tra xem chúng còn đứng vững không."),
    sub="Chia theo thời gian 80/10/10, xếp hạng đủ 100 cụm, seed 2022, 3 epoch",
)

# ================================================================== 4. burst repeats
bg = EDA["E1"]["burst_gap_seconds"]
share_rows = CNT["flagged_stats_before_collapse"]["burst_repeat_share"]
test_burst = AUD["targets_approx_split"]["test"]["burst_target"]["share"]
ceil = AUD["targets_approx_split"]["test"]
ceil_share = ceil["ceiling_rows_that_are_burst"]["n"] / ceil["sameDest_ceiling (target cluster booked earlier at query destination)"]["n"]
slide(
    "2 · Pipeline làm sạch",
    f"{num(bg['n'])} booking lặp y hệt booking trước đó, {pct(bg['share_within_1h'], 0)} cách nhau dưới 1 giờ",
    lay(
        hist_log(bg["hist_edges"], bg["hist_counts"], [(3600, f"≤ 1 giờ: {pct(bg['share_within_1h'], 0)}"), (86400, f"≤ 1 ngày: {pct(bg['share_within_1d'], 0)}")]),
        [
            (pct(test_burst, 2), "target của tập test là booking lặp; nếu bản gốc nằm trong lịch sử, mô hình chỉ cần chép lại đáp án", "red"),
            (pct(ceil_share, 0), "số dòng mà sameDest có thể trúng chỉ nhờ booking lặp", "red"),
            (pct(share_rows, 1), "tổng số booking bị ảnh hưởng", ""),
        ],
        "Khoảng cách giữa booking lặp và booking gốc cùng user (cùng destination, ngày, cluster)",
    )
    + take("Hai đỉnh: lặp trong vài phút (tìm và đặt lại ngay) và lặp sau khoảng một ngày; cả hai đều ghi cùng một chuyến."),
    sub="Lặp = cùng user, destination, ngày check-in, ngày check-out và cluster với booking ngay trước",
)

# ================================================================== 5. cleaning rules
R = EDA["E1"]["rows_per_rule"]
rules = [
    ("Booking lặp (collapse)", R["flag_burst_repeat"], NAVY),
    ("Trùng khóa (xóa)", R["duplicate_keys_dropped"], NAVY),
    ("0 người lớn (xóa giá trị, cờ)", R["flag_zero_adults"], AMBER),
    ("Check-in trước ngày tìm (xóa giá trị, cờ)", R["flag_ci_before_search"], AMBER),
    ("0 phòng (xóa giá trị, cờ)", R["flag_zero_rooms"], AMBER),
    ("Check-out trước check-in (xóa giá trị, cờ)", R["flag_co_before_ci"], AMBER),
    ("User từ 50 booking (chỉ cờ)", R["flag_heavy_user"], GRAY),
    ("Lặp cùng phiên, khác ngày (chỉ cờ)", R["flag_session_repeat"], GRAY),
    ("Quá 4 khách mỗi phòng (chỉ cờ)", R["flag_occupancy_odd"], GRAY),
    ("Mã địa lý bằng 0 (chỉ cờ)", R["flag_unknown_geo"], GRAY),
    ("Check-in sau hơn 365 ngày (chỉ cờ)", R["flag_extreme_lead"], GRAY),
]
rules.sort(key=lambda t: ({NAVY: 0, AMBER: 1, GRAY: 2}[t[2]], -t[1]))
rows_before, rows_after = CNT["flagged_stats_before_collapse"]["rows"], CNT["clean_stats"]["rows"]
slide(
    "2 · Pipeline làm sạch",
    "Chỉ bỏ dòng khi nó ghi cùng một sự kiện hai lần; mọi bất thường khác chỉ gắn cờ",
    lay(
        hbar([r[0] for r in rules], [r[1] for r in rules], [r[2] for r in rules], 200000, label_w=330, row=40, fmt=num, log=True, w=800),
        [
            (f"{num(rows_before)} → {num(rows_after)}", "booking sau khi bỏ trùng khóa và collapse; user giữ nguyên 813.985", "teal"),
            (num(R["flag_session_repeat"]), "lặp cùng phiên nhưng khác ngày vẫn giữ: có thể là đổi chuyến thật, sẽ đo riêng", ""),
        ],
        "Số dòng chịu tác động của từng quy tắc (thang log)",
    )
    + take("Xanh đậm: bỏ khỏi dữ liệu · cam: xóa giá trị lỗi, giữ dòng · xám: chỉ gắn cờ để phân tích."),
    sub="Kênh số 0 là kênh hợp lệ và dữ liệu không có cột giá, nên hai mục này không cần xử lý",
)

# ================================================================== 6. rerun on clean data
slide(
    "3 · Chạy lại trên dữ liệu sạch",
    "Sáu mô hình chạy lại: mọi mô hình dùng lịch sử đều giảm điểm",
    lay(
        grouped_bars(
            [("Dữ liệu thô", [m[1] for m in MODELS], LGRAY), ("Dữ liệu đã làm sạch", [m[2] for m in MODELS], BLUE)],
            [m[0] for m in MODELS],
            0.5,
            w=800,
        ),
        [
            (f"{num(N_OLD)} → {num(N_NEW)}", "dòng warm của tập test: trước và sau khi bỏ booking lặp", "teal"),
            ("1 thay đổi", "chỉ dữ liệu: cùng cấu hình, seed 2022, 3 epoch", ""),
        ],
        "NDCG@5, dòng warm của tập test",
    )
    + take("Thứ hạng giữa các mô hình không đổi; chỉ mức điểm của các mô hình dùng lịch sử thấp hơn."),
    sub="Cả các thí nghiệm tuần trước (quota top-5, prior, hybrid, tách trọng số theo hành vi) được chạy lại, cho cùng kết luận",
)

# ================================================================== 7. consequence 1
deltas = [(m[0].replace("|", " "), m[2] - m[1]) for m in MODELS] + [("Prior theo destination", nd(PRIOR_NEW) - nd(PRIOR_OLD))]
order = [2, 0, 3, 4, 5, 6, 1]  # history only .. history + context .. no history
dl = [deltas[i] for i in order]
cols = [RED, RED, AMBER, AMBER, AMBER, GRAY, GRAY]
slide(
    "4 · Hệ quả",
    f"Càng dựa vào lịch sử càng bị phồng điểm: {sdec(dl[0][1], 4)} NDCG@5, còn gần 0 khi không dùng lịch sử",
    lay(
        hbar_signed([d[0] for d in dl], [d[1] for d in dl], cols, -0.02, 0.004),
        [
            (f"{sdec(dl[0][1], 4)} · {sdec(dl[1][1], 4)}", "SMLP4Rec chỉ lịch sử và ItemKNN: mất nhiều nhất, vì chỉ có thể đoán lại các booking đã có", "red s"),
            (f"{sdec(max(d[1] for d in dl[2:5]), 4)} đến {sdec(min(d[1] for d in dl[2:5]), 4)}", "query token, mô hình chính, hybrid cũ: có cả lịch sử và ngữ cảnh, mất vừa", "amber s"),
            (f"{sdec(dl[5][1], 4)} · {sdec(dl[6][1], 4)}", "prior theo destination và logistic regression: không dùng lịch sử, gần như không đổi", "s"),
        ],
        "Thay đổi NDCG@5 sau khi làm sạch, dòng warm của tập test",
    )
    + take("Hai mô hình không nhìn lịch sử không đổi: lỗi nằm ở dữ liệu trùng, không phải ở mô hình."),
)

# ================================================================== 8. consequence 2: sameDest
g = SENS["sameDest_gain"]
known = split_cell(NEW_Q["samedest_hybrid"]["test_slices_recall@5_ndcg@5"]["known destination"]["Query token"])
known_sd = split_cell(NEW_Q["samedest_hybrid"]["test_slices_recall@5_ndcg@5"]["known destination"]["Query token + sameDest"])
known_rows = NEW_Q["samedest_hybrid"]["test_slices_recall@5_ndcg@5"]["known destination"]["rows"]
sd_items = [
    ("Tất cả dòng warm", g["all_warm"]["gain_bootstrap"]["ndcg@5"]["diff"], g["all_warm"]["rows"]),
    ("Bỏ target đặt lại trong 1 giờ", g["excluding_rebook_within_1h"]["gain_bootstrap"]["ndcg@5"]["diff"], g["excluding_rebook_within_1h"]["rows"]),
    ("Bỏ target đặt lại trong 1 ngày", g["excluding_rebook_within_1d"]["gain_bootstrap"]["ndcg@5"]["diff"], g["excluding_rebook_within_1d"]["rows"]),
    ("Chỉ dòng có destination trong lịch sử", known_sd[1] - known[1], known_rows),
]
slide(
    "4 · Hệ quả",
    f"sameDest còn {sdec(sd_items[1][1], 4)} NDCG@5 sau khi bỏ các lần đặt lại trong 1 giờ",
    lay(
        hbar([a for a, _, _ in sd_items], [v for _, v, _ in sd_items], [BLUE, TEAL, TEAL, GRAY], 0.1, label_w=300, w=760, row=60, fmt=lambda v: sdec(v, 4), subs=[f"{num(n)} dòng" for _, _, n in sd_items]),
        [
            (pct(known_rows / g["all_warm"]["rows"], 0), "số dòng warm có destination đang tìm trong lịch sử; sameDest chỉ tác động ở đây", ""),
            (pct(EDA["E5"]["test_pair_given_destination_in_history"], 1), "khi đó đáp án là một cluster đã đặt tại chính destination ấy", "teal"),
        ],
        "Gain NDCG@5 của sameDest so với query token (tập test)",
    )
    + take("Tín hiệu thật nhưng hẹp, và khoảng 35–45% gain đến từ việc đặt lại trong phiên; số sau khi bỏ là cận dưới vì lọc theo đáp án. Giữ sameDest."),
)

# ================================================================== 9. consequence 3: cold users
c = COLD["results"]["test"]
oov = COLD["test_slices_ndcg5_recall5"]["destination out of vocab"]
qt_cold, pr_cold = nd(c["query token alone (empty history)"]), nd(c["destination prior only"])
cold_items = [
    ("ItemKNN (rơi về popularity)", nd(NEW_B[KNN][COLD_B]), LGRAY),
    ("SMLP4Rec query token", qt_cold, BLUE),
    ("Logistic regression", nd(NEW_B[LR][COLD_B]), LGRAY),
    ("Prior theo destination", pr_cold, GRAY),
]
qt_lr = SENS["cold_rows"]["query_token minus logistic_regression"]["recall@5"]
slide(
    "4 · Hệ quả",
    f"User mới: query token thấp hơn logistic regression {dec(abs(qt_lr['diff']), 4)} Recall@5",
    lay(
        hbar([a for a, _, _ in cold_items], [v for _, v, _ in cold_items], [col for _, _, col in cold_items], 0.5, label_w=250, fmt=lambda v: dec(v, 4)),
        [
            (f"{sdec(qt_lr['diff'], 4)}", f"Recall@5, query token trừ logistic regression, khoảng tin cậy [{dec(qt_lr['ci95'][0], 4)}; {dec(qt_lr['ci95'][1], 4)}]", "red"),
            (pct(COLD["test_share_destination_oov"], 1), f"dòng cold có destination ngoài từ vựng: NDCG@5 chỉ {dec(nd(oov['query token alone (empty history)']), 4)}, trong khi prior đạt {dec(nd(oov['destination prior only']), 4)}", "amber"),
        ],
        "NDCG@5 trên dòng cold (lần đặt đầu của user, lịch sử rỗng)",
    )
    + take("Nguyên nhân: mô hình chỉ học từ booking thứ hai trở đi nên chưa bao giờ thấy lịch sử rỗng, còn logistic regression được học cả lần đặt đầu. Không gắn prior vào mô hình chính vì khó giải thích; sửa ở dữ liệu huấn luyện."),
)

# ================================================================== 10. adjustment: train with first bookings
tr_rows, first_rows = EDA["splits"]["train"]["rows"], EDA["splits"]["first_train"]["rows"]
oov_old = NEW_Q["query_coverage"]["dest"]["oov_test_targets"]
oov_new = RUN1D["query_coverage"]["dest"]["oov_test_targets"]
bar_w = 760
a_w = bar_w * tr_rows / (tr_rows + first_rows)
stack = (
    f'<svg viewBox="0 0 {bar_w} 150" class="chart">'
    f'<rect x="0" y="40" width="{a_w:.1f}" height="54" fill="{LGRAY}"/><rect x="{a_w:.1f}" y="40" width="{bar_w - a_w:.1f}" height="54" fill="{TEAL}"/>'
    f'<text x="{a_w / 2:.1f}" y="73" font-size="16" font-weight="600" fill="#334155" text-anchor="middle">{num(tr_rows)} target (booking thứ hai trở đi)</text>'
    f'<text x="{a_w + (bar_w - a_w) / 2:.1f}" y="73" font-size="16" font-weight="600" fill="#fff" text-anchor="middle">+{num(first_rows)}</text>'
    f'<text x="{bar_w}" y="118" font-size="14" fill="{TEAL}" text-anchor="end" font-weight="600">lần đặt đầu của user, lịch sử rỗng (+{pct(first_rows / tr_rows, 1)})</text>'
    f'<text x="0" y="24" font-size="13" fill="#64748b">Dòng huấn luyện</text></svg>'
)
slide(
    "4 · Hệ quả",
    f"Thêm {num(first_rows)} lần đặt đầu vào tập huấn luyện (+{pct(first_rows / tr_rows, 0)}); mọi thứ khác giữ nguyên",
    lay(
        stack
        + '<ul class="clean t" style="margin-top:18px"><li><b>Giữ nguyên:</b> mô hình, cấu hình, seed, 3 epoch, chia tập, cách chọn epoch và các dòng đánh giá.</li>'
        '<li><b>Lịch sử rỗng:</b> toàn padding, query token vẫn đủ 14 trường; từ vựng query tính từ target train cộng các dòng này.</li>'
        "<li><b>Không rò rỉ:</b> các dòng nằm hoàn toàn trước điểm cắt train và không trùng user với cold valid/test.</li></ul>",
        [
            (f"{pct(oov_old, 2)} → {pct(oov_new, 2)}", "target test có destination ngoài từ vựng, vì từ vựng thấy thêm các lần đặt đầu", "teal"),
        ],
    ),
)

# ================================================================== 11. final results: 3 seeds
PV, PD = SEEDS["per_variant"], SEEDS["paired_01d_minus_01c"]
rows_dp = []
for lab, key in (("Warm|query token", "warm/qt/ndcg@5"), ("Warm|+ sameDest", "warm/qt_sameDest/ndcg@5"), ("Tất cả sự kiện|+ sameDest", "all_events/qt_sameDest/ndcg@5"), ("Cold|query token", "cold/qt/ndcg@5")):
    a, b = PV["01c"][key], PV["01d"][key]
    rows_dp.append(
        (
            lab.replace("|", " · "),
            [("Không có dòng cold", a["mean"], a["std"], a["values"], GRAY), ("Có dòng cold", b["mean"], b["std"], b["values"], TEAL)],
            f"{sdec(PD[key]['mean'], 4)}",
        )
    )
slide(
    "5 · Kết quả cuối",
    "Huấn luyện có lần đặt đầu cao hơn ở cả ba seed, trên mọi tập dòng",
    lay(
        dotplot(rows_dp, 0.33, 0.45)
        + '<div class="legend"><span class="lg" style="--c:#94a3b8">Không có dòng cold</span><span class="lg" style="--c:#0f766e">Có dòng cold</span></div>',
        [
            (f"{min(PD[k]['seeds_where_01d_higher'] for k in ('warm/qt/ndcg@5', 'warm/qt_sameDest/ndcg@5', 'all_events/qt_sameDest/ndcg@5', 'cold/qt/ndcg@5'))}/3", "seed mà bản có dòng cold cao hơn, ở cả bốn chỉ số", "teal"),
            (f"{dec(PV['01c']['cold/qt/ndcg@5']['std'], 4)} → {dec(PV['01d']['cold/qt/ndcg@5']['std'], 4)}", "độ lệch chuẩn giữa các seed ở dòng cold giảm khi có dòng cold trong huấn luyện", ""),
            (f"{dec(PV['01c']['warm/sameDest_gain/ndcg@5']['mean'], 4)} / {dec(PV['01d']['warm/sameDest_gain/ndcg@5']['mean'], 4)}", "gain sameDest của hai bản gần như bằng nhau: không phụ thuộc cách huấn luyện", ""),
        ],
        "NDCG@5, tập test. Chấm: từng seed; vạch: trung bình ± độ lệch chuẩn; bên phải: chênh trung bình",
    )
    + take("Chênh lệch +0,005 đến +0,009 lớn hơn nhiễu giữa các seed; với 3 seed, độ lệch chuẩn chỉ để tham khảo."),
)

# ================================================================== 12. final vs baselines
w_items = [
    ("ItemKNN", nd(NEW_B[KNN][WARM_B]), LGRAY),
    ("Logistic regression", nd(NEW_B[LR][WARM_B]), LGRAY),
    ("SMLP4Rec + query token", PV["01d"]["warm/qt/ndcg@5"]["mean"], BLUE),
    ("+ sameDest", PV["01d"]["warm/qt_sameDest/ndcg@5"]["mean"], TEAL),
]
c_items = [
    ("ItemKNN", nd(NEW_B[KNN][COLD_B]), LGRAY),
    ("Logistic regression", nd(NEW_B[LR][COLD_B]), LGRAY),
    ("SMLP4Rec + query token", PV["01d"]["cold/qt/ndcg@5"]["mean"], BLUE),
    ("Prior theo destination", PV["01d"]["cold/prior_only/ndcg@5"]["mean"], GRAY),
]
wd, cd = PV["01d"]["warm/qt_minus_logreg/ndcg@5"], PV["01d"]["cold/qt_minus_logreg/ndcg@5"]
slide(
    "5 · Kết quả cuối",
    f"Warm: hơn logistic regression {sdec(wd['mean'], 4)} NDCG@5; cold: ngang ({sdec(cd['mean'], 4)})",
    '<div class="two"><div><div class="ct">Dòng warm (user đã có lịch sử)</div>'
    + hbar([a for a, _, _ in w_items], [v for _, v, _ in w_items], [col for _, _, col in w_items], 0.5, w=560, label_w=200, row=46)
    + '</div><div><div class="ct">Dòng cold (lần đặt đầu)</div>'
    + hbar([a for a, _, _ in c_items], [v for _, v, _ in c_items], [col for _, _, col in c_items], 0.5, w=560, label_w=200, row=46)
    + "</div></div>"
    + f'<p class="sm">NDCG@5, tập test; mô hình SMLP4Rec là trung bình 3 seed của bản có dòng cold. Query token trừ logistic regression: warm {sdec(wd["mean"], 4)} ± {dec(wd["std"], 4)}, cold {sdec(cd["mean"], 4)} ± {dec(cd["std"], 4)}.</p>'
    + take(f"Thêm sameDest, mô hình hơn logistic regression {sdec(PV['01d']['warm/qt_sameDest_minus_logreg/ndcg@5']['mean'], 4)} NDCG@5 ở dòng warm."),
)

# ================================================================== 13. conclusion
old_gap = nd(bench(NEW_Q, "Query token + sameDest")) - nd(bench(NEW_Q, "Hybrid"))  # without first bookings in training
new_gap = nd(bench(RUN1D, "Query token + sameDest")) - nd(bench(RUN1D, "Hybrid"))  # with first bookings in training
slide(
    "5 · Kết quả cuối",
    "Kết luận, giới hạn và bước tiếp",
    f"""<div class="two">
<div class="col"><h3>Đã chắc</h3><ul class="clean t lg2">
<li>Dữ liệu trùng làm phồng điểm các mô hình dùng lịch sử; sau làm sạch, thứ hạng giữ nguyên.</li>
<li>sameDest là tín hiệu thật dù hẹp (khoảng {sdec(sd_items[1][1], 3)} NDCG@5 sau khi bỏ đặt lại trong 1 giờ).</li>
<li>Huấn luyện có lần đặt đầu cải thiện ở 3/3 seed và đưa dòng cold về ngang logistic regression.</li>
<li>Prior ngoài mạng không còn cần: khoảng cách NDCG@5 với hybrid cũ từ {sdec(old_gap, 4)} xuống {sdec(new_gap, 4)} (seed 2022).</li></ul></div>
<div class="col"><h3>Giới hạn và việc tiếp theo</h3><ul class="clean t lg2">
<li>Chưa tuning: 3 epoch, chưa có lưới siêu tham số, cả sáu lần chạy đều chọn epoch 3.</li>
<li>Cluster chỉ là proxy cho loại phòng; một tập dữ liệu; chỉ offline; chỉ hai baseline.</li>
<li>Lặp cùng phiên ({num(R['flag_session_repeat'])} dòng) chưa quyết collapse hay không.</li>
<li>Còn lại: lưới siêu tham số, lát cắt theo độ dài lịch sử và trip context, chỉ số đa dạng và thiên lệch phổ biến.</li></ul></div></div>""",
)

# ================================================================== appendix A1: prior
slide(
    "Phụ lục · Công thức",
    "Prior theo destination (dùng trong hybrid cũ và làm mốc đối chiếu)",
    """<div class="eq">p<sub>prior</sub>(k | d) = ( n(d, k) + m · p<sub>mkt</sub>(k | M(d)) ) / ( n(d) + m ),  m = 5
<small>n(d, k): số booking cluster k tại destination d; n(d) = Σ<sub>k</sub> n(d, k); M(d): thị trường phổ biến nhất của d; p<sub>mkt</sub>: phân phối cluster của thị trường đó</small></div>
<ul class="clean t">
<li><b>Chỉ dùng quá khứ:</b> các số đếm lấy từ những booking nằm trước booking cuối cùng của tập train; không dùng valid hay test.</li>
<li><b>Làm mượt theo thị trường:</b> chỉ có tác dụng với destination ít dữ liệu (n(d) nhỏ); với destination nhiều dữ liệu, p<sub>prior</sub> gần bằng tần suất thực.</li>
<li><b>Destination chưa từng thấy:</b> dùng phân phối cluster toàn cục; khi lấy log có sàn 10<sup>−6</sup>.</li>
<li><b>Ghép với mô hình (hybrid cũ):</b> điểm = log p<sub>model</sub> + w · log p<sub>prior</sub>, w chọn trên valid. Mô hình chính không dùng số hạng này.</li></ul>""",
    sub="Kết quả trên dòng cold: prior đạt Recall@5 0,5129; query token 0,5029; logistic regression 0,5096",
)

# ================================================================== appendix A2: sameDest
tun = RUN1D["samedest_hybrid"]["tuned_on_valid"]["A"]
ages = [(0, 12, "q"), (1, 40, "khác"), (3, 12, "q"), (4, 7, "q")]
s12 = sum(0.7**a for a, k, d in ages if k == 12 and d == "q")
s7 = sum(0.7**a for a, k, d in ages if k == 7 and d == "q")
tot = s12 + s7
slide(
    "Phụ lục · Công thức",
    "sameDest: điểm cộng cho cụm đã đặt tại đúng destination đang tìm",
    f"""<div class="two"><div>
<div class="eq">s(k) = Σ<sub>j: dest<sub>j</sub> = q, cluster<sub>j</sub> = k</sub> 0,7<sup>tuổi<sub>j</sub></sup><br>p<sub>s</sub>(k) = s(k) / Σ<sub>k'</sub> s(k')
<small>q: destination đang tìm; tuổi = 0 cho booking gần nhất; không có booking nào tại q thì p<sub>s</sub> đều, hằng số, không đổi thứ hạng</small></div>
<div class="eq" style="margin-top:14px">điểm(k) = log q<sub>model</sub>(k) + w<sub>s</sub> · log q<sub>s</sub>(k)<br>log q(k) = log( (1 − α) · p(k) + α / 100 )
<small>mọi thành phần cùng thang log bị chặn, nên w<sub>s</sub> đọc như độ tin tưởng tương đối</small></div></div>
<div><div class="ct">Ví dụ: lịch sử có 4 booking, lượt tìm tại destination q</div>
{table(["Tuổi", "Cluster", "Destination", "Đóng góp"], [[a, k, "q" if d == "q" else "khác", dec(0.7**a, 3) if d == "q" else "0"] for a, k, d in ages], right_from=3)}
<p class="sm" style="margin-top:10px">s(12) = {dec(s12, 3)}, s(7) = {dec(s7, 3)} → p<sub>s</sub>(12) = {dec(s12 / tot, 3)}, p<sub>s</sub>(7) = {dec(s7 / tot, 3)}; cluster 40 không được cộng vì đặt ở destination khác.</p>
<p class="sm" style="margin-top:12px"><b>Chọn tham số trên valid</b> (quy tắc một sai số chuẩn, SE gom theo user): α ∈ {{0,05; 0,1; 0,2; 0,3; 0,5}}, w<sub>s</sub> ∈ [0; 3]. Hai ô: destination mới (w<sub>s</sub> = {dec(tun["weights"]["0"][1], 2)}) và đã biết (w<sub>s</sub> = {dec(tun["weights"]["1"][1], 2)} ở seed 2022, α = {dec(tun["alpha"], 1)}). Mô hình chính không có trọng số prior.</p></div></div>""",
)

# ================================================================== appendix A3: query-token features
vs = {k: v["vocab_size"] for k, v in RUN1D["query_coverage"].items()}
MINC = {"dest": 5, "country": 5}
spec = [
    ("dest", "Destination đang tìm", "mã destination", "≥ 5"),
    ("dest_type", "Loại destination", "mã loại", "≥ 1"),
    ("ci_month", "Tháng check-in", "1–12; 0 nếu thiếu", "≥ 1"),
    ("lead", "Lead time", "10 nhóm theo số ngày tới check-in; 0 nếu thiếu hoặc âm", "≥ 1"),
    ("stay", "Số đêm lưu trú", "cắt tại 14; 0 nếu thiếu hoặc âm", "≥ 1"),
    ("adults", "Người lớn", "cắt tại 9; 0 nếu bằng 0", "≥ 1"),
    ("children", "Trẻ em", "cắt tại 9", "≥ 1"),
    ("rooms", "Số phòng", "cắt tại 8; 0 nếu bằng 0", "≥ 1"),
    ("package", "Gói", "0 / 1", "≥ 1"),
    ("mobile", "Thiết bị di động", "0 / 1", "≥ 1"),
    ("channel", "Kênh", "mã kênh", "≥ 1"),
    ("site", "Site", "mã site", "≥ 1"),
    ("posa", "Châu lục điểm bán", "mã châu lục", "≥ 1"),
    ("country", "Quốc gia của user", "mã quốc gia", "≥ 5"),
]
slide(
    "Phụ lục · Feature engineering",
    "Query token: 14 trường của lượt tìm, mã hóa thành chỉ số embedding",
    f"""<div class="row" style="gap:30px"><div style="flex:1.7">{
        table(
            ["Trường", "Cách mã hóa", "Giá trị trong từ vựng", "Xuất hiện tối thiểu"],
            [[lab, enc, num(vs[f]), mn] for f, lab, enc, mn in spec],
            right_from=2,
            tight=True,
        )
    }</div>
<div style="flex:1;display:flex;flex-direction:column;gap:12px;justify-content:center"><ul class="clean t">
<li><b>Cách đưa vào mô hình:</b> vector [MASK] học được cộng tổng 14 embedding, đặt ở vị trí cuối sau 20 vị trí lịch sử.</li>
<li><b>Lead time:</b> số ngày từ ngày tìm đến check-in, chia theo mốc 0 · 1–2 · 3–6 · 7–13 · 14–29 · 30–59 · 60–89 · 90–179 · 180–364 · 365+.</li>
<li><b>Giá trị lỗi</b> thành mã “không rõ” (0), không bỏ dòng.</li>
<li><b>Từ vựng chỉ lấy từ target train</b> (và lần đặt đầu của tập train); giá trị ngoài từ vựng dùng chung một embedding.</li>
<li><b>Không dùng:</b> mọi cột <code>hotel_*</code>, khoảng cách tới khách sạn, <code>cnt</code>, vùng và thành phố của user (thành phố cho thông tin âm trên valid vì là proxy của user).</li></ul></div></div>""",
    sub="Từ vựng của bản có dòng cold; tổng khoảng 1,05 triệu tham số, phần lớn ở bảng embedding của query",
)


def render() -> None:
    parts, total = [], len(slides)
    for n, (meta, title, sub, body) in enumerate(slides, 1):
        if meta == "cover":
            parts.append(f'<section class="slide">{body}<div class="ft"><span>Topic C1 · Next-Best-Product Recommendation</span><span>{n} / {total}</span></div></section>')
            continue
        subh = f'<div class="sub">{sub}</div>' if sub else ""
        parts.append(
            f'<section class="slide"><div class="hd"><div><div class="meta">{meta}</div><h2>{title}</h2>{subh}</div>'
            f'<div class="pg">Tuần 4</div></div><div class="body">{body}</div>'
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ results/week4_rebuild</span><span>{n} / {total}</span></div></section>'
        )
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tuần 4: làm sạch Expedia và chạy lại SMLP4Rec</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="deck">{"".join(parts)}</div><script>{JS}</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    render()
