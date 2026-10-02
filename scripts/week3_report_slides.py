"""Build the week-3 report deck (Vietnamese, HTML) in the visual style of the week-2 slides.

Sections: (1) design recap from week 2 (SMLP4Rec architecture + hybrid flow slides, taken as-is),
(2) from the raw SMLP4Rec code to the Expedia adaptation, (3) plain run results, (4) the
experiments and their findings, (5) conclusion. CSS, JS and the architecture diagram are read
from scripts/week2_methodology_slides.py so the style stays identical.

Every number on the slides is read from persisted outputs in reports/summary/week3/:
smlprec_expedia_run.json (plain run), smlprec_expedia_mixed_run.json (fixed quota),
smlprec_expedia_dynamic_cap_run.json (dynamic cap), smlprec_expedia_late_fusion.json (prior
fusion + cold users), smlprec_expedia_hybrid_samedest.json (hybrid with sameDest + bootstrap).

Usage: python scripts/week3_report_slides.py
Output: reports/summary/week3/week3_report_slides.html (PDF: print with headless Chrome)
"""

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W3 = ROOT / "reports" / "summary" / "week3"
OUT = W3 / "week3_report_slides.html"
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
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%d/%m/%Y")


def table(head, rows, hl=(), cls="dense", right_from=1) -> str:
    th = "".join(
        f'<th class="{"r" if i >= right_from else ""}">{h}</th>'
        for i, h in enumerate(head)
    )
    trs = ""
    for ri, r in enumerate(rows):
        c = ' class="hl"' if ri in hl else ""
        trs += (
            f"<tr{c}>"
            + "".join(
                f'<td class="{"r" if i >= right_from else ""}">{x}</td>'
                for i, x in enumerate(r)
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
        out.append(
            f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
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
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" class="cv">{pct(v, 1)}</text>'
            )
        out.append(
            f'<text x="{x0 + (len(series) * bw + 6 * (len(series) - 1)) / 2:.1f}" y="{h - 8}" class="cl">{g}</text>'
        )
    out.append("</svg>")
    legend = "".join(
        f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series
    )
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
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>'
        )
        for i, v in enumerate(vals):
            out.append(
                f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}"/>'
            )
        bi = max(range(len(vals)), key=lambda k: vals[k])
        out.append(
            f'<text x="{X(bi):.1f}" y="{Y(vals[bi]) - 9:.1f}" class="cv">{pct(vals[bi], 1)}</text>'
        )
    out.append("</svg>")
    legend = "".join(
        f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series
    )
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
slide_cover = (
    "cover",
    "",
    "",
    """<div class="cover">
  <div class="kicker">Tuần 3 · Triển khai &amp; thử nghiệm</div>
  <h1>Triển khai code gốc SMLP4Rec trên Expedia<br><span>và các thử nghiệm bổ sung</span></h1>
  <p class="lead">Next-Best-Product Recommendation · Vinpearl × GSM · Bối cảnh (a): du lịch</p>
  <div class="cover-tags"><span>Code gốc SMLP4Rec</span><span>Dữ liệu Expedia</span><span>Thử nghiệm bổ sung</span></div>
</div>""",
)
slides.append(slide_cover)

slide(
    "Nội dung",
    "Nội dung báo cáo",
    """<div class="agenda">
  <div class="ag"><span>1</span><div><h3>Nhắc lại thiết kế (tuần 2)</h3><p>Kiến trúc SMLP4Rec đã chọn</p></div></div>
  <div class="ag"><span>2</span><div><h3>Từ code gốc đến bản chạy với Expedia</h3><p>Những gì đã đổi, giữ nguyên, và luồng pipeline</p></div></div>
  <div class="ag"><span>3</span><div><h3>Kết quả chạy plain</h3><p>SMLP4Rec chỉ dùng lịch sử, 3 epoch: điểm số, từng epoch, top 5 gồm gì</p></div></div>
  <div class="ag"><span>4</span><div><h3>Các cách thử nghiệm và phát hiện chính</h3><p>Ép tỉ lệ cũ/mới, fusion với prior điểm đến, người dùng mới, hybrid có sameDest</p></div></div>
  <div class="ag"><span>5</span><div><h3>Kết luận và bước tiếp theo</h3><p>Quyết định, hạn chế, kế hoạch</p></div></div>
</div>""",
)

# ---- section 1: recap, taken from the week-2 deck
slide(
    "1 · Nhắc lại thiết kế (tuần 2)",
    "Chọn SMLP4Rec: ý tưởng và kiến trúc",
    f"""
<div class="row">
  <div class="panel diagram-wrap">{architecture_svg()}</div>
  <div class="panel side steps">
    <div class="step"><span>1</span><p>Mỗi lượt dự đoán → khối <b>(S+1) × F × C</b>: chuỗi booking + query token của lượt search.</p></div>
    <div class="step"><span>2</span><p>Ba MLP <b>song song</b> trộn thông tin theo <b>thời gian</b>, <b>context</b> và <b>chiều embedding</b>, cộng lại, lặp L lần.</p></div>
    <div class="step"><span>3</span><p>Lấy hàng của query, so với 100 cluster → điểm xếp hạng.</p></div>
  </div>
</div>""",
    sub="Gao et al., ACM TOIS 2024 — slide giữ nguyên từ tuần 2",
)

# ---- section 2: from raw code to the Expedia adaptation
slide(
    "2 · Thích ứng với Expedia",
    "Từ code gốc đến bản chạy với Expedia: những gì đã đổi",
    table(
        [
            "Hạng mục",
            "Code gốc (fork SMLP4Rec trên RecBole)",
            "Bản thích ứng cho Expedia",
        ],
        [
            [
                "Dữ liệu",
                "MovieLens ML-100K; item có feature (thể loại, tên phim, năm)",
                "Booking Expedia: user, hotel cluster (100), thời gian; cluster không có feature",
            ],
            [
                "Độ dài chuỗi",
                "Cố định 50 trong code",
                "Lấy từ cấu hình <code>MAX_ITEM_LIST_LENGTH</code> (= 20)",
            ],
            [
                "Feature của item",
                "Bắt buộc có lớp embedding feature",
                "Cho phép danh sách feature rỗng, bỏ lớp feature",
            ],
            [
                "Chia dữ liệu và đánh giá",
                "Leave-one-out; xếp hạng với 100 mẫu theo popularity (pop100)",
                "Chia theo thời gian 80/10/10; xếp hạng đầy đủ trên 100 cụm",
            ],
            [
                "Cấu hình mô hình",
                "Cấu hình paper: nhiều lớp, dropout 0,5, quá chậm trên CPU",
                "2 lớp, hidden 64, dropout 0, lr 0,001 (nhẹ, chạy được trên CPU)",
            ],
            [
                "Môi trường",
                "RecBole 1.0.1 cần numpy 1.23 / pandas 1.5; torch lỗi DLL khi đường dẫn dài",
                "Venv Python 3.11 riêng ở đường dẫn ngắn; hai bản vá nhỏ cho RecBole",
            ],
        ],
        right_from=9,
    )
    + '<div class="note">Mọi chỉnh sửa trong mã mô hình được đánh dấu <code>ADAPTED</code> và chỉ có hai điểm; phần còn lại giữ nguyên bản gốc.</div>',
    sub="Chỉnh sửa tối thiểu: thích ứng để chạy được, chưa tối ưu",
)

slide(
    "2 · Thích ứng với Expedia",
    "Phần giữ nguyên từ bản gốc (đã biết là hạn chế)",
    """<div class="grid2">
  <div class="card"><h3>Một khối dùng lại cho mọi lớp</h3><p>Chỉ có một khối SMLP, lặp <code>n_layers</code> lần với trọng số chung. Tham số <code>n_layers</code> không thêm lớp thật.</p></div>
  <div class="card"><h3>Không có kết nối residual</h3><p>Đầu ra mỗi lớp thay thế đầu vào thay vì cộng vào, nên xếp nhiều lớp khó huấn luyện hơn.</p></div>
  <div class="card"><h3>Mô hình không thấy ngữ cảnh tìm kiếm</h3><p>Đầu vào chỉ là id cluster đã đặt; không có điểm đến, ngày, số người của lượt search hiện tại.</p></div>
</div>
<div class="note warn">Các hạn chế này nằm trong kế hoạch sửa (trọng số riêng mỗi lớp, residual, query token) ở bước dựng lại pipeline, không thuộc lần chạy kiểm tra này.</div>""",
)

flow_steps = [
    ("in", "train.csv", "4 GB, đọc theo khối"),
    ("in", "Lọc is_booking = 1", "giữ booking, bỏ click"),
    ("prior", "expedia.inter", "user · cluster · giờ"),
    ("gate", "RecBole", "chuỗi + chia theo thời gian"),
    ("model", "SMLP4Rec", "CE trên 100 cluster"),
    ("out", "Đánh giá", "xếp hạng đầy đủ"),
]
flow_html = '<div class="farrow">→</div>'.join(
    f'<div class="fbox {k}">{a}<small>{b}</small></div>' for k, a, b in flow_steps
)
slide(
    "2 · Thích ứng với Expedia",
    "Luồng pipeline và thiết lập",
    f"""<div class="flow six">{flow_html}</div>
<div class="kpis">
  <div class="kpi"><b>{num(DS["users"])}</b><span>người dùng</span></div>
  <div class="kpi"><b>{DS["items"]}</b><span>cụm khách sạn (item)</span></div>
  <div class="kpi"><b>{num(DS["sequence_targets"])}</b><span>mục tiêu “đặt tiếp theo”</span></div>
  <div class="kpi"><b>{num(RUN["n_parameters"])}</b><span>tham số của mô hình</span></div>
</div>
<div class="note">Mô hình: {CFG["n_layers"]} lớp, hidden {CFG["hidden_size"]}, dropout {CFG["hidden_dropout_prob"]}, lr {CFG["learning_rate"]}, batch {CFG["train_batch_size"]}, {CFG["epochs"]} epoch, lịch sử tối đa {CFG["MAX_ITEM_LIST_LENGTH"]} lần đặt, seed {CFG["seed"]}. Thời gian: dựng dữ liệu {RUN["timing_seconds"]["data_build"]:.0f}s, huấn luyện + đánh giá từng epoch {RUN["timing_seconds"]["train_and_valid"]:.0f}s.</div>""",
    sub="Một lần chạy end-to-end trên CPU; có sửa lỗi đơn vị thời gian (pandas 3 đọc ngày ra micro giây)",
)

slide(
    "2 · Thích ứng với Expedia",
    "Cách chia dữ liệu",
    table(
        ["Tập", "Số mục tiêu", "Từ ngày", "Đến ngày", "Lịch sử trung bình"],
        [
            [
                n,
                num(SP[k]["targets"]),
                day(SP[k]["first_target_unix"]),
                day(SP[k]["last_target_unix"]),
                dec(SP[k]["mean_history_len"], 2),
            ]
            for k, n in (("train", "Train"), ("valid", "Valid"), ("test", "Test"))
        ],
    )
    + """<div class="grid3 tight">
  <div class="card"><h3>Theo thời gian</h3><p>Chia 80/10/10 theo thứ tự thời gian của các mục tiêu, không chia ngẫu nhiên, để mùa vụ không rò rỉ.</p></div>
  <div class="card"><h3>Mục tiêu là gì</h3><p>Mọi lần đặt phòng trừ lần đặt đầu của mỗi người dùng; lịch sử là các lần đặt trước đó của người đó.</p></div>
  <div class="card"><h3>Chưa có người dùng mới</h3><p>Lần đặt đầu không có lịch sử nên không phải mục tiêu; người dùng mới được đánh giá riêng ở phần 4.</p></div>
</div>""",
    sub="Xếp hạng đầy đủ trên 100 cụm, không lấy mẫu âm",
)

# ---- section 3: plain run
slide(
    "3 · Kết quả plain",
    "Kết quả plain: SMLP4Rec chỉ dùng lịch sử (test)",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Recall@K trên tập test · cùng các dòng cho cả hai phương pháp</div>
    {
        grouped_bars(
            [
                (
                    "Popularity toàn cục",
                    [POP[f"recall@{k}"] for k in (5, 10, 20)],
                    GRAY,
                ),
                (
                    f"SMLP4Rec (epoch {RUN['best_epoch_by_valid']})",
                    [T[f"recall@{k}"] for k in (5, 10, 20)],
                    BLUE,
                ),
            ],
            ["Recall@5", "Recall@10", "Recall@20"],
        )
    }</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">SMLP4Rec, test</div><div class="val">Recall@5 {
        pct(T["recall@5"])
    }</div><p>NDCG@10 {dec(T["ndcg@10"])} · MRR@10 {dec(T["mrr@10"])}</p></div>
  </div>
</div>
<div class="note">Lịch sử đặt phòng tự nó có tín hiệu, nhưng điểm vẫn thấp (Recall@5 {
        pct(T["recall@5"], 1)
    }) vì mô hình chưa biết điểm đến của lần tìm kiếm.</div>""",
    sub="Xếp hạng đầy đủ trên 100 cụm · người dùng có ít nhất một booking trước đó",
)

slide(
    "3 · Kết quả plain",
    "Kết quả sau từng epoch: thêm epoch gần như không giúp",
    table(
        [
            "Epoch",
            "Loss train",
            "Valid Recall@5",
            "Test Recall@5",
            "Test Recall@10",
            "Test NDCG@10",
        ],
        [
            [
                str(e["epoch"]),
                num(round(e["train_loss"])),
                pct(e["valid"]["recall@5"]),
                pct(e["test"]["recall@5"]),
                pct(e["test"]["recall@10"]),
                dec(e["test"]["ndcg@10"]),
            ]
            for e in PE
        ],
        hl=(len(PE) - 1,),
    )
    + f"""<div class="kpis three">
  <div class="kpi"><b>{pp(PE[-1]["test"]["recall@5"] - PE[0]["test"]["recall@5"])}</b><span>Recall@5 test, epoch {len(PE)} so với epoch 1</span></div>
  <div class="kpi"><b>{pct((PE[0]["train_loss"] - PE[-1]["train_loss"]) / PE[0]["train_loss"], 1)}</b><span>loss train giảm sau {len(PE) - 1} epoch thêm</span></div>
  <div class="kpi"><b>{RUN["best_epoch_by_valid"]}/{len(PE)}</b><span>epoch tốt nhất theo valid</span></div>
</div>
<div class="note">Trần điểm nằm ở <b>thông tin đầu vào</b> (chỉ có lịch sử, không có điểm đến) và ở kiến trúc nhỏ, không nằm ở số epoch huấn luyện.</div>""",
    sub="Valid và test được tính lại sau mỗi epoch; epoch tốt nhất chọn theo valid",
)

tb_t, tb_v = TB["test"], TB["valid"]
slide(
    "3 · Kết quả plain",
    "Plain chủ yếu gợi lại cụm đã đặt",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Người dùng đặt cụm nào tiếp theo (tỉ lệ dòng mục tiêu)</div>
    {table(["Đặt…", "Train", "Valid", "Test"], [["Cụm vừa đặt gần nhất"] + [pct(MX[k]["last"]) for k in ("train", "valid", "test")], ["Cụm cũ, không phải gần nhất"] + [pct(MX[k]["older_not_last"]) for k in ("train", "valid", "test")], ["<b>Cụm cũ (gồm gần nhất)</b>"] + [pct(MX[k]["old_incl_last"]) for k in ("train", "valid", "test")], ["<b>Cụm mới (chưa từng đặt)</b>"] + [pct(MX[k]["new"]) for k in ("train", "valid", "test")]])}</div>
  <div class="panel side2">
    <div class="wk"><b>{pct(tb_t["top1_is_last_booking"], 1)}</b><p>dòng test có <b>top 1 = cụm đặt gần nhất</b></p></div>
    <div class="wk"><b>{pct(tb_t["old_target_in_top5_of_old_rows"], 1)}</b><p>trúng trong top 5 khi người dùng đặt lại <b>cụm cũ</b></p></div>
    <div class="wk"><b>{pct(tb_t["new_target_in_top5_of_new_rows"], 1)}</b><p>trúng khi đặt <b>cụm mới</b>, thấp hơn popularity top 5 ({pct(tb_t["global_popularity_top5_on_new_rows"], 1)})</p></div>
  </div>
</div>
<div class="note warn">Khoảng {pct(MX["test"]["new"], 0)} lần đặt tiếp theo là cụm mới: chỉ nhìn lịch sử không thể đoán được các trường hợp này. {pct(tb_t["old_target_in_top5_of_all_rows"] / tb_t["recall@5_check"], 1)} số lần trúng của mô hình là cụm cũ.</div>""",
    sub="Trọng số epoch tốt nhất · tính trên toàn bộ dòng test",
)

# ---- section 4: experiments
mp, mm = MIXT["plain"], MIXT["mixed"]
cp, cc = CAPT["plain"], CAPT["capped"]


def hits(d):
    return pct(d["old_target_hit_of_all_rows"]), pct(d["new_target_hit_of_all_rows"])


exch_fixed = (mp["old_target_hit_of_all_rows"] - mm["old_target_hit_of_all_rows"]) / (
    mm["new_target_hit_of_all_rows"] - mp["new_target_hit_of_all_rows"]
)
exch_cap = (cp["old_target_hit_of_all_rows"] - cc["old_target_hit_of_all_rows"]) / (
    cc["new_target_hit_of_all_rows"] - cp["new_target_hit_of_all_rows"]
)
slide(
    "4 · Thử nghiệm",
    "Thử 1–2: ép tỉ lệ cụm cũ / cụm mới trong top 5",
    table(
        [
            "Cách",
            "Recall@5",
            "MAP@5",
            "Cụm cũ trúng",
            "Cụm mới trúng",
            "Slot cụm cũ TB",
        ],
        [
            [
                "Plain: 5 điểm cao nhất",
                pct(mp["recall@5"]),
                dec(mp["mrr@5"]),
                *hits(mp),
                dec(mp["avg_past_slots_in_top5"], 2),
            ],
            [
                "Cố định 2 cũ + 3 mới (1 cũ + 4 mới nếu chỉ có 1 cụm cũ)",
                pct(mm["recall@5"]),
                dec(mm["mrr@5"]),
                *hits(mm),
                dec(mm["avg_past_slots_in_top5"], 2),
            ],
            [
                "Giới hạn động: tối đa 1 cụm cũ nếu L &lt; 5, tối đa 2 nếu L ≥ 5",
                pct(cc["recall@5"]),
                dec(cc["mrr@5"]),
                *hits(cc),
                dec(cc["avg_past_slots_in_top5"], 2),
            ],
        ],
        hl=(0,),
    )
    + f"""<div class="grid2">
  <div class="card good"><h3>Cả hai đều làm giảm độ chính xác</h3><p>Recall@5 giảm {dec((mp["recall@5"] - mm["recall@5"]) * 100, 2)} và {dec((mp["recall@5"] - cc["recall@5"]) * 100, 2)} điểm, đổi lại cụm mới trúng nhiều hơn.</p></div>
  <div class="card good"><h3>Giá của việc “khám phá”</h3><p>Mỗi cụm mới trúng thêm làm mất khoảng <b>{dec(exch_fixed, 1)}</b> (cố định) và <b>{dec(exch_cap, 1)}</b> (giới hạn động) cụm cũ. Gợi ý mới của mô hình yếu hơn nhiều so với gợi ý lại.</p></div>
</div>
<div class="note">L = số booking trước đó của người dùng. Quy tắc chỉ thay cách <b>chọn</b> top 5 sau khi chấm điểm; trọng số mô hình không đổi (kết quả khớp lần chạy plain từng epoch).</div>""",
    sub="Test, epoch tốt nhất theo valid · “cụm cũ/mới trúng” là % trên toàn bộ dòng",
)

wgrid = [str(w) for w in LF["w_grid"]]
slide(
    "4 · Thử nghiệm",
    "Thử 3: kết hợp SMLP4Rec với prior điểm đến (late fusion)",
    f"""<div class="eq">điểm(cluster) = log p<sub>SMLP4Rec</sub> + w × log p<sub>prior</sub>(cluster | điểm đến tìm kiếm)</div>
<div class="row">
  <div class="panel grow"><div class="ph">Recall@5 theo trọng số w của prior (w = 0 là plain)</div>
    {line_chart(wgrid, [("Valid", [LF["sweep"]["valid"][w]["all"]["recall@5"] for w in wgrid], BLUE), ("Test", [LF["sweep"]["test"][w]["all"]["recall@5"] for w in wgrid], TEAL)], 0.30, 0.65)}</div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Prior điểm đến</div><p>Đếm cluster theo điểm đến, làm mượt về market (m = 5), điểm đến chưa thấy dùng global. Chỉ dùng booking trước kỳ đánh giá.</p></div>
    <div class="th-card"><div class="lbl">Trọng số tốt nhất (valid)</div><div class="val">w = {LFB}</div><p>Đường cong mượt, một đỉnh.</p></div>
  </div>
</div>""",
    sub="Không huấn luyện lại; w chọn trên valid, đọc test một lần",
)

vf, vp0 = LF["variants"]["test"], None
fkey = f"fusion_global_w={LFB}"
slide(
    "4 · Thử nghiệm",
    "Thử 3: prior điểm đến là đòn bẩy lớn nhất",
    table(
        [
            "Cách (test, người dùng có lịch sử)",
            "Recall@5",
            "Recall@10",
            "NDCG@10",
            "MAP@5",
        ],
        [
            [
                "Plain SMLP4Rec",
                *(
                    pct(HW[K_PLAIN]["all"]["recall@5"]),
                    pct(HW[K_PLAIN]["all"]["recall@10"]),
                    dec(HW[K_PLAIN]["all"]["ndcg@10"]),
                    dec(HW[K_PLAIN]["all"]["map@5"]),
                ),
            ],
            [
                "Prior điểm đến một mình",
                *(
                    pct(HW[K_PRIOR]["all"]["recall@5"]),
                    pct(HW[K_PRIOR]["all"]["recall@10"]),
                    dec(HW[K_PRIOR]["all"]["ndcg@10"]),
                    dec(HW[K_PRIOR]["all"]["map@5"]),
                ),
            ],
            [
                f"SMLP4Rec + prior (w = {LFB})",
                *(
                    pct(HW[K_FUSION]["all"]["recall@5"]),
                    pct(HW[K_FUSION]["all"]["recall@10"]),
                    dec(HW[K_FUSION]["all"]["ndcg@10"]),
                    dec(HW[K_FUSION]["all"]["map@5"]),
                ),
            ],
        ],
        hl=(2,),
    )
    + f"""<div class="kpis three">
  <div class="kpi"><b>{pp(HW[K_FUSION]["all"]["recall@5"] - HW[K_PLAIN]["all"]["recall@5"], 1)}</b><span>Recall@5 so với plain</span></div>
  <div class="kpi"><b>{pct(HW[K_PLAIN]["new_target_rows"]["recall@5"], 1)} → {pct(LF["variants"]["test"][fkey]["new_target_rows"]["recall@5"], 1)}</b><span>Recall@5 trên dòng đặt cụm mới</span></div>
  <div class="kpi"><b>{pct(LF["variants"]["test"][fkey]["old_target_rows"]["recall@5"], 1)}</b><span>trên dòng đặt lại cụm cũ (plain {pct(HW[K_PLAIN]["old_target_rows"]["recall@5"], 1)}): giữ nguyên</span></div>
</div>
<div class="note">Gần như toàn bộ mức tăng nằm ở nhóm cụm mới. Prior một mình ({pct(HW[K_PRIOR]["all"]["recall@5"], 1)}) đã tái hiện mốc tuần 2 (53,07%); SMLP4Rec cộng thêm {pp(HW[K_FUSION]["all"]["recall@5"] - HW[K_PRIOR]["all"]["recall@5"], 1)}.</div>""",
)

cold_t, cold_v = (
    LF["cold_users_prior_only"]["test"],
    LF["cold_users_prior_only"]["valid"],
)
sup = cold_t["prior_only_by_destination_support"]
slide(
    "4 · Thử nghiệm",
    "Người dùng mới (chưa có lịch sử): 100% prior, không dùng SMLP4Rec",
    f"""<div class="row">
  <div class="panel grow"><div class="ph">Lần đặt đầu tiên của người dùng, chỉ xếp hạng bằng prior điểm đến</div>
    {table(["Tập", "Số dòng", "Recall@5", "Recall@10", "Recall@20", "Popularity toàn cục @5"], [["Valid", num(cold_v["n_cold_rows"]), pct(cold_v["prior_only"]["recall@5"]), pct(cold_v["prior_only"]["recall@10"]), pct(cold_v["prior_only"]["recall@20"]), pct(cold_v["global_popularity"]["recall@5"])], ["Test", num(cold_t["n_cold_rows"]), pct(cold_t["prior_only"]["recall@5"]), pct(cold_t["prior_only"]["recall@10"]), pct(cold_t["prior_only"]["recall@20"]), pct(cold_t["global_popularity"]["recall@5"])]], hl=(1,))}</div>
  <div class="panel side2">
    <div class="th-card"><div class="lbl">Test theo độ phủ của điểm đến trong prior</div>
      {table(["Số booking trước", "Tỉ lệ", "Recall@5"], [["Chưa thấy (0)", pct(sup["n_dest=0"]["share"], 1), pct(sup["n_dest=0"]["recall@5"], 1)], ["1–19", pct(sup["n_dest 1-19"]["share"], 1), pct(sup["n_dest 1-19"]["recall@5"], 1)], ["≥ 20", pct(sup["n_dest>=20"]["share"], 1), pct(sup["n_dest>=20"]["recall@5"], 1)]], cls="dense")}</div>
  </div>
</div>
<div class="note">Người dùng mới chiếm khoảng {pct(HY["cold_rows"]["test"] / (HY["cold_rows"]["test"] + HY["rows"]["test"]), 0)} sự kiện test. Điểm của họ phụ thuộc hoàn toàn vào prior: cải thiện cần ngữ cảnh tìm kiếm hoặc prior tốt hơn, không phải mô hình tuần tự.</div>""",
    sub="Cửa sổ thời gian valid/test giống các dòng có lịch sử; prior chỉ tính từ booking trước kỳ đánh giá",
)

slide(
    "4 · Thử nghiệm",
    "Thử 4: thêm tín hiệu sameDest vào hybrid",
    f"""<div class="eq">điểm = 1·log p<sub>SMLP4Rec</sub> + w<sub>p</sub>·log p<sub>prior</sub> + w<sub>s</sub>·sameDest</div>
<div class="row">
  <div class="panel grow">
    <div class="ph">sameDest(cluster) = tổng 0,7<sup>tuổi</sup> của các lần đặt cluster đó <b>tại đúng điểm đến đang tìm</b></div>
    {table(["Lịch sử (cũ → mới)", "Cluster", "Điểm đến", "Tuổi", "0,7^tuổi"], [["Booking 1", "12", "Cancún", "3", dec(0.7**3, 3)], ["Booking 2", "45", "Miami", "2", "bỏ (khác điểm đến)"], ["Booking 3", "3", "Cancún", "1", dec(0.7, 3)], ["Booking 4", "12", "Cancún", "0", dec(1.0, 3)]], cls="dense ex")}
    <p class="small" style="margin-top:8px">Truy vấn: Cancún. sameDest(12) = {dec(0.7**3, 3)} + 1 = <b>{dec(EX12, 3)}</b>, sameDest(3) = <b>{dec(EX3, 3)}</b>, sameDest(45) = 0.</p>
  </div>
  <div class="panel side">
    <div class="th-card"><div class="lbl">Vì sao cần</div><p>SMLP4Rec chỉ thấy id cluster, không biết cluster đó đặt ở đâu. sameDest cho nó biết “đặt lại ở nơi đã ở”.</p></div>
    <div class="th-card"><div class="lbl">Trọng số tốt nhất (valid)</div><div class="val">w<sub>p</sub> {dec(min(v[0] for v in CW.values()), 1)}–{dec(max(v[0] for v in CW.values()), 1)} · w<sub>s</sub> = {dec(max(v[1] for v in CW.values()), 0)}</div><p>Theo từng nhóm số booking trước; các nhóm gần như giống nhau.</p></div>
  </div>
</div>""",
    sub="Mỗi tín hiệu là một số hạng; trọng số chọn trên valid, đọc test một lần",
)

slide(
    "4 · Thử nghiệm",
    "So sánh bốn cách (test)",
    table(
        [
            "Cách",
            "Recall@5",
            "Recall@10",
            "NDCG@10",
            "MAP@5",
            "Tất cả sự kiện R@5",
            "Tất cả sự kiện MAP@5",
        ],
        [
            [
                name,
                pct(HW[k]["all"]["recall@5"]),
                pct(HW[k]["all"]["recall@10"]),
                dec(HW[k]["all"]["ndcg@10"]),
                dec(HW[k]["all"]["map@5"]),
                pct(HA[k]["recall@5"]),
                dec(HA[k]["map@5"]),
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
    + """<div class="note">Cột 2–5: người dùng có lịch sử. “Tất cả sự kiện” gồm cả người dùng mới (prior; riêng plain dùng popularity toàn cục cho nhóm này). Hybrid cao nhất ở mọi cột.</div>""",
)

# ---- section 5: insights and conclusion
slide(
    "5 · Kết luận",
    "Các phát hiện chính",
    f"""<div class="grid3">
  <div class="card good"><h3>1 · Phần lớn đặt cụm mới</h3><p>{pct(MX["test"]["new"], 0)} lần đặt tiếp theo là cụm chưa từng đặt; chỉ {pct(MX["test"]["last"], 1)} đặt lại cụm cuối.</p></div>
  <div class="card good"><h3>2 · Điểm đến là tín hiệu chủ đạo</h3><p>Prior điểm đến đưa Recall@5 từ {pct(HW[K_PLAIN]["all"]["recall@5"], 0)} lên {pct(HW[K_FUSION]["all"]["recall@5"], 0)}, hầu hết ở nhóm cụm mới.</p></div>
  <div class="card good"><h3>3 · Ép tỉ lệ làm giảm điểm</h3><p>Gợi ý mới của mô hình yếu: mỗi hit mới tốn khoảng 2 hit cũ. Cần cải thiện cách chấm gợi ý mới, không phải hạn ngạch.</p></div>
  <div class="card good"><h3>4 · SMLP4Rec cộng thêm trên prior</h3><p>Trên cùng prior điểm đến, thêm SMLP4Rec tăng Recall@5 {pp(HW[K_FUSION]["all"]["recall@5"] - HW[K_PRIOR]["all"]["recall@5"], 1)}: lịch sử cá nhân có giá trị, nhưng nhỏ hơn prior.</p></div>
  <div class="card good"><h3>5 · Hybrid ba tín hiệu cao nhất</h3><p>SMLP4Rec + prior + sameDest đạt Recall@5 {pct(HW[K_HYBRID]["all"]["recall@5"], 1)} và MAP@5 {dec(HW[K_HYBRID]["all"]["map@5"])}, cao hơn fusion không có sameDest ({pct(HW[K_FUSION]["all"]["recall@5"], 1)} / {dec(HW[K_FUSION]["all"]["map@5"])}).</p></div>
  <div class="card good"><h3>6 · Người dùng mới nhờ prior</h3><p>Chiếm khoảng {pct(HY["cold_rows"]["test"] / (HY["cold_rows"]["test"] + HY["rows"]["test"]), 0)} sự kiện test, Recall@5 {pct(cold_t["prior_only"]["recall@5"], 1)}; chỉ prior mới giúp được họ.</p></div>
</div>""",
    sub="Tổng hợp từ lần chạy plain và các thử nghiệm",
)

slide(
    "5 · Kết luận",
    "Hạn chế cần nhớ",
    """<div class="grid2">
  <div class="card"><h3>Trọng số chạm mép lưới</h3><p>w<sub>s</sub> = 8 là giá trị lớn nhất đã thử, ở mọi nhóm. Với w<sub>s</sub> lớn, cụm cùng điểm đến gần như luôn xếp đầu, phần còn lại do mô hình và prior sắp xếp; cần mở rộng lưới trước khi coi trọng số là cuối cùng.</p></div>
  <div class="card"><h3>Đóng góp riêng của SMLP4Rec chưa chứng minh</h3><p>Chưa so sánh với mô hình đếm lịch sử (history-counts) trong cùng hybrid. Nếu hai bên ngang nhau, kiến trúc tuần tự không đem thêm giá trị.</p></div>
  <div class="card"><h3>Giới hạn của lần chạy</h3><p>Một seed, prior tĩnh (không cập nhật qua valid/test), 3 epoch chưa tinh chỉnh, bootstrap theo dòng.</p></div>
  <div class="card"><h3>Chia dữ liệu</h3><p>Cách chia 80/10/10 trên các mục tiêu khác tuần 2, nên so với mốc tuần 2 chỉ là gần đúng; kế hoạch dựng lại dùng chia theo sự kiện gồm cả người dùng mới.</p></div>
</div>""",
)

slide(
    "5 · Kết luận",
    "Kết luận và bước tiếp theo",
    f"""<div class="sum">
  <div class="sum-main"><div class="lbl">Quyết định hiện tại</div>
    <h3>SMLP4Rec + prior điểm đến + sameDest, trọng số theo nhóm</h3>
    <ul class="clean">
      <li>Prior vùng/điểm đến là <b>nền tảng độ chính xác</b>; SMLP4Rec là lớp cá nhân hóa trên đó.</li>
      <li>Hybrid đạt Recall@5 <b>{pct(HW[K_HYBRID]["all"]["recall@5"])}</b> / MAP@5 <b>{dec(HW[K_HYBRID]["all"]["map@5"])}</b> (test, có lịch sử), so với plain {pct(HW[K_PLAIN]["all"]["recall@5"])} / {dec(HW[K_PLAIN]["all"]["map@5"])}.</li>
      <li>Người dùng mới chỉ dùng prior; Recall@5 {pct(cold_t["prior_only"]["recall@5"], 1)}.</li>
    </ul></div>
  <div class="sum-side">
    <div class="th-card"><div class="lbl">Cổng quyết định</div><p>Nếu mô hình học không thắng mô hình đếm lịch sử sau khi có đủ ngữ cảnh, báo cáo kết luận như vậy và chuyển trọng tâm sang mô hình ngữ cảnh.</p></div>
  </div>
</div>
<div class="timeline">
  <div class="tl"><span>1</span><h3>Dữ liệu</h3><p>Làm sạch, chia theo sự kiện gồm người dùng mới</p></div>
  <div class="tl"><span>2</span><h3>Baseline</h3><p>Prior, ItemKNN và các baseline trên chia mới</p></div>
  <div class="tl"><span>3</span><h3>Dựng lại</h3><p>SMLP4Rec dạng module, PyTorch thuần</p></div>
  <div class="tl"><span>4</span><h3>Hybrid</h3><p>Mở rộng lưới, so với mô hình đếm lịch sử</p></div>
  <div class="tl"><span>5</span><h3>Ngữ cảnh</h3><p>Điểm đến từng booking, query token, seed, ablation</p></div>
</div>""",
    sub="Kế hoạch 10 ngày: docs/plan_oct_01_14.md",
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
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ reports/summary/week3/*.json</span><span>{n} / {total}</span></div></section>'
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
