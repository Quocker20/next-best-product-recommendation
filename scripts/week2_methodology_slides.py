"""Build the week-2 methodology-selection slide deck (Vietnamese, HTML).

Every number on the slides is read from persisted script outputs, not typed in:
- results/week1_dataset_selection/expedia_eda_detail.json   (scripts/expedia_eda_detail.py)
- results/week1_dataset_selection/expedia_context_signal.json (scripts/expedia_context_signal.py)
- results/week2_methodology/expedia_history_slices.json    (scripts/expedia_history_slices.py)

Usage: python scripts/week2_methodology_slides.py
Output: reports/summary/week2_methodology/methodology_slides.html
PDF: print the HTML with headless Chrome (see README note in the output folder).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT / "results" / "week1_dataset_selection"
W2 = ROOT / "results" / "week2_methodology"
OUT = (
    ROOT
    / "reports"
    / "summary"
    / "week2_methodology"
    / "methodology_slides.html"
)

EDA = json.loads((W1 / "expedia_eda_detail.json").read_text())
CTX = json.loads((W1 / "expedia_context_signal.json").read_text())
HS = json.loads((W2 / "expedia_history_slices.json").read_text())

NAVY, BLUE, TEAL, AMBER, RED, GRAY = (
    "#1e3a8a",
    "#2563eb",
    "#0f766e",
    "#b45309",
    "#dc2626",
    "#94a3b8",
)


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


# ---------------------------------------------------------------- charts (inline SVG)


def bar_chart(labels, values, fmt, w=560, h=250, colors=None, ymax=None, rotate=False):
    ymax = ymax or max(values) * 1.15
    left, right, top, bottom = 8, 8, 24, 44 if rotate else 30
    pw, ph = w - left - right, h - top - bottom
    step = pw / len(values)
    bw = step * 0.62
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="chart">',
        f'<line x1="{left}" y1="{top + ph}" x2="{w - right}" y2="{top + ph}" stroke="#cbd5e1"/>',
    ]
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = left + i * step + (step - bw) / 2
        bh = ph * v / ymax
        c = colors[i] if colors else BLUE
        out.append(
            f'<rect x="{x:.1f}" y="{top + ph - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{c}"/>'
        )
        if fmt:
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{top + ph - bh - 6:.1f}" class="cv">{fmt(v)}</text>'
            )
        cx = x + bw / 2
        if rotate:
            out.append(
                f'<text x="{cx:.1f}" y="{top + ph + 14}" class="cl" transform="rotate(-45 {cx:.1f} {top + ph + 14})" text-anchor="end">{lab}</text>'
            )
        else:
            out.append(
                f'<text x="{cx:.1f}" y="{top + ph + 18}" class="cl">{lab}</text>'
            )
    out.append("</svg>")
    return "".join(out)


def line_chart(
    labels, series, w=640, h=300, ymin=0.30, ymax=0.66, ticks=(0.35, 0.45, 0.55, 0.65)
):
    left, right, top, bottom = 44, 16, 16, 34
    pw, ph = w - left - right, h - top - bottom
    step = pw / (len(labels) - 1)

    def y(v):
        return top + ph * (1 - (v - ymin) / (ymax - ymin))

    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="chart">']
    for t in ticks:
        out.append(
            f'<line x1="{left}" y1="{y(t):.1f}" x2="{w - right}" y2="{y(t):.1f}" stroke="#eef2f7"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{y(t) + 4:.1f}" class="cl" text-anchor="end">{int(t * 100)}%</text>'
        )
    for i, lab in enumerate(labels):
        out.append(
            f'<text x="{left + i * step:.1f}" y="{h - 10}" class="cl">{lab}</text>'
        )
    for name, vals, color, dash, label_pos in series:
        pts = " ".join(f"{left + i * step:.1f},{y(v):.1f}" for i, v in enumerate(vals))
        da = ' stroke-dasharray="6 5"' if dash else ""
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"{da}/>'
        )
        for i, v in enumerate(vals):
            out.append(
                f'<circle cx="{left + i * step:.1f}" cy="{y(v):.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2.5"/>'
            )
        i = len(vals) - 1
        dy = {"above": -12, "below": 20}[label_pos]
        out.append(
            f'<text x="{left + i * step - 4:.1f}" y="{y(vals[-1]) + dy:.1f}" class="cv" text-anchor="end" fill="{color}">{pct(vals[-1], 1)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def architecture_svg():
    """SMLP4Rec adapted to Expedia (query market derived from train, same-dest flag on history)."""

    def row(y, text, kind):
        fill, stroke, col = {
            "id": ("#eef2ff", "#6366f1", "#4338ca"),
            "ctx": ("#ecfdf5", "#0f766e", "#0f766e"),
            "mask": ("#f1f5f9", "#64748b", "#334155"),
        }[kind]
        return (
            f'<rect x="40" y="{y}" width="238" height="26" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
            f'<text x="159" y="{y + 17.5}" class="d" fill="{col}">{text}</text>'
        )

    s = [
        '<svg viewBox="0 0 900 500" width="900" height="500" class="diagram">',
        '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>',
        '<rect x="24" y="14" width="270" height="452" rx="14" fill="none" stroke="#94a3b8"/>',
        '<text x="159" y="40" class="d">Lịch sử booking (left-pad)</text>',
    ]
    y = 52
    for label in ("Booking 1: cluster ID", "Booking t: cluster ID"):
        s += [
            row(y, label, "id"),
            row(y + 26, "Destination, market", "ctx"),
            row(y + 52, "Party, package, tháng", "ctx"),
            row(y + 78, "Lead time, số đêm", "ctx"),
        ]
        s.append(
            f'<line x1="278" y1="{y + 52}" x2="304" y2="{y + 52}" stroke="#64748b" marker-end="url(#ar)"/>'
        )
        if label.startswith("Booking 1"):
            s.append(f'<text x="159" y="{y + 124}" class="d">…</text>')
        y += 136
    s += [
        f'<text x="159" y="{y + 6}" class="d">Query: lượt search hiện tại</text>',
        f'<rect x="33" y="{y + 14}" width="252" height="114" rx="6" fill="none" stroke="#ea580c" stroke-dasharray="5 4" stroke-width="1.5"/>',
        row(y + 20, "[MASK]", "mask"),
        row(y + 46, "Destination → market (từ train)", "ctx"),
        row(y + 72, "Party, package, tháng", "ctx"),
        row(y + 98, "Lead time, số đêm", "ctx"),
        f'<line x1="285" y1="{y + 71}" x2="304" y2="{y + 71}" stroke="#64748b" marker-end="url(#ar)"/>',
    ]
    s += [
        '<rect x="306" y="52" width="34" height="370" rx="5" fill="#f1f5f9" stroke="#64748b"/>',
        '<text class="d" transform="translate(323,237) rotate(-90)">Embedding layer</text>',
        '<rect x="356" y="14" width="440" height="420" rx="18" fill="none" stroke="#94a3b8"/>',
        '<text x="576" y="44" class="d dt">Main operation layer (× L, dùng chung tham số)</text>',
        '<line x1="340" y1="237" x2="374" y2="237" stroke="#64748b" marker-end="url(#ar)"/>',
        '<polygon points="376,200 398,180 468,180 446,200" fill="#ecfdf5" stroke="#0f766e"/>',
        '<polygon points="446,200 468,180 468,254 446,274" fill="#ecfdf5" stroke="#0f766e"/>',
        '<rect x="376" y="200" width="70" height="74" fill="#eef2ff" stroke="#6366f1"/>',
        '<text x="422" y="300" class="d">(S+1) × F × C</text>',
        '<line x1="468" y1="237" x2="486" y2="237" stroke="#64748b" marker-end="url(#ar)"/>',
        '<rect x="488" y="150" width="32" height="174" rx="5" fill="#f1f5f9" stroke="#64748b"/>',
        '<text class="d" transform="translate(504,237) rotate(-90)">LayerNorm</text>',
        '<path d="M520 237 H534 V145 H548 M534 237 H548 M534 237 V329 H548" fill="none" stroke="#64748b"/>',
    ]
    for yy, name in (
        (126, "Sequence mixing"),
        (218, "Feature mixing"),
        (310, "Channel mixing"),
    ):
        s.append(
            f'<rect x="550" y="{yy}" width="170" height="38" rx="6" fill="#f8fafc" stroke="#64748b"/>'
            f'<text x="635" y="{yy + 24}" class="d">{name}</text>'
        )
    s += [
        '<path d="M720 145 H752 V225 M720 237 H740 M720 329 H752 V249" fill="none" stroke="#64748b"/>',
        '<circle cx="752" cy="237" r="12" fill="#fff" stroke="#64748b"/><text x="752" y="242" class="d dt">+</text>',
        '<text x="504" y="370" class="d dm">Norm</text><text x="635" y="370" class="d dm">Extraction</text><text x="752" y="370" class="d dm">Fusion</text>',
        '<line x1="764" y1="237" x2="816" y2="237" stroke="#64748b" marker-end="url(#ar)"/>',
        '<rect x="818" y="52" width="34" height="370" rx="5" fill="#f1f5f9" stroke="#64748b"/>',
        '<text class="d" transform="translate(835,237) rotate(-90)">Output: hàng query · bảng cluster</text>',
        '<text x="876" y="490" class="d dm" style="text-anchor:end">→ 100 điểm cluster (CE)</text>',
        '<rect x="24" y="478" width="12" height="12" rx="2" fill="#eef2ff" stroke="#6366f1"/><text x="42" y="488" class="d dm" style="text-anchor:start">Cluster ID</text>',
        '<rect x="130" y="478" width="12" height="12" rx="2" fill="#ecfdf5" stroke="#0f766e"/><text x="148" y="488" class="d dm" style="text-anchor:start">Context booking</text>',
        '<rect x="270" y="478" width="12" height="12" rx="2" fill="none" stroke="#ea580c" stroke-dasharray="3 2"/><text x="288" y="488" class="d dm" style="text-anchor:start">Query token</text>',
        "</svg>",
    ]
    return "".join(s)


# ---------------------------------------------------------------- numbers

field = {f["field"]: f for f in EDA["field_table"]}
depth = EDA["user_depth"]["histogram"]
n_bookers = sum(v["count"] for v in depth.values())
r5 = CTX["popularity_probe"]["recall_at_5"]
on = HS["by_history_online"]
sup = HS["by_dest_support"]
buckets = ["0", "1", "2", "3-4", "5-9", "10+"]
monthly = EDA["monthly_bookings"]

# ---------------------------------------------------------------- slides

slides = []


def jsd(key: str) -> str:
    return f"{CTX['distribution_shift'][key]['js_mean']:.4f}".replace(".", ",")


def slide(meta, title, body, sub=""):
    slides.append((meta, title, sub, body))


slides.append(
    (
        "cover",
        "",
        "",
        """
<div class="cover">
  <div class="kicker">Tuần 2 · Lựa chọn methodology</div>
  <h1>Lựa chọn methodology<br><span>cho gợi ý hotel cluster trên Expedia</span></h1>
  <p class="lead">Next-Best-Product Recommendation · Vinpearl × GSM · Bối cảnh (a): du lịch</p>
  <div class="cover-tags"><span>Dữ liệu công khai</span><span>Split theo thời gian</span><span>Xếp hạng đầy đủ 100 cluster</span></div>
</div>""",
    )
)

slide(
    "Nội dung",
    "Mạch trình bày",
    """
<div class="agenda">
  <div class="ag"><span>1</span><div><h3>Bài toán &amp; dữ liệu</h3><p>Đặc điểm Expedia và yêu cầu đặt ra cho mô hình</p></div></div>
  <div class="ag"><span>2</span><div><h3>Ba ứng viên</h3><p>SMLP4Rec, AdaGIN, LightGBM — phiên bản gốc</p></div></div>
  <div class="ag"><span>3</span><div><h3>Chọn SMLP4Rec</h3><p>Ý tưởng, kiến trúc, lý do phù hợp</p></div></div>
  <div class="ag"><span>4</span><div><h3>Điểm yếu &amp; đề xuất hybrid</h3><p>Lịch sử user ngắn → bổ sung prior theo context, ngưỡng từ dữ liệu</p></div></div>
  <div class="ag"><span>5</span><div><h3>Tổng kết</h3><p>Quyết định, mốc cần vượt, bước tiếp theo</p></div></div>
</div>""",
)

slide(
    "1 · Bài toán",
    "Bài toán",
    """
<div class="grid3">
  <div class="card"><div class="ic">01</div><h3>Mục tiêu</h3><p>Gợi ý <b>1 trong 100 hotel cluster</b> cho một lượt search. Cluster là proxy cho hạng phòng / gói nghỉ dưỡng của Vinpearl.</p></div>
  <div class="card"><div class="ic">02</div><h3>Đầu vào</h3><p><b>Lịch sử booking</b> của user và <b>context lượt search</b>: destination, số người, ngày nhận phòng, package, kênh.</p></div>
  <div class="card"><div class="ic">03</div><h3>Đánh giá</h3><p>Split <b>theo thời gian</b>, xếp hạng <b>toàn bộ</b> 100 cluster. Recall / NDCG / MRR @5, 10, 20, cắt theo độ dài lịch sử.</p></div>
</div>
<div class="note">Câu hỏi nghiên cứu RQ1: trên dữ liệu du lịch thưa, có mùa vụ, phụ thuộc context — mô hình context-aware / tuần tự có vượt CF truyền thống?</div>""",
)

months = list(monthly.keys())
mlabels = [m[2:4] + "/" + m[5:] if m.endswith(("01", "07")) else "" for m in months]
slide(
    "1 · Dữ liệu",
    "Expedia Hotel Recommendations",
    f"""
<div class="kpis">
  <div class="kpi"><b>{num(EDA["bookings"])}</b><span>booking (nhãn)</span></div>
  <div class="kpi"><b>{num(n_bookers)}</b><span>user có booking</span></div>
  <div class="kpi"><b>100</b><span>hotel cluster</span></div>
  <div class="kpi"><b>{num(field["srch_destination_id"]["n_unique_all"])}</b><span>destination</span></div>
</div>
<div class="row">
  <div class="panel grow"><div class="ph">Số booking theo tháng, {EDA["daily"]["date_min"]} → {EDA["daily"]["date_max"]}</div>
    {bar_chart(mlabels, [v / 1000 for v in monthly.values()], None, w=700, h=230)}
    <div class="axis-note">nghìn booking / tháng · tăng mạnh năm 2014</div></div>
  <div class="panel side"><ul class="clean">
    <li><b>{pct(EDA["clicks"] / EDA["total_rows"])}</b> dòng là click, chỉ dùng booking làm nhãn</li>
    <li>Mỗi dòng có đủ context: party, ngày, package, channel</li>
    <li>Cluster ẩn danh, không có thuộc tính item</li>
    <li>Không có cột giá</li></ul></div>
</div>""",
)

ctx_labels = ["Global", "Tháng", "Party", "Lặp cluster cũ", "Market", "Destination"]
ctx_vals = [
    r5["global_popularity"],
    r5["popularity_by_checkin_month"],
    r5["popularity_by_party"],
    r5["repeat_last_cluster"],
    r5["popularity_by_hotel_market"],
    r5["popularity_by_srch_destination_id"],
]
slide(
    "1 · Dữ liệu",
    "Tín hiệu nằm ở context của lượt search",
    f"""
<div class="row">
  <div class="panel grow"><div class="ph">Recall@5 của popularity theo từng điều kiện (test sau {CTX["popularity_probe"]["cut"][:10]})</div>
    {bar_chart(ctx_labels, ctx_vals, lambda v: pct(v, 1), w=700, h=280, colors=[GRAY, GRAY, GRAY, AMBER, BLUE, NAVY], ymax=0.62)}</div>
  <div class="panel side">
    <div class="big">{pct(r5["popularity_by_srch_destination_id"], 1)}</div><p class="muted">chỉ cần biết <b>destination</b> đang tìm</p>
    <div class="big amber">{pct(r5["repeat_last_cluster"], 1)}</div><p class="muted">lặp lại cluster lần trước — lịch sử có tín hiệu nhưng yếu</p>
    <p class="small">Tháng và party gần như không đổi phân phối cluster (JSD {jsd("checkin_month")} và {jsd("party")} so với destination {jsd("srch_destination_top")}).</p>
  </div>
</div>""",
)

dlabels = ["1", "2", "3–4", "5–10", "11–20", "21+"]
dvals = [depth[k]["share"] for k in ["1", "2", "3-4", "5-10", "11-20", "21+"]]
slide(
    "1 · Dữ liệu",
    "Lịch sử ngắn, nhiều user mới",
    f"""
<div class="row">
  <div class="panel grow"><div class="ph">Phân bố số booking mỗi user (trung vị {int(EDA["user_depth"]["quantiles"]["p50"])}, p90 {int(EDA["user_depth"]["quantiles"]["p90"])})</div>
    {bar_chart(dlabels, dvals, lambda v: pct(v, 1), w=600, h=270, colors=[RED, AMBER, BLUE, BLUE, BLUE, BLUE])}</div>
  <div class="panel side">
    <div class="big red">{pct(HS["by_history_train_only"]["0"]["share"])}</div><p class="muted">booking test thuộc user <b>chưa có trong train</b></p>
    <div class="big red">{pct(on["0"]["share"])}</div><p class="muted">booking test <b>không có booking nào trước đó</b> (kể cả trong test)</p>
    <div class="big amber">{num(EDA["destinations"]["n_dest_single_booking"])}</div><p class="muted">destination chỉ có 1 booking</p>
  </div>
</div>""",
)

slide(
    "1 · Yêu cầu",
    "Đặc điểm dữ liệu → yêu cầu với methodology",
    """
<table class="tbl">
<tr><th>Đặc điểm</th><th>Yêu cầu với mô hình</th></tr>
<tr><td>Context lượt search chi phối (destination)</td><td>Đưa context hiện tại vào mô hình, học tổ hợp giữa các trường</td></tr>
<tr><td>Có lịch sử booking theo thời gian</td><td>Khai thác thứ tự các booking trước đó</td></tr>
<tr><td>Lịch sử ngắn, nhiều user mới</td><td>Vẫn dự đoán tốt khi không có lịch sử</td></tr>
<tr><td>Chỉ 100 cluster, không thuộc tính item</td><td>Xếp hạng toàn catalog, chi phí huấn luyện vừa phải</td></tr>
<tr><td>Dữ liệu theo thời gian, có mùa vụ</td><td>Split theo thời gian, không rò rỉ thông tin tương lai</td></tr>
</table>
<div class="note">Hướng tiếp cận phù hợp: <b>context-aware sequential recommendation</b></div>""",
)

slide(
    "2 · Ứng viên",
    "Ba ứng viên (phiên bản gốc)",
    """
<table class="tbl cmp3">
<tr><th></th><th>SMLP4Rec</th><th>AdaGIN</th><th>LightGBM</th></tr>
<tr><td>Ý tưởng</td><td>MLP trộn thông tin theo 3 trục: chuỗi × feature × embedding</td><td>Đồ thị tương tác giữa các field (kiểu CTR)</td><td>Gradient boosting trên cây quyết định</td></tr>
<tr><td>Context lượt search</td><td>Qua trục feature</td><td>Tương tác field tường minh</td><td>Feature bảng, tương tác gián tiếp</td></tr>
<tr><td>Lịch sử user</td><td class="ok">Trục sequence, có thứ tự</td><td class="bad">Không mô hình hóa chuỗi</td><td class="bad">Chỉ qua feature tổng hợp thủ công</td></tr>
<tr><td>User ít lịch sử</td><td class="bad">Yếu — nhánh sequence rỗng</td><td class="ok">Ổn — không dùng user ID</td><td class="ok">Ổn nếu đủ feature context</td></tr>
<tr><td>Chi phí huấn luyện</td><td class="ok">1 dòng / booking, softmax 100 lớp</td><td class="bad">100 dòng / booking</td><td>1 dòng / booking, 100 cây mỗi vòng</td></tr>
<tr><td>Hạn chế chính</td><td>Lịch sử user ngắn</td><td>Tính toán nặng, code gốc có lỗi</td><td>Cách tiếp cận cũ, không học chuỗi</td></tr>
</table>""",
)

slide(
    "2 · Ứng viên",
    "AdaGIN và LightGBM: vì sao không chọn làm chính",
    f"""
<div class="grid2">
  <div class="card"><h3>AdaGIN <span class="tag">context-aware</span></h3>
    <p class="muted">Đồ thị tương tác giữa các field kiểu CTR, học cạnh thích nghi.</p>
    <p class="plus">+ Học tương tác field tường minh, không cần user ID</p>
    <p class="minus">− <b>Khối lượng tính toán lớn</b>: mỗi booking chấm 100 cluster → <b>{num(HS["candidate_rows_train_x100"])}</b> dòng train, <b>{num(HS["candidate_rows_test_x100"])}</b> dòng test</p>
    <p class="minus">− Loss BCE từng điểm, không tối ưu trực tiếp thứ hạng</p>
    <p class="minus">− Code gốc có lỗi (nhiễu Gumbel vẫn bật khi đánh giá)</p></div>
  <div class="card"><h3>LightGBM <span class="tag">tabular</span></h3>
    <p class="muted">Gradient boosting trên cây quyết định, feature dạng bảng.</p>
    <p class="plus">+ Mạnh với dữ liệu bảng, nhanh, dễ giải thích</p>
    <p class="minus">− <b>Cách tiếp cận cũ</b> (2017): không học biểu diễn, không mô hình hóa chuỗi</p>
    <p class="minus">− Tương tác chỉ học gián tiếp qua nhánh cây</p>
    <p class="minus">− Destination nhiều giá trị phải target-encode, dễ rò rỉ nhãn</p></div>
</div>
<div class="note">Cả hai được giữ làm <b>mô hình so sánh</b>: AdaGIN cho context-aware (RQ1), LightGBM làm baseline dạng bảng.</div>""",
)

slide(
    "3 · Lựa chọn",
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
    sub="Gao et al., ACM TOIS 2024 — cài lại từ paper, thêm query token cho lượt search hiện tại",
)

slide(
    "3 · Lựa chọn",
    "Vì sao SMLP4Rec phù hợp",
    """
<div class="grid2">
  <div class="card good"><h3>Đưa context chi phối vào mô hình</h3><p>Query token + trục feature học tổ hợp như <i>family × package × destination</i>.</p></div>
  <div class="card good"><h3>Khai thác lịch sử có thứ tự</h3><p>MLP trên trục sequence nhận biết thứ tự sẵn, không cần positional embedding.</p></div>
  <div class="card good"><h3>Nhẹ, nhanh</h3><p>Chi phí tuyến tính theo độ dài chuỗi; softmax trên 100 cluster rẻ → đủ ngân sách cho nhiều seed và ablation.</p></div>
  <div class="card good"><h3>Một khung thống nhất</h3><p>Lịch sử, context và feature nằm trong cùng một khối, dễ ablation từng nhánh.</p></div>
</div>
<div class="note">Kết luận: <b>SMLP4Rec là phương pháp chính</b> — đáp ứng cả context lẫn lịch sử với chi phí thấp nhất trong ba ứng viên.</div>""",
)

hist_labels = ["0", "1", "2", "3–4", "5–9", "10+"]
hist_share = [on[b]["share"] for b in buckets]
le1 = on["0"]["share"] + on["1"]["share"]
slide(
    "4 · Điểm yếu",
    "Điểm yếu lớn nhất: lịch sử user ngắn",
    f"""
<div class="row">
  <div class="panel grow"><div class="ph">Tỉ lệ booking test theo số booking trước đó của user</div>
    {bar_chart(hist_labels, hist_share, lambda v: pct(v, 1), w=600, h=270, colors=[RED, AMBER, BLUE, BLUE, BLUE, BLUE])}</div>
  <div class="panel side2 weak">
    <div class="wk"><b>{pct(on["0"]["share"])}</b><p>booking test <b>không có lịch sử</b> → nhánh sequence rỗng, mô hình chỉ còn context</p></div>
    <div class="wk"><b>{pct(le1)}</b><p>có <b>tối đa 1</b> booking trước đó → quá ít để học quy luật tuần tự</p></div>
    <div class="wk"><b>{pct(r5["popularity_by_srch_destination_id"])}</b><p>chỉ bảng đếm theo destination đã đạt mức này — mô hình neural phải tự <b>ghi nhớ</b> bảng đếm với {num(field["srch_destination_id"]["n_unique_all"])} destination, nhiều giá trị hiếm</p></div>
  </div>
</div>
<div class="note warn">Rủi ro: SMLP4Rec thuần có thể <b>thua bảng đếm popularity</b> ở nhóm user ít lịch sử → cần một thành phần bổ sung.</div>""",
)

slide(
    "4 · Đề xuất",
    "Đề xuất: hybrid cho user ít lịch sử",
    """
<div class="flow">
  <div class="fcol">
    <div class="fbox in">Lịch sử booking<br>+ query</div>
    <div class="fbox in">Destination<br>đang tìm</div>
  </div>
  <div class="farrow">→</div>
  <div class="fcol">
    <div class="fbox model">SMLP4Rec<br><small>điểm tuần tự</small></div>
    <div class="fbox prior">Context prior<br><small>destination → market → global</small></div>
  </div>
  <div class="farrow">→</div>
  <div class="fcol mid">
    <div class="fbox gate">× Gate<br><small>theo độ dài lịch sử &amp;<br>độ phổ biến destination</small></div>
  </div>
  <div class="farrow">→</div>
  <div class="fcol mid"><div class="fbox out">Cộng điểm<br>→ Top-K / 100 cluster</div></div>
</div>
<div class="grid3 tight">
  <div class="card"><h3>Ghi nhớ</h3><p>Prior là bảng đếm cluster theo destination, làm mượt về market khi ít dữ liệu. Chỉ tính từ quá khứ — không rò rỉ nhãn.</p></div>
  <div class="card"><h3>Tự điều chỉnh</h3><p>Gate học mức tin prior: user mới dựa vào prior, user nhiều lịch sử dựa vào SMLP4Rec.</p></div>
  <div class="card"><h3>Nối lịch sử với context</h3><p>Mỗi booking lịch sử được đánh dấu <b>“cùng destination với lượt search”</b>.</p></div>
</div>
<div class="modes">
  <div class="mode m0"><b>User mới</b><span>0 booking trước đó</span><em>Prior dẫn dắt</em></div>
  <div class="mode m1"><b>Ít lịch sử</b><span>1–4 booking</span><em>Kết hợp prior + SMLP4Rec</em></div>
  <div class="mode m2"><b>Nhiều lịch sử</b><span>≥ 5 booking</span><em>SMLP4Rec dẫn dắt</em></div>
</div>""",
)

prior = [on[b]["R@5_dest_raw_backoff_market"] for b in buckets]
any_h = [on[b]["R@5_history_then_dest"] for b in buckets]
same_h = [on[b]["R@5_same_dest_history_then_dest_smooth20"] for b in buckets]
sup_rows = "".join(
    f"<tr><td>{lab}</td><td>{pct(sup[k]['share'], 1)}</td><td>{pct(sup[k]['R@5_dest_raw_backoff_market'], 1)}</td>"
    f"<td class='{'hl' if k in ('1-4', '5-19') else ''}'>{pct(sup[k]['R@5_dest_smooth_m5'], 1)}</td></tr>"
    for k, lab in [
        ("0", "0"),
        ("1-4", "1–4"),
        ("5-19", "5–19"),
        ("20-99", "20–99"),
        ("100-999", "100–999"),
        ("1000+", "1000+"),
    ]
)
slide(
    "4 · Đề xuất",
    "Căn cứ chọn ngưỡng cho hybrid",
    f"""
<div class="row">
  <div class="panel grow"><div class="ph">Recall@5 của 3 luật đếm, theo số booking trước đó của user</div>
    {line_chart(["0", "1", "2", "3–4", "5–9", "10+"], [("same", same_h, TEAL, False, "above"), ("prior", prior, NAVY, False, "below"), ("any", any_h, RED, True, "below")], w=600, h=290)}
    <div class="legend col"><span class="lg" style="--c:{TEAL}">Cluster từng đặt <b>ở cùng destination</b> xếp trước, còn lại theo prior</span><span class="lg" style="--c:{NAVY}">Chỉ prior: tần suất cluster theo destination</span><span class="lg dash" style="--c:{RED}">Mọi cluster từng đặt xếp trước, còn lại theo prior</span></div></div>
  <div class="panel side2">
    <div class="th-card"><div class="lbl">Ngưỡng lịch sử</div><div class="val">0 | 1 | 2–4 | 5–9 | 10+</div><p>0 booking: lịch sử không có gì để dùng. Từ 5 booking: luật dùng lịch sử cùng destination hơn prior &gt; 6 điểm.</p></div>
    <div class="th-card"><div class="lbl">Ngưỡng destination (số booking train)</div><div class="val">0 | 1–19 | 20+</div>
      <table class="mini"><tr><th>Booking</th><th>% test</th><th>Prior thô</th><th>Prior mượt</th></tr>{sup_rows}</table></div>
  </div>
</div>
<div class="note">Các số trên là <b>luật xếp hạng bằng bảng đếm</b>, không học tham số — <b>chưa phải kết quả của SMLP4Rec hay hybrid</b>. Dùng để (1) đặt ngưỡng, (2) làm mốc mà mô hình phải vượt. Kết quả mô hình có sau khi cài đặt.</div>""",
)

slide(
    "5 · Tổng kết",
    "Tổng kết",
    f"""
<div class="sum">
  <div class="sum-main">
    <div class="lbl">Quyết định</div>
    <h3>SMLP4Rec + hybrid cho user ít lịch sử</h3>
    <ul class="clean">
      <li>SMLP4Rec: đưa context lượt search và lịch sử booking vào một mô hình nhẹ</li>
      <li>Hybrid: prior theo destination + gate theo độ dài lịch sử, bù cho {pct(on["0"]["share"])} booking không có lịch sử</li>
      <li>AdaGIN (context-aware) và LightGBM (bảng) giữ làm mô hình so sánh</li>
    </ul>
  </div>
  <div class="sum-side">
    <div class="kpi"><b>{pct(HS["overall"]["R@5_same_dest_history_then_dest_smooth20"])}</b><span>Mốc Recall@5 cần vượt — luật đếm tốt nhất, <b>chưa phải kết quả mô hình</b></span></div>
    <div class="kpi"><b>{pct(on["0"]["R@5_dest_smooth_m5"])}</b><span>Mốc cho user mới — prior làm mượt theo destination</span></div>
  </div>
</div>
<div class="timeline">
  <div class="tl"><span>1</span><h3>Pipeline dữ liệu</h3><p>Chuỗi booking, query token, split theo thời gian</p></div>
  <div class="tl"><span>2</span><h3>Baseline</h3><p>Popularity, heuristic lịch sử, LightGBM</p></div>
  <div class="tl"><span>3</span><h3>SMLP4Rec</h3><p>Bản thuần, tuning trên validation</p></div>
  <div class="tl"><span>4</span><h3>Hybrid</h3><p>Prior + gate, chốt ngưỡng trên validation</p></div>
  <div class="tl"><span>5</span><h3>So sánh</h3><p>AdaGIN, ablation, cắt theo lịch sử và mùa</p></div>
</div>""",
)


# ---------------------------------------------------------------- page

CSS = """
:root{--navy:#1e3a8a;--blue:#2563eb;--teal:#0f766e;--amber:#b45309;--red:#dc2626;--ink:#0f172a;--muted:#475569;--sub:#64748b;--line:#e2e8f0;--soft:#f8fafc}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#eef2f7;font-family:'Be Vietnam Pro',system-ui,sans-serif;color:var(--ink);-webkit-font-smoothing:antialiased}
.deck{width:1280px;height:720px;position:relative;transform-origin:0 0}
.slide{width:1280px;height:720px;background:#fff;padding:44px 60px 40px;display:none;flex-direction:column;position:absolute;inset:0;overflow:hidden}
.slide.on{display:flex}
.hd{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1.5px solid var(--line);padding-bottom:14px;margin-bottom:26px}
.meta{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:6px}
h2{font-size:34px;font-weight:700;letter-spacing:-.01em}
.sub{font-size:14px;color:var(--sub);margin-top:4px}
.pg{font-size:13px;color:var(--sub);font-weight:500}
.ft{position:absolute;left:60px;right:60px;bottom:16px;display:flex;justify-content:space-between;font-size:11.5px;color:#94a3b8}
.body{flex:1;display:flex;flex-direction:column;gap:22px;min-height:0;justify-content:center;padding-bottom:18px}
.row{display:flex;gap:22px;flex:1;min-height:0}
.panel{background:var(--soft);border:1px solid var(--line);border-radius:14px;padding:18px 20px}
.grow{flex:1;display:flex;flex-direction:column}.grow .chart{flex:1;margin:auto 0}.side{width:340px;display:flex;flex-direction:column;justify-content:center}.side2{width:470px;display:flex;flex-direction:column;gap:14px;background:none;border:none;padding:0}
.ph{font-size:14px;font-weight:600;color:var(--muted);margin-bottom:8px}
.axis-note{font-size:12px;color:var(--sub);text-align:right}
.chart{width:100%;height:auto;display:block}
.chart .cv{font-size:13px;font-weight:600;fill:#334155;text-anchor:middle}
.chart .cl{font-size:12.5px;fill:#64748b;text-anchor:middle}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.kpi{border:1px solid var(--line);border-radius:14px;padding:16px 20px;background:#fff}
.kpi>b{display:block;font-size:32px;font-weight:700;color:var(--navy)}
.kpi span{font-size:14px;color:var(--sub)}.kpi span b{color:var(--ink)}
.big{font-size:40px;font-weight:700;color:var(--navy);line-height:1.1}.big.amber{color:var(--amber)}.big.red{color:var(--red)}
.muted{color:var(--muted);font-size:15px;margin:2px 0 14px}
.small{font-size:12.5px;color:var(--sub);line-height:1.5}
ul.clean{list-style:none;display:flex;flex-direction:column;gap:14px;font-size:15.5px;color:var(--muted)}
ul.clean li{padding-left:18px;position:relative}ul.clean li:before{content:'';position:absolute;left:0;top:9px;width:7px;height:7px;border-radius:50%;background:var(--blue)}
ul.clean b{color:var(--ink)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid3.tight .card{padding:16px 18px}
.card{border:1px solid var(--line);border-radius:14px;padding:26px 28px;background:#fff}
.card h3{font-size:20px;font-weight:700;margin-bottom:8px}
.card p{font-size:16.5px;color:var(--muted);line-height:1.55}
.card.good{border-left:4px solid var(--teal);border-radius:0 14px 14px 0}
.ic{font-size:13px;font-weight:700;color:var(--blue);margin-bottom:10px}
.tag{font-size:11px;font-weight:600;color:var(--blue);background:#eff6ff;border-radius:20px;padding:3px 9px;vertical-align:middle;margin-left:6px}
.plus{color:var(--teal)!important;margin-top:6px}.minus{color:#9f1239!important;margin-top:6px}
.note{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:15px 22px;font-size:16px;color:#1e3a8a;line-height:1.5}
.note.warn{background:#fff7ed;border-color:#fed7aa;color:#9a3412}
.tbl{width:100%;border-collapse:collapse;font-size:17.5px}
.tbl th{text-align:left;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--sub);padding:12px 16px;border-bottom:2px solid var(--line)}
.tbl td{padding:17px 18px;border-bottom:1px solid var(--line);color:var(--muted)}
.tbl td:first-child{color:var(--ink);font-weight:600}
.tbl.cmp th.win{color:var(--teal)}.tbl.cmp td.win{background:#f0fdfa;color:#134e4a}
.diagram-wrap{flex:1;display:flex;align-items:center;justify-content:center;background:#fff;padding:6px}
.diagram{width:100%;height:auto}
.diagram .d{font-size:13px;fill:#334155;text-anchor:middle}.diagram .dt{font-size:15px;font-weight:600}.diagram .dm{fill:#64748b;font-size:12.5px}
.steps{width:300px;gap:16px}
.step{display:flex;gap:12px;align-items:flex-start}
.step span{flex:none;width:28px;height:28px;border-radius:50%;background:var(--navy);color:#fff;font-weight:700;font-size:14px;display:flex;align-items:center;justify-content:center}
.step p{font-size:15px;color:var(--muted);line-height:1.5}
.flow{display:flex;align-items:center;gap:14px;justify-content:center;padding:6px 0 4px}
.fcol{display:flex;flex-direction:column;gap:16px}
.fbox{border-radius:12px;padding:14px 18px;text-align:center;font-size:16px;font-weight:600;min-width:170px;border:1.5px solid}
.fbox small{display:block;font-weight:500;font-size:12.5px;margin-top:3px;opacity:.85}
.fbox.in{background:#f8fafc;border-color:#cbd5e1;color:#334155}
.fbox.model{background:#eef2ff;border-color:#6366f1;color:#3730a3}
.fbox.prior{background:#ecfdf5;border-color:#0f766e;color:#115e59}
.fbox.gate{background:#fffbeb;border-color:#d97706;color:#92400e}
.fbox.out{background:var(--navy);border-color:var(--navy);color:#fff}
.farrow{font-size:26px;color:#94a3b8}
.legend{display:flex;gap:18px;font-size:12.5px;color:var(--muted);margin-top:4px}
.legend.col{flex-direction:column;gap:4px}.legend b{color:var(--ink)}
.lg:before{content:'';display:inline-block;width:18px;height:3px;background:var(--c);vertical-align:middle;margin-right:6px}
.lg.dash:before{background:repeating-linear-gradient(90deg,var(--c) 0 5px,transparent 5px 9px)}
.th-card{border:1px solid var(--line);border-radius:14px;padding:14px 18px;background:#fff}
.th-card .lbl{font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--sub)}
.th-card .val{font-size:22px;font-weight:700;color:var(--navy);margin:4px 0 6px}
.th-card p{font-size:13.5px;color:var(--muted)}
.mini{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:4px}
.mini th{color:var(--sub);font-weight:600;text-align:right;padding:3px 6px;border-bottom:1px solid var(--line)}
.mini td{text-align:right;padding:3px 6px;color:var(--muted)}.mini th:first-child,.mini td:first-child{text-align:left}
.mini td.hl{color:var(--teal);font-weight:700}
.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-radius:14px;overflow:hidden;border:1px solid var(--line)}
.mode{padding:16px 22px;display:flex;flex-direction:column;gap:2px}
.mode b{font-size:17px}.mode span{font-size:13.5px;color:var(--sub)}.mode em{font-style:normal;font-weight:600;font-size:15px;margin-top:6px}
.m0{background:#ecfdf5}.m0 em{color:#0f766e}.m1{background:#fffbeb}.m1 em{color:#b45309}.m2{background:#eef2ff}.m2 em{color:#3730a3}
.agenda{display:flex;flex-direction:column;gap:14px}
.ag{display:flex;gap:20px;align-items:center;border:1px solid var(--line);border-radius:14px;padding:16px 22px}
.ag span{flex:none;width:40px;height:40px;border-radius:50%;background:var(--navy);color:#fff;font-weight:700;font-size:18px;display:flex;align-items:center;justify-content:center}
.ag h3{font-size:19px;margin-bottom:2px}.ag p{font-size:15px;color:var(--muted)}
.tbl.cmp3 td{font-size:16px;padding:14px 16px}.tbl.cmp3 th{padding:10px 16px}
.tbl.cmp3 td.ok{color:#0f766e}.tbl.cmp3 td.bad{color:#b91c1c}
.weak{justify-content:center}.wk{border-left:4px solid var(--red);padding:6px 0 6px 16px}
.wk>b{font-size:30px;color:var(--red);display:block;line-height:1.1}.wk p{font-size:14.5px;color:var(--muted);margin-top:4px}
.wk:nth-child(3){border-color:var(--amber)}.wk:nth-child(3)>b{color:var(--amber)}
.sum{display:flex;gap:22px}
.sum-main{flex:1;border:1px solid var(--line);border-left:5px solid var(--navy);border-radius:0 14px 14px 0;padding:22px 26px}
.sum-main .lbl{font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--sub)}
.sum-main h3{font-size:24px;margin:4px 0 14px;color:var(--navy)}
.sum-side{width:330px;display:flex;flex-direction:column;gap:14px}
.timeline{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-top:4px}
.tl{border-top:4px solid var(--blue);padding-top:16px}
.tl span{font-size:28px;font-weight:700;color:var(--blue)}
.tl h3{font-size:20px;margin:6px 0 8px}.tl p{font-size:16px;color:var(--muted);line-height:1.55}
.cover{flex:1;display:flex;flex-direction:column;justify-content:center;padding-left:20px;border-left:6px solid var(--navy)}
.kicker{font-size:15px;font-weight:600;color:var(--blue);letter-spacing:.08em;text-transform:uppercase;margin-bottom:18px}
.cover h1{font-size:52px;font-weight:700;line-height:1.15;letter-spacing:-.02em}
.cover h1 span{font-size:30px;font-weight:500;color:var(--muted)}
.lead{font-size:18px;color:var(--sub);margin-top:22px}
.cover-tags{display:flex;gap:10px;margin-top:28px}
.cover-tags span{font-size:13.5px;padding:7px 14px;border-radius:20px;background:#eff6ff;color:var(--navy);font-weight:600}
@media print{
  @page{size:1280px 720px;margin:0}
  html,body{background:#fff}
  .deck{transform:none!important;position:static!important;width:auto;height:auto}
  .slide{display:flex!important;position:relative;page-break-after:always;break-after:page}
  .slide:last-child{page-break-after:auto;break-after:auto}
  html,body{height:auto;overflow:visible}
  .slide{height:720px;break-inside:avoid}
}
"""

JS = """
const s=[...document.querySelectorAll('.slide')];let i=0;
function show(n){i=Math.max(0,Math.min(s.length-1,n));s.forEach((e,k)=>e.classList.toggle('on',k===i));}
function fit(){const d=document.querySelector('.deck');const k=Math.min(innerWidth/1280,innerHeight/720);
 d.style.transform=`scale(${k})`;d.style.position='absolute';d.style.left=((innerWidth-1280*k)/2)+'px';d.style.top=((innerHeight-720*k)/2)+'px';}
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);});
addEventListener('click',e=>show(e.clientX>innerWidth/2?i+1:i-1));
addEventListener('resize',fit);fit();show((parseInt(location.hash.slice(1))||1)-1);
"""


def render():
    parts = []
    total = len(slides)
    for n, (meta, title, sub, body) in enumerate(slides, 1):
        if meta == "cover":
            parts.append(
                f'<section class="slide">{body}<div class="ft"><span>Topic C1 · Next-Best-Product Recommendation</span><span>{n} / {total}</span></div></section>'
            )
            continue
        subh = f'<div class="sub">{sub}</div>' if sub else ""
        parts.append(
            f'<section class="slide"><div class="hd"><div><div class="meta">{meta}</div><h2>{title}</h2>{subh}</div>'
            f'<div class="pg">Tuần 2 · Methodology</div></div><div class="body">{body}</div>'
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ scripts/expedia_*.py</span><span>{n} / {total}</span></div></section>'
        )
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tuần 2: Lựa chọn methodology</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="deck">{"".join(parts)}</div><script>{JS}</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    render()
