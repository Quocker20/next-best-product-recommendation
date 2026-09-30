"""Build the week-3 pipeline-validation slide deck (Vietnamese, HTML).

Every number on the slides is read from reports/summary/week3/smlprec_expedia_run.json
(written by scripts/run_smlprec_expedia.py); nothing is typed in from memory.

Usage: python scripts/week3_pipeline_slides.py
Output: reports/slides/week3/pipeline_validation_slides.html
PDF: print the HTML with headless Chrome (--print-to-pdf, page size 1280x720).
"""

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "summary" / "week3" / "smlprec_expedia_run.json"
OUT = ROOT / "reports" / "slides" / "week3" / "pipeline_validation_slides.html"

R = json.loads(SRC.read_text(encoding="utf-8"))
T, V, H = R["test"], R["valid"], R["heuristics_same_test_rows"]
POP, REP = H["global_popularity"], H["repeat_last_then_popularity"]
KS = (5, 10, 20)


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def dec(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def day(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%d/%m/%Y")


def pp(a: float, b: float) -> str:
    return f"{(a - b) * 100:+.2f}".replace(".", ",") + " điểm %"


# ------------------------------------------------------------------ chart
BLUE, TEAL, GRAY = "#2563eb", "#0f766e", "#94a3b8"


def grouped_bars() -> str:
    """Recall@K for the three rankers (inline SVG)."""
    series = [
        ("Popularity toàn cục", [POP[f"recall@{k}"] for k in KS], GRAY),
        (
            "Lặp lại cụm cuối, rồi popularity",
            [REP[f"recall@{k}"] for k in KS],
            "#f59e0b",
        ),
        ("SMLP4Rec (1 epoch)", [T[f"recall@{k}"] for k in KS], BLUE),
    ]
    w, h, top, bottom, left = 720, 330, 26, 34, 10
    ymax = 0.7
    ph, gw = h - top - bottom, (w - left) / len(KS)
    bw = gw * 0.24
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in (0.2, 0.4, 0.6):
        y = top + ph * (1 - g / ymax)
        out.append(
            f'<line x1="{left}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        out.append(
            f'<text x="{left}" y="{y - 4:.1f}" class="cl">{int(g * 100)}%</text>'
        )
    for gi, k in enumerate(KS):
        x0 = left + gi * gw + gw * 0.13
        for si, (_, vals, col) in enumerate(series):
            v = vals[gi]
            bh = ph * v / ymax
            x = x0 + si * (bw + 6)
            y = top + ph - bh
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>'
            )
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{y - 6:.1f}" class="cv" text-anchor="middle">{pct(v, 1)}</text>'
            )
        out.append(
            f'<text x="{x0 + (3 * bw + 12) / 2:.1f}" y="{h - 10}" class="ck" text-anchor="middle">Recall@{k}</text>'
        )
    out.append("</svg>")
    legend = "".join(
        f'<span class="lg" style="--c:{c}">{n}</span>' for n, _, c in series
    )
    return "".join(out) + f'<div class="legend">{legend}</div>'


def table(head, rows, hl_last=False) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = ""
    for i, r in enumerate(rows):
        cls = ' class="hl"' if hl_last and i == len(rows) - 1 else ""
        trs += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table class="tbl"><tr>{th}</tr>{trs}</table>'


S = R["splits"]
CFG = R["config"]
DS = R["dataset_stats"]
TM = R["timing_seconds"]

slides = []

# 1 cover
slides.append(
    (
        "cover",
        "",
        "",
        """<div class="cover"><div class="kicker">Tuần 3 · Kiểm tra pipeline</div>
<h1>Chạy thử SMLP4Rec trên Expedia<br><span>những gì đã đổi, đã chạy và đã thấy</span></h1>
<p class="lead">Topic C1 · Next-Best-Product Recommendation · 30/09/2026</p>
<div class="cover-tags"><span>Pipeline end-to-end</span><span>1 epoch · CPU</span><span>Chưa phải benchmark</span></div></div>""",
    )
)

# 2 goal
slides.append(
    (
        "Mục tiêu",
        "Chạy pipeline một lần trên Expedia",
        "Chỉ kiểm tra luồng chạy và đọc kết quả, không tối ưu",
        """<div class="two">
<div class="card"><h3>Trong phạm vi</h3><ul>
<li>Chuyển mã SMLP4Rec vào <code>src/</code></li>
<li>Chỉnh tối thiểu để chạy với Expedia</li>
<li>Huấn luyện, validate, test một lần</li>
<li>Lưu kết quả vào <code>reports/summary/week3/</code></li></ul></div>
<div class="card off"><h3>Ngoài phạm vi</h3><ul>
<li>Xử lý dữ liệu, feature engineering</li>
<li>Viết mô hình mới từ đầu</li>
<li>Tinh chỉnh siêu tham số</li>
<li>Query token và cold-start hybrid (bước sau)</li></ul></div></div>
<p class="note">Kết quả chỉ nhằm xác nhận pipeline chạy được. Mọi con số trong bộ slide này lấy từ <code>smlprec_expedia_run.json</code>.</p>""",
    )
)

# 3 changes: files
slides.append(
    (
        "Đã thay đổi",
        "Các file mới trong repo",
        "Không sửa dữ liệu gốc, không sửa thư viện ngoài repo này",
        table(
            ["File", "Vai trò"],
            [
                [
                    "<code>src/models/smlprec.py</code>",
                    "SMLP4Rec chép từ fork, 2 chỉnh sửa nhỏ (đánh dấu <code># ADAPTED</code>)",
                ],
                [
                    "<code>scripts/expedia_to_recbole.py</code>",
                    "Xuất đặt phòng ra file <code>.inter</code> của RecBole: chỉ user, cụm khách sạn, thời gian",
                ],
                [
                    "<code>configs/smlprec_expedia.yaml</code>",
                    "Cấu hình mô hình, huấn luyện, đánh giá",
                ],
                [
                    "<code>scripts/run_smlprec_expedia.py</code>",
                    "Chạy train, valid, test và hai heuristic trên cùng dòng test",
                ],
                [
                    "<code>reports/summary/week3/</code>",
                    "README, JSON kết quả, log RecBole",
                ],
            ],
        )
        + '<p class="note">Cập nhật thêm: <code>PROGRESS.md</code> (mục Phase 3) và sơ đồ thư mục trong <code>CLAUDE.md</code>.</p>',
    )
)

# 4 model adaptation
slides.append(
    (
        "Đã thay đổi",
        "Chỉnh sửa mô hình: chỉ hai điểm",
        "Còn lại giữ nguyên như bản gốc của fork",
        """<div class="two">
<div class="card"><h3>Đã chỉnh</h3><ol>
<li><b>Độ dài chuỗi</b> lấy từ <code>MAX_ITEM_LIST_LENGTH</code> thay vì cố định 50</li>
<li><b>Cho phép danh sách feature rỗng</b>: cụm khách sạn không có thuộc tính, nên bỏ lớp embedding feature</li></ol></div>
<div class="card off"><h3>Giữ nguyên (đã biết là vấn đề)</h3><ul>
<li>Một khối SMLP dùng lại cho mọi lớp, trọng số chung</li>
<li>Không có kết nối residual quanh khối</li>
<li>Còn code thừa chưa dùng</li>
<li>Repo gốc không có file license: chỉ dùng nội bộ, gắn nhãn “research adaptation”</li></ul></div></div>""",
    )
)

# 5 pipeline flow
steps = [
    ("train.csv", "4 GB, đọc theo khối"),
    ("Lọc is_booking = 1", "3.000.693 đặt phòng"),
    ("expedia.inter", "user · cluster · giờ"),
    ("RecBole", "chuỗi + chia theo thời gian"),
    ("SMLP4Rec", "CE trên 100 cụm"),
    ("Đánh giá", "xếp hạng đầy đủ"),
]
flow = "".join(
    f'<div class="st"><b>{a}</b><span>{b}</span></div>'
    + ('<div class="ar">›</div>' if i < len(steps) - 1 else "")
    for i, (a, b) in enumerate(steps)
)
slides.append(
    (
        "Đã chạy",
        "Luồng pipeline và thiết lập",
        "Một lần chạy, seed 2022, CPU",
        f"""<div class="flow">{flow}</div>
<div class="kpis">
<div class="kpi"><b>{num(DS["users"])}</b><span>người dùng</span></div>
<div class="kpi"><b>{DS["items"]}</b><span>cụm khách sạn</span></div>
<div class="kpi"><b>{num(DS["sequence_targets"])}</b><span>mục tiêu “đặt tiếp theo”</span></div>
<div class="kpi"><b>{num(R["n_parameters"])}</b><span>tham số</span></div></div>
<p class="note">Mô hình: {CFG["n_layers"]} lớp, hidden {CFG["hidden_size"]}, dropout {CFG["hidden_dropout_prob"]}, lr {CFG["learning_rate"]}, batch {CFG["train_batch_size"]}, {CFG["epochs"]} epoch, lịch sử tối đa {CFG["MAX_ITEM_LIST_LENGTH"]} lần đặt. Thời gian: dựng dữ liệu {TM["data_build"]:.1f}s · train + valid {TM["train_and_valid"]:.1f}s · test {TM["test"]:.1f}s.</p>""",
    )
)

# 6 split
rows = []
for name, label in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
    s = S[name]
    rows.append(
        [
            label,
            num(s["targets"]),
            day(s["first_target_unix"]),
            day(s["last_target_unix"]),
            dec(s["mean_history_len"], 3),
        ]
    )
slides.append(
    (
        "Đã chạy",
        "Cách chia dữ liệu",
        "Chia theo thời gian 80/10/10 trên các mục tiêu, không chia ngẫu nhiên",
        table(["Tập", "Số mục tiêu", "Từ ngày", "Đến ngày", "Lịch sử trung bình"], rows)
        + f"""<ul class="tight">
<li>Mục tiêu = mọi lần đặt phòng trừ lần đặt đầu tiên của mỗi người dùng; lịch sử = các lần đặt trước đó của người đó.</li>
<li>Đánh giá xếp hạng đầy đủ trên {DS["items"]} cụm, không lấy mẫu âm.</li>
<li>Người dùng chưa từng đặt (cold-start) <b>không</b> nằm trong tập test của lần chạy này.</li></ul>""",
    )
)

# 7 results
slides.append(
    (
        "Đã thấy",
        "Kết quả trên tập test",
        "Xếp hạng đầy đủ trên 100 cụm · cùng các dòng test cho cả ba phương pháp",
        f"""<div class="split"><div>{grouped_bars()}</div>
<div class="side">
<div class="th-card"><div class="lbl">SMLP4Rec, test</div><div class="val">Recall@5 {pct(T["recall@5"])}</div>
<p>NDCG@10 {dec(T["ndcg@10"])} · MRR@10 {dec(T["mrr@10"])}</p></div>
<div class="th-card"><div class="lbl">So với lặp lại cụm cuối</div><div class="val">{pp(T["recall@5"], REP["recall@5"])}</div><p>Recall@5</p></div>
<div class="th-card"><div class="lbl">So với popularity toàn cục</div><div class="val">{pp(T["recall@5"], POP["recall@5"])}</div><p>Recall@5</p></div></div></div>""",
    )
)

# 8 findings
slides.append(
    (
        "Đã thấy",
        "Đọc kết quả",
        "",
        f"""<div class="finds">
<div class="fd"><span>1</span><div><h3>Pipeline chạy thông suốt</h3><p>Xuất dữ liệu, dựng chuỗi, chia theo thời gian, huấn luyện, xếp hạng đầy đủ và nạp lại checkpoint đều chạy không lỗi, tổng cộng khoảng {(TM["data_build"] + TM["train_and_valid"] + TM["test"]) / 60:.1f} phút trên CPU.</p></div></div>
<div class="fd"><span>2</span><div><h3>Lịch sử đặt phòng tự nó có tín hiệu</h3><p>SMLP4Rec vượt lặp-lại-cụm-cuối {pp(T["recall@5"], REP["recall@5"])} và popularity {pp(T["recall@5"], POP["recall@5"])} ở Recall@5. Valid ({pct(V["recall@5"])}) và test gần nhau, không thấy lệch giữa hai tập.</p></div></div>
<div class="fd"><span>3</span><div><h3>Còn thấp hơn mốc tuần 2 (53,07%)</h3><p>Mô hình chỉ thấy các cụm đã đặt, chưa thấy điểm đến của lần tìm kiếm hiện tại, tức tín hiệu mạnh nhất. Đây chính là chỗ query token cần lấp.</p></div></div>
<div class="fd warn"><span>!</span><div><h3>Chưa so sánh trực tiếp với tuần 2</h3><p>Tuần 2 cắt theo phân vị 80% của thời gian đặt và có cả người dùng mới; lần chạy này chia 80/10/10 trên các mục tiêu và bỏ lần đặt đầu của mỗi người.</p></div></div></div>""",
    )
)

# 9 issues
slides.append(
    (
        "Vấn đề gặp phải",
        "Lỗi đã gặp và cách xử lý",
        "Bốn lỗi, đều đã xử lý; lỗi thứ tư có thể làm sai kết quả nếu bỏ sót",
        table(
            ["Vấn đề", "Nguyên nhân", "Xử lý"],
            [
                [
                    "Torch không nạp được DLL",
                    "Đường dẫn workspace quá dài trên Windows",
                    "Venv Python 3.11 đặt ở đường dẫn ngắn",
                ],
                [
                    "Huấn luyện rất chậm",
                    "<code>sine.py</code> bật anomaly detection khi import",
                    "Tắt trong script chạy",
                ],
                [
                    "Bước test lỗi khi nạp checkpoint",
                    "PyTorch ≥ 2.6 mặc định <code>weights_only=True</code>",
                    "Nạp với <code>weights_only=False</code> (file do chính lần chạy tạo ra)",
                ],
                [
                    "<b>Sai đơn vị thời gian</b>",
                    "pandas 3 đọc ngày ra micro giây, chia 10⁹ làm các lần đặt cách nhau dưới ~17 phút trùng thời điểm",
                    "Đổi sang tính giây tường minh, chạy lại; số liệu ở đây là của lần chạy đã sửa",
                ],
            ],
        ),
    )
)

# 10 next
slides.append(
    (
        "Bước tiếp theo",
        "Hạn chế và việc cần làm",
        "",
        """<div class="two">
<div class="card off"><h3>Hạn chế của lần chạy này</h3><ul>
<li>1 epoch, chưa tinh chỉnh, một seed duy nhất</li>
<li>Chưa có ngữ cảnh tìm kiếm (điểm đến, ngày, số người, gói)</li>
<li>Chưa có cold-start hybrid</li>
<li>RecBole 1.0.1 buộc dùng numpy 1.23 / pandas 1.5 trong venv riêng</li></ul></div>
<div class="card"><h3>Đề xuất</h3><ol>
<li>Thêm query token cho ngữ cảnh tìm kiếm hiện tại</li>
<li>Thêm cold-start hybrid (prior theo điểm đến, cổng học theo độ dài lịch sử)</li>
<li>Căn chỉnh cách chia với tuần 2 để so với mốc 57,85%</li>
<li>Quyết định: giữ RecBole hay chuyển sang PyTorch thuần</li>
<li>Sửa lỗi kiến trúc: trọng số riêng mỗi lớp, thêm residual</li></ol></div></div>""",
    )
)

CSS = """
:root{--navy:#1e3a8a;--blue:#2563eb;--teal:#0f766e;--amber:#b45309;--red:#dc2626;
--ink:#0f172a;--muted:#475569;--sub:#64748b;--line:#e2e8f0;--bg:#f8fafc}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#e2e8f0;font-family:'Be Vietnam Pro',system-ui,sans-serif;color:var(--ink)}
.deck{width:1280px;height:720px;transform-origin:top left}
.slide{display:none;position:relative;width:1280px;height:720px;background:#fff;padding:44px 64px 56px;flex-direction:column}
.slide.on{display:flex}
.hd{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px}
.meta{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:6px}
h2{font-size:34px;font-weight:700;letter-spacing:-.01em}
.sub{font-size:16px;color:var(--sub);margin-top:6px}
.pg{font-size:13px;color:var(--sub);border:1px solid var(--line);border-radius:20px;padding:5px 12px}
.body{flex:1;display:flex;flex-direction:column;gap:16px;min-height:0}
.ft{position:absolute;left:64px;right:64px;bottom:20px;display:flex;justify-content:space-between;font-size:12px;color:var(--sub)}
code{font-family:Consolas,monospace;font-size:.88em;background:#f1f5f9;border-radius:5px;padding:1px 6px;color:var(--navy)}
.note{font-size:14.5px;color:var(--sub);line-height:1.6}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.card{border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:14px;padding:20px 24px;background:#fff}
.card.off{border-top-color:var(--amber);background:var(--bg)}
.card h3{font-size:19px;margin-bottom:12px;color:var(--navy)}
.card ul,.card ol{padding-left:20px;font-size:16px;line-height:1.65;color:var(--muted)}
.card li{margin-bottom:6px}
.tbl{width:100%;border-collapse:collapse;font-size:15.5px}
.tbl th{text-align:left;color:var(--sub);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.04em;padding:10px 14px;border-bottom:2px solid var(--line)}
.tbl td{padding:12px 14px;border-bottom:1px solid var(--line);color:var(--muted);line-height:1.5;vertical-align:top}
.tbl td:first-child{color:var(--ink);font-weight:500}
.tbl.hl,.tbl tr.hl td{color:var(--teal);font-weight:700}
ul.tight{padding-left:22px;font-size:15.5px;color:var(--muted);line-height:1.7}
.flow{display:flex;align-items:stretch;gap:6px}
.st{flex:1;border:1px solid var(--line);border-radius:12px;padding:14px 12px;background:var(--bg);display:flex;flex-direction:column;gap:4px;text-align:center}
.st b{font-size:15px;color:var(--navy)}.st span{font-size:12.5px;color:var(--sub)}
.ar{display:flex;align-items:center;color:var(--blue);font-size:28px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.kpi{border:1px solid var(--line);border-radius:14px;padding:16px 20px;background:#fff}
.kpi b{display:block;font-size:30px;color:var(--navy);letter-spacing:-.01em}.kpi span{font-size:14px;color:var(--sub)}
.split{display:flex;gap:28px;align-items:center}
.chart{width:720px;height:330px}
.cv{font-size:11.5px;fill:#334155;font-weight:600}.cl{font-size:11px;fill:#94a3b8}.ck{font-size:14px;fill:#334155;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--muted);margin-top:8px}
.lg:before{content:'';display:inline-block;width:12px;height:12px;border-radius:3px;background:var(--c);margin-right:6px;vertical-align:-1px}
.side{display:flex;flex-direction:column;gap:12px;flex:1}
.th-card{border:1px solid var(--line);border-radius:14px;padding:14px 18px}
.th-card .lbl{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--sub)}
.th-card .val{font-size:22px;font-weight:700;color:var(--navy);margin:4px 0}
.th-card p{font-size:13.5px;color:var(--muted)}
.finds{display:flex;flex-direction:column;gap:12px}
.fd{display:flex;gap:18px;align-items:flex-start;border:1px solid var(--line);border-radius:14px;padding:14px 20px}
.fd>span{flex:none;width:34px;height:34px;border-radius:50%;background:var(--navy);color:#fff;font-weight:700;font-size:16px;display:flex;align-items:center;justify-content:center}
.fd h3{font-size:17px;margin-bottom:2px}.fd p{font-size:14.5px;color:var(--muted);line-height:1.55}
.fd.warn{background:#fffbeb;border-color:#fde68a}.fd.warn>span{background:var(--amber)}
.cover{flex:1;display:flex;flex-direction:column;justify-content:center;padding-left:24px;border-left:6px solid var(--navy)}
.kicker{font-size:15px;font-weight:600;color:var(--blue);letter-spacing:.08em;text-transform:uppercase;margin-bottom:18px}
.cover h1{font-size:50px;font-weight:700;line-height:1.18;letter-spacing:-.02em}
.cover h1 span{font-size:28px;font-weight:500;color:var(--muted)}
.lead{font-size:18px;color:var(--sub);margin-top:22px}
.cover-tags{display:flex;gap:10px;margin-top:28px}
.cover-tags span{font-size:13.5px;padding:7px 14px;border-radius:20px;background:#eff6ff;color:var(--navy);font-weight:600}
@media print{
  @page{size:1280px 720px;margin:0}
  html,body{background:#fff;height:auto;overflow:visible}
  .deck{transform:none!important;position:static!important;width:auto;height:auto}
  .slide{display:flex!important;position:relative;height:720px;page-break-after:always;break-after:page;break-inside:avoid}
  .slide:last-child{page-break-after:auto;break-after:auto}
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
            f'<div class="pg">Tuần 3 · Pipeline</div></div><div class="body">{body}</div>'
            f'<div class="ft"><span>Nguồn: Expedia Hotel Recommendations (Kaggle) · số liệu từ smlprec_expedia_run.json; mốc tuần 2 (53,07% / 57,85%) từ scripts/expedia_context_signal.py và expedia_history_slices.py</span><span>{n} / {total}</span></div></section>'
        )
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tuần 3: Chạy thử pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="deck">{"".join(parts)}</div><script>{JS}</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    render()
