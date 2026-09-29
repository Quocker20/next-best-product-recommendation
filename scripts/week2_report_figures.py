"""Figures added to the methodology selection report for the week-2 revision
(SMLP4Rec + cold-start hybrid). Vietnamese only, same visual style as
scripts/methodology_figures.py (fig1-fig6, which the report also reuses).

Reads only persisted outputs:
- reports/summary/week1_dataset_selection/expedia_eda_detail.json
- reports/summary/week2_methodology/expedia_history_slices.json
Diagrams (architecture, hybrid flow) are rendered from HTML with headless Chrome.

Usage: python scripts/week2_report_figures.py
Output: reports/figures/vi/fig7_* ... fig13_*
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methodology_figures import (
    BASELINE,
    BLUE,
    BLUE_PALE,
    GRID,
    INK,
    INK2,
    ORANGE,
    SURFACE,
    base_axes,
    finish,
    fmt_pct,
    num,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "vi"
EDA = json.loads(
    (
        ROOT / "reports/summary/week1_dataset_selection/expedia_eda_detail.json"
    ).read_text()
)
HS = json.loads(
    (ROOT / "reports/summary/week2_methodology/expedia_history_slices.json").read_text()
)
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
TEAL = "#0f766e"
NAVY = "#1e3a8a"
RED = "#dc2626"
BUCKETS = ["0", "1", "2", "3-4", "5-9", "10+"]
BUCKET_LABELS = ["0", "1", "2", "3–4", "5–9", "10+"]
LANG = "vi"


def pct(v: float, d: int = 2) -> str:
    return num(v * 100, LANG, d) + "%"


def fig7_package_share() -> None:
    share = EDA["monthly_package_share"]
    months = sorted(share)
    vals = [share[m] for m in months]
    fig, ax = plt.subplots(figsize=(6.3, 2.3), facecolor=SURFACE)
    base_axes(ax)
    x = list(range(len(months)))
    ax.plot(x, vals, color=BLUE, linewidth=2)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([months[i] for i in x[::3]], fontsize=8)
    ax.yaxis.set_major_formatter(fmt_pct(LANG))
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xlim(-0.5, len(months) - 0.5)
    ax.set_ylabel("Tỷ trọng booking package", color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for m in ("2014-01", "2014-12"):
        i = months.index(m)
        ax.annotate(
            pct(share[m]),
            xy=(i, share[m]),
            xytext=(i, share[m] + max(vals) * 0.12),
            ha="center",
            fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.8},
        )
    finish(
        fig,
        ax,
        "Tỷ trọng booking package theo tháng",
        "Hành vi thay đổi theo thời gian: validation phải nằm ngay trước cửa sổ test.",
        OUT / "fig7_package_share.png",
        -0.24,
    )


def fig8_history_buckets() -> None:
    on = HS["by_history_online"]
    vals = [on[b]["share"] for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(6.3, 2.4), facecolor=SURFACE)
    base_axes(ax)
    xpos = list(range(len(vals)))
    ax.bar(xpos, vals, width=0.62, color=[ORANGE, "#f2a07c"] + [BLUE] * 4, linewidth=0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(BUCKET_LABELS, fontsize=8.5)
    ax.yaxis.set_major_formatter(fmt_pct(LANG))
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xlabel("Số booking trước đó của user", color=INK2, fontsize=8.5, labelpad=6)
    ax.set_ylabel("Tỷ trọng booking test", color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for x, v in zip(xpos, vals):
        ax.text(x, v + max(vals) * 0.03, pct(v), ha="center", fontsize=8, color=INK)
    le1 = on["0"]["share"] + on["1"]["share"]
    train0 = HS["by_history_train_only"]["0"]["share"]
    finish(
        fig,
        ax,
        "Booking test theo số booking trước đó của user",
        f"{pct(le1)} booking test có tối đa 1 booking trước đó. Nếu chỉ tính lịch sử trong train, "
        f"{pct(train0)} thuộc user chưa từng xuất hiện.",
        OUT / "fig8_history_buckets.png",
        -0.3,
    )


def fig9_rules_by_history() -> None:
    on = HS["by_history_online"]
    series = [
        (
            "Cluster từng đặt ở cùng điểm đến xếp trước, còn lại theo prior",
            "R@5_same_dest_history_then_dest_smooth20",
            TEAL,
            "-",
        ),
        (
            "Chỉ prior: tần suất cluster theo điểm đến",
            "R@5_dest_raw_backoff_market",
            NAVY,
            "-",
        ),
        (
            "Mọi cluster từng đặt xếp trước, còn lại theo prior",
            "R@5_history_then_dest",
            RED,
            (0, (4, 2)),
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.0), facecolor=SURFACE)
    base_axes(ax)
    x = list(range(len(BUCKETS)))
    for label, key, color, ls in series:
        vals = [on[b][key] for b in BUCKETS]
        ax.plot(
            x,
            vals,
            color=color,
            linewidth=2,
            linestyle=ls,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            label=label,
        )
        ax.text(
            x[-1] + 0.12,
            vals[-1],
            pct(vals[-1], 1),
            va="center",
            fontsize=8,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS, fontsize=8.5)
    ax.yaxis.set_major_formatter(fmt_pct(LANG))
    ax.set_ylim(0.30, 0.74)
    ax.set_xlim(-0.3, len(x) - 0.3)
    ax.set_xlabel("Số booking trước đó của user", color=INK2, fontsize=8.5, labelpad=6)
    ax.set_ylabel("Recall@5", color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    finish(
        fig,
        ax,
        "Recall@5 của 3 luật đếm theo số booking trước đó của user",
        "Luật xếp hạng bằng bảng đếm, không học tham số; chưa phải kết quả của SMLP4Rec hay hybrid.",
        OUT / "fig9_rules_by_history.png",
        -0.2,
    )


def fig10_dest_support() -> None:
    sup = HS["by_dest_support"]
    keys = ["0", "1-4", "5-19", "20-99", "100-999", "1000+"]
    labels = ["0", "1–4", "5–19", "20–99", "100–999", "1000+"]
    raw = [sup[k]["R@5_dest_raw_backoff_market"] for k in keys]
    smooth = [sup[k]["R@5_dest_smooth_m5"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.3, 2.6), facecolor=SURFACE)
    base_axes(ax)
    x = list(range(len(keys)))
    w = 0.36
    ax.bar(
        [i - w / 2 for i in x],
        raw,
        width=w,
        color=BLUE_PALE,
        label="Prior thô",
        linewidth=0,
    )
    ax.bar(
        [i + w / 2 for i in x],
        smooth,
        width=w,
        color=BLUE,
        label="Prior làm mượt về market (m = 5)",
        linewidth=0,
    )
    for i in x:
        ax.text(
            i + w / 2,
            smooth[i] + 0.015,
            pct(smooth[i], 1),
            ha="center",
            fontsize=7.5,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lab}\n{pct(sup[k]['share'], 1)} test" for lab, k in zip(labels, keys)],
        fontsize=8,
    )
    ax.yaxis.set_major_formatter(fmt_pct(LANG))
    ax.set_ylim(0, 0.95)
    ax.set_xlabel(
        "Số booking của điểm đến trong train", color=INK2, fontsize=8.5, labelpad=6
    )
    ax.set_ylabel("Recall@5", color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    finish(
        fig,
        ax,
        "Recall@5 của prior theo điểm đến, thô và làm mượt",
        "Làm mượt chỉ có tác dụng với điểm đến dưới 20 booking; điểm đến chưa từng thấy dùng phân phối chung.",
        OUT / "fig10_dest_support_smoothing.png",
        -0.34,
    )


def fig11_training_rows() -> None:
    n = HS["n_train"]
    rows = [
        ("SMLP4Rec\n1 mẫu / booking", n, TEAL),
        ("LightGBM\n1 dòng / booking, 100 cây mỗi vòng", n, BLUE_PALE),
        (
            "AdaGIN\n1 dòng / (booking, cluster)",
            HS["candidate_rows_train_x100"],
            ORANGE,
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.2), facecolor=SURFACE)
    base_axes(ax)
    y = list(range(len(rows)))
    ax.barh(
        y, [r[1] for r in rows], height=0.56, color=[r[2] for r in rows], linewidth=0
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, HS["candidate_rows_train_x100"] * 1.32)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v / 1e6:.0f} tr" if v else "0")
    )
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for yy, r in zip(y, rows):
        ax.text(
            r[1] + HS["candidate_rows_train_x100"] * 0.015,
            yy,
            f"{r[1]:,}".replace(",", "."),
            va="center",
            fontsize=8,
            color=INK,
        )
    finish(
        fig,
        ax,
        "Số dòng huấn luyện trên tập train",
        "AdaGIN chấm điểm từng cặp (lượt tìm kiếm, cluster) nên cần gấp 100 lần số dòng, trước khi lấy mẫu âm.",
        OUT / "fig11_training_rows.png",
        -0.24,
    )


FLOW_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#fcfcfb;font-family:'Segoe UI',system-ui,sans-serif;width:1240px;height:260px;display:flex;align-items:center;justify-content:center}
.flow{display:flex;align-items:center;gap:14px}.col{display:flex;flex-direction:column;gap:16px}
.b{border-radius:12px;padding:14px 18px;text-align:center;font-size:17px;font-weight:600;min-width:190px;white-space:nowrap;border:1.5px solid}
.b small{display:block;font-weight:500;font-size:13px;margin-top:3px;opacity:.85}
.in{background:#f8fafc;border-color:#cbd5e1;color:#334155}.m{background:#eef2ff;border-color:#6366f1;color:#3730a3}
.p{background:#ecfdf5;border-color:#0f766e;color:#115e59}.g{background:#fffbeb;border-color:#d97706;color:#92400e}
.o{background:#1e3a8a;border-color:#1e3a8a;color:#fff}.a{font-size:28px;color:#94a3b8}
</style></head><body><div class="flow">
<div class="col"><div class="b in">Lịch sử booking<br>+ query</div><div class="b in">Điểm đến<br>đang tìm</div></div>
<div class="a">→</div>
<div class="col"><div class="b m">SMLP4Rec<small>điểm tuần tự</small></div><div class="b p">Prior theo ngữ cảnh<small>điểm đến → market → toàn cục</small></div></div>
<div class="a">→</div><div class="b g">× Gate<small>theo độ dài lịch sử &amp;<br>độ phổ biến điểm đến</small></div>
<div class="a">→</div><div class="b o">Cộng điểm<br>→ Top-K / 100 cluster</div>
</div></body></html>"""


def chrome_png(html: str, width: int, height: int, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=2",
                f"--screenshot={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )


def diagrams() -> None:
    from week2_methodology_slides import architecture_svg

    arch = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>body{margin:0;background:#fcfcfb;"
        "font-family:'Segoe UI',system-ui,sans-serif}.d{font-size:13px;fill:#334155;text-anchor:middle}"
        ".dt{font-size:15px;font-weight:600}.dm{fill:#64748b;font-size:12.5px}</style></head><body>"
        f"{architecture_svg()}</body></html>"
    )
    chrome_png(arch, 900, 500, OUT / "fig12_smlp4rec_architecture.png")
    chrome_png(FLOW_HTML, 1240, 260, OUT / "fig13_hybrid_flow.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig7_package_share()
    fig8_history_buckets()
    fig9_rules_by_history()
    fig10_dest_support()
    fig11_training_rows()
    diagrams()
    print(f"wrote fig7-fig13 to {OUT}")


if __name__ == "__main__":
    main()
