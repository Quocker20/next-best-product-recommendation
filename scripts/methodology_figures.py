"""Generate the six figures used by the methodology selection report.

Reads only the persisted Phase-1 profiling outputs in reports/summary/ so the
figures cannot drift from the numbers quoted in the report text. Emits one PNG
set per language into reports/figures/<lang>/.

Usage:
    python scripts/methodology_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "summary"
OUTROOT = ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_PALE = "#9ec5f4"
ORANGE = "#eb6834"

LABELS = {
    "en": {
        "fig1_title": "Booking share of each hotel cluster, ranked",
        "fig1_x": "Cluster rank (1–100)",
        "fig1_y": "Share of bookings",
        "fig1_top": "rank 1: 4.03%",
        "fig1_bot": "rank 100: 0.08%",
        "fig1_note": "Top 10 clusters hold 22.87% of bookings; the lowest 50 hold 25.43%.",
        "fig2_title": "Cumulative booking share by destination rank band",
        "fig2_x": "Share of bookings",
        "fig2_bands": ["Top 10 destinations", "Top 100", "Top 1,000", "All 36,933"],
        "fig2_note": "10,789 destinations (29.2%) carry exactly one booking.",
        "fig3_title": "Distribution of bookings per user",
        "fig3_x": "Bookings per user",
        "fig3_y": "Share of booking users",
        "fig3_buckets": ["1", "2", "3–4", "5–10", "11–20", "21+"],
        "fig3_note": "Median 2 bookings per user. 40.57% of users reach three or more.",
        "fig4_title": "Divergence of the cluster distribution across context values",
        "fig4_x": "Mean Jensen–Shannon divergence (bits)",
        "fig4_fields": [
            "Search destination",
            "Hotel market",
            "Package flag",
            "Party composition",
            "Check-in month",
        ],
        "fig4_note": "Geography separates the cluster distribution; calendar and party composition barely do.",
        "fig5_title": "Recall@5 of a frequency table under different conditioning variables",
        "fig5_x": "Recall@5 on the temporal test split",
        "fig5_rules": [
            "Destination × package",
            "Search destination",
            "Destination × party",
            "Destination × check-in month",
            "Hotel market",
            "Repeat last cluster",
            "Party composition",
            "Global frequency",
            "Check-in month",
        ],
        "fig5_note": "Full ranking over 100 clusters, single run, no tuning.",
        "fig6_title": "Monthly booking volume with the temporal split boundary",
        "fig6_y": "Bookings per month",
        "fig6_train": "Training period",
        "fig6_test": "Test period",
        "fig6_note": "Volume grows 92.9% year over year; the test window covers the highest-volume quarter.",
    },
    "vi": {
        "fig1_title": "Tỷ trọng booking của từng hotel cluster, xếp theo thứ hạng",
        "fig1_x": "Thứ hạng cluster (1–100)",
        "fig1_y": "Tỷ trọng booking",
        "fig1_top": "hạng 1: 4,03%",
        "fig1_bot": "hạng 100: 0,08%",
        "fig1_note": "10 cluster hàng đầu nắm 22,87% booking; 50 cluster thấp nhất nắm 25,43%.",
        "fig2_title": "Tỷ trọng booking tích lũy theo dải thứ hạng điểm đến",
        "fig2_x": "Tỷ trọng booking",
        "fig2_bands": ["Top 10 điểm đến", "Top 100", "Top 1.000", "Toàn bộ 36.933"],
        "fig2_note": "10.789 điểm đến (29,2%) chỉ có đúng một booking.",
        "fig3_title": "Phân phối số booking trên mỗi user",
        "fig3_x": "Số booking mỗi user",
        "fig3_y": "Tỷ trọng user có booking",
        "fig3_buckets": ["1", "2", "3–4", "5–10", "11–20", "21+"],
        "fig3_note": "Trung vị 2 booking mỗi user. 40,57% user đạt từ ba booking trở lên.",
        "fig4_title": "Độ phân kỳ của phân phối cluster theo từng giá trị ngữ cảnh",
        "fig4_x": "Jensen–Shannon divergence trung bình (bit)",
        "fig4_fields": [
            "Điểm đến tìm kiếm",
            "Hotel market",
            "Cờ combo",
            "Thành phần đoàn khách",
            "Tháng nhận phòng",
        ],
        "fig4_note": "Yếu tố địa lý phân tách phân phối cluster; lịch và đoàn khách gần như không.",
        "fig5_title": "Recall@5 của bảng tần suất theo từng biến điều kiện",
        "fig5_x": "Recall@5 trên tập test chia theo thời gian",
        "fig5_rules": [
            "Điểm đến × combo",
            "Điểm đến tìm kiếm",
            "Điểm đến × đoàn khách",
            "Điểm đến × tháng nhận phòng",
            "Hotel market",
            "Lặp cluster gần nhất",
            "Thành phần đoàn khách",
            "Tần suất toàn cục",
            "Tháng nhận phòng",
        ],
        "fig5_note": "Xếp hạng toàn bộ 100 cluster, chạy một lần, không tinh chỉnh.",
        "fig6_title": "Khối lượng booking theo tháng và ranh giới chia dữ liệu theo thời gian",
        "fig6_y": "Booking mỗi tháng",
        "fig6_train": "Giai đoạn train",
        "fig6_test": "Giai đoạn test",
        "fig6_note": "Khối lượng tăng 92,9% so với cùng kỳ; cửa sổ test rơi vào quý có khối lượng lớn nhất.",
    },
}


def load(name: str) -> dict:
    with (SUMMARY / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def fmt_pct(lang: str):
    def _f(x, _pos=None):
        s = f"{x * 100:.0f}%"
        return s.replace(".", ",") if lang == "vi" else s

    return FuncFormatter(_f)


def num(value: float, lang: str, decimals: int = 2) -> str:
    s = f"{value:.{decimals}f}"
    return s.replace(".", ",") if lang == "vi" else s


def base_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def finish(fig, ax, title: str, note: str, path: Path, note_y: float) -> None:
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=12)
    if note:
        ax.text(0, note_y, note, color=MUTED, fontsize=7.6, ha="left", va="top",
                transform=ax.transAxes)
    fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def fig1(detail: dict, L: dict, lang: str, out: Path) -> None:
    shares = detail["cluster_popularity"]["share_sorted_desc"]
    fig, ax = plt.subplots(figsize=(6.3, 2.5), facecolor=SURFACE)
    base_axes(ax)
    ranks = range(1, len(shares) + 1)
    ax.bar(ranks, shares, width=0.72, color=BLUE, linewidth=0)
    ax.yaxis.set_major_formatter(fmt_pct(lang))
    ax.set_ylim(0, max(shares) * 1.28)
    ax.set_xlim(0.2, len(shares) + 0.8)
    ax.set_xlabel(L["fig1_x"], color=INK2, fontsize=8.5, labelpad=6)
    ax.set_ylabel(L["fig1_y"], color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.annotate(
        L["fig1_top"], xy=(1, shares[0]), xytext=(5, shares[0] * 1.14),
        color=INK, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.8),
    )
    ax.annotate(
        L["fig1_bot"], xy=(100, shares[-1]), xytext=(78, max(shares) * 0.30),
        color=INK, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.8),
    )
    finish(fig, ax, L["fig1_title"], L["fig1_note"], out / "fig1_item_distribution.png", -0.32)


def fig2(detail: dict, L: dict, lang: str, out: Path) -> None:
    d = detail["destinations"]
    vals = [d["share_top10"], d["share_top100"], d["share_top1000"], 1.0]
    fig, ax = plt.subplots(figsize=(6.3, 2.1), facecolor=SURFACE)
    base_axes(ax)
    ypos = range(len(vals))
    colors = [BLUE, BLUE, BLUE, BLUE_PALE]
    ax.barh(list(ypos), vals, height=0.58, color=colors, linewidth=0)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(L["fig2_bands"], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(fmt_pct(lang))
    ax.set_xlim(0, 1.06)
    ax.set_xlabel(L["fig2_x"], color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.012, y, num(v * 100, lang, 2) + "%", va="center",
                fontsize=8, color=INK)
    finish(fig, ax, L["fig2_title"], L["fig2_note"], out / "fig2_destination_concentration.png", -0.34)


def fig3(detail: dict, L: dict, lang: str, out: Path) -> None:
    hist = detail["user_depth"]["histogram"]
    keys = ["1", "2", "3-4", "5-10", "11-20", "21+"]
    vals = [hist[k]["share"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.3, 2.3), facecolor=SURFACE)
    base_axes(ax)
    xpos = range(len(vals))
    colors = [ORANGE] + [BLUE] * (len(vals) - 1)
    ax.bar(list(xpos), vals, width=0.62, color=colors, linewidth=0)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(L["fig3_buckets"], fontsize=8.5)
    ax.yaxis.set_major_formatter(fmt_pct(lang))
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_xlabel(L["fig3_x"], color=INK2, fontsize=8.5, labelpad=6)
    ax.set_ylabel(L["fig3_y"], color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for x, v in zip(xpos, vals):
        ax.text(x, v + max(vals) * 0.03, num(v * 100, lang, 2) + "%", ha="center",
                fontsize=8, color=INK)
    finish(fig, ax, L["fig3_title"], L["fig3_note"], out / "fig3_user_depth.png", -0.34)


def fig4(signal: dict, L: dict, lang: str, out: Path) -> None:
    shift = signal["distribution_shift"]
    vals = [
        shift["srch_destination_top"]["js_mean"],
        shift["hotel_market_top"]["js_mean"],
        shift["is_package"]["js_mean"],
        shift["party"]["js_mean"],
        shift["checkin_month"]["js_mean"],
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.2), facecolor=SURFACE)
    base_axes(ax)
    ypos = range(len(vals))
    ax.barh(list(ypos), vals, height=0.56, color=BLUE, linewidth=0)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(L["fig4_fields"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel(L["fig4_x"], color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.012, y, num(v, lang, 4), va="center",
                fontsize=8, color=INK)
    finish(fig, ax, L["fig4_title"], L["fig4_note"], out / "fig4_context_divergence.png", -0.33)


def fig5(signal: dict, L: dict, lang: str, out: Path) -> None:
    r = signal["popularity_probe"]["recall_at_5"]
    order = [
        "popularity_by_destination_x_is_package",
        "popularity_by_srch_destination_id",
        "popularity_by_destination_x_party",
        "popularity_by_destination_x_ci_month",
        "popularity_by_hotel_market",
        "repeat_last_cluster",
        "popularity_by_party",
        "global_popularity",
        "popularity_by_checkin_month",
    ]
    vals = [r[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.3, 2.8), facecolor=SURFACE)
    base_axes(ax)
    ypos = range(len(vals))
    colors = [BLUE if v >= 0.4 else BLUE_PALE for v in vals]
    ax.barh(list(ypos), vals, height=0.6, color=colors, linewidth=0)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(L["fig5_rules"], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(fmt_pct(lang))
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel(L["fig5_x"], color=INK2, fontsize=8.5, labelpad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.008, y, num(v * 100, lang, 2) + "%", va="center",
                fontsize=8, color=INK)
    finish(fig, ax, L["fig5_title"], L["fig5_note"], out / "fig5_recall_by_rule.png", -0.26)


def fig6(detail: dict, L: dict, lang: str, out: Path) -> None:
    monthly = detail["monthly_bookings"]
    months = sorted(monthly)
    vals = [monthly[m] for m in months]
    cut_idx = months.index("2014-09") + 0.9  # split falls at the end of September 2014
    fig, ax = plt.subplots(figsize=(6.3, 2.4), facecolor=SURFACE)
    base_axes(ax)
    x = list(range(len(months)))
    ax.axvspan(cut_idx, len(months) - 0.5, color=BLUE_PALE, alpha=0.30, linewidth=0)
    ax.plot(x, vals, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.set_xticks(x[::3])
    ax.set_xticklabels([months[i] for i in x[::3]], fontsize=8)
    ax.set_ylabel(L["fig6_y"], color=INK2, fontsize=8.5, labelpad=6)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_xlim(-0.5, len(months) - 0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{int(v / 1000)}k" if v else "0")
    )
    ax.axvline(cut_idx, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.text(cut_idx - 0.4, max(vals) * 1.10, L["fig6_train"], ha="right",
            fontsize=8, color=INK2)
    ax.text(cut_idx + 0.4, max(vals) * 1.10, L["fig6_test"], ha="left",
            fontsize=8, color=INK2)
    finish(fig, ax, L["fig6_title"], L["fig6_note"], out / "fig6_monthly_volume.png", -0.22)


def main() -> None:
    detail = load("expedia_eda_detail.json")
    signal = load("expedia_context_signal.json")
    for lang, L in LABELS.items():
        out = OUTROOT / lang
        out.mkdir(parents=True, exist_ok=True)
        fig1(detail, L, lang, out)
        fig2(detail, L, lang, out)
        fig3(detail, L, lang, out)
        fig4(signal, L, lang, out)
        fig5(signal, L, lang, out)
        fig6(detail, L, lang, out)
        print(f"wrote 6 figures to {out}")


if __name__ == "__main__":
    main()
