"""Figures for the week-3 implementation report (Vietnamese), same visual style as
scripts/methodology_figures.py and scripts/week2_report_figures.py.

Reads only persisted outputs in results/week3_implementation/:
- smlprec_expedia_run.json (plain run + heuristics on the same test rows, notebook 01)
- smlprec_expedia_late_fusion.json (destination prior fusion, notebook 03)
- smlprec_expedia_hybrid_samedest.json (sameDest hybrid, notebook 04)

Usage: python scripts/week3_report_figures.py
Output: results/figures/vi/week3_fig1_* ... week3_fig4_*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methodology_figures import (
    BASELINE,
    BLUE,
    GRID,
    INK2,
    SURFACE,
    base_axes,
    finish,
    fmt_pct,
    num,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "vi"
W3 = ROOT / "results" / "week3_implementation"
RUN = json.loads((W3 / "smlprec_expedia_run.json").read_text(encoding="utf-8"))
LF = json.loads((W3 / "smlprec_expedia_late_fusion.json").read_text(encoding="utf-8"))
HY = json.loads((W3 / "smlprec_expedia_hybrid_samedest.json").read_text(encoding="utf-8"))
TEAL = "#0f766e"
NAVY = "#1e3a8a"
GRAY = "#a8a79f"
AMBER = "#b45309"
LANG = "vi"

K_HYBRID = "SMLP4Rec + prior + sameDest (this notebook)"
K_REF = HY["reference"]["name"]
FKEY = f"fusion_global_w={LF['best_global_w_on_valid']}"
LV = LF["variants"]["test"]


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_dec(d: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda x, _pos=None: num(x, LANG, d))


def pct(v: float, d: int = 1) -> str:
    return num(v * 100, LANG, d) + "%"


def dec(v: float, d: int = 3) -> str:
    return num(v, LANG, d)


def bar_labels(ax, bars, vals, fmt) -> None:
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt(v),
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=INK2,
        )


def fig1_plain_vs_heuristics() -> None:
    """Plain SMLP4Rec vs the two count heuristics on the same test rows."""
    pop = RUN["heuristics_same_test_rows"]["global_popularity"]
    rep = RUN["heuristics_same_test_rows"]["repeat_last_then_popularity"]
    plain = RUN["test"]
    metrics = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20"]
    labels = ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]
    series = [
        ("Popularity toàn cục", pop, GRAY),
        ("Lặp cụm gần nhất, rồi popularity", rep, BLUE),
        (f"SMLP4Rec plain (epoch {RUN['best_epoch_by_valid']})", plain, NAVY),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.9), facecolor=SURFACE)
    base_axes(ax)
    w = 0.26
    for i, (name, d, col) in enumerate(series):
        xs = [j + (i - 1) * w for j in range(len(metrics))]
        vals = [d[m] for m in metrics]
        bars = ax.bar(xs, vals, width=w * 0.92, color=col, label=name, linewidth=0)
        bar_labels(ax, bars, vals, lambda v: dec(v, 2))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(fmt_dec(1))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=1)
    finish(
        fig,
        ax,
        "SMLP4Rec plain so với hai luật đếm, tập test",
        f"{thousands(plain_n())} dòng test, xếp hạng đầy đủ 100 cụm. Plain chỉ dùng lịch sử booking.",
        OUT / "week3_fig1_plain_vs_heuristics.png",
        -0.13,
    )


def plain_n() -> int:
    return RUN["heuristics_same_test_rows"]["global_popularity"]["n"]


def fig2_prior_sweep() -> None:
    """NDCG@5 and Recall@5 against the prior weight w_p (w_p = 0 is plain)."""
    grid = [str(w) for w in LF["w_grid"]]
    sw = LF["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), facecolor=SURFACE)
    for ax, m, name in zip(axes, ("ndcg@5", "recall@5"), ("NDCG@5", "Recall@5")):
        base_axes(ax)
        x = list(range(len(grid)))
        for split, col, lab in (("valid", BLUE, "Valid"), ("test", TEAL, "Test")):
            vals = [sw[split][w]["all"][m] for w in grid]
            ax.plot(x, vals, color=col, linewidth=2, marker="o", markersize=3.2, label=lab)
        best = grid.index(str(LF["best_global_w_on_valid"]))
        ax.axvline(best, color=BASELINE, linewidth=0.9, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(".", ",") for g in grid], fontsize=7)
        ax.set_xlabel("w_p (trọng số prior)", color=INK2, fontsize=8)
        ax.set_title(name, color=INK2, fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.yaxis.set_major_formatter(fmt_pct(LANG) if m == "recall@5" else fmt_dec(2))
        ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    fig.suptitle(
        "Thêm prior điểm đến: điểm theo trọng số w_p",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.04,
        f"w_p = 0 là SMLP4Rec plain. Đường đứt: w_p = {num(LF['best_global_w_on_valid'], LANG, 1)} chọn trên valid. Dòng có lịch sử (L ≥ 1).",
        fontsize=7.6,
        color="#898781",
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "week3_fig2_prior_weight_sweep.png",
        dpi=220,
        facecolor=SURFACE,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)


def fig3_chain() -> None:
    """NDCG@5 / Recall@5 on warm test rows along the experiment chain."""
    rows = [
        ("Popularity toàn cục", RUN["heuristics_same_test_rows"]["global_popularity"], GRAY),
        ("SMLP4Rec plain", LV["plain_model (w=0)"]["all"], NAVY),
        ("Prior điểm đến một mình", LV["destination_prior_only"]["all"], GRAY),
        ("Luật đếm tuần 2 (sameDest → prior)", LV["same_dest_history_then_prior"]["all"], AMBER),
        ("SMLP4Rec + prior", LV[FKEY]["all"], BLUE),
        ("Hybrid: + sameDest", HY["benchmark"]["test"]["warm"][K_HYBRID], TEAL),
    ][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), facecolor=SURFACE, sharey=True)
    for ax, m, name in zip(axes, ("ndcg@5", "recall@5"), ("NDCG@5", "Recall@5")):
        base_axes(ax)
        vals = [d[m] for _, d, _ in rows]
        y = range(len(rows))
        ax.barh(y, vals, color=[c for *_, c in rows], height=0.66, linewidth=0)
        for yi, v in zip(y, vals):
            ax.text(
                v + max(vals) * 0.015,
                yi,
                dec(v, 3) if m == "ndcg@5" else pct(v),
                va="center",
                fontsize=7,
                color=INK2,
            )
        ax.set_yticks(list(y))
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(name, color=INK2, fontsize=9, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(fmt_pct(LANG) if m == "recall@5" else fmt_dec(1))
    fig.suptitle(
        "Chuỗi thử nghiệm: NDCG@5 và Recall@5 trên tập test",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.03,
        f"{thousands(plain_n())} dòng test có lịch sử (L ≥ 1). Xám: tham chiếu không học; cam: luật đếm của tuần 2.",
        fontsize=7.6,
        color="#898781",
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "week3_fig3_experiment_chain.png",
        dpi=220,
        facecolor=SURFACE,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)


def fig4_slices() -> None:
    """Hybrid vs SMLP4Rec + prior, NDCG@5 per test slice."""
    sl = HY["test_slices_recall@5_ndcg@5"]
    names = [
        ("known destination", "Điểm đến đã đặt"),
        ("new destination", "Điểm đến mới"),
        ("target already booked", "Đặt lại cụm cũ"),
        ("target new to the user", "Đặt cụm mới"),
        ("L=1", "L = 1"),
        ("L=2-4", "L = 2–4"),
        ("L=5-9", "L = 5–9"),
        ("L>=10", "L ≥ 10"),
    ][::-1]

    def ndcg(key: str, k: str) -> float:
        return float(sl[key][k].split(" / ")[1])

    fig, ax = plt.subplots(figsize=(6.6, 3.6), facecolor=SURFACE)
    base_axes(ax)
    h = 0.38
    for i, (k, lab, col) in enumerate(
        ((K_HYBRID, "Hybrid: + sameDest", TEAL), (K_REF, "SMLP4Rec + prior", BLUE))
    ):
        ys = [j + (0.5 - i) * h for j in range(len(names))]
        vals = [ndcg(n, k) for n, _ in names]
        ax.barh(ys, vals, height=h * 0.9, color=col, label=lab, linewidth=0)
        for yv, v in zip(ys, vals):
            ax.text(v + 0.008, yv, dec(v, 3), va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [f"{lab} ({thousands(sl[n]['rows'])} dòng)" for n, lab in names], fontsize=7.4
    )
    ax.set_xlim(0, 0.88)
    ax.xaxis.set_major_formatter(fmt_dec(1))
    ax.set_xlabel("NDCG@5", color=INK2, fontsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=7.4, loc="lower right")
    finish(
        fig,
        ax,
        "Hybrid so với SMLP4Rec + prior theo lát cắt, tập test",
        "L = số booking trước đó của user. sameDest chỉ tác động khi điểm đến đang tìm đã có trong lịch sử.",
        OUT / "week3_fig4_slices.png",
        -0.16,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_plain_vs_heuristics()
    fig2_prior_sweep()
    fig3_chain()
    fig4_slices()
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
