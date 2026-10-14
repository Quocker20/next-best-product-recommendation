"""Week-6 deck v2 (Vietnamese HTML, 16:9), rebuilt from the review notes.

Does not touch `slides_week6_vi.html`; reads it only to inherit the model diagrams and the appendix
slides. Every number is read from `results/week6_retrain/` (user-level test, seed 2024) or from
`eda_all_bookings.json` (whole booking table, `scripts/eda_all_bookings.py`).

Output: reports/summary/week6_retrain/slides_week6_v2_vi.html
Keys: arrows / space, Home, End, `n` toggles speaker notes. Print gives one 1280x720 page per slide.

Usage: .venv/Scripts/python.exe scripts/week6_slides_v2.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "week6_retrain"
OLD = ROOT / "reports" / "summary" / "week6_retrain" / "slides_week6_vi.html"
OUT = ROOT / "reports" / "summary" / "week6_retrain" / "slides_week6_v2_vi.html"

# palette: light teal ground, a few clear accents; text ink only, never in chart marks
BG, INK, MUTED, RULE = "#f6fbfa", "#1d2f36", "#4f6a6b", "#d3e3e0"
TEAL, TEAL_D, TEAL_L = "#12907f", "#0b6e62", "#46c2b2"
BLUE, OCHRE, CORAL, GREEN, GRAY = "#4a93cf", "#e0a03c", "#e2735c", "#5aa95a", "#a6bdba"
SECTIONS = {  # slide class: (accent, dark, tint, label)
    "sA": (TEAL, TEAL_D, "#d9f0ec", "Bài toán"),
    "sB": (BLUE, "#2d6ea3", "#dcebf7", "Dữ liệu"),
    "sC": (OCHRE, "#a8680f", "#fbeccd", "Hướng tiếp cận"),
    "sD": (CORAL, "#b5472f", "#fbe1da", "Mô hình nền"),
    "sE": (TEAL, TEAL_D, "#d9f0ec", "Cải tiến 1"),
    "sF": (GREEN, "#3b7c3b", "#e0f1dc", "Cải tiến 2"),
    "sG": (TEAL_D, TEAL_D, "#d9f0ec", "Kết luận"),
}
MODEL_COLOR = {"popularity": GRAY, "itemknn": BLUE, "plain": OCHRE, "query": TEAL_L, "main": TEAL_D}
MODEL_NAME = {
    "popularity": "Độ phổ biến",
    "itemknn": "ItemKNN",
    "plain": "SMLP4Rec gốc",
    "query": "+ query embedding",
    "main": "+ sameDest",
}


def esc(s: object) -> str:
    return html.escape(str(s), quote=False)


def num(n: float) -> str:
    return f"{int(round(n)):,}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def f4(x: float) -> str:
    return f"{x:.4f}"


def sg4(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


# ----------------------------------------------------------------------------------------------
# svg helpers
# ----------------------------------------------------------------------------------------------
def T(x, y, s, size=18, fill=INK, anchor="start", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{s}</text>'
    )


def R(x, y, w, h, fill="none", stroke="none", sw=1.5, dash=False, rx=0):
    da = ' stroke-dasharray="6 5"' if dash else ""
    r = f' rx="{rx}"' if rx else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{da}{r}/>'
    )


def L(x1, y1, x2, y2, stroke=MUTED, sw=1.5, arrow=False, dash=False):
    mk = ' marker-end="url(#arr)"' if arrow else ""
    da = ' stroke-dasharray="5 5"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}"{mk}{da}/>'
    )


def C(x, y, r, fill):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'


_SVG_N = [0]


def svg(w: int, h: int, inner: str, label: str, color: str = MUTED) -> str:
    _SVG_N[0] += 1
    mid = f"arr{_SVG_N[0]}"
    defs = (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>'
    )
    inner = inner.replace("url(#arr)", f"url(#{mid})")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{esc(label)}">{defs}{inner}</svg>'
    )


def wrap_label(label: str, y: float, size: int, fill: str, x: float) -> str:
    return "".join(T(x, y + i * (size + 3), part, size, fill, "middle") for i, part in enumerate(label.split("\n")))


def vbars(w, h, items, vmax, fmt, ticks=(), top=34, bottom=58, lsize=16, vsize=17, left=40, label="chart", tick_fmt=None):
    """Vertical bars; items: dict label, value, color, ghost. Returns (svg, tops) with bar x/top for arrows."""
    n = len(items)
    plot_h = h - top - bottom
    slot = (w - left) / n
    bw = slot * 0.62
    out = [L(left, h - bottom, w, h - bottom, RULE, 1)]
    for t in ticks:
        y = h - bottom - plot_h * t / vmax
        out.append(L(left, y, w, y, RULE, 0.6, dash=True))
        out.append(T(left - 6, y + 5, (tick_fmt or (lambda v: f"{v:g}"))(t), 13, MUTED, "end"))
    tops = []
    for i, it in enumerate(items):
        x = left + slot * i + (slot - bw) / 2
        cx = x + bw / 2
        if it.get("ghost"):
            out.append(R(x, h - bottom - 3, bw, 3, RULE))
            out.append(wrap_label(it["label"], h - bottom + 22, lsize, GRAY, cx))
            tops.append((cx, h - bottom))
            continue
        bh = plot_h * it["value"] / vmax
        out.append(R(x, h - bottom - bh, bw, bh, it["color"]))
        out.append(T(cx, h - bottom - bh - 8, fmt(it["value"]), vsize, INK, "middle", 600))
        out.append(wrap_label(it["label"], h - bottom + 22, lsize, INK, cx))
        tops.append((cx, h - bottom - bh - 30))
    return "".join(out), tops


def fmt_tick(t, vmax):
    return f"{t:.1f}" if vmax <= 1 else f"{t:.0f}"


def lift_arrow(x1, y1, x2, y2, text, color, size=19):
    """Arrow from bar A top to bar B top with a label pill above the middle."""
    ymid = min(y1, y2) - 36
    mx = (x1 + x2) / 2
    wlab = len(text) * size * 0.56 + 22
    ye = y2 - 6
    path = (
        f'<path d="M{x1:.1f},{y1:.1f} L{x1:.1f},{ymid:.1f} L{x2:.1f},{ymid:.1f} L{x2:.1f},{ye - 8:.1f}" fill="none" stroke="{color}" stroke-width="3"/>'
        f'<polygon points="{x2 - 8:.1f},{ye - 10:.1f} {x2 + 8:.1f},{ye - 10:.1f} {x2:.1f},{ye + 2:.1f}" fill="{color}"/>'
    )
    pill = R(mx - wlab / 2, ymid - 17, wlab, 34, color, rx=3) + T(mx, ymid + 7, text, size, "#ffffff", "middle", 700)
    return path + pill


def hbars(w, rows, vmax, fmt, label_w=250, bar_h=36, gap=16, top=6, vsize=18, lsize=18, label="chart"):
    out, y = [], top
    span = w - label_w - 150
    for r in rows:
        bw = max(3.0, span * r["value"] / vmax)
        out.append(T(label_w - 12, y + bar_h * 0.68, r["label"], lsize, INK, "end"))
        out.append(R(label_w, y, bw, bar_h, r["color"]))
        out.append(T(label_w + bw + 10, y + bar_h * 0.68, fmt(r["value"]), vsize, INK, "start", 600))
        y += bar_h + gap
    out.insert(0, L(label_w, top - 4, label_w, y - gap + 4, RULE, 1))
    return svg(w, int(y - gap + top + 4), "".join(out), label)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load():
    j = lambda n: json.loads((RES / n).read_text(encoding="utf-8"))  # noqa: E731
    return {
        "u": j("user_top20.json"),
        "sl": j("user_top20_slices.json"),
        "eda": j("eda_split.json"),
        "all": j("eda_all_bookings.json"),
    }


def metric(d, model, rs, m):
    return d["u"]["results"][model][rs][m]


# ----------------------------------------------------------------------------------------------
# slide registry
# ----------------------------------------------------------------------------------------------
SLIDES: list[dict] = []


def slide(title, body, notes="", sec="sA", ins="", kind="main", link=None):
    SLIDES.append({"title": title, "body": body, "notes": notes, "sec": sec, "ins": ins, "kind": kind})


def pane(title, svg_html, w=560):
    return f'<div class="pane" style="width:{w}px"><p class="ct">{title}</p>{svg_html}</div>'


# ----------------------------------------------------------------------------------------------
# diagrams (hand drawn)
# ----------------------------------------------------------------------------------------------
def fig_cover() -> str:
    cards = ""
    for i, (nm, sub, hi) in enumerate([("Phòng deluxe, view biển", "gói 3 đêm", True), ("Phòng gia đình", "gói bữa sáng", False), ("Phòng tiêu chuẩn", "không kèm gói", False)]):
        y = 110 + i * 96
        cards += R(40, y, 440, 82, "#ffffff", TEAL if hi else RULE, 2.5 if hi else 1.2)
        cards += R(52, y + 12, 58, 58, TEAL_L if hi else GRAY) + T(66, y + 50, str(i + 1), 28, "#ffffff", "start", 700)
        cards += T(126, y + 36, nm, 21, INK, "start", 600) + T(126, y + 62, sub, 17, MUTED)
        if hi:
            cards += R(396, y + 22, 72, 36, TEAL, rx=3) + T(432, y + 47, "top 1", 18, "#ffffff", "middle", 700)
    inner = (
        C(440, 40, 120, "#d3ece8") + C(520, 460, 150, "#dcebf7") + C(60, 470, 70, "#fbeccd")
        + R(40, 24, 440, 62, "#ffffff", RULE, 1.5)
        + C(70, 55, 13, "none").replace('fill="none"', f'fill="none" stroke="{MUTED}" stroke-width="2.5"')
        + L(80, 65, 90, 75, MUTED, 2.5)
        + T(102, 62, "Phú Quốc · 2 người lớn · 3 đêm", 20, INK)
        + cards
        + T(40, 420, "một lần tìm kiếm", 18, MUTED) + L(40, 430, 480, 430, TEAL, 2.5)
    )
    return svg(520, 500, inner, "Minh họa: tìm kiếm và danh sách gợi ý")


def box(x, y, w, h, title, sub, fill, stroke, tsize=19):
    out = R(x, y, w, h, fill, stroke, 1.4)
    if sub:
        out += T(x + w / 2, y + h / 2 - 2, title, tsize, INK, "middle", 600) + T(x + w / 2, y + h / 2 + 22, sub, 16, MUTED, "middle")
    else:
        out += T(x + w / 2, y + h / 2 + 7, title, tsize, INK, "middle", 600)
    return out


def fig_flow() -> str:
    cols = ["Tìm kiếm", "Xem danh sách", "Lọc, so sánh", "Chọn hạng phòng", "Đặt phòng"]
    guest = ["điểm đến, ngày,\nsố khách", "cuộn nhiều\nkhách sạn", "đổi bộ lọc,\nmở nhiều trang", "hạng phòng\nhoặc gói", "thanh toán"]
    sysm = ["nhận tham số\ntìm kiếm", "danh sách chung\ntheo bộ lọc", "ghi lượt xem", "hiện giá\nvà gói", "ghi booking\nvào lịch sử"]
    pain_g, pain_s = {1, 2}, {1}
    x0, cw, gap = 176, 180, 18
    out = [T(2, 118, "Khách", 22, "#2d6ea3", "start", 700), T(2, 308, "Bên bán", 22, "#a8680f", "start", 700)]
    out.insert(0, R(0, 70, 1152, 120, "#dcebf7") + R(0, 262, 1152, 120, "#fbeccd"))
    for i, c in enumerate(cols):
        x = x0 + i * (cw + gap)
        out.append(T(x + cw / 2, 40, c, 20, INK, "middle", 700))
        if i < 4:
            out.append(L(x + cw + 2, 130, x + cw + gap - 2, 130, MUTED, 2, arrow=True))
            out.append(L(x + cw + 2, 322, x + cw + gap - 2, 322, MUTED, 2, arrow=True))
        for lane, (txts, y0, pset) in enumerate(((guest, 84, pain_g), (sysm, 276, pain_s))):
            a, b = txts[i].split("\n") if "\n" in txts[i] else (txts[i], "")
            out.append(R(x, y0 + 8, cw, 76, "#ffffff", CORAL if i in pset else RULE, 2.5 if i in pset else 1.2))
            if b:
                out.append(T(x + cw / 2, y0 + 42, a, 18, INK, "middle") + T(x + cw / 2, y0 + 66, b, 18, INK, "middle"))
            else:
                out.append(T(x + cw / 2, y0 + 54, a, 18, INK, "middle"))
            if i in pset:
                out.append(C(x + cw - 6, y0 + 12, 13, CORAL) + T(x + cw - 6, y0 + 18, "!", 18, "#ffffff", "middle", 700))
        out.append(L(x + cw / 2, 168, x + cw / 2, 276, GRAY, 1.2, dash=True))
    out.append(T(x0, 432, "!  điểm khách và bên bán gặp khó", 17, CORAL, "start", 600))
    return svg(1152, 452, "".join(out), "Hành trình đặt phòng hiện tại")


def fig_pain() -> str:
    def panel(x, title, sub, color, tint, rows):
        out = [R(x, 0, 560, 410, tint), R(x, 0, 560, 64, color), T(x + 24, 28, title, 24, "#ffffff", "start", 700), T(x + 24, 52, sub, 16, "#ffffff")]
        for i, (key, rest) in enumerate(rows):
            y = 92 + i * 104
            out.append(C(x + 42, y + 30, 26, color) + T(x + 42, y + 40, str(i + 1), 26, "#ffffff", "middle", 700))
            out.append(T(x + 84, y + 28, key, 24, INK, "start", 700) + T(x + 84, y + 58, rest, 19, MUTED))
            if i < 2:
                out.append(L(x + 24, y + 86, x + 536, y + 86, "#ffffff", 2))
        return "".join(out)

    left = panel(0, "Khách lưu trú", "chọn hạng phòng, gói", BLUE, "#dcebf7",
                 [("Quá nhiều lựa chọn", "nhiều khách sạn, hạng phòng, gói"), ("Tốn thời gian", "lọc, so sánh nhiều trang"), ("Chọn sai", "mất tiền, hỏng cả chuyến đi")])
    right = panel(592, "Bên bán lưu trú", "bán đúng hạng phòng, gói", OCHRE, "#fbeccd",
                  [("Không biết khách cần gì", "chỉ thấy tham số tìm kiếm"), ("Bỏ lỡ cơ hội bán", "hạng phòng, gói phù hợp"), ("Lịch sử khách mỏng", "39.34% khách chỉ đặt một lần")])
    return svg(1152, 410, left + right, "Pain point của khách và bên bán lưu trú")


def fig_solution() -> str:
    outs = ""
    for i in range(5):
        outs += R(824, 20 + i * 78, 300, 62, TEAL if i == 0 else "#ffffff", TEAL if i == 0 else RULE, 1.4)
        outs += T(848, 60 + i * 78, f"{i + 1}. hạng phòng hoặc gói", 21, "#ffffff" if i == 0 else INK, "start", 600 if i == 0 else 400)
    inner = (
        R(0, 120, 330, 130, "#dcebf7", BLUE, 1.6)
        + T(20, 160, "Lần tìm kiếm hiện tại", 22, INK, "start", 700)
        + T(20, 192, "điểm đến, ngày, nhóm khách,", 18, MUTED) + T(20, 216, "gói, kênh bán", 18, MUTED)
        + R(0, 280, 330, 70, "none", GRAY, 1.6, dash=True)
        + T(20, 322, "vài booking cũ (thường không có)", 18, MUTED)
        + L(332, 185, 440, 185, MUTED, 2, arrow=True) + L(332, 315, 380, 315, MUTED, 2) + L(380, 315, 380, 215, MUTED, 2, arrow=True)
        + R(442, 120, 250, 130, TEAL, rx=0) + T(567, 178, "Hệ gợi ý", 30, "#ffffff", "middle", 700) + T(567, 212, "xếp hạng 100 cluster", 18, "#ffffff", "middle")
        + L(694, 185, 820, 185, MUTED, 2, arrow=True)
        + outs
    )
    return svg(1152, 410, inner, "Hệ gợi ý: đầu vào và đầu ra")


def fig_recsys() -> str:
    rng = np.random.default_rng(7)
    cells = ""
    for r in range(8):
        for c in range(10):
            on = rng.random() < 0.28
            cells += R(10 + c * 32, 10 + r * 32, 28, 28, TEAL_L if on else "#e6f1ef")
    ranked = ""
    for i in range(4):
        ranked += R(810, 20 + i * 62, 320, 50, TEAL if i == 0 else "#ffffff", TEAL if i == 0 else RULE, 1.4) + T(832, 52 + i * 62, f"sản phẩm {i + 1}", 21, "#ffffff" if i == 0 else INK, "start", 600 if i == 0 else 400)
    doms = [("Xem phim", "user – phim", GRAY, False), ("Mua sắm", "user – sản phẩm", GRAY, False), ("Du lịch nghỉ dưỡng", "user – hạng phòng, gói", TEAL, True)]
    dom = ""
    for i, (a, b, col, hi) in enumerate(doms):
        x = i * 386
        dom += R(x, 320, 366, 90, "#d9f0ec" if hi else "#ffffff", col, 2.5 if hi else 1.2) + T(x + 22, 360, a, 24, INK, "start", 700) + T(x + 22, 390, b, 19, MUTED)
    inner = (
        T(0, 8, "", 1)
        + f'<g transform="translate(0,6)">{cells}</g>'
        + T(165, 290, "tương tác user – item", 18, MUTED, "middle")
        + L(350, 140, 450, 140, MUTED, 2, arrow=True)
        + R(452, 90, 280, 100, "#dcebf7", BLUE, 1.6) + T(592, 150, "Mô hình gợi ý", 26, INK, "middle", 700)
        + L(734, 140, 806, 140, MUTED, 2, arrow=True)
        + ranked + T(970, 275, "danh sách xếp hạng", 18, MUTED, "middle")
        + dom
    )
    return svg(1152, 410, inner, "Hệ gợi ý chung và ba lĩnh vực")


def card3(items) -> str:
    out = []
    for i, (big, head, sub, col, tint) in enumerate(items):
        x = i * 392
        out.append(R(x, 0, 368, 410, tint))
        out.append(R(x, 0, 368, 10, col))
        out.append(T(x + 28, 140, big, 64 if len(big) < 8 else 44, col, "start", 700))
        out.append(T(x + 28, 214, head, 28, INK, "start", 700))
        for k, line in enumerate(sub.replace("|", "\n").split("\n")):
            out.append(T(x + 28, 254 + k * 28, line, 21, MUTED))
    out = [o for o in out]
    return svg(1152, 410, "".join(out), "Ba đặc điểm của bài toán du lịch")


def kpi_slide_svg(items) -> str:
    out = []
    n = len(items)
    cw = (1152 - 24 * (n - 1)) / n
    for i, (big, head, sub, col, tint) in enumerate(items):
        x = i * (cw + 24)
        out.append(R(x, 0, cw, 410, tint) + R(x, 0, cw, 10, col))
        out.append(T(x + 26, 150, big, 56, col, "start", 700))
        out.append(T(x + 26, 214, head, 25, INK, "start", 700))
        for k, line in enumerate(sub.split("\n")):
            out.append(T(x + 26, 252 + k * 27, line, 20, MUTED))
    return svg(1152, 410, "".join(out), "Ba kết quả chính")


# ----------------------------------------------------------------------------------------------
# result charts (fixed geometry so slides line up)
# ----------------------------------------------------------------------------------------------
LADDER = ["popularity", "itemknn", "plain", "query", "main"]
LADDER_LABEL = {"popularity": "Độ phổ\nbiến", "itemknn": "ItemKNN", "plain": "SMLP4Rec\ngốc", "query": "+ query\nembedding", "main": "+ sameDest\n(chính)"}


def ladder(d, stage: int, m: str, vmax: float, ticks, fmt) -> str:
    """Five fixed slots; slots beyond `stage` are ghosts. Lift arrow ends at the newest bar."""
    items = []
    for i, k in enumerate(LADDER):
        if i < stage:
            items.append({"label": LADDER_LABEL[k], "value": metric(d, k, "all", m), "color": MODEL_COLOR[k]})
        else:
            items.append({"label": LADDER_LABEL[k], "value": 0, "color": GRAY, "ghost": True})
    body, tops = vbars(560, 400, items, vmax, fmt, ticks, top=70, bottom=60, lsize=15, vsize=16)
    prev_k, new_k = LADDER[stage - 2], LADDER[stage - 1]
    a, b = metric(d, prev_k, "all", m), metric(d, new_k, "all", m)
    txt = f"{b - a:+.4f} ({100 * (b / a - 1):+.1f}%)".replace("-", "−") if m.startswith("ndcg") else f"{100 * (b / a - 1):+.1f}%"
    arrow = lift_arrow(tops[stage - 2][0], tops[stage - 2][1], tops[stage - 1][0], tops[stage - 1][1], txt, OCHRE if stage == 3 else MODEL_COLOR[new_k] if stage != 4 else TEAL, 17)
    return svg(560, 400, body + arrow, f"{m} theo mô hình")


def result_panes(d, stage: int) -> str:
    n = ladder(d, stage, "ndcg@10", 0.52, (0, 0.25, 0.5), f4)
    # recall panel: labels as percentages
    r_items = []
    for i, k in enumerate(LADDER):
        if i < stage:
            r_items.append({"label": LADDER_LABEL[k], "value": metric(d, k, "all", "recall@10"), "color": MODEL_COLOR[k]})
        else:
            r_items.append({"label": LADDER_LABEL[k], "value": 0, "color": GRAY, "ghost": True})
    body, tops = vbars(560, 400, r_items, 0.9, lambda v: f"{100 * v:.1f}%", (0, 0.45, 0.9), top=70, bottom=60, lsize=15, vsize=16, tick_fmt=lambda t: f"{100 * t:.0f}%")
    prev_k, new_k = LADDER[stage - 2], LADDER[stage - 1]
    a, b = metric(d, prev_k, "all", "recall@10"), metric(d, new_k, "all", "recall@10")
    arrow = lift_arrow(tops[stage - 2][0], tops[stage - 2][1], tops[stage - 1][0], tops[stage - 1][1], f"{100 * (b / a - 1):+.1f}%".replace("-", "−"), OCHRE if stage == 3 else TEAL, 17)
    rec = svg(560, 400, body + arrow, "recall@10 theo mô hình")
    return (
        '<div class="g2">'
        + pane("NDCG@10 trên tập test, trung bình theo user", n)
        + pane("Recall@10 trên tập test, trung bình theo user", rec)
        + "</div>"
    )


def group_vbars(w, h, groups, series, vmax, fmt, ticks, bottom=44, top=34, lsize=17, vsize=15, left=40):
    """groups: labels; series: (name, color, values). Bars side by side per group."""
    n, k = len(groups), len(series)
    slot = (w - left) / n
    bw = slot * 0.8 / k
    plot_h = h - top - bottom
    out = [L(left, h - bottom, w, h - bottom, RULE, 1)]
    for t in ticks:
        y = h - bottom - plot_h * t / vmax
        out.append(L(left, y, w, y, RULE, 0.6, dash=True))
        out.append(T(left - 6, y + 5, f"{t:.2f}", 13, MUTED, "end"))
    for gi, g in enumerate(groups):
        gx = left + slot * gi + slot * 0.1
        for si, (_, col, vals) in enumerate(series):
            bh = plot_h * vals[gi] / vmax
            x = gx + si * bw
            out.append(R(x, h - bottom - bh, bw - 2, bh, col))
            out.append(T(x + bw / 2 - 1, h - bottom - bh - 6, fmt(vals[gi]), vsize, INK, "middle", 600))
        out.append(T(left + slot * (gi + 0.5), h - bottom + 26, g, lsize, INK, "middle", 600))
    return "".join(out)


def legend(items, x=0, y=14, size=16):
    out, cx = [], x
    for name, col in items:
        out.append(R(cx, y - 12, 16, 16, col) + T(cx + 22, y + 2, name, size, INK))
        cx += 38 + len(name) * size * 0.52
    return "".join(out)


# ----------------------------------------------------------------------------------------------
# old-deck pieces
# ----------------------------------------------------------------------------------------------
def old_parts():
    h = OLD.read_text(encoding="utf-8")
    secs = re.split(r'(?=<section class="slide)', h)[1:]
    secs = [x[: x.index("</section>") + len("</section>")] for x in secs]
    byn = {re.search(r'data-n="([^"]+)"', s).group(1): s for s in secs}
    return byn


def first_svg(sec: str) -> str:
    return re.search(r"<svg[\s\S]*?</svg>", sec).group(0)


# ----------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------
def build(d):
    a = d["all"]
    old = old_parts()
    mo = a["monthly_bookings"]
    avg13 = np.mean([v for k, v in mo.items() if k.startswith("2013")])
    avg14 = np.mean([v for k, v in mo.items() if k.startswith("2014")])
    win = d["eda"]["S1"]["windows"]
    u = d["u"]
    n_test, n_warm, n_cold = u["info"]["users"], u["info"]["warm_users"], u["info"]["cold_users"]
    sl = d["sl"]
    qn = lambda k, rs="all", m="ndcg@10": metric(d, k, rs, m)  # noqa: E731

    # 1 cover
    slide(
        "",
        f'<div class="cov"><div class="covl"><p class="slogan">Gợi ý đúng hạng phòng ngay khi khách bắt đầu tìm</p>'
        '<p class="covsub">Next-best-product recommendation cho du lịch nghỉ dưỡng</p></div>'
        f'<div class="covr">{fig_cover()}</div></div>',
        "Bìa. Đề tài C1. Toàn bộ thực nghiệm dùng dữ liệu công khai Expedia Hotel Recommendations; không dùng dữ liệu Vinpearl hay GSM.",
        kind="title",
    )

    # A
    slide(
        "Hành trình đặt phòng hiện tại",
        f'<div class="stage">{fig_flow()}</div>',
        "Luồng chung của một lần đặt phòng online: khách nhập tìm kiếm, xem danh sách, lọc và so sánh, chọn hạng phòng, đặt. Phía bên bán nhận tham số tìm kiếm, hiện danh sách chung, ghi booking. Hai điểm khó của khách (cuộn, lọc) và một điểm khó của bên bán (danh sách chung, không cá nhân hóa).",
        ins="Danh sách hiển thị <b>chung</b> cho mọi khách, <b>không cá nhân hóa</b>",
    )
    slide(
        "Pain point",
        f'<div class="stage">{fig_pain()}</div>',
        f"Pain point hai phía. Số 39.34% lấy từ toàn bộ {num(a['bookings'])} booking của {num(a['users'])} user (eda_all_bookings.json).",
        ins="Hai phía cùng cần <b>gợi ý đúng</b> ngay lúc khách tìm kiếm",
    )
    slide(
        "Giải pháp: hệ gợi ý",
        f'<div class="stage">{fig_solution()}</div>',
        "Đầu vào là lần tìm kiếm hiện tại và vài booking cũ (thường không có). Đầu ra là danh sách xếp hạng 100 hotel_cluster, id ẩn danh của Expedia đóng vai hạng phòng hoặc gói.",
        ins="Đầu ra: <b>top 5</b> trong <b>100 hotel_cluster</b> (hạng phòng hoặc gói)",
    )
    slide(
        "Hệ gợi ý",
        f'<div class="stage">{fig_recsys()}</div>',
        "Hệ gợi ý học từ tương tác user và item rồi xếp hạng item cho từng user. Ba lĩnh vực minh họa; đề tài thuộc du lịch nghỉ dưỡng.",
        ins="<b>Next-best-product</b>: gợi ý sản phẩm kế tiếp cho từng khách",
    )
    DOMAIN_CARDS = card3(
        [
            (pct(a["bookings_per_user"]["share_1"], 2), "Mua thưa", "user chỉ đặt một lần", BLUE, "#dcebf7"),
            ("14", "Ngữ cảnh quyết định", "trường tìm kiếm mô tả|chuyến đi", TEAL, "#d9f0ec"),
            ("Đắt", "Chọn sai", "tốn tiền, tốn thời gian,|ảnh hưởng cả chuyến đi", CORAL, "#fbe1da"),
        ]
    )
    slide(
        "Bài toán du lịch",
        '<div class="stage">' + DOMAIN_CARDS + "</div>",
        "Ba đặc điểm của gợi ý du lịch khách sạn: tần suất mua thấp nên lịch sử mỏng; ngữ cảnh chuyến đi quyết định lựa chọn; chi phí chọn sai cao.",
        ins="Lịch sử <b>thưa</b>, ngữ cảnh <b>quyết định</b>, sai thì <b>đắt</b>",
    )

    # B dataset
    kp = lambda big, lab, col: f'<div class="kp"><b style="color:{col}">{big}</b><span>{lab}</span></div>'  # noqa: E731
    fields = (
        '<div class="fgrp"><div class="fg" style="background:#dcebf7"><h4>Đầu vào: lần tìm kiếm</h4>'
        "<p>điểm đến · ngày check-in, check-out · người lớn, trẻ em, phòng · gói · kênh · thiết bị</p></div>"
        '<div class="fg" style="background:#d9f0ec"><h4>Nhãn</h4><p><b>hotel_cluster</b>, 100 giá trị</p></div>'
        '<div class="fg" style="background:#f0f0ee"><h4>Không dùng làm đầu vào</h4><p>thông tin khách sạn đã đặt</p></div></div>'
    )
    slide(
        "Bộ dữ liệu Expedia",
        '<div class="stage"><div class="kps">'
        + kp(num(a["bookings"]), "booking", TEAL) + kp(num(a["users"]), "user", BLUE)
        + kp(num(a["destinations"]["n"]), "điểm đến", OCHRE) + kp("100", "cluster", CORAL) + "</div>"
        + fields + "</div>",
        "Expedia Hotel Recommendations, công khai. Booking từ 2013-01-07 đến 2014-12-31. Hotel_cluster là id ẩn danh, không có cột giá. Giới hạn: một dataset, đánh giá offline.",
        sec="sB", ins="Mỗi dòng là <b>một booking</b> kèm lần tìm kiếm dẫn tới booking đó",
    )

    # EDA 1
    m_keys = list(mo)
    items = [{"label": "", "value": mo[k], "color": BLUE if k.startswith("2013") else TEAL} for k in m_keys]
    vb, _ = vbars(360, 360, items, 200_000, lambda v: "", (0, 100_000, 200_000), top=14, bottom=50, left=44, tick_fmt=lambda t: f"{t // 1000:.0f}k")
    # month labels
    ml = "".join(T(44 + (360 - 44) / 24 * (i + 0.5), 360 - 50 + 22, lb, 13, MUTED, "middle") for i, lb in ((0, "1/13"), (6, "7/13"), (12, "1/14"), (18, "7/14")))
    ml += "".join(T(2, 14 + (360 - 64) * (1 - t / 200_000) + 4, f"{t // 1000}k", 12, MUTED, "start") for t in ())
    ch1 = svg(360, 360, vb.replace("</text>", "</text>") + ml, "Booking theo tháng đặt")
    cm = a["checkin_month_share"]
    peak = max(cm, key=cm.get)
    items = [{"label": str(m), "value": cm[str(m)], "color": CORAL if str(m) == peak else GRAY} for m in range(1, 13)]
    vb, _ = vbars(360, 360, items, 0.12, lambda v: "", (0, 0.06, 0.12), top=14, bottom=50, lsize=13, left=44, tick_fmt=lambda t: f"{100 * t:.0f}%")
    ch2 = svg(360, 360, vb, "Booking theo tháng check-in")
    ps = a["party_share"]
    names = {"couple": "couple", "solo": "solo", "family": "family", "group": "group"}
    rows = [{"label": names[k], "value": ps[k], "color": TEAL if k == "couple" else GRAY} for k in names]
    ch3 = hbars(360, rows, 0.5, lambda v: pct(v, 1), label_w=90, bar_h=40, gap=30, vsize=16, lsize=17)
    slide(
        "EDA 1: quy mô và thời gian",
        '<div class="stage"><div class="g3">'
        + pane("Số booking theo tháng đặt phòng (booking date), 2013–2014", ch1, 360)
        + pane("Tỷ lệ booking theo tháng check-in (ngày nhận phòng)", ch2, 360)
        + pane("Tỷ lệ booking theo nhóm khách", ch3, 360) + "</div></div>",
        f"Trung bình {num(avg13)} booking mỗi tháng đặt năm 2013, {num(avg14)} năm 2014. Check-in cao nhất tháng {peak} ({pct(cm[peak], 2)}). Nhóm khách: couple có 2 người lớn, solo 1, family có trẻ em, group từ 3 người lớn. Nguồn: eda_all_bookings.json.",
        sec="sB", ins=f"Booking mỗi tháng tăng <b>{avg14 / avg13:.1f} lần</b> từ 2013 sang 2014; check-in cao nhất tháng <b>{peak}</b>",
    )

    # EDA 2
    bpu = a["bookings_per_user"]
    rows = [
        {"label": "1 booking", "value": bpu["share_1"], "color": TEAL},
        {"label": "2 booking", "value": bpu["share_2"], "color": GRAY},
        {"label": "3 đến 4", "value": bpu["share_3_4"], "color": GRAY},
        {"label": "5 đến 9", "value": bpu["share_5_9"], "color": GRAY},
        {"label": "từ 10", "value": bpu["share_10_plus"], "color": GRAY},
    ]
    e2a = hbars(560, rows, 0.45, lambda v: pct(v, 1), label_w=130, bar_h=44, gap=26, vsize=18)
    gb = a["gap_buckets"]
    gl = [("le_1h", "≤ 1 giờ"), ("1h_1d", "1 giờ–1 ngày"), ("1d_7d", "1–7 ngày"), ("7d_30d", "7–30 ngày"), ("30d_90d", "30–90 ngày"), ("gt_90d", "> 90 ngày")]
    rows = [{"label": lb, "value": gb[k], "color": BLUE if k == "le_1h" else GRAY} for k, lb in gl]
    e2b = hbars(560, rows, 0.30, lambda v: pct(v, 1), label_w=150, bar_h=38, gap=18, vsize=18)
    slide(
        "EDA 2: hành vi user",
        '<div class="stage"><div class="g2">'
        + pane("Tỷ lệ user theo số booking của user (toàn bộ dữ liệu)", e2a)
        + pane("Tỷ lệ cặp booking liên tiếp theo khoảng cách thời gian giữa hai booking", e2b)
        + "</div></div>",
        f"Trung vị {bpu['median']:.0f} booking mỗi user; trung bình {bpu['mean']:.2f}. Khoảng cách trung vị giữa hai booking liên tiếp {a['gap_days']['median']:.1f} ngày, {pct(a['gap_days']['share_within_1h'], 2)} cách nhau tối đa một giờ. Nguồn: eda_all_bookings.json.",
        sec="sB", ins=f"Lịch sử <b>thưa</b>: <b>{pct(bpu['share_1'], 2)}</b> user chỉ có một booking, trung vị {bpu['median']:.0f}",
    )

    # EDA 3
    share = np.array(a["clusters"]["share_by_rank"])
    plot_h = 340 - 14 - 44
    out = [L(44, 340 - 44, 560, 340 - 44, RULE, 1)]
    step = (560 - 44) / len(share)
    for t in (0, 0.02, 0.04):
        y = 340 - 44 - plot_h * t / 0.045
        out.append(L(44, y, 560, y, RULE, 0.6, dash=True)) if t else None
        out.append(T(38, y + 5, f"{100 * t:.0f}%", 13, MUTED, "end"))
    for i, v in enumerate(share):
        bh = plot_h * v / 0.045
        out.append(R(44 + i * step + 0.5, 340 - 44 - bh, step - 1, bh, CORAL if i < 10 else GRAY))
    for r in (1, 10, 50, 100):
        out.append(T(44 + (r - 0.5) * step - (6 if r == 100 else 0), 340 - 44 + 22, str(r), 13, MUTED, "middle"))
    out.append(T(302, 336, "hạng cluster", 14, MUTED, "middle"))
    e3a = svg(560, 340, "".join(out), "Tỷ lệ booking của từng cluster")
    r_, c_ = np.array(a["destination_curve"]["ranks"]), np.array(a["destination_curve"]["cum"])
    lx = lambda v: 44 + (560 - 64) * np.log10(v) / np.log10(r_.max())  # noqa: E731
    ly = lambda v: 340 - 44 - plot_h * v  # noqa: E731
    out = [L(44, 340 - 44, 560, 340 - 44, RULE, 1)]
    for t in (0, 0.5, 1.0):
        out.append(T(38, ly(t) + 5, f"{100 * t:.0f}%", 13, MUTED, "end"))
        if t:
            out.append(L(44, ly(t), 560, ly(t), RULE, 0.6, dash=True))
    pts = " ".join(f"{lx(x):.1f},{ly(y):.1f}" for x, y in zip(r_, c_))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>')
    for t, v in ((100, a["destinations"]["top100"]), (1000, a["destinations"]["top1000"])):
        out.append(C(lx(t), ly(v), 6, CORAL) + T(lx(t) + 12, ly(v) + 26, pct(v, 1), 18, CORAL, "start", 700))
    for t in (1, 10, 100, 1000, 10000):
        out.append(T(lx(t), 340 - 44 + 22, f"{t:,}", 13, MUTED, "middle"))
    out.append(T(302, 336, "hạng điểm đến (thang log)", 14, MUTED, "middle"))
    e3b = svg(560, 340, "".join(out), "Tỷ lệ booking tích lũy theo hạng điểm đến")
    slide(
        "EDA 3: điểm đến và cluster",
        '<div class="stage"><div class="g2">'
        + pane("Tỷ lệ booking của từng cluster, xếp theo hạng (10 cluster đầu tô màu)", e3a)
        + pane(f"Tỷ lệ booking tích lũy theo hạng điểm đến, {num(a['destinations']['n'])} điểm đến", e3b)
        + "</div></div>",
        f"Top 10 cluster chiếm {pct(a['clusters']['top10'], 1)} booking, top 1 chiếm {pct(a['clusters']['top1'], 1)}. Top 100 điểm đến chiếm {pct(a['destinations']['top100'], 1)}, top 1,000 chiếm {pct(a['destinations']['top1000'], 1)}. Nguồn: eda_all_bookings.json.",
        sec="sB", ins=f"Cluster <b>phân bố đều</b> (top 10 = {pct(a['clusters']['top10'], 1)}), điểm đến <b>đuôi dài</b> (top 1,000 = {pct(a['destinations']['top1000'], 1)})",
    )

    # split
    sets = [("train", "07/01/2013–31/08/2014"), ("valid", "01/09–31/10/2014"), ("test", "01/11–31/12/2014")]
    colset = {"train": GRAY, "valid": OCHRE, "test": TEAL}
    rows_b = [{"label": f"{n}\n{rng}", "value": win[n]["bookings"], "color": colset[n]} for n, rng in sets]
    rows_u = [{"label": f"{n}\n{rng}", "value": win[n]["users"], "color": colset[n]} for n, rng in sets]

    def hb2(rows, vmax, name):
        out, y = [], 6
        for r in rows:
            bw = 250 * r["value"] / vmax
            l1, l2 = r["label"].split("\n")
            out.append(T(172, y + 24, l1, 22, INK, "end", 700) + T(172, y + 50, l2, 15, MUTED, "end"))
            out.append(R(184, y, bw, 60, r["color"]))
            out.append(T(184 + bw + 8, y + 38, num(r["value"]), 19, INK, "start", 600))
            y += 100
        return svg(560, 300, "".join(out), name)

    slide(
        "Chia tập",
        '<div class="stage"><div class="g2">'
        + pane("Số booking mỗi tập, chia theo ngày đặt phòng (booking date, không phải ngày check-in)", hb2(rows_b, win["train"]["bookings"] * 1.0, "booking"), 560)
        + pane("Số user mỗi tập (một user có thể thuộc nhiều tập)", hb2(rows_u, win["train"]["users"] * 1.0, "user"), 560)
        + "</div></div>",
        f"Train {pct(win['train']['share'], 2)}, valid {pct(win['valid']['share'], 2)}, test {pct(win['test']['share'], 2)} booking. Tập user tính trên {num(a['users'])} user của cả dataset.",
        sec="sB", ins="Chia theo <b>ngày đặt phòng</b>: test là <b>hai tháng cuối</b> của dữ liệu",
    )

    # data problems
    hist = sl["history_length"]
    keys = [("0", "0 (cold)"), ("1", "1"), ("2-4", "2 đến 4"), ("5-9", "5 đến 9"), ("10+", "từ 10")]
    rows = [{"label": lb, "value": hist[k]["n"] / n_test, "color": CORAL if k == "0" else GRAY} for k, lb in keys]
    p2 = hbars(560, rows, 0.4, lambda v: pct(v, 2), label_w=130, bar_h=44, gap=22, vsize=18)
    vs = [("valid", win["valid"]["users_not_in_train_share"]), ("test", win["test"]["users_not_in_train_share"])]
    rows = [{"label": n, "value": v, "color": OCHRE} for n, v in vs]
    p1 = hbars(560, rows, 0.5, lambda v: pct(v, 2), label_w=110, bar_h=50, gap=30, vsize=20)
    slide(
        "Vấn đề của dữ liệu",
        '<div class="stage"><div class="g2">'
        + pane("Tỷ lệ user của tập valid, test chưa từng có booking ở train", p1)
        + pane("Tỷ lệ user test theo số booking trước ngày 01/11/2014 (cold: chưa có booking nào)", p2)
        + "</div></div>",
        f"Cold = user test chưa có booking trước ngày 2014-11-01, {num(n_cold)} user, {pct(n_cold / n_test, 2)} trong {num(n_test)} user test. Khác với 'chưa có ở train' ({pct(win['test']['users_not_in_train_share'], 2)}): nhóm này gồm cả user chỉ xuất hiện ở valid. Xử lý cold: query embedding đọc lần tìm kiếm hiện tại.",
        sec="sB", ins=f"<b>{pct(n_cold / n_test, 2)}</b> user test không có lịch sử (cold start): cần đọc <b>lần tìm kiếm hiện tại</b>",
    )

    # C approaches
    appr = [
        ("Content-based", "cần thuộc tính item; cluster ẩn danh", False),
        ("Collaborative filtering", "cần nhiều booking mỗi user; bỏ qua ngữ cảnh", False),
        ("Graph-based", "nặng tính toán; cần nhiều tương tác", False),
        ("Session-based", "ít booking cùng phiên", False),
        ("Sequential: SMLP4Rec", "dùng thứ tự booking; bỏ qua tìm kiếm hiện tại", True),
        ("Context-aware", "dùng ngữ cảnh tìm kiếm; chạy được khi chưa có lịch sử", True),
    ]
    rows = ""
    for nm, weak, pick in appr:
        rows += (
            f'<div class="ap{" pk" if pick else ""}"><b>{nm}</b><span>{weak}</span>'
            f'<em>{"Chọn" if pick else ""}</em></div>'
        )
    slide(
        "Hướng tiếp cận",
        f'<div class="stage"><div class="aps">{rows}</div></div>',
        "Sáu hướng phổ biến và hạn chế khi áp dụng cho dữ liệu du lịch khách sạn. Đề tài kết hợp sequential (thứ tự các booking) với context-aware (lần tìm kiếm hiện tại).",
        sec="sC", ins="Kết hợp <b>sequential</b> (lịch sử) và <b>context-aware</b> (tìm kiếm hiện tại)",
    )

    # D models
    slide(
        "MLP4Rec",
        f'<div class="stage" style="padding-top:60px">{first_svg(old["12"])}</div>',
        "MLP4Rec (Li et al., IJCAI 2022): mô hình tuần tự thuần MLP, không attention, độ phức tạp tuyến tính. Ba mixer (sequence, channel, feature) xếp nối tiếp, mỗi lần transpose để mixer nằm trên chiều cần trộn. Theo bài SMLP4Rec, cấu trúc nối tiếp và nhiều lớp chuẩn hoá giữa các mixer gây nhiễu và giảm hiệu quả. Không có số đo thời gian hay metric của MLP4Rec trong dự án.",
        sec="sD", ins="Hạn chế: mixer <b>nối tiếp</b>, nhiều lớp <b>chuẩn hoá</b> gây nhiễu",
    )
    slide(
        "SMLP4Rec",
        f'<div class="stage">{first_svg(old["13"])}</div>',
        "SMLP4Rec (Gao et al., TOIS 2024): giữ ba hướng trộn nhưng đổi sang cấu trúc song song và gộp lớp chuẩn hoá thành một Layer Norm trước ba mixer. Bản gốc chỉ dùng tối đa 20 booking trước và bỏ qua tìm kiếm hiện tại.",
        sec="sD", ins="Khắc phục: <b>song song</b>, <b>một</b> Layer Norm. Còn thiếu: <b>tìm kiếm hiện tại</b>",
    )
    slide(
        "SMLP4Rec gốc: kết quả",
        result_panes(d, 3) + '<p class="lk"><a href="#" onclick="ret=i;show({A2});return false">Cách test và eval</a></p>',
        f"Tập test {num(n_test)} user, mỗi user một danh sách top 20 cố định, trung bình theo user, seed 2024.",
        sec="sD", ins=f"Chỉ dùng lịch sử, SMLP4Rec gốc <b>ngang ItemKNN</b>: NDCG@10 {f4(qn('plain'))} so với {f4(qn('itemknn'))}",
    )

    # E
    slide(
        "Cải tiến 1: query embedding",
        f'<div class="stage">{first_svg(old["16"])}<p class="lk"><a href="#" onclick="ret=i;show({{A1}});return false">Phụ lục A1: 14 trường</a></p></div>',
        "Query embedding là một vị trí bổ sung sau lịch sử: vector mask học được cộng với tổng 14 embedding trường tìm kiếm. Booking đầu tiên vẫn có query embedding nên mô hình chạy được khi lịch sử rỗng.",
        sec="sE", ins="14 trường tìm kiếm thành <b>một vị trí</b> thêm vào chuỗi",
    )
    g_q = qn("query") / qn("plain") - 1
    slide(
        "Query embedding: kết quả",
        result_panes(d, 4),
        f"NDCG@10 từ {f4(qn('plain'))} lên {f4(qn('query'))}. Khoảng tin cậy 95% của mức tăng: "
        f"{sg4(sl['bootstrap_ndcg@10']['query_minus_plain']['ci95'][0])} đến {sg4(sl['bootstrap_ndcg@10']['query_minus_plain']['ci95'][1])}.",
        sec="sE", ins=f"Lần tìm kiếm hiện tại nâng NDCG@10 <b>{sg4(qn('query') - qn('plain'))} ({100 * g_q:+.1f}%)</b>",
    )
    cold, warm = sl["row_set"]["cold"], sl["row_set"]["warm"]
    ser = [("popularity", "Độ phổ biến"), ("itemknn", "ItemKNN"), ("plain", "SMLP4Rec gốc"), ("query", "+ query embedding")]
    series = [(nm, MODEL_COLOR[k], [cold[k]["ndcg@10"], warm[k]["ndcg@10"]]) for k, nm in ser]
    gsvg = svg(1152, 380, legend([(n, c) for n, c, _ in series], 0, 14) + f'<g transform="translate(0,30)">{group_vbars(1152, 350, [f"Cold ({num(n_cold)} user)", f"Warm ({num(n_warm)} user)"], series, 0.52, f4, (0, 0.25, 0.5), left=44)}</g>', "NDCG@10 cold và warm")
    slide(
        "Query embedding: user cold",
        f'<div class="stage"><p class="ct">NDCG@10 trên tập test, trung bình theo user, tách user cold (chưa có booking trước) và warm</p>{gsvg}</div>',
        "Cold: user test chưa có booking nào trước ngày 2014-11-01. Mô hình chỉ đọc lịch sử không có gì để xếp hạng, query embedding đọc lần tìm kiếm hiện tại.",
        sec="sE", ins=f"User cold: <b>{f4(cold['query']['ndcg@10'])}</b>, ItemKNN chỉ {f4(cold['itemknn']['ndcg@10'])}",
    )

    # F
    seen_d = a["repeat_given_history"]["destination_seen_before"]
    slide(
        "Cải tiến 2: sameDest",
        f'<div class="stage">{first_svg(old["18"])}<p class="lk"><a href="#" onclick="ret=i;show({{A3}});return false">Phụ lục A3: công thức</a></p></div>',
        "sameDest là số hạng late-fusion: tìm các cluster user đã đặt tại điểm đến đang tìm, cho booking gần đây trọng số cao hơn (0.7^lag), cộng log của tỷ lệ đó vào log xác suất của mô hình. Điểm đến mới và booking đầu tiên không bị ảnh hưởng.",
        sec="sF", ins=f"<b>{pct(seen_d, 1)}</b> booking có lịch sử quay lại <b>điểm đến đã từng đặt</b>",
    )
    bs = sl["bootstrap_ndcg@10"]["main_minus_query"]
    g_m = qn("main") / qn("query") - 1
    slide(
        "sameDest: kết quả",
        result_panes(d, 5),
        f"Mức tăng {sg4(bs['diff'])}, khoảng tin cậy 95% {sg4(bs['ci95'][0])} đến {sg4(bs['ci95'][1])}.",
        sec="sF", ins=f"sameDest thêm <b>{sg4(bs['diff'])} ({100 * g_m:+.1f}%)</b> NDCG@10, khoảng tin cậy 95% {sg4(bs['ci95'][0])} đến {sg4(bs['ci95'][1])}",
    )
    hb = [("0", "0"), ("1", "1"), ("2-4", "2 đến 4"), ("5-9", "5 đến 9"), ("10+", "từ 10")]
    items = []
    for k, lb in hb:
        diff = hist[k]["main"]["ndcg@10"] - hist[k]["query"]["ndcg@10"]
        items.append({"label": f"{lb}\nn={num(hist[k]['n'])}", "value": diff, "color": GREEN if k == "10+" else GRAY})
    vb, _ = vbars(560, 350, items, 0.02, lambda v: f"{v:+.4f}", (0, 0.01, 0.02), top=24, bottom=62, lsize=14, vsize=15, left=50)
    p_hist = svg(560, 350, vb.replace(">0.0<", ">0<"), "Mức tăng sameDest theo lịch sử")
    dk = sl["destination_warm"]
    GL_DEST = [f"Đã biết (n={num(dk['known']['n'])})", f"Mới (n={num(dk['new']['n'])})"]
    series = [("+ query embedding", TEAL_L, [dk["known"]["query"]["ndcg@10"], dk["new"]["query"]["ndcg@10"]]), ("+ sameDest", TEAL_D, [dk["known"]["main"]["ndcg@10"], dk["new"]["main"]["ndcg@10"]])]
    gk = svg(560, 350, legend([(n, c) for n, c, _ in series], 40, 14) + f'<g transform="translate(0,26)">{group_vbars(560, 324, GL_DEST, series, 0.65, lambda v: f"{v:.3f}", (0, 0.3, 0.6), left=44)}</g>', "Điểm đến đã biết và mới")
    known_gain = dk["known"]["main"]["ndcg@10"] - dk["known"]["query"]["ndcg@10"]
    known_pct = dk["known"]["main"]["ndcg@10"] / dk["known"]["query"]["ndcg@10"] - 1
    slide(
        "sameDest: giúp ở đâu",
        '<div class="stage"><div class="g2">'
        + pane("Mức tăng NDCG@10 của sameDest (chính − query embedding) theo số booking trước đó của user", p_hist)
        + pane("NDCG@10 của user warm theo điểm đến của lần tìm kiếm (đã từng đặt hoặc mới)", gk)
        + "</div></div>",
        f"Điểm đến đã biết chiếm {pct(sl['known_share_of_warm'], 1)} user warm. Điểm đến mới: {f4(dk['new']['query']['ndcg@10'])} và {f4(dk['new']['main']['ndcg@10'])}.",
        sec="sF", ins=f"sameDest chỉ giúp khi <b>quay lại điểm đến cũ</b>: {sg4(known_gain)} ({100 * known_pct:+.1f}%)",
    )

    # cost
    cost_rows = [
        {"label": "Huấn luyện 1 epoch, query embedding", "value": 1070.5, "color": GRAY},
        {"label": "Chấm điểm tập valid, query embedding", "value": 24.4, "color": GRAY},
        {"label": "sameDest, chấm điểm 222,428 user", "value": 1.1, "color": GREEN},
    ]
    cost = hbars(1152, cost_rows, 1100, lambda v: f"{v:,.1f} s", label_w=440, bar_h=64, gap=40, vsize=24, lsize=21)
    slide(
        "sameDest: chi phí",
        f'<div class="stage"><p class="ct">Thời gian trên CPU (giây). sameDest đo trên đầu vào mô phỏng cùng kích thước, không huấn luyện lại</p>{cost}</div>',
        "sameDest là hậu xử lý trên điểm của mô hình query embedding: mỗi user duyệt tối đa 20 booking và tính trên 100 cluster. Đo bằng scratchpad/bench_samedest.py: 1.1 giây cho 222,428 user (khoảng 4.8 micro giây mỗi user), gồm cả softmax. Thời gian huấn luyện và chấm điểm query embedding lấy từ results/week6_retrain/query_s2024.json (train_seconds, eval_seconds mỗi epoch).",
        sec="sF", ins="Lợi ích <b>+1.1%</b> toàn tập, <b>+6.5%</b> ở điểm đến đã biết; chi phí <b>~1 giây</b>, không huấn luyện lại",
    )

    # conclusion
    slide(
        "Những điều rút ra từ kết quả",
        '<div class="stage">'
        + kpi_slide_svg(
            [
                (f"{100 * g_q:+.0f}%", "Lần tìm kiếm hiện tại", f"query embedding: NDCG@10\n{f4(qn('plain'))} lên {f4(qn('query'))}", TEAL, "#d9f0ec"),
                (f"{100 * known_pct:+.1f}%", "Lịch sử cũ, đúng điểm đến", f"sameDest ở điểm đến đã biết,\nchi phí khoảng 1 giây", GREEN, "#e0f1dc"),
                (f4(cold["query"]["ndcg@10"]), "User cold vẫn xếp hạng tốt", f"NDCG@10, ItemKNN {f4(cold['itemknn']['ndcg@10'])}", BLUE, "#dcebf7"),
            ]
        )
        + "</div>",
        "Mô hình chính = SMLP4Rec + query embedding + sameDest. Lần tìm kiếm hiện tại là tín hiệu chính; lịch sử là phần bổ sung nhỏ hơn nhưng có thật, tập trung ở user quay lại điểm đến cũ.",
        sec="sG", ins="Mô hình chính: SMLP4Rec + <b>query embedding</b> + <b>sameDest</b>",
    )

    # appendix: reuse A1, A2, A2.1, A3 verbatim, add A4
    def ap(n, title):
        s = old[n]
        s = re.sub(r'<h1>.*?</h1>', f"<h1>{title}</h1>", s, count=1)
        s = re.sub(r'<span class="pg">[^<]*</span>', "", s)
        s = re.sub(r'<p class="src">[^<]*</p>', "", s)
        s = re.sub(r'data-n="[^"]+"', "", s, count=1)
        s = s.replace('class="slide appendix"', 'class="slide appendix sG"', 1)
        s = re.sub(r"Quay lại slide \d+", "Quay lại", s)
        return s

    for n, t in (("28", "Phụ lục A1. 14 trường query embedding"), ("29", "Phụ lục A2. Cách test và eval"), ("29.1", "Phụ lục A2.1. Hai cách gộp còn lại"), ("30", "Phụ lục A3. Công thức sameDest")):
        SLIDES.append({"raw": ap(n, t), "kind": "appendix"})
    # A4 robustness
    groups = [("season", [("Th.11-12", "T11–12"), ("Th.1-3", "T1–3"), ("Th.4-10", "T4–10")]), ("party", [(k, k) for k in ("solo", "couple", "family", "group")]), ("package", [("package", "có gói"), ("no_package", "không gói")])]
    rows = []
    for sec, ks in groups:
        for k, lb in ks:
            rows.append({"label": f"{lb}", "value": sl[sec][k]["main"]["ndcg@10"], "color": TEAL_D, "n": sl[sec][k]["n"]})
    rows_b = [{"label": f"{r['label']} (n={num(r['n'])})", "value": r["value"], "color": TEAL_D} for r in rows]
    a4 = hbars(1000, rows_b, 0.5, lambda v: f"{v:.4f}", label_w=300, bar_h=30, gap=10, vsize=17, lsize=17)
    SLIDES.append({"title": "Phụ lục A4. Mô hình chính theo mùa, nhóm khách, gói đặt", "body": f'<div class="stage"><p class="ct">NDCG@10 của mô hình chính trên tập test, theo thuộc tính của booking đầu tiên của user</p>{a4}</div>', "notes": "Mô hình chính dẫn đầu mọi nhóm; chênh lệch giữa các nhóm nhỏ.", "sec": "sG", "ins": "", "kind": "appendix"})


# ----------------------------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------------------------
CSS_EXTRA = """
:root{--bg:%(bg)s;--ink:%(ink)s;--muted:%(muted)s;--rule:%(rule)s;--main:%(teal)s;--main-dark:%(teald)s}
.sA{--sec:%(a0)s;--secd:%(a1)s;--tint:%(a2)s}.sB{--sec:%(b0)s;--secd:%(b1)s;--tint:%(b2)s}
.sC{--sec:%(c0)s;--secd:%(c1)s;--tint:%(c2)s}.sD{--sec:%(d0)s;--secd:%(d1)s;--tint:%(d2)s}
.sE{--sec:%(e0)s;--secd:%(e1)s;--tint:%(e2)s}.sF{--sec:%(f0)s;--secd:%(f1)s;--tint:%(f2)s}
.sG{--sec:%(g0)s;--secd:%(g1)s;--tint:%(g2)s}
.slide::before{content:"";position:absolute;left:0;top:0;right:0;height:10px;background:var(--sec)}
.slide.title::before{height:0}
h1{margin:6px 0 20px 0}
.stage{position:absolute;left:64px;top:112px;width:1152px;height:462px}
.ins{position:absolute;left:64px;right:64px;bottom:58px;height:60px;display:flex;align-items:center;padding:0 24px;
background:var(--tint);font-size:25px;color:var(--ink)}
.ins b{color:var(--secd)}
.sec{position:absolute;left:64px;bottom:20px;font-size:13px;color:var(--secd);font-weight:600;letter-spacing:.04em}
.sec::before{content:"";display:inline-block;width:10px;height:10px;background:var(--sec);margin-right:8px}
.g2{display:flex;gap:32px}.g3{display:flex;gap:36px}
.pane svg{display:block}
.ct{font-size:19px;line-height:1.25;color:var(--muted);margin:0 0 10px 0;min-height:48px}
.lk{margin:8px 0 0;font-size:18px}.lk a{color:var(--main-dark)}
.stage>.lk{position:absolute;right:0;bottom:-6px;margin:0}
.kps{display:flex;gap:24px;margin-bottom:28px}
.kp{flex:1;padding:34px 24px;background:#fff;border:1.5px solid var(--rule)}
.kp b{display:block;font-size:52px;line-height:1.1;font-variant-numeric:tabular-nums}
.kp span{display:block;font-size:21px;color:var(--muted);margin-top:4px}
.fgrp{display:flex;gap:24px}
.fg{flex:1;padding:26px 28px;min-height:230px}
.fg h4{margin:0 0 14px;font-size:25px}.fg p{margin:0;font-size:22px;color:var(--muted);line-height:1.4}
.fg p b{color:var(--ink)}
.aps{display:flex;flex-direction:column;gap:10px}
.ap{display:grid;grid-template-columns:330px 1fr 90px;align-items:center;gap:18px;padding:13px 22px;background:#fff;border:1.5px solid var(--rule);font-size:22px}
.ap b{font-weight:600}.ap span{color:var(--muted);font-size:20px}
.ap em{font-style:normal;font-weight:700;text-align:center;padding:4px 0}
.ap.pk{background:var(--tint);border-color:var(--sec)}
.ap.pk em{background:var(--sec);color:#fff}
.cov{display:flex;align-items:center;height:620px;gap:24px;margin-left:-0px}
.covl{width:660px}.covr{flex:1}
.slogan{font-size:48px;line-height:1.14;font-weight:700;margin:0 0 24px;letter-spacing:-.01em;color:var(--ink)}
.covsub{font-size:24px;color:var(--main-dark);margin:0;font-weight:600}
.slide.title{padding-top:20px;background:linear-gradient(135deg,#f6fbfa 0%%,#e4f4f1 100%%)}
"""

SECS = {}


def css() -> str:
    head = re.search(r"<style>([\s\S]*?)</style>", OLD.read_text(encoding="utf-8")).group(1)
    head = re.sub(r":root\{[^}]*\}", "", head, count=1)
    vals = {"bg": BG, "ink": INK, "muted": MUTED, "rule": RULE, "teal": TEAL, "teald": TEAL_D}
    for key, tag in zip("abcdefg", ("sA", "sB", "sC", "sD", "sE", "sF", "sG")):
        a0, a1, a2, _ = SECTIONS[tag]
        vals[f"{key}0"], vals[f"{key}1"], vals[f"{key}2"] = a0, a1, a2
    return head + CSS_EXTRA % vals


def render() -> str:
    old_head = OLD.read_text(encoding="utf-8")
    js = re.search(r"<script>([\s\S]*?)</script>", old_head).group(1)
    main = [i for i, s in enumerate(SLIDES) if s["kind"] != "appendix"]
    idx_app = {name: len(main) + k for k, name in enumerate(("A1", "A2", "A2.1", "A3", "A4"))}
    total = len(main)
    parts, mi = [], 0
    for i, s in enumerate(SLIDES):
        if s["kind"] == "appendix" and "raw" in s:
            raw = s["raw"]
            raw = re.sub(r"show\((\d+)\)", lambda m: m.group(0), raw)
            parts.append(raw)
            continue
        cls = "slide " + s["sec"] + (" title" if s["kind"] == "title" else "") + (" appendix" if s["kind"] == "appendix" else "")
        ins = f'<div class="ins"><span>{s["ins"]}</span></div>' if s["ins"] else ""
        if s["kind"] == "title":
            parts.append(f'<section class="{cls}">{s["body"]}<aside class="notes">{esc(s["notes"])}</aside></section>')
            mi += 1
            continue
        h1 = f"<h1>{s['title']}</h1>"
        body = s["body"]
        if s["kind"] == "main":
            mi += 1
            pg = f'<span class="pg">{mi} / {total}</span>'
        else:
            pg = '<span class="pg">Phụ lục</span>'
        sec = f'<p class="sec">{SECTIONS[s["sec"]][3]}</p>' if s["kind"] == "main" else ""
        parts.append(f'<section class="{cls}">{h1}{body}{ins}{sec}{pg}<aside class="notes">{esc(s["notes"])}</aside></section>')
    page = (
        '<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Slide week 6 v2</title><style>" + css() + '</style></head><body><div id="stage">'
        + "".join(parts) + "</div><script>" + js + "</script></body></html>"
    )
    for name, i in idx_app.items():
        page = page.replace("{" + name + "}", str(i))
    return page


def main() -> None:
    d = load()
    build(d)
    page = render()
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(SLIDES)} slides, {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
