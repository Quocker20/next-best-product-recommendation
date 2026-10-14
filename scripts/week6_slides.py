"""Week-6 slide deck (Vietnamese HTML, 16:9): cover and part A (problem and data, EDA).

Style, helpers, CSS and QA come from `scripts/final_slides.py` (style guide:
`docs/plans/final_prep/03_slide_style_guide.md`). Text rules for this deck: no connective words,
no first or second person, no hint text, every number read from `results/week6_retrain/eda_split.json`.
Parts B to E are added later; the outline is `docs/plans/slide_outline_week6.md`.

Output: reports/summary/week6_retrain/slides_week6_vi.html, figures in results/figures/week6_slides/.
Keys: arrows / space, Home, End, `n` toggles speaker notes. Print (headless Edge --print-to-pdf)
gives one 1280x720 page per slide.

Usage: .venv/Scripts/python.exe scripts/week6_slides.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_slides as fs  # noqa: E402
from final_slides import (  # noqa: E402
    CONTEXT,
    CONTRAST,
    INK,
    MAIN,
    MUTED,
    RULE,
    hbars,
    line,
    rect,
    slide,
    svg,
    table,
    txt,
)

ROOT = fs.ROOT
EDA = ROOT / "results" / "week6_retrain" / "eda_split.json"
OUT = ROOT / "reports" / "summary" / "week6_retrain" / "slides_week6_vi.html"
FIG = ROOT / "results" / "figures" / "week6_slides"
SRC = "results/week6_retrain/eda_split.json"
MONTH_VI = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
WIN_COLOR = {"train": CONTEXT, "valid": CONTRAST, "test": INK}


def num(n: float) -> str:
    return f"{int(round(n)):,}"


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def dec(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def src(section: str, scope: str) -> str:
    return f"Nguồn: {SRC} ({section}). {scope}"


# ----------------------------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------------------------
def month_ticks(months: list[str], x0: float, step: float, y: float, every: int = 3) -> str:
    return "".join(
        txt(x0 + i * step + step / 2, y, months[i][2:], 15, MUTED, "middle")
        for i in range(0, len(months), every)
    )


def fig_problem() -> str:
    boxes = "".join(
        rect(804, 30 + i * 40, 300, 32, MAIN if i == 0 else "#ffffff", MAIN if i == 0 else RULE, 1.2)
        + txt(820, 52 + i * 40, f"{i + 1}. loại phòng hoặc gói", 17, "#ffffff" if i == 0 else INK)
        for i in range(5)
    )
    inner = (
        rect(0, 70, 300, 110, "#f2f2f1", MUTED, 1.2)
        + txt(20, 108, "tìm kiếm hiện tại", 20, INK, weight=600)
        + txt(20, 136, "điểm đến, ngày, nhóm khách,", 18, MUTED)
        + txt(20, 160, "gói, kênh bán", 18, MUTED)
        + rect(0, 0, 300, 50, "none", CONTEXT, 1.2, dash=True)
        + txt(20, 32, "vài lượt đặt cũ, thường không có", 18, MUTED)
        + line(302, 125, 420, 125, MUTED, 1.5, arrow=True)
        + line(150, 52, 150, 66, MUTED, 1.2, arrow=True)
        + rect(424, 80, 260, 90, "#ffffff", INK, 1.5)
        + txt(554, 132, "hệ gợi ý", 20, INK, "middle", 600)
        + line(686, 125, 800, 125, MUTED, 1.5, arrow=True)
        + boxes
    )
    return svg("problem_schematic", 1152, 250, inner, "Sơ đồ bài toán")


def fig_volume(e: dict) -> str:
    mo = e["S1"]["monthly"]
    months = list(mo)
    w, h, left, bottom = 1152, 450, 70, 40
    step = (w - left - 10) / len(months)
    vmax = 200_000
    out = [line(left, h - bottom, w, h - bottom, RULE, 1)]
    for v in (0, 100_000, 200_000):
        y = h - bottom - (h - bottom - 20) * v / vmax
        out.append(txt(left - 10, y + 5, f"{v // 1000}k", 15, MUTED, "end"))
    for i, m in enumerate(months):
        bh = (h - bottom - 20) * mo[m]["bookings"] / vmax
        out.append(rect(left + i * step + 3, h - bottom - bh, step - 6, bh, WIN_COLOR[mo[m]["window"]]))
    out.append(month_ticks(months, left, step, h - bottom + 22))
    # direct labels of the three windows
    for name, color in (("train", MUTED), ("valid", CONTRAST), ("test", INK)):
        idx = [i for i, m in enumerate(months) if mo[m]["window"] == name]
        xm = left + (idx[0] + idx[-1] + 1) * step / 2
        out.append(txt(xm, 14, name, 17, color, "middle", 600))
    return svg("volume_by_month", w, h, "".join(out), "Lượt đặt mỗi tháng")


def fig_users(e: dict) -> str:
    hist = {int(k): v for k, v in e["S2"]["bookings_per_user_hist"].items()}
    tot = sum(hist.values())

    def part(lo: int, hi: int) -> float:
        return sum(v for k, v in hist.items() if lo <= k <= hi) / tot

    rows = [
        {"label": "1 lượt đặt", "value": part(1, 1), "color": INK},
        {"label": "2 lượt", "value": part(2, 2), "color": CONTEXT},
        {"label": "3 đến 4 lượt", "value": part(3, 4), "color": CONTEXT},
        {"label": "5 đến 9 lượt", "value": part(5, 9), "color": CONTEXT},
        {"label": "từ 10 lượt trở lên", "value": part(10, 10**9), "color": CONTEXT},
    ]
    return hbars("users_by_bookings", rows, 780, 230, 0.45, pct, "Tỷ lệ người dùng theo số lượt đặt", 58, 28)


def fig_gaps(e: dict) -> str:
    q = e["S2"]["days_between_bookings"]["test"]
    rows = [
        {"label": f"phân vị {int(float(k) * 100)}%", "value": q[k], "color": INK if k == "0.5" else CONTEXT}
        for k in ("0.25", "0.5", "0.75", "0.9")
    ]
    return hbars("gap_days", rows, 780, 200, q["0.9"] * 1.02, lambda v: f"{v:.1f} ngày", "Số ngày giữa hai lượt đặt liên tiếp", 66, 34)


def fig_cluster_rank(e: dict) -> str:
    s = e["S3"]["share_by_train_rank"]["train"]
    w, h, left, bottom = 820, 440, 70, 44
    step = (w - left - 10) / len(s)
    vmax = 0.045
    out = [line(left, h - bottom, w, h - bottom, RULE, 1)]
    for v in (0, 0.02, 0.04):
        y = h - bottom - (h - bottom - 20) * v / vmax
        out.append(txt(left - 10, y + 5, pct(v), 15, MUTED, "end"))
    for i, v in enumerate(s):
        bh = (h - bottom - 20) * v / vmax
        out.append(rect(left + i * step + 0.5, h - bottom - bh, step - 1, bh, INK if i < 10 else CONTEXT))
    for r in (1, 10, 50, 100):
        out.append(txt(left + (r - 0.5) * step, h - bottom + 22, str(r), 15, MUTED, "middle"))
    out.append(txt(left + (w - left) / 2, h - 4, "hạng cluster theo tỷ lệ trong train", 16, MUTED, "middle"))
    return svg("cluster_share_by_rank", w, h, "".join(out), "Tỷ lệ lượt đặt theo hạng cluster")


def fig_dest_cum(e: dict) -> str:
    r, c = np.array(e["S6"]["cum_curve_ranks"]), np.array(e["S6"]["cum_curve"])
    w, h, left, bottom = 820, 440, 70, 44
    lx = lambda v: left + (w - left - 20) * np.log10(v) / np.log10(r.max())  # noqa: E731
    ly = lambda v: h - bottom - (h - bottom - 20) * v  # noqa: E731
    out = [line(left, h - bottom, w, h - bottom, RULE, 1)]
    for v in (0, 0.5, 1.0):
        out.append(txt(left - 10, ly(v) + 5, pct(v), 15, MUTED, "end"))
    pts = " ".join(f"{lx(a):.1f},{ly(b):.1f}" for a, b in zip(r, c))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    for t in (1, 10, 100, 1000, 10000):
        out.append(txt(lx(t), h - bottom + 22, f"{t:,}", 15, MUTED, "middle"))
    for t, v in ((100, e["S6"]["top100_share"]), (1000, e["S6"]["top1000_share"])):
        out.append(f'<circle cx="{lx(t):.1f}" cy="{ly(v):.1f}" r="5" fill="{MAIN}"/>')
        out.append(txt(lx(t) + 10, ly(v) + 22, pct(v), 17, MAIN, weight=600))
    out.append(txt(left + (w - left) / 2, h - 4, "hạng điểm đến (thang log)", 16, MUTED, "middle"))
    return svg("destination_cumulative", w, h, "".join(out), "Tỷ lệ lượt đặt tích lũy theo hạng điểm đến")


def fig_party(e: dict) -> str:
    p = e["S4"]["composition"]["train"]["party"]
    names = {"couple": "cặp đôi (couple)", "solo": "một mình (solo)", "family": "gia đình (family)", "group": "nhóm (group)"}
    rows = [{"label": names[k], "value": p[k], "color": INK if k == "couple" else CONTEXT} for k in names]
    return hbars("party_train", rows, 780, 250, 0.5, pct, "Loại nhóm khách, tập train", 66, 34)


def fig_checkin(e: dict) -> str:
    s = e["S5"]["checkin_month_share"]["train"]
    w, h, left, bottom = 540, 420, 50, 40
    step = (w - left - 6) / 12
    vmax = 0.16
    out = [line(left, h - bottom, w, h - bottom, RULE, 1)]
    for v in (0, 0.05, 0.10, 0.15):
        y = h - bottom - (h - bottom - 20) * v / vmax
        out.append(txt(left - 8, y + 5, pct(v), 14, MUTED, "end"))
    top = max(s, key=s.get)
    for m in range(1, 13):
        bh = (h - bottom - 20) * s[str(m)] / vmax
        out.append(rect(left + (m - 1) * step + 3, h - bottom - bh, step - 6, bh, INK if str(m) == top else CONTEXT))
        out.append(txt(left + (m - 0.5) * step, h - bottom + 20, str(m), 14, MUTED, "middle"))
    out.append(txt(left, 14, "tháng check-in, tập train", 16, MUTED))
    return svg("checkin_month_train", w, h, "".join(out), "Tỷ lệ lượt đặt theo tháng check-in")


def fig_trip_mix(e: dict) -> str:
    mo = e["S1"]["monthly"]
    months = list(mo)
    w, h, left, bottom = 560, 420, 50, 40
    step = (w - left - 6) / len(months)
    vmax = 0.35
    ly = lambda v: h - bottom - (h - bottom - 20) * v / vmax  # noqa: E731
    lx = lambda i: left + (i + 0.5) * step  # noqa: E731
    out = [line(left, h - bottom, w, h - bottom, RULE, 1)]
    for v in (0, 0.1, 0.2, 0.3):
        out.append(txt(left - 8, ly(v) + 5, pct(v), 14, MUTED, "end"))
    for key, color, name in (("family_share", MAIN, "family"), ("package_share", CONTRAST, "package")):
        pts = " ".join(f"{lx(i):.1f},{ly(mo[m][key]):.1f}" for i, m in enumerate(months))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        dy = -12 if key == "family_share" else 26
        out.append(txt(lx(len(months) - 1) - 4, ly(mo[months[-1]][key]) + dy, name, 16, color, "end", 600))
    out.append(month_ticks(months, left, step, h - bottom + 20, 6))
    out.append(txt(left, 14, "tỷ lệ theo tháng đặt", 16, MUTED))
    return svg("trip_mix_monthly", w, h, "".join(out), "Tỷ lệ family và package theo tháng đặt")


def fig_repeat_h(e: dict) -> str:
    t = e["S7"]["test"]
    keys = ["h=0", "h=1", "h=2", "h=3-5", "h=6+"]
    labels = ["chưa có", "1 lượt", "2 lượt", "3 đến 5 lượt", "từ 6 lượt"]
    rows = [
        {"label": lb, "value": t[k]["cluster_in_history_full"], "color": MAIN if k == keys[-1] else CONTEXT}
        for k, lb in zip(keys, labels)
    ]
    return hbars("repeat_by_h", rows, 780, 190, 0.7, pct, "Cluster đã nằm trong lịch sử, theo số lượt đặt trước đó trong cửa sổ test", 62, 28)


def fig_fields(e: dict) -> str:
    ig = e["S8"]["information_gain_bits"]
    names = {
        "dest": "điểm đến",
        "country": "quốc gia người dùng",
        "site": "site",
        "posa": "châu lục điểm bán",
        "stay": "số đêm",
        "lead": "số ngày đến check-in",
        "dest_type": "loại điểm đến",
        "package": "gói",
    }
    rows = [{"label": v, "value": ig[k]["test"], "color": INK if k == "dest" else CONTEXT} for k, v in names.items()]
    return hbars("fields_information", rows, 900, 270, ig["dest"]["test"] * 1.05, lambda v: f"{v:.2f}", "bit", 40, 18)


# ----------------------------------------------------------------------------------------------
# slides
# ----------------------------------------------------------------------------------------------
def build(e: dict) -> None:
    s1, s2, s3, s4, s5, s6, s7, s8 = (e[k] for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"))
    events, users = e["events"], s2["users"]
    mo = s1["monthly"]
    avg13 = np.mean([v["bookings"] for k, v in mo.items() if k.startswith("2013")])
    avg14 = np.mean([v["bookings"] for k, v in mo.items() if k.startswith("2014")])
    single = s2["share_single_booking_users"]
    med = s2["bookings_per_user_quantiles"]["0.5"]
    gap = s2["days_between_bookings"]["test"]
    train_party = s4["composition"]["train"]
    test_rep = s7["test"]["all"]
    ig = s8["information_gain_bits"]
    others = max(v["test"] for k, v in ig.items() if k not in ("dest",) and "candidate" not in k)

    # 1 cover
    slide(
        "Next-best-product recommendation cho du lịch",
        '<p class="sub">Gợi ý phòng hoặc gói tiếp theo trong tìm kiếm khách sạn, đề tài C1 '
        "(bối cảnh Vinpearl, GSM)</p>"
        '<p class="sub2">Báo cáo cuối kỳ</p>'
        '<p class="sub3">Dữ liệu công khai Expedia. Không dùng dữ liệu của Vinpearl, GSM.</p>',
        "Trang bìa. Đề tài C1: gợi ý phòng hoặc gói tiếp theo cho khách đang tìm khách sạn. "
        "Toàn bộ thực nghiệm dùng dữ liệu công khai Expedia Hotel Recommendations; không có dữ liệu "
        "Vinpearl hay GSM. Câu hỏi có thể gặp: đây có phải dữ liệu Vinpearl không? Không.",
        kind="title",
    )

    # 1b domain characteristics
    slide(
        "Chọn đúng cụm khách sạn cho từng chuyến đi: dữ liệu ít, chi phí sai cao",
        '<p class="lede" style="font-size:28px;margin:0 0 28px 0">Khách đang tìm khách sạn: '
        "đề xuất cụm khách sạn nào?</p>"
        + table(
            ["Đặc điểm", "Biểu hiện"],
            [
                ["Tần suất mua thấp", "một đến hai chuyến mỗi năm, dữ liệu rất ít"],
                ["Ngữ cảnh chuyến đi quyết định", "một mình, cặp đôi, nhóm; điểm đến, ngày đi"],
                ["Chi phí chọn sai cao", "đặt nhầm: tốn tiền, tốn thời gian, ảnh hưởng cả chuyến đi"],
            ],
            num_from=9,
        ).replace('<table class="t">', '<table class="t" style="font-size:24px">'),
        "Câu hỏi đặt ra: khách đang tìm khách sạn thì đề xuất cụm khách sạn nào. Ba đặc điểm của lĩnh "
        "vực du lịch khách sạn làm bài toán khó hơn gợi ý thông thường. Một: tần suất mua thấp, một "
        "đến hai chuyến mỗi năm, nên lịch sử của mỗi khách rất ngắn. Hai: ngữ cảnh chuyến đi (một "
        "mình, cặp đôi, nhóm, điểm đến, ngày đi) quyết định lựa chọn nhiều hơn sở thích cũ. Ba: "
        "chi phí chọn sai cao, đặt nhầm khách sạn ảnh hưởng cả chuyến đi, tốn tiền, tốn thời gian.",
    )

    # 2 problem
    slide(
        "Hệ gợi ý chọn phòng hoặc gói ngay lúc khách đang tìm kiếm",
        f'<div class="fig">{fig_problem()}</div>'
        '<p class="lede" style="font-size:26px;margin-top:48px">Đầu ra: top 5 trong 100 hotel_cluster, đại diện hạng phòng hoặc gói.</p>',
        "Đầu vào gồm lần tìm kiếm hiện tại (điểm đến, ngày, nhóm khách, gói, kênh bán) và vài lượt "
        "đặt cũ, thường không có. Đầu ra là danh sách xếp hạng 100 hotel_cluster. Cluster là id ẩn "
        "danh của Expedia, đóng vai hạng phòng hoặc gói của khu nghỉ dưỡng.",
    )

    # 3 research questions
    slide(
        "RQ1 đo độ chính xác, RQ4 đo diversity và coverage",
        '<div class="rq">'
        "<p><b>RQ1</b> Mô hình phiên có ngữ cảnh vượt collaborative filtering truyền thống?</p>"
        "<p><b>RQ4</b> Độ chính xác tăng, diversity và coverage thay đổi ra sao?</p></div>"
        + table(
            ["", "Đo bằng", "Phân tích theo"],
            [
                ["RQ1", "NDCG@K, Recall@K", "lịch sử, mùa, ngữ cảnh chuyến đi, điểm đến"],
                ["RQ4", "diversity, độ lệch phổ biến", "coverage theo điểm đến, mùa"],
            ],
            num_from=9,
        ),
        "RQ1 so sánh mô hình chuỗi có ngữ cảnh với collaborative filtering truyền thống, phân tích "
        "theo độ dài lịch sử, mùa, ngữ cảnh chuyến đi, điểm đến đã biết hoặc mới. RQ4 đo đánh đổi "
        "giữa độ chính xác với diversity và coverage. RQ2 (giao đồ ăn) và RQ3 (cross-sell) ngoài "
        "phạm vi.",
    )

    # 4 dataset
    slide(
        f"Expedia: {num(events)} lượt đặt, {num(users)} người dùng, 100 cluster, 2013 đến 2014",
        table(
            ["Nhóm trường", "Nội dung"],
            [
                ["Điểm đến, thời gian", "điểm đến, ngày check-in, ngày check-out"],
                ["Thành phần nhóm khách", "số người lớn, số trẻ em, số phòng"],
                ["Kênh đặt", "gói (is_package), kênh bán, thiết bị di động"],
                ["Mục tiêu", "hotel_cluster, 100 giá trị, đại diện hạng phòng hoặc gói"],
                ["Thông tin khách sạn đã đặt", "chỉ làm nhãn, không là đầu vào"],
            ],
            num_from=9,
        ).replace('<table class="t">', '<table class="t" style="font-size:24px">'),
        f"Mỗi dòng là một lượt đặt kèm lần tìm kiếm dẫn tới lượt đặt đó. Tổng {num(events)} lượt đặt "
        f"của {num(users)} người dùng. Mục tiêu hotel_cluster là id ẩn danh, đại diện hạng phòng "
        "hoặc gói; không có cột giá. Giới hạn: một dataset, đánh giá offline. "
        f"Nguồn: {SRC} (S1, S2).",
        src=src("S1, S2", "Expedia Hotel Recommendations, lượt đặt."),
    )

    # 4 volume
    slide(
        f"Lượt đặt mỗi tháng tăng từ {num(round(avg13, -3))} (2013) lên {num(round(avg14, -3))} (2014)",
        f'<div class="fig">{fig_volume(e)}</div>',
        f"Trung bình {num(avg13)} lượt đặt mỗi tháng năm 2013, {num(avg14)} năm 2014. Mức tăng mạnh "
        f"bắt đầu tháng 2014-03 ({num(mo['2014-03']['bookings'])}). Số người dùng mới mỗi tháng năm 2014 "
        f"dao động {num(min(v['new_users'] for k, v in mo.items() if k.startswith('2014')))} đến "
        f"{num(max(v['new_users'] for k, v in mo.items() if k.startswith('2014')))}. Màu cột đánh dấu "
        "ba cửa sổ train, valid, test của phép chia theo ngày. Nguồn: "
        f"{SRC} (S1).",
        src=src("S1", "Số lượt đặt mỗi tháng, mọi người dùng."),
    )

    # 5 sparsity
    slide(
        f"{pct(single)} người dùng chỉ đặt một lần, trung vị {num(med)} lượt đặt mỗi người",
        f'<div class="row"><div class="fig">{fig_users(e)}</div>'
        f'<p class="aside">Phân vị 90%: {num(s2["bookings_per_user_quantiles"]["0.9"])} lượt. '
        f'Mật độ cặp người dùng và cluster: {pct(s2["user_cluster_density"], 2)}.</p></div>',
        f"{pct(single, 2)} người dùng chỉ có một lượt đặt; {pct(s2['share_users_10_plus'], 2)} có từ 10 "
        "lượt trở lên. Mô hình cần lịch sử dài có rất ít dữ liệu để dựa vào ở đa số người dùng. "
        f"Nguồn: {SRC} (S2).",
        src=src("S2", "Số lượt đặt mỗi người dùng, mọi người dùng."),
    )

    # 6 gaps
    slide(
        f"Khoảng cách giữa hai lượt đặt: trung vị {dec(gap['0.5'], 1)} ngày, "
        f"{pct(s2['share_gap_within_1h']['test'], 2)} cách nhau tối đa một giờ",
        f'<div class="row"><div class="fig">{fig_gaps(e)}</div>'
        f'<p class="aside">Tối đa 1 ngày: {pct(s2["share_gap_within_1d"]["test"], 2)}.</p></div>',
        "Khoảng cách tính giữa lượt đặt hiện tại với lượt đặt liền trước của cùng người dùng, trên "
        "các dòng warm của cửa sổ test. Một phần lượt đặt cách nhau dưới một giờ: lịch sử trong cùng "
        f"cửa sổ gần như là cùng một phiên tìm kiếm. Nguồn: {SRC} (S2).",
        src=src("S2", "Dòng warm, cửa sổ test."),
    )

    # 7 cluster distribution
    g = s3["per_window"]["train"]
    slide(
        f"100 cluster phân bố đều: top 10 chiếm {pct(g['top10_share'])} lượt đặt, top 1 chiếm {pct(g['top1_share'], 1)}",
        f'<div class="row"><div class="fig">{fig_cluster_rank(e)}</div>'
        f'<p class="aside">Entropy chuẩn hóa {dec(g["entropy_normalised"], 2)}. '
        f'50 cluster đầu của train phủ {pct(s3["targets_covered_by_top_m_train_clusters"]["test"]["50"])} '
        "mục tiêu test.</p></div>",
        f"Không có cluster nào chiếm ưu thế: cluster đứng đầu train chiếm {pct(g['top1_share'], 2)}, "
        f"10 cluster đầu chiếm {pct(g['top10_share'], 2)}. Gợi ý theo độ phổ biến vì thế khó đạt độ "
        f"chính xác cao. Nguồn: {SRC} (S3).",
        src=src("S3", "Tập train, tỷ lệ lượt đặt theo cluster."),
    )

    # 8 destinations
    slide(
        f"Điểm đến tập trung: top 1,000 trong {num(s6['destinations_all'])} điểm đến chiếm {pct(s6['top1000_share'])} lượt đặt",
        f'<div class="row"><div class="fig">{fig_dest_cum(e)}</div>'
        f'<p class="aside">Entropy của cluster: {dec(s6["cluster_entropy_bits_marginal_train"])} bit. '
        f'Biết điểm đến: {dec(s6["mean_cluster_entropy_bits_given_destination_ge20"])} bit.</p></div>',
        f"{num(s6['destinations_all'])} điểm đến; 100 điểm đầu chiếm {pct(s6['top100_share'])}, 1,000 "
        f"điểm đầu chiếm {pct(s6['top1000_share'])}. Entropy trung bình của cluster theo điểm đến tính "
        f"trên {num(s6['destinations_with_ge20_train_bookings'])} điểm đến có từ 20 lượt đặt trong train. "
        f"Nguồn: {SRC} (S6).",
        src=src("S6", "Mọi lượt đặt; entropy tính trên tập train."),
    )

    # 9 trip context
    slide(
        f"Ngữ cảnh chuyến đi đa dạng: cặp đôi {pct(train_party['party']['couple'])}, "
        f"một mình {pct(train_party['party']['solo'])}, gia đình {pct(train_party['party']['family'])}",
        f'<div class="row"><div class="fig">{fig_party(e)}</div>'
        f'<p class="aside">Gói: {pct(train_party["package_share"], 1)}. '
        f'Thiết bị di động: {pct(train_party["mobile_share"], 1)}. '
        f'Trung vị đặt trước {num(train_party["lead_days_quantiles"]["0.5"])} ngày, '
        f'ở {num(train_party["stay_nights_quantiles"]["0.5"])} đêm.</p></div>',
        "Loại nhóm khách suy ra từ số người lớn và số trẻ em của lần tìm kiếm. Các trường ngữ cảnh "
        "chuyến đi (nhóm khách, gói, thời gian đặt trước, số đêm) là đầu vào của query embedding. "
        f"Nguồn: {SRC} (S4).",
        src=src("S4", "Tập train, tỷ lệ lượt đặt."),
    )

    # 10 season and drift
    fam = [v["family_share"] for v in mo.values()]
    pkg = [v["package_share"] for v in mo.values()]
    top_m = max(s5["checkin_month_share"]["train"], key=s5["checkin_month_share"]["train"].get)
    slide(
        f"Mùa vụ mạnh, tỷ lệ family giảm từ {pct(max(fam))} xuống {pct(min(fam))}",
        f'<div class="row"><div class="fig">{fig_checkin(e)}</div>'
        f'<div class="fig">{fig_trip_mix(e)}</div></div>',
        f"Check-in tháng {top_m} chiếm {pct(s5['checkin_month_share']['train'][top_m], 2)} lượt đặt "
        f"trong train. Theo tháng đặt, tỷ lệ family đi từ {pct(max(fam), 2)} xuống {pct(min(fam), 2)}, "
        f"tỷ lệ gói đi từ {pct(max(pkg), 2)} xuống {pct(min(pkg), 2)}: cơ cấu khách đổi theo thời gian, "
        f"không chỉ theo mùa. Nguồn: {SRC} (S1, S5).",
        src=src("S1, S5", "Trái: tập train. Phải: mọi lượt đặt theo tháng đặt."),
    )

    # 11 repeats
    rh = s7["test"]
    slide(
        f"Cluster lặp lại tăng theo lịch sử: {pct(rh['h=0']['cluster_in_history_full'])} khi chưa có, "
        f"{pct(rh['h=6+']['cluster_in_history_full'])} từ 6 lượt",
        '<p class="cap">Tỷ lệ dòng test có cluster đã đặt trước đó, theo số lượt đặt trước đó trong cửa sổ test</p>'
        f'<div class="row"><div class="fig">{fig_repeat_h(e)}</div>'
        f'<p class="aside">Điểm đến đã có trong lịch sử: {pct(rh["pair_given_destination_in_history_full"], 1)} '
        f'đặt lại đúng cluster cũ. Cluster đã đặt trước đó: {pct(test_rep["cluster"]["in_history_full"], 1)} '
        f'dòng test. Điểm đến đã tìm trước đó: {pct(test_rep["destination"]["in_history_full"], 1)}.</p></div>',
        "Lịch sử đầy đủ gồm mọi lượt đặt trước đó của người dùng, kể cả lượt trong cùng cửa sổ test. "
        "Tỷ lệ lặp tăng theo số lượt đặt trước đó trong cửa sổ. Quan sát này là cơ sở của sameDest: "
        f"khi điểm đến đã có trong lịch sử, {pct(rh['pair_given_destination_in_history_full'], 2)} dòng "
        f"đặt lại đúng cluster cũ. Nguồn: {SRC} (S7).",
        src=src("S7", "Cửa sổ test."),
    )

    # 12 information in the fields
    slide(
        f"Điểm đến mang {dec(ig['dest']['test'])} bit về cluster, mọi trường khác không quá {dec(others, 2)} bit",
        f'<div class="fig">{fig_fields(e)}</div>',
        f"Information gain về cluster của từng trường tìm kiếm, ước lượng trên train, đo trên test; "
        f"entropy biên {dec(s8['cluster_marginal_entropy_bits_train'])} bit. Điểm đến {dec(ig['dest']['test'], 3)} "
        f"bit, trường kế tiếp (quốc gia người dùng) {dec(ig['country']['test'], 3)} bit. Lần tìm kiếm "
        f"hiện tại mang nhiều thông tin hơn lịch sử; đây là lý do đưa tìm kiếm vào mô hình. "
        f"Nguồn: {SRC} (S8).",
        src=src("S8", "Ước lượng trên train, đo trên cửa sổ test."),
    )


def main() -> None:
    fs.LANG = "vi"
    e = json.loads(EDA.read_text(encoding="utf-8"))
    build(e)
    page = fs.render().replace("Slide cuối kỳ v1 (tiếng Việt)", "Slide week 6 (phần A)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    for name, body in fs.FIGURES.items():
        (FIG / f"{name}.svg").write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(fs.SLIDES)} slides)")
    print("QA issues:", fs.qa(page) or "none")


if __name__ == "__main__":
    main()
