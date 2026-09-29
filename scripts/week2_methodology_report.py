"""Rebuild the Vietnamese methodology selection report for the week-2 decision
(SMLP4Rec + cold-start hybrid; AdaGIN and LightGBM kept as comparators).

The existing .docx is used as the formatting template: its styles, page setup,
header/footer and table-of-contents field are kept; the body is regenerated.
Every number is read from persisted script outputs:
- reports/summary/week1_dataset_selection/expedia_eda_detail.json
- reports/summary/week1_dataset_selection/expedia_context_signal.json
- reports/summary/week2_methodology/expedia_history_slices.json
Figures come from scripts/methodology_figures.py (fig1-fig6) and
scripts/week2_report_figures.py (fig7-fig13). Run both first.

Usage: python scripts/week2_methodology_report.py
Output: reports/summary/week2_methodology/methodology_selection_report.docx
The table of contents is refreshed when the file is opened in Word (F9).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from xml.sax.saxutils import escape

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOCX = (
    ROOT
    / "reports"
    / "summary"
    / "week2_methodology"
    / "methodology_selection_report.docx"
)
FIG = ROOT / "reports" / "figures" / "vi"
W1 = ROOT / "reports" / "summary" / "week1_dataset_selection"
EDA = json.loads((W1 / "expedia_eda_detail.json").read_text(encoding="utf-8"))
CTX = json.loads((W1 / "expedia_context_signal.json").read_text(encoding="utf-8"))
HS = json.loads(
    (ROOT / "reports/summary/week2_methodology/expedia_history_slices.json").read_text(
        encoding="utf-8"
    )
)
TABLE_WIDTH = 9638


# ---------------------------------------------------------------- number formatting


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def dec(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


R5 = CTX["popularity_probe"]["recall_at_5"]
JSD = CTX["distribution_shift"]
ON = HS["by_history_online"]
SUP = HS["by_dest_support"]
DEPTH = EDA["user_depth"]
FIELD = {f["field"]: f for f in EDA["field_table"]}
MONTHLY = EDA["monthly_bookings"]
Y2013 = sum(v for k, v in MONTHLY.items() if k.startswith("2013"))
Y2014 = sum(v for k, v in MONTHLY.items() if k.startswith("2014"))
PKG = EDA["monthly_package_share"]
N_BOOKERS = sum(v["count"] for v in DEPTH["histogram"].values())
LE1 = ON["0"]["share"] + ON["1"]["share"]
BEST_RULE = HS["overall"]["R@5_same_dest_history_then_dest_smooth20"]
COLD_FLOOR = ON["0"]["R@5_dest_smooth_m5"]


def gain(bucket: str) -> float:
    b = ON[bucket]
    return (
        b["R@5_same_dest_history_then_dest_smooth20"] - b["R@5_dest_raw_backoff_market"]
    )


# ---------------------------------------------------------------- XML builders

RUN_PR = {
    "title": '<w:b/><w:bCs/><w:color w:val="1F3A5F"/><w:sz w:val="34"/><w:szCs w:val="34"/>',
    "subtitle": '<w:color w:val="52514E"/><w:sz w:val="22"/><w:szCs w:val="22"/>',
    "caption": '<w:i/><w:iCs/><w:color w:val="52514E"/><w:sz w:val="18"/><w:szCs w:val="18"/>',
    "ref": '<w:sz w:val="19"/><w:szCs w:val="19"/>',
    "cell": '<w:sz w:val="19"/><w:szCs w:val="19"/>',
}
PARA_PR = {
    "title": '<w:spacing w:after="80"/>',
    "subtitle": '<w:spacing w:after="200"/>',
    "body": '<w:spacing w:after="100" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>',
    "h1": '<w:pStyle w:val="u1"/><w:keepNext/><w:keepLines/>',
    "h2": '<w:pStyle w:val="u2"/><w:keepNext/><w:keepLines/>',
    "list": '<w:pStyle w:val="oancuaDanhsach"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
    '<w:spacing w:after="60" w:line="264" w:lineRule="auto"/>',
    "caption": '<w:spacing w:after="160"/><w:jc w:val="center"/>',
    "ref": '<w:spacing w:after="100" w:line="276" w:lineRule="auto"/>',
    "img": '<w:keepNext/><w:spacing w:before="80" w:after="40"/><w:jc w:val="center"/>',
    "cell": '<w:spacing w:after="20"/>',
}


def runs_xml(parts, base: str = "") -> str:
    """parts: str or list of (text, bold). '**x**' markup is not used; bold is explicit."""
    if isinstance(parts, str):
        parts = [(parts, False)]
    out = []
    for text, bold in parts:
        rpr = base + ("<w:b/><w:bCs/>" if bold and "<w:b/>" not in base else "")
        out.append(
            f"<w:r>{f'<w:rPr>{rpr}</w:rPr>' if rpr else ''}"
            f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
        )
    return "".join(out)


def para_el(parts, kind: str = "body"):
    base = RUN_PR.get(kind, "")
    return parse_xml(
        f"<w:p {nsdecls('w')}><w:pPr>{PARA_PR[kind]}</w:pPr>{runs_xml(parts, base)}</w:p>"
    )


def page_break_el():
    return parse_xml(f'<w:p {nsdecls("w")}><w:r><w:br w:type="page"/></w:r></w:p>')


BORDER = "".join(
    f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="B8C2CF"/>'
    for s in ("top", "left", "bottom", "right")
)
CELL_MAR = (
    '<w:tcMar><w:top w:w="60" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
    '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
)


def table_el(rows, widths, header: bool = True, shade_first_col: bool = False):
    """rows: list of rows; each cell is str or list of (text, bold). widths in DXA, sum = TABLE_WIDTH."""
    assert sum(widths) == TABLE_WIDTH, sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = []
    for r_i, row in enumerate(rows):
        is_head = header and r_i == 0
        tcs = []
        for c_i, (cell, w) in enumerate(zip(row, widths)):
            shade = is_head or (shade_first_col and c_i == 0)
            shd = (
                '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF6"/>'
                if is_head
                else (
                    '<w:shd w:val="clear" w:color="auto" w:fill="F5F7FA"/>'
                    if shade
                    else ""
                )
            )
            parts = [(cell, is_head)] if isinstance(cell, str) else cell
            tcs.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:tcBorders>{BORDER}</w:tcBorders>'
                f"{shd}{CELL_MAR}</w:tcPr><w:p><w:pPr>{PARA_PR['cell']}</w:pPr>"
                f"{runs_xml(parts, RUN_PR['cell'])}</w:p></w:tc>"
            )
        trpr = (
            "<w:trPr><w:cantSplit/>"
            + ("<w:tblHeader/>" if is_head else "")
            + "</w:trPr>"
        )
        trs.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
    return parse_xml(
        f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{TABLE_WIDTH}" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders>'
        '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
        'w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
    )


# ---------------------------------------------------------------- document assembly


class Builder:
    def __init__(self, path: Path):
        self.doc = docx.Document(str(path))
        self.body = self.doc.element.body
        self.sect = self.body.find(qn("w:sectPr"))
        toc = self.body.find(qn("w:sdt"))
        self.toc = copy.deepcopy(toc) if toc is not None else None
        for el in list(self.body):
            if el is not self.sect:
                self.body.remove(el)
        self.fig_no = 0
        self.tab_no = 0

    def add(self, el):
        self.sect.addprevious(el)
        return el

    def p(self, parts, kind: str = "body"):
        return self.add(para_el(parts, kind))

    def lead(self, lead: str, text: str):
        return self.p([(lead + " ", True), (text, False)])

    def bullets(self, items):
        for it in items:
            self.p(it, "list")

    def h1(self, text: str):
        self.p(text, "h1")

    def h2(self, text: str):
        self.p(text, "h2")

    def table(self, rows, widths, **kw):
        self.add(table_el(rows, widths, **kw))
        self.p("", "ref")

    def figure(self, name: str, caption: str, width_cm: float = 16.0):
        el = self.add(para_el([], "img"))
        Paragraph(el, self.doc._body).add_run().add_picture(
            str(FIG / name), width=Cm(width_cm)
        )
        self.fig_no += 1
        self.p(f"Hình {self.fig_no}. {caption}", "caption")

    def save(self, path: Path):
        self.doc.save(str(path))


def build() -> Builder:
    b = Builder(DOCX)

    # ------------------------------------------------------------ cover block
    b.p(
        "Lựa chọn phương pháp cho bài toán gợi ý hotel cluster theo ngữ cảnh chuyến đi",
        "title",
    )
    b.p(
        "SMLP4Rec kết hợp hybrid cho user ít lịch sử — so sánh với AdaGIN và LightGBM trên Expedia Hotel Recommendations",
        "subtitle",
    )
    b.table(
        [
            ["Hạng mục", "Nội dung"],
            [
                [("Bài toán", True)],
                "Xếp hạng 100 hotel cluster cho một lượt tìm kiếm, dựa trên ngữ cảnh lượt tìm kiếm và lịch sử booking của user nếu có",
            ],
            [
                [("Dữ liệu", True)],
                (
                    f"Expedia Hotel Recommendations: {num(EDA['bookings'])} booking, {num(N_BOOKERS)} user có booking, "
                    f"{EDA['daily']['date_min']} → {EDA['daily']['date_max']}"
                ),
            ],
            [
                [("Giao thức", True)],
                (
                    f"Chia theo thời gian toàn cục ({num(HS['n_train'])} booking train / {num(HS['n_test'])} booking test); "
                    "xếp hạng toàn bộ 100 cluster [9]; Recall@K, NDCG@K, MRR, K = 5, 10, 20"
                ),
            ],
            [
                [("Lựa chọn", True)],
                [
                    ("SMLP4Rec + hybrid cho user ít lịch sử", True),
                    (" (prior theo ngữ cảnh, gate theo độ dài lịch sử)", False),
                ],
            ],
            [
                [("Mô hình so sánh", True)],
                "AdaGIN (context-aware), LightGBM (baseline dạng bảng)",
            ],
            [
                [("Phiên bản", True)],
                "28/09/2026 — thay bản 23/09/2026 (bản trước chọn AdaGIN)",
            ],
        ],
        [1927, 7711],
    )
    if b.toc is not None:
        b.add(b.toc)
    b.add(page_break_el())

    # ------------------------------------------------------------ 1. problem and data
    b.h1("1. Bài toán và dữ liệu")
    b.h2("1.1 Bài toán")
    b.p(
        "Khách tìm kiếm với điểm đến, ngày lưu trú, thành phần đoàn, kênh và thiết bị. Mô hình xếp hạng 100 hotel cluster "
        "ẩn danh sao cho cluster được đặt nằm càng cao càng tốt. Cluster không có tên, mô tả hay thuộc tính, đóng vai trò "
        f"đại diện cho hạng phòng hoặc gói sản phẩm. Chỉ dùng sự kiện booking làm nhãn; click ({pct(EDA['clicks'] / EDA['total_rows'])} số dòng) không dùng làm nhãn dương."
    )
    b.p(
        "Các trường mô tả khách sạn đã được đặt (hotel_market, hotel_country, hotel_continent, orig_destination_distance) là "
        "thông tin phía đáp án, không dùng làm đầu vào của lượt đang dự đoán. Market của lượt tìm kiếm chỉ được suy ra từ "
        f"điểm đến qua dữ liệu train: {pct(HS['dest_to_modal_market_purity'])} booking của một điểm đến thuộc market phổ biến nhất của nó."
    )

    b.h2("1.2 Đặc tính dữ liệu")
    b.lead(
        "Khối lượng tăng và hành vi thay đổi theo thời gian.",
        f"Số booking năm 2014 tăng {pct(Y2014 / Y2013 - 1, 1)} so với 2013; tỷ trọng package giảm từ {pct(PKG['2014-01'])} "
        f"(01/2014) xuống {pct(PKG['2014-12'])} (12/2014). Dữ liệu phải chia theo thời gian, validation nằm ngay trước cửa sổ test.",
    )
    b.figure(
        "fig6_monthly_volume.png",
        "Khối lượng booking theo tháng và ranh giới chia train/test.",
    )
    b.figure("fig7_package_share.png", "Tỷ trọng booking package theo tháng.")
    b.lead(
        "Ngữ cảnh chi phối.",
        f"Phân phối cluster thay đổi mạnh theo điểm đến (JSD trung bình {dec(JSD['srch_destination_top']['js_mean'])} bit) "
        f"nhưng gần như không đổi theo tháng nhận phòng ({dec(JSD['checkin_month']['js_mean'])}) hay thành phần đoàn "
        f"({dec(JSD['party']['js_mean'])}).",
    )
    b.figure(
        "fig4_context_divergence.png",
        "Độ phân kỳ Jensen–Shannon của phân phối cluster theo từng trường ngữ cảnh.",
    )
    b.lead(
        "Điểm đến là tín hiệu mạnh nhất.",
        f"Chỉ xếp hạng theo tần suất cluster tại điểm đến đã đạt Recall@5 {pct(R5['popularity_by_srch_destination_id'])}, "
        f"so với {pct(R5['global_popularity'])} của tần suất toàn cục. Lặp lại cluster gần nhất chỉ đạt "
        f"{pct(R5['repeat_last_cluster'])}: lịch sử có tín hiệu nhưng yếu nếu dùng riêng. Chia nhỏ thêm theo tháng làm giảm "
        f"còn {pct(R5['popularity_by_destination_x_ci_month'])} vì số đếm bị phân mảnh.",
    )
    b.figure(
        "fig5_recall_by_rule.png",
        "Recall@5 của bảng tần suất theo từng biến điều kiện, tập test chia theo thời gian.",
    )
    b.lead(
        "Điểm đến nhiều giá trị, đuôi dài.",
        f"{num(FIELD['srch_destination_id']['n_unique_all'])} giá trị điểm đến; {num(EDA['destinations']['n_destinations_bookings'])} "
        f"có booking, trong đó {num(EDA['destinations']['n_dest_single_booking'])} chỉ có đúng một booking.",
    )
    b.figure(
        "fig2_destination_concentration.png",
        "Tỷ trọng booking tích lũy theo dải thứ hạng điểm đến.",
    )
    b.lead(
        "Catalog nhỏ, phân bố tương đối đều.",
        "100 cluster, cluster phổ biến nhất chiếm chưa tới 5% booking: xếp hạng toàn bộ catalog là khả thi và không cần lấy mẫu âm.",
    )
    b.figure(
        "fig1_item_distribution.png",
        "Tỷ trọng booking của từng hotel cluster, xếp theo thứ hạng.",
    )
    b.lead(
        "Lịch sử user ngắn.",
        f"Trung vị {int(DEPTH['quantiles']['p50'])} booking mỗi user; {pct(DEPTH['histogram']['1']['share'])} user chỉ có một booking. "
        f"Chuỗi ngắn làm giảm tín hiệu thứ tự [8]. Trên tập test, {pct(ON['0']['share'])} booking không có booking nào trước đó (chi tiết ở Mục 5).",
    )
    b.figure("fig3_user_depth.png", "Phân phối số booking trên mỗi user.")

    b.h2("1.3 Yêu cầu đối với phương pháp")
    b.table(
        [
            ["Đặc tính", "Yêu cầu đối với phương pháp"],
            [
                "Ngữ cảnh lượt tìm kiếm chi phối (điểm đến)",
                "Đưa ngữ cảnh hiện tại vào mô hình, học tổ hợp giữa các trường",
            ],
            [
                "Có lịch sử booking theo thời gian",
                "Khai thác thứ tự các booking trước đó",
            ],
            ["Lịch sử ngắn, nhiều user mới", "Vẫn xếp hạng tốt khi không có lịch sử"],
            [
                "Điểm đến nhiều giá trị, đuôi dài",
                "Chia sẻ tín hiệu cho giá trị hiếm, có mức dự phòng",
            ],
            [
                "Chỉ 100 cluster, không có thuộc tính item",
                "Xếp hạng toàn catalog với chi phí huấn luyện vừa phải",
            ],
            [
                "Hành vi thay đổi theo thời gian",
                "Chia theo thời gian, không rò rỉ thông tin tương lai",
            ],
        ],
        [3800, 5838],
    )
    b.p(
        [
            ("Hướng tiếp cận phù hợp: ", True),
            (
                "context-aware sequential recommendation — mô hình tuần tự có nhận ngữ cảnh lượt tìm kiếm hiện tại.",
                False,
            ),
        ]
    )

    # ------------------------------------------------------------ 2. candidates
    b.h1("2. Ba ứng viên")
    b.bullets(
        [
            [
                ("SMLP4Rec", True),
                (
                    " (Gao và cộng sự, ACM TOIS 2024 [1]): mô hình tuần tự thuần MLP, trộn thông tin song song theo ba trục chuỗi × đặc trưng × chiều embedding.",
                    False,
                ),
            ],
            [
                ("AdaGIN", True),
                (
                    " (Sang và cộng sự, ACM TOIS 2024 [3]): mô hình CTR dựng đồ thị tương tác giữa các trường cho từng mẫu.",
                    False,
                ),
            ],
            [
                ("LightGBM", True),
                (
                    " (Ke và cộng sự, NeurIPS 2017 [4]): gradient boosting trên cây quyết định cho dữ liệu dạng bảng.",
                    False,
                ),
            ],
        ]
    )
    b.table(
        [
            ["Tiêu chí", "SMLP4Rec", "AdaGIN", "LightGBM"],
            [
                [("Ý tưởng", True)],
                "3 MLP song song trên chuỗi × đặc trưng × embedding",
                "Đồ thị tương tác giữa các trường",
                "Cộng dồn cây quyết định sửa lỗi",
            ],
            [
                [("Ngữ cảnh lượt tìm kiếm", True)],
                "Qua trục đặc trưng và token truy vấn",
                "Tương tác trường tường minh",
                "Đặc trưng bảng, tương tác ngầm",
            ],
            [
                [("Lịch sử user", True)],
                "Trục chuỗi, có thứ tự",
                "Không mô hình hóa chuỗi",
                "Chỉ qua đặc trưng tổng hợp thủ công",
            ],
            [
                [("User ít lịch sử", True)],
                "Yếu — nhánh chuỗi rỗng",
                "Ổn — không dùng user ID",
                "Ổn nếu đủ đặc trưng ngữ cảnh",
            ],
            [
                [("Đơn vị huấn luyện", True)],
                "1 mẫu / booking, softmax 100 lớp",
                "1 dòng / (booking, cluster)",
                "1 dòng / booking, 100 cây mỗi vòng",
            ],
            [
                [("Hạn chế chính", True)],
                "Lịch sử user ngắn",
                "Chi phí tính toán",
                "Cách tiếp cận cũ, không học chuỗi",
            ],
        ],
        [2000, 2546, 2546, 2546],
    )

    # ------------------------------------------------------------ 3. not chosen
    b.h1("3. Vì sao không chọn AdaGIN và LightGBM")
    b.h2("3.1 AdaGIN — chi phí tính toán")
    b.p(
        "AdaGIN chấm điểm từng cặp (lượt tìm kiếm, cluster ứng viên). Xếp hạng 100 cluster đòi hỏi mở rộng mỗi booking thành "
        f"100 dòng: {num(HS['candidate_rows_train_x100'])} dòng train trước khi lấy mẫu âm và "
        f"{num(HS['candidate_rows_test_x100'])} dòng chấm điểm trên tập test, gấp 100 lần đơn vị huấn luyện của SMLP4Rec."
    )
    b.figure(
        "fig11_training_rows.png", "Số dòng huấn luyện trên tập train của ba ứng viên."
    )
    b.bullets(
        [
            [
                ("Tài nguyên: ", True),
                (
                    "cần GPU và lấy mẫu âm để huấn luyện trong thời gian dự án; chấm điểm test theo batch.",
                    False,
                ),
            ],
            [
                ("Loss theo điểm: ", True),
                (
                    "binary cross-entropy trên từng cặp không tối ưu trực tiếp thứ hạng giữa 100 ứng viên.",
                    False,
                ),
            ],
            [
                ("Mã nguồn: ", True),
                (
                    "khi rà soát mã của tác giả, nhóm ghi nhận nhiễu Gumbel vẫn bật ở chế độ đánh giá.",
                    False,
                ),
            ],
        ]
    )
    b.h2("3.2 LightGBM — cách tiếp cận cũ")
    b.bullets(
        [
            [
                ("Cách tiếp cận cũ (2017): ", True),
                (
                    "không học biểu diễn cho giá trị categorical và không mô hình hóa chuỗi hành vi.",
                    False,
                ),
            ],
            [
                ("Tương tác ngầm: ", True),
                (
                    "tổ hợp điểm đến × tháng × đoàn khách chỉ học được qua cây sâu, chia nhỏ dữ liệu như bảng tần suất.",
                    False,
                ),
            ],
            [
                ("Trường nhiều giá trị: ", True),
                (
                    f"{num(FIELD['srch_destination_id']['n_unique_all'])} điểm đến phải target-encode, dễ rò rỉ nhãn nếu không tính trên dữ liệu trước sự kiện.",
                    False,
                ),
            ],
        ]
    )
    b.h2("3.3 Vai trò trong benchmark")
    b.p(
        "Cả hai được giữ làm mô hình so sánh: AdaGIN đại diện cho hướng context-aware kiểu CTR (RQ1), LightGBM là baseline dạng bảng."
    )

    # ------------------------------------------------------------ 4. SMLP4Rec
    b.h1("4. SMLP4Rec")
    b.h2("4.1 Nguồn gốc")
    b.table(
        [
            ["Hạng mục", "Chi tiết"],
            [
                [("Bài báo", True)],
                "J. Gao, X. Zhao, M. Li, M. Zhao, R. Wu, R. Guo, Y. Liu, D. Yin. SMLP4Rec: An Efficient All-MLP Architecture for Sequential Recommendations. ACM TOIS 42(3), 2024 [1]; bản mở rộng của MLP4Rec [2]",
            ],
            [
                [("Mã nguồn", True)],
                "github.com/Applied-Machine-Learning-Lab/SMLP4Rec (RecBole 1.0 [7]), không có file giấy phép → cài lại từ bài báo, ghi rõ là điều chỉnh của nghiên cứu",
            ],
            [
                [("Dạng bài toán", True)],
                "Dự đoán item kế tiếp từ chuỗi tương tác có thứ tự; loss full softmax trên toàn bộ item",
            ],
        ],
        [1927, 7711],
    )
    b.h2("4.2 Ý tưởng")
    b.figure(
        "fig12_smlp4rec_architecture.png",
        "Kiến trúc SMLP4Rec áp dụng cho Expedia: chuỗi booking và token truy vấn của lượt tìm kiếm hiện tại.",
        16.5,
    )
    b.lead(
        "Bước 1 — Dựng khối đầu vào.",
        "Mỗi lượt dự đoán thành một khối (S+1) × F × C: S vị trí trong chuỗi (các booking gần nhất và token truy vấn), F trường cho mỗi vị trí, C chiều embedding.",
    )
    b.lead(
        "Bước 2 — Chuẩn hóa.",
        "Một LayerNorm theo trục C ở đầu mỗi lớp đưa các vector về cùng thang đo.",
    )
    b.lead(
        "Bước 3 — Trộn song song theo ba trục.",
        "Ba MLP chạy song song trên cùng đầu vào: dọc chuỗi (thời gian), dọc trường (ngữ cảnh), dọc chiều embedding. Kết quả được cộng lại; lớp này lặp L lần với tham số dùng chung.",
    )
    b.lead(
        "Bước 4 — Xếp hạng.",
        "Lấy vector tại hàng cluster của token truy vấn, nhân vô hướng với bảng embedding của 100 cluster, được 100 điểm xếp hạng.",
    )
    b.h2("4.3 Cách áp dụng cho Expedia")
    b.table(
        [
            ["Thành phần", "Đặc tả"],
            [
                [("Dựng mẫu", True)],
                "Script lọc booking, sắp theo user và thời gian; mỗi booking thứ k là một mẫu: lịch sử = các booking trước đó, nhãn = cluster của booking k",
            ],
            [
                [("Trường mỗi booking", True)],
                "Cluster ID; điểm đến, market; đoàn khách, package, tháng nhận phòng; lead time, số đêm (chia bucket); khoảng thời gian tới lượt hiện tại",
            ],
            [
                [("Token truy vấn", True)],
                "Vị trí cuối chuỗi; cluster = [MASK]; ngữ cảnh của lượt tìm kiếm hiện tại; market suy ra từ điểm đến qua train",
            ],
            [
                [("Độ dài chuỗi", True)],
                f"S khoảng 8–16 (p90 = {int(DEPTH['quantiles']['p90'])} booking mỗi user); đệm bên trái để token truy vấn luôn ở vị trí cuối; mask cho vị trí đệm",
            ],
            [
                [("Từ vựng", True)],
                "Xây từ train; điểm đến dưới 20 booking gộp vào [UNK]; ngưỡng chốt trên validation",
            ],
            [
                [("Loss", True)],
                "Cross-entropy trên 100 cluster, chỉ tính tại token truy vấn (bộ trộn chuỗi không nhân quả [6])",
            ],
            [
                [("Huấn luyện", True)],
                "Adam; early stopping và chọn siêu tham số theo NDCG@10 trên validation; tối thiểu ba seed",
            ],
        ],
        [2300, 7338],
    )
    b.h2("4.4 Nền tảng toán học")
    b.p(
        "Mỗi nhánh là một MLP hai lớp (mở rộng n → αn, GELU, thu về n) áp lên từng dải dọc một trục. Mỗi phần tử đầu ra là tổ hợp "
        "có trọng số của mọi phần tử trên dải; GELU làm cách trộn phi tuyến. Trên trục chuỗi, trọng số gắn với vị trí tuyệt đối "
        "nên mô hình nhạy với thứ tự mà không cần positional encoding — thành phần mà bài báo cho là làm nhiễu ngữ nghĩa của item "
        "embedding trong self-attention. Chi phí tuyến tính theo độ dài chuỗi."
    )
    b.p(
        "Khi huấn luyện, 100 điểm qua softmax thành xác suất; cross-entropy phạt khi cluster được đặt có xác suất thấp. Với 100 "
        "cluster, loss tính trên toàn bộ catalog, không cần lấy mẫu âm. Gradient cập nhật cùng lúc embedding, hai LayerNorm và ba MLP."
    )
    b.h2("4.5 Mức phù hợp với Expedia")
    b.table(
        [
            ["Đặc tính Expedia", "Cách SMLP4Rec đáp ứng", "Mức phù hợp"],
            [
                "Ngữ cảnh chi phối",
                "Token truy vấn + trục đặc trưng học tổ hợp như đoàn gia đình × package × điểm đến",
                [("Cao", True)],
            ],
            [
                "Có lịch sử theo thời gian",
                "Trục chuỗi nhận biết thứ tự; khoảng thời gian là một trường",
                [("Cao", True)],
            ],
            [
                "Chỉ 100 cluster",
                "Softmax toàn catalog; 1 mẫu mỗi booking",
                [("Cao", True)],
            ],
            [
                "Điểm đến nhiều giá trị",
                "Embedding + [UNK]; điểm đến hiếm cần thêm prior (Mục 5)",
                [("Trung bình", True)],
            ],
            [
                "Lịch sử ngắn, nhiều user mới",
                "Nhánh chuỗi rỗng, chỉ còn ngữ cảnh (Mục 5)",
                [("Thấp khi dùng riêng", True)],
            ],
        ],
        [2600, 5238, 1800],
    )

    # ------------------------------------------------------------ 5. weakness and hybrid
    b.h1("5. Điểm yếu và đề xuất hybrid")
    b.h2("5.1 Điểm yếu lớn nhất: lịch sử user ngắn")
    b.p(
        f"{pct(ON['0']['share'])} booking test không có booking nào trước đó và {pct(LE1)} có tối đa một booking. Với nhóm này nhánh "
        "chuỗi gần như rỗng, mô hình chỉ còn ngữ cảnh và phải tự ghi nhớ bảng tần suất theo điểm đến — bảng mà riêng nó đã đạt "
        f"Recall@5 {pct(R5['popularity_by_srch_destination_id'])}. Rủi ro là SMLP4Rec thua bảng đếm ở nhóm user ít lịch sử."
    )
    b.figure(
        "fig8_history_buckets.png",
        "Tỷ trọng booking test theo số booking trước đó của user.",
    )
    b.h2("5.2 Cơ chế hybrid")
    b.figure(
        "fig13_hybrid_flow.png",
        "Hybrid: điểm SMLP4Rec cộng prior theo ngữ cảnh, trọng số do gate học.",
        16.0,
    )
    b.bullets(
        [
            [
                ("Prior theo ngữ cảnh (ghi nhớ [5]): ", True),
                (
                    "bảng đếm cluster tại điểm đến đang tìm, làm mượt về market khi ít dữ liệu, dùng phân phối toàn cục khi điểm đến chưa từng thấy. Chỉ tính từ các booking trước sự kiện nên không rò rỉ nhãn.",
                    False,
                ),
            ],
            [
                ("Gate (tự điều chỉnh): ", True),
                (
                    "một trọng số cho mỗi ô (nhóm độ dài lịch sử × nhóm độ phổ biến điểm đến), học cùng mô hình, khởi tạo bằng 1, ràng buộc không âm.",
                    False,
                ),
            ],
            [
                ("Nối lịch sử với ngữ cảnh: ", True),
                (
                    "mỗi booking lịch sử có thêm trường “cùng điểm đến với lượt tìm kiếm”.",
                    False,
                ),
            ],
        ]
    )
    b.table(
        [
            ["Nhóm user", "Số booking trước đó", "Kỳ vọng"],
            ["User mới", "0", "Prior dẫn dắt"],
            ["Ít lịch sử", "1–4", "Kết hợp prior và SMLP4Rec"],
            ["Nhiều lịch sử", "≥ 5", "SMLP4Rec dẫn dắt"],
        ],
        [3000, 3000, 3638],
    )
    b.p(
        "Giá trị gate chưa có; chúng được học khi huấn luyện và sẽ được báo cáo như một kết quả của nghiên cứu."
    )

    b.h2("5.3 Ngưỡng và căn cứ")
    b.p(
        "Ngưỡng được chọn từ Recall@5 của ba luật xếp hạng bằng bảng đếm trên tập test chia theo thời gian. Đây không phải kết quả "
        "của SMLP4Rec hay hybrid; chúng dùng để đặt ngưỡng và làm mốc mà mô hình phải vượt."
    )
    b.figure(
        "fig9_rules_by_history.png",
        "Recall@5 của ba luật đếm theo số booking trước đó của user.",
    )
    b.p(
        f"Luật dùng lịch sử cùng điểm đến vượt prior thêm {dec(gain('1') * 100, 1)} điểm ở nhóm 1 booking, "
        f"{dec(gain('5-9') * 100, 1)} điểm ở nhóm 5–9 và {dec(gain('10+') * 100, 1)} điểm ở nhóm 10+. Ngược lại, đưa mọi cluster "
        f"từng đặt lên đầu mà không xét điểm đến chỉ đạt {pct(ON['5-9']['R@5_history_then_dest'])} ở nhóm 5–9, thấp hơn prior "
        f"({pct(ON['5-9']['R@5_dest_raw_backoff_market'])}): lịch sử chỉ có giá trị khi được nối với ngữ cảnh hiện tại."
    )
    b.figure(
        "fig10_dest_support_smoothing.png",
        "Recall@5 của prior theo điểm đến, thô và làm mượt về market, theo số booking của điểm đến.",
    )
    b.table(
        [
            ["Ngưỡng", "Nhóm", "Căn cứ"],
            [
                [("Độ dài lịch sử user", True)],
                "0 | 1 | 2–4 | 5–9 | 10+",
                (
                    f"0: không có lịch sử. 2 và 3–4 gộp vì lợi ích gần nhau ({dec(gain('2') * 100, 1)} và {dec(gain('3-4') * 100, 1)} điểm). "
                    f"Từ 5: lợi ích trên 6 điểm. 10+: cluster đúng nằm trong lịch sử {pct(ON['10+']['target_in_history'], 1)} trường hợp"
                ),
            ],
            [
                [("Độ phổ biến điểm đến (số booking train)", True)],
                "0 | 1–19 | 20+",
                (
                    f"1–4 booking: làm mượt tăng Recall@5 từ {pct(SUP['1-4']['R@5_dest_raw_backoff_market'], 1)} lên "
                    f"{pct(SUP['1-4']['R@5_dest_smooth_m5'], 1)}; từ 20 booking làm mượt không còn tác dụng; điểm đến chưa từng thấy "
                    f"chỉ đạt {pct(SUP['0']['R@5_dest_raw_backoff_market'], 1)}"
                ),
            ],
        ],
        [2400, 2100, 5138],
    )
    b.p(
        "Các ngưỡng rút ra từ phép chia 80/20 dùng cho phân tích; bản chính thức được chốt lại trên tập validation."
    )

    # ------------------------------------------------------------ 6. decision
    b.h1("6. Quyết định và kế hoạch")
    b.h2("6.1 Quyết định")
    b.p(
        [
            ("SMLP4Rec kết hợp hybrid cho user ít lịch sử", True),
            (
                (
                    " là phương pháp chính: SMLP4Rec đưa ngữ cảnh lượt tìm kiếm và lịch sử có thứ tự vào một mô hình nhẹ; hybrid bù cho "
                    f"{pct(ON['0']['share'])} booking không có lịch sử. AdaGIN và LightGBM được giữ làm mô hình so sánh."
                ),
                False,
            ),
        ]
    )
    b.h2("6.2 Rủi ro và biện pháp")
    b.table(
        [
            ["Rủi ro", "Biện pháp"],
            [
                "Lịch sử user ngắn",
                "Hybrid prior + gate; báo cáo mọi kết quả theo nhóm độ dài lịch sử",
            ],
            [
                "Bộ trộn chuỗi không nhân quả",
                "Chỉ tính loss tại token truy vấn; lịch sử chỉ gồm booking trước sự kiện",
            ],
            ["Vị trí đệm khác 0 sau LayerNorm", "Nhân mask sau mỗi lớp"],
            [
                "Trọng số trộn tĩnh, khó chú ý có chọn lọc",
                "Trường “cùng điểm đến”; so sánh với SASRec có token truy vấn",
            ],
            ["Bỏ residual chưa có bằng chứng", "Ablation có và không có residual"],
            [
                "Tự cài lại mô hình",
                "Unit test cho dựng mẫu, mask, metric; so sánh với cấu hình của bài báo",
            ],
            ["Ngưỡng chưa chốt", "Chọn lại trên validation, báo cáo test một lần"],
        ],
        [3600, 6038],
    )
    b.h2("6.3 Mốc kiểm định (Recall@5, tập test)")
    b.table(
        [
            ["Mốc", "Recall@5", "Vai trò"],
            [
                "Tần suất toàn cục",
                pct(R5["global_popularity"]),
                "Thấp hơn mức này: lỗi cài đặt",
            ],
            [
                "Lặp lại cluster gần nhất",
                pct(R5["repeat_last_cluster"]),
                "Giới hạn của riêng yếu tố lặp lại",
            ],
            [
                "Tần suất điểm đến × tháng nhận phòng",
                pct(R5["popularity_by_destination_x_ci_month"]),
                "Phần mất do phân mảnh số đếm",
            ],
            [
                "Tần suất điểm đến",
                pct(R5["popularity_by_srch_destination_id"]),
                "Mốc tối thiểu của mô hình học máy",
            ],
            [
                [("Luật đếm tốt nhất: lịch sử cùng điểm đến + prior", True)],
                [(pct(BEST_RULE), True)],
                "Mốc cần vượt tổng thể (chưa phải kết quả mô hình)",
            ],
            [
                "Prior làm mượt, user không có lịch sử",
                pct(COLD_FLOOR),
                "Mốc tối thiểu cho user mới",
            ],
        ],
        [4200, 1400, 4038],
    )
    b.h2("6.4 Bước tiếp theo")
    b.bullets(
        [
            "Pipeline dữ liệu: dựng chuỗi booking và token truy vấn, chia train / validation / test theo thời gian.",
            "Baseline: tần suất, luật lịch sử cùng điểm đến, LightGBM.",
            "SMLP4Rec: bản không có hybrid, tinh chỉnh trên validation, nhiều seed.",
            "Hybrid: prior + gate, chốt ngưỡng trên validation.",
            "So sánh và ablation: AdaGIN; bỏ từng nhánh, prior, trường cùng điểm đến; cắt theo lịch sử và mùa.",
        ]
    )

    # ------------------------------------------------------------ 7. limitations
    b.h1("7. Giới hạn")
    b.bullets(
        [
            "Hotel cluster là đại diện ẩn danh cho hạng phòng hoặc gói; kết luận kinh doanh kế thừa giới hạn này.",
            "Kết quả trong bài báo SMLP4Rec được đo với 100 item âm lấy mẫu trên dữ liệu đã lọc user dưới 5 tương tác; Expedia có trung vị 2 booking mỗi user.",
            "SMLP4Rec và hybrid chưa được đo; mọi con số ở Mục 5 và 6.3 là của luật đếm.",
            "Ngưỡng rút từ phép chia 80/20 dùng cho phân tích, cần chốt lại trên validation.",
            "Bảng xếp hạng cuộc thi Expedia bị ảnh hưởng bởi đường rò rỉ qua vị trí user và khoảng cách; điểm công bố không dùng làm mục tiêu.",
            "Điều khoản dữ liệu chỉ cho phép dùng cho nghiên cứu; không phân phối lại dữ liệu thô.",
        ]
    )

    # ------------------------------------------------------------ references
    b.h1("Tài liệu tham khảo")
    for ref in [
        "[1] J. Gao, X. Zhao, M. Li, M. Zhao, R. Wu, R. Guo, Y. Liu, D. Yin. SMLP4Rec: An Efficient All-MLP Architecture for Sequential Recommendations. ACM TOIS, 2024. doi:10.1145/3637871",
        "[2] M. Li, X. Zhao, C. Lyu, M. Zhao, R. Wu, R. Guo. MLP4Rec: A Pure MLP Architecture for Sequential Recommendations. IJCAI, 2022",
        "[3] L. Sang, H. Li, Y. Zhang, Y. Zhang, Y. Yang. AdaGIN: Adaptive Graph Interaction Network for Click-Through Rate Prediction. ACM TOIS, 2024. doi:10.1145/3681785",
        "[4] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017",
        "[5] H.-T. Cheng et al. Wide & Deep Learning for Recommender Systems. DLRS, 2016",
        "[6] TriMLP: Revenge of a MLP-like Architecture in Sequential Recommendation. arXiv:2305.14675, 2023",
        "[7] W. X. Zhao et al. RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. CIKM, 2021",
        "[8] Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations. RecSys, 2024. doi:10.1145/3640457.3688195",
        "[9] Y. Fan, Y. Ji, J. Zhang, A. Sun. Our Model Achieves Excellent Performance on MovieLens: What Does It Mean? ACM TOIS, 2024. doi:10.1145/3675163",
    ]:
        b.p(ref, "ref")
    return b


def main() -> None:
    b = build()
    b.save(DOCX)
    print(f"wrote {DOCX} ({b.fig_no} figures)")


if __name__ == "__main__":
    main()
