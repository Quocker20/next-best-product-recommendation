"""Build the week-3 implementation report (Vietnamese, .docx): SMLP4Rec on Expedia, the plain
run, and the experiment chain (top-5 rules -> destination prior -> sameDest hybrid ->
behaviour split). Companion to the deck reports/summary/week3_implementation/week3_report_slides_v2.html.

Format: the week-2 report (reports/summary/week2_methodology/methodology_selection_report.docx)
is the template: its styles, page setup, footer and table-of-contents field are kept, the body
is regenerated with the XML helpers of scripts/week2_methodology_report.py. The week-2 file is
only read, never written.

Every number is read from persisted outputs in results/week3_implementation/:
smlprec_expedia_run.json (notebook 01), smlprec_expedia_mixed_run.json and
smlprec_expedia_dynamic_cap_run.json (notebook 02), smlprec_expedia_late_fusion.json
(notebook 03), smlprec_expedia_hybrid_samedest.json (notebook 04),
smlprec_expedia_hybrid_behaviour_split.json (notebook 05),
smlprec_expedia_seen_unseen_users.json (notebook 06); and from results/week4_rebuild/:
basic_baselines.json (ItemKNN + logistic regression, notebook 07), basic_baselines_verify.json
(independent check, scripts/verify_basic_baselines.py), smlprec_query_run.json (query token and the
prior-fusion check, notebook 01c).
Figures: results/figures/vi/week3_fig1-7 (scripts/week3_report_figures.py, run it first) and
fig12_smlp4rec_architecture.png from the week-2 report.

Usage: python scripts/week3_implementation_report.py [--no-word]
Output: reports/summary/week3_implementation/week3_implementation_report.docx
With Microsoft Word installed (Windows), the table of contents is refreshed through Word COM;
--no-word skips that (then press F9 on the table of contents in Word).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "reports" / "summary" / "week2_methodology" / "methodology_selection_report.docx"
OUT = ROOT / "reports" / "summary" / "week3_implementation" / "week3_implementation_report.docx"
W3 = ROOT / "results" / "week3_implementation"
FIG = ROOT / "results" / "figures" / "vi"
TABLE_WIDTH = 9638
FOOTER_OLD = "Lựa chọn phương pháp"
FOOTER_NEW = "Triển khai và thử nghiệm (tuần 3)"


def load(name: str) -> dict:
    return json.loads((W3 / name).read_text(encoding="utf-8"))


RUN = load("smlprec_expedia_run.json")
MIX = load("smlprec_expedia_mixed_run.json")
CAP = load("smlprec_expedia_dynamic_cap_run.json")
LF = load("smlprec_expedia_late_fusion.json")
HY = load("smlprec_expedia_hybrid_samedest.json")
BS = load("smlprec_expedia_hybrid_behaviour_split.json")
SU = load("smlprec_expedia_seen_unseen_users.json")
W4 = ROOT / "results" / "week4_rebuild"
BL = json.loads((W4 / "basic_baselines.json").read_text(encoding="utf-8"))
BV = json.loads((W4 / "basic_baselines_verify.json").read_text(encoding="utf-8"))
QT = json.loads((W4 / "smlprec_query_run.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- number formatting


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def _r(x: float, d: int) -> float:
    """Round and drop the sign of a zero, so -0.00001 prints as 0,0000, not -0,0000."""
    return round(x, d) + 0.0


def dec(x: float, d: int = 4) -> str:
    return f"{_r(x, d):.{d}f}".replace(".", ",")


def sdec(x: float, d: int = 4) -> str:
    v = _r(x, d)
    return (f"{v:+.{d}f}" if v else f"{0:.{d}f}").replace(".", ",")


def pp(x: float, d: int = 2) -> str:
    """Difference of two shares, in percentage points."""
    v = _r(x * 100, d)
    return (f"{v:+.{d}f}" if v else f"{0:.{d}f}").replace(".", ",") + " điểm %"


def day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%d/%m/%Y")


def wdec(v: float) -> str:
    """Weight with the trailing zeros dropped (1.25 -> 1,25; 2.0 -> 2)."""
    return dec(v, 2).rstrip("0").rstrip(",")


def ci_recall(d: dict) -> str:
    lo, hi = d["recall@5"]["ci95"]
    return f"{pp(d['recall@5']['diff'])} [{dec(lo * 100, 2)}; {dec(hi * 100, 2)}]"


def ci_ndcg(d: dict) -> str:
    lo, hi = d["ndcg@5"]["ci95"]
    return f"{sdec(d['ndcg@5']['diff'])} [{dec(lo)}; {dec(hi)}]"


# ---------------------------------------------------------------- shorthand data

T, V = RUN["test"], RUN["valid"]
PE = RUN["per_epoch"]
HEUR = RUN["heuristics_same_test_rows"]
POP, REP = HEUR["global_popularity"], HEUR["repeat_last_then_popularity"]
DS, CFG, SP = RUN["dataset_stats"], RUN["config"], RUN["splits"]
TB = RUN["top5_behavior"]["test"]
MX = RUN["target_mix"]
mp, mm = (
    MIX["final_test_at_best_mixed_epoch"]["plain"],
    MIX["final_test_at_best_mixed_epoch"]["mixed"],
)
cc = CAP["final_test_at_best_capped_epoch"]["capped"]


def exchange(rule: dict) -> float:
    """Old-target hits lost per new-target hit gained, against plain."""
    return (mp["old_target_hit_of_all_rows"] - rule["old_target_hit_of_all_rows"]) / (
        rule["new_target_hit_of_all_rows"] - mp["new_target_hit_of_all_rows"]
    )


W_BEST = LF["best_global_w_on_valid"]
FKEY = f"fusion_global_w={W_BEST}"
LV = LF["variants"]["test"]
K_PLAIN, K_PRIOR, K_RULE = (
    "plain_model (w=0)",
    "destination_prior_only",
    "same_dest_history_then_prior",
)
LC = LF["combined_warm_plus_cold"]["test"]
COLD = LF["cold_users_prior_only"]["test"]
LB = LF["bootstrap_test_95ci"]
K_HYBRID = "SMLP4Rec + prior + sameDest (this notebook)"
K_REF = HY["reference"]["name"]
HWARM = HY["benchmark"]["test"]["warm"]
HALL = HY["benchmark"]["test"]["all_events"]
HB = HY["bootstrap_test_95ci"]["hybrid minus SMLP4Rec + prior"]
SEL = HY["selection"]
H_ALPHA = SEL["one_se_pick"][0]
HW = HY["cohorts"]["final_weights_per_cell"]  # "0" new destination, "1" known destination
SL = HY["test_slices_recall@5_ndcg@5"]
CC = HY["component_check_test"][K_HYBRID]
EX12 = (
    0.7**3 + 0.7**0
)  # worked sameDest example: one cluster booked at the destination at ages 0 and 3

BW, BA = BS["benchmark_test"]["warm"], BS["benchmark_test"]["all_events"]
BB = BS["bootstrap_test_95ci"]["best behaviour scheme minus reference"]
B_CUT = BS["cut"]["chosen_on_train"]
B_BEST = BS["best_behaviour_scheme"]
B_CELLS = BS["final_weights_per_cell"]
STUMP = BS["cut"]["stump_on_train"][str(B_CUT)]

# all-events views (warm + cold users) for the summary table
ALL = {
    "plain": LC["plain SMLP4Rec on warm + global popularity on cold"],
    "prior": LC["prior-only for everyone"],
    "rule": LC["week-2 heuristic (same-dest history, prior when cold)"],
    "fusion": LC["hybrid: fusion on warm + prior-only on cold"],
    "hybrid": HALL[K_HYBRID],
}
WARM = {
    "pop": POP,
    "plain": LV[K_PLAIN]["all"],
    "prior": LV[K_PRIOR]["all"],
    "rule": LV[K_RULE]["all"],
    "fusion": LV[FKEY]["all"],
    "hybrid": HWARM[K_HYBRID],
}


def slice_pair(name: str, key: str) -> tuple[float, float]:
    r, n = SL[name][key].split(" / ")
    return float(r), float(n)


SCHEME_VI = {
    "global (1 set)": "Một bộ chung",
    "known vs new destination (2)": "Điểm đến đã đặt / mới (2)",
    "known vs new destination (2) [notebook 04]": "Điểm đến đã đặt / mới (2) [B2]",
    "L=1 vs L>=2 (2)": "L = 1 / L ≥ 2 (2)",
    "L = 1 | 2-4 | 5+ (3)": "L = 1 | 2–4 | 5+ (3)",
    "4 L buckets (4)": "4 nhóm L (4)",
    "L=1 vs L>=2 x known/new (4)": "L = 1 / L ≥ 2 × điểm đến đã đặt / mới (4)",
    "old-leaning vs rest (2)": "Thiên cũ / còn lại (2)",
    "L=1 | new-leaning | old-leaning (3)": "L = 1 | thiên mới | thiên cũ (3)",
    "known/new destination x old-leaning/rest (4)": "Điểm đến × thiên cũ / còn lại (4)",
    "known/new destination x (L=1 | new | old) (6)": "Điểm đến × (L = 1 | thiên mới | thiên cũ) (6)",
}
CELL_VI = {
    "destination new to the user": "Điểm đến mới với user",
    "destination already in history": "Điểm đến đã đặt trước đó",
    "new dest, rest": "Điểm đến mới · còn lại",
    "new dest, old-leaning": "Điểm đến mới · thiên cũ",
    "known dest, rest": "Đã đặt · còn lại",
    "known dest, old-leaning": "Đã đặt · thiên cũ",
}


def split_gain(s: str) -> str:
    """'+0.00029 (SE 0.00024)' -> '+0,00029 (SE 0,00024)'."""
    return s.replace(".", ",")


def metric_row(name, d, bold: bool = False):
    cells = [
        dec(d["ndcg@5"]),
        pct(d["recall@5"]),
        dec(d["ndcg@10"]),
        pct(d["recall@10"]),
        dec(d["ndcg@20"]),
        pct(d["recall@20"]),
    ]
    if bold:
        return [[(name, True)]] + [[(c, True)] for c in cells]
    return [name] + cells


HYB_K = "hybrid (SMLP4Rec + prior + sameDest)"
PLN_K = "plain SMLP4Rec"
_bt = BL["results"]["test"]
BL_R = {
    "pop": _bt["global popularity"],
    "knn": _bt[next(k for k in _bt if k.startswith("ItemKNN"))],
    "lr": _bt[next(k for k in _bt if k.startswith("Logistic"))],
}
BL_KNN = next(k for k in BL["models"] if k.startswith("ItemKNN"))
BL_LR = next(k for k in BL["models"] if k.startswith("Logistic"))
METRIC_HEAD = ["Cách", "NDCG@5", "Recall@5", "NDCG@10", "Recall@10", "NDCG@20", "Recall@20"]
METRIC_W = [3238, 1050, 1100, 1050, 1100, 1050, 1050]


# ---------------------------------------------------------------- XML builders (copied from
# scripts/week2_methodology_report.py so both reports share one look)

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
                else ('<w:shd w:val="clear" w:color="auto" w:fill="F5F7FA"/>' if shade else "")
            )
            parts = [(cell, is_head)] if isinstance(cell, str) else cell
            tcs.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:tcBorders>{BORDER}</w:tcBorders>'
                f"{shd}{CELL_MAR}</w:tcPr><w:p><w:pPr>{PARA_PR['cell']}</w:pPr>"
                f"{runs_xml(parts, RUN_PR['cell'])}</w:p></w:tc>"
            )
        trpr = "<w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if is_head else "") + "</w:trPr>"
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
        Paragraph(el, self.doc._body).add_run().add_picture(str(FIG / name), width=Cm(width_cm))
        self.fig_no += 1
        self.p(f"Hình {self.fig_no}. {caption}", "caption")

    def save(self, path: Path):
        self.doc.save(str(path))


# ---------------------------------------------------------------- document


def build() -> Builder:
    b = Builder(TEMPLATE)

    # ------------------------------------------------------------ cover block
    b.p("Triển khai SMLP4Rec trên Expedia và các thử nghiệm bổ sung", "title")
    b.p(
        "Báo cáo tuần 3: từ code gốc đến hybrid SMLP4Rec + prior điểm đến + sameDest, "
        "đánh giá trên Expedia Hotel Recommendations",
        "subtitle",
    )
    b.table(
        [
            ["Hạng mục", "Nội dung"],
            [
                [("Bài toán", True)],
                (
                    "Xếp hạng 100 hotel cluster cho lần booking tiếp theo của user, dựa trên lịch sử booking "
                    "và điểm đến của lượt tìm kiếm"
                ),
            ],
            [
                [("Dữ liệu", True)],
                (
                    f"Expedia: {num(DS['users'])} user có booking, {DS['items']} cụm, "
                    f"{num(DS['sequence_targets'])} mục tiêu “đặt tiếp theo”"
                ),
            ],
            [
                [("Giao thức", True)],
                (
                    f"Chia theo thời gian 80/10/10 trên mục tiêu ({day(SP['train']['first_target_unix'])} → "
                    f"{day(SP['test']['last_target_unix'])}); xếp hạng đầy đủ 100 cụm; Recall@K, NDCG@K, K = 5, 10, 20; "
                    f"một seed ({CFG['seed']})"
                ),
            ],
            [
                [("Kết quả chính", True)],
                [
                    (
                        (
                            f"NDCG@5 {dec(WARM['plain']['ndcg@5'])} → {dec(WARM['hybrid']['ndcg@5'])}, "
                            f"Recall@5 {pct(WARM['plain']['recall@5'])} → {pct(WARM['hybrid']['recall@5'])}"
                        ),
                        True,
                    ),
                    (" (SMLP4Rec plain → hybrid, user có lịch sử, tập test)", False),
                ],
            ],
            [
                [("Quyết định", True)],
                (
                    "SMLP4Rec + prior điểm đến + sameDest, 2 bộ trọng số (điểm đến mới / đã đặt); "
                    "user mới chỉ dùng prior; không tách theo thói quen"
                ),
            ],
            [
                [("Phiên bản", True)],
                (
                    "07/10/2026 — đi kèm slide week3_report_slides_v2; notebook 01–07 và 01c trong "
                    "notebooks/hospitality/smlp4rec/"
                ),
            ],
        ],
        [1927, 7711],
    )
    if b.toc is not None:
        # the table of contents gets its own page: on the cover page its closing field paragraph
        # spills over and leaves an empty page before section 1
        toc_title = b.toc.find(qn("w:sdtContent")).find(qn("w:p"))
        toc_title.get_or_add_pPr().append(parse_xml(f"<w:pageBreakBefore {nsdecls('w')}/>"))
        b.add(b.toc)

    # ------------------------------------------------------------ 1. summary
    # page break carried by the heading itself: a separate break paragraph after a full
    # table-of-contents page would leave an empty page
    first = b.add(para_el("1. Tóm tắt", "h1"))
    first.pPr.append(parse_xml(f"<w:pageBreakBefore {nsdecls('w')}/>"))
    b.h2("1.1 Mục tiêu tuần 3")
    b.p(
        "Tuần 2 chọn SMLP4Rec làm phương pháp chính, kèm hybrid cho user ít lịch sử. Tuần 3 có ba việc: "
        "(1) chạy được code gốc SMLP4Rec trên dữ liệu booking Expedia với chỉnh sửa tối thiểu; "
        "(2) đo bản plain (chỉ lịch sử booking) và tìm điểm yếu; (3) thử các hướng bổ sung, mỗi hướng một giả thuyết, "
        "chọn trên valid và đọc test một lần."
    )
    b.h2("1.2 Kết quả chính")
    b.bullets(
        [
            [
                ("Plain có tín hiệu nhưng thấp. ", True),
                (
                    (
                        f"NDCG@5 {dec(T['ndcg@5'])}, Recall@5 {pct(T['recall@5'])} trên test, gấp "
                        f"{dec(T['ndcg@5'] / POP['ndcg@5'], 1)} lần popularity toàn cục về NDCG@5."
                    ),
                    False,
                ),
            ],
            [
                ("Phần lớn lần đặt tiếp theo là cụm mới. ", True),
                (
                    (
                        f"{pct(MX['test']['new'], 1)} mục tiêu test là cụm user chưa từng đặt; plain chủ yếu gợi lại "
                        f"cụm cũ (top 1 = cụm đặt gần nhất ở {pct(TB['top1_is_last_booking'], 1)} dòng) nên chỉ trúng "
                        f"{pct(TB['new_target_in_top5_of_new_rows'], 1)} dòng cụm mới."
                    ),
                    False,
                ),
            ],
            [
                ("Ép tỉ lệ cụm mới trong top 5 làm giảm điểm. ", True),
                (
                    (
                        f"NDCG@5 {dec(mp['ndcg@5'])} → {dec(mm['ndcg@5'])} (cố định 2 cũ + 3 mới) và "
                        f"{dec(cc['ndcg@5'])} (trần theo L); hướng này bị loại."
                    ),
                    False,
                ),
            ],
            [
                ("Prior điểm đến là đòn bẩy lớn nhất. ", True),
                (
                    (
                        f"SMLP4Rec + prior (w_p = {dec(W_BEST, 1)}): NDCG@5 {dec(WARM['plain']['ndcg@5'])} → "
                        f"{dec(WARM['fusion']['ndcg@5'])}, Recall@5 {pct(WARM['plain']['recall@5'])} → "
                        f"{pct(WARM['fusion']['recall@5'])}, không huấn luyện lại."
                    ),
                    False,
                ),
            ],
            [
                ("sameDest thêm phần còn thiếu. ", True),
                (
                    (
                        f"Hybrid ba tín hiệu đạt NDCG@5 {dec(WARM['hybrid']['ndcg@5'])}, Recall@5 "
                        f"{pct(WARM['hybrid']['recall@5'])}; so với SMLP4Rec + prior: NDCG@5 {ci_ndcg(HB['warm'])}, "
                        f"Recall@5 {ci_recall(HB['warm'])} (bootstrap 95%)."
                    ),
                    False,
                ),
            ],
            [
                ("Tách trọng số theo thói quen đặt lại không thêm gì. ", True),
                (
                    f"NDCG@5 {ci_ndcg(BB['warm'])}, Recall@5 {ci_recall(BB['warm'])}; giữ 2 bộ trọng số.",
                    False,
                ),
            ],
            [
                ("Hai baseline cơ bản giải thích điểm cao của hybrid. ", True),
                (
                    f"Trên mọi sự kiện test, hồi quy logistic chỉ dùng điểm đến và ngữ cảnh đạt NDCG@5 "
                    f"{dec(BL_R['lr']['all events']['ndcg@5'])}, ItemKNN chỉ dùng lịch sử đạt "
                    f"{dec(BL_R['knn']['all events']['ndcg@5'])}, hybrid đạt "
                    f"{dec(SU['results']['test']['all events'][HYB_K]['ndcg@5'])}; không thấy dấu hiệu rò rỉ hay lỗi "
                    "chia tập (mục 6.3).",
                    False,
                ),
            ],
        ]
    )
    b.figure(
        "week3_fig3_experiment_chain.png",
        "NDCG@5 và Recall@5 trên tập test (user có lịch sử) qua chuỗi thử nghiệm.",
    )

    # ------------------------------------------------------------ 2. implementation
    b.h1("2. Triển khai SMLP4Rec trên Expedia")
    b.h2("2.1 Nhắc lại mô hình")
    b.p(
        "SMLP4Rec [1] là mô hình gợi ý tuần tự toàn MLP: chuỗi cụm đã đặt đi qua ba MLP trộn song song theo chiều "
        "thời gian (vị trí trong chuỗi), chiều context và chiều embedding, lặp L lớp. Hàng cuối của chuỗi được so "
        "với embedding của 100 cụm để cho điểm xếp hạng; hàm mất mát là cross-entropy trên toàn bộ 100 cụm."
    )
    b.figure(
        "fig12_smlp4rec_architecture.png",
        "Kiến trúc SMLP4Rec (từ báo cáo tuần 2). Bản chạy tuần 3 chỉ dùng lịch sử booking, chưa có query token.",
    )
    b.h2("2.2 Từ code gốc đến bản cho Expedia")
    b.p(
        "Code gốc (bản fork MLP4Rec [3] trên RecBole 1.0.1 [2]) được giữ nguyên tối đa: một khối dùng chung cho mọi "
        "lớp, không residual. Chỉ sửa hai chỗ trong mã mô hình (đánh dấu ADAPTED) để chạy được với dữ liệu không "
        "có feature item."
    )
    b.table(
        [
            ["Hạng mục", "Code gốc", "Bản cho Expedia"],
            [
                "Dữ liệu",
                "MovieLens, item có feature",
                "Booking Expedia: user, cụm (100), thời gian; cụm không có feature",
            ],
            ["Độ dài chuỗi", "Cố định 50", f"Lấy từ cấu hình (= {CFG['MAX_ITEM_LIST_LENGTH']})"],
            ["Feature item", "Bắt buộc có", "Cho phép rỗng, bỏ lớp feature"],
            [
                "Chia và đánh giá",
                "Leave-one-out, 100 mẫu âm theo popularity",
                "Chia theo thời gian 80/10/10, xếp hạng đủ 100 cụm",
            ],
            [
                "Cấu hình",
                "Cấu hình bài báo, quá chậm trên CPU",
                f"{CFG['n_layers']} lớp, hidden {CFG['hidden_size']}, dropout {dec(CFG['hidden_dropout_prob'], 0)}",
            ],
            [
                "Môi trường",
                "RecBole 1.0.1 (numpy 1.23, pandas 1.5)",
                "Venv Python 3.11 riêng + 2 bản vá nhỏ",
            ],
        ],
        [2000, 3400, 4238],
        shade_first_col=True,
    )
    b.h2("2.3 Pipeline dữ liệu và chia tập")
    b.p(
        "train.csv (4 GB) được đọc theo khối, chỉ giữ dòng is_booking = 1, xuất thành bảng tương tác user · cụm · "
        "thời gian cho RecBole. Mỗi booking trừ booking đầu tiên của user là một mục tiêu “đặt tiếp theo”, với "
        f"lịch sử là tối đa {CFG['MAX_ITEM_LIST_LENGTH']} booking trước đó. Mục tiêu được chia theo thời gian "
        "toàn cục; booking đầu tiên của mỗi user (user mới, L = 0) không phải mục tiêu của SMLP4Rec và được chấm "
        "riêng bằng prior (Mục 5.1)."
    )
    b.table(
        [["Tập", "Số mục tiêu", "Từ ngày", "Đến ngày", "Độ dài lịch sử TB"]]
        + [
            [
                n,
                num(SP[k]["targets"]),
                day(SP[k]["first_target_unix"]),
                day(SP[k]["last_target_unix"]),
                dec(SP[k]["mean_history_len"], 2),
            ]
            for k, n in (("train", "Train"), ("valid", "Valid"), ("test", "Test"))
        ],
        [1838, 1950, 1950, 1950, 1950],
    )
    b.h2("2.4 Cấu hình và chi phí")
    tm = RUN["timing_seconds"]
    b.table(
        [
            ["Tham số", "Giá trị", "Tham số", "Giá trị"],
            ["Số lớp", str(CFG["n_layers"]), "Batch", num(CFG["train_batch_size"])],
            ["Hidden", str(CFG["hidden_size"]), "Learning rate", dec(CFG["learning_rate"], 3)],
            ["Dropout", dec(CFG["hidden_dropout_prob"], 1), "Số epoch", str(CFG["epochs"])],
            [
                "Độ dài chuỗi tối đa",
                str(CFG["MAX_ITEM_LIST_LENGTH"]),
                "Chọn epoch theo",
                "valid " + CFG["valid_metric"],
            ],
            ["Hàm mất mát", "Cross-entropy trên 100 cụm", "Seed", str(CFG["seed"])],
            [
                "Số tham số",
                num(RUN["n_parameters"]),
                "Thời gian train + valid (CPU)",
                f"{dec(tm['train_and_valid'], 1)} giây",
            ],
        ],
        [2400, 2419, 2400, 2419],
        shade_first_col=True,
    )

    # ------------------------------------------------------------ 3. plain run
    b.h1("3. Kết quả chạy plain")
    b.h2("3.1 Kết quả tổng")
    b.p(
        f"Epoch tốt nhất theo valid là epoch {RUN['best_epoch_by_valid']}. Hai luật đếm được chấm trên đúng các "
        "dòng test để làm mốc: popularity toàn cục, và lặp lại cụm đặt gần nhất rồi đến popularity."
    )
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("Popularity toàn cục", POP),
            metric_row("Lặp cụm gần nhất, rồi popularity", REP),
            metric_row(f"SMLP4Rec plain (epoch {RUN['best_epoch_by_valid']})", T, bold=True),
        ],
        METRIC_W,
    )
    b.figure(
        "week3_fig1_plain_vs_heuristics.png", "SMLP4Rec plain so với hai luật đếm trên tập test."
    )
    b.p(
        f"Thêm epoch gần như không giúp: NDCG@5 trên test {dec(PE[0]['test']['ndcg@5'])} (epoch 1) → "
        f"{dec(PE[-1]['test']['ndcg@5'])} (epoch {PE[-1]['epoch']}), train loss {dec(PE[0]['train_loss'], 1)} → "
        f"{dec(PE[-1]['train_loss'], 1)}. Valid và test gần nhau (NDCG@5 {dec(V['ndcg@5'])} và {dec(T['ndcg@5'])}), "
        "không có dấu hiệu overfit."
    )
    b.h2("3.2 Vì sao plain thấp")
    b.p("Phân tích top 5 của plain trên tập test:")
    b.table(
        [
            ["Chỉ số (test)", "Giá trị"],
            ["Mục tiêu là cụm mới (user chưa đặt)", pct(MX["test"]["new"])],
            ["Mục tiêu là cụm cũ (đã có trong lịch sử)", pct(MX["test"]["old_incl_last"])],
            ["Top 1 = cụm đặt gần nhất", pct(TB["top1_is_last_booking"])],
            ["Cụm đặt gần nhất nằm trong top 5", pct(TB["last_booking_in_top5"])],
            ["Trúng top 5 khi mục tiêu là cụm cũ", pct(TB["old_target_in_top5_of_old_rows"])],
            [
                [("Trúng top 5 khi mục tiêu là cụm mới", True)],
                [(pct(TB["new_target_in_top5_of_new_rows"]), True)],
            ],
            [
                "Popularity toàn cục, trúng top 5 trên dòng cụm mới",
                pct(TB["global_popularity_top5_on_new_rows"]),
            ],
        ],
        [6438, 3200],
    )
    b.p(
        "Plain chỉ nhìn thấy lịch sử, không biết user đang tìm ở điểm đến nào. Nó học tốt việc gợi lại cụm cũ "
        f"(trúng {pct(TB['old_target_in_top5_of_old_rows'], 1)} dòng cụm cũ) nhưng ở nhóm cụm mới, chiếm "
        f"{pct(MX['test']['new'], 0)} mục tiêu, nó còn thấp hơn popularity toàn cục "
        f"({pct(TB['new_target_in_top5_of_new_rows'], 1)} so với {pct(TB['global_popularity_top5_on_new_rows'], 1)}). "
        "Từ đó có hai hướng thử:"
    )
    b.bullets(
        [
            [
                ("Hướng A: ", True),
                ("chỉnh cách chọn top 5, dành một phần chỗ cho cụm mới (Mục 4).", False),
            ],
            [
                ("Hướng B: ", True),
                (
                    "thêm thông tin điểm đến: prior điểm đến, rồi tín hiệu đặt lại tại cùng điểm đến (Mục 5).",
                    False,
                ),
            ],
        ]
    )

    # ------------------------------------------------------------ 4. direction A
    b.h1("4. Hướng A: ép tỉ lệ cụm cũ / cụm mới trong top 5")
    b.p(
        "Giả thuyết: top 5 thiên về cụm cũ, chia lại chỗ cho cụm mới sẽ tốt hơn. Hai luật được áp lên điểm của "
        "chính các trọng số plain (notebook 02, không huấn luyện lại): cố định min(2, số cụm cũ) cụm cũ + phần "
        "còn lại là cụm mới; hoặc trần số cụm cũ theo độ dài lịch sử L (tối đa 1 nếu L < 5, 2 nếu L ≥ 5). Tỉ lệ "
        "trúng tính trên toàn bộ dòng test."
    )
    b.table(
        [
            [
                "Cách (test)",
                "NDCG@5",
                "Recall@5",
                "Cụm cũ trúng",
                "Cụm mới trúng",
                "Số cụm cũ TB trong top 5",
            ],
            [
                [("Plain: 5 điểm cao nhất", True)],
                [(dec(mp["ndcg@5"]), True)],
                [(pct(mp["recall@5"]), True)],
                pct(mp["old_target_hit_of_all_rows"]),
                pct(mp["new_target_hit_of_all_rows"]),
                dec(mp["avg_past_slots_in_top5"], 2),
            ],
            [
                "Cố định 2 cũ + 3 mới",
                dec(mm["ndcg@5"]),
                pct(mm["recall@5"]),
                pct(mm["old_target_hit_of_all_rows"]),
                pct(mm["new_target_hit_of_all_rows"]),
                dec(mm["avg_past_slots_in_top5"], 2),
            ],
            [
                "Tối đa 1 cũ nếu L < 5, 2 nếu L ≥ 5",
                dec(cc["ndcg@5"]),
                pct(cc["recall@5"]),
                pct(cc["old_target_hit_of_all_rows"]),
                pct(cc["new_target_hit_of_all_rows"]),
                dec(cc["avg_past_slots_in_top5"], 2),
            ],
        ],
        [2838, 1100, 1200, 1400, 1400, 1700],
    )
    b.p(
        f"Kết quả: cả hai luật đều làm giảm điểm. Mỗi lần trúng thêm một cụm mới phải đổi bằng khoảng "
        f"{dec(exchange(mm), 1)} (cố định) và {dec(exchange(cc), 1)} (trần theo L) lần trúng cụm cũ. Vấn đề không nằm "
        "ở cách cắt top 5 mà ở chỗ điểm cho cụm mới còn yếu: cần thêm thông tin, không phải chia lại chỗ. "
        "Hướng A bị loại."
    )

    # ------------------------------------------------------------ 5. direction B
    b.h1("5. Hướng B: thêm thông tin điểm đến")
    b.h2("5.1 B1: prior điểm đến")
    pr = LF["prior"]
    b.p(
        "Giả thuyết: điểm đến đang tìm quyết định phần lớn cụm được đặt (tuần 2: phân phối cụm thay đổi mạnh theo "
        "điểm đến). Prior là tần suất cụm tại điểm đến, làm mượt về market của điểm đến:"
    )
    b.p(
        [
            ("p_prior(k | d) = (n_d,k + m · p_market,k) / (n_d + m),  m = ", False),
            (str(pr["smoothing_m"]), False),
        ],
        "caption",
    )
    b.p(
        "Market là market phổ biến nhất của điểm đến trong dữ liệu prior, không dùng hotel_market của chính sự "
        "kiện; điểm đến chưa gặp dùng phân phối toàn cục. Prior chỉ đếm booking trước mục tiêu train cuối cùng "
        f"({num(pr['n_bookings'])} booking, {num(pr['n_destinations'])} điểm đến), nên không booking valid/test nào "
        "góp vào prior của chính nó. Prior được trộn với SMLP4Rec trong không gian log, không huấn luyện lại:"
    )
    b.p("điểm(k) = log p_SMLP4Rec(k | lịch sử) + w_p · log p_prior(k | điểm đến)", "caption")
    b.figure(
        "week3_fig2_prior_weight_sweep.png",
        f"NDCG@5 và Recall@5 theo trọng số w_p (w_p = 0 là plain); w_p = {dec(W_BEST, 1)} chọn trên valid.",
    )
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("SMLP4Rec plain", WARM["plain"]),
            metric_row("Prior điểm đến một mình", WARM["prior"]),
            metric_row("Luật đếm tuần 2 (sameDest rồi prior)", WARM["rule"]),
            metric_row(f"SMLP4Rec + prior (w_p = {dec(W_BEST, 1)})", WARM["fusion"], bold=True),
            metric_row("SMLP4Rec + prior, w_p theo nhóm L", LV["fusion_per_bucket_w"]["all"]),
        ],
        METRIC_W,
    )
    fb = LB[f"{FKEY} minus {K_RULE}"]["warm"]
    b.p(
        f"Prior tăng mạnh nhất ở nhóm cụm mới: NDCG@5 {dec(LV[K_PLAIN]['new_target_rows']['ndcg@5'])} → "
        f"{dec(LV[FKEY]['new_target_rows']['ndcg@5'])}, Recall@5 {pct(LV[K_PLAIN]['new_target_rows']['recall@5'], 1)} → "
        f"{pct(LV[FKEY]['new_target_rows']['recall@5'], 1)}. Chọn w_p riêng cho từng nhóm L "
        f"({', '.join(f'{k}: {wdec(v)}' for k, v in LF['chosen_w_per_bucket_on_valid_recall@5'].items())}) không tốt "
        "hơn một w_p chung, nên giữ w_p chung."
    )
    b.p(
        [
            ("Điểm cần lưu ý: ", True),
            (
                (
                    f"SMLP4Rec + prior chưa thắng luật đếm của tuần 2 ở NDCG@5. So với luật đó: Recall@5 {ci_recall(fb)}, "
                    f"nhưng NDCG@5 {ci_ndcg(fb)} (bootstrap 95%). Luật đếm đặt cụm từng đặt tại cùng điểm đến lên đầu; "
                    "SMLP4Rec + prior không có tín hiệu này. Đây là lý do thêm sameDest ở bước B2."
                ),
                False,
            ),
        ]
    )
    b.p(
        f"User mới (L = 0, {pct(LC['cold_share_of_events'], 1)} sự kiện test) không có lịch sử nên chỉ dùng prior:"
    )
    b.table(
        [
            ["User mới, test", "Số dòng", "NDCG@5", "Recall@5", "NDCG@10", "Recall@10"],
            [
                "Popularity toàn cục",
                num(COLD["global_popularity"]["n"]),
                dec(COLD["global_popularity"]["ndcg@5"]),
                pct(COLD["global_popularity"]["recall@5"]),
                dec(COLD["global_popularity"]["ndcg@10"]),
                pct(COLD["global_popularity"]["recall@10"]),
            ],
            [
                [("Prior điểm đến", True)],
                num(COLD["prior_only"]["n"]),
                [(dec(COLD["prior_only"]["ndcg@5"]), True)],
                [(pct(COLD["prior_only"]["recall@5"]), True)],
                dec(COLD["prior_only"]["ndcg@10"]),
                pct(COLD["prior_only"]["recall@10"]),
            ],
        ]
        + [
            [
                f"  Prior, điểm đến có {lab}",
                num(COLD["prior_only_by_destination_support"][k]["n"]),
                dec(COLD["prior_only_by_destination_support"][k]["ndcg@5"]),
                pct(COLD["prior_only_by_destination_support"][k]["recall@5"]),
                dec(COLD["prior_only_by_destination_support"][k]["ndcg@10"]),
                pct(COLD["prior_only_by_destination_support"][k]["recall@10"]),
            ]
            for k, lab in (
                ("n_dest=0", "0 booking trước đó"),
                ("n_dest 1-19", "1–19 booking"),
                ("n_dest>=20", "≥ 20 booking"),
            )
        ],
        [3638, 1200, 1200, 1200, 1200, 1200],
    )

    b.h2("5.2 B2: sameDest, hybrid ba tín hiệu")
    b.p(
        "Giả thuyết: user hay đặt lại cụm đã ở tại chính điểm đến đang tìm. sameDest(k) là tổng 0,7^tuổi các lần "
        "user đặt cụm k tại đúng điểm đến của lượt tìm kiếm (tuổi 0 = lần gần nhất), chia cho tổng để thành phân "
        f"bố trên 100 cụm. Ví dụ: một cụm được đặt tại điểm đến này ở tuổi 0 và 3 có trọng số 0,7^0 + 0,7^3 = {dec(EX12, 3)}; "
        "cụm đặt ở điểm đến khác có trọng số 0. User chưa từng đặt tại điểm đến này nhận phân bố đều, tức sameDest "
        "không tác động thứ tự."
    )
    b.p(
        "điểm(k) = log q_SMLP4Rec(k) + w_p · log q_prior(k) + w_s · log q_sameDest(k),  "
        "log q(k) = log((1 − α) · p(k) + α / 100)",
        "caption",
    )
    b.p(
        "Ba thành phần đều được trộn với phân bố đều trước khi lấy log (một α chung), nên cùng một thang và trọng số "
        "đọc được như mức tin cậy tương đối. Quy tắc chọn, chỉ dùng valid:"
    )
    g = SEL["grid"]
    b.bullets(
        [
            "Hàm mục tiêu mỗi dòng: Recall@5 + NDCG@5.",
            (
                f"Lưới: α ∈ {{{'; '.join(wdec(a) for a in g['alpha'])}}}, w_p và w_s ∈ [0; 3] ({len(g['w_p'])} mức), "
                f"trọng số SMLP4Rec cố định = {dec(g['w_SMLP4Rec'], 0)}."
            ),
            (
                "Quy tắc một sai số chuẩn [4, 5]: trong các cấu hình nằm trong một SE (gom theo user) của cấu hình tốt "
                "nhất, lấy cấu hình đơn giản nhất: trọng số lớn nhất nhỏ nhất, rồi tổng trọng số nhỏ nhất, rồi α lớn nhất."
            ),
            (
                f"Kết quả: {SEL['n_within_one_se']} cấu hình nằm trong một SE; chọn α = {dec(H_ALPHA, 1)}, "
                f"w_p = {wdec(SEL['one_se_pick'][1])}, w_s = {wdec(SEL['one_se_pick'][2])}."
            ),
        ]
    )
    b.p(
        "Sau đó thử chia trọng số theo nhóm dòng (cohort), đánh giá bằng kiểm định chéo 2 phần theo user trên valid "
        "(3 seed). Quy tắc một SE ở mức lược đồ chọn lược đồ có ít bộ trọng số nhất trong một SE của lược đồ tốt nhất:"
    )
    sch = HY["cohorts"]["schemes"]
    b.table(
        [["Lược đồ chia trọng số", "Số bộ", "Phủ nhỏ nhất", "Lợi so với một bộ (SE)", "Trong 1 SE"]]
        + [
            [
                [(SCHEME_VI[k], k == HY["cohorts"]["chosen_fewest_cells_within_one_se"])],
                str(v["cells"]),
                pct(v["min_coverage"], 1),
                f"{sdec(v['gain_vs_global'], 5)} ({dec(v['se_of_gain'], 5)})",
                "có" if v["within_one_se_of_best"] else "không",
            ]
            for k, v in sch.items()
        ],
        [3638, 900, 1400, 2500, 1200],
    )
    b.p(
        f"Lược đồ được chọn: 2 bộ trọng số. Điểm đến mới với user: w_p = {wdec(HW['0'][1])} (w_s không tác động); "
        f"điểm đến đã đặt: w_p = {wdec(HW['1'][1])}, w_s = {wdec(HW['1'][2])}; α = {dec(H_ALPHA, 1)}."
    )
    b.p("Kết quả trên tập test:", "body")
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("SMLP4Rec plain", WARM["plain"]),
            metric_row("Prior điểm đến một mình", WARM["prior"]),
            metric_row("Luật đếm tuần 2", WARM["rule"]),
            metric_row("SMLP4Rec + prior", HWARM[K_REF]),
            metric_row("Hybrid: + sameDest", WARM["hybrid"], bold=True),
        ],
        METRIC_W,
    )
    b.p(
        f"So với SMLP4Rec + prior, hybrid tăng NDCG@5 {ci_ndcg(HB['warm'])} và Recall@5 {ci_recall(HB['warm'])} "
        "(bootstrap ghép cặp 95%, 1000 lần). Hybrid cũng vượt luật đếm tuần 2 ở cả NDCG@5 và Recall@5 "
        f"({dec(WARM['hybrid']['ndcg@5'])} so với {dec(WARM['rule']['ndcg@5'])}; {pct(WARM['hybrid']['recall@5'])} so "
        f"với {pct(WARM['rule']['recall@5'])}); cặp này chưa có khoảng tin cậy bootstrap."
    )
    kd_r, kd_h = slice_pair("known destination", K_REF), slice_pair("known destination", K_HYBRID)
    nd_r, nd_h = slice_pair("new destination", K_REF), slice_pair("new destination", K_HYBRID)
    b.figure(
        "week3_fig4_slices.png", "NDCG@5 của hybrid và SMLP4Rec + prior theo lát cắt, tập test."
    )
    b.p(
        f"Phần tăng nằm ở các dòng có điểm đến đã đặt ({num(SL['known destination']['rows'])} dòng): NDCG@5 "
        f"{dec(kd_r[1])} → {dec(kd_h[1])}, Recall@5 {pct(kd_r[0], 1)} → {pct(kd_h[0], 1)}. Ở điểm đến mới "
        f"({num(SL['new destination']['rows'])} dòng), sameDest không tác động và điểm gần như giữ nguyên "
        f"(NDCG@5 {dec(nd_r[1])} → {dec(nd_h[1])}). Lợi tăng dần theo độ dài lịch sử."
    )
    b.p("Kiểm tra từng thành phần (test): mỗi thành phần đều có đóng góp.")
    sk, sn = CC["spread share M/P/S, known destination"], CC["spread share M/P/S, new destination"]

    def drop(key: str) -> tuple[str, str]:
        r, n = CC[key].split(" / ")
        return pp(float(r)), sdec(float(n))

    b.table(
        [
            [
                "Thành phần",
                "Tỉ trọng độ phân tán, điểm đến đã đặt",
                "Tỉ trọng, điểm đến mới",
                "Bỏ thành phần: Recall@5",
                "Bỏ thành phần: NDCG@5",
            ],
            ["SMLP4Rec", pct(sk[0], 1), pct(sn[0], 1), *drop("drop SMLP4Rec")],
            ["Prior điểm đến", pct(sk[1], 1), pct(sn[1], 1), *drop("drop prior")],
            ["sameDest", pct(sk[2], 1), pct(sn[2], 1), *drop("drop sameDest")],
        ],
        [1838, 2200, 1800, 1900, 1900],
    )
    b.p(
        "Tỉ trọng độ phân tán: độ lệch chuẩn theo cụm của từng số hạng đã nhân trọng số, chia cho tổng ba số hạng "
        "(số hạng nào quyết định thứ tự). Prior chi phối, nhưng bỏ SMLP4Rec hoặc sameDest đều làm giảm điểm."
    )

    b.h2("5.3 B3: tách trọng số theo thói quen đặt lại")
    b.p(
        "Giả thuyết: user hay đặt lại cụm cũ cần trọng số khác user hay đặt cụm mới. Công thức B2 giữ nguyên, chỉ "
        "đổi cách chia nhóm. Đặc trưng old_share = tỉ lệ booking trong lịch sử (vị trí 2..L) là cụm đã đặt trước "
        "đó, chỉ tính từ lịch sử, không dùng mục tiêu. User có L = 1 không có bằng chứng và là nhóm riêng."
    )
    b.bullets(
        [
            (
                f"Ngưỡng chọn trên train: thiên cũ nếu L ≥ 2 và old_share ≥ {dec(B_CUT, 2)}. Ở ngưỡng này tỉ lệ mục tiêu "
                f"là cụm cũ là {pct(STUMP['tgt_old_if_at_or_above'], 1)} (thiên cũ) so với "
                f"{pct(STUMP['tgt_old_if_below'], 1)} (thiên mới)."
            ),
            "Mỗi bộ trọng số phải phủ ít nhất 10% dòng có lịch sử, không tạo nhóm nhỏ.",
            "Chọn lược đồ bằng kiểm định chéo 2 phần theo user trên valid và quy tắc một SE như B2.",
        ]
    )
    bsch = BS["schemes"]
    b.table(
        [["Lược đồ", "Số bộ", "Phủ nhỏ nhất", "Đủ điều kiện", "Lợi so với B2 (SE)"]]
        + [
            [
                [(SCHEME_VI[k], k == BS["chosen_scheme_fewest_cells_within_one_se"])],
                str(v["cells"]),
                pct(v["min_cell_share"], 1),
                "có" if v["eligible"] else "không",
                split_gain(v["gain_vs_reference"]),
            ]
            for k, v in bsch.items()
        ],
        [3838, 900, 1400, 1300, 2200],
    )
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("Hybrid B2 (2 bộ)", BW["reference (notebook 04)"], bold=True),
            metric_row(
                f"Tốt nhất có thói quen ({len(B_CELLS[B_BEST])} bộ)", BW["best behaviour scheme"]
            ),
        ],
        METRIC_W,
    )
    b.p(
        f"Kết quả: không cải thiện. Lược đồ tốt nhất có thói quen so với B2 trên test: NDCG@5 {ci_ndcg(BB['warm'])}, "
        f"Recall@5 {ci_recall(BB['warm'])} (bootstrap 95%); quy tắc một SE vẫn chọn 2 bộ của B2. Thói quen đặt lại "
        "đã nằm sẵn trong sameDest, nên không cần trọng số riêng."
    )

    # ------------------------------------------------------------ 6. summary of results
    b.h1("6. Tổng hợp kết quả")
    b.h2("6.1 Mọi sự kiện test")
    b.p(
        "Bảng dưới gộp user có lịch sử và user mới (mọi sự kiện test). User mới dùng prior cho mọi cách có prior; "
        "với plain, user mới dùng popularity toàn cục."
    )
    b.table(
        [
            [
                "Cách (mọi sự kiện test)",
                "NDCG@5",
                "Recall@5",
                "NDCG@10",
                "Recall@10",
                "NDCG@20",
                "Recall@20",
            ]
        ]
        + [
            metric_row("SMLP4Rec plain + popularity cho user mới", ALL["plain"]),
            metric_row("Prior cho mọi user", ALL["prior"]),
            metric_row("Luật đếm tuần 2", ALL["rule"]),
            metric_row("SMLP4Rec + prior", ALL["fusion"]),
            metric_row("Hybrid: + sameDest", ALL["hybrid"], bold=True),
        ],
        METRIC_W,
    )
    b.p(
        f"Trên mọi sự kiện test ({num(ALL['hybrid']['n'])} dòng), hybrid so với SMLP4Rec + prior: NDCG@5 "
        f"{ci_ndcg(HB['all_events'])}, Recall@5 {ci_recall(HB['all_events'])}."
    )

    # 6.2 seen vs unseen users (notebook 06)
    PR, SR = SU["presence"], SU["results"]["test"]
    PL, HYK = "plain SMLP4Rec", "hybrid (SMLP4Rec + prior + sameDest)"
    b.h2("6.2 Người dùng đã thấy và chưa thấy trong train")
    b.p(
        "Tập valid và test được chia thành hai tập con không giao nhau theo việc người dùng có mặt trong tập train "
        "hay không. Đã thấy: có ít nhất một dòng trong tập train của RecBole. Chưa thấy (cold-start): không có dòng "
        "nào trong train, gồm hai loại sự kiện: người đã có lịch sử (L ≥ 1) nhưng toàn bộ booking nằm sau cửa sổ "
        "train, và lần đặt đầu tiên của người dùng (L = 0)."
    )
    b.figure(
        "week3_fig5_seen_unseen_share.png",
        "Tỉ lệ người dùng và sự kiện đã thấy và chưa thấy trong train, tập valid và test.",
    )
    b.p(
        f"Trên test, {pct(PR['test']['unseen_users_pct'] / 100, 1)} người dùng "
        f"({num(PR['test']['unseen_users'])} / {num(PR['test']['users'])}) chưa có trong train, tương ứng "
        f"{pct(PR['test']['unseen_events_pct'] / 100, 1)} sự kiện ({num(PR['test']['unseen_events'])} / "
        f"{num(PR['test']['events'])}); trên valid là {pct(PR['valid']['unseen_users_pct'] / 100, 1)} người dùng và "
        f"{pct(PR['valid']['unseen_events_pct'] / 100, 1)} sự kiện. Trong số người dùng chưa thấy ở test, "
        f"{num(PR['test']['unseen_users_with_history_(L>=1)'])} người đã có lịch sử và "
        f"{num(PR['test']['unseen_users_first_booking_only_(L=0)'])} người chỉ có lần đặt đầu. Độ dài lịch sử trung "
        f"bình trên test: {dec(PR['test']['mean_history_len_seen'], 1)} (đã thấy) so với "
        f"{dec(PR['test']['mean_history_len_unseen_L>=1'], 1)} (chưa thấy, L ≥ 1)."
    )
    b.figure(
        "week3_fig6_seen_unseen_results.png",
        "NDCG@5 và Recall@5 của SMLP4Rec plain và hybrid trên người dùng đã thấy và chưa thấy, tập test.",
    )
    b.p(
        f"Hybrid cao hơn plain ở mọi tập con: NDCG@5 {dec(SR['seen users'][PL]['ndcg@5'], 3)} → "
        f"{dec(SR['seen users'][HYK]['ndcg@5'], 3)} ở người đã thấy, {dec(SR['unseen users'][PL]['ndcg@5'], 3)} → "
        f"{dec(SR['unseen users'][HYK]['ndcg@5'], 3)} ở người chưa thấy; Recall@5 "
        f"{pct(SR['seen users'][PL]['recall@5'], 1)} → {pct(SR['seen users'][HYK]['recall@5'], 1)} và "
        f"{pct(SR['unseen users'][PL]['recall@5'], 1)} → {pct(SR['unseen users'][HYK]['recall@5'], 1)}. "
        f"Chênh lệch NDCG@5 có khoảng tin cậy bootstrap 95% hẹp ở mọi tập con "
        f"(chưa thấy: {ci_ndcg(SU['bootstrap_test_95ci_hybrid_minus_plain']['unseen users'])})."
    )
    b.p(
        [
            ("Lưu ý về plain ở L = 0. ", True),
            (
                "SMLP4Rec chỉ đọc chuỗi booking nên không chấm được lần đặt đầu tiên; ở cột này plain dùng popularity "
                f"toàn cục (NDCG@5 {dec(SR['unseen, L=0 (first booking)'][PL]['ndcg@5'], 3)}, Recall@5 "
                f"{pct(SR['unseen, L=0 (first booking)'][PL]['recall@5'], 1)}), nên đây là điểm của popularity, không "
                "phải của mô hình. Với người chưa thấy nhưng có lịch sử (L ≥ 1), plain đạt NDCG@5 "
                f"{dec(SR['unseen, L>=1'][PL]['ndcg@5'], 3)}, gần với người đã thấy "
                f"({dec(SR['seen users'][PL]['ndcg@5'], 3)}): mô hình không có tham số theo từng người dùng, nên việc "
                "người đó có mặt trong train không quyết định; độ dài lịch sử mới quyết định."
            , False),
        ]
    )

    # 6.3 two basic baselines (notebook 07)
    KM, LM = BL["models"][BL_KNN], BL["models"][BL_LR]
    SPL = BL["split"]
    all_sz = SPL["slice_sizes_vs_reference"]["test"]["all events"]
    CT = BL["verification"]["controls_test_warm"]
    ver_mismatch = (
        sum(v["rank_mismatches"] for v in BV["logreg"].values() if isinstance(v, dict))
        + BV["itemknn"]["test_sample_rank_mismatches"]
        + len(BV["metrics_recomputed_from_saved_ranks"]["mismatches"])
    )

    def bl_row(name, d):
        return [name] + [
            dec(d[sl]["ndcg@5"], 3) if m == "n" else pct(d[sl]["recall@5"], 1)
            for sl, m in (("all events", "n"), ("all events", "r"))
        ] + [dec(d[sl]["ndcg@5"], 3) for sl in ("seen users", "unseen, L>=1", "unseen, L=0 (first booking)")]

    k_grid = ", ".join(str(k) for k in sorted(int(k) for k in KM["K_grid_valid"]))
    c_grid = "; ".join(
        dec(c, 1).rstrip("0").rstrip(",") if c % 1 else dec(c, 0) for c in sorted(float(k) for k in LM["C_grid_valid"])
    )
    b.h2("6.3 Hai baseline cơ bản trên cùng phép chia")
    b.p(
        "Để kiểm tra điểm cao của hybrid đến từ dữ liệu chứ không từ rò rỉ hay lỗi chia tập, chạy thêm hai mô hình "
        "có sẵn trong thư viện, không dùng code của hybrid, mỗi mô hình bị chặn đúng một nguồn tín hiệu "
        "(notebook 07_basic_baselines)."
    )
    b.table(
        [
            ["Baseline", "Thư viện", "Thấy", "Không thấy", "Chọn trên valid"],
            [
                "ItemKNN (cosine, Sarwar 2001)",
                "implicit, CosineRecommender",
                f"{CFG['MAX_ITEM_LIST_LENGTH']} booking gần nhất của người dùng",
                "điểm đến, mọi ngữ cảnh",
                f"K = {KM['K_pick']} trong {{{k_grid}}}",
            ],
            [
                "Hồi quy logistic đa lớp",
                "scikit-learn, saga",
                "điểm đến + ngữ cảnh lượt tìm (one-hot)",
                "lịch sử, mọi trường của khách sạn đã đặt, thành phố và vùng của khách",
                f"C = {dec(LM['C_pick'], 0)} trong {{{c_grid}}}; {LM['epochs']} epoch",
            ],
        ],
        [2100, 1700, 2100, 2200, 1538],
    )
    b.p(
        f"Phép chia giống bản plain: chia theo thời gian 80/10/10 trên {num(sum(SPL['counts'][k] for k in ('train', 'valid', 'test')))} "
        f"mục tiêu “đặt tiếp theo”, lần đặt đầu của mỗi người dùng (L = 0) chấm riêng trong cùng cửa sổ thời gian. "
        f"Tập test có {num(all_sz['here'])} sự kiện, bản plain có {num(all_sz['reference'])} (chênh {all_sz['diff']}); "
        "số dòng ở từng tập con (đã thấy, chưa thấy) lệch tối đa 4 dòng. Cả hai mô hình huấn luyện trên booking trước "
        "mốc cắt của prior; K và C chọn trên valid theo Recall@5 + NDCG@5, test chấm một lần. ItemKNN không có epoch: "
        "độ tương tự cosine tính một lần; hồi quy logistic chạy đúng "
        f"{LM['epochs']} epoch, không dừng sớm. Dòng không có lịch sử (L = 0) dùng popularity toàn cục ở ItemKNN."
    )
    b.table(
        [["Cách (test)", "NDCG@5 mọi sự kiện", "Recall@5 mọi sự kiện", "NDCG@5 đã thấy", "NDCG@5 chưa thấy, L ≥ 1", "NDCG@5 L = 0*"]]
        + [
            bl_row("Popularity toàn cục (không học)", BL_R["pop"]),
            bl_row(f"ItemKNN (K = {KM['K_pick']}): chỉ lịch sử", BL_R["knn"]),
            bl_row("SMLP4Rec plain: chỉ lịch sử", {k: v[PLN_K] for k, v in SU["results"]["test"].items()}),
            bl_row(f"Hồi quy logistic (C = {dec(LM['C_pick'], 0)}): điểm đến + ngữ cảnh", BL_R["lr"]),
            [
                [(c, True)]
                for c in bl_row("Hybrid: lịch sử + prior + sameDest", {k: v[HYB_K] for k, v in SU["results"]["test"].items()})
            ],
        ],
        [3000, 1300, 1300, 1300, 1438, 1300],
    )
    b.p(
        "* L = 0 là lần đặt đầu tiên: ItemKNN và plain không có đầu vào nên dùng popularity toàn cục (cùng số); hồi quy "
        "logistic và hybrid dùng điểm đến. Hàng plain và hybrid lấy từ notebook 06 (mục 6.2).",
        "ref",
    )
    b.figure(
        "week3_fig7_basic_baselines.png",
        "NDCG@5 và Recall@5 của hai baseline, SMLP4Rec plain và hybrid trên mọi sự kiện test.",
    )
    d_dest = BL_R["lr"]["all events"]["ndcg@5"] - BL_R["pop"]["all events"]["ndcg@5"]
    d_hist = SU["results"]["test"]["all events"][HYB_K]["ndcg@5"] - BL_R["lr"]["all events"]["ndcg@5"]
    l0 = "unseen, L=0 (first booking)"
    b.p(
        [
            ("Đọc kết quả. ", True),
            (
                f"ItemKNN, chỉ đọc lịch sử, nằm sát SMLP4Rec plain (NDCG@5 {dec(BL_R['knn']['all events']['ndcg@5'], 3)} so "
                f"với {dec(SU['results']['test']['all events'][PLN_K]['ndcg@5'], 3)}), nên bản plain không bị thổi phồng. "
                f"Hồi quy logistic, chỉ có điểm đến và ngữ cảnh, đạt {dec(BL_R['lr']['all events']['ndcg@5'], 3)}, cao hơn "
                f"popularity {dec(d_dest, 3)} điểm NDCG@5; hybrid cộng thêm {dec(d_hist, 3)} điểm nhờ lịch sử cùng điểm "
                f"đến và SMLP4Rec. Ở lần đặt đầu (L = 0), hồi quy logistic ({dec(BL_R['lr'][l0]['ndcg@5'], 3)}) chỉ kém hybrid "
                f"({dec(SU['results']['test'][l0][HYB_K]['ndcg@5'], 3)}) {dec(SU['results']['test'][l0][HYB_K]['ndcg@5'] - BL_R['lr'][l0]['ndcg@5'], 3)} điểm. "
                "Phần lớn điểm của hybrid giải thích được bằng điểm đến của lượt tìm, thông tin có sẵn trước khi đặt.",
                False,
            ),
        ]
    )
    b.table(
        [["Đối chứng (test, user có lịch sử, NDCG@5)", "Thật", "Xáo trộn"]]
        + [
            ["Hồi quy logistic: xáo trộn đặc trưng lượt tìm", dec(CT["LR real"]["ndcg@5"], 3), dec(CT["LR shuffled query features"]["ndcg@5"], 3)],
            ["ItemKNN: xáo trộn lịch sử giữa các dòng", dec(CT["ItemKNN real"]["ndcg@5"], 3), dec(CT["ItemKNN shuffled histories"]["ndcg@5"], 3)],
        ],
        [6038, 1800, 1800],
    )
    b.p(
        f"Mốc popularity toàn cục trên cùng tập là {dec(CT['global popularity']['ndcg@5'], 3)}; xáo trộn đưa cả hai mô hình "
        "về mức đó, nên tín hiệu của chúng là thật."
    )
    b.p(
        [
            ("Kiểm tra. ", True),
            (
                f"Trong notebook: không dòng valid hoặc test nào nằm trong dữ liệu huấn luyện "
                f"({num(BL['verification']['eval_rows_in_training_data'])} dòng); huấn luyện kết thúc "
                f"{abs(BL['verification']['training_max_ts_minus_first_valid_target_ts'])} giây trước mục tiêu valid đầu tiên; "
                "điểm ItemKNN khớp hàm recommend() của thư viện; Recall@5 và NDCG@5 tính lại bằng vòng lặp Python thuần "
                "khớp; không có hạng hòa ở mục tiêu. Kiểm tra độc lập (scripts/verify_basic_baselines.py, không dùng code "
                "của notebook): dựng lại phép chia, lịch sử, độ tương tự cosine và đặc trưng, chấm lại hai mô hình và "
                f"so với kết quả đã lưu, được {ver_mismatch} sai khác về hạng; hồi quy logistic không dùng "
                "trường nào của khách sạn đã đặt, thành phố hay vùng của khách; không có số kết quả viết cứng trong code notebook.",
                False,
            ),
        ]
    )
    b.p(
        [
            ("Giới hạn. ", True),
            (
                f"Một seed; C = {dec(LM['C_pick'], 0)} là giá trị lớn nhất trong lưới nên có thể còn tăng nhẹ; hồi quy "
                f"logistic chỉ chạy {LM['epochs']} epoch; lưới K và C nhỏ. Đây là kiểm tra tính hợp lý, không phải bảng so "
                "sánh mô hình cuối cùng.",
                False,
            ),
        ]
    )

    # 6.4 query token (notebook 01c)
    QTT, QTP, QCFG, QCOV, QTIME = QT["test"], QT["plain_rescored_test"], QT["config"], QT["query_coverage"], QT["timing_seconds"]
    QD = QT["diagnostics_test"]
    QBOOT = QT["bootstrap_query_minus_plain_test"]["all warm rows"]
    q_sim = WARM["fusion"]["ndcg@5"] - QTT["ndcg@5"]
    q_gap = WARM["hybrid"]["ndcg@5"] - QTT["ndcg@5"]
    b.h2("6.4 Query token: đưa lượt tìm kiếm vào trong mô hình")
    b.p(
        "Bản plain chỉ đọc chuỗi cụm đã đặt; mọi trường của lượt tìm kiếm hiện tại (điểm đến, ngày, nhóm khách, gói, kênh) "
        "bị bỏ. Hybrid ở mục 5 bù bằng cách cộng prior và sameDest bên ngoài mô hình. Notebook 01c đưa lượt tìm vào trong "
        "mô hình: thêm một vị trí cuối chuỗi (query token) gồm vector [MASK] học được cộng một embedding cho mỗi trường "
        "của lượt tìm; trạng thái ẩn tại vị trí đó chấm điểm 100 cụm. Phần lịch sử giữ nguyên (20 cụm, đệm bên phải)."
    )
    b.table(
        [
            ["Nhóm", "14 trường của lượt tìm (cộng thành một vector query)"],
            ["Điểm đến", "mã điểm đến (embedding riêng nếu có ít nhất 5 lần ở các mục tiêu train), loại điểm đến"],
            ["Lịch", "tháng check-in, số ngày từ lúc tìm đến check-in (nhóm), số đêm (nhóm)"],
            ["Nhóm khách", "số người lớn, số trẻ em, số phòng"],
            ["Gói và kênh", "gói, thiết bị di động, kênh, site, châu lục điểm bán, quốc gia của khách"],
        ],
        [2200, 7438],
    )
    b.p(
        f"Cùng cấu hình và phép chia với bản plain ({QCFG['epochs']} epoch, hidden {QCFG['hidden_size']}, {QCFG['n_layers']} lớp, "
        f"seed {QCFG['seed']}); chỉ chấm user có lịch sử (L ≥ 1), chưa chấm user mới. Số tham số {num(QT['n_parameters'])} "
        f"(bản chỉ lịch sử: {num(QT['n_parameters_history_only'])}), trong đó {num(QT['n_query_parameters'])} là các bảng embedding "
        f"query. Vocab điểm đến và quốc gia chỉ đếm trên mục tiêu train (điểm đến: ít nhất 5 lần, {num(QCOV['dest']['vocab_size'])} "
        f"điểm đến); giá trị ngoài vocab dùng chỉ số 0 ({pct(QCOV['dest']['oov_test_targets'], 1)} dòng test về điểm đến). "
        "Không dùng mã khách sạn (hotel_*), khoảng cách, thành phố và vùng của khách, đặc trưng ẩn của destinations.csv. "
        f"Huấn luyện và chấm {dec(QTIME['train_and_eval'] / 60, 1)} phút trên CPU. Bản plain được chấm lại trên đúng các dòng "
        "này và khớp kết quả notebook 01."
    )
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("SMLP4Rec plain: chỉ lịch sử", QTP),
            metric_row("SMLP4Rec + query token", QTT, bold=True),
            metric_row(f"SMLP4Rec + prior (w_p = {dec(W_BEST, 1)}, mục 5.1)", WARM["fusion"]),
            metric_row("Hybrid: + sameDest (mục 5.2)", WARM["hybrid"]),
        ],
        METRIC_W,
    )
    b.p(
        [
            ("Đọc kết quả. ", True),
            (
                f"Chỉ thêm lượt tìm vào trong mô hình, NDCG@5 tăng {dec(QTP['ndcg@5'], 3)} → {dec(QTT['ndcg@5'], 3)} "
                f"và Recall@5 {pct(QTP['recall@5'])} → {pct(QTT['recall@5'])}; so với plain: NDCG@5 {ci_ndcg(QBOOT)}, "
                f"Recall@5 {ci_recall(QBOOT)} (bootstrap 95%), tăng ở user đã thấy, chưa thấy và mọi độ dài lịch sử. "
                f"Điểm ngang SMLP4Rec + prior (chênh {dec(abs(q_sim), 3)} NDCG@5) mà không cần phần vá bên ngoài; hybrid vẫn "
                f"cao hơn {dec(q_gap, 3)} NDCG@5.",
                False,
            ),
        ]
    )
    b.table(
        [["Chẩn đoán trên cùng checkpoint (test)", "NDCG@5", "Recall@5"]]
        + [
            [n, dec(d["ndcg@5"]), pct(d["recall@5"])]
            for n, d in (
                ("Thật (có lịch sử, có query)", QD["real"]),
                ("Xáo query giữa các dòng", QD["query shuffled"]),
                ("Che lịch sử, giữ query", QD["history masked"]),
                ("Vừa xáo vừa che", QD["both (shuffled + masked)"]),
            )
        ],
        [6038, 1800, 1800],
    )
    b.p(
        f"Xáo query làm điểm tụt về mức popularity ({dec(POP['ndcg@5'], 3)}), nên mô hình dựa vào query. Che lịch sử, chỉ còn "
        f"query, vẫn đạt NDCG@5 {dec(QD['history masked']['ndcg@5'], 3)}, cao hơn cả plain có đủ lịch sử "
        f"({dec(QTP['ndcg@5'], 3)}): thông tin của lượt tìm mạnh hơn lịch sử. Lịch sử cộng thêm "
        f"{dec(QTT['ndcg@5'] - QD['history masked']['ndcg@5'], 3)} NDCG@5 trên nền query; ở mục 5.1, prior cộng lịch sử hơn prior "
        f"một mình {dec(WARM['fusion']['ndcg@5'] - WARM['prior']['ndcg@5'], 3)}: hai cách đo cho cùng bậc độ lớn."
    )

    # 6.5 prior check on the query-token model (notebook 01c, section 10)
    PF = QT["prior_fusion_check"]
    PF_W = PF["chosen_w_on_valid_recall@5"]
    PF_ALONE, PF_FUSED = "query token alone (w=0)", f"query token + prior (w={PF_W})"
    PF_T, PF_B, PF_S = PF["test"], PF["bootstrap_fused_minus_query_token_test"], PF["test_slices_recall@5_ndcg@5"]
    b.h2("6.5 Kiểm chứng: cộng prior vào query token có tăng không")
    b.p(
        "Điểm của query token gần như trùng SMLP4Rec + prior. Giả thuyết: query token chứa điểm đến nên mô hình đã tự học "
        "p(cụm | điểm đến) mà prior cung cấp bằng cách đếm; vậy cộng thêm prior sẽ không tăng đáng kể. Điểm giống nhau chỉ là "
        "bằng chứng gián tiếp, nên kiểm tra trực tiếp: điểm = log p_query token + w · log p_prior(cụm | điểm đến), prior dựng "
        "như mục 5.1 (làm trơn về market, cắt trước mục tiêu train cuối), w chọn trên valid theo Recall@5 (hòa thì lấy w nhỏ), "
        f"test chấm một lần. w = 0 phải tái tạo mô hình một mình (đã kiểm tra). w chọn là {dec(PF_W, 2)}, nhỏ hơn nhiều so với "
        f"{dec(W_BEST, 1)} của bản plain; trên valid, w càng lớn điểm càng giảm."
    )
    b.table(
        [METRIC_HEAD]
        + [
            metric_row("Query token một mình", PF_T[PF_ALONE]),
            metric_row(f"Query token + prior (w = {dec(PF_W, 2)})", PF_T[PF_FUSED], bold=True),
            metric_row("Prior điểm đến một mình", PF_T["destination prior only"]),
        ],
        METRIC_W,
    )
    qs_keys = [
        ("Điểm đến phổ biến (từ 200 booking trong dữ liệu prior)", "destination 200+"),
        ("20–199 booking", "destination 20-199"),
        ("Hiếm (1–19 booking)", "destination 1-19 bookings in prior data"),
        ("Chưa từng thấy trong prior", "destination unseen in prior data"),
        ("Ngoài vocab của query token", "destination out of query-token vocab"),
        ("Trong vocab của query token", "destination in query-token vocab"),
    ]
    b.table(
        [["Nhóm điểm đến (test)", "Số dòng", "NDCG@5 một mình", "NDCG@5 + prior", "NDCG@5 prior", "Recall@5 một mình", "Recall@5 + prior"]]
        + [
            [
                lab,
                num(PF_S[f"{k} | query token alone"]["n"]),
                dec(PF_S[f"{k} | query token alone"]["ndcg@5"], 3),
                dec(PF_S[f"{k} | query token + prior"]["ndcg@5"], 3),
                dec(PF_S[f"{k} | prior only"]["ndcg@5"], 3),
                pct(PF_S[f"{k} | query token alone"]["recall@5"]),
                pct(PF_S[f"{k} | query token + prior"]["recall@5"]),
            ]
            for lab, k in qs_keys
        ],
        [2638, 1000, 1200, 1200, 1200, 1200, 1200],
    )
    b.p(
        [
            ("Đọc kết quả. ", True),
            (
                f"Cộng prior chỉ tăng nhẹ: NDCG@5 {ci_ndcg(PF_B)}, Recall@5 {ci_recall(PF_B)} (bootstrap 95%), nhỏ hơn nhiều so "
                f"với sameDest (NDCG@5 {ci_ndcg(HB['warm'])} trên SMLP4Rec + prior). Với điểm đến phổ biến "
                f"({num(PF_S['destination 200+ | query token alone']['n'])} dòng), prior không thêm gì "
                f"(NDCG@5 {dec(PF_S['destination 200+ | query token alone']['ndcg@5'], 3)} → "
                f"{dec(PF_S['destination 200+ | query token + prior']['ndcg@5'], 3)}); phần tăng nằm ở điểm đến hiếm hoặc ngoài vocab, "
                "nơi prior có backoff theo market còn mô hình thiếu dữ liệu. Giả thuyết đúng cho phần lớn dữ liệu: query token đã "
                f"bao hàm prior. Khoảng cách {dec(q_gap, 3)} NDCG@5 còn lại với hybrid thuộc về sameDest, thông tin về lịch sử của "
                "user tại đúng điểm đến, mà lịch sử của query token chưa có (mỗi booking cũ mới chỉ là mã cụm).",
                False,
            ),
        ]
    )
    b.p(
        [
            ("Giới hạn. ", True),
            (
                "Một seed, một checkpoint, 3 epoch (loss còn giảm ở epoch cuối); trọng số w toàn cục, chưa tách theo nhóm điểm đến; "
                "chưa chấm user mới (L = 0) và chưa tách riêng đóng góp của từng trường ngoài điểm đến.",
                False,
            ),
        ]
    )

    # ------------------------------------------------------------ 7. conclusion
    b.h1("7. Kết luận và bước tiếp theo")
    b.h2("7.1 Kết luận")
    b.bullets(
        [
            "SMLP4Rec chạy được trên Expedia với hai chỉnh sửa nhỏ; bản plain có tín hiệu nhưng chủ yếu gợi lại cụm cũ.",
            (
                "Điểm đến là tín hiệu chủ đạo. Chia lại chỗ trong top 5 không giúp; thêm prior điểm đến và sameDest "
                f"đưa NDCG@5 từ {dec(WARM['plain']['ndcg@5'])} lên {dec(WARM['hybrid']['ndcg@5'])} và Recall@5 từ "
                f"{pct(WARM['plain']['recall@5'])} lên {pct(WARM['hybrid']['recall@5'])}."
            ),
            "Hybrid vượt luật đếm tuần 2, mốc mà SMLP4Rec + prior chưa vượt được ở NDCG@5.",
            (
                "Hybrid cao hơn plain cả với người dùng đã thấy lẫn chưa thấy trong train; plain không chấm được "
                "lần đặt đầu (L = 0), chỉ có popularity thay thế."
            ),
            "Tách trọng số theo thói quen không thêm gì; giữ 2 bộ trọng số (điểm đến mới / đã đặt).",
            (
                "Đưa lượt tìm vào trong mô hình (query token, mục 6.4) đưa NDCG@5 từ "
                f"{dec(QT['plain_rescored_test']['ndcg@5'])} lên {dec(QT['test']['ndcg@5'])}, ngang SMLP4Rec + prior; "
                f"cộng thêm prior chỉ tăng {sdec(QT['prior_fusion_check']['bootstrap_fused_minus_query_token_test']['ndcg@5']['diff'])} "
                "NDCG@5 (mục 6.5): query token đã bao hàm prior, phần còn thiếu so với hybrid là sameDest."
            ),
            (
                "Hai baseline cơ bản (ItemKNN chỉ dùng lịch sử, hồi quy logistic chỉ dùng điểm đến) cho thấy điểm cao "
                "của hybrid chủ yếu do điểm đến của lượt tìm; không thấy dấu hiệu rò rỉ hay lỗi chia tập."
            ),
        ]
    )
    b.p(
        [
            ("Quyết định: ", True),
            (
                "SMLP4Rec + prior điểm đến + sameDest, 2 bộ trọng số chọn trên valid; user mới chỉ dùng prior.",
                False,
            ),
        ]
    )
    b.h2("7.2 Bước tiếp theo")
    b.bullets(
        [
            "Chạy nhiều seed và báo cáo trung bình ± độ lệch chuẩn.",
            "Đưa ngữ cảnh theo từng booking cũ (điểm đến, nhóm khách, mùa) vào chuỗi lịch sử, để sameDest có thể nằm trong mô hình.",
            "Chấm query token cho user mới (L = 0) và thêm các lần đặt đầu vào tập huấn luyện nếu cần.",
            "Chạy lại trên phép chia theo sự kiện, gồm cả user mới, trong cùng một bảng.",
            "Thêm các mô hình so sánh còn lại: MF, item2vec, AdaGIN, LightGBM (ItemKNN và hồi quy logistic đã chạy ở mục 6.3), cùng phép chia và tập ứng viên.",
            "Bổ sung chỉ số ngoài độ chính xác: độ đa dạng trong danh sách, thiên lệch popularity.",
        ]
    )

    # ------------------------------------------------------------ 8. limitations
    b.h1("8. Giới hạn")
    b.bullets(
        [
            "Hotel cluster là đại diện ẩn danh cho hạng phòng hoặc gói; kết luận kinh doanh kế thừa giới hạn này.",
            f"Một seed ({CFG['seed']}), {CFG['epochs']} epoch, cấu hình nhẹ trên CPU; chưa tinh chỉnh siêu tham số của SMLP4Rec.",
            "Prior tĩnh, đếm đến cuối train; không cập nhật trong cửa sổ valid/test.",
            "Query token: một seed, 3 epoch, loss còn giảm; chưa chấm user mới; lịch sử chưa mang ngữ cảnh từng booking.",
            "Hai baseline cơ bản: một seed, lưới K và C nhỏ, hồi quy logistic chỉ 3 epoch và C chọn ở biên lưới; dùng để kiểm tra tính hợp lý, chưa phải bảng so sánh cuối.",
            "Phép chia 80/10/10 của RecBole tính trên mục tiêu có lịch sử; user mới được chấm riêng cùng cửa sổ thời gian.",
            "Mọi kết quả là offline trên dữ liệu công khai; không có dữ liệu Vinpearl.",
            "Điều khoản dữ liệu Expedia chỉ cho phép dùng cho nghiên cứu; không phân phối lại dữ liệu thô.",
        ]
    )

    # ------------------------------------------------------------ appendix
    b.h1("Phụ lục A. Trọng số cuối cùng")
    b.table(
        [["Bộ trọng số B2 (α = " + dec(H_ALPHA, 1) + ")", "w_p", "w_s"]]
        + [
            [CELL_VI[lab], wdec(w[0]), wdec(w[1]) if "already" in lab else "— (không tác động)"]
            for lab, w in B_CELLS[BS["chosen_scheme_fewest_cells_within_one_se"]].items()
        ],
        [5638, 2000, 2000],
    )
    b.table(
        [[f"Bộ trọng số B3 ({len(B_CELLS[B_BEST])} bộ, α = {dec(H_ALPHA, 1)})", "w_p", "w_s"]]
        + [
            [
                CELL_VI[lab],
                wdec(w[0]),
                "— (không tác động)" if lab.startswith("new") else wdec(w[1]),
            ]
            for lab, w in B_CELLS[B_BEST].items()
        ],
        [5638, 2000, 2000],
    )
    b.p(
        "w_s “không tác động”: điểm đến chưa từng đặt thì sameDest là phân bố đều, không đổi thứ tự xếp hạng.",
    )
    b.h1("Phụ lục B. Nguồn số liệu")
    b.table(
        [
            ["Mục", "Notebook", "File kết quả"],
            ["2, 3", "01_train_test_smlp4rec", "smlprec_expedia_run.json"],
            [
                "4",
                "02_top5_rerank_rules",
                "smlprec_expedia_mixed_run.json, smlprec_expedia_dynamic_cap_run.json",
            ],
            ["5.1", "03_destination_prior", "smlprec_expedia_late_fusion.json"],
            ["5.2, 6.1", "04_samedest_hybrid", "smlprec_expedia_hybrid_samedest.json"],
            ["5.3", "05_behaviour_split", "smlprec_expedia_hybrid_behaviour_split.json"],
            ["6.2", "06_seen_vs_unseen_users", "smlprec_expedia_seen_unseen_users.json"],
            [
                "6.3",
                "07_basic_baselines",
                "results/week4_rebuild/basic_baselines.json, basic_baselines_verify.json",
            ],
            ["6.4, 6.5", "01c_train_test_smlp4rec_query", "results/week4_rebuild/smlprec_query_run.json"],
        ],
        [1200, 3000, 5438],
    )
    b.p(
        "Notebook nằm trong notebooks/hospitality/smlp4rec/, file kết quả trong results/week3_implementation/. "
        "Hình được vẽ bởi scripts/week3_report_figures.py; báo cáo được dựng bởi scripts/week3_implementation_report.py. "
        "Kiểm tra độc lập của mục 6.3: scripts/verify_basic_baselines.py. Notebook 01c đọc dữ liệu từ scripts/expedia_query_to_recbole.py; mục 6.4 và 6.5 dùng file kết quả trong results/week4_rebuild/.",
    )

    # ------------------------------------------------------------ references
    b.h1("Tài liệu tham khảo")
    for ref in [
        "[1] J. Gao, X. Zhao, M. Li, M. Zhao, R. Wu, R. Guo, Y. Liu, D. Yin. SMLP4Rec: An Efficient All-MLP Architecture for Sequential Recommendations. ACM TOIS, 2024. doi:10.1145/3637871",
        "[2] W. X. Zhao et al. RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. CIKM, 2021",
        "[3] M. Li, X. Zhao, C. Lyu, M. Zhao, R. Wu, R. Guo. MLP4Rec: A Pure MLP Architecture for Sequential Recommendations. IJCAI, 2022",
        "[4] L. Breiman, J. Friedman, R. Olshen, C. Stone. Classification and Regression Trees. Wadsworth, 1984",
        "[5] T. Hastie, R. Tibshirani, J. Friedman. The Elements of Statistical Learning, 2nd ed., §7.10. Springer, 2009",
    ]:
        b.p(ref, "ref")
    return b


# ---------------------------------------------------------------- post-processing


def set_footer(b: Builder) -> None:
    for sec in b.doc.sections:
        for p in sec.footer.paragraphs:
            for r in p.runs:
                if FOOTER_OLD in r.text:
                    r.text = r.text.replace(FOOTER_OLD, FOOTER_NEW)


def drop_orphan_images(b: Builder) -> int:
    """The template's week-2 figures stay related to the part after its body is cleared; drop them."""
    xml = b.doc.element.xml
    dropped = 0
    for rid, rel in list(b.doc.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in xml:
            b.doc.part.rels.pop(rid)
            dropped += 1
    return dropped


def refresh_toc_with_word(path: Path) -> bool:
    """Update every field (table of contents, page numbers) through Word COM; False if Word is missing."""
    ps = (
        "$ErrorActionPreference='Stop';"
        "$w=New-Object -ComObject Word.Application;$w.Visible=$false;"
        f"$d=$w.Documents.Open('{path}');"
        "foreach($t in $d.TablesOfContents){$t.Update()};"
        "$d.Fields.Update()|Out-Null;$d.Save();$d.Close();$w.Quit()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--no-word", action="store_true", help="skip the Word COM table-of-contents refresh"
    )
    args = ap.parse_args()
    b = build()
    set_footer(b)
    n_drop = drop_orphan_images(b)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b.save(OUT)
    toc = False if args.no_word else refresh_toc_with_word(OUT)
    print(
        f"wrote {OUT} ({b.fig_no} figures, {n_drop} template images dropped, "
        f"TOC {'refreshed with Word' if toc else 'not refreshed: press F9 in Word'})"
    )


if __name__ == "__main__":
    main()
