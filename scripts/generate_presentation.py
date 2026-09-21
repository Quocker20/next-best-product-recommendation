"""
Generate week1-report.pptx mirroring reports/slides/week1-report.html 100%.
Compatible with Microsoft PowerPoint, Google Slides, LibreOffice.
16:9 Widescreen layout, clean modern academic theme.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Color Palette
BG_PAGE = RGBColor(248, 250, 252)       # #f8fafc slate-50
CARD_BG = RGBColor(255, 255, 255)       # #ffffff
CARD_BORDER = RGBColor(226, 232, 240)   # #e2e8f0 slate-200
TEXT_MAIN = RGBColor(15, 23, 42)        # #0f172a slate-900
TEXT_MUTED = RGBColor(71, 85, 105)      # #475569 slate-600
TEXT_SUB = RGBColor(100, 116, 139)      # #64748b slate-500
PRIMARY = RGBColor(30, 64, 175)         # #1e40af blue-800
PRIMARY_LIGHT = RGBColor(239, 246, 255) # #eff6ff blue-50
PRIMARY_ACCENT = RGBColor(37, 99, 235)  # #2563eb blue-600
SUCCESS = RGBColor(22, 163, 74)         # #16a34a green-600
SUCCESS_BG = RGBColor(240, 253, 244)    # #f0fdf4 green-50
AMBER = RGBColor(217, 119, 6)           # #d97706 amber-600
AMBER_BG = RGBColor(254, 252, 232)      # #fefce8 amber-50
DANGER = RGBColor(220, 38, 38)          # #dc2626 red-600
DANGER_BG = RGBColor(254, 242, 242)     # #fef2f2 red-50
PURPLE = RGBColor(126, 34, 206)         # #7e22ce purple-700
PURPLE_BG = RGBColor(250, 245, 255)     # #faf5ff purple-50
ROW_HL = RGBColor(241, 245, 249)        # #f1f5f9
TH_BG = RGBColor(241, 245, 249)         # #f1f5f9
CALLOUT_BORDER = RGBColor(191, 219, 254) # #bfdbfe
CALLOUT_AMBER_BORDER = RGBColor(253, 230, 138) # #fde68a

FONT_FAMILY = "Segoe UI"


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_blank_slide():
        slide = prs.slides.add_slide(blank_layout)
        # Background shape
        bg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
        )
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_PAGE
        bg.line.fill.background()
        return slide

    def add_header(slide, meta, badge, title, subtitle):
        # Header text box
        box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(9.8), Inches(1.3))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Meta
        p0 = tf.paragraphs[0]
        p0.text = meta.upper()
        p0.font.name = FONT_FAMILY
        p0.font.size = Pt(9)
        p0.font.bold = True
        p0.font.color.rgb = PRIMARY_ACCENT
        p0.space_after = Pt(2)

        # Title
        p1 = tf.add_paragraph()
        p1.text = title
        p1.font.name = FONT_FAMILY
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.space_after = Pt(2)

        # Subtitle
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.name = FONT_FAMILY
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

        # Badge pill top right
        badge_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.8), Inches(0.45), Inches(1.75), Inches(0.4)
        )
        badge_box.fill.solid()
        badge_box.fill.fore_color.rgb = RGBColor(238, 242, 255)
        badge_box.line.color.rgb = RGBColor(199, 210, 254)
        badge_box.line.width = Pt(1)
        btf = badge_box.text_frame
        btf.vertical_anchor = MSO_ANCHOR.MIDDLE
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        bp.text = badge
        bp.font.name = FONT_FAMILY
        bp.font.size = Pt(9.5)
        bp.font.bold = True
        bp.font.color.rgb = PRIMARY

    def add_footer(slide, source, page_text):
        box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.35))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_SUB

        r_source = p.add_run()
        r_source.text = source

        # Page align right via separate box
        pbox = slide.shapes.add_textbox(Inches(10.5), Inches(7.0), Inches(2.033), Inches(0.35))
        ptf = pbox.text_frame
        ptf.word_wrap = True
        ptf.margin_left = ptf.margin_top = ptf.margin_right = ptf.margin_bottom = 0
        pp = ptf.paragraphs[0]
        pp.alignment = PP_ALIGN.RIGHT
        pp.font.name = FONT_FAMILY
        pp.font.size = Pt(9)
        pp.font.color.rgb = TEXT_SUB
        pp.text = page_text

    def add_card(slide, left, top, width, height, bg_rgb=CARD_BG, border_rgb=CARD_BORDER, top_accent_rgb=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_rgb
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(1)
        if top_accent_rgb:
            # Add top accent line
            accent = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.06)
            )
            accent.fill.solid()
            accent.fill.fore_color.rgb = top_accent_rgb
            accent.line.fill.background()
        return shape

    def add_stat_pill(slide, left, top, width, height, num_text, title_text, sub_text, highlight=False):
        card = add_card(
            slide, left, top, width, height,
            bg_rgb=RGBColor(240, 247, 255) if highlight else CARD_BG,
            border_rgb=RGBColor(147, 197, 253) if highlight else CARD_BORDER
        )
        box = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.08), width - Inches(0.3), height - Inches(0.16))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = num_text
        p0.font.name = FONT_FAMILY
        p0.font.size = Pt(18)
        p0.font.bold = True
        p0.font.color.rgb = PRIMARY if highlight else TEXT_MAIN
        p0.space_after = Pt(1)

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.name = FONT_FAMILY
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.space_after = Pt(1)

        p2 = tf.add_paragraph()
        p2.text = sub_text
        p2.font.name = FONT_FAMILY
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = TEXT_SUB

    def add_callout(slide, left, top, width, height, icon_text, content_text, is_amber=False):
        bg_c = AMBER_BG if is_amber else PRIMARY_LIGHT
        bd_c = CALLOUT_AMBER_BORDER if is_amber else CALLOUT_BORDER
        shape = add_card(slide, left, top, width, height, bg_rgb=bg_c, border_rgb=bd_c)

        box = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.08), width - Inches(0.3), height - Inches(0.16))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN

        r_icon = p.add_run()
        r_icon.text = icon_text + "  "
        r_icon.font.bold = True

        r_text = p.add_run()
        r_text.text = content_text

    # =========================================================================
    # SLIDE 1: OVERVIEW 11 DATASETS
    # =========================================================================
    s1 = add_blank_slide()
    add_header(s1, "Slide 01 • Khảo Sát & Sàng Lọc", "Tuần 1 • Screening",
               "Điểm Khởi Đầu: 3 Lĩnh Vực & 11 Bộ Dữ Liệu",
               "Đo kiểm các họ mô hình gợi ý (MF, item2vec, GRU4Rec/SASRec, LLM) trên Du lịch, Giao đồ ăn và Gọi xe.")

    # 3 Stat Pills
    add_stat_pill(s1, Inches(0.8), Inches(1.85), Inches(3.75), Inches(0.9),
                  "11 Datasets", "Sàng Lọc Toàn Diện", "8 bộ đề xuất + 3 bộ tìm kiếm mở rộng", highlight=True)
    add_stat_pill(s1, Inches(4.75), Inches(1.85), Inches(3.75), Inches(0.9),
                  "100% Scripted", "Đo Kiểm Bằng Code", "Xử lý chunked cho file lớn hơn RAM")
    add_stat_pill(s1, Inches(8.7), Inches(1.85), Inches(3.833), Inches(0.9),
                  "0 – 100 Điểm", "Rubric Độc Lập", "5 cổng Pass/Fail + 10 tiêu chí trọng số")

    # Table
    t_left, t_top, t_w, t_h = Inches(0.8), Inches(2.95), Inches(11.733), Inches(3.1)
    table_shape = s1.shapes.add_table(7, 5, t_left, t_top, t_w, t_h)
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.6)
    tbl.columns[1].width = Inches(4.3)
    tbl.columns[2].width = Inches(1.3)
    tbl.columns[3].width = Inches(2.0)
    tbl.columns[4].width = Inches(2.533)

    headers = ["Lĩnh vực", "Bộ dữ liệu (Dataset)", "Điểm", "Phân tầng", "Giấy phép sử dụng"]
    rows = [
        ["Du lịch", "Expedia Hotel Recommendations", "79", "Cốt lõi (Ưu tiên)", "Giới hạn (Kaggle Rules)"],
        ["Du lịch", "Trivago RecSys Challenge 2019", "77", "Cốt lõi (Dự phòng)", "Giới hạn (Kaggle mirror)"],
        ["Du lịch", "Airbnb New User (46) • Hotel Booking (46)", "46", "Loại bỏ", "Không đủ chuỗi / Trùng lặp"],
        ["Đồ ăn", "Akeed (73) • Instacart (69) • Yelp (62)", "62 – 73", "Thứ cấp", "Tọa độ giả lập / Lệch mục tiêu"],
        ["Gọi xe", "Porto Taxi (69) • Citi Bike (52) • NYC TLC (50)", "50 – 69", "Thứ cấp", "Không có ID hành khách để gợi ý"],
        ["Benchmark", "MovieLens 100k / 1M", "73", "Thử nghiệm", "Chỉ dùng kiểm thử đường ống"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TH_BG
        cell.text_frame.margin_left = cell.text_frame.margin_right = Inches(0.1)
        cell.text_frame.margin_top = cell.text_frame.margin_bottom = Inches(0.06)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN

    for r_idx, r_data in enumerate(rows):
        is_highlight = (r_idx == 0)
        for c_idx, val in enumerate(r_data):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if is_highlight else CARD_BG
            cell.text_frame.margin_left = cell.text_frame.margin_right = Inches(0.1)
            cell.text_frame.margin_top = cell.text_frame.margin_bottom = Inches(0.04)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_FAMILY
            p.font.size = Pt(9)
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = PRIMARY if c_idx in [1, 2] else TEXT_MAIN
            else:
                p.font.color.rgb = TEXT_MUTED if c_idx == 4 else TEXT_MAIN

    # Callout bottom
    add_callout(s1, Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.55),
                "💡 Nguyên tắc cốt lõi:",
                "Rubric xếp hạng chất lượng dữ liệu trong từng lĩnh vực, không tự quyết định lựa chọn lĩnh vực nghiên cứu. Xem chi tiết thang điểm tại Phụ lục A.")
    add_footer(s1, "Nguồn: reports/summary/eda_summary.md §2 • docs/dataset_scores.md", "Trang 01 / 08")

    # =========================================================================
    # SLIDE 2: SCOPE DECISION
    # =========================================================================
    s2 = add_blank_slide()
    add_header(s2, "Slide 02 • Quyết Định Chiến Lược", "Scope Decision",
               "Một Lĩnh Vực, Một Bộ Dữ Liệu: Du Lịch — Expedia",
               "Lựa chọn dựa trên tính khả thi thực tế khi trả lời câu hỏi nghiên cứu bằng dữ liệu thật.")

    col_w = Inches(3.75)
    gap = Inches(0.24)

    # Col 1: Travel Selected
    c1_left = Inches(0.8)
    add_card(s2, c1_left, Inches(1.85), col_w, Inches(4.2), top_accent_rgb=SUCCESS)
    b1 = s2.shapes.add_textbox(c1_left + Inches(0.2), Inches(1.95), col_w - Inches(0.4), Inches(4.0))
    tf1 = b1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. Du lịch — Expedia   [ĐƯỢC CHỌN]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SUCCESS
    p.space_after = Pt(8)

    bullets1 = [
        "Actor thực: 813,985 user có booking thật.",
        "Ngữ cảnh 100% thật: Đoàn khách (solo/cặp đôi/gia đình), ngày đi/về, gói combo, 2 năm chu kỳ mùa vụ.",
        "Dữ liệu cực sạch: 0% thiếu ngày đi/về; chỉ 8 khóa trùng / 3M booking.",
        "Khớp lộ trình mentor: Đáp ứng hoàn hảo Phase 3 (Sequential) & Phase 4 (Context-aware)."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Col 2: Food Deferred
    c2_left = c1_left + col_w + gap
    add_card(s2, c2_left, Inches(1.85), col_w, Inches(4.2), top_accent_rgb=AMBER)
    b2 = s2.shapes.add_textbox(c2_left + Inches(0.2), Inches(1.95), col_w - Inches(0.4), Inches(4.0))
    tf2 = b2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. Đồ ăn — Akeed   [TẠM HOÃN]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = AMBER
    p.space_after = Pt(8)

    bullets2 = [
        "Actor: 27,445 khách hàng thật.",
        "Tọa độ giả lập 100%: Tọa độ nằm ngoài Oman; không thể ghép dữ liệu thời tiết thực tế.",
        "Chất lượng lỗi: 41.1% đơn hàng có khoảng cách giao <= 0; 81 đơn trùng ID.",
        "Thiếu 2/4 yếu tố của RQ2: Rủi ro học thuật quá lớn nếu tiếp tục."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Col 3: Taxi Dropped
    c3_left = c2_left + col_w + gap
    add_card(s2, c3_left, Inches(1.85), col_w, Inches(4.2), top_accent_rgb=DANGER)
    b3 = s2.shapes.add_textbox(c3_left + Inches(0.2), Inches(1.95), col_w - Inches(0.4), Inches(4.0))
    tf3 = b3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Gọi xe — Porto / NYC   [KHÔNG KHẢ THI]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = DANGER
    p.space_after = Pt(8)

    bullets3 = [
        "Không có ID hành khách: Chỉ có ID tài xế (Porto), vùng đón (TLC), trạm xe (Citi Bike).",
        "Không thể cá nhân hóa: Không có lịch sử người đi để gợi ý sản phẩm kế tiếp.",
        "Lạc đề nghiên cứu: Bài toán dự đoán điểm đến taxi nằm ngoài kế hoạch các phase của mentor."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # 2 Bottom callouts
    add_callout(s2, Inches(0.8), Inches(6.2), Inches(5.75), Inches(0.6),
                "🎯 Trọng tâm:", "Tập trung giải quyết trọn vẹn RQ1 (So sánh mô hình theo ngữ cảnh) và RQ4 (Đánh đổi Accuracy vs Diversity).")
    add_callout(s2, Inches(6.783), Inches(6.2), Inches(5.75), Inches(0.6),
                "⏸️ Tạm hoãn:", "Tạm dừng RQ2 (Đồ ăn) và RQ3 (Bán chéo giả lập) để dồn toàn lực vào lĩnh vực Du lịch (Vinpearl proxy).", is_amber=True)
    add_footer(s2, "Nguồn: data cards expedia.md, akeed.md, porto_taxi.md • reports/summary/eda_summary.md", "Trang 02 / 08")

    # =========================================================================
    # SLIDE 3: PROBLEM DEFINITION
    # =========================================================================
    s3 = add_blank_slide()
    add_header(s3, "Slide 03 • Định Nghĩa Bài Toán", "Problem Formulation",
               "Next-Best-Product Cho Du Lịch: Định Nghĩa & Thách Thức",
               "Dự đoán điểm đến / hạng phòng / gói nghỉ dưỡng khách hàng có xác suất đặt cao nhất.")

    # Left: Formulation
    add_card(s3, Inches(0.8), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=PRIMARY_ACCENT)
    b_left = s3.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.35), Inches(4.0))
    tfl = b_left.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "Không Gian Bài Toán (Expedia Proxy)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    f_bullets = [
        "Đầu vào du khách: user_id, lịch sử đặt, nguồn gốc khách, khoảng cách ước tính.",
        "Ngữ cảnh chuyến đi:\n  • Đoàn: Người lớn, trẻ em, số phòng\n  • Lịch trình: Ngày nhận / trả phòng (mùa vụ)\n  • Kênh: Tìm kiếm điểm đến (srch_destination_id), cờ gói combo (is_package).",
        "Mục tiêu dự đoán (Target): 100 cụm khách sạn (hotel_cluster) — đại diện cho hạng phòng / combo nghỉ dưỡng Vinpearl."
    ]
    for b in f_bullets:
        p = tfl.add_paragraph()
        p.text = "• " + b
        p.font.name = FONT_FAMILY
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    # Right: 4 Challenges
    add_card(s3, Inches(6.783), Inches(1.85), Inches(5.75), Inches(4.2))
    b_right_title = s3.shapes.add_textbox(Inches(6.983), Inches(1.95), Inches(5.35), Inches(0.4))
    tfr = b_right_title.text_frame
    p = tfr.paragraphs[0]
    p.text = "4 Thách Thức Đặc Thù Của Du Lịch"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    # 4 sub-pills
    add_stat_pill(s3, Inches(6.983), Inches(2.45), Inches(2.6), Inches(1.6),
                  "Median = 2", "Tần Suất Rất Thấp", "38.8% user chỉ đặt đúng 1 lần trong đời")
    add_stat_pill(s3, Inches(9.7), Inches(2.45), Inches(2.6), Inches(1.6),
                  "5.4% → 10.7%", "Mùa Vụ Biến Động", "Đáy Tháng 2 (5.4%), Đỉnh Tháng 8 (10.7%)")
    add_stat_pill(s3, Inches(6.983), Inches(4.25), Inches(2.6), Inches(1.6),
                  "44% vs 19%", "Ngữ Cảnh Đoàn Khách", "Cặp đôi (44%) vs Gia đình (19%) có nhu cầu rất khác")
    add_stat_pill(s3, Inches(9.7), Inches(4.25), Inches(2.6), Inches(1.6),
                  "97.10%", "Độ Thưa Tương Tác", "814K user × 100 cụm phòng nghỉ dưỡng")

    add_callout(s3, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "⚠️ Giới hạn phạm vi (Known Gap):",
                "Dịch vụ tại điểm lưu trú (in-stay: spa, ẩm thực, vui chơi) không có dataset công khai tương ứng. Do đó, benchmark tập trung 100% vào quyết định đặt phòng trước chuyến đi (pre-stay).",
                is_amber=True)
    add_footer(s3, "Nguồn: docs/data_cards/expedia.md • reports/summary/expedia_profile.json", "Trang 03 / 08")

    # =========================================================================
    # SLIDE 4: BENCHMARK PROTOCOL
    # =========================================================================
    s4 = add_blank_slide()
    add_header(s4, "Slide 04 • Thiết Kế Thực Nghiệm", "Benchmark Protocol",
               "5 Họ Mô Hình Benchmark & Giao Thức Đánh Giá",
               "So sánh công bằng, chặt chẽ giữa các kiến trúc truyền thống, học biểu diễn chuỗi và LLM.")

    # 5 Models
    m_w = Inches(2.22)
    m_gap = Inches(0.15)
    models_info = [
        ("1. Baselines", "Popularity (toàn cục, theo điểm đến, theo mùa), Repeat-Last, ItemKNN.", "Chuẩn so sánh", CARD_BG, TEXT_SUB),
        ("2. Matrix Fact.", "Học vector tiềm ẩn User-Item. 61.2% user có >= 2 booking để huấn luyện.", "CF cổ điển", PRIMARY_LIGHT, PRIMARY),
        ("3. item2vec", "Nhúng item từ sự đồng xuất hiện trong chuỗi thời gian thực của user.", "Skip-gram", PRIMARY_LIGHT, PRIMARY),
        ("4. GRU4Rec/SAS", "Học chuỗi tuần tự; tích hợp side features (đoàn khách, ngày đi).", "Sequential", PRIMARY_LIGHT, PRIMARY),
        ("5. LLM Rerank", "Tái xếp hạng top-N dựa trên prompt ngữ cảnh và 149 latent features.", "GenAI", PURPLE_BG, PURPLE)
    ]

    for idx, (m_title, m_desc, m_tag, m_bg, m_c) in enumerate(models_info):
        mx = Inches(0.8) + idx * (m_w + m_gap)
        add_card(s4, mx, Inches(1.85), m_w, Inches(1.9), bg_rgb=m_bg)
        mb = s4.shapes.add_textbox(mx + Inches(0.12), Inches(1.95), m_w - Inches(0.24), Inches(1.7))
        mtf = mb.text_frame
        mtf.word_wrap = True
        mp0 = mtf.paragraphs[0]
        mp0.text = m_title
        mp0.font.name = FONT_FAMILY
        mp0.font.size = Pt(11)
        mp0.font.bold = True
        mp0.font.color.rgb = m_c
        mp0.space_after = Pt(4)

        mp1 = mtf.add_paragraph()
        mp1.text = m_desc
        mp1.font.name = FONT_FAMILY
        mp1.font.size = Pt(8.5)
        mp1.font.color.rgb = TEXT_MUTED

    # Protocol Card
    add_card(s4, Inches(0.8), Inches(3.95), Inches(11.733), Inches(2.8), top_accent_rgb=PRIMARY_ACCENT)
    proto_box = s4.shapes.add_textbox(Inches(1.0), Inches(4.05), Inches(11.333), Inches(0.4))
    ptf = proto_box.text_frame
    pp = ptf.paragraphs[0]
    pp.text = "Giao Thức Đánh Giá Cố Định (Fixed Evaluation Protocol)"
    pp.font.name = FONT_FAMILY
    pp.font.size = Pt(13)
    pp.font.bold = True
    pp.font.color.rgb = PRIMARY

    # 3 Protocol Columns
    p_col_w = Inches(3.6)
    p_gap = Inches(0.25)
    proto_cols = [
        ("1. Cắt Thời Gian 80/20",
         "Mốc cắt 2014-09-29: 2.4M train / 600K test.\n30.1% user tập test chưa từng có ở train -> Thiết lập lát cắt Cold-Start độc lập đánh giá riêng biệt."),
        ("2. Full Ranking (100 Cụm)",
         "Xếp hạng trên toàn bộ 100 cụm khách sạn, không lấy mẫu âm tính giả định (no sampled negatives).\nĐo lường Recall@K, NDCG@K, MRR tại K = 5, 10, 20."),
        ("3. Đa Dạng & Tin Cậy",
         "Bổ sung chỉ số vượt ngoài độ chính xác: Catalog Coverage, Intra-list Diversity, Popularity Bias.\nBáo cáo kết quả trung bình Mean ± Std trên >= 3 random seeds.")
    ]

    for p_idx, (p_title, p_desc) in enumerate(proto_cols):
        px = Inches(1.0) + p_idx * (p_col_w + p_gap)
        p_card_box = s4.shapes.add_textbox(px, Inches(4.5), p_col_w, Inches(2.1))
        pctf = p_card_box.text_frame
        pctf.word_wrap = True
        p0 = pctf.paragraphs[0]
        p0.text = p_title
        p0.font.name = FONT_FAMILY
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = TEXT_MAIN
        p0.space_after = Pt(4)

        p1 = pctf.add_paragraph()
        p1.text = p_desc
        p1.font.name = FONT_FAMILY
        p1.font.size = Pt(9.5)
        p1.font.color.rgb = TEXT_MUTED

    add_footer(s4, "Nguồn: CLAUDE.md §8 • docs/eval_protocol.md", "Trang 04 / 08")

    # =========================================================================
    # SLIDE 5: WHY EXPEDIA & HONEST LIMITS
    # =========================================================================
    s5 = add_blank_slide()
    add_header(s5, "Slide 05 • Minh Bạch Phương Pháp", "Methodological Rigor",
               "Vì Sao Chọn Expedia & Những Giới Hạn Cần Nêu Rõ",
               "Minh bạch học thuật: 5 ưu thế cốt lõi và 5 giới hạn kỹ thuật bắt buộc phải công khai.")

    # 5 Strengths
    add_card(s5, Inches(0.8), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=SUCCESS)
    sb_box = s5.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.35), Inches(4.0))
    stf = sb_box.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = "5 Lý Do Expedia Là Bộ Dữ Liệu Chính"
    sp.font.name = FONT_FAMILY
    sp.font.size = Pt(12.5)
    sp.font.bold = True
    sp.font.color.rgb = SUCCESS
    sp.space_after = Pt(8)

    strengths = [
        "1. Khớp bài toán (Task fit): User x Cluster booking là bài toán next-item trực tiếp (C1 = 2/3).",
        "2. Đủ mọi yếu tố của RQ1: Tần suất thấp, mùa vụ 2 năm, đoàn khách và user thật để phân lát.",
        "3. Lõi dữ liệu cực sạch: 0% thiếu ngày check-in/out, chỉ 8 key trùng / 3M booking.",
        "4. Khả thi & Tái lập: Script chunked pass nạp vừa RAM; không đòi hỏi GPU đắt tiền cho baseline.",
        "5. Điểm Rubric cao nhất: 79/100 (tăng lên 80/100 khi bỏ tiêu chí bán chéo)."
    ]
    for s in strengths:
        p = stf.add_paragraph()
        p.text = "• " + s
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # 5 Limits
    add_card(s5, Inches(6.783), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=DANGER)
    lb_box = s5.shapes.add_textbox(Inches(6.983), Inches(1.95), Inches(5.35), Inches(4.0))
    ltf = lb_box.text_frame
    ltf.word_wrap = True
    lp = ltf.paragraphs[0]
    lp.text = "5 Giới Hạn Bắt Buộc Công Khai"
    lp.font.name = FONT_FAMILY
    lp.font.size = Pt(12.5)
    lp.font.bold = True
    lp.font.color.rgb = DANGER
    lp.space_after = Pt(8)

    limits = [
        "1. Target là cụm số ẩn danh: Không có text mô tả phòng -> LLM rerank dựa trên ngữ cảnh + latent features.",
        "2. Không có cột giá tiền: Khoảng cách bị thiếu 33.8% trên tập booking.",
        "3. 38.8% user chỉ có 1 booking: Lát cắt Cold-Start phải báo cáo riêng, không tính cào bằng.",
        "4. Click chiếm 92%: Click là tín hiệu yếu -> Không gộp làm positive ground-truth.",
        "5. Giấy phép Kaggle: Chỉ phục vụ nghiên cứu nội bộ, không phân phối lại dữ liệu gốc."
    ]
    for l in limits:
        p = ltf.add_paragraph()
        p.text = "• " + l
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_callout(s5, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "🛡️ Quan điểm học thuật:",
                "Nêu rõ giới hạn giúp thiết kế quy trình đánh giá trung thực, tránh rò rỉ dữ liệu hoặc kết luận quá lạc quan khi nghiệm thu đề tài.")
    add_footer(s5, "Nguồn: docs/data_cards/expedia.md • reports/summary/travel_rubric_without_c9.txt", "Trang 05 / 08")

    # =========================================================================
    # SLIDE 6: SCHEMA & ANOMALIES
    # =========================================================================
    s6 = add_blank_slide()
    add_header(s6, "Slide 06 • Cấu Trúc & Bất Thường Dữ Liệu", "Schema & Quality",
               "Ý Nghĩa Trường Cốt Lõi & Bất Thường Đo Trên Data Thật",
               "Khảo sát 24 trường trong train.csv, giải nghĩa các trường trọng tâm và kiểm định bất thường trên 37.7M dòng.")

    # Core Fields
    add_card(s6, Inches(0.8), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=PRIMARY_ACCENT)
    fld_box = s6.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.35), Inches(4.0))
    ftf = fld_box.text_frame
    ftf.word_wrap = True
    fp = ftf.paragraphs[0]
    fp.text = "Ý Nghĩa Các Trường Cốt Lõi (Core Fields)"
    fp.font.name = FONT_FAMILY
    fp.font.size = Pt(12)
    fp.font.bold = True
    fp.font.color.rgb = PRIMARY
    fp.space_after = Pt(6)

    fields_desc = [
        ("user_id", "Mã du khách duy nhất (1,198,786 user toàn file, 813,985 user có booking)."),
        ("is_booking", "Cờ tương tác: 1 = Đặt phòng (Signal dương), 0 = Click (Tín hiệu yếu)."),
        ("date_time", "Thời điểm tương tác (07/01/2013 -> 31/12/2014, dùng chia temporal split)."),
        ("srch_ci / srch_co", "Ngày nhận/trả phòng — Xác định mùa vụ và độ dài chuyến đi."),
        ("srch_adults/children", "Quy mô đoàn — Phân loại Solo / Cặp đôi / Gia đình / Nhóm."),
        ("srch_destination_id", "Mã điểm đến du khách tìm kiếm (36,933 điểm đến)."),
        ("is_package", "Cờ đặt kèm combo (1 = Gói combo nghỉ dưỡng, 0 = Phòng lẻ)."),
        ("hotel_cluster", "Target cần xếp hạng: 100 cụm khách sạn / hạng phòng (0–99).")
    ]
    for fn, fd in fields_desc:
        p = ftf.add_paragraph()
        p.text = f"• {fn}: {fd}"
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(3)

    # Measured Anomalies
    add_card(s6, Inches(6.783), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=DANGER)
    ano_box = s6.shapes.add_textbox(Inches(6.983), Inches(1.95), Inches(5.35), Inches(4.0))
    atf = ano_box.text_frame
    atf.word_wrap = True
    ap = atf.paragraphs[0]
    ap.text = "Bất Thường Đo Thực Tế (Data Anomalies)"
    ap.font.name = FONT_FAMILY
    ap.font.size = Pt(12)
    ap.font.bold = True
    ap.font.color.rgb = DANGER
    ap.space_after = Pt(6)

    anomalies_desc = [
        "Mất cân bằng tương tác cực lớn (92% vs 8%):\nClick xem chiếm 92.03% (34.7M dòng) vs Booking chỉ chiếm 7.97% (3.0M dòng).\n-> Xử lý: Chỉ dùng Booking làm ground-truth dương.",
        "Khuyết thiếu khoảng cách địa lý (33.8% – 35.9%):\norig_destination_distance thiếu 35.9% toàn file (thiếu 33.8% trên booking).\n-> Điểm sáng: Ngày check-in/out thiếu 0% trên booking; 21 cột còn lại 0% missing.",
        "Bất thường logic cực hiếm (Đã cô lập sạch):\n• Check-out trước Check-in: Chỉ 3 dòng trên 3.0M booking.\n• Không có người lớn (0 adults): 0.17% (5,239 dòng trên booking).\n• Trùng lặp khóa (user, time, dest, cluster): Chỉ 8 bản ghi trên booking."
    ]
    for ad in anomalies_desc:
        p = atf.add_paragraph()
        p.text = "• " + ad
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(5)

    add_callout(s6, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "📊 Kết luận chất lượng:",
                "Dữ liệu có tỷ lệ khuyết thiếu khoảng cách cao và mất cân bằng Click/Booking lớn, nhưng lõi giao dịch Booking (3.0M dòng) cực sạch, đầy đủ 100% ngày tháng và ngữ cảnh để huấn luyện mô hình.")
    add_footer(s6, "Nguồn: docs/data_cards/expedia.md • reports/summary/expedia_profile.json", "Trang 06 / 08")

    # =========================================================================
    # SLIDE 7: VERIFIED PROFILE
    # =========================================================================
    s7 = add_blank_slide()
    add_header(s7, "Slide 07 • Thẩm Định Thực Tế", "Verified Profile",
               "Expedia Hotel: Hồ Sơ Dữ Liệu Đã Kiểm Chứng",
               "Số liệu quét toàn vẹn 37,670,293 dòng từ script profile_expedia.py.")

    # 4 Big Stat Cards
    add_stat_pill(s7, Inches(0.8), Inches(1.85), Inches(2.78), Inches(1.1),
                  "37.67M", "Tổng Quy Mô Dữ Liệu", "34.67M click (92%) vs 3.00M booking (8%)", highlight=True)
    add_stat_pill(s7, Inches(3.78), Inches(1.85), Inches(2.78), Inches(1.1),
                  "813,985", "Khách Hàng Đặt Phòng", "61.2% khách hàng có >= 2 booking")
    add_stat_pill(s7, Inches(6.76), Inches(1.85), Inches(2.78), Inches(1.1),
                  "100 Cụm", "Lớp Khách Sạn (Classes)", "Độ thưa tương tác: 97.10%")
    add_stat_pill(s7, Inches(9.74), Inches(1.85), Inches(2.793), Inches(1.1),
                  "2 Năm", "Chu Kỳ Thời Gian", "07/01/2013 -> 31/12/2014")

    # 3 Detail Panels
    p_w = Inches(3.75)
    p_gap = Inches(0.24)

    # Panel 1
    px1 = Inches(0.8)
    add_card(s7, px1, Inches(3.15), p_w, Inches(2.85))
    b = s7.shapes.add_textbox(px1 + Inches(0.15), Inches(3.25), p_w - Inches(0.3), Inches(2.65))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Chiều Sâu & Hành Vi Lặp"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p.space_after = Pt(6)
    p_b1 = [
        "Trung vị số booking/user: 2.0 (Mean = 3.69, P90 = 8.0).",
        "Chỉ 21.3% chọn lại cụm cũ: Phần lớn là tìm kiếm cái mới -> Không bị lấn át bởi baseline repeat-last.",
        "Độ tập trung: Cụm Top 1 chiếm 4.0%, Top 10 chiếm 22.9% (Đuôi dài vừa phải)."
    ]
    for txt in p_b1:
        p = tf.add_paragraph()
        p.text = "• " + txt
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    # Panel 2
    px2 = px1 + p_w + p_gap
    add_card(s7, px2, Inches(3.15), p_w, Inches(2.85))
    b = s7.shapes.add_textbox(px2 + Inches(0.15), Inches(3.25), p_w - Inches(0.3), Inches(2.65))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Mùa Vụ & Gói Combo"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p.space_after = Pt(6)
    p_b2 = [
        "Tháng cao điểm: T8 (10.7%), T12 (10.1%), T10 (9.8%).",
        "Tháng thấp điểm: T2 (5.4%), T1 (6.2%).",
        "13.7% booking là gói combo (is_package = 1).",
        "36,933 điểm đến tìm kiếm khác nhau."
    ]
    for txt in p_b2:
        p = tf.add_paragraph()
        p.text = "• " + txt
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    # Panel 3
    px3 = px2 + p_w + p_gap
    add_card(s7, px3, Inches(3.15), p_w, Inches(2.85))
    b = s7.shapes.add_textbox(px3 + Inches(0.15), Inches(3.25), p_w - Inches(0.3), Inches(2.65))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Cơ Cấu Đoàn Khách (Party)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p.space_after = Pt(6)
    p_b3 = [
        "👫 Cặp đôi (Couple): 44.0%",
        "🚶 Đi một mình (Solo): 28.9%",
        "👨‍👩‍👧 Gia đình (Family): 18.8%",
        "👥 Nhóm người lớn (>= 3 người): 8.1%"
    ]
    for txt in p_b3:
        p = tf.add_paragraph()
        p.text = "• " + txt
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    add_callout(s7, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "🏨 Ánh xạ tương đương Vinpearl:",
                "hotel_cluster -> Hạng phòng/Combo • srch_destination_id -> Điểm đến • is_package -> Gói nghỉ dưỡng • Cơ cấu đoàn & Ngày -> Ngữ cảnh chuyến đi.")
    add_footer(s7, "Nguồn: docs/data_cards/expedia.md • reports/summary/expedia_profile.json", "Trang 07 / 08")

    # =========================================================================
    # SLIDE 8: CONTINGENCY DATASET - TRIVAGO 2019
    # =========================================================================
    s8 = add_blank_slide()
    add_header(s8, "Slide 08 • Quản Trị Rủi Ro", "Risk Management",
               "Phương Án Dự Phòng: Trivago RecSys Challenge 2019",
               "Kích hoạt chuyển đổi ngay nếu Expedia bị trùng lặp đề tài với nhóm nghiên cứu khác.")

    # Left: Trivago Strengths
    add_card(s8, Inches(0.8), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=PRIMARY_ACCENT)
    tb_box = s8.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.35), Inches(4.0))
    ttf = tb_box.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "Thế Mạnh Của Trivago (Điểm Rubric: 77/100)"
    tp.font.name = FONT_FAMILY
    tp.font.size = Pt(12)
    tp.font.bold = True
    tp.font.color.rgb = PRIMARY
    tp.space_after = Pt(8)

    triv_strengths = [
        "Chuẩn Session-based: 15.9M dòng, 910K phiên, 1.58M clickout kèm giá hiển thị thực tế.",
        "Có text metadata: 927K khách sạn có nhãn tiện ích (item_metadata.csv) -> Rất có lợi cho LLM Reranker.",
        "Tọa độ thực: Tên thành phố/quốc gia thật; đã test ghép thời tiết Open-Meteo khả thi.",
        "Dữ liệu sạch: 0% missing trên 9 cột cốt lõi."
    ]
    for s in triv_strengths:
        p = ttf.add_paragraph()
        p.text = "• " + s
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Right: Trade-offs
    add_card(s8, Inches(6.783), Inches(1.85), Inches(5.75), Inches(4.2), top_accent_rgb=AMBER)
    tr_box = s8.shapes.add_textbox(Inches(6.983), Inches(1.95), Inches(5.35), Inches(4.0))
    trtf = tr_box.text_frame
    trtf.word_wrap = True
    trp = trtf.paragraphs[0]
    trp.text = "Đánh Đổi Nếu Phải Chuyển Đổi"
    trp.font.name = FONT_FAMILY
    trp.font.size = Pt(12)
    trp.font.bold = True
    trp.font.color.rgb = AMBER
    trp.space_after = Pt(8)

    tradeoffs = [
        "Chỉ có 6 ngày dữ liệu: Mất hoàn toàn lát cắt mùa vụ trong năm của RQ1.",
        "Bản chất bài toán đổi: Chuyển thành bài toán session-based next-clickout thay vì chuỗi dài hạn.",
        "Độ thưa cực lớn (99.999%): 289K khách sạn, 56% user chỉ xuất hiện 1 lần duy nhất.",
        "Tín hiệu là Click: Không phải hành vi Booking thực tế (C7 = 2 so với 3 của Expedia)."
    ]
    for t in tradeoffs:
        p = trtf.add_paragraph()
        p.text = "• " + t
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_callout(s8, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "❌ Các ứng viên du lịch khác bị loại:",
                "Airbnb (46 - Drop) chỉ có 1 booking/user và target 12 quốc gia (58% NDF); Hotel Booking Demand (46 - Drop) không có ID khách hàng và 26.8% dòng trùng lặp.",
                is_amber=True)
    add_footer(s8, "Nguồn: docs/data_cards/trivago_2019.md, airbnb_new_user.md • docs/dataset_scores.md", "Trang 08 / 08")

    # =========================================================================
    # SLIDE 9: APPENDIX A - RUBRIC
    # =========================================================================
    s9 = add_blank_slide()
    add_header(s9, "Phụ Lục A • Thang Đo Rubric (Tham Chiếu)", "Appendix A",
               "Phương Pháp Chấm Điểm 11 Bộ Dữ Liệu",
               "Quy trình chấm điểm 3 bước: 5 cổng Pass/Fail, 10 tiêu chí trọng số (thang 100), và kiểm định độ nhạy.")

    a_col_w = Inches(3.75)
    a_gap = Inches(0.24)

    # Step 1
    ax1 = Inches(0.8)
    add_card(s9, ax1, Inches(1.85), a_col_w, Inches(4.2))
    b = s9.shapes.add_textbox(ax1 + Inches(0.15), Inches(1.95), a_col_w - Inches(0.3), Inches(4.0))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Bước 1: 5 Cổng Pass / Fail"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)
    g_bullets = [
        "G1: Giấy phép cho phép nghiên cứu.",
        "G2: Sản phẩm gợi ý có ID ổn định.",
        "G3: Nạp được trên máy phát triển.",
        "G4: Ánh xạ được thành User x Item.",
        "G5: Sẵn có trong data/raw/ và tái lập bằng script."
    ]
    for g in g_bullets:
        p = tf.add_paragraph()
        p.text = "• " + g
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Step 2
    ax2 = ax1 + a_col_w + a_gap
    add_card(s9, ax2, Inches(1.85), a_col_w, Inches(4.2))
    b = s9.shapes.add_textbox(ax2 + Inches(0.15), Inches(1.95), a_col_w - Inches(0.3), Inches(4.0))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Bước 2: 10 Tiêu Chí (0–3 đ)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)
    c_bullets = [
        "C1: Khớp bài toán đề bài (x3)",
        "C2: Định danh User / Session (x2)",
        "C3: Thông tin chuỗi thời gian (x2)",
        "C4: Chiều sâu lịch sử tương tác (x2)",
        "C5: Ngữ cảnh thực tế không giả lập (x2)",
        "C6–C10: Metadata, Tín hiệu, Tính khả thi, Bán chéo, Chất lượng (mỗi mục x1)"
    ]
    for c in c_bullets:
        p = tf.add_paragraph()
        p.text = "• " + c
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Step 3
    ax3 = ax2 + a_col_w + a_gap
    add_card(s9, ax3, Inches(1.85), a_col_w, Inches(4.2), bg_rgb=PRIMARY_LIGHT, border_rgb=CALLOUT_BORDER)
    b = s9.shapes.add_textbox(ax3 + Inches(0.15), Inches(1.95), a_col_w - Inches(0.3), Inches(4.0))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Bước 3: Quy Đổi Thang 100"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(8)
    s_bullets = [
        "Công thức: Điểm = (Tổng điểm / 48) x 100",
        ">= 70 Cốt lõi (Core): Expedia (79), Trivago (77), Akeed (73)",
        "50–69 Thứ cấp (Secondary): Porto (69), Instacart (69), Yelp (62)...",
        "< 50 Loại bỏ (Drop): Airbnb (46), Hotel Booking Demand (46)"
    ]
    for sb in s_bullets:
        p = tf.add_paragraph()
        p.text = "• " + sb
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_callout(s9, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.6),
                "🧪 Kiểm định độ nhạy C9 (Sensitivity Test):",
                "Khi loại bỏ tiêu chí bán chéo C9 (tối đa 45 điểm), điểm số mảng du lịch tăng lên nhưng thứ hạng hoàn toàn không đổi: Expedia (80 - Core) > Trivago (78 - Core) > Airbnb (47 - Drop) > Hotel Demand (47 - Drop). Khẳng định tính vững chắc của quyết định.")
    add_footer(s9, "Nguồn: docs/dataset_rubric.md • reports/summary/travel_rubric_without_c9.txt", "Phụ Lục A (Tham chiếu)")

    # Save
    out_path = "reports/slides/week1-report.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to: {out_path}")


if __name__ == "__main__":
    create_deck()
