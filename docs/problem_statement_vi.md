Đề C1. Next-best-product recommendation cho du lịch (Vinpearl) và taxi / giao đồ ăn (GSM)

* Mục tiêu: Benchmark các họ mô hình gợi ý (matrix factorization, item2vec, sequence model như GRU4Rec/SASRec, LLM-based reranking) cho 2 bối cảnh mục tiêu của Q1/2027:
   * (a) Du lịch nghỉ dưỡng (Vinpearl): gợi ý điểm đến / hạng phòng / gói kỳ nghỉ, và gợi ý dịch vụ trong kỳ lưu trú (spa, vui chơi, F&B).
   * (b) Di chuyển & giao đồ ăn (GSM): gợi ý quán/món ăn theo ngữ cảnh, dự báo điểm đến chuyến xe, và cross-sell hai chiều taxi ↔ food.
* Câu hỏi nghiên cứu:
   * Du lịch có tần suất mua rất thấp, tính mùa vụ và ngữ cảnh chuyến đi (đi một mình / cặp đôi / gia đình) — mô hình session-based/context-aware nào vượt trội so với collaborative filtering truyền thống trên dữ liệu thưa?
   * Với giao đồ ăn, các yếu tố vị trí, khung giờ ăn, thời tiết, thời gian giao dự kiến đóng góp bao nhiêu vào chất lượng gợi ý; cân bằng tái đặt món quen (exploitation) và khám phá quán mới (exploration) thế nào?
   * Cross-sell liên PnL (khách đặt phòng Vinpearl → gợi ý xe GSM đón sân bay; khách hay gọi xe khung giờ trưa → gợi ý food): mô hình hóa ra sao khi hai nguồn dữ liệu tách biệt (transfer learning / shared embedding trên dữ liệu giả lập có liên kết ID)?
   * Đánh đổi giữa độ chính xác (recall@k, NDCG) và độ đa dạng/độ phủ danh mục ở từng bối cảnh?
* Dữ liệu đề xuất: Du lịch: Expedia Hotel Recommendations (Kaggle); Trivago RecSys Challenge 2019 (session-based hotel); Airbnb New User Bookings (Kaggle). Food delivery: Akeed Restaurant Recommendation Challenge (Kaggle — dữ liệu app giao đồ ăn thật); Yelp Open Dataset (nhà hàng + review + vị trí). Taxi: Porto Taxi Trajectory (Kaggle ECML/PKDD 2015 — dự báo điểm đến); NYC TLC Trip Records. MovieLens chỉ dùng prototype thuật toán nhanh.
* Bàn giao thêm: Catalog schema chung cho 3 loại "sản phẩm" (gói nghỉ/phòng, chuyến xe, món ăn) khớp mục Integration Design của roadmap + kịch bản demo cross-sell Vinpearl↔GSM trên dữ liệu giả lập liên kết.
