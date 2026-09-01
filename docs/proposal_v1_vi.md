# Đề xuất Nghiên cứu (v1): Truyền thông KV-Cache Ẩn Thích ứng cho LLM Đa Agent

> **Trạng thái**: Bản thảo v1 — 15-08-2026
> **Phát triển dựa trên**: KVComm (ICLR 2026) + tích hợp LatentMAS (`com_latent.py`)
> **Các mô hình nghiên cứu**: `Qwen/Qwen3-4B`, `suayptalha/DeepSeek-R1-Distill-Llama-3B`
> **Bộ dữ liệu đánh giá (Benchmark)**: HotpotQA, MedQA, TMATH, MultiFieldQA-EN, Tipsheets

---

## 1. Bối cảnh Nghiên cứu

**KVComm** cho phép hai agent LLM giao tiếp bằng cách truyền trực tiếp key-value attention cache của bên gửi (Mô hình A) sang bên nhận (Mô hình B), thay vì trao đổi các thông điệp bằng ngôn ngữ tự nhiên. Để việc truyền tải đạt hiệu quả cao, chỉ một tập hợp con các lớp được chia sẻ, được lựa chọn theo điểm số mức độ quan trọng dựa trên attention (được hiệu chỉnh trên một vài mẫu và kết hợp với phân phối chuẩn Gaussian theo chiều sâu — xem `layer_importance.py`).

Mở rộng hiện tại tích hợp **LatentMAS**: trước khi bàn giao KV cache của mình, Mô hình A thực hiện *N* lượt lan truyền tiến "suy nghĩ ẩn" (latent thinking) bổ sung — trạng thái ẩn cuối cùng của nó được chiếu ngược trở lại không gian embedding thông qua ma trận căn chỉnh W đã học và được đưa trở lại làm đầu vào tiếp theo, thêm N token KV ẩn vào cache (`models_latent.py`).

Hai chế độ được so sánh:
- **Mode 1** — suy nghĩ ẩn + toàn bộ KV cache (tất cả các lớp) được chuyển sang B.
- **Mode 2** — suy nghĩ ẩn + truyền lớp chọn lọc (KVComm, top-k% số lớp).

## 2. Các Phát hiện Chính cho đến Nay (EXPERIMENT_RESULTS.md, Tháng 8/2026)

1. **Suy nghĩ ẩn hỗ trợ các nhiệm vụ suy luận.** LatentMAS vượt qua KVComm thuần trên HotpotQA (72.80% so với 70.00%) và TMATH (+2.7 đến +5.0 điểm). Ít bước ẩn (1–5) cho kết quả tốt nhất; vượt quá ~10 bước, độ chính xác giảm dần và phản hồi rác tăng lên (lên tới 6% trên DeepSeek-R1-Distill ở 25 bước).
2. **Suy nghĩ ẩn gây hại cho các nhiệm vụ trích xuất.** Trên MultiFieldQA-EN (truy xuất thực tế ngữ cảnh dài), KVComm thuần dẫn đầu (50.00% so với 47.33%): ép A vào trạng thái suy luận làm biến dạng biểu diễn ngữ cảnh nguyên văn mà B cần.
3. **Tỉa bớt lớp đồng nhất xung đột với truyền suy nghĩ ẩn.** Khi bật suy nghĩ ẩn, Mode 1 (đầy đủ các lớp) vượt trội so với Mode 2 (top 70%) khoảng ~3–5 điểm trên HotpotQA và MultiFieldQA-EN — việc cắt giảm các lớp một cách đồng nhất trên toàn bộ chuỗi sẽ loại bỏ thông tin quan trọng đối với đầu vào dài.
4. **Đánh đổi về độ trễ.** KVComm thuần nhanh hơn nhiều (ví dụ: MedQA: 5 phút so với 40–60 phút cho các biến thể ẩn), vì các biến thể ẩn thêm N lượt lan truyền tiến trên A và mở rộng ngữ cảnh mà B phải chú ý (T_A + N + T_B).

**Thấu hiểu cốt lõi**: truyền thông KV ẩn vượt trội hơn trao đổi văn bản đối với nhiệm vụ suy luận, nhưng việc truyền tải phải trở nên *thích ứng với nội dung và nhiệm vụ* thay vì áp dụng đồng nhất.

## 3. Các Hướng Đề xuất

### 3.1 Định tuyến KV Chọn lọc Kép (Dual-Selective KV Routing)
Định tuyến các lớp **theo từng loại token** thay vì tỉa bớt đồng nhất:
- Token ngữ cảnh ban đầu (T_A): giữ các lớp từ nông đến trung bình (~0–14) để bảo toàn thông tin thực tế/nguyên văn (sửa sự suy giảm trên MultiFieldQA-EN).
- Token suy nghĩ ẩn (N): giữ các lớp từ trung bình đến sâu (~14–35) nơi chứa các biểu diễn suy luận.

Mục tiêu: Giảm ~70% kích thước KV truyền đi trong khi vẫn giữ nguyên cả độ trung thực truy xuất lẫn khả năng suy luận.

### 3.2 Bước Tư duy Ẩn Thích ứng & Dừng Sớm (Adaptive Latent Steps & Early Exit)
Tự động dừng vòng lặp ẩn khi trạng thái ẩn hội tụ (độ tương đồng cosine của h⁽ⁿ⁾ so với h⁽ⁿ⁻¹⁾ ≈ 1). Kỳ vọng: Giảm ~50% thời gian thực thi và loại bỏ sự suy giảm độ chính xác quan sát được ở số bước cao.

### 3.3 Căn chỉnh Thặng dư Điểm neo (Anchor Residual Realignment)
Ổn định vòng lặp ẩn bằng cập nhật có điểm neo:

  h̃⁽ⁿ⁾ = α · W · h⁽ⁿ⁻¹⁾ + (1 − α) · h⁽⁰⁾

nhằm triệt tiêu sự trôi lệch biểu diễn và loại bỏ các phản hồi rác trên các mô hình chắt lọc (DeepSeek-R1-Distill).

### 3.4 Định tuyến Chế độ theo Nhiệm vụ (Task-Aware Mode Routing)
Tự động phát hiện loại nhiệm vụ và định tuyến:
- **Truy xuất thực tế / ngữ cảnh dài** → KVComm thuần hoặc N = 1.
- **Suy luận đa bước / toán học** → LatentMAS với N = 2–5.

## 4. Kế hoạch Đánh giá

- **Mô hình cơ sở (Baselines)**: baseline chỉ dùng B (B-only), ranh giới lý tưởng đầy đủ ngữ cảnh (full-context skyline), KVComm thuần (top 70%), LatentMAS Mode 1/2, NLD, CIPHER.
- **Thước đo (Metrics)**: độ chính xác nhiệm vụ (EM/F1/Rouge-L), thời gian thực thi thực tế, kích thước KV truyền đi, tỷ lệ phản hồi rác.
- **Thử nghiệm cắt giảm (Ablations)**: chọn lớp ngẫu nhiên so với chọn lớp theo độ quan trọng, thử nghiệm quét số bước ẩn, phân chia định tuyến theo từng loại token, quét tham số α cho căn chỉnh điểm neo.
- **Tiêu chí thành công**:
  1. Định tuyến chọn lọc kép đạt hoặc vượt độ chính xác của Mode 1 trên các nhiệm vụ suy luận, đồng thời phục hồi độ chính xác của KVComm thuần trên MultiFieldQA-EN, với dung lượng KV truyền đi ≤ 50% so với Mode 1.
  2. Cơ chế dừng sớm giảm thời gian chạy của biến thể ẩn ≥ 40% mà không làm giảm độ chính xác.
  3. Tỷ lệ phản hồi rác < 0.5% trên các mô hình chắt lọc ở bất kỳ số bước nào.
