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

## 2. Các Phát hiện Chính cho đến Nay

> *(Phần này sẽ được cập nhật lại sau khi có đủ kết quả benchmark.)*  
> Xem kết quả thực nghiệm hiện tại tại [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

## 3. Các Hướng Đề xuất


### 3.1 Định tuyến KV Chọn lọc Kép (Dual-Selective KV Routing)
Định tuyến KV **theo từng đoạn token tại mỗi layer**, thay vì cắt toàn bộ cache
của một layer. Bản v1 đã implement chạy 5 mẫu calibration với full KV và greedy,
sau đó tính hai điểm độc lập:

- `ContextScore[l]`: attention mass trung bình từ query của B tới các token input
  ban đầu của A tại layer `l` (không tính attention-sink).
- `LatentScore[l]`: attention mass trung bình từ query của B tới các latent token
  của A tại layer `l`.

Mỗi bảng xếp hạng giữ `floor(0.7 * L)` layer và cho phép hai tập chồng nhau. Vì
vậy mỗi layer có đúng một trong bốn trạng thái thật: context+latent,
context-only, sink+latent, hoặc sink-only. Attention-sink gốc luôn được giữ.
Key đã cache của A giữ nguyên RoPE; vị trí của B tiếp tục từ độ dài logic đầy đủ
`T_A + N`, còn causal mask dùng độ dài cache vật lý riêng của từng layer.

Đây là Mode 5 (`--segmented_kv_select`). Mode 4 (`--dual_kv_select`) được giữ làm
baseline cũ: nó hợp hai danh sách layer nông/sâu rồi truyền full context+latent KV
ở mọi layer được giữ. V1 chỉ hỗ trợ batch size 1 và A/B có cùng kiến trúc Qwen3
full-attention (ví dụ Qwen3-4B).

Mốc 70% là budget số layer cho từng segment, không phải giảm 70% số byte. Nó giữ
xấp xỉ 70% số vị trí KV của full cache (cộng overhead của sink); tối ưu trực tiếp
theo byte budget được để lại cho hướng phát triển tiếp theo.

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
  1. Segmented Dual-KV cải thiện biên Pareto chất lượng/chi phí so với Mode 2 và Mode 4 cũ ở cấu hình cố định 70% mỗi segment. Phiên bản byte-budget sau đó hướng tới dung lượng KV truyền đi ≤ 50% Mode 1.
  2. Cơ chế dừng sớm giảm thời gian chạy của biến thể ẩn ≥ 40% mà không làm giảm độ chính xác.
  3. Tỷ lệ phản hồi rác < 0.5% trên các mô hình chắt lọc ở bất kỳ số bước nào.
