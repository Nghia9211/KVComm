# Phát biểu Vấn đề (v1): Các Phát hiện Hiện tại & Vấn đề Cần Giải quyết

> **Trạng thái**: Bản thảo v1 — 15-08-2026 (cập nhật kết quả 09-09-2026)  
> **Tài liệu đồng hành**: [proposal_v1.md](proposal_v1.md) (hoặc [proposal_v1_vi.md](proposal_v1_vi.md)) — mỗi vấn đề dưới đây tương ứng với một hướng đề xuất ở tài liệu đó.  
> **Kết quả thực nghiệm**: Xem [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

---

## 1. Các Phát hiện Hiện tại

> *(Phần này sẽ được cập nhật lại sau khi có đủ kết quả benchmark.)*

---

## 2. Các Vấn đề Cần Giải quyết

### P1 — Chọn lớp đồng nhất bỏ qua vai trò không đồng nhất của token
**Vấn đề**: `layer_importance.py` tính toán một thứ tự xếp hạng lớp tĩnh áp dụng đồng nhất cho toàn bộ chuỗi được truyền. Tuy nhiên, các token ngữ cảnh (T_A) và token suy nghĩ ẩn (N) mang các loại thông tin khác nhau — thông tin thực tế nguyên văn nằm ở các lớp nông-đến-trung-bình, trong khi trừu tượng hóa suy luận nằm ở các lớp trung-bình-đến-sâu. Một xếp hạng đơn lẻ không thể đáp ứng cả hai, dẫn đến việc chọn lọc vừa làm mất độ trung thực truy xuất vừa truyền KV dư thừa.  
**Tương ứng với**: Đề xuất §3.1 Định tuyến KV chọn lọc kép (Dual-Selective KV Routing).

### P2 — Lặp tư duy ẩn bị lệch biểu diễn và mất ổn định
**Vấn đề**: vòng lặp tư duy ẩn trong `models_latent.py` liên tục chiếu trạng thái ẩn cuối cùng trở lại không gian embedding thông qua ma trận căn chỉnh W. Sai số tích tụ qua các vòng lặp, gây ra độ lệch biểu diễn (representation drift), suy giảm độ chính xác ở N cao và tạo ra phản hồi rác — nghiêm trọng nhất trên các mô hình chắt lọc (DeepSeek-R1-Distill).  
**Tương ứng với**: Đề xuất §3.3 Căn chỉnh thặng dư điểm neo (Anchor Residual Realignment).

### P3 — Thiếu tiêu chuẩn dừng cho các bước ẩn
**Vấn đề**: `--latent_steps` là một siêu tham số cố định. Không có tín hiệu dừng lặp khi trạng thái ẩn đã hội tụ, dẫn đến việc các lượt chạy bị lãng phí tài nguyên tính toán (N quá cao, cộng với sự suy thoái từ P2) hoặc suy nghĩ chưa đủ (N quá thấp). Giá trị N tối ưu cũng thay đổi theo nhiệm vụ và mô hình, làm cho việc tinh chỉnh thủ công trở nên tốn kém.  
**Tương ứng với**: Đề xuất §3.2 Bước tư duy ẩn thích ứng & Dừng sớm (Adaptive Latent Steps & Early Exit).

### P4 — Một chế độ truyền thông cố định không tối ưu trên các loại nhiệm vụ khác nhau
**Vấn đề**: mô hình thắng/thua mang tính hệ thống — tư duy ẩn thắng ở các nhiệm vụ suy luận, KVComm thuần thắng ở các nhiệm vụ trích xuất — nhưng khung làm việc không có cơ chế phát hiện loại nhiệm vụ để định tuyến phù hợp. Bất kỳ một cấu hình mặc định đơn lẻ nào cũng sẽ bỏ lỡ cơ hội tối ưu độ chính xác ở đâu đó.  
**Tương ứng với**: Đề xuất §3.4 Định tuyến chế độ theo nhiệm vụ (Task-Aware Mode Routing).

### P5 — Sai lệch vị trí RoPE trong truyền KV chỉ mới được giải quyết một phần
**Bằng chứng**: cấp độ mã nguồn (code-level).  
**Vấn đề**: các phần tử KV được truyền mang mã hóa vị trí quay (rotary position encodings - RoPE) từ hệ tọa độ của A. Giải pháp tạm thời `--shift_back` (`models.py:forward_shift_back_llama` / `forward_shift_back_qwen2`) sửa các lớp chỉ dùng attention-sink và luôn phải bật ở chế độ latent, nhưng nó phụ thuộc vào dòng mô hình (báo lỗi `NotImplementedError` ở nơi khác) và chế độ `--latent_only` (chỉ truyền N token KV ẩn, bỏ T_A — biến thể hiệu quả về băng thông nhất) vẫn chưa thể sử dụng do sai lệch RoPE chưa giải quyết. Việc tái ánh xạ vị trí nguyên tắc sẽ mở ra khả năng truyền dữ liệu nhỏ hơn nhiều.  
**Tương ứng với**: điều kiện tiên quyết cho Đề xuất §3.1 (định tuyến theo loại token làm thay đổi cấu trúc chuỗi hơn nữa).

### P6 — Tính dễ tổn thương của Prompt/template cho các mô hình suy nghĩ (thứ yếu)
**Bằng chứng**: các sửa lỗi được ghi nhãn "vấn đề 1/3/5" trong `eval_latent.py`.  
**Vấn đề**: khả năng nhận biết suy nghĩ ẩn của phía gửi/nhận phụ thuộc vào việc xử lý thủ công chat-template (giữ `<think>` cho A, thêm thông báo ngữ cảnh ẩn cho B). Điều này dễ hỏng giữa các tokenizer khác nhau (ví dụ: `<think>` là chuỗi multi-token trên các mô hình chắt lọc dựa trên Llama) và nên được hợp nhất vào một lớp prompt mạnh mẽ, độc lập với mô hình.

---

## 3. Các Câu hỏi Ngỏ

1. **Quy mô (Scale)**: lợi ích truyền thông ẩn tăng hay giảm khi kích thước mô hình tăng (4B → 8B → 70B)?
2. **Cặp mô hình không đồng nhất**: việc truyền KV/latent có hoạt động khi A ≠ B (khác kích thước hoặc khác dòng mô hình), khi số lớp và cấu trúc KV không khớp nhau không?
3. **Ranh giới Pareto**: đường cong đánh đổi giữa độ chính xác và kích thước KV truyền đi là gì, và định tuyến chọn lọc kép nằm ở đâu trên đường cong đó?
4. **Tín hiệu hội tụ**: độ tương đồng cosine của các trạng thái ẩn liên tiếp có phải là tiêu chí dừng sớm đáng tin cậy trên các nhiệm vụ không, hay nó dừng quá sớm ở các bài toán khó?
5. **Yêu cầu về độ bất cân xứng**: khung làm việc có thể phát hiện độ bất cân xứng thông tin thấp ngay từ đầu để bỏ qua hoàn toàn truyền thông không?

---

## 4. Kết quả Đánh giá Mã nguồn (Code-Review Findings - 15-08-2026)

> Đánh giá mã nguồn đối kháng đa agent (Multi-agent adversarial code review) trên `models_latent.py`, `models.py`, `eval_latent.py`, `com_latent.py`, tập trung vào sự suy thoái N-bước. Tất cả các phát hiện dưới đây đã được xác minh trên bản cài đặt `transformers==4.53.3`.  
> **Tóm tắt chính**: không có lỗi đơn lẻ nào gây ra sự suy thoái — sự trôi lệch OOD tích tụ do đưa trạng thái ẩn ngược trở lại làm embedding là bản chất của LatentMAS — nhưng có 4 lỗi thực tế làm khuếch đại hoặc khởi phát nó, và 2 "sửa lỗi" được ghi chép lại thực chất không có tác dụng (no-op).

### 4.1 Các lỗi thúc đẩy sự suy thoái N-bước

| # | Lỗi | Vị trí | Tác động |
|---|---|---|---|
| B1 | **Khả năng suy luận của Bên nhận B bị triệt tiêu**: Lượt assistant của B được thêm sẵn `</think>\n\nThe answer is: ` (qua `eval.apply_chat_template(context=False)`), do đó B không thể suy nghĩ và phụ thuộc hoàn toàn vào cache ẩn của A. Ở N cao, nơi các key ẩn ngày càng lệch phân phối (OOD), B không có kênh phục hồi → câu trả lời rác tăng theo N. | `eval_latent.py:188` | Cao — trực tiếp khuếch đại suy thoái N-bước |
| B2 | **Trùng lặp `<think>` trên R1-Distill**: `<think>` được thêm vô điều kiện, nhưng chat template của R1-Distill đã phát ra nó → đầu vào của A kết thúc bằng `<think>\n<think>`. | `eval_latent.py:174-182` | Cao đối với các mô hình chắt lọc |
| B3 | **Padding không được mask trong mỗi bước ẩn**: vòng lặp dựng lại mask dưới dạng toàn số 1 trên `past_len+1` thay vì `cat([attention_mask, ones])`, làm lộ pad-token KV. Chỉ xảy ra khi batch > 1. | `models_latent.py:307` | Ẩn ở hiện tại — ảnh hưởng batch runs |
| B4 | **Mode 2 + `shift_back=False`: vi phạm tính nhân quả.** Token prompt của B chú ý (attend) đến các token *tương lai* của chính chúng trong mọi lớp không được chọn. `attention_mask` không bao giờ đến được B (`models.py:170-174`). | `models.py:129` | Cao đối với kết quả Mode 2 |

### 4.2 Các giải pháp đã ghi chép nhưng không có tác dụng (No-op)

| # | Khẳng định trong mã gốc | Thực tế (đã xác minh trên transformers 4.53.3) |
|---|---|---|
| N1 | `new_cache._seen_tokens = 0` gán lại vị trí ẩn 0..N-1 cho B | Không tác dụng: `DynamicCache.get_seq_length()` dựa trên shape và bỏ qua `_seen_tokens`. |
| N2 | "Để sửa hoàn toàn (lệch bằng 0), hãy đặt `shift_back=True`" | `shift_back` không bao giờ xoay lại các key đã lưu vào cache của A. |
| N3 | Cờ `--latent_only` điều khiển các thử nghiệm ablation Mode 1 | Bị âm thầm bỏ qua: Mode 1 khởi tạo evaluator mà không có `latent_only=cfg.latent_only`. |

### 4.3 Trường hợp đã bị bác bỏ

Khẳng định cho rằng bước prefill của Mode 1 bị lệch nghiêm trọng (`cache_position` bắt đầu từ 0) đã bị **bác bỏ**: `_supports_cache_class=False` có nghĩa là quá trình generation không chèn `cache_position`, và mô hình của B tính lại nó từ độ dài cache dựa trên shape — việc truyền full-cache trong Mode 1 thuần túy nhất quán về mặt vị trí, đó là lý do tại sao nó vẫn hoạt động.

### 4.4 Thứ tự ưu tiên sửa lỗi (tác động dự kiến lên đường cong suy thoái N-bước)

1. Loại bỏ/đặt điều kiện cho tiền chèn `</think>` cho B (B1) — bài kiểm tra tốn ít chi phí nhất để xem liệu phản hồi rác ở N cao có biến mất không.
2. Bảo vệ việc thêm `<think>` đối với các template đã phát ra nó (B2).
3. Chạy lại Mode 2 với `shift_back=True` và ghi lại thiết lập nào mà mỗi thử nghiệm đã sử dụng (B4) — không còn log snapshot nào tồn tại trong repo để kiểm tra hồi truy.
4. Sửa padding mask của vòng lặp ẩn trước khi thực hiện bất kỳ thử nghiệm chạy theo batch nào (B3).
5. Xóa hoặc viết lại đường đi `latent_only` / `_seen_tokens` (N1, N3).

**Trạng thái implement (2026-09-09):** đường đi và cờ CLI `latent_only` toàn cục
đã được xóa. Mode 5 định tuyến rõ từng segment context/latent tại mỗi layer,
đồng thời giữ attention-sink gốc và vị trí RoPE logic.
