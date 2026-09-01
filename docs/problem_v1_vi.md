# Phát biểu Vấn đề (v1): Các Phát hiện Hiện tại & Vấn đề Cần Giải quyết

> **Trạng thái**: Bản thảo v1 — 15-08-2026
> **Tài liệu đồng hành**: [proposal_v1.md](proposal_v1.md) (hoặc [proposal_v1_vi.md](proposal_v1_vi.md)) — mỗi vấn đề dưới đây tương ứng với một hướng đề xuất ở tài liệu đó.
> **Cơ sở thực nghiệm**: `EXPERIMENT_RESULTS.md` (cập nhật 11-08-2026); `Qwen/Qwen3-4B`, `suayptalha/DeepSeek-R1-Distill-Llama-3B`; HotpotQA (500), MedQA (300), TMATH (300), MultiFieldQA-EN (150).

---

## 1. Các Phát hiện Hiện tại

### F1 — Tư duy ẩn hỗ trợ các nhiệm vụ suy luận
LatentMAS (Model A thực hiện N bước suy nghĩ ẩn trước khi truyền KV) vượt trội so với KVComm thuần trên các nhiệm vụ yêu cầu suy luận đa bước (multi-hop) hoặc suy luận toán học:

| Nhiệm vụ | KVComm thuần | LatentMAS tốt nhất | Mức tăng |
|---|---:|---:|---:|
| HotpotQA (Qwen3-4B) | 70.00% | **72.80%** (Mode 1, N=1–2) | +2.80 |
| TMATH (DeepSeek-R1-Distill) | ~31.0% | **36.00%** (Mode 2, N=1) | +5.00 |
| TMATH (Qwen3-4B) | 31.36% | **34.12%** (Mode 1, N=10) | +2.76 |
| MedQA (Qwen3-4B) | 58.33% | **59.00%** (Mode 1, N=5) | +0.67 |

### F2 — Tư duy ẩn gây hại cho việc trích xuất trong ngữ cảnh dài
Trên tác vụ MultiFieldQA-EN (truy xuất thông tin thực tế nguyên văn từ tài liệu dài), KVComm thuần dẫn đầu: **50.00%** so với 47.33% của cấu hình LatentMAS tốt nhất. Việc ép A vào trạng thái suy luận kiểu `<think>` làm biến dạng biểu diễn ngữ cảnh trực tiếp mà B cần để trích xuất thực thể.

### F3 — Số bước ẩn ít hơn cho kết quả tốt hơn; các vòng lặp dài bị suy thoái
Độ chính xác đạt đỉnh ở N=1–5 trên mọi nhiệm vụ. Vượt quá N≈10, độ chính xác giảm dần đơn điệu và câu trả lời rác (garbage response) tăng lên — trên DeepSeek-R1-Distill/TMATH: 0.3% rác ở N=1 → 3.7% ở N=10 → 6.0% ở N=25, cùng với độ chính xác giảm từ 33.0% xuống 29.0%. Các mô hình chắt lọc (distilled models) là nhạy cảm nhất.

### F4 — Tỉa bớt lớp đồng nhất xung đột với truyền suy nghĩ ẩn
Khi bật tư duy ẩn, Mode 1 (tất cả các lớp) vượt trội so với Mode 2 (chọn top 70% các lớp quan trọng) khoảng ~3–5 điểm trên HotpotQA (72.80% so với 70.00%) và MultiFieldQA-EN (47.33% so với 42.67%). Việc cắt giảm 30% số lớp một cách đồng nhất trên toàn bộ chuỗi sẽ loại bỏ thông tin quan trọng đối với đầu vào dài. (Ngoại lệ: TMATH trên DeepSeek-R1-Distill, nơi Mode 2 chiến thắng — mô hình này phụ thuộc vào nhiệm vụ/mô hình).

### F5 — Các biến thể tư duy ẩn phải trả chi phí độ trễ cao
KVComm thuần trên MedQA: 5 phút; các biến thể tư duy ẩn: 40–60 phút (chậm hơn 8–10 lần). Nguyên nhân: N lượt lan truyền tiến (forward pass) bổ sung qua A với KV cache tăng dần (O(N·T_A)), và B phải thực hiện attention trên T_A + N + T_B token thay vì T_B.

### F6 — Không có hiệu quả nếu không có bất cân xứng thông tin
MedQA chỉ cho thấy mức tăng dư địa +0.67: A và B đều nhìn thấy thông tin giống nhau, do đó kênh truyền thông không đóng góp được thêm nhiều.

---

## 2. Các Vấn đề Cần Giải quyết

### P1 — Chọn lớp đồng nhất bỏ qua vai trò không đồng nhất của token
**Bằng chứng**: F2, F4.
**Vấn đề**: `layer_importance.py` tính toán một thứ tự xếp hạng lớp tĩnh áp dụng đồng nhất cho toàn bộ chuỗi được truyền. Tuy nhiên, các token ngữ cảnh (T_A) và token suy nghĩ ẩn (N) mang các loại thông tin khác nhau — thông tin thực tế nguyên văn nằm ở các lớp nông-đến-trung-bình, trong khi trừu tượng hóa suy luận nằm ở các lớp trung-bình-đến-sâu. Một xếp hạng đơn lẻ không thể đáp ứng cả hai, dẫn đến việc chọn lọc vừa làm mất độ trung thực truy xuất (MultiFieldQA-EN) vừa truyền KV dư thừa.
**Tương ứng với**: Đề xuất §3.1 Định tuyến KV chọn lọc kép (Dual-Selective KV Routing).

### P2 — Lặp tư duy ẩn bị lệch biểu diễn và mất ổn định
**Bằng chứng**: F3.
**Vấn đề**: vòng lặp tư duy ẩn trong `models_latent.py` liên tục chiếu trạng thái ẩn cuối cùng trở lại không gian embedding thông qua ma trận căn chỉnh W. Sai số tích tụ qua các vòng lặp, gây ra độ lệch biểu diễn (representation drift), suy giảm độ chính xác ở N cao và tạo ra phản hồi rác — nghiêm trọng nhất trên các mô hình chắt lọc (DeepSeek-R1-Distill).
**Tương ứng với**: Đề xuất §3.3 Căn chỉnh thặng dư điểm neo (Anchor Residual Realignment).

### P3 — Thiếu tiêu chuẩn dừng cho các bước ẩn
**Bằng chứng**: F3, F5.
**Vấn đề**: `--latent_steps` là một siêu tham số cố định. Không có tín hiệu dừng lặp khi trạng thái ẩn đã hội tụ, dẫn đến việc các lượt chạy bị lãng phí tài nguyên tính toán (N quá cao, cộng với sự suy thoái từ P2) hoặc suy nghĩ chưa đủ (N quá thấp). Giá trị N tối ưu cũng thay đổi theo nhiệm vụ và mô hình, làm cho việc tinh chỉnh thủ công trở nên tốn kém.
**Tương ứng với**: Đề xuất §3.2 Bước tư duy ẩn thích ứng & Dừng sớm (Adaptive Latent Steps & Early Exit).

### P4 — Một chế độ truyền thông cố định không tối ưu trên các loại nhiệm vụ khác nhau
**Bằng chứng**: F1 so với F2.
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
5. **Yêu cầu về độ bất cân xứng**: khung làm việc có thể phát hiện độ bất cân xứng thông tin thấp (F6) ngay từ đầu để bỏ qua hoàn toàn truyền thông không?

---

## 4. Kết quả Đánh giá Mã nguồn (Code-Review Findings - 15-08-2026)

> Đánh giá mã nguồn đối kháng đa agent (Multi-agent adversarial code review) trên `models_latent.py`, `models.py`, `eval_latent.py`, `com_latent.py`, tập trung vào sự suy thoái N-bước (F3). Tất cả các phát hiện dưới đây đã được xác minh trên bản cài đặt `transformers==4.53.3`.
> **Tóm tắt chính**: không có lỗi đơn lẻ nào gây ra sự suy thoái — sự trôi lệch OOD tích tụ do đưa trạng thái ẩn ngược trở lại làm embedding là bản chất của LatentMAS — nhưng có 4 lỗi thực tế làm khuếch đại hoặc khởi phát nó, và 2 "sửa lỗi" được ghi chép lại thực chất không có tác dụng (no-op).

### 4.1 Các lỗi thúc đẩy sự suy thoái N-bước

| # | Lỗi | Vị trí | Tác động |
|---|---|---|---|
| B1 | **Khả năng suy luận của Bên nhận B bị triệt tiêu**: Lượt assistant của B được thêm sẵn `</think>\n\nThe answer is: ` (qua `eval.apply_chat_template(context=False)`), do đó B không thể suy nghĩ và phụ thuộc hoàn toàn vào cache ẩn của A. Ở N cao, nơi các key ẩn ngày càng lệch phân phối (OOD), B không có kênh phục hồi → câu trả lời rác tăng theo N (khớp từ 0.3% → 6.0% khi N=1 tới N=25). | `eval_latent.py:188` | Cao — trực tiếp khuếch đại F3 |
| B2 | **Trùng lặp `<think>` trên R1-Distill**: `<think>` được thêm vô điều kiện, nhưng chat template của R1-Distill đã phát ra nó → đầu vào của A kết thúc bằng `<think>\n<think>`, một trạng thái chưa từng gặp. Trạng thái `last_hidden` đầu tiên bị hỏng chính là điểm neo mà tất cả N bước ẩn lặp từ đó. Ngoài ra: kiểm tra None của `convert_tokens_to_ids` không đáng tin cậy (fast tokenizer trả về `unk_token_id`, không phải None). | `eval_latent.py:174-182` | Cao đối với các mô hình chắt lọc — giải thích tại sao DeepSeek-R1-Distill dễ sinh câu trả lời rác nhất |
| B3 | **Padding không được mask trong mỗi bước ẩn**: vòng lặp dựng lại mask dưới dạng toàn số 1 trên `past_len+1` thay vì `cat([attention_mask, ones])`, làm lộ pad-token KV; sự hư hỏng tích tụ theo từng bước. Chỉ xảy ra khi batch > 1 (chưa ảnh hưởng đánh giá từng mẫu hiện tại). | `models_latent.py:307` | Ẩn ở hiện tại — sẽ âm thầm làm hỏng bất kỳ lượt chạy batch nào trong tương lai |
| B4 | **Mode 2 + `shift_back=False`: vi phạm tính nhân quả.** Các lớp không được chọn giữ lại 1 token "sink"; B dựng một causal mask ở chiều rộng đầy đủ của layer-0 (T_A+N+T_B); sdpa cắt nhỏ nó theo từng lớp và 1+T_B cột đầu tiên đều không được mask → token prompt của B chú ý (attend) đến các token *tương lai* của chính chúng trong mọi lớp không được chọn. `attention_mask` không bao giờ đến được B (bị bỏ trong `CVCommunicator.forward`, `models.py:170-174`), nên "past_mask fix" tại `eval_latent.py:234` là mã chết (dead code). Rất có thể giải thích một phần khoảng cách Mode 2 < Mode 1 nếu các lượt chạy dùng mặc định cấu hình `shift_back=False`. | `models.py:129` | Cao đối với kết quả Mode 2 — cần xác minh lại với `shift_back=True` |

### 4.2 Các giải pháp đã ghi chép nhưng không có tác dụng (No-op)

| # | Khẳng định trong mã gốc | Thực tế (đã xác minh trên transformers 4.53.3) |
|---|---|---|
| N1 | `new_cache._seen_tokens = 0` gán lại vị trí ẩn 0..N-1 cho B (`models_latent.py:220`) | Không tác dụng: `DynamicCache.get_seq_length()` dựa trên shape và bỏ qua `_seen_tokens`. Trong chế độ `latent_only`, độ lệch RoPE vẫn giữ ~T_A, với key của A xuất hiện trong *tương lai* của B. |
| N2 | "Để sửa hoàn toàn (lệch bằng 0), hãy đặt `shift_back=True`" (`models_latent.py:193`) | `shift_back` không bao giờ xoay lại các key đã lưu vào cache của A; ở Mode 1 đường đi của nó hoàn toàn giống với tính toán vị trí tiêu chuẩn của HF. "Độ lệch bằng 0" là không thể đạt được. |
| N3 | Cờ `--latent_only` điều khiển các thử nghiệm cắt giảm (ablation) Mode 1 | Bị âm thầm bỏ qua: Mode 1 khởi tạo evaluator mà không có `latent_only=cfg.latent_only` (`com_latent.py:306`), do đó các lượt chạy Mode 1 "latent_only" thực chất truyền toàn bộ cache trong khi log ghi khác đi. Ngoài ra, cờ `--no_latent_kv_select` được ghi chép lại không tồn tại (argparse chỉ tạo dạng `--no_` cho các bool có mặc định là True). |

### 4.3 Trường hợp đã bị bác bỏ

Khẳng định cho rằng bước prefill của Mode 1 bị lệch nghiêm trọng (`cache_position` bắt đầu từ 0) đã bị **bác bỏ**: `_supports_cache_class=False` có nghĩa là quá trình generation không chèn `cache_position`, và mô hình của B tính lại nó từ độ dài cache dựa trên shape — việc truyền full-cache trong Mode 1 thuần túy nhất quán về mặt vị trí, đó là lý do tại sao nó vẫn hoạt động.

### 4.4 Thứ tự ưu tiên sửa lỗi (tác động dự kiến lên đường cong suy thoái N-bước)

1. Loại bỏ/đặt điều kiện cho tiền chèn `</think>` cho B (B1) — bài kiểm tra tốn ít chi phí nhất để xem liệu phản hồi rác ở N cao có biến mất không.
2. Bảo vệ việc thêm `<think>` đối với các template đã phát ra nó (B2).
3. Chạy lại Mode 2 với `shift_back=True` và ghi lại thiết lập nào mà mỗi thử nghiệm đã sử dụng (B4) — không còn log snapshot nào tồn tại trong repo để kiểm tra hồi truy.
4. Sửa padding mask của vòng lặp ẩn trước khi thực hiện bất kỳ thử nghiệm chạy theo batch nào (B3).
5. Xóa hoặc viết lại đường đi `latent_only` / `_seen_tokens` (N1, N3) — vừa bị hỏng *vừa* bị âm thầm bỏ qua, điều này làm mất hiệu lực của nó như một thử nghiệm ablation.

**Hệ quả đối với các phát biểu vấn đề ở trên**: P2 (lệch biểu diễn ẩn) một phần là *hệ quả do đo lường* (measurement artifact) — B1/B2 làm thổi phồng độ lệch nhìn thấy; đường cong suy thoái nội tại thực sự vẫn chưa biết cho đến khi chúng được sửa. P5 (sai lệch RoPE) tồi tệ hơn những gì được ghi chép lại: cả hai biện pháp giảm thiểu được tuyên bố (N1, N2) đều không hiệu quả, và khoảng cách Mode 1 > Mode 2 của F4 có thể giải thích một phần bởi B4 thay vì do mất mát thông tin từ việc tỉa bớt lớp.
