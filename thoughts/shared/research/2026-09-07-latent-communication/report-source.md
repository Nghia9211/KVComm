# LatentMAS + KVComm: lợi thế nào thực sự có thể chứng minh?

Ngày 07/09/2026 · Dành cho người phát triển KVComm · Nghiên cứu tài liệu gốc và audit tĩnh code hiện tại.

## Kết luận điều hành

**Chưa có bằng chứng đủ để kết luận LatentMAS + KVComm thắng cả LatentMAS lẫn TextMAS trên các task hiện tại.** Hướng có cơ sở nhất là tìm cấu hình giữ chất lượng gần full-KV với chi phí thấp hơn. Thắng accuracy là khả năng cần kiểm chứng, không phải hệ quả tự động của việc ghép hai phương pháp.

Ba nhận định cần điều chỉnh: HotpotQA không mặc nhiên là QA đơn giản; reasoning không đảm bảo latent thắng text; ít bước sinh không đồng nghĩa ít byte truyền, ít VRAM hoặc nhanh hơn trên server thực tế.

Phân biệt trong báo cáo: “quan sát code” là kết quả đọc triển khai; “bằng chứng paper” chỉ đúng trong thiết lập đã công bố; “giả thuyết” là dự đoán cần thực nghiệm. Chưa chạy benchmark mới hoặc xác nhận số liệu trên server 4×RTX 3080. Tất cả đề xuất dưới đây chưa được implement.

## 1. HotpotQA: TextMAS có lợi thế không?

HotpotQA được thiết kế cho multi-hop QA, không phải tập câu hỏi đơn giản theo định nghĩa. Bản distractor chính thức yêu cầu đọc 10 đoạn và dự đoán cả đáp án lẫn supporting facts. [HotpotQA — Yang và cộng sự, EMNLP 2018](https://hotpotqa.github.io/).

**Nhưng task trong repo đã được đơn giản hóa ở khâu tìm bằng chứng.** Hàm construct_support chỉ lấy đúng các câu được gold supporting_facts chỉ ra. A nhận những câu này, B nhận câu hỏi. Đây là điều kiện oracle evidence: biết trước bằng chứng đúng, không phải toàn bộ bài toán retrieval + reasoning. Nó không đưa trực tiếp đáp án gold vào prompt, nhưng dùng annotation gold để chọn đầu vào. MuSiQue cũng chỉ lấy đoạn is_supporting. Nguồn: [hotpotqa.py](../../../../dataloader/hotpotqa.py), [musique.py](../../../../dataloader/musique.py).

Suy luận cho thiết lập này:

- TextMAS có thể rất cạnh tranh: bằng chứng đã ngắn và sạch; A có thể chuyển vài tên, ngày tháng và quan hệ chính xác. Ngôn ngữ cũng giúp kiểm tra được chỗ A bỏ sót hoặc suy luận sai.
- Full-KV vẫn có thể tốt hơn nếu bản tóm tắt text bỏ mất thực thể trung gian, phủ định hoặc quan hệ so sánh. Không được suy từ độ ngắn của câu trả lời sang độ dễ của quá trình suy luận.
- Selective-KV có thể giảm chi phí ở B, nhưng khi context vốn ngắn thì phần tiết kiệm tuyệt đối nhỏ; chi phí chọn lớp và latent steps có thể lấn át.
- Chưa có cơ sở nói accuracy của latent bắt buộc giảm. Cũng chưa có cơ sở bảo đảm latent nhanh hơn TextMAS chỉ sinh vài chục token hữu ích.

Nên gọi kết quả hiện tại là **HotpotQA trên oracle supporting sentences, chấm answer F1**. Không so trực tiếp với leaderboard distractor/fullwiki. Nên phân tích riêng câu bridge, comparison, yes/no và độ dài evidence; giữ cùng sample ID giữa các phương pháp. Loader đang bỏ tiêu đề nguồn khi ghép câu, có thể làm mất thông tin chủ thể; thêm tiêu đề phải áp dụng đồng đều và ghi phiên bản dữ liệu mới.

## 2. Reasoning: lợi thế của latent có rõ ràng hơn không?

Bằng chứng quyết định từ Table 1, thiết lập sequential:

| GSM8K | TextMAS accuracy | LatentMAS accuracy |
| --- | --- | --- |
| Qwen3-4B | 89,8% | 88,2% |
| Qwen3-8B | 92,3% | 93,8% |

Cùng task reasoning nhưng thứ hạng đảo theo model. Nguồn: [Latent Collaboration in Multi-Agent Systems — Zou và cộng sự, v4, 03/08/2026](https://arxiv.org/html/2511.20639v4).

Một nguồn độc lập cũng chống lại suy luận “reasoning ⇒ latent thắng”: Coconut hơn CoT trên một số bài logic tổng hợp nhưng chưa vượt CoT trên GSM8K; phương pháp này có huấn luyện curriculum, không tương đương hệ hai agent training-free hiện tại. [Coconut — Hao và cộng sự, bản v3](https://arxiv.org/html/2412.06769v3).

**Giả thuyết hợp lý hơn:** latent có cơ hội hữu ích khi A cần nhiều bước xử lý và text trung gian dài, trong khi hidden states chuyển được đủ thông tin có ích cho B. Tuy nhiên:

- Một kế hoạch bằng text có thể là cấu trúc kiểm tra rất tốt cho phép tính, logic tuần tự hoặc ràng buộc code.
- Nếu A và B đã nhận cùng đề, A không cung cấp dữ kiện mới; phải chứng minh A bổ sung tính toán hữu ích, không chỉ lặp lại biểu diễn của đề.
- Tăng latent steps làm tăng tính toán, không bảo đảm tăng chất lượng. Drift biểu diễn hoặc latent steps không hữu ích là giả thuyết cần đo, không phải lời giải thích sẵn có cho mọi lần giảm điểm.
- Chọn lớp có thể loại đúng phần B cần để dùng thông tin A đã xử lý. Do đó combined method có thể thua full-KV ngay trên reasoning.

Cần phân biệt **latent reasoning** — A sinh hidden states thay cho token — với **latent communication** — B nhận KV thay cho text. Nếu chỉ so TextMAS và full LatentMAS thì cả hai trục đổi cùng lúc; chưa cô lập nguyên nhân.

Paper dùng chuỗi bốn agent ở thiết lập sequential, khác pipeline hai agent hiện tại; số đo tốc độ trên 8×A100 80GB không thể chuyển nguyên sang 4×RTX 3080. Lập luận bảo toàn thông tin cũng không bảo đảm đáp án đúng, chuyển cache khác model tùy ý hoặc chọn bỏ lớp không mất thông tin. Nguồn: [LatentMAS v4](https://arxiv.org/html/2511.20639v4).

## 3. Lợi thế dự kiến theo toàn bộ nhóm task hiện tại

Bảng dưới đây là định hướng thực nghiệm dựa trên giao thức đầu vào của repo, không phải dự báo điểm số. Nguồn triển khai: [dataloader](../../../../dataloader/), [prompts_latent.py](../../../../prompts_latent.py).

| Nhóm task | Cơ hội cho LatentMAS + KVComm | Đối chứng hoặc rủi ro quyết định |
| --- | --- | --- |
| hotpotqa, musique | Truyền quan hệ nối giữa bằng chứng; kiểm tra mất thông tin khi chọn lớp | Oracle evidence khiến text ngắn trở thành baseline mạnh; không đại diện retrieval |
| multifieldqa_en, qasper, twowikimqa | Tránh nút thắt tóm tắt text; giảm phần cache B phải sử dụng so với full-KV | Context dài tăng KV, prefill và nguy cơ OOM; không mặc định nhanh hơn text |
| countries, tipsheets | Sanity check việc B có sử dụng thông tin A hay không | Ít nhu cầu tính toán bổ sung; nhiều latent steps có thể chỉ tăng chi phí |
| gsm8k, aime2024, aime2025 | Kiểm tra giá trị của tính toán thêm ở A | Cần metric đáp án cuối đúng, B-thinking đối xứng và baseline single-agent |
| arc_easy, arc_challenge, gpqa, medqa | Kiểm tra kế hoạch/tri thức trung gian giúp chọn đáp án | Khả năng trả lời của B và thông tin choices ở A có thể chi phối; ARC-E có thể có trần điểm |
| tmath | Chuyển hint có thể hỗ trợ lời giải | Hiện chấm ROUGE-L recall, chưa phải math accuracy; A chỉ thấy hint |
| repobench | Giữ tên biến, hàm và quan hệ cross-file | Metric là edit similarity cho dòng tiếp theo, không phải chương trình chạy đúng |
| mbppplus, humanevalplus | Truyền kế hoạch và ràng buộc trước khi B viết code | Phải kiểm tra execution-based correctness và xử lý output; không lấy F1 thay pass rate |
| samsum | Chuyển thông tin từ nửa hội thoại A sang B | Tóm tắt một câu có thể thiếu nội dung; ROUGE-L recall bị ảnh hưởng độ dài |

QASPER trong repo được nạp từ tau/scrolls, không phải cứ đổi metric sang LongBench F1 là trở thành bản benchmark LongBench chính thức. Tương tự, cùng tên task không đủ để bảo đảm cùng dữ liệu, split, preprocessing và answer aliases. Nguồn: [qasper.py](../../../../dataloader/qasper.py).

**Ưu tiên nghiên cứu:** HotpotQA-oracle để debug giao tiếp; GSM8K để kiểm tra reasoning; MedQA để kiểm tra choices/knowledge sau khi chốt đầu vào; một task code để kiểm tra tính đúng chức năng; MultiFieldQA để stress-test dài sau khi giải quyết lỗi bộ nhớ. Không chỉ chọn các task mà phương pháp mới thắng.

## 4. Prompt hiện tại: hợp lý về cấu trúc, chưa chứng minh tối ưu

### Những điểm đã tốt

Code đã tập trung prompt vào một bộ dựng chung; phân biệt evidence extraction, shared problem và native split. Với QA, A biết câu hỏi và được yêu cầu giữ đúng tên, số và quan hệ; B chỉ trả đáp án. Đây là nền tảng tốt để giảm khác biệt ngữ nghĩa giữa text, full-KV và selective-KV. Nguồn: [prompts_latent.py](../../../../prompts_latent.py).

Tuy nhiên, cùng semantic core chưa có nghĩa cùng token đầu vào, cùng thông tin nhận được hoặc cùng compute budget. Chat template, thinking và truncation vẫn có thể khác nhau.

### Những điểm cần kiểm chứng hoặc chỉnh trong phiên bản sau

**A. TextMAS có thể bị cắt ở thinking trước khi xuất evidence.** Text A luôn bật thinking, trong khi nhiều QA task đặt sender_max_tokens=256. Đây là rủi ro từ cấu hình, chưa phải bằng chứng mọi mẫu đều bị cắt. Cần log finish reason, output thô, token thinking và token evidence; thử một dải ngân sách như 128/256/512/1024 thay vì chỉ một điểm. Không được gọi đó là thất bại của text channel nếu baseline chưa kịp hoàn thành message. Nguồn: [eval_latent.py](../../../../eval_latent.py), [hotpotqa.py](../../../../dataloader/hotpotqa.py).

**B. MedQA có lệch thông tin giữa hai agent.** A nhận question, B nhận question + choices, dù profile có tên shared_problem. Điều này đối xứng giữa các kênh nếu cùng loader nhưng không phải shared full problem. Cần chọn rõ mục tiêu: A lập kế hoạch không nhìn options, hay A phân tích cùng bộ options với B. Không đổi âm thầm rồi so với kết quả cũ. Nguồn: [medqa.py](../../../../dataloader/medqa.py).

**C. “Không đưa final answer” có thể quá cứng cho evidence extraction.** Khi câu chứa bằng chứng đồng thời chứa tên đáp án, A không nên né tên đó. Đề xuất ý nghĩa prompt: giữ nguyên mọi dữ kiện cần thiết, kể cả chuỗi có thể là đáp án; không tự tổng hợp kết luận thiếu căn cứ. Đây là đề xuất, chưa đo hiệu quả.

**D. Question placement cần ablation.** QA đang đặt câu hỏi sau context. Với causal prefill, các token context trước đó chưa thấy câu hỏi xuất hiện sau; token cuối và latent steps thì có thể thấy cả hai. Có thể thử câu hỏi trước context hoặc ở cả hai đầu, nhưng phải giữ ngân sách và apply đồng đều. Nghiên cứu long-context cho thấy vị trí thông tin có thể ảnh hưởng kết quả, không chứng minh một thứ tự luôn tốt nhất. [Lost in the Middle — Liu và cộng sự](https://arxiv.org/abs/2307.03172).

**E. Không đồng nhất “đáp án ngắn” với “cấm suy nghĩ”.** QA core không phân nhánh theo allow_b_think, dù chat template có xử lý cờ này. Nên kiểm chứng B-thinking on/off độc lập, chỉ chấm final answer và tính cả token thinking trong chi phí. Với list, multi-part answer hoặc unanswerable, yêu cầu “shortest answer” phải giữ đủ nội dung.

**F. Native split là một bài toán khác.** TMATH A chỉ có hint, RepoBench A có cross-file context, SAMSum A có nửa đầu hội thoại. Không nên ép cùng planner prompt lên tất cả. SAMSum “one sentence” có nguy cơ thiếu ý; nên xem lại quan hệ giữa yêu cầu độ dài và metric recall.

**G. Đánh giá hai loại công bằng.** Một thí nghiệm giữ chung semantic prompt để cô lập kênh. Một thí nghiệm khác cho mỗi phương pháp tối ưu prompt trên cùng tập validation và ngân sách tuning. Hai câu hỏi khác nhau; báo cáo cả hai nếu muốn kết luận về cơ chế lẫn hệ tốt nhất.

## 5. Các vấn đề code có thể làm sai kết luận nghiên cứu

Đây là audit tĩnh, không phải xác nhận bằng GPU profiler. Ưu tiên sửa tính đúng và khả năng đo trước khi mở rộng sweep.

| Mức | Quan sát triển khai | Hệ quả nghiên cứu |
| --- | --- | --- |
| Cao | prepare_key_cache gọi .to(target_device) cho toàn bộ K/V rồi mới kiểm tra lớp có được chọn; lớp 0 luôn giữ | Logical cache nhỏ hơn chưa chứng minh bytes thực sự được copy ít hơn |
| Cao | Lớp bỏ đi dùng slice một token, không clone ngay | Slice có thể giữ backing storage lớn; cần đo peak và thời điểm giải phóng, không suy VRAM từ shape |
| Cao | CVCommunicator.forward nhận attention_mask nhưng không truyền tiếp cho B hoặc hàm shift | Chưa nên tin kết quả batching có padding; cần kiểm tra mask/position/cache end-to-end |
| Cao | TextMAS bắt mọi exception rồi continue | OOM/lỗi có thể bị loại khỏi mẫu số; phải báo đủ attempted/success/failed và so cùng tập |
| Cao | Truncation ở latent xử lý A+B; Text A không đi qua cùng đường cắt, B bị cắt riêng | Semantic prompt giống nhau nhưng evidence thực tế có thể khác |
| Cao | Nhánh regular KVComm khởi tạo top_layers nhưng không chạy calibration tại nhánh đó; một số cờ B không được truyền như latent/text | Không mặc định mode 3 là zero-step selective-KV với điều kiện tương đương |
| Cao | MCQ override evaluate_item chỉ cập nhật bộ đếm riêng dùng f1_total, không metric_totals của QA base | primary có thể đúng nhưng các trường legacy_accuracy/longbench_f1 phụ bằng 0 và item_metrics thiếu; tránh đọc nhầm JSON |
| Vừa | Timing gộp inference, metric và log; token B có chỗ đếm lại từ answer đã bỏ thinking | Chưa đủ để kết luận stage speedup hoặc tổng token sinh thực |
| Vừa | Layer selection hiệu chỉnh trên đầu tập rồi đánh giá lại tập đó; tracing có thể vẫn bật | Cần calibration split riêng và backend đo đồng nhất |
| Vừa | Manifest được dựng trước khi chốt selection và có nhãn cấu hình dùng chung | Cần lưu actual layers, effective prompt/budget/decoding và seed, không chỉ requested flags |

Nguồn: [models.py](../../../../models.py), [eval_latent.py](../../../../eval_latent.py), [com_latent.py](../../../../com_latent.py), [base_evaluator.py](../../../../dataloader/base_evaluator.py), [medqa.py](../../../../dataloader/medqa.py).

Hai lỗi OOM đã gửi cũng không phải cùng một lỗi: một lần ở tạo realignment FP32, một lần ở lm_head của prefill. Code run vẫn gọi full CausalLM và yêu cầu hidden states; chọn lớp sau đó không làm giảm peak của bước A đã thực hiện. Đây là lý do không thể dùng “chỉ truyền 70% layer” để hứa chữa OOM hiện tại. Nguồn: traceback người dùng và [models_latent.py](../../../../models_latent.py).

Mode dual-layer hiện kết hợp danh sách lớp, không tách riêng token context với token latent. Vì vậy chưa nên mô tả nó là “context lấy tầng nông, latent thoughts lấy tầng sâu” theo nghĩa hai luồng cache token tách biệt. Nguồn: [com_latent.py](../../../../com_latent.py), [models.py](../../../../models.py).

## 6. Metric: không gọi mọi con số là accuracy

Với QA thông thường, metric mới là trung bình token F1 liên tục; normalization và đếm số lần xuất hiện theo cách LongBench. Metric cũ là binary match qua ngưỡng trên hàm F1 cũ, vốn có lemmatization và set overlap. Vì vậy thay đổi không chỉ là bỏ ngưỡng 0,5. [LongBench metrics — mã nguồn chính thức](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/metrics.py), [f1.py của repo](../../../../utils/f1.py), [base_evaluator.py](../../../../dataloader/base_evaluator.py).

Hệ quả: không lấy 70% legacy match và 0,70 LongBench F1 làm cùng một đại lượng; cũng không kết luận phương pháp tiến bộ từ hai lần chạy đổi cả prompt lẫn metric. Có thể chấm lại cùng response bằng cả hai để tách hiệu ứng metric, nhưng muốn biết hiệu ứng prompt phải chạy lại.

TMATH và SAMSum đang cộng ROUGE-L recall. RepoBench dùng edit similarity. MedQA là correctness chọn đáp án. Nhãn generic legacy_match trong profile không đổi bản chất các phép tính này. Cần bảng kết quả riêng theo metric, không trung bình trực tiếp rồi gọi là “accuracy tổng”. Nguồn: [tmath.py](../../../../dataloader/tmath.py), [samsum.py](../../../../dataloader/samsum.py), [repobench.py](../../../../dataloader/repobench.py), [medqa.py](../../../../dataloader/medqa.py).

Tài liệu cũ của repo ghi nhận selected-KV có thể thua full-KV; đây chỉ là dấu hiệu cần điều tra, không phải so sánh ba phương pháp sau đổi prompt/metric. Báo cáo này không xác minh lại run gốc hoặc coi bảng tóm tắt cũ là benchmark mới. Nguồn: [problem_v1.md](../../../../docs/problem_v1.md).

## 7. Tốc độ, bộ nhớ và chi phí truyền là ba thứ khác nhau

Theo triển khai hiện tại, KV mặc định chứa cả input context lẫn latent steps, không chỉ N vector suy nghĩ. Xấp xỉ payload cache được giữ, bỏ qua sink và metadata:

**KV bytes ≈ 2 × batch × L_kept × H_KV × d_head × (T_A + N) × bytes_per_element.**

Đây là phép đếm tensor, không phải kết quả profiling. L_kept phải tính cả lớp bắt buộc giữ. Nếu chỉ truyền latent tokens thì thay T_A+N bằng N, nhưng đó là phương pháp khác cần đánh giá chất lượng riêng. Nguồn cấu trúc tensor: [models_latent.py](../../../../models_latent.py), [models.py](../../../../models.py).

Trong khi đó text payload phụ thuộc số byte UTF-8 hoặc token IDs thực truyền và protocol. Không thể so “10 latent steps” với “256 text tokens” rồi gọi latent nhỏ hơn 25,6 lần.

Để giải thích latency, tách: A-prefill; A-thinking; cache preparation/copy; B-prefill; B-decode; calibration. Tách cold-start model load/realignment khỏi steady-state, đồng thời báo cả hai nếu ứng dụng chạy ít request. Đo peak VRAM theo từng GPU, p50/p95 latency, samples/s, số lỗi và số byte copy thực tế. Với đa GPU, cần đồng bộ đúng thiết bị trước/sau vùng đo; không dùng số GPU như một bể VRAM liên tục.

Nếu calibration tốn C giây, và mỗi query sau đó tiết kiệm Δt > 0, số query hòa vốn xấp xỉ C/Δt. Nếu Δt ≤ 0 thì không có hòa vốn về thời gian dưới thiết lập đó. Đây là công thức kế toán chi phí, không phải số đo đã có.

Selective-KV chủ yếu tác động cache/attention phía B và đường truyền. Nó không tự giảm compute tất cả layer của A. Còn so với TextMAS có message cực ngắn, ưu thế truyền byte có thể đảo chiều.

## 8. Toàn cảnh nghiên cứu: tránh ghép các kết quả không cùng điều kiện

KVComm gốc tập trung selective KV, với sender không biết query; paper nêu chính điều này gây khó cho natural-language baseline. Vì vậy không dùng điểm NLD cũ để khẳng định thắng TextMAS query-aware hiện tại. Mức 2,5–6× ở phần hiệu quả là tính toán/FLOPs, không phải cam kết latency server. Nguồn đã đọc là arXiv v1; bản full-text cuối trên OpenReview bị chặn truy cập. [KVComm — Shi và cộng sự, 2025](https://arxiv.org/html/2510.03346v1).

**Alignment khác model là một bài toán riêng.** Ma trận realignment của sender hiện không chứng minh cache của A bất kỳ dùng đúng ở B bất kỳ; mapping số layer không xử lý hết khác biệt KV heads, head dimensions, tokenizer và không gian biểu diễn. Nguồn: [models_latent.py](../../../../models_latent.py), [models.py](../../../../models.py). Cache-to-Cache nghiên cứu chuyển cache khác model bằng fuser có huấn luyện; đó là bằng chứng cho một giải pháp học alignment, không cho phép thay model tùy ý trong pipeline training-free. [Cache-to-Cache — bản v1, 2025](https://arxiv.org/html/2510.03215v1).

Các khía cạnh cần mở rộng dưới dạng câu hỏi kiểm chứng:

- **Causal usefulness:** bỏ A hoặc tráo message A có làm chất lượng giảm không? Nếu không, B có thể đang tự giải; attention score cao không tự chứng minh quan hệ nhân quả.
- **Raw-context confound:** N=0 vẫn có full context KV phải được tách khỏi “không giao tiếp”. Nếu tăng điểm chủ yếu nhờ context cache, không quy toàn bộ cho latent reasoning.
- **Layer selection:** so với random cùng số lớp và cùng byte budget; chọn lớp dùng context-only có thể không tối ưu cho context+latent. Hiệu chỉnh trên held-out calibration.
- **Token selection:** context-only, latent-only và kết hợp có thể tạo các điểm chất lượng–chi phí khác nhau; đây khác với chỉ chọn layer.
- **Robustness:** độ dài, vị trí evidence, distractor, phủ định, alias, nhiều đáp án, ngôn ngữ, precision/quantization. Báo nơi phương pháp thất bại, không chỉ mean.
- **Debuggability:** text dễ đọc và kiểm toán; latent cần thử nghiệm can thiệp và log cache/config. Giải mã latent thành từ hoặc cosine hội tụ không đủ chứng minh suy luận đúng.
- **Deployment:** cache sharing đòi hỏi truy cập trạng thái nội bộ; kết quả giữa hai process/GPU/host có thể khác. Cần mô tả topology, dtype và thiết bị thực tế.
- **Security:** không xem latent là mã hóa hay bảo đảm riêng tư; phải kiểm tra rò rỉ và prompt injection riêng nếu triển khai. Đây là yêu cầu thiết kế an toàn, chưa phải kết quả tấn công trên repo.
- **Economics:** khi text message đã ngắn, tổng throughput và độ ổn định có thể quan trọng hơn token savings. Khi A context dài được dùng lại cho nhiều query, cần tính reuse riêng thay vì lẫn với giao tiếp mới.

Lưu ý tên: có một công trình KVCOMM khác về tái sử dụng cache ngữ cảnh chồng lặp, không phải selective KVComm trong repo. Không gộp speedup của hai hệ. [KVCOMM — Ye và cộng sự, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html).

## 9. Bộ thí nghiệm tối thiểu để đưa ra kết luận đáng tin

Đây là đề xuất thiết kế, không phải lệnh chạy đã thực hiện.

### Vòng 1 — kiểm tra tính đúng

Batch size 1; cùng model pair, dtype, device map, backend, dữ liệu và truncation; kiểm tra prompt/token thực tế; sửa hoặc chặn các lỗi ở mục 5. Chạy một tập nhỏ để xác nhận đủ output, metric và log lỗi, không dùng smoke test để kết luận thứ hạng.

### Vòng 2 — tách đóng góp

| Cấu hình | Câu hỏi được trả lời |
| --- | --- |
| B-only theo thông tin được phép thấy | A có đóng góp thực sự không? |
| B nhận trực tiếp context + question | Reference không có bottleneck giao tiếp; không phải upper bound toán học |
| TextMAS với budget sweep | Baseline text mạnh đến đâu khi không bị cắt giữa chừng? |
| N=0 full context KV | Bao nhiêu lợi ích đến từ truyền context, chưa có latent reasoning? |
| N>0 full-KV | Latent steps bổ sung gì? |
| N=0 selective-KV | Lợi ích riêng của selection khi chưa có latent steps |
| N>0 selective-KV | Combined method cải thiện điểm chất lượng–chi phí nào? |
| Random layers, matched budget | Heuristic chọn lớp có tốt hơn chọn ngẫu nhiên không? |

Sau đó mới thêm latent-only/context-only và hybrid text reasoning + KV communication nếu cần cô lập sâu hơn. Kiểm tra đường chạy N=0 thực sự giữ context cache; không lấy tên mode làm bảo đảm.

### Vòng 3 — đo và thống kê

Sweep N nhỏ trước rồi mở rộng có kiểm soát; sweep tỷ lệ layer và Text A token budget. Tuning trên validation, chọn lớp trên calibration riêng, báo test held-out. Dùng cùng sample IDs và seed; khi sampling, chạy nhiều seed và reset RNG theo thiết kế, tránh calibration làm lệch chuỗi random.

Báo metric từng task, latency và VRAM từng cấu hình, cùng attempted/success/failed. Không âm thầm bỏ OOM. Với F1, bootstrap theo cặp cùng câu hỏi để có khoảng tin cậy; với correctness nhị phân có thể dùng McNemar trên prediction ghép cặp. Nhiều seed không thay thế đủ sample, đặc biệt các tập toán nhỏ.

Không cần phương pháp mới thắng mọi ô. Một kết quả có giá trị là: **ở ngưỡng suy giảm chất lượng đã định trước, combined method giảm latency/bytes/VRAM đo được so với full-KV, và vẫn cạnh tranh với TextMAS được tuning công bằng.** Chọn ngưỡng trước khi xem test, không chọn sau để làm đẹp kết quả.

## 10. Khuyến nghị cuối và giới hạn

Trước mắt nên đầu tư vào ba việc theo thứ tự: tính đúng của pipeline và metric; baseline công bằng; đo đường biên chất lượng–chi phí. Prompt v2 là nền tảng tốt nhưng chưa đủ bằng chứng để gọi tối ưu. Không nên kết luận từ tên task, một seed, số latent steps, hoặc một bảng điểm dùng metric cũ.

Nghiên cứu đã đủ để bác bỏ các khẳng định thắng phổ quát và thiết kế thử nghiệm phân biệt nguyên nhân. Khoảng trống còn lại cần dữ liệu thực nghiệm của chính repo: dự đoán từng mẫu dưới prompt hiện tại, ngân sách thực dùng, profiler và calibration held-out. Tìm thêm paper sẽ không thay thế những phép đo này.

Phạm vi: audit tĩnh workspace ngày 07/09/2026; không chạy GPU, không thay mã nguồn thực thi; không tuyên bố tái lập paper hay kiểm toán toàn bộ code. Tài liệu tham chiếu dùng đúng phiên bản đã truy cập; hạn chế truy cập bản cuối KVComm được nêu ở mục 8.
