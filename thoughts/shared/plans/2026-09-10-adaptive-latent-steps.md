# Plan thảo luận: Adaptive Latent Steps cho KVComm / LatentMAS

Ngày: 2026-09-10.
Trạng thái: IMPLEMENTED v1 infrastructure (2026-09-11), chưa chạy calibration/benchmark trên server.
Phạm vi được xác nhận: controller luật/ngưỡng không training, Mode 1/2, batch_size=1.
Hướng dẫn chạy: `tests/ADAPTIVE_LATENT_TESTING.md`. Các phần dưới giữ lại thiết kế/nghiên cứu ban đầu; xem ghi chú triển khai cuối file để biết phạm vi đã hoàn thành.

## 1. Mục tiêu và bằng chứng hiện có

Mục tiêu đã được người dùng chọn: tăng chất lượng trong cùng thời gian, hỗ trợ Adaptive trên cả Mode 1 và Mode 2.

Cách đo đề xuất cần làm rõ: cùng tổng thời gian inference trên cùng tập mẫu, tương đương cùng latency trung bình; không yêu cầu mọi câu có cùng thời gian. Câu ít hưởng lợi từ thêm bước được cấp ít compute hơn, phần ngân sách đó dành cho câu có khả năng cải thiện. Báo thêm p95 để tránh che khuất một số câu quá chậm.

Theo `EXPERIMENT_RESULTS.md`, HotpotQA 500 mẫu, Qwen3-4B → Qwen3-4B:

| Mode | N | F1 | Tổng thời gian (s) |
|---|---:|---:|---:|
| Full KV | 10 | 0.6697 | 778.5 |
| Full KV | 80 | 0.6898 | 2955.8 |
| Top-70% | 10 | 0.6691 | 679.7 |
| Top-70% | 20 | 0.6827 | 1014.2 |
| Top-70% | 40 | 0.6897 | 1804.1 |
| Top-70% | 80 | 0.6928 | 3071.0 |

Nguồn hiện là bảng tổng hợp; chưa có prediction ghép cặp để xác minh ý nghĩa thống kê. N cao vẫn cải thiện F1 trung bình: chưa có bằng chứng về suy thoái ở N cao trên cấu hình này.

Điểm cần vượt qua: Mode 2 N=40 chỉ thấp hơn N=80 là 0.0031 F1 nhưng nhanh hơn khoảng 41.3%. Adaptive chỉ thắng fixed-80 có thể chưa mang lại lợi ích hơn việc đặt cố định N=40. Phải so với toàn bộ đường cong fixed-N, đặc biệt N=20/40.

Phân biệt hai giả thuyết:

- H1: số bước có ích khác nhau giữa các mẫu; có tiềm năng phân bổ compute.
- H2: tín hiệu quan sát được trước khi biết đáp án dự đoán được thời điểm dừng tốt.

H1 đúng không kéo theo H2 đúng. Thêm điều kiện hội tụ vào vòng lặp không tự chứng minh khả năng phân bổ compute.

## 2. Các quyết định cần người dùng chốt

Đã nhận phản hồi vòng 1, xác nhận Q4/Q5 và yêu cầu hạn chế ở controller/adapter nhỏ, không training. Thiết kế v1 diễn giải yêu cầu này theo hướng controller luật/ngưỡng, không thêm trọng số phải học.

| ID | Quyết định | Đề xuất ban đầu | Trạng thái |
|---|---|---|---|
| Q1 | Training | Giữ frozen A/B; controller nhỏ bên ngoài dùng luật/ngưỡng, không train LLM, LoRA, MLP hay halting head. Chọn hyperparameter trên calibration riêng | Phạm vi v1 theo cách hiểu “không training” đã nêu với người dùng |
| Q2 | Mục tiêu | Tăng chất lượng trong cùng thời gian. Đề xuất so tổng runtime/mean latency trên cùng tập, tính cả controller | Đã chốt mục tiêu; cách đo là đề xuất |
| Q3 | KV routing | Hỗ trợ cả Mode 1 và Mode 2; policy là trục độc lập với routing | Đã chốt |
| Q4 | Compute/data | Người dùng cho phép chạy khảo sát qua đêm. Quy mô mẫu sẽ chọn theo smoke timing; calibration/validation tách evaluation là thiết kế đề xuất | Đã chốt khả năng chạy qua đêm |
| Q5 | Receiver online | Khi inference thực tế, B chỉ chạy một lần sau A; các lượt B ở nhiều checkpoint chỉ phục vụ thu dữ liệu offline | Đã chốt |
| Q6 | Task pilot/transfer | Người dùng sẽ cung cấp danh sách task, tin nhắn hiện chưa có danh sách. HotpotQA/GSM8K chỉ là ví dụ tạm, chưa được chốt | Chờ danh sách |

Q6 (danh sách task) còn chờ người dùng bổ sung. Q5 đã chốt: không triển khai receiver probe online trong v1. Một adapter có trọng số học được vẫn cần training; thuật ngữ dùng cho v1 là controller luật/ngưỡng để tránh nhập nhằng. Chọn ngưỡng theo dữ liệu là calibration, vẫn phải báo dữ liệu và chi phí, không gọi là calibration-free.

## 3. Brainstorm và hướng khuyến nghị

| Hướng | Tín hiệu và hành động | Ưu điểm | Điểm cần kiểm chứng |
|---|---|---|---|
| A. Hidden convergence | Cosine, thay đổi norm, cửa sổ/patience | Rẻ, dễ kiểm thử, baseline cần có | Hidden ổn định chưa chứng minh B đã đủ thông tin |
| B. Latent value novelty | Thay đổi/novelty V ở một số layer, kết hợp hidden trajectory | Quan sát bộ nhớ A thực sự truyền | Redundancy hình học không đồng nghĩa dư thừa chức năng |
| C. Receiver checkpoint | B prefill thử tại vài checkpoint; theo dõi hidden/logit hoặc câu trả lời | Gắn trực tiếp với đầu ra B | Prefill, chuyển cache, clone và đồng bộ GPU có thể làm mất speedup |
| D. Learned budget/controller | Controller nhỏ dự đoán N hoặc continue/stop từ trajectory | Có thể học tín hiệu phức tạp | Cần dữ liệu/huấn luyện, nguy cơ overfit, thay đổi claim training-free |

Phạm vi v1: xây infrastructure thu thập trajectory và stop policy có thể hoán đổi; implement A làm baseline; chỉ nâng B thành ứng viên chính nếu pilot cho thấy hữu ích hơn A. Dùng C offline để đánh giá A/B. D là phương án brainstorm lịch sử, không thuộc implementation v1.

Controller v1 dùng tín hiệu hidden/value làm proxy cho lợi ích của việc tiếp tục; không có mô hình học dự đoán gain. Mục tiêu nghiên cứu vẫn là phân bổ compute giữa các mẫu với hard cap lớn hơn budget fixed đang so. Phải kiểm chứng proxy trên dữ liệu offline trước khi kết luận nó dự đoán được lợi ích downstream.

Thiết kế v1: tại checkpoint n, controller tính feature trajectory, so với ngưỡng đã khóa, cập nhật patience rồi trả continue/stop. Chọn một cấu hình từ grid nhỏ gồm min_steps, interval, patience và thresholds trên calibration/validation để đạt chất lượng tốt nhất trong ngân sách runtime baseline. Ví dụ minh họa: một số câu dùng 5–10 bước, một số dùng 40–80 bước khi so fixed-20. Số bước trung bình tương đương chưa bảo đảm runtime tương đương vì context và B generation khác nhau. Online không đọc kết quả các câu tương lai hoặc gold để cân lại ngân sách.

Controller là code xử lý feature và trạng thái, không phải mạng neural mới. Không có optimizer, loss, backward, checkpoint trọng số controller hoặc RL. Kết quả calibration là JSON chứa ngưỡng/cửa sổ/layer và provenance. Chi phí offline chủ yếu ở việc chạy B tại nhiều checkpoint để chấm ảnh hưởng số bước; cần báo riêng dù không training.

Không đồng nhất “câu khó” với “nên chạy thêm”: có câu khó nhưng thêm bước vẫn không giúp. Khảo sát cả gain tới checkpoint gần và xa để phát hiện dừng sớm bỏ lỡ cải thiện muộn. Mọi nhãn tương lai/gold chỉ dùng offline; policy online chỉ đọc trạng thái tới n.

Mode 1/2 dùng chung controller và giao diện. Có thể calibrate bộ ngưỡng riêng theo mode; việc cùng bộ ngưỡng có transfer được hay không là thí nghiệm, không mặc định. So fixed/adaptive trong từng mode với cùng danh sách layer được đóng băng ở Mode 2.

Một nghiên cứu tốt có thể kết luận cosine/novelty không dự đoán được lợi ích tương lai. Không ép tín hiệu đó thành phương pháp chính nếu thua fixed-N.

### 3.1 Điều chỉnh hai ý tưởng từ trao đổi trước

1. Không dùng trực tiếp khoảng cách raw cached K giữa hai vị trí như semantic novelty: K đã qua RoPE, nên thay đổi vị trí gây thay đổi vector. Muốn dùng K phải đưa về cùng hệ tọa độ đúng implementation Qwen3 và xác minh bằng test. V không qua RoPE là ứng viên rẻ hơn cho pilot, dù vẫn phụ thuộc context/vị trí qua hidden computation.
2. Vector KV mới nằm trong span cũ vẫn có thể thay đổi attention output. Ví dụ thêm một K/V trùng lặp làm thay đổi mẫu số softmax và tổng mass phân bổ cho latent so với context. Vì vậy residual bằng 0 không phải chứng nhận có thể dừng.
3. `_apply_realignment()` rescale embedding về target norm. Với vector cùng norm, squared L2 của vector chuẩn hóa bằng `2 * (1 - cosine)`. Không xem hai tín hiệu đó là hai bằng chứng độc lập. Đo norm trên hidden chưa realign nếu cần.
4. Một threshold cosine chỉ đo trạng thái cuối, trong khi state thực của vòng lặp còn có toàn bộ cache đang tăng độ dài. Không áp dụng trực tiếp bảo đảm fixed-point của kiến trúc recurrent khác.

### 3.2 Related work và giới hạn novelty

- [Learning When to Stop (2025)](https://arxiv.org/html/2511.21581v1): học binary halting head bằng SFT/RL; adaptive latent reasoning đã tồn tại.
- [LaTER (2026)](https://arxiv.org/abs/2605.07315): phiên bản training-free hồi tiếp hidden state, giữ latent KV, dùng entropy và model-native stop-token probe để chuyển sang explicit CoT. Đây là related work gần hơn nhận định ở trao đổi trước. Training-free tự nó không đủ làm novelty.
- [LatentMAS](https://arxiv.org/abs/2511.20639): nền tảng trao đổi latent working memory giữa agents.

Research question cập nhật: tín hiệu rẻ tại sender có dự đoán được lợi ích downstream của việc tiếp tục latent reasoning, để tăng chất lượng trong cùng tổng thời gian của hệ A→B hay không? Phải đo chi phí quyết định, tính nhất quán cache/position và ảnh hưởng KV routing.

Không tuyên bố “đầu tiên” hay communication sufficiency được bảo đảm. Cơ chế entropy/stop-token nếu được chuyển sang A→B chỉ là baseline lấy cảm hứng từ LaTER; không gọi là tái lập nguyên paper.

## 4. Khảo sát trước khi chọn thuật toán cuối cùng

### 4.1 Tập dữ liệu

- Khôi phục và lưu ID của đúng 500 mẫu HotpotQA lịch sử. Loader hiện shuffle validation bằng seed nội bộ 42, rồi lấy 500 mẫu; không mặc định CLI seed thay đổi tập mẫu.
- Dùng calibration và validation khác ID với 500 mẫu đó. Thử 100 calibration + 100 validation như quy mô pilot đề xuất; đo smoke timing trước để điều chỉnh cho lượt chạy qua đêm đã được người dùng chấp nhận. Không cam kết số mẫu hoặc thời lượng khi chưa biết task và runtime thực tế.
- Vì 500 mẫu đã được xem kết quả nhiều lần để chọn hướng nghiên cứu, gọi là historical evaluation set; giữ thêm holdout chưa dùng cho thiết kế để xác nhận cuối cùng nếu data/compute cho phép.
- Lưu dataset ID, revision/fingerprint nếu có, source split, preprocessing version, sample IDs và hashes. Assert không giao nhau; không chỉ tách bằng `--limit`.
- HotpotQA giữ nguyên oracle supporting sentences và prompt/metric v2. Không đồng thời đổi sang distractor đầy đủ.
- GSM8K dùng loader có sẵn nhưng audit answer extraction và metric trước khi dùng làm bằng chứng reasoning; TMATH ROUGE-L chỉ là task phụ nếu được chọn.

### 4.2 Checkpoint và đối chứng

- Pilot fixed checkpoints đề xuất: `0, 5, 10, 20, 40, 80`. N=0 giữ cùng prompt A/B và truyền input KV, chỉ bỏ vòng latent; không đồng nhất nó với B-only hoặc KVComm prompt khác.
- Thu hidden statistics mỗi bước; đo V tại checkpoint hoặc mỗi 5 bước.
- Checkpoint generation phải thật sự chạy B ra đáp án và chấm bằng metric task. Một logit đầu tiên của B có thể là token định dạng, không phải đáp án hoàn chỉnh.
- Greedy cho kiểm thử và pilot dễ tái lập; run xác nhận dùng đúng sampling profile lịch sử trên nhiều seed. Không so trực tiếp greedy mới với một scalar sampling cũ để tuyên bố thắng.
- Seed decoding cố định theo `(base_seed, sample_id)`; lưu/khôi phục RNG khi chạy probe. Cùng seed không bảo đảm ghép cặp hoàn hảo khi các trajectory sinh độ dài khác nhau, nhưng tránh hiệu ứng đổi số RNG ở mẫu trước.

### 4.3 Cách chạy tiết kiệm và bảo vệ cache

Phiên bản đúng trước: chạy độc lập từng N trên pilot nhỏ để làm reference. Sau đó có thể chạy A tới N_max một lần, thử B ở checkpoint để tiết kiệm prefill/latent compute.

B chỉ nhận bản cache tách biệt của prefix checkpoint; B.generate có thể mutate DynamicCache. Không đưa cache đang chạy A trực tiếp cho B. Giới hạn một bản clone checkpoint còn sống; giải phóng sau probe; không giữ full cache cho mọi checkpoint trên GPU.

Nếu đo B từ prefix crop của rollout dài, chỉ lấy `T_A+n` token ở mọi layer; dựng lại metadata/position/cache counters từ độ dài thật. Kiểm tra cache prefix bằng reference fixed-n trước khi tin số liệu. Mọi lựa chọn stop ở n chỉ dùng feature có tại <=n.

Không dùng thời gian offline replay/checkpoint làm latency online. Đo adaptive online bằng lượt chạy riêng, chỉ thực hiện các bước thực sự đã chọn.

Lượt chạy qua đêm: trước hết đo một số mẫu đại diện theo độ dài ở các checkpoint để ước lượng tổng thời gian, tính cả các lượt B và cả hai mode. Ghi kết quả tăng dần theo sample/checkpoint/mode; hỗ trợ resume theo ID và config hash, không chạy lại các ô đã hoàn thành và không đếm trùng metrics. Ghi failed/OOM riêng; không âm thầm bỏ mẫu khỏi báo cáo. Thiết kế cho phép thu được dữ liệu hữu ích ngay cả khi chưa hoàn tất toàn bộ grid trong một đêm.

### 4.4 Phân tích tín hiệu

- Chấm `score(x,n)` và `gain(x,n→m) = score(x,m)-score(x,n)`.
- Phân nhóm: thêm bước giúp, gây hại, hoặc không đổi; kiểm tra theo evidence length và loại câu hỏi.
- Oracle lựa chọn N tốt nhất theo gold trên checkpoint grid chỉ là upper bound chẩn đoán, không phải thuật toán deploy.
- Nếu tất cả N đều sai, không gọi N nhỏ là “đủ suy nghĩ” chỉ vì ngang điểm N_max. Báo riêng nhóm all-fail.
- Chọn threshold/feature/layer trên calibration, không train controller. Chọn policy đáp ứng ngân sách cuối trên validation; khóa cấu hình trước holdout.
- Nếu tín hiệu chỉ tương quan với step index, policy có thể tương đương fixed-N. Bắt buộc so random/mixed budget không nhìn nội dung, có phân phối N hoặc chi phí tương đương.

## 5. Stop policy v1 — giao diện dự kiến

Adaptive là một trục độc lập với KV routing. Giữ Mode 1/2, thêm tùy chọn policy; không tạo mode mới cho mỗi tổ hợp.

CLI v1 đã implement:

```text
--latent_step_policy fixed|cosine|hidden_value
--latent_steps 80                   # fixed N, hoặc hard cap khi adaptive
--min_latent_steps 10
--latent_check_interval 5
--latent_patience 2
--latent_policy_config path.json    # thresholds, feature settings, layers, provenance
--latent_trace                     # chi tiết để phân tích, có thể tăng overhead
```

Ý nghĩa phải thống nhất:

- Fixed tiếp tục hiểu `latent_steps=N` như cũ.
- Adaptive: `latent_steps=N_max`; `actual_steps` là số forward latent đã hoàn thành.
- Kiểm tra sau forward, không trước khi thêm token KV của bước đó.
- Chỉ tăng patience khi n >= min_steps và n thuộc checkpoint hợp lệ. Với min=10, interval=5, patience=2, dừng sớm nhất ở n=15 nếu kiểm tra n=10/15 đều đạt.
- Nếu n không chia hết interval nhưng đạt hard cap, vẫn kết thúc ở cap.
- Stop reason tối thiểu: `fixed_budget`, `criterion_met`, `max_steps`. Tách lỗi số học khỏi quyết định hội tụ; không coi NaN là đạt ngưỡng và không đưa kết quả lỗi vào success average.
- V1 adaptive chỉ batch_size=1; fail rõ trước load model cho cấu hình không hỗ trợ.
- min=max=N là trường hợp kiểm thử tương đương fixed-N. Cap được ưu tiên làm stop reason khi chạm giới hạn.
- Chưa có policy config đã calibrate: fail rõ khi yêu cầu adaptive thật, vẫn cho phép trace fixed runs. Không phát hành threshold tùy ý dưới nhãn “tối ưu”.

Đặc trưng baseline A: cosine hidden liên tiếp, thay đổi log-norm và xu thế trong cửa sổ. Không bắt buộc kết hợp mọi feature: dùng ablation để chọn.

Đặc trưng ứng viên B: cosine novelty của V mới so với cửa sổ latent V trước, tính từng head/layer, kết hợp với baseline A. Chỉ dùng latent slice, không trộn context/sink. Chọn nhóm layer đại diện có giới hạn trên calibration, giữ cố định khi đo. Cần đủ history, kiểm tra zero norms, tính FP32; không gom full KV của mọi layer sang một GPU.

Subspace residual/QR là ablation sau, chưa phải default: overhead lớn hơn, dễ bị ảnh hưởng rank/window. Nếu thử K, phải có bước xử lý RoPE và test riêng trước.

## 6. Thay đổi code dự kiến

| File | Công việc |
|---|---|
| `adaptive_latent.py` (mới) | Policy config, feature collector, patience state, stop decision, per-run stats; controller không nhận gold answer |
| `models_latent.py` | Hook sau mỗi bước; reset state mỗi sample; trả DynamicCache như cũ và gắn actual length/stats; giữ fixed path tương đương |
| `eval_latent.py` | Dùng actual_steps cho mask/metadata/log; thu token B từ generated IDs trước strip; per-item timings; chặn adaptive batch>1 |
| `com_latent.py` | CLI/config validation, policy load/hash, run naming/manifest, deterministic sampling options dùng chung cho cả fixed/adaptive |
| `sweep_latent.sh` | Truyền policy và trace cho m1/m2; sửa hiện tượng `--track_convergence` hiện chỉ được forward ở m4/m5; dry-run rõ policy/cap |
| `utils/response_logging.py` | Metadata adaptive theo sample, giữ tương thích record v2 bằng trường bổ sung và `adaptive_schema_version=1` |
| `dataloader/hotpotqa.py` / helper manifest | Load split/sample IDs rõ ràng cho pilot, giữ preprocessing và default lịch sử |
| `scripts/profile_adaptive_latent.py` (mới) | Reference fixed-N, checkpoint profiling tùy chọn, clone/cache isolation, sample manifest |
| `scripts/analyze_adaptive_latent.py` (mới) | Oracle chẩn đoán, chọn threshold, paired comparisons, xuất policy JSON, so fixed frontier |
| `tests/test_adaptive_latent.py` (mới) | Unit tests state machine và các invariant cache bằng synthetic/model nhỏ |
| `tests/test_adaptive_latent_qwen3.py` (mới) | Regression Qwen3/CPU nhỏ và GPU server test theo khả năng môi trường |
| `tests/ADAPTIVE_LATENT_TESTING.md` (mới) | Lệnh Windows/CPU và Linux/GPU, expected outputs, điều kiện chạy từng nhóm |

Các phụ thuộc cần chú ý từ code hiện tại:

- `models_latent.py`: `_kvcomm_context_length`, `_kvcomm_latent_length`, `_kvcomm_logical_length` hiện dựa `self.latent_steps`. Phải dùng `actual_steps`; đếm từ cache trước B để xác minh.
- `eval_latent.py`: `last_latent_length`, `generated_tokens_a`, `latent.steps` đang dùng budget. Phải phân biệt configured cap và actual.
- `models_latent.py`: convergence history đang reset theo flag, `.item()` và INFO mỗi step gây đồng bộ/log overhead. Buffer diagnostic tensors nhỏ và copy cuối sample; online chỉ đồng bộ khi cần quyết định.
- `models.py`: giữ logic shift_back/sink hiện tại; mọi geometry phải dùng cache thực sự có. Adaptive N=20 phải có geometry như fixed N=20, không như N_max=80. Chỉ sửa nơi audit chứng minh đang đọc configured cap.
- Mode 2: đóng băng cùng danh sách layer khi so fixed/adaptive; không recalibrate selection theo mỗi N trong thí nghiệm chính. Calibration cho layer selection và stopping phải được ghi riêng, không làm lẫn RNG/metrics/logs.
- Adaptive + Mode 5/Mode 4 chưa thuộc v1 theo đề xuất; reject tổ hợp chưa kiểm thử. Fixed modes cũ tiếp tục là regression references.
- Local code còn CausalLM prefill trả logits cho mọi vị trí và tạo realignment FP32 trên GPU. Đây là rủi ro OOM đã gặp. Đối chiếu code server trước benchmark; mọi fix memory/forward backend nếu cần phải áp dụng cho cả fixed/adaptive và kiểm thử equivalence, không gộp speedup đó vào stopping.

## 7. Logging và phép đo

Per sample:

```text
sample_id, prompt hashes, model/backend versions, seed
policy, policy_config_hash, configured_max_steps, actual_steps, stop_reason
feature_trace (optional), decision_checkpoints, selected_layers
context_length, actual_cache_length_before_B
generated_B_token_count_raw, answer, task_metrics, error/status
prefill_A_ms, latent_A_ms_including_controller, controller_ms (diagnostic)
kv_handoff_ms, B_generation_ms, end_to_end_ms
logical_payload_bytes, unique_retained_storage_bytes (nếu đo được), per_device_peak_VRAM
```

Controller time là phần nằm trong latent stage; không cộng thêm hai lần. KV handoff có thể thực hiện lazy trong `cv.forward`, nên đo tại nơi route/copy xảy ra để tránh gán nhầm vào A hoặc B.

Đồng bộ các CUDA device liên quan trước/sau đo stage; full profiling riêng vì synchronization có thể làm thay đổi latency. Run chính dùng end-to-end timing đồng nhất giữa phương pháp. Warm-up cùng cách, log GPU mapping/load/precision.

Chi phí init model, realignment, calibration, offline probes được báo riêng. Report mean/p50/p95 latency và số bước; failures nằm trong mẫu số coverage. Token B đo trước khi loại thinking/format để xác định adaptive có làm B phải sinh dài hơn không.

`logical_payload_bytes` không phải network traffic hoặc VRAM thật: slice có thể giữ storage gốc. Không suy số bước giảm 50% thành 50% byte hoặc 50% end-to-end speedup. T_A và thời gian B có thể chiếm phần lớn.

## 8. Kiểm thử bắt buộc

1. Policy disabled và fixed-N cho cache/logits/response như reference trong tolerance của backend.
2. Force stop ở n: số forward A bằng 1 prefill + n latent; không chạy đến cap rồi crop giả adaptive.
3. Adaptive forced n so fixed n: K/V prefix, length, B positions, causal masking và greedy output phù hợp; kiểm tra Full KV và Mode 2 cả layer sink/selected, kể cả layer 0 không chọn nếu mode hỗ trợ.
4. min/max/check_interval/patience: hội tụ, phá chuỗi ổn định, hard cap không chia hết interval, reset giữa sample, N=0 fixed, min=max, invalid flags.
5. NaN/Inf/zero norms, không đủ novelty history; các trường hợp này không được âm thầm cho dừng thành công.
6. Checkpoint B probe không mutate cache A, counters, global RNG hoặc ảnh hưởng sample kế tiếp.
7. Feature/trace mode không làm đổi đáp án fixed run; timing trace không được dùng thay timing production.
8. JSONL thực ghi actual_steps cho từng mẫu khác nhau; backward compatibility response record hiện có; sample ID/split overlap validation.
9. Multi-device GPU smoke trên server 4×3080, batch=1, không gom tensor cache lớn vào một GPU.
10. Nếu triển khai K novelty: unit test phân biệt thay đổi semantic với rotation vị trí; nếu chưa có test thì chỉ dùng V.

Viết lệnh test vào `tests/ADAPTIVE_LATENT_TESTING.md` cùng thư mục test theo yêu cầu trước của người dùng. Không đòi GPU cho pure controller tests; báo rõ GPU tests chưa chạy nếu local thiếu môi trường.

## 9. Milestones và tiêu chí quyết định

- [ ] M0: Q2/Q3/Q4/Q5 đã được xác nhận, Q1 được triển khai theo hướng không training; nhận danh sách task Q6 và đóng thiết kế v1.
- [x] M1: Per-sample metadata, fixed trace, ID manifests, fixed reference tests.
- [ ] M2: Pilot calibration/validation, checkpoint answers, phân tích H1/H2, đo overhead signal.
- [x] M3 (infrastructure): Implement cosine + hidden_value candidate, cache invariants và regression tests. Chưa có bằng chứng để chọn policy thắng.
- [ ] M4: Khóa policy; benchmark online so fixed-10/20/40/80, matched-cost random/mixed allocation; paired bootstrap.
- [ ] M5: Kiểm tra transfer sang task thứ hai với threshold đã khóa; báo riêng nếu phải recalibrate từng task.

Tiêu chí chính theo Q2: chất lượng cao hơn baseline fixed-N trong ngân sách thời gian tương đương trên cùng task, hardware, prompt/decoding và tập mẫu. Chọn các mốc ngân sách từ fixed-10/20/40/80; calibrate policy trên validation, khóa trước evaluation. Báo chênh lệch metric với paired bootstrap và runtime đo thật qua các lượt lặp. Sai số runtime chấp nhận được phải định trước sau khi đo độ nhiễu máy; không điều chỉnh policy bằng gold hoặc kết quả tổng của holdout. Tiêu chí tăng tốc với ít mất F1 chỉ còn là phân tích phụ.

Điều kiện đổi hướng: nếu adaptive không cải thiện fixed frontier trên validation hoặc feature chủ yếu dự đoán n, giữ infrastructure và negative results, đánh giá lại proxy hoặc dùng fixed-N tốt nhất. Controller học và receiver probe online nằm ngoài phạm vi v1. Không tune tiếp trên holdout cho tới khi đạt số mong muốn.

## 10. Deliverable của bước implement sau khi chốt

Một thay đổi có thể review gồm policy module, tích hợp actual_steps, profile/analyze scripts, backward-compatible logs, tests và hướng dẫn chạy server. Policy JSON cuối chỉ được xuất sau calibration/validation thật. Chưa có số liệu đó thì deliverable chỉ là infrastructure + candidates, không gắn nhãn một policy đã được chứng minh hiệu quả.

## 11. Ghi chú triển khai v1 — 2026-09-11

Đã có `adaptive_latent.py`, tích hợp vòng latent/evaluator/CLI/sweep, manifest split,
profile fixed-N độc lập có resume, analyzer calibration/validation xuất candidate JSON,
và `scripts/compare_adaptive_runs.py` so response logs online ghép cặp.
Không thay đổi KV routing/shift-back/sink hay prompt/metric; các độ dài dùng actual_steps.
Mode 2 adaptive yêu cầu explicit layer list, không tự chọn lại layer theo N.

Quyết định triển khai cụ thể:

- V novelty = minimum cosine distance tới V latent trước trong window, mean qua
  head/layer được cấu hình. Chưa implement QR/subspace, K novelty, hidden trend.
- Profiling v1 chạy A/B độc lập mỗi N; reuse checkpoint chỉ là tối ưu tùy chọn sau.
- B generation timing bao gồm lazy KV handoff; chưa instrument riêng handoff.
  Host controller timing là diagnostic, không phải GPU kernel timing chuẩn xác.
- Analyzer không dùng score ở N chưa được chạy B, không chấp nhận profile thiếu
  mẫu/ô hoặc calibration/validation trùng nhau. Random control offline khớp phân
  phối N, không đảm bảo khớp thời gian; cần thêm thí nghiệm matched-time online.
- Chưa cung cấp threshold tối ưu. File `tests/fixtures/adaptive_smoke_policy.json`
  chỉ ép dừng để test, được ghi rõ TEST_ONLY_NOT_CALIBRATED.
- Có kiểm thử CPU tiny Qwen3 về cache/logits/greedy output/mask/position, Full/Selected
  và layer 0 không nằm trong danh sách chọn; tests cho GPU/multi-device để chạy sau.
- Không sửa hai rủi ro OOM prefill full-logits/realignment FP32 ở lượt này, tránh
  gộp hiệu quả memory fix vào stopping. Cần đối chiếu với bản đang chạy trên server.

M2/M4/M5 vẫn chờ dữ liệu/calibration và benchmark thực tế. Danh sách task để cấu
hình qua CLI; HotpotQA chỉ là ví dụ trong hướng dẫn, không áp đặt thay lựa chọn người dùng.
