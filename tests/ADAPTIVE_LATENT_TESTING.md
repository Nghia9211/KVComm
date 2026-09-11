# Adaptive Latent Steps v1 — kiểm thử và chạy thí nghiệm

> Cleanup: Mode 4/5 và `--track_convergence` đã được gỡ. Dùng `--latent_trace`.
> `--mode all` chỉ gồm m3/textmas/m1/m2. Kết quả lịch sử và policy JSON giữ nguyên;
> code hash mới yêu cầu output profiling mới, không resume vào JSONL trước cleanup.
> Script sweep dùng Python đang active hoặc `--python PATH`, không hard-code server.


V1 là **controller không training**, frozen A/B; không có adapter học, LoRA,
receiver probe online hay gold answer trong quyết định dừng. Hỗ trợ Qwen3 → cùng
Qwen3, Mode 1/2, batch_size=1. Mặc định `fixed` giữ hành vi cũ.

Chưa có threshold được xác nhận bằng benchmark thật. File trong `tests/fixtures`
chỉ ép dừng để smoke test, tuyệt đối không dùng kết quả đó làm bằng chứng nghiên cứu.

## 1. Test local không tải model/dataset

Chạy từ thư mục `KVComm`. Pure Python controller/data/analysis tests:

```powershell
python -m unittest discover -s tests -p test_adaptive_latent.py -v
```

Tiny random Qwen3 + toàn bộ regression tests, dùng môi trường hiện có của workspace:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_adaptive_latent_qwen3.py -v
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trên Linux/server đã cài dependencies của repo:

```bash
python -m unittest discover -s tests -v
ADAPTIVE_TEST_DEVICE=cuda:0 python -m unittest discover -s tests -p test_adaptive_latent_qwen3.py -v
ADAPTIVE_TEST_DEVICE=multi python -m unittest discover -s tests -p test_adaptive_latent_qwen3.py -v
```

`multi` cần >=2 GPU và Accelerate, phân 4 layer của model nhỏ lên 2 GPU. Test này
không thay thế smoke Qwen3-4B trên 4×3080. CPU đã kiểm tra bằng Transformers
4.53.3; Python mặc định của máy có Transformers 4.47.1, không có Qwen3.
Không cần thay đổi môi trường mặc định. Test evaluator chặn tác dụng phụ tải NLTK
của code cũ; các bài test không sử dụng NLTK metrics.

Các invariant chính: 1 prefill + đúng n latent forward; không chạy đến cap rồi
crop; cache/mask/position/logits/output khớp fixed-n; sink layer không chọn vẫn
giữ token đầu; layer 0 vẫn full theo code KVComm cũ; RNG và patience reset;
N=0 fixed; cap ưu tiên; NaN/Inf không được xem là hội tụ; JSONL ghi actual N và
số token B **trước khi strip thinking**.

## 2. Smoke GPU bằng policy ép dừng (không phải ngưỡng thực nghiệm)

```bash
python com_latent.py --model_A Qwen/Qwen3-4B --model_B Qwen/Qwen3-4B --device auto --device_B auto --test_task hotpotqa --do_test_latent --batch_size 1 --limit 2 --shift_back --latent_step_policy cosine --latent_steps 8 --min_latent_steps 2 --latent_check_interval 1 --latent_patience 2 --latent_policy_config tests/fixtures/adaptive_smoke_policy.json --greedy --per_sample_seed --profile_timing --latent_trace
```

Kỳ vọng mỗi mẫu finite/nonzero: `actual_steps=3`, `stop_reason=criterion_met`,
`actual_cache_length_before_B=context_length+3`; chỉ chạy B một lần/mẫu.
So fixed bằng cùng lệnh nhưng thay policy thành `fixed`, `--latent_steps 3`
và bỏ `--latent_policy_config`. Prompt/metric/decoding phải giữ nguyên.

Mode 2: thêm `--latent_kv_select --top_layers 0 --layers_list ...` với **cùng
danh sách layer của reference đã lưu**. Không dùng `--top_layers 0.7` ở adaptive:
đó là yêu cầu chạy lại layer ranking, làm lẫn hai thay đổi. Nếu cần top-70%, chọn
layer một lần trên calibration riêng rồi dùng danh sách explicit ở tất cả N/policy.

Các tổ hợp adaptive + batch>1 / Mode4 / Mode5 / auto-ranking / random-selection /
layer-curve bị reject trước khi tải model. CLI phải khớp trường `controller` trong
policy JSON đã khóa; không thể đổi cap/interval mà vô tình dùng nhầm cấu hình đó.

## 3. Tạo split có ID rõ ràng

Ví dụ pilot nhỏ trước, tăng số mẫu sau khi đo thời gian:

```bash
python scripts/profile_adaptive_latent.py --task hotpotqa --prepare_manifest snapshots/adaptive/pilot_hotpotqa.json --calibration 10 --validation 10 --holdout 10
```

HotpotQA tự dùng loader `hotpotqa_full` với preprocessing oracle supporting
sentences hiện tại, giữ prompt/metric v2. 500 mẫu đầu theo shuffle seed nội bộ 42
được đưa vào `historical`, không trộn với cal/val/holdout. Sau đó mới shuffle phần
còn lại để chia split. Manifest ghi nội dung, ID gốc, content ID, fingerprint và
hash. Lệnh không overwrite manifest cũ.

Đây là tái dựng historical set theo **loader hiện tại**, không xác nhận được đó
là đúng 500 ID lịch sử nếu dataset/revision/code server đã đổi. Đối chiếu trước
khi báo kết quả. Hash manifest phát hiện sửa nội dung và overlap; không phải cơ
chế kiểm chứng nguồn dataset từ bên ngoài.

Task khác dùng `--task gsm8k`, v.v.; cần tự đặt `--historical_count` nếu đã xem
kết quả một phần dataset. Nếu loader không đủ mẫu riêng biệt, lệnh báo lỗi,
không lấy calibration/test trùng nhau. Dataset revision upstream chưa được pin
bởi loader cũ; manifest lưu cố định nội dung dùng trong thí nghiệm.

## 4. Profiling qua đêm, có resume

Ví dụ Mode 1. Mode 2 đổi thành `--modes m2 --layers_list ...`; hoặc chạy cả hai
bằng `--modes m1 m2 --layers_list ...`. Layer V mặc định `0 17 35` chỉ là nhóm
đại diện khởi đầu cho Qwen3-4B 36 layer, chưa chứng minh tốt nhất.

```bash
python scripts/profile_adaptive_latent.py --task hotpotqa --sample_manifest snapshots/adaptive/pilot_hotpotqa.json --split calibration --output snapshots/adaptive/cal.jsonl --modes m1 --steps 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 --value_layers 0 17 35 --value_window 4 --warmup 1
python scripts/profile_adaptive_latent.py --task hotpotqa --sample_manifest snapshots/adaptive/pilot_hotpotqa.json --split validation --output snapshots/adaptive/val.jsonl --modes m1 --steps 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 --value_layers 0 17 35 --value_window 4 --warmup 1
```

- Mặc định greedy; thêm `--sampling --seed 42` để dùng temperature .6/top_p .95.
- Dùng `--allow_b_think` nhất quán cho task reasoning nếu muốn B suy nghĩ.
- Chạy lại **đúng lệnh cũ** để resume. Ô successful được bỏ qua; ô error/OOM
  được ghi và thử lại. Nếu còn thiếu ô, chương trình exit 1; analyzer không
  âm thầm bỏ các mẫu khó/OOM khỏi mẫu số.
- Config hash gồm code, dataset manifest, model resolved commit, decoder,
  layer, bước, backend, phiên bản thư viện. Thay cấu hình/code cần output mới.
- File JSONL bị cắt giữa dòng do kill máy sẽ báo rõ dòng hỏng: backup và sửa
  dòng chưa hoàn tất trước resume; không tự xóa dữ liệu người dùng.
- Mỗi ô chạy A lại từ đầu, B đúng một lần. Chưa implement tối ưu reuse một
  rollout cho nhiều checkpoint; do đó không có live cache A được B mutate rồi
  đem tiếp tục. Đổi lại offline profiling tốn nhiều prefill hơn.
- Grid phải chứa **mọi điểm policy có thể dừng**, ví dụ 10,15,20,...80.
  Chỉ profile 0,5,10,20,40,80 thì không thể gán score cho stop ở 15/25/30.
  Analyzer báo lỗi nếu gặp điểm chưa có đáp án B, không nội suy accuracy.
- Warmup không tính vào cell latency. Init model/realignment báo riêng.
  10 mẫu × 17 N × 2 modes = 340 ô cho **một split**; đo smoke trước khi tăng
  lên 100 cal + 100 val. Không cam kết vừa một đêm khi chưa đo server.
- Có `--model`, `--revision` (nên dùng commit SHA), `--device`, `--device_B`,
  `--max_input_length`, `--max_tokens_B` để điều chỉnh tài nguyên.

## 5. Chọn ngưỡng, không training

```bash
python scripts/analyze_adaptive_latent.py --calibration snapshots/adaptive/cal.jsonl --validation snapshots/adaptive/val.jsonl --mode m1 --policy cosine --max_steps 80 --min_steps 10 --interval 5 --patience 2 --budget_fixed_steps 40 --output_policy snapshots/adaptive/cosine_m1.json --report snapshots/adaptive/cosine_m1_report.json
python scripts/analyze_adaptive_latent.py --calibration snapshots/adaptive/cal.jsonl --validation snapshots/adaptive/val.jsonl --mode m1 --policy hidden_value --max_steps 80 --min_steps 10 --interval 5 --patience 2 --budget_fixed_steps 40 --norm_thresholds 0.01 0.05 0.1 --output_policy snapshots/adaptive/hidden_value_m1.json --report snapshots/adaptive/hidden_value_m1_report.json
```

`cosine` dùng distance hidden liên tiếp; log-norm chỉ bật khi đưa threshold.
`hidden_value` thêm novelty V: với mỗi head/layer, lấy **minimum cosine distance**
của V mới tới cửa sổ V latent trước, rồi mean qua head/layer. Không dùng K/RoPE,
không trộn context/sink, không có QR/subspace trong v1. Phải đủ toàn bộ cửa sổ
và norm khác 0; feature tính FP32 trên device từng layer. Norm hidden được đo
**trước realignment**.

Grid threshold trên calibration → shortlist mặc định 3 → chọn ứng viên đáp ứng
ngân sách trên validation. `--time_tolerance` phải định trước, mặc định 0.
Nếu không có ứng viên khả thi thì không xuất policy. Không tự mở rộng grid/tune
holdout để đạt kết quả đẹp. Có thể chỉnh `--cosine_thresholds`,
`--value_thresholds`, `--norm_thresholds` trên calibration rồi chạy lại bằng
tên output mới. Chọn layer/window/interval cũng là hyperparameter cần khóa.

Report gồm fixed frontier, nhóm tăng/giảm/không đổi, all-fail, oracle gold upper
bound, random allocation cùng histogram N và paired bootstrap. Random cùng N
**không đảm bảo cùng thời gian**; report ghi time riêng, chưa phải đối chứng
matched-time đã được xác nhận online. Oracle không phải thuật toán deploy.

Policy xuất ra có nhãn `offline_calibrated_candidate_not_online_validated`.
Replay sử dụng latency fixed **có trace**: chỉ là ước lượng để shortlist, không
được gọi là speedup online. Validation đã dùng để chọn nên CI trên đó chỉ có
tính khảo sát. Nếu không vượt fixed frontier, report chỉ rõ điều đó.

## 6. Đo online trên holdout sau khi khóa policy

Mode 1 ví dụ với cosine_m1.json ở trên:

```bash
python com_latent.py --model_A Qwen/Qwen3-4B --model_B Qwen/Qwen3-4B --device auto --device_B auto --test_task hotpotqa --do_test_latent --batch_size 1 --shift_back --sample_manifest snapshots/adaptive/pilot_hotpotqa.json --sample_split holdout --latent_step_policy cosine --latent_steps 80 --min_latent_steps 10 --latent_check_interval 5 --latent_patience 2 --latent_policy_config snapshots/adaptive/cosine_m1.json --greedy --per_sample_seed --profile_timing --latent_warmup 1
python com_latent.py --model_A Qwen/Qwen3-4B --model_B Qwen/Qwen3-4B --device auto --device_B auto --test_task hotpotqa --do_test_latent --batch_size 1 --shift_back --sample_manifest snapshots/adaptive/pilot_hotpotqa.json --sample_split holdout --latent_step_policy fixed --latent_steps 40 --greedy --per_sample_seed --profile_timing --latent_warmup 1
```

Không bật `--latent_trace` trong comparison online. Lặp fixed N=10/20/40/80;
so full frontier, đặc biệt fixed-40 vốn gần fixed-80 ở kết quả lịch sử. Mode 2
phải thêm **cùng explicit layers** ở cả hai lệnh và dùng policy m2.

So hai file output thực tế (thay đường dẫn bên dưới):

```bash
python scripts/compare_adaptive_runs.py --fixed snapshots/FIXED_RUN/latent_responses.jsonl --adaptive snapshots/ADAPTIVE_RUN/latent_responses.jsonl --metric longbench_f1 --time_tolerance 0.0
```

Tool kiểm tra ID/coverage, prompt token hashes, decoder seed, model, layer và
profile; reject lỗi/thiếu mẫu, trace-enabled timing hoặc điều kiện không khớp.
Task khác truyền key metric trong `item_metrics`. Chạy nhiều lượt trên máy
nhàn với thứ tự đảo xen kẽ; không xem một lượt timing là bằng chứng cuối.
Đối chứng random phân bổ cùng ngân sách thời gian và transfer task là bước
thực nghiệm tiếp theo, không phải kết luận từ unit tests.

## 7. Ý nghĩa log và giới hạn

`latent.steps`, `generated_tokens_A` là N **thực tế**. `adaptive.configured_max_steps`
là cap; có stop_reason, decision_checkpoints, policy hash, raw B token count,
context/cache length, hashes, decoding seed. Response record v2 cũ được giữ,
chỉ thêm `adaptive`/`adaptive_schema_version`/`sample_id`.

`prefill_A_ms`, `latent_A_ms_including_controller` có đồng bộ CUDA khi bật
`--profile_timing`. `controller_host_ms_diagnostic` chỉ là host wall time quanh
thu feature/decision synchronization, **không phải GPU kernel timing độc lập**;
không cộng nó lần nữa vào latent stage. Trace tensors nhỏ copy về CPU sau loop,
không INFO/`.item()` mỗi bước như convergence tracking cũ.

`B_generation_including_handoff_ms` gồm cả route/copy lazy trong cv.forward;
`kv_handoff_ms=null` vì chưa instrument riêng. `end_to_end_ms` tính preparation,
A, controller, handoff, B, decode; không tính chấm điểm, ghi file hay init.
`per_device_peak_allocated_bytes` là peak PyTorch allocated gồm cả weights,
không phải toàn bộ VRAM từ nvidia-smi.

`logical_payload_bytes` là lượng KV route về mặt logic. Slices sink có thể giữ
backing storage đầy đủ, nên `unique_retained_storage_bytes` lớn hơn. Trường
storage đo trên view của source trước handoff, **không phải** peak VRAM của B
sau cross-device copy và không phải lượng network traffic.

Không đồng thời sửa prefill full-logits hay realignment FP32 trên GPU trong
thay đổi này. Hai rủi ro OOM cũ vẫn có thể xảy ra trước khi stopping giúp được;
đối chiếu code server nếu từng có fix riêng. Test CPU không chứng minh model
Qwen3-4B sẽ vừa 4×3080 cho mọi context. Cần chạy smoke GPU và calibration thật
trước khi kết luận controller tăng chất lượng trong cùng thời gian.

## 8. Kết quả kiểm tra local trước cleanup (2026-09-11)

- 55/55 unittest pass bằng `..\venv\Scripts\python.exe -m unittest discover -s tests -v`.
- `compileall` và `git diff --check` pass.
- Git Bash syntax check và dry-run 12 tổ hợp (2 task × 2 mode × 3 N) pass;
  explicit layers được forward cùng `--top_layers 0`, trace đến m1/m2; convergence flag cũ đã được gỡ trong cleanup sau đó.
- Chưa chạy test CUDA/multi-GPU, Qwen3-4B hoặc benchmark dataset thật.

## 9. Kiểm tra sau cleanup A+B+C+G+H (2026-09-11)

- 53/53 unittest pass (4.479s), gồm inference với Qwen3 nhỏ trên CPU,
  fixed/adaptive, full/selected KV, shift-back và attention sink.
- `compileall`, `git diff --check` và `bash -n sweep_latent.sh` pass.
- 21 cấu hình dry-run pass: 16 cấu hình `all` trên hai task với N=0/10/25,
  4 cấu hình adaptive full/selected trên hai task và 1 cấu hình Mode 2 auto.
- Mode 2 manual truyền `--top_layers 0`; auto giữ tỷ lệ đã cấu hình.
  Interpreter lấy từ `--python` hoặc PATH, không còn đường dẫn server cố định.
- `m4`, `m5` và `--track_convergence` bị từ chối rõ ràng; test Python cũng kiểm tra
  các flag đã bỏ bị từ chối trước khi load model.
- Các entry point `com.py`, `com_ms.py`, `com_online.py`, AC và Cipher vẫn import được.
- Giữ nguyên `EXPERIMENT_RESULTS.md`, proposal/research và snapshots.
- Dry-run chỉ kiểm tra tạo lệnh, không chạy benchmark GPU. Chưa xác nhận CUDA,
  multi-GPU hay chất lượng trên toàn bộ HotpotQA sau cleanup.

Chạy lại tests từ thư mục `KVComm` trên Windows:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dry-run bằng Bash/Git Bash:

```bash
bash sweep_latent.sh --tasks 'hotpotqa tmath' --mode all --steps '0 10 25' --layers_list '0 3 7' --latent_trace --dry_run
```
