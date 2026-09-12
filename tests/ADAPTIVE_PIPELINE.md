# Adaptive policy: một lệnh end-to-end

## Chỉnh trực tiếp sweep_latent.sh rồi chạy

`POLICY="cosine"` (mặc định mới) hoặc `POLICY="hidden_value"`, cùng `MODE="m1"`,
tự chạy chia split → profile → chọn policy → holdout khi chưa cung cấp JSON.
Chạy `./sweep_latent.sh`; kiểm tra trước bằng `bash sweep_latent.sh --dry_run`.
Đổi `TASK_INPUT` để chọn task. Các biến `POLICY_CALIBRATION`, `POLICY_VALIDATION`,
`POLICY_HOLDOUT` mặc định 5/5/10; `POLICY_MAX_STEPS=80` là cap, không phải `STEPS`.
Ngưỡng tối thiểu/interval/patience/ngân sách chỉnh bằng các biến `POLICY_*` liền kề.
`POLICY_OUTPUT` rỗng tạo thư mục timestamp mới. Đây vẫn chỉ là pilot nhỏ.

`MODE="m2"` cần `LAYERS_LIST` đã khóa. `MODE="both"` chạy hai pipeline riêng,
cùng task/seed/kích thước split nhưng tạo manifest riêng (không phải một manifest dùng chung).
Nếu cần so sánh chặt chẽ phải đối chiếu ID/manifest hash giữa hai mode.
Không tự chọn layer, không tự nới ngân sách nếu analyzer không tìm được policy.
Các flag evaluation-only và `LIMIT` khác 0 bị từ chối trong automatic pipeline để tránh bị bỏ qua âm thầm.

`POLICY="fixed"` chạy sweep latent thường theo `STEPS`. Nếu truyền JSON bằng
`--latent_policy_config`, cosine/hidden_value chỉ evaluation như trước, không profile lại.
TextMAS/m3 chạy độc lập, bỏ qua controller; `all` với controller adaptive vẫn bị từ chối:
hãy chọn `fixed` cho sweep `all` hoặc `both` cho automatic policy.

## Chạy mode thường, không calibration

```bash
bash sweep_latent.sh --mode textmas --tasks "hotpotqa gsm8k" --limit 2
bash sweep_latent.sh --mode m1 --tasks hotpotqa --steps "10 40"
bash sweep_latent.sh --mode m2 --tasks hotpotqa --layers_list "0 3 7"
bash sweep_latent.sh --mode m3 --tasks hotpotqa
```

`python scripts/run_adaptive_pipeline.py --mode textmas --tasks hotpotqa --limit 2`
cũng chuyển sang sweep thường (cần Bash/Git Bash), không yêu cầu output hay chia split.
Không truyền các flag riêng của pipeline như `--calibration`, `--output`, `--kv_mode`
cho mode thường. TextMAS/m3 bỏ qua controller latent với cảnh báo, không thay decoding.
Thêm `--dry_run` để kiểm tra lệnh trước khi chạy model.

Chạy từ thư mục `KVComm`, trong môi trường Python đã cài dependencies.

## 1. Xem lệnh trước (không tải model, không tạo output)

```bash
bash sweep_latent.sh --mode policy --tasks hotpotqa gsm8k --output snapshots/policy_pilot_01 --dry_run
```

## 2. Chạy thật

```bash
bash sweep_latent.sh --mode policy --tasks hotpotqa gsm8k --output snapshots/policy_pilot_01
```

Windows không có Bash có thể gọi trực tiếp:

```powershell
python scripts/run_adaptive_pipeline.py --mode policy --tasks hotpotqa gsm8k --output snapshots/policy_pilot_01
```

Script lần lượt tạo split → profile calibration/validation → chọn ngưỡng → chạy adaptive và fixed trên holdout → xuất báo cáo so sánh cho từng fixed N.

Mặc định: Qwen3-4B, m1, cosine, greedy, batch=1, 10 calibration + 10 validation + 10 holdout; cap=80, min=10, interval=5, patience=2, ngân sách fixed-40. Fixed references là 0/10/20/40/80. Đây là pilot kiểm tra pipeline, không đủ để kết luận nghiên cứu. Profiling chạy lại A/B cho từng mẫu × checkpoint, có thể tốn nhiều giờ. B-thinking tự bật cho nhóm reasoning giống sweep cũ; có thể khóa bằng `--b_think yes` hoặc `--b_think no` cho tất cả các giai đoạn.

Task presets như `qa`/`all` không áp dụng cho mode policy: hãy truyền tên task thực, cách nhau bằng dấu cách, không gom vào một chuỗi có dấu nháy.

## Tùy chỉnh

```bash
bash sweep_latent.sh --mode policy --tasks hotpotqa --output snapshots/hotpotqa_policy_02 --calibration 100 --validation 100 --holdout 100 --policy hidden_value
```

Mode 2 cần danh sách layer đã chọn trên calibration, không tự động ranking:

```bash
bash sweep_latent.sh --mode policy --tasks hotpotqa --output snapshots/hotpotqa_m2_01 --kv_mode m2 --layers_list 0 3 7
```

Các layer trên chỉ minh họa, không phải khuyến nghị thực nghiệm. Mỗi invocation dùng một mode/layer list; nếu từng task dùng layer khác nhau, chạy riêng từng task. `--value_layers 0 17 35` mặc định phù hợp số layer Qwen3-4B; đổi model phải chọn value layers hợp lệ. `--model` dùng cùng model cho A/B. `--python PATH` được Bash wrapper hỗ trợ. Xem toàn bộ flags bằng `--mode policy --help`.

## Output và an toàn

- Mỗi task có `samples.json`, `calibration.jsonl`, `validation.jsonl`, `policy.json`, `offline_report.json`, thư mục evaluation adaptive/fixed và `comparison_fixed_N.json`.
- Mỗi lần chạy thật yêu cầu output directory mới. Không tự ghi đè, không tự resume toàn pipeline ở phiên bản này. Khi lỗi, script dừng và giữ nguyên artifacts; có thể resume riêng profiler bằng đúng lệnh đã in, hoặc chạy pipeline mới ở thư mục mới. Không trộn output sau khi đổi code/config.
- Không có policy khả thi thì dừng trước holdout, không tự nới ngân sách/tune holdout. Không tự nhận adaptive tốt hơn fixed.
- Historical exclusion giữ quy tắc hiện tại: HotpotQA bỏ 500 mẫu đầu; task khác mặc định 0. Dùng `--historical_count` nếu đã xem dữ liệu trước đó. Lỗi thiếu mẫu/overlap không bị bỏ qua.
- Metric được suy từ primary metric của task (LongBench dùng `longbench_f1`); có thể truyền `--metric` nếu task dùng tên khác. Không so sánh khi model commit thay đổi giữa profiling và evaluation.
- Đây là một lượt đo timing; cần lặp trên cùng server nhàn trước khi công bố speedup. Không chạy TextMAS/Skyline trong pipeline này: các baseline đó vẫn chạy qua sweep/CLI cũ; ở đây đối chứng là fixed latent cùng mode.

## Test không tải model

```bash
python -m unittest discover -s tests -p test_adaptive_pipeline.py -v
python -m unittest discover -s tests -v
```
