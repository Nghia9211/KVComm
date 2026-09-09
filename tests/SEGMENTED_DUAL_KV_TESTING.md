# Kiểm thử Segmented Dual-KV (Mode 5)

Các lệnh dưới đây chạy từ thư mục gốc `KVComm`. Mode 5 hiện chỉ hỗ trợ
`transformers==4.53.3`, A/B cùng kiến trúc Qwen3 full-attention và `batch_size=1`.

## 1. Unit test không cần tải model

```bash
python -m unittest discover -s tests -p "test_segmented_kv.py" -v
```

Kỳ vọng: 8 test `OK`. Các test kiểm tra cách làm tròn floor, hai bảng xếp hạng
độc lập, bốn trạng thái cache, attention-sink, phép tính attention theo chunk và
thống kê retention.

Chạy toàn bộ test hiện có:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Kiểm tra cú pháp Python:

```bash
python -m py_compile com_latent.py eval_latent.py models.py models_latent.py segmented_kv.py utils/response_logging.py tests/test_segmented_kv.py
```

## 2. Kiểm tra lệnh sweep mà chưa tải model

```bash
bash sweep_latent.sh --task "hotpotqa tmath" --mode m5 --steps "5 10" --limit 2 --dry_run
```

Mỗi lệnh in ra phải có `--segmented_kv_select`, `--batch_size 1`,
`--context_top_ratio 0.7`, `--latent_top_ratio 0.7` và `--calib_size 5`.

## 3. GPU smoke test trên server

```bash
bash sweep_latent.sh \
  --task hotpotqa \
  --mode m5 \
  --steps "10" \
  --limit 5 \
  --batch_size 1 \
  --context_top_ratio 0.7 \
  --latent_top_ratio 0.7 \
  --calib_size 5
```

Với Qwen3-4B có 36 layer, log phải báo 25 context layer và 25 latent layer vì
`floor(0.7 * 36) = 25`. Hai danh sách không bắt buộc giống nhau và có thể chồng
lên nhau. Không được thấy Mode 4 tạo danh sách nông/sâu cố định.

Thư mục snapshot phải có:

- `manifest.json`: mode, hai tập layer, thời gian và thống kê KV tổng hợp.
- `segmented_calibration.json`: ContextScore, LatentScore và layer được chọn.
- `latent_responses.jsonl`: kết quả từng mẫu cùng trạng thái routing/chi phí KV.

Năm mẫu calibration được chạy lại trong evaluation, vì vậy chúng xuất hiện đúng
một lần trong metric cuối; calibration chỉ làm tăng thời gian chạy.

## 4. So sánh công bằng Mode 1, Mode 4 và Mode 5

```bash
bash sweep_latent.sh --task hotpotqa --mode m1 --steps "10" --limit 100 --batch_size 1
bash sweep_latent.sh --task hotpotqa --mode m4 --steps "10" --limit 100 --batch_size 1 --split_ratio 0.5 --context_top_ratio 0.7 --latent_top_ratio 0.7
bash sweep_latent.sh --task hotpotqa --mode m5 --steps "10" --limit 100 --batch_size 1 --context_top_ratio 0.7 --latent_top_ratio 0.7 --calib_size 5
```

Giữ nguyên model, seed, prompt/metric version và latent steps. Đánh giá cả metric,
wall-clock time, `logical_retention_ratio` và `actual_tensor_bytes`; không chỉ so
sánh accuracy.

## 5. Dấu hiệu cần dừng và báo lỗi

- Mode 5 chấp nhận `batch_size > 1` thay vì báo `ValueError`.
- Một layer `sink+latent` làm B dùng vị trí RoPE liền sau cache vật lý, thay vì
  tiếp tục từ độ dài logic `T_A + N`.
- `context_layers` hoặc `latent_layers` có 26 layer ở tỉ lệ 0.7/36 (đang dùng
  round thay vì floor).
- Metric cuối đã tính cả response calibration một-token và response evaluation.
- GPU OOM tại `lm_head` do tạo logits cho toàn bộ prompt thay vì token cuối.
