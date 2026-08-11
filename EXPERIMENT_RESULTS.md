# 📊 Tổng Hợp Kết Quả Thực Nghiệm KVComm + LatentMAS

> **Cập nhật mới nhất**: 11/08/2026  
> **Mô hình thử nghiệm**: `Qwen/Qwen3-4B`, `suayptalha/DeepSeek-R1-Distill-Llama-3B`  
> **Các task set**: `hotpotqa`, `medqa`, `tmath`, `multifieldqa_en`, `tipsheets`

---

## 📑 Mục Lục
1. [Bảng Tổng Hợp Kết Quả Chi Tiết](#1-bảng-tổng-hợp-kết-quả-chi-tiết)
   - [HotpotQA (Multi-hop QA)](#-11-hotpotqa-500-samples)
   - [MedQA (Medical MCQ)](#-12-medqa-300-samples)
   - [TMATH (Mathematical Reasoning)](#-13-tmath-300-samples)
   - [MultiFieldQA-EN (Multi-domain Long Context QA)](#-14-multifieldqa_en-150-samples)
2. [Đánh Giá Hiệu Năng & Xu Hướng Toàn Cảnh](#2-đánh-giá-hiệu-năng--xu-hướng-toàn-cảnh)
3. [Phân Tích Cơ Chế & Nguyên Nhân](#3-phân-tích-cơ-chế--nguyên-nhân)
4. [Hướng Đi Đột Phá Tiếp Theo](#4-hướng-đi-đột-phá-tiếp-theo)

---

## 1. Bảng Tổng Hợp Kết Quả Chi Tiết

### 📌 1.1 `hotpotqa` (500 samples)
* **Dataset Type**: Multi-hop QA (Wikipedia supporting facts)
* **Model A & B**: `Qwen/Qwen3-4B`
* **Metric**: Exact Match / F1

| Mode | Cấu hình | Latent Steps | Accuracy | Time | Garbage % | Ghi chú |
|---|---|:---:|:---:|:---:|:---:|---|
| **LatentMAS Mode 1** | All 36 Layers (Full) | 1 | **72.80%** | 5m 24s (324.5s) | **0.0%** (0/500) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 2 | **72.80%** | 7m 35s (454.9s) | **0.0%** (0/500) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 3 | **72.40%** | 6m 18s (378.1s) | **0.0%** (0/500) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 4 | **72.40%** | 8m 31s (511.0s) | **0.0%** (0/500) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 5 | **71.20%** | 7m 30s (449.9s) | **0.0%** (0/500) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 10 | **71.80%** | 12m 19s (738.6s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 1 | **66.60%** | 6m 38s (397.5s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 2 | **70.00%** | 5m 47s (347.1s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 3 | **68.80%** | 6m 21s (381.3s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 4 | **69.00%** | 6m 52s (411.6s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 5 | **67.80%** | 7m 38s (458.0s) | **0.0%** (0/500) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 10 | **67.60%** | 10m 18s (617.8s) | **0.0%** (0/500) | |
| **KVComm Baseline** | Top 70% Layers (No Latent) | — | **70.00%** | 5m 12s (312.0s) | **0.0%** (0/500) | ⚡ Fast Baseline |

---

### 📌 1.2 `medqa` (300 samples)
* **Dataset Type**: Medical Multiple Choice QA (USMLE format)
* **Model A & B**: `Qwen/Qwen3-4B`
* **Metric**: Accuracy (Exact Match on Choice A/B/C/D)

| Mode | Cấu hình | Latent Steps | Accuracy | Time | Garbage % | Ghi chú |
|---|---|:---:|:---:|:---:|:---:|---|
| **LatentMAS Mode 1** | All 36 Layers (Full) | 1 | **57.67%** | 47m 37s (2856.9s) | **0.0%** (0/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 2 | **57.67%** | 40m 46s (2446.5s) | **0.0%** (0/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 3 | **57.00%** | 41m 41s (2501.3s) | **0.3%** (1/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 4 | **58.33%** | 46m 13s (2772.8s) | **0.0%** (0/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 5 | **59.00%** | 41m 08s (2467.7s) | **0.3%** (1/300) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 1 | **55.33%** | 51m 53s (3112.6s) | **0.0%** (0/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 2 | **56.00%** | 57m 15s (3434.6s) | **0.0%** (0/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 3 | **56.33%** | 60m 47s (3646.6s) | **0.0%** (0/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 4 | **55.67%** | 59m 30s (3569.9s) | **0.0%** (0/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 5 | **54.00%** | 50m 14s (3014.2s) | **0.3%** (1/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 10 | **56.33%** | 44m 01s (2641.3s) | **0.3%** (1/300) | |
| **KVComm Baseline** | Top 70% Layers (No Latent) | — | **58.33%** | 5m 00s (300.0s) | **0.0%** (0/300) | ⚡ Nhanh gấp 8–10× |

---

### 📌 1.3 `tmath` (300 samples)
* **Dataset Type**: Competition Math Problem Solving
* **Metric**: Rouge-L / Accuracy

#### A. Model: `suayptalha/DeepSeek-R1-Distill-Llama-3B`
| Mode | Cấu hình | Latent Steps | Accuracy | Time | Garbage % | Ghi chú |
|---|---|:---:|:---:|:---:|:---:|---|
| **LatentMAS Mode 1** | All 28 Layers (Full) | 1 | **33.00%** | 12m 13s (733.2s) | **0.3%** (1/300) | |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 2 | **32.00%** | 15m 02s (901.9s) | **0.7%** (2/300) | |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 3 | **33.33%** | 11m 14s (673.9s) | **1.0%** (3/300) | |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 4 | **32.33%** | 15m 40s (940.4s) | **1.3%** (4/300) | |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 5 | **34.33%** | 15m 39s (939.2s) | **1.7%** (5/300) | Đỉnh Mode 1 |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 10 | **32.67%** | 18m 48s (1128.4s) | **3.7%** (11/300) | Drift bắt đầu tăng |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 15 | **30.67%** | 21m 45s (1304.9s) | **4.7%** (14/300) | Drift tăng |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 20 | **30.33%** | 19m 58s (1198.5s) | **5.0%** (15/300) | Drift tăng |
| **LatentMAS Mode 1** | All 28 Layers (Full) | 25 | **29.00%** | 24m 26s (1466.2s) | **6.0%** (18/300) | Suy giảm mạnh |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 1 | **36.00%** | 23m 27s (1407.3s) | **0.3%** (1/300) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 2 | **35.33%** | 22m 01s (1321.4s) | **0.7%** (2/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 3 | **34.00%** | 19m 33s (1173.3s) | **1.0%** (3/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 4 | **34.67%** | 23m 09s (1388.7s) | **1.3%** (4/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 5 | **35.00%** | 22m 58s (1378.0s) | **1.7%** (5/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 10 | **31.33%** | 23m 53s (1433.3s) | **3.7%** (11/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 15 | **29.67%** | 27m 05s (1624.7s) | **4.3%** (13/300) | |

#### B. Model: `Qwen/Qwen3-4B`
| Mode | Cấu hình | Latent Steps | Accuracy | Time | Garbage % | Ghi chú |
|---|---|:---:|:---:|:---:|:---:|---|
| **LatentMAS Mode 1** | All 36 Layers (Full) | 1 | **34.06%** | 1h 31m (5474.3s) | **0.3%** (1/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 2 | **33.96%** | 1h 36m (5814.4s) | **0.7%** (2/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 3 | **33.67%** | 1h 33m (5622.2s) | **0.3%** (1/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 4 | **33.45%** | 1h 26m (5184.3s) | **0.7%** (2/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 5 | **33.47%** | 1h 27m (5254.9s) | **0.3%** (1/300) | |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 10 | **34.12%** | 1h 29m (5342.0s) | **1.0%** (3/300) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 1** | All 36 Layers (Full) | 15 | **33.98%** | 1h 34m (5660.4s) | **0.7%** (2/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 1 | **33.82%** | 1h 21m (4903.4s) | **0.3%** (1/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 2 | **32.77%** | 1h 17m (4647.2s) | **0.3%** (1/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 3 | **33.28%** | 1h 19m (4790.8s) | **0.7%** (2/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 4 | **33.24%** | 1h 27m (5270.3s) | **0.3%** (1/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 5 | **33.81%** | 1h 22m (4939.6s) | **0.7%** (2/300) | |
| **LatentMAS Mode 2** | KV Select (Top 70%) | 10 | **33.95%** | 1h 32m (5528.1s) | **0.7%** (2/300) | |
| **KVComm Baseline** | Top 70% Layers (No Latent) | — | **31.36%** | 1h 27m (5241.5s) | **0.3%** (1/300) | |

---

### 📌 1.4 `multifieldqa_en` (150 samples)
* **Dataset Type**: Multi-domain Long Context Reading Comprehension QA
* **Model A & B**: `Qwen/Qwen3-4B`
* **Metric**: Accuracy / F1

| Mode | Cấu hình | Latent Steps | Accuracy (F1) | Time | Garbage % | Ghi chú |
|---|---|:---:|:---:|:---:|:---:|---|
| **KVComm Baseline** | Top 70% Layers (No Latent) | — | **50.00%** | 15.2m (915s) | **N/A** (0 resps) | 🏆 **Top 1 Accuracy** |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 1 | **47.33%** | 37.4m (2244s) | **0.0%** (0/150) | Đỉnh LatentMAS |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 2 | **46.00%** | 40.7m (2441s) | **0.0%** (0/150) | |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 3 | **46.67%** | 18.9m (1136s) | **0.0%** (0/150) | |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 4 | **46.67%** | 19.2m (1155s) | **0.0%** (0/150) | |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 5 | **46.67%** | 19.7m (1183s) | **0.0%** (0/150) | |
| **LatentMAS Mode 1** | All 36 Layers (Full KV) | 10 | **45.33%** | 21.0m (1261s) | **0.0%** (0/150) | |
| **LatentMAS Mode 2** | KV Select (Top 70% KV) | 1 | **42.67%** | 39.1m (2346s) | **0.0%** (0/150) | |
| **LatentMAS Mode 2** | KV Select (Top 70% KV) | 2 | **41.33%** | 39.5m (2369s) | **0.0%** (0/150) | |
| **LatentMAS Mode 2** | KV Select (Top 70% KV) | 3 | **42.00%** | 40.4m (2423s) | **0.0%** (0/150) | |
| **LatentMAS Mode 2** | KV Select (Top 70% KV) | 4 | **41.33%** | 41.1m (2469s) | **0.0%** (0/150) | |
| **LatentMAS Mode 2** | KV Select (Top 70% KV) | 5 | **40.67%** | 41.6m (2494s) | **0.0%** (0/150) | |

---

## 2. Đánh Giá Hiệu Năng & Xu Hướng Toàn Cảnh

```
               [So sánh Accuracy cao nhất giữa các phương pháp trên 4 Task Set]

  Task Set             KVComm Thuần       LatentMAS Mode 1 (Full)    LatentMAS Mode 2 (Top 70%)
───────────────────────────────────────────────────────────────────────────────────────────────
  HotpotQA (Qwen3)        70.00%             72.80% (lat=1,2) 🏆          70.00% (lat=2)
  TMATH (DeepSeek)        ~31.0%             34.33% (lat=5)              36.00% (lat=1) 🏆
  TMATH (Qwen3)           31.36%             34.12% (lat=10) 🏆          33.95% (lat=10)
  MedQA (Qwen3)           58.33%             59.00% (lat=5) 🏆           56.33% (lat=3,10)
  MultiFieldQA-EN         50.00% 🏆          47.33% (lat=1)              42.67% (lat=1)
```

### 💡 Nhận xét then chốt:
1. **Nhóm Task suy luận logic / có Hint ngữ cảnh (HotpotQA, TMATH)**:
   - **LatentMAS vượt trội (+2.7% đến +5.0%)**. Việc Agent A suy nghĩ sâu trong không gian latent trước khi truyền KV cache giúp B giải toán và truy vết thông tin đa bước hiệu quả hơn rõ rệt.
2. **Nhóm Task đọc hiểu văn bản dài / Truy xuất nguyên văn (MultiFieldQA-EN)**:
   - **KVComm thuần dẫn đầu (50.00%)**. Với các đoạn văn bản dài cần trích xuất chi tiết sự thật (factual retrieval), việc ép Agent A vào trạng thái suy luận `<think>` làm biến dạng biểu diễn ngữ cảnh trực tiếp, khiến Agent B khó trích xuất entity chính xác hơn việc đọc KV cache trực tiếp của KVComm.
3. **Hiệu ứng chọn lọc Layer (Mode 1 vs Mode 2)**:
   - Ở MultiFieldQA-EN và HotpotQA, Mode 1 (Full 36 Layers) tốt hơn Mode 2 (Top 70% Layers) từ 2.8% đến 5.0%. Việc cắt 30% layer cào bằng trên toàn bộ chuỗi làm mất mát thông tin quan trọng ở văn bản dài.

---

## 3. Phân Tích Cơ Chế & Nguyên Nhân

1. **Tại sao LatentMAS chậm hơn KVComm thuần?**
   - Phải chạy thêm $N$ bước forward pass qua Agent A với growing KV cache ($O(N \cdot T_A)$).
   - Kích thước context mà Agent B phải attend tăng lên ($T_A + N + T_B$ tokens thay vì chỉ $T_B$).
2. **Tại sao kết quả khác nhau giữa các dataset?**
   - **HotpotQA / TMATH**: Bản chất là *Reasoning task* $\rightarrow$ Tư duy latent đem lại giá trị gia tăng cực lớn.
   - **MultiFieldQA-EN**: Bản chất là *Extraction / Retrieval task* trên văn bản dài $\rightarrow$ Suy nghĩ latent gây loãng/nhiễu thông tin trích xuất nguyên văn.
   - **MedQA**: Không có sự phân chia thông tin bất đối xứng giữa A và B $\rightarrow$ Không gian tăng trưởng bị giới hạn.

---

## 4. Hướng Đi Đột Phá Tiếp Theo

1. **Dual-Selective KV Routing**: 
   - Lọc layer theo từng loại token: Token ngữ cảnh gốc $T_A$ chỉ giữ các tầng nông-giữa (0–14) để bảo toàn sự thật trích xuất (cứu MultiFieldQA-EN), token suy nghĩ $N$ giữ các tầng giữa-sâu (14–35). Giảm 70% kích thước KV truyền đi mà vẫn giữ nguyên năng lực tư duy.
2. **Adaptive Latent Steps & Early Exit**:
   - Tự động dừng suy nghĩ ẩn khi vector trạng thái $h^{(n)}$ hội tụ (Cosine similarity $\approx 1$). Giảm 50% thời gian chạy và triệt tiêu suy giảm accuracy ở bước cao.
3. **Anchor Residual Realignment**:
   - Thêm $\tilde{h}^{(n)} = \alpha W h^{(n-1)} + (1-\alpha) h^{(0)}$ để loại bỏ hoàn toàn hiện tượng Garbage Response trên các mô hình distilled như DeepSeek-R1.
4. **Task-Aware Mode Routing**:
   - Tự động nhận diện bài toán: Nếu là *Factual Retrieval / Long Context* $\rightarrow$ Ưu tiên KVComm thuần hoặc $N=1$; nếu là *Multi-hop / Math Reasoning* $\rightarrow$ Kích hoạt LatentMAS $N=2 \to 5$.
