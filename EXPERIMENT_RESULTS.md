# 📊 Kết Quả Thực Nghiệm — KVComm / LatentMAS

> **Cập nhật mới nhất**: 09/09/2026  
> **Mô hình thử nghiệm**: `Qwen/Qwen3-4B` → `Qwen/Qwen3-4B`  
> **Môi trường**: 4× NVIDIA GeForce RTX 3080, `transformers==4.53.3`, `torch 2.9.0`  
> **Seed**: 42 | **Cấu hình generation**: `temperature=0.6`, `top_p=0.95`

---

## 1. HotpotQA (500 samples)

* **Prompt**: Query-Aware Prompt v2 (`kvcomm_qa_query_aware_v2`)  
* **Metric**: `longbench_qa_f1`  
* **Cấu hình model**: `max_tokens_A=256`, `max_tokens_B=48`

| Phương pháp | Mode | Latent Steps | Tỉ lệ giữ KV | F1 Score | Thời gian (500 mẫu) | Thời gian/mẫu |
|---|---|:---:|:---:|:---:|:---:|:---:|
| TextMAS | TextMAS | — | — | **0.7242** | 9290.1s (154.8m) | 18.58s |
| LatentMAS | Mode 1 (Full KV) | 10 | 100% (36/36) | 0.6697 | 778.5s (13.0m) | 1.56s |
| LatentMAS | Mode 1 (Full KV) | 20 | 100% (36/36) | 0.6739 | 1090.4s (18.2m) | 2.18s |
| LatentMAS | Mode 1 (Full KV) | 40 | 100% (36/36) | 0.6816 | 2000.7s (33.3m) | 4.00s |
| LatentMAS | Mode 1 (Full KV) | 80 | 100% (36/36) | 0.6898 | 2955.8s (49.3m) | 5.91s |
| LatentMAS + KVComm | Mode 2 (Top 70% KV) | 10 | 69.4% (25/36) | 0.6691 | 679.7s (11.3m) | 1.36s |
| LatentMAS + KVComm | Mode 2 (Top 70% KV) | 20 | 69.4% (25/36) | **0.6827** | 1014.2s (16.9m) | 2.03s |
| Dual-KV (Legacy) | Mode 4 (Split 0.5) | 10 | 70%+70% | 0.3814 | 873.9s (14.6m) | 1.75s |

---

## 2. MedQA (300 samples)

* **Prompt**: Prompt v1 (cũ — chưa có query-aware)  
* **Metric**: Accuracy (Exact Match on Choice A/B/C/D)  
* **Cấu hình model**: `max_tokens_B=0` (unlimited), `allow_b_think=True`

> ⚠️ Các run MedQA này dùng prompt v1 cũ — chưa có query-aware context. Không so sánh trực tiếp với kết quả HotpotQA pv2.

| Phương pháp | Mode | Latent Steps | Tỉ lệ giữ KV | Accuracy | Thời gian (300 mẫu) |
|---|---|:---:|:---:|:---:|:---:|
| TextMAS | TextMAS | — | — | 0.6767 | 46346.9s (12.9h) |
| LatentMAS | Mode 1 (Full KV) | 10 | 100% (36/36) | 0.6667 | 26822.5s (7.4h) |
| LatentMAS + KVComm | Mode 2 (Top 70% KV) | 10 | 69.4% (25/36) | **0.6867** | 24027.5s (6.7h) |
| Dual-KV (Legacy) | Mode 4 (Split 0.5) | 10 | 70%+70% | 0.6600 | 35074.3s (9.7h) |

---

## 3. TMATH (300 samples)

* **Prompt**: Native Split v1 (`kvcomm_native_split_v1`)  
* **Metric**: `legacy_match` (ROUGE-L recall based)  
* **Cấu hình model**: `max_tokens_A=512`, `max_tokens_B=512`, `allow_b_think=True`

| Phương pháp | Mode | Latent Steps | Tỉ lệ giữ KV | Score | Thời gian (300 mẫu) |
|---|---|:---:|:---:|:---:|:---:|
| TextMAS | TextMAS | — | — | 0.3710 | 21561.3s (5.99h) |
| LatentMAS | Mode 1 (Full KV) | 10 | 100% (36/36) | **0.3864** | 11267.5s (3.13h) |
| LatentMAS + KVComm | Mode 2 (Top 70% KV) | 10 | 69.4% (25/36) | 0.3782 | 10532.6s (2.93h) |
| Dual-KV (Legacy) | Mode 4 (Split 0.5) | 10 | 70%+70% | 0.3751 | 11776.7s (3.27h) |

---

## 4. MultiFieldQA-EN (150 samples)

* **Prompt**: Query-Aware Prompt v2 (`kvcomm_qa_query_aware_v2`)  
* **Metric**: `longbench_qa_f1`  
* **Cấu hình model**: `max_tokens_A=256`, `max_tokens_B=64`

| Phương pháp | Mode | Latent Steps | Tỉ lệ giữ KV | F1 Score | Thời gian (150 mẫu) |
|---|---|:---:|:---:|:---:|:---:|
| TextMAS | TextMAS | — | — | **0.5052** | 2923.6s (48.7m) |

> ⚠️ Chỉ có 1 run MultiFieldQA-EN (TextMAS). Các mode LatentMAS chưa được chạy với prompt v2.

---

## 5. HumanEval+ (164 / ~164 samples)

* **Prompt**: LatentMAS Two-Agent v1 (`latentmas_two_agent_v1`)  
* **Metric**: `legacy_match`  
* **Cấu hình model**: `max_tokens_A=4096`, `max_tokens_B=4096`, `allow_b_think=True`

| Phương pháp | Mode | Latent Steps | Tỉ lệ giữ KV | Score | Thời gian |
|---|---|:---:|:---:|:---:|:---:|
| LatentMAS + KVComm | Mode 2 (Top 70% KV) | 10 | 69.4% (25/36) | **0.6524** | 25782.1s (7.16h) |
| LatentMAS | Mode 1 (Full KV) | 10 | 100% (36/36) | *(đang chạy)* | — |

> ⚠️ Run Mode 1 HumanEval+ đang chạy (snapshot `..._0909_061550`, 120 samples hoàn thành tính đến 09/09/2026).
