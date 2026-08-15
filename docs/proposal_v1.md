# Research Proposal (v1): Adaptive Latent KV-Cache Communication for Multi-Agent LLMs

> **Status**: Draft v1 — 2026-08-15
> **Builds on**: KVComm (ICLR 2026) + LatentMAS integration (`com_latent.py`)
> **Models under study**: `Qwen/Qwen3-4B`, `suayptalha/DeepSeek-R1-Distill-Llama-3B`
> **Benchmarks**: HotpotQA, MedQA, TMATH, MultiFieldQA-EN, Tipsheets

---

## 1. Background

**KVComm** enables two LLM agents to communicate by transferring the sender's (Model A) key-value attention cache directly into the receiver (Model B), instead of exchanging natural-language messages. To keep the transfer efficient, only a subset of layers is shared, selected by an attention-based importance score (calibrated on a few samples and blended with a Gaussian depth prior — see `layer_importance.py`).

The current extension integrates **LatentMAS**: before handing over its KV cache, Model A runs *N* extra "latent thinking" forward passes — its last hidden state is projected back into embedding space via a learned realignment matrix W and fed back as the next input, appending N latent KV tokens to the cache (`models_latent.py`).

Two modes are compared:
- **Mode 1** — latent thinking + full KV cache (all layers) passed to B.
- **Mode 2** — latent thinking + selective layer transfer (KVComm, top-k% layers).

## 2. Key Findings So Far (EXPERIMENT_RESULTS.md, Aug 2026)

1. **Latent thinking helps reasoning tasks.** LatentMAS beats plain KVComm on HotpotQA (72.80% vs. 70.00%) and TMATH (+2.7 to +5.0 points). Few latent steps (1–5) are best; beyond ~10 steps accuracy decays and garbage responses increase (up to 6% on DeepSeek-R1-Distill at 25 steps).
2. **Latent thinking hurts extraction tasks.** On MultiFieldQA-EN (long-context factual retrieval), plain KVComm leads (50.00% vs. 47.33%): forcing A into a reasoning state distorts the verbatim context representation B needs.
3. **Uniform layer pruning conflicts with latent transfer.** With latent thinking on, Mode 1 (full layers) beats Mode 2 (top 70%) by ~3–5 points on HotpotQA and MultiFieldQA-EN — pruning layers uniformly across the whole sequence discards information critical for long inputs.
4. **Latency trade-off.** Plain KVComm is far faster (e.g. MedQA: 5 min vs. 40–60 min for latent variants), since latent variants add N forward passes on A and enlarge the context B must attend to (T_A + N + T_B).

**Core insight**: latent KV communication beats text exchange for reasoning, but the transfer must become *content- and task-adaptive* rather than uniform.

## 3. Proposed Directions

### 3.1 Dual-Selective KV Routing
Route layers **per token type** instead of pruning uniformly:
- Original context tokens (T_A): keep shallow-to-mid layers (~0–14) to preserve factual/verbatim information (fixes MultiFieldQA-EN regression).
- Latent thinking tokens (N): keep mid-to-deep layers (~14–35) where reasoning representations live.

Target: ~70% reduction in transferred KV size while retaining both retrieval fidelity and reasoning capability.

### 3.2 Adaptive Latent Steps & Early Exit
Stop the latent loop automatically when the hidden state converges (cosine similarity of h⁽ⁿ⁾ vs. h⁽ⁿ⁻¹⁾ ≈ 1). Expected: ~50% runtime reduction and elimination of the accuracy decay observed at high step counts.

### 3.3 Anchor Residual Realignment
Stabilize latent iteration with an anchored update:

  h̃⁽ⁿ⁾ = α · W · h⁽ⁿ⁻¹⁾ + (1 − α) · h⁽⁰⁾

to suppress representation drift and eliminate garbage responses on distilled models (DeepSeek-R1-Distill).

### 3.4 Task-Aware Mode Routing
Automatically detect task type and route:
- **Factual retrieval / long context** → plain KVComm or N = 1.
- **Multi-hop / math reasoning** → LatentMAS with N = 2–5.

## 4. Evaluation Plan

- **Baselines**: B-only baseline, full-context skyline, plain KVComm (top 70%), LatentMAS Mode 1/2, NLD, CIPHER.
- **Metrics**: task accuracy (EM/F1/Rouge-L), wall-clock time, transferred-KV size, garbage-response rate.
- **Ablations**: random vs. importance-based layer selection, latent step sweep, per-token-type routing splits, α sweep for anchor realignment.
- **Success criteria**:
  1. Dual-selective routing matches or beats Mode 1 accuracy on reasoning tasks while recovering plain-KVComm accuracy on MultiFieldQA-EN, at ≤ 50% of Mode 1's transferred KV size.
  2. Early exit reduces latent-variant runtime ≥ 40% with no accuracy loss.
  3. Garbage-response rate < 0.5% on distilled models at any step count.
