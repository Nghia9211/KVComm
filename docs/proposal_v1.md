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

## 2. Key Findings So Far

> *(This section will be updated once full benchmark results are compiled.)*  
> See current experimental results at [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

## 3. Proposed Directions


### 3.1 Dual-Selective KV Routing
Route KV **per token segment at each layer** instead of pruning the complete
cache of a layer uniformly. The implemented v1 first runs five full-KV,
greedy calibration examples and computes two independent scores:

- `ContextScore[l]`: mean attention mass from B queries to A's original input
  tokens at layer `l` (the sink token is excluded from scoring).
- `LatentScore[l]`: mean attention mass from B queries to A's latent tokens at
  layer `l`.

Each ranking keeps `floor(0.7 * L)` layers. Their overlap is allowed, producing
four real per-layer routes: context+latent, context-only, sink+latent, and
sink-only. The original attention sink is retained at every layer. Cached A
keys keep their original RoPE; B positions continue from the full logical
length `T_A + N`, while causal masks use each layer's physical cache length.

This is Mode 5 (`--segmented_kv_select`). Mode 4 (`--dual_kv_select`) remains a
legacy depth-split baseline: it unions shallow/deep layer lists and transfers
full context+latent KV at every retained layer. V1 is restricted to batch size
1 and matching full-attention Qwen3 architectures (such as Qwen3-4B).

The 70% setting is a fixed layer-count budget per segment, not a 70% byte
reduction. It retains approximately 70% of full KV token positions (plus sink
overhead); direct byte-budget optimization is future work.

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
  1. Segmented Dual-KV improves the quality/communication Pareto frontier over Mode 2 and legacy Mode 4 at the fixed 70%-per-segment setting. A later byte-budgeted version targets ≤ 50% of Mode 1's transferred KV size.
  2. Early exit reduces latent-variant runtime ≥ 40% with no accuracy loss.
  3. Garbage-response rate < 0.5% on distilled models at any step count.
## Reproducibility update: prompt and metric v2

The comparison now separates task semantics from communication modality. All
three methods use the same sender/receiver cores: query-aware evidence
extraction for KVComm QA, shared-problem Planner/Solver for the nine LatentMAS
benchmarks, and native splits for TMATH, RepoBench, and SAMSum. TextMAS alone
inserts decoded text; latent methods transfer internal KV state. This is a
two-agent adaptation, not the official four-agent chain.

New QA runs log continuous LongBench F1 plus `legacy_accuracy`, independent A/B
budgets, actual generated token counts, formatted prompts, and prompt/metric
versions in schema-v2 JSONL. Context-only QA prompt-v1 runs remain immutable and
must not be presented as directly comparable to query-aware prompt-v2 runs.
