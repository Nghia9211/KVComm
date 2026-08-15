# Problem Statement (v1): Current Findings & Problems to Be Solved

> **Status**: Draft v1 — 2026-08-15
> **Companion doc**: [proposal_v1.md](proposal_v1.md) — each problem below maps to a proposed direction there.
> **Evidence base**: `EXPERIMENT_RESULTS.md` (updated 2026-08-11); `Qwen/Qwen3-4B`, `suayptalha/DeepSeek-R1-Distill-Llama-3B`; HotpotQA (500), MedQA (300), TMATH (300), MultiFieldQA-EN (150).

---

## 1. Current Findings

### F1 — Latent thinking helps reasoning tasks
LatentMAS (A runs N latent thinking steps before KV transfer) beats plain KVComm where the task requires multi-hop or mathematical reasoning:

| Task | Plain KVComm | Best LatentMAS | Gain |
|---|---:|---:|---:|
| HotpotQA (Qwen3-4B) | 70.00% | **72.80%** (Mode 1, N=1–2) | +2.80 |
| TMATH (DeepSeek-R1-Distill) | ~31.0% | **36.00%** (Mode 2, N=1) | +5.00 |
| TMATH (Qwen3-4B) | 31.36% | **34.12%** (Mode 1, N=10) | +2.76 |
| MedQA (Qwen3-4B) | 58.33% | **59.00%** (Mode 1, N=5) | +0.67 |

### F2 — Latent thinking hurts long-context extraction
On MultiFieldQA-EN (verbatim factual retrieval from long documents), plain KVComm leads: **50.00%** vs. 47.33% for the best LatentMAS config. Forcing A into a `<think>`-style reasoning state distorts the direct context representation B needs to extract entities.

### F3 — Fewer latent steps are better; long loops degenerate
Peak accuracy occurs at N=1–5 on every task. Beyond N≈10, accuracy decays monotonically and garbage responses grow — on DeepSeek-R1-Distill/TMATH: 0.3% garbage at N=1 → 3.7% at N=10 → 6.0% at N=25, with accuracy falling from 33.0% to 29.0%. Distilled models are the most fragile.

### F4 — Uniform layer pruning conflicts with latent transfer
With latent thinking enabled, Mode 1 (all layers) beats Mode 2 (top-70% importance-selected layers) by ~3–5 points on HotpotQA (72.80% vs. 70.00%) and MultiFieldQA-EN (47.33% vs. 42.67%). Cutting 30% of layers uniformly across the whole sequence discards information critical for long inputs. (Exception: TMATH on DeepSeek-R1-Distill, where Mode 2 wins — the pattern is task/model-dependent.)

### F5 — Latent variants pay a heavy latency cost
Plain KVComm on MedQA: 5 min; latent variants: 40–60 min (8–10× slower). Causes: N additional forward passes through A with a growing KV cache (O(N·T_A)), and B must attend over T_A + N + T_B tokens instead of T_B.

### F6 — No gain without information asymmetry
MedQA shows only +0.67 headroom: A and B see the same information, so there is little for the communication channel to add.

---

## 2. Problems to Be Solved

### P1 — Uniform layer selection ignores heterogeneous token roles
**Evidence**: F2, F4.
**Problem**: `layer_importance.py` computes one static layer ranking applied uniformly to the entire transferred sequence. But context tokens (T_A) and latent thinking tokens (N) carry different kinds of information — verbatim facts live in shallow-to-mid layers, reasoning abstractions in mid-to-deep layers. A single ranking cannot serve both, so selection either loses retrieval fidelity (MultiFieldQA-EN) or transfers redundant KV.
**Maps to**: Proposal §3.1 Dual-Selective KV Routing.

### P2 — Latent iteration drifts and destabilizes
**Evidence**: F3.
**Problem**: the latent loop in `models_latent.py` repeatedly projects the last hidden state back into embedding space via the realignment matrix W. Errors compound across iterations, producing representation drift, accuracy decay at high N, and garbage responses — worst on distilled models (DeepSeek-R1-Distill).
**Maps to**: Proposal §3.3 Anchor Residual Realignment.

### P3 — No stopping criterion for latent steps
**Evidence**: F3, F5.
**Problem**: `--latent_steps` is a fixed hyperparameter. There is no signal to stop iterating once the hidden state has converged, so runs either waste compute (N too high, plus degradation from P2) or under-think (N too low). The optimal N also varies by task and model, making manual tuning expensive.
**Maps to**: Proposal §3.2 Adaptive Latent Steps & Early Exit.

### P4 — One fixed communication mode is suboptimal across task types
**Evidence**: F1 vs. F2.
**Problem**: the win/lose pattern is systematic — latent thinking wins on reasoning tasks, plain KVComm wins on extraction tasks — but the framework has no mechanism to detect the task type and route accordingly. Any single default leaves accuracy on the table somewhere.
**Maps to**: Proposal §3.4 Task-Aware Mode Routing.

### P5 — RoPE positional mismatch in KV transfer is only partially solved
**Evidence**: code-level.
**Problem**: transferred KV entries carry rotary position encodings from A's coordinate frame. The `--shift_back` workaround (`models.py:forward_shift_back_llama` / `forward_shift_back_qwen2`) corrects attention-sink-only layers and must always be enabled with latent mode, but it is model-family-specific (raises `NotImplementedError` elsewhere) and the `--latent_only` mode (transfer only the N latent KV tokens, discarding T_A — the most bandwidth-efficient variant) remains unusable due to unresolved RoPE mismatch. A principled position-remapping would unlock much smaller transfers.
**Maps to**: prerequisite for Proposal §3.1 (per-token-type routing changes sequence geometry further).

### P6 — Prompt/template fragility for think-models (secondary)
**Evidence**: fixes labeled "vấn đề 1/3/5" in `eval_latent.py`.
**Problem**: sender/receiver latent-awareness depends on hand-patched chat-template handling (preserving `<think>` for A, prefixing B with a latent-context notice). This is brittle across tokenizers (e.g. `<think>` is multi-token on Llama-based distills) and should be consolidated into a robust, model-agnostic prompt layer.

---

## 3. Open Questions

1. **Scale**: do latent-communication gains grow or shrink with model size (4B → 8B → 70B)?
2. **Heterogeneous pairs**: does KV/latent transfer work when A ≠ B (different sizes or families), given mismatched layer counts and KV geometries?
3. **Pareto frontier**: what is the accuracy-vs-transferred-KV-size trade-off curve, and where does dual-selective routing sit on it?
4. **Convergence signal**: is cosine similarity of successive hidden states a reliable early-exit criterion across tasks, or does it exit prematurely on hard problems?
5. **Asymmetry requirement**: can the framework detect low information asymmetry (F6) upfront and skip communication entirely?

---

## 4. Code-Review Findings (2026-08-15)

> Multi-agent adversarial code review of `models_latent.py`, `models.py`, `eval_latent.py`, `com_latent.py`, focused on the N-step degradation (F3). All findings below were verified against the pinned `transformers==4.53.3` wheel.
> **Headline**: no single bug causes the degradation — compounding OOD drift from feeding hidden states back as embeddings is inherent to LatentMAS — but four real bugs amplify or seed it, and two documented "fixes" are verifiably no-ops.

### 4.1 Bugs that feed the N-step degradation

| # | Bug | Location | Impact |
|---|---|---|---|
| B1 | **Receiver B's reasoning is suppressed**: B's assistant turn is pre-seeded with `</think>\n\nThe answer is: ` (via `eval.apply_chat_template(context=False)`), so B cannot think and depends entirely on A's latent cache. At high N, where latent keys are increasingly OOD, B has no recovery channel → garbage climbs with N (matches 0.3% → 6.0% from N=1 to N=25). | `eval_latent.py:188` | High — directly amplifies F3 |
| B2 | **Double `<think>` on R1-Distill**: `<think>` is appended unconditionally, but R1-Distill chat templates already emit it → A's input ends `<think>\n<think>`, a never-seen state. The corrupted first `last_hidden` is the anchor all N latent steps iterate from. Also: the `convert_tokens_to_ids` None-check is unreliable (fast tokenizers return `unk_token_id`, not None). | `eval_latent.py:174-182` | High for distilled models — explains why DeepSeek-R1-Distill is the most garbage-prone |
| B3 | **Padding unmasked in every latent step**: the loop rebuilds the mask as all-ones over `past_len+1` instead of `cat([attention_mask, ones])`, exposing pad-token KV; corruption compounds per step. Only fires with batch > 1 (current per-sample eval unaffected). | `models_latent.py:307` | Latent for now — will silently corrupt any future batched runs |
| B4 | **Mode 2 + `shift_back=False`: causality violation.** Non-selected layers keep 1 "sink" token; B builds one causal mask at layer-0's full width (T_A+N+T_B); sdpa slices it per layer and the first 1+T_B columns are all-unmasked → B's prompt tokens attend their own *future* tokens in every non-selected layer. `attention_mask` never reaches B (dropped in `CVCommunicator.forward`, `models.py:170-174`), so the "past_mask fix" at `eval_latent.py:234` is dead code. Likely explains part of the Mode 2 < Mode 1 gap if runs used the config default `shift_back=False`. | `models.py:129` | High for Mode 2 results — needs re-verification with `shift_back=True` |

### 4.2 Documented fixes that are no-ops

| # | Claim in code | Reality (verified on transformers 4.53.3) |
|---|---|---|
| N1 | `new_cache._seen_tokens = 0` re-assigns latent positions 0..N-1 for B (`models_latent.py:220`) | No-op: `DynamicCache.get_seq_length()` is shape-based and ignores `_seen_tokens`. In `latent_only` mode the RoPE mismatch stays ~T_A, with A's keys appearing in B's *future*. |
| N2 | "For a full fix (zero mismatch), set `shift_back=True`" (`models_latent.py:193`) | `shift_back` never re-rotates A's cached keys; in Mode 1 its path is bit-identical to stock HF position computation. "Zero mismatch" is unreachable. |
| N3 | `--latent_only` flag controls Mode 1 ablations | Silently dropped: Mode 1 constructs the evaluator without `latent_only=cfg.latent_only` (`com_latent.py:306`), so "latent_only" Mode-1 runs actually transfer the full cache while logs claim otherwise. Also, the documented `--no_latent_kv_select` flag does not exist (argparse only generates `--no_` forms for default-True bools). |

### 4.3 Rejected candidate

A claim that Mode 1 prefill is catastrophically misaligned (`cache_position` starting at 0) was **refuted**: `_supports_cache_class=False` means generation injects no `cache_position`, and B's model recomputes it from the shape-based cache length — plain Mode 1 full-cache transfer is positionally consistent, which is why it works at all.

### 4.4 Fix priority (expected impact on the N-degradation curve)

1. Remove/condition the `</think>` pre-seeding for B (B1) — cheapest test of whether garbage-at-high-N collapses.
2. Guard the `<think>` append against templates that already emit it (B2).
3. Re-run Mode 2 with `shift_back=True` and record which setting each experiment used (B4) — no snapshot logs survive in the repo to check retroactively.
4. Fix the latent-loop padding mask before any batched experiments (B3).
5. Delete or rewrite the `latent_only` / `_seen_tokens` path (N1, N3) — broken *and* silently ignored, which invalidates it as an ablation.

**Implication for the problem statements above**: P2 (latent drift) is partly *measurement artifact* — B1/B2 inflate the apparent drift; the true intrinsic decay curve is unknown until they are fixed. P5 (RoPE mismatch) is worse than documented: both claimed mitigations (N1, N2) are ineffective, and F4's Mode 1 > Mode 2 gap may be partly explained by B4 rather than by information loss from layer pruning.
