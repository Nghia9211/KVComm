# KVComm

Official implementation of the paper [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://openreview.net/forum?id=F7rUng23nw) (ICLR 2026).

A framework for communicating between Large Language Models (LLMs), focusing on how models can effectively share information to improve collaborative reasoning and question-answering performance.

## Installation

```bash
pip install -r requirements.txt
```

Note: Requires `transformers==4.53.3` specifically.

## Datasets

| Dataset           | Task Type             | Description               | Data Path                         |
|-------------------|-----------------------|---------------------------|-----------------------------------|
| `hotpotqa`        | Multi-hop QA          | Wikipedia-based reasoning | HuggingFace                       |
| `qasper`          | Scientific QA         | Paper-based questions     | HuggingFace                       |
| `musique`         | Multi-hop QA          | Compositional reasoning   | HuggingFace                       |
| `multifieldqa_en` | Multi-domain QA       | Cross-field knowledge     | HuggingFace                       |
| `twowikimqa`      | Multi-hop QA          | Wikipedia bridge entities | HuggingFace                       |
| `tipsheets`       | Custom QA             | Synthetic reasoning tasks | `dataloader/data/tipsheets.jsonl` |
| `countries`       | Geographic QA         | Country-based questions   | `dataloader/data/countries.jsonl` |
| `tmath`           | Mathematical          | Math problem solving      | `dataloader/data/TMATH`           |

## Quick Start

### Baseline Test
```bash
python com.py --test_task hotpotqa \
    --do_test_baseline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Skyline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### KVComm Communication
```bash
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.2-3B-Instruct --model_B meta-llama/Llama-3.2-3B-Instruct --top_layers 0.7
```

### Activation Communication
```bash
python com.py \
    --test_task tipsheets \
    --do_test_ac \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --layer_k 26 \
    --layer_j 26 \
    --f replace
```

### Natural Language Debate
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_nld \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### CIPHER Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

## Communication Methods

### 1. KVComm (Cross-View Communication)
- **Mechanism**: Shares key-value cache from model A's specified layers to model B
- **Parameters**: `--layers_list`, `--layer_from`, `--layer_to`, `--top_layers`
- **Use Case**: Efficient information transfer with minimal computational overhead

### 2. Activation Communication (AC)
- **Mechanism**: Injects hidden activations from model A into model B at specific layers
- **Parameters**: `--layer_k` (source), `--layer_j` (target), `--f` (fusion method)
- **Fusion Methods**: `replace`, `sum`, `mean`

### 3. Natural Language Debate (NLD)
- **Mechanism**: Models exchange natural language responses and refine answers
- **Parameters**: `--nld_max_tokens_model_A_and_B_phase1`, `--sender_aware`
- **Process**: Initial responses → Exchange → Refinement

### 4. CIPHER Communication
- **Mechanism**: Models communicate through learned embedding representations
- **Features**: Temperature-controlled generation, nearest neighbor decoding

## Configuration Options

### Model Configuration
- `--model_A`, `--model_B`: Hugging Face model identifiers
- `--device`: CUDA device (default: `cuda:0`)
- `--max_input_length`: Maximum input token length (default: 64000)

### Communication Parameters
- `--layers_list`: Specific layers for KVComm communication
- `--top_layers`: Percentage of top-importance layers to use
- `--layer_k`, `--layer_j`: Source and target layers for AC
- `--f`: Fusion function for AC (`replace`, `sum`, `mean`)

### Evaluation Settings
- `--test_task`: Dataset to evaluate on
- `--limit`: Limit number of evaluation examples
- `--calib_size`: Calibration set size for layer importance

### Experiment Tracking
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity
- `--run_name`: Custom experiment name

## Layer Importance Analysis

The framework includes automatic layer importance detection:

```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3
```

This automatically identifies which layers are most important for communication and selects them for the main evaluation.

---

## LatentMAS Integration (`com_latent.py`)

Extension of KVComm that integrates **LatentMAS** — a latent thinking mechanism where model A compresses its reasoning into latent KV tokens before passing them to model B.

### Architecture Overview

```
                    Model A (Sender)
                         │
               [input tokens: prompt_A]
                         │
                    forward pass
                         │
                   last hidden state
                         │
               ┌─────────▼──────────┐
               │  Latent Loop × N   │  (latent_steps)
               │  realign → embed   │
               │  → forward → h_new │
               └─────────┬──────────┘
                         │
                   DynamicCache
               [T_input + N_latent KV tokens]
                         │
          ┌──────────────▼──────────────────┐
          │    Mode 1          Mode 2        │
          │  (all layers)   (top-k layers)  │
          │                  prepare_key_cache()
          └──────────────┬──────────────────┘
                         │
                    Model B (Receiver)
               [input tokens: prompt_B]
                         │
                    → response
```

### Running Modes

#### Mode 0 — Baseline: B only, no context from A
```bash
python com_latent.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --test_task hotpotqa \
    --do_test_baseline
```

#### Mode 0b — Skyline: A+B both see full context (upper bound)
```bash
python com_latent.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --test_task hotpotqa \
    --do_test_skyline
```

#### Mode 1 — LatentMAS Standalone: latent thinking, all KV layers passed to B
```bash
python com_latent.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --test_task hotpotqa \
    --do_test_latent \
    --latent_steps 5 \
    --shift_back
```

> A runs N latent thinking steps. Full KV cache (all 28 layers) is passed to B directly.

#### Mode 2 — LatentMAS + KVComm: latent thinking + selective KV layer transfer
```bash
python com_latent.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --test_task hotpotqa \
    --do_test_latent \
    --latent_steps 5 \
    --latent_kv_select \
    --top_layers 0.7 \
    --calib_size 5 \
    --shift_back
```

> A runs N latent thinking steps. KV cache is filtered to only the most important layers (top 70% by attention importance) before passing to B.

##### Mode 2 Sub-modes: Layer Selection Strategy

**AUTO** — calibrate importance on N samples, pick top-k% layers:
```bash
    --latent_kv_select --top_layers 0.7 --calib_size 5
```

**MANUAL** — specify exact layer indices:
```bash
    --latent_kv_select --layers_list 8 10 12 13 15 19 20
```

**RANDOM** — random layer subset (ablation):
```bash
    --latent_kv_select --top_layers 0.7 --random_selection
```

#### Mode 3 — Regular KVComm (no latent, for comparison baseline)
```bash
python com_latent.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.7 \
    --calib_size 5 \
    --shift_back
```



### Runinng Thinking Model :
```bash
python com_latent.py --model_A suayptalha/DeepSeek-R1-Distill-Llama-3B --model_B suayptalha/DeepSeek-R1-Distill-Llama-3B --latent_steps 1 --do_test_latent --test_task tmath --device cuda:0 --device_B cuda:1    
```
  

### LatentMAS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | `"cuda:0"` | Primary GPU device (or device for `model_A`). Supports `"cuda:0"`, `"auto"`, etc. |
| `--device_B` | str | `""` | Device for `model_B`. If empty, defaults to `--device`. Supports multi-GPU (e.g. `--device cuda:0 --device_B cuda:1`) |
| `--latent_steps` | int | `5` | Number of latent thinking iterations. Lower = less degeneration (recommended: 5–10) |
| `--latent_space_realign` | flag | `False` | Apply realignment matrix W to project hidden states back to embedding space between steps |
| `--latent_kv_select` | flag | `False` | Enable Mode 2: filter KV cache by layer importance before passing to B |
| `--latent_only` | flag | `False` | ⚠️ Experimental: pass only the N latent KV tokens (discard input tokens). Causes RoPE mismatch — do not use |
| `--calib_size` | int | `1` | Number of calibration samples for layer ranking. Use ≥5 for reliable rankings |
| `--shift_back` | flag | `False` | Fix RoPE position mismatch for attention-sink-only layers. **Always enable with latent** |
| `--top_layers` | float | `0.0` | Fraction of top-importance layers to keep (e.g., `0.7` = keep top 70%) |
| `--layers_list` | int[] | `[-1]` | Manual layer list for MANUAL sub-mode |
| `--random_selection` | flag | `False` | Random layer selection (ablation baseline) |

### Comparison Table

| Mode | Script flag | Latent | KV Select | Expected Score (hotpotqa) |
|------|------------|--------|-----------|--------------------------|
| Baseline (B only) | `--do_test_baseline` | ❌ | ❌ | ~0.30 |
| Skyline (full context) | `--do_test_skyline` | ❌ | ❌ | ~0.65 |
| KVComm (no latent) | `--do_test` | ❌ | ✅ | ~0.60 |
| LatentMAS standalone | `--do_test_latent` | ✅ | ❌ | ~0.35 |
| **LatentMAS + KVComm** | `--do_test_latent --latent_kv_select` | ✅ | ✅ | **TBD** |

### Output Files

Each run creates a timestamped snapshot directory under `snapshots/`:

```
snapshots/
└── llama3.23binstruct-to-llama3.23binstruct_top0.7_lat5_realign_kvsel_MMDD_HHMM/
    ├── log.log          # Run config, calibration results, final scores
    └── responses.jsonl  # Per-sample: prompt_a, prompt_b, response, answer, result_so_far
```

`responses.jsonl` format (one JSON per line):
```json
{
  "idx": 42,
  "prompt_a": "...",
  "prompt_b": "...",
  "response": "...",
  "answer": "...",
  "result_so_far": 0.3571
}
```

### Evaluation Metric

HotpotQA uses **token-level F1** with a 0.5 threshold:

```
result = mean over all samples of f1_match(answer, response)

f1_match = True  (score 1.0)  if word-overlap F1 > 0.5
         = False (score 0.0)  otherwise
```

Words are lowercased, punctuation-stripped, and lemmatized before comparison.
