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

#### Mode 4 — Legacy Dual-KV depth split

Mode 4 is retained for comparison with earlier experiments. It chooses shallow
and deep layer groups, unions them, and transfers the full context+latent cache
at each retained layer. It does not route the two token segments separately.

#### Mode 5 — Segmented Dual-KV

```bash
python com_latent.py \
    --model_A Qwen/Qwen3-4B \
    --model_B Qwen/Qwen3-4B \
    --test_task hotpotqa \
    --do_test_latent \
    --segmented_kv_select \
    --latent_steps 10 \
    --context_top_ratio 0.7 \
    --latent_top_ratio 0.7 \
    --calib_size 5 \
    --batch_size 1 \
    --shift_back
```

Mode 5 independently ranks context and latent attention mass, keeps
`floor(ratio × number_of_layers)` layers for each segment, and preserves the
original attention sink everywhere. V1 requires matching full-attention Qwen3
architectures and `batch_size=1`.



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
| `--no_latent_space_realign` | flag | disabled | Disable the default realignment matrix W between latent steps |
| `--latent_kv_select` | flag | `False` | Enable Mode 2: filter KV cache by layer importance before passing to B |
| `--dual_kv_select` | flag | `False` | Enable legacy Mode 4 depth-split layer selection |
| `--segmented_kv_select` | flag | `False` | Enable Mode 5 independent context/latent segment routing |
| `--calib_size` | int | `5` | Number of calibration samples for layer ranking |
| `--shift_back` | flag | `False` | Fix RoPE position mismatch for attention-sink-only layers. **Always enable with latent** |
| `--top_layers` | float | `0.0` | Fraction of top-importance layers to keep (e.g., `0.7` = keep top 70%) |
| `--context_top_ratio` | float | `0.7` | Mode 5 fraction of layers retaining context KV |
| `--latent_top_ratio` | float | `0.7` | Mode 5 fraction of layers retaining latent KV |
| `--layers_list` | int[] | `[-1]` | Manual layer list for MANUAL sub-mode |
| `--random_selection` | flag | `False` | Random layer selection (ablation baseline) |

### Comparison Table

Evaluated on `Qwen/Qwen3-4B → Qwen/Qwen3-4B`, `seed=42`, `temperature=0.6`, `top_p=0.95`. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for full details.

#### HotpotQA (500 samples, `longbench_qa_f1`, prompt v2)

| Mode | Script flag | Latent Steps | KV Select | F1 Score | Time (500 samples) |
|------|------------|:---:|-----------|:---:|:---:|
| TextMAS | `--do_test_nld` | — | ❌ | **0.7242** | 9290s |
| Mode 1 (Full KV) | `--do_test_latent` | 10 | ❌ | 0.6697 | 779s |
| Mode 1 (Full KV) | `--do_test_latent` | 20 | ❌ | 0.6739 | 1090s |
| Mode 1 (Full KV) | `--do_test_latent` | 40 | ❌ | 0.6816 | 2001s |
| Mode 1 (Full KV) | `--do_test_latent` | 80 | ❌ | 0.6898 | 2956s |
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 10 | ✅ | 0.6691 | 680s |
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 20 | ✅ | 0.6827 | 1014s |
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 40 | ✅ | 0.6897 | 1804s |
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 80 | ✅ | **0.6928** | 3071s |
| Mode 4 (Dual KV Legacy) | `--do_test_latent --dual_kv_select` | 10 | ✅ (whole layer) | 0.3814 | 874s |
| **Mode 5 (Segmented Dual-KV, Top 70%)** | `--do_test_latent --segmented_kv_select` | 10 | ✅ (per segment) | 0.5707 | 1343s |
| **Mode 5 (Segmented Dual-KV, Full 100%)** | `--do_test_latent --segmented_kv_select` | 10 | ❌ (all layers) | 0.6739 | 1225s |

#### TMATH (300 samples, `legacy_match`, prompt v1)

| Mode | Script flag | Latent Steps | KV Select | Score | Time (300 samples) |
|------|------------|:---:|-----------|:---:|:---:|
| TextMAS | `--do_test_nld` | — | ❌ | 0.3710 | 21561s |
| **Mode 1 (Full KV)** | `--do_test_latent` | 10 | ❌ | **0.3864** | 11268s |
| Mode 2 (KV Top 70%) | `--do_test_latent --latent_kv_select` | 10 | ✅ | 0.3782 | 10533s |
| Mode 4 (Dual KV Legacy) | `--do_test_latent --dual_kv_select` | 10 | ✅ (whole layer) | 0.3751 | 11777s |

#### MedQA (300 samples, Accuracy, prompt v1)

| Mode | Script flag | Latent Steps | KV Select | Accuracy | Time (300 samples) |
|------|------------|:---:|-----------|:---:|:---:|
| TextMAS | `--do_test_nld` | — | ❌ | 0.6767 | 46347s |
| Mode 1 (Full KV) | `--do_test_latent` | 10 | ❌ | 0.6667 | 26823s |
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 10 | ✅ | **0.6867** | 24028s |
| Mode 4 (Dual KV Legacy) | `--do_test_latent --dual_kv_select` | 10 | ✅ (whole layer) | 0.6600 | 35074s |

#### MultiFieldQA-EN (150 samples, `longbench_qa_f1`, prompt v2)

| Mode | Script flag | Latent Steps | KV Select | F1 Score | Time (150 samples) |
|------|------------|:---:|-----------|:---:|:---:|
| TextMAS | `--do_test_nld` | — | ❌ | **0.5052** | 2924s |

#### HumanEval+ (164 samples, `legacy_match`, prompt v1)

| Mode | Script flag | Latent Steps | KV Select | Score | Time |
|------|------------|:---:|-----------|:---:|:---:|
| **Mode 2 (KV Top 70%)** | `--do_test_latent --latent_kv_select` | 10 | ✅ | **0.6524** | 25782s |
| **Mode 1 (Full KV)** | `--do_test_latent` | 10 | ❌ | **0.6524** | 26938s |

### Output Files

Each run creates a timestamped snapshot directory under `snapshots/`:

```
snapshots/
└── llama3.23binstruct-to-llama3.23binstruct_top0.7_lat5_realign_kvsel_MMDD_HHMM/
    ├── log.log                       # Run config and final score
    ├── manifest.json                 # Reproducibility metadata and aggregate stats
    ├── segmented_calibration.json    # Mode-5 scores and selected layer sets
    └── latent_responses.jsonl        # Per-sample output and routing statistics
```

`latent_responses.jsonl` format (one schema-v2 JSON object per line):
```json
{
  "schema_version": "v2",
  "idx": 42,
  "method": "latentmas_segmented_dual_kv",
  "response": "...",
  "answers": ["..."],
  "item_metrics": {"longbench_qa_f1": 0.75},
  "latent": {
    "steps": 10,
    "layer_selection_mode": "segmented",
    "context_layers": [0, 1],
    "latent_layers": [1, 2],
    "segmented_stats": {"byte_retention_ratio": 0.70}
  }
}
```

### Evaluation Metric

HotpotQA uses continuous **LongBench token-level F1**:

```
result = mean over all samples of token_overlap_f1(answer, response)
```

Each sample contributes a fractional score from 0 to 1. The old thresholded
score remains available as `legacy_accuracy`, but it is not the primary metric.
## Evaluation protocol v2 (two-agent adaptation)

New runs use explicit evaluator metadata instead of task-flag fallbacks. This is
a two-agent Planner/Evidence-Extractor -> Solver adaptation; it is not the
official four-agent LatentMAS chain.

- LatentMAS shared-problem tasks: `gsm8k`, `aime2024`, `aime2025`, `arc_easy`,
  `arc_challenge`, `gpqa`, `medqa`, `mbppplus`, and `humanevalplus`.
- Query-aware KVComm QA v2: `hotpotqa`, `multifieldqa_en`, `2wikimqa`,
  `musique`, `qasper`, `tipsheets`, and `countries`. Agent A sees both context
  and target question and extracts evidence.
- Native-split KVComm tasks: `tmath` (hint/problem), `repobench`
  (code-context/completion), and `samsum` (dialogue halves).

TextMAS has independent sender and receiver budgets:

```bash
python com_latent.py --do_test_nld --test_task multifieldqa_en --limit 10 \
  --max_tokens_A 256 --max_tokens_B 64
```

For LongBench English QA tasks, schema-v2 logs report continuous
`longbench_f1` as the primary metric and preserve the old thresholded score as
`legacy_accuracy`. Historical prompt-v1/metric-v1 percentages are not directly
comparable because Agent A did not see the target question and the reported
number was threshold accuracy rather than mean F1.
