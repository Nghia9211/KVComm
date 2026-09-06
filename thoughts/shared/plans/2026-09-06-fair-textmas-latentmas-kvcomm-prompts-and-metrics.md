# Fair TextMAS, LatentMAS, and LatentMAS + KVComm Evaluation Plan

## Overview

Refactor the current two-agent evaluation pipeline so that prompt selection follows the benchmark family rather than fragile `hasattr()` fallbacks, TextMAS has an explicit sender budget, and QA results use the official continuous LongBench F1 while retaining the existing thresholded score for historical comparison.

The experiment remains a two-agent A→B system:

- Tasks originally evaluated by LatentMAS use the LatentMAS Planner → Solver prompt family.
- KVComm QA tasks use a query-aware KVComm QA prompt v2: A receives both `prompt_A` (context) and `prompt_B` (target question), while B receives the target question plus the information communicated by A.
- KVComm non-QA tasks keep their native split topology: TMATH uses hint/problem, RepoBench uses code-context/completion, and SAMSum uses content-part-1/content-part-2.
- TextMAS, full-KV LatentMAS, and selective-KV LatentMAS + KVComm use the same task-family semantics. The communication representation is the intended difference.
- No configurable question-blind protocol is introduced. The old context-only QA sender remains identifiable only as prompt v1 in historical results.

## Current State Analysis

### Prompt routing

- `prompts_latent.py` documents both KVComm and LatentMAS task families, but `build_latent_sender_msg()` currently special-cases only TMATH, RepoBench, and SAMSum before sending every other evaluator through the LatentMAS Planner fallback (`prompts_latent.py:119-137`). Consequently, MultiFieldQA-EN, HotpotQA, MuSiQue, 2WikiMQA, Qasper, Tipsheets, and Countries receive the wrong sender role.
- MultiFieldQA-EN stores `prompt_A=context` and `prompt_B=question` (`dataloader/multifieldqa_en.py:14-19`). The current A-only-context flow asks TextMAS to compress a long, query-independent document into 64 tokens before B reveals the question. This is a major information bottleneck and confounds the intended comparison of text versus latent communication.
- MultiFieldQA-EN must use the KVComm QA role rather than treating its context as an input question, but its v2 sender prompt must combine the existing context and target question so A can extract query-relevant evidence.
- The current routing relies on task-specific attributes such as `evaluator.tmath`, `evaluator.medqa`, and `evaluator.aime`. Missing a flag silently routes a task to the default branch.
- Official LatentMAS sequential prompts are defined for a multi-agent Planner/Critic/Refiner/Solver chain. This repository intentionally retains its existing two-agent Planner → Solver simplification; experiment names and documentation must state that distinction.

### TextMAS generation budgets

- `TextMASEvaluator` computes one `effective_max_tokens` value and applies it to both A and B (`eval_latent.py:575-587`). On MultiFieldQA-EN this inherits `evaluator.max_tokens=64`, so A's decoded communication is limited to 64 tokens even though comments describe full textual reasoning.
- `com_latent.py` exposes `max_tokens_B` but no active `max_tokens_A`, although its usage example still mentions the removed/nonexistent `--nld_max_tokens_A` option (`com_latent.py:44-48`).
- `sweep_latent.sh` similarly forwards only `--max_tokens_B` and describes TextMAS as full-CoT without recording an independent sender budget.

### Metrics

- `BaseEvaluator.evaluate_item()` calls `f1_match()` and adds its boolean result to `f1_total` (`dataloader/base_evaluator.py:35-45`).
- `f1_match()` returns only whether normalized F1 is strictly greater than 0.5 (`utils/f1.py:65-102`). Therefore the reported value is thresholded match accuracy, not average F1.
- Official LongBench evaluation assigns `qa_f1_score` to MultiFieldQA-EN, Qasper, HotpotQA, 2WikiMQA, and MuSiQue. Its metric preserves the continuous token-overlap F1 and takes the best score across reference answers.
- Historical experiment documents report the legacy thresholded score, so removing it would make old and new runs difficult to compare.

### Response logging

- Communication, latent, and TextMAS loggers serialize `item.get("answer", "")` (`eval.py:479`, `eval_latent.py:475`, `eval_latent.py:738`). MultiFieldQA-EN stores `answers`, so every logged gold answer is empty even though scoring reads the correct list.
- TextMAS logs raw `prompt_A`/`prompt_B`, not the actual formatted messages passed to the tokenizer, and it does not retain `response_A`. This prevents auditing prompt routing and the transmitted textual message.
- Item and aggregate metrics are represented by ambiguous scalar names (`item_score`, `result_so_far`) with no metric name or schema version.

### Test coverage and working tree

- The repository currently has no dedicated unit-test directory for prompt routing, response-log schema, or QA metrics.
- There are existing uncommitted changes in prompt, evaluator, model, and dataloader files. Implementation must preserve them and make narrowly scoped patches rather than replacing files wholesale.
- Existing snapshot files are experiment artifacts and must not be rewritten or migrated.

## Desired End State

1. Every evaluator exposes stable task metadata used by a single prompt router.
2. The nine LatentMAS benchmark tasks use the LatentMAS prompt family:
   - `gsm8k`, `aime2024`, `aime2025`
   - `arc_easy`, `arc_challenge`, `gpqa`, `medqa`
   - `mbppplus`, `humanevalplus`
3. KVComm tasks keep their task-specific role family, with topology selected by task type:
   - Query-aware QA v2: `hotpotqa`, `multifieldqa_en`, `twowikimqa`, `musique`, `qasper`, `tipsheets`, `countries`. A receives context + target question; B receives target question + A communication.
   - Native math/hint split: `tmath`.
   - Native code-context split: `repobench`.
   - Native split summarization: `samsum`.
4. For a given task and sample, TextMAS, full-KV LatentMAS, and selective-KV LatentMAS + KVComm resolve to the same sender and receiver core instructions. Only the transmitted object and the minimal modality wrapper differ.
5. TextMAS has separate A and B generation budgets. Both values are recorded in run metadata and logs.
6. LongBench-derived English QA tasks report continuous LongBench QA F1 as the primary result and legacy `F1 > 0.5` accuracy as a named secondary result.
7. Logs contain non-empty gold answers, actual formatted prompts, TextMAS sender output, item metrics, aggregate metrics, and token-count metadata.
8. A 10-sample smoke run completes for all three comparison modes before any full benchmark is launched.

## What We're NOT Doing

- Migrating the system from two agents to the official four-agent LatentMAS chain.
- Adding a runtime protocol switch or retaining question-blind QA as a new experiment mode. QA v2 is query-aware; historical context-only runs remain immutable prompt-v1 artifacts.
- Changing KV layer ranking, dual-selective routing, latent realignment, convergence tracking, RoPE handling, or cache geometry.
- Fixing the separate latent-transfer bugs recorded in `docs/problem_v1.md`, unless a small compatibility change is strictly necessary for the prompt/metric work.
- Recomputing, deleting, or modifying historical snapshots.
- Running all 150 MultiFieldQA-EN samples as part of implementation verification.
- Claiming that TextMAS and latent communication have equal information bandwidth; the logs will instead make budgets and transmitted token counts explicit.

## Implementation Approach

Introduce task metadata at the evaluator boundary, then make prompt construction, metric selection, logging, and sweep configuration consume that metadata. Keep task prompt families and sender-input topology independent from communication modes so TextMAS, full-KV LatentMAS, and selective-KV LatentMAS cannot accidentally diverge in the information shown to A or B.

For KVComm QA, construct A's query-aware input from the existing `prompt_A` and `prompt_B` fields without changing the dataset schema. Use backward-compatible secondary metrics and versioned log fields. New QA results must be labeled as prompt version `kvcomm_qa_query_aware_v2` and metric version 2; they must not be compared directly with context-only prompt-v1 results without naming the protocol change.

## Phase 1: Add Explicit Task Metadata and Deterministic Prompt Routing

### Overview

Replace implicit `hasattr()` routing with an explicit evaluator profile while keeping existing evaluator data schemas unchanged.

### Changes Required

#### 1. Base evaluator task profile

**File**: `dataloader/base_evaluator.py`

**Changes**:

- Add documented metadata fields with conservative KVComm-QA defaults:
  - `prompt_family`: `"kvcomm" | "latentmas"`
  - `task_type`: `"qa" | "math" | "code" | "summarization"`
  - `sender_input_mode`: `"query_aware_context" | "shared_problem" | "native_split"`
  - `answer_format`: `"short_text" | "boxed_integer" | "boxed_choice" | "python" | "summary"`
  - `prompt_version`: stable identifier included in every run/log
  - `primary_metric`: default `"legacy_match"`
  - `sender_max_tokens`: task-specific TextMAS default
- Add validation that fails with a clear error for unsupported metadata instead of silently falling through to a generic Planner prompt.

#### 2. Annotate evaluator families

**Files**:

- LatentMAS family: `dataloader/medqa.py`, `dataloader/mbppplus.py`, `dataloader/humanevalplus.py`, `dataloader/aime2024.py`, `dataloader/aime2025.py`, `dataloader/gsm8k.py`, `dataloader/arc_easy.py`, `dataloader/arc_challenge.py`, `dataloader/gpqa.py`
- KVComm special task types: `dataloader/tmath.py`, `dataloader/repobench.py`, `dataloader/samsum.py`
- LongBench QA metric selection: `dataloader/hotpotqa.py`, `dataloader/multifieldqa_en.py`, `dataloader/twowikimqa.py`, `dataloader/musique.py`, `dataloader/qasper.py`

**Changes**:

- Set `prompt_family="latentmas"` only on the nine LatentMAS tasks.
- Set KVComm QA evaluators to `prompt_family="kvcomm"` and `sender_input_mode="query_aware_context"`.
- Set LatentMAS tasks to `sender_input_mode="shared_problem"` and TMATH/RepoBench/SAMSum to `sender_input_mode="native_split"`.
- Set the correct output format for numeric, MCQ, code, short-answer, and summarization evaluators.
- Set `primary_metric="longbench_qa_f1"` only for the LongBench-style QA evaluators; Countries and Tipsheets retain their existing exact/legacy semantics.
- Preserve existing attributes temporarily if other code still consumes them; remove them only in a later cleanup after searches prove they are unused.

#### 3. Refactor prompt builders

**File**: `prompts_latent.py`

**Changes**:

- Split prompt construction into modality-independent cores:
  - `build_sender_core(evaluator, item)`
  - `build_receiver_core(evaluator, item, allow_b_think)`
- KVComm QA family:
  - Introduce a versioned query-aware evidence-extractor prompt that includes both `item["prompt_A"]` as context and `item["prompt_B"]` as target question.
  - Preserve the KVComm reader/answerer roles, but instruct A to identify exact query-relevant evidence rather than produce a generic long-document summary or final answer.
  - Instruct A to preserve entity names, aliases, dates, quantities, locations, definitions, and relations verbatim and to avoid unsupported facts.
  - Keep B's short-answer QA instruction based on `item["prompt_B"]` and require the shortest supported answer without explanation.
- Use this shared semantic template for the query-aware QA sender before chat-template application:

```text
You are an Evidence Extraction Agent.

Read the context and target question carefully. Identify the exact evidence
needed by another agent to answer the target question.

Preserve entity names, aliases, dates, quantities, locations, definitions,
and relationships exactly as stated in the context. Do not invent information.
Do not produce the final answer.

Context:
{prompt_A}

Target Question:
{prompt_B}

Prepare concise evidence for the next agent:
```

- Use this shared receiver core for query-aware QA; TextMAS adds A's decoded evidence in a delimited previous-agent block, while latent modes provide A's information through KV state:

```text
You are the final Answering Agent.

Use the information provided by the previous agent to answer the target
question. Do not invent unsupported information.

Target Question:
{prompt_B}

Return only the shortest answer that fully answers the question.
Do not provide explanations.
```

- KVComm non-QA family:
  - Reuse the original `eval.py` sender instructions and `COMMUNICATION_*` templates.
  - Preserve TMATH hint/problem, RepoBench context/completion, and SAMSum split-content topology.
- LatentMAS family:
  - Preserve Planner → Solver semantics and task-specific output formatting used by the LatentMAS task set.
  - Keep the same original problem/question visible to A and B, as already provided by their dataloaders.
- Build modality wrappers separately:
  - TextMAS inserts decoded A output under an explicit previous-agent section.
  - LatentMAS tells B that previous-agent information is available through transferred internal state without duplicating task instructions.
- Ensure the formatted A prompt is byte-for-byte identical across TextMAS, full-KV LatentMAS, and selective-KV LatentMAS for the same QA item before chat-template application; method-specific behavior begins only at generation/communication.
- Ensure full-KV and selected-KV modes call the same latent prompt functions.
- Delete the broad `else -> Planner` behavior. Unknown metadata must raise an actionable exception.

### Success Criteria

#### Automated Verification

- [ ] A table-driven unit test maps all 19 supported task names to the expected prompt family, task type, and answer format.
- [ ] MultiFieldQA-EN sender prompt contains both its context and exact target question, and does not label the context itself as `Question:`.
- [ ] MultiFieldQA-EN receiver prompt contains `prompt_B` and does not contain the original context text.
- [ ] TextMAS, full-KV LatentMAS, and selective-KV LatentMAS produce the same MultiFieldQA-EN sender core and expose the same target question to A.
- [ ] HotpotQA, 2WikiMQA, MuSiQue, Qasper, Tipsheets, and Countries follow the same query-aware QA rule.
- [ ] TMATH, RepoBench, and SAMSum retain their native split input topology and do not inherit the QA query-aware template.
- [ ] MedQA/AIME/GSM8K/ARC/GPQA/code-generation tasks retain LatentMAS Planner/Solver wording and output constraints.
- [ ] Full-KV and selective-KV latent modes resolve identical formatted A/B messages for the same item.
- [ ] `python -m py_compile` succeeds for the changed Python modules.

#### Manual Verification

- [ ] Inspect one rendered A/B prompt from MultiFieldQA-EN, HotpotQA, TMATH, MedQA, and MBPP+.
- [ ] Confirm the MultiFieldQA-EN sender prompt exposes the question but requests evidence rather than a final answer.
- [ ] Confirm experiment labels say `two-agent` rather than claiming exact four-agent paper reproduction.

**Implementation Note**: Pause after this phase for manual confirmation of the four representative prompt pairs before changing metrics or running model inference.

---

## Phase 2: Separate TextMAS Sender and Receiver Budgets

### Overview

Give TextMAS an explicit communication budget and stop inheriting B's short-answer budget for A.

### Changes Required

#### 1. Evaluator configuration

**File**: `eval_latent.py`

**Changes**:

- Add `max_tokens_A` to `TextMASEvaluator.__init__()`.
- Resolve A budget as:
  - CLI override when `max_tokens_A > 0`.
  - Otherwise `evaluator.sender_max_tokens`.
- Continue resolving B independently from `max_tokens_B` or `evaluator.max_tokens`.
- Maintain separate `generate_args_A` and `generate_args_B` dictionaries.
- Record both effective budgets in initialization logs.
- Preserve task-specific thinking behavior and response cleanup; do not reuse B's answer parser for A's communication if it would strip useful sender text.
- Return an inference result object containing `response_A`, `response_B`, and token counts rather than returning only B's string.

#### 2. CLI configuration

**File**: `com_latent.py`

**Changes**:

- Add `max_tokens_A: int = 0` to `LatentAlignConfig`.
- Pass it only to TextMAS.
- Correct stale usage examples that reference `--nld_max_tokens_A`.
- Log effective A/B budgets and prompt family at run startup.

#### 3. Sweep support

**File**: `sweep_latent.sh`

**Changes**:

- Add `--max_tokens_A` parsing, help text, config output, and TextMAS forwarding.
- Set `ALLOW_B_THINK="auto"` as the actual default so behavior matches comments and help text.
- Keep B thinking disabled for short QA/extraction tasks and enabled for reasoning/code tasks unless explicitly overridden.
- Stop describing TextMAS as “same thinking budget” unless the logged configuration actually matches the intended experiment.

### Success Criteria

#### Automated Verification

- [ ] TextMAS A and B generation calls receive different configured budgets in a mocked unit test.
- [ ] `--max_tokens_A 256 --max_tokens_B 64` parses and reaches `TextMASEvaluator` correctly.
- [ ] A zero CLI override resolves to evaluator defaults.
- [ ] `bash -n sweep_latent.sh` succeeds.
- [ ] `bash sweep_latent.sh --mode textmas --task multifieldqa_en --limit 10 --max_tokens_A 256 --no_b_think --dry_run` prints the expected command.

#### Manual Verification

- [ ] Confirm the dry-run summary clearly shows A budget, B budget, task prompt family, and B thinking mode.

**Implementation Note**: Pause after this phase for manual confirmation of the generated command and effective budgets.

---

## Phase 3: Add Official LongBench QA F1 and Preserve the Legacy Metric

### Overview

Make continuous LongBench QA F1 the primary metric for LongBench QA tasks while retaining the old thresholded result under an explicit legacy name.

### Changes Required

#### 1. Official-compatible QA metric utility

**New file**: `utils/longbench_metrics.py`

**Changes**:

- Implement LongBench-compatible English answer normalization:
  - lowercase;
  - remove punctuation;
  - remove English articles;
  - normalize whitespace.
- Compute token F1 with `collections.Counter`, preserving repeated-token counts.
- Compute the maximum score across all reference answers.
- Keep this implementation self-contained and deterministic; do not add runtime downloads or new dependencies.

#### 2. Multi-metric evaluator state

**Files**: `dataloader/base_evaluator.py`, task-specific evaluator overrides as required

**Changes**:

- Replace ambiguous internal-only `f1_total` semantics with named metric accumulators while retaining compatibility properties if existing custom evaluators rely on them.
- For `primary_metric="longbench_qa_f1"`, accumulate:
  - `longbench_f1`: continuous value in `[0, 1]`;
  - `legacy_match`: existing boolean `f1_match` result.
- Have `evaluate_item()` return named per-item metrics so callers do not infer item score by subtracting mutable totals.
- Have `get_results()` return all aggregate metrics and `get_result()` return the configured primary metric for compatibility with existing progress bars/W&B code.
- Verify custom metrics for TMATH, SAMSum, RepoBench, MCQ, math, and code execution are unchanged.

#### 3. Metric naming in outputs

**Files**: `eval.py`, `eval_latent.py`, `com_latent.py`

**Changes**:

- Display the primary metric name in progress bars and final logs.
- Log both QA metrics to W&B using distinct keys.
- Add `metric_version="v2"` to run metadata/run naming for results using the new primary metric.
- Never label the legacy boolean aggregate simply as F1.

### Success Criteria

#### Automated Verification

- [ ] Exact match produces LongBench F1 `1.0` and legacy match `1`.
- [ ] No overlap produces `0.0` and `0`.
- [ ] Partial overlap preserves a fractional LongBench F1 rather than rounding to 0/1.
- [ ] Repeated tokens use multiset/`Counter` semantics.
- [ ] Multiple gold answers use the maximum per-reference score.
- [ ] The Margaret Way regression fixture yields fractional LongBench F1 while retaining legacy match `1`, demonstrating why both fields exist.
- [ ] Non-QA evaluator tests confirm their primary results do not change.

#### Manual Verification

- [ ] Compare the local metric against the official LongBench implementation on at least five fixed prediction/reference pairs.
- [ ] Confirm historical results are described as `legacy_accuracy`, not compared directly with new `longbench_f1` values.

**Implementation Note**: Pause after metric fixtures pass and the five official-comparison cases are reviewed.

---

## Phase 4: Version and Enrich Response Logs

### Overview

Make each experiment record independently auditable without modifying old snapshots.

### Changes Required

#### 1. Shared response record builder

**New file**: `utils/response_logging.py`

**Changes**:

- Add a shared record builder used by Communication, LatentCommunication, and TextMAS evaluators.
- Normalize gold references as `answers: list[str]`, falling back to `[item["answer"]]` for single-answer datasets.
- Emit a versioned schema containing at least:
  - `schema_version`, `idx`, `task`, `method`, `prompt_family`;
  - raw `prompt_A`, raw `prompt_B`;
  - actual formatted `model_A_prompt`, `model_B_prompt`;
  - `response_A` for TextMAS, otherwise `null` or omitted with an explicit communication-type field;
  - final `response`;
  - `answers`;
  - named `item_metrics` and `aggregate_metrics`;
  - effective A/B token budgets and actual generated token counts;
  - latent steps, layer-selection mode, and selected layers when applicable.
- Keep legacy scalar fields only if a downstream script demonstrably depends on them; if retained, document them as compatibility aliases.

#### 2. Integrate all evaluation loops

**Files**: `eval.py`, `eval_latent.py`

**Changes**:

- Route all three relevant logging sites through the shared builder.
- Pass formatted prompts and structured inference outputs through the evaluation loop.
- Avoid logging hidden-state tensors or KV payloads.
- Keep append behavior but ensure snapshot/run names are unique enough that incompatible schemas are not mixed.

### Success Criteria

#### Automated Verification

- [ ] A MultiFieldQA-EN fixture logs a non-empty `answers` list.
- [ ] A single-answer task is normalized to a one-element list.
- [ ] TextMAS records `response_A`, actual A/B prompts, configured budgets, and generated token counts.
- [ ] Latent modes record latent steps and selected layer metadata without serializing tensors.
- [ ] Every JSONL test line round-trips through `json.loads`.
- [ ] Schema v2 records never contain an ambiguous empty `answer` caused solely by plural dataset schema.

#### Manual Verification

- [ ] Inspect one JSONL entry from TextMAS, LatentMAS-full, and LatentMAS+KVComm and confirm their prompt cores match for the same fixture.

**Implementation Note**: Pause after the three sample records are manually compared.

---

## Phase 5: Make Sweeps Reproducible and Run the 10-Sample Smoke Comparison

### Overview

Update experiment naming/documentation and validate the full workflow on ten samples only.

### Changes Required

#### 1. Run naming and configuration manifest

**Files**: `com_latent.py`, `sweep_latent.sh`, `utils/utils.py`

**Changes**:

- Include method, prompt family/profile version, metric version, A/B budgets, latent steps, and layer-selection mode in the run name or a machine-readable manifest stored beside the JSONL.
- Ensure TextMAS, Mode 1, Mode 2, and Mode 4 names cannot collide.
- Log seed, temperature, top-p, sampling mode, model identifiers, task, limit, and selected layers.
- Use the same seed and B decoding configuration for comparisons. When stochastic decoding remains enabled, label a single-seed run as a smoke test rather than a final result.

#### 2. Documentation

**Files**: `README.md`, `docs/problem_v1.md`, `docs/proposal_v1.md`

**Changes**:

- Document the two prompt families, sender-input modes, and their task membership.
- State that the implementation is a two-agent adaptation, not the official four-agent TextMAS/LatentMAS chain.
- Document metric v1 versus v2 and warn against direct numeric comparison.
- Replace stale TextMAS CLI examples with `--max_tokens_A`/`--max_tokens_B` examples.
- Describe KVComm QA as query-aware v2, LatentMAS benchmark tasks as shared-problem tasks, and TMATH/RepoBench/SAMSum as native split tasks.
- Document that prompt-v1 context-only QA scores measure a different information topology and are not directly comparable to prompt-v2 results.

#### 3. Ten-sample smoke run

**Files produced**: new snapshot directories only

**Commands/sequence**:

1. Run TextMAS on MultiFieldQA-EN with `--limit 10`, explicit A budget, automatic/no B thinking for short QA.
2. Run full-KV LatentMAS on the same ten deterministic samples and seed.
3. Run selective-KV LatentMAS + KVComm on the same ten samples, seed, latent steps, and B decoding configuration.
4. Compare manifests, prompt hashes/strings, logs, generated-token counts, and both metrics.

The exact GPU commands should be generated by `sweep_latent.sh --dry_run` first and manually approved before execution.

### Success Criteria

#### Automated Verification

- [ ] Unit tests for prompt routing, budgets, metrics, and logging pass.
- [ ] `python -m py_compile` passes for all changed Python files.
- [ ] `bash -n sweep_latent.sh` passes.
- [ ] Dry-run commands for the three modes use the same task, sample limit, seed, model pair, B budget, and prompt-profile version.
- [ ] Dry-run/manifests prove all three methods use `kvcomm_qa_query_aware_v2` and show the exact target question in A's formatted prompt.
- [ ] Each completed smoke run contains exactly 10 valid schema-v2 records with non-empty gold answers.
- [ ] Recomputed metrics from each JSONL match its logged final aggregates.

#### Manual Verification

- [ ] Review the first three TextMAS sender messages for relevance and truncation.
- [ ] Confirm the A/B prompt cores are equivalent across the three methods for corresponding samples.
- [ ] Confirm no historical snapshot was modified.
- [ ] Approve the configuration before starting a full 150-sample run.

**Implementation Note**: Stop after the ten-sample comparison and wait for human review before any full benchmark sweep.

---

## Testing Strategy

### Unit Tests

Create a lightweight `tests/` suite that does not load Hugging Face models or datasets:

- `tests/test_prompt_routing.py`
  - table-driven task-family coverage;
  - query-aware KVComm QA routing;
  - native split routing for TMATH/RepoBench/SAMSum;
  - LatentMAS Planner/Solver routing;
  - unsupported-profile failure;
  - full-KV/selected-KV prompt parity.
- `tests/test_longbench_metrics.py`
  - normalization, repeated tokens, partial overlap, multiple references, legacy comparison.
- `tests/test_response_logging.py`
  - plural/singular answer normalization;
  - schema-version fields;
  - TextMAS sender output and token counts;
  - JSON serialization.
- `tests/test_textmas_budgets.py`
  - independent A/B generation args using mocked models/tokenizer.

### Integration Tests

- Construct one synthetic evaluator/item for each prompt family and pass it through prompt preparation without model generation.
- For KVComm QA, assert the same context and target question reach A in TextMAS, full-KV LatentMAS, and selective-KV LatentMAS.
- Use a fake tokenizer/model to verify formatted prompts and generation arguments reach A and B correctly.
- Recompute aggregate metrics from temporary JSONL and compare them with evaluator state.
- Run CLI/sweep dry runs for TextMAS, Mode 1, and Mode 2.

### Manual Testing Steps

1. Review representative rendered prompts before inference.
2. Generate dry-run commands and verify parity fields.
3. Execute only the ten-sample MultiFieldQA-EN smoke comparison.
4. Inspect sender output, final response, gold answers, and named metrics.
5. Recompute metrics from logs.
6. Decide whether to proceed to the full benchmark separately.

## Performance Considerations

- Adding the target question to A adds only a small number of input tokens relative to the long context, but changes the task from query-independent compression to query-conditioned evidence extraction. Track this through `prompt_version` rather than attributing all gains to the communication channel.
- Increasing TextMAS A from 64 to a larger explicit budget increases runtime and decoded-token cost. Report the effective budget and actual generated tokens rather than describing it as directly compute-matched to latent steps.
- Logging formatted prompts duplicates long contexts and can substantially increase snapshot size. If disk growth becomes material, store prompt hashes plus a run-level prompt template and retain full formatted prompts for the ten-sample smoke run; do not silently omit audit data.
- The new metric and record-building operations are negligible compared with model generation.
- No additional model forward passes should be introduced by this work.

## Migration and Compatibility Notes

- Historical JSONL files remain schema v1 and continue to contain the old `answer`/`item_score` fields. New analysis code must branch on `schema_version`.
- Historical KVComm QA runs used a context-only A input. Preserve them as `prompt_version=v1` in documentation/analysis metadata where inferable; never rewrite their JSONL files.
- New query-aware QA runs use `prompt_version=kvcomm_qa_query_aware_v2`. Any reported improvement must separate prompt-topology gains from full-KV versus selected-KV gains.
- Existing documented percentages are legacy threshold accuracy. New continuous LongBench F1 results must use distinct labels and a metric-v2 run tag.
- Preserve `get_result()` during the transition so unrelated callers continue to receive a scalar primary result.
- Preserve current evaluator flags until all uses have been migrated and verified; metadata becomes the source of truth for prompt and metric routing.
- Make edits on top of the dirty worktree. Do not reset or overwrite the user's current prompt, model, dual-routing, or convergence changes.

## References

- Current prompt routing: `prompts_latent.py`
- TextMAS and latent evaluators: `eval_latent.py`
- KVComm templates/evaluation loop: `eval.py`
- Entrypoint/configuration: `com_latent.py`
- Sweep orchestration: `sweep_latent.sh`
- Evaluator base and task loaders: `dataloader/base_evaluator.py`, `dataloader/*.py`
- Existing research context: `docs/problem_v1.md`, `docs/proposal_v1.md`
- Official LongBench evaluator: https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py
- Official LongBench metrics: https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
- Official LatentMAS prompts: https://github.com/Gen-Verse/LatentMAS/blob/main/prompts.py
