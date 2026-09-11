# R1–R10 refactor verification

## Changes

- Removed unused batch length bookkeeping and empty segmented fields from new response records/manifests. Historical files are untouched; readers must treat these retired fields as optional.
- NLTK resources are provisioned only when legacy lemmatization is called, not at module import. Metric formulas are unchanged. Offline users of those legacy metrics still need the same resources installed.
- Extracted latent dispatch from `main()` into `run_latent_evaluation()`. Its body was compared directly with the pre-refactor body and is identical.
- Moved the entire unchanged TextMAS class to `eval_textmas.py`; `eval_latent.TextMASEvaluator` remains a compatibility import.
- Shared tokenizer/model-loading defaults in `utils/model_loading.py`, used by both entry points and the profiler. Model evaluation and initialization order remain owned by callers.
- Reused inference prompt IDs for response logging instead of preparing prompts twice. The two-sample integration test asserts exactly two preparations and checks raw token counts.
- Centralized latent method names while retaining the existing manifest/response aliases. No rename of historical method identifiers.
- Separated controller flags from latent-run flags in Bash. Both remain scoped to m1/m2, including greedy; baseline decoding behavior is unchanged.
- Added missing `defaultdict` import: the new dispatch test exposed an existing cleanup regression in auto selection/layer curves.

## Local verification (2026-09-11)

- Before: 53 tests passed. After: 61 tests passed, including tiny Qwen3 CPU inference and new dispatch contracts.
- Dispatch tests exercise full/manual/random/auto selection, auto/manual layer curves, Skyline, B-only, regular KVComm and TextMAS. These dispatch tests mock models; they do not establish benchmark quality.
- Existing tests cover fixed/cosine/hidden-value stopping, full/selected KV, shift-back/sink, prompt routing, metrics, response records and imports of the other baseline entry points.
- Python compilation, Bash syntax and `git diff --check` pass.
- 20 dry-run configurations passed: 16 fixed configurations across m3/textmas/m1/m2, plus 4 adaptive m1/m2 configurations across two tasks.
- No real Qwen3-4B weights, dataset benchmark, CUDA or multi-GPU test was run locally. A server smoke test remains required before a long sweep.

## Commands

From `KVComm`, Windows PowerShell:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

On the server, with the existing environment activated:

```bash
python -m unittest discover -s tests -v
bash -n sweep_latent.sh
bash sweep_latent.sh --tasks 'hotpotqa tmath' --mode all --steps '0 10 25' --layers_list '0 3 7' --dry_run
```

Then smoke-test real models before a full dataset sweep:

```bash
bash sweep_latent.sh --task hotpotqa --mode all --steps '0 10' --limit 2
```

The inference code hash changes after moving code, including the new TextMAS module. Keep new profiles/runs separate from old profiles; do not resume an old profiling JSONL under a different source hash. Existing policy settings are not automatically recalibrated by this refactor.
