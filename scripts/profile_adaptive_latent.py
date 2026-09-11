"""Independent fixed-N reference profiler. Run --help before a GPU job.

No checkpoint cache reuse: every cell runs A afresh and B once. This is slower
offline but isolates B's mutable cache. JSONL success cells are resumable.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_latent import PolicyConfig, content_hash, sample_id, inference_code_hash


def append_row(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_completed(path, signature):
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Damaged JSONL line {line_no}; preserve and repair it before resume") from exc
                if row["config_hash"] != signature:
                    raise ValueError("Resume config/code/data hash mismatch; choose a new output file")
                if row["status"] == "ok":
                    key = (row["sample_id"], row["mode"], row["steps"])
                    if key in done:
                        raise ValueError("Duplicate successful profile cell")
                    done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--prepare_manifest", type=Path)
    parser.add_argument("--calibration", type=int, default=100)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--historical_count", type=int, default=None,
                        help="Default 500 for HotpotQA, 0 otherwise; exclude already-seen examples")
    parser.add_argument("--sample_manifest", type=Path)
    parser.add_argument("--split", choices=["calibration", "validation", "holdout", "historical"], default="calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(0, 81, 5)))
    parser.add_argument("--modes", nargs="+", choices=["m1", "m2"], default=["m1"])
    parser.add_argument("--layers_list", nargs="+", type=int)
    parser.add_argument("--value_layers", nargs="+", type=int, default=[0, 17, 35])
    parser.add_argument("--value_window", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default="main", help="Prefer immutable Hugging Face commit hash")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_B", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", action="store_true", help="Default greedy; historical sampling is opt-in")
    parser.add_argument("--allow_b_think", action="store_true")
    parser.add_argument("--max_tokens_B", type=int, default=0)
    parser.add_argument("--max_input_length", type=int, default=64000)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    from dataloader import get_evaluator
    from utils.latent_samples import make_manifest, load_manifest
    if args.prepare_manifest:
        if args.prepare_manifest.exists():
            parser.error("Refusing to overwrite a sample manifest")
        task = args.task.removesuffix("_full")
        source_task = task + "_full" if task in ("hotpotqa", "qasper", "musique") else args.task
        evaluator = get_evaluator(source_task)
        historical = args.historical_count if args.historical_count is not None else (500 if task == "hotpotqa" else 0)
        document = make_manifest(evaluator.data, task, args.calibration, args.validation, args.holdout,
                                 args.seed, historical, getattr(evaluator.data, "_fingerprint", None))
        args.prepare_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.prepare_manifest.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.prepare_manifest}; hash={document['manifest_hash']}")
        return
    if not args.sample_manifest or not args.output:
        parser.error("Need --sample_manifest and --output, or --prepare_manifest")
    if min(args.steps) < 0 or len(set(args.steps)) != len(args.steps) or len(set(args.modes)) != len(args.modes):
        parser.error("Steps/modes must be unique; steps must be nonnegative")
    if "m2" in args.modes and not args.layers_list:
        parser.error("Mode 2 needs a frozen --layers_list (no re-ranking across N)")
    if "qwen3" not in args.model.lower():
        parser.error("Reference profiler v1 supports Qwen3")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    items, document = load_manifest(args.sample_manifest, args.task, args.split)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models import CVCommunicator
    from models_latent import LatentMAS
    from eval_latent import LatentCommunicationEvaluator
    features = PolicyConfig(0.0, value_window=args.value_window, value_layers=tuple(args.value_layers))
    evaluator = get_evaluator(args.task)
    evaluator.data = items
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    init_start = time.perf_counter()
    def load_model(device):
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
            device_map="auto" if device == "auto" else {"": device},
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        model.name = args.model
        return model
    a, b = load_model(args.device), load_model(args.device_B)
    total_layers = a.config.num_hidden_layers
    if any(i < 0 or i >= total_layers for i in (args.layers_list or []) + args.value_layers):
        parser.error("Layer index outside model range")
    source_hash = inference_code_hash(ROOT)
    profile = {"task": args.task, "manifest_hash": document["manifest_hash"], "split": args.split,
        "sample_ids": [sample_id(item) for item in items],
        "steps": sorted(args.steps), "modes": args.modes, "layers_list": args.layers_list,
        "features": asdict(features), "model": args.model, "revision": args.revision,
        "resolved_commit": getattr(a.config, "_commit_hash", None),
        "seed": args.seed, "do_sample": args.sampling, "allow_b_think": args.allow_b_think,
        "max_tokens_B": args.max_tokens_B or evaluator.max_tokens, "max_input_length": args.max_input_length,
        "torch": torch.__version__, "transformers": transformers.__version__, "source_hash": source_hash,
        "profiler_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": args.device, "device_B": args.device_B, "shift_back": True,
        "realign": True, "backend": "sdpa", "dtype": "bfloat16", "warmup": args.warmup,
        "primary_metric": evaluator.primary_metric, "prompt_version": evaluator.prompt_version,
        "measurement": "independent_fixed_reference_trace_enabled_synchronized"}
    signature = content_hash(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = read_completed(args.output, signature)
    latent = LatentMAS(a, latent_steps=max(args.steps), policy_config=features,
                      latent_trace=True, profile_timing=True, greedy=not args.sampling, sample_seed=args.seed)
    print(f"Model + realign init: {time.perf_counter() - init_start:.2f}s; config={signature}", flush=True)
    failed = 0
    for mode in args.modes:
        layers = list(range(total_layers)) if mode == "m1" else sorted(set(args.layers_list))
        cv = CVCommunicator(a, b, 0, -1, layers_list=layers, shift_back=True)
        runner = LatentCommunicationEvaluator(evaluator, tokenizer, False, args.max_input_length,
                    latent, cv, allow_b_think=args.allow_b_think, max_tokens_B=args.max_tokens_B)
        # Warmup is explicitly excluded from cell metrics and repeated on resume.
        for _ in range(args.warmup):
            latent.latent_steps = max(args.steps)
            runner.inference(a, cv, items[0])
        for item in items:
            for steps in sorted(args.steps):
                identity = sample_id(item)
                key = (identity, mode, steps)
                if key in done:
                    continue
                row = {"config_hash": signature, "profile": profile, "sample_id": identity,
                       "source_id": item.get("id", item.get("_id")), "mode": mode,
                       "steps": steps, "split": args.split}
                try:
                    latent.latent_steps = steps
                    response = runner.inference(a, cv, item)
                    iter(evaluator)  # reset counters before scoring exactly this cell
                    metrics = evaluator.evaluate_item(item, response)
                    row.update(status="ok", response=response, score=float(evaluator.get_result()),
                               metrics=metrics, adaptive=runner.last_inference_stats)
                except Exception as exc:
                    failed += 1
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                append_row(args.output, row)
                if row["status"] == "ok":
                    done.add(key)
                print(f"{mode} N={steps} id={identity[:10]} status={row['status']}", flush=True)
    expected = len(items) * len(args.steps) * len(args.modes)
    print(f"Coverage {len(done)}/{expected}; failed this invocation={failed}")
    if len(done) != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
