"""Paired ONLINE response-log comparison; rejects incomplete or mismatched runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_adaptive_latent import bootstrap_delta, summarize


def load_run(path, metric):
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("status") == "error":
                raise ValueError("Run contains an error; no complete-set speedup comparison")
            stats = record["adaptive"]
            identity = stats["sample_id"]
            if identity in rows:
                raise ValueError("Duplicate sample in run")
            if not stats["timing_synchronized"] or stats["feature_trace"]:
                raise ValueError("Use --profile_timing without --latent_trace for online comparison")
            rows[identity] = dict(sample_id=identity, score=float(record["item_metrics"][metric]),
                                  steps=stats["actual_steps"], adaptive=stats)
    if set(rows) != set(manifest["sample_ids"]) or not rows:
        raise ValueError("Missing samples relative to run manifest")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--metric", default="longbench_f1")
    parser.add_argument("--time_tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fm, fixed = load_run(args.fixed, args.metric)
    am, adaptive = load_run(args.adaptive, args.metric)
    if fm["latent_step_policy"] != "fixed" or am["latent_step_policy"] == "fixed":
        raise ValueError("Expected one fixed reference and one adaptive run")
    for key in ("task", "metric_version", "prompt_version", "method", "model_A", "model_B",
                "model_A_commit", "model_B_commit", "max_tokens_B", "selected_layers", "batch_size",
                "allow_b_think", "temperature", "top_p", "do_sample", "per_sample_seed", "seed",
                "latent_warmup", "sample_manifest_hash", "sample_split", "torch_version",
                "transformers_version", "inference_code_hash", "shift_back", "latent_space_realign",
                "max_input_length", "backend", "dtype", "device_map_A", "device_map_B"):
        if fm[key] != am[key]:
            raise ValueError(f"Runs differ in {key}; not a controlled fixed/adaptive comparison")
    if set(fixed) != set(adaptive):
        raise ValueError("Paired sample IDs differ")
    for identity in fixed:
        for key in ("prompt_A_token_hash", "prompt_B_token_hash", "decoding_seed"):
            if fixed[identity]["adaptive"][key] != adaptive[identity]["adaptive"][key]:
                raise ValueError(f"Sample {identity} differs in {key}")
    fixed_rows = [fixed[key] for key in sorted(fixed)]
    adaptive_rows = [adaptive[key] for key in sorted(adaptive)]
    fs, ads = summarize(fixed_rows), summarize(adaptive_rows)
    result = {"fixed": fs, "adaptive": ads, "paired": bootstrap_delta(adaptive_rows, fixed_rows),
              "latency_ratio_adaptive_over_fixed": ads["total_ms"] / fs["total_ms"],
              "within_time_budget": ads["total_ms"] <= fs["total_ms"] * (1 + args.time_tolerance),
              "warning": "One timing repetition only. Repeat on the same idle hardware; compare the fixed-N frontier, not just fixed-80."}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
