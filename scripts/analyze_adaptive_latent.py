"""Calibrate training-free stopping on independent reference profiles.

Replay timing is an ESTIMATE from trace-enabled fixed runs, never online speedup.
Select a shortlist on calibration, choose a budget-feasible candidate on validation,
then lock JSON before a separate online holdout run. No gold enters the controller.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_latent import PolicyConfig, StopController, content_hash, replay


def load_profiles(path):
    cells, profile = {}, None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if content_hash(row["profile"]) != row["config_hash"]:
                raise ValueError("Profile config hash mismatch")
            if profile is None:
                profile = row["profile"]
            if profile != row["profile"]:
                raise ValueError("Mixed profile configurations")
            key = (row["sample_id"], row["mode"], row["steps"])
            if key in cells and cells[key]["status"] == "ok":
                raise ValueError("Duplicate/overwritten successful cell")
            cells[key] = row
    if not profile:
        raise ValueError("Empty profile file")
    expected = {(identity, mode, n) for identity in profile["sample_ids"]
                for mode in profile["modes"] for n in profile["steps"]}
    if set(cells) != expected or any(row["status"] != "ok" for row in cells.values()):
        raise ValueError("Incomplete profile coverage or failed cells; resume before analysis")
    for row in cells.values():
        for value in (row["score"], row["adaptive"]["end_to_end_ms"]):
            if not math.isfinite(value):
                raise ValueError("Non-finite score/timing")
    return profile, cells


def matrix_for(profile, cells, mode):
    if mode not in profile["modes"]:
        raise ValueError(f"Mode {mode} missing")
    return {identity: {n: cells[identity, mode, n] for n in profile["steps"]}
            for identity in sorted(profile["sample_ids"])}


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def summarize(rows):
    latency = [row["adaptive"]["end_to_end_ms"] for row in rows]
    return {"count": len(rows), "score": statistics.mean(row["score"] for row in rows),
            "total_ms": sum(latency), "mean_ms": statistics.mean(latency),
            "p50_ms": percentile(latency, .5), "p95_ms": percentile(latency, .95),
            "mean_steps": statistics.mean(row["steps"] for row in rows),
            "mean_B_tokens_raw": statistics.mean(row["adaptive"]["generated_B_token_count_raw"] for row in rows)}


def evaluate_policy(matrix, spec, features):
    chosen = []
    for checkpoints in matrix.values():
        trace = checkpoints[spec["max_steps"]]["adaptive"]["feature_trace"]
        n, _ = replay(trace, StopController(config=features, **spec))
        if n not in checkpoints:
            raise ValueError(f"Policy stops at N={n}, but no B answer was measured there; extend profiling grid")
        chosen.append(checkpoints[n])
    return chosen


def bootstrap_delta(rows, reference, seed=42, repeats=2000):
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in reference]:
        raise ValueError("Paired bootstrap sample IDs differ")
    delta = [a["score"] - b["score"] for a, b in zip(rows, reference)]
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(repeats)]
    return {"delta_score": statistics.mean(delta), "ci95": [percentile(means, .025), percentile(means, .975)]}


def diagnostics(matrix, chosen, seed=42):
    values = list(matrix.values())
    ns = sorted(values[0])
    oracle = [max(cells.values(), key=lambda row: (row["score"], -row["steps"])) for cells in values]
    allocations = [row["steps"] for row in chosen]
    rng = random.Random(seed)
    random_results = []
    for _ in range(100):
        rng.shuffle(allocations)
        random_results.append(summarize([cells[n] for cells, n in zip(values, allocations)]))
    gain = [cells[ns[-1]]["score"] - cells[ns[0]]["score"] for cells in values]
    return {"oracle_unconstrained_gold_upper_bound_not_deployable": summarize(oracle),
            "all_checkpoints_zero_score": sum(all(row["score"] == 0 for row in cells.values()) for cells in values),
            "min_to_max_gain": {"helped": sum(x > 0 for x in gain), "harmed": sum(x < 0 for x in gain),
                                "unchanged": sum(x == 0 for x in gain)},
            "step_distribution_matched_random_100_permutations": {
                "score_mean": statistics.mean(row["score"] for row in random_results),
                "total_ms_mean": statistics.mean(row["total_ms"] for row in random_results),
                "note": "Same N histogram, NOT guaranteed equal time; verify matched-time allocation online."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output_policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=["m1", "m2"], default="m1")
    parser.add_argument("--policy", choices=["cosine", "hidden_value"], default="cosine")
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--min_steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--cosine_thresholds", type=float, nargs="+", default=[.001, .005, .01, .02, .05, .1])
    parser.add_argument("--norm_thresholds", type=float, nargs="+", default=None,
                        help="Omit for cosine-only ablation")
    parser.add_argument("--value_thresholds", type=float, nargs="+", default=[.005, .02, .05, .1])
    parser.add_argument("--budget_fixed_steps", type=int, default=40)
    parser.add_argument("--time_tolerance", type=float, default=0.0)
    parser.add_argument("--shortlist", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_policy.exists() or args.report.exists():
        parser.error("Refusing to overwrite policy/report; use a new path")
    if args.time_tolerance < 0 or args.shortlist < 1:
        parser.error("Require time_tolerance >= 0 and shortlist >= 1")
    cal_profile, cal_cells = load_profiles(args.calibration)
    val_profile, val_cells = load_profiles(args.validation)
    if cal_profile["split"] != "calibration" or val_profile["split"] != "validation":
        raise ValueError("Threshold selection may use only calibration + validation, never holdout/historical")
    if set(cal_profile["sample_ids"]) & set(val_profile["sample_ids"]):
        raise ValueError("Calibration/validation overlap")
    comparable = lambda doc: {k: v for k, v in doc.items() if k not in ("split", "sample_ids")}
    if comparable(cal_profile) != comparable(val_profile):
        raise ValueError("Calibration and validation differ in model/manifest/features/decoding/code")
    if args.max_steps not in cal_profile["steps"] or args.budget_fixed_steps not in cal_profile["steps"]:
        raise ValueError("Missing cap or fixed budget baseline checkpoint")
    cal, val = matrix_for(cal_profile, cal_cells, args.mode), matrix_for(val_profile, val_cells, args.mode)
    reference = lambda matrix: [cells[args.budget_fixed_steps] for cells in matrix.values()]
    cal_budget = summarize(reference(cal))["total_ms"] * (1 + args.time_tolerance)
    val_budget = summarize(reference(val))["total_ms"] * (1 + args.time_tolerance)
    spec = dict(policy=args.policy, max_steps=args.max_steps, min_steps=args.min_steps,
                interval=args.interval, patience=args.patience)
    candidates = []
    for cosine in args.cosine_thresholds:
        for norm in args.norm_thresholds or [None]:
            for value in args.value_thresholds if args.policy == "hidden_value" else [None]:
                feature_dict = dict(cal_profile["features"], cosine_distance=cosine,
                                    log_norm_delta=norm, value_novelty=value)
                features = PolicyConfig(**feature_dict)
                rows = evaluate_policy(cal, spec, features)
                result = summarize(rows)
                if result["total_ms"] <= cal_budget:
                    candidates.append((features, result))
    candidates.sort(key=lambda pair: (-pair[1]["score"], pair[1]["total_ms"]))
    shortlist = []
    for features, cal_result in candidates[:args.shortlist]:
        rows = evaluate_policy(val, spec, features)
        val_result = summarize(rows)
        if val_result["total_ms"] <= val_budget:
            shortlist.append((features, cal_result, val_result, rows))
    if not shortlist:
        raise ValueError("No budget-feasible candidate on both splits. No policy exported; expand grid or retain fixed-N.")
    shortlist.sort(key=lambda entry: (-entry[2]["score"], entry[2]["total_ms"]))
    features, cal_result, val_result, selected = shortlist[0]
    fixed = {str(n): summarize([cells[n] for cells in val.values()]) for n in val_profile["steps"]}
    feasible_fixed = [row for row in fixed.values() if row["total_ms"] <= val_budget]
    best_fixed = max(row["score"] for row in feasible_fixed)
    report = {"warning": "OFFLINE REPLAY ESTIMATE. Trace overhead included; no demonstrated online gain. Validation is used for selection, so its CI is exploratory.",
              "mode": args.mode, "controller": spec, "features": asdict(features),
              "budget_fixed_steps": args.budget_fixed_steps, "time_tolerance": args.time_tolerance,
              "calibration": cal_result, "validation": val_result, "validation_fixed_frontier": fixed,
              "exceeds_best_budget_feasible_fixed_score": val_result["score"] > best_fixed,
              "paired_vs_budget_fixed": bootstrap_delta(selected, reference(val), args.seed),
              "diagnostics": diagnostics(val, selected, args.seed),
              "candidate_count_calibration_feasible": len(candidates),
              "next": "Freeze JSON; run adaptive and fixed online on disjoint holdout with identical decoding and repeated timing."}
    policy = {"schema_version": 1, "status": "offline_calibrated_candidate_not_online_validated",
              "controller": spec, "features": asdict(features),
              "provenance": {"calibration_hash": content_hash(cal_profile), "validation_hash": content_hash(val_profile),
                  "calibration_sample_ids": cal_profile["sample_ids"], "validation_sample_ids": val_profile["sample_ids"],
                  "manifest_hash": cal_profile["manifest_hash"], "task": cal_profile["task"],
                  "mode": args.mode, "layers_list": cal_profile["layers_list"],
                  "model": cal_profile["model"], "source_hash": cal_profile["source_hash"],
                  "do_sample": cal_profile["do_sample"], "seed": cal_profile["seed"],
                  "budget_fixed_steps": args.budget_fixed_steps, "time_tolerance": args.time_tolerance}}
    for path, doc in ((args.report, report), (args.output_policy, policy)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"validation": val_result, "beats_fixed_frontier": report["exceeds_best_budget_feasible_fixed_score"],
                      "policy": str(args.output_policy)}, indent=2))


if __name__ == "__main__":
    main()
