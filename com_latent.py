"""
com_latent.py — Entry point for KVComm + LatentMAS integration

Extends com.py with latent thinking and multiple KV communication modes.

  Mode 1 — LatentMAS standalone (without --latent_kv_select):
    Sender A runs latent thinking; full KV cache (all layers) is shared
    with receiver B. Equivalent to LatentMAS with KVComm agent setup.

  Mode 2 — KVComm + LatentMAS (--latent_kv_select):
    Sender A runs latent thinking; KV cache is layer-selected via
    CVCommunicator.prepare_key_cache() before being passed to B.

Comparison baselines (inherited from com.py):
  --do_test_skyline   Skyline (A+B see everything)
  --do_test_baseline  Baseline (B only, no A context)
  --do_test           Regular KVComm (no latent)

New flags:
  --do_test_latent    LatentMAS + KVComm (Mode 1 or 2 per --latent_kv_select)
  --do_test_nld       TextMAS — natural-language baseline for LatentMAS
                      (A generates a short text summary, B reads and refines)

Usage examples:
  # Mode 1: LatentMAS standalone
  python com_latent.py \\
      --model_A meta-llama/Llama-3.2-3B-Instruct \\
      --model_B meta-llama/Llama-3.2-3B-Instruct \\
      --latent_steps 5 --do_test_latent --test_task tmath --limit 50

  # Mode 2: KVComm layer selection + LatentMAS
  python com_latent.py \\
      --model_A meta-llama/Llama-3.2-3B-Instruct \\
      --model_B meta-llama/Llama-3.2-3B-Instruct \\
      --latent_steps 5 --latent_kv_select --layers_list 14 20 25 \\
      --do_test_latent --test_task tmath --limit 50

  # Compare with regular KVComm baseline
  python com_latent.py \\
      --model_A meta-llama/Llama-3.2-3B-Instruct \\
      --model_B meta-llama/Llama-3.2-3B-Instruct \\
      --layers_list 14 20 25 --do_test --test_task tmath --limit 50

  # TextMAS baseline (natural-language channel)
  python com_latent.py \\
      --model_A meta-llama/Llama-3.2-3B-Instruct \\
      --model_B meta-llama/Llama-3.2-3B-Instruct \\
      --do_test_nld --max_tokens_A 256 --max_tokens_B 64 --test_task hotpotqa --limit 10
"""

from utils.method_names import latent_method
from utils.model_loading import load_tokenizer, load_causal_model
import os
from collections import defaultdict
import torch
import argparse
import sys
import wandb
import datetime
import logging
import random
import json
import hashlib
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from models import CVCommunicator
from models_latent import LatentMAS
from eval import SkylineEvaluator, CommunicationEvaluator, BaselineEvaluator, is_think_model
from eval_latent import LatentCommunicationEvaluator, TextMASEvaluator
from utils import setup_logging, log_gpu_info, generate_run_name
from dataloader import get_evaluator
from layer_importance import get_top_layers, get_layer_ranking
from prompts_latent import build_sender_core, build_receiver_core
from adaptive_latent import validate_runtime, sample_id, inference_code_hash
from utils.latent_samples import load_manifest


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LatentAlignConfig:
    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"
    device_B: str = ""
    seed: int = 42
    snapshot_path: str = "snapshots"

    # ── Models ────────────────────────────────────────────────────────────
    model_A: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000

    # ── KVComm: layer selection ────────────────────────────────────────────
    layer_from: int = 0
    layer_to: int = -1  # -1 = all layers (auto-detect from model)
    layers_list: list[int] = field(default_factory=lambda: [-1])
    top_layers: float = 0.0
    calib_size: int = 5
    do_layer_curve: bool = False
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    random_selection: bool = False
    shift_back: bool = False

    # ── Latent params ──────────────────────────────────────────────────────
    latent_steps: int = 5
    latent_step_policy: str = "fixed"
    min_latent_steps: int = 10
    latent_check_interval: int = 5
    latent_patience: int = 2
    latent_policy_config: str = ""
    latent_trace: bool = False
    profile_timing: bool = False
    latent_warmup: int = 0
    greedy: bool = False
    per_sample_seed: bool = False
    sample_manifest: str = ""
    sample_split: str = "holdout"
    latent_space_realign: bool = True
    # latent_kv_select=False → Mode 1 (all layers, LatentMAS standalone)
    # latent_kv_select=True  → Mode 2 (layer selection via CVCommunicator)
    latent_kv_select: bool = False
    allow_b_think: bool = False
    # TextMAS sender budget. 0 means evaluator.sender_max_tokens.
    max_tokens_A: int = 0
    # max_tokens_B=0 → use evaluator's default (task-specific).
    # Set > 0 to grant B extra output headroom, e.g. 4096 when allow_b_think=True
    # so the model can reason inside <think>…</think> before giving the final answer.
    # Only applies to --do_test_latent (LatentMAS). Other modes are unaffected.
    max_tokens_B: int = 0
    batch_size: int = 1

    # ── Task ──────────────────────────────────────────────────────────────
    test_task: str = "tmath"
    task_name: str = ""
    limit: int = 0

    # ── What to run ────────────────────────────────────────────────────────────
    do_test_skyline: bool = False    # Skyline (A+B with full context)
    do_test_baseline: bool = False   # Baseline (B only, no A context)
    do_test: bool = False            # Regular KVComm (no latent, for comparison)
    do_test_latent: bool = False     # LatentMAS + KVComm  ← main new method
    do_test_nld: bool = False        # TextMAS ← sequential text-channel baseline for LatentMAS

    # ── W&B ───────────────────────────────────────────────────────────────
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"


# ──────────────────────────────────────────────────────────────────────────────
# Run name helper (extends generate_run_name to include latent info)
# ──────────────────────────────────────────────────────────────────────────────

def generate_latent_run_name(cfg: LatentAlignConfig) -> str:
    """Generate a descriptive run name including latent parameters."""
    if cfg.do_test_nld:
        # TextMAS: build run name with "TextMAS" label (not "NLD" from utils)
        from utils import get_model_short_name
        model_A_short = get_model_short_name(cfg.model_A)
        model_B_short = get_model_short_name(cfg.model_B)
        layer_to_str = "ALL" if cfg.layer_to < 0 else cfg.layer_to
        layer_info = f"from{cfg.layer_from}to{layer_to_str}"
        return f"{cfg.test_task}_{model_A_short}-to-{model_B_short}_{layer_info}_TextMAS_pv2_mv2_A{cfg.max_tokens_A}_B{cfg.max_tokens_B}"
    base = generate_run_name(cfg)   # reuse KVComm's convention
    latent_suffix = f"_lat{cfg.latent_steps}"
    if cfg.latent_space_realign:
        latent_suffix += "_realign"
    if cfg.latent_kv_select:
        latent_suffix += "_kvsel"
    latent_suffix += f"_pv2_mv2_B{cfg.max_tokens_B}"
    return base + latent_suffix



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_latent_evaluation(cfg, model_A, model_B, evaluator, tokenizer,
                          response_log_path, policy_config, policy_hash):
    logging.info(
        f"Running LatentMAS evaluation: "
        f"latent_steps={cfg.latent_steps}, "
        f"latent_space_realign={cfg.latent_space_realign}, "
        f"latent_kv_select={cfg.latent_kv_select}, "
        f"top_layers={cfg.top_layers}, "
        f"random_selection={cfg.random_selection}"
    )

    # Build LatentMAS wrapper for sender A
    latent_mas = LatentMAS(
        model=model_A,
        latent_steps=cfg.latent_steps,
        latent_space_realign=cfg.latent_space_realign,
        latent_step_policy=cfg.latent_step_policy,
        min_latent_steps=cfg.min_latent_steps,
        latent_check_interval=cfg.latent_check_interval,
        latent_patience=cfg.latent_patience,
        policy_config=policy_config,
        policy_config_hash=policy_hash,
        latent_trace=cfg.latent_trace,
        profile_timing=cfg.profile_timing,
        warmup=cfg.latent_warmup,
        greedy=cfg.greedy,
        sample_seed=cfg.seed if cfg.per_sample_seed or cfg.latent_step_policy != "fixed" else None,
    )

    # ── Resolve A_num_layers (shared across all modes) ────────────────
    if hasattr(model_A.config, "num_hidden_layers"):
        A_num_layers = model_A.config.num_hidden_layers
    else:
        A_num_layers = model_A.config.text_config.num_hidden_layers

    if not cfg.latent_kv_select:
        latent_layers_list = list(range(A_num_layers))
        if cfg.top_layers > 0:
            logging.warning(
                "top_layers is set but latent_kv_select=False (Mode 1). "
                "Auto layer selection only applies to Mode 2 (--latent_kv_select). "
                "Ignoring top_layers and using all layers."
            )
        logging.info(
            f"Mode 1 (LatentMAS standalone): "
            f"all {A_num_layers} layers selected (no KV layer selection)"
        )

        # Build CVCommunicator and evaluator for Mode 1
        cv = CVCommunicator(
            model_A, model_B,
            cfg.layer_from, cfg.layer_to,
            layers_list=latent_layers_list,
            top_layers=0.0,
            apply_attn_tracer=False,
            shift_back=cfg.shift_back,
        )
        latent_evaluator = LatentCommunicationEvaluator(
            evaluator=evaluator,
            tokenizer=tokenizer,
            use_wandb=cfg.use_wandb,
            max_input_length=cfg.max_input_length,
            latent_mas=latent_mas,
            cv=cv,
            allow_b_think=cfg.allow_b_think,
            max_tokens_B=cfg.max_tokens_B,
            response_log_path=response_log_path,
        )
        results = latent_evaluator.test(model_A, cv, limit=cfg.limit, batch_size=cfg.batch_size)

    else:
        # Mode 2: KVComm layer selection + LatentMAS
        # ── Sub-mode: RANDOM selection ─────────────────────────────
        if cfg.random_selection:
            if hasattr(model_A.config, "num_hidden_layers"):
                A_num_layers = model_A.config.num_hidden_layers
            else:
                A_num_layers = model_A.config.text_config.num_hidden_layers
            n_select = max(1, int(cfg.top_layers * A_num_layers)) if cfg.top_layers > 0 else len(cfg.layers_list)
            cfg.layers_list = random.sample(list(range(A_num_layers)), n_select)
            logging.info(f"Mode 2 RANDOM: randomly selected layers_list={cfg.layers_list}")

            cv = CVCommunicator(
                model_A, model_B,
                cfg.layer_from, cfg.layer_to,
                layers_list=cfg.layers_list,
                top_layers=0.0,
                apply_attn_tracer=False,
                shift_back=cfg.shift_back,
            )
            latent_evaluator = LatentCommunicationEvaluator(
                evaluator=evaluator,
                tokenizer=tokenizer,
                use_wandb=cfg.use_wandb,
                max_input_length=cfg.max_input_length,
                latent_mas=latent_mas,
                cv=cv,
                allow_b_think=cfg.allow_b_think,
                max_tokens_B=cfg.max_tokens_B,
                response_log_path=response_log_path,
            )
            results = latent_evaluator.test(model_A, cv, limit=cfg.limit)

        elif cfg.top_layers > 0:
            # ── Sub-mode: AUTO selection (calibrate via LatentMAS) ──
            # Get all layers for A to use during calibration
            if hasattr(model_A.config, "num_hidden_layers"):
                A_num_layers = model_A.config.num_hidden_layers
            else:
                A_num_layers = model_A.config.text_config.num_hidden_layers
            calib_layers_list = list(range(A_num_layers))

            # Step 1: Build calibration cv with attn_tracer=True (all layers)
            logging.info(
                f"Mode 2 AUTO: calibrating layer importance over "
                f"{cfg.calib_size} sample(s) using LatentMAS forward..."
            )
            cv_calib = CVCommunicator(
                model_A, model_B,
                cfg.layer_from, cfg.layer_to,
                layers_list=calib_layers_list,
                top_layers=0.0,
                apply_attn_tracer=True,
                shift_back=cfg.shift_back,
            )

            # Step 2: Build latent evaluator using cv_calib for assertion check
            latent_evaluator = LatentCommunicationEvaluator(
                evaluator=evaluator,
                tokenizer=tokenizer,
                use_wandb=cfg.use_wandb,
                max_input_length=cfg.max_input_length,
                latent_mas=latent_mas,
                cv=cv_calib,
                allow_b_think=cfg.allow_b_think,
                max_tokens_B=cfg.max_tokens_B,
                response_log_path=response_log_path,
            )

            if not cfg.do_layer_curve:
                # Step 3a: Run calibration to collect layer importance
                latent_evaluator.test(
                    model_A, cv_calib,
                    limit=cfg.calib_size,
                    no_wandb=True,
                    do_calc_layer_importance=True,
                )

                # Step 4a: Compute top layers from attention importance
                cfg = get_top_layers(latent_evaluator.layer_importance_total, cfg)
                logging.info(f"Mode 2 AUTO: selected layers_list={cfg.layers_list}")

                # Step 5a: Rebuild cv with selected layers (attn_tracer already in model_B — harmless)
                cv = CVCommunicator(
                    model_A, model_B,
                    cfg.layer_from, cfg.layer_to,
                    layers_list=cfg.layers_list,
                    top_layers=0.0,
                    apply_attn_tracer=False,
                    shift_back=cfg.shift_back,
                )

                # Step 6a: Reset layer importance and run actual evaluation
                latent_evaluator.layer_importance_total = defaultdict(list)
                results = latent_evaluator.test(model_A, cv, limit=cfg.limit, batch_size=cfg.batch_size)

            else:
                # Step 3b (do_layer_curve): Calibrate → full layer ranking
                latent_evaluator.test(
                    model_A, cv_calib,
                    limit=cfg.calib_size,
                    no_wandb=True,
                    do_calc_layer_importance=True,
                )
                layer_ranking = get_layer_ranking(
                    latent_evaluator.layer_importance_total, cfg
                )
                logging.info(f"Mode 2 AUTO layer_curve: ranking={list(layer_ranking)}")

                # Step 4b: Sweep layers_list size from 1 to all
                results = []
                for i in range(len(layer_ranking)):
                    layers_list_i = list(layer_ranking[: i + 1])
                    logging.info(f"Layer curve step {i+1}/{len(layer_ranking)}: layers_list={layers_list_i}")
                    cv_i = CVCommunicator(
                        model_A, model_B,
                        cfg.layer_from, cfg.layer_to,
                        layers_list=layers_list_i,
                        top_layers=0.0,
                        apply_attn_tracer=False,
                        shift_back=cfg.shift_back,
                    )
                    latent_evaluator.layer_importance_total = defaultdict(list)
                    result = latent_evaluator.test(model_A, cv_i, limit=cfg.limit, batch_size=cfg.batch_size)
                    results.append(result)
                logging.info(f"Mode 2 AUTO layer_curve results: {results}")
                if cfg.use_wandb:
                    wandb.log({"latent_layer_curve_results": results})

        else:
            # ── Sub-mode: MANUAL selection (original behaviour) ─────
            latent_layers_list = cfg.layers_list
            logging.info(
                f"Mode 2 MANUAL (KVComm + LatentMAS): "
                f"layers_list={latent_layers_list}"
            )

            cv = CVCommunicator(
                model_A, model_B,
                cfg.layer_from, cfg.layer_to,
                layers_list=latent_layers_list,
                top_layers=0.0,
                apply_attn_tracer=False,
                shift_back=cfg.shift_back,
            )
            latent_evaluator = LatentCommunicationEvaluator(
                evaluator=evaluator,
                tokenizer=tokenizer,
                use_wandb=cfg.use_wandb,
                max_input_length=cfg.max_input_length,
                latent_mas=latent_mas,
                cv=cv,
                allow_b_think=cfg.allow_b_think,
                max_tokens_B=cfg.max_tokens_B,
                response_log_path=response_log_path,
            )

            if not cfg.do_layer_curve:
                results = latent_evaluator.test(model_A, cv, limit=cfg.limit, batch_size=cfg.batch_size)
            else:
                # Manual do_layer_curve: iterate prefix of layers_list by position
                results = []
                for i in range(len(latent_layers_list)):
                    layers_list_i = latent_layers_list[: i + 1]
                    logging.info(f"Layer curve step {i+1}/{len(latent_layers_list)}: layers_list={layers_list_i}")
                    cv_i = CVCommunicator(
                        model_A, model_B,
                        cfg.layer_from, cfg.layer_to,
                        layers_list=layers_list_i,
                        top_layers=0.0,
                        apply_attn_tracer=False,
                        shift_back=cfg.shift_back,
                    )
                    result = latent_evaluator.test(model_A, cv_i, limit=cfg.limit)
                    results.append(result)
                logging.info(f"Mode 2 MANUAL layer_curve results: {results}")
                if cfg.use_wandb:
                    wandb.log({"latent_layer_curve_results": results})
    return results


def main(cfg: LatentAlignConfig):
    policy_config, policy_document, policy_hash = validate_runtime(cfg)
    sample_items, sample_document = (load_manifest(cfg.sample_manifest, cfg.test_task, cfg.sample_split)
                                      if cfg.sample_manifest else (None, None))
    if sample_items is not None and cfg.sample_split == "holdout" and policy_document:
        provenance = policy_document.get("provenance", {})
        used = set(provenance.get("calibration_sample_ids", [])) | set(provenance.get("validation_sample_ids", []))
        if used & {sample_id(item) for item in sample_items}:
            raise ValueError("Holdout overlaps samples used to select this policy")
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if cfg.run_name == "":
        run_name = generate_latent_run_name(cfg)
    else:
        run_name = cfg.run_name
        if getattr(cfg, "test_task", "") and not run_name.startswith(f"{cfg.test_task}_"):
            run_name = f"{cfg.test_task}_{run_name}"
    if cfg.latent_step_policy != "fixed":
        run_name += f"_{cfg.latent_step_policy}_cap{cfg.latent_steps}_{policy_hash[:8]}"
    run_name  = f"{run_name}_{timestamp}"

    final_snapshot_path = os.path.join(cfg.snapshot_path, run_name)
    os.makedirs(final_snapshot_path, exist_ok=True)
    log_file_path = os.path.join(final_snapshot_path, "log.log")

    setup_logging(log_file_path=log_file_path, log_level=cfg.log_level)
    logging.info(f"Configuration: {cfg}")
    logging.info(f"Outputs will be saved to: {final_snapshot_path}")
    log_gpu_info()

    # Response log paths — one file per mode so they don't overwrite each other
    response_log_dir  = final_snapshot_path
    response_log_path = os.path.join(response_log_dir, "latent_responses.jsonl")
    skyline_log_path  = os.path.join(response_log_dir, "skyline_responses.jsonl")
    baseline_log_path = os.path.join(response_log_dir, "baseline_responses.jsonl")
    kvcomm_log_path   = os.path.join(response_log_dir, "kvcomm_responses.jsonl")
    textmas_log_path  = os.path.join(response_log_dir, "textmas_responses.jsonl")
    logging.info(f"Response logs will be saved to: {response_log_dir}/{{skyline|baseline|kvcomm|latent|textmas}}_responses.jsonl")

    # ── W&B ───────────────────────────────────────────────────────────────
    if cfg.use_wandb:
        wandb_config = {k: v for k, v in cfg.__dict__.items() if not k.startswith("wandb_")}
        wandb_tags   = [t.strip() for t in cfg.wandb_tags.split(",")] if cfg.wandb_tags else []
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            entity=cfg.wandb_entity,
            tags=wandb_tags,
            config=wandb_config,
        )

    # ── Load tokenizer & models ────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.model_B)

    device_A = cfg.device
    device_B = cfg.device_B if cfg.device_B else cfg.device

    model_A = load_causal_model(cfg.model_A, device_A)
    model_B = load_causal_model(cfg.model_B, device_B)
    model_A.eval()
    model_B.eval()

    # Convention from com.py: attach model path as .name attribute
    # (used by apply_chat_template / is_think_model in eval.py)
    model_A.name = cfg.model_A
    model_B.name = cfg.model_B

    # Gemma-specific dynamo workaround (same as com.py)
    if "gemma" in cfg.model_A.lower() or "gemma" in cfg.model_B.lower():
        torch._dynamo.config.cache_size_limit = 64

    evaluator = get_evaluator(cfg.test_task)
    if sample_items is not None:
        evaluator.data = sample_items
    first_item = evaluator.data[0] if len(evaluator) else None
    sender_core = build_sender_core(evaluator, first_item, is_think=is_think_model(model_A)) if first_item else ""
    receiver_core = build_receiver_core(evaluator, first_item, allow_b_think=cfg.allow_b_think) if first_item else ""
    effective_a_budget = cfg.max_tokens_A if cfg.max_tokens_A > 0 else evaluator.sender_max_tokens
    effective_b_budget = cfg.max_tokens_B if cfg.max_tokens_B > 0 else evaluator.max_tokens
    method_name = (
        "textmas" if cfg.do_test_nld
        else latent_method(cfg.latent_kv_select)
    )
    manifest = {
        "schema_version": "v2",
        "metric_version": "v2",
        "task": cfg.test_task,
        "prompt_family": evaluator.prompt_family,
        "prompt_version": evaluator.prompt_version,
        "sender_input_mode": evaluator.sender_input_mode,
        "primary_metric": evaluator.primary_metric,
        "method": method_name,
        "seed": cfg.seed,
        "limit": cfg.limit,
        "model_A": cfg.model_A,
        "model_B": cfg.model_B,
        "max_tokens_A": effective_a_budget,
        "max_tokens_B": effective_b_budget,
        "latent_steps": cfg.latent_steps,
        "selected_layers": cfg.layers_list,
        "batch_size": cfg.batch_size,
        "allow_b_think": cfg.allow_b_think,
        "temperature": None if cfg.greedy else 0.6,
        "top_p": None if cfg.greedy else 0.95,
        "do_sample": not cfg.greedy,
        "latent_step_policy": cfg.latent_step_policy,
        "policy_config_hash": policy_hash,
        "policy_document": policy_document,
        "min_latent_steps": cfg.min_latent_steps,
        "latent_check_interval": cfg.latent_check_interval,
        "latent_patience": cfg.latent_patience,
        "per_sample_seed": cfg.per_sample_seed or cfg.latent_step_policy != "fixed",
        "latent_trace": cfg.latent_trace,
        "profile_timing": cfg.profile_timing,
        "latent_warmup": cfg.latent_warmup,
        "sample_ids": [sample_id(row) for row in (sample_items if sample_items is not None else evaluator.data)][:cfg.limit or None],
        "model_A_commit": getattr(model_A.config, "_commit_hash", None),
        "model_B_commit": getattr(model_B.config, "_commit_hash", None),
        "sample_manifest_hash": sample_document["manifest_hash"] if sample_document else None,
        "sample_split": cfg.sample_split if sample_document else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "inference_code_hash": inference_code_hash(Path(__file__).parent),
        "shift_back": cfg.shift_back,
        "latent_space_realign": cfg.latent_space_realign,
        "max_input_length": cfg.max_input_length,
        "backend": "sdpa",
        "dtype": "bfloat16",
        "device_map_A": str(getattr(model_A, "hf_device_map", cfg.device)),
        "device_map_B": str(getattr(model_B, "hf_device_map", cfg.device_B)),
        "first_sender_core_sha256": hashlib.sha256(sender_core.encode("utf-8")).hexdigest(),
        "first_receiver_core_sha256": hashlib.sha256(receiver_core.encode("utf-8")).hexdigest(),
    }
    manifest_path = os.path.join(final_snapshot_path, "manifest.json")

    def persist_manifest():
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    persist_manifest()
    logging.info(f"Evaluation profile: {manifest}")
    if cfg.limit == 0:
        cfg.limit = None

    results = None

    # ── Skyline ───────────────────────────────────────────────────────────
    if cfg.do_test_skyline:
        logging.info("Running skyline evaluation...")
        skyline_evaluator = SkylineEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length,
            response_log_path=skyline_log_path,
        )
        results = skyline_evaluator.test(model_A, model_B, limit=cfg.limit)

    # ── Baseline ──────────────────────────────────────────────────────────
    if cfg.do_test_baseline:
        logging.info("Running baseline evaluation...")
        baseline_evaluator = BaselineEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length,
            response_log_path=baseline_log_path,
        )
        results = baseline_evaluator.test(model_A, model_B, limit=cfg.limit)

    # ── Regular KVComm (comparison baseline, no latent) ───────────────────
    if cfg.do_test:
        logging.info("Running regular KVComm evaluation (no latent)...")
        comm_evaluator = CommunicationEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length,
            response_log_path=kvcomm_log_path,
        )
        cv = CVCommunicator(
            model_A, model_B,
            cfg.layer_from, cfg.layer_to,
            layers_list=cfg.layers_list,
            top_layers=cfg.top_layers,
            apply_attn_tracer=False,
            shift_back=cfg.shift_back,
        )
        results = comm_evaluator.test(model_A, cv, limit=cfg.limit)

    # ── TextMAS (sequential text-channel baseline, paper-faithful) ────────────
    if cfg.do_test_nld:
        logging.info(
            f"Running TextMAS (sequential text-channel baseline): "
            f"prompt_family={evaluator.prompt_family}, prompt_version={evaluator.prompt_version}, "
            f"allow_b_think={cfg.allow_b_think}, max_tokens_A={cfg.max_tokens_A}, max_tokens_B={cfg.max_tokens_B}"
        )
        textmas_evaluator = TextMASEvaluator(
            evaluator=evaluator,
            tokenizer=tokenizer,
            use_wandb=cfg.use_wandb,
            max_input_length=cfg.max_input_length,
            allow_b_think=cfg.allow_b_think,
            max_tokens_A=cfg.max_tokens_A,
            max_tokens_B=cfg.max_tokens_B,
            response_log_path=textmas_log_path,
        )
        results = textmas_evaluator.test(model_A, model_B, limit=cfg.limit)

    # ── LatentMAS + KVComm (main new method) ──────────────────────────────
    if cfg.do_test_latent:
        results = run_latent_evaluation(
            cfg, model_A, model_B, evaluator, tokenizer,
            response_log_path, policy_config, policy_hash)

    # ── Finish ────────────────────────────────────────────────────────────
    if cfg.use_wandb:
        wandb.finish()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> LatentAlignConfig:
    parser = argparse.ArgumentParser(
        description="KVComm + LatentMAS: latent thinking for KV cache communication"
    )
    removed = {"--dual_kv_select", "--segmented_kv_select", "--split_ratio",
               "--context_top_ratio", "--latent_top_ratio", "--track_convergence"}
    for argument in sys.argv[1:]:
        flag = argument.split("=", 1)[0]
        if flag in removed:
            hint = " Use --latent_trace instead." if flag == "--track_convergence" else " Modes 4/5 are retired."
            parser.error(f"{flag} has been removed.{hint}")
    defaults = LatentAlignConfig()
    for fname, default in defaults.__dict__.items():
        arg_type = type(default)
        if isinstance(default, bool):
            if default:
                parser.add_argument(f"--no_{fname}", dest=fname, action="store_false")
            else:
                parser.add_argument(f"--{fname}", dest=fname, action="store_true")
            parser.set_defaults(**{fname: default})
        elif isinstance(default, list):
            element_type = type(default[0])
            parser.add_argument(f"--{fname}", type=element_type, default=default, nargs="+")
        else:
            parser.add_argument(f"--{fname}", type=arg_type, default=default)
    args = parser.parse_args()
    return LatentAlignConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    main(config)
