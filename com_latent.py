"""
com_latent.py — Entry point for KVComm + LatentMAS integration

Extends com.py with latent thinking params and two operating modes:

  Mode 1 — LatentMAS standalone (--no_latent_kv_select):
    Sender A runs latent thinking; full KV cache (all layers) is shared
    with receiver B. Equivalent to LatentMAS with KVComm agent setup.

  Mode 2 — KVComm + LatentMAS (--latent_kv_select):
    Sender A runs latent thinking; KV cache is layer-selected via
    CVCommunicator.prepare_key_cache() before being passed to B.

Comparison baselines (inherited from com.py):
  --do_test_skyline   Skyline (A+B see everything)
  --do_test_baseline  Baseline (B only, no A context)
  --do_test           Regular KVComm (no latent)

New flag:
  --do_test_latent    LatentMAS + KVComm (Mode 1 or 2 per --latent_kv_select)

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
"""

import os
import torch
import argparse
import wandb
import datetime
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from models import CVCommunicator
from models_latent import LatentMAS
from eval import SkylineEvaluator, CommunicationEvaluator, BaselineEvaluator
from eval_latent import LatentCommunicationEvaluator
from utils import setup_logging, log_gpu_info, generate_run_name
from dataloader import get_evaluator
from layer_importance import get_top_layers, get_layer_ranking


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LatentAlignConfig:
    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"
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
    calib_size: int = 1
    do_layer_curve: bool = False
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    random_selection: bool = False
    shift_back: bool = False

    # ── Latent params (NEW) ───────────────────────────────────────────────
    latent_steps: int = 5
    latent_space_realign: bool = True
    # latent_kv_select=False → Mode 1 (all layers, LatentMAS standalone)
    # latent_kv_select=True  → Mode 2 (layer selection via CVCommunicator)
    latent_kv_select: bool = False
    # latent_only=False → B receives full T_input + N_latent KV tokens (backward-compat)
    # latent_only=True  → B receives only the N_latent compressed thought tokens
    latent_only: bool = False

    # ── Task ──────────────────────────────────────────────────────────────
    test_task: str = "tmath"
    task_name: str = ""
    limit: int = 0

    # ── What to run ───────────────────────────────────────────────────────
    do_test_skyline: bool = False    # Skyline (A+B with full context)
    do_test_baseline: bool = False   # Baseline (B only, no A context)
    do_test: bool = False            # Regular KVComm (no latent, for comparison)
    do_test_latent: bool = False     # LatentMAS + KVComm  ← main new method

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
    base = generate_run_name(cfg)   # reuse KVComm's convention
    latent_suffix = f"_lat{cfg.latent_steps}"
    if cfg.latent_space_realign:
        latent_suffix += "_realign"
    if cfg.latent_kv_select:
        latent_suffix += "_kvsel"
    return base + latent_suffix


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: LatentAlignConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name  = generate_latent_run_name(cfg) if cfg.run_name == "" else cfg.run_name
    run_name  = f"{run_name}_{timestamp}"

    final_snapshot_path = os.path.join(cfg.snapshot_path, run_name)
    os.makedirs(final_snapshot_path, exist_ok=True)
    log_file_path = os.path.join(final_snapshot_path, "log.log")

    setup_logging(log_file_path=log_file_path, log_level=cfg.log_level)
    logging.info(f"Configuration: {cfg}")
    logging.info(f"Outputs will be saved to: {final_snapshot_path}")
    log_gpu_info()

    # Response log file (written alongside log.log in the same snapshot folder)
    response_log_path = os.path.join(final_snapshot_path, "responses.jsonl")
    logging.info(f"Response log will be saved to: {response_log_path}")

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_A = AutoModelForCausalLM.from_pretrained(
        cfg.model_A,
        device_map={"": cfg.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model_B = AutoModelForCausalLM.from_pretrained(
        cfg.model_B,
        device_map={"": cfg.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
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
    if cfg.limit == 0:
        cfg.limit = None

    results = None

    # ── Skyline ───────────────────────────────────────────────────────────
    if cfg.do_test_skyline:
        logging.info("Running skyline evaluation...")
        skyline_evaluator = SkylineEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length
        )
        results = skyline_evaluator.test(model_A, model_B, limit=cfg.limit)

    # ── Baseline ──────────────────────────────────────────────────────────
    if cfg.do_test_baseline:
        logging.info("Running baseline evaluation...")
        baseline_evaluator = BaselineEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length
        )
        results = baseline_evaluator.test(model_A, model_B, limit=cfg.limit)

    # ── Regular KVComm (comparison baseline, no latent) ───────────────────
    if cfg.do_test:
        logging.info("Running regular KVComm evaluation (no latent)...")
        comm_evaluator = CommunicationEvaluator(
            evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length,
            response_log_path=response_log_path,
        )
        cv = CVCommunicator(
            model_A, model_B,
            cfg.layer_from, cfg.layer_to,
            layers_list=cfg.layers_list,
            top_layers=cfg.top_layers,
            apply_attn_tracer=False,
            shift_back=cfg.shift_back,
        ).to(cfg.device)
        results = comm_evaluator.test(model_A, cv, limit=cfg.limit)

    # ── LatentMAS + KVComm (main new method) ──────────────────────────────
    if cfg.do_test_latent:
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
        )

        # ── Determine layers_list for CVCommunicator ───────────────────
        if not cfg.latent_kv_select:
            # Mode 1: all layers → no selection, equivalent to LatentMAS standalone
            if hasattr(model_A.config, "num_hidden_layers"):
                A_num_layers = model_A.config.num_hidden_layers
            else:
                A_num_layers = model_A.config.text_config.num_hidden_layers
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
            ).to(cfg.device)
            latent_evaluator = LatentCommunicationEvaluator(
                evaluator=evaluator,
                tokenizer=tokenizer,
                use_wandb=cfg.use_wandb,
                max_input_length=cfg.max_input_length,
                latent_mas=latent_mas,
                cv=cv,
                response_log_path=response_log_path,
            )
            results = latent_evaluator.test(model_A, cv, limit=cfg.limit)

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
                ).to(cfg.device)
                latent_evaluator = LatentCommunicationEvaluator(
                    evaluator=evaluator,
                    tokenizer=tokenizer,
                    use_wandb=cfg.use_wandb,
                    max_input_length=cfg.max_input_length,
                    latent_mas=latent_mas,
                    cv=cv,
                    latent_only=cfg.latent_only,
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
                ).to(cfg.device)

                # Step 2: Build latent evaluator using cv_calib for assertion check
                latent_evaluator = LatentCommunicationEvaluator(
                    evaluator=evaluator,
                    tokenizer=tokenizer,
                    use_wandb=cfg.use_wandb,
                    max_input_length=cfg.max_input_length,
                    latent_mas=latent_mas,
                    cv=cv_calib,
                    latent_only=cfg.latent_only,
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
                    ).to(cfg.device)

                    # Step 6a: Reset layer importance and run actual evaluation
                    latent_evaluator.layer_importance_total = defaultdict(list)
                    results = latent_evaluator.test(model_A, cv, limit=cfg.limit)

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
                        ).to(cfg.device)
                        latent_evaluator.layer_importance_total = defaultdict(list)
                        result = latent_evaluator.test(model_A, cv_i, limit=cfg.limit)
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
                ).to(cfg.device)
                latent_evaluator = LatentCommunicationEvaluator(
                    evaluator=evaluator,
                    tokenizer=tokenizer,
                    use_wandb=cfg.use_wandb,
                    max_input_length=cfg.max_input_length,
                    latent_mas=latent_mas,
                    cv=cv,
                    latent_only=cfg.latent_only,
                    response_log_path=response_log_path,
                )

                if not cfg.do_layer_curve:
                    results = latent_evaluator.test(model_A, cv, limit=cfg.limit)
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
                        ).to(cfg.device)
                        result = latent_evaluator.test(model_A, cv_i, limit=cfg.limit)
                        results.append(result)
                    logging.info(f"Mode 2 MANUAL layer_curve results: {results}")
                    if cfg.use_wandb:
                        wandb.log({"latent_layer_curve_results": results})

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
