#!/usr/bin/env bash
set -euo pipefail

MODE="textmas"
TASK="multifieldqa_en"
LIMIT=10
SEED=42
MODEL_A="meta-llama/Llama-3.1-8B-Instruct"
MODEL_B="meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS_A=0
MAX_TOKENS_B=0
LATENT_STEPS=5
ALLOW_B_THINK="auto"
DRY_RUN=false
LAYERS_LIST=""

usage() {
  echo "Usage: $0 --mode textmas|full_kv|selective_kv [options]"
  echo "  --task NAME --limit N --seed N --max_tokens_A N --max_tokens_B N"
  echo "  --latent_steps N --layers_list '1 2 3' --allow_b_think|--no_b_think --dry_run"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model_A) MODEL_A="$2"; shift 2 ;;
    --model_B) MODEL_B="$2"; shift 2 ;;
    --max_tokens_A) MAX_TOKENS_A="$2"; shift 2 ;;
    --max_tokens_B) MAX_TOKENS_B="$2"; shift 2 ;;
    --latent_steps) LATENT_STEPS="$2"; shift 2 ;;
    --layers_list) LAYERS_LIST="$2"; shift 2 ;;
    --allow_b_think) ALLOW_B_THINK=true; shift ;;
    --no_b_think) ALLOW_B_THINK=false; shift ;;
    --dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$ALLOW_B_THINK" == "auto" ]]; then
  case "$TASK" in
    hotpotqa|multifieldqa_en|2wikimqa|musique|qasper|tipsheets|countries) ALLOW_B_THINK=false ;;
    *) ALLOW_B_THINK=true ;;
  esac
fi

CMD=(python com_latent.py --test_task "$TASK" --limit "$LIMIT" --seed "$SEED"
  --model_A "$MODEL_A" --model_B "$MODEL_B" --max_tokens_B "$MAX_TOKENS_B")

case "$MODE" in
  textmas) CMD+=(--do_test_nld --max_tokens_A "$MAX_TOKENS_A") ;;
  full_kv) CMD+=(--do_test_latent --latent_steps "$LATENT_STEPS") ;;
  selective_kv)
    CMD+=(--do_test_latent --latent_kv_select --latent_steps "$LATENT_STEPS")
    if [[ -n "$LAYERS_LIST" ]]; then read -r -a LAYERS <<< "$LAYERS_LIST"; CMD+=(--layers_list "${LAYERS[@]}"); fi
    ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac
[[ "$ALLOW_B_THINK" == true ]] && CMD+=(--allow_b_think)

echo "mode=$MODE task=$TASK limit=$LIMIT seed=$SEED prompt_profile=v2 metric_version=v2"
echo "max_tokens_A=$MAX_TOKENS_A max_tokens_B=$MAX_TOKENS_B latent_steps=$LATENT_STEPS allow_b_think=$ALLOW_B_THINK"
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$DRY_RUN" == true ]] || exec "${CMD[@]}"
