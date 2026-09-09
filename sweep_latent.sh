#!/usr/bin/env bash
# Sweep multiple tasks, latent steps, and communication modes through com_latent.py.
set -uo pipefail

ALL_TASKS="hotpotqa medqa tmath multifieldqa_en twowikimqa musique qasper tipsheets countries repobench samsum mbppplus aime2024 aime2025 arc_challenge arc_easy gpqa gsm8k humanevalplus"
CORE_TASKS="hotpotqa medqa tmath multifieldqa_en"
QA_TASKS="hotpotqa multifieldqa_en twowikimqa musique qasper tipsheets countries"
MATH_TASKS="tmath aime2024 aime2025 gsm8k"
CODE_TASKS="repobench mbppplus humanevalplus"
MCQ_TASKS="medqa arc_easy arc_challenge gpqa"

MODEL_A="Qwen/Qwen3-4B"
MODEL_B="Qwen/Qwen3-4B"
TASK_INPUT="multifieldqa_en hotpotqa"
STEPS="10"
MODE="m1"
DEVICE="auto"
DEVICE_B="auto"
SEED=42
LIMIT=0
BATCH_SIZE=1
TOP_LAYERS=0.7
CALIB_SIZE=5
LAYERS_LIST=""
MAX_TOKENS_A=0
MAX_TOKENS_B=0
ALLOW_B_THINK="auto"
SPLIT_RATIO=0.5
CONTEXT_TOP_RATIO=0.7
LATENT_TOP_RATIO=0.7
TRACK_CONVERGENCE=false
SHIFT_BACK=true
DRY_RUN=false
PYTHON_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: bash sweep_latent.sh [OPTIONS]

Modes:
  m1 | full_kv                 Full-KV LatentMAS
  m2 | selective_kv            Selective-KV LatentMAS
  m3 | kvcomm                  Regular KVComm, no latent steps
  m4 | dual_kv                 Legacy depth-split routing (full KV per kept layer)
  m5 | segmented_kv            Segmented Dual-KV (independent context/latent routing)
  textmas | tx                 TextMAS baseline
  both                         m1 + m2
  all                          m3 + textmas + m1 + m2 + m4 + m5

Sweep options:
  --task, --tasks VALUE         all|core|qa|math|code|mcq or "task1 task2"
  --steps "1 2 5 10"          Latent steps for m1/m2/m4/m5 (default: 10)
  --mode MODE                  One of the modes above
  --limit N                    0 means full dataset
  --dry_run                    Print every command without running models

Model and generation:
  --model_A PATH --model_B PATH
  --python PATH                  Explicit Python interpreter
  --device VALUE --device_B VALUE --seed N --batch_size N
  --max_tokens_A N             TextMAS sender budget; 0 uses task default
  --max_tokens_B N             Receiver budget; 0 uses task default
  --allow_b_think | --no_b_think | --auto_b_think

Layer selection:
  --top_layers FLOAT --calib_size N --layers_list "4 8 12"
  --split_ratio FLOAT --context_top_ratio FLOAT --latent_top_ratio FLOAT
  --track_convergence --no_shift_back

Examples:
  bash sweep_latent.sh --task "hotpotqa tmath" --steps "1 2 5" --mode both --dry_run
  bash sweep_latent.sh --task qa --mode all --steps "1 5" --limit 10
  bash sweep_latent.sh --task "hotpotqa tmath" --mode m5 --steps "5 10" --limit 10
  bash sweep_latent.sh --task multifieldqa_en --mode textmas --max_tokens_A 256 --max_tokens_B 64
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_A) MODEL_A="$2"; shift 2 ;;
    --model_B) MODEL_B="$2"; shift 2 ;;
    --python) PYTHON_OVERRIDE="$2"; shift 2 ;;
    --task|--tasks) TASK_INPUT="$2"; shift 2 ;;
    --steps|--latent_steps) STEPS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device_B) DEVICE_B="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --top_layers) TOP_LAYERS="$2"; shift 2 ;;
    --calib_size) CALIB_SIZE="$2"; shift 2 ;;
    --layers_list) LAYERS_LIST="$2"; shift 2 ;;
    --max_tokens_A) MAX_TOKENS_A="$2"; shift 2 ;;
    --max_tokens_B) MAX_TOKENS_B="$2"; shift 2 ;;
    --allow_b_think) ALLOW_B_THINK=true; shift ;;
    --no_b_think|--no-b-think|--no_allow_b_think) ALLOW_B_THINK=false; shift ;;
    --auto_b_think) ALLOW_B_THINK=auto; shift ;;
    --split_ratio) SPLIT_RATIO="$2"; shift 2 ;;
    --context_top_ratio) CONTEXT_TOP_RATIO="$2"; shift 2 ;;
    --latent_top_ratio) LATENT_TOP_RATIO="$2"; shift 2 ;;
    --track_convergence) TRACK_CONVERGENCE=true; shift ;;
    --no_shift_back) SHIFT_BACK=false; shift ;;
    --dry_run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODE" in
  1|full_kv) MODE=m1 ;;
  2|selective_kv) MODE=m2 ;;
  3|kvcomm) MODE=m3 ;;
  4|dual_kv) MODE=m4 ;;
  5|segmented_kv|segmented_dual_kv) MODE=m5 ;;
  tx|nld) MODE=textmas ;;
  m1|m2|m3|m4|m5|textmas|both|all) ;;
  *) echo "[ERROR] Unsupported mode: $MODE" >&2; exit 2 ;;
esac

case "$TASK_INPUT" in
  all) TASK_STRING="$ALL_TASKS" ;;
  core) TASK_STRING="$CORE_TASKS" ;;
  qa) TASK_STRING="$QA_TASKS" ;;
  math) TASK_STRING="$MATH_TASKS" ;;
  code) TASK_STRING="$CODE_TASKS" ;;
  mcq) TASK_STRING="$MCQ_TASKS" ;;
  *) TASK_STRING="$TASK_INPUT" ;;
esac
read -r -a TASKS <<< "$TASK_STRING"
read -r -a LATENT_STEPS <<< "$STEPS"
read -r -a EXPLICIT_LAYERS <<< "$LAYERS_LIST"

if [[ ${#TASKS[@]} -eq 0 ]]; then echo "[ERROR] No tasks selected" >&2; exit 2; fi
if [[ ${#LATENT_STEPS[@]} -eq 0 ]]; then echo "[ERROR] No latent steps selected" >&2; exit 2; fi

if [[ -n "$PYTHON_OVERRIDE" ]]; then
  PYTHON="$PYTHON_OVERRIDE"
elif [[ -f "/mnt/disk2/miniconda3/envs/nghialt/bin/python" ]]; then
  PYTHON="/mnt/disk2/miniconda3/envs/nghialt/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "[ERROR] Python interpreter not found; pass --python PATH" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/com_latent.py"

is_b_think_task() {
  case "$1" in
    medqa|tmath|aime2024|aime2025|gsm8k|mbppplus|humanevalplus|arc_easy|arc_challenge|gpqa) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_b_think() {
  local task="$1"
  if [[ "$ALLOW_B_THINK" == true ]]; then echo true
  elif [[ "$ALLOW_B_THINK" == false ]]; then echo false
  elif is_b_think_task "$task"; then echo true
  else echo false
  fi
}

declare -a MODES
case "$MODE" in
  both) MODES=(m1 m2) ;;
  all) MODES=(m3 textmas m1 m2 m4 m5) ;;
  *) MODES=("$MODE") ;;
esac

TOTAL=0
COMPLETED=0
FAILED=0

run_one() {
  local task="$1" mode="$2" step="${3:-}"
  local think
  think="$(resolve_b_think "$task")"
  local -a args=(
    --model_A "$MODEL_A" --model_B "$MODEL_B"
    --test_task "$task" --device "$DEVICE" --device_B "$DEVICE_B"
    --seed "$SEED" --batch_size "$BATCH_SIZE"
    --max_tokens_B "$MAX_TOKENS_B"
  )
  [[ "$SHIFT_BACK" == true ]] && args+=(--shift_back)
  [[ "$think" == true ]] && args+=(--allow_b_think)
  [[ "$LIMIT" -gt 0 ]] && args+=(--limit "$LIMIT")

  case "$mode" in
    m1) args+=(--do_test_latent --latent_steps "$step") ;;
    m2)
      args+=(--do_test_latent --latent_kv_select --latent_steps "$step" --top_layers "$TOP_LAYERS" --calib_size "$CALIB_SIZE")
      [[ ${#EXPLICIT_LAYERS[@]} -gt 0 ]] && args+=(--layers_list "${EXPLICIT_LAYERS[@]}")
      ;;
    m3)
      args+=(--do_test --top_layers "$TOP_LAYERS" --calib_size "$CALIB_SIZE")
      [[ ${#EXPLICIT_LAYERS[@]} -gt 0 ]] && args+=(--layers_list "${EXPLICIT_LAYERS[@]}")
      ;;
    m4)
      args+=(--do_test_latent --dual_kv_select --latent_steps "$step"
        --split_ratio "$SPLIT_RATIO" --context_top_ratio "$CONTEXT_TOP_RATIO" --latent_top_ratio "$LATENT_TOP_RATIO")
      [[ "$TRACK_CONVERGENCE" == true ]] && args+=(--track_convergence)
      ;;
    m5)
      args+=(--do_test_latent --segmented_kv_select --latent_steps "$step"
        --context_top_ratio "$CONTEXT_TOP_RATIO" --latent_top_ratio "$LATENT_TOP_RATIO"
        --calib_size "$CALIB_SIZE")
      [[ "$TRACK_CONVERGENCE" == true ]] && args+=(--track_convergence)
      ;;
    textmas) args+=(--do_test_nld --max_tokens_A "$MAX_TOKENS_A") ;;
  esac

  TOTAL=$((TOTAL + 1))
  printf '[%s] %s/%s%s | B-think=%s | CMD: ' "$(date '+%H:%M:%S')" "$task" "$mode" "${step:+/N=$step}" "$think"
  printf '%q ' "$PYTHON" "$MAIN_SCRIPT" "${args[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == true ]]; then COMPLETED=$((COMPLETED + 1)); return 0; fi
  if "$PYTHON" "$MAIN_SCRIPT" "${args[@]}"; then
    COMPLETED=$((COMPLETED + 1))
  else
    FAILED=$((FAILED + 1))
    echo "[WARN] Run failed; continuing with the remaining sweep." >&2
  fi
}

echo "Sweep: tasks=${#TASKS[@]} [${TASKS[*]}] | modes=[${MODES[*]}] | steps=[$STEPS]"
echo "Models: $MODEL_A -> $MODEL_B | seed=$SEED | limit=$LIMIT | dry_run=$DRY_RUN"
echo "Budgets: A=$MAX_TOKENS_A (TextMAS) | B=$MAX_TOKENS_B | B-thinking=$ALLOW_B_THINK"

for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    if [[ "$mode" == m1 || "$mode" == m2 || "$mode" == m4 || "$mode" == m5 ]]; then
      for step in "${LATENT_STEPS[@]}"; do run_one "$task" "$mode" "$step"; done
    else
      run_one "$task" "$mode"
    fi
  done
done

echo "Sweep complete: total=$TOTAL completed=$COMPLETED failed=$FAILED"
[[ "$FAILED" -eq 0 ]]
