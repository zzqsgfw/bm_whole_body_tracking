#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for remaining walk trajectories on an 8-GPU cluster.
#
# Default behavior:
#   - trains all data/npz/walk*.npz
#   - skips trajectories that already have model_{MAX_ITERATIONS-1}.pt
#   - launches one worker per GPU, each worker runs its assigned motions sequentially
#
# Examples:
#   bash scripts/train_walk_parallel_8gpu.sh
#   MAX_ITERATIONS=10 bash scripts/train_walk_parallel_8gpu.sh
#   GPU_IDS="0 1 2 3" MAX_ITERATIONS=10 bash scripts/train_walk_parallel_8gpu.sh
#   INCLUDE_MOTIONS="walk2_subject3,walk3_subject1" MAX_ITERATIONS=10 bash scripts/train_walk_parallel_8gpu.sh
#   SKIP_COMPLETED=0 MAX_ITERATIONS=10 bash scripts/train_walk_parallel_8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

ISAACLAB_SH="${ISAACLAB_SH:-../IsaacLab/isaaclab.sh}"
TASK="${TASK:-Tracking-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
COMPLETION_ITERATIONS="${COMPLETION_ITERATIONS:-$MAX_ITERATIONS}"
DRY_RUN="${DRY_RUN:-0}"
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-beyondmimic-tracking}"
RUN_TAG="${RUN_TAG:-parallelwalk}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
INCLUDE_MOTIONS="${INCLUDE_MOTIONS:-}"
EXCLUDE_MOTIONS="${EXCLUDE_MOTIONS:-}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-3473219558-south-university-of-science-and-technology}"
export WANDB_USERNAME="${WANDB_USERNAME:-3473219558-south-university-of-science-and-technology}"

source scripts/wandb_env.sh
export WANDB_MODE

if [[ ! -x "$ISAACLAB_SH" ]]; then
  echo "[ERROR] IsaacLab launcher not found: $ISAACLAB_SH"
  exit 1
fi

IFS=' ' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_STR"
GPU_COUNT="${#GPU_IDS_ARR[@]}"
if [[ "$GPU_COUNT" -eq 0 ]]; then
  echo "[ERROR] No GPU ids provided."
  exit 1
fi

FINAL_CKPT="model_$((COMPLETION_ITERATIONS - 1)).pt"
LAUNCH_LOG_DIR="logs/cluster_launchers"
mkdir -p "$LAUNCH_LOG_DIR"
LAUNCH_LOG="$LAUNCH_LOG_DIR/train_walk_parallel_${MAX_ITERATIONS}iter_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "[INFO] repo_dir=$REPO_DIR" | tee -a "$LAUNCH_LOG"
echo "[INFO] isaaclab_sh=$ISAACLAB_SH" | tee -a "$LAUNCH_LOG"
echo "[INFO] task=$TASK num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS completion_iterations=$COMPLETION_ITERATIONS gpu_ids=$GPU_IDS_STR dry_run=$DRY_RUN" | tee -a "$LAUNCH_LOG"
echo "[INFO] wandb_entity=${WANDB_ENTITY:-<unset>} wandb_username=${WANDB_USERNAME:-<unset>} project=$LOG_PROJECT_NAME" | tee -a "$LAUNCH_LOG"

mapfile -t ALL_WALK_FILES < <(find data/npz -maxdepth 1 -type f -name 'walk*.npz' | sort)
if [[ "${#ALL_WALK_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No walk npz files found under data/npz." | tee -a "$LAUNCH_LOG"
  exit 1
fi

contains_csv_item() {
  local item="$1"
  local csv="$2"
  [[ ",${csv}," == *",${item},"* ]]
}

is_completed_for_iters() {
  local motion_base="$1"
  find logs/rsl_rl/g1_flat -maxdepth 1 -type d -name "*_${motion_base}_*${COMPLETION_ITERATIONS}iter*" | while read -r run_dir; do
    if [[ -f "$run_dir/$FINAL_CKPT" ]]; then
      echo "$run_dir"
      return 0
    fi
  done
}

TRAIN_QUEUE=()
for motion_path in "${ALL_WALK_FILES[@]}"; do
  motion_file="$(basename "$motion_path")"
  motion_base="${motion_file%.npz}"

  if [[ -n "$INCLUDE_MOTIONS" ]] && ! contains_csv_item "$motion_base" "$INCLUDE_MOTIONS"; then
    continue
  fi
  if [[ -n "$EXCLUDE_MOTIONS" ]] && contains_csv_item "$motion_base" "$EXCLUDE_MOTIONS"; then
    echo "[SKIP][excluded] $motion_base" | tee -a "$LAUNCH_LOG"
    continue
  fi
  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    completed_dir="$(is_completed_for_iters "$motion_base" || true)"
    if [[ -n "$completed_dir" ]]; then
      echo "[SKIP][completed] $motion_base -> $completed_dir/$FINAL_CKPT" | tee -a "$LAUNCH_LOG"
      continue
    fi
  fi
  TRAIN_QUEUE+=("$motion_path")
done

if [[ "${#TRAIN_QUEUE[@]}" -eq 0 ]]; then
  echo "[INFO] Nothing to train. All selected walk trajectories already have $FINAL_CKPT." | tee -a "$LAUNCH_LOG"
  exit 0
fi

echo "[INFO] motions_to_train=${#TRAIN_QUEUE[@]}" | tee -a "$LAUNCH_LOG"
printf '  - %s\n' "${TRAIN_QUEUE[@]##*/}" | tee -a "$LAUNCH_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, not launching jobs." | tee -a "$LAUNCH_LOG"
  echo "[INFO] Launcher log: $LAUNCH_LOG" | tee -a "$LAUNCH_LOG"
  exit 0
fi

WORKER_DIR="$LAUNCH_LOG_DIR/workers_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$WORKER_DIR"

declare -a PIDS=()
cleanup() {
  echo "[WARN] Caught signal, terminating worker processes..." | tee -a "$LAUNCH_LOG"
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup INT TERM

for worker_idx in "${!GPU_IDS_ARR[@]}"; do
  gpu_id="${GPU_IDS_ARR[$worker_idx]}"
  assigned=()
  for i in "${!TRAIN_QUEUE[@]}"; do
    if (( i % GPU_COUNT == worker_idx )); then
      assigned+=("${TRAIN_QUEUE[$i]}")
    fi
  done

  if [[ "${#assigned[@]}" -eq 0 ]]; then
    echo "[INFO] gpu=$gpu_id has no assigned motions" | tee -a "$LAUNCH_LOG"
    continue
  fi

  worker_log="$WORKER_DIR/gpu${gpu_id}.log"
  echo "[INFO] gpu=$gpu_id assigned=${#assigned[@]} motions -> $worker_log" | tee -a "$LAUNCH_LOG"
  printf '    %s\n' "${assigned[@]##*/}" | tee -a "$LAUNCH_LOG"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export WANDB_MODE
    for motion_path in "${assigned[@]}"; do
      motion_file="$(basename "$motion_path")"
      motion_base="${motion_file%.npz}"
      run_name="${motion_base}_${RUN_TAG}_${MAX_ITERATIONS}iter_${NUM_ENVS}env_$(date +%Y-%m-%d)"
      echo "=================================================="
      echo "[$(date '+%F %T')] gpu=$gpu_id START $motion_base"
      echo "run_name=$run_name"
      echo "motion_path=$motion_path"
      echo "=================================================="
      TERM=xterm "$ISAACLAB_SH" -p scripts/rsl_rl/train.py \
        --task="$TASK" \
        --motion_file="$motion_path" \
        --headless \
        --logger wandb \
        --log_project_name "$LOG_PROJECT_NAME" \
        --run_name "$run_name" \
        --max_iterations="$MAX_ITERATIONS" \
        --num_envs="$NUM_ENVS"
      echo "[$(date '+%F %T')] gpu=$gpu_id DONE $motion_base"
    done
  ) > >(tee -a "$worker_log") 2>&1 &

  PIDS+=("$!")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No worker processes were launched." | tee -a "$LAUNCH_LOG"
  exit 1
fi

echo "[INFO] launched_pids=${PIDS[*]}" | tee -a "$LAUNCH_LOG"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[ERROR] At least one worker failed. Check logs in $WORKER_DIR" | tee -a "$LAUNCH_LOG"
  exit 1
fi

echo "[INFO] All assigned walk trainings finished successfully." | tee -a "$LAUNCH_LOG"
echo "[INFO] Launcher log: $LAUNCH_LOG" | tee -a "$LAUNCH_LOG"
echo "[INFO] Worker logs: $WORKER_DIR" | tee -a "$LAUNCH_LOG"
