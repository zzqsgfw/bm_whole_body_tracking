#!/bin/bash
# 8-GPU parallel DART trajectory export.
# Each run directory is assigned to a separate GPU and runs concurrently.
#
# Usage:
#   conda activate motiongen
#   cd /root/wuji_ws_0/zzq_ws/bm_whole_body_tracking
#   bash scripts/batch_export_dart_8gpu.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/g1_flat"
EXPORT_SCRIPT="$SCRIPT_DIR/rsl_rl/export_trajs.py"

NUM_ENVS="${NUM_ENVS:-256}"
NOISE_STD="${NOISE_STD:-0.05}"
TASK="${TASK:-Tracking-Flat-G1-v0}"

# Collect run directories
RUN_DIRS=()
for d in "$LOG_ROOT"/2026-*; do
    [ -d "$d" ] && RUN_DIRS+=("$d")
done

NUM_RUNS=${#RUN_DIRS[@]}
if [ "$NUM_RUNS" -eq 0 ]; then
    echo "[ERROR] No run directories found under $LOG_ROOT"
    exit 1
fi

echo "========================================="
echo " 8-GPU Parallel DART Export"
echo " num_envs=$NUM_ENVS  noise_std=$NOISE_STD"
echo " runs=$NUM_RUNS"
echo "========================================="

LOGDIR="$PROJECT_DIR/logs/export_logs"
mkdir -p "$LOGDIR"

PIDS=()
GPU=0

for run_dir in "${RUN_DIRS[@]}"; do
    run_name="$(basename "$run_dir")"

    # find latest checkpoint
    latest_ckpt=$(ls "$run_dir"/model_*.pt 2>/dev/null \
        | sed 's/.*model_//' | sed 's/\.pt$//' \
        | sort -n | tail -1)

    if [ -z "$latest_ckpt" ]; then
        echo "[$run_name] SKIP - no checkpoint"
        continue
    fi

    ckpt_path="$run_dir/model_${latest_ckpt}.pt"
    logfile="$LOGDIR/${run_name}_gpu${GPU}.log"

    echo "[GPU $GPU] $run_name  ckpt=model_${latest_ckpt}.pt  log=$logfile"

    CUDA_VISIBLE_DEVICES=$GPU python "$EXPORT_SCRIPT" \
        --task "$TASK" \
        --checkpoint_path "$ckpt_path" \
        --num_envs "$NUM_ENVS" \
        --noise_std "$NOISE_STD" \
        --headless \
        > "$logfile" 2>&1 &

    PIDS+=($!)
    GPU=$(( (GPU + 1) % 8 ))
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting..."
echo ""

# Wait for all and report results
FAILED=0
OK=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    run_name="$(basename "${RUN_DIRS[$i]}")"
    if wait "$pid"; then
        echo "[OK]     $run_name (pid=$pid)"
        OK=$((OK + 1))
    else
        echo "[FAILED] $run_name (pid=$pid) - check log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
echo " Done. OK=$OK  Failed=$FAILED  Total=${#PIDS[@]}"
echo " Trajectories: $PROJECT_DIR/logs/trajs/"
echo " Logs:         $LOGDIR/"
echo "========================================="
