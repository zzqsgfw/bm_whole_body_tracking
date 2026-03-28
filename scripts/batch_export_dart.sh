#!/bin/bash
# Batch DART-style trajectory export for all trained runs.
# Iterates over all run directories under logs/rsl_rl/g1_flat/,
# picks the latest checkpoint, and runs export_trajs.py with DART noise.
#
# Usage:
#   cd whole_body_tracking
#   bash scripts/batch_export_dart.sh [--num_envs 64] [--noise_std 0.05]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/g1_flat"
EXPORT_SCRIPT="$SCRIPT_DIR/rsl_rl/export_trajs.py"

# defaults (overridable via env vars or CLI)
NUM_ENVS="${NUM_ENVS:-4096}"
NOISE_STD="${NOISE_STD:-0.05}"
ACTION_CLIP="${ACTION_CLIP:-1.0}"
TASK="${TASK:-Tracking-Flat-G1-v0}"

# parse optional CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)   NUM_ENVS="$2"; shift 2 ;;
        --noise_std)  NOISE_STD="$2"; shift 2 ;;
        --action_clip) ACTION_CLIP="$2"; shift 2 ;;
        --task)       TASK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo " DART Batch Export"
echo " num_envs=$NUM_ENVS  noise_std=$NOISE_STD  clip=$ACTION_CLIP"
echo "========================================="
echo ""

# find all run directories (skip _baseline)
RUN_DIRS=()
for d in "$LOG_ROOT"/2026-*; do
    [ -d "$d" ] && RUN_DIRS+=("$d")
done

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "[ERROR] No run directories found under $LOG_ROOT"
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s):"
for d in "${RUN_DIRS[@]}"; do echo "  $(basename "$d")"; done
echo ""

OK=0
FAILED=0

for run_dir in "${RUN_DIRS[@]}"; do
    run_name="$(basename "$run_dir")"

    # find latest checkpoint
    latest_ckpt=$(ls "$run_dir"/model_*.pt 2>/dev/null \
        | sed 's/.*model_//' | sed 's/\.pt$//' \
        | sort -n | tail -1)

    if [ -z "$latest_ckpt" ]; then
        echo "[$run_name] SKIP - no checkpoints found"
        FAILED=$((FAILED + 1))
        continue
    fi

    ckpt_path="$run_dir/model_${latest_ckpt}.pt"
    echo "[$run_name] checkpoint: model_${latest_ckpt}.pt"

    # run export
    python "$EXPORT_SCRIPT" \
        --task "$TASK" \
        --checkpoint_path "$ckpt_path" \
        --num_envs "$NUM_ENVS" \
        --noise_std "$NOISE_STD" \
        --action_clip "$ACTION_CLIP" \
        --headless 2>&1 | grep -E "\[INFO\]|\[WARN\]|step|traj|Error|Traceback" || true

    # check if output was created
    dart_file=$(ls -t "$PROJECT_DIR/logs/trajs/"*"${run_name}"*dart*.pt 2>/dev/null | head -1)
    if [ -n "$dart_file" ]; then
        size=$(du -h "$dart_file" | cut -f1)
        echo "[$run_name] OK -> $dart_file ($size)"
        OK=$((OK + 1))
    else
        echo "[$run_name] FAILED - no output .pt found"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "========================================="
echo " Done. OK: $OK  Failed: $FAILED  Total: ${#RUN_DIRS[@]}"
echo " Output dir: $PROJECT_DIR/logs/trajs/"
echo "========================================="
