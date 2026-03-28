#!/bin/bash
# 批量 DART 导出 - 用户直接跑这个脚本
# 用法: cd whole_body_tracking && bash scripts/run_export_all.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/g1_flat"
EXPORT_SCRIPT="$SCRIPT_DIR/rsl_rl/export_trajs.py"
OUT_DIR="$PROJECT_DIR/logs/trajs"
mkdir -p "$OUT_DIR"

NUM_ENVS=4096
NOISE_STD=0.05
TASK="Tracking-Flat-G1-v0"

RUNS=(
  "2026-03-27_18-55-16_dance1_subject1_personal_probe_5000iter_4096env_2026-03-27 model_4999.pt"
  "2026-03-27_21-08-22_walk1_subject1_personal_probe_3000iter_4096env_2026-03-27 model_2999.pt"
  "2026-03-27_23-53-08_walk2_subject1_personal_probe_3000iter_4096env_2026-03-27 model_2999.pt"
)

echo "========================================="
echo " DART Export: ${#RUNS[@]} runs, ${NUM_ENVS} envs, noise=${NOISE_STD}"
echo "========================================="

OK=0
FAIL=0

for entry in "${RUNS[@]}"; do
    run_name=$(echo "$entry" | cut -d' ' -f1)
    ckpt=$(echo "$entry" | cut -d' ' -f2)
    ckpt_path="$LOG_ROOT/$run_name/$ckpt"
    log_file="$OUT_DIR/export_${run_name}.log"

    echo ""
    echo ">>> [$run_name] ckpt=$ckpt"
    echo "    log -> $log_file"

    if [ ! -f "$ckpt_path" ]; then
        echo "    ERROR: checkpoint not found: $ckpt_path"
        FAIL=$((FAIL + 1))
        continue
    fi

    python "$EXPORT_SCRIPT" \
        --task "$TASK" \
        --checkpoint_path "$ckpt_path" \
        --num_envs "$NUM_ENVS" \
        --noise_std "$NOISE_STD" \
        --headless \
        > "$log_file" 2>&1 &
    PID=$!
    echo "    PID=$PID"
    wait $PID
    exit_code=$?

    # decode exit code: 128+N means killed by signal N
    if [ $exit_code -gt 128 ]; then
        sig=$((exit_code - 128))
        echo "    KILLED by signal $sig ($(kill -l $sig 2>/dev/null || echo '?'))"
    else
        echo "    python exit code: $exit_code"
    fi
    # show key lines from log
    grep -E "\[INFO\]|step |Saved|FAILED|Error executing|ValueError|RuntimeError|KeyError|OOM|Segmentation" "$log_file" | grep -v "omni\." | tail -20

    # check output
    dart_file=$(ls -t "$OUT_DIR"/*"${run_name}"*dart*.pt 2>/dev/null | head -1)
    if [ -n "$dart_file" ] && [ -f "$dart_file" ]; then
        size=$(du -h "$dart_file" | cut -f1)
        echo "    OK -> $dart_file ($size)"
        OK=$((OK + 1))
    else
        echo "    FAILED (exit=$exit_code)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "========================================="
echo " Done. OK=$OK  Failed=$FAIL  Total=${#RUNS[@]}"
echo " Output: $OUT_DIR/"
echo "========================================="
