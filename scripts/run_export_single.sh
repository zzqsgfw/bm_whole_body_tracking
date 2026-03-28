#!/bin/bash
# 单个 run 的 DART 导出，完全隔离运行
# 用法: bash scripts/run_export_single.sh <run_dir_name> <checkpoint_file>
# 示例: bash scripts/run_export_single.sh 2026-03-27_18-55-16_dance1_subject1_personal_probe_5000iter_4096env_2026-03-27 model_4999.pt
#
# 输出:  logs/trajs/ 下的 .pt 文件
# 日志:  logs/trajs/export_<run>.log  +  logs/trajs/export_<run>_wrapper.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RUN_NAME="${1:?用法: $0 <run_dir_name> <checkpoint_file>}"
CKPT="${2:?用法: $0 <run_dir_name> <checkpoint_file>}"
NUM_ENVS="${3:-4096}"

CKPT_PATH="logs/rsl_rl/g1_flat/$RUN_NAME/$CKPT"
LOG_FILE="logs/trajs/export_${RUN_NAME}.log"
WRAPPER_LOG="logs/trajs/export_${RUN_NAME}_wrapper.log"
mkdir -p logs/trajs

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: $CKPT_PATH not found" | tee "$WRAPPER_LOG"
    exit 1
fi

echo "=== Export DART ===" | tee "$WRAPPER_LOG"
echo "  run:  $RUN_NAME" | tee -a "$WRAPPER_LOG"
echo "  ckpt: $CKPT" | tee -a "$WRAPPER_LOG"
echo "  envs: $NUM_ENVS" | tee -a "$WRAPPER_LOG"
echo "  log:  $LOG_FILE" | tee -a "$WRAPPER_LOG"
echo "  time: $(date)" | tee -a "$WRAPPER_LOG"
echo "" | tee -a "$WRAPPER_LOG"

# 用 setsid 创建新 session，防止信号传播杀掉 bash
setsid python scripts/rsl_rl/export_trajs.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint_path "$CKPT_PATH" \
    --num_envs "$NUM_ENVS" \
    --noise_std 0.05 \
    --headless \
    > "$LOG_FILE" 2>&1 &
PID=$!
echo "  python PID: $PID (new session via setsid)" | tee -a "$WRAPPER_LOG"

# 等待完成
wait $PID 2>/dev/null
EXIT_CODE=$?

echo "" | tee -a "$WRAPPER_LOG"
echo "  finished at: $(date)" | tee -a "$WRAPPER_LOG"

if [ $EXIT_CODE -gt 128 ]; then
    SIG=$((EXIT_CODE - 128))
    SIGNAME=$(kill -l $SIG 2>/dev/null || echo "?")
    echo "  KILLED by signal $SIG ($SIGNAME)" | tee -a "$WRAPPER_LOG"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "  FAILED with exit code $EXIT_CODE" | tee -a "$WRAPPER_LOG"
    echo "  Last 10 lines of log:" | tee -a "$WRAPPER_LOG"
    tail -10 "$LOG_FILE" | tee -a "$WRAPPER_LOG"
else
    echo "  SUCCESS (exit 0)" | tee -a "$WRAPPER_LOG"
fi

# 检查产出
DART_FILE=$(ls -t logs/trajs/*"${RUN_NAME}"*dart*.pt 2>/dev/null | head -1)
if [ -n "$DART_FILE" ] && [ -f "$DART_FILE" ]; then
    SIZE=$(du -h "$DART_FILE" | cut -f1)
    echo "  output: $DART_FILE ($SIZE)" | tee -a "$WRAPPER_LOG"
else
    echo "  NO .pt output found" | tee -a "$WRAPPER_LOG"
fi
