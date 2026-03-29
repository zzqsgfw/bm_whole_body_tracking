#!/bin/bash
# 完全脱离终端跑 DART 导出，ToDesk/终端断了都不影响
#
# 用法:
#   bash scripts/run_export_detached.sh       # 启动全部3个run（顺序执行）
#   tail -f logs/trajs/dart_master.log        # 查看总进度
#   tail -f logs/trajs/export_<run>.log       # 查看单个run详情
#   cat logs/trajs/dart_master.pid            # 查看主进程PID
#   kill $(cat logs/trajs/dart_master.pid)    # 停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/trajs

MASTER_LOG="logs/trajs/dart_master.log"
PID_FILE="logs/trajs/dart_master.pid"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "已经在跑了！PID=$OLD_PID"
        echo "  查看进度: tail -f $MASTER_LOG"
        echo "  停止:     kill $OLD_PID"
        exit 0
    fi
fi

# 内联 worker 脚本
nohup bash -c '
cd "'"$PROJECT_DIR"'"
echo $$ > "'"$PID_FILE"'"

MASTER_LOG="'"$MASTER_LOG"'"
exec >> "$MASTER_LOG" 2>&1

echo "========================================="
echo "DART Export started: $(date)"
echo "PID: $$"
echo "========================================="

RUNS=(
  "2026-03-27_21-08-22_walk1_subject1_personal_probe_3000iter_4096env_2026-03-27 model_2999.pt"
)

OK=0; FAIL=0

for entry in "${RUNS[@]}"; do
    run_name=$(echo "$entry" | cut -d" " -f1)
    ckpt=$(echo "$entry" | cut -d" " -f2)
    ckpt_path="logs/rsl_rl/g1_flat/$run_name/$ckpt"
    log_file="logs/trajs/export_${run_name}.log"

    echo ""
    echo ">>> [$run_name] started: $(date)"

    python scripts/rsl_rl/export_trajs.py \
        --task Tracking-Flat-G1-v0 \
        --checkpoint_path "$ckpt_path" \
        --num_envs 256 \
        --noise_std 0.05 \
        --headless \
        > "$log_file" 2>&1
    ec=$?

    if [ $ec -gt 128 ]; then
        sig=$((ec - 128))
        echo "    KILLED signal=$sig at $(date)"
        FAIL=$((FAIL + 1))
    elif [ $ec -ne 0 ]; then
        echo "    FAILED exit=$ec at $(date)"
        tail -3 "$log_file"
        FAIL=$((FAIL + 1))
    else
        dart_file=$(ls -t logs/trajs/*"${run_name}"*dart*.pt 2>/dev/null | head -1)
        if [ -n "$dart_file" ]; then
            echo "    OK -> $dart_file ($(du -h "$dart_file" | cut -f1)) at $(date)"
            OK=$((OK + 1))
        else
            echo "    FAILED - no output at $(date)"
            FAIL=$((FAIL + 1))
        fi
    fi
done

echo ""
echo "========================================="
echo "Done: OK=$OK Failed=$FAIL Total=${#RUNS[@]}"
echo "Finished: $(date)"
ls -lh logs/trajs/*dart*.pt 2>/dev/null
echo "========================================="
rm -f "'"$PID_FILE"'"
' </dev/null > /dev/null 2>&1 &
disown

sleep 0.5
PID=$(cat "$PID_FILE" 2>/dev/null || echo "$!")
echo "已启动！完全脱离终端运行。"
echo ""
echo "  主进程 PID: $PID"
echo "  查看总进度: tail -f logs/trajs/dart_master.log"
echo "  查看单run:  tail -f logs/trajs/export_<run>.log"
echo "  停止:       kill $PID"
echo ""
echo "ToDesk 断了、终端关了都不影响。"
