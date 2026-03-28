#!/bin/bash
# 在 tmux 里跑全部 DART 导出，ToDesk 断了也不影响
#
# 用法:
#   bash scripts/run_export_tmux.sh        # 启动
#   tmux attach -t dart_export             # 重新接入查看进度
#   tmux kill-session -t dart_export       # 强制停止

SESSION="dart_export"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 如果 session 已存在，提示用户
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' 已经在跑了！"
    echo "  查看进度: tmux attach -t $SESSION"
    echo "  强制停止: tmux kill-session -t $SESSION"
    exit 0
fi

# 创建 tmux session 跑导出任务
tmux new-session -d -s "$SESSION" bash -c "
cd '$PROJECT_DIR'
mkdir -p logs/trajs

RUNS=(
  '2026-03-27_18-55-16_dance1_subject1_personal_probe_5000iter_4096env_2026-03-27 model_4999.pt'
  '2026-03-27_21-08-22_walk1_subject1_personal_probe_3000iter_4096env_2026-03-27 model_2999.pt'
  '2026-03-27_23-53-08_walk2_subject1_personal_probe_3000iter_4096env_2026-03-27 model_2999.pt'
)

NUM_ENVS=4096
NOISE_STD=0.05
TASK='Tracking-Flat-G1-v0'

echo '========================================='
echo ' DART Export in tmux (ToDesk-safe)'
echo ' 3 runs, 4096 envs, noise=0.05'
echo '========================================='
echo ''

OK=0
FAIL=0
TOTAL=\${#RUNS[@]}

for entry in \"\${RUNS[@]}\"; do
    run_name=\$(echo \"\$entry\" | cut -d' ' -f1)
    ckpt=\$(echo \"\$entry\" | cut -d' ' -f2)
    ckpt_path=\"logs/rsl_rl/g1_flat/\$run_name/\$ckpt\"
    log_file=\"logs/trajs/export_\${run_name}.log\"

    echo ''
    echo \">>> [\$run_name]\"
    echo \"    ckpt=\$ckpt  envs=$NUM_ENVS\"
    echo \"    log -> \$log_file\"
    echo \"    started: \$(date)\"

    if [ ! -f \"\$ckpt_path\" ]; then
        echo '    ERROR: checkpoint not found'
        FAIL=\$((FAIL + 1))
        continue
    fi

    python scripts/rsl_rl/export_trajs.py \\
        --task \"\$TASK\" \\
        --checkpoint_path \"\$ckpt_path\" \\
        --num_envs $NUM_ENVS \\
        --noise_std $NOISE_STD \\
        --headless \\
        > \"\$log_file\" 2>&1
    exit_code=\$?

    echo \"    finished: \$(date)  exit=\$exit_code\"

    if [ \$exit_code -gt 128 ]; then
        sig=\$((exit_code - 128))
        echo \"    KILLED by signal \$sig\"
        FAIL=\$((FAIL + 1))
    elif [ \$exit_code -ne 0 ]; then
        echo \"    FAILED\"
        tail -5 \"\$log_file\"
        FAIL=\$((FAIL + 1))
    else
        dart_file=\$(ls -t logs/trajs/*\"\${run_name}\"*dart*.pt 2>/dev/null | head -1)
        if [ -n \"\$dart_file\" ] && [ -f \"\$dart_file\" ]; then
            size=\$(du -h \"\$dart_file\" | cut -f1)
            echo \"    OK -> \$dart_file (\$size)\"
            OK=\$((OK + 1))
        else
            echo \"    FAILED - no .pt output\"
            FAIL=\$((FAIL + 1))
        fi
    fi
done

echo ''
echo '========================================='
echo \" Done. OK=\$OK  Failed=\$FAIL  Total=\$TOTAL\"
echo ' Output: logs/trajs/'
ls -lh logs/trajs/*dart*.pt 2>/dev/null
echo '========================================='
echo ''
echo 'All done. This tmux session will stay open.'
echo 'Press Enter or Ctrl-D to close.'
read
"

echo "tmux session '$SESSION' 已启动！"
echo ""
echo "  查看进度: tmux attach -t $SESSION"
echo "  后台运行: ToDesk 断了也不会死"
echo "  强制停止: tmux kill-session -t $SESSION"
