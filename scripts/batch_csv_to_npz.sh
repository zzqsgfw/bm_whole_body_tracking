#!/bin/bash
# Batch convert all LAFAN1 CSV files to NPZ format
# Usage: bash scripts/batch_csv_to_npz.sh
#
# Each conversion launches Isaac Sim which may hang on exit,
# so we use timeout + file existence check to move on.

CSV_DIR="data/lafan1_retargeted/g1"
NPZ_DIR="data/npz"
INPUT_FPS=30
TIMEOUT_SEC=180  # 3 minutes per file should be plenty

mkdir -p "$NPZ_DIR"

TOTAL=$(ls "$CSV_DIR"/*.csv 2>/dev/null | wc -l)
COUNT=0
SKIPPED=0
OK=0
FAILED=0

for csv_file in "$CSV_DIR"/*.csv; do
    name=$(basename "$csv_file" .csv)
    npz_file="$NPZ_DIR/${name}.npz"
    COUNT=$((COUNT + 1))

    if [ -f "$npz_file" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] Converting: $name ..."
    timeout "$TIMEOUT_SEC" python scripts/csv_to_npz.py \
        --input_file "$csv_file" \
        --input_fps "$INPUT_FPS" \
        --output_name "$name" \
        --output_file "$npz_file" \
        --headless 2>&1 | grep -E "\[INFO\]|Motion loaded|Motion interpolated|Error"

    if [ -f "$npz_file" ]; then
        echo "[$COUNT/$TOTAL] OK: $name ($(du -h "$npz_file" | cut -f1))"
        OK=$((OK + 1))
    else
        echo "[$COUNT/$TOTAL] FAILED: $name"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
echo "Done. Total: $TOTAL, Skipped: $SKIPPED, OK: $OK, Failed: $FAILED"
echo "========================================="
