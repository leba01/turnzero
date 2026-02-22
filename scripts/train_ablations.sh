#!/usr/bin/env bash
# Train all 15 ablation models (3 loss modes × 5 seeds).
#
# Usage:
#     cd /home/walter/CS229/turnzero
#     bash scripts/train_ablations.sh
#
# Output structure:
#     outputs/runs/ablation_{a,b,c}_{001..005}/best.pt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

ABLATIONS=("a" "b" "c")
MEMBERS=(001 002 003 004 005)

echo "========================================"
echo "  TurnZero Ablation Training (15 runs)"
echo "========================================"
echo "Root: $ROOT"
echo ""

for abl in "${ABLATIONS[@]}"; do
    for mem in "${MEMBERS[@]}"; do
        CONFIG="$ROOT/configs/ablation_${abl}/member_${mem}.yaml"
        OUT_DIR="$ROOT/outputs/runs/ablation_${abl}_${mem}"

        if [ -f "$OUT_DIR/best.pt" ]; then
            echo "[SKIP] ablation_${abl}_${mem} — best.pt already exists"
            continue
        fi

        echo ""
        echo "────────────────────────────────────────"
        echo "Training: ablation_${abl}_${mem}"
        echo "  Config: $CONFIG"
        echo "  Output: $OUT_DIR"
        echo "────────────────────────────────────────"

        $PYTHON -m turnzero.cli train --config "$CONFIG" --out_dir "$OUT_DIR"
    done
done

echo ""
echo "========================================"
echo "  All ablation training complete."
echo "========================================"
