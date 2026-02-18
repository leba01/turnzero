#!/bin/bash
# Train all 5 ensemble members sequentially
# Each member has identical architecture but a different random seed.
# Seeds: 001→42, 002→137, 003→256, 004→512, 005→777
set -e
for i in 001 002 003 004 005; do
  echo "=== Training ensemble member $i ==="
  turnzero train \
    --config configs/ensemble/member_${i}.yaml \
    --out_dir outputs/runs/ensemble_${i}
done
echo "=== All 5 members trained ==="
