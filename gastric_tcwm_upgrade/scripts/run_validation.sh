#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q --junitxml=validation/pytest.xml
ROOT="${1:-/tmp/tcwm-validation}"
for ARM in gaussian flow survival; do
  case "$ARM" in
    gaussian) CONFIG=configs/smoke.json; EXTRA=() ;;
    flow) CONFIG=configs/smoke_flow.json; EXTRA=() ;;
    survival) CONFIG=configs/smoke_survival.json; EXTRA=(--survival --causes 2) ;;
  esac
  python scripts/run_tcwm.py --threads 1 synth --out "$ROOT/$ARM" --n 64 --image-dim 32 "${EXTRA[@]}"
  python scripts/run_tcwm.py --threads 1 train --data "$ROOT/$ARM/cohort.pt" --split "$ROOT/$ARM/split.json" --config "$CONFIG" --out "$ROOT/$ARM/run"
  python scripts/run_tcwm.py --threads 1 evaluate --data "$ROOT/$ARM/cohort.pt" --split "$ROOT/$ARM/split.json" --run "$ROOT/$ARM/run" --role test --samples 8
  for STAGE in 0 1 2; do
    HORIZONS=(); if [[ "$ARM" == survival ]]; then HORIZONS=(--horizons 12 24 36); fi
    python scripts/run_tcwm.py --threads 1 predict --bundle "$ROOT/$ARM/run/inference.pt" --query "$ROOT/$ARM/query.pt" --stage "$STAGE" --out "$ROOT/$ARM/prediction_s$STAGE.json" --samples 8 --allow-extrapolation "${HORIZONS[@]}"
  done
done
