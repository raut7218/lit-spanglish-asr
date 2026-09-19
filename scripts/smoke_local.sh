#!/usr/bin/env bash
# End-to-end smoke test on a tiny model: prepare -> unit tests -> train -> eval -> export/CT2 -> zip -> run zip -> score.
# Usage: scripts/smoke_local.sh MIAMI_DIR DEV_DIR [WORKDIR]
set -euo pipefail
MIAMI=${1:?miami dir}; DEV=${2:?dev dir}; WORK=${3:-/tmp/lit_smoke}
PY=${PYTHON:-python}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src"
CFG="--config configs/smoke.yaml --set data_dir=$WORK/prepared out_dir=$WORK/run"

rm -rf "$WORK"; mkdir -p "$WORK"
echo "=== 1/7 prepare data (3 conversations)"; $PY -m lit.prepare_data --miami_dir "$MIAMI" --dev_dir "$DEV" --out_dir "$WORK/prepared" --max_convs 3 --holdout_hours 0.05
echo "=== 2/7 unit tests";                     $PY -m pytest -q tests
echo "=== 3/7 train (tiny)";                   $PY -m lit.train $CFG
echo "=== 4/7 evaluate adapter";               $PY -m lit.evaluate --model openai/whisper-tiny --adapter "$WORK/run/final_adapter" --data_dir "$WORK/prepared" --max_clips 12 --dtype fp32
echo "=== 5/7 export + CTranslate2";           $PY -m lit.export --model openai/whisper-tiny --adapter "$WORK/run/final_adapter" --data_dir "$WORK/prepared" --out "$WORK/export" --quantization float32
echo "=== 6/7 build submission.zip";           $PY scripts/make_submission.py --export "$WORK/export" --out "$WORK/submission.zip" --cfg '{"beam_size": 2, "language": "es", "compute_type": "float32"}'
echo "=== 7/7 run zip like the platform";      $PY scripts/run_submission_local.py --zip "$WORK/submission.zip" --prepared "$WORK/prepared" --dev_raw "$DEV" --n 12
echo "SMOKE TEST PASSED"
