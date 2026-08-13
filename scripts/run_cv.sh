#!/usr/bin/env bash
#
# Experiment 1: 5-fold subject-wise cross-validation of the main model.
#
#     ./scripts/run_cv.sh
#
# Resumable: a finished fold drops a marker under <work_dir>/logs/cv_<stamp>/.done.
# Rerun with the same stamp to continue where it stopped:
#     ./scripts/run_cv.sh 20260101_1200

set -uo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
WORK="$(python -c 'from src.config import load_paths; print(load_paths().work_dir)')" || exit 1
LOGDIR="$WORK/logs/cv_$STAMP"
DONE="$LOGDIR/.done"
mkdir -p "$LOGDIR" "$DONE"

N_FOLDS=5

if ! python -c "import torch, timm, insightface" 2>/dev/null; then
    echo "ERROR: missing torch / timm / insightface"
    exit 1
fi

if [[ ! -f "$WORK/dast10_table.csv" ]]; then
    echo "ERROR: table missing — run 'python -m src.data.build_table' first"
    exit 1
fi

banner() {
    echo
    echo "===================================================================="
    echo "  $*"
    echo "  $(date '+%F %T')"
    echo "===================================================================="
}

echo "stamp : $STAMP"
echo "logs  : $LOGDIR"

python -m src.cv_folds > "$LOGDIR/fold_layout.log" 2>&1
sed -n '1,12p' "$LOGDIR/fold_layout.log"

for fold in $(seq 0 $((N_FOLDS - 1))); do
    marker="$DONE/fold$fold"
    log="$LOGDIR/fold$fold.log"
    if [[ -f "$marker" ]]; then
        echo "  [skip] fold $fold"
        continue
    fi
    banner "FOLD $fold of $N_FOLDS"
    t0=$SECONDS
    if python -m src.train_cv --fold "$fold" --align > "$log" 2>&1; then
        touch "$marker"
        echo "  [ok]   fold $fold  ($(( (SECONDS-t0)/60 )) min)"
    else
        echo "  [FAIL] fold $fold  ($(( (SECONDS-t0)/60 )) min) — see $log"
        tail -n 15 "$log" | sed 's/^/      | /'
    fi
done

banner "POOLING FOLDS"
python -m src.eval.collect_cv 2>&1 | tee "$LOGDIR/CV_REPORT.log"

banner "DONE"
echo "report : $WORK/results/CV_REPORT.txt"
echo "to resume: ./scripts/run_cv.sh $STAMP"
