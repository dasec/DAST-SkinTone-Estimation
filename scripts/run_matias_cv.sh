#!/usr/bin/env bash
# Experiment 5, external baseline: zero-shot and fine-tuned on the folds of
# the main cross-validation. Zero-shot needs no training, so the scored image
# set is evaluated once; fine-tuning runs all five folds.
# Resumable: ./scripts/run_matias_cv.sh <stamp>

set -uo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
WORK="$(python -c 'from src.config import load_paths; print(load_paths().work_dir)')" || exit 1
LOGDIR="$WORK/logs/matiascv_$STAMP"
DONE="$LOGDIR/.done"
mkdir -p "$LOGDIR" "$DONE"

banner() {
    echo
    echo "===================================================================="
    echo "  $*"
    echo "  $(date '+%F %T')"
    echo "===================================================================="
}

step() {
    local name="$1"; shift
    local marker="$DONE/$name" log="$LOGDIR/$name.log"
    if [[ -f "$marker" ]]; then echo "  [skip] $name"; return 0; fi
    banner "$name"
    local t0=$SECONDS
    if "$@" > "$log" 2>&1; then
        touch "$marker"
        echo "  [ok]   $name  ($(( (SECONDS-t0)/60 )) min)"
    else
        echo "  [FAIL] $name, see $log"
        tail -n 15 "$log" | sed 's/^/      | /'
    fi
}

echo "stamp : $STAMP"
echo "logs  : $LOGDIR"

step matias_zeroshot_cv python -m src.baselines.eval_matias --cv

for fold in 0 1 2 3 4; do
    step "matias_cv_fold$fold" python -m src.baselines.train_matias_cv --fold "$fold" --align
done

step matias_cv_pooled python -m src.baselines.collect_matias_cv

banner "DONE"
echo "report : $WORK/results/MATIAS_CV.txt"
echo "to resume: ./scripts/run_matias_cv.sh $STAMP"
