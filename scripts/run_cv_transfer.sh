#!/usr/bin/env bash
#
# Experiments 4 and 6: cross-dataset transfer, occlusion sensitivity and
# EigenCAM, all reusing the fold checkpoints from the main cross-validation run.
#
#     ./scripts/run_cv_transfer.sh
#
# Resumable: ./scripts/run_cv_transfer.sh <stamp>

set -uo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
WORK="$(python -c 'from src.config import load_paths; print(load_paths().work_dir)')" || exit 1
LOGDIR="$WORK/logs/transfer_$STAMP"
DONE="$LOGDIR/.done"
mkdir -p "$LOGDIR" "$DONE"

# Refuse to run rather than silently retraining with a different seed.
missing=0
for f in 0 1 2 3 4; do
    ck="$WORK/checkpoints/cv/cv${f}_vit_small_patch16_224_aligned_best.pth"
    if [[ ! -f "$ck" ]]; then
        echo "ERROR: missing fold checkpoint $ck"
        missing=1
    fi
done
if [[ "$missing" == "1" ]]; then
    echo
    echo "Not starting. Run scripts/run_cv.sh first."
    exit 1
fi

if ! python -c "import torch, timm, insightface, pytorch_grad_cam" 2>/dev/null; then
    echo "ERROR: missing torch / timm / insightface / pytorch_grad_cam"
    exit 1
fi

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
        echo "  [FAIL] $name  ($(( (SECONDS-t0)/60 )) min) — see $log"
        tail -n 15 "$log" | sed 's/^/      | /'
    fi
}

echo "stamp : $STAMP"
echo "logs  : $LOGDIR"

banner "PART 1 of 3 — zero-shot transfer and interpretability"
step zeroshot_chroma_cv    python -m src.transfer.eval_zeroshot_chroma_cv
step region_sensitivity_cv python -m src.interpret.region_sensitivity_cv
step eigencam_cv           python -m src.interpret.eigencam_cv --fold 0 --subjects 3

banner "PART 2 of 3 — combined training, five folds"
for f in 0 1 2 3 4; do
    step "crossdataset_cv_fold$f" python -m src.transfer.train_crossdataset_cv --fold "$f"
done

banner "PART 3 of 3 — report"
step transfer_report       python -m src.transfer.collect_transfer_interp

banner "DONE"
echo "report : $WORK/results/CV_TRANSFER_INTERP.txt"
echo "to resume: ./scripts/run_cv_transfer.sh $STAMP"
