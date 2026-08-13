#!/usr/bin/env bash
#
# Experiment 1, backbone ablation: six backbones, aligned and unaligned, five
# folds each. The ViT-S/16 aligned cell is reused from the main cross-validation
# run rather than retrained, so 55 trainings remain.
#
#     ./scripts/run_cv_ablation.sh
#
# Ordered cheapest first. Resumable: ./scripts/run_cv_ablation.sh <stamp>
# collect_cv_ablation is safe to run mid-flight; incomplete cells show as PARTIAL.

set -uo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
WORK="$(python -c 'from src.config import load_paths; print(load_paths().work_dir)')" || exit 1
LOGDIR="$WORK/logs/ablcv_$STAMP"
DONE="$LOGDIR/.done"
RESDIR="$WORK/results/cv_ablation"
CKPTDIR="$WORK/checkpoints/cv_ablation"
mkdir -p "$LOGDIR" "$DONE" "$RESDIR" "$CKPTDIR"

N_FOLDS=5

if ! python -c "import torch, timm, insightface" 2>/dev/null; then
    echo "ERROR: missing torch / timm / insightface"
    exit 1
fi

for f in $(seq 0 $((N_FOLDS-1))); do
    if [[ ! -f "$WORK/results/cv/fold${f}_test_predictions.csv" ]]; then
        echo "ERROR: main cross-validation fold $f predictions missing."
        echo "  The ViT-S/16 aligned cell is reused from that run. Run scripts/run_cv.sh first."
        exit 1
    fi
done

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
    local t0=$SECONDS
    if "$@" > "$log" 2>&1; then
        touch "$marker"
        echo "  [ok]   $name  ($(( (SECONDS-t0)/60 )) min)"
    else
        echo "  [FAIL] $name  ($(( (SECONDS-t0)/60 )) min) — see $log"
        tail -n 12 "$log" | sed 's/^/      | /'
    fi
}

# "<timm name>|<run aligned?>|<run unaligned?>"
CELLS=(
    "vit_small_patch16_224.augreg_in21k_ft_in1k|no|yes"
    "efficientnet_b0|yes|yes"
    "efficientnetv2_rw_s.ra2_in1k|yes|yes"
    "efficientnet_b4|yes|yes"
    "vit_base_patch16_224.augreg_in21k_ft_in1k|yes|yes"
    "vit_small_patch14_dinov2.lvd142m|yes|yes"
)

echo "stamp : $STAMP"
echo "logs  : $LOGDIR"

for entry in "${CELLS[@]}"; do
    bb="${entry%%|*}"
    rest="${entry#*|}"
    do_aligned="${rest%%|*}"
    do_plain="${rest##*|}"
    short="${bb%%.*}"

    if [[ "$do_plain" == "yes" ]]; then
        banner "$short, unaligned, $N_FOLDS folds"
        for f in $(seq 0 $((N_FOLDS-1))); do
            step "${short}_fold${f}" \
                python -m src.train_cv --fold "$f" --backbone "$bb" \
                    --ckpt_dir "$CKPTDIR" --results_dir "$RESDIR"
        done
    fi

    if [[ "$do_aligned" == "yes" ]]; then
        banner "$short, aligned, $N_FOLDS folds"
        for f in $(seq 0 $((N_FOLDS-1))); do
            step "${short}_aligned_fold${f}" \
                python -m src.train_cv --fold "$f" --backbone "$bb" --align \
                    --ckpt_dir "$CKPTDIR" --results_dir "$RESDIR"
        done
    fi

    # refresh the pooled table after every backbone, so a partial run is usable
    python -m src.eval.collect_cv_ablation > "$LOGDIR/ablation_table.log" 2>&1 || true
done

banner "POOLING"
python -m src.eval.collect_cv_ablation 2>&1 | tee "$LOGDIR/CV_ABLATION.log"

banner "DONE"
echo "table : $WORK/results/CV_ABLATION.txt"
echo "to resume: ./scripts/run_cv_ablation.sh $STAMP"
