# Skin Tone Estimation for Performance Differential Analysis in Biometric Systems

Code for the BIOSIG 2026 paper of the same name.

The method regresses continuous CIELAB `L*a*b*` skin tone directly from
ICAO-compliant face images with a ViT-S/16, supervised by contact colorimeter
measurements rather than by perceptual class labels. Predictions are assigned to
Cook et al. (2025) skin tone scale (CST) bins by nearest centroid in full
CIELAB. Regressing the measurement and binning afterwards keeps a continuous,
device-referenced quantity available for demographic differential analysis,
which a direct classifier discards. Evaluation is five-fold subject-wise
cross-validation, so every subject is held out exactly once and the spread
across folds gives error bars that a single split cannot.

## Datasets

**Primary dataset (DAST-10).** Access-restricted and not included here, in whole
or in part. It holds ICAO-compliant captures at five fixed exposure levels
(EV offsets −5, −3, 0, +3, +5) on two cameras, with a per-subject face
colorimeter `L*a*b*` reading and a CST class. Access is governed by a data
protection agreement and must be requested from the data controller; this
repository ships no images, no metadata table and no colorimeter values.

**CHROMA-FIT.** Used for the cross-dataset transfer experiment. Obtain it from
its publishers under their own terms.

**External baseline.** Experiment 5 needs the released checkpoint and the
per-class training images of the prototype-based ViT-S/16 classifier of
Matias et al. Request them from that work's authors.

The pipeline reads its data through `configs/paths.yaml`; nothing is hardcoded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp configs/paths.example.yaml configs/paths.yaml
$EDITOR configs/paths.yaml        # point at your data and choose a work_dir
```

Python 3.9 or newer. A CUDA GPU is assumed; everything falls back to CPU but the
trainings become impractical. Hyperparameters live in `configs/default.yaml` and
are shared by the main model, the ablation and the fine-tuned baseline, so those
comparisons isolate the backbone rather than the recipe.

All commands are run from the repository root and accept `--paths` and
`--params` to override the config files.

## Data preparation

```bash
python -m src.data.verify_dataset          # integrity gate on the release
python -m src.data.build_table             # -> <work_dir>/dast10_table.csv
python -m src.data.check_split             # leakage gate
python -m src.data.check_exposure_labels   # luminance cross-check of the labels
```

`build_table` applies the corrections in `configs/exposure_overrides.yaml`. Two
images in the release carry an exposure label that contradicts their own
luminance; without the corrections the per-exposure counts, and every
per-exposure result, differ from the published table. The loader warns if the
file is missing or empty.

## Experiments

| # | Experiment | Command |
|---|---|---|
| 1 | ViT-S/16 regression, five-fold subject-wise CV | `./scripts/run_cv.sh` |
| 1b | Backbone ablation over the same folds | `./scripts/run_cv_ablation.sh` |
| 2 | CST bins under the three distance rules | `python -m src.eval.collect_cv` |
| 3 | Robustness across the five exposure levels | `python -m src.eval.exposure_breakdown` |
| 4 | Cross-dataset transfer with mean-shift correction | `./scripts/run_cv_transfer.sh` |
| 5 | Pixel baseline | `python -m src.baselines.run_stone_baseline --cv` |
| 5 | External ViT-S/16 baseline, zero-shot and fine-tuned | `./scripts/run_matias_cv.sh` |
| 6 | Occlusion sensitivity and EigenCAM | `./scripts/run_cv_transfer.sh` |

Experiment 1 must run first: 1b, 4 and 6 reuse its five fold checkpoints rather
than retraining with a different seed, and refuse to start if they are missing.
Experiment 4 also needs `python -m src.transfer.build_merged_table`.
The `run_*.sh` drivers are resumable — rerun with the stamp they print to
continue after an interruption.

## Which script produces which number

| Reported quantity | Produced by | Written to |
|---|---|---|
| Per-channel MAE / RMSE / R² of `L*`, `a*`, `b*`; ΔE76 | `src.eval.collect_cv` | `results/CV_REPORT.txt` |
| Per-fold spread (mean, std, min, max) | `src.eval.collect_cv` | `results/CV_REPORT.txt` |
| Exact match and within-one-bin under the 1D, 2D and 3D rules | `src.eval.collect_cv` | `results/CV_REPORT.txt` |
| Exact-match ceiling of the 2D rule | `src.eval.ceiling` | stdout |
| Accuracy and bias per exposure level, and its LaTeX rows | `src.eval.exposure_breakdown` | stdout |
| Backbone ablation table, including DINOv2 R²(`b*`) | `src.eval.collect_cv_ablation` | `results/CV_ABLATION.txt` |
| Zero-shot transfer, per fold and the drop in R²(`a*`) | `src.transfer.eval_zeroshot_chroma_cv` | `results/cv_zeroshot_chroma.csv` |
| Estimated per-channel device offset | `src.transfer.train_crossdataset_cv` | `results/cv_cross/fold*.csv` |
| Transfer after correction, and the no-shift control | `src.transfer.collect_transfer_interp` | `results/CV_TRANSFER_INTERP.txt` |
| Pixel baseline accuracy, overall and per exposure level | `src.baselines.run_stone_baseline` | stdout, `results/stone_cst_cv.csv` |
| External baseline zero-shot vs fine-tuned vs ours | `src.baselines.collect_matias_cv` | `results/MATIAS_CV.txt` |
| Occlusion sensitivity per region, face/background ratio | `src.interpret.region_sensitivity_cv` | `results/CV_TRANSFER_INTERP.txt` |
| EigenCAM and occlusion figure | `src.interpret.eigencam_cv` | `results/eigencam_cv*/` |

Paths are relative to the `work_dir` set in `configs/paths.yaml`.

## Notes on reading the results

**The 2D rule has a ceiling well below 100%.** Ground-truth bins are assigned by
nearest CST centroid in full CIELAB, but predictions are binned on `L*`+`b*`,
because `a*` is not reliably predictable from the images. A predictor returning
the exact colorimeter values would still not reach 100% exact match, and some
bins are structurally unreachable. Read exact match against the ceiling that
`src.eval.ceiling` reports, not against 100%.

**Bins with too few subjects are never scored.** A CST bin with fewer subjects
than `min_subjects_to_test` cannot be cross-validated: with two subjects, one is
in test while the other is in train, so the score measures whether a single
example generalises. Those subjects are pinned to the training set of every fold
instead. This is why the number of scored subjects is smaller than the number of
subjects in the dataset.

**Trained checkpoints are not distributed.** Neither are prediction files,
reports or figures. Everything under `work_dir` is produced by running the
commands above on data you have obtained yourself.

## Licence

MIT, see `LICENSE`. The licence covers this code only, not any dataset.
