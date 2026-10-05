# Skin Tone Estimation for Performance Differential Analysis in Biometric Systems

**Authors:** L. Saladino, A. Dörsch, F. Stockhardt, C. Busch

## BIOSIG 2026

Code and models of the paper "Skin Tone Estimation for Performance Differential
Analysis in Biometric Systems", presented at BIOSIG 2026.

<!-- TODO: replace with the link to the published paper. -->
Paper: [link to be added](TODO)

## Overview

A ViT-S/16 regresses continuous CIELAB `L*a*b*` skin tone from ICAO-compliant
face images, supervised by contact colorimeter measurements. Each prediction is
then assigned to the skin tone scale of Cook et al. (2025) (CST) by nearest
centroid in full CIELAB.

| Path | Content |
|---|---|
| `src/` | model, dataset, training, cross-validation, baselines, transfer and interpretability code |
| `src/inference.py` | predicts `L*a*b*` and the CST bin for a folder of images |
| `analysis/` | error analysis by CST bin and by camera |
| `scripts/` | drivers for the cross-validation experiments |
| `configs/` | hyperparameters, path template and exposure-label corrections |
| `demo_data/` | five images of one subject, one per exposure level |
| `models/` | empty; the downloaded weights go here |

## Setup

<!-- TODO: replace with the final repository URL under github.com/dasec. -->
```bash
git clone https://github.com/dasec/TODO-repository-name.git
cd TODO-repository-name
conda env create -f requirements.yaml
conda activate skintone
```

On a machine without an NVIDIA GPU, replace `onnxruntime-gpu` with `onnxruntime`
in `requirements.yaml`.

<!-- TODO: replace with the h-da cloud link. -->
Download the weights from the [h-da cloud](TODO) into `models/`.

## Usage

One model:

```bash
python src/inference.py --model models/skintone_vits16_fold3.pth \
    --input demo_data --output predictions.csv
```

```
image,L*,a*,b*,cst_bin,face_detected,status
EL1.jpg,54.803,9.7405,14.6874,3,True,ok
EL2.jpg,54.5665,9.8054,15.0138,3,True,ok
EL3.jpg,54.9244,9.7615,14.6393,3,True,ok
EL4.jpg,55.054,9.8452,14.3314,3,True,ok
EL5.jpg,54.1915,9.8271,14.8138,4,True,ok
```

The demo images show subject 001 of DAST-10 at the five exposure levels, from
strongly underexposed (`EL1`) to strongly overexposed (`EL5`). This subject is
in the test set of fold 3, so the output above is a held-out prediction. The
colorimeter reading of the subject is `L*` = 54.5 (Fig. 1 of the paper). Values
can differ by about 0.01 between machines.

All five models, with their predictions averaged:

```bash
python src/inference.py --model "models/skintone_vits16_fold*.pth" \
    --input demo_data --output predictions.csv
```

The other four models were trained on subject 001, so this second output is not
a held-out prediction for the demo images.

Images are aligned with InsightFace `buffalo_sc` inside the script. `cst_bin` is
the nearest CST centroid in full CIELAB by ΔE76, numbered from 1. If no face is
detected, the whole image is used and the row is marked `face_detected=False`;
unreadable files get a row without a prediction. Both are listed at the end of
the run.

## Model Zoo

Five ViT-S/16 models (`vit_small_patch16_224.augreg_in21k_ft_in1k`, 224 × 224
input, about 87 MB each), one per cross-validation fold.

| File | Held-out fold | MD5 |
|---|---|---|
| `skintone_vits16_fold0.pth` | 0 | `4b45955792843ec207641e349ee6a864` |
| `skintone_vits16_fold1.pth` | 1 | `c674feda1912e9880a7b076df26190d2` |
| `skintone_vits16_fold2.pth` | 2 | `b76ae3c171f05982ba790b87d1fa1679` |
| `skintone_vits16_fold3.pth` | 3 | `6f0e6a632bc9dd9d49cd60e74089b73b` |
| `skintone_vits16_fold4.pth` | 4 | `8ef00e41321fd944690561eeea56d1d7` |

The numbers in the paper come from these models, each scored on its held-out
fold and pooled: R² of `L*` 0.918, mean ΔE76 3.07, 55.9% exact CST bin and 98.7%
within one bin. Averaging the models is supported by the script but was not
evaluated in the paper.

## Reproducing the paper

DAST-10 is governed by a data-protection agreement and is not included; access
must be requested from the data controller. CHROMA-FIT and the external baseline
of Matias et al. are obtained from their authors. Copy
`configs/paths.example.yaml` to `configs/paths.yaml`, set the paths, and run
`python -m src.data.build_table` before the experiments. All commands run from
the repository root.

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
| 7 | Error by CST bin and by camera | `python analysis/camera_ready_error_analysis.py <work_dir>/results/cv_pooled_predictions.csv` |

Experiment 1 runs first; the others reuse its fold checkpoints and predictions.

## Disclaimer

The models were evaluated by five-fold subject-wise cross-validation on 80
DAST-10 subjects, and each fold model was trained on 48 to 50 of them. All
subjects were captured under ICAO-compliant conditions; behaviour on other kinds
of images has not been established.

Only CST bins 2 to 9 were evaluated. Bin 1 was never scored, because it has too
few subjects to be cross-validated, and bin 10 is absent from the dataset.

The estimated scale is compressed towards its centre: light skin tones tend to
be predicted darker than measured and dark skin tones lighter than measured.

The models are research artefacts and are provided without any warranty. The
code is released under the MIT licence, see `LICENSE`; the licence covers the
code only, not any dataset.

## Citation

```bibtex
@inproceedings{Saladino-SkinToneEstimation-BIOSIG-2026,
  Author    = {L. Saladino and A. D{\"o}rsch and F. Stockhardt and C. Busch},
  Booktitle = {Proc. Intl. Conf. of the Biometrics Special Interest Group ({BIOSIG})},
  Title     = {Skin Tone Estimation for Performance Differential Analysis in Biometric Systems},
  Publisher = {IEEE},
  Year      = {2026}
}
```
