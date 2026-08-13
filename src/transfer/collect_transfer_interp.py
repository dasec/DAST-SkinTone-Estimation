"""Report cross-dataset transfer and occlusion sensitivity under cross-validation.

Each block names the file it was read from. Missing inputs are reported rather
than skipped silently.

    python -m src.transfer.collect_transfer_interp
"""

import argparse

import numpy as np
import pandas as pd

from src.config import add_config_args, load_paths, load_params

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def rule(c="=", n=74):
    emit(c * n)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    P = load_params(args.params)
    res = paths.results
    out = res / "CV_TRANSFER_INTERP.txt"

    rule()
    emit("CROSS-DATASET TRANSFER AND OCCLUSION UNDER 5-FOLD CROSS-VALIDATION")
    rule()
    emit("Every figure below uses the fold models from the main cross-validation")
    emit("run. No new split was created and no model was retrained.")
    emit()

    rule()
    emit("1. ZERO-SHOT CHROMA-FIT  (source: results/cv_zeroshot_chroma.csv)")
    rule()
    zs_path = res / "cv_zeroshot_chroma.csv"
    if zs_path.is_file():
        zs = pd.read_csv(zs_path)
        emit(f"  every fold scored the same {int(zs.n.iloc[0])} images")
        emit(f"  filter: L*>={P['chroma_min_L']}, UQS>={P['chroma_uqs_min']}, "
             "background-uniformity != -1, applied once")
        emit()
        emit(f"  {'fold':>4} {'n':>5} {'L* R2':>8} {'a* R2':>9} {'b* R2':>8} "
             f"{'dE76':>7} {'delta a*':>9}")
        for r in zs.itertuples(index=False):
            emit(f"  {int(r.fold):>4} {int(r.n):>5} {r.L_R2:>8.3f} {r.a_R2:>9.3f} "
                 f"{r.b_R2:>8.3f} {r.dE76:>7.2f} {r.delta_a:>+9.2f}")
        emit("  " + "-" * 56)
        for stat, fn in (("mean", np.mean), ("std", np.std),
                         ("min", np.min), ("max", np.max)):
            emit(f"  {stat:>4} {'':>5} {fn(zs.L_R2):>8.3f} {fn(zs.a_R2):>9.3f} "
                 f"{fn(zs.b_R2):>8.3f} {fn(zs.dE76):>7.2f} {fn(zs.delta_a):>+9.2f}")
        emit()
        emit(f"  estimated device offset, mean over folds: "
             f"delta L*={zs.delta_L.mean():+.2f}  delta a*={zs.delta_a.mean():+.2f}  "
             f"delta b*={zs.delta_b.mean():+.2f}")
    else:
        emit("  MISSING cv_zeroshot_chroma.csv")

    emit()
    rule()
    emit("2. MEAN-SHIFT CORRECTION AND COMBINED TRAINING")
    emit("   (source: results/cv_cross/fold*.csv)")
    rule()
    emit("  The primary dataset keeps the folds of the main cross-validation, so")
    emit("  its rows here are on the same subjects as the main experiments. The")
    emit("  partition is not regenerated.")
    emit()
    files = sorted((res / "cv_cross").glob("fold[0-9].csv"))
    if files:
        cc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        sh = cc.dropna(subset=["shift_a"])
        if len(sh):
            emit("  mean-shift per fold (estimated on the training split only)")
            emit(f"  {'fold':>4} {'shift L*':>9} {'shift a*':>9} {'shift b*':>9}")
            for f, g in sh.groupby("fold"):
                r = g.iloc[0]
                emit(f"  {int(f):>4} {r.shift_L:>+9.2f} {r.shift_a:>+9.2f} "
                     f"{r.shift_b:>+9.2f}")
            emit(f"  {'mean':>4} {sh.shift_L.mean():>+9.2f} "
                 f"{sh.shift_a.mean():>+9.2f} {sh.shift_b.mean():>+9.2f}")
        for subset in ("combined", "dast", "chroma"):
            s = cc[cc.subset == subset]
            if s.empty:
                continue
            emit()
            emit(f"  [{subset}] per fold")
            emit(f"  {'fold':>4} {'n':>5} {'L* R2':>8} {'a* R2':>9} {'b* R2':>8} "
                 f"{'dE76':>7}")
            for r in s.itertuples(index=False):
                emit(f"  {int(r.fold):>4} {int(r.n):>5} {r.L_R2:>8.3f} "
                     f"{r.a_R2:>9.3f} {r.b_R2:>8.3f} {r.dE76:>7.2f}")
            emit(f"  {'mean':>4} {'':>5} {s.L_R2.mean():>8.3f} {s.a_R2.mean():>9.3f} "
                 f"{s.b_R2.mean():>8.3f} {s.dE76.mean():>7.2f}")
            emit(f"  {'std':>4} {'':>5} {s.L_R2.std():>8.3f} {s.a_R2.std():>9.3f} "
                 f"{s.b_R2.std():>8.3f} {s.dE76.std():>7.2f}")
    else:
        emit("  MISSING results/cv_cross/fold*.csv")

    ns = sorted((res / "cv_cross").glob("fold[0-9]_noshift.csv"))
    if ns:
        cn = pd.concat([pd.read_csv(f) for f in ns], ignore_index=True)
        emit()
        emit("  control: same folds, mean-shift disabled")
        for subset in ("dast", "chroma"):
            s = cn[cn.subset == subset]
            if not s.empty:
                emit(f"    [{subset}] L* R2 mean {s.L_R2.mean():.3f}  "
                     f"dE76 {s.dE76.mean():.2f}")

    emit()
    rule()
    emit("3. OCCLUSION SENSITIVITY  (source: results/cv_region_sensitivity.csv)")
    rule()
    rs_path = res / "cv_region_sensitivity.csv"
    if rs_path.is_file():
        s = pd.read_csv(rs_path, dtype={"subject_key": str})
        bg = ["bg_TL", "bg_TR", "bg_BL", "bg_BR"]
        s["background"] = s[bg].mean(axis=1)
        cheeks = s[["cheek_L", "cheek_R"]].mean(axis=1)
        emit(f"  pooled over {s.fold.nunique()} folds: {len(s)} images / "
             f"{s.subject_key.nunique()} subjects")
        emit("  each subject measured once, by the fold model that never saw it")
        emit()
        emit(f"  {'region':<30} {'mean |dL*|':>11} {'std':>8}")
        for name, series in (("face oval", s.face_oval),
                             ("outside oval", s.bg_oval),
                             ("eyes", s.eyes),
                             ("mouth / chin", s.mouth),
                             ("left cheek", s.cheek_L),
                             ("right cheek", s.cheek_R),
                             ("cheeks (average)", cheeks),
                             ("forehead", s.forehead),
                             ("background corners", s.background)):
            emit(f"  {name:<30} {series.mean():>11.3f} {series.std():>8.3f}")
        ratio = s.face_oval.mean() / max(s.bg_oval.mean(), 1e-9)
        emit("  " + "-" * 51)
        emit(f"  face oval / outside oval ratio : {ratio:.2f}x")
        emit()
        emit(f"  {'fold':>4} {'n':>5} {'face':>8} {'outside':>9} {'ratio':>7}")
        for f, g in s.groupby("fold"):
            emit(f"  {int(f):>4} {len(g):>5} {g.face_oval.mean():>8.3f} "
                 f"{g.bg_oval.mean():>9.3f} "
                 f"{g.face_oval.mean()/max(g.bg_oval.mean(),1e-9):>7.2f}")
    else:
        emit("  MISSING cv_region_sensitivity.csv")

    emit()
    rule()
    emit("4. EIGENCAM  (qualitative)")
    rule()
    runs = sorted(res.glob("eigencam_cv*/"))
    if runs:
        latest = runs[-1]
        emit(f"  maps written to {latest.name}")
        imgs = sorted(latest.glob("individual/*.png"))
        emit(f"  files   : {len(imgs)}")
        note = latest / "PROVENANCE.txt"
        if note.is_file():
            for ln in note.read_text().splitlines():
                emit(f"  {ln}")
    else:
        emit("  MISSING results/eigencam_cv*/")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
