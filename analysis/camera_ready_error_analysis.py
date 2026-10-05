"""Error analyses on the pooled five-fold cross-validation predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CH = ["L", "a", "b"]
CH_TEX = {"L": "$L^*$", "a": "$a^*$", "b": "$b^*$"}
EL_NAME = {"EL1": "Strongly underexposed", "EL2": "Underexposed", "EL3": "Normal",
           "EL4": "Overexposed", "EL5": "Strongly overexposed"}
CAMERAS = ["Canon", "Sony"]
N_BOOT = 10000
SEED = 0

# Colours that stay distinguishable under colour vision deficiency.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CH_STYLE = {"L": dict(color=BLUE, marker="o", ls="-"),
            "a": dict(color=ORANGE, marker="s", ls="--"),
            "b": dict(color=AQUA, marker="^", ls=":")}


def load(path):
    d = pd.read_csv(path, dtype={"subject_key": str})
    # Signed error is prediction minus ground truth.
    for c in CH:
        d[f"err_{c}"] = d[f"{c}_pred"] - d[f"{c}_gt"]
        d[f"abs_{c}"] = d[f"err_{c}"].abs()
    d["dE"] = np.sqrt(sum(d[f"err_{c}"] ** 2 for c in CH))
    return d


METRICS = [f"abs_{c}" for c in CH] + ["dE"] + [f"err_{c}" for c in CH]


def summarise(d, by):
    g = d.groupby(by, sort=True)
    s = g[METRICS].mean()
    s.insert(0, "n_subj", g.subject_key.nunique())
    s.insert(0, "n_img", g.size())
    return s


def subject_level(d, extra=()):
    # The images of a subject share one label, so tests use one value per subject.
    return d.groupby(["subject_key", "mst_gt", *extra])[METRICS].mean().reset_index()


def boot_ci(subj_vals, rng):
    """95% percentile CI of the mean, resampling subjects."""
    v = np.asarray(subj_vals)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def fmt(x, signed=False):
    if round(x, 2) == 0:
        return "0.00"
    s =f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace("-", "$-$") if signed else s


def row(label, r, extra_label=None):
    cells = ([extra_label] if extra_label is not None else []) + [label,
             str(int(r.n_img)), str(int(r.n_subj))]
    cells += [fmt(r[f"abs_{c}"]) for c in CH] + [fmt(r.dE)]
    cells += [fmt(r[f"err_{c}"], signed=True) for c in CH]
    return " & ".join(cells) + r" \\"


def header(first_cols):
    n_first = len(first_cols)
    lead = " & ".join(first_cols)
    top = (" & " * (n_first + 1)) + r"\multicolumn{4}{c}{Absolute error} & "
    top += r"\multicolumn{3}{c}{Signed error} \\"
    a0 = n_first + 3
    cm = rf"\cmidrule(lr){{{a0}-{a0 + 3}}} \cmidrule(lr){{{a0 + 4}-{a0 + 6}}}"
    sub = (lead + r" & $n_\mathrm{img}$ & $n_\mathrm{subj}$ & "
           + " & ".join(CH_TEX[c] for c in CH) + r" & $\Delta E_{76}$ & "
           + " & ".join(CH_TEX[c] for c in CH) + r" \\")
    return [top, cm, sub]


def tabular(colspec, head, body):
    return "\n".join([
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + colspec + "}",
        r"\toprule", *head, r"\midrule", *body, r"\bottomrule", r"\end{tabular*}"])


def table_bins(by_bin, overall, stats_line):
    body = [row(str(b), r) for b, r in by_bin.iterrows()]
    body += [r"\midrule", row("All", overall)]
    tab = tabular("l" + "r" * 9, header(["CST bin"]), body)
    cap = (r"\caption{Error of the ViT-S/16 regressor by ground-truth CST bin, pooled "
           r"over the test folds of subject-wise 5-fold cross-validation on DAST-10. "
           r"Absolute error is the mean absolute error per channel and the mean "
           r"$\Delta E_{76}$; signed error is the mean of prediction minus ground "
           r"truth, so a positive value means the channel is overestimated. Bin~1 "
           r"(2 subjects) was used for training only and is not scored, and no "
           r"DAST-10 subject falls in bin~10. " + stats_line + "}")
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
                      r"\setlength{\tabcolsep}{4pt}", tab, cap,
                      r"\label{tab:error_by_bin}", r"\end{table*}", ""])


def table_camera(by_cam, by_cam_el, by_cam_bin, stats_line):
    body = [r"\multicolumn{11}{l}{\emph{By camera}} \\"]
    body += [row("All", r, cam) for cam, r in by_cam.iterrows()]
    body += [r"\midrule", r"\multicolumn{11}{l}{\emph{Camera $\times$ exposure level}} \\"]
    for cam in CAMERAS:
        body += [row(el, by_cam_el.loc[(cam, el)], cam)
                 for el in by_cam_el.loc[cam].index]
        body += [r"\addlinespace[2pt]"] if cam != CAMERAS[-1] else []
    body += [r"\midrule", r"\multicolumn{11}{l}{\emph{Camera $\times$ CST bin}} \\"]
    for cam in CAMERAS:
        body += [row(f"Bin {b}", by_cam_bin.loc[(cam, b)], cam)
                 for b in by_cam_bin.loc[cam].index]
        body += [r"\addlinespace[2pt]"] if cam != CAMERAS[-1] else []
    tab = tabular("ll" + "r" * 9, header(["Camera", "Stratum"]), body)
    cap = (r"\caption{Error of the ViT-S/16 regressor by acquisition camera, by camera "
           r"and exposure level (EL1 strongly underexposed to EL5 strongly "
           r"overexposed) and by camera and ground-truth CST bin, pooled over the "
           r"test folds of subject-wise 5-fold cross-validation on DAST-10. Every "
           r"subject is photographed with both cameras, so the two cameras share the "
           r"same subjects and ground truth. Metrics as in "
           r"Table~\ref{tab:error_by_bin}. " + stats_line + "}")
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
                      r"\setlength{\tabcolsep}{4pt}", tab, cap,
                      r"\label{tab:error_by_camera}", r"\end{table*}", ""])


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_bins(d, subj, path, rng):
    bins = sorted(d.mst_gt.unique())
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(5.0, 1.9), constrained_layout=True)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    off = {"L": -0.18, "a": 0.0, "b": 0.18}
    for c in CH:
        means, los, his = [], [], []
        for b in bins:
            m = d.loc[d.mst_gt == b, f"err_{c}"].mean()
            lo, hi = boot_ci(subj.loc[subj.mst_gt == b, f"err_{c}"].values, rng)
            means.append(m); los.append(m - lo); his.append(hi - m)
        st = CH_STYLE[c]
        ax.errorbar(x + off[c], means, yerr=[los, his], color=st["color"],
                    marker=st["marker"], ls=st["ls"], lw=1.5, ms=5, mec="white",
                    mew=0.8, elinewidth=1.0, capsize=0, label=CH_TEX[c], zorder=3)
    ax.set_xticks(x, [str(b) for b in bins])
    ax.set_xlabel("Ground-truth CST bin", color=INK)
    ax.set_ylabel("Mean signed error\n(prediction minus truth)", color=INK)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=2.2,
              labelcolor=INK)
    style_axes(ax)
    fig.savefig(path)
    plt.close(fig)


def figure_env(fname, caption, label):
    return "\n".join([r"\begin{figure}[t]", r"\centering",
                      rf"\includegraphics{{{fname}}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{figure}", ""])


def p_str(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def p_tex(p):
    if p >= 0.001:
        return f"$p={p:.3f}$"
    m, e = f"{p:.1e}".split("e")
    return rf"$p={m}\times10^{{{int(e)}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", type=Path,
                        help="cv_pooled_predictions.csv written by src.eval.collect_cv")
    parser.add_argument("--out", type=Path, default=Path("runs/camera_ready"),
                        help="output folder (default: runs/camera_ready)")
    parser.add_argument("--exclude-image", action="append", default=[],
                        metavar="FILENAME",
                        help="repeat the camera test without this image, as a "
                             "sensitivity check; may be given several times")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = []
    emit = lambda s="": (print(s), log.append(s))

    d = load(args.predictions)
    unknown = sorted(set(args.exclude_image) - set(d.image_filename))
    if unknown:
        parser.error(f"--exclude-image not in the predictions file: {unknown}")
    n_subj = d.subject_key.nunique()
    n_bins = d.mst_gt.nunique()
    max_diff = (d.dE - d.dE76).abs().max()
    assert max_diff < 1e-3, max_diff
    emit(f"input: {args.predictions.name}  images={len(d)}  subjects={n_subj}")
    emit(f"dE76 recomputed from Lab; max |recomputed - stored| = {max_diff:.2e} "
         f"(stored rounded to 4 dp)")
    emit("signed error = prediction - ground truth")

    overall = summarise(d.assign(_all="All"), "_all").iloc[0]
    by_bin = summarise(d, "mst_gt")
    subj = subject_level(d)

    kw = stats.kruskal(*[g.dE.values for _, g in subj.groupby("mst_gt")])
    sp = stats.spearmanr(subj.mst_gt, subj.dE)
    emit("\n=== (a) ERROR BY GROUND-TRUTH CST BIN (image-level means) ===")
    emit(by_bin.round(3).to_string())
    emit("overall: " + overall.round(3).to_string().replace("\n", "  "))
    absent = sorted(set(range(1, 11)) - set(d.mst_gt.unique()))
    emit(f"CST bins without scored images: {absent}")
    emit(f"\nsubject-level tests (one value per subject = mean over its images, "
         f"n={n_subj})")
    emit(f"  dE76  Kruskal-Wallis across {n_bins} bins: H={kw.statistic:.2f}, "
         f"df={n_bins - 1}, p={p_str(kw.pvalue)}")
    emit(f"  dE76  Spearman(bin, subject mean dE76): rho={sp.statistic:+.3f}, "
         f"p={p_str(sp.pvalue)}")
    emit("  supplementary, signed error per channel (not multiplicity-corrected):")
    for c in CH:
        k = stats.kruskal(*[g[f"err_{c}"].values for _, g in subj.groupby("mst_gt")])
        s = stats.spearmanr(subj.mst_gt, subj[f"err_{c}"])
        emit(f"    signed {c}*: KW H={k.statistic:6.2f} p={p_str(k.pvalue):>8}   "
             f"Spearman rho={s.statistic:+.3f} p={p_str(s.pvalue)}")

    stats_a = (rf"Across subjects (one mean $\Delta E_{{76}}$ per subject, $n={n_subj}$), a "
               rf"Kruskal--Wallis test across bins gives $H={kw.statistic:.2f}$, "
               rf"{p_tex(kw.pvalue)}, and the Spearman correlation between bin and "
               rf"mean $\Delta E_{{76}}$ is $\rho={sp.statistic:.2f}$, "
               rf"{p_tex(sp.pvalue)}.")
    (out_dir / "tab_error_by_bin.tex").write_text(table_bins(by_bin, overall, stats_a))

    fig_bins(d, subj, out_dir / "fig_error_by_bin.pdf", np.random.default_rng(SEED))
    cap_a = (r"Mean signed error per channel (prediction minus ground truth) of the "
             r"ViT-S/16 regressor by ground-truth CST bin, pooled over the test folds "
             r"of subject-wise 5-fold cross-validation on DAST-10. Error bars are 95\% "
             r"confidence intervals from a bootstrap over subjects. Bin~1 is not "
             r"scored and bin~10 is empty.")
    (out_dir / "fig_error_by_bin.tex").write_text(
        figure_env("fig_error_by_bin.pdf", cap_a, "fig:error_by_bin"))

    emit("\n=== (b) ERROR BY ACQUISITION CAMERA ===")
    emit("camera source: `camera` column of the predictions file")
    by_cam = summarise(d, "camera")
    by_cam_el = summarise(d, ["camera", "EL"])
    by_cam_bin = summarise(d, ["camera", "mst_gt"])
    emit(by_cam.round(3).to_string())
    emit("\ncamera x exposure level")
    emit(by_cam_el.round(3).to_string())
    emit("\ncamera x CST bin")
    emit(by_cam_bin.round(3).to_string())

    subj_cam = subject_level(d, extra=("camera",))
    w = subj_cam.pivot(index="subject_key", columns="camera", values=METRICS)
    diff = (w.xs("Canon", axis=1, level=1) - w.xs("Sony", axis=1, level=1))
    diff = diff.join(subj.set_index("subject_key").mst_gt)
    wx = stats.wilcoxon(w[("dE", "Canon")], w[("dE", "Sony")])
    lo, hi = boot_ci(diff.dE.values, rng)
    emit(f"\nsubject-level paired tests (per subject: mean over its images per "
         f"camera, n={n_subj} pairs; difference = Canon - Sony)")
    emit(f"  dE76  mean diff {diff.dE.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), "
         f"median diff {diff.dE.median():+.3f}, Canon worse for "
         f"{(diff.dE > 0).sum()}/{n_subj} subjects")
    emit(f"  dE76  Wilcoxon signed-rank: W={wx.statistic:.0f}, p={p_str(wx.pvalue)}")
    emit("  supplementary, per channel (not multiplicity-corrected):")
    for m in [f"abs_{c}" for c in CH] + [f"err_{c}" for c in CH]:
        t = stats.wilcoxon(w[(m, "Canon")], w[(m, "Sony")])
        emit(f"    {m:6s} mean diff {diff[m].mean():+.3f}  median {diff[m].median():+.3f}"
             f"  Wilcoxon p={p_str(t.pvalue)}")

    if args.exclude_image:
        d2 = d[~d.image_filename.isin(args.exclude_image)]
        w2 = subject_level(d2, extra=("camera",)).pivot(index="subject_key",
                                                        columns="camera", values="dE")
        wx2 = stats.wilcoxon(w2.Canon, w2.Sony)
        emit(f"  sensitivity without {len(d) - len(d2)} excluded image(s): "
             f"mean diff {(w2.Canon - w2.Sony).mean():+.3f}, "
             f"Wilcoxon p={p_str(wx2.pvalue)}")

    emit("\ncamera x exposure level, paired per EL (Holm over 5 tests)")
    subj_cel = subject_level(d, extra=("camera", "EL"))
    ps, lines_el = [], []
    for el in sorted(d.EL.unique()):
        q = subj_cel[subj_cel.EL == el].pivot(index="subject_key", columns="camera",
                                              values="dE")
        t = stats.wilcoxon(q.Canon, q.Sony)
        ps.append(t.pvalue)
        lines_el.append((el, (q.Canon - q.Sony).mean(), t.pvalue))
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    for (el, md, p), pa in zip(lines_el, adj):
        emit(f"  {el}: mean diff {md:+.3f}  Wilcoxon p={p_str(p)}  Holm p={p_str(pa)}")

    emit("\ncamera x CST bin interaction (subject-level Canon - Sony difference vs bin)")
    kwi = stats.kruskal(*[g.dE.values for _, g in diff.groupby("mst_gt")])
    spi = stats.spearmanr(diff.mst_gt, diff.dE)
    emit(f"  dE76 diff  Kruskal-Wallis across {n_bins} bins: H={kwi.statistic:.2f}, "
         f"df={n_bins - 1}, p={p_str(kwi.pvalue)}")
    emit(f"  dE76 diff  Spearman(bin, diff): rho={spi.statistic:+.3f}, "
         f"p={p_str(spi.pvalue)}")
    emit("  supplementary, signed error difference per channel:")
    for c in CH:
        k = stats.kruskal(*[g[f"err_{c}"].values for _, g in diff.groupby("mst_gt")])
        s = stats.spearmanr(diff.mst_gt, diff[f"err_{c}"])
        emit(f"    signed {c}* diff: KW H={k.statistic:6.2f} p={p_str(k.pvalue):>8}   "
             f"Spearman rho={s.statistic:+.3f} p={p_str(s.pvalue)}")
    emit("  per camera, bin effect on subject mean dE76:")
    for cam in CAMERAS:
        sc = subj_cam[subj_cam.camera == cam]
        k = stats.kruskal(*[g.dE.values for _, g in sc.groupby("mst_gt")])
        s = stats.spearmanr(sc.mst_gt, sc.dE)
        emit(f"    {cam:5s}: KW H={k.statistic:.2f} p={p_str(k.pvalue)}   "
             f"Spearman rho={s.statistic:+.3f} p={p_str(s.pvalue)}")

    stats_b = (rf"Paired over subjects ($n={n_subj}$, one mean $\Delta E_{{76}}$ per subject "
               rf"and camera), the Canon minus Sony difference is "
               rf"{fmt(diff.dE.mean(), signed=True)} (Wilcoxon signed-rank "
               rf"{p_tex(wx.pvalue)}); its dependence on CST bin gives "
               rf"Kruskal--Wallis $H={kwi.statistic:.2f}$, {p_tex(kwi.pvalue)}, and "
               rf"Spearman $\rho={spi.statistic:.2f}$, {p_tex(spi.pvalue)}.")
    (out_dir / "tab_error_by_camera.tex").write_text(
        table_camera(by_cam, by_cam_el, by_cam_bin, stats_b))

    pd.concat({"bin": by_bin.reset_index().astype({"mst_gt": str}).rename(
                   columns={"mst_gt": "stratum"}),
               "camera": by_cam.reset_index().rename(columns={"camera": "stratum"}),
               "camera_x_el": by_cam_el.reset_index().assign(
                   stratum=lambda x: x.camera + "_" + x.EL).drop(columns=["camera", "EL"]),
               "camera_x_bin": by_cam_bin.reset_index().assign(
                   stratum=lambda x: x.camera + "_bin" + x.mst_gt.astype(str)).drop(
                   columns=["camera", "mst_gt"])},
              names=["analysis"]).droplevel(1).reset_index().round(4).to_csv(
        out_dir / "error_strata.csv", index=False)

    (out_dir / "ERROR_ANALYSIS.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
