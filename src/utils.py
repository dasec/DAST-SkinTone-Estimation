"""Regression metrics and CST bin assignment under the three distance rules."""

import numpy as np

CHANNEL_NAMES = ["L*", "a*", "b*"]

# CST centroids in CIELAB, from Cook et al. (2025), Fig. 4C. Published in that
# paper and reproduced here as constants; bins are 1-indexed.
CST_L = np.array([70.00, 64.44, 58.89, 53.33, 47.78, 42.22, 36.67, 31.11, 25.56, 20.00])
CST_A = np.array([6.07, 11.02, 14.44, 16.45, 17.20, 16.79, 15.34, 12.93, 9.65, 5.55])
CST_B = np.array([4.70, 9.26, 13.06, 15.94, 17.74, 18.35, 17.66, 15.62, 12.16, 7.26])


def assign_bin_L(l):
    """Nearest CST centroid on L* alone."""
    return int(np.argmin(np.abs(CST_L - l))) + 1


def assign_bin_Lb(l, b):
    """Nearest CST centroid in the (L*, b*) plane."""
    return int(np.argmin(np.sqrt((CST_L - l) ** 2 + (CST_B - b) ** 2))) + 1


def assign_bin_Lab(l, a, b):
    """Nearest CST centroid in full CIELAB, the rule that defines the ground-truth bin."""
    dist = np.sqrt((CST_L - l) ** 2 + (CST_A - a) ** 2 + (CST_B - b) ** 2)
    return int(np.argmin(dist)) + 1


def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    """Per-channel and averaged MAE, RMSE and R2 as a flat dict."""
    results = {}
    mae_list, rmse_list, r2_list = [], [], []

    for i, ch in enumerate(CHANNEL_NAMES):
        p, t = preds[:, i], targets[:, i]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results[f"{ch}_MAE"] = round(mae, 4)
        results[f"{ch}_RMSE"] = round(rmse, 4)
        results[f"{ch}_R2"] = round(r2, 4)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

    results["avg_MAE"] = round(float(np.mean(mae_list)), 4)
    results["avg_RMSE"] = round(float(np.mean(rmse_list)), 4)
    results["avg_R2"] = round(float(np.mean(r2_list)), 4)
    return results


def print_metrics(metrics: dict, label: str = None, n: int = None, indent: str = ""):
    """Print a metrics dict as an aligned block."""
    header = indent
    if label:
        header += label
    if n is not None:
        header += f" (n={n})"
    if header.strip():
        print(header)
    pad = indent + "  "
    for ch in CHANNEL_NAMES:
        print(f"{pad}{ch}   MAE={metrics[f'{ch}_MAE']:.2f}  "
              f"RMSE={metrics[f'{ch}_RMSE']:.2f}  R2={metrics[f'{ch}_R2']:.3f}")
    print(f"{pad}avg  MAE={metrics['avg_MAE']:.2f}  "
          f"RMSE={metrics['avg_RMSE']:.2f}  R2={metrics['avg_R2']:.3f}")


def bin_accuracy(gt_bins, pred_bins):
    """Return (exact_match, within_one_bin) as fractions."""
    gt = np.asarray(gt_bins)
    pred = np.asarray(pred_bins)
    return float((gt == pred).mean()), float((np.abs(gt - pred) <= 1).mean())
