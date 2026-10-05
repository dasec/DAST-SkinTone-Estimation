"""Cross-check exposure labels against measured image luminance."""

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image

from src.config import add_config_args, load_paths

Image.MAX_IMAGE_PIXELS = None

# Six images per (subject, camera) at fixed EV offsets: sorted by luminance,
# their labels must read as follows.
EXPECTED_SORTED = ["E1", "E2", "E3", "E3", "E4", "E5"]

_DATA_ROOT = None


def _init(root):
    global _DATA_ROOT
    _DATA_ROOT = root


def mean_luma(relpath):
    a = np.asarray(Image.open(_DATA_ROOT / relpath).convert("RGB"), dtype=np.float32)
    return float((0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).mean())


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    out_csv = paths.work_dir / "luminance_check.csv"

    d = pd.read_csv(paths.metadata_csv)
    d["sid"] = d.subject_id.astype(str).str.zfill(3)
    d["E"] = d.image_filename.str.extract(r"-(E\d)-")[0]

    with mp.Pool(args.jobs, initializer=_init,
                 initargs=(paths.dataset_root.parent,)) as pool:
        d["luma"] = pool.map(mean_luma, d.image_relpath.tolist())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}\n")

    print("mean luminance by labelled exposure level")
    print(f"  {'E':>3} {'n':>4} {'mean':>8} {'std':>7} {'min':>7} {'max':>7}")
    for e, g in d.groupby("E"):
        print(f"  {e:>3} {len(g):>4} {g.luma.mean():>8.1f} {g.luma.std():>7.1f} "
              f"{g.luma.min():>7.1f} {g.luma.max():>7.1f}")

    print("\ngroups whose labels disagree with their own luminance ordering")
    bad_groups = []
    for (sid, cam), g in d.groupby(["sid", "camera"]):
        if len(g) != 6:
            print(f"  a group has {len(g)} images, not 6, skipped")
            continue
        order = g.sort_values("luma")
        if list(order.E) != EXPECTED_SORTED:
            bad_groups.append((sid, cam, order))

    if not bad_groups:
        print("  none: every group is consistent")
    for sid, cam, order in bad_groups:
        print(f"\n  group on camera {cam}")
        print(f"    {'labelled':>9} {'luma':>8}  {'implied':>8}")
        for (_, row), implied in zip(order.iterrows(), EXPECTED_SORTED):
            flag = "" if row.E == implied else "   <-- mismatch"
            print(f"    {row.E:>9} {row.luma:>8.1f}  {implied:>8}{flag}")

    print(f"\n{len(bad_groups)} inconsistent group(s) out of "
          f"{d.groupby(['sid','camera']).ngroups}")


if __name__ == "__main__":
    main()
