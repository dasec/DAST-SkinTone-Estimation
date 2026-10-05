"""Filesystem locations and hyperparameters, read from configs/."""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

DEFAULT_PATHS = CONFIG_DIR / "paths.yaml"
DEFAULT_PARAMS = CONFIG_DIR / "default.yaml"
DEFAULT_OVERRIDES = CONFIG_DIR / "exposure_overrides.yaml"


class Paths:
    """Resolved locations; derived outputs all live under work_dir."""

    def __init__(self, cfg):
        self.dataset_root = Path(cfg["dataset_root"]).expanduser()
        self.metadata_csv = self.dataset_root / cfg["metadata_csv"]
        self.chroma_table = Path(cfg["chroma_table"]).expanduser()
        self.matias_checkpoint = Path(cfg["matias_checkpoint"]).expanduser()
        self.matias_prototype_dir = Path(cfg["matias_prototype_dir"]).expanduser()
        self.work_dir = Path(cfg["work_dir"]).expanduser()

    @property
    def table_csv(self):
        return self.work_dir / "dast10_table.csv"

    @property
    def merged_csv(self):
        return self.work_dir / "merged_chroma_table.csv"

    @property
    def results(self):
        return self.work_dir / "results"

    @property
    def checkpoints(self):
        return self.work_dir / "checkpoints"

    @property
    def stone_work(self):
        return self.work_dir / "stone_work"


def load_paths(path=None):
    p = Path(path) if path else DEFAULT_PATHS
    if not p.is_file():
        sys.exit(
            f"missing {p}\n"
            f"copy {CONFIG_DIR / 'paths.example.yaml'} to {DEFAULT_PATHS} "
            "and set the local paths."
        )
    return Paths(yaml.safe_load(p.read_text()))


def load_params(path=None, **overrides):
    """Read configs/default.yaml (or `path`); non-None overrides win."""
    p = Path(path) if path else DEFAULT_PARAMS
    if not p.is_file():
        sys.exit(f"missing {p}")
    params = yaml.safe_load(p.read_text())
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def load_exposure_overrides(path=None):
    p = Path(path) if path else DEFAULT_OVERRIDES
    data = yaml.safe_load(p.read_text()) if p.is_file() else None
    overrides = (data or {}).get("overrides") or {}
    if not overrides:
        print(
            "WARNING: no exposure overrides loaded "
            f"({'file missing: ' + str(p) if not p.is_file() else 'file is empty'}).\n"
            "         Two images in the release carry a label that contradicts their\n"
            "         own luminance. Without the corrections the per-exposure counts\n"
            "         and every per-exposure result will differ from the published\n"
            "         table. See configs/exposure_overrides.yaml.",
            file=sys.stderr,
        )
    return {k: tuple(v) for k, v in overrides.items()}


def add_config_args(parser):
    parser.add_argument("--paths", default=None, help="path to a paths.yaml")
    parser.add_argument("--params", default=None, help="path to a params yaml")
    return parser
