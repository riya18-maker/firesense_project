from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def class_ids(cfg):
    """Return {'fire': id, 'smoke': id} from the config's class map."""
    return {str(v).lower(): int(k) for k, v in cfg["classes"].items()}


def clone(cfg):
    return copy.deepcopy(cfg)
