import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def progress(it, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc)
    except Exception:
        return it


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
