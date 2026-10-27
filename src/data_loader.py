"""Reading CSV / Excel / JSON files safely (parsing only, nothing is executed)."""
import io
from pathlib import Path

import pandas as pd

from .config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, PRODUCTS_SAMPLE, REVIEWS_SAMPLE, get_logger

log = get_logger(__name__)


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame based on the file extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DataLoadError(f"File is larger than {MAX_UPLOAD_MB} MB. Create a smaller sample "
                            "with scripts/prepare_dataset.py first.")
    if not content:
        raise DataLoadError("The file is empty.")
    try:
        if ext == "csv":
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc, low_memory=False,
                                     on_bad_lines="skip")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise DataLoadError("Could not decode the CSV file.")
        elif ext == "xlsx":
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:  # json / jsonl (UCSD files are JSON-lines)
            try:
                df = pd.read_json(io.BytesIO(content), lines=True)
            except ValueError:
                df = pd.read_json(io.BytesIO(content))
    except DataLoadError:
        raise
    except Exception as exc:
        log.warning("Failed to parse %s: %s", filename, exc)
        raise DataLoadError("The file could not be read. Check that it is a valid, "
                            f"well-formed {ext.upper()} file.") from exc
    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The file contains no data rows.")
    return df


def read_path(path) -> pd.DataFrame:
    return read_table(Path(path).read_bytes(), Path(path).name)


def load_default_dataset():
    """Real sample from data/ when present, otherwise a clearly-labelled synthetic demo."""
    from .demo_data import generate_demo
    from .preprocessing import build_dataset
    from .validation import auto_map_products, auto_map_reviews
    if PRODUCTS_SAMPLE.exists():
        try:
            prod = read_path(PRODUCTS_SAMPLE)
            rev = read_path(REVIEWS_SAMPLE) if REVIEWS_SAMPLE.exists() else None
            return build_dataset(prod, auto_map_products(prod.columns), rev,
                                 auto_map_reviews(rev.columns) if rev is not None else None,
                                 source_label=f"Repository sample ({PRODUCTS_SAMPLE.name})")
        except Exception as exc:  # fall back to demo, but log the reason
            log.error("Could not load repository sample: %s", exc)
    prod, rev = generate_demo()
    return build_dataset(prod, auto_map_products(prod.columns), rev, auto_map_reviews(rev.columns),
                         source_label="SYNTHETIC DEMO DATA (not real Amazon data)", is_demo=True)
