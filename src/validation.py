"""Column mapping, validation and data inspection helpers."""
from dataclasses import dataclass, field

import pandas as pd

from .config import PRODUCT_FIELDS, REVIEW_FIELDS, normalise_name


def auto_map_columns(columns, synonyms: dict) -> dict:
    """Return {canonical_field: source_column_or_None} using the synonym lists."""
    lookup = {}
    for col in columns:
        lookup.setdefault(normalise_name(col), col)
    return {f: next((lookup[s] for s in names if s in lookup), None) for f, names in synonyms.items()}


def auto_map_products(columns) -> dict:
    return auto_map_columns(columns, PRODUCT_FIELDS)


def auto_map_reviews(columns) -> dict:
    return auto_map_columns(columns, REVIEW_FIELDS)


def validate_product_mapping(mapping: dict) -> tuple[list, list]:
    """Return (errors, warnings) for a product column mapping."""
    errors, warnings = [], []
    if not (mapping.get("title") or mapping.get("asin")):
        errors.append("Map at least a product title or a product ID (ASIN).")
    for f, msg in [("price", "price charts"), ("rating", "rating charts"),
                   ("category", "category charts"), ("brand", "brand comparison"),
                   ("review_count", "review-count charts")]:
        if not mapping.get(f):
            warnings.append(f"No '{f}' column mapped: {msg} will be unavailable.")
    return errors, warnings


def validate_review_mapping(mapping: dict) -> tuple[list, list]:
    errors, warnings = [], []
    if not mapping.get("review_text"):
        errors.append("Map the review text column (needed for sentiment analysis).")
    if not mapping.get("asin"):
        warnings.append("No product ID mapped: reviews cannot be linked to products.")
    if not mapping.get("review_date"):
        warnings.append("No review date mapped: time trends will be unavailable.")
    return errors, warnings


@dataclass
class QualityReport:
    kind: str
    rows_loaded: int = 0
    rows_retained: int = 0
    removed: dict = field(default_factory=dict)
    flagged: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def add_removed(self, label, n):
        if n:
            self.removed[label] = int(n)

    def add_flagged(self, label, n):
        if n:
            self.flagged[label] = int(n)

    def to_text(self) -> str:
        lines = [f"Data-quality report: {self.kind}", "=" * 40,
                 f"Rows loaded:   {self.rows_loaded}", f"Rows retained: {self.rows_retained}"]
        for title, items in (("Removed", self.removed), ("Flagged / corrected", self.flagged)):
            lines.append(f"\n{title}:")
            lines += [f"  - {k}: {v}" for k, v in items.items()] or ["  (none)"]
        for title, items in (("Warnings", self.warnings), ("Errors", self.errors)):
            if items:
                lines.append(f"\n{title}:")
                lines += [f"  - {x}" for x in items]
        return "\n".join(lines)


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """Row/column counts, dtypes, missing values and exact-duplicate rows."""
    try:
        dup = int(df.astype(str).duplicated().sum())
    except Exception:  # pragma: no cover - defensive
        dup = 0
    return {
        "rows": len(df), "columns": df.shape[1],
        "dtypes": pd.DataFrame({"column": df.columns.astype(str), "dtype": df.dtypes.astype(str).values}),
        "missing": pd.DataFrame({"column": df.columns.astype(str), "missing": df.isna().sum().values,
                                 "missing_%": (df.isna().mean().values * 100).round(1)}),
        "duplicate_rows": dup,
    }
