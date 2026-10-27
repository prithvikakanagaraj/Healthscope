"""Cleaning pipeline: products and reviews -> clean, linked Dataset + quality reports."""
import ast
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (CATEGORY_LEVEL, MAX_REVIEWS_IN_SESSION, PRODUCT_COLUMNS, REVIEW_COLUMNS,
                     UNKNOWN, get_logger)
from .validation import QualityReport

log = get_logger(__name__)


@dataclass
class Dataset:
    products: pd.DataFrame
    reviews: pd.DataFrame | None
    reports: list = field(default_factory=list)
    source_label: str = ""
    is_demo: bool = False
    sentiment_ready: bool = False  # set once VADER scores are stored in reviews["compound"]


# ---------------------------------------------------------------- field parsers
def parse_price(series: pd.Series) -> pd.Series:
    """'$1,299.99', '12.5 - 15' (first number), 10 -> floats; unparseable -> NaN."""
    text = series.astype("object").where(series.notna(), "").astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(text.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def _category_label(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [str(x) for x in value]
    elif isinstance(value, str) and value.strip():
        s = value.strip()
        items = None
        if s.startswith("["):
            try:  # literal_eval only parses literals, it never executes code
                parsed = ast.literal_eval(s)
                items = [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else None
            except (ValueError, SyntaxError):
                items = None
        if items is None:
            items = [p for p in re.split(r"\s*[>|]\s*", s) if p]
    else:
        return None
    items = [re.sub(r"\s+", " ", i).strip() for i in items if str(i).strip()]
    if not items:
        return None
    return items[CATEGORY_LEVEL] if len(items) > CATEGORY_LEVEL else items[0]


def normalise_labels(series: pd.Series) -> pd.Series:
    """Merge spelling/case variants ('NATURE MADE' vs 'Nature Made') into the most common form."""
    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: re.sub(r"\s+", " ", str(v)).strip() if v is not None else None)
    s = s.where(s != "", None)
    key = s.str.lower()
    best = s.groupby(key).agg(lambda x: x.value_counts().index[0])
    return key.map(best).fillna(UNKNOWN)


def _flatten_text(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(str(x) for x in v)
    return v


def _clean_text_col(s: pd.Series) -> pd.Series:
    s = s.map(_flatten_text).astype("object")
    s = s.where(s.notna(), None)
    s = s.map(lambda v: re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", str(v))).strip()
              if v is not None else None)
    return s.where(s.map(lambda v: bool(v) and v.lower() not in ("nan", "none", "null")), None)


def parse_dates(s: pd.Series) -> pd.Series:
    """Handles datetimes, unix seconds/milliseconds and text dates such as '09 4, 2015'."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna() & (num > 1e8)
    is_ms = is_num & (num > 1e11)
    sec = pd.to_datetime(num.where(is_num & ~is_ms), unit="s", errors="coerce")
    msec = pd.to_datetime(num.where(is_ms), unit="ms", errors="coerce")
    text = s.where(~is_num).astype("object")
    try:
        txt = pd.to_datetime(text, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        txt = pd.to_datetime(text, errors="coerce")
    out = sec.fillna(msec).fillna(txt)
    return out.dt.tz_localize(None) if getattr(out.dt, "tz", None) is not None else out


def _select(raw: pd.DataFrame, mapping: dict, canonical: list) -> pd.DataFrame:
    """Rename mapped source columns to canonical names; add missing ones as empty."""
    data = {}
    for name in canonical:
        src = mapping.get(name)
        data[name] = raw[src].reset_index(drop=True) if src and src in raw.columns else pd.Series(
            [None] * len(raw), dtype="object")
    return pd.DataFrame(data)


# ---------------------------------------------------------------- products
def clean_products(raw: pd.DataFrame, mapping: dict, remove_duplicates: bool = True):
    rep = QualityReport("products", rows_loaded=len(raw))
    if raw is None or raw.empty:
        rep.errors.append("Product table is empty.")
        return pd.DataFrame(columns=PRODUCT_COLUMNS), rep
    df = _select(raw, mapping, PRODUCT_COLUMNS)
    for c in ("asin", "title", "description", "url"):
        df[c] = _clean_text_col(df[c])
    df["asin"] = df["asin"].str.upper()
    df["url"] = df["url"].where(df["url"].fillna("").str.match(r"^https?://", case=False), None)

    df["price"] = parse_price(df["price"])
    bad_price = df["price"] <= 0
    rep.add_flagged("non-positive prices set to missing", bad_price.sum())
    df.loc[bad_price, "price"] = np.nan
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = df["rating"].notna() & ~df["rating"].between(1, 5)
    rep.add_flagged("ratings outside 1-5 set to missing", bad_rating.sum())
    df.loc[bad_rating, "rating"] = np.nan
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    bad_cnt = df["review_count"] < 0
    rep.add_flagged("negative review counts set to missing", bad_cnt.sum())
    df.loc[bad_cnt, "review_count"] = np.nan

    df["category"] = normalise_labels(df["category"].map(_category_label))
    df["brand"] = normalise_labels(df["brand"])

    no_id = df["asin"].isna() & df["title"].isna()
    rep.add_removed("rows with neither ASIN nor title", no_id.sum())
    df = df[~no_id]
    df["title"] = df["title"].fillna(df["asin"])

    if remove_duplicates:
        before = len(df)
        has_id = df["asin"].notna()
        with_id = df[has_id].sort_values("review_count", ascending=False, na_position="last") \
            .drop_duplicates("asin")
        without = df[~has_id].drop_duplicates(["title", "brand"])
        df = pd.concat([with_id, without]).sort_index()
        rep.add_removed("duplicate products (same ASIN, or same title+brand without ASIN)", before - len(df))

    for col in ("category", "brand"):
        rep.add_flagged(f"products with unknown {col}", (df[col] == UNKNOWN).sum())
    for col in ("price", "rating", "review_count"):
        rep.add_flagged(f"products with missing {col}", df[col].isna().sum())
    df = df.reset_index(drop=True)
    rep.rows_retained = len(df)
    if df.empty:
        rep.errors.append("No valid products remained after cleaning.")
    return df, rep


# ---------------------------------------------------------------- reviews
def clean_reviews(raw: pd.DataFrame, mapping: dict, remove_duplicates: bool = True):
    rep = QualityReport("reviews", rows_loaded=len(raw))
    if raw is None or raw.empty:
        rep.errors.append("Review table is empty.")
        return pd.DataFrame(columns=REVIEW_COLUMNS), rep
    df = _select(raw, mapping, REVIEW_COLUMNS)
    for c in ("asin", "review_text", "review_title", "reviewer_id"):
        df[c] = _clean_text_col(df[c])
    df["asin"] = df["asin"].str.upper()
    empty = df["review_text"].isna()
    rep.add_removed("reviews with empty / unusable text", empty.sum())
    df = df[~empty].copy()

    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    bad = df["review_rating"].notna() & ~df["review_rating"].between(1, 5)
    rep.add_flagged("review ratings outside 1-5 set to missing", bad.sum())
    df.loc[bad, "review_rating"] = np.nan

    dates = parse_dates(df["review_date"])
    implausible = dates.notna() & ((dates.dt.year < 1995) | (dates > pd.Timestamp.now() + pd.Timedelta(days=1)))
    rep.add_flagged("implausible review dates set to missing", implausible.sum())
    df["review_date"] = dates.mask(implausible)
    rep.add_flagged("reviews with missing/unparseable date", df["review_date"].isna().sum())

    if remove_duplicates:
        keys = [c for c in ["reviewer_id", "asin", "review_text", "review_date", "review_rating"]
                if df[c].notna().any()]
        before = len(df)
        # Only EXACT repeats on all key fields are removed; merely similar texts are kept.
        df = df.drop_duplicates(subset=keys)
        rep.add_removed("exact duplicate reviews (" + ", ".join(keys) + ")", before - len(df))
    df = df.reset_index(drop=True)
    rep.rows_retained = len(df)
    if df.empty:
        rep.errors.append("No valid reviews remained after cleaning.")
    return df, rep


def derive_products_from_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Minimal product table from review-level data only (rating / count are sample-based)."""
    g = reviews.dropna(subset=["asin"]).groupby("asin")
    out = pd.DataFrame({"asin": g.size().index, "review_count": g.size().values,
                        "rating": g["review_rating"].mean().values})
    out["title"] = out["asin"]
    out["category"] = UNKNOWN
    out["brand"] = UNKNOWN
    for c in ("price", "description", "url"):
        out[c] = np.nan
    return out[PRODUCT_COLUMNS]


def link_reviews(reviews: pd.DataFrame, products: pd.DataFrame, rep: QualityReport):
    """Attach product title/category/brand to each review (left join on ASIN)."""
    info = products.dropna(subset=["asin"]).drop_duplicates("asin")[["asin", "title", "category", "brand"]]
    out = reviews.merge(info.rename(columns={"title": "product_title"}), on="asin", how="left")
    orphan = out["product_title"].isna()
    rep.add_flagged("reviews whose ASIN is not in the product table", orphan.sum())
    out["category"] = out["category"].fillna(UNKNOWN)
    out["brand"] = out["brand"].fillna(UNKNOWN)
    out["product_title"] = out["product_title"].fillna(out["asin"])
    return out


def build_dataset(products_raw, product_map, reviews_raw=None, review_map=None,
                  source_label="", is_demo=False, remove_duplicates=True) -> Dataset:
    """Full pipeline used by the app, tests and scripts."""
    reports, reviews = [], None
    if products_raw is not None:
        products, prep = clean_products(products_raw, product_map, remove_duplicates)
        reports.append(prep)
    else:
        products, prep = None, None
    if reviews_raw is not None:
        reviews, rrep = clean_reviews(reviews_raw, review_map or {}, remove_duplicates)
        reports.append(rrep)
        if len(reviews) > MAX_REVIEWS_IN_SESSION:
            reviews = reviews.sample(MAX_REVIEWS_IN_SESSION, random_state=42).reset_index(drop=True)
            rrep.warnings.append(f"Review file had more than {MAX_REVIEWS_IN_SESSION:,} rows; a "
                                 "reproducible random sample (seed 42) is used in the app.")
        if products is None or products.empty:
            products = derive_products_from_reviews(reviews)
            rrep.warnings.append("No product table: products were derived from reviews "
                                 "(titles are ASINs, ratings/counts come from this review sample).")
        else:
            if products["rating"].isna().all() and reviews["review_rating"].notna().any():
                agg = reviews.groupby("asin")["review_rating"].mean()
                products["rating"] = products["asin"].map(agg)
                rrep.warnings.append("Product ratings were missing; filled with the mean review "
                                     "rating from the loaded review sample.")
            if products["review_count"].isna().all():
                products["review_count"] = products["asin"].map(reviews.groupby("asin").size())
                rrep.warnings.append("Product review counts were missing; filled with the number "
                                     "of reviews in the loaded sample (not the Amazon total).")
        reviews = link_reviews(reviews, products, rrep)
    if products is None:
        raise ValueError("A product table or a review table is required.")
    return Dataset(products, reviews, reports, source_label, is_demo)
