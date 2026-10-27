"""Streamlit helpers shared by all pages: session data, filters, formatting, pagination."""
import math

import pandas as pd
import streamlit as st

from . import analytics as A
from . import sentiment_analysis as S
from .config import NEG_THRESHOLD, POS_THRESHOLD, get_logger
from .data_loader import load_default_dataset

log = get_logger(__name__)

CSS = """
<style>
[data-testid="stMetric"]{background:#F1F5F9;border-left:5px solid #13A89E;padding:12px 16px;border-radius:8px;}
[data-testid="stMetricLabel"]{color:#0B2545;}
h1,h2,h3{color:#0B2545;}
.hs-hero{background:linear-gradient(120deg,#0B2545,#13A89E);color:#fff;padding:26px 30px;border-radius:12px;margin-bottom:18px;}
.hs-hero h1{color:#fff;margin:0;} .hs-hero p{margin:6px 0 0 0;opacity:.92;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"<div class='hs-hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading dataset...")
def _default_dataset():
    return load_default_dataset()  # cached copy only holds repository / demo data, never uploads


# ------------------------------------------------------------------ session data
def get_dataset():
    """The dataset of THIS browser session (uploads never leave the session)."""
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = _default_dataset()
    return st.session_state["dataset"]


def set_dataset(ds):
    st.session_state["dataset"] = ds
    for k in [k for k in st.session_state if str(k).startswith("flt_")]:
        del st.session_state[k]


def reset_dataset():
    set_dataset(_default_dataset())


def dataset_banner(ds):
    if ds.is_demo:
        st.warning("Showing **SYNTHETIC DEMO DATA** - invented records, not real Amazon data. "
                   "Add a real dataset (see README) or upload one on the Dataset Manager page.")
    else:
        st.caption(f"Dataset: {ds.source_label}")


def require_products():
    ds = get_dataset()
    if ds.products is None or ds.products.empty:
        st.error("No product data is loaded. Open the Dataset Manager page to load a dataset.")
        st.stop()
    return ds


def get_thresholds():
    return st.session_state.get("pos_thr", POS_THRESHOLD), st.session_state.get("neg_thr", NEG_THRESHOLD)


def get_scored_reviews():
    """Reviews with VADER compound scores (computed once per dataset) and current labels."""
    ds = get_dataset()
    if ds.reviews is None or ds.reviews.empty:
        return None
    if not ds.sentiment_ready:
        with st.spinner("Scoring reviews with VADER (done once per dataset)..."):
            try:
                ds.reviews["compound"] = S.score_texts(ds.reviews["review_text"])
            except Exception as exc:
                log.exception("Sentiment scoring failed: %s", exc)
                st.error("Sentiment analysis could not start (the VADER lexicon is unavailable). "
                         "Check your internet connection and that requirements are installed.")
                return None
        ds.sentiment_ready = True
    return S.add_labels(ds.reviews, *get_thresholds())


@st.cache_data(show_spinner=False)
def cached_top_terms(texts: pd.Series, n: int = 20):
    return S.top_terms(texts, n)


@st.cache_data(show_spinner=False)
def cached_theme_counts(texts: pd.Series):
    return S.theme_counts(texts)


# ------------------------------------------------------------------ formatting
def fmt(v, prefix="", decimals=2, dash="n/a"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return f"{prefix}{v:,.{decimals}f}" if isinstance(v, float) else f"{prefix}{v:,}"


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


def kpi_row(k: dict):
    cols = st.columns(6)
    items = [("Products", fmt(k["products"]), "Products in the current selection"),
             ("Valid reviews", fmt(k["reviews"]) if k["reviews"] is not None else "n/a",
              "Needs a review-level dataset"),
             ("Avg rating", fmt(k["avg_rating"]), "Mean of product average ratings"),
             ("Avg price", fmt(k["avg_price"], "$"), "Mean listed price (USD), missing prices ignored"),
             ("Categories", fmt(k["categories"]), "Distinct categories (excluding Unknown)"),
             ("Brands", fmt(k["brands"]), "Distinct brands (excluding Unknown)")]
    for c, (label, value, tip) in zip(cols, items):
        c.metric(label, value, help=tip)


def paginate(df: pd.DataFrame, key: str, default_size=25):
    """Render page controls and return the visible slice."""
    c1, c2, c3 = st.columns([1, 1, 3])
    size = c1.selectbox("Rows per page", [10, 25, 50, 100], index=[10, 25, 50, 100].index(default_size),
                        key=f"{key}_size")
    pages = max(1, math.ceil(len(df) / size))
    page = c2.number_input("Page", 1, pages, 1, key=f"{key}_page_{pages}")
    c3.caption(f"{len(df):,} rows - page {page} of {pages}")
    start = (int(page) - 1) * size
    return df.iloc[start:start + size]


# ------------------------------------------------------------------ filters (kept in session state)
def _get(key, default):
    return st.session_state.get(f"flt_{key}", default)


def _put(key, value):
    st.session_state[f"flt_{key}"] = value
    return value


def _reset(scope):
    for k in [k for k in st.session_state if str(k).startswith(f"flt_{scope}_")]:
        del st.session_state[k]


def render_filters(products: pd.DataFrame, reviews: pd.DataFrame | None, scope: str, with_search=False):
    """Sidebar filters. Returns a dict understood by analytics.apply_product_filters."""
    sb, f = st.sidebar, {}
    sb.markdown("### Filters")
    if with_search:
        f["search"] = _put(f"{scope}_search", sb.text_input(
            "Search title / description", value=_get(f"{scope}_search", "")))
    for field, label in (("category", "Category"), ("brand", "Brand")):
        if A.has_data(products, field):
            opts = sorted(products[field].dropna().unique())
            cur = [c for c in _get(f"{scope}_{field}", []) if c in opts]
            f["categories" if field == "category" else "brands"] = _put(
                f"{scope}_{field}", sb.multiselect(label, opts, default=cur))
    if A.has_data(products, "price"):
        lo, hi = float(math.floor(products["price"].min())), float(math.ceil(products["price"].max()))
        if lo < hi:
            cur = _get(f"{scope}_price", (lo, hi))
            cur = (max(lo, cur[0]), min(hi, cur[1]))
            cur = cur if cur[0] <= cur[1] else (lo, hi)
            val = _put(f"{scope}_price", sb.slider("Price range (USD)", lo, hi, cur, step=0.5))
            f["price"] = None if val == (lo, hi) else val
    if A.has_data(products, "rating"):
        val = _put(f"{scope}_rating", sb.slider("Minimum rating", 1.0, 5.0, float(_get(f"{scope}_rating", 1.0)), 0.1))
        f["min_rating"] = val if val > 1.0 else None
    if A.has_data(products, "review_count"):
        mx = int(products["review_count"].max())
        cur = min(int(_get(f"{scope}_rc", 0)), mx)
        val = _put(f"{scope}_rc", sb.number_input("Minimum review count", 0, mx, cur, step=10))
        f["min_reviews"] = int(val) or None
    if reviews is not None and A.has_data(reviews, "review_date"):
        mn, mxd = reviews["review_date"].min().date(), reviews["review_date"].max().date()
        cur = _get(f"{scope}_dates", (mn, mxd))
        cur = cur if mn <= cur[0] <= mxd and mn <= cur[1] <= mxd else (mn, mxd)
        val = sb.date_input("Review date range", value=cur, min_value=mn, max_value=mxd)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            _put(f"{scope}_dates", tuple(val))
            f["date_range"] = None if tuple(val) == (mn, mxd) else tuple(val)
    sb.button("Reset filters", on_click=_reset, args=(scope,), key=f"reset_{scope}")
    return f


def apply_all(ds, reviews, f):
    """Filtered products and reviews for the given filter dict."""
    prods = A.apply_product_filters(ds.products, f)
    revs = A.apply_review_filters(reviews, prods, f.get("date_range")) if reviews is not None else None
    return prods, revs
