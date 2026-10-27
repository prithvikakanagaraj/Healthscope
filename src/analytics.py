"""Filtering, KPIs and aggregations on cleaned product / review tables."""
import numpy as np
import pandas as pd

from .config import HIGH_RATING_CUTOFF, MIN_BRAND_PRODUCTS, UNKNOWN


def has_data(df: pd.DataFrame | None, col: str) -> bool:
    """True when the column exists and holds real values (not all missing / 'Unknown')."""
    if df is None or col not in df.columns:
        return False
    s = df[col]
    if s.dtype == object or str(s.dtype).startswith("str"):
        s = s.where(s != UNKNOWN)
    return bool(s.notna().any())


def apply_product_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """f keys (all optional): search, categories, brands, price (lo,hi), min_rating, min_reviews.
    A numeric filter left at its full range is not applied, so missing values are not dropped."""
    mask = pd.Series(True, index=df.index)
    if f.get("search"):
        q = str(f["search"]).strip()
        hay = df["title"].fillna("").astype(str)
        if "description" in df.columns:
            hay = hay + " " + df["description"].fillna("").astype(str)
        mask &= hay.str.contains(q, case=False, regex=False)
    if f.get("categories"):
        mask &= df["category"].isin(f["categories"])
    if f.get("brands"):
        mask &= df["brand"].isin(f["brands"])
    if f.get("price"):
        lo, hi = f["price"]
        mask &= df["price"].between(lo, hi)
    if f.get("min_rating"):
        mask &= df["rating"] >= f["min_rating"]
    if f.get("min_reviews"):
        mask &= df["review_count"] >= f["min_reviews"]
    return df[mask]


def apply_review_filters(reviews: pd.DataFrame | None, products: pd.DataFrame, date_range=None):
    """Keep reviews of the (already filtered) products, optionally within a date range."""
    if reviews is None:
        return None
    out = reviews[reviews["asin"].isin(products["asin"].dropna())] if has_data(reviews, "asin") else reviews
    if date_range and len(date_range) == 2 and has_data(out, "review_date"):
        lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["review_date"] >= lo) & (out["review_date"] < hi)]  # end date inclusive
    return out


def compute_kpis(products: pd.DataFrame, reviews: pd.DataFrame | None) -> dict:
    return {
        "products": len(products),
        "reviews": None if reviews is None else len(reviews),
        "avg_rating": products["rating"].mean() if has_data(products, "rating") else None,
        "avg_price": products["price"].mean() if has_data(products, "price") else None,
        "categories": products.loc[products["category"] != UNKNOWN, "category"].nunique()
        if has_data(products, "category") else None,
        "brands": products.loc[products["brand"] != UNKNOWN, "brand"].nunique()
        if has_data(products, "brand") else None,
    }


def _group_summary(products: pd.DataFrame, col: str) -> pd.DataFrame:
    d = products[products[col] != UNKNOWN]
    g = d.groupby(col)
    out = pd.DataFrame({
        "products": g.size(),
        "avg_price": g["price"].mean().round(2),
        "median_price": g["price"].median().round(2),
        "avg_rating": g["rating"].mean().round(2),
        "total_review_count": g["review_count"].sum(min_count=1),
        "products_with_rating": g["rating"].count(),
    })
    return out.sort_values("products", ascending=False).reset_index().rename(columns={col: col})


def category_summary(products: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(products, "category")


def brand_summary(products: pd.DataFrame, min_products: int = MIN_BRAND_PRODUCTS) -> pd.DataFrame:
    out = _group_summary(products, "brand")
    return out[out["products"] >= min_products].reset_index(drop=True)


def top_reviewed(products: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return products.dropna(subset=["review_count"]).nlargest(n, "review_count")


def high_rated_with_volume(products: pd.DataFrame, min_rating=HIGH_RATING_CUTOFF, min_reviews=50):
    """Defined rule (not a 'best product' claim): rating >= min_rating AND review_count >= min_reviews."""
    d = products.dropna(subset=["rating", "review_count"])
    return d[(d["rating"] >= min_rating) & (d["review_count"] >= min_reviews)] \
        .sort_values(["rating", "review_count"], ascending=False)


def rank_correlation(a: pd.Series, b: pd.Series) -> dict:
    """Spearman rank correlation (robust to outliers) with sample size."""
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return {"n": len(d), "rho": None}
    return {"n": len(d), "rho": float(d.rank().corr().iloc[0, 1])}


def correlation_matrix(products: pd.DataFrame) -> pd.DataFrame | None:
    cols = [c for c in ("price", "rating", "review_count") if has_data(products, c)]
    if len(cols) < 2:
        return None
    return products[cols].rank().corr().round(2)


def iqr_outliers(s: pd.Series, k: float = 1.5):
    """Boolean mask of values outside [Q1-k*IQR, Q3+k*IQR] plus the bounds."""
    s = s.dropna()
    if len(s) < 8:
        return pd.Series(False, index=s.index), (None, None)
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (s < lo) | (s > hi), (float(lo), float(hi))


def kmeans_clusters(products: pd.DataFrame, k: int = 3, random_state: int = 42):
    """Optional: K-means on standardised log-price, rating and log-review-count."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    d = products.dropna(subset=["price", "rating", "review_count"]).copy()
    if len(d) < max(20, k * 5):
        return None, None
    X = np.column_stack([np.log1p(d["price"]), d["rating"], np.log1p(d["review_count"])])
    d["cluster"] = (KMeans(n_clusters=k, n_init=10, random_state=random_state)
                    .fit_predict(StandardScaler().fit_transform(X)) + 1).astype(str)
    summary = d.groupby("cluster").agg(products=("asin", "size"), avg_price=("price", "mean"),
                                       avg_rating=("rating", "mean"),
                                       median_review_count=("review_count", "median")).round(2)
    return d, summary.reset_index()


def monthly_review_counts(reviews: pd.DataFrame) -> pd.DataFrame:
    d = reviews.dropna(subset=["review_date"])
    if d.empty:
        return pd.DataFrame(columns=["month", "reviews", "avg_review_rating"])
    g = d.groupby(d["review_date"].dt.to_period("M").dt.to_timestamp())
    return pd.DataFrame({"reviews": g.size(), "avg_review_rating": g["review_rating"].mean().round(2)}) \
        .reset_index().rename(columns={"review_date": "month"})
