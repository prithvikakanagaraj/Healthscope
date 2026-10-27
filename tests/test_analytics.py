import pandas as pd
import pytest

from src import analytics as A
from src.data_loader import load_default_dataset


@pytest.fixture(scope="module")
def products():
    return load_default_dataset().products


def test_kpis_match_dataframe(products):
    k = A.compute_kpis(products, None)
    assert k["products"] == len(products) and k["reviews"] is None
    assert k["avg_rating"] == pytest.approx(products["rating"].mean())
    assert k["categories"] == products["category"].nunique()


def test_filters_combine_and_do_not_mutate(products):
    n = len(products)
    out = A.apply_product_filters(products, {"min_rating": 4.5, "min_reviews": 100, "price": (5, 30)})
    assert len(products) == n and (out["rating"] >= 4.5).all() and (out["review_count"] >= 100).all()
    assert out["price"].between(5, 30).all()


def test_search_and_category_filter(products):
    cat = products["category"].iloc[0]
    out = A.apply_product_filters(products, {"categories": [cat], "search": "demo"})
    assert (out["category"] == cat).all()


def test_category_summary_totals(products):
    cs = A.category_summary(products)
    assert cs["products"].sum() == len(products[products["category"] != "Unknown"])


def test_brand_summary_respects_minimum(products):
    assert (A.brand_summary(products, 5)["products"] >= 5).all()


def test_missing_columns_are_reported_not_crashing():
    df = pd.DataFrame({"asin": ["a"], "title": ["A"], "category": ["Unknown"], "brand": ["Unknown"],
                       "price": [None], "rating": [None], "review_count": [None]})
    assert not A.has_data(df, "price") and not A.has_data(df, "brand")
    assert A.correlation_matrix(df) is None
    assert A.rank_correlation(df["price"], df["rating"])["rho"] is None


def test_rank_correlation_detects_monotonic_relationship():
    s = pd.Series(range(30))
    assert A.rank_correlation(s, s * 2)["rho"] == pytest.approx(1.0)


def test_iqr_outliers_and_kmeans(products):
    mask, (lo, hi) = A.iqr_outliers(pd.Series([1, 2, 2, 3, 2, 3, 2, 100]))
    assert mask.sum() == 1 and hi < 100
    d, summary = A.kmeans_clusters(products, 3)
    assert d["cluster"].nunique() == 3 and summary["products"].sum() == len(d)


def test_review_filter_by_date_and_product():
    prods = pd.DataFrame({"asin": ["a"]})
    rev = pd.DataFrame({"asin": ["a", "b", "a"], "review_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-01-01"])})
    out = A.apply_review_filters(rev, prods, ("2020-01-01", "2020-12-31"))
    assert len(out) == 1
