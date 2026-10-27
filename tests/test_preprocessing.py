import numpy as np
import pandas as pd

from src.preprocessing import (build_dataset, clean_products, clean_reviews, normalise_labels,
                               parse_dates, parse_price)

PM = {"asin": "asin", "title": "title", "category": "category", "brand": "brand", "price": "price",
      "rating": "rating", "review_count": "n"}


def test_parse_price_variants():
    s = parse_price(pd.Series(["$1,299.99", "12.5 - 15", None, "free", 10, -5]))
    assert s.iloc[0] == 1299.99 and s.iloc[1] == 12.5 and np.isnan(s.iloc[2]) and np.isnan(s.iloc[3])
    assert s.iloc[4] == 10


def test_products_invalid_values_flagged_not_silently_dropped():
    raw = pd.DataFrame({"asin": ["a1", "a2", "a3"], "title": ["A", "B", "C"], "category": ["x", "x", "y"],
                        "brand": ["Acme", "ACME", None], "price": ["$5", "-3", "abc"],
                        "rating": [4.5, 9, 3], "n": [10, -1, 5]})
    df, rep = clean_products(raw, PM)
    assert len(df) == 3 and rep.rows_loaded == 3 and rep.rows_retained == 3
    assert df["price"].isna().sum() == 2 and df["rating"].isna().sum() == 1
    assert df["brand"].iloc[0] == df["brand"].iloc[1]          # case variants merged
    assert df["brand"].iloc[2] == "Unknown"
    assert rep.flagged


def test_duplicate_products_removed_by_asin_keeping_most_reviewed():
    raw = pd.DataFrame({"asin": ["a1", "A1", "a2"], "title": ["A", "A", "B"], "category": ["x"] * 3,
                        "brand": ["b"] * 3, "price": [1, 1, 2], "rating": [4, 4, 3], "n": [5, 50, 1]})
    df, rep = clean_products(raw, PM)
    assert len(df) == 2 and df.loc[df.asin == "A1", "review_count"].iloc[0] == 50
    assert rep.removed


def test_hierarchical_category_uses_second_level():
    raw = pd.DataFrame({"asin": ["a"], "title": ["A"], "category": ["['Health', 'Vitamins', 'D3']"],
                        "brand": ["b"], "price": [1], "rating": [4], "n": [1]})
    df, _ = clean_products(raw, PM)
    assert df["category"].iloc[0] == "Vitamins"


def test_similar_but_different_reviews_are_kept_exact_repeats_removed():
    raw = pd.DataFrame({"asin": ["a"] * 4, "reviewText": ["Great product", "Great product", "Great product!", ""],
                        "overall": [5, 5, 5, 5], "reviewTime": ["01 5, 2020", "01 5, 2020", "02 6, 2020", "x"],
                        "reviewerID": ["u1", "u1", "u2", "u3"]})
    m = {"asin": "asin", "review_text": "reviewText", "review_rating": "overall",
         "review_date": "reviewTime", "reviewer_id": "reviewerID"}
    df, rep = clean_reviews(raw, m)
    assert len(df) == 2                     # 1 empty removed, 1 exact repeat removed, similar text kept
    assert rep.rows_loaded == 4 and df["review_date"].notna().all()


def test_parse_dates_handles_unix_text_and_garbage():
    d = parse_dates(pd.Series([1577836800, 1577836800000, "09 4, 2015", "2020-01-02", "garbage"]))
    assert d.iloc[0].year == 2020 and d.iloc[1].year == 2020 and d.iloc[2].year == 2015
    assert d.iloc[3].day == 2 and pd.isna(d.iloc[4])


def test_empty_input_is_handled():
    df, rep = clean_products(pd.DataFrame(), PM)
    assert df.empty and rep.errors


def test_reviews_only_derives_products_and_links():
    raw = pd.DataFrame({"asin": ["a", "a", "b"], "text": ["good", "bad item", "ok"], "overall": [5, 1, 3]})
    ds = build_dataset(None, {}, raw, {"asin": "asin", "review_text": "text", "review_rating": "overall"})
    assert len(ds.products) == 2 and set(ds.reviews.columns) >= {"category", "brand", "product_title"}
    assert ds.products.set_index("asin").loc["A", "review_count"] == 2
