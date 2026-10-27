import pytest

from src.data_loader import DataLoadError, load_default_dataset, read_table
from src.validation import auto_map_products, auto_map_reviews, validate_product_mapping


def test_read_csv_bytes():
    df = read_table(b"title,price\nA,$1.5\nB,2\n", "x.csv")
    assert list(df.columns) == ["title", "price"] and len(df) == 2


def test_read_jsonl_bytes():
    df = read_table(b'{"asin":"A1","overall":5}\n{"asin":"A2","overall":1}\n', "r.jsonl")
    assert len(df) == 2


def test_rejects_bad_extension_and_empty():
    with pytest.raises(DataLoadError):
        read_table(b"x", "evil.exe")
    with pytest.raises(DataLoadError):
        read_table(b"", "a.csv")


def test_rejects_header_only_csv():
    with pytest.raises(DataLoadError):
        read_table(b"title,price\n", "a.csv")


def test_auto_mapping_ucsd_style_names():
    m = auto_map_products(["asin", "title", "main_category", "average_rating", "rating_number", "price", "store"])
    assert m["category"] == "main_category" and m["rating"] == "average_rating"
    assert m["review_count"] == "rating_number" and m["brand"] == "store"
    r = auto_map_reviews(["reviewerID", "asin", "reviewText", "overall", "reviewTime"])
    assert r["review_text"] == "reviewText" and r["review_date"] == "reviewTime"


def test_mapping_validation_requires_identifier():
    errors, _ = validate_product_mapping({})
    assert errors


def test_default_dataset_loads_and_is_labelled_when_demo():
    ds = load_default_dataset()
    assert len(ds.products) > 0 and ds.reviews is not None
    if ds.is_demo:
        assert "SYNTHETIC" in ds.source_label
