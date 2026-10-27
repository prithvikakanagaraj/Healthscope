import numpy as np

from src import insights as I
from src import sentiment_analysis as S
from src.data_loader import load_default_dataset


def test_insights_are_computed_and_traceable():
    ds = load_default_dataset()
    rev = ds.reviews.copy()
    rev["compound"] = np.where(rev["review_rating"] >= 4, 0.6, np.where(rev["review_rating"] <= 2, -0.6, 0.0))
    ins, na = I.generate_insights(ds.products, S.add_labels(rev))
    assert ins and all(i.basis for i in ins)
    assert any(str(len(ds.products)) in i.text for i in ins)


def test_missing_reviews_explained_not_invented():
    ds = load_default_dataset()
    ins, na = I.generate_insights(ds.products, None)
    assert any("review-level" in x for x in na)
    assert not any("sentiment" in i.title.lower() for i in ins)
