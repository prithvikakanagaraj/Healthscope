import numpy as np
import pandas as pd
import pytest

from src import sentiment_analysis as S


def test_label_thresholds_are_configurable():
    scores = [0.9, 0.04, 0.0, -0.04, -0.9]
    assert list(S.label_scores(scores)) == ["Positive", "Neutral", "Neutral", "Neutral", "Negative"]
    assert list(S.label_scores(scores, pos=0.5, neg=-0.5)) == ["Positive", "Neutral", "Neutral", "Neutral", "Negative"]
    assert list(S.label_scores([0.3], pos=0.2, neg=-0.2)) == ["Positive"]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        S.label_scores([0.1], pos=-0.5, neg=0.5)


def _reviews():
    df = pd.DataFrame({"review_text": ["great taste", "bad smell and awful packaging", "okay", "great value"],
                       "compound": [0.8, -0.7, 0.0, 0.7], "category": ["A", "A", "B", "B"],
                       "review_date": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05", "2020-03-06"])})
    return S.add_labels(df)


def test_counts_percentages_and_groups():
    df = _reviews()
    c = S.sentiment_counts(df).set_index("sentiment")
    assert c.loc["Positive", "reviews"] == 2 and c["percent"].sum() == pytest.approx(100)
    g = S.sentiment_by(df, "category", min_reviews=2).set_index("category")
    assert g.loc["A", "reviews"] == 2 and g.loc["A", "negative_%"] == 50.0
    assert S.sentiment_by(df, "category", min_reviews=5).empty


def test_themes_and_terms():
    df = _reviews()
    t = S.theme_counts(df["review_text"]).set_index("theme")
    assert t.loc["Taste / flavour", "reviews"] == 1 and t.loc["Smell", "reviews"] == 1
    terms = S.top_terms(pd.Series(["great taste"] * 3 + ["bad smell"] * 2), 3)
    assert terms.iloc[0]["term"] in ("great", "taste", "great taste")


def test_monthly_sentiment():
    m = S.monthly_sentiment(_reviews())
    assert m["reviews"].sum() == 4


def test_vader_polarity_if_available():
    try:
        s = S.score_texts(["I love this, it is excellent", "This is terrible and I hate it"])
    except Exception:
        pytest.skip("VADER lexicon not available offline")
    assert s[0] > 0.05 and s[1] < -0.05
