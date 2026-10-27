"""VADER sentiment scoring, labelling, themes and frequent terms (no Streamlit here)."""
import re

import numpy as np
import pandas as pd

from .config import (DATA_DIR, MIN_GROUP_REVIEWS, NEG_THRESHOLD, POS_THRESHOLD,
                     SENTIMENT_ORDER, get_logger)

log = get_logger(__name__)
_ANALYZER = None

# Transparent keyword themes: a review "mentions" a theme if it contains any keyword.
THEMES = {
    "Taste / flavour": ["taste", "tastes", "flavor", "flavour", "bitter", "sweet", "aftertaste"],
    "Smell": ["smell", "smells", "odor", "odour", "scent", "fragrance"],
    "Price / value": ["price", "priced", "value", "expensive", "cheap", "worth", "money", "cost"],
    "Packaging": ["package", "packaging", "bottle", "box", "seal", "sealed", "damaged", "leak", "leaked"],
    "Shipping / delivery": ["shipping", "shipped", "delivery", "delivered", "arrived", "late", "slow"],
    "Quality": ["quality", "durable", "broke", "broken", "flimsy", "sturdy", "well made"],
    "Ease of use": ["easy", "simple", "convenient", "hard to use", "difficult", "instructions"],
    "Size / amount": ["size", "small", "large", "big", "tiny", "amount", "quantity", "capsule", "capsules"],
    "User-reported results": ["works", "worked", "effective", "ineffective", "helped", "helps", "results",
                              "noticed", "relief"],
    "Side effects mentioned": ["side effect", "side effects", "nausea", "headache", "rash", "stomach",
                               "upset", "allergic", "reaction"],
}


def get_analyzer():
    """NLTK VADER (downloads its lexicon once); falls back to the identical vaderSentiment package."""
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            _ANALYZER = SentimentIntensityAnalyzer()
        except LookupError:
            target = DATA_DIR.parent / "nltk_data"
            nltk.download("vader_lexicon", download_dir=str(target), quiet=True)
            nltk.data.path.append(str(target))
            _ANALYZER = SentimentIntensityAnalyzer()
    except Exception as exc:
        log.warning("NLTK VADER unavailable (%s); trying vaderSentiment.", exc)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as Alt
        _ANALYZER = Alt()
    return _ANALYZER


def score_texts(texts) -> np.ndarray:
    """VADER compound score (-1..+1) per text. Slow part: compute once, then re-label cheaply."""
    sia = get_analyzer()
    return np.array([sia.polarity_scores(str(t))["compound"] for t in texts], dtype=float)


def label_scores(compound, pos=POS_THRESHOLD, neg=NEG_THRESHOLD) -> pd.Series:
    """compound >= pos -> Positive, <= neg -> Negative, otherwise Neutral."""
    if neg >= pos:
        raise ValueError("Negative threshold must be lower than the positive threshold.")
    c = np.asarray(compound, dtype=float)
    lab = np.select([c >= pos, c <= neg], ["Positive", "Negative"], default="Neutral")
    return pd.Series(pd.Categorical(lab, categories=SENTIMENT_ORDER), name="sentiment")


def add_labels(reviews: pd.DataFrame, pos=POS_THRESHOLD, neg=NEG_THRESHOLD) -> pd.DataFrame:
    out = reviews.copy()
    out["sentiment"] = label_scores(out["compound"].to_numpy(), pos, neg).to_numpy()
    return out


def sentiment_counts(reviews: pd.DataFrame) -> pd.DataFrame:
    n = reviews["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = int(n.sum())
    return pd.DataFrame({"sentiment": SENTIMENT_ORDER, "reviews": n.values,
                         "percent": (n.values / total * 100).round(1) if total else 0.0})


def sentiment_by(reviews: pd.DataFrame, col: str, min_reviews: int = MIN_GROUP_REVIEWS) -> pd.DataFrame:
    """Percent positive/neutral/negative per group, only groups with >= min_reviews."""
    ct = pd.crosstab(reviews[col], reviews["sentiment"]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    ct["reviews"] = ct.sum(axis=1)
    ct = ct[ct["reviews"] >= min_reviews]
    for s in SENTIMENT_ORDER:
        ct[f"{s.lower()}_%"] = (ct[s] / ct["reviews"] * 100).round(1)
    ct["avg_compound"] = reviews.groupby(col)["compound"].mean().reindex(ct.index).round(3)
    return ct.sort_values("reviews", ascending=False).reset_index()


def monthly_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    d = reviews.dropna(subset=["review_date"])
    if d.empty:
        return pd.DataFrame()
    d = d.assign(month=d["review_date"].dt.to_period("M").dt.to_timestamp())
    ct = pd.crosstab(d["month"], d["sentiment"]).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    ct["reviews"] = ct.sum(axis=1)
    ct["positive_%"] = (ct["Positive"] / ct["reviews"] * 100).round(1)
    ct["negative_%"] = (ct["Negative"] / ct["reviews"] * 100).round(1)
    return ct.reset_index()


def theme_counts(texts: pd.Series) -> pd.DataFrame:
    """Number and share of reviews mentioning each keyword theme."""
    low = texts.fillna("").str.lower()
    total = max(len(low), 1)
    rows = []
    for theme, words in THEMES.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        n = int(low.str.contains(pattern, regex=True).sum())
        rows.append({"theme": theme, "reviews": n, "share_%": round(n / total * 100, 1)})
    return pd.DataFrame(rows).sort_values("reviews", ascending=False).reset_index(drop=True)


def top_terms(texts: pd.Series, n: int = 20, ngram=(1, 2), max_docs: int = 20000) -> pd.DataFrame:
    """Most frequent words/phrases (number of reviews containing the term), stop-words removed."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    texts = texts.dropna()
    if len(texts) < 2:
        return pd.DataFrame(columns=["term", "reviews"])
    if len(texts) > max_docs:
        texts = texts.sample(max_docs, random_state=42)
    try:
        vec = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=ngram, binary=True,
                              min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        m = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return pd.DataFrame(columns=["term", "reviews"])
    counts = np.asarray(m.sum(axis=0)).ravel()
    out = pd.DataFrame({"term": vec.get_feature_names_out(), "reviews": counts})
    return out.sort_values("reviews", ascending=False).head(n).reset_index(drop=True)


def tfidf_terms(texts: pd.Series, labels: pd.Series, n: int = 12, max_docs: int = 20000) -> dict:
    """Optional: terms with the highest mean TF-IDF weight within each sentiment class."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    d = pd.DataFrame({"t": texts, "l": labels}).dropna()
    if len(d) < 20:
        return {}
    if len(d) > max_docs:
        d = d.sample(max_docs, random_state=42)
    vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2), min_df=3,
                          max_features=5000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        m = vec.fit_transform(d["t"])
    except ValueError:
        return {}
    names, out = vec.get_feature_names_out(), {}
    for lab in SENTIMENT_ORDER:
        idx = np.where(d["l"].astype(str).to_numpy() == lab)[0]
        if len(idx) >= 5:
            w = np.asarray(m[idx].mean(axis=0)).ravel()
            top = np.argsort(w)[::-1][:n]
            out[lab] = pd.DataFrame({"term": names[top], "mean_tfidf": w[top].round(4)})
    return out
