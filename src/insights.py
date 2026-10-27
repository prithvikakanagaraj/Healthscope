"""Plain-language findings. Every sentence is built from numbers computed on the loaded data."""
from dataclasses import dataclass

import pandas as pd

from . import analytics as A
from .config import HIGH_RATING_CUTOFF, MIN_BRAND_PRODUCTS, MIN_GROUP_REVIEWS
from .sentiment_analysis import sentiment_counts, theme_counts


@dataclass
class Insight:
    title: str
    text: str
    basis: str  # which calculation and sample size the sentence comes from


def describe_rho(rho: float) -> str:
    a = abs(rho)
    strength = "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{strength} {'positive' if rho > 0 else 'negative'}" if a >= 0.1 else strength


def generate_insights(products: pd.DataFrame, reviews: pd.DataFrame | None):
    """Return (insights, unavailable) where unavailable explains what could not be computed."""
    out, na = [], []
    n = len(products)
    if n == 0:
        return out, ["No products match the current data/filters."]

    if A.has_data(products, "category"):
        cs = A.category_summary(products)
        if len(cs):
            top = cs.iloc[0]
            out.append(Insight("Category distribution",
                               f"{len(cs)} categories are represented. '{top['category']}' has the most "
                               f"products ({int(top['products'])} of {n}, {top['products'] / n:.0%}).",
                               f"Count of products per category, n={n}."))
            ok = cs[cs["products"] >= 5]
            if A.has_data(products, "price") and len(ok) >= 2:
                hi, lo = ok.sort_values("median_price").iloc[-1], ok.sort_values("median_price").iloc[0]
                out.append(Insight("Price variation across categories",
                                   f"Median price is highest in '{hi['category']}' (${hi['median_price']:.2f}) "
                                   f"and lowest in '{lo['category']}' (${lo['median_price']:.2f}).",
                                   f"Median price per category; categories with >=5 products "
                                   f"({len(ok)} categories)."))
            else:
                na.append("Price variation: needs a price column and >=2 categories with >=5 products.")
            ok = cs[cs["products_with_rating"] >= 5]
            if len(ok) >= 2:
                hi, lo = ok.sort_values("avg_rating").iloc[-1], ok.sort_values("avg_rating").iloc[0]
                out.append(Insight("Rating variation across categories",
                                   f"Average rating ranges from {lo['avg_rating']:.2f} ('{lo['category']}') "
                                   f"to {hi['avg_rating']:.2f} ('{hi['category']}'). Differences may reflect "
                                   "product mix rather than category quality.",
                                   "Mean product rating per category; categories with >=5 rated products."))
            else:
                na.append("Rating variation: needs a rating column and >=2 categories with >=5 rated products.")
    else:
        na.append("Category insights: no category column available.")

    for a, b, label in [("rating", "review_count", "rating and review count"),
                        ("price", "rating", "price and rating")]:
        r = A.rank_correlation(products[a], products[b])
        if r["rho"] is None:
            na.append(f"Relationship between {label}: not enough complete data (n={r['n']}).")
        else:
            out.append(Insight(f"Relationship: {label}",
                               f"There is a {describe_rho(r['rho'])} rank correlation between {label} "
                               f"(Spearman rho={r['rho']:.2f}, n={r['n']}). This is an association only and "
                               "does not show that one causes the other.", f"Spearman rank correlation, n={r['n']}."))

    top = A.top_reviewed(products, 3)
    if len(top):
        names = "; ".join(f"{t.title[:50]} ({int(t.review_count):,} reviews)" for t in top.itertuples())
        out.append(Insight("Highly reviewed products",
                           f"Most-reviewed products: {names}. Review counts show how much feedback exists, "
                           "not how many units were sold.", f"Top 3 by review_count of {n} products."))
    hv = A.high_rated_with_volume(products)
    if A.has_data(products, "rating") and A.has_data(products, "review_count"):
        out.append(Insight("High ratings with sufficient review volume",
                           f"{len(hv)} product(s) have a rating of at least {HIGH_RATING_CUTOFF} and at least "
                           "50 reviews. This is a simple stated rule, not a measure of 'best' or of "
                           "medical effectiveness.", f"Rule: rating>={HIGH_RATING_CUTOFF} and review_count>=50."))
    if A.has_data(products, "brand"):
        bs = A.brand_summary(products, MIN_BRAND_PRODUCTS).dropna(subset=["avg_rating"])
        if len(bs):
            b = bs.sort_values("avg_rating").iloc[-1]
            out.append(Insight("Brands with the strongest observed ratings",
                               f"Among brands with at least {MIN_BRAND_PRODUCTS} products, '{b['brand']}' has the "
                               f"highest average rating ({b['avg_rating']:.2f}, {int(b['products'])} products).",
                               f"{len(bs)} brands with >={MIN_BRAND_PRODUCTS} products."))
        else:
            na.append(f"Brand ratings: no brand has >={MIN_BRAND_PRODUCTS} products with ratings.")
    else:
        na.append("Brand insights: no brand column available.")

    if reviews is not None and len(reviews) and "sentiment" in reviews.columns:
        sc = sentiment_counts(reviews).set_index("sentiment")["percent"]
        out.append(Insight("Review sentiment pattern",
                           f"{sc['Positive']:.1f}% of {len(reviews):,} reviews are classified positive, "
                           f"{sc['Neutral']:.1f}% neutral and {sc['Negative']:.1f}% negative by VADER. "
                           "VADER can miss sarcasm, context and medical wording.",
                           f"VADER compound thresholds, n={len(reviews):,} reviews."))
        for lab, name in (("Positive", "positive"), ("Negative", "negative")):
            sub = reviews[reviews["sentiment"] == lab]["review_text"]
            if len(sub) >= MIN_GROUP_REVIEWS:
                t = theme_counts(sub)
                t = t[t["reviews"] > 0].head(3)
                if len(t):
                    out.append(Insight(f"Frequent {name} themes",
                                       "Most mentioned among " + name + " reviews: " + ", ".join(
                                           f"{r['theme']} ({r['share_%']}%)" for _, r in t.iterrows()) + ".",
                                       f"Keyword theme match in {len(sub):,} {name} reviews."))
            else:
                na.append(f"{lab} themes: fewer than {MIN_GROUP_REVIEWS} {name} reviews.")
        if A.has_data(reviews, "review_date"):
            m = A.monthly_review_counts(reviews)
            if len(m) >= 6:
                peak = m.loc[m["reviews"].idxmax()]
                out.append(Insight("Time-based pattern",
                                   f"Reviews span {m['month'].min():%b %Y} to {m['month'].max():%b %Y}; "
                                   f"the busiest month is {peak['month']:%b %Y} ({int(peak['reviews'])} reviews).",
                                   f"Monthly counts over {len(m)} months."))
            else:
                na.append("Time trends: fewer than 6 months of dated reviews.")
        else:
            na.append("Time trends: no usable review dates.")
    else:
        na.append("Sentiment, theme and time insights need a review-level dataset (review text).")
    return out, na


def data_quality_observations(reports) -> list:
    lines = []
    for r in reports:
        lines.append(f"{r.kind.capitalize()}: {r.rows_loaded:,} rows loaded, {r.rows_retained:,} retained.")
        lines += [f"{r.kind.capitalize()}: {v:,} - {k}" for k, v in {**r.removed, **r.flagged}.items()]
        lines += [f"{r.kind.capitalize()} warning: {w}" for w in r.warnings]
    return lines


LIMITATIONS = [
    "Ratings and review counts reflect people who chose to write or rate; they are not a random sample (self-selection bias).",
    "Review counts measure feedback volume, not sales or market share.",
    "Fake, incentivised or duplicate reviews may exist and cannot be fully detected here.",
    "VADER is a general-purpose lexicon model; it can misread sarcasm, negation, context and medical terms.",
    "Sentiment and ratings are customer opinions, not evidence of safety or clinical effectiveness.",
    "Observed relationships are associations; they do not establish causation.",
    "Results describe only the loaded dataset (and sample), which may be older than current Amazon listings.",
    "Brand and category labels depend on source metadata quality and normalisation rules.",
]
