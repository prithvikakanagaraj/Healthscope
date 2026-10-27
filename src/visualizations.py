"""Plotly figures with one consistent look. Every function takes already-filtered data."""
import plotly.express as px
import plotly.graph_objects as go

from .config import NAVY, SENTIMENT_COLORS, SENTIMENT_ORDER, TEAL

_MAX_POINTS = 5000


def _style(fig, height=380, legend=True):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(color=NAVY), title_font_size=16, showlegend=legend,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def bar_by_category(summary, y, title, ylabel, col="category"):
    d = summary.dropna(subset=[y]).sort_values(y, ascending=True)
    fig = px.bar(d, x=y, y=col, orientation="h", title=title, color_discrete_sequence=[TEAL],
                 labels={y: ylabel, col: ""}, hover_data=["products"])
    return _style(fig, max(320, 28 * len(d) + 120), legend=False)


def histogram(df, col, title, xlabel, nbins=30, log_x=False):
    d = df.dropna(subset=[col])
    fig = px.histogram(d, x=col, nbins=nbins, title=title, color_discrete_sequence=[NAVY],
                       labels={col: xlabel}, log_x=log_x)
    fig.update_layout(yaxis_title="Number of products")
    return _style(fig, legend=False)


def top_products_bar(top):
    d = top.sort_values("review_count")
    fig = px.bar(d, x="review_count", y="title", orientation="h", color_discrete_sequence=[TEAL],
                 title="Top products by review count (not sales)", hover_data=["rating", "price"],
                 labels={"review_count": "Review count", "title": ""})
    return _style(fig, max(320, 32 * len(d) + 120), legend=False)


def scatter_price_rating(df, color="category"):
    d = df.dropna(subset=["price", "rating"])
    if len(d) > _MAX_POINTS:
        d = d.sample(_MAX_POINTS, random_state=42)
    fig = px.scatter(d, x="price", y="rating", color=color if color in d else None, opacity=0.7,
                     hover_name="title", hover_data=["review_count"], title="Price vs rating",
                     labels={"price": "Price (USD)", "rating": "Average rating"})
    return _style(fig, 420)


def scatter_rating_reviews(df):
    d = df.dropna(subset=["rating", "review_count"])
    if len(d) > _MAX_POINTS:
        d = d.sample(_MAX_POINTS, random_state=42)
    fig = px.scatter(d, x="review_count", y="rating", log_x=True, opacity=0.6, hover_name="title",
                     color_discrete_sequence=[NAVY], title="Rating vs review count (log scale)",
                     labels={"review_count": "Review count (log)", "rating": "Average rating"})
    return _style(fig, 400, legend=False)


def sentiment_bar(counts):
    fig = px.bar(counts, x="sentiment", y="reviews", color="sentiment", text="percent",
                 color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": SENTIMENT_ORDER},
                 title="Review sentiment distribution", labels={"reviews": "Reviews", "sentiment": ""})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    return _style(fig, legend=False)


def sentiment_stacked(summary, group, title):
    d = summary.sort_values("reviews", ascending=False).head(20)
    long = d.melt(id_vars=[group], value_vars=[f"{s.lower()}_%" for s in SENTIMENT_ORDER],
                  var_name="sentiment", value_name="percent")
    long["sentiment"] = long["sentiment"].str.replace("_%", "").str.capitalize()
    fig = px.bar(long, y=group, x="percent", color="sentiment", orientation="h", title=title,
                 color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": SENTIMENT_ORDER},
                 labels={"percent": "% of reviews", group: ""})
    return _style(fig, max(320, 28 * len(d) + 140))


def line_trend(df, x, y, title, ylabel):
    fig = px.line(df, x=x, y=y, title=title, markers=True, color_discrete_sequence=[TEAL],
                  labels={x: "Month", y: ylabel})
    return _style(fig, legend=False)


def sentiment_trend(monthly):
    fig = go.Figure()
    fig.add_scatter(x=monthly["month"], y=monthly["positive_%"], name="Positive %",
                    line=dict(color=SENTIMENT_COLORS["Positive"]))
    fig.add_scatter(x=monthly["month"], y=monthly["negative_%"], name="Negative %",
                    line=dict(color=SENTIMENT_COLORS["Negative"]))
    fig.update_layout(title="Share of positive / negative reviews by month", yaxis_title="% of reviews")
    return _style(fig)


def terms_bar(terms, title, color=TEAL):
    d = terms.sort_values("reviews")
    fig = px.bar(d, x="reviews", y=d.columns[0], orientation="h", title=title,
                 color_discrete_sequence=[color], labels={"reviews": "Reviews mentioning", d.columns[0]: ""})
    return _style(fig, max(320, 26 * len(d) + 120), legend=False)


def corr_heatmap(corr):
    fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Spearman rank correlation")
    return _style(fig, 340, legend=False)


def cluster_scatter(d):
    fig = px.scatter(d, x="price", y="rating", color="cluster", size="review_count", size_max=18,
                     opacity=0.7, hover_name="title", title="K-means clusters (price, rating, review count)")
    return _style(fig, 440)
