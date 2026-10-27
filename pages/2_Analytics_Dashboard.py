import streamlit as st

from src import analytics as A
from src import visualizations as V
from src.export_utils import to_csv_bytes
from src.sentiment_analysis import monthly_sentiment, sentiment_counts
from src.ui import (apply_all, dataset_banner, get_scored_reviews, kpi_row, page_header,
                    render_filters, require_products, show)

page_header("📊 Analytics Dashboard", "KPIs and charts update with the sidebar filters.")
ds = require_products()
dataset_banner(ds)
reviews_all = get_scored_reviews() if ds.reviews is not None else None
f = render_filters(ds.products, reviews_all, "dash")
prods, revs = apply_all(ds, reviews_all, f)

if prods.empty:
    st.warning("No products match these filters. Widen them or press **Reset filters**.")
    st.stop()
kpi_row(A.compute_kpis(prods, revs))
st.caption("Review-count figures describe feedback volume, not sales. The review date filter affects only review-level metrics.")
if len(prods) < 10:
    st.warning(f"Only {len(prods)} products selected: statistics may be unreliable.")

tab_cat, tab_dist, tab_rel, tab_brand, tab_rev = st.tabs(
    ["Categories", "Distributions", "Relationships & top products", "Brands", "Reviews & trends"])

with tab_cat:
    if A.has_data(prods, "category"):
        cs = A.category_summary(prods)
        show(V.bar_by_category(cs, "products", "Number of products by category", "Products"))
        if A.has_data(prods, "price"):
            show(V.bar_by_category(cs, "avg_price", "Average price by category", "Average price (USD)"))
        if A.has_data(prods, "rating"):
            show(V.bar_by_category(cs, "avg_rating", "Average rating by category", "Average rating"))
        st.dataframe(cs, use_container_width=True, hide_index=True)
        st.download_button("Download category summary (CSV)", to_csv_bytes(cs), "category_summary.csv", "text/csv")
    else:
        st.info("Category charts need a category column in the dataset.")

with tab_dist:
    c1, c2 = st.columns(2)
    if A.has_data(prods, "rating"):
        c1.plotly_chart(V.histogram(prods, "rating", "Distribution of product ratings", "Average rating", 20),
                        use_container_width=True)
    else:
        c1.info("No rating data.")
    if A.has_data(prods, "price"):
        mask, (lo, hi) = A.iqr_outliers(prods["price"])
        c2.plotly_chart(V.histogram(prods, "price", "Product price distribution", "Price (USD)"),
                        use_container_width=True)
        if hi is not None:
            c2.caption(f"{int(mask.sum())} price outlier(s) above/below the IQR fences "
                       f"(${max(lo, 0):.2f} to ${hi:.2f}); they are kept in the chart.")
    else:
        c2.info("No price data.")
    if A.has_data(prods, "review_count"):
        show(V.histogram(prods, "review_count", "Product review-count distribution (log x-axis)",
                         "Review count", log_x=True))
    else:
        st.info("No review-count data.")

with tab_rel:
    if A.has_data(prods, "price") and A.has_data(prods, "rating"):
        show(V.scatter_price_rating(prods))
    else:
        st.info("Price vs rating needs both price and rating columns.")
    if A.has_data(prods, "review_count"):
        n_top = st.slider("Products to show", 5, 25, 10)
        show(V.top_products_bar(A.top_reviewed(prods, n_top)))
    else:
        st.info("Top products by review count needs a review-count column.")

with tab_brand:
    if A.has_data(prods, "brand"):
        min_p = st.slider("Minimum products per brand", 1, 20, 3)
        bs = A.brand_summary(prods, min_p)
        if bs.empty:
            st.info("No brand has that many products in the current selection.")
        else:
            metric = st.radio("Compare by", ["avg_rating", "avg_price", "products"], horizontal=True)
            show(V.bar_by_category(bs.head(20), metric, f"Brands by {metric} (top 20 by product count)",
                                   metric, col="brand"))
            st.dataframe(bs, use_container_width=True, hide_index=True)
    else:
        st.info("Brand comparison needs a brand column.")

with tab_rev:
    if revs is None or revs.empty:
        st.info("Sentiment and trend charts need a review-level dataset (review text). "
                "Upload one in the Dataset Manager.")
    else:
        show(V.sentiment_bar(sentiment_counts(revs)))
        if A.has_data(revs, "review_date"):
            m = A.monthly_review_counts(revs)
            if len(m) >= 3:
                show(V.line_trend(m, "month", "reviews", "Reviews per month", "Reviews"))
                show(V.sentiment_trend(monthly_sentiment(revs)))
            else:
                st.info("Fewer than 3 months of dated reviews: no trend chart.")
        else:
            st.info("No reliable review dates: time trends unavailable.")
