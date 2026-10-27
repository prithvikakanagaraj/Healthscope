import streamlit as st

from src import analytics as A
from src import insights as I
from src import sentiment_analysis as S
from src import visualizations as V
from src.export_utils import html_report, summary_markdown, to_csv_bytes
from src.sentiment_analysis import monthly_sentiment
from src.ui import (dataset_banner, get_scored_reviews, page_header, require_products, show)

page_header("💡 Big Data Insights & Findings", "Every sentence below is computed from the loaded dataset.")
ds = require_products()
dataset_banner(ds)
prods = ds.products
reviews = get_scored_reviews() if ds.reviews is not None else None

st.subheader("Automatic findings")
insights, na = I.generate_insights(prods, reviews)
for i in insights:
    st.markdown(f"**{i.title}** - {i.text}")
    st.caption(f"Basis: {i.basis}")
if na:
    with st.expander("Findings that could not be calculated"):
        for x in na:
            st.write("- " + x)

st.subheader("Supporting charts")
cs = A.category_summary(prods) if A.has_data(prods, "category") else None
figs = []
if cs is not None and len(cs):
    figs += [("Products by category", V.bar_by_category(cs, "products", "Products by category", "Products")),
             ("Average rating by category", V.bar_by_category(cs, "avg_rating", "Average rating by category", "Average rating"))]
    if A.has_data(prods, "price"):
        figs.append(("Median price by category", V.bar_by_category(cs, "median_price", "Median price by category", "Median price (USD)")))
if A.has_data(prods, "rating") and A.has_data(prods, "review_count"):
    figs.append(("Rating vs review count", V.scatter_rating_reviews(prods)))
if A.has_data(prods, "price") and A.has_data(prods, "rating"):
    figs.append(("Price vs rating", V.scatter_price_rating(prods)))
corr = A.correlation_matrix(prods)
if corr is not None:
    figs.append(("Rank correlations", V.corr_heatmap(corr)))
if reviews is not None and A.has_data(reviews, "review_date") and len(monthly_sentiment(reviews)) >= 3:
    figs.append(("Sentiment trend", V.sentiment_trend(monthly_sentiment(reviews))))
if not figs:
    st.info("Not enough columns to draw supporting charts.")
for title, fig in figs:
    show(fig)
st.caption("Correlation charts use Spearman ranks and show association, not causation.")

with st.expander("Optional advanced analysis (separate from the core app)"):
    if st.checkbox("Outlier detection (IQR rule)"):
        for col in ("price", "review_count"):
            if A.has_data(prods, col):
                mask, (lo, hi) = A.iqr_outliers(prods[col])
                st.write(f"**{col}**: {int(mask.sum())} of {int(mask.size)} values outside [{lo:.2f}, {hi:.2f}]")
    if st.checkbox("K-means clustering of products"):
        k = st.slider("Number of clusters", 2, 6, 3)
        d, summ = A.kmeans_clusters(prods, k)
        if d is None:
            st.info("Needs at least 20 products with price, rating and review count.")
        else:
            show(V.cluster_scatter(d))
            st.dataframe(summ, hide_index=True, use_container_width=True)
            st.caption("Clusters group similar products; the labels have no meaning beyond the numbers shown.")
    if st.checkbox("TF-IDF distinctive terms by sentiment") and reviews is not None:
        res = S.tfidf_terms(reviews["review_text"], reviews["sentiment"])
        if not res:
            st.info("Not enough review text.")
        for lab, tbl in res.items():
            st.markdown(f"**{lab}**")
            st.dataframe(tbl, hide_index=True, use_container_width=True)

st.subheader("Data-quality observations")
for line in I.data_quality_observations(ds.reports) or ["No quality information recorded."]:
    st.write("- " + line)
st.subheader("Limitations and possible sources of bias")
for line in I.LIMITATIONS:
    st.write("- " + line)

st.subheader("Downloads")
c1, c2, c3 = st.columns(3)
kp = A.compute_kpis(prods, reviews)
c1.download_button("Analysis summary (Markdown)",
                   summary_markdown(ds, kp, {}, insights, na, cs).encode("utf-8"), "analysis_summary.md")
if cs is not None:
    c2.download_button("Category summary (CSV)", to_csv_bytes(cs), "category_summary.csv", "text/csv")
if c3.button("Build HTML findings report"):
    with st.spinner("Building report..."):
        st.session_state["report_html"] = html_report(
            ds, kp, {}, insights, na, figs, I.LIMITATIONS, I.data_quality_observations(ds.reports))
if "report_html" in st.session_state:
    c3.download_button("Download report (HTML)", st.session_state["report_html"].encode("utf-8"),
                       "healthscope_report.html", "text/html")
