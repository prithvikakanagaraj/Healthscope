import streamlit as st

from src import analytics as A
from src.config import PROJECT_TITLE, TAGLINE
from src.ui import dataset_banner, fmt, get_dataset, kpi_row, page_header

page_header("🩺 HealthScope", TAGLINE)
ds = get_dataset()
dataset_banner(ds)

st.markdown(f"**Project:** {PROJECT_TITLE}")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Objective")
    st.write("Turn raw Amazon healthcare product data and customer reviews into interactive charts and "
             "plain-language findings about prices, ratings, categories, review volume and customer sentiment.")
with c2:
    st.subheader("Problem statement")
    st.write("Comparing many product listings and reviews by hand is slow. HealthScope centralises the data "
             "so users can explore, filter, compare and understand customer feedback in one place.")

st.subheader("How it works")
st.markdown("1. **Load** a CSV/Excel/JSON dataset (repository sample or your upload).\n"
            "2. **Validate and clean** it with Pandas (types, duplicates, invalid values) and get a quality report.\n"
            "3. **Analyse**: aggregations, correlations and VADER sentiment scoring of review text.\n"
            "4. **Explore and export**: interactive dashboards, comparisons, CSV and HTML reports.")

st.subheader("Loaded dataset")
if ds.products is not None and not ds.products.empty:
    kpi_row(A.compute_kpis(ds.products, ds.reviews))
    st.caption(f"Fields available: " + ", ".join(
        f for f in ("price", "rating", "review_count", "category", "brand", "description", "url")
        if A.has_data(ds.products, f)) + (" | review text and dates: "
        f"{'yes' if ds.reviews is not None else 'no (product-level data only)'}"))
else:
    st.info("No valid data loaded yet. Go to **Dataset Manager**.")

st.subheader("What you can do")
cols = st.columns(3)
cols[0].markdown("**📊 Analytics Dashboard**\n\nKPIs, 11 chart types and live filters.")
cols[0].markdown("**🔎 Product Explorer**\n\nSearch, sort, compare products, export CSV.")
cols[1].markdown("**💬 Review Sentiment**\n\nPositive / neutral / negative analysis, themes, examples.")
cols[1].markdown("**💡 Big Data Insights**\n\nAuto-generated findings with sample sizes and limits.")
cols[2].markdown("**🗂️ Dataset Manager**\n\nUpload, map columns, clean, download reports.")
cols[2].markdown("**ℹ️ About**\n\nMethodology, the 5 Vs of Big Data, limitations.")
st.info("**First time here?** Start with the Analytics Dashboard, then try the Product Explorer. "
        "Use the Dataset Manager to load your own data.")
st.warning("**Disclaimer:** ratings, review counts and sentiment are customer opinions. They are not medical "
           "advice and are not evidence that a product is safe or effective. Talk to a health professional.")
