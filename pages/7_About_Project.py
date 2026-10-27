import streamlit as st

from src.config import PROJECT_TITLE
from src.ui import dataset_banner, fmt, get_dataset, page_header

page_header("ℹ️ About the Project", PROJECT_TITLE)
ds = get_dataset()
dataset_banner(ds)

st.subheader("Abstract")
st.write("HealthScope is a Python/Streamlit analytics application that cleans Amazon healthcare product data and "
         "customer reviews, then presents interactive charts, VADER sentiment analysis and automatically "
         "generated, evidence-based findings. It is an academic project and gives no medical advice.")
st.subheader("Introduction and problem statement")
st.write("Online marketplaces hold huge numbers of listings and reviews. Comparing them manually is hard, so a "
         "central dashboard that explores prices, ratings, categories, review volume and sentiment is useful.")
st.subheader("Objectives")
st.markdown("- Build a repeatable pipeline from raw files to cleaned data.\n- Provide interactive, filterable analytics.\n"
            "- Classify review sentiment with a documented baseline.\n- Generate findings that trace to calculations.\n"
            "- Deploy publicly on free infrastructure.")
st.subheader("Scope")
st.write("Product-level and review-level analysis of one or more uploaded/prepared datasets. Out of scope: "
         "medical claims, scraping Amazon, real-time data, paid APIs.")

st.subheader("Methodology and data-processing workflow")
st.markdown("1. Load CSV/Excel/JSON (`data_loader.py`)\n2. Map and validate columns (`validation.py`)\n"
            "3. Clean: price/rating/date parsing, invalid values, de-duplication, category and brand normalisation (`preprocessing.py`)\n"
            "4. Link reviews to products by ASIN\n5. Aggregate and filter (`analytics.py`)\n"
            "6. Score sentiment with VADER once per dataset, re-label cheaply when thresholds change (`sentiment_analysis.py`)\n"
            "7. Generate findings (`insights.py`) and exports (`export_utils.py`)")
st.subheader("Dataset sources")
st.write("The repository ships either a user-prepared sample in `data/` (from the UCSD Amazon Reviews datasets or "
         "another source whose licence permits use) or clearly labelled synthetic demo data. See `data/README.md` "
         "for sources, licence checks and how to prepare a sample.")
st.subheader("Technologies used")
st.write("Python, Streamlit, Pandas, NumPy, Plotly, NLTK (VADER), scikit-learn (TF-IDF, K-means, optional page), "
         "pytest, Python logging, Git/GitHub, Streamlit Community Cloud. PySpark is an **optional offline script** "
         "(`scripts/run_pyspark_analysis.py`); the deployed website does not use Spark, Hadoop, real-time streams or distributed computing.")

st.subheader("Analysis modules")
st.table({"Module": ["Dashboard", "Product Explorer", "Review Sentiment", "Insights", "Dataset Manager",
                     "K-means / TF-IDF / outliers"],
          "Status": ["Implemented", "Implemented", "Implemented (VADER)", "Implemented", "Implemented",
                     "Implemented, optional (Insights page)"]})

st.subheader("Relationship to the five Vs of Big Data (computed from the loaded data)")
p, r = ds.products, ds.reviews
mem = (p.memory_usage(deep=True).sum() + (r.memory_usage(deep=True).sum() if r is not None else 0)) / 1e6
rows = len(p) + (len(r) if r is not None else 0)
flagged = sum(sum(x.removed.values()) + sum(x.flagged.values()) for x in ds.reports)
st.markdown(f"""
- **Volume:** {len(p):,} products and {fmt(len(r) if r is not None else None, decimals=0)} reviews are loaded
  ({mem:.1f} MB in memory). {'This is small by big-data standards; the full UCSD datasets (millions of reviews) are the big-data scale, handled offline by the preparation / PySpark scripts.' if rows < 1_000_000 else 'This is large enough to benefit from the sampling and Spark scripts.'}
- **Variety:** structured fields (price, rating, counts), free text (reviews), dates and hierarchical categories in CSV/Excel/JSON.
- **Velocity:** data is processed in batches. There is no streaming or real-time component.
- **Veracity:** {flagged:,} records were removed, corrected or flagged during cleaning (see quality report); reviews may be biased or fake.
- **Value:** category, price, rating and sentiment patterns that help compare products, always with sample sizes and caveats.
""")
st.subheader("Expected outcomes")
st.write("A working public dashboard, a reproducible pipeline, downloadable cleaned data and reports, and findings that "
         "can be checked against the data.")
st.subheader("Limitations")
from src.insights import LIMITATIONS
for line in LIMITATIONS:
    st.write("- " + line)
st.subheader("Future enhancements")
st.write("Run the Spark pipeline on a cluster for full-size datasets, transformer-based sentiment models, aspect-based "
         "sentiment, review-helpfulness analysis, and scheduled data refreshes.")
st.warning("**Academic disclaimer:** built for a college mini-project. Results describe the loaded dataset only and "
           "are not medical, safety or purchasing advice.")
