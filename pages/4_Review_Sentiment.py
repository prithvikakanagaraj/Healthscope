import streamlit as st

from src import analytics as A
from src import sentiment_analysis as S
from src import visualizations as V
from src.config import MIN_GROUP_REVIEWS, SENTIMENT_COLORS
from src.export_utils import to_csv_bytes
from src.ui import (apply_all, cached_theme_counts, cached_top_terms, dataset_banner, get_dataset,
                    get_scored_reviews, get_thresholds, page_header, paginate, render_filters, show)

page_header("💬 Customer Review Sentiment", "NLTK VADER baseline: positive, neutral and negative reviews.")
ds = get_dataset()
dataset_banner(ds)
if ds.reviews is None or ds.reviews.empty:
    st.info("This page needs a **review-level dataset** (a file with review text). The current dataset has "
            "product-level data only, so no sentiment results are shown rather than inventing any.")
    st.page_link("pages/6_Dataset_Manager.py", label="Open Dataset Manager", icon="🗂️")
    st.stop()

with st.expander("Sentiment thresholds (VADER compound score)", expanded=False):
    p0, n0 = get_thresholds()
    c1, c2 = st.columns(2)
    pos = c1.number_input("Positive if score >=", 0.0, 1.0, float(p0), 0.01)
    neg = c2.number_input("Negative if score <=", -1.0, 0.0, float(n0), 0.01)
    st.session_state["pos_thr"], st.session_state["neg_thr"] = pos, neg
    st.caption("Defaults 0.05 / -0.05 are the usual VADER convention. Scores in between are Neutral.")

reviews_all = get_scored_reviews()
if reviews_all is None:
    st.stop()
f = render_filters(ds.products, reviews_all, "sent")
prods, revs = apply_all(ds, reviews_all, f)
if revs is None or revs.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

counts = S.sentiment_counts(revs)
m = st.columns(4)
m[0].metric("Reviews analysed", f"{len(revs):,}")
for col, row in zip(m[1:], counts.itertuples()):
    col.metric(row.sentiment, f"{row.reviews:,}", f"{row.percent}%", delta_color="off")
st.caption("VADER is a general-purpose lexicon. It can misread sarcasm, negation, context and medical terms, "
           "and sentiment is not evidence of clinical effectiveness.")
show(V.sentiment_bar(counts))

min_n = st.number_input("Minimum reviews per group (for category / brand / product views)", 1, 500,
                        min(MIN_GROUP_REVIEWS, 10))
tabs = st.tabs(["By category", "By brand", "By product", "Words & themes", "Examples", "Review table", "Trend"])

for tab, col, label in ((tabs[0], "category", "category"), (tabs[1], "brand", "brand"),
                        (tabs[2], "product_title", "product")):
    with tab:
        if col != "product_title" and not A.has_data(revs, col):
            st.info(f"No {label} information available.")
            continue
        summ = S.sentiment_by(revs[revs[col] != "Unknown"] if col != "product_title" else revs, col, int(min_n))
        if summ.empty:
            st.info(f"No {label} has at least {int(min_n)} reviews in the current selection.")
        else:
            show(V.sentiment_stacked(summ, col, f"Sentiment by {label} (top 20 by review count)"))
            st.dataframe(summ, use_container_width=True, hide_index=True)

with tabs[3]:
    pos_txt = revs.loc[revs["sentiment"] == "Positive", "review_text"]
    neg_txt = revs.loc[revs["sentiment"] == "Negative", "review_text"]
    c1, c2 = st.columns(2)
    for col, txt, name, color in ((c1, pos_txt, "positive", SENTIMENT_COLORS["Positive"]),
                                  (c2, neg_txt, "negative", SENTIMENT_COLORS["Negative"])):
        col.markdown(f"#### Themes in {name} reviews")
        if len(txt) < 5:
            col.info(f"Fewer than 5 {name} reviews.")
            continue
        t = cached_theme_counts(txt)
        t = t[t["reviews"] > 0]
        col.plotly_chart(V.terms_bar(t.head(8)[["theme", "reviews"]], f"Themes: {name} ({len(txt):,} reviews)", color),
                         use_container_width=True)
    st.caption("Themes are keyword matches (transparent lists in sentiment_analysis.py), not learned topics. "
               "'User-reported results' and 'Side effects mentioned' are what customers wrote, not verified facts.")
    st.markdown("#### Frequent words and phrases")
    terms = cached_top_terms(revs["review_text"], 20)
    if len(terms):
        show(V.terms_bar(terms, "Most frequent terms (number of reviews containing them)"))
    else:
        st.info("Not enough text for a frequency analysis.")

with tabs[4]:
    k = st.slider("Examples per side", 1, 10, 5)
    c1, c2 = st.columns(2)
    c1.markdown("#### Most positive")
    for r in revs.nlargest(k, "compound").itertuples():
        c1.success(f"{r.review_text[:500]}\n\n*{str(r.product_title)[:60]} - {r.review_rating} stars*")
    c2.markdown("#### Most negative")
    for r in revs.nsmallest(k, "compound").itertuples():
        c2.error(f"{r.review_text[:500]}\n\n*{str(r.product_title)[:60]} - {r.review_rating} stars*")
    st.caption("Review text is shown without reviewer identifiers. Respect the dataset licence when sharing.")

with tabs[5]:
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("Search review text")
    sel = c2.multiselect("Sentiment", S.SENTIMENT_ORDER, default=S.SENTIMENT_ORDER)
    view = revs[revs["sentiment"].isin(sel)]
    if q:
        view = view[view["review_text"].str.contains(q, case=False, regex=False)]
    cols = [c for c in ["review_date", "product_title", "category", "brand", "review_rating", "sentiment",
                        "compound", "review_text"] if c in view.columns]
    if view.empty:
        st.info("No reviews match.")
    else:
        st.dataframe(paginate(view, "rev", 10)[cols], use_container_width=True, hide_index=True)
    st.download_button("Download sentiment results (CSV)", to_csv_bytes(view[cols]),
                       "sentiment_results.csv", "text/csv")

with tabs[6]:
    if A.has_data(revs, "review_date"):
        mon = S.monthly_sentiment(revs)
        if len(mon) >= 3:
            show(V.sentiment_trend(mon))
        else:
            st.info("Fewer than 3 months of dated reviews.")
    else:
        st.info("No reliable review dates in this dataset.")
