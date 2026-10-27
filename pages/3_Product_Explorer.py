import streamlit as st

from src import analytics as A
from src.export_utils import to_csv_bytes
from src.ui import (apply_all, dataset_banner, fmt, get_scored_reviews, page_header, paginate,
                    render_filters, require_products)

page_header("🔎 Product Explorer", "Search, filter, sort and compare products.")
ds = require_products()
dataset_banner(ds)
f = render_filters(ds.products, None, "exp", with_search=True)
prods = A.apply_product_filters(ds.products, f)

SORTS = {"Review count": "review_count", "Rating": "rating", "Price": "price", "Product name": "title"}
c1, c2 = st.columns(2)
sort_label = c1.selectbox("Sort by", list(SORTS))
descending = c2.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Descending"
if sort_label != "Product name" and not A.has_data(prods, SORTS[sort_label]):
    st.info(f"'{sort_label}' is not available in this dataset; sorting by name instead.")
    sort_col = "title"
else:
    sort_col = SORTS[sort_label]
prods = prods.sort_values(sort_col, ascending=not descending, na_position="last")

st.write(f"**{len(prods):,}** matching products")
if prods.empty:
    st.warning("No products match your search and filters.")
    st.stop()

cols = [c for c in ["title", "category", "brand", "price", "rating", "review_count"] if A.has_data(prods, c) or c == "title"]
config = {"price": st.column_config.NumberColumn("Price", format="$%.2f"),
          "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
          "review_count": st.column_config.NumberColumn("Reviews", format="%d")}
if A.has_data(prods, "url"):
    cols.append("url")
    config["url"] = st.column_config.LinkColumn("Product link")
st.dataframe(paginate(prods, "exp")[cols], use_container_width=True, hide_index=True, column_config=config)
st.download_button("Download filtered results (CSV)", to_csv_bytes(prods), "filtered_products.csv", "text/csv")

st.subheader("Compare products side by side")
labels = (prods["title"].fillna("").str.slice(0, 70) + " | " + prods["asin"].fillna("no ID")).tolist()
by_label = dict(zip(labels, prods.index))
choice = st.multiselect("Choose 2 to 4 products from the filtered list", labels, max_selections=4)
if len(choice) >= 2:
    sel = prods.loc[[by_label[c] for c in choice]].copy()
    table = sel[[c for c in ["title", "asin", "category", "brand", "price", "rating", "review_count",
                             "description", "url"] if c in sel.columns]].copy()
    if "description" in table:
        table["description"] = table["description"].fillna("").str.slice(0, 200)
    reviews = get_scored_reviews() if ds.reviews is not None else None
    if reviews is not None:
        g = reviews[reviews["asin"].isin(sel["asin"])].groupby("asin")
        table["reviews_in_dataset"] = sel["asin"].map(g.size())
        table["% positive (VADER)"] = sel["asin"].map(g["sentiment"].apply(lambda s: round((s == "Positive").mean() * 100, 1)))
    table = table.dropna(axis=1, how="all")
    out = table.T.astype(str).replace({"None": "-", "nan": "-", "<NA>": "-"})
    out.columns = [f"Product {i + 1}" for i in range(out.shape[1])]
    st.dataframe(out, use_container_width=True)
    st.caption("This table shows data fields only. It does not rank products or claim any is 'best value'.")
elif choice:
    st.info("Select at least one more product to compare.")
