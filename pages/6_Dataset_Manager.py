import streamlit as st

from src import insights as I
from src.analytics import compute_kpis
from src.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, PRODUCT_FIELDS, REVIEW_FIELDS
from src.data_loader import DataLoadError, read_table
from src.export_utils import quality_report_text, summary_markdown, to_csv_bytes
from src.preprocessing import build_dataset
from src.ui import (dataset_banner, get_dataset, get_scored_reviews, page_header, reset_dataset,
                    set_dataset)
from src.validation import (auto_map_columns, inspect_dataframe, validate_product_mapping,
                            validate_review_mapping)

page_header("🗂️ Dataset Manager", "Upload, inspect, map, clean and download your data (kept only in your session).")
ds = get_dataset()
dataset_banner(ds)
st.info(f"Accepted: {', '.join('.' + e for e in ALLOWED_EXTENSIONS)} up to {MAX_UPLOAD_MB} MB per file. Files are only "
        "parsed as data (never executed) and are not saved to disk. For bigger data, build a sample with "
        "`python scripts/prepare_dataset.py` first (see README).")


def parse_upload(upload, slot):
    """Parse once per file and keep the DataFrame in this session only."""
    if upload is None:
        for k in (f"{slot}_sig", f"{slot}_df"):
            st.session_state.pop(k, None)
        return None
    sig = (upload.name, upload.size)
    if st.session_state.get(f"{slot}_sig") != sig:
        try:
            st.session_state[f"{slot}_df"] = read_table(upload.getvalue(), upload.name)
            st.session_state[f"{slot}_sig"] = sig
        except DataLoadError as exc:
            st.error(str(exc))
            st.session_state.pop(f"{slot}_df", None)
            st.session_state.pop(f"{slot}_sig", None)
            return None
    return st.session_state.get(f"{slot}_df")


def inspect_and_map(df, slot, fields, validator):
    info = inspect_dataframe(df)
    c = st.columns(4)
    c[0].metric("Rows", f"{info['rows']:,}")
    c[1].metric("Columns", info["columns"])
    c[2].metric("Duplicate rows", f"{info['duplicate_rows']:,}")
    c[3].metric("Missing cells", f"{int(info['missing']['missing'].sum()):,}")
    t1, t2, t3 = st.tabs(["Preview", "Data types", "Missing values"])
    t1.dataframe(df.head(20), use_container_width=True)
    t2.dataframe(info["dtypes"], use_container_width=True, hide_index=True)
    t3.dataframe(info["missing"], use_container_width=True, hide_index=True)
    st.markdown("**Column mapping** (auto-detected; change if wrong)")
    auto, mapping, cols = auto_map_columns(df.columns, fields), {}, list(df.columns)
    grid = st.columns(3)
    for i, field in enumerate(fields):
        idx = cols.index(auto[field]) + 1 if auto[field] in cols else 0
        pick = grid[i % 3].selectbox(field, ["(none)"] + cols, index=idx, key=f"{slot}_map_{field}")
        mapping[field] = None if pick == "(none)" else pick
    errors, warnings = validator(mapping)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    return mapping, errors


st.subheader("1. Upload")
c1, c2 = st.columns(2)
prod_up = c1.file_uploader("Products file (price, rating, category ...)", type=list(ALLOWED_EXTENSIONS), key="up_prod")
rev_up = c2.file_uploader("Reviews file (optional, review text ...)", type=list(ALLOWED_EXTENSIONS), key="up_rev")
prod_df, rev_df = parse_upload(prod_up, "prod"), parse_upload(rev_up, "rev")

p_map = r_map = None
errs = []
if prod_df is not None:
    st.subheader("2. Inspect and map products")
    p_map, e = inspect_and_map(prod_df, "prod", PRODUCT_FIELDS, validate_product_mapping)
    errs += e
if rev_df is not None:
    st.subheader("3. Inspect and map reviews")
    r_map, e = inspect_and_map(rev_df, "rev", REVIEW_FIELDS, validate_review_mapping)
    errs += e

if prod_df is not None or rev_df is not None:
    st.subheader("4. Clean and use")
    dedupe = st.checkbox("Remove duplicates (same ASIN for products; exact repeats for reviews)", value=True)
    if st.button("Process and use this dataset", type="primary", disabled=bool(errs)):
        try:
            with st.status("Cleaning data...", expanded=True) as status:
                new = build_dataset(prod_df if prod_df is not None else None, p_map or {},
                                    rev_df, r_map, remove_duplicates=dedupe,
                                    source_label="Uploaded: " + ", ".join(u.name for u in (prod_up, rev_up) if u))
                status.update(label="Dataset ready", state="complete")
            set_dataset(new)
            st.success("Dataset loaded. All analysis pages now use it (this session only).")
            st.rerun()
        except Exception as exc:  # friendly message, details only in the server log
            from src.config import get_logger
            get_logger(__name__).exception("Processing failed: %s", exc)
            st.error("The data could not be processed. Check the column mapping and file contents.")
    if st.button("Remove uploaded files from this session"):
        for k in ("prod_sig", "prod_df", "rev_sig", "rev_df"):
            st.session_state.pop(k, None)
        reset_dataset()
        st.rerun()

st.subheader("Current dataset: quality report and downloads")
ds = get_dataset()
for r in ds.reports:
    with st.expander(f"Quality report: {r.kind} ({r.rows_retained:,} of {r.rows_loaded:,} rows retained)", expanded=True):
        st.text(r.to_text())
t1, t2 = st.tabs(["Cleaned products", "Cleaned reviews"])
t1.dataframe(ds.products.head(50), use_container_width=True)
if ds.reviews is not None:
    t2.dataframe(ds.reviews.drop(columns=["reviewer_id"], errors="ignore").head(50), use_container_width=True)
else:
    t2.info("No review data loaded.")
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Cleaned products (CSV)", to_csv_bytes(ds.products), "cleaned_products.csv", "text/csv")
if ds.reviews is not None:
    d2.download_button("Cleaned reviews (CSV)", to_csv_bytes(ds.reviews), "cleaned_reviews.csv", "text/csv")
d3.download_button("Quality report (TXT)", quality_report_text(ds.reports).encode("utf-8"), "quality_report.txt")
if d4.button("Prepare analysis summary"):
    revs = get_scored_reviews() if ds.reviews is not None else None
    ins, na = I.generate_insights(ds.products, revs)
    st.session_state["summary_md"] = summary_markdown(ds, compute_kpis(ds.products, revs), {}, ins, na, None)
if "summary_md" in st.session_state:
    d4.download_button("Download analysis summary (MD)", st.session_state["summary_md"].encode("utf-8"),
                       "analysis_summary.md")
if st.button("Reset to the default dataset"):
    reset_dataset()
    st.rerun()
