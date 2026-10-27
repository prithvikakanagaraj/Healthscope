"""CSV / text / HTML exports built from the current data (no hardcoded values)."""
import html
from datetime import datetime

import pandas as pd

from .config import APP_NAME, PROJECT_TITLE

PRIVATE_COLUMNS = ["reviewer_id"]


def to_csv_bytes(df: pd.DataFrame, drop_private: bool = True) -> bytes:
    if drop_private:
        df = df.drop(columns=[c for c in PRIVATE_COLUMNS if c in df.columns])
    return df.to_csv(index=False).encode("utf-8")


def quality_report_text(reports) -> str:
    return "\n\n".join(r.to_text() for r in reports) or "No reports."


def describe_filters(f: dict) -> str:
    parts = []
    for key, label in [("search", "Search"), ("categories", "Categories"), ("brands", "Brands"),
                       ("price", "Price range"), ("min_rating", "Min rating"),
                       ("min_reviews", "Min review count"), ("date_range", "Review dates")]:
        if f.get(key):
            parts.append(f"{label}: {f[key]}")
    return "; ".join(parts) or "none"


def summary_markdown(ds, kpis: dict, filters: dict, insights, na, cat_summary) -> str:
    fmt = lambda v, p="": "n/a" if v is None or pd.isna(v) else f"{p}{v:,.2f}" if isinstance(v, float) else f"{v:,}"
    lines = [f"# {APP_NAME} analysis summary", f"_{PROJECT_TITLE}_", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"\n**Dataset:** {ds.source_label}", f"**Filters:** {describe_filters(filters)}", "\n## Key metrics",
             f"- Products: {fmt(kpis['products'])}", f"- Reviews: {fmt(kpis['reviews'])}",
             f"- Average rating: {fmt(kpis['avg_rating'])}", f"- Average price: {fmt(kpis['avg_price'], '$')}",
             f"- Categories: {fmt(kpis['categories'])}", f"- Brands: {fmt(kpis['brands'])}", "\n## Findings"]
    lines += [f"- **{i.title}:** {i.text} _(basis: {i.basis})_" for i in insights] or ["- none computed"]
    if na:
        lines += ["\n## Not available"] + [f"- {x}" for x in na]
    if cat_summary is not None and len(cat_summary):
        lines += ["\n## Category summary", cat_summary.to_markdown(index=False)
                  if _has_tabulate() else cat_summary.to_csv(index=False)]
    lines.append("\n_Ratings and sentiment are customer opinions, not medical advice._")
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def html_report(ds, kpis, filters, insights, na, figures, limitations, quality_lines) -> str:
    """Self-contained-ish HTML (Plotly JS loaded from CDN). figures = [(title, fig), ...]."""
    e = html.escape
    kp = "".join(f"<div class='k'><b>{e(str(v))}</b><span>{e(k)}</span></div>" for k, v in {
        "Products": kpis["products"], "Reviews": "n/a" if kpis["reviews"] is None else f"{kpis['reviews']:,}",
        "Avg rating": "n/a" if kpis["avg_rating"] is None else f"{kpis['avg_rating']:.2f}",
        "Avg price": "n/a" if kpis["avg_price"] is None else f"${kpis['avg_price']:.2f}",
        "Categories": kpis["categories"] if kpis["categories"] is not None else "n/a",
        "Brands": kpis["brands"] if kpis["brands"] is not None else "n/a"}.items())
    charts = "".join(f"<h3>{e(t)}</h3>" + f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                     for i, (t, f) in enumerate(figures))
    li = lambda xs: "<ul>" + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul>"
    ins = "<ul>" + "".join(f"<li><b>{e(i.title)}:</b> {e(i.text)} <i>({e(i.basis)})</i></li>" for i in insights) + "</ul>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{APP_NAME} report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:auto;padding:24px;color:#0B2545}}
h1,h2{{color:#0B2545}}.g{{display:flex;gap:12px;flex-wrap:wrap}}.k{{background:#F1F5F9;padding:12px 18px;border-radius:8px;
border-left:4px solid #13A89E}}.k b{{display:block;font-size:22px}}</style></head><body>
<h1>{APP_NAME} findings report</h1><p>{e(PROJECT_TITLE)}<br>Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<h2>Dataset</h2><p>{e(ds.source_label)}</p><p>Applied filters: {e(describe_filters(filters))}</p>
<h2>Key metrics</h2><div class='g'>{kp}</div><h2>Major findings</h2>{ins}
{('<h3>Not available</h3>' + li(na)) if na else ''}<h2>Charts</h2>{charts}
<h2>Data quality</h2>{li(quality_lines)}
<h2>Methodology</h2><p>Files are cleaned with Pandas (validation, type conversion, de-duplication). Sentiment uses
NLTK VADER compound scores with configurable thresholds. Statistics are computed on the loaded dataset only.</p>
<h2>Limitations</h2>{li(limitations)}
<p><i>Ratings and reviews are customer opinions, not medical advice.</i></p></body></html>"""
