"""Build a small, reproducible sample (products + reviews) from large raw files.

Works with UCSD Amazon Reviews files (JSON-lines, optionally .gz) or CSV files, e.g.
    python scripts/prepare_dataset.py --meta data/raw/meta_Health.jsonl.gz \
        --reviews data/raw/Health_5.json.gz --products 800 --max-reviews-per-product 40

Output: data/products_sample.csv, data/reviews_sample.csv, data/preprocessing_report.txt
Only keep outputs in Git if the dataset licence allows redistribution (see data/README.md).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, PRODUCT_COLUMNS, REVIEW_COLUMNS  # noqa: E402
from src.preprocessing import clean_products, clean_reviews  # noqa: E402
from src.validation import auto_map_products, auto_map_reviews  # noqa: E402

CHUNK = 200_000


def chunks(path: Path):
    """Yield DataFrames from a CSV or JSON-lines file (gzip detected from the extension)."""
    if ".csv" in path.name:
        yield from pd.read_csv(path, chunksize=CHUNK, low_memory=False)
    else:
        yield from pd.read_json(path, lines=True, chunksize=CHUNK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, type=Path, help="product metadata file")
    ap.add_argument("--reviews", required=True, type=Path, help="review file")
    ap.add_argument("--products", type=int, default=800, help="number of products in the sample")
    ap.add_argument("--min-reviews", type=int, default=5, help="only sample products with at least this many reviews")
    ap.add_argument("--max-reviews-per-product", type=int, default=40)
    ap.add_argument("--category-keyword", default=None, help="keep only categories containing this text")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    a = ap.parse_args()

    # Pass 1: products (all rows, cleaned)
    parts, pmap = [], None
    for ch in chunks(a.meta):
        pmap = pmap or auto_map_products(ch.columns)
        parts.append(ch[[c for c in pmap.values() if c]])
    raw_products = pd.concat(parts, ignore_index=True)
    products, prep = clean_products(raw_products, pmap)
    if a.category_keyword:
        products = products[products["category"].str.contains(a.category_keyword, case=False, na=False)]
    print(f"products cleaned: {len(products):,}")

    # Pass 2: review counts and rating sums for ALL reviews (before sampling) so statistics stay correct
    rmap, tallies = None, []
    for ch in chunks(a.reviews):
        rmap = rmap or auto_map_reviews(ch.columns)
        t = pd.DataFrame({"asin": ch[rmap["asin"]].astype(str).str.upper(),
                          "r": pd.to_numeric(ch[rmap["review_rating"]], errors="coerce")})
        tallies.append(t.groupby("asin")["r"].agg(["count", "sum"]))
    tally = pd.concat(tallies).groupby(level=0).sum()
    tally["avg"] = tally["sum"] / tally["count"]

    eligible = products[products["asin"].map(tally["count"]).fillna(0) >= a.min_reviews]
    sample = eligible.sample(min(a.products, len(eligible)), random_state=a.seed).copy()
    # Fill statistics that the metadata does not contain, using the full review file
    if sample["review_count"].isna().all():
        sample["review_count"] = sample["asin"].map(tally["count"])
    if sample["rating"].isna().all():
        sample["rating"] = sample["asin"].map(tally["avg"]).round(2)
    keep = set(sample["asin"])
    print(f"sampled products: {len(sample):,}")

    # Pass 3: collect reviews for the sampled products, capped per product
    got = []
    for ch in chunks(a.reviews):
        ch = ch.copy()
        ch["__asin"] = ch[rmap["asin"]].astype(str).str.upper()
        got.append(ch[ch["__asin"].isin(keep)].drop(columns="__asin"))
    reviews, rrep = clean_reviews(pd.concat(got, ignore_index=True), rmap)
    # shuffle (reproducibly) then keep the first N reviews of each product
    reviews = reviews.sample(frac=1, random_state=a.seed).groupby("asin").head(a.max_reviews_per_product)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    sample[PRODUCT_COLUMNS].to_csv(a.out_dir / "products_sample.csv", index=False)
    reviews[REVIEW_COLUMNS].drop(columns=["reviewer_id"]).to_csv(a.out_dir / "reviews_sample.csv", index=False)
    (a.out_dir / "preprocessing_report.txt").write_text(
        prep.to_text() + "\n\n" + rrep.to_text() +
        f"\n\nSample: {len(sample)} products, {len(reviews)} reviews (seed={a.seed}, "
        f"max {a.max_reviews_per_product} reviews/product)\n", encoding="utf-8")
    print(f"saved {len(sample):,} products and {len(reviews):,} reviews to {a.out_dir}")


if __name__ == "__main__":
    main()
