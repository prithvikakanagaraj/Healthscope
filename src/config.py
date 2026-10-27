"""Central configuration: paths, thresholds, column synonyms, colours."""
import logging
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRODUCTS_SAMPLE = DATA_DIR / "products_sample.csv"
REVIEWS_SAMPLE = DATA_DIR / "reviews_sample.csv"

APP_NAME = "HealthScope"
TAGLINE = "Amazon Healthcare Analytics Dashboard"
PROJECT_TITLE = "Amazon Healthcare Products Analysis Using Big Data Analytics"

# Upload / size limits (keep in sync with .streamlit/config.toml -> maxUploadSize)
MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = ("csv", "xlsx", "json", "jsonl")
MAX_REVIEWS_IN_SESSION = 100_000  # larger review files are sampled (seed fixed)

# Sentiment thresholds on the VADER compound score (-1..+1)
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

# Ranking safeguards
MIN_BRAND_PRODUCTS = 3
MIN_GROUP_REVIEWS = 20
HIGH_RATING_CUTOFF = 4.3
CATEGORY_LEVEL = 1  # for hierarchical categories "A > B > C" use level B

NAVY, TEAL = "#0B2545", "#13A89E"
SENTIMENT_COLORS = {"Positive": "#13A89E", "Neutral": "#9AA5B1", "Negative": "#D1495B"}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

# Canonical field -> accepted source column names (normalised: lowercase, a-z0-9 and "_").
# Order matters: the first match wins.
PRODUCT_FIELDS = {
    "asin": ["parent_asin", "asin", "product_id", "productid", "sku", "id"],
    "title": ["title", "product_name", "name", "product_title", "item_name"],
    "category": ["category", "main_category", "product_category", "category_name", "categories"],
    "brand": ["brand", "store", "manufacturer", "brand_name"],
    "price": ["price", "price_usd", "selling_price", "final_price", "list_price"],
    "rating": ["average_rating", "avg_rating", "product_rating", "rating", "stars", "overall_rating"],
    "review_count": ["review_count", "rating_number", "reviews_count", "num_reviews",
                     "number_of_reviews", "ratings_count", "total_reviews"],
    "description": ["description", "product_description", "about"],
    "url": ["url", "product_url", "link"],
}
REVIEW_FIELDS = {
    "asin": ["parent_asin", "asin", "product_id", "productid"],
    "review_text": ["reviewtext", "review_text", "text", "review", "content", "review_body", "body"],
    "review_rating": ["overall", "review_rating", "rating", "stars", "score"],
    "review_date": ["reviewtime", "review_date", "date", "timestamp", "unixreviewtime", "review_time"],
    "review_title": ["summary", "review_title", "headline"],
    "reviewer_id": ["reviewerid", "reviewer_id", "user_id", "customer_id"],
}
PRODUCT_COLUMNS = list(PRODUCT_FIELDS)
REVIEW_COLUMNS = list(REVIEW_FIELDS)

UNKNOWN = "Unknown"


def normalise_name(name) -> str:
    """'Review Text ' -> 'review_text'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)
