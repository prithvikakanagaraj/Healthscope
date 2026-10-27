# Data folder

`products_sample.csv` and `reviews_sample.csv` (optional) are loaded automatically at start-up. If they are absent the
app uses synthetic demo data. Raw downloads go in `data/raw/` (git-ignored).

## Candidate sources - YOU must verify licence and availability
Availability, formats and licence terms can change, so check each page yourself before using or redistributing data
and note the result in your report. Do not assume a dataset is free for any use because the page is public.

1. **UCSD Amazon Reviews (J. McAuley lab)** - https://cseweb.ucsd.edu/~jmcauley/datasets.html
   (newer "Amazon Reviews 2023" is linked from that page). Gives product metadata and review files as JSON-lines
   per category (look for *Health and Personal Care* / *Health and Household*). Check the terms stated on the page
   (research / academic-use wording, citation requirements) and cite the papers it asks you to cite.
2. **AWS Marketplace - Amazon Health and Household product dataset** -
   https://aws.amazon.com/marketplace/pp/prodview-ce5kwbm5wuxq2 - check whether it is free, any subscription or
   fee, the delivery format, fields and the usage terms before relying on it.
3. Any other legitimate research dataset (e.g. Kaggle/Hugging Face) - read its licence.

Do **not** scrape Amazon directly; this project does not need it.

## Step by step (UCSD-style files)
1. Download a metadata file and a reviews file for the same category into `data/raw/` (smaller "5-core" review files are easier).
2. From the project root:
   `python scripts/prepare_dataset.py --meta data/raw/<meta> --reviews data/raw/<reviews> --products 800 --max-reviews-per-product 40`
3. Check `data/preprocessing_report.txt` and start the app. If the licence forbids redistribution, add the two CSVs
   to `.gitignore` before pushing.

## Column names
Mapping is flexible (see `PRODUCT_FIELDS` / `REVIEW_FIELDS` in `src/config.py`). Typical names that are recognised:
products - `asin/parent_asin, title, category/main_category/categories, brand/store, price, average_rating/rating,
rating_number/review_count, description, url`; reviews - `asin/parent_asin, reviewText/text, overall/rating,
reviewTime/timestamp, summary, reviewerID`. Products and reviews are joined on ASIN (both files must use the same ID
type; for the 2023 files `parent_asin` is preferred). Hierarchical categories use the second level
(`CATEGORY_LEVEL` in config). Missing rating/count columns in metadata are filled from the full review file by
`prepare_dataset.py`.
