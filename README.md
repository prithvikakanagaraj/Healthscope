# 🩺 HealthScope - Amazon Healthcare Analytics Dashboard

College mini-project (Big Data Analysis): **Amazon Healthcare Products Analysis Using Big Data Analytics**.
A Streamlit web app that cleans product and review data, then shows interactive charts, VADER sentiment
analysis, product comparison and automatically generated, data-backed findings.
Ratings and reviews are customer opinions, **not medical advice**.

## What works out of the box
The app runs immediately with **clearly labelled synthetic demo data** (invented products, ASINs start with `DEMO`).
Replace it with real data by following `data/README.md` (or upload files on the *Dataset Manager* page).

Pages: Home, Analytics Dashboard, Product Explorer, Review Sentiment, Big Data Insights, Dataset Manager, About.

## 1. Run it on Windows 11 (VS Code)
1. Install **Python 3.11 or 3.12** from python.org (tick *Add python.exe to PATH*), and **VS Code**.
2. Open this folder in VS Code: *File > Open Folder > healthscope*. Open a terminal: *Terminal > New Terminal*.
3. Create and activate a virtual environment, install packages:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
   If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.
4. Start the app (a browser tab opens at http://localhost:8501):
```powershell
streamlit run app.py
```
5. Run the automated tests: `pytest`

Stop the app with `Ctrl+C`. The first sentiment run downloads the small VADER lexicon (needs internet once).

## 2. Use real data
Read `data/README.md` first (sources, licence checks, how to download, how to make a small sample):
```powershell
python scripts/prepare_dataset.py --meta data/raw/<meta file> --reviews data/raw/<reviews file> --products 800
```
This writes `data/products_sample.csv` and `data/reviews_sample.csv`; the app loads them automatically.

## 3. Deploy free on Streamlit Community Cloud
1. Install Git, create a GitHub account and an empty repository (e.g. `healthscope`).
2. In the project terminal:
```powershell
git init
git add .
git commit -m "HealthScope first version"
git branch -M main
git remote add origin https://github.com/<your-username>/healthscope.git
git push -u origin main
```
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository,
   branch `main`, main file `app.py`. In *Advanced settings* choose Python 3.11 or 3.12. Click **Deploy**.
4. Copy the public URL (`https://<name>.streamlit.app`) for your report.

**Before pushing real data:** only commit `data/products_sample.csv` / `reviews_sample.csv` if the dataset licence allows
redistribution, and keep them small (well under 25 MB). If unsure, deploy with the demo data (or a private repository)
and let users upload files. Never commit passwords, API keys or `.env` (they are git-ignored; this app needs none).


## 4. Android APK wrapper
The `android/HealthScopeAndroid/` folder contains the Android Studio WebView wrapper source. It opens the deployed HealthScope site, supports the web file picker for dataset uploads, and keeps analysis on the server.
Follow `android/README.md` after deployment. Replace the placeholder `app_url` with the real HTTPS Streamlit URL before building. This ZIP contains Android source, **not a precompiled APK**; the APK can only be built after the real URL is known and Android Studio completes a successful build.

## Project structure
```
app.py                  navigation + page config
pages/                  the 7 pages (UI only)
src/config.py           paths, thresholds, column synonyms
src/data_loader.py      safe CSV/Excel/JSON reading, default dataset
src/validation.py       column mapping, quality-report class, inspection
src/preprocessing.py    cleaning pipeline (products, reviews, linking)
src/analytics.py        filters, KPIs, aggregations, correlation, K-means
src/sentiment_analysis.py  VADER scoring, labels, themes, terms, TF-IDF
src/insights.py         generated findings (each with its calculation basis)
src/visualizations.py   Plotly charts      src/export_utils.py  CSV / MD / HTML exports
src/ui.py               Streamlit helpers (session data, filters, pagination)
src/demo_data.py        synthetic demo data (labelled)
scripts/prepare_dataset.py       large raw files -> reproducible sample
scripts/run_pyspark_analysis.py  OPTIONAL PySpark module (offline)
tests/                  pytest tests
```

## Optional PySpark module
Needs Java and `pip install -r requirements-spark.txt`:
`python scripts/run_pyspark_analysis.py --input data/products_sample.csv --output outputs/spark`.
It validates the schema, cleans fields, removes duplicate ASINs, computes category aggregates in parallel and saves outputs.
The deployed website does **not** use Spark; describe it as an offline extension in your report.

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` not recognised | Activate the venv (`venv\Scripts\activate`) and run `pip install -r requirements.txt` |
| `ModuleNotFoundError: src` | Run commands from the project root (the folder containing `app.py`) |
| VADER lexicon error | Connect to the internet once, or `python -c "import nltk; nltk.download('vader_lexicon')"` |
| Upload rejected | Max 50 MB, `.csv .xlsx .json .jsonl`; make a sample with `prepare_dataset.py` |
| Cloud build fails | Check the *Manage app* logs; keep `requirements.txt` as provided and pick Python 3.11/3.12 |
| Sentiment page empty | Needs a review file (review text); product-only data has no sentiment |

## Honest status notes
* Sentiment = NLTK VADER only (baseline). K-means, TF-IDF and outlier detection are optional tools on the Insights page.
* No Hadoop/real-time/distributed features are used by the website.
* Findings are generated from calculations at run time; nothing about the data is hard-coded.
