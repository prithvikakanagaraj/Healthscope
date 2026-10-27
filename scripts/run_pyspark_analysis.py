"""OPTIONAL PySpark module: clean a large product CSV and compute aggregates in parallel.

Requires Java 8/11/17 and `pip install -r requirements-spark.txt`. The deployed Streamlit app
does NOT use Spark; this script is an offline extension for datasets too big for Pandas.

    python scripts/run_pyspark_analysis.py --input data/products_sample.csv --output outputs/spark
"""
import argparse
import sys

try:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
except ImportError:
    sys.exit("PySpark is not installed. Run: pip install -r requirements-spark.txt (and install Java).")

REQUIRED = ["asin", "title"]
NUMERIC = {"price": "double", "rating": "double", "review_count": "double"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="products CSV (canonical column names, see data/README.md)")
    ap.add_argument("--output", default="outputs/spark")
    a = ap.parse_args()

    spark = SparkSession.builder.appName("HealthScopeSpark").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.csv(a.input, header=True, inferSchema=False)

    # 1) schema validation
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        spark.stop()
        sys.exit(f"Missing required columns: {missing}")
    for c in NUMERIC:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))
    if "category" not in df.columns:
        df = df.withColumn("category", F.lit("Unknown"))

    # 2) cleaning
    loaded = df.count()
    df = df.withColumn("price", F.regexp_extract(F.regexp_replace("price", ",", ""), r"(-?\d+(\.\d+)?)", 1).cast("double"))
    df = df.withColumn("rating", F.col("rating").cast("double")).withColumn("review_count", F.col("review_count").cast("double"))
    df = df.withColumn("price", F.when(F.col("price") > 0, F.col("price")))
    df = df.withColumn("rating", F.when(F.col("rating").between(1, 5), F.col("rating")))
    df = df.withColumn("category", F.coalesce(F.trim("category"), F.lit("Unknown")))

    # 3) duplicates: keep the most-reviewed row per ASIN
    w = Window.partitionBy("asin").orderBy(F.col("review_count").desc_nulls_last())
    df = df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn").cache()
    retained = df.count()
    print(f"rows loaded={loaded:,} retained={retained:,} partitions={df.rdd.getNumPartitions()}")

    # 4) aggregates computed by Spark executors
    cat = (df.groupBy("category")
           .agg(F.count("*").alias("products"), F.round(F.avg("price"), 2).alias("avg_price"),
                F.round(F.avg("rating"), 2).alias("avg_rating"), F.sum("review_count").alias("total_review_count"))
           .orderBy(F.desc("products")))
    cat.show(20, truncate=False)
    df.select("price", "rating", "review_count").summary("count", "mean", "stddev", "min", "50%", "max").show()

    # 5) save
    df.write.mode("overwrite").option("header", True).csv(f"{a.output}/cleaned_products")
    cat.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{a.output}/category_summary")
    print(f"saved outputs to {a.output}")
    spark.stop()


if __name__ == "__main__":
    main()
