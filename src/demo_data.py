"""Synthetic DEMO data so the app runs before a real dataset is added.

Everything generated here is invented (fictional brands, ASINs start with 'DEMO').
It must never be presented as genuine Amazon data; the UI labels it clearly.
"""
import numpy as np
import pandas as pd

CATEGORIES = {  # name: (typical price, product types)
    "Vitamins & Supplements": (18, ["Vitamin D3", "Omega-3", "Multivitamin", "Magnesium", "Probiotic"]),
    "Pain Relief": (12, ["Pain Relief Gel", "Heat Patch", "Muscle Rub", "Joint Support"]),
    "First Aid": (9, ["Bandage Set", "Antiseptic Wipes", "First Aid Kit", "Gauze Pads"]),
    "Medical Devices": (35, ["Digital Thermometer", "Blood Pressure Monitor", "Pulse Oximeter"]),
    "Sleep & Relaxation": (15, ["Melatonin", "Sleep Mask", "Herbal Tea", "Calming Gummies"]),
    "Digestive Health": (16, ["Fiber Powder", "Digestive Enzymes", "Antacid Chews"]),
    "Oral Care": (8, ["Whitening Strips", "Electric Toothbrush Head", "Mouthwash"]),
}
BRANDS = ["AlphaCare", "NovaWell", "PureLeaf", "VitaNorth", "MediCore", "ZenHealth", "BrightLife", "OakRoot"]
POS = ["Great product, it works well and I am very happy with it.",
       "Excellent quality and good value for the price, I love it.",
       "Easy to use and effective for me, highly recommend.",
       "Nice taste and pleasant smell, the packaging was perfect.",
       "Fast shipping and the size is just right, wonderful purchase."]
NEU = ["It is okay, nothing special.", "Average product, arrived as described.",
       "The size is fine and the price is acceptable."]
NEG = ["Terrible taste and awful smell, I hate it.", "Poor quality, it broke after a week and was a waste of money.",
       "The packaging was damaged and shipping was slow, very disappointed.",
       "Did not work for me, too expensive and the size was wrong.", "Bad experience, cheap quality and hard to use."]


def generate_demo(seed: int = 42, n_products: int = 240):
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    prod_rows, rev_rows = [], []
    for i in range(n_products):
        cat = cats[rng.integers(len(cats))]
        base, types = CATEGORIES[cat]
        brand = BRANDS[rng.integers(len(BRANDS))]
        asin = f"DEMO{i:06d}"
        quality = float(np.clip(rng.normal(4.1, 0.5), 2.0, 5.0))
        price = round(float(base * rng.lognormal(0, 0.45)), 2)
        total_reviews = int(rng.lognormal(4.5, 1.1)) + 3
        prod_rows.append({"asin": asin, "title": f"{brand} {types[rng.integers(len(types))]} (demo {i})",
                          "category": cat, "brand": brand, "price": price,
                          "rating": round(quality, 1), "review_count": total_reviews,
                          "description": "Synthetic demo product.", "url": None})
        for _ in range(int(rng.integers(6, 30))):
            star = int(np.clip(round(rng.normal(quality, 1.0)), 1, 5))
            pool = POS if star >= 4 else NEU if star == 3 else NEG
            when = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 6)))
            rev_rows.append({"asin": asin, "review_text": pool[rng.integers(len(pool))],
                             "review_rating": star, "review_date": when.strftime("%Y-%m-%d"),
                             "review_title": None, "reviewer_id": f"DEMOUSER{rng.integers(10**6)}"})
    return pd.DataFrame(prod_rows), pd.DataFrame(rev_rows)
