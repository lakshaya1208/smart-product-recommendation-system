from pathlib import Path
import csv
import math
import random

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

categories = ["Electronics", "Books", "Home", "Fitness", "Stationery", "Accessories"]
category_price = {
    "Electronics": (650, 4200),
    "Books": (250, 1100),
    "Home": (400, 3500),
    "Fitness": (500, 2800),
    "Stationery": (80, 700),
    "Accessories": (250, 2200),
}

rows = []
num_customers = 180
num_products = 72

# Each customer has a small number of preferred categories.
customer_preferences = {}
for customer_id in range(1001, 1001 + num_customers):
    primary = random.choice(categories)
    secondary = random.choice([c for c in categories if c != primary])
    customer_preferences[customer_id] = {primary: 1.0, secondary: 0.55}

products = []
for i in range(num_products):
    category = categories[i % len(categories)]
    low, high = category_price[category]
    price = round(random.uniform(low, high), 2)
    products.append((f"P{i+1:03d}", category, price))

for customer_id in customer_preferences:
    previous_purchases = random.randint(0, 18)
    prefs = customer_preferences[customer_id]
    for product_id, category, price in products:
        affinity = prefs.get(category, 0.12)
        views = max(0, int(random.gauss(2.5 + 9.0 * affinity, 2.8)))
        cart_rate = 0.06 + 0.055 * affinity
        cart_additions = min(views, sum(1 for _ in range(views) if random.random() < cart_rate))
        cart_ratio = cart_additions / max(views, 1)
        price_penalty = math.log1p(price) / 12.5
        score = (
            -2.15
            + 0.12 * math.log1p(views)
            + 4.0 * cart_ratio
            + 0.035 * previous_purchases
            + 0.55 * affinity
            - 0.55 * price_penalty
        )
        purchase_probability = 1 / (1 + math.exp(-score))
        purchases = 1 if random.random() < purchase_probability else 0
        rows.append([
            customer_id,
            product_id,
            category,
            price,
            views,
            cart_additions,
            purchases,
            previous_purchases,
        ])

random.shuffle(rows)
out = DATA_DIR / "interactions.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "customer_id", "product_id", "product_category", "product_price",
        "views", "cart_additions", "purchases", "previous_purchases"
    ])
    writer.writerows(rows)

print(f"Generated {len(rows):,} rows at {out}")
print("Seed:", SEED)
