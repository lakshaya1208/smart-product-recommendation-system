from pathlib import Path
import csv
import json
import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interactions.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

with DATA.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    raw = list(reader)

# Remove exact duplicates.
seen = set()
rows = []
for r in raw:
    key = tuple(r.items())
    if key not in seen:
        seen.add(key)
        rows.append(r)

numeric_base = ["product_price", "views", "cart_additions", "previous_purchases", "cart_to_view", "category_purchase_history"]
category_col = "product_category"
feature_names = numeric_base + ["category"]

# Customer/category purchase history is computed excluding the current row's target.
category_total = {}
for r in rows:
    key = (r["customer_id"], r["product_category"])
    category_total[key] = category_total.get(key, 0) + int(r["purchases"])

X_rows = []
y = []
for r in rows:
    views = int(r["views"])
    carts = int(r["cart_additions"])
    purchases = int(r["purchases"])
    history = category_total[(r["customer_id"], r["product_category"])] - purchases
    X_rows.append({
        "product_price": float(r["product_price"]),
        "views": float(views),
        "cart_additions": float(carts),
        "previous_purchases": float(r["previous_purchases"]),
        "cart_to_view": carts / max(views, 1),
        "category_purchase_history": float(history),
        "product_category": r["product_category"],
    })
    y.append(purchases)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
category_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_base),
    ("category", category_pipeline, ["product_category"]),
])

X = pd.DataFrame(X_rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": round(float(accuracy_score(y_test, pred)), 6),
    "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6),
    "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6),
    "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6),
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 6),
}

fitted_pre = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
scaler = fitted_pre.named_transformers_["numeric"].named_steps["scaler"]
encoder = fitted_pre.named_transformers_["category"].named_steps["onehot"]

# Export the exact transformed-space parameters so inference can be performed in TypeScript.
coef = classifier.coef_[0].tolist()
intercept = float(classifier.intercept_[0])
category_values = encoder.categories_[0].tolist()
means = scaler.mean_.tolist()
scales = scaler.scale_.tolist()

artifact = {
    "model_type": "Logistic Regression",
    "random_seed": SEED,
    "numeric_features": numeric_base,
    "category_values": category_values,
    "numeric_means": means,
    "numeric_scales": scales,
    "coefficients": coef,
    "intercept": intercept,
    "metrics": metrics,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "positive_rate": round(float(sum(y) / len(y)), 6),
}

with (MODEL_DIR / "model.json").open("w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)

# Create a compact application data file. It includes only information needed by the recommendation UI.
customers = {}
products = {}
interactions = {}
for r in rows:
    cid = r["customer_id"]
    pid = r["product_id"]
    category = r["product_category"]
    products[pid] = {
        "productId": pid,
        "category": category,
        "price": float(r["product_price"]),
    }
    customers.setdefault(cid, {"customerId": cid, "previousPurchases": int(r["previous_purchases"])})
    interactions[f"{cid}:{pid}"] = {
        "views": int(r["views"]),
        "cartAdditions": int(r["cart_additions"]),
        "purchases": int(r["purchases"]),
    }

app_data = {
    "customers": list(customers.values()),
    "products": list(products.values()),
    "interactions": interactions,
    "metrics": metrics,
}
with (ROOT / "data" / "app_data.json").open("w", encoding="utf-8") as f:
    json.dump(app_data, f, separators=(",", ":"))

with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps({
    "rows_before_dedup": len(raw),
    "rows_after_dedup": len(rows),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "positive_rate": artifact["positive_rate"],
    "metrics": metrics,
}, indent=2))
