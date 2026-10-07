# Smart Product Recommendation System

An ML-based product recommendation system for the NEXUS VIT Chennai technical recruitment submission. The system predicts whether a customer is likely to purchase a product and ranks available products by the predicted purchase probability.

## Objective

Build a small, understandable recommendation pipeline that genuinely uses a trained classification model rather than hard-coded or random recommendations.

## Features

- Reproducible synthetic customer-product interaction dataset.
- Data cleaning, duplicate removal and missing-value handling in the training pipeline.
- Explainable feature engineering.
- Balanced Logistic Regression classifier.
- Held-out evaluation with accuracy, precision, recall, F1-score and ROC-AUC.
- Server-side inference using exported model parameters.
- Customer selection and personalized Top 5 product recommendations.
- Vercel-compatible static web application with no database or external API.

## Dataset

The included dataset is **synthetic**, generated with random seed `42`. It contains 180 customers, 72 products and 12,960 customer-product interactions.

Columns:

| Column | Meaning |
| --- | --- |
| `customer_id` | Demo customer identifier |
| `product_id` | Demo product identifier |
| `product_category` | Product category |
| `product_price` | Product price in INR |
| `views` | Product views by the customer |
| `cart_additions` | Product cart additions |
| `purchases` | Purchase outcome for the interaction window |
| `previous_purchases` | Customer purchase count before the current window |

`purchases` is the target used to train the classifier and is not used as a direct model feature.

## ML approach

### Feature engineering

The model uses:

- Product price
- Views
- Cart additions
- Previous purchases
- Cart-to-view ratio
- Customer purchase history within the product category, calculated excluding the current row's purchase
- Product category encoded with one-hot encoding

The current purchase outcome is deliberately excluded from the features to avoid target leakage.

### Model selection

Logistic Regression was selected because this is a binary classification problem and the project prioritizes explainability and lightweight deployment. `class_weight="balanced"` is used because purchases are less frequent than non-purchases in the synthetic dataset.

Training is performed offline with scikit-learn. The final application does not need scikit-learn at runtime: the learned coefficients, scaling values and category vocabulary are exported to `models/model.json`, and the browser application performs the same logistic calculation directly.

## Evaluation

The model is evaluated on a held-out 20% test split using seed `42`.

| Metric | Result |
| --- | ---: |
| Accuracy | 68.1713% |
| Precision | 31.5483% |
| Recall | 53.3465% |
| F1-score | 0.396489 |
| ROC-AUC | 0.666845 |

These are actual results from the included training run, not fabricated values.

## Recommendation methodology

For a selected customer:

1. Retrieve the customer's interaction history.
2. Exclude products the customer has already purchased.
3. Build the same six numeric features used by the model for each available product.
4. Apply the saved training means/scales and category encoding.
5. Calculate the Logistic Regression purchase probability.
6. Sort products from highest to lowest probability.
7. Return the first five products.

The UI displays Product ID, category, price and the model's predicted probability.

## Architecture

```text
Synthetic CSV
    |
    v
Python training pipeline
    |
    +--> preprocessing + feature engineering
    |
    +--> Logistic Regression
    |
    +--> evaluation metrics
    |
    +--> models/model.json
    |
    +--> data/app_data.json

Static web app
    |
    v
Customer selection
    |
    v
Model inference
    |
    v
Probability ranking
    |
    v
Top 5 recommendations
```

## Project structure

```text
smart-product-recommendation-system/
├── app/
│   ├── api/recommend/route.ts
│   ├── components/recommendation-panel.tsx
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── data/
│   ├── interactions.csv
│   ├── app_data.json
│   └── README.md
├── lib/
│   └── recommender.ts
├── models/
│   ├── model.json
│   └── metrics.json
├── public/
│   └── favicon.svg
├── scripts/
│   ├── generate_dataset.py
│   └── train_model.py
├── .gitignore
├── .vercelignore
├── eslint.config.mjs
├── next.config.ts
├── package.json
├── report.md
├── tsconfig.json
└── vercel.json
```

## Local setup

Requirements:

- Python 3.10+ for regenerating the dataset/model
- A simple local static web server for the browser application

The generated dataset and model artifacts are already included, so the web app can be started immediately.

## Train the model

To regenerate the synthetic dataset and model artifacts:

```bash
python scripts/generate_dataset.py
python scripts/train_model.py
```

The fixed seed keeps the dataset reproducible.

## Run the application

From the project root:

```bash
python -m http.server 3000
```

Open `http://localhost:3000`. Do not open `index.html` directly with `file://`; the browser must be able to fetch the JSON model/data artifacts.

The frontend is plain HTML/CSS/JavaScript intentionally. This removes a Node build dependency and makes the project a reliable static Vercel deployment while keeping the actual trained ML model and inference logic in the application.

## Vercel deployment

1. Push this project to GitHub.
2. Import the repository into Vercel.
3. No build command is required.
4. No database, environment variables, Docker or external API is required.
5. Deploy.

Vercel serves the static application and JSON model/data artifacts directly. The Python training scripts are development-time tooling and are not executed during deployment.

## Limitations

- The dataset is synthetic and therefore cannot represent real customer behavior.
- Recommendations depend on the interaction features present in the demo dataset.
- Logistic Regression captures relatively simple relationships.
- There is no online learning or feedback loop after deployment.
- The demo intentionally has no authentication or persistent user accounts.

## Future improvements

- Replace the synthetic dataset with a real, consented interaction dataset.
- Add time-aware validation for real-world recommendation behavior.
- Compare Logistic Regression with a tree-based model.
- Add offline ranking metrics such as Precision@K or Recall@K when suitable labeled recommendation data is available.
- Add a database only if persistent user or product data becomes a real requirement.
