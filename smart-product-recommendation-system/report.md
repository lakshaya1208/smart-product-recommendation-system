# Technical Report: Smart Product Recommendation System

## 1. Introduction

This project implements a lightweight ML-based product recommendation system. It predicts the probability that a customer will purchase a product and uses those probabilities to rank products for personalized recommendations.

## 2. Problem Statement

Given customer-product interaction information such as views, cart additions, product category, price and purchase history, the system should identify products a customer is more likely to purchase.

## 3. Objective

The objective is to build a complete and explainable classification-and-ranking pipeline that can run as a small web application and be deployed on Vercel without unnecessary infrastructure.

## 4. Dataset

The included dataset is synthetic and reproducible with seed `42`. It contains 12,960 customer-product interactions involving 180 customers and 72 products.

The columns are customer ID, product ID, product category, product price, views, cart additions, purchases and previous purchases.

## 5. Data Preprocessing

The training pipeline removes exact duplicate rows. Numeric features are median-imputed and standardized. Product categories are one-hot encoded. The application uses the saved preprocessing parameters rather than retraining at request time.

## 6. Feature Engineering

The model uses product price, views, cart additions, previous purchases, cart-to-view ratio and customer purchase history within the product category. Category purchase history is calculated excluding the current row's purchase, preventing the target from leaking into the feature.

## 7. Model Selection

Logistic Regression was chosen because the target is binary and the project values explainability, simplicity and lightweight deployment. Balanced class weights are used because purchases are less common than non-purchases.

## 8. Training

The data is split into 80% training and 20% test data using stratification and random seed `42`. The model is trained with scikit-learn. After training, its intercept, coefficients, numeric scaling parameters and category vocabulary are exported to `models/model.json`.

## 9. Evaluation

The held-out test results from the implemented training pipeline are:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.681713 |
| Precision | 0.315483 |
| Recall | 0.533465 |
| F1-score | 0.396489 |
| ROC-AUC | 0.666845 |

These results are not intended to represent real-world performance because the dataset is synthetic.

## 10. Recommendation Logic

When a customer is selected, the server retrieves that customer's interaction history and identifies products not previously purchased. It constructs the model features for each candidate, applies the exported preprocessing parameters, calculates a purchase probability using the learned Logistic Regression parameters, sorts the candidates by probability and returns the highest five.

## 11. System Architecture

```text
CSV dataset
   |
   v
Python cleaning + feature engineering
   |
   v
Train/test split
   |
   v
Logistic Regression
   |
   +--> metrics.json
   +--> model.json
   +--> app_data.json

Static web app
   |
   v
Customer ID
   |
   v
Load saved model artifact
   |
   v
JavaScript model inference
   |
   v
Probability ranking
   |
   v
Top 5 results
```

## 12. User Interface

The interface is intentionally restrained: clear typography, simple borders, a compact customer selector, a recommendation table, model metrics and a short pipeline explanation. It avoids marketing content and decorative AI-style effects because the technical system is the focus of the submission.

## 13. Results

The application returns exactly five recommendations when at least five unpurchased candidate products exist. Each result comes from the trained model's calculated probability and includes Product ID, category, price and probability.

## 14. Limitations

The data is synthetic, so the model cannot establish real customer behavior. The model is also intentionally simple and has no online feedback loop. The application does not persist newly generated user behavior.

## 15. Future Improvements

Future versions could use a real interaction dataset, time-aware validation, stronger ranking evaluation, model comparison and persistent product/customer storage if those requirements become necessary.

## 16. Conclusion

The completed system demonstrates the full path from interaction data to preprocessing, feature engineering, model training, evaluation, probability-based ranking and a working web interface. The deployment architecture keeps Python training separate from lightweight JavaScript inference, allowing the final application to be deployed as a simple static Vercel project.
