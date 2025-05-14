# Credit Card Fraud Detection with Machine Learning

## Overview
This project develops a machine learning system to detect fraudulent credit card transactions, inspired by PayPal's fraud detection challenges. Using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (284,807 transactions, 0.172% fraud), it implements **XGBoost** for supervised learning and **Isolation Forest** for anomaly detection, addressing severe class imbalance with `scale_pos_weight`. The project includes Nature Publishing-quality visualizations (600 DPI, white background, no grid) and aligns with PayPal’s responsibilities: model development, supervised learning, anomaly detection, continual learning, collaboration, and advocacy.

## Objectives
- Detect fraud with high recall to minimize missed cases, critical for financial security.
- Visualize fraud patterns (e.g., transaction amounts, class imbalance) for stakeholder insights.
- Simulate continual learning by adapting to new fraud patterns.
- Share results via GitHub for collaboration and advocacy.

## Dataset
- **Source**: Kaggle ([mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud))
- **Size**: 284,807 transactions, 31 columns (`Time`, `V1-V28` (PCA features), `Amount`, `Class`)
- **Imbalance**: 0.172% fraud (492 cases), 99.828% non-fraud (284,315 cases)
- **Preprocessing**: Scaled `Time` and `Amount` using `StandardScaler`.

## Methods
- **Supervised Learning**: XGBoost with `scale_pos_weight` (~581) to handle imbalance, tuned for recall.
- **Anomaly Detection**: Isolation Forest (`contamination=0.00172`) for unsupervised fraud detection.
- **Hybrid Model**: Combines XGBoost and Isolation Forest to reduce false positives.
- **Evaluation**: AUC-ROC, precision, recall, F1-score, confusion matrix, ROC curve.
- **Visualizations**: Class imbalance bar plot, amount distribution ridge plot, anomaly scatter, score histogram.

## Deliverables
- **Notebook**: `Credit_Card_Fraud_Detection_ML_Analysis.ipynb` (data loading, EDA, modeling)
- **Visualizations**:
  - `class_imbalance_log.png`: Log-scale bar plot of fraud vs. non-fraud.
  - `amount_distribution_ridge.png`: Ridge plot of transaction amounts (0–120 USD).
  - `confusion_matrix.png`, `roc_curve_tuned.png`, `confusion_matrix_tuned.png`: XGBoost results.
  - `anomaly_scatter.png`, `anomaly_score_histogram.png`, `confusion_matrix_hybrid.png`: Anomaly detection.
- **Models**: `xgboost_fraud_model.pkl`, `xgboost_fraud_model_best.pkl`
## Setup
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection-ML.git
   cd Credit-Card-Fraud-Detection-ML
