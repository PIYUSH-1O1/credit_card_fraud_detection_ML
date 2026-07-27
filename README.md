# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-2C3E50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

An end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions on a **severely imbalanced real-world dataset** (0.173% fraud rate). 

This repository focuses on rigorous **class-imbalance handling**, business-aligned **cost optimization**, and evaluating models using **Precision, Recall, and PR-AUC** rather than misleading accuracy metrics.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset Summary](#-dataset-summary)
- [Master Project Roadmap](#-master-project-roadmap)
- [Phase 1: Exploratory Data Analysis Key Findings](#-phase-1-exploratory-data-analysis-key-findings)
- [Project Goals & Progress](#-project-goals--progress)
- [Models & Evaluation Strategy](#-models--evaluation-strategy)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Progress Log](#-progress-log)
- [License](#-license)

---

## 🔍 Overview

Credit card fraud detection is a classic **imbalanced classification** problem:
* Fraudulent transactions are rare, costly to miss, and easy for a naive model to ignore while still claiming **99.83% accuracy**.
* A baseline model predicting "Legitimate" for every transaction achieves 99.83% accuracy while catching **0% of fraud cases (0% Recall)**.

This project implements a complete, recruiter-grade machine learning workflow — from exploratory data analysis to feature engineering, imbalanced sampling (SMOTE & class weighting), threshold tuning, and production API deployment.

---

## 📊 Dataset Summary

* **Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Total Transactions:** 284,807 transactions over 48 hours (European cardholders)
* **Class Breakdown:** 
  * `Class 0` (Legitimate): **284,315** (99.827%)
  * `Class 1` (Fraudulent): **492** (**0.173%**)
* **Features:** 
  * `V1`–`V28`: PCA-anonymized numerical features (due to confidentiality)
  * `Time`: Seconds elapsed since the first transaction
  * `Amount`: Transaction dollar amount
  * `Class`: Target variable (`1` = Fraud, `0` = Legitimate)

---

## 🗺️ Master Project Roadmap

```
Phase 1: Exploratory Data Analysis (EDA) 🔍 [COMPLETED]
  ├── Class imbalance verification (99.83% vs 0.17%)
  ├── Financial retry & duplicate log analysis (Retained 1,081 duplicate records)
  ├── Behavioral distribution analysis (Nighttime fraud peak & micro-charge testing)
  └── Discriminative feature discovery (V17, V14, V12, V10, V11, V4)

Phase 2: Preprocessing & Data Pipeline Setup ⚙️ [NEXT]
  ├── Robust scaling for unscaled features (Amount & Time)
  └── Stratified Train/Test splitting (80/20 ratio) before resampling

Phase 3: Class Imbalance Handling ⚖️
  ├── Algorithmic Cost-Sensitive Weighting (scale_pos_weight / class_weight)
  ├── Synthetic Minority Oversampling (SMOTE on training split only)
  └── Majority Undersampling Baseline

Phase 4: Model Exploration & Benchmarking 📊
  ├── Baseline: Logistic Regression (Cost-sensitive)
  ├── Non-linear: Random Forest Classifier
  └── Gradient Boosting: XGBoost / LightGBM

Phase 5: Evaluation & Threshold Optimization 🎯
  ├── Evaluate using Precision, Recall, F1-Score, and PR-AUC
  └── Precision-Recall Curve threshold tuning for cost trade-offs

Phase 6: Explainability & Deployment 🚀
  ├── Model explainability with SHAP (SHapley Additive exPlanations)
  └── Lightweight REST API (FastAPI) for live transaction scoring
```

---

## 💡 Phase 1: Exploratory Data Analysis Key Findings

> [!IMPORTANT]
> **Key Analytical Takeaways from EDA:**

1. **Domain Decision on Duplicates:** Retained all **1,081 duplicate transactions** (including 19 fraud duplicates). In financial processing, duplicate log entries represent real-world events such as automated subscription retries, double-billing glitches, or rapid repeated card testing.
2. **Nighttime Exploitation Peak (Time Analysis):**
   * Legitimate transactions drop significantly between **1:00 AM and 5:00 AM** (due to normal sleep cycles).
   * Fraudulent transaction density **peaks sharply between 1:00 AM and 4:00 AM**, indicating fraudsters intentionally strike when cardholders are asleep and unable to review immediate bank alerts.
3. **Card Testing Micro-Charges (Amount Analysis):**
   * Over **14% of fraudulent transactions** are concentrated under **$50**, representing automated "card testing" micro-authorizations.
   * Fraudulent amounts cap around **$2,125**, whereas legitimate transactions include large enterprise outliers up to **$25,691**.
4. **Top Discriminative Features:**
   * **Strongest Negative Correlators:** `V17` (-0.326), `V14` (-0.302), `V12` (-0.261), `V10` (-0.217)
   * **Strongest Positive Correlators:** `V11` (+0.155), `V4` (+0.133)

---

## 🎯 Project Goals & Progress

- [x] **Exploratory Data Analysis (EDA)** — Class distribution, distributions of `Amount` & `Time`, and PCA correlation analysis
- [x] **Data Hygiene & Domain Rules** — Imbalance verification and duplicate log retaining logic
- [ ] **Feature Engineering & Preprocessing** — Scale `Amount` and `Time` using `RobustScaler`; Stratified 80/20 split
- [ ] **Class Imbalance Handling** — Compare Class-Weighting, SMOTE, and Random Undersampling
- [ ] **Model Exploration & Benchmarking** — Train Logistic Regression, Random Forest, and XGBoost
- [ ] **Threshold Tuning** — Optimize probability cutoffs on the Precision-Recall curve
- [ ] **Explainability & API Deployment** — SHAP feature attribution & FastAPI service

---

## 🧪 Models & Evaluation Strategy

### Models Included
| Model | Type | Objective |
|---|---|---|
| **Logistic Regression** | Baseline | Linear interpretability & cost-sensitive benchmark |
| **Random Forest** | Tree Ensemble | Captures non-linear feature interactions without feature scaling dependency |
| **XGBoost** | Gradient Boosting | State-of-the-art performance on imbalanced tabular data (`scale_pos_weight`) |

### Evaluation Metrics Focus
* **Recall (Sensitivity):** Percentage of actual frauds caught (minimizing False Negatives / missed fraud).
* **Precision:** Percentage of flagged transactions that are actually fraud (minimizing False Positives / user friction).
* **PR-AUC (Precision-Recall Area Under Curve):** Primary metric for imbalanced data evaluation.
* **Confusion Matrix:** Detailed accounting of False Negatives vs. False Positives.

---

## 📁 Project Structure

```
credit_card_fraud_detection_ML/
├── data/                   # Raw and processed datasets (gitignored)
├── notebooks/
│   └── cc_fraud.ipynb      # Main EDA, Preprocessing, and Modeling Notebook
├── models/                 # Saved model binaries (.pkl / .json)
├── src/                    # Modularized production scripts (Preprocessing, Train, Evaluate)
├── requirements.txt        # Project dependencies
├── implementation_plan.md  # Detailed technical roadmap & interview guide
└── README.md               # Project documentation
```

---

## 💻 Installation & Quickstart

```bash
# Clone repository
git clone https://github.com/PIYUSH-1O1/credit_card_fraud_detection_ML.git
cd credit_card_fraud_detection_ML

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Progress Log

* **Phase 1 (Completed):** 
  * Project environment & GitHub setup.
  * Complete Exploratory Data Analysis (EDA).
  * Imbalanced target analysis (99.827% vs 0.173%).
  * Visualized transaction `Amount` (log-scale boxplots & density plots) and `Time` 24-hour cycle distributions.
  * Identified top predictive PCA features (`V17`, `V14`, `V12`, `V10`, `V11`, `V4`).
* **Phase 2 (In Progress):**
  * Data preprocessing pipeline setup (`RobustScaler` & Stratified Train/Test split).

---

## 📄 License

This project is licensed under the **MIT License**.