# Credit Card Fraud Detection

Detecting fraudulent credit card transactions using machine learning on a
real-world, **highly imbalanced** dataset (fraud makes up just 0.17% of all
transactions). This project focuses as much on handling that imbalance
correctly as it does on model choice — accuracy is a misleading metric here,
so the evaluation is built around precision, recall, F1, and ROC-AUC instead.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Goals](#project-goals)
- [Methodology](#methodology)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Tech Stack](#tech-stack)
- [Progress Log](#progress-log)
- [Future Work](#future-work)

## Overview

Credit card fraud detection is a classic **imbalanced classification**
problem: fraudulent transactions are rare, costly to miss, and easy for a
naive model to ignore entirely while still claiming 99.8% accuracy. This
project works through the problem properly — from EDA, to imbalance
handling, to choosing metrics that actually reflect performance on the
minority class, to comparing models on equal footing.

## Dataset

- Source: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions over two days, made by European cardholders
- 492 fraudulent transactions (**0.17%** of total) — severe class imbalance
- Features `V1`–`V28` are PCA-anonymized for confidentiality; `Time` and
  `Amount` are the only non-transformed features
- Target: `Class` (1 = fraud, 0 = legitimate)

## Project Goals

- [x] Exploratory Data Analysis (EDA)
- [x] Class imbalance analysis
- [ ] Understand fraud patterns (transaction amount, time-of-day trends, feature correlations)
- [ ] Handle class imbalance (SMOTE, class weighting, and/or undersampling — compared, not assumed)
- [ ] Train and compare multiple ML models
- [ ] Evaluate using Precision, Recall, F1-score, ROC-AUC, and PR-AUC
- [ ] Select and justify a final model based on business cost trade-offs
- [ ] Deploy the final model (API or simple web demo)

## Methodology

1. **EDA** — class distribution, transaction amount distributions for fraud
   vs. legitimate, correlation heatmap of `V1`–`V28`, time-of-day patterns.
2. **Preprocessing** — scale `Amount` and `Time` (the only unscaled
   features, since `V1`–`V28` are already PCA-transformed).
3. **Train/test split** — **stratified**, so the tiny fraud class is
   represented proportionally in both sets. Done *before* any resampling,
   so the test set stays a realistic, untouched sample of real-world data.
4. **Class imbalance handling** — compare at least two approaches rather
   than assuming one is best:
   - Class weighting (`class_weight='balanced'`)
   - SMOTE (synthetic oversampling of the minority class) — applied **only
     to the training set**, never the test set, to avoid leaking synthetic
     patterns into evaluation
   - Random undersampling of the majority class, as a baseline comparison
5. **Model training** — see [Models](#models).
6. **Threshold tuning** — the default 0.5 probability cutoff is rarely
   optimal for imbalanced problems. Use the precision-recall curve to pick
   a threshold that matches the actual cost trade-off (missing fraud vs.
   flagging legitimate transactions).

## Models

| Model | Why it's included |
|---|---|
| Logistic Regression | Simple, interpretable, fast baseline |
| Random Forest | Handles nonlinearity and feature interactions well |
| XGBoost | Strong performance on tabular, imbalanced data; supports `scale_pos_weight` |

## Evaluation

Accuracy is intentionally **not** the headline metric — a model that
predicts "not fraud" every single time scores 99.83% accuracy while
catching zero fraud. Instead:

- **Precision** — of transactions flagged as fraud, how many actually were?
- **Recall** — of actual fraud cases, how many did we catch?
- **F1-score** — harmonic mean of precision and recall
- **ROC-AUC** — overall ranking ability across thresholds
- **PR-AUC (average precision)** — often more informative than ROC-AUC on
  severely imbalanced data, since it focuses on minority-class performance
- **Confusion matrix** — to see the actual count of false negatives (missed
  fraud) vs. false positives (legitimate transactions flagged)

## Results

> To be filled in once models are trained — keep this table updated as you go.

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | – | – | – | – | – |
| Random Forest | – | – | – | – | – |
| XGBoost | – | – | – | – | – |

## Project Structure

```
credit-card-fraud-detection/
├── data/                   # raw and processed data (gitignored if large)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # saved trained models
├── requirements.txt
└── README.md
```

## Installation & Usage

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download the dataset from Kaggle and place `creditcard.csv` in `data/`,
then run the notebooks in order, or:

```bash
python src/train.py --model xgboost
python src/evaluate.py --model xgboost
```

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost

## Progress Log

**Day 1**
- Project and GitHub repository setup
- Initial dataset exploration
- Class imbalance analysis

**Day 2** *(update as you go)*
- [ ] ...

## Future Work

- Cost-sensitive evaluation: assign real dollar costs to false
  positives/negatives and optimize threshold against that, not just F1
- Compare anomaly-detection approaches (Isolation Forest, Autoencoders) as
  an alternative framing to supervised classification
- Model explainability with SHAP — which features drive fraud predictions?
- Deploy via a lightweight API (FastAPI/Flask) with a simple front end for
  live transaction scoring