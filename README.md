# Risk Alert Classifier

An early-warning machine-learning system that identifies **high-risk customers** of a digital banking platform, built for the *Red & White Skill Education – Supervised Learning* project.

**Live app:** `https://<your-app-name>.streamlit.app` &nbsp;|&nbsp; **Video walkthrough:** `<Google Drive / YouTube unlisted link>` &nbsp;|&nbsp; **Author:** `<Your Name>`

---

## Contents

1. [Overview](#1-overview)
2. [Repository structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Key findings and caveats](#6-key-findings-and-caveats)
7. [Installation and usage](#7-installation-and-usage)
8. [Streamlit app](#8-streamlit-app)
9. [Deployment](#9-deployment)
10. [Task coverage](#10-task-coverage)
11. [Submission checklist](#11-submission-checklist)
12. [Limitations and future work](#12-limitations-and-future-work)

---

## 1. Overview

A digital bank wants to flag customers who are likely to **default on payments or show fraudulent behaviour** (`risk_status = 1`). High-risk customers are a small minority (about 12 %), so the data is **highly imbalanced** and accuracy alone is misleading.

This project builds a leak-free classification pipeline that:

- identifies high-risk customers accurately,
- handles class imbalance (under-sampling, over-sampling, SMOTE, ADASYN),
- is evaluated with appropriate metrics (confusion matrix, precision, recall, F1, ROC-AUC),
- is improved with hyper-parameter tuning (`RandomizedSearchCV` then `GridSearchCV`),
- is delivered as a working Streamlit application.

**Business rule:** a missed high-risk customer (false negative, Type-II error) costs more than a false alarm (false positive, Type-I error, one extra manual review). The final model is therefore selected to **minimise false negatives**.

---

## 2. Repository structure

```
risk-alert-classifier/
├── README.md
├── Risk_Alert_Classifier.ipynb     # full solution, Parts A–H (outputs included)
├── app.py                          # Streamlit application
├── risk_utils.py                   # shared preprocessing pipeline and CV helpers
├── train_model.py                  # re-trains the final model, writes models/
├── requirements.txt
├── data/
│   └── Risk_Alert_Classifier_Dataset_4600.csv
├── models/
│   ├── risk_model.joblib           # fitted pipeline (preprocessing + model)
│   ├── model_meta.json             # metrics, threshold, ROC points, importances
│   └── model_comparison.csv        # test metrics of every model
├── screenshots/                    # all notebook charts
├── docs/
│   ├── Theory_Concepts.pdf         # theory definitions (Part A + glossary)
│   └── Video_Script.md             # talking points for the video
└── .streamlit/config.toml
```

---

## 3. Dataset

`data/Risk_Alert_Classifier_Dataset_4600.csv` contains **4,600 customers and 19 columns**.

| Column | Type | Description |
|---|---|---|
| `customer_id` | identifier | Dropped (no predictive meaning) |
| `age` | numeric | Age in years |
| `gender` | categorical | Male / Female / Other |
| `region` | categorical | North / South / East / West / Central |
| `employment_type` | categorical | Salaried / Self-Employed / Student / Unemployed / Retired |
| `annual_income_inr` | numeric | Annual income (INR) |
| `credit_score` | numeric | Credit score |
| `credit_utilization_ratio` | numeric | Share of available credit in use (0–1) |
| `missed_payments_12m` | numeric | Missed payments, last 12 months |
| `avg_late_payment_days` | numeric | Average days a payment is late |
| `monthly_transaction_count` | numeric | Transactions per month |
| `monthly_spend_inr` | numeric | Monthly spend (INR) |
| `cash_advance_count_6m` | numeric | Cash advances, last 6 months |
| `complaints_last_6m` | numeric | Complaints, last 6 months |
| `failed_login_attempts_3m` | numeric | Failed log-ins, last 3 months |
| `account_tenure_months` | numeric | Account age in months |
| `last_transaction_date` | date | Converted to `days_since_last_txn` |
| `debt_balance_inr` | numeric | Outstanding debt (INR) |
| **`risk_status`** | **target** | **0 = Low Risk, 1 = High Risk** |

Column descriptions are interpreted from the column names.

**Data quality**

- Missing values in 7 columns (about 2–5 % each): `credit_score` (216), `annual_income_inr` (166), `credit_utilization_ratio` (147), `employment_type` (144), `age` (140), `monthly_spend_inr` (129), `region` (102).
- Class balance: 4,043 low-risk (87.9 %) vs 557 high-risk (12.1 %), roughly 7 : 1.
- No duplicate rows or customer IDs.

**Engineered features**

- `days_since_last_txn`: days between 2025-12-01 and `last_transaction_date`.
- `debt_to_income_ratio`: `debt_balance_inr / annual_income_inr`.

---

## 4. Methodology

```
raw data → stratified 80/20 split → feature engineering → ColumnTransformer → model
                                                          ├─ numeric: StandardScaler → KNNImputer (k=5)
                                                          └─ categorical: most-frequent imputer → one-hot
```

| Stage | Approach | Rationale |
|---|---|---|
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)` | Same 12 % high-risk share in train and test |
| Imputation | Scale, then `KNNImputer(n_neighbors=5)`; categoricals use most-frequent | Multivariate imputation; KNN is distance-based so scaling is required |
| Leakage control | Pipeline fitted on the training split only; resampling applied to training data only | Test set mimics unseen data |
| Baseline | Logistic Regression | Interpretable, probability-based |
| Imbalance | Under-sampling, over-sampling, SMOTE, ADASYN, then retrain and compare recall, F1, AUC | Show and address the minority-class problem |
| Tree models | Decision Tree (depth-curve overfitting analysis), Random Forest | Non-linear models; bagging reduces variance |
| Tuning | `RandomizedSearchCV` (DT, RF), then `GridSearchCV` around the best RF; 5-fold stratified CV; F2 scoring; `class_weight` in the search space | F2 weights recall twice as much as precision |
| Selection | Fewest false negatives, then fewest false positives, then highest AUC, then highest F1; confirmed with 5-fold out-of-fold validation | Encodes the business requirement |
| Threshold | Maximise F2 on out-of-fold training predictions (never on the test set) | Recall-oriented operating point |
| Delivery | Pipeline and metadata exported with `joblib` / JSON | Used by the Streamlit app |

---

## 5. Results

Measured on the **920-customer hold-out test set** (111 high-risk, 809 low-risk) at the default threshold of 0.50.

| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression (baseline), selected** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 2 | Random Forest (tuned, RandomizedSearchCV) | 0.9989 | 0.9911 | 1.0000 | 0.9955 | 1.0000 | 0 | 1 |
| 3 | Random Forest (tuned, GridSearchCV) | 0.9989 | 0.9911 | 1.0000 | 0.9955 | 1.0000 | 0 | 1 |
| 4 | LR + Over-Sampling | 0.9978 | 0.9823 | 1.0000 | 0.9911 | 1.0000 | 0 | 2 |
| 5 | LR + SMOTE | 0.9978 | 0.9823 | 1.0000 | 0.9911 | 1.0000 | 0 | 2 |
| 6 | LR + ADASYN | 0.9978 | 0.9823 | 1.0000 | 0.9911 | 1.0000 | 0 | 2 |
| 7 | LR + Under-Sampling | 0.9967 | 0.9737 | 1.0000 | 0.9867 | 1.0000 | 0 | 3 |
| 8 | Random Forest (default) | 0.9978 | 1.0000 | 0.9820 | 0.9909 | 0.9999 | 2 | 0 |
| 9 | Decision Tree (default) | 0.9837 | 0.9286 | 0.9369 | 0.9327 | 0.9635 | 7 | 8 |
| 10 | Decision Tree (tuned, RandomizedSearchCV) | 0.9859 | 0.9623 | 0.9189 | 0.9401 | 0.9850 | 9 | 4 |

FN = false negatives (Type-II), FP = false positives (Type-I). Rows are ranked by the selection rule above.

**Cross-validation check (final model, training data, 5-fold out-of-fold):** recall 0.975, precision 0.989, ROC-AUC 0.999.

**Two operating points.** The table uses threshold 0.50. The app and exported metadata use the recall-oriented threshold **0.18**, chosen on out-of-fold data. On the test set it still catches every high-risk customer (0 FN) with 2 false alarms (precision 0.982, F1 0.991); on out-of-fold training predictions it lifts recall from 0.975 to 0.989.

### Selected figures

All charts are in [`screenshots/`](screenshots/).

| | |
|---|---|
| ![Target distribution](screenshots/02_target_distribution.png) | ![Accuracy paradox](screenshots/06_accuracy_paradox.png) |
| ![Baseline confusion matrix](screenshots/05_baseline_confusion_matrix.png) | ![Balancing comparison](screenshots/08_balancing_comparison.png) |
| ![Reduced-signal experiment](screenshots/09_reduced_signal_balancing.png) | ![Decision-tree overfitting](screenshots/10_decision_tree_overfitting.png) |
| ![ROC curves](screenshots/13_roc_curves_all_models.png) | ![Threshold analysis](screenshots/16_threshold_analysis.png) |

---

## 6. Key findings and caveats

1. **Accuracy is misleading.** A model that never raises an alert scores about 88 % accuracy with 0 % recall.
2. **The baseline is already very strong.** Credit-behaviour features (utilisation, late-payment days, missed payments, debt-to-income, credit score) separate the classes almost linearly.
3. **Balancing trades precision for recall.** With all features, recall is already at its maximum, so the sampling methods only add a few false alarms. In a demonstration experiment with the six strongest features removed, recall rose from about 0.86 to 0.95–0.97 while precision fell from about 0.86 to 0.64–0.68.
4. **Trees overfit, forests generalise.** The unrestricted tree reaches 100 % training accuracy but is clearly weaker on test data. Tuning shrinks the train–test gap; the tuned Random Forest is the best tree-based model (0 FN, 1 FP).
5. **AUC ties.** Many models reach AUC of about 1.000, so the final choice uses the business metric (false negatives).
6. **Do not over-interpret the perfect hold-out score.** The data is clean and strongly separable, the top models differ by one or two customers (within noise), and cross-validation shows recall near 0.975. Logistic Regression was preferred because it ties on the business metric and is simpler, faster and explainable. Validate on real data before production use.
7. **No leakage.** Imputation, scaling and resampling are fitted on training data only.
8. SMOTE and ADASYN operate on the one-hot-encoded matrix, so synthetic rows can contain fractional category values; `SMOTENC` is an alternative.

---

## 7. Installation and usage

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/risk-alert-classifier.git
cd risk-alert-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the notebook** (start Jupyter from the project root so relative paths resolve):

```bash
jupyter notebook Risk_Alert_Classifier.ipynb
```

Use *Kernel → Restart & Run All*. Runtime is a few minutes; the Random Forest searches are the slowest step. Running it regenerates `screenshots/` and `models/`. On Google Colab, upload the project folder and run `!pip install -r requirements.txt` first.

**Re-train only the final model:**

```bash
python train_model.py
```

**Start the app:**

```bash
streamlit run app.py
```

If the saved model cannot be loaded (for example, a different scikit-learn version), the app re-trains the final model from `data/` automatically on start-up.

---

## 8. Streamlit app

| Tab | Function |
|---|---|
| Single customer | Enter a profile; optionally mark fields as unknown (handled by the KNN imputer). Returns the risk probability, a HIGH RISK / WATCH / LOW RISK band and an alert message. |
| Batch scoring | Upload a CSV with the dataset's columns (missing values allowed) or score a bundled sample. Shows alert count and rate, a sortable table and a download button. If `risk_status` is present it also reports TP / FN / FP / TN. |
| Model performance | Hold-out metrics, confusion matrix, ROC curve, top risk drivers and the model-comparison table. |
| About | Pipeline summary and how to read the score. |

The sidebar **alert threshold slider** trades fewer missed risky customers (lower threshold) against fewer false alarms (higher threshold). Default: 0.18.

---

## 9. Deployment

**Streamlit Community Cloud**

1. Push the project to a public GitHub repository.
2. Open <https://share.streamlit.io>, sign in with GitHub and choose **New app**.
3. Select the repository and branch (`main`), set the main file to `app.py`, then deploy.
4. Paste the resulting URL at the top of this README.

**GitHub**

```bash
git init
git add .
git commit -m "Risk Alert Classifier: notebook, Streamlit app, docs"
git branch -M main
git remote add origin https://github.com/<your-username>/risk-alert-classifier.git
git push -u origin main
```

`requirements.txt` is installed automatically on Streamlit Cloud.

---

## 10. Task coverage

| Part | Tasks | Notebook section |
|---|---|---|
| A. Theory | 1–6 | Part A (also `docs/Theory_Concepts.pdf`) |
| B. Data preparation | 7 features and target; 8 stratified split; 9 missing values and KNN imputer | Part B |
| C. Baseline | 10 Logistic Regression; 11 confusion matrix and metrics; 12 Type-I / Type-II errors | Part C |
| D. Imbalance | 13 impact of imbalance; 14 under/over-sampling, SMOTE, ADASYN; 15 before/after comparison | Part D |
| E. Tree models | 16 Decision Tree; 17 overfitting; 18 Random Forest; 19 comparison | Part E |
| F. Tuning | 20 RandomizedSearchCV; 21 GridSearchCV; 22 tuned vs untuned | Part F |
| G. Evaluation | 23 ROC curves; 24 AUC comparison; 25 final model selection | Part G |
| H. Reporting | 26 final report; 27 deliverables | Part H and this README |

---

## 11. Submission checklist

| Component | Marks | Deliverable |
|---|---|---|
| A. Technical completion | 5 | Notebook (Restart & Run All clean) and live Streamlit URL |
| B. Recorded video | 3 | 5–10 min face + screen recording (OBS Studio), shared via Google Drive link or YouTube unlisted |
| C. GitHub repository | 2 | Public repo with `.ipynb`, screenshots, README and video link |

Before submitting:

- [ ] Run the notebook end to end in your own environment
- [ ] Push to a public GitHub repository
- [ ] Deploy the app and paste the URL above
- [ ] Record and upload the video, then paste the link above
- [ ] Replace the `<Your Name>` and `<your-username>` placeholders

---

## 12. Limitations and future work

- The dataset is clean and highly separable; real-world performance will likely be lower, so re-validate and monitor drift.
- It is a single time snapshot; with real data, use time-based validation to avoid look-ahead bias.
- Imbalance experiments use Logistic Regression only; extend to forests, gradient boosting and cost-sensitive learning.
- Add probability calibration, fairness checks across gender and region, and SHAP explanations.
- Add automated retraining and monitoring.

---

*Built with Python, scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn and Streamlit.*
