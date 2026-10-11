# Churn Risk Advisor

A Streamlit web app that scores telecom customers by their risk of churning, so a retention team can decide who to call first.

**Live app:** TODO paste your Streamlit URL here

Project 1 (ML-Powered Customer Analytics), Introduction to Applied AI, BS 8th Semester.

## What the app does

- **One customer:** enter a customer profile in the sidebar and get a churn probability, a risk band (HIGH, WATCH or LOW) and a suggested action.
- **What-if:** shows how the probability changes if the contract, payment method or tech support changes.
- **Batch scoring:** upload a CSV with the original Telco columns, score every customer, and download the result.
- **About:** shows the model name, metrics and limitations.

## Screenshots

TODO add your two screenshots to the repo and fix the file names below.

![One customer](screenshot_one_customer.png)
![Batch scoring](screenshot_batch.png)

## Model card: Churn Risk Advisor v1.0

**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBClassifier (476 trees, max depth 2), chosen by 5-fold cross-validation over Logistic Regression and Random Forest. Features: 30 encoded columns.

**Performance:** CV AUC 0.850 +/- 0.012; test AUC 0.848 (test set used once). At a 0.5 threshold the test recall was 0.521 and precision 0.659.

**Threshold:** 0.20. With FN = PKR 6,000 and FP = PKR 1,000 the cost-based optimum is 1,000 / 7,000 = 0.14, so the threshold is set well below 0.5 to catch more churners; 0.20 is the nearest slider step at which long-tenure two-year customers still fall in the LOW band.

**Limitations:** one US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change. The lead of XGBoost over the other two models (0.850 vs 0.846) is smaller than one cross-validation standard deviation, so the three models are close to equal. Class imbalance was not corrected in training, so recall at 0.5 is low and the threshold matters.

**Fairness check (Week 15):** compare recall across gender and seniors.

**Owner and version:** TODO your name, v1.0, TODO date.

## Repository files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `churn_model.joblib` | The trained XGBoost model |
| `model_meta.json` | Feature columns, threshold, metrics, versions |
| `sample_customers.csv` | 50 customers for the batch demo |
| `requirements.txt` | Pinned library versions |
| `README.md` | This file |

## Run locally

Use Python 3.11 (or the version you deployed with).

```
git clone https://github.com/YOUR-USERNAME/churn-risk-advisor.git
cd churn-risk-advisor
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt`:

```
streamlit>=1.32
scikit-learn==1.6.1
xgboost==3.4.1
pandas
numpy
joblib
```

## Notebooks

- Week 1, EDA: TODO Kaggle link
- Week 2, baseline models: TODO Kaggle link
- Week 3, model optimization: TODO Kaggle link

## Written insights

### Part 1: training-serving skew

For one two-year-contract customer, P(churn) was TODO with the Week 3 encoder and TODO with the serving encoder.

They differ because `get_dummies(drop_first=True)` drops the first category it sees in each column. On a single row the only category present is the first one, so both Contract columns become 0 and the model reads the customer as month-to-month.

Parity test: largest difference TODO across 7,043 customers, so the app gives the same answers as the notebook.

From the what-if table, TODO (which change) lowered risk the most, from TODO to TODO. That does not mean the change would make the customer stay: customers who chose long contracts were already more loyal, so the model learned an association, not the effect of moving someone onto a contract.

### Part 2: testing the app

TODO one behavior of the app that surprised you.

TODO one input a real user might enter that the app still handles badly.

### Part 3: deployment

Public URL: TODO

TODO did the first deployment succeed? If not, quote the error line and say what fixed it.

## Check your understanding

1. **Why does `get_dummies(drop_first=True)` break predictions for a single customer?** With one row each text column has only one category, which is treated as the first and dropped. Every encoded column becomes 0, so the model sees the baseline customer (for example month-to-month) instead of the real one, and no error is raised.

2. **What does the parity test compare, and why every row?** It compares the probabilities from the training-time features with those from the serving encoder for the same customers. It must cover every row because a mismatch can sit in a rare category combination that a small sample would never contain.

3. **`st.cache_resource` or `st.cache_data` for the model?** `st.cache_resource`. A model is one shared object that should be loaded once per server and reused; `st.cache_data` is for data and makes a copy on every call.

4. **Why does the app derive TotalCharges?** TotalCharges is roughly tenure times monthly charges, so asking for it separately lets a user type a value that contradicts the other two. Deriving it keeps the input consistent with the kind of data the model was trained on.

5. **Why pin the exact library version?** A saved model stores the internal structure of the library version that created it. A different version on the server may fail to load it or load it and give different predictions. For this project that applies to xgboost as well as scikit-learn.

6. **Would a two-year contract make the customer stay?** Not necessarily. The drop from 0.79 to 0.27 is an association in past data: people who signed two-year contracts were different from those who did not. A company could find out with a randomized trial, offering the contract to a random half of high-risk customers and comparing churn with the other half.
