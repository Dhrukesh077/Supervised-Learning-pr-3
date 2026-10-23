"""
Risk Alert Classifier - Streamlit app
=====================================
Early-warning tool for a digital banking platform: scores a customer (or a CSV
of customers) and raises an alert when P(high risk) exceeds an adjustable threshold.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import risk_utils as ru

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "risk_model.joblib")
META_PATH = os.path.join(HERE, "models", "model_meta.json")

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🚨", layout="wide")


# --------------------------------------------------------------------------- #
# Model loading (re-trains automatically if the saved model is missing/incompatible)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model ...")
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as fh:
            meta = json.load(fh)
        return model, meta
    except Exception:
        import train_model  # local import: only needed for the fallback

        with st.spinner("Saved model not usable here - training a fresh one (takes a few seconds) ..."):
            model, meta = train_model.train_and_save(verbose=False)
        return model, meta


model, meta = load_artifacts()
DEFAULT_THRESHOLD = float(meta["threshold"])
M = meta["metrics"]


def score(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df)[:, 1]


def risk_band(p: float, thr: float) -> tuple[str, str]:
    if p >= thr:
        return "HIGH RISK", "#c0392b"
    if p >= thr * 0.5:
        return "WATCH", "#e67e22"
    return "LOW RISK", "#27ae60"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🚨 Risk Alert Classifier")
    st.caption("Early-warning system for high-risk banking customers")
    st.markdown("---")
    threshold = st.slider(
        "Alert threshold  P(high risk) ≥",
        min_value=0.05, max_value=0.95, value=float(round(DEFAULT_THRESHOLD, 2)), step=0.01,
        help="Lower threshold = fewer missed risky customers (fewer false negatives) "
             "but more false alarms. The default was chosen to maximise F2 (recall-weighted).",
    )
    st.markdown(f"**Model:** {meta['model_name']}")
    st.markdown(f"**Trained:** {meta['trained_at_utc']} UTC")
    st.markdown(f"**Hold-out recall:** {M['recall']:.3f}")
    st.markdown(f"**Hold-out ROC-AUC:** {M['roc_auc']:.3f}")
    st.markdown("---")
    st.caption("Business rule: a missed high-risk customer (false negative) costs more than an "
               "extra manual review (false positive).")

tab_single, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔍 Single customer", "📂 Batch scoring (CSV)", "📊 Model performance", "ℹ️ About"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - single customer
# --------------------------------------------------------------------------- #
with tab_single:
    st.subheader("Score one customer")
    st.write("Fill in the customer profile and press **Predict**. Fields you tick as *unknown* are "
             "treated as missing and filled by the model's KNN imputer.")
    ni = meta["numeric_info"]
    opts = meta["options"]

    def num(label, col, step=1.0, integer=False, key=None):
        info = ni[col]
        lo, hi, med = info["min"], info["max"], info["median"]
        if integer:
            return st.number_input(label, int(lo), int(hi * 1.5), int(round(med)), step=1, key=key or col)
        return st.number_input(label, float(lo), float(hi * 1.5), float(round(med, 3)), step=step, key=key or col)

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Demographics**")
            age = num("Age", "age", integer=True)
            gender = st.selectbox("Gender", opts["gender"])
            region = st.selectbox("Region", opts["region"])
            employment = st.selectbox("Employment type", opts["employment_type"])
            income = num("Annual income (INR)", "annual_income_inr", step=1000.0)
        with c2:
            st.markdown("**💳 Credit behaviour**")
            credit_score = num("Credit score", "credit_score", integer=True)
            util = st.slider("Credit utilisation ratio", 0.0, 1.0, float(round(ni["credit_utilization_ratio"]["median"], 2)), 0.01)
            missed = num("Missed payments (12 m)", "missed_payments_12m", integer=True)
            late_days = num("Avg. late-payment days", "avg_late_payment_days", step=0.5)
            debt = num("Debt balance (INR)", "debt_balance_inr", step=1000.0)
        with c3:
            st.markdown("**🧾 Activity**")
            tx_count = num("Monthly transactions", "monthly_transaction_count", integer=True)
            spend = num("Monthly spend (INR)", "monthly_spend_inr", step=500.0)
            cash_adv = num("Cash advances (6 m)", "cash_advance_count_6m", integer=True)
            complaints = num("Complaints (6 m)", "complaints_last_6m", integer=True)
            failed_login = num("Failed log-ins (3 m)", "failed_login_attempts_3m", integer=True)
            tenure = num("Account tenure (months)", "account_tenure_months", integer=True)
            last_txn = st.date_input("Last transaction date", value=ru.REFERENCE_DATE.date() - timedelta(days=10),
                                     max_value=date.today())

        unknown = st.multiselect(
            "Mark as unknown (missing) – optional",
            ["age", "annual_income_inr", "credit_score", "credit_utilization_ratio", "monthly_spend_inr"],
        )
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        record = {
            "age": age, "gender": gender, "region": region, "employment_type": employment,
            "annual_income_inr": income, "credit_score": credit_score,
            "credit_utilization_ratio": util, "missed_payments_12m": missed,
            "avg_late_payment_days": late_days, "monthly_transaction_count": tx_count,
            "monthly_spend_inr": spend, "cash_advance_count_6m": cash_adv,
            "complaints_last_6m": complaints, "failed_login_attempts_3m": failed_login,
            "account_tenure_months": tenure, "last_transaction_date": str(last_txn),
            "debt_balance_inr": debt,
        }
        for col in unknown:
            record[col] = np.nan
        row = pd.DataFrame([record])
        p = float(score(row)[0])
        label, color = risk_band(p, threshold)

        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div style='padding:22px;border-radius:12px;background:{color};color:white;text-align:center'>"
                f"<div style='font-size:15px'>Prediction</div>"
                f"<div style='font-size:34px;font-weight:700'>{label}</div>"
                f"<div style='font-size:15px'>P(high risk) = {p:.1%}</div></div>",
                unsafe_allow_html=True,
            )
        with right:
            st.progress(min(max(p, 0.0), 1.0), text=f"Risk score {p:.1%}  |  alert threshold {threshold:.0%}")
            if p >= threshold:
                st.error("🚨 **Alert raised** – route this customer to manual review.")
            elif p >= threshold * 0.5:
                st.warning("⚠️ Below the alert threshold but elevated – keep on the watch-list.")
            else:
                st.success("✅ No alert – customer looks low risk.")
            st.caption("The score is a statistical estimate to support – not replace – a human risk analyst.")

        with st.expander("Customer record sent to the model"):
            st.dataframe(row.T.rename(columns={0: "value"}))

# --------------------------------------------------------------------------- #
# Tab 2 - batch
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Score many customers at once")
    st.write("Upload a CSV with the same columns as the training data (`customer_id` and `risk_status` "
             "are optional). Missing values are allowed.")
    sample_path = os.path.join(HERE, "data", "Risk_Alert_Classifier_Dataset_4600.csv")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    use_sample = False
    if os.path.exists(sample_path):
        use_sample = st.checkbox("…or score the first 200 rows of the bundled sample dataset")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = pd.read_csv(sample_path).head(200)

    if data is not None:
        missing_cols = [c for c in ru.RAW_INPUT_COLUMNS if c not in data.columns]
        if missing_cols:
            st.warning(f"Columns not found (treated as missing): {', '.join(missing_cols)}")
        probs = score(data)
        out = data.copy()
        out["risk_probability"] = probs.round(4)
        out["alert"] = np.where(probs >= threshold, "HIGH RISK", "ok")
        n_alert = int((probs >= threshold).sum())
        a, b, c = st.columns(3)
        a.metric("Customers scored", f"{len(out):,}")
        b.metric("Alerts raised", f"{n_alert:,}")
        c.metric("Alert rate", f"{n_alert / len(out):.1%}")

        if ru.TARGET in out.columns and out[ru.TARGET].notna().all():
            y_true = out[ru.TARGET].astype(int)
            pred = (probs >= threshold).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
            st.info(f"Labels found in the file → TP={tp}, FN (missed)={fn}, FP (false alarms)={fp}, TN={tn}. "
                    "(If this is training data the numbers are optimistic.)")

        st.dataframe(out.sort_values("risk_probability", ascending=False), height=380)
        buf = io.StringIO(); out.to_csv(buf, index=False)
        st.download_button("⬇️ Download scored CSV", buf.getvalue(), "scored_customers.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Tab 3 - performance
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader(f"Hold-out performance – {meta['model_name']}")
    st.caption(f"{meta['n_test']} unseen test customers · evaluated at the default threshold "
               f"{DEFAULT_THRESHOLD:.2f} · training set {meta['n_train']} customers "
               f"({meta['high_risk_share_train']:.1%} high-risk)")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{M['accuracy']:.3f}")
    k[1].metric("Precision", f"{M['precision']:.3f}")
    k[2].metric("Recall (TPR)", f"{M['recall']:.3f}")
    k[3].metric("F1-score", f"{M['f1']:.3f}")
    k[4].metric("ROC-AUC", f"{M['roc_auc']:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(4.6, 4))
        cm = np.array([[M["tn"], M["fp"]], [M["fn"], M["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred Low", "Pred High"]); ax.set_yticks([0, 1], ["Actual Low", "Actual High"])
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
        st.caption(f"Type-I errors (false alarms): **{M['fp']}** · Type-II errors (missed risky customers): **{M['fn']}**")
    with col_b:
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.plot(meta["roc"]["fpr"], meta["roc"]["tpr"], color="#c0392b", lw=2, label=f"AUC = {M['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.markdown(f"**Top drivers of risk** ({meta['importance_kind']})")
    imp = pd.DataFrame(meta["feature_importance"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"], imp["importance"], color="#2e86c1")
    ax.set_xlabel("importance")
    st.pyplot(fig)

    cmp_path = os.path.join(HERE, "models", "model_comparison.csv")
    if os.path.exists(cmp_path):
        st.markdown("**All models compared in the notebook**")
        st.dataframe(pd.read_csv(cmp_path))

# --------------------------------------------------------------------------- #
# Tab 4 - about
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Risk Alert Classifier** is an early-warning classification system for a digital banking
platform. It predicts whether a customer is likely to **default on payments or show
fraud-like behaviour** (`risk_status = 1`).

**Pipeline**
1. Feature engineering – `days_since_last_txn`, `debt_to_income_ratio`
2. Multivariate **KNN imputation** of missing numeric values, most-frequent imputation + one-hot encoding for categories
3. Class-imbalance study: under-sampling, over-sampling, SMOTE, ADASYN
4. Models: Logistic Regression, Decision Tree, Random Forest (+ RandomizedSearchCV / GridSearchCV tuning)
5. Final model chosen to **minimise false negatives**; alert threshold chosen on out-of-fold data

**How to read the result** – the probability is a model score, not a certainty. Use the sidebar
slider to trade *fewer missed risky customers* (lower threshold) against *fewer false alarms*
(higher threshold).

*Built for the Red & White Skill Education – Supervised Learning project.*
"""
    )
