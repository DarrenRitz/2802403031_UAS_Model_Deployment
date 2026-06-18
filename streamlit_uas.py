import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATHS = {
    "Random Forest": BASE_DIR / "artifacts" / "random_forest_model.pkl",
    "XGBoost":       BASE_DIR / "artifacts" / "xgboost_model.pkl",
    "Decision Tree": BASE_DIR / "artifacts" / "decision_tree_model.pkl",
}

PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.pkl"

SCORE_COLOR = {
    "Good":     "#2ecc71",
    "Standard": "#f39c12",
    "Poor":     "#e74c3c",
}

SCORE_EMOJI = {
    "Good":     "✅",
    "Standard": "⚠️",
    "Poor":     "❌",
}

OCCUPATIONS = [
    "Teacher", "Doctor", "Accountant", "Writer", "Musician",
    "Engineer", "Architect", "Journalist", "Entrepreneur",
    "Developer", "Media_Manager", "Scientist", "Manager",
    "Lawyer", "Mechanic",
]

PAYMENT_BEHAVIOURS = [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
]

@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7c8db5;
        margin-bottom: 10px;
        margin-top: 24px;
        border-bottom: 1px solid #1e2235;
        padding-bottom: 6px;
    }

    .result-card {
        border-radius: 12px;
        padding: 28px 32px;
        margin-top: 24px;
        text-align: center;
    }
    .result-label {
        font-size: 13px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7c8db5;
        margin-bottom: 6px;
    }
    .result-score {
        font-size: 48px;
        font-weight: 800;
        line-height: 1.1;
    }
    .result-model {
        font-size: 12px;
        color: #555e7a;
        margin-top: 8px;
    }

    .prob-label {
        font-size: 12px;
        color: #7c8db5;
        margin-bottom: 2px;
    }

    label { color: #b0b8d0 !important; font-size: 13px !important; }

    div.stButton > button {
        background: linear-gradient(135deg, #3b5bdb, #5c7cfa);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 0;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }

    section[data-testid="stSidebar"] {
        background-color: #090b10;
        border-right: 1px solid #1a1d2e;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Credit Score")
    st.markdown("---")
    st.markdown("**Model**")
    selected_model = st.selectbox(
        "Pilih model prediksi:",
        list(MODEL_PATHS.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Random Forest - XGBoost - Decision Tree")
    st.caption("Target: Good / Standard / Poor")
    st.markdown("---")
    st.caption("2802403031 - Darren Ritz Junaidi")

st.markdown("# Credit Score Predictor")
st.markdown(
    "Masukkan data nasabah di bawah, lalu klik **Predict** untuk melihat "
    "prediksi skor kredit menggunakan model yang dipilih."
)
st.markdown("---")

with st.form("predict_form"):

    # Identitas Nasabah
    st.markdown('<div class="section-title">Identitas Nasabah</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=17, max_value=100, value=30)
    with c2:
        occupation = st.selectbox("Occupation", OCCUPATIONS)
    with c3:
        annual_income = st.number_input("Annual Income (USD)", min_value=0.0, value=50000.0, step=1000.0)

    # Informasi Bank & Kredit
    st.markdown('<div class="section-title">Informasi Bank & Kredit</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, value=3)
    with c2:
        num_credit_card = st.number_input("Num Credit Cards", min_value=0, value=2)
    with c3:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=15.0, step=0.5)
    with c4:
        num_of_loan = st.number_input("Num of Loans", min_value=0, value=2)

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_mix = st.selectbox("Credit Mix", ["Standard", "Good", "Bad"])
    with c2:
        outstanding_debt = st.number_input("Outstanding Debt (USD)", min_value=0.0, value=500.0, step=10.0)
    with c3:
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)

    c1, c2 = st.columns(2)
    with c1:
        credit_history_age = st.number_input("Credit History Age (months)", min_value=0, value=60)
    with c2:
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, value=2)

    # Perilaku Pembayaran
    st.markdown('<div class="section-title">Perilaku Pembayaran</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, value=5)
    with c2:
        num_of_delayed_payment = st.number_input("Num of Delayed Payments", min_value=0, value=3)
    with c3:
        payment_of_min_amount = st.selectbox("Payment of Min Amount", ["No", "Yes"])

    c1, c2 = st.columns(2)
    with c1:
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOURS)
    with c2:
        changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=10.0, step=0.5)

    # Keuangan Bulanan
    st.markdown('<div class="section-title">Keuangan Bulanan</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary (USD)", min_value=0.0, value=4000.0, step=100.0)
    with c2:
        total_emi_per_month = st.number_input("Total EMI per Month (USD)", min_value=0.0, value=200.0, step=10.0)
    with c3:
        amount_invested_monthly = st.number_input("Amount Invested Monthly (USD)", min_value=0.0, value=150.0, step=10.0)

    monthly_balance = st.number_input("Monthly Balance (USD)", min_value=0.0, value=500.0, step=10.0)

    # Loan Type
    st.markdown('<div class="section-title">Jenis Pinjaman yang Dimiliki</div>', unsafe_allow_html=True)
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        student_loan            = st.checkbox("Student Loan")
        personal_loan           = st.checkbox("Personal Loan")
    with lc2:
        credit_builder_loan     = st.checkbox("Credit-Builder Loan")
        mortgage_loan           = st.checkbox("Mortgage Loan")
    with lc3:
        debt_consolidation_loan = st.checkbox("Debt Consolidation Loan")
        payday_loan             = st.checkbox("Payday Loan")
    with lc4:
        auto_loan               = st.checkbox("Auto Loan")
        home_equity_loan        = st.checkbox("Home Equity Loan")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Credit Score")

if submitted:
    # Build raw input DataFrame
    input_dict = {
        "Age":                      [age],
        "Occupation":               [occupation],
        "Annual_Income":            [annual_income],
        "Monthly_Inhand_Salary":    [monthly_inhand_salary],
        "Num_Bank_Accounts":        [num_bank_accounts],
        "Num_Credit_Card":          [num_credit_card],
        "Interest_Rate":            [interest_rate],
        "Num_of_Loan":              [num_of_loan],
        "Delay_from_due_date":      [delay_from_due_date],
        "Num_of_Delayed_Payment":   [num_of_delayed_payment],
        "Changed_Credit_Limit":     [changed_credit_limit],
        "Num_Credit_Inquiries":     [num_credit_inquiries],
        "Credit_Mix":               [credit_mix],
        "Outstanding_Debt":         [outstanding_debt],
        "Credit_Utilization_Ratio": [credit_utilization_ratio],
        "Credit_History_Age":       [credit_history_age],
        "Payment_of_Min_Amount":    [payment_of_min_amount],
        "Total_EMI_per_month":      [total_emi_per_month],
        "Amount_invested_monthly":  [amount_invested_monthly],
        "Payment_Behaviour":        [payment_behaviour],
        "Monthly_Balance":          [monthly_balance],
        # Loan Type (One Hot)
        "student_loan":             [int(student_loan)],
        "personal_loan":            [int(personal_loan)],
        "credit-builder_loan":      [int(credit_builder_loan)],
        "mortgage_loan":            [int(mortgage_loan)],
        "debt_consolidation_loan":  [int(debt_consolidation_loan)],
        "payday_loan":              [int(payday_loan)],
        "auto_loan":                [int(auto_loan)],
        "home_equity_loan":         [int(home_equity_loan)],
    }
    input_df = pd.DataFrame(input_dict)

    try:
        preprocessor = load_preprocessor()
        models       = load_models()
        model        = models[selected_model]
        label_encoder = preprocessor.get_label_encoder()

        input_proc = preprocessor.transform(input_df)

        pred_enc  = model.predict(input_proc)[0]
        credit_score = label_encoder.inverse_transform([pred_enc])[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(input_proc)[0]
            classes   = label_encoder.inverse_transform(model.classes_)
            probabilities = dict(zip(classes, proba_arr))

        color = SCORE_COLOR.get(credit_score, "#888")
        emoji = SCORE_EMOJI.get(credit_score, "")

        st.markdown(f"""
        <div class="result-card" style="background:{color}18; border: 1.5px solid {color}55;">
            <div class="result-label">Prediksi Credit Score</div>
            <div class="result-score" style="color:{color};">{emoji} {credit_score}</div>
            <div class="result-model">Model: {selected_model}</div>
        </div>
        """, unsafe_allow_html=True)

        if probabilities:
            st.markdown("#### Probabilitas per Kelas")
            for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
                bar_color = SCORE_COLOR.get(label, "#888")
                st.markdown(f'<div class="prob-label">{label}</div>', unsafe_allow_html=True)
                st.progress(float(prob))
                st.caption(f"{prob * 100:.1f}%")

        # Visual
        st.markdown("---")
        st.markdown("#### Profil Nasabah")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#0f1117")

        # Chart 1: Financial snapshot
        fin_labels = ["Annual\nIncome", "Monthly\nSalary", "Outstanding\nDebt", "Monthly\nBalance", "EMI/Month"]
        fin_values = [
            annual_income / 1000,
            monthly_inhand_salary / 100,
            outstanding_debt / 100,
            monthly_balance / 100,
            total_emi_per_month,
        ]
        bar_colors = ["#5c7cfa", "#74c0fc", "#e74c3c", "#2ecc71", "#f39c12"]

        axes[0].bar(fin_labels, fin_values, color=bar_colors, edgecolor="none", width=0.55)
        axes[0].set_facecolor("#0f1117")
        axes[0].tick_params(colors="#7c8db5", labelsize=9)
        axes[0].set_ylabel("Value (scaled)", color="#7c8db5", fontsize=9)
        axes[0].set_title("Financial Snapshot", color="#b0b8d0", fontsize=11, pad=10)
        for spine in axes[0].spines.values():
            spine.set_edgecolor("#1e2235")

        # Chart 2: Credit behaviour
        beh_labels = ["Delay\nDays", "Delayed\nPayments", "Credit\nInquiries", "Num\nLoans", "Credit\nCards"]
        beh_values = [delay_from_due_date, num_of_delayed_payment,
                      num_credit_inquiries, num_of_loan, num_credit_card]
        beh_colors = ["#e74c3c" if v > 5 else "#5c7cfa" for v in beh_values]

        axes[1].bar(beh_labels, beh_values, color=beh_colors, edgecolor="none", width=0.55)
        axes[1].set_facecolor("#0f1117")
        axes[1].tick_params(colors="#7c8db5", labelsize=9)
        axes[1].set_ylabel("Count / Days", color="#7c8db5", fontsize=9)
        axes[1].set_title("Credit Behaviour", color="#b0b8d0", fontsize=11, pad=10)
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#1e2235")

        high_patch = mpatches.Patch(color="#e74c3c", label="> 5 (tinggi)")
        norm_patch = mpatches.Patch(color="#5c7cfa", label="≤ 5 (normal)")
        axes[1].legend(handles=[norm_patch, high_patch], fontsize=8,
                       facecolor="#0f1117", edgecolor="#1e2235", labelcolor="#b0b8d0")

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}\nPastikan pipeline sudah dijalankan dan folder `artifacts/` ada.")
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")