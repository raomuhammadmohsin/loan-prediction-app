import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import csv
import time
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan System v2.0", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #1a73e8; color: white; font-weight: bold; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; margin-top: 10px; }
    .approved { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .rejected { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .success-text { 
        color: #28a745; 
        font-weight: bold; 
        font-size: 22px; 
        text-align: center; 
        border: 2px solid #28a745; 
        padding: 15px; 
        border-radius: 10px; 
        margin-top: 20px;
        background-color: #f0fff4;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- DATA & MODEL LOADING ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("best_loan_model.joblib")
    all_cols = pd.read_csv("cleaned_loan_data.csv").columns.tolist()
    feature_cols = [col for col in all_cols if not col.startswith("Loan_Status")]
    return model, feature_cols

try:
    model, feature_cols = load_assets()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ---------------- HEADER ----------------
st.title("🏦 Strategic Loan Prediction System")
st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2, col3 = st.columns(3)
with col1:
    user_name = st.text_input("Full Name *") 
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
with col2:
    income = st.number_input("Monthly Income (PKR) *", min_value=0, value=75000)
    co_income = st.number_input("Co-Applicant Income (PKR)", min_value=0, value=0)
    credit_history = st.selectbox("Credit Record", ["Good", "Poor"])
    ch_val = 1.0 if credit_history == "Good" else 0.0
with col3:
    loan_pkr = st.number_input("Loan Amount (PKR) *", min_value=10000, value=500000)
    term = st.slider("Tenure (Years)", 1, 30, 15)
    property_area = st.selectbox("Area", ["Urban", "Semiurban", "Rural"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])

# ---------------- PREDICTION LOGIC (Dynamic Confidence) ----------------
st.divider()
if st.button("🔍 ANALYZE ELIGIBILITY"):
    if not user_name.strip():
        st.error("⚠️ Please enter your Full Name before proceeding!")
    else:
        loan_scaled = loan_pkr / 1000
        total_income = income + co_income
        
        input_data = {
            "ApplicantIncome": income, "CoapplicantIncome": co_income, "LoanAmount": loan_scaled,
            "Loan_Amount_Term": term * 12, "Credit_History": ch_val, "TotalIncome": total_income,
            "Income_to_Loan": total_income / (loan_pkr + 1),
            "log_ApplicantIncome": np.log1p(income), "log_LoanAmount": np.log1p(loan_scaled),
            "log_TotalIncome": np.log1p(total_income),
            "Gender_Male": 1 if gender == "Male" else 0, "Married_Yes": 1 if married == "Yes" else 0,
            "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
            "Self_Employed_Yes": 1 if self_emp == "Yes" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if property_area == "Urban" else 0,
            "Dependents_1": 1 if dependents == "1" else 0, "Dependents_2": 1 if dependents == "2" else 0,
            "Dependents_3+": 1 if dependents == "3+" else 0
        }

        input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols, fill_value=0)
        
        # Asli Model Prediction & Confidence
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = probs[1] if prediction == 1 else probs[0]
        dynamic_acc = round(confidence * 100, 2)
        
        res_text = "APPROVED ✅" if prediction == 1 else "REJECTED ❌"
        color_class = "approved" if prediction == 1 else "rejected"
        
        st.markdown(f'<div class="status-box {color_class}"><h2>{res_text}</h2></div>', unsafe_allow_html=True)
        st.write(f"<center>Prediction Confidence: <b>{dynamic_acc}%</b></center>", unsafe_allow_html=True)
        
        # Save results in session to use in feedback form
        st.session_state['res'] = res_text
        st.session_state['user_data'] = {
            "name": user_name, 
            "income": income, 
            "loan": loan_pkr, 
            "conf": dynamic_acc
        }

# ---------------- FEEDBACK SECTION (Saves ONLY on Submit) ----------------
if 'res' in st.session_state:
    st.divider()
    st.subheader("📝 Project Feedback Form")
    with st.form("feedback_form", clear_on_submit=True):
        rating = st.slider("Rate the System (1-5)", 1, 5, 5)
        opinion = st.radio("Is prediction accurate?", ["Yes", "Maybe", "No"])
        sugs = st.text_area("Suggestions")
        
        if st.form_submit_button("Submit Feedback"):
            feedback_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User": st.session_state['user_data']['name'],
                "Income": st.session_state['user_data']['income'],
                "Loan_Amount": st.session_state['user_data']['loan'],
                "Prediction": st.session_state['res'],
                "Model_Accuracy": st.session_state['user_data']['conf'], # Asli Model Confidence
                "Rating": rating,
                "Accuracy_Opinion": opinion,
                "Suggestions": sugs
            }
            
            log_file = "feedback_results.csv"
            cols_order = ["Timestamp", "User", "Income", "Loan_Amount", "Prediction", "Model_Accuracy", "Rating", "Accuracy_Opinion", "Suggestions"]
            
            new_df = pd.DataFrame([feedback_entry])[cols_order]
            
            if not os.path.isfile(log_file):
                new_df.to_csv(log_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            else:
                new_df.to_csv(log_file, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
            
            st.balloons()
            st.markdown('<p class="success-text">FEEDBACK SUBMITTED & DATA SAVED! ✅</p>', unsafe_allow_html=True)
            time.sleep(2) 
            st.rerun()

# ---------------- ADMIN SIDEBAR ----------------
st.sidebar.title("🛠 Admin Access")
try:
    real_password = st.secrets["admin_password"]
except:
    real_password = "admin123"

pass_input = st.sidebar.text_input("Enter Admin Password", type="password")

if pass_input == real_password:
    st.sidebar.success("Welcome, Admin!")
    if os.path.exists("feedback_results.csv"):
        try:
            df_admin = pd.read_csv("feedback_results.csv", engine='python', on_bad_lines='skip')
            cols_order = ["Timestamp", "User", "Income", "Loan_Amount", "Prediction", "Model_Accuracy", "Rating", "Accuracy_Opinion", "Suggestions"]
            df_admin = df_admin.reindex(columns=cols_order)
            
            st.sidebar.subheader(f"📊 Total Entries: {len(df_admin)}")
            edited_df = st.sidebar.data_editor(df_admin, num_rows="dynamic", key="admin_editor_final")
            
            if st.sidebar.button("💾 Save Changes"):
                edited_df.to_csv("feedback_results.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
                st.sidebar.success("Database Updated!")
                st.rerun()
        except Exception as e:
            st.sidebar.error("⚠️ File Error.")
            if st.sidebar.button("🗑️ Reset File"):
                os.remove("feedback_results.csv")
                st.rerun()
    else:
        st.sidebar.info("No records found yet.")
