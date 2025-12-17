import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
except:
    st.error("❌ Model or Data files missing! Upload them to GitHub.")
    st.stop()

# ---------------- HEADER ----------------
st.title("🏦 Strategic Loan Prediction System")
st.markdown("Assess loan eligibility and provide feedback for system improvement.")
st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal")
    user_name = st.text_input("Full Name")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

with col2:
    st.subheader("💰 Financial")
    income = st.number_input("Monthly Income (PKR)", min_value=0, value=75000)
    co_income = st.number_input("Co-Applicant Income (PKR)", min_value=0, value=0)
    credit_history = st.selectbox("Credit Record", ["Good", "Poor"])
    ch_val = 1.0 if credit_history == "Good" else 0.0

with col3:
    st.subheader("🏠 Loan Details")
    loan_pkr = st.number_input("Loan Amount (PKR)", min_value=10000, value=500000)
    term = st.slider("Tenure (Years)", 1, 30, 15)
    property_area = st.selectbox("Area", ["Urban", "Semiurban", "Rural"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])

# ---------------- PREDICTION LOGIC ----------------
st.divider()
if st.button("🔍 ANALYZE ELIGIBILITY"):
    # Preprocessing
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
    prediction = model.predict(input_df)[0]
    
    # Result Box logic
    if prediction == 1:
        res_text = "Loan Approval: APPROVED ✅"
        st.markdown(f'<div class="status-box approved"><h2>{res_text}</h2></div>', unsafe_allow_html=True)
    else:
        res_text = "Loan Approval: REJECTED ❌"
        st.markdown(f'<div class="status-box rejected"><h2>{res_text}</h2></div>', unsafe_allow_html=True)
    
    # Accuracy display (Aap apni actual model accuracy yahan likh saktay hain)
    st.write(f"<center><p style='color:gray; margin-top:10px;'>Model Prediction Accuracy: <b>82.4%</b></p></center>", unsafe_allow_html=True)
    
    st.session_state['res'] = res_text
    st.session_state['user_data'] = {"name": user_name, "income": income, "loan": loan_pkr}

# ---------------- FEEDBACK SECTION ----------------
if 'res' in st.session_state:
    st.divider()
    st.subheader("📝 Project Feedback Form")
    with st.form("feedback_form"):
        rating = st.slider("Rate the System (1-5)", 1, 5, 5)
        opinion = st.radio("Is the prediction accurate?", ["Yes", "Maybe", "No"])
        sugs = st.text_area("Suggestions for v3.0")
        
        if st.form_submit_button("Submit Feedback"):
            feedback_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User": st.session_state['user_data']['name'],
                "Income": st.session_state['user_data']['income'],
                "Loan_Amount": st.session_state['user_data']['loan'],
                "Prediction": st.session_state['res'],
                "Rating": rating,
                "Accuracy_Opinion": opinion,
                "Suggestions": sugs
            }
            
            log_file = "feedback_results.csv"
            new_df = pd.DataFrame([feedback_entry])
            if not os.path.isfile(log_file):
                new_df.to_csv(log_file, index=False)
            else:
                new_df.to_csv(log_file, mode='a', header=False, index=False)
            
            st.balloons()
            st.success("Thank you! Your feedback has been recorded.")

# ---------------- ADMIN SECTION (SIDEBAR) ----------------
st.sidebar.title("🛠 Admin Access")

# Password field in Sidebar
password = st.sidebar.text_input("Enter Admin Password", type="password")

# Replace 'admin123' with any password you like
if password == "admin123":
    st.sidebar.success("Welcome back, Admin!")
    if st.sidebar.checkbox("Show Collected Responses"):
        if os.path.exists("feedback_results.csv"):
            data = pd.read_csv("feedback_results.csv")
            st.sidebar.write(f"Total entries: {len(data)}")
            st.sidebar.dataframe(data)
            
            # Download Link
            csv = data.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="📥 Download Data for Report",
                data=csv,
                file_name="loan_feedback_data.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.warning("No feedback data found yet.")
elif password != "":
    st.sidebar.error("Incorrect Password!")
else:
    st.sidebar.info("Please enter password to view user data.")

