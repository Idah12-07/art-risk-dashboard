
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="ART Risk Dashboard", layout="centered")
st.title("📊 ART Risk Prediction Dashboard")
st.markdown("Enter patient details below to predict high-risk ART cases.")

# Manual input form
with st.form("input_form"):
    age = st.number_input("Age at reporting", min_value=0, max_value=100, value=35)
    sex = st.selectbox("Sex", ["M", "F"])
    first_regimen = st.selectbox("First Regimen", ["TDF/3TC/DTG", "ABC/3TC/EFV", "AZT/3TC/NVP"])
    current_regimen = st.selectbox("Current Regimen", ["TDF/3TC/DTG", "ABC/3TC/EFV", "AZT/3TC/NVP"])
    last_visit_days = st.number_input("Days since last visit", min_value=0, max_value=365, value=30)
    age_group = st.selectbox("Age Group", ["Child", "Youth", "Adult", "Senior"])
    submitted = st.form_submit_button("Predict")

# Run prediction
if submitted:
    input_data = pd.DataFrame({
        'Age at reporting': [age],
        'Sex': [sex],
        'First Regimen': [first_regimen],
        'Current Regimen': [current_regimen],
        'Last_Visit_Days': [last_visit_days],
        'Age_Group': [age_group]
    })

    # Encode categorical columns
    for col in ['Sex', 'Age_Group', 'First Regimen', 'Current Regimen']:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col].astype(str))

    # Scale and predict
    features = ['Age at reporting', 'Sex', 'First Regimen', 'Current Regimen', 'Last_Visit_Days', 'Age_Group']
    scaled = scaler.transform(input_data[features])
    prob = model.predict_proba(scaled)[:, 1]
    flag = (prob > 0.005).astype(int)

    st.subheader("🧠 Prediction Result")
    st.write(f"**Risk Probability:** {prob[0]:.4f}")
    st.write(f"**Risk Flag:** {'High Risk (1)' if flag[0] == 1 else 'Low Risk (0)'}")

