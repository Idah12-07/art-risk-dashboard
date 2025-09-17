
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="ART Risk Dashboard", layout="wide")
st.title("📊 ART Risk Prediction Dashboard")
st.markdown("Upload patient data to predict high-risk ART cases.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = ['Age', 'Sex', 'Regimen', 'Facility', 'Last_Visit_Days', 'Age_Group']
    df_encoded = pd.get_dummies(df[features])
    df_scaled = scaler.transform(df_encoded)
    predictions = model.predict(df_scaled)
    df['Risk_Flag'] = predictions
    st.dataframe(df[['Age', 'Facility', 'Risk_Flag']])
    st.bar_chart(df['Risk_Flag'].value_counts())
