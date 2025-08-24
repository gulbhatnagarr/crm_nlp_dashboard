import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline
import os

# --------------------------
# Load models and columns
# --------------------------
churn_model = joblib.load("models/churn_model.pkl")
try:
    model_columns = joblib.load("models/churn_model_columns.pkl")
except:
    st.error("churn_model_columns.pkl not found. Please run train_model.py first!")
    st.stop()

# Load Hugging Face sentiment pipeline safely
try:
    hf_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )
except Exception as e:
    st.error(f"Failed to load Hugging Face model: {e}")
    hf_pipeline = None

# Load ML sentiment model
sentiment_model = joblib.load("models/sentiment_model.pkl")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="CRM Dashboard", page_icon="💼", layout="wide")
st.title("💼 CRM Dashboard: Customer Sentiment + Churn Prediction")

st.sidebar.header("Demo Options")
option = st.sidebar.selectbox("Select Input Type", ["Single Customer", "Upload CSV"])

# --------------------------
# Single Customer Input
# --------------------------
if option == "Single Customer":
    st.subheader("Enter Customer Data & Feedback")

    feedback = st.text_area("Customer Feedback / Tweet:")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    if st.button("Analyze Customer"):

        # --------------------------
        # Sentiment Analysis
        # --------------------------
        if feedback.strip():
            ml_sentiment = sentiment_model.predict([feedback])[0]
            hf_label, hf_score = "N/A", "N/A"
            if hf_pipeline:
                hf_result = hf_pipeline(feedback)[0]
                hf_label = hf_result['label']
                hf_score = round(hf_result['score'], 3)
        else:
            ml_sentiment = hf_label = hf_score = "N/A"

        # --------------------------
        # Churn Prediction
        # --------------------------
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract_Month-to-Month": 1 if contract=="Month-to-Month" else 0,
            "Contract_One year": 1 if contract=="One Year" else 0,
            "Contract_Two year": 1 if contract=="Two Year" else 0,
            "Dependents_Yes": 1 if dependents=="Yes" else 0,
            "DeviceProtection_1": 1 if device_protection=="Yes" else 0,
            "DeviceProtection_No internet service": 1 if device_protection=="No internet service" else 0,
        }

        input_df = pd.DataFrame([input_dict])
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        churn_class = churn_model.predict(input_df)[0]
        churn_prob = churn_model.predict_proba(input_df)[0][1]

        # --------------------------
        # Display Results
        # --------------------------
        st.subheader("🔎 Customer Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💬 Sentiment Analysis:**")
            st.write(f"ML Model Prediction: {ml_sentiment}")
            st.write(f"Hugging Face Prediction: {hf_label} (score: {hf_score})")

        with col2:
            st.markdown("**📊 Churn Prediction:**")
            st.write(f"Churn Class: {churn_class}")
            st.write(f"Churn Probability: {churn_prob:.2f}")

        if churn_class == 1 and ml_sentiment.lower() == "negative":
            st.error("⚠️ High-Risk Customer! Negative sentiment + High churn risk")

# --------------------------
# CSV Upload for batch prediction
# --------------------------
else:
    st.subheader("Upload CSV for Batch Analysis")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Preprocess for churn model
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        churn_preds = churn_model.predict(df)
        churn_probs = churn_model.predict_proba(df)[:,1]
        df['Churn_Prediction'] = churn_preds
        df['Churn_Probability'] = churn_probs

        st.write("✅ Batch Churn Predictions:")
        st.dataframe(df)

        # Optionally, visualize
        st.bar_chart(df['Churn_Prediction'].value_counts())