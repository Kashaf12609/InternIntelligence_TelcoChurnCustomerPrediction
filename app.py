import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("churn_pipeline.pkl")

st.title("📊 Telco Customer Churn Prediction")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

# --- Prepare input DataFrame ---
input_data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior_citizen],
    "Partner":[partner],
    "Dependents":[dependents],
    "PhoneService":[phone_service],
    "MultipleLines":[multiple_lines],
    "InternetService":[internet_service],
    "OnlineSecurity":[online_security],
    "OnlineBackup":[online_backup],
    "DeviceProtection":[device_protection],
    "TechSupport":[tech_support],
    "StreamingTV":[streaming_tv],
    "StreamingMovies":[streaming_movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless_billing],
    "PaymentMethod":[payment_method],
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges]
})

# --- Predict ---
if st.button("🔮 Predict Churn"):
    prediction = pipeline.predict(input_data)[0]
    st.success("✅ Customer will **Churn**" if prediction==1 else "✅ Customer will **Not Churn**")
