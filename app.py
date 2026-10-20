import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "artifacts" / "churn_model1.pkl"

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churning.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen (0=No, 1=Yes)", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col2:
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}
input_df = pd.DataFrame([input_data])
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.3, 0.05)

if st.button("Predict Churn"):
    try:
        probability = model.predict_proba(input_df)[0, 1]
        st.write("---")
        st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")
        if probability >= threshold:
            st.error("Prediction: The customer is likely to CHURN!")
        else:
            st.success("Prediction: The customer is likely to STAY!")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
