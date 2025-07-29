import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load("random_forest_model.joblib2")  
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.title("Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Drop 'Amount' if present
        if 'Amount' in data.columns:
            data.drop(columns=['Amount'], inplace=True)

        # Scale 'Time' if it's in the dataset
        if 'Time' in data.columns:
            data['Time'] = scaler.transform(data[['Time']])

        # Keep only the expected columns
        data = data[feature_columns]

        st.subheader("Processed Input Preview")
        st.dataframe(data.head())

        # Make prediction
        prediction = model.predict(data)
        prediction_prob = model.predict_proba(data)[:, 1]

        results = data.copy()
        results['Fraud Prediction'] = prediction
        results['Fraud Probability'] = prediction_prob

        st.subheader("Prediction Results")
        st.write(results[['Fraud Prediction', 'Fraud Probability']])

        # Show summary
        fraud_count = sum(prediction)
        st.success(f"Fraudulent transactions detected: {fraud_count}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
