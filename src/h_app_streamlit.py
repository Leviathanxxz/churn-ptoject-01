import streamlit as st
import requests

st.title("Telco Customer Churn Predictor")

# Input User
tenure = st.slider("Tenure (Months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# ... input lainnya

if st.button("Check Risk"):
    # Kirim data ke FastAPI (g_app.py)
    response = requests.post("http://localhost:8000/predict", json={
        "Tenure Months": tenure,
        "Contract": contract,
        # ... kirim data lainnya
    })
    result = response.json()
    st.write(f"Hasil Prediksi: {result['prediction']}")