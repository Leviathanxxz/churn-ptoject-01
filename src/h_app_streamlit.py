import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="ChurnGuard Pro | Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS untuk UI yang lebih Modern
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Gunakan cache agar model tidak di-load ulang setiap kali ada input bar

@st.cache_resource 
def load_model():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # 1. Ambil lokasi folder 'src' (tempat script ini berada)
    current_dir = os.path.dirname(__file__)
    
    # 2. Naik satu level ke root, lalu masuk ke folder 'models'
    # '..' artinya naik satu tingkat folder
    model_path = os.path.abspath(os.path.join(current_dir, '..', 'models', 'final_model_produksi.pkl'))
    # Debugging: Baris ini akan muncul di terminal untuk memastikan path sudah benar
    print(f"Mencoba memuat model dari: {model_path}")
    
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}")
        return None
        
    return joblib.load(model_path)

model = load_model()

# 4. Sidebar - Deskripsi & Instruksi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("ChurnGuard Pro")
    st.info("Sistem prediksi churn pelanggan menggunakan AI XGBoost. Masukkan data di panel utama untuk menganalisis risiko.")
    st.markdown("---")
    st.write("**Model Version:** v1.0.2")
    st.write("**Accuracy:** 89% (XGBoost)")

# 5. Dashboard Header
st.title("🛡️ Customer Retention Dashboard")
st.write("Analisis probabilitas pelanggan berhenti berlangganan secara real-time.")

# 6. Form Input Utama
with st.container():
    tab1, tab2, tab3 = st.tabs(["📋 Data Kontrak", "🌐 Layanan", "👤 Profil Pelanggan"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input("Tenure Months", min_value=0, max_value=120, value=12, help="Lama berlangganan dalam bulan")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        with c2:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=tenure * monthly_charges)
            contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            internet_service = st.selectbox("Layanan Internet", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        with c4:
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            total_services = st.slider("Total Layanan Aktif", 0, 10, 3)

    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            partner = st.selectbox("Memiliki Pasangan?", ["Yes", "No"])
            dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])
        with c6:
            payment_method = st.selectbox("Metode Pembayaran", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])

# 7. LOGIKA FEATURE ENGINEERING (Internal Logic)
# Mengolah input mentah menjadi fitur yang dimengerti model
tenure_group = 'Newbie' if tenure < 12 else ('Active' if tenure < 24 else 'Loyal')
is_streaming_user = 1 if (streaming_tv == 'Yes' or streaming_movies == 'Yes') else 0
risky_loyalist = 1 if (tenure > 24 and contract == 'Month-to-month') else 0
high_cost_m2m = 1 if (monthly_charges > 70 and contract == 'Month-to-month') else 0
fiber_high_risk = 1 if (internet_service == 'Fiber optic' and online_security == 'No') else 0
has_security_bundle = 1 if (online_security == 'Yes' and tech_support == 'Yes') else 0
risk_map = {'Electronic check': 3, 'Mailed check': 2, 'Bank transfer': 1, 'Credit card': 0}
payment_risk = risk_map.get(payment_method, 0)

# 8. Action & Result
st.markdown("###")
if st.button("🚀 ANALISIS RISIKO PELANGGAN"):
    input_df = pd.DataFrame({
        'Tenure Months': [tenure], 'Monthly Charges': [monthly_charges], 'Total Charges': [total_charges],
        'Is Streaming User': [is_streaming_user], 'Total Services': [total_services], 
        'Risky Loyalist': [risky_loyalist],'High Cost Month To Month': [high_cost_m2m], 
        'Is Fiber Optic High Risk': [fiber_high_risk],'Has Security Bundle': [has_security_bundle], 
        'Payment Method Risk Level': [payment_risk],'Tenure Group': [tenure_group], 
        'Contract': [contract], 'Partner': [partner],'Dependents': [dependents], 
        'Multiple Lines': [multiple_lines], 'Device Protection': [device_protection], 
        'Internet Service': [internet_service],'Online Security': [online_security], 
        'Online Backup': [online_backup], 'Tech Support': [tech_support],'Phone Service': [phone_service],
        'Streaming TV': [streaming_tv], 'Streaming Movies': [streaming_movies], 
        'Paperless Billing': [paperless], 'Payment Method': [payment_method]
    })
    
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Tampilan Hasil yang Mewah
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK")
                st.metric("Probability to Churn", f"{probability:.1%}", delta="Alert", delta_color="inverse")
            else:
                st.success("### ✅ LOW RISK")
                st.metric("Probability to Churn", f"{probability:.1%}", delta="Safe", delta_color="normal")
        
        with res_col2:
            st.write("**Rekomendasi Strategi:**")
            if prediction == 1:
                st.warning("1. Berikan penawaran diskon loyalitas.\n2. Hubungi CS untuk feedback layanan.\n3. Tawarkan upgrade ke kontrak 'One year'.")
            else:
                st.info("1. Pertahankan layanan saat ini.\n2. Tawarkan produk cross-sell.\n3. Masukkan ke dalam program referral.")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")