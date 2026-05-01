from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
from pydantic import BaseModel, Field
# Import wajib agar class transformer kamu dikenali saat load pickle
from src.b_transformers import * 

app = FastAPI(title="Telco Churn API - High Performance")

# 1. Definisi Input (Hanya Fitur Asli/Mentah dari Dataset)
class CustomerData(BaseModel):
    # Demografi & Lokasi
    Partner: str
    Dependents: str

    # Fitur Dasar untuk Feature Engineering
    Tenure_Months: int = Field(alias="Tenure Months")
    Phone_Service: str = Field(alias="Phone Service")
    Multiple_Lines: str = Field(alias="Multiple Lines")
    Internet_Service: str = Field(alias="Internet Service")
    Online_Security: str = Field(alias="Online Security")
    Online_Backup: str = Field(alias="Online Backup")
    Device_Protection: str = Field(alias="Device Protection")
    Tech_Support: str = Field(alias="Tech Support")
    Streaming_TV: str = Field(alias="Streaming TV")
    Streaming_Movies: str = Field(alias="Streaming Movies")
    Contract: str
    Paperless_Billing: str = Field(alias="Paperless Billing")
    Payment_Method: str = Field(alias="Payment Method")
    Monthly_Charges: float = Field(alias="Monthly Charges")
    Total_Charges: str = Field(alias="Total Charges")

    class Config:
        populate_by_name = True

# 2. Load Model & Threshold
try:
    model = joblib.load('models/final_model_produksi.pkl')
    with open('models/threshold_config.json', 'r') as f:
        config = json.load(f)
        best_threshold = config.get('best_threshold', 0.5)
except Exception as e:
    print(f"Error loading resources: {e}")
    model = None

@app.post("/predict")
async def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model tidak tersedia.")
    
    try:
        # Konversi ke dict dengan ALIAS (mengembalikan spasi agar dibaca transformer)
        # Contoh: Tenure_Months jadi 'Tenure Months'
        input_dict = data.dict(by_alias=True)
        df = pd.DataFrame([input_dict])
        
        # PROSES MAGIC:
        # Saat model.predict_proba dipanggil, Pipeline akan otomatis menjalankan:
        # 1. b_transformers.py (Menghitung 'Total Services', 'Risky Loyalist', dll)
        # 2. Preprocessing (Scaling & Encoding)
        # 3. Model Prediction
        y_prob = model.predict_proba(df)[:, 1][0]
        
        prediction = 1 if y_prob >= best_threshold else 0
        
        return {
            "prediction": "Churn" if prediction == 1 else "Stay",
            "probability": round(float(y_prob), 4),
            "applied_threshold": best_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))