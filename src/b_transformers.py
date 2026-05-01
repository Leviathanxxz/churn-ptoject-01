import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Tidak ada parameter yang perlu diinisialisasi

    def fit(self, X, y=None):
        return self  # Tidak ada parameter yang perlu di-fit

    def transform(self, X):
        # Kita copy X agar tidak merusak data asli
        X = X.copy()

        # 2. Total Services (jumlah layanan yang digunakan)
        cols_to_map = ['Online Security', 'Online Backup', 'Device Protection', 
                       'Tech Support', 'Streaming TV', 'Streaming Movies',
                       'Phone Service']
        
        mapping = {'Yes': 1, 'No': 0, 'No internet service': 0}
        for col in cols_to_map:
            X[col] = X[col].replace(mapping)
            
        X['Total Services'] = X[cols_to_map].sum(axis=1)
        
        # 3. Tenure Grouping (kelompokkan masa keanggotaan)
        X['Tenure Group'] = pd.cut(X['Tenure Months'], 
                                  bins=[-1, 6, 24, 100], 
                                  labels=['Newbie', 'Active', 'Loyal'],
                                  include_lowest=True)
        
        # 4. Contract Flag (untuk kontrak month-to-month paling rawan churn)
        X['Is Month To Month'] = (X['Contract'] == 'Month-to-month').astype(int)
        
        # 5. Total Charges (Handle missing values/non-numeric)
        X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce').fillna(0)

        # 6. Avg Monthly Charges (hitung rata-rata biaya bulanan)
        X['Avg Monthly Charges'] = X['Total Charges'] / (X['Tenure Months'] + 1)

        # 7. Risky Loyalist (pelanggan setia yang berisiko tinggi untuk churn karena langganan hanya per bulan)
        X['Risky Loyalist'] = ((X['Tenure Group'] == 'Loyal') & (X['Is Month To Month'] == 1)).astype(int)

        # 8. High Cost Month To Month (pelanggan month-to-month dengan biaya bulanan di atas median)
        X['High Cost Month To Month'] = ((X['Is Month To Month'] == 1) & (X['Avg Monthly Charges'] > X['Avg Monthly Charges'].median())).astype(int)

        # 9. Is Fiber Optic High Risk (karena lebih mahal pelanggan dengan layanan internet fiber optic lebih rawan churn)
        X['Is Fiber Optic High Risk'] = ((X['Internet Service'] == 'Fiber optic')).astype(int)

        # 10. Has Security Bundle (kombinasi layanan keamanan paling berpengaruh)
        X['Has Security Bundle'] = ((X['Online Security'] == 1) & (X['Tech Support'] == 1)).astype(int)

        # 11. Payment Method Risk Level (pelanggan dengan metode pembayaran tertentu mungkin lebih rawan churn 
        payment_risk_mapping = {
            'Electronic check': 3,          # High Risk
            'Mailed check': 2,              # Medium Risk
            'Bank transfer (automatic)': 1, # Low Risk
            'Credit card (automatic)': 1    # Low Risk
        }
        X['Payment Method Risk Level'] = X['Payment Method'].map(payment_risk_mapping)

        # Streaming User (Pelanggan yang menggunakan internet hanya untuk hiburan (streaming) sensitif terhadap gangguan kecepatan)
        X['Is Streaming User'] = ((X['Streaming TV'] == 1) | (X['Streaming Movies'] == 1)).astype(int)
       
        return X