# src/e_train.py
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from src.c_pipeline import get_full_pipeline
from src.a_drop_split_data import load_split_data

def run_final_training():
    X_train, _, y_train, _ = load_split_data()
    
    # GANTI DI SINI: Masukkan model yang menang dari hasil tuning
    # Jika XGBoost yang menang:
    best_model =  RandomForestClassifier(
        n_estimators=167, 
        max_depth=11, 
        min_samples_leaf=9,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    
    # Jika Random Forest yang menang, ganti dengan:
    # best_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced')
    
    # Masukkan ke pipeline
    pipeline = get_full_pipeline(best_model)
    
    print("Sedang melatih model final dengan seluruh data...")
    pipeline.fit(X_train, y_train)
    
    # Simpan model final
    joblib.dump(pipeline, 'models/final_model_produksi.pkl')
    print("Model produksi final berhasil disimpan!")

if __name__ == "__main__":
    run_final_training()