# src/d_tuning.py
import os
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from src.c_pipeline import get_full_pipeline
from src.a_drop_split_data import load_split_data

def run_multi_tuning():
    X_train, _, y_train, _ = load_split_data()
    
    # 1. Hitung rasio untuk handling imbalanced data
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale_pos_weight = neg / pos if pos > 0 else 1
    
    print(f"Balancing: XGBoost scale_pos_weight = {scale_pos_weight:.2f}")


    # 2. Konfigurasi model
    search_configs = [
        {
            'model': [XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)],
    'model__n_estimators':randint(low=100,high=500),
    'model__max_depth':randint(low=3,high=10),
    'model__learning_rate':uniform(loc=0.01, scale=0.29),
    'model__subsample':uniform(loc=0.6, scale=0.39),
    'model__colsample_bytree':uniform(loc=0.6, scale=0.39),
    'model__gamma':uniform(loc=0, scale=0.5),
    'model__min_child_weight':randint(low=1, high=10),
    'model__scale_pos_weight': [scale_pos_weight]
        },
        {
            'model': [RandomForestClassifier(random_state=42)],
    'model__n_estimators':randint(low=100,high=400),
    'model__max_depth':randint(low=5,high=15),
    'model__min_samples_split':randint(low=5,high=15),
    'model__min_samples_leaf':randint(low=5,high=15),
    'model__class_weight':['balanced']
        }
    ]

    pipeline = get_full_pipeline(XGBClassifier()) 
    search = RandomizedSearchCV(
        pipeline, 
        search_configs, 
        cv=5, 
        scoring='f1', 
        n_iter=50, 
        n_jobs=-1, 
        verbose=1
    )
    
    print("Sedang melakukan tuning dan menyimpan model terbaik...")
    search.fit(X_train, y_train)
    
    # 3. Otomatis simpan model terbaik
    if not os.path.exists('models'):
        os.makedirs('models')
        
    best_model = search.best_estimator_
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print("\n" + "="*30)
    print("HASIL TUNING")
    
    # Cetak semua parameter yang terpilih
    print("Parameter Terbaik:")
    for param, value in search.best_params_.items():
        if param == 'model':
            print(f"  - Algoritma: {value.__class__.__name__}")
        else:
            # Menghilangkan prefix 'model__' agar lebih rapi
            clean_param = param.replace('model__', '')
            print(f"  - {clean_param}: {value}")
            
    print(f"Skor F1 (Cross-Validation): {search.best_score_:.4f}")
    print("Model telah disimpan di 'models/best_model.pkl'")
    print("="*30)
    
    return best_model
if __name__ == "__main__":
    run_multi_tuning()