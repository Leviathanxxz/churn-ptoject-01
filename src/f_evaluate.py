# src/f_evaluate.py
# agar tidak memanggil jendela GUI dan bisa menyimpan gambar plot 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve,accuracy_score
from src.a_drop_split_data import load_split_data
from sklearn.pipeline import Pipeline

def run_threshold_tuning(model, X_test, y_test, save_path='models/threshold_config.json'):
    """
    Mencari threshold terbaik untuk memaksimalkan F1-Score dan menyimpannya.
    """
    print("--- Memulai Threshold Tuning ---")
    
    # 1. Dapatkan probabilitas (Bukan prediksi 0/1)
    # Kita ambil kolom index 1 (probabilitas Churn)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Hitung Precision & Recall untuk berbagai threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # 3. Hitung F1-Score untuk setiap threshold
    # Formula F1: 2 * (P * R) / (P + R)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # 4. Cari index dengan F1-Score tertinggi
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    
    print(f"Threshold Terbaik Ditemukan: {best_threshold:.4f}")
    print(f"Estimasi F1-Score: {best_f1:.4f}")

    # 5. Visualisasi (Opsional tapi sangat disarankan)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.axvline(best_threshold, color='red', linestyle=':', label=f'Best: {best_threshold:.2f}')
    plt.title('Precision-Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/threshold_tuning.png')

    # 6. Simpan hasil ke JSON (Poin 3 yang kita bahas tadi)
    config = {
        "best_threshold": best_threshold,
        "best_f1_score": best_f1,
        "model_version": "1.0"
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Konfigurasi threshold berhasil disimpan di: {save_path}")
    
    return best_threshold

def predict_with_threshold(model, X, threshold_path='models/threshold_config.json'):
    """
    Inference menggunakan threshold kustom yang sudah disimpan di JSON.
    """
    # 1. Load threshold dari file JSON hasil tuning
    with open(threshold_path, 'r') as f:
        config = json.load(f)
        threshold = config['best_threshold']
    
    # 2. Ambil PROBABILITAS (bukan kelas 0/1)
    y_probs = model.predict_proba(X)[:, 1]
    
    # 3. Klasifikasi manual: 
    # Jika prob >= threshold jadi 1, jika tidak jadi 0
    y_pred = (y_probs >= threshold).astype(int)
    
    return y_pred

def plot_feature_importance(pipeline, top_n=15):
   # 1. Ambil model dan preprocessor dari pipeline
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # 2. Dapatkan nama fitur setelah transformasi
    try:
        # Ini cara cerdas mengambil nama fitur setelah OneHotEncoder
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback jika terjadi error
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    
    # 3. Buat DataFrame untuk visualisasi
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 4. Plot
    plt.figure(figsize=(12, 8)) # Saya perbesar sedikit agar teks tidak bertumpuk
    
    # Kita simpan objek barplot ke variabel 'ax'
    ax = sns.barplot(x='importance', y='feature', data=feature_imp.head(top_n), 
                     hue='feature', palette='viridis', legend=False)
    
    # --- INI BARIS AJAIBNYA ---
    # melakukan iterasi pada setiap container (setiap bar) di plot
    # fmt='%.3f' artinya tampilkan 3 angka di belakang koma
    # padding=3 memberi jarak antara angka dengan ujung bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
    
    plt.title(f'Top {top_n} Feature Importance with Scores')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png') 

def run_evaluation():
    # 1. Load data test dan model terbaik
    X_train, X_test, y_train, y_test = load_split_data()
    model = joblib.load('models/final_model_produksi.pkl')

    # 2. Jalankan tuning terlebih dahulu untuk mendapatkan/memperbarui file JSON
    best_threshold = run_threshold_tuning(model, X_test, y_test)

    # 2. Gunakan fungsi predict_with_threshold yang sudah kamu buat
    y_pred = predict_with_threshold(model, X_test)
    
    # 3. Cetak Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))

    # 4. akurasi antara data train dan data test
    y_pred_train = predict_with_threshold(model, X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f'Akurasi data train : {acc_train:.2%}')
    print('-'*40)
    y_pred_test = predict_with_threshold(model, X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f'Akurasi data test : {acc_test:.2%}')
    print('-'*40)
    print(f'Selisih atau GAP : {acc_train-acc_test:.2%}')
    
    # 5. Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stay', 'Churn'], 
                yticklabels=['Stay', 'Churn'])
    plt.xlabel('Prediksi Model')
    plt.ylabel('Kenyataan (Actual)')
    plt.title('Confusion Matrix: Akurasi Prediksi')
    
    # 6. Pastikan folder 'results' ada (opsional, tapi disarankan)
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 7. Simpan gambar
    save_path = 'results/confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\n--- Selesai! ---")
    print(f"Confusion Matrix telah disimpan di: {save_path}")
    print(f"jumlah kolom : {X_test.shape[1]}")

    # 8 . Heatmap untuk melihat korelasi antara fitur dan target
        # Ambil preprocessor utama
    preprocessor = model.named_steps['preprocessor']
    
    # Jika preprocessor butuh kolom engineering, kita harus memastikan 
    # X_train sudah ditransformasi oleh step SEBELUM preprocessor.
    
    # Kita gunakan seluruh pipeline KECUALI step terakhir (classifier)
    fitur_extractor = Pipeline(steps=model.steps[:-1])
    X_train_transformed = fitur_extractor.transform(X_train)
    
    # Ambil nama kolom baru
    try:
        new_cols = preprocessor.get_feature_names_out()
    except:
        new_cols = [f"col_{i}" for i in range(X_train_transformed.shape[1])]
    
    X_train_final = pd.DataFrame(X_train_transformed, columns=new_cols)
    
    # Buat Heatmap
    plt.figure(figsize=(20, 18))
    datafull = pd.DataFrame(X_train_transformed, columns=new_cols)
    datafull['Churn Value'] = y_train.values
    datacorr = datafull.corr()
    sns.heatmap(datacorr, vmax=1, vmin=-1, center=0, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig('results/heatmap.png', bbox_inches='tight')
    plt.close()
    print("Heatmap berhasil disimpan.")

    plot_feature_importance(model)

if __name__ == "__main__":
    run_evaluation()