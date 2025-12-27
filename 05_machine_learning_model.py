import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- KONFIGURASI PATH ---
DATA_PATH = 'cleaned_data/dataset_final_untuk_ml.csv'
MODEL_OUT = 'cleaned_data/model_kemiskinan_final.pkl'
FEATURES_OUT = 'cleaned_data/feature_names.pkl' # Penting untuk Dashboard

def build_machine_learning_model():
    print("🚀 [05] Memasuki tahap Pelatihan Model...")
    
    if not os.path.exists(DATA_PATH):
        print(f"🛑 Error: File {DATA_PATH} tidak ditemukan! Jalankan skrip 04 dulu.")
        return None
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. PEMILIHAN FITUR (Disamakan dengan output skrip 04)
    # Pastikan nama kolom ini ada di dataset_final_untuk_ml.csv
    features = ['P0_Lag1', 'TPT', 'Garis_Kemiskinan', 'Sentimen_Global', 'P1', 'P2']
    target = 'P0'
    
    # Validasi keberadaan kolom sebelum lanjut
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        print(f"🛑 Error: Kolom berikut tidak ada di dataset: {missing_cols}")
        return None

    X = df[features]
    y = df[target]
    
    # 2. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. TRAINING MODEL
    print(f"Melatih Random Forest dengan {len(X_train)} data...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. EVALUASI
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=================================================")
    print("HASIL EVALUASI MODEL")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-Squared (Akurasi): {r2*100:.2f}%")
    print("=================================================")
    
    # 5. FEATURE IMPORTANCE
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Fitur': features, 'Kepentingan': importances}).sort_values(by='Kepentingan', ascending=False)
    print("\nFitur Paling Berpengaruh:")
    print(feat_imp)
    
    # 6. VISUALISASI
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Akurasi Model: Aktual vs Prediksi')
    plt.savefig('cleaned_data/plot_prediksi.png')
    
    # 7. SIMPAN MODEL & DAFTAR FITUR
    joblib.dump(model, MODEL_OUT)
    joblib.dump(features, FEATURES_OUT) # Menyimpan urutan kolom fitur
    
    print(f"\n✅ Model disimpan di: {MODEL_OUT}")
    print(f"✅ Daftar fitur disimpan di: {FEATURES_OUT}")
    
    return model

if __name__ == '__main__':
    build_machine_learning_model()