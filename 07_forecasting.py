import pandas as pd
import joblib
import os

# --- KONFIGURASI PATH ---
DATA_FINAL_PATH = 'cleaned_data/dataset_final_untuk_ml.csv'
MODEL_PATH = 'cleaned_data/model_kemiskinan_final.pkl'
FEATURES_PATH = 'cleaned_data/feature_names.pkl'
OUTPUT_FORECAST = 'cleaned_data/data_forecasting_2026_2027.csv'

def run_forecasting():
    print("🚀 [07] Memulai Peramalan Kemiskinan 5 Tahun Kedepan...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print("🛑 Error: Model atau Daftar Fitur tidak ditemukan. Jalankan skrip 05 dulu.")
        return

    # 1. Muat Model dan Data
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATA_FINAL_PATH)
    
    # 2. Ambil data tahun terakhir sebagai basis
    latest_year = df['Tahun'].max()
    df_latest = df[df['Tahun'] == latest_year].copy()
    
    forecast_results = []

    # 3. Loop untuk 5 tahun kedepan
    for i in range(1, 6):  # 1, 2, 3, 4, 5
        year_target = latest_year + i
        print(f"   Memproses Prediksi Tahun {year_target}...")
        
        # Siapkan data input (menggunakan P0 tahun sebelumnya sebagai P0_Lag1)
        df_latest['P0_Lag1'] = df_latest['P0']
        df_latest['Tahun'] = year_target
        
        # Lakukan Prediksi
        X_input = df_latest[features]
        df_latest['P0'] = model.predict(X_input)
        
        forecast_results.append(df_latest.copy())

    # 4. Gabungkan dan Simpan
    df_forecast = pd.concat(forecast_results, ignore_index=True)
    df_forecast.to_csv(OUTPUT_FORECAST, index=False)
    
    print(f"✅ [07] Peramalan selesai! Hasil disimpan di: {OUTPUT_FORECAST}")
    print(f"Tahun forecast: {latest_year+1} - {latest_year+5}")
    print(f"Rata-rata Prediksi Nasional {latest_year+1}: {df_forecast[df_forecast['Tahun']==latest_year+1]['P0'].mean():.2f}%")

if __name__ == '__main__':
    run_forecasting()