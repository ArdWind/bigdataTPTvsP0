import pandas as pd
import os

# --- KONFIGURASI PATH ---
MASTER_BPS_PATH = 'cleaned_data/data_master_ml.csv'
SENTIMENT_CSV_PATH = 'cleaned_data/sentiment_per_year.csv'
OUTPUT_FINAL = 'cleaned_data/dataset_final_untuk_ml.csv'

def integrate_final_dataset():
    print("🚀 [04] Memulai Integrasi Dataset Final...")
    
    if not os.path.exists(MASTER_BPS_PATH):
        print("🛑 Error: Data Master BPS tidak ditemukan.")
        return

    # 1. Load Data Master
    df = pd.read_csv(MASTER_BPS_PATH)
    
    # 2. Standarisasi Nama Kolom (PENTING!)
    # Kita ubah nama kolom yang mengandung 'TPT' atau 'GK' menjadi nama baku
    rename_map = {}
    for col in df.columns:
        if 'TPT' in col: rename_map[col] = 'TPT'
        if 'GK' in col or 'Garis_Kemiskinan' in col: rename_map[col] = 'Garis_Kemiskinan'
    
    df = df.rename(columns=rename_map)
    print(f"   Log: Kolom yang di-rename: {rename_map}")

    # 3. Gabungkan dengan Sentimen
    if os.path.exists(SENTIMENT_CSV_PATH):
        df_sent = pd.read_csv(SENTIMENT_CSV_PATH)
        df_final = pd.merge(df, df_sent, on='Tahun', how='left')
        
        # Jika kolom dari Skrip 03 bernama 'Score', ubah ke 'Sentimen_Global'
        if 'Score' in df_final.columns:
            df_final = df_final.rename(columns={'Score': 'Sentimen_Global'})
            
        df_final['Sentimen_Global'] = df_final['Sentimen_Global'].fillna(0)
    else:
        print("⚠️ Warning: Data sentimen tidak ditemukan, mengisi dengan 0.")
        df_final = df.copy()
        df_final['Sentimen_Global'] = 0

    # 4. Simpan Dataset Final
    df_final.to_csv(OUTPUT_FINAL, index=False)
    print(f"✅ [04] Dataset Final berhasil dibuat dengan kolom: {df_final.columns.tolist()}")

if __name__ == '__main__':
    integrate_final_dataset()