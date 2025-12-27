# ====================================================
# FINAL REFACTORING (01_data_ingestion_cleaning.py)
# ====================================================

import pandas as pd
import os
import glob
import numpy as np

# --- Konfigurasi Direktori ---
CONFIG = {
    'TPT_DIR': 'Data_Source/Tingkat Pengangguran Terbuka/',
    'GK_DIR': 'Data_Source/Persentase Penduduk Miskin/Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah/',
    'P_DIR_MAP': {
        'P0': 'Data_Source/Persentase Penduduk Miskin/(P0) Menurut Provinsi/',
        'P1': 'Data_Source/Persentase Penduduk Miskin/(P1) Menurut Provinsi/',
        'P2': 'Data_Source/Persentase Penduduk Miskin/(P2) Menurut Provinsi/'
    },
    'CLEANED_DIR': 'cleaned_data/',
    'MIN_TAHUN': 2013 # Kritis untuk memastikan kelengkapan feature GK dan TPT
}

# Pastikan folder cleaned_data ada
os.makedirs(CONFIG['CLEANED_DIR'], exist_ok=True)

# ====================================================
# FUNGSI UTILITAS
# ====================================================

def standardize_province_names(df):
    """Fungsi untuk menyeragamkan nama provinsi agar merge berhasil."""
    if 'Provinsi' in df.columns:
        df['Provinsi'] = df['Provinsi'].astype(str).str.upper().str.strip()
        df['Provinsi'] = df['Provinsi'].str.replace(r'\s+', ' ', regex=True)
        
        # Hapus KATA KUNCI geografis yang mengganggu
        df['Provinsi'] = df['Provinsi'].str.replace('DAERAH ISTIMEWA', '', regex=False)
        df['Provinsi'] = df['Provinsi'].str.replace('DKI', '', regex=False)
        
        # Standardisasi Nama Pulau
        df['Provinsi'] = df['Provinsi'].str.replace('SUMATERA', 'SUMATRA', regex=False)
        df['Provinsi'] = df['Provinsi'].str.replace('KALIMANTAN', 'KALIMATAN', regex=False)
        df['Provinsi'] = df['Provinsi'].str.replace('SULAWESI', 'SULAWESI', regex=False)
        
        # Final Cleaning dan Trim
        df['Provinsi'] = df['Provinsi'].str.strip()
        df['Provinsi'] = df['Provinsi'].str.replace(r'\s+', ' ', regex=True)
        
    return df

def clean_and_impute_semesters(df_clean, data_type, prefix_map, target_col):
    """Melakukan pembersihan numerik dan imputasi Tahunan dari Semesteran."""
    
    mar_col = prefix_map['Mar']
    sep_col = prefix_map['Sep']
    tahunan_col = prefix_map['Tahunan']
    
    # 1. Konversi Numerik Awal (membersihkan -, dll.)
    cols_to_num = [mar_col, sep_col, tahunan_col]
    for col in cols_to_num:
        df_clean[col] = df_clean[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Salin nilai Tahunan yang sudah ada ke kolom target
    df_clean[target_col] = df_clean[tahunan_col]

    # 2. Imputasi Logika
    
    # Rata-rata Mar & Sep (jika Tahunan kosong, dan keduanya ada)
    mask_avg = df_clean[target_col].isna() & df_clean[mar_col].notna() & df_clean[sep_col].notna()
    df_clean.loc[mask_avg, target_col] = df_clean[[mar_col, sep_col]].mean(axis=1)

    # Gunakan nilai Mar
    mask_mar = df_clean[target_col].isna() & df_clean[mar_col].notna() & df_clean[sep_col].isna()
    df_clean.loc[mask_mar, target_col] = df_clean[mar_col]

    # Gunakan nilai Sep
    mask_sep = df_clean[target_col].isna() & df_clean[mar_col].isna() & df_clean[sep_col].notna()
    df_clean.loc[mask_sep, target_col] = df_clean[sep_col]
    
    # Pembulatan
    df_clean[target_col] = df_clean[target_col].round(2)
    
    return df_clean

# ====================================================
# STEP 1: PENGGABUNGAN & PEMBERSIHAN TPT
# ====================================================

def process_tpt_data():
    list_df_tpt = []
    all_files = glob.glob(os.path.join(CONFIG['TPT_DIR'], "*.csv"))
    if not all_files: return pd.DataFrame()

    print(f"Ditemukan {len(all_files)} file CSV TPT. Memulai penggabungan dengan imputasi nilai...")
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename, header=3)
            year_str = os.path.basename(filename).split(',')[-1].replace('.csv', '').strip()
            df['Tahun'] = int(year_str) 
            
            # Penamaan Kolom TPT
            df_clean = df[df[df.columns[0]].notna()].copy()
            df_clean = df_clean.rename(columns={
                df_clean.columns[0]: 'Provinsi',
                df_clean.columns[1]: 'TPT_Feb',
                df_clean.columns[2]: 'TPT_Aug',
                df_clean.columns[3]: 'TPT_Tahunan_Source'
            })
            
            # Pembersihan baris non-data
            df_clean = df_clean[~df_clean['Provinsi'].str.contains('INDONESIA|RATA-RATA|TOTAL', na=False, case=False)]
            
            # Imputasi
            prefix_map = {'Mar': 'TPT_Feb', 'Sep': 'TPT_Aug', 'Tahunan': 'TPT_Tahunan_Source'}
            df_clean = clean_and_impute_semesters(df_clean, 'TPT', prefix_map, 'TPT_Tahunan')
            
            list_df_tpt.append(df_clean[['Provinsi', 'TPT_Tahunan', 'Tahun']])
            
        except Exception as e:
            print(f"Gagal memproses file {filename}: {e}")

    df_tpt_master = pd.concat(list_df_tpt, ignore_index=True) if list_df_tpt else pd.DataFrame()
    df_tpt_master.dropna(subset=['TPT_Tahunan'], inplace=True)
    return standardize_province_names(df_tpt_master)

# ====================================================
# STEP 2: PENGGABUNGAN & PEMBERSIHAN P0, P1, P2
# ====================================================

def process_p_data(data_type, data_dir):
    list_df = []
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files: return pd.DataFrame()
    
    print(f"\nDitemukan {len(all_files)} file CSV {data_type}. Memulai penggabungan dan imputasi...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename, header=3)
            year_str = os.path.basename(filename).split(',')[-1].replace('.csv', '').strip()
            df['Tahun'] = int(year_str) 
            
            # Mapping Kolom Px (Kelompok Jumlah: 7, 8, 9)
            df_clean = df[df[df.columns[0]].notna()].copy()
            df_clean = df_clean.rename(columns={
                df_clean.columns[0]: 'Provinsi',
                df_clean.columns[7]: f'{data_type}_Mar',
                df_clean.columns[8]: f'{data_type}_Sep',
                df_clean.columns[9]: f'{data_type}_Tahunan_Source'
            })
            
            # Pembersihan baris non-data
            df_clean = df_clean[~df_clean['Provinsi'].str.contains('INDONESIA|RATA-RATA|TOTAL', na=False, case=False)]
            
            # Imputasi
            prefix_map = {'Mar': f'{data_type}_Mar', 'Sep': f'{data_type}_Sep', 'Tahunan': f'{data_type}_Tahunan_Source'}
            df_clean = clean_and_impute_semesters(df_clean, data_type, prefix_map, data_type)
            
            list_df.append(df_clean[['Provinsi', data_type, 'Tahun']])
            
        except Exception as e:
            print(f"Gagal memproses file {filename}: {e}")

    df_master = pd.concat(list_df, ignore_index=True) if list_df else pd.DataFrame()
    df_master.dropna(subset=[data_type], inplace=True)
    return standardize_province_names(df_master)

# ====================================================
# STEP 3: PENGGABUNGAN & PEMBERSIHAN GARIS KEMISKINAN (GK)
# ====================================================

def process_gk_data_final():
    list_df = []
    all_files = glob.glob(os.path.join(CONFIG['GK_DIR'], "*.csv"))
    if not all_files: return pd.DataFrame()
    
    print(f"\nDitemukan {len(all_files)} file CSV GK. Memulai penggabungan dengan imputasi...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename, header=3)
            year_str = os.path.basename(filename).split(',')[-1].replace('.csv', '').strip()
            df['Tahun'] = int(year_str) 
            
            # Mapping Kolom GK: [0: Provinsi] [1-3: Perkotaan] [4-6: Perdesaan]
            df_clean = df[df[df.columns[0]].notna()].copy()
            df_clean = df_clean.rename(columns={
                df_clean.columns[0]: 'Provinsi',
                df_clean.columns[3]: 'GK_Kota_Tahunan',
                df_clean.columns[6]: 'GK_Desa_Tahunan'
            })

            # Pembersihan baris non-data
            df_clean = df_clean[~df_clean['Provinsi'].str.contains('INDONESIA|RATA-RATA|TOTAL', na=False, case=False)]
            
            # --- Imputasi Nilai Tahunan (Perkotaan & Perdesaan) ---
            
            # Kita hanya mengambil kolom Tahunan yang sudah diimputasi/dihitung pada proses clean_and_impute_semesters (yang kita pindahkan ke bawah)
            
            # Mapping ulang kolom yang belum di-rename untuk diimputasi
            df_clean['GK_Kota_Mar'] = df[df.columns[1]]
            df_clean['GK_Kota_Sep'] = df[df.columns[2]]
            df_clean['GK_Desa_Mar'] = df[df.columns[4]]
            df_clean['GK_Desa_Sep'] = df[df.columns[5]]

            for area in ['Kota', 'Desa']:
                target = f'GK_{area}_Tahunan'
                mar = f'GK_{area}_Mar'
                sep = f'GK_{area}_Sep'
                
                prefix_map = {'Mar': mar, 'Sep': sep, 'Tahunan': target}
                
                # Kita ubah cara panggil agar sesuai dengan fungsi utilitas clean_and_impute_semesters
                df_clean = clean_and_impute_semesters(df_clean, f'GK_{area}', prefix_map, target)
                
            # --- HITUNG RATA-RATA FINAL PROVINSI (Logika Anda) ---
            
            df_clean['GK_Tahunan'] = df_clean[['GK_Kota_Tahunan', 'GK_Desa_Tahunan']].mean(axis=1).round(2)
            
            list_df.append(df_clean[['Provinsi', 'GK_Tahunan', 'Tahun']])
            
        except Exception as e:
            print(f"Gagal memproses file {filename}: {e}")

    df_master = pd.concat(list_df, ignore_index=True) if list_df else pd.DataFrame()
    df_master.dropna(subset=['GK_Tahunan'], inplace=True)
    return standardize_province_names(df_master)

# ====================================================
# STEP 4: MENGGABUNGKAN SEMUA DATA BPS (MASTER ML)
# ====================================================

def create_master_dataframe():
    
    cleaned_dir = CONFIG['CLEANED_DIR']
    
    # Memuat data yang sudah dibersihkan
    df_tpt = pd.read_csv(os.path.join(cleaned_dir, 'tpt_master_final.csv'))
    df_p0 = pd.read_csv(os.path.join(cleaned_dir, 'P0_master_final.csv'))
    df_p1 = pd.read_csv(os.path.join(cleaned_dir, 'P1_master_final.csv'))
    df_p2 = pd.read_csv(os.path.join(cleaned_dir, 'P2_master_final.csv'))
    df_gk = pd.read_csv(os.path.join(cleaned_dir, 'gk_master_final.csv'))

    # --- KONVERSI EKSPILISIT KOLOM TAHUN ---
    for df in [df_tpt, df_p0, df_p1, df_p2, df_gk]:
        df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
        df.dropna(subset=['Tahun'], inplace=True)
        df['Tahun'] = df['Tahun'].astype(int) 
    
    # Standardisasi nama provinsi
    df_tpt = standardize_province_names(df_tpt)
    df_p0 = standardize_province_names(df_p0)
    df_p1 = standardize_province_names(df_p1)
    df_p2 = standardize_province_names(df_p2)
    df_gk = standardize_province_names(df_gk)

    # --- FILTER TAHUN KRITIS ---
    MIN_TAHUN = CONFIG['MIN_TAHUN'] 
    df_p0 = df_p0[df_p0['Tahun'] >= MIN_TAHUN] 
    
    print(f"\n[INFO] Data Master ML difilter: Tahun >= {MIN_TAHUN}. Baris P0 (basis) sekarang: {len(df_p0)}")
    
    # 1. Menggabungkan data P0, P1, P2 (Basis: P0 yang sudah difilter)
    df_master = df_p0.merge(df_p1, on=['Provinsi', 'Tahun'], how='inner')
    df_master = df_master.merge(df_p2, on=['Provinsi', 'Tahun'], how='inner')
    
    # 2. Menggabungkan dengan GK dan TPT
    df_master = df_master.merge(df_gk, on=['Provinsi', 'Tahun'], how='left')
    df_master = df_master.merge(df_tpt, on=['Provinsi', 'Tahun'], how='left')
    
    # 3. Menambahkan Feature Lag P0 
    df_master.sort_values(by=['Provinsi', 'Tahun'], inplace=True)
    df_master['P0_Lag1'] = df_master.groupby('Provinsi')['P0'].shift(1)
    
    # 4. Filter Data Master (Menghapus baris dengan nilai hilang/NaN)
    df_master.dropna(inplace=True) 
    
    # 5. Pilih dan atur ulang kolom final
    df_master = df_master[['Provinsi', 'Tahun', 'P0', 'P0_Lag1', 'TPT_Tahunan', 'GK_Tahunan', 'P1', 'P2']]

    return df_master

# ====================================================
# MAIN EXECUTION
# ====================================================

def main():
    
    # Step 1
    df_tpt_final = process_tpt_data()
    if not df_tpt_final.empty:
        final_path = os.path.join(CONFIG['CLEANED_DIR'], 'tpt_master_final.csv')
        df_tpt_final.to_csv(final_path, index=False)
        print(f"\n[DONE] TPT Master (Rows: {len(df_tpt_final)}) disimpan.")

    # Step 2
    for key, directory in CONFIG['P_DIR_MAP'].items():
        df_cleaned = process_p_data(key, directory)
        if not df_cleaned.empty:
            final_path = os.path.join(CONFIG['CLEANED_DIR'], f'{key}_master_final.csv')
            df_cleaned.to_csv(final_path, index=False)
            print(f"[DONE] {key} Master (Rows: {len(df_cleaned)}) disimpan.")

    # Step 3
    df_gk_final = process_gk_data_final()
    if not df_gk_final.empty:
        final_path = os.path.join(CONFIG['CLEANED_DIR'], 'gk_master_final.csv')
        df_gk_final.to_csv(final_path, index=False)
        print(f"[DONE] GK Master (Rows: {len(df_gk_final)}) disimpan.")


    # Step 4: Membuat Data Master ML
    df_master_ml = create_master_dataframe()

    print("\n=================================================")
    if not df_master_ml.empty:
        print("DATA MASTER ML SIAP (SEMUA DATA BPS TERGABUNG)")
        print(f"Total Baris Data Master ML: {len(df_master_ml)}")
        print("=================================================")
        print(df_master_ml.head())
        
        final_path = os.path.join(CONFIG['CLEANED_DIR'], 'data_master_ml.csv')
        df_master_ml.to_csv(final_path, index=False)
        print(f"\n[FINAL] Data Master ML berhasil disimpan ke '{final_path}'")
    else:
        print("PENTING: Penggabungan Data Master ML GAGAL atau menghasilkan DataFrame kosong. Cek inkonsistensi nama Provinsi atau rentang Tahun.")
    print("=================================================")

if __name__ == '__main__':
    main()