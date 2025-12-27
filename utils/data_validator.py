"""
Data Validator Module
Validasi format data untuk upload BPS dan TikTok
"""

import pandas as pd
import numpy as np

def validate_tpt_data(df):
    """Validasi format data TPT (Tingkat Pengangguran Terbuka)"""
    required_columns = ['Provinsi', 'TPT_Feb', 'TPT_Aug', 'TPT_Tahunan_Source']
    errors = []
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom yang hilang: {', '.join(missing_cols)}")
    
    # Check data types
    if 'Provinsi' in df.columns and not df['Provinsi'].dtype == 'object':
        errors.append("Kolom 'Provinsi' harus berisi teks")
    
    # Check numeric columns
    numeric_cols = ['TPT_Feb', 'TPT_Aug', 'TPT_Tahunan_Source']
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                errors.append(f"Kolom '{col}' harus berisi angka")
    
    return len(errors) == 0, errors

def validate_p_data(df, data_type='P0'):
    """Validasi format data P0/P1/P2"""
    required_columns = ['Provinsi', f'{data_type}_Mar', f'{data_type}_Sep', f'{data_type}_Tahunan_Source']
    errors = []
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom yang hilang: {', '.join(missing_cols)}")
    
    # Check data types
    if 'Provinsi' in df.columns and not df['Provinsi'].dtype == 'object':
        errors.append("Kolom 'Provinsi' harus berisi teks")
    
    # Check numeric columns
    numeric_cols = [f'{data_type}_Mar', f'{data_type}_Sep', f'{data_type}_Tahunan_Source']
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                errors.append(f"Kolom '{col}' harus berisi angka")
    
    return len(errors) == 0, errors

def validate_gk_data(df):
    """Validasi format data Garis Kemiskinan"""
    required_columns = ['Provinsi', 'GK_Kota_Tahunan', 'GK_Desa_Tahunan']
    errors = []
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom yang hilang: {', '.join(missing_cols)}")
    
    # Check data types
    if 'Provinsi' in df.columns and not df['Provinsi'].dtype == 'object':
        errors.append("Kolom 'Provinsi' harus berisi teks")
    
    # Check numeric columns
    numeric_cols = ['GK_Kota_Tahunan', 'GK_Desa_Tahunan']
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                errors.append(f"Kolom '{col}' harus berisi angka")
    
    return len(errors) == 0, errors

def validate_tiktok_content(df):
    """Validasi format konten TikTok"""
    required_columns = ['ID Unik', 'Jenis Konten', 'Teks Konten', 'Tahun', 'Platform']
    errors = []
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom yang hilang: {', '.join(missing_cols)}")
    
    # Check ID Unik format
    if 'ID Unik' in df.columns:
        if not all(df['ID Unik'].astype(str).str.startswith('T')):
            errors.append("ID Unik harus dimulai dengan huruf 'T'")
    
    # Check Tahun
    if 'Tahun' in df.columns:
        try:
            years = pd.to_numeric(df['Tahun'], errors='coerce')
            if years.isna().any():
                errors.append("Kolom 'Tahun' harus berisi angka")
            elif (years < 2019).any() or (years > 2030).any():
                errors.append("Tahun harus antara 2019-2030")
        except:
            errors.append("Kolom 'Tahun' tidak valid")
    
    # Check Platform
    if 'Platform' in df.columns:
        if not all(df['Platform'] == 'TikTok'):
            errors.append("Platform harus 'TikTok'")
    
    return len(errors) == 0, errors

def validate_tiktok_comment(df):
    """Validasi format komentar TikTok"""
    required_columns = ['ID Unik', 'Jenis Konten', 'Teks Konten', 'Tahun', 'Platform', 'Balasan ID Komentar']
    errors = []
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom yang hilang: {', '.join(missing_cols)}")
    
    # Check ID Unik format
    if 'ID Unik' in df.columns:
        if not all(df['ID Unik'].astype(str).str.startswith('T')):
            errors.append("ID Unik harus dimulai dengan huruf 'T'")
    
    # Check Jenis Konten
    if 'Jenis Konten' in df.columns:
        if not all(df['Jenis Konten'] == 'Komentar Balasan'):
            errors.append("Jenis Konten harus 'Komentar Balasan'")
    
    # Check Tahun
    if 'Tahun' in df.columns:
        try:
            years = pd.to_numeric(df['Tahun'], errors='coerce')
            if years.isna().any():
                errors.append("Kolom 'Tahun' harus berisi angka")
            elif (years < 2019).any() or (years > 2030).any():
                errors.append("Tahun harus antara 2019-2030")
        except:
            errors.append("Kolom 'Tahun' tidak valid")
    
    # Check Platform
    if 'Platform' in df.columns:
        if not all(df['Platform'] == 'TikTok'):
            errors.append("Platform harus 'TikTok'")
    
    return len(errors) == 0, errors

def get_data_summary(df):
    """Mendapatkan ringkasan data untuk preview"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Add year range if Tahun column exists
    if 'Tahun' in df.columns:
        try:
            years = pd.to_numeric(df['Tahun'], errors='coerce').dropna()
            summary['year_range'] = f"{int(years.min())} - {int(years.max())}"
            summary['years'] = sorted(years.unique().astype(int).tolist())
        except:
            pass
    
    # Add province count if Provinsi column exists
    if 'Provinsi' in df.columns:
        summary['province_count'] = df['Provinsi'].nunique()
        summary['provinces'] = sorted(df['Provinsi'].unique().tolist())
    
    return summary
