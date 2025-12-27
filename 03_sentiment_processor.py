import pandas as pd
import os

PATH_KONTEN = '/home/ardwind/Project/BigDataProject/Data_Source/sosialresponse/kontentiktok.csv'
PATH_KOMEN = '/home/ardwind/Project/BigDataProject/Data_Source/sosialresponse/komentiktok.csv'
OUTPUT_DIR = 'cleaned_data/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

KATA_POSITIF = ['daftar', 'minat', 'siap', 'bantu', 'upgrade', 'lirik', 'semangat', 'solusi', 'berhasil', 'kerja', 'terima']
KATA_NEGATIF = ['phk', 'susah', 'nganggur', 'belum', 'habis', 'sulit', 'menjerit', 'penjilat', 'parah', 'gagal', 'miskin']

def get_sentiment_label(text):
    if pd.isna(text): return 0
    text = str(text).lower()
    score = 0
    for kata in KATA_POSITIF:
        if kata in text: score += 1
    for kata in KATA_NEGATIF:
        if kata in text: score -= 1
    return 1 if score > 0 else (-1 if score < 0 else 0)

def process_tiktok_data():
    print("🚀 [03] Memulai Pemrosesan Sentimen...")
    try:
        df_konten = pd.read_csv(PATH_KONTEN, skiprows=4).dropna(how='all', axis=1)
        df_komen = pd.read_csv(PATH_KOMEN, skiprows=1).dropna(how='all', axis=1)
        
        df_konten.columns = df_konten.columns.str.strip()
        df_komen.columns = df_komen.columns.str.strip()
    except Exception as e:
        print(f"🛑 Error: {e}")
        return

    df1 = df_konten[['Teks Konten', 'Tahun']].copy()
    df2 = df_komen[['Teks Konten', 'Tahun']].copy()
    df_combined = pd.concat([df1, df2], ignore_index=True)

    df_combined['Score'] = df_combined['Teks Konten'].apply(get_sentiment_label)
    
    df_combined['Tahun'] = pd.to_numeric(df_combined['Tahun'], errors='coerce')
    df_combined = df_combined.dropna(subset=['Tahun'])
    df_combined['Tahun'] = df_combined['Tahun'].astype(int)

    df_yearly = df_combined.groupby('Tahun')['Score'].mean().reset_index()
    
    output_path = os.path.join(OUTPUT_DIR, 'sentiment_per_year.csv')
    df_yearly.to_csv(output_path, index=False)
    
    print(f"✅ [03] Berhasil menyimpan sentimen tahunan ke: {output_path}")
    print(df_yearly)

if __name__ == '__main__':
    process_tiktok_data()