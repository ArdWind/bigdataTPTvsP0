import joblib
import pandas as pd

# 1. Muat model yang sudah disimpan
try:
    model = joblib.load('cleaned_data/model_kemiskinan_final.pkl')
    print("✅ Model berhasil dimuat.")
except:
    print("❌ Model tidak ditemukan. Jalankan script 05 terlebih dahulu.")
    exit()

# 2. Input Data Manual (Simulasi)
print("\n--- Simulasi Prediksi Kemiskinan ---")
p0_lalu = float(input("Masukkan P0 Tahun Lalu (contoh 8.15): "))
tpt = float(input("Masukkan angka TPT (contoh 5.0): "))
gk = float(input("Masukkan Garis Kemiskinan (contoh 600000): "))
sentimen = -0.0129  # Kita gunakan nilai dari TikTok yang sudah ada
p1 = float(input("Masukkan Indeks Kedalaman (P1) (contoh 1.2): "))
p2 = float(input("Masukkan Indeks Keparahan (P2) (contoh 0.3): "))

# 3. Masukkan ke DataFrame
data_simulasi = pd.DataFrame([[p0_lalu, tpt, gk, sentimen, p1, p2]], 
                            columns=['P0_Lag1', 'TPT_Tahunan', 'GK_Tahunan', 'Sentimen_Global', 'P1', 'P2'])

# 4. Prediksi
hasil = model.predict(data_simulasi)

print("\n=================================================")
print(f"HASIL PREDIKSI P0: {hasil[0]:.2f}%")
print("=================================================")