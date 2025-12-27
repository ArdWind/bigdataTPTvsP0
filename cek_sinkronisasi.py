import pandas as pd
import json
import os

# Path file Anda
DATA_PATH = 'cleaned_data/dataset_final_untuk_ml.csv'
MAP_DATA_PATH = 'Data_Source/indonesia_38_provinsi.geojson'

def cek_sinkronisasi():
    # 1. Ambil data dari CSV
    if not os.path.exists(DATA_PATH):
        print("CSV tidak ditemukan!")
        return
    df = pd.read_csv(DATA_PATH)
    nama_csv = set(df['Provinsi'].str.upper().unique())

    # 2. Ambil data dari GeoJSON
    if not os.path.exists(MAP_DATA_PATH):
        print("GeoJSON tidak ditemukan!")
        return
    with open(MAP_DATA_PATH, 'r') as f:
        geo = json.load(f)
    
    nama_geo = set()
    for feature in geo['features']:
        # Mengambil properti 'Propinsi' sesuai file ardian28
        nama_geo.add(feature['properties']['PROVINSI'].upper())

    # 3. Bandingkan
    tidak_ada_di_geo = nama_csv - nama_geo
    tidak_ada_di_csv = nama_geo - nama_csv

    print(f"Total Provinsi di CSV: {len(nama_csv)}")
    print(f"Total Provinsi di GeoJSON: {len(nama_geo)}")
    
    if tidak_ada_di_geo:
        print("\n⚠️ Nama di CSV yang TIDAK ADA di GeoJSON (Peta akan kosong):")
        for n in sorted(tidak_ada_di_geo): print(f"- {n}")
    else:
        print("\n✅ Semua nama di CSV cocok dengan GeoJSON!")

    if tidak_ada_di_csv:
        print("\nℹ️ Nama di GeoJSON yang tidak ada di CSV (Area akan abu-abu):")
        for n in sorted(tidak_ada_di_csv): print(f"- {n}")

if __name__ == "__main__":
    cek_sinkronisasi()