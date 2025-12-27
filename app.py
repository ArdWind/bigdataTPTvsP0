import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import subprocess
import joblib

# --- KONFIGURASI PATH ---
DATA_PATH = 'cleaned_data/dataset_final_untuk_ml.csv'
MAP_DATA_PATH = 'Data_Source/indonesia_simple.geojson'
MODEL_PATH = 'cleaned_data/model_kemiskinan_final.pkl'
FORECAST_PATH = 'cleaned_data/forecast_results.csv'

# --- SETTING HALAMAN ---
st.set_page_config(
    page_title="Big Data Poverty Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI HELPER ---
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Penyesuaian nama kolom agar sinkron dengan hasil skrip 04 & 05 terbaru
        # Menghapus '_Tahunan' agar sesuai dengan logika dataset_final_untuk_ml.csv
        for col in ['P0', 'P1', 'P2', 'TPT', 'Garis_Kemiskinan', 'Sentimen_Global']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Provinsi'] = df['Provinsi'].str.upper().str.strip()
        
        mapping_sinkron = {
            "DI YOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
            "JAKARTA": "DKI JAKARTA",
            "NUSA TENGGARA BARAT": "NUSATENGGARA BARAT",
            "NUSA TENGGARA TIMUR": "NUSATENGGARA TIMUR",
            "KEP. BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
            "KEP. RIAU": "KEPULAUAN RIAU",
            "SUMATRA BARAT": "SUMATERA BARAT",
            "SUMATRA SELATAN": "SUMATERA SELATAN",
            "SUMATRA UTARA": "SUMATERA UTARA",
            "PAPUA": "IRIAN JAYA TIMUR"
        }
        df['Provinsi'] = df['Provinsi'].replace(mapping_sinkron)
        return df
    return None

def load_geojson():
    if os.path.exists(MAP_DATA_PATH):
        try:
            with open(MAP_DATA_PATH, 'r') as f:
                return json.load(f)
        except: return None
    return None

def run_script(script_name):
    try:
        with st.spinner(f"Menjalankan {script_name}..."):
            result = subprocess.run(['python3', script_name], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"✅ {script_name} Berhasil dijalankan.")
                return True
            else:
                st.error(f"❌ Gagal menjalankan {script_name}")
                st.code(result.stderr)
                return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# --- LOAD DATA ---
df = load_data()

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("🚀 Menu Utama")
menu = st.sidebar.radio("Pilih Halaman:", 
    ["🏠 Dashboard", "📈 Prediksi Masa Depan", "🔮 Prediksi Manual", "⚙️ Control Panel"])

# ==========================================
# HALAMAN 1: DASHBOARD
# ==========================================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Analisis Kemiskinan")
    
    if df is not None:
        tab1, tab2 = st.tabs(["🇮🇩 Nasional: TPT vs P0", "📍 Detail Provinsi: P0, P1, P2 & Sentimen"])

# TAB 1: NASIONAL (EKONOMI & SENTIMEN)
        with tab1:
            st.title("🇮🇩 Analisis Strategis Nasional")
            
            with st.expander("ℹ️ Penjelasan Indikator & Sumber Data Nasional", expanded=False):
                st.markdown("""
                **1. Persentase Kemiskinan (P0):**
                * **Arti:** Persentase penduduk yang memiliki rata-rata pengeluaran per kapita per bulan di bawah Garis Kemiskinan.
                * **Satuan:** Persen (%).
                * **Hubungan:** Jika angka ini naik, berarti jumlah penduduk miskin bertambah.
                
                **2. Tingkat Pengangguran Terbuka (TPT):**
                * **Arti:** Penduduk yang sedang mencari kerja, mempersiapkan usaha, atau merasa tidak mungkin mendapat kerja.
                * **Satuan:** Persen (%).
                * **Hubungan:** Secara teori, kenaikan TPT cenderung diikuti kenaikan P0 karena berkurangnya pendapatan masyarakat.
                
                **3. Skor Sentimen Publik (TikTok):**
                * **Arti:** Hasil ekstraksi emosi masyarakat dari komentar media sosial terkait isu ekonomi menggunakan AI.
                * **Nilai:** Rentang -1 (Sangat Negatif) hingga +1 (Sangat Positif).
                * **Hubungan:** Mencerminkan persepsi dan keresahan masyarakat terhadap kondisi ekonomi secara real-time.
                
                **Sumber Data:** BPS Indonesia (P0 & TPT) dan Ekstraksi Metadata TikTok (Sentimen).
                """)

            # Agregasi Data (Penyesuaian nama kolom TPT)
            df_nat = df.groupby('Tahun').agg({'P0':'mean', 'TPT':'mean', 'Sentimen_Global':'mean'}).reset_index()

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("##### 📈 Tren Ekonomi (TPT vs P0)")
                fig_eco = go.Figure()
                fig_eco.add_trace(go.Scatter(x=df_nat['Tahun'], y=df_nat['P0'], name="P0 (%)", line=dict(color='firebrick', width=3)))
                fig_eco.add_trace(go.Scatter(x=df_nat['Tahun'], y=df_nat['TPT'], name="TPT (%)", line=dict(color='royalblue', width=3), yaxis="y2"))
                fig_eco.update_layout(
                    yaxis=dict(title="Kemiskinan (P0) %", title_font=dict(color="firebrick")),
                    yaxis2=dict(title="Pengangguran (TPT) %", title_font=dict(color="royalblue"), overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.1), height=400
                )
                st.plotly_chart(fig_eco, use_container_width=True)

            with col_g2:
                st.markdown("##### 📱 Analisis Sentimen Publik (2019-2025)")
                
                # Filter data sentimen
                df_sentimen = df_nat[df_nat['Tahun'] >= 2019].copy()
                
                if not df_sentimen.empty:
                    latest_sent = df_sentimen.iloc[-1]['Sentimen_Global']
                    latest_year = int(df_sentimen.iloc[-1]['Tahun'])
                    status_sentimen = "POSITIF" if latest_sent > 0 else "NEGATIF"
                    color_sentimen = "inverse" if latest_sent < 0 else "normal"

                    st.metric(
                        label=f"Skor Sentimen Tahun {latest_year}", 
                        value=f"{latest_sent:.2f}", 
                        delta=f"{status_sentimen} (Batas: -1 s/d +1)",
                        delta_color=color_sentimen
                    )
                    
                    fig_sent = px.bar(df_sentimen, x='Tahun', y='Sentimen_Global', 
                                      color='Sentimen_Global', 
                                      color_continuous_scale="RdYlGn",
                                      range_y=[-1, 1])
                    
                    fig_sent.update_layout(
                        height=300, 
                        yaxis_title="Skor (-1 ke 1)",
                        coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    fig_sent.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.warning("Data sentimen (2019-2025) belum tersedia di dataset.")

            st.markdown("---")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.caption("Data Historis Ekonomi Nasional")
                st.dataframe(df_nat[['Tahun', 'P0', 'TPT']], use_container_width=True)
            with col_t2:
                st.caption("Data Historis Sentimen Nasional")
                st.dataframe(df_nat[['Tahun', 'Sentimen_Global']], use_container_width=True)

# TAB 2: DETAIL PROVINSI
        with tab2:
            st.title("📍 Detail Indikator Ekonomi & Kemiskinan Provinsi")
            
            with st.expander("ℹ️ Pahami Istilah Kemiskinan dengan Mudah", expanded=False):
                st.markdown("""
                Agar tidak bingung, mari kita pahami indikator ini dengan perumpamaan **"Garis Batas"**:

                **1. Garis Kemiskinan (GK):** Ibarat sebuah **pagar**. Jika uang belanja Anda sebulan di bawah angka ini, Anda berada di dalam zona miskin. Jika di atasnya, Anda tidak miskin.

                **2. Persentase Kemiskinan (P0):** Menghitung **berapa banyak orang** yang berada di bawah pagar tersebut.

                **3. Indeks Kedalaman Kemiskinan (P1) - "Seberapa Jauh":** Bayangkan orang-orang yang sudah di bawah pagar tadi. P1 mengukur **seberapa jauh jarak mereka** dari pagar.  
                **4. Indeks Keparahan Kemiskinan (P2) - "Kesenjangan di Dalam":** P2 melihat **perbedaan nasib** di antara sesama orang miskin.  
                **5. Tingkat Pengangguran Terbuka (TPT):** Persentase orang yang tidak punya pekerjaan. Ini adalah "pintu masuk" utama menuju kemiskinan.
                """)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter Detail Provinsi")
            sel_prov = st.sidebar.selectbox("Pilih Provinsi", sorted(df['Provinsi'].unique()))
            sel_year = st.sidebar.slider("Pilih Tahun Peta", int(df['Tahun'].min()), int(df['Tahun'].max()), int(df['Tahun'].max()))
            
            df_prov = df[df['Provinsi'] == sel_prov].sort_values('Tahun')

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.subheader(f"📈 Tren Indikator: {sel_prov}")
                # Penyesuaian nama kolom TPT
                fig_pct = px.line(df_prov, x='Tahun', y=['P0', 'P1', 'P2', 'TPT'], 
                                markers=True, title="Indikator Persentase (%)")
                st.plotly_chart(fig_pct, use_container_width=True)

                # Penyesuaian nama kolom Garis_Kemiskinan
                fig_gk = px.area(df_prov, x='Tahun', y='Garis_Kemiskinan', 
                                title="Tren Garis Kemiskinan (Rp)",
                                color_discrete_sequence=['#228B22'])
                fig_gk.update_layout(yaxis_title="Rupiah (Rp)")
                st.plotly_chart(fig_gk, use_container_width=True)
            
            with col_r:
                st.subheader(f"🗺️ Peta Sebaran P0 - {sel_year}")
                geo = load_geojson()
                if geo:
                    df_map = df[df['Tahun'] == sel_year]
                    fig_map = px.choropleth(df_map, geojson=geo, locations='Provinsi', 
                                            featureidkey="properties.Propinsi",
                                            color='P0', color_continuous_scale="YlOrRd")
                    fig_map.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_map, use_container_width=True)
                
                st.info(f"**Info {sel_prov} ({sel_year}):** \n"
                        f"P0: {df_map[df_map['Provinsi']==sel_prov]['P0'].values[0]}%  \n"
                        f"GK: Rp {df_map[df_map['Provinsi']==sel_prov]['Garis_Kemiskinan'].values[0]:,.0f}")

            st.subheader(f"📋 Tabel Data Detail: {sel_prov}")
            # Penyesuaian nama kolom sesuai dataset baru
            cols_to_show = ['Tahun', 'Provinsi', 'P0', 'P1', 'P2', 'TPT', 'Garis_Kemiskinan']
            st.dataframe(df_prov[cols_to_show], use_container_width=True)

# ==========================================
# HALAMAN 2: PREDIKSI MASA DEPAN
# ==========================================
elif menu == "📈 Prediksi Masa Depan":
    st.title("📈 Proyeksi Kemiskinan 2026-2027")
    if os.path.exists('cleaned_data/data_forecasting_2026_2027.csv'):
        df_forecast = pd.read_csv('cleaned_data/data_forecasting_2026_2027.csv')
        df_hist = df.groupby('Tahun')['P0'].mean().reset_index()
        df_fore = df_forecast.groupby('Tahun')['P0'].mean().reset_index()
        df_all = pd.concat([df_hist, df_fore])
        
        fig = px.line(df_all, x='Tahun', y='P0', title="Prediksi P0 Nasional", markers=True)
        fig.add_vrect(x0=2025.5, x1=2027.5, fillcolor="green", opacity=0.1, annotation_text="Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_fore)
    else:
        st.warning("Data forecasting belum tersedia.")

# ==========================================
# HALAMAN 3: PREDIKSI MANUAL (DIGABUNG AGAR TIDAK DUPLIKAT)
# ==========================================
elif menu == "🔮 Prediksi Manual":
    st.title("🔮 Simulasi Prediksi Manual")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                p0_l = st.number_input("P0 Tahun Lalu", value=9.0)
                tpt = st.number_input("TPT (%)", value=5.0)
                gk = st.number_input("Garis Kemiskinan", value=500000.0)
            with c2:
                p1 = st.number_input("P1", value=1.5)
                p2 = st.number_input("P2", value=0.4)
                sent = st.number_input("Skor Sentimen (-1 s/d 1)", value=0.0)
            
            if st.form_submit_button("Prediksi Sekarang"):
                # Urutan sesuai skrip 05: P0_Lag1, TPT, Garis_Kemiskinan, Sentimen_Global, P1, P2
                features = [[p0_l, tpt, gk, sent, p1, p2]]
                res = model.predict(features)
                st.success(f"### Hasil Prediksi P0: {res[0]:.2f}%")
    else: st.error("Model .pkl tidak ditemukan.")

# ==========================================
# HALAMAN 4: CONTROL PANEL
# ==========================================
else:
    st.title("⚙️ Control Panel")
    st.info("Gunakan tombol di bawah untuk memproses ulang seluruh alur data.")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("1️⃣ Clean Data"): run_script('01_data_ingestion_cleaning.py')
        if st.button("2️⃣ Sentiment Analysis"): run_script('03_sentiment_processor.py')
        if st.button("3️⃣ Final Integration"): run_script('04_final_integration.py')
    with col_c2:
        if st.button("4️⃣ Latih Model (Model Training)"): run_script('05_machine_learning_model.py')
        if st.button("5️⃣ Jalankan Uji Prediksi"): run_script('06_uji_prediksi.py')
        if st.button("6️⃣ Jalankan Forecasting"): run_script('07_forecasting.py')