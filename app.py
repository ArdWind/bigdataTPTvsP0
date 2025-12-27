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
                
                # Filter data sentimen - pastikan data 2025 termasuk
                df_sentimen = df_nat[df_nat['Tahun'] >= 2019].copy()
                # Sort by year to ensure latest year is at the end
                df_sentimen = df_sentimen.sort_values('Tahun')
                
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
                df_display = df_nat[['Tahun', 'P0', 'TPT']].copy()
                df_display['P0'] = df_display['P0'].round(2)
                df_display['TPT'] = df_display['TPT'].round(2)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            with col_t2:
                st.caption("Data Historis Sentimen Nasional")
                df_display = df_nat[['Tahun', 'Sentimen_Global']].copy()
                df_display['Sentimen_Global'] = df_display['Sentimen_Global'].round(2)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

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

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader(f"📈 Tren Indikator: {sel_prov}")
                fig_pct = px.line(df_prov, x='Tahun', y=['P0', 'P1', 'P2', 'TPT'], 
                                markers=True, title="Indikator Persentase (%)")
                st.plotly_chart(fig_pct, use_container_width=True)
            
            with col_r:
                st.subheader(f"💰 Tren Garis Kemiskinan")
                fig_gk = px.area(df_prov, x='Tahun', y='Garis_Kemiskinan', 
                                title="Garis Kemiskinan (Rp)",
                                color_discrete_sequence=['#228B22'])
                fig_gk.update_layout(yaxis_title="Rupiah (Rp)")
                st.plotly_chart(fig_gk, use_container_width=True)
            
            st.subheader(f"🗺️ Peta Sebaran P0 - {sel_year}")
            geo = load_geojson()
            if geo:
                df_map = df[df['Tahun'] == sel_year]
                fig_map = px.choropleth(df_map, geojson=geo, locations='Provinsi', 
                                        featureidkey="properties.Propinsi",
                                        color='P0', color_continuous_scale="YlOrRd",
                                        height=400)
                fig_map.update_geos(
                    fitbounds="locations", 
                    visible=False,
                    projection_scale=1.8,
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

            st.subheader(f"📋 Tabel Data Detail: {sel_prov}")
            cols_to_show = ['Tahun', 'Provinsi', 'P0', 'P1', 'P2', 'TPT', 'Garis_Kemiskinan']
            df_display = df_prov[cols_to_show].copy()
            for col in ['P0', 'P1', 'P2', 'TPT']:
                df_display[col] = df_display[col].round(2)
            df_display['Garis_Kemiskinan'] = df_display['Garis_Kemiskinan'].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

elif menu == "📈 Prediksi Masa Depan":
    st.title("📈 Proyeksi Kemiskinan 5 Tahun Kedepan")
    
    forecast_file = 'cleaned_data/data_forecasting_2026_2027.csv'
    
    if os.path.exists(forecast_file):
        df_forecast = pd.read_csv(forecast_file)
        df_hist = df.groupby('Tahun')['P0'].mean().reset_index()
        df_fore = df_forecast.groupby('Tahun')['P0'].mean().reset_index()
        df_all = pd.concat([df_hist, df_fore])
        
        last_hist_year = df_hist['Tahun'].max()
        
        fig = px.line(df_all, x='Tahun', y='P0', title="Prediksi P0 Nasional (5 Tahun Kedepan)", markers=True)
        fig.add_vrect(x0=last_hist_year + 0.5, x1=df_all['Tahun'].max() + 0.5, 
                      fillcolor="green", opacity=0.1, annotation_text="Forecast")
        st.plotly_chart(fig, use_container_width=True)
        
        df_display = df_fore.copy()
        df_display['P0'] = df_display['P0'].round(2)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
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
    st.title("⚙️ Control Panel - Data Management")
    
    # Import utils
    from utils import (
        validate_tpt_data, validate_p_data, validate_gk_data,
        validate_tiktok_content, validate_tiktok_comment,
        get_data_summary, DataProcessor, check_data_files
    )
    
    # Tabs untuk berbagai fungsi
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Data BPS", 
        "📱 Upload Data TikTok", 
        "🔄 Re-Process Data",
        "📊 Data Status"
    ])
    
    # ========== TAB 1: UPLOAD DATA BPS ==========
    with tab1:
        st.header("📤 Upload Data BPS")
        st.markdown("Upload file CSV untuk data BPS (TPT, P0, P1, P2, GK)")
        
        data_type = st.selectbox(
            "Pilih Jenis Data:",
            ["TPT (Tingkat Pengangguran Terbuka)", 
             "P0 (Persentase Penduduk Miskin)",
             "P1 (Indeks Kedalaman Kemiskinan)",
             "P2 (Indeks Keparahan Kemiskinan)",
             "GK (Garis Kemiskinan)"]
        )
        
        year = st.number_input("Tahun Data:", min_value=2000, max_value=2030, value=2025, step=1)
        
        uploaded_file = st.file_uploader(
            "Upload file CSV", 
            type=['csv'],
            help="Format file harus sesuai dengan template BPS"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_upload = pd.read_csv(uploaded_file, header=3)
                
                st.success(f"✅ File berhasil dibaca: {uploaded_file.name}")
                
                # Validate data
                data_code = data_type.split()[0]  # TPT, P0, P1, P2, atau GK
                
                if data_code == "TPT":
                    is_valid, errors = validate_tpt_data(df_upload)
                elif data_code in ["P0", "P1", "P2"]:
                    is_valid, errors = validate_p_data(df_upload, data_code)
                else:  # GK
                    is_valid, errors = validate_gk_data(df_upload)
                
                if is_valid:
                    st.success("✅ Validasi data berhasil!")
                    
                    # Preview data
                    with st.expander("👁️ Preview Data", expanded=True):
                        st.dataframe(df_upload.head(10), use_container_width=True)
                        summary = get_data_summary(df_upload)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Baris", summary['total_rows'])
                        with col2:
                            st.metric("Total Kolom", summary['total_columns'])
                        with col3:
                            if 'province_count' in summary:
                                st.metric("Jumlah Provinsi", summary['province_count'])
                    
                    # Save button
                    if st.button("💾 Simpan Data", type="primary"):
                        # Determine save path
                        if data_code == "TPT":
                            save_dir = "Data_Source/Tingkat Pengangguran Terbuka/"
                            filename = f"Tingkat Pengangguran Terbuka Menurut Provinsi, {year}.csv"
                        elif data_code in ["P0", "P1", "P2"]:
                            save_dir = f"Data_Source/Persentase Penduduk Miskin/({data_code}) Menurut Provinsi/"
                            if data_code == "P0":
                                filename = f"Persentase Penduduk Miskin (P0) Menurut Provinsi dan Daerah, {year}.csv"
                            elif data_code == "P1":
                                filename = f"Indeks Kedalaman Kemiskinan (P1) Menurut Provinsi dan Daerah, {year}.csv"
                            else:
                                filename = f"Indeks Keparahan Kemiskinan (P2) Menurut Provinsi dan Daerah, {year}.csv"
                        else:  # GK
                            save_dir = "Data_Source/Persentase Penduduk Miskin/Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah/"
                            filename = f"Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah , {year}.csv"
                        
                        # Create directory if not exists
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, filename)
                        
                        # Save file
                        df_upload.to_csv(save_path, index=False)
                        st.success(f"✅ Data berhasil disimpan ke: {save_path}")
                        st.info("💡 Jangan lupa jalankan 'Re-Process Data' untuk memperbarui dataset!")
                else:
                    st.error("❌ Validasi data gagal!")
                    for error in errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    # ========== TAB 2: UPLOAD DATA TIKTOK ==========
    with tab2:
        st.header("📱 Upload Data TikTok")
        st.markdown("Upload file CSV untuk data konten atau komentar TikTok")
        
        tiktok_type = st.radio(
            "Pilih Jenis Data TikTok:",
            ["Konten TikTok (Postingan Utama)", "Komentar TikTok (Balasan)"]
        )
        
        uploaded_tiktok = st.file_uploader(
            "Upload file CSV TikTok", 
            type=['csv'],
            key="tiktok_upload",
            help="File harus berisi kolom: ID Unik, Jenis Konten, Teks Konten, Tahun, Platform"
        )
        
        if uploaded_tiktok is not None:
            try:
                # Determine skiprows based on type
                skiprows = 4 if "Konten" in tiktok_type else 1
                df_tiktok = pd.read_csv(uploaded_tiktok, skiprows=skiprows)
                
                st.success(f"✅ File berhasil dibaca: {uploaded_tiktok.name}")
                
                # Validate
                if "Konten" in tiktok_type:
                    is_valid, errors = validate_tiktok_content(df_tiktok)
                else:
                    is_valid, errors = validate_tiktok_comment(df_tiktok)
                
                if is_valid:
                    st.success("✅ Validasi data berhasil!")
                    
                    # Preview
                    with st.expander("👁️ Preview Data", expanded=True):
                        st.dataframe(df_tiktok.head(10), use_container_width=True)
                        summary = get_data_summary(df_tiktok)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Baris", summary['total_rows'])
                        with col2:
                            if 'year_range' in summary:
                                st.metric("Rentang Tahun", summary['year_range'])
                        with col3:
                            if 'years' in summary:
                                st.write("**Tahun:**", ", ".join(map(str, summary['years'])))
                    
                    # Save button
                    if st.button("💾 Simpan Data TikTok", type="primary"):
                        save_dir = "Data_Source/sosialresponse/"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        if "Konten" in tiktok_type:
                            # Read existing file to preserve header
                            save_path = os.path.join(save_dir, "kontentiktok.csv")
                            # Append to existing file
                            if os.path.exists(save_path):
                                existing_df = pd.read_csv(save_path, skiprows=4)
                                combined_df = pd.concat([existing_df, df_tiktok], ignore_index=True)
                                # Remove duplicates based on ID Unik
                                combined_df = combined_df.drop_duplicates(subset=['ID Unik'], keep='last')
                                # Save with header
                                with open(save_path, 'w') as f:
                                    f.write("DATA EKSTERNAL (DATA MEDIA SOSIAL) TIKTOK,,,,,,,,,,,,,\n")
                                    f.write("DATA KONTEN UTAMA DAN BALASAN KOMENTAR,,,,,,,,,,,,,\n")
                                    f.write(",,,,,,,,,,,,,\n")
                                    f.write("Data Postingan Utama (Content Nodes),,,,,,,,,,,,,\n")
                                combined_df.to_csv(save_path, mode='a', index=False)
                            else:
                                with open(save_path, 'w') as f:
                                    f.write("DATA EKSTERNAL (DATA MEDIA SOSIAL) TIKTOK,,,,,,,,,,,,,\n")
                                    f.write("DATA KONTEN UTAMA DAN BALASAN KOMENTAR,,,,,,,,,,,,,\n")
                                    f.write(",,,,,,,,,,,,,\n")
                                    f.write("Data Postingan Utama (Content Nodes),,,,,,,,,,,,,\n")
                                df_tiktok.to_csv(save_path, mode='a', index=False)
                        else:
                            save_path = os.path.join(save_dir, "komentiktok.csv")
                            if os.path.exists(save_path):
                                existing_df = pd.read_csv(save_path, skiprows=1)
                                combined_df = pd.concat([existing_df, df_tiktok], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=['ID Unik'], keep='last')
                                with open(save_path, 'w') as f:
                                    f.write("Data Komentar Balasan (Edge/Reply Nodes),,,,,,,,,,,,,\n")
                                combined_df.to_csv(save_path, mode='a', index=False)
                            else:
                                with open(save_path, 'w') as f:
                                    f.write("Data Komentar Balasan (Edge/Reply Nodes),,,,,,,,,,,,,\n")
                                df_tiktok.to_csv(save_path, mode='a', index=False)
                        
                        st.success(f"✅ Data TikTok berhasil disimpan!")
                        st.info("💡 Jangan lupa jalankan 'Re-Process Data' untuk memperbarui sentimen!")
                else:
                    st.error("❌ Validasi data gagal!")
                    for error in errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    # ========== TAB 3: RE-PROCESS DATA ==========
    with tab3:
        st.header("🔄 Re-Process Data")
        st.markdown("Jalankan ulang proses data setelah upload data baru")
        
        processor = DataProcessor()
        
        # Layout tombol yang lebih rapi dan simetris (3 kolom x 2 baris)
        st.subheader("🔧 Individual Scripts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("1️⃣ Data Ingestion & Cleaning", use_container_width=True, key="btn_ingest"):
                with st.spinner("Memproses..."):
                    success, output = processor.run_data_ingestion()
                    if success:
                        st.success("✅ Data Ingestion selesai!")
                    else:
                        st.error("❌ Gagal!")
                    with st.expander("📋 Log Output"):
                        st.code(processor.get_logs())
        
        with col2:
            if st.button("2️⃣ Sentiment Processing", use_container_width=True, key="btn_sentiment"):
                with st.spinner("Memproses..."):
                    success, output = processor.run_sentiment_processor()
                    if success:
                        st.success("✅ Sentiment Processing selesai!")
                    else:
                        st.error("❌ Gagal!")
                    with st.expander("📋 Log Output"):
                        st.code(processor.get_logs())
        
        with col3:
            if st.button("3️⃣ Final Integration", use_container_width=True, key="btn_integration"):
                with st.spinner("Memproses..."):
                    success, output = processor.run_final_integration()
                    if success:
                        st.success("✅ Final Integration selesai!")
                    else:
                        st.error("❌ Gagal!")
                    with st.expander("📋 Log Output"):
                        st.code(processor.get_logs())
        
        # Baris kedua
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("4️⃣ ML Model Training", use_container_width=True, key="btn_ml"):
                with st.spinner("Memproses..."):
                    success, output = processor.run_ml_training()
                    if success:
                        st.success("✅ ML Training selesai!")
                    else:
                        st.error("❌ Gagal!")
                    with st.expander("📋 Log Output"):
                        st.code(processor.get_logs())
        
        with col5:
            if st.button("5️⃣ Forecasting 5 Tahun Kedepan", use_container_width=True, key="btn_forecast"):
                with st.spinner("Memproses..."):
                    success, output = processor.run_forecasting()
                    if success:
                        st.success("✅ Forecasting selesai!")
                    else:
                        st.error("❌ Gagal!")
                    with st.expander("📋 Log Output"):
                        st.code(processor.get_logs())
        
        with col6:
            # Kolom kosong untuk simetri, atau bisa ditambahkan tombol lain di masa depan
            st.write("")
        
        st.markdown("---")
        st.subheader("🚀 Full Pipeline")
        st.markdown("Jalankan semua proses secara berurutan (Data Ingestion → Sentiment → Integration → ML → Forecasting)")
        
        if st.button("▶️ Jalankan Full Pipeline", type="primary", use_container_width=True, key="btn_full"):
            with st.spinner("Menjalankan full pipeline... Ini mungkin memakan waktu beberapa menit."):
                success, output = processor.run_full_pipeline(include_sentiment=True)
                if success:
                    st.success("🎉 Full Pipeline selesai!")
                    st.balloons()
                else:
                    st.error("❌ Pipeline gagal!")
                
                with st.expander("📋 Full Log Output", expanded=True):
                    st.code(processor.get_logs())
    
    # ========== TAB 4: DATA STATUS ==========
    with tab4:
        st.header("📊 Status Data Files")
        
        data_status = check_data_files()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Data BPS")
            for name in ['TPT', 'P0', 'P1', 'P2', 'GK']:
                if name in data_status:
                    if data_status[name]:
                        st.success(f"✅ {name}: Tersedia")
                    else:
                        st.error(f"❌ {name}: Tidak tersedia")
        
        with col2:
            st.subheader("📱 Data TikTok")
            for name in ['TikTok Konten', 'TikTok Komentar']:
                if name in data_status:
                    if data_status[name]:
                        st.success(f"✅ {name}: Tersedia")
                    else:
                        st.error(f"❌ {name}: Tidak tersedia")
        
        st.markdown("---")
        st.subheader("📂 Processed Data Files")
        
        processed_files = {
            'Data Master ML': 'cleaned_data/data_master_ml.csv',
            'Dataset Final ML': 'cleaned_data/dataset_final_untuk_ml.csv',
            'Model PKL': 'cleaned_data/model_kemiskinan_final.pkl',
            'Forecast Results': 'cleaned_data/forecast_results.csv'
        }
        
        for name, path in processed_files.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                st.success(f"✅ {name}: {file_size:.1f} KB")
            else:
                st.warning(f"⚠️ {name}: Belum tersedia")