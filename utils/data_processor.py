"""
Data Processor Module
Fungsi untuk menjalankan script processing data
"""

import subprocess
import os
import time
from datetime import datetime

class DataProcessor:
    def __init__(self, project_dir='/home/ardwind/Project/BigDataProject'):
        self.project_dir = project_dir
        self.venv_python = os.path.join(project_dir, 'venv/bin/python3')
        self.logs = []
    
    def _run_script(self, script_name, description):
        """Menjalankan script Python dan menangkap output"""
        script_path = os.path.join(self.project_dir, script_name)
        
        if not os.path.exists(script_path):
            return False, f"Script {script_name} tidak ditemukan"
        
        try:
            start_time = time.time()
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Memulai {description}...")
            
            # Run script
            result = subprocess.run(
                [self.venv_python, script_path],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {description} selesai ({elapsed_time:.1f}s)")
                return True, result.stdout
            else:
                self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {description} gagal")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ {description} timeout (>5 menit)")
            return False, "Proses timeout setelah 5 menit"
        except Exception as e:
            self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
            return False, str(e)
    
    def run_data_ingestion(self):
        """Jalankan 01_data_ingestion_cleaning.py"""
        return self._run_script(
            '01_data_ingestion_cleaning.py',
            'Data Ingestion & Cleaning'
        )
    
    def run_sentiment_processor(self):
        """Jalankan 03_sentiment_processor.py"""
        return self._run_script(
            '03_sentiment_processor.py',
            'Sentiment Processing'
        )
    
    def run_final_integration(self):
        """Jalankan 04_final_integration.py"""
        return self._run_script(
            '04_final_integration.py',
            'Final Integration'
        )
    
    def run_ml_training(self):
        """Jalankan 05_machine_learning_model.py"""
        return self._run_script(
            '05_machine_learning_model.py',
            'ML Model Training'
        )
    
    def run_forecasting(self):
        """Jalankan 07_forecasting.py"""
        return self._run_script(
            '07_forecasting.py',
            'Forecasting 2026-2027'
        )
    
    def run_full_pipeline(self, include_sentiment=True):
        """Jalankan seluruh pipeline processing"""
        self.logs = []
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Memulai Full Pipeline Processing...")
        
        # Step 1: Data Ingestion
        success, output = self.run_data_ingestion()
        if not success:
            return False, "Gagal di Data Ingestion: " + output
        
        # Step 2: Sentiment Processing (optional)
        if include_sentiment:
            success, output = self.run_sentiment_processor()
            if not success:
                self.logs.append(f"⚠️ Sentiment Processing gagal, melanjutkan tanpa data sentimen...")
        
        # Step 3: Final Integration
        success, output = self.run_final_integration()
        if not success:
            return False, "Gagal di Final Integration: " + output
        
        # Step 4: ML Training
        success, output = self.run_ml_training()
        if not success:
            return False, "Gagal di ML Training: " + output
        
        # Step 5: Forecasting
        success, output = self.run_forecasting()
        if not success:
            return False, "Gagal di Forecasting: " + output
        
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 Full Pipeline selesai!")
        return True, "\n".join(self.logs)
    
    def get_logs(self):
        """Mendapatkan semua logs"""
        return "\n".join(self.logs)
    
    def clear_logs(self):
        """Menghapus logs"""
        self.logs = []

def check_data_files():
    """Cek keberadaan file data yang diperlukan"""
    project_dir = '/home/ardwind/Project/BigDataProject'
    required_files = {
        'TPT': 'Data_Source/Tingkat Pengangguran Terbuka/',
        'P0': 'Data_Source/Persentase Penduduk Miskin/(P0) Menurut Provinsi/',
        'P1': 'Data_Source/Persentase Penduduk Miskin/(P1) Menurut Provinsi/',
        'P2': 'Data_Source/Persentase Penduduk Miskin/(P2) Menurut Provinsi/',
        'GK': 'Data_Source/Persentase Penduduk Miskin/Garis Kemiskinan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah/',
        'TikTok Konten': 'Data_Source/sosialresponse/kontentiktok.csv',
        'TikTok Komentar': 'Data_Source/sosialresponse/komentiktok.csv'
    }
    
    status = {}
    for name, path in required_files.items():
        full_path = os.path.join(project_dir, path)
        if path.endswith('.csv'):
            status[name] = os.path.exists(full_path)
        else:
            status[name] = os.path.exists(full_path) and len(os.listdir(full_path)) > 0
    
    return status
