# 02_sentiment_ingestion.py

import pandas as pd
import datetime
import os
import snscrape.modules.twitter as sntwitter
import itertools

# --- KONFIGURASI PROYEK ---
CLEANED_DIR = 'cleaned_data/'
KEYWORD_LIST = [
    "kemiskinan OR miskin OR harga sembako OR PHK",
    "kesejahteraan rakyat OR subsidi",
    "bantuan sosial OR bansos OR lapangan kerja"
]
# Daftar provinsi harus sesuai dengan standardisasi di 01_data_ingestion_cleaning.py
PROVINCE_LIST = [
    "ACEH", "SUMATRA UTARA", "SUMATRA BARAT", "RIAU", "JAMBI", 
    "SUMATRA SELATAN", "BENGKULU", "LAMPUNG", "KEP. BANGKA BELITUNG", 
    "KEPULAUAN RIAU", "JAKARTA", "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA", 
    "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR", 
    "KALIMATAN BARAT", "KALIMATAN SELATAN", "KALIMATAN TENGAH", "KALIMATAN TIMUR", 
    "KALIMATAN UTARA", "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN", 
    "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", 
    "PAPUA", "PAPUA BARAT", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN",
    "PAPUA BARAT DAYA"
]

# --- RENTANG WAKTU ---
START_YEAR = 2013 
END_YEAR = 2024 

def scrape_tweets(max_tweets_per_year_province=500):
    """Melakukan scraping tweet menggunakan snscrape."""
    
    all_tweet_data = []
    total_scraped = 0
    
    print(f"Memulai scraping data sentimen dari {START_YEAR} hingga {END_YEAR}...")
    print(f"Target max per kombinasi (Tahun/Provinsi/Keyword): {max_tweets_per_year_province}")
    
    for year in range(START_YEAR, END_YEAR + 1):
        # snscrape menggunakan format YYYY-MM-DD
        start_date = f"{year}-01-01"
        # Ambil sampai akhir tahun (31 Des)
        end_date = f"{year}-12-31" 
        
        for province in PROVINCE_LIST:
            for keyword in KEYWORD_LIST:
                
                # Query Pencarian: (Keywords) PROVINSI sejak TglX sampai TglY, bukan Retweet, bahasa Indonesia
                query = f'({keyword}) "{province}" since:{start_date} until:{end_date} -filter:retweets lang:id'
                
                try:
                    # Menggunakan sntwitter.TwitterSearchScraper
                    scraper = sntwitter.TwitterSearchScraper(query)
                    
                    # Iterasi dan ambil tweets hingga batas max_tweets_per_year_province
                    for i, tweet in enumerate(scraper.get_items()):
                        if i >= max_tweets_per_year_province:
                            break
                        
                        all_tweet_data.append({
                            'Provinsi_Scrape': province,
                            'Tahun_Scrape': year,
                            'Keyword': keyword,
                            'text': tweet.rawContent,
                            'created_at': tweet.date,
                            'retweet_count': tweet.retweetCount,
                            'like_count': tweet.likeCount,
                        })
                        total_scraped += 1
                        
                    print(f"-> {province}, {year}, Keyword '{keyword[:15]}...': {i + 1} tweets scraped. Total: {total_scraped}")
                
                except Exception as e:
                    print(f"Gagal scraping {province}, {year} (Keyword: {keyword[:15]}...). Error: {e}")
                    
    return pd.DataFrame(all_tweet_data)

if __name__ == '__main__':
    # Catatan: Proses ini BISA memakan waktu lama (jam) karena scraping volume besar
    df_sentiment_raw = scrape_tweets(max_tweets_per_year_province=500)
    
    if not df_sentiment_raw.empty:
        df_sentiment_raw['Tahun'] = df_sentiment_raw['Tahun_Scrape']
        df_sentiment_raw.to_csv(os.path.join(CLEANED_DIR, 'sentiment_raw.csv'), index=False)
        print(f"\n[DONE] Data Mentah Sentimen (Rows: {len(df_sentiment_raw)}) disimpan ke '{os.path.join(CLEANED_DIR, 'sentiment_raw.csv')}'")
    else:
        print("\n[GAGAL] Tidak ada data sentimen yang berhasil diambil. Cek instalasi snscrape Anda.")