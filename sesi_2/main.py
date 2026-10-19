"""
main.py
=======
Skrip utama pipeline data lingkungan/cuaca yang mengorkestrasi:
1. Fetching data dari API / data dummy
2. Transformasi & pembersihan data
3. Penyimpanan hasil ke CSV dan SQLite

Jalankan: python main.py
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
from datetime import datetime

# Tambahkan direktori sesi_2 ke path
sys.path.insert(0, os.path.dirname(__file__))

from fetcher import fetch_all_cities
from processor import transform_to_dataframe, generate_summary_stats

# ==============================================================================
# KONFIGURASI LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# KONFIGURASI OUTPUT
# ==============================================================================

OUTPUT_DIR = "sesi_2/output"
CSV_OUTPUT_PATH    = os.path.join(OUTPUT_DIR, "weather_data.csv")
SQLITE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "weather_data.db")
SQLITE_TABLE_NAME  = "weather_readings"


# ==============================================================================
# FUNGSI PENYIMPANAN
# ==============================================================================

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Menyimpan DataFrame ke file CSV.
    Jika file sudah ada, data baru akan diappend (tanpa header duplikat).
    
    Parameters:
        df (pd.DataFrame): Data yang akan disimpan.
        path (str): Path file CSV tujuan.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    df.to_csv(path, mode="a", header=not file_exists, index=False)
    action = "ditambahkan ke" if file_exists else "dibuat di"
    logger.info(f"✔ Data CSV berhasil {action}: '{path}'")


def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Menyimpan DataFrame ke database SQLite.
    Tabel akan dibuat otomatis jika belum ada.
    
    Parameters:
        df (pd.DataFrame): Data yang akan disimpan.
        db_path (str): Path file database SQLite.
        table_name (str): Nama tabel dalam database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()
        logger.info(f"✔ Data SQLite berhasil disimpan: '{db_path}' → tabel '{table_name}'")

        # Verifikasi: hitung total record dalam tabel
        total = pd.read_sql(f"SELECT COUNT(*) as total FROM {table_name}", conn).iloc[0, 0]
        logger.info(f"  Total record dalam tabel '{table_name}': {total}")
        conn.close()

    except sqlite3.Error as e:
        logger.error(f"✘ Gagal menyimpan ke SQLite: {e}")
    finally:
        if conn:
            conn.close()


# ==============================================================================
# PIPELINE UTAMA
# ==============================================================================

def run_pipeline(use_live_api: bool = False) -> None:
    """
    Menjalankan seluruh pipeline secara berurutan:
    Fetch → Transform → Save (CSV + SQLite)
    
    Parameters:
        use_live_api (bool): True untuk menggunakan API nyata, False untuk dummy.
    """
    start_time = datetime.now()

    print("\n" + "=" * 65)
    print("  PIPELINE ANALISIS DATA LINGKUNGAN / CUACA")
    print(f"  Dijalankan pada: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # LANGKAH 1: Fetch data
    # ------------------------------------------------------------------
    print("\n[LANGKAH 1] Mengambil data cuaca dari sumber...")
    try:
        raw_data = fetch_all_cities(use_live_api=use_live_api)
        logger.info(f"  ✔ {len(raw_data)} record mentah berhasil diambil.")
    except Exception as e:
        logger.error(f"  ✘ Pipeline GAGAL pada tahap fetching: {e}")
        return

    # ------------------------------------------------------------------
    # LANGKAH 2: Transform data
    # ------------------------------------------------------------------
    print("\n[LANGKAH 2] Mentransformasikan data JSON ke DataFrame...")
    try:
        df = transform_to_dataframe(raw_data)
        if df.empty:
            logger.error("  ✘ DataFrame kosong setelah transformasi. Pipeline dihentikan.")
            return
        logger.info(f"  ✔ Transformasi selesai: {df.shape[0]} baris, {df.shape[1]} kolom.")
    except Exception as e:
        logger.error(f"  ✘ Pipeline GAGAL pada tahap transformasi: {e}")
        return

    # ------------------------------------------------------------------
    # LANGKAH 3: Tampilkan ringkasan
    # ------------------------------------------------------------------
    print("\n[LANGKAH 3] Ringkasan statistik data:")
    summary = generate_summary_stats(df)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # LANGKAH 4: Simpan ke CSV
    # ------------------------------------------------------------------
    print("\n[LANGKAH 4] Menyimpan data ke CSV...")
    try:
        save_to_csv(df, CSV_OUTPUT_PATH)
    except Exception as e:
        logger.error(f"  ✘ Gagal menyimpan CSV: {e}")

    # ------------------------------------------------------------------
    # LANGKAH 5: Simpan ke SQLite
    # ------------------------------------------------------------------
    print("\n[LANGKAH 5] Menyimpan data ke SQLite...")
    try:
        save_to_sqlite(df, SQLITE_OUTPUT_PATH, SQLITE_TABLE_NAME)
    except Exception as e:
        logger.error(f"  ✘ Gagal menyimpan SQLite: {e}")

    # ------------------------------------------------------------------
    # SELESAI
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*65}")
    print(f"  [SELESAI] Pipeline berhasil dalam {elapsed:.2f} detik.")
    print(f"  Output CSV   : {os.path.abspath(CSV_OUTPUT_PATH)}")
    print(f"  Output SQLite: {os.path.abspath(SQLITE_OUTPUT_PATH)}")
    print(f"{'='*65}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Set use_live_api=True jika Anda memiliki API key di fetcher.py
    run_pipeline(use_live_api=False)
