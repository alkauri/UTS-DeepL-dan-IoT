"""
processor.py
============
Modul untuk membersihkan dan mentransformasikan data cuaca JSON mentah
menjadi format tabel terstruktur menggunakan Pandas.

Fungsi utama:
- Ekstraksi field relevan dari struktur JSON bersarang
- Pembersihan data (handling missing field, tipe data)
- Transformasi ke DataFrame yang siap disimpan
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Union

# Konfigurasi logging
logger = logging.getLogger(__name__)


# ==============================================================================
# FUNGSI EKSTRAKSI FIELD
# ==============================================================================

def safe_get(data: dict, *keys, default=None):
    """
    Helper untuk mengakses kunci bersarang secara aman (tidak raise KeyError).
    
    Contoh:
        safe_get(data, "main", "temp", default=0.0)
        → data["main"]["temp"] atau 0.0 jika kunci tidak ada
    
    Parameters:
        data (dict): Dictionary sumber.
        *keys: Urutan kunci yang akan diakses secara berurutan.
        default: Nilai default jika kunci tidak ditemukan.
    
    Returns:
        Nilai yang ditemukan, atau default.
    """
    try:
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError, IndexError):
        return default


def extract_weather_fields(raw_item: dict) -> dict:
    """
    Mengekstrak dan meratakan field penting dari satu item JSON cuaca.
    
    Parameters:
        raw_item (dict): Satu record data cuaca mentah dari OpenWeatherMap.
    
    Returns:
        dict: Baris data yang sudah diratakan dan bersih.
    """
    # Konversi timestamp Unix ke datetime yang mudah dibaca
    unix_ts = safe_get(raw_item, "dt", default=None)
    timestamp_readable = (
        datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S")
        if unix_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    return {
        # Identifikasi lokasi
        "city_name":        safe_get(raw_item, "name", default="Unknown"),
        "country_code":     safe_get(raw_item, "sys", "country", default="??"),
        "latitude":         safe_get(raw_item, "coord", "lat", default=None),
        "longitude":        safe_get(raw_item, "coord", "lon", default=None),

        # Waktu pengambilan data
        "timestamp":        timestamp_readable,

        # Kondisi cuaca
        "weather_main":     safe_get(raw_item, "weather", 0, "main", default="Unknown"),
        "weather_desc":     safe_get(raw_item, "weather", 0, "description", default=""),

        # Suhu (°C)
        "temp_celsius":     safe_get(raw_item, "main", "temp", default=None),
        "feels_like":       safe_get(raw_item, "main", "feels_like", default=None),
        "temp_min":         safe_get(raw_item, "main", "temp_min", default=None),
        "temp_max":         safe_get(raw_item, "main", "temp_max", default=None),

        # Kelembapan & tekanan
        "humidity_pct":     safe_get(raw_item, "main", "humidity", default=None),
        "pressure_hPa":     safe_get(raw_item, "main", "pressure", default=None),

        # Angin
        "wind_speed_ms":    safe_get(raw_item, "wind", "speed", default=None),
        "wind_direction":   safe_get(raw_item, "wind", "deg", default=None),

        # Visibilitas & tutupan awan
        "visibility_m":     safe_get(raw_item, "visibility", default=None),
        "cloud_coverage":   safe_get(raw_item, "clouds", "all", default=None),

        # Penanda sumber data
        "data_source":      safe_get(raw_item, "_source", default="API"),
    }


# ==============================================================================
# FUNGSI TRANSFORMASI BATCH
# ==============================================================================

def transform_to_dataframe(raw_data_list: list) -> pd.DataFrame:
    """
    Mengubah list data JSON mentah menjadi DataFrame Pandas yang terstruktur.
    
    Parameters:
        raw_data_list (list): List dictionary data cuaca mentah.
    
    Returns:
        pd.DataFrame: DataFrame siap pakai dengan tipe data yang tepat.
    """
    if not raw_data_list:
        logger.warning("Data kosong! Mengembalikan DataFrame kosong.")
        return pd.DataFrame()

    # Ekstraksi semua record
    records = []
    for i, item in enumerate(raw_data_list):
        try:
            record = extract_weather_fields(item)
            records.append(record)
            logger.debug(f"  ✔ Record {i+1} '{record.get('city_name')}' diekstrak.")
        except Exception as e:
            logger.error(f"  ✘ Gagal mengekstrak record {i+1}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Berhasil membuat DataFrame: {df.shape[0]} baris, {df.shape[1]} kolom.")

    # Pastikan tipe data yang benar
    float_cols = ["latitude", "longitude", "temp_celsius", "feels_like",
                  "temp_min", "temp_max", "wind_speed_ms"]
    int_cols   = ["humidity_pct", "pressure_hPa", "wind_direction",
                  "visibility_m", "cloud_coverage"]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


# ==============================================================================
# FUNGSI RINGKASAN STATISTIK
# ==============================================================================

def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan statistik ringkasan untuk insight cepat data cuaca.
    
    Parameters:
        df (pd.DataFrame): DataFrame data cuaca yang sudah ditransformasikan.
    
    Returns:
        pd.DataFrame: Ringkasan statistik per kota.
    """
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("city_name").agg(
        avg_temp=("temp_celsius", "mean"),
        max_temp=("temp_celsius", "max"),
        min_temp=("temp_celsius", "min"),
        avg_humidity=("humidity_pct", "mean"),
        avg_wind=("wind_speed_ms", "mean"),
        weather_condition=("weather_main", "first"),
    ).round(2).reset_index()

    return summary


# ==============================================================================
# ENTRY POINT (untuk pengujian mandiri)
# ==============================================================================

if __name__ == "__main__":
    # Test dengan data dummy dari fetcher
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from fetcher import fetch_all_cities

    raw_data = fetch_all_cities(use_live_api=False)
    df = transform_to_dataframe(raw_data)

    print("\n[PREVIEW] DataFrame hasil transformasi:")
    print(df.to_string(index=False))

    print("\n[RINGKASAN] Statistik per kota:")
    summary = generate_summary_stats(df)
    print(summary.to_string(index=False))
