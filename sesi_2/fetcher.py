"""
fetcher.py
==========
Modul untuk mengambil data lingkungan/cuaca dari API publik.

Mendukung dua mode:
- Mode LIVE: HTTP Request ke OpenWeatherMap API (memerlukan API key)
- Mode DUMMY: Menghasilkan data cuaca simulasi tanpa koneksi internet

Error handling diimplementasikan untuk kegagalan koneksi jaringan.
"""

import requests
import json
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# KONFIGURASI API
# ==============================================================================

# Ganti dengan API key nyata dari https://openweathermap.org/ (gratis)
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Daftar kota yang akan diambil datanya
TARGET_CITIES = [
    {"name": "Jakarta", "country": "ID", "lat": -6.2088, "lon": 106.8456},
    {"name": "Bandung", "country": "ID", "lat": -6.9175, "lon": 107.6191},
    {"name": "Surabaya", "country": "ID", "lat": -7.2575, "lon": 112.7521},
    {"name": "Yogyakarta", "country": "ID", "lat": -7.7956, "lon": 110.3695},
    {"name": "Medan", "country": "ID", "lat": 3.5952, "lon": 98.6722},
]

# Timeout untuk HTTP request (detik)
REQUEST_TIMEOUT = 10
MAX_RETRY = 3


# ==============================================================================
# FUNGSI FETCHER UTAMA
# ==============================================================================

def fetch_weather_live(city: dict, api_key: str) -> Optional[dict]:
    """
    Mengambil data cuaca LIVE dari OpenWeatherMap API.
    Dilengkapi dengan mekanisme retry dan error handling.
    
    Parameters:
        city (dict): Dictionary berisi nama kota, lat, lon.
        api_key (str): API key OpenWeatherMap.
    
    Returns:
        dict | None: Data JSON mentah dari API, atau None jika gagal.
    """
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {
        "lat": city["lat"],
        "lon": city["lon"],
        "appid": api_key,
        "units": "metric",  # Celsius
        "lang": "id",
    }

    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info(f"Fetching live data untuk '{city['name']}' (percobaan {attempt}/{MAX_RETRY})...")
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise error untuk status 4xx/5xx
            data = response.json()
            logger.info(f"  ✔ Berhasil mendapatkan data untuk '{city['name']}'")
            return data

        except requests.exceptions.ConnectionError:
            logger.error(f"  ✘ Koneksi gagal untuk '{city['name']}'. Cek koneksi internet.")
        except requests.exceptions.Timeout:
            logger.warning(f"  ✘ Request timeout untuk '{city['name']}' (attempt {attempt}).")
            time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.HTTPError as e:
            logger.error(f"  ✘ HTTP Error {e.response.status_code} untuk '{city['name']}': {e}")
            break  # Tidak perlu retry untuk error 4xx
        except requests.exceptions.RequestException as e:
            logger.error(f"  ✘ Request error tidak terduga: {e}")
            break

    logger.warning(f"  ↩ Menggunakan data dummy sebagai fallback untuk '{city['name']}'.")
    return None


def generate_dummy_weather(city: dict) -> dict:
    """
    Menghasilkan data cuaca dummy yang mensimulasikan struktur respons OpenWeatherMap API.
    
    Parameters:
        city (dict): Informasi kota (nama, lat, lon).
    
    Returns:
        dict: Struktur JSON yang mirip dengan respons OpenWeatherMap.
    """
    conditions = ["Clear", "Clouds", "Rain", "Drizzle", "Thunderstorm", "Mist"]
    condition = random.choice(conditions)

    # Simulasi suhu berdasarkan kondisi cuaca
    temp_base = 26.0 if "Tropical" else 25.0
    temp = round(temp_base + random.uniform(-5, 8), 1)
    feels_like = round(temp + random.uniform(-2, 3), 1)
    humidity = random.randint(55, 98)
    wind_speed = round(random.uniform(0.5, 8.5), 1)
    visibility = random.randint(2000, 10000)

    return {
        "coord": {"lat": city["lat"], "lon": city["lon"]},
        "weather": [{"main": condition, "description": condition.lower()}],
        "main": {
            "temp": temp,
            "feels_like": feels_like,
            "temp_min": round(temp - 2, 1),
            "temp_max": round(temp + 2, 1),
            "pressure": random.randint(1008, 1020),
            "humidity": humidity,
        },
        "visibility": visibility,
        "wind": {"speed": wind_speed, "deg": random.randint(0, 360)},
        "clouds": {"all": random.randint(0, 100)},
        "dt": int(datetime.now().timestamp()),
        "name": city["name"],
        "sys": {"country": city["country"]},
        "_source": "DUMMY",  # Penanda data bukan dari API nyata
    }


def fetch_all_cities(use_live_api: bool = False) -> list:
    """
    Mengambil data cuaca untuk semua kota target.
    
    Parameters:
        use_live_api (bool): Jika True, gunakan API nyata; jika False, gunakan dummy.
    
    Returns:
        list: List data cuaca mentah (JSON) untuk semua kota.
    """
    logger.info(f"Mode fetching: {'LIVE API' if use_live_api else 'DUMMY DATA'}")
    raw_data_list = []

    for city in TARGET_CITIES:
        if use_live_api and OPENWEATHER_API_KEY != "YOUR_API_KEY_HERE":
            raw = fetch_weather_live(city, OPENWEATHER_API_KEY)
            if raw is None:
                raw = generate_dummy_weather(city)
        else:
            raw = generate_dummy_weather(city)
            logger.info(f"  ✔ [DUMMY] Data cuaca '{city['name']}' dibuat.")

        raw_data_list.append(raw)
        time.sleep(0.1)  # Jeda kecil antar request

    logger.info(f"Total data terkumpul: {len(raw_data_list)} kota.")
    return raw_data_list


# ==============================================================================
# ENTRY POINT (untuk pengujian mandiri)
# ==============================================================================

if __name__ == "__main__":
    data = fetch_all_cities(use_live_api=False)
    print("\n[PREVIEW] Contoh data mentah kota pertama:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
