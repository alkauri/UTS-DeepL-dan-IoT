"""
spatial_analysis.py
===================
Script untuk pemrosesan data spasial, spatial filtering, dan visualisasi peta
wilayah administratif Indonesia menggunakan GeoPandas dan Folium/Matplotlib.

Alur kerja:
1. Generate GeoJSON dummy wilayah & titik sebaran fasilitas
2. Baca file data spasial
3. Lakukan spatial filtering berdasarkan kriteria atribut dan koordinat
4. Hasilkan visualisasi peta (statis PNG + interaktif HTML)

Catatan: Jika Anda memiliki Shapefile/GeoJSON nyata, ganti path di LOAD_PATH.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import json
import os
from shapely.geometry import Point, Polygon, MultiPolygon

# ==============================================================================
# KONFIGURASI
# ==============================================================================

OUTPUT_DIR = "sesi_3/output"
GEOJSON_PATH = os.path.join("sesi_3", "data", "wilayah_dummy.geojson")
FACILITIES_PATH = os.path.join("sesi_3", "data", "fasilitas_dummy.geojson")

# Kriteria filter
FILTER_CATEGORY = "Perkotaan"          # Filter atribut kategori wilayah
BOUNDING_BOX = {                        # Bounding box Jawa (lon_min, lat_min, lon_max, lat_max)
    "lon_min": 105.0,
    "lat_min": -9.0,
    "lon_max": 115.5,
    "lat_max": -5.5,
}


# ==============================================================================
# 1. GENERATE DATA SPASIAL DUMMY
# ==============================================================================

def generate_dummy_wilayah_geojson(output_path: str) -> None:
    """
    Membuat GeoJSON dummy yang merepresentasikan wilayah administratif
    beberapa provinsi di Pulau Jawa dan sekitarnya.
    
    Parameters:
        output_path (str): Path tujuan file GeoJSON.
    """
    provinces = [
        {
            "nama": "DKI Jakarta",
            "kategori": "Perkotaan",
            "populasi": 10_770_000,
            "luas_km2": 662,
            "geometry": Polygon([
                (106.68, -6.37), (106.97, -6.37),
                (106.97, -6.10), (106.68, -6.10), (106.68, -6.37)
            ])
        },
        {
            "nama": "Jawa Barat",
            "kategori": "Campuran",
            "populasi": 49_935_858,
            "luas_km2": 35_377,
            "geometry": Polygon([
                (106.5, -7.9), (108.8, -7.9),
                (108.8, -5.9), (106.5, -5.9), (106.5, -7.9)
            ])
        },
        {
            "nama": "Jawa Tengah",
            "kategori": "Pertanian",
            "populasi": 37_274_572,
            "luas_km2": 32_800,
            "geometry": Polygon([
                (108.5, -8.2), (111.2, -8.2),
                (111.2, -6.5), (108.5, -6.5), (108.5, -8.2)
            ])
        },
        {
            "nama": "DI Yogyakarta",
            "kategori": "Perkotaan",
            "populasi": 3_800_000,
            "luas_km2": 3_186,
            "geometry": Polygon([
                (110.0, -8.1), (110.6, -8.1),
                (110.6, -7.7), (110.0, -7.7), (110.0, -8.1)
            ])
        },
        {
            "nama": "Jawa Timur",
            "kategori": "Industri",
            "populasi": 41_149_974,
            "luas_km2": 47_963,
            "geometry": Polygon([
                (111.0, -8.8), (114.6, -8.8),
                (114.6, -6.8), (111.0, -6.8), (111.0, -8.8)
            ])
        },
        {
            "nama": "Banten",
            "kategori": "Industri",
            "populasi": 12_448_595,
            "luas_km2": 9_663,
            "geometry": Polygon([
                (105.2, -7.1), (106.7, -7.1),
                (106.7, -5.8), (105.2, -5.8), (105.2, -7.1)
            ])
        },
    ]

    gdf = gpd.GeoDataFrame(provinces, crs="EPSG:4326")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"  ✔ GeoJSON wilayah dibuat: '{output_path}'")


def generate_dummy_facilities_geojson(output_path: str, n_points: int = 80) -> None:
    """
    Membuat GeoJSON dummy titik sebaran fasilitas publik (RS, sekolah, dll.)
    tersebar di Pulau Jawa.
    
    Parameters:
        output_path (str): Path tujuan file GeoJSON.
        n_points (int): Jumlah titik fasilitas.
    """
    np.random.seed(99)
    facility_types = ["Rumah Sakit", "Sekolah", "Puskesmas", "Universitas", "Balai Desa"]

    data = {
        "nama": [f"Fasilitas-{i+1:03d}" for i in range(n_points)],
        "jenis": np.random.choice(facility_types, n_points),
        "kapasitas": np.random.randint(50, 5000, n_points),
        "tahun_berdiri": np.random.randint(1970, 2023, n_points),
        "geometry": [
            Point(
                np.random.uniform(105.2, 114.5),   # longitude Jawa
                np.random.uniform(-8.5, -5.9)       # latitude Jawa
            )
            for _ in range(n_points)
        ],
    }

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"  ✔ GeoJSON fasilitas dibuat: '{output_path}' ({n_points} titik)")


# ==============================================================================
# 2. LOAD DATA SPASIAL
# ==============================================================================

def load_spatial_data(wilayah_path: str, facilities_path: str) -> tuple:
    """
    Memuat data GeoJSON wilayah dan titik fasilitas ke GeoDataFrame.
    
    Returns:
        tuple: (gdf_wilayah, gdf_facilities)
    """
    gdf_wilayah = gpd.read_file(wilayah_path)
    gdf_facilities = gpd.read_file(facilities_path)

    print(f"  ✔ Wilayah dimuat: {len(gdf_wilayah)} fitur | CRS: {gdf_wilayah.crs}")
    print(f"  ✔ Fasilitas dimuat: {len(gdf_facilities)} titik | CRS: {gdf_facilities.crs}")

    return gdf_wilayah, gdf_facilities


# ==============================================================================
# 3. SPATIAL FILTERING
# ==============================================================================

def filter_by_attribute(gdf: gpd.GeoDataFrame, column: str, value: str) -> gpd.GeoDataFrame:
    """
    Filter berdasarkan nilai atribut kategori tertentu.
    
    Parameters:
        column (str): Nama kolom yang dijadikan kriteria filter.
        value (str): Nilai yang dicari.
    
    Returns:
        gpd.GeoDataFrame: Subset data yang sesuai kriteria.
    """
    filtered = gdf[gdf[column] == value]
    print(f"  ✔ Filter atribut '{column}={value}': {len(filtered)} dari {len(gdf)} fitur.")
    return filtered


def filter_by_bounding_box(gdf: gpd.GeoDataFrame, bbox: dict) -> gpd.GeoDataFrame:
    """
    Filter berdasarkan bounding box koordinat geografis.
    
    Parameters:
        bbox (dict): Dictionary dengan kunci lon_min, lat_min, lon_max, lat_max.
    
    Returns:
        gpd.GeoDataFrame: Subset data dalam area bounding box.
    """
    bbox_polygon = Polygon([
        (bbox["lon_min"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_max"]),
        (bbox["lon_min"], bbox["lat_max"]),
    ])
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:4326")
    filtered = gpd.clip(gdf, bbox_gdf)
    print(f"  ✔ Filter bounding box: {len(filtered)} dari {len(gdf)} fitur tersisa.")
    return filtered


def spatial_join_facilities(gdf_wilayah: gpd.GeoDataFrame, gdf_facilities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Melakukan spatial join: memetakan setiap fasilitas ke wilayah yang memuatnya.
    
    Returns:
        gpd.GeoDataFrame: DataFrame fasilitas yang dilengkapi dengan info wilayah.
    """
    joined = gpd.sjoin(gdf_facilities, gdf_wilayah[["nama", "kategori", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"nama_right": "provinsi", "kategori": "kategori_wilayah"})
    print(f"  ✔ Spatial join selesai: {len(joined)} fasilitas berhasil dipetakan ke wilayah.")
    return joined


# ==============================================================================
# 4. VISUALISASI PETA STATIS (PNG)
# ==============================================================================

def create_static_map(gdf_wilayah: gpd.GeoDataFrame,
                       gdf_filtered_wilayah: gpd.GeoDataFrame,
                       gdf_facilities: gpd.GeoDataFrame,
                       output_path: str) -> None:
    """
    Membuat peta statis menggunakan Matplotlib dan menyimpan sebagai PNG.
    
    Parameters:
        gdf_wilayah: Semua wilayah (background).
        gdf_filtered_wilayah: Wilayah yang sudah difilter.
        gdf_facilities: Titik fasilitas.
        output_path (str): Path file PNG output.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_facecolor("#E8F4FD")

    # Layer 1: Semua wilayah (abu-abu, background)
    gdf_wilayah.plot(ax=ax, color="#D0D0D0", edgecolor="#999999", linewidth=1, alpha=0.7)

    # Layer 2: Wilayah terfilter (highlight warna)
    if not gdf_filtered_wilayah.empty:
        gdf_filtered_wilayah.plot(ax=ax, color="#FF6B35", edgecolor="#CC4400",
                                   linewidth=1.5, alpha=0.85, label="Wilayah Perkotaan")

    # Layer 3: Titik fasilitas dengan warna berbeda per jenis
    jenis_list = gdf_facilities["jenis"].unique()
    color_map = {
        "Rumah Sakit": "#E74C3C",
        "Sekolah": "#2ECC71",
        "Puskesmas": "#3498DB",
        "Universitas": "#9B59B6",
        "Balai Desa": "#F39C12",
    }
    for jenis in jenis_list:
        subset = gdf_facilities[gdf_facilities["jenis"] == jenis]
        subset.plot(ax=ax, color=color_map.get(jenis, "#333333"),
                    markersize=5, marker="o", label=jenis, alpha=0.8)

    # Label wilayah
    for _, row in gdf_wilayah.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(
            row["nama"], xy=(centroid.x, centroid.y),
            fontsize=8, ha="center", fontweight="bold", color="#333333"
        )

    # Legenda dan format
    handles = [
        mpatches.Patch(color="#D0D0D0", label="Semua Wilayah"),
        mpatches.Patch(color="#FF6B35", label=f"Wilayah: {FILTER_CATEGORY}"),
    ]
    for jenis in jenis_list:
        handles.append(mpatches.Patch(color=color_map.get(jenis, "#333"), label=jenis))

    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Peta Wilayah Administratif & Sebaran Fasilitas Publik\n(Pulau Jawa dan Sekitarnya)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✔ Peta statis disimpan: '{output_path}'")
    plt.show()


# ==============================================================================
# 5. VISUALISASI PETA INTERAKTIF (HTML)
# ==============================================================================

def create_interactive_map(gdf_wilayah: gpd.GeoDataFrame,
                            gdf_facilities: gpd.GeoDataFrame,
                            output_path: str) -> None:
    """
    Membuat peta interaktif menggunakan Folium dan menyimpan sebagai HTML.
    
    Parameters:
        gdf_wilayah: GeoDataFrame wilayah administratif.
        gdf_facilities: GeoDataFrame titik fasilitas.
        output_path (str): Path file HTML output.
    """
    # Pusat peta (sekitar Jawa)
    m = folium.Map(location=[-7.0, 110.0], zoom_start=7, tiles="CartoDB positron")

    # Tambahkan layer choropleth wilayah
    folium.GeoJson(
        gdf_wilayah.__geo_interface__,
        name="Wilayah Administratif",
        style_function=lambda feat: {
            "fillColor": {
                "Perkotaan": "#FF6B35",
                "Campuran": "#4ECDC4",
                "Pertanian": "#95D86E",
                "Industri": "#6C5CE7",
            }.get(feat["properties"].get("kategori", ""), "#CCCCCC"),
            "color": "#555555",
            "weight": 1.5,
            "fillOpacity": 0.6,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=["nama", "kategori", "populasi", "luas_km2"],
            aliases=["Provinsi:", "Kategori:", "Populasi:", "Luas (km²):"],
            localize=True,
        ),
    ).add_to(m)

    # Tambahkan titik fasilitas sebagai marker
    icon_map = {
        "Rumah Sakit": ("red", "plus"),
        "Sekolah": ("green", "book"),
        "Puskesmas": ("blue", "medkit"),
        "Universitas": ("purple", "graduation-cap"),
        "Balai Desa": ("orange", "home"),
    }
    for _, row in gdf_facilities.iterrows():
        jenis = row.get("jenis", "Unknown")
        color, icon = icon_map.get(jenis, ("gray", "info-sign"))
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(
                f"<b>{row['nama']}</b><br>Jenis: {jenis}<br>Kapasitas: {row['kapasitas']}<br>Berdiri: {row['tahun_berdiri']}",
                max_width=250
            ),
            tooltip=f"{row['nama']} ({jenis})",
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"  ✔ Peta interaktif disimpan: '{output_path}'")


# ==============================================================================
# 6. PIPELINE UTAMA
# ==============================================================================

def main():
    print("=" * 65)
    print("  SPATIAL DATA PROCESSING & MAPPING - GEOPANDAS")
    print("=" * 65)

    # Langkah 1: Generate data dummy jika belum ada
    print("\n[LANGKAH 1] Membuat data spasial dummy...")
    if not os.path.exists(GEOJSON_PATH):
        generate_dummy_wilayah_geojson(GEOJSON_PATH)
    else:
        print(f"  ✔ File wilayah sudah ada: '{GEOJSON_PATH}'")
    if not os.path.exists(FACILITIES_PATH):
        generate_dummy_facilities_geojson(FACILITIES_PATH)
    else:
        print(f"  ✔ File fasilitas sudah ada: '{FACILITIES_PATH}'")

    # Langkah 2: Load data spasial
    print("\n[LANGKAH 2] Memuat data spasial...")
    gdf_wilayah, gdf_facilities = load_spatial_data(GEOJSON_PATH, FACILITIES_PATH)

    # Langkah 3: Spatial filtering
    print("\n[LANGKAH 3] Melakukan spatial filtering...")
    gdf_perkotaan = filter_by_attribute(gdf_wilayah, "kategori", FILTER_CATEGORY)
    gdf_facilities_bb = filter_by_bounding_box(gdf_facilities, BOUNDING_BOX)
    gdf_joined = spatial_join_facilities(gdf_wilayah, gdf_facilities_bb)

    # Langkah 4: Tampilkan ringkasan
    print("\n[RINGKASAN] Fasilitas per Provinsi:")
    if "provinsi" in gdf_joined.columns:
        summary = gdf_joined.groupby(["provinsi", "jenis"]).size().unstack(fill_value=0)
        print(summary.to_string())

    # Langkah 5: Visualisasi peta statis
    print("\n[LANGKAH 5] Membuat peta statis (PNG)...")
    create_static_map(
        gdf_wilayah, gdf_perkotaan, gdf_facilities_bb,
        output_path=os.path.join(OUTPUT_DIR, "peta_wilayah_fasilitas.png")
    )

    # Langkah 6: Visualisasi peta interaktif
    print("\n[LANGKAH 6] Membuat peta interaktif (HTML)...")
    create_interactive_map(
        gdf_wilayah, gdf_facilities_bb,
        output_path=os.path.join(OUTPUT_DIR, "peta_interaktif.html")
    )

    print(f"\n[SELESAI] Output tersimpan di folder: '{os.path.abspath(OUTPUT_DIR)}/'")
    print("=" * 65)


if __name__ == "__main__":
    main()
