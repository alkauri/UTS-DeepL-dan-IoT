"""
data_preprocessing.py
======================
Script untuk memuat, membersihkan, dan mempersiapkan dataset Customer Churn
untuk pemodelan prediktif.

Alur kerja:
1. Load dataset (simulasi Telco Customer Churn)
2. Tangani missing values
3. Normalisasi fitur numerik
4. Encoding fitur kategorikal
5. Simpan hasil ke CSV yang sudah bersih
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ==============================================================================
# 1. GENERATE DATASET DUMMY (mensimulasikan Telco Customer Churn)
# ==============================================================================

def generate_dummy_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Membuat dataset dummy yang mensimulasikan struktur Telco Customer Churn.
    
    Parameters:
        n_samples (int): Jumlah baris data yang akan dibuat.
        random_state (int): Seed untuk reproduktibilitas.
    
    Returns:
        pd.DataFrame: Dataset mentah dengan missing values dan fitur campuran.
    """
    np.random.seed(random_state)

    df = pd.DataFrame({
        # Fitur demografi
        "CustomerID": [f"CUST-{i:04d}" for i in range(1, n_samples + 1)],
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),

        # Fitur layanan
        "Tenure": np.random.randint(0, 73, n_samples),           # bulan berlangganan
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
        ),

        # Fitur tagihan
        "MonthlyCharges": np.round(np.random.uniform(18, 120, n_samples), 2),
        "TotalCharges": np.round(np.random.uniform(20, 8700, n_samples), 2),

        # Label target
        "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.265, 0.735]),
    })

    # Sengaja sisipkan missing values (~5%) untuk simulasi kondisi data nyata
    for col in ["MonthlyCharges", "TotalCharges", "Tenure"]:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


# ==============================================================================
# 2. FUNGSI PEMBERSIHAN DATA (DATA CLEANING)
# ==============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values menggunakan strategi imputasi yang sesuai:
    - Fitur numerik  → diisi dengan median kolom
    - Fitur kategori → diisi dengan modus kolom
    
    Parameters:
        df (pd.DataFrame): DataFrame yang memiliki missing values.
    
    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    print("\n[INFO] Missing values sebelum imputasi:")
    print(df.isnull().sum())

    # Imputasi numerik dengan median (robust terhadap outlier)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  ✔ '{col}' diisi dengan median: {median_val:.2f}")

    # Imputasi kategorikal dengan modus
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  ✔ '{col}' diisi dengan modus: '{mode_val}'")

    print("\n[INFO] Missing values setelah imputasi:")
    print(df.isnull().sum())
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah fitur kategorikal menjadi representasi numerik menggunakan:
    - Label Encoding untuk fitur biner (Gender, Churn)
    - One-Hot Encoding untuk fitur multi-kelas
    
    Parameters:
        df (pd.DataFrame): DataFrame yang sudah bebas dari missing values.
    
    Returns:
        pd.DataFrame: DataFrame dengan fitur yang sudah di-encode.
    """
    le = LabelEncoder()

    # Label Encoding untuk fitur biner
    binary_cols = ["Gender", "Churn"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            print(f"  ✔ '{col}' di-label-encode: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-Hot Encoding untuk fitur multi-kelas
    multi_class_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
    print(f"  ✔ One-Hot Encoding diterapkan pada: {multi_class_cols}")

    return df


def normalize_numerical_features(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """
    Menstandarisasi fitur numerik menggunakan StandardScaler (mean=0, std=1).
    
    Parameters:
        df (pd.DataFrame): DataFrame setelah encoding.
        exclude_cols (list): Kolom yang dikecualikan dari normalisasi (ID, target, dll.).
    
    Returns:
        pd.DataFrame: DataFrame dengan fitur numerik yang sudah dinormalisasi.
    """
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"  ✔ Normalisasi diterapkan pada: {cols_to_scale}")

    return df


# ==============================================================================
# 3. PIPELINE UTAMA
# ==============================================================================

def preprocess_pipeline(output_path: str = "sesi_1/data_cleaned.csv") -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing dari awal hingga akhir.
    
    Returns:
        pd.DataFrame: Dataset bersih yang siap digunakan untuk pelatihan model.
    """
    print("=" * 60)
    print("  PIPELINE DATA PREPROCESSING - CUSTOMER CHURN")
    print("=" * 60)

    # Langkah 1: Load / Generate dataset
    print("\n[LANGKAH 1] Membuat dataset dummy...")
    df = generate_dummy_dataset(n_samples=1000)
    print(f"  ✔ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Langkah 2: Hapus kolom ID (tidak relevan untuk model)
    df.drop(columns=["CustomerID"], inplace=True)

    # Langkah 3: Tangani missing values
    print("\n[LANGKAH 2] Menangani missing values...")
    df = handle_missing_values(df)

    # Langkah 4: Encode fitur kategorikal
    print("\n[LANGKAH 3] Encoding fitur kategorikal...")
    df = encode_categorical_features(df)

    # Langkah 5: Normalisasi fitur numerik (kecuali target 'Churn')
    print("\n[LANGKAH 4] Normalisasi fitur numerik...")
    df = normalize_numerical_features(df, exclude_cols=["Churn", "SeniorCitizen"])

    # Langkah 6: Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[SELESAI] Dataset bersih disimpan ke: '{output_path}'")
    print(f"  Dimensi akhir: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("=" * 60)

    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    cleaned_df = preprocess_pipeline()
    print("\n[PREVIEW] 5 baris pertama dataset bersih:")
    print(cleaned_df.head())
