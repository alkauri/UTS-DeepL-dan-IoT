"""
train_model.py
==============
Script untuk melatih dan mengevaluasi model prediksi Customer Churn.

Alur kerja:
1. Load dataset bersih dari data_preprocessing.py
2. Split data menjadi training dan testing set
3. Latih model Logistic Regression & Random Forest
4. Evaluasi dengan Confusion Matrix dan Accuracy Score
5. Tampilkan hasil perbandingan performa kedua model
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# Import pipeline preprocessing
from data_preprocessing import preprocess_pipeline


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

def load_data(csv_path: str = "sesi_1/data_cleaned.csv") -> tuple:
    """
    Memuat dataset bersih dan memisahkan fitur (X) dari label target (y).
    
    Parameters:
        csv_path (str): Path ke file CSV yang sudah bersih.
    
    Returns:
        tuple: (X, y) sebagai DataFrame fitur dan Series target.
    """
    # Jika file belum ada, jalankan pipeline preprocessing terlebih dahulu
    if not os.path.exists(csv_path):
        print("[INFO] File data bersih tidak ditemukan. Menjalankan preprocessing...")
        preprocess_pipeline(output_path=csv_path)

    df = pd.read_csv(csv_path)
    print(f"[INFO] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Pisahkan fitur dan target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"[INFO] Distribusi kelas target:\n{y.value_counts().to_string()}")
    return X, y


# ==============================================================================
# 2. SPLIT DATA
# ==============================================================================

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Membagi dataset menjadi training set dan testing set.
    
    Parameters:
        X (pd.DataFrame): DataFrame fitur.
        y (pd.Series): Series label target.
        test_size (float): Proporsi data uji (default 20%).
        random_state (int): Seed untuk reproduktibilitas.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[INFO] Data splitting selesai:")
    print(f"  Training set : {X_train.shape[0]} sampel ({(1-test_size)*100:.0f}%)")
    print(f"  Testing set  : {X_test.shape[0]} sampel ({test_size*100:.0f}%)")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 3. PELATIHAN DAN EVALUASI MODEL
# ==============================================================================

def train_and_evaluate(model, model_name: str, X_train, X_test, y_train, y_test) -> dict:
    """
    Melatih sebuah model dan mengevaluasi performanya.
    
    Parameters:
        model: Objek model Scikit-Learn.
        model_name (str): Nama model untuk ditampilkan.
        X_train, X_test: Fitur training dan testing.
        y_train, y_test: Label training dan testing.
    
    Returns:
        dict: Dictionary berisi nama model, akurasi, dan confusion matrix.
    """
    print(f"\n{'='*50}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*50}")

    # Latih model
    model.fit(X_train, y_train)
    print(f"  ✔ Model selesai dilatih.")

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy Score : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return {"model_name": model_name, "accuracy": accuracy, "cm": cm, "model": model}


# ==============================================================================
# 4. VISUALISASI CONFUSION MATRIX
# ==============================================================================

def plot_confusion_matrices(results: list, output_dir: str = "sesi_1/"):
    """
    Membuat dan menyimpan visualisasi Confusion Matrix untuk semua model.
    
    Parameters:
        results (list): List dictionary hasil evaluasi tiap model.
        output_dir (str): Direktori penyimpanan gambar.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=result["cm"],
            display_labels=["No Churn", "Churn"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.2%}", fontsize=13)

    plt.suptitle("Perbandingan Confusion Matrix Model Churn Prediction", fontsize=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "confusion_matrix_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Confusion Matrix disimpan ke: '{save_path}'")
    plt.show()


# ==============================================================================
# 5. PIPELINE PELATIHAN UTAMA
# ==============================================================================

def training_pipeline():
    """
    Menjalankan seluruh pipeline pelatihan model dari load data hingga evaluasi.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE PELATIHAN MODEL - CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Definisi model yang akan dilatih
    models = [
        (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
    ]

    # Latih dan evaluasi setiap model
    results = []
    for model, name in models:
        result = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        results.append(result)

    # Bandingkan hasil semua model
    print("\n" + "=" * 60)
    print("  RINGKASAN PERBANDINGAN MODEL")
    print("=" * 60)
    best = max(results, key=lambda r: r["accuracy"])
    for r in results:
        marker = " ← TERBAIK" if r["model_name"] == best["model_name"] else ""
        print(f"  {r['model_name']:<25} | Accuracy: {r['accuracy']:.4f}{marker}")

    # Visualisasi confusion matrix
    plot_confusion_matrices(results)

    print("\n[SELESAI] Pipeline pelatihan model selesai.")
    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    training_pipeline()
