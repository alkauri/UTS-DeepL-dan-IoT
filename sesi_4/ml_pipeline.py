"""
ml_pipeline.py
==============
Pipeline Machine Learning end-to-end untuk klasifikasi supervised learning
menggunakan dataset standar Scikit-Learn.

Alur kerja:
1. Load dataset klasifikasi (Breast Cancer)
2. Train-test split (80:20) + Standarisasi fitur
3. Latih Decision Tree dan SVM
4. Evaluasi Training vs Testing Accuracy (deteksi overfitting)
5. Tampilkan classification_report lengkap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os


# ==============================================================================
# KONFIGURASI
# ==============================================================================

OUTPUT_DIR = "sesi_4/output"
RANDOM_STATE = 42
TEST_SIZE = 0.20    # 80% training, 20% testing

# Pilih dataset: "breast_cancer" atau "iris"
DATASET_NAME = "breast_cancer"


# ==============================================================================
# 1. LOAD DATASET
# ==============================================================================

def load_dataset(name: str = "breast_cancer") -> tuple:
    """
    Memuat dataset klasifikasi standar dari Scikit-Learn.
    
    Parameters:
        name (str): Nama dataset ('breast_cancer' atau 'iris').
    
    Returns:
        tuple: (X, y, feature_names, target_names, description)
    """
    loaders = {
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
    }

    if name not in loaders:
        raise ValueError(f"Dataset '{name}' tidak dikenali. Pilih: {list(loaders.keys())}")

    dataset = loaders[name]()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target

    print(f"\n[INFO] Dataset: {name.upper()}")
    print(f"  Jumlah sampel   : {X.shape[0]}")
    print(f"  Jumlah fitur    : {X.shape[1]}")
    print(f"  Kelas target    : {list(dataset.target_names)}")
    print(f"  Distribusi kelas:")
    for i, cls in enumerate(dataset.target_names):
        count = np.sum(y == i)
        print(f"    [{i}] {cls}: {count} sampel ({count/len(y)*100:.1f}%)")

    return X, y, dataset.feature_names, dataset.target_names


# ==============================================================================
# 2. PREPROCESSING: SPLIT & STANDARDISASI
# ==============================================================================

def preprocess_data(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Melakukan split data dan standarisasi fitur menggunakan StandardScaler.
    Scaler di-fit HANYA pada data training untuk mencegah data leakage.
    
    Parameters:
        X (pd.DataFrame): Matriks fitur.
        y (np.ndarray): Array label target.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Split data (stratified agar distribusi kelas proporsional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n[INFO] Data Splitting (Train:Test = {int((1-TEST_SIZE)*100)}:{int(TEST_SIZE*100)}):")
    print(f"  Training : {X_train.shape[0]} sampel")
    print(f"  Testing  : {X_test.shape[0]} sampel")

    # Standarisasi (Z-score normalization): fit HANYA pada train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)  # Gunakan parameter dari train

    print("  ✔ Standarisasi fitur selesai (fit pada train, transform pada test).")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==============================================================================
# 3. PELATIHAN & EVALUASI MODEL (dengan deteksi overfitting)
# ==============================================================================

def train_evaluate_with_overfitting_check(
        model, model_name: str,
        X_train, X_test,
        y_train, y_test,
        target_names: list) -> dict:
    """
    Melatih model, mengevaluasi performa, dan mendeteksi potensi overfitting
    dengan membandingkan Training Accuracy vs Testing Accuracy.
    
    Parameters:
        model: Objek model Scikit-Learn.
        model_name (str): Nama model.
        X_train, X_test: Fitur training & testing (sudah di-scale).
        y_train, y_test: Label training & testing.
        target_names (list): Nama kelas untuk laporan.
    
    Returns:
        dict: Ringkasan metrik evaluasi model.
    """
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    # Latih model
    model.fit(X_train, y_train)

    # Prediksi pada data training dan testing
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Hitung akurasi
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy  = accuracy_score(y_test, y_pred_test)
    gap = train_accuracy - test_accuracy

    print(f"\n  Training Accuracy : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Testing Accuracy  : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Accuracy Gap      : {gap:.4f} ({gap*100:.2f}%)")

    # Analisis overfitting berdasarkan gap
    if gap > 0.10:
        status = "⚠️  OVERFITTING TERDETEKSI (gap > 10%)"
    elif gap > 0.05:
        status = "⚡ Potensi overfitting ringan (gap 5-10%)"
    else:
        status = "✅ Model tergeneralisasi dengan baik (gap ≤ 5%)"
    print(f"  Status Overfitting: {status}")

    # Cross-validation untuk estimasi performa yang lebih robust
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    print(f"\n  Classification Report (Testing Set):")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    return {
        "model_name": model_name,
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "gap": gap,
        "overfitting_status": status,
        "cm": cm,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


# ==============================================================================
# 4. VISUALISASI
# ==============================================================================

def plot_overfitting_comparison(results: list, output_dir: str) -> None:
    """
    Membuat bar chart perbandingan Training vs Testing Accuracy untuk semua model.
    """
    model_names   = [r["model_name"] for r in results]
    train_accs    = [r["train_accuracy"] for r in results]
    test_accs     = [r["test_accuracy"] for r in results]
    x = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Bar chart training vs testing accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, train_accs, width, label="Training Accuracy",
                     color="#4ECDC4", alpha=0.85)
    bars2 = ax1.bar(x + width/2, test_accs, width, label="Testing Accuracy",
                     color="#FF6B35", alpha=0.85)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Training vs Testing Accuracy\n(Deteksi Overfitting)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.set_ylim(0.7, 1.05)
    ax1.legend(fontsize=10)
    ax1.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax1.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Confusion Matrix side by side
    # Hanya tampilkan confusion matrix dari model pertama sebagai contoh
    ax2 = axes[1]
    best_model = max(results, key=lambda r: r["test_accuracy"])
    disp = ConfusionMatrixDisplay(confusion_matrix=best_model["cm"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title(f"Confusion Matrix\n{best_model['model_name']} (Best Model)",
                  fontsize=13, fontweight="bold")

    plt.suptitle("Evaluasi Pipeline Machine Learning - Klasifikasi", fontsize=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ml_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✔ Grafik evaluasi disimpan: '{save_path}'")
    plt.show()


# ==============================================================================
# 5. PIPELINE UTAMA
# ==============================================================================

def run_ml_pipeline():
    """
    Menjalankan seluruh pipeline Machine Learning dari load data hingga evaluasi.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE ML SUPERVISED CLASSIFICATION")
    print(f"  Dataset: {DATASET_NAME.upper()}")
    print("=" * 60)

    # LANGKAH 1: Load data
    X, y, feature_names, target_names = load_dataset(DATASET_NAME)

    # LANGKAH 2: Preprocessing
    print("\n[LANGKAH 2] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # LANGKAH 3: Definisi model
    models = [
        (DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE), "Decision Tree (Full)"),
        (DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), "Decision Tree (max_depth=5)"),
        (SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE), "SVM (RBF Kernel)"),
    ]

    # LANGKAH 4: Latih dan evaluasi
    print("\n[LANGKAH 3] Melatih dan mengevaluasi model...")
    results = []
    for model, name in models:
        result = train_evaluate_with_overfitting_check(
            model, name, X_train, X_test, y_train, y_test, list(target_names)
        )
        results.append(result)

    # LANGKAH 5: Ringkasan perbandingan
    print("\n" + "=" * 60)
    print("  RINGKASAN PERBANDINGAN MODEL")
    print("=" * 60)
    print(f"  {'Model':<30} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8} Status")
    print("  " + "-" * 75)
    best = max(results, key=lambda r: r["test_accuracy"])
    for r in results:
        marker = " ← BEST" if r["model_name"] == best["model_name"] else ""
        status_icon = "⚠️" if r["gap"] > 0.10 else ("⚡" if r["gap"] > 0.05 else "✅")
        print(f"  {r['model_name']:<30} {r['train_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} "
              f"{r['gap']:>8.4f} {status_icon}{marker}")

    # LANGKAH 6: Visualisasi
    print("\n[LANGKAH 4] Membuat visualisasi evaluasi...")
    plot_overfitting_comparison(results, OUTPUT_DIR)

    print(f"\n[SELESAI] Pipeline ML selesai. Output di: '{os.path.abspath(OUTPUT_DIR)}'")
    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    run_ml_pipeline()
