"""
cnn_model.py
============
Model Convolutional Neural Network (CNN) sederhana menggunakan PyTorch
untuk klasifikasi citra sebagai proksi analisis citra satelit.

Dataset: CIFAR-10 (proksi untuk klasifikasi penutupan lahan)
Kelas CIFAR-10 yang relevan sebagai analogi remote sensing:
  - airplane  → Infrastruktur udara
  - automobile → Jalan/infrastruktur darat
  - bird       → Area terbuka/ruang hijau
  - cat/deer/horse/dog → Area vegetasi/fauna
  - frog       → Area basah/rawa
  - ship       → Perairan
  - truck      → Infrastruktur industri

Alur kerja:
1. Definisi arsitektur CNN (SimpleCNN)
2. Load dan preprocessing dataset CIFAR-10
3. Training loop dengan optimizer Adam
4. Fungsi evaluasi akurasi pada data uji
5. Visualisasi kurva training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================

OUTPUT_DIR = "sesi_6/output"
DATA_DIR   = "sesi_6/data"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
BATCH_SIZE   = 64
NUM_EPOCHS   = 10       # Kurangi jika tidak ada GPU; full training butuh 30+ epoch
LEARNING_RATE = 0.001
NUM_CLASSES  = 10       # CIFAR-10 memiliki 10 kelas
NUM_WORKERS  = 0        # 0 untuk kompatibilitas Windows

# Nama kelas CIFAR-10 (analog remote sensing)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"[INFO] Menggunakan device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  CPU mode (training akan lebih lambat)")


# ==============================================================================
# 1. ARSITEKTUR CNN
# ==============================================================================

class SimpleCNN(nn.Module):
    """
    Arsitektur CNN sederhana untuk klasifikasi citra 32×32×3.
    
    Struktur arsitektur:
    ┌─────────────────────────────────────────────────┐
    │  Input: (N, 3, 32, 32)                          │
    │  ↓                                              │
    │  Conv1 (3→32, 3×3) → BN → ReLU → MaxPool(2×2)  │
    │  ↓                                              │
    │  Conv2 (32→64, 3×3) → BN → ReLU → MaxPool(2×2) │
    │  ↓                                              │
    │  Conv3 (64→128, 3×3) → BN → ReLU              │
    │  ↓                                              │
    │  GlobalAvgPool → Flatten                        │
    │  ↓                                              │
    │  FC1 (128→256) → ReLU → Dropout(0.5)           │
    │  ↓                                              │
    │  FC2 (256→10) → Output logits                  │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # --- BLOK KONVOLUSIONAL 1 ---
        # Mengekstraksi fitur low-level (tepi, tekstur)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),       # Stabilisasi training
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 → 16x16
        )

        # --- BLOK KONVOLUSIONAL 2 ---
        # Mengekstraksi fitur mid-level (pola, bentuk)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 → 8x8
        )

        # --- BLOK KONVOLUSIONAL 3 ---
        # Mengekstraksi fitur high-level (objek, konsep)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # GlobalAveragePooling menggantikan flatten besar → mengurangi parameter
        )

        # Global Average Pooling: menggantikan Flatten konvensional
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # → (N, 128, 1, 1)

        # --- BLOK FULLY CONNECTED ---
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),        # Regularisasi untuk mencegah overfitting
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass dari input citra ke logits output.
        
        Parameters:
            x: Tensor input dengan shape (N, 3, 32, 32)
        
        Returns:
            Tensor logits dengan shape (N, num_classes)
        """
        # Ekstraksi fitur melalui blok konvolusional
        x = self.conv_block1(x)    # → (N, 32, 16, 16)
        x = self.conv_block2(x)    # → (N, 64, 8, 8)
        x = self.conv_block3(x)    # → (N, 128, 8, 8)
        x = self.global_avg_pool(x)  # → (N, 128, 1, 1)
        x = x.view(x.size(0), -1)   # Flatten → (N, 128)

        # Klasifikasi
        x = self.classifier(x)    # → (N, num_classes)
        return x


# ==============================================================================
# 2. LOAD & PREPROCESSING DATASET CIFAR-10
# ==============================================================================

def get_data_loaders(batch_size: int = BATCH_SIZE) -> tuple:
    """
    Memuat dataset CIFAR-10 dengan augmentasi data untuk training
    dan normalisasi standar untuk validasi/testing.
    
    Augmentasi training:
    - RandomHorizontalFlip: simulasikan variasi orientasi citra satelit
    - RandomCrop: simulasikan crop/patch berbeda dari citra besar
    - ColorJitter: simulasikan variasi pencahayaan pada citra satelit
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Transformasi untuk data TRAINING (dengan augmentasi)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),    # Padding + crop acak
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        # Normalisasi menggunakan mean/std dari dataset CIFAR-10
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # Transformasi untuk data TESTING (hanya normalisasi, tanpa augmentasi)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # Download dan load dataset
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[INFO] Memuat dataset CIFAR-10 (download otomatis jika belum ada)...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"  ✔ Training set: {len(train_dataset):,} citra")
    print(f"  ✔ Testing set : {len(test_dataset):,} citra")
    print(f"  ✔ Batch size  : {batch_size}")

    return train_loader, test_loader


# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> tuple:
    """
    Menjalankan satu epoch training.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass dan update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Akumulasi metrik
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress setiap 100 batch
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {100.*correct/total:.2f}%", end="\r")

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs: int = NUM_EPOCHS) -> dict:
    """
    Menjalankan loop training lengkap selama num_epochs epoch.
    
    Parameters:
        model: Model CNN yang akan dilatih.
        train_loader: DataLoader untuk training.
        test_loader: DataLoader untuk testing/evaluasi.
        num_epochs (int): Jumlah epoch training.
    
    Returns:
        dict: Riwayat loss dan akurasi training & testing per epoch.
    """
    model = model.to(DEVICE)

    # Optimizer: Adam dengan weight decay untuk regularisasi L2
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler: kurangi LR jika performa stagnan
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loss function: CrossEntropyLoss (multi-class classification)
    criterion = nn.CrossEntropyLoss()

    # Riwayat training
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": [],
    }

    best_test_acc = 0.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MEMULAI TRAINING CNN ({num_epochs} Epoch)")
    print(f"  Device: {DEVICE} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch:02d}/{num_epochs}]")

        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluasi pada test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)

        # Update scheduler
        scheduler.step()

        # Simpan riwayat
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        epoch_time = time.time() - epoch_start
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
        print(f"  Overfitting Gap: {train_acc - test_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Waktu epoch: {epoch_time:.1f}s")

        # Simpan model terbaik
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_cnn_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, checkpoint_path)
            print(f"  ✔ Model terbaik disimpan (Test Acc: {best_test_acc:.2f}%)")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training selesai dalam {total_time:.1f}s ({total_time/60:.1f} menit)")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"{'='*60}")

    return history


# ==============================================================================
# 4. FUNGSI EVALUASI
# ==============================================================================

def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> tuple:
    """
    Mengevaluasi model pada dataset (biasanya test set).
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Nonaktifkan gradient untuk efisiensi
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_per_class(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """
    Evaluasi akurasi model per kelas (penting untuk mendeteksi bias model).
    """
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label]   += 1
                class_correct[label] += (predicted[i] == labels[i]).item()

    print("\n[AKURASI PER KELAS]")
    print(f"  {'Kelas':<15} {'Benar':>8} {'Total':>8} {'Akurasi':>10}")
    print("  " + "-" * 45)
    for i, cls in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {cls:<15} {class_correct[i]:>8} {class_total[i]:>8} {acc:>9.1f}%")


# ==============================================================================
# 5. VISUALISASI KURVA TRAINING
# ==============================================================================

def plot_training_curves(history: dict, output_dir: str) -> None:
    """
    Membuat plot kurva training (loss dan accuracy) dan menyimpannya sebagai PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curve
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, history["test_loss"],  "r-o", label="Testing Loss",  linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (CrossEntropyLoss)", fontsize=12)
    ax1.set_title("Kurva Training Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(epochs)

    # Plot 2: Accuracy curve
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy", linewidth=2, markersize=4)
    ax2.plot(epochs, history["test_acc"],  "r-o", label="Testing Accuracy",  linewidth=2, markersize=4)
    ax2.fill_between(epochs, history["train_acc"], history["test_acc"],
                     alpha=0.1, color="purple", label="Overfitting Gap")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Kurva Akurasi Training vs Testing\n(Visualisasi Overfitting)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xticks(epochs)

    plt.suptitle(f"SimpleCNN Training pada CIFAR-10\n"
                 f"(Proxy untuk Klasifikasi Citra Satelit/Remote Sensing)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✔ Kurva training disimpan: '{save_path}'")
    plt.show()


def visualize_sample_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_dir: str) -> None:
    """
    Visualisasikan prediksi model pada beberapa sampel citra.
    """
    model.eval()
    
    # Ambil satu batch
    images, labels = next(iter(loader))
    images_device = images[:16].to(device)
    
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = outputs.max(1)
    
    # De-normalisasi untuk visualisasi
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images_display = images[:16] * std + mean
    images_display = images_display.clamp(0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = images_display[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        true_label = CLASS_NAMES[labels[i].item()]
        pred_label = CLASS_NAMES[predicted[i].item()]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=7, color=color)
        ax.axis("off")
    
    plt.suptitle("Contoh Prediksi Model CNN\n(Hijau=Benar, Merah=Salah)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✔ Visualisasi prediksi disimpan: '{save_path}'")
    plt.show()


# ==============================================================================
# 6. RINGKASAN ARSITEKTUR
# ==============================================================================

def print_model_summary(model: nn.Module) -> None:
    """
    Menampilkan ringkasan arsitektur model dan jumlah parameter.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[ARSITEKTUR MODEL]")
    print(model)
    print(f"\n  Total Parameter    : {total_params:,}")
    print(f"  Parameter Trainable: {trainable_params:,}")
    print(f"  Ukuran model (approx): {total_params * 4 / 1024 / 1024:.2f} MB")


# ==============================================================================
# 7. MAIN PIPELINE
# ==============================================================================

def run_cnn_pipeline():
    """
    Menjalankan seluruh pipeline CNN dari inisialisasi model hingga evaluasi.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE CNN KLASIFIKASI CITRA (Remote Sensing Proxy)")
    print(f"  Dataset: CIFAR-10 | Device: {DEVICE}")
    print("=" * 60)

    # LANGKAH 1: Inisialisasi model
    print("\n[LANGKAH 1] Inisialisasi arsitektur CNN...")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    print_model_summary(model)

    # LANGKAH 2: Load dataset
    print("\n[LANGKAH 2] Memuat dataset CIFAR-10...")
    train_loader, test_loader = get_data_loaders()

    # LANGKAH 3: Training
    print("\n[LANGKAH 3] Memulai training...")
    history = train_model(model, train_loader, test_loader, num_epochs=NUM_EPOCHS)

    # LANGKAH 4: Evaluasi akhir per kelas
    print("\n[LANGKAH 4] Evaluasi per kelas...")
    evaluate_per_class(model, test_loader, DEVICE)

    # LANGKAH 5: Visualisasi kurva training
    print("\n[LANGKAH 5] Membuat visualisasi kurva training...")
    plot_training_curves(history, OUTPUT_DIR)

    # LANGKAH 6: Visualisasi prediksi sampel
    print("\n[LANGKAH 6] Visualisasi prediksi sampel...")
    visualize_sample_predictions(model, test_loader, DEVICE, OUTPUT_DIR)

    print(f"\n[SELESAI] Pipeline CNN selesai. Output di: '{os.path.abspath(OUTPUT_DIR)}'")
    return model, history


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    model, history = run_cnn_pipeline()
