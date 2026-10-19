# UTS Deep Learning dan IoT —  Study Cases



---

## 📁 Struktur Proyek

```
UTS-DeepL-dan-IoT/
├── sesi_1/    → Customer Churn Predictive Analytics
├── sesi_2/    → Modular Environmental Data Pipeline
├── sesi_3/    → Spatial Data Processing & GIS Mapping
├── sesi_4/    → Supervised ML Pipeline & Overfitting Analysis
├── sesi_5/    → Interactive Streamlit Dashboard (HCI Shneiderman)
└── sesi_6/    → CNN Deep Learning untuk Klasifikasi Citra Satelit
```

---

## 🗂️ Detail Setiap Sesi

### Sesi I — Patterns of Data with Analysis and AI
**Studi Kasus**: Customer Churn Predictive Analytics & Data Cleaning Pipeline

- `data_preprocessing.py` — Load dataset, handle missing values (median/modus), normalisasi, encoding kategorikal
- `train_model.py` — Logistic Regression + Random Forest, Confusion Matrix, Accuracy Score
- `requirements.txt` — pandas, numpy, scikit-learn, matplotlib

```bash
cd sesi_1
pip install -r requirements.txt
python train_model.py
```

---

### Sesi II — Global Infrastructure to Address Local Challenge with HPC
**Studi Kasus**: Modular Pipeline untuk Analisis Data Kualitas Udara/Cuaca Lokal

- `fetcher.py` — HTTP request ke OpenWeatherMap API + fallback dummy + retry/error handling
- `processor.py` — Transformasi JSON mentah → DataFrame Pandas terstruktur
- `main.py` — Orkestrasi pipeline: Fetch → Transform → Save CSV + SQLite
- `requirements.txt` — requests, pandas, numpy

```bash
cd sesi_2
pip install -r requirements.txt
python main.py
```

> **Catatan**: Set `OPENWEATHER_API_KEY` di `fetcher.py` untuk data live. Default menggunakan data dummy.

---

### Sesi III — Career and Competition in Industrial Era with GIS
**Studi Kasus**: Spatial Data Processing & Mapping menggunakan Python & GeoPandas

- `spatial_analysis.py` — Generate GeoJSON dummy wilayah Jawa, spatial filter (atribut + bounding box), spatial join, peta statis PNG + interaktif HTML (Folium)
- `requirements.txt` — geopandas, shapely, matplotlib, folium

```bash
cd sesi_3
pip install -r requirements.txt
python spatial_analysis.py
```

**Output**: `output/peta_wilayah_fasilitas.png` dan `output/peta_interaktif.html`

---

### Sesi IV — Introduction to Machine Learning
**Studi Kasus**: Supervised Learning Pipeline & Model Evaluation untuk Klasifikasi

- `ml_pipeline.py` — Dataset Breast Cancer, Decision Tree + SVM, train-test split 80:20, StandardScaler, deteksi overfitting via accuracy gap, classification_report
- `requirements.txt` — scikit-learn, pandas, numpy, matplotlib

```bash
cd sesi_4
pip install -r requirements.txt
python ml_pipeline.py
```

---

### Sesi V — Human-Computer Interaction & Data Visualization
**Studi Kasus**: Interactive Data Dashboard berbasis Streamlit (Prinsip Shneiderman)

- `app.py` — Dashboard e-commerce interaktif: sidebar filter (Zoom & Filter), KPI cards + 4 grafik Plotly (Overview First), tabel detail + download CSV (Details on Demand)
- `requirements.txt` — streamlit, plotly, pandas, numpy

```bash
cd sesi_5
pip install -r requirements.txt
streamlit run app.py
```

---

### Sesi VI — Deforestation Alerts and Driver Attribution with Remote Sensing and AI
**Studi Kasus**: Simple Deep Learning (CNN) untuk Klasifikasi Patch Citra Satelit

- `cnn_model.py` — Arsitektur SimpleCNN (Conv+BN+ReLU × 3, GlobalAvgPool, FC+Dropout), dataset CIFAR-10 sebagai proksi citra satelit, training loop Adam + CrossEntropyLoss + LR Scheduler, evaluasi per kelas, visualisasi kurva training
- `requirements.txt` — torch, torchvision, matplotlib, numpy

```bash
cd sesi_6
pip install -r requirements.txt
python cnn_model.py
```

> **Catatan**: Training CIFAR-10 membutuhkan ~20-30 menit di CPU. GPU sangat direkomendasikan.

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|---------|
| Python 3.10+ | Bahasa pemrograman utama |
| Scikit-Learn | ML pipeline, preprocessing, evaluasi |
| PyTorch + TorchVision | Deep Learning CNN |
| Streamlit + Plotly | Dasbor interaktif |
| GeoPandas + Folium | Analisis dan visualisasi data spasial |
| Pandas + NumPy | Manipulasi dan analisis data |
| Requests | HTTP client untuk API |
| SQLite | Penyimpanan database lokal |

---

## 👤 Informasi

- **Mata Kuliah**: Deep Learning dan IoT
- **Jenis Tugas**: UTS (Ujian Tengah Semester)

