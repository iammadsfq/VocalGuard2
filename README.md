# Vocal-Guard: Spoofing-Aware Audio Front-End untuk Robot Sosial

Evaluasi empiris dan re-engineering sistem anti-spoofing suara berbasis fitur akustik ringan pada benchmark resmi **ASVspoof 2019 Logical Access (LA)** untuk verifikasi suara on-device pada robot sosial.

---

## 📊 Hasil Evaluasi Resmi (ASVspoof 2019 LA Dev Set)

| Model | Arsitektur / Fitur | Parameter | Ukuran Model | EER Dev (%) | Status Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original System** | MobileNetV2 (Frozen ImageNet, GAP, Resize 224x224) | ~3.5 M | ~14 MB | **41.22%** | Failed (Gagal Acak) |
| **Baseline B0** | LFCC-GMM (256 komponen, 60-dim LFCC) | N/A | ~966 KB | **32.18%** | Gate G1 Passed ✅ |
| **Model M1 (Ours)** | **LCNN (Light CNN + MFM)** pada Native LFCC (60-dim) | **539,714** | **2.1 MB** | **0.37%** 🏆 | **Gate G2 Passed ✅** |

> **Analisis:** Mengganti arsitektur visual ImageNet yang dibekukan dengan LCNN (Light CNN berbasis Max-Feature-Map) serta mempertahankan resolusi asli spektro-temporal LFCC berhasil menurunkan EER dari **41.22%** menjadi **0.37%** (penurunan relatif > 99%).

---

## 🧠 Temuan Diagnostik & Hipotesis (H1–H7)

Sistem awal Vocal-Guard mengalami kegagalan performa (EER 41.22%) akibat beberapa kesalahan perancangan mendasar:
1. **H1 (Frozen ImageNet Backbone):** MobileNetV2 dibekukan (`trainable = False`), memaksa model mengklasifikasi suara menggunakan representasi tekstur foto visual yang tidak relevan dengan artefak sintetis audio.
2. **H2 (Global Average Pooling):** Merata-ratakan seluruh sumbu waktu melenyapkan transien fase milidetik dan artefak vocoder frekuensi tinggi. LCNN mempertahankan dimensi spasial lokal dengan konvolusi berstrata dan MFM selektif.
3. **H3 (Distorted Resizing):** Melakukan interpolasi 20 koefisien LFCC ke 224 piksel merusak integritas representasi frekuensi linier.
4. **H4 (Feature Parity):** Training awal menggunakan `dct(stft)` langsung tanpa filterbank linier. Versi saat ini menggunakan implementasi referensi tunggal (*Single Source of Truth*) LFCC 60 dimensi (20 statis + 20 delta + 20 delta-delta).

---

## 🛠️ Struktur Repositori

```text
vocal_guard/
├── features.py          # Single source of truth untuk ekstraksi LFCC 60-dim
├── protocol.py          # Parser resmi protokol ASVspoof 2019 LA
├── metrics.py           # Perhitungan metrik resmi EER dan threshold
├── dataset.py           # PyTorch Dataset (4-detik wrap/truncate)
├── baseline_gmm.py      # PyTorch GPU-accelerated LFCC-GMM baseline
└── models/
    └── lcnn.py          # Light CNN (LCNN) dengan Max-Feature-Map (MFM)
notebooks/
├── 01_baseline_b0_gmm.ipynb    # Reproduksi Baseline B0 di Kaggle
└── 02_train_lcnn_m1.ipynb      # Training LCNN M1 dari nol di Kaggle GPU
results/
├── b0_evaluation_log.csv       # Log hasil Baseline B0
├── m1_evaluation_log.csv       # Log hasil Model M1
└── lcnn_training_history.csv   # History 10 epoch training LCNN
models/
├── gmm_bonafide.pkl     # Bobot GMM Bonafide (256 komponen)
├── gmm_spoof.pkl        # Bobot GMM Spoof (256 komponen)
└── lcnn_best_model.pt   # Checkpoint PyTorch LCNN terbaik (EER 0.37%)
```

---

## 🚀 Cara Reproduksi

### 1. Lingkungan Lokal & Dependensi
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Inferensi Menggunakan Model Checkpoint
```python
import torch
from vocal_guard.models.lcnn import LCNN

# Load model LCNN
model = LCNN(num_classes=2)
model.load_state_dict(torch.load("models/lcnn_best_model.pt", map_location="cpu"))
model.eval()

# Input format: (Batch, 1, 60, T)
x = torch.randn(1, 1, 60, 399)
with torch.no_grad():
    logits = model(x)
    # Skor bonafide vs spoof (skor positif = bonafide, negatif = spoof)
    score = (logits[:, 0] - logits[:, 1]).item()
    prediction = "bonafide" if score > -1.1896 else "spoof"
    print(f"Prediction: {prediction} (Score: {score:.4f})")
```

---

## ⚡ Kuantisasi & Benchmark Inferensi CPU On-Device (RQ3)

Pengujian latensi dan kuantisasi dilakukan pada **Intel Core i7-8665U @ 1.90GHz** untuk mengevaluasi kelayakan deteksi spoofing audio secara real-time pada robot sosial dan panggilan telepon streaming.

### 💾 Kompresi Model
* **PyTorch FP32 Checkpoint:** `2.08 MB`
* **ONNX FP32 Model:** `2.08 MB`
* **ONNX Dynamic INT8 Model:** **`0.56 MB`** (**Kompresi 73.0%** 📉)

### ⏱️ Latensi & Real-Time Factor (RTF)

$$\text{RTF} = \frac{\text{Latency (detik)}}{\text{Durasi Audio (detik)}}$$
*(Model streaming real-time wajib memiliki $\text{RTF} \ll 1.0$)*

| Skenario Audio | Format & Runtime | Threads | Latensi Rata-rata (ms) | p95 Latensi (ms) | RTF | Status Streaming |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4.0s Full Utterance** (399 frames) | PyTorch FP32 | 1 Core | 38.90 ms | 42.90 ms | 0.0097 | Real-Time ✅ |
| | ONNX Runtime FP32 | 1 Core | **22.28 ms** | 26.08 ms | **0.0056** | Ultra-Fast ⚡ |
| | ONNX Dynamic INT8 | 1 Core | 125.57 ms | 142.99 ms | 0.0314 | Real-Time ✅ |
| | ONNX Dynamic INT8 | 4 Cores | **66.12 ms** | 78.30 ms | **0.0165** | Real-Time ✅ |
| **1.28s Streaming Chunk** (128 frames) | PyTorch FP32 | 1 Core | 17.50 ms | 24.18 ms | 0.0137 | Ultra-Fast ⚡ |
| | ONNX Runtime FP32 | 1 Core | **9.62 ms** | 12.34 ms | **0.0075** | **Sub-10ms Streaming 🚀** |
| | ONNX Dynamic INT8 | 1 Core | 37.89 ms | 43.87 ms | 0.0296 | Real-Time ✅ |
| | ONNX Dynamic INT8 | 4 Cores | **21.15 ms** | 23.70 ms | **0.0165** | Real-Time ✅ |

> **Temuan Kunci untuk Robot Sosial:**
> 1. Pada potongan streaming live audio berdurasi 1.28 detik, inferensi CPU hanya membutuhkan **9.62 ms** ($\text{RTF} = 0.0075$). Ini berarti sistem anti-spoofing hanya mengonsumsi kurang dari 1% siklus prosesor CPU, memungkinkan deteksi deepfake secara simultan tanpa mengganggu interaksi percakapan natural atau streaming telepon.
> 2. Model INT8 berhasil menekan ukuran biner menjadi **576 KB** dengan delta skor dari FP32 hanya $\Delta = 0.4459$, sangat ideal untuk microcontroller atau embedded Linux pada robot sosial.
