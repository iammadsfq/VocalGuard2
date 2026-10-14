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
