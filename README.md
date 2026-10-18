# Vocal-Guard: Spoofing-Aware Audio Front-End untuk Robot Sosial

Evaluasi ulang secara sistematis dan *re-engineering* sistem deteksi spoofing suara pada benchmark resmi **ASVspoof 2019 Logical Access (LA)** untuk aplikasi antarmuka suara pada robot sosial.

> **Fokus Metodologi:** Proyek ini mendokumentasikan investigasi kegagalan sistem awal (`41.22%` EER) melalui 8 hipotesis terukur (H1–H8), pemisahan protokol LA dan PA, satu sumber kebenaran (*Single Source of Truth*) fitur LFCC 60-dimensi di Python, serta evaluasi jujur antara performa *in-distribution* (Dev Set), generalisasi serangan tak dikenal (Eval Set), dan *trade-off* inferensi CPU menggunakan PyTorch & ONNX Runtime.

---

## 📊 1. Ringkasan Performa Model (ASVspoof 2019 LA)

*Catatan Penting:* Pada ASVspoof 2019 LA, **Dev Set** memuat jenis serangan yang sama dengan Train Set (A01–A06, *in-distribution*), sedangkan **Eval Set** memuat 13 jenis serangan baru yang belum pernah dilihat saat pelatihan (A07–A19, *out-of-distribution*). Angka **Eval EER** menjadi tolok ukur utama (*headline*) generalisasi sistem.

| Kode | Konfigurasi Model | Fitur & Panjang Input | Parameter | Ukuran File | Dev EER (%) *(In-Dist A01–A06)* | **Eval EER (%) *(Unseen A07–A19)*** | Interpretasi & Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224`, 1s | ~3.50 M | ~14.0 MB | `41.22%` | — | Gagal (mendekati tebakan acak) |
| **B0** | **LFCC-GMM** (256 diag, custom LFCC) | LFCC `(T, 60)`, Full | Statistik | `0.97 MB` | `32.18%` | — | Di atas acak (pipeline data/label tidak rusak, belum setara GMM resmi ~8–9%) |
| **M2** | **MobileNetV2 From-Scratch (Ablation H1)** | LFCC `(1, 60, 399)`, 4s / No Trim | 2,225,858 | `8.50 MB` | **`1.28%`** | — | Melatih MobileNetV2 dari nol pada LFCC 4s menurunkan EER dari `41.22%` ke `1.28%` |
| **M3** | **LCNN + GAP `1x1` (Ablation H2)** | LFCC `(1, 60, 399)`, 4s / No Trim | 314,434 | `1.20 MB` | **`0.47%`** | — | Mengganti backbone ke LCNN+MFM memangkas EER `1.28%` $\to$ `0.47%` (7x lebih hemat param) |
| **M1** | **LCNN + MFM + AdaptivePool `3x4` (FP32)** | LFCC `(1, 60, 399)`, 4s / No Trim | **539,714** | `2.08 MB` | **`0.37%`** | **`4.59%`** 🏆 | **Angka Headline:** Sangat kuat di A07–A16 & A18–A19 ($\le 1.85\%$), terdegradasi pada VC **A17 (`25.73%`)** |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC `(1, 60, 399)`, 4s (2k Dev sub) | **539,714** | **`0.56 MB`** | **`0.59%`** *(vs FP32 `0.59%`)* | — | **Paritas 100% ($\Delta\text{EER} = 0.000\%$)** pada 2,000 sampel Dev; kompresi ukuran 73.0% |
| **M1-CrossTrim**| **LCNN + MFM (Uji Silang H6a)** | Latih Tanpa Trim $\to$ Uji VAD Trim | 539,714 | `2.08 MB` | **`12.11%`** | — | Membuktikan M1 tanpa trim sebagian bergantung pada *silence shortcut* (`0.37%` $\to$ `12.11%`) |
| **M1-Trim** | **LCNN + MFM + VAD Trim (Uji H6b)** | Latih & Uji dengan VAD Trim | 539,714 | `2.08 MB` | **`6.71%`** | — | Akurasi murni dari segmen suara aktif setelah *silence shortcut* dieliminasi |
| **M1-Chunk1.28s**| **LCNN + MFM (Evaluasi Chunk 1.28s)** | LFCC `(1, 60, 128)`, 1.28s / No Trim | 539,714 | `2.08 MB` | **`1.33%`** | — | Akurasi riil yang berpasangan dengan latensi inferensi per chunk 1.28s (`9.62 ms`) |

> **Temuan Generalisasi Eval Set (`4.59%` Pooled EER):** Pada 12 dari 13 jenis serangan tak dikenal (A07–A16, A18–A19), model mencapai EER antara **`0.00%` hingga `1.85%`**. Satu-satunya serangan yang mendegradasi performa secara signifikan adalah **A17 (`25.73%` EER)**, yaitu metode *Voice Conversion* berbasis *direct waveform filtering* yang mempertahankan eksitasi fase asli tanpa melalui *vocoder* konvensional (lihat rincian lengkap di [`results/eval_attack_breakdown.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/eval_attack_breakdown.csv) dan [`RESEARCH_DOCUMENTATION.md`](file:///home/saf/saf/projects/Vocal-Guard/RESEARCH_DOCUMENTATION.md)).

---

## 🧠 2. Diagnosis Kegagalan Awal & Hasil Studi Ablasi (H1–H8)

Perubahan dari `41.22%` ke `0.37%` (Dev EER) pada M1 didekomposisi secara terkontrol di [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) ([`results/ablation_study_summary.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/ablation_study_summary.csv)):

* **H1 (Backbone ImageNet Beku vs Arsitektur LCNN):** Melatih MobileNetV2 dari nol pada fitur LFCC 60-dimensi 4 detik (**M2**) menurunkan Dev EER dari `41.22%` menjadi **`1.28%`**. Mengganti backbone dari MobileNetV2 (**M2**) ke LCNN+MFM dengan pooling yang sama (**M3**) memangkas EER lebih lanjut sebesar **63% relatif (`1.28%` $\to$ `0.47%`)** dengan parameter 7x lebih kecil (`2.23M` $\to$ `0.31M`).
* **H2 (Global Average Pooling `1x1` vs Adaptive Pooling `3x4`):** Membandingkan **M3** (`LCNN + GAP 1x1`, `0.47%` EER) dengan **M1** (`LCNN + AdaptivePool 3x4`, `0.37%` EER) menunjukkan bahwa mempertahankan resolusi spektro-temporal kasar (`3x4`) menurunkan EER sebesar `0.10%` absolut (21% relatif).
* **H3 (Distorsi Resize 224x224) & H4 (Rumus LFCC Tanpa Filterbank):** Diperbaiki lewat implementasi referensi tunggal LFCC 60-dimensi resolusi asli `(1, 60, 399)` di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py).
* **H6 (Window 1 Detik & *Silence Shortcut* — Müller dkk., 2021):** Ketika model **M1** (dilatih tanpa *silence trimming*, `0.37%` Dev EER) diuji pada Dev Set yang telah dipotong heningnya dengan *Energy VAD* (**M1-CrossTrim**), EER melonjak menjadi **`12.11%`**, membuktikan secara empiris bahwa model tanpa trimming sebagian mengeksploitasi korelasi durasi hening. Ketika dilatih dan diuji dengan VAD trimming (**M1-Trim**), model mencapai **`6.71%` Dev EER** murni dari fitur spektral suara aktif.
* **H7 (Pencampuran Trek LA & PA) & H8 (Validitas Protokol Data):** Trek LA dan PA dipisahkan. Baseline **B0 (LFCC-GMM)** menghasilkan `32.18%` Dev EER (di atas tebakan acak 50%, membuktikan pembacaan label tidak terbalik/rusak).

---

## ⚡ 3. Kuantisasi & Benchmark Inferensi Per Chunk di CPU Laptop (Proxy RQ3)

Pengujian dilakukan pada **CPU Laptop Intel Core i7-8665U @ 1.90GHz** sebagai *proxy* komputasi CPU lokal (belum diuji langsung pada prosesor ARM *embedded* seperti Raspberry Pi). Latensi pada skenario `1.28 detik` mengukur waktu *forward-pass* per potongan (*chunk*) statis `128 frame` (yang memiliki akurasi **`M1-Chunk1.28s` Dev EER = `1.33%`**).

$$\text{RTF} = \frac{\text{Waktu Inferensi CPU (detik)}}{\text{Durasi Audio Input (detik)}}$$

| Skenario Input Tensor | Format & Runtime | Thread CPU | Ukuran File | Latensi Rata-rata | p95 Latensi | RTF | Catatan Empiris |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chunk 1.28 detik**<br>*(128 frame LFCC, Dev EER 1.33%)* | **ONNX Runtime FP32** | **1 Core** | `2.08 MB` | **9.62 ms** | **12.34 ms** | **0.0075** | **Tercepat di CPU x86 non-VNNI** |
| | PyTorch Native FP32 | 1 Core | `2.08 MB` | 17.50 ms | 24.18 ms | 0.0137 | Baseline PyTorch CPU |
| | ONNX Dynamic INT8 | 4 Cores | **`0.56 MB`** | 21.15 ms | 23.70 ms | 0.0165 | Kompresi ukuran file **73.0%** ($\Delta\text{EER}=0.00\%$) |
| | ONNX Dynamic INT8 | 1 Core | **`0.56 MB`** | 37.89 ms | 43.87 ms | 0.0296 | Overhead quant/dequant dinamis per layer |
| **Utterance 4.00 detik**<br>*(399 frame LFCC, Eval EER 4.59%)* | **ONNX Runtime FP32** | **1 Core** | `2.08 MB` | **22.28 ms** | **26.08 ms** | **0.0056** | **Tercepat untuk evaluasi 4 detik** |
| | PyTorch Native FP32 | 1 Core | `2.08 MB` | 38.90 ms | 42.90 ms | 0.0097 | Baseline PyTorch CPU |
| | ONNX Dynamic INT8 | 4 Cores | **`0.56 MB`** | 66.12 ms | 78.30 ms | 0.0165 | Kompresi ukuran file **73.0%** ($\Delta\text{EER}=0.00\%$) |
| | ONNX Dynamic INT8 | 1 Core | **`0.56 MB`** | 125.57 ms | 142.99 ms | 0.0314 | Overhead quant/dequant dinamis per layer |

---

## 🖥️ 4. Menjalankan Vocal-Guard Streaming Demo (Lokal CPU)

Untuk menjalankan aplikasi web demonstrasi pemrosesan per-chunk 1.28 detik (`FastAPI` + `WebSocket` + `ONNX Runtime`):

```bash
./scripts/run_demo.sh
```
Lalu buka `http://127.0.0.1:8000` di peramban untuk menguji sampel referensi ASVspoof (*bonafide* vs *spoof*), membandingkan runtime **ONNX FP32 vs INT8**, serta mengamati grafik skor logit dan latensi/RTF per chunk secara langsung.

---

## 🛠️ 5. Struktur Repositori & Dokumentasi Lengkap

Lihat **[`RESEARCH_DOCUMENTATION.md`](file:///home/saf/saf/projects/Vocal-Guard/RESEARCH_DOCUMENTATION.md)** untuk rincian metodologi, tabel konvergensi epoch, dan analisis lengkap.

```text
vocal_guard/
├── features.py          # Single source of truth ekstraksi LFCC 60-dim (termasuk opsi VAD trim)
├── protocol.py          # Parser resmi protokol ASVspoof 2019 LA
├── metrics.py           # Perhitungan metrik resmi EER dan threshold
├── dataset.py           # PyTorch Dataset (4s/1.28s wrap/truncate & VAD silence trim)
├── baseline_gmm.py      # Baseline LFCC-GMM
└── models/
    └── lcnn.py          # Light CNN (LCNN+MFM), LCNN+GAP (M3), dan MobileNetV2 (M2)
demo/
├── app.py               # FastAPI + WebSocket backend untuk inferensi per-chunk 1.28s
├── samples/             # Sampel WAV referensi ASVspoof (bonafide & spoof)
└── static/index.html    # Antarmuka web satu halaman (grafik skor & latensi real-time)
notebooks/
├── 01_baseline_b0_gmm.ipynb         # Eksperimen Baseline B0 (Selesai)
├── 02_train_lcnn_m1.ipynb           # Training LCNN M1 dari nol (Selesai)
├── 03_eval_asvspoof2019_la.ipynb    # Evaluasi Eval Set 71k & Breakdown A07–A19 (Selesai)
└── 04_ablation_study_m2_m3.ipynb    # Ablation M3, M2, Silence Trim H6, Paritas INT8, & Chunk 1.28s (Selesai)
scripts/
├── quantize_and_benchmark.py        # Ekspor ONNX, kuantisasi INT8, dan benchmark CPU
└── run_demo.sh                      # Skrip peluncur aplikasi web demo
results/                             # CSV log hasil eksperimen, breakdown Eval, dan studi ablasi
models/                              # Bobot PyTorch (.pt), GMM (.pkl), dan ONNX (.onnx)
```

