# 🛡️ Vocal-Guard: Spoofing-Aware Audio Front-End untuk Robot Sosial

Evaluasi ulang secara sistematis, studi ablasi terkontrol, optimasi inferensi CPU (**ONNX FP32 & Dynamic INT8**), serta sistem deteksi *streaming real-time* hibrida pada benchmark **ASVspoof 2019 Logical Access (LA)** untuk keamanan antarmuka suara pada robot sosial.

> **📄 Dokumentasi Riset Lengkap:** Lihat **[`RESEARCH_DOCUMENTATION.md`](file:///home/saf/saf/projects/Vocal-Guard/RESEARCH_DOCUMENTATION.md)** untuk penjelasan mendalam mengenai diagnosis 8 hipotesis (H1–H8), arsitektur sistem, rincian *breakdown* 71.237 sampel Eval Set (`A07–A19`), analisis *silence shortcut*, dan hasil telemetri *live streaming*.

---

## 🎯 1. Ringkasan Eksekutif

Proyek ini berawal dari investigasi atas kegagalan sistem deteksi *spoofing* generasi pertama (**`41.22%` EER**, hampir setara tebakan acak `50%`) yang menggunakan *backbone* visual **MobileNetV2 beku (*frozen ImageNet*)** pada gambar spektrogram LFCC berukuran `224x224`. Melalui **8 hipotesis terukur (H1–H8)** dan eksperimen ablasi terkontrol, kami membangun ulang *pipeline* secara menyeluruh hingga mencapai:

1. **Generalisasi Eval Set Resmi (`4.59%` Pooled EER pada 71.237 sampel A07–A19):** Sangat tangguh pada 12 dari 13 jenis serangan tak dikenal (**`0.00%` – `1.85%` EER** pada `A07–A16` dan `A18–A19`), dengan identifikasi jujur terhadap satu titik lemah pada metode *direct waveform-filtering Voice Conversion* **A17 (`25.73%` EER)**.
2. **Dekomposisi Ablasi Terkontrol (`41.22%` $\to$ `1.28%` $\to$ `0.47%` $\to$ `0.37%` Dev EER):** Membuktikan secara terpisah dampak pembukaan bobot + fitur LFCC 60-dimensi (**M2: `1.28%`**), penggantian arsitektur ke **LCNN + Max-Feature-Map (MFM)** (**M3: `0.47%`**, hemat 86% parameter), dan *Adaptive Pooling* `3x4` (**M1: `0.37%`**), serta mengonfirmasi secara empiris fenomena *silence shortcut* (Müller dkk., 2021) lewat uji silang **M1-CrossTrim (`12.11%`)** dan **M1-Trim (`6.71%`)**.
3. **Inferensi CPU Super Cepat & Kuantisasi INT8 Tanpa Kehilangan Akurasi:** Ekspor ke **ONNX FP32 (`2.08 MB`)** mencapai latensi inferensi **`4.8–9.6 ms`** per *chunk* `1.28 detik` ($\text{RTF} \approx 0.0062$, **~160x lebih cepat dari waktu nyata** di 1 *core* CPU Intel i7-8665U), sementara **ONNX Dynamic INT8 (`0.56 MB`)** memangkas ukuran model sebesar **73.0%** dengan paritas EER sempurna (**$\Delta\text{EER} = 0.000\%$**).
4. **Deteksi Live Streaming Hibrida (LA + PA Speaker Replay / Neural TTS):** Dilengkapi antarmuka web *real-time* (`FastAPI` + `WebSocket`) yang menggabungkan kompaksi *frame* vokal tanpa *dither* buatan dengan detektor akustik resonansi speaker & kemulusan *pitch* untuk membedakan **suara manusia asli (`🟢 BONAFIDE`)** dari pemutaran **Gemini TTS / Google Maps Voiceover (`🔴 FAKE / SPOOF`)** melalui speaker HP/laptop.

---

## 📊 2. Tabel Hasil Eksperimen & Studi Ablasi (ASVspoof 2019 LA)

Sumber data: [`results/ablation_study_summary.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/ablation_study_summary.csv) & [`results/asvspoof2019_final_eval_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/asvspoof2019_final_eval_results.csv)

| Kode | Konfigurasi Model | Fitur & Panjang Input | Parameter | Ukuran File | Dev EER (%) *(In-Dist A01–A06)* | **Eval EER (%) *(Unseen A07–A19)*** | Interpretasi & Temuan Kunci |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224`, 1s | ~3.50 M | ~14.0 MB | `41.22%` | — | Gagal (mendekati tebakan acak 50%) |
| **B0** | **LFCC-GMM** (256 diag, custom LFCC) | LFCC `(T, 60)`, Full | Statistik | `0.97 MB` | `32.18%` | — | Di atas acak; membuktikan pembacaan protokol/label valid |
| **M2** | **MobileNetV2 From-Scratch (Ablation H1)** | LFCC `(1, 60, 399)`, 4s / No Trim | 2.225.858 | `8.50 MB` | **`1.28%`** | — | Melatih dari nol pada LFCC 4s menurunkan EER `41.22%` $\to$ `1.28%` |
| **M3** | **LCNN + GAP `1x1` (Ablation H2)** | LFCC `(1, 60, 399)`, 4s / No Trim | 314.434 | `1.20 MB` | **`0.47%`** | — | Mengganti backbone ke LCNN+MFM memangkas EER `1.28%` $\to$ `0.47%` (7x lebih hemat param) |
| **M1** | **LCNN + MFM + AdaptivePool `3x4` (FP32)** | LFCC `(1, 60, 399)`, 4s / No Trim | **539.714** | `2.08 MB` | **`0.37%`** | **`4.59%`** 🏆 | **Angka Headline:** Sangat kuat di A07–A16 & A18–A19 ($\le 1.85\%$), lemah di VC **A17 (`25.73%`)** |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC `(1, 60, 399)`, 4s (2k Dev sub) | **539.714** | **`0.56 MB`** | **`0.59%`** *(vs FP32 `0.59%`)* | — | **Paritas 100% ($\Delta\text{EER} = 0.000\%$)** pada 2.000 sampel Dev; kompresi ukuran 73.0% |
| **M1-CrossTrim**| **LCNN + MFM (Uji Silang H6a)** | Latih Tanpa Trim $\to$ Uji VAD Trim | 539.714 | `2.08 MB` | **`12.11%`** | — | Membuktikan M1 tanpa trim sebagian bergantung pada *silence shortcut* (`0.37%` $\to$ `12.11%`) |
| **M1-Trim** | **LCNN + MFM + VAD Trim (Uji H6b)** | Latih & Uji dengan VAD Trim | 539.714 | `2.08 MB` | **`6.71%`** | — | Akurasi murni dari segmen suara aktif setelah *silence shortcut* dieliminasi |
| **M1-Chunk1.28s**| **LCNN + MFM (Evaluasi Chunk 1.28s)** | LFCC `(1, 60, 128)`, 1.28s / No Trim | 539.714 | `2.08 MB` | **`1.33%`** | — | Akurasi riil yang berpasangan dengan latensi inferensi per chunk 1.28s (`9.62 ms`) |

---

## ⚡ 3. Benchmark Latensi CPU & Hasil Pengujian Live Streaming Mikrofon

### 3.1 Benchmark Offline per Chunk (`1.28s`) & per Utterance (`4.00s`)
Perangkat Uji: **CPU Laptop Intel Core i7-8665U @ 1.90GHz** (digunakan sebagai *proxy* komputasi CPU lokal tanpa GPU; lihat [`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv)):

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

### 3.2 Hasil Pengujian Live Streaming Mikrofon (Suara Asli vs Suara Gemini TTS)
Pada mode *live streaming* (`Update tiap 0.50 detik`, jendela `1.28 detik`), sistem menggabungkan skor logit **LCNN (`LA`)** dengan modul **Kalibrasi Akustik Hibrida (`PA/TTS`)** untuk membedakan suara manusia langsung dari suara AI (**Gemini TTS / Google Maps Voiceover**) yang diputar lewat speaker:

| Skenario Uji Live Mikrofon | Model Utama | Total Chunk | Akurasi Stabil *(Setelah Buffer Penuh)* | Rentang Skor Akhir ($s$) | Latensi ONNX | Total Latensi | RTF | Selisih `|FP32 - INT8|` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Suara Asli Manusia (Live Mic)** | **ONNX INT8** | 32 | **96.6%** *(28/29 chunk #4–#32 BONAFIDE)* | `+0.73` s.d. `+11.28` | `30.18 ms` | `33.95 ms` | `0.0265` | `0.114` |
| **Suara Asli Manusia (Live Mic)** | **ONNX FP32** | 42 | **93.3%** *(28/30 chunk #13–#42 BONAFIDE)* | `+0.17` s.d. `+12.11` | **`4.84 ms`** | **`7.98 ms`** | **`0.0062`** | `0.158` |
| **Suara Gemini TTS (Speaker)** | **ONNX FP32** | 32 | **90.6% Total / 100%** *(#13–#32 FAKE)* | `-10.92` s.d. **`-53.39`** | **`5.12 ms`** | **`8.35 ms`** | **`0.0065`** | `0.142` |
| **Suara Gemini TTS (Speaker)** | **ONNX INT8** | 57 | **100.0%** *(34/34 chunk #24–#57 FAKE)* | `-8.13` s.d. **`-53.03`** | `24.89 ms` | `28.14 ms` | `0.0220` | `0.156` |

---

## 🖥️ 4. Cara Menjalankan Aplikasi Web Streaming Demo

Jalankan perintah berikut dari direktori utama proyek:

```bash
./scripts/run_demo.sh
```

Kemudian buka **`http://127.0.0.1:8000`** di browser:
1. **🎙️ Mode Rekam Mikrofon Streaming (Live):** Klik **"🔴 Mulai Rekam & Deteksi Streaming Live"** untuk menguji suara asli Anda secara langsung (`🟢 BONAFIDE`) atau memutar suara **Gemini TTS / Google Maps Voiceover** dari HP/speaker (`🔴 FAKE (Replay/TTS)`).
2. **📁 Mode Sampel Referensi & Upload File WAV:** Uji sampel referensi bawaan (`reference_harmonic_5s.wav` vs `reference_synthetic_artifact_5s.wav`) atau unggah file `.wav`/`.flac` untuk dianalisis chunk demi chunk.
3. **🔬 Perbandingan Dual-Model Real-Time:** Aktifkan opsi **"Bandingkan FP32 vs INT8 Live"** untuk melihat perbandingan latensi dan selisih skor logit antara `ONNX FP32 (2.08 MB)` dan `ONNX INT8 (0.56 MB)` pada setiap chunk secara berdampingan, lalu ekspor log sesi ke format `.csv`.

---

## 🗂️ 5. Struktur Direktori Proyek

```text
Vocal-Guard/
├── vocal_guard/                         # Paket Inti Python (Single Source of Truth)
│   ├── features.py                      # Ekstraksi fitur LFCC 60-dimensi (20 statis + 20 Δ + 20 ΔΔ) & VAD trim
│   ├── protocol.py                      # Parser protokol resmi ASVspoof 2019 LA
│   ├── metrics.py                       # Perhitungan resmi Equal Error Rate (EER) & ambang keputusan
│   ├── dataset.py                       # PyTorch Dataset (padding/truncation 4.0s & 1.28s)
│   ├── baseline_gmm.py                  # Implementasi Baseline Klasik LFCC-GMM (B0)
│   └── models/
│       └── lcnn.py                      # Arsitektur LCNN+MFM (M1), LCNN+GAP (M3), dan MobileNetV2 (M2)
├── demo/                                # Aplikasi Web Real-Time Streaming Demo
│   ├── app.py                           # Backend FastAPI + WebSocket + Detektor Hibrida (LA + PA/TTS)
│   ├── samples/                         # Sampel WAV referensi 16 kHz (Bonafide & Spoof)
│   └── static/index.html                # Dashboard UI interaktif (Osiloskop, Gauge, Grafik, & Tabel Log)
├── notebooks/                           # Notebook Eksperimen & Evaluasi (Kaggle / Lokal)
│   ├── 01_baseline_b0_gmm.ipynb         # Eksperimen Baseline B0 LFCC-GMM
│   ├── 02_train_lcnn_m1.ipynb           # Pelatihan Model Utama LCNN+MFM (M1)
│   ├── 03_eval_asvspoof2019_la.ipynb    # Evaluasi Eval Set 71.237 sampel & Breakdown A07–A19
│   └── 04_ablation_study_m2_m3.ipynb    # Studi Ablasi (M2, M3, H6 Silence Trim, Chunk 1.28s, Paritas INT8)
├── scripts/
│   ├── quantize_and_benchmark.py        # Ekspor ONNX FP32/INT8 & pengukuran latensi/RTF CPU
│   └── run_demo.sh                      # Skrip peluncur server demo FastAPI (port 8000)
├── models/                              # File bobot terlatih (.pt, .pkl, .onnx FP32 & INT8)
├── results/                             # File CSV hasil eksperimen, breakdown Eval, ablasi, & benchmark CPU
├── tests/                               # Unit test otomatis (pytest)
├── RESEARCH_DOCUMENTATION.md            # Dokumentasi riset & engineering lengkap
└── README.md                            # Ringkasan utama repositori
```
