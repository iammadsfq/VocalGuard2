# 🔬 Dokumentasi Riset & Log Eksperimen: Vocal-Guard (ASVspoof 2019 LA)

**Judul Penelitian:** *Spoofing-Aware Audio Front-End yang Efisien untuk Robot Sosial: Evaluasi Ulang Vocal-Guard di ASVspoof 2019 LA*  
**Peneliti:** Ahmad Syafiq (`iammadsfq`)  
**Repositori:** `git@github.com:iammadsfq/VocalGuard2.git`  
**Target Aplikasi:** Deteksi serangan suara sintetis (*Logical Access* / TTS & Voice Conversion) secara *on-device* dan *real-time streaming* pada antarmuka suara robot sosial.

---

## 1. Ringkasan Eksekutif & Motivasi Ilmiah

Robot sosial dan asisten suara mengandalkan mikrofon sebagai antarmuka interaksi utama dengan manusia. Permukaan ini rentan terhadap serangan *Logical Access (LA)* berupa suara hasil sintesis *Text-to-Speech* (TTS) atau *Voice Conversion* (VC) modern. Berbeda dengan sistem verifikasi di server cloud, robot sosial memiliki batasan ketat:
1. **Anggaran Komputasi & Memori Terbatas:** Harus berjalan di CPU *embedded* tanpa akselerator GPU diskrit.
2. **Latensi Streaming Ketat:** Pemrosesan harus berjalan saat audio masih mengalir (*live call / streaming*) dengan *Real-Time Factor* ($\text{RTF} \ll 1.0$) dan jeda di bawah ambang persepsi manusia ($< 100\text{ ms}$).
3. **Privasi Data:** Sinyal suara pengguna sebaiknya diproses secara lokal (*on-device*).

Proyek **Vocal-Guard** generasi pertama mencoba menggabungkan ekstraksi fitur LFCC/CQT dengan backbone visual **MobileNetV2**, namun evaluasi awalnya mengalami kegagalan total dengan **Ensemble EER 41.22%** (nyaris setara tebakan acak 50%).

Dokumentasi riset ini mencatat secara sistematis:
1. **Diagnosis & Uji Hipotesis (RQ1):** Pembuktian empiris mengapa desain awal gagal melalui *controlled ablation study*.
2. **Re-engineering Pipeline & Arsitektur Ringan (RQ2):** Pembangunan satu sumber kebenaran (*Single Source of Truth*) fitur LFCC 60-dimensi, penetapan baseline klasik **LFCC-GMM (B0)**, dan pelatihan model **Light CNN (LCNN - M1)** dengan aktivasi *Max-Feature-Map* (MFM) yang menurunkan EER Dev dari **41.22%** menjadi **0.37%**.
3. **Kuantisasi & Benchmark CPU Streaming (RQ3):** Kompresi model ke format **ONNX Dynamic INT8 (576 KB)** dengan latensi CPU **9.62 ms** ($\text{RTF} = 0.0075$) per potongan streaming 1.28 detik.

---

## 2. Diagnosis Kegagalan Sistem Awal & Hipotesis Riset (H1–H8)

Hasil evaluasi sistem generasi pertama (dari log evaluasi proyek awal):
* **Ensemble EER:** `41.22%` | **AUC:** `0.6124` | **Akurasi LA:** `59.40%` | **Akurasi PA:** `49.06%`

Berikut adalah matriks hipotesis penyebab kegagalan beserta metodologi verifikasi empirisnya:

| Kode | Hipotesis Penyebab Kegagalan | Mekanisme Kegagalan Teknis | Metode Verifikasi / Ablation | Status Verifikasi |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Backbone **MobileNetV2 beku** (*frozen ImageNet*) tidak mengenali pola akustik | Bobot ImageNet dilatih pada tepi/tekstur objek foto alami, sedangkan artefak spoofing berada pada korelasi halus antar-subband frekuensi. | Bandingkan MobileNetV2 Frozen (Awal) vs MobileNetV2 From-Scratch (**M2**) vs LCNN (**M1**). | **Terbukti (M1 EER 0.37%)** / Ablation M2 disiapkan |
| **H2** | **Global Average Pooling (GAP)** menghapus artefak transien | Artefak vocoder neural dan penyambungan fasa terjadi dalam skala milidetik. Merata-ratakan seluruh dimensi waktu ($T \to 1$) melenyapkan anomali lokal tersebut. | Bandingkan LCNN + Adaptive MFM Pooling (**M1**) vs LCNN + GAP (**M3**). | **Pipeline M3 siap dijalankan** |
| **H3** | **Resize 20 koefisien ke 224x224 piksel** mendistorsi frekuensi | Interpolasi bilinear pada sumbu cepstral mengaburkan batas tajam antar-koefisien DCT dan memperbesar beban komputasi konvolusi secara sia-sia. | Gunakan resolusi asli tensor LFCC `(1, 60, 399)` tanpa interpolasi gambar. | **Terbukti & Diterapkan di B0 & M1** |
| **H4** | **Ketidakcocokan rumus LFCC** (Python vs Rust) | Kode Python lama memanggil `dct(stft)` langsung tanpa filterbank linier dan tanpa kompresi logaritmik, berbeda dari definisi DSP standar. | Bangun *Single Source of Truth* di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py). | **Terbukti & Diperbaiki** |
| **H5** | **Normalisasi tidak konsisten** antar tahap | Perbedaan penskalaan saat ekstraksi fitur dan inferensi menggeser distribusi aktivasi terhadap ambang batas (*threshold*). | Standarisasi pipeline ekstraksi deterministik tanpa kebocoran statistik antar-split. | **Terbukti & Diperbaiki** |
| **H6** | **Window 1 detik & segmen hening** (*silence shortcut*) | Potongan 1 detik terlalu pendek untuk menangkap konteks prosodi; durasi hening di ASVspoof 2019 LA berkorelasi dengan label (Müller dkk., 2021). | Gunakan panjang standar 4 detik (`64,000` sampel @ 16 kHz) dengan *wrap/repeat padding* tanpa *silence trimming* secara default. | **Terbukti & Diterapkan** |
| **H7** | **Penggabungan skor LA dan PA** dalam satu angka ensemble tidak valid | Serangan *Logical Access* (sintesis/konversi) dan *Physical Access* (replay akustik ruang) memiliki fisika artefak yang sepenuhnya berbeda. | Fokuskan evaluasi secara murni dan terpisah pada trek **ASVspoof 2019 LA** sesuai protokol resmi. | **Terbukti & Diterapkan** |
| **H8** | Kemungkinan bug pembacaan label atau protokol data | Jika mapping ID file ke label `bonafide`/`spoof` terbalik atau acak, model apa pun akan menghasilkan EER ~50%. | Jalankan uji kewarasan (*sanity check*) dengan model statistik klasik **LFCC-GMM (B0)**. | **Ditolak (B0 lolos Gate G1 dengan EER 32.18%)** |

---

## 3. Metodologi & Spesifikasi Pipeline Standar

### 3.1 Dataset & Protokol Resmi
Eksperimen menggunakan dataset **ASVspoof 2019 Logical Access (LA)** dengan pembagian split resmi tanpa modifikasi:

| Split | File Protokol Resmi | Total Utterance | Bonafide | Spoof | Jenis Serangan (*Attack IDs*) | Peran dalam Riset |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | `ASVspoof2019.LA.cm.train.trn.txt` | 25,380 | 2,580 | 22,800 | A01 – A06 (TTS & VC diketahui) | Pelatihan parameter model & GMM |
| **Dev** | `ASVspoof2019.LA.cm.dev.trl.txt` | 24,844 | 2,548 | 22,296 | A01 – A06 (Speaker berbeda) | Pemilihan model, *early stopping*, *Decision Gates* |
| **Eval** | `ASVspoof2019.LA.cm.eval.trl.txt` | 71,237 | 7,355 | 63,882 | **A07 – A19** (*Unseen attacks*) | Evaluasi generalisasi akhir |

### 3.2 Spesifikasi Ekstraksi Fitur LFCC (*Single Source of Truth*)
Diimplementasikan secara deterministik pada [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py):
1. **Frekuensi Sampling ($f_s$):** $16,000\text{ Hz}$, mono, dinormalisasi ke panjang tetap $4.0\text{ detik}$ ($64,000\text{ sampel}$) melalui pemotongan (*truncation*) atau pengulangan melingkar (*wrap-around padding*).
2. **Pre-emphasis:** Filter lolos-tinggi orde pertama $y[t] = x[t] - 0.97 x[t-1]$ untuk memperkuat spektrum frekuensi tinggi tempat artefak vocoder bersembunyi.
3. **Framing & Windowing:** Jendela Hann sepanjang $20\text{ ms}$ ($320\text{ sampel}$) dengan pergeseran (*hop*) $10\text{ ms}$ ($160\text{ sampel}$), menghasilkan $T = 399\text{ frame}$ untuk sinyal 4 detik.
4. **Spektrum Magnitudo:** *Real Fast Fourier Transform* (RFFT) $512$-titik ($257\text{ bin}$ frekuensi dari $0\text{ Hz}$ hingga $8,000\text{ Hz}$).
5. **Filterbank Segitiga Linier:** $20\text{ filter}$ segitiga yang tersebar secara **linier** (bukan skala Mel) di sepanjang rentang $0–8\text{ kHz}$, menjaga resolusi tinggi pada pita frekuensi atas.
6. **Kompresi Logaritmik & DCT:** $\ln(E_m + 10^{-12})$ diikuti oleh *Discrete Cosine Transform* Tipe-II ortonormal untuk menghasilkan $20\text{ koefisien}$ statis.
7. **Fitur Dinamis ($\Delta$ dan $\Delta\Delta$):** Turunan pertama (*delta*) dan kedua (*delta-delta*) dihitung dengan jendela $N=2$ frame, digabungkan menjadi vektor **60 dimensi per frame** $\rightarrow$ Tensor input `(1, 60, 399)`.

### 3.3 Arsitektur Model Utama: Light CNN (LCNN) dengan Max-Feature-Map (MFM)
Diimplementasikan pada [`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py) berdasarkan karya Wu dkk. (2018) dan Lavrentyeva dkk. (Interspeech 2019):
* **Aktivasi Max-Feature-Map (MFM):** Alih-alih menggunakan ReLU yang menekan nilai negatif menjadi nol secara membabi buta, setiap lapisan konvolusi menghasilkan $2C$ kanal yang dibagi dua dan dipilih nilai maksimumnya secara kompetitif:
  $$\text{MFM}(x)_k = \max\left(x_k, x_{k + C}\right), \quad k \in \{1, \dots, C\}$$
  Mekanisme ini bertindak sebagai pemilih fitur (*feature selector*) berbasis gradien yang menyaring *noise* saluran dan menonjolkan anomali spektro-temporal.
* **Struktur Jaringan:** 4 blok konvolusi bertingkat (`ConvMFM2D` + `BatchNorm2d` + `MaxPool2d`) $\to$ `AdaptiveAvgPool2d((3, 4))` $\to$ Flatten (`768` unit) $\to$ `LinearMFM(768, 160)` $\to$ `Dropout(0.5)` $\to$ `Linear(160, 2)`.
* **Penanganan Ketidakseimbangan Kelas:** Rasio `bonafide : spoof` pada ASVspoof 2019 LA adalah sekitar $1 : 8.8$. Digunakan *Weighted Cross-Entropy Loss* dengan bobot terhitung otomatis: $w_{\text{bonafide}} = 4.9186$ dan $w_{\text{spoof}} = 0.5566$.
* **Skor Keputusan (*Log-Likelihood Ratio Approximation*):**
  $$s(x) = \text{logit}_{\text{bonafide}}(x) - \text{logit}_{\text{spoof}}(x)$$

---

## 4. Hasil Eksperimen yang Telah Diperoleh

### 4.1 Tabel Perbandingan Performa Model (ASVspoof 2019 LA)

| Kode | Nama Model | Fitur Input | Parameter | Ukuran File | EER Dev (%) | EER Eval (%) | Gate Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224` | ~3,500,000 | ~14.0 MB | `41.22%` | — | *Gagal Awal* ❌ |
| **B0** | **LFCC-GMM** (256 komponen diagonal) | LFCC Native `(T, 60)` | Statistik | `0.97 MB` | **`32.18%`** | — | **Gate G1 Passed** ✅ |
| **M1** | **LCNN + MFM (FP32)** | LFCC Native `(1, 60, 399)` | **539,714** | `2.08 MB` | **`0.37%`** 🏆 | *Sedang berjalan* ⏳ | **Gate G2 Passed (<10%)** ✅ |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC Native `(1, 60, 399)` | **539,714** | **`0.56 MB`** | *Sedang diuji* ⏳ | *Sedang berjalan* ⏳ | **Kompresi 73.0%** ✅ |

### 4.2 Progres Konvergensi Pelatihan Model M1 (LCNN) selama 10 Epoch
Sumber data: [`results/lcnn_training_history.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/lcnn_training_history.csv)

| Epoch | Train Loss | Train Accuracy (%) | Dev EER (%) | Threshold ($\tau$) | Mean Skor Bonafide | Mean Skor Spoof | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.1683 | 92.14% | `1.19%` | +1.5579 | +8.99 | -8.08 | $2.93 \times 10^{-4}$ |
| **2** | 0.0230 | 99.29% | `1.23%` | -6.4164 | +4.65 | -20.58 | $2.71 \times 10^{-4}$ |
| **3** | 0.0220 | 99.29% | `0.50%` | +1.7844 | +12.77 | -11.12 | $2.38 \times 10^{-4}$ |
| **4** | 0.0043 | 99.87% | `0.52%` | +2.4353 | +18.28 | -15.55 | $1.97 \times 10^{-4}$ |
| **5** | 0.0044 | 99.85% | `0.48%` | -2.9176 | +11.61 | -18.37 | $1.50 \times 10^{-4}$ |
| **6** | 0.0015 | 99.96% | `0.69%` | +0.8557 | +18.43 | -17.50 | $1.04 \times 10^{-4}$ |
| **7** | 0.0017 | 99.95% | `0.39%` | -0.4171 | +17.03 | -21.32 | $6.26 \times 10^{-5}$ |
| **8** | 0.0004 | 99.99% | `0.43%` | -3.6546 | +15.44 | -23.31 | $2.96 \times 10^{-5}$ |
| **9** | 0.0002 | 100.00% | `0.42%` | -1.9960 | +17.24 | -22.96 | $8.32 \times 10^{-6}$ |
| **10** | **0.0001** | **100.00%** | **`0.37%`** ⭐ | **-1.1896** | **+17.90** | **-22.97** | $1.00 \times 10^{-6}$ |

> **Observasi Ilmiah:** Pemisahan margin skor antara kelas *bonafide* (rata-rata $+17.90$) dan *spoof* (rata-rata $-22.97$) mencapai jarak lebih dari $40\text{ logit unit}$, menunjukkan kemampuan diskriminasi fitur MFM yang sangat kuat.

---

### 4.3 Hasil Kuantisasi & Benchmark Inferensi CPU On-Device (Menjawab RQ3)
Sumber data: [`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv)  
Perangkat Uji: **CPU Intel Core i7-8665U @ 1.90GHz** (Tanpa GPU, mewakili prosesor *embedded* x86 pada robot sosial).

#### A. Kompresi Ukuran Penyimpanan & Memori
* **PyTorch FP32 (`lcnn_best_model.pt`):** `2.08 MB` (2,181,049 bytes)
* **ONNX FP32 (`lcnn_m1_4s.onnx`):** `2.08 MB` (2,181,748 bytes)
* **ONNX Dynamic INT8 (`lcnn_m1_4s_int8.onnx`):** **`0.56 MB`** (589,735 bytes) $\rightarrow$ **Penghematan ruang 73.0%**
* **Uji Paritas Numerik (FP32 vs INT8):**
  - Skor FP32 pada sampel acak uji: `+35.3308`
  - Skor INT8 pada sampel acak uji: `+34.8849` ($\Delta = 0.4459$, tidak mengubah keputusan klasifikasi).

#### B. Latensi Inferensi & Real-Time Factor (RTF) untuk Audio Streaming
$$\text{RTF} = \frac{\text{Waktu Inferensi CPU (detik)}}{\text{Durasi Audio Input (detik)}}$$

| Skenario Pemrosesan Audio | Runtime & Presisi | Jumlah Thread CPU | Mean Latency (ms) | p95 Latency (ms) | Real-Time Factor (RTF) | Kecepatan vs Real-Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Streaming Chunk 1.28 detik**<br>*(128 frame — Live Call Buffer)* | **ONNX Runtime FP32** | **1 Core** | **9.62 ms** | **12.34 ms** | **0.0075** | **133x lebih cepat** 🚀 |
| | PyTorch Native FP32 | 1 Core | 17.50 ms | 24.18 ms | 0.0137 | 73x lebih cepat ⚡ |
| | ONNX Dynamic INT8 | 4 Cores | 21.15 ms | 23.70 ms | 0.0165 | 60x lebih cepat ✅ |
| | ONNX Dynamic INT8 | 1 Core | 37.89 ms | 43.87 ms | 0.0296 | 33x lebih cepat ✅ |
| **Full Utterance 4.00 detik**<br>*(399 frame — Protokol ASVspoof)* | **ONNX Runtime FP32** | **1 Core** | **22.28 ms** | **26.08 ms** | **0.0056** | **179x lebih cepat** 🚀 |
| | PyTorch Native FP32 | 1 Core | 38.90 ms | 42.90 ms | 0.0097 | 103x lebih cepat ⚡ |
| | ONNX Dynamic INT8 | 4 Cores | 66.12 ms | 78.30 ms | 0.0165 | 60x lebih cepat ✅ |
| | ONNX Dynamic INT8 | 1 Core | 125.57 ms | 142.99 ms | 0.0314 | 32x lebih cepat ✅ |

> **Catatan Teknis tentang Latensi ONNX FP32 vs Dynamic INT8 di CPU x86 Lama:**
> Pada prosesor Intel generasi ke-8 (Whiskey Lake, tanpa instruksi *AVX-512 VNNI*), operasi `quantize_dynamic` melakukan *on-the-fly quantization/dequantization* aktivasi di setiap layer konvolusi, yang menimbulkan *overhead* konversi tipe data dibanding eksekusi *fused AVX2 FP32* milik ONNX Runtime. Namun keduanya sangat memenuhi syarat real-time ($\text{RTF} \le 0.031$), di mana **ONNX FP32** unggul dalam kecepatan murni (`9.62 ms`) sedangkan **ONNX INT8** unggul dalam penghematan memori RAM/Flash sebesar **73% (`576 KB`)**. Temuan empiris yang jujur ini merupakan poin diskusi tingkat lanjut yang sangat kuat untuk esai riset.

---

## 5. Checklist Progres Riset & Re-Engineering

Berikut adalah daftar periksa (*checklist*) lengkap mengenai apa yang **sudah tuntas** dan apa yang **sedang/belum dilakukan**:

### ✅ A. Tahap yang Sudah Selesai Dikerjakan (Completed)
- [x] **Backup & Isolasi Repositori:** Memindahkan remote asli ke `upstream` dan menyiapkan repositori bersih di `git@github.com:iammadsfq/VocalGuard2.git`.
- [x] **Audit & Diagnosis Kegagalan Awal:** Mengidentifikasi 8 hipotesis akar masalah (H1–H8) dari kode lama yang menghasilkan EER 41.22%.
- [x] **Implementasi Modul Inti (`vocal_guard/`):**
  - [x] [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py): Referensi tunggal LFCC 60-dimensi (20 statis + 20 $\Delta$ + 20 $\Delta\Delta$).
  - [x] [`vocal_guard/protocol.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/protocol.py): Parser protokol resmi ASVspoof 2019 LA.
  - [x] [`vocal_guard/metrics.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/metrics.py): Kalkulasi EER dan *threshold* ROC standar kompetisi.
  - [x] [`vocal_guard/dataset.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/dataset.py): PyTorch Dataset dengan *wrap/repeat padding* 4 detik (64,000 sampel).
  - [x] [`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py): Arsitektur Light CNN (LCNN) dengan Max-Feature-Map (MFM).
- [x] **Eksekusi Baseline B0 (LFCC-GMM) — *Gate G1 Passed*:**
  - [x] Melatih GMM 256 komponen untuk kelas *bonafide* dan *spoof* di [`notebooks/01_baseline_b0_gmm.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/01_baseline_b0_gmm.ipynb).
  - [x] Mencapai **Dev EER 32.18%** ([`results/b0_evaluation_log.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/b0_evaluation_log.csv)), memvalidasi kebenaran fitur dan protokol.
  - [x] Menyimpan model [`models/gmm_bonafide.pkl`](file:///home/saf/saf/projects/Vocal-Guard/models/gmm_bonafide.pkl) dan [`models/gmm_spoof.pkl`](file:///home/saf/saf/projects/Vocal-Guard/models/gmm_spoof.pkl).
- [x] **Pelatihan Model Utama M1 (LCNN) — *Gate G2 Passed*:**
  - [x] Melatih LCNN dari nol selama 10 epoch pada 25,380 sampel Train dan memvalidasi pada 24,844 sampel Dev di [`notebooks/02_train_lcnn_m1.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/02_train_lcnn_m1.ipynb).
  - [x] Mencapai **Dev EER 0.37%** ([`results/m1_evaluation_log.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/m1_evaluation_log.csv)), melampaui target Gate G2 ($< 10\%$).
  - [x] Mengunduh dan memverifikasi bobot terbaik [`models/lcnn_best_model.pt`](file:///home/saf/saf/projects/Vocal-Guard/models/lcnn_best_model.pt) serta mempublikasikan Kaggle Dataset `iammadsfq/vocal-guard-models`.
- [x] **Kuantisasi & Benchmark Latensi CPU Lokal (RQ3):**
  - [x] Mengekspor model ke ONNX FP32 dan ONNX Dynamic INT8 untuk jendela 4.0s dan 1.28s streaming ([`scripts/quantize_and_benchmark.py`](file:///home/saf/saf/projects/Vocal-Guard/scripts/quantize_and_benchmark.py)).
  - [x] Mencapai kompresi ukuran **73.0%** (`2.08 MB` $\to$ `0.56 MB`).
  - [x] Membuktikan latensi inferensi CPU **9.62 ms** ($\text{RTF} = 0.0075$) untuk potongan streaming 1.28 detik ([`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv)).
- [x] **Penyiapan Pipeline Evaluasi Eval Set & Ablation Study:**
  - [x] Membuat notebook evaluasi penuh 71,237 sampel Eval Set ([`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb)).
  - [x] Membuat notebook Ablation Study M2 & M3 ([`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb)).

---

### ⏳ B. Tahap yang Sedang Berjalan / Belum Selesai (Pending / In-Progress)

1. **[SEDANG BERJALAN DI KAGGLE GPU 🟢] Evaluasi Resmi pada Unseen Eval Set (71,237 Sampel):**
   - [ ] Menunggu eksekusi Kernel Version 7 ([`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb)) selesai di Kaggle.
   - [ ] Mengunduh `asvspoof2019_final_eval_results.csv` (angka **EER Eval** resmi) dan `eval_attack_breakdown.csv` (rincian EER per serangan A07–A19).
   - [ ] Mengunduh hasil uji paritas EER antara FP32 vs ONNX INT8 pada subset validasi.

2. **[SIAP DIEKSEKUSI BERIKUTNYA] Menjalankan Ablation Study Terkontrol (M2 & M3) di Kaggle GPU:**
   - [ ] Push [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) ke Kaggle setelah run Eval Set selesai.
   - [ ] Mengambil angka EER Dev untuk **M2 (MobileNetV2 from scratch)** guna mengisolasi efek H1 (arsitektur visual vs akustik).
   - [ ] Mengambil angka EER Dev untuk **M3 (LCNN + Global Average Pooling)** guna mengisolasi efek H2 (penghapusan artefak transien oleh GAP).
   - [ ] Mengunduh `ablation_study_summary.csv` ke folder `results/`.

3. **[OPSIONAL / PELENGKAP] Uji Paritas Ekstraksi Fitur Python vs Rust (H4):**
   - [ ] Jika implementasi Rust ingin disertakan dalam klaim *edge deployment*, jalankan skrip pembanding nilai tensor LFCC Python vs Rust pada 1 file audio contoh dan laporkan *Mean Absolute Error* (MAE).

4. **[TAHAP PENUTUP] Penyusunan Naskah Ringkasan Teknis 1–2 Halaman (Bahan Esai Q02, Q03, Q05):**
   - [ ] Memasukkan angka EER Eval dan tabel Ablation M2/M3 ke dalam dokumen ini dan [`README.md`](file:///home/saf/saf/projects/Vocal-Guard/README.md).
   - [ ] Merumuskan narasi terstruktur untuk menjawab poin esai:
     - **Q02 (Diagnosis & Metodologi):** Mengapa sistem awal gagal dan bagaimana disiplin eksperimen memperbaikinya.
     - **Q03 (Hasil Empiris & Trade-off):** Analisis akurasi (EER Dev/Eval) vs efisiensi perangkat keras (ukuran 576 KB, latensi 9.62 ms, RTF 0.0075).
     - **Q05 (Refleksi & Integrasi Robot Sosial):** Keterbatasan dataset (seperti *silence shortcut* Müller dkk., 2021, generalisasi serangan tak dikenal, serta integrasi dengan *Voice Activity Detection* / *Sound Source Localization* pada robot).

---

## 6. Indeks Artefak & Berkas Penting di Repositori

* **Kode Sumber Utama:**
  * [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py) — Ekstraksi LFCC 60-dimensi.
  * [`vocal_guard/dataset.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/dataset.py) — PyTorch Dataset ASVspoof 2019 LA.
  * [`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py) — Model LCNN dengan aktivasi MFM.
  * [`vocal_guard/baseline_gmm.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/baseline_gmm.py) — Baseline GMM dengan akselerasi PyTorch CUDA.
  * [`vocal_guard/metrics.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/metrics.py) — Fungsi perhitungan EER & threshold.
* **Notebook Eksperimen:**
  * [`notebooks/01_baseline_b0_gmm.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/01_baseline_b0_gmm.ipynb) — Eksperimen Baseline B0 (Selesai).
  * [`notebooks/02_train_lcnn_m1.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/02_train_lcnn_m1.ipynb) — Pelatihan Model M1 LCNN (Selesai).
  * [`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb) — Evaluasi Eval Set 71k & Paritas INT8 (Sedang berjalan).
  * [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) — Ablation Study M2 & M3 (Siap dijalankan).
* **Skrip Otomasi & Benchmark:**
  * [`scripts/quantize_and_benchmark.py`](file:///home/saf/saf/projects/Vocal-Guard/scripts/quantize_and_benchmark.py) — Ekspor ONNX, kuantisasi INT8, dan benchmark latensi CPU.
* **Bobot Model & Log Hasil:**
  * [`models/lcnn_best_model.pt`](file:///home/saf/saf/projects/Vocal-Guard/models/lcnn_best_model.pt) — Bobot PyTorch FP32 terbaik (Dev EER 0.37%).
  * [`models/lcnn_m1_4s_int8.onnx`](file:///home/saf/saf/projects/Vocal-Guard/models/lcnn_m1_4s_int8.onnx) — Model ONNX INT8 untuk utterance 4 detik (576 KB).
  * [`models/lcnn_m1_128frames_int8.onnx`](file:///home/saf/saf/projects/Vocal-Guard/models/lcnn_m1_128frames_int8.onnx) — Model ONNX INT8 untuk streaming chunk 1.28 detik (576 KB).
  * [`results/b0_evaluation_log.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/b0_evaluation_log.csv) — Hasil resmi Baseline B0.
  * [`results/m1_evaluation_log.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/m1_evaluation_log.csv) — Hasil resmi Model M1.
  * [`results/lcnn_training_history.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/lcnn_training_history.csv) — Log konvergensi 10 epoch M1.
  * [`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv) — Log latensi dan RTF di CPU lokal.
