# 🔬 Dokumentasi Riset & Log Eksperimen: Vocal-Guard (ASVspoof 2019 LA)

**Judul Penelitian:** *Spoofing-Aware Audio Front-End yang Efisien untuk Robot Sosial: Evaluasi Ulang Vocal-Guard di ASVspoof 2019 LA*  
**Peneliti:** Ahmad Syafiq (`iammadsfq`)  
**Repositori:** `git@github.com:iammadsfq/VocalGuard2.git`  
**Stack Implementasi Saat Ini:** Python, PyTorch (Training & Evaluasi), ONNX Runtime (Kuantisasi & Inferensi CPU). *(Catatan: Eksperimen paritas Rust/PyO3 dan TFLite dari draf awal ditangguhkan untuk memprioritaskan validasi ilmiah pipeline utama dan ablation).*

---

## 1. Ringkasan Eksekutif & Disiplin Metodologi

Antarmuka suara pada robot sosial rentan terhadap serangan *Logical Access (LA)* berupa suara sintetis (*Text-to-Speech* / *Voice Conversion*). Deteksi spoofing untuk robot memiliki kendala *on-device* (tanpa GPU diskrit) dan anggaran latensi yang ketat.

Proyek **Vocal-Guard** generasi pertama mencoba menggabungkan ekstraksi fitur LFCC/CQT dengan backbone visual **MobileNetV2**, namun evaluasi awalnya mengalami kegagalan total dengan **Ensemble EER 41.22%** (hampir setara tebakan acak 50%).

Alih-alih langsung mengganti model secara membabi buta, riset ini menerapkan **disiplin diagnosis sistematis**:
1. **Isolasi Hipotesis (RQ1):** Merumuskan 8 hipotesis penyebab kegagalan (H1–H8), memisahkan evaluasi trek LA dan PA, serta membangun satu sumber kebenaran (*Single Source of Truth*) ekstraksi fitur LFCC 60-dimensi.
2. **Evaluasi Jujur In-Distribution vs Out-of-Distribution (RQ2):** Membedakan secara tegas antara performa pada **Dev Set** (serangan diketahui A01–A06, di mana LCNN mencapai `0.37%` EER namun berisiko *overfit* dan mengeksploitasi *silence shortcut*) dengan performa generalisasi sejati pada **Eval Set** (serangan tak dikenal A07–A19, yang menjadi metrik utama/*headline* riset).
3. **Pengukuran Empiris Trade-off Kuantisasi & Latensi Chunk (RQ3):** Mengukur dampak kuantisasi **ONNX Dynamic INT8** terhadap ukuran memori (`576 KB`) dan latensi inferensi per potongan (*chunk*) pada CPU laptop sebagai *proxy* awal, termasuk melaporkan secara transparan *overhead* latensi INT8 dinamis dibanding FP32 pada prosesor x86 tanpa instruksi VNNI.

---

## 2. Diagnosis Kegagalan Sistem Awal & Status Hipotesis (H1–H8)

Hasil evaluasi sistem generasi pertama (dari log evaluasi proyek awal):
* **Ensemble EER:** `41.22%` | **AUC:** `0.6124` | **Akurasi LA:** `59.40%` | **Akurasi PA:** `49.06%`

Perbaikan dari `41.22%` ke `0.37%` (Dev EER pada model M1) merupakan hasil dari **paket perubahan simultan** (backbone, pooling, resolusi fitur, rumus LFCC, dan pemisahan trek). Oleh karena itu, kontribusi masing-masing faktor **tidak boleh diklaim sebelum diisolasi lewat eksperimen ablation terkontrol**:

| Kode | Hipotesis Penyebab Kegagalan | Mekanisme Teknis | Cara Menguji Secara Terkontrol | Status Saat Ini (Jujur & Defensible) |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Backbone **MobileNetV2 beku** (*frozen ImageNet*) tidak mengenali pola akustik | Bobot ImageNet dilatih pada tekstur objek visual, bukan korelasi subband frekuensi suara sintetis. | Bandingkan MobileNetV2 Frozen (`41.22%`) vs MobileNetV2 dilatih dari nol pada LFCC (**M2**) vs LCNN+GAP (**M3**). | **Belum terisolasi.** Menunggu hasil run **M2** di [`04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb). |
| **H2** | **Global Average Pooling (GAP)** menghapus artefak transien | Merata-ratakan seluruh sumbu waktu ($T \to 1$) melenyapkan anomali fasa/vocoder skala milidetik. | Bandingkan LCNN + Adaptive MFM `3x4` (**M1**) vs LCNN + GAP `1x1` (**M3**) dengan semua faktor lain tetap. | **Belum terisolasi.** Menunggu hasil run **M3** di [`04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb). |
| **H3** | **Resize 20 koefisien ke 224x224 piksel** mendistorsi frekuensi | Interpolasi gambar mengaburkan batas tajam antar-koefisien DCT dan menambah beban komputasi. | Gunakan tensor LFCC resolusi asli `(1, 60, 399)` tanpa resize. | **Sudah diterapkan** sebagai bagian paket perbaikan M1. |
| **H4** | **Rumus LFCC awal keliru** (`dct(stft)` tanpa filterbank/log) | Tanpa filterbank segitiga linier dan kompresi logaritmik, fitur kehilangan sifat cepstral. | Implementasi referensi tunggal LFCC 60-dimensi di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py). | **Sudah diperbaiki di Python.** *(Uji paritas dengan kode Rust belum dikerjakan).* |
| **H5** | **Normalisasi tidak konsisten** antar tahap | Perbedaan penskalaan saat training dan inferensi menggeser distribusi skor terhadap *threshold*. | Satu fungsi prapemrosesan deterministik untuk semua split. | **Sudah diterapkan** di pipeline baru. |
| **H6** | **Window 1 detik & eksploitasi durasi hening** (*silence shortcut*) | Müller dkk. (2021) membuktikan durasi hening di ASVspoof 2019 LA berkorelasi dengan label. Tanpa *trimming*, model dapat menghafal panjang hening. | Uji silang **M1** pada Dev yang di-trim heningnya (**M1-CrossTrim**) dan latih **M1-Trim** dengan *energy VAD trimming*. | **Masih terbuka.** Dev EER `0.37%` belum bebas dari *silence shortcut*; diuji di [`04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb). |
| **H7** | **Penggabungan skor LA dan PA** dalam satu angka tidak valid | Artefak sintesis (LA) dan pantulan ruang (PA) memiliki sifat fisis yang berbeda. | Evaluasi khusus pada trek **ASVspoof 2019 LA** sesuai protokol resmi. | **Sudah diterapkan.** |
| **H8** | Kemungkinan bug pembacaan label atau protokol data | Jika mapping ID file ke label terbalik/acak, model apa pun akan menghasilkan EER ~50%. | Jalankan model klasik **LFCC-GMM (B0)** untuk memastikan skor *bonafide* > *spoof* secara konsisten. | **Pipeline data terbukti tidak rusak** (EER Dev `32.18%`, di atas acak; namun fitur/GMM belum setara baseline resmi ASVspoof ~8–9%). |

---

## 3. Metodologi & Spesifikasi Pipeline Standar

### 3.1 Dataset & Protokol Resmi
Eksperimen menggunakan dataset **ASVspoof 2019 Logical Access (LA)** dengan pembagian split resmi tanpa modifikasi:

| Split | File Protokol Resmi | Total Utterance | Bonafide | Spoof | Jenis Serangan (*Attack IDs*) | Peran & Interpretasi Metrik |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | `ASVspoof2019.LA.cm.train.trn.txt` | 25,380 | 2,580 | 22,800 | A01 – A06 (TTS & VC diketahui) | Pelatihan parameter model |
| **Dev** | `ASVspoof2019.LA.cm.dev.trl.txt` | 24,844 | 2,548 | 22,296 | A01 – A06 (*In-distribution attacks*) | Pemilihan model & *ablation* (bukan klaim generalisasi akhir) |
| **Eval** | `ASVspoof2019.LA.cm.eval.trl.txt` | 71,237 | 7,355 | 63,882 | **A07 – A19** (*Unseen attacks*) | **Metrik Utama (Headline):** Mengukur generalisasi terhadap serangan baru |

### 3.2 Spesifikasi Ekstraksi Fitur LFCC (*Single Source of Truth*)
Diimplementasikan pada [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py):
* **Panjang Sinyal Default:** $16\text{ kHz}$, $4.0\text{ detik}$ ($64,000\text{ sampel}$) dengan *truncation* atau *wrap-around repeat padding*.
* **Parameter Spektral:** Pre-emphasis $\alpha = 0.97$, jendela Hann $20\text{ ms}$ ($320\text{ sampel}$), hop $10\text{ ms}$ ($160\text{ sampel}$), RFFT $512$-titik, $20$ filter segitiga linier ($0–8\text{ kHz}$), kompresi logaritmik, dan DCT-II ortonormal ($20$ koefisien statis).
* **Fitur Dinamis:** Turunan pertama ($\Delta$) dan kedua ($\Delta\Delta$) dengan jendela $N=2$, membentuk matriks $(60, 399)$ per utterance 4 detik.

### 3.3 Arsitektur Model: Light CNN (LCNN) dengan Max-Feature-Map (MFM)
Diimplementasikan pada [`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py) (539,714 parameter):
* **Aktivasi Kompetitif MFM:** $\text{MFM}(x)_k = \max(x_k, x_{k+C})$, menyaring fitur secara selektif di setiap blok konvolusi.
* **Penanganan Imbalance:** *Weighted Cross-Entropy Loss* dengan bobot kelas terhitung dari rasio train set ($w_{\text{bonafide}} = 4.9186$, $w_{\text{spoof}} = 0.5566$).
* **Skor Keputusan:** $s(x) = \text{logit}_{\text{bonafide}}(x) - \text{logit}_{\text{spoof}}(x)$.

---

## 4. Hasil Eksperimen & Analisis Kritis

### 4.1 Tabel Perbandingan Performa Model (ASVspoof 2019 LA)

| Kode | Konfigurasi Model | Fitur & Window | Parameter | Ukuran File | Dev EER (%) *(In-Dist A01–A06)* | **Eval EER (%) *(Unseen A07–A19)*** | Interpretasi Ilmiah |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224`, 1s | ~3.50 M | ~14.0 MB | `41.22%` | — | Gagal (mendekati tebakan acak) |
| **B0** | **LFCC-GMM** (256 diag, custom LFCC) | LFCC `(T, 60)`, Full | Statistik | `0.97 MB` | `32.18%` | — | Di atas acak (pipeline label tidak rusak, namun belum setara GMM resmi ~8–9%) |
| **M1** | **LCNN + MFM (FP32)** | LFCC `(1, 60, 399)`, 4s | **539,714** | `2.08 MB` | `0.37%` | **`4.59%`** 🏆 | **Angka Headline:** Sangat kuat di A07–A16 & A18–A19 ($\le 1.85\%$), terdegradasi pada serangan akustik VC **A17 (`25.73%`)** |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC `(1, 60, 399)`, 4s | **539,714** | **`0.56 MB`** | *Sedang berjalan (v8)* ⏳ | — | Kompresi memori 73.0%; paritas EER 2,000 sampel sedang berjalan di Kernel v8 |
| **M3** | **LCNN + GAP (Ablation H2)** | LFCC `(1, 60, 399)`, 4s | 314,434 | `1.20 MB` | *Sedang berjalan (v8)* ⏳ | — | Mengukur penalti akurasi akibat penggantian MFM Pool `3x4` dengan GAP `1x1` |
| **M2** | **MobileNetV2 From-Scratch (Ablation H1)** | LFCC `(1, 60, 399)`, 4s | 2,225,858 | `8.50 MB` | *Sedang berjalan (v8)* ⏳ | — | Mengukur efek backbone MobileNetV2 jika dilatih dari nol pada LFCC |
| **M1-Trim** | **LCNN + MFM + VAD Trim (Ablation H6)** | LFCC `(1, 60, 399)`, 4s Trimmed | 539,714 | `2.08 MB` | *Sedang berjalan (v8)* ⏳ | — | Mengukur performa murni tanpa *silence shortcut* (Müller dkk., 2021) |
| **M1-1.28s**| **LCNN + MFM (Evaluasi Chunk 1.28s)** | LFCC `(1, 60, 128)`, 1.28s | 539,714 | `2.08 MB` | *Sedang berjalan (v8)* ⏳ | — | Mengukur EER riil pada potongan 1.28s untuk mendampingi klaim latensi chunk |

---

### 4.2 Rincian Performa Generalisasi pada Unseen Eval Set (71,237 Utterance: A07–A19)
Sumber data: [`results/eval_attack_breakdown.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/eval_attack_breakdown.csv) & [`results/asvspoof2019_final_eval_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/asvspoof2019_final_eval_results.csv)

* **Pooled Eval EER (Seluruh 71,237 sampel):** **`4.59%`** (Threshold optimal Eval: `+7.9534` vs Threshold Dev: `-1.1896`).
* Berikut adalah rincian EER untuk masing-masing dari 13 algoritma serangan tak dikenal (*unseen attacks*) dibandingkan terhadap 7,355 sampel *bonafide* Eval:

| Attack ID | Kategori Serangan (ASVspoof 2019 LA) | Jumlah Sampel | EER per Serangan (%) | Rata-rata Skor Logit | Analisis & Interpretasi |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A07** | TTS (Neural Vocoder / WaveNet) | 4,914 | `0.48%` | `-15.0977` | Terdeteksi sangat baik |
| **A08** | TTS (Neural Waveform) | 4,914 | `0.09%` | `-30.9816` | Terdeteksi hampir sempurna |
| **A09** | TTS (Vocoder Tradisional) | 4,914 | **`0.00%`** | `-46.3613` | Terdeteksi sempurna |
| **A10** | TTS (End-to-End Neural TTS) | 4,914 | `1.48%` | `-7.6414` | Generalisasi baik |
| **A11** | TTS (Neural / Griffin-Lim) | 4,914 | `0.55%` | `-15.1364` | Terdeteksi sangat baik |
| **A12** | TTS (Neural WaveNet) | 4,914 | `1.39%` | `-13.2628` | Generalisasi baik |
| **A13** | VC + TTS Combined | 4,914 | `1.85%` | `-6.0059` | Generalisasi baik |
| **A14** | VC + Vocoder | 4,914 | `0.71%` | `-14.0749` | Terdeteksi sangat baik |
| **A15** | VC + Neural Waveform | 4,914 | `1.14%` | `-10.5323` | Generalisasi baik |
| **A16** | TTS (Known-type Waveform Concatenation) | 4,914 | `0.59%` | `-16.7935` | Terdeteksi sangat baik |
| **A17** | **VC (Waveform Filtering / Direct Acoustic VC)** | **4,914** | **`25.73%`** ⚠️ | **`+9.2221`** | **Titik Lemah Utama:** Skor positif menyerupai *bonafide* |
| **A18** | VC (Non-parallel / i-vector + Vocoder) | 4,914 | `1.85%` | `-8.3047` | Generalisasi baik |
| **A19** | VC (Spectral Mapping) | 4,914 | `1.30%` | `-16.4945` | Generalisasi baik |

> **Analisis Ilmiah Dev (`0.37%`) vs Eval (`4.59%`) dan Anomali A17 (`25.73%`):**
> 1. **Kesenjangan In-Distribution vs Out-of-Distribution:** Kenaikan EER dari `0.37%` (Dev) menjadi `4.59%` (Eval) serta pergeseran ambang batas optimal dari `-1.19` ke `+7.95` membuktikan bahwa metrik Dev tidak boleh dijadikan klaim tunggal.
> 2. **Mengapa A17 Gagal (`25.73%` EER, Mean Score `+9.22`)?** Pada literatur resmi ASVspoof 2019 (Nautiyal dkk. / Todisco dkk., 2019), **A17** adalah serangan *Voice Conversion* berbasis penyaringan gelombang langsung (*direct waveform filtering*) yang mempertahankan eksitasi fase asli pembicara sumber tanpa melewati *vocoder* sintetis konvensional. Karena model LCNN kita belajar mendeteksi artefak frekuensi tinggi dari *vocoder* (yang sangat jelas pada A07–A16 dan A18–A19 di mana EER $\le 1.85\%$), sampel A17 berhasil mengelabui model dan memperoleh skor rata-rata positif (`+9.2221`). Tanpa A17, rata-rata EER pada 12 serangan tak dikenal lainnya berada di bawah **`1.0%`**. Temuan ini menjadi bukti empiris yang sangat kaya untuk diskusi esai Q03 dan Q05.


---

### 4.2 Hasil Kuantisasi & Benchmark Latensi Per Chunk di CPU Laptop (Proxy RQ3)
Sumber data: [`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv)  
Perangkat Uji: **CPU Laptop Intel Core i7-8665U @ 1.90GHz** (digunakan sebagai *proxy* komputasi CPU tanpa GPU; **belum diuji pada prosesor ARM embedded** seperti Raspberry Pi / Jetson).

#### A. Kompresi Ukuran Model & Status Paritas Akurasi
* **PyTorch FP32 (`lcnn_best_model.pt`):** `2.08 MB`
* **ONNX FP32 (`lcnn_m1_4s.onnx`):** `2.08 MB`
* **ONNX Dynamic INT8 (`lcnn_m1_4s_int8.onnx`):** **`0.56 MB`** (**Kompresi ukuran file 73.0%**)
* **Status Paritas Akurasi:** Uji pada satu sampel acak menunjukkan selisih skor logit $\Delta = 0.4459$ (`+35.33` vs `+34.88`). **Namun, satu sampel belum membuktikan bahwa EER terjaga.** Uji paritas EER pada 2,000 sampel Dev Set sedang dijalankan di [`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb).

#### B. Latensi Inferensi Per Utterance (4.0s) dan Per Chunk (1.28s)
*Catatan Metodologis:* Pengujian pada input `1.28 detik` (`128 frame`) di bawah ini mengukur **waktu komputasi forward-pass per potongan (*chunk*) statis**, bukan *stateful streaming* dengan *ring-buffer* aktif, dan EER untuk potongan 1.28 detik baru akan diukur pada notebook ablation.

| Skenario Input Tensor | Runtime & Presisi | Thread CPU | Mean Latency (ms) | p95 Latency (ms) | Real-Time Factor (RTF) | Temuan Empiris |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Chunk 1.28 detik**<br>*(128 frame LFCC)* | **ONNX Runtime FP32** | **1 Core** | **9.62 ms** | **12.34 ms** | **0.0075** | **Paling cepat di CPU x86 non-VNNI** |
| | PyTorch Native FP32 | 1 Core | 17.50 ms | 24.18 ms | 0.0137 | Baseline eager execution |
| | ONNX Dynamic INT8 | 4 Cores | 21.15 ms | 23.70 ms | 0.0165 | Lebih lambat dari FP32 karena overhead quant/dequant |
| | ONNX Dynamic INT8 | 1 Core | 37.89 ms | 43.87 ms | 0.0296 | 3.9x lebih lambat dari ONNX FP32 1-core |
| **Utterance 4.00 detik**<br>*(399 frame LFCC)* | **ONNX Runtime FP32** | **1 Core** | **22.28 ms** | **26.08 ms** | **0.0056** | **Paling efisien untuk evaluasi 4 detik** |
| | PyTorch Native FP32 | 1 Core | 38.90 ms | 42.90 ms | 0.0097 | Baseline eager execution |
| | ONNX Dynamic INT8 | 4 Cores | 66.12 ms | 78.30 ms | 0.0165 | Overhead konversi dinamis per layer |
| | ONNX Dynamic INT8 | 1 Core | 125.57 ms | 142.99 ms | 0.0314 | 5.6x lebih lambat dari ONNX FP32 1-core |

> **Temuan Kunci (Trade-off Jujur INT8 vs FP32):**
> Meskipun **ONNX Dynamic INT8** berhasil memangkas ukuran penyimpanan hingga **73.0%** (`2.08 MB` $\to$ `0.56 MB`), waktu inferensinya pada CPU Intel i7-8665U justru **lebih lambat dibanding ONNX FP32** (`37.89 ms` vs `9.62 ms` pada 1 core). Hal ini disebabkan oleh dua faktor arsitektural: (1) prosesor generasi ke-8 (Whiskey Lake) belum mendukung instruksi perangkat keras *AVX-512 VNNI* untuk perkalian matriks 8-bit terakselerasi, dan (2) skema *dynamic quantization* harus menghitung faktor skala (*min/max scaling*) serta melakukan kuantisasi-dekuantisasi aktivasi secara berulang di setiap lapisan konvolusi saat *runtime*. Sebaliknya, **ONNX FP32** dapat langsung mengeksekusi instruksi vektor *fused multiply-add* (FMA/AVX2) secara kontinu. Temuan ini membuktikan bahwa pemilihan format deployment pada robot sosial harus disesuaikan dengan dukungan instruksi SIMD perangkat keras target, bukan sekadar berasumsi INT8 selalu lebih cepat.

---

## 5. Checklist Progres Riset & Urutan Kerja (Target 20 Oktober)

### ✅ A. Sudah Selesai Dikerjakan (Completed)
- [x] **Disiplin Protokol & Pemisahan Trek:** Memisahkan evaluasi LA dari PA, menggunakan split resmi ASVspoof 2019 LA tanpa kebocoran data.
- [x] **Single Source of Truth Fitur LFCC:** Implementasi Python deterministik 60-dimensi di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py).
- [x] **Uji Kewarasan Pipeline Data (B0):** Memastikan pipeline label dan ekstraksi fitur berjalan di atas tebakan acak (`32.18%` Dev EER) di [`notebooks/01_baseline_b0_gmm.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/01_baseline_b0_gmm.ipynb).
- [x] **Pelatihan Model LCNN (M1):** Mencapai `0.37%` Dev EER pada serangan *in-distribution* A01–A06 ([`notebooks/02_train_lcnn_m1.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/02_train_lcnn_m1.ipynb)).
- [x] **Pengukuran Latensi & Ukuran Kuantisasi di CPU Proxy:** Mengukur ukuran model (`0.56 MB` INT8 vs `2.08 MB` FP32) dan latensi inferensi per chunk 1.28s & 4.0s di [`scripts/quantize_and_benchmark.py`](file:///home/saf/saf/projects/Vocal-Guard/scripts/quantize_and_benchmark.py).
- [x] **Koreksi Klaim & Penyiapan Notebook Ablation Lengkap:** Memperbarui [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) agar mencakup **M3** (LCNN+GAP), **M2** (MobileNetV2 from scratch), **M1-Trim** (Silence Trimming H6), dan **EER Chunk 1.28s**.

---

### ⏳ B. Urutan Kerja Prioritas yang Sedang & Akan Dijalankan (Pending)

1. **[PRIORITAS 1 — SEDANG BERJALAN DI KAGGLE GPU 🟢] Ambil EER Eval (A07–A19) & Paritas EER INT8:**
   - [ ] Menunggu eksekusi Kernel Version 7 ([`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb)) selesai.
   - [ ] Mengunduh `asvspoof2019_final_eval_results.csv` dan menjadikan **Eval EER** sebagai angka utama (*headline*) riset.
   - [ ] Mengunduh `eval_attack_breakdown.csv` untuk menganalisis serangan tak dikenal tersulit (misalnya A17/A18) vs termudah.
   - [ ] Mencatat hasil uji paritas EER FP32 vs INT8 pada 2,000 sampel subset Dev.

2. **[PRIORITAS 2 — SIAP PUSH SETELAH EVAL SELESAI] Jalankan Eksperimen Ablation (M3, M2, Silence Trim H6, & Chunk 1.28s):**
   - [ ] Push [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) ke Kaggle GPU.
   - [ ] **Hasil M3 (`LCNN + GAP`):** Menjawab seberapa besar kerusakan akurasi akibat GAP (menguji H2 secara terisolasi).
   - [ ] **Hasil M2 (`MobileNetV2 from scratch`):** Menjawab seberapa besar pengaruh arsitektur backbone ketika dilatih dari nol (menguji H1 secara terisolasi).
   - [ ] **Hasil H6 (`M1-CrossTrim` & `M1-Trim`):** Menjawab seberapa besar porsi `0.37%` Dev EER yang berasal dari *silence shortcut* (Müller dkk., 2021).
   - [ ] **Hasil `M1-Chunk1.28s`:** Melengkapi klaim latensi chunk 1.28 detik dengan angka EER yang sesuai.

3. **[PRIORITAS 3 — PENYUSUNAN NARASI ESAI Q02, Q03, Q05]:**
   - [ ] Memasukkan seluruh angka Eval dan Ablation ke tabel dokumentasi dan [`README.md`](file:///home/saf/saf/projects/Vocal-Guard/README.md).
   - [ ] Menyusun draf paragraf esai Q02, Q03, dan Q05 yang menonjolkan **disiplin diagnosis, kejujuran evaluasi generalisasi (Dev vs Eval), serta temuan empiris trade-off INT8 vs FP32 dan RTF**.
