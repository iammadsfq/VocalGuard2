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

| Kode | Hipotesis Penyebab Kegagalan | Mekanisme Teknis | Cara Menguji Secara Terkontrol | Status Akhir (Teruji Secara Empiris) |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Backbone **MobileNetV2 beku** (*frozen ImageNet*) tidak mengenali pola akustik | Bobot ImageNet dilatih pada tekstur objek visual, bukan korelasi subband frekuensi suara sintetis. | Bandingkan MobileNetV2 Frozen (`41.22%`) vs MobileNetV2 dilatih dari nol pada LFCC (**M2**: `1.28%`) vs LCNN+GAP (**M3**: `0.47%`). | **Terisolasi & Terbukti.** Melatih MobileNetV2 dari nol pada LFCC 4s menurunkan EER dari `41.22%` ke **`1.28%`**, dan penggantian ke LCNN+MFM (**M3**) memangkas EER lebih lanjut sebesar 63% relatif menjadi **`0.47%`** dengan 7x lebih sedikit parameter. |
| **H2** | **Global Average Pooling (GAP)** menghapus resolusi spektro-temporal | Merata-ratakan seluruh sumbu waktu-frekuensi ke `1x1` menghilangkan lokalisasi artefak transien. | Bandingkan LCNN + Adaptive MFM `3x4` (**M1**: `0.37%`) vs LCNN + GAP `1x1` (**M3**: `0.47%`) dengan semua faktor lain tetap. | **Terisolasi & Terukur.** Ketika fitur konvolusional MFM sudah diekstraksi, GAP tidak merusak model secara total (`0.47%`), namun mempertahankan resolusi `3x4` (**M1**) meningkatkan Dev EER dari **`0.47%` menjadi `0.37%`** (perbaikan relatif 21%). |
| **H3** | **Resize 20 koefisien ke 224x224 piksel** mendistorsi frekuensi | Interpolasi gambar mengaburkan batas tajam antar-koefisien DCT dan menambah beban komputasi. | Gunakan tensor LFCC resolusi asli `(1, 60, 399)` tanpa resize. | **Sudah diterapkan** pada seluruh model M1, M2, dan M3. |
| **H4** | **Rumus LFCC awal keliru** (`dct(stft)` tanpa filterbank/log) | Tanpa filterbank segitiga linier dan kompresi logaritmik, fitur kehilangan sifat cepstral. | Implementasi referensi tunggal LFCC 60-dimensi di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py). | **Sudah diperbaiki di Python.** *(Uji paritas dengan kode Rust ditangguhkan).* |
| **H5** | **Normalisasi tidak konsisten** antar tahap | Perbedaan penskalaan saat training dan inferensi menggeser distribusi skor terhadap *threshold*. | Satu fungsi prapemrosesan deterministik untuk semua split. | **Sudah diterapkan** di pipeline utama dan web demo. |
| **H6** | **Window 1 detik & eksploitasi durasi hening** (*silence shortcut*) | Müller dkk. (2021) membuktikan durasi hening di ASVspoof 2019 LA berkorelasi dengan label. Tanpa *trimming*, model dapat menghafal panjang hening. | Uji silang **M1** pada Dev yang di-trim heningnya (**M1-CrossTrim**) dan latih **M1-Trim** dengan *energy VAD trimming*. | **Terbukti Secara Empiris.** Saat hening dihapus saat uji (**M1-CrossTrim**), EER M1 melonjak dari `0.37%` ke **`12.11%`**, membuktikan adanya ketergantungan pada *silence shortcut*. Pelatihan ulang dengan VAD trim (**M1-Trim**) memulihkan Dev EER ke **`6.71%`** murni dari artefak spektral suara aktif. |
| **H7** | **Penggabungan skor LA dan PA** dalam satu angka tidak valid | Artefak sintesis (LA) dan pantulan ruang (PA) memiliki sifat fisis yang berbeda. | Evaluasi khusus pada trek **ASVspoof 2019 LA** sesuai protokol resmi. | **Sudah diterapkan.** |
| **H8** | Kemungkinan bug pembacaan label atau protokol data | Jika mapping ID file ke label terbalik/acak, model apa pun akan menghasilkan EER ~50%. | Jalankan model klasik **LFCC-GMM (B0)** untuk memastikan skor *bonafide* > *spoof* secara konsisten. | **Pipeline data terbukti tidak rusak** (EER Dev `32.18%`, di atas acak; namun fitur/GMM kustom belum setara baseline resmi ASVspoof ~8–9%). |

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
* **Fitur Dinamis:** Turunan pertama ($\Delta$) dan kedua ($\Delta\Delta$) dengan jendela $N=2$, membentuk matriks $(60, 399)$ per utterance 4 detik (atau $(60, 128)$ untuk potongan *chunk* 1.28 detik).

### 3.3 Arsitektur Model: Light CNN (LCNN) dengan Max-Feature-Map (MFM)
Diimplementasikan pada [`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py) (539,714 parameter):
* **Aktivasi Kompetitif MFM:** $\text{MFM}(x)_k = \max(x_k, x_{k+C})$, menyaring fitur secara selektif di setiap blok konvolusi.
* **Penanganan Imbalance:** *Weighted Cross-Entropy Loss* dengan bobot kelas terhitung dari rasio train set ($w_{\text{bonafide}} = 4.9186$, $w_{\text{spoof}} = 0.5566$).
* **Skor Keputusan:** $s(x) = \text{logit}_{\text{bonafide}}(x) - \text{logit}_{\text{spoof}}(x)$.

---

## 4. Hasil Eksperimen & Analisis Kritis

### 4.1 Tabel Perbandingan Performa Model & Studi Ablasi (ASVspoof 2019 LA)
Sumber data: [`results/ablation_study_summary.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/ablation_study_summary.csv) & [`results/asvspoof2019_final_eval_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/asvspoof2019_final_eval_results.csv)

| Kode | Konfigurasi Model | Fitur & Window / Trim | Parameter | Ukuran File | Dev EER (%) *(In-Dist A01–A06)* | **Eval EER (%) *(Unseen A07–A19)*** | Interpretasi Ilmiah |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224`, 1s | ~3,500,000 | ~14.0 MB | `41.22%` | — | Gagal (mendekati tebakan acak 50%) |
| **B0** | **LFCC-GMM** (256 diag, custom LFCC) | LFCC `(T, 60)`, Full | Statistik | `0.97 MB` | `32.18%` | — | Di atas acak (pipeline label valid, belum setara GMM resmi ~8–9%) |
| **M2** | **MobileNetV2 From-Scratch (Ablation H1)** | LFCC `(1, 60, 399)`, 4s / No Trim | 2,225,858 | `8.50 MB` | **`1.28%`** | — | Membuka bekukan bobot + fitur LFCC 60-dim 4s menurunkan EER dari `41.22%` ke `1.28%` |
| **M3** | **LCNN + GAP `1x1` (Ablation H2)** | LFCC `(1, 60, 399)`, 4s / No Trim | 314,434 | `1.20 MB` | **`0.47%`** | — | Mengganti MobileNetV2 dengan LCNN+MFM memangkas EER `1.28%` $\to$ `0.47%` dengan 7x lebih hemat parameter |
| **M1** | **LCNN + MFM + AdaptivePool `3x4` (FP32)** | LFCC `(1, 60, 399)`, 4s / No Trim | **539,714** | `2.08 MB` | **`0.37%`** | **`4.59%`** 🏆 | **Angka Headline:** Mempertahankan resolusi `3x4` memperbaiki Dev EER ke `0.37%` dan mencapai **`4.59%` Eval EER** |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC `(1, 60, 399)`, 4s (2k Dev sub) | **539,714** | **`0.56 MB`** | **`0.59%`** *(vs FP32 `0.59%`)* | — | **Paritas 100% ($\Delta\text{EER} = 0.000\%$)** pada subset 2,000 sampel Dev; kompresi memori 73.0% |
| **M1-CrossTrim** | **LCNN + MFM (Uji Silang H6a)** | Latih Tanpa Trim $\to$ Uji VAD Trim | 539,714 | `2.08 MB` | **`12.11%`** | — | Membuktikan M1 tanpa trim sebagian bergantung pada *silence shortcut* (`0.37%` $\to$ `12.11%`) |
| **M1-Trim** | **LCNN + MFM + VAD Trim (Uji H6b)** | Latih & Uji dengan VAD Trim | 539,714 | `2.08 MB` | **`6.71%`** | — | Performa murni dari segmen suara aktif setelah *silence shortcut* dieliminasi |
| **M1-Chunk1.28s**| **LCNN + MFM (Evaluasi Chunk 1.28s)** | LFCC `(1, 60, 128)`, 1.28s / No Trim | 539,714 | `2.08 MB` | **`1.33%`** | — | Akurasi riil yang berpasangan langsung dengan latensi inferensi per chunk 1.28s (`9.62 ms`) |

> **Temuan Kunci dari Studi Ablasi Terkontrol (M2, M3, H6, Chunk 1.28s):**
> 1. **Dekomposisi Perbaikan `41.22%` $\to$ `0.37%` (H1 vs H2):**
>    * Lonjakan terbesar (`41.22%` $\to$ `1.28%` pada **M2**) berasal dari penghapusan pembekuan bobot visual ImageNet (*unfreezing*) dan perbaikan representasi input ke LFCC 60-dimensi 4 detik tanpa *bilinear resize*.
>    * Penggantian arsitektur dari `MobileNetV2` (**M2**, 2.23M parameter, `1.28%` EER) ke `LCNN + MFM` (**M3**, 0.31M parameter, `0.47%` EER) menurunkan tingkat kesalahan sebesar **63% secara relatif** sekaligus menghemat **86% parameter**.
>    * Penggantian `GAP (1x1)` (**M3**, `0.47%`) menjadi `AdaptiveAvgPool2d((3, 4))` (**M1**, `0.37%`) memberikan peningkatan tambahan (`-0.10%` EER) dengan mempertahankan lokalisasi spektro-temporal kasar sebelum lapisan *fully-connected*.
> 2. **Pembuktian Empiris *Silence Shortcut* (H6 — Müller dkk., 2021):**
>    * Ketika model **M1** (yang dilatih tanpa pemotongan hening, `0.37%` Dev EER) dievaluasi pada Dev Set yang telah dipotong bagian heningnya menggunakan *Energy VAD* (**M1-CrossTrim**), EER meningkat tajam menjadi **`12.11%`**. Hal ini mengonfirmasi secara empiris peringatan Müller dkk. (2021) bahwa sebagian performa `0.37%` memang berasal dari korelasi durasi hening di dataset ASVspoof 2019 LA.
>    * Ketika model dilatih dan dievaluasi secara konsisten dengan *VAD silence trimming* (**M1-Trim**), EER turun kembali menjadi **`6.71%`**, menunjukkan kemampuan diskriminasi akustik murni tanpa jalan pintas durasi hening.
> 3. **Trade-off Panjang Konteks Chunk `1.28s` vs Utterance `4.0s`:**
>    * Memotong durasi input dari `4.0 detik` (399 frame) menjadi potongan `1.28 detik` (128 frame, **M1-Chunk1.28s**) meningkatkan Dev EER dari `0.37%` menjadi **`1.33%`** sebagai kompensasi atas waktu inferensi yang sangat cepat (**`9.62 ms`** per chunk di CPU).

---

### 4.2 Rincian Performa Generalisasi pada Unseen Eval Set (71,237 Utterance: A07–A19)
Sumber data: [`results/eval_attack_breakdown.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/eval_attack_breakdown.csv) & [`results/asvspoof2019_final_eval_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/asvspoof2019_final_eval_results.csv)

* **Pooled Eval EER (Seluruh 71,237 sampel):** **`4.59%`** (Threshold optimal Eval: `+7.9534` vs Threshold Dev: `-1.1896`).
* Berikut adalah rincian EER untuk masing-masing dari 13 algoritma serangan tak dikenal (*unseen attacks*) dibandingkan terhadap 7,355 sampel *bonafide* Eval:

| Attack ID | Kategori Serangan (ASVspoof 2019 LA) | Jumlah Sampel | EER per Serangan (%) | Rata-rata Skor Logit | Analisis & Interpretasi |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A07** | TTS (Neural Vocoder / WaveNet) | 4,914 | `0.48%` | `-12.64` | Terdeteksi sangat baik |
| **A08** | TTS (Neural Waveform) | 4,914 | `0.09%` | `-19.55` | Terdeteksi hampir sempurna |
| **A09** | TTS (Vocoder Tradisional) | 4,914 | **`0.00%`** | `-20.77` | Terdeteksi sempurna |
| **A10** | TTS (End-to-End Neural TTS) | 4,914 | `1.48%` | `-2.01` | Generalisasi baik |
| **A11** | TTS (Neural / Griffin-Lim) | 4,914 | `0.55%` | `-5.59` | Terdeteksi sangat baik |
| **A12** | TTS (Neural WaveNet) | 4,914 | `1.39%` | `-1.84` | Generalisasi baik |
| **A13** | VC + TTS Combined | 4,914 | `1.85%` | `+0.51` | Generalisasi baik |
| **A14** | VC + Vocoder | 4,914 | `0.71%` | `-6.06` | Terdeteksi sangat baik |
| **A15** | VC + Neural Waveform | 4,914 | `1.14%` | `-2.69` | Generalisasi baik |
| **A16** | TTS (Known-type Waveform Concatenation) | 4,914 | `0.59%` | `-8.47` | Terdeteksi sangat baik |
| **A17** | **VC (Waveform Filtering / Direct Acoustic VC)** | **4,914** | **`25.73%`** ⚠️ | **`+9.22`** | **Titik Lemah Utama:** Skor positif menyerupai *bonafide* |
| **A18** | VC (Non-parallel / i-vector + Vocoder) | 4,914 | `1.85%` | `-1.94` | Generalisasi baik |
| **A19** | VC (Spectral Mapping) | 4,914 | `1.30%` | `-6.03` | Generalisasi baik |

> **Analisis Ilmiah Dev (`0.37%`) vs Eval (`4.59%`) dan Anomali A17 (`25.73%`):**
> 1. **Kesenjangan In-Distribution vs Out-of-Distribution:** Kenaikan EER dari `0.37%` (Dev) menjadi `4.59%` (Eval) serta pergeseran ambang batas optimal dari `-1.19` ke `+7.95` membuktikan bahwa metrik Dev tidak boleh dijadikan klaim tunggal.
> 2. **Mengapa A17 Gagal (`25.73%` EER, Mean Score `+9.22`)?** Pada literatur resmi ASVspoof 2019 (Todisco dkk., 2019), **A17** adalah serangan *Voice Conversion* berbasis penyaringan gelombang langsung (*direct waveform filtering*) yang mempertahankan eksitasi fase asli pembicara sumber tanpa melewati *vocoder* sintetis konvensional. Karena model LCNN kita belajar mendeteksi artefak frekuensi tinggi dari *vocoder* (yang sangat jelas pada A07–A16 dan A18–A19 di mana EER $\le 1.85\%$), sampel A17 berhasil mengelabui model dan memperoleh skor rata-rata positif (`+9.22`). Tanpa A17, EER pada 12 serangan tak dikenal lainnya berada di rentang **`0.00%` – `1.85%`**.

---

### 4.3 Hasil Kuantisasi & Benchmark Latensi Per Chunk di CPU Laptop (Proxy RQ3)
Sumber data: [`results/cpu_benchmark_results.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/cpu_benchmark_results.csv) & [`results/ablation_study_summary.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/ablation_study_summary.csv)  
Perangkat Uji: **CPU Laptop Intel Core i7-8665U @ 1.90GHz** (digunakan sebagai *proxy* komputasi CPU tanpa GPU; **belum diuji pada prosesor ARM embedded** seperti Raspberry Pi / Jetson).

#### A. Kompresi Ukuran Model & Paritas EER (2,000 Sampel Dev)
* **PyTorch FP32 (`lcnn_best_model.pt`):** `2.08 MB`
* **ONNX FP32 (`lcnn_m1_4s.onnx`):** `2.08 MB` — **EER Subset 2,000 Dev: `0.59%`**
* **ONNX Dynamic INT8 (`lcnn_m1_4s_int8.onnx`):** **`0.56 MB`** (**Kompresi ukuran file 73.0%**) — **EER Subset 2,000 Dev: `0.59%` ($\Delta\text{EER} = 0.000\%$)**
* **Kesimpulan Paritas:** Kuantisasi dinamis INT8 terbukti menjaga urutan skor (*ranking*) dan nilai EER secara identik (`0.59%` vs `0.59%`) pada evaluasi 2,000 sampel Dev Set.

#### B. Latensi Inferensi Per Utterance (4.0s) dan Per Chunk (1.28s)
*Catatan Metodologis:* Pengujian pada input `1.28 detik` (`128 frame`) di bawah ini mengukur **waktu komputasi forward-pass per potongan (*chunk*) statis** (dengan akurasi pasangannya **`M1-Chunk1.28s` Dev EER = `1.33%`**).

| Skenario Input Tensor | Runtime & Presisi | Thread CPU | Mean Latency (ms) | p95 Latency (ms) | Real-Time Factor (RTF) | Temuan Empiris |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Chunk 1.28 detik**<br>*(128 frame LFCC, Dev EER 1.33%)* | **ONNX Runtime FP32** | **1 Core** | **9.62 ms** | **12.34 ms** | **0.0075** | **Paling cepat di CPU x86 non-VNNI** |
| | PyTorch Native FP32 | 1 Core | 17.50 ms | 24.18 ms | 0.0137 | Baseline eager execution |
| | ONNX Dynamic INT8 | 4 Cores | 21.15 ms | 23.70 ms | 0.0165 | Lebih lambat dari FP32 karena overhead quant/dequant |
| | ONNX Dynamic INT8 | 1 Core | 37.89 ms | 43.87 ms | 0.0296 | 3.9x lebih lambat dari ONNX FP32 1-core |
| **Utterance 4.00 detik**<br>*(399 frame LFCC, Dev EER 0.37%, Eval EER 4.59%)* | **ONNX Runtime FP32** | **1 Core** | **22.28 ms** | **26.08 ms** | **0.0056** | **Paling efisien untuk evaluasi 4 detik** |
| | PyTorch Native FP32 | 1 Core | 38.90 ms | 42.90 ms | 0.0097 | Baseline eager execution |
| | ONNX Dynamic INT8 | 4 Cores | 66.12 ms | 78.30 ms | 0.0165 | Overhead konversi dinamis per layer |
| | ONNX Dynamic INT8 | 1 Core | 125.57 ms | 142.99 ms | 0.0314 | 5.6x lebih lambat dari ONNX FP32 1-core |

---

## 5. Checklist Progres Riset (Status: 100% Selesai ✅)

- [x] **Disiplin Protokol & Pemisahan Trek:** Memisahkan evaluasi LA dari PA, menggunakan split resmi ASVspoof 2019 LA tanpa kebocoran data.
- [x] **Single Source of Truth Fitur LFCC:** Implementasi Python deterministik 60-dimensi di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py).
- [x] **Uji Kewarasan Pipeline Data (B0):** Memastikan pipeline label dan ekstraksi fitur berjalan di atas tebakan acak (`32.18%` Dev EER) di [`notebooks/01_baseline_b0_gmm.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/01_baseline_b0_gmm.ipynb).
- [x] **Pelatihan Model LCNN (M1):** Mencapai `0.37%` Dev EER pada serangan *in-distribution* A01–A06 ([`notebooks/02_train_lcnn_m1.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/02_train_lcnn_m1.ipynb)).
- [x] **Evaluasi Headline Unseen Eval Set (A07–A19):** Mencapai **`4.59%` Pooled Eval EER** pada 71,237 sampel, dengan breakdown lengkap per serangan (`0.00%–1.85%` pada 12 serangan, dan identifikasi kelemahan spesifik pada serangan *waveform-filtering VC* **A17: `25.73%`**) di [`results/eval_attack_breakdown.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/eval_attack_breakdown.csv).
- [x] **Studi Ablasi Terkontrol (M2, M3, Silence Trim H6, Chunk 1.28s):** Seluruh eksperimen di [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb) telah selesai dan disimpan di [`results/ablation_study_summary.csv`](file:///home/saf/saf/projects/Vocal-Guard/results/ablation_study_summary.csv).
- [x] **Uji Paritas EER ONNX FP32 vs INT8 & Benchmark Latensi CPU:** Terbukti $\Delta\text{EER} = 0.000\%$ (`0.59%` vs `0.59%` pada 2,000 sampel Dev), kompresi 73% (`0.56 MB`), dan analisis jujur latensi FP32 (`9.62 ms`) vs INT8 (`37.89 ms`).
- [x] **Aplikasi Web Demo Inferensi Per-Chunk (`1.28s`):** Backend FastAPI + Frontend interaktif di [`demo/app.py`](file:///home/saf/saf/projects/Vocal-Guard/demo/app.py) dengan paritas 100% terhadap skrip offline.

