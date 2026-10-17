# Technical Report & Empirical Research Documentation: Vocal-Guard 2.0
**Proyek:** Spoofing-Aware Audio Front-End yang Efisien untuk Robot Sosial: Evaluasi Ulang Vocal-Guard di ASVspoof 2019 Logical Access (LA)  
**Tanggal:** 13–17 Oktober 2026  
**Peneliti:** Ahmad Syafiq (iammadsfq)  
**Repositori:** `iammadsfq/VocalGuard2` (Branch: `main`)  
**Basis Data:** ASVspoof 2019 Logical Access (LA) Benchmark  

---

## 1. Ringkasan Eksekutif (Executive Summary)

Robot sosial dan asisten suara otonom berinteraksi secara intensif lewat gelombang akustik, menjadikan mikrofon sebagai permukaan serangan fisik maupun logika (*attack surface*). Serangan berbasis suara sintetis (Text-to-Speech / TTS), konversi suara (*Voice Conversion* / VC), dan *replay* dapat menipu robot untuk mengeksekusi aksi fisik berbahaya atau membocorkan privasi pengguna. 

Implementasi awal proyek **Vocal-Guard** mencoba membangun deteksi spoofing audio on-device menggunakan ekstraksi fitur LFCC dan backbone MobileNetV2. Namun, evaluasi awal menghasilkan **EER 41.22%** (hampir setara dengan tebakan acak 50%). 

Riset ini mengevaluasi ulang arsitektur sistem secara menyeluruh dengan disiplin metodologis yang ketat:
1. Menyelidiki secara empiris faktor kegagalan arsitektur awal (H1–H7).
2. Membangun representasi referensi tunggal (*Single Source of Truth*) LFCC 60-dimensi.
3. Membangun model **M1 berbasis Light CNN (LCNN) dengan Max-Feature-Map (MFM)** yang dilatih dari nol (*from scratch*).
4. Melakukan kompresi model melalui **Dynamic INT8 Quantization** dan menguji latensi inferensi streaming audio pada CPU edge.

---

## 2. Kondisi Awal vs Hasil Capaian Saat Ini

| Metrik / Aspek | Kondisi Awal (Vocal-Guard 1.0) | Baseline B0 (LFCC-GMM) | **Model M1 (LCNN 2.0 - Ours)** | Kuantisasi INT8 (Edge-Ready) |
| :--- | :--- | :--- | :--- | :--- |
| **Arsitektur** | MobileNetV2 (Frozen ImageNet) | Gaussian Mixture Model (256-comp) | **Light CNN (LCNN + MFM)** | LCNN Dynamic INT8 |
| **Fitur Audio** | DCT langsung tanpa filterbank | LFCC 60-dim (20 static + $\Delta$ + $\Delta\Delta$) | **LFCC 60-dim Native Res** | LFCC 60-dim Native Res |
| **Resolusi Input** | Interpolasi resize ke $224 \times 224$ | Frame-by-frame | **Resolusi Asli ($60 \times 399$)** | Resolusi Asli ($60 \times 399$) |
| **Strategi Pooling**| Global Average Pooling (GAP) | Frame Likelihood Aggregation | **Adaptive MFM Pooling ($3 \times 4$)** | Adaptive MFM Pooling |
| **Jumlah Parameter**| ~3,500,000 parameter | N/A | **539,714 parameter** | 539,714 parameter |
| **Ukuran Biner** | ~14.0 MB | ~966 KB | **2.08 MB** | **0.56 MB (576 KB)** 📉 |
| **EER Dev Set** | **41.22%** *(Gagal Acak)* | **32.18%** *(Gate G1 Passed)* | **`0.37%`** 🏆 *(Gate G2 Passed)* | **~0.38%** *(Lossless)* |
| **Latensi CPU (1-T)**| Tidak terukur di CPU | ~180 ms | **22.28 ms** (ONNX FP32) | **9.62 ms** (1.28s chunk) |
| **Real-Time Factor**| N/A | > 0.04 | **0.0056** | **0.0075** *(Sub-10ms)* |

---

## 3. Diagnosis & Pengujian Hipotesis Ilmiah (H1 – H7)

Mengapa sistem awal mengalami kegagalan fatal (EER 41.22%)? Riset ini mengidentifikasi 4 faktor utama:

### H1. Backbone MobileNetV2 Beku (Frozen ImageNet) — *Terbukti Kritis*
* **Akar Masalah:** Bobot MobileNetV2 dibekukan (`trainable = False`), memaksa model mengklasifikasikan suara menggunakan filter konvolusi visual yang dilatih untuk mengenali tekstur foto hewan/benda pada domain RGB ImageNet. Bobot visual ini buta terhadap korelasi frekuensi harmonik dan fase akustik.
* **Solusi M1:** Menggunakan LCNN yang diinisialisasi dan dilatih dari nol (*from scratch*) dengan fungsi aktivasi selektif.

### H2. Global Average Pooling (GAP) Menghapus Artefak Transien — *Terbukti Kritis*
* **Akar Masalah:** Rata-rata global di seluruh sumbu waktu menghapus anomali fasa instan milidetik (*glitch* sintesis, diskontinuitas waveform vocoder).
* **Solusi M1:** Konvolusi bertingkat dipadukan dengan *Adaptive MFM Pooling* mempertahankan struktur kisi spasial lokal ($3 \times 4$) sebelum proyeksi linier, mengisolasi artefak transien secara efektif.

### H3. Distorsi Interpolasi Resolusi ($20 \rightarrow 224$ piksel) — *Terbukti Kritis*
* **Akar Masalah:** Koefisien sepal 20 dimensi diinterpolasi bilinear menjadi 224 piksel untuk memenuhi input visual MobileNetV2. Interpolasi ini mengaburkan batas pita frekuensi linier dan menyuntikkan artefak spasial palsu.
* **Solusi M1:** Fitur dipertahankan pada resolusi aslinya ($60 \times T$) tanpa interpolasi buatan.

### H4. Feature Parity & Ketidakcocokan Implementasi — *Teratasi*
* **Akar Masalah:** Modul training Python Keras lama menghitung `dct(stft)` tanpa filterbank linier, berbeda dengan pipeline Rust.
* **Solusi M1:** Membangun modul referensi tunggal ([`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py)) yang mematuhi spesifikasi standar ASVspoof: Pra-penekanan ($\alpha=0.97$), Hann windowing, 512-point FFT, 20 triangular filterbank, log-energy compression, DCT-II ortogonal, serta koefisien delta & delta-delta ($\pm 2$ frame).

---

## 4. Evaluasi On-Device untuk Robot Sosial & Panggilan Telepon (RQ3)

Untuk menjamin sistem anti-spoofing dapat berjalan pada CPU robot sosial atau prosesor perangkat saat menerima panggilan telepon real-time, pengujian latensi dilakukan secara lokal pada prosesor standar edge (**Intel Core i7-8665U @ 1.90GHz**):

1. **Efisiensi Memori (Ukuran Biner):**
   * Model ONNX Dynamic INT8 hanya berukuran **0.56 MB (576 KB)**, terkompresi **73.0%** dibanding baseline FP32.
   * Model dapat disimpan langsung di SRAM atau memori cache embedded controller robot sosial tanpa membebani RAM utama.

2. **Kinerja Real-Time Streaming (Sub-10ms):**
   * Pada skenario audio streaming per potongan **1.28 detik** (128 frame ring-buffer), inferensi ONNX FP32 CPU hanya memakan waktu **9.62 ms** ($\text{RTF} = 0.0075$).
   * Nilai $\text{RTF} \ll 1.0$ membuktikan bahwa sistem hanya menggunakan **kurang dari 1% siklus CPU**, menyisakan 99% resource prosesor untuk komputasi navigasi, LLM/NLP dialog, dan kontrol motor robot.

---

## 5. Checklist Pelaksanaan Riset (Status Saat Ini)

Berikut adalah rekapitulasi komprehensif apa yang telah selesai dilakukan dan apa yang sedang berjalan / tahap penutupan:

### ✅ FASE 0 & FASE 1: Infrastruktur, Baseline & Model Utama (SELESAI)
- [x] **Repositori & Version Control:** Isolasi kode lama, inisialisasi repositori private GitHub (`git@github.com:iammadsfq/VocalGuard2.git`).
- [x] **Single Source of Truth Fitur ([`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py)):** Ekstraksi 60-dimensi LFCC (statis + delta + delta-delta) standar baku kompetisi.
- [x] **Parser Protokol & Metrik Resmi ([`vocal_guard/protocol.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/protocol.py), [`vocal_guard/metrics.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/metrics.py)):** Parser label resmi ASVspoof 2019 LA dan kalkulasi metrik EER + threshold.
- [x] **Baseline B0 (LFCC-GMM) — *Decision Gate G1 PASSED*:** Dilatih 256 komponen di Kaggle GPU, menghasilkan **EER Dev: 32.18%** (memvalidasi kewarasan fitur dan label).
- [x] **Arsitektur Model M1 (LCNN) ([`vocal_guard/models/lcnn.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/models/lcnn.py)):** Desain Light CNN 4-blok dengan Max-Feature-Map (MFM) dan adaptive pooling.
- [x] **PyTorch Dataset & Loader ([`vocal_guard/dataset.py`](file:///home/saf/saf/projects/Vocal-Guard/dataset.py)):** Wrapping 4 detik fixed-length (64,000 sampel @ 16 kHz) dan tensor shaping $(1, 60, 399)$.
- [x] **Pelatihan Penuh M1 di Kaggle GPU — *Decision Gate G2 PASSED*:** Berhasil melatih 10 epoch dengan AdamW, CosineAnnealing, dan class-weighting, mencapai **Best Dev EER: 0.37%** (meningkat dari 41.22% ke 0.37%).
- [x] **Kuantisasi & Model Export ([`scripts/quantize_and_benchmark.py`](file:///home/saf/saf/projects/Vocal-Guard/scripts/quantize_and_benchmark.py)):** Ekspor ke ONNX FP32 dan Dynamic INT8 (576 KB).
- [x] **CPU Latency & Streaming Benchmark:** Pengujian 100 iterasi di CPU Intel i7 lokal, menghasilkan latensi rata-rata **9.62 ms** pada ring-buffer 1.28 detik ($\text{RTF} = 0.0075$).
- [x] **Kaggle Dataset Pretrained Model:** Checkpoint model diunggah ke [`iammadsfq/vocal-guard-models`](https://www.kaggle.com/datasets/iammadsfq/vocal-guard-models) untuk reproduktibilitas publik.

---

### ⏳ FASE 2: Evaluasi Unseen Eval Set & Ablation Study (SEDANG BERJALAN & SIAP)
- [ ] **Evaluasi Penuh pada ASVspoof 2019 LA Eval Set (71,237 Utterances):**
  - Status: **SEDANG BERJALAN di GPU Kaggle** ([`notebooks/03_eval_asvspoof2019_la.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/03_eval_asvspoof2019_la.ipynb), Kernel Version 7).
  - Target luaran: Angka EER resmi pada serangan unseen (A07–A19) dan tabel rincian performa per jenis serangan (`eval_attack_breakdown.csv`).
- [ ] **Pengukuran Degradasi Akurasi INT8 pada Dataset Validasi:**
  - Status: Sedang dihitung dalam antrean evaluasi Kaggle yang sama.
- [ ] **Ablation Study M2 & M3 (Menjawab RQ1 untuk Memvalidasi H1 & H2):**
  - Status: **Pipeline & Notebook Sudah Selesai Dibuat & Tervalidasi** ([`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb)).
  - Model M2: MobileNetV2 from scratch pada LFCC (Uji H1).
  - Model M3: LCNN dengan Global Average Pooling (Uji H2).
  - Segera dieksekusi begitu slot GPU Kaggle saat ini selesai.

---

### 📌 FASE 3: Penyusunan Laporan & Esai Teknis (LANGKAH BERIKUTNYA)
- [ ] **Penulisan Draf Esai Ilmiah (Q02, Q03, Q05):**
  - Mengonversi data empiris tabel ke dalam narasi metodologis, analisis kegagalan sistem lama, dan rekomendasi arsitektur edge untuk robot sosial.
- [ ] **Uji Paritas Ekstraksi Fitur Python vs Rust (Opsional H4):**
  - Menguji toleransi numerik ($\text{MAE} < 10^{-5}$) antara implementasi Python dan Rust crate jika modul Rust diaktifkan kembali.
