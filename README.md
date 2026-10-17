# Vocal-Guard: Spoofing-Aware Audio Front-End untuk Robot Sosial

Evaluasi ulang secara sistematis dan *re-engineering* sistem deteksi spoofing suara pada benchmark resmi **ASVspoof 2019 Logical Access (LA)** untuk aplikasi antarmuka suara pada robot sosial.

> **Fokus Metodologi:** Proyek ini mendokumentasikan investigasi kegagalan sistem awal (`41.22%` EER) melalui 8 hipotesis terukur (H1–H8), pemisahan protokol LA dan PA, satu sumber kebenaran (*Single Source of Truth*) fitur LFCC 60-dimensi di Python, serta evaluasi jujur antara performa *in-distribution* (Dev Set), generalisasi serangan tak dikenal (Eval Set), dan *trade-off* inferensi CPU menggunakan PyTorch & ONNX Runtime.

---

## 📊 1. Ringkasan Performa Model (ASVspoof 2019 LA)

*Catatan Penting:* Pada ASVspoof 2019 LA, **Dev Set** memuat jenis serangan yang sama dengan Train Set (A01–A06, *in-distribution*), sedangkan **Eval Set** memuat 13 jenis serangan baru yang belum pernah dilihat saat pelatihan (A07–A19, *out-of-distribution*). Angka **Eval EER** menjadi tolok ukur utama (*headline*) generalisasi sistem.

| Kode | Konfigurasi Model | Fitur & Panjang Input | Parameter | Ukuran File | Dev EER (%) *(In-Dist A01–A06)* | Eval EER (%) *(Unseen A07–A19)* | Interpretasi & Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Awal** | MobileNetV2 (Frozen ImageNet + GAP) | LFCC Resize `224x224`, 1s | ~3.50 M | ~14.0 MB | `41.22%` | — | Gagal (mendekati tebakan acak) |
| **B0** | **LFCC-GMM** (256 diag, custom LFCC) | LFCC `(T, 60)`, Full | Statistik | `0.97 MB` | `32.18%` | — | Di atas acak (pipeline data/label tidak rusak, belum setara GMM resmi ~8–9%) |
| **M1** | **LCNN + MFM (FP32)** | LFCC `(1, 60, 399)`, 4s | **539,714** | `2.08 MB` | `0.37%` | *Sedang berjalan* ⏳ | Kuat pada A01–A06; sedang diuji generalisasinya di Eval Set |
| **M1-Q** | **LCNN + MFM (ONNX Dynamic INT8)** | LFCC `(1, 60, 399)`, 4s | **539,714** | **`0.56 MB`** | *Menunggu subset* ⏳ | *Menunggu* ⏳ | Kompresi ukuran file 73.0%; paritas EER sedang dihitung |
| **M3** | **LCNN + GAP (Ablation H2)** | LFCC `(1, 60, 399)`, 4s | 314,434 | `1.20 MB` | *Siap dijalankan* ⏳ | — | Mengisolasi efek *Global Average Pooling* (H2) |
| **M2** | **MobileNetV2 From-Scratch (Ablation H1)** | LFCC `(1, 60, 399)`, 4s | 2,225,858 | `8.50 MB` | *Siap dijalankan* ⏳ | — | Mengisolasi efek arsitektur backbone (H1) |
| **M1-Trim** | **LCNN + MFM + VAD Trim (Ablation H6)** | LFCC `(1, 60, 399)`, 4s Trimmed | 539,714 | `2.08 MB` | *Siap dijalankan* ⏳ | — | Mengisolasi efek *silence shortcut* (Müller dkk., 2021) |
| **M1-1.28s**| **LCNN + MFM (Evaluasi Chunk 1.28s)** | LFCC `(1, 60, 128)`, 1.28s | 539,714 | `2.08 MB` | *Siap dijalankan* ⏳ | — | Mengukur EER riil pada potongan pendek 1.28 detik |

---

## 🧠 2. Diagnosis Kegagalan Awal & Status Hipotesis (H1–H8)

Perubahan dari `41.22%` ke `0.37%` (Dev EER) pada M1 merupakan hasil dari paket perbaikan simultan. Untuk menentukan faktor mana yang paling menentukan secara ilmiah, setiap hipotesis diuji secara terkontrol di [`notebooks/04_ablation_study_m2_m3.ipynb`](file:///home/saf/saf/projects/Vocal-Guard/notebooks/04_ablation_study_m2_m3.ipynb):

* **H1 (Backbone ImageNet Beku):** Diuji dengan melatih MobileNetV2 dari nol pada LFCC (**M2**) dan membandingkannya terhadap **M3** (`LCNN + GAP`). *(Menunggu hasil run ablation)*.
* **H2 (Global Average Pooling Menghapus Transien):** Diuji dengan membandingkan **M1** (`LCNN + AdaptivePool 3x4`) vs **M3** (`LCNN + GAP 1x1`). *(Menunggu hasil run ablation)*.
* **H3 (Distorsi Resize 224x224) & H4 (Rumus LFCC Tanpa Filterbank):** Diperbaiki lewat implementasi referensi tunggal LFCC 60-dimensi resolusi asli `(1, 60, 399)` di [`vocal_guard/features.py`](file:///home/saf/saf/projects/Vocal-Guard/vocal_guard/features.py). *(Catatan: Paritas dengan kode Rust belum dikerjakan; seluruh pipeline saat ini berjalan murni di Python/ONNX).*
* **H6 (Window 1 Detik & *Silence Shortcut*):** Mengacu pada Müller dkk. (2021), durasi hening pada ASVspoof 2019 LA berkorelasi dengan label. Karena M1 dilatih tanpa *silence trimming*, sebagian performa `0.37%` Dev EER berpotensi berasal dari jalan pintas durasi hening. Hal ini diuji secara eksplisit melalui **M1-CrossTrim** dan **M1-Trim**.
* **H7 (Pencampuran Trek LA & PA) & H8 (Validitas Protokol Data):** Trek LA dan PA kini dipisahkan. Baseline **B0 (LFCC-GMM)** menghasilkan `32.18%` Dev EER (di atas tebakan acak 50%, membuktikan pembacaan label tidak terbalik/rusak, meskipun belum setara GMM resmi ~8–9%).

---

## ⚡ 3. Kuantisasi & Benchmark Inferensi Per Chunk di CPU Laptop (Proxy RQ3)

Pengujian dilakukan pada **CPU Laptop Intel Core i7-8665U @ 1.90GHz** sebagai *proxy* komputasi CPU lokal (belum diuji langsung pada prosesor ARM *embedded* seperti Raspberry Pi). Latensi pada skenario `1.28 detik` mengukur waktu *forward-pass* per potongan (*chunk*) statis `128 frame`.

$$\text{RTF} = \frac{\text{Waktu Inferensi CPU (detik)}}{\text{Durasi Audio Input (detik)}}$$

| Skenario Input Tensor | Format & Runtime | Thread CPU | Ukuran File | Latensi Rata-rata | p95 Latensi | RTF | Catatan Empiris |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chunk 1.28 detik**<br>*(128 frame LFCC)* | **ONNX Runtime FP32** | **1 Core** | `2.08 MB` | **9.62 ms** | **12.34 ms** | **0.0075** | **Tercepat di CPU x86 non-VNNI** |
| | PyTorch Native FP32 | 1 Core | `2.08 MB` | 17.50 ms | 24.18 ms | 0.0137 | Baseline PyTorch CPU |
| | ONNX Dynamic INT8 | 4 Cores | **`0.56 MB`** | 21.15 ms | 23.70 ms | 0.0165 | Kompresi ukuran file **73.0%** |
| | ONNX Dynamic INT8 | 1 Core | **`0.56 MB`** | 37.89 ms | 43.87 ms | 0.0296 | Overhead quant/dequant dinamis per layer |
| **Utterance 4.00 detik**<br>*(399 frame LFCC)* | **ONNX Runtime FP32** | **1 Core** | `2.08 MB` | **22.28 ms** | **26.08 ms** | **0.0056** | **Tercepat untuk evaluasi 4 detik** |
| | PyTorch Native FP32 | 1 Core | `2.08 MB` | 38.90 ms | 42.90 ms | 0.0097 | Baseline PyTorch CPU |
| | ONNX Dynamic INT8 | 4 Cores | **`0.56 MB`** | 66.12 ms | 78.30 ms | 0.0165 | Kompresi ukuran file **73.0%** |
| | ONNX Dynamic INT8 | 1 Core | **`0.56 MB`** | 125.57 ms | 142.99 ms | 0.0314 | Overhead quant/dequant dinamis per layer |

> **Diskusi Trade-off INT8 vs FP32:**
> Kuantisasi dinamis INT8 berhasil menekan ukuran model sebesar **73.0%** (`2.08 MB` $\to$ `0.56 MB`), namun memiliki latensi lebih tinggi dibanding ONNX FP32 pada Intel i7-8665U (`37.89 ms` vs `9.62 ms` pada 1 core). Hal ini terjadi karena prosesor uji belum mendukung instruksi *AVX-512 VNNI*, sehingga *dynamic quantization* mengalami *overhead* perhitungan skala aktivasi (*on-the-fly quantization/dequantization*) di setiap layer konvolusi, sedangkan FP32 dieksekusi langsung lewat instruksi vektor *AVX2 FMA*.

---

## 🛠️ 4. Struktur Repositori & Dokumentasi Lengkap

Lihat **[`RESEARCH_DOCUMENTATION.md`](file:///home/saf/saf/projects/Vocal-Guard/RESEARCH_DOCUMENTATION.md)** untuk rincian metodologi, tabel konvergensi epoch, dan *checklist* progres eksperimen.

```text
vocal_guard/
├── features.py          # Single source of truth ekstraksi LFCC 60-dim
├── protocol.py          # Parser resmi protokol ASVspoof 2019 LA
├── metrics.py           # Perhitungan metrik resmi EER dan threshold
├── dataset.py           # PyTorch Dataset (4-detik wrap/truncate)
├── baseline_gmm.py      # Baseline LFCC-GMM
└── models/
    └── lcnn.py          # Light CNN (LCNN) dengan Max-Feature-Map (MFM)
notebooks/
├── 01_baseline_b0_gmm.ipynb         # Eksperimen Baseline B0 (Selesai)
├── 02_train_lcnn_m1.ipynb           # Training LCNN M1 dari nol (Selesai)
├── 03_eval_asvspoof2019_la.ipynb    # Evaluasi Eval Set 71k & Paritas INT8 (Sedang berjalan)
└── 04_ablation_study_m2_m3.ipynb    # Ablation M3, M2, Silence Trim H6, & Chunk 1.28s
scripts/
└── quantize_and_benchmark.py        # Ekspor ONNX, kuantisasi INT8, dan benchmark CPU
results/                             # CSV log hasil eksperimen dan benchmark
models/                              # Bobot PyTorch (.pt), GMM (.pkl), dan ONNX (.onnx)
```
