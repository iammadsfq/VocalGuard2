# Spectrogram Pipeline

This folder contains a conversion pipeline to turn ASVspoof FLAC audio into
saved 3-channel LFCC tensors with shape `[3, 224, 224]`, written one file per
utterance as `<ID>.parquet` under `data/spectrogram/`.

Important: extraction is done by the Rust module (`lfcc_rs`) from
`audioprocessor`, so training features and production features use the same
core implementation.

Channel layout:
- channel 0: static LFCC
- channel 1: delta LFCC
- channel 2: delta-delta LFCC

Output format:
- feature tensor per utterance: `<ID>.parquet`
- dataset index: `manifest.csv`
- optional failures: `failures.csv`

## 1) Install dependencies

Run from project root:

```powershell
pip install -r ml/requirements.txt
```

## 2) Build Rust Python module (same extractor as production)

Run from project root:

```powershell
maturin develop --manifest-path audioprocessor/Cargo.toml --features python
```

This builds and installs `lfcc_rs` into your active Python environment.

## 3) Quick test with 50 samples

Run from project root:

```powershell
python ml/build_spectrogram_dataset.py --corpus both --limit 50 --output-root data/spectrogram
```

This will create:
- `data/spectrogram/<ID>.parquet`
- `data/spectrogram/manifest.csv`

## 4) Run full conversion

After the sample run looks good:

```powershell
python ml/build_spectrogram_dataset.py --corpus both --limit 0 --output-root data/spectrogram
```

## 5) Training input

Use `manifest.csv` as the source of truth for training:
- `parquet_path` points to each saved tensor file
- `label_id` is already numeric (`bonafide=1`, `spoof=0`)

Load one sample in Python:

```python
import pyarrow.parquet as pq

table = pq.read_table("data/spectrogram/LA_E_9332881.parquet")
flat = table.column("spectrogram")[0].as_py()
id_value = table.column("id")[0].as_py()
```
