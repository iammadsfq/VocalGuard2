import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import soundfile as sf
import os
from scipy.fftpack import dct
from model import build_deepfake_detector

def extract_lfcc_3channel(audio_path, sr=16000, n_lfcc=40):
    try:
        y, native_sr = sf.read(audio_path)
        
        if len(y.shape) > 1:
            y = y.mean(axis=1)
            
        if native_sr != sr:
            y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)

        y = librosa.effects.preemphasis(y, coef=0.97)

        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
        lfcc = dct(stft, axis=0, type=2, norm='ortho')[:n_lfcc]

        delta = librosa.feature.delta(lfcc)
        delta2 = librosa.feature.delta(delta)

        features = np.stack([lfcc, delta, delta2], axis=-1)
        
        features_resized = tf.image.resize(features, [224, 224]).numpy()
        
        mean = np.mean(features_resized)
        std = np.std(features_resized)
        return (features_resized - mean) / (std + 1e-8)
    
    except Exception as e:
        print(f"Gagal memproses {audio_path}: {e}")
        return None

def train_task(task_name, csv_path, audio_dir):
    print(f"\n--- Memulai Training untuk Task: {task_name.upper()} ---")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV tidak ditemukan di {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Kolom ditemukan: {df.columns.tolist()}")

    file_col = 'id'
    label_col = 'label'

    X, y = [], []
    samples = df.head(4000) 

    for _, row in samples.iterrows():
        file_name = str(row[file_col])
        path = os.path.join(audio_dir, f"{file_name}.flac")

        if os.path.exists(path):
            feat = extract_lfcc_3channel(path)
            if feat is not None:
                X.append(feat)
                y.append(int(row[label_col]))
        else:
            path_wav = path.replace('.flac', '.wav')
            if os.path.exists(path_wav):
                feat = extract_lfcc_3channel(path_wav)
                if feat is not None:
                    X.append(feat)
                    y.append(int(row[label_col]))

    if not X:
        print(f"Error: Tidak ada audio yang berhasil dimuat dari {audio_dir}")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Data dimuat: {len(X)} sampel. Memulai training...")

    model = build_deepfake_detector()
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
    
    model_name = f"model_{task_name}.h5"
    model.save(model_name)
    print(f"Berhasil! Model disimpan sebagai {model_name}")

if __name__ == "__main__":
    train_task('la', './flac_sample/flac_sample/LA_sample_labels.csv', './flac_sample/flac_sample/LA/')
    train_task('pa', './flac_sample/flac_sample/PA_sample_labels.csv', './flac_sample/flac_sample/PA/')