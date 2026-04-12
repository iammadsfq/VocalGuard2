import numpy as np
import librosa
import tensorflow as tf
from scipy.fftpack import dct
from model import build_deepfake_detector

def extract_lfcc_3channel(audio_path, sr=16000, n_lfcc=40):
    y, _ = librosa.load(audio_path, sr=sr)
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

def train():
    model = build_deepfake_detector()
    # X_train = np.array([extract_lfcc_3channel(p) for p in paths])
    print("Siap memulai training dengan input shape 224x224x3...")

if __name__ == "__main__":
    train()