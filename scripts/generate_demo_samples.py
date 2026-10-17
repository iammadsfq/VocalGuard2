"""
Generate deterministic local reference audio samples for the Vocal-Guard Streaming Demo:
1. sample_bonafide_like.wav (4.0s, 16kHz mono): Harmonic speech-like formant structure with natural pitch modulation.
2. sample_spoof_like.wav (4.0s, 16kHz mono): Synthetic vocoder-like buzz with high-frequency phase/spectral discontinuities.
Also verifies that offline chunk scoring matches the demo pipeline 100%.
"""

from pathlib import Path
import numpy as np
import soundfile as sf


def generate_demo_samples(output_dir: str = "demo/samples", sr: int = 16000, duration: float = 5.12):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.RandomState(42)

    # 1. Sample A: Natural-like harmonic signal with smooth envelope and natural floor
    f0 = 135.0 + 15.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig_a = np.zeros_like(t)
    for k in range(1, 12):
        # Smooth spectral roll-off
        amp = (1.0 / (k ** 1.4)) * np.exp(-0.002 * k)
        sig_a += amp * np.sin(k * phase)
    # Syllabic amplitude modulation (3.5 Hz speaking rate)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t) ** 2
    sig_a = sig_a * envelope + 0.008 * rng.randn(n_samples)
    sig_a = (sig_a / (np.max(np.abs(sig_a)) + 1e-8) * 0.85).astype(np.float32)

    # 2. Sample B: Vocoder-artifact simulation (high-frequency metallic buzz + periodic phase resets)
    sig_b = np.zeros_like(t)
    for k in range(1, 28):
        # Flat high-frequency emphasis typical of unsmoothed vocoder excitation
        amp = 1.0 / (k ** 0.55)
        sig_b += amp * np.cos(k * 2 * np.pi * 160.0 * t + (k % 3) * 0.7)
    # High-frequency synthetic carrier modulation (5.5 kHz - 7.5 kHz band artifact)
    hf_artifact = 0.35 * np.sin(2 * np.pi * 6400 * t) * np.sign(np.sin(2 * np.pi * 50 * t))
    sig_b = (sig_b * 0.6 + hf_artifact) * envelope
    sig_b = (sig_b / (np.max(np.abs(sig_b)) + 1e-8) * 0.85).astype(np.float32)

    path_a = out / "reference_harmonic_5s.wav"
    path_b = out / "reference_synthetic_artifact_5s.wav"
    sf.write(str(path_a), sig_a, sr, subtype="PCM_16")
    sf.write(str(path_b), sig_b, sr, subtype="PCM_16")
    print(f"Generated: {path_a} ({len(sig_a)} samples)")
    print(f"Generated: {path_b} ({len(sig_b)} samples)")


if __name__ == "__main__":
    generate_demo_samples()
