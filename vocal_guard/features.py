import numpy as np
import scipy.signal
from scipy.fftpack import dct


def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Apply first-order high-pass filter: y[t] = x[t] - alpha * x[t-1]"""
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def get_linear_filterbank(
    n_filters: int = 20,
    n_fft: int = 512,
    sr: int = 16000,
    low_freq: float = 0.0,
    high_freq: float | None = None,
) -> np.ndarray:
    """
    Construct a linear triangular filterbank.
    Returns: filterbank matrix of shape (n_filters, n_fft // 2 + 1)
    """
    if high_freq is None:
        high_freq = sr / 2.0

    # Linearly spaced frequency points
    freq_points = np.linspace(low_freq, high_freq, n_filters + 2)
    # Convert frequencies to FFT bin indices
    bin_points = np.floor((n_fft + 1) * freq_points / sr).astype(int)

    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_bins))

    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Upward slope
        if center > left:
            for b in range(left, center):
                if b < n_bins:
                    filterbank[i, b] = (b - left) / (center - left)
        # Downward slope
        if right > center:
            for b in range(center, right):
                if b < n_bins:
                    filterbank[i, b] = (right - b) / (right - center)

    return filterbank


def compute_deltas(feats: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute delta (derivative) coefficients along time axis (axis 0).
    feats: (T, D)
    N: window half-width (typically 2 for +/- 2 frames)
    """
    T, D = feats.shape
    deltas = np.zeros_like(feats)
    denom = 2 * sum(n**2 for n in range(1, N + 1))

    for t in range(T):
        numerator = np.zeros(D)
        for n in range(1, N + 1):
            t_plus = min(T - 1, t + n)
            t_minus = max(0, t - n)
            numerator += n * (feats[t_plus] - feats[t_minus])
        deltas[t] = numerator / denom

    return deltas


def extract_lfcc(
    y: np.ndarray,
    sr: int = 16000,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    num_filters: int = 20,
    num_coeffs: int = 20,
    pre_emph: float = 0.97,
    with_deltas: bool = True,
) -> np.ndarray:
    """
    Single Source of Truth reference implementation of LFCC extraction:
    1. Pre-emphasis
    2. Framing & Hann windowing
    3. Real FFT & magnitude spectrum
    4. 20 Linear triangular filterbank
    5. Logarithmic compression: ln(energy + eps)
    6. Discrete Cosine Transform (ortho)
    7. Delta and Delta-Delta coefficients (optional)

    Returns:
        feats: shape (T, 60) if with_deltas is True, else (T, num_coeffs)
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 1. Pre-emphasis
    y = pre_emphasis(y, alpha=pre_emph)

    # 2. Framing
    frame_len = int(sr * window_ms / 1000.0)  # e.g., 320 samples @ 16kHz
    hop_len = int(sr * hop_ms / 1000.0)      # e.g., 160 samples @ 16kHz

    if len(y) < frame_len:
        # Pad with zero if shorter than a single frame
        y = np.pad(y, (0, frame_len - len(y)))

    num_frames = 1 + (len(y) - frame_len) // hop_len
    # Extract frames
    shape = (num_frames, frame_len)
    strides = (y.strides[0] * hop_len, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()

    # Apply Hann window
    window = np.hanning(frame_len)
    frames *= window

    # 3. FFT (zero padded to n_fft)
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    mag_spec = np.abs(spec)  # (num_frames, n_fft // 2 + 1)

    # 4. Linear Filterbank
    fbank = get_linear_filterbank(n_filters=num_filters, n_fft=n_fft, sr=sr)
    # Energy in each linear filterbank channel
    filter_energies = np.dot(mag_spec, fbank.T)  # (num_frames, num_filters)

    # 5. Logarithmic compression
    log_energies = np.log(filter_energies + 1e-12)

    # 6. DCT (type 2, ortho normalization)
    cepstral_coeffs = dct(log_energies, type=2, axis=-1, norm="ortho")[:, :num_coeffs]

    if not with_deltas:
        return cepstral_coeffs

    # 7. Deltas & Delta-Deltas
    deltas = compute_deltas(cepstral_coeffs, N=2)
    delta_deltas = compute_deltas(deltas, N=2)

    # Stack into (T, 60): static (20) + delta (20) + delta-delta (20)
    return np.hstack([cepstral_coeffs, deltas, delta_deltas])
