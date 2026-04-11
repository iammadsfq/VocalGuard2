use crate::config::LfccConfig;
use crate::core::LfccCore;

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// The fully-processed output ready for MobileNetV2.
///
/// Shape: `[1, 3, 224, 224]` stored in row-major order (batch size = 1;
/// the whole utterance is treated as a single image).
///
/// Channel layout:
///   0 (R) — static LFCCs
///   1 (G) — delta  (first temporal derivative)
///   2 (B) — delta-delta (second temporal derivative)
///
/// Each channel is utterance-normalised (zero mean, unit variance) and
/// bilinearly resized from `[n_frames × n_coeffs]` to `[224 × 224]`.
pub struct BatchOutput {
    /// Flat buffer, length = `3 * 224 * 224`.
    /// Index as `data[channel * 224*224 + row * 224 + col]`.
    pub data: Vec<f32>,
    /// Number of LFCC frames extracted from the audio (before resize).
    pub n_frames: usize,
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Process a mono audio clip into a MobileNetV2-ready tensor.
///
/// # Arguments
/// * `samples` — raw PCM samples, normalised to `[-1.0, 1.0]`.
/// * `config`  — LFCC / framing parameters; use `LfccConfig::default()` for
///               the deepfake-detection preset.
///
/// Returns `None` if the audio is too short to produce even one frame.
pub fn process_batch(samples: &[f32], config: LfccConfig) -> Option<BatchOutput> {
    // -----------------------------------------------------------------------
    // STAGE 1 — Pre-emphasis
    // y[t] = x[t] - α * x[t-1]   (first-order high-pass, α ≈ 0.97)
    // Amplifies high-frequency content where deepfake artefacts concentrate.
    // -----------------------------------------------------------------------
    let emphasized = pre_emphasize(samples, config.pre_emphasis_alpha);

    // -----------------------------------------------------------------------
    // STAGE 2 — Framing
    // Slice the signal into overlapping frames of `frame_size_samples` each,
    // advancing by `frame_step_samples` per frame.
    // -----------------------------------------------------------------------
    let mut engine = LfccCore::new(config.clone());
    let frame_size = engine.frame_size_samples;
    let frame_step = engine.frame_step_samples;
    let n_coeffs = config.num_cepstral_coeffs;

    if emphasized.len() < frame_size {
        return None;
    }

    let n_frames = 1 + (emphasized.len() - frame_size) / frame_step;

    // -----------------------------------------------------------------------
    // STAGE 3–6 — Windowing, FFT, Filterbank, Log, DCT
    // Delegated to LfccCore::process_frame, which handles:
    //   • Hann windowing + zero-padding to fft_size
    //   • Real-to-complex FFT (half-spectrum only via realfft)
    //   • Sparse triangular filterbank dot-products
    //   • Log compression  ln(energy + ε)
    //   • DCT matrix-vector multiply → n_coeffs LFCCs per frame
    // Output: row-major matrix  [n_frames × n_coeffs]
    // -----------------------------------------------------------------------
    let mut lfcc_matrix = vec![0.0f32; n_frames * n_coeffs];

    for f in 0..n_frames {
        let start = f * frame_step;
        let frame = &emphasized[start..start + frame_size];
        let row = &mut lfcc_matrix[f * n_coeffs..(f + 1) * n_coeffs];
        engine.process_frame(frame, row);
    }

    // -----------------------------------------------------------------------
    // STAGE 7 — Delta and Delta-Delta coefficients
    // First and second temporal derivatives via the ±2-frame HTK formula.
    // Boundary frames are edge-replicated so all three matrices share the
    // same [n_frames × n_coeffs] shape.
    // -----------------------------------------------------------------------
    let delta = compute_deltas(&lfcc_matrix, n_frames, n_coeffs);
    let delta_delta = compute_deltas(&delta, n_frames, n_coeffs);

    // -----------------------------------------------------------------------
    // STAGE 8 — Utterance-level normalisation
    // Each channel is independently shifted to zero mean, unit variance.
    // Decouples the model from microphone gain and room acoustics.
    // -----------------------------------------------------------------------
    let ch0 = normalize_channel(lfcc_matrix);
    let ch1 = normalize_channel(delta);
    let ch2 = normalize_channel(delta_delta);

    // -----------------------------------------------------------------------
    // STAGE 9 — Resize each channel to 224 × 224
    // Treat [n_frames × n_coeffs] as a grayscale image and bilinearly
    // resample to the fixed spatial size MobileNetV2 expects.
    // -----------------------------------------------------------------------
    const IMG: usize = 224;
    let ch0 = bilinear_resize(&ch0, n_frames, n_coeffs, IMG, IMG);
    let ch1 = bilinear_resize(&ch1, n_frames, n_coeffs, IMG, IMG);
    let ch2 = bilinear_resize(&ch2, n_frames, n_coeffs, IMG, IMG);

    // -----------------------------------------------------------------------
    // STAGE 10 — Pack into [1, 3, 224, 224] tensor  (CHW order)
    // MobileNetV2 expects (Batch, Channels, H, W).
    // -----------------------------------------------------------------------
    let pixels = IMG * IMG;
    let mut data = vec![0.0f32; 3 * pixels];
    data[0 * pixels..1 * pixels].copy_from_slice(&ch0);
    data[1 * pixels..2 * pixels].copy_from_slice(&ch1);
    data[2 * pixels..3 * pixels].copy_from_slice(&ch2);

    Some(BatchOutput { data, n_frames })
}

// ---------------------------------------------------------------------------
// Stage 1 helper — Pre-emphasis
// ---------------------------------------------------------------------------

/// Apply a first-order high-pass filter:  y[t] = x[t] − α·x[t−1].
///
/// The first sample has no predecessor and is passed through unchanged.
fn pre_emphasize(samples: &[f32], alpha: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for i in 1..samples.len() {
        out.push(samples[i] - alpha * samples[i - 1]);
    }
    out
}

// ---------------------------------------------------------------------------
// Stage 7 helper — Delta coefficients  (HTK ±2-frame regression)
// ---------------------------------------------------------------------------

/// Compute first-order temporal derivatives of a coefficient matrix.
///
/// HTK delta formula over a ±N window (N = 2 here):
///
///   Δ[t] = Σ_{n=1}^{N}  n · (c[t+n] − c[t−n])
///           ─────────────────────────────────────
///               2 · Σ_{n=1}^{N}  n²
///
/// Boundary frames are clamped (edge-replicated) so the first and last real
/// frames are not suppressed by zero-padding.
fn compute_deltas(matrix: &[f32], n_frames: usize, n_coeffs: usize) -> Vec<f32> {
    const WIN: i32 = 2;
    // denominator: 2 * (1² + 2²) = 10
    let denom = 2.0 * (1..=WIN).map(|n| (n * n) as f32).sum::<f32>();

    let mut out = vec![0.0f32; n_frames * n_coeffs];

    // Clamp frame index to valid range — edge replication.
    let clamp_row = |f: i32| f.clamp(0, n_frames as i32 - 1) as usize;

    for f in 0..n_frames {
        for c in 0..n_coeffs {
            let mut num = 0.0f32;
            for n in 1..=WIN {
                let ahead = matrix[clamp_row(f as i32 + n) * n_coeffs + c];
                let behind = matrix[clamp_row(f as i32 - n) * n_coeffs + c];
                num += n as f32 * (ahead - behind);
            }
            out[f * n_coeffs + c] = num / denom;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Stage 8 helper — Utterance-level normalisation
// ---------------------------------------------------------------------------

/// Normalise a flat coefficient matrix to zero mean, unit variance (in-place).
///
/// If the standard deviation is below ε (silent segment), the buffer is
/// zeroed rather than dividing by near-zero noise.
fn normalize_channel(mut matrix: Vec<f32>) -> Vec<f32> {
    let n = matrix.len() as f32;
    if n == 0.0 {
        return matrix;
    }

    let mean = matrix.iter().sum::<f32>() / n;
    let variance = matrix.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    if std_dev < f32::EPSILON {
        matrix.fill(0.0);
    } else {
        for x in matrix.iter_mut() {
            *x = (*x - mean) / std_dev;
        }
    }
    matrix
}

// ---------------------------------------------------------------------------
// Stage 9 helper — Bilinear resize
// ---------------------------------------------------------------------------

/// Resize a `src_h × src_w` image (row-major flat slice) to `dst_h × dst_w`
/// using bilinear interpolation.
///
/// Maps the variable-length `[n_frames × n_coeffs]` LFCC matrix to the fixed
/// `224 × 224` grid expected by MobileNetV2.  Works equally for upsampling
/// (short clips) and downsampling (long utterances).
fn bilinear_resize(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_h * dst_w];

    // Scale factors mapping each destination pixel centre back into source space.
    let scale_h = src_h as f32 / dst_h as f32;
    let scale_w = src_w as f32 / dst_w as f32;

    for dy in 0..dst_h {
        // +0.5 / −0.5 centres the sample within the destination pixel.
        let sy = (dy as f32 + 0.5) * scale_h - 0.5;
        let sy0 = (sy.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let fy = (sy - sy.floor()).clamp(0.0, 1.0);

        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_w - 0.5;
            let sx0 = (sx.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let fx = (sx - sx.floor()).clamp(0.0, 1.0);

            // Four-point bilinear blend.
            let v00 = src[sy0 * src_w + sx0];
            let v01 = src[sy0 * src_w + sx1];
            let v10 = src[sy1 * src_w + sx0];
            let v11 = src[sy1 * src_w + sx1];

            let top = v00 * (1.0 - fx) + v01 * fx;
            let bottom = v10 * (1.0 - fx) + v11 * fx;
            dst[dy * dst_w + dx] = top * (1.0 - fy) + bottom * fy;
        }
    }
    dst
}
