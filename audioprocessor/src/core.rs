use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::{f32::consts::PI, sync::Arc, vec};

use crate::config::LfccConfig;

const EPSILON: f32 = f32::EPSILON;

/// A single triangular filter stored as a contiguous slice of non-zero weights
/// and the FFT bin index where that slice begins.
///
/// For a linear filterbank the non-zero span is exactly
/// `[left_bin, right_bin]` — roughly 2/num_filters of all bins.
/// Skipping the zero-padded wings halves filterbank arithmetic and
/// keeps each filter's weights in a handful of cache lines.
struct SparseFilter {
    /// Index of the first non-zero FFT bin.
    start_bin: usize,
    /// Non-zero weights, covering bins `start_bin .. start_bin + weights.len()`.
    weights: Vec<f32>,
}

pub struct LfccCore {
    pub config: LfccConfig,
    pub frame_size_samples: usize,
    pub frame_step_samples: usize,

    hann_window: Vec<f32>,

    // realfft: operates on real f32 input, produces N/2+1 complex bins directly.
    // Replaces the full complex FFT which required a Complex32 input buffer and
    // produced N bins — the upper N/2-1 of which are conjugate mirrors we never used.
    fft_instance: Arc<dyn RealToComplex<f32>>,
    real_input: Vec<f32>,        // length: fft_size  (windowed + zero-padded)
    spectrum: Vec<Complex32>,    // length: fft_size/2 + 1  (the half-spectrum)
    fft_scratch: Vec<Complex32>, // internal scratch required by realfft
    power_spectrum: Vec<f32>,    // length: fft_size/2 + 1

    // Sparse triangular filters — each stores only its non-zero bin range.
    // Replaces the dense Array2 [num_filters × num_bins] to skip zero-weight bins.
    filterbank: Vec<SparseFilter>,
    // [num_cepstral_coeffs × num_linear_filters]
    dct_matrix: Array2<f32>,

    // pre-allocated scratch so process_frame never allocates
    filterbank_energies: Array1<f32>, // length: num_linear_filters
    lfcc_scratch: Array1<f32>,        // length: num_cepstral_coeffs
}

impl LfccCore {
    pub fn new(config: LfccConfig) -> Self {
        let frame_size_samples =
            ((config.sample_rate as f32) * (config.frame_length_ms / 1000.0)) as usize;
        let frame_step_samples =
            ((config.sample_rate as f32) * (config.frame_step_ms / 1000.0)) as usize;

        let mut hann_window = Vec::with_capacity(frame_size_samples);
        for n in 0..frame_size_samples {
            let n_f32 = n as f32;
            let n_total_minus_one = (frame_size_samples - 1) as f32;

            // w[n] = 0.5 - 0.5 * cos( (2 * pi * n) / (N - 1) )
            let window_value = 0.5 - 0.5 * f32::cos((2.0 * PI * n_f32) / (n_total_minus_one));
            hann_window.push(window_value);
        }

        // realfft plans are created from RealFftPlanner.
        // make_input_vec() / make_output_vec() allocate exactly the right sizes
        // (fft_size and fft_size/2+1 respectively) so there's no manual arithmetic.
        let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
        let fft_instance = planner.plan_fft_forward(config.fft_size);
        let real_input = fft_instance.make_input_vec(); // len == fft_size
        let spectrum = fft_instance.make_output_vec(); // len == fft_size/2 + 1
        let fft_scratch = fft_instance.make_scratch_vec();
        let power_spectrum = vec![0.0f32; spectrum.len()];

        let num_bins = (config.fft_size / 2) + 1;
        let filterbank = Self::generate_sparse_filterbank(
            config.num_linear_filters,
            num_bins,
            config.sample_rate,
        );
        let dct_matrix =
            Self::generate_dct_matrix(config.num_cepstral_coeffs, config.num_linear_filters);

        let filterbank_energies = Array1::zeros(config.num_linear_filters);
        let lfcc_scratch = Array1::zeros(config.num_cepstral_coeffs);

        Self {
            config,
            frame_size_samples,
            frame_step_samples,
            hann_window,
            fft_instance,
            real_input,
            spectrum,
            fft_scratch,
            power_spectrum,
            filterbank,
            dct_matrix,
            filterbank_energies,
            lfcc_scratch,
        }
    }

    #[inline]
    pub fn process_frame(&mut self, pre_emphasized_frame: &[f32], output_lfccs: &mut [f32]) {
        // Windowing — write directly into the f32 real_input buffer.
        // realfft takes &mut [f32], so there's no Complex32 wrapper to unpack.
        for ((sample, &signal), &window) in self.real_input[..self.frame_size_samples]
            .iter_mut()
            .zip(pre_emphasized_frame.iter())
            .zip(self.hann_window.iter())
        {
            *sample = signal * window;
        }

        // Zero-pad the remainder of the input window.
        self.real_input[self.frame_size_samples..].fill(0.0);

        // Real-to-complex FFT: consumes real_input, produces spectrum[0..N/2+1].
        // This is ~2× faster than a full complex FFT because it exploits the
        // conjugate symmetry of a real signal — the upper N/2-1 mirror bins are
        // never computed or stored.
        self.fft_instance
            .process_with_scratch(
                &mut self.real_input,
                &mut self.spectrum,
                &mut self.fft_scratch,
            )
            .expect("realfft: buffer length mismatch — should be impossible");

        // Power spectrum: |X[k]|² for each of the N/2+1 unique bins.
        for (power, complex_val) in self.power_spectrum.iter_mut().zip(self.spectrum.iter()) {
            *power = complex_val.norm_sqr();
        }

        // STEP 4 — sparse filterbank dot-products.
        // Each filter touches only its non-zero bin range, skipping the zero wings.
        for (energy, filter) in self
            .filterbank_energies
            .iter_mut()
            .zip(self.filterbank.iter())
        {
            let power_slice =
                &self.power_spectrum[filter.start_bin..filter.start_bin + filter.weights.len()];
            *energy = power_slice
                .iter()
                .zip(filter.weights.iter())
                .fold(0.0f32, |acc, (&p, &w)| acc + p * w);
        }

        // STEP 5 — log of each filterbank energy (in-place)
        self.filterbank_energies
            .mapv_inplace(|e| (e + EPSILON).ln());

        // STEP 6 — DCT mat-vec: lfccs = dct_matrix · log_energies
        // dct_matrix is [num_cepstral_coeffs × num_filters]
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            &self.dct_matrix,
            &self.filterbank_energies,
            0.0,
            &mut self.lfcc_scratch,
        );
        let n = output_lfccs.len().min(self.lfcc_scratch.len());
        output_lfccs[..n].copy_from_slice(&self.lfcc_scratch.as_slice().unwrap()[..n]);
    }

    /// Build a linear filterbank as sparse filters.
    ///
    /// For each triangular filter the non-zero span runs from the left anchor
    /// bin to the right anchor bin (inclusive).  Only those weights are stored,
    /// so the energy dot-product skips all zero-weight bins automatically.
    fn generate_sparse_filterbank(
        num_filters: usize,
        num_bins: usize,
        sample_rate: usize,
    ) -> Vec<SparseFilter> {
        let nyquist = (sample_rate as f32) / 2.0;
        let hz_per_bin = nyquist / ((num_bins - 1) as f32);
        let num_anchors = num_filters + 2;
        let hz_step = nyquist / ((num_anchors - 1) as f32);
        let anchor_hz: Vec<f32> = (0..num_anchors).map(|i| (i as f32) * hz_step).collect();

        // Convert a frequency in Hz to the nearest FFT bin index, clamped to
        // [0, num_bins).  Using floor here matches the dense implementation.
        let hz_to_bin =
            |hz: f32| -> usize { ((hz / hz_per_bin).floor() as usize).min(num_bins - 1) };

        let mut filterbank = Vec::with_capacity(num_filters);

        for i in 0..num_filters {
            let left_hz = anchor_hz[i];
            let center_hz = anchor_hz[i + 1];
            let right_hz = anchor_hz[i + 2];

            // Clamp the active span to actual bin indices.
            let start_bin = hz_to_bin(left_hz);
            let end_bin = hz_to_bin(right_hz); // inclusive

            let span = end_bin + 1 - start_bin;
            let mut weights = Vec::with_capacity(span);

            for bin in start_bin..=end_bin {
                let bin_hz = (bin as f32) * hz_per_bin;
                let w = if bin_hz >= left_hz && bin_hz <= center_hz {
                    (bin_hz - left_hz) / (center_hz - left_hz)
                } else if bin_hz > center_hz && bin_hz <= right_hz {
                    (right_hz - bin_hz) / (right_hz - center_hz)
                } else {
                    0.0
                };
                weights.push(w);
            }

            // Trim any leading/trailing zeros that fell outside the triangle
            // due to bin-boundary rounding.
            let first_nz = weights.iter().position(|&w| w != 0.0).unwrap_or(0);
            let last_nz = weights.iter().rposition(|&w| w != 0.0).unwrap_or(0);
            let trimmed = weights[first_nz..=last_nz].to_vec();

            filterbank.push(SparseFilter {
                start_bin: start_bin + first_nz,
                weights: trimmed,
            });
        }

        filterbank
    }

    fn generate_dct_matrix(num_coeffs: usize, num_filters: usize) -> Array2<f32> {
        let mut matrix = Array2::zeros((num_coeffs, num_filters));

        let factor_0 = (1.0 / (num_filters as f32)).sqrt();
        let factor_k = (2.0 / (num_filters as f32)).sqrt();

        for k in 0..num_coeffs {
            let normalization = if k == 0 { factor_0 } else { factor_k };
            let k_f32 = k as f32;
            let num_filters_f32 = num_filters as f32;
            let mut row = matrix.slice_mut(s![k, ..]);

            for n in 0..num_filters {
                // Cosine math: cos ( (pi / N) * (n + 0.5) * k )
                row[n] = normalization * f32::cos(PI * k_f32 * (n as f32 + 0.5) / num_filters_f32);
            }
        }
        matrix
    }
}
