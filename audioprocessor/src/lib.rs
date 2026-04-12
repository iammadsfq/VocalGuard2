mod batch_shell;
mod config;
mod core;
pub mod converter;
// mod stream_shell;  // reserved for real-time path — not yet implemented

// ---------------------------------------------------------------------------
// PyO3 Python module
// Compiled when the crate is built as a `cdylib` (i.e. `cargo build --release`
// with `crate-type = ["cdylib"]` in Cargo.toml).
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub use batch_shell::process_batch;
pub use config::LfccConfig;

/// Python-visible function: `extract_lfcc_tensor(samples, config_kwargs) -> list[float]`
///
/// # Python usage
/// ```python
/// import lfcc_rs
///
/// samples = [0.1, -0.2, ...]          # list[float], normalised PCM
/// tensor  = lfcc_rs.extract_lfcc_tensor(samples)
/// # tensor is a flat list of 3*224*224 = 150_528 float32 values.
/// # Reshape in Python:
/// import numpy as np
/// arr = np.array(tensor, dtype=np.float32).reshape(1, 3, 224, 224)
/// ```
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    samples,
    sample_rate       = 16000,
    pre_emphasis_alpha = 0.97,
    frame_length_ms   = 25.0,
    frame_step_ms     = 10.0,
    num_linear_filters = 40,
    num_cepstral_coeffs = 20,
    fft_size          = 512,
))]
fn extract_lfcc_tensor(
    samples: Vec<f32>,
    sample_rate: usize,
    pre_emphasis_alpha: f32,
    frame_length_ms: f32,
    frame_step_ms: f32,
    num_linear_filters: usize,
    num_cepstral_coeffs: usize,
    fft_size: usize,
) -> PyResult<Vec<f32>> {
    let config = LfccConfig {
        sample_rate,
        pre_emphasis_alpha,
        frame_length_ms,
        frame_step_ms,
        num_linear_filters,
        num_cepstral_coeffs,
        fft_size,
    };

    process_batch(&samples, config)
        .map(|out| out.data)
        .ok_or_else(|| {
            PyValueError::new_err(
                "Audio clip is too short to produce even one LFCC frame. \
                 Provide at least frame_length_ms of audio.",
            )
        })
}

/// The compiled Python module is named `lfcc_rs`.
///
/// Build with:
///   maturin develop          # editable install for development
///   maturin build --release  # wheel for distribution
#[cfg(feature = "python")]
#[pymodule]
fn lfcc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_lfcc_tensor, m)?)?;
    Ok(())
}
