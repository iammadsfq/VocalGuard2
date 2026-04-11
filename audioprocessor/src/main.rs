use batch_shell::process_batch;
use config::LfccConfig;

// ---------------------------------------------------------------------------
// Native binary entry point
// Used for local smoke-testing without Python.  Run with:
//   cargo run --features native
// ---------------------------------------------------------------------------

#[cfg(not(feature = "python"))]
fn main() {
    use std::time::Instant;

    // Synthetic 1-second test signal at 16 kHz: 440 Hz sine + white noise.
    let sample_rate = 16000;
    let duration_s = 1.0_f32;
    let n_samples = (sample_rate as f32 * duration_s) as usize;

    let samples: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // 440 Hz tone scaled to [-0.5, 0.5] — well within the [-1, 1] range
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    let config = LfccConfig::default();

    println!(
        "Input: {} samples @ {} Hz",
        samples.len(),
        config.sample_rate
    );
    println!(
        "Config: {} filters, {} coeffs, FFT size {}",
        config.num_linear_filters, config.num_cepstral_coeffs, config.fft_size
    );

    let t0 = Instant::now();
    match process_batch(&samples, config) {
        Some(out) => {
            let elapsed = t0.elapsed();
            println!(
                "OK — {} LFCC frames → tensor [{} × {} × {} × {}] ({} f32 values) in {:.2?}",
                out.n_frames,
                1,
                3,
                224,
                224,
                out.data.len(),
                elapsed,
            );

            // Sanity-check: tensor should be exactly 3*224*224 elements.
            assert_eq!(out.data.len(), 3 * 224 * 224, "unexpected tensor size");

            // Spot-check: all values should be finite after normalisation.
            let all_finite = out.data.iter().all(|v| v.is_finite());
            println!("All values finite: {}", all_finite);

            // Print a small window of channel-0 pixel values.
            println!("ch0 [0..8]: {:?}", &out.data[..8]);
        }
        None => eprintln!("Error: audio too short for even one frame."),
    }
}
