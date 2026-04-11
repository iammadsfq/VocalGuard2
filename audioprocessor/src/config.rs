#[derive(Debug, Clone)]
pub struct LfccConfig {
    pub sample_rate: usize,
    pub pre_emphasis_alpha: f32,
    pub frame_length_ms: f32,
    pub frame_step_ms: f32,
    pub num_linear_filters: usize,
    pub num_cepstral_coeffs: usize,
    pub fft_size: usize,
}

impl Default for LfccConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            pre_emphasis_alpha: 0.97,
            frame_length_ms: 25.0,
            frame_step_ms: 10.0,
            num_linear_filters: 40,
            num_cepstral_coeffs: 20,
            fft_size: 512
        }
    }
}
