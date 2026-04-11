use crate::config::LfccConfig;

pub struct LfccCore {
    config: LfccConfig,

    hann_window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
    dct_matrix: Vec<Vec<f32>>,
    scratch_fft: Vec<f32>,
}

impl LfccCore {
    pub fn new(config: LfccConfig) -> Self {
        todo!()
    }

    #[inline]
    pub fn process_frame(&mut self, pre_emphasized_frame: &[f32], output_lfccs: &[f32]) {
        todo!();
    }
}
