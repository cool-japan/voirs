//! Configuration for audio quality research

/// Configuration for audio quality research
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// Enable neural quality prediction models
    pub neural_models: bool,
    /// Depth of psychoacoustic analysis (1-10)
    pub psychoacoustic_depth: u8,
    /// Enable PESQ-style analysis
    pub pesq_analysis: bool,
    /// Enable STOI-style analysis
    pub stoi_analysis: bool,
    /// Enable PEMO-Q style analysis
    pub pemo_q_analysis: bool,
    /// Sample rate for analysis
    pub sample_rate: u32,
    /// Frame size for analysis (samples)
    pub frame_size: usize,
    /// Overlap factor (0.0-1.0)
    pub overlap_factor: f32,
    /// Enable advanced spectral analysis
    pub advanced_spectral: bool,
    /// Enable temporal coherence analysis
    pub temporal_coherence: bool,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            neural_models: true,
            psychoacoustic_depth: 5,
            pesq_analysis: true,
            stoi_analysis: true,
            pemo_q_analysis: true,
            sample_rate: 16000,
            frame_size: 1024,
            overlap_factor: 0.5,
            advanced_spectral: true,
            temporal_coherence: true,
        }
    }
}

impl ResearchConfig {
    /// Enable or disable neural models
    pub fn with_neural_models(mut self, enable: bool) -> Self {
        self.neural_models = enable;
        self
    }

    /// Set psychoacoustic analysis depth
    pub fn with_psychoacoustic_depth(mut self, depth: u8) -> Self {
        self.psychoacoustic_depth = depth.clamp(1, 10);
        self
    }

    /// Set sample rate for analysis
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }
}
