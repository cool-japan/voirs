//! Singing synthesis engine and processing
//!
//! This module has been refactored into smaller, more manageable components
//! for better maintainability and organization. It provides comprehensive
//! singing voice synthesis capabilities including harmonic synthesis,
//! spectral processing, and noise generation.

pub mod core;
pub mod diffsinger;
pub mod harmonic;
pub mod models;
pub mod noise;
pub mod processor;
pub mod results;
pub mod spectral;

// Re-export main types and traits
pub use core::{SynthesisEngine, SynthesisModel, SynthesisParams};
pub use processor::SynthesisProcessor;
pub use results::{
    ExpressionFidelityMetrics, PitchAccuracyMetrics, PrecisionMetricsReport, PrecisionTargets,
    QualityMetrics, SpectralQualityMetrics, SynthesisResult, SynthesisStats, TimingAccuracyMetrics,
};

// Re-export spectral processing types
pub use spectral::SpectralProcessor;

// Re-export harmonic processing types
pub use harmonic::HarmonicProcessor;

// Re-export noise processing types
pub use noise::{NoiseProcessor, NoiseType};

// Re-export synthesis models
pub use models::{HarmonicSynthesisModel, HybridSynthesisModel, SpectralSynthesisModel};

// Re-export DiffSinger types
pub use diffsinger::{
    ConditioningFeature, DiffSingerConfig, DiffSingerModel, DiffSingerRequest, MusicalConditioning,
    NoiseSchedule, QualityLevel, VocoderType,
};

/// Create a default synthesis engine
pub fn create_default_engine() -> SynthesisEngine {
    SynthesisEngine::default()
}

/// Create a synthesis engine with harmonic model
pub fn create_harmonic_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let harmonic_model = Box::new(HarmonicSynthesisModel::new());
    engine.add_model("harmonic".to_string(), harmonic_model);
    Ok(engine)
}

/// Create a synthesis engine with spectral model
pub fn create_spectral_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let spectral_model = Box::new(SpectralSynthesisModel::new());
    engine.add_model("spectral".to_string(), spectral_model);
    Ok(engine)
}

/// Create a synthesis engine with hybrid model
pub fn create_hybrid_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let hybrid_model = Box::new(HybridSynthesisModel::new());
    engine.add_model("hybrid".to_string(), hybrid_model);
    Ok(engine)
}

/// Create a synthesis engine with DiffSinger model
pub fn create_diffsinger_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let diffsinger_model = Box::new(DiffSingerModel::default());
    engine.add_model("diffsinger".to_string(), diffsinger_model);
    Ok(engine)
}

/// Create a high-quality DiffSinger engine
pub fn create_diffsinger_hq_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let diffsinger_model = Box::new(DiffSingerModel::high_quality());
    engine.add_model("diffsinger_hq".to_string(), diffsinger_model);
    Ok(engine)
}

/// Create a fast DiffSinger engine
pub fn create_diffsinger_fast_engine() -> crate::Result<SynthesisEngine> {
    let mut engine = SynthesisEngine::default();
    let diffsinger_model = Box::new(DiffSingerModel::fast());
    engine.add_model("diffsinger_fast".to_string(), diffsinger_model);
    Ok(engine)
}

/// Synthesis engine configuration
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Default frame size for processing
    pub frame_size: usize,
    /// Default hop size for processing
    pub hop_size: usize,
    /// Default sample rate
    pub sample_rate: f32,
    /// Maximum number of harmonics
    pub max_harmonics: usize,
    /// Default noise level
    pub default_noise_level: f32,
    /// Quality targets
    pub quality_targets: PrecisionTargets,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 256,
            sample_rate: 44100.0,
            max_harmonics: 64,
            default_noise_level: 0.1,
            quality_targets: PrecisionTargets::default(),
        }
    }
}

/// Synthesis capabilities enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisCapability {
    /// Basic harmonic synthesis
    Harmonic,
    /// Spectral synthesis with FFT
    Spectral,
    /// Noise synthesis
    Noise,
    /// Formant synthesis
    Formant,
    /// Granular synthesis
    Granular,
    /// Neural synthesis
    Neural,
    /// DiffSinger diffusion-based synthesis
    DiffSinger,
}

/// Get available synthesis capabilities
pub fn get_synthesis_capabilities() -> Vec<SynthesisCapability> {
    vec![
        SynthesisCapability::Harmonic,
        SynthesisCapability::Spectral,
        SynthesisCapability::Noise,
        SynthesisCapability::Formant,
        SynthesisCapability::DiffSinger,
    ]
}

/// Check if a capability is available
pub fn has_capability(capability: SynthesisCapability) -> bool {
    get_synthesis_capabilities().contains(&capability)
}

/// Synthesis performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Synthesis throughput in samples per second
    pub samples_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Real-time factor (>1.0 means faster than real-time)
    pub real_time_factor: f64,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            samples_per_second: 0.0,
            memory_usage: 0,
            cpu_usage: 0.0,
            real_time_factor: 0.0,
        }
    }

    /// Check if performance meets real-time requirements
    pub fn is_real_time(&self) -> bool {
        self.real_time_factor >= 1.0
    }

    /// Get performance summary string
    pub fn summary(&self) -> String {
        format!(
            "Throughput: {:.1} kSPS, Memory: {} KB, CPU: {:.1}%, RT Factor: {:.2}x",
            self.samples_per_second / 1000.0,
            self.memory_usage / 1024,
            self.cpu_usage,
            self.real_time_factor
        )
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
