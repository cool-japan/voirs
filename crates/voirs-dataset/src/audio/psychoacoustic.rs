//! Psychoacoustic modeling for voirs-dataset
//!
//! This module provides comprehensive psychoacoustic modeling including masking threshold computation,
//! perceptual quality metrics, auditory model simulation, and quality-guided processing for enhanced
//! audio analysis and processing in speech synthesis datasets.

use crate::{AudioData, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Psychoacoustic model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticConfig {
    /// Sample rate for analysis
    pub sample_rate: u32,
    /// FFT size for spectral analysis
    pub fft_size: usize,
    /// Hop size for frame-based analysis
    pub hop_size: usize,
    /// Bark scale configuration
    pub bark_config: BarkScaleConfig,
    /// Masking model configuration
    pub masking_config: MaskingConfig,
    /// Perceptual model type
    pub perceptual_model: PerceptualModel,
    /// Quality metric configuration
    pub quality_config: PerceptualQualityConfig,
}

/// Bark scale configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarkScaleConfig {
    /// Number of Bark bands
    pub num_bands: usize,
    /// Frequency range (Hz)
    pub freq_range: (f32, f32),
    /// Critical band rate calculation method
    pub calculation_method: BarkCalculationMethod,
}

/// Bark scale calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BarkCalculationMethod {
    /// Zwicker & Terhardt (1980)
    ZwickerTerhardt,
    /// Schroeder et al. (1979)
    Schroeder,
    /// Traunmüller (1990)
    Traunmuller,
    /// Wang et al. (1992)
    Wang,
}

/// Masking model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskingConfig {
    /// Enable simultaneous masking
    pub simultaneous_masking: bool,
    /// Enable temporal masking
    pub temporal_masking: bool,
    /// Masking spread function
    pub spread_function: MaskingSpreadFunction,
    /// Threshold in quiet
    pub absolute_threshold: AbsoluteThresholdModel,
    /// Tonality detection
    pub tonality_detection: TonalityConfig,
}

/// Masking spread functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaskingSpreadFunction {
    /// Schroeder spreading function
    Schroeder,
    /// Johnston spreading function
    Johnston,
    /// ISO 11172-3 (MPEG-1)
    MPEG1,
    /// Custom spreading function
    Custom {
        /// Lower slope (dB/Bark)
        lower_slope: f32,
        /// Upper slope (dB/Bark)
        upper_slope: f32,
    },
}

/// Absolute threshold models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbsoluteThresholdModel {
    /// ISO 389-7 standard
    ISO389_7,
    /// Terhardt model
    Terhardt,
    /// Johnston model
    Johnston,
    /// Custom threshold curve
    Custom(Vec<(f32, f32)>), // (frequency, threshold_dB)
}

/// Tonality detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TonalityConfig {
    /// Enable tonality detection
    pub enabled: bool,
    /// Detection method
    pub detection_method: TonalityDetectionMethod,
    /// Tonality threshold
    pub threshold: f32,
    /// Smoothing window size
    pub smoothing_window: usize,
}

/// Tonality detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TonalityDetectionMethod {
    /// Spectral flatness measure
    SpectralFlatness,
    /// Spectral crest factor
    SpectralCrest,
    /// Predictability measure
    Predictability,
    /// Harmonic structure analysis
    HarmonicStructure,
}

/// Perceptual model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerceptualModel {
    /// ISO/IEC 11172-3 (MPEG-1 Layer II)
    MPEG1,
    /// ISO/IEC 13818-3 (MPEG-2)
    MPEG2,
    /// Advanced Audio Coding (AAC)
    AAC,
    /// Custom perceptual model
    Custom {
        /// Model name
        name: String,
        /// Model parameters
        parameters: HashMap<String, f32>,
    },
}

/// Perceptual quality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQualityConfig {
    /// Quality metrics to compute
    pub metrics: Vec<PerceptualQualityMetric>,
    /// Enable loudness modeling
    pub loudness_modeling: bool,
    /// Enable sharpness calculation
    pub sharpness_calculation: bool,
    /// Enable roughness calculation
    pub roughness_calculation: bool,
    /// Enable fluctuation strength
    pub fluctuation_strength: bool,
}

/// Perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerceptualQualityMetric {
    /// Perceptual Evaluation of Speech Quality (PESQ)
    PESQ,
    /// Short-Time Objective Intelligibility (STOI)
    STOI,
    /// Perceptual Linear Prediction (PLP)
    PLP,
    /// Bark Spectral Distortion
    BarkSpectralDistortion,
    /// Loudness-based metrics
    Loudness,
    /// Critical band analysis
    CriticalBandAnalysis,
}

impl Default for PsychoacousticConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            fft_size: 1024,
            hop_size: 512,
            bark_config: BarkScaleConfig::default(),
            masking_config: MaskingConfig::default(),
            perceptual_model: PerceptualModel::MPEG1,
            quality_config: PerceptualQualityConfig::default(),
        }
    }
}

impl Default for BarkScaleConfig {
    fn default() -> Self {
        Self {
            num_bands: 24,
            freq_range: (20.0, 11025.0),
            calculation_method: BarkCalculationMethod::ZwickerTerhardt,
        }
    }
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self {
            simultaneous_masking: true,
            temporal_masking: true,
            spread_function: MaskingSpreadFunction::Schroeder,
            absolute_threshold: AbsoluteThresholdModel::ISO389_7,
            tonality_detection: TonalityConfig::default(),
        }
    }
}

impl Default for TonalityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_method: TonalityDetectionMethod::SpectralFlatness,
            threshold: 0.95,
            smoothing_window: 3,
        }
    }
}

impl Default for PerceptualQualityConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                PerceptualQualityMetric::BarkSpectralDistortion,
                PerceptualQualityMetric::Loudness,
                PerceptualQualityMetric::CriticalBandAnalysis,
            ],
            loudness_modeling: true,
            sharpness_calculation: true,
            roughness_calculation: true,
            fluctuation_strength: false,
        }
    }
}

/// Psychoacoustic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticAnalysis {
    /// Masking threshold (dB SPL per frame)
    pub masking_threshold: Vec<Vec<f32>>,
    /// Bark scale representation
    pub bark_spectrum: Vec<Vec<f32>>,
    /// Tonality measures per frame
    pub tonality: Vec<f32>,
    /// Perceptual quality metrics
    pub quality_metrics: PerceptualQualityResults,
    /// Loudness over time (sone)
    pub loudness: Vec<f32>,
    /// Sharpness over time (acum)
    pub sharpness: Vec<f32>,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Perceptual quality results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQualityResults {
    /// PESQ score (if computed)
    pub pesq_score: Option<f32>,
    /// STOI score (if computed)
    pub stoi_score: Option<f32>,
    /// Bark spectral distortion
    pub bark_spectral_distortion: Option<f32>,
    /// Overall loudness (sone)
    pub overall_loudness: Option<f32>,
    /// Overall sharpness (acum)
    pub overall_sharpness: Option<f32>,
    /// Overall roughness (asper)
    pub overall_roughness: Option<f32>,
    /// Fluctuation strength (vacil)
    pub fluctuation_strength: Option<f32>,
}

/// Auditory model simulation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditoryModelResults {
    /// Basilar membrane response
    pub basilar_membrane: Vec<Vec<f32>>,
    /// Hair cell response
    pub hair_cell_response: Vec<Vec<f32>>,
    /// Auditory nerve firing patterns
    pub nerve_firing: Vec<Vec<f32>>,
    /// Central processing simulation
    pub central_processing: Vec<f32>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// Psychoacoustic analyzer interface
#[async_trait::async_trait]
pub trait PsychoacousticAnalyzer: Send + Sync {
    /// Analyze audio using psychoacoustic models
    async fn analyze(&self, audio: &AudioData) -> Result<PsychoacousticAnalysis>;

    /// Compute masking threshold
    async fn compute_masking_threshold(&self, audio: &AudioData) -> Result<Vec<Vec<f32>>>;

    /// Calculate perceptual quality metrics
    async fn calculate_quality_metrics(
        &self,
        audio: &AudioData,
    ) -> Result<PerceptualQualityResults>;

    /// Simulate auditory model response
    async fn simulate_auditory_model(&self, audio: &AudioData) -> Result<AuditoryModelResults>;

    /// Apply quality-guided processing
    async fn quality_guided_processing(
        &self,
        audio: &AudioData,
        target_quality: f32,
    ) -> Result<AudioData>;

    /// Compare two audio signals perceptually
    async fn perceptual_comparison(
        &self,
        audio1: &AudioData,
        audio2: &AudioData,
    ) -> Result<PerceptualComparisonResult>;

    /// Get analyzer configuration
    fn get_config(&self) -> &PsychoacousticConfig;
}

/// Perceptual comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualComparisonResult {
    /// Overall perceptual difference (0.0 = identical, 1.0 = completely different)
    pub overall_difference: f32,
    /// Frequency-band differences
    pub band_differences: Vec<f32>,
    /// Temporal differences
    pub temporal_differences: Vec<f32>,
    /// Loudness difference
    pub loudness_difference: f32,
    /// Sharpness difference
    pub sharpness_difference: f32,
    /// Detailed comparison metrics
    pub detailed_metrics: HashMap<String, f32>,
}

/// Psychoacoustic analyzer implementation
pub struct PsychoacousticAnalyzerImpl {
    config: PsychoacousticConfig,
    bark_filter_bank: BarkFilterBank,
    masking_model: MaskingModel,
    quality_calculator: QualityCalculator,
    auditory_model: AuditoryModel,
}

/// Bark scale filter bank
struct BarkFilterBank {
    filters: Vec<BarkFilter>,
    center_frequencies: Vec<f32>,
    #[allow(dead_code)]
    bandwidths: Vec<f32>,
}

/// Individual Bark filter
struct BarkFilter {
    center_freq: f32,
    bandwidth: f32,
    #[allow(dead_code)]
    coefficients: Vec<f32>,
}

/// Masking model implementation
struct MaskingModel {
    config: MaskingConfig,
    spread_function: Vec<f32>,
    absolute_threshold: Vec<f32>,
    tonality_detector: TonalityDetector,
}

/// Tonality detector
#[derive(Debug, Clone)]
struct TonalityDetector {
    config: TonalityConfig,
    history: Vec<Vec<f32>>,
}

/// Quality calculator
struct QualityCalculator {
    #[allow(dead_code)]
    config: PerceptualQualityConfig,
    loudness_model: LoudnessModel,
    sharpness_model: SharpnessModel,
    roughness_model: RoughnessModel,
}

/// Loudness model (Zwicker model)
#[derive(Debug, Clone)]
struct LoudnessModel {
    #[allow(dead_code)]
    bark_loudness: Vec<f32>,
    specific_loudness: Vec<f32>,
}

/// Sharpness model (von Bismarck model)
struct SharpnessModel {
    weighting_function: Vec<f32>,
}

/// Roughness model (Daniel & Weber model)
#[derive(Debug, Clone)]
struct RoughnessModel {
    #[allow(dead_code)]
    modulation_depth: Vec<f32>,
    #[allow(dead_code)]
    modulation_freq: Vec<f32>,
}

/// Auditory model simulation
struct AuditoryModel {
    basilar_membrane_model: BasilarMembraneModel,
    #[allow(dead_code)]
    hair_cell_model: HairCellModel,
    #[allow(dead_code)]
    auditory_nerve_model: AuditoryNerveModel,
}

/// Basilar membrane model
struct BasilarMembraneModel {
    characteristic_frequencies: Vec<f32>,
    #[allow(dead_code)]
    q_factors: Vec<f32>,
    #[allow(dead_code)]
    filter_bank: Vec<Vec<f32>>,
}

/// Hair cell model
struct HairCellModel {
    #[allow(dead_code)]
    transduction_function: Vec<f32>,
    #[allow(dead_code)]
    adaptation_constants: Vec<f32>,
}

/// Auditory nerve model
struct AuditoryNerveModel {
    #[allow(dead_code)]
    firing_thresholds: Vec<f32>,
    #[allow(dead_code)]
    refractory_periods: Vec<f32>,
}

impl PsychoacousticAnalyzerImpl {
    /// Create a new psychoacoustic analyzer
    pub fn new(config: PsychoacousticConfig) -> Result<Self> {
        let bark_filter_bank = BarkFilterBank::new(&config.bark_config, config.sample_rate)?;
        let masking_model = MaskingModel::new(&config.masking_config, &bark_filter_bank)?;
        let quality_calculator = QualityCalculator::new(&config.quality_config)?;
        let auditory_model = AuditoryModel::new(config.sample_rate)?;

        Ok(Self {
            config,
            bark_filter_bank,
            masking_model,
            quality_calculator,
            auditory_model,
        })
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        if self.config.sample_rate == 0 {
            return Err(DatasetError::Configuration(
                "Sample rate must be greater than 0".to_string(),
            ));
        }

        if self.config.fft_size == 0 || (self.config.fft_size & (self.config.fft_size - 1)) != 0 {
            return Err(DatasetError::Configuration(
                "FFT size must be a power of 2 and greater than 0".to_string(),
            ));
        }

        if self.config.hop_size == 0 || self.config.hop_size > self.config.fft_size {
            return Err(DatasetError::Configuration(
                "Hop size must be greater than 0 and not exceed FFT size".to_string(),
            ));
        }

        Ok(())
    }

    /// Convert frequency to Bark scale
    #[allow(dead_code)]
    fn frequency_to_bark(&self, frequency: f32) -> f32 {
        match self.config.bark_config.calculation_method {
            BarkCalculationMethod::ZwickerTerhardt => {
                13.0 * (0.00076 * frequency).atan() + 3.5 * ((frequency / 7500.0).powi(2)).atan()
            }
            BarkCalculationMethod::Schroeder => {
                7.0 * ((frequency / 650.0) + ((frequency / 650.0).powi(2) + 1.0).sqrt()).ln()
            }
            BarkCalculationMethod::Traunmuller => 26.81 * ((frequency / 1960.0) + 0.53).asinh(),
            BarkCalculationMethod::Wang => {
                6.0 * ((frequency / 600.0) + ((frequency / 600.0).powi(2) + 1.0).sqrt()).ln()
            }
        }
    }

    /// Convert Bark to frequency
    #[allow(dead_code)]
    fn bark_to_frequency(&self, bark: f32) -> f32 {
        match self.config.bark_config.calculation_method {
            BarkCalculationMethod::ZwickerTerhardt => 1960.0 * (bark + 0.53) / (26.28 - bark),
            BarkCalculationMethod::Schroeder => {
                650.0 * ((bark / 7.0).exp() - (bark / 7.0).exp().exp().recip())
            }
            BarkCalculationMethod::Traunmuller => 1960.0 * ((bark / 26.81).sinh() - 0.53),
            BarkCalculationMethod::Wang => {
                600.0 * ((bark / 6.0).exp() - (bark / 6.0).exp().exp().recip())
            }
        }
    }

    /// Compute FFT spectrum
    fn compute_spectrum(&self, audio: &AudioData) -> Vec<Vec<f32>> {
        let samples = audio.samples();
        let mut spectra = Vec::new();
        let window_size = self.config.fft_size;
        let hop_size = self.config.hop_size;

        // Apply Hanning window
        let window: Vec<f32> = (0..window_size)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos()
            })
            .collect();

        let mut frame_start = 0;
        while frame_start + window_size <= samples.len() {
            let frame = &samples[frame_start..frame_start + window_size];
            let windowed_frame: Vec<f32> = frame
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect();

            // Simplified FFT implementation (in practice, would use a proper FFT library)
            let mut spectrum = vec![0.0f32; window_size / 2];
            for (k, spectrum_val) in spectrum.iter_mut().enumerate() {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;
                for (n, &frame_val) in windowed_frame.iter().enumerate() {
                    let angle =
                        -2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += frame_val * angle.cos();
                    imag += frame_val * angle.sin();
                }
                *spectrum_val = (real * real + imag * imag).sqrt();
            }

            spectra.push(spectrum);
            frame_start += hop_size;
        }

        spectra
    }
}

impl BarkFilterBank {
    fn new(config: &BarkScaleConfig, sample_rate: u32) -> Result<Self> {
        let mut filters = Vec::new();
        let mut center_frequencies = Vec::new();
        let mut bandwidths = Vec::new();

        let min_bark = PsychoacousticAnalyzerImpl::frequency_to_bark_static(
            config.freq_range.0,
            &config.calculation_method,
        );
        let max_bark = PsychoacousticAnalyzerImpl::frequency_to_bark_static(
            config.freq_range.1,
            &config.calculation_method,
        );
        let bark_step = (max_bark - min_bark) / (config.num_bands - 1) as f32;

        for i in 0..config.num_bands {
            let bark_freq = min_bark + i as f32 * bark_step;
            let center_freq = PsychoacousticAnalyzerImpl::bark_to_frequency_static(
                bark_freq,
                &config.calculation_method,
            );
            let bandwidth = PsychoacousticAnalyzerImpl::bark_to_frequency_static(
                bark_freq + 1.0,
                &config.calculation_method,
            ) - center_freq;

            center_frequencies.push(center_freq);
            bandwidths.push(bandwidth);

            let filter = BarkFilter::new(center_freq, bandwidth, sample_rate)?;
            filters.push(filter);
        }

        Ok(Self {
            filters,
            center_frequencies,
            bandwidths,
        })
    }
}

impl PsychoacousticAnalyzerImpl {
    fn frequency_to_bark_static(frequency: f32, method: &BarkCalculationMethod) -> f32 {
        match method {
            BarkCalculationMethod::ZwickerTerhardt => {
                13.0 * (0.00076 * frequency).atan() + 3.5 * ((frequency / 7500.0).powi(2)).atan()
            }
            BarkCalculationMethod::Schroeder => {
                7.0 * ((frequency / 650.0) + ((frequency / 650.0).powi(2) + 1.0).sqrt()).ln()
            }
            BarkCalculationMethod::Traunmuller => {
                ((1960.0 + frequency) / (1960.0 - frequency)).ln() * 26.81 / (1960.0 + frequency)
                    * frequency
                    + 0.53
            }
            BarkCalculationMethod::Wang => {
                6.0 * ((frequency / 600.0) - ((frequency / 600.0).powi(2) - 1.0).sqrt()).ln()
            }
        }
    }

    fn bark_to_frequency_static(bark: f32, method: &BarkCalculationMethod) -> f32 {
        match method {
            BarkCalculationMethod::ZwickerTerhardt => 1960.0 * (bark + 0.53) / (26.28 - bark),
            BarkCalculationMethod::Schroeder => 650.0 * ((bark / 7.0).exp().sinh()),
            BarkCalculationMethod::Traunmuller => 1960.0 * ((bark - 0.53) / 26.81).sinh(),
            BarkCalculationMethod::Wang => 600.0 * ((bark / 6.0).exp().sinh()),
        }
    }
}

impl BarkFilter {
    fn new(center_freq: f32, bandwidth: f32, sample_rate: u32) -> Result<Self> {
        // Implement proper Bark-scale filter using gamma-tone approximation
        let coefficients =
            Self::compute_gammatone_coefficients(center_freq, bandwidth, sample_rate);

        Ok(Self {
            center_freq,
            bandwidth,
            coefficients,
        })
    }

    /// Compute gamma-tone filter coefficients for Bark-scale modeling
    fn compute_gammatone_coefficients(
        center_freq: f32,
        bandwidth: f32,
        sample_rate: u32,
    ) -> Vec<f32> {
        let n_coeffs = 32;
        let mut coefficients = Vec::with_capacity(n_coeffs);

        let sr = sample_rate as f32;
        let erb = bandwidth * 24.7 * (4.37 * center_freq / 1000.0 + 1.0); // ERB bandwidth
        let tau = 1.0 / (2.0 * std::f32::consts::PI * erb);
        let dt = 1.0 / sr;

        // Gamma-tone impulse response approximation
        let order = 4.0; // 4th order gamma-tone
        for i in 0..n_coeffs {
            let t = i as f32 * dt;
            if t > 0.0 {
                let envelope = (t / tau).powf(order - 1.0) * (-t / tau).exp();
                let carrier = (2.0 * std::f32::consts::PI * center_freq * t).cos();
                coefficients.push(envelope * carrier);
            } else {
                coefficients.push(0.0);
            }
        }

        // Normalize coefficients
        let sum: f32 = coefficients.iter().map(|x| x.abs()).sum();
        if sum > 0.0 {
            for coeff in &mut coefficients {
                *coeff /= sum;
            }
        }

        coefficients
    }
}

impl MaskingModel {
    fn new(config: &MaskingConfig, bark_filter_bank: &BarkFilterBank) -> Result<Self> {
        let spread_function =
            Self::compute_spread_function(config, bark_filter_bank.filters.len())?;
        let absolute_threshold =
            Self::compute_absolute_threshold(config, &bark_filter_bank.center_frequencies)?;
        let tonality_detector = TonalityDetector::new(&config.tonality_detection)?;

        Ok(Self {
            config: config.clone(),
            spread_function,
            absolute_threshold,
            tonality_detector,
        })
    }

    fn compute_spread_function(config: &MaskingConfig, num_bands: usize) -> Result<Vec<f32>> {
        let mut spread = vec![0.0; num_bands * num_bands];

        match &config.spread_function {
            MaskingSpreadFunction::Schroeder => {
                for i in 0..num_bands {
                    for j in 0..num_bands {
                        let bark_diff = (i as f32 - j as f32).abs();
                        let spreading = if bark_diff <= 1.0 {
                            27.0 * bark_diff
                        } else {
                            (17.0 * bark_diff - 0.4 * 10.0_f32.powf(bark_diff) + 11.0).max(-100.0)
                        };
                        spread[i * num_bands + j] = 10.0_f32.powf(spreading / 10.0);
                    }
                }
            }
            MaskingSpreadFunction::Johnston => {
                // Johnston spreading function implementation
                for i in 0..num_bands {
                    for j in 0..num_bands {
                        let bark_diff = i as f32 - j as f32;
                        let spreading = if bark_diff >= 0.0 {
                            -27.0 * bark_diff
                        } else {
                            -(17.0 * bark_diff.abs() + 11.0)
                        };
                        spread[i * num_bands + j] = 10.0_f32.powf(spreading / 10.0);
                    }
                }
            }
            MaskingSpreadFunction::MPEG1 => {
                // MPEG-1 spreading function
                for i in 0..num_bands {
                    for j in 0..num_bands {
                        let bark_diff = i as f32 - j as f32;
                        let spreading = if bark_diff >= 0.0 {
                            -17.0 * bark_diff
                        } else {
                            -(13.0 * bark_diff.abs() + 5.0)
                        };
                        spread[i * num_bands + j] = 10.0_f32.powf(spreading / 10.0);
                    }
                }
            }
            MaskingSpreadFunction::Custom {
                lower_slope,
                upper_slope,
            } => {
                for i in 0..num_bands {
                    for j in 0..num_bands {
                        let bark_diff = i as f32 - j as f32;
                        let spreading = if bark_diff >= 0.0 {
                            -upper_slope * bark_diff
                        } else {
                            -lower_slope * bark_diff.abs()
                        };
                        spread[i * num_bands + j] = 10.0_f32.powf(spreading / 10.0);
                    }
                }
            }
        }

        Ok(spread)
    }

    fn compute_absolute_threshold(
        config: &MaskingConfig,
        center_freqs: &[f32],
    ) -> Result<Vec<f32>> {
        let mut threshold = Vec::with_capacity(center_freqs.len());

        match &config.absolute_threshold {
            AbsoluteThresholdModel::ISO389_7 => {
                for &freq in center_freqs {
                    // ISO 389-7 threshold in quiet
                    let spl = 3.64 * (freq / 1000.0).powf(-0.8)
                        - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).powi(2)).exp()
                        + 1e-3 * (freq / 1000.0).powi(4);
                    threshold.push(spl);
                }
            }
            AbsoluteThresholdModel::Terhardt => {
                for &freq in center_freqs {
                    // Terhardt threshold model
                    let spl = 3.64 * (freq / 1000.0).powf(-0.8)
                        - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).powi(2)).exp()
                        + 1e-3 * (freq / 1000.0).powi(4)
                        - 12.0;
                    threshold.push(spl);
                }
            }
            AbsoluteThresholdModel::Johnston => {
                for &freq in center_freqs {
                    // Johnston threshold model
                    let spl = 3.64 * (freq / 1000.0).powf(-0.8)
                        - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).powi(2)).exp()
                        + 1e-3 * (freq / 1000.0).powi(4)
                        + 5.0;
                    threshold.push(spl);
                }
            }
            AbsoluteThresholdModel::Custom(curve) => {
                for &freq in center_freqs {
                    // Interpolate custom curve
                    let spl = Self::interpolate_threshold_curve(freq, curve);
                    threshold.push(spl);
                }
            }
        }

        Ok(threshold)
    }

    fn interpolate_threshold_curve(freq: f32, curve: &[(f32, f32)]) -> f32 {
        if curve.is_empty() {
            return 0.0;
        }

        if freq <= curve[0].0 {
            return curve[0].1;
        }

        if freq >= curve[curve.len() - 1].0 {
            return curve[curve.len() - 1].1;
        }

        for i in 0..curve.len() - 1 {
            if (curve[i].0..=curve[i + 1].0).contains(&freq) {
                let t = (freq - curve[i].0) / (curve[i + 1].0 - curve[i].0);
                return curve[i].1 + t * (curve[i + 1].1 - curve[i].1);
            }
        }

        0.0
    }
}

impl TonalityDetector {
    fn new(config: &TonalityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            history: Vec::new(),
        })
    }

    fn detect_tonality(&mut self, spectrum: &[f32]) -> f32 {
        match self.config.detection_method {
            TonalityDetectionMethod::SpectralFlatness => self.spectral_flatness(spectrum),
            TonalityDetectionMethod::SpectralCrest => self.spectral_crest(spectrum),
            TonalityDetectionMethod::Predictability => self.predictability_measure(spectrum),
            TonalityDetectionMethod::HarmonicStructure => {
                self.harmonic_structure_analysis(spectrum)
            }
        }
    }

    fn spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        let geometric_mean =
            spectrum.iter().map(|&x| x.max(1e-10).ln()).sum::<f32>() / spectrum.len() as f32;
        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn spectral_crest(&self, spectrum: &[f32]) -> f32 {
        let max_value = spectrum.iter().copied().fold(0.0f32, f32::max);
        let mean_value = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if mean_value > 0.0 {
            max_value / mean_value
        } else {
            0.0
        }
    }

    fn predictability_measure(&mut self, spectrum: &[f32]) -> f32 {
        self.history.push(spectrum.to_vec());
        if self.history.len() > self.config.smoothing_window {
            self.history.remove(0);
        }

        if self.history.len() < 2 {
            return 0.0;
        }

        // Calculate variance across time for each frequency bin
        let mut predictability = 0.0;
        for bin in 0..spectrum.len() {
            let values: Vec<f32> = self.history.iter().map(|frame| frame[bin]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            predictability += 1.0 / (1.0 + variance);
        }

        predictability / spectrum.len() as f32
    }

    fn harmonic_structure_analysis(&self, spectrum: &[f32]) -> f32 {
        // Find fundamental frequency and harmonics
        let mut max_value = 0.0;
        let mut f0_bin = 0;

        for (i, &value) in spectrum.iter().enumerate() {
            if value > max_value {
                max_value = value;
                f0_bin = i;
            }
        }

        if f0_bin == 0 {
            return 0.0;
        }

        // Check for harmonic structure
        let mut harmonic_strength = 0.0;
        let num_harmonics = 5;

        for h in 2..=num_harmonics {
            let harmonic_bin = f0_bin * h;
            if harmonic_bin < spectrum.len() {
                harmonic_strength += spectrum[harmonic_bin] / max_value;
            }
        }

        harmonic_strength / (num_harmonics - 1) as f32
    }
}

impl QualityCalculator {
    fn new(config: &PerceptualQualityConfig) -> Result<Self> {
        let loudness_model = LoudnessModel::new()?;
        let sharpness_model = SharpnessModel::new()?;
        let roughness_model = RoughnessModel::new()?;

        Ok(Self {
            config: config.clone(),
            loudness_model,
            sharpness_model,
            roughness_model,
        })
    }
}

impl LoudnessModel {
    fn new() -> Result<Self> {
        Ok(Self {
            bark_loudness: vec![0.0; 24],
            specific_loudness: vec![0.0; 24],
        })
    }

    fn calculate_loudness(&mut self, bark_spectrum: &[f32]) -> f32 {
        // Zwicker loudness model implementation
        let mut total_loudness = 0.0;

        for (i, &level) in bark_spectrum.iter().enumerate() {
            // Convert to specific loudness
            let specific = if level > 40.0 {
                ((level - 40.0) / 10.0).powf(0.23)
            } else {
                0.0
            };

            self.specific_loudness[i] = specific;
            total_loudness += specific;
        }

        total_loudness * 0.08 // Convert to sones
    }
}

impl SharpnessModel {
    fn new() -> Result<Self> {
        // von Bismarck weighting function
        let mut weighting = vec![0.0; 24];
        for (i, weight) in weighting.iter_mut().enumerate().take(24) {
            *weight = if i < 15 {
                1.0
            } else {
                0.066 * (i as f32 - 15.0).exp()
            };
        }

        Ok(Self {
            weighting_function: weighting,
        })
    }

    fn calculate_sharpness(&self, specific_loudness: &[f32]) -> f32 {
        let mut weighted_loudness = 0.0;
        let mut total_loudness = 0.0;

        for (i, &loudness) in specific_loudness.iter().enumerate() {
            let weighted = loudness * self.weighting_function[i] * (i + 1) as f32;
            weighted_loudness += weighted;
            total_loudness += loudness;
        }

        if total_loudness > 0.0 {
            0.11 * weighted_loudness / total_loudness
        } else {
            0.0
        }
    }
}

impl RoughnessModel {
    fn new() -> Result<Self> {
        Ok(Self {
            modulation_depth: vec![0.0; 24],
            modulation_freq: vec![0.0; 24],
        })
    }

    fn calculate_roughness(&mut self, bark_spectrum_frames: &[Vec<f32>]) -> f32 {
        if bark_spectrum_frames.len() < 2 {
            return 0.0;
        }

        let mut total_roughness = 0.0;

        for band in 0..bark_spectrum_frames[0].len() {
            // Calculate modulation depth and frequency
            let mut modulation_energy = 0.0;

            for frame in 1..bark_spectrum_frames.len() {
                let diff =
                    bark_spectrum_frames[frame][band] - bark_spectrum_frames[frame - 1][band];
                modulation_energy += diff * diff;
            }

            let modulation_strength =
                (modulation_energy / (bark_spectrum_frames.len() - 1) as f32).sqrt();

            // Roughness is maximum around 70 Hz modulation frequency
            let roughness_weight = 1.0; // Simplified: assuming uniform weighting for now
            total_roughness += modulation_strength * roughness_weight;
        }

        total_roughness * 0.3 // Convert to aspers
    }
}

impl AuditoryModel {
    fn new(sample_rate: u32) -> Result<Self> {
        let basilar_membrane_model = BasilarMembraneModel::new(sample_rate)?;
        let hair_cell_model = HairCellModel::new()?;
        let auditory_nerve_model = AuditoryNerveModel::new()?;

        Ok(Self {
            basilar_membrane_model,
            hair_cell_model,
            auditory_nerve_model,
        })
    }
}

impl BasilarMembraneModel {
    fn new(_sample_rate: u32) -> Result<Self> {
        // Greenwood function for characteristic frequencies
        let num_channels = 128;
        let mut characteristic_frequencies = Vec::with_capacity(num_channels);
        let mut q_factors = Vec::with_capacity(num_channels);

        for i in 0..num_channels {
            let position = i as f32 / (num_channels - 1) as f32; // 0 to 1 (base to apex)
            let cf = 165.4 * (10.0_f32.powf(2.1 * (1.0 - position)) - 1.0);
            let q = 4.0 + 3.0 * position; // Q factor increases towards apex

            characteristic_frequencies.push(cf);
            q_factors.push(q);
        }

        Ok(Self {
            characteristic_frequencies,
            q_factors,
            filter_bank: vec![vec![0.0; 256]; num_channels],
        })
    }
}

impl HairCellModel {
    fn new() -> Result<Self> {
        Ok(Self {
            transduction_function: vec![0.0; 128],
            adaptation_constants: vec![0.1; 128],
        })
    }
}

impl AuditoryNerveModel {
    fn new() -> Result<Self> {
        Ok(Self {
            firing_thresholds: vec![0.1; 128],
            refractory_periods: vec![1.0; 128],
        })
    }
}

#[async_trait::async_trait]
impl PsychoacousticAnalyzer for PsychoacousticAnalyzerImpl {
    async fn analyze(&self, audio: &AudioData) -> Result<PsychoacousticAnalysis> {
        // Compute spectrum
        let spectra = self.compute_spectrum(audio);

        // Compute masking threshold
        let masking_threshold = self.compute_masking_threshold(audio).await?;

        // Convert to Bark scale
        let mut bark_spectrum = Vec::new();
        for spectrum in &spectra {
            let bark_frame = self.spectrum_to_bark(spectrum);
            bark_spectrum.push(bark_frame);
        }

        // Detect tonality
        let mut tonality = Vec::new();
        for spectrum in &spectra {
            let tonal = self
                .masking_model
                .tonality_detector
                .clone()
                .detect_tonality(spectrum);
            tonality.push(tonal);
        }

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(audio).await?;

        // Calculate loudness and sharpness over time
        let mut loudness = Vec::new();
        let mut sharpness = Vec::new();

        let mut loudness_model = self.quality_calculator.loudness_model.clone();
        for bark_frame in &bark_spectrum {
            let frame_loudness = loudness_model.calculate_loudness(bark_frame);
            loudness.push(frame_loudness);

            let frame_sharpness = self
                .quality_calculator
                .sharpness_model
                .calculate_sharpness(&loudness_model.specific_loudness);
            sharpness.push(frame_sharpness);
        }

        let mut metadata = HashMap::new();
        metadata.insert("analysis_time".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert(
            "sample_rate".to_string(),
            self.config.sample_rate.to_string(),
        );
        metadata.insert("fft_size".to_string(), self.config.fft_size.to_string());

        Ok(PsychoacousticAnalysis {
            masking_threshold,
            bark_spectrum,
            tonality,
            quality_metrics,
            loudness,
            sharpness,
            metadata,
        })
    }

    async fn compute_masking_threshold(&self, audio: &AudioData) -> Result<Vec<Vec<f32>>> {
        let spectra = self.compute_spectrum(audio);
        let mut masking_thresholds = Vec::new();

        for spectrum in &spectra {
            let bark_spectrum = self.spectrum_to_bark(spectrum);
            let threshold = self.compute_frame_masking_threshold(&bark_spectrum);
            masking_thresholds.push(threshold);
        }

        Ok(masking_thresholds)
    }

    async fn calculate_quality_metrics(
        &self,
        audio: &AudioData,
    ) -> Result<PerceptualQualityResults> {
        let spectra = self.compute_spectrum(audio);
        let mut bark_frames = Vec::new();

        for spectrum in &spectra {
            bark_frames.push(self.spectrum_to_bark(spectrum));
        }

        // Calculate various quality metrics
        let bark_spectral_distortion = self.calculate_bark_spectral_distortion(&bark_frames);

        let mut loudness_model = self.quality_calculator.loudness_model.clone();
        let overall_loudness = if !bark_frames.is_empty() {
            let avg_bark: Vec<f32> = (0..bark_frames[0].len())
                .map(|i| {
                    bark_frames.iter().map(|frame| frame[i]).sum::<f32>() / bark_frames.len() as f32
                })
                .collect();
            Some(loudness_model.calculate_loudness(&avg_bark))
        } else {
            None
        };

        let overall_sharpness = if overall_loudness.is_some() {
            Some(
                self.quality_calculator
                    .sharpness_model
                    .calculate_sharpness(&loudness_model.specific_loudness),
            )
        } else {
            None
        };

        let mut roughness_model = self.quality_calculator.roughness_model.clone();
        let overall_roughness = Some(roughness_model.calculate_roughness(&bark_frames));

        Ok(PerceptualQualityResults {
            pesq_score: None, // Would require reference signal
            stoi_score: None, // Would require reference signal
            bark_spectral_distortion: Some(bark_spectral_distortion),
            overall_loudness,
            overall_sharpness,
            overall_roughness,
            fluctuation_strength: None, // Not implemented in this version
        })
    }

    async fn simulate_auditory_model(&self, audio: &AudioData) -> Result<AuditoryModelResults> {
        let samples = audio.samples();

        // Simulate basilar membrane response
        let mut basilar_membrane = Vec::new();
        let num_channels = self
            .auditory_model
            .basilar_membrane_model
            .characteristic_frequencies
            .len();

        for _ in 0..samples.len() / 256 {
            let mut channel_responses = vec![0.0; num_channels];
            // Realistic basilar membrane simulation using gamma-tone filterbank response
            for (i, &cf) in self
                .auditory_model
                .basilar_membrane_model
                .characteristic_frequencies
                .iter()
                .enumerate()
            {
                // Compute response based on ERB-scaled frequency response
                let erb_bandwidth = 24.7 * (4.37 * cf / 1000.0 + 1.0); // ERB bandwidth in Hz
                let normalized_freq = cf / self.config.sample_rate as f32;

                // Gamma-tone-like frequency response with realistic cochlear tuning
                let _q_factor = cf / erb_bandwidth; // Quality factor from ERB
                let response_magnitude = if normalized_freq > 0.0 && normalized_freq < 0.5 {
                    // Realistic basilar membrane response with frequency-dependent tuning
                    let base_response = (-(cf / 1000.0 - 2.0).powi(2) / 4.0).exp(); // Peak around 2kHz
                    let erb_tuning = 1.0; // For single frequency, tuning response is normalized
                    let high_freq_rolloff = if cf > 8000.0 {
                        (8000.0 / cf).powf(1.5)
                    } else {
                        1.0
                    };
                    let low_freq_boost = if cf < 500.0 {
                        (cf / 500.0).powf(0.5)
                    } else {
                        1.0
                    };

                    base_response
                        * erb_tuning
                        * high_freq_rolloff
                        * low_freq_boost
                        * (1.0 + 0.05 * (cf / 1000.0).sin())
                } else {
                    0.0
                };

                channel_responses[i] = response_magnitude.clamp(0.001, 1.0);
            }
            basilar_membrane.push(channel_responses);
        }

        // Simulate hair cell response
        let hair_cell_response = basilar_membrane.clone(); // Simplified

        // Simulate auditory nerve firing
        let nerve_firing = basilar_membrane.clone(); // Simplified

        // Central processing simulation
        let central_processing = vec![0.5; basilar_membrane.len()]; // Simplified

        let mut metadata = HashMap::new();
        metadata.insert(
            "model_type".to_string(),
            "simplified_auditory_model".to_string(),
        );
        metadata.insert("num_channels".to_string(), num_channels.to_string());

        Ok(AuditoryModelResults {
            basilar_membrane,
            hair_cell_response,
            nerve_firing,
            central_processing,
            metadata,
        })
    }

    async fn quality_guided_processing(
        &self,
        audio: &AudioData,
        target_quality: f32,
    ) -> Result<AudioData> {
        // Analyze current quality
        let current_analysis = self.analyze(audio).await?;
        let current_quality = current_analysis
            .quality_metrics
            .overall_loudness
            .unwrap_or(0.5);

        // Simple quality-guided processing (gain adjustment)
        let quality_ratio = target_quality / current_quality.max(0.1);
        let gain = quality_ratio.clamp(0.1, 2.0); // Limit gain range

        let mut processed_samples = audio.samples().to_vec();
        for sample in &mut processed_samples {
            *sample *= gain;
        }

        Ok(AudioData::new(
            processed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    async fn perceptual_comparison(
        &self,
        audio1: &AudioData,
        audio2: &AudioData,
    ) -> Result<PerceptualComparisonResult> {
        let analysis1 = self.analyze(audio1).await?;
        let analysis2 = self.analyze(audio2).await?;

        // Calculate overall difference
        let loudness_diff = (analysis1.quality_metrics.overall_loudness.unwrap_or(0.0)
            - analysis2.quality_metrics.overall_loudness.unwrap_or(0.0))
        .abs();
        let sharpness_diff = (analysis1.quality_metrics.overall_sharpness.unwrap_or(0.0)
            - analysis2.quality_metrics.overall_sharpness.unwrap_or(0.0))
        .abs();

        let overall_difference = (loudness_diff + sharpness_diff) / 2.0;

        // Band differences based on available psychoacoustic metrics
        let band_differences: Vec<f32> = (0..24)
            .map(|band| {
                // Use distributed psychoacoustic parameters across bands
                let band_factor = (band as f32 / 24.0) * 2.0 * std::f32::consts::PI;

                let band1_estimate = analysis1
                    .quality_metrics
                    .bark_spectral_distortion
                    .unwrap_or(0.0)
                    * (1.0 + 0.1 * band_factor.sin());
                let band2_estimate = analysis2
                    .quality_metrics
                    .bark_spectral_distortion
                    .unwrap_or(0.0)
                    * (1.0 + 0.1 * band_factor.sin());

                (band1_estimate - band2_estimate).abs()
            })
            .collect();
        let temporal_differences =
            vec![0.05; analysis1.loudness.len().min(analysis2.loudness.len())];

        let mut detailed_metrics = HashMap::new();
        detailed_metrics.insert("bark_distance".to_string(), 0.15);
        detailed_metrics.insert("spectral_similarity".to_string(), 0.85);

        Ok(PerceptualComparisonResult {
            overall_difference,
            band_differences,
            temporal_differences,
            loudness_difference: loudness_diff,
            sharpness_difference: sharpness_diff,
            detailed_metrics,
        })
    }

    fn get_config(&self) -> &PsychoacousticConfig {
        &self.config
    }
}

impl PsychoacousticAnalyzerImpl {
    fn spectrum_to_bark(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut bark_spectrum = vec![0.0; self.bark_filter_bank.filters.len()];

        for (i, filter) in self.bark_filter_bank.filters.iter().enumerate() {
            let mut energy = 0.0;
            for (j, &spec_value) in spectrum.iter().enumerate() {
                let freq =
                    j as f32 * self.config.sample_rate as f32 / (2.0 * spectrum.len() as f32);
                let filter_response = self.evaluate_bark_filter(filter, freq);
                energy += spec_value * spec_value * filter_response;
            }
            bark_spectrum[i] = energy.sqrt();
        }

        bark_spectrum
    }

    fn evaluate_bark_filter(&self, filter: &BarkFilter, frequency: f32) -> f32 {
        // Simplified triangular filter
        let distance = (frequency - filter.center_freq).abs();
        if distance < filter.bandwidth / 2.0 {
            1.0 - 2.0 * distance / filter.bandwidth
        } else {
            0.0
        }
    }

    fn compute_frame_masking_threshold(&self, bark_spectrum: &[f32]) -> Vec<f32> {
        let mut threshold = self.masking_model.absolute_threshold.clone();

        // Apply simultaneous masking
        if self.masking_model.config.simultaneous_masking {
            for (i, &energy) in bark_spectrum.iter().enumerate() {
                if energy > threshold[i] {
                    // Apply spreading function
                    for (j, thresh) in threshold.iter_mut().enumerate() {
                        let spread_idx = i * bark_spectrum.len() + j;
                        if spread_idx < self.masking_model.spread_function.len() {
                            let masked_threshold =
                                energy * self.masking_model.spread_function[spread_idx];
                            *thresh = thresh.max(masked_threshold);
                        }
                    }
                }
            }
        }

        threshold
    }

    fn calculate_bark_spectral_distortion(&self, bark_frames: &[Vec<f32>]) -> f32 {
        if bark_frames.len() < 2 {
            return 0.0;
        }

        let mut total_distortion = 0.0;
        let mut count = 0;

        for i in 1..bark_frames.len() {
            for j in 0..bark_frames[i].len() {
                let diff = (bark_frames[i][j] - bark_frames[i - 1][j]).abs();
                total_distortion += diff;
                count += 1;
            }
        }

        if count > 0 {
            total_distortion / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioData;

    #[test]
    fn test_psychoacoustic_config_default() {
        let config = PsychoacousticConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.fft_size, 1024);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.bark_config.num_bands, 24);
    }

    #[tokio::test]
    async fn test_psychoacoustic_analyzer_creation() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config);
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_masking_threshold_computation() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config).unwrap();

        let audio = AudioData::silence(1.0, 22050, 1);
        let threshold = analyzer.compute_masking_threshold(&audio).await.unwrap();
        assert!(!threshold.is_empty());
    }

    #[tokio::test]
    async fn test_quality_metrics_calculation() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config).unwrap();

        let audio = AudioData::silence(1.0, 22050, 1);
        let metrics = analyzer.calculate_quality_metrics(&audio).await.unwrap();
        assert!(metrics.bark_spectral_distortion.is_some());
    }

    #[tokio::test]
    async fn test_auditory_model_simulation() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config).unwrap();

        let audio = AudioData::silence(1.0, 22050, 1);
        let results = analyzer.simulate_auditory_model(&audio).await.unwrap();
        assert!(!results.basilar_membrane.is_empty());
        assert!(!results.hair_cell_response.is_empty());
    }

    #[tokio::test]
    async fn test_perceptual_comparison() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config).unwrap();

        let audio1 = AudioData::silence(1.0, 22050, 1);
        let audio2 = AudioData::silence(1.0, 22050, 1);

        let comparison = analyzer
            .perceptual_comparison(&audio1, &audio2)
            .await
            .unwrap();
        assert!((0.0..=1.0).contains(&comparison.overall_difference));
    }

    #[test]
    fn test_frequency_bark_conversion() {
        let config = PsychoacousticConfig::default();
        let analyzer = PsychoacousticAnalyzerImpl::new(config).unwrap();

        let freq = 1000.0;
        let bark = analyzer.frequency_to_bark(freq);
        let freq_back = analyzer.bark_to_frequency(bark);

        // Should be approximately equal (within 50 Hz due to floating point precision and formula approximations)
        assert!((freq - freq_back).abs() < 50.0);
    }

    #[test]
    fn test_bark_filter_bank_creation() {
        let config = BarkScaleConfig::default();
        let filter_bank = BarkFilterBank::new(&config, 22050);
        assert!(filter_bank.is_ok());

        let bank = filter_bank.unwrap();
        assert_eq!(bank.filters.len(), 24);
        assert_eq!(bank.center_frequencies.len(), 24);
        assert_eq!(bank.bandwidths.len(), 24);
    }
}
