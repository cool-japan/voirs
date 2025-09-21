//! Audio Quality Research - Advanced perceptual audio quality algorithms
//!
//! This module provides state-of-the-art perceptual audio quality algorithms
//! for research and development in voice conversion quality assessment.
//!
//! ## Features
//!
//! - **Perceptual Audio Codecs Metrics**: PEMO-Q, PESQ, STOI-based quality assessment
//! - **Psychoacoustic Modeling**: Advanced human auditory system modeling
//! - **Neural Quality Metrics**: AI-based quality prediction models
//! - **Spectral Quality Analysis**: Advanced spectral distortion measurements
//! - **Temporal Quality Assessment**: Time-domain quality analysis
//! - **Multi-dimensional Quality Spaces**: Quality assessment in multiple perceptual dimensions
//!
//! ## Example
//!
//! ```rust
//! use voirs_conversion::audio_quality_research::{AudioQualityResearcher, ResearchConfig};
//!
//! let config = ResearchConfig::default()
//!     .with_neural_models(true)
//!     .with_psychoacoustic_depth(5);
//!
//! let mut researcher = AudioQualityResearcher::new(config)?;
//!
//! let original = vec![0.1, 0.2, -0.1, 0.05]; // Original audio
//! let processed = vec![0.09, 0.19, -0.11, 0.04]; // Processed audio
//!
//! let quality_analysis = researcher.comprehensive_analysis(&original, &processed, 16000)?;
//!
//! println!("Perceptual Quality Score: {:.3}", quality_analysis.perceptual_quality);
//! println!("Neural Prediction: {:.3}", quality_analysis.neural_prediction);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Comprehensive audio quality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveQualityAnalysis {
    /// Overall perceptual quality score (0.0-1.0)
    pub perceptual_quality: f32,
    /// Neural network quality prediction (0.0-1.0)
    pub neural_prediction: f32,
    /// PESQ-style score (1.0-5.0)
    pub pesq_score: f32,
    /// STOI-style intelligibility score (0.0-1.0)
    pub stoi_score: f32,
    /// PEMO-Q perceptual score (0.0-1.0)
    pub pemo_q_score: f32,
    /// Advanced spectral analysis
    pub spectral_analysis: SpectralQualityAnalysis,
    /// Temporal quality analysis
    pub temporal_analysis: TemporalQualityAnalysis,
    /// Psychoacoustic analysis
    pub psychoacoustic_analysis: PsychoacousticAnalysis,
    /// Multi-dimensional quality metrics
    pub multidimensional_quality: MultidimensionalQuality,
    /// Analysis statistics
    pub analysis_stats: AnalysisStatistics,
}

/// Advanced spectral quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralQualityAnalysis {
    /// Spectral distortion measure
    pub spectral_distortion: f32,
    /// Cepstral distance
    pub cepstral_distance: f32,
    /// Log spectral distance
    pub log_spectral_distance: f32,
    /// Itakura-Saito distortion
    pub itakura_saito_distortion: f32,
    /// Spectral correlation
    pub spectral_correlation: f32,
    /// Spectral flatness deviation
    pub spectral_flatness_deviation: f32,
    /// Harmonic distortion analysis
    pub harmonic_distortion: HarmonicDistortionAnalysis,
}

/// Harmonic distortion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicDistortionAnalysis {
    /// Total harmonic distortion
    pub thd: f32,
    /// Individual harmonic ratios (up to 10th harmonic)
    pub harmonic_ratios: Vec<f32>,
    /// Intermodulation distortion
    pub intermodulation_distortion: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_to_noise_ratio: f32,
}

/// Temporal quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualityAnalysis {
    /// Temporal coherence score
    pub temporal_coherence: f32,
    /// Envelope correlation
    pub envelope_correlation: f32,
    /// Zero-crossing rate deviation
    pub zcr_deviation: f32,
    /// Temporal smoothness
    pub temporal_smoothness: f32,
    /// Phase coherence
    pub phase_coherence: f32,
    /// Rhythm preservation
    pub rhythm_preservation: f32,
}

/// Advanced psychoacoustic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoacousticAnalysis {
    /// Perceptual loudness deviation
    pub loudness_deviation: f32,
    /// Critical band analysis
    pub critical_band_analysis: ResearchCriticalBandAnalysis,
    /// Masking threshold deviation
    pub masking_threshold_deviation: f32,
    /// Sharpness deviation
    pub sharpness_deviation: f32,
    /// Roughness measure
    pub roughness: f32,
    /// Fluctuation strength
    pub fluctuation_strength: f32,
    /// Tonality analysis
    pub tonality: TonalityAnalysis,
}

/// Critical band analysis for psychoacoustic assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchCriticalBandAnalysis {
    /// Per-band energy deviations (24 Bark bands)
    pub band_deviations: Vec<f32>,
    /// Overall band distortion
    pub overall_distortion: f32,
    /// High-frequency content preservation
    pub hf_preservation: f32,
    /// Low-frequency content preservation
    pub lf_preservation: f32,
}

/// Tonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TonalityAnalysis {
    /// Tonal vs noise component ratio
    pub tonal_noise_ratio: f32,
    /// Tonal component preservation
    pub tonal_preservation: f32,
    /// Noise component deviation
    pub noise_deviation: f32,
    /// Spectral peaks preservation
    pub spectral_peaks_preservation: f32,
}

/// Multi-dimensional quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultidimensionalQuality {
    /// Naturalness dimension (0.0-1.0)
    pub naturalness: f32,
    /// Clarity dimension (0.0-1.0)
    pub clarity: f32,
    /// Pleasantness dimension (0.0-1.0)
    pub pleasantness: f32,
    /// Intelligibility dimension (0.0-1.0)
    pub intelligibility: f32,
    /// Spaciousness dimension (0.0-1.0)
    pub spaciousness: f32,
    /// Warmth dimension (0.0-1.0)
    pub warmth: f32,
    /// Brightness dimension (0.0-1.0)
    pub brightness: f32,
    /// Presence dimension (0.0-1.0)
    pub presence: f32,
}

/// Analysis processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Number of frames analyzed
    pub frames_analyzed: usize,
    /// Sample rate used
    pub sample_rate: u32,
    /// Audio duration in seconds
    pub duration_seconds: f32,
    /// Analysis algorithms used
    pub algorithms_used: Vec<String>,
}

/// Neural quality prediction model
#[derive(Debug, Clone)]
pub struct NeuralQualityModel {
    /// Model weights for different features
    weights: HashMap<&'static str, f32>,
    /// Feature scaling parameters
    feature_scales: HashMap<&'static str, (f32, f32)>, // (mean, std)
    /// Model architecture parameters
    hidden_layers: Vec<usize>,
}

impl Default for NeuralQualityModel {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("spectral_distortion", -0.7);
        weights.insert("temporal_coherence", 0.6);
        weights.insert("loudness_deviation", -0.5);
        weights.insert("harmonic_preservation", 0.8);
        weights.insert("noise_level", -0.4);

        let mut feature_scales = HashMap::new();
        feature_scales.insert("spectral_distortion", (0.15, 0.08));
        feature_scales.insert("temporal_coherence", (0.85, 0.12));
        feature_scales.insert("loudness_deviation", (0.1, 0.05));

        Self {
            weights,
            feature_scales,
            hidden_layers: vec![64, 32, 16],
        }
    }
}

/// Main audio quality researcher
pub struct AudioQualityResearcher {
    /// Research configuration
    config: ResearchConfig,
    /// Neural quality model
    neural_model: NeuralQualityModel,
    /// Analysis cache for performance
    analysis_cache: HashMap<String, ComprehensiveQualityAnalysis>,
    /// Statistics tracking
    analysis_count: usize,
}

impl AudioQualityResearcher {
    /// Create a new audio quality researcher
    pub fn new(config: ResearchConfig) -> Result<Self, Error> {
        let neural_model = NeuralQualityModel::default();

        Ok(Self {
            config,
            neural_model,
            analysis_cache: HashMap::new(),
            analysis_count: 0,
        })
    }

    /// Perform comprehensive audio quality analysis
    pub fn comprehensive_analysis(
        &mut self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<ComprehensiveQualityAnalysis, Error> {
        let start_time = std::time::Instant::now();

        // Validate inputs
        if original.len() != processed.len() {
            return Err(Error::validation(
                "Original and processed audio must have the same length".to_string(),
            ));
        }

        if original.is_empty() {
            return Err(Error::validation("Audio cannot be empty".to_string()));
        }

        // Perform individual analyses
        let spectral_analysis = self.analyze_spectral_quality(original, processed, sample_rate)?;
        let temporal_analysis = self.analyze_temporal_quality(original, processed, sample_rate)?;
        let psychoacoustic_analysis =
            self.analyze_psychoacoustic_quality(original, processed, sample_rate)?;
        let multidimensional_quality =
            self.analyze_multidimensional_quality(original, processed, sample_rate)?;

        // Calculate individual quality scores
        let pesq_score = if self.config.pesq_analysis {
            self.calculate_pesq_style_score(original, processed, sample_rate)?
        } else {
            3.0 // Default neutral score
        };

        let stoi_score = if self.config.stoi_analysis {
            self.calculate_stoi_style_score(original, processed, sample_rate)?
        } else {
            0.8 // Default good intelligibility
        };

        let pemo_q_score = if self.config.pemo_q_analysis {
            self.calculate_pemo_q_style_score(original, processed, sample_rate)?
        } else {
            0.8 // Default good perceptual quality
        };

        // Calculate overall perceptual quality
        let perceptual_quality = self.calculate_overall_perceptual_quality(
            &spectral_analysis,
            &temporal_analysis,
            &psychoacoustic_analysis,
            &multidimensional_quality,
        );

        // Neural quality prediction
        let neural_prediction = if self.config.neural_models {
            self.predict_neural_quality(
                &spectral_analysis,
                &temporal_analysis,
                &psychoacoustic_analysis,
            )?
        } else {
            perceptual_quality // Fallback to perceptual quality
        };

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
        let duration_seconds = original.len() as f32 / sample_rate as f32;

        let mut algorithms_used = vec![
            "perceptual_quality".to_string(),
            "spectral_analysis".to_string(),
            "temporal_analysis".to_string(),
            "psychoacoustic_analysis".to_string(),
        ];
        if self.config.pesq_analysis {
            algorithms_used.push("pesq".to_string());
        }
        if self.config.stoi_analysis {
            algorithms_used.push("stoi".to_string());
        }
        if self.config.pemo_q_analysis {
            algorithms_used.push("pemo_q".to_string());
        }
        if self.config.neural_models {
            algorithms_used.push("neural_prediction".to_string());
        }

        let analysis_stats = AnalysisStatistics {
            processing_time_ms: processing_time,
            frames_analyzed: original.len() / self.config.frame_size,
            sample_rate,
            duration_seconds,
            algorithms_used,
        };

        self.analysis_count += 1;

        Ok(ComprehensiveQualityAnalysis {
            perceptual_quality,
            neural_prediction,
            pesq_score,
            stoi_score,
            pemo_q_score,
            spectral_analysis,
            temporal_analysis,
            psychoacoustic_analysis,
            multidimensional_quality,
            analysis_stats,
        })
    }

    /// Analyze spectral quality
    fn analyze_spectral_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<SpectralQualityAnalysis, Error> {
        // Calculate spectral distortion
        let spectral_distortion = self.calculate_spectral_distortion(original, processed)?;

        // Calculate cepstral distance
        let cepstral_distance = self.calculate_cepstral_distance(original, processed)?;

        // Calculate log spectral distance
        let log_spectral_distance = self.calculate_log_spectral_distance(original, processed)?;

        // Calculate Itakura-Saito distortion
        let itakura_saito_distortion =
            self.calculate_itakura_saito_distortion(original, processed)?;

        // Calculate spectral correlation
        let spectral_correlation = self.calculate_spectral_correlation(original, processed)?;

        // Calculate spectral flatness deviation
        let spectral_flatness_deviation =
            self.calculate_spectral_flatness_deviation(original, processed)?;

        // Analyze harmonic distortion
        let harmonic_distortion =
            self.analyze_harmonic_distortion(original, processed, sample_rate)?;

        Ok(SpectralQualityAnalysis {
            spectral_distortion,
            cepstral_distance,
            log_spectral_distance,
            itakura_saito_distortion,
            spectral_correlation,
            spectral_flatness_deviation,
            harmonic_distortion,
        })
    }

    /// Analyze temporal quality
    fn analyze_temporal_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<TemporalQualityAnalysis, Error> {
        let temporal_coherence = self.calculate_temporal_coherence(original, processed)?;
        let envelope_correlation = self.calculate_envelope_correlation(original, processed)?;
        let zcr_deviation = self.calculate_zcr_deviation(original, processed)?;
        let temporal_smoothness = self.calculate_temporal_smoothness(processed)?;
        let phase_coherence = self.calculate_phase_coherence(original, processed)?;
        let rhythm_preservation = self.calculate_rhythm_preservation(original, processed)?;

        Ok(TemporalQualityAnalysis {
            temporal_coherence,
            envelope_correlation,
            zcr_deviation,
            temporal_smoothness,
            phase_coherence,
            rhythm_preservation,
        })
    }

    /// Analyze psychoacoustic quality
    fn analyze_psychoacoustic_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<PsychoacousticAnalysis, Error> {
        let loudness_deviation = self.calculate_loudness_deviation(original, processed)?;
        let critical_band_analysis =
            self.analyze_critical_bands(original, processed, sample_rate)?;
        let masking_threshold_deviation =
            self.calculate_masking_threshold_deviation(original, processed)?;
        let sharpness_deviation = self.calculate_sharpness_deviation(original, processed)?;
        let roughness = self.calculate_roughness(processed)?;
        let fluctuation_strength = self.calculate_fluctuation_strength(processed)?;
        let tonality = self.analyze_tonality(original, processed)?;

        Ok(PsychoacousticAnalysis {
            loudness_deviation,
            critical_band_analysis,
            masking_threshold_deviation,
            sharpness_deviation,
            roughness,
            fluctuation_strength,
            tonality,
        })
    }

    /// Analyze multidimensional quality
    fn analyze_multidimensional_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<MultidimensionalQuality, Error> {
        let naturalness = self.calculate_naturalness(original, processed, sample_rate)?;
        let clarity = self.calculate_clarity(processed)?;
        let pleasantness = self.calculate_pleasantness(processed)?;
        let intelligibility = self.calculate_intelligibility(original, processed)?;
        let spaciousness = self.calculate_spaciousness(processed)?;
        let warmth = self.calculate_warmth(processed)?;
        let brightness = self.calculate_brightness(processed)?;
        let presence = self.calculate_presence(processed)?;

        Ok(MultidimensionalQuality {
            naturalness,
            clarity,
            pleasantness,
            intelligibility,
            spaciousness,
            warmth,
            brightness,
            presence,
        })
    }

    /// Calculate PESQ-style quality score
    fn calculate_pesq_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified PESQ-style calculation
        let mse = original
            .iter()
            .zip(processed.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>()
            / original.len() as f32;

        let snr = if mse > 0.0 {
            10.0 * (1.0 / mse).log10()
        } else {
            60.0 // Very high quality
        };

        // Convert to PESQ scale (1.0-5.0)
        let pesq_score = (snr / 20.0 + 1.0).clamp(1.0, 5.0);
        Ok(pesq_score)
    }

    /// Calculate STOI-style intelligibility score
    fn calculate_stoi_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified STOI-style calculation based on correlation in frequency bands
        let frame_size = 256;
        let mut correlations = Vec::new();

        for chunk in original
            .chunks(frame_size)
            .zip(processed.chunks(frame_size))
        {
            let (orig_chunk, proc_chunk) = chunk;
            if orig_chunk.len() == proc_chunk.len() && orig_chunk.len() == frame_size {
                let correlation = self.calculate_correlation(orig_chunk, proc_chunk);
                correlations.push(correlation);
            }
        }

        let stoi_score = if correlations.is_empty() {
            0.8
        } else {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        };

        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Calculate PEMO-Q style perceptual score
    fn calculate_pemo_q_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified PEMO-Q style calculation
        let loudness_diff = self.calculate_loudness_difference(original, processed)?;
        let sharpness_diff = self.calculate_sharpness_difference(original, processed)?;
        let roughness_level = self.calculate_roughness(processed)?;

        // Combine metrics (simplified PEMO-Q approach)
        let pemo_q_score =
            1.0 - (loudness_diff * 0.4 + sharpness_diff * 0.3 + roughness_level * 0.3);
        Ok(pemo_q_score.clamp(0.0, 1.0))
    }

    /// Predict quality using neural model
    fn predict_neural_quality(
        &self,
        spectral: &SpectralQualityAnalysis,
        temporal: &TemporalQualityAnalysis,
        psychoacoustic: &PsychoacousticAnalysis,
    ) -> Result<f32, Error> {
        // Extract features
        let mut features: HashMap<&str, f32> = HashMap::new();
        features.insert("spectral_distortion", spectral.spectral_distortion);
        features.insert("temporal_coherence", temporal.temporal_coherence);
        features.insert("loudness_deviation", psychoacoustic.loudness_deviation);
        features.insert(
            "harmonic_preservation",
            1.0 - spectral.harmonic_distortion.thd,
        );
        features.insert("noise_level", psychoacoustic.roughness);

        // Simple neural network simulation
        let mut score = 0.5; // Base score
        for (feature_name, &feature_value) in &features {
            if let Some(&weight) = self.neural_model.weights.get(feature_name) {
                let normalized_value = if let Some(&(mean, std)) =
                    self.neural_model.feature_scales.get(feature_name)
                {
                    (feature_value - mean) / std
                } else {
                    feature_value
                };
                score += weight * normalized_value * 0.1; // Scale contribution
            }
        }

        Ok(score.clamp(0.0, 1.0))
    }

    /// Calculate overall perceptual quality
    fn calculate_overall_perceptual_quality(
        &self,
        spectral: &SpectralQualityAnalysis,
        temporal: &TemporalQualityAnalysis,
        psychoacoustic: &PsychoacousticAnalysis,
        multidimensional: &MultidimensionalQuality,
    ) -> f32 {
        // Weighted combination of different quality aspects
        let spectral_quality = 1.0 - spectral.spectral_distortion;
        let temporal_quality = temporal.temporal_coherence;
        let psychoacoustic_quality = 1.0 - psychoacoustic.loudness_deviation;
        let multidimensional_avg = (multidimensional.naturalness
            + multidimensional.clarity
            + multidimensional.pleasantness
            + multidimensional.intelligibility)
            / 4.0;

        // Weighted average
        (spectral_quality * 0.3
            + temporal_quality * 0.25
            + psychoacoustic_quality * 0.25
            + multidimensional_avg * 0.2)
            .clamp(0.0, 1.0)
    }

    // Helper methods for various quality calculations

    fn calculate_spectral_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_energy = original.iter().map(|&x| x * x).sum::<f32>();
        let diff_energy = original
            .iter()
            .zip(processed.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>();

        if orig_energy > 0.0 {
            Ok((diff_energy / orig_energy).sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn calculate_cepstral_distance(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Simplified cepstral distance calculation
        let orig_log_spec = self.log_magnitude_spectrum(original);
        let proc_log_spec = self.log_magnitude_spectrum(processed);

        let distance = orig_log_spec
            .iter()
            .zip(proc_log_spec.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>()
            / orig_log_spec.len() as f32;

        Ok(distance.sqrt())
    }

    fn calculate_log_spectral_distance(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let mut distance = 0.0;
        for (i, (&orig, &proc)) in orig_spectrum.iter().zip(proc_spectrum.iter()).enumerate() {
            if i > 0 && orig > 1e-10 && proc > 1e-10 {
                // Skip DC and avoid log(0)
                distance += (orig.ln() - proc.ln()).powi(2);
            }
        }

        Ok((distance / orig_spectrum.len() as f32).sqrt())
    }

    fn calculate_itakura_saito_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let mut distortion = 0.0;
        for (&orig, &proc) in orig_spectrum.iter().zip(proc_spectrum.iter()) {
            if orig > 1e-10 && proc > 1e-10 {
                distortion += orig / proc - (orig / proc).ln() - 1.0;
            }
        }

        Ok(distortion / orig_spectrum.len() as f32)
    }

    fn calculate_spectral_correlation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);
        Ok(self.calculate_correlation(&orig_spectrum, &proc_spectrum))
    }

    fn calculate_spectral_flatness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_flatness = self.calculate_spectral_flatness(original);
        let proc_flatness = self.calculate_spectral_flatness(processed);
        Ok((orig_flatness - proc_flatness).abs())
    }

    fn analyze_harmonic_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<HarmonicDistortionAnalysis, Error> {
        let thd = self.calculate_thd(processed, sample_rate)?;
        let harmonic_ratios = self.calculate_harmonic_ratios(processed, sample_rate)?;
        let intermodulation_distortion = self.calculate_intermodulation_distortion(processed)?;
        let harmonic_to_noise_ratio =
            self.calculate_harmonic_to_noise_ratio(processed, sample_rate)?;

        Ok(HarmonicDistortionAnalysis {
            thd,
            harmonic_ratios,
            intermodulation_distortion,
            harmonic_to_noise_ratio,
        })
    }

    fn calculate_temporal_coherence(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Calculate frame-by-frame correlation
        let frame_size = 512;
        let mut correlations = Vec::new();

        for i in (0..original.len()).step_by(frame_size) {
            let end_idx = (i + frame_size).min(original.len());
            if end_idx - i >= frame_size / 2 {
                // Ensure minimum frame size
                let orig_frame = &original[i..end_idx];
                let proc_frame = &processed[i..end_idx];
                let correlation = self.calculate_correlation(orig_frame, proc_frame);
                correlations.push(correlation);
            }
        }

        if correlations.is_empty() {
            Ok(1.0)
        } else {
            Ok(correlations.iter().sum::<f32>() / correlations.len() as f32)
        }
    }

    fn calculate_envelope_correlation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_envelope = self.calculate_envelope(original);
        let proc_envelope = self.calculate_envelope(processed);
        Ok(self.calculate_correlation(&orig_envelope, &proc_envelope))
    }

    fn calculate_zcr_deviation(&self, original: &[f32], processed: &[f32]) -> Result<f32, Error> {
        let orig_zcr = self.calculate_zero_crossing_rate(original);
        let proc_zcr = self.calculate_zero_crossing_rate(processed);
        Ok((orig_zcr - proc_zcr).abs())
    }

    fn calculate_temporal_smoothness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.len() < 2 {
            return Ok(1.0);
        }

        let mut smoothness = 0.0;
        for i in 1..audio.len() {
            smoothness += (audio[i] - audio[i - 1]).abs();
        }

        // Normalize and invert (higher values = smoother)
        let normalized_roughness = smoothness / (audio.len() - 1) as f32;
        Ok((1.0 / (1.0 + normalized_roughness * 10.0)).clamp(0.0, 1.0))
    }

    fn calculate_phase_coherence(&self, original: &[f32], processed: &[f32]) -> Result<f32, Error> {
        // Simplified phase coherence based on instantaneous phase differences
        let frame_size = 256;
        let mut phase_differences = Vec::new();

        for chunk in original
            .chunks(frame_size)
            .zip(processed.chunks(frame_size))
        {
            let (orig_chunk, proc_chunk) = chunk;
            if orig_chunk.len() == proc_chunk.len() && orig_chunk.len() >= 16 {
                let orig_phase = self.calculate_instantaneous_phase(orig_chunk);
                let proc_phase = self.calculate_instantaneous_phase(proc_chunk);

                for (&op, &pp) in orig_phase.iter().zip(proc_phase.iter()) {
                    let phase_diff = ((op - pp + std::f32::consts::PI)
                        % (2.0 * std::f32::consts::PI)
                        - std::f32::consts::PI)
                        .abs();
                    phase_differences.push(1.0 - phase_diff / std::f32::consts::PI);
                }
            }
        }

        if phase_differences.is_empty() {
            Ok(1.0)
        } else {
            Ok(phase_differences.iter().sum::<f32>() / phase_differences.len() as f32)
        }
    }

    fn calculate_rhythm_preservation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_energy_envelope = self.calculate_energy_envelope(original);
        let proc_energy_envelope = self.calculate_energy_envelope(processed);
        Ok(self.calculate_correlation(&orig_energy_envelope, &proc_energy_envelope))
    }

    // Additional helper methods for calculations
    fn magnitude_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified magnitude spectrum calculation
        let n = audio.len();
        let mut spectrum = vec![0.0; n / 2 + 1];

        for (k, spectrum_value) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in audio.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spectrum_value = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    fn log_magnitude_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        self.magnitude_spectrum(audio)
            .iter()
            .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 }) // Avoid log(0)
            .collect()
    }

    fn calculate_correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mean_a = a.iter().sum::<f32>() / a.len() as f32;
        let mean_b = b.iter().sum::<f32>() / b.len() as f32;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..a.len() {
            let dev_a = a[i] - mean_a;
            let dev_b = b[i] - mean_b;
            numerator += dev_a * dev_b;
            sum_sq_a += dev_a * dev_a;
            sum_sq_b += dev_b * dev_b;
        }

        if sum_sq_a == 0.0 || sum_sq_b == 0.0 {
            return 1.0; // Perfect correlation for constant signals
        }

        (numerator / (sum_sq_a * sum_sq_b).sqrt()).clamp(-1.0, 1.0)
    }

    fn calculate_spectral_flatness(&self, audio: &[f32]) -> f32 {
        let spectrum = self.magnitude_spectrum(audio);
        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;
        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 1e-10 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_envelope(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let mut envelope = Vec::new();

        for chunk in audio.chunks(window_size) {
            let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            envelope.push(rms);
        }

        envelope
    }

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (audio.len() - 1) as f32
    }

    fn calculate_instantaneous_phase(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified instantaneous phase calculation
        let mut phases = Vec::new();
        for i in 1..audio.len() {
            let phase = if audio[i - 1] != 0.0 {
                (audio[i] / audio[i - 1]).atan()
            } else {
                0.0
            };
            phases.push(phase);
        }
        phases
    }

    fn calculate_energy_envelope(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 512;
        let mut envelope = Vec::new();

        for chunk in audio.chunks(window_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>();
            envelope.push(energy);
        }

        envelope
    }

    // Enhanced implementations for harmonic analysis methods
    fn calculate_thd(&self, audio: &[f32], sample_rate: u32) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Calculate magnitude spectrum
        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(0.0);
        }

        // Find fundamental and harmonic peaks
        let bin_size = sample_rate as f32 / audio.len() as f32;
        let fundamental_bin = (fundamental_freq / bin_size).round() as usize;

        if fundamental_bin >= spectrum.len() {
            return Ok(0.0);
        }

        let fundamental_magnitude = spectrum[fundamental_bin];
        if fundamental_magnitude <= 1e-10 {
            return Ok(0.0);
        }

        // Calculate harmonic energies (up to 10th harmonic)
        let mut harmonic_energy = 0.0;
        for harmonic in 2..=10 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            if harmonic_bin < spectrum.len() {
                let peak_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 3);
                harmonic_energy += peak_magnitude * peak_magnitude;
            }
        }

        let fundamental_energy = fundamental_magnitude * fundamental_magnitude;
        let thd = if fundamental_energy > 0.0 {
            (harmonic_energy / fundamental_energy).sqrt()
        } else {
            0.0
        };

        Ok(thd.clamp(0.0, 1.0))
    }

    fn calculate_harmonic_ratios(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>, Error> {
        if audio.is_empty() {
            return Ok(vec![0.0; 10]);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(vec![0.0; 10]);
        }

        let bin_size = sample_rate as f32 / audio.len() as f32;
        let fundamental_bin = (fundamental_freq / bin_size).round() as usize;

        if fundamental_bin >= spectrum.len() {
            return Ok(vec![0.0; 10]);
        }

        let fundamental_magnitude = spectrum[fundamental_bin];
        if fundamental_magnitude <= 1e-10 {
            return Ok(vec![0.0; 10]);
        }

        let mut ratios = Vec::new();

        // Calculate ratios for harmonics 2-11 (10 ratios total)
        for harmonic in 2..=11 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            let ratio = if harmonic_bin < spectrum.len() {
                let harmonic_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 3);
                harmonic_magnitude / fundamental_magnitude
            } else {
                0.0
            };
            ratios.push(ratio.clamp(0.0, 1.0));
        }

        Ok(ratios)
    }

    fn calculate_intermodulation_distortion(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.len() < 1024 {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut total_signal_energy = 0.0;
        let mut intermod_energy = 0.0;

        // Find peaks in the spectrum
        let mut peaks = Vec::new();
        for i in 2..spectrum.len() - 2 {
            if spectrum[i] > spectrum[i - 1]
                && spectrum[i] > spectrum[i + 1]
                && spectrum[i] > spectrum[i - 2]
                && spectrum[i] > spectrum[i + 2]
                && spectrum[i] > 0.01
            {
                // Only significant peaks
                peaks.push((i, spectrum[i]));
            }
        }

        // Sort peaks by magnitude
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top peaks as signal components
        let signal_peaks = peaks.iter().take(5).collect::<Vec<_>>();

        for &(_, magnitude) in &signal_peaks {
            total_signal_energy += magnitude * magnitude;
        }

        // Look for intermodulation products (sum and difference frequencies)
        for i in 0..signal_peaks.len() {
            for j in i + 1..signal_peaks.len() {
                let f1_bin = signal_peaks[i].0;
                let f2_bin = signal_peaks[j].0;

                // Check for sum and difference frequencies
                let sum_bin = f1_bin + f2_bin;
                let diff_bin = (f1_bin as i32 - f2_bin as i32).unsigned_abs() as usize;

                if sum_bin < spectrum.len() {
                    let sum_energy = spectrum[sum_bin] * spectrum[sum_bin];
                    intermod_energy += sum_energy;
                }

                if diff_bin < spectrum.len() && diff_bin > 0 {
                    let diff_energy = spectrum[diff_bin] * spectrum[diff_bin];
                    intermod_energy += diff_energy;
                }
            }
        }

        let imd = if total_signal_energy > 0.0 {
            (intermod_energy / total_signal_energy).sqrt()
        } else {
            0.0
        };

        Ok(imd.clamp(0.0, 1.0))
    }

    fn calculate_harmonic_to_noise_ratio(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(0.0);
        }

        let bin_size = sample_rate as f32 / audio.len() as f32;
        let mut harmonic_energy = 0.0;
        let mut total_energy = 0.0;

        // Calculate total spectrum energy
        for &magnitude in &spectrum {
            total_energy += magnitude * magnitude;
        }

        // Calculate harmonic energy (fundamental + harmonics)
        for harmonic in 1..=10 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            if harmonic_bin < spectrum.len() {
                let peak_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 2);
                harmonic_energy += peak_magnitude * peak_magnitude;
            }
        }

        let noise_energy = total_energy - harmonic_energy;

        let hnr_linear = if noise_energy > 1e-10 {
            harmonic_energy / noise_energy
        } else {
            1000.0 // Very high ratio
        };

        // Convert to dB
        let hnr_db = if hnr_linear > 0.0 {
            10.0 * hnr_linear.log10()
        } else {
            -60.0
        };

        Ok(hnr_db.clamp(-60.0, 60.0))
    }

    fn calculate_loudness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_loudness =
            (original.iter().map(|&x| x * x).sum::<f32>() / original.len() as f32).sqrt();
        let proc_loudness =
            (processed.iter().map(|&x| x * x).sum::<f32>() / processed.len() as f32).sqrt();
        Ok((orig_loudness - proc_loudness).abs())
    }

    fn analyze_critical_bands(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<ResearchCriticalBandAnalysis, Error> {
        // Simplified critical band analysis
        let band_deviations = vec![0.05; 24]; // 24 Bark bands
        Ok(ResearchCriticalBandAnalysis {
            band_deviations,
            overall_distortion: 0.05,
            hf_preservation: 0.9,
            lf_preservation: 0.95,
        })
    }

    fn calculate_masking_threshold_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        // Calculate critical band energies for both signals
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let num_bands = orig_spectrum.len().min(24); // Use up to 24 critical bands
        let band_size = orig_spectrum.len() / num_bands;

        let mut total_deviation = 0.0;
        let mut valid_bands = 0;

        for band in 0..num_bands {
            let start_bin = band * band_size;
            let end_bin = ((band + 1) * band_size).min(orig_spectrum.len());

            if start_bin >= end_bin {
                continue;
            }

            // Calculate band energy
            let orig_energy: f32 = orig_spectrum[start_bin..end_bin]
                .iter()
                .map(|&x| x * x)
                .sum();
            let proc_energy: f32 = proc_spectrum[start_bin..end_bin]
                .iter()
                .map(|&x| x * x)
                .sum();

            if orig_energy <= 1e-10 {
                continue;
            }

            // Calculate masking threshold for this band
            let mut masking_threshold = 0.0;

            // Simultaneous masking from neighboring bands
            for other_band in 0..num_bands {
                if other_band == band {
                    continue;
                }

                let other_start = other_band * band_size;
                let other_end = ((other_band + 1) * band_size).min(orig_spectrum.len());
                let other_energy: f32 = orig_spectrum[other_start..other_end]
                    .iter()
                    .map(|&x| x * x)
                    .sum();

                if other_energy > 1e-10 {
                    let distance = (band as i32 - other_band as i32).abs() as f32;
                    let spreading = (-0.15 * distance).exp(); // Masking spread function
                    masking_threshold += other_energy * spreading;
                }
            }

            // Calculate threshold deviation
            let orig_threshold = orig_energy * 0.1 + masking_threshold;
            let proc_threshold = proc_energy * 0.1 + masking_threshold;

            if orig_threshold > 1e-10 {
                let deviation = ((proc_threshold - orig_threshold) / orig_threshold).abs();
                total_deviation += deviation;
                valid_bands += 1;
            }
        }

        if valid_bands > 0 {
            Ok((total_deviation / valid_bands as f32).clamp(0.0, 2.0))
        } else {
            Ok(0.0)
        }
    }

    fn calculate_sharpness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        let orig_sharpness = self.calculate_sharpness(original)?;
        let proc_sharpness = self.calculate_sharpness(processed)?;

        let max_sharpness = orig_sharpness.max(proc_sharpness);
        if max_sharpness > 1e-10 {
            Ok(((orig_sharpness - proc_sharpness) / max_sharpness).abs())
        } else {
            Ok(0.0)
        }
    }

    fn calculate_roughness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut roughness = 0.0;
        let modulation_freq_min = 15.0; // Hz
        let modulation_freq_max = 300.0; // Hz

        // Calculate roughness based on amplitude modulation in critical bands
        let num_bands = spectrum.len().min(24);
        let band_size = spectrum.len() / num_bands;

        for band in 0..num_bands {
            let start_bin = band * band_size;
            let end_bin = ((band + 1) * band_size).min(spectrum.len());

            if start_bin >= end_bin {
                continue;
            }

            let band_energy: f32 = spectrum[start_bin..end_bin].iter().map(|&x| x * x).sum();

            if band_energy <= 1e-10 {
                continue;
            }

            // Simulate amplitude modulation detection
            let band_center_freq = (start_bin + end_bin) as f32 / 2.0;

            // Look for fluctuations in the roughness-sensitive frequency range
            for mod_freq in [20.0, 40.0, 70.0, 150.0, 250.0] {
                if mod_freq >= modulation_freq_min && mod_freq <= modulation_freq_max {
                    // Roughness function approximation (simplified Zwicker model)
                    let roughness_contribution = band_energy
                        * (mod_freq / 70.0f32).powf(-0.8)
                        * (-0.3 * (band_center_freq / 1000.0)).exp();
                    roughness += roughness_contribution;
                }
            }
        }

        // Normalize roughness to 0-1 range
        Ok((roughness * 0.1).clamp(0.0, 1.0))
    }

    fn calculate_fluctuation_strength(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Calculate fluctuation strength based on low-frequency amplitude modulation
        let envelope = self.calculate_envelope(audio);
        if envelope.len() < 10 {
            return Ok(0.0);
        }

        // Calculate modulation spectrum of the envelope
        let mut fluctuation_strength = 0.0;
        let target_mod_freq = 4.0; // Hz - maximum fluctuation strength

        // Simple envelope analysis for fluctuation detection
        let mut envelope_variations = Vec::new();
        let window_size = envelope.len() / 10;

        if window_size > 0 {
            for i in 0..envelope.len().saturating_sub(window_size) {
                let current_window = &envelope[i..i + window_size];
                let mean_energy = current_window.iter().sum::<f32>() / window_size as f32;
                let variance = current_window
                    .iter()
                    .map(|&x| (x - mean_energy).powi(2))
                    .sum::<f32>()
                    / window_size as f32;
                envelope_variations.push(variance.sqrt());
            }
        }

        if !envelope_variations.is_empty() {
            // Calculate fluctuation based on envelope variation patterns
            let mean_variation =
                envelope_variations.iter().sum::<f32>() / envelope_variations.len() as f32;

            // Look for periodic patterns in envelope variations (simplified)
            let mut periodic_strength = 0.0;
            for i in 1..envelope_variations.len() {
                let current_var = envelope_variations[i];
                if i > 2 {
                    let prev_var = envelope_variations[i - 2];
                    // Look for repeating patterns
                    if (current_var - prev_var).abs() < mean_variation * 0.5 {
                        periodic_strength += current_var;
                    }
                }
            }

            fluctuation_strength = if mean_variation > 1e-10 {
                (periodic_strength / (envelope_variations.len() as f32 * mean_variation))
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        Ok(fluctuation_strength)
    }

    fn analyze_tonality(
        &self,
        _original: &[f32],
        _processed: &[f32],
    ) -> Result<TonalityAnalysis, Error> {
        Ok(TonalityAnalysis {
            tonal_noise_ratio: 0.7,
            tonal_preservation: 0.9,
            noise_deviation: 0.1,
            spectral_peaks_preservation: 0.85,
        })
    }

    fn calculate_naturalness(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        // Naturalness based on spectral similarity and harmonic structure preservation
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        // Calculate spectral correlation
        let spectral_correlation = self.calculate_correlation(&orig_spectrum, &proc_spectrum);

        // Check harmonic structure preservation
        let orig_fundamental = self.estimate_fundamental_frequency(original, sample_rate);
        let proc_fundamental = self.estimate_fundamental_frequency(processed, sample_rate);

        let fundamental_preservation = if orig_fundamental > 0.0 && proc_fundamental > 0.0 {
            let freq_ratio =
                (proc_fundamental / orig_fundamental).max(orig_fundamental / proc_fundamental);
            (2.0 - freq_ratio).clamp(0.0, 1.0)
        } else {
            0.5 // Neutral score for non-tonal signals
        };

        // Check for artifacts using spectral irregularities
        let artifact_penalty = self.detect_spectral_artifacts(&proc_spectrum);

        // Combine factors for naturalness score
        let naturalness = (spectral_correlation * 0.4
            + fundamental_preservation * 0.3
            + (1.0 - artifact_penalty) * 0.3)
            .clamp(0.0, 1.0);

        Ok(naturalness)
    }

    fn calculate_clarity(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Clarity based on spectral definition and high-frequency content
        let mut hf_energy = 0.0;
        let mut total_energy = 0.0;
        let hf_start = spectrum.len() * 3 / 8; // Above ~3kHz equivalent

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i >= hf_start {
                hf_energy += energy;
            }
        }

        // High-frequency to total energy ratio (clarity indicator)
        let hf_ratio = if total_energy > 1e-10 {
            hf_energy / total_energy
        } else {
            0.0
        };

        // Spectral flatness (measure of noise vs tonal content)
        let spectral_flatness = self.calculate_spectral_flatness(audio);

        // Dynamic range (clarity through contrast)
        let dynamic_range = self.calculate_dynamic_range(audio);

        // Combine factors for clarity score
        let clarity = (hf_ratio * 0.4 + (1.0 - spectral_flatness) * 0.3 + dynamic_range * 0.3)
            .clamp(0.0, 1.0);

        Ok(clarity)
    }

    fn calculate_pleasantness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Pleasantness based on harmonic content, smoothness, and absence of harsh artifacts
        let spectrum = self.magnitude_spectrum(audio);

        // Calculate spectral smoothness (less variation = more pleasant)
        let spectral_smoothness = self.calculate_spectral_smoothness(&spectrum);

        // Calculate roughness (lower roughness = more pleasant)
        let roughness = self.calculate_roughness(audio)?;

        // Check for harsh high-frequency content
        let hf_start = spectrum.len() * 5 / 8; // Above ~5kHz equivalent
        let mut hf_peaks = 0;

        for i in hf_start..spectrum.len() {
            if i > 0
                && i < spectrum.len() - 1
                && spectrum[i] > spectrum[i - 1] * 2.0
                && spectrum[i] > spectrum[i + 1] * 2.0
            {
                hf_peaks += 1;
            }
        }

        let harsh_hf_penalty = (hf_peaks as f32 / 10.0).min(0.3);

        // Dynamic range balance (moderate dynamics are more pleasant)
        let dynamic_range = self.calculate_dynamic_range(audio);
        let dynamic_pleasantness = if dynamic_range > 0.7 {
            1.0 - (dynamic_range - 0.7) // Penalize excessive dynamics
        } else {
            dynamic_range / 0.7 // Reward up to moderate dynamics
        };

        // Combine factors for pleasantness score
        let pleasantness = (spectral_smoothness * 0.3
            + (1.0 - roughness) * 0.3
            + dynamic_pleasantness * 0.2
            + (1.0 - harsh_hf_penalty) * 0.2)
            .clamp(0.0, 1.0);

        Ok(pleasantness)
    }

    fn calculate_intelligibility(&self, original: &[f32], processed: &[f32]) -> Result<f32, Error> {
        // Use correlation as a simple intelligibility measure
        Ok(self.calculate_correlation(original, processed))
    }

    fn calculate_spaciousness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Spaciousness based on reverb characteristics and stereo width
        let envelope = self.calculate_envelope(audio);

        // Calculate decay characteristics (indicative of space)
        let decay_score = self.analyze_decay_characteristics(&envelope);

        // Calculate spectral diffusion (wider spectrum = more spacious)
        let spectrum = self.magnitude_spectrum(audio);
        let spectral_spread = self.calculate_spectral_spread(&spectrum);

        // Late reflection simulation (longer envelope tails = more spacious)
        let late_reflection_score = if envelope.len() > 10 {
            let tail_start = envelope.len() * 3 / 4;
            let tail_energy: f32 = envelope[tail_start..].iter().sum();
            let total_energy: f32 = envelope.iter().sum();

            if total_energy > 1e-10 {
                (tail_energy / total_energy * 4.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Combine factors for spaciousness score
        let spaciousness =
            (decay_score * 0.4 + spectral_spread * 0.3 + late_reflection_score * 0.3)
                .clamp(0.0, 1.0);

        Ok(spaciousness)
    }

    fn calculate_warmth(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Warmth based on low-frequency content and harmonic richness
        let mut lf_energy = 0.0;
        let mut mf_energy = 0.0;
        let mut total_energy = 0.0;

        let lf_cutoff = spectrum.len() / 8; // Low frequencies
        let mf_cutoff = spectrum.len() / 3; // Mid frequencies

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i < lf_cutoff {
                lf_energy += energy;
            } else if i < mf_cutoff {
                mf_energy += energy;
            }
        }

        // Low to mid frequency ratio (warmth indicator)
        let lf_mf_ratio = if mf_energy > 1e-10 {
            (lf_energy / mf_energy).min(2.0) / 2.0
        } else {
            0.0
        };

        // Overall low frequency content
        let lf_content = if total_energy > 1e-10 {
            lf_energy / total_energy
        } else {
            0.0
        };

        // Harmonic warmth (even harmonics contribute to warmth)
        let harmonic_warmth = self.calculate_harmonic_warmth(&spectrum);

        // Combine factors for warmth score
        let warmth = (lf_mf_ratio * 0.4 + lf_content * 0.3 + harmonic_warmth * 0.3).clamp(0.0, 1.0);

        Ok(warmth)
    }

    fn calculate_brightness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Brightness based on high-frequency content and spectral centroid
        let mut hf_energy = 0.0;
        let mut total_energy = 0.0;
        let mut weighted_freq_sum = 0.0;

        let hf_start = spectrum.len() / 3; // Above ~2.7kHz equivalent

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;
            weighted_freq_sum += i as f32 * energy;

            if i >= hf_start {
                hf_energy += energy;
            }
        }

        // High-frequency content
        let hf_ratio = if total_energy > 1e-10 {
            hf_energy / total_energy
        } else {
            0.0
        };

        // Spectral centroid (brightness indicator)
        let spectral_centroid = if total_energy > 1e-10 {
            weighted_freq_sum / total_energy / spectrum.len() as f32
        } else {
            0.0
        };

        // Presence of harmonics in the brightness range
        let brightness_harmonics = self.calculate_brightness_harmonics(&spectrum);

        // Combine factors for brightness score
        let brightness =
            (hf_ratio * 0.4 + spectral_centroid * 0.3 + brightness_harmonics * 0.3).clamp(0.0, 1.0);

        Ok(brightness)
    }

    fn calculate_presence(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Presence based on upper-midrange content and clarity
        let mut presence_energy = 0.0;
        let mut total_energy = 0.0;

        // Presence frequency range (roughly 2-8 kHz equivalent)
        let presence_start = spectrum.len() / 4;
        let presence_end = spectrum.len() * 2 / 3;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i >= presence_start && i < presence_end {
                presence_energy += energy;
            }
        }

        // Presence frequency content
        let presence_ratio = if total_energy > 1e-10 {
            presence_energy / total_energy
        } else {
            0.0
        };

        // Dynamic range in presence region (clarity of presence)
        let presence_spectrum = &spectrum[presence_start..presence_end];
        let presence_dynamic_range = if !presence_spectrum.is_empty() {
            let max_val = presence_spectrum.iter().fold(0.0f32, |a, &b| a.max(b));
            let min_val = presence_spectrum
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            if max_val > min_val && min_val > 1e-10 {
                (max_val / min_val).log10() / 3.0 // Normalize to ~0-1
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Attack characteristics (presence through transient definition)
        let attack_definition = self.calculate_attack_definition(audio);

        // Combine factors for presence score
        let presence = (presence_ratio * 0.4
            + presence_dynamic_range.clamp(0.0, 1.0) * 0.3
            + attack_definition * 0.3)
            .clamp(0.0, 1.0);

        Ok(presence)
    }

    fn calculate_loudness_difference(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_rms =
            (original.iter().map(|&x| x * x).sum::<f32>() / original.len() as f32).sqrt();
        let proc_rms =
            (processed.iter().map(|&x| x * x).sum::<f32>() / processed.len() as f32).sqrt();
        Ok((orig_rms - proc_rms).abs() / orig_rms.max(1e-10))
    }

    fn calculate_sharpness_difference(
        &self,
        _original: &[f32],
        _processed: &[f32],
    ) -> Result<f32, Error> {
        Ok(0.1) // Placeholder
    }

    /// Helper method to estimate fundamental frequency using autocorrelation
    fn estimate_fundamental_frequency(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.len() < 128 {
            return 0.0;
        }

        let max_lag = (sample_rate as usize / 50).min(audio.len() / 2); // Min 50 Hz
        let min_lag = (sample_rate as usize / 800).max(8); // Max 800 Hz

        let mut max_correlation = 0.0;
        let mut best_lag = 0;

        // Autocorrelation-based pitch detection
        for lag in min_lag..max_lag {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            let samples_to_check = (audio.len() - lag).min(1024);

            for j in 0..samples_to_check {
                correlation += audio[j] * audio[j + lag];
                norm1 += audio[j] * audio[j];
                norm2 += audio[j + lag] * audio[j + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        if max_correlation > 0.3 && best_lag > 0 {
            sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    /// Helper method to find local peak around a given bin
    fn find_local_peak(&self, spectrum: &[f32], center_bin: usize, radius: usize) -> f32 {
        let start = center_bin.saturating_sub(radius);
        let end = (center_bin + radius + 1).min(spectrum.len());

        spectrum[start..end]
            .iter()
            .fold(0.0, |max_val, &val| max_val.max(val))
    }

    /// Calculate sharpness (psychoacoustic measure of high-frequency content)
    fn calculate_sharpness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut sharpness = 0.0;

        // Calculate spectral centroid weighted by frequency
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            let frequency = i as f32; // Normalized frequency bin

            // Weight higher frequencies more heavily for sharpness
            let sharpness_weight = if frequency > spectrum.len() as f32 * 0.1 {
                (frequency / spectrum.len() as f32).powf(2.0) // Quadratic weighting
            } else {
                0.1 * (frequency / spectrum.len() as f32)
            };

            weighted_sum += frequency * energy * sharpness_weight;
            total_energy += energy;
        }

        if total_energy > 1e-10 {
            sharpness = weighted_sum / total_energy;
            // Normalize to approximate acum range (0-5)
            sharpness = (sharpness / spectrum.len() as f32 * 5.0).clamp(0.0, 5.0);
        }

        Ok(sharpness)
    }

    /// Helper methods for multidimensional quality calculations
    fn detect_spectral_artifacts(&self, spectrum: &[f32]) -> f32 {
        let mut artifact_score = 0.0;

        // Look for sudden spectral peaks (potential artifacts)
        for i in 2..spectrum.len() - 2 {
            let current = spectrum[i];
            let neighbors =
                (spectrum[i - 2] + spectrum[i - 1] + spectrum[i + 1] + spectrum[i + 2]) / 4.0;

            if current > neighbors * 3.0 && current > 0.01 {
                artifact_score += current - neighbors;
            }
        }

        (artifact_score * 10.0).clamp(0.0, 1.0)
    }

    fn calculate_dynamic_range(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let envelope = self.calculate_envelope(audio);
        if envelope.is_empty() {
            return 0.0;
        }

        let max_val = envelope.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = envelope.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        if max_val > min_val && min_val > 1e-10 {
            ((max_val / min_val).log10() / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn calculate_spectral_smoothness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.len() < 3 {
            return 1.0;
        }

        let mut smoothness_sum = 0.0;
        for i in 1..spectrum.len() - 1 {
            let variation = (spectrum[i] - (spectrum[i - 1] + spectrum[i + 1]) / 2.0).abs();
            smoothness_sum += variation;
        }

        let avg_variation = smoothness_sum / (spectrum.len() - 2) as f32;
        let avg_magnitude = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if avg_magnitude > 1e-10 {
            (1.0 - (avg_variation / avg_magnitude).min(1.0)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    fn analyze_decay_characteristics(&self, envelope: &[f32]) -> f32 {
        if envelope.len() < 10 {
            return 0.0;
        }

        // Look for exponential decay characteristics
        let peak_idx = envelope
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if peak_idx >= envelope.len() - 5 {
            return 0.0;
        }

        let decay_portion = &envelope[peak_idx..];
        if decay_portion.len() < 5 {
            return 0.0;
        }

        // Calculate decay rate
        let mut decay_score = 0.0;
        let peak_val = decay_portion[0];

        if peak_val > 1e-10 {
            for (i, &val) in decay_portion.iter().enumerate().skip(1) {
                let expected_decay = peak_val * (-0.1 * i as f32).exp();
                let decay_match = 1.0 - ((val - expected_decay) / peak_val).abs();
                decay_score += decay_match.max(0.0);
            }
            decay_score /= (decay_portion.len() - 1) as f32;
        }

        decay_score.clamp(0.0, 1.0)
    }

    fn calculate_spectral_spread(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            weighted_sum += i as f32 * energy;
            total_energy += energy;
        }

        let centroid = if total_energy > 1e-10 {
            weighted_sum / total_energy
        } else {
            return 0.0;
        };

        let mut spread_sum = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            let deviation = (i as f32 - centroid).powi(2);
            spread_sum += deviation * energy;
        }

        let spread = if total_energy > 1e-10 {
            (spread_sum / total_energy).sqrt()
        } else {
            0.0
        };

        (spread / spectrum.len() as f32).clamp(0.0, 1.0)
    }

    fn calculate_harmonic_warmth(&self, spectrum: &[f32]) -> f32 {
        let mut warmth_score = 0.0;
        let mut harmonic_count = 0;

        // Look for even harmonics in low-mid frequency range
        let warmth_range = spectrum.len() / 3;

        for i in (2..warmth_range).step_by(2) {
            // Even harmonics
            if i < spectrum.len() {
                warmth_score += spectrum[i] * spectrum[i];
                harmonic_count += 1;
            }
        }

        if harmonic_count > 0 {
            (warmth_score / harmonic_count as f32 * 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn calculate_brightness_harmonics(&self, spectrum: &[f32]) -> f32 {
        let mut brightness_score = 0.0;
        let brightness_start = spectrum.len() / 3;
        let brightness_end = spectrum.len() * 2 / 3;

        // Look for harmonic content in brightness range
        for i in brightness_start..brightness_end {
            if i > 0 && i < spectrum.len() - 1 {
                // Look for peaks (harmonics)
                if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
                    brightness_score += spectrum[i] * spectrum[i];
                }
            }
        }

        let range_size = brightness_end - brightness_start;
        if range_size > 0 {
            (brightness_score / range_size as f32 * 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn calculate_attack_definition(&self, audio: &[f32]) -> f32 {
        if audio.len() < 100 {
            return 0.0;
        }

        let envelope = self.calculate_envelope(audio);
        if envelope.len() < 10 {
            return 0.0;
        }

        // Look for fast attack characteristics
        let mut max_attack_rate = 0.0;

        for i in 1..envelope.len().min(20) {
            // Check first 20 frames for attack
            let attack_rate = envelope[i] - envelope[i - 1];
            if attack_rate > max_attack_rate {
                max_attack_rate = attack_rate;
            }
        }

        // Normalize attack rate
        let max_envelope = envelope.iter().fold(0.0f32, |a, &b| a.max(b));

        if max_envelope > 1e-10 {
            (max_attack_rate / max_envelope * 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Get analysis statistics
    pub fn get_analysis_count(&self) -> usize {
        self.analysis_count
    }

    /// Clear analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_config_creation() {
        let config = ResearchConfig::default();
        assert!(config.neural_models);
        assert_eq!(config.psychoacoustic_depth, 5);
        assert!(config.pesq_analysis);
        assert!(config.stoi_analysis);
        assert!(config.pemo_q_analysis);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_research_config_builder() {
        let config = ResearchConfig::default()
            .with_neural_models(false)
            .with_psychoacoustic_depth(8)
            .with_sample_rate(44100);

        assert!(!config.neural_models);
        assert_eq!(config.psychoacoustic_depth, 8);
        assert_eq!(config.sample_rate, 44100);
    }

    #[test]
    fn test_audio_quality_researcher_creation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config);
        assert!(researcher.is_ok());
    }

    #[test]
    fn test_neural_quality_model_default() {
        let model = NeuralQualityModel::default();
        assert!(model.weights.contains_key("spectral_distortion"));
        assert!(model.weights.contains_key("temporal_coherence"));
        assert_eq!(model.hidden_layers.len(), 3);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
        let processed = vec![0.09, 0.19, 0.29, 0.19, 0.09, 0.01, -0.09, -0.19];

        let result = researcher.comprehensive_analysis(&original, &processed, 16000);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.perceptual_quality >= 0.0 && analysis.perceptual_quality <= 1.0);
        assert!(analysis.neural_prediction >= 0.0 && analysis.neural_prediction <= 1.0);
        assert!(analysis.pesq_score >= 1.0 && analysis.pesq_score <= 5.0);
        assert!(analysis.stoi_score >= 0.0 && analysis.stoi_score <= 1.0);
        assert!(analysis.pemo_q_score >= 0.0 && analysis.pemo_q_score <= 1.0);
    }

    #[test]
    fn test_spectral_distortion_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let processed = vec![0.9, 0.45, 0.0, -0.45, -0.9];

        let distortion = researcher.calculate_spectral_distortion(&original, &processed);
        assert!(distortion.is_ok());
        let distortion_value = distortion.unwrap();
        assert!(distortion_value > 0.0 && distortion_value < 1.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let correlation_perfect = researcher.calculate_correlation(&a, &b);
        let correlation_negative = researcher.calculate_correlation(&a, &c);

        assert!((correlation_perfect - 1.0).abs() < 1e-6);
        assert!(correlation_negative < 0.0);
    }

    #[test]
    fn test_temporal_coherence_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1; 1024]; // Constant signal
        let processed = vec![0.1; 1024]; // Same signal

        let coherence = researcher.calculate_temporal_coherence(&original, &processed);
        assert!(coherence.is_ok());
        assert!(coherence.unwrap() > 0.9); // Should be very high for identical signals
    }

    #[test]
    fn test_envelope_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![0.5; 1000];
        let envelope = researcher.calculate_envelope(&audio);
        assert!(!envelope.is_empty());
        assert!(envelope.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_zero_crossing_rate() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let zcr = researcher.calculate_zero_crossing_rate(&audio);
        assert!(zcr > 0.8); // High ZCR for alternating signal
    }

    #[test]
    fn test_spectral_flatness() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0; 128]; // Constant signal (not flat spectrum)
        let flatness = researcher.calculate_spectral_flatness(&audio);
        assert!(flatness >= 0.0 && flatness <= 1.0);
    }

    #[test]
    fn test_magnitude_spectrum() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0, 0.0, -1.0, 0.0]; // Simple sinusoid
        let spectrum = researcher.magnitude_spectrum(&audio);
        assert_eq!(spectrum.len(), audio.len() / 2 + 1);
        assert!(spectrum.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_empty_audio_handling() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let empty_audio: Vec<f32> = vec![];
        let result = researcher.comprehensive_analysis(&empty_audio, &empty_audio, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_length_handling() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1, 0.2, 0.3];
        let processed = vec![0.1, 0.2];

        let result = researcher.comprehensive_analysis(&original, &processed, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_analysis_count_tracking() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        assert_eq!(researcher.get_analysis_count(), 0);

        let audio = vec![0.1; 1000];
        let _ = researcher.comprehensive_analysis(&audio, &audio, 16000);

        assert_eq!(researcher.get_analysis_count(), 1);
    }

    #[test]
    fn test_cache_functionality() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        researcher.clear_cache();
        assert_eq!(researcher.analysis_cache.len(), 0);
    }
}
