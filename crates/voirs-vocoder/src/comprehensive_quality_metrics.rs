//! Comprehensive quality metrics for all vocoder features
//!
//! Provides feature-specific quality assessment including:
//! - Emotion expression quality metrics
//! - Voice conversion fidelity metrics
//! - Spatial audio positioning accuracy
//! - Singing voice naturalness metrics
//! - Overall perceptual quality assessment

use crate::{AudioBuffer, Result, VocoderFeature};
use scirs2_core::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Comprehensive quality assessment for all features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveQualityMetrics {
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Feature-specific quality scores
    pub feature_qualities: HashMap<VocoderFeature, FeatureQualityMetrics>,
    /// Perceptual quality metrics
    pub perceptual: PerceptualQualityMetrics,
    /// Technical quality metrics
    pub technical: TechnicalQualityMetrics,
    /// Quality consistency metrics
    pub consistency: ConsistencyMetrics,
    /// Quality timestamp
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Quality metrics specific to each vocoder feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureQualityMetrics {
    /// Feature type
    pub feature_type: VocoderFeature,
    /// Quality score for this feature (0.0-1.0)
    pub quality_score: f32,
    /// Feature-specific sub-metrics
    pub sub_metrics: FeatureSubMetrics,
    /// Confidence in quality assessment
    pub confidence: f32,
    /// Quality degradation factors
    pub degradation_factors: Vec<QualityDegradationFactor>,
}

/// Feature-specific sub-metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSubMetrics {
    /// Emotion-specific quality metrics
    Emotion(EmotionQualityMetrics),
    /// Voice conversion quality metrics
    VoiceConversion(VoiceConversionQualityMetrics),
    /// Spatial audio quality metrics
    Spatial(SpatialQualityMetrics),
    /// Singing voice quality metrics
    Singing(SingingQualityMetrics),
    /// Base vocoding quality metrics
    Base(BaseVocodingQualityMetrics),
}

/// Quality metrics for emotion expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionQualityMetrics {
    /// Emotion expression clarity (0.0-1.0)
    pub expression_clarity: f32,
    /// Emotion intensity accuracy (0.0-1.0)
    pub intensity_accuracy: f32,
    /// Emotional naturalness (0.0-1.0)
    pub naturalness: f32,
    /// Emotion consistency across time (0.0-1.0)
    pub temporal_consistency: f32,
    /// Spectral emotion characteristics (0.0-1.0)
    pub spectral_characteristics: f32,
    /// Prosodic emotion features (0.0-1.0)
    pub prosodic_features: f32,
}

impl Default for EmotionQualityMetrics {
    fn default() -> Self {
        Self {
            expression_clarity: 0.8,
            intensity_accuracy: 0.8,
            naturalness: 0.8,
            temporal_consistency: 0.8,
            spectral_characteristics: 0.8,
            prosodic_features: 0.8,
        }
    }
}

/// Quality metrics for voice conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConversionQualityMetrics {
    /// Speaker similarity to target (0.0-1.0)
    pub speaker_similarity: f32,
    /// Content preservation (0.0-1.0)
    pub content_preservation: f32,
    /// Conversion naturalness (0.0-1.0)
    pub conversion_naturalness: f32,
    /// Artifact level (0.0=no artifacts, 1.0=severe artifacts)
    pub artifact_level: f32,
    /// Spectral consistency (0.0-1.0)
    pub spectral_consistency: f32,
    /// Prosody preservation (0.0-1.0)
    pub prosody_preservation: f32,
}

impl Default for VoiceConversionQualityMetrics {
    fn default() -> Self {
        Self {
            speaker_similarity: 0.8,
            content_preservation: 0.9,
            conversion_naturalness: 0.8,
            artifact_level: 0.1,
            spectral_consistency: 0.8,
            prosody_preservation: 0.8,
        }
    }
}

/// Quality metrics for spatial audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQualityMetrics {
    /// Spatial positioning accuracy (0.0-1.0)
    pub positioning_accuracy: f32,
    /// Binaural rendering quality (0.0-1.0)
    pub binaural_quality: f32,
    /// Distance perception accuracy (0.0-1.0)
    pub distance_accuracy: f32,
    /// Room acoustics realism (0.0-1.0)
    pub room_acoustics: f32,
    /// Head tracking responsiveness (0.0-1.0)
    pub head_tracking_quality: f32,
    /// Spatial consistency (0.0-1.0)
    pub spatial_consistency: f32,
}

impl Default for SpatialQualityMetrics {
    fn default() -> Self {
        Self {
            positioning_accuracy: 0.8,
            binaural_quality: 0.8,
            distance_accuracy: 0.7,
            room_acoustics: 0.8,
            head_tracking_quality: 0.9,
            spatial_consistency: 0.8,
        }
    }
}

/// Quality metrics for singing voice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingQualityMetrics {
    /// Pitch accuracy (0.0-1.0)
    pub pitch_accuracy: f32,
    /// Vocal technique quality (0.0-1.0)
    pub vocal_technique: f32,
    /// Breath control naturalness (0.0-1.0)
    pub breath_control: f32,
    /// Vibrato quality (0.0-1.0)
    pub vibrato_quality: f32,
    /// Musical phrasing (0.0-1.0)
    pub musical_phrasing: f32,
    /// Harmonic richness (0.0-1.0)
    pub harmonic_richness: f32,
}

impl Default for SingingQualityMetrics {
    fn default() -> Self {
        Self {
            pitch_accuracy: 0.9,
            vocal_technique: 0.8,
            breath_control: 0.8,
            vibrato_quality: 0.8,
            musical_phrasing: 0.8,
            harmonic_richness: 0.8,
        }
    }
}

/// Quality metrics for base vocoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseVocodingQualityMetrics {
    /// Audio fidelity (0.0-1.0)
    pub audio_fidelity: f32,
    /// Spectral accuracy (0.0-1.0)
    pub spectral_accuracy: f32,
    /// Temporal accuracy (0.0-1.0)
    pub temporal_accuracy: f32,
    /// Noise level (0.0=no noise, 1.0=very noisy)
    pub noise_level: f32,
    /// Dynamic range preservation (0.0-1.0)
    pub dynamic_range: f32,
    /// Frequency response accuracy (0.0-1.0)
    pub frequency_response: f32,
}

impl Default for BaseVocodingQualityMetrics {
    fn default() -> Self {
        Self {
            audio_fidelity: 0.9,
            spectral_accuracy: 0.9,
            temporal_accuracy: 0.9,
            noise_level: 0.05,
            dynamic_range: 0.9,
            frequency_response: 0.9,
        }
    }
}

/// Perceptual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQualityMetrics {
    /// Mean Opinion Score (MOS) estimate (1.0-5.0)
    pub mos_estimate: f32,
    /// Perceived naturalness (0.0-1.0)
    pub naturalness: f32,
    /// Listening effort required (0.0=effortless, 1.0=high effort)
    pub listening_effort: f32,
    /// Overall pleasantness (0.0-1.0)
    pub pleasantness: f32,
    /// Perceived quality stability (0.0-1.0)
    pub quality_stability: f32,
}

impl Default for PerceptualQualityMetrics {
    fn default() -> Self {
        Self {
            mos_estimate: 4.0,
            naturalness: 0.8,
            listening_effort: 0.2,
            pleasantness: 0.8,
            quality_stability: 0.8,
        }
    }
}

/// Technical quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalQualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Total harmonic distortion (0.0-1.0)
    pub thd: f32,
    /// Frequency response flatness (0.0-1.0)
    pub frequency_flatness: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Spectral centroid stability (0.0-1.0)
    pub spectral_stability: f32,
    /// Phase coherence (0.0-1.0)
    pub phase_coherence: f32,
}

impl Default for TechnicalQualityMetrics {
    fn default() -> Self {
        Self {
            snr_db: 40.0,
            thd: 0.02,
            frequency_flatness: 0.9,
            dynamic_range_db: 60.0,
            spectral_stability: 0.9,
            phase_coherence: 0.9,
        }
    }
}

/// Quality consistency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    /// Temporal consistency (0.0-1.0)
    pub temporal_consistency: f32,
    /// Cross-feature consistency (0.0-1.0)
    pub cross_feature_consistency: f32,
    /// Quality variance (lower is better)
    pub quality_variance: f32,
    /// Stability over time (0.0-1.0)
    pub stability: f32,
}

impl Default for ConsistencyMetrics {
    fn default() -> Self {
        Self {
            temporal_consistency: 0.9,
            cross_feature_consistency: 0.8,
            quality_variance: 0.05,
            stability: 0.9,
        }
    }
}

/// Quality degradation factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDegradationFactor {
    /// Factor type
    pub factor_type: DegradationFactorType,
    /// Severity (0.0-1.0)
    pub severity: f32,
    /// Description
    pub description: String,
    /// Potential impact on quality (0.0-1.0)
    pub impact: f32,
}

/// Types of quality degradation factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationFactorType {
    /// Processing artifacts
    Artifacts,
    /// Resource constraints
    ResourceLimitation,
    /// Model limitations
    ModelLimitation,
    /// Input quality issues
    InputQuality,
    /// Configuration issues
    Configuration,
    /// Hardware limitations
    Hardware,
}

/// Comprehensive quality assessor for all features
pub struct ComprehensiveQualityAssessor {
    /// Feature-specific quality calculators
    feature_calculators: HashMap<VocoderFeature, Box<dyn FeatureQualityCalculator>>,
    /// Assessment configuration
    config: QualityAssessmentConfig,
}

/// Configuration for quality assessment
#[derive(Debug, Clone)]
pub struct QualityAssessmentConfig {
    /// Enable detailed analysis (slower but more accurate)
    pub enable_detailed_analysis: bool,
    /// Quality assessment frequency (every N samples)
    pub assessment_frequency: usize,
    /// Minimum confidence threshold for reliable assessment
    pub min_confidence_threshold: f32,
    /// Enable real-time assessment
    pub enable_realtime_assessment: bool,
}

impl Default for QualityAssessmentConfig {
    fn default() -> Self {
        Self {
            enable_detailed_analysis: true,
            assessment_frequency: 1000,
            min_confidence_threshold: 0.7,
            enable_realtime_assessment: true,
        }
    }
}

/// Trait for feature-specific quality calculators
pub trait FeatureQualityCalculator: Send + Sync {
    /// Calculate quality metrics for a feature
    fn calculate_quality(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics>;

    /// Get feature type this calculator handles
    fn feature_type(&self) -> VocoderFeature;

    /// Update calculator configuration
    fn update_config(&mut self, config: &QualityAssessmentConfig);
}

impl ComprehensiveQualityAssessor {
    /// Create new comprehensive quality assessor
    pub fn new(config: QualityAssessmentConfig) -> Self {
        let mut assessor = Self {
            feature_calculators: HashMap::new(),
            config,
        };

        // Register feature-specific calculators
        assessor.register_default_calculators();
        assessor
    }

    /// Register default quality calculators for all features
    fn register_default_calculators(&mut self) {
        self.feature_calculators.insert(
            VocoderFeature::Emotion,
            Box::new(EmotionQualityCalculator::new()),
        );
        self.feature_calculators.insert(
            VocoderFeature::VoiceConversion,
            Box::new(VoiceConversionQualityCalculator::new()),
        );
        self.feature_calculators.insert(
            VocoderFeature::Spatial,
            Box::new(SpatialQualityCalculator::new()),
        );
        self.feature_calculators.insert(
            VocoderFeature::Singing,
            Box::new(SingingQualityCalculator::new()),
        );
        self.feature_calculators.insert(
            VocoderFeature::Base,
            Box::new(BaseVocodingQualityCalculator::new()),
        );
    }

    /// Assess comprehensive quality for all active features
    pub fn assess_comprehensive_quality(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        active_features: &[VocoderFeature],
        feature_params: &HashMap<VocoderFeature, HashMap<String, f32>>,
    ) -> Result<ComprehensiveQualityMetrics> {
        let mut feature_qualities = HashMap::new();
        let mut overall_quality = 0.0;
        let mut total_weight = 0.0;

        // Calculate quality for each active feature
        for feature in active_features {
            if let Some(calculator) = self.feature_calculators.get(feature) {
                let params = feature_params.get(feature).cloned().unwrap_or_default();
                let quality = calculator.calculate_quality(audio, reference, &params)?;

                let weight = self.get_feature_weight(feature);
                overall_quality += quality.quality_score * weight;
                total_weight += weight;

                feature_qualities.insert(*feature, quality);
            }
        }

        // Normalize overall quality
        if total_weight > 0.0 {
            overall_quality /= total_weight;
        }

        // Calculate perceptual and technical metrics
        let perceptual = self.calculate_perceptual_metrics(audio, reference)?;
        let technical = self.calculate_technical_metrics(audio, reference)?;
        let consistency = self.calculate_consistency_metrics(&feature_qualities);

        Ok(ComprehensiveQualityMetrics {
            overall_quality,
            feature_qualities,
            perceptual,
            technical,
            consistency,
            timestamp: Instant::now(),
        })
    }

    /// Get weight for a feature in overall quality calculation
    fn get_feature_weight(&self, feature: &VocoderFeature) -> f32 {
        match feature {
            VocoderFeature::Base => 0.4,            // Base quality is most important
            VocoderFeature::Emotion => 0.25,        // Emotion expression is important
            VocoderFeature::VoiceConversion => 0.2, // Voice conversion quality
            VocoderFeature::Spatial => 0.1,         // Spatial accuracy
            VocoderFeature::Singing => 0.15,        // Singing naturalness
            VocoderFeature::StreamingInference => 0.1,
            VocoderFeature::BatchProcessing => 0.1,
            VocoderFeature::GpuAcceleration => 0.05,
            VocoderFeature::HighQuality => 0.3,
            VocoderFeature::RealtimeProcessing => 0.2,
            VocoderFeature::FastInference => 0.15,
            VocoderFeature::EmotionConditioning => 0.25,
            VocoderFeature::AgeTransformation => 0.15,
            VocoderFeature::GenderTransformation => 0.15,
            VocoderFeature::VoiceMorphing => 0.15,
            VocoderFeature::SingingVoice => 0.15,
            VocoderFeature::SpatialAudio => 0.1,
        }
    }

    /// Calculate perceptual quality metrics
    fn calculate_perceptual_metrics(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<PerceptualQualityMetrics> {
        // Simplified perceptual analysis
        let snr = self.calculate_snr(audio, reference);
        let thd = self.calculate_thd(audio);

        // Convert technical metrics to perceptual scores
        let mos_estimate = self.technical_to_mos(snr, thd);
        let naturalness = (snr / 50.0).clamp(0.0, 1.0);
        let listening_effort = (1.0 - naturalness).clamp(0.0, 1.0);
        let pleasantness = naturalness * 0.9;
        let quality_stability = (1.0 - thd * 10.0).clamp(0.0, 1.0);

        Ok(PerceptualQualityMetrics {
            mos_estimate,
            naturalness,
            listening_effort,
            pleasantness,
            quality_stability,
        })
    }

    /// Calculate technical quality metrics
    fn calculate_technical_metrics(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<TechnicalQualityMetrics> {
        let snr_db = self.calculate_snr(audio, reference);
        let thd = self.calculate_thd(audio);
        let frequency_flatness = self.calculate_frequency_flatness(audio);
        let dynamic_range_db = self.calculate_dynamic_range(audio);
        let spectral_stability = self.calculate_spectral_stability(audio);
        let phase_coherence = self.calculate_phase_coherence(audio);

        Ok(TechnicalQualityMetrics {
            snr_db,
            thd,
            frequency_flatness,
            dynamic_range_db,
            spectral_stability,
            phase_coherence,
        })
    }

    /// Calculate consistency metrics across features
    fn calculate_consistency_metrics(
        &self,
        feature_qualities: &HashMap<VocoderFeature, FeatureQualityMetrics>,
    ) -> ConsistencyMetrics {
        if feature_qualities.is_empty() {
            return ConsistencyMetrics::default();
        }

        // Calculate quality variance across features
        let qualities: Vec<f32> = feature_qualities
            .values()
            .map(|q| q.quality_score)
            .collect();

        let mean_quality = qualities.iter().sum::<f32>() / qualities.len() as f32;
        let variance = qualities
            .iter()
            .map(|q| (q - mean_quality).powi(2))
            .sum::<f32>()
            / qualities.len() as f32;

        let quality_variance = variance.sqrt();
        let cross_feature_consistency = (1.0 - quality_variance).clamp(0.0, 1.0);

        ConsistencyMetrics {
            temporal_consistency: 0.9, // Would be calculated from time-series data
            cross_feature_consistency,
            quality_variance,
            stability: cross_feature_consistency * 0.9,
        }
    }

    // Technical metric calculations.

    /// Signal-to-noise ratio in dB.
    ///
    /// With a reference signal this is the standard error-to-signal SNR
    /// (`10·log10(Σref² / Σ(audio-ref)²)`). Without a reference it is a blind
    /// estimate: the mean short-frame power compared against the noise-floor
    /// power, taken as a low percentile of the per-frame powers.
    fn calculate_snr(&self, audio: &AudioBuffer, reference: Option<&AudioBuffer>) -> f32 {
        if audio.samples.is_empty() {
            return 0.0;
        }
        match reference {
            Some(ref_audio) if !ref_audio.samples.is_empty() => {
                let signal_power = ref_audio.samples.iter().map(|x| x * x).sum::<f32>();
                let noise_power = audio
                    .samples
                    .iter()
                    .zip(ref_audio.samples.iter())
                    .map(|(a, r)| (a - r).powi(2))
                    .sum::<f32>();

                if noise_power > 0.0 {
                    10.0 * (signal_power / noise_power).log10()
                } else {
                    60.0 // Very high SNR
                }
            }
            _ => {
                // Blind SNR: mean frame power vs. noise-floor (10th percentile).
                let frame = (audio.sample_rate() as usize / 50).max(64); // ~20 ms
                let mut powers: Vec<f32> = audio
                    .samples
                    .chunks(frame)
                    .filter(|c| c.len() == frame)
                    .map(|c| c.iter().map(|&x| x * x).sum::<f32>() / frame as f32)
                    .collect();

                if powers.len() < 2 {
                    let p = audio.samples.iter().map(|&x| x * x).sum::<f32>()
                        / audio.samples.len() as f32;
                    return if p > 0.0 { 40.0 } else { 0.0 };
                }

                powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let noise_floor = powers[powers.len() / 10];
                let signal = powers.iter().sum::<f32>() / powers.len() as f32;
                let floor = noise_floor.max(signal * 1e-9).max(f32::MIN_POSITIVE);
                (10.0 * (signal / floor).log10()).clamp(0.0, 90.0)
            }
        }
    }

    /// Total harmonic distortion (0.0-1.0).
    ///
    /// Computes the FFT of a Hann-windowed segment, locates the fundamental as
    /// the strongest non-DC bin, then forms `THD = sqrt(Σ harmonic energy) /
    /// sqrt(fundamental energy)` from the energy at integer multiples of the
    /// fundamental bin (each measured over a ±2-bin window to absorb leakage).
    fn calculate_thd(&self, audio: &AudioBuffer) -> f32 {
        if audio.samples.len() < 64 {
            return 0.0;
        }
        let signal = to_mono_f64(audio);

        // FFT size: largest power of two ≤ signal length, capped to [1024, 8192].
        let mut fft_size = 1024;
        while fft_size * 2 <= signal.len() && fft_size < 8192 {
            fft_size *= 2;
        }

        let windowed: Vec<f64> = (0..fft_size)
            .map(|i| signal[i] * hann(i, fft_size))
            .collect();
        let spec = match scirs2_fft::rfft(&windowed, None) {
            Ok(s) => s,
            Err(_) => return 0.0,
        };
        let mag: Vec<f64> = spec.iter().map(|c| c.norm()).collect();
        let n_bins = mag.len();
        if n_bins < 4 {
            return 0.0;
        }

        // Fundamental = strongest bin above the DC / very-low-frequency region.
        let search_start = 2usize;
        let mut fund_bin = search_start;
        let mut fund_mag = 0.0_f64;
        for (i, &m) in mag.iter().enumerate().skip(search_start) {
            if m > fund_mag {
                fund_mag = m;
                fund_bin = i;
            }
        }
        if fund_mag <= 0.0 {
            return 0.0;
        }

        // Energy within a ±2-bin window around a partial (captures leakage).
        let band_energy = |center: usize| -> f64 {
            let lo = center.saturating_sub(2);
            let hi = (center + 2).min(n_bins - 1);
            (lo..=hi).map(|b| mag[b] * mag[b]).sum::<f64>()
        };

        let fundamental_energy = band_energy(fund_bin);
        if fundamental_energy <= 0.0 {
            return 0.0;
        }

        let mut harmonic_energy = 0.0_f64;
        let mut h = 2usize;
        while fund_bin * h < n_bins - 1 {
            harmonic_energy += band_energy(fund_bin * h);
            h += 1;
        }

        ((harmonic_energy / fundamental_energy).sqrt()).clamp(0.0, 1.0) as f32
    }

    /// Spectral flatness (Wiener entropy) of the power spectrum, in `[0, 1]`.
    ///
    /// Defined as `geometric_mean(power) / arithmetic_mean(power)` over the
    /// Welch-averaged power spectrum (DC bin excluded). Broadband / noise-like
    /// signals approach 1.0; tonal signals approach 0.0.
    fn calculate_frequency_flatness(&self, audio: &AudioBuffer) -> f32 {
        if audio.samples.is_empty() {
            return 0.0;
        }
        let signal = to_mono_f64(audio);
        let fft_size = 1024;
        let power = averaged_power_spectrum(&signal, fft_size, fft_size / 2);

        let bins = &power[1..]; // skip DC
        if bins.is_empty() {
            return 0.0;
        }
        let arith = bins.iter().sum::<f64>() / bins.len() as f64;
        if arith <= 0.0 {
            return 0.0;
        }
        // Floor each bin relative to the mean to avoid log(0); a tone's empty
        // bins then drive the geometric mean toward zero (low flatness).
        let floor = arith * 1e-10;
        let log_sum: f64 = bins.iter().map(|&p| p.max(floor).ln()).sum::<f64>();
        let geo = (log_sum / bins.len() as f64).exp();
        (geo / arith).clamp(0.0, 1.0) as f32
    }

    fn calculate_dynamic_range(&self, audio: &AudioBuffer) -> f32 {
        let max_val = audio.samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let min_val = audio.samples.iter().fold(f32::INFINITY, |a, &b| {
            if b.abs() > 0.001 {
                a.min(b.abs())
            } else {
                a
            }
        });

        if min_val > 0.0 && min_val != f32::INFINITY {
            20.0 * (max_val / min_val).log10()
        } else {
            60.0
        }
    }

    /// Frame-to-frame spectral stability in `[0, 1]`.
    ///
    /// Computes the normalised spectral flux between consecutive STFT magnitude
    /// spectra, `flux_t = Σ|m_t − m_{t-1}| / Σ(m_t + m_{t-1}) ∈ [0, 1]`, and
    /// reports `1 − mean(flux_t)`. A stationary signal (steady tone) yields a
    /// stability near 1.0; a rapidly changing / noisy signal yields a lower value.
    fn calculate_spectral_stability(&self, audio: &AudioBuffer) -> f32 {
        if audio.samples.is_empty() {
            return 1.0;
        }
        let signal = to_mono_f64(audio);
        let frames = stft_frames(&signal, 1024, 512);
        if frames.len() < 2 {
            return 1.0; // not enough frames to observe any variation
        }

        let mut flux_sum = 0.0_f64;
        let mut count = 0_usize;
        for w in frames.windows(2) {
            let (prev, cur) = (&w[0], &w[1]);
            let mut diff = 0.0_f64;
            let mut denom = 0.0_f64;
            for (a, b) in cur.iter().zip(prev.iter()) {
                let (ma, mb) = (a.norm(), b.norm());
                diff += (ma - mb).abs();
                denom += ma + mb;
            }
            if denom > 0.0 {
                flux_sum += diff / denom;
                count += 1;
            }
        }
        if count == 0 {
            return 1.0;
        }
        (1.0 - flux_sum / count as f64).clamp(0.0, 1.0) as f32
    }

    /// Phase coherence in `[0, 1]`.
    ///
    /// Defined as the magnitude-weighted mean resultant length of the per-bin
    /// inter-frame phase increments. For every bin we accumulate the unit
    /// phasor `exp(i·Δφ)` of the phase advance between consecutive STFT frames,
    /// weighted by the bin magnitude. A bin whose phase advances by a consistent
    /// amount each hop (a stable sinusoid) gives a resultant length `R_k → 1`;
    /// random phases give `R_k → 0`. The reported value is `Σ(w_k·R_k) / Σ w_k`.
    fn calculate_phase_coherence(&self, audio: &AudioBuffer) -> f32 {
        if audio.samples.is_empty() {
            return 1.0;
        }
        let signal = to_mono_f64(audio);
        let frames = stft_frames(&signal, 1024, 512);
        if frames.len() < 2 {
            return 1.0;
        }

        let n_bins = frames[0].len();
        let mut sum_re = vec![0.0_f64; n_bins];
        let mut sum_im = vec![0.0_f64; n_bins];
        let mut sum_w = vec![0.0_f64; n_bins];

        for w in frames.windows(2) {
            let (prev, cur) = (&w[0], &w[1]);
            for (k, (a, b)) in cur.iter().zip(prev.iter()).enumerate() {
                let weight = (a.norm() * b.norm()).sqrt();
                if weight <= 0.0 {
                    continue;
                }
                let dphi = a.arg() - b.arg();
                sum_re[k] += weight * dphi.cos();
                sum_im[k] += weight * dphi.sin();
                sum_w[k] += weight;
            }
        }

        let mut weighted_r = 0.0_f64;
        let mut total_w = 0.0_f64;
        for ((&re, &im), &w) in sum_re.iter().zip(sum_im.iter()).zip(sum_w.iter()) {
            if w > 0.0 {
                let r = (re * re + im * im).sqrt() / w;
                weighted_r += w * r;
                total_w += w;
            }
        }
        if total_w <= 0.0 {
            return 1.0;
        }
        (weighted_r / total_w).clamp(0.0, 1.0) as f32
    }

    fn technical_to_mos(&self, snr: f32, thd: f32) -> f32 {
        let snr_contribution = (snr / 60.0 * 3.0 + 1.0).clamp(1.0, 5.0);
        let thd_penalty = thd * 20.0; // THD in percent
        (snr_contribution - thd_penalty).clamp(1.0, 5.0)
    }

    /// Check if real-time assessment is enabled
    pub fn is_realtime_assessment_enabled(&self) -> bool {
        self.config.enable_realtime_assessment
    }

    /// Get assessment frequency
    pub fn get_assessment_frequency(&self) -> usize {
        self.config.assessment_frequency
    }

    /// Check if detailed analysis is enabled  
    pub fn is_detailed_analysis_enabled(&self) -> bool {
        self.config.enable_detailed_analysis
    }

    /// Get minimum confidence threshold
    pub fn get_min_confidence_threshold(&self) -> f32 {
        self.config.min_confidence_threshold
    }

    /// Update assessment configuration
    pub fn update_config(&mut self, config: QualityAssessmentConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &QualityAssessmentConfig {
        &self.config
    }

    /// Check if assessment should be performed based on configuration
    pub fn should_assess(&self, sample_count: usize) -> bool {
        sample_count.is_multiple_of(self.config.assessment_frequency)
    }
}

// Feature-specific quality calculator implementations

/// Quality calculator for emotion features
pub struct EmotionQualityCalculator {
    config: QualityAssessmentConfig,
}

impl Default for EmotionQualityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionQualityCalculator {
    pub fn new() -> Self {
        Self {
            config: QualityAssessmentConfig::default(),
        }
    }
}

impl FeatureQualityCalculator for EmotionQualityCalculator {
    fn calculate_quality(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics> {
        let emotion_intensity = feature_params.get("emotion_intensity").unwrap_or(&0.5);
        let _emotion_type_id = feature_params.get("emotion_type").unwrap_or(&0.0);

        // Simplified emotion quality assessment
        let expression_clarity = 0.8 + emotion_intensity * 0.15;
        let intensity_accuracy = (1.0 - (emotion_intensity - 0.5).abs()).clamp(0.6, 1.0);
        let naturalness = 0.85;
        let temporal_consistency = 0.9;
        let spectral_characteristics = 0.8;
        let prosodic_features = 0.8;

        let quality_score = (expression_clarity
            + intensity_accuracy
            + naturalness
            + temporal_consistency
            + spectral_characteristics
            + prosodic_features)
            / 6.0;

        Ok(FeatureQualityMetrics {
            feature_type: VocoderFeature::Emotion,
            quality_score,
            sub_metrics: FeatureSubMetrics::Emotion(EmotionQualityMetrics {
                expression_clarity,
                intensity_accuracy,
                naturalness,
                temporal_consistency,
                spectral_characteristics,
                prosodic_features,
            }),
            confidence: 0.8,
            degradation_factors: vec![],
        })
    }

    fn feature_type(&self) -> VocoderFeature {
        VocoderFeature::Emotion
    }

    fn update_config(&mut self, config: &QualityAssessmentConfig) {
        self.config = config.clone();
    }
}

/// Quality calculator for voice conversion features
pub struct VoiceConversionQualityCalculator {
    config: QualityAssessmentConfig,
}

impl Default for VoiceConversionQualityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceConversionQualityCalculator {
    pub fn new() -> Self {
        Self {
            config: QualityAssessmentConfig::default(),
        }
    }
}

impl FeatureQualityCalculator for VoiceConversionQualityCalculator {
    fn calculate_quality(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics> {
        let conversion_strength = feature_params.get("conversion_strength").unwrap_or(&0.7);

        let speaker_similarity = 0.8 * conversion_strength;
        let content_preservation = 1.0 - conversion_strength * 0.1;
        let conversion_naturalness = 0.85;
        let artifact_level = conversion_strength * 0.15;
        let spectral_consistency = 0.8;
        let prosody_preservation = 0.9;

        let quality_score = (speaker_similarity
            + content_preservation
            + conversion_naturalness
            + (1.0 - artifact_level)
            + spectral_consistency
            + prosody_preservation)
            / 6.0;

        Ok(FeatureQualityMetrics {
            feature_type: VocoderFeature::VoiceConversion,
            quality_score,
            sub_metrics: FeatureSubMetrics::VoiceConversion(VoiceConversionQualityMetrics {
                speaker_similarity,
                content_preservation,
                conversion_naturalness,
                artifact_level,
                spectral_consistency,
                prosody_preservation,
            }),
            confidence: 0.85,
            degradation_factors: vec![],
        })
    }

    fn feature_type(&self) -> VocoderFeature {
        VocoderFeature::VoiceConversion
    }

    fn update_config(&mut self, config: &QualityAssessmentConfig) {
        self.config = config.clone();
    }
}

/// Quality calculator for spatial audio features
pub struct SpatialQualityCalculator {
    config: QualityAssessmentConfig,
}

impl Default for SpatialQualityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialQualityCalculator {
    pub fn new() -> Self {
        Self {
            config: QualityAssessmentConfig::default(),
        }
    }
}

impl FeatureQualityCalculator for SpatialQualityCalculator {
    fn calculate_quality(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics> {
        let spatial_precision = *feature_params.get("spatial_precision").unwrap_or(&0.8);

        let positioning_accuracy = spatial_precision;
        let binaural_quality = 0.85;
        let distance_accuracy = spatial_precision * 0.9;
        let room_acoustics = 0.8;
        let head_tracking_quality = 0.9;
        let spatial_consistency = spatial_precision;

        let quality_score = (positioning_accuracy
            + binaural_quality
            + distance_accuracy
            + room_acoustics
            + head_tracking_quality
            + spatial_consistency)
            / 6.0;

        Ok(FeatureQualityMetrics {
            feature_type: VocoderFeature::Spatial,
            quality_score,
            sub_metrics: FeatureSubMetrics::Spatial(SpatialQualityMetrics {
                positioning_accuracy,
                binaural_quality,
                distance_accuracy,
                room_acoustics,
                head_tracking_quality,
                spatial_consistency,
            }),
            confidence: 0.75,
            degradation_factors: vec![],
        })
    }

    fn feature_type(&self) -> VocoderFeature {
        VocoderFeature::Spatial
    }

    fn update_config(&mut self, config: &QualityAssessmentConfig) {
        self.config = config.clone();
    }
}

/// Quality calculator for singing voice features
pub struct SingingQualityCalculator {
    config: QualityAssessmentConfig,
}

impl Default for SingingQualityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl SingingQualityCalculator {
    pub fn new() -> Self {
        Self {
            config: QualityAssessmentConfig::default(),
        }
    }
}

impl FeatureQualityCalculator for SingingQualityCalculator {
    fn calculate_quality(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics> {
        let pitch_stability = *feature_params.get("pitch_stability").unwrap_or(&0.9);
        let vibrato_strength = *feature_params.get("vibrato_strength").unwrap_or(&0.3);

        let pitch_accuracy = pitch_stability;
        let vocal_technique = 0.8;
        let breath_control = 0.85;
        let vibrato_quality = if vibrato_strength > 0.1 { 0.8 } else { 0.9 };
        let musical_phrasing = 0.8;
        let harmonic_richness = 0.85;

        let quality_score = (pitch_accuracy
            + vocal_technique
            + breath_control
            + vibrato_quality
            + musical_phrasing
            + harmonic_richness)
            / 6.0;

        Ok(FeatureQualityMetrics {
            feature_type: VocoderFeature::Singing,
            quality_score,
            sub_metrics: FeatureSubMetrics::Singing(SingingQualityMetrics {
                pitch_accuracy,
                vocal_technique,
                breath_control,
                vibrato_quality,
                musical_phrasing,
                harmonic_richness,
            }),
            confidence: 0.8,
            degradation_factors: vec![],
        })
    }

    fn feature_type(&self) -> VocoderFeature {
        VocoderFeature::Singing
    }

    fn update_config(&mut self, config: &QualityAssessmentConfig) {
        self.config = config.clone();
    }
}

/// Quality calculator for base vocoding features
pub struct BaseVocodingQualityCalculator {
    config: QualityAssessmentConfig,
}

impl Default for BaseVocodingQualityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseVocodingQualityCalculator {
    pub fn new() -> Self {
        Self {
            config: QualityAssessmentConfig::default(),
        }
    }
}

impl FeatureQualityCalculator for BaseVocodingQualityCalculator {
    fn calculate_quality(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        _feature_params: &HashMap<String, f32>,
    ) -> Result<FeatureQualityMetrics> {
        // Use simplified technical analysis for base quality
        let audio_fidelity = 0.9;
        let spectral_accuracy = 0.85;
        let temporal_accuracy = 0.9;
        let noise_level = 0.05;
        let dynamic_range = 0.9;
        let frequency_response = 0.85;

        let quality_score = (audio_fidelity
            + spectral_accuracy
            + temporal_accuracy
            + (1.0 - noise_level)
            + dynamic_range
            + frequency_response)
            / 6.0;

        Ok(FeatureQualityMetrics {
            feature_type: VocoderFeature::Base,
            quality_score,
            sub_metrics: FeatureSubMetrics::Base(BaseVocodingQualityMetrics {
                audio_fidelity,
                spectral_accuracy,
                temporal_accuracy,
                noise_level,
                dynamic_range,
                frequency_response,
            }),
            confidence: 0.9,
            degradation_factors: vec![],
        })
    }

    fn feature_type(&self) -> VocoderFeature {
        VocoderFeature::Base
    }

    fn update_config(&mut self, config: &QualityAssessmentConfig) {
        self.config = config.clone();
    }
}

// ---------------------------------------------------------------------------
// DSP helpers shared by the FFT-based technical metrics.
// ---------------------------------------------------------------------------

/// Downmix an (interleaved) audio buffer to a mono `f64` signal.
fn to_mono_f64(audio: &AudioBuffer) -> Vec<f64> {
    let ch = audio.channels().max(1) as usize;
    if ch <= 1 {
        return audio.samples.iter().map(|&s| s as f64).collect();
    }
    audio
        .samples
        .chunks(ch)
        .map(|frame| frame.iter().map(|&s| s as f64).sum::<f64>() / ch as f64)
        .collect()
}

/// Hann window weight at position `i` of a window of length `len`.
fn hann(i: usize, len: usize) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (len - 1) as f64).cos())
}

/// Welch-averaged power spectrum: the mean periodogram over Hann-windowed,
/// `hop`-spaced frames. Returns one power value per real-FFT bin.
fn averaged_power_spectrum(signal: &[f64], fft_size: usize, hop: usize) -> Vec<f64> {
    let mut acc = vec![0.0_f64; fft_size / 2 + 1];
    let mut frames = 0_usize;

    if signal.len() >= fft_size {
        let mut start = 0;
        while start + fft_size <= signal.len() {
            let windowed: Vec<f64> = (0..fft_size)
                .map(|i| signal[start + i] * hann(i, fft_size))
                .collect();
            if let Ok(spec) = scirs2_fft::rfft(&windowed, None) {
                for (a, c) in acc.iter_mut().zip(spec.iter()) {
                    *a += c.norm_sqr();
                }
                frames += 1;
            }
            start += hop.max(1);
        }
    }

    if frames == 0 {
        // Signal shorter than one frame: zero-pad a single frame.
        let mut buf = vec![0.0_f64; fft_size];
        let copy = fft_size.min(signal.len());
        for (i, b) in buf.iter_mut().enumerate().take(copy) {
            *b = signal[i] * hann(i, fft_size);
        }
        if let Ok(spec) = scirs2_fft::rfft(&buf, None) {
            for (a, c) in acc.iter_mut().zip(spec.iter()) {
                *a = c.norm_sqr();
            }
            frames = 1;
        }
    }

    if frames > 1 {
        let inv = 1.0 / frames as f64;
        for a in acc.iter_mut() {
            *a *= inv;
        }
    }
    acc
}

/// Per-frame complex STFT (Hann window, `hop` spacing). Each inner vector holds
/// the `fft_size/2 + 1` complex bins of one frame.
fn stft_frames(signal: &[f64], fft_size: usize, hop: usize) -> Vec<Vec<Complex<f64>>> {
    let mut frames: Vec<Vec<Complex<f64>>> = Vec::new();
    if signal.len() < fft_size {
        return frames;
    }
    let mut start = 0;
    while start + fft_size <= signal.len() {
        let windowed: Vec<f64> = (0..fft_size)
            .map(|i| signal[start + i] * hann(i, fft_size))
            .collect();
        if let Ok(spec) = scirs2_fft::rfft(&windowed, None) {
            frames.push(spec);
        }
        start += hop.max(1);
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assessor() -> ComprehensiveQualityAssessor {
        ComprehensiveQualityAssessor::new(QualityAssessmentConfig::default())
    }

    fn sine(freq: f32, secs: f32, sr: u32, amp: f32) -> AudioBuffer {
        let n = (secs * sr as f32) as usize;
        let s: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer::new(s, sr, 1)
    }

    /// Deterministic white-ish noise via a linear congruential generator. Using
    /// a self-contained LCG avoids any external RNG dependency (SciRS2 policy).
    fn white_noise(n: usize, sr: u32, amp: f32) -> AudioBuffer {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let s: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
                (u * 2.0 - 1.0) * amp
            })
            .collect();
        AudioBuffer::new(s, sr, 1)
    }

    #[test]
    fn test_frequency_flatness_noise_high_tone_low() {
        let a = assessor();
        let noise = white_noise(48_000, 48_000, 0.5);
        let tone = sine(1000.0, 1.0, 48_000, 0.5);

        let noise_flat = a.calculate_frequency_flatness(&noise);
        let tone_flat = a.calculate_frequency_flatness(&tone);

        assert!(
            noise_flat > 0.4,
            "white-noise flatness should be high, got {noise_flat:.4}"
        );
        assert!(
            tone_flat < 0.1,
            "pure-tone flatness should be low, got {tone_flat:.4}"
        );
        assert!(noise_flat > tone_flat);
    }

    #[test]
    fn test_thd_pure_sine_low_clipped_high() {
        let a = assessor();
        let pure = sine(1000.0, 1.0, 48_000, 0.9);

        // Hard-clip the same sine to inject harmonics.
        let mut clipped_samples = pure.samples().to_vec();
        for s in clipped_samples.iter_mut() {
            *s = s.clamp(-0.4, 0.4);
        }
        let clipped = AudioBuffer::new(clipped_samples, 48_000, 1);

        let thd_pure = a.calculate_thd(&pure);
        let thd_clipped = a.calculate_thd(&clipped);

        assert!(
            thd_pure < 0.05,
            "pure-sine THD should be ~0, got {thd_pure:.4}"
        );
        assert!(
            thd_clipped > 0.1,
            "clipped-sine THD should be high, got {thd_clipped:.4}"
        );
        assert!(thd_clipped > thd_pure);
    }

    #[test]
    fn test_spectral_stability_steady_vs_noise() {
        let a = assessor();
        let steady = sine(440.0, 1.0, 48_000, 0.5);
        let noise = white_noise(48_000, 48_000, 0.5);

        let steady_stab = a.calculate_spectral_stability(&steady);
        let noise_stab = a.calculate_spectral_stability(&noise);

        assert!(
            steady_stab > 0.7,
            "steady tone should be stable, got {steady_stab:.4}"
        );
        assert!(
            steady_stab > noise_stab,
            "steady ({steady_stab:.4}) should exceed noise ({noise_stab:.4})"
        );
    }

    #[test]
    fn test_phase_coherence_tone_vs_noise() {
        let a = assessor();
        let tone = sine(440.0, 1.0, 48_000, 0.5);
        let noise = white_noise(48_000, 48_000, 0.5);

        let tone_coh = a.calculate_phase_coherence(&tone);
        let noise_coh = a.calculate_phase_coherence(&noise);

        assert!(
            tone_coh > 0.7,
            "steady tone should be phase-coherent, got {tone_coh:.4}"
        );
        assert!(
            tone_coh > noise_coh,
            "tone ({tone_coh:.4}) should exceed noise ({noise_coh:.4})"
        );
    }

    #[test]
    fn test_blind_snr_clean_vs_noisy() {
        let a = assessor();
        let sr = 48_000_u32;

        // Loud tone for the first half, silence for the second half.
        let half = sr as usize / 2;
        let mut clean: Vec<f32> = (0..half)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        clean.resize(sr as usize, 0.0);

        // Same signal plus a small noise floor everywhere.
        let noise = white_noise(sr as usize, sr, 0.05);
        let noisy: Vec<f32> = clean
            .iter()
            .zip(noise.samples().iter())
            .map(|(c, n)| c + n)
            .collect();

        let clean_snr = a.calculate_snr(&AudioBuffer::new(clean, sr, 1), None);
        let noisy_snr = a.calculate_snr(&AudioBuffer::new(noisy, sr, 1), None);

        assert!(clean_snr.is_finite() && noisy_snr.is_finite());
        assert!(
            noisy_snr > 0.0,
            "noisy SNR should be positive, got {noisy_snr:.2}"
        );
        assert!(
            clean_snr > noisy_snr,
            "clean SNR ({clean_snr:.2}) should exceed noisy SNR ({noisy_snr:.2})"
        );
    }
}
