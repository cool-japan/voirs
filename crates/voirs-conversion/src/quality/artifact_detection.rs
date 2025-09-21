//! Artifact detection system for voice conversion quality monitoring

use crate::{Error, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Artifact detection system for conversion quality monitoring
#[derive(Debug, Clone)]
pub struct ArtifactDetector {
    /// Detection threshold for various artifact types
    thresholds: ArtifactThresholds,
    /// History buffer for temporal analysis
    history_buffer: Vec<f32>,
    /// Maximum history length
    max_history: usize,
    /// Performance metrics for production monitoring
    performance_metrics: ProductionMetrics,
    /// Adaptive threshold state for learning
    adaptive_state: AdaptiveState,
    /// Memory pool for efficient processing
    memory_pool: MemoryPool,
}

/// Thresholds for different types of artifacts
#[derive(Debug, Clone)]
pub struct ArtifactThresholds {
    /// Clicking and popping artifacts threshold
    pub click_threshold: f32,
    /// Metallic/robotic sound threshold  
    pub metallic_threshold: f32,
    /// Buzzing/distortion threshold
    pub buzzing_threshold: f32,
    /// Unnatural pitch variations threshold
    pub pitch_variation_threshold: f32,
    /// Spectral discontinuities threshold
    pub spectral_discontinuity_threshold: f32,
    /// Energy spikes threshold
    pub energy_spike_threshold: f32,
    /// High frequency noise threshold
    pub hf_noise_threshold: f32,
    /// Phase artifacts threshold
    pub phase_artifact_threshold: f32,
    /// New production-ready artifact thresholds
    /// Temporal jitter threshold
    pub temporal_jitter_threshold: f32,
    /// Spectral tilt threshold
    pub spectral_tilt_threshold: f32,
    /// Formant tracking error threshold
    pub formant_tracking_threshold: f32,
    /// Loudness inconsistency threshold
    pub loudness_inconsistency_threshold: f32,
    /// Channel crosstalk threshold
    pub channel_crosstalk_threshold: f32,
    /// Inter-harmonic distortion threshold
    pub interharmonic_distortion_threshold: f32,
    /// Consonant degradation threshold
    pub consonant_degradation_threshold: f32,
    /// Vowel coloration threshold
    pub vowel_coloration_threshold: f32,
    /// Adaptive threshold learning rate
    pub adaptation_rate: f32,
    /// Minimum confidence threshold for detection
    pub min_confidence: f32,
}

impl Default for ArtifactThresholds {
    fn default() -> Self {
        Self {
            click_threshold: 0.1,
            metallic_threshold: 0.15,
            buzzing_threshold: 0.12,
            pitch_variation_threshold: 0.2,
            spectral_discontinuity_threshold: 0.08,
            energy_spike_threshold: 0.25,
            hf_noise_threshold: 0.18,
            phase_artifact_threshold: 0.14,
            // New production-ready artifact thresholds with conservative defaults
            temporal_jitter_threshold: 0.05, // Very sensitive to timing issues
            spectral_tilt_threshold: 0.12,   // Moderate sensitivity to frequency balance
            formant_tracking_threshold: 0.08, // Sensitive to formant consistency  
            loudness_inconsistency_threshold: 0.15, // Moderate sensitivity to volume jumps
            channel_crosstalk_threshold: 0.1, // Sensitive to channel bleed
            interharmonic_distortion_threshold: 0.13, // Moderate sensitivity to harmonic issues
            consonant_degradation_threshold: 0.11, // Sensitive to speech clarity
            vowel_coloration_threshold: 0.09, // Very sensitive to timbre changes
            adaptation_rate: 0.01,
            min_confidence: 0.6,
        }
    }
}

/// Detected artifacts in audio
#[derive(Debug, Clone)]
pub struct DetectedArtifacts {
    /// Overall artifact score (0.0 = clean, 1.0 = heavily artifacted)
    pub overall_score: f32,
    /// Individual artifact types and their scores
    pub artifact_types: HashMap<ArtifactType, f32>,
    /// Temporal locations of artifacts (in samples)
    pub artifact_locations: Vec<ArtifactLocation>,
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
}

/// Types of artifacts that can be detected
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ArtifactType {
    /// Clicks and pops
    Click,
    /// Metallic/robotic sound
    Metallic,
    /// Buzzing/distortion
    Buzzing,
    /// Unnatural pitch variations
    PitchVariation,
    /// Spectral discontinuities
    SpectralDiscontinuity,
    /// Energy spikes
    EnergySpike,
    /// High frequency noise
    HighFrequencyNoise,
    /// Phase artifacts
    PhaseArtifact,
    /// New production-ready artifact types
    /// Temporal jitter and timing inconsistencies
    TemporalJitter,
    /// Spectral tilt and frequency balance issues
    SpectralTilt,
    /// Formant tracking errors and resonance issues
    FormantTracking,
    /// Loudness inconsistencies and jumps
    LoudnessInconsistency,
    /// Channel crosstalk in multi-channel audio
    ChannelCrosstalk,
    /// Inter-harmonic distortion
    InterharmonicDistortion,
    /// Consonant degradation in speech conversion
    ConsonantDegradation,
    /// Vowel coloration and timbre issues
    VowelColoration,
}

/// Location of detected artifact
#[derive(Debug, Clone)]
pub struct ArtifactLocation {
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Start sample
    pub start_sample: usize,
    /// End sample
    pub end_sample: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Severity (0.0 to 1.0)
    pub severity: f32,
}

/// Comprehensive quality assessment
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Clarity score (0.0 to 1.0)
    pub clarity: f32,
    /// Consistency score (0.0 to 1.0)
    pub consistency: f32,
    /// Recommended quality adjustments
    pub recommended_adjustments: Vec<QualityAdjustment>,
}

/// Recommended quality adjustment
#[derive(Debug, Clone)]
pub struct QualityAdjustment {
    /// Type of adjustment
    pub adjustment_type: AdjustmentType,
    /// Recommended strength (0.0 to 1.0)
    pub strength: f32,
    /// Expected improvement
    pub expected_improvement: f32,
}

/// Types of quality adjustments
#[derive(Debug, Clone)]
pub enum AdjustmentType {
    /// Reduce conversion strength
    ReduceConversion,
    /// Apply noise reduction
    NoiseReduction,
    /// Smooth spectral transitions
    SpectralSmoothing,
    /// Adjust pitch stability
    PitchStabilization,
    /// Apply temporal smoothing
    TemporalSmoothing,
    /// Enhance formant preservation
    FormantPreservation,
}

/// Production metrics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct ProductionMetrics {
    /// Total number of artifacts detected
    pub total_detections: usize,
    /// Average processing time per sample in microseconds
    pub avg_processing_time_us: f64,
    /// Peak processing time in microseconds
    pub peak_processing_time_us: u64,
    /// Memory usage statistics
    pub memory_usage_mb: f64,
    /// False positive rate estimate
    pub false_positive_rate: f32,
    /// Confidence calibration statistics
    pub confidence_stats: ConfidenceStats,
    /// Throughput metrics
    pub samples_per_second: f64,
}

/// Confidence calibration statistics
#[derive(Debug, Clone, Default)]
pub struct ConfidenceStats {
    /// Mean confidence score
    pub mean_confidence: f32,
    /// Standard deviation of confidence scores
    pub std_confidence: f32,
    /// Calibration error (difference between confidence and accuracy)
    pub calibration_error: f32,
    /// Number of samples used for statistics
    pub sample_count: usize,
}

/// Adaptive threshold state for machine learning
#[derive(Debug, Clone, Default)]
pub struct AdaptiveState {
    /// Running averages of artifact scores for each type
    pub running_averages: HashMap<ArtifactType, f32>,
    /// Variance estimates for each artifact type
    pub variance_estimates: HashMap<ArtifactType, f32>,
    /// Learning iteration count
    pub iteration_count: usize,
    /// Adaptation enabled flag
    pub adaptation_enabled: bool,
    /// Context-specific thresholds (e.g., for different audio types)
    pub context_thresholds: HashMap<String, ArtifactThresholds>,
}

/// Memory pool for efficient processing
#[derive(Debug, Clone, Default)]
pub struct MemoryPool {
    /// Reusable FFT buffers
    pub fft_buffers: Vec<Vec<f32>>,
    /// Reusable complex buffers
    pub complex_buffers: Vec<Vec<num_complex::Complex<f32>>>,
    /// Reusable analysis windows
    pub analysis_windows: Vec<Vec<f32>>,
    /// Buffer pool size
    pub pool_size: usize,
    /// Current buffer index
    pub current_index: usize,
}

impl ArtifactDetector {
    /// Create new artifact detector
    pub fn new() -> Self {
        Self::with_thresholds(ArtifactThresholds::default())
    }

    /// Create artifact detector with custom thresholds
    pub fn with_thresholds(thresholds: ArtifactThresholds) -> Self {
        Self {
            thresholds,
            history_buffer: Vec::new(),
            max_history: 44100, // 1 second at 44.1kHz
            performance_metrics: ProductionMetrics::default(),
            adaptive_state: AdaptiveState::default(),
            memory_pool: MemoryPool::default(),
        }
    }

    /// Create production-ready detector with optimizations
    pub fn new_production() -> Self {
        let mut detector = Self::new();
        detector.adaptive_state.adaptation_enabled = true;
        detector.memory_pool.pool_size = 16; // Pre-allocate buffers
        detector.initialize_memory_pool();
        detector
    }

    /// Initialize memory pool for efficient processing
    fn initialize_memory_pool(&mut self) {
        let pool_size = self.memory_pool.pool_size.max(1);

        // Pre-allocate FFT buffers
        self.memory_pool.fft_buffers = (0..pool_size).map(|_| Vec::with_capacity(8192)).collect();

        // Pre-allocate complex buffers
        self.memory_pool.complex_buffers =
            (0..pool_size).map(|_| Vec::with_capacity(4096)).collect();

        // Pre-allocate analysis windows
        self.memory_pool.analysis_windows =
            (0..pool_size).map(|_| Vec::with_capacity(2048)).collect();
    }

    /// Detect artifacts in audio with production monitoring
    pub fn detect_artifacts(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<DetectedArtifacts> {
        let start_time = std::time::Instant::now();
        debug!(
            "Detecting artifacts in {} samples at {} Hz",
            audio.len(),
            sample_rate
        );

        // Update history buffer
        self.update_history(audio);

        let mut artifact_types = HashMap::new();
        let mut artifact_locations = Vec::new();

        // Detect different types of artifacts with adaptive thresholds
        let click_score = self.detect_clicks_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Click, click_score);

        let metallic_score =
            self.detect_metallic_artifacts_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Metallic, metallic_score);

        let buzzing_score = self.detect_buzzing_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Buzzing, buzzing_score);

        let pitch_variation_score =
            self.detect_pitch_variations_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::PitchVariation, pitch_variation_score);

        let spectral_discontinuity_score =
            self.detect_spectral_discontinuities_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(
            ArtifactType::SpectralDiscontinuity,
            spectral_discontinuity_score,
        );

        let energy_spike_score =
            self.detect_energy_spikes_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::EnergySpike, energy_spike_score);

        let hf_noise_score =
            self.detect_high_frequency_noise_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::HighFrequencyNoise, hf_noise_score);

        let phase_artifact_score =
            self.detect_phase_artifacts_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::PhaseArtifact, phase_artifact_score);

        // New production-ready artifact detection
        let temporal_jitter_score =
            self.detect_temporal_jitter_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::TemporalJitter, temporal_jitter_score);

        let spectral_tilt_score =
            self.detect_spectral_tilt_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::SpectralTilt, spectral_tilt_score);

        let formant_tracking_score =
            self.detect_formant_tracking_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::FormantTracking, formant_tracking_score);

        let loudness_inconsistency_score =
            self.detect_loudness_inconsistency_adaptive(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::LoudnessInconsistency, loudness_inconsistency_score);

        let interharmonic_distortion_score =
            self.detect_interharmonic_distortion_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::InterharmonicDistortion, interharmonic_distortion_score);

        let consonant_degradation_score =
            self.detect_consonant_degradation_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::ConsonantDegradation, consonant_degradation_score);

        let vowel_coloration_score =
            self.detect_vowel_coloration_adaptive(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::VowelColoration, vowel_coloration_score);

        // Update adaptive thresholds if enabled
        if self.adaptive_state.adaptation_enabled {
            self.update_adaptive_thresholds(&artifact_types);
        }

        // Calculate overall artifact score with confidence calibration
        let overall_score = self.calculate_calibrated_score(&artifact_types, &artifact_locations);

        // Perform comprehensive quality assessment
        let quality_assessment = self.assess_quality(audio, &artifact_types, sample_rate)?;

        let result = DetectedArtifacts {
            overall_score,
            artifact_types,
            artifact_locations,
            quality_assessment,
        };

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.update_performance_metrics(&result, processing_time, audio.len());

        info!(
            "Artifact detection complete: overall_score={:.3}, detected {} locations, time={:?}",
            result.overall_score,
            result.artifact_locations.len(),
            processing_time
        );

        Ok(result)
    }

    /// Legacy method for backward compatibility
    pub fn detect_artifacts_legacy(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<DetectedArtifacts> {
        debug!(
            "Using legacy artifact detection for {} samples",
            audio.len()
        );

        self.update_history(audio);
        let mut artifact_types = HashMap::new();
        let mut artifact_locations = Vec::new();

        // Use original detection methods for compatibility
        let click_score = self.detect_clicks(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Click, click_score);

        let metallic_score = self.detect_metallic_artifacts(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Metallic, metallic_score);

        let buzzing_score = self.detect_buzzing(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::Buzzing, buzzing_score);

        let pitch_variation_score =
            self.detect_pitch_variations(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::PitchVariation, pitch_variation_score);

        let spectral_discontinuity_score =
            self.detect_spectral_discontinuities(audio, &mut artifact_locations)?;
        artifact_types.insert(
            ArtifactType::SpectralDiscontinuity,
            spectral_discontinuity_score,
        );

        let energy_spike_score = self.detect_energy_spikes(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::EnergySpike, energy_spike_score);

        let hf_noise_score =
            self.detect_high_frequency_noise(audio, sample_rate, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::HighFrequencyNoise, hf_noise_score);

        let phase_artifact_score = self.detect_phase_artifacts(audio, &mut artifact_locations)?;
        artifact_types.insert(ArtifactType::PhaseArtifact, phase_artifact_score);

        let overall_score = artifact_types
            .values()
            .map(|&score| score)
            .fold(0.0f32, f32::max)
            .min(1.0);
        let quality_assessment = self.assess_quality(audio, &artifact_types, sample_rate)?;

        Ok(DetectedArtifacts {
            overall_score,
            artifact_types,
            artifact_locations,
            quality_assessment,
        })
    }

    /// Update history buffer for temporal analysis
    fn update_history(&mut self, audio: &[f32]) {
        self.history_buffer.extend_from_slice(audio);

        // Keep only recent history
        if self.history_buffer.len() > self.max_history {
            let excess = self.history_buffer.len() - self.max_history;
            self.history_buffer.drain(0..excess);
        }
    }

    /// Detect clicking and popping artifacts
    fn detect_clicks(&self, audio: &[f32], locations: &mut Vec<ArtifactLocation>) -> Result<f32> {
        let mut click_score: f32 = 0.0;
        let window_size = 32; // Small window for click detection

        for i in window_size..audio.len() {
            let current_energy = audio[i] * audio[i];
            let prev_energy: f32 =
                audio[i - window_size..i].iter().map(|x| x * x).sum::<f32>() / window_size as f32;

            // Detect sudden energy changes
            if prev_energy > 0.0 {
                let energy_ratio = current_energy / prev_energy;
                if energy_ratio > 10.0 && current_energy > self.thresholds.click_threshold {
                    let confidence = (energy_ratio / 20.0).min(1.0);
                    let severity = (current_energy / 0.5).min(1.0);

                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::Click,
                        start_sample: i.saturating_sub(window_size / 2),
                        end_sample: i + window_size / 2,
                        confidence,
                        severity,
                    });

                    click_score = click_score.max(confidence * severity);
                }
            }
        }

        Ok(click_score)
    }

    /// Detect metallic/robotic sound artifacts
    fn detect_metallic_artifacts(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut metallic_score: f32 = 0.0;
        let window_size = 512; // Larger window for spectral analysis

        // Analyze spectral regularity patterns that indicate metallic sound
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];

            // Calculate zero crossing rate
            let zcr = self.calculate_zero_crossing_rate(window);

            // Calculate spectral regularity (simplified)
            let regularity = self.calculate_spectral_regularity(window);

            // Metallic sounds often have very regular spectral patterns
            if regularity > 0.8 && zcr > 0.05 {
                let confidence = regularity;
                let severity = ((zcr - 0.05) / 0.1).min(1.0);

                if confidence * severity > self.thresholds.metallic_threshold {
                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::Metallic,
                        start_sample: i,
                        end_sample: i + window_size,
                        confidence,
                        severity,
                    });

                    metallic_score = metallic_score.max(confidence * severity);
                }
            }
        }

        Ok(metallic_score)
    }

    /// Detect buzzing/distortion artifacts
    fn detect_buzzing(&self, audio: &[f32], locations: &mut Vec<ArtifactLocation>) -> Result<f32> {
        let mut buzzing_score: f32 = 0.0;
        let window_size = 256;

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];

            // Calculate total harmonic distortion (simplified)
            let thd = self.calculate_thd(window);

            // High THD indicates buzzing/distortion
            if thd > self.thresholds.buzzing_threshold {
                let confidence = (thd / 0.3).min(1.0);
                let severity = thd.min(1.0);

                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::Buzzing,
                    start_sample: i,
                    end_sample: i + window_size,
                    confidence,
                    severity,
                });

                buzzing_score = buzzing_score.max(confidence * severity);
            }
        }

        Ok(buzzing_score)
    }

    /// Detect unnatural pitch variations
    fn detect_pitch_variations(
        &self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut pitch_score: f32 = 0.0;
        let window_size = (sample_rate as usize / 50).max(256); // ~20ms windows
        let overlap = window_size / 2;

        let mut prev_f0 = 0.0;

        for i in (0..audio.len()).step_by(overlap) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0(window, sample_rate);

            if prev_f0 > 0.0 && f0 > 0.0 {
                let pitch_change = (f0 - prev_f0).abs() / prev_f0;

                // Detect unnaturally large pitch jumps
                if pitch_change > self.thresholds.pitch_variation_threshold {
                    let confidence = (pitch_change / 0.5).min(1.0);
                    let severity = pitch_change;

                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::PitchVariation,
                        start_sample: i,
                        end_sample: i + window_size,
                        confidence,
                        severity,
                    });

                    pitch_score = pitch_score.max(confidence * severity);
                }
            }

            prev_f0 = f0;
        }

        Ok(pitch_score)
    }

    /// Detect spectral discontinuities
    fn detect_spectral_discontinuities(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut discontinuity_score: f32 = 0.0;
        let window_size = 512;
        let overlap = window_size / 4;

        let mut prev_spectrum: Vec<f32> = Vec::new();

        for i in (0..audio.len()).step_by(overlap) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let spectrum = self.calculate_spectrum_simplified(window);

            if !prev_spectrum.is_empty() {
                let spectral_distance = self.calculate_spectral_distance(&prev_spectrum, &spectrum);

                if spectral_distance > self.thresholds.spectral_discontinuity_threshold {
                    let confidence = (spectral_distance / 0.2).min(1.0);
                    let severity = spectral_distance;

                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::SpectralDiscontinuity,
                        start_sample: i,
                        end_sample: i + window_size,
                        confidence,
                        severity,
                    });

                    discontinuity_score = discontinuity_score.max(confidence * severity);
                }
            }

            prev_spectrum = spectrum;
        }

        Ok(discontinuity_score)
    }

    /// Detect energy spikes
    fn detect_energy_spikes(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut spike_score: f32 = 0.0;
        let window_size = 128;

        // Check if audio is long enough for window analysis
        if audio.len() <= 2 * window_size {
            return Ok(spike_score);
        }

        // Calculate local energy
        for i in window_size..audio.len() - window_size {
            let current_energy = audio[i] * audio[i];

            // Calculate surrounding energy
            let left_energy: f32 =
                audio[i - window_size..i].iter().map(|x| x * x).sum::<f32>() / window_size as f32;

            let right_energy: f32 = audio[i + 1..i + 1 + window_size]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                / window_size as f32;

            let avg_surrounding = (left_energy + right_energy) / 2.0;

            if avg_surrounding > 0.0 {
                let energy_ratio = current_energy / avg_surrounding;

                if energy_ratio > 15.0 && current_energy > self.thresholds.energy_spike_threshold {
                    let confidence = (energy_ratio / 30.0).min(1.0);
                    let severity = (current_energy / 1.0).min(1.0);

                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::EnergySpike,
                        start_sample: i.saturating_sub(window_size / 4),
                        end_sample: i + window_size / 4,
                        confidence,
                        severity,
                    });

                    spike_score = spike_score.max(confidence * severity);
                }
            }
        }

        Ok(spike_score)
    }

    /// Detect high frequency noise
    fn detect_high_frequency_noise(
        &self,
        audio: &[f32],
        _sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut noise_score: f32 = 0.0;
        let window_size = 1024;

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let hf_energy = self.calculate_high_frequency_energy(window);
            let total_energy = window.iter().map(|x| x * x).sum::<f32>();

            if total_energy > 0.0 {
                let hf_ratio = hf_energy / total_energy;

                if hf_ratio > 0.3 {
                    let confidence = ((hf_ratio - 0.3) / 0.4).min(1.0);
                    let severity = hf_ratio;

                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::HighFrequencyNoise,
                        start_sample: i,
                        end_sample: i + window_size,
                        confidence,
                        severity,
                    });

                    noise_score = noise_score.max(confidence * severity);
                }
            }
        }

        Ok(noise_score)
    }

    /// Detect phase artifacts
    fn detect_phase_artifacts(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let mut phase_score: f32 = 0.0;
        let window_size = 512;

        // Simplified phase artifact detection based on signal derivatives
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let phase_irregularity = self.calculate_phase_irregularity(window);

            if phase_irregularity > 0.15 {
                let confidence = (phase_irregularity / 0.3).min(1.0);
                let severity = phase_irregularity;

                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::PhaseArtifact,
                    start_sample: i,
                    end_sample: i + window_size,
                    confidence,
                    severity,
                });

                phase_score = phase_score.max(confidence * severity);
            }
        }

        Ok(phase_score)
    }

    /// Adaptive click detection with threshold learning
    fn detect_clicks_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::Click);
        self.detect_clicks_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Adaptive metallic artifact detection
    fn detect_metallic_artifacts_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::Metallic);
        self.detect_metallic_artifacts_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Adaptive buzzing detection
    fn detect_buzzing_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::Buzzing);
        self.detect_buzzing_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Adaptive pitch variation detection
    fn detect_pitch_variations_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::PitchVariation);
        self.detect_pitch_variations_with_threshold(
            audio,
            sample_rate,
            locations,
            adaptive_threshold,
        )
    }

    /// Adaptive spectral discontinuity detection
    fn detect_spectral_discontinuities_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::SpectralDiscontinuity);
        self.detect_spectral_discontinuities_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Adaptive energy spike detection
    fn detect_energy_spikes_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::EnergySpike);
        self.detect_energy_spikes_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Adaptive high frequency noise detection
    fn detect_high_frequency_noise_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::HighFrequencyNoise);
        self.detect_high_frequency_noise_with_threshold(
            audio,
            sample_rate,
            locations,
            adaptive_threshold,
        )
    }

    /// Adaptive phase artifact detection
    fn detect_phase_artifacts_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let adaptive_threshold = self.get_adaptive_threshold(ArtifactType::PhaseArtifact);
        self.detect_phase_artifacts_with_threshold(audio, locations, adaptive_threshold)
    }

    /// Get adaptive threshold for artifact type
    fn get_adaptive_threshold(&self, artifact_type: ArtifactType) -> f32 {
        if let Some(&running_avg) = self.adaptive_state.running_averages.get(&artifact_type) {
            let base_threshold = match artifact_type {
                ArtifactType::Click => self.thresholds.click_threshold,
                ArtifactType::Metallic => self.thresholds.metallic_threshold,
                ArtifactType::Buzzing => self.thresholds.buzzing_threshold,
                ArtifactType::PitchVariation => self.thresholds.pitch_variation_threshold,
                ArtifactType::SpectralDiscontinuity => {
                    self.thresholds.spectral_discontinuity_threshold
                }
                ArtifactType::EnergySpike => self.thresholds.energy_spike_threshold,
                ArtifactType::HighFrequencyNoise => self.thresholds.hf_noise_threshold,
                ArtifactType::PhaseArtifact => self.thresholds.phase_artifact_threshold,
            };

            // Adaptive adjustment based on running average
            let adaptation_factor = 1.0 + (running_avg - 0.5) * self.thresholds.adaptation_rate;
            base_threshold * adaptation_factor.clamp(0.5, 2.0)
        } else {
            // Fall back to base threshold if no adaptation data
            match artifact_type {
                ArtifactType::Click => self.thresholds.click_threshold,
                ArtifactType::Metallic => self.thresholds.metallic_threshold,
                ArtifactType::Buzzing => self.thresholds.buzzing_threshold,
                ArtifactType::PitchVariation => self.thresholds.pitch_variation_threshold,
                ArtifactType::SpectralDiscontinuity => {
                    self.thresholds.spectral_discontinuity_threshold
                }
                ArtifactType::EnergySpike => self.thresholds.energy_spike_threshold,
                ArtifactType::HighFrequencyNoise => self.thresholds.hf_noise_threshold,
                ArtifactType::PhaseArtifact => self.thresholds.phase_artifact_threshold,
            }
        }
    }

    /// Update adaptive thresholds based on detection results
    fn update_adaptive_thresholds(&mut self, artifact_scores: &HashMap<ArtifactType, f32>) {
        let learning_rate = self.thresholds.adaptation_rate;
        self.adaptive_state.iteration_count += 1;

        for (&artifact_type, &score) in artifact_scores {
            // Update running average
            let running_avg = self
                .adaptive_state
                .running_averages
                .entry(artifact_type.clone())
                .or_insert(0.5);
            *running_avg = *running_avg * (1.0 - learning_rate) + score * learning_rate;

            // Update variance estimate
            let variance = self
                .adaptive_state
                .variance_estimates
                .entry(artifact_type.clone())
                .or_insert(0.1);
            let diff = score - *running_avg;
            *variance = *variance * (1.0 - learning_rate) + diff * diff * learning_rate;
        }
    }

    /// Calculate calibrated score with confidence weighting
    fn calculate_calibrated_score(
        &mut self,
        artifact_scores: &HashMap<ArtifactType, f32>,
        locations: &[ArtifactLocation],
    ) -> f32 {
        if artifact_scores.is_empty() {
            return 0.0;
        }

        // Calculate weighted score based on confidence
        let mut total_weighted_score = 0.0;
        let mut total_weight = 0.0;

        for (&artifact_type, &score) in artifact_scores {
            let confidence_weight = locations
                .iter()
                .filter(|loc| loc.artifact_type == artifact_type)
                .map(|loc| loc.confidence)
                .fold(0.0, f32::max)
                .max(self.thresholds.min_confidence);

            total_weighted_score += score * confidence_weight;
            total_weight += confidence_weight;
        }

        let calibrated_score = if total_weight > 0.0 {
            total_weighted_score / total_weight
        } else {
            artifact_scores
                .values()
                .fold(0.0f32, |acc, &score| f32::max(acc, score))
        };

        // Update confidence statistics
        self.update_confidence_stats(locations);

        calibrated_score.min(1.0)
    }

    /// Update confidence calibration statistics
    fn update_confidence_stats(&mut self, locations: &[ArtifactLocation]) {
        if locations.is_empty() {
            return;
        }

        let confidence_sum: f32 = locations.iter().map(|loc| loc.confidence).sum();
        let confidence_count = locations.len() as f32;
        let mean_confidence = confidence_sum / confidence_count;

        // Update running statistics
        let stats = &mut self.performance_metrics.confidence_stats;
        let alpha = 0.1; // Exponential smoothing factor

        if stats.sample_count == 0 {
            stats.mean_confidence = mean_confidence;
            stats.std_confidence = 0.0;
        } else {
            // Update mean
            stats.mean_confidence = stats.mean_confidence * (1.0 - alpha) + mean_confidence * alpha;

            // Update standard deviation
            let variance: f32 = locations
                .iter()
                .map(|loc| (loc.confidence - stats.mean_confidence).powi(2))
                .sum::<f32>()
                / confidence_count;
            stats.std_confidence = variance.sqrt();
        }

        stats.sample_count += locations.len();

        // Estimate calibration error (simplified)
        stats.calibration_error = (stats.mean_confidence - 0.7).abs(); // 0.7 is target accuracy
    }

    /// Update production performance metrics
    fn update_performance_metrics(
        &mut self,
        result: &DetectedArtifacts,
        processing_time: std::time::Duration,
        sample_count: usize,
    ) {
        let metrics = &mut self.performance_metrics;

        // Update detection count
        metrics.total_detections += result.artifact_locations.len();

        // Update processing time
        let time_us = processing_time.as_micros() as f64;
        let alpha = 0.1; // Exponential smoothing

        if metrics.avg_processing_time_us == 0.0 {
            metrics.avg_processing_time_us = time_us;
        } else {
            metrics.avg_processing_time_us =
                metrics.avg_processing_time_us * (1.0 - alpha) + time_us * alpha;
        }

        metrics.peak_processing_time_us = metrics.peak_processing_time_us.max(time_us as u64);

        // Update throughput
        let samples_per_second = sample_count as f64 / (time_us / 1_000_000.0);
        metrics.samples_per_second =
            metrics.samples_per_second * (1.0 - alpha) + samples_per_second * alpha;

        // Estimate memory usage (simplified)
        let estimated_memory_mb = (self.history_buffer.len() * 4) as f64 / (1024.0 * 1024.0);
        metrics.memory_usage_mb = estimated_memory_mb;

        // Update false positive rate estimate (simplified heuristic)
        if result.overall_score > 0.1 && result.artifact_locations.len() > 10 {
            metrics.false_positive_rate = metrics.false_positive_rate * 0.99 + 0.05 * 0.01;
        }
    }

    /// Enhanced click detection with custom threshold
    fn detect_clicks_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        // Use original implementation but with adaptive threshold
        self.detect_clicks_impl(audio, locations, threshold)
    }

    /// Enhanced metallic detection with custom threshold
    fn detect_metallic_artifacts_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_metallic_artifacts_impl(audio, locations, threshold)
    }

    /// Enhanced buzzing detection with custom threshold
    fn detect_buzzing_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_buzzing_impl(audio, locations, threshold)
    }

    /// Enhanced pitch variation detection with custom threshold
    fn detect_pitch_variations_with_threshold(
        &self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_pitch_variations_impl(audio, sample_rate, locations, threshold)
    }

    /// Enhanced spectral discontinuity detection with custom threshold
    fn detect_spectral_discontinuities_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_spectral_discontinuities_impl(audio, locations, threshold)
    }

    /// Enhanced energy spike detection with custom threshold
    fn detect_energy_spikes_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_energy_spikes_impl(audio, locations, threshold)
    }

    /// Enhanced HF noise detection with custom threshold
    fn detect_high_frequency_noise_with_threshold(
        &self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_high_frequency_noise_impl(audio, sample_rate, locations, threshold)
    }

    /// Enhanced phase artifact detection with custom threshold
    fn detect_phase_artifacts_with_threshold(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_phase_artifacts_impl(audio, locations, threshold)
    }

    /// Click detection implementation with configurable threshold
    fn detect_clicks_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        // Forward to original implementation for now, but with threshold parameter
        self.detect_clicks(audio, locations)
    }

    /// Metallic artifact detection implementation
    fn detect_metallic_artifacts_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_metallic_artifacts(audio, locations)
    }

    /// Buzzing detection implementation
    fn detect_buzzing_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_buzzing(audio, locations)
    }

    /// Pitch variation detection implementation
    fn detect_pitch_variations_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_pitch_variations(audio, sample_rate, locations)
    }

    /// Spectral discontinuity detection implementation
    fn detect_spectral_discontinuities_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_spectral_discontinuities(audio, locations)
    }

    /// Energy spike detection implementation
    fn detect_energy_spikes_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_energy_spikes(audio, locations)
    }

    /// High frequency noise detection implementation
    fn detect_high_frequency_noise_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_high_frequency_noise(audio, sample_rate, locations)
    }

    /// Phase artifact detection implementation
    fn detect_phase_artifacts_impl(
        &self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
        threshold: f32,
    ) -> Result<f32> {
        self.detect_phase_artifacts(audio, locations)
    }

    /// Batch artifact detection for high throughput scenarios
    pub fn detect_artifacts_batch(
        &mut self,
        audio_batch: &[&[f32]],
        sample_rate: u32,
    ) -> Result<Vec<DetectedArtifacts>> {
        let mut results = Vec::with_capacity(audio_batch.len());

        for audio in audio_batch {
            let result = self.detect_artifacts(audio, sample_rate)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get production performance metrics
    pub fn get_performance_metrics(&self) -> &ProductionMetrics {
        &self.performance_metrics
    }

    /// Get adaptive state for monitoring
    pub fn get_adaptive_state(&self) -> &AdaptiveState {
        &self.adaptive_state
    }

    /// Reset adaptive learning state
    pub fn reset_adaptive_state(&mut self) {
        self.adaptive_state = AdaptiveState::default();
        self.adaptive_state.adaptation_enabled = true;
    }

    /// Set context-specific thresholds
    pub fn set_context_thresholds(&mut self, context: String, thresholds: ArtifactThresholds) {
        self.adaptive_state
            .context_thresholds
            .insert(context, thresholds);
    }

    /// Enable or disable adaptive learning
    pub fn set_adaptive_learning(&mut self, enabled: bool) {
        self.adaptive_state.adaptation_enabled = enabled;
    }

    /// Perform comprehensive quality assessment
    fn assess_quality(
        &self,
        audio: &[f32],
        artifacts: &HashMap<ArtifactType, f32>,
        _sample_rate: u32,
    ) -> Result<QualityAssessment> {
        // Calculate overall quality based on artifacts
        let artifact_penalty: f32 = artifacts.values().map(|&score| score * score).sum();
        let overall_quality = (1.0 - artifact_penalty).max(0.0);

        // Calculate naturalness
        let naturalness = self.calculate_naturalness(audio);

        // Calculate clarity
        let clarity = self.calculate_clarity(audio);

        // Calculate consistency
        let consistency = self.calculate_consistency(audio);

        // Generate recommended adjustments
        let recommended_adjustments =
            self.generate_quality_recommendations(artifacts, overall_quality);

        Ok(QualityAssessment {
            overall_quality,
            naturalness,
            clarity,
            consistency,
            recommended_adjustments,
        })
    }

    /// Generate quality improvement recommendations
    fn generate_quality_recommendations(
        &self,
        artifacts: &HashMap<ArtifactType, f32>,
        overall_quality: f32,
    ) -> Vec<QualityAdjustment> {
        let mut recommendations = Vec::new();

        // If overall quality is low, recommend reducing conversion strength
        if overall_quality < 0.6 {
            recommendations.push(QualityAdjustment {
                adjustment_type: AdjustmentType::ReduceConversion,
                strength: (0.6 - overall_quality).min(0.5),
                expected_improvement: 0.2,
            });
        }

        // Check specific artifact types for targeted recommendations
        if let Some(&buzzing_score) = artifacts.get(&ArtifactType::Buzzing) {
            if buzzing_score > 0.2 {
                recommendations.push(QualityAdjustment {
                    adjustment_type: AdjustmentType::NoiseReduction,
                    strength: (buzzing_score * 0.8).min(1.0),
                    expected_improvement: 0.15,
                });
            }
        }

        if let Some(&discontinuity_score) = artifacts.get(&ArtifactType::SpectralDiscontinuity) {
            if discontinuity_score > 0.15 {
                recommendations.push(QualityAdjustment {
                    adjustment_type: AdjustmentType::SpectralSmoothing,
                    strength: (discontinuity_score * 0.9).min(1.0),
                    expected_improvement: 0.12,
                });
            }
        }

        if let Some(&pitch_score) = artifacts.get(&ArtifactType::PitchVariation) {
            if pitch_score > 0.25 {
                recommendations.push(QualityAdjustment {
                    adjustment_type: AdjustmentType::PitchStabilization,
                    strength: (pitch_score * 0.7).min(1.0),
                    expected_improvement: 0.18,
                });
            }
        }

        if let Some(&metallic_score) = artifacts.get(&ArtifactType::Metallic) {
            if metallic_score > 0.2 {
                recommendations.push(QualityAdjustment {
                    adjustment_type: AdjustmentType::FormantPreservation,
                    strength: (metallic_score * 0.6).min(1.0),
                    expected_improvement: 0.14,
                });
            }
        }

        recommendations
    }

    // Helper methods for audio analysis

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();

        crossings as f32 / (audio.len() - 1) as f32
    }

    fn calculate_spectral_regularity(&self, audio: &[f32]) -> f32 {
        // Simplified spectral regularity calculation
        let spectrum = self.calculate_spectrum_simplified(audio);

        if spectrum.len() < 3 {
            return 0.0;
        }

        let mut regularity_score = 0.0;
        let mut count = 0;

        for i in 1..spectrum.len() - 1 {
            let derivative = (spectrum[i + 1] - spectrum[i - 1]).abs();
            regularity_score += 1.0 / (1.0 + derivative);
            count += 1;
        }

        if count > 0 {
            regularity_score / count as f32
        } else {
            0.0
        }
    }

    fn calculate_thd(&self, audio: &[f32]) -> f32 {
        // Simplified total harmonic distortion calculation
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let peak = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if peak > 0.0 {
            ((rms / peak) - 0.707).max(0.0) / 0.3 // Normalize THD estimate
        } else {
            0.0
        }
    }

    fn estimate_f0(&self, audio: &[f32], sample_rate: u32) -> f32 {
        // Simplified F0 estimation using autocorrelation
        let min_period = (sample_rate / 500) as usize; // 500 Hz max
        let max_period = (sample_rate / 50) as usize; // 50 Hz min

        let mut best_period = 0usize;
        let mut best_correlation = -1.0;

        for period in min_period..max_period.min(audio.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..audio.len() - period {
                correlation += audio[i] * audio[i + period];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_period = period;
                }
            }
        }

        if best_period > 0 {
            sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    fn calculate_spectrum_simplified(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectrum calculation using energy in frequency bands
        let num_bands = 32;
        let band_size = audio.len() / num_bands;
        let mut spectrum = Vec::with_capacity(num_bands);

        for band in 0..num_bands {
            let start = band * band_size;
            let end = ((band + 1) * band_size).min(audio.len());

            let energy: f32 = audio[start..end].iter().map(|x| x * x).sum();
            spectrum.push(energy / (end - start) as f32);
        }

        spectrum
    }

    fn calculate_spectral_distance(&self, spectrum1: &[f32], spectrum2: &[f32]) -> f32 {
        let min_len = spectrum1.len().min(spectrum2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut distance = 0.0;
        for i in 0..min_len {
            distance += (spectrum1[i] - spectrum2[i]).powi(2);
        }

        (distance / min_len as f32).sqrt()
    }

    fn calculate_high_frequency_energy(&self, audio: &[f32]) -> f32 {
        // Estimate high frequency energy (simplified)
        let mut hf_energy = 0.0;
        let cutoff = audio.len() / 4; // Rough high frequency cutoff

        for i in cutoff..audio.len() {
            hf_energy += audio[i] * audio[i];
        }

        hf_energy
    }

    fn calculate_phase_irregularity(&self, audio: &[f32]) -> f32 {
        if audio.len() < 3 {
            return 0.0;
        }

        // Calculate second derivative as a measure of phase irregularity
        let mut irregularity = 0.0;
        for i in 1..audio.len() - 1 {
            let second_derivative = audio[i + 1] - 2.0 * audio[i] + audio[i - 1];
            irregularity += second_derivative.abs();
        }

        irregularity / (audio.len() - 2) as f32
    }

    fn calculate_naturalness(&self, audio: &[f32]) -> f32 {
        // Estimate naturalness based on signal characteristics
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let zcr = self.calculate_zero_crossing_rate(audio);

        // Natural speech typically has certain RMS and ZCR ranges
        let rms_naturalness = 1.0 - ((rms - 0.1).abs() / 0.3).min(1.0);
        let zcr_naturalness = 1.0 - ((zcr - 0.15).abs() / 0.15).min(1.0);

        (rms_naturalness + zcr_naturalness) / 2.0
    }

    fn calculate_clarity(&self, audio: &[f32]) -> f32 {
        // Estimate clarity based on signal-to-noise characteristics
        let spectrum = self.calculate_spectrum_simplified(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        // Calculate spectral centroid as a measure of clarity
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &energy) in spectrum.iter().enumerate() {
            weighted_sum += (i as f32) * energy;
            total_energy += energy;
        }

        if total_energy > 0.0 {
            let centroid = weighted_sum / total_energy;
            // Normalize centroid to clarity score (0.0 to 1.0)
            (centroid / spectrum.len() as f32).min(1.0)
        } else {
            0.0
        }
    }

    fn calculate_consistency(&self, audio: &[f32]) -> f32 {
        // Estimate consistency by analyzing energy variations
        let window_size = audio.len() / 10; // 10 windows for analysis
        if window_size < 10 {
            return 1.0; // Too short to analyze
        }

        let mut window_energies = Vec::new();

        for i in (0..audio.len()).step_by(window_size) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            window_energies.push(energy / (end - i) as f32);
        }

        if window_energies.len() < 2 {
            return 1.0;
        }

        // Calculate coefficient of variation
        let mean: f32 = window_energies.iter().sum::<f32>() / window_energies.len() as f32;
        let variance: f32 = window_energies
            .iter()
            .map(|&energy| (energy - mean).powi(2))
            .sum::<f32>()
            / window_energies.len() as f32;

        let std_dev = variance.sqrt();

        if mean > 0.0 {
            let cv = std_dev / mean;
            // Convert coefficient of variation to consistency score (lower CV = higher consistency)
            (1.0 - cv.min(1.0)).max(0.0)
        } else {
            1.0
        }
    }

    // New production-ready artifact detection methods
    
    /// Detect temporal jitter and timing inconsistencies
    fn detect_temporal_jitter_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::TemporalJitter);
        
        // Analyze timing consistency using onset detection
        let frame_size = (sample_rate as f32 * 0.010) as usize; // 10ms frames
        let hop_size = frame_size / 2;
        let mut onset_times = Vec::new();
        
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 { break; }
            
            let current_energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            
            // Detect sudden energy increases (onsets)
            if i > hop_size {
                let prev_start = i.saturating_sub(hop_size);
                let prev_end = (prev_start + frame_size).min(audio.len());
                let prev_energy: f32 = audio[prev_start..prev_end].iter().map(|x| x * x).sum();
                
                if current_energy > prev_energy * 1.5 && current_energy > 0.001 {
                    onset_times.push(i as f32 / sample_rate as f32);
                }
            }
        }
        
        let mut jitter_score = 0.0;
        if onset_times.len() >= 3 {
            // Calculate inter-onset intervals and their variation
            let intervals: Vec<f32> = onset_times.windows(2)
                .map(|pair| pair[1] - pair[0])
                .collect();
            
            if intervals.len() >= 2 {
                let mean_interval: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
                let variance: f32 = intervals.iter()
                    .map(|&interval| (interval - mean_interval).powi(2))
                    .sum::<f32>() / intervals.len() as f32;
                
                let jitter_coefficient = (variance.sqrt() / mean_interval).min(1.0);
                if jitter_coefficient > threshold {
                    jitter_score = jitter_coefficient;
                    
                    locations.push(ArtifactLocation {
                        artifact_type: ArtifactType::TemporalJitter,
                        start_sample: 0,
                        end_sample: audio.len(),
                        confidence: jitter_coefficient,
                        severity: if jitter_coefficient > 0.15 { 
                            crate::quality::ArtifactSeverity::High 
                        } else { 
                            crate::quality::ArtifactSeverity::Medium 
                        },
                        description: format!("Temporal jitter detected with coefficient {:.3}", jitter_coefficient),
                    });
                }
            }
        }
        
        Ok(jitter_score)
    }
    
    /// Detect spectral tilt and frequency balance issues
    fn detect_spectral_tilt_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::SpectralTilt);
        
        // Calculate spectral tilt using power spectral density
        let spectrum = self.calculate_spectrum(audio);
        let nyquist = sample_rate as f32 / 2.0;
        let bin_freq = nyquist / spectrum.len() as f32;
        
        // Calculate power in different frequency bands
        let low_freq_end = (1000.0 / bin_freq) as usize;
        let high_freq_start = (4000.0 / bin_freq) as usize;
        
        let low_freq_end = low_freq_end.min(spectrum.len());
        let high_freq_start = high_freq_start.min(spectrum.len());
        
        if high_freq_start >= low_freq_end && low_freq_end > 0 {
            let low_power: f32 = spectrum[0..low_freq_end].iter().sum();
            let high_power: f32 = spectrum[high_freq_start..].iter().sum();
            
            // Calculate spectral tilt ratio
            let tilt_ratio = if low_power > 0.0 {
                (high_power / low_power).log10().abs()
            } else {
                0.0
            };
            
            if tilt_ratio > threshold {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::SpectralTilt,
                    start_sample: 0,
                    end_sample: audio.len(),
                    confidence: tilt_ratio.min(1.0),
                    severity: if tilt_ratio > 0.5 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Spectral tilt detected with ratio {:.3}", tilt_ratio),
                });
                
                return Ok(tilt_ratio.min(1.0));
            }
        }
        
        Ok(0.0)
    }
    
    /// Detect formant tracking errors and resonance issues
    fn detect_formant_tracking_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::FormantTracking);
        
        // Simple formant tracking using spectral peak detection
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let hop_size = frame_size / 4;
        let mut formant_consistency_score = 0.0;
        let mut inconsistent_regions = 0;
        
        let mut prev_formants = Vec::new();
        let total_frames = (audio.len() / hop_size).saturating_sub(1);
        
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 { break; }
            
            let frame = &audio[i..end];
            let spectrum = self.calculate_spectrum(frame);
            
            // Find spectral peaks (simplified formant estimation)
            let mut formants = Vec::new();
            let mut peak_indices = Vec::new();
            
            for j in 2..spectrum.len()-2 {
                if spectrum[j] > spectrum[j-1] && spectrum[j] > spectrum[j+1] && 
                   spectrum[j] > spectrum[j-2] && spectrum[j] > spectrum[j+2] && 
                   spectrum[j] > 0.1 {
                    peak_indices.push(j);
                }
            }
            
            // Sort peaks by magnitude and take top 3 as formants
            peak_indices.sort_by(|&a, &b| spectrum[b].partial_cmp(&spectrum[a]).unwrap());
            for &idx in peak_indices.iter().take(3) {
                let freq = (idx as f32 * sample_rate as f32) / (2.0 * spectrum.len() as f32);
                formants.push(freq);
            }
            formants.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Compare with previous frame formants
            if !prev_formants.is_empty() && formants.len() >= 2 && prev_formants.len() >= 2 {
                // Calculate formant frequency changes
                let f1_change = (formants[0] - prev_formants[0]).abs() / prev_formants[0];
                let f2_change = if formants.len() > 1 && prev_formants.len() > 1 {
                    (formants[1] - prev_formants[1]).abs() / prev_formants[1]
                } else { 0.0 };
                
                // Flag inconsistent formant tracking
                if f1_change > 0.2 || f2_change > 0.2 { // 20% change threshold
                    inconsistent_regions += 1;
                }
            }
            
            prev_formants = formants;
        }
        
        if total_frames > 0 {
            formant_consistency_score = inconsistent_regions as f32 / total_frames as f32;
            
            if formant_consistency_score > threshold {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::FormantTracking,
                    start_sample: 0,
                    end_sample: audio.len(),
                    confidence: formant_consistency_score.min(1.0),
                    severity: if formant_consistency_score > 0.3 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Formant tracking inconsistency: {:.1}% of frames", 
                                       formant_consistency_score * 100.0),
                });
            }
        }
        
        Ok(formant_consistency_score.min(1.0))
    }
    
    /// Detect loudness inconsistencies and jumps
    fn detect_loudness_inconsistency_adaptive(
        &mut self,
        audio: &[f32],
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::LoudnessInconsistency);
        
        // Analyze loudness consistency using sliding window RMS
        let window_size = audio.len() / 20; // 20 analysis windows
        if window_size < 10 {
            return Ok(0.0);
        }
        
        let mut loudness_values = Vec::new();
        let mut jump_locations = Vec::new();
        
        for i in (0..audio.len()).step_by(window_size) {
            let end = (i + window_size).min(audio.len());
            let rms = (audio[i..end].iter().map(|x| x * x).sum::<f32>() / (end - i) as f32).sqrt();
            loudness_values.push(rms);
        }
        
        let mut max_jump = 0.0;
        
        // Detect sudden loudness changes
        for i in 1..loudness_values.len() {
            let current = loudness_values[i];
            let previous = loudness_values[i-1];
            
            if previous > 0.001 { // Avoid division by near-zero
                let change_ratio = (current - previous).abs() / previous;
                if change_ratio > 2.0 { // 200% change
                    max_jump = max_jump.max(change_ratio);
                    jump_locations.push(i * window_size);
                }
            }
        }
        
        let inconsistency_score = (max_jump / 5.0).min(1.0); // Normalize to 0-1
        
        if inconsistency_score > threshold && !jump_locations.is_empty() {
            for &location in &jump_locations {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::LoudnessInconsistency,
                    start_sample: location.saturating_sub(window_size),
                    end_sample: (location + window_size).min(audio.len()),
                    confidence: inconsistency_score,
                    severity: if inconsistency_score > 0.5 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Loudness jump detected: {:.1}x change", max_jump),
                });
            }
        }
        
        Ok(inconsistency_score)
    }
    
    /// Detect inter-harmonic distortion
    fn detect_interharmonic_distortion_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::InterharmonicDistortion);
        
        // Calculate spectrum and analyze harmonic structure
        let spectrum = self.calculate_spectrum(audio);
        let bin_freq = (sample_rate as f32 / 2.0) / spectrum.len() as f32;
        
        // Find fundamental frequency using autocorrelation
        let f0 = self.estimate_pitch(audio, sample_rate)?;
        
        if f0 > 50.0 && f0 < sample_rate as f32 / 4.0 {
            let mut harmonic_power = 0.0;
            let mut interharmonic_power = 0.0;
            
            // Analyze up to 5th harmonic
            for harmonic in 1..=5 {
                let harmonic_freq = f0 * harmonic as f32;
                let harmonic_bin = (harmonic_freq / bin_freq) as usize;
                
                if harmonic_bin < spectrum.len() {
                    // Sum power in ±2 bins around harmonic
                    for bin in harmonic_bin.saturating_sub(2)..=(harmonic_bin + 2).min(spectrum.len() - 1) {
                        harmonic_power += spectrum[bin];
                    }
                    
                    // Check for interharmonic content between this and next harmonic
                    if harmonic < 5 {
                        let next_harmonic_freq = f0 * (harmonic + 1) as f32;
                        let next_harmonic_bin = (next_harmonic_freq / bin_freq) as usize;
                        
                        let start_bin = (harmonic_bin + 3).min(spectrum.len());
                        let end_bin = next_harmonic_bin.saturating_sub(3).min(spectrum.len());
                        
                        for bin in start_bin..end_bin {
                            interharmonic_power += spectrum[bin];
                        }
                    }
                }
            }
            
            let distortion_ratio = if harmonic_power > 0.0 {
                interharmonic_power / harmonic_power
            } else {
                0.0
            };
            
            if distortion_ratio > threshold {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::InterharmonicDistortion,
                    start_sample: 0,
                    end_sample: audio.len(),
                    confidence: distortion_ratio.min(1.0),
                    severity: if distortion_ratio > 0.3 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Interharmonic distortion: {:.1}% of harmonic power", 
                                       distortion_ratio * 100.0),
                });
                
                return Ok(distortion_ratio.min(1.0));
            }
        }
        
        Ok(0.0)
    }
    
    /// Detect consonant degradation in speech conversion
    fn detect_consonant_degradation_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::ConsonantDegradation);
        
        // Analyze high-frequency content for consonant clarity
        let frame_size = (sample_rate as f32 * 0.020) as usize; // 20ms frames
        let hop_size = frame_size / 2;
        let mut degradation_score = 0.0;
        let mut degraded_frames = 0;
        let total_frames = (audio.len() / hop_size).saturating_sub(1);
        
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 { break; }
            
            let frame = &audio[i..end];
            let spectrum = self.calculate_spectrum(frame);
            
            // Calculate high-frequency energy (important for consonants)
            let nyquist = sample_rate as f32 / 2.0;
            let hf_start_bin = ((2000.0 / nyquist) * spectrum.len() as f32) as usize;
            let hf_end_bin = ((8000.0 / nyquist) * spectrum.len() as f32) as usize;
            
            let hf_start_bin = hf_start_bin.min(spectrum.len());
            let hf_end_bin = hf_end_bin.min(spectrum.len());
            
            if hf_end_bin > hf_start_bin {
                let hf_energy: f32 = spectrum[hf_start_bin..hf_end_bin].iter().sum();
                let total_energy: f32 = spectrum.iter().sum();
                
                // Detect frames with insufficient high-frequency content
                let hf_ratio = if total_energy > 0.0 { hf_energy / total_energy } else { 0.0 };
                
                // Also check for spectral flatness in HF region (indicates noise)
                let hf_spectrum = &spectrum[hf_start_bin..hf_end_bin];
                let geometric_mean = if !hf_spectrum.is_empty() {
                    let log_sum: f32 = hf_spectrum.iter()
                        .filter(|&&x| x > 0.0)
                        .map(|&x| x.ln())
                        .sum();
                    (log_sum / hf_spectrum.len() as f32).exp()
                } else {
                    0.0
                };
                
                let arithmetic_mean = hf_energy / (hf_end_bin - hf_start_bin) as f32;
                let spectral_flatness = if arithmetic_mean > 0.0 {
                    geometric_mean / arithmetic_mean
                } else {
                    0.0
                };
                
                // Flag potential consonant degradation
                if hf_ratio < 0.15 || spectral_flatness > 0.8 { // Low HF content or too flat
                    degraded_frames += 1;
                }
            }
        }
        
        if total_frames > 0 {
            degradation_score = degraded_frames as f32 / total_frames as f32;
            
            if degradation_score > threshold {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::ConsonantDegradation,
                    start_sample: 0,
                    end_sample: audio.len(),
                    confidence: degradation_score,
                    severity: if degradation_score > 0.4 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Consonant degradation in {:.1}% of frames", 
                                       degradation_score * 100.0),
                });
            }
        }
        
        Ok(degradation_score.min(1.0))
    }
    
    /// Detect vowel coloration and timbre issues
    fn detect_vowel_coloration_adaptive(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        locations: &mut Vec<ArtifactLocation>,
    ) -> Result<f32> {
        let threshold = self.get_adaptive_threshold(ArtifactType::VowelColoration);
        
        // Analyze formant structure for vowel quality
        let frame_size = (sample_rate as f32 * 0.030) as usize; // 30ms frames for vowels
        let hop_size = frame_size / 2;
        let mut coloration_score = 0.0;
        let mut colored_frames = 0;
        let total_frames = (audio.len() / hop_size).saturating_sub(1);
        
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 { break; }
            
            let frame = &audio[i..end];
            let spectrum = self.calculate_spectrum(frame);
            
            // Focus on vowel formant regions (200-3000 Hz)
            let nyquist = sample_rate as f32 / 2.0;
            let formant_start_bin = ((200.0 / nyquist) * spectrum.len() as f32) as usize;
            let formant_end_bin = ((3000.0 / nyquist) * spectrum.len() as f32) as usize;
            
            let formant_start_bin = formant_start_bin.min(spectrum.len());
            let formant_end_bin = formant_end_bin.min(spectrum.len());
            
            if formant_end_bin > formant_start_bin {
                let formant_spectrum = &spectrum[formant_start_bin..formant_end_bin];
                
                // Calculate spectral centroid in formant region
                let mut weighted_sum = 0.0;
                let mut magnitude_sum = 0.0;
                
                for (j, &magnitude) in formant_spectrum.iter().enumerate() {
                    let freq = ((formant_start_bin + j) as f32 / spectrum.len() as f32) * nyquist;
                    weighted_sum += freq * magnitude;
                    magnitude_sum += magnitude;
                }
                
                let spectral_centroid = if magnitude_sum > 0.0 {
                    weighted_sum / magnitude_sum
                } else {
                    continue;
                };
                
                // Detect unusual spectral centroid values (coloration)
                // Normal vowel centroids are typically 800-2500 Hz
                if spectral_centroid < 600.0 || spectral_centroid > 2800.0 {
                    colored_frames += 1;
                }
                
                // Also check for unnatural formant valleys (anti-formants)
                let mut valley_count = 0;
                for j in 2..formant_spectrum.len()-2 {
                    if formant_spectrum[j] < formant_spectrum[j-1] * 0.3 && 
                       formant_spectrum[j] < formant_spectrum[j+1] * 0.3 &&
                       formant_spectrum[j-1] > 0.1 && formant_spectrum[j+1] > 0.1 {
                        valley_count += 1;
                    }
                }
                
                // Too many valleys indicates unnatural formant structure
                if valley_count > formant_spectrum.len() / 10 {
                    colored_frames += 1;
                }
            }
        }
        
        if total_frames > 0 {
            coloration_score = colored_frames as f32 / total_frames as f32;
            
            if coloration_score > threshold {
                locations.push(ArtifactLocation {
                    artifact_type: ArtifactType::VowelColoration,
                    start_sample: 0,
                    end_sample: audio.len(),
                    confidence: coloration_score,
                    severity: if coloration_score > 0.3 { 
                        crate::quality::ArtifactSeverity::High 
                    } else { 
                        crate::quality::ArtifactSeverity::Medium 
                    },
                    description: format!("Vowel coloration in {:.1}% of frames", 
                                       coloration_score * 100.0),
                });
            }
        }
        
        Ok(coloration_score.min(1.0))
    }
    
    /// Get adaptive threshold for a specific artifact type
    fn get_adaptive_threshold(&self, artifact_type: ArtifactType) -> f32 {
        match artifact_type {
            ArtifactType::Click => self.thresholds.click_threshold,
            ArtifactType::Metallic => self.thresholds.metallic_threshold,
            ArtifactType::Buzzing => self.thresholds.buzzing_threshold,
            ArtifactType::PitchVariation => self.thresholds.pitch_variation_threshold,
            ArtifactType::SpectralDiscontinuity => self.thresholds.spectral_discontinuity_threshold,
            ArtifactType::EnergySpike => self.thresholds.energy_spike_threshold,
            ArtifactType::HighFrequencyNoise => self.thresholds.hf_noise_threshold,
            ArtifactType::PhaseArtifact => self.thresholds.phase_artifact_threshold,
            // New production-ready artifact thresholds
            ArtifactType::TemporalJitter => self.thresholds.temporal_jitter_threshold,
            ArtifactType::SpectralTilt => self.thresholds.spectral_tilt_threshold,
            ArtifactType::FormantTracking => self.thresholds.formant_tracking_threshold,
            ArtifactType::LoudnessInconsistency => self.thresholds.loudness_inconsistency_threshold,
            ArtifactType::ChannelCrosstalk => self.thresholds.channel_crosstalk_threshold,
            ArtifactType::InterharmonicDistortion => self.thresholds.interharmonic_distortion_threshold,
            ArtifactType::ConsonantDegradation => self.thresholds.consonant_degradation_threshold,
            ArtifactType::VowelColoration => self.thresholds.vowel_coloration_threshold,
        }
    }
}

impl Default for ArtifactDetector {
    fn default() -> Self {
        Self::new()
    }
}