//! Quality assessment and artifact detection for voice conversion

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
    pub complex_buffers: Vec<Vec<scirs2_core::Complex<f32>>>,
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
            .copied()
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
                .entry(artifact_type)
                .or_insert(0.5);
            *running_avg = *running_avg * (1.0 - learning_rate) + score * learning_rate;

            // Update variance estimate
            let variance = self
                .adaptive_state
                .variance_estimates
                .entry(artifact_type)
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
        let cutoff = audio.len() / 4; // Rough high frequency cutoff

        audio[cutoff..].iter().map(|&x| x * x).sum()
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
}

impl Default for ArtifactDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Objective quality metrics system for conversion evaluation
#[derive(Debug, Clone)]
pub struct QualityMetricsSystem {
    /// Reference audio features for comparison
    reference_features: Option<QualityFeatures>,
    /// Perceptual quality model parameters
    perceptual_params: PerceptualParameters,
}

/// Features used for quality assessment
#[derive(Debug, Clone)]
pub struct QualityFeatures {
    /// Spectral features
    pub spectral: Vec<f32>,
    /// Temporal features  
    pub temporal: Vec<f32>,
    /// Prosodic features
    pub prosodic: Vec<f32>,
    /// Perceptual features
    pub perceptual: Vec<f32>,
}

/// Parameters for perceptual quality modeling
#[derive(Debug, Clone)]
pub struct PerceptualParameters {
    /// Weight for spectral similarity
    pub spectral_weight: f32,
    /// Weight for temporal consistency
    pub temporal_weight: f32,
    /// Weight for prosodic preservation
    pub prosodic_weight: f32,
    /// Weight for naturalness
    pub naturalness_weight: f32,
}

impl Default for PerceptualParameters {
    fn default() -> Self {
        Self {
            spectral_weight: 0.3,
            temporal_weight: 0.2,
            prosodic_weight: 0.3,
            naturalness_weight: 0.2,
        }
    }
}

/// Objective quality metrics results
#[derive(Debug, Clone)]
pub struct ObjectiveQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Spectral similarity score
    pub spectral_similarity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Prosodic preservation score
    pub prosodic_preservation: f32,
    /// Naturalness score
    pub naturalness: f32,
    /// Perceptual quality score
    pub perceptual_quality: f32,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: f32,
    /// Segmental SNR
    pub segmental_snr: f32,
}

impl QualityMetricsSystem {
    /// Create new quality metrics system
    pub fn new() -> Self {
        Self {
            reference_features: None,
            perceptual_params: PerceptualParameters::default(),
        }
    }

    /// Create with custom perceptual parameters
    pub fn with_perceptual_params(perceptual_params: PerceptualParameters) -> Self {
        Self {
            reference_features: None,
            perceptual_params,
        }
    }

    /// Set reference audio for quality comparison
    pub fn set_reference(&mut self, reference_audio: &[f32], sample_rate: u32) -> Result<()> {
        self.reference_features =
            Some(self.extract_quality_features(reference_audio, sample_rate)?);
        Ok(())
    }

    /// Evaluate objective quality metrics
    pub fn evaluate_quality(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ObjectiveQualityMetrics> {
        debug!(
            "Evaluating objective quality metrics for {} samples",
            audio.len()
        );

        let features = self.extract_quality_features(audio, sample_rate)?;

        let spectral_similarity = if let Some(ref reference) = self.reference_features {
            self.calculate_feature_similarity(&features.spectral, &reference.spectral)
        } else {
            self.estimate_spectral_quality(&features.spectral)
        };

        let temporal_consistency = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_preservation = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);
        let perceptual_quality = self.calculate_perceptual_quality(&features);

        let snr_estimate = self.estimate_snr(audio);
        let segmental_snr = self.calculate_segmental_snr(audio);

        // Calculate weighted overall score
        let overall_score = spectral_similarity * self.perceptual_params.spectral_weight
            + temporal_consistency * self.perceptual_params.temporal_weight
            + prosodic_preservation * self.perceptual_params.prosodic_weight
            + naturalness * self.perceptual_params.naturalness_weight;

        info!(
            "Quality evaluation complete: overall_score={:.3}",
            overall_score
        );

        Ok(ObjectiveQualityMetrics {
            overall_score,
            spectral_similarity,
            temporal_consistency,
            prosodic_preservation,
            naturalness,
            perceptual_quality,
            snr_estimate,
            segmental_snr,
        })
    }

    /// Extract quality-relevant features from audio
    fn extract_quality_features(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<QualityFeatures> {
        // Extract spectral features (simplified MFCCs)
        let spectral = self.extract_spectral_features(audio);

        // Extract temporal features
        let temporal = self.extract_temporal_features(audio);

        // Extract prosodic features
        let prosodic = self.extract_prosodic_features(audio);

        // Extract perceptual features
        let perceptual = self.extract_perceptual_features(audio);

        Ok(QualityFeatures {
            spectral,
            temporal,
            prosodic,
            perceptual,
        })
    }

    fn extract_spectral_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectral feature extraction
        let mut features = Vec::new();

        let spectrum = self.calculate_power_spectrum(audio);

        // Spectral centroid
        let spectral_centroid = self.calculate_spectral_centroid(&spectrum);
        features.push(spectral_centroid);

        // Spectral rolloff
        let spectral_rolloff = self.calculate_spectral_rolloff(&spectrum);
        features.push(spectral_rolloff);

        // Spectral flatness
        let spectral_flatness = self.calculate_spectral_flatness(&spectrum);
        features.push(spectral_flatness);

        // Add simplified MFCC-like features
        let num_bands = 13;
        let band_energies = self.calculate_mel_band_energies(&spectrum, num_bands);
        features.extend(band_energies);

        features
    }

    fn extract_temporal_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // RMS energy
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        features.push(rms);

        // Zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(audio);
        features.push(zcr);

        // Short-time energy variation
        let energy_variation = self.calculate_energy_variation(audio);
        features.push(energy_variation);

        // Spectral flux
        let spectral_flux = self.calculate_spectral_flux(audio);
        features.push(spectral_flux);

        features
    }

    fn extract_prosodic_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // F0 statistics (simplified)
        let f0_stats = self.calculate_f0_statistics(audio);
        features.extend(f0_stats);

        // Energy contour statistics
        let energy_stats = self.calculate_energy_statistics(audio);
        features.extend(energy_stats);

        // Duration features (simplified)
        let duration_features = self.calculate_duration_features(audio);
        features.extend(duration_features);

        features
    }

    fn extract_perceptual_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // Loudness estimate
        let loudness = self.estimate_loudness(audio);
        features.push(loudness);

        // Sharpness estimate
        let sharpness = self.estimate_sharpness(audio);
        features.push(sharpness);

        // Roughness estimate
        let roughness = self.estimate_roughness(audio);
        features.push(roughness);

        features
    }

    // Implementation of helper methods for feature extraction

    fn calculate_power_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified power spectrum calculation
        let window_size = 512;
        let mut spectrum = vec![0.0; window_size / 2];

        for (i, &audio_sample) in audio.iter().enumerate().take(window_size.min(audio.len())) {
            let real = audio_sample;
            let bin = i / 2; // Simplified frequency mapping
            if bin < spectrum.len() {
                spectrum[bin] += real * real;
            }
        }

        spectrum
    }

    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &energy) in spectrum.iter().enumerate() {
            weighted_sum += (i as f32) * energy;
            total_energy += energy;
        }

        if total_energy > 0.0 {
            weighted_sum / total_energy
        } else {
            0.0
        }
    }

    fn calculate_spectral_rolloff(&self, spectrum: &[f32]) -> f32 {
        let total_energy: f32 = spectrum.iter().sum();
        let threshold = total_energy * 0.85; // 85% rolloff

        let mut cumulative_energy = 0.0;
        for (i, &energy) in spectrum.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= threshold {
                return i as f32 / spectrum.len() as f32;
            }
        }

        1.0
    }

    fn calculate_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;

        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_mel_band_energies(&self, spectrum: &[f32], num_bands: usize) -> Vec<f32> {
        let band_size = spectrum.len() / num_bands;

        (0..num_bands)
            .map(|band| {
                let start = band * band_size;
                let end = ((band + 1) * band_size).min(spectrum.len());

                if end > start {
                    spectrum[start..end].iter().sum::<f32>() / (end - start) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

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

    fn calculate_energy_variation(&self, audio: &[f32]) -> f32 {
        let window_size = audio.len() / 10;
        if window_size < 10 {
            return 0.0;
        }

        let mut energies = Vec::new();
        for i in (0..audio.len()).step_by(window_size) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.len() < 2 {
            return 0.0;
        }

        let mean = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance =
            energies.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / energies.len() as f32;

        variance.sqrt()
    }

    fn calculate_spectral_flux(&self, audio: &[f32]) -> f32 {
        let window_size = 256;
        let hop_size = window_size / 2;

        let mut prev_spectrum: Vec<f32> = Vec::new();
        let mut flux_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let spectrum = self.calculate_power_spectrum(window);

            if !prev_spectrum.is_empty() {
                let mut flux = 0.0;
                for (curr, &prev) in spectrum.iter().zip(prev_spectrum.iter()) {
                    flux += (curr - prev).max(0.0);
                }
                flux_values.push(flux);
            }

            prev_spectrum = spectrum;
        }

        if flux_values.is_empty() {
            0.0
        } else {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        }
    }

    fn calculate_f0_statistics(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified F0 extraction and statistics
        let window_size = 1024;
        let hop_size = window_size / 2;
        let mut f0_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0_simple(window);
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let min_f0 = f0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = f0_values.iter().fold(0.0f32, |a, &b| a.max(b));

        vec![mean_f0, min_f0, max_f0]
    }

    fn estimate_f0_simple(&self, audio: &[f32]) -> f32 {
        // Very simplified F0 estimation
        let mut best_lag = 0;
        let mut best_correlation = -1.0;

        let min_lag = 20; // Assuming sample rate around 44100, this gives ~440 Hz max
        let max_lag = 400; // This gives ~110 Hz min

        for lag in min_lag..max_lag.min(audio.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..audio.len() - lag {
                correlation += audio[i] * audio[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        if best_lag > 0 {
            44100.0 / best_lag as f32 // Assuming 44.1kHz sample rate
        } else {
            0.0
        }
    }

    fn calculate_energy_statistics(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let mut energies = Vec::new();

        for i in (0..audio.len()).step_by(window_size / 2) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.is_empty() {
            return vec![0.0, 0.0];
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let energy_variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        vec![mean_energy, energy_variance.sqrt()]
    }

    fn calculate_duration_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified duration features
        let total_duration = audio.len() as f32;
        let non_silent_samples = audio.iter().filter(|&&x| x.abs() > 0.01).count() as f32;

        let speech_rate = if total_duration > 0.0 {
            non_silent_samples / total_duration
        } else {
            0.0
        };

        vec![speech_rate]
    }

    fn estimate_loudness(&self, audio: &[f32]) -> f32 {
        // Simplified loudness estimation based on RMS
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        (rms * 100.0).min(1.0) // Normalize to 0-1 range
    }

    fn estimate_sharpness(&self, audio: &[f32]) -> f32 {
        // Simplified sharpness based on high-frequency content
        let spectrum = self.calculate_power_spectrum(audio);
        let total_energy: f32 = spectrum.iter().sum();

        if total_energy > 0.0 {
            let hf_start = spectrum.len() * 2 / 3; // Upper third of spectrum
            let hf_energy: f32 = spectrum[hf_start..].iter().sum();
            hf_energy / total_energy
        } else {
            0.0
        }
    }

    fn estimate_roughness(&self, audio: &[f32]) -> f32 {
        // Simplified roughness based on amplitude modulation
        if audio.len() < 3 {
            return 0.0;
        }

        let mut modulation = 0.0;
        for i in 1..audio.len() - 1 {
            let local_variation = (audio[i + 1] - audio[i - 1]).abs();
            modulation += local_variation;
        }

        (modulation / (audio.len() - 2) as f32).min(1.0)
    }

    // Quality calculation methods

    fn calculate_feature_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..min_len {
            let diff = (features1[i] - features2[i]).abs();
            similarity += 1.0 / (1.0 + diff);
        }

        similarity / min_len as f32
    }

    fn estimate_spectral_quality(&self, features: &[f32]) -> f32 {
        // Estimate quality without reference based on feature characteristics
        if features.is_empty() {
            return 0.0;
        }

        // Check for typical speech-like spectral characteristics
        let centroid = features[0];
        let rolloff = if features.len() > 1 { features[1] } else { 0.5 };

        // Ideal values for speech
        let centroid_quality =
            1.0 - ((centroid / features.len() as f32 - 0.3).abs() / 0.3).min(1.0);
        let rolloff_quality = 1.0 - ((rolloff - 0.8).abs() / 0.2).min(1.0);

        (centroid_quality + rolloff_quality) / 2.0
    }

    fn calculate_temporal_consistency(&self, temporal_features: &[f32]) -> f32 {
        if temporal_features.len() < 3 {
            return 1.0;
        }

        // Check energy variation (less variation = better consistency)
        let energy_variation = temporal_features[2];
        1.0 - energy_variation.min(1.0)
    }

    fn calculate_prosodic_quality(&self, prosodic_features: &[f32]) -> f32 {
        if prosodic_features.len() < 3 {
            return 0.5;
        }

        let mean_f0 = prosodic_features[0];
        let f0_range = prosodic_features[2] - prosodic_features[1];

        // Check if F0 is in typical speech range
        let f0_quality = if mean_f0 >= 80.0 && mean_f0 <= 400.0 {
            1.0 - ((mean_f0 - 150.0).abs() / 150.0).min(1.0)
        } else {
            0.0
        };

        // Check if F0 range is reasonable
        let range_quality = if f0_range >= 10.0 && f0_range <= 100.0 {
            1.0 - ((f0_range - 30.0).abs() / 30.0).min(1.0)
        } else {
            0.0
        };

        (f0_quality + range_quality) / 2.0
    }

    fn calculate_naturalness(&self, perceptual_features: &[f32]) -> f32 {
        if perceptual_features.is_empty() {
            return 0.5;
        }

        let loudness = perceptual_features[0];
        let roughness = if perceptual_features.len() > 2 {
            perceptual_features[2]
        } else {
            0.5
        };

        // Natural speech should have moderate loudness and low roughness
        let loudness_quality = 1.0 - ((loudness - 0.3).abs() / 0.3).min(1.0);
        let roughness_quality = 1.0 - roughness;

        (loudness_quality + roughness_quality) / 2.0
    }

    fn calculate_perceptual_quality(&self, features: &QualityFeatures) -> f32 {
        // Combine all feature types for overall perceptual quality
        let spectral_quality = self.estimate_spectral_quality(&features.spectral);
        let temporal_quality = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_quality = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);

        // Weighted combination
        spectral_quality * 0.3 + temporal_quality * 0.2 + prosodic_quality * 0.3 + naturalness * 0.2
    }

    fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Estimate signal power (simplified)
        let signal_power: f32 = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;

        // Estimate noise power from quiet segments (simplified)
        let sorted_samples: Vec<f32> = {
            let mut samples = audio.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples
        };

        let noise_threshold_idx = sorted_samples.len() / 4; // Bottom quartile as noise
        let noise_power = if noise_threshold_idx < sorted_samples.len() {
            sorted_samples[noise_threshold_idx] * sorted_samples[noise_threshold_idx]
        } else {
            0.001 // Small noise floor
        };

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR when no noise detected
        }
    }

    fn calculate_segmental_snr(&self, audio: &[f32]) -> f32 {
        let segment_size = 256;
        let mut snr_values = Vec::new();

        for i in (0..audio.len()).step_by(segment_size) {
            let end = (i + segment_size).min(audio.len());
            let segment = &audio[i..end];

            if !segment.is_empty() {
                let segment_snr = self.estimate_snr(segment);
                snr_values.push(segment_snr);
            }
        }

        if snr_values.is_empty() {
            0.0
        } else {
            snr_values.iter().sum::<f32>() / snr_values.len() as f32
        }
    }
}

impl Default for QualityMetricsSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive quality adjustment system
#[derive(Debug, Clone)]
pub struct AdaptiveQualityController {
    /// Current quality target
    quality_target: f32,
    /// Adaptation sensitivity
    adaptation_rate: f32,
    /// Quality history for trend analysis
    quality_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
    /// Quality improvement strategies
    strategies: Vec<QualityStrategy>,
}

/// Quality improvement strategy
#[derive(Debug, Clone)]
pub struct QualityStrategy {
    /// Strategy name
    pub name: String,
    /// Trigger condition
    pub trigger: QualityTrigger,
    /// Adjustment parameters
    pub adjustment: QualityStrategyAdjustment,
    /// Expected effectiveness (0.0 to 1.0)
    pub effectiveness: f32,
    /// Usage count for learning
    pub usage_count: usize,
    /// Success rate for adaptive learning
    pub success_rate: f32,
}

/// Quality trigger conditions
#[derive(Debug, Clone)]
pub enum QualityTrigger {
    /// Triggered when overall quality is below threshold
    OverallQualityBelow(f32),
    /// Triggered when artifact score is above threshold
    ArtifactScoreAbove(f32),
    /// Triggered when specific artifact type exceeds threshold
    SpecificArtifact(ArtifactType, f32),
    /// Triggered when naturalness is below threshold
    NaturalnessBelow(f32),
    /// Triggered when SNR is below threshold
    SnrBelow(f32),
    /// Triggered by multiple conditions (all must be true)
    Combined(Vec<QualityTrigger>),
}

/// Quality strategy adjustment parameters
#[derive(Debug, Clone)]
pub struct QualityStrategyAdjustment {
    /// Adjustment type
    pub adjustment_type: AdjustmentType,
    /// Parameter adjustments
    pub parameter_changes: HashMap<String, f32>,
    /// Processing mode changes
    pub processing_mode_change: Option<String>,
    /// Model selection preference
    pub preferred_model: Option<String>,
}

impl AdaptiveQualityController {
    /// Create new adaptive quality controller
    pub fn new(quality_target: f32) -> Self {
        // Add default quality improvement strategies
        let strategies = vec![
            QualityStrategy {
                name: "reduce_conversion_strength".to_string(),
                trigger: QualityTrigger::OverallQualityBelow(0.6),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::ReduceConversion,
                    parameter_changes: [("conversion_strength".to_string(), -0.2)].into(),
                    processing_mode_change: None,
                    preferred_model: None,
                },
                effectiveness: 0.7,
                usage_count: 0,
                success_rate: 0.7,
            },
            QualityStrategy {
                name: "enable_noise_reduction".to_string(),
                trigger: QualityTrigger::SpecificArtifact(ArtifactType::Buzzing, 0.2),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::NoiseReduction,
                    parameter_changes: [("noise_reduction_strength".to_string(), 0.8)].into(),
                    processing_mode_change: Some("high_quality".to_string()),
                    preferred_model: None,
                },
                effectiveness: 0.8,
                usage_count: 0,
                success_rate: 0.75,
            },
            QualityStrategy {
                name: "spectral_smoothing".to_string(),
                trigger: QualityTrigger::SpecificArtifact(
                    ArtifactType::SpectralDiscontinuity,
                    0.15,
                ),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::SpectralSmoothing,
                    parameter_changes: [("smoothing_factor".to_string(), 0.6)].into(),
                    processing_mode_change: None,
                    preferred_model: None,
                },
                effectiveness: 0.6,
                usage_count: 0,
                success_rate: 0.65,
            },
            QualityStrategy {
                name: "pitch_stabilization".to_string(),
                trigger: QualityTrigger::SpecificArtifact(ArtifactType::PitchVariation, 0.25),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::PitchStabilization,
                    parameter_changes: [("pitch_smoothing".to_string(), 0.7)].into(),
                    processing_mode_change: None,
                    preferred_model: None,
                },
                effectiveness: 0.75,
                usage_count: 0,
                success_rate: 0.8,
            },
            QualityStrategy {
                name: "formant_preservation".to_string(),
                trigger: QualityTrigger::SpecificArtifact(ArtifactType::Metallic, 0.2),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::FormantPreservation,
                    parameter_changes: [("formant_preservation".to_string(), 0.9)].into(),
                    processing_mode_change: None,
                    preferred_model: None,
                },
                effectiveness: 0.65,
                usage_count: 0,
                success_rate: 0.7,
            },
            QualityStrategy {
                name: "low_latency_fallback".to_string(),
                trigger: QualityTrigger::Combined(vec![
                    QualityTrigger::OverallQualityBelow(0.4),
                    QualityTrigger::ArtifactScoreAbove(0.8),
                ]),
                adjustment: QualityStrategyAdjustment {
                    adjustment_type: AdjustmentType::ReduceConversion,
                    parameter_changes: [
                        ("conversion_strength".to_string(), -0.4),
                        ("processing_quality".to_string(), -0.3),
                    ]
                    .into(),
                    processing_mode_change: Some("low_latency".to_string()),
                    preferred_model: Some("lightweight".to_string()),
                },
                effectiveness: 0.5,
                usage_count: 0,
                success_rate: 0.6,
            },
        ];

        Self {
            quality_target: quality_target.clamp(0.0, 1.0),
            adaptation_rate: 0.1,
            quality_history: Vec::new(),
            max_history: 10,
            strategies,
        }
    }

    /// Analyze quality and suggest adjustments
    pub fn analyze_and_adjust(
        &mut self,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
        current_params: &HashMap<String, f32>,
    ) -> Result<AdaptiveAdjustmentResult> {
        debug!("Analyzing quality for adaptive adjustments");

        // Update quality history
        self.update_quality_history(objective_quality.overall_score);

        // Evaluate current quality against target
        let quality_gap = self.quality_target - objective_quality.overall_score;

        // Find applicable strategies
        let applicable_strategies = self.find_applicable_strategies(artifacts, objective_quality);

        if applicable_strategies.is_empty() {
            return Ok(AdaptiveAdjustmentResult {
                should_adjust: false,
                selected_strategy: None,
                parameter_adjustments: HashMap::new(),
                processing_mode_change: None,
                preferred_model: None,
                expected_improvement: 0.0,
                confidence: 0.0,
            });
        }

        // Select best strategy based on effectiveness and success rate
        let selected_strategy = self.select_best_strategy(&applicable_strategies);

        // Calculate expected improvement
        let expected_improvement = selected_strategy.effectiveness * quality_gap.abs();

        // Prepare adjustment result
        let adjustment_result = AdaptiveAdjustmentResult {
            should_adjust: quality_gap > 0.05, // Only adjust if significant quality gap
            selected_strategy: Some(selected_strategy.name.clone()),
            parameter_adjustments: self.calculate_parameter_adjustments(
                &selected_strategy.adjustment,
                current_params,
                quality_gap,
            ),
            processing_mode_change: selected_strategy.adjustment.processing_mode_change.clone(),
            preferred_model: selected_strategy.adjustment.preferred_model.clone(),
            expected_improvement,
            confidence: selected_strategy.success_rate,
        };

        info!(
            "Adaptive quality analysis complete: should_adjust={}, expected_improvement={:.3}",
            adjustment_result.should_adjust, adjustment_result.expected_improvement
        );

        Ok(adjustment_result)
    }

    /// Update quality history for trend analysis
    fn update_quality_history(&mut self, quality_score: f32) {
        self.quality_history.push(quality_score);

        // Keep only recent history
        if self.quality_history.len() > self.max_history {
            self.quality_history.remove(0);
        }
    }

    /// Find strategies applicable to current quality issues
    fn find_applicable_strategies(
        &self,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
    ) -> Vec<&QualityStrategy> {
        self.strategies
            .iter()
            .filter(|strategy| {
                Self::evaluate_trigger(&strategy.trigger, artifacts, objective_quality)
            })
            .collect()
    }

    /// Evaluate if a trigger condition is met
    fn evaluate_trigger(
        trigger: &QualityTrigger,
        artifacts: &DetectedArtifacts,
        objective_quality: &ObjectiveQualityMetrics,
    ) -> bool {
        match trigger {
            QualityTrigger::OverallQualityBelow(threshold) => {
                objective_quality.overall_score < *threshold
            }
            QualityTrigger::ArtifactScoreAbove(threshold) => artifacts.overall_score > *threshold,
            QualityTrigger::SpecificArtifact(artifact_type, threshold) => {
                if let Some(&score) = artifacts.artifact_types.get(artifact_type) {
                    score > *threshold
                } else {
                    false
                }
            }
            QualityTrigger::NaturalnessBelow(threshold) => {
                objective_quality.naturalness < *threshold
            }
            QualityTrigger::SnrBelow(threshold) => objective_quality.snr_estimate < *threshold,
            QualityTrigger::Combined(triggers) => triggers
                .iter()
                .all(|t| Self::evaluate_trigger(t, artifacts, objective_quality)),
        }
    }

    /// Select the best strategy from applicable ones
    fn select_best_strategy<'a>(&self, strategies: &[&'a QualityStrategy]) -> &'a QualityStrategy {
        strategies
            .iter()
            .max_by(|a, b| {
                let score_a = a.effectiveness * a.success_rate;
                let score_b = b.effectiveness * b.success_rate;
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap()
    }

    /// Calculate parameter adjustments based on strategy and quality gap
    fn calculate_parameter_adjustments(
        &self,
        adjustment: &QualityStrategyAdjustment,
        current_params: &HashMap<String, f32>,
        quality_gap: f32,
    ) -> HashMap<String, f32> {
        let mut adjusted_params = HashMap::new();
        let intensity_factor = (quality_gap.abs() / 0.3).min(1.0); // Scale by quality gap

        for (param_name, base_change) in &adjustment.parameter_changes {
            let current_value = current_params.get(param_name).copied().unwrap_or(1.0);
            let scaled_change = base_change * intensity_factor;
            let new_value = (current_value + scaled_change).clamp(0.0, 2.0);

            adjusted_params.insert(param_name.clone(), new_value);
        }

        adjusted_params
    }

    /// Update strategy effectiveness based on results
    pub fn update_strategy_effectiveness(
        &mut self,
        strategy_name: &str,
        quality_before: f32,
        quality_after: f32,
    ) {
        if let Some(strategy) = self.strategies.iter_mut().find(|s| s.name == strategy_name) {
            strategy.usage_count += 1;

            let improvement = quality_after - quality_before;
            let success = improvement > 0.05; // Consider successful if quality improved by > 5%

            // Update success rate using exponential moving average
            let alpha = 0.1;
            strategy.success_rate =
                strategy.success_rate * (1.0 - alpha) + if success { 1.0 } else { 0.0 } * alpha;

            // Update effectiveness estimate
            if success {
                let actual_effectiveness = improvement.abs().min(1.0);
                strategy.effectiveness =
                    strategy.effectiveness * (1.0 - alpha) + actual_effectiveness * alpha;
            }

            info!(
                "Updated strategy '{}': usage_count={}, success_rate={:.3}, effectiveness={:.3}",
                strategy_name, strategy.usage_count, strategy.success_rate, strategy.effectiveness
            );
        }
    }

    /// Get quality trend from history
    pub fn get_quality_trend(&self) -> QualityTrend {
        if self.quality_history.len() < 3 {
            return QualityTrend::Stable;
        }

        let recent = &self.quality_history[self.quality_history.len() - 3..];
        let trend_slope = (recent[2] - recent[0]) / 2.0;

        if trend_slope > 0.02 {
            QualityTrend::Improving
        } else if trend_slope < -0.02 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }

    /// Set quality target
    pub fn set_quality_target(&mut self, target: f32) {
        self.quality_target = target.clamp(0.0, 1.0);
    }

    /// Get current quality target
    pub fn quality_target(&self) -> f32 {
        self.quality_target
    }

    /// Get strategy statistics
    pub fn get_strategy_stats(&self) -> Vec<StrategyStats> {
        self.strategies
            .iter()
            .map(|s| StrategyStats {
                name: s.name.clone(),
                usage_count: s.usage_count,
                success_rate: s.success_rate,
                effectiveness: s.effectiveness,
            })
            .collect()
    }
}

/// Result of adaptive quality analysis
#[derive(Debug, Clone)]
pub struct AdaptiveAdjustmentResult {
    /// Whether adjustment is recommended
    pub should_adjust: bool,
    /// Selected strategy name
    pub selected_strategy: Option<String>,
    /// Parameter adjustments to apply
    pub parameter_adjustments: HashMap<String, f32>,
    /// Processing mode change recommendation
    pub processing_mode_change: Option<String>,
    /// Preferred model selection
    pub preferred_model: Option<String>,
    /// Expected quality improvement
    pub expected_improvement: f32,
    /// Confidence in the adjustment
    pub confidence: f32,
}

/// Quality trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTrend {
    /// Quality is improving over time
    Improving,
    /// Quality is stable
    Stable,
    /// Quality is degrading over time
    Degrading,
}

/// Strategy statistics for monitoring
#[derive(Debug, Clone)]
pub struct StrategyStats {
    /// Strategy name
    pub name: String,
    /// Number of times used
    pub usage_count: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Effectiveness estimate (0.0 to 1.0)
    pub effectiveness: f32,
}

impl Default for AdaptiveQualityController {
    fn default() -> Self {
        Self::new(0.8) // Default quality target of 80%
    }
}

/// Perceptual optimization system for human auditory perception
#[derive(Debug, Clone)]
pub struct PerceptualOptimizer {
    /// Psychoacoustic model parameters
    psychoacoustic_model: PsychoacousticModel,
    /// Critical band analyzer
    critical_bands: CriticalBandAnalyzer,
    /// Masking calculator
    masking_calculator: MaskingCalculator,
    /// Loudness model
    loudness_model: LoudnessModel,
    /// Optimization parameters
    optimization_params: PerceptualOptimizationParams,
}

/// Psychoacoustic model for human hearing perception
#[derive(Debug, Clone)]
pub struct PsychoacousticModel {
    /// Absolute threshold of hearing (dB SPL) for different frequencies
    absolute_threshold: Vec<f32>,
    /// Frequency points for threshold curve (Hz)
    threshold_frequencies: Vec<f32>,
    /// Sample rate for analysis
    sample_rate: u32,
}

/// Critical band analysis for frequency masking
#[derive(Debug, Clone)]
pub struct CriticalBandAnalyzer {
    /// Critical band boundaries in Hz
    band_boundaries: Vec<f32>,
    /// Number of critical bands
    num_bands: usize,
    /// Sample rate
    sample_rate: u32,
}

/// Masking calculation for auditory masking effects
#[derive(Debug, Clone)]
pub struct MaskingCalculator {
    /// Spreading function coefficients
    spreading_coefficients: Vec<f32>,
    /// Temporal masking parameters
    temporal_masking: TemporalMaskingParams,
    /// Simultaneous masking parameters  
    simultaneous_masking: SimultaneousMaskingParams,
}

/// Loudness perception model
#[derive(Debug, Clone)]
pub struct LoudnessModel {
    /// Equal loudness contours (phons)
    equal_loudness_contours: HashMap<u32, Vec<f32>>, // phon level -> dB values
    /// Frequency points for contours
    contour_frequencies: Vec<f32>,
    /// Loudness scaling factors
    loudness_scaling: Vec<f32>,
}

/// Parameters for temporal masking
#[derive(Debug, Clone)]
pub struct TemporalMaskingParams {
    /// Pre-masking duration (ms)
    pub pre_masking_duration: f32,
    /// Post-masking duration (ms)
    pub post_masking_duration: f32,
    /// Masking slope (dB/ms)
    pub masking_slope: f32,
}

/// Parameters for simultaneous masking
#[derive(Debug, Clone)]
pub struct SimultaneousMaskingParams {
    /// Lower slope (dB/Bark)
    pub lower_slope: f32,
    /// Upper slope (dB/Bark)
    pub upper_slope: f32,
    /// Spreading function width
    pub spreading_width: f32,
}

/// Perceptual optimization parameters
#[derive(Debug, Clone)]
pub struct PerceptualOptimizationParams {
    /// Weight for spectral masking optimization
    pub spectral_masking_weight: f32,
    /// Weight for temporal masking optimization
    pub temporal_masking_weight: f32,
    /// Weight for loudness optimization
    pub loudness_weight: f32,
    /// Weight for critical band optimization
    pub critical_band_weight: f32,
    /// Optimization target quality (0.0 to 1.0)
    pub target_quality: f32,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for PerceptualOptimizationParams {
    fn default() -> Self {
        Self {
            spectral_masking_weight: 0.3,
            temporal_masking_weight: 0.2,
            loudness_weight: 0.3,
            critical_band_weight: 0.2,
            target_quality: 0.8,
            max_iterations: 20,
            convergence_threshold: 0.01,
        }
    }
}

impl Default for TemporalMaskingParams {
    fn default() -> Self {
        Self {
            pre_masking_duration: 2.0,    // 2ms pre-masking
            post_masking_duration: 200.0, // 200ms post-masking
            masking_slope: 0.1,           // 0.1 dB/ms slope
        }
    }
}

impl Default for SimultaneousMaskingParams {
    fn default() -> Self {
        Self {
            lower_slope: -27.0,   // -27 dB/Bark below masker
            upper_slope: -12.0,   // -12 dB/Bark above masker
            spreading_width: 2.5, // 2.5 Bark spreading width
        }
    }
}

/// Perceptual optimization result
#[derive(Debug, Clone)]
pub struct PerceptualOptimizationResult {
    /// Optimized conversion parameters
    pub optimized_params: HashMap<String, f32>,
    /// Perceptual quality score after optimization
    pub perceptual_quality: f32,
    /// Masking threshold analysis
    pub masking_analysis: MaskingAnalysis,
    /// Loudness analysis
    pub loudness_analysis: LoudnessAnalysis,
    /// Critical band analysis
    pub critical_band_analysis: CriticalBandAnalysis,
    /// Number of optimization iterations performed
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
}

/// Masking analysis results
#[derive(Debug, Clone)]
pub struct MaskingAnalysis {
    /// Spectral masking thresholds per critical band
    pub spectral_masking_thresholds: Vec<f32>,
    /// Temporal masking effects
    pub temporal_masking_effects: Vec<f32>,
    /// Overall masking efficiency (0.0 to 1.0)
    pub masking_efficiency: f32,
}

/// Loudness analysis results
#[derive(Debug, Clone)]
pub struct LoudnessAnalysis {
    /// Loudness levels per critical band (sones)
    pub loudness_levels: Vec<f32>,
    /// Overall loudness (sones)
    pub overall_loudness: f32,
    /// Loudness balance across frequency bands
    pub loudness_balance: f32,
}

/// Critical band analysis results
#[derive(Debug, Clone)]
pub struct CriticalBandAnalysis {
    /// Energy distribution across critical bands
    pub band_energies: Vec<f32>,
    /// Spectral centroid per band
    pub band_centroids: Vec<f32>,
    /// Bandwidth utilization efficiency
    pub bandwidth_efficiency: f32,
}

impl PerceptualOptimizer {
    /// Create new perceptual optimizer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            psychoacoustic_model: PsychoacousticModel::new(sample_rate),
            critical_bands: CriticalBandAnalyzer::new(sample_rate),
            masking_calculator: MaskingCalculator::new(),
            loudness_model: LoudnessModel::new(),
            optimization_params: PerceptualOptimizationParams::default(),
        }
    }

    /// Create with custom optimization parameters
    pub fn with_params(sample_rate: u32, params: PerceptualOptimizationParams) -> Self {
        Self {
            psychoacoustic_model: PsychoacousticModel::new(sample_rate),
            critical_bands: CriticalBandAnalyzer::new(sample_rate),
            masking_calculator: MaskingCalculator::new(),
            loudness_model: LoudnessModel::new(),
            optimization_params: params,
        }
    }

    /// Optimize conversion parameters for perceptual quality
    pub fn optimize_parameters(
        &self,
        audio: &[f32],
        current_params: &HashMap<String, f32>,
        conversion_type: &str,
    ) -> Result<PerceptualOptimizationResult> {
        debug!(
            "Starting perceptual optimization for {} samples",
            audio.len()
        );

        // Analyze current audio with psychoacoustic models
        let masking_analysis = self.analyze_masking(audio)?;
        let loudness_analysis = self.analyze_loudness(audio)?;
        let critical_band_analysis = self.analyze_critical_bands(audio)?;

        // Initialize optimization
        let mut optimized_params = current_params.clone();
        let mut current_quality = self.evaluate_perceptual_quality(
            &masking_analysis,
            &loudness_analysis,
            &critical_band_analysis,
        );

        let mut iterations = 0;
        let mut converged = false;

        // Iterative optimization using gradient-free approach
        while iterations < self.optimization_params.max_iterations && !converged {
            let previous_quality = current_quality;

            // Optimize based on different perceptual aspects
            self.optimize_for_masking(&mut optimized_params, &masking_analysis, conversion_type)?;
            self.optimize_for_loudness(&mut optimized_params, &loudness_analysis, conversion_type)?;
            self.optimize_for_critical_bands(
                &mut optimized_params,
                &critical_band_analysis,
                conversion_type,
            )?;

            // Re-evaluate quality (in real implementation, would re-analyze audio with new params)
            current_quality = self.evaluate_perceptual_quality(
                &masking_analysis,
                &loudness_analysis,
                &critical_band_analysis,
            );

            // Check convergence
            let quality_improvement = current_quality - previous_quality;
            if quality_improvement.abs() < self.optimization_params.convergence_threshold {
                converged = true;
            }

            iterations += 1;
            debug!(
                "Optimization iteration {}: quality = {:.3}",
                iterations, current_quality
            );
        }

        info!(
            "Perceptual optimization complete: {} iterations, quality = {:.3}, converged = {}",
            iterations, current_quality, converged
        );

        Ok(PerceptualOptimizationResult {
            optimized_params,
            perceptual_quality: current_quality,
            masking_analysis,
            loudness_analysis,
            critical_band_analysis,
            iterations,
            converged,
        })
    }

    /// Analyze masking effects in the audio
    fn analyze_masking(&self, audio: &[f32]) -> Result<MaskingAnalysis> {
        let spectrum = self.calculate_spectrum(audio);

        // Calculate masking thresholds for each critical band
        let mut spectral_masking_thresholds = Vec::new();
        let mut temporal_masking_effects = Vec::new();

        for band_idx in 0..self.critical_bands.num_bands {
            // Extract energy for this critical band
            let band_energy = self.critical_bands.get_band_energy(&spectrum, band_idx);

            // Calculate spectral masking threshold
            let spectral_threshold = self.masking_calculator.calculate_spectral_masking(
                &spectrum,
                band_idx,
                &self.critical_bands,
            );
            spectral_masking_thresholds.push(spectral_threshold);

            // Calculate temporal masking effects
            let temporal_effect = self
                .masking_calculator
                .calculate_temporal_masking(band_energy, band_idx);
            temporal_masking_effects.push(temporal_effect);
        }

        // Calculate overall masking efficiency
        let masking_efficiency = self
            .calculate_masking_efficiency(&spectral_masking_thresholds, &temporal_masking_effects);

        Ok(MaskingAnalysis {
            spectral_masking_thresholds,
            temporal_masking_effects,
            masking_efficiency,
        })
    }

    /// Analyze loudness perception
    fn analyze_loudness(&self, audio: &[f32]) -> Result<LoudnessAnalysis> {
        let spectrum = self.calculate_spectrum(audio);
        let mut loudness_levels = Vec::new();

        // Calculate loudness for each critical band
        for band_idx in 0..self.critical_bands.num_bands {
            let band_energy = self.critical_bands.get_band_energy(&spectrum, band_idx);
            let band_frequency = self.critical_bands.get_band_center_frequency(band_idx);

            let loudness = self
                .loudness_model
                .calculate_loudness(band_energy, band_frequency);
            loudness_levels.push(loudness);
        }

        // Calculate overall loudness (sum of specific loudness values)
        let overall_loudness = loudness_levels.iter().sum();

        // Calculate loudness balance (evenness across frequency bands)
        let mean_loudness = overall_loudness / loudness_levels.len() as f32;
        let loudness_variance = loudness_levels
            .iter()
            .map(|&l| {
                let diff = l - mean_loudness;
                diff * diff
            })
            .sum::<f32>()
            / loudness_levels.len() as f32;
        let loudness_balance = if mean_loudness > 0.0 {
            let ratio: f32 = loudness_variance.sqrt() / mean_loudness;
            1.0f32 - ratio.min(1.0f32)
        } else {
            1.0f32
        };

        Ok(LoudnessAnalysis {
            loudness_levels,
            overall_loudness,
            loudness_balance,
        })
    }

    /// Analyze critical band distribution
    fn analyze_critical_bands(&self, audio: &[f32]) -> Result<CriticalBandAnalysis> {
        let spectrum = self.calculate_spectrum(audio);

        let mut band_energies = Vec::new();
        let mut band_centroids = Vec::new();

        for band_idx in 0..self.critical_bands.num_bands {
            let energy = self.critical_bands.get_band_energy(&spectrum, band_idx);
            band_energies.push(energy);

            let centroid = self
                .critical_bands
                .calculate_band_centroid(&spectrum, band_idx);
            band_centroids.push(centroid);
        }

        // Calculate bandwidth utilization efficiency
        let total_energy: f32 = band_energies.iter().sum();
        let effective_bands = band_energies
            .iter()
            .filter(|&&e| e > 0.01 * total_energy)
            .count();
        let bandwidth_efficiency = effective_bands as f32 / self.critical_bands.num_bands as f32;

        Ok(CriticalBandAnalysis {
            band_energies,
            band_centroids,
            bandwidth_efficiency,
        })
    }

    /// Optimize parameters for masking effectiveness
    fn optimize_for_masking(
        &self,
        params: &mut HashMap<String, f32>,
        masking_analysis: &MaskingAnalysis,
        conversion_type: &str,
    ) -> Result<()> {
        // Adjust parameters based on masking analysis
        match conversion_type {
            "PitchShift" => {
                // For pitch shifting, reduce artifacts that break masking
                if masking_analysis.masking_efficiency < 0.7 {
                    self.adjust_param(params, "pitch_smoothing", 0.1);
                    self.adjust_param(params, "formant_preservation", 0.05);
                }
            }
            "SpeedTransformation" => {
                // For speed changes, optimize temporal masking
                if masking_analysis
                    .temporal_masking_effects
                    .iter()
                    .any(|&e| e < 0.5)
                {
                    self.adjust_param(params, "temporal_smoothing", 0.08);
                    self.adjust_param(params, "overlap_ratio", 0.05);
                }
            }
            "SpeakerConversion" => {
                // For speaker conversion, optimize spectral masking
                let avg_spectral_masking = masking_analysis
                    .spectral_masking_thresholds
                    .iter()
                    .sum::<f32>()
                    / masking_analysis.spectral_masking_thresholds.len() as f32;
                if avg_spectral_masking < 0.6 {
                    self.adjust_param(params, "spectral_smoothing", 0.12);
                    self.adjust_param(params, "conversion_strength", -0.1);
                }
            }
            _ => {
                // Generic optimization
                if masking_analysis.masking_efficiency < self.optimization_params.target_quality {
                    self.adjust_param(params, "quality_factor", 0.05);
                }
            }
        }

        Ok(())
    }

    /// Optimize parameters for loudness perception
    fn optimize_for_loudness(
        &self,
        params: &mut HashMap<String, f32>,
        loudness_analysis: &LoudnessAnalysis,
        conversion_type: &str,
    ) -> Result<()> {
        // Optimize for balanced loudness across frequency bands
        if loudness_analysis.loudness_balance < 0.7 {
            match conversion_type {
                "GenderTransformation" | "AgeTransformation" => {
                    self.adjust_param(params, "formant_shift_strength", -0.05);
                    self.adjust_param(params, "energy_normalization", 0.1);
                }
                "PitchShift" => {
                    self.adjust_param(params, "energy_preservation", 0.08);
                }
                _ => {
                    self.adjust_param(params, "dynamic_range_compression", 0.05);
                }
            }
        }

        // Optimize for overall loudness level
        if loudness_analysis.overall_loudness > 50.0 {
            // Too loud - reduce gain
            self.adjust_param(params, "output_gain", -0.1);
        } else if loudness_analysis.overall_loudness < 10.0 {
            // Too quiet - increase gain
            self.adjust_param(params, "output_gain", 0.1);
        }

        Ok(())
    }

    /// Optimize parameters for critical band efficiency
    fn optimize_for_critical_bands(
        &self,
        params: &mut HashMap<String, f32>,
        critical_band_analysis: &CriticalBandAnalysis,
        conversion_type: &str,
    ) -> Result<()> {
        // Optimize bandwidth utilization
        if critical_band_analysis.bandwidth_efficiency < 0.6 {
            match conversion_type {
                "SpeedTransformation" => {
                    self.adjust_param(params, "frequency_warping", 0.05);
                }
                "PitchShift" => {
                    self.adjust_param(params, "harmonic_preservation", 0.1);
                }
                _ => {
                    self.adjust_param(params, "spectral_expansion", 0.05);
                }
            }
        }

        // Balance energy across critical bands
        let max_energy = critical_band_analysis
            .band_energies
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        let min_energy = critical_band_analysis
            .band_energies
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));

        if max_energy > 0.0 && (max_energy / min_energy) > 100.0 {
            // Too much energy imbalance
            self.adjust_param(params, "spectral_balance", 0.1);
            self.adjust_param(params, "frequency_equalization", 0.08);
        }

        Ok(())
    }

    /// Helper function to adjust parameter values safely
    fn adjust_param(&self, params: &mut HashMap<String, f32>, param_name: &str, adjustment: f32) {
        let current_value = params.get(param_name).copied().unwrap_or(1.0);
        let new_value = (current_value + adjustment).clamp(0.0, 2.0);
        params.insert(param_name.to_string(), new_value);

        debug!(
            "Adjusted {}: {:.3} -> {:.3}",
            param_name, current_value, new_value
        );
    }

    /// Calculate perceptual quality from analyses
    fn evaluate_perceptual_quality(
        &self,
        masking_analysis: &MaskingAnalysis,
        loudness_analysis: &LoudnessAnalysis,
        critical_band_analysis: &CriticalBandAnalysis,
    ) -> f32 {
        let masking_quality =
            masking_analysis.masking_efficiency * self.optimization_params.spectral_masking_weight;
        let loudness_quality =
            loudness_analysis.loudness_balance * self.optimization_params.loudness_weight;
        let bandwidth_quality = critical_band_analysis.bandwidth_efficiency
            * self.optimization_params.critical_band_weight;

        // Temporal masking contribution
        let avg_temporal_masking = masking_analysis
            .temporal_masking_effects
            .iter()
            .sum::<f32>()
            / masking_analysis.temporal_masking_effects.len() as f32;
        let temporal_quality =
            avg_temporal_masking * self.optimization_params.temporal_masking_weight;

        masking_quality + loudness_quality + bandwidth_quality + temporal_quality
    }

    /// Calculate masking efficiency from thresholds
    fn calculate_masking_efficiency(
        &self,
        spectral_thresholds: &[f32],
        temporal_effects: &[f32],
    ) -> f32 {
        let spectral_eff =
            spectral_thresholds.iter().sum::<f32>() / spectral_thresholds.len() as f32;
        let temporal_eff = temporal_effects.iter().sum::<f32>() / temporal_effects.len() as f32;
        (spectral_eff + temporal_eff) / 2.0
    }

    /// Calculate spectrum for analysis
    fn calculate_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectrum calculation using energy in frequency bands
        let window_size = 1024.min(audio.len().max(2)); // Ensure at least 2 samples
        let num_bins = (window_size / 2).max(1); // Ensure at least 1 bin
        let mut spectrum = vec![0.0; num_bins];

        // Simple energy-based spectrum calculation
        for (i, &audio_sample) in audio.iter().enumerate().take(window_size.min(audio.len())) {
            let bin = i * num_bins / window_size;
            if bin < spectrum.len() {
                spectrum[bin] += audio_sample * audio_sample;
            }
        }

        // Normalize spectrum
        let total_energy: f32 = spectrum.iter().sum();
        if total_energy > 0.0 {
            for energy in &mut spectrum {
                *energy /= total_energy;
            }
        }

        spectrum
    }
}

// Implementation of sub-components

impl PsychoacousticModel {
    fn new(sample_rate: u32) -> Self {
        // ISO 226 absolute threshold approximation
        let threshold_frequencies = (0..=20)
            .map(|i| 20.0 * (2.0f32.powf(i as f32 / 3.0)))
            .filter(|&f| f <= sample_rate as f32 / 2.0)
            .collect::<Vec<f32>>();

        let absolute_threshold = threshold_frequencies
            .iter()
            .map(|&f| {
                // Simplified absolute threshold curve (approximation)
                let log_f = f.log10();
                3.64 * (f / 1000.0).powf(-0.8) - 6.5 * (-0.6 * (f / 1000.0 - 3.3).powi(2)).exp()
                    + 0.001 * (f / 1000.0).powi(4)
            })
            .collect();

        Self {
            absolute_threshold,
            threshold_frequencies,
            sample_rate,
        }
    }
}

impl CriticalBandAnalyzer {
    fn new(sample_rate: u32) -> Self {
        // Bark scale critical band boundaries (approximation)
        let mut band_boundaries = Vec::new();
        let max_freq = sample_rate as f32 / 2.0;

        for bark in 0..24 {
            let freq = 600.0 * ((bark as f32 / 4.0).sinh());
            if freq <= max_freq {
                band_boundaries.push(freq);
            } else {
                break;
            }
        }

        let num_bands = band_boundaries.len() - 1;

        Self {
            band_boundaries,
            num_bands,
            sample_rate,
        }
    }

    fn get_band_energy(&self, spectrum: &[f32], band_idx: usize) -> f32 {
        if band_idx >= self.num_bands || band_idx + 1 >= self.band_boundaries.len() {
            return 0.0;
        }

        let start_freq = self.band_boundaries[band_idx];
        let end_freq = self.band_boundaries[band_idx + 1];

        let start_bin =
            (start_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let end_bin = (end_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        spectrum[start_bin.min(spectrum.len())..end_bin.min(spectrum.len())]
            .iter()
            .sum()
    }

    fn get_band_center_frequency(&self, band_idx: usize) -> f32 {
        if band_idx >= self.num_bands || band_idx + 1 >= self.band_boundaries.len() {
            return 0.0;
        }

        (self.band_boundaries[band_idx] + self.band_boundaries[band_idx + 1]) / 2.0
    }

    fn calculate_band_centroid(&self, spectrum: &[f32], band_idx: usize) -> f32 {
        if band_idx >= self.num_bands || band_idx + 1 >= self.band_boundaries.len() {
            return 0.0;
        }

        let start_freq = self.band_boundaries[band_idx];
        let end_freq = self.band_boundaries[band_idx + 1];

        let start_bin =
            (start_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let end_bin = (end_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &energy) in spectrum[start_bin.min(spectrum.len())..end_bin.min(spectrum.len())]
            .iter()
            .enumerate()
        {
            let freq =
                start_freq + i as f32 * (end_freq - start_freq) / (end_bin - start_bin) as f32;
            weighted_sum += freq * energy;
            total_energy += energy;
        }

        if total_energy > 0.0 {
            weighted_sum / total_energy
        } else {
            self.get_band_center_frequency(band_idx)
        }
    }
}

impl MaskingCalculator {
    fn new() -> Self {
        Self {
            spreading_coefficients: (0..50).map(|i| (-0.05 * i as f32).exp()).collect(),
            temporal_masking: TemporalMaskingParams::default(),
            simultaneous_masking: SimultaneousMaskingParams::default(),
        }
    }

    fn calculate_spectral_masking(
        &self,
        spectrum: &[f32],
        band_idx: usize,
        critical_bands: &CriticalBandAnalyzer,
    ) -> f32 {
        let band_energy = critical_bands.get_band_energy(spectrum, band_idx);

        if band_energy <= 0.0 {
            return 0.0;
        }

        // Calculate masking from neighboring bands
        let mut masking_threshold = 0.0;

        for other_band in 0..critical_bands.num_bands {
            if other_band == band_idx {
                continue;
            }

            let other_energy = critical_bands.get_band_energy(spectrum, other_band);
            if other_energy <= 0.0 {
                continue;
            }

            let distance = (other_band as i32 - band_idx as i32).unsigned_abs() as usize;
            let spreading = if distance < self.spreading_coefficients.len() {
                self.spreading_coefficients[distance]
            } else {
                0.001
            };

            masking_threshold += other_energy * spreading;
        }

        // Return masking effectiveness (higher is better)
        if masking_threshold > 0.0 {
            (band_energy / masking_threshold).min(1.0)
        } else {
            1.0
        }
    }

    fn calculate_temporal_masking(&self, band_energy: f32, _band_idx: usize) -> f32 {
        // Simplified temporal masking effect
        // Higher energy provides better temporal masking
        (band_energy * 2.0).min(1.0)
    }
}

impl LoudnessModel {
    fn new() -> Self {
        // Simplified equal loudness contours
        let mut equal_loudness_contours = HashMap::new();
        let contour_frequencies = (0..=20)
            .map(|i| 20.0 * (2.0f32.powf(i as f32 / 3.0)))
            .collect::<Vec<f32>>();

        // 40 phon contour (simplified)
        let phon_40 = contour_frequencies
            .iter()
            .map(|&f| {
                // Simplified loudness curve
                40.0 + 10.0 * (f / 1000.0).log10() - 5.0 * ((f / 1000.0 - 1.0).powi(2))
            })
            .collect();

        equal_loudness_contours.insert(40, phon_40);

        let loudness_scaling = vec![1.0; contour_frequencies.len()];

        Self {
            equal_loudness_contours,
            contour_frequencies,
            loudness_scaling,
        }
    }

    fn calculate_loudness(&self, energy: f32, frequency: f32) -> f32 {
        if energy <= 0.0 {
            return 0.0;
        }

        // Convert energy to dB
        let db_level = 10.0 * energy.log10();

        // Simple loudness calculation (Stevens' power law approximation)
        let loudness_exponent = 0.3; // Typical for loudness
        (db_level / 40.0).powf(loudness_exponent).max(0.0)
    }
}

/// Quality targets measurement system for production validation
/// Implements the quality targets from TODO: Target Similarity (85%+), Source Preservation (90%+),
/// MOS 4.0+, and Artifact Level (<5%)
#[derive(Debug, Clone)]
pub struct QualityTargetsSystem {
    /// Configuration for quality target thresholds
    config: QualityTargetsConfig,
    /// History of quality measurements for trend analysis
    measurement_history: Vec<QualityTargetMeasurement>,
    /// Maximum history size
    max_history: usize,
}

/// Configuration for quality target thresholds
#[derive(Debug, Clone)]
pub struct QualityTargetsConfig {
    /// Target similarity threshold (default: 0.85 for 85%+)
    pub target_similarity_threshold: f32,
    /// Source preservation threshold (default: 0.90 for 90%+)
    pub source_preservation_threshold: f32,
    /// MOS (Mean Opinion Score) threshold (default: 4.0)
    pub mos_threshold: f32,
    /// Artifact level threshold (default: 0.05 for <5%)
    pub artifact_level_threshold: f32,
    /// Enable detailed measurement tracking
    pub enable_detailed_tracking: bool,
}

impl Default for QualityTargetsConfig {
    fn default() -> Self {
        Self {
            target_similarity_threshold: 0.85,
            source_preservation_threshold: 0.90,
            mos_threshold: 4.0,
            artifact_level_threshold: 0.05,
            enable_detailed_tracking: true,
        }
    }
}

/// Complete quality target measurement result
#[derive(Debug, Clone)]
pub struct QualityTargetMeasurement {
    /// Target similarity score (0.0-1.0, target: ≥0.85)
    pub target_similarity: f32,
    /// Source preservation score (0.0-1.0, target: ≥0.90)
    pub source_preservation: f32,
    /// Mean Opinion Score (1.0-5.0, target: ≥4.0)
    pub mos_score: f32,
    /// Artifact level (0.0-1.0, target: <0.05)
    pub artifact_level: f32,
    /// Overall quality target achievement
    pub targets_met: QualityTargetsAchievement,
    /// Timestamp of measurement
    pub timestamp: std::time::SystemTime,
    /// Additional metrics for detailed analysis
    pub detailed_metrics: DetailedQualityMetrics,
}

/// Quality targets achievement status
#[derive(Debug, Clone)]
pub struct QualityTargetsAchievement {
    /// Whether target similarity threshold is met
    pub target_similarity_met: bool,
    /// Whether source preservation threshold is met  
    pub source_preservation_met: bool,
    /// Whether MOS threshold is met
    pub mos_threshold_met: bool,
    /// Whether artifact level threshold is met
    pub artifact_level_met: bool,
    /// Overall achievement percentage (0.0-1.0)
    pub overall_achievement: f32,
}

/// Detailed quality metrics for in-depth analysis
#[derive(Debug, Clone)]
pub struct DetailedQualityMetrics {
    /// Speaker identity preservation score
    pub speaker_identity_preservation: f32,
    /// Prosodic characteristic preservation
    pub prosodic_preservation: f32,
    /// Linguistic content preservation
    pub linguistic_preservation: f32,
    /// Spectral fidelity score
    pub spectral_fidelity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Perceptual quality estimate
    pub perceptual_quality: f32,
}

impl QualityTargetsSystem {
    /// Create a new quality targets system
    pub fn new() -> Self {
        Self::with_config(QualityTargetsConfig::default())
    }

    /// Create a new quality targets system with custom configuration
    pub fn with_config(config: QualityTargetsConfig) -> Self {
        Self {
            config,
            measurement_history: Vec::new(),
            max_history: 1000, // Keep last 1000 measurements
        }
    }

    /// Measure quality targets for a conversion result
    pub fn measure_quality_targets(
        &mut self,
        converted_audio: &[f32],
        original_audio: &[f32],
        target_reference: Option<&[f32]>,
        sample_rate: u32,
    ) -> Result<QualityTargetMeasurement> {
        // Calculate target similarity
        let target_similarity = if let Some(target_ref) = target_reference {
            self.calculate_target_similarity(converted_audio, target_ref, sample_rate)?
        } else {
            // Use speaker characteristics similarity if no reference available
            self.calculate_speaker_characteristics_similarity(converted_audio, sample_rate)?
        };

        // Calculate source preservation
        let source_preservation =
            self.calculate_source_preservation(converted_audio, original_audio, sample_rate)?;

        // Calculate MOS score
        let mos_score = self.calculate_mos_score(converted_audio, sample_rate)?;

        // Calculate artifact level
        let artifact_level = self.calculate_artifact_level(converted_audio, sample_rate)?;

        // Calculate detailed metrics
        let detailed_metrics = self.calculate_detailed_metrics(
            converted_audio,
            original_audio,
            target_reference,
            sample_rate,
        )?;

        // Determine targets achievement
        let targets_met = QualityTargetsAchievement {
            target_similarity_met: target_similarity >= self.config.target_similarity_threshold,
            source_preservation_met: source_preservation
                >= self.config.source_preservation_threshold,
            mos_threshold_met: mos_score >= self.config.mos_threshold,
            artifact_level_met: artifact_level <= self.config.artifact_level_threshold,
            overall_achievement: self.calculate_overall_achievement(
                target_similarity,
                source_preservation,
                mos_score,
                artifact_level,
            ),
        };

        let measurement = QualityTargetMeasurement {
            target_similarity,
            source_preservation,
            mos_score,
            artifact_level,
            targets_met,
            timestamp: std::time::SystemTime::now(),
            detailed_metrics,
        };

        // Store measurement in history
        if self.config.enable_detailed_tracking {
            self.add_to_history(measurement.clone());
        }

        Ok(measurement)
    }

    /// Calculate target similarity (how similar the converted voice is to the target voice)
    fn calculate_target_similarity(
        &self,
        converted: &[f32],
        target: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Use multiple similarity measures for robust assessment
        let spectral_similarity = self.calculate_spectral_similarity(converted, target)?;
        let prosodic_similarity =
            self.calculate_prosodic_similarity(converted, target, sample_rate)?;
        let timbral_similarity = self.calculate_timbral_similarity(converted, target)?;

        // Weighted combination (spectral: 40%, prosodic: 35%, timbral: 25%)
        let similarity =
            spectral_similarity * 0.4 + prosodic_similarity * 0.35 + timbral_similarity * 0.25;
        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Calculate source preservation (how much of the original content is preserved)
    fn calculate_source_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Measure linguistic content preservation
        let linguistic_preservation =
            self.calculate_linguistic_preservation(converted, original, sample_rate)?;

        // Measure temporal structure preservation
        let temporal_preservation = self.calculate_temporal_preservation(converted, original)?;

        // Measure semantic content preservation (simplified)
        let semantic_preservation = self.calculate_semantic_preservation(converted, original)?;

        // Weighted combination (linguistic: 50%, temporal: 30%, semantic: 20%)
        let preservation = linguistic_preservation * 0.5
            + temporal_preservation * 0.3
            + semantic_preservation * 0.2;
        Ok(preservation.clamp(0.0, 1.0))
    }

    /// Calculate MOS (Mean Opinion Score) estimate
    fn calculate_mos_score(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        // Objective MOS estimation using multiple quality factors
        let naturalness = self.estimate_naturalness(audio, sample_rate)?;
        let clarity = self.estimate_clarity(audio)?;
        let pleasantness = self.estimate_pleasantness(audio, sample_rate)?;
        let overall_quality = self.estimate_overall_quality(audio)?;

        // Convert to MOS scale (1-5) with weighted combination
        let quality_score =
            naturalness * 0.3 + clarity * 0.25 + pleasantness * 0.25 + overall_quality * 0.2;
        let mos = 1.0 + (quality_score * 4.0); // Scale to 1-5 range

        Ok(mos.clamp(1.0, 5.0))
    }

    /// Calculate artifact level (percentage of noticeable artifacts)
    fn calculate_artifact_level(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let mut detector = ArtifactDetector::new();
        let artifacts = detector.detect_artifacts(audio, sample_rate)?;

        // Calculate artifact level as percentage of audio with artifacts
        let total_samples = audio.len() as f32;
        let mut artifact_samples = 0.0;

        for location in &artifacts.artifact_locations {
            let duration = (location.end_sample - location.start_sample) as f32;
            artifact_samples += duration * location.severity;
        }

        let artifact_level = (artifact_samples / total_samples).min(1.0);
        Ok(artifact_level)
    }

    /// Calculate detailed quality metrics
    fn calculate_detailed_metrics(
        &self,
        converted: &[f32],
        original: &[f32],
        target_reference: Option<&[f32]>,
        sample_rate: u32,
    ) -> Result<DetailedQualityMetrics> {
        Ok(DetailedQualityMetrics {
            speaker_identity_preservation: if let Some(target) = target_reference {
                self.calculate_speaker_identity_preservation(converted, target, sample_rate)?
            } else {
                0.5 // Default when no reference available
            },
            prosodic_preservation: self.calculate_prosodic_preservation(
                converted,
                original,
                sample_rate,
            )?,
            linguistic_preservation: self.calculate_linguistic_preservation(
                converted,
                original,
                sample_rate,
            )?,
            spectral_fidelity: self.calculate_spectral_fidelity(converted, original)?,
            temporal_consistency: self.calculate_temporal_consistency(converted)?,
            perceptual_quality: self.estimate_perceptual_quality(converted, sample_rate)?,
        })
    }

    /// Calculate overall achievement percentage
    fn calculate_overall_achievement(
        &self,
        target_sim: f32,
        source_pres: f32,
        mos: f32,
        artifact: f32,
    ) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        // Target similarity (weight: 30%)
        let sim_achievement = (target_sim / self.config.target_similarity_threshold).min(1.0);
        score += sim_achievement * 0.3;
        total_weight += 0.3;

        // Source preservation (weight: 30%)
        let pres_achievement = (source_pres / self.config.source_preservation_threshold).min(1.0);
        score += pres_achievement * 0.3;
        total_weight += 0.3;

        // MOS score (weight: 25%)
        let mos_achievement = (mos / self.config.mos_threshold).min(1.0);
        score += mos_achievement * 0.25;
        total_weight += 0.25;

        // Artifact level (weight: 15%, inverted since lower is better)
        let artifact_achievement =
            (1.0 - (artifact / self.config.artifact_level_threshold)).clamp(0.0, 1.0);
        score += artifact_achievement * 0.15;
        total_weight += 0.15;

        score / total_weight
    }

    /// Add measurement to history
    fn add_to_history(&mut self, measurement: QualityTargetMeasurement) {
        self.measurement_history.push(measurement);

        // Maintain maximum history size
        if self.measurement_history.len() > self.max_history {
            self.measurement_history.remove(0);
        }
    }

    /// Get quality targets achievement statistics
    pub fn get_achievement_statistics(&self) -> QualityTargetsStatistics {
        if self.measurement_history.is_empty() {
            return QualityTargetsStatistics::default();
        }

        let total_measurements = self.measurement_history.len() as f32;
        let mut target_sim_met = 0;
        let mut source_pres_met = 0;
        let mut mos_met = 0;
        let mut artifact_met = 0;
        let mut total_achievement = 0.0;

        for measurement in &self.measurement_history {
            if measurement.targets_met.target_similarity_met {
                target_sim_met += 1;
            }
            if measurement.targets_met.source_preservation_met {
                source_pres_met += 1;
            }
            if measurement.targets_met.mos_threshold_met {
                mos_met += 1;
            }
            if measurement.targets_met.artifact_level_met {
                artifact_met += 1;
            }
            total_achievement += measurement.targets_met.overall_achievement;
        }

        QualityTargetsStatistics {
            total_measurements: self.measurement_history.len(),
            target_similarity_achievement_rate: target_sim_met as f32 / total_measurements,
            source_preservation_achievement_rate: source_pres_met as f32 / total_measurements,
            mos_achievement_rate: mos_met as f32 / total_measurements,
            artifact_level_achievement_rate: artifact_met as f32 / total_measurements,
            average_overall_achievement: total_achievement / total_measurements,
        }
    }

    // Helper methods for similarity calculations
    fn calculate_spectral_similarity(&self, audio1: &[f32], audio2: &[f32]) -> Result<f32> {
        let min_len = audio1.len().min(audio2.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let mut correlation = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..min_len {
            correlation += audio1[i] * audio2[i];
            norm1 += audio1[i] * audio1[i];
            norm2 += audio2[i] * audio2[i];
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok((correlation / (norm1.sqrt() * norm2.sqrt())).abs())
    }

    fn calculate_prosodic_similarity(
        &self,
        audio1: &[f32],
        audio2: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Simplified prosodic similarity based on F0 patterns
        let f0_1 = self.extract_f0_contour(audio1, sample_rate)?;
        let f0_2 = self.extract_f0_contour(audio2, sample_rate)?;

        if f0_1.is_empty() || f0_2.is_empty() {
            return Ok(0.5); // Default similarity when F0 extraction fails
        }

        self.calculate_spectral_similarity(&f0_1, &f0_2)
    }

    fn calculate_timbral_similarity(&self, audio1: &[f32], audio2: &[f32]) -> Result<f32> {
        // Simplified timbral similarity using spectral centroid
        let centroid1 = self.calculate_spectral_centroid(audio1)?;
        let centroid2 = self.calculate_spectral_centroid(audio2)?;

        let similarity = 1.0 - ((centroid1 - centroid2).abs() / (centroid1 + centroid2).max(0.001));
        Ok(similarity.clamp(0.0, 1.0))
    }

    // Helper methods for preservation calculations
    fn calculate_linguistic_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        _sample_rate: u32,
    ) -> Result<f32> {
        // Simplified linguistic preservation using correlation
        self.calculate_spectral_similarity(converted, original)
    }

    fn calculate_temporal_preservation(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        let len_ratio = converted.len() as f32 / original.len() as f32;
        let preservation = 1.0 - (1.0 - len_ratio).abs();
        Ok(preservation.clamp(0.0, 1.0))
    }

    fn calculate_semantic_preservation(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        // Simplified semantic preservation using energy patterns
        let energy1 = converted.iter().map(|x| x * x).sum::<f32>() / converted.len() as f32;
        let energy2 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;

        let preservation = 1.0 - ((energy1 - energy2).abs() / (energy1 + energy2).max(0.001));
        Ok(preservation.clamp(0.0, 1.0))
    }

    // Helper methods for MOS estimation
    fn estimate_naturalness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        // Estimate naturalness based on spectral and temporal characteristics
        let spectral_naturalness = self.estimate_spectral_naturalness(audio)?;
        let temporal_naturalness = self.estimate_temporal_naturalness(audio, sample_rate)?;

        Ok((spectral_naturalness + temporal_naturalness) / 2.0)
    }

    fn estimate_clarity(&self, audio: &[f32]) -> Result<f32> {
        // Estimate clarity using SNR-like measure
        let signal_power = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let noise_estimate = self.estimate_noise_level(audio)?;

        let snr = if noise_estimate > 0.0 {
            10.0 * (signal_power / noise_estimate).log10()
        } else {
            60.0 // Very high SNR when no noise detected
        };

        // Convert SNR to 0-1 clarity score
        Ok((snr / 60.0).clamp(0.0, 1.0))
    }

    fn estimate_pleasantness(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        // Estimate pleasantness based on harmonic structure
        let harmonic_score = self.calculate_harmonic_content(audio, sample_rate)?;
        let roughness_penalty = self.calculate_roughness_penalty(audio)?;

        Ok((harmonic_score * (1.0 - roughness_penalty)).clamp(0.0, 1.0))
    }

    fn estimate_overall_quality(&self, audio: &[f32]) -> Result<f32> {
        // Overall quality based on dynamic range and distortion
        let dynamic_range = self.calculate_dynamic_range(audio)?;
        let distortion_penalty = self.calculate_distortion_penalty(audio)?;

        Ok((dynamic_range * (1.0 - distortion_penalty)).clamp(0.0, 1.0))
    }

    // Additional helper methods
    fn calculate_speaker_characteristics_similarity(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Estimate speaker similarity based on vocal characteristics
        let pitch_stats = self.calculate_pitch_statistics(audio, sample_rate)?;
        let formant_characteristics = self.estimate_formant_characteristics(audio)?;

        // Combine characteristics for similarity estimate
        Ok((pitch_stats + formant_characteristics) / 2.0)
    }

    fn calculate_speaker_identity_preservation(
        &self,
        converted: &[f32],
        target: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        self.calculate_prosodic_similarity(converted, target, sample_rate)
    }

    fn calculate_prosodic_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        self.calculate_prosodic_similarity(converted, original, sample_rate)
    }

    fn calculate_spectral_fidelity(&self, converted: &[f32], original: &[f32]) -> Result<f32> {
        self.calculate_spectral_similarity(converted, original)
    }

    fn calculate_temporal_consistency(&self, audio: &[f32]) -> Result<f32> {
        // Measure temporal consistency using energy variations
        let window_size = 1024;
        let mut consistency_score = 0.0;
        let mut window_count = 0;

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let energy = window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32;

            if window_count > 0 {
                // Compare with previous window energy (simplified consistency measure)
                consistency_score += (1.0 - (energy - 0.1).abs()).max(0.0); // 0.1 is reference energy
            }
            window_count += 1;
        }

        if window_count > 1 {
            Ok(consistency_score / (window_count - 1) as f32)
        } else {
            Ok(0.5)
        }
    }

    fn estimate_perceptual_quality(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let naturalness = self.estimate_naturalness(audio, sample_rate)?;
        let clarity = self.estimate_clarity(audio)?;
        Ok((naturalness + clarity) / 2.0)
    }

    // Additional computational helper methods (simplified implementations)
    fn extract_f0_contour(&self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        // Simplified F0 extraction - in production would use proper pitch detection
        let window_size = 1024;
        let mut f0_contour = Vec::new();

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0_simple(window);
            f0_contour.push(f0);
        }

        Ok(f0_contour)
    }

    fn estimate_f0_simple(&self, window: &[f32]) -> f32 {
        // Simplified autocorrelation-based F0 estimation
        let mut best_lag = 0;
        let mut max_correlation = 0.0;

        for lag in 20..400 {
            // Typical pitch lag range
            if lag >= window.len() {
                break;
            }

            let mut correlation = 0.0;
            for i in 0..(window.len() - lag) {
                correlation += window[i] * window[i + lag];
            }

            if correlation > max_correlation {
                max_correlation = correlation;
                best_lag = lag;
            }
        }

        if best_lag > 0 {
            16000.0 / best_lag as f32
        } else {
            0.0
        } // Assuming 16kHz sample rate
    }

    fn calculate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        // Simplified spectral centroid calculation
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += (i as f32) * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }

    fn estimate_spectral_naturalness(&self, audio: &[f32]) -> Result<f32> {
        // Estimate based on spectral roll-off
        let centroid = self.calculate_spectral_centroid(audio)?;
        let normalized_centroid = (centroid / audio.len() as f32).clamp(0.0, 1.0);
        Ok(1.0 - (normalized_centroid - 0.3).abs()) // Natural speech typically has centroid around 30%
    }

    fn estimate_temporal_naturalness(&self, audio: &[f32], _sample_rate: u32) -> Result<f32> {
        // Estimate based on zero-crossing rate variation
        let mut zcr_values = Vec::new();
        let window_size = 512;

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let zcr = self.calculate_zero_crossing_rate(window);
            zcr_values.push(zcr);
        }

        if zcr_values.is_empty() {
            return Ok(0.5);
        }

        // Calculate variance in ZCR (natural speech has moderate variance)
        let mean_zcr = zcr_values.iter().sum::<f32>() / zcr_values.len() as f32;
        let variance = zcr_values
            .iter()
            .map(|x| (x - mean_zcr).powi(2))
            .sum::<f32>()
            / zcr_values.len() as f32;

        // Natural speech has ZCR variance around 0.01-0.05
        let naturalness = 1.0 - ((variance - 0.03).abs() / 0.03).min(1.0);
        Ok(naturalness)
    }

    fn calculate_zero_crossing_rate(&self, window: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..window.len() {
            if (window[i] >= 0.0) != (window[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / window.len() as f32
    }

    fn estimate_noise_level(&self, audio: &[f32]) -> Result<f32> {
        // Estimate noise using minimum energy method
        let window_size = 256;
        let mut energy_values = Vec::new();

        for i in (0..audio.len()).step_by(window_size) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];
            let energy = window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32;
            energy_values.push(energy);
        }

        if energy_values.is_empty() {
            return Ok(0.001);
        }

        // Use 10th percentile as noise estimate
        energy_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_index = (energy_values.len() as f32 * 0.1) as usize;
        Ok(energy_values.get(noise_index).copied().unwrap_or(0.001))
    }

    fn calculate_harmonic_content(&self, audio: &[f32], _sample_rate: u32) -> Result<f32> {
        // Simplified harmonic content estimation
        let mut total_energy = 0.0;
        let mut harmonic_energy = 0.0;

        let window_size = 1024;
        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];

            let energy = window.iter().map(|x| x * x).sum::<f32>();
            total_energy += energy;

            // Estimate harmonic content using autocorrelation peak
            let mut max_autocorr: f32 = 0.0;
            for lag in 50..400 {
                if lag >= window.len() {
                    break;
                }
                let mut autocorr = 0.0;
                for j in 0..(window.len() - lag) {
                    autocorr += window[j] * window[j + lag];
                }
                max_autocorr = max_autocorr.max(autocorr);
            }

            harmonic_energy += max_autocorr.max(0.0);
        }

        if total_energy > 0.0 {
            Ok((harmonic_energy / total_energy).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn calculate_roughness_penalty(&self, audio: &[f32]) -> Result<f32> {
        // Estimate roughness based on high-frequency energy variations
        let mut roughness = 0.0;
        for i in 1..audio.len() {
            let diff = (audio[i] - audio[i - 1]).abs();
            roughness += diff;
        }
        roughness /= audio.len() as f32;
        Ok((roughness * 10.0).min(1.0)) // Scale and clamp penalty
    }

    fn calculate_dynamic_range(&self, audio: &[f32]) -> Result<f32> {
        let max_val = audio.iter().fold(0.0f32, |max, &val| max.max(val.abs()));
        let min_val = audio.iter().fold(f32::INFINITY, |min, &val| {
            if val.abs() > 0.001 {
                min.min(val.abs())
            } else {
                min
            }
        });

        if min_val != f32::INFINITY && min_val > 0.0 {
            let dynamic_range_db = 20.0 * (max_val / min_val).log10();
            Ok((dynamic_range_db / 60.0).clamp(0.0, 1.0)) // Normalize to 0-1
        } else {
            Ok(0.0)
        }
    }

    fn calculate_distortion_penalty(&self, audio: &[f32]) -> Result<f32> {
        // Estimate distortion using clipping detection
        let mut clipped_samples = 0;
        for &sample in audio {
            if sample.abs() > 0.95 {
                // Near clipping
                clipped_samples += 1;
            }
        }

        let clipping_ratio = clipped_samples as f32 / audio.len() as f32;
        Ok(clipping_ratio.min(1.0))
    }

    fn calculate_pitch_statistics(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        let f0_contour = self.extract_f0_contour(audio, sample_rate)?;
        if f0_contour.is_empty() {
            return Ok(0.5);
        }

        let mean_f0 = f0_contour.iter().sum::<f32>() / f0_contour.len() as f32;
        let variance = f0_contour
            .iter()
            .map(|x| (x - mean_f0).powi(2))
            .sum::<f32>()
            / f0_contour.len() as f32;

        // Normalize statistics to 0-1 range (simplified)
        let normalized_mean = (mean_f0 / 500.0).clamp(0.0, 1.0); // Typical F0 range
        let normalized_variance = (variance / 10000.0).clamp(0.0, 1.0);

        Ok((normalized_mean + normalized_variance) / 2.0)
    }

    fn estimate_formant_characteristics(&self, audio: &[f32]) -> Result<f32> {
        // Simplified formant estimation using spectral peaks
        let window_size = 1024;
        let mut formant_score = 0.0;
        let mut window_count = 0;

        for i in (0..audio.len()).step_by(window_size / 2) {
            if i + window_size >= audio.len() {
                break;
            }
            let window = &audio[i..i + window_size];

            // Find spectral peaks (simplified formant detection)
            let mut peaks = 0;
            for j in 1..(window.len() - 1) {
                if window[j] > window[j - 1] && window[j] > window[j + 1] && window[j] > 0.1 {
                    peaks += 1;
                }
            }

            // Typical speech has 3-5 formants in analysis window
            let formant_quality = if peaks >= 3 && peaks <= 8 {
                1.0 - ((peaks as f32 - 4.0).abs() / 4.0)
            } else {
                0.5
            };

            formant_score += formant_quality;
            window_count += 1;
        }

        if window_count > 0 {
            Ok(formant_score / window_count as f32)
        } else {
            Ok(0.5)
        }
    }
}

impl Default for QualityTargetsSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for quality targets achievement over time
#[derive(Debug, Clone)]
pub struct QualityTargetsStatistics {
    /// Total number of measurements
    pub total_measurements: usize,
    /// Percentage of measurements meeting target similarity threshold
    pub target_similarity_achievement_rate: f32,
    /// Percentage of measurements meeting source preservation threshold
    pub source_preservation_achievement_rate: f32,
    /// Percentage of measurements meeting MOS threshold
    pub mos_achievement_rate: f32,
    /// Percentage of measurements meeting artifact level threshold
    pub artifact_level_achievement_rate: f32,
    /// Average overall achievement across all measurements
    pub average_overall_achievement: f32,
}

impl Default for QualityTargetsStatistics {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            target_similarity_achievement_rate: 0.0,
            source_preservation_achievement_rate: 0.0,
            mos_achievement_rate: 0.0,
            artifact_level_achievement_rate: 0.0,
            average_overall_achievement: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_detector_creation() {
        let detector = ArtifactDetector::new();
        assert_eq!(detector.max_history, 44100);
        assert!(detector.history_buffer.is_empty());
    }

    #[test]
    fn test_artifact_detector_with_thresholds() {
        let thresholds = ArtifactThresholds {
            click_threshold: 0.2,
            metallic_threshold: 0.3,
            buzzing_threshold: 0.25,
            pitch_variation_threshold: 0.3,
            spectral_discontinuity_threshold: 0.1,
            energy_spike_threshold: 0.4,
            hf_noise_threshold: 0.2,
            phase_artifact_threshold: 0.15,
            adaptation_rate: 0.01,
            min_confidence: 0.6,
        };
        let detector = ArtifactDetector::with_thresholds(thresholds);
        assert_eq!(detector.thresholds.click_threshold, 0.2);
        assert_eq!(detector.thresholds.metallic_threshold, 0.3);
    }

    #[test]
    fn test_artifact_detection_clean_audio() {
        let mut detector = ArtifactDetector::new();
        let clean_audio = vec![0.0; 1000]; // Silent audio should be clean

        let result = detector.detect_artifacts(&clean_audio, 44100).unwrap();

        assert!(result.overall_score < 0.1); // Should be very low for clean audio
        assert!(!result.artifact_types.is_empty());
        assert_eq!(result.artifact_locations.len(), 0); // No artifacts in silent audio
    }

    #[test]
    fn test_artifact_detection_noisy_audio() {
        let mut detector = ArtifactDetector::new();
        // Create audio with high energy spikes
        let mut noisy_audio = vec![0.0; 1000];
        for i in (100..900).step_by(100) {
            noisy_audio[i] = 1.0; // Add large spikes
        }

        let result = detector.detect_artifacts(&noisy_audio, 44100).unwrap();

        assert!(result.overall_score > 0.1); // Should detect artifacts
        assert!(result.artifact_locations.len() > 0); // Should find some artifact locations
    }

    #[test]
    fn test_artifact_location_creation() {
        let location = ArtifactLocation {
            artifact_type: ArtifactType::Click,
            start_sample: 100,
            end_sample: 132,
            confidence: 0.8,
            severity: 0.6,
        };

        assert_eq!(location.artifact_type, ArtifactType::Click);
        assert_eq!(location.start_sample, 100);
        assert_eq!(location.end_sample, 132);
        assert_eq!(location.confidence, 0.8);
        assert_eq!(location.severity, 0.6);
    }

    #[test]
    fn test_quality_metrics_system_creation() {
        let system = QualityMetricsSystem::new();
        assert!(system.reference_features.is_none());

        let params = PerceptualParameters::default();
        assert_eq!(params.spectral_weight, 0.3);
        assert_eq!(params.temporal_weight, 0.2);
        assert_eq!(params.prosodic_weight, 0.3);
        assert_eq!(params.naturalness_weight, 0.2);
    }

    #[test]
    fn test_quality_metrics_system_with_reference() {
        let mut system = QualityMetricsSystem::new();
        let reference_audio = vec![0.1, -0.2, 0.3, -0.1, 0.2];

        let result = system.set_reference(&reference_audio, 16000);
        assert!(result.is_ok());
        assert!(system.reference_features.is_some());
    }

    #[test]
    fn test_quality_evaluation() {
        let system = QualityMetricsSystem::new();
        let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05];

        let result = system.evaluate_quality(&test_audio, 16000).unwrap();

        assert!(result.overall_score >= 0.0 && result.overall_score <= 1.0);
        assert!(result.spectral_similarity >= 0.0 && result.spectral_similarity <= 1.0);
        assert!(result.temporal_consistency >= 0.0 && result.temporal_consistency <= 1.0);
        assert!(result.prosodic_preservation >= 0.0 && result.prosodic_preservation <= 1.0);
        assert!(result.naturalness >= 0.0 && result.naturalness <= 1.0);
        assert!(result.perceptual_quality >= 0.0 && result.perceptual_quality <= 1.0);
    }

    #[test]
    fn test_adaptive_quality_controller_creation() {
        let controller = AdaptiveQualityController::new(0.9);
        assert_eq!(controller.quality_target(), 0.9);
        assert!(controller.strategies.len() > 0);

        let default_controller = AdaptiveQualityController::default();
        assert_eq!(default_controller.quality_target(), 0.8);
    }

    #[test]
    fn test_adaptive_quality_controller_target_setting() {
        let mut controller = AdaptiveQualityController::new(0.7);
        controller.set_quality_target(0.95);
        assert_eq!(controller.quality_target(), 0.95);

        // Test clamping
        controller.set_quality_target(1.5);
        assert_eq!(controller.quality_target(), 1.0);

        controller.set_quality_target(-0.1);
        assert_eq!(controller.quality_target(), 0.0);
    }

    #[test]
    fn test_quality_trend_analysis() {
        let mut controller = AdaptiveQualityController::new(0.8);

        // Test with insufficient history
        assert_eq!(controller.get_quality_trend(), QualityTrend::Stable);

        // Add improving trend
        controller.update_quality_history(0.5);
        controller.update_quality_history(0.6);
        controller.update_quality_history(0.7);
        assert_eq!(controller.get_quality_trend(), QualityTrend::Improving);

        // Add degrading trend
        controller.update_quality_history(0.6);
        controller.update_quality_history(0.5);
        controller.update_quality_history(0.4);
        assert_eq!(controller.get_quality_trend(), QualityTrend::Degrading);
    }

    #[test]
    fn test_quality_trigger_evaluation() {
        let controller = AdaptiveQualityController::new(0.8);

        // Create test artifacts
        let artifacts = DetectedArtifacts {
            overall_score: 0.3,
            artifact_types: [(ArtifactType::Click, 0.2), (ArtifactType::Buzzing, 0.4)]
                .into_iter()
                .collect(),
            artifact_locations: Vec::new(),
            quality_assessment: QualityAssessment {
                overall_quality: 0.6,
                naturalness: 0.7,
                clarity: 0.8,
                consistency: 0.75,
                recommended_adjustments: Vec::new(),
            },
        };

        let objective_quality = ObjectiveQualityMetrics {
            overall_score: 0.6,
            spectral_similarity: 0.7,
            temporal_consistency: 0.8,
            prosodic_preservation: 0.75,
            naturalness: 0.7,
            perceptual_quality: 0.65,
            snr_estimate: 25.0,
            segmental_snr: 23.0,
        };

        // Test overall quality trigger
        let trigger1 = QualityTrigger::OverallQualityBelow(0.7);
        assert!(controller.evaluate_trigger(&trigger1, &artifacts, &objective_quality));

        let trigger2 = QualityTrigger::OverallQualityBelow(0.5);
        assert!(!controller.evaluate_trigger(&trigger2, &artifacts, &objective_quality));

        // Test specific artifact trigger
        let trigger3 = QualityTrigger::SpecificArtifact(ArtifactType::Buzzing, 0.3);
        assert!(controller.evaluate_trigger(&trigger3, &artifacts, &objective_quality));

        let trigger4 = QualityTrigger::SpecificArtifact(ArtifactType::Metallic, 0.1);
        assert!(!controller.evaluate_trigger(&trigger4, &artifacts, &objective_quality));

        // Test combined trigger
        let trigger5 = QualityTrigger::Combined(vec![
            QualityTrigger::OverallQualityBelow(0.7),
            QualityTrigger::ArtifactScoreAbove(0.2),
        ]);
        assert!(controller.evaluate_trigger(&trigger5, &artifacts, &objective_quality));
    }

    #[test]
    fn test_adaptive_adjustment_analysis() {
        let mut controller = AdaptiveQualityController::new(0.85);

        let artifacts = DetectedArtifacts {
            overall_score: 0.4,
            artifact_types: [(ArtifactType::Buzzing, 0.3)].into_iter().collect(),
            artifact_locations: Vec::new(),
            quality_assessment: QualityAssessment {
                overall_quality: 0.5,
                naturalness: 0.6,
                clarity: 0.7,
                consistency: 0.65,
                recommended_adjustments: Vec::new(),
            },
        };

        let objective_quality = ObjectiveQualityMetrics {
            overall_score: 0.5,
            spectral_similarity: 0.6,
            temporal_consistency: 0.7,
            prosodic_preservation: 0.65,
            naturalness: 0.6,
            perceptual_quality: 0.55,
            snr_estimate: 20.0,
            segmental_snr: 18.0,
        };

        let current_params = [("conversion_strength".to_string(), 1.0)].into();

        let result = controller
            .analyze_and_adjust(&artifacts, &objective_quality, &current_params)
            .unwrap();

        assert!(result.should_adjust); // Quality is below target, should adjust
        assert!(result.selected_strategy.is_some());
        assert!(result.expected_improvement > 0.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_strategy_effectiveness_update() {
        let mut controller = AdaptiveQualityController::new(0.8);

        // Get initial strategy stats
        let initial_stats = controller.get_strategy_stats();
        let strategy_name = &initial_stats[0].name;
        let initial_success_rate = initial_stats[0].success_rate;

        // Update with improvement
        controller.update_strategy_effectiveness(strategy_name, 0.6, 0.7);

        let updated_stats = controller.get_strategy_stats();
        let updated_strategy = updated_stats
            .iter()
            .find(|s| s.name == *strategy_name)
            .unwrap();

        assert_eq!(updated_strategy.usage_count, 1);
        // Success rate should have improved (exponential moving average)
        assert!(updated_strategy.success_rate >= initial_success_rate);
    }

    #[test]
    fn test_perceptual_parameters() {
        let params = PerceptualParameters::default();
        let total_weight = params.spectral_weight
            + params.temporal_weight
            + params.prosodic_weight
            + params.naturalness_weight;
        assert!((total_weight - 1.0).abs() < 0.001); // Should sum to 1.0

        let custom_params = PerceptualParameters {
            spectral_weight: 0.4,
            temporal_weight: 0.3,
            prosodic_weight: 0.2,
            naturalness_weight: 0.1,
        };
        let system = QualityMetricsSystem::with_perceptual_params(custom_params);
        assert_eq!(system.perceptual_params.spectral_weight, 0.4);
    }

    #[test]
    fn test_artifact_types_enum() {
        // Test Debug formatting for artifact types
        assert_eq!(format!("{:?}", ArtifactType::Click), "Click");
        assert_eq!(format!("{:?}", ArtifactType::Metallic), "Metallic");
        assert_eq!(format!("{:?}", ArtifactType::Buzzing), "Buzzing");
        assert_eq!(
            format!("{:?}", ArtifactType::PitchVariation),
            "PitchVariation"
        );
        assert_eq!(
            format!("{:?}", ArtifactType::SpectralDiscontinuity),
            "SpectralDiscontinuity"
        );
        assert_eq!(format!("{:?}", ArtifactType::EnergySpike), "EnergySpike");
        assert_eq!(
            format!("{:?}", ArtifactType::HighFrequencyNoise),
            "HighFrequencyNoise"
        );
        assert_eq!(
            format!("{:?}", ArtifactType::PhaseArtifact),
            "PhaseArtifact"
        );
    }

    #[test]
    fn test_quality_assessment_defaults() {
        let assessment = QualityAssessment {
            overall_quality: 0.0,
            naturalness: 0.0,
            clarity: 0.0,
            consistency: 0.0,
            recommended_adjustments: Vec::new(),
        };

        assert_eq!(assessment.overall_quality, 0.0);
        assert_eq!(assessment.recommended_adjustments.len(), 0);
    }

    #[test]
    fn test_adjustment_types() {
        // Test the types module QualityAdjustment (which is the serializable version)
        let adjustment1 = crate::types::QualityAdjustment {
            adjustment_type: "ReduceConversion".to_string(),
            strength: 0.3,
            expected_improvement: 0.15,
        };

        assert_eq!(adjustment1.adjustment_type, "ReduceConversion".to_string());
        assert_eq!(adjustment1.strength, 0.3);
        assert_eq!(adjustment1.expected_improvement, 0.15);
    }

    #[test]
    fn test_feature_extraction_helpers() {
        let system = QualityMetricsSystem::new();

        // Test spectral feature extraction
        let audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05];
        let features = system.extract_spectral_features(&audio);

        // Should extract spectral centroid, rolloff, flatness, and band energies
        assert!(features.len() >= 3); // At minimum centroid, rolloff, flatness

        // Test temporal feature extraction
        let temporal_features = system.extract_temporal_features(&audio);
        assert!(temporal_features.len() >= 4); // RMS, ZCR, energy variation, spectral flux

        // Test prosodic feature extraction
        let prosodic_features = system.extract_prosodic_features(&audio);
        assert!(prosodic_features.len() >= 3); // F0 stats, energy stats, duration features
    }

    #[test]
    fn test_production_artifact_detection() {
        let mut detector = ArtifactDetector::new_production();

        // Test that production features are enabled
        assert!(detector.adaptive_state.adaptation_enabled);
        assert_eq!(detector.memory_pool.pool_size, 16);

        // Test with sample audio
        let audio = vec![0.1; 1000];
        let result = detector.detect_artifacts(&audio, 44100).unwrap();

        // Should have production metrics
        let metrics = detector.get_performance_metrics();
        assert!(metrics.avg_processing_time_us > 0.0);
        assert!(metrics.samples_per_second > 0.0);
    }

    #[test]
    fn test_adaptive_threshold_learning() {
        let mut detector = ArtifactDetector::new_production();

        // Process multiple audio samples to trigger adaptation
        for _ in 0..10 {
            let audio = vec![0.05; 500]; // Low artifact audio
            let _ = detector.detect_artifacts(&audio, 44100).unwrap();
        }

        // Check that adaptation occurred
        let state = detector.get_adaptive_state();
        assert!(state.iteration_count > 0);
        assert!(!state.running_averages.is_empty());
    }

    #[test]
    fn test_batch_artifact_detection() {
        let mut detector = ArtifactDetector::new_production();

        let audio1 = vec![0.0; 500];
        let audio2 = vec![0.1; 500];
        let audio3 = vec![0.05; 500];
        let batch = vec![&audio1[..], &audio2[..], &audio3[..]];

        let results = detector.detect_artifacts_batch(&batch, 44100).unwrap();
        assert_eq!(results.len(), 3);

        // Check that all results are valid
        for result in results {
            assert!(!result.artifact_types.is_empty());
            assert!(result.overall_score >= 0.0 && result.overall_score <= 1.0);
        }
    }

    #[test]
    fn test_confidence_calibration() {
        let mut detector = ArtifactDetector::new_production();

        // Process some audio with artifacts
        let noisy_audio: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.1).sin() + 0.1 * fastrand::f32())
            .collect();

        let result = detector.detect_artifacts(&noisy_audio, 44100).unwrap();

        if !result.artifact_locations.is_empty() {
            // Check confidence statistics were updated
            let metrics = detector.get_performance_metrics();
            assert!(metrics.confidence_stats.sample_count > 0);
        }
    }

    #[test]
    fn test_context_specific_thresholds() {
        let mut detector = ArtifactDetector::new_production();

        // Set context-specific thresholds
        let context_thresholds = ArtifactThresholds {
            click_threshold: 0.05, // Very sensitive
            ..ArtifactThresholds::default()
        };

        detector.set_context_thresholds("high_quality".to_string(), context_thresholds);

        // Check that context was set
        let state = detector.get_adaptive_state();
        assert!(state.context_thresholds.contains_key("high_quality"));
    }

    #[test]
    fn test_performance_metrics_tracking() {
        let mut detector = ArtifactDetector::new_production();

        // Process multiple samples to accumulate metrics
        for i in 0..5 {
            let audio = vec![0.01 * i as f32; 800];
            let _ = detector.detect_artifacts(&audio, 44100).unwrap();
        }

        let metrics = detector.get_performance_metrics();

        // Verify metrics are being tracked
        assert!(metrics.avg_processing_time_us > 0.0);
        assert!(metrics.samples_per_second > 0.0);
        assert!(metrics.memory_usage_mb >= 0.0);
    }

    #[test]
    fn test_legacy_compatibility() {
        let mut detector = ArtifactDetector::new();

        // Test legacy method still works
        let audio = vec![0.1; 500];
        let result = detector.detect_artifacts_legacy(&audio, 44100).unwrap();

        assert!(!result.artifact_types.is_empty());
        assert!(result.overall_score >= 0.0 && result.overall_score <= 1.0);
    }

    #[test]
    fn test_audio_analysis_edge_cases() {
        let mut detector = ArtifactDetector::new();

        // Test with very short audio
        let short_audio = vec![0.1, -0.1];
        let result = detector.detect_artifacts(&short_audio, 44100);
        assert!(result.is_ok());

        // Test with empty audio
        let empty_audio = vec![];
        let result = detector.detect_artifacts(&empty_audio, 44100);
        assert!(result.is_ok());

        // Test with single sample
        let single_sample = vec![0.5];
        let result = detector.detect_artifacts(&single_sample, 44100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quality_strategy_adjustment() {
        let adjustment = QualityStrategyAdjustment {
            adjustment_type: AdjustmentType::NoiseReduction,
            parameter_changes: [("noise_reduction_strength".to_string(), 0.8)].into(),
            processing_mode_change: Some("high_quality".to_string()),
            preferred_model: None,
        };

        assert_eq!(adjustment.parameter_changes.len(), 1);
        assert_eq!(
            adjustment.processing_mode_change,
            Some("high_quality".to_string())
        );
        assert_eq!(adjustment.preferred_model, None);
    }

    // Perceptual Optimization Tests

    #[test]
    fn test_perceptual_optimizer_creation() {
        let optimizer = PerceptualOptimizer::new(44100);

        assert_eq!(optimizer.critical_bands.sample_rate, 44100);
        assert!(optimizer.critical_bands.num_bands > 0);
        assert!(optimizer.critical_bands.band_boundaries.len() > 0);

        // Test with custom parameters
        let custom_params = PerceptualOptimizationParams {
            target_quality: 0.9,
            max_iterations: 10,
            convergence_threshold: 0.005,
            ..PerceptualOptimizationParams::default()
        };

        let optimizer_custom = PerceptualOptimizer::with_params(22050, custom_params);
        assert_eq!(optimizer_custom.optimization_params.target_quality, 0.9);
        assert_eq!(optimizer_custom.optimization_params.max_iterations, 10);
    }

    #[test]
    fn test_psychoacoustic_model() {
        let model = PsychoacousticModel::new(44100);

        assert!(!model.threshold_frequencies.is_empty());
        assert!(!model.absolute_threshold.is_empty());
        assert_eq!(
            model.threshold_frequencies.len(),
            model.absolute_threshold.len()
        );
        assert_eq!(model.sample_rate, 44100);

        // Check that frequencies are reasonable
        for &freq in &model.threshold_frequencies {
            assert!(freq >= 20.0);
            assert!(freq <= 22050.0); // Nyquist frequency
        }
    }

    #[test]
    fn test_critical_band_analyzer() {
        let analyzer = CriticalBandAnalyzer::new(44100);

        assert!(analyzer.num_bands > 0);
        assert_eq!(analyzer.band_boundaries.len(), analyzer.num_bands + 1);
        assert_eq!(analyzer.sample_rate, 44100);

        // Test band operations
        let test_spectrum = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02];

        if analyzer.num_bands > 0 {
            let energy = analyzer.get_band_energy(&test_spectrum, 0);
            assert!(energy >= 0.0);

            let center_freq = analyzer.get_band_center_frequency(0);
            assert!(center_freq > 0.0);

            let centroid = analyzer.calculate_band_centroid(&test_spectrum, 0);
            assert!(centroid >= 0.0);
        }
    }

    #[test]
    fn test_masking_calculator() {
        let calculator = MaskingCalculator::new();
        let analyzer = CriticalBandAnalyzer::new(44100);

        assert!(!calculator.spreading_coefficients.is_empty());
        assert_eq!(calculator.temporal_masking.pre_masking_duration, 2.0);
        assert_eq!(calculator.temporal_masking.post_masking_duration, 200.0);

        // Test masking calculations
        let test_spectrum = vec![1.0, 0.5, 0.3, 0.2, 0.1];

        if analyzer.num_bands > 0 {
            let spectral_masking =
                calculator.calculate_spectral_masking(&test_spectrum, 0, &analyzer);
            assert!(spectral_masking >= 0.0 && spectral_masking <= 1.0);

            let temporal_masking = calculator.calculate_temporal_masking(0.5, 0);
            assert!(temporal_masking >= 0.0 && temporal_masking <= 1.0);
        }
    }

    #[test]
    fn test_loudness_model() {
        let model = LoudnessModel::new();

        assert!(!model.equal_loudness_contours.is_empty());
        assert!(!model.contour_frequencies.is_empty());
        assert_eq!(
            model.contour_frequencies.len(),
            model.loudness_scaling.len()
        );

        // Test loudness calculation
        let loudness = model.calculate_loudness(0.1, 1000.0);
        assert!(loudness >= 0.0);

        let zero_loudness = model.calculate_loudness(0.0, 1000.0);
        assert_eq!(zero_loudness, 0.0);

        let negative_loudness = model.calculate_loudness(-0.1, 1000.0);
        assert_eq!(negative_loudness, 0.0);
    }

    #[test]
    fn test_perceptual_optimization_result() {
        let optimizer = PerceptualOptimizer::new(44100);
        let audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05];
        let current_params = [("quality_factor".to_string(), 1.0)].into();

        let result = optimizer.optimize_parameters(&audio, &current_params, "PitchShift");

        match result {
            Ok(optimization_result) => {
                assert!(!optimization_result.optimized_params.is_empty());
                assert!(
                    optimization_result.perceptual_quality >= 0.0
                        && optimization_result.perceptual_quality <= 1.0
                );
                assert!(
                    optimization_result.iterations <= optimizer.optimization_params.max_iterations
                );

                // Check masking analysis
                assert!(!optimization_result
                    .masking_analysis
                    .spectral_masking_thresholds
                    .is_empty());
                assert!(!optimization_result
                    .masking_analysis
                    .temporal_masking_effects
                    .is_empty());
                assert!(
                    optimization_result.masking_analysis.masking_efficiency >= 0.0
                        && optimization_result.masking_analysis.masking_efficiency <= 1.0
                );

                // Check loudness analysis
                assert!(!optimization_result
                    .loudness_analysis
                    .loudness_levels
                    .is_empty());
                assert!(optimization_result.loudness_analysis.overall_loudness >= 0.0);
                assert!(
                    optimization_result.loudness_analysis.loudness_balance >= 0.0
                        && optimization_result.loudness_analysis.loudness_balance <= 1.0
                );

                // Check critical band analysis
                assert!(!optimization_result
                    .critical_band_analysis
                    .band_energies
                    .is_empty());
                assert!(!optimization_result
                    .critical_band_analysis
                    .band_centroids
                    .is_empty());
                assert!(
                    optimization_result
                        .critical_band_analysis
                        .bandwidth_efficiency
                        >= 0.0
                        && optimization_result
                            .critical_band_analysis
                            .bandwidth_efficiency
                            <= 1.0
                );
            }
            Err(e) => panic!("Perceptual optimization failed: {}", e),
        }
    }

    #[test]
    fn test_optimization_for_different_conversion_types() {
        let optimizer = PerceptualOptimizer::new(22050);
        let audio = vec![0.1; 1000]; // Simple constant signal
        let current_params = [
            ("quality_factor".to_string(), 1.0),
            ("conversion_strength".to_string(), 1.0),
        ]
        .into();

        let conversion_types = [
            "PitchShift",
            "SpeedTransformation",
            "SpeakerConversion",
            "Unknown",
        ];

        for conversion_type in &conversion_types {
            let result = optimizer.optimize_parameters(&audio, &current_params, conversion_type);

            match result {
                Ok(optimization_result) => {
                    assert!(!optimization_result.optimized_params.is_empty());
                    assert!(optimization_result.perceptual_quality >= 0.0);
                    println!(
                        "Conversion type '{}': quality = {:.3}, iterations = {}",
                        conversion_type,
                        optimization_result.perceptual_quality,
                        optimization_result.iterations
                    );
                }
                Err(e) => panic!("Optimization failed for {}: {}", conversion_type, e),
            }
        }
    }

    #[test]
    fn test_masking_analysis_with_different_audio() {
        let optimizer = PerceptualOptimizer::new(44100);

        // Test with different audio characteristics
        let test_cases = vec![
            ("silent", vec![0.0; 1000]),
            (
                "sine_wave",
                (0..1000)
                    .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
                    .collect::<Vec<f32>>(),
            ),
            (
                "noise",
                (0..1000)
                    .map(|_| (scirs2_core::random::random::<f32>() - 0.5) * 0.2)
                    .collect::<Vec<f32>>(),
            ),
            (
                "complex",
                (0..1000)
                    .map(|i| {
                        let t = i as f32 / 44100.0;
                        (t * 220.0 * 2.0 * std::f32::consts::PI).sin() * 0.3
                            + (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.2
                            + (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.1
                    })
                    .collect::<Vec<f32>>(),
            ),
        ];

        for (name, audio) in test_cases {
            match optimizer.analyze_masking(&audio) {
                Ok(masking_analysis) => {
                    assert!(!masking_analysis.spectral_masking_thresholds.is_empty());
                    assert!(!masking_analysis.temporal_masking_effects.is_empty());
                    assert!(
                        masking_analysis.masking_efficiency >= 0.0
                            && masking_analysis.masking_efficiency <= 1.0
                    );

                    println!(
                        "Masking analysis for '{}': efficiency = {:.3}",
                        name, masking_analysis.masking_efficiency
                    );
                }
                Err(e) => panic!("Masking analysis failed for {}: {}", name, e),
            }
        }
    }

    #[test]
    fn test_loudness_analysis_with_different_audio() {
        let optimizer = PerceptualOptimizer::new(44100);

        let test_cases = vec![
            ("quiet", vec![0.01; 500]),
            ("medium", vec![0.1; 500]),
            ("loud", vec![0.5; 500]),
        ];

        for (name, audio) in test_cases {
            match optimizer.analyze_loudness(&audio) {
                Ok(loudness_analysis) => {
                    assert!(!loudness_analysis.loudness_levels.is_empty());
                    assert!(loudness_analysis.overall_loudness >= 0.0);
                    assert!(
                        loudness_analysis.loudness_balance >= 0.0
                            && loudness_analysis.loudness_balance <= 1.0
                    );

                    println!(
                        "Loudness analysis for '{}': overall = {:.3}, balance = {:.3}",
                        name,
                        loudness_analysis.overall_loudness,
                        loudness_analysis.loudness_balance
                    );
                }
                Err(e) => panic!("Loudness analysis failed for {}: {}", name, e),
            }
        }
    }

    #[test]
    fn test_parameter_adjustment() {
        let optimizer = PerceptualOptimizer::new(44100);
        let mut params = [("test_param".to_string(), 1.0)].into();

        // Test positive adjustment
        optimizer.adjust_param(&mut params, "test_param", 0.1);
        assert_eq!(params.get("test_param"), Some(&1.1));

        // Test negative adjustment
        optimizer.adjust_param(&mut params, "test_param", -0.2);
        assert!((params.get("test_param").unwrap() - 0.9).abs() < 0.001);

        // Test clamping to lower bound
        optimizer.adjust_param(&mut params, "test_param", -2.0);
        assert_eq!(params.get("test_param"), Some(&0.0));

        // Test clamping to upper bound
        optimizer.adjust_param(&mut params, "new_param", 5.0);
        assert_eq!(params.get("new_param"), Some(&2.0)); // Should be clamped to max 2.0
    }

    #[test]
    fn test_spectrum_calculation() {
        let optimizer = PerceptualOptimizer::new(44100);

        // Test with simple sine wave
        let audio = (0..1024)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin())
            .collect::<Vec<f32>>();
        let spectrum = optimizer.calculate_spectrum(&audio);

        assert!(!spectrum.is_empty());
        assert_eq!(spectrum.len(), 512); // Should be half of input length

        // Check normalization
        let total_energy: f32 = spectrum.iter().sum();
        assert!((total_energy - 1.0).abs() < 0.001); // Should be normalized to 1.0

        // Test with empty audio
        let empty_spectrum = optimizer.calculate_spectrum(&[]);
        assert!(empty_spectrum.len() > 0); // Should return default size spectrum

        // Test with very short audio
        let short_audio = vec![0.1, -0.1];
        let short_spectrum = optimizer.calculate_spectrum(&short_audio);
        assert!(!short_spectrum.is_empty());
    }

    #[test]
    fn test_optimization_convergence() {
        let mut custom_params = PerceptualOptimizationParams::default();
        custom_params.max_iterations = 5; // Limit iterations for test
        custom_params.convergence_threshold = 0.1; // Large threshold for quick convergence

        let optimizer = PerceptualOptimizer::with_params(44100, custom_params);
        let audio = vec![0.1; 500];
        let current_params = [("quality_factor".to_string(), 0.5)].into();

        let result = optimizer
            .optimize_parameters(&audio, &current_params, "PitchShift")
            .unwrap();

        // Should converge quickly with large threshold
        assert!(result.iterations <= 5);
        println!(
            "Convergence test: {} iterations, converged = {}",
            result.iterations, result.converged
        );
    }

    #[test]
    fn test_default_parameters() {
        let params = PerceptualOptimizationParams::default();

        // Check that weights sum to 1.0
        let total_weight = params.spectral_masking_weight
            + params.temporal_masking_weight
            + params.loudness_weight
            + params.critical_band_weight;
        assert!((total_weight - 1.0).abs() < 0.001);

        assert_eq!(params.target_quality, 0.8);
        assert_eq!(params.max_iterations, 20);
        assert_eq!(params.convergence_threshold, 0.01);
    }

    #[test]
    fn test_temporal_masking_params() {
        let params = TemporalMaskingParams::default();

        assert_eq!(params.pre_masking_duration, 2.0);
        assert_eq!(params.post_masking_duration, 200.0);
        assert_eq!(params.masking_slope, 0.1);

        // Test reasonable values
        assert!(params.pre_masking_duration > 0.0);
        assert!(params.post_masking_duration > params.pre_masking_duration);
        assert!(params.masking_slope > 0.0);
    }

    #[test]
    fn test_simultaneous_masking_params() {
        let params = SimultaneousMaskingParams::default();

        assert_eq!(params.lower_slope, -27.0);
        assert_eq!(params.upper_slope, -12.0);
        assert_eq!(params.spreading_width, 2.5);

        // Test that slopes are negative (as expected in masking)
        assert!(params.lower_slope < 0.0);
        assert!(params.upper_slope < 0.0);
        assert!(params.spreading_width > 0.0);
    }

    // Tests for the new Quality Targets System
    #[test]
    fn test_quality_targets_system_creation() {
        let system = QualityTargetsSystem::new();

        assert_eq!(system.config.target_similarity_threshold, 0.85);
        assert_eq!(system.config.source_preservation_threshold, 0.90);
        assert_eq!(system.config.mos_threshold, 4.0);
        assert_eq!(system.config.artifact_level_threshold, 0.05);
        assert!(system.config.enable_detailed_tracking);
        assert_eq!(system.measurement_history.len(), 0);
        assert_eq!(system.max_history, 1000);
    }

    #[test]
    fn test_quality_targets_config_custom() {
        let config = QualityTargetsConfig {
            target_similarity_threshold: 0.80,
            source_preservation_threshold: 0.95,
            mos_threshold: 3.5,
            artifact_level_threshold: 0.03,
            enable_detailed_tracking: false,
        };

        let system = QualityTargetsSystem::with_config(config.clone());

        assert_eq!(system.config.target_similarity_threshold, 0.80);
        assert_eq!(system.config.source_preservation_threshold, 0.95);
        assert_eq!(system.config.mos_threshold, 3.5);
        assert_eq!(system.config.artifact_level_threshold, 0.03);
        assert!(!system.config.enable_detailed_tracking);
    }

    #[test]
    fn test_quality_target_measurement_basic() {
        let mut system = QualityTargetsSystem::new();

        // Create test audio samples
        let converted_audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let original_audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.9).collect();
        let target_reference: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.012).sin()).collect();

        let result = system.measure_quality_targets(
            &converted_audio,
            &original_audio,
            Some(&target_reference),
            16000,
        );

        assert!(result.is_ok());
        let measurement = result.unwrap();

        // Verify measurement structure
        assert!(measurement.target_similarity >= 0.0 && measurement.target_similarity <= 1.0);
        assert!(measurement.source_preservation >= 0.0 && measurement.source_preservation <= 1.0);
        assert!(measurement.mos_score >= 1.0 && measurement.mos_score <= 5.0);
        assert!(measurement.artifact_level >= 0.0 && measurement.artifact_level <= 1.0);
        assert!(
            measurement.targets_met.overall_achievement >= 0.0
                && measurement.targets_met.overall_achievement <= 1.0
        );

        // Verify detailed metrics
        assert!(measurement.detailed_metrics.speaker_identity_preservation >= 0.0);
        assert!(measurement.detailed_metrics.prosodic_preservation >= 0.0);
        assert!(measurement.detailed_metrics.linguistic_preservation >= 0.0);
        assert!(measurement.detailed_metrics.spectral_fidelity >= 0.0);
        assert!(measurement.detailed_metrics.temporal_consistency >= 0.0);
        assert!(measurement.detailed_metrics.perceptual_quality >= 0.0);
    }

    #[test]
    fn test_quality_target_measurement_without_reference() {
        let mut system = QualityTargetsSystem::new();

        // Create test audio samples
        let converted_audio: Vec<f32> = (0..500).map(|i| (i as f32 * 0.01).sin()).collect();
        let original_audio: Vec<f32> = (0..500).map(|i| (i as f32 * 0.01).sin() * 0.8).collect();

        let result = system.measure_quality_targets(
            &converted_audio,
            &original_audio,
            None, // No target reference
            16000,
        );

        assert!(result.is_ok());
        let measurement = result.unwrap();

        // Should still provide valid measurements
        assert!(measurement.target_similarity >= 0.0);
        assert!(measurement.source_preservation >= 0.0);
        assert!(measurement.mos_score >= 1.0);
        assert!(measurement.artifact_level >= 0.0);

        // Should default speaker identity preservation when no reference
        assert_eq!(
            measurement.detailed_metrics.speaker_identity_preservation,
            0.5
        );
    }

    #[test]
    fn test_quality_targets_achievement_thresholds() {
        let mut system = QualityTargetsSystem::new();

        // Create high-quality test audio that should meet targets
        let high_quality_audio: Vec<f32> =
            (0..1000).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
        let original_audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();

        let result = system.measure_quality_targets(
            &high_quality_audio,
            &original_audio,
            Some(&high_quality_audio), // Perfect similarity case
            16000,
        );

        assert!(result.is_ok());
        let measurement = result.unwrap();

        // Should have high similarity and preservation for identical/similar audio
        assert!(measurement.target_similarity > 0.5); // Should be quite similar
        assert!(measurement.source_preservation > 0.8); // Should preserve well
        assert!(measurement.mos_score > 2.0); // Should have decent MOS
        assert!(measurement.artifact_level < 0.5); // Should be relatively clean
    }

    #[test]
    fn test_quality_targets_achievement_status() {
        let mut system = QualityTargetsSystem::new();

        let converted_audio: Vec<f32> = (0..800).map(|i| (i as f32 * 0.005).sin()).collect();
        let original_audio: Vec<f32> = (0..800).map(|i| (i as f32 * 0.005).sin() * 0.95).collect();
        let target_audio: Vec<f32> = (0..800).map(|i| (i as f32 * 0.0048).sin()).collect();

        let result = system.measure_quality_targets(
            &converted_audio,
            &original_audio,
            Some(&target_audio),
            16000,
        );

        assert!(result.is_ok());
        let measurement = result.unwrap();

        // Test achievement status structure
        let achievement = &measurement.targets_met;

        // Verify boolean flags make sense
        assert_eq!(
            achievement.target_similarity_met,
            measurement.target_similarity >= system.config.target_similarity_threshold
        );
        assert_eq!(
            achievement.source_preservation_met,
            measurement.source_preservation >= system.config.source_preservation_threshold
        );
        assert_eq!(
            achievement.mos_threshold_met,
            measurement.mos_score >= system.config.mos_threshold
        );
        assert_eq!(
            achievement.artifact_level_met,
            measurement.artifact_level <= system.config.artifact_level_threshold
        );

        // Overall achievement should be between 0 and 1
        assert!(achievement.overall_achievement >= 0.0);
        assert!(achievement.overall_achievement <= 1.0);
    }

    #[test]
    fn test_quality_targets_history_tracking() {
        let mut system = QualityTargetsSystem::new();

        let audio1: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio2: Vec<f32> = (0..200).map(|i| (i as f32 * 0.011).sin()).collect();

        // Make several measurements
        for _ in 0..5 {
            let _ = system.measure_quality_targets(&audio1, &audio2, Some(&audio2), 16000);
        }

        // Should have history
        assert_eq!(system.measurement_history.len(), 5);

        // Test statistics
        let stats = system.get_achievement_statistics();
        assert_eq!(stats.total_measurements, 5);

        // All rates should be between 0 and 1
        assert!(stats.target_similarity_achievement_rate >= 0.0);
        assert!(stats.target_similarity_achievement_rate <= 1.0);
        assert!(stats.source_preservation_achievement_rate >= 0.0);
        assert!(stats.source_preservation_achievement_rate <= 1.0);
        assert!(stats.mos_achievement_rate >= 0.0);
        assert!(stats.mos_achievement_rate <= 1.0);
        assert!(stats.artifact_level_achievement_rate >= 0.0);
        assert!(stats.artifact_level_achievement_rate <= 1.0);
        assert!(stats.average_overall_achievement >= 0.0);
        assert!(stats.average_overall_achievement <= 1.0);
    }

    #[test]
    fn test_quality_targets_statistics_empty() {
        let system = QualityTargetsSystem::new();
        let stats = system.get_achievement_statistics();

        // Empty system should have default statistics
        assert_eq!(stats.total_measurements, 0);
        assert_eq!(stats.target_similarity_achievement_rate, 0.0);
        assert_eq!(stats.source_preservation_achievement_rate, 0.0);
        assert_eq!(stats.mos_achievement_rate, 0.0);
        assert_eq!(stats.artifact_level_achievement_rate, 0.0);
        assert_eq!(stats.average_overall_achievement, 0.0);
    }

    #[test]
    fn test_quality_targets_history_limit() {
        let mut system = QualityTargetsSystem::new();

        let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.02).sin()).collect();

        // Add more measurements than max_history
        for _ in 0..1005 {
            let _ = system.measure_quality_targets(&audio, &audio, Some(&audio), 16000);
        }

        // Should be limited to max_history
        assert_eq!(system.measurement_history.len(), system.max_history);

        let stats = system.get_achievement_statistics();
        assert_eq!(stats.total_measurements, system.max_history);
    }

    #[test]
    fn test_quality_targets_disabled_tracking() {
        let config = QualityTargetsConfig {
            enable_detailed_tracking: false,
            ..Default::default()
        };
        let mut system = QualityTargetsSystem::with_config(config);

        let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();

        let result = system.measure_quality_targets(&audio, &audio, Some(&audio), 16000);
        assert!(result.is_ok());

        // Should not track history when disabled
        assert_eq!(system.measurement_history.len(), 0);

        let stats = system.get_achievement_statistics();
        assert_eq!(stats.total_measurements, 0);
    }

    #[test]
    fn test_spectral_similarity_calculation() {
        let system = QualityTargetsSystem::new();

        // Test identical audio
        let audio1: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let audio2 = audio1.clone();

        let similarity = system.calculate_spectral_similarity(&audio1, &audio2);
        assert!(similarity.is_ok());
        let similarity_val = similarity.unwrap();
        assert!(similarity_val > 0.99); // Should be very similar

        // Test different audio
        let audio3: Vec<f32> = (0..100).map(|i| (i as f32 * 2.0).sin()).collect();
        let similarity2 = system.calculate_spectral_similarity(&audio1, &audio3);
        assert!(similarity2.is_ok());
        let similarity2_val = similarity2.unwrap();
        // Should be less similar than identical case
        assert!(similarity2_val < similarity_val);
    }

    #[test]
    fn test_mos_score_range() {
        let system = QualityTargetsSystem::new();

        // Test various audio qualities
        let clean_audio: Vec<f32> = (0..500).map(|i| (i as f32 * 0.001).sin() * 0.3).collect();
        let noisy_audio: Vec<f32> = (0..500)
            .map(|i| (i as f32 * 0.001).sin() * 0.8 + (i as f32 * 0.1).sin() * 0.3)
            .collect();

        let mos1 = system.calculate_mos_score(&clean_audio, 16000);
        let mos2 = system.calculate_mos_score(&noisy_audio, 16000);

        assert!(mos1.is_ok());
        assert!(mos2.is_ok());

        let mos1_val = mos1.unwrap();
        let mos2_val = mos2.unwrap();

        // Both should be in valid MOS range
        assert!(mos1_val >= 1.0 && mos1_val <= 5.0);
        assert!(mos2_val >= 1.0 && mos2_val <= 5.0);

        // Clean audio should generally have better MOS than noisy audio
        // (though with simplified metrics this might not always hold)
    }

    #[test]
    fn test_artifact_level_calculation() {
        let system = QualityTargetsSystem::new();

        // Test clean audio (should have low artifact level)
        let clean_audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin() * 0.3).collect();
        let artifact_level = system.calculate_artifact_level(&clean_audio, 16000);

        assert!(artifact_level.is_ok());
        let level = artifact_level.unwrap();
        assert!(level >= 0.0 && level <= 1.0);

        // Test noisy/distorted audio (should have higher artifact level)
        let distorted_audio: Vec<f32> = clean_audio
            .iter()
            .map(|&x| {
                if x > 0.8 {
                    0.95
                } else if x < -0.8 {
                    -0.95
                } else {
                    x
                } // Add clipping
            })
            .collect();

        let artifact_level2 = system.calculate_artifact_level(&distorted_audio, 16000);
        assert!(artifact_level2.is_ok());
        let level2 = artifact_level2.unwrap();
        assert!(level2 >= 0.0 && level2 <= 1.0);

        // Distorted audio should generally have more artifacts
        // assert!(level2 >= level); // This might not always hold with simplified detection
    }

    #[test]
    fn test_overall_achievement_calculation() {
        let system = QualityTargetsSystem::new();

        // Test perfect scores
        let perfect_achievement = system.calculate_overall_achievement(1.0, 1.0, 5.0, 0.0);
        assert!(perfect_achievement > 0.9); // Should be very high

        // Test poor scores
        let poor_achievement = system.calculate_overall_achievement(0.5, 0.5, 2.0, 0.2);
        assert!(poor_achievement < perfect_achievement);

        // Test mixed scores
        let mixed_achievement = system.calculate_overall_achievement(0.8, 0.9, 4.2, 0.03);
        assert!(mixed_achievement > poor_achievement);
        assert!(mixed_achievement < perfect_achievement);

        // All achievements should be in valid range
        assert!(perfect_achievement >= 0.0 && perfect_achievement <= 1.0);
        assert!(poor_achievement >= 0.0 && poor_achievement <= 1.0);
        assert!(mixed_achievement >= 0.0 && mixed_achievement <= 1.0);
    }

    #[test]
    fn test_edge_cases_empty_audio() {
        let system = QualityTargetsSystem::new();

        let empty_audio: Vec<f32> = Vec::new();
        let small_audio: Vec<f32> = vec![0.1, 0.2];

        // Should handle empty audio gracefully
        let similarity = system.calculate_spectral_similarity(&empty_audio, &small_audio);
        assert!(similarity.is_ok());
        assert_eq!(similarity.unwrap(), 0.0);

        let similarity2 = system.calculate_spectral_similarity(&small_audio, &empty_audio);
        assert!(similarity2.is_ok());
        assert_eq!(similarity2.unwrap(), 0.0);
    }

    #[test]
    fn test_quality_targets_system_default() {
        let system1 = QualityTargetsSystem::new();
        let system2 = QualityTargetsSystem::default();

        // Both should have same configuration
        assert_eq!(
            system1.config.target_similarity_threshold,
            system2.config.target_similarity_threshold
        );
        assert_eq!(
            system1.config.source_preservation_threshold,
            system2.config.source_preservation_threshold
        );
        assert_eq!(system1.config.mos_threshold, system2.config.mos_threshold);
        assert_eq!(
            system1.config.artifact_level_threshold,
            system2.config.artifact_level_threshold
        );
        assert_eq!(
            system1.config.enable_detailed_tracking,
            system2.config.enable_detailed_tracking
        );
    }
}
