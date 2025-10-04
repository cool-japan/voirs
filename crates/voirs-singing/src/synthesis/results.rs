//! Synthesis results and metrics

use std::time::Duration;

/// Synthesis result containing audio and metadata
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Synthesized audio samples in range [-1.0, 1.0]
    pub audio: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Total duration of audio
    pub duration: Duration,
    /// Performance statistics from synthesis
    pub stats: SynthesisStats,
    /// Quality metrics for the synthesized audio
    pub quality_metrics: QualityMetrics,
}

/// Synthesis statistics
#[derive(Debug, Clone)]
pub struct SynthesisStats {
    /// Processing time
    pub processing_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Frame count
    pub frame_count: usize,
    /// Synthesis quality
    pub quality: f32,
    /// Error count
    pub error_count: usize,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Pitch accuracy
    pub pitch_accuracy: f32,
    /// Spectral quality
    pub spectral_quality: f32,
    /// Harmonic quality
    pub harmonic_quality: f32,
    /// Noise level
    pub noise_level: f32,
    /// Formant quality
    pub formant_quality: f32,
    /// Overall quality
    pub overall_quality: f32,
}

/// Precision metrics for enhanced quality analysis
#[derive(Debug, Clone)]
pub struct PrecisionMetricsReport {
    /// Overall synthesis quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Pitch accuracy metrics
    pub pitch_accuracy: PitchAccuracyMetrics,
    /// Timing precision metrics
    pub timing_accuracy: TimingAccuracyMetrics,
    /// Spectral quality metrics
    pub spectral_quality: SpectralQualityMetrics,
    /// Expression fidelity metrics
    pub expression_fidelity: ExpressionFidelityMetrics,
}

/// Pitch accuracy metrics
#[derive(Debug, Clone)]
pub struct PitchAccuracyMetrics {
    /// Mean absolute error in cents
    pub mean_error_cents: f32,
    /// Standard deviation of error in cents
    pub std_error_cents: f32,
    /// Percentage of notes within acceptable range
    pub notes_in_range_percent: f32,
    /// Maximum error observed in cents
    pub max_error_cents: f32,
}

/// Timing accuracy metrics
#[derive(Debug, Clone)]
pub struct TimingAccuracyMetrics {
    /// Mean absolute timing error in milliseconds
    pub mean_error_ms: f32,
    /// Standard deviation of timing error
    pub std_error_ms: f32,
    /// Percentage of events within timing tolerance
    pub events_in_tolerance_percent: f32,
    /// Rhythmic stability score
    pub rhythmic_stability: f32,
}

/// Spectral quality metrics
#[derive(Debug, Clone)]
pub struct SpectralQualityMetrics {
    /// Spectral centroid stability
    pub centroid_stability: f32,
    /// Harmonic-to-noise ratio
    pub hnr_db: f32,
    /// Formant tracking accuracy
    pub formant_accuracy: f32,
    /// Spectral envelope smoothness
    pub envelope_smoothness: f32,
}

/// Expression fidelity metrics
#[derive(Debug, Clone)]
pub struct ExpressionFidelityMetrics {
    /// Dynamic range utilization
    pub dynamic_range_usage: f32,
    /// Vibrato consistency
    pub vibrato_consistency: f32,
    /// Articulation clarity
    pub articulation_clarity: f32,
    /// Emotional expression accuracy
    pub emotion_accuracy: f32,
}

/// Precision targets for quality control
#[derive(Debug, Clone)]
pub struct PrecisionTargets {
    /// Target pitch accuracy in cents (default: 10.0)
    pub pitch_accuracy_cents: f32,
    /// Target timing accuracy in milliseconds (default: 50.0)
    pub timing_accuracy_ms: f32,
    /// Target minimum HNR in dB (default: 15.0)
    pub min_hnr_db: f32,
    /// Target minimum overall quality (default: 0.8)
    pub min_overall_quality: f32,
}

impl Default for SynthesisStats {
    fn default() -> Self {
        Self {
            processing_time: Duration::from_secs(0),
            memory_usage: 0,
            cpu_usage: 0.0,
            frame_count: 0,
            quality: 0.0,
            error_count: 0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            pitch_accuracy: 0.0,
            spectral_quality: 0.0,
            harmonic_quality: 0.0,
            noise_level: 0.0,
            formant_quality: 0.0,
            overall_quality: 0.0,
        }
    }
}

impl Default for PrecisionTargets {
    fn default() -> Self {
        Self {
            pitch_accuracy_cents: 10.0,
            timing_accuracy_ms: 50.0,
            min_hnr_db: 15.0,
            min_overall_quality: 0.8,
        }
    }
}

impl PrecisionMetricsReport {
    /// Check if all precision targets are met
    pub fn meets_targets(&self, targets: &PrecisionTargets) -> bool {
        self.pitch_accuracy.mean_error_cents <= targets.pitch_accuracy_cents
            && self.timing_accuracy.mean_error_ms <= targets.timing_accuracy_ms
            && self.spectral_quality.hnr_db >= targets.min_hnr_db
            && self.overall_quality >= targets.min_overall_quality
    }

    /// Get a summary of precision metrics
    pub fn summary(&self) -> String {
        format!(
            "Quality: {:.2}, Pitch: {:.1}¢, Timing: {:.1}ms, HNR: {:.1}dB",
            self.overall_quality,
            self.pitch_accuracy.mean_error_cents,
            self.timing_accuracy.mean_error_ms,
            self.spectral_quality.hnr_db
        )
    }
}

impl SynthesisResult {
    /// Get duration in seconds
    ///
    /// # Returns
    ///
    /// Duration as floating point seconds
    pub fn duration_secs(&self) -> f32 {
        self.duration.as_secs_f32()
    }

    /// Get audio length in samples
    ///
    /// # Returns
    ///
    /// Number of audio samples
    pub fn sample_count(&self) -> usize {
        self.audio.len()
    }

    /// Check if the result meets quality thresholds
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum quality score (0.0-1.0)
    ///
    /// # Returns
    ///
    /// true if quality meets or exceeds threshold
    pub fn meets_quality_threshold(&self, threshold: f32) -> bool {
        self.quality_metrics.overall_quality >= threshold
    }

    /// Get audio as 16-bit PCM samples
    ///
    /// Converts normalized float samples to 16-bit integers.
    ///
    /// # Returns
    ///
    /// Vector of 16-bit PCM samples
    pub fn to_pcm16(&self) -> Vec<i16> {
        self.audio
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }
}
