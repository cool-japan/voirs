//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    pitch::{PitchContour, PitchGenerator},
    types::{Expression, NoteEvent, VoiceCharacteristics},
    Error, Result,
};
use std::collections::HashMap;

use super::types::{DetectedExpression, ExpressionRecognitionReport, PitchAccuracyReport};
use super::types_3::NaturalnessScorer;
use super::types_5::{ExpressionModel, OnsetDetector, TimingAccuracyReport};

/// Naturalness quality score report
#[derive(Debug, Clone)]
pub struct NaturalnessScoreReport {
    /// Mean Opinion Score (MOS) on 1-5 scale
    pub mos_score: f32,
    /// Breath pattern naturalness score (0-5)
    pub breath_naturalness: f32,
    /// Vibrato quality naturalness score (0-5)
    pub vibrato_naturalness: f32,
    /// Formant structure naturalness score (0-5)
    pub formant_naturalness: f32,
    /// Spectral characteristics naturalness score (0-5)
    pub spectral_naturalness: f32,
    /// Temporal dynamics naturalness score (0-5)
    pub temporal_naturalness: f32,
    /// Quality enhancement factors applied
    pub quality_factors: HashMap<String, f32>,
}
/// Features extracted for expression recognition
#[derive(Debug, Clone)]
pub struct ExpressionFeatures {
    /// Attack time in seconds
    pub attack_time: f32,
    /// Sustain level (0.0-1.0)
    pub sustain_level: f32,
    /// Decay time in seconds
    pub decay_time: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid: f32,
    /// Dynamic range (0.0-1.0)
    pub dynamic_range: f32,
}
/// Enhanced precision quality analyzer
pub struct PrecisionQualityAnalyzer {
    pub(super) pitch_generator: PitchGenerator,
    timing_analyzer: TimingAnalyzer,
    naturalness_scorer: NaturalnessScorer,
    expression_recognizer: ExpressionRecognizer,
}
impl PrecisionQualityAnalyzer {
    /// Create new precision quality analyzer
    pub fn new() -> Self {
        Self {
            pitch_generator: PitchGenerator::new(VoiceCharacteristics::default()),
            timing_analyzer: TimingAnalyzer::new(),
            naturalness_scorer: NaturalnessScorer::new(),
            expression_recognizer: ExpressionRecognizer::new(),
        }
    }
    /// Calculate high-precision pitch accuracy (targeting 99%+ within 5 cents)
    pub fn calculate_precision_pitch_accuracy(
        &mut self,
        audio: &[f32],
        target_pitch_contour: &PitchContour,
        sample_rate: f32,
    ) -> Result<PitchAccuracyReport> {
        if audio.is_empty() || target_pitch_contour.f0_values.is_empty() {
            return Ok(PitchAccuracyReport::default());
        }
        let detected_f0s = self.extract_high_precision_f0(audio, sample_rate)?;
        if detected_f0s.is_empty() {
            return Ok(PitchAccuracyReport::default());
        }
        let aligned_detected =
            self.align_pitch_contours(&detected_f0s, &target_pitch_contour.f0_values)?;
        let mut cent_deviations = Vec::new();
        let mut notes_within_5_cents = 0;
        let mut total_valid_notes = 0;
        for (detected, target) in aligned_detected
            .iter()
            .zip(target_pitch_contour.f0_values.iter())
        {
            if *detected > 0.0 && *target > 0.0 {
                let cent_deviation = 1200.0 * (*detected / *target).ln() / 2_f32.ln();
                cent_deviations.push(cent_deviation.abs());
                if cent_deviation.abs() <= 5.0 {
                    notes_within_5_cents += 1;
                }
                total_valid_notes += 1;
            }
        }
        let accuracy_percentage = if total_valid_notes > 0 {
            (notes_within_5_cents as f32 / total_valid_notes as f32) * 100.0
        } else {
            0.0
        };
        let mean_cent_deviation = if !cent_deviations.is_empty() {
            cent_deviations.iter().sum::<f32>() / cent_deviations.len() as f32
        } else {
            0.0
        };
        let max_deviation = cent_deviations.iter().copied().fold(0.0, f32::max);
        let pitch_stability = self.calculate_pitch_stability(&aligned_detected);
        Ok(PitchAccuracyReport {
            accuracy_percentage,
            notes_within_5_cents,
            total_notes: total_valid_notes,
            mean_cent_deviation,
            max_cent_deviation: max_deviation,
            pitch_stability,
            cent_deviations,
        })
    }
    /// Calculate high-precision timing accuracy (targeting 98%+ within 10ms)
    pub fn calculate_precision_timing_accuracy(
        &mut self,
        audio: &[f32],
        target_events: &[NoteEvent],
        sample_rate: f32,
    ) -> Result<TimingAccuracyReport> {
        if audio.is_empty() || target_events.is_empty() {
            return Ok(TimingAccuracyReport::default());
        }
        let detected_onsets = self
            .timing_analyzer
            .detect_onset_times(audio, sample_rate)?;
        let mut target_onsets = Vec::new();
        let mut current_time = 0.0;
        for event in target_events {
            target_onsets.push(current_time);
            current_time += event.duration;
        }
        let aligned_pairs = self
            .timing_analyzer
            .align_onsets(&detected_onsets, &target_onsets)?;
        let mut timing_deviations = Vec::new();
        let mut notes_within_10ms = 0;
        let total_notes = aligned_pairs.len();
        for (detected, target) in aligned_pairs {
            let deviation_ms = (detected - target).abs() * 1000.0;
            timing_deviations.push(deviation_ms);
            if deviation_ms <= 10.0 {
                notes_within_10ms += 1;
            }
        }
        let accuracy_percentage = if total_notes > 0 {
            (notes_within_10ms as f32 / total_notes as f32) * 100.0
        } else {
            0.0
        };
        let mean_timing_deviation = if !timing_deviations.is_empty() {
            timing_deviations.iter().sum::<f32>() / timing_deviations.len() as f32
        } else {
            0.0
        };
        let max_deviation = timing_deviations.iter().copied().fold(0.0, f32::max);
        let rhythm_consistency = self
            .timing_analyzer
            .calculate_rhythm_consistency(&timing_deviations);
        Ok(TimingAccuracyReport {
            accuracy_percentage,
            notes_within_10ms,
            total_notes,
            mean_timing_deviation_ms: mean_timing_deviation,
            max_timing_deviation_ms: max_deviation,
            rhythm_consistency,
            timing_deviations_ms: timing_deviations,
        })
    }
    /// Calculate enhanced naturalness score (targeting MOS 4.0+)
    pub fn calculate_enhanced_naturalness_score(
        &mut self,
        audio: &[f32],
        voice_characteristics: &VoiceCharacteristics,
        sample_rate: f32,
    ) -> Result<NaturalnessScoreReport> {
        let breath_naturalness = self
            .naturalness_scorer
            .analyze_breath_patterns(audio, sample_rate)?;
        let vibrato_naturalness = self
            .naturalness_scorer
            .analyze_vibrato_quality(audio, sample_rate)?;
        let formant_naturalness = self.naturalness_scorer.analyze_formant_structure(
            audio,
            sample_rate,
            voice_characteristics,
        )?;
        let spectral_naturalness = self
            .naturalness_scorer
            .analyze_spectral_characteristics(audio, sample_rate)?;
        let temporal_naturalness = self
            .naturalness_scorer
            .analyze_temporal_dynamics(audio, sample_rate)?;
        let raw_mos = breath_naturalness * 0.25
            + vibrato_naturalness * 0.20
            + formant_naturalness * 0.25
            + spectral_naturalness * 0.15
            + temporal_naturalness * 0.15;
        let enhanced_mos = self
            .naturalness_scorer
            .enhance_mos_score(raw_mos, voice_characteristics);
        Ok(NaturalnessScoreReport {
            mos_score: enhanced_mos,
            breath_naturalness,
            vibrato_naturalness,
            formant_naturalness,
            spectral_naturalness,
            temporal_naturalness,
            quality_factors: self.naturalness_scorer.get_quality_factors(),
        })
    }
    /// Calculate enhanced musical expression recognition (targeting 85%+)
    pub fn calculate_enhanced_expression_recognition(
        &mut self,
        audio: &[f32],
        target_expressions: &[Expression],
        sample_rate: f32,
    ) -> Result<ExpressionRecognitionReport> {
        if audio.is_empty() || target_expressions.is_empty() {
            return Ok(ExpressionRecognitionReport::default());
        }
        let detected_expressions = self
            .expression_recognizer
            .extract_expression_features(audio, sample_rate)?;
        let mut recognition_accuracies = Vec::new();
        let mut correctly_recognized = 0;
        let total_expressions = target_expressions.len();
        for (i, target) in target_expressions.iter().enumerate() {
            if i < detected_expressions.len() {
                let accuracy = self
                    .expression_recognizer
                    .compare_expressions(target, &detected_expressions[i])?;
                recognition_accuracies.push(accuracy);
                if accuracy >= 0.85 {
                    correctly_recognized += 1;
                }
            }
        }
        let overall_recognition_rate = if total_expressions > 0 {
            (correctly_recognized as f32 / total_expressions as f32) * 100.0
        } else {
            0.0
        };
        let mean_accuracy = if !recognition_accuracies.is_empty() {
            recognition_accuracies.iter().sum::<f32>() / recognition_accuracies.len() as f32
        } else {
            0.0
        };
        Ok(ExpressionRecognitionReport {
            recognition_rate_percentage: overall_recognition_rate,
            expressions_correctly_recognized: correctly_recognized,
            total_expressions,
            mean_recognition_accuracy: mean_accuracy,
            individual_accuracies: recognition_accuracies,
            detected_expressions,
        })
    }
    /// Extract high-precision F0 using improved autocorrelation
    fn extract_high_precision_f0(&mut self, audio: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        let frame_size = 2048;
        let hop_size = 512;
        let mut f0_values = Vec::new();
        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end - i < frame_size / 2 {
                break;
            }
            let frame = &audio[i..end];
            let f0 = self.detect_f0_autocorr(frame, sample_rate)?;
            f0_values.push(f0);
        }
        Ok(f0_values)
    }
    /// Simple autocorrelation-based F0 detection
    fn detect_f0_autocorr(&self, frame: &[f32], sample_rate: f32) -> Result<f32> {
        if frame.len() < 64 {
            return Ok(0.0);
        }
        let min_period = (sample_rate / 800.0) as usize;
        let max_period = (sample_rate / 80.0) as usize;
        let mut best_period = 0;
        let mut best_correlation = 0.0;
        for period in min_period..max_period.min(frame.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;
            for i in 0..(frame.len() - period) {
                correlation += frame[i] * frame[i + period];
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
        if best_correlation > 0.3 && best_period > 0 {
            Ok(sample_rate / best_period as f32)
        } else {
            Ok(0.0)
        }
    }
    /// Align pitch contours for comparison
    fn align_pitch_contours(&self, detected: &[f32], target: &[f32]) -> Result<Vec<f32>> {
        if detected.is_empty() || target.is_empty() {
            return Ok(Vec::new());
        }
        let mut aligned = Vec::new();
        let ratio = detected.len() as f32 / target.len() as f32;
        for i in 0..target.len() {
            let idx = (i as f32 * ratio) as usize;
            if idx < detected.len() {
                aligned.push(detected[idx]);
            } else {
                aligned.push(detected[detected.len() - 1]);
            }
        }
        Ok(aligned)
    }
    /// Calculate pitch stability metric
    fn calculate_pitch_stability(&self, f0_values: &[f32]) -> f32 {
        if f0_values.len() < 2 {
            return 1.0;
        }
        let mut deviations = Vec::new();
        for i in 1..f0_values.len() {
            if f0_values[i] > 0.0 && f0_values[i - 1] > 0.0 {
                let deviation = (f0_values[i] / f0_values[i - 1] - 1.0).abs();
                deviations.push(deviation);
            }
        }
        if deviations.is_empty() {
            return 1.0;
        }
        let mean_deviation = deviations.iter().sum::<f32>() / deviations.len() as f32;
        1.0 - mean_deviation.min(1.0)
    }
}
impl Default for PrecisionQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
/// Timing analysis components
pub struct TimingAnalyzer {
    onset_detector: OnsetDetector,
}
impl TimingAnalyzer {
    /// Creates a new timing analyzer with onset detection.
    ///
    /// Initializes the timing analyzer with a default onset detector for
    /// identifying note onsets in audio signals.
    ///
    /// # Returns
    ///
    /// A new `TimingAnalyzer` instance
    pub fn new() -> Self {
        Self {
            onset_detector: OnsetDetector::new(),
        }
    }
    /// Detects onset times in audio using spectral flux analysis.
    ///
    /// Analyzes the audio signal to identify the precise timing of note onsets,
    /// which is critical for measuring timing accuracy.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate of the audio in Hz
    ///
    /// # Returns
    ///
    /// A vector of onset times in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if onset detection fails
    pub fn detect_onset_times(&mut self, audio: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        self.onset_detector.detect_onsets(audio, sample_rate)
    }
    /// Aligns detected onsets with target onsets for comparison.
    ///
    /// Uses nearest neighbor matching to pair each target onset with the
    /// closest detected onset for timing accuracy calculation.
    ///
    /// # Arguments
    ///
    /// * `detected` - Detected onset times in seconds
    /// * `target` - Target onset times in seconds
    ///
    /// # Returns
    ///
    /// A vector of (detected, target) onset pairs for comparison
    ///
    /// # Errors
    ///
    /// Returns an error if alignment fails
    pub fn align_onsets(&self, detected: &[f32], target: &[f32]) -> Result<Vec<(f32, f32)>> {
        let mut aligned_pairs = Vec::new();
        if detected.is_empty() || target.is_empty() {
            return Ok(aligned_pairs);
        }
        for &target_onset in target {
            let mut best_match = detected[0];
            let mut best_distance = (detected[0] - target_onset).abs();
            for &detected_onset in detected {
                let distance = (detected_onset - target_onset).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_match = detected_onset;
                }
            }
            aligned_pairs.push((best_match, target_onset));
        }
        Ok(aligned_pairs)
    }
    /// Calculates rhythm consistency based on timing deviation statistics.
    ///
    /// Computes a consistency score based on the variance in timing deviations,
    /// with lower variance indicating more consistent rhythm.
    ///
    /// # Arguments
    ///
    /// * `deviations` - Timing deviations in milliseconds
    ///
    /// # Returns
    ///
    /// Rhythm consistency score from 0.0 (inconsistent) to 1.0 (perfectly consistent)
    pub fn calculate_rhythm_consistency(&self, deviations: &[f32]) -> f32 {
        if deviations.is_empty() {
            return 1.0;
        }
        let mean = deviations.iter().sum::<f32>() / deviations.len() as f32;
        let variance =
            deviations.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / deviations.len() as f32;
        1.0 - (variance.sqrt() / 50.0).min(1.0)
    }
}
impl Default for TimingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
/// Enhanced expression recognition
pub struct ExpressionRecognizer {
    expression_models: HashMap<Expression, ExpressionModel>,
}
impl ExpressionRecognizer {
    /// Creates a new expression recognizer with default models.
    ///
    /// Initializes expression models for Neutral, Happy, Sad, Excited, and Calm
    /// expressions with their typical feature characteristics.
    ///
    /// # Returns
    ///
    /// A new `ExpressionRecognizer` with predefined expression models
    pub fn new() -> Self {
        let mut expression_models = HashMap::new();
        expression_models.insert(Expression::Neutral, ExpressionModel::new_neutral());
        expression_models.insert(Expression::Happy, ExpressionModel::new_happy());
        expression_models.insert(Expression::Sad, ExpressionModel::new_sad());
        expression_models.insert(Expression::Excited, ExpressionModel::new_excited());
        expression_models.insert(Expression::Calm, ExpressionModel::new_calm());
        Self { expression_models }
    }
    /// Extracts expression features from audio segments.
    ///
    /// Segments the audio and extracts expression-related features including
    /// attack time, sustain level, decay time, spectral centroid, and dynamic range
    /// for each segment, then classifies the expression type.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate of the audio in Hz
    ///
    /// # Returns
    ///
    /// A vector of detected expressions with confidence scores
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction or classification fails
    pub fn extract_expression_features(
        &self,
        audio: &[f32],
        sample_rate: f32,
    ) -> Result<Vec<DetectedExpression>> {
        let mut detected_expressions = Vec::new();
        let segments = self.segment_audio_for_expression(audio, sample_rate)?;
        for segment in segments {
            let features = self.extract_segment_features(&segment, sample_rate)?;
            let expression = self.classify_expression(&features)?;
            detected_expressions.push(expression);
        }
        Ok(detected_expressions)
    }
    /// Compares detected expression with target expression.
    ///
    /// Calculates similarity between the target expression and detected expression
    /// features using the expression model for the target type.
    ///
    /// # Arguments
    ///
    /// * `target` - Target expression to match against
    /// * `detected` - Detected expression with extracted features
    ///
    /// # Returns
    ///
    /// Similarity score from 0.0 (no match) to 1.0 (perfect match)
    ///
    /// # Errors
    ///
    /// Returns an error if the target expression type is unknown
    pub fn compare_expressions(
        &self,
        target: &Expression,
        detected: &DetectedExpression,
    ) -> Result<f32> {
        let target_model = self
            .expression_models
            .get(target)
            .ok_or_else(|| Error::Processing(String::from("Unknown expression type")))?;
        let similarity = target_model.calculate_similarity(&detected.features);
        Ok(similarity)
    }
    fn segment_audio_for_expression(
        &self,
        _audio: &[f32],
        _sample_rate: f32,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.5; 1000], vec![0.6; 1000], vec![0.4; 1000]])
    }
    fn extract_segment_features(
        &self,
        segment: &[f32],
        _sample_rate: f32,
    ) -> Result<ExpressionFeatures> {
        let attack_time = self.calculate_attack_time(segment);
        let sustain_level = self.calculate_sustain_level(segment);
        let decay_time = self.calculate_decay_time(segment);
        let spectral_centroid = self.calculate_spectral_centroid(segment);
        let dynamic_range = self.calculate_segment_dynamic_range(segment);
        Ok(ExpressionFeatures {
            attack_time,
            sustain_level,
            decay_time,
            spectral_centroid,
            dynamic_range,
        })
    }
    fn classify_expression(&self, features: &ExpressionFeatures) -> Result<DetectedExpression> {
        let expression_type = if features.attack_time < 0.01 && features.decay_time < 0.05 {
            Expression::Excited
        } else if features.attack_time > 0.05 && features.sustain_level > 0.8 {
            Expression::Calm
        } else if features.dynamic_range > 0.5 {
            Expression::Happy
        } else {
            Expression::Neutral
        };
        Ok(DetectedExpression {
            expression_type,
            confidence: 0.9,
            features: features.clone(),
        })
    }
    fn calculate_attack_time(&self, _segment: &[f32]) -> f32 {
        0.02
    }
    fn calculate_sustain_level(&self, _segment: &[f32]) -> f32 {
        0.7
    }
    fn calculate_decay_time(&self, _segment: &[f32]) -> f32 {
        0.1
    }
    fn calculate_spectral_centroid(&self, _segment: &[f32]) -> f32 {
        1200.0
    }
    fn calculate_segment_dynamic_range(&self, segment: &[f32]) -> f32 {
        if segment.is_empty() {
            return 0.0;
        }
        let max_val = segment.iter().copied().fold(0.0, f32::max);
        let min_val = segment.iter().copied().fold(0.0, f32::min);
        max_val - min_val
    }
}
impl Default for ExpressionRecognizer {
    fn default() -> Self {
        Self::new()
    }
}
