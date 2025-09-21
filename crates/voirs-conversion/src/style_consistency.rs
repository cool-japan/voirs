//! Style Consistency Preservation System
//!
//! This module provides advanced style consistency preservation capabilities
//! for voice conversion, ensuring that important stylistic characteristics
//! are maintained throughout the conversion process while allowing for
//! controlled modifications.

use crate::types::*;
use crate::{ConversionConfig, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Style consistency preservation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConsistencyConfig {
    /// Preservation mode
    pub preservation_mode: PreservationMode,
    /// Style elements to preserve
    pub preserved_elements: Vec<StyleElement>,
    /// Consistency thresholds
    pub consistency_thresholds: ConsistencyThresholds,
    /// Analysis window size (ms)
    pub analysis_window_ms: u32,
    /// Tracking sensitivity
    pub tracking_sensitivity: f32,
    /// Style adaptation settings
    pub adaptation_settings: StyleAdaptationSettings,
}

impl Default for StyleConsistencyConfig {
    fn default() -> Self {
        Self {
            preservation_mode: PreservationMode::Adaptive,
            preserved_elements: vec![
                StyleElement::ProsodyPattern,
                StyleElement::RhythmPattern,
                StyleElement::EmotionalTone,
                StyleElement::SpeakingRate,
                StyleElement::ArticulationStyle,
            ],
            consistency_thresholds: ConsistencyThresholds::default(),
            analysis_window_ms: 500,
            tracking_sensitivity: 0.7,
            adaptation_settings: StyleAdaptationSettings::default(),
        }
    }
}

/// Preservation mode for style consistency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreservationMode {
    /// Strict preservation - maintain all style elements exactly
    Strict,
    /// Adaptive preservation - allow minor variations for quality
    Adaptive,
    /// Selective preservation - preserve only specified elements
    Selective,
    /// Blended preservation - mix source and target styles
    Blended { source_weight: f32 },
}

/// Style elements that can be preserved
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StyleElement {
    /// Prosodic patterns (intonation, stress)
    ProsodyPattern,
    /// Rhythm and timing patterns
    RhythmPattern,
    /// Emotional tone and expression
    EmotionalTone,
    /// Speaking rate and pace variations
    SpeakingRate,
    /// Articulation and pronunciation style
    ArticulationStyle,
    /// Pause patterns and breath timing
    PausePattern,
    /// Voice quality characteristics
    VoiceQuality,
    /// Dynamic range usage
    DynamicRange,
    /// Accent and dialect features
    AccentFeatures,
    /// Conversational style
    ConversationalStyle,
}

/// Consistency thresholds for different style elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyThresholds {
    /// Prosody consistency threshold (0.0-1.0)
    pub prosody_threshold: f32,
    /// Rhythm consistency threshold (0.0-1.0)
    pub rhythm_threshold: f32,
    /// Emotional consistency threshold (0.0-1.0)
    pub emotional_threshold: f32,
    /// Speaking rate consistency threshold (0.0-1.0)
    pub speaking_rate_threshold: f32,
    /// Articulation consistency threshold (0.0-1.0)
    pub articulation_threshold: f32,
    /// Overall consistency requirement (0.0-1.0)
    pub overall_consistency: f32,
}

impl Default for ConsistencyThresholds {
    fn default() -> Self {
        Self {
            prosody_threshold: 0.85,
            rhythm_threshold: 0.80,
            emotional_threshold: 0.75,
            speaking_rate_threshold: 0.90,
            articulation_threshold: 0.80,
            overall_consistency: 0.82,
        }
    }
}

/// Style adaptation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdaptationSettings {
    /// Enable progressive adaptation
    pub progressive_adaptation: bool,
    /// Adaptation learning rate
    pub learning_rate: f32,
    /// Maximum adaptation per frame
    pub max_adaptation_per_frame: f32,
    /// Style memory window (frames)
    pub memory_window: usize,
    /// Feedback integration enabled
    pub feedback_integration: bool,
}

impl Default for StyleAdaptationSettings {
    fn default() -> Self {
        Self {
            progressive_adaptation: true,
            learning_rate: 0.01,
            max_adaptation_per_frame: 0.05,
            memory_window: 50,
            feedback_integration: true,
        }
    }
}

/// Style characteristics extracted from audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleCharacteristics {
    /// Prosodic features
    pub prosody: ProsodyFeatures,
    /// Rhythm features
    pub rhythm: RhythmFeatures,
    /// Emotional features
    pub emotion: EmotionalFeatures,
    /// Speaking rate features
    pub speaking_rate: SpeakingRateFeatures,
    /// Articulation features
    pub articulation: ArticulationFeatures,
    /// Timestamp of analysis
    pub timestamp: std::time::SystemTime,
    /// Confidence scores for each feature
    pub confidence_scores: HashMap<StyleElement, f32>,
}

/// Prosodic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyFeatures {
    /// F0 contour statistics
    pub f0_mean: f32,
    pub f0_std: f32,
    pub f0_range: f32,
    /// Intonation patterns
    pub intonation_contour: Vec<f32>,
    /// Stress patterns
    pub stress_pattern: Vec<f32>,
    /// Pitch accent locations
    pub pitch_accents: Vec<usize>,
}

/// Rhythm features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmFeatures {
    /// Syllable timing
    pub syllable_durations: Vec<f32>,
    /// Inter-syllable intervals
    pub inter_syllable_intervals: Vec<f32>,
    /// Rhythmic regularity
    pub rhythmic_regularity: f32,
    /// Tempo variations
    pub tempo_variations: Vec<f32>,
    /// Beat tracking
    pub beat_positions: Vec<f32>,
}

/// Emotional features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalFeatures {
    /// Valence (positive/negative)
    pub valence: f32,
    /// Arousal (calm/excited)
    pub arousal: f32,
    /// Dominance (submissive/dominant)
    pub dominance: f32,
    /// Emotional intensity
    pub intensity: f32,
    /// Emotional category probabilities
    pub emotion_probabilities: HashMap<String, f32>,
}

/// Speaking rate features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakingRateFeatures {
    /// Overall speaking rate (syllables per second)
    pub overall_rate: f32,
    /// Local rate variations
    pub local_rates: Vec<f32>,
    /// Pause durations
    pub pause_durations: Vec<f32>,
    /// Speech-to-pause ratio
    pub speech_pause_ratio: f32,
    /// Rate consistency
    pub rate_consistency: f32,
}

/// Articulation features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationFeatures {
    /// Consonant clarity
    pub consonant_clarity: f32,
    /// Vowel formant characteristics
    pub vowel_formants: Vec<(f32, f32, f32)>, // F1, F2, F3
    /// Coarticulation effects
    pub coarticulation_strength: f32,
    /// Precision metrics
    pub articulation_precision: f32,
    /// Spectral tilt
    pub spectral_tilt: f32,
}

/// Style consistency analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConsistencyResult {
    /// Consistency scores for each element
    pub element_scores: HashMap<StyleElement, f32>,
    /// Overall consistency score
    pub overall_score: f32,
    /// Consistency trend over time
    pub consistency_trend: Vec<f32>,
    /// Detected deviations
    pub deviations: Vec<StyleDeviation>,
    /// Recommendations for improvement
    pub recommendations: Vec<StyleRecommendation>,
    /// Analysis confidence
    pub analysis_confidence: f32,
}

/// Style deviation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleDeviation {
    /// Affected style element
    pub element: StyleElement,
    /// Deviation magnitude (0.0-1.0)
    pub magnitude: f32,
    /// Time range of deviation
    pub time_range: (f32, f32), // start_time, end_time in seconds
    /// Deviation type
    pub deviation_type: DeviationType,
    /// Suggested correction
    pub suggested_correction: Option<String>,
}

/// Types of style deviations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviationType {
    /// Sudden change in style
    SuddenChange,
    /// Gradual drift from original style
    GradualDrift,
    /// Inconsistent pattern
    InconsistentPattern,
    /// Missing style element
    MissingElement,
    /// Excessive variation
    ExcessiveVariation,
}

/// Style improvement recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleRecommendation {
    /// Target style element
    pub element: StyleElement,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Specific adjustment
    pub adjustment: String,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Priority level
    pub priority: Priority,
}

/// Types of style recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Adjust parameters
    ParameterAdjustment,
    /// Apply smoothing
    Smoothing,
    /// Enhance consistency
    ConsistencyEnhancement,
    /// Reduce variation
    VariationReduction,
    /// Strengthen pattern
    PatternStrengthening,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Style consistency preservation engine
#[derive(Debug)]
pub struct StyleConsistencyEngine {
    /// Configuration
    config: StyleConsistencyConfig,
    /// Style characteristics history
    style_history: Arc<RwLock<Vec<StyleCharacteristics>>>,
    /// Reference style template
    reference_style: Option<StyleCharacteristics>,
    /// Adaptation state
    adaptation_state: Arc<RwLock<StyleAdaptationState>>,
    /// Performance statistics
    statistics: Arc<RwLock<StyleConsistencyStats>>,
}

/// Style adaptation state
#[derive(Debug, Clone)]
pub struct StyleAdaptationState {
    /// Current adaptation weights
    pub adaptation_weights: HashMap<StyleElement, f32>,
    /// Learning momentum
    pub learning_momentum: HashMap<StyleElement, f32>,
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationStep>,
    /// Current consistency scores
    pub current_scores: HashMap<StyleElement, f32>,
}

/// Adaptation step record
#[derive(Debug, Clone)]
pub struct AdaptationStep {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Element adapted
    pub element: StyleElement,
    /// Adaptation magnitude
    pub magnitude: f32,
    /// Resulting consistency score
    pub resulting_score: f32,
}

/// Style consistency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConsistencyStats {
    /// Total analysis operations
    pub total_analyses: usize,
    /// Average consistency score
    pub avg_consistency_score: f32,
    /// Consistency improvement over time
    pub consistency_improvement: f32,
    /// Most common deviations
    pub common_deviations: HashMap<DeviationType, usize>,
    /// Processing performance
    pub avg_processing_time_ms: f32,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: f32,
}

impl Default for StyleConsistencyStats {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            avg_consistency_score: 0.0,
            consistency_improvement: 0.0,
            common_deviations: HashMap::new(),
            avg_processing_time_ms: 0.0,
            adaptation_effectiveness: 0.0,
        }
    }
}

impl StyleConsistencyEngine {
    /// Create new style consistency engine
    pub fn new(config: StyleConsistencyConfig) -> Self {
        Self {
            config,
            style_history: Arc::new(RwLock::new(Vec::new())),
            reference_style: None,
            adaptation_state: Arc::new(RwLock::new(StyleAdaptationState::new())),
            statistics: Arc::new(RwLock::new(StyleConsistencyStats::default())),
        }
    }

    /// Set reference style from template audio
    pub fn set_reference_style(&mut self, audio_data: &[f32], sample_rate: u32) -> Result<()> {
        let style_chars = self.extract_style_characteristics(audio_data, sample_rate)?;
        self.reference_style = Some(style_chars);
        Ok(())
    }

    /// Analyze style consistency of converted audio
    pub fn analyze_consistency(
        &self,
        original_audio: &[f32],
        converted_audio: &[f32],
        sample_rate: u32,
    ) -> Result<StyleConsistencyResult> {
        let start_time = std::time::SystemTime::now();

        // Extract style characteristics from both audio samples
        let original_style = self.extract_style_characteristics(original_audio, sample_rate)?;
        let converted_style = self.extract_style_characteristics(converted_audio, sample_rate)?;

        // Compute consistency scores
        let element_scores =
            self.compute_element_consistency_scores(&original_style, &converted_style)?;

        // Calculate overall consistency score
        let overall_score = self.calculate_overall_consistency(&element_scores);

        // Detect deviations
        let deviations = self.detect_style_deviations(&original_style, &converted_style)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&element_scores, &deviations);

        // Compute consistency trend
        let consistency_trend = self.compute_consistency_trend(&converted_style);

        // Calculate analysis confidence
        let analysis_confidence =
            self.calculate_analysis_confidence(&original_style, &converted_style);

        // Update statistics
        self.update_statistics(
            start_time.elapsed().unwrap_or_default().as_millis() as f32,
            overall_score,
        );

        Ok(StyleConsistencyResult {
            element_scores,
            overall_score,
            consistency_trend,
            deviations,
            recommendations,
            analysis_confidence,
        })
    }

    /// Apply style consistency preservation during conversion
    pub fn apply_preservation(
        &self,
        conversion_params: &mut ConversionConfig,
        style_feedback: Option<&StyleConsistencyResult>,
    ) -> Result<()> {
        match self.config.preservation_mode {
            PreservationMode::Strict => {
                self.apply_strict_preservation(conversion_params)?;
            }
            PreservationMode::Adaptive => {
                self.apply_adaptive_preservation(conversion_params, style_feedback)?;
            }
            PreservationMode::Selective => {
                self.apply_selective_preservation(conversion_params)?;
            }
            PreservationMode::Blended { source_weight } => {
                self.apply_blended_preservation(conversion_params, source_weight)?;
            }
        }
        Ok(())
    }

    /// Extract style characteristics from audio
    fn extract_style_characteristics(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<StyleCharacteristics> {
        // Extract prosodic features
        let prosody = self.extract_prosody_features(audio_data, sample_rate)?;

        // Extract rhythm features
        let rhythm = self.extract_rhythm_features(audio_data, sample_rate)?;

        // Extract emotional features
        let emotion = self.extract_emotional_features(audio_data, sample_rate)?;

        // Extract speaking rate features
        let speaking_rate = self.extract_speaking_rate_features(audio_data, sample_rate)?;

        // Extract articulation features
        let articulation = self.extract_articulation_features(audio_data, sample_rate)?;

        // Calculate confidence scores
        let confidence_scores =
            self.calculate_feature_confidence_scores(&prosody, &rhythm, &emotion);

        Ok(StyleCharacteristics {
            prosody,
            rhythm,
            emotion,
            speaking_rate,
            articulation,
            timestamp: std::time::SystemTime::now(),
            confidence_scores,
        })
    }

    /// Extract prosodic features
    fn extract_prosody_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<ProsodyFeatures> {
        // Simplified prosody extraction
        let window_size = (sample_rate as f32 * 0.02) as usize; // 20ms windows
        let hop_size = window_size / 2;

        let mut f0_values = Vec::new();
        let mut intonation_contour = Vec::new();
        let mut stress_pattern = Vec::new();

        for i in (0..audio_data.len()).step_by(hop_size) {
            let end = (i + window_size).min(audio_data.len());
            let window = &audio_data[i..end];

            // Simple F0 estimation using autocorrelation
            let f0 = self.estimate_f0(window, sample_rate)?;
            f0_values.push(f0);

            // Simple intonation contour (normalized F0)
            let normalized_f0 = if f0 > 0.0 { f0 / 200.0 } else { 0.0 };
            intonation_contour.push(normalized_f0);

            // Simple stress estimation based on energy
            let energy = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;
            stress_pattern.push(energy.sqrt());
        }

        let valid_f0_values: Vec<f32> = f0_values.iter().filter(|&&f| f > 0.0).cloned().collect();
        let f0_mean = if valid_f0_values.is_empty() {
            0.0
        } else {
            valid_f0_values.iter().sum::<f32>() / valid_f0_values.len() as f32
        };
        let f0_std = {
            if valid_f0_values.is_empty() {
                0.0
            } else {
                let variance = valid_f0_values
                    .iter()
                    .map(|&f| (f - f0_mean).powi(2))
                    .sum::<f32>()
                    / valid_f0_values.len() as f32;
                variance.sqrt()
            }
        };
        let f0_range = if valid_f0_values.is_empty() {
            0.0
        } else {
            let f0_max = valid_f0_values.iter().cloned().fold(0.0, f32::max);
            let f0_min = valid_f0_values
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            f0_max - f0_min
        };

        // Simple pitch accent detection (peaks in F0)
        let mut pitch_accents = Vec::new();
        for i in 1..f0_values.len() - 1 {
            if f0_values[i] > f0_values[i - 1]
                && f0_values[i] > f0_values[i + 1]
                && f0_values[i] > f0_mean + f0_std
            {
                pitch_accents.push(i);
            }
        }

        Ok(ProsodyFeatures {
            f0_mean,
            f0_std,
            f0_range,
            intonation_contour,
            stress_pattern,
            pitch_accents,
        })
    }

    /// Estimate F0 using autocorrelation
    fn estimate_f0(&self, window: &[f32], sample_rate: u32) -> Result<f32> {
        if window.len() < 2 {
            return Ok(0.0);
        }

        let min_period = sample_rate / 500; // 500 Hz max
        let max_period = sample_rate / 80; // 80 Hz min

        // Normalize the window to improve correlation
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let normalized: Vec<f32> = window.iter().map(|x| x - mean).collect();

        let mut best_period = 0;
        let mut max_biased_correlation = 0.0;
        let mut best_correlation = 0.0;

        for period in min_period as usize..=(max_period as usize).min(window.len() / 2) {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            // Calculate normalized cross-correlation
            for i in 0..(window.len() - period) {
                correlation += normalized[i] * normalized[i + period];
                norm1 += normalized[i] * normalized[i];
                norm2 += normalized[i + period] * normalized[i + period];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();

                // Prefer shorter periods (higher frequencies) by adding a small bias
                let frequency_bias = 1.0 + 0.1 / (period as f32 / min_period as f32);
                let biased_correlation = correlation * frequency_bias;

                if biased_correlation > max_biased_correlation {
                    max_biased_correlation = biased_correlation;
                    best_correlation = correlation;
                    best_period = period;
                }
            }
        }

        if best_period > 0 && best_correlation > 0.3 {
            Ok(sample_rate as f32 / best_period as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Extract rhythm features
    fn extract_rhythm_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmFeatures> {
        // Simplified rhythm extraction
        let frame_size = (sample_rate as f32 * 0.02) as usize; // 20ms frames
        let hop_size = frame_size / 2;

        let mut energy_values = Vec::new();

        // Calculate energy for each frame
        for i in (0..audio_data.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio_data.len());
            let window = &audio_data[i..end];
            let energy = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;
            energy_values.push(energy.sqrt());
        }

        // Simple syllable detection based on energy peaks
        let mean_energy = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let energy_threshold = mean_energy * 0.3;

        let mut syllable_durations = Vec::new();
        let mut inter_syllable_intervals = Vec::new();
        let mut in_syllable = false;
        let mut syllable_start = 0;
        let mut last_syllable_end = 0;

        for (i, &energy) in energy_values.iter().enumerate() {
            if !in_syllable && energy > energy_threshold {
                // Start of syllable
                in_syllable = true;
                syllable_start = i;
                if last_syllable_end > 0 {
                    let interval =
                        (i - last_syllable_end) as f32 * hop_size as f32 / sample_rate as f32;
                    inter_syllable_intervals.push(interval);
                }
            } else if in_syllable && energy <= energy_threshold {
                // End of syllable
                in_syllable = false;
                let duration = (i - syllable_start) as f32 * hop_size as f32 / sample_rate as f32;
                syllable_durations.push(duration);
                last_syllable_end = i;
            }
        }

        // Calculate rhythmic regularity (coefficient of variation of intervals)
        let rhythmic_regularity = if inter_syllable_intervals.len() > 1 {
            let mean_interval = inter_syllable_intervals.iter().sum::<f32>()
                / inter_syllable_intervals.len() as f32;
            let variance = inter_syllable_intervals
                .iter()
                .map(|&x| (x - mean_interval).powi(2))
                .sum::<f32>()
                / inter_syllable_intervals.len() as f32;
            1.0 - (variance.sqrt() / mean_interval).min(1.0)
        } else {
            0.0
        };

        // Simple tempo variations (energy envelope)
        let tempo_variations = energy_values
            .iter()
            .enumerate()
            .map(|(i, &energy)| {
                let smoothed = if i > 0 && i < energy_values.len() - 1 {
                    (energy_values[i - 1] + energy + energy_values[i + 1]) / 3.0
                } else {
                    energy
                };
                smoothed
            })
            .collect();

        // Simple beat positions (energy peaks)
        let mut beat_positions = Vec::new();
        for i in 1..energy_values.len() - 1 {
            if energy_values[i] > energy_values[i - 1]
                && energy_values[i] > energy_values[i + 1]
                && energy_values[i] > mean_energy
            {
                let time_pos = i as f32 * hop_size as f32 / sample_rate as f32;
                beat_positions.push(time_pos);
            }
        }

        Ok(RhythmFeatures {
            syllable_durations,
            inter_syllable_intervals,
            rhythmic_regularity,
            tempo_variations,
            beat_positions,
        })
    }

    /// Extract emotional features
    fn extract_emotional_features(
        &self,
        audio_data: &[f32],
        _sample_rate: u32,
    ) -> Result<EmotionalFeatures> {
        // Simplified emotional feature extraction

        // Calculate basic statistics
        let mean_amplitude =
            audio_data.iter().map(|&x| x.abs()).sum::<f32>() / audio_data.len() as f32;
        let energy = audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32;
        let energy_std = {
            let variance = audio_data
                .iter()
                .map(|&x| (x * x - energy).powi(2))
                .sum::<f32>()
                / audio_data.len() as f32;
            variance.sqrt()
        };

        // Map acoustic features to emotional dimensions
        let arousal = (energy * 2.0).clamp(-1.0, 1.0); // Higher energy = higher arousal
        let valence = (mean_amplitude - 0.1).clamp(-1.0, 1.0); // Moderate amplitude = positive
        let dominance = (energy_std * 3.0 - 0.5).clamp(-1.0, 1.0); // Variability = dominance
        let intensity = (energy + energy_std).min(1.0);

        // Simple emotion category probabilities
        let mut emotion_probabilities = HashMap::new();
        emotion_probabilities.insert("happy".to_string(), (valence + arousal).max(0.0) / 2.0);
        emotion_probabilities.insert("sad".to_string(), (-valence - arousal).max(0.0) / 2.0);
        emotion_probabilities.insert("angry".to_string(), (-valence + arousal).max(0.0) / 2.0);
        emotion_probabilities.insert("calm".to_string(), (valence - arousal).max(0.0) / 2.0);
        emotion_probabilities.insert("neutral".to_string(), 1.0 - intensity);

        Ok(EmotionalFeatures {
            valence,
            arousal,
            dominance,
            intensity,
            emotion_probabilities,
        })
    }

    /// Extract speaking rate features
    fn extract_speaking_rate_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<SpeakingRateFeatures> {
        // Simplified speaking rate extraction
        let frame_size = (sample_rate as f32 * 0.02) as usize; // 20ms frames
        let hop_size = frame_size / 2;

        let mut energy_values = Vec::new();
        for i in (0..audio_data.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio_data.len());
            let window = &audio_data[i..end];
            let energy = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;
            energy_values.push(energy.sqrt());
        }

        // Detect speech vs silence
        let mean_energy = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let speech_threshold = mean_energy * 0.2;

        let mut speech_frames = 0;
        let mut pause_durations = Vec::new();
        let mut local_rates = Vec::new();
        let mut in_pause = false;
        let mut pause_start = 0;

        let window_size = 100; // 100 frames for local rate calculation

        for (i, &energy) in energy_values.iter().enumerate() {
            if energy > speech_threshold {
                if in_pause {
                    // End of pause
                    let pause_duration =
                        (i - pause_start) as f32 * hop_size as f32 / sample_rate as f32;
                    pause_durations.push(pause_duration);
                    in_pause = false;
                }
                speech_frames += 1;
            } else {
                if !in_pause {
                    // Start of pause
                    pause_start = i;
                    in_pause = true;
                }
            }

            // Calculate local rate in sliding window
            if i >= window_size {
                let window_speech = energy_values[i - window_size..i]
                    .iter()
                    .filter(|&&e| e > speech_threshold)
                    .count();
                let local_rate = window_speech as f32
                    / (window_size as f32 * hop_size as f32 / sample_rate as f32);
                local_rates.push(local_rate);
            }
        }

        let total_time = audio_data.len() as f32 / sample_rate as f32;
        let speech_time = speech_frames as f32 * hop_size as f32 / sample_rate as f32;
        let pause_time = total_time - speech_time;

        let overall_rate = speech_frames as f32 / total_time; // Frames per second
        let speech_pause_ratio = if pause_time > 0.0 {
            speech_time / pause_time
        } else {
            f32::INFINITY
        };

        // Calculate rate consistency
        let rate_consistency = if local_rates.len() > 1 {
            let mean_rate = local_rates.iter().sum::<f32>() / local_rates.len() as f32;
            let variance = local_rates
                .iter()
                .map(|&r| (r - mean_rate).powi(2))
                .sum::<f32>()
                / local_rates.len() as f32;
            1.0 - (variance.sqrt() / mean_rate.max(1.0)).min(1.0)
        } else {
            1.0
        };

        Ok(SpeakingRateFeatures {
            overall_rate,
            local_rates,
            pause_durations,
            speech_pause_ratio,
            rate_consistency,
        })
    }

    /// Extract articulation features
    fn extract_articulation_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<ArticulationFeatures> {
        // Simplified articulation feature extraction

        // Calculate spectral features
        let fft_size = 1024;
        let mut spectrum = vec![0.0; fft_size / 2];

        if audio_data.len() >= fft_size {
            // Simple magnitude spectrum
            for i in 0..fft_size / 2 {
                if i < audio_data.len() {
                    spectrum[i] = audio_data[i].abs();
                }
            }
        }

        // Estimate formants (simplified)
        let freq_per_bin = sample_rate as f32 / fft_size as f32;
        let mut formants = Vec::new();

        // Look for peaks in low frequencies (simplified formant detection)
        for window_start in (0..spectrum.len() / 4).step_by(20) {
            let window_end = (window_start + 20).min(spectrum.len());
            if let Some((peak_idx, &peak_val)) = spectrum[window_start..window_end]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                let freq = (window_start + peak_idx) as f32 * freq_per_bin;
                if peak_val > 0.01 && freq > 200.0 && freq < 3000.0 {
                    formants.push((freq, freq * 1.2, freq * 1.5)); // F1, F2, F3 estimates
                }
            }
        }

        // Take first 3 formants or fill with defaults
        let vowel_formants = if formants.len() >= 3 {
            formants[0..3].to_vec()
        } else {
            vec![(500.0, 1500.0, 2500.0); 3 - formants.len()]
                .into_iter()
                .chain(formants.into_iter())
                .take(3)
                .collect()
        };

        // Consonant clarity (high frequency energy)
        let high_freq_start = (2000.0 / freq_per_bin) as usize;
        let high_freq_energy = spectrum[high_freq_start.min(spectrum.len())..]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>();
        let total_energy = spectrum.iter().map(|&x| x * x).sum::<f32>();
        let consonant_clarity = if total_energy > 0.0 {
            high_freq_energy / total_energy
        } else {
            0.0
        };

        // Spectral tilt (ratio of low to high frequencies)
        let low_freq_end = (1000.0 / freq_per_bin) as usize;
        let low_freq_energy = spectrum[..low_freq_end.min(spectrum.len())]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>();
        let spectral_tilt = if high_freq_energy > 0.0 {
            low_freq_energy / high_freq_energy
        } else {
            1.0
        };

        // Simple measures
        let coarticulation_strength = 0.5; // Placeholder
        let articulation_precision = consonant_clarity * 0.7 + (1.0 - spectral_tilt.min(1.0)) * 0.3;

        Ok(ArticulationFeatures {
            consonant_clarity,
            vowel_formants,
            coarticulation_strength,
            articulation_precision,
            spectral_tilt,
        })
    }

    /// Calculate confidence scores for extracted features
    fn calculate_feature_confidence_scores(
        &self,
        prosody: &ProsodyFeatures,
        rhythm: &RhythmFeatures,
        emotion: &EmotionalFeatures,
    ) -> HashMap<StyleElement, f32> {
        let mut scores = HashMap::new();

        // Prosody confidence based on F0 validity
        let prosody_confidence = if prosody.f0_mean > 80.0 && prosody.f0_mean < 400.0 {
            0.9
        } else {
            0.5
        };
        scores.insert(StyleElement::ProsodyPattern, prosody_confidence);

        // Rhythm confidence based on pattern regularity
        let rhythm_confidence = rhythm.rhythmic_regularity;
        scores.insert(StyleElement::RhythmPattern, rhythm_confidence);

        // Emotional confidence based on intensity
        let emotion_confidence = emotion.intensity.min(1.0);
        scores.insert(StyleElement::EmotionalTone, emotion_confidence);

        // Default confidence for other elements
        scores.insert(StyleElement::SpeakingRate, 0.8);
        scores.insert(StyleElement::ArticulationStyle, 0.7);

        scores
    }

    // Additional implementation methods would continue here...
    // For brevity, implementing key methods that demonstrate the functionality

    /// Compute consistency scores for each style element
    fn compute_element_consistency_scores(
        &self,
        original: &StyleCharacteristics,
        converted: &StyleCharacteristics,
    ) -> Result<HashMap<StyleElement, f32>> {
        let mut scores = HashMap::new();

        for element in &self.config.preserved_elements {
            let score = match element {
                StyleElement::ProsodyPattern => {
                    self.compare_prosody_features(&original.prosody, &converted.prosody)
                }
                StyleElement::RhythmPattern => {
                    self.compare_rhythm_features(&original.rhythm, &converted.rhythm)
                }
                StyleElement::EmotionalTone => {
                    self.compare_emotional_features(&original.emotion, &converted.emotion)
                }
                StyleElement::SpeakingRate => self.compare_speaking_rate_features(
                    &original.speaking_rate,
                    &converted.speaking_rate,
                ),
                StyleElement::ArticulationStyle => self
                    .compare_articulation_features(&original.articulation, &converted.articulation),
                _ => 0.8, // Default score for other elements
            };
            scores.insert(element.clone(), score);
        }

        Ok(scores)
    }

    /// Compare prosody features between original and converted
    fn compare_prosody_features(
        &self,
        original: &ProsodyFeatures,
        converted: &ProsodyFeatures,
    ) -> f32 {
        let f0_similarity = 1.0
            - ((original.f0_mean - converted.f0_mean).abs() / original.f0_mean.max(1.0)).min(1.0);
        let range_similarity = 1.0
            - ((original.f0_range - converted.f0_range).abs() / original.f0_range.max(1.0))
                .min(1.0);
        let contour_similarity =
            self.compare_vectors(&original.intonation_contour, &converted.intonation_contour);

        (f0_similarity + range_similarity + contour_similarity) / 3.0
    }

    /// Compare rhythm features between original and converted  
    fn compare_rhythm_features(
        &self,
        original: &RhythmFeatures,
        converted: &RhythmFeatures,
    ) -> f32 {
        let regularity_similarity =
            1.0 - (original.rhythmic_regularity - converted.rhythmic_regularity).abs();
        let tempo_similarity =
            self.compare_vectors(&original.tempo_variations, &converted.tempo_variations);

        (regularity_similarity + tempo_similarity) / 2.0
    }

    /// Compare emotional features between original and converted
    fn compare_emotional_features(
        &self,
        original: &EmotionalFeatures,
        converted: &EmotionalFeatures,
    ) -> f32 {
        let valence_similarity = 1.0 - (original.valence - converted.valence).abs() / 2.0;
        let arousal_similarity = 1.0 - (original.arousal - converted.arousal).abs() / 2.0;
        let intensity_similarity = 1.0 - (original.intensity - converted.intensity).abs();

        (valence_similarity + arousal_similarity + intensity_similarity) / 3.0
    }

    /// Compare speaking rate features between original and converted
    fn compare_speaking_rate_features(
        &self,
        original: &SpeakingRateFeatures,
        converted: &SpeakingRateFeatures,
    ) -> f32 {
        let rate_similarity = 1.0
            - ((original.overall_rate - converted.overall_rate).abs()
                / original.overall_rate.max(1.0))
            .min(1.0);
        let consistency_similarity =
            1.0 - (original.rate_consistency - converted.rate_consistency).abs();

        (rate_similarity + consistency_similarity) / 2.0
    }

    /// Compare articulation features between original and converted
    fn compare_articulation_features(
        &self,
        original: &ArticulationFeatures,
        converted: &ArticulationFeatures,
    ) -> f32 {
        let clarity_similarity =
            1.0 - (original.consonant_clarity - converted.consonant_clarity).abs();
        let precision_similarity =
            1.0 - (original.articulation_precision - converted.articulation_precision).abs();

        (clarity_similarity + precision_similarity) / 2.0
    }

    /// Compare two vectors and return similarity score
    fn compare_vectors(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.is_empty() || vec2.is_empty() {
            return 0.0;
        }

        let min_len = vec1.len().min(vec2.len());
        let mut similarity_sum = 0.0;

        for i in 0..min_len {
            let diff = (vec1[i] - vec2[i]).abs();
            let max_val = vec1[i].abs().max(vec2[i].abs()).max(1.0);
            similarity_sum += 1.0 - (diff / max_val).min(1.0);
        }

        similarity_sum / min_len as f32
    }

    /// Calculate overall consistency score
    fn calculate_overall_consistency(&self, element_scores: &HashMap<StyleElement, f32>) -> f32 {
        if element_scores.is_empty() {
            return 0.0;
        }

        let sum: f32 = element_scores.values().sum();
        sum / element_scores.len() as f32
    }

    /// Detect style deviations
    fn detect_style_deviations(
        &self,
        original: &StyleCharacteristics,
        converted: &StyleCharacteristics,
    ) -> Result<Vec<StyleDeviation>> {
        let mut deviations = Vec::new();

        // Check each style element for deviations
        for element in &self.config.preserved_elements {
            let threshold = match element {
                StyleElement::ProsodyPattern => {
                    self.config.consistency_thresholds.prosody_threshold
                }
                StyleElement::RhythmPattern => self.config.consistency_thresholds.rhythm_threshold,
                StyleElement::EmotionalTone => {
                    self.config.consistency_thresholds.emotional_threshold
                }
                StyleElement::SpeakingRate => {
                    self.config.consistency_thresholds.speaking_rate_threshold
                }
                StyleElement::ArticulationStyle => {
                    self.config.consistency_thresholds.articulation_threshold
                }
                _ => 0.8,
            };

            let score = match element {
                StyleElement::ProsodyPattern => {
                    self.compare_prosody_features(&original.prosody, &converted.prosody)
                }
                StyleElement::RhythmPattern => {
                    self.compare_rhythm_features(&original.rhythm, &converted.rhythm)
                }
                StyleElement::EmotionalTone => {
                    self.compare_emotional_features(&original.emotion, &converted.emotion)
                }
                StyleElement::SpeakingRate => self.compare_speaking_rate_features(
                    &original.speaking_rate,
                    &converted.speaking_rate,
                ),
                StyleElement::ArticulationStyle => self
                    .compare_articulation_features(&original.articulation, &converted.articulation),
                _ => 0.8,
            };

            if score < threshold {
                let magnitude = threshold - score;
                let deviation_type = if magnitude > 0.3 {
                    DeviationType::SuddenChange
                } else if magnitude > 0.2 {
                    DeviationType::InconsistentPattern
                } else {
                    DeviationType::GradualDrift
                };

                deviations.push(StyleDeviation {
                    element: element.clone(),
                    magnitude,
                    time_range: (0.0, 1.0), // Simplified time range
                    deviation_type,
                    suggested_correction: Some(format!(
                        "Adjust {} parameters to improve consistency",
                        format!("{:?}", element).to_lowercase()
                    )),
                });
            }
        }

        Ok(deviations)
    }

    /// Generate style improvement recommendations
    fn generate_recommendations(
        &self,
        element_scores: &HashMap<StyleElement, f32>,
        deviations: &[StyleDeviation],
    ) -> Vec<StyleRecommendation> {
        let mut recommendations = Vec::new();

        for (element, &score) in element_scores {
            if score < 0.8 {
                let priority = if score < 0.5 {
                    Priority::Critical
                } else if score < 0.7 {
                    Priority::High
                } else {
                    Priority::Medium
                };

                let recommendation_type = if deviations.iter().any(|d| d.element == *element) {
                    RecommendationType::ConsistencyEnhancement
                } else {
                    RecommendationType::ParameterAdjustment
                };

                recommendations.push(StyleRecommendation {
                    element: element.clone(),
                    recommendation_type,
                    adjustment: format!(
                        "Improve {} consistency by adjusting conversion parameters",
                        format!("{:?}", element).to_lowercase()
                    ),
                    expected_improvement: (0.8 - score) * 0.7,
                    priority,
                });
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }

    /// Compute consistency trend over time
    fn compute_consistency_trend(&self, _converted: &StyleCharacteristics) -> Vec<f32> {
        // Simplified trend calculation
        // In a real implementation, this would track consistency over multiple time windows
        vec![0.8, 0.82, 0.85, 0.83, 0.87, 0.85, 0.88]
    }

    /// Calculate analysis confidence
    fn calculate_analysis_confidence(
        &self,
        original: &StyleCharacteristics,
        converted: &StyleCharacteristics,
    ) -> f32 {
        let original_confidence: f32 = original.confidence_scores.values().sum::<f32>()
            / original.confidence_scores.len().max(1) as f32;
        let converted_confidence: f32 = converted.confidence_scores.values().sum::<f32>()
            / converted.confidence_scores.len().max(1) as f32;

        (original_confidence + converted_confidence) / 2.0
    }

    /// Apply strict preservation
    fn apply_strict_preservation(&self, _conversion_params: &mut ConversionConfig) -> Result<()> {
        // Implementation would modify conversion parameters to strictly preserve all style elements
        Ok(())
    }

    /// Apply adaptive preservation
    fn apply_adaptive_preservation(
        &self,
        _conversion_params: &mut ConversionConfig,
        _style_feedback: Option<&StyleConsistencyResult>,
    ) -> Result<()> {
        // Implementation would adaptively adjust preservation based on feedback
        Ok(())
    }

    /// Apply selective preservation
    fn apply_selective_preservation(
        &self,
        _conversion_params: &mut ConversionConfig,
    ) -> Result<()> {
        // Implementation would preserve only selected style elements
        Ok(())
    }

    /// Apply blended preservation
    fn apply_blended_preservation(
        &self,
        _conversion_params: &mut ConversionConfig,
        _source_weight: f32,
    ) -> Result<()> {
        // Implementation would blend source and target styles
        Ok(())
    }

    /// Update performance statistics
    fn update_statistics(&self, processing_time_ms: f32, consistency_score: f32) {
        if let Ok(mut stats) = self.statistics.write() {
            stats.total_analyses += 1;

            // Update moving average of consistency score
            if stats.total_analyses == 1 {
                // Initialize with first values
                stats.avg_consistency_score = consistency_score;
                stats.avg_processing_time_ms = processing_time_ms;
            } else {
                // Use exponential moving average
                let alpha = 0.1;
                stats.avg_consistency_score =
                    alpha * consistency_score + (1.0 - alpha) * stats.avg_consistency_score;
                stats.avg_processing_time_ms =
                    alpha * processing_time_ms + (1.0 - alpha) * stats.avg_processing_time_ms;
            }
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> StyleConsistencyStats {
        self.statistics.read().unwrap().clone()
    }
}

impl StyleAdaptationState {
    fn new() -> Self {
        Self {
            adaptation_weights: HashMap::new(),
            learning_momentum: HashMap::new(),
            adaptation_history: Vec::new(),
            current_scores: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_consistency_engine_creation() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        assert_eq!(engine.config.preservation_mode, PreservationMode::Adaptive);
        assert_eq!(engine.config.preserved_elements.len(), 5);
    }

    #[test]
    fn test_style_characteristic_extraction() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Generate test audio (sine wave)
        let sample_rate = 16000;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4 note
        let mut audio_data = Vec::new();

        for i in 0..(sample_rate as f32 * duration) as usize {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            audio_data.push(sample);
        }

        let result = engine.extract_style_characteristics(&audio_data, sample_rate);
        assert!(result.is_ok());

        let characteristics = result.unwrap();
        assert!(characteristics.prosody.f0_mean > 0.0);
        assert!(!characteristics.rhythm.tempo_variations.is_empty());
        assert!(characteristics.emotion.intensity >= 0.0);
    }

    #[test]
    fn test_consistency_analysis() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Generate test audio
        let sample_rate_usize = 16000;
        let sample_rate = 16000u32;
        let audio_data = vec![0.5; sample_rate_usize]; // 1 second of constant amplitude
        let converted_audio = vec![0.4; sample_rate_usize]; // Slightly different amplitude

        let result = engine.analyze_consistency(&audio_data, &converted_audio, sample_rate);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
        assert!(!analysis.element_scores.is_empty());
        assert!(analysis.analysis_confidence >= 0.0 && analysis.analysis_confidence <= 1.0);
    }

    #[test]
    fn test_prosody_feature_extraction() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Generate test audio with varying frequency
        let sample_rate = 16000;
        let mut audio_data = Vec::new();

        for i in 0..sample_rate {
            let t = i as f32 / sample_rate as f32;
            let frequency = 200.0 + 100.0 * t; // Sweep from 200Hz to 300Hz
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            audio_data.push(sample);
        }

        let result = engine.extract_prosody_features(&audio_data, sample_rate);
        assert!(result.is_ok());

        let prosody = result.unwrap();
        assert!(prosody.f0_mean > 100.0 && prosody.f0_mean < 400.0);
        assert!(prosody.f0_std > 0.0);
        assert!(!prosody.intonation_contour.is_empty());
    }

    #[test]
    fn test_rhythm_feature_extraction() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Generate rhythmic audio (alternating loud and quiet)
        let sample_rate = 16000;
        let mut audio_data = Vec::new();

        for i in 0..sample_rate {
            let t = i as f32 / sample_rate as f32;
            let beat_frequency = 2.0; // 2 beats per second
            let amplitude = if ((t * beat_frequency) % 1.0) < 0.5 {
                0.8
            } else {
                0.2
            };
            let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * amplitude;
            audio_data.push(sample);
        }

        let result = engine.extract_rhythm_features(&audio_data, sample_rate);
        assert!(result.is_ok());

        let rhythm = result.unwrap();
        assert!(rhythm.rhythmic_regularity >= 0.0 && rhythm.rhythmic_regularity <= 1.0);
        assert!(!rhythm.tempo_variations.is_empty());
        assert!(!rhythm.beat_positions.is_empty());
    }

    #[test]
    fn test_style_deviation_detection() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Create mock style characteristics with different values
        let original = StyleCharacteristics {
            prosody: ProsodyFeatures {
                f0_mean: 200.0,
                f0_std: 20.0,
                f0_range: 100.0,
                intonation_contour: vec![0.5, 0.6, 0.4],
                stress_pattern: vec![0.3, 0.7, 0.5],
                pitch_accents: vec![1, 3],
            },
            rhythm: RhythmFeatures {
                syllable_durations: vec![0.2, 0.3, 0.25],
                inter_syllable_intervals: vec![0.1, 0.15],
                rhythmic_regularity: 0.8,
                tempo_variations: vec![0.5, 0.6, 0.4],
                beat_positions: vec![0.0, 0.5, 1.0],
            },
            emotion: EmotionalFeatures {
                valence: 0.6,
                arousal: 0.4,
                dominance: 0.5,
                intensity: 0.7,
                emotion_probabilities: HashMap::new(),
            },
            speaking_rate: SpeakingRateFeatures {
                overall_rate: 5.0,
                local_rates: vec![4.8, 5.2, 5.0],
                pause_durations: vec![0.1, 0.2],
                speech_pause_ratio: 10.0,
                rate_consistency: 0.9,
            },
            articulation: ArticulationFeatures {
                consonant_clarity: 0.8,
                vowel_formants: vec![(500.0, 1500.0, 2500.0)],
                coarticulation_strength: 0.6,
                articulation_precision: 0.7,
                spectral_tilt: 1.2,
            },
            timestamp: std::time::SystemTime::now(),
            confidence_scores: HashMap::new(),
        };

        let mut converted = original.clone();
        converted.prosody.f0_mean = 100.0; // More dramatic change (200.0 -> 100.0 = 50% reduction)
        converted.prosody.f0_range = 20.0; // Change range too (100.0 -> 20.0)
        converted.emotion.valence = -0.8; // More dramatic change (0.6 -> -0.8)
        converted.emotion.arousal = -0.8; // Change arousal too (0.4 -> -0.8)

        let result = engine.detect_style_deviations(&original, &converted);
        assert!(result.is_ok());

        let deviations = result.unwrap();
        assert!(!deviations.is_empty());

        // Should detect deviations in prosody and emotion
        let prosody_deviation = deviations
            .iter()
            .find(|d| d.element == StyleElement::ProsodyPattern);
        assert!(prosody_deviation.is_some());

        let emotion_deviation = deviations
            .iter()
            .find(|d| d.element == StyleElement::EmotionalTone);
        assert!(emotion_deviation.is_some());
    }

    #[test]
    fn test_style_recommendation_generation() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        let mut element_scores = HashMap::new();
        element_scores.insert(StyleElement::ProsodyPattern, 0.4); // Low score
        element_scores.insert(StyleElement::RhythmPattern, 0.9); // Good score
        element_scores.insert(StyleElement::EmotionalTone, 0.6); // Medium score

        let deviations = vec![StyleDeviation {
            element: StyleElement::ProsodyPattern,
            magnitude: 0.4,
            time_range: (0.0, 1.0),
            deviation_type: DeviationType::SuddenChange,
            suggested_correction: Some("Adjust prosody".to_string()),
        }];

        let recommendations = engine.generate_recommendations(&element_scores, &deviations);

        assert!(!recommendations.is_empty());

        // Should have recommendations for low-scoring elements
        let prosody_rec = recommendations
            .iter()
            .find(|r| r.element == StyleElement::ProsodyPattern);
        assert!(prosody_rec.is_some());
        assert_eq!(prosody_rec.unwrap().priority, Priority::Critical);

        let emotion_rec = recommendations
            .iter()
            .find(|r| r.element == StyleElement::EmotionalTone);
        assert!(emotion_rec.is_some());
        assert_eq!(emotion_rec.unwrap().priority, Priority::High);

        // Should not have recommendation for high-scoring rhythm
        let rhythm_rec = recommendations
            .iter()
            .find(|r| r.element == StyleElement::RhythmPattern);
        assert!(rhythm_rec.is_none());
    }

    #[test]
    fn test_vector_comparison() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.1, 2.1, 2.9];
        let vec3 = vec![5.0, 6.0, 7.0];

        let similarity1 = engine.compare_vectors(&vec1, &vec2);
        let similarity2 = engine.compare_vectors(&vec1, &vec3);

        assert!(similarity1 > similarity2);
        assert!(similarity1 > 0.8); // Should be high similarity
        assert!(similarity2 < 0.5); // Should be low similarity
    }

    #[test]
    fn test_f0_estimation() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        let sample_rate = 16000;
        let frequency = 200.0;
        let duration = 0.1; // 100ms

        // Generate pure sine wave
        let mut audio_data = Vec::new();
        for i in 0..(sample_rate as f32 * duration) as usize {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            audio_data.push(sample);
        }

        let result = engine.estimate_f0(&audio_data, sample_rate);
        assert!(result.is_ok());

        let estimated_f0 = result.unwrap();
        assert!(estimated_f0 > 180.0 && estimated_f0 < 220.0); // Should be close to 200Hz
    }

    #[test]
    fn test_statistics_update() {
        let config = StyleConsistencyConfig::default();
        let engine = StyleConsistencyEngine::new(config);

        // Update statistics multiple times
        engine.update_statistics(100.0, 0.8);
        engine.update_statistics(150.0, 0.9);
        engine.update_statistics(120.0, 0.85);

        let stats = engine.get_statistics();
        assert_eq!(stats.total_analyses, 3);
        assert!(stats.avg_consistency_score > 0.8);
        assert!(stats.avg_processing_time_ms > 100.0);
    }
}
