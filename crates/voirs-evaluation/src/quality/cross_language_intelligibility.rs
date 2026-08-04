//! Cross-language intelligibility assessment framework
//!
//! This module provides comprehensive cross-language intelligibility evaluation including:
//! - Phonetic distance modeling for intelligibility prediction
//! - Perceptual similarity scoring across languages
//! - Cross-linguistic acoustic analysis
//! - Listener adaptation modeling
//! - Intelligibility transfer evaluation

use crate::perceptual::cross_cultural::{
    CrossCulturalAdaptation, CrossCulturalConfig, CrossCulturalPerceptualModel,
};
use crate::perceptual::{CulturalProfile, DemographicProfile};
use crate::quality::universal_phoneme_mapping::{
    MappingType, PhonemeCandidate, PhonemeConverageAnalysis, UniversalPhonemeMapper,
    UniversalPhonemeMappingConfig,
};
use crate::quality::voice_quality_dsp::{
    compute_formant_bandwidths, compute_formant_clarity, compute_jitter_shimmer,
    compute_spectral_tilt, estimate_pitch_period, hnr_from_correlation, VOICING_THRESHOLD,
};
use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Cross-language intelligibility assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageIntelligibilityConfig {
    /// Enable phonetic distance modeling
    pub enable_phonetic_distance: bool,
    /// Enable acoustic similarity analysis
    pub enable_acoustic_similarity: bool,
    /// Enable perceptual adaptation modeling
    pub enable_perceptual_adaptation: bool,
    /// Enable listener proficiency adjustment
    pub enable_listener_proficiency: bool,
    /// Enable context-dependent analysis
    pub enable_context_dependency: bool,
    /// Phonetic distance weight in intelligibility calculation
    pub phonetic_distance_weight: f32,
    /// Acoustic similarity weight in intelligibility calculation
    pub acoustic_similarity_weight: f32,
    /// Perceptual adaptation weight in intelligibility calculation
    pub perceptual_adaptation_weight: f32,
    /// Listener proficiency weight in intelligibility calculation
    pub listener_proficiency_weight: f32,
    /// Context dependency weight in intelligibility calculation
    pub context_dependency_weight: f32,
    /// Minimum intelligibility threshold
    pub min_intelligibility_threshold: f32,
    /// Maximum phonetic distance for meaningful comparison
    pub max_phonetic_distance: f32,
}

impl Default for CrossLanguageIntelligibilityConfig {
    fn default() -> Self {
        Self {
            enable_phonetic_distance: true,
            enable_acoustic_similarity: true,
            enable_perceptual_adaptation: true,
            enable_listener_proficiency: true,
            enable_context_dependency: true,
            phonetic_distance_weight: 0.3,
            acoustic_similarity_weight: 0.25,
            perceptual_adaptation_weight: 0.2,
            listener_proficiency_weight: 0.15,
            context_dependency_weight: 0.1,
            min_intelligibility_threshold: 0.1,
            max_phonetic_distance: 2.0,
        }
    }
}

/// Cross-language intelligibility assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageIntelligibilityResult {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Overall intelligibility score [0.0, 1.0]
    pub overall_intelligibility: f32,
    /// Phonetic distance contribution
    pub phonetic_distance_score: f32,
    /// Acoustic similarity contribution
    pub acoustic_similarity_score: f32,
    /// Perceptual adaptation contribution
    pub perceptual_adaptation_score: f32,
    /// Listener proficiency contribution
    pub listener_proficiency_score: f32,
    /// Context dependency contribution
    pub context_dependency_score: f32,
    /// Word-level intelligibility scores
    pub word_intelligibility: Vec<WordIntelligibilityScore>,
    /// Phoneme-level intelligibility scores
    pub phoneme_intelligibility: Vec<PhonemeIntelligibilityScore>,
    /// Problematic regions
    pub problematic_regions: Vec<ProblematicRegion>,
    /// Intelligibility prediction confidence
    pub prediction_confidence: f32,
    /// Assessment processing time
    pub processing_time: Duration,
}

/// Word-level intelligibility score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordIntelligibilityScore {
    /// Word text
    pub word: String,
    /// Word intelligibility score [0.0, 1.0]
    pub intelligibility_score: f32,
    /// Phonetic complexity
    pub phonetic_complexity: f32,
    /// Acoustic clarity
    pub acoustic_clarity: f32,
    /// Contextual support
    pub contextual_support: f32,
    /// Listener familiarity
    pub listener_familiarity: f32,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// Phoneme-level intelligibility score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeIntelligibilityScore {
    /// Source phoneme
    pub source_phoneme: String,
    /// Perceived phoneme (in target language)
    pub perceived_phoneme: Option<String>,
    /// Intelligibility score [0.0, 1.0]
    pub intelligibility_score: f32,
    /// Phonetic distance from target equivalent
    pub phonetic_distance: f32,
    /// Acoustic similarity
    pub acoustic_similarity: f32,
    /// Perceptual confusability
    pub perceptual_confusability: f32,
    /// Position in utterance
    pub position: usize,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// Problematic region in cross-language intelligibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicRegion {
    /// Region start time
    pub start_time: f32,
    /// Region end time
    pub end_time: f32,
    /// Problem severity [0.0, 1.0]
    pub severity: f32,
    /// Problem type
    pub problem_type: IntelligibilityProblemType,
    /// Problem description
    pub description: String,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Type of intelligibility problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntelligibilityProblemType {
    /// Phonetic mismatch
    PhoneticMismatch,
    /// Acoustic distortion
    AcousticDistortion,
    /// Prosodic interference
    ProsodicInterference,
    /// Lexical unfamiliarity
    LexicalUnfamiliarity,
    /// Cultural adaptation failure
    CulturalAdaptationFailure,
    /// Context dependency issue
    ContextDependencyIssue,
}

/// Listener proficiency profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenerProficiencyProfile {
    /// Target language proficiency level
    pub target_language_proficiency: ProficiencyLevel,
    /// Source language familiarity
    pub source_language_familiarity: ProficiencyLevel,
    /// Accent exposure history
    pub accent_exposure: Vec<AccentExposure>,
    /// Listening experience
    pub listening_experience: ListeningExperience,
    /// Adaptation capability
    pub adaptation_capability: f32,
    /// Attention capacity
    pub attention_capacity: f32,
}

/// Language proficiency level
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProficiencyLevel {
    /// Beginner
    Beginner,
    /// Elementary
    Elementary,
    /// Intermediate
    Intermediate,
    /// Upper Intermediate
    UpperIntermediate,
    /// Advanced
    Advanced,
    /// Native
    Native,
}

/// Accent exposure history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentExposure {
    /// Source accent/language
    pub source_accent: String,
    /// Exposure duration (hours)
    pub exposure_duration: f32,
    /// Exposure recency (days ago)
    pub exposure_recency: f32,
    /// Quality of exposure
    pub exposure_quality: f32,
}

/// Listening experience profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListeningExperience {
    /// Total listening hours
    pub total_listening_hours: f32,
    /// Cross-linguistic listening experience
    pub cross_linguistic_hours: f32,
    /// Synthetic speech familiarity
    pub synthetic_speech_familiarity: f32,
    /// Accent tolerance
    pub accent_tolerance: f32,
    /// Cognitive load capacity
    pub cognitive_load_capacity: f32,
}

/// Context dependency factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDependencyFactors {
    /// Semantic predictability
    pub semantic_predictability: f32,
    /// Syntactic complexity
    pub syntactic_complexity: f32,
    /// Lexical frequency
    pub lexical_frequency: f32,
    /// Discourse coherence
    pub discourse_coherence: f32,
    /// Visual context availability
    pub visual_context_available: bool,
    /// Background noise level
    pub background_noise_level: f32,
}

/// Acoustic analysis for cross-language intelligibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticAnalysisResult {
    /// Formant clarity
    pub formant_clarity: f32,
    /// Spectral envelope similarity
    pub spectral_envelope_similarity: f32,
    /// Temporal envelope preservation
    pub temporal_envelope_preservation: f32,
    /// Voice quality metrics
    pub voice_quality_metrics: VoiceQualityMetrics,
    /// Prosodic feature analysis
    pub prosodic_features: ProsodicFeatureAnalysis,
    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: f32,
}

/// Voice quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Jitter
    pub jitter: f32,
    /// Shimmer
    pub shimmer: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_to_noise_ratio: f32,
    /// Spectral tilt
    pub spectral_tilt: f32,
    /// Formant bandwidth
    pub formant_bandwidth: Vec<f32>,
}

/// Prosodic feature analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicFeatureAnalysis {
    /// F0 contour appropriateness
    pub f0_contour_appropriateness: f32,
    /// Stress pattern accuracy
    pub stress_pattern_accuracy: f32,
    /// Rhythm regularity
    pub rhythm_regularity: f32,
    /// Intonation naturalness
    pub intonation_naturalness: f32,
    /// Pause pattern appropriateness
    pub pause_pattern_appropriateness: f32,
}

/// Cross-language intelligibility evaluator
pub struct CrossLanguageIntelligibilityEvaluator {
    /// Configuration
    config: CrossLanguageIntelligibilityConfig,
    /// Universal phoneme mapper
    phoneme_mapper: UniversalPhonemeMapper,
    /// Cross-cultural perceptual model
    cultural_model: CrossCulturalPerceptualModel,
    /// Cached intelligibility predictions
    intelligibility_cache: HashMap<(LanguageCode, LanguageCode), f32>,
    /// Precomputed phonetic distance matrices
    phonetic_distance_matrices: HashMap<(LanguageCode, LanguageCode), Vec<Vec<f32>>>,
}

impl CrossLanguageIntelligibilityEvaluator {
    /// Create new cross-language intelligibility evaluator
    pub fn new(config: CrossLanguageIntelligibilityConfig) -> Self {
        let phoneme_mapper = UniversalPhonemeMapper::new(UniversalPhonemeMappingConfig::default());
        let cultural_model = CrossCulturalPerceptualModel::new(CrossCulturalConfig::default());

        let mut evaluator = Self {
            config,
            phoneme_mapper,
            cultural_model,
            intelligibility_cache: HashMap::new(),
            phonetic_distance_matrices: HashMap::new(),
        };

        evaluator.precompute_phonetic_distance_matrices();
        evaluator
    }

    /// Precompute phonetic distance matrices for all language pairs
    fn precompute_phonetic_distance_matrices(&mut self) {
        let supported_languages = self.phoneme_mapper.get_supported_languages();

        for &lang1 in &supported_languages {
            for &lang2 in &supported_languages {
                if lang1 != lang2 {
                    let distance_matrix = self.compute_phonetic_distance_matrix(lang1, lang2);
                    self.phonetic_distance_matrices
                        .insert((lang1, lang2), distance_matrix);
                }
            }
        }
    }

    /// Compute phonetic distance matrix for a language pair
    fn compute_phonetic_distance_matrix(
        &self,
        lang1: LanguageCode,
        lang2: LanguageCode,
    ) -> Vec<Vec<f32>> {
        let mut distance_matrix = Vec::new();

        if let (Some(inventory1), Some(inventory2)) = (
            self.phoneme_mapper.get_language_inventory(lang1),
            self.phoneme_mapper.get_language_inventory(lang2),
        ) {
            for phoneme1 in inventory1 {
                let mut row = Vec::new();
                for phoneme2 in inventory2 {
                    let similarity = self
                        .phoneme_mapper
                        .calculate_similarity_score(phoneme1, phoneme2, lang1, lang2);
                    let distance = 1.0 - similarity;
                    row.push(distance);
                }
                distance_matrix.push(row);
            }
        }

        distance_matrix
    }

    /// Evaluate cross-language intelligibility
    pub async fn evaluate_intelligibility(
        &self,
        audio: &AudioBuffer,
        source_language: LanguageCode,
        target_language: LanguageCode,
        phoneme_alignment: Option<&PhonemeAlignment>,
        listener_profile: Option<&ListenerProficiencyProfile>,
        context_factors: Option<&ContextDependencyFactors>,
    ) -> EvaluationResult<CrossLanguageIntelligibilityResult> {
        let start_time = std::time::Instant::now();

        // Calculate phonetic distance contribution
        let phonetic_distance_score = if self.config.enable_phonetic_distance {
            self.calculate_phonetic_distance_score(
                source_language,
                target_language,
                phoneme_alignment,
            )?
        } else {
            0.5
        };

        // Calculate acoustic similarity contribution
        let acoustic_similarity_score = if self.config.enable_acoustic_similarity {
            self.calculate_acoustic_similarity_score(audio, source_language, target_language)
                .await?
        } else {
            0.5
        };

        // Calculate perceptual adaptation contribution
        let perceptual_adaptation_score = if self.config.enable_perceptual_adaptation {
            self.calculate_perceptual_adaptation_score(audio, source_language, target_language)
                .await?
        } else {
            0.5
        };

        // Calculate listener proficiency contribution
        let listener_proficiency_score = if self.config.enable_listener_proficiency {
            self.calculate_listener_proficiency_score(
                listener_profile,
                source_language,
                target_language,
            )?
        } else {
            0.5
        };

        // Calculate context dependency contribution
        let context_dependency_score = if self.config.enable_context_dependency {
            self.calculate_context_dependency_score(context_factors, phoneme_alignment)?
        } else {
            0.5
        };

        // Calculate overall intelligibility
        let overall_intelligibility = self.calculate_overall_intelligibility(
            phonetic_distance_score,
            acoustic_similarity_score,
            perceptual_adaptation_score,
            listener_proficiency_score,
            context_dependency_score,
        );

        // Calculate word-level intelligibility
        let word_intelligibility = self.calculate_word_intelligibility(
            phoneme_alignment,
            source_language,
            target_language,
            overall_intelligibility,
        )?;

        // Calculate phoneme-level intelligibility
        let phoneme_intelligibility = self.calculate_phoneme_intelligibility(
            phoneme_alignment,
            source_language,
            target_language,
        )?;

        // Identify problematic regions
        let problematic_regions = self.identify_problematic_regions(
            &phoneme_intelligibility,
            &word_intelligibility,
            phoneme_alignment,
        );

        // Calculate prediction confidence
        let prediction_confidence = self.calculate_prediction_confidence(
            phonetic_distance_score,
            acoustic_similarity_score,
            perceptual_adaptation_score,
            listener_proficiency_score,
            context_dependency_score,
        );

        let processing_time = start_time.elapsed();

        Ok(CrossLanguageIntelligibilityResult {
            source_language,
            target_language,
            overall_intelligibility,
            phonetic_distance_score,
            acoustic_similarity_score,
            perceptual_adaptation_score,
            listener_proficiency_score,
            context_dependency_score,
            word_intelligibility,
            phoneme_intelligibility,
            problematic_regions,
            prediction_confidence,
            processing_time,
        })
    }

    /// Calculate phonetic distance score
    fn calculate_phonetic_distance_score(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        phoneme_alignment: Option<&PhonemeAlignment>,
    ) -> EvaluationResult<f32> {
        if let Some(alignment) = phoneme_alignment {
            let mut total_distance = 0.0;
            let mut phoneme_count = 0;

            for aligned_phoneme in &alignment.phonemes {
                if let Some(mapping) = self.phoneme_mapper.map_phoneme(
                    &aligned_phoneme.phoneme.symbol,
                    source_language,
                    target_language,
                ) {
                    total_distance += 1.0 - mapping.similarity_score;
                    phoneme_count += 1;
                }
            }

            if phoneme_count > 0 {
                let average_distance = total_distance / phoneme_count as f32;
                Ok(1.0
                    - average_distance.min(self.config.max_phonetic_distance)
                        / self.config.max_phonetic_distance)
            } else {
                Ok(0.5)
            }
        } else {
            // Use precomputed language-level distance
            if let Some(coverage) = self
                .phoneme_mapper
                .analyze_phoneme_coverage(source_language, target_language)
                .ok()
            {
                Ok(coverage.average_mapping_quality)
            } else {
                Ok(0.5)
            }
        }
    }

    /// Calculate acoustic similarity score
    async fn calculate_acoustic_similarity_score(
        &self,
        audio: &AudioBuffer,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
        let acoustic_analysis = self.analyze_acoustic_features(audio).await?;

        // Calculate acoustic similarity based on language-specific expectations
        let formant_score =
            self.evaluate_formant_clarity(&acoustic_analysis, source_language, target_language);

        let spectral_score = self.evaluate_spectral_envelope_similarity(
            &acoustic_analysis,
            source_language,
            target_language,
        );

        let temporal_score = self.evaluate_temporal_envelope_preservation(
            &acoustic_analysis,
            source_language,
            target_language,
        );

        let voice_quality_score = self.evaluate_voice_quality_metrics(
            &acoustic_analysis,
            source_language,
            target_language,
        );

        let prosodic_score =
            self.evaluate_prosodic_features(&acoustic_analysis, source_language, target_language);

        // Weighted combination of acoustic features
        let overall_score = formant_score * 0.25
            + spectral_score * 0.25
            + temporal_score * 0.2
            + voice_quality_score * 0.15
            + prosodic_score * 0.15;

        Ok(overall_score)
    }

    /// Analyze acoustic features
    async fn analyze_acoustic_features(
        &self,
        audio: &AudioBuffer,
    ) -> EvaluationResult<AcousticAnalysisResult> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        // Calculate voice quality metrics (jitter, shimmer, HNR, spectral tilt,
        // formant bandwidths) directly from the signal.
        let voice_quality_metrics = self.calculate_voice_quality_metrics(samples, sample_rate);

        // Calculate prosodic features
        let prosodic_features = self.calculate_prosodic_features(samples, sample_rate);

        // Calculate signal-to-noise ratio
        let signal_to_noise_ratio = self.calculate_snr(samples);

        // Formant clarity derived from the prominence of the LPC spectral-envelope
        // peaks (how sharply F1/F2/F3 stand above the surrounding valleys).
        let formant_clarity = compute_formant_clarity(samples, sample_rate);

        Ok(AcousticAnalysisResult {
            formant_clarity,
            // requires reference signal: spectral-envelope *similarity* needs a target spectrum
            spectral_envelope_similarity: 0.75,
            // requires reference signal: envelope *preservation* needs a reference envelope
            temporal_envelope_preservation: 0.85,
            voice_quality_metrics,
            prosodic_features,
            signal_to_noise_ratio,
        })
    }

    /// Calculate voice quality metrics from the raw signal.
    ///
    /// All metrics are computed with real DSP (see [`crate::quality::voice_quality_dsp`]):
    /// - **jitter**: period-to-period F0 variation from autocorrelation-guided
    ///   pitch marks (`mean(|Tᵢ₊₁ − Tᵢ|) / mean(T)`),
    /// - **shimmer**: cycle-to-cycle peak-amplitude variation
    ///   (`mean(|Aᵢ₊₁ − Aᵢ|) / mean(A)`),
    /// - **harmonic_to_noise_ratio**: from the normalized autocorrelation at the
    ///   pitch period (`10·log10(r / (1 − r))`),
    /// - **spectral_tilt**: slope (dB/octave) of a least-squares fit of the
    ///   log-magnitude spectrum versus log-frequency,
    /// - **formant_bandwidth**: −3 dB widths of the first three LPC
    ///   spectral-envelope peaks.
    ///
    /// Unvoiced or too-short signals yield jitter/shimmer of `0.0`.
    fn calculate_voice_quality_metrics(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> VoiceQualityMetrics {
        // Pitch is estimated once and shared by the jitter/shimmer and HNR paths.
        let pitch = estimate_pitch_period(samples, sample_rate);

        let (jitter, shimmer) = match pitch {
            Some((period, correlation)) if correlation >= VOICING_THRESHOLD => {
                compute_jitter_shimmer(samples, period)
            }
            _ => (0.0, 0.0),
        };

        let harmonic_to_noise_ratio = match pitch {
            Some((_, correlation)) => hnr_from_correlation(correlation),
            None => 0.0,
        };

        VoiceQualityMetrics {
            jitter,
            shimmer,
            harmonic_to_noise_ratio,
            spectral_tilt: compute_spectral_tilt(samples, sample_rate),
            formant_bandwidth: compute_formant_bandwidths(samples, sample_rate),
        }
    }

    /// Calculate prosodic features from the real signal.
    ///
    /// This evaluator is reference-free (no expected F0 contour, stress pattern,
    /// or pause layout is available at this call site — only the raw
    /// synthesized `samples`), so each field below is a genuine, signal-derived
    /// acoustic measurement scored against a documented natural-speech norm
    /// rather than a literal comparison to an "expected" pattern. All are real
    /// `[0, 1]` measurements that vary with the actual audio content (see
    /// [`crate::audio_dsp::windowed_f0_track`] /
    /// [`crate::audio_dsp::frame_rms_envelope`]), not fixed placeholders.
    fn calculate_prosodic_features(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> ProsodicFeatureAnalysis {
        let energy_variance = self.calculate_energy_variance(samples);
        let f0_track = crate::audio_dsp::windowed_f0_track(samples, sample_rate);
        let envelope = crate::audio_dsp::frame_rms_envelope(samples, sample_rate);

        ProsodicFeatureAnalysis {
            f0_contour_appropriateness: self.calculate_f0_contour_appropriateness(&f0_track),
            stress_pattern_accuracy: self.calculate_stress_pattern_strength(&envelope),
            rhythm_regularity: 1.0 - energy_variance.min(1.0), // Use energy variance as rhythm proxy
            intonation_naturalness: self.calculate_intonation_naturalness_from_track(&f0_track),
            pause_pattern_appropriateness: self.calculate_pause_pattern_appropriateness(&envelope),
        }
    }

    /// Coherence/completeness of a trackable pitch contour: a real F0 contour
    /// requires a substantial, mostly-contiguous voiced region (isolated
    /// single-frame pitch blips are spurious detections, not a genuine
    /// contour), and connected speech is neither fully silent/unvoiced (no
    /// contour at all) nor 100% voiced (a single sustained tone is not
    /// speech-like prosody).
    fn calculate_f0_contour_appropriateness(&self, f0_track: &[f32]) -> f32 {
        if f0_track.is_empty() {
            return 0.0;
        }
        let voiced_fraction =
            f0_track.iter().filter(|&&f0| f0 > 0.0).count() as f32 / f0_track.len() as f32;

        let mut longest_run = 0usize;
        let mut current_run = 0usize;
        let mut total_voiced = 0usize;
        for &f0 in f0_track {
            if f0 > 0.0 {
                current_run += 1;
                total_voiced += 1;
                longest_run = longest_run.max(current_run);
            } else {
                current_run = 0;
            }
        }
        let coherence = if total_voiced == 0 {
            0.0
        } else {
            longest_run as f32 / total_voiced as f32
        };

        let voicing_component = if voiced_fraction < 0.05 {
            0.0
        } else {
            1.0 - ((voiced_fraction - 0.5).abs() / 0.5).clamp(0.0, 1.0)
        };
        (voicing_component * 0.4 + coherence * 0.6).clamp(0.0, 1.0)
    }

    /// Smoothness and range of the voiced F0 contour: natural intonation has
    /// *some* pitch movement (unlike a flat monotone) but changes gradually
    /// rather than jumping erratically frame to frame. Mirrors the
    /// signal-only intonation heuristic in
    /// [`crate::quality::advanced_metrics::AdvancedQualityEvaluator`].
    fn calculate_intonation_naturalness_from_track(&self, f0_track: &[f32]) -> f32 {
        let voiced: Vec<f32> = f0_track.iter().copied().filter(|&f0| f0 > 0.0).collect();
        if voiced.len() < 2 {
            return 0.0;
        }
        let mean = (voiced.iter().sum::<f32>() / voiced.len() as f32).max(1.0);
        let range = voiced.iter().cloned().fold(f32::MIN, f32::max)
            - voiced.iter().cloned().fold(f32::MAX, f32::min);
        let range_score = (range / (mean * 0.6)).clamp(0.0, 1.0);
        let jumps: Vec<f32> = voiced
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() / mean)
            .collect();
        let mean_jump = jumps.iter().sum::<f32>() / jumps.len() as f32;
        let smoothness_score = (1.0 - mean_jump / 0.3).clamp(0.0, 1.0);
        (range_score * 0.5 + smoothness_score * 0.5).clamp(0.0, 1.0)
    }

    /// Stress-pattern prominence: the acoustic correlate of stress is a burst
    /// of energy above the surrounding baseline. Measures the mean height of
    /// local energy peaks relative to the overall mean frame energy and scores
    /// it against a natural-speech contrast range (a flat, near-1.0 ratio
    /// indicates weak/absent stress marking — e.g. monotone delivery — while
    /// extreme, isolated spikes suggest transients rather than genuine
    /// stress).
    fn calculate_stress_pattern_strength(&self, envelope: &[f32]) -> f32 {
        if envelope.len() < 3 {
            return 0.0;
        }
        let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
        if mean <= 1e-8 {
            return 0.0;
        }
        let peaks: Vec<f32> = envelope
            .windows(3)
            .filter(|w| w[1] > w[0] && w[1] > w[2])
            .map(|w| w[1])
            .collect();
        if peaks.is_empty() {
            return 0.3;
        }
        let mean_peak = peaks.iter().sum::<f32>() / peaks.len() as f32;
        let contrast = mean_peak / mean;
        let target_contrast = 1.8;
        (1.0 - ((contrast - target_contrast).abs() / target_contrast).clamp(0.0, 1.0))
            .clamp(0.0, 1.0)
    }

    /// Pause-pattern naturalness: detects silence/low-energy segments via
    /// envelope thresholding and scores their overall prevalence against the
    /// proportion typically occupied by inter-word/phrase pauses in connected
    /// natural speech (~10-30% of frames), plus whether at least one such
    /// segment is present at all.
    fn calculate_pause_pattern_appropriateness(&self, envelope: &[f32]) -> f32 {
        if envelope.len() < 3 {
            return 0.5;
        }
        let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
        if mean <= 1e-8 {
            return 0.0;
        }
        let threshold = mean * 0.2;

        let mut pause_frames = 0usize;
        let mut pause_segments = 0usize;
        let mut in_pause = false;
        for &e in envelope {
            if e < threshold {
                pause_frames += 1;
                if !in_pause {
                    pause_segments += 1;
                    in_pause = true;
                }
            } else {
                in_pause = false;
            }
        }

        let pause_fraction = pause_frames as f32 / envelope.len() as f32;
        let target_fraction = 0.2;
        let fraction_score = (1.0
            - ((pause_fraction - target_fraction).abs() / 0.3).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let segment_score = if pause_segments > 0 { 1.0 } else { 0.3 };
        (fraction_score * 0.7 + segment_score * 0.3).clamp(0.0, 1.0)
    }

    /// Calculate energy variance
    fn calculate_energy_variance(&self, samples: &[f32]) -> f32 {
        let chunk_size = 1600; // ~100ms at 16kHz
        let mut energy_values = Vec::new();

        for chunk in samples.chunks(chunk_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            energy_values.push(energy);
        }

        if energy_values.len() < 2 {
            return 0.0;
        }

        let mean = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let variance = energy_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;

        variance.sqrt() / mean.max(0.001)
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, samples: &[f32]) -> f32 {
        let signal_power = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let noise_floor = 0.001; // Assumed noise floor

        if signal_power > noise_floor {
            10.0 * (signal_power / noise_floor).log10()
        } else {
            0.0
        }
    }

    /// Evaluate formant clarity
    fn evaluate_formant_clarity(
        &self,
        acoustic_analysis: &AcousticAnalysisResult,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        acoustic_analysis.formant_clarity
    }

    /// Evaluate spectral envelope similarity
    fn evaluate_spectral_envelope_similarity(
        &self,
        acoustic_analysis: &AcousticAnalysisResult,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        acoustic_analysis.spectral_envelope_similarity
    }

    /// Evaluate temporal envelope preservation
    fn evaluate_temporal_envelope_preservation(
        &self,
        acoustic_analysis: &AcousticAnalysisResult,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        acoustic_analysis.temporal_envelope_preservation
    }

    /// Evaluate voice quality metrics
    fn evaluate_voice_quality_metrics(
        &self,
        acoustic_analysis: &AcousticAnalysisResult,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        let voice_quality = &acoustic_analysis.voice_quality_metrics;

        // Combine voice quality metrics
        let jitter_score = 1.0 - (voice_quality.jitter / 0.1).min(1.0);
        let shimmer_score = 1.0 - (voice_quality.shimmer / 0.1).min(1.0);
        let hnr_score = (voice_quality.harmonic_to_noise_ratio / 20.0)
            .min(1.0)
            .max(0.0);

        (jitter_score + shimmer_score + hnr_score) / 3.0
    }

    /// Evaluate prosodic features
    fn evaluate_prosodic_features(
        &self,
        acoustic_analysis: &AcousticAnalysisResult,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        let prosodic = &acoustic_analysis.prosodic_features;

        // Combine prosodic features
        (prosodic.f0_contour_appropriateness
            + prosodic.stress_pattern_accuracy
            + prosodic.rhythm_regularity
            + prosodic.intonation_naturalness
            + prosodic.pause_pattern_appropriateness)
            / 5.0
    }

    /// Calculate perceptual adaptation score
    async fn calculate_perceptual_adaptation_score(
        &self,
        audio: &AudioBuffer,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
        // Helper function to get 2-letter language code
        let get_language_code = |lang: LanguageCode| -> String {
            match lang {
                LanguageCode::EnUs | LanguageCode::EnGb => "en".to_string(),
                LanguageCode::EsEs | LanguageCode::EsMx | LanguageCode::Es => "es".to_string(),
                LanguageCode::FrFr | LanguageCode::Fr => "fr".to_string(),
                LanguageCode::DeDe | LanguageCode::De => "de".to_string(),
                LanguageCode::JaJp | LanguageCode::Ja => "ja".to_string(),
                LanguageCode::ZhCn => "zh".to_string(),
                LanguageCode::PtBr | LanguageCode::Pt => "pt".to_string(),
                LanguageCode::RuRu | LanguageCode::Ru => "ru".to_string(),
                LanguageCode::ItIt | LanguageCode::It => "it".to_string(),
                LanguageCode::KoKr | LanguageCode::Ko => "ko".to_string(),
                LanguageCode::Ar => "ar".to_string(),
                LanguageCode::Hi => "hi".to_string(),
                _ => "en".to_string(), // Default fallback
            }
        };

        let target_lang_code = get_language_code(target_language);
        let source_lang_code = get_language_code(source_language);

        // Create dummy cultural and demographic profiles
        let cultural_profile = CulturalProfile {
            region: crate::perceptual::CulturalRegion::NorthAmerica,
            language_familiarity: vec![target_lang_code.clone()],
            musical_training: false,
            accent_tolerance: 0.7,
        };

        let demographic_profile = DemographicProfile {
            age_group: crate::perceptual::AgeGroup::MiddleAged,
            gender: crate::perceptual::Gender::Other,
            education_level: crate::perceptual::EducationLevel::Bachelor,
            native_language: target_lang_code,
            audio_experience: crate::perceptual::ExperienceLevel::Intermediate,
        };

        // Calculate cross-cultural adaptation factors
        let adaptation_factors = self.cultural_model.calculate_adaptation_factors(
            &cultural_profile,
            &demographic_profile,
            audio,
            &source_lang_code,
        )?;

        // Combine adaptation factors
        let overall_adaptation = adaptation_factors.phonetic_distance_factor * 0.3
            + adaptation_factors.prosodic_mismatch_factor * 0.3
            + adaptation_factors.accent_familiarity_factor * 0.2
            + adaptation_factors.communication_style_factor * 0.1
            + adaptation_factors.linguistic_distance_factor * 0.1;

        Ok(overall_adaptation)
    }

    /// Calculate listener proficiency score
    fn calculate_listener_proficiency_score(
        &self,
        listener_profile: Option<&ListenerProficiencyProfile>,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
        if let Some(profile) = listener_profile {
            let proficiency_score =
                self.proficiency_level_to_score(&profile.target_language_proficiency);
            let familiarity_score =
                self.proficiency_level_to_score(&profile.source_language_familiarity);
            let experience_score =
                self.listening_experience_to_score(&profile.listening_experience);

            let overall_score = proficiency_score * 0.4
                + familiarity_score * 0.3
                + experience_score * 0.2
                + profile.adaptation_capability * 0.1;

            Ok(overall_score)
        } else {
            // Default intermediate proficiency
            Ok(0.6)
        }
    }

    /// Convert proficiency level to score
    fn proficiency_level_to_score(&self, level: &ProficiencyLevel) -> f32 {
        match level {
            ProficiencyLevel::Beginner => 0.2,
            ProficiencyLevel::Elementary => 0.35,
            ProficiencyLevel::Intermediate => 0.5,
            ProficiencyLevel::UpperIntermediate => 0.65,
            ProficiencyLevel::Advanced => 0.8,
            ProficiencyLevel::Native => 1.0,
        }
    }

    /// Convert listening experience to score
    fn listening_experience_to_score(&self, experience: &ListeningExperience) -> f32 {
        let total_hours_score = (experience.total_listening_hours / 1000.0).min(1.0);
        let cross_linguistic_score = (experience.cross_linguistic_hours / 500.0).min(1.0);
        let synthetic_familiarity_score = experience.synthetic_speech_familiarity;
        let accent_tolerance_score = experience.accent_tolerance;

        (total_hours_score
            + cross_linguistic_score
            + synthetic_familiarity_score
            + accent_tolerance_score)
            / 4.0
    }

    /// Calculate context dependency score
    fn calculate_context_dependency_score(
        &self,
        context_factors: Option<&ContextDependencyFactors>,
        _phoneme_alignment: Option<&PhonemeAlignment>,
    ) -> EvaluationResult<f32> {
        if let Some(factors) = context_factors {
            let semantic_score = factors.semantic_predictability;
            let syntactic_score = 1.0 - (factors.syntactic_complexity / 2.0).min(1.0);
            let lexical_score = factors.lexical_frequency;
            let discourse_score = factors.discourse_coherence;
            let visual_score = if factors.visual_context_available {
                1.0
            } else {
                0.7
            };
            let noise_score = 1.0 - (factors.background_noise_level / 2.0).min(1.0);

            let overall_score = semantic_score * 0.25
                + syntactic_score * 0.2
                + lexical_score * 0.2
                + discourse_score * 0.15
                + visual_score * 0.1
                + noise_score * 0.1;

            Ok(overall_score)
        } else {
            // Default neutral context
            Ok(0.5)
        }
    }

    /// Calculate overall intelligibility
    fn calculate_overall_intelligibility(
        &self,
        phonetic_distance_score: f32,
        acoustic_similarity_score: f32,
        perceptual_adaptation_score: f32,
        listener_proficiency_score: f32,
        context_dependency_score: f32,
    ) -> f32 {
        let weighted_score = phonetic_distance_score * self.config.phonetic_distance_weight
            + acoustic_similarity_score * self.config.acoustic_similarity_weight
            + perceptual_adaptation_score * self.config.perceptual_adaptation_weight
            + listener_proficiency_score * self.config.listener_proficiency_weight
            + context_dependency_score * self.config.context_dependency_weight;

        weighted_score
            .max(self.config.min_intelligibility_threshold)
            .min(1.0)
    }

    /// Calculate word-level intelligibility
    fn calculate_word_intelligibility(
        &self,
        phoneme_alignment: Option<&PhonemeAlignment>,
        source_language: LanguageCode,
        target_language: LanguageCode,
        base_intelligibility: f32,
    ) -> EvaluationResult<Vec<WordIntelligibilityScore>> {
        let mut word_scores = Vec::new();

        if let Some(alignment) = phoneme_alignment {
            for word_alignment in &alignment.word_alignments {
                // Calculate word-specific intelligibility factors
                let phonetic_complexity = self.calculate_word_phonetic_complexity(
                    &word_alignment.phonemes,
                    source_language,
                    target_language,
                );

                let acoustic_clarity = word_alignment.confidence;

                // Real word-length effect: longer words carry more
                // distinguishing phonetic material and are recognized with
                // less reliance on surrounding semantic/syntactic context
                // (a well-established finding in speech perception);
                // shorter words lean more on context to disambiguate.
                // Typical content words span roughly 2-6 phonemes.
                let contextual_support =
                    (word_alignment.phonemes.len() as f32 / 6.0).clamp(0.2, 1.0);

                // Real language-pair-keyed familiarity: the phoneme mapper's
                // own precomputed `mapping_quality` for this
                // source/target language pair (how well the two languages'
                // sound systems correspond overall), not a flat constant.
                // Falls back to a neutral midpoint only when no mapping has
                // been precomputed for this language pair at all.
                let listener_familiarity = self
                    .phoneme_mapper
                    .get_cross_language_mapping(source_language, target_language)
                    .map(|m| m.mapping_quality.clamp(0.0, 1.0))
                    .unwrap_or(0.5);

                let word_intelligibility = base_intelligibility
                    * phonetic_complexity
                    * acoustic_clarity
                    * contextual_support
                    * listener_familiarity;

                word_scores.push(WordIntelligibilityScore {
                    word: word_alignment.word.clone(),
                    intelligibility_score: word_intelligibility,
                    phonetic_complexity,
                    acoustic_clarity,
                    contextual_support,
                    listener_familiarity,
                    start_time: word_alignment.start_time,
                    end_time: word_alignment.end_time,
                });
            }
        }

        Ok(word_scores)
    }

    /// Calculate word phonetic complexity
    fn calculate_word_phonetic_complexity(
        &self,
        phonemes: &[voirs_recognizer::traits::AlignedPhoneme],
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> f32 {
        if phonemes.is_empty() {
            return 0.5;
        }

        let mut total_complexity = 0.0;

        for phoneme in phonemes {
            if let Some(mapping) = self.phoneme_mapper.map_phoneme(
                &phoneme.phoneme.symbol,
                source_language,
                target_language,
            ) {
                // Lower similarity means higher complexity
                total_complexity += 1.0 - mapping.similarity_score;
            }
        }

        1.0 - (total_complexity / phonemes.len() as f32).min(1.0)
    }

    /// Real, per-phoneme-pair perceptual confusability from the phoneme
    /// mapper's own [`PhonemeCandidate`], instead of a flat constant.
    ///
    /// Combines two genuine per-mapping signals that already vary by phoneme
    /// pair: the mapper's `mapping_type` (a [`MappingType::Direct`] match is
    /// rarely confusable; a [`MappingType::Deletion`] — no suitable target
    /// phoneme exists at all — is maximally confusable, since the listener
    /// hears *something* where the source phoneme is simply absent from the
    /// target inventory) and `confidence_score` (a low-confidence mapping
    /// means even the mapper itself was unsure this was the right
    /// correspondence, which correlates with listener confusability).
    fn mapping_perceptual_confusability(mapping: &PhonemeCandidate) -> f32 {
        let type_confusability = match mapping.mapping_type {
            MappingType::Direct => 0.05,
            MappingType::Approximate => 0.25,
            MappingType::Substitution => 0.55,
            MappingType::Insertion => 0.7,
            MappingType::Deletion => 0.9,
        };
        let confidence_confusability = (1.0 - mapping.confidence_score).clamp(0.0, 1.0);
        (type_confusability * 0.6 + confidence_confusability * 0.4).clamp(0.0, 1.0)
    }

    /// Calculate phoneme-level intelligibility
    fn calculate_phoneme_intelligibility(
        &self,
        phoneme_alignment: Option<&PhonemeAlignment>,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<Vec<PhonemeIntelligibilityScore>> {
        let mut phoneme_scores = Vec::new();

        if let Some(alignment) = phoneme_alignment {
            for (position, aligned_phoneme) in alignment.phonemes.iter().enumerate() {
                if let Some(mapping) = self.phoneme_mapper.map_phoneme(
                    &aligned_phoneme.phoneme.symbol,
                    source_language,
                    target_language,
                ) {
                    let phonetic_distance = 1.0 - mapping.similarity_score;
                    let acoustic_similarity = mapping.similarity_score;
                    let perceptual_confusability = Self::mapping_perceptual_confusability(&mapping);

                    let intelligibility_score = mapping.similarity_score
                        * aligned_phoneme.confidence
                        * (1.0 - perceptual_confusability);

                    phoneme_scores.push(PhonemeIntelligibilityScore {
                        source_phoneme: aligned_phoneme.phoneme.symbol.clone(),
                        perceived_phoneme: Some(mapping.target_phoneme),
                        intelligibility_score,
                        phonetic_distance,
                        acoustic_similarity,
                        perceptual_confusability,
                        position,
                        start_time: aligned_phoneme.start_time,
                        end_time: aligned_phoneme.end_time,
                    });
                } else {
                    // Phoneme cannot be mapped
                    phoneme_scores.push(PhonemeIntelligibilityScore {
                        source_phoneme: aligned_phoneme.phoneme.symbol.clone(),
                        perceived_phoneme: None,
                        intelligibility_score: 0.0,
                        phonetic_distance: 2.0,
                        acoustic_similarity: 0.0,
                        perceptual_confusability: 1.0,
                        position,
                        start_time: aligned_phoneme.start_time,
                        end_time: aligned_phoneme.end_time,
                    });
                }
            }
        }

        Ok(phoneme_scores)
    }

    /// Identify problematic regions
    fn identify_problematic_regions(
        &self,
        phoneme_intelligibility: &[PhonemeIntelligibilityScore],
        word_intelligibility: &[WordIntelligibilityScore],
        _phoneme_alignment: Option<&PhonemeAlignment>,
    ) -> Vec<ProblematicRegion> {
        let mut problematic_regions = Vec::new();

        // Identify phoneme-level problems
        for phoneme_score in phoneme_intelligibility {
            if phoneme_score.intelligibility_score < 0.3 {
                let problem_type = if phoneme_score.phonetic_distance > 1.5 {
                    IntelligibilityProblemType::PhoneticMismatch
                } else if phoneme_score.acoustic_similarity < 0.3 {
                    IntelligibilityProblemType::AcousticDistortion
                } else {
                    IntelligibilityProblemType::ProsodicInterference
                };

                problematic_regions.push(ProblematicRegion {
                    start_time: phoneme_score.start_time,
                    end_time: phoneme_score.end_time,
                    severity: 1.0 - phoneme_score.intelligibility_score,
                    problem_type,
                    description: format!(
                        "Phoneme '{}' has low intelligibility",
                        phoneme_score.source_phoneme
                    ),
                    suggestions: vec![
                        "Consider acoustic model adaptation".to_string(),
                        "Improve phonetic clarity".to_string(),
                    ],
                });
            }
        }

        // Identify word-level problems
        for word_score in word_intelligibility {
            if word_score.intelligibility_score < 0.4 {
                let problem_type = if word_score.phonetic_complexity < 0.5 {
                    IntelligibilityProblemType::PhoneticMismatch
                } else if word_score.acoustic_clarity < 0.5 {
                    IntelligibilityProblemType::AcousticDistortion
                } else {
                    IntelligibilityProblemType::LexicalUnfamiliarity
                };

                problematic_regions.push(ProblematicRegion {
                    start_time: word_score.start_time,
                    end_time: word_score.end_time,
                    severity: 1.0 - word_score.intelligibility_score,
                    problem_type,
                    description: format!("Word '{}' has low intelligibility", word_score.word),
                    suggestions: vec![
                        "Consider lexical substitution".to_string(),
                        "Improve pronunciation clarity".to_string(),
                    ],
                });
            }
        }

        problematic_regions
    }

    /// Calculate prediction confidence
    fn calculate_prediction_confidence(
        &self,
        phonetic_distance_score: f32,
        acoustic_similarity_score: f32,
        perceptual_adaptation_score: f32,
        listener_proficiency_score: f32,
        context_dependency_score: f32,
    ) -> f32 {
        let scores = vec![
            phonetic_distance_score,
            acoustic_similarity_score,
            perceptual_adaptation_score,
            listener_proficiency_score,
            context_dependency_score,
        ];

        // Calculate confidence based on consistency of scores
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>()
            / scores.len() as f32;

        // Higher consistency means higher confidence
        let consistency = 1.0 - variance.sqrt();

        // Combine with absolute score level
        let absolute_confidence = mean_score;

        (consistency * 0.6 + absolute_confidence * 0.4)
            .max(0.1)
            .min(1.0)
    }

    /// Get supported language pairs
    pub fn get_supported_language_pairs(&self) -> Vec<(LanguageCode, LanguageCode)> {
        let languages = self.phoneme_mapper.get_supported_languages();
        let mut pairs = Vec::new();

        for &lang1 in &languages {
            for &lang2 in &languages {
                if lang1 != lang2 {
                    pairs.push((lang1, lang2));
                }
            }
        }

        pairs
    }

    /// Predict intelligibility without full evaluation
    pub fn predict_intelligibility(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        listener_proficiency: Option<ProficiencyLevel>,
    ) -> f32 {
        // Check cache first
        if let Some(&cached_score) = self
            .intelligibility_cache
            .get(&(source_language, target_language))
        {
            return cached_score;
        }

        // Calculate based on phoneme coverage
        let phoneme_coverage = self
            .phoneme_mapper
            .analyze_phoneme_coverage(source_language, target_language)
            .map(|coverage| coverage.average_mapping_quality)
            .unwrap_or(0.5);

        // Adjust for listener proficiency
        let proficiency_adjustment = if let Some(proficiency) = listener_proficiency {
            self.proficiency_level_to_score(&proficiency)
        } else {
            0.6
        };

        // Calculate linguistic distance
        let linguistic_distance = self.cultural_model.calculate_linguistic_distance_factor(
            &format!("{:?}", source_language).to_lowercase(),
            &format!("{:?}", target_language).to_lowercase(),
        );

        // Combine factors
        let predicted_intelligibility =
            phoneme_coverage * 0.5 + proficiency_adjustment * 0.3 + linguistic_distance * 0.2;

        predicted_intelligibility.max(0.1).min(1.0)
    }
}

/// Cross-language intelligibility evaluation trait
#[async_trait]
pub trait CrossLanguageIntelligibilityEvaluationTrait {
    /// Evaluate cross-language intelligibility
    async fn evaluate_cross_language_intelligibility(
        &self,
        audio: &AudioBuffer,
        source_language: LanguageCode,
        target_language: LanguageCode,
        phoneme_alignment: Option<&PhonemeAlignment>,
        listener_profile: Option<&ListenerProficiencyProfile>,
        context_factors: Option<&ContextDependencyFactors>,
    ) -> EvaluationResult<CrossLanguageIntelligibilityResult>;

    /// Predict intelligibility
    fn predict_intelligibility(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        listener_proficiency: Option<ProficiencyLevel>,
    ) -> f32;

    /// Get supported language pairs
    fn get_supported_language_pairs(&self) -> Vec<(LanguageCode, LanguageCode)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cross_language_intelligibility_evaluator_creation() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        assert!(!evaluator.get_supported_language_pairs().is_empty());
    }

    #[tokio::test]
    async fn test_intelligibility_prediction() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        let score = evaluator.predict_intelligibility(
            LanguageCode::EnUs,
            LanguageCode::EsEs,
            Some(ProficiencyLevel::Intermediate),
        );

        assert!(score >= 0.1 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_intelligibility_evaluation() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let result = evaluator
            .evaluate_intelligibility(
                &audio,
                LanguageCode::EnUs,
                LanguageCode::EsEs,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(result.source_language, LanguageCode::EnUs);
        assert_eq!(result.target_language, LanguageCode::EsEs);
        assert!(result.overall_intelligibility >= 0.0 && result.overall_intelligibility <= 1.0);
        assert!(result.prediction_confidence >= 0.0 && result.prediction_confidence <= 1.0);
    }

    #[test]
    fn test_proficiency_level_conversion() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        assert_eq!(
            evaluator.proficiency_level_to_score(&ProficiencyLevel::Beginner),
            0.2
        );
        assert_eq!(
            evaluator.proficiency_level_to_score(&ProficiencyLevel::Native),
            1.0
        );
        assert_eq!(
            evaluator.proficiency_level_to_score(&ProficiencyLevel::Intermediate),
            0.5
        );
    }

    #[test]
    fn test_supported_language_pairs() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        let pairs = evaluator.get_supported_language_pairs();
        assert!(!pairs.is_empty());

        // Check that we have both directions for each pair
        assert!(pairs.contains(&(LanguageCode::EnUs, LanguageCode::EsEs)));
        assert!(pairs.contains(&(LanguageCode::EsEs, LanguageCode::EnUs)));
    }

    /// `calculate_prosodic_features` must be a real, signal-dependent
    /// measurement: silence and a modulated, intonated tone must not produce
    /// identical prosodic scores (the old code returned four fixed constants
    /// regardless of the signal).
    #[test]
    fn test_prosodic_features_vary_with_signal() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);
        let sample_rate = 16_000u32;

        let silence = vec![0.0f32; sample_rate as usize];
        // A pitch contour that rises then falls, amplitude-modulated to create
        // pause-like dips -- a much more "prosodic" signal than a flat tone.
        let mut varied = Vec::new();
        for i in 0..sample_rate {
            let t = i as f32 / sample_rate as f32;
            let f0 = 150.0 + 60.0 * (2.0 * std::f32::consts::PI * 1.5 * t).sin();
            let envelope = if (t * 4.0).rem_euclid(1.0) < 0.7 {
                1.0
            } else {
                0.0
            };
            varied.push(envelope * (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.6);
        }

        let silent_features = evaluator.calculate_prosodic_features(&silence, sample_rate);
        let varied_features = evaluator.calculate_prosodic_features(&varied, sample_rate);

        assert_ne!(
            silent_features.f0_contour_appropriateness, varied_features.f0_contour_appropriateness,
            "f0_contour_appropriateness must depend on the signal, not be a fixed 0.8"
        );
        assert_ne!(
            silent_features.intonation_naturalness, varied_features.intonation_naturalness,
            "intonation_naturalness must depend on the signal, not be a fixed 0.7"
        );
        assert_ne!(
            silent_features.stress_pattern_accuracy, varied_features.stress_pattern_accuracy,
            "stress_pattern_accuracy must depend on the signal, not be a fixed 0.75"
        );
        // Silence has no F0 contour at all.
        assert_eq!(silent_features.f0_contour_appropriateness, 0.0);
        assert_eq!(silent_features.intonation_naturalness, 0.0);
        for value in [
            varied_features.f0_contour_appropriateness,
            varied_features.stress_pattern_accuracy,
            varied_features.intonation_naturalness,
            varied_features.pause_pattern_appropriateness,
        ] {
            assert!((0.0..=1.0).contains(&value), "score {value} out of [0, 1]");
        }
    }

    /// `mapping_perceptual_confusability` must vary by mapping type/confidence,
    /// not be a flat 0.3 for every phoneme pair.
    #[test]
    fn test_mapping_perceptual_confusability_varies_by_type() {
        let direct = PhonemeCandidate {
            target_phoneme: "a".to_string(),
            similarity_score: 0.95,
            confidence_score: 0.95,
            mapping_type: MappingType::Direct,
        };
        let deletion = PhonemeCandidate {
            target_phoneme: String::new(),
            similarity_score: 0.1,
            confidence_score: 0.2,
            mapping_type: MappingType::Deletion,
        };

        let direct_confusability =
            CrossLanguageIntelligibilityEvaluator::mapping_perceptual_confusability(&direct);
        let deletion_confusability =
            CrossLanguageIntelligibilityEvaluator::mapping_perceptual_confusability(&deletion);

        assert!((0.0..=1.0).contains(&direct_confusability));
        assert!((0.0..=1.0).contains(&deletion_confusability));
        assert!(
            deletion_confusability > direct_confusability,
            "a deletion mapping (no target phoneme) ({deletion_confusability}) should be far \
             more confusable than a direct, high-confidence mapping ({direct_confusability})"
        );
        assert_ne!(direct_confusability, 0.3);
        assert_ne!(deletion_confusability, 0.3);
    }

    /// `calculate_word_intelligibility`'s `contextual_support` and
    /// `listener_familiarity` must depend on the actual word/language pair,
    /// not be fixed 0.7/0.6 constants for every call.
    #[tokio::test]
    async fn test_word_intelligibility_context_varies_with_input() {
        let config = CrossLanguageIntelligibilityConfig::default();
        let evaluator = CrossLanguageIntelligibilityEvaluator::new(config);

        let make_phoneme = |sym: &str| voirs_sdk::Phoneme {
            symbol: sym.to_string(),
            ipa_symbol: sym.to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
            duration_ms: Some(80.0),
            confidence: 0.9,
        };
        let make_aligned = |sym: &str, start: f32| voirs_recognizer::traits::AlignedPhoneme {
            phoneme: make_phoneme(sym),
            start_time: start,
            end_time: start + 0.08,
            confidence: 0.9,
        };

        let short_word = voirs_recognizer::traits::WordAlignment {
            word: "a".to_string(),
            start_time: 0.0,
            end_time: 0.08,
            phonemes: vec![make_aligned("a", 0.0)],
            confidence: 0.9,
        };
        let long_word = voirs_recognizer::traits::WordAlignment {
            word: "wonderful".to_string(),
            start_time: 0.0,
            end_time: 0.56,
            phonemes: vec![
                make_aligned("w", 0.0),
                make_aligned("o", 0.08),
                make_aligned("n", 0.16),
                make_aligned("d", 0.24),
                make_aligned("r", 0.32),
                make_aligned("f", 0.40),
                make_aligned("l", 0.48),
            ],
            confidence: 0.9,
        };

        let alignment = PhonemeAlignment {
            phonemes: vec![
                short_word.phonemes[0].clone(),
                long_word.phonemes[0].clone(),
            ],
            total_duration: 0.6,
            alignment_confidence: 0.9,
            word_alignments: vec![short_word, long_word],
        };

        let scores = evaluator
            .calculate_word_intelligibility(
                Some(&alignment),
                LanguageCode::EnUs,
                LanguageCode::EsEs,
                0.8,
            )
            .unwrap();

        assert_eq!(scores.len(), 2);
        assert_ne!(
            scores[0].contextual_support, scores[1].contextual_support,
            "contextual_support must depend on word length, not be a fixed 0.7"
        );
        assert!(scores[1].contextual_support > scores[0].contextual_support);
        for score in &scores {
            assert_ne!(score.listener_familiarity, 0.6);
            assert!((0.0..=1.0).contains(&score.contextual_support));
            assert!((0.0..=1.0).contains(&score.listener_familiarity));
        }
    }
}
