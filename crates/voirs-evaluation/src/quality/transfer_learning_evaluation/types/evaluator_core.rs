//! Core implementation for TransferLearningEvaluator:
//! construction, language similarity, knowledge transfer, and effectiveness.

use crate::integration::{EcosystemConfig, EcosystemResults, RecommendationType};
use crate::perceptual::cross_cultural::{CrossCulturalConfig, CrossCulturalPerceptualModel};
use crate::quality::cross_language_intelligibility::{
    CrossLanguageIntelligibilityConfig, CrossLanguageIntelligibilityEvaluator,
};
use crate::quality::multilingual_speaker_models::{
    MultilingualSpeakerModelConfig, MultilingualSpeakerModelEvaluator,
};
use crate::quality::universal_phoneme_mapping::{
    UniversalPhonemeMapper, UniversalPhonemeMappingConfig,
};
use crate::traits::EvaluationResult;
use std::collections::HashMap;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::data_types::{
    AdaptationChallenge, AdaptationChallengeType, ConvergenceAnalysis, ConvergencePattern,
    DomainAdaptationResult, FewShotPerformance, ImplementationEffort, KnowledgeTransferAssessment,
    LearningCurvePoint, NegativeTransferDetectionResult, NegativeTransferSource,
    NegativeTransferSourceType, ProblematicTransferPair, StabilityMetrics, TransferHistoryEntry,
    TransferLearningEvaluationConfig, TransferLearningEvaluationResult, TransferLearningEvaluator,
    TransferLearningMetrics, TransferOptimizationRecommendation,
    TransferOptimizationRecommendationType, TransferProblemType, TransferStabilityAnalysis,
};

impl TransferLearningEvaluator {
    /// Create new transfer learning evaluator
    pub fn new(config: TransferLearningEvaluationConfig) -> Self {
        let phoneme_mapper = UniversalPhonemeMapper::new(UniversalPhonemeMappingConfig::default());
        let intelligibility_evaluator = CrossLanguageIntelligibilityEvaluator::new(
            CrossLanguageIntelligibilityConfig::default(),
        );
        let speaker_model_evaluator =
            MultilingualSpeakerModelEvaluator::new(MultilingualSpeakerModelConfig::default());
        let cultural_model = CrossCulturalPerceptualModel::new(CrossCulturalConfig::default());
        let mut evaluator = Self {
            config,
            phoneme_mapper,
            intelligibility_evaluator,
            speaker_model_evaluator,
            cultural_model,
            transfer_history_cache: HashMap::new(),
            language_similarity_matrix: HashMap::new(),
        };
        evaluator.precompute_language_similarity_matrix();
        evaluator
    }
    /// Precompute language similarity matrix
    fn precompute_language_similarity_matrix(&mut self) {
        let languages = &self.config.evaluation_languages;
        for &lang1 in languages {
            for &lang2 in languages {
                if lang1 != lang2 {
                    let similarity = self.calculate_language_similarity(lang1, lang2);
                    self.language_similarity_matrix
                        .insert((lang1, lang2), similarity);
                }
            }
        }
    }
    /// Calculate language similarity
    pub(crate) fn calculate_language_similarity(
        &self,
        lang1: LanguageCode,
        lang2: LanguageCode,
    ) -> f32 {
        if lang1 == lang2 {
            return 1.0;
        }
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
                _ => "en".to_string(),
            }
        };
        let linguistic_distance = self.cultural_model.calculate_linguistic_distance_factor(
            &get_language_code(lang1),
            &get_language_code(lang2),
        );
        let phoneme_coverage = self
            .phoneme_mapper
            .analyze_phoneme_coverage(lang1, lang2)
            .map(|coverage| coverage.average_mapping_quality)
            .unwrap_or(0.5);
        let intelligibility_similarity = self
            .intelligibility_evaluator
            .predict_intelligibility(lang1, lang2, None);
        let overall_similarity =
            linguistic_distance * 0.4 + phoneme_coverage * 0.35 + intelligibility_similarity * 0.25;
        overall_similarity.max(0.0).min(1.0)
    }
    /// Evaluate transfer learning performance
    pub async fn evaluate_transfer_learning(
        &mut self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
        transfer_history: Option<&HashMap<(LanguageCode, LanguageCode), Vec<TransferHistoryEntry>>>,
    ) -> EvaluationResult<TransferLearningEvaluationResult> {
        let start_time = std::time::Instant::now();
        if let Some(history) = transfer_history {
            for ((source, target), entries) in history {
                self.transfer_history_cache
                    .insert((*source, *target), entries.clone());
            }
        }
        let knowledge_transfer_assessment = if self.config.enable_knowledge_transfer_assessment {
            self.assess_knowledge_transfer(
                source_language,
                target_audios,
                reference_audios,
                phoneme_alignments,
            )
            .await?
        } else {
            KnowledgeTransferAssessment::default()
        };
        let transfer_effectiveness = if self.config.enable_transfer_effectiveness {
            self.evaluate_transfer_effectiveness(source_language, target_audios, reference_audios)
                .await?
        } else {
            HashMap::new()
        };
        let stability_analysis = if self.config.enable_stability_analysis {
            self.analyze_transfer_stability(
                source_language,
                target_audios.keys().cloned().collect(),
            )
            .await?
        } else {
            TransferStabilityAnalysis::default()
        };
        let few_shot_performance = if self.config.enable_few_shot_evaluation {
            self.evaluate_few_shot_performance(source_language, target_audios, reference_audios)
                .await?
        } else {
            HashMap::new()
        };
        let domain_adaptation = if self.config.enable_domain_adaptation {
            self.assess_domain_adaptation(source_language, target_audios, reference_audios)
                .await?
        } else {
            HashMap::new()
        };
        let negative_transfer_detection = if self.config.enable_negative_transfer_detection {
            self.detect_negative_transfer(source_language, target_audios, &transfer_effectiveness)
                .await?
        } else {
            NegativeTransferDetectionResult::default()
        };
        let transfer_optimization_recommendations = if self.config.enable_transfer_optimization {
            self.generate_transfer_optimization_recommendations(
                source_language,
                target_audios,
                &knowledge_transfer_assessment,
                &transfer_effectiveness,
                &stability_analysis,
                &few_shot_performance,
                &domain_adaptation,
                &negative_transfer_detection,
            )
            .await?
        } else {
            Vec::new()
        };
        let transfer_metrics = self.calculate_transfer_learning_metrics(
            &knowledge_transfer_assessment,
            &transfer_effectiveness,
            &stability_analysis,
            &few_shot_performance,
            &domain_adaptation,
            &negative_transfer_detection,
        );
        let language_transfer_matrix = self.build_language_transfer_matrix(
            source_language,
            target_audios,
            &transfer_effectiveness,
        );
        let problematic_transfer_pairs = self.identify_problematic_transfer_pairs(
            source_language,
            &transfer_effectiveness,
            &stability_analysis,
            &negative_transfer_detection,
        );
        let overall_transfer_score = self.calculate_overall_transfer_score(
            &knowledge_transfer_assessment,
            &transfer_effectiveness,
            &stability_analysis,
            &few_shot_performance,
            &domain_adaptation,
        );
        let evaluation_confidence = self.calculate_evaluation_confidence(
            &knowledge_transfer_assessment,
            &transfer_effectiveness,
            &stability_analysis,
            &few_shot_performance,
            &domain_adaptation,
        );
        let processing_time = start_time.elapsed();
        Ok(TransferLearningEvaluationResult {
            source_language,
            target_languages: target_audios.keys().cloned().collect(),
            overall_transfer_score,
            knowledge_transfer_assessment,
            transfer_effectiveness,
            stability_analysis,
            few_shot_performance,
            domain_adaptation,
            negative_transfer_detection,
            transfer_optimization_recommendations,
            transfer_metrics,
            language_transfer_matrix,
            problematic_transfer_pairs,
            evaluation_confidence,
            processing_time,
        })
    }
    /// Assess cross-linguistic knowledge transfer
    async fn assess_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<KnowledgeTransferAssessment> {
        let mut transfer_coverage = HashMap::new();
        let mut phonetic_scores = Vec::new();
        let mut prosodic_scores = Vec::new();
        let mut acoustic_scores = Vec::new();
        let mut linguistic_scores = Vec::new();
        let mut cultural_scores = Vec::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != source_language {
                let phonetic_transfer = self.assess_phonetic_knowledge_transfer(
                    source_language,
                    *target_language,
                    target_audio,
                    phoneme_alignments,
                )?;
                phonetic_scores.push(phonetic_transfer);
                let prosodic_transfer = self.assess_prosodic_knowledge_transfer(
                    source_language,
                    *target_language,
                    target_audio,
                )?;
                prosodic_scores.push(prosodic_transfer);
                let acoustic_transfer = self.assess_acoustic_knowledge_transfer(
                    source_language,
                    *target_language,
                    target_audio,
                    reference_audios,
                )?;
                acoustic_scores.push(acoustic_transfer);
                let linguistic_transfer =
                    self.assess_linguistic_knowledge_transfer(source_language, *target_language)?;
                linguistic_scores.push(linguistic_transfer);
                let cultural_transfer = self.assess_cultural_knowledge_transfer(
                    source_language,
                    *target_language,
                    target_audio,
                )?;
                cultural_scores.push(cultural_transfer);
                let coverage = self.calculate_transfer_coverage(
                    source_language,
                    *target_language,
                    target_audio,
                    phoneme_alignments,
                )?;
                transfer_coverage.insert(*target_language, coverage);
            }
        }
        let phonetic_knowledge_transfer = self.calculate_average_score(&phonetic_scores);
        let prosodic_knowledge_transfer = self.calculate_average_score(&prosodic_scores);
        let acoustic_knowledge_transfer = self.calculate_average_score(&acoustic_scores);
        let linguistic_knowledge_transfer = self.calculate_average_score(&linguistic_scores);
        let cultural_knowledge_transfer = self.calculate_average_score(&cultural_scores);
        let overall_knowledge_transfer = (phonetic_knowledge_transfer
            + prosodic_knowledge_transfer
            + acoustic_knowledge_transfer
            + linguistic_knowledge_transfer
            + cultural_knowledge_transfer)
            / 5.0;
        let transfer_efficiency =
            self.calculate_transfer_efficiency(&transfer_coverage, overall_knowledge_transfer);
        let transfer_consistency = self.calculate_transfer_consistency(&[
            phonetic_knowledge_transfer,
            prosodic_knowledge_transfer,
            acoustic_knowledge_transfer,
            linguistic_knowledge_transfer,
            cultural_knowledge_transfer,
        ]);
        Ok(KnowledgeTransferAssessment {
            phonetic_knowledge_transfer,
            prosodic_knowledge_transfer,
            acoustic_knowledge_transfer,
            linguistic_knowledge_transfer,
            cultural_knowledge_transfer,
            overall_knowledge_transfer,
            transfer_efficiency,
            transfer_consistency,
            transfer_coverage,
        })
    }
    /// Assess phonetic knowledge transfer
    fn assess_phonetic_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        _target_audio: &AudioBuffer,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<f32> {
        let phoneme_coverage = self
            .phoneme_mapper
            .analyze_phoneme_coverage(source_language, target_language)?;
        let phonetic_distance =
            1.0 - self.get_language_similarity(source_language, target_language);
        let alignment_quality = if let Some(alignments) = phoneme_alignments {
            if let Some(alignment) = alignments.get(&target_language) {
                let total_confidence = alignment.phonemes.iter().map(|p| p.confidence).sum::<f32>();
                if alignment.phonemes.len() > 0 {
                    total_confidence / alignment.phonemes.len() as f32
                } else {
                    0.5
                }
            } else {
                0.5
            }
        } else {
            0.5
        };
        let phonetic_transfer = phoneme_coverage.average_mapping_quality * 0.4
            + (1.0 - phonetic_distance) * 0.35
            + alignment_quality * 0.25;
        Ok(phonetic_transfer.max(0.0).min(1.0))
    }
    /// Assess prosodic knowledge transfer
    fn assess_prosodic_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
    ) -> EvaluationResult<f32> {
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
                _ => "en".to_string(),
            }
        };
        let prosodic_similarity = self.cultural_model.calculate_linguistic_distance_factor(
            &get_language_code(source_language),
            &get_language_code(target_language),
        );
        let prosodic_appropriateness =
            self.analyze_prosodic_appropriateness(target_audio, target_language)?;
        let prosodic_transfer = prosodic_similarity * 0.6 + prosodic_appropriateness * 0.4;
        Ok(prosodic_transfer.max(0.0).min(1.0))
    }
    /// Analyze prosodic appropriateness
    fn analyze_prosodic_appropriateness(
        &self,
        audio: &AudioBuffer,
        _language: LanguageCode,
    ) -> EvaluationResult<f32> {
        let samples = audio.samples();
        let speaking_rate = self.estimate_speaking_rate(samples);
        let pitch_variation = self.estimate_pitch_variation(samples);
        let rhythm_regularity = self.estimate_rhythm_regularity(samples);
        let rate_score = (speaking_rate / 4.5).min(1.0);
        let pitch_score = pitch_variation.min(1.0);
        let rhythm_score = rhythm_regularity.min(1.0);
        let overall_score = (rate_score + pitch_score + rhythm_score) / 3.0;
        Ok(overall_score)
    }
    /// Estimate speaking rate
    fn estimate_speaking_rate(&self, samples: &[f32]) -> f32 {
        let chunk_size = 800;
        let mut energy_peaks = 0;
        let mut prev_energy = 0.0;
        for chunk in samples.chunks(chunk_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            if energy > prev_energy * 1.5 && energy > 0.01 {
                energy_peaks += 1;
            }
            prev_energy = energy;
        }
        let duration_seconds = samples.len() as f32 / 16000.0;
        if duration_seconds > 0.0 {
            (energy_peaks as f32 / duration_seconds).clamp(2.0, 8.0)
        } else {
            4.0
        }
    }
    /// Estimate pitch variation
    fn estimate_pitch_variation(&self, samples: &[f32]) -> f32 {
        if samples.len() < 1600 {
            return 0.5;
        }
        let chunk_size = 1600;
        let mut pitch_estimates = Vec::new();
        for chunk in samples.chunks(chunk_size) {
            if chunk.len() == chunk_size {
                let pitch = self.estimate_pitch(chunk);
                if pitch > 0.0 {
                    pitch_estimates.push(pitch);
                }
            }
        }
        if pitch_estimates.len() < 2 {
            return 0.5;
        }
        let mean = pitch_estimates.iter().sum::<f32>() / pitch_estimates.len() as f32;
        let variance = pitch_estimates
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / pitch_estimates.len() as f32;
        let std_dev = variance.sqrt();
        if mean > 0.0 {
            (std_dev / mean).clamp(0.1, 1.0)
        } else {
            0.5
        }
    }
    /// Estimate pitch using autocorrelation
    fn estimate_pitch(&self, samples: &[f32]) -> f32 {
        let min_period = 40;
        let max_period = 400;
        let mut best_corr = 0.0;
        let mut best_period = min_period;
        for period in min_period..=max_period.min(samples.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;
            for i in 0..(samples.len() - period) {
                correlation += samples[i] * samples[i + period];
                count += 1;
            }
            if count > 0 {
                correlation /= count as f32;
                if correlation > best_corr {
                    best_corr = correlation;
                    best_period = period;
                }
            }
        }
        if best_corr > 0.3 {
            16000.0 / best_period as f32
        } else {
            0.0
        }
    }
    /// Estimate rhythm regularity
    fn estimate_rhythm_regularity(&self, samples: &[f32]) -> f32 {
        let chunk_size = 1600;
        let mut energy_values = Vec::new();
        for chunk in samples.chunks(chunk_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            energy_values.push(energy);
        }
        if energy_values.len() < 3 {
            return 0.5;
        }
        let mean_energy = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let variance = energy_values
            .iter()
            .map(|&x| (x - mean_energy).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let regularity = 1.0 / (1.0 + variance.sqrt());
        regularity.clamp(0.0, 1.0)
    }
    /// Assess acoustic knowledge transfer
    fn assess_acoustic_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<f32> {
        let acoustic_similarity = if let Some(ref_audios) = reference_audios {
            if let Some(ref_audio) = ref_audios.get(&source_language) {
                self.calculate_acoustic_similarity(ref_audio, target_audio)?
            } else {
                0.5
            }
        } else {
            0.5
        };
        let acoustic_compatibility =
            self.calculate_acoustic_compatibility(source_language, target_language, target_audio)?;
        let acoustic_transfer = acoustic_similarity * 0.6 + acoustic_compatibility * 0.4;
        Ok(acoustic_transfer.max(0.0).min(1.0))
    }
    /// Calculate acoustic similarity
    pub(super) fn calculate_acoustic_similarity(
        &self,
        reference_audio: &AudioBuffer,
        target_audio: &AudioBuffer,
    ) -> EvaluationResult<f32> {
        let ref_samples = reference_audio.samples();
        let target_samples = target_audio.samples();
        let ref_centroid = self.calculate_spectral_centroid(ref_samples);
        let target_centroid = self.calculate_spectral_centroid(target_samples);
        let centroid_similarity = 1.0
            - (ref_centroid - target_centroid).abs() / ref_centroid.max(target_centroid).max(1.0);
        let ref_energy = self.calculate_rms_energy(ref_samples);
        let target_energy = self.calculate_rms_energy(target_samples);
        let energy_similarity =
            1.0 - (ref_energy - target_energy).abs() / ref_energy.max(target_energy).max(0.01);
        let overall_similarity = centroid_similarity * 0.6 + energy_similarity * 0.4;
        Ok(overall_similarity.max(0.0).min(1.0))
    }
    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        let chunk_size = 512;
        let mut centroid_sum = 0.0;
        let mut count = 0;
        for chunk in samples.chunks(chunk_size) {
            if chunk.len() == chunk_size {
                let mut weighted_sum = 0.0;
                let mut total_energy = 0.0;
                for (i, &sample) in chunk.iter().enumerate() {
                    let energy = sample * sample;
                    weighted_sum += energy * i as f32;
                    total_energy += energy;
                }
                if total_energy > 0.0 {
                    let centroid = weighted_sum / total_energy;
                    centroid_sum += centroid * 16000.0 / chunk_size as f32;
                    count += 1;
                }
            }
        }
        if count > 0 {
            centroid_sum / count as f32
        } else {
            1000.0
        }
    }
    /// Calculate RMS energy
    fn calculate_rms_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares = samples.iter().map(|&x| x * x).sum::<f32>();
        (sum_squares / samples.len() as f32).sqrt()
    }
    /// Calculate acoustic compatibility
    fn calculate_acoustic_compatibility(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        _target_audio: &AudioBuffer,
    ) -> EvaluationResult<f32> {
        let compatibility = self.get_language_similarity(source_language, target_language);
        Ok(compatibility)
    }
    /// Assess linguistic knowledge transfer
    fn assess_linguistic_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
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
                _ => "en".to_string(),
            }
        };
        let linguistic_distance = self.cultural_model.calculate_linguistic_distance_factor(
            &get_language_code(source_language),
            &get_language_code(target_language),
        );
        let intelligibility = self.intelligibility_evaluator.predict_intelligibility(
            source_language,
            target_language,
            None,
        );
        let linguistic_transfer = linguistic_distance * 0.6 + intelligibility * 0.4;
        Ok(linguistic_transfer.max(0.0).min(1.0))
    }
    /// Assess cultural knowledge transfer
    fn assess_cultural_knowledge_transfer(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
    ) -> EvaluationResult<f32> {
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
                _ => "en".to_string(),
            }
        };
        let cultural_profile = crate::perceptual::CulturalProfile {
            region: crate::perceptual::CulturalRegion::NorthAmerica,
            language_familiarity: vec![get_language_code(target_language)],
            musical_training: false,
            accent_tolerance: 0.7,
        };
        let demographic_profile = crate::perceptual::DemographicProfile {
            age_group: crate::perceptual::AgeGroup::MiddleAged,
            gender: crate::perceptual::Gender::Other,
            education_level: crate::perceptual::EducationLevel::Bachelor,
            native_language: get_language_code(target_language),
            audio_experience: crate::perceptual::ExperienceLevel::Intermediate,
        };
        let adaptation_factors = self.cultural_model.calculate_adaptation_factors(
            &cultural_profile,
            &demographic_profile,
            target_audio,
            &get_language_code(source_language),
        )?;
        let cultural_transfer = (adaptation_factors.accent_familiarity_factor
            + adaptation_factors.communication_style_factor
            + adaptation_factors.linguistic_distance_factor)
            / 3.0;
        Ok(cultural_transfer.max(0.0).min(1.0))
    }
    /// Calculate transfer coverage
    fn calculate_transfer_coverage(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        _target_audio: &AudioBuffer,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<f32> {
        let phoneme_coverage = self
            .phoneme_mapper
            .analyze_phoneme_coverage(source_language, target_language)?;
        let alignment_coverage = if let Some(alignments) = phoneme_alignments {
            if let Some(alignment) = alignments.get(&target_language) {
                alignment.phonemes.len() as f32 / alignment.phonemes.len().max(1) as f32
            } else {
                0.5
            }
        } else {
            0.5
        };
        let overall_coverage =
            phoneme_coverage.average_mapping_quality * 0.7 + alignment_coverage * 0.3;
        Ok(overall_coverage.max(0.0).min(1.0))
    }
    /// Get language similarity from precomputed matrix
    pub(super) fn get_language_similarity(&self, lang1: LanguageCode, lang2: LanguageCode) -> f32 {
        self.language_similarity_matrix
            .get(&(lang1, lang2))
            .copied()
            .unwrap_or(0.5)
    }
    /// Calculate average score
    pub(super) fn calculate_average_score(&self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            0.5
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    /// Calculate transfer efficiency
    fn calculate_transfer_efficiency(
        &self,
        transfer_coverage: &HashMap<LanguageCode, f32>,
        overall_knowledge_transfer: f32,
    ) -> f32 {
        let average_coverage = if transfer_coverage.is_empty() {
            0.5
        } else {
            transfer_coverage.values().sum::<f32>() / transfer_coverage.len() as f32
        };
        if average_coverage > 0.0 {
            (overall_knowledge_transfer / average_coverage).min(1.0)
        } else {
            0.5
        }
    }
    /// Calculate transfer consistency
    pub(super) fn calculate_transfer_consistency(&self, scores: &[f32]) -> f32 {
        if scores.len() < 2 {
            return 1.0;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let consistency = 1.0 / (1.0 + variance.sqrt());
        consistency.max(0.0).min(1.0)
    }
    /// Evaluate transfer effectiveness
    pub(super) async fn evaluate_transfer_effectiveness(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut effectiveness_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != source_language {
                let effectiveness = self.calculate_transfer_effectiveness(
                    source_language,
                    *target_language,
                    target_audio,
                    reference_audios,
                )?;
                effectiveness_scores.insert(*target_language, effectiveness);
            }
        }
        Ok(effectiveness_scores)
    }
    /// Calculate transfer effectiveness
    fn calculate_transfer_effectiveness(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<f32> {
        let baseline_similarity = self.get_language_similarity(source_language, target_language);
        let transfer_quality = if let Some(ref_audios) = reference_audios {
            if let Some(ref_audio) = ref_audios.get(&source_language) {
                self.calculate_acoustic_similarity(ref_audio, target_audio)?
            } else {
                baseline_similarity
            }
        } else {
            baseline_similarity
        };
        let intelligibility_improvement = self.intelligibility_evaluator.predict_intelligibility(
            source_language,
            target_language,
            None,
        );
        let effectiveness = (transfer_quality * 0.5
            + intelligibility_improvement * 0.3
            + baseline_similarity * 0.2)
            .max(0.0)
            .min(1.0);
        Ok(effectiveness)
    }
    /// Get supported languages
    pub fn get_supported_languages(&self) -> Vec<LanguageCode> {
        self.config.evaluation_languages.clone()
    }
    /// Get transfer history
    pub fn get_transfer_history(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> Option<&Vec<TransferHistoryEntry>> {
        self.transfer_history_cache
            .get(&(source_language, target_language))
    }
    /// Add transfer history entry
    pub fn add_transfer_history_entry(
        &mut self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        entry: TransferHistoryEntry,
    ) {
        self.transfer_history_cache
            .entry((source_language, target_language))
            .or_insert_with(Vec::new)
            .push(entry);
    }
    /// Clear transfer history
    pub fn clear_transfer_history(&mut self) {
        self.transfer_history_cache.clear();
    }
}
