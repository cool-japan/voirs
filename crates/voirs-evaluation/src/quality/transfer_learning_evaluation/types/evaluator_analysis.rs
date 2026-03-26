//! Analysis implementation for TransferLearningEvaluator:
//! stability, few-shot, domain adaptation, negative transfer detection, and metrics.

use crate::integration::{RecommendationPriority, RecommendationType};
use crate::traits::EvaluationResult;
use std::collections::HashMap;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::data_types::{
    AdaptationChallenge, AdaptationChallengeType, ConvergenceAnalysis, ConvergencePattern,
    DomainAdaptationResult, FewShotPerformance, ImplementationEffort, KnowledgeTransferAssessment,
    LearningCurvePoint, NegativeTransferDetectionResult, NegativeTransferSource,
    NegativeTransferSourceType, ProblematicTransferPair, StabilityMetrics, TransferHistoryEntry,
    TransferLearningEvaluationConfig, TransferLearningEvaluator, TransferLearningMetrics,
    TransferOptimizationRecommendation, TransferOptimizationRecommendationType,
    TransferProblemType, TransferStabilityAnalysis,
};

impl TransferLearningEvaluator {
    /// Analyze transfer stability
    pub(super) async fn analyze_transfer_stability(
        &self,
        source_language: LanguageCode,
        target_languages: Vec<LanguageCode>,
    ) -> EvaluationResult<TransferStabilityAnalysis> {
        let mut language_stability_metrics = HashMap::new();
        let mut convergence_rates = Vec::new();
        let mut stability_scores = Vec::new();
        for target_language in &target_languages {
            if *target_language != source_language {
                let stability_metrics =
                    self.calculate_language_stability_metrics(source_language, *target_language)?;
                convergence_rates.push(stability_metrics.stability_coefficient);
                stability_scores.push(stability_metrics.stability_coefficient);
                language_stability_metrics.insert(*target_language, stability_metrics);
            }
        }
        let convergence_rate = self.calculate_average_score(&convergence_rates);
        let stability_score = self.calculate_average_score(&stability_scores);
        let cross_language_consistency = self.calculate_transfer_consistency(&stability_scores);
        let noise_robustness = stability_score * 0.8;
        let performance_variance = self.calculate_variance(&stability_scores);
        let convergence_analysis =
            self.analyze_convergence_pattern(source_language, &target_languages)?;
        Ok(TransferStabilityAnalysis {
            convergence_rate,
            stability_score,
            cross_language_consistency,
            noise_robustness,
            performance_variance,
            language_stability_metrics,
            convergence_analysis,
        })
    }
    /// Calculate language stability metrics
    fn calculate_language_stability_metrics(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<StabilityMetrics> {
        let history = self
            .transfer_history_cache
            .get(&(source_language, target_language));
        let (
            mean_performance,
            performance_std,
            performance_variance,
            best_performance,
            final_performance,
            convergence_epochs,
        ) = if let Some(history_entries) = history {
            let performances: Vec<f32> = history_entries.iter().map(|e| e.performance).collect();
            if performances.is_empty() {
                (0.5, 0.1, 0.01, 0.5, 0.5, None)
            } else {
                let mean = performances.iter().sum::<f32>() / performances.len() as f32;
                let variance = performances
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    / performances.len() as f32;
                let std_dev = variance.sqrt();
                let best = performances
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let final_perf = *performances.last().expect("collection should not be empty");
                let convergence_epoch = if performances.len() > 10 {
                    let mut converged_epoch = None;
                    for (i, &perf) in performances.iter().enumerate().skip(10) {
                        let recent_avg = performances[i - 10..i].iter().sum::<f32>() / 10.0;
                        if (perf - recent_avg).abs() < 0.01 {
                            converged_epoch = Some(i);
                            break;
                        }
                    }
                    converged_epoch
                } else {
                    None
                };
                (mean, std_dev, variance, best, final_perf, convergence_epoch)
            }
        } else {
            let similarity = self.get_language_similarity(source_language, target_language);
            (similarity, 0.1, 0.01, similarity, similarity, None)
        };
        let stability_coefficient = if performance_std > 0.0 {
            1.0 / (1.0 + performance_std)
        } else {
            1.0
        };
        Ok(StabilityMetrics {
            mean_performance,
            performance_std,
            performance_variance,
            stability_coefficient,
            convergence_epochs,
            best_performance,
            final_performance,
        })
    }
    /// Calculate variance
    pub(super) fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    }
    /// Analyze convergence pattern
    fn analyze_convergence_pattern(
        &self,
        source_language: LanguageCode,
        target_languages: &[LanguageCode],
    ) -> EvaluationResult<ConvergenceAnalysis> {
        let mut all_patterns = Vec::new();
        let mut convergence_speeds = Vec::new();
        let mut convergence_qualities = Vec::new();
        for &target_language in target_languages {
            if target_language != source_language {
                let pattern =
                    self.analyze_individual_convergence_pattern(source_language, target_language)?;
                all_patterns.push(pattern.clone());
                let (speed, quality) = match pattern {
                    ConvergencePattern::Monotonic => (0.8, 0.9),
                    ConvergencePattern::Oscillating => (0.6, 0.7),
                    ConvergencePattern::Plateau => (0.4, 0.6),
                    ConvergencePattern::Divergent => (0.2, 0.3),
                    ConvergencePattern::Irregular => (0.3, 0.4),
                };
                convergence_speeds.push(speed);
                convergence_qualities.push(quality);
            }
        }
        let convergence_pattern = if all_patterns
            .iter()
            .all(|p| matches!(p, ConvergencePattern::Monotonic))
        {
            ConvergencePattern::Monotonic
        } else if all_patterns
            .iter()
            .any(|p| matches!(p, ConvergencePattern::Divergent))
        {
            ConvergencePattern::Divergent
        } else if all_patterns
            .iter()
            .any(|p| matches!(p, ConvergencePattern::Plateau))
        {
            ConvergencePattern::Plateau
        } else {
            ConvergencePattern::Irregular
        };
        let convergence_speed = self.calculate_average_score(&convergence_speeds);
        let convergence_quality = self.calculate_average_score(&convergence_qualities);
        let convergence_reliability = self.calculate_transfer_consistency(&convergence_qualities);
        let early_stopping_epoch = if convergence_speed > 0.7 {
            Some(50)
        } else {
            None
        };
        let plateau_detected = matches!(convergence_pattern, ConvergencePattern::Plateau);
        let plateau_start_epoch = if plateau_detected { Some(30) } else { None };
        Ok(ConvergenceAnalysis {
            convergence_pattern,
            convergence_speed,
            convergence_quality,
            early_stopping_epoch,
            convergence_reliability,
            plateau_detected,
            plateau_start_epoch,
        })
    }
    /// Analyze individual convergence pattern
    pub(crate) fn analyze_individual_convergence_pattern(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<ConvergencePattern> {
        if let Some(history) = self
            .transfer_history_cache
            .get(&(source_language, target_language))
        {
            if history.len() < 5 {
                return Ok(ConvergencePattern::Irregular);
            }
            let performances: Vec<f32> = history.iter().map(|e| e.performance).collect();
            let mut increasing_count = 0;
            let mut decreasing_count = 0;
            let mut stable_count = 0;
            for i in 1..performances.len() {
                let diff = performances[i] - performances[i - 1];
                if diff > 0.01 {
                    increasing_count += 1;
                } else if diff < -0.01 {
                    decreasing_count += 1;
                } else {
                    stable_count += 1;
                }
            }
            let total = performances.len() - 1;
            let increasing_ratio = increasing_count as f32 / total as f32;
            let decreasing_ratio = decreasing_count as f32 / total as f32;
            let stable_ratio = stable_count as f32 / total as f32;
            if increasing_ratio > 0.7 {
                Ok(ConvergencePattern::Monotonic)
            } else if decreasing_ratio > 0.5 {
                Ok(ConvergencePattern::Divergent)
            } else if stable_ratio > 0.6 {
                Ok(ConvergencePattern::Plateau)
            } else if increasing_ratio > 0.4 && decreasing_ratio > 0.3 {
                Ok(ConvergencePattern::Oscillating)
            } else {
                Ok(ConvergencePattern::Irregular)
            }
        } else {
            let similarity = self.get_language_similarity(source_language, target_language);
            if similarity > 0.8 {
                Ok(ConvergencePattern::Monotonic)
            } else if similarity > 0.6 {
                Ok(ConvergencePattern::Oscillating)
            } else if similarity > 0.4 {
                Ok(ConvergencePattern::Plateau)
            } else {
                Ok(ConvergencePattern::Irregular)
            }
        }
    }
    /// Evaluate few-shot learning performance
    pub(super) async fn evaluate_few_shot_performance(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<HashMap<LanguageCode, FewShotPerformance>> {
        let mut few_shot_performances = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != source_language {
                let performance = self.calculate_few_shot_performance(
                    source_language,
                    *target_language,
                    target_audio,
                    reference_audios,
                )?;
                few_shot_performances.insert(*target_language, performance);
            }
        }
        Ok(few_shot_performances)
    }
    /// Calculate few-shot performance
    fn calculate_few_shot_performance(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<FewShotPerformance> {
        let mut performance_by_sample_size = HashMap::new();
        let mut learning_curve = Vec::new();
        for &sample_size in &self.config.few_shot_sample_sizes {
            let performance = self.simulate_few_shot_performance(
                source_language,
                target_language,
                target_audio,
                reference_audios,
                sample_size,
            )?;
            performance_by_sample_size.insert(sample_size, performance);
            learning_curve.push(LearningCurvePoint {
                samples: sample_size,
                performance,
                variance: 0.05,
                confidence_interval: (performance - 0.1, performance + 0.1),
            });
        }
        let learning_efficiency = self.calculate_learning_efficiency(&performance_by_sample_size);
        let sample_efficiency = self.calculate_sample_efficiency(&performance_by_sample_size);
        let adaptation_speed = self.calculate_adaptation_speed(&performance_by_sample_size);
        let min_samples_needed = self.find_min_samples_needed(&performance_by_sample_size);
        let saturation_point = self.find_saturation_point(&performance_by_sample_size);
        Ok(FewShotPerformance {
            performance_by_sample_size,
            learning_efficiency,
            sample_efficiency,
            adaptation_speed,
            min_samples_needed,
            saturation_point,
            learning_curve,
        })
    }
    /// Simulate few-shot performance
    pub(crate) fn simulate_few_shot_performance(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        _target_audio: &AudioBuffer,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
        sample_size: usize,
    ) -> EvaluationResult<f32> {
        let baseline_effectiveness = if let Some(ref_audios) = reference_audios {
            if ref_audios.get(&source_language).is_some() {
                self.get_language_similarity(source_language, target_language)
            } else {
                0.5
            }
        } else {
            self.get_language_similarity(source_language, target_language)
        };
        let sample_factor = (sample_size as f32).ln() / 10.0;
        let max_improvement = 0.4;
        let improvement = max_improvement * (1.0 - (-sample_factor).exp());
        let performance = (baseline_effectiveness + improvement).min(1.0);
        Ok(performance)
    }
    /// Calculate learning efficiency
    fn calculate_learning_efficiency(
        &self,
        performance_by_sample_size: &HashMap<usize, f32>,
    ) -> f32 {
        if performance_by_sample_size.len() < 2 {
            return 0.5;
        }
        let mut sample_sizes: Vec<usize> = performance_by_sample_size.keys().cloned().collect();
        sample_sizes.sort();
        let mut efficiency_sum = 0.0;
        let mut count = 0;
        for i in 1..sample_sizes.len() {
            let prev_size = sample_sizes[i - 1];
            let curr_size = sample_sizes[i];
            let prev_perf = performance_by_sample_size[&prev_size];
            let curr_perf = performance_by_sample_size[&curr_size];
            let perf_improvement = curr_perf - prev_perf;
            let sample_increase = curr_size - prev_size;
            if sample_increase > 0 {
                let efficiency = perf_improvement / (sample_increase as f32 / 100.0);
                efficiency_sum += efficiency.max(0.0);
                count += 1;
            }
        }
        if count > 0 {
            (efficiency_sum / count as f32).min(1.0)
        } else {
            0.5
        }
    }
    /// Calculate sample efficiency
    fn calculate_sample_efficiency(&self, performance_by_sample_size: &HashMap<usize, f32>) -> f32 {
        if performance_by_sample_size.is_empty() {
            return 0.5;
        }
        let mut sorted_samples: Vec<(&usize, &f32)> = performance_by_sample_size.iter().collect();
        sorted_samples.sort_by_key(|(size, _)| *size);
        let target_performance = 0.7;
        let mut efficient_sample_size = None;
        for (size, performance) in sorted_samples {
            if *performance >= target_performance {
                efficient_sample_size = Some(*size);
                break;
            }
        }
        if let Some(size) = efficient_sample_size {
            let max_size = self
                .config
                .few_shot_sample_sizes
                .iter()
                .max()
                .unwrap_or(&50);
            1.0 - (size as f32 / *max_size as f32)
        } else {
            0.3
        }
    }
    /// Calculate adaptation speed
    fn calculate_adaptation_speed(&self, performance_by_sample_size: &HashMap<usize, f32>) -> f32 {
        if performance_by_sample_size.len() < 2 {
            return 0.5;
        }
        let smallest_size = *performance_by_sample_size
            .keys()
            .min()
            .expect("value should be present");
        let largest_size = *performance_by_sample_size
            .keys()
            .max()
            .expect("value should be present");
        let initial_perf = performance_by_sample_size[&smallest_size];
        let final_perf = performance_by_sample_size[&largest_size];
        let improvement = final_perf - initial_perf;
        let size_ratio = largest_size as f32 / smallest_size as f32;
        if size_ratio > 1.0 {
            (improvement / size_ratio.ln()).max(0.0).min(1.0)
        } else {
            0.5
        }
    }
    /// Find minimum samples needed
    fn find_min_samples_needed(&self, performance_by_sample_size: &HashMap<usize, f32>) -> usize {
        let threshold = 0.6;
        let mut sorted_samples: Vec<(&usize, &f32)> = performance_by_sample_size.iter().collect();
        sorted_samples.sort_by_key(|(size, _)| *size);
        for (size, performance) in &sorted_samples {
            if **performance >= threshold {
                return **size;
            }
        }
        sorted_samples.last().map(|(size, _)| **size).unwrap_or(50)
    }
    /// Find saturation point
    fn find_saturation_point(
        &self,
        performance_by_sample_size: &HashMap<usize, f32>,
    ) -> Option<usize> {
        if performance_by_sample_size.len() < 3 {
            return None;
        }
        let mut sorted_samples: Vec<(&usize, &f32)> = performance_by_sample_size.iter().collect();
        sorted_samples.sort_by_key(|(size, _)| *size);
        let improvement_threshold = 0.01;
        for i in 1..sorted_samples.len() {
            let prev_perf = sorted_samples[i - 1].1;
            let curr_perf = sorted_samples[i].1;
            let improvement = curr_perf - prev_perf;
            if improvement < improvement_threshold {
                return Some(*sorted_samples[i - 1].0);
            }
        }
        None
    }
    /// Assess domain adaptation
    pub(super) async fn assess_domain_adaptation(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<HashMap<LanguageCode, DomainAdaptationResult>> {
        let mut domain_adaptations = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != source_language {
                let adaptation_result = self.calculate_domain_adaptation(
                    source_language,
                    *target_language,
                    target_audio,
                    reference_audios,
                )?;
                domain_adaptations.insert(*target_language, adaptation_result);
            }
        }
        Ok(domain_adaptations)
    }
    /// Calculate domain adaptation
    pub(crate) fn calculate_domain_adaptation(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        target_audio: &AudioBuffer,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
    ) -> EvaluationResult<DomainAdaptationResult> {
        let domain_similarity = self.get_language_similarity(source_language, target_language);
        let adaptation_score = if let Some(ref_audios) = reference_audios {
            if let Some(ref_audio) = ref_audios.get(&source_language) {
                self.calculate_acoustic_similarity(ref_audio, target_audio)?
            } else {
                domain_similarity
            }
        } else {
            domain_similarity
        };
        let adaptation_efficiency = adaptation_score / domain_similarity.max(0.1);
        let domain_gap = 1.0 - domain_similarity;
        let adaptation_challenges =
            self.generate_adaptation_challenges(source_language, target_language, domain_gap);
        let adaptation_recommendations = self.generate_adaptation_recommendations(
            source_language,
            target_language,
            &adaptation_challenges,
        );
        Ok(DomainAdaptationResult {
            adaptation_score,
            domain_similarity,
            adaptation_efficiency,
            domain_gap,
            adaptation_challenges,
            adaptation_recommendations,
        })
    }
    /// Generate adaptation challenges
    fn generate_adaptation_challenges(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        domain_gap: f32,
    ) -> Vec<AdaptationChallenge> {
        let mut challenges = Vec::new();
        if domain_gap > 0.3 {
            challenges.push(AdaptationChallenge {
                challenge_type: AdaptationChallengeType::PhoneticDivergence,
                severity: domain_gap * 0.8,
                description: format!(
                    "Significant phonetic differences between {:?} and {:?}",
                    source_language, target_language
                ),
                suggested_solutions: vec![
                    "Implement phonetic adaptation layers".to_string(),
                    "Use cross-lingual phoneme embeddings".to_string(),
                    "Apply phonetic distance regularization".to_string(),
                ],
            });
        }
        if domain_gap > 0.4 {
            challenges.push(AdaptationChallenge {
                challenge_type: AdaptationChallengeType::ProsodicMismatch,
                severity: domain_gap * 0.6,
                description: format!(
                    "Prosodic patterns mismatch between {:?} and {:?}",
                    source_language, target_language
                ),
                suggested_solutions: vec![
                    "Implement prosodic style transfer".to_string(),
                    "Use rhythm and stress adaptation".to_string(),
                    "Apply intonation pattern alignment".to_string(),
                ],
            });
        }
        if domain_gap > 0.5 {
            challenges.push(AdaptationChallenge {
                challenge_type: AdaptationChallengeType::CulturalDifferences,
                severity: domain_gap * 0.7,
                description: format!(
                    "Cultural communication differences between {:?} and {:?}",
                    source_language, target_language
                ),
                suggested_solutions: vec![
                    "Implement cultural adaptation modules".to_string(),
                    "Use culturally-aware training data".to_string(),
                    "Apply cultural style transfer techniques".to_string(),
                ],
            });
        }
        challenges
    }
    /// Generate adaptation recommendations
    fn generate_adaptation_recommendations(
        &self,
        _source_language: LanguageCode,
        _target_language: LanguageCode,
        challenges: &[AdaptationChallenge],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        for challenge in challenges {
            match challenge.challenge_type {
                AdaptationChallengeType::PhoneticDivergence => {
                    recommendations.push("Implement cross-lingual phoneme mapping".to_string());
                    recommendations.push("Use phonetic adaptation layers".to_string());
                }
                AdaptationChallengeType::ProsodicMismatch => {
                    recommendations.push("Apply prosodic style transfer".to_string());
                    recommendations.push("Use rhythm and stress adaptation".to_string());
                }
                AdaptationChallengeType::CulturalDifferences => {
                    recommendations.push("Implement cultural adaptation modules".to_string());
                    recommendations.push("Use culturally-aware training strategies".to_string());
                }
                AdaptationChallengeType::AcousticIncompatibility => {
                    recommendations.push("Apply acoustic domain adaptation".to_string());
                    recommendations
                        .push("Use adversarial training for acoustic alignment".to_string());
                }
                AdaptationChallengeType::LimitedTrainingData => {
                    recommendations.push("Implement few-shot learning techniques".to_string());
                    recommendations.push("Use data augmentation strategies".to_string());
                }
                AdaptationChallengeType::NegativeInterference => {
                    recommendations.push("Apply negative transfer mitigation".to_string());
                    recommendations.push("Use selective transfer learning".to_string());
                }
            }
        }
        recommendations
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    /// Detect negative transfer
    pub(super) async fn detect_negative_transfer(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
    ) -> EvaluationResult<NegativeTransferDetectionResult> {
        let mut negative_transfer_detected = false;
        let mut negative_transfer_severity: f32 = 0.0;
        let mut affected_language_pairs = Vec::new();
        let mut negative_transfer_sources = Vec::new();
        let mut performance_degradation = HashMap::new();
        for (target_language, _target_audio) in target_audios {
            if *target_language != source_language {
                let effectiveness = transfer_effectiveness.get(target_language).unwrap_or(&0.5);
                let baseline_similarity =
                    self.get_language_similarity(source_language, *target_language);
                if *effectiveness < baseline_similarity - 0.1 {
                    negative_transfer_detected = true;
                    let severity = baseline_similarity - effectiveness;
                    negative_transfer_severity = negative_transfer_severity.max(severity);
                    affected_language_pairs.push((source_language, *target_language));
                    let sources = self.identify_negative_transfer_sources(
                        source_language,
                        *target_language,
                        severity,
                    );
                    negative_transfer_sources.extend(sources);
                    performance_degradation.insert(*target_language, severity);
                }
            }
        }
        let mitigation_strategies = self.generate_mitigation_strategies(&negative_transfer_sources);
        Ok(NegativeTransferDetectionResult {
            negative_transfer_detected,
            negative_transfer_severity,
            affected_language_pairs,
            negative_transfer_sources,
            mitigation_strategies,
            performance_degradation,
        })
    }
    /// Identify negative transfer sources
    pub(crate) fn identify_negative_transfer_sources(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        severity: f32,
    ) -> Vec<NegativeTransferSource> {
        let mut sources = Vec::new();
        if severity > 0.2 {
            sources.push(NegativeTransferSource {
                source_type: NegativeTransferSourceType::PhoneticInterference,
                source_language,
                target_language,
                interference_magnitude: severity * 0.6,
                description: format!(
                    "Phonetic differences between {:?} and {:?} causing interference",
                    source_language, target_language
                ),
            });
        }
        if severity > 0.15 {
            sources.push(NegativeTransferSource {
                source_type: NegativeTransferSourceType::ProsodicInterference,
                source_language,
                target_language,
                interference_magnitude: severity * 0.4,
                description: format!(
                    "Prosodic patterns from {:?} interfering with {:?}",
                    source_language, target_language
                ),
            });
        }
        if severity > 0.25 {
            sources.push(NegativeTransferSource {
                source_type: NegativeTransferSourceType::CulturalInterference,
                source_language,
                target_language,
                interference_magnitude: severity * 0.5,
                description: format!(
                    "Cultural communication styles causing interference between {:?} and {:?}",
                    source_language, target_language
                ),
            });
        }
        sources
    }
    /// Generate mitigation strategies
    fn generate_mitigation_strategies(
        &self,
        negative_transfer_sources: &[NegativeTransferSource],
    ) -> Vec<String> {
        let mut strategies = Vec::new();
        for source in negative_transfer_sources {
            match source.source_type {
                NegativeTransferSourceType::PhoneticInterference => {
                    strategies.push("Use selective phonetic transfer".to_string());
                    strategies.push("Apply phonetic adaptation regularization".to_string());
                    strategies.push("Implement phonetic distance constraints".to_string());
                }
                NegativeTransferSourceType::ProsodicInterference => {
                    strategies.push("Use prosodic disentanglement techniques".to_string());
                    strategies.push("Apply prosodic style separation".to_string());
                    strategies.push("Implement prosodic adaptation layers".to_string());
                }
                NegativeTransferSourceType::CulturalInterference => {
                    strategies.push("Use cultural adaptation modules".to_string());
                    strategies.push("Apply cultural style disentanglement".to_string());
                    strategies.push("Implement cultural-aware training".to_string());
                }
                NegativeTransferSourceType::AcousticInterference => {
                    strategies.push("Use acoustic domain adaptation".to_string());
                    strategies.push("Apply adversarial acoustic training".to_string());
                    strategies.push("Implement acoustic feature disentanglement".to_string());
                }
                NegativeTransferSourceType::LinguisticInterference => {
                    strategies.push("Use linguistic feature separation".to_string());
                    strategies.push("Apply linguistic adaptation constraints".to_string());
                    strategies.push("Implement linguistic-aware transfer".to_string());
                }
                NegativeTransferSourceType::ModelCapacityLimitations => {
                    strategies.push("Increase model capacity".to_string());
                    strategies.push("Use modular architectures".to_string());
                    strategies.push("Implement capacity-aware training".to_string());
                }
            }
        }
        strategies
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    /// Generate transfer optimization recommendations
    pub(super) async fn generate_transfer_optimization_recommendations(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        knowledge_transfer_assessment: &KnowledgeTransferAssessment,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
        stability_analysis: &TransferStabilityAnalysis,
        few_shot_performance: &HashMap<LanguageCode, FewShotPerformance>,
        domain_adaptation: &HashMap<LanguageCode, DomainAdaptationResult>,
        negative_transfer_detection: &NegativeTransferDetectionResult,
    ) -> EvaluationResult<Vec<TransferOptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        if knowledge_transfer_assessment.overall_knowledge_transfer < 0.7 {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type:
                    TransferOptimizationRecommendationType::SourceLanguageSelection,
                priority: RecommendationPriority::High,
                target_languages: target_audios.keys().cloned().collect(),
                description: "Improve source language selection for better knowledge transfer"
                    .to_string(),
                expected_improvement: 0.2,
                implementation_effort: ImplementationEffort::Medium,
                parameters: HashMap::from([
                    ("min_similarity_threshold".to_string(), 0.6),
                    ("knowledge_transfer_weight".to_string(), 0.3),
                ]),
            });
        }
        let avg_effectiveness = transfer_effectiveness.values().sum::<f32>()
            / transfer_effectiveness.len().max(1) as f32;
        if avg_effectiveness < 0.6 {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type: TransferOptimizationRecommendationType::MultiTaskLearning,
                priority: RecommendationPriority::High,
                target_languages: transfer_effectiveness.keys().cloned().collect(),
                description: "Implement multi-task learning to improve transfer effectiveness"
                    .to_string(),
                expected_improvement: 0.25,
                implementation_effort: ImplementationEffort::High,
                parameters: HashMap::from([
                    ("task_weight_balance".to_string(), 0.5),
                    ("shared_encoder_layers".to_string(), 0.7),
                ]),
            });
        }
        if stability_analysis.stability_score < 0.7 {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type: TransferOptimizationRecommendationType::TransferTiming,
                priority: RecommendationPriority::Medium,
                target_languages: target_audios.keys().cloned().collect(),
                description: "Optimize transfer timing for better stability".to_string(),
                expected_improvement: 0.15,
                implementation_effort: ImplementationEffort::Medium,
                parameters: HashMap::from([
                    ("early_stopping_patience".to_string(), 10.0),
                    ("transfer_warmup_epochs".to_string(), 5.0),
                ]),
            });
        }
        let mut poor_few_shot_languages = Vec::new();
        for (language, performance) in few_shot_performance {
            if performance.learning_efficiency < 0.6 {
                poor_few_shot_languages.push(*language);
            }
        }
        if !poor_few_shot_languages.is_empty() {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type: TransferOptimizationRecommendationType::FewShotLearning,
                priority: RecommendationPriority::Medium,
                target_languages: poor_few_shot_languages,
                description: "Enhance few-shot learning capabilities".to_string(),
                expected_improvement: 0.2,
                implementation_effort: ImplementationEffort::Medium,
                parameters: HashMap::from([
                    ("meta_learning_rate".to_string(), 0.01),
                    ("support_set_size".to_string(), 5.0),
                ]),
            });
        }
        let mut poor_adaptation_languages = Vec::new();
        for (language, adaptation) in domain_adaptation {
            if adaptation.adaptation_score < 0.6 {
                poor_adaptation_languages.push(*language);
            }
        }
        if !poor_adaptation_languages.is_empty() {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type: TransferOptimizationRecommendationType::DomainAdaptation,
                priority: RecommendationPriority::High,
                target_languages: poor_adaptation_languages,
                description: "Improve domain adaptation mechanisms".to_string(),
                expected_improvement: 0.3,
                implementation_effort: ImplementationEffort::High,
                parameters: HashMap::from([
                    ("adaptation_learning_rate".to_string(), 0.005),
                    ("domain_classifier_weight".to_string(), 0.1),
                ]),
            });
        }
        if negative_transfer_detection.negative_transfer_detected {
            recommendations.push(TransferOptimizationRecommendation {
                recommendation_type:
                    TransferOptimizationRecommendationType::NegativeTransferReduction,
                priority: RecommendationPriority::Critical,
                target_languages: negative_transfer_detection
                    .affected_language_pairs
                    .iter()
                    .map(|(_, target)| *target)
                    .collect(),
                description: "Reduce negative transfer effects".to_string(),
                expected_improvement: 0.35,
                implementation_effort: ImplementationEffort::High,
                parameters: HashMap::from([
                    ("negative_transfer_weight".to_string(), 0.2),
                    ("interference_threshold".to_string(), 0.1),
                ]),
            });
        }
        Ok(recommendations)
    }
    /// Calculate transfer learning metrics
    pub(super) fn calculate_transfer_learning_metrics(
        &self,
        knowledge_transfer_assessment: &KnowledgeTransferAssessment,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
        stability_analysis: &TransferStabilityAnalysis,
        few_shot_performance: &HashMap<LanguageCode, FewShotPerformance>,
        domain_adaptation: &HashMap<LanguageCode, DomainAdaptationResult>,
        negative_transfer_detection: &NegativeTransferDetectionResult,
    ) -> TransferLearningMetrics {
        let successful_transfers = transfer_effectiveness
            .values()
            .filter(|&&v| v > 0.6)
            .count();
        let total_transfers = transfer_effectiveness.len().max(1);
        let transfer_success_rate = successful_transfers as f32 / total_transfers as f32;
        let average_transfer_effectiveness = if transfer_effectiveness.is_empty() {
            0.5
        } else {
            transfer_effectiveness.values().sum::<f32>() / transfer_effectiveness.len() as f32
        };
        let transfer_efficiency = knowledge_transfer_assessment.transfer_efficiency;
        let cross_linguistic_consistency = stability_analysis.cross_language_consistency;
        let knowledge_preservation = knowledge_transfer_assessment.overall_knowledge_transfer;
        let adaptation_speed = if few_shot_performance.is_empty() {
            0.5
        } else {
            few_shot_performance
                .values()
                .map(|p| p.adaptation_speed)
                .sum::<f32>()
                / few_shot_performance.len() as f32
        };
        let negative_transfer_rate = if negative_transfer_detection.negative_transfer_detected {
            negative_transfer_detection.negative_transfer_severity
        } else {
            0.0
        };
        let overall_transfer_quality = (transfer_success_rate * 0.3
            + average_transfer_effectiveness * 0.25
            + transfer_efficiency * 0.2
            + cross_linguistic_consistency * 0.15
            + knowledge_preservation * 0.1)
            * (1.0 - negative_transfer_rate * 0.5);
        TransferLearningMetrics {
            transfer_success_rate,
            average_transfer_effectiveness,
            transfer_efficiency,
            cross_linguistic_consistency,
            knowledge_preservation,
            adaptation_speed,
            negative_transfer_rate,
            overall_transfer_quality,
        }
    }
    /// Build language transfer matrix
    pub(super) fn build_language_transfer_matrix(
        &self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
    ) -> HashMap<(LanguageCode, LanguageCode), f32> {
        let mut matrix = HashMap::new();
        for (target_language, _) in target_audios {
            if *target_language != source_language {
                let effectiveness = transfer_effectiveness.get(target_language).unwrap_or(&0.5);
                matrix.insert((source_language, *target_language), *effectiveness);
            }
        }
        let target_languages: Vec<LanguageCode> = target_audios.keys().cloned().collect();
        for &lang1 in &target_languages {
            for &lang2 in &target_languages {
                if lang1 != lang2 && lang1 != source_language && lang2 != source_language {
                    let similarity = self.get_language_similarity(lang1, lang2);
                    matrix.insert((lang1, lang2), similarity);
                }
            }
        }
        matrix
    }
    /// Identify problematic transfer pairs
    pub(super) fn identify_problematic_transfer_pairs(
        &self,
        source_language: LanguageCode,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
        stability_analysis: &TransferStabilityAnalysis,
        negative_transfer_detection: &NegativeTransferDetectionResult,
    ) -> Vec<ProblematicTransferPair> {
        let mut problematic_pairs = Vec::new();
        for (target_language, effectiveness) in transfer_effectiveness {
            let mut problem_types = Vec::new();
            let mut problem_severity: f32 = 0.0;
            if *effectiveness < 0.5 {
                problem_types.push(TransferProblemType::PoorTransferEffectiveness);
                problem_severity = problem_severity.max(1.0 - effectiveness);
            }
            if let Some(stability_metrics) = stability_analysis
                .language_stability_metrics
                .get(target_language)
            {
                if stability_metrics.stability_coefficient < 0.6 {
                    problem_types.push(TransferProblemType::UnstableConvergence);
                    problem_severity =
                        problem_severity.max(1.0 - stability_metrics.stability_coefficient);
                }
            }
            if negative_transfer_detection
                .affected_language_pairs
                .contains(&(source_language, *target_language))
            {
                problem_types.push(TransferProblemType::NegativeTransfer);
                problem_severity =
                    problem_severity.max(negative_transfer_detection.negative_transfer_severity);
            }
            if !problem_types.is_empty() {
                problematic_pairs.push(ProblematicTransferPair {
                    source_language,
                    target_language: *target_language,
                    problem_severity,
                    problem_types,
                    problem_description: format!(
                        "Transfer from {:?} to {:?} shows multiple issues",
                        source_language, target_language
                    ),
                    improvement_strategies: vec![
                        "Improve training data quality".to_string(),
                        "Implement better transfer learning techniques".to_string(),
                        "Use language-specific adaptation".to_string(),
                        "Apply negative transfer mitigation".to_string(),
                    ],
                });
            }
        }
        problematic_pairs
    }
    /// Calculate overall transfer score
    pub(super) fn calculate_overall_transfer_score(
        &self,
        knowledge_transfer_assessment: &KnowledgeTransferAssessment,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
        stability_analysis: &TransferStabilityAnalysis,
        few_shot_performance: &HashMap<LanguageCode, FewShotPerformance>,
        domain_adaptation: &HashMap<LanguageCode, DomainAdaptationResult>,
    ) -> f32 {
        let knowledge_score = knowledge_transfer_assessment.overall_knowledge_transfer;
        let effectiveness_score = if transfer_effectiveness.is_empty() {
            0.5
        } else {
            transfer_effectiveness.values().sum::<f32>() / transfer_effectiveness.len() as f32
        };
        let stability_score = stability_analysis.stability_score;
        let few_shot_score = if few_shot_performance.is_empty() {
            0.5
        } else {
            few_shot_performance
                .values()
                .map(|p| p.learning_efficiency)
                .sum::<f32>()
                / few_shot_performance.len() as f32
        };
        let adaptation_score = if domain_adaptation.is_empty() {
            0.5
        } else {
            domain_adaptation
                .values()
                .map(|d| d.adaptation_score)
                .sum::<f32>()
                / domain_adaptation.len() as f32
        };
        let overall_score = knowledge_score * self.config.knowledge_transfer_weight
            + effectiveness_score * self.config.transfer_effectiveness_weight
            + stability_score * self.config.stability_analysis_weight
            + few_shot_score * self.config.few_shot_evaluation_weight
            + adaptation_score * self.config.domain_adaptation_weight;
        overall_score.max(0.0).min(1.0)
    }
    /// Calculate evaluation confidence
    pub(super) fn calculate_evaluation_confidence(
        &self,
        knowledge_transfer_assessment: &KnowledgeTransferAssessment,
        transfer_effectiveness: &HashMap<LanguageCode, f32>,
        stability_analysis: &TransferStabilityAnalysis,
        few_shot_performance: &HashMap<LanguageCode, FewShotPerformance>,
        domain_adaptation: &HashMap<LanguageCode, DomainAdaptationResult>,
    ) -> f32 {
        let mut confidence_factors = Vec::new();
        confidence_factors.push(knowledge_transfer_assessment.transfer_consistency);
        if !transfer_effectiveness.is_empty() {
            let effectiveness_consistency = self.calculate_transfer_consistency(
                &transfer_effectiveness.values().cloned().collect::<Vec<_>>(),
            );
            confidence_factors.push(effectiveness_consistency);
        }
        confidence_factors.push(
            stability_analysis
                .convergence_analysis
                .convergence_reliability,
        );
        if !few_shot_performance.is_empty() {
            let few_shot_consistency = few_shot_performance
                .values()
                .map(|p| p.learning_efficiency)
                .collect::<Vec<_>>();
            confidence_factors.push(self.calculate_transfer_consistency(&few_shot_consistency));
        }
        if !domain_adaptation.is_empty() {
            let adaptation_consistency = domain_adaptation
                .values()
                .map(|d| d.adaptation_score)
                .collect::<Vec<_>>();
            confidence_factors.push(self.calculate_transfer_consistency(&adaptation_consistency));
        }
        let overall_confidence = if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32
        };
        overall_confidence.max(0.1).min(1.0)
    }
}
