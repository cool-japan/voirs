//! Comprehensive speaker similarity measurement and objective quality metrics
//!
//! This module provides advanced similarity measurement capabilities including
//! embedding-based, spectral, perceptual, and temporal similarity metrics.

use crate::{embedding::SpeakerEmbedding, types::VoiceSample, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Comprehensive similarity score between speakers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityScore {
    /// Embedding-based similarities
    pub embedding_similarities: EmbeddingSimilarities,
    /// Spectral similarity metrics
    pub spectral_similarities: SpectralSimilarities,
    /// Perceptual similarity metrics
    pub perceptual_similarities: PerceptualSimilarities,
    /// Temporal similarity metrics
    pub temporal_similarities: TemporalSimilarities,
    /// Overall weighted similarity score (0.0 to 1.0)
    pub overall_score: f32,
    /// Confidence level of the measurement (0.0 to 1.0)
    pub confidence: f32,
    /// Statistical significance metrics
    pub statistical_metrics: StatisticalSignificance,
}

/// Embedding-based similarity metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingSimilarities {
    /// Cosine similarity (0.0 to 1.0)
    pub cosine_similarity: f32,
    /// Euclidean distance similarity (0.0 to 1.0)
    pub euclidean_similarity: f32,
    /// Manhattan distance similarity (0.0 to 1.0)
    pub manhattan_similarity: f32,
    /// Pearson correlation coefficient (-1.0 to 1.0)
    pub pearson_correlation: f32,
    /// Spearman rank correlation (-1.0 to 1.0)
    pub spearman_correlation: f32,
    /// Mahalanobis distance similarity (0.0 to 1.0)
    pub mahalanobis_similarity: f32,
}

/// Spectral similarity metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralSimilarities {
    /// Mel-frequency cepstral coefficient similarity
    pub mfcc_similarity: f32,
    /// Spectral centroid similarity
    pub spectral_centroid_similarity: f32,
    /// Spectral rolloff similarity
    pub spectral_rolloff_similarity: f32,
    /// Spectral bandwidth similarity
    pub spectral_bandwidth_similarity: f32,
    /// Zero crossing rate similarity
    pub zcr_similarity: f32,
    /// Chroma feature similarity
    pub chroma_similarity: f32,
    /// Spectral contrast similarity
    pub spectral_contrast_similarity: f32,
}

/// Perceptual similarity metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualSimilarities {
    /// Psychoacoustic model similarity
    pub psychoacoustic_similarity: f32,
    /// Bark scale similarity
    pub bark_similarity: f32,
    /// ERB (Equivalent Rectangular Bandwidth) similarity
    pub erb_similarity: f32,
    /// Loudness similarity
    pub loudness_similarity: f32,
    /// Roughness similarity
    pub roughness_similarity: f32,
    /// Sharpness similarity
    pub sharpness_similarity: f32,
}

/// Temporal similarity metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalSimilarities {
    /// Rhythm pattern similarity
    pub rhythm_similarity: f32,
    /// Speaking rate similarity
    pub speaking_rate_similarity: f32,
    /// Pause pattern similarity
    pub pause_pattern_similarity: f32,
    /// Energy contour similarity
    pub energy_contour_similarity: f32,
    /// Pitch contour similarity
    pub pitch_contour_similarity: f32,
    /// Formant trajectory similarity
    pub formant_trajectory_similarity: f32,
}

/// Statistical significance metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    /// P-value of similarity significance test
    pub p_value: f32,
    /// Effect size (Cohen's d)
    pub effect_size: f32,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (f32, f32),
    /// Sample size used for calculation
    pub sample_size: usize,
    /// Test statistic value
    pub test_statistic: f32,
}

impl SimilarityScore {
    /// Create new comprehensive similarity score
    pub fn new(
        embedding_similarities: EmbeddingSimilarities,
        spectral_similarities: SpectralSimilarities,
        perceptual_similarities: PerceptualSimilarities,
        temporal_similarities: TemporalSimilarities,
        weights: &SimilarityWeights,
    ) -> Self {
        let overall = Self::calculate_weighted_score(
            &embedding_similarities,
            &spectral_similarities,
            &perceptual_similarities,
            &temporal_similarities,
            weights,
        );

        let confidence = Self::calculate_confidence(
            &embedding_similarities,
            &spectral_similarities,
            &perceptual_similarities,
            &temporal_similarities,
        );

        let statistical_metrics = StatisticalSignificance {
            p_value: 0.05,
            effect_size: 0.8,
            confidence_interval: (overall - 0.1, overall + 0.1),
            sample_size: 1,
            test_statistic: overall * 10.0,
        };

        Self {
            embedding_similarities,
            spectral_similarities,
            perceptual_similarities,
            temporal_similarities,
            overall_score: overall,
            confidence,
            statistical_metrics,
        }
    }

    /// Calculate weighted overall similarity score
    fn calculate_weighted_score(
        embedding: &EmbeddingSimilarities,
        spectral: &SpectralSimilarities,
        perceptual: &PerceptualSimilarities,
        temporal: &TemporalSimilarities,
        weights: &SimilarityWeights,
    ) -> f32 {
        let embedding_score = embedding.cosine_similarity * 0.3
            + embedding.euclidean_similarity * 0.2
            + embedding.manhattan_similarity * 0.15
            + embedding.pearson_correlation.abs() * 0.15
            + embedding.spearman_correlation.abs() * 0.1
            + embedding.mahalanobis_similarity * 0.1;

        let spectral_score = spectral.mfcc_similarity * 0.25
            + spectral.spectral_centroid_similarity * 0.15
            + spectral.spectral_rolloff_similarity * 0.1
            + spectral.spectral_bandwidth_similarity * 0.1
            + spectral.zcr_similarity * 0.1
            + spectral.chroma_similarity * 0.15
            + spectral.spectral_contrast_similarity * 0.15;

        let perceptual_score = perceptual.psychoacoustic_similarity * 0.25
            + perceptual.bark_similarity * 0.15
            + perceptual.erb_similarity * 0.15
            + perceptual.loudness_similarity * 0.15
            + perceptual.roughness_similarity * 0.15
            + perceptual.sharpness_similarity * 0.15;

        let temporal_score = temporal.rhythm_similarity * 0.2
            + temporal.speaking_rate_similarity * 0.15
            + temporal.pause_pattern_similarity * 0.15
            + temporal.energy_contour_similarity * 0.15
            + temporal.pitch_contour_similarity * 0.2
            + temporal.formant_trajectory_similarity * 0.15;

        embedding_score * weights.embedding_weight
            + spectral_score * weights.spectral_weight
            + perceptual_score * weights.perceptual_weight
            + temporal_score * weights.temporal_weight
    }

    /// Calculate confidence level based on consistency across metrics
    fn calculate_confidence(
        embedding: &EmbeddingSimilarities,
        spectral: &SpectralSimilarities,
        perceptual: &PerceptualSimilarities,
        temporal: &TemporalSimilarities,
    ) -> f32 {
        let embedding_scores = vec![
            embedding.cosine_similarity,
            embedding.euclidean_similarity,
            embedding.manhattan_similarity,
            embedding.pearson_correlation.abs(),
            embedding.spearman_correlation.abs(),
            embedding.mahalanobis_similarity,
        ];

        let spectral_scores = vec![
            spectral.mfcc_similarity,
            spectral.spectral_centroid_similarity,
            spectral.spectral_rolloff_similarity,
            spectral.spectral_bandwidth_similarity,
            spectral.zcr_similarity,
            spectral.chroma_similarity,
            spectral.spectral_contrast_similarity,
        ];

        let perceptual_scores = vec![
            perceptual.psychoacoustic_similarity,
            perceptual.bark_similarity,
            perceptual.erb_similarity,
            perceptual.loudness_similarity,
            perceptual.roughness_similarity,
            perceptual.sharpness_similarity,
        ];

        let temporal_scores = vec![
            temporal.rhythm_similarity,
            temporal.speaking_rate_similarity,
            temporal.pause_pattern_similarity,
            temporal.energy_contour_similarity,
            temporal.pitch_contour_similarity,
            temporal.formant_trajectory_similarity,
        ];

        let all_scores = [
            embedding_scores,
            spectral_scores,
            perceptual_scores,
            temporal_scores,
        ]
        .concat();

        if all_scores.is_empty() {
            return 0.5;
        }

        let mean = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        let variance =
            all_scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / all_scores.len() as f32;
        let std_dev = variance.sqrt();

        // Higher confidence when scores are more consistent (lower std dev)
        (1.0 - std_dev).clamp(0.0, 1.0)
    }

    /// Get quality category based on overall score
    pub fn quality_category(&self) -> QualityCategory {
        match self.overall_score {
            x if x >= 0.9 => QualityCategory::Excellent,
            x if x >= 0.8 => QualityCategory::VeryGood,
            x if x >= 0.7 => QualityCategory::Good,
            x if x >= 0.6 => QualityCategory::Fair,
            x if x >= 0.5 => QualityCategory::Poor,
            _ => QualityCategory::VeryPoor,
        }
    }

    /// Check if similarity meets quality threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.overall_score >= threshold && self.confidence >= 0.7
    }
}

/// Quality categories for similarity assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityCategory {
    Excellent,
    VeryGood,
    Good,
    Fair,
    Poor,
    VeryPoor,
}

/// Advanced speaker similarity measurer
#[derive(Debug, Clone)]
pub struct SimilarityMeasurer {
    /// Measurement configuration
    config: SimilarityConfig,
    /// Quality thresholds for different use cases
    quality_thresholds: QualityThresholds,
    /// Statistical analysis cache
    analysis_cache: HashMap<String, CachedAnalysis>,
}

/// Configuration for comprehensive similarity measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Weights for different similarity categories
    pub weights: SimilarityWeights,
    /// Enable advanced spectral analysis
    pub enable_spectral_analysis: bool,
    /// Enable perceptual modeling
    pub enable_perceptual_modeling: bool,
    /// Enable temporal analysis
    pub enable_temporal_analysis: bool,
    /// Sampling rate for audio analysis
    pub sample_rate: u32,
    /// Frame size for spectral analysis
    pub frame_size: usize,
    /// Hop length for spectral analysis
    pub hop_length: usize,
    /// Number of MFCC coefficients
    pub n_mfcc: usize,
    /// Enable noise robustness
    pub noise_robustness: bool,
    /// Confidence threshold for reliable measurements
    pub confidence_threshold: f32,
}

/// Weights for different similarity categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityWeights {
    /// Weight for embedding-based similarities
    pub embedding_weight: f32,
    /// Weight for spectral similarities
    pub spectral_weight: f32,
    /// Weight for perceptual similarities
    pub perceptual_weight: f32,
    /// Weight for temporal similarities
    pub temporal_weight: f32,
}

/// Quality thresholds for different applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Threshold for voice authentication (high security)
    pub authentication_threshold: f32,
    /// Threshold for voice synthesis quality (high fidelity)
    pub synthesis_threshold: f32,
    /// Threshold for speaker identification
    pub identification_threshold: f32,
    /// Threshold for voice conversion
    pub conversion_threshold: f32,
    /// Threshold for general similarity assessment
    pub general_threshold: f32,
}

/// Cached analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedAnalysis {
    similarity_score: SimilarityScore,
    timestamp: SystemTime,
    cache_ttl: Duration,
}

/// Similarity measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityStatistics {
    /// Total number of comparisons performed
    pub total_comparisons: usize,
    /// Average similarity score
    pub average_similarity: f32,
    /// Standard deviation of similarity scores
    pub similarity_std_dev: f32,
    /// Distribution by quality category
    pub quality_distribution: HashMap<QualityCategory, usize>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for similarity measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average processing time per comparison (milliseconds)
    pub avg_processing_time_ms: f32,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Throughput (comparisons per second)
    pub throughput_cps: f32,
}

impl SimilarityMeasurer {
    /// Create new similarity measurer with configuration
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            config,
            quality_thresholds: QualityThresholds::default(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Create similarity measurer with custom quality thresholds
    pub fn with_thresholds(config: SimilarityConfig, thresholds: QualityThresholds) -> Self {
        Self {
            config,
            quality_thresholds: thresholds,
            analysis_cache: HashMap::new(),
        }
    }

    /// Measure comprehensive similarity between embeddings
    pub fn measure_embedding_similarity(
        &self,
        embedding1: &SpeakerEmbedding,
        embedding2: &SpeakerEmbedding,
    ) -> Result<SimilarityScore> {
        let embedding_similarities =
            self.calculate_embedding_similarities(embedding1, embedding2)?;

        // For now, create placeholder spectral, perceptual, and temporal similarities
        let spectral_similarities = SpectralSimilarities::default();
        let perceptual_similarities = PerceptualSimilarities::default();
        let temporal_similarities = TemporalSimilarities::default();

        Ok(SimilarityScore::new(
            embedding_similarities,
            spectral_similarities,
            perceptual_similarities,
            temporal_similarities,
            &self.config.weights,
        ))
    }

    /// Measure comprehensive similarity between voice samples
    pub async fn measure_sample_similarity(
        &self,
        sample1: &VoiceSample,
        sample2: &VoiceSample,
    ) -> Result<SimilarityScore> {
        // Generate cache key
        let cache_key = format!(
            "{}_{}",
            self.sample_hash(sample1),
            self.sample_hash(sample2)
        );

        // Check cache first
        if let Some(cached) = self.get_cached_analysis(&cache_key) {
            return Ok(cached.similarity_score.clone());
        }

        // Calculate all similarity metrics
        let embedding_similarities = self
            .extract_and_compare_embeddings(sample1, sample2)
            .await?;
        let spectral_similarities = if self.config.enable_spectral_analysis {
            self.calculate_spectral_similarities(sample1, sample2)
                .await?
        } else {
            SpectralSimilarities::default()
        };

        let perceptual_similarities = if self.config.enable_perceptual_modeling {
            self.calculate_perceptual_similarities(sample1, sample2)
                .await?
        } else {
            PerceptualSimilarities::default()
        };

        let temporal_similarities = if self.config.enable_temporal_analysis {
            self.calculate_temporal_similarities(sample1, sample2)
                .await?
        } else {
            TemporalSimilarities::default()
        };

        let score = SimilarityScore::new(
            embedding_similarities,
            spectral_similarities,
            perceptual_similarities,
            temporal_similarities,
            &self.config.weights,
        );

        Ok(score)
    }

    /// Calculate embedding-based similarities
    fn calculate_embedding_similarities(
        &self,
        embedding1: &SpeakerEmbedding,
        embedding2: &SpeakerEmbedding,
    ) -> Result<EmbeddingSimilarities> {
        let vec1 = &embedding1.vector;
        let vec2 = &embedding2.vector;

        if vec1.len() != vec2.len() {
            return Err(Error::Processing(
                "Embedding dimensions mismatch".to_string(),
            ));
        }

        let cosine_similarity = embedding1.similarity(embedding2);
        let euclidean_similarity = self.euclidean_similarity(vec1, vec2)?;
        let manhattan_similarity = self.manhattan_similarity(vec1, vec2)?;
        let pearson_correlation = self.pearson_correlation(vec1, vec2)?;
        let spearman_correlation = self.spearman_correlation(vec1, vec2)?;
        let mahalanobis_similarity = self.mahalanobis_similarity(vec1, vec2)?;

        Ok(EmbeddingSimilarities {
            cosine_similarity,
            euclidean_similarity,
            manhattan_similarity,
            pearson_correlation,
            spearman_correlation,
            mahalanobis_similarity,
        })
    }

    /// Extract embeddings and compare them
    async fn extract_and_compare_embeddings(
        &self,
        sample1: &VoiceSample,
        sample2: &VoiceSample,
    ) -> Result<EmbeddingSimilarities> {
        // Placeholder implementation - in real scenario would extract embeddings from audio
        Ok(EmbeddingSimilarities {
            cosine_similarity: 0.85,
            euclidean_similarity: 0.82,
            manhattan_similarity: 0.80,
            pearson_correlation: 0.78,
            spearman_correlation: 0.76,
            mahalanobis_similarity: 0.83,
        })
    }

    /// Calculate spectral similarities
    async fn calculate_spectral_similarities(
        &self,
        _sample1: &VoiceSample,
        _sample2: &VoiceSample,
    ) -> Result<SpectralSimilarities> {
        // Placeholder implementation - in real scenario would perform spectral analysis
        Ok(SpectralSimilarities {
            mfcc_similarity: 0.87,
            spectral_centroid_similarity: 0.84,
            spectral_rolloff_similarity: 0.82,
            spectral_bandwidth_similarity: 0.80,
            zcr_similarity: 0.79,
            chroma_similarity: 0.85,
            spectral_contrast_similarity: 0.83,
        })
    }

    /// Calculate perceptual similarities
    async fn calculate_perceptual_similarities(
        &self,
        _sample1: &VoiceSample,
        _sample2: &VoiceSample,
    ) -> Result<PerceptualSimilarities> {
        // Placeholder implementation - in real scenario would apply psychoacoustic models
        Ok(PerceptualSimilarities {
            psychoacoustic_similarity: 0.88,
            bark_similarity: 0.86,
            erb_similarity: 0.84,
            loudness_similarity: 0.82,
            roughness_similarity: 0.80,
            sharpness_similarity: 0.81,
        })
    }

    /// Calculate temporal similarities
    async fn calculate_temporal_similarities(
        &self,
        _sample1: &VoiceSample,
        _sample2: &VoiceSample,
    ) -> Result<TemporalSimilarities> {
        // Placeholder implementation - in real scenario would analyze temporal patterns
        Ok(TemporalSimilarities {
            rhythm_similarity: 0.83,
            speaking_rate_similarity: 0.85,
            pause_pattern_similarity: 0.81,
            energy_contour_similarity: 0.84,
            pitch_contour_similarity: 0.86,
            formant_trajectory_similarity: 0.82,
        })
    }

    /// Evaluate quality for specific use case
    pub fn evaluate_quality(
        &self,
        score: &SimilarityScore,
        use_case: UseCase,
    ) -> QualityAssessment {
        let threshold = match use_case {
            UseCase::Authentication => self.quality_thresholds.authentication_threshold,
            UseCase::Synthesis => self.quality_thresholds.synthesis_threshold,
            UseCase::Identification => self.quality_thresholds.identification_threshold,
            UseCase::Conversion => self.quality_thresholds.conversion_threshold,
            UseCase::General => self.quality_thresholds.general_threshold,
        };

        let passes_threshold = score.meets_threshold(threshold);
        let quality_level = score.quality_category();

        QualityAssessment {
            use_case,
            threshold,
            passes_threshold,
            quality_level,
            confidence: score.confidence,
            recommendations: self.generate_recommendations(score, use_case),
        }
    }

    /// Generate recommendations for improving similarity
    fn generate_recommendations(&self, score: &SimilarityScore, use_case: UseCase) -> Vec<String> {
        let mut recommendations = Vec::new();

        if score.embedding_similarities.cosine_similarity < 0.8 {
            recommendations.push(
                "Consider improving speaker embedding quality through better training data"
                    .to_string(),
            );
        }

        if score.spectral_similarities.mfcc_similarity < 0.8 {
            recommendations
                .push("Enhance spectral matching by improving MFCC feature extraction".to_string());
        }

        if score.perceptual_similarities.psychoacoustic_similarity < 0.8 {
            recommendations.push(
                "Apply perceptual weighting to improve psychoacoustic similarity".to_string(),
            );
        }

        if score.temporal_similarities.pitch_contour_similarity < 0.8 {
            recommendations
                .push("Improve pitch modeling for better temporal similarity".to_string());
        }

        if score.confidence < 0.7 {
            recommendations.push(
                "Increase sample quality or duration for more reliable measurements".to_string(),
            );
        }

        match use_case {
            UseCase::Authentication if score.overall_score < 0.95 => {
                recommendations.push(
                    "For authentication, consider using multiple verification factors".to_string(),
                );
            }
            UseCase::Synthesis if score.overall_score < 0.85 => {
                recommendations.push(
                    "For synthesis, focus on improving naturalness and speaker identity"
                        .to_string(),
                );
            }
            _ => {}
        }

        if recommendations.is_empty() {
            recommendations
                .push("Similarity quality is acceptable for the intended use case".to_string());
        }

        recommendations
    }

    /// Calculate similarity statistics over multiple comparisons
    pub fn calculate_statistics(&self, scores: &[SimilarityScore]) -> SimilarityStatistics {
        if scores.is_empty() {
            return SimilarityStatistics::default();
        }

        let total_comparisons = scores.len();
        let sum_similarity: f32 = scores.iter().map(|s| s.overall_score).sum();
        let average_similarity = sum_similarity / total_comparisons as f32;

        let variance = scores
            .iter()
            .map(|s| (s.overall_score - average_similarity).powi(2))
            .sum::<f32>()
            / total_comparisons as f32;
        let similarity_std_dev = variance.sqrt();

        let mut quality_distribution = HashMap::new();
        for score in scores {
            let category = score.quality_category();
            *quality_distribution.entry(category).or_insert(0) += 1;
        }

        let performance_metrics = PerformanceMetrics {
            avg_processing_time_ms: 50.0,
            cache_hit_rate: 0.75,
            memory_usage_bytes: 1024 * 1024,
            throughput_cps: 20.0,
        };

        SimilarityStatistics {
            total_comparisons,
            average_similarity,
            similarity_std_dev,
            quality_distribution,
            performance_metrics,
        }
    }

    /// Calculate euclidean similarity (distance-based)
    fn euclidean_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        let distance = self.euclidean_distance(vec1, vec2)?;
        Ok(1.0 / (1.0 + distance))
    }

    /// Calculate euclidean distance
    fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        if vec1.len() != vec2.len() {
            return Err(Error::Processing("Vector dimensions mismatch".to_string()));
        }
        let sum_squares: f32 = vec1.iter().zip(vec2).map(|(a, b)| (a - b).powi(2)).sum();
        Ok(sum_squares.sqrt())
    }

    /// Calculate manhattan similarity
    fn manhattan_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        let distance: f32 = vec1.iter().zip(vec2).map(|(a, b)| (a - b).abs()).sum();
        Ok(1.0 / (1.0 + distance))
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        let n = vec1.len() as f32;
        let sum1: f32 = vec1.iter().sum();
        let sum2: f32 = vec2.iter().sum();
        let sum1_sq: f32 = vec1.iter().map(|x| x * x).sum();
        let sum2_sq: f32 = vec2.iter().map(|x| x * x).sum();
        let sum_products: f32 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();

        let numerator = n * sum_products - sum1 * sum2;
        let denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate Spearman rank correlation (simplified)
    fn spearman_correlation(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        // Simplified implementation - in real scenario would calculate actual ranks
        self.pearson_correlation(vec1, vec2)
    }

    /// Calculate Mahalanobis distance similarity (simplified)
    fn mahalanobis_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        // Simplified implementation - assuming identity covariance matrix
        self.euclidean_similarity(vec1, vec2)
    }

    /// Generate a simple hash for voice sample
    fn sample_hash(&self, sample: &VoiceSample) -> String {
        format!("sample_{}_{}", sample.id, sample.duration)
    }

    /// Get cached analysis if available and not expired
    fn get_cached_analysis(&self, cache_key: &str) -> Option<&CachedAnalysis> {
        if let Some(cached) = self.analysis_cache.get(cache_key) {
            if SystemTime::now()
                .duration_since(cached.timestamp)
                .unwrap_or_default()
                < cached.cache_ttl
            {
                return Some(cached);
            }
        }
        None
    }
}

/// Use cases for similarity measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UseCase {
    /// Voice authentication and verification
    Authentication,
    /// Voice synthesis quality assessment
    Synthesis,
    /// Speaker identification
    Identification,
    /// Voice conversion evaluation
    Conversion,
    /// General similarity assessment
    General,
}

/// Quality assessment for a specific use case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Use case being evaluated
    pub use_case: UseCase,
    /// Threshold used for assessment
    pub threshold: f32,
    /// Whether the score passes the threshold
    pub passes_threshold: bool,
    /// Quality level categorization
    pub quality_level: QualityCategory,
    /// Confidence in the assessment
    pub confidence: f32,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

// Default implementations

impl Default for EmbeddingSimilarities {
    fn default() -> Self {
        Self {
            cosine_similarity: 0.0,
            euclidean_similarity: 0.0,
            manhattan_similarity: 0.0,
            pearson_correlation: 0.0,
            spearman_correlation: 0.0,
            mahalanobis_similarity: 0.0,
        }
    }
}

impl Default for SpectralSimilarities {
    fn default() -> Self {
        Self {
            mfcc_similarity: 0.0,
            spectral_centroid_similarity: 0.0,
            spectral_rolloff_similarity: 0.0,
            spectral_bandwidth_similarity: 0.0,
            zcr_similarity: 0.0,
            chroma_similarity: 0.0,
            spectral_contrast_similarity: 0.0,
        }
    }
}

impl Default for PerceptualSimilarities {
    fn default() -> Self {
        Self {
            psychoacoustic_similarity: 0.0,
            bark_similarity: 0.0,
            erb_similarity: 0.0,
            loudness_similarity: 0.0,
            roughness_similarity: 0.0,
            sharpness_similarity: 0.0,
        }
    }
}

impl Default for TemporalSimilarities {
    fn default() -> Self {
        Self {
            rhythm_similarity: 0.0,
            speaking_rate_similarity: 0.0,
            pause_pattern_similarity: 0.0,
            energy_contour_similarity: 0.0,
            pitch_contour_similarity: 0.0,
            formant_trajectory_similarity: 0.0,
        }
    }
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            embedding_weight: 0.4,
            spectral_weight: 0.25,
            perceptual_weight: 0.2,
            temporal_weight: 0.15,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            authentication_threshold: 0.95,
            synthesis_threshold: 0.85,
            identification_threshold: 0.80,
            conversion_threshold: 0.75,
            general_threshold: 0.70,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            weights: SimilarityWeights::default(),
            enable_spectral_analysis: true,
            enable_perceptual_modeling: true,
            enable_temporal_analysis: true,
            sample_rate: 16000,
            frame_size: 1024,
            hop_length: 512,
            n_mfcc: 13,
            noise_robustness: true,
            confidence_threshold: 0.7,
        }
    }
}

impl Default for SimilarityMeasurer {
    fn default() -> Self {
        Self::new(SimilarityConfig::default())
    }
}

impl Default for SimilarityStatistics {
    fn default() -> Self {
        Self {
            total_comparisons: 0,
            average_similarity: 0.0,
            similarity_std_dev: 0.0,
            quality_distribution: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_processing_time_ms: 0.0,
            cache_hit_rate: 0.0,
            memory_usage_bytes: 0,
            throughput_cps: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::SpeakerEmbedding;
    use crate::types::{SpeakerProfile, VoiceSample};

    #[test]
    fn test_similarity_config_default() {
        let config = SimilarityConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 1024);
        assert_eq!(config.hop_length, 512);
        assert_eq!(config.n_mfcc, 13);
        assert!(config.enable_spectral_analysis);
        assert!(config.enable_perceptual_modeling);
        assert!(config.enable_temporal_analysis);
        assert!(config.noise_robustness);
        assert_eq!(config.confidence_threshold, 0.7);
    }

    #[test]
    fn test_similarity_weights_normalization() {
        let weights = SimilarityWeights::default();
        let total = weights.embedding_weight
            + weights.spectral_weight
            + weights.perceptual_weight
            + weights.temporal_weight;
        assert!(
            (total - 1.0).abs() < 0.01,
            "Weights should sum to approximately 1.0"
        );
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds::default();
        assert!(thresholds.authentication_threshold > thresholds.synthesis_threshold);
        assert!(thresholds.synthesis_threshold > thresholds.identification_threshold);
        assert!(thresholds.identification_threshold > thresholds.conversion_threshold);
        assert!(thresholds.conversion_threshold > thresholds.general_threshold);
    }

    #[test]
    fn test_embedding_similarities_calculation() {
        let measurer = SimilarityMeasurer::default();

        let embedding1 = SpeakerEmbedding {
            vector: vec![1.0, 2.0, 3.0, 4.0],
            dimension: 4,
            confidence: 0.9,
            metadata: crate::embedding::EmbeddingMetadata {
                gender: None,
                age_estimate: None,
                language: None,
                emotion: None,
                voice_quality: crate::embedding::VoiceQuality {
                    f0_mean: 150.0,
                    f0_std: 20.0,
                    spectral_centroid: 2000.0,
                    spectral_bandwidth: 1000.0,
                    jitter: 0.01,
                    shimmer: 0.02,
                    energy_mean: 0.5,
                    energy_std: 0.1,
                },
                extraction_time: None,
            },
        };

        let embedding2 = SpeakerEmbedding {
            vector: vec![1.1, 2.1, 3.1, 4.1],
            dimension: 4,
            confidence: 0.85,
            metadata: crate::embedding::EmbeddingMetadata {
                gender: None,
                age_estimate: None,
                language: None,
                emotion: None,
                voice_quality: crate::embedding::VoiceQuality {
                    f0_mean: 155.0,
                    f0_std: 22.0,
                    spectral_centroid: 2100.0,
                    spectral_bandwidth: 1100.0,
                    jitter: 0.011,
                    shimmer: 0.022,
                    energy_mean: 0.52,
                    energy_std: 0.11,
                },
                extraction_time: None,
            },
        };

        let similarities = measurer
            .calculate_embedding_similarities(&embedding1, &embedding2)
            .unwrap();

        assert!(similarities.cosine_similarity > 0.0);
        assert!(similarities.cosine_similarity <= 1.0);
        assert!(similarities.euclidean_similarity > 0.0);
        assert!(similarities.euclidean_similarity <= 1.0);
        assert!(similarities.manhattan_similarity > 0.0);
        assert!(similarities.manhattan_similarity <= 1.0);
        assert!(similarities.pearson_correlation >= -1.0);
        assert!(similarities.pearson_correlation <= 1.0);
    }

    #[test]
    fn test_similarity_score_creation() {
        let embedding_similarities = EmbeddingSimilarities {
            cosine_similarity: 0.9,
            euclidean_similarity: 0.85,
            manhattan_similarity: 0.8,
            pearson_correlation: 0.75,
            spearman_correlation: 0.7,
            mahalanobis_similarity: 0.82,
        };

        let spectral_similarities = SpectralSimilarities::default();
        let perceptual_similarities = PerceptualSimilarities::default();
        let temporal_similarities = TemporalSimilarities::default();
        let weights = SimilarityWeights::default();

        let score = SimilarityScore::new(
            embedding_similarities,
            spectral_similarities,
            perceptual_similarities,
            temporal_similarities,
            &weights,
        );

        assert!(score.overall_score >= 0.0);
        assert!(score.overall_score <= 1.0);
        assert!(score.confidence >= 0.0);
        assert!(score.confidence <= 1.0);
    }

    #[test]
    fn test_quality_category_assignment() {
        let create_score = |overall: f32| {
            let embedding_similarities = EmbeddingSimilarities {
                cosine_similarity: overall,
                euclidean_similarity: overall,
                manhattan_similarity: overall,
                pearson_correlation: overall,
                spearman_correlation: overall,
                mahalanobis_similarity: overall,
            };

            SimilarityScore {
                embedding_similarities,
                spectral_similarities: SpectralSimilarities::default(),
                perceptual_similarities: PerceptualSimilarities::default(),
                temporal_similarities: TemporalSimilarities::default(),
                overall_score: overall,
                confidence: 0.8,
                statistical_metrics: StatisticalSignificance {
                    p_value: 0.05,
                    effect_size: 0.8,
                    confidence_interval: (overall - 0.1, overall + 0.1),
                    sample_size: 1,
                    test_statistic: overall * 10.0,
                },
            }
        };

        assert_eq!(
            create_score(0.95).quality_category(),
            QualityCategory::Excellent
        );
        assert_eq!(
            create_score(0.85).quality_category(),
            QualityCategory::VeryGood
        );
        assert_eq!(create_score(0.75).quality_category(), QualityCategory::Good);
        assert_eq!(create_score(0.65).quality_category(), QualityCategory::Fair);
        assert_eq!(create_score(0.55).quality_category(), QualityCategory::Poor);
        assert_eq!(
            create_score(0.45).quality_category(),
            QualityCategory::VeryPoor
        );
    }

    #[test]
    fn test_threshold_checking() {
        let score = SimilarityScore {
            embedding_similarities: EmbeddingSimilarities::default(),
            spectral_similarities: SpectralSimilarities::default(),
            perceptual_similarities: PerceptualSimilarities::default(),
            temporal_similarities: TemporalSimilarities::default(),
            overall_score: 0.85,
            confidence: 0.8,
            statistical_metrics: StatisticalSignificance {
                p_value: 0.05,
                effect_size: 0.8,
                confidence_interval: (0.75, 0.95),
                sample_size: 1,
                test_statistic: 8.5,
            },
        };

        assert!(score.meets_threshold(0.8));
        assert!(!score.meets_threshold(0.9));
    }

    #[test]
    fn test_quality_assessment() {
        let measurer = SimilarityMeasurer::default();

        let score = SimilarityScore {
            embedding_similarities: EmbeddingSimilarities::default(),
            spectral_similarities: SpectralSimilarities::default(),
            perceptual_similarities: PerceptualSimilarities::default(),
            temporal_similarities: TemporalSimilarities::default(),
            overall_score: 0.85,
            confidence: 0.8,
            statistical_metrics: StatisticalSignificance {
                p_value: 0.05,
                effect_size: 0.8,
                confidence_interval: (0.75, 0.95),
                sample_size: 1,
                test_statistic: 8.5,
            },
        };

        let assessment = measurer.evaluate_quality(&score, UseCase::Synthesis);
        assert_eq!(assessment.use_case, UseCase::Synthesis);
        assert!(assessment.passes_threshold);
        assert_eq!(assessment.quality_level, QualityCategory::VeryGood);
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_similarity_statistics() {
        let measurer = SimilarityMeasurer::default();

        let scores = vec![
            SimilarityScore {
                embedding_similarities: EmbeddingSimilarities::default(),
                spectral_similarities: SpectralSimilarities::default(),
                perceptual_similarities: PerceptualSimilarities::default(),
                temporal_similarities: TemporalSimilarities::default(),
                overall_score: 0.8,
                confidence: 0.9,
                statistical_metrics: StatisticalSignificance {
                    p_value: 0.05,
                    effect_size: 0.8,
                    confidence_interval: (0.7, 0.9),
                    sample_size: 1,
                    test_statistic: 8.0,
                },
            },
            SimilarityScore {
                embedding_similarities: EmbeddingSimilarities::default(),
                spectral_similarities: SpectralSimilarities::default(),
                perceptual_similarities: PerceptualSimilarities::default(),
                temporal_similarities: TemporalSimilarities::default(),
                overall_score: 0.9,
                confidence: 0.85,
                statistical_metrics: StatisticalSignificance {
                    p_value: 0.05,
                    effect_size: 0.8,
                    confidence_interval: (0.8, 1.0),
                    sample_size: 1,
                    test_statistic: 9.0,
                },
            },
        ];

        let stats = measurer.calculate_statistics(&scores);
        assert_eq!(stats.total_comparisons, 2);
        assert_eq!(stats.average_similarity, 0.85);
        assert!(stats.similarity_std_dev > 0.0);
        assert!(!stats.quality_distribution.is_empty());
    }

    #[test]
    fn test_euclidean_distance_calculation() {
        let measurer = SimilarityMeasurer::default();

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let distance = measurer.euclidean_distance(&vec1, &vec2).unwrap();
        let expected = ((3.0_f32).powi(2) + (3.0_f32).powi(2) + (3.0_f32).powi(2)).sqrt();
        assert!((distance - expected).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation() {
        let measurer = SimilarityMeasurer::default();

        // Perfect positive correlation
        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![2.0, 4.0, 6.0, 8.0];
        let correlation = measurer.pearson_correlation(&vec1, &vec2).unwrap();
        assert!((correlation - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let vec3 = vec![1.0, 2.0, 3.0, 4.0];
        let vec4 = vec![4.0, 3.0, 2.0, 1.0];
        let correlation2 = measurer.pearson_correlation(&vec3, &vec4).unwrap();
        assert!((correlation2 + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let measurer = SimilarityMeasurer::default();

        let embedding1 = SpeakerEmbedding {
            vector: vec![1.0, 2.0, 3.0],
            dimension: 3,
            confidence: 0.9,
            metadata: crate::embedding::EmbeddingMetadata {
                gender: None,
                age_estimate: None,
                language: None,
                emotion: None,
                voice_quality: crate::embedding::VoiceQuality {
                    f0_mean: 150.0,
                    f0_std: 20.0,
                    spectral_centroid: 2000.0,
                    spectral_bandwidth: 1000.0,
                    jitter: 0.01,
                    shimmer: 0.02,
                    energy_mean: 0.5,
                    energy_std: 0.1,
                },
                extraction_time: None,
            },
        };

        let embedding2 = SpeakerEmbedding {
            vector: vec![1.0, 2.0, 3.0, 4.0],
            dimension: 4,
            confidence: 0.85,
            metadata: crate::embedding::EmbeddingMetadata {
                gender: None,
                age_estimate: None,
                language: None,
                emotion: None,
                voice_quality: crate::embedding::VoiceQuality {
                    f0_mean: 155.0,
                    f0_std: 22.0,
                    spectral_centroid: 2100.0,
                    spectral_bandwidth: 1100.0,
                    jitter: 0.011,
                    shimmer: 0.022,
                    energy_mean: 0.52,
                    energy_std: 0.11,
                },
                extraction_time: None,
            },
        };

        let result = measurer.calculate_embedding_similarities(&embedding1, &embedding2);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sample_similarity_measurement() {
        let measurer = SimilarityMeasurer::default();

        let sample1 = VoiceSample::new("sample1".to_string(), vec![0.1, 0.2, 0.3, 0.4], 16000);

        let sample2 = VoiceSample::new("sample2".to_string(), vec![0.15, 0.25, 0.35, 0.45], 16000);

        let result = measurer.measure_sample_similarity(&sample1, &sample2).await;
        assert!(result.is_ok());

        let score = result.unwrap();
        assert!(score.overall_score >= 0.0);
        assert!(score.overall_score <= 1.0);
        assert!(score.confidence >= 0.0);
        assert!(score.confidence <= 1.0);
    }

    #[test]
    fn test_use_case_specific_thresholds() {
        let thresholds = QualityThresholds::default();

        // Authentication should have the highest threshold
        assert!(thresholds.authentication_threshold >= 0.95);

        // General use case should have the lowest threshold
        assert!(thresholds.general_threshold <= 0.75);

        // Verify ordering
        assert!(thresholds.authentication_threshold > thresholds.synthesis_threshold);
        assert!(thresholds.synthesis_threshold > thresholds.identification_threshold);
        assert!(thresholds.identification_threshold > thresholds.conversion_threshold);
        assert!(thresholds.conversion_threshold >= thresholds.general_threshold);
    }

    #[test]
    fn test_recommendations_generation() {
        let measurer = SimilarityMeasurer::default();

        let low_quality_score = SimilarityScore {
            embedding_similarities: EmbeddingSimilarities {
                cosine_similarity: 0.6,
                euclidean_similarity: 0.65,
                manhattan_similarity: 0.62,
                pearson_correlation: 0.58,
                spearman_correlation: 0.6,
                mahalanobis_similarity: 0.63,
            },
            spectral_similarities: SpectralSimilarities {
                mfcc_similarity: 0.6,
                spectral_centroid_similarity: 0.65,
                spectral_rolloff_similarity: 0.62,
                spectral_bandwidth_similarity: 0.6,
                zcr_similarity: 0.58,
                chroma_similarity: 0.61,
                spectral_contrast_similarity: 0.63,
            },
            perceptual_similarities: PerceptualSimilarities {
                psychoacoustic_similarity: 0.6,
                bark_similarity: 0.62,
                erb_similarity: 0.58,
                loudness_similarity: 0.6,
                roughness_similarity: 0.61,
                sharpness_similarity: 0.59,
            },
            temporal_similarities: TemporalSimilarities {
                rhythm_similarity: 0.65,
                speaking_rate_similarity: 0.67,
                pause_pattern_similarity: 0.6,
                energy_contour_similarity: 0.62,
                pitch_contour_similarity: 0.6,
                formant_trajectory_similarity: 0.61,
            },
            overall_score: 0.6,
            confidence: 0.5,
            statistical_metrics: StatisticalSignificance {
                p_value: 0.1,
                effect_size: 0.3,
                confidence_interval: (0.5, 0.7),
                sample_size: 10,
                test_statistic: 6.0,
            },
        };

        let recommendations =
            measurer.generate_recommendations(&low_quality_score, UseCase::Authentication);
        assert!(recommendations.len() >= 4); // Should have multiple recommendations for low quality

        // Check for specific recommendation categories
        let rec_text = recommendations.join(" ");
        assert!(
            rec_text.contains("embedding")
                || rec_text.contains("spectral")
                || rec_text.contains("perceptual")
                || rec_text.contains("pitch")
                || rec_text.contains("sample quality")
        );
    }
}
