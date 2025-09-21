//! Core domain adaptation types and statistics
//!
//! This module contains the core types for domain adaptation including
//! domain shift detection, adaptation results, and comprehensive statistics.

use crate::{DatasetSample, DatasetStatistics, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::{AgeDistribution, DomainConfig, GenderDistribution, TextComplexity};

/// Domain shift information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainShift {
    /// Shift magnitude
    pub magnitude: f32,
    /// Shift direction (feature space)
    pub direction: Vec<f32>,
    /// Affected features
    pub affected_features: Vec<String>,
    /// Confidence in detection
    pub confidence: f32,
    /// Detection timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Domain adaptation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Adaptation success
    pub success: bool,
    /// Performance improvement
    pub improvement: f32,
    /// Adapted samples
    pub adapted_samples: Vec<DatasetSample>,
    /// Adaptation metadata
    pub metadata: HashMap<String, String>,
}

/// Domain adapter interface
#[async_trait::async_trait]
pub trait DomainAdapter: Send + Sync {
    /// Detect domain shift between source and target data
    async fn detect_domain_shift(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<DomainShift>;

    /// Adapt source domain data to target domain
    async fn adapt_domain(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<AdaptationResult>;

    /// Apply domain-specific preprocessing
    async fn apply_preprocessing(
        &self,
        samples: &[DatasetSample],
        domain_config: &DomainConfig,
    ) -> Result<Vec<DatasetSample>>;

    /// Mix data from multiple domains
    async fn mix_domains(
        &self,
        domain_samples: &[(DomainConfig, Vec<DatasetSample>)],
    ) -> Result<Vec<DatasetSample>>;

    /// Transfer learning from source to target domain
    async fn transfer_learning(
        &mut self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<()>;

    /// Get domain statistics
    async fn get_domain_statistics(&self, samples: &[DatasetSample]) -> Result<DomainStatistics>;

    /// Validate domain compatibility
    async fn validate_compatibility(
        &self,
        source_domain: &DomainConfig,
        target_domain: &DomainConfig,
    ) -> Result<CompatibilityReport>;
}

/// Domain statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStatistics {
    /// Basic dataset statistics
    pub dataset_stats: DatasetStatistics,
    /// Audio characteristics statistics
    pub audio_stats: AudioStatistics,
    /// Text characteristics statistics
    pub text_stats: TextStatistics,
    /// Speaker characteristics statistics
    pub speaker_stats: SpeakerStatistics,
}

/// Audio statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStatistics {
    /// Sample rate distribution
    pub sample_rate_distribution: HashMap<u32, f32>,
    /// Channel distribution
    pub channel_distribution: HashMap<u32, f32>,
    /// Quality score distribution
    pub quality_distribution: HashMap<String, f32>,
    /// SNR distribution
    pub snr_distribution: (f32, f32, f32), // (min, max, avg)
    /// Dynamic range distribution
    pub dynamic_range_distribution: (f32, f32, f32),
}

/// Text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Language distribution
    pub language_distribution: HashMap<LanguageCode, f32>,
    /// Text length distribution
    pub length_distribution: (usize, usize, f32), // (min, max, avg)
    /// Vocabulary statistics
    pub vocabulary_stats: VocabularyStatistics,
    /// Complexity statistics
    pub complexity_stats: TextComplexity,
}

/// Vocabulary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyStatistics {
    /// Unique words count
    pub unique_words: usize,
    /// Most frequent words
    pub frequent_words: Vec<(String, usize)>,
    /// Word frequency distribution
    pub word_frequency_distribution: HashMap<String, f32>,
    /// OOV rate vs reference vocabulary
    pub oov_rate: f32,
}

/// Speaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerStatistics {
    /// Number of unique speakers
    pub unique_speakers: usize,
    /// Speaker distribution
    pub speaker_distribution: HashMap<String, f32>,
    /// Gender distribution
    pub gender_distribution: GenderDistribution,
    /// Age distribution (if available)
    pub age_distribution: AgeDistribution,
}

/// Domain compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    /// Overall compatibility score
    pub compatibility_score: f32,
    /// Audio compatibility
    pub audio_compatibility: f32,
    /// Text compatibility
    pub text_compatibility: f32,
    /// Speaker compatibility
    pub speaker_compatibility: f32,
    /// Recommended adaptations
    pub recommendations: Vec<AdaptationRecommendation>,
    /// Potential issues
    pub potential_issues: Vec<String>,
}

/// Adaptation recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: Priority,
    /// Description
    pub description: String,
    /// Estimated improvement
    pub estimated_improvement: f32,
}

/// Types of recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    AudioPreprocessing,
    TextNormalization,
    SpeakerAdaptation,
    FeatureAlignment,
    DataAugmentation,
    TransferLearning,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

impl DomainShift {
    /// Create a new domain shift instance
    pub fn new(
        magnitude: f32,
        direction: Vec<f32>,
        affected_features: Vec<String>,
        confidence: f32,
    ) -> Self {
        Self {
            magnitude,
            direction,
            affected_features,
            confidence,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if the shift is significant
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.magnitude > threshold && self.confidence > 0.8
    }

    /// Get the most affected feature
    pub fn most_affected_feature(&self) -> Option<&String> {
        self.affected_features.first()
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl AdaptationResult {
    /// Create a successful adaptation result
    pub fn success(improvement: f32, adapted_samples: Vec<DatasetSample>) -> Self {
        Self {
            success: true,
            improvement,
            adapted_samples,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed adaptation result
    pub fn failure(error_message: String) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("error".to_string(), error_message);
        Self {
            success: false,
            improvement: 0.0,
            adapted_samples: vec![],
            metadata,
        }
    }

    /// Check if adaptation was successful
    pub fn is_successful(&self) -> bool {
        self.success && self.improvement > 0.0
    }

    /// Get error message if adaptation failed
    pub fn error_message(&self) -> Option<&String> {
        self.metadata.get("error")
    }
}

impl DomainStatistics {
    /// Create new domain statistics
    pub fn new(
        dataset_stats: DatasetStatistics,
        audio_stats: AudioStatistics,
        text_stats: TextStatistics,
        speaker_stats: SpeakerStatistics,
    ) -> Self {
        Self {
            dataset_stats,
            audio_stats,
            text_stats,
            speaker_stats,
        }
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.dataset_stats.total_items
    }

    /// Get primary language
    pub fn primary_language(&self) -> Option<LanguageCode> {
        self.text_stats
            .language_distribution
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(lang, _)| *lang)
    }

    /// Get average audio quality
    pub fn average_audio_quality(&self) -> f32 {
        self.audio_stats.quality_distribution.values().sum::<f32>()
            / self.audio_stats.quality_distribution.len().max(1) as f32
    }
}

impl AudioStatistics {
    /// Create new audio statistics
    pub fn new() -> Self {
        Self {
            sample_rate_distribution: HashMap::new(),
            channel_distribution: HashMap::new(),
            quality_distribution: HashMap::new(),
            snr_distribution: (0.0, 0.0, 0.0),
            dynamic_range_distribution: (0.0, 0.0, 0.0),
        }
    }

    /// Get most common sample rate
    pub fn most_common_sample_rate(&self) -> Option<u32> {
        self.sample_rate_distribution
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(rate, _)| *rate)
    }

    /// Get most common channel count
    pub fn most_common_channels(&self) -> Option<u32> {
        self.channel_distribution
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(channels, _)| *channels)
    }

    /// Check if predominantly mono
    pub fn is_predominantly_mono(&self) -> bool {
        self.channel_distribution.get(&1).unwrap_or(&0.0) > &0.5
    }
}

impl TextStatistics {
    /// Create new text statistics
    pub fn new() -> Self {
        Self {
            language_distribution: HashMap::new(),
            length_distribution: (0, 0, 0.0),
            vocabulary_stats: VocabularyStatistics::new(),
            complexity_stats: TextComplexity {
                avg_word_length: 0.0,
                lexical_diversity: 0.0,
                syntactic_complexity: 0.0,
                reading_level: 0.0,
            },
        }
    }

    /// Check if multilingual
    pub fn is_multilingual(&self) -> bool {
        self.language_distribution.len() > 1
    }

    /// Get language diversity score
    pub fn language_diversity(&self) -> f32 {
        if self.language_distribution.is_empty() {
            return 0.0;
        }

        let entropy = self
            .language_distribution
            .values()
            .map(|p| if *p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum::<f32>();

        entropy / (self.language_distribution.len() as f32).log2()
    }

    /// Get average text length
    pub fn average_length(&self) -> f32 {
        self.length_distribution.2
    }
}

impl VocabularyStatistics {
    /// Create new vocabulary statistics
    pub fn new() -> Self {
        Self {
            unique_words: 0,
            frequent_words: vec![],
            word_frequency_distribution: HashMap::new(),
            oov_rate: 0.0,
        }
    }

    /// Get vocabulary richness (type-token ratio)
    pub fn vocabulary_richness(&self) -> f32 {
        if self.word_frequency_distribution.is_empty() {
            return 0.0;
        }

        let total_tokens: f32 = self.word_frequency_distribution.values().sum();
        if total_tokens > 0.0 {
            self.unique_words as f32 / total_tokens
        } else {
            0.0
        }
    }

    /// Get most frequent word
    pub fn most_frequent_word(&self) -> Option<&(String, usize)> {
        self.frequent_words.first()
    }

    /// Check if vocabulary is large
    pub fn is_large_vocabulary(&self, threshold: usize) -> bool {
        self.unique_words > threshold
    }
}

impl SpeakerStatistics {
    /// Create new speaker statistics
    pub fn new() -> Self {
        Self {
            unique_speakers: 0,
            speaker_distribution: HashMap::new(),
            gender_distribution: GenderDistribution {
                male: 0.0,
                female: 0.0,
                other: 0.0,
            },
            age_distribution: AgeDistribution {
                children: 0.0,
                teenagers: 0.0,
                young_adults: 0.0,
                middle_aged: 0.0,
                older_adults: 0.0,
            },
        }
    }

    /// Check if gender balanced
    pub fn is_gender_balanced(&self, threshold: f32) -> bool {
        let diff = (self.gender_distribution.male - self.gender_distribution.female).abs();
        diff < threshold
    }

    /// Get dominant gender
    pub fn dominant_gender(&self) -> &str {
        if self.gender_distribution.male > self.gender_distribution.female {
            "male"
        } else if self.gender_distribution.female > self.gender_distribution.male {
            "female"
        } else {
            "balanced"
        }
    }

    /// Get age diversity score
    pub fn age_diversity(&self) -> f32 {
        let ages = [
            self.age_distribution.children,
            self.age_distribution.teenagers,
            self.age_distribution.young_adults,
            self.age_distribution.middle_aged,
            self.age_distribution.older_adults,
        ];

        let entropy = ages
            .iter()
            .map(|p| if *p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum::<f32>();

        entropy / 5.0_f32.log2()
    }
}

impl CompatibilityReport {
    /// Create a new compatibility report
    pub fn new(
        compatibility_score: f32,
        audio_compatibility: f32,
        text_compatibility: f32,
        speaker_compatibility: f32,
    ) -> Self {
        Self {
            compatibility_score,
            audio_compatibility,
            text_compatibility,
            speaker_compatibility,
            recommendations: vec![],
            potential_issues: vec![],
        }
    }

    /// Check if domains are highly compatible
    pub fn is_highly_compatible(&self) -> bool {
        self.compatibility_score > 0.8
    }

    /// Check if adaptation is recommended
    pub fn adaptation_recommended(&self) -> bool {
        self.compatibility_score < 0.6 || !self.recommendations.is_empty()
    }

    /// Get high priority recommendations
    pub fn high_priority_recommendations(&self) -> Vec<&AdaptationRecommendation> {
        self.recommendations
            .iter()
            .filter(|rec| matches!(rec.priority, Priority::High | Priority::Critical))
            .collect()
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: AdaptationRecommendation) {
        self.recommendations.push(recommendation);
    }

    /// Add potential issue
    pub fn add_issue(&mut self, issue: String) {
        self.potential_issues.push(issue);
    }
}

impl AdaptationRecommendation {
    /// Create a new recommendation
    pub fn new(
        recommendation_type: RecommendationType,
        priority: Priority,
        description: String,
        estimated_improvement: f32,
    ) -> Self {
        Self {
            recommendation_type,
            priority,
            description,
            estimated_improvement,
        }
    }

    /// Check if this is a critical recommendation
    pub fn is_critical(&self) -> bool {
        matches!(self.priority, Priority::Critical)
    }

    /// Check if this is a high-impact recommendation
    pub fn is_high_impact(&self) -> bool {
        self.estimated_improvement > 0.2
    }
}

impl Default for AudioStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TextStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VocabularyStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SpeakerStatistics {
    fn default() -> Self {
        Self::new()
    }
}
