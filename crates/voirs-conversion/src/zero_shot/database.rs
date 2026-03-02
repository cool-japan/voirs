//! Reference voice database for zero-shot learning

use crate::types::VoiceCharacteristics;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Speaker embedding for voice identification and conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Embedding vector data
    pub data: Vec<f32>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Reference voice database for zero-shot learning
pub struct ReferenceVoiceDatabase {
    /// Voice entries indexed by speaker ID
    voices: HashMap<String, ReferenceVoice>,

    /// Voice embeddings for fast similarity search
    embeddings: HashMap<String, SpeakerEmbedding>,

    /// Voice characteristics
    characteristics: HashMap<String, VoiceCharacteristics>,

    /// Usage statistics
    usage_stats: HashMap<String, UsageStatistics>,

    /// Database metadata
    metadata: DatabaseMetadata,
}

/// Reference voice entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceVoice {
    /// Speaker identifier
    pub speaker_id: String,

    /// Voice name/description
    pub name: String,

    /// Audio samples
    pub audio_samples: Vec<AudioSample>,

    /// Speaker embedding
    pub embedding: SpeakerEmbedding,

    /// Voice characteristics
    pub characteristics: VoiceCharacteristics,

    /// Quality scores
    pub quality_scores: QualityScores,

    /// Metadata
    pub metadata: VoiceMetadata,

    /// Last used timestamp
    #[serde(skip)]
    pub last_used: Option<Instant>,
}

/// Audio sample for reference voice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSample {
    /// Sample identifier
    pub id: String,

    /// Audio data (placeholder - would contain actual audio)
    #[serde(skip)]
    pub audio_data: Vec<f32>,

    /// Sample rate
    pub sample_rate: u32,

    /// Duration in seconds
    pub duration: f32,

    /// Transcription
    pub transcription: Option<String>,

    /// Quality score
    pub quality_score: f32,

    /// Phonetic content analysis
    pub phonetic_content: PhoneticAnalysis,
}

/// Quality scores for reference voice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    /// Overall quality score (0.0 to 1.0)
    pub overall: f32,

    /// Clarity score
    pub clarity: f32,

    /// Naturalness score
    pub naturalness: f32,

    /// Consistency score
    pub consistency: f32,

    /// Recording quality score
    pub recording_quality: f32,

    /// Prosody quality score
    pub prosody_quality: f32,
}

/// Voice metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMetadata {
    /// Language
    pub language: String,

    /// Accent/dialect
    pub accent: Option<String>,

    /// Gender
    pub gender: Option<String>,

    /// Age group
    pub age_group: Option<String>,

    /// Recording environment
    pub recording_environment: Option<String>,

    /// Tags
    pub tags: Vec<String>,

    /// Creation timestamp
    #[serde(
        skip_serializing,
        skip_deserializing,
        default = "std::time::Instant::now"
    )]
    pub created: Instant,

    /// Last modified timestamp
    #[serde(skip_serializing, skip_deserializing, default)]
    pub modified: Option<Instant>,
}

/// Phonetic analysis of audio sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticAnalysis {
    /// Phoneme distribution
    pub phoneme_distribution: HashMap<String, f32>,

    /// Phonetic diversity score
    pub diversity_score: f32,

    /// Vowel-consonant ratio
    pub vowel_consonant_ratio: f32,

    /// Prosodic features
    pub prosodic_features: ProsodicFeatures,
}

/// Prosodic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicFeatures {
    /// Mean F0
    pub mean_f0: f32,

    /// F0 range
    pub f0_range: (f32, f32),

    /// Speaking rate (syllables per second)
    pub speaking_rate: f32,

    /// Pause patterns
    pub pause_patterns: Vec<f32>,

    /// Stress patterns
    pub stress_patterns: Vec<f32>,
}

/// Usage statistics for reference voices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    /// Number of times used
    pub usage_count: u64,

    /// Average similarity scores
    pub avg_similarity: f32,

    /// Success rate
    pub success_rate: f32,

    /// Last used timestamp
    #[serde(skip)]
    pub last_used: Option<Instant>,

    /// Preferred contexts
    pub preferred_contexts: Vec<String>,
}

/// Database metadata
#[derive(Debug, Clone)]
pub struct DatabaseMetadata {
    /// Total number of voices
    pub total_voices: usize,

    /// Total audio duration (seconds)
    pub total_duration: f32,

    /// Languages represented
    pub languages: Vec<String>,

    /// Last updated timestamp
    pub last_updated: Instant,

    /// Database version
    pub version: String,

    /// Index statistics
    pub index_stats: IndexStatistics,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Embedding index size
    pub embedding_index_size: usize,

    /// Characteristic index size
    pub characteristic_index_size: usize,

    /// Search performance metrics
    pub search_performance: SearchPerformanceMetrics,
}

/// Search performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceMetrics {
    /// Average search time (ms)
    pub avg_search_time: f32,

    /// Cache hit rate
    pub cache_hit_rate: f32,

    /// Index efficiency
    pub index_efficiency: f32,
}

impl Default for ReferenceVoiceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceVoiceDatabase {
    /// Creates a new empty reference voice database.
    ///
    /// # Returns
    ///
    /// A new `ReferenceVoiceDatabase` instance with empty collections and initialized metadata.
    pub fn new() -> Self {
        Self {
            voices: HashMap::new(),
            embeddings: HashMap::new(),
            characteristics: HashMap::new(),
            usage_stats: HashMap::new(),
            metadata: DatabaseMetadata {
                total_voices: 0,
                total_duration: 0.0,
                languages: Vec::new(),
                last_updated: Instant::now(),
                version: "1.0.0".to_string(),
                index_stats: IndexStatistics {
                    embedding_index_size: 0,
                    characteristic_index_size: 0,
                    search_performance: SearchPerformanceMetrics {
                        avg_search_time: 0.0,
                        cache_hit_rate: 0.0,
                        index_efficiency: 1.0,
                    },
                },
            },
        }
    }

    /// Adds a reference voice to the database.
    ///
    /// # Arguments
    ///
    /// * `voice` - The reference voice to add with speaker ID, embedding, and characteristics
    ///
    /// # Returns
    ///
    /// `Ok(())` if the voice was successfully added.
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but returns `Result` for future error handling.
    pub fn add_voice(&mut self, voice: ReferenceVoice) -> Result<()> {
        let speaker_id = voice.speaker_id.clone();
        self.embeddings
            .insert(speaker_id.clone(), voice.embedding.clone());
        self.characteristics
            .insert(speaker_id.clone(), voice.characteristics.clone());
        self.usage_stats.insert(
            speaker_id.clone(),
            UsageStatistics {
                usage_count: 0,
                avg_similarity: 0.0,
                success_rate: 0.0,
                last_used: None,
                preferred_contexts: Vec::new(),
            },
        );
        self.voices.insert(speaker_id, voice);
        self.metadata.total_voices += 1;
        Ok(())
    }

    /// Removes a reference voice from the database by speaker ID.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - The unique identifier of the speaker to remove
    ///
    /// # Returns
    ///
    /// `Ok(())` if the voice was successfully removed or if it didn't exist.
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but returns `Result` for future error handling.
    pub fn remove_voice(&mut self, speaker_id: &str) -> Result<()> {
        self.voices.remove(speaker_id);
        self.embeddings.remove(speaker_id);
        self.characteristics.remove(speaker_id);
        self.usage_stats.remove(speaker_id);
        if self.metadata.total_voices > 0 {
            self.metadata.total_voices -= 1;
        }
        Ok(())
    }

    /// Finds the most similar reference voices based on target characteristics.
    ///
    /// # Arguments
    ///
    /// * `target_characteristics` - The target voice characteristics to match against
    /// * `max_voices` - Maximum number of similar voices to return
    ///
    /// # Returns
    ///
    /// A vector of reference voices sorted by similarity (most similar first), limited to `max_voices`.
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but returns `Result` for future error handling.
    pub fn find_similar_voices(
        &self,
        target_characteristics: &VoiceCharacteristics,
        max_voices: usize,
    ) -> Result<Vec<ReferenceVoice>> {
        let mut similarities = Vec::new();

        for voice in self.voices.values() {
            let similarity =
                self.calculate_similarity(&voice.characteristics, target_characteristics);
            similarities.push((similarity, voice.clone()));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top matches
        Ok(similarities
            .into_iter()
            .take(max_voices)
            .map(|(_, voice)| voice)
            .collect())
    }

    /// Returns a reference to the database metadata.
    ///
    /// Provides access to database statistics, version information, and index performance metrics.
    ///
    /// # Returns
    ///
    /// A reference to the [`DatabaseMetadata`] containing information about the database state,
    /// including total voices, duration, supported languages, and search performance metrics.
    pub fn metadata(&self) -> &DatabaseMetadata {
        &self.metadata
    }

    fn calculate_similarity(
        &self,
        voice1: &VoiceCharacteristics,
        voice2: &VoiceCharacteristics,
    ) -> f32 {
        // Enhanced multi-dimensional similarity calculation
        let mut similarity = 0.0;
        let mut total_weight: f32 = 0.0;

        // Gender similarity (weight: 0.25)
        if voice1.gender == voice2.gender {
            similarity += 0.25;
        } else if voice1.gender.is_some() && voice2.gender.is_some() {
            similarity += 0.1; // Partial credit for different but specified genders
        }
        total_weight += 0.25;

        // Age group similarity (weight: 0.15)
        if voice1.age_group == voice2.age_group {
            similarity += 0.15;
        } else if voice1.age_group.is_some() && voice2.age_group.is_some() {
            // Age groups have some similarity (adult vs young_adult vs senior)
            let age_similarity = match (voice1.age_group, voice2.age_group) {
                (Some(a1), Some(a2)) => {
                    use crate::types::AgeGroup;
                    match (a1, a2) {
                        (AgeGroup::YoungAdult, AgeGroup::MiddleAged)
                        | (AgeGroup::MiddleAged, AgeGroup::YoungAdult) => 0.8,
                        (AgeGroup::MiddleAged, AgeGroup::Senior)
                        | (AgeGroup::Senior, AgeGroup::MiddleAged) => 0.6,
                        (AgeGroup::Child, AgeGroup::YoungAdult)
                        | (AgeGroup::YoungAdult, AgeGroup::Child) => 0.4,
                        _ => 0.2,
                    }
                }
                _ => 0.05,
            };
            similarity += 0.15 * age_similarity;
        }
        total_weight += 0.15;

        // Accent similarity (weight: 0.2)
        if voice1.accent == voice2.accent {
            similarity += 0.2;
        } else if voice1.accent.is_some() && voice2.accent.is_some() {
            // Some accents are more similar than others
            let accent_similarity =
                if let (Some(ref a1), Some(ref a2)) = (&voice1.accent, &voice2.accent) {
                    if (a1.contains("american") && a2.contains("canadian"))
                        || (a2.contains("american") && a1.contains("canadian"))
                    {
                        0.8
                    } else if (a1.contains("british") && a2.contains("australian"))
                        || (a2.contains("british") && a1.contains("australian"))
                    {
                        0.7
                    } else {
                        0.3
                    }
                } else {
                    0.1
                };
            similarity += 0.2 * accent_similarity;
        }
        total_weight += 0.2;

        // Pitch similarity (weight: 0.2)
        let pitch_diff = (voice1.pitch.mean_f0 - voice2.pitch.mean_f0).abs();
        let pitch_similarity = if pitch_diff < 10.0 {
            1.0 // Very similar pitch
        } else if pitch_diff < 50.0 {
            1.0 - (pitch_diff - 10.0) / 40.0 // Linear decay from 10-50 Hz
        } else {
            (1.0 - (pitch_diff / 200.0)).max(0.0) // Slower decay above 50 Hz
        };
        similarity += pitch_similarity * 0.2;
        total_weight += 0.2;

        // Spectral similarity (weight: 0.1)
        let formant_diff = (voice1.spectral.formant_shift - voice2.spectral.formant_shift).abs();
        let spectral_similarity = 1.0 - formant_diff.min(1.0);
        similarity += spectral_similarity * 0.1;
        total_weight += 0.1;

        // Quality similarity (weight: 0.1)
        let breathiness_diff = (voice1.quality.breathiness - voice2.quality.breathiness).abs();
        let roughness_diff = (voice1.quality.roughness - voice2.quality.roughness).abs();
        let quality_similarity = 1.0 - ((breathiness_diff + roughness_diff) / 2.0).min(1.0);
        similarity += quality_similarity * 0.1;
        total_weight += 0.1;

        // Normalize by total weight
        similarity / total_weight.max(1e-10)
    }
}

impl Default for VoiceMetadata {
    fn default() -> Self {
        Self {
            language: String::new(),
            accent: None,
            gender: None,
            age_group: None,
            recording_environment: None,
            tags: Vec::new(),
            created: Instant::now(),
            modified: None,
        }
    }
}

impl Default for DatabaseMetadata {
    fn default() -> Self {
        Self {
            total_voices: 0,
            total_duration: 0.0,
            languages: Vec::new(),
            last_updated: Instant::now(),
            version: "1.0.0".to_string(),
            index_stats: IndexStatistics::default(),
        }
    }
}

impl Default for IndexStatistics {
    fn default() -> Self {
        Self {
            embedding_index_size: 0,
            characteristic_index_size: 0,
            search_performance: SearchPerformanceMetrics::default(),
        }
    }
}

impl Default for SearchPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_search_time: 0.0,
            cache_hit_rate: 0.0,
            index_efficiency: 1.0,
        }
    }
}
