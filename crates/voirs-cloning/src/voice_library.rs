//! Voice Library Management for VoiRS Voice Cloning
//!
//! This module provides comprehensive voice library management capabilities,
//! including organization, cataloging, search, version control, and batch
//! operations for managing large collections of cloned voices.

use crate::consent::{ConsentManager, ConsentRecord};
use crate::quality::{CloningQualityAssessor, QualityMetrics};
use crate::similarity::SimilarityMeasurer;
use crate::types::{CloningMethod, SpeakerProfile, VoiceCloneResult, VoiceSample};
use crate::usage_tracking::SimilarityMetrics;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::fs;

/// Voice library entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceLibraryEntry {
    /// Unique voice identifier
    pub id: String,
    /// Voice display name
    pub name: String,
    /// Voice description
    pub description: Option<String>,
    /// Speaker profile
    pub speaker_profile: SpeakerProfile,
    /// Voice category/genre
    pub category: VoiceCategory,
    /// Language(s) supported
    pub languages: Vec<String>,
    /// Voice characteristics tags
    pub tags: HashSet<String>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub modified_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// File size in bytes
    pub file_size: u64,
    /// Storage path
    pub storage_path: PathBuf,
    /// Voice rating (1-5 stars)
    pub rating: Option<u8>,
    /// User notes
    pub notes: Option<String>,
    /// Version information
    pub version: VoiceVersion,
    /// Consent status
    pub consent_status: ConsentStatus,
    /// Usage statistics
    pub usage_stats: VoiceUsageStats,
}

/// Voice category for organization
#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum VoiceCategory {
    /// Personal voice
    Personal,
    /// Celebrity voice
    Celebrity,
    /// Character voice
    Character,
    /// Professional narration
    Narration,
    /// Singing voice
    Singing,
    /// Language learning
    LanguageLearning,
    /// Accessibility
    Accessibility,
    /// Entertainment
    Entertainment,
    /// Educational
    Educational,
    /// Custom category
    Custom(String),
}

/// Voice version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVersion {
    /// Version number (semantic versioning)
    pub version: String,
    /// Version creation time
    pub created_at: SystemTime,
    /// Version description/changelog
    pub description: String,
    /// Previous version ID (for history)
    pub parent_version: Option<String>,
    /// Version tags/labels
    pub tags: Vec<String>,
}

/// Consent status for voice usage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsentStatus {
    /// Consent granted and valid
    Valid,
    /// Consent expired
    Expired,
    /// Consent revoked
    Revoked,
    /// Consent pending verification
    Pending,
    /// No consent required (public domain)
    NotRequired,
}

/// Voice usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceUsageStats {
    /// Total synthesis requests
    pub synthesis_count: u64,
    /// Total synthesis duration
    pub total_duration: Duration,
    /// Average quality score
    pub avg_quality: f32,
    /// Most used cloning method
    pub preferred_method: Option<CloningMethod>,
    /// Usage by application/context
    pub usage_by_context: HashMap<String, u64>,
    /// Performance metrics
    pub avg_processing_time: Duration,
}

/// Voice collection/playlist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCollection {
    /// Collection identifier
    pub id: String,
    /// Collection name
    pub name: String,
    /// Collection description
    pub description: Option<String>,
    /// Voice IDs in this collection
    pub voice_ids: Vec<String>,
    /// Collection tags
    pub tags: HashSet<String>,
    /// Collection creation time
    pub created_at: SystemTime,
    /// Collection modification time
    pub modified_at: SystemTime,
    /// Collection owner/creator
    pub owner: String,
    /// Whether collection is public
    pub is_public: bool,
    /// Collection icon/thumbnail path
    pub icon_path: Option<PathBuf>,
}

/// Search query for voice library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSearchQuery {
    /// Text search in name/description
    pub text: Option<String>,
    /// Category filter
    pub category: Option<VoiceCategory>,
    /// Language filter
    pub languages: Vec<String>,
    /// Tags filter (all must match)
    pub required_tags: Vec<String>,
    /// Tags filter (any can match)
    pub optional_tags: Vec<String>,
    /// Quality score range
    pub quality_range: Option<(f32, f32)>,
    /// Rating range
    pub rating_range: Option<(u8, u8)>,
    /// Date range
    pub date_range: Option<(SystemTime, SystemTime)>,
    /// Consent status filter
    pub consent_status: Option<ConsentStatus>,
    /// Minimum usage count
    pub min_usage_count: Option<u64>,
    /// Sort criteria
    pub sort_by: SortCriteria,
    /// Sort order
    pub sort_order: SortOrder,
    /// Result limit
    pub limit: Option<usize>,
    /// Result offset for pagination
    pub offset: Option<usize>,
}

/// Sort criteria for search results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortCriteria {
    /// Sort by name
    Name,
    /// Sort by creation date
    CreatedAt,
    /// Sort by modification date
    ModifiedAt,
    /// Sort by last access
    LastAccessed,
    /// Sort by access count
    AccessCount,
    /// Sort by quality score
    Quality,
    /// Sort by rating
    Rating,
    /// Sort by file size
    FileSize,
    /// Sort by relevance (for text search)
    Relevance,
}

/// Sort order
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortOrder {
    /// Ascending order
    Ascending,
    /// Descending order
    Descending,
}

/// Library statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryStatistics {
    /// Total number of voices
    pub total_voices: usize,
    /// Total storage size in bytes
    pub total_storage_size: u64,
    /// Voices by category
    pub voices_by_category: HashMap<VoiceCategory, usize>,
    /// Voices by language
    pub voices_by_language: HashMap<String, usize>,
    /// Average quality score
    pub average_quality: f32,
    /// Most popular voices (by access count)
    pub most_popular: Vec<String>,
    /// Recently added voices
    pub recently_added: Vec<String>,
    /// Quality distribution
    pub quality_distribution: BTreeMap<u8, usize>, // Quality ranges 0-9
    /// Consent status distribution
    pub consent_distribution: HashMap<ConsentStatus, usize>,
}

/// Batch operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchOperation {
    /// Delete multiple voices
    Delete { voice_ids: Vec<String> },
    /// Update tags for multiple voices
    UpdateTags {
        voice_ids: Vec<String>,
        add_tags: Vec<String>,
        remove_tags: Vec<String>,
    },
    /// Move voices to category
    UpdateCategory {
        voice_ids: Vec<String>,
        new_category: VoiceCategory,
    },
    /// Update quality metrics
    UpdateQuality { voice_ids: Vec<String> },
    /// Export voices to directory
    Export {
        voice_ids: Vec<String>,
        export_path: PathBuf,
        include_metadata: bool,
    },
    /// Import voices from directory
    Import {
        import_path: PathBuf,
        category: VoiceCategory,
        default_tags: Vec<String>,
    },
}

/// Batch operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperationResult {
    /// Operation identifier
    pub operation_id: String,
    /// Number of successfully processed items
    pub success_count: usize,
    /// Number of failed items
    pub failure_count: usize,
    /// Detailed results per item
    pub item_results: Vec<BatchItemResult>,
    /// Operation duration
    pub duration: Duration,
    /// Operation timestamp
    pub timestamp: SystemTime,
}

/// Individual item result in batch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemResult {
    /// Item identifier (voice ID)
    pub item_id: String,
    /// Whether operation succeeded for this item
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Voice Library Manager
pub struct VoiceLibraryManager {
    /// Library root directory
    library_path: PathBuf,
    /// Voice entries index
    voices: Arc<RwLock<HashMap<String, VoiceLibraryEntry>>>,
    /// Voice collections
    collections: Arc<RwLock<HashMap<String, VoiceCollection>>>,
    /// Quality assessor for quality updates
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity measurer for similarity analysis
    similarity_measurer: Arc<SimilarityMeasurer>,
    /// Consent manager for consent verification
    consent_manager: Arc<ConsentManager>,
    /// Search index for fast text search
    search_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,
}

impl Default for VoiceSearchQuery {
    fn default() -> Self {
        Self {
            text: None,
            category: None,
            languages: Vec::new(),
            required_tags: Vec::new(),
            optional_tags: Vec::new(),
            quality_range: None,
            rating_range: None,
            date_range: None,
            consent_status: None,
            min_usage_count: None,
            sort_by: SortCriteria::ModifiedAt,
            sort_order: SortOrder::Descending,
            limit: Some(50),
            offset: None,
        }
    }
}

impl Default for VoiceUsageStats {
    fn default() -> Self {
        Self {
            synthesis_count: 0,
            total_duration: Duration::from_secs(0),
            avg_quality: 0.0,
            preferred_method: None,
            usage_by_context: HashMap::new(),
            avg_processing_time: Duration::from_secs(0),
        }
    }
}

impl VoiceLibraryManager {
    /// Create new voice library manager
    pub async fn new<P: AsRef<Path>>(library_path: P) -> Result<Self> {
        let library_path = library_path.as_ref().to_path_buf();

        // Ensure library directory exists
        fs::create_dir_all(&library_path)
            .await
            .map_err(|e| Error::Io(e))?;

        let quality_assessor = Arc::new(CloningQualityAssessor::new()?);
        let similarity_config = crate::similarity::SimilarityConfig::default();
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(similarity_config));
        let consent_manager = Arc::new(ConsentManager::new());

        let manager = Self {
            library_path,
            voices: Arc::new(RwLock::new(HashMap::new())),
            collections: Arc::new(RwLock::new(HashMap::new())),
            quality_assessor,
            similarity_measurer,
            consent_manager,
            search_index: Arc::new(RwLock::new(HashMap::new())),
        };

        // Load existing library
        manager.load_library().await?;

        Ok(manager)
    }

    /// Load library from disk
    async fn load_library(&self) -> Result<()> {
        let voices_file = self.library_path.join("voices.json");
        let collections_file = self.library_path.join("collections.json");

        // Load voices
        if voices_file.exists() {
            let content = fs::read_to_string(&voices_file)
                .await
                .map_err(|e| Error::Io(e))?;

            let voices: HashMap<String, VoiceLibraryEntry> =
                serde_json::from_str(&content).map_err(|e| Error::Serialization(e))?;

            {
                let mut voices_lock = self.voices.write().unwrap();
                *voices_lock = voices;
            }

            // Rebuild search index
            self.rebuild_search_index().await;
        }

        // Load collections
        if collections_file.exists() {
            let content = fs::read_to_string(&collections_file)
                .await
                .map_err(|e| Error::Io(e))?;

            let collections: HashMap<String, VoiceCollection> =
                serde_json::from_str(&content).map_err(|e| Error::Serialization(e))?;

            let mut collections_lock = self.collections.write().unwrap();
            *collections_lock = collections;
        }

        Ok(())
    }

    /// Save library to disk
    async fn save_library(&self) -> Result<()> {
        let voices_file = self.library_path.join("voices.json");
        let collections_file = self.library_path.join("collections.json");

        // Save voices
        {
            let voices = self.voices.read().unwrap();
            let content =
                serde_json::to_string_pretty(&*voices).map_err(|e| Error::Serialization(e))?;

            fs::write(&voices_file, content)
                .await
                .map_err(|e| Error::Io(e))?;
        }

        // Save collections
        {
            let collections = self.collections.read().unwrap();
            let content =
                serde_json::to_string_pretty(&*collections).map_err(|e| Error::Serialization(e))?;

            fs::write(&collections_file, content)
                .await
                .map_err(|e| Error::Io(e))?;
        }

        Ok(())
    }

    /// Add voice to library
    pub async fn add_voice(&self, mut entry: VoiceLibraryEntry) -> Result<String> {
        // Ensure unique ID
        if entry.id.is_empty() {
            entry.id = format!(
                "voice_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
        }

        // Set timestamps
        let now = SystemTime::now();
        entry.created_at = now;
        entry.modified_at = now;
        entry.last_accessed = now;

        // Create storage directory for voice files
        let voice_dir = self.library_path.join("voices").join(&entry.id);
        fs::create_dir_all(&voice_dir)
            .await
            .map_err(|e| Error::Io(e))?;

        entry.storage_path = voice_dir;

        // Assess quality if not provided
        if entry.quality_metrics.overall_score == 0.0 {
            entry.quality_metrics = self.assess_voice_quality(&entry.speaker_profile).await?;
        }

        // Save voice files
        self.save_voice_files(&entry).await?;

        let voice_id = entry.id.clone();

        // Add to library
        {
            let mut voices = self.voices.write().unwrap();
            voices.insert(voice_id.clone(), entry);
        }

        // Update search index
        self.update_search_index(&voice_id).await;

        // Save library
        self.save_library().await?;

        Ok(voice_id)
    }

    /// Get voice by ID
    pub async fn get_voice(&self, voice_id: &str) -> Result<Option<VoiceLibraryEntry>> {
        let voices = self.voices.read().unwrap();
        if let Some(mut entry) = voices.get(voice_id).cloned() {
            // Update access statistics
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;

            // Update in memory (would need write lock for persistence)
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    /// Update voice entry
    pub async fn update_voice(
        &self,
        voice_id: &str,
        updates: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let mut voices = self.voices.write().unwrap();
        if let Some(entry) = voices.get_mut(voice_id) {
            // Apply updates
            for (key, value) in updates {
                match key.as_str() {
                    "name" => {
                        if let Some(name) = value.as_str() {
                            entry.name = name.to_string();
                        }
                    }
                    "description" => {
                        entry.description = value.as_str().map(|s| s.to_string());
                    }
                    "tags" => {
                        if let Some(tags_array) = value.as_array() {
                            entry.tags = tags_array
                                .iter()
                                .filter_map(|v| v.as_str())
                                .map(|s| s.to_string())
                                .collect();
                        }
                    }
                    "rating" => {
                        if let Some(rating) = value.as_u64() {
                            entry.rating = Some(rating.min(5) as u8);
                        }
                    }
                    "notes" => {
                        entry.notes = value.as_str().map(|s| s.to_string());
                    }
                    "category" => {
                        if let Ok(category) = serde_json::from_value::<VoiceCategory>(value) {
                            entry.category = category;
                        }
                    }
                    _ => {} // Ignore unknown fields
                }
            }

            entry.modified_at = SystemTime::now();

            // Update search index
            drop(voices); // Release lock
            self.update_search_index(voice_id).await;

            // Save library
            self.save_library().await?;

            Ok(())
        } else {
            Err(Error::Validation(format!("Voice not found: {voice_id}")))
        }
    }

    /// Delete voice from library
    pub async fn delete_voice(&self, voice_id: &str) -> Result<()> {
        let mut voices = self.voices.write().unwrap();
        if let Some(entry) = voices.remove(voice_id) {
            // Delete voice files
            if entry.storage_path.exists() {
                fs::remove_dir_all(&entry.storage_path)
                    .await
                    .map_err(|e| Error::Io(e))?;
            }

            // Remove from collections
            drop(voices); // Release lock
            self.remove_voice_from_collections(voice_id).await?;

            // Update search index
            self.remove_from_search_index(voice_id).await;

            // Save library
            self.save_library().await?;

            Ok(())
        } else {
            Err(Error::Validation(format!("Voice not found: {voice_id}")))
        }
    }

    /// Search voices in library
    pub async fn search_voices(&self, query: &VoiceSearchQuery) -> Result<Vec<VoiceLibraryEntry>> {
        let voices = self.voices.read().unwrap();
        let mut results: Vec<VoiceLibraryEntry> = voices
            .values()
            .filter(|entry| self.matches_query(entry, query))
            .cloned()
            .collect();

        // Sort results
        self.sort_results(&mut results, &query.sort_by, &query.sort_order);

        // Apply pagination
        if let Some(offset) = query.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                results.clear();
            }
        }

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Check if entry matches search query
    fn matches_query(&self, entry: &VoiceLibraryEntry, query: &VoiceSearchQuery) -> bool {
        // Text search
        if let Some(ref text) = query.text {
            let text_lower = text.to_lowercase();
            let matches_text = entry.name.to_lowercase().contains(&text_lower)
                || entry
                    .description
                    .as_ref()
                    .map_or(false, |d| d.to_lowercase().contains(&text_lower))
                || entry
                    .tags
                    .iter()
                    .any(|tag| tag.to_lowercase().contains(&text_lower));

            if !matches_text {
                return false;
            }
        }

        // Category filter
        if let Some(ref category) = query.category {
            if entry.category != *category {
                return false;
            }
        }

        // Language filter
        if !query.languages.is_empty() {
            let has_matching_language = query
                .languages
                .iter()
                .any(|lang| entry.languages.contains(lang));

            if !has_matching_language {
                return false;
            }
        }

        // Required tags filter
        if !query.required_tags.is_empty() {
            let has_all_required = query
                .required_tags
                .iter()
                .all(|tag| entry.tags.contains(tag));

            if !has_all_required {
                return false;
            }
        }

        // Optional tags filter
        if !query.optional_tags.is_empty() {
            let has_any_optional = query
                .optional_tags
                .iter()
                .any(|tag| entry.tags.contains(tag));

            if !has_any_optional {
                return false;
            }
        }

        // Quality range filter
        if let Some((min_quality, max_quality)) = query.quality_range {
            if entry.quality_metrics.overall_score < min_quality
                || entry.quality_metrics.overall_score > max_quality
            {
                return false;
            }
        }

        // Rating range filter
        if let Some((min_rating, max_rating)) = query.rating_range {
            if let Some(rating) = entry.rating {
                if rating < min_rating || rating > max_rating {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Date range filter
        if let Some((start_date, end_date)) = query.date_range {
            if entry.created_at < start_date || entry.created_at > end_date {
                return false;
            }
        }

        // Consent status filter
        if let Some(ref consent_status) = query.consent_status {
            if entry.consent_status != *consent_status {
                return false;
            }
        }

        // Minimum usage count filter
        if let Some(min_usage) = query.min_usage_count {
            if entry.access_count < min_usage {
                return false;
            }
        }

        true
    }

    /// Sort search results
    fn sort_results(
        &self,
        results: &mut Vec<VoiceLibraryEntry>,
        sort_by: &SortCriteria,
        sort_order: &SortOrder,
    ) {
        let ascending = *sort_order == SortOrder::Ascending;

        results.sort_by(|a, b| {
            let comparison = match sort_by {
                SortCriteria::Name => a.name.cmp(&b.name),
                SortCriteria::CreatedAt => a.created_at.cmp(&b.created_at),
                SortCriteria::ModifiedAt => a.modified_at.cmp(&b.modified_at),
                SortCriteria::LastAccessed => a.last_accessed.cmp(&b.last_accessed),
                SortCriteria::AccessCount => a.access_count.cmp(&b.access_count),
                SortCriteria::Quality => a
                    .quality_metrics
                    .overall_score
                    .partial_cmp(&b.quality_metrics.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortCriteria::Rating => a.rating.cmp(&b.rating),
                SortCriteria::FileSize => a.file_size.cmp(&b.file_size),
                SortCriteria::Relevance => std::cmp::Ordering::Equal, // Would need relevance scoring
            };

            if ascending {
                comparison
            } else {
                comparison.reverse()
            }
        });
    }

    /// Create voice collection
    pub async fn create_collection(
        &self,
        name: String,
        description: Option<String>,
        owner: String,
    ) -> Result<String> {
        let collection_id = format!(
            "collection_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let collection = VoiceCollection {
            id: collection_id.clone(),
            name,
            description,
            voice_ids: Vec::new(),
            tags: HashSet::new(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            owner,
            is_public: false,
            icon_path: None,
        };

        let mut collections = self.collections.write().unwrap();
        collections.insert(collection_id.clone(), collection);

        drop(collections);
        self.save_library().await?;

        Ok(collection_id)
    }

    /// Add voice to collection
    pub async fn add_voice_to_collection(&self, collection_id: &str, voice_id: &str) -> Result<()> {
        // Verify voice exists
        {
            let voices = self.voices.read().unwrap();
            if !voices.contains_key(voice_id) {
                return Err(Error::Validation(format!("Voice not found: {voice_id}")));
            }
        }

        // Add to collection
        let mut collections = self.collections.write().unwrap();
        if let Some(collection) = collections.get_mut(collection_id) {
            if !collection.voice_ids.contains(&voice_id.to_string()) {
                collection.voice_ids.push(voice_id.to_string());
                collection.modified_at = SystemTime::now();
            }

            drop(collections);
            self.save_library().await?;
            Ok(())
        } else {
            Err(Error::Validation(format!(
                "Collection not found: {}",
                collection_id
            )))
        }
    }

    /// Remove voice from collections
    async fn remove_voice_from_collections(&self, voice_id: &str) -> Result<()> {
        let mut collections = self.collections.write().unwrap();
        for collection in collections.values_mut() {
            collection.voice_ids.retain(|id| id != voice_id);
            collection.modified_at = SystemTime::now();
        }
        Ok(())
    }

    /// Get library statistics
    pub async fn get_statistics(&self) -> LibraryStatistics {
        let voices = self.voices.read().unwrap();

        let total_voices = voices.len();
        let total_storage_size = voices.values().map(|v| v.file_size).sum();

        // Group by category
        let mut voices_by_category = HashMap::new();
        for entry in voices.values() {
            *voices_by_category
                .entry(entry.category.clone())
                .or_insert(0) += 1;
        }

        // Group by language
        let mut voices_by_language = HashMap::new();
        for entry in voices.values() {
            for language in &entry.languages {
                *voices_by_language.entry(language.clone()).or_insert(0) += 1;
            }
        }

        // Calculate average quality
        let average_quality = if total_voices > 0 {
            voices
                .values()
                .map(|v| v.quality_metrics.overall_score)
                .sum::<f32>()
                / total_voices as f32
        } else {
            0.0
        };

        // Most popular voices (top 10 by access count)
        let mut popular_voices: Vec<_> = voices.values().collect();
        popular_voices.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        let most_popular = popular_voices
            .iter()
            .take(10)
            .map(|v| v.id.clone())
            .collect();

        // Recently added voices (last 10)
        let mut recent_voices: Vec<_> = voices.values().collect();
        recent_voices.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        let recently_added = recent_voices
            .iter()
            .take(10)
            .map(|v| v.id.clone())
            .collect();

        // Quality distribution (0.0-0.9 -> 0, 0.1-0.19 -> 1, etc.)
        let mut quality_distribution = BTreeMap::new();
        for entry in voices.values() {
            let bucket = ((entry.quality_metrics.overall_score * 10.0) as u8).min(9);
            *quality_distribution.entry(bucket).or_insert(0) += 1;
        }

        // Consent status distribution
        let mut consent_distribution = HashMap::new();
        for entry in voices.values() {
            *consent_distribution
                .entry(entry.consent_status.clone())
                .or_insert(0) += 1;
        }

        LibraryStatistics {
            total_voices,
            total_storage_size,
            voices_by_category,
            voices_by_language,
            average_quality,
            most_popular,
            recently_added,
            quality_distribution,
            consent_distribution,
        }
    }

    /// Perform batch operation
    pub async fn batch_operation(&self, operation: BatchOperation) -> Result<BatchOperationResult> {
        let operation_id = format!(
            "batch_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let start_time = std::time::Instant::now();
        let mut item_results = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        match operation {
            BatchOperation::Delete { voice_ids } => {
                for voice_id in voice_ids {
                    match self.delete_voice(&voice_id).await {
                        Ok(_) => {
                            success_count += 1;
                            item_results.push(BatchItemResult {
                                item_id: voice_id,
                                success: true,
                                error_message: None,
                                metadata: HashMap::new(),
                            });
                        }
                        Err(e) => {
                            failure_count += 1;
                            item_results.push(BatchItemResult {
                                item_id: voice_id,
                                success: false,
                                error_message: Some(e.to_string()),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }

            BatchOperation::UpdateTags {
                voice_ids,
                add_tags,
                remove_tags,
            } => {
                for voice_id in voice_ids {
                    let mut updates = HashMap::new();

                    // Get current tags
                    if let Ok(Some(entry)) = self.get_voice(&voice_id).await {
                        let mut new_tags = entry.tags.clone();

                        // Add new tags
                        for tag in &add_tags {
                            new_tags.insert(tag.clone());
                        }

                        // Remove specified tags
                        for tag in &remove_tags {
                            new_tags.remove(tag);
                        }

                        let tags_array: Vec<serde_json::Value> = new_tags
                            .iter()
                            .map(|tag| serde_json::Value::String(tag.clone()))
                            .collect();

                        updates.insert("tags".to_string(), serde_json::Value::Array(tags_array));

                        match self.update_voice(&voice_id, updates).await {
                            Ok(_) => {
                                success_count += 1;
                                item_results.push(BatchItemResult {
                                    item_id: voice_id,
                                    success: true,
                                    error_message: None,
                                    metadata: HashMap::new(),
                                });
                            }
                            Err(e) => {
                                failure_count += 1;
                                item_results.push(BatchItemResult {
                                    item_id: voice_id,
                                    success: false,
                                    error_message: Some(e.to_string()),
                                    metadata: HashMap::new(),
                                });
                            }
                        }
                    } else {
                        failure_count += 1;
                        item_results.push(BatchItemResult {
                            item_id: voice_id,
                            success: false,
                            error_message: Some("Voice not found".to_string()),
                            metadata: HashMap::new(),
                        });
                    }
                }
            }

            BatchOperation::UpdateCategory {
                voice_ids,
                new_category,
            } => {
                for voice_id in voice_ids {
                    let mut updates = HashMap::new();
                    updates.insert("category".to_string(), serde_json::to_value(&new_category)?);

                    match self.update_voice(&voice_id, updates).await {
                        Ok(_) => {
                            success_count += 1;
                            item_results.push(BatchItemResult {
                                item_id: voice_id,
                                success: true,
                                error_message: None,
                                metadata: HashMap::new(),
                            });
                        }
                        Err(e) => {
                            failure_count += 1;
                            item_results.push(BatchItemResult {
                                item_id: voice_id,
                                success: false,
                                error_message: Some(e.to_string()),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }

            _ => {
                return Err(Error::Processing(
                    "Batch operation not yet implemented".to_string(),
                ));
            }
        }

        let duration = start_time.elapsed();

        Ok(BatchOperationResult {
            operation_id,
            success_count,
            failure_count,
            item_results,
            duration,
            timestamp: SystemTime::now(),
        })
    }

    /// Assess voice quality for library entry
    async fn assess_voice_quality(
        &self,
        speaker_profile: &SpeakerProfile,
    ) -> Result<QualityMetrics> {
        // Create mock result for quality assessment
        if let Some(sample) = speaker_profile.samples.first() {
            let mock_result = VoiceCloneResult {
                request_id: "quality_assessment".to_string(),
                audio: sample.audio.clone(),
                sample_rate: sample.sample_rate,
                quality_metrics: HashMap::new(),
                similarity_score: 1.0,
                processing_time: Duration::from_secs(0),
                method_used: CloningMethod::FewShot,
                success: true,
                error_message: None,
                cross_lingual_info: None,
                timestamp: SystemTime::now(),
            };

            // Create voice samples for quality assessment
            let original_sample = VoiceSample::new("original".to_string(), vec![0.0; 1000], 16000);
            let cloned_sample = VoiceSample::new(
                "cloned".to_string(),
                mock_result.audio.clone(),
                mock_result.sample_rate,
            );
            let mut quality_assessor = CloningQualityAssessor::new()?;
            quality_assessor
                .assess_quality(&original_sample, &cloned_sample)
                .await
        } else {
            Ok(QualityMetrics::default())
        }
    }

    /// Save voice files to storage
    async fn save_voice_files(&self, entry: &VoiceLibraryEntry) -> Result<()> {
        let profile_file = entry.storage_path.join("profile.json");
        let profile_json = serde_json::to_string_pretty(&entry.speaker_profile)
            .map_err(|e| Error::Serialization(e))?;

        fs::write(&profile_file, profile_json)
            .await
            .map_err(|e| Error::Io(e))?;

        // Save individual samples
        for (i, sample) in entry.speaker_profile.samples.iter().enumerate() {
            let sample_file = entry.storage_path.join(format!("sample_{i}.json"));
            let sample_json =
                serde_json::to_string_pretty(sample).map_err(|e| Error::Serialization(e))?;

            fs::write(&sample_file, sample_json)
                .await
                .map_err(|e| Error::Io(e))?;
        }

        Ok(())
    }

    /// Rebuild search index
    async fn rebuild_search_index(&self) {
        let voices = self.voices.read().unwrap();
        let mut search_index = self.search_index.write().unwrap();
        search_index.clear();

        for (voice_id, entry) in voices.iter() {
            let mut terms = HashSet::new();

            // Index name words
            for word in entry.name.split_whitespace() {
                terms.insert(word.to_lowercase());
            }

            // Index description words
            if let Some(ref description) = entry.description {
                for word in description.split_whitespace() {
                    terms.insert(word.to_lowercase());
                }
            }

            // Index tags
            for tag in &entry.tags {
                terms.insert(tag.to_lowercase());
            }

            // Index languages
            for language in &entry.languages {
                terms.insert(language.to_lowercase());
            }

            for term in terms {
                search_index
                    .entry(term)
                    .or_insert_with(HashSet::new)
                    .insert(voice_id.clone());
            }
        }
    }

    /// Update search index for specific voice
    async fn update_search_index(&self, voice_id: &str) {
        let voices = self.voices.read().unwrap();
        if let Some(entry) = voices.get(voice_id) {
            let mut search_index = self.search_index.write().unwrap();

            // Remove old entries for this voice
            for voice_set in search_index.values_mut() {
                voice_set.remove(voice_id);
            }

            // Add new entries
            let mut terms = HashSet::new();

            // Index name words
            for word in entry.name.split_whitespace() {
                terms.insert(word.to_lowercase());
            }

            // Index description words
            if let Some(ref description) = entry.description {
                for word in description.split_whitespace() {
                    terms.insert(word.to_lowercase());
                }
            }

            // Index tags
            for tag in &entry.tags {
                terms.insert(tag.to_lowercase());
            }

            // Index languages
            for language in &entry.languages {
                terms.insert(language.to_lowercase());
            }

            for term in terms {
                search_index
                    .entry(term)
                    .or_insert_with(HashSet::new)
                    .insert(voice_id.to_string());
            }
        }
    }

    /// Remove voice from search index
    async fn remove_from_search_index(&self, voice_id: &str) {
        let mut search_index = self.search_index.write().unwrap();
        for voice_set in search_index.values_mut() {
            voice_set.remove(voice_id);
        }
    }

    /// Get all collections
    pub async fn get_collections(&self) -> Vec<VoiceCollection> {
        let collections = self.collections.read().unwrap();
        collections.values().cloned().collect()
    }

    /// Get collection by ID
    pub async fn get_collection(&self, collection_id: &str) -> Result<Option<VoiceCollection>> {
        let collections = self.collections.read().unwrap();
        Ok(collections.get(collection_id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_library_creation() {
        let temp_dir = TempDir::new().unwrap();
        let library_path = temp_dir.path();

        let library = VoiceLibraryManager::new(library_path).await.unwrap();

        // Verify library directory structure
        assert!(library_path.exists());
        assert!(library_path.is_dir());
    }

    #[tokio::test]
    async fn test_voice_management() {
        let temp_dir = TempDir::new().unwrap();
        let library = VoiceLibraryManager::new(temp_dir.path()).await.unwrap();

        // Create test voice entry
        let entry = VoiceLibraryEntry {
            id: "test_voice".to_string(),
            name: "Test Voice".to_string(),
            description: Some("Test voice description".to_string()),
            speaker_profile: SpeakerProfile::default(),
            category: VoiceCategory::Personal,
            languages: vec!["en".to_string()],
            tags: {
                let mut tags = HashSet::new();
                tags.insert("test".to_string());
                tags.insert("demo".to_string());
                tags
            },
            quality_metrics: QualityMetrics::default(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            file_size: 1024,
            storage_path: PathBuf::new(),
            rating: Some(4),
            notes: None,
            version: VoiceVersion {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                description: "Initial version".to_string(),
                parent_version: None,
                tags: vec!["stable".to_string()],
            },
            consent_status: ConsentStatus::Valid,
            usage_stats: VoiceUsageStats::default(),
        };

        // Add voice to library
        let voice_id = library.add_voice(entry).await.unwrap();
        assert!(!voice_id.is_empty());

        // Retrieve voice
        let retrieved = library.get_voice(&voice_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.name, "Test Voice");
        assert_eq!(retrieved_entry.category, VoiceCategory::Personal);

        // Update voice
        let mut updates = HashMap::new();
        updates.insert(
            "name".to_string(),
            serde_json::Value::String("Updated Test Voice".to_string()),
        );
        updates.insert(
            "rating".to_string(),
            serde_json::Value::Number(serde_json::Number::from(5)),
        );

        library.update_voice(&voice_id, updates).await.unwrap();

        // Verify update
        let updated = library.get_voice(&voice_id).await.unwrap().unwrap();
        assert_eq!(updated.name, "Updated Test Voice");
        assert_eq!(updated.rating, Some(5));

        // Delete voice
        library.delete_voice(&voice_id).await.unwrap();
        let deleted = library.get_voice(&voice_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_voice_search() {
        let temp_dir = TempDir::new().unwrap();
        let library = VoiceLibraryManager::new(temp_dir.path()).await.unwrap();

        // Add multiple test voices
        for i in 0..5 {
            let entry = VoiceLibraryEntry {
                id: format!("voice_{i}"),
                name: format!("Voice {i}"),
                description: Some(format!("Description for voice {i}")),
                speaker_profile: SpeakerProfile::default(),
                category: if i % 2 == 0 {
                    VoiceCategory::Personal
                } else {
                    VoiceCategory::Character
                },
                languages: vec!["en".to_string()],
                tags: {
                    let mut tags = HashSet::new();
                    tags.insert(format!("tag_{i}"));
                    if i % 2 == 0 {
                        tags.insert("even".to_string());
                    }
                    tags
                },
                quality_metrics: QualityMetrics {
                    overall_score: 0.5 + (i as f32 * 0.1),
                    ..Default::default()
                },
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                access_count: i as u64,
                file_size: 1024 * (i as u64 + 1),
                storage_path: PathBuf::new(),
                rating: Some((i % 5 + 1) as u8),
                notes: None,
                version: VoiceVersion {
                    version: "1.0.0".to_string(),
                    created_at: SystemTime::now(),
                    description: "Initial version".to_string(),
                    parent_version: None,
                    tags: vec![],
                },
                consent_status: ConsentStatus::Valid,
                usage_stats: VoiceUsageStats::default(),
            };

            library.add_voice(entry).await.unwrap();
        }

        // Test text search
        let query = VoiceSearchQuery {
            text: Some("Voice 2".to_string()),
            ..Default::default()
        };
        let results = library.search_voices(&query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Voice 2");

        // Test category filter
        let query = VoiceSearchQuery {
            category: Some(VoiceCategory::Personal),
            ..Default::default()
        };
        let results = library.search_voices(&query).await.unwrap();
        assert_eq!(results.len(), 3); // Voices 0, 2, 4

        // Test tag filter
        let query = VoiceSearchQuery {
            required_tags: vec!["even".to_string()],
            ..Default::default()
        };
        let results = library.search_voices(&query).await.unwrap();
        assert_eq!(results.len(), 3); // Voices 0, 2, 4

        // Test quality range filter
        let query = VoiceSearchQuery {
            quality_range: Some((0.7, 1.0)),
            ..Default::default()
        };
        let results = library.search_voices(&query).await.unwrap();
        assert_eq!(results.len(), 2); // Voices 3, 4
    }

    #[tokio::test]
    async fn test_collections() {
        let temp_dir = TempDir::new().unwrap();
        let library = VoiceLibraryManager::new(temp_dir.path()).await.unwrap();

        // Add test voice
        let entry = VoiceLibraryEntry {
            id: "test_voice".to_string(),
            name: "Test Voice".to_string(),
            description: None,
            speaker_profile: SpeakerProfile::default(),
            category: VoiceCategory::Personal,
            languages: vec!["en".to_string()],
            tags: HashSet::new(),
            quality_metrics: QualityMetrics::default(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            file_size: 1024,
            storage_path: PathBuf::new(),
            rating: None,
            notes: None,
            version: VoiceVersion {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                description: "Initial version".to_string(),
                parent_version: None,
                tags: vec![],
            },
            consent_status: ConsentStatus::Valid,
            usage_stats: VoiceUsageStats::default(),
        };

        let voice_id = library.add_voice(entry).await.unwrap();

        // Create collection
        let collection_id = library
            .create_collection(
                "Test Collection".to_string(),
                Some("Test collection description".to_string()),
                "test_user".to_string(),
            )
            .await
            .unwrap();

        // Add voice to collection
        library
            .add_voice_to_collection(&collection_id, &voice_id)
            .await
            .unwrap();

        // Verify collection
        let collection = library
            .get_collection(&collection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(collection.name, "Test Collection");
        assert_eq!(collection.voice_ids.len(), 1);
        assert_eq!(collection.voice_ids[0], voice_id);
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let library = VoiceLibraryManager::new(temp_dir.path()).await.unwrap();

        // Add multiple test voices
        let mut voice_ids = Vec::new();
        for i in 0..3 {
            let entry = VoiceLibraryEntry {
                id: format!("voice_{i}"),
                name: format!("Voice {i}"),
                description: None,
                speaker_profile: SpeakerProfile::default(),
                category: VoiceCategory::Personal,
                languages: vec!["en".to_string()],
                tags: HashSet::new(),
                quality_metrics: QualityMetrics::default(),
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                access_count: 0,
                file_size: 1024,
                storage_path: PathBuf::new(),
                rating: None,
                notes: None,
                version: VoiceVersion {
                    version: "1.0.0".to_string(),
                    created_at: SystemTime::now(),
                    description: "Initial version".to_string(),
                    parent_version: None,
                    tags: vec![],
                },
                consent_status: ConsentStatus::Valid,
                usage_stats: VoiceUsageStats::default(),
            };

            let voice_id = library.add_voice(entry).await.unwrap();
            voice_ids.push(voice_id);
        }

        // Test batch tag update
        let operation = BatchOperation::UpdateTags {
            voice_ids: voice_ids.clone(),
            add_tags: vec!["batch_test".to_string()],
            remove_tags: vec![],
        };

        let result = library.batch_operation(operation).await.unwrap();
        assert_eq!(result.success_count, 3);
        assert_eq!(result.failure_count, 0);

        // Verify tags were added
        for voice_id in &voice_ids {
            let voice = library.get_voice(voice_id).await.unwrap().unwrap();
            assert!(voice.tags.contains("batch_test"));
        }
    }

    #[tokio::test]
    async fn test_library_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let library = VoiceLibraryManager::new(temp_dir.path()).await.unwrap();

        // Add test voices with different categories
        for i in 0..5 {
            let entry = VoiceLibraryEntry {
                id: format!("voice_{i}"),
                name: format!("Voice {i}"),
                description: None,
                speaker_profile: SpeakerProfile::default(),
                category: if i < 3 {
                    VoiceCategory::Personal
                } else {
                    VoiceCategory::Character
                },
                languages: vec!["en".to_string()],
                tags: HashSet::new(),
                quality_metrics: QualityMetrics {
                    overall_score: 0.5 + (i as f32 * 0.1),
                    ..Default::default()
                },
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                access_count: i as u64,
                file_size: 1024 * (i as u64 + 1),
                storage_path: PathBuf::new(),
                rating: Some((i % 5 + 1) as u8),
                notes: None,
                version: VoiceVersion {
                    version: "1.0.0".to_string(),
                    created_at: SystemTime::now(),
                    description: "Initial version".to_string(),
                    parent_version: None,
                    tags: vec![],
                },
                consent_status: ConsentStatus::Valid,
                usage_stats: VoiceUsageStats::default(),
            };

            library.add_voice(entry).await.unwrap();
        }

        // Get statistics
        let stats = library.get_statistics().await;

        assert_eq!(stats.total_voices, 5);
        assert_eq!(stats.total_storage_size, 1024 + 2048 + 3072 + 4096 + 5120); // Sum of file sizes
        assert_eq!(
            stats.voices_by_category.get(&VoiceCategory::Personal),
            Some(&3)
        );
        assert_eq!(
            stats.voices_by_category.get(&VoiceCategory::Character),
            Some(&2)
        );
        assert_eq!(stats.voices_by_language.get("en"), Some(&5));
        assert!(stats.average_quality > 0.0);
    }
}
