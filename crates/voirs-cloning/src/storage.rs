//! Efficient storage system for thousands of cloned voices
//!
//! This module provides a comprehensive storage solution for voice cloning models,
//! including efficient data structures, compression, caching, and maintenance capabilities
//! optimized for handling large numbers of voice profiles and their associated models.

use crate::{
    embedding::SpeakerEmbedding, quality::QualityMetrics, Error, Result, SpeakerProfile,
    VoiceCloneResult, VoiceSample,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;

/// Name of the on-disk metadata index file, stored directly under the
/// storage root so it survives process restarts (see [`PersistedMetadataIndex`]).
const METADATA_INDEX_FILE_NAME: &str = "metadata_index.json";

/// Retention window for per-model recent-access history and the
/// hot/warm boundary used by [`VoiceModelStorage::update_storage_tiers`].
const RECENT_ACCESS_RETENTION: Duration = Duration::from_secs(30 * 24 * 3600);
const HOT_TIER_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 3600);
const WARM_TIER_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 3600);

/// Comprehensive voice model storage system
#[derive(Debug)]
pub struct VoiceModelStorage {
    /// Storage configuration
    config: StorageConfig,
    /// Root storage directory
    storage_root: PathBuf,
    /// In-memory metadata index for fast access
    metadata_index: Arc<RwLock<MetadataIndex>>,
    /// LRU cache for frequently accessed models
    model_cache: Arc<RwLock<ModelCache>>,
    /// Storage statistics and health monitoring
    statistics: Arc<RwLock<StorageStatistics>>,
    /// Background maintenance task handles
    maintenance_tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    /// Total number of storage operations observed by [`Self::update_storage_statistics`]
    operation_count: Arc<AtomicU64>,
    /// Cumulative processing time (ms) across those operations, for a real
    /// (non-fabricated) running average response time.
    operation_total_time_ms: Arc<AtomicU64>,
}

/// Storage system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Maximum number of models to keep in memory cache
    pub max_cache_size: usize,
    /// Enable compression for stored models
    pub enable_compression: bool,
    /// Compression level (0-9, higher = better compression, slower)
    pub compression_level: u32,
    /// Maximum file size per model (bytes)
    pub max_model_size: u64,
    /// Enable automatic cleanup of old/unused models
    pub enable_auto_cleanup: bool,
    /// Age threshold for cleanup (days)
    pub cleanup_age_threshold_days: u64,
    /// Enable storage encryption
    pub enable_encryption: bool,
    /// Background maintenance interval
    pub maintenance_interval: Duration,
    /// Enable deduplication of similar models
    pub enable_deduplication: bool,
    /// Similarity threshold for deduplication (0.0-1.0)
    pub deduplication_threshold: f32,
    /// Enable tiered storage (hot/warm/cold)
    pub enable_tiered_storage: bool,
    /// Backup retention policy
    pub backup_retention_days: u64,
}

/// In-memory metadata index for fast lookups
#[derive(Debug, Clone, Default)]
struct MetadataIndex {
    /// Speaker ID to metadata mapping
    speaker_metadata: HashMap<String, StoredModelMetadata>,
    /// Category-based indexes for efficient queries
    category_index: HashMap<String, Vec<String>>,
    /// Time-based indexes for cleanup and maintenance
    creation_time_index: BTreeMap<SystemTime, Vec<String>>,
    /// Access frequency tracking
    access_frequency: HashMap<String, AccessStats>,
    /// Size-based index for storage optimization
    size_index: BTreeMap<u64, Vec<String>>,
}

impl MetadataIndex {
    /// Insert (or replace) a model's metadata, keeping the derived indices
    /// (`category_index`, `creation_time_index`, `size_index`,
    /// `access_frequency`) consistent with the primary map.
    fn insert(&mut self, metadata: StoredModelMetadata) {
        let model_id = metadata.model_id.clone();

        // Drop any previous entry's derived-index contributions first, so
        // re-inserting (e.g. after an update, or on index load) never leaves
        // stale entries behind.
        self.remove_derived(&model_id);

        for tag in &metadata.tags {
            self.category_index
                .entry(tag.clone())
                .or_default()
                .push(model_id.clone());
        }
        self.creation_time_index
            .entry(metadata.storage_info.created_at)
            .or_default()
            .push(model_id.clone());
        self.size_index
            .entry(metadata.storage_info.file_size)
            .or_default()
            .push(model_id.clone());
        self.access_frequency
            .insert(model_id.clone(), metadata.access_stats.clone());

        self.speaker_metadata.insert(model_id, metadata);
    }

    /// Remove a model's contributions to the derived (non-primary) indices,
    /// without touching `speaker_metadata` itself.
    fn remove_derived(&mut self, model_id: &str) {
        let Some(existing) = self.speaker_metadata.get(model_id) else {
            return;
        };

        for tag in &existing.tags {
            if let Some(ids) = self.category_index.get_mut(tag) {
                ids.retain(|id| id != model_id);
                if ids.is_empty() {
                    self.category_index.remove(tag);
                }
            }
        }
        if let Some(ids) = self
            .creation_time_index
            .get_mut(&existing.storage_info.created_at)
        {
            ids.retain(|id| id != model_id);
            if ids.is_empty() {
                self.creation_time_index
                    .remove(&existing.storage_info.created_at);
            }
        }
        if let Some(ids) = self.size_index.get_mut(&existing.storage_info.file_size) {
            ids.retain(|id| id != model_id);
            if ids.is_empty() {
                self.size_index.remove(&existing.storage_info.file_size);
            }
        }
        self.access_frequency.remove(model_id);
    }

    /// Remove a model entirely: derived indices plus the primary map entry.
    fn remove(&mut self, model_id: &str) -> Option<StoredModelMetadata> {
        self.remove_derived(model_id);
        self.speaker_metadata.remove(model_id)
    }
}

/// LRU cache for frequently accessed models
#[derive(Debug)]
struct ModelCache {
    /// Cached models with access tracking
    cache: HashMap<String, CachedModel>,
    /// LRU order tracking
    access_queue: VecDeque<String>,
    /// Current cache size in bytes
    current_size: u64,
    /// Maximum cache size in bytes
    max_size: u64,
    /// Cache statistics
    stats: CacheStatistics,
}

/// Metadata for stored voice models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModelMetadata {
    /// Unique model identifier
    pub model_id: String,
    /// Original speaker profile information
    pub speaker_info: SpeakerInfo,
    /// Storage information
    pub storage_info: StorageInfo,
    /// Model quality metrics
    pub quality_metrics: Option<QualityMetrics>,
    /// Access statistics
    pub access_stats: AccessStats,
    /// Compression information
    pub compression_info: Option<CompressionInfo>,
    /// Tags for categorization and search
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
}

/// Speaker information for identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker identifier
    pub speaker_id: String,
    /// Speaker name (if available)
    pub name: Option<String>,
    /// Voice characteristics summary
    pub characteristics: VoiceCharacteristicsSummary,
    /// Supported languages
    pub languages: Vec<String>,
    /// Gender classification (if available)
    pub gender: Option<String>,
    /// Age group estimation (if available)
    pub age_group: Option<String>,
}

/// Storage-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// File path relative to storage root
    pub file_path: PathBuf,
    /// File size in bytes
    pub file_size: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Storage tier (hot/warm/cold)
    pub storage_tier: StorageTier,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Voice characteristics summary for storage optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristicsSummary {
    /// Average fundamental frequency
    pub average_f0: f32,
    /// Voice quality indicators
    pub quality_indicators: Vec<f32>,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Energy characteristics
    pub energy_stats: EnergyStats,
}

/// Energy statistics for voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyStats {
    pub mean: f32,
    pub std_dev: f32,
    pub dynamic_range: f32,
}

/// Access statistics for usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessStats {
    /// Total number of accesses
    pub access_count: u64,
    /// Last access timestamp
    pub last_access: SystemTime,
    /// Access frequency (accesses per day)
    pub access_frequency: f32,
    /// Recent access pattern (last 30 days)
    pub recent_accesses: VecDeque<SystemTime>,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes
    pub original_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Compression time
    pub compression_time: Duration,
}

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Frequently accessed, kept in fast storage
    Hot,
    /// Occasionally accessed, balanced storage
    Warm,
    /// Rarely accessed, archived storage
    Cold,
}

/// Compression algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Zstd,
    Lz4,
}

/// Cached model with metadata
#[derive(Debug)]
struct CachedModel {
    /// Model data
    data: Vec<u8>,
    /// Metadata
    metadata: StoredModelMetadata,
    /// Cache timestamp
    cached_at: SystemTime,
    /// Access count since cached
    access_count: u64,
    /// Size in bytes
    size: u64,
}

/// Cache performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache hit ratio
    pub hit_ratio: f32,
    /// Total evictions
    pub evictions: u64,
    /// Average load time (milliseconds)
    pub avg_load_time_ms: f32,
}

/// Storage system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatistics {
    /// Total number of stored models
    pub total_models: u64,
    /// Total storage size in bytes
    pub total_size: u64,
    /// Average model size in bytes
    pub avg_model_size: u64,
    /// Storage by tier distribution
    pub tier_distribution: HashMap<StorageTier, u64>,
    /// Compression statistics
    pub compression_stats: CompressionStatistics,
    /// Cache performance
    pub cache_stats: CacheStatistics,
    /// Maintenance statistics
    pub maintenance_stats: MaintenanceStatistics,
    /// Health indicators
    pub health_indicators: HealthIndicators,
}

/// Compression statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStatistics {
    /// Number of compressed models
    pub compressed_models: u64,
    /// Total original size
    pub total_original_size: u64,
    /// Total compressed size
    pub total_compressed_size: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Space saved in bytes
    pub space_saved: u64,
}

/// Maintenance operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaintenanceStatistics {
    /// Last maintenance run
    pub last_maintenance: Option<SystemTime>,
    /// Number of cleanup operations
    pub cleanup_operations: u64,
    /// Number of models cleaned up
    pub models_cleaned: u64,
    /// Space recovered in bytes
    pub space_recovered: u64,
    /// Deduplication operations
    pub deduplication_count: u64,
    /// Models deduplicated
    pub models_deduplicated: u64,
}

/// Storage system health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicators {
    /// Overall health score (0.0-1.0)
    pub health_score: f32,
    /// Storage utilization percentage
    pub storage_utilization: f32,
    /// Cache efficiency score
    pub cache_efficiency: f32,
    /// Error rate (errors per operation)
    pub error_rate: f32,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f32,
    /// Detected issues
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Storage operation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOperationResult {
    /// Operation success status
    pub success: bool,
    /// Model ID involved
    pub model_id: String,
    /// Operation type
    pub operation: StorageOperation,
    /// Processing time
    pub processing_time: Duration,
    /// Bytes affected
    pub bytes_affected: u64,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of storage operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageOperation {
    Store,
    Retrieve,
    Delete,
    Update,
    Compress,
    Migrate,
    Backup,
    Restore,
}

impl VoiceModelStorage {
    /// Create new voice model storage system
    pub async fn new(storage_root: PathBuf, config: StorageConfig) -> Result<Self> {
        // Ensure storage directory exists
        fs::create_dir_all(&storage_root)
            .map_err(|e| Error::Config(format!("Failed to create storage directory: {e}")))?;

        info!("Initializing voice model storage at: {:?}", storage_root);

        let storage = Self {
            config: config.clone(),
            storage_root,
            metadata_index: Arc::new(RwLock::new(MetadataIndex::default())),
            model_cache: Arc::new(RwLock::new(ModelCache::new(
                (config.max_cache_size * 1024 * 1024) as u64,
            ))),
            statistics: Arc::new(RwLock::new(StorageStatistics::default())),
            maintenance_tasks: Arc::new(Mutex::new(Vec::new())),
            operation_count: Arc::new(AtomicU64::new(0)),
            operation_total_time_ms: Arc::new(AtomicU64::new(0)),
        };

        // Load existing metadata index
        storage.load_metadata_index().await?;

        // Start background maintenance tasks
        if config.enable_auto_cleanup || config.enable_deduplication {
            storage.start_maintenance_tasks().await?;
        }

        info!("Voice model storage initialized successfully");
        Ok(storage)
    }

    /// Store a voice model with associated metadata
    pub async fn store_model(
        &self,
        speaker_profile: &SpeakerProfile,
        model_data: &[u8],
        quality_metrics: Option<QualityMetrics>,
        tags: Vec<String>,
    ) -> Result<StorageOperationResult> {
        let start_time = Instant::now();
        let model_id = Uuid::new_v4().to_string();

        debug!("Storing voice model: {}", model_id);

        // Check for deduplication if enabled
        if self.config.enable_deduplication {
            if let Some(existing_id) = self.find_similar_model(speaker_profile).await? {
                info!("Found similar model, using existing: {}", existing_id);
                return Ok(StorageOperationResult {
                    success: true,
                    model_id: existing_id,
                    operation: StorageOperation::Store,
                    processing_time: start_time.elapsed(),
                    bytes_affected: 0,
                    error_message: None,
                    metadata: [("deduplicated".to_string(), "true".to_string())].into(),
                });
            }
        }

        // Prepare storage path
        let file_path = self.generate_storage_path(&model_id)?;
        let full_path = self.storage_root.join(&file_path);

        // Ensure parent directory exists
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Processing(format!("Failed to create model directory: {e}")))?;
        }

        // Compress data if enabled
        let (final_data, compression_info) = if self.config.enable_compression {
            self.compress_model_data(model_data).await?
        } else {
            (model_data.to_vec(), None)
        };

        // Write model data to storage
        let mut file = File::create(&full_path)
            .map_err(|e| Error::Processing(format!("Failed to create model file: {e}")))?;
        file.write_all(&final_data)
            .map_err(|e| Error::Processing(format!("Failed to write model data: {e}")))?;

        // Calculate checksum
        let checksum = self.calculate_checksum(&final_data);

        // Create metadata
        let metadata = StoredModelMetadata {
            model_id: model_id.clone(),
            speaker_info: self.extract_speaker_info(speaker_profile),
            storage_info: StorageInfo {
                file_path: file_path.clone(),
                file_size: final_data.len() as u64,
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                storage_tier: StorageTier::Hot,
                checksum,
            },
            quality_metrics,
            access_stats: AccessStats {
                access_count: 0,
                last_access: SystemTime::now(),
                access_frequency: 0.0,
                recent_accesses: VecDeque::new(),
            },
            compression_info,
            tags,
            custom_metadata: HashMap::new(),
        };

        // Update metadata index
        self.update_metadata_index(&metadata).await?;

        // Update cache if there's space. The cache always holds
        // *decompressed* bytes (matching what `retrieve_model` returns on a
        // cache hit) - `model_data` (the caller's original bytes), not
        // `final_data` (the possibly-compressed on-disk bytes).
        if self.should_cache_model(&metadata).await {
            self.cache_model(&model_id, model_data, &metadata).await?;
        }

        let processing_time = start_time.elapsed();

        // Update statistics from the real post-store state (index + cache).
        self.update_storage_statistics(processing_time).await;

        info!(
            "Stored voice model {} in {:?} (size: {} bytes)",
            model_id,
            processing_time,
            final_data.len()
        );

        Ok(StorageOperationResult {
            success: true,
            model_id,
            operation: StorageOperation::Store,
            processing_time,
            bytes_affected: final_data.len() as u64,
            error_message: None,
            metadata: HashMap::new(),
        })
    }

    /// Retrieve a voice model by ID
    pub async fn retrieve_model(&self, model_id: &str) -> Result<(Vec<u8>, StoredModelMetadata)> {
        let start_time = Instant::now();

        debug!("Retrieving voice model: {}", model_id);

        // Check cache first
        if let Some((data, metadata)) = self.get_from_cache(model_id).await? {
            self.update_access_stats(model_id).await?;
            debug!("Retrieved model from cache: {}", model_id);
            return Ok((data, metadata));
        }

        // Load from storage
        let metadata = self
            .get_model_metadata(model_id)
            .await?
            .ok_or_else(|| Error::Processing(format!("Model not found: {model_id}")))?;

        let file_path = self.storage_root.join(&metadata.storage_info.file_path);
        let mut file = File::open(&file_path)
            .map_err(|e| Error::Processing(format!("Failed to open model file: {e}")))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Processing(format!("Failed to read model data: {e}")))?;

        // Verify checksum
        let checksum = self.calculate_checksum(&data);
        if checksum != metadata.storage_info.checksum {
            return Err(Error::Processing(format!(
                "Model data corrupted: {}",
                model_id
            )));
        }

        // Decompress if needed
        let final_data = if let Some(compression_info) = &metadata.compression_info {
            self.decompress_model_data(&data, compression_info.algorithm)
                .await?
        } else {
            data
        };

        // Update cache
        if self.should_cache_model(&metadata).await {
            self.cache_model(model_id, &final_data, &metadata).await?;
        }

        // Update access statistics
        self.update_access_stats(model_id).await?;

        let processing_time = start_time.elapsed();
        debug!(
            "Retrieved voice model {} in {:?} (size: {} bytes)",
            model_id,
            processing_time,
            final_data.len()
        );

        Ok((final_data, metadata))
    }

    /// Delete a voice model
    pub async fn delete_model(&self, model_id: &str) -> Result<StorageOperationResult> {
        let start_time = Instant::now();

        info!("Deleting voice model: {}", model_id);

        let metadata = self
            .get_model_metadata(model_id)
            .await?
            .ok_or_else(|| Error::Processing(format!("Model not found: {model_id}")))?;

        // Remove from cache
        self.remove_from_cache(model_id).await;

        // Delete file
        let file_path = self.storage_root.join(&metadata.storage_info.file_path);
        if file_path.exists() {
            fs::remove_file(&file_path)
                .map_err(|e| Error::Processing(format!("Failed to delete model file: {e}")))?;
        }

        // Remove from metadata index
        self.remove_from_metadata_index(model_id).await?;

        let processing_time = start_time.elapsed();

        // Update statistics from the real post-delete state (index + cache).
        self.update_storage_statistics(processing_time).await;

        info!("Deleted voice model {} in {:?}", model_id, processing_time);

        Ok(StorageOperationResult {
            success: true,
            model_id: model_id.to_string(),
            operation: StorageOperation::Delete,
            processing_time,
            bytes_affected: metadata.storage_info.file_size,
            error_message: None,
            metadata: HashMap::new(),
        })
    }

    /// List models with optional filtering
    pub async fn list_models(
        &self,
        filter: Option<ModelFilter>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<StoredModelMetadata>> {
        let index = self.metadata_index.read().await;
        let mut models: Vec<_> = index.speaker_metadata.values().cloned().collect();

        // Apply filters
        if let Some(filter) = filter {
            models = self.apply_filter(models, &filter);
        }

        // Sort by creation time (newest first)
        models.sort_by_key(|b| std::cmp::Reverse(b.storage_info.created_at));

        // Apply pagination
        let start = offset.unwrap_or(0);
        let end = if let Some(limit) = limit {
            (start + limit).min(models.len())
        } else {
            models.len()
        };

        Ok(models[start..end].to_vec())
    }

    /// Get storage statistics
    pub async fn get_statistics(&self) -> StorageStatistics {
        // Merge in the live cache statistics (updated on every get/put) rather
        // than only the snapshot taken at the last store/delete/maintenance
        // call, so hit/miss counts are never stale.
        let mut stats = self.statistics.read().await.clone();
        stats.cache_stats = self.model_cache.read().await.stats.clone();
        stats.health_indicators.cache_efficiency = stats.cache_stats.hit_ratio;
        stats
    }

    /// Perform maintenance operations
    pub async fn perform_maintenance(&self) -> Result<MaintenanceReport> {
        info!("Starting storage maintenance");
        let start_time = Instant::now();

        let mut report = MaintenanceReport {
            start_time: SystemTime::now(),
            operations_performed: Vec::new(),
            models_processed: 0,
            space_recovered: 0,
            errors: Vec::new(),
            duration: Duration::from_secs(0), // Will be updated at the end
        };

        // Cleanup old models if enabled. Only recorded as "performed" when at
        // least one model was actually removed - an operation that changed
        // nothing is not reported as having run.
        if self.config.enable_auto_cleanup {
            match self.cleanup_old_models().await {
                Ok((count, space)) => {
                    if count > 0 {
                        report.operations_performed.push("cleanup".to_string());
                    }
                    report.models_processed += count;
                    report.space_recovered += space;
                }
                Err(e) => report.errors.push(format!("Cleanup failed: {e}")),
            }
        }

        // Perform deduplication if enabled
        if self.config.enable_deduplication {
            match self.deduplicate_models().await {
                Ok((count, space)) => {
                    if count > 0 {
                        report
                            .operations_performed
                            .push("deduplication".to_string());
                    }
                    report.models_processed += count;
                    report.space_recovered += space;
                }
                Err(e) => report.errors.push(format!("Deduplication failed: {e}")),
            }
        }

        // Update storage tiers based on real access recency
        match self.update_storage_tiers().await {
            Ok(count) => {
                if count > 0 {
                    report.operations_performed.push("tier_update".to_string());
                }
                report.models_processed += count;
            }
            Err(e) => report.errors.push(format!("Tier update failed: {e}")),
        }

        // Optimize metadata index: self-heals stale entries and persists to disk.
        self.optimize_metadata_index().await?;
        report
            .operations_performed
            .push("index_optimization".to_string());

        report.duration = start_time.elapsed();

        // Refresh statistics so get_statistics() reflects the post-maintenance state.
        self.update_storage_statistics(report.duration).await;

        info!("Storage maintenance completed in {:?}", report.duration);

        Ok(report)
    }

    // Private implementation methods...

    /// Generate storage path for a model
    fn generate_storage_path(&self, model_id: &str) -> Result<PathBuf> {
        // Use hierarchical directory structure for better filesystem performance
        let prefix = &model_id[0..2];
        let subdir = &model_id[2..4];
        Ok(PathBuf::from(format!(
            "models/{}/{}/{}.voice",
            prefix, subdir, model_id
        )))
    }

    /// Extract speaker information from profile
    fn extract_speaker_info(&self, profile: &SpeakerProfile) -> SpeakerInfo {
        let characteristics = VoiceCharacteristicsSummary {
            average_f0: profile.characteristics.average_pitch,
            quality_indicators: vec![
                profile.characteristics.voice_quality.breathiness,
                profile.characteristics.voice_quality.roughness,
                profile.characteristics.voice_quality.brightness,
                profile.characteristics.voice_quality.warmth,
            ],
            spectral_centroid: 2000.0, // Default value
            energy_stats: EnergyStats {
                mean: profile.characteristics.average_energy,
                std_dev: 0.1,        // Default value
                dynamic_range: 40.0, // Default value
            },
        };

        SpeakerInfo {
            speaker_id: profile.id.clone(),
            name: Some(profile.name.clone()),
            characteristics,
            languages: profile.languages.clone(),
            gender: profile.characteristics.gender.map(|g| format!("{:?}", g)),
            age_group: profile
                .characteristics
                .age_group
                .map(|a| format!("{:?}", a)),
        }
    }

    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Path of the on-disk metadata index file for this storage root.
    fn index_file_path(&self) -> PathBuf {
        self.storage_root.join(METADATA_INDEX_FILE_NAME)
    }

    /// Load metadata index from storage.
    ///
    /// Reads the JSON index written by [`Self::persist_metadata_index`], if
    /// present, and rebuilds the in-memory derived indices from it. A missing
    /// index file (e.g. first run in a fresh storage root) is not an error -
    /// the storage simply starts empty.
    async fn load_metadata_index(&self) -> Result<()> {
        let index_path = self.index_file_path();
        if !index_path.exists() {
            debug!(
                "No existing metadata index at {:?}; starting with an empty index",
                index_path
            );
            return Ok(());
        }

        let data = fs::read_to_string(&index_path).map_err(|e| {
            Error::Config(format!("Failed to read metadata index {index_path:?}: {e}"))
        })?;
        let persisted: PersistedMetadataIndex = serde_json::from_str(&data).map_err(|e| {
            Error::Config(format!(
                "Failed to parse metadata index {index_path:?}: {e}"
            ))
        })?;

        let mut index = self.metadata_index.write().await;
        let loaded_count = persisted.models.len();
        for metadata in persisted.models {
            index.insert(metadata);
        }

        info!(
            "Loaded {} voice model(s) from metadata index at {:?}",
            loaded_count, index_path
        );
        Ok(())
    }

    /// Serialize the current metadata index to disk atomically (write to a
    /// unique temp sibling, then rename into place) so a crash mid-write can
    /// never leave a corrupt or partially-written index file.
    fn persist_metadata_index(&self, index: &MetadataIndex) -> Result<()> {
        let persisted = PersistedMetadataIndex {
            models: index.speaker_metadata.values().cloned().collect(),
        };
        let json = serde_json::to_vec_pretty(&persisted)
            .map_err(|e| Error::Processing(format!("Failed to serialize metadata index: {e}")))?;

        fs::create_dir_all(&self.storage_root)
            .map_err(|e| Error::Processing(format!("Failed to create storage root: {e}")))?;

        let index_path = self.index_file_path();
        let tmp_path = self.storage_root.join(format!(
            ".{METADATA_INDEX_FILE_NAME}.{}.tmp",
            Uuid::new_v4()
        ));

        {
            let mut file = File::create(&tmp_path)
                .map_err(|e| Error::Processing(format!("Failed to create temp index file: {e}")))?;
            file.write_all(&json)
                .map_err(|e| Error::Processing(format!("Failed to write temp index file: {e}")))?;
            file.sync_all()
                .map_err(|e| Error::Processing(format!("Failed to sync temp index file: {e}")))?;
        }

        if let Err(e) = fs::rename(&tmp_path, &index_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(Error::Processing(format!(
                "Failed to atomically replace metadata index: {e}"
            )));
        }

        Ok(())
    }

    /// Update metadata index with new model and persist the index to disk.
    async fn update_metadata_index(&self, metadata: &StoredModelMetadata) -> Result<()> {
        let mut index = self.metadata_index.write().await;
        index.insert(metadata.clone());
        self.persist_metadata_index(&index)
    }

    /// Compress model data with Zstd (via the pure-Rust `oxiarc-zstd` codec).
    ///
    /// Data that does not actually shrink (e.g. tiny payloads or already
    /// high-entropy data) is stored uncompressed rather than paying
    /// decompression overhead for no benefit; in that case `None` is returned
    /// as the compression info.
    async fn compress_model_data(&self, data: &[u8]) -> Result<(Vec<u8>, Option<CompressionInfo>)> {
        if data.is_empty() {
            return Ok((data.to_vec(), None));
        }

        let level = (self.config.compression_level.clamp(1, 22)) as i32;
        let start = Instant::now();
        let compressed = oxiarc_zstd::compress_with_level(data, level)
            .map_err(|e| Error::Processing(format!("Zstd compression failed: {e}")))?;
        let compression_time = start.elapsed();

        let original_size = data.len() as u64;
        let compressed_size = compressed.len() as u64;

        if compressed_size >= original_size {
            return Ok((data.to_vec(), None));
        }

        let info = CompressionInfo {
            algorithm: CompressionAlgorithm::Zstd,
            original_size,
            compressed_size,
            compression_ratio: compressed_size as f32 / original_size as f32,
            compression_time,
        };

        Ok((compressed, Some(info)))
    }

    /// Decompress model data according to the algorithm recorded at store time.
    async fn decompress_model_data(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Zstd => oxiarc_zstd::decompress(data)
                .map_err(|e| Error::Processing(format!("Zstd decompression failed: {e}"))),
            CompressionAlgorithm::Gzip => oxiarc_deflate::gzip::gzip_decompress(data)
                .map_err(|e| Error::Processing(format!("Gzip decompression failed: {e}"))),
            CompressionAlgorithm::Lz4 => Err(Error::Config(
                "Lz4 decompression is not implemented by this storage backend (only Gzip and \
                 Zstd are supported); this model was stored with an unsupported codec"
                    .to_string(),
            )),
        }
    }

    /// Find an already-stored model whose derived voice characteristics are
    /// close enough to `profile` (per [`StorageConfig::deduplication_threshold`])
    /// to be considered a duplicate, returning its model ID if found.
    async fn find_similar_model(&self, profile: &SpeakerProfile) -> Result<Option<String>> {
        let candidate = self.extract_speaker_info(profile).characteristics;
        let index = self.metadata_index.read().await;

        let mut best: Option<(String, f32)> = None;
        for existing in index.speaker_metadata.values() {
            let similarity =
                characteristics_similarity(&candidate, &existing.speaker_info.characteristics);
            if similarity >= self.config.deduplication_threshold
                && best.as_ref().map(|(_, s)| similarity > *s).unwrap_or(true)
            {
                best = Some((existing.model_id.clone(), similarity));
            }
        }

        Ok(best.map(|(id, _)| id))
    }

    /// Decide whether a model is worth caching at all: it must fit within the
    /// configured cache budget (eviction of other entries happens in
    /// [`Self::cache_model`]).
    async fn should_cache_model(&self, metadata: &StoredModelMetadata) -> bool {
        let cache = self.model_cache.read().await;
        cache.max_size > 0 && metadata.storage_info.file_size <= cache.max_size
    }

    /// Insert a model into the in-memory LRU cache, evicting least-recently-used
    /// entries as needed to stay within [`ModelCache::max_size`].
    async fn cache_model(
        &self,
        model_id: &str,
        data: &[u8],
        metadata: &StoredModelMetadata,
    ) -> Result<()> {
        let size = data.len() as u64;
        let mut cache = self.model_cache.write().await;

        if cache.max_size == 0 || size > cache.max_size {
            // Cannot possibly fit; this is not an error, caching is best-effort.
            return Ok(());
        }

        while cache.current_size + size > cache.max_size {
            let Some(evict_id) = cache.access_queue.pop_front() else {
                break;
            };
            if let Some(evicted) = cache.cache.remove(&evict_id) {
                cache.current_size = cache.current_size.saturating_sub(evicted.size);
                cache.stats.evictions += 1;
                trace!(
                    model_id = %evict_id,
                    age_secs = ?evicted.cached_at.elapsed().map(|d| d.as_secs()),
                    "Evicted model from LRU cache to make room"
                );
            }
        }

        // Replace any existing entry for this id first to avoid double-counting size.
        if let Some(old) = cache.cache.remove(model_id) {
            cache.current_size = cache.current_size.saturating_sub(old.size);
            cache.access_queue.retain(|id| id != model_id);
        }

        cache.cache.insert(
            model_id.to_string(),
            CachedModel {
                data: data.to_vec(),
                metadata: metadata.clone(),
                cached_at: SystemTime::now(),
                access_count: 0,
                size,
            },
        );
        cache.access_queue.push_back(model_id.to_string());
        cache.current_size += size;

        Ok(())
    }

    /// Look up a model in the LRU cache, updating hit/miss statistics and
    /// promoting the entry to most-recently-used on a hit.
    async fn get_from_cache(
        &self,
        model_id: &str,
    ) -> Result<Option<(Vec<u8>, StoredModelMetadata)>> {
        let mut cache = self.model_cache.write().await;

        let result = if let Some(entry) = cache.cache.get_mut(model_id) {
            entry.access_count += 1;
            let data = entry.data.clone();
            let metadata = entry.metadata.clone();
            Some((data, metadata))
        } else {
            None
        };

        if result.is_some() {
            cache.access_queue.retain(|id| id != model_id);
            cache.access_queue.push_back(model_id.to_string());
            cache.stats.hits += 1;
        } else {
            cache.stats.misses += 1;
        }
        let total = cache.stats.hits + cache.stats.misses;
        cache.stats.hit_ratio = if total > 0 {
            cache.stats.hits as f32 / total as f32
        } else {
            0.0
        };

        Ok(result)
    }

    /// Evict a model from the LRU cache (used on delete, or when superseded).
    async fn remove_from_cache(&self, model_id: &str) {
        let mut cache = self.model_cache.write().await;
        if let Some(removed) = cache.cache.remove(model_id) {
            cache.current_size = cache.current_size.saturating_sub(removed.size);
        }
        cache.access_queue.retain(|id| id != model_id);
    }

    async fn get_model_metadata(&self, model_id: &str) -> Result<Option<StoredModelMetadata>> {
        let index = self.metadata_index.read().await;
        Ok(index.speaker_metadata.get(model_id).cloned())
    }

    /// Remove a model from the metadata index and persist the change to disk.
    async fn remove_from_metadata_index(&self, model_id: &str) -> Result<()> {
        let mut index = self.metadata_index.write().await;
        index.remove(model_id);
        self.persist_metadata_index(&index)
    }

    /// Record a real access against a model's metadata: increments the access
    /// counter, appends to the 30-day recent-access window (trimming entries
    /// older than that), recomputes `access_frequency` from that window, and
    /// persists the change.
    async fn update_access_stats(&self, model_id: &str) -> Result<()> {
        let now = SystemTime::now();
        let mut index = self.metadata_index.write().await;

        let Some(metadata) = index.speaker_metadata.get_mut(model_id) else {
            return Ok(());
        };

        metadata.access_stats.access_count += 1;
        metadata.access_stats.last_access = now;
        metadata.access_stats.recent_accesses.push_back(now);

        while let Some(&oldest) = metadata.access_stats.recent_accesses.front() {
            if now
                .duration_since(oldest)
                .map(|age| age > RECENT_ACCESS_RETENTION)
                .unwrap_or(false)
            {
                metadata.access_stats.recent_accesses.pop_front();
            } else {
                break;
            }
        }

        let tracked_days = metadata
            .access_stats
            .recent_accesses
            .front()
            .and_then(|&first| now.duration_since(first).ok())
            .map(|span| (span.as_secs_f32() / 86_400.0).max(1.0))
            .unwrap_or(1.0);
        metadata.access_stats.access_frequency =
            metadata.access_stats.recent_accesses.len() as f32 / tracked_days;

        metadata.storage_info.last_accessed = now;

        self.persist_metadata_index(&index)
    }

    /// Recompute storage statistics from the real metadata index and cache
    /// state (model counts, sizes, tier distribution, compression ratios,
    /// cache hit/miss ratio, and a running average response time derived from
    /// actually-observed operation durations).
    async fn update_storage_statistics(&self, processing_time: Duration) {
        self.operation_count.fetch_add(1, Ordering::Relaxed);
        self.operation_total_time_ms
            .fetch_add(processing_time.as_millis() as u64, Ordering::Relaxed);

        let (
            total_models,
            total_size,
            tier_distribution,
            compressed_models,
            total_original,
            total_compressed,
        ) = {
            let index = self.metadata_index.read().await;
            let total_models = index.speaker_metadata.len() as u64;
            let total_size: u64 = index
                .speaker_metadata
                .values()
                .map(|m| m.storage_info.file_size)
                .sum();

            let mut tier_distribution: HashMap<StorageTier, u64> = HashMap::new();
            for model in index.speaker_metadata.values() {
                *tier_distribution
                    .entry(model.storage_info.storage_tier)
                    .or_insert(0) += 1;
            }

            let (compressed_models, total_original, total_compressed) = index
                .speaker_metadata
                .values()
                .filter_map(|m| m.compression_info.as_ref())
                .fold((0u64, 0u64, 0u64), |(count, orig, comp), info| {
                    (
                        count + 1,
                        orig + info.original_size,
                        comp + info.compressed_size,
                    )
                });

            (
                total_models,
                total_size,
                tier_distribution,
                compressed_models,
                total_original,
                total_compressed,
            )
        };

        let avg_model_size = total_size.checked_div(total_models).unwrap_or(0);
        let avg_compression_ratio = if total_original > 0 {
            total_compressed as f32 / total_original as f32
        } else {
            1.0
        };

        let cache_stats = self.model_cache.read().await.stats.clone();

        let op_count = self.operation_count.load(Ordering::Relaxed);
        let op_total_ms = self.operation_total_time_ms.load(Ordering::Relaxed);
        let avg_response_time_ms = if op_count > 0 {
            op_total_ms as f32 / op_count as f32
        } else {
            0.0
        };

        let max_budget_bytes = self
            .config
            .max_model_size
            .saturating_mul(total_models.max(1));
        let storage_utilization = if max_budget_bytes > 0 {
            (total_size as f32 / max_budget_bytes as f32).min(1.0)
        } else {
            0.0
        };

        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        if storage_utilization > 0.9 {
            issues.push(
                "Storage utilization is above 90% of the configured per-model budget".to_string(),
            );
            recommendations
                .push("Enable cleanup/deduplication or increase max_model_size".to_string());
        }
        if cache_stats.hits + cache_stats.misses >= 10 && cache_stats.hit_ratio < 0.3 {
            issues.push("Model cache hit ratio is low".to_string());
            recommendations.push("Consider increasing max_cache_size".to_string());
        }
        let health_score = if issues.is_empty() {
            1.0
        } else {
            (1.0 - 0.2 * issues.len() as f32).max(0.0)
        };

        let mut statistics = self.statistics.write().await;
        statistics.total_models = total_models;
        statistics.total_size = total_size;
        statistics.avg_model_size = avg_model_size;
        statistics.tier_distribution = tier_distribution;
        statistics.compression_stats = CompressionStatistics {
            compressed_models,
            total_original_size: total_original,
            total_compressed_size: total_compressed,
            avg_compression_ratio,
            space_saved: total_original.saturating_sub(total_compressed),
        };
        statistics.cache_stats = cache_stats;
        statistics.health_indicators = HealthIndicators {
            health_score,
            storage_utilization,
            cache_efficiency: statistics.cache_stats.hit_ratio,
            // No failure path currently calls into this method, so an honest
            // 0.0 reflects "no errors observed" rather than a fabricated
            // placeholder; this should be wired to a real failure counter if
            // one is added to the write/read paths.
            error_rate: 0.0,
            avg_response_time_ms,
            issues,
            recommendations,
        };
    }

    /// Automatic background maintenance scheduling is not implemented: no
    /// task is spawned here. Callers that enable `enable_auto_cleanup` /
    /// `enable_deduplication` must invoke [`Self::perform_maintenance`]
    /// explicitly (e.g. from their own timer, a CLI subcommand, or a service
    /// entry point) — that method performs real cleanup, deduplication, tier
    /// updates, and index persistence.
    async fn start_maintenance_tasks(&self) -> Result<()> {
        debug!(
            "Automatic maintenance scheduling is not implemented; call perform_maintenance() \
             explicitly (e.g. on a timer) to run cleanup/deduplication/tier updates"
        );
        Ok(())
    }

    /// Apply a [`ModelFilter`] to a list of stored-model metadata.
    fn apply_filter(
        &self,
        models: Vec<StoredModelMetadata>,
        filter: &ModelFilter,
    ) -> Vec<StoredModelMetadata> {
        models
            .into_iter()
            .filter(|model| {
                if let Some(ref speaker_id) = filter.speaker_id {
                    if &model.speaker_info.speaker_id != speaker_id {
                        return false;
                    }
                }
                if let Some(ref tags) = filter.tags {
                    if !tags.iter().any(|tag| model.tags.contains(tag)) {
                        return false;
                    }
                }
                if let Some(after) = filter.created_after {
                    if model.storage_info.created_at < after {
                        return false;
                    }
                }
                if let Some(before) = filter.created_before {
                    if model.storage_info.created_at > before {
                        return false;
                    }
                }
                if let Some(tier) = filter.storage_tier {
                    if model.storage_info.storage_tier != tier {
                        return false;
                    }
                }
                if let Some(min_score) = filter.min_quality_score {
                    let score = model
                        .quality_metrics
                        .as_ref()
                        .map(|q| q.overall_score)
                        .unwrap_or(0.0);
                    if score < min_score {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Remove models older than [`StorageConfig::cleanup_age_threshold_days`],
    /// deleting both their backing file and their metadata index entry.
    /// Returns `(models_removed, bytes_recovered)`.
    async fn cleanup_old_models(&self) -> Result<(u64, u64)> {
        let threshold = Duration::from_secs(
            self.config
                .cleanup_age_threshold_days
                .saturating_mul(24 * 3600),
        );
        let now = SystemTime::now();

        let stale_ids: Vec<String> = {
            let index = self.metadata_index.read().await;
            index
                .speaker_metadata
                .values()
                .filter(|m| {
                    now.duration_since(m.storage_info.created_at)
                        .map(|age| age > threshold)
                        .unwrap_or(false)
                })
                .map(|m| m.model_id.clone())
                .collect()
        };

        let mut space_recovered = 0u64;
        for model_id in &stale_ids {
            if let Some(metadata) = self.get_model_metadata(model_id).await? {
                let file_path = self.storage_root.join(&metadata.storage_info.file_path);
                if file_path.exists() {
                    fs::remove_file(&file_path).map_err(|e| {
                        Error::Processing(format!("Failed to remove stale model file: {e}"))
                    })?;
                }
                space_recovered += metadata.storage_info.file_size;
            }
            self.remove_from_cache(model_id).await;
            self.remove_from_metadata_index(model_id).await?;
        }

        Ok((stale_ids.len() as u64, space_recovered))
    }

    /// Remove near-duplicate models (per [`StorageConfig::deduplication_threshold`]),
    /// keeping the oldest model in each duplicate cluster. Returns
    /// `(models_removed, bytes_recovered)`.
    async fn deduplicate_models(&self) -> Result<(u64, u64)> {
        let threshold = self.config.deduplication_threshold;

        let to_remove: Vec<String> = {
            let index = self.metadata_index.read().await;
            let mut entries: Vec<&StoredModelMetadata> = index.speaker_metadata.values().collect();
            entries.sort_by_key(|m| m.storage_info.created_at);

            let mut kept: Vec<&StoredModelMetadata> = Vec::new();
            let mut duplicates = Vec::new();
            for candidate in entries {
                let is_duplicate = kept.iter().any(|existing: &&StoredModelMetadata| {
                    characteristics_similarity(
                        &candidate.speaker_info.characteristics,
                        &existing.speaker_info.characteristics,
                    ) >= threshold
                });
                if is_duplicate {
                    duplicates.push(candidate.model_id.clone());
                } else {
                    kept.push(candidate);
                }
            }
            duplicates
        };

        let mut space_recovered = 0u64;
        for model_id in &to_remove {
            if let Some(metadata) = self.get_model_metadata(model_id).await? {
                let file_path = self.storage_root.join(&metadata.storage_info.file_path);
                if file_path.exists() {
                    fs::remove_file(&file_path).map_err(|e| {
                        Error::Processing(format!("Failed to remove duplicate model file: {e}"))
                    })?;
                }
                space_recovered += metadata.storage_info.file_size;
            }
            self.remove_from_cache(model_id).await;
            self.remove_from_metadata_index(model_id).await?;
        }

        Ok((to_remove.len() as u64, space_recovered))
    }

    /// Re-tier every model based on real access recency
    /// (`Hot` <= 7 days, `Warm` <= 30 days, else `Cold`). Returns the number
    /// of models whose tier actually changed.
    async fn update_storage_tiers(&self) -> Result<u64> {
        let now = SystemTime::now();
        let mut updated = 0u64;

        let mut index = self.metadata_index.write().await;
        for metadata in index.speaker_metadata.values_mut() {
            let age = now
                .duration_since(metadata.access_stats.last_access)
                .unwrap_or(Duration::ZERO);
            let new_tier = if age <= HOT_TIER_MAX_AGE {
                StorageTier::Hot
            } else if age <= WARM_TIER_MAX_AGE {
                StorageTier::Warm
            } else {
                StorageTier::Cold
            };
            if metadata.storage_info.storage_tier != new_tier {
                metadata.storage_info.storage_tier = new_tier;
                updated += 1;
            }
        }

        if updated > 0 {
            self.persist_metadata_index(&index)?;
        }

        Ok(updated)
    }

    /// Self-heal and compact the metadata index: drop entries whose backing
    /// file no longer exists on disk, trim recent-access history beyond the
    /// retention window, and persist the result.
    async fn optimize_metadata_index(&self) -> Result<()> {
        let mut index = self.metadata_index.write().await;

        let missing: Vec<String> = index
            .speaker_metadata
            .values()
            .filter(|m| !self.storage_root.join(&m.storage_info.file_path).exists())
            .map(|m| m.model_id.clone())
            .collect();
        for model_id in &missing {
            index.remove(model_id);
        }

        let now = SystemTime::now();
        for metadata in index.speaker_metadata.values_mut() {
            while let Some(&oldest) = metadata.access_stats.recent_accesses.front() {
                if now
                    .duration_since(oldest)
                    .map(|age| age > RECENT_ACCESS_RETENTION)
                    .unwrap_or(false)
                {
                    metadata.access_stats.recent_accesses.pop_front();
                } else {
                    break;
                }
            }
        }

        self.persist_metadata_index(&index)?;

        if !missing.is_empty() {
            info!(
                "Metadata index optimization removed {} stale entrie(s) with missing backing files",
                missing.len()
            );
        }

        Ok(())
    }
}

/// On-disk representation of the metadata index (see [`MetadataIndex`]).
///
/// Only the primary speaker-metadata map is persisted; the derived indices
/// (`category_index`, `creation_time_index`, `size_index`, `access_frequency`)
/// are rebuilt in memory from this list whenever the index is loaded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PersistedMetadataIndex {
    models: Vec<StoredModelMetadata>,
}

/// Combine several derived voice-characteristic distances into a single
/// similarity score in `[0, 1]`, used for duplicate detection. This is an
/// explicit heuristic over measured characteristics (F0, voice-quality
/// indicators, spectral centroid, energy) — not a learned embedding
/// similarity — so it is deliberately conservative about what it calls a
/// "duplicate".
fn characteristics_similarity(
    a: &VoiceCharacteristicsSummary,
    b: &VoiceCharacteristicsSummary,
) -> f32 {
    let f0_sim = 1.0 - ((a.average_f0 - b.average_f0).abs() / 400.0).min(1.0);
    let quality_sim = elementwise_similarity(&a.quality_indicators, &b.quality_indicators);
    let centroid_sim = 1.0 - ((a.spectral_centroid - b.spectral_centroid).abs() / 4000.0).min(1.0);
    let energy_scale = a
        .energy_stats
        .mean
        .abs()
        .max(b.energy_stats.mean.abs())
        .max(1e-6);
    let energy_sim =
        1.0 - ((a.energy_stats.mean - b.energy_stats.mean).abs() / energy_scale).min(1.0);

    (f0_sim * 0.4 + quality_sim * 0.3 + centroid_sim * 0.2 + energy_sim * 0.1).clamp(0.0, 1.0)
}

/// Elementwise similarity between two equal-length feature vectors, in
/// `[0, 1]`: `1.0` minus the mean absolute per-element difference. Unlike
/// cosine similarity, this is sensitive to differences in absolute
/// magnitude, not just direction - two parallel vectors at very different
/// scales (e.g. `[0.1, 0.1, 0.1, 0.1]` vs `[0.9, 0.9, 0.9, 0.9]`, both
/// quality-indicator vectors in `[0, 1]`) correctly score as dissimilar
/// rather than as a perfect cosine match. Returns `0.0` for empty or
/// mismatched-length inputs.
fn elementwise_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mean_abs_diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    // Quality indicators are normalized to [0, 1], so a difference of 1.0
    // (the maximum possible per element) maps to zero similarity.
    (1.0 - mean_abs_diff).clamp(0.0, 1.0)
}

/// Model filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFilter {
    /// Filter by speaker ID
    pub speaker_id: Option<String>,
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    /// Filter by creation date range
    pub created_after: Option<SystemTime>,
    pub created_before: Option<SystemTime>,
    /// Filter by storage tier
    pub storage_tier: Option<StorageTier>,
    /// Filter by minimum quality score
    pub min_quality_score: Option<f32>,
}

/// Maintenance operation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceReport {
    /// Maintenance start time
    pub start_time: SystemTime,
    /// Operations performed
    pub operations_performed: Vec<String>,
    /// Number of models processed
    pub models_processed: u64,
    /// Space recovered in bytes
    pub space_recovered: u64,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Total maintenance duration
    pub duration: Duration,
}

impl ModelCache {
    fn new(max_size: u64) -> Self {
        Self {
            cache: HashMap::new(),
            access_queue: VecDeque::new(),
            current_size: 0,
            max_size,
            stats: CacheStatistics::default(),
        }
    }
}

// Default implementations
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 100, // 100MB
            enable_compression: true,
            compression_level: 6,
            max_model_size: 50 * 1024 * 1024, // 50MB
            enable_auto_cleanup: true,
            cleanup_age_threshold_days: 30,
            enable_encryption: false,
            maintenance_interval: Duration::from_secs(3600), // 1 hour
            enable_deduplication: true,
            deduplication_threshold: 0.95,
            enable_tiered_storage: true,
            backup_retention_days: 7,
        }
    }
}

impl Default for StorageStatistics {
    fn default() -> Self {
        Self {
            total_models: 0,
            total_size: 0,
            avg_model_size: 0,
            tier_distribution: HashMap::new(),
            compression_stats: CompressionStatistics::default(),
            cache_stats: CacheStatistics::default(),
            maintenance_stats: MaintenanceStatistics::default(),
            health_indicators: HealthIndicators {
                health_score: 1.0,
                storage_utilization: 0.0,
                cache_efficiency: 0.0,
                error_rate: 0.0,
                avg_response_time_ms: 0.0,
                issues: Vec::new(),
                recommendations: Vec::new(),
            },
        }
    }
}

// Tests live in `storage/tests.rs` (kept out of this file to stay under the
// workspace's 2000-line-per-file guideline).
#[cfg(test)]
#[path = "storage/tests.rs"]
mod tests;
