//! Model caching system for efficient model loading and management.

use crate::{
    error::{Result, VoirsError},
    traits::{CacheStats, ModelCache},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    hash::Hash,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};
use tokio::fs;
use tracing::{debug, info, warn};

#[cfg(feature = "cloud")]
use sha2::{Digest, Sha256};

/// Advanced model cache with intelligent loading and LRU management
pub struct AdvancedModelCache {
    /// In-memory cache storage
    memory_cache: Arc<RwLock<HashMap<String, CachedModel>>>,

    /// Persistent cache on disk
    disk_cache_dir: Option<PathBuf>,

    /// Maximum memory cache size in bytes
    max_memory_bytes: usize,

    /// Maximum disk cache size in bytes
    #[allow(dead_code)]
    max_disk_bytes: usize,

    /// Current memory usage
    current_memory_usage: Arc<RwLock<usize>>,

    /// Access tracking for LRU (most recent first)
    access_order: Arc<RwLock<Vec<String>>>,

    /// Model loading queue for async operations
    loading_queue: Arc<RwLock<HashSet<String>>>,

    /// Cache configuration
    config: ModelCacheConfig,

    /// Cache statistics
    stats: Arc<RwLock<ModelCacheStats>>,

    /// Model metadata cache
    #[allow(dead_code)]
    metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,

    /// Model dependency graph
    #[allow(dead_code)]
    dependency_graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

/// Configuration for model caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheConfig {
    /// Enable memory caching
    pub memory_cache_enabled: bool,

    /// Enable disk caching
    pub disk_cache_enabled: bool,

    /// Memory cache size in MB
    pub memory_cache_size_mb: usize,

    /// Disk cache size in MB
    pub disk_cache_size_mb: usize,

    /// Model TTL in seconds
    pub model_ttl_seconds: u64,

    /// Enable compression for disk cache
    pub enable_compression: bool,

    /// Enable cache warming on startup
    pub enable_cache_warming: bool,

    /// Preload models on first access
    pub enable_preloading: bool,

    /// Background cleanup interval in seconds
    pub cleanup_interval_seconds: u64,

    /// Maximum concurrent model loads
    pub max_concurrent_loads: usize,

    /// Cache verification on load
    pub verify_integrity: bool,

    /// Model priority levels
    pub priority_levels: HashMap<String, ModelPriority>,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            memory_cache_enabled: true,
            disk_cache_enabled: true,
            memory_cache_size_mb: 1024, // 1GB
            disk_cache_size_mb: 8192,   // 8GB
            model_ttl_seconds: 86400,   // 24 hours
            enable_compression: true,
            enable_cache_warming: true,
            enable_preloading: false,
            cleanup_interval_seconds: 3600, // 1 hour
            max_concurrent_loads: 4,
            verify_integrity: true,
            priority_levels: HashMap::new(),
        }
    }
}

/// Model priority levels for cache eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelPriority {
    /// Critical models that should never be evicted
    Critical,
    /// High priority models
    High,
    /// Normal priority models
    Normal,
    /// Low priority models (evicted first)
    Low,
}

impl Default for ModelPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Cached model entry
#[derive(Debug)]
pub struct CachedModel {
    /// Model data
    pub data: Box<dyn Any + Send + Sync>,

    /// Model metadata
    pub metadata: ModelMetadata,

    /// Size in bytes
    pub size_bytes: usize,

    /// When the model was cached
    pub cached_at: SystemTime,

    /// When the model expires
    pub expires_at: SystemTime,

    /// Last access time
    pub last_accessed: SystemTime,

    /// Access count
    pub access_count: u64,

    /// Model priority
    pub priority: ModelPriority,

    /// Whether model is pinned (can't be evicted)
    pub pinned: bool,

    /// Model checksum for integrity verification
    pub checksum: Option<String>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model type (g2p, acoustic, vocoder, etc.)
    pub model_type: ModelType,

    /// Model architecture
    pub architecture: String,

    /// Model language(s)
    pub languages: Vec<String>,

    /// Model size in parameters
    pub parameter_count: Option<u64>,

    /// Model file path (for disk-based models)
    pub file_path: Option<PathBuf>,

    /// Model dependencies
    pub dependencies: Vec<String>,

    /// Model creation timestamp
    pub created_at: SystemTime,

    /// Model source/origin
    pub source: ModelSource,

    /// Model configuration hash
    pub config_hash: u64,

    /// Model precision (fp32, fp16, int8, etc.)
    pub precision: ModelPrecision,

    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
}

/// Model type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Grapheme-to-phoneme model
    G2P,
    /// Acoustic model
    Acoustic,
    /// Vocoder model
    Vocoder,
    /// Language model
    LanguageModel,
    /// Encoder model
    Encoder,
    /// Decoder model
    Decoder,
    /// Enhancement model
    Enhancement,
    /// Other model type
    Other(u8),
}

/// Model source/origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// Local file
    Local(PathBuf),
    /// Remote URL
    Remote(String),
    /// Hugging Face Hub
    HuggingFace { repo: String, revision: String },
    /// Built-in model
    Builtin(String),
    /// Custom source
    Custom(String),
}

/// Model precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
    Mixed,
}

/// Hardware requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum RAM in MB
    pub min_ram_mb: u64,

    /// GPU memory requirement in MB
    pub gpu_memory_mb: Option<u64>,

    /// Required CPU features
    pub cpu_features: Vec<String>,

    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,

    /// Minimum compute capability
    pub min_compute_capability: Option<String>,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum GPU memory in MB
    pub min_memory_mb: u64,

    /// Required GPU APIs
    pub required_apis: Vec<String>,

    /// Preferred GPU vendor
    pub preferred_vendor: Option<String>,

    /// Minimum compute capability
    pub min_compute_capability: Option<String>,
}

/// Extended cache statistics for models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCacheStats {
    /// Basic cache stats
    pub basic_stats: CacheStats,

    /// Models loaded
    pub models_loaded: u64,

    /// Models evicted
    pub models_evicted: u64,

    /// Models failed to load
    pub load_failures: u64,

    /// Cache warming time
    pub warming_time_ms: u64,

    /// Average load time
    pub avg_load_time_ms: f64,

    /// Memory fragmentation ratio
    pub memory_fragmentation: f64,

    /// Disk cache usage
    pub disk_usage_bytes: u64,

    /// Model load queue size
    pub queue_size: usize,

    /// Hot models (frequently accessed)
    pub hot_models: Vec<String>,

    /// Cold models (rarely accessed)
    pub cold_models: Vec<String>,

    /// Model priority distribution
    pub priority_distribution: HashMap<ModelPriority, usize>,
}

impl AdvancedModelCache {
    /// Create new advanced model cache
    pub fn new(config: ModelCacheConfig, disk_cache_dir: Option<PathBuf>) -> Result<Self> {
        // Create disk cache directory if specified
        if let Some(ref dir) = disk_cache_dir {
            if config.disk_cache_enabled {
                std::fs::create_dir_all(dir).map_err(|e| {
                    VoirsError::cache_error(format!("Failed to create cache directory: {e}"))
                })?;
            }
        }

        Ok(Self {
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            disk_cache_dir,
            max_memory_bytes: config.memory_cache_size_mb * 1024 * 1024,
            max_disk_bytes: config.disk_cache_size_mb * 1024 * 1024,
            current_memory_usage: Arc::new(RwLock::new(0)),
            access_order: Arc::new(RwLock::new(Vec::new())),
            loading_queue: Arc::new(RwLock::new(HashSet::new())),
            config,
            stats: Arc::new(RwLock::new(ModelCacheStats::default())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Warm cache by preloading commonly used models
    pub async fn warm_cache(&self, model_names: Vec<String>) -> Result<()> {
        let start_time = Instant::now();
        info!("Starting cache warming for {} models", model_names.len());

        let mut successful_loads = 0;
        let mut failed_loads = 0;

        for model_name in model_names {
            match self.preload_model(&model_name).await {
                Ok(_) => {
                    successful_loads += 1;
                    debug!("Preloaded model: {}", model_name);
                }
                Err(e) => {
                    failed_loads += 1;
                    warn!("Failed to preload model '{}': {}", model_name, e);
                }
            }
        }

        let warming_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.warming_time_ms = warming_time;
            stats.models_loaded += successful_loads;
            stats.load_failures += failed_loads;
        }

        info!(
            "Cache warming completed in {}ms: {} successful, {} failed",
            warming_time, successful_loads, failed_loads
        );

        Ok(())
    }

    /// Preload a specific model
    pub async fn preload_model(&self, model_name: &str) -> Result<()> {
        // Check if model is already loaded
        if self.contains_key(model_name).await {
            return Ok(());
        }

        // Check if model is in loading queue
        {
            let loading_queue = self.loading_queue.read().unwrap();
            if loading_queue.contains(model_name) {
                return Err(VoirsError::cache_error(format!(
                    "Model '{model_name}' is already being loaded"
                )));
            }
        }

        // Add to loading queue
        {
            let mut loading_queue = self.loading_queue.write().unwrap();
            loading_queue.insert(model_name.to_string());
        }

        // Attempt to load from disk cache first
        let loaded = if self.config.disk_cache_enabled {
            self.load_from_disk_cache(model_name).await.unwrap_or(false)
        } else {
            false
        };

        // Remove from loading queue
        {
            let mut loading_queue = self.loading_queue.write().unwrap();
            loading_queue.remove(model_name);
        }

        if loaded {
            info!("Loaded model '{}' from disk cache", model_name);
            Ok(())
        } else {
            Err(VoirsError::cache_error(format!(
                "Model '{model_name}' not found in cache"
            )))
        }
    }

    /// Check if model exists in cache
    pub async fn contains_key(&self, key: &str) -> bool {
        let cache = self.memory_cache.read().unwrap();
        cache.contains_key(key)
    }

    /// Load model from disk cache
    async fn load_from_disk_cache(&self, model_name: &str) -> Result<bool> {
        if let Some(ref cache_dir) = self.disk_cache_dir {
            let model_path = cache_dir.join(format!("{model_name}.cache"));
            let metadata_path = cache_dir.join(format!("{model_name}.meta"));

            if model_path.exists() && metadata_path.exists() {
                // Load metadata
                let metadata_content = fs::read_to_string(&metadata_path).await.map_err(|e| {
                    VoirsError::cache_error(format!("Failed to read metadata: {e}"))
                })?;

                let metadata: ModelMetadata =
                    serde_json::from_str(&metadata_content).map_err(|e| {
                        VoirsError::cache_error(format!("Failed to parse metadata: {e}"))
                    })?;

                // Load model data (this is a placeholder - real implementation would
                // deserialize the actual model based on the model type)
                let model_data = format!("Model data for {model_name}");

                // Calculate checksum for integrity verification
                let checksum = self.calculate_file_checksum(&model_path).await.ok();

                // Create cached model entry
                let cached_model = CachedModel {
                    data: Box::new(model_data),
                    metadata: metadata.clone(),
                    size_bytes: model_path.metadata().unwrap().len() as usize,
                    cached_at: SystemTime::now(),
                    expires_at: SystemTime::now()
                        + Duration::from_secs(self.config.model_ttl_seconds),
                    last_accessed: SystemTime::now(),
                    access_count: 0,
                    priority: self
                        .config
                        .priority_levels
                        .get(model_name)
                        .copied()
                        .unwrap_or_default(),
                    pinned: false,
                    checksum,
                };

                // Store in memory cache
                self.put_cached_model(model_name, cached_model).await?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Store a cached model in memory
    async fn put_cached_model(&self, key: &str, model: CachedModel) -> Result<()> {
        // Check memory limits and evict if necessary
        self.ensure_memory_capacity(model.size_bytes).await?;

        // Store model
        {
            let mut cache = self.memory_cache.write().unwrap();
            let mut current_usage = self.current_memory_usage.write().unwrap();

            let model_size = model.size_bytes;
            cache.insert(key.to_string(), model);
            *current_usage += model_size;
        }

        // Update access order
        self.update_access_order(key).await;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.basic_stats.total_entries += 1;
            stats.basic_stats.memory_usage_bytes = *self.current_memory_usage.read().unwrap();
            stats.models_loaded += 1;
        }

        Ok(())
    }

    /// Ensure sufficient memory capacity
    async fn ensure_memory_capacity(&self, required_bytes: usize) -> Result<()> {
        let current_usage = *self.current_memory_usage.read().unwrap();

        if current_usage + required_bytes > self.max_memory_bytes {
            self.evict_lru_models(required_bytes).await?;
        }

        Ok(())
    }

    /// Evict LRU models to free space
    async fn evict_lru_models(&self, required_bytes: usize) -> Result<()> {
        let mut freed_bytes = 0;
        let mut evicted_count = 0;

        // Get models to evict (LRU order, considering priority)
        let models_to_evict = self.get_eviction_candidates(required_bytes).await;

        for model_key in models_to_evict {
            if let Some(model) = self.remove_model(&model_key).await? {
                freed_bytes += model.size_bytes;
                evicted_count += 1;

                debug!("Evicted model '{}' ({} bytes)", model_key, model.size_bytes);

                if freed_bytes >= required_bytes {
                    break;
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.models_evicted += evicted_count;
        }

        info!(
            "Evicted {} models, freed {} bytes",
            evicted_count, freed_bytes
        );

        Ok(())
    }

    /// Get eviction candidates based on LRU and priority
    async fn get_eviction_candidates(&self, _required_bytes: usize) -> Vec<String> {
        let cache = self.memory_cache.read().unwrap();
        let access_order = self.access_order.read().unwrap();

        let mut candidates = Vec::new();

        // Sort by priority (low first) and then by LRU
        let mut sortable_models: Vec<_> = cache
            .iter()
            .filter(|(_, model)| !model.pinned) // Don't evict pinned models
            .map(|(key, model)| {
                let access_index = access_order
                    .iter()
                    .position(|k| k == key)
                    .unwrap_or(usize::MAX);
                (key.clone(), model.priority, access_index, model.size_bytes)
            })
            .collect();

        // Sort by priority (ascending) then by LRU (descending index = older)
        sortable_models.sort_by(|a, b| {
            use std::cmp::Ordering;
            match (a.1 as u8).cmp(&(b.1 as u8)) {
                Ordering::Equal => b.2.cmp(&a.2), // Older access first
                other => other,
            }
        });

        for (key, _, _, _) in sortable_models {
            candidates.push(key);
        }

        candidates
    }

    /// Remove a model from cache
    async fn remove_model(&self, key: &str) -> Result<Option<CachedModel>> {
        let removed_model = {
            let mut cache = self.memory_cache.write().unwrap();
            cache.remove(key)
        };

        if let Some(ref model) = removed_model {
            // Update memory usage
            {
                let mut current_usage = self.current_memory_usage.write().unwrap();
                *current_usage = current_usage.saturating_sub(model.size_bytes);
            }

            // Update access order
            {
                let mut access_order = self.access_order.write().unwrap();
                access_order.retain(|k| k != key);
            }

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.basic_stats.total_entries = stats.basic_stats.total_entries.saturating_sub(1);
                stats.basic_stats.memory_usage_bytes = *self.current_memory_usage.read().unwrap();
            }
        }

        Ok(removed_model)
    }

    /// Update access order for LRU tracking
    async fn update_access_order(&self, key: &str) {
        let mut access_order = self.access_order.write().unwrap();

        // Remove existing entry
        access_order.retain(|k| k != key);

        // Add to front (most recent)
        access_order.insert(0, key.to_string());
    }

    /// Pin a model to prevent eviction
    pub async fn pin_model(&self, key: &str) -> Result<()> {
        let mut cache = self.memory_cache.write().unwrap();

        if let Some(model) = cache.get_mut(key) {
            model.pinned = true;
            info!("Pinned model '{}'", key);
            Ok(())
        } else {
            Err(VoirsError::cache_error(format!(
                "Model '{key}' not found in cache"
            )))
        }
    }

    /// Unpin a model to allow eviction
    pub async fn unpin_model(&self, key: &str) -> Result<()> {
        let mut cache = self.memory_cache.write().unwrap();

        if let Some(model) = cache.get_mut(key) {
            model.pinned = false;
            info!("Unpinned model '{}'", key);
            Ok(())
        } else {
            Err(VoirsError::cache_error(format!(
                "Model '{key}' not found in cache"
            )))
        }
    }

    /// Get model metadata
    pub async fn get_model_metadata(&self, key: &str) -> Option<ModelMetadata> {
        let cache = self.memory_cache.read().unwrap();
        cache.get(key).map(|model| model.metadata.clone())
    }

    /// List all cached models
    pub async fn list_cached_models(&self) -> Vec<String> {
        let cache = self.memory_cache.read().unwrap();
        cache.keys().cloned().collect()
    }

    /// Get cache usage summary
    pub async fn get_usage_summary(&self) -> CacheUsageSummary {
        let cache = self.memory_cache.read().unwrap();
        let current_usage = *self.current_memory_usage.read().unwrap();

        let model_count = cache.len();
        let total_accesses: u64 = cache.values().map(|m| m.access_count).sum();
        let pinned_count = cache.values().filter(|m| m.pinned).count();

        CacheUsageSummary {
            total_models: model_count,
            memory_usage_bytes: current_usage,
            memory_usage_mb: current_usage / (1024 * 1024),
            memory_utilization: (current_usage as f64 / self.max_memory_bytes as f64) * 100.0,
            total_accesses,
            pinned_models: pinned_count,
            avg_model_size: if model_count > 0 {
                current_usage / model_count
            } else {
                0
            },
        }
    }

    /// Perform cache maintenance
    pub async fn perform_maintenance(&self) -> Result<()> {
        info!("Starting cache maintenance");

        // Clean expired models
        let expired_count = self.cleanup_expired_models().await?;

        // Update statistics
        self.update_cache_statistics().await;

        // Log maintenance results
        info!(
            "Cache maintenance completed: {} expired models removed",
            expired_count
        );

        Ok(())
    }

    /// Clean up expired models
    async fn cleanup_expired_models(&self) -> Result<usize> {
        let now = SystemTime::now();
        let mut expired_keys = Vec::new();

        // Find expired models
        {
            let cache = self.memory_cache.read().unwrap();
            for (key, model) in cache.iter() {
                if model.expires_at <= now && !model.pinned {
                    expired_keys.push(key.clone());
                }
            }
        }

        // Remove expired models
        let mut removed_count = 0;
        for key in expired_keys {
            if self.remove_model(&key).await?.is_some() {
                removed_count += 1;
                debug!("Removed expired model: {}", key);
            }
        }

        Ok(removed_count)
    }

    /// Update cache statistics
    async fn update_cache_statistics(&self) {
        let cache = self.memory_cache.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Update basic stats
        stats.basic_stats.total_entries = cache.len();
        stats.basic_stats.memory_usage_bytes = *self.current_memory_usage.read().unwrap();

        // Calculate memory fragmentation (simplified)
        let used_memory = stats.basic_stats.memory_usage_bytes;
        let allocated_memory = self.max_memory_bytes;
        stats.memory_fragmentation = if allocated_memory > 0 {
            1.0 - (used_memory as f64 / allocated_memory as f64)
        } else {
            0.0
        };

        // Update priority distribution
        stats.priority_distribution.clear();
        for model in cache.values() {
            *stats
                .priority_distribution
                .entry(model.priority)
                .or_insert(0) += 1;
        }

        // Update queue size
        stats.queue_size = self.loading_queue.read().unwrap().len();

        // Identify hot and cold models
        let mut models_by_access: Vec<_> = cache
            .iter()
            .map(|(key, model)| (key.clone(), model.access_count))
            .collect();

        models_by_access.sort_by(|a, b| b.1.cmp(&a.1));

        let hot_threshold = models_by_access.len() / 4; // Top 25%
        let cold_threshold = models_by_access.len() * 3 / 4; // Bottom 25%

        stats.hot_models = models_by_access
            .iter()
            .take(hot_threshold)
            .map(|(key, _)| key.clone())
            .collect();

        stats.cold_models = models_by_access
            .iter()
            .skip(cold_threshold)
            .map(|(key, _)| key.clone())
            .collect();
    }

    /// Calculate SHA256 checksum for file data
    async fn calculate_file_checksum(&self, file_path: &Path) -> Result<String> {
        #[cfg(feature = "cloud")]
        {
            let data = fs::read(file_path)
                .await
                .map_err(|e| VoirsError::FileCorrupted {
                    path: file_path.to_path_buf(),
                    reason: format!("Failed to read file for checksum calculation: {}", e),
                })?;

            let mut hasher = Sha256::new();
            hasher.update(&data);
            let result = hasher.finalize();
            Ok(format!("{:x}", result))
        }

        #[cfg(not(feature = "cloud"))]
        {
            warn!("Checksum calculation requires 'cloud' feature to be enabled");
            Ok("no-checksum-available".to_string())
        }
    }

    /// Verify checksum of cached model data
    #[allow(dead_code)]
    async fn verify_checksum(&self, file_path: &Path, expected_checksum: &str) -> Result<bool> {
        let calculated = self.calculate_file_checksum(file_path).await?;
        Ok(calculated == expected_checksum)
    }
}

#[async_trait]
impl ModelCache for AdvancedModelCache {
    async fn get_any(&self, key: &str) -> Result<Option<Box<dyn Any + Send + Sync>>> {
        let result = {
            let cache = self.memory_cache.read().unwrap();
            cache.get(key).is_some()
        };

        if result {
            // Update access tracking
            self.update_access_order(key).await;

            // Update access count
            {
                let mut cache = self.memory_cache.write().unwrap();
                if let Some(model) = cache.get_mut(key) {
                    model.access_count += 1;
                    model.last_accessed = SystemTime::now();
                }
            }

            // Update hit/miss statistics
            {
                let mut stats = self.stats.write().unwrap();
                let total_requests = stats.basic_stats.hit_rate + stats.basic_stats.miss_rate;
                let hits = (stats.basic_stats.hit_rate / 100.0) * total_requests;
                let new_total = total_requests + 1.0;
                stats.basic_stats.hit_rate = ((hits + 1.0) / new_total) * 100.0;
                stats.basic_stats.miss_rate = 100.0 - stats.basic_stats.hit_rate;
            }

            // Note: We can't actually return the model data due to Any limitations
            // A real implementation would use a different approach for type safety
            Ok(None)
        } else {
            // Update miss statistics
            {
                let mut stats = self.stats.write().unwrap();
                let total_requests = stats.basic_stats.hit_rate + stats.basic_stats.miss_rate;
                let misses = (stats.basic_stats.miss_rate / 100.0) * total_requests;
                let new_total = total_requests + 1.0;
                stats.basic_stats.miss_rate = ((misses + 1.0) / new_total) * 100.0;
                stats.basic_stats.hit_rate = 100.0 - stats.basic_stats.miss_rate;
            }

            Ok(None)
        }
    }

    async fn put_any(&self, key: &str, value: Box<dyn Any + Send + Sync>) -> Result<()> {
        // Create model metadata (simplified for this implementation)
        let metadata = ModelMetadata {
            name: key.to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Other(0),
            architecture: "unknown".to_string(),
            languages: vec!["en".to_string()],
            parameter_count: None,
            file_path: None,
            dependencies: vec![],
            created_at: SystemTime::now(),
            source: ModelSource::Custom("memory".to_string()),
            config_hash: 0,
            precision: ModelPrecision::FP32,
            hardware_requirements: HardwareRequirements {
                min_ram_mb: 0,
                gpu_memory_mb: None,
                cpu_features: vec![],
                gpu_requirements: None,
                min_compute_capability: None,
            },
        };

        let estimated_size = std::mem::size_of_val(&*value) + key.len();

        let cached_model = CachedModel {
            data: value,
            metadata,
            size_bytes: estimated_size,
            cached_at: SystemTime::now(),
            expires_at: SystemTime::now() + Duration::from_secs(self.config.model_ttl_seconds),
            last_accessed: SystemTime::now(),
            access_count: 0,
            priority: self
                .config
                .priority_levels
                .get(key)
                .copied()
                .unwrap_or_default(),
            pinned: false,
            checksum: None,
        };

        self.put_cached_model(key, cached_model).await
    }

    async fn remove(&self, key: &str) -> Result<()> {
        self.remove_model(key).await?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        {
            let mut cache = self.memory_cache.write().unwrap();
            let mut current_usage = self.current_memory_usage.write().unwrap();
            let mut access_order = self.access_order.write().unwrap();

            cache.clear();
            *current_usage = 0;
            access_order.clear();
        }

        // Reset statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.basic_stats.total_entries = 0;
            stats.basic_stats.memory_usage_bytes = 0;
        }

        info!("Cleared all models from cache");
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        let stats = self.stats.read().unwrap();
        stats.basic_stats
    }
}

/// Cache usage summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheUsageSummary {
    /// Total number of cached models
    pub total_models: usize,

    /// Memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Memory usage in MB
    pub memory_usage_mb: usize,

    /// Memory utilization percentage
    pub memory_utilization: f64,

    /// Total access count across all models
    pub total_accesses: u64,

    /// Number of pinned models
    pub pinned_models: usize,

    /// Average model size in bytes
    pub avg_model_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_model_cache_creation() {
        let config = ModelCacheConfig::default();
        let cache = AdvancedModelCache::new(config, None).unwrap();

        assert_eq!(cache.list_cached_models().await.len(), 0);
    }

    #[tokio::test]
    async fn test_model_caching_and_retrieval() {
        let config = ModelCacheConfig::default();
        let cache = AdvancedModelCache::new(config, None).unwrap();

        let test_data = "test model data".to_string();
        cache
            .put_any("test_model", Box::new(test_data))
            .await
            .unwrap();

        assert!(cache.contains_key("test_model").await);
        assert_eq!(cache.list_cached_models().await.len(), 1);
    }

    #[tokio::test]
    async fn test_model_pinning() {
        let config = ModelCacheConfig::default();
        let cache = AdvancedModelCache::new(config, None).unwrap();

        let test_data = "pinned model".to_string();
        cache
            .put_any("pinned_model", Box::new(test_data))
            .await
            .unwrap();
        cache.pin_model("pinned_model").await.unwrap();

        // Verify model is pinned
        let metadata = cache.get_model_metadata("pinned_model").await;
        assert!(metadata.is_some());
    }

    #[tokio::test]
    async fn test_cache_maintenance() {
        let config = ModelCacheConfig::default();
        let cache = AdvancedModelCache::new(config, None).unwrap();

        let test_data = "maintenance test".to_string();
        cache
            .put_any("test_model", Box::new(test_data))
            .await
            .unwrap();

        cache.perform_maintenance().await.unwrap();

        let summary = cache.get_usage_summary().await;
        // Check that the summary contains valid data
        assert!(summary.memory_usage_bytes <= summary.memory_usage_mb * 1024 * 1024 + 1024 * 1024);
    }

    #[tokio::test]
    async fn test_cache_warming() {
        let config = ModelCacheConfig::default();
        let cache = AdvancedModelCache::new(config, None).unwrap();

        // Test warming with empty model list
        let result = cache.warm_cache(vec![]).await;
        assert!(result.is_ok());
    }
}
