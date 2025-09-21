//! Persistent cache implementation with LRU eviction and compression

use crate::cache::{CacheConfig, CachePriority, CleanupResult};
use crate::RecognitionError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Persistent cache entry metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    /// Entry key
    key: String,
    /// File path
    file_path: PathBuf,
    /// Creation time
    created_at: SystemTime,
    /// Last access time
    last_accessed: SystemTime,
    /// TTL duration
    ttl: Duration,
    /// Priority level
    priority: CachePriority,
    /// Compressed size in bytes
    compressed_size: u64,
    /// Original size in bytes
    original_size: u64,
    /// Access count
    access_count: u64,
}

/// Persistent cache with compression and LRU eviction
pub struct PersistentCache {
    config: CacheConfig,
    cache_dir: PathBuf,
    entries: HashMap<String, CacheEntry>,
    total_size_bytes: u64,
    access_order: Vec<String>,
}

impl PersistentCache {
    pub async fn new(config: CacheConfig) -> Result<Self, RecognitionError> {
        let cache_dir = config.cache_dir.clone();
        
        // Ensure cache directory exists
        fs::create_dir_all(&cache_dir).await.map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to create cache directory: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        let mut cache = Self {
            config,
            cache_dir,
            entries: HashMap::new(),
            total_size_bytes: 0,
            access_order: Vec::new(),
        };

        // Load existing cache metadata
        cache.load_metadata().await?;

        Ok(cache)
    }

    /// Store data with priority and compression
    pub async fn store_with_priority<T: serde::Serialize>(
        &mut self,
        key: &str,
        data: &T,
        priority: CachePriority,
    ) -> Result<(), RecognitionError> {
        // Serialize data
        let serialized = serde_json::to_vec(data).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to serialize cache data: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Compress data
        let compressed = super::compression::compress_data(&serialized, self.config.compression_level)?;
        
        // Generate file path
        let file_path = self.cache_dir.join(format!("{}.cache", key));
        
        // Write compressed data to file
        fs::write(&file_path, &compressed).await.map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to write cache file: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Create cache entry
        let entry = CacheEntry {
            key: key.to_string(),
            file_path: file_path.clone(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            ttl: self.config.default_ttl,
            priority,
            compressed_size: compressed.len() as u64,
            original_size: serialized.len() as u64,
            access_count: 0,
        };

        // Update cache metadata
        if let Some(old_entry) = self.entries.get(key) {
            self.total_size_bytes -= old_entry.compressed_size;
        }
        
        self.total_size_bytes += entry.compressed_size;
        self.entries.insert(key.to_string(), entry);
        
        // Update access order for LRU
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key.to_string());

        // Check if we need to evict items
        if self.should_evict() {
            self.evict_lru_items().await?;
        }

        // Save metadata
        self.save_metadata().await?;

        Ok(())
    }

    /// Retrieve cached data
    pub async fn retrieve<T: serde::de::DeserializeOwned>(
        &mut self,
        key: &str,
    ) -> Result<Option<T>, RecognitionError> {
        let entry = match self.entries.get_mut(key) {
            Some(entry) => entry,
            None => return Ok(None),
        };

        // Check if entry has expired
        if entry.created_at.elapsed().unwrap_or(Duration::ZERO) > entry.ttl {
            self.remove_entry(key).await?;
            return Ok(None);
        }

        // Update access statistics
        entry.last_accessed = SystemTime::now();
        entry.access_count += 1;

        // Update LRU order
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key.to_string());

        // Read compressed data from file
        let compressed_data = fs::read(&entry.file_path).await.map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to read cache file: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Decompress data
        let decompressed = super::compression::decompress_data(&compressed_data)?;

        // Deserialize data
        let data: T = serde_json::from_slice(&decompressed).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to deserialize cache data: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(Some(data))
    }

    /// Get compression ratio
    pub async fn get_compression_ratio(&self) -> f64 {
        if self.total_original_size() == 0 {
            return 1.0;
        }
        
        self.total_size_bytes as f64 / self.total_original_size() as f64
    }

    /// Cleanup expired and low-priority entries
    pub async fn cleanup(&mut self) -> Result<CleanupResult, RecognitionError> {
        let mut result = CleanupResult::default();
        let now = SystemTime::now();
        let mut to_remove = Vec::new();

        // Find expired entries
        for (key, entry) in &self.entries {
            if now.duration_since(entry.created_at).unwrap_or(Duration::ZERO) > entry.ttl {
                to_remove.push(key.clone());
            }
        }

        // Remove expired entries
        for key in &to_remove {
            if let Some(entry) = self.entries.get(key) {
                result.bytes_freed += entry.compressed_size;
                result.items_removed += 1;
            }
            self.remove_entry(key).await?;
        }

        // If still over size limit, remove low-priority entries
        while self.total_size_bytes > (self.config.max_cache_size_mb * 1024 * 1024) as u64 {
            if let Some(key) = self.find_lowest_priority_entry() {
                if let Some(entry) = self.entries.get(&key) {
                    result.bytes_freed += entry.compressed_size;
                    result.items_removed += 1;
                }
                self.remove_entry(&key).await?;
            } else {
                break;
            }
        }

        self.save_metadata().await?;
        Ok(result)
    }

    /// Clear all cache entries
    pub async fn clear(&mut self) -> Result<(), RecognitionError> {
        // Remove all cache files
        for entry in self.entries.values() {
            let _ = fs::remove_file(&entry.file_path).await;
        }

        // Clear metadata
        self.entries.clear();
        self.access_order.clear();
        self.total_size_bytes = 0;

        // Save empty metadata
        self.save_metadata().await?;

        Ok(())
    }

    /// Check if cache should evict items
    fn should_evict(&self) -> bool {
        self.total_size_bytes > (self.config.max_cache_size_mb * 1024 * 1024) as u64
            || self.entries.len() > self.config.max_items
    }

    /// Evict LRU items
    async fn evict_lru_items(&mut self) -> Result<(), RecognitionError> {
        while self.should_evict() && !self.access_order.is_empty() {
            // Find least recently used item
            let key = self.access_order.remove(0);
            self.remove_entry(&key).await?;
        }
        Ok(())
    }

    /// Find lowest priority entry for eviction
    fn find_lowest_priority_entry(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| (entry.priority, entry.last_accessed))
            .map(|(key, _)| key.clone())
    }

    /// Remove cache entry
    async fn remove_entry(&mut self, key: &str) -> Result<(), RecognitionError> {
        if let Some(entry) = self.entries.remove(key) {
            self.total_size_bytes -= entry.compressed_size;
            let _ = fs::remove_file(&entry.file_path).await;
        }

        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }

        Ok(())
    }

    /// Load cache metadata from disk
    async fn load_metadata(&mut self) -> Result<(), RecognitionError> {
        let metadata_path = self.cache_dir.join("metadata.json");
        
        if !metadata_path.exists() {
            return Ok(());
        }

        let data = fs::read(&metadata_path).await.map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to read cache metadata: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        let metadata: CacheMetadata = serde_json::from_slice(&data).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to parse cache metadata: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        self.entries = metadata.entries;
        self.access_order = metadata.access_order;
        self.total_size_bytes = metadata.total_size_bytes;

        Ok(())
    }

    /// Save cache metadata to disk
    async fn save_metadata(&self) -> Result<(), RecognitionError> {
        let metadata = CacheMetadata {
            entries: self.entries.clone(),
            access_order: self.access_order.clone(),
            total_size_bytes: self.total_size_bytes,
        };

        let data = serde_json::to_vec(&metadata).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to serialize cache metadata: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        let metadata_path = self.cache_dir.join("metadata.json");
        fs::write(&metadata_path, data).await.map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to write cache metadata: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(())
    }

    /// Get total original size of all entries
    fn total_original_size(&self) -> u64 {
        self.entries.values().map(|e| e.original_size).sum()
    }
}

/// Cache metadata for persistence
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheMetadata {
    entries: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    total_size_bytes: u64,
}