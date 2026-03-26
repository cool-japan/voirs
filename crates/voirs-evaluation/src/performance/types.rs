//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use parking_lot::RwLock;
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;
use std::sync::Arc;

/// Thread-safe LRU cache for expensive computations
pub struct LRUCache<K, V> {
    map: Arc<RwLock<HashMap<K, V>>>,
    max_size: usize,
}
impl<K, V> LRUCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU cache with the specified maximum size
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }
    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.read().get(key).cloned()
    }
    /// Insert a key-value pair into the cache
    pub fn insert(&self, key: K, value: V) {
        let mut map = self.map.write();
        if map.len() >= self.max_size {
            map.clear();
        }
        map.insert(key, value);
    }
    /// Clear all entries from the cache
    pub fn clear(&self) {
        self.map.write().clear();
    }
    /// Get the current number of entries in the cache
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.read().len()
    }
    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }
}
/// Memory-efficient sliding window processor
pub struct SlidingWindowProcessor<T> {
    window_size: usize,
    hop_size: usize,
    buffer: Vec<T>,
}
impl<T> SlidingWindowProcessor<T>
where
    T: Clone + Default,
{
    /// Create a new sliding window processor with specified window and hop sizes
    #[must_use]
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        Self {
            window_size,
            hop_size,
            buffer: Vec::with_capacity(window_size),
        }
    }
    /// Process data in sliding windows with parallel computation
    pub fn process_parallel<F, R>(&self, data: &[T], processor: F) -> Vec<R>
    where
        T: Send + Sync,
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
    {
        if data.len() < self.window_size {
            return Vec::new();
        }
        let num_windows = (data.len() - self.window_size) / self.hop_size + 1;
        (0..num_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * self.hop_size;
                let end = (start + self.window_size).min(data.len());
                processor(&data[start..end])
            })
            .collect()
    }
}
/// Performance monitoring utilities
pub struct PerformanceMonitor {
    timings: Arc<RwLock<HashMap<String, Vec<std::time::Duration>>>>,
}
impl PerformanceMonitor {
    /// Create a new performance monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            timings: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    /// Time an operation and record the duration
    pub fn time_operation<F, R>(&self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();
        self.timings
            .write()
            .entry(name.to_string())
            .or_default()
            .push(duration);
        result
    }
    /// Get the average time for a named operation
    #[must_use]
    pub fn get_average_time(&self, name: &str) -> Option<std::time::Duration> {
        let timings = self.timings.read();
        if let Some(times) = timings.get(name) {
            if times.is_empty() {
                None
            } else {
                let total: std::time::Duration = times.iter().sum();
                Some(total / times.len() as u32)
            }
        } else {
            None
        }
    }
    /// Clear all recorded timings
    pub fn clear(&self) {
        self.timings.write().clear();
    }
}
/// Persistent cache with compression support
///
/// This cache stores data to disk with compression for efficient storage
/// and retrieval across application restarts.
pub struct PersistentCache<K, V> {
    memory_cache: LRUCache<K, V>,
    cache_dir: std::path::PathBuf,
    compression_level: u32,
}
impl<K, V> PersistentCache<K, V>
where
    K: Eq + Hash + Clone + serde::Serialize + serde::de::DeserializeOwned,
    V: Clone + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new persistent cache with the specified directory and settings
    pub fn new<P: AsRef<Path>>(
        cache_dir: P,
        max_memory_size: usize,
        compression_level: u32,
    ) -> Result<Self, std::io::Error> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }
        Ok(Self {
            memory_cache: LRUCache::new(max_memory_size),
            cache_dir,
            compression_level: compression_level.clamp(0, 9),
        })
    }
    /// Get a value from the cache (checks memory first, then disk)
    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(value) = self.memory_cache.get(key) {
            return Some(value);
        }
        if let Ok(value) = self.load_from_disk(key) {
            self.memory_cache.insert(key.clone(), value.clone());
            return Some(value);
        }
        None
    }
    /// Insert a key-value pair into the cache (saves to both memory and disk)
    pub fn insert(&self, key: K, value: V) -> Result<(), std::io::Error> {
        self.memory_cache.insert(key.clone(), value.clone());
        self.save_to_disk(&key, &value)
    }
    /// Clear all entries from the cache
    pub fn clear(&self) -> Result<(), std::io::Error> {
        self.memory_cache.clear();
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
    /// Get the current number of entries in memory cache
    #[must_use]
    pub fn memory_len(&self) -> usize {
        self.memory_cache.len()
    }
    /// Get the total number of entries (including disk)
    pub fn total_len(&self) -> usize {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| entries.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
    }
    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory_cache.is_empty() && self.total_len() == 0
    }
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            memory_entries: self.memory_cache.len(),
            disk_entries: self.total_len(),
            cache_dir_size: self.calculate_cache_dir_size(),
        }
    }
    /// Set compression level (0-9, where 9 is highest compression)
    pub fn set_compression_level(&mut self, level: u32) {
        self.compression_level = level.clamp(0, 9);
    }
    fn cache_key_to_filename(&self, key: &K) -> Result<String, std::io::Error> {
        let serialized = oxicode::serde::encode_to_vec(key, oxicode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            let mut hasher = DefaultHasher::new();
            hasher.write(&serialized);
            hasher.finish()
        };
        Ok(format!("{:x}.cache", hash))
    }
    fn save_to_disk(&self, key: &K, value: &V) -> Result<(), std::io::Error> {
        let filename = self.cache_key_to_filename(key)?;
        let filepath = self.cache_dir.join(filename);
        let serialized = oxicode::serde::encode_to_vec(value, oxicode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut encoder =
            oxiarc_deflate::GzipStreamEncoder::new(Vec::new(), self.compression_level as u8);
        std::io::Write::write_all(&mut encoder, &serialized)?;
        let compressed = encoder.finish()?;
        std::fs::write(filepath, compressed)
    }
    fn load_from_disk(&self, key: &K) -> Result<V, std::io::Error> {
        let filename = self.cache_key_to_filename(key)?;
        let filepath = self.cache_dir.join(filename);
        if !filepath.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Cache entry not found",
            ));
        }
        let compressed_data = std::fs::read(filepath)?;
        let mut decoder = oxiarc_deflate::GzipStreamDecoder::new(&compressed_data[..]);
        let mut decompressed = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decompressed)?;
        oxicode::serde::decode_from_slice(&decompressed, oxicode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    fn calculate_cache_dir_size(&self) -> u64 {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok().and_then(|e| e.metadata().ok()).map(|m| m.len()))
                    .sum()
            })
            .unwrap_or(0)
    }
}
/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Number of entries on disk
    pub disk_entries: usize,
    /// Total size of cache directory in bytes
    pub cache_dir_size: u64,
}
