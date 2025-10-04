//! Compression utilities for memory optimization

use std::collections::HashMap;
use std::hash::Hash;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Compressed cache entry for memory optimization
#[derive(Debug, Clone)]
struct CompressedCacheEntry {
    compressed_data: Vec<u8>,
    original_size: usize,
    timestamp: Instant,
    access_count: u64,
}

/// Cache with compression for memory optimization
pub struct CompressedCache<K, V> {
    cache: Arc<Mutex<HashMap<K, CompressedCacheEntry>>>,
    max_size: usize,
    compression_level: u32,
    stats: Arc<Mutex<CompressedCacheStats>>,
    _phantom: PhantomData<V>,
}

/// Statistics for compressed cache
#[derive(Debug, Clone, Default)]
pub struct CompressedCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub compression_ratio: f64,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
}

impl<K, V> CompressedCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone + Serialize + for<'de> Deserialize<'de>,
{
    /// Create new compressed cache
    pub fn new(max_size: usize, compression_level: u32) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            compression_level,
            stats: Arc::new(Mutex::new(CompressedCacheStats::default())),
            _phantom: PhantomData,
        }
    }

    /// Insert value with compression
    pub fn insert(&self, key: K, value: V) -> Result<()> {
        let serialized = bincode::serde::encode_to_vec(&value, bincode::config::standard())
            .map_err(|e| crate::G2pError::InvalidInput(format!("Serialization failed: {e}")))?;
        let original_size = serialized.len();

        // Compress the data
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder
            .write_all(&serialized)
            .map_err(|e| crate::G2pError::InvalidInput(format!("Compression failed: {e}")))?;
        let compressed_data = encoder.finish().map_err(|e| {
            crate::G2pError::InvalidInput(format!("Compression finish failed: {e}"))
        })?;

        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Update statistics
        stats.total_original_size += original_size;
        stats.total_compressed_size += compressed_data.len();
        stats.compression_ratio = if stats.total_original_size > 0 {
            stats.total_compressed_size as f64 / stats.total_original_size as f64
        } else {
            1.0
        };

        // Evict if needed
        if cache.len() >= self.max_size {
            if let Some(lru_key) = self.find_lru_key(&cache) {
                let removed = cache.remove(&lru_key);
                if let Some(entry) = removed {
                    stats.total_compressed_size -= entry.compressed_data.len();
                    stats.total_original_size -= entry.original_size;
                    stats.evictions += 1;
                }
            }
        }

        let entry = CompressedCacheEntry {
            compressed_data,
            original_size,
            timestamp: Instant::now(),
            access_count: 0,
        };

        cache.insert(key, entry);
        Ok(())
    }

    /// Get value with decompression
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let mut cache = self.cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            entry.access_count += 1;
            entry.timestamp = Instant::now();
            stats.hits += 1;

            // Decompress the data
            let mut decoder = GzDecoder::new(&entry.compressed_data[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| crate::G2pError::InvalidInput(format!("Decompression failed: {e}")))?;

            let (value, _): (V, usize) =
                bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())
                    .map_err(|e| {
                        crate::G2pError::InvalidInput(format!("Deserialization failed: {e}"))
                    })?;
            Ok(Some(value))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    /// Find LRU key for eviction
    fn find_lru_key(&self, cache: &HashMap<K, CompressedCacheEntry>) -> Option<K> {
        cache
            .iter()
            .min_by_key(|(_, entry)| (entry.access_count, entry.timestamp))
            .map(|(key, _)| key.clone())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CompressedCacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
        let mut stats = self.stats.lock().unwrap();
        stats.total_original_size = 0;
        stats.total_compressed_size = 0;
        stats.compression_ratio = 1.0;
    }
}

/// Adaptive compression manager
pub struct AdaptiveCompressionManager {
    compression_threshold: usize,
    compression_level: u32,
    stats: Arc<Mutex<CompressionStats>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CompressionStats {
    pub items_compressed: u64,
    pub items_uncompressed: u64,
    pub total_savings: usize,
    pub avg_compression_ratio: f64,
}

impl AdaptiveCompressionManager {
    /// Create new adaptive compression manager
    pub fn new(compression_threshold: usize, compression_level: u32) -> Self {
        Self {
            compression_threshold,
            compression_level,
            stats: Arc::new(Mutex::new(CompressionStats::default())),
        }
    }

    /// Decide whether to compress based on size and type
    pub fn should_compress<T: Serialize>(&self, data: &T) -> bool {
        if let Ok(serialized) = bincode::serde::encode_to_vec(data, bincode::config::standard()) {
            serialized.len() > self.compression_threshold
        } else {
            false
        }
    }

    /// Compress data adaptively
    pub fn compress_adaptive<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        let serialized = bincode::serde::encode_to_vec(data, bincode::config::standard())
            .map_err(|e| crate::G2pError::InvalidInput(format!("Serialization failed: {e}")))?;
        let mut stats = self.stats.lock().unwrap();

        if serialized.len() > self.compression_threshold {
            // Compress large data
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
            encoder
                .write_all(&serialized)
                .map_err(|e| crate::G2pError::InvalidInput(format!("Compression failed: {e}")))?;
            let compressed = encoder.finish().map_err(|e| {
                crate::G2pError::InvalidInput(format!("Compression finish failed: {e}"))
            })?;

            let savings = serialized.len().saturating_sub(compressed.len());
            stats.items_compressed += 1;
            stats.total_savings += savings;
            stats.avg_compression_ratio = compressed.len() as f64 / serialized.len() as f64;

            // Prefix with compression marker
            let mut result = vec![1u8]; // Compressed marker
            result.extend(compressed);
            Ok(result)
        } else {
            // Store uncompressed
            stats.items_uncompressed += 1;
            let mut result = vec![0u8]; // Uncompressed marker
            result.extend(serialized);
            Ok(result)
        }
    }

    /// Decompress data adaptively
    pub fn decompress_adaptive<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        if data.is_empty() {
            return Err(crate::G2pError::InvalidInput("Empty data".to_string()));
        }

        let is_compressed = data[0] == 1;
        let payload = &data[1..];

        let decompressed = if is_compressed {
            // Decompress
            let mut decoder = GzDecoder::new(payload);
            let mut result = Vec::new();
            decoder
                .read_to_end(&mut result)
                .map_err(|e| crate::G2pError::InvalidInput(format!("Decompression failed: {e}")))?;
            result
        } else {
            // Use directly
            payload.to_vec()
        };

        let (value, _): (T, usize) =
            bincode::serde::decode_from_slice(&decompressed, bincode::config::standard()).map_err(
                |e| crate::G2pError::InvalidInput(format!("Deserialization failed: {e}")),
            )?;
        Ok(value)
    }

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStats {
        self.stats.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phoneme;

    #[test]
    fn test_compressed_cache() {
        let cache: CompressedCache<String, Vec<Phoneme>> = CompressedCache::new(10, 6);

        let phonemes = vec![Phoneme::new("h"), Phoneme::new("e"), Phoneme::new("l")];

        // Test insert
        cache.insert("hello".to_string(), phonemes.clone()).unwrap();

        // Test get
        let retrieved = cache.get(&"hello".to_string()).unwrap().unwrap();
        assert_eq!(retrieved.len(), phonemes.len());

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert!(stats.total_compressed_size > 0);
        assert!(stats.total_original_size > 0);
        assert!(stats.compression_ratio > 0.0);
    }

    #[test]
    fn test_compressed_cache_eviction() {
        let cache: CompressedCache<String, Vec<Phoneme>> = CompressedCache::new(2, 6);

        let phonemes = vec![Phoneme::new("a")];

        // Fill cache
        cache.insert("key1".to_string(), phonemes.clone()).unwrap();
        cache.insert("key2".to_string(), phonemes.clone()).unwrap();
        assert_eq!(cache.size(), 2);

        // Trigger eviction
        cache.insert("key3".to_string(), phonemes).unwrap();
        assert_eq!(cache.size(), 2);

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_adaptive_compression() {
        let manager = AdaptiveCompressionManager::new(50, 6);

        // Small data - should not compress
        let small_data = vec![1, 2, 3];
        assert!(!manager.should_compress(&small_data));

        let compressed = manager.compress_adaptive(&small_data).unwrap();
        assert_eq!(compressed[0], 0); // Uncompressed marker

        let decompressed: Vec<i32> = manager.decompress_adaptive(&compressed).unwrap();
        assert_eq!(decompressed, small_data);

        // Large data - should compress
        let large_data = vec![1; 100];
        assert!(manager.should_compress(&large_data));

        let compressed = manager.compress_adaptive(&large_data).unwrap();
        assert_eq!(compressed[0], 1); // Compressed marker

        let decompressed: Vec<i32> = manager.decompress_adaptive(&compressed).unwrap();
        assert_eq!(decompressed, large_data);

        let stats = manager.stats();
        assert_eq!(stats.items_compressed, 1);
        assert_eq!(stats.items_uncompressed, 1);
        assert!(stats.total_savings > 0);
    }

    #[test]
    fn test_compression_stats() {
        let manager = AdaptiveCompressionManager::new(50, 6);

        // Compress several items
        let data1 = vec![1; 100];
        let data2 = vec![2; 120];
        let data3 = vec![3; 5]; // Should not compress

        let _ = manager.compress_adaptive(&data1).unwrap();
        let _ = manager.compress_adaptive(&data2).unwrap();
        let _ = manager.compress_adaptive(&data3).unwrap();

        let stats = manager.stats();
        assert!(stats.items_compressed >= 2);
        assert!(stats.items_uncompressed >= 1);
        assert!(stats.avg_compression_ratio > 0.0);
        assert!(stats.total_savings > 0);
    }
}
