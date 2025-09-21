//! Feature cache for audio processing results

use crate::cache::{CacheConfig, CleanupResult};
use crate::RecognitionError;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Audio feature cache entry
#[derive(Debug, Clone)]
struct FeatureCacheEntry {
    /// Audio features
    features: Vec<f32>,
    /// Sample rate
    sample_rate: u32,
    /// Creation time
    created_at: Instant,
    /// Last access time
    last_accessed: Instant,
    /// Access count
    access_count: u64,
    /// Feature type (e.g., "mfcc", "spectrogram", "mel")
    feature_type: String,
}

/// Cache for audio features
pub struct FeatureCache {
    config: CacheConfig,
    entries: HashMap<String, FeatureCacheEntry>,
    total_memory_bytes: usize,
    access_order: Vec<String>,
}

impl FeatureCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            total_memory_bytes: 0,
            access_order: Vec::new(),
        }
    }

    /// Store audio features
    pub async fn store_features(
        &mut self,
        audio_hash: &str,
        features: &[f32],
        sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        self.store_features_with_type(audio_hash, features, sample_rate, "default").await
    }

    /// Store audio features with specific type
    pub async fn store_features_with_type(
        &mut self,
        audio_hash: &str,
        features: &[f32],
        sample_rate: u32,
        feature_type: &str,
    ) -> Result<(), RecognitionError> {
        let key = format!("{}_{}", audio_hash, feature_type);
        let feature_size_bytes = features.len() * std::mem::size_of::<f32>();

        // Check if we need to make space
        if self.total_memory_bytes + feature_size_bytes > self.config.max_cache_size_mb * 1024 * 1024 {
            self.evict_lru_entries(feature_size_bytes).await?;
        }

        // Remove old entry if exists
        if let Some(old_entry) = self.entries.remove(&key) {
            self.total_memory_bytes -= old_entry.features.len() * std::mem::size_of::<f32>();
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }

        // Create new entry
        let entry = FeatureCacheEntry {
            features: features.to_vec(),
            sample_rate,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            feature_type: feature_type.to_string(),
        };

        // Update cache
        self.total_memory_bytes += feature_size_bytes;
        self.entries.insert(key.clone(), entry);
        self.access_order.push(key);

        Ok(())
    }

    /// Get cached audio features
    pub async fn get_features(&mut self, audio_hash: &str) -> Option<(Vec<f32>, u32)> {
        self.get_features_with_type(audio_hash, "default").await
    }

    /// Get cached audio features with specific type
    pub async fn get_features_with_type(
        &mut self,
        audio_hash: &str,
        feature_type: &str,
    ) -> Option<(Vec<f32>, u32)> {
        let key = format!("{}_{}", audio_hash, feature_type);
        
        if let Some(entry) = self.entries.get_mut(&key) {
            // Check if expired
            if entry.created_at.elapsed() > self.config.default_ttl {
                self.entries.remove(&key);
                if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                    self.access_order.remove(pos);
                }
                return None;
            }

            // Update access statistics
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Update LRU order
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key);

            Some((entry.features.clone(), entry.sample_rate))
        } else {
            None
        }
    }

    /// Store MFCC features
    pub async fn store_mfcc(
        &mut self,
        audio_hash: &str,
        mfcc: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        // Flatten MFCC for storage
        let flattened: Vec<f32> = mfcc.iter().flatten().copied().collect();
        self.store_features_with_type(audio_hash, &flattened, sample_rate, "mfcc").await
    }

    /// Get cached MFCC features
    pub async fn get_mfcc(&mut self, audio_hash: &str) -> Option<(Vec<Vec<f32>>, u32)> {
        if let Some((flattened, sample_rate)) = self.get_features_with_type(audio_hash, "mfcc").await {
            // Need to know dimensions to unflatten - for now return as single vector
            // In a real implementation, you'd store dimensions metadata
            Some((vec![flattened], sample_rate))
        } else {
            None
        }
    }

    /// Store mel spectrogram features
    pub async fn store_mel_spectrogram(
        &mut self,
        audio_hash: &str,
        mel_spec: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<(), RecognitionError> {
        let flattened: Vec<f32> = mel_spec.iter().flatten().copied().collect();
        self.store_features_with_type(audio_hash, &flattened, sample_rate, "mel_spec").await
    }

    /// Get cached mel spectrogram features
    pub async fn get_mel_spectrogram(&mut self, audio_hash: &str) -> Option<(Vec<Vec<f32>>, u32)> {
        if let Some((flattened, sample_rate)) = self.get_features_with_type(audio_hash, "mel_spec").await {
            Some((vec![flattened], sample_rate))
        } else {
            None
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> FeatureCacheStats {
        FeatureCacheStats {
            total_entries: self.entries.len(),
            total_memory_mb: self.total_memory_bytes as f64 / 1024.0 / 1024.0,
            hit_count: self.entries.values().map(|e| e.access_count).sum(),
            average_feature_size: if self.entries.is_empty() {
                0.0
            } else {
                self.entries.values().map(|e| e.features.len()).sum::<usize>() as f64 / self.entries.len() as f64
            },
        }
    }

    /// Cleanup expired entries
    pub async fn cleanup(&mut self) -> Result<CleanupResult, RecognitionError> {
        let mut result = CleanupResult::default();
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // Find expired entries
        for (key, entry) in &self.entries {
            if now.duration_since(entry.created_at) > self.config.default_ttl {
                to_remove.push(key.clone());
            }
        }

        // Remove expired entries
        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                let entry_size = entry.features.len() * std::mem::size_of::<f32>();
                self.total_memory_bytes -= entry_size;
                result.bytes_freed += entry_size as u64;
                result.items_removed += 1;

                if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                    self.access_order.remove(pos);
                }
            }
        }

        Ok(result)
    }

    /// Clear all cached features
    pub async fn clear(&mut self) -> Result<(), RecognitionError> {
        self.entries.clear();
        self.access_order.clear();
        self.total_memory_bytes = 0;
        Ok(())
    }

    /// Evict LRU entries to make space
    async fn evict_lru_entries(&mut self, needed_bytes: usize) -> Result<(), RecognitionError> {
        let mut freed_bytes = 0;
        
        while freed_bytes < needed_bytes && !self.access_order.is_empty() {
            let key = self.access_order.remove(0);
            if let Some(entry) = self.entries.remove(&key) {
                let entry_size = entry.features.len() * std::mem::size_of::<f32>();
                self.total_memory_bytes -= entry_size;
                freed_bytes += entry_size;
            }
        }

        Ok(())
    }

    /// Prefetch commonly used features
    pub async fn prefetch_features(&mut self, audio_hashes: &[String]) -> Result<(), RecognitionError> {
        // In a real implementation, this would load features from a prediction model
        // or based on usage patterns
        for hash in audio_hashes {
            if !self.entries.contains_key(hash) {
                // Could trigger background feature computation here
                tracing::debug!("Would prefetch features for hash: {}", hash);
            }
        }
        Ok(())
    }

    /// Get feature similarity
    pub fn get_feature_similarity(&self, hash1: &str, hash2: &str) -> Option<f32> {
        let key1 = format!("{}_default", hash1);
        let key2 = format!("{}_default", hash2);
        
        if let (Some(entry1), Some(entry2)) = (self.entries.get(&key1), self.entries.get(&key2)) {
            Some(cosine_similarity(&entry1.features, &entry2.features))
        } else {
            None
        }
    }
}

/// Feature cache statistics
#[derive(Debug, Clone)]
pub struct FeatureCacheStats {
    pub total_entries: usize,
    pub total_memory_mb: f64,
    pub hit_count: u64,
    pub average_feature_size: f64,
}

/// Calculate cosine similarity between two feature vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_feature_cache_store_get() {
        let config = CacheConfig::default();
        let mut cache = FeatureCache::new(config);

        let features = vec![1.0, 2.0, 3.0, 4.0];
        let sample_rate = 16000;
        let hash = "test_hash";

        cache.store_features(hash, &features, sample_rate).await.unwrap();
        
        let result = cache.get_features(hash).await;
        assert!(result.is_some());
        
        let (cached_features, cached_sr) = result.unwrap();
        assert_eq!(cached_features, features);
        assert_eq!(cached_sr, sample_rate);
    }

    #[tokio::test]
    async fn test_feature_cache_expiration() {
        let mut config = CacheConfig::default();
        config.default_ttl = Duration::from_millis(10); // Very short TTL
        
        let mut cache = FeatureCache::new(config);
        let features = vec![1.0, 2.0, 3.0];
        cache.store_features("test", &features, 16000).await.unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        let result = cache.get_features("test").await;
        assert!(result.is_none());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = vec![-1.0, -2.0, -3.0];
        let similarity2 = cosine_similarity(&a, &c);
        assert!((similarity2 + 1.0).abs() < 1e-6);
    }
}