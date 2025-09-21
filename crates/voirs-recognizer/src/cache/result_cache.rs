//! Result cache for transcription outputs

use crate::cache::{CacheConfig, CleanupResult, TranscriptionResult};
use crate::RecognitionError;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Result cache entry
#[derive(Debug, Clone)]
struct ResultCacheEntry {
    /// Transcription result
    result: TranscriptionResult,
    /// Model ID used for transcription
    model_id: String,
    /// Creation time
    created_at: Instant,
    /// Last access time
    last_accessed: Instant,
    /// Access count
    access_count: u64,
    /// Processing time in milliseconds
    processing_time_ms: u64,
    /// Confidence score
    confidence: f32,
}

/// Cache for transcription results
pub struct ResultCache {
    config: CacheConfig,
    entries: HashMap<String, ResultCacheEntry>,
    model_entries: HashMap<String, Vec<String>>, // model_id -> list of cache keys
    access_order: Vec<String>,
    total_entries: usize,
}

impl ResultCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            model_entries: HashMap::new(),
            access_order: Vec::new(),
            total_entries: 0,
        }
    }

    /// Store transcription result
    pub async fn store_result(
        &mut self,
        input_hash: &str,
        result: &TranscriptionResult,
        model_id: &str,
    ) -> Result<(), RecognitionError> {
        let key = format!("{}_{}", input_hash, model_id);

        // Check cache size limit
        if self.total_entries >= self.config.max_items {
            self.evict_lru_entries(1).await?;
        }

        // Remove old entry if exists
        if let Some(_) = self.entries.remove(&key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        } else {
            self.total_entries += 1;
        }

        // Create new entry
        let entry = ResultCacheEntry {
            result: result.clone(),
            model_id: model_id.to_string(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            processing_time_ms: result.processing_time_ms,
            confidence: result.confidence,
        };

        // Update caches
        self.entries.insert(key.clone(), entry);
        self.access_order.push(key.clone());

        // Update model entries index
        self.model_entries
            .entry(model_id.to_string())
            .or_insert_with(Vec::new)
            .push(key);

        Ok(())
    }

    /// Get cached transcription result
    pub async fn get_result(
        &mut self,
        input_hash: &str,
        model_id: &str,
    ) -> Option<TranscriptionResult> {
        let key = format!("{}_{}", input_hash, model_id);

        if let Some(entry) = self.entries.get_mut(&key) {
            // Check if expired
            if entry.created_at.elapsed() > self.config.default_ttl {
                self.remove_entry(&key).await;
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

            Some(entry.result.clone())
        } else {
            None
        }
    }

    /// Get results by model
    pub async fn get_results_by_model(&self, model_id: &str) -> Vec<TranscriptionResult> {
        if let Some(keys) = self.model_entries.get(model_id) {
            keys.iter()
                .filter_map(|key| self.entries.get(key))
                .filter(|entry| entry.created_at.elapsed() <= self.config.default_ttl)
                .map(|entry| entry.result.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get cached results with minimum confidence
    pub async fn get_high_confidence_results(&self, min_confidence: f32) -> Vec<TranscriptionResult> {
        self.entries
            .values()
            .filter(|entry| {
                entry.confidence >= min_confidence
                    && entry.created_at.elapsed() <= self.config.default_ttl
            })
            .map(|entry| entry.result.clone())
            .collect()
    }

    /// Get fastest processing results (for performance analysis)
    pub async fn get_fastest_results(&self, limit: usize) -> Vec<(TranscriptionResult, u64)> {
        let mut results: Vec<_> = self.entries
            .values()
            .filter(|entry| entry.created_at.elapsed() <= self.config.default_ttl)
            .map(|entry| (entry.result.clone(), entry.processing_time_ms))
            .collect();

        results.sort_by_key(|(_, time)| *time);
        results.truncate(limit);
        results
    }

    /// Get statistics for a specific model
    pub async fn get_model_stats(&self, model_id: &str) -> ModelStats {
        if let Some(keys) = self.model_entries.get(model_id) {
            let valid_entries: Vec<_> = keys
                .iter()
                .filter_map(|key| self.entries.get(key))
                .filter(|entry| entry.created_at.elapsed() <= self.config.default_ttl)
                .collect();

            if valid_entries.is_empty() {
                return ModelStats::default();
            }

            let total_entries = valid_entries.len();
            let avg_confidence = valid_entries.iter().map(|e| e.confidence).sum::<f32>() / total_entries as f32;
            let avg_processing_time = valid_entries.iter().map(|e| e.processing_time_ms).sum::<u64>() / total_entries as u64;
            let total_access_count = valid_entries.iter().map(|e| e.access_count).sum::<u64>();

            ModelStats {
                total_results: total_entries,
                average_confidence: avg_confidence,
                average_processing_time_ms: avg_processing_time,
                total_access_count,
                cache_hit_rate: if total_access_count > 0 {
                    total_access_count as f64 / (total_access_count + total_entries as u64) as f64
                } else {
                    0.0
                },
            }
        } else {
            ModelStats::default()
        }
    }

    /// Search cached results by text content
    pub async fn search_by_text(&self, query: &str) -> Vec<TranscriptionResult> {
        let query_lower = query.to_lowercase();
        
        self.entries
            .values()
            .filter(|entry| {
                entry.created_at.elapsed() <= self.config.default_ttl
                    && entry.result.text.to_lowercase().contains(&query_lower)
            })
            .map(|entry| entry.result.clone())
            .collect()
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> ResultCacheStats {
        let now = Instant::now();
        let valid_entries: Vec<_> = self.entries
            .values()
            .filter(|entry| now.duration_since(entry.created_at) <= self.config.default_ttl)
            .collect();

        let total_valid_entries = valid_entries.len();
        let total_access_count = valid_entries.iter().map(|e| e.access_count).sum::<u64>();
        let avg_confidence = if total_valid_entries > 0 {
            valid_entries.iter().map(|e| e.confidence).sum::<f32>() / total_valid_entries as f32
        } else {
            0.0
        };

        ResultCacheStats {
            total_entries: total_valid_entries,
            total_models: self.model_entries.len(),
            total_access_count,
            average_confidence: avg_confidence,
            memory_usage_estimate_mb: (self.entries.len() * std::mem::size_of::<ResultCacheEntry>()) as f64 / 1024.0 / 1024.0,
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
            self.remove_entry(&key).await;
            result.items_removed += 1;
            result.bytes_freed += std::mem::size_of::<ResultCacheEntry>() as u64;
        }

        Ok(result)
    }

    /// Clear all cached results
    pub async fn clear(&mut self) -> Result<(), RecognitionError> {
        self.entries.clear();
        self.model_entries.clear();
        self.access_order.clear();
        self.total_entries = 0;
        Ok(())
    }

    /// Clear results for a specific model
    pub async fn clear_model_results(&mut self, model_id: &str) -> Result<usize, RecognitionError> {
        let mut removed_count = 0;

        if let Some(keys) = self.model_entries.remove(model_id) {
            for key in keys {
                if self.entries.remove(&key).is_some() {
                    removed_count += 1;
                    self.total_entries -= 1;

                    if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                        self.access_order.remove(pos);
                    }
                }
            }
        }

        Ok(removed_count)
    }

    /// Evict LRU entries
    async fn evict_lru_entries(&mut self, count: usize) -> Result<(), RecognitionError> {
        let mut evicted = 0;
        
        while evicted < count && !self.access_order.is_empty() {
            let key = self.access_order.remove(0);
            self.remove_entry(&key).await;
            evicted += 1;
        }

        Ok(())
    }

    /// Remove entry and update indices
    async fn remove_entry(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.total_entries -= 1;

            // Remove from model entries
            if let Some(model_keys) = self.model_entries.get_mut(&entry.model_id) {
                model_keys.retain(|k| k != key);
                if model_keys.is_empty() {
                    self.model_entries.remove(&entry.model_id);
                }
            }

            // Remove from access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
        }
    }

    /// Preload results for warm-up
    pub async fn preload_common_results(&mut self) -> Result<(), RecognitionError> {
        // In a real implementation, this would load frequently accessed results
        // from persistent storage or prediction models
        tracing::debug!("Preloading common transcription results");
        Ok(())
    }
}

/// Model-specific statistics
#[derive(Debug, Clone, Default)]
pub struct ModelStats {
    pub total_results: usize,
    pub average_confidence: f32,
    pub average_processing_time_ms: u64,
    pub total_access_count: u64,
    pub cache_hit_rate: f64,
}

/// Result cache statistics
#[derive(Debug, Clone)]
pub struct ResultCacheStats {
    pub total_entries: usize,
    pub total_models: usize,
    pub total_access_count: u64,
    pub average_confidence: f32,
    pub memory_usage_estimate_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result() -> TranscriptionResult {
        TranscriptionResult {
            text: "Hello world".to_string(),
            confidence: 0.95,
            timestamps: vec![(0.0, 1.0), (1.0, 2.0)],
            language: Some("en".to_string()),
            model_used: "test_model".to_string(),
            processing_time_ms: 100,
            created_at: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_result_cache_store_get() {
        let config = CacheConfig::default();
        let mut cache = ResultCache::new(config);

        let result = create_test_result();
        let hash = "test_hash";
        let model = "test_model";

        cache.store_result(hash, &result, model).await.unwrap();
        
        let cached_result = cache.get_result(hash, model).await;
        assert!(cached_result.is_some());
        
        let cached = cached_result.unwrap();
        assert_eq!(cached.text, result.text);
        assert_eq!(cached.confidence, result.confidence);
    }

    #[tokio::test]
    async fn test_result_cache_model_stats() {
        let config = CacheConfig::default();
        let mut cache = ResultCache::new(config);

        let result = create_test_result();
        cache.store_result("hash1", &result, "model1").await.unwrap();
        cache.store_result("hash2", &result, "model1").await.unwrap();

        let stats = cache.get_model_stats("model1").await;
        assert_eq!(stats.total_results, 2);
        assert_eq!(stats.average_confidence, 0.95);
    }

    #[tokio::test]
    async fn test_result_cache_search() {
        let config = CacheConfig::default();
        let mut cache = ResultCache::new(config);

        let mut result = create_test_result();
        result.text = "This is a test".to_string();
        
        cache.store_result("hash1", &result, "model1").await.unwrap();

        let search_results = cache.search_by_text("test").await;
        assert_eq!(search_results.len(), 1);
        assert_eq!(search_results[0].text, "This is a test");
    }
}