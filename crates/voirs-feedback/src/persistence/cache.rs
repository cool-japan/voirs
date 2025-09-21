//! Caching layer for persistence operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::persistence::{PersistenceError, PersistenceResult};

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    created_at: DateTime<Utc>,
    ttl_seconds: u64,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl_seconds: u64) -> Self {
        Self {
            data,
            created_at: Utc::now(),
            ttl_seconds,
        }
    }

    fn is_expired(&self) -> bool {
        let age = Utc::now()
            .signed_duration_since(self.created_at)
            .num_seconds() as u64;
        age > self.ttl_seconds
    }
}

/// LRU cache with TTL support
pub struct LruTtlCache<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    access_order: Arc<RwLock<Vec<K>>>,
    max_size: usize,
    default_ttl: u64,
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl<K, V> LruTtlCache<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    /// Create a new LRU TTL cache
    pub fn new(max_size: usize, default_ttl_seconds: u64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            max_size,
            default_ttl: default_ttl_seconds,
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get a value from the cache
    pub async fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;

        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                // Remove expired entry
                cache.remove(key);
                access_order.retain(|k| k != key);
                // Count as miss since expired
                let mut misses = self.misses.write().await;
                *misses += 1;
                None
            } else {
                // Move to front of access order
                access_order.retain(|k| k != key);
                access_order.push(key.clone());
                // Count as hit
                let mut hits = self.hits.write().await;
                *hits += 1;
                Some(entry.data.clone())
            }
        } else {
            // Count as miss
            let mut misses = self.misses.write().await;
            *misses += 1;
            None
        }
    }

    /// Put a value into the cache
    pub async fn put(&self, key: K, value: V) -> PersistenceResult<()> {
        self.put_with_ttl(key, value, self.default_ttl).await
    }

    /// Put a value into the cache with custom TTL
    pub async fn put_with_ttl(&self, key: K, value: V, ttl_seconds: u64) -> PersistenceResult<()> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;

        // Remove existing entry from access order
        access_order.retain(|k| k != &key);

        // Add new entry
        let entry = CacheEntry::new(value, ttl_seconds);
        cache.insert(key.clone(), entry);
        access_order.push(key);

        // Evict if necessary
        while cache.len() > self.max_size {
            if let Some(oldest_key) = access_order.first().cloned() {
                cache.remove(&oldest_key);
                access_order.remove(0);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Remove a value from the cache
    pub async fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;

        access_order.retain(|k| k != key);
        cache.remove(key).map(|entry| entry.data)
    }

    /// Clear all expired entries
    pub async fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;

        let initial_size = cache.len();

        let expired_keys: Vec<K> = cache
            .iter()
            .filter_map(|(key, entry)| {
                if entry.is_expired() {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in &expired_keys {
            cache.remove(key);
            access_order.retain(|k| k != key);
        }

        initial_size - cache.len()
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let hits = self.hits.read().await;
        let misses = self.misses.read().await;

        let total_entries = cache.len();
        let expired_entries = cache.values().filter(|entry| entry.is_expired()).count();
        let total_accesses = *hits + *misses;
        let hit_ratio = if total_accesses > 0 {
            *hits as f64 / total_accesses as f64
        } else {
            0.0
        };

        CacheStats {
            total_entries,
            expired_entries,
            active_entries: total_entries - expired_entries,
            max_size: self.max_size,
            hit_ratio,
            hits: *hits,
            misses: *misses,
        }
    }

    /// Clear the entire cache
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        let mut hits = self.hits.write().await;
        let mut misses = self.misses.write().await;

        cache.clear();
        access_order.clear();
        *hits = 0;
        *misses = 0;
    }

    /// Reset cache statistics without clearing the cache
    pub async fn reset_stats(&self) {
        let mut hits = self.hits.write().await;
        let mut misses = self.misses.write().await;
        *hits = 0;
        *misses = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Number of active entries
    pub active_entries: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Hit ratio (0.0 to 1.0)
    pub hit_ratio: f64,
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
}

/// Multi-level cache for different data types
pub struct PersistenceCache {
    session_cache: LruTtlCache<String, crate::traits::SessionState>,
    progress_cache: LruTtlCache<String, crate::traits::UserProgress>,
    preferences_cache: LruTtlCache<String, crate::traits::UserPreferences>,
    feedback_cache: LruTtlCache<String, Vec<crate::traits::FeedbackResponse>>,
}

impl PersistenceCache {
    /// Create a new persistence cache
    pub fn new(max_size_per_type: usize, default_ttl_seconds: u64) -> Self {
        Self {
            session_cache: LruTtlCache::new(max_size_per_type, default_ttl_seconds),
            progress_cache: LruTtlCache::new(max_size_per_type, default_ttl_seconds),
            preferences_cache: LruTtlCache::new(max_size_per_type, default_ttl_seconds),
            feedback_cache: LruTtlCache::new(max_size_per_type, default_ttl_seconds),
        }
    }

    /// Get session from cache
    pub async fn get_session(&self, session_id: &str) -> Option<crate::traits::SessionState> {
        self.session_cache.get(&session_id.to_string()).await
    }

    /// Put session in cache
    pub async fn put_session(
        &self,
        session_id: &str,
        session: crate::traits::SessionState,
    ) -> PersistenceResult<()> {
        self.session_cache
            .put(session_id.to_string(), session)
            .await
    }

    /// Get user progress from cache
    pub async fn get_progress(&self, user_id: &str) -> Option<crate::traits::UserProgress> {
        self.progress_cache.get(&user_id.to_string()).await
    }

    /// Put user progress in cache
    pub async fn put_progress(
        &self,
        user_id: &str,
        progress: crate::traits::UserProgress,
    ) -> PersistenceResult<()> {
        self.progress_cache.put(user_id.to_string(), progress).await
    }

    /// Get user preferences from cache
    pub async fn get_preferences(&self, user_id: &str) -> Option<crate::traits::UserPreferences> {
        self.preferences_cache.get(&user_id.to_string()).await
    }

    /// Put user preferences in cache
    pub async fn put_preferences(
        &self,
        user_id: &str,
        preferences: crate::traits::UserPreferences,
    ) -> PersistenceResult<()> {
        self.preferences_cache
            .put(user_id.to_string(), preferences)
            .await
    }

    /// Get feedback history from cache
    pub async fn get_feedback_history(
        &self,
        user_id: &str,
    ) -> Option<Vec<crate::traits::FeedbackResponse>> {
        self.feedback_cache.get(&user_id.to_string()).await
    }

    /// Put feedback history in cache
    pub async fn put_feedback_history(
        &self,
        user_id: &str,
        feedback: Vec<crate::traits::FeedbackResponse>,
    ) -> PersistenceResult<()> {
        self.feedback_cache.put(user_id.to_string(), feedback).await
    }

    /// Clear user data from all caches
    pub async fn clear_user_data(&self, user_id: &str) {
        let user_key = user_id.to_string();
        self.session_cache.remove(&user_key).await;
        self.progress_cache.remove(&user_key).await;
        self.preferences_cache.remove(&user_key).await;
        self.feedback_cache.remove(&user_key).await;
    }

    /// Cleanup expired entries from all caches
    pub async fn cleanup_expired(&self) -> usize {
        let session_cleaned = self.session_cache.cleanup_expired().await;
        let progress_cleaned = self.progress_cache.cleanup_expired().await;
        let preferences_cleaned = self.preferences_cache.cleanup_expired().await;
        let feedback_cleaned = self.feedback_cache.cleanup_expired().await;

        session_cleaned + progress_cleaned + preferences_cleaned + feedback_cleaned
    }

    /// Get overall cache statistics
    pub async fn stats(&self) -> PersistenceCacheStats {
        let session_stats = self.session_cache.stats().await;
        let progress_stats = self.progress_cache.stats().await;
        let preferences_stats = self.preferences_cache.stats().await;
        let feedback_stats = self.feedback_cache.stats().await;

        PersistenceCacheStats {
            session_cache: session_stats,
            progress_cache: progress_stats,
            preferences_cache: preferences_stats,
            feedback_cache: feedback_stats,
        }
    }

    /// Clear all caches
    pub async fn clear_all(&self) {
        self.session_cache.clear().await;
        self.progress_cache.clear().await;
        self.preferences_cache.clear().await;
        self.feedback_cache.clear().await;
    }

    /// Reset statistics for all caches
    pub async fn reset_all_stats(&self) {
        self.session_cache.reset_stats().await;
        self.progress_cache.reset_stats().await;
        self.preferences_cache.reset_stats().await;
        self.feedback_cache.reset_stats().await;
    }
}

/// Overall persistence cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceCacheStats {
    /// Session cache statistics
    pub session_cache: CacheStats,
    /// Progress cache statistics
    pub progress_cache: CacheStats,
    /// Preferences cache statistics
    pub preferences_cache: CacheStats,
    /// Feedback cache statistics
    pub feedback_cache: CacheStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lru_ttl_cache_basic_operations() {
        let cache = LruTtlCache::new(3, 60); // 3 items, 60 second TTL

        // Test put and get
        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));

        // Test non-existent key
        let value = cache.get(&"nonexistent".to_string()).await;
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_lru_ttl_cache_eviction() {
        let cache = LruTtlCache::new(2, 60); // 2 items max

        // Fill cache
        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();
        cache
            .put("key2".to_string(), "value2".to_string())
            .await
            .unwrap();

        // Add third item, should evict first
        cache
            .put("key3".to_string(), "value3".to_string())
            .await
            .unwrap();

        // key1 should be evicted
        assert_eq!(cache.get(&"key1".to_string()).await, None);
        assert_eq!(
            cache.get(&"key2".to_string()).await,
            Some("value2".to_string())
        );
        assert_eq!(
            cache.get(&"key3".to_string()).await,
            Some("value3".to_string())
        );
    }

    #[tokio::test]
    async fn test_lru_ttl_cache_expiration() {
        let cache = LruTtlCache::new(10, 1); // 1 second TTL

        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();

        // Should still be valid within TTL
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));

        // Wait for expiration (longer than TTL)
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_persistence_cache_operations() {
        use crate::traits::{
            AdaptiveState, SessionState, SessionStatistics, SessionStats, UserPreferences,
        };
        use uuid::Uuid;

        let cache = PersistenceCache::new(100, 3600);

        // Test preferences cache
        let preferences = UserPreferences::default();
        cache
            .put_preferences("user1", preferences.clone())
            .await
            .unwrap();
        let cached_preferences = cache.get_preferences("user1").await;
        assert_eq!(cached_preferences, Some(preferences));

        // Test session cache
        let session = SessionState {
            session_id: Uuid::new_v4(),
            user_id: "user1".to_string(),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        };

        let session_id = session.session_id.to_string();
        cache
            .put_session(&session_id, session.clone())
            .await
            .unwrap();
        let cached_session = cache.get_session(&session_id).await;
        assert_eq!(cached_session, Some(session));
    }

    #[tokio::test]
    async fn test_cache_hit_ratio_tracking() {
        let cache = LruTtlCache::new(10, 60);

        // Initial stats should show 0 hits, 0 misses, 0.0 hit ratio
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_ratio, 0.0);

        // Add an item
        cache
            .put("key1".to_string(), "value1".to_string())
            .await
            .unwrap();

        // First get should be a hit
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_ratio, 1.0);

        // Second get should also be a hit
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_ratio, 1.0);

        // Get non-existent key should be a miss
        let value = cache.get(&"nonexistent".to_string()).await;
        assert_eq!(value, None);
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_ratio, 2.0 / 3.0);

        // Reset stats
        cache.reset_stats().await;
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_ratio, 0.0);
    }
}
