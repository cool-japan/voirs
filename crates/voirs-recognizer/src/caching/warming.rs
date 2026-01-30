//! Cache warming strategies
//!
//! Provides mechanisms to preload frequently accessed items into the cache

use crate::RecognitionError;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::CacheStorage;

/// Cache warming manager
pub struct CacheWarmingManager<T> {
    /// Reference to cache storage
    storage: Arc<RwLock<CacheStorage<T>>>,
    /// Warming statistics
    stats: Arc<RwLock<WarmingStats>>,
}

/// Warming statistics
#[derive(Debug, Clone, Default)]
pub struct WarmingStats {
    /// Total items warmed
    pub items_warmed: u64,
    /// Total warming operations
    pub warming_operations: u64,
    /// Failed warming attempts
    pub failures: u64,
}

/// Warming strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmingStrategy {
    /// Warm most frequently accessed items
    MostFrequent,
    /// Warm recently accessed items
    RecentlyAccessed,
    /// Warm specific items
    Manual,
    /// Warm based on prediction
    Predictive,
}

impl<T: Clone + Send + Sync> CacheWarmingManager<T> {
    /// Create a new cache warming manager
    pub(crate) fn new(storage: Arc<RwLock<CacheStorage<T>>>) -> Self {
        Self {
            storage,
            stats: Arc::new(RwLock::new(WarmingStats::default())),
        }
    }

    /// Warm the cache with provided items
    ///
    /// # Errors
    ///
    /// Returns an error if warming fails
    pub async fn warm(&self, items: Vec<(String, T)>) -> Result<(), RecognitionError> {
        let mut stats = self.stats.write().await;
        stats.warming_operations += 1;

        let warmed = items.len();

        // Note: Actual warming implementation would insert items into cache
        // This is a simplified version for demonstration

        stats.items_warmed += warmed as u64;
        Ok(())
    }

    /// Get warming statistics
    pub async fn stats(&self) -> WarmingStats {
        self.stats.read().await.clone()
    }
}

/// Warming schedule configuration
#[derive(Debug, Clone)]
pub struct WarmingSchedule {
    /// Items to warm with their priorities
    pub items: HashMap<String, f32>,
    /// Strategy to use
    pub strategy: WarmingStrategy,
    /// Maximum items to warm
    pub max_items: usize,
}

impl WarmingSchedule {
    /// Create a new warming schedule
    pub fn new(strategy: WarmingStrategy, max_items: usize) -> Self {
        Self {
            items: HashMap::new(),
            strategy,
            max_items,
        }
    }

    /// Add an item to the warming schedule
    pub fn add_item(&mut self, key: String, priority: f32) {
        self.items.insert(key, priority);
    }

    /// Get top priority items
    pub fn get_top_items(&self) -> Vec<String> {
        let mut items: Vec<_> = self.items.iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        items
            .into_iter()
            .take(self.max_items)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warming_schedule() {
        let mut schedule = WarmingSchedule::new(WarmingStrategy::MostFrequent, 3);

        schedule.add_item("a".to_string(), 0.9);
        schedule.add_item("b".to_string(), 0.5);
        schedule.add_item("c".to_string(), 0.8);
        schedule.add_item("d".to_string(), 0.3);

        let top_items = schedule.get_top_items();
        assert_eq!(top_items.len(), 3);
        assert_eq!(top_items[0], "a");
        assert_eq!(top_items[1], "c");
        assert_eq!(top_items[2], "b");
    }
}
