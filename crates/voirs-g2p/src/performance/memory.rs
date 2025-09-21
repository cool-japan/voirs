//! Memory pool for efficient object allocation and reuse

use std::sync::{Arc, Mutex};

use crate::{G2p, LanguageCode, Phoneme, Result};

/// Memory pool for efficient object allocation and reuse
pub struct MemoryPool<T> {
    pool: Arc<Mutex<Vec<T>>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    _max_size: usize,
    stats: Arc<Mutex<PoolStats>>,
}

/// Pool statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub peak_size: usize,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
            factory: Arc::new(factory),
            _max_size: max_size,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Get an object from the pool or create new one
    pub fn acquire(&self) -> PooledObject<T> {
        let mut pool = self.pool.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let object = if let Some(obj) = pool.pop() {
            stats.pool_hits += 1;
            obj
        } else {
            stats.pool_misses += 1;
            stats.allocations += 1;
            (self.factory)()
        };

        PooledObject::new(object, self.pool.clone(), self.stats.clone())
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T> {
    object: Option<T>,
    pool: Arc<Mutex<Vec<T>>>,
    stats: Arc<Mutex<PoolStats>>,
}

impl<T> PooledObject<T> {
    fn new(object: T, pool: Arc<Mutex<Vec<T>>>, stats: Arc<Mutex<PoolStats>>) -> Self {
        Self {
            object: Some(object),
            pool,
            stats,
        }
    }

    /// Get reference to the object
    pub fn get(&self) -> &T {
        self.object.as_ref().unwrap()
    }

    /// Get mutable reference to the object
    pub fn get_mut(&mut self) -> &mut T {
        self.object.as_mut().unwrap()
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            let mut pool = self.pool.lock().unwrap();
            let mut stats = self.stats.lock().unwrap();

            stats.deallocations += 1;
            pool.push(object);

            if pool.len() > stats.peak_size {
                stats.peak_size = pool.len();
            }
        }
    }
}

/// Enhanced async batch processor with memory pooling
pub struct EnhancedBatchProcessor {
    phoneme_pool: Arc<MemoryPool<Vec<Phoneme>>>,
    string_pool: Arc<MemoryPool<String>>,
    concurrency_limit: usize,
}

impl EnhancedBatchProcessor {
    /// Create new enhanced batch processor
    pub fn new(concurrency_limit: usize) -> Self {
        Self {
            phoneme_pool: Arc::new(MemoryPool::new(Vec::new, 1000)),
            string_pool: Arc::new(MemoryPool::new(String::new, 1000)),
            concurrency_limit,
        }
    }

    /// Process texts in optimized batches with memory pooling
    pub async fn process_batch_optimized<G>(
        &self,
        backend: Arc<G>,
        texts: Vec<String>,
        language: Option<LanguageCode>,
    ) -> Result<Vec<Result<Vec<Phoneme>>>>
    where
        G: G2p + Send + Sync + 'static,
    {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.concurrency_limit));
        let mut tasks = Vec::with_capacity(texts.len());

        for text in texts {
            let backend = backend.clone();
            let semaphore = semaphore.clone();
            let pool = self.phoneme_pool.clone();
            // Use the provided language parameter directly

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Use pooled memory for result
                let mut pooled_result = pool.acquire();
                pooled_result.get_mut().clear();

                match backend.to_phonemes(&text, language).await {
                    Ok(phonemes) => {
                        pooled_result.get_mut().extend(phonemes);
                        Ok(pooled_result.get().clone())
                    }
                    Err(e) => Err(e),
                }
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(tasks.len());
        for task in tasks {
            results.push(task.await.unwrap());
        }

        Ok(results)
    }

    /// Get memory pool statistics
    pub fn pool_stats(&self) -> (PoolStats, PoolStats) {
        (self.phoneme_pool.stats(), self.string_pool.stats())
    }
}

/// Timed execution macro for performance monitoring
#[macro_export]
macro_rules! timed {
    ($monitor:expr, $name:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        $monitor.record_timing($name, duration);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(Vec::<i32>::new, 10);

        // Test allocation
        let mut obj1 = pool.acquire();
        obj1.get_mut().push(42);
        assert_eq!(obj1.get().len(), 1);

        // Test pool stats
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.pool_misses, 1);

        // Drop object to return to pool
        drop(obj1);

        // Test pool reuse
        let obj2 = pool.acquire();
        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 1);
        // Note: Objects are reused as-is, not cleared
        assert_eq!(obj2.get().len(), 1);
    }

    #[test]
    fn test_pool_stats() {
        let pool = MemoryPool::new(Vec::<i32>::new, 5);

        // Create and drop several objects
        for _ in 0..3 {
            let obj = pool.acquire();
            drop(obj);
        }

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1); // Only one object was allocated, others were reused
        assert_eq!(stats.deallocations, 3);
        assert_eq!(stats.pool_misses, 1); // Only first acquisition was a miss
        assert_eq!(stats.pool_hits, 2); // Second and third acquisitions were hits
        assert_eq!(stats.peak_size, 1); // Peak size is the max objects in pool at any one time
    }

    #[tokio::test]
    async fn test_enhanced_batch_processor() {
        use crate::DummyG2p;

        let processor = EnhancedBatchProcessor::new(2);
        let backend = Arc::new(DummyG2p::new());
        let texts = vec!["hello".to_string(), "world".to_string()];

        let results = processor
            .process_batch_optimized(backend, texts, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        // Check pool stats
        let (phoneme_stats, _string_stats) = processor.pool_stats();
        assert!(phoneme_stats.allocations > 0);
    }
}
