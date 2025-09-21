//! Memory prefetching strategies for improved cache performance
//!
//! Provides intelligent prefetching algorithms for different access patterns.

use super::{CacheConfig, CACHE_LINE_SIZE};

/// Prefetch strategy for different access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,

    /// Simple next-line prefetching
    NextLine,

    /// Sequential prefetching with configurable distance
    Sequential { distance: usize },

    /// Strided prefetching for regular patterns
    Strided { stride: usize, distance: usize },

    /// Adaptive prefetching based on access history
    Adaptive,

    /// Aggressive prefetching for high-bandwidth scenarios
    Aggressive { distance: usize },
}

/// Prefetch locality hint
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchLocality {
    /// Non-temporal access (bypass cache)
    NonTemporal,

    /// Low temporal locality (L3/L2 cache)
    Low,

    /// Medium temporal locality (L2/L1 cache)
    Medium,

    /// High temporal locality (L1 cache)
    High,
}

/// Memory prefetcher for optimizing cache performance
pub struct MemoryPrefetcher {
    strategy: PrefetchStrategy,
    #[allow(dead_code)]
    config: CacheConfig,
    access_history: Vec<usize>,
    stride_detector: StrideDetector,
}

/// Stride detection for adaptive prefetching
#[derive(Debug)]
struct StrideDetector {
    last_address: Option<usize>,
    stride_history: Vec<i64>,
    current_stride: Option<i64>,
    confidence: f32,
}

impl StrideDetector {
    fn new() -> Self {
        Self {
            last_address: None,
            stride_history: Vec::with_capacity(8),
            current_stride: None,
            confidence: 0.0,
        }
    }

    fn update(&mut self, address: usize) -> Option<i64> {
        if let Some(last) = self.last_address {
            let stride = address as i64 - last as i64;

            // Add to history
            self.stride_history.push(stride);
            if self.stride_history.len() > 8 {
                self.stride_history.remove(0);
            }

            // Calculate confidence
            if self.stride_history.len() >= 3 {
                let consistent_strides = self
                    .stride_history
                    .windows(2)
                    .filter(|window| window[0] == window[1])
                    .count();

                self.confidence =
                    consistent_strides as f32 / (self.stride_history.len() - 1) as f32;

                if self.confidence > 0.7 {
                    self.current_stride = Some(stride);
                }
            }
        }

        self.last_address = Some(address);
        self.current_stride
    }

    fn predicted_stride(&self) -> Option<i64> {
        if self.confidence > 0.5 {
            self.current_stride
        } else {
            None
        }
    }
}

impl MemoryPrefetcher {
    /// Create new memory prefetcher
    pub fn new(strategy: PrefetchStrategy, config: CacheConfig) -> Self {
        Self {
            strategy,
            config,
            access_history: Vec::new(),
            stride_detector: StrideDetector::new(),
        }
    }

    /// Issue prefetch for given address
    pub fn prefetch(&mut self, address: usize, locality: PrefetchLocality) {
        // Record access for adaptive learning
        self.access_history.push(address);
        if self.access_history.len() > 100 {
            self.access_history.remove(0);
        }

        // Update stride detection
        let _predicted_stride = self.stride_detector.update(address);

        // Simplified prefetch (no-op for stable Rust compatibility)
        // In a real implementation, this would use platform-specific prefetch instructions
        self._perform_prefetch(address, locality);
    }

    /// Prefetch data based on strategy
    pub fn prefetch_range(&mut self, base_address: usize, size: usize) {
        match &self.strategy {
            PrefetchStrategy::None => {
                // No prefetching
            }
            PrefetchStrategy::NextLine => {
                let next_line = (base_address + CACHE_LINE_SIZE) & !(CACHE_LINE_SIZE - 1);
                self._perform_prefetch(next_line, PrefetchLocality::Medium);
            }
            PrefetchStrategy::Sequential { distance } => {
                for i in 1..=*distance {
                    let prefetch_addr = base_address + i * CACHE_LINE_SIZE;
                    if prefetch_addr < base_address + size {
                        self._perform_prefetch(prefetch_addr, PrefetchLocality::Medium);
                    }
                }
            }
            PrefetchStrategy::Strided { stride, distance } => {
                for i in 1..=*distance {
                    let prefetch_addr = base_address + i * stride;
                    if prefetch_addr < base_address + size {
                        self._perform_prefetch(prefetch_addr, PrefetchLocality::Low);
                    }
                }
            }
            PrefetchStrategy::Adaptive => {
                if let Some(stride) = self.stride_detector.predicted_stride() {
                    if stride > 0 {
                        let prefetch_addr = (base_address as i64 + stride) as usize;
                        if prefetch_addr < base_address + size {
                            self._perform_prefetch(prefetch_addr, PrefetchLocality::Medium);
                        }
                    }
                }
            }
            PrefetchStrategy::Aggressive { distance } => {
                for i in 1..=*distance {
                    let prefetch_addr = base_address + i * CACHE_LINE_SIZE;
                    if prefetch_addr < base_address + size {
                        self._perform_prefetch(prefetch_addr, PrefetchLocality::High);
                    }
                }
            }
        }
    }

    /// Prefetch for array access pattern
    pub fn prefetch_array<T>(&mut self, array: &[T], index: usize) {
        let element_size = std::mem::size_of::<T>();
        let base_address = array.as_ptr() as usize;
        let current_address = base_address + index * element_size;

        self.prefetch(current_address, PrefetchLocality::Medium);

        // Prefetch ahead based on strategy
        match &self.strategy {
            PrefetchStrategy::Sequential { distance } => {
                for i in 1..=*distance {
                    let prefetch_index = index + i;
                    if prefetch_index < array.len() {
                        let prefetch_address = base_address + prefetch_index * element_size;
                        self._perform_prefetch(prefetch_address, PrefetchLocality::Medium);
                    }
                }
            }
            _ => {
                // Use range prefetch for other strategies
                let remaining_size = (array.len() - index) * element_size;
                self.prefetch_range(current_address, remaining_size.min(4096)); // Limit to 4KB
            }
        }
    }

    /// Get prefetch statistics
    pub fn statistics(&self) -> PrefetchStatistics {
        PrefetchStatistics {
            total_accesses: self.access_history.len(),
            stride_confidence: self.stride_detector.confidence,
            predicted_stride: self.stride_detector.predicted_stride(),
            strategy: self.strategy.clone(),
        }
    }

    /// Update prefetch strategy
    pub fn set_strategy(&mut self, strategy: PrefetchStrategy) {
        self.strategy = strategy;
    }

    /// Simplified prefetch implementation (no-op for stable Rust)
    fn _perform_prefetch(&self, _address: usize, _locality: PrefetchLocality) {
        // No-op implementation for stable Rust compatibility
        // In a real implementation, this would use target-specific prefetch instructions
    }
}

impl Default for MemoryPrefetcher {
    fn default() -> Self {
        Self::new(
            PrefetchStrategy::Sequential { distance: 2 },
            CacheConfig::default(),
        )
    }
}

/// Prefetch statistics
#[derive(Debug, Clone)]
pub struct PrefetchStatistics {
    pub total_accesses: usize,
    pub stride_confidence: f32,
    pub predicted_stride: Option<i64>,
    pub strategy: PrefetchStrategy,
}

/// Prefetch optimizer for different scenarios
pub struct PrefetchOptimizer;

impl PrefetchOptimizer {
    /// Suggest optimal prefetch strategy for given access pattern
    pub fn suggest_strategy(
        pattern: &str,
        data_size: usize,
        access_frequency: f32,
    ) -> PrefetchStrategy {
        match pattern {
            "sequential" => {
                if data_size > 1024 * 1024 {
                    PrefetchStrategy::Aggressive { distance: 8 }
                } else {
                    PrefetchStrategy::Sequential { distance: 4 }
                }
            }
            "strided" => {
                let stride = (data_size / 100).max(64); // Estimate stride
                PrefetchStrategy::Strided {
                    stride,
                    distance: 3,
                }
            }
            "random" => PrefetchStrategy::None,
            "adaptive" => PrefetchStrategy::Adaptive,
            _ => {
                if access_frequency > 0.8 {
                    PrefetchStrategy::Aggressive { distance: 6 }
                } else {
                    PrefetchStrategy::Sequential { distance: 2 }
                }
            }
        }
    }

    /// Calculate optimal prefetch distance
    pub fn optimal_distance(cache_size: usize, data_size: usize, access_rate: f32) -> usize {
        let cache_lines = cache_size / CACHE_LINE_SIZE;
        let data_cache_lines = data_size / CACHE_LINE_SIZE;

        if data_cache_lines > cache_lines {
            // Data doesn't fit in cache
            (access_rate * 10.0) as usize
        } else {
            // Data fits in cache
            (access_rate * 4.0) as usize
        }
        .clamp(1, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stride_detector() {
        let mut detector = StrideDetector::new();

        // Sequential access pattern
        detector.update(0);
        detector.update(64);
        detector.update(128);
        detector.update(192);

        assert!(detector.confidence > 0.5);
        assert_eq!(detector.predicted_stride(), Some(64));
    }

    #[test]
    fn test_prefetch_strategy() {
        let config = CacheConfig::default();
        let mut prefetcher =
            MemoryPrefetcher::new(PrefetchStrategy::Sequential { distance: 2 }, config);

        // Test basic prefetching
        prefetcher.prefetch(0x1000, PrefetchLocality::Medium);
        prefetcher.prefetch_range(0x2000, 4096);

        let stats = prefetcher.statistics();
        assert_eq!(stats.total_accesses, 1);
    }

    #[test]
    fn test_prefetch_optimizer() {
        let strategy = PrefetchOptimizer::suggest_strategy("sequential", 1024, 0.9);
        assert_eq!(strategy, PrefetchStrategy::Sequential { distance: 4 });

        let distance = PrefetchOptimizer::optimal_distance(32 * 1024, 1024, 0.5);
        assert!((1..=16).contains(&distance));
    }

    #[test]
    fn test_array_prefetch() {
        let data = vec![1.0f32; 1000];
        let mut prefetcher = MemoryPrefetcher::default();

        prefetcher.prefetch_array(&data, 100);

        let stats = prefetcher.statistics();
        assert!(stats.total_accesses > 0);
    }
}
