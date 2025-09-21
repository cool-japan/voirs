//! Performance optimizations for G2P conversion
//!
//! This module provides various performance optimizations for G2P conversion including:
//! - Caching systems with LRU eviction and TTL support
//! - Batch processing for high-throughput scenarios
//! - SIMD-accelerated text processing
//! - Memory pooling for efficient allocation
//! - Performance monitoring and metrics export
//! - Adaptive compression for memory optimization

pub mod batch;
pub mod cache;
pub mod compression;
pub mod memory;
pub mod metrics;
pub mod monitoring;
pub mod simd;
pub mod targets;

// Re-export all commonly used types
pub use batch::{
    BatchPhonemeProcessor, BatchProcessingStats, BatchProcessor, DynamicBatchProcessor,
    DynamicBatchStats, LengthGroupedBatch,
};
pub use cache::{CacheStats, CachedG2p, G2pCache};
pub use compression::{AdaptiveCompressionManager, CompressedCache, CompressionStats};
pub use memory::{EnhancedBatchProcessor, MemoryPool, PoolStats};
pub use metrics::{MetricsExporter, MetricsFormat, MetricsSnapshot};
pub use monitoring::{PerformanceMetric, PerformanceMonitor};
pub use targets::{
    PerformanceSummary, PerformanceTargetMonitor, PerformanceTargets, TargetViolation,
    ViolationSeverity,
};

// Re-export the timed macro
pub use crate::timed;

// Re-export SIMD functions for convenience
pub use simd::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_integrated_performance_stack() {
        // Test that all components work together
        let monitor = Arc::new(PerformanceMonitor::new(true));
        let cache = Arc::new(G2pCache::new(100));
        let processor = Arc::new(BatchPhonemeProcessor::new(50));
        let compression = Arc::new(AdaptiveCompressionManager::new(100, 6));

        // Create metrics exporter with all components
        let exporter = MetricsExporter::new(monitor.clone())
            .with_cache(cache.clone())
            .with_batch_processor(processor.clone())
            .with_compression_manager(compression.clone());

        // Test metrics export
        let snapshot = exporter.capture_snapshot();
        assert!(snapshot.timestamp > 0);

        // Test JSON export
        let json_result = exporter.export_metrics(MetricsFormat::Json);
        assert!(json_result.is_ok());

        // Test Prometheus export
        let prometheus_result = exporter.export_metrics(MetricsFormat::Prometheus);
        assert!(prometheus_result.is_ok());
    }

    #[test]
    fn test_performance_module_exports() {
        // Test that all exports are available
        let _cache: G2pCache<String, Vec<String>> = G2pCache::new(100);
        let _processor = BatchPhonemeProcessor::new(50);
        let _compression = AdaptiveCompressionManager::new(100, 6);
        let _monitor = PerformanceMonitor::new(true);
        let _exporter = MetricsExporter::new(Arc::new(PerformanceMonitor::new(true)));
    }
}
