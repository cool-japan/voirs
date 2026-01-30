//! Integration tests for production features
//!
//! Tests for production monitoring, caching, and diagnostics

use std::time::Duration;
use voirs_acoustic::{
    diagnostics::DiagnosticContext,
    production_monitoring::ProductionMonitor,
    synthesis_cache::{EvictionPolicy, SynthesisCache, SynthesisCacheConfig, SynthesisCacheKey},
    MelSpectrogram,
};

#[tokio::test]
async fn test_production_monitor_basic() {
    let monitor = ProductionMonitor::new();

    // Update health
    monitor.update_health("test_component", true);
    monitor.update_health("another_component", false);

    // Record some synthesis operations
    for i in 0..10 {
        let duration = Duration::from_millis(50 + i * 10);
        let success = i != 5; // One failure
        monitor.record_synthesis(duration, success, 20 + i as usize);
    }

    // Generate report
    let report = monitor.generate_report().unwrap();

    // Verify metrics
    assert_eq!(report.metrics.total_requests, 10);
    assert_eq!(report.metrics.successful_requests, 9);
    assert_eq!(report.metrics.failed_requests, 1);
    assert!(report.metrics.success_rate > 80.0);
    assert!(report.metrics.success_rate < 100.0);

    // Verify health
    assert_eq!(report.health.component_count, 2);
    assert_eq!(report.health.healthy_count, 1);
}

#[test]
fn test_synthesis_cache_basic() {
    let config = SynthesisCacheConfig {
        max_entries: 3,
        max_size_bytes: 1024 * 1024,
        ttl: Duration::from_secs(3600),
        eviction_policy: EvictionPolicy::LRU,
        enable_stats: true,
        preload_enabled: false,
    };

    let cache = SynthesisCache::new(config);

    // Create test mel spectrograms
    let mel1 = MelSpectrogram::new(vec![vec![1.0; 100]; 80], 22050, 256);
    let mel2 = MelSpectrogram::new(vec![vec![2.0; 100]; 80], 22050, 256);

    // Create cache keys
    let key1 = SynthesisCacheKey::new("test1", Some(0), 1.0, 0.0, 1.0);
    let key2 = SynthesisCacheKey::new("test2", Some(0), 1.0, 0.0, 1.0);

    // Insert entries
    cache.insert(key1.clone(), mel1).unwrap();
    cache.insert(key2.clone(), mel2).unwrap();

    // Verify both are cached
    assert!(cache.get(&key1).is_some());
    assert!(cache.get(&key2).is_some());
}

#[test]
fn test_cache_statistics() {
    let config = SynthesisCacheConfig::default();
    let cache = SynthesisCache::new(config);

    let mel = MelSpectrogram::new(vec![vec![1.0; 50]; 80], 22050, 256);
    let key = SynthesisCacheKey::new("test", Some(0), 1.0, 0.0, 1.0);

    // Insert
    cache.insert(key.clone(), mel).unwrap();

    // Get (hit)
    assert!(cache.get(&key).is_some());

    // Get non-existent (miss)
    let key2 = SynthesisCacheKey::new("nonexistent", Some(0), 1.0, 0.0, 1.0);
    assert!(cache.get(&key2).is_none());

    let stats = cache.statistics();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entry_count, 1);
    assert!(stats.hit_rate() > 0.0);
}

#[test]
fn test_diagnostic_context() {
    use voirs_acoustic::Phoneme;

    // Create test phonemes
    let phonemes = vec![Phoneme {
        symbol: "h".to_string(),
        features: None,
        duration: Some(0.1),
    }];

    let mut ctx = DiagnosticContext::new("test_operation", &phonemes);

    // Add performance checkpoints
    ctx.checkpoint("text_encoding");
    std::thread::sleep(Duration::from_millis(10));
    ctx.checkpoint("mel_generation");

    // Add warnings and metrics
    ctx.add_warning("Test warning");
    ctx.add_metric("frames_generated", 100.0);
    ctx.add_metric("rtf", 0.25);

    // Verify context
    assert!(ctx.checkpoints.len() >= 2);
    assert_eq!(ctx.warnings.len(), 1);
    assert_eq!(ctx.metrics.len(), 2);
}

#[test]
fn test_cache_eviction_policies() {
    for policy in &[
        EvictionPolicy::LRU,
        EvictionPolicy::LFU,
        EvictionPolicy::Hybrid,
    ] {
        let config = SynthesisCacheConfig {
            max_entries: 2,
            eviction_policy: *policy,
            ..Default::default()
        };
        let cache = SynthesisCache::new(config);

        let mel = MelSpectrogram::new(vec![vec![1.0; 50]; 80], 22050, 256);
        let key = SynthesisCacheKey::new("test", Some(0), 1.0, 0.0, 1.0);

        assert!(cache.insert(key.clone(), mel).is_ok());
        assert!(cache.get(&key).is_some());
    }
}

#[test]
fn test_cache_clear() {
    let config = SynthesisCacheConfig::default();
    let cache = SynthesisCache::new(config);

    let mel = MelSpectrogram::new(vec![vec![1.0; 50]; 80], 22050, 256);
    let key = SynthesisCacheKey::new("test", Some(0), 1.0, 0.0, 1.0);

    cache.insert(key.clone(), mel).unwrap();
    assert!(cache.get(&key).is_some());

    cache.clear();
    assert!(cache.get(&key).is_none());

    let stats = cache.statistics();
    assert_eq!(stats.entry_count, 0);
}

#[test]
fn test_cache_key_quantization() {
    // Keys with similar parameters should be the same due to quantization
    let key1 = SynthesisCacheKey::new("test", Some(0), 1.04, 0.02, 1.03);
    let key2 = SynthesisCacheKey::new("test", Some(0), 1.06, 0.04, 1.07);

    // Both should quantize to the same values
    assert_eq!(key1, key2);

    // Different enough values should be different
    let key3 = SynthesisCacheKey::new("test", Some(0), 1.15, 0.02, 1.03);
    assert_ne!(key1, key3);
}

#[test]
fn test_monitoring_report_structure() {
    let monitor = ProductionMonitor::new();

    // Record some data
    for i in 0..5 {
        monitor.record_synthesis(Duration::from_millis(100 + i * 10), true, 20);
    }

    let report = monitor.generate_report().unwrap();

    // Verify report structure
    assert_eq!(report.metrics.total_requests, 5);
    assert!(report.performance.sample_count > 0);
}

#[test]
fn test_cache_size_limits() {
    let config = SynthesisCacheConfig {
        max_entries: 10,
        max_size_bytes: 1024,
        ..Default::default()
    };

    let cache = SynthesisCache::new(config);

    // Insert entries
    for i in 0..15 {
        let mel = MelSpectrogram::new(vec![vec![1.0; 10]; 10], 22050, 256);
        let key = SynthesisCacheKey::new(&format!("test_{}", i), Some(0), 1.0, 0.0, 1.0);
        let _ = cache.insert(key, mel);
    }

    let stats = cache.statistics();
    // Should have evicted some entries due to size/count limits
    assert!(stats.entry_count <= 10);
}
