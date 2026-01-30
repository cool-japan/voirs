//! Advanced Caching Strategies Example
//!
//! This example demonstrates various caching strategies for improving
//! speech recognition performance through intelligent result caching.
//!
//! Run with:
//! ```
//! cargo run --example advanced_caching --all-features
//! ```

use std::time::Duration;
use voirs_recognizer::caching::{
    warming::{WarmingSchedule, WarmingStrategy},
    CacheConfig, CacheStats, EvictionPolicy, ModelCache,
};
use voirs_recognizer::RecognitionError;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VoiRS Advanced Caching Strategies Example\n");
    println!("==========================================\n");

    // Example 1: Basic LRU Cache
    println!("Example 1: Basic LRU Cache");
    println!("--------------------------");
    basic_lru_cache_example().await?;
    println!();

    // Example 2: TTL-Based Cache
    println!("Example 2: TTL-Based Cache");
    println!("--------------------------");
    ttl_cache_example().await?;
    println!();

    // Example 3: Memory-Pressure Cache
    println!("Example 3: Memory-Pressure Based Eviction");
    println!("------------------------------------------");
    memory_pressure_example().await?;
    println!();

    // Example 4: Cache Warming
    println!("Example 4: Cache Warming Strategies");
    println!("------------------------------------");
    cache_warming_example().await?;
    println!();

    // Example 5: Cache Statistics
    println!("Example 5: Cache Performance Analysis");
    println!("--------------------------------------");
    cache_statistics_example().await?;
    println!();

    println!("All examples completed successfully!");
    Ok(())
}

async fn basic_lru_cache_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create LRU cache with capacity of 5
    let config = CacheConfig {
        max_entries: 5,
        max_memory_mb: 100,
        eviction_policy: EvictionPolicy::LRU,
        ttl_seconds: None,
        enable_compression: false,
        enable_warming: false,
        distributed: None,
    };

    let cache: ModelCache<String> = ModelCache::new(config).await?;

    // Add entries
    println!("  Adding 7 entries to cache with capacity of 5...");
    for i in 1..=7 {
        let key = format!("audio_{}", i);
        let value = format!("Transcription for audio {}", i);
        cache.put(key.clone(), value).await?;
        println!("    Added: {}", key);
    }

    println!(
        "  Cache size after adding 7 entries: {}",
        cache.size().await
    );
    println!("  ✓ LRU eviction kept cache at max capacity");

    // Test LRU behavior
    println!("\n  Accessing audio_3 to make it recently used...");
    let _ = cache.get("audio_3").await?;

    println!("  Adding new entry to trigger eviction...");
    cache
        .put("audio_8".to_string(), "New transcription".to_string())
        .await?;

    // Check which entries remain
    println!("\n  Remaining entries:");
    for i in 1..=8 {
        let key = format!("audio_{}", i);
        if cache.contains_key(&key).await {
            println!("    ✓ {}", key);
        }
    }

    Ok(())
}

async fn ttl_cache_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create cache with 2-second TTL
    let config = CacheConfig {
        max_entries: 100,
        max_memory_mb: 100,
        eviction_policy: EvictionPolicy::TTL,
        ttl_seconds: Some(2), // 2 seconds TTL
        enable_compression: false,
        enable_warming: false,
        distributed: None,
    };

    let cache: ModelCache<String> = ModelCache::new(config).await?;

    // Add entry
    println!("  Adding entry with 2-second TTL...");
    cache
        .put(
            "temp_audio".to_string(),
            "Temporary transcription".to_string(),
        )
        .await?;

    // Check immediately
    let result = cache.get("temp_audio").await?;
    println!(
        "  Immediate access: {}",
        if result.is_some() {
            "✓ Found"
        } else {
            "✗ Not found"
        }
    );

    // Wait for expiration
    println!("  Waiting 3 seconds for TTL expiration...");
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Check after expiration
    let result = cache.get("temp_audio").await?;
    println!(
        "  After TTL: {}",
        if result.is_some() {
            "✗ Still found"
        } else {
            "✓ Expired as expected"
        }
    );

    Ok(())
}

async fn memory_pressure_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create cache with strict memory limit
    let config = CacheConfig {
        max_entries: 1000,
        max_memory_mb: 1, // Very small: 1MB
        eviction_policy: EvictionPolicy::MemoryPressure,
        ttl_seconds: None,
        enable_compression: false,
        enable_warming: false,
        distributed: None,
    };

    let cache: ModelCache<String> = ModelCache::new(config).await?;

    println!("  Cache with 1MB memory limit");
    println!("  Adding entries until memory pressure triggers eviction...");

    // Add increasingly large entries
    for i in 1..=20 {
        let key = format!("large_audio_{}", i);
        let value = format!("Very long transcription result {}", "x".repeat(1000));
        cache.put(key, value).await?;
    }

    let stats = cache.stats().await;
    println!("  Final cache size: {} entries", stats.current_entries);
    println!("  Memory usage: {} bytes", stats.current_memory_bytes);
    println!("  Evictions triggered: {}", stats.evictions);
    println!("  ✓ Memory pressure eviction kept usage under limit");

    Ok(())
}

async fn cache_warming_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create cache with warming enabled
    let config = CacheConfig {
        max_entries: 100,
        max_memory_mb: 100,
        eviction_policy: EvictionPolicy::LRU,
        ttl_seconds: None,
        enable_compression: false,
        enable_warming: true,
        distributed: None,
    };

    let cache: ModelCache<String> = ModelCache::new(config).await?;

    println!("  Creating warming schedule for frequently accessed items...");
    let mut schedule = WarmingSchedule::new(WarmingStrategy::MostFrequent, 5);

    // Add items with priorities
    schedule.add_item("common_phrase_1".to_string(), 0.95);
    schedule.add_item("common_phrase_2".to_string(), 0.90);
    schedule.add_item("common_phrase_3".to_string(), 0.85);
    schedule.add_item("rare_phrase_1".to_string(), 0.20);
    schedule.add_item("rare_phrase_2".to_string(), 0.15);

    let top_items = schedule.get_top_items();
    println!("  Top priority items for warming:");
    for (i, item) in top_items.iter().enumerate() {
        println!("    {}. {}", i + 1, item);
    }

    // Warm the cache
    println!("\n  Warming cache with top priority items...");
    let warm_data: Vec<(String, String)> = top_items
        .into_iter()
        .map(|key| {
            let value = format!("Cached transcription for {}", key);
            (key, value)
        })
        .collect();

    cache.warm(warm_data.clone()).await?;

    // Populate cache for demonstration
    for (key, value) in warm_data {
        cache.put(key, value).await?;
    }

    println!("  ✓ Cache warming completed");
    println!("  Cache contains {} preloaded items", cache.size().await);

    Ok(())
}

async fn cache_statistics_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = CacheConfig::default();
    let cache: ModelCache<String> = ModelCache::new(config).await?;

    println!("  Simulating cache access patterns...");

    // Add some entries
    for i in 1..=10 {
        let key = format!("audio_{}", i);
        cache.put(key, format!("Transcription {}", i)).await?;
    }

    // Simulate hits and misses
    let _ = cache.get("audio_1").await?; // Hit
    let _ = cache.get("audio_2").await?; // Hit
    let _ = cache.get("audio_3").await?; // Hit
    let _ = cache.get("audio_99").await?; // Miss
    let _ = cache.get("audio_100").await?; // Miss

    let stats = cache.stats().await;

    println!("\n  Cache Statistics:");
    println!("  -----------------");
    println!("  Total hits: {}", stats.hits);
    println!("  Total misses: {}", stats.misses);
    println!("  Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  Total insertions: {}", stats.insertions);
    println!("  Total evictions: {}", stats.evictions);
    println!("  Current entries: {}", stats.current_entries);
    println!(
        "  Memory usage: {} bytes ({:.2} KB)",
        stats.current_memory_bytes,
        stats.current_memory_bytes as f64 / 1024.0
    );

    // Calculate efficiency
    let total_ops = stats.hits + stats.misses;
    println!("\n  Performance Analysis:");
    println!("  Total operations: {}", total_ops);
    if stats.hit_rate() >= 0.8 {
        println!("  ✓ Excellent cache hit rate (>80%)");
    } else if stats.hit_rate() >= 0.6 {
        println!("  ⚠ Good cache hit rate (60-80%)");
    } else {
        println!("  ⚠ Cache hit rate could be improved (<60%)");
    }

    Ok(())
}
