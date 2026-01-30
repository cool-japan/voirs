//! Intelligent Cache Warming Demonstration
//!
//! This example demonstrates the intelligent cache warming system that:
//! - Analyzes usage patterns to predict future cache needs
//! - Preloads frequently used models and synthesis results
//! - Optimizes cache hit rates through predictive loading
//! - Reduces cold-start latency in production environments
//!
//! Run with:
//! ```bash
//! cargo run --example cache_warming_demo --all-features
//! ```

use voirs_sdk::cache::warming::{CacheWarmer, WarmingConfig, WarmingStrategy};
use voirs_sdk::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS Intelligent Cache Warming Demo ===\n");

    // 1. Basic Cache Warming
    println!("1. Basic Frequency-Based Cache Warming");
    basic_warming_demo().await?;

    println!("\n{}\n", "=".repeat(60));

    // 2. Pattern Analysis
    println!("2. Pattern Analysis and Prediction");
    pattern_analysis_demo().await?;

    println!("\n{}\n", "=".repeat(60));

    // 3. Time-Based Patterns
    println!("3. Time-Based Pattern Detection");
    time_based_patterns_demo().await?;

    println!("\n{}\n", "=".repeat(60));

    // 4. Text Similarity Matching
    println!("4. Text Similarity-Based Prediction");
    text_similarity_demo().await?;

    println!("\n{}\n", "=".repeat(60));

    // 5. Production Workflow
    println!("5. Complete Production Workflow");
    production_workflow_demo().await?;

    Ok(())
}

/// Basic frequency-based cache warming
async fn basic_warming_demo() -> Result<()> {
    let config = WarmingConfig::default()
        .with_strategy(WarmingStrategy::FrequencyBased)
        .with_pattern_window(100);

    let warmer = CacheWarmer::new(config);

    println!("Recording access patterns...");

    // Simulate frequent accesses
    for _ in 0..10 {
        warmer
            .record_access("voice_en_us", "Hello, how are you?")
            .await?;
    }

    for _ in 0..7 {
        warmer.record_access("voice_en_us", "Good morning").await?;
    }

    for _ in 0..3 {
        warmer.record_access("voice_en_uk", "Cheerio!").await?;
    }

    // Analyze patterns
    let analysis = warmer.analyze_patterns().await?;

    println!("\nPattern Analysis Results:");
    println!("  Frequent voices:");
    for (voice, count) in &analysis.frequent_voices {
        println!("    - {}: {} accesses", voice, count);
    }

    println!("\n  Common text patterns:");
    for (text, count) in &analysis.common_patterns {
        println!("    - \"{}\": {} times", text, count);
    }

    println!(
        "\n  Warming predictions: {} entries",
        analysis.predictions.len()
    );
    for prediction in &analysis.predictions {
        println!(
            "    → {} (confidence: {:.1}%)",
            prediction.text_pattern,
            prediction.confidence * 100.0
        );
    }

    // Wait for warming interval
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Warm the cache
    let stats = warmer.warm_cache().await?;
    println!("\nCache warming complete:");
    println!("  Entries warmed: {}", stats.entries_warmed);
    println!("  Predictions used: {}", stats.predictions_used);
    println!("  Time taken: {:?}", stats.warming_time);

    Ok(())
}

/// Pattern analysis and prediction
async fn pattern_analysis_demo() -> Result<()> {
    let config = WarmingConfig::default()
        .with_strategy(WarmingStrategy::Predictive)
        .with_similarity_threshold(0.6);

    let warmer = CacheWarmer::new(config);

    println!("Simulating user interaction patterns...");

    // Simulate a user session with related queries
    let session_queries = vec![
        ("voice_assistant", "What's the weather?"),
        ("voice_assistant", "What's the weather today?"),
        ("voice_assistant", "What's the weather tomorrow?"),
        ("voice_assistant", "Set a timer for 5 minutes"),
        ("voice_assistant", "Set a timer for 10 minutes"),
        ("voice_assistant", "Play some music"),
        ("voice_assistant", "Play jazz music"),
    ];

    for (voice, text) in &session_queries {
        warmer.record_access(voice, text).await?;
        println!("  Recorded: \"{}\"", text);
    }

    // Analyze patterns
    let analysis = warmer.analyze_patterns().await?;

    println!("\nPredictive Analysis:");
    println!("  Total patterns detected: {}", analysis.predictions.len());

    for prediction in analysis.predictions.iter().take(5) {
        println!(
            "  → \"{}\" (confidence: {:.0}%, reason: {})",
            prediction.text_pattern,
            prediction.confidence * 100.0,
            prediction.reason
        );
    }

    Ok(())
}

/// Time-based pattern detection
async fn time_based_patterns_demo() -> Result<()> {
    let config = WarmingConfig::default()
        .with_strategy(WarmingStrategy::TimeBased)
        .enable_time_patterns(true);

    let warmer = CacheWarmer::new(config);

    println!("Recording time-based access patterns...");

    // Simulate time-based access patterns
    for _ in 0..5 {
        warmer
            .record_access("morning_voice", "Good morning news")
            .await?;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    for _ in 0..3 {
        warmer
            .record_access("general_voice", "Random query")
            .await?;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Analyze patterns
    let analysis = warmer.analyze_patterns().await?;

    println!("\nTime-Based Analysis:");

    if !analysis.hourly_patterns.is_empty() {
        println!("  Hourly distribution:");
        let mut hourly: Vec<_> = analysis.hourly_patterns.iter().collect();
        hourly.sort_by_key(|&(hour, _)| hour);

        for (hour, count) in hourly {
            println!("    Hour {}: {} accesses", hour, count);
        }
    }

    if !analysis.daily_patterns.is_empty() {
        println!("\n  Daily distribution:");
        let mut daily: Vec<_> = analysis.daily_patterns.iter().collect();
        daily.sort_by_key(|&(day, _)| day);

        let day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        for (day, count) in daily {
            let day_name = day_names.get(*day as usize).unwrap_or(&"Unknown");
            println!("    {}: {} accesses", day_name, count);
        }
    }

    Ok(())
}

/// Text similarity-based prediction
async fn text_similarity_demo() -> Result<()> {
    let config = WarmingConfig::default()
        .with_strategy(WarmingStrategy::Predictive)
        .with_similarity_threshold(0.5)
        .enable_text_similarity(true);

    let warmer = CacheWarmer::new(config);

    println!("Testing text similarity matching...");

    // Related queries
    let queries = vec![
        "Tell me about the weather",
        "What's the weather like",
        "Weather forecast please",
        "Is it going to rain",
        "Set an alarm for 7 AM",
        "Set alarm for seven in the morning",
    ];

    for query in &queries {
        warmer.record_access("smart_assistant", query).await?;
        println!("  Recorded: \"{}\"", query);
    }

    // Analyze with most recent query
    let analysis = warmer.analyze_patterns().await?;

    println!("\nSimilarity-Based Predictions:");
    println!("  Found {} similar patterns", analysis.predictions.len());

    // Show predictions with similarity scores
    for prediction in analysis.predictions.iter().take(5) {
        if prediction.reason.contains("Similar") {
            println!(
                "  → \"{}\" ({})",
                prediction.text_pattern, prediction.reason
            );
        }
    }

    Ok(())
}

/// Complete production workflow
async fn production_workflow_demo() -> Result<()> {
    println!("Simulating production cache warming workflow...");

    // Create warmer with production settings
    let config = WarmingConfig::default()
        .with_strategy(WarmingStrategy::Hybrid)
        .with_pattern_window(500)
        .with_similarity_threshold(0.7)
        .enable_time_patterns(true)
        .enable_text_similarity(true);

    let warmer = CacheWarmer::new(config);

    println!("\nStep 1: Simulate production traffic patterns");

    // Simulate various usage patterns
    let production_patterns = vec![
        // Frequent user queries
        ("user_voice", "What's my schedule?", 8),
        ("user_voice", "Read my messages", 6),
        ("user_voice", "Call my mom", 4),
        // Administrative queries
        ("admin_voice", "System status", 5),
        ("admin_voice", "Show logs", 3),
        // Occasional queries
        ("casual_voice", "Tell me a joke", 2),
        ("casual_voice", "What time is it", 2),
    ];

    let mut total_recorded = 0;
    for (voice, text, count) in &production_patterns {
        for _ in 0..*count {
            warmer.record_access(voice, text).await?;
            total_recorded += 1;
        }
    }

    println!("  Recorded {} access patterns", total_recorded);

    // Get statistics
    let stats = warmer.get_stats().await?;
    println!("\nStep 2: Current cache warming statistics");
    println!("  Total accesses: {}", stats.total_accesses);
    println!("  Unique patterns: {}", stats.unique_patterns);
    println!("  Warming cycles: {}", stats.warming_cycles);

    // Analyze patterns
    println!("\nStep 3: Analyze patterns and generate predictions");
    let analysis = warmer.analyze_patterns().await?;

    println!("  Most frequent voices:");
    for (voice, count) in analysis.frequent_voices.iter().take(3) {
        println!("    - {}: {} times", voice, count);
    }

    println!("\n  Top predictions for warming:");
    for prediction in analysis.predictions.iter().take(5) {
        println!(
            "    → {} (confidence: {:.0}%, {})",
            prediction.text_pattern,
            prediction.confidence * 100.0,
            prediction.reason
        );
    }

    // Wait for warming interval
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Perform cache warming
    println!("\nStep 4: Warm the cache");
    let warming_stats = warmer.warm_cache().await?;

    println!("  ✓ Cache warming completed successfully");
    println!("  Entries warmed: {}", warming_stats.entries_warmed);
    println!(
        "  Predictions processed: {}",
        warming_stats.predictions_used
    );
    println!("  Time taken: {:?}", warming_stats.warming_time);

    println!("\nStep 5: Production deployment recommendations");
    println!("  1. Deploy with hybrid warming strategy for best results");
    println!("  2. Monitor prediction accuracy and adjust thresholds");
    println!("  3. Consider time-based patterns for scheduled workloads");
    println!("  4. Use text similarity for intelligent query prediction");
    println!("  5. Adjust warming interval based on traffic patterns");

    println!("\n✓ Production workflow demonstration complete!");

    Ok(())
}
