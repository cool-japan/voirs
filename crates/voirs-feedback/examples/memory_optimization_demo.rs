//! Demonstration of memory optimization features in analytics module
//!
//! This example shows the memory efficiency gains from using the optimized analytics manager.

use chrono::Utc;
use std::collections::HashMap;
use std::time::Instant;
use voirs_feedback::analytics::{
    AnalyticsConfig, AnalyticsManagerFactory, InteractionType, MemoryProfile, UserInteractionEvent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 VoiRS Analytics Memory Optimization Demo");
    println!("============================================\n");

    // Create test configuration
    let config = AnalyticsConfig {
        enabled: true,
        max_interactions: 1000,
        max_performance_records: 500,
        retention_days: 30,
        enable_realtime: true,
        max_active_sessions: Some(100),
        ..Default::default()
    };

    println!("📊 Creating analytics managers...");

    // Create both standard and optimized managers
    let standard_manager = AnalyticsManagerFactory::create_standard(config.clone()).await?;
    let optimized_manager = AnalyticsManagerFactory::create_optimized(config.clone()).await?;

    println!("✅ Analytics managers created successfully");

    // Generate test interactions with realistic data patterns
    let test_interactions = generate_test_interactions(500);
    println!("📝 Generated {} test interactions", test_interactions.len());

    // Benchmark standard manager
    println!("\n⏱️  Benchmarking standard manager...");
    let start_time = Instant::now();
    for interaction in &test_interactions {
        standard_manager.record_interaction(interaction).await?;
    }
    let standard_time = start_time.elapsed();
    let standard_memory = standard_manager.get_memory_stats().await;

    // Benchmark optimized manager
    println!("⏱️  Benchmarking optimized manager...");
    let start_time = Instant::now();
    for interaction in &test_interactions {
        optimized_manager.record_interaction(interaction).await?;
    }
    let optimized_time = start_time.elapsed();
    let optimized_stats = optimized_manager.get_comprehensive_memory_stats().await;

    // Get optimization metrics
    let optimization_metrics = optimized_manager.get_optimization_metrics().await;

    // Display results
    println!("\n📈 Performance Comparison Results");
    println!("=================================");

    println!("\n⏱️  Processing Time:");
    println!("   Standard Manager: {:?}", standard_time);
    println!("   Optimized Manager: {:?}", optimized_time);
    let time_improvement = if optimized_time < standard_time {
        let improvement = ((standard_time.as_millis() as f64 - optimized_time.as_millis() as f64)
            / standard_time.as_millis() as f64)
            * 100.0;
        format!("{:.1}% faster", improvement)
    } else {
        let overhead = ((optimized_time.as_millis() as f64 - standard_time.as_millis() as f64)
            / standard_time.as_millis() as f64)
            * 100.0;
        format!("{:.1}% slower (optimization overhead)", overhead)
    };
    println!("   Performance: {}", time_improvement);

    println!("\n💾 Memory Usage:");
    println!(
        "   Standard Manager: {}KB ({} items)",
        standard_memory.current_usage / 1024,
        standard_memory.item_count
    );
    println!(
        "   Optimized Manager: {}KB ({} items)",
        optimized_stats.memory_stats.current_usage / 1024,
        optimized_stats.memory_stats.item_count
    );

    let memory_saved = standard_memory
        .current_usage
        .saturating_sub(optimized_stats.memory_stats.current_usage);
    let memory_saving_percent = if standard_memory.current_usage > 0 {
        (memory_saved as f64 / standard_memory.current_usage as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "   Memory Saved: {}KB ({:.1}%)",
        memory_saved / 1024,
        memory_saving_percent
    );

    println!("\n🎯 String Interning Efficiency:");
    println!(
        "   Hit Rate: {:.1}%",
        optimization_metrics.string_interning_hit_rate * 100.0
    );
    println!(
        "   Memory Saved by Interning: {}KB",
        optimization_metrics.memory_saved_by_interning / 1024
    );
    println!(
        "   Total Optimization Benefit: {}KB",
        optimization_metrics.total_optimization_benefit / 1024
    );

    println!("\n📊 Storage Efficiency:");
    println!(
        "   Interactions per KB: {}",
        optimization_metrics.interactions_per_kb
    );
    println!(
        "   Unique Strings: {}",
        optimized_stats.string_pool_stats.unique_strings
    );
    println!(
        "   String Pool Cache Hits: {}",
        optimized_stats.string_pool_stats.cache_hits
    );

    // Test memory cleanup
    println!("\n🧹 Testing Memory Cleanup...");
    let cleanup_result = optimized_manager.force_memory_cleanup().await?;
    println!(
        "   Memory Before Cleanup: {}KB",
        cleanup_result.memory_before / 1024
    );
    println!(
        "   Memory After Cleanup: {}KB",
        cleanup_result.memory_after / 1024
    );
    println!("   Bytes Freed: {}KB", cleanup_result.bytes_freed / 1024);

    // Demonstrate memory profile factory
    println!("\n🏭 Testing Memory Profile Factory...");
    let low_memory_manager = AnalyticsManagerFactory::create_for_memory_profile(
        config.clone(),
        MemoryProfile::LowMemory,
    )
    .await?;
    println!("   ✅ Low memory profile manager created");

    let standard_profile_manager =
        AnalyticsManagerFactory::create_for_memory_profile(config.clone(), MemoryProfile::Standard)
            .await?;
    println!("   ✅ Standard profile manager created");

    println!("\n🎉 Memory optimization demo completed successfully!");
    println!("\n💡 Key Benefits:");
    println!("   • Reduced memory usage through string interning");
    println!("   • Bounded metadata prevents memory bloat");
    println!("   • Compressed interaction summaries for long-term storage");
    println!("   • Automatic memory pressure detection and cleanup");
    println!("   • Configurable memory profiles for different deployment scenarios");

    Ok(())
}

/// Generate test interactions with realistic patterns
fn generate_test_interactions(count: usize) -> Vec<UserInteractionEvent> {
    let users = ["user1", "user2", "user3", "user4", "user5"];
    let features = [
        "voice_feedback",
        "pronunciation_check",
        "conversation_practice",
        "speech_analysis",
    ];
    let interaction_types = [
        InteractionType::Practice,
        InteractionType::ExerciseCompleted,
        InteractionType::FeedbackViewed,
        InteractionType::SettingsChanged,
        InteractionType::AchievementEarned,
    ];

    (0..count)
        .map(|i| {
            let mut metadata = HashMap::new();
            metadata.insert("session_id".to_string(), format!("session_{}", i % 10));
            metadata.insert("difficulty".to_string(), format!("level_{}", (i % 5) + 1));

            // Sometimes add extra metadata to test bounded metadata
            if i % 3 == 0 {
                metadata.insert("extra_data".to_string(), format!("extra_value_{}", i));
                metadata.insert("context".to_string(), "detailed_context".to_string());
            }

            UserInteractionEvent {
                user_id: users[i % users.len()].to_string(), // Repeated users for string interning
                timestamp: Utc::now(),
                interaction_type: interaction_types[i % interaction_types.len()].clone(),
                feature_used: features[i % features.len()].to_string(), // Repeated features for string interning
                feedback_score: Some(0.7 + (i % 30) as f32 / 100.0), // Score between 0.7 and 0.99
                engagement_duration: std::time::Duration::from_secs(((i % 60) + 1) as u64), // 1-60 seconds
                metadata,
            }
        })
        .collect()
}
