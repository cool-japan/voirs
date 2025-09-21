//! Automated Benchmark Updates Example
//!
//! This example demonstrates how to use the automated benchmark system to track
//! performance metrics, detect regressions, and automatically update baselines.

use std::collections::HashMap;
use voirs_evaluation::automated_benchmarks::{
    current_timestamp, get_git_commit_hash, AutomatedBenchmarkManager, BenchmarkConfig,
    BenchmarkMeasurement,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 VoiRS Automated Benchmark Updates Demo");
    println!("==========================================");

    // Configure the benchmark manager
    let config = BenchmarkConfig {
        data_directory: std::path::PathBuf::from("benchmark_data"),
        min_measurements_for_update: 3,
        max_measurement_age_days: 30,
        default_regression_threshold: 10.0, // 10% regression threshold
        default_improvement_threshold: 5.0, // 5% improvement threshold
        auto_update_baselines: true,
        generate_reports: true,
    };

    println!("📋 Configuration:");
    println!("  Data Directory: {:?}", config.data_directory);
    println!(
        "  Min Measurements for Update: {}",
        config.min_measurements_for_update
    );
    println!(
        "  Regression Threshold: {}%",
        config.default_regression_threshold
    );
    println!(
        "  Improvement Threshold: {}%",
        config.default_improvement_threshold
    );

    // Create the benchmark manager
    let mut manager = AutomatedBenchmarkManager::new(config)?;
    println!("\n✅ Benchmark manager initialized");

    // Simulate adding various benchmark measurements
    println!("\n📊 Adding benchmark measurements...");

    // Example 1: PESQ evaluation performance
    let pesq_measurements = vec![
        ("PESQ Evaluation", 45.2, "ms"),
        ("PESQ Evaluation", 44.8, "ms"),
        ("PESQ Evaluation", 43.9, "ms"),
        ("PESQ Evaluation", 42.1, "ms"), // Improvement
        ("PESQ Evaluation", 41.8, "ms"), // Further improvement
    ];

    for (name, value, unit) in pesq_measurements {
        let measurement = BenchmarkMeasurement {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: current_timestamp(),
            commit_hash: get_git_commit_hash(),
            metadata: create_metadata("benchmark_type", "quality_metric"),
        };

        match manager.add_measurement(measurement) {
            Ok(_) => println!("  ✅ Added {} measurement: {:.2} {}", name, value, unit),
            Err(e) => println!("  ❌ Failed to add {} measurement: {}", name, e),
        }

        // Add small delay to simulate time progression
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Example 2: STOI evaluation performance
    let stoi_measurements = vec![
        ("STOI Evaluation", 38.5, "ms"),
        ("STOI Evaluation", 39.1, "ms"),
        ("STOI Evaluation", 37.9, "ms"),
        ("STOI Evaluation", 36.8, "ms"),
    ];

    for (name, value, unit) in stoi_measurements {
        let measurement = BenchmarkMeasurement {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: current_timestamp(),
            commit_hash: get_git_commit_hash(),
            metadata: create_metadata("benchmark_type", "quality_metric"),
        };

        manager.add_measurement(measurement)?;
        println!("  ✅ Added {} measurement: {:.2} {}", name, value, unit);
    }

    // Example 3: Memory usage tracking
    let memory_measurements = vec![
        ("Memory Usage", 256.0, "MB"),
        ("Memory Usage", 248.0, "MB"),
        ("Memory Usage", 252.0, "MB"),
        ("Memory Usage", 245.0, "MB"), // Improvement
    ];

    for (name, value, unit) in memory_measurements {
        let measurement = BenchmarkMeasurement {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: current_timestamp(),
            commit_hash: get_git_commit_hash(),
            metadata: create_metadata("benchmark_type", "memory"),
        };

        manager.add_measurement(measurement)?;
        println!("  ✅ Added {} measurement: {:.2} {}", name, value, unit);
    }

    // Example 4: Demonstrate regression detection
    println!("\n🚨 Testing regression detection...");

    let regression_measurement = BenchmarkMeasurement {
        name: "PESQ Evaluation".to_string(),
        value: 55.0, // Much worse than baseline (should trigger regression)
        unit: "ms".to_string(),
        timestamp: current_timestamp(),
        commit_hash: get_git_commit_hash(),
        metadata: create_metadata("benchmark_type", "quality_metric"),
    };

    match manager.add_measurement(regression_measurement) {
        Ok(_) => println!("  ⚠️  Regression measurement was accepted (unexpected)"),
        Err(e) => println!("  ✅ Regression correctly detected: {}", e),
    }

    // Display current benchmark histories
    println!("\n📈 Current Benchmark Status:");

    for benchmark_name in ["PESQ Evaluation", "STOI Evaluation", "Memory Usage"] {
        if let Some(history) = manager.get_history(benchmark_name) {
            println!("\n  📊 {}:", benchmark_name);
            println!("    Current Baseline: {:.2}", history.baseline);
            println!("    Total Measurements: {}", history.measurements.len());

            if let Some(latest) = history.measurements.last() {
                let change = ((latest.value - history.baseline) / history.baseline) * 100.0;
                println!("    Latest Value: {:.2} ({:+.2}%)", latest.value, change);
            }
        }
    }

    // Generate and display performance report
    println!("\n📋 Generating Performance Report...");
    let report = manager.generate_report()?;
    println!("{}", report);

    // Export data in different formats
    println!("\n💾 Exporting benchmark data...");

    // Export as JSON
    let json_data = manager.export_data("json")?;
    std::fs::write("benchmark_export.json", json_data)?;
    println!("  ✅ JSON export saved to benchmark_export.json");

    // Export as CSV
    let csv_data = manager.export_data("csv")?;
    std::fs::write("benchmark_export.csv", csv_data)?;
    println!("  ✅ CSV export saved to benchmark_export.csv");

    // Demonstrate integration with CI/CD
    println!("\n🔄 CI/CD Integration Example:");
    println!("  The automated benchmark system can be integrated into CI/CD pipelines to:");
    println!("  - ✅ Automatically track performance metrics");
    println!("  - ✅ Detect performance regressions before deployment");
    println!("  - ✅ Update baselines when improvements are consistently observed");
    println!("  - ✅ Generate performance reports for stakeholders");
    println!("  - ✅ Export data for external analysis tools");

    // Show example CI integration commands
    println!("\n🔗 Example CI Integration Commands:");
    println!("  # Add this to your CI pipeline:");
    println!("  cargo run --example automated_benchmark_updates");
    println!("  # Or integrate directly in your tests:");
    println!("  cargo test --test performance_regression_monitoring");

    println!("\n🎉 Automated benchmark demo completed successfully!");
    println!("   Check the 'benchmark_data' directory for persistent data files.");

    Ok(())
}

/// Helper function to create metadata for measurements
fn create_metadata(key: &str, value: &str) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert(key.to_string(), value.to_string());
    metadata.insert("platform".to_string(), std::env::consts::OS.to_string());
    metadata.insert("arch".to_string(), std::env::consts::ARCH.to_string());

    // Add Rust version if available
    if let Ok(version) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if let Ok(version_str) = String::from_utf8(version.stdout) {
            metadata.insert("rust_version".to_string(), version_str.trim().to_string());
        }
    }

    metadata
}
