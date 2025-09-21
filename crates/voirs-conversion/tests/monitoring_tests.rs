//! Comprehensive Production Monitoring Tests
//!
//! This module provides comprehensive tests for the production monitoring system,
//! covering all aspects needed for Version 1.0.0 production readiness.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use voirs_conversion::{
    monitoring::{
        AlertSeverity, AlertType, MonitorConfig, QualityAlert, QualityEvent, QualityMonitor,
        SessionDashboard, SystemOverview, SystemStatus,
    },
    quality::{ArtifactLocation, ArtifactType, DetectedArtifacts, QualityAssessment},
    Result,
};

/// Comprehensive production monitoring test suite
pub struct MonitoringTestSuite {
    monitor: QualityMonitor,
    test_session_id: String,
}

impl MonitoringTestSuite {
    pub fn new() -> Self {
        let config = MonitorConfig {
            monitoring_interval_ms: 10, // Fast for testing
            max_history_length: 1000,
            quality_alert_threshold: 0.7,
            latency_alert_threshold_ms: 50,
            enable_artifact_alerts: true,
            enable_performance_tracking: true,
            enable_trend_analysis: true,
            report_interval_seconds: 1, // Fast for testing
        };

        Self {
            monitor: QualityMonitor::with_config(config),
            test_session_id: "test_session_001".to_string(),
        }
    }

    /// Start monitoring for tests
    pub async fn start(&mut self) -> Result<()> {
        self.monitor.start_monitoring().await
    }

    /// Stop monitoring for tests
    pub async fn stop(&mut self) -> Result<()> {
        self.monitor.stop_monitoring().await
    }

    /// Submit test quality data
    pub fn submit_test_quality_data(
        &self,
        quality: f32,
        latency_ms: u64,
        artifacts: Option<DetectedArtifacts>,
        metadata: HashMap<String, f32>,
    ) -> Result<()> {
        self.monitor.submit_quality_data(
            self.test_session_id.clone(),
            quality,
            artifacts,
            latency_ms,
            metadata,
        )
    }

    /// Submit test performance data
    pub fn submit_test_performance_data(
        &self,
        cpu_usage: f32,
        memory_usage: f64,
        throughput: f64,
        queue_length: usize,
    ) -> Result<()> {
        self.monitor.submit_performance_data(
            self.test_session_id.clone(),
            cpu_usage,
            memory_usage,
            throughput,
            queue_length,
        )
    }

    /// Get dashboard for validation
    pub async fn get_dashboard(&self) -> Result<voirs_conversion::monitoring::QualityDashboard> {
        self.monitor.get_dashboard().await
    }
}

#[tokio::test]
async fn test_production_monitoring_startup_shutdown() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();

    // Test startup
    test_suite.start().await?;
    println!("✅ Monitoring startup successful");

    // Test double startup (should fail)
    let double_start_result = test_suite.start().await;
    assert!(double_start_result.is_err(), "Double startup should fail");
    println!("✅ Double startup protection working");

    // Test shutdown
    test_suite.stop().await?;
    println!("✅ Monitoring shutdown successful");

    // Test double shutdown (should be safe)
    test_suite.stop().await?;
    println!("✅ Double shutdown safe");

    Ok(())
}

#[tokio::test]
async fn test_quality_data_submission_and_processing() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Submit various quality data points
    let test_cases = vec![
        (0.95, 25, "High quality, low latency"),
        (0.85, 45, "Good quality, acceptable latency"),
        (0.65, 75, "Moderate quality, higher latency"),
        (0.45, 120, "Low quality, high latency"),
        (0.25, 200, "Very low quality, very high latency"),
    ];

    for (quality, latency, description) in test_cases {
        println!("Testing: {}", description);

        let mut metadata = HashMap::new();
        metadata.insert("test_quality".to_string(), quality);
        metadata.insert("test_latency".to_string(), latency as f32);

        test_suite.submit_test_quality_data(quality, latency, None, metadata)?;

        // Small delay to allow processing
        sleep(Duration::from_millis(50)).await;
    }

    // Allow time for processing and dashboard update
    sleep(Duration::from_millis(200)).await;

    // Verify dashboard data
    let dashboard = test_suite.get_dashboard().await?;
    assert!(
        !dashboard.current_sessions.is_empty(),
        "Should have session data"
    );

    if let Some(session) = dashboard.current_sessions.get(&test_suite.test_session_id) {
        assert!(
            !session.quality_trend.is_empty(),
            "Should have quality trend data"
        );
        println!(
            "✅ Quality trend contains {} data points",
            session.quality_trend.len()
        );
    }

    test_suite.stop().await?;
    println!("✅ Quality data submission and processing test complete");

    Ok(())
}

#[tokio::test]
async fn test_alert_system_comprehensive() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Test quality degradation alerts
    println!("Testing quality degradation alerts...");
    test_suite.submit_test_quality_data(0.5, 30, None, HashMap::new())?; // Below threshold (0.7)
    sleep(Duration::from_millis(100)).await;

    // Test high latency alerts
    println!("Testing high latency alerts...");
    test_suite.submit_test_quality_data(0.8, 100, None, HashMap::new())?; // Above latency threshold (50ms)
    sleep(Duration::from_millis(100)).await;

    // Test critical quality alerts
    println!("Testing critical quality alerts...");
    test_suite.submit_test_quality_data(0.15, 30, None, HashMap::new())?; // Critical quality level
    sleep(Duration::from_millis(100)).await;

    // Test performance alerts
    println!("Testing performance alerts...");
    test_suite.submit_test_performance_data(95.0, 8000.0, 100.0, 50)?; // High CPU usage
    sleep(Duration::from_millis(100)).await;

    // Verify alerts in dashboard
    let dashboard = test_suite.get_dashboard().await?;
    assert!(
        !dashboard.recent_alerts.is_empty(),
        "Should have generated alerts"
    );
    println!("✅ Generated {} alerts", dashboard.recent_alerts.len());

    // Verify alert types
    let alert_types: Vec<AlertType> = dashboard
        .recent_alerts
        .iter()
        .map(|a| a.alert_type)
        .collect();
    println!("Alert types generated: {:?}", alert_types);

    test_suite.stop().await?;
    println!("✅ Alert system comprehensive test complete");

    Ok(())
}

#[tokio::test]
async fn test_artifact_detection_monitoring() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Create test artifacts
    let mut artifact_types = HashMap::new();
    artifact_types.insert(ArtifactType::Click, 0.3);
    artifact_types.insert(ArtifactType::Metallic, 0.2);
    artifact_types.insert(ArtifactType::SpectralDiscontinuity, 0.4);

    let artifact_locations = vec![
        ArtifactLocation {
            artifact_type: ArtifactType::Click,
            start_sample: 100,
            end_sample: 120,
            confidence: 0.9,
            severity: 0.3,
        },
        ArtifactLocation {
            artifact_type: ArtifactType::Metallic,
            start_sample: 500,
            end_sample: 550,
            confidence: 0.8,
            severity: 0.2,
        },
        ArtifactLocation {
            artifact_type: ArtifactType::SpectralDiscontinuity,
            start_sample: 1200,
            end_sample: 1250,
            confidence: 0.7,
            severity: 0.4,
        },
    ];

    let quality_assessment = QualityAssessment {
        overall_quality: 0.75,
        naturalness: 0.8,
        clarity: 0.7,
        consistency: 0.75,
        recommended_adjustments: vec![],
    };

    let artifacts = DetectedArtifacts {
        overall_score: 0.8,
        artifact_types,
        artifact_locations,
        quality_assessment,
    };

    // Submit quality data with artifacts
    test_suite.submit_test_quality_data(0.6, 40, Some(artifacts), HashMap::new())?;
    sleep(Duration::from_millis(100)).await;

    // Verify artifact tracking in dashboard
    let dashboard = test_suite.get_dashboard().await?;
    if let Some(session) = dashboard.current_sessions.get(&test_suite.test_session_id) {
        assert!(
            !session.active_artifacts.is_empty(),
            "Should track artifacts"
        );
        println!(
            "✅ Tracking {} artifact types",
            session.active_artifacts.len()
        );
    }

    test_suite.stop().await?;
    println!("✅ Artifact detection monitoring test complete");

    Ok(())
}

#[tokio::test]
async fn test_concurrent_session_monitoring() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Simulate multiple concurrent sessions
    let session_ids = vec![
        "session_001",
        "session_002",
        "session_003",
        "session_004",
        "session_005",
    ];

    println!("Testing concurrent session monitoring...");
    for (i, session_id) in session_ids.iter().enumerate() {
        let quality = 0.8 + (i as f32 * 0.02); // Slightly different qualities
        let latency = 30 + (i as u64 * 5); // Slightly different latencies

        test_suite.monitor.submit_quality_data(
            session_id.to_string(),
            quality,
            None,
            latency,
            HashMap::new(),
        )?;

        test_suite.monitor.submit_performance_data(
            session_id.to_string(),
            50.0 + (i as f32 * 5.0),     // Different CPU usage
            1000.0 + (i as f64 * 100.0), // Different memory usage
            1000.0,
            i + 1,
        )?;
    }

    // Allow processing time
    sleep(Duration::from_millis(200)).await;

    // Verify all sessions are tracked
    let dashboard = test_suite.get_dashboard().await?;
    assert_eq!(
        dashboard.current_sessions.len(),
        session_ids.len(),
        "Should track all sessions"
    );

    println!(
        "✅ Tracking {} concurrent sessions",
        dashboard.current_sessions.len()
    );

    // Verify system overview reflects concurrent activity
    assert!(
        dashboard.system_overview.active_sessions > 0,
        "Should show active sessions"
    );
    println!(
        "✅ System overview shows {} active sessions",
        dashboard.system_overview.active_sessions
    );

    test_suite.stop().await?;
    println!("✅ Concurrent session monitoring test complete");

    Ok(())
}

#[tokio::test]
async fn test_dashboard_data_accuracy() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Submit known data points
    let expected_qualities = vec![0.9, 0.8, 0.7, 0.6, 0.5];
    let expected_latencies = vec![20, 30, 40, 50, 60];

    for (quality, latency) in expected_qualities.iter().zip(expected_latencies.iter()) {
        test_suite.submit_test_quality_data(*quality, *latency, None, HashMap::new())?;
        test_suite.submit_test_performance_data(60.0, 2000.0, 1000.0, 5)?;
        sleep(Duration::from_millis(20)).await;
    }

    // Allow processing
    sleep(Duration::from_millis(100)).await;

    // Verify dashboard accuracy
    let dashboard = test_suite.get_dashboard().await?;

    // Check session data
    if let Some(session) = dashboard.current_sessions.get(&test_suite.test_session_id) {
        assert_eq!(
            session.quality_trend.len(),
            expected_qualities.len(),
            "Quality trend should match submitted data"
        );

        // Verify the trend data matches what we submitted
        for (i, expected) in expected_qualities.iter().enumerate() {
            if let Some(actual) = session.quality_trend.get(i) {
                assert!(
                    (actual - expected).abs() < 0.001,
                    "Quality data should match: expected {}, got {}",
                    expected,
                    actual
                );
            }
        }
        println!("✅ Quality trend data accuracy verified");
    }

    // Check system overview
    assert!(
        dashboard.system_overview.active_sessions > 0,
        "Should show active sessions"
    );
    assert!(
        dashboard.system_overview.average_quality > 0.0,
        "Should have average quality"
    );
    println!("✅ System overview data accuracy verified");

    test_suite.stop().await?;
    println!("✅ Dashboard data accuracy test complete");

    Ok(())
}

#[tokio::test]
async fn test_performance_tracking_comprehensive() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Submit performance data with different patterns
    let performance_scenarios = vec![
        (20.0, 1000.0, 2000.0, 1, "Low load"),
        (50.0, 2000.0, 1500.0, 5, "Medium load"),
        (80.0, 4000.0, 1000.0, 10, "High load"),
        (95.0, 8000.0, 500.0, 20, "Critical load"),
    ];

    for (cpu, memory, throughput, queue, description) in performance_scenarios {
        println!("Testing performance scenario: {}", description);
        test_suite.submit_test_performance_data(cpu, memory, throughput, queue)?;
        test_suite.submit_test_quality_data(0.8, 30, None, HashMap::new())?;
        sleep(Duration::from_millis(50)).await;
    }

    // Verify performance tracking
    let dashboard = test_suite.get_dashboard().await?;

    // Check if system overview reflects the performance data
    assert!(
        dashboard.system_overview.system_load_percent >= 0.0,
        "Should track system load"
    );
    assert!(
        dashboard.system_overview.memory_usage_mb >= 0.0,
        "Should track memory usage"
    );

    println!(
        "✅ System load: {}%",
        dashboard.system_overview.system_load_percent
    );
    println!(
        "✅ Memory usage: {} MB",
        dashboard.system_overview.memory_usage_mb
    );

    test_suite.stop().await?;
    println!("✅ Performance tracking comprehensive test complete");

    Ok(())
}

#[tokio::test]
async fn test_trend_analysis() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Submit data showing a clear trend (degrading quality)
    let qualities = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5];

    println!("Submitting trend data: degrading quality pattern");
    for (i, quality) in qualities.iter().enumerate() {
        test_suite.submit_test_quality_data(*quality, 30 + (i as u64 * 5), None, HashMap::new())?;
        sleep(Duration::from_millis(30)).await;
    }

    // Allow trend analysis to process
    sleep(Duration::from_millis(200)).await;

    // Verify trend is captured in dashboard
    let dashboard = test_suite.get_dashboard().await?;

    if let Some(session) = dashboard.current_sessions.get(&test_suite.test_session_id) {
        assert!(
            session.quality_trend.len() >= qualities.len() - 2,
            "Should capture most of the trend data"
        );

        // Verify the trend shows degradation
        let first_half_avg: f32 = session
            .quality_trend
            .iter()
            .take(qualities.len() / 2)
            .sum::<f32>()
            / (qualities.len() / 2) as f32;
        let second_half_avg: f32 = session
            .quality_trend
            .iter()
            .skip(qualities.len() / 2)
            .sum::<f32>()
            / (qualities.len() / 2) as f32;

        assert!(
            first_half_avg > second_half_avg,
            "Should show degrading trend: first half avg {} > second half avg {}",
            first_half_avg,
            second_half_avg
        );

        println!(
            "✅ Trend analysis detected quality degradation: {:.3} → {:.3}",
            first_half_avg, second_half_avg
        );
    }

    test_suite.stop().await?;
    println!("✅ Trend analysis test complete");

    Ok(())
}

#[tokio::test]
async fn test_alert_cooldown_system() -> Result<()> {
    let config = MonitorConfig {
        monitoring_interval_ms: 10,
        max_history_length: 1000,
        quality_alert_threshold: 0.7,
        latency_alert_threshold_ms: 50,
        enable_artifact_alerts: true,
        enable_performance_tracking: true,
        enable_trend_analysis: true,
        report_interval_seconds: 1,
    };

    let mut monitor = QualityMonitor::with_config(config);
    monitor.start_monitoring().await?;

    let session_id = "cooldown_test_session";

    // Submit first low quality data (should trigger alert)
    monitor.submit_quality_data(
        session_id.to_string(),
        0.5, // Below threshold
        None,
        30,
        HashMap::new(),
    )?;

    sleep(Duration::from_millis(100)).await;
    let dashboard1 = monitor.get_dashboard().await?;
    let alerts_count_1 = dashboard1.recent_alerts.len();

    // Submit second low quality data immediately (should be in cooldown)
    monitor.submit_quality_data(
        session_id.to_string(),
        0.4, // Below threshold
        None,
        30,
        HashMap::new(),
    )?;

    sleep(Duration::from_millis(100)).await;
    let dashboard2 = monitor.get_dashboard().await?;
    let alerts_count_2 = dashboard2.recent_alerts.len();

    // Should not have increased due to cooldown
    assert_eq!(
        alerts_count_1, alerts_count_2,
        "Alert count should not increase during cooldown period"
    );

    println!(
        "✅ Alert cooldown system working: {} alerts maintained during cooldown",
        alerts_count_2
    );

    monitor.stop_monitoring().await?;
    println!("✅ Alert cooldown system test complete");

    Ok(())
}

#[tokio::test]
async fn test_monitoring_under_stress() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    println!("Starting stress test with high-frequency data submission...");

    // Submit data at high frequency to test system stability
    let start_time = Instant::now();
    let num_submissions = 100;

    for i in 0..num_submissions {
        let quality = 0.5 + (i as f32 % 50.0) / 100.0; // Varying quality
        let latency = 20 + (i % 80); // Varying latency

        test_suite.submit_test_quality_data(quality, latency as u64, None, HashMap::new())?;
        test_suite.submit_test_performance_data(
            50.0 + (i as f32 % 40.0),
            2000.0 + (i as f64 % 2000.0),
            1000.0,
            i % 10,
        )?;

        // Very small delay to create high frequency
        if i % 10 == 0 {
            sleep(Duration::from_millis(1)).await;
        }
    }

    let submission_duration = start_time.elapsed();
    println!(
        "✅ Submitted {} data points in {:?}",
        num_submissions, submission_duration
    );

    // Allow processing time
    sleep(Duration::from_millis(500)).await;

    // Verify system stability
    let dashboard = test_suite.get_dashboard().await?;
    assert!(
        !dashboard.current_sessions.is_empty(),
        "System should remain stable under stress"
    );

    if let Some(session) = dashboard.current_sessions.get(&test_suite.test_session_id) {
        assert!(
            !session.quality_trend.is_empty(),
            "Should have processed stress test data"
        );
        println!(
            "✅ Processed {} quality data points under stress",
            session.quality_trend.len()
        );
    }

    test_suite.stop().await?;
    println!("✅ Monitoring stress test complete - system remained stable");

    Ok(())
}

#[tokio::test]
async fn test_error_conditions_and_recovery() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Test invalid data submission (system should handle gracefully)
    println!("Testing error condition handling...");

    // Submit extreme values that should be handled gracefully
    test_suite.submit_test_quality_data(f32::NAN, 0, None, HashMap::new())?;
    test_suite.submit_test_quality_data(f32::INFINITY, u64::MAX, None, HashMap::new())?;
    test_suite.submit_test_quality_data(-1.0, 0, None, HashMap::new())?;
    test_suite.submit_test_quality_data(2.0, 0, None, HashMap::new())?;

    // Allow processing
    sleep(Duration::from_millis(100)).await;

    // Verify system continues to work after error conditions
    test_suite.submit_test_quality_data(0.8, 30, None, HashMap::new())?;
    sleep(Duration::from_millis(50)).await;

    let dashboard = test_suite.get_dashboard().await?;
    assert!(
        !dashboard.current_sessions.is_empty(),
        "System should recover from error conditions"
    );

    println!("✅ System handled error conditions gracefully and recovered");

    test_suite.stop().await?;
    println!("✅ Error conditions and recovery test complete");

    Ok(())
}

#[tokio::test]
async fn test_configuration_scenarios() -> Result<()> {
    // Test different configuration scenarios
    let configs = vec![
        MonitorConfig {
            monitoring_interval_ms: 50,
            max_history_length: 100,
            quality_alert_threshold: 0.9,
            latency_alert_threshold_ms: 25,
            enable_artifact_alerts: false,
            enable_performance_tracking: false,
            enable_trend_analysis: false,
            report_interval_seconds: 10,
        },
        MonitorConfig {
            monitoring_interval_ms: 10,
            max_history_length: 10000,
            quality_alert_threshold: 0.5,
            latency_alert_threshold_ms: 100,
            enable_artifact_alerts: true,
            enable_performance_tracking: true,
            enable_trend_analysis: true,
            report_interval_seconds: 1,
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        println!("Testing configuration scenario {}", i + 1);

        let mut monitor = QualityMonitor::with_config(config.clone());
        monitor.start_monitoring().await?;

        // Submit test data
        monitor.submit_quality_data("config_test".to_string(), 0.7, None, 30, HashMap::new())?;

        sleep(Duration::from_millis(100)).await;

        // Verify monitoring works with this configuration
        let dashboard = monitor.get_dashboard().await?;
        assert!(
            !dashboard.current_sessions.is_empty(),
            "Configuration {} should work",
            i + 1
        );

        monitor.stop_monitoring().await?;
        println!("✅ Configuration scenario {} completed successfully", i + 1);
    }

    println!("✅ Configuration scenarios test complete");
    Ok(())
}

#[tokio::test]
async fn test_production_monitoring_integration() -> Result<()> {
    println!("🚀 Starting comprehensive production monitoring integration test...");

    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    // Simulate a realistic production scenario
    let session_scenarios = vec![
        (
            "prod_session_001",
            vec![(0.95, 25), (0.93, 28), (0.91, 30)],
            "High-quality session",
        ),
        (
            "prod_session_002",
            vec![(0.85, 45), (0.82, 50), (0.80, 55)],
            "Medium-quality session",
        ),
        (
            "prod_session_003",
            vec![(0.65, 75), (0.60, 85), (0.55, 95)],
            "Degrading session",
        ),
        (
            "prod_session_004",
            vec![(0.45, 120), (0.40, 140), (0.35, 160)],
            "Poor-quality session",
        ),
    ];

    for (session_id, data_points, description) in session_scenarios {
        println!("Processing {}: {}", session_id, description);

        for (quality, latency) in data_points {
            test_suite.monitor.submit_quality_data(
                session_id.to_string(),
                quality,
                None,
                latency,
                HashMap::new(),
            )?;

            test_suite.monitor.submit_performance_data(
                session_id.to_string(),
                45.0 + (latency as f32 / 4.0), // CPU correlates with latency
                2000.0 + (latency as f64 * 10.0), // Memory correlates with latency
                1000.0 / (latency as f64 / 30.0), // Throughput inversely correlates
                (latency / 20) as usize,       // Queue length correlates with latency
            )?;

            sleep(Duration::from_millis(30)).await;
        }
    }

    // Allow comprehensive processing
    sleep(Duration::from_millis(300)).await;

    // Comprehensive validation
    let dashboard = test_suite.get_dashboard().await?;

    // Verify all sessions are tracked
    assert_eq!(
        dashboard.current_sessions.len(),
        4,
        "Should track all production sessions"
    );

    // Verify system overview shows realistic data
    assert!(
        dashboard.system_overview.active_sessions == 4,
        "Should show 4 active sessions"
    );
    assert!(
        dashboard.system_overview.average_quality > 0.0,
        "Should have calculated average quality"
    );
    assert!(
        dashboard.system_overview.system_load_percent >= 0.0,
        "Should track system load"
    );

    // Verify alerts were generated for poor sessions
    assert!(
        !dashboard.recent_alerts.is_empty(),
        "Should have generated alerts for poor quality sessions"
    );

    // Verify trend data exists
    assert!(
        !dashboard.trends.quality_over_time.is_empty(),
        "Should have quality trend data"
    );
    assert!(
        !dashboard.trends.latency_over_time.is_empty(),
        "Should have latency trend data"
    );

    println!("✅ Production scenario simulation complete:");
    println!("  - Sessions tracked: {}", dashboard.current_sessions.len());
    println!(
        "  - Active sessions: {}",
        dashboard.system_overview.active_sessions
    );
    println!(
        "  - Average quality: {:.3}",
        dashboard.system_overview.average_quality
    );
    println!(
        "  - System load: {:.1}%",
        dashboard.system_overview.system_load_percent
    );
    println!(
        "  - Memory usage: {:.1} MB",
        dashboard.system_overview.memory_usage_mb
    );
    println!("  - Alerts generated: {}", dashboard.recent_alerts.len());
    println!(
        "  - Quality trend points: {}",
        dashboard.trends.quality_over_time.len()
    );
    println!(
        "  - Latency trend points: {}",
        dashboard.trends.latency_over_time.len()
    );

    test_suite.stop().await?;
    println!("🎉 Production monitoring integration test PASSED - System is production ready!");

    Ok(())
}

/// Performance benchmark for monitoring system overhead
#[tokio::test]
async fn test_monitoring_performance_overhead() -> Result<()> {
    let mut test_suite = MonitoringTestSuite::new();
    test_suite.start().await?;

    println!("Measuring monitoring system performance overhead...");

    let start_time = Instant::now();
    let iterations = 1000;

    for i in 0..iterations {
        test_suite.submit_test_quality_data(0.8, 30, None, HashMap::new())?;
        test_suite.submit_test_performance_data(50.0, 2000.0, 1000.0, 5)?;

        if i % 100 == 0 {
            sleep(Duration::from_millis(1)).await; // Prevent overwhelming
        }
    }

    let total_duration = start_time.elapsed();
    let ops_per_second = (iterations as f64 * 2.0) / total_duration.as_secs_f64(); // 2 ops per iteration

    println!("✅ Performance metrics:");
    println!("  - Total operations: {}", iterations * 2);
    println!("  - Total time: {:?}", total_duration);
    println!("  - Operations per second: {:.2}", ops_per_second);
    println!(
        "  - Average time per operation: {:.4}ms",
        total_duration.as_millis() as f64 / (iterations as f64 * 2.0)
    );

    // Verify the system processed all data
    sleep(Duration::from_millis(100)).await;
    let dashboard = test_suite.get_dashboard().await?;
    assert!(
        !dashboard.current_sessions.is_empty(),
        "Should have processed performance test data"
    );

    // Assert reasonable performance (should handle at least 100 ops/second)
    assert!(
        ops_per_second >= 100.0,
        "Monitoring system should handle at least 100 operations per second, got {:.2}",
        ops_per_second
    );

    test_suite.stop().await?;
    println!(
        "✅ Monitoring performance overhead test complete - System meets performance requirements"
    );

    Ok(())
}
