//! Real-time audio processing modules
//!
//! This module provides comprehensive real-time audio processing capabilities including streaming
//! audio processing, low-latency operations, real-time quality monitoring, and interactive
//! processing tools for enhanced analysis and processing in speech synthesis datasets.
//!
//! # Architecture
//!
//! The real-time processing system is organized into several specialized modules:
//!
//! - [`buffer`] - Buffer management and windowing functions
//! - [`processing`] - Processing pipeline configuration and stages
//! - [`quality`] - Quality control and assessment
//! - [`error`] - Error handling and recovery strategies
//! - [`streaming`] - Streaming configuration and network protocols
//! - [`monitoring`] - Quality monitoring, alerts, and visualization
//! - [`latency`] - Latency configuration and optimization
//! - [`interactive`] - Interactive processing and user interfaces
//! - [`optimization`] - Performance optimization settings
//! - [`processor`] - Core processor trait and implementation

// Module declarations
pub mod buffer;
pub mod error;
pub mod interactive;
pub mod latency;
pub mod monitoring;
pub mod optimization;
pub mod processing;
pub mod processor;
pub mod quality;
pub mod streaming;

// Re-export key types from submodules for convenience
pub use buffer::{BufferConfig, BufferStrategy, RealTimeBuffer, WindowFunction};
pub use error::{
    ErrorHandlingConfig, ErrorRecoveryStrategy, FallbackProcessingConfig, GracefulDegradationConfig,
};
pub use interactive::{ControlInterface, FeedbackConfig, InteractiveConfig};
pub use latency::{LatencyConfig, LatencyMonitoringConfig, LatencyOptimizationConfig};
pub use monitoring::{
    Alert, AlertConfig, AlertSeverity, AlertType, QualityMonitoringConfig, VisualizationConfig,
};
pub use optimization::{CPUOptimization, IOOptimization, MemoryOptimization, RealTimeOptimization};
pub use processing::{
    ParallelProcessingConfig, ProcessingConfig, ProcessingStage, WorkDistributionStrategy,
};
pub use processor::{
    DefaultRealTimeProcessor, RealTimeConfig, RealTimeProcessor, RealTimeResult, RealTimeStatistics,
};
pub use quality::{
    AdaptiveQualityConfig, QualityAssessment, QualityControlConfig, QualityThresholds,
};
pub use streaming::{
    CompressionConfig, NetworkConfig, StreamFormat, StreamProtocol, StreamingConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_realtime_buffer_creation() {
        let buffer = RealTimeBuffer::new(1024, 44100, 2);
        assert_eq!(buffer.capacity, 1024);
        assert_eq!(buffer.sample_rate, 44100);
        assert_eq!(buffer.channels, 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_management() {
        let mut buffer = RealTimeBuffer::new(4, 44100, 1);

        // Test adding samples
        let samples = vec![0.1, 0.2, 0.3];
        buffer.push_samples(&samples).unwrap();

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.utilization(), 75.0);

        // Test capacity overflow
        let more_samples = vec![0.4, 0.5, 0.6];
        buffer.push_samples(&more_samples).unwrap();

        assert_eq!(buffer.len(), 4); // Should be at capacity

        // Clear buffer
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.utilization(), 0.0);
    }

    #[test]
    fn test_configuration_serialization() {
        let config = RealTimeConfig::default();

        // Test that the configuration can be serialized
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        // Test that it can be deserialized
        let deserialized: Result<RealTimeConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_default_realtime_processor() {
        let _processor = DefaultRealTimeProcessor::with_default_config();

        // Test that the processor can be created with default configuration
        // If we get here, creation was successful
    }

    #[test]
    fn test_realtime_statistics() {
        let stats = RealTimeStatistics {
            processing_latency: Duration::from_millis(5),
            buffer_utilization: 50.0,
            cpu_usage: 25.0,
            memory_usage: 1024,
            throughput: 44100.0,
            quality_metrics: std::collections::HashMap::new(),
            error_rate: 0.0,
            timestamp: std::time::Instant::now(),
        };

        assert_eq!(stats.processing_latency, Duration::from_millis(5));
        assert_eq!(stats.buffer_utilization, 50.0);
        assert_eq!(stats.cpu_usage, 25.0);
        assert_eq!(stats.memory_usage, 1024);
        assert_eq!(stats.throughput, 44100.0);
        assert_eq!(stats.error_rate, 0.0);
    }

    #[test]
    fn test_quality_assessment() {
        let mut assessment = QualityAssessment::new();

        assessment.add_metric("snr".to_string(), 30.0);
        assessment.add_metric("thd".to_string(), 0.1);
        assessment.calculate_overall_score();

        assert_eq!(assessment.metric_scores.len(), 2);
        assert_eq!(assessment.overall_score, 15.05); // (30.0 + 0.1) / 2
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            "test_alert".to_string(),
            AlertType::Quality,
            AlertSeverity::Medium,
            "Test alert message".to_string(),
        );

        assert_eq!(alert.id, "test_alert");
        assert_eq!(alert.message, "Test alert message");
        assert!(matches!(alert.alert_type, AlertType::Quality));
        assert!(matches!(alert.severity, AlertSeverity::Medium));
    }

    #[test]
    fn test_processing_stages() {
        use ProcessingStage::*;

        let stages = vec![
            PreProcessing {
                noise_reduction: true,
                gain_control: true,
                high_pass_filter: Some(80.0),
                low_pass_filter: Some(8000.0),
            },
            Analysis {
                spectral_analysis: true,
                pitch_detection: true,
                voice_activity_detection: true,
                quality_metrics: true,
            },
            PostProcessing {
                normalization: true,
                fade_effects: false,
                output_formatting: true,
            },
        ];

        assert_eq!(stages.len(), 3);

        // Test that stages can be pattern matched
        for stage in &stages {
            match stage {
                PreProcessing { .. } => {}
                Analysis { .. } => {}
                Enhancement { .. } => {}
                PostProcessing { .. } => {}
            }
        }
    }

    #[tokio::test]
    async fn test_realtime_processor_start_stop() {
        let mut processor = DefaultRealTimeProcessor::with_default_config();

        // Test starting the processor
        let start_result = processor.start().await;
        assert!(start_result.is_ok());

        // Test stopping the processor
        let stop_result = processor.stop().await;
        assert!(stop_result.is_ok());
    }

    #[tokio::test]
    async fn test_realtime_processor_config() {
        let mut processor = DefaultRealTimeProcessor::with_default_config();
        let config = RealTimeConfig::default();

        // Test setting configuration
        let set_result = processor.set_config(config.clone()).await;
        assert!(set_result.is_ok());

        // Test getting configuration
        let get_result = processor.get_config().await;
        assert!(get_result.is_ok());
    }

    #[tokio::test]
    async fn test_realtime_processor_statistics() {
        let processor = DefaultRealTimeProcessor::with_default_config();

        // Test getting statistics
        let stats_result = processor.get_statistics().await;
        assert!(stats_result.is_ok());

        let stats = stats_result.unwrap();
        assert_eq!(stats.processing_latency, Duration::from_millis(0));
        assert_eq!(stats.buffer_utilization, 0.0);
    }

    #[tokio::test]
    async fn test_realtime_processor_quality_assessment() {
        let processor = DefaultRealTimeProcessor::with_default_config();

        // Test getting quality assessment
        let assessment_result = processor.get_quality_assessment().await;
        assert!(assessment_result.is_ok());

        let assessment = assessment_result.unwrap();
        assert_eq!(assessment.overall_score, 0.8);
        assert_eq!(assessment.confidence, 0.9);
    }
}
