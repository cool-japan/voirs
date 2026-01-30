//! Adaptive quality control and optimization for VoiRS SDK.
//!
//! This module provides intelligent, runtime-adaptive quality control and ML-based
//! prediction for optimal synthesis parameters.
//!
//! # Modules
//!
//! - [`controller`]: Adaptive quality controller with dynamic adjustment
//! - [`predictor`]: ML-based quality prediction using historical data
//! - [`monitoring`]: Real-time quality monitoring with trend analysis and alerting
//! - [`regression`]: Performance regression detection with statistical analysis
//!
//! # Quick Start
//!
//! ```no_run
//! use voirs_sdk::adaptive::{AdaptiveController, AdaptiveConfig, QualityTarget};
//!
//! #[tokio::main]
//! async fn main() -> voirs_sdk::Result<()> {
//!     let config = AdaptiveConfig::default()
//!         .with_target_latency(100)
//!         .with_min_quality(QualityTarget::Medium);
//!
//!     let controller = AdaptiveController::new(config);
//!     let quality = controller.get_recommended_quality().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod controller;
pub mod monitoring;
pub mod predictor;
pub mod regression;

// Re-export main types for convenience
pub use controller::{
    get_system_metrics, AdaptationStats, AdaptiveConfig, AdaptiveController, QualityTarget,
    SystemMetrics,
};
pub use monitoring::{
    AlertSeverity, AlertThreshold, DashboardData, MetricStatistics, MonitorConfig, QualityAlert,
    QualityMetricSample, QualityMonitor, TrendDirection,
};
pub use predictor::{
    PredictionInput, PredictorStats, QualityPrediction, QualityPredictor, TextComplexityAnalyzer,
    TrainingSample,
};
pub use regression::{
    MetricBaseline, MetricRegression, PerformanceBaseline, PerformanceSnapshot, RegressionConfig,
    RegressionDetector, RegressionReport,
};
