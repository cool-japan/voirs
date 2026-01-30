//! Performance profiling utilities for vocoder operations
//!
//! This module provides comprehensive profiling capabilities including:
//! - Real-time factor (RTF) measurement
//! - Latency tracking and analysis
//! - Throughput monitoring
//! - Performance regression detection
//! - Detailed stage-by-stage timing
//! - Production monitoring and alerting

pub mod advanced;
pub mod production;
pub mod prometheus;

pub use advanced::{
    AdvancedProfiler, LatencyBreakdown, LatencySummary, MemoryStatistics, PerformanceReport,
    ProcessingStage, ProfilerConfig, ProfilerMetrics, ProfilingSession, RegressionReport,
    RegressionType, RtfStatistics, ThroughputStatistics,
};

pub use production::{
    Alert, AlertCategory, AlertSeverity, HealthStatus, ProductionMonitor,
    ProductionMonitoringConfig, StatusSummary,
};

pub use prometheus::{
    prometheus_metrics_text, prometheus_metrics_with_subsystem, PrometheusExporter,
};
