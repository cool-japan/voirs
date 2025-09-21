//! Monitoring and observability infrastructure for VoiRS Recognition
//!
//! This module provides comprehensive monitoring capabilities including distributed
//! tracing, metrics collection, performance analysis, and system observability
//! for production deployment of speech recognition services.

pub mod distributed_tracing;
pub mod metrics_collection;
pub mod performance_profiling;

pub use distributed_tracing::{
    AttributeValue, Span, SpanKind, SpanStatus, SpeechRecognitionInstrumentation, TraceContext,
    Tracer, TracingError,
};

pub use metrics_collection::{
    AlertCondition, AlertManager, AlertRule, AlertSeverity, InMemoryMetricsCollector,
    MetricMetadata, MetricValue, MetricsCollector, PerformanceMetrics, PerformanceStatistics,
    RecognitionStatus, SystemResourceMonitor, TimeSeriesAnalyzer,
};

pub use performance_profiling::{
    CpuProfilingReport, CustomEventData, CustomProfilingReport, FunctionProfile,
    GpuProfilingReport, MemoryProfilingReport, NetworkProfilingReport, PerformanceProfiler,
    ProfilingConfig, ProfilingReport, ProfilingSession,
};

/// Re-export common monitoring types and functions
pub use distributed_tracing::{
    AlwaysSampler, BatchSpanProcessor, ConsoleSpanExporter, LoggingSpanProcessor, NeverSampler,
    ProbabilitySampler, RateLimitingSampler,
};
