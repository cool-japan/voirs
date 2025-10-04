//! Distributed Tracing Support for VoiRS Recognition
//!
//! This module provides comprehensive distributed tracing capabilities using OpenTelemetry
//! standards for monitoring speech recognition performance across distributed systems.
//! Includes span management, context propagation, custom instrumentation, and sampling
//! strategies for production observability.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Trace context for distributed tracing
#[derive(Debug, Clone)]
/// Trace Context
pub struct TraceContext {
    /// Trace ID - identifies the entire request across services
    pub trace_id: TraceId,
    /// Span ID - identifies the current operation
    pub span_id: SpanId,
    /// Parent span ID if this is a child span
    pub parent_span_id: Option<SpanId>,
    /// Trace flags (sampling, debug, etc.)
    pub trace_flags: TraceFlags,
    /// Trace state for vendor-specific information
    pub trace_state: TraceState,
}

/// Unique trace identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Trace Id
pub struct TraceId(pub [u8; 16]);

/// Unique span identifier  
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Span Id
pub struct SpanId(pub [u8; 8]);

/// Trace flags for controlling tracing behavior
#[derive(Debug, Clone)]
/// Trace Flags
pub struct TraceFlags {
    /// Whether this trace is sampled
    pub sampled: bool,
    /// Debug flag for detailed tracing
    pub debug: bool,
    /// Random trace flag
    pub random: bool,
}

/// Vendor-specific trace state
#[derive(Debug, Clone)]
/// Trace State
pub struct TraceState {
    /// Key-value pairs for vendor state
    pub entries: HashMap<String, String>,
}

impl TraceContext {
    /// Create a new root trace context
    pub fn new() -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            trace_flags: TraceFlags::default(),
            trace_state: TraceState::default(),
        }
    }

    /// Create a child span from this context
    pub fn create_child_span(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: SpanId::new(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: self.trace_flags.clone(),
            trace_state: self.trace_state.clone(),
        }
    }

    /// Convert to W3C trace parent header format
    pub fn to_trace_parent(&self) -> String {
        let flags = if self.trace_flags.sampled { 1 } else { 0 };
        format!(
            "00-{}-{}-{:02x}",
            hex::encode(&self.trace_id.0),
            hex::encode(&self.span_id.0),
            flags
        )
    }

    /// Parse from W3C trace parent header
    pub fn from_trace_parent(trace_parent: &str) -> Result<Self, TracingError> {
        let parts: Vec<&str> = trace_parent.split('-').collect();
        if parts.len() != 4 {
            return Err(TracingError::InvalidTraceParent);
        }

        let version = parts[0];
        if version != "00" {
            return Err(TracingError::UnsupportedVersion);
        }

        let trace_id_hex = parts[1];
        let span_id_hex = parts[2];
        let flags_hex = parts[3];

        let trace_id_bytes = hex::decode(trace_id_hex).map_err(|_| TracingError::InvalidTraceId)?;
        let span_id_bytes = hex::decode(span_id_hex).map_err(|_| TracingError::InvalidSpanId)?;
        let flags = u8::from_str_radix(flags_hex, 16).map_err(|_| TracingError::InvalidFlags)?;

        if trace_id_bytes.len() != 16 || span_id_bytes.len() != 8 {
            return Err(TracingError::InvalidLength);
        }

        let mut trace_id = [0u8; 16];
        let mut span_id = [0u8; 8];
        trace_id.copy_from_slice(&trace_id_bytes);
        span_id.copy_from_slice(&span_id_bytes);

        Ok(Self {
            trace_id: TraceId(trace_id),
            span_id: SpanId(span_id),
            parent_span_id: None,
            trace_flags: TraceFlags {
                sampled: (flags & 1) != 0,
                debug: (flags & 2) != 0,
                random: (flags & 4) != 0,
            },
            trace_state: TraceState::default(),
        })
    }
}

impl TraceId {
    /// Generate a new random trace ID
    pub fn new() -> Self {
        let uuid = Uuid::new_v4();
        Self(uuid.into_bytes())
    }
}

impl SpanId {
    /// Generate a new random span ID
    pub fn new() -> Self {
        let bytes = scirs2_core::random::random::<u64>().to_be_bytes();
        Self(bytes)
    }
}

impl Default for TraceFlags {
    fn default() -> Self {
        Self {
            sampled: true,
            debug: false,
            random: false,
        }
    }
}

impl Default for TraceState {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0))
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0))
    }
}

/// Span represents a single operation within a trace
#[derive(Debug, Clone)]
/// Span
pub struct Span {
    /// Trace context
    pub context: TraceContext,
    /// Span name
    pub name: String,
    /// Span kind (client, server, internal, etc.)
    pub kind: SpanKind,
    /// Start time
    pub start_time: SystemTime,
    /// End time (None if span is still active)
    pub end_time: Option<SystemTime>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Span events
    pub events: Vec<SpanEvent>,
    /// Span links to other spans
    pub links: Vec<SpanLink>,
}

/// Type of span operation
#[derive(Debug, Clone, PartialEq)]
/// Span Kind
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Outgoing request
    Client,
    /// Incoming request
    Server,
    /// Message producer
    Producer,
    /// Message consumer
    Consumer,
}

/// Status of a span
#[derive(Debug, Clone, PartialEq)]
/// Span Status
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span encountered an error
    Error,
    /// Status not set
    Unset,
}

/// Attribute value types
#[derive(Debug, Clone)]
/// Attribute Value
pub enum AttributeValue {
    /// String( string)
    String(String),
    /// Int(i64)
    Int(i64),
    /// Float(f64)
    Float(f64),
    /// Bool(bool)
    Bool(bool),
    /// String array( vec< string>)
    StringArray(Vec<String>),
    /// Int array( vec<i64>)
    IntArray(Vec<i64>),
    /// Float array( vec<f64>)
    FloatArray(Vec<f64>),
    /// Bool array( vec<bool>)
    BoolArray(Vec<bool>),
}

/// Span event for timestamped annotations
#[derive(Debug, Clone)]
/// Span Event
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Link to another span
#[derive(Debug, Clone)]
/// Span Link
pub struct SpanLink {
    /// Linked span context
    pub context: TraceContext,
    /// Link attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl Span {
    /// Create a new span
    pub fn new(name: String, kind: SpanKind, context: TraceContext) -> Self {
        Self {
            context,
            name,
            kind,
            start_time: SystemTime::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Set span attribute
    pub fn set_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }

    /// Add span event
    pub fn add_event(&mut self, name: String, attributes: HashMap<String, AttributeValue>) {
        self.events.push(SpanEvent {
            name,
            timestamp: SystemTime::now(),
            attributes,
        });
    }

    /// Add span link
    pub fn add_link(&mut self, context: TraceContext, attributes: HashMap<String, AttributeValue>) {
        self.links.push(SpanLink {
            context,
            attributes,
        });
    }

    /// Set span status
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// End the span
    pub fn end(&mut self) {
        if self.end_time.is_none() {
            self.end_time = Some(SystemTime::now());
        }
    }

    /// Get span duration
    pub fn duration(&self) -> Option<Duration> {
        self.end_time
            .map(|end| end.duration_since(self.start_time).unwrap_or_default())
    }

    /// Check if span is active
    pub fn is_active(&self) -> bool {
        self.end_time.is_none()
    }
}

/// Tracer for creating and managing spans
#[derive(Debug)]
/// Tracer
pub struct Tracer {
    /// Service name
    service_name: String,
    /// Service version
    service_version: String,
    /// Active spans
    active_spans: Arc<Mutex<HashMap<SpanId, Span>>>,
    /// Span processor
    processor: Arc<dyn SpanProcessor>,
    /// Sampling strategy
    sampler: Arc<dyn Sampler>,
}

/// Span processor interface
pub trait SpanProcessor: Send + Sync + fmt::Debug {
    /// Process a span when it starts
    fn on_start(&self, span: &Span);

    /// Process a span when it ends
    fn on_end(&self, span: &Span);

    /// Shutdown the processor
    fn shutdown(&self);
}

/// Sampling strategy interface
pub trait Sampler: Send + Sync + fmt::Debug {
    /// Decide whether to sample a span
    fn should_sample(&self, context: &TraceContext, name: &str, kind: SpanKind) -> SamplingResult;
}

/// Result of sampling decision
#[derive(Debug, Clone)]
/// Sampling Result
pub struct SamplingResult {
    /// Whether to sample
    pub decision: SamplingDecision,
    /// Additional attributes to add
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq)]
/// Sampling Decision
pub enum SamplingDecision {
    /// Record and sample
    RecordAndSample,
    /// Record but don't sample
    RecordOnly,
    /// Drop the span
    Drop,
}

impl Tracer {
    /// Create a new tracer
    pub fn new(
        service_name: String,
        service_version: String,
        processor: Arc<dyn SpanProcessor>,
        sampler: Arc<dyn Sampler>,
    ) -> Self {
        Self {
            service_name,
            service_version,
            active_spans: Arc::new(Mutex::new(HashMap::new())),
            processor,
            sampler,
        }
    }

    /// Start a new span
    pub fn start_span(
        &self,
        name: String,
        kind: SpanKind,
        parent_context: Option<TraceContext>,
    ) -> Span {
        let context = match parent_context {
            Some(parent) => parent.create_child_span(),
            None => TraceContext::new(),
        };

        // Apply sampling
        let sampling_result = self.sampler.should_sample(&context, &name, kind.clone());
        let mut span_context = context;
        span_context.trace_flags.sampled =
            sampling_result.decision == SamplingDecision::RecordAndSample;

        let mut span = Span::new(name, kind, span_context);

        // Add service attributes
        span.set_attribute(
            "service.name".to_string(),
            AttributeValue::String(self.service_name.clone()),
        );
        span.set_attribute(
            "service.version".to_string(),
            AttributeValue::String(self.service_version.clone()),
        );

        // Add sampling attributes
        for (key, value) in sampling_result.attributes {
            span.set_attribute(key, value);
        }

        // Notify processor
        self.processor.on_start(&span);

        // Store active span
        if sampling_result.decision != SamplingDecision::Drop {
            self.active_spans
                .lock()
                .unwrap()
                .insert(span.context.span_id.clone(), span.clone());
        }

        span
    }

    /// End a span
    pub fn end_span(&self, mut span: Span) {
        span.end();

        // Remove from active spans
        self.active_spans
            .lock()
            .unwrap()
            .remove(&span.context.span_id);

        // Notify processor
        self.processor.on_end(&span);
    }

    /// Get active span by ID
    pub fn get_active_span(&self, span_id: &SpanId) -> Option<Span> {
        self.active_spans.lock().unwrap().get(span_id).cloned()
    }

    /// Get all active spans
    pub fn get_active_spans(&self) -> Vec<Span> {
        self.active_spans
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }
}

/// Simple span processor that logs spans
#[derive(Debug)]
/// Logging Span Processor
pub struct LoggingSpanProcessor;

impl SpanProcessor for LoggingSpanProcessor {
    fn on_start(&self, span: &Span) {
        println!("Span started: {} ({})", span.name, span.context.span_id);
    }

    fn on_end(&self, span: &Span) {
        let duration = span.duration().unwrap_or_default();
        println!(
            "Span ended: {} ({}) - Duration: {}ms - Status: {:?}",
            span.name,
            span.context.span_id,
            duration.as_millis(),
            span.status
        );
    }

    fn shutdown(&self) {
        println!("Span processor shut down");
    }
}

/// Batch span processor for efficient export
#[derive(Debug)]
/// Batch Span Processor
pub struct BatchSpanProcessor {
    /// Batch of spans waiting to be exported
    batch: Arc<Mutex<Vec<Span>>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Batch timeout
    batch_timeout: Duration,
    /// Span exporter
    exporter: Arc<dyn SpanExporter>,
}

/// Span exporter interface
pub trait SpanExporter: Send + Sync + fmt::Debug {
    /// Export a batch of spans
    fn export(&self, spans: Vec<Span>) -> Result<(), ExportError>;

    /// Shutdown the exporter
    fn shutdown(&self);
}

impl BatchSpanProcessor {
    /// Create a new batch processor
    pub fn new(
        max_batch_size: usize,
        batch_timeout: Duration,
        exporter: Arc<dyn SpanExporter>,
    ) -> Self {
        Self {
            batch: Arc::new(Mutex::new(Vec::new())),
            max_batch_size,
            batch_timeout,
            exporter,
        }
    }

    /// Force export current batch
    pub fn force_export(&self) {
        let mut batch = self.batch.lock().unwrap();
        if !batch.is_empty() {
            let spans = batch.drain(..).collect();
            drop(batch);
            let _ = self.exporter.export(spans);
        }
    }
}

impl SpanProcessor for BatchSpanProcessor {
    fn on_start(&self, _span: &Span) {
        // No action needed on start
    }

    fn on_end(&self, span: &Span) {
        let mut batch = self.batch.lock().unwrap();
        batch.push(span.clone());

        // Export if batch is full
        if batch.len() >= self.max_batch_size {
            let spans = batch.drain(..).collect();
            drop(batch);
            let _ = self.exporter.export(spans);
        }
    }

    fn shutdown(&self) {
        self.force_export();
        self.exporter.shutdown();
    }
}

/// Console span exporter for development
#[derive(Debug)]
/// Console Span Exporter
pub struct ConsoleSpanExporter;

impl SpanExporter for ConsoleSpanExporter {
    fn export(&self, spans: Vec<Span>) -> Result<(), ExportError> {
        for span in spans {
            println!(
                "Exported span: {}",
                serde_json::to_string_pretty(&SpanJson::from(span)).unwrap()
            );
        }
        Ok(())
    }

    fn shutdown(&self) {
        println!("Console exporter shut down");
    }
}

/// JSON representation of a span for export
#[derive(Debug, serde::Serialize)]
struct SpanJson {
    trace_id: String,
    span_id: String,
    parent_span_id: Option<String>,
    name: String,
    kind: String,
    start_time: u64,
    end_time: Option<u64>,
    duration_ms: Option<u64>,
    status: String,
    attributes: HashMap<String, serde_json::Value>,
    events: Vec<SpanEventJson>,
}

#[derive(Debug, serde::Serialize)]
struct SpanEventJson {
    name: String,
    timestamp: u64,
    attributes: HashMap<String, serde_json::Value>,
}

impl From<Span> for SpanJson {
    fn from(span: Span) -> Self {
        let start_timestamp = span
            .start_time
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let end_timestamp = span
            .end_time
            .map(|end| end.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64);
        let duration_ms = span.duration().map(|d| d.as_millis() as u64);

        Self {
            trace_id: span.context.trace_id.to_string(),
            span_id: span.context.span_id.to_string(),
            parent_span_id: span.context.parent_span_id.map(|id| id.to_string()),
            name: span.name,
            kind: format!("{:?}", span.kind),
            start_time: start_timestamp,
            end_time: end_timestamp,
            duration_ms,
            status: format!("{:?}", span.status),
            attributes: span
                .attributes
                .into_iter()
                .map(|(k, v)| (k, attribute_to_json(v)))
                .collect(),
            events: span
                .events
                .into_iter()
                .map(|e| SpanEventJson {
                    name: e.name,
                    timestamp: e.timestamp.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
                    attributes: e
                        .attributes
                        .into_iter()
                        .map(|(k, v)| (k, attribute_to_json(v)))
                        .collect(),
                })
                .collect(),
        }
    }
}

fn attribute_to_json(attr: AttributeValue) -> serde_json::Value {
    match attr {
        AttributeValue::String(s) => serde_json::Value::String(s),
        AttributeValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(i)),
        AttributeValue::Float(f) => serde_json::Value::Number(
            serde_json::Number::from_f64(f).unwrap_or_else(|| serde_json::Number::from(0)),
        ),
        AttributeValue::Bool(b) => serde_json::Value::Bool(b),
        AttributeValue::StringArray(arr) => {
            serde_json::Value::Array(arr.into_iter().map(serde_json::Value::String).collect())
        }
        AttributeValue::IntArray(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(|i| serde_json::Value::Number(serde_json::Number::from(i)))
                .collect(),
        ),
        AttributeValue::FloatArray(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(|f| {
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(f)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                })
                .collect(),
        ),
        AttributeValue::BoolArray(arr) => {
            serde_json::Value::Array(arr.into_iter().map(serde_json::Value::Bool).collect())
        }
    }
}

/// Always sample strategy
#[derive(Debug)]
/// Always Sampler
pub struct AlwaysSampler;

impl Sampler for AlwaysSampler {
    fn should_sample(
        &self,
        _context: &TraceContext,
        _name: &str,
        _kind: SpanKind,
    ) -> SamplingResult {
        SamplingResult {
            decision: SamplingDecision::RecordAndSample,
            attributes: HashMap::new(),
        }
    }
}

/// Never sample strategy
#[derive(Debug)]
/// Never Sampler
pub struct NeverSampler;

impl Sampler for NeverSampler {
    fn should_sample(
        &self,
        _context: &TraceContext,
        _name: &str,
        _kind: SpanKind,
    ) -> SamplingResult {
        SamplingResult {
            decision: SamplingDecision::Drop,
            attributes: HashMap::new(),
        }
    }
}

/// Probability-based sampler
#[derive(Debug)]
/// Probability Sampler
pub struct ProbabilitySampler {
    /// Sampling probability (0.0 - 1.0)
    probability: f64,
}

impl ProbabilitySampler {
    /// Create a new probability sampler
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Sampler for ProbabilitySampler {
    fn should_sample(
        &self,
        _context: &TraceContext,
        _name: &str,
        _kind: SpanKind,
    ) -> SamplingResult {
        let random_value: f64 = scirs2_core::random::random();
        let decision = if random_value < self.probability {
            SamplingDecision::RecordAndSample
        } else {
            SamplingDecision::Drop
        };

        let mut attributes = HashMap::new();
        attributes.insert(
            "sampling.probability".to_string(),
            AttributeValue::Float(self.probability),
        );

        SamplingResult {
            decision,
            attributes,
        }
    }
}

/// Rate limiting sampler
#[derive(Debug)]
/// Rate Limiting Sampler
pub struct RateLimitingSampler {
    /// Maximum spans per second
    max_spans_per_second: f64,
    /// Last sampling time
    last_sample_time: Arc<Mutex<Instant>>,
    /// Current span count in window
    current_count: Arc<Mutex<usize>>,
}

impl RateLimitingSampler {
    /// Create a new rate limiting sampler
    pub fn new(max_spans_per_second: f64) -> Self {
        Self {
            max_spans_per_second,
            last_sample_time: Arc::new(Mutex::new(Instant::now())),
            current_count: Arc::new(Mutex::new(0)),
        }
    }
}

impl Sampler for RateLimitingSampler {
    fn should_sample(
        &self,
        _context: &TraceContext,
        _name: &str,
        _kind: SpanKind,
    ) -> SamplingResult {
        let now = Instant::now();
        let mut last_time = self.last_sample_time.lock().unwrap();
        let mut count = self.current_count.lock().unwrap();

        // Reset count if a second has passed
        if now.duration_since(*last_time) >= Duration::from_secs(1) {
            *count = 0;
            *last_time = now;
        }

        let decision = if (*count as f64) < self.max_spans_per_second {
            *count += 1;
            SamplingDecision::RecordAndSample
        } else {
            SamplingDecision::Drop
        };

        let mut attributes = HashMap::new();
        attributes.insert(
            "sampling.rate_limit".to_string(),
            AttributeValue::Float(self.max_spans_per_second),
        );

        SamplingResult {
            decision,
            attributes,
        }
    }
}

/// Instrumentation for speech recognition operations
pub struct SpeechRecognitionInstrumentation {
    /// Main tracer
    tracer: Tracer,
}

impl SpeechRecognitionInstrumentation {
    /// Create new instrumentation
    pub fn new(service_name: String, service_version: String) -> Self {
        let processor = Arc::new(BatchSpanProcessor::new(
            32,
            Duration::from_secs(5),
            Arc::new(ConsoleSpanExporter),
        ));

        let sampler = Arc::new(ProbabilitySampler::new(0.1)); // 10% sampling

        let tracer = Tracer::new(service_name, service_version, processor, sampler);

        Self { tracer }
    }

    /// Instrument audio preprocessing
    pub fn instrument_preprocessing(&self, parent_context: Option<TraceContext>) -> Span {
        let mut span = self.tracer.start_span(
            "audio.preprocessing".to_string(),
            SpanKind::Internal,
            parent_context,
        );

        span.set_attribute(
            "component".to_string(),
            AttributeValue::String("audio_preprocessing".to_string()),
        );
        span
    }

    /// Instrument feature extraction
    pub fn instrument_feature_extraction(&self, parent_context: Option<TraceContext>) -> Span {
        let mut span = self.tracer.start_span(
            "audio.feature_extraction".to_string(),
            SpanKind::Internal,
            parent_context,
        );

        span.set_attribute(
            "component".to_string(),
            AttributeValue::String("feature_extraction".to_string()),
        );
        span
    }

    /// Instrument model inference
    pub fn instrument_inference(
        &self,
        model_name: &str,
        parent_context: Option<TraceContext>,
    ) -> Span {
        let mut span = self.tracer.start_span(
            "model.inference".to_string(),
            SpanKind::Internal,
            parent_context,
        );

        span.set_attribute(
            "component".to_string(),
            AttributeValue::String("model_inference".to_string()),
        );
        span.set_attribute(
            "model.name".to_string(),
            AttributeValue::String(model_name.to_string()),
        );
        span
    }

    /// Instrument post-processing
    pub fn instrument_postprocessing(&self, parent_context: Option<TraceContext>) -> Span {
        let mut span = self.tracer.start_span(
            "audio.postprocessing".to_string(),
            SpanKind::Internal,
            parent_context,
        );

        span.set_attribute(
            "component".to_string(),
            AttributeValue::String("postprocessing".to_string()),
        );
        span
    }

    /// End a span
    pub fn end_span(&self, span: Span) {
        self.tracer.end_span(span);
    }
}

/// Tracing errors
#[derive(Debug, thiserror::Error)]
/// Tracing Error
pub enum TracingError {
    #[error("Invalid trace parent format")]
    /// Invalid trace parent
    InvalidTraceParent,
    #[error("Unsupported trace version")]
    /// Unsupported version
    UnsupportedVersion,
    #[error("Invalid trace ID")]
    /// Invalid trace id
    InvalidTraceId,
    #[error("Invalid span ID")]
    /// Invalid span id
    InvalidSpanId,
    #[error("Invalid flags")]
    /// Invalid flags
    InvalidFlags,
    #[error("Invalid length")]
    /// Invalid length
    InvalidLength,
}

/// Export errors
#[derive(Debug, thiserror::Error)]
/// Export Error
pub enum ExportError {
    #[error("Export failed: {message}")]
    /// Export failed
    ExportFailed {
        /// Error message
        message: String,
    },
    #[error("Serialization failed")]
    /// Serialization failed
    SerializationFailed,
    #[error("Network error")]
    /// Network error
    NetworkError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(!context.trace_id.0.iter().all(|&x| x == 0));
        assert!(!context.span_id.0.iter().all(|&x| x == 0));
        assert!(context.parent_span_id.is_none());
    }

    #[test]
    fn test_child_span_creation() {
        let parent = TraceContext::new();
        let child = parent.create_child_span();

        assert_eq!(parent.trace_id, child.trace_id);
        assert_ne!(parent.span_id, child.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }

    #[test]
    fn test_trace_parent_serialization() {
        let context = TraceContext::new();
        let trace_parent = context.to_trace_parent();

        assert!(trace_parent.starts_with("00-"));
        assert_eq!(trace_parent.split('-').count(), 4);

        let parsed = TraceContext::from_trace_parent(&trace_parent).unwrap();
        assert_eq!(context.trace_id, parsed.trace_id);
        assert_eq!(context.span_id, parsed.span_id);
    }

    #[test]
    fn test_span_lifecycle() {
        let context = TraceContext::new();
        let mut span = Span::new("test_operation".to_string(), SpanKind::Internal, context);

        assert!(span.is_active());
        assert!(span.duration().is_none());

        span.set_attribute(
            "test.key".to_string(),
            AttributeValue::String("test_value".to_string()),
        );
        span.add_event("test_event".to_string(), HashMap::new());
        span.set_status(SpanStatus::Ok);
        span.end();

        assert!(!span.is_active());
        assert!(span.duration().is_some());
        assert_eq!(span.status, SpanStatus::Ok);
        assert_eq!(span.attributes.len(), 1);
        assert_eq!(span.events.len(), 1);
    }

    #[test]
    fn test_sampling_strategies() {
        let always = AlwaysSampler;
        let never = NeverSampler;
        let prob = ProbabilitySampler::new(0.5);
        let rate = RateLimitingSampler::new(10.0);

        let context = TraceContext::new();

        assert_eq!(
            always
                .should_sample(&context, "test", SpanKind::Internal)
                .decision,
            SamplingDecision::RecordAndSample
        );

        assert_eq!(
            never
                .should_sample(&context, "test", SpanKind::Internal)
                .decision,
            SamplingDecision::Drop
        );

        // Probability sampler should sometimes sample
        let prob_result = prob.should_sample(&context, "test", SpanKind::Internal);
        assert!(matches!(
            prob_result.decision,
            SamplingDecision::RecordAndSample | SamplingDecision::Drop
        ));

        // Rate limiter should initially sample
        let rate_result = rate.should_sample(&context, "test", SpanKind::Internal);
        assert_eq!(rate_result.decision, SamplingDecision::RecordAndSample);
    }

    #[test]
    fn test_tracer_span_management() {
        let processor = Arc::new(LoggingSpanProcessor);
        let sampler = Arc::new(AlwaysSampler);
        let tracer = Tracer::new(
            "test_service".to_string(),
            "1.0.0".to_string(),
            processor,
            sampler,
        );

        let span = tracer.start_span("test_span".to_string(), SpanKind::Internal, None);
        let span_id = span.context.span_id.clone();

        // Span should be active
        assert!(tracer.get_active_span(&span_id).is_some());
        assert_eq!(tracer.get_active_spans().len(), 1);

        tracer.end_span(span);

        // Span should no longer be active
        assert!(tracer.get_active_span(&span_id).is_none());
        assert_eq!(tracer.get_active_spans().len(), 0);
    }

    #[test]
    fn test_speech_recognition_instrumentation() {
        let instrumentation = SpeechRecognitionInstrumentation::new(
            "voirs_recognizer".to_string(),
            "1.0.0".to_string(),
        );

        // Create a trace for a complete recognition pipeline
        let preprocessing_span = instrumentation.instrument_preprocessing(None);
        let preprocessing_context = Some(preprocessing_span.context.clone());
        instrumentation.end_span(preprocessing_span);

        let feature_span =
            instrumentation.instrument_feature_extraction(preprocessing_context.clone());
        let feature_context = Some(feature_span.context.clone());
        instrumentation.end_span(feature_span);

        let inference_span =
            instrumentation.instrument_inference("transformer", feature_context.clone());
        let inference_context = Some(inference_span.context.clone());
        instrumentation.end_span(inference_span);

        let postprocessing_span = instrumentation.instrument_postprocessing(inference_context);
        instrumentation.end_span(postprocessing_span);
    }
}
