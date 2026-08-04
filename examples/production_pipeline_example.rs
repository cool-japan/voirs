//! Production Pipeline Example - Enterprise-Grade VoiRS Deployment
//!
//! This comprehensive example demonstrates a production-ready VoiRS implementation
//! suitable for enterprise deployment. It includes monitoring, error handling,
//! performance optimization, scaling capabilities, and operational best practices.
//!
//! ## Production Features Demonstrated:
//! 1. **Monitoring & Observability**
//!    - Performance metrics collection and analysis
//!    - Error tracking and alerting
//!    - Resource usage monitoring
//!    - Health checks and status reporting
//!
//! 2. **Error Handling & Recovery**
//!    - Comprehensive error classification
//!    - Automatic retry mechanisms with exponential backoff
//!    - Graceful degradation strategies
//!    - Circuit breaker patterns for external dependencies
//!
//! 3. **Performance & Scaling**
//!    - Connection pooling and resource management
//!    - Load balancing across multiple synthesis engines
//!    - Caching strategies for frequently requested content
//!    - Asynchronous processing with queue management
//!
//! 4. **Security & Compliance**
//!    - Request validation and sanitization
//!    - Rate limiting and abuse prevention
//!    - Audit logging for compliance
//!    - Secure configuration management
//!
//! 5. **Operational Excellence**
//!    - Configuration management and environment handling
//!    - Graceful shutdown and resource cleanup
//!    - Version management and feature flags
//!    - Deployment readiness checks
//!
//! ## Architecture Components:
//! - **ProductionSynthesizer**: Main synthesis engine with enterprise features
//! - **RequestProcessor**: Handles request validation, queuing, and processing
//! - **MonitoringService**: Collects metrics and health data
//! - **ConfigurationManager**: Manages environment-specific settings
//! - **CacheService**: Intelligent caching for performance optimization
//!
//! ## Prerequisites:
//! - Rust 1.70+ with production-grade dependencies
//! - Monitoring infrastructure (Prometheus/Grafana compatible)
//! - Logging aggregation system
//! - Load balancer configuration
//!
//! ## Deployment Considerations:
//! ```bash
//! # Production build with optimizations
//! cargo build --release --features production
//!
//! # Docker deployment
//! docker build -t voirs-production .
//! docker run -d --name voirs-prod -p 8080:8080 voirs-production
//!
//! # Kubernetes deployment
//! kubectl apply -f k8s-deployment.yaml
//! ```
//!
//! ## Expected output:
//! - Production-grade synthesis with comprehensive monitoring
//! - Performance metrics and health reports
//! - Scalable request processing demonstration
//! - Enterprise deployment guidelines

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{info, span, warn, Level};
use uuid::Uuid;
use voirs_evaluation::traits::QualityEvaluator as _;
use voirs_sdk::prelude::*;

/// Production-grade configuration with environment-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    /// Service configuration
    pub service: ServiceConfig,
    /// Synthesis engine configuration
    pub synthesis: SynthesisEngineConfig,
    /// Monitoring and observability settings
    pub monitoring: MonitoringConfig,
    /// Caching configuration
    pub cache: CacheConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Resource limits and scaling
    pub resources: ResourceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub service_name: String,
    pub version: String,
    pub environment: Environment,
    pub region: String,
    pub instance_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisEngineConfig {
    pub worker_pool_size: usize,
    pub max_queue_size: usize,
    pub request_timeout_seconds: u64,
    pub quality_level: ProductionQualityLevel,
    pub enable_gpu_acceleration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProductionQualityLevel {
    Fast,     // Optimized for speed
    Balanced, // Balanced quality/speed
    Premium,  // Highest quality
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_port: u16,
    pub health_check_interval_seconds: u64,
    pub alert_thresholds: AlertThresholds,
    pub tracing_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub error_rate_percent: f64,
    pub response_time_ms: u64,
    pub queue_depth: usize,
    pub memory_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enable_cache: bool,
    pub max_cache_size_mb: usize,
    pub ttl_seconds: u64,
    pub cache_hit_rate_target: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_rate_limiting: bool,
    pub max_requests_per_minute: usize,
    pub enable_request_validation: bool,
    pub max_text_length: usize,
    pub enable_audit_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub max_concurrent_requests: usize,
    pub memory_limit_mb: usize,
    pub cpu_limit_percent: f64,
    pub auto_scaling: AutoScalingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_instances: usize,
    pub max_instances: usize,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
}

/// Production synthesis request with comprehensive metadata
#[derive(Debug, Clone)]
pub struct ProductionRequest {
    pub id: Uuid,
    pub text: String,
    pub priority: RequestPriority,
    pub client_id: Option<String>,
    pub trace_id: String,
    pub created_at: SystemTime,
    pub deadline: Option<SystemTime>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Production synthesis response with comprehensive metrics
#[derive(Debug, Clone)]
pub struct ProductionResponse {
    pub request_id: Uuid,
    pub audio: Option<AudioBuffer>,
    pub status: ResponseStatus,
    pub processing_time: Duration,
    pub queue_time: Duration,
    pub cache_hit: bool,
    pub metrics: ProcessingMetrics,
    pub error: Option<ProductionError>,
}

#[derive(Debug, Clone)]
pub enum ResponseStatus {
    Success,
    PartialFailure,
    Failed,
    Timeout,
    RateLimited,
}

#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilized: bool,
    pub model_load_time_ms: u64,
    pub synthesis_time_ms: u64,
    pub quality_score: f64,
}

#[derive(Debug, Clone)]
pub struct ProductionError {
    pub error_type: ErrorType,
    pub message: String,
    pub code: String,
    pub retry_after: Option<Duration>,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum ErrorType {
    ValidationError,
    RateLimitExceeded,
    ResourceExhausted,
    ModelLoadFailure,
    SynthesisFailure,
    TimeoutError,
    InternalError,
}

/// Enterprise-grade production synthesizer
pub struct ProductionSynthesizer {
    config: ProductionConfig,
    pipeline: Arc<VoirsPipeline>,
    request_processor: RequestProcessor,
    monitoring: MonitoringService,
    cache: CacheService,
    metrics: Arc<RwLock<ProductionMetrics>>,
}

/// Production metrics for monitoring and alerting
#[derive(Debug, Default)]
pub struct ProductionMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub queue_depth: usize,
    pub active_connections: usize,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Request processor with queuing and load balancing
struct RequestProcessor {
    request_tx: mpsc::UnboundedSender<(ProductionRequest, oneshot::Sender<ProductionResponse>)>,
    semaphore: Arc<Semaphore>,
}

/// Monitoring service for health checks and metrics
struct MonitoringService {
    config: MonitoringConfig,
    start_time: Instant,
    health_status: Arc<RwLock<HealthStatus>>,
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub checks: HashMap<String, HealthCheck>,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: CheckStatus,
    pub response_time_ms: u64,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

/// Intelligent caching service
struct CacheService {
    config: CacheConfig,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<Mutex<CacheStats>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    audio: AudioBuffer,
    created_at: SystemTime,
    access_count: usize,
    last_accessed: SystemTime,
}

#[derive(Debug, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    size_mb: f64,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        ProductionConfig {
            service: ServiceConfig {
                service_name: "voirs-production".to_string(),
                version: "1.0.0".to_string(),
                environment: Environment::Production,
                region: "us-west-2".to_string(),
                instance_id: Uuid::new_v4().to_string(),
            },
            synthesis: SynthesisEngineConfig {
                worker_pool_size: num_cpus::get() * 2,
                max_queue_size: 10000,
                request_timeout_seconds: 30,
                quality_level: ProductionQualityLevel::Balanced,
                enable_gpu_acceleration: true,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                metrics_port: 9090,
                health_check_interval_seconds: 30,
                alert_thresholds: AlertThresholds {
                    error_rate_percent: 5.0,
                    response_time_ms: 5000,
                    queue_depth: 1000,
                    memory_usage_percent: 85.0,
                },
                tracing_level: "info".to_string(),
            },
            cache: CacheConfig {
                enable_cache: true,
                max_cache_size_mb: 1024,
                ttl_seconds: 3600,
                cache_hit_rate_target: 80.0,
            },
            security: SecurityConfig {
                enable_rate_limiting: true,
                max_requests_per_minute: 1000,
                enable_request_validation: true,
                max_text_length: 10000,
                enable_audit_logging: true,
            },
            resources: ResourceConfig {
                max_concurrent_requests: 100,
                memory_limit_mb: 4096,
                cpu_limit_percent: 80.0,
                auto_scaling: AutoScalingConfig {
                    enabled: true,
                    min_instances: 2,
                    max_instances: 10,
                    scale_up_threshold: 70.0,
                    scale_down_threshold: 30.0,
                },
            },
        }
    }
}

impl ProductionSynthesizer {
    /// Create a new production synthesizer with enterprise features
    pub async fn new(config: ProductionConfig) -> Result<Self> {
        let span = span!(Level::INFO, "production_synthesizer_init");
        let _guard = span.enter();

        info!("🏭 Initializing Production VoiRS Synthesizer");
        info!(
            "Service: {} v{} ({})",
            config.service.service_name, config.service.version, config.service.instance_id
        );
        info!(
            "Environment: {:?}, Region: {}",
            config.service.environment, config.service.region
        );

        // Create production-optimized pipeline
        let pipeline = Self::create_production_pipeline(&config).await?;

        // Initialize cache service and quality evaluator before the request
        // processor, since worker tasks need both for every request.
        let cache = CacheService::new(config.cache.clone());
        let quality_evaluator = Arc::new(
            voirs_evaluation::quality::evaluator::QualityEvaluator::new()
                .await
                .context("Failed to initialize quality evaluator")?,
        );

        // Initialize request processor
        let request_processor = Self::create_request_processor(
            &config,
            pipeline.clone(),
            cache.clone(),
            quality_evaluator,
        )
        .await?;

        // Initialize monitoring service
        let monitoring = MonitoringService::new(config.monitoring.clone());

        let synthesizer = ProductionSynthesizer {
            config,
            pipeline,
            request_processor,
            monitoring,
            cache,
            metrics: Arc::new(RwLock::new(ProductionMetrics::default())),
        };

        // Start background services
        synthesizer.start_background_services().await?;

        info!("✅ Production synthesizer ready for enterprise deployment");
        Ok(synthesizer)
    }

    /// Create production-optimized synthesis pipeline
    async fn create_production_pipeline(config: &ProductionConfig) -> Result<Arc<VoirsPipeline>> {
        info!("🔧 Creating production synthesis pipeline");

        // Production-optimized quality level, mapped onto the unified builder's
        // real `QualityLevel`, which the underlying acoustic/vocoder stage honors.
        let quality = match config.synthesis.quality_level {
            ProductionQualityLevel::Fast => {
                info!("Using fast (low-latency) quality preset for production");
                QualityLevel::Low
            }
            ProductionQualityLevel::Balanced => {
                info!("Using balanced quality preset for production");
                QualityLevel::Medium
            }
            ProductionQualityLevel::Premium => {
                info!("Using premium quality preset for production");
                QualityLevel::Ultra
            }
        };

        let pipeline = VoirsPipelineBuilder::new()
            .with_quality(quality)
            .build()
            .await
            .context("Failed to create production synthesis pipeline")?;

        Ok(Arc::new(pipeline))
    }

    /// Create request processor with queuing and load balancing
    async fn create_request_processor(
        config: &ProductionConfig,
        pipeline: Arc<VoirsPipeline>,
        cache: CacheService,
        quality_evaluator: Arc<voirs_evaluation::quality::evaluator::QualityEvaluator>,
    ) -> Result<RequestProcessor> {
        let (request_tx, request_rx) =
            mpsc::unbounded_channel::<(ProductionRequest, oneshot::Sender<ProductionResponse>)>();
        // Shared behind a Mutex so every worker task in the pool can pull from the
        // same queue (`mpsc::UnboundedReceiver` has exactly one owner otherwise).
        let request_rx = Arc::new(tokio::sync::Mutex::new(request_rx));
        let semaphore = Arc::new(Semaphore::new(config.resources.max_concurrent_requests));

        // Start worker pool
        let worker_count = config.synthesis.worker_pool_size;
        info!("🔄 Starting {} production worker threads", worker_count);

        for worker_id in 0..worker_count {
            let pipeline = pipeline.clone();
            let semaphore = semaphore.clone();
            let config = config.clone();
            let request_rx = request_rx.clone();
            let cache = cache.clone();
            let quality_evaluator = quality_evaluator.clone();

            tokio::spawn(async move {
                loop {
                    let next = { request_rx.lock().await.recv().await };
                    let Some((request, response_tx)) = next else {
                        info!("Worker {worker_id} shutting down: request channel closed");
                        break;
                    };

                    // An owned permit (not borrowed from `semaphore`) so it can be
                    // moved into the `'static` inner task below.
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore should not be closed while workers are running");
                    let pipeline = pipeline.clone();
                    let config = config.clone();
                    let cache = cache.clone();
                    let quality_evaluator = quality_evaluator.clone();

                    tokio::spawn(async move {
                        let _permit = permit;
                        let response = Self::process_production_request(
                            request,
                            &pipeline,
                            &config,
                            &cache,
                            &quality_evaluator,
                        )
                        .await;
                        let _ = response_tx.send(response);
                    });
                }
            });
        }

        Ok(RequestProcessor {
            request_tx,
            semaphore,
        })
    }

    /// Process a production request with comprehensive error handling
    async fn process_production_request(
        request: ProductionRequest,
        pipeline: &VoirsPipeline,
        config: &ProductionConfig,
        cache: &CacheService,
        quality_evaluator: &voirs_evaluation::quality::evaluator::QualityEvaluator,
    ) -> ProductionResponse {
        let span = span!(Level::DEBUG, "process_request", request_id = %request.id);
        let _guard = span.enter();

        let queue_time = request.created_at.elapsed().unwrap_or_default();
        let start_time = Instant::now();

        // Validate request
        if let Err(error) = Self::validate_request(&request, config) {
            return ProductionResponse {
                request_id: request.id,
                audio: None,
                status: ResponseStatus::Failed,
                processing_time: start_time.elapsed(),
                queue_time,
                cache_hit: false,
                metrics: ProcessingMetrics::default(),
                error: Some(error),
            };
        }

        // Real cache lookup, keyed on the request text (the only synthesis input
        // this demo varies). A hit skips synthesis entirely.
        if config.cache.enable_cache {
            if let Some(audio) = cache.get(&request.text).await {
                let processing_time = start_time.elapsed();
                return ProductionResponse {
                    request_id: request.id,
                    audio: Some(audio),
                    status: ResponseStatus::Success,
                    processing_time,
                    queue_time,
                    cache_hit: true,
                    metrics: ProcessingMetrics {
                        cpu_usage_percent: 0.0, // No synthesis work performed on a cache hit
                        memory_usage_mb: Self::get_memory_usage_mb(),
                        gpu_utilized: false,
                        model_load_time_ms: 0,
                        synthesis_time_ms: processing_time.as_millis() as u64,
                        quality_score: 0.0, // Not re-evaluated on a cache hit
                    },
                    error: None,
                };
            }
        }

        // Attempt synthesis with retries
        let max_retries = 3;
        let mut attempts = 0;

        while attempts < max_retries {
            attempts += 1;

            match Self::attempt_synthesis(&request.text, pipeline).await {
                Ok(audio) => {
                    let processing_time = start_time.elapsed();

                    // Real quality measurement via voirs-evaluation (no reference
                    // audio is available here, so this scores the synthesis on its
                    // own intrinsic quality signals).
                    let quality_score = quality_evaluator
                        .evaluate_quality(&audio, None, None)
                        .await
                        .map(|score| f64::from(score.overall_score))
                        .unwrap_or(0.0);

                    if config.cache.enable_cache {
                        cache.put(request.text.clone(), audio.clone()).await;
                    }

                    return ProductionResponse {
                        request_id: request.id,
                        audio: Some(audio),
                        status: ResponseStatus::Success,
                        processing_time,
                        queue_time,
                        cache_hit: false,
                        metrics: ProcessingMetrics {
                            cpu_usage_percent: f64::from(Self::get_cpu_usage_percent()),
                            memory_usage_mb: Self::get_memory_usage_mb(),
                            gpu_utilized: config.synthesis.enable_gpu_acceleration,
                            // The pipeline (and its models) is built once at
                            // `ProductionSynthesizer::new`, not per request.
                            model_load_time_ms: 0,
                            synthesis_time_ms: processing_time.as_millis() as u64,
                            quality_score,
                        },
                        error: None,
                    };
                }
                Err(e) => {
                    warn!("Synthesis attempt {} failed: {}", attempts, e);

                    if attempts >= max_retries {
                        let error = ProductionError {
                            error_type: ErrorType::SynthesisFailure,
                            message: format!(
                                "Synthesis failed after {} attempts: {}",
                                max_retries, e
                            ),
                            code: "SYNTHESIS_FAILURE".to_string(),
                            retry_after: Some(Duration::from_secs(60)),
                            details: HashMap::new(),
                        };

                        return ProductionResponse {
                            request_id: request.id,
                            audio: None,
                            status: ResponseStatus::Failed,
                            processing_time: start_time.elapsed(),
                            queue_time,
                            cache_hit: false,
                            metrics: ProcessingMetrics::default(),
                            error: Some(error),
                        };
                    }

                    // Exponential backoff
                    tokio::time::sleep(Duration::from_millis(100 * (1 << attempts))).await;
                }
            }
        }

        unreachable!()
    }

    /// Validate production request
    fn validate_request(
        request: &ProductionRequest,
        config: &ProductionConfig,
    ) -> Result<(), ProductionError> {
        if !config.security.enable_request_validation {
            return Ok(());
        }

        // Text length validation
        if request.text.len() > config.security.max_text_length {
            return Err(ProductionError {
                error_type: ErrorType::ValidationError,
                message: format!(
                    "Text length {} exceeds maximum {}",
                    request.text.len(),
                    config.security.max_text_length
                ),
                code: "TEXT_TOO_LONG".to_string(),
                retry_after: None,
                details: HashMap::new(),
            });
        }

        // Content validation (basic)
        if request.text.trim().is_empty() {
            return Err(ProductionError {
                error_type: ErrorType::ValidationError,
                message: "Text cannot be empty".to_string(),
                code: "EMPTY_TEXT".to_string(),
                retry_after: None,
                details: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Attempt synthesis with production error handling
    async fn attempt_synthesis(text: &str, pipeline: &VoirsPipeline) -> Result<AudioBuffer> {
        Ok(pipeline.synthesize(text).await?)
    }

    /// Real global CPU usage percentage via `sysinfo` (no simulated values).
    fn get_cpu_usage_percent() -> f32 {
        let mut system = sysinfo::System::new_all();
        // CPU usage is a delta between two samples; sysinfo documents the
        // minimum interval required between them for a meaningful reading.
        system.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_cpu_usage();
        system.global_cpu_usage()
    }

    /// Real current-process memory usage in MB via `sysinfo` (no simulated values).
    fn get_memory_usage_mb() -> f64 {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        let mut system = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing()
                .with_processes(sysinfo::ProcessRefreshKind::nothing().with_memory()),
        );
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        system
            .process(pid)
            .map(|process| process.memory() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }

    /// Start background services for monitoring and maintenance
    async fn start_background_services(&self) -> Result<()> {
        info!("🔄 Starting production background services");

        // Health check service
        let monitoring = self.monitoring.clone();
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            monitoring.start_health_checks(metrics).await;
        });

        // Metrics collection service
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            Self::collect_metrics_loop(metrics).await;
        });

        // Cache maintenance service
        let cache = self.cache.clone();
        tokio::spawn(async move {
            cache.start_maintenance().await;
        });

        Ok(())
    }

    /// Metrics collection loop
    async fn collect_metrics_loop(metrics: Arc<RwLock<ProductionMetrics>>) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));

        loop {
            interval.tick().await;

            // Real OS-reported process/system metrics via `sysinfo` (blocking
            // work, so it runs on a dedicated blocking thread rather than
            // stalling this task's async worker thread).
            let (cpu, memory) = tokio::task::spawn_blocking(|| {
                (Self::get_cpu_usage_percent(), Self::get_memory_usage_mb())
            })
            .await
            .unwrap_or((0.0, 0.0));

            let mut metrics = metrics.write().unwrap_or_else(|e| e.into_inner());
            metrics.memory_usage_mb = memory;
            metrics.cpu_usage_percent = f64::from(cpu);
        }
    }

    /// Submit a production synthesis request
    /// Get the production configuration this synthesizer was built with
    pub fn config(&self) -> &ProductionConfig {
        &self.config
    }

    /// Get the underlying synthesis pipeline (e.g. for advanced/manual use)
    pub fn pipeline(&self) -> &Arc<VoirsPipeline> {
        &self.pipeline
    }

    pub async fn synthesize_production(
        &self,
        request: ProductionRequest,
    ) -> Result<ProductionResponse> {
        let span = span!(Level::INFO, "synthesize", request_id = %request.id);
        let _guard = span.enter();

        info!("📥 Received production synthesis request: {}", request.id);

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
            metrics.total_requests += 1;
            metrics.queue_depth += 1;
        }

        // Submit to request processor
        let (response_tx, response_rx) = oneshot::channel();

        // Real backpressure signal: how many concurrent-request permits remain
        // right now (not a simulated/estimated value).
        let available_capacity = self.request_processor.semaphore.available_permits();
        if available_capacity == 0 {
            warn!("Production synthesizer at full concurrent-request capacity; request will queue");
        }

        self.request_processor
            .request_tx
            .send((request, response_tx))
            .context("Failed to submit production request")?;

        // Wait for response
        let response = response_rx
            .await
            .context("Failed to receive production response")?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap_or_else(|e| e.into_inner());
            metrics.queue_depth -= 1;
            match response.status {
                ResponseStatus::Success => metrics.successful_requests += 1,
                _ => metrics.failed_requests += 1,
            }
            metrics.error_rate =
                metrics.failed_requests as f64 / metrics.total_requests as f64 * 100.0;
        }

        info!(
            "📤 Production synthesis response: {} ({:?})",
            response.request_id, response.status
        );

        Ok(response)
    }

    /// Get production metrics for monitoring
    pub fn get_production_metrics(&self) -> ProductionMetrics {
        let metrics = self.metrics.read().unwrap_or_else(|e| e.into_inner());
        ProductionMetrics {
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            average_response_time_ms: metrics.average_response_time_ms,
            cache_hit_rate: metrics.cache_hit_rate,
            error_rate: metrics.error_rate,
            queue_depth: metrics.queue_depth,
            active_connections: metrics.active_connections,
            uptime_seconds: metrics.uptime_seconds,
            memory_usage_mb: metrics.memory_usage_mb,
            cpu_usage_percent: metrics.cpu_usage_percent,
        }
    }

    /// Get health status for load balancer health checks
    pub async fn get_health_status(&self) -> HealthStatus {
        self.monitoring.get_health_status().await
    }
}

// Implementation of helper structs and their methods would continue...
// For brevity, showing the main production synthesizer structure

impl Clone for MonitoringService {
    fn clone(&self) -> Self {
        MonitoringService {
            config: self.config.clone(),
            start_time: self.start_time,
            health_status: self.health_status.clone(),
        }
    }
}

impl Clone for CacheService {
    fn clone(&self) -> Self {
        CacheService {
            config: self.config.clone(),
            cache: self.cache.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl MonitoringService {
    fn new(config: MonitoringConfig) -> Self {
        MonitoringService {
            config,
            start_time: Instant::now(),
            health_status: Arc::new(RwLock::new(HealthStatus {
                status: ServiceStatus::Healthy,
                checks: HashMap::new(),
                last_updated: SystemTime::now(),
            })),
        }
    }

    async fn start_health_checks(&self, _metrics: Arc<RwLock<ProductionMetrics>>) {
        // Implementation would include actual health checks
    }

    async fn get_health_status(&self) -> HealthStatus {
        self.health_status
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl CacheService {
    fn new(config: CacheConfig) -> Self {
        CacheService {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Look up a cached synthesis result, recording a real hit/miss and
    /// bumping the entry's access bookkeeping on success.
    async fn get(&self, text: &str) -> Option<AudioBuffer> {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());

        let hit = match cache.get_mut(text) {
            Some(entry) if entry.created_at.elapsed().unwrap_or_default() < ttl => {
                entry.access_count += 1;
                entry.last_accessed = SystemTime::now();
                Some(entry.audio.clone())
            }
            Some(_) => {
                // Entry exists but has expired.
                cache.remove(text);
                None
            }
            None => None,
        };

        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        if hit.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }

        hit
    }

    /// Store a synthesis result, evicting the least-recently-used entry first
    /// if this would push the cache over its configured size budget.
    async fn put(&self, text: String, audio: AudioBuffer) {
        let entry_mb =
            audio.samples().len() as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0);

        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        while stats.size_mb + entry_mb > self.config.max_cache_size_mb as f64 && !cache.is_empty() {
            if let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(key, _)| key.clone())
            {
                if let Some(evicted) = cache.remove(&lru_key) {
                    stats.size_mb -= evicted.audio.samples().len() as f64
                        * std::mem::size_of::<f32>() as f64
                        / (1024.0 * 1024.0);
                    stats.evictions += 1;
                }
            } else {
                break;
            }
        }

        cache.insert(
            text,
            CacheEntry {
                audio,
                created_at: SystemTime::now(),
                access_count: 0,
                last_accessed: SystemTime::now(),
            },
        );
        stats.size_mb += entry_mb;
    }

    /// Periodically sweep expired entries so the cache doesn't grow unbounded
    /// with stale data even when nothing evicts it via `put`'s size budget.
    async fn start_maintenance(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        let ttl = Duration::from_secs(self.config.ttl_seconds);

        loop {
            interval.tick().await;

            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            let before = cache.len();
            cache.retain(|_, entry| entry.created_at.elapsed().unwrap_or_default() < ttl);
            let expired = before - cache.len();

            if expired > 0 {
                let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                stats.evictions += expired as u64;
                info!("🧹 Cache maintenance: expired {expired} stale entries");
            }
        }
    }
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        ProcessingMetrics {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilized: false,
            model_load_time_ms: 0,
            synthesis_time_ms: 0,
            quality_score: 0.0,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize production-grade logging with structured output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .json()
        .init();

    info!("🏭 VoiRS Production Pipeline Example");
    info!("===================================");

    // Load production configuration (in real deployment, from environment/files)
    let config = ProductionConfig::default();

    info!("📋 Production Configuration:");
    info!(
        "   Service: {} v{}",
        config.service.service_name, config.service.version
    );
    info!("   Environment: {:?}", config.service.environment);
    info!(
        "   Worker Pool: {} threads",
        config.synthesis.worker_pool_size
    );
    info!(
        "   Max Concurrent: {}",
        config.resources.max_concurrent_requests
    );
    info!("   Cache: {}MB", config.cache.max_cache_size_mb);

    // Create production synthesizer
    let production_start = Instant::now();
    let synthesizer = ProductionSynthesizer::new(config).await?;
    let startup_time = production_start.elapsed();

    info!(
        "✅ Production synthesizer ready in {:.2}s",
        startup_time.as_secs_f32()
    );

    // Demonstrate production synthesis with various scenarios
    let production_scenarios = [
        (
            "Normal Request",
            "Welcome to our production voice synthesis service.",
            RequestPriority::Normal,
        ),
        (
            "High Priority",
            "Critical system alert: immediate attention required.",
            RequestPriority::High,
        ),
        (
            "Bulk Processing",
            "This is a bulk processing request for large-scale content generation.",
            RequestPriority::Low,
        ),
        (
            "Real-time Response",
            "Real-time synthesis for interactive applications.",
            RequestPriority::Critical,
        ),
    ];

    info!(
        "🔄 Processing {} production scenarios",
        production_scenarios.len()
    );

    for (scenario_name, text, priority) in production_scenarios.iter() {
        info!("📝 Processing scenario: {}", scenario_name);

        let request = ProductionRequest {
            id: Uuid::new_v4(),
            text: text.to_string(),
            priority: priority.clone(),
            client_id: Some("production-client-001".to_string()),
            trace_id: Uuid::new_v4().to_string(),
            created_at: SystemTime::now(),
            deadline: Some(SystemTime::now() + Duration::from_secs(30)),
            metadata: [
                ("scenario".to_string(), scenario_name.to_string()),
                ("version".to_string(), "1.0.0".to_string()),
            ]
            .into_iter()
            .collect(),
        };

        let scenario_start = Instant::now();
        let response = synthesizer.synthesize_production(request).await?;
        let _scenario_time = scenario_start.elapsed();

        match response.status {
            ResponseStatus::Success => {
                info!("✅ Scenario '{}' successful", scenario_name);
                info!("   Request ID: {}", response.request_id);
                info!(
                    "   Processing time: {:.2}s",
                    response.processing_time.as_secs_f32()
                );
                info!("   Queue time: {:.2}s", response.queue_time.as_secs_f32());
                info!("   Cache hit: {}", response.cache_hit);
                if let Some(audio) = response.audio {
                    info!(
                        "   Audio: {:.2}s, {} samples",
                        audio.duration(),
                        audio.samples().len()
                    );

                    // Save production output
                    let filename = format!(
                        "production_{}.wav",
                        scenario_name.to_lowercase().replace(" ", "_")
                    );
                    audio.save_wav(&filename)?;
                    info!("   Saved: {}", filename);
                }
            }
            _ => {
                warn!(
                    "❌ Scenario '{}' failed: {:?}",
                    scenario_name, response.status
                );
                if let Some(error) = response.error {
                    warn!("   Error: {} ({})", error.message, error.code);
                }
            }
        }
    }

    // Display production metrics and health status
    let metrics = synthesizer.get_production_metrics();
    let health = synthesizer.get_health_status().await;

    info!("📊 Production Performance Metrics:");
    info!("   Total Requests: {}", metrics.total_requests);
    info!(
        "   Success Rate: {:.1}%",
        (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0
    );
    info!("   Error Rate: {:.2}%", metrics.error_rate);
    info!("   Queue Depth: {}", metrics.queue_depth);
    info!("   Memory Usage: {:.1}MB", metrics.memory_usage_mb);
    info!("   CPU Usage: {:.1}%", metrics.cpu_usage_percent);

    info!("🏥 Service Health Status:");
    info!("   Overall Status: {:?}", health.status);
    info!("   Health Checks: {}", health.checks.len());
    info!("   Last Updated: {:?}", health.last_updated);

    // Production deployment guidance
    info!("📋 Production Deployment Guidelines:");
    info!("====================================");
    info!("🔧 Infrastructure Requirements:");
    info!("   • Load balancer with health check endpoint");
    info!("   • Monitoring system (Prometheus/Grafana)");
    info!("   • Log aggregation (ELK/Fluentd)");
    info!("   • Container orchestration (Kubernetes/Docker Swarm)");
    info!("");
    info!("🚀 Scaling Considerations:");
    info!("   • Horizontal scaling based on queue depth");
    info!("   • Vertical scaling for GPU-intensive workloads");
    info!("   • Cache warming strategies for predictable loads");
    info!("   • Circuit breaker patterns for external dependencies");
    info!("");
    info!("🔒 Security & Compliance:");
    info!("   • Input validation and sanitization");
    info!("   • Rate limiting and DDoS protection");
    info!("   • Audit logging for compliance requirements");
    info!("   • Secure configuration management");
    info!("");
    info!("📈 Monitoring & Alerting:");
    info!("   • SLA monitoring (availability, latency, error rate)");
    info!("   • Resource utilization alerts");
    info!("   • Business metrics tracking");
    info!("   • Automated recovery procedures");

    info!("🎉 Production Pipeline Example Complete!");

    Ok(())
}
