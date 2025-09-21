//! Cache management and coordination system.

use crate::{
    cache::{
        models::{AdvancedModelCache, ModelCacheConfig, ModelCacheStats},
        results::{ResultCacheConfig, ResultCacheStats, SynthesisResultCache},
    },
    error::{Result, VoirsError},
    traits::{CacheStats, ModelCache},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};
use tokio::{sync::Notify, time::interval};
use tracing::{debug, error, info, warn};

/// Central cache management coordinator
#[allow(dead_code)] // Advanced caching system - fields may be used in future implementations
pub struct CacheManager {
    /// Model cache instance
    model_cache: Arc<AdvancedModelCache>,

    /// Result cache instance
    result_cache: Arc<SynthesisResultCache>,

    /// Global cache configuration
    config: CacheManagerConfig,

    /// Cache directories
    cache_dirs: CacheDirs,

    /// Combined cache statistics
    stats: Arc<RwLock<CombinedCacheStats>>,

    /// Cache health monitor
    health_monitor: Arc<CacheHealthMonitor>,

    /// Background task controller
    task_controller: Arc<BackgroundTaskController>,

    /// Cache metrics collector
    metrics_collector: Arc<CacheMetricsCollector>,
}

/// Cache manager configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheManagerConfig {
    /// Model cache configuration
    pub model_cache: ModelCacheConfig,

    /// Result cache configuration
    pub result_cache: ResultCacheConfig,

    /// Global cache settings
    pub global_settings: GlobalCacheSettings,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,

    /// Maintenance configuration
    pub maintenance: MaintenanceConfig,
}

/// Global cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCacheSettings {
    /// Enable global cache coordination
    pub enable_coordination: bool,

    /// Maximum total memory usage across all caches (MB)
    pub max_total_memory_mb: usize,

    /// Enable cache warming on startup
    pub enable_startup_warming: bool,

    /// Enable adaptive cache sizing
    pub enable_adaptive_sizing: bool,

    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f64,

    /// Enable cache compression
    pub enable_compression: bool,

    /// Cache encryption (for sensitive data)
    pub enable_encryption: bool,

    /// Cross-cache deduplication
    pub enable_deduplication: bool,
}

impl Default for GlobalCacheSettings {
    fn default() -> Self {
        Self {
            enable_coordination: true,
            max_total_memory_mb: 2048, // 2GB total
            enable_startup_warming: true,
            enable_adaptive_sizing: true,
            memory_pressure_threshold: 0.85,
            enable_compression: true,
            enable_encryption: false,
            enable_deduplication: true,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable health monitoring
    pub enable_health_monitoring: bool,

    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,

    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,

    /// Performance metrics interval in seconds
    pub metrics_interval_seconds: u64,

    /// Enable cache alerting
    pub enable_alerting: bool,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_health_monitoring: true,
            health_check_interval_seconds: 60,
            enable_performance_monitoring: true,
            metrics_interval_seconds: 30,
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
            enable_detailed_logging: false,
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// High memory usage threshold (0.0-1.0)
    pub high_memory_usage: f64,

    /// Low hit rate threshold (0.0-1.0)
    pub low_hit_rate: f64,

    /// High eviction rate (evictions per minute)
    pub high_eviction_rate: f64,

    /// High error rate (errors per minute)
    pub high_error_rate: f64,

    /// Slow response time threshold (ms)
    pub slow_response_time_ms: u64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            high_memory_usage: 0.9,
            low_hit_rate: 0.5,
            high_eviction_rate: 10.0,
            high_error_rate: 5.0,
            slow_response_time_ms: 1000,
        }
    }
}

/// Maintenance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceConfig {
    /// Enable automatic maintenance
    pub enable_auto_maintenance: bool,

    /// Maintenance interval in seconds
    pub maintenance_interval_seconds: u64,

    /// Cleanup expired entries
    pub cleanup_expired: bool,

    /// Defragment cache storage
    pub defragment_storage: bool,

    /// Optimize cache layout
    pub optimize_layout: bool,

    /// Update cache statistics
    pub update_statistics: bool,

    /// Maintenance window (hour of day, 0-23)
    pub maintenance_window_hour: Option<u8>,

    /// Maximum maintenance duration in seconds
    pub max_maintenance_duration_seconds: u64,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            enable_auto_maintenance: true,
            maintenance_interval_seconds: 3600, // 1 hour
            cleanup_expired: true,
            defragment_storage: true,
            optimize_layout: true,
            update_statistics: true,
            maintenance_window_hour: Some(2),      // 2 AM
            max_maintenance_duration_seconds: 300, // 5 minutes
        }
    }
}

/// Cache directory configuration
#[derive(Debug, Clone)]
pub struct CacheDirs {
    /// Base cache directory
    pub base_dir: PathBuf,

    /// Model cache directory
    pub model_cache_dir: PathBuf,

    /// Result cache directory
    pub result_cache_dir: PathBuf,

    /// Temporary cache directory
    pub temp_dir: PathBuf,

    /// Metadata directory
    pub metadata_dir: PathBuf,
}

impl CacheDirs {
    /// Create new cache directories
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        let model_cache_dir = base_dir.join("models");
        let result_cache_dir = base_dir.join("results");
        let temp_dir = base_dir.join("temp");
        let metadata_dir = base_dir.join("metadata");

        // Create directories
        for dir in [
            &base_dir,
            &model_cache_dir,
            &result_cache_dir,
            &temp_dir,
            &metadata_dir,
        ] {
            std::fs::create_dir_all(dir).map_err(|e| {
                VoirsError::cache_error(format!("Failed to create cache directory {dir:?}: {e}"))
            })?;
        }

        Ok(Self {
            base_dir,
            model_cache_dir,
            result_cache_dir,
            temp_dir,
            metadata_dir,
        })
    }
}

/// Combined cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedCacheStats {
    /// Model cache statistics
    pub model_stats: ModelCacheStats,

    /// Result cache statistics
    pub result_stats: ResultCacheStats,

    /// Global statistics
    pub global_stats: GlobalCacheStats,

    /// Health metrics
    pub health_metrics: HealthMetrics,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,

    /// Last updated timestamp
    pub last_updated: SystemTime,
}

impl Default for CombinedCacheStats {
    fn default() -> Self {
        Self {
            model_stats: ModelCacheStats::default(),
            result_stats: ResultCacheStats::default(),
            global_stats: GlobalCacheStats::default(),
            health_metrics: HealthMetrics::default(),
            performance_metrics: PerformanceMetrics::default(),
            last_updated: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Global cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalCacheStats {
    /// Total memory usage across all caches
    pub total_memory_usage_bytes: usize,

    /// Total memory usage in MB
    pub total_memory_usage_mb: usize,

    /// Total cache entries
    pub total_entries: usize,

    /// Overall hit rate
    pub overall_hit_rate: f64,

    /// Overall miss rate
    pub overall_miss_rate: f64,

    /// Memory efficiency (useful data / total memory)
    pub memory_efficiency: f64,

    /// Cache coordination overhead
    pub coordination_overhead_ms: f64,

    /// Deduplication savings
    pub deduplication_savings_bytes: usize,

    /// Compression ratio
    pub compression_ratio: f64,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Overall health score (0.0-1.0)
    pub overall_health_score: f64,

    /// Memory health (0.0-1.0)
    pub memory_health: f64,

    /// Performance health (0.0-1.0)
    pub performance_health: f64,

    /// Error rate health (0.0-1.0)
    pub error_rate_health: f64,

    /// Cache consistency score (0.0-1.0)
    pub consistency_score: f64,

    /// Last health check
    pub last_health_check: SystemTime,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            overall_health_score: 1.0,
            memory_health: 1.0,
            performance_health: 1.0,
            error_rate_health: 1.0,
            consistency_score: 1.0,
            last_health_check: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// 95th percentile response time
    pub p95_response_time_ms: f64,

    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,

    /// Cache operations per second
    pub cache_ops_per_sec: f64,

    /// Background task performance
    pub background_task_performance: HashMap<String, f64>,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,

    /// Memory usage percentage
    pub memory_usage_percent: f64,

    /// Disk I/O rate (MB/s)
    pub disk_io_rate_mbps: f64,

    /// Network I/O rate (MB/s)
    pub network_io_rate_mbps: f64,
}

/// Cache health monitor
pub struct CacheHealthMonitor {
    /// Health check results
    health_history: Arc<RwLock<Vec<HealthCheckResult>>>,

    /// Alert conditions
    active_alerts: Arc<RwLock<Vec<CacheAlert>>>,

    /// Health check configuration
    #[allow(dead_code)]
    config: MonitoringConfig,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check timestamp
    pub timestamp: SystemTime,

    /// Overall health score
    pub health_score: f64,

    /// Individual component scores
    pub component_scores: HashMap<String, f64>,

    /// Issues detected
    pub issues: Vec<HealthIssue>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Health issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Issue severity
    pub severity: IssueSeverity,

    /// Issue category
    pub category: IssueCategory,

    /// Issue description
    pub description: String,

    /// Affected component
    pub component: String,

    /// Metric value that triggered the issue
    pub metric_value: f64,

    /// Threshold that was exceeded
    pub threshold: f64,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    Memory,
    Performance,
    Consistency,
    Configuration,
    Resource,
    Other,
}

/// Cache alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheAlert {
    /// Alert ID
    pub id: String,

    /// Alert timestamp
    pub timestamp: SystemTime,

    /// Alert severity
    pub severity: IssueSeverity,

    /// Alert message
    pub message: String,

    /// Alert source component
    pub component: String,

    /// Alert resolved timestamp
    pub resolved_at: Option<SystemTime>,
}

/// Background task controller
pub struct BackgroundTaskController {
    /// Task shutdown signal
    shutdown_notify: Arc<Notify>,

    /// Task handles
    task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,

    /// Task status
    task_status: Arc<RwLock<HashMap<String, TaskStatus>>>,
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    /// Task name
    pub name: String,

    /// Running status
    pub running: bool,

    /// Last execution time
    pub last_execution: Option<SystemTime>,

    /// Execution count
    pub execution_count: u64,

    /// Error count
    pub error_count: u64,

    /// Average execution duration
    pub avg_duration_ms: f64,
}

/// Cache metrics collector
pub struct CacheMetricsCollector {
    /// Collected metrics
    metrics: Arc<RwLock<Vec<MetricPoint>>>,

    /// Metrics configuration
    #[allow(dead_code)]
    config: MonitoringConfig,
}

/// Metric point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric timestamp
    pub timestamp: SystemTime,

    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Metric tags
    pub tags: HashMap<String, String>,
}

impl CacheManager {
    /// Create new cache manager
    pub async fn new(config: CacheManagerConfig, base_cache_dir: PathBuf) -> Result<Self> {
        // Set up cache directories
        let cache_dirs = CacheDirs::new(base_cache_dir)?;

        // Create model cache
        let model_cache = Arc::new(AdvancedModelCache::new(
            config.model_cache.clone(),
            Some(cache_dirs.model_cache_dir.clone()),
        )?);

        // Create result cache
        let result_cache = Arc::new(SynthesisResultCache::new(
            config.result_cache.clone(),
            Some(cache_dirs.result_cache_dir.clone()),
        )?);

        // Create health monitor
        let health_monitor = Arc::new(CacheHealthMonitor {
            health_history: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            config: config.monitoring.clone(),
        });

        // Create task controller
        let task_controller = Arc::new(BackgroundTaskController {
            shutdown_notify: Arc::new(Notify::new()),
            task_handles: Arc::new(RwLock::new(Vec::new())),
            task_status: Arc::new(RwLock::new(HashMap::new())),
        });

        // Create metrics collector
        let metrics_collector = Arc::new(CacheMetricsCollector {
            metrics: Arc::new(RwLock::new(Vec::new())),
            config: config.monitoring.clone(),
        });

        let cache_manager = Self {
            model_cache,
            result_cache,
            config: config.clone(),
            cache_dirs,
            stats: Arc::new(RwLock::new(CombinedCacheStats::default())),
            health_monitor,
            task_controller,
            metrics_collector,
        };

        // Start background tasks
        if config.maintenance.enable_auto_maintenance {
            cache_manager.start_background_tasks().await?;
        }

        // Perform startup warming if enabled
        if config.global_settings.enable_startup_warming {
            cache_manager.perform_startup_warming().await?;
        }

        info!("Cache manager initialized successfully");
        Ok(cache_manager)
    }

    /// Get model cache
    pub fn model_cache(&self) -> Arc<AdvancedModelCache> {
        Arc::clone(&self.model_cache)
    }

    /// Get result cache
    pub fn result_cache(&self) -> Arc<SynthesisResultCache> {
        Arc::clone(&self.result_cache)
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&self) -> Result<()> {
        let mut task_handles = Vec::new();

        // Start maintenance task
        if self.config.maintenance.enable_auto_maintenance {
            let maintenance_handle = self.start_maintenance_task().await;
            task_handles.push(maintenance_handle);
        }

        // Start health monitoring task
        if self.config.monitoring.enable_health_monitoring {
            let health_handle = self.start_health_monitoring_task().await;
            task_handles.push(health_handle);
        }

        // Start metrics collection task
        if self.config.monitoring.enable_performance_monitoring {
            let metrics_handle = self.start_metrics_collection_task().await;
            task_handles.push(metrics_handle);
        }

        // Update task handles after all async operations are complete
        let task_count = task_handles.len();
        {
            let mut stored_handles = self.task_controller.task_handles.write().unwrap();
            stored_handles.extend(task_handles);
        }

        info!("Started {} background tasks", task_count);
        Ok(())
    }

    /// Start maintenance task
    async fn start_maintenance_task(&self) -> tokio::task::JoinHandle<()> {
        let model_cache = Arc::clone(&self.model_cache);
        let result_cache = Arc::clone(&self.result_cache);
        let config = self.config.maintenance.clone();
        let shutdown_notify = Arc::clone(&self.task_controller.shutdown_notify);
        let task_status = Arc::clone(&self.task_controller.task_status);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.maintenance_interval_seconds));

            // Initialize task status
            {
                let mut status = task_status.write().unwrap();
                status.insert(
                    "maintenance".to_string(),
                    TaskStatus {
                        name: "maintenance".to_string(),
                        running: true,
                        last_execution: None,
                        execution_count: 0,
                        error_count: 0,
                        avg_duration_ms: 0.0,
                    },
                );
            }

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let start_time = Instant::now();

                        if let Err(e) = Self::perform_maintenance_cycle(&model_cache, &result_cache, &config).await {
                            error!("Maintenance cycle failed: {}", e);

                            // Update error count
                            let mut status = task_status.write().unwrap();
                            if let Some(task_status) = status.get_mut("maintenance") {
                                task_status.error_count += 1;
                            }
                        }

                        // Update task status
                        let duration = start_time.elapsed().as_millis() as f64;
                        let mut status = task_status.write().unwrap();
                        if let Some(task_status) = status.get_mut("maintenance") {
                            task_status.last_execution = Some(SystemTime::now());
                            task_status.execution_count += 1;

                            // Update average duration
                            let total_duration = task_status.avg_duration_ms * (task_status.execution_count - 1) as f64 + duration;
                            task_status.avg_duration_ms = total_duration / task_status.execution_count as f64;
                        }
                    }
                    _ = shutdown_notify.notified() => {
                        info!("Maintenance task shutting down");
                        break;
                    }
                }
            }

            // Mark task as stopped
            let mut status = task_status.write().unwrap();
            if let Some(task_status) = status.get_mut("maintenance") {
                task_status.running = false;
            }
        })
    }

    /// Start health monitoring task
    async fn start_health_monitoring_task(&self) -> tokio::task::JoinHandle<()> {
        let model_cache = Arc::clone(&self.model_cache);
        let result_cache = Arc::clone(&self.result_cache);
        let health_monitor = Arc::clone(&self.health_monitor);
        let config = self.config.monitoring.clone();
        let shutdown_notify = Arc::clone(&self.task_controller.shutdown_notify);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.health_check_interval_seconds));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::perform_health_check(&model_cache, &result_cache, &health_monitor, &config).await {
                            error!("Health check failed: {}", e);
                        }
                    }
                    _ = shutdown_notify.notified() => {
                        info!("Health monitoring task shutting down");
                        break;
                    }
                }
            }
        })
    }

    /// Start metrics collection task
    async fn start_metrics_collection_task(&self) -> tokio::task::JoinHandle<()> {
        let model_cache = Arc::clone(&self.model_cache);
        let result_cache = Arc::clone(&self.result_cache);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let config = self.config.monitoring.clone();
        let shutdown_notify = Arc::clone(&self.task_controller.shutdown_notify);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.metrics_interval_seconds));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::collect_metrics(&model_cache, &result_cache, &metrics_collector).await {
                            error!("Metrics collection failed: {}", e);
                        }
                    }
                    _ = shutdown_notify.notified() => {
                        info!("Metrics collection task shutting down");
                        break;
                    }
                }
            }
        })
    }

    /// Perform maintenance cycle
    async fn perform_maintenance_cycle(
        model_cache: &AdvancedModelCache,
        result_cache: &SynthesisResultCache,
        config: &MaintenanceConfig,
    ) -> Result<()> {
        debug!("Starting maintenance cycle");

        // Cleanup expired entries
        if config.cleanup_expired {
            let expired_results = result_cache.cleanup_expired().await?;
            if expired_results > 0 {
                debug!("Cleaned up {} expired results", expired_results);
            }

            // Model cache maintenance
            model_cache.perform_maintenance().await?;
        }

        debug!("Maintenance cycle completed");
        Ok(())
    }

    /// Perform health check
    async fn perform_health_check(
        model_cache: &AdvancedModelCache,
        result_cache: &SynthesisResultCache,
        health_monitor: &CacheHealthMonitor,
        _config: &MonitoringConfig,
    ) -> Result<()> {
        let mut component_scores = HashMap::new();
        let issues = Vec::new();

        // Check model cache health
        let model_stats = model_cache.stats();
        let model_health = Self::calculate_cache_health(&model_stats);
        component_scores.insert("model_cache".to_string(), model_health);

        // Check result cache health
        let result_stats = result_cache.stats();
        let result_health = Self::calculate_cache_health(&result_stats.basic_stats);
        component_scores.insert("result_cache".to_string(), result_health);

        // Calculate overall health
        let health_score = (model_health + result_health) / 2.0;

        // Generate recommendations based on health scores and cache statistics
        let recommendations = Self::generate_health_recommendations(
            &model_stats,
            &result_stats.basic_stats,
            model_health,
            result_health,
            health_score,
        );

        // Generate health check result
        let health_result = HealthCheckResult {
            timestamp: SystemTime::now(),
            health_score,
            component_scores,
            issues,
            recommendations,
        };

        // Store health result
        {
            let mut history = health_monitor.health_history.write().unwrap();
            history.push(health_result);

            // Keep only recent history
            if history.len() > 100 {
                history.remove(0);
            }
        }

        debug!("Health check completed: score = {:.2}", health_score);
        Ok(())
    }

    /// Calculate cache health score
    fn calculate_cache_health(stats: &CacheStats) -> f64 {
        // Simple health calculation based on hit rate and memory usage
        let hit_rate_score = stats.hit_rate / 100.0;
        let memory_score = if stats.memory_usage_bytes > 0 {
            0.8
        } else {
            1.0
        };

        ((hit_rate_score + memory_score) / 2.0) as f64
    }

    /// Generate health recommendations based on cache statistics
    fn generate_health_recommendations(
        model_stats: &CacheStats,
        result_stats: &CacheStats,
        model_health: f64,
        result_health: f64,
        overall_health: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Overall health recommendations
        if overall_health < 0.5 {
            recommendations.push(
                "Critical: Cache system health is poor. Consider restarting the cache manager."
                    .to_string(),
            );
        } else if overall_health < 0.7 {
            recommendations.push("Warning: Cache system health is degraded. Monitor closely and consider optimization.".to_string());
        }

        // Model cache specific recommendations
        if model_health < 0.6 {
            recommendations.push("Model cache performance is poor. Consider increasing cache size or clearing outdated models.".to_string());
        }

        if model_stats.hit_rate < 50.0 {
            recommendations.push(
                "Model cache hit rate is low. Consider preloading frequently used models."
                    .to_string(),
            );
        }

        if model_stats.memory_usage_bytes > 1_000_000_000 {
            // 1GB
            recommendations.push("Model cache is using high memory. Consider reducing cache size or clearing unused models.".to_string());
        }

        // Result cache specific recommendations
        if result_health < 0.6 {
            recommendations.push("Result cache performance is poor. Consider increasing cache size or adjusting retention policies.".to_string());
        }

        if result_stats.hit_rate < 40.0 {
            recommendations.push("Result cache hit rate is low. Consider extending cache retention time or optimizing cache keys.".to_string());
        }

        if result_stats.memory_usage_bytes > 2_000_000_000 {
            // 2GB
            recommendations.push("Result cache is using high memory. Consider reducing cache size or implementing more aggressive cleanup.".to_string());
        }

        // Performance optimization recommendations
        if model_stats.total_entries > 1000 {
            recommendations.push("Model cache has many entries. Consider implementing cache pruning to improve performance.".to_string());
        }

        if result_stats.total_entries > 10000 {
            recommendations.push("Result cache has many entries. Consider implementing time-based expiration for better performance.".to_string());
        }

        // General recommendations for good health
        if overall_health > 0.8 {
            recommendations.push(
                "Cache system is performing well. Current configuration is optimal.".to_string(),
            );
        }

        recommendations
    }

    /// Collect performance metrics
    async fn collect_metrics(
        model_cache: &AdvancedModelCache,
        result_cache: &SynthesisResultCache,
        metrics_collector: &CacheMetricsCollector,
    ) -> Result<()> {
        let timestamp = SystemTime::now();
        let mut metrics = Vec::new();

        // Collect model cache metrics
        let model_stats = model_cache.stats();
        metrics.push(MetricPoint {
            timestamp,
            name: "model_cache_hit_rate".to_string(),
            value: model_stats.hit_rate as f64,
            tags: [("cache_type".to_string(), "model".to_string())]
                .iter()
                .cloned()
                .collect(),
        });

        metrics.push(MetricPoint {
            timestamp,
            name: "model_cache_memory_usage".to_string(),
            value: model_stats.memory_usage_bytes as f64,
            tags: [("cache_type".to_string(), "model".to_string())]
                .iter()
                .cloned()
                .collect(),
        });

        // Collect result cache metrics
        let result_stats = result_cache.stats();
        metrics.push(MetricPoint {
            timestamp,
            name: "result_cache_hit_rate".to_string(),
            value: result_stats.basic_stats.hit_rate as f64,
            tags: [("cache_type".to_string(), "result".to_string())]
                .iter()
                .cloned()
                .collect(),
        });

        metrics.push(MetricPoint {
            timestamp,
            name: "result_cache_memory_usage".to_string(),
            value: result_stats.basic_stats.memory_usage_bytes as f64,
            tags: [("cache_type".to_string(), "result".to_string())]
                .iter()
                .cloned()
                .collect(),
        });

        // Store metrics
        {
            let mut stored_metrics = metrics_collector.metrics.write().unwrap();
            stored_metrics.extend(metrics);

            // Keep only recent metrics (last 1000 points)
            let len = stored_metrics.len();
            if len > 1000 {
                stored_metrics.drain(0..len - 1000);
            }
        }

        Ok(())
    }

    /// Perform startup cache warming
    async fn perform_startup_warming(&self) -> Result<()> {
        info!("Starting cache warming");

        // Warm model cache with commonly used models
        let common_models = vec![
            "default_g2p".to_string(),
            "default_acoustic".to_string(),
            "default_vocoder".to_string(),
        ];

        if let Err(e) = self.model_cache.warm_cache(common_models).await {
            warn!("Model cache warming failed: {}", e);
        }

        info!("Cache warming completed");
        Ok(())
    }

    /// Get combined cache statistics
    pub async fn get_combined_stats(&self) -> CombinedCacheStats {
        let model_stats = self.model_cache.stats();
        let result_stats = self.result_cache.stats();

        // Calculate global stats
        let total_memory_usage_bytes =
            model_stats.memory_usage_bytes + result_stats.basic_stats.memory_usage_bytes;
        let total_entries = model_stats.total_entries + result_stats.basic_stats.total_entries;

        let overall_hit_rate = if total_entries > 0 {
            (model_stats.hit_rate + result_stats.basic_stats.hit_rate) / 2.0
        } else {
            0.0
        };

        let global_stats = GlobalCacheStats {
            total_memory_usage_bytes,
            total_memory_usage_mb: total_memory_usage_bytes / (1024 * 1024),
            total_entries,
            overall_hit_rate: overall_hit_rate as f64,
            overall_miss_rate: (100.0 - overall_hit_rate) as f64,
            memory_efficiency: 0.85,        // Placeholder
            coordination_overhead_ms: 0.1,  // Placeholder
            deduplication_savings_bytes: 0, // Placeholder
            compression_ratio: 1.2,         // Placeholder
        };

        CombinedCacheStats {
            model_stats: ModelCacheStats {
                basic_stats: model_stats,
                ..Default::default()
            },
            result_stats,
            global_stats,
            health_metrics: HealthMetrics::default(),
            performance_metrics: PerformanceMetrics::default(),
            last_updated: SystemTime::now(),
        }
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<()> {
        info!("Clearing all caches");

        self.model_cache.clear().await?;
        self.result_cache.clear().await?;

        info!("All caches cleared");
        Ok(())
    }

    /// Shutdown cache manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down cache manager");

        // Signal background tasks to stop
        self.task_controller.shutdown_notify.notify_waiters();

        // Wait for tasks to complete
        let task_handles = {
            let mut handles = self.task_controller.task_handles.write().unwrap();
            std::mem::take(&mut *handles)
        };

        for handle in task_handles {
            if let Err(e) = handle.await {
                warn!("Background task failed to shutdown cleanly: {}", e);
            }
        }

        info!("Cache manager shutdown completed");
        Ok(())
    }

    /// Get cache health status
    pub fn get_health_status(&self) -> Vec<HealthCheckResult> {
        let history = self.health_monitor.health_history.read().unwrap();
        history.clone()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<CacheAlert> {
        let alerts = self.health_monitor.active_alerts.read().unwrap();
        alerts.clone()
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> Vec<MetricPoint> {
        let metrics = self.metrics_collector.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get task status
    pub fn get_task_status(&self) -> HashMap<String, TaskStatus> {
        let status = self.task_controller.task_status.read().unwrap();
        status.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cache_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = CacheManagerConfig::default();

        // Disable background tasks for testing
        config.maintenance.enable_auto_maintenance = false;
        config.monitoring.enable_health_monitoring = false;
        config.monitoring.enable_performance_monitoring = false;
        config.global_settings.enable_startup_warming = false;

        let manager = CacheManager::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let stats = manager.get_combined_stats().await;

        assert_eq!(stats.global_stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_cache_manager_shutdown() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = CacheManagerConfig::default();

        // Disable background tasks for testing
        config.maintenance.enable_auto_maintenance = false;
        config.monitoring.enable_health_monitoring = false;
        config.monitoring.enable_performance_monitoring = false;
        config.global_settings.enable_startup_warming = false;

        let manager = CacheManager::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();
        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_combined_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = CacheManagerConfig::default();

        // Disable background tasks for testing
        config.maintenance.enable_auto_maintenance = false;
        config.monitoring.enable_health_monitoring = false;
        config.monitoring.enable_performance_monitoring = false;
        config.global_settings.enable_startup_warming = false;

        let manager = CacheManager::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let stats = manager.get_combined_stats().await;

        assert!(stats.global_stats.overall_hit_rate >= 0.0);
        assert!(stats.global_stats.overall_hit_rate <= 100.0);
    }
}
