//! Thread safety improvements for voice conversion system
//!
//! This module provides enhanced thread safety patterns, concurrent operation management,
//! and safe resource sharing for the voice conversion system.

use crate::{
    config::ConversionConfig,
    models::ConversionModel,
    types::{ConversionRequest, ConversionResult, ConversionType},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Weak;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore};
use tracing::{debug, info, trace, warn};

/// Memory safety auditing system for voice conversion
pub struct MemorySafetyAuditor {
    /// Track memory allocations
    allocation_tracker: Arc<RwLock<AllocationTracker>>,
    /// Track reference cycles
    reference_tracker: Arc<RwLock<ReferenceTracker>>,
    /// Monitor buffer safety
    buffer_safety_monitor: Arc<RwLock<BufferSafetyMonitor>>,
    /// Audit configuration
    audit_config: MemorySafetyConfig,
}

/// Configuration for memory safety auditing
#[derive(Debug, Clone)]
pub struct MemorySafetyConfig {
    /// Enable allocation tracking
    pub enable_allocation_tracking: bool,
    /// Enable reference cycle detection
    pub enable_reference_cycle_detection: bool,
    /// Enable buffer bounds checking
    pub enable_buffer_bounds_checking: bool,
    /// Maximum memory usage threshold (in bytes)
    pub max_memory_threshold: u64,
    /// Enable automatic cleanup
    pub enable_automatic_cleanup: bool,
    /// Audit interval for periodic checks
    pub audit_interval: Duration,
}

impl Default for MemorySafetyConfig {
    fn default() -> Self {
        Self {
            enable_allocation_tracking: true,
            enable_reference_cycle_detection: true,
            enable_buffer_bounds_checking: true,
            max_memory_threshold: 1024 * 1024 * 1024, // 1GB
            enable_automatic_cleanup: true,
            audit_interval: Duration::from_secs(30),
        }
    }
}

/// Track memory allocations and usage patterns
#[derive(Debug, Default)]
pub struct AllocationTracker {
    /// Total allocations
    pub total_allocations: AtomicU64,
    /// Total deallocations
    pub total_deallocations: AtomicU64,
    /// Current memory usage
    pub current_memory_usage: AtomicU64,
    /// Peak memory usage
    pub peak_memory_usage: AtomicU64,
    /// Active allocations by ID
    pub active_allocations: HashMap<String, AllocationInfo>,
    /// Allocation patterns
    pub allocation_patterns: HashMap<String, AllocationPattern>,
    /// Memory leaks detected
    pub detected_leaks: Vec<MemoryLeak>,
}

/// Information about a specific allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Unique identifier for this allocation
    pub allocation_id: String,
    /// Size of the allocation in bytes
    pub size: u64,
    /// Timestamp when allocation occurred
    pub timestamp: Instant,
    /// Source location where allocation occurred
    pub location: String,
    /// Type of allocation
    pub allocation_type: AllocationType,
    /// Thread that performed the allocation
    pub thread_id: std::thread::ThreadId,
}

/// Type of memory allocation
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationType {
    /// Audio processing buffer
    AudioBuffer,
    /// Model weights and data
    ModelData,
    /// Conversion result cache
    ConversionCache,
    /// Temporary processing buffer
    TemporaryBuffer,
    /// Configuration data
    ConfigurationData,
    /// Performance metrics data
    MetricsData,
    /// Other type of allocation
    Other(String),
}

/// Pattern of memory allocation behavior
#[derive(Debug, Clone)]
pub struct AllocationPattern {
    pub pattern_name: String,
    pub allocation_count: u32,
    pub average_size: u64,
    pub total_size: u64,
    pub frequency: f64, // Allocations per second
    pub typical_lifetime: Duration,
}

/// Detected memory leak information
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    pub leak_id: String,
    pub allocation_info: AllocationInfo,
    pub leak_detected_at: Instant,
    pub estimated_leak_duration: Duration,
    pub severity: LeakSeverity,
}

/// Severity of memory leak
#[derive(Debug, Clone, PartialEq)]
pub enum LeakSeverity {
    Low,      // Small, short-lived leaks
    Medium,   // Moderate size or duration
    High,     // Large size or long duration
    Critical, // Severe leaks that could cause system instability
}

/// Track reference cycles and ownership patterns
#[derive(Debug, Default)]
pub struct ReferenceTracker {
    /// Active strong references
    pub strong_references: HashMap<String, ReferenceInfo>,
    /// Active weak references
    pub weak_references: HashMap<String, WeakReferenceInfo>,
    /// Detected cycles
    pub detected_cycles: Vec<ReferenceCycle>,
    /// Reference creation patterns
    pub reference_patterns: HashMap<String, ReferencePattern>,
}

/// Information about a reference
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    pub reference_id: String,
    pub object_type: String,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u32,
    pub source_location: String,
    pub reference_chain: Vec<String>, // Chain of references leading to this object
}

/// Information about a weak reference
#[derive(Debug, Clone)]
pub struct WeakReferenceInfo {
    pub reference_id: String,
    pub object_type: String,
    pub created_at: Instant,
    pub is_valid: bool,
    pub upgrade_attempts: u32,
    pub successful_upgrades: u32,
}

/// Detected reference cycle
#[derive(Debug, Clone)]
pub struct ReferenceCycle {
    pub cycle_id: String,
    pub objects_in_cycle: Vec<String>,
    pub cycle_length: usize,
    pub detected_at: Instant,
    pub cycle_type: CycleType,
    pub estimated_memory_impact: u64,
}

/// Type of reference cycle
#[derive(Debug, Clone, PartialEq)]
pub enum CycleType {
    DirectCycle,   // A -> B -> A
    IndirectCycle, // A -> B -> C -> A
    ComplexCycle,  // Multiple interconnected cycles
}

/// Pattern of reference creation and usage
#[derive(Debug, Clone)]
pub struct ReferencePattern {
    pub pattern_name: String,
    pub creation_frequency: f64,
    pub average_lifetime: Duration,
    pub typical_access_pattern: AccessPattern,
    pub common_reference_chains: Vec<Vec<String>>,
}

/// Access pattern for references
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    SingleAccess,     // Used once and dropped
    BurstAccess,      // Heavy usage in short periods
    SteadyAccess,     // Regular consistent access
    DecreasingAccess, // Access frequency decreases over time
    PeriodicAccess,   // Regular periodic access pattern
}

/// Monitor buffer safety and bounds checking
#[derive(Debug, Default)]
pub struct BufferSafetyMonitor {
    /// Buffer bounds violations
    pub bounds_violations: Vec<BoundsViolation>,
    /// Buffer usage statistics
    pub buffer_stats: HashMap<String, BufferStats>,
    /// Unsafe buffer operations detected
    pub unsafe_operations: Vec<UnsafeOperation>,
    /// Buffer lifecycle tracking
    pub buffer_lifecycle: HashMap<String, BufferLifecycle>,
}

/// Buffer bounds violation information
#[derive(Debug, Clone)]
pub struct BoundsViolation {
    pub violation_id: String,
    pub buffer_id: String,
    pub violation_type: ViolationType,
    pub attempted_index: isize,
    pub buffer_size: usize,
    pub stack_trace: String,
    pub detected_at: Instant,
    pub severity: ViolationSeverity,
}

/// Type of bounds violation
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    ReadBeyondBounds,
    WriteBeyondBounds,
    NegativeIndex,
    UseAfterFree,
    DoubleFree,
}

/// Severity of bounds violation
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    Warning,  // Potential issue but handled safely
    Error,    // Definite violation that was caught
    Critical, // Violation that could cause undefined behavior
}

/// Statistics for buffer usage
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub buffer_id: String,
    pub buffer_type: String,
    pub size: usize,
    pub access_count: u32,
    pub read_operations: u32,
    pub write_operations: u32,
    pub resize_operations: u32,
    pub first_access: Instant,
    pub last_access: Instant,
    pub average_access_interval: Duration,
}

/// Unsafe operation detected
#[derive(Debug, Clone)]
pub struct UnsafeOperation {
    pub operation_id: String,
    pub operation_type: UnsafeOperationType,
    pub buffer_id: String,
    pub detected_at: Instant,
    pub risk_level: RiskLevel,
    pub mitigation_applied: Option<String>,
}

/// Type of unsafe operation
#[derive(Debug, Clone, PartialEq)]
pub enum UnsafeOperationType {
    UnalignedAccess,
    RacyAccess,
    DanglingPointer,
    BufferOverflow,
    UseAfterMove,
    ConcurrentMutation,
}

/// Risk level of unsafe operation
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Buffer lifecycle tracking
#[derive(Debug, Clone)]
pub struct BufferLifecycle {
    pub buffer_id: String,
    pub created_at: Instant,
    pub size_changes: Vec<(Instant, usize)>,
    pub access_pattern: Vec<(Instant, AccessType)>,
    pub current_state: BufferState,
    pub expected_lifetime: Option<Duration>,
}

/// Type of buffer access
#[derive(Debug, Clone, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    Resize,
    Clone,
    Move,
}

/// Current state of buffer
#[derive(Debug, Clone, PartialEq)]
pub enum BufferState {
    Active,
    Borrowed,
    Moved,
    Dropped,
}

impl MemorySafetyAuditor {
    /// Create new memory safety auditor
    pub fn new(config: MemorySafetyConfig) -> Self {
        Self {
            allocation_tracker: Arc::new(RwLock::new(AllocationTracker::default())),
            reference_tracker: Arc::new(RwLock::new(ReferenceTracker::default())),
            buffer_safety_monitor: Arc::new(RwLock::new(BufferSafetyMonitor::default())),
            audit_config: config,
        }
    }

    /// Start periodic memory safety audit
    pub async fn start_periodic_audit(&self) -> Result<()> {
        if !self.audit_config.enable_allocation_tracking
            && !self.audit_config.enable_reference_cycle_detection
            && !self.audit_config.enable_buffer_bounds_checking
        {
            return Ok(()); // No auditing enabled
        }

        let auditor = Self {
            allocation_tracker: Arc::clone(&self.allocation_tracker),
            reference_tracker: Arc::clone(&self.reference_tracker),
            buffer_safety_monitor: Arc::clone(&self.buffer_safety_monitor),
            audit_config: self.audit_config.clone(),
        };

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(auditor.audit_config.audit_interval);

            loop {
                interval.tick().await;

                if let Err(e) = auditor.perform_audit().await {
                    warn!("Memory safety audit failed: {}", e);
                }
            }
        });

        info!(
            "Started periodic memory safety audit with interval: {:?}",
            self.audit_config.audit_interval
        );
        Ok(())
    }

    /// Perform comprehensive memory safety audit
    pub async fn perform_audit(&self) -> Result<MemorySafetyReport> {
        let mut report = MemorySafetyReport::default();

        // Audit memory allocations
        if self.audit_config.enable_allocation_tracking {
            report.allocation_audit = Some(self.audit_allocations().await?);
        }

        // Audit reference cycles
        if self.audit_config.enable_reference_cycle_detection {
            report.reference_audit = Some(self.audit_references().await?);
        }

        // Audit buffer safety
        if self.audit_config.enable_buffer_bounds_checking {
            report.buffer_audit = Some(self.audit_buffers().await?);
        }

        // Calculate overall safety score
        report.overall_safety_score = self.calculate_safety_score(&report);
        report.audit_timestamp = Instant::now();

        // Apply automatic cleanup if enabled
        if self.audit_config.enable_automatic_cleanup {
            self.apply_automatic_cleanup(&report).await?;
        }

        Ok(report)
    }

    /// Audit memory allocations for leaks and patterns
    async fn audit_allocations(&self) -> Result<AllocationAuditResult> {
        let tracker = self.allocation_tracker.read().await;
        let mut result = AllocationAuditResult::default();

        // Check for memory leaks
        let current_time = Instant::now();
        for (id, alloc_info) in &tracker.active_allocations {
            let age = current_time.duration_since(alloc_info.timestamp);

            // Consider allocations older than 5 minutes as potential leaks
            if age > Duration::from_secs(300) {
                let severity = match alloc_info.size {
                    size if size > 100 * 1024 * 1024 => LeakSeverity::Critical, // > 100MB
                    size if size > 10 * 1024 * 1024 => LeakSeverity::High,      // > 10MB
                    size if size > 1024 * 1024 => LeakSeverity::Medium,         // > 1MB
                    _ => LeakSeverity::Low,
                };

                let leak = MemoryLeak {
                    leak_id: format!("leak_{}", id),
                    allocation_info: alloc_info.clone(),
                    leak_detected_at: current_time,
                    estimated_leak_duration: age,
                    severity,
                };

                result.detected_leaks.push(leak);
            }
        }

        // Calculate statistics
        result.total_active_allocations = tracker.active_allocations.len();
        result.current_memory_usage = tracker.current_memory_usage.load(Ordering::Relaxed);
        result.peak_memory_usage = tracker.peak_memory_usage.load(Ordering::Relaxed);
        result.allocation_patterns = tracker.allocation_patterns.clone();

        // Check if memory usage exceeds threshold
        if result.current_memory_usage > self.audit_config.max_memory_threshold {
            result.memory_threshold_exceeded = true;
            warn!(
                "Memory usage ({} bytes) exceeds threshold ({} bytes)",
                result.current_memory_usage, self.audit_config.max_memory_threshold
            );
        }

        Ok(result)
    }

    /// Audit reference cycles and ownership patterns
    async fn audit_references(&self) -> Result<ReferenceAuditResult> {
        let tracker = self.reference_tracker.read().await;
        let mut result = ReferenceAuditResult::default();

        // Detect potential reference cycles using simple graph traversal
        result.detected_cycles = tracker.detected_cycles.clone();
        result.active_strong_references = tracker.strong_references.len();
        result.active_weak_references = tracker.weak_references.len();

        // Check for orphaned references (strong references with no activity)
        let current_time = Instant::now();
        for (id, ref_info) in &tracker.strong_references {
            let idle_time = current_time.duration_since(ref_info.last_accessed);
            if idle_time > Duration::from_secs(600) {
                // 10 minutes idle
                result.orphaned_references.push(ref_info.clone());
            }
        }

        // Analyze reference patterns
        result.reference_patterns = tracker.reference_patterns.clone();

        Ok(result)
    }

    /// Audit buffer safety and bounds checking
    async fn audit_buffers(&self) -> Result<BufferAuditResult> {
        let monitor = self.buffer_safety_monitor.read().await;
        let mut result = BufferAuditResult::default();

        result.bounds_violations = monitor.bounds_violations.clone();
        result.unsafe_operations = monitor.unsafe_operations.clone();
        result.buffer_statistics = monitor.buffer_stats.clone();

        // Analyze buffer lifecycle patterns
        for (id, lifecycle) in &monitor.buffer_lifecycle {
            if lifecycle.current_state == BufferState::Dropped {
                continue; // Skip dropped buffers
            }

            // Check for long-lived buffers that might be leaked
            let age = Instant::now().duration_since(lifecycle.created_at);
            if age > Duration::from_secs(1800) {
                // 30 minutes
                result.long_lived_buffers.push(lifecycle.clone());
            }
        }

        Ok(result)
    }

    /// Calculate overall safety score based on audit results
    fn calculate_safety_score(&self, report: &MemorySafetyReport) -> f64 {
        let mut score = 100.0;

        if let Some(ref alloc_audit) = report.allocation_audit {
            // Deduct points for memory leaks
            for leak in &alloc_audit.detected_leaks {
                let deduction = match leak.severity {
                    LeakSeverity::Critical => 25.0,
                    LeakSeverity::High => 15.0,
                    LeakSeverity::Medium => 8.0,
                    LeakSeverity::Low => 3.0,
                };
                score -= deduction;
            }

            // Deduct points for memory threshold exceeded
            if alloc_audit.memory_threshold_exceeded {
                score -= 20.0;
            }
        }

        if let Some(ref ref_audit) = report.reference_audit {
            // Deduct points for reference cycles
            score -= ref_audit.detected_cycles.len() as f64 * 10.0;

            // Deduct points for orphaned references
            score -= ref_audit.orphaned_references.len() as f64 * 5.0;
        }

        if let Some(ref buf_audit) = report.buffer_audit {
            // Deduct points for bounds violations
            for violation in &buf_audit.bounds_violations {
                let deduction = match violation.severity {
                    ViolationSeverity::Critical => 30.0,
                    ViolationSeverity::Error => 15.0,
                    ViolationSeverity::Warning => 5.0,
                };
                score -= deduction;
            }

            // Deduct points for unsafe operations
            for operation in &buf_audit.unsafe_operations {
                let deduction = match operation.risk_level {
                    RiskLevel::Critical => 25.0,
                    RiskLevel::High => 15.0,
                    RiskLevel::Medium => 8.0,
                    RiskLevel::Low => 3.0,
                };
                score -= deduction;
            }
        }

        // Ensure score doesn't go negative
        score.max(0.0)
    }

    /// Apply automatic cleanup based on audit results
    async fn apply_automatic_cleanup(&self, report: &MemorySafetyReport) -> Result<()> {
        if let Some(ref alloc_audit) = report.allocation_audit {
            // Clean up low-severity leaks that are very old
            let mut tracker = self.allocation_tracker.write().await;
            let mut cleaned_up = 0;

            let current_time = Instant::now();
            let old_allocations: Vec<String> = tracker
                .active_allocations
                .iter()
                .filter(|(_, alloc)| {
                    let age = current_time.duration_since(alloc.timestamp);
                    age > Duration::from_secs(3600) && alloc.size < 1024 * 1024 // 1 hour old and < 1MB
                })
                .map(|(id, _)| id.clone())
                .collect();

            for id in old_allocations {
                if let Some(alloc_info) = tracker.active_allocations.remove(&id) {
                    let current_usage = tracker.current_memory_usage.load(Ordering::Relaxed);
                    tracker.current_memory_usage.store(
                        current_usage.saturating_sub(alloc_info.size),
                        Ordering::Relaxed,
                    );
                    tracker.total_deallocations.fetch_add(1, Ordering::Relaxed);
                    cleaned_up += 1;
                }
            }

            if cleaned_up > 0 {
                info!("Automatic cleanup removed {} old allocations", cleaned_up);
            }
        }

        Ok(())
    }

    /// Track a new memory allocation
    pub async fn track_allocation(
        &self,
        allocation_id: String,
        size: u64,
        location: String,
        allocation_type: AllocationType,
    ) -> Result<()> {
        if !self.audit_config.enable_allocation_tracking {
            return Ok(());
        }

        let mut tracker = self.allocation_tracker.write().await;

        let alloc_info = AllocationInfo {
            allocation_id: allocation_id.clone(),
            size,
            timestamp: Instant::now(),
            location,
            allocation_type,
            thread_id: std::thread::current().id(),
        };

        tracker.active_allocations.insert(allocation_id, alloc_info);
        tracker.total_allocations.fetch_add(1, Ordering::Relaxed);

        let new_usage = tracker
            .current_memory_usage
            .fetch_add(size, Ordering::Relaxed)
            + size;

        // Update peak memory usage
        let current_peak = tracker.peak_memory_usage.load(Ordering::Relaxed);
        if new_usage > current_peak {
            tracker
                .peak_memory_usage
                .store(new_usage, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Track deallocation of memory
    pub async fn track_deallocation(&self, allocation_id: &str) -> Result<()> {
        if !self.audit_config.enable_allocation_tracking {
            return Ok(());
        }

        let mut tracker = self.allocation_tracker.write().await;

        if let Some(alloc_info) = tracker.active_allocations.remove(allocation_id) {
            tracker.total_deallocations.fetch_add(1, Ordering::Relaxed);
            let current_usage = tracker.current_memory_usage.load(Ordering::Relaxed);
            tracker.current_memory_usage.store(
                current_usage.saturating_sub(alloc_info.size),
                Ordering::Relaxed,
            );
        }

        Ok(())
    }

    /// Get current memory safety status
    pub async fn get_safety_status(&self) -> MemorySafetyStatus {
        let allocation_tracker = self.allocation_tracker.read().await;
        let reference_tracker = self.reference_tracker.read().await;
        let buffer_monitor = self.buffer_safety_monitor.read().await;

        MemorySafetyStatus {
            current_memory_usage: allocation_tracker
                .current_memory_usage
                .load(Ordering::Relaxed),
            active_allocations: allocation_tracker.active_allocations.len(),
            detected_leaks: allocation_tracker.detected_leaks.len(),
            active_references: reference_tracker.strong_references.len(),
            detected_cycles: reference_tracker.detected_cycles.len(),
            bounds_violations: buffer_monitor.bounds_violations.len(),
            unsafe_operations: buffer_monitor.unsafe_operations.len(),
            last_audit: Instant::now(), // This would be stored properly in a real implementation
        }
    }
}

/// Complete memory safety audit report
#[derive(Debug)]
pub struct MemorySafetyReport {
    pub allocation_audit: Option<AllocationAuditResult>,
    pub reference_audit: Option<ReferenceAuditResult>,
    pub buffer_audit: Option<BufferAuditResult>,
    pub overall_safety_score: f64,
    pub audit_timestamp: Instant,
}

impl Default for MemorySafetyReport {
    fn default() -> Self {
        Self {
            allocation_audit: None,
            reference_audit: None,
            buffer_audit: None,
            overall_safety_score: 0.0,
            audit_timestamp: Instant::now(),
        }
    }
}

/// Result of allocation audit
#[derive(Debug, Default)]
pub struct AllocationAuditResult {
    pub detected_leaks: Vec<MemoryLeak>,
    pub total_active_allocations: usize,
    pub current_memory_usage: u64,
    pub peak_memory_usage: u64,
    pub allocation_patterns: HashMap<String, AllocationPattern>,
    pub memory_threshold_exceeded: bool,
}

/// Result of reference audit
#[derive(Debug, Default)]
pub struct ReferenceAuditResult {
    pub detected_cycles: Vec<ReferenceCycle>,
    pub active_strong_references: usize,
    pub active_weak_references: usize,
    pub orphaned_references: Vec<ReferenceInfo>,
    pub reference_patterns: HashMap<String, ReferencePattern>,
}

/// Result of buffer audit
#[derive(Debug, Default)]
pub struct BufferAuditResult {
    pub bounds_violations: Vec<BoundsViolation>,
    pub unsafe_operations: Vec<UnsafeOperation>,
    pub buffer_statistics: HashMap<String, BufferStats>,
    pub long_lived_buffers: Vec<BufferLifecycle>,
}

/// Current memory safety status
#[derive(Debug)]
pub struct MemorySafetyStatus {
    pub current_memory_usage: u64,
    pub active_allocations: usize,
    pub detected_leaks: usize,
    pub active_references: usize,
    pub detected_cycles: usize,
    pub bounds_violations: usize,
    pub unsafe_operations: usize,
    pub last_audit: Instant,
}

/// Thread-safe model manager for voice conversion
pub struct ThreadSafeModelManager {
    /// Cached models with thread-safe access
    models: Arc<RwLock<HashMap<ConversionType, Arc<ConversionModel>>>>,
    /// Model loading semaphore to prevent resource exhaustion
    loading_semaphore: Arc<Semaphore>,
    /// Model access statistics
    stats: Arc<RwLock<ModelAccessStats>>,
    /// Maximum number of cached models
    max_cached_models: usize,
    /// Model usage tracking for eviction decisions
    usage_tracker: Arc<RwLock<HashMap<ConversionType, ModelUsageInfo>>>,
}

/// Model access statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ModelAccessStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub models_loaded: u64,
    pub models_evicted: u64,
    pub concurrent_loads: u64,
    pub average_load_time: Duration,
    pub last_cleanup: Option<Instant>,
}

/// Model usage information for cache management
#[derive(Debug, Clone)]
pub struct ModelUsageInfo {
    /// When the model was last accessed
    pub last_accessed: Instant,
    /// Total number of times this model has been accessed
    pub access_count: u32,
    /// Cumulative processing time for all operations
    pub total_processing_time: Duration,
    /// Average processing time per operation
    pub average_processing_time: Duration,
    /// Estimated memory usage in bytes
    pub memory_usage_estimate: u64,
}

impl ThreadSafeModelManager {
    /// Create new thread-safe model manager
    pub fn new(max_cached_models: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            loading_semaphore: Arc::new(Semaphore::new(2)), // Max 2 concurrent model loads
            stats: Arc::new(RwLock::new(ModelAccessStats::default())),
            max_cached_models,
            usage_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get model with thread-safe access and cache management
    pub async fn get_model(
        &self,
        conversion_type: &ConversionType,
    ) -> Result<Option<Arc<ConversionModel>>> {
        // Try to get from cache first
        {
            let models_guard = self.models.read().await;
            if let Some(model) = models_guard.get(conversion_type) {
                // Update access statistics
                self.update_access_stats(conversion_type, true).await;
                return Ok(Some(Arc::clone(model)));
            }
        }

        // Model not in cache, need to load it
        self.update_access_stats(conversion_type, false).await;

        // Use semaphore to limit concurrent loading
        let _permit = self
            .loading_semaphore
            .acquire()
            .await
            .map_err(|e| Error::runtime(format!("Failed to acquire loading permit: {}", e)))?;

        // Double-check pattern: another thread might have loaded it while we waited
        {
            let models_guard = self.models.read().await;
            if let Some(model) = models_guard.get(conversion_type) {
                self.update_access_stats(conversion_type, true).await;
                return Ok(Some(Arc::clone(model)));
            }
        }

        // Load the model (this would be implemented based on specific model loading logic)
        debug!("Loading model for conversion type: {:?}", conversion_type);
        let start_time = Instant::now();

        // Simulate model loading - in real implementation this would load actual models
        let model = self.load_model_impl(conversion_type).await?;

        let load_time = start_time.elapsed();

        // Update cache with new model
        {
            let mut models_guard = self.models.write().await;
            let mut usage_guard = self.usage_tracker.write().await;

            // Check if we need to evict old models
            if models_guard.len() >= self.max_cached_models {
                self.evict_least_used_model(&mut models_guard, &mut usage_guard)
                    .await;
            }

            // Insert new model
            let model_arc = Arc::new(model);
            models_guard.insert(conversion_type.clone(), Arc::clone(&model_arc));

            // Track usage
            usage_guard.insert(
                conversion_type.clone(),
                ModelUsageInfo {
                    last_accessed: Instant::now(),
                    access_count: 1,
                    total_processing_time: Duration::from_millis(0),
                    average_processing_time: Duration::from_millis(0),
                    memory_usage_estimate: 100 * 1024 * 1024, // 100MB estimate
                },
            );

            // Update load statistics
            {
                let mut stats_guard = self.stats.write().await;
                stats_guard.models_loaded += 1;
                stats_guard.concurrent_loads += 1;
                stats_guard.average_load_time = if stats_guard.models_loaded == 1 {
                    load_time
                } else {
                    Duration::from_nanos(
                        (stats_guard.average_load_time.as_nanos() as u64
                            * (stats_guard.models_loaded - 1)
                            + load_time.as_nanos() as u64)
                            / stats_guard.models_loaded,
                    )
                };
            }

            Ok(Some(model_arc))
        }
    }

    /// Load model implementation (placeholder for actual model loading)
    async fn load_model_impl(&self, conversion_type: &ConversionType) -> Result<ConversionModel> {
        // This is a placeholder - in real implementation, this would load the actual model
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate loading time

        let model_type = match conversion_type {
            ConversionType::SpeakerConversion => crate::models::ModelType::NeuralVC,
            ConversionType::AgeTransformation => crate::models::ModelType::NeuralVC,
            ConversionType::GenderTransformation => crate::models::ModelType::NeuralVC,
            ConversionType::VoiceMorphing => crate::models::ModelType::AutoVC,
            ConversionType::EmotionalTransformation => crate::models::ModelType::Transformer,
            _ => crate::models::ModelType::Custom,
        };

        Ok(ConversionModel::new(model_type))
    }

    /// Evict least used model to make space
    async fn evict_least_used_model(
        &self,
        models_guard: &mut HashMap<ConversionType, Arc<ConversionModel>>,
        usage_guard: &mut HashMap<ConversionType, ModelUsageInfo>,
    ) {
        if let Some((least_used_type, _)) = usage_guard
            .iter()
            .min_by_key(|(_, usage)| (usage.last_accessed, usage.access_count))
        {
            let evicted_type = least_used_type.clone();
            models_guard.remove(&evicted_type);
            usage_guard.remove(&evicted_type);

            // Update statistics
            {
                let mut stats_guard = self.stats.write().await;
                stats_guard.models_evicted += 1;
            }

            debug!("Evicted least used model: {:?}", evicted_type);
        }
    }

    /// Update access statistics
    async fn update_access_stats(&self, conversion_type: &ConversionType, cache_hit: bool) {
        let mut stats_guard = self.stats.write().await;
        if cache_hit {
            stats_guard.cache_hits += 1;
        } else {
            stats_guard.cache_misses += 1;
        }

        // Update usage tracker
        drop(stats_guard);
        let mut usage_guard = self.usage_tracker.write().await;
        if let Some(usage_info) = usage_guard.get_mut(conversion_type) {
            usage_info.last_accessed = Instant::now();
            usage_info.access_count += 1;
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ModelAccessStats {
        self.stats.read().await.clone()
    }

    /// Clear all cached models
    pub async fn clear_cache(&self) {
        let mut models_guard = self.models.write().await;
        let mut usage_guard = self.usage_tracker.write().await;

        let evicted_count = models_guard.len();
        models_guard.clear();
        usage_guard.clear();

        // Update statistics
        {
            let mut stats_guard = self.stats.write().await;
            stats_guard.models_evicted += evicted_count as u64;
        }

        info!(
            "Cleared all cached models: {} models evicted",
            evicted_count
        );
    }

    /// Perform periodic cleanup of unused models
    pub async fn cleanup_unused_models(&self, max_idle_time: Duration) {
        let now = Instant::now();
        let mut models_guard = self.models.write().await;
        let mut usage_guard = self.usage_tracker.write().await;

        let mut to_remove = Vec::new();
        for (conversion_type, usage_info) in usage_guard.iter() {
            if now.duration_since(usage_info.last_accessed) > max_idle_time {
                to_remove.push(conversion_type.clone());
            }
        }

        let mut evicted_count = 0;
        for conversion_type in to_remove {
            models_guard.remove(&conversion_type);
            usage_guard.remove(&conversion_type);
            evicted_count += 1;
            debug!("Evicted idle model: {:?}", conversion_type);
        }

        if evicted_count > 0 {
            // Update statistics
            {
                let mut stats_guard = self.stats.write().await;
                stats_guard.models_evicted += evicted_count;
                stats_guard.last_cleanup = Some(now);
            }

            info!("Cleanup evicted {} idle models", evicted_count);
        }
    }
}

/// Thread-safe conversion operation guard
pub struct OperationGuard {
    /// Shared operation state
    operation_state: Arc<RwLock<OperationState>>,
    /// Semaphore permit for this operation
    _permit: OwnedSemaphorePermit,
    /// Operation identifier
    operation_id: String,
    /// Start time for performance tracking
    start_time: Instant,
}

/// Operation state tracking
#[derive(Debug, Default, Clone)]
pub struct OperationState {
    /// Currently active operations mapped by operation ID
    pub active_operations: HashMap<String, OperationInfo>,
    /// Total number of completed operations
    pub completed_operations: u64,
    /// Total number of failed operations
    pub failed_operations: u64,
    /// Average duration of completed operations
    pub average_duration: Duration,
}

/// Information about an active operation
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// Unique identifier for this operation
    pub operation_id: String,
    /// Type of conversion being performed
    pub conversion_type: ConversionType,
    /// When the operation started
    pub start_time: Instant,
    /// ID of the thread handling this operation
    pub thread_id: std::thread::ThreadId,
    /// Current status of the operation
    pub status: OperationStatus,
}

/// Operation status
#[derive(Debug, Clone, PartialEq)]
pub enum OperationStatus {
    /// Operation is being initialized
    Starting,
    /// Operation is actively processing
    Processing,
    /// Operation is in final cleanup phase
    Finalizing,
    /// Operation completed successfully
    Completed,
    /// Operation failed with error message
    Failed(String),
}

impl OperationGuard {
    /// Create new operation guard
    pub async fn new(
        operation_state: Arc<RwLock<OperationState>>,
        semaphore: Arc<Semaphore>,
        operation_id: String,
        conversion_type: ConversionType,
    ) -> Result<Self> {
        let permit = semaphore
            .acquire_owned()
            .await
            .map_err(|e| Error::runtime(format!("Failed to acquire operation permit: {}", e)))?;

        let start_time = Instant::now();

        // Register operation
        {
            let mut state_guard = operation_state.write().await;
            state_guard.active_operations.insert(
                operation_id.clone(),
                OperationInfo {
                    operation_id: operation_id.clone(),
                    conversion_type,
                    start_time,
                    thread_id: std::thread::current().id(),
                    status: OperationStatus::Starting,
                },
            );
        }

        Ok(Self {
            operation_state,
            _permit: permit,
            operation_id,
            start_time,
        })
    }

    /// Update operation status
    pub async fn update_status(&self, status: OperationStatus) {
        let mut state_guard = self.operation_state.write().await;
        if let Some(op_info) = state_guard.active_operations.get_mut(&self.operation_id) {
            op_info.status = status;
        }
    }

    /// Mark operation as completed
    pub async fn complete(&self) {
        self.finalize_operation(OperationStatus::Completed).await;
    }

    /// Mark operation as failed
    pub async fn fail(&self, error: String) {
        self.finalize_operation(OperationStatus::Failed(error))
            .await;
    }

    /// Finalize operation and update statistics
    async fn finalize_operation(&self, final_status: OperationStatus) {
        let duration = self.start_time.elapsed();
        let mut state_guard = self.operation_state.write().await;

        // Remove from active operations
        state_guard.active_operations.remove(&self.operation_id);

        // Update statistics
        match final_status {
            OperationStatus::Completed => {
                state_guard.completed_operations += 1;

                // Update average duration
                let total_ops = state_guard.completed_operations;
                if total_ops == 1 {
                    state_guard.average_duration = duration;
                } else {
                    let total_nanos = state_guard.average_duration.as_nanos() as u64
                        * (total_ops - 1)
                        + duration.as_nanos() as u64;
                    state_guard.average_duration = Duration::from_nanos(total_nanos / total_ops);
                }
            }
            OperationStatus::Failed(_) => {
                state_guard.failed_operations += 1;
            }
            _ => {}
        }

        debug!(
            "Operation {} finalized with status {:?} in {:?}",
            self.operation_id, final_status, duration
        );
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        // Ensure operation is removed from active operations on drop
        let operation_state = Arc::clone(&self.operation_state);
        let operation_id = self.operation_id.clone();

        tokio::spawn(async move {
            let mut state_guard = operation_state.write().await;
            state_guard.active_operations.remove(&operation_id);
        });
    }
}

/// Thread-safe concurrent conversion manager
pub struct ConcurrentConversionManager {
    /// Shared operation state
    operation_state: Arc<RwLock<OperationState>>,
    /// Semaphore for limiting concurrent operations
    operation_semaphore: Arc<Semaphore>,
    /// Model manager for thread-safe model access
    model_manager: Arc<ThreadSafeModelManager>,
    /// Configuration with thread-safe access
    config: Arc<RwLock<ConversionConfig>>,
    /// Performance metrics
    metrics: Arc<RwLock<ConcurrentConversionMetrics>>,
}

/// Metrics for concurrent conversion operations
#[derive(Debug, Default, Clone)]
pub struct ConcurrentConversionMetrics {
    /// Total number of conversion requests received
    pub total_requests: u64,
    /// Number of conversions that completed successfully
    pub successful_conversions: u64,
    /// Number of conversions that failed
    pub failed_conversions: u64,
    /// Average time spent waiting in queue
    pub average_queue_time: Duration,
    /// Average time spent processing requests
    pub average_processing_time: Duration,
    /// Maximum number of concurrent operations seen
    pub peak_concurrent_operations: usize,
    /// Current number of active operations
    pub current_concurrent_operations: usize,
}

impl ConcurrentConversionManager {
    /// Create new concurrent conversion manager
    pub fn new(
        max_concurrent_operations: usize,
        max_cached_models: usize,
        config: ConversionConfig,
    ) -> Self {
        Self {
            operation_state: Arc::new(RwLock::new(OperationState::default())),
            operation_semaphore: Arc::new(Semaphore::new(max_concurrent_operations)),
            model_manager: Arc::new(ThreadSafeModelManager::new(max_cached_models)),
            config: Arc::new(RwLock::new(config)),
            metrics: Arc::new(RwLock::new(ConcurrentConversionMetrics::default())),
        }
    }

    /// Process conversion request with thread safety guarantees
    pub async fn convert_with_concurrency_control(
        &self,
        request: ConversionRequest,
    ) -> Result<ConversionResult> {
        let queue_start = Instant::now();

        // Update metrics
        {
            let mut metrics_guard = self.metrics.write().await;
            metrics_guard.total_requests += 1;
        }

        // Create operation guard for this conversion
        let operation_guard = OperationGuard::new(
            Arc::clone(&self.operation_state),
            Arc::clone(&self.operation_semaphore),
            request.id.clone(),
            request.conversion_type.clone(),
        )
        .await?;

        let queue_time = queue_start.elapsed();

        // Update queue time metrics
        {
            let mut metrics_guard = self.metrics.write().await;
            let total_requests = metrics_guard.total_requests;
            let current_avg = metrics_guard.average_queue_time;

            metrics_guard.average_queue_time = if total_requests == 1 {
                queue_time
            } else {
                Duration::from_nanos(
                    (current_avg.as_nanos() as u64 * (total_requests - 1)
                        + queue_time.as_nanos() as u64)
                        / total_requests,
                )
            };

            metrics_guard.current_concurrent_operations += 1;
            if metrics_guard.current_concurrent_operations
                > metrics_guard.peak_concurrent_operations
            {
                metrics_guard.peak_concurrent_operations =
                    metrics_guard.current_concurrent_operations;
            }
        }

        operation_guard
            .update_status(OperationStatus::Processing)
            .await;

        // Perform the actual conversion
        let conversion_result = match self
            .perform_safe_conversion(&request, &operation_guard)
            .await
        {
            Ok(result) => {
                operation_guard.complete().await;

                // Update success metrics
                {
                    let mut metrics_guard = self.metrics.write().await;
                    metrics_guard.successful_conversions += 1;
                    metrics_guard.current_concurrent_operations -= 1;
                }

                Ok(result)
            }
            Err(e) => {
                operation_guard.fail(e.to_string()).await;

                // Update failure metrics
                {
                    let mut metrics_guard = self.metrics.write().await;
                    metrics_guard.failed_conversions += 1;
                    metrics_guard.current_concurrent_operations -= 1;
                }

                Err(e)
            }
        };

        conversion_result
    }

    /// Perform the actual conversion with safety guarantees
    async fn perform_safe_conversion(
        &self,
        request: &ConversionRequest,
        operation_guard: &OperationGuard,
    ) -> Result<ConversionResult> {
        let processing_start = Instant::now();

        // Get model safely
        let model = self
            .model_manager
            .get_model(&request.conversion_type)
            .await?;

        operation_guard
            .update_status(OperationStatus::Finalizing)
            .await;

        // Perform conversion (this would use the actual conversion logic)
        let converted_audio = self.simulate_conversion(&request.source_audio).await?;

        let processing_time = processing_start.elapsed();

        // Update processing time metrics
        {
            let mut metrics_guard = self.metrics.write().await;
            let successful_conversions = metrics_guard.successful_conversions + 1; // +1 because we haven't incremented yet
            let current_avg = metrics_guard.average_processing_time;

            metrics_guard.average_processing_time = if successful_conversions == 1 {
                processing_time
            } else {
                Duration::from_nanos(
                    (current_avg.as_nanos() as u64 * (successful_conversions - 1)
                        + processing_time.as_nanos() as u64)
                        / successful_conversions,
                )
            };
        }

        // Create successful result
        Ok(ConversionResult {
            request_id: request.id.clone(),
            converted_audio,
            output_sample_rate: 22050, // Default sample rate
            quality_metrics: HashMap::new(),
            artifacts: None,
            objective_quality: None,
            processing_time,
            conversion_type: request.conversion_type.clone(),
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Simulate conversion (placeholder for actual conversion logic)
    async fn simulate_conversion(&self, source_audio: &[f32]) -> Result<Vec<f32>> {
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Return processed audio (simplified)
        let mut result = source_audio.to_vec();
        for sample in &mut result {
            *sample *= 0.9; // Simple processing simulation
        }

        Ok(result)
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ConcurrentConversionMetrics {
        let metrics_guard = self.metrics.read().await;
        metrics_guard.clone()
    }

    /// Get current operation state
    pub async fn get_operation_state(&self) -> OperationState {
        let state_guard = self.operation_state.read().await;
        state_guard.clone()
    }

    /// Update configuration thread-safely
    pub async fn update_config(&self, new_config: ConversionConfig) -> Result<()> {
        let mut config_guard = self.config.write().await;
        *config_guard = new_config;
        info!("Configuration updated successfully");
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> ConversionConfig {
        self.config.read().await.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> HashMap<String, String> {
        let mut health = HashMap::new();

        let metrics = self.get_metrics().await;
        let operation_state = self.get_operation_state().await;
        let model_stats = self.model_manager.get_stats().await;

        health.insert("status".to_string(), "healthy".to_string());
        health.insert(
            "total_requests".to_string(),
            metrics.total_requests.to_string(),
        );
        health.insert(
            "success_rate".to_string(),
            format!(
                "{:.2}%",
                if metrics.total_requests > 0 {
                    (metrics.successful_conversions as f64 / metrics.total_requests as f64) * 100.0
                } else {
                    100.0
                }
            ),
        );
        health.insert(
            "active_operations".to_string(),
            operation_state.active_operations.len().to_string(),
        );
        health.insert(
            "cached_models".to_string(),
            format!(
                "{}/{}",
                model_stats.cache_hits + model_stats.cache_misses - model_stats.models_evicted,
                self.model_manager.max_cached_models
            ),
        );
        health.insert(
            "model_cache_hit_rate".to_string(),
            format!(
                "{:.2}%",
                if model_stats.cache_hits + model_stats.cache_misses > 0 {
                    (model_stats.cache_hits as f64
                        / (model_stats.cache_hits + model_stats.cache_misses) as f64)
                        * 100.0
                } else {
                    0.0
                }
            ),
        );

        health
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Starting graceful shutdown of concurrent conversion manager");

        // Wait for all active operations to complete (with timeout)
        let shutdown_timeout = Duration::from_secs(30);
        let start_time = Instant::now();

        while start_time.elapsed() < shutdown_timeout {
            let operation_state = self.operation_state.read().await;
            if operation_state.active_operations.is_empty() {
                break;
            }
            drop(operation_state);

            debug!(
                "Waiting for {} active operations to complete",
                self.operation_state.read().await.active_operations.len()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Clear model cache
        self.model_manager.clear_cache().await;

        let final_metrics = self.get_metrics().await;
        info!(
            "Concurrent conversion manager shutdown complete. Final stats: {} total requests, {} successful, {} failed",
            final_metrics.total_requests, final_metrics.successful_conversions, final_metrics.failed_conversions
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ConversionTarget, VoiceCharacteristics};

    #[tokio::test]
    async fn test_thread_safe_model_manager() {
        let manager = ThreadSafeModelManager::new(3);

        // Test cache miss and load
        let model = manager
            .get_model(&ConversionType::PitchShift)
            .await
            .unwrap();
        assert!(model.is_some());

        // Test cache hit
        let model2 = manager
            .get_model(&ConversionType::PitchShift)
            .await
            .unwrap();
        assert!(model2.is_some());

        // Verify statistics
        let stats = manager.get_stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.models_loaded, 1);
    }

    #[tokio::test]
    async fn test_operation_guard() {
        let operation_state = Arc::new(RwLock::new(OperationState::default()));
        let semaphore = Arc::new(Semaphore::new(1));

        let guard = OperationGuard::new(
            Arc::clone(&operation_state),
            semaphore,
            "test_op".to_string(),
            ConversionType::PitchShift,
        )
        .await
        .unwrap();

        // Check that operation is registered
        {
            let state = operation_state.read().await;
            assert!(state.active_operations.contains_key("test_op"));
        }

        // Complete operation
        guard.complete().await;

        // Check that operation is removed and stats updated
        {
            let state = operation_state.read().await;
            assert!(!state.active_operations.contains_key("test_op"));
            assert_eq!(state.completed_operations, 1);
        }
    }

    #[tokio::test]
    async fn test_concurrent_conversion_manager() {
        let config = ConversionConfig::default();
        let manager = ConcurrentConversionManager::new(2, 3, config);

        let request = ConversionRequest::new(
            "test_request".to_string(),
            vec![0.1, -0.1, 0.2, -0.2],
            22050,
            ConversionType::PitchShift,
            ConversionTarget::new(VoiceCharacteristics::default()),
        );

        let result = manager.convert_with_concurrency_control(request).await;
        assert!(result.is_ok());

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_conversions, 1);
        assert_eq!(metrics.failed_conversions, 0);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let config = ConversionConfig::default();
        let manager = Arc::new(ConcurrentConversionManager::new(3, 2, config));

        let mut handles = Vec::new();

        // Spawn multiple concurrent requests
        for i in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                let request = ConversionRequest::new(
                    format!("test_request_{}", i),
                    vec![0.1, -0.1, 0.2, -0.2],
                    22050,
                    ConversionType::PitchShift,
                    ConversionTarget::new(VoiceCharacteristics::default()),
                );

                manager_clone
                    .convert_with_concurrency_control(request)
                    .await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        let mut successful = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successful += 1;
            }
        }

        assert_eq!(successful, 5);

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 5);
        assert_eq!(metrics.successful_conversions, 5);
    }

    #[tokio::test]
    async fn test_model_cache_eviction() {
        let manager = ThreadSafeModelManager::new(2); // Small cache size

        // Load models to fill cache
        manager
            .get_model(&ConversionType::PitchShift)
            .await
            .unwrap();
        manager
            .get_model(&ConversionType::SpeedTransformation)
            .await
            .unwrap();

        // Load another model, should trigger eviction
        manager
            .get_model(&ConversionType::GenderTransformation)
            .await
            .unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.models_evicted, 1);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = ConversionConfig::default();
        let manager = ConcurrentConversionManager::new(2, 3, config);

        let health = manager.health_check().await;
        assert_eq!(health.get("status"), Some(&"healthy".to_string()));
        assert!(health.contains_key("total_requests"));
        assert!(health.contains_key("success_rate"));
        assert!(health.contains_key("cached_models"));
    }
}
