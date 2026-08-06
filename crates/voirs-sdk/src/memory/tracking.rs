//! Resource tracking and monitoring for memory management
//!
//! Provides comprehensive tracking of memory usage, leak detection,
//! and resource lifecycle management for VoiRS SDK components.

use backtrace::Backtrace;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// Resource statistics for monitoring
#[derive(Debug, Clone)]
pub struct ResourceStats {
    /// Current memory usage in bytes
    pub memory_usage: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
    /// Current active allocations
    pub active_allocations: u64,
    /// Memory fragmentation ratio (0.0 - 1.0)
    pub fragmentation_ratio: f64,
    /// Average allocation size
    pub avg_allocation_size: u64,
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Deallocation rate (deallocations per second)
    pub deallocation_rate: f64,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl Default for ResourceStats {
    fn default() -> Self {
        Self {
            memory_usage: 0,
            peak_memory_usage: 0,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            fragmentation_ratio: 0.0,
            avg_allocation_size: 0,
            allocation_rate: 0.0,
            deallocation_rate: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Memory allocation information
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Allocation size in bytes
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: Instant,
    /// Stack trace (if available)
    pub stack_trace: Option<Vec<String>>,
    /// Allocation type/category
    pub category: AllocationCategory,
    /// Thread ID that made the allocation
    pub thread_id: thread::ThreadId,
}

/// Categories of memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationCategory {
    /// Audio buffer allocations
    AudioBuffer,
    /// Tensor/model data
    TensorData,
    /// Configuration data
    Config,
    /// Cache data
    Cache,
    /// Temporary/working memory
    Temporary,
    /// Unknown/other
    Other,
}

impl fmt::Display for AllocationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioBuffer => write!(f, "AudioBuffer"),
            Self::TensorData => write!(f, "TensorData"),
            Self::Config => write!(f, "Config"),
            Self::Cache => write!(f, "Cache"),
            Self::Temporary => write!(f, "Temporary"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Configuration for resource tracking
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Enable detailed tracking (impacts performance)
    pub enable_detailed_tracking: bool,
    /// Enable stack trace collection
    pub enable_stack_traces: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Update interval for statistics
    pub stats_update_interval: Duration,
    /// Enable automatic leak detection
    pub enable_leak_detection: bool,
    /// Threshold for considering an allocation a potential leak
    pub leak_detection_threshold: Duration,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_tracking: true,
            enable_stack_traces: false, // Expensive
            max_tracked_allocations: 10000,
            stats_update_interval: Duration::from_secs(1),
            enable_leak_detection: true,
            leak_detection_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// System-level memory information
#[derive(Debug, Clone, Copy)]
pub struct SystemMemoryInfo {
    /// Resident set size (physical memory currently used)
    pub rss: u64,
    /// Virtual memory size (total virtual memory used)
    pub virtual_memory: u64,
    /// Peak resident set size (maximum physical memory used)
    pub peak_rss: u64,
}

/// Memory tracker for monitoring allocations and deallocations
pub struct MemoryTracker {
    /// Current allocations
    allocations: Arc<RwLock<HashMap<usize, AllocationInfo>>>,
    /// Statistics by category
    category_stats: Arc<RwLock<HashMap<AllocationCategory, ResourceStats>>>,
    /// Global statistics
    #[allow(dead_code)]
    global_stats: Arc<RwLock<ResourceStats>>,
    /// Atomic counters for fast access
    current_memory: AtomicU64,
    peak_memory: AtomicU64,
    total_allocations: AtomicU64,
    total_deallocations: AtomicU64,
    /// Configuration
    config: TrackingConfig,
    /// Start time for rate calculations
    start_time: Instant,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            category_stats: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(ResourceStats::default())),
            current_memory: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            config,
            start_time: Instant::now(),
        }
    }

    /// Create tracker with default configuration
    pub fn with_default_config() -> Self {
        Self::new(TrackingConfig::default())
    }

    /// Record a new allocation
    pub fn record_allocation(&self, ptr: usize, size: usize, category: AllocationCategory) {
        // Update atomic counters
        let new_memory = self
            .current_memory
            .fetch_add(size as u64, Ordering::Relaxed)
            + size as u64;
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        // Update peak memory if necessary
        let current_peak = self.peak_memory.load(Ordering::Relaxed);
        if new_memory > current_peak {
            self.peak_memory.store(new_memory, Ordering::Relaxed);
        }

        if self.config.enable_detailed_tracking {
            let allocation_info = AllocationInfo {
                size,
                timestamp: Instant::now(),
                stack_trace: if self.config.enable_stack_traces {
                    Some(self.capture_stack_trace())
                } else {
                    None
                },
                category,
                thread_id: thread::current().id(),
            };

            // Store allocation info
            if let Ok(mut allocations) = self.allocations.write() {
                // Check if we're exceeding the limit
                if allocations.len() >= self.config.max_tracked_allocations {
                    // Remove oldest allocation to make space
                    if let Some(oldest_ptr) = allocations
                        .iter()
                        .min_by_key(|(_, info)| info.timestamp)
                        .map(|(ptr, _)| *ptr)
                    {
                        allocations.remove(&oldest_ptr);
                    }
                }
                allocations.insert(ptr, allocation_info);
            }

            // Update category statistics
            self.update_category_stats(category, size as i64);
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, ptr: usize) -> Option<AllocationInfo> {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);

        if self.config.enable_detailed_tracking {
            if let Ok(mut allocations) = self.allocations.write() {
                if let Some(allocation_info) = allocations.remove(&ptr) {
                    self.current_memory
                        .fetch_sub(allocation_info.size as u64, Ordering::Relaxed);
                    self.update_category_stats(
                        allocation_info.category,
                        -(allocation_info.size as i64),
                    );
                    return Some(allocation_info);
                }
            }
        }

        None
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> u64 {
        self.current_memory.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> u64 {
        self.peak_memory.load(Ordering::Relaxed)
    }

    /// Get global resource statistics
    pub fn get_global_stats(&self) -> ResourceStats {
        let current_memory = self.current_memory.load(Ordering::Relaxed);
        let peak_memory = self.peak_memory.load(Ordering::Relaxed);
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        let total_deallocs = self.total_deallocations.load(Ordering::Relaxed);

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let allocation_rate = if elapsed > 0.0 {
            total_allocs as f64 / elapsed
        } else {
            0.0
        };
        let deallocation_rate = if elapsed > 0.0 {
            total_deallocs as f64 / elapsed
        } else {
            0.0
        };
        let avg_allocation_size = current_memory.checked_div(total_allocs).unwrap_or(0);

        // Enhanced memory statistics with system-level information
        let mut stats = ResourceStats {
            memory_usage: current_memory,
            peak_memory_usage: peak_memory,
            total_allocations: total_allocs,
            total_deallocations: total_deallocs,
            active_allocations: total_allocs - total_deallocs,
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
            avg_allocation_size,
            allocation_rate,
            deallocation_rate,
            last_updated: SystemTime::now(),
        };

        // Add system memory information if available
        if let Some(system_memory) = self.get_system_memory_info() {
            // Update statistics with system-level memory information
            stats.memory_usage = std::cmp::max(stats.memory_usage, system_memory.rss);
            stats.peak_memory_usage =
                std::cmp::max(stats.peak_memory_usage, system_memory.peak_rss);

            // Calculate more accurate fragmentation ratio using system info
            if system_memory.virtual_memory > 0 && system_memory.rss <= system_memory.virtual_memory
            {
                let virtual_fragmentation = (system_memory.virtual_memory - system_memory.rss)
                    as f64
                    / system_memory.virtual_memory as f64;
                // Ensure fragmentation ratio stays within bounds
                let combined_fragmentation =
                    stats.fragmentation_ratio * 0.5 + virtual_fragmentation * 0.5;
                stats.fragmentation_ratio = combined_fragmentation.clamp(0.0, 1.0);
            }
        }

        stats
    }

    /// Get system-level memory information
    fn get_system_memory_info(&self) -> Option<SystemMemoryInfo> {
        self.collect_system_memory_info()
    }

    /// Collect system-level memory information
    fn collect_system_memory_info(&self) -> Option<SystemMemoryInfo> {
        #[cfg(target_os = "linux")]
        {
            self.collect_linux_memory_info()
        }
        #[cfg(target_os = "macos")]
        {
            self.collect_macos_memory_info()
        }
        #[cfg(target_os = "windows")]
        {
            self.collect_windows_memory_info()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            None
        }
    }

    /// Collect Linux-specific memory information
    #[cfg(target_os = "linux")]
    fn collect_linux_memory_info(&self) -> Option<SystemMemoryInfo> {
        use std::fs;

        // Read from /proc/self/status for detailed memory information
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            let mut rss = 0;
            let mut virtual_memory = 0;
            let mut peak_rss = 0;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        rss = value.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    }
                } else if line.starts_with("VmSize:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        virtual_memory = value.parse::<u64>().unwrap_or(0) * 1024;
                        // Convert kB to bytes
                    }
                } else if line.starts_with("VmHWM:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        peak_rss = value.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    }
                }
            }

            Some(SystemMemoryInfo {
                rss,
                virtual_memory,
                peak_rss,
            })
        } else {
            None
        }
    }

    /// Collect macOS-specific memory information
    #[cfg(target_os = "macos")]
    fn collect_macos_memory_info(&self) -> Option<SystemMemoryInfo> {
        // Use task_info to get memory information on macOS
        use std::mem;

        unsafe {
            #[allow(deprecated)]
            let task = libc::mach_task_self();
            let mut info: libc::mach_task_basic_info = mem::zeroed();
            let mut count =
                (mem::size_of::<libc::mach_task_basic_info>() / mem::size_of::<u32>()) as u32;

            let result = libc::task_info(
                task,
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
            );

            if result == libc::KERN_SUCCESS {
                Some(SystemMemoryInfo {
                    rss: info.resident_size,
                    virtual_memory: info.virtual_size,
                    peak_rss: info.resident_size_max,
                })
            } else {
                None
            }
        }
    }

    /// Collect Windows-specific memory information
    #[cfg(target_os = "windows")]
    fn collect_windows_memory_info(&self) -> Option<SystemMemoryInfo> {
        // Windows implementation would use GetProcessMemoryInfo
        // For now, return None as a placeholder
        None
    }

    /// Get statistics by category
    pub fn get_category_stats(&self) -> HashMap<AllocationCategory, ResourceStats> {
        if let Ok(stats) = self.category_stats.read() {
            stats.clone()
        } else {
            HashMap::new()
        }
    }

    /// Detect potential memory leaks
    pub fn detect_leaks(&self) -> Vec<(usize, AllocationInfo)> {
        if !self.config.enable_leak_detection {
            return Vec::new();
        }

        let now = Instant::now();
        let mut leaks = Vec::new();

        if let Ok(allocations) = self.allocations.read() {
            for (ptr, info) in allocations.iter() {
                if now.duration_since(info.timestamp) > self.config.leak_detection_threshold {
                    leaks.push((*ptr, info.clone()));
                }
            }
        }

        leaks
    }

    /// Get allocations by category
    pub fn get_allocations_by_category(
        &self,
        category: AllocationCategory,
    ) -> Vec<(usize, AllocationInfo)> {
        if let Ok(allocations) = self.allocations.read() {
            allocations
                .iter()
                .filter(|(_, info)| info.category == category)
                .map(|(ptr, info)| (*ptr, info.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get memory usage over time (for visualization)
    pub fn get_memory_timeline(&self) -> Vec<(SystemTime, u64)> {
        // This would require periodic sampling in a real implementation
        // For now, return current state
        vec![(SystemTime::now(), self.current_memory_usage())]
    }

    /// Force garbage collection hint
    pub fn hint_gc(&self) {
        // In Rust, we can't force GC, but we can provide a hint
        // that cleanup might be beneficial
        if self.config.enable_detailed_tracking {
            self.cleanup_stale_allocations();
        }
    }

    /// Clear all tracking data
    pub fn clear(&self) {
        if let Ok(mut allocations) = self.allocations.write() {
            allocations.clear();
        }
        if let Ok(mut category_stats) = self.category_stats.write() {
            category_stats.clear();
        }

        self.current_memory.store(0, Ordering::Relaxed);
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_deallocations.store(0, Ordering::Relaxed);
    }

    /// Update category statistics
    fn update_category_stats(&self, category: AllocationCategory, size_delta: i64) {
        if let Ok(mut stats) = self.category_stats.write() {
            let category_stat = stats.entry(category).or_insert_with(ResourceStats::default);

            if size_delta > 0 {
                category_stat.memory_usage += size_delta as u64;
                category_stat.total_allocations += 1;
                if category_stat.memory_usage > category_stat.peak_memory_usage {
                    category_stat.peak_memory_usage = category_stat.memory_usage;
                }
            } else {
                category_stat.memory_usage = category_stat
                    .memory_usage
                    .saturating_sub((-size_delta) as u64);
                category_stat.total_deallocations += 1;
            }

            category_stat.last_updated = SystemTime::now();
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation_ratio(&self) -> f64 {
        // Simplified fragmentation calculation
        // In a real implementation, this would analyze memory layout
        if let Ok(allocations) = self.allocations.read() {
            if allocations.is_empty() {
                return 0.0;
            }

            let total_size: usize = allocations.values().map(|info| info.size).sum();
            if total_size == 0 {
                return 0.0;
            }

            let avg_size = total_size / allocations.len();
            if avg_size == 0 {
                return 0.0;
            }

            let variance: f64 = allocations
                .values()
                .map(|info| {
                    let diff = info.size as f64 - avg_size as f64;
                    diff * diff
                })
                .sum::<f64>()
                / allocations.len() as f64;

            let std_dev = variance.sqrt();
            let fragmentation = std_dev / avg_size as f64;

            // Normalize fragmentation to 0.0-1.0 range
            // Standard deviation relative to mean can exceed 1.0, so we clamp it
            fragmentation.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Capture stack trace using backtrace crate
    fn capture_stack_trace(&self) -> Vec<String> {
        let bt = Backtrace::new();
        let mut frames = Vec::new();

        // Convert backtrace to string representation
        for frame in bt.frames() {
            for symbol in frame.symbols() {
                let mut frame_info = String::new();

                // Add function name if available
                if let Some(name) = symbol.name() {
                    frame_info.push_str(&format!("{name}"));
                }

                // Add file and line information if available
                if let Some(filename) = symbol.filename() {
                    // Extract just the filename, not the full path
                    let file_name = filename
                        .file_name()
                        .map(|f| f.to_string_lossy())
                        .unwrap_or_else(|| "unknown".into());

                    frame_info.push_str(&format!(" ({file_name}:"));
                    if let Some(line) = symbol.lineno() {
                        frame_info.push_str(&format!("{line})"));
                    } else {
                        frame_info.push(')');
                    }
                }

                // Only include meaningful frames using the helper method
                if self.should_include_frame(&frame_info) {
                    frames.push(frame_info);
                }
            }
        }

        // Limit the number of frames to avoid excessive memory usage
        const MAX_FRAMES: usize = 20;
        if frames.len() > MAX_FRAMES {
            frames.truncate(MAX_FRAMES);
            frames.push("... (truncated)".to_string());
        }

        // If no meaningful frames were captured, provide a fallback
        if frames.is_empty() {
            frames.push(format!("allocation at thread {:?}", thread::current().id()));
        }

        // Format the frames for better readability
        self.format_stack_trace(frames)
    }

    /// Crates whose frames are runtime plumbing rather than caller context.
    const RUNTIME_CRATES: &'static [&'static str] =
        &["std", "core", "alloc", "test", "backtrace", "addr2line"];

    /// Check if a stack frame should be included in the trace
    fn should_include_frame(&self, frame_info: &str) -> bool {
        // Filter out internal Rust runtime frames
        !frame_info.is_empty()
            && !frame_info.contains("::fmt::")
            && !frame_info.contains("rust_begin_unwind")
            && !frame_info.contains("__rust_")
            // Hide the tracker's own plumbing -- every entry point that can sit on
            // top of a capture, or it burns the frame budget describing itself.
            // Matched on the trailing path so the check holds for both symbol
            // formats (see `is_runtime_frame`).
            && !frame_info.contains("MemoryTracker::capture_stack_trace")
            && !frame_info.contains("MemoryTracker::record_allocation")
            && !frame_info.contains("MemoryTracker::get_compact_stack_trace")
            && !Self::is_runtime_frame(frame_info)
    }

    /// Whether a frame belongs to the Rust runtime rather than to calling code.
    ///
    /// A single trace mixes two symbol formats: locally compiled code demangles to
    /// the legacy form (`std::panicking::catch_unwind`), while the precompiled
    /// standard library resolves through DWARF to the v0 form
    /// (`std[bb513d90a5cee88a]::panicking::catch_unwind`). Both are recognised here.
    ///
    /// Only the head of the path is inspected, so a caller frame is kept when its
    /// generic arguments merely mention runtime types, and a crate whose name ends
    /// in a runtime crate name (`scirs2_core`) is not mistaken for one.
    fn is_runtime_frame(frame_info: &str) -> bool {
        // `<Type as Trait>::method` renders the receiver first; fall through to the
        // trait when the receiver is not a path (`<fn() -> _ as FnOnce<()>>::call_once`).
        frame_info
            .split(" as ")
            .find_map(Self::crate_root_of)
            .is_some_and(|krate| Self::RUNTIME_CRATES.contains(&krate))
    }

    /// Extract the crate name heading a demangled path fragment, if it starts with one.
    fn crate_root_of(path: &str) -> Option<&str> {
        let path = path.trim_start().trim_start_matches('<');
        let bytes = path.as_bytes();
        let end = bytes
            .iter()
            .position(|b| !(b.is_ascii_alphanumeric() || *b == b'_'))?;
        if end == 0 {
            return None;
        }

        // A crate root is followed by `[` (v0 disambiguator) or `::` (path separator).
        match bytes[end] {
            b'[' => Some(&path[..end]),
            b':' if bytes.get(end + 1) == Some(&b':') => Some(&path[..end]),
            _ => None,
        }
    }

    /// Format stack trace frames for better readability
    fn format_stack_trace(&self, frames: Vec<String>) -> Vec<String> {
        frames
            .into_iter()
            .enumerate()
            .map(|(i, frame)| {
                // Add frame number and clean up the format
                let cleaned = frame.replace("voirs_sdk::", "").replace("voirs_", "");

                format!("#{i:02}: {cleaned}")
            })
            .collect()
    }

    /// Get a compact stack trace suitable for leak detection
    pub fn get_compact_stack_trace(&self) -> String {
        if !self.config.enable_stack_traces {
            return "stack traces disabled".to_string();
        }

        let frames = self.capture_stack_trace();
        if frames.is_empty() {
            return "no stack trace available".to_string();
        }

        // Take only the top 5 frames for compact representation
        frames
            .iter()
            .take(5)
            .map(|f| f.replace("voirs_sdk::", ""))
            .collect::<Vec<_>>()
            .join(" → ")
    }

    /// Clean up stale allocation tracking data
    fn cleanup_stale_allocations(&self) {
        let now = Instant::now();
        let threshold = Duration::from_secs(3600); // 1 hour

        if let Ok(mut allocations) = self.allocations.write() {
            allocations.retain(|_, info| now.duration_since(info.timestamp) < threshold);
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::with_default_config()
    }
}

/// Resource tracker for comprehensive resource management
pub struct ResourceTracker {
    /// Memory tracker
    memory_tracker: Arc<MemoryTracker>,
    /// Resource usage history
    usage_history: Arc<RwLock<BTreeMap<SystemTime, ResourceStats>>>,
    /// Background monitoring task
    monitor_handle: Option<thread::JoinHandle<()>>,
    /// Configuration
    config: TrackingConfig,
}

impl ResourceTracker {
    /// Create a new resource tracker
    pub fn new(config: TrackingConfig) -> Self {
        let memory_tracker = Arc::new(MemoryTracker::new(config.clone()));
        let usage_history = Arc::new(RwLock::new(BTreeMap::new()));

        let mut tracker = Self {
            memory_tracker,
            usage_history,
            monitor_handle: None,
            config,
        };

        tracker.start_monitoring();
        tracker
    }

    /// Get memory tracker
    pub fn memory_tracker(&self) -> &Arc<MemoryTracker> {
        &self.memory_tracker
    }

    /// Get resource usage over time
    pub fn get_usage_history(&self) -> BTreeMap<SystemTime, ResourceStats> {
        if let Ok(history) = self.usage_history.read() {
            history.clone()
        } else {
            BTreeMap::new()
        }
    }

    /// Generate comprehensive resource report
    pub fn generate_report(&self) -> ResourceReport {
        let global_stats = self.memory_tracker.get_global_stats();
        let category_stats = self.memory_tracker.get_category_stats();
        let potential_leaks = self.memory_tracker.detect_leaks();
        let usage_history = self.get_usage_history();

        ResourceReport {
            global_stats,
            category_stats,
            potential_leaks,
            usage_history,
            report_time: SystemTime::now(),
        }
    }

    /// Start background monitoring
    fn start_monitoring(&mut self) {
        let usage_history = Arc::clone(&self.usage_history);
        let memory_tracker = Arc::clone(&self.memory_tracker);
        let interval = self.config.stats_update_interval;

        let handle = thread::spawn(move || {
            loop {
                thread::sleep(interval);

                // Collect real memory statistics from the memory tracker
                let stats = memory_tracker.get_global_stats();
                let timestamp = SystemTime::now();

                if let Ok(mut history) = usage_history.write() {
                    history.insert(timestamp, stats);

                    // Limit history size to prevent unbounded growth
                    while history.len() > 1000 {
                        if let Some(oldest) = history.keys().next().cloned() {
                            history.remove(&oldest);
                        } else {
                            break;
                        }
                    }
                }
            }
        });

        self.monitor_handle = Some(handle);
    }
}

/// Comprehensive resource report
#[derive(Debug, Clone)]
pub struct ResourceReport {
    /// Global resource statistics
    pub global_stats: ResourceStats,
    /// Statistics by category
    pub category_stats: HashMap<AllocationCategory, ResourceStats>,
    /// Potential memory leaks
    pub potential_leaks: Vec<(usize, AllocationInfo)>,
    /// Resource usage history
    pub usage_history: BTreeMap<SystemTime, ResourceStats>,
    /// Report generation time
    pub report_time: SystemTime,
}

impl ResourceReport {
    /// Check if there are any concerning metrics
    pub fn has_concerns(&self) -> bool {
        !self.potential_leaks.is_empty()
            || self.global_stats.fragmentation_ratio > 0.5
            || self.global_stats.active_allocations > 10000
    }

    /// Get memory efficiency score (0.0 - 1.0)
    pub fn efficiency_score(&self) -> f64 {
        let fragmentation_penalty = self.global_stats.fragmentation_ratio;
        let leak_penalty = (self.potential_leaks.len() as f64 / 100.0).min(1.0);

        (1.0 - fragmentation_penalty * 0.5 - leak_penalty * 0.5).max(0.0)
    }
}

impl fmt::Display for ResourceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Resource Report - {}",
            self.report_time
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        )?;
        writeln!(f, "==================")?;
        writeln!(
            f,
            "Memory Usage: {:.2} MB",
            self.global_stats.memory_usage as f64 / 1024.0 / 1024.0
        )?;
        writeln!(
            f,
            "Peak Memory: {:.2} MB",
            self.global_stats.peak_memory_usage as f64 / 1024.0 / 1024.0
        )?;
        writeln!(
            f,
            "Active Allocations: {}",
            self.global_stats.active_allocations
        )?;
        writeln!(
            f,
            "Fragmentation: {:.1}%",
            self.global_stats.fragmentation_ratio * 100.0
        )?;
        writeln!(f, "Potential Leaks: {}", self.potential_leaks.len())?;
        writeln!(
            f,
            "Efficiency Score: {:.1}%",
            self.efficiency_score() * 100.0
        )?;

        if !self.category_stats.is_empty() {
            writeln!(f, "\nCategory Breakdown:")?;
            for (category, stats) in &self.category_stats {
                writeln!(
                    f,
                    "  {}: {:.2} MB ({} allocations)",
                    category,
                    stats.memory_usage as f64 / 1024.0 / 1024.0,
                    stats.active_allocations
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_tracker_basic() {
        let config = TrackingConfig {
            enable_detailed_tracking: true,
            enable_stack_traces: false,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Record allocation
        tracker.record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);
        assert_eq!(tracker.current_memory_usage(), 1024);

        // Record deallocation
        let info = tracker.record_deallocation(0x1000);
        assert!(info.is_some());
        assert_eq!(tracker.current_memory_usage(), 0);

        let stats = tracker.get_global_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_category_statistics() {
        let tracker = MemoryTracker::default();

        // Record allocations in different categories
        tracker.record_allocation(0x1000, 512, AllocationCategory::AudioBuffer);
        tracker.record_allocation(0x2000, 256, AllocationCategory::TensorData);
        tracker.record_allocation(0x3000, 128, AllocationCategory::Cache);

        let category_stats = tracker.get_category_stats();
        assert_eq!(category_stats.len(), 3);

        let audio_stats = &category_stats[&AllocationCategory::AudioBuffer];
        assert_eq!(audio_stats.memory_usage, 512);
        assert_eq!(audio_stats.total_allocations, 1);
    }

    #[test]
    fn test_leak_detection() {
        let config = TrackingConfig {
            enable_leak_detection: true,
            leak_detection_threshold: Duration::from_millis(50),
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Record allocation
        tracker.record_allocation(0x1000, 1024, AllocationCategory::Temporary);

        // Wait for leak threshold
        thread::sleep(Duration::from_millis(100));

        // Check for leaks
        let leaks = tracker.detect_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].0, 0x1000);
    }

    #[test]
    fn test_resource_report() {
        let tracker = ResourceTracker::new(TrackingConfig::default());

        // Generate some activity
        tracker
            .memory_tracker()
            .record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);
        tracker
            .memory_tracker()
            .record_allocation(0x2000, 512, AllocationCategory::TensorData);

        let report = tracker.generate_report();
        assert!(report.global_stats.memory_usage > 0);
        assert!(!report.category_stats.is_empty());

        let efficiency = report.efficiency_score();
        assert!((0.0..=1.0).contains(&efficiency));
    }

    #[test]
    fn test_stack_trace_capture() {
        let config = TrackingConfig {
            enable_stack_traces: true,
            enable_detailed_tracking: true,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Record allocation with stack trace
        tracker.record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);

        // Whether a caller's name is recoverable at all depends on the platform
        // symbolizer, not on the tracker: with symbols stripped or debug info
        // unavailable, frames resolve to the nearest preceding symbol instead
        // (`_main`, `__mh_execute_header`). Probe that capability separately so a
        // degraded environment is never mistaken for a tracking bug, and so a real
        // filtering or truncation regression is never excused as one.
        let raw_backtrace = Backtrace::new();
        let symbolication_available = raw_backtrace.frames().iter().any(|frame| {
            frame.symbols().iter().any(|symbol| {
                symbol
                    .name()
                    .is_some_and(|name| format!("{name}").contains("test_stack_trace_capture"))
            })
        });

        // Check if allocation was recorded with stack trace
        let Ok(allocations) = tracker.allocations.read() else {
            panic!("Failed to read allocations");
        };
        let Some(allocation_info) = allocations.get(&0x1000) else {
            panic!("Allocation not found in tracker");
        };
        let Some(stack_trace) = allocation_info.stack_trace.as_ref() else {
            panic!("Stack trace should be recorded when enabled");
        };
        assert!(!stack_trace.is_empty());

        // The tracker's own plumbing is noise for leak attribution and must never
        // surface, whether or not symbolication succeeded.
        for frame in stack_trace {
            assert!(
                !frame.contains("MemoryTracker::capture_stack_trace")
                    && !frame.contains("MemoryTracker::record_allocation"),
                "Tracker internals should be filtered out: {frame}"
            );
        }

        if symbolication_available {
            // Symbolication works here, so the calling frame must survive both the
            // runtime-frame filter and the frame cap. If it does not, the tracker
            // dropped the only information a leak report is built from.
            let has_meaningful_frame = stack_trace
                .iter()
                .any(|frame| frame.contains("test_stack_trace_capture"));
            assert!(
                has_meaningful_frame,
                "Stack trace should contain meaningful frames: {stack_trace:?}"
            );
        }
    }

    #[test]
    fn test_stack_trace_disabled() {
        let config = TrackingConfig {
            enable_stack_traces: false,
            enable_detailed_tracking: true,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Record allocation without stack trace
        tracker.record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);

        // Check if allocation was recorded without stack trace
        if let Ok(allocations) = tracker.allocations.read() {
            if let Some(allocation_info) = allocations.get(&0x1000) {
                assert!(allocation_info.stack_trace.is_none());
            } else {
                panic!("Allocation not found in tracker");
            }
        } else {
            panic!("Failed to read allocations");
        };
    }

    #[test]
    fn test_compact_stack_trace() {
        let config = TrackingConfig {
            enable_stack_traces: true,
            enable_detailed_tracking: true,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Test compact stack trace
        let compact_trace = tracker.get_compact_stack_trace();
        assert!(!compact_trace.is_empty());
        assert!(!compact_trace.contains("stack traces disabled"));

        // Test with stack traces disabled
        let config_disabled = TrackingConfig {
            enable_stack_traces: false,
            ..Default::default()
        };
        let tracker_disabled = MemoryTracker::new(config_disabled);
        let compact_trace_disabled = tracker_disabled.get_compact_stack_trace();
        assert_eq!(compact_trace_disabled, "stack traces disabled");
    }

    #[test]
    fn test_stack_trace_filtering() {
        let config = TrackingConfig {
            enable_stack_traces: true,
            enable_detailed_tracking: true,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        // Test frame filtering
        assert!(tracker.should_include_frame("my_function (src/lib.rs:123)"));
        assert!(!tracker.should_include_frame("std::alloc::alloc"));
        assert!(!tracker.should_include_frame("core::ptr::drop_in_place"));
        assert!(!tracker.should_include_frame("rust_begin_unwind"));
        assert!(!tracker.should_include_frame("__rust_start_panic"));
        assert!(!tracker.should_include_frame("backtrace::backtrace"));
        assert!(!tracker.should_include_frame(""));
    }

    #[test]
    fn test_stack_trace_filtering_v0_symbol_format() {
        let tracker = MemoryTracker::with_default_config();

        // The precompiled standard library resolves through DWARF to the v0 format,
        // which plain `std::` / `core::` substring checks do not match. Left
        // unfiltered these frames crowd out real callers under the frame cap.
        assert!(!tracker.should_include_frame(
            "std[bb513d90a5cee88a]::panicking::catch_unwind (panicking.rs:544)"
        ));
        assert!(!tracker
            .should_include_frame("test[83d61c8db1bee592]::run_test_in_process (lib.rs:686)"));
        assert!(!tracker.should_include_frame(
            "<alloc[1519c0c6]::boxed::Box<dyn core[f63e07]::ops::function::FnOnce<()>> \
             as core[f63e07]::ops::function::FnOnce<()>>::call_once (boxed.rs:2240)"
        ));
        // Receiver is not a path, so the trait decides.
        assert!(!tracker.should_include_frame(
            "<fn() -> core[f63e07]::result::Result<(), alloc[1519c0c6]::string::String> \
             as core[f63e07]::ops::function::FnOnce<()>>::call_once (function.rs:250)"
        ));

        // Caller frames survive, including generics that mention runtime types and
        // crates whose name merely ends in a runtime crate name.
        assert!(tracker.should_include_frame(
            "voirs_sdk::memory::tracking::tests::test_stack_trace_capture (tracking.rs:958)"
        ));
        assert!(tracker.should_include_frame(
            "voirs_sdk::pipeline::synth::<alloc::string::String> (mod.rs:12)"
        ));
        assert!(tracker.should_include_frame("scirs2_core::simd_ops::add (lib.rs:1)"));

        // The tracker's own plumbing stays hidden in either symbol format.
        assert!(!tracker.should_include_frame(
            "voirs_sdk::memory::tracking::MemoryTracker::record_allocation (tracking.rs:184)"
        ));
        assert!(!tracker.should_include_frame(
            "voirs_sdk[9a1f]::memory::tracking::MemoryTracker::capture_stack_trace (tracking.rs:560)"
        ));
        assert!(!tracker.should_include_frame(
            "voirs_sdk::memory::tracking::MemoryTracker::get_compact_stack_trace (tracking.rs:689)"
        ));
    }

    #[test]
    fn test_compact_stack_trace_excludes_tracker_internals() {
        let config = TrackingConfig {
            enable_stack_traces: true,
            enable_detailed_tracking: true,
            ..Default::default()
        };
        let tracker = MemoryTracker::new(config);

        // The compact trace keeps only the top 5 frames, so any tracker frame left
        // in place costs a fifth of the budget a leak report is read from.
        let compact = tracker.get_compact_stack_trace();
        assert!(
            !compact.contains("MemoryTracker::get_compact_stack_trace")
                && !compact.contains("MemoryTracker::capture_stack_trace"),
            "Compact trace should not describe the tracker itself: {compact}"
        );
    }

    #[test]
    fn test_stack_trace_formatting() {
        let config = TrackingConfig {
            enable_stack_traces: true,
            enable_detailed_tracking: true,
            ..Default::default()
        };

        let tracker = MemoryTracker::new(config);

        let frames = vec![
            "my_function (src/lib.rs:123)".to_string(),
            "another_function (src/main.rs:456)".to_string(),
        ];

        let formatted = tracker.format_stack_trace(frames);
        assert_eq!(formatted.len(), 2);
        assert!(formatted[0].starts_with("#00:"));
        assert!(formatted[1].starts_with("#01:"));
        assert!(formatted[0].contains("my_function"));
        assert!(formatted[1].contains("another_function"));
    }

    #[test]
    fn test_system_memory_info() {
        let tracker = MemoryTracker::default();

        // Test system memory info collection (may return None on some platforms)
        let system_info = tracker.get_system_memory_info();

        // If we get system info, verify the structure is reasonable
        if let Some(info) = system_info {
            // RSS should be non-zero if we're running
            assert!(info.rss > 0, "RSS should be greater than 0");
            // Virtual memory should be at least as large as RSS
            assert!(
                info.virtual_memory >= info.rss,
                "Virtual memory should be >= RSS"
            );
            // Peak RSS should be at least as large as current RSS
            assert!(
                info.peak_rss >= info.rss,
                "Peak RSS should be >= current RSS"
            );
        }
    }

    #[test]
    fn test_enhanced_global_stats() {
        let tracker = MemoryTracker::default();

        // Record some allocations
        tracker.record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);
        tracker.record_allocation(0x2000, 512, AllocationCategory::TensorData);

        // Get enhanced global statistics
        let stats = tracker.get_global_stats();

        // Verify basic statistics
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 0);
        assert_eq!(stats.active_allocations, 2);
        assert!(stats.memory_usage >= 1536); // At least our tracked allocations

        // Verify timing information
        assert!(stats.allocation_rate >= 0.0);
        assert_eq!(stats.deallocation_rate, 0.0);

        // Verify fragmentation calculation
        assert!(stats.fragmentation_ratio >= 0.0);
        assert!(stats.fragmentation_ratio <= 1.0);
    }

    #[test]
    fn test_resource_tracker_real_stats() {
        let tracker = ResourceTracker::new(TrackingConfig::default());

        // Generate some memory activity
        tracker
            .memory_tracker()
            .record_allocation(0x1000, 2048, AllocationCategory::AudioBuffer);
        tracker
            .memory_tracker()
            .record_allocation(0x2000, 1024, AllocationCategory::TensorData);

        // Generate a report with real statistics
        let report = tracker.generate_report();

        // Verify the report contains real data
        assert!(report.global_stats.memory_usage > 0);
        assert_eq!(report.global_stats.total_allocations, 2);
        assert!(!report.category_stats.is_empty());

        // Verify efficiency score calculation
        let efficiency = report.efficiency_score();
        assert!((0.0..=1.0).contains(&efficiency));
    }

    #[test]
    fn test_memory_timeline_tracking() {
        let tracker = MemoryTracker::default();

        // Test memory timeline functionality
        let initial_timeline = tracker.get_memory_timeline();
        assert_eq!(initial_timeline.len(), 1); // Should have current state

        // Record allocation and verify timeline
        tracker.record_allocation(0x1000, 1024, AllocationCategory::AudioBuffer);
        let updated_timeline = tracker.get_memory_timeline();
        assert_eq!(updated_timeline.len(), 1); // Still just current state
        assert!(updated_timeline[0].1 >= 1024); // Memory usage should include our allocation
    }

    #[test]
    fn test_system_memory_collection_platforms() {
        let tracker = MemoryTracker::default();

        // Test that system memory collection doesn't panic on any platform
        let system_info = tracker.collect_system_memory_info();

        // The result may be None on unsupported platforms, but it shouldn't panic
        match system_info {
            Some(info) => {
                // If we got info, it should be reasonable
                assert!(info.rss > 0);
                assert!(info.virtual_memory >= info.rss);
                assert!(info.peak_rss >= info.rss);
            }
            None => {
                // No system info available, which is fine for some platforms
            }
        }
    }

    #[test]
    fn test_enhanced_fragmentation_calculation() {
        let config = TrackingConfig {
            enable_detailed_tracking: true,
            ..Default::default()
        };
        let tracker = MemoryTracker::new(config);

        // Create allocations with varying sizes to test fragmentation calculation
        tracker.record_allocation(0x1000, 100, AllocationCategory::AudioBuffer);
        tracker.record_allocation(0x2000, 1000, AllocationCategory::TensorData);
        tracker.record_allocation(0x3000, 10000, AllocationCategory::Cache);

        let stats = tracker.get_global_stats();

        // With varied allocation sizes, we should have some fragmentation
        assert!(stats.fragmentation_ratio >= 0.0);
        assert!(stats.fragmentation_ratio <= 1.0);

        // Test that the enhanced calculation includes system memory info if available
        let base_fragmentation = tracker.calculate_fragmentation_ratio();
        assert!(base_fragmentation >= 0.0);
    }
}
