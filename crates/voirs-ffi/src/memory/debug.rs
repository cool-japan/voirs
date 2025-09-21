//! Memory Debugging Tools
//!
//! Allocation tracking, leak detection, memory usage statistics,
//! and debug output formatting for comprehensive memory analysis.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::alloc::Layout;
use std::collections::HashMap;
// Note: AtomicUsize, Ordering, and Mutex imports removed as they are unused in this file
use std::time::Instant;

/// Detailed allocation information for debugging
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    pub ptr: usize,
    pub size: usize,
    pub align: usize,
    pub allocated_at: Instant,
    pub location: Option<SourceLocation>,
    pub backtrace: Option<String>,
}

/// Source code location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub function: &'static str,
}

/// Memory usage statistics with detailed breakdowns
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocations: usize,
    pub peak_allocations: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_deallocated: usize,
    pub current_bytes_allocated: usize,
    pub peak_bytes_allocated: usize,
    pub allocation_histogram: HashMap<usize, usize>, // size -> count
    pub allocation_timeline: Vec<AllocationEvent>,
}

/// Memory allocation event for timeline tracking
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    pub timestamp: Instant,
    pub event_type: AllocationEventType,
    pub ptr: usize,
    pub size: usize,
    pub total_allocated: usize,
}

#[derive(Debug, Clone)]
pub enum AllocationEventType {
    Allocate,
    Deallocate,
    Reallocate { old_size: usize },
}

/// Memory leak detection and reporting
#[derive(Debug, Clone)]
pub struct LeakReport {
    pub total_leaked_bytes: usize,
    pub leaked_allocations: Vec<AllocationRecord>,
    pub leak_by_size: HashMap<usize, usize>,
    pub leak_by_location: HashMap<String, usize>,
}

/// Global memory debugger instance
static MEMORY_DEBUGGER: Lazy<RwLock<MemoryDebugger>> =
    Lazy::new(|| RwLock::new(MemoryDebugger::new()));

/// Memory debugger implementation
struct MemoryDebugger {
    enabled: bool,
    allocations: HashMap<usize, AllocationRecord>,
    statistics: MemoryStatistics,
    max_timeline_events: usize,
    track_backtraces: bool,
}

impl MemoryDebugger {
    fn new() -> Self {
        Self {
            enabled: false,
            allocations: HashMap::new(),
            statistics: MemoryStatistics::default(),
            max_timeline_events: 10000,
            track_backtraces: false,
        }
    }

    fn record_allocation(&mut self, ptr: usize, layout: Layout, location: Option<SourceLocation>) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();

        // Update statistics
        self.statistics.total_allocations += 1;
        self.statistics.current_allocations += 1;
        self.statistics.total_bytes_allocated += layout.size();
        self.statistics.current_bytes_allocated += layout.size();

        if self.statistics.current_allocations > self.statistics.peak_allocations {
            self.statistics.peak_allocations = self.statistics.current_allocations;
        }

        if self.statistics.current_bytes_allocated > self.statistics.peak_bytes_allocated {
            self.statistics.peak_bytes_allocated = self.statistics.current_bytes_allocated;
        }

        // Update histogram
        *self
            .statistics
            .allocation_histogram
            .entry(layout.size())
            .or_insert(0) += 1;

        // Add to timeline
        if self.statistics.allocation_timeline.len() < self.max_timeline_events {
            self.statistics.allocation_timeline.push(AllocationEvent {
                timestamp: now,
                event_type: AllocationEventType::Allocate,
                ptr,
                size: layout.size(),
                total_allocated: self.statistics.current_bytes_allocated,
            });
        }

        // Record allocation details
        let record = AllocationRecord {
            ptr,
            size: layout.size(),
            align: layout.align(),
            allocated_at: now,
            location,
            backtrace: if self.track_backtraces {
                Some(capture_backtrace())
            } else {
                None
            },
        };

        self.allocations.insert(ptr, record);
    }

    fn record_deallocation(&mut self, ptr: usize, layout: Layout) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();

        // Remove allocation record
        self.allocations.remove(&ptr);

        // Update statistics
        self.statistics.total_deallocations += 1;
        self.statistics.current_allocations = self.statistics.current_allocations.saturating_sub(1);
        self.statistics.total_bytes_deallocated += layout.size();
        self.statistics.current_bytes_allocated = self
            .statistics
            .current_bytes_allocated
            .saturating_sub(layout.size());

        // Add to timeline
        if self.statistics.allocation_timeline.len() < self.max_timeline_events {
            self.statistics.allocation_timeline.push(AllocationEvent {
                timestamp: now,
                event_type: AllocationEventType::Deallocate,
                ptr,
                size: layout.size(),
                total_allocated: self.statistics.current_bytes_allocated,
            });
        }
    }

    fn record_reallocation(
        &mut self,
        old_ptr: usize,
        new_ptr: usize,
        old_layout: Layout,
        new_size: usize,
    ) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();

        // Remove old allocation record
        let old_record = self.allocations.remove(&old_ptr);

        // Update statistics
        self.statistics.total_bytes_deallocated += old_layout.size();
        self.statistics.total_bytes_allocated += new_size;
        self.statistics.current_bytes_allocated = self
            .statistics
            .current_bytes_allocated
            .saturating_sub(old_layout.size())
            .saturating_add(new_size);

        if self.statistics.current_bytes_allocated > self.statistics.peak_bytes_allocated {
            self.statistics.peak_bytes_allocated = self.statistics.current_bytes_allocated;
        }

        // Update histogram
        *self
            .statistics
            .allocation_histogram
            .entry(new_size)
            .or_insert(0) += 1;

        // Add to timeline
        if self.statistics.allocation_timeline.len() < self.max_timeline_events {
            self.statistics.allocation_timeline.push(AllocationEvent {
                timestamp: now,
                event_type: AllocationEventType::Reallocate {
                    old_size: old_layout.size(),
                },
                ptr: new_ptr,
                size: new_size,
                total_allocated: self.statistics.current_bytes_allocated,
            });
        }

        // Create new allocation record
        let new_record = AllocationRecord {
            ptr: new_ptr,
            size: new_size,
            align: old_layout.align(),
            allocated_at: old_record.as_ref().map(|r| r.allocated_at).unwrap_or(now),
            location: old_record.as_ref().and_then(|r| r.location.clone()),
            backtrace: if self.track_backtraces {
                Some(capture_backtrace())
            } else {
                old_record.and_then(|r| r.backtrace)
            },
        };

        self.allocations.insert(new_ptr, new_record);
    }

    fn generate_leak_report(&self) -> LeakReport {
        let mut leak_by_size = HashMap::new();
        let mut leak_by_location = HashMap::new();
        let mut total_leaked_bytes = 0;

        for record in self.allocations.values() {
            total_leaked_bytes += record.size;
            *leak_by_size.entry(record.size).or_insert(0) += 1;

            if let Some(ref location) = record.location {
                let location_key =
                    format!("{}:{}:{}", location.file, location.line, location.function);
                *leak_by_location.entry(location_key).or_insert(0) += record.size;
            }
        }

        LeakReport {
            total_leaked_bytes,
            leaked_allocations: self.allocations.values().cloned().collect(),
            leak_by_size,
            leak_by_location,
        }
    }
}

/// Enable memory debugging
pub fn enable_memory_debugging() {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.enabled = true;
}

/// Disable memory debugging
pub fn disable_memory_debugging() {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.enabled = false;
}

/// Check if memory debugging is enabled
pub fn is_memory_debugging_enabled() -> bool {
    MEMORY_DEBUGGER.read().enabled
}

/// Enable backtrace tracking (expensive!)
pub fn enable_backtrace_tracking() {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.track_backtraces = true;
}

/// Disable backtrace tracking
pub fn disable_backtrace_tracking() {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.track_backtraces = false;
}

/// Set maximum number of timeline events to track
pub fn set_max_timeline_events(max: usize) {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.max_timeline_events = max;
}

/// Record an allocation (called by allocators)
pub fn record_allocation(ptr: *mut u8, layout: Layout) {
    record_allocation_at(ptr, layout, None);
}

/// Record an allocation with source location
pub fn record_allocation_at(ptr: *mut u8, layout: Layout, location: Option<SourceLocation>) {
    if !ptr.is_null() {
        let mut debugger = MEMORY_DEBUGGER.write();
        debugger.record_allocation(ptr as usize, layout, location);
    }
}

/// Record a deallocation (called by allocators)
pub fn record_deallocation(ptr: *mut u8, layout: Layout) {
    if !ptr.is_null() {
        let mut debugger = MEMORY_DEBUGGER.write();
        debugger.record_deallocation(ptr as usize, layout);
    }
}

/// Record a reallocation (called by allocators)
pub fn record_reallocation(
    old_ptr: *mut u8,
    new_ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) {
    if !old_ptr.is_null() && !new_ptr.is_null() {
        let mut debugger = MEMORY_DEBUGGER.write();
        debugger.record_reallocation(old_ptr as usize, new_ptr as usize, old_layout, new_size);
    }
}

/// Get current memory statistics
pub fn get_memory_statistics() -> MemoryStatistics {
    MEMORY_DEBUGGER.read().statistics.clone()
}

/// Reset memory statistics
pub fn reset_memory_statistics() {
    let mut debugger = MEMORY_DEBUGGER.write();
    debugger.statistics = MemoryStatistics::default();
    debugger.allocations.clear();
}

/// Generate a memory leak report
pub fn generate_leak_report() -> LeakReport {
    MEMORY_DEBUGGER.read().generate_leak_report()
}

/// Check for memory leaks
pub fn has_memory_leaks() -> bool {
    let debugger = MEMORY_DEBUGGER.read();
    !debugger.allocations.is_empty()
}

/// Get count of current allocations
pub fn get_current_allocation_count() -> usize {
    MEMORY_DEBUGGER.read().allocations.len()
}

/// Get total allocated bytes currently outstanding
pub fn get_current_allocated_bytes() -> usize {
    MEMORY_DEBUGGER.read().statistics.current_bytes_allocated
}

/// Dump all current allocations to string
pub fn dump_allocations() -> String {
    let debugger = MEMORY_DEBUGGER.read();
    let mut output = String::new();

    output.push_str("=== Memory Allocations Dump ===\n");
    output.push_str(&format!(
        "Total allocations: {}\n",
        debugger.allocations.len()
    ));
    output.push_str(&format!(
        "Total bytes: {}\n\n",
        debugger.statistics.current_bytes_allocated
    ));

    for (i, record) in debugger.allocations.values().enumerate() {
        output.push_str(&format!("Allocation #{}\n", i + 1));
        output.push_str(&format!("  Pointer: 0x{:x}\n", record.ptr));
        output.push_str(&format!("  Size: {} bytes\n", record.size));
        output.push_str(&format!("  Alignment: {}\n", record.align));
        output.push_str(&format!("  Allocated at: {:?}\n", record.allocated_at));

        if let Some(ref location) = record.location {
            output.push_str(&format!(
                "  Location: {}:{} in {}\n",
                location.file, location.line, location.function
            ));
        }

        if let Some(ref backtrace) = record.backtrace {
            output.push_str(&format!("  Backtrace:\n{backtrace}\n"));
        }

        output.push('\n');
    }

    output
}

/// Format memory statistics as human-readable string
pub fn format_memory_statistics(stats: &MemoryStatistics) -> String {
    let mut output = String::new();

    output.push_str("=== Memory Statistics ===\n");
    output.push_str(&format!("Total allocations: {}\n", stats.total_allocations));
    output.push_str(&format!(
        "Total deallocations: {}\n",
        stats.total_deallocations
    ));
    output.push_str(&format!(
        "Current allocations: {}\n",
        stats.current_allocations
    ));
    output.push_str(&format!("Peak allocations: {}\n", stats.peak_allocations));
    output.push_str(&format!(
        "Total bytes allocated: {}\n",
        format_bytes(stats.total_bytes_allocated)
    ));
    output.push_str(&format!(
        "Total bytes deallocated: {}\n",
        format_bytes(stats.total_bytes_deallocated)
    ));
    output.push_str(&format!(
        "Current bytes allocated: {}\n",
        format_bytes(stats.current_bytes_allocated)
    ));
    output.push_str(&format!(
        "Peak bytes allocated: {}\n",
        format_bytes(stats.peak_bytes_allocated)
    ));

    if !stats.allocation_histogram.is_empty() {
        output.push_str("\n=== Allocation Size Histogram ===\n");
        let mut histogram: Vec<_> = stats.allocation_histogram.iter().collect();
        histogram.sort_by_key(|(size, _)| **size);

        for (size, count) in histogram {
            output.push_str(&format!(
                "{:>10} bytes: {} allocations\n",
                format_bytes(*size),
                count
            ));
        }
    }

    output
}

/// Format leak report as human-readable string
pub fn format_leak_report(report: &LeakReport) -> String {
    let mut output = String::new();

    output.push_str("=== Memory Leak Report ===\n");
    output.push_str(&format!(
        "Total leaked bytes: {}\n",
        format_bytes(report.total_leaked_bytes)
    ));
    output.push_str(&format!(
        "Total leaked allocations: {}\n",
        report.leaked_allocations.len()
    ));

    if !report.leak_by_size.is_empty() {
        output.push_str("\n=== Leaks by Size ===\n");
        let mut leaks_by_size: Vec<_> = report.leak_by_size.iter().collect();
        leaks_by_size.sort_by_key(|(size, _)| **size);

        for (size, count) in leaks_by_size {
            output.push_str(&format!(
                "{:>10} bytes: {} leaks\n",
                format_bytes(*size),
                count
            ));
        }
    }

    if !report.leak_by_location.is_empty() {
        output.push_str("\n=== Leaks by Location ===\n");
        let mut leaks_by_location: Vec<_> = report.leak_by_location.iter().collect();
        leaks_by_location.sort_by_key(|(_, bytes)| std::cmp::Reverse(**bytes));

        for (location, bytes) in leaks_by_location {
            output.push_str(&format!(
                "{:>10} bytes: {}\n",
                format_bytes(*bytes),
                location
            ));
        }
    }

    output
}

/// Format bytes as human-readable string
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Capture backtrace using the backtrace crate
fn capture_backtrace() -> String {
    use backtrace::Backtrace;

    // Check if backtrace capture is enabled via environment variable
    if std::env::var("RUST_BACKTRACE")
        .unwrap_or_default()
        .is_empty()
    {
        return "Backtrace disabled (set RUST_BACKTRACE=1 to enable)".to_string();
    }

    let bt = Backtrace::new();

    // Format the backtrace for better readability
    let mut result = String::new();
    let frames: Vec<_> = bt.frames().iter().take(10).collect(); // Limit to 10 frames

    for (i, frame) in frames.iter().enumerate() {
        for symbol in frame.symbols() {
            if let Some(name) = symbol.name() {
                result.push_str(&format!("  {}: {}\n", i, name));
                if let Some(filename) = symbol.filename() {
                    if let Some(line) = symbol.lineno() {
                        result.push_str(&format!("      at {}:{}\n", filename.display(), line));
                    }
                }
            } else {
                result.push_str(&format!("  {}: <unknown symbol>\n", i));
            }
        }
    }

    if result.is_empty() {
        "Backtrace captured but no symbols available".to_string()
    } else {
        result
    }
}

/// Macro to track allocations with source location
#[macro_export]
macro_rules! track_allocation {
    ($ptr:expr, $layout:expr) => {
        $crate::memory::debug::record_allocation_at(
            $ptr,
            $layout,
            Some($crate::memory::debug::SourceLocation {
                file: file!(),
                line: line!(),
                column: column!(),
                function: "<function>",
            }),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;
    use std::sync::Mutex;

    // Test mutex to prevent concurrent access to global memory debugger state
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_memory_debugging_lifecycle() {
        let _guard = TEST_MUTEX.lock().unwrap();

        reset_memory_statistics();
        enable_memory_debugging();

        assert!(is_memory_debugging_enabled());
        let initial_count = get_current_allocation_count();
        let initial_bytes = get_current_allocated_bytes();

        // Simulate some allocations
        let layout1 = Layout::from_size_align(100, 8).unwrap();
        let layout2 = Layout::from_size_align(200, 8).unwrap();

        record_allocation(0x1000 as *mut u8, layout1);
        record_allocation(0x2000 as *mut u8, layout2);

        assert_eq!(get_current_allocation_count(), initial_count + 2);
        assert_eq!(get_current_allocated_bytes(), initial_bytes + 300);

        let stats = get_memory_statistics();
        assert!(stats.total_allocations >= initial_count + 2);
        assert_eq!(stats.current_allocations, initial_count + 2);
        assert!(stats.total_bytes_allocated >= initial_bytes + 300);

        // Simulate deallocation
        record_deallocation(0x1000 as *mut u8, layout1);

        assert_eq!(get_current_allocation_count(), initial_count + 1);
        assert_eq!(get_current_allocated_bytes(), initial_bytes + 200);

        // Check for leaks (we should have at least our remaining allocation)
        assert!(has_memory_leaks() || initial_count > 0);
        let leak_report = generate_leak_report();
        assert!(leak_report.total_leaked_bytes >= initial_bytes + 200);
        assert!(leak_report.leaked_allocations.len() >= initial_count + 1);

        // Clean up our test allocation
        record_deallocation(0x2000 as *mut u8, layout2);

        // After cleanup, we should be back to initial state
        assert_eq!(get_current_allocation_count(), initial_count);
        assert_eq!(get_current_allocated_bytes(), initial_bytes);

        disable_memory_debugging();
        assert!(!is_memory_debugging_enabled());

        // Final cleanup to prevent test interference
        reset_memory_statistics();
    }

    #[test]
    fn test_allocation_statistics() {
        let _guard = TEST_MUTEX.lock().unwrap();

        // Ensure clean state before test
        reset_memory_statistics();
        enable_memory_debugging();

        let layout = Layout::from_size_align(50, 8).unwrap();

        // Multiple allocations of same size
        for i in 0..5 {
            record_allocation((0x1000 + i * 100) as *mut u8, layout);
        }

        let stats = get_memory_statistics();

        // Test all statistics
        assert_eq!(stats.total_allocations, 5);
        assert_eq!(stats.peak_allocations, 5);
        assert_eq!(stats.peak_bytes_allocated, 250);
        assert_eq!(stats.allocation_histogram.get(&50), Some(&5));

        disable_memory_debugging();
    }

    #[test]
    fn test_memory_formatting() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }

    #[test]
    fn test_reallocation_tracking() {
        let _guard = TEST_MUTEX.lock().unwrap();

        reset_memory_statistics();
        enable_memory_debugging();

        let initial_count = get_current_allocation_count();
        let initial_bytes = get_current_allocated_bytes();
        let old_layout = Layout::from_size_align(100, 8).unwrap();

        // Initial allocation
        record_allocation(0x1000 as *mut u8, old_layout);
        assert_eq!(get_current_allocated_bytes(), initial_bytes + 100);

        // Reallocation to larger size
        record_reallocation(0x1000 as *mut u8, 0x2000 as *mut u8, old_layout, 200);
        assert_eq!(get_current_allocation_count(), initial_count + 1);
        assert_eq!(get_current_allocated_bytes(), initial_bytes + 200);

        let stats = get_memory_statistics();
        assert!(stats.total_bytes_allocated >= initial_bytes + 300); // At least 100 + 200
        assert!(stats.total_bytes_deallocated >= 100);

        disable_memory_debugging();
    }
}
