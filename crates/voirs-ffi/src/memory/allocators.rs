//! Advanced Memory Allocators
//!
//! Pluggable allocator interface with pool optimization,
//! alignment handling, and debug allocator support.

use parking_lot::RwLock;
use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::ptr;
// Note: AtomicUsize and Ordering imports removed as they are unused in this file
use std::sync::Mutex;

/// Statistics for memory allocation tracking
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocations: usize,
    pub peak_allocations: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_deallocated: usize,
    pub current_bytes_allocated: usize,
    pub peak_bytes_allocated: usize,
}

/// Allocation information for debugging
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub align: usize,
    pub timestamp: std::time::Instant,
    pub backtrace: Option<String>,
}

/// Trait for pluggable allocators
pub trait VoirsAllocator: Send + Sync {
    /// Allocate memory with given layout
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;

    /// Deallocate memory with given layout
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);

    /// Reallocate memory (optional, falls back to alloc+copy+dealloc)
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_ptr = self.alloc(new_layout);
        if !new_ptr.is_null() && !ptr.is_null() {
            let copy_size = std::cmp::min(layout.size(), new_size);
            ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
            self.dealloc(ptr, layout);
        }
        new_ptr
    }

    /// Get allocator statistics
    fn stats(&self) -> AllocatorStats;

    /// Reset statistics
    fn reset_stats(&self);

    /// Get allocator name
    fn name(&self) -> &'static str;
}

/// System allocator wrapper with tracking
pub struct TrackedSystemAllocator {
    stats: RwLock<AllocatorStats>,
    allocations: Mutex<HashMap<usize, AllocationInfo>>,
    enable_backtrace: bool,
}

impl TrackedSystemAllocator {
    pub fn new(enable_backtrace: bool) -> Self {
        Self {
            stats: RwLock::new(AllocatorStats::default()),
            allocations: Mutex::new(HashMap::new()),
            enable_backtrace,
        }
    }

    fn record_allocation(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }

        let mut stats = self.stats.write();
        stats.total_allocations += 1;
        stats.current_allocations += 1;
        stats.total_bytes_allocated += layout.size();
        stats.current_bytes_allocated += layout.size();

        if stats.current_allocations > stats.peak_allocations {
            stats.peak_allocations = stats.current_allocations;
        }

        if stats.current_bytes_allocated > stats.peak_bytes_allocated {
            stats.peak_bytes_allocated = stats.current_bytes_allocated;
        }

        if self.enable_backtrace {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(
                ptr as usize,
                AllocationInfo {
                    size: layout.size(),
                    align: layout.align(),
                    timestamp: std::time::Instant::now(),
                    backtrace: if std::env::var("RUST_BACKTRACE").is_ok() {
                        Some(format!("{:?}", std::backtrace::Backtrace::capture()))
                    } else {
                        None
                    },
                },
            );
        }
    }

    fn record_deallocation(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }

        let mut stats = self.stats.write();
        stats.total_deallocations += 1;
        stats.current_allocations = stats.current_allocations.saturating_sub(1);
        stats.total_bytes_deallocated += layout.size();
        stats.current_bytes_allocated = stats.current_bytes_allocated.saturating_sub(layout.size());

        if self.enable_backtrace {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.remove(&(ptr as usize));
        }
    }
}

impl VoirsAllocator for TrackedSystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = std::alloc::System.alloc(layout);
        self.record_allocation(ptr, layout);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.record_deallocation(ptr, layout);
        std::alloc::System.dealloc(ptr, layout);
    }

    fn stats(&self) -> AllocatorStats {
        self.stats.read().clone()
    }

    fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = AllocatorStats::default();

        let mut allocations = self.allocations.lock().unwrap();
        allocations.clear();
    }

    fn name(&self) -> &'static str {
        "TrackedSystem"
    }
}

/// Pool allocator for fixed-size allocations
pub struct PoolAllocator {
    block_size: usize,
    blocks_per_chunk: usize,
    chunks: Mutex<Vec<Chunk>>,
    free_blocks: Mutex<Vec<usize>>, // Store as usize instead of raw pointers
    stats: RwLock<AllocatorStats>,
}

struct Chunk {
    memory: usize, // Store as usize instead of raw pointer
    layout: Layout,
}

impl PoolAllocator {
    pub fn new(block_size: usize, blocks_per_chunk: usize) -> Self {
        let aligned_size = (block_size + 15) & !15; // Align to 16 bytes

        Self {
            block_size: aligned_size,
            blocks_per_chunk,
            chunks: Mutex::new(Vec::new()),
            free_blocks: Mutex::new(Vec::new()),
            stats: RwLock::new(AllocatorStats::default()),
        }
    }

    fn allocate_chunk(&self) -> Result<(), &'static str> {
        let chunk_size = self.block_size * self.blocks_per_chunk;
        let layout = Layout::from_size_align(chunk_size, 16).map_err(|_| "Invalid layout")?;

        unsafe {
            let memory = std::alloc::System.alloc(layout);
            if memory.is_null() {
                return Err("Allocation failed");
            }

            // Initialize free block list
            let mut free_blocks = self.free_blocks.lock().unwrap();
            for i in 0..self.blocks_per_chunk {
                let block_ptr = memory.add(i * self.block_size);
                free_blocks.push(block_ptr as usize);
            }

            let mut chunks = self.chunks.lock().unwrap();
            chunks.push(Chunk {
                memory: memory as usize,
                layout,
            });
        }

        Ok(())
    }
}

impl VoirsAllocator for PoolAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > self.block_size || layout.align() > 16 {
            // Fall back to system allocator for large or specially-aligned allocations
            return std::alloc::System.alloc(layout);
        }

        let mut free_blocks = self.free_blocks.lock().unwrap();

        if let Some(ptr_addr) = free_blocks.pop() {
            let ptr = ptr_addr as *mut u8;
            // Update statistics
            let mut stats = self.stats.write();
            stats.total_allocations += 1;
            stats.current_allocations += 1;
            stats.total_bytes_allocated += self.block_size;
            stats.current_bytes_allocated += self.block_size;

            if stats.current_allocations > stats.peak_allocations {
                stats.peak_allocations = stats.current_allocations;
            }

            if stats.current_bytes_allocated > stats.peak_bytes_allocated {
                stats.peak_bytes_allocated = stats.current_bytes_allocated;
            }

            ptr
        } else {
            // Need to allocate a new chunk
            drop(free_blocks);

            if self.allocate_chunk().is_ok() {
                let mut free_blocks = self.free_blocks.lock().unwrap();
                if let Some(ptr_addr) = free_blocks.pop() {
                    let ptr = ptr_addr as *mut u8;

                    // Update statistics
                    let mut stats = self.stats.write();
                    stats.total_allocations += 1;
                    stats.current_allocations += 1;
                    stats.total_bytes_allocated += self.block_size;
                    stats.current_bytes_allocated += self.block_size;

                    if stats.current_allocations > stats.peak_allocations {
                        stats.peak_allocations = stats.current_allocations;
                    }

                    if stats.current_bytes_allocated > stats.peak_bytes_allocated {
                        stats.peak_bytes_allocated = stats.current_bytes_allocated;
                    }

                    ptr
                } else {
                    ptr::null_mut()
                }
            } else {
                ptr::null_mut()
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() > self.block_size || layout.align() > 16 {
            // Was allocated by system allocator
            std::alloc::System.dealloc(ptr, layout);
            return;
        }

        if ptr.is_null() {
            return;
        }

        // Return block to free list
        let mut free_blocks = self.free_blocks.lock().unwrap();
        free_blocks.push(ptr as usize);

        // Update statistics
        let mut stats = self.stats.write();
        stats.total_deallocations += 1;
        stats.current_allocations = stats.current_allocations.saturating_sub(1);
        stats.total_bytes_deallocated += self.block_size;
        stats.current_bytes_allocated = stats
            .current_bytes_allocated
            .saturating_sub(self.block_size);
    }

    fn stats(&self) -> AllocatorStats {
        self.stats.read().clone()
    }

    fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = AllocatorStats::default();
    }

    fn name(&self) -> &'static str {
        "Pool"
    }
}

impl Drop for PoolAllocator {
    fn drop(&mut self) {
        unsafe {
            let chunks = self.chunks.lock().unwrap();
            for chunk in chunks.iter() {
                std::alloc::System.dealloc(chunk.memory as *mut u8, chunk.layout);
            }
        }
    }
}

/// Debug allocator that fills allocated memory with patterns
pub struct DebugAllocator {
    inner: Box<dyn VoirsAllocator>,
    fill_pattern: u8,
    free_pattern: u8,
}

impl DebugAllocator {
    pub fn new(inner: Box<dyn VoirsAllocator>) -> Self {
        Self {
            inner,
            fill_pattern: 0xAA, // Pattern for allocated memory
            free_pattern: 0xDD, // Pattern for freed memory
        }
    }

    pub fn with_patterns(
        inner: Box<dyn VoirsAllocator>,
        fill_pattern: u8,
        free_pattern: u8,
    ) -> Self {
        Self {
            inner,
            fill_pattern,
            free_pattern,
        }
    }
}

impl VoirsAllocator for DebugAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            // Fill allocated memory with pattern
            ptr::write_bytes(ptr, self.fill_pattern, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            // Fill freed memory with pattern
            ptr::write_bytes(ptr, self.free_pattern, layout.size());
        }
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = self.inner.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() && new_size > layout.size() {
            // Fill the new portion with pattern
            let new_portion = new_ptr.add(layout.size());
            let new_portion_size = new_size - layout.size();
            ptr::write_bytes(new_portion, self.fill_pattern, new_portion_size);
        }
        new_ptr
    }

    fn stats(&self) -> AllocatorStats {
        self.inner.stats()
    }

    fn reset_stats(&self) {
        self.inner.reset_stats();
    }

    fn name(&self) -> &'static str {
        "Debug"
    }
}

/// Global allocator manager
static GLOBAL_ALLOCATOR: RwLock<Option<Box<dyn VoirsAllocator>>> = RwLock::new(None);

/// Set the global allocator
pub fn set_global_allocator(allocator: Box<dyn VoirsAllocator>) {
    let mut global = GLOBAL_ALLOCATOR.write();
    *global = Some(allocator);
}

/// Get the global allocator statistics
pub fn get_global_allocator_stats() -> Option<AllocatorStats> {
    let global = GLOBAL_ALLOCATOR.read();
    global.as_ref().map(|alloc| alloc.stats())
}

/// Reset global allocator statistics
pub fn reset_global_allocator_stats() {
    let global = GLOBAL_ALLOCATOR.read();
    if let Some(alloc) = global.as_ref() {
        alloc.reset_stats();
    }
}

/// Get global allocator name
pub fn get_global_allocator_name() -> Option<&'static str> {
    let global = GLOBAL_ALLOCATOR.read();
    global.as_ref().map(|alloc| alloc.name())
}

/// Allocate memory using global allocator
pub unsafe fn global_alloc(layout: Layout) -> *mut u8 {
    let global = GLOBAL_ALLOCATOR.read();
    if let Some(alloc) = global.as_ref() {
        alloc.alloc(layout)
    } else {
        std::alloc::System.alloc(layout)
    }
}

/// Deallocate memory using global allocator
pub unsafe fn global_dealloc(ptr: *mut u8, layout: Layout) {
    let global = GLOBAL_ALLOCATOR.read();
    if let Some(alloc) = global.as_ref() {
        alloc.dealloc(ptr, layout);
    } else {
        std::alloc::System.dealloc(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracked_system_allocator() {
        let allocator = TrackedSystemAllocator::new(false);

        unsafe {
            let layout = Layout::from_size_align(100, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());

            let stats = allocator.stats();
            assert_eq!(stats.total_allocations, 1);
            assert_eq!(stats.current_allocations, 1);
            assert_eq!(stats.total_bytes_allocated, 100);

            allocator.dealloc(ptr, layout);

            let stats = allocator.stats();
            assert_eq!(stats.total_deallocations, 1);
            assert_eq!(stats.current_allocations, 0);
            assert_eq!(stats.total_bytes_deallocated, 100);
        }
    }

    #[test]
    fn test_pool_allocator() {
        let allocator = PoolAllocator::new(64, 10);

        unsafe {
            let layout = Layout::from_size_align(32, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            let ptr2 = allocator.alloc(layout);

            assert!(!ptr1.is_null());
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);

            let stats = allocator.stats();
            assert_eq!(stats.current_allocations, 2);

            allocator.dealloc(ptr1, layout);
            allocator.dealloc(ptr2, layout);

            let stats = allocator.stats();
            assert_eq!(stats.current_allocations, 0);
        }
    }

    #[test]
    fn test_debug_allocator() {
        let system_allocator = Box::new(TrackedSystemAllocator::new(false));
        let debug_allocator = DebugAllocator::new(system_allocator);

        unsafe {
            let layout = Layout::from_size_align(100, 8).unwrap();
            let ptr = debug_allocator.alloc(layout);
            assert!(!ptr.is_null());

            // Check that memory is filled with pattern
            let slice = std::slice::from_raw_parts(ptr, 100);
            assert!(slice.iter().all(|&b| b == 0xAA));

            debug_allocator.dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_allocator_realloc() {
        let allocator = TrackedSystemAllocator::new(false);

        unsafe {
            let layout = Layout::from_size_align(50, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());

            // Write some data
            ptr::write_bytes(ptr, 0x42, 50);

            // Reallocate to larger size
            let new_ptr = allocator.realloc(ptr, layout, 100);
            assert!(!new_ptr.is_null());

            // Check that original data is preserved
            let slice = std::slice::from_raw_parts(new_ptr, 50);
            assert!(slice.iter().all(|&b| b == 0x42));

            let new_layout = Layout::from_size_align(100, 8).unwrap();
            allocator.dealloc(new_ptr, new_layout);
        }
    }

    #[test]
    fn test_global_allocator_management() {
        let allocator = Box::new(TrackedSystemAllocator::new(false));
        set_global_allocator(allocator);

        assert!(get_global_allocator_stats().is_some());
        assert_eq!(get_global_allocator_name(), Some("TrackedSystem"));

        reset_global_allocator_stats();

        let stats = get_global_allocator_stats().unwrap();
        assert_eq!(stats.total_allocations, 0);
    }
}
