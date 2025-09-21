//! Custom allocator C API functions.
//!
//! This module provides C API functions for setting and managing custom allocators,
//! getting allocation statistics, and controlling memory management behavior.

use crate::memory::allocators::{
    get_global_allocator_name, get_global_allocator_stats, reset_global_allocator_stats,
    set_global_allocator, DebugAllocator, PoolAllocator, TrackedSystemAllocator, VoirsAllocator,
};
use crate::{set_last_error, VoirsErrorCode};
use std::os::raw::{c_char, c_float, c_int, c_uint};

/// Allocator type constants
pub const VOIRS_ALLOCATOR_SYSTEM: c_int = 0;
pub const VOIRS_ALLOCATOR_POOL: c_int = 1;
pub const VOIRS_ALLOCATOR_DEBUG: c_int = 2;
pub const VOIRS_ALLOCATOR_TRACKED_SYSTEM: c_int = 3;

/// Memory allocation statistics structure for C API
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VoirsAllocatorStats {
    pub total_allocations: c_uint,
    pub total_deallocations: c_uint,
    pub current_allocations: c_uint,
    pub peak_allocations: c_uint,
    pub total_bytes_allocated: c_uint,
    pub total_bytes_deallocated: c_uint,
    pub current_bytes_allocated: c_uint,
    pub peak_bytes_allocated: c_uint,
}

impl Default for VoirsAllocatorStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            current_allocations: 0,
            peak_allocations: 0,
            total_bytes_allocated: 0,
            total_bytes_deallocated: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
        }
    }
}

/// Set the global allocator type
///
/// # Arguments
/// * `allocator_type` - Type of allocator (VOIRS_ALLOCATOR_*)
/// * `block_size` - Block size for pool allocator (ignored for other types)
/// * `blocks_per_chunk` - Blocks per chunk for pool allocator (ignored for other types)
/// * `enable_backtrace` - Enable backtrace tracking for tracked system allocator
///
/// Returns 0 on success, or error code on failure.
#[no_mangle]
pub extern "C" fn voirs_set_allocator(
    allocator_type: c_int,
    block_size: c_uint,
    blocks_per_chunk: c_uint,
    enable_backtrace: c_int,
) -> c_int {
    match set_allocator_impl(
        allocator_type,
        block_size,
        blocks_per_chunk,
        enable_backtrace != 0,
    ) {
        Ok(()) => 0,
        Err(code) => {
            set_last_error(format!("Failed to set allocator: {code:?}"));
            code as c_int
        }
    }
}

/// Get global allocator statistics
///
/// # Arguments
/// * `stats` - Pointer to VoirsAllocatorStats structure to fill
///
/// Returns 0 on success, or error code on failure.
#[no_mangle]
pub unsafe extern "C" fn voirs_get_allocator_stats(stats: *mut VoirsAllocatorStats) -> c_int {
    if stats.is_null() {
        set_last_error("Stats pointer is null".to_string());
        return VoirsErrorCode::InvalidParameter as c_int;
    }

    let allocator_stats = get_global_allocator_stats().unwrap_or_default();

    unsafe {
        let output_stats = &mut *stats;
        output_stats.total_allocations = allocator_stats.total_allocations as c_uint;
        output_stats.total_deallocations = allocator_stats.total_deallocations as c_uint;
        output_stats.current_allocations = allocator_stats.current_allocations as c_uint;
        output_stats.peak_allocations = allocator_stats.peak_allocations as c_uint;
        output_stats.total_bytes_allocated = allocator_stats.total_bytes_allocated as c_uint;
        output_stats.total_bytes_deallocated = allocator_stats.total_bytes_deallocated as c_uint;
        output_stats.current_bytes_allocated = allocator_stats.current_bytes_allocated as c_uint;
        output_stats.peak_bytes_allocated = allocator_stats.peak_bytes_allocated as c_uint;
    }

    0
}

/// Reset global allocator statistics
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn voirs_reset_allocator_stats() -> c_int {
    reset_global_allocator_stats();
    0
}

/// Get global allocator name
///
/// Returns pointer to null-terminated string with allocator name, or null if no allocator is set.
/// The returned string is valid until the allocator is changed.
#[no_mangle]
pub extern "C" fn voirs_get_allocator_name() -> *const c_char {
    match get_global_allocator_name() {
        Some(name) => name.as_ptr() as *const c_char,
        None => std::ptr::null(),
    }
}

/// Check if a custom allocator is currently set
///
/// Returns 1 if a custom allocator is set, 0 otherwise.
#[no_mangle]
pub extern "C" fn voirs_has_custom_allocator() -> c_int {
    if get_global_allocator_name().is_some() {
        1
    } else {
        0
    }
}

/// Get memory fragmentation ratio from the current allocator
///
/// Returns fragmentation ratio as a float between 0.0 and 1.0,
/// or -1.0 if the information is not available.
#[no_mangle]
pub extern "C" fn voirs_get_memory_fragmentation() -> c_float {
    if let Some(stats) = get_global_allocator_stats() {
        if stats.total_bytes_allocated > 0 {
            let fragmentation =
                1.0 - (stats.current_bytes_allocated as f32 / stats.total_bytes_allocated as f32);
            fragmentation.clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        -1.0
    }
}

// Implementation functions

fn set_allocator_impl(
    allocator_type: c_int,
    block_size: c_uint,
    blocks_per_chunk: c_uint,
    enable_backtrace: bool,
) -> Result<(), VoirsErrorCode> {
    let allocator: Box<dyn VoirsAllocator> = match allocator_type {
        VOIRS_ALLOCATOR_SYSTEM => {
            // Use system allocator without tracking
            return Err(VoirsErrorCode::InvalidParameter); // Not implemented as a VoirsAllocator
        }
        VOIRS_ALLOCATOR_TRACKED_SYSTEM => Box::new(TrackedSystemAllocator::new(enable_backtrace)),
        VOIRS_ALLOCATOR_POOL => {
            if block_size == 0 || blocks_per_chunk == 0 {
                return Err(VoirsErrorCode::InvalidParameter);
            }
            Box::new(PoolAllocator::new(
                block_size as usize,
                blocks_per_chunk as usize,
            ))
        }
        VOIRS_ALLOCATOR_DEBUG => {
            let base_allocator = Box::new(TrackedSystemAllocator::new(enable_backtrace));
            Box::new(DebugAllocator::new(base_allocator))
        }
        _ => {
            return Err(VoirsErrorCode::InvalidParameter);
        }
    };

    set_global_allocator(allocator);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_tracked_system_allocator() {
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_TRACKED_SYSTEM, 0, 0, 0);
        assert_eq!(result, 0, "Setting tracked system allocator should succeed");

        let has_custom = voirs_has_custom_allocator();
        assert_eq!(has_custom, 1, "Should have custom allocator set");

        let name = voirs_get_allocator_name();
        assert!(!name.is_null(), "Allocator name should not be null");
    }

    #[test]
    fn test_set_pool_allocator() {
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_POOL, 64, 100, 0);
        assert_eq!(result, 0, "Setting pool allocator should succeed");

        let has_custom = voirs_has_custom_allocator();
        assert_eq!(has_custom, 1, "Should have custom allocator set");
    }

    #[test]
    fn test_set_debug_allocator() {
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_DEBUG, 0, 0, 1);
        assert_eq!(result, 0, "Setting debug allocator should succeed");

        let has_custom = voirs_has_custom_allocator();
        assert_eq!(has_custom, 1, "Should have custom allocator set");
    }

    #[test]
    fn test_invalid_allocator_type() {
        let result = voirs_set_allocator(999, 0, 0, 0);
        assert_ne!(result, 0, "Setting invalid allocator type should fail");
    }

    #[test]
    fn test_invalid_pool_parameters() {
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_POOL, 0, 100, 0);
        assert_ne!(result, 0, "Pool allocator with zero block size should fail");

        let result = voirs_set_allocator(VOIRS_ALLOCATOR_POOL, 64, 0, 0);
        assert_ne!(
            result, 0,
            "Pool allocator with zero blocks per chunk should fail"
        );
    }

    #[test]
    fn test_allocator_stats() {
        // Set a tracked allocator
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_TRACKED_SYSTEM, 0, 0, 0);
        assert_eq!(result, 0, "Setting tracked system allocator should succeed");

        // Get stats
        let mut stats = VoirsAllocatorStats::default();
        let result = unsafe { voirs_get_allocator_stats(&mut stats) };
        assert_eq!(result, 0, "Getting allocator stats should succeed");

        // Reset stats
        let result = voirs_reset_allocator_stats();
        assert_eq!(result, 0, "Resetting allocator stats should succeed");

        // Get stats again to verify reset
        let mut new_stats = VoirsAllocatorStats::default();
        let result = unsafe { voirs_get_allocator_stats(&mut new_stats) };
        assert_eq!(
            result, 0,
            "Getting allocator stats after reset should succeed"
        );
    }

    #[test]
    fn test_null_stats_pointer() {
        let result = unsafe { voirs_get_allocator_stats(std::ptr::null_mut()) };
        assert_ne!(result, 0, "Getting stats with null pointer should fail");
    }

    #[test]
    fn test_memory_fragmentation() {
        // Set a tracked allocator
        let result = voirs_set_allocator(VOIRS_ALLOCATOR_TRACKED_SYSTEM, 0, 0, 0);
        assert_eq!(result, 0, "Setting tracked system allocator should succeed");

        let fragmentation = voirs_get_memory_fragmentation();
        assert!(fragmentation >= 0.0, "Fragmentation should be non-negative");
        assert!(fragmentation <= 1.0, "Fragmentation should be at most 1.0");
    }
}
