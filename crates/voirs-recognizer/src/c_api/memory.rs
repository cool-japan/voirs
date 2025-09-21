//! Memory management utilities for the C API.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

/// Memory manager for C API allocations
pub struct VoirsMemoryManager {
    allocations: HashMap<*const c_void, Box<dyn std::any::Any + Send + Sync>>,
    string_allocations: Vec<CString>,
    ptr_array_allocations: Vec<Vec<*const c_char>>,
    segment_allocations: Vec<Vec<super::types::VoirsSegment>>,
    result_allocations: Vec<super::types::VoirsRecognitionResult>,
    next_id: usize,
}

impl VoirsMemoryManager {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            string_allocations: Vec::new(),
            ptr_array_allocations: Vec::new(),
            segment_allocations: Vec::new(),
            result_allocations: Vec::new(),
            next_id: 0,
        }
    }

    /// Store a string and return a C-compatible pointer
    pub fn store_string(&mut self, s: &str) -> *const c_char {
        match CString::new(s) {
            Ok(c_string) => {
                let ptr = c_string.as_ptr();
                self.string_allocations.push(c_string);
                ptr
            }
            Err(_) => ptr::null(),
        }
    }

    /// Store a struct and return a pointer to it
    pub fn store_struct<T: 'static + Send + Sync>(&mut self, data: T) -> *const T {
        let boxed = Box::new(data);
        let ptr = Box::as_ref(&boxed) as *const T;
        self.allocations.insert(ptr as *const c_void, boxed);
        ptr
    }

    /// Store an array and return a pointer to it
    pub fn store_array<T: 'static + Send + Sync + Clone>(&mut self, data: &[T]) -> *const T {
        if data.is_empty() {
            return ptr::null();
        }

        let vec = data.to_vec();
        let ptr = vec.as_ptr();
        let boxed = Box::new(vec);
        self.allocations.insert(ptr as *const c_void, boxed);
        ptr
    }

    /// Store an array of pointers
    pub fn store_ptr_array(&mut self, ptrs: &[*const c_char]) -> *const *const c_char {
        if ptrs.is_empty() {
            return ptr::null();
        }

        let vec = ptrs.to_vec();
        let ptr = vec.as_ptr();

        // Store in the dedicated ptr_array_allocations field to avoid Send+Sync issues
        self.ptr_array_allocations.push(vec);
        ptr
    }

    /// Store an array of VoirsSegment structures
    pub fn store_segments(
        &mut self,
        segments: &[super::types::VoirsSegment],
    ) -> *const super::types::VoirsSegment {
        if segments.is_empty() {
            return ptr::null();
        }

        let vec = segments.to_vec();
        let ptr = vec.as_ptr();

        // Store in the dedicated segment_allocations field to avoid Send+Sync issues
        self.segment_allocations.push(vec);
        ptr
    }

    /// Store a VoirsRecognitionResult structure
    pub fn store_result(
        &mut self,
        result: super::types::VoirsRecognitionResult,
    ) -> *const super::types::VoirsRecognitionResult {
        self.result_allocations.push(result);
        // Get pointer to the last element
        self.result_allocations.last().unwrap() as *const super::types::VoirsRecognitionResult
    }

    /// Mark a pointer for cleanup (for reference counting)
    pub fn mark_for_cleanup(&mut self, ptr: *const c_void) {
        // For now, we don't actually remove it since Rust's ownership
        // handles the cleanup when the manager is dropped
        // In a more sophisticated implementation, we might implement
        // reference counting here
    }

    /// Get the number of active allocations (for debugging)
    pub fn allocation_count(&self) -> usize {
        self.allocations.len() + self.string_allocations.len()
    }

    /// Get total memory usage estimate (for debugging)
    pub fn memory_usage_estimate(&self) -> usize {
        // This is a rough estimate
        let struct_memory = self.allocations.len() * std::mem::size_of::<*const c_void>();
        let string_memory: usize = self
            .string_allocations
            .iter()
            .map(|s| s.as_bytes().len())
            .sum();

        struct_memory + string_memory
    }
}

impl Drop for VoirsMemoryManager {
    fn drop(&mut self) {
        // All memory will be automatically freed when the HashMap and Vec are dropped
        // due to Rust's RAII
    }
}

impl Default for VoirsMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

// Utility functions for C string conversion

/// Convert a C string to a Rust String
pub fn c_string_to_string(c_str: *const c_char) -> Result<String, std::str::Utf8Error> {
    if c_str.is_null() {
        return Ok(String::new());
    }

    unsafe { CStr::from_ptr(c_str).to_str().map(|s| s.to_owned()) }
}

/// Convert a Rust string to a C string pointer (caller must free)
pub fn string_to_c_string(s: &str) -> *const c_char {
    match CString::new(s) {
        Ok(c_string) => {
            let ptr = c_string.as_ptr();
            // Leak the CString to keep the pointer valid
            std::mem::forget(c_string);
            ptr
        }
        Err(_) => ptr::null(),
    }
}

/// Copy a C string (returns newly allocated string that must be freed)
pub fn copy_c_string(c_str: *const c_char) -> *mut c_char {
    if c_str.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let rust_str = match CStr::from_ptr(c_str).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        match CString::new(rust_str) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }
}

/// Free a C string that was allocated by copy_c_string
pub unsafe fn free_c_string(c_str: *mut c_char) {
    if !c_str.is_null() {
        let _ = CString::from_raw(c_str);
    }
}

/// Validate that a pointer is not null and properly aligned
pub fn validate_pointer<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0
}

/// Calculate the size needed for a null-terminated array of pointers
pub fn calculate_ptr_array_size(ptrs: &[*const c_char]) -> usize {
    (ptrs.len() + 1) * std::mem::size_of::<*const c_char>()
}

/// Create a null-terminated array of C string pointers
pub fn create_null_terminated_ptr_array(ptrs: &[*const c_char]) -> Vec<*const c_char> {
    let mut result = ptrs.to_vec();
    result.push(ptr::null());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_string_storage() {
        let mut manager = VoirsMemoryManager::new();

        let test_str = "Hello, World!";
        let ptr = manager.store_string(test_str);

        assert!(!ptr.is_null());

        // Verify the string can be read back
        let read_back = c_string_to_string(ptr).unwrap();
        assert_eq!(read_back, test_str);
    }

    #[test]
    fn test_memory_manager_struct_storage() {
        let mut manager = VoirsMemoryManager::new();

        #[derive(Debug, PartialEq)]
        struct TestStruct {
            a: i32,
            b: f64,
        }

        let test_data = TestStruct { a: 42, b: 3.14 };
        let ptr = manager.store_struct(test_data);

        assert!(!ptr.is_null());

        // Verify the struct can be read back
        let read_back = unsafe { &*ptr };
        assert_eq!(read_back.a, 42);
        assert_eq!(read_back.b, 3.14);
    }

    #[test]
    fn test_c_string_conversion() {
        let test_str = "Test String";
        let c_str = string_to_c_string(test_str);

        assert!(!c_str.is_null());

        let converted_back = c_string_to_string(c_str).unwrap();
        assert_eq!(converted_back, test_str);

        // Clean up
        unsafe { free_c_string(c_str as *mut c_char) };
    }

    #[test]
    fn test_pointer_validation() {
        let valid_ptr: *const i32 = &42;
        assert!(validate_pointer(valid_ptr));

        let null_ptr: *const i32 = ptr::null();
        assert!(!validate_pointer(null_ptr));
    }

    #[test]
    fn test_memory_manager_allocation_count() {
        let mut manager = VoirsMemoryManager::new();

        assert_eq!(manager.allocation_count(), 0);

        manager.store_string("test");
        assert_eq!(manager.allocation_count(), 1);

        manager.store_struct(42i32);
        assert_eq!(manager.allocation_count(), 2);
    }
}
