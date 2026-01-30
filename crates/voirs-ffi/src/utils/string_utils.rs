//! String conversion utilities for FFI.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr,
};

/// Convert Rust string to owned C string
pub fn string_to_owned_c_str(s: &str) -> Result<CString, std::ffi::NulError> {
    CString::new(s)
}

/// Convert C string to Rust string slice (unsafe)
///
/// # Safety
/// The `c_str` pointer, if not null, must point to a valid null-terminated C string.
/// The string must remain valid for the lifetime 'a.
pub unsafe fn c_str_to_str<'a>(c_str: *const c_char) -> Result<&'a str, std::str::Utf8Error> {
    if c_str.is_null() {
        return Ok("");
    }
    CStr::from_ptr(c_str).to_str()
}

/// Create a null-terminated string array for C (optimized)
pub fn create_string_array(strings: &[String]) -> (*mut *mut c_char, usize) {
    let len = strings.len();
    // Pre-allocate with exact capacity + 1 for null terminator
    let mut ptrs = Vec::with_capacity(len + 1);

    // Convert strings directly without intermediate collection
    for s in strings {
        let ptr = match CString::new(s.as_str()) {
            Ok(cs) => cs.into_raw(),
            Err(_) => ptr::null_mut(),
        };
        ptrs.push(ptr);
    }

    ptrs.push(ptr::null_mut()); // Null terminator

    let array = ptrs.into_boxed_slice();
    let ptr = Box::into_raw(array) as *mut *mut c_char;

    (ptr, len)
}

/// Free a string array created by create_string_array
///
/// # Safety
/// The `array` pointer must have been previously returned by `create_string_array`.
/// The `len` value must match the length that was used when creating the array.
/// After calling this function, the array pointer becomes invalid and must not be used.
pub unsafe fn free_string_array(array: *mut *mut c_char, len: usize) {
    if array.is_null() {
        return;
    }

    for i in 0..len {
        let str_ptr = *array.add(i);
        if !str_ptr.is_null() {
            let _ = CString::from_raw(str_ptr);
        }
    }

    let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(array, len + 1));
}
