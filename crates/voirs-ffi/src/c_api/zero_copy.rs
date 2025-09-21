//! C API for zero-copy operations
//!
//! This module provides C-compatible functions for zero-copy buffer management,
//! memory mapping, and shared memory operations.

use crate::memory::zero_copy::*;
use crate::{string_to_c_str, VoirsErrorCode};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;

/// Opaque handle for zero-copy buffer
pub type VoirsZeroCopyBuffer = *mut ZeroCopyBuffer<f32>;

/// Opaque handle for zero-copy view
pub type VoirsZeroCopyView = *mut ZeroCopyView<f32>;

/// Opaque handle for zero-copy ring buffer
pub type VoirsZeroCopyRingBuffer = *mut ZeroCopyRingBuffer<f32>;

/// Opaque handle for memory mapped file
pub type VoirsMemoryMappedFile = *mut MemoryMappedFile;

/// Create a new zero-copy buffer
#[no_mangle]
pub extern "C" fn voirs_zero_copy_buffer_create(capacity: c_uint) -> VoirsZeroCopyBuffer {
    if capacity == 0 {
        return ptr::null_mut();
    }

    match ZeroCopyBuffer::<f32>::new(capacity as usize) {
        Ok(buffer) => Box::into_raw(Box::new(buffer)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a zero-copy buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_destroy(buffer: VoirsZeroCopyBuffer) {
    if !buffer.is_null() {
        let _ = Box::from_raw(buffer);
    }
}

/// Get the length of valid data in the buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_len(buffer: VoirsZeroCopyBuffer) -> c_uint {
    if buffer.is_null() {
        return 0;
    }

    (*buffer).len() as c_uint
}

/// Get the capacity of the buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_capacity(buffer: VoirsZeroCopyBuffer) -> c_uint {
    if buffer.is_null() {
        return 0;
    }

    (*buffer).capacity() as c_uint
}

/// Set the length of valid data in the buffer
/// # Safety
/// The caller must ensure that data up to `new_len` is properly initialized
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_set_len(
    buffer: VoirsZeroCopyBuffer,
    new_len: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let buffer_ref = &mut *buffer;
    buffer_ref.set_len(new_len as usize);
    VoirsErrorCode::Success
}

/// Get a pointer to the buffer data
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_data(
    buffer: VoirsZeroCopyBuffer,
) -> *const c_float {
    if buffer.is_null() {
        return ptr::null();
    }

    (*buffer).as_slice().as_ptr()
}

/// Get a mutable pointer to the buffer data
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_data_mut(
    buffer: VoirsZeroCopyBuffer,
) -> *mut c_float {
    if buffer.is_null() {
        return ptr::null_mut();
    }

    (*buffer).as_mut_slice().as_mut_ptr()
}

/// Clone a zero-copy buffer handle (increases reference count)
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_clone(
    buffer: VoirsZeroCopyBuffer,
) -> VoirsZeroCopyBuffer {
    if buffer.is_null() {
        return ptr::null_mut();
    }

    let cloned = (*buffer).clone_handle();
    Box::into_raw(Box::new(cloned))
}

/// Get the reference count of a zero-copy buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_ref_count(buffer: VoirsZeroCopyBuffer) -> c_uint {
    if buffer.is_null() {
        return 0;
    }

    (*buffer).ref_count() as c_uint
}

/// Create a zero-copy view of a buffer slice
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_buffer_slice(
    buffer: VoirsZeroCopyBuffer,
    start: c_uint,
    end: c_uint,
) -> VoirsZeroCopyView {
    if buffer.is_null() || start >= end {
        return ptr::null_mut();
    }

    match (*buffer).slice(start as usize, end as usize) {
        Ok(view) => Box::into_raw(Box::new(view)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a zero-copy view
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_view_destroy(view: VoirsZeroCopyView) {
    if !view.is_null() {
        let _ = Box::from_raw(view);
    }
}

/// Get the length of a zero-copy view
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_view_len(view: VoirsZeroCopyView) -> c_uint {
    if view.is_null() {
        return 0;
    }

    (*view).len() as c_uint
}

/// Get a pointer to the view data
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_view_data(view: VoirsZeroCopyView) -> *const c_float {
    if view.is_null() {
        return ptr::null();
    }

    (*view).as_slice().as_ptr()
}

/// Create a zero-copy ring buffer
#[no_mangle]
pub extern "C" fn voirs_zero_copy_ring_create(capacity: c_uint) -> VoirsZeroCopyRingBuffer {
    if capacity == 0 || !capacity.is_power_of_two() {
        return ptr::null_mut();
    }

    match ZeroCopyRingBuffer::<f32>::new(capacity as usize) {
        Ok(ring) => Box::into_raw(Box::new(ring)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a zero-copy ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_destroy(ring: VoirsZeroCopyRingBuffer) {
    if !ring.is_null() {
        let _ = Box::from_raw(ring);
    }
}

/// Write data to a zero-copy ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_write(
    ring: VoirsZeroCopyRingBuffer,
    data: *const c_float,
    len: c_uint,
) -> c_uint {
    if ring.is_null() || data.is_null() || len == 0 {
        return 0;
    }

    let slice = std::slice::from_raw_parts(data, len as usize);
    (*ring).write(slice) as c_uint
}

/// Read data from a zero-copy ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_read(
    ring: VoirsZeroCopyRingBuffer,
    data: *mut c_float,
    len: c_uint,
) -> c_uint {
    if ring.is_null() || data.is_null() || len == 0 {
        return 0;
    }

    let slice = std::slice::from_raw_parts_mut(data, len as usize);
    (*ring).read(slice) as c_uint
}

/// Get available data for reading in ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_available_read(
    ring: VoirsZeroCopyRingBuffer,
) -> c_uint {
    if ring.is_null() {
        return 0;
    }

    (*ring).available_read() as c_uint
}

/// Get available space for writing in ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_available_write(
    ring: VoirsZeroCopyRingBuffer,
) -> c_uint {
    if ring.is_null() {
        return 0;
    }

    (*ring).available_write() as c_uint
}

/// Get capacity of ring buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_ring_capacity(ring: VoirsZeroCopyRingBuffer) -> c_uint {
    if ring.is_null() {
        return 0;
    }

    (*ring).capacity() as c_uint
}

/// Open a memory mapped file for reading
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_open_read(path: *const c_char) -> VoirsMemoryMappedFile {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = std::ffi::CStr::from_ptr(path);
    if let Ok(path_str) = c_str.to_str() {
        match MemoryMappedFile::open_read_only(path_str) {
            Ok(mmap) => Box::into_raw(Box::new(mmap)),
            Err(_) => ptr::null_mut(),
        }
    } else {
        ptr::null_mut()
    }
}

/// Open a memory mapped file for reading and writing
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_open_write(
    path: *const c_char,
    size: c_uint,
) -> VoirsMemoryMappedFile {
    if path.is_null() || size == 0 {
        return ptr::null_mut();
    }

    let c_str = std::ffi::CStr::from_ptr(path);
    if let Ok(path_str) = c_str.to_str() {
        match MemoryMappedFile::open_read_write(path_str, size as usize) {
            Ok(mmap) => Box::into_raw(Box::new(mmap)),
            Err(_) => ptr::null_mut(),
        }
    } else {
        ptr::null_mut()
    }
}

/// Destroy a memory mapped file
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_destroy(mmap: VoirsMemoryMappedFile) {
    if !mmap.is_null() {
        let _ = Box::from_raw(mmap);
    }
}

/// Get the size of a memory mapped file
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_size(mmap: VoirsMemoryMappedFile) -> c_uint {
    if mmap.is_null() {
        return 0;
    }

    (*mmap).len() as c_uint
}

/// Get a pointer to the memory mapped data
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_data(mmap: VoirsMemoryMappedFile) -> *const c_void {
    if mmap.is_null() {
        return ptr::null();
    }

    (*mmap).as_slice().as_ptr() as *const c_void
}

/// Get a mutable pointer to the memory mapped data
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_data_mut(mmap: VoirsMemoryMappedFile) -> *mut c_void {
    if mmap.is_null() {
        return ptr::null_mut();
    }

    match (*mmap).as_mut_slice() {
        Ok(slice) => slice.as_mut_ptr() as *mut c_void,
        Err(_) => ptr::null_mut(),
    }
}

/// Synchronize memory mapped file changes to disk
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_sync(mmap: VoirsMemoryMappedFile) -> VoirsErrorCode {
    if mmap.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    match (*mmap).sync() {
        Ok(()) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::InternalError,
    }
}

/// Set memory access pattern advice for memory mapped file
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_advise_sequential(
    mmap: VoirsMemoryMappedFile,
) -> VoirsErrorCode {
    if mmap.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    match (*mmap).advise_sequential() {
        Ok(()) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::InternalError,
    }
}

/// Set random access pattern advice for memory mapped file
#[no_mangle]
pub unsafe extern "C" fn voirs_memory_map_advise_random(
    mmap: VoirsMemoryMappedFile,
) -> VoirsErrorCode {
    if mmap.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    match (*mmap).advise_random() {
        Ok(()) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::InternalError,
    }
}

/// Batch copy operation using zero-copy optimization where possible
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_batch_copy(
    sources: *const *const c_float,
    destinations: *const *mut c_float,
    sizes: *const c_uint,
    count: c_uint,
) -> VoirsErrorCode {
    if sources.is_null() || destinations.is_null() || sizes.is_null() || count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let src_ptrs = std::slice::from_raw_parts(sources, count as usize);
    let dst_ptrs = std::slice::from_raw_parts(destinations, count as usize);
    let size_array = std::slice::from_raw_parts(sizes, count as usize);

    for i in 0..count as usize {
        if src_ptrs[i].is_null() || dst_ptrs[i].is_null() || size_array[i] == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let src_slice = std::slice::from_raw_parts(src_ptrs[i], size_array[i] as usize);
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptrs[i], size_array[i] as usize);

        // Use optimized copy
        dst_slice.copy_from_slice(src_slice);
    }

    VoirsErrorCode::Success
}

/// Zero-copy audio buffer interleaving
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_interleave(
    channels: *const *const c_float,
    channel_count: c_uint,
    frame_count: c_uint,
    output: *mut c_float,
) -> VoirsErrorCode {
    if channels.is_null() || output.is_null() || channel_count == 0 || frame_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let channel_ptrs = std::slice::from_raw_parts(channels, channel_count as usize);
    let output_slice =
        std::slice::from_raw_parts_mut(output, (channel_count * frame_count) as usize);

    // Verify all channel pointers are valid
    for &ptr in channel_ptrs {
        if ptr.is_null() {
            return VoirsErrorCode::InvalidParameter;
        }
    }

    // Interleave the audio data
    for frame in 0..frame_count as usize {
        for (ch, &channel_ptr) in channel_ptrs.iter().enumerate() {
            let channel_slice = std::slice::from_raw_parts(channel_ptr, frame_count as usize);
            let output_index = frame * channel_count as usize + ch;
            output_slice[output_index] = channel_slice[frame];
        }
    }

    VoirsErrorCode::Success
}

/// Zero-copy audio buffer deinterleaving
#[no_mangle]
pub unsafe extern "C" fn voirs_zero_copy_deinterleave(
    input: *const c_float,
    channels: *const *mut c_float,
    channel_count: c_uint,
    frame_count: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || channels.is_null() || channel_count == 0 || frame_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let input_slice = std::slice::from_raw_parts(input, (channel_count * frame_count) as usize);
    let channel_ptrs = std::slice::from_raw_parts(channels, channel_count as usize);

    // Verify all channel pointers are valid
    for &ptr in channel_ptrs {
        if ptr.is_null() {
            return VoirsErrorCode::InvalidParameter;
        }
    }

    // Deinterleave the audio data
    for frame in 0..frame_count as usize {
        for (ch, &channel_ptr) in channel_ptrs.iter().enumerate() {
            let channel_slice = std::slice::from_raw_parts_mut(channel_ptr, frame_count as usize);
            let input_index = frame * channel_count as usize + ch;
            channel_slice[frame] = input_slice[input_index];
        }
    }

    VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_c_api_zero_copy_buffer() {
        unsafe {
            let buffer = voirs_zero_copy_buffer_create(1024);
            assert!(!buffer.is_null());

            assert_eq!(voirs_zero_copy_buffer_capacity(buffer), 1024);
            assert_eq!(voirs_zero_copy_buffer_len(buffer), 0);

            assert_eq!(
                voirs_zero_copy_buffer_set_len(buffer, 10),
                VoirsErrorCode::Success
            );
            assert_eq!(voirs_zero_copy_buffer_len(buffer), 10);

            let data_ptr = voirs_zero_copy_buffer_data_mut(buffer);
            assert!(!data_ptr.is_null());

            // Test cloning
            let cloned = voirs_zero_copy_buffer_clone(buffer);
            assert!(!cloned.is_null());
            assert_eq!(voirs_zero_copy_buffer_ref_count(buffer), 2);

            voirs_zero_copy_buffer_destroy(cloned);
            assert_eq!(voirs_zero_copy_buffer_ref_count(buffer), 1);

            voirs_zero_copy_buffer_destroy(buffer);
        }
    }

    #[test]
    fn test_c_api_zero_copy_ring() {
        unsafe {
            let ring = voirs_zero_copy_ring_create(16);
            assert!(!ring.is_null());

            assert_eq!(voirs_zero_copy_ring_capacity(ring), 16);
            assert_eq!(voirs_zero_copy_ring_available_read(ring), 0);
            assert_eq!(voirs_zero_copy_ring_available_write(ring), 16);

            let data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let written = voirs_zero_copy_ring_write(ring, data.as_ptr(), 5);
            assert_eq!(written, 5);

            assert_eq!(voirs_zero_copy_ring_available_read(ring), 5);
            assert_eq!(voirs_zero_copy_ring_available_write(ring), 11);

            let mut output = [0.0f32; 10];
            let read = voirs_zero_copy_ring_read(ring, output.as_mut_ptr(), 3);
            assert_eq!(read, 3);
            assert_eq!(output[0], 1.0);
            assert_eq!(output[1], 2.0);
            assert_eq!(output[2], 3.0);

            voirs_zero_copy_ring_destroy(ring);
        }
    }

    #[test]
    fn test_c_api_batch_copy() {
        unsafe {
            let src1 = [1.0, 2.0, 3.0];
            let src2 = [4.0, 5.0, 6.0, 7.0];
            let mut dst1 = [0.0f32; 3];
            let mut dst2 = [0.0f32; 4];

            let sources = [src1.as_ptr(), src2.as_ptr()];
            let destinations = [dst1.as_mut_ptr(), dst2.as_mut_ptr()];
            let sizes = [3, 4];

            let result = voirs_zero_copy_batch_copy(
                sources.as_ptr(),
                destinations.as_ptr(),
                sizes.as_ptr(),
                2,
            );

            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(dst1, [1.0, 2.0, 3.0]);
            assert_eq!(dst2, [4.0, 5.0, 6.0, 7.0]);
        }
    }

    #[test]
    fn test_c_api_interleave_deinterleave() {
        unsafe {
            let left = [1.0, 3.0, 5.0];
            let right = [2.0, 4.0, 6.0];
            let channels = [left.as_ptr(), right.as_ptr()];
            let mut interleaved = [0.0f32; 6];

            let result =
                voirs_zero_copy_interleave(channels.as_ptr(), 2, 3, interleaved.as_mut_ptr());

            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(interleaved, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

            // Test deinterleaving
            let mut out_left = [0.0f32; 3];
            let mut out_right = [0.0f32; 3];
            let out_channels = [out_left.as_mut_ptr(), out_right.as_mut_ptr()];

            let result =
                voirs_zero_copy_deinterleave(interleaved.as_ptr(), out_channels.as_ptr(), 2, 3);

            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(out_left, [1.0, 3.0, 5.0]);
            assert_eq!(out_right, [2.0, 4.0, 6.0]);
        }
    }
}
