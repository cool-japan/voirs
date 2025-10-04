//! Zero-copy operations for high-performance audio processing
//!
//! This module provides zero-copy operations, memory mapping, and shared memory
//! optimizations for minimal data copying overhead in FFI operations.

use parking_lot::RwLock;
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Zero-copy buffer that can be shared between threads without copying data
pub struct ZeroCopyBuffer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    len: usize,
    shared_state: Arc<ZeroCopyState>,
    _phantom: PhantomData<T>,
}

/// Shared state for zero-copy buffer reference counting and metadata
struct ZeroCopyState {
    ref_count: AtomicUsize,
    layout: Layout,
    is_mapped: AtomicUsize,
}

impl<T> ZeroCopyBuffer<T> {
    /// Create a new zero-copy buffer with specified capacity
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("Capacity cannot be zero");
        }

        let layout = Layout::array::<T>(capacity).map_err(|_| "Invalid layout")?;

        unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                return Err("Memory allocation failed");
            }

            let non_null_ptr = NonNull::new_unchecked(ptr);

            Ok(Self {
                ptr: non_null_ptr,
                capacity,
                len: 0,
                shared_state: Arc::new(ZeroCopyState {
                    ref_count: AtomicUsize::new(1),
                    layout,
                    is_mapped: AtomicUsize::new(0),
                }),
                _phantom: PhantomData,
            })
        }
    }

    /// Create a zero-copy buffer from existing memory (takes ownership)
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        capacity: usize,
        len: usize,
    ) -> Result<Self, &'static str> {
        if ptr.is_null() {
            return Err("Null pointer provided");
        }
        if len > capacity {
            return Err("Length exceeds capacity");
        }

        let layout = Layout::array::<T>(capacity).map_err(|_| "Invalid layout")?;
        let non_null_ptr = NonNull::new_unchecked(ptr);

        Ok(Self {
            ptr: non_null_ptr,
            capacity,
            len,
            shared_state: Arc::new(ZeroCopyState {
                ref_count: AtomicUsize::new(1),
                layout,
                is_mapped: AtomicUsize::new(0),
            }),
            _phantom: PhantomData,
        })
    }

    /// Get the length of valid data in the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Set the length of valid data (unsafe - caller must ensure data is initialized)
    pub unsafe fn set_len(&mut self, new_len: usize) {
        if new_len <= self.capacity {
            self.len = new_len;
        }
    }

    /// Get a slice view of the valid data
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice view of the valid data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get a slice view of the entire capacity
    pub fn as_full_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    /// Get a mutable slice view of the entire capacity
    pub fn as_full_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    /// Create a zero-copy view of a portion of this buffer
    pub fn slice(&self, start: usize, end: usize) -> Result<ZeroCopyView<T>, &'static str> {
        if start > end || end > self.len {
            return Err("Invalid slice bounds");
        }

        // Increment reference count
        self.shared_state.ref_count.fetch_add(1, Ordering::Relaxed);

        Ok(ZeroCopyView {
            ptr: unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(start)) },
            len: end - start,
            shared_state: Arc::clone(&self.shared_state),
            _phantom: PhantomData,
        })
    }

    /// Get the current reference count
    pub fn ref_count(&self) -> usize {
        self.shared_state.ref_count.load(Ordering::Relaxed)
    }

    /// Clone the buffer handle (increases reference count, doesn't copy data)
    pub fn clone_handle(&self) -> Self {
        self.shared_state.ref_count.fetch_add(1, Ordering::Relaxed);
        Self {
            ptr: self.ptr,
            capacity: self.capacity,
            len: self.len,
            shared_state: Arc::clone(&self.shared_state),
            _phantom: PhantomData,
        }
    }
}

/// Zero-copy view into a buffer (doesn't own the memory)
pub struct ZeroCopyView<T> {
    ptr: NonNull<T>,
    len: usize,
    shared_state: Arc<ZeroCopyState>,
    _phantom: PhantomData<T>,
}

impl<T> ZeroCopyView<T> {
    /// Get the length of the view
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice view of the data
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Create a sub-view of this view
    pub fn subview(&self, start: usize, end: usize) -> Result<ZeroCopyView<T>, &'static str> {
        if start > end || end > self.len {
            return Err("Invalid subview bounds");
        }

        // Increment reference count
        self.shared_state.ref_count.fetch_add(1, Ordering::Relaxed);

        Ok(ZeroCopyView {
            ptr: unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(start)) },
            len: end - start,
            shared_state: Arc::clone(&self.shared_state),
            _phantom: PhantomData,
        })
    }
}

impl<T> Deref for ZeroCopyView<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> Drop for ZeroCopyBuffer<T> {
    fn drop(&mut self) {
        let prev_count = self.shared_state.ref_count.fetch_sub(1, Ordering::Relaxed);
        if prev_count == 1 {
            // Last reference, deallocate memory
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.shared_state.layout);
            }
        }
    }
}

impl<T> Drop for ZeroCopyView<T> {
    fn drop(&mut self) {
        let prev_count = self.shared_state.ref_count.fetch_sub(1, Ordering::Relaxed);
        if prev_count == 1 {
            // Last reference, deallocate memory
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.shared_state.layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for ZeroCopyBuffer<T> {}
unsafe impl<T: Sync> Sync for ZeroCopyBuffer<T> {}
unsafe impl<T: Send> Send for ZeroCopyView<T> {}
unsafe impl<T: Sync> Sync for ZeroCopyView<T> {}

/// Memory-mapped file for zero-copy file I/O operations
pub struct MemoryMappedFile {
    ptr: NonNull<u8>,
    len: usize,
    read_only: bool,
    #[cfg(unix)]
    file_descriptor: std::os::unix::io::RawFd,
    #[cfg(windows)]
    file_handle: std::os::windows::io::RawHandle,
    #[cfg(windows)]
    mapping_handle: std::os::windows::io::RawHandle,
}

impl MemoryMappedFile {
    /// Memory map a file for reading
    #[cfg(unix)]
    pub fn open_read_only(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::ffi::CString;
        use std::os::unix::io::RawFd;

        let c_path = CString::new(path)?;

        unsafe {
            let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY);
            if fd == -1 {
                return Err("Failed to open file".into());
            }

            let mut stat: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut stat) == -1 {
                libc::close(fd);
                return Err("Failed to get file stats".into());
            }

            let len = stat.st_size as usize;
            if len == 0 {
                libc::close(fd);
                return Err("Cannot map empty file".into());
            }

            let ptr = libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err("Memory mapping failed".into());
            }

            Ok(Self {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                len,
                read_only: true,
                file_descriptor: fd,
            })
        }
    }

    /// Memory map a file for reading and writing
    #[cfg(unix)]
    pub fn open_read_write(path: &str, size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        use std::ffi::CString;

        let c_path = CString::new(path)?;

        unsafe {
            let fd = libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o644);
            if fd == -1 {
                return Err("Failed to open/create file".into());
            }

            // Extend file to desired size
            if libc::ftruncate(fd, size as i64) == -1 {
                libc::close(fd);
                return Err("Failed to set file size".into());
            }

            let ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err("Memory mapping failed".into());
            }

            Ok(Self {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                len: size,
                read_only: false,
                file_descriptor: fd,
            })
        }
    }

    /// Stub implementation for non-Unix platforms
    #[cfg(not(unix))]
    pub fn open_read_only(_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Memory mapping not implemented for this platform".into())
    }

    #[cfg(not(unix))]
    pub fn open_read_write(_path: &str, _size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Memory mapping not implemented for this platform".into())
    }

    /// Get the length of the mapped region
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the mapping is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a slice view of the mapped data
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice view of the mapped data (if writable)
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], &'static str> {
        if self.read_only {
            return Err("Cannot get mutable slice of read-only mapping");
        }
        Ok(unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }

    /// Synchronize changes to disk
    #[cfg(unix)]
    pub fn sync(&self) -> Result<(), &'static str> {
        unsafe {
            if libc::msync(
                self.ptr.as_ptr() as *mut libc::c_void,
                self.len,
                libc::MS_SYNC,
            ) == -1
            {
                return Err("Failed to sync memory mapping");
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn sync(&self) -> Result<(), &'static str> {
        Err("Sync not implemented for this platform")
    }

    /// Advise the kernel about memory usage patterns
    #[cfg(unix)]
    pub fn advise_sequential(&self) -> Result<(), &'static str> {
        unsafe {
            if libc::madvise(
                self.ptr.as_ptr() as *mut libc::c_void,
                self.len,
                libc::MADV_SEQUENTIAL,
            ) == -1
            {
                return Err("Failed to set memory advice");
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    pub fn advise_random(&self) -> Result<(), &'static str> {
        unsafe {
            if libc::madvise(
                self.ptr.as_ptr() as *mut libc::c_void,
                self.len,
                libc::MADV_RANDOM,
            ) == -1
            {
                return Err("Failed to set memory advice");
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn advise_sequential(&self) -> Result<(), &'static str> {
        Ok(()) // No-op on non-Unix platforms
    }

    #[cfg(not(unix))]
    pub fn advise_random(&self) -> Result<(), &'static str> {
        Ok(()) // No-op on non-Unix platforms
    }
}

impl Drop for MemoryMappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
            libc::close(self.file_descriptor);
        }
    }
}

/// Shared memory segment for inter-process zero-copy communication
pub struct SharedMemorySegment {
    ptr: NonNull<u8>,
    size: usize,
    name: String,
    #[cfg(unix)]
    shm_fd: std::os::unix::io::RawFd,
}

impl SharedMemorySegment {
    /// Create a new shared memory segment
    #[cfg(unix)]
    pub fn create(name: &str, size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        use std::ffi::CString;

        let c_name = CString::new(name)?;

        unsafe {
            let shm_fd = libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644);
            if shm_fd == -1 {
                return Err("Failed to create shared memory".into());
            }

            if libc::ftruncate(shm_fd, size as i64) == -1 {
                libc::close(shm_fd);
                libc::shm_unlink(c_name.as_ptr());
                return Err("Failed to set shared memory size".into());
            }

            let ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(shm_fd);
                libc::shm_unlink(c_name.as_ptr());
                return Err("Failed to map shared memory".into());
            }

            Ok(Self {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                size,
                name: name.to_string(),
                shm_fd,
            })
        }
    }

    /// Open an existing shared memory segment
    #[cfg(unix)]
    pub fn open(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::ffi::CString;

        let c_name = CString::new(name)?;

        unsafe {
            let shm_fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0);
            if shm_fd == -1 {
                return Err("Failed to open shared memory".into());
            }

            let mut stat: libc::stat = std::mem::zeroed();
            if libc::fstat(shm_fd, &mut stat) == -1 {
                libc::close(shm_fd);
                return Err("Failed to get shared memory stats".into());
            }

            let size = stat.st_size as usize;

            let ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(shm_fd);
                return Err("Failed to map shared memory".into());
            }

            Ok(Self {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                size,
                name: name.to_string(),
                shm_fd,
            })
        }
    }

    #[cfg(not(unix))]
    pub fn create(_name: &str, _size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Shared memory not implemented for this platform".into())
    }

    #[cfg(not(unix))]
    pub fn open(_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Shared memory not implemented for this platform".into())
    }

    /// Get the size of the shared memory segment
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a slice view of the shared memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get a mutable slice view of the shared memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for SharedMemorySegment {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size);
            libc::close(self.shm_fd);
            // Note: We don't unlink here as other processes might still be using it
        }
    }
}

/// Zero-copy ring buffer for high-performance streaming
pub struct ZeroCopyRingBuffer<T> {
    buffer: ZeroCopyBuffer<T>,
    read_pos: AtomicUsize,
    write_pos: AtomicUsize,
    mask: usize, // capacity - 1 (assumes power of 2)
}

impl<T> ZeroCopyRingBuffer<T> {
    /// Create a new zero-copy ring buffer (capacity must be power of 2)
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if !capacity.is_power_of_two() {
            return Err("Capacity must be a power of 2");
        }

        let buffer = ZeroCopyBuffer::new(capacity)?;

        Ok(Self {
            buffer,
            read_pos: AtomicUsize::new(0),
            write_pos: AtomicUsize::new(0),
            mask: capacity - 1,
        })
    }

    /// Try to write data to the ring buffer (returns number of items written)
    pub fn write(&self, data: &[T]) -> usize
    where
        T: Copy,
    {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let available = self.mask + 1 - (write_pos.wrapping_sub(read_pos));
        let to_write = data.len().min(available);

        if to_write == 0 {
            return 0;
        }

        unsafe {
            let buffer_slice = self.buffer.as_full_slice();
            for i in 0..to_write {
                let idx = (write_pos + i) & self.mask;
                // Safety: We're writing to uninitialized memory, but T: Copy ensures it's safe
                ptr::write(buffer_slice.as_ptr().add(idx) as *mut T, data[i]);
            }
        }

        self.write_pos
            .store(write_pos + to_write, Ordering::Release);
        to_write
    }

    /// Try to read data from the ring buffer (returns number of items read)
    pub fn read(&self, output: &mut [T]) -> usize
    where
        T: Copy,
    {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        let available = write_pos.wrapping_sub(read_pos);
        let to_read = output.len().min(available);

        if to_read == 0 {
            return 0;
        }

        let buffer_slice = self.buffer.as_full_slice();
        for i in 0..to_read {
            let idx = (read_pos + i) & self.mask;
            output[i] = buffer_slice[idx];
        }

        self.read_pos.store(read_pos + to_read, Ordering::Release);
        to_read
    }

    /// Get the number of items available for reading
    pub fn available_read(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        write_pos.wrapping_sub(read_pos)
    }

    /// Get the number of items that can be written
    pub fn available_write(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        self.mask + 1 - write_pos.wrapping_sub(read_pos)
    }

    /// Get the capacity of the ring buffer
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }
}

unsafe impl<T: Send> Send for ZeroCopyRingBuffer<T> {}
unsafe impl<T: Sync> Sync for ZeroCopyRingBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_buffer() {
        let mut buffer = ZeroCopyBuffer::<f32>::new(1024).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        unsafe {
            buffer.set_len(10);
        }
        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_empty());

        let slice = buffer.as_mut_slice();
        slice[0] = 1.0;
        slice[9] = 9.0;

        assert_eq!(buffer.as_slice()[0], 1.0);
        assert_eq!(buffer.as_slice()[9], 9.0);
    }

    #[test]
    fn test_zero_copy_view() {
        let mut buffer = ZeroCopyBuffer::<i32>::new(100).unwrap();
        unsafe {
            buffer.set_len(100);
        }

        let slice = buffer.as_mut_slice();
        for (i, item) in slice.iter_mut().enumerate() {
            *item = i as i32;
        }

        let view = buffer.slice(10, 20).unwrap();
        assert_eq!(view.len(), 10);
        assert_eq!(view[0], 10);
        assert_eq!(view[9], 19);

        let subview = view.subview(2, 5).unwrap();
        assert_eq!(subview.len(), 3);
        assert_eq!(subview[0], 12);
        assert_eq!(subview[2], 14);
    }

    #[test]
    fn test_zero_copy_buffer_clone() {
        let buffer = ZeroCopyBuffer::<u8>::new(64).unwrap();
        assert_eq!(buffer.ref_count(), 1);

        let cloned = buffer.clone_handle();
        assert_eq!(buffer.ref_count(), 2);
        assert_eq!(cloned.ref_count(), 2);

        drop(cloned);
        assert_eq!(buffer.ref_count(), 1);
    }

    #[test]
    fn test_zero_copy_ring_buffer() {
        let ring = ZeroCopyRingBuffer::<u32>::new(16).unwrap();
        assert_eq!(ring.capacity(), 16);
        assert_eq!(ring.available_read(), 0);
        assert_eq!(ring.available_write(), 16);

        let data = [1, 2, 3, 4, 5];
        let written = ring.write(&data);
        assert_eq!(written, 5);
        assert_eq!(ring.available_read(), 5);
        assert_eq!(ring.available_write(), 11);

        let mut output = [0u32; 10];
        let read = ring.read(&mut output[..3]);
        assert_eq!(read, 3);
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 3);

        assert_eq!(ring.available_read(), 2);
        assert_eq!(ring.available_write(), 14);
    }

    #[test]
    fn test_zero_copy_ring_buffer_wrap_around() {
        let ring = ZeroCopyRingBuffer::<u8>::new(4).unwrap();

        // Fill the buffer
        let data = [1, 2, 3];
        assert_eq!(ring.write(&data), 3);

        // Read some data
        let mut output = [0u8; 2];
        assert_eq!(ring.read(&mut output), 2);
        assert_eq!(output, [1, 2]);

        // Write more data (should wrap around)
        let more_data = [4, 5, 6];
        assert_eq!(ring.write(&more_data), 3);

        // Read all remaining data
        let mut all_output = [0u8; 10];
        let total_read = ring.read(&mut all_output);
        assert_eq!(total_read, 4);
        assert_eq!(&all_output[..4], &[3, 4, 5, 6]);
    }

    #[cfg(unix)]
    #[test]
    fn test_memory_mapped_file() {
        use std::fs::File;
        use std::io::Write;

        // Create a test file
        let test_path = "/tmp/voirs_test_mmap.dat";
        {
            let mut file = File::create(test_path).unwrap();
            file.write_all(b"Hello, memory mapping!").unwrap();
        }

        let mmap = MemoryMappedFile::open_read_only(test_path).unwrap();
        assert_eq!(mmap.len(), 22);
        assert_eq!(&mmap.as_slice()[..5], b"Hello");

        std::fs::remove_file(test_path).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_shared_memory() {
        let shm_name = "/voirs_test_shm";

        // Clean up any existing shared memory
        unsafe {
            let c_name = std::ffi::CString::new(shm_name).unwrap();
            libc::shm_unlink(c_name.as_ptr());
        }

        let mut shm = SharedMemorySegment::create(shm_name, 1024).unwrap();
        assert_eq!(shm.size(), 1024);

        let slice = shm.as_mut_slice();
        slice[0] = 42;
        slice[1023] = 84;

        // Test opening existing shared memory
        let shm2 = SharedMemorySegment::open(shm_name).unwrap();
        assert_eq!(shm2.as_slice()[0], 42);
        assert_eq!(shm2.as_slice()[1023], 84);

        drop(shm);
        drop(shm2);

        // Clean up
        unsafe {
            let c_name = std::ffi::CString::new(shm_name).unwrap();
            libc::shm_unlink(c_name.as_ptr());
        }
    }
}
