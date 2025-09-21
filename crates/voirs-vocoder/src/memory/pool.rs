use std::sync::{Arc, Mutex, RwLock};
use std::collections::{VecDeque, HashMap};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::VocoderError;
use super::{MemoryConfig, Result};

#[cfg(feature = "candle")]
use candle_core::{Tensor, Device};

pub struct MemoryPool {
    config: MemoryConfig,
    buffers: Mutex<VecDeque<PooledBuffer>>,
    stats: Arc<PoolStats>,
    total_allocated: AtomicUsize,
    #[cfg(feature = "candle")]
    tensor_pools: Mutex<HashMap<TensorKey, VecDeque<Tensor>>>,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TensorKey {
    shape: Vec<usize>,
    dtype: candle_core::DType,
    device: String, // Simplified device representation
}

impl MemoryPool {
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let pool = Self {
            config,
            buffers: Mutex::new(VecDeque::new()),
            stats: Arc::new(PoolStats::default()),
            total_allocated: AtomicUsize::new(0),
            #[cfg(feature = "candle")]
            tensor_pools: Mutex::new(HashMap::new()),
        };

        // Pre-allocate some buffers
        pool.preallocate_buffers()?;
        
        Ok(pool)
    }

    fn preallocate_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.lock().map_err(|_| VocoderError::Other("Failed to lock buffers".to_string()))?;
        
        // Create a variety of buffer sizes
        let sizes = vec![1024, 2048, 4096, 8192, 16384, 32768];
        
        for &size in &sizes {
            for _ in 0..4 {
                let buffer = PooledBuffer::new(size, self.config.alignment)?;
                buffers.push_back(buffer);
            }
        }

        self.stats.preallocated_count.store(buffers.len(), Ordering::SeqCst);
        Ok(())
    }

    pub fn allocate_audio_buffer(&self, size: usize) -> Result<AllocatedBuffer> {
        self.stats.allocation_requests.fetch_add(1, Ordering::SeqCst);

        // Try to find a suitable buffer from the pool
        if let Some(buffer) = self.try_get_from_pool(size)? {
            self.stats.pool_hits.fetch_add(1, Ordering::SeqCst);
            return Ok(AllocatedBuffer::new(buffer, Arc::clone(&self.stats)));
        }

        // Allocate a new buffer
        self.stats.pool_misses.fetch_add(1, Ordering::SeqCst);
        let buffer = PooledBuffer::new(size, self.config.alignment)?;
        self.total_allocated.fetch_add(size, Ordering::SeqCst);
        
        Ok(AllocatedBuffer::new(buffer, Arc::clone(&self.stats)))
    }

    fn try_get_from_pool(&self, size: usize) -> Result<Option<PooledBuffer>> {
        let mut buffers = self.buffers.lock().map_err(|_| VocoderError::Other("Failed to lock buffers".to_string()))?;
        
        // Find a buffer that's large enough
        let mut index = None;
        for (i, buffer) in buffers.iter().enumerate() {
            if buffer.capacity >= size {
                index = Some(i);
                break;
            }
        }

        if let Some(i) = index {
            Ok(Some(buffers.remove(i).unwrap()))
        } else {
            Ok(None)
        }
    }

    pub fn return_buffer(&self, buffer: PooledBuffer) {
        if let Ok(mut buffers) = self.buffers.lock() {
            if buffers.len() < self.config.max_buffers {
                buffers.push_back(buffer);
                self.stats.buffers_returned.fetch_add(1, Ordering::SeqCst);
            } else {
                // Pool is full, let the buffer be dropped
                self.stats.buffers_dropped.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    pub fn clear(&self) {
        if let Ok(mut buffers) = self.buffers.lock() {
            buffers.clear();
            self.total_allocated.store(0, Ordering::SeqCst);
        }
        #[cfg(feature = "candle")]
        if let Ok(mut tensor_pools) = self.tensor_pools.lock() {
            tensor_pools.clear();
        }
    }

    #[cfg(feature = "candle")]
    pub fn get_tensor(&self, shape: &[usize], dtype: candle_core::DType, device: &Device) -> Option<Tensor> {
        let key = TensorKey {
            shape: shape.to_vec(),
            dtype,
            device: format!("{:?}", device), // Simplified device representation
        };
        
        if let Ok(mut pools) = self.tensor_pools.lock() {
            if let Some(pool) = pools.get_mut(&key) {
                if let Some(tensor) = pool.pop_front() {
                    self.stats.pool_hits.fetch_add(1, Ordering::SeqCst);
                    return Some(tensor);
                }
            }
        }
        
        self.stats.pool_misses.fetch_add(1, Ordering::SeqCst);
        None
    }

    #[cfg(feature = "candle")]
    pub fn return_tensor(&self, tensor: Tensor) {
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();
        let device_str = format!("{:?}", tensor.device());
        
        let key = TensorKey {
            shape,
            dtype,
            device: device_str,
        };
        
        if let Ok(mut pools) = self.tensor_pools.lock() {
            let pool = pools.entry(key).or_insert_with(VecDeque::new);
            
            // Limit pool size to prevent excessive memory usage
            if pool.len() < self.config.max_buffers / 4 {
                pool.push_back(tensor);
                self.stats.buffers_returned.fetch_add(1, Ordering::SeqCst);
            } else {
                // Pool is full, let tensor be dropped
                self.stats.buffers_dropped.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[cfg(feature = "candle")]
    pub fn create_or_reuse_tensor(&self, shape: &[usize], dtype: candle_core::DType, device: &Device) -> Result<Tensor> {
        // Try to get from pool first
        if let Some(tensor) = self.get_tensor(shape, dtype, device) {
            return Ok(tensor);
        }
        
        // Create new tensor if not available in pool
        let tensor = Tensor::zeros(shape, dtype, device)
            .map_err(|e| VocoderError::ModelError(format!("Failed to create tensor: {e}")))?;
        
        Ok(tensor)
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        self.clear();
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub allocation_requests: AtomicUsize,
    pub pool_hits: AtomicUsize,
    pub pool_misses: AtomicUsize,
    pub buffers_returned: AtomicUsize,
    pub buffers_dropped: AtomicUsize,
    pub preallocated_count: AtomicUsize,
    pub total_capacity: AtomicUsize,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            allocation_requests: AtomicUsize::new(0),
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
            buffers_returned: AtomicUsize::new(0),
            buffers_dropped: AtomicUsize::new(0),
            preallocated_count: AtomicUsize::new(0),
            total_capacity: AtomicUsize::new(1024 * 1024 * 64),
        }
    }
}

pub struct PooledBuffer {
    ptr: NonNull<u8>,
    capacity: usize,
    layout: Layout,
}

impl PooledBuffer {
    fn new(size: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| VocoderError::Other("Invalid layout".to_string()))?;
        
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(VocoderError::Other("Failed to allocate memory".to_string()));
        }

        let ptr = NonNull::new(ptr)
            .ok_or_else(|| VocoderError::Other("Null pointer after allocation".to_string()))?;

        Ok(Self {
            ptr,
            capacity: size,
            layout,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

pub struct AllocatedBuffer {
    buffer: Option<PooledBuffer>,
    stats: Arc<PoolStats>,
    pool: Option<Arc<MemoryPool>>,
}

impl AllocatedBuffer {
    fn new(buffer: PooledBuffer, stats: Arc<PoolStats>) -> Self {
        Self {
            buffer: Some(buffer),
            stats,
            pool: None,
        }
    }

    pub fn size(&self) -> usize {
        self.buffer.as_ref().map(|b| b.capacity).unwrap_or(0)
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.buffer.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null_mut())
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.buffer.as_mut().map(|b| b.as_slice_mut())
    }

    pub fn as_slice(&self) -> Option<&[u8]> {
        self.buffer.as_ref().map(|b| b.as_slice())
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.as_slice_mut().map(|slice| {
            let len = slice.len() / std::mem::size_of::<f32>();
            unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, len) }
        })
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        self.as_slice().map(|slice| {
            let len = slice.len() / std::mem::size_of::<f32>();
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) }
        })
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Some(pool) = &self.pool {
                pool.return_buffer(buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let stats = pool.stats();
        assert!(stats.preallocated_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let buffer = pool.allocate_audio_buffer(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_buffer_reuse() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        // Allocate and drop a buffer
        {
            let _buffer = pool.allocate_audio_buffer(1024).unwrap();
        }
        
        // Allocate again - should reuse from pool
        let buffer = pool.allocate_audio_buffer(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_pooled_buffer_operations() {
        let mut buffer = PooledBuffer::new(1024, 64).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        
        let slice = buffer.as_slice_mut();
        slice[0] = 42;
        assert_eq!(buffer.as_slice()[0], 42);
    }

    #[test]
    fn test_allocated_buffer_f32_operations() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let mut buffer = pool.allocate_audio_buffer(1024).unwrap();
        
        if let Some(slice) = buffer.as_f32_slice_mut() {
            slice[0] = 3.14;
        }
        
        if let Some(slice) = buffer.as_f32_slice() {
            assert_eq!(slice[0], 3.14);
        }
    }

    #[test]
    fn test_pool_stats() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let _buffer1 = pool.allocate_audio_buffer(1024).unwrap();
        let _buffer2 = pool.allocate_audio_buffer(2048).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.allocation_requests.load(Ordering::SeqCst), 2);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_tensor_pooling() {
        use candle_core::{Device, DType};
        
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        let device = Device::Cpu;
        
        // Create and return a tensor
        let tensor1 = pool.create_or_reuse_tensor(&[2, 3], DType::F32, &device).unwrap();
        assert_eq!(tensor1.shape().dims(), &[2, 3]);
        
        pool.return_tensor(tensor1);
        
        // Try to get it back from pool
        let tensor2 = pool.get_tensor(&[2, 3], DType::F32, &device);
        assert!(tensor2.is_some());
        
        let stats = pool.stats();
        assert!(stats.pool_hits.load(Ordering::SeqCst) > 0);
    }

    #[cfg(feature = "candle")]
    #[test] 
    fn test_tensor_pool_miss() {
        use candle_core::{Device, DType};
        
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        let device = Device::Cpu;
        
        // Try to get a tensor that doesn't exist in pool
        let tensor = pool.get_tensor(&[5, 7], DType::F64, &device);
        assert!(tensor.is_none());
        
        let stats = pool.stats();
        assert!(stats.pool_misses.load(Ordering::SeqCst) > 0);
    }
}