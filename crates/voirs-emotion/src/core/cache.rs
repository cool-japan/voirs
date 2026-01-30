//! Cache and buffer pool implementations for efficient memory management

use std::collections::HashMap;
use std::sync::Mutex;

/// Simple LRU cache implementation for emotion parameters
#[derive(Debug)]
pub(super) struct LruCache<K, V> {
    map: HashMap<K, V>,
    access_order: Vec<K>,
    capacity: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V> LruCache<K, V> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            access_order: Vec::new(),
            capacity,
        }
    }

    pub(super) fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            // Move to front
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                let key_clone = self.access_order.remove(pos);
                self.access_order.push(key_clone);
            }
            self.map.get(key)
        } else {
            None
        }
    }

    pub(super) fn insert(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            // Update existing
            self.map.insert(key.clone(), value);
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                let key_clone = self.access_order.remove(pos);
                self.access_order.push(key_clone);
            }
        } else {
            // Insert new
            if self.map.len() >= self.capacity {
                // Remove LRU
                if let Some(lru_key) = self.access_order.first().cloned() {
                    self.map.remove(&lru_key);
                    self.access_order.remove(0);
                }
            }
            self.map.insert(key.clone(), value);
            self.access_order.push(key);
        }
    }

    pub(super) fn clear(&mut self) {
        self.map.clear();
        self.access_order.clear();
    }

    pub(super) fn len(&self) -> usize {
        self.map.len()
    }
}

/// Buffer pool for reusing audio processing buffers to reduce allocations
#[derive(Debug)]
pub(super) struct BufferPool {
    float_buffers: Mutex<Vec<Vec<f32>>>,
    max_pool_size: usize,
}

impl BufferPool {
    pub(super) fn new(max_pool_size: usize) -> Self {
        Self {
            float_buffers: Mutex::new(Vec::new()),
            max_pool_size,
        }
    }

    pub(super) fn get_buffer(&self, min_size: usize) -> Vec<f32> {
        // Handle potential mutex poisoning gracefully
        match self.float_buffers.lock() {
            Ok(mut pool) => {
                if let Some(mut buffer) = pool.pop() {
                    if buffer.len() >= min_size {
                        buffer.clear();
                        buffer.resize(min_size, 0.0);
                        return buffer;
                    }
                }
            }
            Err(_) => {
                // Mutex is poisoned, but we can still provide a buffer
                tracing::warn!("Buffer pool mutex poisoned, creating new buffer");
            }
        }
        vec![0.0; min_size]
    }

    pub(super) fn return_buffer(&self, buffer: Vec<f32>) {
        // Handle potential mutex poisoning gracefully
        match self.float_buffers.lock() {
            Ok(mut pool) => {
                if pool.len() < self.max_pool_size && buffer.len() <= 8192 {
                    pool.push(buffer);
                }
            }
            Err(_) => {
                // Mutex is poisoned, drop the buffer
                tracing::warn!("Buffer pool mutex poisoned, dropping buffer");
            }
        }
    }
}
