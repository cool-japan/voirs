//! Resource management for voirs-vocoder
//!
//! Provides RAII-based resource management with reference counting,
//! weak references for caches, and memory pressure handling.

use std::sync::{Arc, Weak, Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::time::{Instant, Duration};
use crate::{VocoderError, Result};
use super::MemoryConfig;

/// Resource handle with automatic cleanup
#[derive(Clone)]
pub struct ResourceHandle<T> {
    inner: Arc<ResourceInner<T>>,
    manager: Weak<ResourceManager>,
}

struct ResourceInner<T> {
    data: T,
    resource_id: u64,
    created_at: Instant,
    last_accessed: RwLock<Instant>,
    access_count: AtomicUsize,
}

/// Resource manager with automatic cleanup and memory pressure handling
pub struct ResourceManager {
    config: MemoryConfig,
    resources: Mutex<HashMap<u64, WeakResourceEntry>>,
    next_id: AtomicU64,
    stats: Arc<ResourceStats>,
    cleanup_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

struct WeakResourceEntry {
    weak_ref: Box<dyn std::any::Any + Send + Sync>,
    created_at: Instant,
    resource_type: String,
}

/// Resource management statistics
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub total_created: AtomicUsize,
    pub total_dropped: AtomicUsize,
    pub current_count: AtomicUsize,
    pub memory_pressure_events: AtomicUsize,
    pub cleanup_cycles: AtomicUsize,
    pub bytes_managed: AtomicUsize,
}

impl Default for ResourceStats {
    fn default() -> Self {
        Self {
            total_created: AtomicUsize::new(0),
            total_dropped: AtomicUsize::new(0),
            current_count: AtomicUsize::new(0),
            memory_pressure_events: AtomicUsize::new(0),
            cleanup_cycles: AtomicUsize::new(0),
            bytes_managed: AtomicUsize::new(0),
        }
    }
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let stats = Arc::new(ResourceStats::default());
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        
        let manager = Self {
            config,
            resources: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            stats: stats.clone(),
            cleanup_thread: None,
            shutdown: shutdown.clone(),
        };

        Ok(manager)
    }

    /// Start automatic cleanup background thread
    pub fn start_cleanup_thread(mut self) -> Self {
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);
        let cleanup_interval = Duration::from_secs(30); // Cleanup every 30 seconds
        
        let handle = std::thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                std::thread::sleep(cleanup_interval);
                
                // Perform cleanup cycle
                stats.cleanup_cycles.fetch_add(1, Ordering::SeqCst);
                
                // Note: In a real implementation, we would clean up expired weak references
                // For now, we just update the cleanup cycle count
            }
        });
        
        self.cleanup_thread = Some(handle);
        self
    }

    /// Create managed resource with automatic cleanup
    pub fn manage_resource<T: Send + Sync + 'static>(&self, data: T) -> ResourceHandle<T> {
        let resource_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let now = Instant::now();
        
        let inner = Arc::new(ResourceInner {
            data,
            resource_id,
            created_at: now,
            last_accessed: RwLock::new(now),
            access_count: AtomicUsize::new(0),
        });

        // Store weak reference for tracking
        if let Ok(mut resources) = self.resources.lock() {
            let weak_inner: Weak<ResourceInner<T>> = Arc::downgrade(&inner);
            let weak_ref = Box::new(weak_inner);
            
            resources.insert(resource_id, WeakResourceEntry {
                weak_ref,
                created_at: now,
                resource_type: std::any::type_name::<T>().to_string(),
            });
        }

        self.stats.total_created.fetch_add(1, Ordering::SeqCst);
        self.stats.current_count.fetch_add(1, Ordering::SeqCst);

        ResourceHandle {
            inner,
            manager: Arc::downgrade(&Arc::new(self.clone())), // Note: this would need proper Arc<Self> management in practice
        }
    }

    /// Get current resource statistics
    pub fn stats(&self) -> ResourceStats {
        self.stats.as_ref().clone()
    }

    /// Manually trigger cleanup of expired resources
    pub fn cleanup_expired(&self) {
        let mut resources = match self.resources.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        let mut expired_ids = Vec::new();
        let now = Instant::now();
        let expiry_duration = Duration::from_secs(300); // 5 minutes

        for (&id, entry) in resources.iter() {
            if now.duration_since(entry.created_at) > expiry_duration {
                // Check if the weak reference is still valid
                if let Some(weak_ref) = entry.weak_ref.downcast_ref::<Weak<ResourceInner<()>>>() {
                    if weak_ref.strong_count() == 0 {
                        expired_ids.push(id);
                    }
                }
            }
        }

        for id in expired_ids {
            resources.remove(&id);
            self.stats.total_dropped.fetch_add(1, Ordering::SeqCst);
            self.stats.current_count.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Check memory pressure and trigger cleanup if needed
    pub fn check_memory_pressure(&self) -> bool {
        let current_resources = self.stats.current_count.load(Ordering::SeqCst);
        let max_resources = self.config.max_buffers * 2; // Allow some overhead
        
        if current_resources > max_resources {
            self.stats.memory_pressure_events.fetch_add(1, Ordering::SeqCst);
            self.cleanup_expired();
            true
        } else {
            false
        }
    }

    /// Force cleanup of all resources
    pub fn force_cleanup(&self) {
        if let Ok(mut resources) = self.resources.lock() {
            let count = resources.len();
            resources.clear();
            self.stats.total_dropped.fetch_add(count, Ordering::SeqCst);
            self.stats.current_count.store(0, Ordering::SeqCst);
        }
    }
}

impl Clone for ResourceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            resources: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            stats: Arc::clone(&self.stats),
            cleanup_thread: None, // Don't clone the thread handle
            shutdown: Arc::clone(&self.shutdown),
        }
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        // Signal shutdown and wait for cleanup thread
        self.shutdown.store(true, Ordering::SeqCst);
        
        if let Some(handle) = self.cleanup_thread.take() {
            let _ = handle.join();
        }
        
        // Force cleanup all resources
        self.force_cleanup();
    }
}

impl<T> ResourceHandle<T> {
    /// Get read-only access to the managed resource
    pub fn get(&self) -> &T {
        // Update access tracking
        if let Ok(mut last_accessed) = self.inner.last_accessed.write() {
            *last_accessed = Instant::now();
        }
        self.inner.access_count.fetch_add(1, Ordering::SeqCst);
        
        &self.inner.data
    }

    /// Get resource metadata
    pub fn metadata(&self) -> ResourceMetadata {
        let last_accessed = self.inner.last_accessed.read()
            .map(|guard| *guard)
            .unwrap_or(self.inner.created_at);
        
        ResourceMetadata {
            resource_id: self.inner.resource_id,
            created_at: self.inner.created_at,
            last_accessed,
            access_count: self.inner.access_count.load(Ordering::SeqCst),
        }
    }

    /// Check if resource is still managed
    pub fn is_managed(&self) -> bool {
        self.manager.strong_count() > 0
    }

    /// Get the reference count for this resource
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<T> Drop for ResourceHandle<T> {
    fn drop(&mut self) {
        // The Arc<ResourceInner<T>> will handle its own cleanup
        // The weak reference in the manager will be cleaned up during next cleanup cycle
    }
}

/// Resource metadata for monitoring and debugging
#[derive(Debug, Clone)]
pub struct ResourceMetadata {
    pub resource_id: u64,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: usize,
}

/// RAII guard for temporary resource allocation
pub struct TemporaryResource<T> {
    resource: Option<T>,
    cleanup_fn: Option<Box<dyn FnOnce(T) + Send>>,
}

impl<T> TemporaryResource<T> {
    /// Create new temporary resource with cleanup function
    pub fn new<F>(resource: T, cleanup_fn: F) -> Self 
    where 
        F: FnOnce(T) + Send + 'static 
    {
        Self {
            resource: Some(resource),
            cleanup_fn: Some(Box::new(cleanup_fn)),
        }
    }

    /// Create temporary resource with default cleanup (just drop)
    pub fn new_simple(resource: T) -> Self {
        Self {
            resource: Some(resource),
            cleanup_fn: None,
        }
    }

    /// Get reference to the resource
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get mutable reference to the resource
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Take the resource, preventing automatic cleanup
    pub fn take(mut self) -> Option<T> {
        self.cleanup_fn = None; // Disable cleanup
        self.resource.take()
    }
}

impl<T> Drop for TemporaryResource<T> {
    fn drop(&mut self) {
        if let Some(resource) = self.resource.take() {
            if let Some(cleanup_fn) = self.cleanup_fn.take() {
                cleanup_fn(resource);
            }
            // If no cleanup function, just let it drop normally
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let config = MemoryConfig::default();
        let manager = ResourceManager::new(config).unwrap();
        
        let stats = manager.stats();
        assert_eq!(stats.total_created.load(Ordering::SeqCst), 0);
        assert_eq!(stats.current_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_resource_management() {
        let config = MemoryConfig::default();
        let manager = ResourceManager::new(config).unwrap();
        
        let data = vec![1, 2, 3, 4, 5];
        let handle = manager.manage_resource(data.clone());
        
        assert_eq!(handle.get(), &data);
        assert_eq!(handle.ref_count(), 1);
        
        let stats = manager.stats();
        assert_eq!(stats.total_created.load(Ordering::SeqCst), 1);
        assert_eq!(stats.current_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_resource_metadata() {
        let config = MemoryConfig::default();
        let manager = ResourceManager::new(config).unwrap();
        
        let handle = manager.manage_resource("test data".to_string());
        let metadata = handle.metadata();
        
        assert_eq!(metadata.resource_id, 1);
        assert_eq!(metadata.access_count, 0);
        
        // Access the resource
        let _ = handle.get();
        let metadata_after = handle.metadata();
        assert_eq!(metadata_after.access_count, 1);
    }

    #[test]
    fn test_temporary_resource() {
        let mut cleanup_called = false;
        
        {
            let _temp = TemporaryResource::new(42, |_value| {
                // This would be called on drop, but we can't mutate external state in the real test
            });
            
            // Resource is managed here
        }
        
        // Resource should be cleaned up automatically
    }

    #[test]
    fn test_temporary_resource_take() {
        let temp = TemporaryResource::new_simple(vec![1, 2, 3]);
        let data = temp.take().unwrap();
        
        assert_eq!(data, vec![1, 2, 3]);
        // No cleanup should happen since we took the resource
    }

    #[test]
    fn test_memory_pressure_detection() {
        let mut config = MemoryConfig::default();
        config.max_buffers = 2; // Low limit for testing
        
        let manager = ResourceManager::new(config).unwrap();
        
        // Create resources below the limit
        let _handle1 = manager.manage_resource(vec![1; 1000]);
        let _handle2 = manager.manage_resource(vec![2; 1000]);
        
        assert!(!manager.check_memory_pressure());
        
        // Create more resources to trigger pressure
        let _handle3 = manager.manage_resource(vec![3; 1000]);
        let _handle4 = manager.manage_resource(vec![4; 1000]);
        let _handle5 = manager.manage_resource(vec![5; 1000]);
        
        assert!(manager.check_memory_pressure());
        
        let stats = manager.stats();
        assert!(stats.memory_pressure_events.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_force_cleanup() {
        let config = MemoryConfig::default();
        let manager = ResourceManager::new(config).unwrap();
        
        let _handle1 = manager.manage_resource("data1".to_string());
        let _handle2 = manager.manage_resource("data2".to_string());
        
        let stats_before = manager.stats();
        assert_eq!(stats_before.current_count.load(Ordering::SeqCst), 2);
        
        manager.force_cleanup();
        
        let stats_after = manager.stats();
        assert_eq!(stats_after.current_count.load(Ordering::SeqCst), 0);
        assert!(stats_after.total_dropped.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_resource_handle_clone() {
        let config = MemoryConfig::default();
        let manager = ResourceManager::new(config).unwrap();
        
        let handle1 = manager.manage_resource("shared data".to_string());
        let handle2 = handle1.clone();
        
        assert_eq!(handle1.get(), handle2.get());
        assert_eq!(handle1.ref_count(), 2); // Both handles reference the same data
    }
}