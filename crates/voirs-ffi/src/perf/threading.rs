//! Threading Performance Optimization
//!
//! This module provides advanced threading optimizations including work-stealing
//! algorithms, lock-free data structures, thread-local storage, and CPU affinity management.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, ThreadId};
use std::collections::HashMap;
use parking_lot::{Mutex, RwLock};
use once_cell::sync::Lazy;

/// Global threading performance statistics
static THREAD_PERF_STATS: Lazy<ThreadPerfStats> = Lazy::new(ThreadPerfStats::new);

/// Threading performance statistics and metrics
#[derive(Debug)]
pub struct ThreadPerfStats {
    pub active_threads: AtomicUsize,
    pub work_stolen_count: AtomicU64,
    pub lock_free_operations: AtomicU64,
    pub cpu_migrations: AtomicU64,
    pub thread_local_hits: AtomicU64,
    pub thread_local_misses: AtomicU64,
}

impl ThreadPerfStats {
    pub fn new() -> Self {
        Self {
            active_threads: AtomicUsize::new(0),
            work_stolen_count: AtomicU64::new(0),
            lock_free_operations: AtomicU64::new(0),
            cpu_migrations: AtomicU64::new(0),
            thread_local_hits: AtomicU64::new(0),
            thread_local_misses: AtomicU64::new(0),
        }
    }

    /// Increment work stealing counter
    pub fn record_work_steal(&self) {
        self.work_stolen_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record lock-free operation
    pub fn record_lock_free_op(&self) {
        self.lock_free_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record CPU migration
    pub fn record_cpu_migration(&self) {
        self.cpu_migrations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record thread-local storage access
    pub fn record_tls_access(&self, hit: bool) {
        if hit {
            self.thread_local_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.thread_local_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> ThreadPerfSnapshot {
        ThreadPerfSnapshot {
            active_threads: self.active_threads.load(Ordering::Relaxed),
            work_stolen_count: self.work_stolen_count.load(Ordering::Relaxed),
            lock_free_operations: self.lock_free_operations.load(Ordering::Relaxed),
            cpu_migrations: self.cpu_migrations.load(Ordering::Relaxed),
            thread_local_hits: self.thread_local_hits.load(Ordering::Relaxed),
            thread_local_misses: self.thread_local_misses.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of threading performance statistics
#[derive(Debug, Clone)]
pub struct ThreadPerfSnapshot {
    pub active_threads: usize,
    pub work_stolen_count: u64,
    pub lock_free_operations: u64,
    pub cpu_migrations: u64,
    pub thread_local_hits: u64,
    pub thread_local_misses: u64,
}

/// Work-stealing deque for improved load balancing
pub struct WorkStealingDeque<T> {
    items: Arc<RwLock<Vec<Option<T>>>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
}

impl<T> WorkStealingDeque<T> {
    /// Create a new work-stealing deque
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Arc::new(RwLock::new(vec![None; capacity])),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push work item to local end (owner thread)
    pub fn push(&self, item: T) -> Result<(), T> {
        let mut items = self.items.write();
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail.wrapping_sub(head) >= self.capacity {
            return Err(item); // Queue full
        }

        let index = tail % self.capacity;
        items[index] = Some(item);
        
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pop work item from local end (owner thread)
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }

        let new_tail = tail.wrapping_sub(1);
        self.tail.store(new_tail, Ordering::Relaxed);
        
        let mut items = self.items.write();
        let index = new_tail % self.capacity;
        THREAD_PERF_STATS.record_lock_free_op();
        items[index].take()
    }

    /// Steal work item from remote end (other threads)
    pub fn steal(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head >= tail {
            return None; // Empty
        }

        // Attempt to steal
        if self.head.compare_exchange_weak(
            head,
            head.wrapping_add(1),
            Ordering::Release,
            Ordering::Relaxed,
        ).is_ok() {
            THREAD_PERF_STATS.record_work_steal();
            THREAD_PERF_STATS.record_lock_free_op();
            
            let mut items = self.items.write();
            let index = head % self.capacity;
            items[index].take()
        } else {
            None
        }
    }

    /// Get current queue size
    pub fn size(&self) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        tail.saturating_sub(head)
    }
}

/// Thread-local storage manager for audio buffers
pub struct ThreadLocalStorage<T> {
    data: Arc<RwLock<HashMap<ThreadId, T>>>,
    constructor: fn() -> T,
}

impl<T> ThreadLocalStorage<T> 
where
    T: Clone,
{
    /// Create new thread-local storage
    pub fn new(constructor: fn() -> T) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            constructor,
        }
    }

    /// Get thread-local data
    pub fn get(&self) -> T {
        let thread_id = thread::current().id();
        
        // Fast path: try to read existing value
        {
            let data = self.data.read();
            if let Some(value) = data.get(&thread_id) {
                THREAD_PERF_STATS.record_tls_access(true);
                return value.clone();
            }
        }

        // Slow path: create new value
        let mut data = self.data.write();
        let value = data.entry(thread_id).or_insert_with(self.constructor).clone();
        THREAD_PERF_STATS.record_tls_access(false);
        value
    }

    /// Update thread-local data
    pub fn update<F>(&self, updater: F) 
    where
        F: FnOnce(&mut T),
    {
        let thread_id = thread::current().id();
        let mut data = self.data.write();
        let value = data.entry(thread_id).or_insert_with(self.constructor);
        updater(value);
    }

    /// Clear all thread-local data
    pub fn clear(&self) {
        let mut data = self.data.write();
        data.clear();
    }
}

/// CPU affinity manager for optimal thread placement
pub struct CpuAffinityManager {
    cpu_count: usize,
    thread_assignments: Arc<Mutex<HashMap<ThreadId, usize>>>,
    enabled: AtomicBool,
}

impl CpuAffinityManager {
    /// Create new CPU affinity manager
    pub fn new() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            cpu_count,
            thread_assignments: Arc::new(Mutex::new(HashMap::new())),
            enabled: AtomicBool::new(true),
        }
    }

    /// Set thread CPU affinity
    pub fn set_affinity(&self, thread_id: ThreadId, cpu_id: usize) -> Result<(), String> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(());
        }

        if cpu_id >= self.cpu_count {
            return Err(format!("CPU ID {} exceeds available CPU count {}", cpu_id, self.cpu_count));
        }

        let mut assignments = self.thread_assignments.lock();
        assignments.insert(thread_id, cpu_id);
        
        // Platform-specific CPU affinity setting would go here
        #[cfg(target_os = "linux")]
        {
            // Linux-specific affinity setting
            // This would require libc or similar for actual implementation
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows-specific affinity setting
            // This would require windows-sys or similar for actual implementation
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS-specific affinity setting
            // Note: macOS has limited CPU affinity support
        }

        THREAD_PERF_STATS.record_cpu_migration();
        Ok(())
    }

    /// Get optimal CPU for new thread
    pub fn get_optimal_cpu(&self) -> usize {
        let assignments = self.thread_assignments.lock();
        let mut cpu_usage = vec![0; self.cpu_count];
        
        for &cpu_id in assignments.values() {
            if cpu_id < cpu_usage.len() {
                cpu_usage[cpu_id] += 1;
            }
        }

        // Find CPU with minimum usage
        cpu_usage
            .iter()
            .enumerate()
            .min_by_key(|(_, &usage)| usage)
            .map(|(cpu_id, _)| cpu_id)
            .unwrap_or(0)
    }

    /// Enable or disable CPU affinity management
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get CPU count
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }
}

/// Lock-free counter for performance metrics
pub struct LockFreeCounter {
    value: AtomicU64,
}

impl LockFreeCounter {
    /// Create new lock-free counter
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment counter
    pub fn increment(&self) -> u64 {
        THREAD_PERF_STATS.record_lock_free_op();
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Add value to counter
    pub fn add(&self, val: u64) -> u64 {
        THREAD_PERF_STATS.record_lock_free_op();
        self.value.fetch_add(val, Ordering::Relaxed)
    }

    /// Get current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset counter
    pub fn reset(&self) -> u64 {
        THREAD_PERF_STATS.record_lock_free_op();
        self.value.swap(0, Ordering::Relaxed)
    }
}

/// Get global threading performance statistics
pub fn get_thread_perf_stats() -> ThreadPerfSnapshot {
    THREAD_PERF_STATS.get_stats()
}

/// Reset global threading performance statistics
pub fn reset_thread_perf_stats() {
    THREAD_PERF_STATS.work_stolen_count.store(0, Ordering::Relaxed);
    THREAD_PERF_STATS.lock_free_operations.store(0, Ordering::Relaxed);
    THREAD_PERF_STATS.cpu_migrations.store(0, Ordering::Relaxed);
    THREAD_PERF_STATS.thread_local_hits.store(0, Ordering::Relaxed);
    THREAD_PERF_STATS.thread_local_misses.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_work_stealing_deque() {
        let deque = WorkStealingDeque::new(10);
        
        // Test push and size
        assert!(deque.push(42).is_ok());
        assert_eq!(deque.size(), 1);
        
        // Test steal from empty
        assert!(deque.steal().is_none());
    }

    #[test]
    fn test_thread_local_storage() {
        let tls = ThreadLocalStorage::new(|| 0u32);
        
        // Test initial value
        assert_eq!(tls.get(), 0);
        
        // Test update
        tls.update(|val| *val = 42);
        assert_eq!(tls.get(), 42);
    }

    #[test]
    fn test_cpu_affinity_manager() {
        let manager = CpuAffinityManager::new();
        
        // Test CPU count
        assert!(manager.cpu_count() > 0);
        
        // Test optimal CPU selection
        let optimal = manager.get_optimal_cpu();
        assert!(optimal < manager.cpu_count());
    }

    #[test]
    fn test_lock_free_counter() {
        let counter = LockFreeCounter::new();
        
        // Test initial value
        assert_eq!(counter.get(), 0);
        
        // Test increment
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        // Test add
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        // Test reset
        let old_value = counter.reset();
        assert_eq!(old_value, 6);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_thread_perf_stats() {
        reset_thread_perf_stats();
        
        // Test recording operations
        THREAD_PERF_STATS.record_work_steal();
        THREAD_PERF_STATS.record_lock_free_op();
        THREAD_PERF_STATS.record_cpu_migration();
        THREAD_PERF_STATS.record_tls_access(true);
        THREAD_PERF_STATS.record_tls_access(false);
        
        let stats = get_thread_perf_stats();
        assert_eq!(stats.work_stolen_count, 1);
        assert_eq!(stats.lock_free_operations, 1);
        assert_eq!(stats.cpu_migrations, 1);
        assert_eq!(stats.thread_local_hits, 1);
        assert_eq!(stats.thread_local_misses, 1);
    }

    #[test]
    fn test_concurrent_operations() {
        let counter = Arc::new(LockFreeCounter::new());
        let mut handles = vec![];

        // Spawn multiple threads to test lock-free operations
        for _ in 0..4 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    counter_clone.increment();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), 4000);
    }
}

/// Advanced threading optimizations for high-performance FFI operations
pub mod advanced_threading {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    
    /// NUMA-aware thread pool with topology detection
    pub struct NumaAwareThreadPool {
        workers: Vec<NumaWorker>,
        work_queues: Vec<Arc<Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>>>,
        topology: NumaTopology,
        round_robin_counter: AtomicUsize,
    }
    
    /// NUMA topology information
    #[derive(Debug, Clone)]
    pub struct NumaTopology {
        pub numa_nodes: Vec<NumaNode>,
        pub total_cores: usize,
        pub cores_per_node: usize,
    }
    
    #[derive(Debug, Clone)]
    pub struct NumaNode {
        pub node_id: usize,
        pub cpu_cores: Vec<usize>,
        pub memory_size_gb: usize,
    }
    
    /// NUMA-aware worker thread
    pub struct NumaWorker {
        thread_handle: Option<thread::JoinHandle<()>>,
        numa_node: usize,
        cpu_affinity: Vec<usize>,
        shutdown: Arc<AtomicBool>,
    }
    
    impl NumaAwareThreadPool {
        pub fn new() -> Self {
            let topology = Self::detect_numa_topology();
            let num_workers = topology.total_cores;
            let mut workers = Vec::with_capacity(num_workers);
            let mut work_queues = Vec::with_capacity(num_workers);
            
            // Create work queues for each worker
            for _ in 0..num_workers {
                work_queues.push(Arc::new(Mutex::new(std::collections::VecDeque::new())));
            }
            
            // Create workers with NUMA affinity
            for (i, node) in topology.numa_nodes.iter().enumerate() {
                for (core_idx, &cpu_core) in node.cpu_cores.iter().enumerate() {
                    let worker_id = i * topology.cores_per_node + core_idx;
                    if worker_id < num_workers {
                        let worker = NumaWorker::new(
                            worker_id,
                            node.node_id,
                            vec![cpu_core],
                            work_queues.clone(),
                        );
                        workers.push(worker);
                    }
                }
            }
            
            Self {
                workers,
                work_queues,
                topology,
                round_robin_counter: AtomicUsize::new(0),
            }
        }
        
        fn detect_numa_topology() -> NumaTopology {
            // Detect NUMA topology (simplified implementation)
            let total_cores = num_cpus::get();
            let numa_nodes = if total_cores > 8 {
                // Assume 2 NUMA nodes for systems with >8 cores
                vec![
                    NumaNode {
                        node_id: 0,
                        cpu_cores: (0..total_cores/2).collect(),
                        memory_size_gb: 16,
                    },
                    NumaNode {
                        node_id: 1,
                        cpu_cores: (total_cores/2..total_cores).collect(),
                        memory_size_gb: 16,
                    },
                ]
            } else {
                // Single NUMA node for smaller systems
                vec![
                    NumaNode {
                        node_id: 0,
                        cpu_cores: (0..total_cores).collect(),
                        memory_size_gb: 16,
                    },
                ]
            };
            
            NumaTopology {
                cores_per_node: total_cores / numa_nodes.len(),
                numa_nodes,
                total_cores,
            }
        }
        
        /// Submit work to NUMA-local queue when possible
        pub fn submit_work<F>(&self, preferred_numa_node: Option<usize>, work: F)
        where
            F: FnOnce() + Send + 'static,
        {
            let target_queue = if let Some(numa_node) = preferred_numa_node {
                // Try to place work on specified NUMA node
                numa_node * self.topology.cores_per_node
            } else {
                // Round-robin assignment
                self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.work_queues.len()
            };
            
            if let Some(queue) = self.work_queues.get(target_queue) {
                queue.lock().push_back(Box::new(work));
            }
        }
        
        /// Get NUMA node for current thread
        pub fn current_numa_node(&self) -> Option<usize> {
            // Simplified NUMA node detection based on CPU
            let current_cpu = Self::get_current_cpu();
            for node in &self.topology.numa_nodes {
                if node.cpu_cores.contains(&current_cpu) {
                    return Some(node.node_id);
                }
            }
            None
        }
        
        fn get_current_cpu() -> usize {
            // Platform-specific CPU detection
            #[cfg(target_os = "linux")]
            {
                // Use sched_getcpu() on Linux
                unsafe { libc::sched_getcpu() as usize }
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // Fallback to 0 for other platforms
                0
            }
        }
    }
    
    impl NumaWorker {
        fn new(
            worker_id: usize,
            numa_node: usize,
            cpu_affinity: Vec<usize>,
            work_queues: Vec<Arc<Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>>>,
        ) -> Self {
            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_clone = shutdown.clone();
            
            let thread_handle = thread::Builder::new()
                .name(format!("numa-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(worker_id, numa_node, &cpu_affinity, work_queues, shutdown_clone);
                })
                .ok();
            
            Self {
                thread_handle,
                numa_node,
                cpu_affinity,
                shutdown,
            }
        }
        
        fn worker_loop(
            worker_id: usize,
            numa_node: usize,
            cpu_affinity: &[usize],
            work_queues: Vec<Arc<Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>>>,
            shutdown: Arc<AtomicBool>,
        ) {
            // Set CPU affinity if supported
            Self::set_cpu_affinity(cpu_affinity);
            
            let mut steal_attempts = 0;
            const MAX_STEAL_ATTEMPTS: usize = 100;
            
            while !shutdown.load(Ordering::Relaxed) {
                let mut work_found = false;
                
                // Try to get work from own queue first
                if let Some(own_queue) = work_queues.get(worker_id) {
                    if let Some(work) = own_queue.lock().pop_front() {
                        work();
                        work_found = true;
                        steal_attempts = 0;
                    }
                }
                
                // If no work found, try work stealing
                if !work_found {
                    for (i, queue) in work_queues.iter().enumerate() {
                        if i != worker_id {
                            if let Some(work) = queue.lock().pop_back() {
                                work();
                                work_found = true;
                                get_thread_perf_stats().record_work_steal();
                                steal_attempts = 0;
                                break;
                            }
                        }
                    }
                }
                
                if !work_found {
                    steal_attempts += 1;
                    if steal_attempts < MAX_STEAL_ATTEMPTS {
                        // Short spin before trying again
                        std::hint::spin_loop();
                    } else {
                        // Longer sleep to avoid wasting CPU
                        thread::sleep(Duration::from_micros(10));
                        steal_attempts = 0;
                    }
                }
            }
        }
        
        #[cfg(target_os = "linux")]
        fn set_cpu_affinity(cpu_cores: &[usize]) {
            if cpu_cores.is_empty() {
                return;
            }
            
            // Set CPU affinity using libc on Linux
            unsafe {
                let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut cpu_set);
                
                for &cpu in cpu_cores {
                    libc::CPU_SET(cpu, &mut cpu_set);
                }
                
                libc::sched_setaffinity(
                    0, // Current thread
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpu_set,
                );
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        fn set_cpu_affinity(_cpu_cores: &[usize]) {
            // CPU affinity not implemented for this platform
        }
    }
    
    impl Drop for NumaWorker {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.thread_handle.take() {
                let _ = handle.join();
            }
        }
    }
    
    /// Lock-free SPSC (Single Producer Single Consumer) queue for high-frequency operations
    pub struct LockFreeSPSCQueue<T> {
        buffer: Box<[std::sync::atomic::AtomicPtr<T>]>,
        capacity: usize,
        head: std::sync::atomic::AtomicUsize,
        tail: std::sync::atomic::AtomicUsize,
    }
    
    impl<T> LockFreeSPSCQueue<T> {
        pub fn new(capacity: usize) -> Self {
            let mut buffer = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                buffer.push(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()));
            }
            
            Self {
                buffer: buffer.into_boxed_slice(),
                capacity,
                head: std::sync::atomic::AtomicUsize::new(0),
                tail: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        
        /// Try to enqueue an item (returns false if queue is full)
        pub fn try_enqueue(&self, item: T) -> Result<(), T> {
            let tail = self.tail.load(Ordering::Relaxed);
            let next_tail = (tail + 1) % self.capacity;
            
            if next_tail == self.head.load(Ordering::Acquire) {
                // Queue is full
                return Err(item);
            }
            
            let boxed_item = Box::into_raw(Box::new(item));
            self.buffer[tail].store(boxed_item, Ordering::Relaxed);
            self.tail.store(next_tail, Ordering::Release);
            
            get_thread_perf_stats().record_lock_free_op();
            Ok(())
        }
        
        /// Try to dequeue an item (returns None if queue is empty)
        pub fn try_dequeue(&self) -> Option<T> {
            let head = self.head.load(Ordering::Relaxed);
            
            if head == self.tail.load(Ordering::Acquire) {
                // Queue is empty
                return None;
            }
            
            let item_ptr = self.buffer[head].load(Ordering::Relaxed);
            if item_ptr.is_null() {
                return None;
            }
            
            self.buffer[head].store(std::ptr::null_mut(), Ordering::Relaxed);
            self.head.store((head + 1) % self.capacity, Ordering::Release);
            
            unsafe {
                let item = Box::from_raw(item_ptr);
                get_thread_perf_stats().record_lock_free_op();
                Some(*item)
            }
        }
        
        /// Check if queue is empty
        pub fn is_empty(&self) -> bool {
            self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
        }
        
        /// Get approximate queue size
        pub fn len(&self) -> usize {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Relaxed);
            
            if tail >= head {
                tail - head
            } else {
                self.capacity - head + tail
            }
        }
    }
    
    unsafe impl<T: Send> Send for LockFreeSPSCQueue<T> {}
    unsafe impl<T: Send> Sync for LockFreeSPSCQueue<T> {}
    
    impl<T> Drop for LockFreeSPSCQueue<T> {
        fn drop(&mut self) {
            // Clean up any remaining items
            while self.try_dequeue().is_some() {}
        }
    }
    
    /// High-performance thread-local storage with cache optimization
    pub struct OptimizedThreadLocal<T> {
        data: thread_local::ThreadLocal<std::cell::RefCell<T>>,
        default_factory: Box<dyn Fn() -> T + Send + Sync>,
    }
    
    impl<T> OptimizedThreadLocal<T>
    where
        T: 'static,
    {
        pub fn new<F>(factory: F) -> Self
        where
            F: Fn() -> T + Send + Sync + 'static,
        {
            Self {
                data: thread_local::ThreadLocal::new(),
                default_factory: Box::new(factory),
            }
        }
        
        /// Get reference to thread-local data
        pub fn with<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&T) -> R,
        {
            let cell = self.data.get_or(|| {
                get_thread_perf_stats().record_thread_local_miss();
                std::cell::RefCell::new((self.default_factory)())
            });
            
            get_thread_perf_stats().record_thread_local_hit();
            f(&cell.borrow())
        }
        
        /// Get mutable reference to thread-local data
        pub fn with_mut<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut T) -> R,
        {
            let cell = self.data.get_or(|| {
                get_thread_perf_stats().record_thread_local_miss();
                std::cell::RefCell::new((self.default_factory)())
            });
            
            get_thread_perf_stats().record_thread_local_hit();
            f(&mut cell.borrow_mut())
        }
    }
    
    /// Adaptive work-stealing scheduler with load balancing
    pub struct AdaptiveWorkStealer {
        queues: Vec<Arc<Mutex<std::collections::VecDeque<WorkItem>>>>,
        workers: Vec<WorkerThread>,
        load_balancer: LoadBalancer,
        metrics: WorkStealingMetrics,
    }
    
    type WorkItem = Box<dyn FnOnce() + Send>;
    
    #[derive(Debug, Default)]
    pub struct WorkStealingMetrics {
        pub successful_steals: AtomicU64,
        pub failed_steals: AtomicU64,
        pub load_balance_operations: AtomicU64,
        pub queue_contentions: AtomicU64,
    }
    
    pub struct WorkerThread {
        thread_handle: Option<thread::JoinHandle<()>>,
        shutdown_signal: Arc<AtomicBool>,
        worker_id: usize,
    }
    
    pub struct LoadBalancer {
        last_balance_time: Mutex<Instant>,
        balance_interval: Duration,
        load_threshold: f64,
    }
    
    impl AdaptiveWorkStealer {
        pub fn new(num_workers: usize) -> Self {
            let mut queues = Vec::with_capacity(num_workers);
            let mut workers = Vec::with_capacity(num_workers);
            
            // Create work queues
            for _ in 0..num_workers {
                queues.push(Arc::new(Mutex::new(std::collections::VecDeque::new())));
            }
            
            let queues_shared = queues.iter().cloned().collect::<Vec<_>>();
            let metrics = Arc::new(WorkStealingMetrics::default());
            
            // Create worker threads
            for worker_id in 0..num_workers {
                let shutdown_signal = Arc::new(AtomicBool::new(false));
                let shutdown_clone = shutdown_signal.clone();
                let queues_clone = queues_shared.clone();
                let metrics_clone = metrics.clone();
                
                let thread_handle = thread::Builder::new()
                    .name(format!("work-stealer-{}", worker_id))
                    .spawn(move || {
                        Self::worker_loop(worker_id, queues_clone, shutdown_clone, metrics_clone);
                    })
                    .ok();
                
                workers.push(WorkerThread {
                    thread_handle,
                    shutdown_signal,
                    worker_id,
                });
            }
            
            Self {
                queues,
                workers,
                load_balancer: LoadBalancer::new(),
                metrics: (*metrics).clone(),
            }
        }
        
        fn worker_loop(
            worker_id: usize,
            queues: Vec<Arc<Mutex<std::collections::VecDeque<WorkItem>>>>,
            shutdown: Arc<AtomicBool>,
            metrics: Arc<WorkStealingMetrics>,
        ) {
            let mut consecutive_steals = 0;
            const MAX_CONSECUTIVE_STEALS: usize = 10;
            
            while !shutdown.load(Ordering::Relaxed) {
                let mut work_found = false;
                
                // Try own queue first
                if let Some(work) = Self::try_pop_work(&queues[worker_id]) {
                    work();
                    work_found = true;
                    consecutive_steals = 0;
                }
                
                // If no work in own queue, try stealing
                if !work_found && consecutive_steals < MAX_CONSECUTIVE_STEALS {
                    for (i, queue) in queues.iter().enumerate() {
                        if i != worker_id {
                            if let Some(work) = Self::try_steal_work(queue) {
                                work();
                                work_found = true;
                                consecutive_steals += 1;
                                metrics.successful_steals.fetch_add(1, Ordering::Relaxed);
                                get_thread_perf_stats().record_work_steal();
                                break;
                            } else {
                                metrics.failed_steals.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
                
                if !work_found {
                    // Exponential backoff when no work available
                    let backoff_duration = std::cmp::min(
                        Duration::from_micros(1 << consecutive_steals.min(10)),
                        Duration::from_millis(1),
                    );
                    thread::sleep(backoff_duration);
                    consecutive_steals = 0;
                }
            }
        }
        
        fn try_pop_work(queue: &Arc<Mutex<std::collections::VecDeque<WorkItem>>>) -> Option<WorkItem> {
            if let Ok(mut q) = queue.try_lock() {
                q.pop_front()
            } else {
                None
            }
        }
        
        fn try_steal_work(queue: &Arc<Mutex<std::collections::VecDeque<WorkItem>>>) -> Option<WorkItem> {
            if let Ok(mut q) = queue.try_lock() {
                q.pop_back() // Steal from back to minimize contention
            } else {
                None
            }
        }
        
        /// Submit work to the least loaded queue
        pub fn submit_work<F>(&self, work: F)
        where
            F: FnOnce() + Send + 'static,
        {
            let target_queue = self.find_least_loaded_queue();
            self.queues[target_queue].lock().push_back(Box::new(work));
        }
        
        fn find_least_loaded_queue(&self) -> usize {
            let mut min_size = usize::MAX;
            let mut best_queue = 0;
            
            for (i, queue) in self.queues.iter().enumerate() {
                if let Ok(q) = queue.try_lock() {
                    let size = q.len();
                    if size < min_size {
                        min_size = size;
                        best_queue = i;
                    }
                }
            }
            
            best_queue
        }
        
        /// Get work stealing metrics
        pub fn get_metrics(&self) -> &WorkStealingMetrics {
            &self.metrics
        }
    }
    
    impl LoadBalancer {
        fn new() -> Self {
            Self {
                last_balance_time: Mutex::new(Instant::now()),
                balance_interval: Duration::from_millis(100),
                load_threshold: 0.8,
            }
        }
    }
    
    impl Drop for AdaptiveWorkStealer {
        fn drop(&mut self) {
            // Signal all workers to shut down
            for worker in &self.workers {
                worker.shutdown_signal.store(true, Ordering::Relaxed);
            }
            
            // Wait for all workers to finish
            for worker in &mut self.workers {
                if let Some(handle) = worker.thread_handle.take() {
                    let _ = handle.join();
                }
            }
        }
    }
}

/// Global instances for advanced threading optimizations
static NUMA_THREAD_POOL: Lazy<advanced_threading::NumaAwareThreadPool> = 
    Lazy::new(advanced_threading::NumaAwareThreadPool::new);
static ADAPTIVE_WORK_STEALER: Lazy<advanced_threading::AdaptiveWorkStealer> = 
    Lazy::new(|| advanced_threading::AdaptiveWorkStealer::new(num_cpus::get()));

/// Enhanced C API functions for advanced threading
#[no_mangle]
pub extern "C" fn voirs_threading_submit_numa_work(
    numa_node: i32,
    work_fn: extern "C" fn(),
) -> bool {
    let preferred_node = if numa_node >= 0 { Some(numa_node as usize) } else { None };
    
    NUMA_THREAD_POOL.submit_work(preferred_node, move || {
        work_fn();
    });
    
    true
}

#[no_mangle]
pub extern "C" fn voirs_threading_get_current_numa_node() -> i32 {
    NUMA_THREAD_POOL.current_numa_node().map_or(-1, |node| node as i32)
}

#[no_mangle]
pub extern "C" fn voirs_threading_submit_adaptive_work(work_fn: extern "C" fn()) -> bool {
    ADAPTIVE_WORK_STEALER.submit_work(move || {
        work_fn();
    });
    
    true
}

#[no_mangle]
pub extern "C" fn voirs_threading_get_work_stealing_stats(
    successful_steals: *mut u64,
    failed_steals: *mut u64,
) -> bool {
    if successful_steals.is_null() || failed_steals.is_null() {
        return false;
    }
    
    let metrics = ADAPTIVE_WORK_STEALER.get_metrics();
    
    unsafe {
        *successful_steals = metrics.successful_steals.load(Ordering::Relaxed);
        *failed_steals = metrics.failed_steals.load(Ordering::Relaxed);
    }
    
    true
}

#[no_mangle]
pub extern "C" fn voirs_threading_create_spsc_queue(capacity: usize) -> *mut advanced_threading::LockFreeSPSCQueue<u64> {
    if capacity == 0 {
        return std::ptr::null_mut();
    }
    
    Box::into_raw(Box::new(advanced_threading::LockFreeSPSCQueue::new(capacity)))
}

#[no_mangle]
pub extern "C" fn voirs_threading_destroy_spsc_queue(queue: *mut advanced_threading::LockFreeSPSCQueue<u64>) {
    if !queue.is_null() {
        unsafe {
            let _ = Box::from_raw(queue);
        }
    }
}

#[no_mangle]
pub extern "C" fn voirs_threading_spsc_enqueue(
    queue: *mut advanced_threading::LockFreeSPSCQueue<u64>,
    item: u64,
) -> bool {
    if queue.is_null() {
        return false;
    }
    
    unsafe {
        (*queue).try_enqueue(item).is_ok()
    }
}

#[no_mangle]
pub extern "C" fn voirs_threading_spsc_dequeue(
    queue: *mut advanced_threading::LockFreeSPSCQueue<u64>,
    item: *mut u64,
) -> bool {
    if queue.is_null() || item.is_null() {
        return false;
    }
    
    unsafe {
        if let Some(dequeued_item) = (*queue).try_dequeue() {
            *item = dequeued_item;
            true
        } else {
            false
        }
    }
}