//! Advanced real-time scheduling for neural vocoder streaming
//!
//! This module provides enhanced real-time scheduling capabilities including:
//! - Priority-based task scheduling
//! - Deadline-aware processing
//! - Load balancing across cores
//! - Interrupt-style processing
//! - NUMA-aware task distribution

use crate::Result;
use std::{
    collections::{BinaryHeap, HashMap},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex as AsyncMutex, Notify};

/// NUMA topology information for task scheduling
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// NUMA nodes in the system
    pub nodes: Vec<NumaNode>,
    /// Core to NUMA node mapping
    pub core_to_node: HashMap<usize, u32>,
}

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub node_id: u32,
    /// CPU cores associated with this node
    pub cpu_cores: Vec<usize>,
    /// Current load on this node
    pub load: f64,
    /// Memory usage on this node (0.0 to 1.0)
    pub memory_usage: f64,
}

/// Priority levels for real-time scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RtPriority {
    /// Background processing (lowest priority)
    Background = 0,
    /// Normal streaming priority
    #[default]
    Normal = 1,
    /// High priority for low-latency streams
    High = 2,
    /// Critical real-time processing
    Critical = 3,
    /// Emergency interrupt-level processing
    Interrupt = 4,
}

/// Real-time task definition
#[derive(Debug, Clone)]
pub struct RtTask {
    /// Unique task identifier
    pub id: u64,
    /// Stream ID this task belongs to
    pub stream_id: u64,
    /// Task priority
    pub priority: RtPriority,
    /// Deadline for completion
    pub deadline: Instant,
    /// Estimated processing time
    pub estimated_duration: Duration,
    /// Task payload size
    pub chunk_size: usize,
    /// CPU affinity preference
    pub preferred_core: Option<usize>,
}

/// Enhanced real-time scheduler with priority and deadline awareness
pub struct EnhancedRtScheduler {
    /// Task queue organized by priority
    task_queue: Arc<AsyncMutex<BinaryHeap<PriorityTask>>>,
    /// Running tasks by ID
    running_tasks: Arc<RwLock<HashMap<u64, TaskExecution>>>,
    /// Scheduler configuration
    config: Arc<RwLock<SchedulerConfig>>,
    /// Performance statistics
    stats: Arc<RwLock<SchedulerStats>>,
    /// NUMA topology information
    numa_topology: Arc<RwLock<Option<NumaTopology>>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Task completion notifier
    notify: Arc<Notify>,
    /// Task ID counter
    next_task_id: Arc<AtomicU64>,
}

/// Task with priority for heap ordering
#[derive(Debug, Clone)]
struct PriorityTask {
    task: RtTask,
    /// Combined priority score (higher = more urgent)
    priority_score: u64,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority scores come first (max heap)
        self.priority_score.cmp(&other.priority_score)
    }
}

/// Task execution context
#[derive(Debug, Clone)]
struct TaskExecution {
    task: RtTask,
    start_time: Instant,
    #[allow(dead_code)]
    assigned_core: Option<usize>,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Enable CPU affinity management
    pub enable_cpu_affinity: bool,
    /// Enable NUMA awareness
    pub enable_numa_awareness: bool,
    /// Deadline miss tolerance (ms)
    pub deadline_tolerance_ms: u64,
    /// Priority boost for aging tasks
    pub aging_boost_factor: f64,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get(),
            enable_cpu_affinity: true,
            enable_numa_awareness: false,
            deadline_tolerance_ms: 10,
            aging_boost_factor: 1.1,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin assignment
    RoundRobin,
    /// Assign to least loaded core
    LeastLoaded,
    /// NUMA-aware assignment
    NumaAware,
    /// Priority-based assignment
    PriorityBased,
}

/// Scheduler performance statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks scheduled
    pub total_scheduled: u64,
    /// Total tasks completed
    pub total_completed: u64,
    /// Tasks that missed deadlines
    pub deadline_misses: u64,
    /// Average task latency (ms)
    pub avg_latency_ms: f64,
    /// Current load per core
    pub core_loads: Vec<f64>,
    /// Priority inversions detected
    pub priority_inversions: u64,
}

impl EnhancedRtScheduler {
    /// Create new enhanced real-time scheduler
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Create scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        let num_cores = num_cpus::get();
        let numa_topology = if config.enable_numa_awareness {
            Some(Self::detect_numa_topology(num_cores))
        } else {
            None
        };

        Self {
            task_queue: Arc::new(AsyncMutex::new(BinaryHeap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(RwLock::new(SchedulerStats {
                core_loads: vec![0.0; num_cores],
                ..Default::default()
            })),
            numa_topology: Arc::new(RwLock::new(numa_topology)),
            shutdown: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
            next_task_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Schedule a real-time task
    pub async fn schedule_task(
        &self,
        stream_id: u64,
        priority: RtPriority,
        deadline: Instant,
        estimated_duration: Duration,
        chunk_size: usize,
    ) -> Result<u64> {
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);

        let task = RtTask {
            id: task_id,
            stream_id,
            priority,
            deadline,
            estimated_duration,
            chunk_size,
            preferred_core: self.select_optimal_core(priority, chunk_size).await,
        };

        let priority_score = self.calculate_priority_score(&task).await;
        let priority_task = PriorityTask {
            task,
            priority_score,
        };

        {
            let mut queue = self.task_queue.lock().await;
            queue.push(priority_task);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_scheduled += 1;
        }

        // Notify scheduler of new task
        self.notify.notify_one();

        Ok(task_id)
    }

    /// Get next task to execute (priority + deadline aware)
    pub async fn get_next_task(&self) -> Option<RtTask> {
        let mut queue = self.task_queue.lock().await;

        // Check for urgent tasks that need immediate attention
        if let Some(priority_task) = queue.peek() {
            let task = &priority_task.task;
            let time_to_deadline = task.deadline.saturating_duration_since(Instant::now());

            // If deadline is very close or passed, execute immediately
            if time_to_deadline <= Duration::from_millis(5) {
                return queue.pop().map(|pt| pt.task);
            }

            // Check if we have capacity for this task
            let running_count = self.running_tasks.read().unwrap().len();
            let max_concurrent = self.config.read().unwrap().max_concurrent_tasks;

            if running_count < max_concurrent {
                return queue.pop().map(|pt| pt.task);
            }
        }

        None
    }

    /// Mark task as started
    pub async fn start_task_execution(
        &self,
        task: RtTask,
        assigned_core: Option<usize>,
    ) -> Result<()> {
        let execution = TaskExecution {
            task: task.clone(),
            start_time: Instant::now(),
            assigned_core,
        };

        self.running_tasks
            .write()
            .unwrap()
            .insert(task.id, execution);
        Ok(())
    }

    /// Mark task as completed
    pub async fn complete_task(&self, task_id: u64, _success: bool) -> Result<()> {
        let execution = self.running_tasks.write().unwrap().remove(&task_id);

        if let Some(exec) = execution {
            let completion_time = Instant::now();
            let actual_duration = completion_time.duration_since(exec.start_time);
            let deadline_met = completion_time <= exec.task.deadline;

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.total_completed += 1;

                if !deadline_met {
                    stats.deadline_misses += 1;
                }

                // Update average latency
                let latency_ms = actual_duration.as_secs_f64() * 1000.0;
                if stats.total_completed == 1 {
                    stats.avg_latency_ms = latency_ms;
                } else {
                    stats.avg_latency_ms =
                        (stats.avg_latency_ms * (stats.total_completed - 1) as f64 + latency_ms)
                            / stats.total_completed as f64;
                }
            }
        }

        Ok(())
    }

    /// Calculate priority score for task scheduling
    async fn calculate_priority_score(&self, task: &RtTask) -> u64 {
        let base_priority = (task.priority as u64) * 1000;
        let time_to_deadline = task.deadline.saturating_duration_since(Instant::now());

        // Urgency factor (higher for closer deadlines)
        let urgency_factor = if time_to_deadline.is_zero() {
            1000 // Overdue tasks get maximum urgency
        } else {
            let deadline_ms = time_to_deadline.as_millis() as u64;
            1000 / (1 + deadline_ms / 10) // Closer deadlines get higher scores
        };

        // Size factor (smaller chunks get slight priority boost for responsiveness)
        let size_factor = if task.chunk_size < 1024 {
            10
        } else if task.chunk_size < 4096 {
            5
        } else {
            0
        };

        base_priority + urgency_factor + size_factor
    }

    /// Select optimal CPU core for task assignment
    async fn select_optimal_core(&self, priority: RtPriority, _chunk_size: usize) -> Option<usize> {
        let config = self.config.read().unwrap();

        if !config.enable_cpu_affinity {
            return None;
        }

        let stats = self.stats.read().unwrap();
        let strategy = config.load_balancing;

        match strategy {
            LoadBalancingStrategy::RoundRobin => {
                Some((stats.total_scheduled as usize) % stats.core_loads.len())
            }
            LoadBalancingStrategy::LeastLoaded => stats
                .core_loads
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx),
            LoadBalancingStrategy::PriorityBased => {
                // High priority tasks get dedicated cores (if available)
                match priority {
                    RtPriority::Critical | RtPriority::Interrupt => {
                        // Try to assign to a less loaded core
                        stats
                            .core_loads
                            .iter()
                            .enumerate()
                            .filter(|(_, load)| **load < 0.5)
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                    }
                    _ => {
                        // Normal priority uses least loaded strategy
                        stats
                            .core_loads
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                    }
                }
            }
            LoadBalancingStrategy::NumaAware => {
                // Update NUMA loads first
                self.update_numa_loads();

                // Find optimal NUMA node
                if let Some(best_numa_node) = self.find_optimal_numa_node() {
                    // Get cores for the best NUMA node
                    if let Some(numa_topology) = self.numa_topology.read().unwrap().as_ref() {
                        if let Some(node) = numa_topology
                            .nodes
                            .iter()
                            .find(|n| n.node_id == best_numa_node)
                        {
                            // Find least loaded core within the optimal NUMA node
                            return node
                                .cpu_cores
                                .iter()
                                .min_by(|&&core_a, &&core_b| {
                                    let load_a = stats.core_loads.get(core_a).unwrap_or(&0.0);
                                    let load_b = stats.core_loads.get(core_b).unwrap_or(&0.0);
                                    load_a
                                        .partial_cmp(load_b)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .copied();
                        }
                    }
                }

                // Fallback to least loaded if NUMA detection failed
                stats
                    .core_loads
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
            }
        }
    }

    /// Detect NUMA topology for the system
    fn detect_numa_topology(num_cores: usize) -> NumaTopology {
        // For systems without NUMA or when detection fails, create a single node
        // In a real implementation, this would use platform-specific APIs
        // like libnuma on Linux or GetNumaNodeProcessorMask on Windows

        let numa_node_count = std::env::var("VOIRS_NUMA_NODES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let cores_per_node = (num_cores + numa_node_count - 1) / numa_node_count;
        let mut nodes = Vec::new();
        let mut core_to_node = HashMap::new();

        for node_id in 0..numa_node_count {
            let start_core = node_id * cores_per_node;
            let end_core = std::cmp::min(start_core + cores_per_node, num_cores);
            let cpu_cores: Vec<usize> = (start_core..end_core).collect();

            // Map cores to this node
            for &core in &cpu_cores {
                core_to_node.insert(core, node_id as u32);
            }

            nodes.push(NumaNode {
                node_id: node_id as u32,
                cpu_cores,
                load: 0.0,
                memory_usage: 0.0,
            });
        }

        NumaTopology {
            nodes,
            core_to_node,
        }
    }

    /// Update NUMA node loads based on current core loads
    fn update_numa_loads(&self) {
        if let Some(ref mut numa_topology) = self.numa_topology.write().unwrap().as_mut() {
            let stats = self.stats.read().unwrap();

            // Update load for each NUMA node
            for node in &mut numa_topology.nodes {
                let total_load: f64 = node
                    .cpu_cores
                    .iter()
                    .map(|&core_idx| stats.core_loads.get(core_idx).unwrap_or(&0.0))
                    .sum();

                node.load = total_load / node.cpu_cores.len() as f64;
            }
        }
    }

    /// Find best NUMA node for task placement
    fn find_optimal_numa_node(&self) -> Option<u32> {
        let numa_topology_guard = self.numa_topology.read().unwrap();
        let numa_topology = numa_topology_guard.as_ref()?;

        // Find the NUMA node with the lowest load
        numa_topology
            .nodes
            .iter()
            .min_by(|a, b| {
                a.load
                    .partial_cmp(&b.load)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.node_id)
    }

    /// Enable or disable NUMA awareness at runtime
    pub fn set_numa_awareness(&self, enable: bool) {
        let mut config = self.config.write().unwrap();
        config.enable_numa_awareness = enable;

        if enable {
            let num_cores = num_cpus::get();
            let topology = Self::detect_numa_topology(num_cores);
            *self.numa_topology.write().unwrap() = Some(topology);
        } else {
            *self.numa_topology.write().unwrap() = None;
        }
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.read().unwrap().clone()
    }

    /// Check for deadline violations and missed tasks
    pub async fn check_deadline_violations(&self) -> Vec<u64> {
        let mut violations = Vec::new();
        let now = Instant::now();

        for (task_id, execution) in self.running_tasks.read().unwrap().iter() {
            if now > execution.task.deadline {
                violations.push(*task_id);
            }
        }

        violations
    }

    /// Shutdown the scheduler
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    /// Wait for task completion notification
    pub async fn wait_for_task(&self) {
        if !self.shutdown.load(Ordering::Relaxed) {
            self.notify.notified().await;
        }
    }
}

impl Default for EnhancedRtScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_enhanced_scheduler_creation() {
        let scheduler = EnhancedRtScheduler::new();
        let stats = scheduler.get_stats();
        assert_eq!(stats.total_scheduled, 0);
        assert_eq!(stats.total_completed, 0);
    }

    #[tokio::test]
    async fn test_task_scheduling() {
        let scheduler = EnhancedRtScheduler::new();
        let deadline = Instant::now() + Duration::from_millis(100);

        let task_id = scheduler
            .schedule_task(
                1,
                RtPriority::High,
                deadline,
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        assert!(task_id > 0);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_scheduled, 1);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let scheduler = EnhancedRtScheduler::new();
        let now = Instant::now();

        // Schedule low priority task
        let _low_id = scheduler
            .schedule_task(
                1,
                RtPriority::Normal,
                now + Duration::from_millis(100),
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        // Schedule high priority task
        let _high_id = scheduler
            .schedule_task(
                2,
                RtPriority::Critical,
                now + Duration::from_millis(100),
                Duration::from_millis(10),
                512,
            )
            .await
            .unwrap();

        // High priority task should come first
        if let Some(task) = scheduler.get_next_task().await {
            assert_eq!(task.priority, RtPriority::Critical);
            assert_eq!(task.stream_id, 2);
        }
    }

    #[tokio::test]
    async fn test_deadline_urgency() {
        let scheduler = EnhancedRtScheduler::new();
        let now = Instant::now();

        // Schedule task with distant deadline
        let _distant_id = scheduler
            .schedule_task(
                1,
                RtPriority::Normal,
                now + Duration::from_millis(1000),
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        // Schedule task with urgent deadline
        let _urgent_id = scheduler
            .schedule_task(
                2,
                RtPriority::Normal,
                now + Duration::from_millis(5),
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        // Urgent task should come first despite same priority
        if let Some(task) = scheduler.get_next_task().await {
            assert_eq!(task.stream_id, 2);
        }
    }

    #[tokio::test]
    async fn test_task_completion() {
        let scheduler = EnhancedRtScheduler::new();
        let deadline = Instant::now() + Duration::from_millis(100);

        let task_id = scheduler
            .schedule_task(
                1,
                RtPriority::Normal,
                deadline,
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        if let Some(task) = scheduler.get_next_task().await {
            scheduler.start_task_execution(task, Some(0)).await.unwrap();

            // Simulate task processing
            sleep(Duration::from_millis(5)).await;

            scheduler.complete_task(task_id, true).await.unwrap();
        }

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_completed, 1);
        assert!(stats.avg_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_deadline_violations() {
        let scheduler = EnhancedRtScheduler::new();
        let past_deadline = Instant::now() - Duration::from_millis(10);

        let task_id = scheduler
            .schedule_task(
                1,
                RtPriority::Normal,
                past_deadline,
                Duration::from_millis(10),
                1024,
            )
            .await
            .unwrap();

        if let Some(task) = scheduler.get_next_task().await {
            scheduler.start_task_execution(task, Some(0)).await.unwrap();

            let violations = scheduler.check_deadline_violations().await;
            assert!(!violations.is_empty());
            assert_eq!(violations[0], task_id);
        }
    }

    #[tokio::test]
    async fn test_load_balancing_config() {
        let config = SchedulerConfig {
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            enable_cpu_affinity: true,
            ..Default::default()
        };

        let scheduler = EnhancedRtScheduler::with_config(config);

        // Test that core selection works
        let core = scheduler.select_optimal_core(RtPriority::High, 1024).await;
        assert!(core.is_some());
    }
}
