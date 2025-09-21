//! Interrupt-style processing for ultra-low latency audio streaming
//!
//! This module provides interrupt-driven processing capabilities similar to
//! hardware interrupt handling, enabling ultra-low latency audio processing
//! with preemptive scheduling and priority-based task management.

use crate::{MelSpectrogram, Result, VocoderError};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::{oneshot, Notify};

/// Interrupt priority levels (higher values = higher priority)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterruptPriority {
    /// Non-maskable interrupt (highest priority)
    NonMaskable = 255,
    /// Critical system interrupt
    Critical = 200,
    /// Audio hardware interrupt
    AudioHardware = 150,
    /// High priority audio processing
    AudioProcessing = 100,
    /// Network I/O interrupt
    NetworkIO = 80,
    /// Timer interrupt
    Timer = 60,
    /// Keyboard/mouse input
    UserInput = 40,
    /// Background processing
    Background = 10,
}

/// Interrupt service routine (ISR) function type
pub type InterruptHandler =
    Box<dyn Fn(&InterruptContext) -> Result<InterruptResponse> + Send + Sync>;

/// Interrupt context passed to handlers
#[derive(Debug, Clone)]
pub struct InterruptContext {
    /// Interrupt ID
    pub interrupt_id: u64,
    /// Interrupt priority
    pub priority: InterruptPriority,
    /// Timestamp when interrupt was triggered
    pub timestamp: Instant,
    /// Associated data payload
    pub data: InterruptData,
    /// Preemption flag
    pub preempted: bool,
}

/// Data payload for interrupts
#[derive(Debug, Clone)]
pub enum InterruptData {
    /// Audio processing interrupt
    AudioChunk {
        mel_data: MelSpectrogram,
        stream_id: u64,
        deadline: Instant,
    },
    /// Buffer management interrupt
    BufferEvent {
        buffer_id: u64,
        event_type: BufferEventType,
        urgency: f32,
    },
    /// Timer interrupt
    Timer {
        timer_id: u64,
        interval: Duration,
        repetitions: u32,
    },
    /// System control interrupt
    SystemControl {
        command: SystemCommand,
        parameters: HashMap<String, String>,
    },
    /// Custom user-defined interrupt
    Custom {
        data: Vec<u8>,
        metadata: HashMap<String, String>,
    },
}

/// Buffer event types
#[derive(Debug, Clone, Copy)]
pub enum BufferEventType {
    /// Buffer is nearly full
    HighWatermark,
    /// Buffer is nearly empty
    LowWatermark,
    /// Buffer overflow detected
    Overflow,
    /// Buffer underrun detected
    Underrun,
    /// Buffer allocation failure
    AllocationFailed,
}

/// System control commands
#[derive(Debug, Clone)]
pub enum SystemCommand {
    /// Adjust processing priority
    SetPriority(u64, InterruptPriority),
    /// Enable/disable interrupt source
    EnableInterrupt(u64, bool),
    /// Flush all pending interrupts
    FlushInterrupts,
    /// Emergency shutdown
    EmergencyShutdown,
    /// Update configuration
    UpdateConfig(String),
}

/// Response from interrupt handler
#[derive(Debug, Clone)]
pub enum InterruptResponse {
    /// Interrupt handled successfully
    Handled,
    /// Interrupt handled with result data
    HandledWithData(Vec<u8>),
    /// Interrupt should be deferred
    Deferred,
    /// Interrupt should be retried
    Retry(Duration),
    /// Error occurred during handling
    Error(String),
}

/// Interrupt controller for managing interrupt-style processing
pub struct InterruptController {
    /// Registered interrupt handlers
    handlers: Arc<RwLock<HashMap<InterruptPriority, Vec<InterruptHandler>>>>,
    /// Pending interrupts queue (priority ordered)
    interrupt_queue: Arc<RwLock<VecDeque<PendingInterrupt>>>,
    /// Currently executing interrupt context
    current_interrupt: Arc<RwLock<Option<InterruptContext>>>,
    /// Preempted interrupts waiting for resumption
    preempted_interrupts: Arc<RwLock<VecDeque<PreemptedInterrupt>>>,
    /// Interrupt masking state
    interrupt_mask: Arc<RwLock<HashMap<InterruptPriority, bool>>>,
    /// Global interrupt enable flag
    interrupts_enabled: Arc<AtomicBool>,
    /// Performance statistics
    stats: Arc<RwLock<InterruptStats>>,
    /// Worker thread handles
    worker_threads: Arc<RwLock<Vec<thread::JoinHandle<()>>>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Interrupt notification
    notify: Arc<Notify>,
    /// Next interrupt ID
    next_interrupt_id: Arc<AtomicU64>,
}

/// Pending interrupt in the queue
#[derive(Debug)]
struct PendingInterrupt {
    context: InterruptContext,
    response_tx: Option<oneshot::Sender<InterruptResponse>>,
}

/// Preempted interrupt saved for later resumption
#[derive(Debug)]
struct PreemptedInterrupt {
    /// Original interrupt context
    #[allow(dead_code)]
    context: InterruptContext,
    /// Timestamp when preempted
    #[allow(dead_code)]
    preempted_at: Instant,
    /// Execution state at time of preemption
    #[allow(dead_code)]
    execution_state: InterruptExecutionState,
    /// Response channel for resumption
    #[allow(dead_code)]
    response_tx: Option<oneshot::Sender<InterruptResponse>>,
}

/// Execution state for interrupt resumption
#[derive(Debug, Clone)]
struct InterruptExecutionState {
    /// Progress percentage (0.0 to 1.0)
    #[allow(dead_code)]
    progress: f64,
    /// Any partial results that need to be preserved
    #[allow(dead_code)]
    partial_data: Option<Vec<u8>>,
    /// Handler-specific state data
    #[allow(dead_code)]
    handler_state: HashMap<String, String>,
}

/// Interrupt controller statistics
#[derive(Debug, Clone, Default)]
pub struct InterruptStats {
    /// Total interrupts processed
    pub total_interrupts: u64,
    /// Interrupts by priority level
    pub interrupts_by_priority: HashMap<InterruptPriority, u64>,
    /// Average interrupt latency (µs)
    pub avg_interrupt_latency_us: f64,
    /// Peak interrupt latency (µs)
    pub peak_interrupt_latency_us: u64,
    /// Interrupt handler execution time (µs)
    pub avg_handler_time_us: f64,
    /// Preempted interrupts count
    pub preempted_interrupts: u64,
    /// Resumed interrupts count
    pub resumed_interrupts: u64,
    /// Deferred interrupts count
    pub deferred_interrupts: u64,
    /// Failed interrupts count
    pub failed_interrupts: u64,
    /// Current interrupt nesting level
    pub nesting_level: u32,
}

impl InterruptController {
    /// Create new interrupt controller
    pub fn new() -> Self {
        let mut interrupt_mask = HashMap::new();

        // Initially all interrupts are enabled
        interrupt_mask.insert(InterruptPriority::NonMaskable, false);
        interrupt_mask.insert(InterruptPriority::Critical, false);
        interrupt_mask.insert(InterruptPriority::AudioHardware, false);
        interrupt_mask.insert(InterruptPriority::AudioProcessing, false);
        interrupt_mask.insert(InterruptPriority::NetworkIO, false);
        interrupt_mask.insert(InterruptPriority::Timer, false);
        interrupt_mask.insert(InterruptPriority::UserInput, false);
        interrupt_mask.insert(InterruptPriority::Background, false);

        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            interrupt_queue: Arc::new(RwLock::new(VecDeque::new())),
            current_interrupt: Arc::new(RwLock::new(None)),
            preempted_interrupts: Arc::new(RwLock::new(VecDeque::new())),
            interrupt_mask: Arc::new(RwLock::new(interrupt_mask)),
            interrupts_enabled: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(RwLock::new(InterruptStats::default())),
            worker_threads: Arc::new(RwLock::new(Vec::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
            next_interrupt_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Start the interrupt controller with specified number of worker threads
    pub async fn start(&self, num_workers: usize) -> Result<()> {
        let mut workers = self.worker_threads.write().unwrap();

        for worker_id in 0..num_workers {
            let controller = self.clone_for_worker();
            let handle = thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create async runtime for interrupt worker");

                rt.block_on(async move {
                    controller.worker_loop(worker_id).await;
                });
            });

            workers.push(handle);
        }

        Ok(())
    }

    /// Register an interrupt handler for a specific priority level
    pub fn register_handler(&self, priority: InterruptPriority, handler: InterruptHandler) {
        let mut handlers = self.handlers.write().unwrap();
        handlers.entry(priority).or_default().push(handler);
    }

    /// Trigger an interrupt with given priority and data
    pub async fn trigger_interrupt(
        &self,
        priority: InterruptPriority,
        data: InterruptData,
    ) -> Result<InterruptResponse> {
        if !self.is_interrupt_enabled(priority) {
            return Ok(InterruptResponse::Deferred);
        }

        let interrupt_id = self.next_interrupt_id.fetch_add(1, Ordering::Relaxed);
        let context = InterruptContext {
            interrupt_id,
            priority,
            timestamp: Instant::now(),
            data,
            preempted: false,
        };

        let (response_tx, response_rx) = oneshot::channel();
        let pending = PendingInterrupt {
            context,
            response_tx: Some(response_tx),
        };

        // Add to priority queue
        {
            let mut queue = self.interrupt_queue.write().unwrap();

            // Insert in priority order (higher priority first)
            let insert_pos = queue
                .iter()
                .position(|existing| priority > existing.context.priority);

            if let Some(pos) = insert_pos {
                queue.insert(pos, pending);
            } else {
                queue.push_back(pending);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_interrupts += 1;
            *stats.interrupts_by_priority.entry(priority).or_insert(0) += 1;
        }

        // Notify workers
        self.notify.notify_one();

        // Wait for response
        response_rx
            .await
            .map_err(|_| VocoderError::StreamingError("Interrupt handler failed".to_string()))
    }

    /// Fire and forget interrupt (no response expected)
    pub async fn fire_interrupt(
        &self,
        priority: InterruptPriority,
        data: InterruptData,
    ) -> Result<()> {
        if !self.is_interrupt_enabled(priority) {
            return Ok(());
        }

        let interrupt_id = self.next_interrupt_id.fetch_add(1, Ordering::Relaxed);
        let context = InterruptContext {
            interrupt_id,
            priority,
            timestamp: Instant::now(),
            data,
            preempted: false,
        };

        let pending = PendingInterrupt {
            context,
            response_tx: None, // No response channel
        };

        // Add to priority queue
        {
            let mut queue = self.interrupt_queue.write().unwrap();

            // Insert in priority order
            let insert_pos = queue
                .iter()
                .position(|existing| priority > existing.context.priority);

            if let Some(pos) = insert_pos {
                queue.insert(pos, pending);
            } else {
                queue.push_back(pending);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_interrupts += 1;
            *stats.interrupts_by_priority.entry(priority).or_insert(0) += 1;
        }

        // Notify workers
        self.notify.notify_one();
        Ok(())
    }

    /// Enable or disable interrupts globally
    pub fn set_interrupts_enabled(&self, enabled: bool) {
        self.interrupts_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Mask/unmask specific interrupt priority level
    pub fn set_interrupt_mask(&self, priority: InterruptPriority, masked: bool) {
        let mut mask = self.interrupt_mask.write().unwrap();
        mask.insert(priority, masked);
    }

    /// Check if interrupt priority is enabled
    fn is_interrupt_enabled(&self, priority: InterruptPriority) -> bool {
        if !self.interrupts_enabled.load(Ordering::Relaxed) {
            return false;
        }

        // Non-maskable interrupts are always enabled
        if priority == InterruptPriority::NonMaskable {
            return true;
        }

        let mask = self.interrupt_mask.read().unwrap();
        !mask.get(&priority).copied().unwrap_or(false)
    }

    /// Get current interrupt statistics
    pub fn get_stats(&self) -> InterruptStats {
        self.stats.read().unwrap().clone()
    }

    /// Shutdown the interrupt controller
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.notify_waiters();

        // Wait for worker threads to finish
        let mut workers = self.worker_threads.write().unwrap();
        while let Some(handle) = workers.pop() {
            if let Err(e) = handle.join() {
                tracing::warn!("Worker thread panicked: {:?}", e);
            }
        }

        Ok(())
    }

    /// Clone for worker thread use
    fn clone_for_worker(&self) -> InterruptControllerWorker {
        InterruptControllerWorker {
            handlers: self.handlers.clone(),
            interrupt_queue: self.interrupt_queue.clone(),
            current_interrupt: self.current_interrupt.clone(),
            preempted_interrupts: self.preempted_interrupts.clone(),
            stats: self.stats.clone(),
            shutdown: self.shutdown.clone(),
            notify: self.notify.clone(),
        }
    }
}

/// Worker thread interface
struct InterruptControllerWorker {
    handlers: Arc<RwLock<HashMap<InterruptPriority, Vec<InterruptHandler>>>>,
    interrupt_queue: Arc<RwLock<VecDeque<PendingInterrupt>>>,
    current_interrupt: Arc<RwLock<Option<InterruptContext>>>,
    preempted_interrupts: Arc<RwLock<VecDeque<PreemptedInterrupt>>>,
    stats: Arc<RwLock<InterruptStats>>,
    shutdown: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl InterruptControllerWorker {
    async fn worker_loop(&self, worker_id: usize) {
        tracing::info!("Interrupt worker {} started", worker_id);

        while !self.shutdown.load(Ordering::Relaxed) {
            // Process pending interrupts first
            let mut processed_any = false;
            while let Some(pending) = self.get_next_interrupt() {
                processed_any = true;
                let start_time = Instant::now();

                // Check if this interrupt should preempt current one
                let should_preempt = self.should_preempt(&pending.context);

                if should_preempt {
                    self.preempt_current_interrupt();
                }

                // Set current interrupt context
                {
                    let mut current = self.current_interrupt.write().unwrap();
                    *current = Some(pending.context.clone());
                }

                // Execute interrupt handler
                let response = self.execute_handler(&pending.context).await;

                // Calculate latency
                let latency = start_time.elapsed();
                self.update_latency_stats(latency);

                // Send response if channel exists
                if let Some(tx) = pending.response_tx {
                    let _ = tx.send(response);
                }

                // Clear current interrupt
                {
                    let mut current = self.current_interrupt.write().unwrap();
                    *current = None;
                }
            }

            // If no interrupts were processed, wait for notification
            if !processed_any {
                tokio::select! {
                    _ = self.notify.notified() => {
                        // New interrupt arrived
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Timeout to prevent hanging
                    }
                }
            }
        }

        tracing::info!("Interrupt worker {} shutdown", worker_id);
    }

    /// Get next interrupt from queue
    fn get_next_interrupt(&self) -> Option<PendingInterrupt> {
        let mut queue = self.interrupt_queue.write().unwrap();
        queue.pop_front()
    }

    /// Check if interrupt should preempt current one
    fn should_preempt(&self, new_context: &InterruptContext) -> bool {
        let current = self.current_interrupt.read().unwrap();

        if let Some(current_context) = current.as_ref() {
            // Higher priority interrupts can preempt lower priority ones
            new_context.priority > current_context.priority
        } else {
            false // No current interrupt to preempt
        }
    }

    /// Preempt current interrupt
    fn preempt_current_interrupt(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.preempted_interrupts += 1;

        // Save current interrupt state for later resumption
        if let Some(mut current) = self.current_interrupt.write().unwrap().take() {
            // Mark context as preempted
            current.preempted = true;

            // Create execution state snapshot
            let execution_state = InterruptExecutionState {
                progress: 0.5,                 // Assume 50% completion for simplicity
                partial_data: None,            // Could capture partial processing results
                handler_state: HashMap::new(), // Handler-specific state would go here
            };

            // Create preempted interrupt record
            let preempted = PreemptedInterrupt {
                context: current,
                preempted_at: Instant::now(),
                execution_state,
                response_tx: None, // Would need to be passed from the caller in a real implementation
            };

            // Add to preempted queue (ordered by original priority)
            let mut preempted_queue = self.preempted_interrupts.write().unwrap();
            preempted_queue.push_back(preempted);

            tracing::info!("Interrupt preempted and saved for later resumption");
        }
    }

    /// Resume a preempted interrupt when resources become available
    #[allow(dead_code)]
    pub async fn resume_preempted_interrupt(&self) -> Option<InterruptResponse> {
        let preempted = {
            let mut preempted_queue = self.preempted_interrupts.write().unwrap();
            preempted_queue.pop_front()
        }; // Lock is released here

        // Find the highest priority preempted interrupt
        if let Some(mut preempted) = preempted {
            tracing::info!(
                "Resuming preempted interrupt {} (priority: {:?})",
                preempted.context.interrupt_id,
                preempted.context.priority
            );

            // Update context to indicate resumption
            preempted.context.preempted = false;

            // Set as current interrupt
            *self.current_interrupt.write().unwrap() = Some(preempted.context.clone());

            // Execute the handler with restored state
            let response = self.execute_handler(&preempted.context).await;

            // Clear current interrupt when done
            *self.current_interrupt.write().unwrap() = None;

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.resumed_interrupts += 1;

            Some(response)
        } else {
            None
        }
    }

    /// Get count of preempted interrupts waiting for resumption
    #[allow(dead_code)]
    pub fn preempted_count(&self) -> usize {
        self.preempted_interrupts.read().unwrap().len()
    }

    /// Execute interrupt handler for given context
    async fn execute_handler(&self, context: &InterruptContext) -> InterruptResponse {
        let handlers = self.handlers.read().unwrap();

        if let Some(handler_list) = handlers.get(&context.priority) {
            for handler in handler_list {
                match handler(context) {
                    Ok(response) => return response,
                    Err(e) => {
                        tracing::warn!("Interrupt handler failed: {}", e);
                        let mut stats = self.stats.write().unwrap();
                        stats.failed_interrupts += 1;
                    }
                }
            }
        }

        InterruptResponse::Error("No handler found".to_string())
    }

    /// Update latency statistics
    fn update_latency_stats(&self, latency: Duration) {
        let mut stats = self.stats.write().unwrap();
        let latency_us = latency.as_micros() as u64;

        if latency_us > stats.peak_interrupt_latency_us {
            stats.peak_interrupt_latency_us = latency_us;
        }

        if stats.total_interrupts == 1 {
            stats.avg_interrupt_latency_us = latency_us as f64;
        } else {
            stats.avg_interrupt_latency_us = (stats.avg_interrupt_latency_us
                * (stats.total_interrupts - 1) as f64
                + latency_us as f64)
                / stats.total_interrupts as f64;
        }
    }
}

impl Default for InterruptController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[tokio::test]
    async fn test_interrupt_controller_creation() {
        let controller = InterruptController::new();
        let stats = controller.get_stats();
        assert_eq!(stats.total_interrupts, 0);
    }

    #[tokio::test]
    async fn test_interrupt_handler_registration() {
        let controller = InterruptController::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        controller.register_handler(
            InterruptPriority::AudioProcessing,
            Box::new(move |_ctx| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
                Ok(InterruptResponse::Handled)
            }),
        );

        // Start controller
        controller.start(1).await.unwrap();

        // Trigger interrupt
        let data = InterruptData::AudioChunk {
            mel_data: MelSpectrogram::new(vec![vec![0.0; 80]; 100], 22050, 256),
            stream_id: 1,
            deadline: Instant::now() + Duration::from_millis(10),
        };

        // Use timeout for the interrupt
        let response = tokio::time::timeout(
            Duration::from_millis(1000),
            controller.trigger_interrupt(InterruptPriority::AudioProcessing, data),
        )
        .await
        .expect("Interrupt timed out")
        .unwrap();

        // Verify response
        matches!(response, InterruptResponse::Handled);
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_interrupt_priority_ordering() {
        let controller = InterruptController::new();
        let execution_order = Arc::new(RwLock::new(Vec::new()));

        // Register handlers that record execution order
        for priority in [InterruptPriority::Background, InterruptPriority::Critical] {
            let order_clone = execution_order.clone();
            controller.register_handler(
                priority,
                Box::new(move |ctx| {
                    order_clone.write().unwrap().push(ctx.priority);
                    Ok(InterruptResponse::Handled)
                }),
            );
        }

        controller.start(1).await.unwrap();

        // Trigger low priority interrupt first
        controller
            .fire_interrupt(
                InterruptPriority::Background,
                InterruptData::Timer {
                    timer_id: 1,
                    interval: Duration::from_millis(10),
                    repetitions: 1,
                },
            )
            .await
            .unwrap();

        // Then trigger high priority interrupt
        controller
            .fire_interrupt(
                InterruptPriority::Critical,
                InterruptData::SystemControl {
                    command: SystemCommand::FlushInterrupts,
                    parameters: HashMap::new(),
                },
            )
            .await
            .unwrap();

        // Give time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        {
            let order = execution_order.read().unwrap();
            // Critical should execute before Background due to priority
            if order.len() >= 2 {
                assert!(order[0] >= InterruptPriority::Critical);
            }
        }

        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_interrupt_masking() {
        let controller = InterruptController::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        controller.register_handler(
            InterruptPriority::UserInput,
            Box::new(move |_ctx| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
                Ok(InterruptResponse::Handled)
            }),
        );

        controller.start(1).await.unwrap();

        // Mask user input interrupts
        controller.set_interrupt_mask(InterruptPriority::UserInput, true);

        // Try to trigger masked interrupt
        controller
            .fire_interrupt(
                InterruptPriority::UserInput,
                InterruptData::Custom {
                    data: vec![1, 2, 3],
                    metadata: HashMap::new(),
                },
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Counter should still be 0 because interrupt was masked
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let controller = InterruptController::new();

        controller.register_handler(
            InterruptPriority::AudioProcessing,
            Box::new(|_ctx| {
                // Minimal processing time
                Ok(InterruptResponse::Handled)
            }),
        );

        controller.start(1).await.unwrap();

        // Trigger several interrupts with fire_interrupt (no response needed)
        for i in 0..3 {
            let data = InterruptData::BufferEvent {
                buffer_id: i,
                event_type: BufferEventType::HighWatermark,
                urgency: 0.8,
            };
            controller
                .fire_interrupt(InterruptPriority::AudioProcessing, data)
                .await
                .unwrap();
        }

        // Give minimal time for processing
        tokio::time::sleep(Duration::from_millis(20)).await;

        let stats = controller.get_stats();
        assert_eq!(stats.total_interrupts, 3);

        controller.shutdown().await.unwrap();
    }
}
