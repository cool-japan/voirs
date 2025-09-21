//! Thread-safe callback handling for VoiRS FFI operations
//!
//! This module provides safe callback management for cross-language operations,
//! ensuring thread safety and proper resource cleanup.

use crate::threading::sync::{CancellationToken, ThreadStats};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

/// Thread-safe callback manager
pub struct CallbackManager {
    callbacks: Arc<Mutex<HashMap<CallbackId, CallbackEntry>>>,
    next_id: AtomicU64,
    stats: Arc<ThreadStats>,
}

/// Unique callback identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallbackId(u64);

/// Callback entry containing the callback and metadata
struct CallbackEntry {
    callback: Box<dyn Fn(&CallbackData) + Send + Sync>,
    callback_type: CallbackType,
    registered_at: Instant,
    call_count: AtomicU64,
    last_called: Mutex<Option<Instant>>,
}

/// Types of callbacks supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackType {
    /// Progress callback for long-running operations
    Progress,
    /// Error callback for error handling
    Error,
    /// Completion callback for operation completion
    Completion,
    /// Custom callback for application-specific needs
    Custom,
}

/// Data passed to callbacks
#[derive(Debug, Clone)]
pub struct CallbackData {
    pub operation_id: u64,
    pub callback_type: CallbackType,
    pub timestamp: Instant,
    pub data: CallbackPayload,
}

/// Callback payload containing operation-specific data
#[derive(Debug, Clone)]
pub enum CallbackPayload {
    /// Progress data with percentage and optional message
    Progress {
        percentage: f32,
        message: Option<String>,
        bytes_processed: u64,
        total_bytes: u64,
    },
    /// Error data with error code and message
    Error {
        error_code: i32,
        message: String,
        recoverable: bool,
    },
    /// Completion data with results
    Completion {
        success: bool,
        duration: Duration,
        result_size: usize,
    },
    /// Custom data for application-specific callbacks
    Custom { name: String, data: Vec<u8> },
}

/// Callback execution context
pub struct CallbackContext {
    operation_id: u64,
    cancellation_token: Arc<CancellationToken>,
    start_time: Instant,
    stats: Arc<ThreadStats>,
}

impl CallbackManager {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            stats: Arc::new(ThreadStats::new()),
        }
    }

    /// Register a new callback
    pub fn register<F>(&self, callback_type: CallbackType, callback: F) -> CallbackId
    where
        F: Fn(&CallbackData) + Send + Sync + 'static,
    {
        let id = CallbackId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let entry = CallbackEntry {
            callback: Box::new(callback),
            callback_type,
            registered_at: Instant::now(),
            call_count: AtomicU64::new(0),
            last_called: Mutex::new(None),
        };

        self.callbacks.lock().insert(id, entry);
        id
    }

    /// Unregister a callback
    pub fn unregister(&self, id: CallbackId) -> bool {
        self.callbacks.lock().remove(&id).is_some()
    }

    /// Trigger callbacks of a specific type
    pub fn trigger(&self, callback_type: CallbackType, data: CallbackData) {
        let callbacks = self.callbacks.lock();

        for (_, entry) in callbacks.iter() {
            if entry.callback_type == callback_type {
                // Update metadata
                entry.call_count.fetch_add(1, Ordering::Relaxed);
                *entry.last_called.lock() = Some(Instant::now());

                // Execute callback safely
                let callback_start = Instant::now();
                (entry.callback)(&data);
                let callback_duration = callback_start.elapsed();

                // Update stats
                self.stats
                    .complete_operation(callback_duration.as_nanos() as u64);
            }
        }
    }

    /// Trigger a specific callback by ID
    pub fn trigger_by_id(&self, id: CallbackId, data: CallbackData) -> bool {
        let callbacks = self.callbacks.lock();

        if let Some(entry) = callbacks.get(&id) {
            // Update metadata
            entry.call_count.fetch_add(1, Ordering::Relaxed);
            *entry.last_called.lock() = Some(Instant::now());

            // Execute callback safely
            let callback_start = Instant::now();
            (entry.callback)(&data);
            let callback_duration = callback_start.elapsed();

            // Update stats
            self.stats
                .complete_operation(callback_duration.as_nanos() as u64);
            true
        } else {
            false
        }
    }

    /// Get callback statistics
    pub fn get_callback_stats(&self, id: CallbackId) -> Option<CallbackStats> {
        let callbacks = self.callbacks.lock();

        callbacks.get(&id).map(|entry| CallbackStats {
            id,
            callback_type: entry.callback_type,
            registered_at: entry.registered_at,
            call_count: entry.call_count.load(Ordering::Relaxed),
            last_called: *entry.last_called.lock(),
        })
    }

    /// Get all registered callback IDs
    pub fn get_registered_callbacks(&self) -> Vec<CallbackId> {
        self.callbacks.lock().keys().cloned().collect()
    }

    /// Clear all callbacks
    pub fn clear(&self) {
        self.callbacks.lock().clear();
    }

    /// Get manager statistics
    pub fn stats(&self) -> &ThreadStats {
        &self.stats
    }
}

/// Statistics for a specific callback
#[derive(Debug, Clone)]
pub struct CallbackStats {
    pub id: CallbackId,
    pub callback_type: CallbackType,
    pub registered_at: Instant,
    pub call_count: u64,
    pub last_called: Option<Instant>,
}

impl CallbackContext {
    /// Create a new callback context
    pub fn new(operation_id: u64, cancellation_token: Arc<CancellationToken>) -> Self {
        Self {
            operation_id,
            cancellation_token,
            start_time: Instant::now(),
            stats: Arc::new(ThreadStats::new()),
        }
    }

    /// Report progress
    pub fn report_progress(
        &self,
        manager: &CallbackManager,
        percentage: f32,
        message: Option<String>,
        bytes_processed: u64,
        total_bytes: u64,
    ) {
        if self.cancellation_token.is_cancelled() {
            return;
        }

        let data = CallbackData {
            operation_id: self.operation_id,
            callback_type: CallbackType::Progress,
            timestamp: Instant::now(),
            data: CallbackPayload::Progress {
                percentage,
                message,
                bytes_processed,
                total_bytes,
            },
        };

        manager.trigger(CallbackType::Progress, data);
    }

    /// Report error
    pub fn report_error(
        &self,
        manager: &CallbackManager,
        error_code: i32,
        message: String,
        recoverable: bool,
    ) {
        let data = CallbackData {
            operation_id: self.operation_id,
            callback_type: CallbackType::Error,
            timestamp: Instant::now(),
            data: CallbackPayload::Error {
                error_code,
                message,
                recoverable,
            },
        };

        manager.trigger(CallbackType::Error, data);
    }

    /// Report completion
    pub fn report_completion(&self, manager: &CallbackManager, success: bool, result_size: usize) {
        let data = CallbackData {
            operation_id: self.operation_id,
            callback_type: CallbackType::Completion,
            timestamp: Instant::now(),
            data: CallbackPayload::Completion {
                success,
                duration: self.start_time.elapsed(),
                result_size,
            },
        };

        manager.trigger(CallbackType::Completion, data);
    }

    /// Report custom data
    pub fn report_custom(&self, manager: &CallbackManager, name: String, data: Vec<u8>) {
        let callback_data = CallbackData {
            operation_id: self.operation_id,
            callback_type: CallbackType::Custom,
            timestamp: Instant::now(),
            data: CallbackPayload::Custom { name, data },
        };

        manager.trigger(CallbackType::Custom, callback_data);
    }

    /// Check if operation is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Get operation duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Safe callback executor that handles panics
pub struct SafeCallbackExecutor {
    manager: Arc<CallbackManager>,
    panic_handler: Box<dyn Fn(&str) + Send + Sync>,
}

impl SafeCallbackExecutor {
    /// Create a new safe callback executor
    pub fn new<F>(manager: Arc<CallbackManager>, panic_handler: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        Self {
            manager,
            panic_handler: Box::new(panic_handler),
        }
    }

    /// Execute callback with panic protection
    pub fn execute(&self, callback_type: CallbackType, data: CallbackData) {
        let manager = Arc::clone(&self.manager);
        let panic_handler = &self.panic_handler;

        // Execute in a separate thread to isolate panics
        let handle = thread::spawn(move || {
            manager.trigger(callback_type, data);
        });

        // Wait for completion and handle panics
        match handle.join() {
            Ok(()) => {
                // Callback executed successfully
            }
            Err(panic) => {
                let panic_msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic in callback".to_string()
                };

                (panic_handler)(&panic_msg);
            }
        }
    }
}

/// Global callback manager instance
static GLOBAL_CALLBACK_MANAGER: std::sync::OnceLock<CallbackManager> = std::sync::OnceLock::new();

/// Get the global callback manager
pub fn global_callback_manager() -> &'static CallbackManager {
    GLOBAL_CALLBACK_MANAGER.get_or_init(CallbackManager::new)
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_callback_registration() {
        let manager = CallbackManager::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let id = manager.register(CallbackType::Progress, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        assert!(manager.get_registered_callbacks().contains(&id));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_callback_triggering() {
        let manager = CallbackManager::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        manager.register(CallbackType::Progress, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        let data = CallbackData {
            operation_id: 1,
            callback_type: CallbackType::Progress,
            timestamp: Instant::now(),
            data: CallbackPayload::Progress {
                percentage: 50.0,
                message: Some("Test progress".to_string()),
                bytes_processed: 500,
                total_bytes: 1000,
            },
        };

        manager.trigger(CallbackType::Progress, data);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_callback_unregistration() {
        let manager = CallbackManager::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let id = manager.register(CallbackType::Progress, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        assert!(manager.unregister(id));
        assert!(!manager.get_registered_callbacks().contains(&id));
    }

    #[test]
    fn test_callback_context() {
        let manager = CallbackManager::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        manager.register(CallbackType::Progress, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        let token = Arc::new(CancellationToken::new());
        let context = CallbackContext::new(1, token);

        context.report_progress(&manager, 25.0, None, 250, 1000);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_callback_stats() {
        let manager = CallbackManager::new();
        let id = manager.register(CallbackType::Progress, |_| {});

        let data = CallbackData {
            operation_id: 1,
            callback_type: CallbackType::Progress,
            timestamp: Instant::now(),
            data: CallbackPayload::Progress {
                percentage: 50.0,
                message: None,
                bytes_processed: 500,
                total_bytes: 1000,
            },
        };

        manager.trigger_by_id(id, data);

        let stats = manager.get_callback_stats(id).unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.last_called.is_some());
    }

    #[test]
    fn test_cancellation_in_context() {
        let manager = CallbackManager::new();
        let token = Arc::new(CancellationToken::new());
        let context = CallbackContext::new(1, Arc::clone(&token));

        assert!(!context.is_cancelled());
        token.cancel(None);
        assert!(context.is_cancelled());
    }

    #[test]
    fn test_global_callback_manager() {
        let manager = global_callback_manager();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        manager.register(CallbackType::Custom, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        let data = CallbackData {
            operation_id: 1,
            callback_type: CallbackType::Custom,
            timestamp: Instant::now(),
            data: CallbackPayload::Custom {
                name: "test".to_string(),
                data: vec![1, 2, 3],
            },
        };

        manager.trigger(CallbackType::Custom, data);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
