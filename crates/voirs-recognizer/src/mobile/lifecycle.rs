//! App lifecycle management for mobile platforms
//!
//! This module handles app lifecycle events such as:
//! - App becoming active/inactive
//! - Entering background/foreground
//! - Termination events
//! - Memory warnings

use crate::RecognitionError;

/// App lifecycle manager
pub struct LifecycleManager {
    /// Current app state
    current_state: AppState,
    /// Background task tracking
    background_tasks: Vec<BackgroundTask>,
    /// State transition history
    state_history: Vec<StateTransition>,
}

/// Application lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    /// App is active and in foreground
    Active,
    /// App is inactive but visible
    Inactive,
    /// App is in background
    Background,
    /// App is suspended
    Suspended,
    /// App is being terminated
    Terminating,
}

/// Lifecycle events that can occur
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleEvent {
    /// App became active
    DidBecomeActive,
    /// App will resign active state
    WillResignActive,
    /// App entered background
    DidEnterBackground,
    /// App will enter foreground
    WillEnterForeground,
    /// App received memory warning
    MemoryWarning,
    /// App will terminate
    WillTerminate,
}

/// State transition record
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Previous state
    from: AppState,
    /// New state
    to: AppState,
    /// When the transition occurred
    timestamp: std::time::Instant,
}

/// Background task information
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    /// Task identifier
    id: String,
    /// Task start time
    started: std::time::Instant,
    /// Task description
    description: String,
}

impl LifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        Self {
            current_state: AppState::Active,
            background_tasks: Vec::new(),
            state_history: Vec::new(),
        }
    }

    /// Handle a lifecycle event
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be handled
    pub async fn handle_event(&mut self, event: LifecycleEvent) -> Result<(), RecognitionError> {
        let new_state = match event {
            LifecycleEvent::DidBecomeActive => {
                self.on_became_active().await?;
                AppState::Active
            }
            LifecycleEvent::WillResignActive => {
                self.on_will_resign_active().await?;
                AppState::Inactive
            }
            LifecycleEvent::DidEnterBackground => {
                self.on_entered_background().await?;
                AppState::Background
            }
            LifecycleEvent::WillEnterForeground => {
                self.on_will_enter_foreground().await?;
                AppState::Inactive
            }
            LifecycleEvent::MemoryWarning => {
                self.on_memory_warning().await?;
                self.current_state // State doesn't change
            }
            LifecycleEvent::WillTerminate => {
                self.on_will_terminate().await?;
                AppState::Terminating
            }
        };

        if new_state != self.current_state {
            self.record_state_transition(new_state);
        }

        Ok(())
    }

    /// Get current app state
    pub fn current_state(&self) -> AppState {
        self.current_state
    }

    /// Check if app is in foreground
    pub fn is_foreground(&self) -> bool {
        matches!(self.current_state, AppState::Active | AppState::Inactive)
    }

    /// Check if app is in background
    pub fn is_background(&self) -> bool {
        matches!(
            self.current_state,
            AppState::Background | AppState::Suspended
        )
    }

    /// Get active background tasks
    pub fn active_background_tasks(&self) -> &[BackgroundTask] {
        &self.background_tasks
    }

    /// Start a background task
    pub fn begin_background_task(&mut self, description: String) -> String {
        let task_id = format!("bg_task_{}", self.background_tasks.len());
        let task = BackgroundTask {
            id: task_id.clone(),
            started: std::time::Instant::now(),
            description,
        };
        self.background_tasks.push(task);
        task_id
    }

    /// End a background task
    pub fn end_background_task(&mut self, task_id: &str) {
        self.background_tasks.retain(|task| task.id != task_id);
    }

    /// Handle app becoming active
    async fn on_became_active(&mut self) -> Result<(), RecognitionError> {
        // Resume normal processing
        // Re-enable full CPU usage
        // Resume any paused operations
        Ok(())
    }

    /// Handle app will resign active state
    async fn on_will_resign_active(&mut self) -> Result<(), RecognitionError> {
        // Reduce processing intensity
        // Save any critical state
        Ok(())
    }

    /// Handle app entered background
    async fn on_entered_background(&mut self) -> Result<(), RecognitionError> {
        // Stop non-critical operations
        // Release resources
        // Save complete app state
        Ok(())
    }

    /// Handle app will enter foreground
    async fn on_will_enter_foreground(&mut self) -> Result<(), RecognitionError> {
        // Prepare to resume operations
        // Restore saved state
        Ok(())
    }

    /// Handle memory warning
    async fn on_memory_warning(&mut self) -> Result<(), RecognitionError> {
        // Aggressive memory cleanup
        // Clear caches
        // Release non-essential resources
        Ok(())
    }

    /// Handle app will terminate
    async fn on_will_terminate(&mut self) -> Result<(), RecognitionError> {
        // Save critical data
        // Clean up resources
        // Cancel background tasks
        self.background_tasks.clear();
        Ok(())
    }

    /// Record a state transition
    fn record_state_transition(&mut self, new_state: AppState) {
        let transition = StateTransition {
            from: self.current_state,
            to: new_state,
            timestamp: std::time::Instant::now(),
        };
        self.state_history.push(transition);
        self.current_state = new_state;

        // Keep only recent history (last 100 transitions)
        if self.state_history.len() > 100 {
            self.state_history.drain(0..self.state_history.len() - 100);
        }
    }

    /// Get recent state transitions
    pub fn recent_transitions(&self, count: usize) -> &[StateTransition] {
        let start = if self.state_history.len() > count {
            self.state_history.len() - count
        } else {
            0
        };
        &self.state_history[start..]
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle_manager_creation() {
        let manager = LifecycleManager::new();
        assert_eq!(manager.current_state(), AppState::Active);
    }

    #[tokio::test]
    async fn test_lifecycle_transitions() {
        let mut manager = LifecycleManager::new();

        // Test becoming inactive
        manager
            .handle_event(LifecycleEvent::WillResignActive)
            .await
            .unwrap();
        assert_eq!(manager.current_state(), AppState::Inactive);

        // Test entering background
        manager
            .handle_event(LifecycleEvent::DidEnterBackground)
            .await
            .unwrap();
        assert_eq!(manager.current_state(), AppState::Background);
        assert!(manager.is_background());
        assert!(!manager.is_foreground());

        // Test entering foreground
        manager
            .handle_event(LifecycleEvent::WillEnterForeground)
            .await
            .unwrap();
        assert_eq!(manager.current_state(), AppState::Inactive);

        // Test becoming active
        manager
            .handle_event(LifecycleEvent::DidBecomeActive)
            .await
            .unwrap();
        assert_eq!(manager.current_state(), AppState::Active);
        assert!(manager.is_foreground());
        assert!(!manager.is_background());
    }

    #[tokio::test]
    async fn test_background_tasks() {
        let mut manager = LifecycleManager::new();

        // Start background task
        let task_id = manager.begin_background_task("Test task".to_string());
        assert_eq!(manager.active_background_tasks().len(), 1);

        // End background task
        manager.end_background_task(&task_id);
        assert_eq!(manager.active_background_tasks().len(), 0);
    }

    #[tokio::test]
    async fn test_memory_warning() {
        let mut manager = LifecycleManager::new();

        // Memory warning shouldn't change state
        let original_state = manager.current_state();
        manager
            .handle_event(LifecycleEvent::MemoryWarning)
            .await
            .unwrap();
        assert_eq!(manager.current_state(), original_state);
    }

    #[tokio::test]
    async fn test_state_history() {
        let mut manager = LifecycleManager::new();

        // Perform several state transitions
        manager
            .handle_event(LifecycleEvent::WillResignActive)
            .await
            .unwrap();
        manager
            .handle_event(LifecycleEvent::DidEnterBackground)
            .await
            .unwrap();
        manager
            .handle_event(LifecycleEvent::WillEnterForeground)
            .await
            .unwrap();

        // Check history
        let recent = manager.recent_transitions(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].to, AppState::Inactive);
        assert_eq!(recent[1].to, AppState::Background);
        assert_eq!(recent[2].to, AppState::Inactive);
    }
}
