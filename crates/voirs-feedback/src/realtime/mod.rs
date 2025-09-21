//! Real-time feedback system implementation
//!
//! This module provides real-time audio processing and feedback generation capabilities.
//! The system is divided into several focused modules:
//!
//! - [`types`]: Core data structures and type definitions
//! - [`config`]: Configuration management and system settings
//! - [`phoneme`]: Phoneme analysis and processing
//! - [`audio_processing`]: Audio processing and feature extraction
//! - [`suggestions`]: Contextual suggestion engine
//! - [`multimodal`]: Multi-modal feedback delivery
//! - [`performance`]: Performance monitoring and optimization
//! - [`stream`]: Stream management and processing
//! - [`system`]: Core real-time feedback system
//! - [`websocket`]: WebSocket-based real-time communication
//! - [`ui_responsive_queue`]: UI-responsive queue management for non-blocking operations

pub mod audio_processing;
pub mod config;
pub mod multimodal;
pub mod performance;
pub mod phoneme;
pub mod stream;
pub mod suggestions;
pub mod system;
pub mod types;
pub mod ui_responsive_queue;
pub mod websocket;

// Tests are included inline in each module

// Re-export core types for backwards compatibility
pub use config::*;
pub use stream::FeedbackStream;
pub use system::RealtimeFeedbackSystem;
pub use types::*;
pub use ui_responsive_queue::{FeedbackQueue, QueueResult, QueueStats, UiResponsiveQueue};
pub use websocket::{RealtimeCommunicationManager, WebSocketClient, WebSocketMessage};
