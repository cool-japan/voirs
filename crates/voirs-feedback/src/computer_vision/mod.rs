//! Computer Vision Analysis Module
//!
//! This module provides comprehensive computer vision analysis capabilities for
//! real-time feedback systems, including facial expression recognition, lip movement
//! analysis, gesture detection, posture assessment, and eye gaze tracking.

/// Description
pub mod core;
/// Description
pub mod eye_tracking;
/// Description
pub mod facial;
/// Description
pub mod gesture;
/// Description
pub mod types;

// Re-export all public types and traits to maintain backward compatibility
pub use core::{ComputerVisionSystem, MLComputerVisionAnalyzer, SimpleLandmarkDetector};
pub use types::*;

// Re-export specific analyzers for modular usage
pub use eye_tracking::EyeTrackingAnalyzer;
pub use facial::FacialAnalyzer;
pub use gesture::GesturePostureAnalyzer;
