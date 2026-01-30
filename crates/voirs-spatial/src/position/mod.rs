//! Position tracking and listener/source management
//!
//! This module provides comprehensive spatial audio positioning with:
//! - 3D listener and sound source tracking
//! - Head tracking with prediction and smoothing
//! - Occlusion detection and processing
//! - Doppler effect processing for moving sources
//! - Spatial grid for efficient proximity queries
//! - VR/AR platform integration

// Module declarations
pub mod advanced_prediction;
pub mod occlusion;
pub mod prediction;
pub mod source_manager;
pub mod spatial_grid;
pub mod tracking;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use occlusion::{
    DiffractionPath, OcclusionDetector, OcclusionMaterial, OcclusionMethod, OcclusionResult,
};
pub use prediction::{MotionPredictor, MotionSnapshot};
pub use source_manager::{
    DopplerProcessor, DynamicSource, DynamicSourceManager, SpatialSourceManager,
};
pub use spatial_grid::SpatialGrid;
pub use tracking::{HeadTracker, ListenerMovementSystem, MovementTracker, PlatformIntegration};
pub use types::{
    AttenuationModel, AttenuationParams, Box3D, CalibrationData, ComfortSettings,
    DirectivityPattern, FrequencyGain, Listener, MovementConstraints, MovementMetrics,
    NavigationMode, OrientationSnapshot, PlatformData, PlatformType, PositionSnapshot, SoundSource,
    SourceType,
};

// Re-export advanced prediction types
pub use advanced_prediction::{
    AccelerationProfile, AdaptationPhase, AdaptationState, AdaptivePredictionController,
    AdvancedPredictiveTracker, KalmanMotionFilter, LinearMotionModel, ModelSelectionStrategy,
    MotionPattern, MotionPatternAnalyzer, MotionPatternParameters, MotionPatternTemplate,
    MotionPatternType, NeuralModelConfig, NeuralPredictionModel, PatternRecognitionConfig,
    PatternRecognitionState, PerformanceOptimizationConfig, PolynomialMotionModel,
    PredictedPosition, PredictionAccuracy, PredictionMetrics, PredictionModelType,
    PredictionModels, PredictionNetwork, PredictionResult, PredictiveTrackingConfig,
    TrainingExample,
};
