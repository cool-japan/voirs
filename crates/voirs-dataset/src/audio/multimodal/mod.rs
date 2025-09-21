//! Multi-modal processing modules
//!
//! This module provides comprehensive multi-modal processing capabilities including video-audio
//! synchronization, visual speech alignment, gesture-speech correlation, and multi-modal quality
//! assessment for enhanced analysis and processing in speech synthesis datasets.
//!
//! # Architecture
//!
//! The multi-modal processing system is organized into several specialized modules:
//!
//! - [`video`] - Video processing and frame analysis
//! - [`synchronization`] - Audio-video synchronization
//! - [`alignment`] - Visual speech alignment and phoneme-viseme mapping
//! - [`gesture`] - Gesture detection and analysis
//! - [`quality`] - Multi-modal quality assessment
//! - [`optimization`] - Processing optimization and performance tuning
//! - [`processor`] - Core processor trait and implementation

// Module declarations
pub mod alignment;
pub mod gesture;
pub mod optimization;
pub mod processor;
pub mod quality;
pub mod synchronization;
pub mod video;

// Re-export key types from submodules for convenience
pub use alignment::{
    AlignmentConfig, AlignmentMethod, AlignmentResult, AlignmentSegment, CoarticulationConfig,
    PhonemeVisemeAlignment, PhonemeVisemeMapping, TemporalAlignment, TemporalAlignmentConfig,
    TemporalBoundary, UtteranceAlignment,
};
pub use gesture::{
    DetectedGesture, GestureAnalysisResult, GestureCategory, GestureClassificationConfig,
    GestureConfig, GestureFeatureExtraction, GestureSpeechCorrelation,
    GestureSpeechCorrelationConfig,
};
pub use optimization::{
    BatchProcessingConfig, LoadBalancingStrategy, MemoryOptimization, PerformanceStats,
    ProcessingOptimization, ResourceUtilization,
};
pub use processor::{
    DefaultMultiModalProcessor, MultiModalConfig, MultiModalProcessingResult, MultiModalProcessor,
};
pub use quality::{
    MultiModalQualityConfig, MultiModalQualityMetric, MultiModalQualityResults,
    QualityAggregationStrategy, QualityAssessmentMethod, QualityReportingConfig,
};
pub use synchronization::{
    AlignmentState, AutoCorrectionConfig, InterpolationMethod, SyncAnalysis, SyncConfig,
    SyncMethod, SynchronizationResult,
};
pub use video::{
    ColorSpace, FaceDetectionConfig, FaceDetectionModel, Frame, LipDetectionModel,
    LipExtractionConfig, LipNormalization, SmoothingMethod, TemporalSmoothingConfig, VideoConfig,
    VideoData, VideoFrame,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioData;

    #[test]
    fn test_configuration_serialization() {
        let config = MultiModalConfig::default();

        // Test that the configuration can be serialized
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        // Test that it can be deserialized
        let deserialized: Result<MultiModalConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_video_data_creation() {
        let video = VideoData::new(30.0, (1920, 1080), ColorSpace::RGB);
        assert_eq!(video.fps, 30.0);
        assert_eq!(video.dimensions, (1920, 1080));
        assert!(matches!(video.color_space, ColorSpace::RGB));
        assert_eq!(video.frame_count(), 0);
        assert_eq!(video.duration, 0.0);
    }

    #[test]
    fn test_alignment_result() {
        let mut alignment = UtteranceAlignment::new();

        let segment = alignment::AlignmentSegment::new(0.0, 1.0, "hello".to_string(), 0.9);
        alignment.add_segment(segment);

        assert_eq!(alignment.segments.len(), 1);
        assert_eq!(alignment.overall_quality, 0.9);
        assert_eq!(alignment.total_duration(), 1.0);
    }

    #[test]
    fn test_synchronization_result() {
        let mut sync_result = SynchronizationResult::new(0.1, 0.8);

        sync_result.add_quality_metric("cross_correlation".to_string(), 0.8);
        sync_result.mark_correction_applied();

        assert_eq!(sync_result.time_offset, 0.1);
        assert_eq!(sync_result.confidence, 0.8);
        assert!(sync_result.correction_applied);
        assert!(sync_result.is_synchronized(0.7));
    }

    #[test]
    fn test_gesture_analysis_result() {
        let mut gesture_result = GestureAnalysisResult::new();

        let gesture = DetectedGesture::new(
            "g1".to_string(),
            0.0,
            1.0,
            gesture::GestureCategory::Iconic,
            0.8,
        );

        gesture_result.add_gesture(gesture);
        gesture_result.calculate_quality_score();

        assert_eq!(gesture_result.gestures.len(), 1);
        assert_eq!(gesture_result.quality_score, 0.4); // (0.8 + 0.0) / 2
    }

    #[test]
    fn test_multimodal_quality_results() {
        let mut quality_results = MultiModalQualityResults::new();

        quality_results.add_metric_score("sync_quality".to_string(), 0.8);
        quality_results.add_metric_score("alignment_quality".to_string(), 0.7);

        assert_eq!(quality_results.metric_scores.len(), 2);
        assert_eq!(quality_results.overall_score, 0.75); // (0.8 + 0.7) / 2
    }

    #[test]
    fn test_default_multimodal_processor() {
        let _processor = DefaultMultiModalProcessor::with_default_config();

        // Test that the processor can be created with default configuration
        // If we get here, creation was successful
    }

    #[test]
    fn test_sync_analysis() {
        let mut analysis = SyncAnalysis::new();

        analysis.add_correlation(0.0, 0.3);
        analysis.add_correlation(0.1, 0.7);
        analysis.add_correlation(0.2, 0.9);
        analysis.add_correlation(0.3, 0.5);

        assert_eq!(analysis.peak_correlation, 0.9);
        assert_eq!(analysis.peak_lag, 0.2);

        analysis.calculate_confidence_interval(0.8);
        assert!(analysis.confidence_interval.0 <= analysis.confidence_interval.1);
    }

    #[test]
    fn test_alignment_state() {
        let mut state = AlignmentState::new();

        state.update_offset(0.1, 1.0);
        state.update_offset(0.15, 2.0);
        state.update_offset(0.12, 3.0);

        assert_eq!(state.current_offset, 0.12);
        assert_eq!(state.offset_history.len(), 3); // 3 updates = 3 history items (0.0, 0.1, 0.15)
        assert!(state.stability > 0.0);

        let predicted = state.predict_offset(4.0);
        assert!(predicted.is_finite());

        assert!(state.needs_correction(0.05));
    }

    #[test]
    fn test_phoneme_viseme_mapping() {
        let mut mapping = PhonemeVisemeMapping::new();

        mapping.add_mapping("p".to_string(), "p_viseme".to_string(), 0.9);
        mapping.add_language_mapping("en".to_string(), "th".to_string(), "th_viseme".to_string());

        assert_eq!(mapping.get_viseme("p", None), Some(&"p_viseme".to_string()));
        assert_eq!(
            mapping.get_viseme("th", Some("en")),
            Some(&"th_viseme".to_string())
        );
        assert_eq!(mapping.get_confidence("p"), 0.9);
    }

    #[tokio::test]
    async fn test_multimodal_processor_synchronize() {
        let processor = DefaultMultiModalProcessor::with_default_config();
        let audio = AudioData::silence(1.0, 44100, 2);
        let video = VideoData::new(25.0, (640, 480), ColorSpace::RGB);

        let result = processor.synchronize(&audio, &video).await;
        assert!(result.is_ok());

        let sync_result = result.unwrap();
        assert!(sync_result.confidence >= 0.0);
        assert!(sync_result.time_offset >= 0.0);
    }

    #[tokio::test]
    async fn test_multimodal_processor_analyze_gestures() {
        let processor = DefaultMultiModalProcessor::with_default_config();
        let video = VideoData::new(25.0, (640, 480), ColorSpace::RGB);

        let result = processor.analyze_gestures(&video).await;
        assert!(result.is_ok());

        let gesture_result = result.unwrap();
        assert_eq!(gesture_result.gestures.len(), 0); // Placeholder implementation
    }

    #[tokio::test]
    async fn test_multimodal_processor_process() {
        let processor = DefaultMultiModalProcessor::with_default_config();
        let audio = AudioData::silence(1.0, 44100, 2);
        let video = VideoData::new(25.0, (640, 480), ColorSpace::RGB);

        let result = processor.process(&audio, &video).await;
        assert!(result.is_ok());

        let processing_result = result.unwrap();
        assert!(processing_result.overall_quality() >= 0.0);
    }

    #[test]
    fn test_processing_optimization() {
        let optimization = ProcessingOptimization::new()
            .with_gpu_acceleration()
            .with_threads(8)
            .with_batch_size(64)
            .with_max_memory(2048);

        assert!(optimization.gpu_acceleration);
        assert_eq!(optimization.num_threads, 8);
        assert_eq!(optimization.batch_processing.batch_size, 64);
        assert_eq!(optimization.memory_optimization.max_memory_mb, 2048);
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::new();

        stats.update_throughput(100.0);
        stats.update_latency(50.0);
        stats.update_latency(60.0);

        assert_eq!(stats.throughput, 100.0);
        assert_eq!(stats.avg_latency, 55.0); // (50 + 60) / 2
        assert_eq!(stats.peak_latency, 60.0);

        let thresholds = optimization::PerformanceThresholds::default();
        assert!(stats.is_within_limits(&thresholds));
    }

    #[test]
    fn test_video_frame_operations() {
        let mut video = VideoData::new(30.0, (640, 480), ColorSpace::RGB);

        let frame = video::Frame::new(0, 0.0, vec![255; 640 * 480 * 3]).with_metadata(
            "test".to_string(),
            serde_json::Value::String("value".to_string()),
        );

        video.add_frame(frame);

        assert_eq!(video.frame_count(), 1);
        assert!(video.duration > 0.0);

        let retrieved_frame = video.get_frame(0);
        assert!(retrieved_frame.is_some());

        let frame_at_time = video.get_frame_at_time(0.0);
        assert!(frame_at_time.is_some());
    }
}
