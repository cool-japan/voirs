//! Video processing configuration for multi-modal audio-video analysis

use serde::{Deserialize, Serialize};

/// Video processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Target frame rate for analysis
    pub target_fps: f32,
    /// Frame resolution for processing
    pub resolution: (u32, u32),
    /// Color space for processing
    pub color_space: ColorSpace,
    /// Face detection configuration
    pub face_detection: FaceDetectionConfig,
    /// Lip region extraction settings
    pub lip_extraction: LipExtractionConfig,
    /// Temporal smoothing parameters
    pub temporal_smoothing: TemporalSmoothingConfig,
}

/// Color space options for video processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorSpace {
    /// RGB color space
    RGB,
    /// YUV color space
    YUV,
    /// HSV color space
    HSV,
    /// LAB color space
    LAB,
    /// Grayscale
    Grayscale,
}

/// Face detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetectionConfig {
    /// Detection model type
    pub model_type: FaceDetectionModel,
    /// Detection confidence threshold
    pub confidence_threshold: f32,
    /// Maximum number of faces to detect
    pub max_faces: usize,
    /// Minimum face size (pixels)
    pub min_face_size: u32,
    /// Non-maximum suppression threshold
    pub nms_threshold: f32,
}

/// Face detection model options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaceDetectionModel {
    /// Haar cascade classifier
    HaarCascade,
    /// HOG-based detector
    HOG,
    /// Deep learning-based detector
    DNN,
    /// MediaPipe Face Detection
    MediaPipe,
    /// OpenCV DNN Face Detection
    OpenCVDNN,
}

/// Lip region extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipExtractionConfig {
    /// Lip detection model
    pub model: LipDetectionModel,
    /// Region of interest expansion factor
    pub roi_expansion: f32,
    /// Landmark detection precision
    pub landmark_precision: LandmarkPrecision,
    /// Lip contour smoothing
    pub contour_smoothing: bool,
    /// Normalization parameters
    pub normalization: LipNormalization,
}

/// Lip detection model options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LipDetectionModel {
    /// Facial landmark-based detection
    FacialLandmarks,
    /// Deep learning lip segmentation
    DeepLipSegmentation,
    /// Active appearance model
    ActiveAppearanceModel,
    /// Constrained local model
    ConstrainedLocalModel,
}

/// Landmark detection precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LandmarkPrecision {
    /// Basic 68-point landmarks
    Basic68,
    /// Extended 468-point landmarks
    Extended468,
    /// High-precision 1468-point landmarks
    HighPrecision1468,
}

/// Lip normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipNormalization {
    /// Normalize lip region size
    pub size_normalization: bool,
    /// Normalize lip position
    pub position_normalization: bool,
    /// Normalize lip orientation
    pub orientation_normalization: bool,
    /// Target lip region size
    pub target_size: (u32, u32),
}

/// Temporal smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSmoothingConfig {
    /// Smoothing window size (frames)
    pub window_size: usize,
    /// Smoothing method
    pub method: SmoothingMethod,
    /// Outlier detection threshold
    pub outlier_threshold: f32,
    /// Motion compensation
    pub motion_compensation: bool,
}

/// Smoothing methods for temporal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// Moving average
    MovingAverage,
    /// Gaussian smoothing
    Gaussian,
    /// Kalman filtering
    Kalman,
    /// Savitzky-Golay filter
    SavitzkyGolay,
    /// Butterworth filter
    Butterworth,
}

/// Video data structure for storing frame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    /// Video frames as raw pixel data
    pub frames: Vec<Frame>,
    /// Frame rate
    pub fps: f32,
    /// Frame dimensions
    pub dimensions: (u32, u32),
    /// Color space
    pub color_space: ColorSpace,
    /// Video duration in seconds
    pub duration: f32,
}

/// Individual video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Frame index
    pub index: usize,
    /// Timestamp in seconds
    pub timestamp: f32,
    /// Raw pixel data (flattened)
    pub pixel_data: Vec<u8>,
    /// Frame metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Video frame representation (alias for compatibility)
pub type VideoFrame = Frame;

impl VideoData {
    /// Create new video data
    pub fn new(fps: f32, dimensions: (u32, u32), color_space: ColorSpace) -> Self {
        Self {
            frames: Vec::new(),
            fps,
            dimensions,
            color_space,
            duration: 0.0,
        }
    }

    /// Add a frame to the video
    pub fn add_frame(&mut self, frame: Frame) {
        self.frames.push(frame);
        self.duration = self.frames.len() as f32 / self.fps;
    }

    /// Get frame by index
    pub fn get_frame(&self, index: usize) -> Option<&Frame> {
        self.frames.get(index)
    }

    /// Get number of frames
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get frame at specific time
    pub fn get_frame_at_time(&self, time: f32) -> Option<&Frame> {
        let frame_index = (time * self.fps) as usize;
        self.get_frame(frame_index)
    }
}

impl Frame {
    /// Create new frame
    pub fn new(index: usize, timestamp: f32, pixel_data: Vec<u8>) -> Self {
        Self {
            index,
            timestamp,
            pixel_data,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to frame
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            resolution: (640, 480),
            color_space: ColorSpace::RGB,
            face_detection: FaceDetectionConfig::default(),
            lip_extraction: LipExtractionConfig::default(),
            temporal_smoothing: TemporalSmoothingConfig::default(),
        }
    }
}

impl Default for FaceDetectionConfig {
    fn default() -> Self {
        Self {
            model_type: FaceDetectionModel::MediaPipe,
            confidence_threshold: 0.7,
            max_faces: 1,
            min_face_size: 50,
            nms_threshold: 0.3,
        }
    }
}

impl Default for LipExtractionConfig {
    fn default() -> Self {
        Self {
            model: LipDetectionModel::FacialLandmarks,
            roi_expansion: 1.2,
            landmark_precision: LandmarkPrecision::Basic68,
            contour_smoothing: true,
            normalization: LipNormalization::default(),
        }
    }
}

impl Default for LipNormalization {
    fn default() -> Self {
        Self {
            size_normalization: true,
            position_normalization: true,
            orientation_normalization: false,
            target_size: (64, 64),
        }
    }
}

impl Default for TemporalSmoothingConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            method: SmoothingMethod::MovingAverage,
            outlier_threshold: 2.0,
            motion_compensation: true,
        }
    }
}