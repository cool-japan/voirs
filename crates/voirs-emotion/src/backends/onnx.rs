//! OxiONNX backend for emotion classification.
//!
//! This module provides ONNX-based emotion classification from audio features,
//! enabling high-performance inference using pre-trained emotion recognition models.

use crate::{Emotion, Error, Result};
use oxionnx::{OptLevel, Session, Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tracing::{debug, info, warn};

/// Number of supported emotion categories
pub const NUM_EMOTIONS: usize = 7;

/// Supported emotion labels in order of model output indices
pub const EMOTION_LABELS: [&str; NUM_EMOTIONS] = [
    "anger",
    "happiness",
    "sadness",
    "neutral",
    "fear",
    "surprise",
    "disgust",
];

/// Configuration for the ONNX emotion classifier
#[derive(Debug, Clone)]
pub struct OnnxEmotionClassifierConfig {
    /// Path to the ONNX model file
    pub model_path: PathBuf,

    /// ONNX graph optimization level
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse
    pub enable_memory_pool: bool,

    /// Number of mel frequency bins (default: 80)
    pub mel_bins: usize,

    /// Minimum confidence threshold for classification results
    pub confidence_threshold: f32,

    /// Sample rate expected by the model (for audio input preprocessing)
    pub sample_rate: u32,
}

impl Default for OnnxEmotionClassifierConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: true,
            mel_bins: 80,
            confidence_threshold: 0.1,
            sample_rate: 16000,
        }
    }
}

/// Result of emotion classification
#[derive(Debug, Clone)]
pub struct EmotionClassificationResult {
    /// The most likely emotion
    pub primary_emotion: String,

    /// Confidence score for the primary emotion (0.0 - 1.0)
    pub primary_confidence: f32,

    /// All emotion probabilities mapped by label
    pub emotion_probabilities: HashMap<String, f32>,

    /// Raw logits from the model output
    pub raw_logits: Vec<f32>,
}

/// Information about the loaded ONNX emotion model
#[derive(Debug, Clone)]
pub struct EmotionModelInfo {
    /// Model file path
    pub model_path: PathBuf,

    /// Input tensor names
    pub input_names: Vec<String>,

    /// Output tensor names
    pub output_names: Vec<String>,

    /// Number of emotion categories
    pub num_emotions: usize,

    /// Number of mel frequency bins
    pub mel_bins: usize,
}

/// ONNX-based emotion classifier
///
/// Classifies emotions from mel spectrogram features using an ONNX model.
/// Input shape: `[1, mel_bins, T]` where T is the time dimension.
/// Output shape: `[1, num_emotions]` with logits for each emotion category.
pub struct OnnxEmotionClassifier {
    /// OxiONNX inference session
    session: Arc<RwLock<Session>>,

    /// Classifier configuration
    config: OnnxEmotionClassifierConfig,

    /// Model information
    model_info: EmotionModelInfo,
}

impl OnnxEmotionClassifier {
    /// Create a new ONNX emotion classifier from the given configuration.
    ///
    /// Loads the ONNX model and prepares it for inference.
    pub fn new(config: OnnxEmotionClassifierConfig) -> Result<Self> {
        info!(
            "Initializing OxiONNX emotion classifier from {:?}",
            config.model_path
        );

        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        let session = builder
            .load(&config.model_path)
            .map_err(|e| Error::Processing(format!("Failed to load ONNX emotion model: {}", e)))?;

        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        debug!("Emotion model inputs: {:?}", input_names);
        debug!("Emotion model outputs: {:?}", output_names);

        let model_info = EmotionModelInfo {
            model_path: config.model_path.clone(),
            input_names,
            output_names,
            num_emotions: NUM_EMOTIONS,
            mel_bins: config.mel_bins,
        };

        info!("OxiONNX emotion classifier loaded successfully");

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
            model_info,
        })
    }

    /// Load a classifier from a model file path with default configuration.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let config = OnnxEmotionClassifierConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Classify emotion from a mel spectrogram.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram data with shape `[mel_bins, T]` stored in row-major order.
    /// * `time_steps` - Number of time frames in the spectrogram.
    ///
    /// # Returns
    /// Classification result with emotion probabilities.
    pub fn classify(&self, mel: &[f32], time_steps: usize) -> Result<EmotionClassificationResult> {
        let mel_bins = self.config.mel_bins;
        let expected_len = mel_bins * time_steps;
        if mel.len() != expected_len {
            return Err(Error::Processing(format!(
                "Mel spectrogram size mismatch: expected {} ({}x{}), got {}",
                expected_len,
                mel_bins,
                time_steps,
                mel.len()
            )));
        }

        // Build input tensor with shape [1, mel_bins, T]
        let input_tensor = Tensor::new(mel.to_vec(), vec![1, mel_bins, time_steps]);

        let input_name = self
            .model_info
            .input_names
            .first()
            .ok_or_else(|| Error::Processing("Model has no input names".to_string()))?;

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(input_name.clone(), input_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Run inference
        let session = self.session.read().map_err(|e| {
            Error::Processing(format!("Failed to acquire session read lock: {}", e))
        })?;

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::Processing(format!("ONNX inference failed: {}", e)))?;

        // Extract logits from first output
        let output_name = self
            .model_info
            .output_names
            .first()
            .ok_or_else(|| Error::Processing("Model has no output names".to_string()))?;

        let output_tensor = outputs.get(output_name).ok_or_else(|| {
            Error::Processing(format!("Output '{}' not found in results", output_name))
        })?;

        self.build_classification_result(&output_tensor.data)
    }

    /// Classify emotion from raw audio samples.
    ///
    /// This method computes a simplified mel spectrogram from the audio samples
    /// before running classification. For production use, prefer [`OnnxEmotionClassifier::classify`] with
    /// a properly computed mel spectrogram.
    ///
    /// # Arguments
    /// * `samples` - Raw audio samples in f32 format, expected at the configured sample rate.
    pub fn classify_audio(&self, samples: &[f32]) -> Result<EmotionClassificationResult> {
        if samples.is_empty() {
            return Err(Error::Processing("Cannot classify empty audio".to_string()));
        }

        let mel = self.compute_simple_mel(samples);
        let time_steps = mel.len() / self.config.mel_bins;
        if time_steps == 0 {
            return Err(Error::Processing(
                "Audio too short to compute mel spectrogram".to_string(),
            ));
        }
        self.classify(&mel, time_steps)
    }

    /// Return information about the loaded model.
    pub fn model_info(&self) -> &EmotionModelInfo {
        &self.model_info
    }

    /// Return the classifier configuration.
    pub fn config(&self) -> &OnnxEmotionClassifierConfig {
        &self.config
    }

    // ---- private helpers ----

    /// Build a classification result from raw logits by applying softmax.
    fn build_classification_result(
        &self,
        raw_logits: &[f32],
    ) -> Result<EmotionClassificationResult> {
        let num_emotions = raw_logits.len().min(NUM_EMOTIONS);
        if num_emotions == 0 {
            return Err(Error::Processing("Empty logits from model".to_string()));
        }

        // Stable softmax
        let max_logit = raw_logits[..num_emotions]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_values: Vec<f32> = raw_logits[..num_emotions]
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        let sum_exp: f32 = exp_values.iter().sum();
        let probabilities: Vec<f32> = if sum_exp > 0.0 {
            exp_values.iter().map(|&v| v / sum_exp).collect()
        } else {
            vec![1.0 / num_emotions as f32; num_emotions]
        };

        let mut emotion_probabilities = HashMap::with_capacity(num_emotions);
        let mut primary_idx = 0;
        let mut primary_prob = 0.0_f32;

        for (i, &prob) in probabilities.iter().enumerate() {
            let label = if i < EMOTION_LABELS.len() {
                EMOTION_LABELS[i].to_string()
            } else {
                format!("emotion_{}", i)
            };
            emotion_probabilities.insert(label, prob);
            if prob > primary_prob {
                primary_prob = prob;
                primary_idx = i;
            }
        }

        let primary_emotion = if primary_idx < EMOTION_LABELS.len() {
            EMOTION_LABELS[primary_idx].to_string()
        } else {
            format!("emotion_{}", primary_idx)
        };

        Ok(EmotionClassificationResult {
            primary_emotion,
            primary_confidence: primary_prob,
            emotion_probabilities,
            raw_logits: raw_logits.to_vec(),
        })
    }

    /// Compute a simplified mel-like spectrogram from raw audio.
    ///
    /// This is a basic energy-band approximation for convenience; callers doing
    /// production inference should use a proper mel-spectrogram implementation
    /// (e.g. via `scirs2-fft`).
    fn compute_simple_mel(&self, samples: &[f32]) -> Vec<f32> {
        let mel_bins = self.config.mel_bins;
        let hop_size = 256;
        let frame_size = 1024;

        let num_frames = if samples.len() >= frame_size {
            (samples.len() - frame_size) / hop_size + 1
        } else {
            1
        };

        let mut mel = vec![0.0_f32; mel_bins * num_frames];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + frame_size).min(samples.len());
            let frame = &samples[start..end];

            // Simplified energy per mel bin
            let bin_width = frame.len().max(1) / mel_bins.max(1);
            let bin_width = bin_width.max(1);

            for bin in 0..mel_bins {
                let bin_start = bin * bin_width;
                let bin_end = ((bin + 1) * bin_width).min(frame.len());
                if bin_start < frame.len() {
                    let energy: f32 = frame[bin_start..bin_end].iter().map(|&s| s * s).sum();
                    // Log-scale with floor to avoid log(0)
                    mel[bin * num_frames + frame_idx] = (energy + 1e-10).ln();
                }
            }
        }

        mel
    }
}

/// Map an emotion label string to the corresponding [`Emotion`] enum variant.
pub fn label_to_emotion(label: &str) -> Option<Emotion> {
    match label {
        "anger" => Some(Emotion::Angry),
        "happiness" => Some(Emotion::Happy),
        "sadness" => Some(Emotion::Sad),
        "neutral" => Some(Emotion::Neutral),
        "fear" => Some(Emotion::Fear),
        "surprise" => Some(Emotion::Surprise),
        "disgust" => Some(Emotion::Disgust),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_labels_count() {
        assert_eq!(EMOTION_LABELS.len(), NUM_EMOTIONS);
    }

    #[test]
    fn test_default_config() {
        let config = OnnxEmotionClassifierConfig::default();
        assert_eq!(config.mel_bins, 80);
        assert_eq!(config.sample_rate, 16000);
        assert!(config.enable_memory_pool);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_label_to_emotion() {
        assert!(matches!(label_to_emotion("anger"), Some(Emotion::Angry)));
        assert!(matches!(
            label_to_emotion("happiness"),
            Some(Emotion::Happy)
        ));
        assert!(matches!(
            label_to_emotion("neutral"),
            Some(Emotion::Neutral)
        ));
        assert!(label_to_emotion("unknown_label").is_none());
    }

    #[test]
    fn test_softmax_classification() {
        let config = OnnxEmotionClassifierConfig::default();
        let model_info = EmotionModelInfo {
            model_path: PathBuf::from("test.onnx"),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            num_emotions: NUM_EMOTIONS,
            mel_bins: 80,
        };

        // Test the math manually
        let logits = vec![2.0_f32, 1.0, 0.5, 0.1, -0.5, -1.0, -2.0];
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();

        // First element should have highest probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);

        // Probabilities should sum to ~1.0
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);

        // Verify model_info fields
        assert_eq!(model_info.num_emotions, 7);
        assert_eq!(model_info.mel_bins, 80);
        let _ = config;
    }
}
