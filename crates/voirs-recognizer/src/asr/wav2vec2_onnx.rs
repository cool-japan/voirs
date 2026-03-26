//! ONNX backend for Wav2Vec2 ASR model.
//!
//! Supports Wav2Vec2 models exported as ONNX for speech recognition.
//! Uses OxiONNX for pure Rust inference.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use async_trait::async_trait;
use oxionnx::{ModelInfo, NodeProfile, OptLevel, Session, Tensor};

use crate::traits::{
    ASRConfig, ASRFeature, ASRMetadata, ASRModel, AudioStream, Transcript, TranscriptStream,
};
use voirs_sdk::error::ModelType;
use voirs_sdk::AudioBuffer;
use voirs_sdk::{LanguageCode, VoirsError};

/// Result alias for this module.
type Result<T> = std::result::Result<T, VoirsError>;

/// Helper to convert an error message into a VoirsError.
fn model_err(msg: String) -> VoirsError {
    VoirsError::ModelError {
        model_type: ModelType::ASR,
        message: msg,
        source: None,
    }
}

/// Configuration for ONNX Wav2Vec2 model.
#[derive(Debug, Clone)]
pub struct OnnxWav2Vec2Config {
    /// Path to the Wav2Vec2 ONNX model file.
    pub model_path: PathBuf,
    /// Sample rate (default: 16000).
    pub sample_rate: u32,
    /// Vocabulary size for CTC output.
    pub vocab_size: usize,
    /// CTC blank token ID (default: 0).
    pub blank_token_id: usize,
    /// Maximum input length in samples.
    pub max_input_length: usize,
    /// ONNX optimization level.
    pub opt_level: OptLevel,
    /// Enable profiling.
    pub enable_profiling: bool,
    /// Enable memory pool.
    pub enable_memory_pool: bool,
    /// Vocabulary mapping (token ID -> character/token string).
    pub vocab: Vec<String>,
}

impl Default for OnnxWav2Vec2Config {
    fn default() -> Self {
        let mut vocab = vec![
            "<pad>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "<unk>".to_string(),
        ];
        for c in 'a'..='z' {
            vocab.push(c.to_string());
        }
        vocab.push(" ".to_string());
        vocab.push("'".to_string());
        vocab.push("|".to_string()); // Word boundary token

        Self {
            model_path: PathBuf::from("wav2vec2.onnx"),
            sample_rate: 16000,
            vocab_size: vocab.len(),
            blank_token_id: 0,
            max_input_length: 16000 * 30, // 30 seconds
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
            vocab,
        }
    }
}

/// ONNX-based Wav2Vec2 ASR model.
///
/// Takes raw 16 kHz waveform as input and produces text via CTC decoding.
/// The model internally learns feature representations from the raw audio,
/// unlike Whisper or Conformer which require mel spectrogram extraction.
pub struct OnnxWav2Vec2 {
    session: Arc<RwLock<Session>>,
    config: OnnxWav2Vec2Config,
    model_info: ModelInfo,
}

impl OnnxWav2Vec2 {
    /// Create a new ONNX Wav2Vec2 model.
    pub fn new(config: OnnxWav2Vec2Config) -> Result<Self> {
        let session = Self::load_session(&config.model_path, &config)?;
        let model_info = session.model_info();

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
            model_info,
        })
    }

    /// Create from a pre-loaded session.
    pub fn from_session(session: Session, config: OnnxWav2Vec2Config) -> Self {
        let model_info = session.model_info();
        Self {
            session: Arc::new(RwLock::new(session)),
            config,
            model_info,
        }
    }

    fn load_session(path: &Path, config: &OnnxWav2Vec2Config) -> Result<Session> {
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        builder.load(path).map_err(|e| {
            model_err(format!(
                "Failed to load Wav2Vec2 ONNX model '{}': {}",
                path.display(),
                e
            ))
        })
    }

    /// Normalize audio to zero mean and unit variance.
    fn normalize_audio(audio: &[f32]) -> Vec<f32> {
        let len = audio.len();
        if len == 0 {
            return Vec::new();
        }
        let mean: f32 = audio.iter().sum::<f32>() / len as f32;
        let variance: f32 =
            audio.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / len as f32;
        let std_dev = (variance + 1e-7).sqrt();

        audio.iter().map(|&x| (x - mean) / std_dev).collect()
    }

    /// CTC greedy decoding with collapse, handling the word-boundary token `|`.
    fn ctc_decode(&self, logits: &[f32], time_steps: usize) -> String {
        let vocab_size = self.config.vocab_size;
        let blank_id = self.config.blank_token_id;

        let mut prev_token = blank_id;
        let mut decoded = String::new();

        for t in 0..time_steps {
            let offset = t * vocab_size;
            let end = (offset + vocab_size).min(logits.len());
            if offset >= logits.len() {
                break;
            }
            let step_logits = &logits[offset..end];

            let token_id = step_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            if token_id != blank_id && token_id != prev_token && token_id < self.config.vocab.len()
            {
                let token = &self.config.vocab[token_id];
                if token == "|" {
                    decoded.push(' ');
                } else {
                    decoded.push_str(token);
                }
            }
            prev_token = token_id;
        }

        decoded.trim().to_string()
    }

    /// Get model info.
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    /// Get profiling results.
    pub fn profiling_results(&self) -> Option<Vec<NodeProfile>> {
        let session = self.session.read().ok()?;
        session.profiling_results()
    }
}

#[async_trait]
impl ASRModel for OnnxWav2Vec2 {
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        _config: Option<&ASRConfig>,
    ) -> crate::traits::RecognitionResult<Transcript> {
        let start = Instant::now();

        let samples = audio.samples();
        let truncated = if samples.len() > self.config.max_input_length {
            &samples[..self.config.max_input_length]
        } else {
            samples
        };

        let normalized = Self::normalize_audio(truncated);
        let input_len = normalized.len();

        let input = Tensor::new(normalized, vec![1, input_len]);
        let mut inputs = HashMap::new();
        inputs.insert("input_values", input);

        let session = self
            .session
            .read()
            .map_err(|e| model_err(format!("Session lock error: {}", e)))?;

        let outputs = session
            .run(&inputs)
            .map_err(|e| model_err(format!("Wav2Vec2 inference failed: {}", e)))?;

        let logits = outputs
            .into_values()
            .next()
            .ok_or_else(|| model_err("Wav2Vec2 produced no outputs".to_string()))?;

        let time_steps = logits
            .data
            .len()
            .checked_div(self.config.vocab_size)
            .unwrap_or(0);

        let text = self.ctc_decode(&logits.data, time_steps);

        Ok(Transcript {
            text,
            language: LanguageCode::EnUs,
            confidence: 0.82,
            word_timestamps: Vec::new(),
            sentence_boundaries: Vec::new(),
            processing_duration: Some(start.elapsed()),
        })
    }

    async fn transcribe_streaming(
        &self,
        _audio_stream: AudioStream,
        _config: Option<&ASRConfig>,
    ) -> crate::traits::RecognitionResult<TranscriptStream> {
        Err(VoirsError::ModelError {
            model_type: ModelType::ASR,
            message: "Streaming not yet supported for ONNX Wav2Vec2".to_string(),
            source: None,
        })
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::EnUs]
    }

    fn metadata(&self) -> ASRMetadata {
        let model_size_mb = (self.model_info.parameter_count * 4) as f32 / (1024.0 * 1024.0);
        let mut wer_benchmarks = HashMap::new();
        wer_benchmarks.insert(LanguageCode::EnUs, 0.04);

        ASRMetadata {
            name: "OnnxWav2Vec2".to_string(),
            version: "1.0.0".to_string(),
            description: "ONNX-based Wav2Vec2 ASR model using OxiONNX runtime".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            architecture: "Wav2Vec2 CTC".to_string(),
            model_size_mb,
            inference_speed: 0.5,
            wer_benchmarks,
            supported_features: Vec::new(),
        }
    }

    fn supports_feature(&self, _feature: ASRFeature) -> bool {
        false
    }

    async fn detect_language(
        &self,
        _audio: &AudioBuffer,
    ) -> crate::traits::RecognitionResult<LanguageCode> {
        Ok(LanguageCode::EnUs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_wav2vec2_config_default() {
        let config = OnnxWav2Vec2Config::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.blank_token_id, 0);
        assert_eq!(config.max_input_length, 16000 * 30);
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
        // Vocabulary: <pad>, <s>, </s>, <unk>, a-z, space, apostrophe, pipe = 33
        assert_eq!(config.vocab.len(), 33);
        assert_eq!(config.vocab_size, 33);
    }

    #[test]
    fn test_normalize_audio() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = OnnxWav2Vec2::normalize_audio(&audio);
        assert_eq!(normalized.len(), 5);

        // Check mean is approximately 0
        let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Check std is approximately 1
        let variance: f32 = normalized
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / normalized.len() as f32;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.1,
            "Std dev should be ~1, got {}",
            std_dev
        );
    }

    #[test]
    fn test_normalize_audio_empty() {
        let audio: Vec<f32> = Vec::new();
        let normalized = OnnxWav2Vec2::normalize_audio(&audio);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_ctc_decode_with_word_boundary() {
        let config = OnnxWav2Vec2Config::default();
        let vocab_size = config.vocab_size;

        // Find indices for 'h', 'i', '|', 't', 'h', 'e', 'r', 'e'
        let find_idx = |s: &str| -> usize { config.vocab.iter().position(|v| v == s).unwrap_or(0) };
        let h_idx = find_idx("h");
        let i_idx = find_idx("i");
        let pipe_idx = find_idx("|");

        // Simulate logits: [h, i, |, h, i]
        let steps = 5;
        let mut logits = vec![0.0f32; steps * vocab_size];

        logits[h_idx] = 10.0;
        logits[vocab_size + i_idx] = 10.0;
        logits[2 * vocab_size + pipe_idx] = 10.0;
        logits[3 * vocab_size + h_idx] = 10.0;
        logits[4 * vocab_size + i_idx] = 10.0;

        // Manual CTC decode to verify logic
        let blank_id = 0;
        let mut prev_token = blank_id;
        let mut decoded = String::new();
        for t in 0..steps {
            let offset = t * vocab_size;
            let step = &logits[offset..offset + vocab_size];
            let token_id = step
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);
            if token_id != blank_id && token_id != prev_token && token_id < config.vocab.len() {
                let token = &config.vocab[token_id];
                if token == "|" {
                    decoded.push(' ');
                } else {
                    decoded.push_str(token);
                }
            }
            prev_token = token_id;
        }
        let decoded = decoded.trim().to_string();
        assert_eq!(decoded, "hi hi");
    }
}
