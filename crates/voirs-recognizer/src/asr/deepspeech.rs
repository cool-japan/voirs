//! Mozilla DeepSpeech ASR implementation
//!
//! This module provides integration with Mozilla's DeepSpeech model for automatic speech recognition.
//! DeepSpeech is primarily focused on English but can be trained for other languages.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Mozilla DeepSpeech ASR model implementation
pub struct DeepSpeechModel {
    /// Model configuration
    config: DeepSpeechConfig,
    /// Model state
    state: Arc<RwLock<DeepSpeechState>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: ASRMetadata,
}

/// DeepSpeech model configuration
#[derive(Debug, Clone)]
pub struct DeepSpeechConfig {
    /// Path to the model file (.tflite or .pb)
    pub model_path: String,
    /// Path to the scorer file (optional)
    pub scorer_path: Option<String>,
    /// Beam width for decoding
    pub beam_width: usize,
    /// Language model alpha
    pub lm_alpha: f32,
    /// Language model beta
    pub lm_beta: f32,
    /// Number of threads for CPU inference
    pub num_threads: usize,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Enable intermediate results
    pub enable_intermediate_results: bool,
}

impl Default for DeepSpeechConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            scorer_path: None,
            beam_width: 100,
            lm_alpha: 0.75,
            lm_beta: 1.85,
            num_threads: num_cpus::get(),
            use_gpu: false,
            enable_intermediate_results: true,
        }
    }
}

/// Internal state for DeepSpeech model
struct DeepSpeechState {
    /// Whether the model is loaded
    loaded: bool,
    /// Model file path
    model_path: String,
    /// Scorer file path
    scorer_path: Option<String>,
    /// Loading time
    load_time: Option<Duration>,
    /// Inference count
    inference_count: usize,
    /// Total inference time
    total_inference_time: Duration,
}

impl DeepSpeechState {
    fn new(model_path: String, scorer_path: Option<String>) -> Self {
        Self {
            loaded: false,
            model_path,
            scorer_path,
            load_time: None,
            inference_count: 0,
            total_inference_time: Duration::ZERO,
        }
    }
}

impl DeepSpeechModel {
    /// Create a new DeepSpeech model instance
    pub async fn new(
        model_path: String,
        scorer_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        // Validate model file exists
        if !Path::new(&model_path).exists() {
            return Err(RecognitionError::ModelLoadError {
                message: format!("Model file not found: {}", model_path),
                source: None,
            });
        }

        // Validate scorer file if provided
        if let Some(ref scorer_path) = scorer_path {
            if !Path::new(scorer_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Scorer file not found: {}", scorer_path),
                    source: None,
                });
            }
        }

        let config = DeepSpeechConfig {
            model_path: model_path.clone(),
            scorer_path: scorer_path.clone(),
            ..Default::default()
        };

        // DeepSpeech is primarily English-focused
        let supported_languages = vec![LanguageCode::EnUs, LanguageCode::EnGb];

        let metadata = ASRMetadata {
            name: "Mozilla DeepSpeech".to_string(),
            version: "0.9.3".to_string(),
            description: "Mozilla DeepSpeech automatic speech recognition model".to_string(),
            supported_languages: supported_languages.clone(),
            architecture: "RNN".to_string(),
            model_size_mb: Self::estimate_model_size(&model_path),
            inference_speed: 1.5, // Typically slower than Whisper
            wer_benchmarks: Self::create_wer_benchmarks(),
            supported_features: vec![
                ASRFeature::WordTimestamps,
                ASRFeature::StreamingInference,
                ASRFeature::CustomVocabulary,
            ],
        };

        let state = Arc::new(RwLock::new(DeepSpeechState::new(model_path, scorer_path)));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: DeepSpeechConfig) -> Result<Self, RecognitionError> {
        Self::new(config.model_path.clone(), config.scorer_path.clone()).await
    }

    /// Load the model if not already loaded
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if !state.loaded {
            let start_time = Instant::now();

            // Load model (placeholder implementation)
            tracing::info!("Loading DeepSpeech model: {}", state.model_path);

            // Simulate loading time (DeepSpeech models are typically smaller)
            tokio::time::sleep(Duration::from_millis(50)).await;

            if let Some(ref scorer_path) = state.scorer_path {
                tracing::info!("Loading DeepSpeech scorer: {}", scorer_path);
                tokio::time::sleep(Duration::from_millis(25)).await;
            }

            state.loaded = true;
            state.load_time = Some(start_time.elapsed());

            tracing::info!("DeepSpeech model loaded in {:?}", state.load_time.unwrap());
        }

        Ok(())
    }

    /// Estimate model size from file
    fn estimate_model_size(model_path: &str) -> f32 {
        std::fs::metadata(model_path)
            .map(|metadata| metadata.len() as f32 / 1_024_000.0) // Convert to MB
            .unwrap_or(50.0) // Default estimate
    }

    /// Create WER benchmarks
    fn create_wer_benchmarks() -> HashMap<LanguageCode, f32> {
        let mut benchmarks = HashMap::new();
        benchmarks.insert(LanguageCode::EnUs, 0.065); // Slightly higher WER than Whisper
        benchmarks.insert(LanguageCode::EnGb, 0.07);
        benchmarks
    }

    /// Process audio with DeepSpeech
    async fn process_audio(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<Transcript, RecognitionError> {
        self.ensure_loaded().await?;

        let start_time = Instant::now();

        // Preprocess audio for DeepSpeech (requires 16kHz mono)
        let processed_audio = super::utils::preprocess_audio(audio).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to preprocess audio: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Validate sample rate
        if processed_audio.sample_rate() != 16000 {
            return Err(RecognitionError::AudioProcessingError {
                message: "DeepSpeech requires 16kHz sample rate".to_string(),
                source: None,
            });
        }

        // Determine language (DeepSpeech is primarily English)
        let language = if let Some(config) = config {
            config.language.unwrap_or(LanguageCode::EnUs)
        } else {
            LanguageCode::EnUs
        };

        if !self.supported_languages.contains(&language) {
            return Err(RecognitionError::FeatureNotSupported {
                feature: format!("Language: {:?}", language),
            });
        }

        // Perform inference (placeholder implementation)
        let transcript = self
            .mock_inference(&processed_audio, language, config)
            .await?;

        // Update statistics
        let mut state = self.state.write().await;
        state.inference_count += 1;
        state.total_inference_time += start_time.elapsed();

        Ok(transcript)
    }

    /// Mock inference for demonstration
    async fn mock_inference(
        &self,
        audio: &AudioBuffer,
        language: LanguageCode,
        _config: Option<&ASRConfig>,
    ) -> Result<Transcript, RecognitionError> {
        // Simulate processing time (DeepSpeech is typically slower)
        let processing_time = Duration::from_millis(
            (audio.samples().len() as f64 / audio.sample_rate() as f64 * 1000.0
                / self.metadata.inference_speed as f64) as u64,
        );
        tokio::time::sleep(processing_time).await;

        // Generate mock transcript (DeepSpeech style - often lowercase, no punctuation)
        let text = match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                "hello this is a test transcription from deep speech"
            }
            _ => "hello this is a test transcription from deep speech",
        };

        let words = text.split_whitespace().collect::<Vec<&str>>();
        let mut word_timestamps = Vec::new();
        let mut current_time = 0.0;

        for word in &words {
            let word_duration = word.len() as f32 * 0.08; // Slightly different timing than Whisper
            word_timestamps.push(WordTimestamp {
                word: word.to_string(),
                start_time: current_time,
                end_time: current_time + word_duration,
                confidence: 0.88, // Slightly lower confidence than Whisper
            });
            current_time += word_duration + 0.06; // Add pause between words
        }

        let sentence_boundaries = vec![SentenceBoundary {
            start_time: 0.0,
            end_time: current_time,
            text: text.to_string(),
            confidence: 0.88,
        }];

        Ok(Transcript {
            text: text.to_string(),
            language,
            confidence: 0.88,
            word_timestamps,
            sentence_boundaries,
            processing_duration: Some(processing_time),
        })
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> DeepSpeechStats {
        let state = self.state.read().await;
        DeepSpeechStats {
            inference_count: state.inference_count,
            total_inference_time: state.total_inference_time,
            average_inference_time: if state.inference_count > 0 {
                state.total_inference_time / state.inference_count as u32
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            model_path: state.model_path.clone(),
            scorer_path: state.scorer_path.clone(),
        }
    }

    /// Set custom vocabulary
    pub async fn set_custom_vocabulary(&self, words: Vec<String>) -> Result<(), RecognitionError> {
        self.ensure_loaded().await?;

        // In a real implementation, this would update the DeepSpeech model's vocabulary
        tracing::info!("Setting custom vocabulary with {} words", words.len());

        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(())
    }

    /// Set language model parameters
    pub async fn set_lm_params(&self, alpha: f32, beta: f32) -> Result<(), RecognitionError> {
        self.ensure_loaded().await?;

        // In a real implementation, this would update the language model parameters
        tracing::info!("Setting LM parameters: alpha={}, beta={}", alpha, beta);

        Ok(())
    }
}

/// DeepSpeech model statistics
#[derive(Debug, Clone)]
pub struct DeepSpeechStats {
    /// Total number of inferences
    pub inference_count: usize,
    /// Total inference time
    pub total_inference_time: Duration,
    /// Average inference time
    pub average_inference_time: Duration,
    /// Model load time
    pub load_time: Option<Duration>,
    /// Model file path
    pub model_path: String,
    /// Scorer file path
    pub scorer_path: Option<String>,
}

#[async_trait]
impl ASRModel for DeepSpeechModel {
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<Transcript> {
        self.process_audio(audio, config)
            .await
            .map_err(|e| e.into())
    }

    async fn transcribe_streaming(
        &self,
        mut audio_stream: AudioStream,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream> {
        // DeepSpeech supports streaming inference
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let model = self.clone();
        let config = config.cloned();

        tokio::spawn(async move {
            use futures::StreamExt;

            let mut chunk_index = 0;
            let mut accumulated_text = String::new();

            while let Some(audio_chunk) = audio_stream.next().await {
                let transcript_result = model.process_audio(&audio_chunk, config.as_ref()).await;

                match transcript_result {
                    Ok(transcript) => {
                        // For streaming DeepSpeech, we typically get partial results
                        accumulated_text.push_str(&transcript.text);
                        accumulated_text.push(' ');

                        let chunk = TranscriptChunk {
                            text: transcript.text,
                            is_final: false, // Intermediate result
                            start_time: chunk_index as f32 * 1.0,
                            end_time: (chunk_index + 1) as f32 * 1.0,
                            confidence: transcript.confidence,
                        };

                        if sender.send(Ok(chunk)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        if sender.send(Err(e.into())).is_err() {
                            break;
                        }
                    }
                }

                chunk_index += 1;
            }

            // Send final result
            let final_chunk = TranscriptChunk {
                text: accumulated_text.trim().to_string(),
                is_final: true,
                start_time: 0.0,
                end_time: chunk_index as f32 * 1.0,
                confidence: 0.88,
            };
            let _ = sender.send(Ok(final_chunk));
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(receiver),
        ))
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.supported_languages.clone()
    }

    fn metadata(&self) -> ASRMetadata {
        self.metadata.clone()
    }

    fn supports_feature(&self, feature: ASRFeature) -> bool {
        self.metadata.supported_features.contains(&feature)
    }

    async fn detect_language(&self, _audio: &AudioBuffer) -> RecognitionResult<LanguageCode> {
        // DeepSpeech doesn't support language detection - it's primarily English
        Ok(LanguageCode::EnUs)
    }
}

impl Clone for DeepSpeechModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            supported_languages: self.supported_languages.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_mock_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "mock model data").unwrap();
        file
    }

    #[tokio::test]
    async fn test_deepspeech_model_creation() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        assert_eq!(model.metadata.name, "Mozilla DeepSpeech");
        assert!(model.supported_languages().contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_deepspeech_missing_file() {
        let result = DeepSpeechModel::new("nonexistent.pb".to_string(), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deepspeech_transcribe() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let result = model.transcribe(&audio, None).await.unwrap();
        assert!(!result.text.is_empty());
        assert!(result.confidence > 0.0);
        assert_eq!(result.language, LanguageCode::EnUs);
    }

    #[tokio::test]
    async fn test_deepspeech_unsupported_language() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let config = ASRConfig {
            language: Some(LanguageCode::JaJp), // Not supported by DeepSpeech
            ..Default::default()
        };

        let result = model.transcribe(&audio, Some(&config)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deepspeech_features() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();

        assert!(model.supports_feature(ASRFeature::WordTimestamps));
        assert!(model.supports_feature(ASRFeature::StreamingInference));
        assert!(model.supports_feature(ASRFeature::CustomVocabulary));
        assert!(!model.supports_feature(ASRFeature::LanguageDetection));
    }

    #[tokio::test]
    async fn test_deepspeech_custom_vocabulary() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        let custom_words = vec!["tensorflow".to_string(), "pytorch".to_string()];

        let result = model.set_custom_vocabulary(custom_words).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deepspeech_lm_params() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();

        let result = model.set_lm_params(0.8, 2.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deepspeech_stats() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        // Initial stats
        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 0);

        // After inference
        let _result = model.transcribe(&audio, None).await.unwrap();
        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 1);
        assert!(stats.total_inference_time > Duration::ZERO);
    }
}
