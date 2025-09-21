//! Facebook Wav2Vec2 ASR implementation
//!
//! This module provides integration with Facebook's Wav2Vec2 model for automatic speech recognition.
//! Wav2Vec2 is a self-supervised model that can be fine-tuned for various languages.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Facebook Wav2Vec2 ASR model implementation
pub struct Wav2Vec2Model {
    /// Model configuration
    config: Wav2Vec2Config,
    /// Model state
    state: Arc<RwLock<Wav2Vec2State>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: ASRMetadata,
}

/// Wav2Vec2 model configuration
#[derive(Debug, Clone)]
pub struct Wav2Vec2Config {
    /// Model identifier (e.g., "facebook/wav2vec2-base-960h")
    pub model_id: String,
    /// Local model path (optional, will download if not provided)
    pub model_path: Option<String>,
    /// Processor/tokenizer path
    pub processor_path: Option<String>,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Number of threads for CPU inference
    pub num_threads: usize,
    /// Attention mask for padding
    pub use_attention_mask: bool,
    /// Return attention weights
    pub output_attentions: bool,
    /// Chunk length for long audio
    pub chunk_length_s: Option<f32>,
    /// Stride length for overlapping chunks
    pub stride_length_s: Option<f32>,
}

impl Default for Wav2Vec2Config {
    fn default() -> Self {
        Self {
            model_id: "facebook/wav2vec2-base-960h".to_string(),
            model_path: None,
            processor_path: None,
            use_gpu: true,
            num_threads: num_cpus::get(),
            use_attention_mask: true,
            output_attentions: false,
            chunk_length_s: Some(30.0),
            stride_length_s: Some(5.0),
        }
    }
}

/// Internal state for Wav2Vec2 model
struct Wav2Vec2State {
    /// Whether the model is loaded
    loaded: bool,
    /// Model identifier
    model_id: String,
    /// Model file path
    model_path: Option<String>,
    /// Loading time
    load_time: Option<Duration>,
    /// Inference count
    inference_count: usize,
    /// Total inference time
    total_inference_time: Duration,
}

impl Wav2Vec2State {
    fn new(model_id: String, model_path: Option<String>) -> Self {
        Self {
            loaded: false,
            model_id,
            model_path,
            load_time: None,
            inference_count: 0,
            total_inference_time: Duration::ZERO,
        }
    }
}

impl Wav2Vec2Model {
    /// Create a new Wav2Vec2 model instance
    pub async fn new(
        model_id: String,
        model_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        // Validate model path if provided
        if let Some(ref path) = model_path {
            if !Path::new(path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Model path not found: {}", path),
                    source: None,
                });
            }
        }

        let config = Wav2Vec2Config {
            model_id: model_id.clone(),
            model_path: model_path.clone(),
            ..Default::default()
        };

        // Determine supported languages based on model ID
        let supported_languages = Self::get_supported_languages(&model_id);

        let metadata = ASRMetadata {
            name: format!("Wav2Vec2 ({})", model_id),
            version: "2.0.0".to_string(),
            description: "Facebook Wav2Vec2 self-supervised speech recognition model".to_string(),
            supported_languages: supported_languages.clone(),
            architecture: "Transformer".to_string(),
            model_size_mb: Self::estimate_model_size(&model_id),
            inference_speed: Self::estimate_inference_speed(&model_id),
            wer_benchmarks: Self::create_wer_benchmarks(&model_id),
            supported_features: vec![
                ASRFeature::WordTimestamps,
                ASRFeature::SentenceSegmentation,
                ASRFeature::NoiseRobustness,
                ASRFeature::StreamingInference,
            ],
        };

        let state = Arc::new(RwLock::new(Wav2Vec2State::new(model_id, model_path)));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: Wav2Vec2Config) -> Result<Self, RecognitionError> {
        Self::new(config.model_id.clone(), config.model_path.clone()).await
    }

    /// Load the model if not already loaded
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if !state.loaded {
            let start_time = Instant::now();

            // Load model (placeholder implementation)
            tracing::info!("Loading Wav2Vec2 model: {}", state.model_id);

            if state.model_path.is_none() {
                tracing::info!("Downloading model from HuggingFace Hub...");
                // Simulate download time
                tokio::time::sleep(Duration::from_millis(200)).await;
            }

            // Simulate loading time
            tokio::time::sleep(Duration::from_millis(150)).await;

            state.loaded = true;
            state.load_time = Some(start_time.elapsed());

            tracing::info!("Wav2Vec2 model loaded in {:?}", state.load_time.unwrap());
        }

        Ok(())
    }

    /// Get supported languages based on model ID
    fn get_supported_languages(model_id: &str) -> Vec<LanguageCode> {
        match model_id {
            id if id.contains("960h") => vec![LanguageCode::EnUs, LanguageCode::EnGb],
            id if id.contains("xlsr") => vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            id if id.contains("large") => vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
            ],
            _ => vec![LanguageCode::EnUs], // Default to English
        }
    }

    /// Estimate model size based on model ID
    fn estimate_model_size(model_id: &str) -> f32 {
        match model_id {
            id if id.contains("base") => 95.0,
            id if id.contains("large") => 315.0,
            id if id.contains("xlsr") && id.contains("300m") => 300.0,
            id if id.contains("xlsr") && id.contains("1b") => 1000.0,
            id if id.contains("xlsr") && id.contains("2b") => 2000.0,
            _ => 95.0, // Default to base size
        }
    }

    /// Estimate inference speed based on model ID
    fn estimate_inference_speed(model_id: &str) -> f32 {
        match model_id {
            id if id.contains("base") => 2.5,
            id if id.contains("large") => 1.0,
            id if id.contains("xlsr") && id.contains("300m") => 1.2,
            id if id.contains("xlsr") && id.contains("1b") => 0.6,
            id if id.contains("xlsr") && id.contains("2b") => 0.3,
            _ => 2.5, // Default to base speed
        }
    }

    /// Create WER benchmarks based on model ID
    fn create_wer_benchmarks(model_id: &str) -> HashMap<LanguageCode, f32> {
        let mut benchmarks = HashMap::new();

        match model_id {
            id if id.contains("base") => {
                benchmarks.insert(LanguageCode::EnUs, 0.055);
                benchmarks.insert(LanguageCode::EnGb, 0.06);
            }
            id if id.contains("large") => {
                benchmarks.insert(LanguageCode::EnUs, 0.035);
                benchmarks.insert(LanguageCode::EnGb, 0.04);
                benchmarks.insert(LanguageCode::DeDe, 0.05);
                benchmarks.insert(LanguageCode::FrFr, 0.045);
                benchmarks.insert(LanguageCode::EsEs, 0.045);
            }
            id if id.contains("xlsr") => {
                benchmarks.insert(LanguageCode::EnUs, 0.04);
                benchmarks.insert(LanguageCode::EnGb, 0.045);
                benchmarks.insert(LanguageCode::DeDe, 0.055);
                benchmarks.insert(LanguageCode::FrFr, 0.05);
                benchmarks.insert(LanguageCode::EsEs, 0.05);
                benchmarks.insert(LanguageCode::JaJp, 0.08);
                benchmarks.insert(LanguageCode::ZhCn, 0.09);
                benchmarks.insert(LanguageCode::KoKr, 0.085);
            }
            _ => {
                benchmarks.insert(LanguageCode::EnUs, 0.055);
            }
        }

        benchmarks
    }

    /// Process audio with Wav2Vec2
    async fn process_audio(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<Transcript, RecognitionError> {
        self.ensure_loaded().await?;

        let start_time = Instant::now();

        // Preprocess audio (Wav2Vec2 typically uses 16kHz)
        let processed_audio = super::utils::preprocess_audio(audio).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to preprocess audio: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Determine language
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
        // Simulate processing time
        let processing_time = Duration::from_millis(
            (audio.samples().len() as f64 / audio.sample_rate() as f64 * 1000.0
                / self.metadata.inference_speed as f64) as u64,
        );
        tokio::time::sleep(processing_time).await;

        // Generate mock transcript (Wav2Vec2 style - good punctuation and capitalization)
        let text = match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                "Hello, this is a test transcription from Wav2Vec2."
            }
            LanguageCode::DeDe => "Hallo, das ist eine Testtranscription von Wav2Vec2.",
            LanguageCode::FrFr => "Bonjour, ceci est une transcription test de Wav2Vec2.",
            LanguageCode::EsEs => "Hola, esta es una transcripción de prueba de Wav2Vec2.",
            LanguageCode::JaJp => "こんにちは、これはWav2Vec2からのテスト転写です。",
            LanguageCode::ZhCn => "你好，这是来自Wav2Vec2的测试转录。",
            LanguageCode::KoKr => "안녕하세요, 이것은 Wav2Vec2의 테스트 전사입니다.",
            _ => "Hello, this is a test transcription from Wav2Vec2.",
        };

        let words = text.split_whitespace().collect::<Vec<&str>>();
        let mut word_timestamps = Vec::new();
        let mut current_time = 0.0;

        for word in &words {
            let word_duration = word.len() as f32 * 0.09; // Wav2Vec2 timing
            word_timestamps.push(WordTimestamp {
                word: word.to_string(),
                start_time: current_time,
                end_time: current_time + word_duration,
                confidence: 0.92, // Good confidence
            });
            current_time += word_duration + 0.04; // Add pause between words
        }

        let sentence_boundaries = vec![SentenceBoundary {
            start_time: 0.0,
            end_time: current_time,
            text: text.to_string(),
            confidence: 0.92,
        }];

        Ok(Transcript {
            text: text.to_string(),
            language,
            confidence: 0.92,
            word_timestamps,
            sentence_boundaries,
            processing_duration: Some(processing_time),
        })
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> Wav2Vec2Stats {
        let state = self.state.read().await;
        Wav2Vec2Stats {
            inference_count: state.inference_count,
            total_inference_time: state.total_inference_time,
            average_inference_time: if state.inference_count > 0 {
                state.total_inference_time / state.inference_count as u32
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            model_id: state.model_id.clone(),
            model_path: state.model_path.clone(),
        }
    }

    /// Process long audio by chunking
    pub async fn process_long_audio(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<Transcript, RecognitionError> {
        let chunk_length = self.config.chunk_length_s.unwrap_or(30.0);
        let stride_length = self.config.stride_length_s.unwrap_or(5.0);

        // Split audio into overlapping chunks
        let chunk_samples = (audio.sample_rate() as f32 * chunk_length) as usize;
        let stride_samples = (audio.sample_rate() as f32 * stride_length) as usize;

        let samples = audio.samples();
        let mut transcripts = Vec::new();
        let mut start_idx = 0;

        while start_idx < samples.len() {
            let end_idx = (start_idx + chunk_samples).min(samples.len());
            let chunk_slice = &samples[start_idx..end_idx];
            let chunk_audio =
                AudioBuffer::new(chunk_slice.to_vec(), audio.sample_rate(), audio.channels());

            let transcript = self.process_audio(&chunk_audio, config).await?;
            transcripts.push(transcript);

            if end_idx >= samples.len() {
                break;
            }

            start_idx += chunk_samples - stride_samples;
        }

        // Merge transcripts
        Ok(crate::merge_transcripts(&transcripts))
    }
}

/// Wav2Vec2 model statistics
#[derive(Debug, Clone)]
pub struct Wav2Vec2Stats {
    /// Total number of inferences
    pub inference_count: usize,
    /// Total inference time
    pub total_inference_time: Duration,
    /// Average inference time
    pub average_inference_time: Duration,
    /// Model load time
    pub load_time: Option<Duration>,
    /// Model identifier
    pub model_id: String,
    /// Model file path
    pub model_path: Option<String>,
}

#[async_trait]
impl ASRModel for Wav2Vec2Model {
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<Transcript> {
        // For long audio, use chunking
        let audio_duration = audio.samples().len() as f32 / audio.sample_rate() as f32;
        let max_duration = self.config.chunk_length_s.unwrap_or(30.0);

        if audio_duration > max_duration {
            self.process_long_audio(audio, config).await
        } else {
            self.process_audio(audio, config).await
        }
        .map_err(|e| e.into())
    }

    async fn transcribe_streaming(
        &self,
        mut audio_stream: AudioStream,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let model = self.clone();
        let config = config.cloned();

        tokio::spawn(async move {
            use futures::StreamExt;

            let mut chunk_index = 0;
            while let Some(audio_chunk) = audio_stream.next().await {
                let transcript_result = model.process_audio(&audio_chunk, config.as_ref()).await;

                match transcript_result {
                    Ok(transcript) => {
                        let chunk = TranscriptChunk {
                            text: transcript.text,
                            is_final: true, // Wav2Vec2 typically produces complete results
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

    async fn detect_language(&self, audio: &AudioBuffer) -> RecognitionResult<LanguageCode> {
        // Wav2Vec2 models typically don't support built-in language detection
        // We would need a separate language ID model
        self.ensure_loaded().await?;

        let _ = super::utils::preprocess_audio(audio).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to preprocess audio for language detection: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(30)).await;

        // For multilingual models, we could try to detect the language
        if self.config.model_id.contains("xlsr") {
            // Mock detection based on audio characteristics
            // In reality, this would require a language identification model
            Ok(LanguageCode::EnUs)
        } else {
            // Single-language models just return their primary language
            Ok(self.supported_languages[0])
        }
    }
}

impl Clone for Wav2Vec2Model {
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
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_wav2vec2_model_creation() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();
        assert!(model.metadata.name.contains("Wav2Vec2"));
        assert!(model.supported_languages().contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_wav2vec2_transcribe() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let result = model.transcribe(&audio, None).await.unwrap();
        assert!(!result.text.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_wav2vec2_multilingual() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-large-xlsr-53".to_string(), None)
            .await
            .unwrap();

        // Should support multiple languages
        let supported = model.supported_languages();
        assert!(supported.len() > 2);
        assert!(supported.contains(&LanguageCode::EnUs));
        assert!(supported.contains(&LanguageCode::DeDe));
        assert!(supported.contains(&LanguageCode::FrFr));
    }

    #[tokio::test]
    async fn test_wav2vec2_unsupported_language() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let config = ASRConfig {
            language: Some(LanguageCode::JaJp), // Not supported by base model
            ..Default::default()
        };

        let result = model.transcribe(&audio, Some(&config)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wav2vec2_long_audio() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();

        // Create long audio (35 seconds at 16kHz)
        let samples = vec![0.1; 35 * 16000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let result = model.transcribe(&audio, None).await.unwrap();
        assert!(!result.text.is_empty());
    }

    #[tokio::test]
    async fn test_wav2vec2_features() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();

        assert!(model.supports_feature(ASRFeature::WordTimestamps));
        assert!(model.supports_feature(ASRFeature::NoiseRobustness));
        assert!(model.supports_feature(ASRFeature::StreamingInference));
        assert!(!model.supports_feature(ASRFeature::LanguageDetection));
    }

    #[tokio::test]
    async fn test_wav2vec2_model_sizes() {
        // Test different model size estimates
        assert_eq!(
            Wav2Vec2Model::estimate_model_size("facebook/wav2vec2-base-960h"),
            95.0
        );
        assert_eq!(
            Wav2Vec2Model::estimate_model_size("facebook/wav2vec2-large-960h"),
            315.0
        );
        assert_eq!(
            Wav2Vec2Model::estimate_model_size("facebook/wav2vec2-large-xlsr-53"),
            315.0
        );
    }

    #[tokio::test]
    async fn test_wav2vec2_stats() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();
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
