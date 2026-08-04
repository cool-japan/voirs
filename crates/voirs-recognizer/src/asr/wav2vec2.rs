//! Facebook `Wav2Vec2` checkpoint container.
//!
//! This type owns and validates a real `Wav2Vec2` checkpoint but does **not** execute
//! it: this crate has no hand-written `Wav2Vec2` graph in `candle`, so there is nothing
//! to feed the parameters to. Rather than fabricate transcripts, it does the honest
//! subset of the job:
//!
//! * [`Wav2Vec2Model::ensure_loaded`] really opens the checkpoint, parses the
//!   `safetensors` header, checks that the tensors look like a `Wav2Vec2` CTC model,
//!   and records the real parameter count and file size, and
//! * every inference entry point returns [`RecognitionError::FeatureNotSupported`].
//!
//! The working `Wav2Vec2` backend is [`crate::asr::wav2vec2_onnx::OnnxWav2Vec2`]
//! (`onnx` feature), which runs a real exported graph through `OxiONNX`. Export one with
//! `optimum-cli export onnx --model facebook/wav2vec2-base-960h wav2vec2-onnx/`.

use super::weights::SafetensorsHeader;
use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Tensor-name prefixes that a Hugging Face `Wav2Vec2` CTC checkpoint must contain.
///
/// Used to tell a genuine `Wav2Vec2` checkpoint apart from an unrelated `safetensors`
/// file, so that the reported diagnostics are accurate.
const WAV2VEC2_REQUIRED_PREFIXES: &[&str] = &[
    "wav2vec2.feature_extractor",
    "wav2vec2.encoder",
    "lm_head",
];

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
    /// Real parameter count read from the checkpoint header
    parameter_count: usize,
    /// Real checkpoint size in mebibytes
    checkpoint_size_mb: f32,
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
            parameter_count: 0,
            checkpoint_size_mb: 0.0,
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
            description: "Facebook Wav2Vec2 checkpoint container. Inference is unavailable: no \
                          native Wav2Vec2 graph exists in this crate. Use OnnxWav2Vec2 instead."
                .to_string(),
            supported_languages: supported_languages.clone(),
            architecture: "Transformer".to_string(),
            // Real size of the configured checkpoint (0.0 when none is configured).
            model_size_mb: Self::checkpoint_size_mb(model_path.as_deref()),
            // 0.0 == no measured value; this backend never runs inference.
            inference_speed: 0.0,
            wer_benchmarks: Self::create_wer_benchmarks(),
            // Inference is unavailable, so no inference-time feature is advertised.
            supported_features: Vec::new(),
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

    /// Open and validate the configured checkpoint for real.
    ///
    /// Parses the `safetensors` header, verifies that the tensor names match a
    /// `Wav2Vec2` CTC model, and records the real parameter count and file size.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when no local checkpoint path is
    /// configured (VoiRS does not silently download weights), when the file cannot be
    /// parsed, or when the tensors do not belong to a `Wav2Vec2` model.
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if state.loaded {
            return Ok(());
        }

        let start_time = Instant::now();
        let Some(model_path) = state.model_path.clone() else {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "No local checkpoint configured for Wav2Vec2 model '{}'. VoiRS does not \
                     download weights implicitly: pass a path to a downloaded \
                     `model.safetensors` (or its directory) via Wav2Vec2Config::model_path.",
                    state.model_id
                ),
                source: None,
            });
        };

        tracing::info!(
            "Inspecting Wav2Vec2 checkpoint for '{}' at {model_path}",
            state.model_id
        );

        let header = SafetensorsHeader::read(&model_path)?;

        let missing: Vec<&str> = WAV2VEC2_REQUIRED_PREFIXES
            .iter()
            .copied()
            .filter(|prefix| !header.has_prefix(prefix))
            .collect();
        if !missing.is_empty() {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{} does not look like a Wav2Vec2 CTC checkpoint: no tensors under {}",
                    header.path.display(),
                    missing.join(", ")
                ),
                source: None,
            });
        }

        state.parameter_count = header.parameter_count();
        state.checkpoint_size_mb = header.size_mb();
        state.loaded = true;
        state.load_time = Some(start_time.elapsed());

        tracing::info!(
            "Wav2Vec2 checkpoint validated in {:?}: {} tensors, {} parameters, {:.1} MiB",
            start_time.elapsed(),
            header.tensors.len(),
            state.parameter_count,
            state.checkpoint_size_mb
        );

        Ok(())
    }

    /// Real parameter count read from the checkpoint header.
    ///
    /// # Errors
    /// Propagates any error from validating the checkpoint.
    pub async fn parameter_count(&self) -> Result<usize, RecognitionError> {
        self.ensure_loaded().await?;
        Ok(self.state.read().await.parameter_count)
    }

    /// The typed error returned by every inference entry point.
    fn unsupported_backend_error(model_id: &str) -> RecognitionError {
        RecognitionError::FeatureNotSupported {
            feature: format!(
                "Wav2Vec2 inference for '{model_id}': this crate has no native Wav2Vec2 graph to \
                 run the checkpoint through. Use OnnxWav2Vec2 (`onnx` feature) with a model \
                 exported via `optimum-cli export onnx`."
            ),
        }
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

    /// Real on-disk size of the configured checkpoint in mebibytes.
    ///
    /// Returns `0.0` when no local checkpoint is configured. The previous
    /// implementation guessed a size from substrings of the model id
    /// (`"base"` => 95 MB and so on), which described whatever HuggingFace hosts under
    /// that name rather than any file the caller actually has.
    fn checkpoint_size_mb(model_path: Option<&str>) -> f32 {
        let Some(path) = model_path else {
            return 0.0;
        };
        super::weights::resolve_weights_path(Path::new(path))
            .ok()
            .and_then(|resolved| std::fs::metadata(resolved).ok())
            .map_or(0.0, |metadata| {
                #[allow(clippy::cast_precision_loss)]
                {
                    metadata.len() as f32 / (1024.0 * 1024.0)
                }
            })
    }

    /// Word Error Rate benchmarks.
    ///
    /// Empty: this backend cannot run inference, so VoiRS has measured no WER for it.
    /// The previous implementation returned per-model-id constants that were never
    /// produced by any evaluation in this repository.
    fn create_wer_benchmarks() -> HashMap<LanguageCode, f32> {
        HashMap::new()
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

        // The checkpoint was really validated and the audio is really preprocessed, but
        // there is no graph to run. Fail closed rather than invent a transcript.
        tracing::warn!(
            "Wav2Vec2 transcription requested for {:.3}s of {language:?} audio, but no native \
             Wav2Vec2 graph is available",
            processed_audio.duration()
        );
        Err(Self::unsupported_backend_error(&self.config.model_id))
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

    /// Streaming transcription.
    ///
    /// # Errors
    /// Always returns [`RecognitionError::FeatureNotSupported`]: streaming only chunks
    /// audio into the unavailable batch path, so it fails for the same reason.
    async fn transcribe_streaming(
        &self,
        _audio_stream: AudioStream,
        _config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream> {
        self.ensure_loaded().await?;
        Err(Self::unsupported_backend_error(&self.config.model_id).into())
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

    /// Detect the spoken language.
    ///
    /// # Errors
    /// Always returns [`RecognitionError::FeatureNotSupported`]. `Wav2Vec2` CTC models
    /// have no language-identification head; the previous implementation slept for 30 ms
    /// and then returned a fixed language without consulting the audio at all.
    async fn detect_language(&self, _audio: &AudioBuffer) -> RecognitionResult<LanguageCode> {
        Err(RecognitionError::FeatureNotSupported {
            feature: "Wav2Vec2 language detection: CTC checkpoints carry no \
                      language-identification head. Use a dedicated language-ID model."
                .to_string(),
        }
        .into())
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
    use std::io::Write;
    use std::path::PathBuf;
    use voirs_sdk::AudioBuffer;

    /// Write a real safetensors file whose tensor names match a Wav2Vec2 CTC model.
    fn write_wav2vec2_checkpoint(dir: &Path, extra: &[&str]) -> PathBuf {
        let mut names: Vec<String> = vec![
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight".to_string(),
            "wav2vec2.encoder.layers.0.attention.q_proj.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        names.extend(extra.iter().map(|n| (*n).to_string()));

        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for name in &names {
            // 4 x 4 F32 tensor == 64 bytes.
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [4, 4],
                    "data_offsets": [offset, offset + 64],
                }),
            );
            offset += 64;
        }

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let path = dir.join("model.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&vec![0_u8; offset as usize]).unwrap();
        file.flush().unwrap();
        path
    }

    async fn model_with_checkpoint(dir: &tempfile::TempDir) -> Wav2Vec2Model {
        let path = write_wav2vec2_checkpoint(dir.path(), &[]);
        Wav2Vec2Model::new(
            "facebook/wav2vec2-base-960h".to_string(),
            Some(path.to_string_lossy().to_string()),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_wav2vec2_model_creation() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();
        assert!(model.metadata.name.contains("Wav2Vec2"));
        assert!(model.supported_languages().contains(&LanguageCode::EnUs));
    }

    /// Regression test for the removed `mock_inference`: transcription must fail
    /// closed instead of returning a canned per-language sentence.
    #[tokio::test]
    async fn test_wav2vec2_transcribe_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let err = model
            .process_audio(&audio, None)
            .await
            .expect_err("Wav2Vec2 inference must not fabricate a transcript");
        match err {
            RecognitionError::FeatureNotSupported { feature } => {
                assert!(feature.contains("Wav2Vec2 inference"), "unexpected: {feature}");
                assert!(feature.contains("OnnxWav2Vec2"), "must name the real backend: {feature}");
            }
            other => panic!("expected FeatureNotSupported, got {other:?}"),
        }
        assert!(model.transcribe(&audio, None).await.is_err());
    }

    /// The old code returned a fixed string per requested language. Verify that no
    /// language selection can produce an `Ok(Transcript)`.
    #[tokio::test]
    async fn test_wav2vec2_no_language_yields_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_wav2vec2_checkpoint(dir.path(), &[]);
        let model = Wav2Vec2Model::new(
            "facebook/wav2vec2-large-xlsr-53".to_string(),
            Some(path.to_string_lossy().to_string()),
        )
        .await
        .unwrap();
        let audio = AudioBuffer::new(vec![0.05; 1600], 16000, 1);

        for language in [
            LanguageCode::EnUs,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::JaJp,
        ] {
            let config = ASRConfig {
                language: Some(language),
                ..Default::default()
            };
            assert!(
                model.transcribe(&audio, Some(&config)).await.is_err(),
                "transcribe returned Ok for {language:?}"
            );
        }
    }

    /// Without a local checkpoint the model must say so rather than pretend to
    /// download one from HuggingFace.
    #[tokio::test]
    async fn test_wav2vec2_requires_local_checkpoint() {
        let model = Wav2Vec2Model::new("facebook/wav2vec2-base-960h".to_string(), None)
            .await
            .unwrap();

        let err = model.ensure_loaded().await.unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("No local checkpoint"), "unexpected: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    /// Loading really parses the checkpoint: an unrelated safetensors file is rejected.
    #[tokio::test]
    async fn test_wav2vec2_rejects_foreign_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let mut header = serde_json::Map::new();
        header.insert(
            "some.other.model.weight".to_string(),
            serde_json::json!({ "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] }),
        );
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let path = dir.path().join("model.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&[0_u8; 16]).unwrap();
        drop(file);

        let model = Wav2Vec2Model::new(
            "facebook/wav2vec2-base-960h".to_string(),
            Some(path.to_string_lossy().to_string()),
        )
        .await
        .unwrap();

        let err = model.ensure_loaded().await.unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("Wav2Vec2 CTC checkpoint"), "unexpected: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    /// The parameter count must come from the real header, not a per-model-id guess.
    #[tokio::test]
    async fn test_wav2vec2_parameter_count_is_real() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        // 3 tensors of shape [4, 4].
        assert_eq!(model.parameter_count().await.unwrap(), 48);

        let dir2 = tempfile::tempdir().unwrap();
        let path = write_wav2vec2_checkpoint(dir2.path(), &["wav2vec2.encoder.layers.1.k.weight"]);
        let bigger = Wav2Vec2Model::new(
            "facebook/wav2vec2-base-960h".to_string(),
            Some(path.to_string_lossy().to_string()),
        )
        .await
        .unwrap();
        assert_eq!(bigger.parameter_count().await.unwrap(), 64);
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
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let config = ASRConfig {
            language: Some(LanguageCode::JaJp), // Not supported by base model
            ..Default::default()
        };

        let result = model.transcribe(&audio, Some(&config)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wav2vec2_long_audio_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;

        // 35 seconds at 16 kHz exercises the chunking path.
        let samples = vec![0.1; 35 * 16000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        assert!(model.transcribe(&audio, None).await.is_err());
    }

    /// Metadata must not advertise inference-time features or fabricated WER numbers.
    #[tokio::test]
    async fn test_wav2vec2_metadata_is_honest() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_wav2vec2_checkpoint(dir.path(), &[]);
        let model = Wav2Vec2Model::new(
            "facebook/wav2vec2-base-960h".to_string(),
            Some(path.to_string_lossy().to_string()),
        )
        .await
        .unwrap();
        let metadata = model.metadata();

        assert!(metadata.wer_benchmarks.is_empty(), "WER must not be fabricated");
        assert!(metadata.supported_features.is_empty());
        assert!(!model.supports_feature(ASRFeature::WordTimestamps));
        assert!(!model.supports_feature(ASRFeature::NoiseRobustness));
        assert!(!model.supports_feature(ASRFeature::StreamingInference));
        assert!(!model.supports_feature(ASRFeature::LanguageDetection));

        // Size must come from the file that really exists on disk.
        let real_bytes = std::fs::metadata(&path).unwrap().len();
        #[allow(clippy::cast_precision_loss)]
        let expected_mb = real_bytes as f32 / (1024.0 * 1024.0);
        assert!((metadata.model_size_mb - expected_mb).abs() < f32::EPSILON);
    }

    /// Without a checkpoint the reported size must be 0.0, not a guess from the id.
    #[tokio::test]
    async fn test_wav2vec2_size_is_not_guessed_from_id() {
        for id in [
            "facebook/wav2vec2-base-960h",
            "facebook/wav2vec2-large-960h",
            "facebook/wav2vec2-large-xlsr-53",
        ] {
            let model = Wav2Vec2Model::new(id.to_string(), None).await.unwrap();
            assert_eq!(
                model.metadata().model_size_mb,
                0.0,
                "{id}: size must not be inferred from the model id"
            );
        }
    }

    #[tokio::test]
    async fn test_wav2vec2_streaming_fails_closed() {
        use futures::stream;

        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        let audio_stream: AudioStream =
            Box::pin(stream::iter(vec![AudioBuffer::new(vec![0.0; 160], 16000, 1)]));

        assert!(model.transcribe_streaming(audio_stream, None).await.is_err());
    }

    #[tokio::test]
    async fn test_wav2vec2_detect_language_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        assert!(model.detect_language(&audio).await.is_err());
    }

    /// Inference never succeeds, so the counter must stay at zero rather than
    /// counting fabricated runs.
    #[tokio::test]
    async fn test_wav2vec2_stats() {
        let dir = tempfile::tempdir().unwrap();
        let model = model_with_checkpoint(&dir).await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 0);

        assert!(model.transcribe(&audio, None).await.is_err());
        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 0);
        assert_eq!(stats.total_inference_time, Duration::ZERO);
        assert!(stats.load_time.is_some(), "checkpoint validation time must be recorded");
    }
}
