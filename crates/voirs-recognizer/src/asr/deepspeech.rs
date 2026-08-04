//! Mozilla `DeepSpeech` ASR model container.
//!
//! `DeepSpeech` ships its acoustic model as a TensorFlow graph (`.pb` / `.pbmm`) or a
//! TensorFlow Lite flatbuffer (`.tflite`), and its language model as a KenLM-backed
//! `.scorer` package. Executing either of those formats requires the TensorFlow /
//! KenLM C++ runtimes, which VoiRS deliberately does not link (pure-Rust policy).
//!
//! This module therefore does the part it *can* do honestly:
//!
//! * it opens and parses the supplied model and scorer files for real, reporting their
//!   detected container format and real on-disk size, and
//! * it refuses to transcribe, returning [`RecognitionError::FeatureNotSupported`],
//!   because no pure-Rust `DeepSpeech` acoustic decoder exists in this crate.
//!
//! For a working local ASR backend, use the ONNX backends
//! ([`crate::asr::wav2vec2_onnx::OnnxWav2Vec2`], [`crate::asr::whisper_onnx::OnnxWhisper`]),
//! which run real exported weights through `OxiONNX`.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Container format detected for a `DeepSpeech` model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSpeechModelFormat {
    /// TensorFlow Lite flatbuffer (`.tflite`), identified by the `TFL3` magic at
    /// byte offset 4.
    TensorFlowLite,
    /// TensorFlow `GraphDef` protobuf (`.pb` / memory-mapped `.pbmm`).
    TensorFlowGraphDef,
    /// Content did not match any known `DeepSpeech` container.
    Unknown,
}

impl DeepSpeechModelFormat {
    /// Human-readable name of the detected container format.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TensorFlowLite => "TensorFlow Lite (.tflite)",
            Self::TensorFlowGraphDef => "TensorFlow GraphDef (.pb/.pbmm)",
            Self::Unknown => "unknown",
        }
    }
}

/// Read the leading bytes of a file, returning fewer than `max` bytes at EOF.
fn read_file_header(path: &Path, max: usize) -> Result<Vec<u8>, RecognitionError> {
    let mut file = std::fs::File::open(path).map_err(|e| RecognitionError::ModelLoadError {
        message: format!("Failed to open {}: {e}", path.display()),
        source: Some(Box::new(e)),
    })?;
    let mut buf = vec![0_u8; max];
    let mut filled = 0;
    while filled < max {
        let n = file
            .read(&mut buf[filled..])
            .map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Failed to read {}: {e}", path.display()),
                source: Some(Box::new(e)),
            })?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(buf)
}

/// Detect the container format of a `DeepSpeech` acoustic model from its real bytes.
///
/// TFLite flatbuffers carry the ASCII identifier `TFL3` at offset 4. TensorFlow
/// `GraphDef` files are protobufs whose first field is `node` (field 1, wire type 2),
/// i.e. a leading `0x0A` byte.
fn detect_model_format(header: &[u8]) -> DeepSpeechModelFormat {
    if header.len() >= 8 && &header[4..8] == b"TFL3" {
        return DeepSpeechModelFormat::TensorFlowLite;
    }
    if header.first() == Some(&0x0A) {
        return DeepSpeechModelFormat::TensorFlowGraphDef;
    }
    DeepSpeechModelFormat::Unknown
}

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
    /// Container format detected by reading the model file
    model_format: DeepSpeechModelFormat,
    /// Real on-disk size of the model file in bytes
    model_size_bytes: u64,
    /// Real on-disk size of the scorer file in bytes (if a scorer was supplied)
    scorer_size_bytes: Option<u64>,
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
            model_format: DeepSpeechModelFormat::Unknown,
            model_size_bytes: 0,
            scorer_size_bytes: None,
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
            description: "Mozilla DeepSpeech model container. Inference is unavailable: the \
                          acoustic model is a TensorFlow graph and VoiRS links no TensorFlow \
                          runtime. Use an ONNX backend instead."
                .to_string(),
            supported_languages: supported_languages.clone(),
            architecture: "RNN".to_string(),
            // Real on-disk size of the supplied model file.
            model_size_mb: Self::estimate_model_size(&model_path),
            // 0.0 == no measured value; this backend never runs inference.
            inference_speed: 0.0,
            wer_benchmarks: Self::create_wer_benchmarks(),
            // Inference is unavailable, so no inference-time feature is advertised.
            supported_features: Vec::new(),
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

    /// Inspect the model (and scorer) files for real.
    ///
    /// This performs genuine I/O: it stats each file for its true size and reads the
    /// leading bytes to detect the container format. It does **not** build a runnable
    /// decoder — see the module documentation for why.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] if a file cannot be read, or if the
    /// model file is empty or is not a recognised `DeepSpeech` container.
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if state.loaded {
            return Ok(());
        }

        let start_time = Instant::now();
        let model_path = Path::new(&state.model_path).to_path_buf();

        tracing::info!("Inspecting DeepSpeech model: {}", state.model_path);

        let model_size = std::fs::metadata(&model_path)
            .map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Failed to stat model file {}: {e}", model_path.display()),
                source: Some(Box::new(e)),
            })?
            .len();

        if model_size == 0 {
            return Err(RecognitionError::ModelLoadError {
                message: format!("DeepSpeech model file is empty: {}", model_path.display()),
                source: None,
            });
        }

        let header = read_file_header(&model_path, 16)?;
        let format = detect_model_format(&header);
        if format == DeepSpeechModelFormat::Unknown {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{} is not a recognised DeepSpeech model container \
                     (expected a TensorFlow Lite flatbuffer or a TensorFlow GraphDef protobuf)",
                    model_path.display()
                ),
                source: None,
            });
        }

        let scorer_size = match state.scorer_path.clone() {
            Some(scorer_path) => {
                let path = Path::new(&scorer_path).to_path_buf();
                let size = std::fs::metadata(&path)
                    .map_err(|e| RecognitionError::ModelLoadError {
                        message: format!("Failed to stat scorer file {}: {e}", path.display()),
                        source: Some(Box::new(e)),
                    })?
                    .len();
                tracing::info!(
                    "Inspected DeepSpeech scorer {} ({size} bytes)",
                    path.display()
                );
                Some(size)
            }
            None => None,
        };

        state.model_format = format;
        state.model_size_bytes = model_size;
        state.scorer_size_bytes = scorer_size;
        state.loaded = true;
        state.load_time = Some(start_time.elapsed());

        tracing::info!(
            "DeepSpeech model inspected in {:?}: format={}, size={} bytes",
            start_time.elapsed(),
            format.as_str(),
            model_size
        );

        Ok(())
    }

    /// Real on-disk model size in MB, or `0.0` when the file cannot be stat'ed.
    fn estimate_model_size(model_path: &str) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        std::fs::metadata(model_path)
            .map(|metadata| metadata.len() as f32 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }

    /// Word Error Rate benchmarks.
    ///
    /// Empty: this backend cannot run inference, so VoiRS has no measured WER for it.
    /// Publishing a number here would be a fabrication.
    fn create_wer_benchmarks() -> HashMap<LanguageCode, f32> {
        HashMap::new()
    }

    /// Report the container format detected when the model file was inspected.
    ///
    /// # Errors
    /// Propagates any error from inspecting the model file.
    pub async fn model_format(&self) -> Result<DeepSpeechModelFormat, RecognitionError> {
        self.ensure_loaded().await?;
        Ok(self.state.read().await.model_format)
    }

    /// Build the typed error returned by every inference entry point.
    fn unsupported_backend_error(format: DeepSpeechModelFormat) -> RecognitionError {
        RecognitionError::FeatureNotSupported {
            feature: format!(
                "DeepSpeech inference: the acoustic model is a {} graph, which requires the \
                 TensorFlow runtime; VoiRS links no C/C++ runtimes. Use an ONNX backend \
                 (OnnxWhisper / OnnxWav2Vec2, `onnx` feature) with exported weights instead.",
                format.as_str()
            ),
        }
    }

    /// Validate the request against the inspected model, then fail closed.
    ///
    /// Audio preprocessing, sample-rate validation and language support checks are all
    /// performed for real so that callers get precise diagnostics, but the final step
    /// returns [`RecognitionError::FeatureNotSupported`] rather than a fabricated
    /// transcript.
    async fn process_audio(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<Transcript, RecognitionError> {
        self.ensure_loaded().await?;

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

        // The audio is valid and the model file was really inspected, but no pure-Rust
        // DeepSpeech acoustic decoder exists. Fail closed instead of inventing text.
        let format = self.state.read().await.model_format;
        tracing::warn!(
            "DeepSpeech transcription requested for {:.3}s of {language:?} audio, but no \
             pure-Rust DeepSpeech decoder is available",
            processed_audio.duration()
        );
        Err(Self::unsupported_backend_error(format))
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

    /// Set a custom decoding vocabulary.
    ///
    /// # Errors
    /// Always returns [`RecognitionError::FeatureNotSupported`]: the vocabulary is
    /// consumed by the KenLM scorer inside the `DeepSpeech` CTC beam decoder, which this
    /// crate cannot execute. Reporting success would imply a decoder configuration that
    /// does not exist.
    pub async fn set_custom_vocabulary(&self, words: Vec<String>) -> Result<(), RecognitionError> {
        self.ensure_loaded().await?;
        tracing::warn!(
            "Rejecting custom vocabulary of {} words: no DeepSpeech decoder to apply it to",
            words.len()
        );
        Err(Self::unsupported_backend_error(
            self.state.read().await.model_format,
        ))
    }

    /// Set the language-model interpolation parameters (`alpha`, `beta`).
    ///
    /// # Errors
    /// Always returns [`RecognitionError::FeatureNotSupported`], for the same reason as
    /// [`Self::set_custom_vocabulary`].
    pub async fn set_lm_params(&self, alpha: f32, beta: f32) -> Result<(), RecognitionError> {
        self.ensure_loaded().await?;
        tracing::warn!(
            "Rejecting LM parameters alpha={alpha}, beta={beta}: no DeepSpeech decoder to apply \
             them to"
        );
        Err(Self::unsupported_backend_error(
            self.state.read().await.model_format,
        ))
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

    /// Streaming transcription.
    ///
    /// # Errors
    /// Always returns [`RecognitionError::FeatureNotSupported`]: streaming would only
    /// chunk audio into the unavailable batch decoder, so it fails at the same point.
    async fn transcribe_streaming(
        &self,
        _audio_stream: AudioStream,
        _config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream> {
        self.ensure_loaded().await?;
        Err(Self::unsupported_backend_error(self.state.read().await.model_format).into())
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
    /// Always returns [`RecognitionError::FeatureNotSupported`]. `DeepSpeech` has no
    /// language-identification head, and this backend cannot run the acoustic model at
    /// all, so there is nothing to detect *from*. Previously this returned
    /// `Ok(LanguageCode::EnUs)` unconditionally, which reported a "detection" that never
    /// looked at the audio.
    async fn detect_language(&self, _audio: &AudioBuffer) -> RecognitionResult<LanguageCode> {
        Err(RecognitionError::FeatureNotSupported {
            feature: "DeepSpeech language detection: the model has no language-identification \
                      head and this backend cannot execute the acoustic model"
                .to_string(),
        }
        .into())
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

    /// Write a minimal file whose bytes really look like a TFLite flatbuffer
    /// (4-byte root-table offset followed by the `TFL3` file identifier).
    fn create_tflite_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let mut bytes = vec![0x18, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(b"TFL3");
        bytes.extend_from_slice(&[0_u8; 64]);
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    /// Write a file whose leading byte matches a TensorFlow `GraphDef` protobuf
    /// (field 1, wire type 2 => tag byte 0x0A).
    fn create_graphdef_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let mut bytes = vec![0x0A, 0x04];
        bytes.extend_from_slice(b"node");
        bytes.extend_from_slice(&[0_u8; 32]);
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    fn create_garbage_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "definitely not a model").unwrap();
        file.flush().unwrap();
        file
    }

    async fn tflite_model() -> (NamedTempFile, DeepSpeechModel) {
        let model_file = create_tflite_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();
        let model = DeepSpeechModel::new(model_path, None).await.unwrap();
        (model_file, model)
    }

    #[test]
    fn test_detect_model_format_reads_real_magic() {
        let mut tflite = vec![0x18, 0x00, 0x00, 0x00];
        tflite.extend_from_slice(b"TFL3");
        assert_eq!(
            detect_model_format(&tflite),
            DeepSpeechModelFormat::TensorFlowLite
        );
        assert_eq!(
            detect_model_format(&[0x0A, 0x04, b'n', b'o', b'd', b'e']),
            DeepSpeechModelFormat::TensorFlowGraphDef
        );
        assert_eq!(
            detect_model_format(b"definitely not a model"),
            DeepSpeechModelFormat::Unknown
        );
        assert_eq!(detect_model_format(&[]), DeepSpeechModelFormat::Unknown);
    }

    #[tokio::test]
    async fn test_deepspeech_model_creation() {
        let (_guard, model) = tflite_model().await;
        assert_eq!(model.metadata.name, "Mozilla DeepSpeech");
        assert!(model.supported_languages().contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_deepspeech_missing_file() {
        let result = DeepSpeechModel::new("nonexistent.pb".to_string(), None).await;
        assert!(result.is_err());
    }

    /// Regression test for the removed `mock_inference`: transcription must fail
    /// closed with a typed error instead of returning a canned English sentence.
    #[tokio::test]
    async fn test_deepspeech_transcribe_fails_closed() {
        let (_guard, model) = tflite_model().await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let err = model
            .process_audio(&audio, None)
            .await
            .expect_err("DeepSpeech inference must not fabricate a transcript");
        match err {
            RecognitionError::FeatureNotSupported { feature } => {
                assert!(
                    feature.contains("DeepSpeech inference"),
                    "unexpected error text: {feature}"
                );
                assert!(
                    feature.contains("TensorFlow Lite"),
                    "format should be reported: {feature}"
                );
            }
            other => panic!("expected FeatureNotSupported, got {other:?}"),
        }

        // The trait entry point must fail closed too.
        assert!(model.transcribe(&audio, None).await.is_err());
    }

    /// The old code returned the same canned string for every input. Verify no
    /// input at all can produce an `Ok(Transcript)`.
    #[tokio::test]
    async fn test_deepspeech_never_returns_ok_transcript() {
        let (_guard, model) = tflite_model().await;
        for len in [16_usize, 1600, 16000] {
            let samples: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.01).sin()).collect();
            let audio = AudioBuffer::new(samples, 16000, 1);
            assert!(
                model.transcribe(&audio, None).await.is_err(),
                "transcribe returned Ok for {len} samples"
            );
        }
    }

    #[tokio::test]
    async fn test_deepspeech_rejects_unrecognised_container() {
        let model_file = create_garbage_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();
        let model = DeepSpeechModel::new(model_path, None).await.unwrap();

        // `new` only checks existence; the real read happens on first use.
        let err = model
            .model_format()
            .await
            .expect_err("garbage bytes must be rejected");
        assert!(matches!(err, RecognitionError::ModelLoadError { .. }));
    }

    #[tokio::test]
    async fn test_deepspeech_detects_graphdef_container() {
        let model_file = create_graphdef_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();
        let model = DeepSpeechModel::new(model_path, None).await.unwrap();

        assert_eq!(
            model.model_format().await.unwrap(),
            DeepSpeechModelFormat::TensorFlowGraphDef
        );
    }

    #[tokio::test]
    async fn test_deepspeech_unsupported_language() {
        let (_guard, model) = tflite_model().await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let config = ASRConfig {
            language: Some(LanguageCode::JaJp), // Not supported by DeepSpeech
            ..Default::default()
        };

        let result = model.transcribe(&audio, Some(&config)).await;
        assert!(result.is_err());
    }

    /// The metadata must not advertise inference-time features or a WER benchmark
    /// for a backend that cannot run inference.
    #[tokio::test]
    async fn test_deepspeech_metadata_is_honest() {
        let (guard, model) = tflite_model().await;
        let metadata = model.metadata();

        assert!(
            metadata.wer_benchmarks.is_empty(),
            "WER must not be fabricated"
        );
        assert!(metadata.supported_features.is_empty());
        assert!(!model.supports_feature(ASRFeature::WordTimestamps));
        assert!(!model.supports_feature(ASRFeature::StreamingInference));
        assert!(!model.supports_feature(ASRFeature::CustomVocabulary));
        assert!(!model.supports_feature(ASRFeature::LanguageDetection));

        // model_size_mb is derived from the file that really exists on disk.
        let real_bytes = std::fs::metadata(guard.path()).unwrap().len();
        #[allow(clippy::cast_precision_loss)]
        let expected_mb = real_bytes as f32 / (1024.0 * 1024.0);
        assert!((metadata.model_size_mb - expected_mb).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_deepspeech_custom_vocabulary_fails_closed() {
        let (_guard, model) = tflite_model().await;
        let custom_words = vec!["tensorflow".to_string(), "pytorch".to_string()];

        let result = model.set_custom_vocabulary(custom_words).await;
        assert!(matches!(
            result,
            Err(RecognitionError::FeatureNotSupported { .. })
        ));
    }

    #[tokio::test]
    async fn test_deepspeech_lm_params_fails_closed() {
        let (_guard, model) = tflite_model().await;

        let result = model.set_lm_params(0.8, 2.0).await;
        assert!(matches!(
            result,
            Err(RecognitionError::FeatureNotSupported { .. })
        ));
    }

    #[tokio::test]
    async fn test_deepspeech_streaming_fails_closed() {
        use futures::stream;

        let (_guard, model) = tflite_model().await;
        let audio_stream: AudioStream = Box::pin(stream::iter(vec![AudioBuffer::new(
            vec![0.0; 160],
            16000,
            1,
        )]));

        assert!(model
            .transcribe_streaming(audio_stream, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_deepspeech_detect_language_fails_closed() {
        let (_guard, model) = tflite_model().await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        assert!(model.detect_language(&audio).await.is_err());
    }

    /// Inference never succeeds, so the inference counter must stay at zero
    /// rather than counting fabricated runs. Load time, by contrast, is real.
    #[tokio::test]
    async fn test_deepspeech_stats() {
        let (_guard, model) = tflite_model().await;
        let audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 16000, 1);

        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 0);

        assert!(model.transcribe(&audio, None).await.is_err());
        let stats = model.get_stats().await;
        assert_eq!(stats.inference_count, 0);
        assert_eq!(stats.total_inference_time, Duration::ZERO);
        assert!(
            stats.load_time.is_some(),
            "file inspection time must be recorded"
        );
    }
}
