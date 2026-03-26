//! Synthesis configuration for Python bindings

use super::common::*;

/// Python synthesis configuration
#[pyclass]
#[derive(Clone)]
pub struct PySynthesisConfig {
    #[pyo3(get, set)]
    pub speaking_rate: f32,
    #[pyo3(get, set)]
    pub pitch_shift: f32,
    #[pyo3(get, set)]
    pub volume_gain: f32,
    #[pyo3(get, set)]
    pub enable_enhancement: bool,
    #[pyo3(get, set)]
    pub output_format: String,
    #[pyo3(get, set)]
    pub sample_rate: u32,
    #[pyo3(get, set)]
    pub quality: String,
}

#[pymethods]
impl PySynthesisConfig {
    #[new]
    fn new() -> Self {
        let config = voirs_sdk::types::SynthesisConfig::default();
        Self {
            speaking_rate: config.speaking_rate,
            pitch_shift: config.pitch_shift,
            volume_gain: config.volume_gain,
            enable_enhancement: config.enable_enhancement,
            output_format: format!("{:?}", config.output_format),
            sample_rate: config.sample_rate,
            quality: format!("{:?}", config.quality),
        }
    }
}

// ========================================================================
// Speech Recognition Classes (ASR)
// ========================================================================

#[cfg(feature = "recognition")]
mod recognition_bindings {
    use super::*;
    use crate::python::audio_buffer::PyAudioBuffer;
    use voirs_recognizer::{
        analysis::AudioAnalyzerImpl, asr::ASRBackend,
        performance::PerformanceMetrics as RecogMetrics, prelude::*, traits::AlignedPhoneme,
        RecognitionError,
    };

    /// Python ASR Model wrapper for speech recognition
    #[pyclass]
    pub struct PyASRModel {
        #[allow(dead_code)]
        inner: Box<dyn ASRModel + Send + Sync>,
        rt: Runtime,
    }

    #[pymethods]
    impl PyASRModel {
        /// Create a new ASR model with Whisper
        #[staticmethod]
        fn whisper(model_size: Option<&str>, device: Option<&str>) -> PyResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            #[cfg(feature = "whisper-pure")]
            {
                use voirs_recognizer::asr::WhisperModelSize;

                let size = match model_size.unwrap_or("base") {
                    "tiny" => WhisperModelSize::Tiny,
                    "base" => WhisperModelSize::Base,
                    "small" => WhisperModelSize::Small,
                    "medium" => WhisperModelSize::Medium,
                    "large" => WhisperModelSize::Large,
                    _ => WhisperModelSize::Base,
                };

                let model = rt
                    .block_on(async {
                        voirs_recognizer::asr::PureRustWhisper::new_from_model_size(size).await
                    })
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to load Whisper model: {}", e))
                    })?;

                Ok(Self {
                    inner: Box::new(model),
                    rt,
                })
            }
            #[cfg(not(feature = "whisper-pure"))]
            {
                Err(PyRuntimeError::new_err("Whisper support not compiled in"))
            }
        }

        /// Recognize speech from audio buffer
        fn recognize(&self, audio: &PyAudioBuffer) -> PyResult<PyRecognitionResult> {
            let audio_buffer = audio.inner();
            let config = ASRConfig::default();

            let result = self
                .rt
                .block_on(async { self.inner.transcribe(audio_buffer, Some(&config)).await })
                .map_err(|e| PyRuntimeError::new_err(format!("Recognition failed: {}", e)))?;

            Ok(PyRecognitionResult::new(result))
        }

        /// Recognize speech from audio file
        #[staticmethod]
        fn recognize_file(
            file_path: &str,
            model_size: Option<&str>,
        ) -> PyResult<PyRecognitionResult> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                // Load audio file
                let audio = voirs_recognizer::audio_formats::load_audio(file_path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load audio: {}", e)))?;

                // Create ASR model
                #[cfg(feature = "whisper-pure")]
                {
                    use voirs_recognizer::asr::WhisperModelSize;

                    let size = match model_size.unwrap_or("base") {
                        "tiny" => WhisperModelSize::Tiny,
                        "base" => WhisperModelSize::Base,
                        "small" => WhisperModelSize::Small,
                        "medium" => WhisperModelSize::Medium,
                        "large" => WhisperModelSize::Large,
                        _ => WhisperModelSize::Base,
                    };

                    let model = voirs_recognizer::asr::PureRustWhisper::new_from_model_size(size)
                        .await
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to load model: {}", e))
                        })?;

                    // Perform recognition
                    let config = ASRConfig::default();
                    let result = model.transcribe(&audio, Some(&config)).await.map_err(|e| {
                        PyRuntimeError::new_err(format!("Recognition failed: {}", e))
                    })?;

                    Ok(PyRecognitionResult::new(result))
                }
                #[cfg(not(feature = "whisper-pure"))]
                {
                    Err(PyRuntimeError::new_err("Whisper support not compiled in"))
                }
            })
        }

        /// Get supported languages
        fn supported_languages(&self) -> Vec<String> {
            // Return common language codes
            vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "it".to_string(),
                "pt".to_string(),
                "ru".to_string(),
                "ja".to_string(),
                "ko".to_string(),
                "zh".to_string(),
                "ar".to_string(),
                "hi".to_string(),
            ]
        }
    }

    /// Python Audio Analyzer wrapper
    #[pyclass]
    pub struct PyAudioAnalyzer {
        inner: AudioAnalyzerImpl,
        rt: Runtime,
    }

    #[pymethods]
    impl PyAudioAnalyzer {
        #[new]
        fn new() -> PyResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            let config = AudioAnalysisConfig::default();
            let analyzer = rt.block_on(AudioAnalyzerImpl::new(config)).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e))
            })?;

            Ok(Self {
                inner: analyzer,
                rt,
            })
        }

        /// Analyze audio buffer
        fn analyze(&self, audio: &PyAudioBuffer) -> PyResult<PyAudioAnalysis> {
            let config = AudioAnalysisConfig::default();
            let analysis = self
                .rt
                .block_on(self.inner.analyze(audio.inner(), Some(&config)))
                .map_err(|e| PyRuntimeError::new_err(format!("Analysis failed: {}", e)))?;

            Ok(PyAudioAnalysis::new(analysis))
        }

        /// Analyze audio file
        #[staticmethod]
        fn analyze_file(file_path: &str) -> PyResult<PyAudioAnalysis> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                // Load audio file
                let audio = voirs_recognizer::audio_formats::load_audio(file_path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load audio: {}", e)))?;

                // Create analyzer
                let config = AudioAnalysisConfig::default();
                let analyzer = AudioAnalyzerImpl::new(config).await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e))
                })?;

                // Perform analysis
                let analysis = analyzer
                    .analyze(&audio, Some(&AudioAnalysisConfig::default()))
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Analysis failed: {}", e)))?;

                Ok(PyAudioAnalysis::new(analysis))
            })
        }
    }

    /// Python Phoneme Recognizer wrapper
    #[pyclass]
    pub struct PyPhonemeRecognizer {
        #[allow(dead_code)]
        inner: Arc<dyn PhonemeRecognizer>,
        rt: Runtime,
        language: LanguageCode,
    }

    #[pymethods]
    impl PyPhonemeRecognizer {
        #[new]
        fn new(language: &str) -> PyResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            // Parse language code
            let lang_code = match language.to_lowercase().as_str() {
                "en" | "en-us" => LanguageCode::EnUs,
                "es" | "es-es" => LanguageCode::EsEs,
                "fr" | "fr-fr" => LanguageCode::FrFr,
                "de" | "de-de" => LanguageCode::DeDe,
                "it" | "it-it" => LanguageCode::ItIt,
                "pt" | "pt-br" => LanguageCode::PtBr,
                "ru" | "ru-ru" => LanguageCode::RuRu,
                "ja" | "ja-jp" => LanguageCode::JaJp,
                "ko" | "ko-kr" => LanguageCode::KoKr,
                "zh" | "zh-cn" => LanguageCode::ZhCn,
                _ => LanguageCode::EnUs, // Default fallback
            };

            let backend = voirs_recognizer::phoneme::PhonemeRecognizerBackend::ForcedAlign {
                model_path: format!("models/phoneme/{}.bin", language.to_lowercase()),
                dictionary_path: None,
            };

            let recognizer = rt
                .block_on(async {
                    voirs_recognizer::phoneme::create_phoneme_recognizer(backend).await
                })
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create phoneme recognizer: {}", e))
                })?;

            Ok(Self {
                inner: recognizer,
                rt,
                language: lang_code,
            })
        }

        /// Recognize phonemes from audio
        fn recognize(
            &self,
            audio: &PyAudioBuffer,
            text: &str,
        ) -> PyResult<Vec<PyPhonemeAlignment>> {
            let config = PhonemeRecognitionConfig {
                language: self.language,
                ..Default::default()
            };

            let alignment = self
                .rt
                .block_on(async {
                    self.inner
                        .align_text(audio.inner(), text, Some(&config))
                        .await
                })
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Phoneme recognition failed: {}", e))
                })?;

            // Extract individual aligned phonemes from the alignment result
            Ok(alignment
                .phonemes
                .into_iter()
                .map(PyPhonemeAlignment::new)
                .collect())
        }
    }

    /// Python Recognition Result wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyRecognitionResult {
        #[pyo3(get)]
        pub transcript: PyTranscript,
        #[pyo3(get)]
        pub confidence: f32,
        #[pyo3(get)]
        pub processing_time_ms: f64,
    }

    #[pymethods]
    impl PyRecognitionResult {
        fn __str__(&self) -> String {
            format!(
                "RecognitionResult(text='{}', confidence={:.2})",
                self.transcript.text, self.confidence
            )
        }
    }

    impl PyRecognitionResult {
        fn new(transcript: Transcript) -> Self {
            Self {
                transcript: PyTranscript::new(transcript.clone()),
                confidence: transcript.confidence,
                processing_time_ms: transcript
                    .processing_duration
                    .map(|d| d.as_millis() as f64)
                    .unwrap_or(0.0),
            }
        }
    }

    /// Python Transcript wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyTranscript {
        #[pyo3(get)]
        pub text: String,
        #[pyo3(get)]
        pub language: String,
        #[pyo3(get)]
        pub confidence: f32,
        #[pyo3(get)]
        pub word_count: usize,
    }

    #[pymethods]
    impl PyTranscript {
        fn __str__(&self) -> String {
            self.text.clone()
        }

        fn __len__(&self) -> usize {
            self.text.len()
        }
    }

    impl PyTranscript {
        fn new(transcript: Transcript) -> Self {
            Self {
                text: transcript.text.clone(),
                language: transcript.language.to_string(),
                confidence: transcript.confidence,
                word_count: transcript.text.split_whitespace().count(),
            }
        }
    }

    /// Python Audio Analysis wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyAudioAnalysis {
        #[pyo3(get)]
        pub duration_seconds: f32,
        #[pyo3(get)]
        pub sample_rate: u32,
        #[pyo3(get)]
        pub channels: u32,
        #[pyo3(get)]
        pub rms_energy: f32,
        #[pyo3(get)]
        pub zero_crossing_rate: f32,
        #[pyo3(get)]
        pub spectral_centroid: f32,
    }

    #[pymethods]
    impl PyAudioAnalysis {
        fn __str__(&self) -> String {
            format!(
                "AudioAnalysis(duration={:.2}s, sample_rate={}Hz, channels={}, rms={:.3})",
                self.duration_seconds, self.sample_rate, self.channels, self.rms_energy
            )
        }
    }

    impl PyAudioAnalysis {
        fn new(analysis: AudioAnalysis) -> Self {
            Self {
                duration_seconds: analysis
                    .processing_duration
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0),
                sample_rate: 22050, // Default sample rate - would need to be passed from audio buffer
                channels: 1,        // Default mono - would need to be passed from audio buffer
                rms_energy: analysis
                    .quality_metrics
                    .get("rms_energy")
                    .copied()
                    .unwrap_or(0.0),
                zero_crossing_rate: analysis
                    .quality_metrics
                    .get("zero_crossing_rate")
                    .copied()
                    .unwrap_or(0.0),
                spectral_centroid: analysis
                    .quality_metrics
                    .get("spectral_centroid")
                    .copied()
                    .unwrap_or(0.0),
            }
        }
    }

    /// Python Phoneme Alignment wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyPhonemeAlignment {
        #[pyo3(get)]
        pub phoneme: String,
        #[pyo3(get)]
        pub start_time: f32,
        #[pyo3(get)]
        pub end_time: f32,
        #[pyo3(get)]
        pub confidence: f32,
    }

    #[pymethods]
    impl PyPhonemeAlignment {
        fn __str__(&self) -> String {
            format!(
                "PhonemeAlignment({}, {:.2}-{:.2}s, conf={:.2})",
                self.phoneme, self.start_time, self.end_time, self.confidence
            )
        }
    }

    impl PyPhonemeAlignment {
        fn new(aligned_phoneme: AlignedPhoneme) -> Self {
            Self {
                phoneme: aligned_phoneme.phoneme.symbol.clone(),
                start_time: aligned_phoneme.start_time,
                end_time: aligned_phoneme.end_time,
                confidence: aligned_phoneme.confidence,
            }
        }
    }

    /// Python Performance Metrics wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyPerformanceMetrics {
        #[pyo3(get)]
        pub real_time_factor: f32,
        #[pyo3(get)]
        pub memory_usage_mb: f32,
        #[pyo3(get)]
        pub processing_time_ms: f64,
        #[pyo3(get)]
        pub throughput_ratio: f32,
    }

    #[pymethods]
    impl PyPerformanceMetrics {
        fn __str__(&self) -> String {
            format!(
                "PerformanceMetrics(rtf={:.2}, memory={:.1}MB, time={:.1}ms)",
                self.real_time_factor, self.memory_usage_mb, self.processing_time_ms
            )
        }
    }

    impl PyPerformanceMetrics {
        fn new(metrics: RecogMetrics) -> Self {
            Self {
                real_time_factor: metrics.rtf,
                memory_usage_mb: metrics.memory_usage as f32 / 1_000_000.0,
                processing_time_ms: metrics.startup_time_ms as f64, // Use startup time as approximation
                throughput_ratio: metrics.throughput_samples_per_sec as f32,
            }
        }
    }

    // Re-export for main module with aliases to avoid conflicts
    pub use {
        self::PyASRModel as RecognitionASRModel, self::PyAudioAnalysis as RecognitionAudioAnalysis,
        self::PyAudioAnalyzer as RecognitionAudioAnalyzer,
        self::PyPerformanceMetrics as RecognitionPerformanceMetrics,
        self::PyPhonemeAlignment as RecognitionPhonemeAlignment,
        self::PyPhonemeRecognizer as RecognitionPhonemeRecognizer,
        self::PyRecognitionResult as RecognitionResult,
        self::PyTranscript as RecognitionTranscript,
    };
}

#[cfg(feature = "recognition")]
use recognition_bindings::*;
