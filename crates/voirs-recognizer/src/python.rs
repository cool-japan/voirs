//! Python bindings for VoiRS Recognizer
//!
//! This module provides Python bindings for the VoiRS voice recognition system
//! using PyO3. It exposes the main functionality for audio processing,
//! speech recognition, and analysis.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyType};
use pyo3::Bound;

#[cfg(feature = "python")]
use pyo3_async_runtimes::tokio::future_into_py;

use crate::prelude::*;
use crate::{PerformanceValidator, RecognitionError};

/// Python wrapper for AudioBuffer
#[pyclass]
/// Py Audio Buffer
pub struct PyAudioBuffer {
    inner: AudioBuffer,
}

#[pymethods]
impl PyAudioBuffer {
    #[new]
    /// new
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            inner: AudioBuffer::mono(samples, sample_rate),
        }
    }

    /// Create from stereo samples
    #[staticmethod]
    /// from stereo
    pub fn from_stereo(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            inner: AudioBuffer::stereo(samples, sample_rate),
        }
    }

    /// Get audio samples
    pub fn samples(&self) -> Vec<f32> {
        self.inner.samples().to_vec()
    }

    /// Get audio samples as NumPy array
    #[cfg(feature = "python")]
    /// samples numpy
    pub fn samples_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let samples = self.inner.samples();
        samples.to_pyarray(py)
    }

    /// Create PyAudioBuffer from NumPy array
    #[cfg(feature = "python")]
    #[staticmethod]
    /// from numpy
    pub fn from_numpy(samples: PyReadonlyArray1<f32>, sample_rate: u32) -> PyResult<Self> {
        let samples_vec = samples.as_array().to_vec();
        Ok(Self {
            inner: AudioBuffer::mono(samples_vec, sample_rate),
        })
    }

    /// Create stereo PyAudioBuffer from NumPy array
    #[cfg(feature = "python")]
    #[staticmethod]
    /// from numpy stereo
    pub fn from_numpy_stereo(samples: PyReadonlyArray1<f32>, sample_rate: u32) -> PyResult<Self> {
        let samples_vec = samples.as_array().to_vec();
        Ok(Self {
            inner: AudioBuffer::stereo(samples_vec, sample_rate),
        })
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.inner.duration()
    }

    /// Get number of channels
    pub fn channels(&self) -> usize {
        self.inner.channels() as usize
    }

    /// Get length in samples
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "AudioBuffer({} samples, {}Hz, {:.2}s)",
            self.inner.len(),
            self.inner.sample_rate(),
            self.inner.duration()
        )
    }
}

/// Python wrapper for ASR configuration
#[pyclass]
#[derive(Clone)]
/// Py A S R Config
pub struct PyASRConfig {
    inner: ASRConfig,
}

#[pymethods]
impl PyASRConfig {
    #[new]
    #[pyo3(signature = (language=None, word_timestamps=false, confidence_threshold=0.5, sentence_segmentation=true, language_detection=false))]
    /// new
    pub fn new(
        language: Option<String>,
        word_timestamps: bool,
        confidence_threshold: f32,
        sentence_segmentation: bool,
        language_detection: bool,
    ) -> Self {
        let lang_code = language.map(|l| match l.as_str() {
            "en" | "en-US" => LanguageCode::EnUs,
            "es" | "es-ES" => LanguageCode::EsEs,
            "fr" | "fr-FR" => LanguageCode::FrFr,
            "de" | "de-DE" => LanguageCode::DeDe,
            "it" | "it-IT" => LanguageCode::ItIt,
            "pt" | "pt-BR" => LanguageCode::PtBr,
            "ru" | "ru-RU" => LanguageCode::RuRu,
            "ja" | "ja-JP" => LanguageCode::JaJp,
            "ko" | "ko-KR" => LanguageCode::KoKr,
            "zh" | "zh-CN" => LanguageCode::ZhCn,
            _ => LanguageCode::EnUs, // Default to English
        });

        Self {
            inner: ASRConfig {
                language: lang_code,
                word_timestamps,
                sentence_segmentation,
                confidence_threshold,
                language_detection,
                ..Default::default()
            },
        }
    }

    /// Create default configuration
    #[staticmethod]
    /// default
    pub fn default() -> Self {
        Self {
            inner: ASRConfig::default(),
        }
    }

    /// Get language
    pub fn language(&self) -> Option<String> {
        self.inner.language.as_ref().map(|l| match l {
            LanguageCode::EnUs => "en-US".to_string(),
            LanguageCode::EsEs => "es-ES".to_string(),
            LanguageCode::FrFr => "fr-FR".to_string(),
            LanguageCode::DeDe => "de-DE".to_string(),
            LanguageCode::ItIt => "it-IT".to_string(),
            LanguageCode::PtBr => "pt-BR".to_string(),
            LanguageCode::RuRu => "ru-RU".to_string(),
            LanguageCode::JaJp => "ja-JP".to_string(),
            LanguageCode::KoKr => "ko-KR".to_string(),
            LanguageCode::ZhCn => "zh-CN".to_string(),
            _ => "en-US".to_string(), // Default fallback
        })
    }

    /// Get word timestamps setting
    pub fn word_timestamps(&self) -> bool {
        self.inner.word_timestamps
    }

    /// Get confidence threshold
    pub fn confidence_threshold(&self) -> f32 {
        self.inner.confidence_threshold
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "ASRConfig(language={:?}, word_timestamps={}, confidence_threshold={:.2})",
            self.language(),
            self.word_timestamps(),
            self.confidence_threshold()
        )
    }
}

/// Python wrapper for word timestamps
#[pyclass]
#[derive(Clone)]
/// Py Word Timestamp
pub struct PyWordTimestamp {
    word: String,
    start_time: f32,
    end_time: f32,
    confidence: f32,
}

#[pymethods]
impl PyWordTimestamp {
    /// Get word
    pub fn word(&self) -> &str {
        &self.word
    }

    /// Get start time
    pub fn start(&self) -> f32 {
        self.start_time
    }

    /// Get end time
    pub fn end(&self) -> f32 {
        self.end_time
    }

    /// Get confidence
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "WordTimestamp('{}', {:.2}s-{:.2}s, conf={:.2})",
            self.word, self.start_time, self.end_time, self.confidence
        )
    }
}

/// Python wrapper for recognition results
#[pyclass]
/// Py Recognition Result
pub struct PyRecognitionResult {
    text: String,
    confidence: f32,
    language: Option<String>,
    word_timestamps: Vec<PyWordTimestamp>,
    processing_time: Option<f32>,
}

#[pymethods]
impl PyRecognitionResult {
    /// Get transcribed text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get confidence score
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get detected language
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Get word timestamps
    pub fn word_timestamps(&self) -> Vec<PyWordTimestamp> {
        self.word_timestamps.clone()
    }

    /// Get processing time in seconds
    pub fn processing_time(&self) -> Option<f32> {
        self.processing_time
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "RecognitionResult('{}', confidence={:.2}, language={:?})",
            self.text, self.confidence, self.language
        )
    }
}

/// Python wrapper for VoiRS ASR system
#[pyclass]
/// Py Voi R S Recognizer
pub struct PyVoiRSRecognizer {
    asr: IntelligentASRFallback,
    rt: tokio::runtime::Runtime,
}

// Async Python wrapper for VoiRS Recognizer
// NOTE: pyo3-async-runtimes compatibility has been resolved with pyo3 0.25+
// The async implementation can be added when needed, using:
// - pyo3-async-runtimes crate for async support
// - Arc<tokio::sync::Mutex<IntelligentASRFallback>> for shared state
// - future_into_py for converting Rust futures to Python coroutines
//
// #[cfg(feature = "python")]
// #[pyclass]
// pub struct PyAsyncVoiRSRecognizer {
//     asr: Arc<tokio::sync::Mutex<IntelligentASRFallback>>,
// }
//
// Full async implementation available when needed - compatibility issue resolved!

#[pymethods]
impl PyVoiRSRecognizer {
    #[new]
    /// new
    pub fn new(config: Option<PyASRConfig>) -> PyResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let asr_config = config.map(|c| c.inner).unwrap_or_default();

        let asr = rt
            .block_on(async {
                let fallback_config = crate::asr::FallbackConfig::default();
                IntelligentASRFallback::new(fallback_config).await
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize ASR: {}", e)))?;

        Ok(Self { asr, rt })
    }

    /// Recognize speech from audio buffer
    pub fn recognize(&mut self, audio: &PyAudioBuffer) -> PyResult<PyRecognitionResult> {
        let start_time = std::time::Instant::now();

        let result = self
            .rt
            .block_on(async { self.asr.transcribe(&audio.inner, None).await })
            .map_err(|e| PyValueError::new_err(format!("Recognition failed: {}", e)))?;

        let processing_time = start_time.elapsed().as_secs_f32();

        // Convert word timestamps
        let word_timestamps = result
            .transcript
            .word_timestamps
            .into_iter()
            .map(|w| PyWordTimestamp {
                word: w.word,
                start_time: w.start_time,
                end_time: w.end_time,
                confidence: w.confidence,
            })
            .collect();

        // Convert language
        let language = match result.transcript.language {
            LanguageCode::EnUs => Some("en-US".to_string()),
            LanguageCode::EsEs => Some("es-ES".to_string()),
            LanguageCode::FrFr => Some("fr-FR".to_string()),
            LanguageCode::DeDe => Some("de-DE".to_string()),
            LanguageCode::ItIt => Some("it-IT".to_string()),
            LanguageCode::PtBr => Some("pt-BR".to_string()),
            LanguageCode::RuRu => Some("ru-RU".to_string()),
            LanguageCode::JaJp => Some("ja-JP".to_string()),
            LanguageCode::KoKr => Some("ko-KR".to_string()),
            LanguageCode::ZhCn => Some("zh-CN".to_string()),
            _ => Some("en-US".to_string()), // Default fallback
        };

        Ok(PyRecognitionResult {
            text: result.transcript.text,
            confidence: result.transcript.confidence,
            language,
            word_timestamps,
            processing_time: Some(processing_time),
        })
    }

    /// Transcribe audio (alternative method name)
    pub fn transcribe(&mut self, audio: &PyAudioBuffer) -> PyResult<PyRecognitionResult> {
        self.recognize(audio)
    }

    /// String representation
    pub fn __str__(&self) -> String {
        "VoiRSRecognizer(initialized)".to_string()
    }
}

/// Python wrapper for audio analysis configuration
#[pyclass]
#[derive(Clone)]
/// Py Audio Analysis Config
pub struct PyAudioAnalysisConfig {
    inner: AudioAnalysisConfig,
}

#[pymethods]
impl PyAudioAnalysisConfig {
    #[new]
    #[pyo3(signature = (quality_metrics=true, prosody_analysis=true, speaker_analysis=true, emotional_analysis=false))]
    /// new
    pub fn new(
        quality_metrics: bool,
        prosody_analysis: bool,
        speaker_analysis: bool,
        emotional_analysis: bool,
    ) -> Self {
        Self {
            inner: AudioAnalysisConfig {
                quality_metrics,
                prosody_analysis,
                speaker_analysis,
                emotional_analysis,
                ..Default::default()
            },
        }
    }

    /// Create default configuration
    #[staticmethod]
    /// default
    pub fn default() -> Self {
        Self {
            inner: AudioAnalysisConfig::default(),
        }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "AudioAnalysisConfig(quality={}, prosody={}, speaker={}, emotional={})",
            self.inner.quality_metrics,
            self.inner.prosody_analysis,
            self.inner.speaker_analysis,
            self.inner.emotional_analysis
        )
    }
}

/// Python wrapper for audio analysis results
#[pyclass]
/// Py Audio Analysis Result
pub struct PyAudioAnalysisResult {
    quality_metrics: HashMap<String, f32>,
    prosody_features: HashMap<String, f32>,
    speaker_features: HashMap<String, f32>,
}

#[pymethods]
impl PyAudioAnalysisResult {
    /// Get quality metrics
    pub fn quality_metrics(&self) -> HashMap<String, f32> {
        self.quality_metrics.clone()
    }

    /// Get prosody features
    pub fn prosody_features(&self) -> HashMap<String, f32> {
        self.prosody_features.clone()
    }

    /// Get speaker features
    pub fn speaker_features(&self) -> HashMap<String, f32> {
        self.speaker_features.clone()
    }

    /// Get a specific quality metric
    pub fn get_quality_metric(&self, name: &str) -> Option<f32> {
        self.quality_metrics.get(name).copied()
    }

    /// Get a specific prosody feature
    pub fn get_prosody_feature(&self, name: &str) -> Option<f32> {
        self.prosody_features.get(name).copied()
    }

    /// Get a specific speaker feature
    pub fn get_speaker_feature(&self, name: &str) -> Option<f32> {
        self.speaker_features.get(name).copied()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "AudioAnalysisResult(quality_metrics={}, prosody_features={}, speaker_features={})",
            self.quality_metrics.len(),
            self.prosody_features.len(),
            self.speaker_features.len()
        )
    }
}

/// Python wrapper for audio analyzer
#[pyclass]
/// Py Audio Analyzer
pub struct PyAudioAnalyzer {
    analyzer: AudioAnalyzerImpl,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyAudioAnalyzer {
    #[new]
    /// new
    pub fn new(config: Option<PyAudioAnalysisConfig>) -> PyResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let analysis_config = config.map(|c| c.inner).unwrap_or_default();

        let analyzer = rt
            .block_on(async { AudioAnalyzerImpl::new(analysis_config).await })
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize analyzer: {}", e)))?;

        Ok(Self { analyzer, rt })
    }

    /// Analyze audio
    pub fn analyze(
        &mut self,
        audio: &PyAudioBuffer,
        config: Option<PyAudioAnalysisConfig>,
    ) -> PyResult<PyAudioAnalysisResult> {
        let analysis_config = config.map(|c| c.inner);

        let result = self
            .rt
            .block_on(async {
                self.analyzer
                    .analyze(&audio.inner, analysis_config.as_ref())
                    .await
            })
            .map_err(|e| PyValueError::new_err(format!("Analysis failed: {}", e)))?;

        // Convert quality metrics
        let quality_metrics = result.quality_metrics;

        // Convert prosody features
        let mut prosody_features = HashMap::new();
        prosody_features.insert("pitch_mean".to_string(), result.prosody.pitch.mean_f0);
        prosody_features.insert("pitch_std".to_string(), result.prosody.pitch.f0_std);
        prosody_features.insert("pitch_range".to_string(), result.prosody.pitch.f0_range);
        prosody_features.insert(
            "speaking_rate".to_string(),
            result.prosody.rhythm.speaking_rate,
        );

        // Convert speaker features
        let mut speaker_features = HashMap::new();
        if let Some(gender) = &result.speaker_characteristics.gender {
            speaker_features.insert(
                "gender".to_string(),
                match gender {
                    crate::traits::Gender::Male => 1.0,
                    crate::traits::Gender::Female => 0.0,
                    crate::traits::Gender::Other => 0.5,
                },
            );
        }
        if let Some(age_range) = &result.speaker_characteristics.age_range {
            speaker_features.insert(
                "age_range".to_string(),
                match age_range {
                    crate::traits::AgeRange::Child => 0.0,
                    crate::traits::AgeRange::Teen => 1.0,
                    crate::traits::AgeRange::Adult => 2.0,
                    crate::traits::AgeRange::Senior => 3.0,
                },
            );
        }

        Ok(PyAudioAnalysisResult {
            quality_metrics,
            prosody_features,
            speaker_features,
        })
    }

    /// String representation
    pub fn __str__(&self) -> String {
        "AudioAnalyzer(initialized)".to_string()
    }
}

/// Python wrapper for performance validator
#[pyclass]
/// Py Performance Validator
pub struct PyPerformanceValidator {
    validator: PerformanceValidator,
}

#[pymethods]
impl PyPerformanceValidator {
    #[new]
    /// new
    pub fn new() -> Self {
        Self {
            validator: PerformanceValidator::new(),
        }
    }

    /// Validate real-time factor
    pub fn validate_rtf(&self, audio: &PyAudioBuffer, processing_time: f32) -> (f32, bool) {
        let duration = Duration::from_secs_f32(processing_time);
        self.validator.validate_rtf(&audio.inner, duration)
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(&self) -> PyResult<(usize, bool)> {
        self.validator
            .estimate_memory_usage()
            .map(|(bytes, passed)| (bytes as usize, passed))
            .map_err(|e| PyValueError::new_err(format!("Memory estimation failed: {}", e)))
    }

    /// String representation
    pub fn __str__(&self) -> String {
        "PerformanceValidator()".to_string()
    }
}

/// Utility function to convert confidence to label
#[pyfunction]
/// confidence to label
pub fn confidence_to_label(confidence: f32) -> String {
    crate::confidence_to_label(confidence).to_string()
}

/// Load audio from file path
#[pyfunction]
/// load audio file
pub fn load_audio_file(path: String, sample_rate: Option<u32>) -> PyResult<PyAudioBuffer> {
    let audio = if let Some(sr) = sample_rate {
        crate::audio_formats::load_audio_with_sample_rate(&path, sr)
    } else {
        crate::audio_formats::load_audio(&path)
    }
    .map_err(|e| PyValueError::new_err(format!("Failed to load audio: {}", e)))?;

    Ok(PyAudioBuffer { inner: audio })
}

/// Create the Python module
#[pymodule]
fn voirs_recognizer(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAudioBuffer>()?;
    m.add_class::<PyASRConfig>()?;
    m.add_class::<PyWordTimestamp>()?;
    m.add_class::<PyRecognitionResult>()?;
    m.add_class::<PyVoiRSRecognizer>()?;
    // #[cfg(feature = "python")]
    // m.add_class::<PyAsyncVoiRSRecognizer>()?;  // Async class temporarily disabled, compatibility resolved
    m.add_class::<PyAudioAnalysisConfig>()?;
    m.add_class::<PyAudioAnalysisResult>()?;
    m.add_class::<PyAudioAnalyzer>()?;
    m.add_class::<PyPerformanceValidator>()?;

    m.add_function(wrap_pyfunction!(confidence_to_label, py)?)?;
    m.add_function(wrap_pyfunction!(load_audio_file, py)?)?;

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
