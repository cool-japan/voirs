//! Metrics and synthesis result types for Python bindings

use super::common::*;

/// Enhanced metrics and monitoring for Python bindings
#[pyclass]
#[derive(Debug, Clone)]
pub struct SynthesisMetrics {
    #[pyo3(get)]
    pub processing_time_ms: f64,
    #[pyo3(get)]
    pub audio_duration_ms: f64,
    #[pyo3(get)]
    pub real_time_factor: f64,
    #[pyo3(get)]
    pub memory_usage_mb: f64,
    #[pyo3(get)]
    pub cache_hit_rate: f64,
}

#[pymethods]
impl SynthesisMetrics {
    #[new]
    pub fn new(
        processing_time_ms: f64,
        audio_duration_ms: f64,
        real_time_factor: f64,
        memory_usage_mb: f64,
        cache_hit_rate: f64,
    ) -> Self {
        Self {
            processing_time_ms,
            audio_duration_ms,
            real_time_factor,
            memory_usage_mb,
            cache_hit_rate,
        }
    }

    fn __str__(&self) -> String {
        format!(
            "SynthesisMetrics(rtf={:.3}, cache_hit_rate={:.1}%)",
            self.real_time_factor,
            self.cache_hit_rate * 100.0
        )
    }
}

/// Enhanced synthesis result with metrics
#[pyclass]
pub struct SynthesisResult {
    #[pyo3(get)]
    pub audio: super::audio_buffer::PyAudioBuffer,
    #[pyo3(get)]
    pub metrics: SynthesisMetrics,
}

#[pymethods]
impl SynthesisResult {
    #[new]
    pub fn new(audio: super::audio_buffer::PyAudioBuffer, metrics: SynthesisMetrics) -> Self {
        Self { audio, metrics }
    }
}
