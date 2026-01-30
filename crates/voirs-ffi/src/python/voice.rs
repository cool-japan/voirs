//! Voice information wrapper for Python bindings

use super::common::*;

/// Python VoiceInfo wrapper
#[pyclass]
#[derive(Clone)]
pub struct PyVoiceInfo {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub quality: String,
    #[pyo3(get)]
    pub is_available: bool,
}

impl From<VoiceInfo> for PyVoiceInfo {
    fn from(voice: VoiceInfo) -> Self {
        Self {
            id: voice.config.id,
            name: voice.config.name,
            language: voice.config.language.to_string(),
            quality: format!("{:?}", voice.config.characteristics.quality),
            is_available: true, // Always available for now
        }
    }
}
