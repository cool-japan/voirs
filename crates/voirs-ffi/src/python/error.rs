//! Error types for Python bindings

use super::common::*;

/// Structured error information for Python
#[pyclass]
#[derive(Debug, Clone)]
pub struct VoirsErrorInfo {
    #[pyo3(get)]
    pub code: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub details: Option<String>,
    #[pyo3(get)]
    pub suggestion: Option<String>,
}

impl VoirsErrorInfo {
    /// Create a new VoirsErrorInfo (for Rust usage)
    pub(crate) fn create(
        code: String,
        message: String,
        details: Option<String>,
        suggestion: Option<String>,
    ) -> Self {
        Self {
            code,
            message,
            details,
            suggestion,
        }
    }
}

#[pymethods]
impl VoirsErrorInfo {
    #[new]
    pub fn new(
        code: String,
        message: String,
        details: Option<String>,
        suggestion: Option<String>,
    ) -> Self {
        Self::create(code, message, details, suggestion)
    }

    fn __str__(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }

    fn __repr__(&self) -> String {
        format!(
            "VoirsErrorInfo(code='{}', message='{}')",
            self.code, self.message
        )
    }
}

/// Enhanced exception with structured error information
#[pyclass(extends=PyRuntimeError)]
pub struct VoirsException {
    #[pyo3(get)]
    pub error_info: VoirsErrorInfo,
}

#[pymethods]
impl VoirsException {
    #[new]
    pub fn new(error_info: VoirsErrorInfo) -> Self {
        Self { error_info }
    }
}
