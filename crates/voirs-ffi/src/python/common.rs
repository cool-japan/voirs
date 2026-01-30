//! Common imports and utilities for Python bindings

pub(super) use crate::{VoirsAudioFormat, VoirsQualityLevel};
pub(super) use pyo3::exceptions::{PyRuntimeError, PyValueError};
pub(super) use pyo3::prelude::*;
pub(super) use pyo3::types::{PyBytes, PyList};
pub(super) use std::sync::Arc;
pub(super) use tokio::runtime::Runtime;
pub(super) use voirs_sdk::{
    audio::AudioBuffer, error::VoirsError, voice::info::VoiceInfo, VoirsPipeline as SdkPipeline,
};

// NumPy integration (when numpy feature is enabled)
#[cfg(feature = "numpy")]
pub(super) use numpy::{
    npyffi::NPY_ORDER, Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayDyn,
    PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray,
};

#[cfg(feature = "numpy")]
pub(super) use numpy::ToPyArray as _; // Ensure trait is in scope

/// Helper function to convert VoirsError to PyErr
pub(super) fn to_pyerr(error: VoirsError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", error))
}
