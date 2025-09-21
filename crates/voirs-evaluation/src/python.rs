//! Python bindings for VoiRS evaluation system
//!
//! This module provides Python bindings for the VoiRS evaluation system, enabling
//! seamless integration with Python scientific computing tools including NumPy,
//! SciPy, Pandas, and Matplotlib.
//!
//! ## Features
//!
//! - **NumPy Array Integration**: Direct support for NumPy arrays as audio input
//! - **Quality Evaluation**: Complete access to PESQ, STOI, MCD, MSD metrics
//! - **Statistical Analysis**: Full statistical framework with confidence intervals
//! - **Batch Processing**: Efficient evaluation of multiple audio samples
//! - **DataFrame Output**: Results formatted for Pandas DataFrame integration
//!
//! ## Installation
//!
//! ```bash
//! pip install voirs-evaluation
//! ```
//!
//! ## Quick Start
//!
//! ```python
//! import numpy as np
//! import voirs_evaluation as ve
//!
//! # Load audio data as NumPy arrays
//! reference = np.random.randn(16000).astype(np.float32)
//! degraded = np.random.randn(16000).astype(np.float32)
//!
//! # Create evaluator
//! evaluator = ve.PyQualityEvaluator()
//!
//! # Evaluate quality
//! result = evaluator.evaluate(reference, degraded, sample_rate=16000)
//! print(f"Overall Score: {result.overall_score}")
//!
//! # Statistical analysis
//! analyzer = ve.PyStatisticalAnalyzer()
//! stats = analyzer.paired_t_test([1.2, 1.5, 1.8], [1.0, 1.3, 1.6])
//! print(f"P-value: {stats.p_value}")
//! ```

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

#[cfg(feature = "python")]
use crate::{
    quality::QualityEvaluator,
    statistical::{CorrelationResult, StatisticalAnalyzer, StatisticalTestResult},
    traits::{EvaluationResult, QualityEvaluator as QualityEvaluatorTrait, QualityScore},
};

#[cfg(feature = "python")]
use voirs_sdk::AudioBuffer;

#[cfg(feature = "python")]
use std::collections::HashMap;

/// Python wrapper for quality evaluation results
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyQualityResult {
    /// PESQ score (1.0-4.5, higher is better)
    #[pyo3(get)]
    pub pesq: f32,
    /// STOI score (0.0-1.0, higher is better)
    #[pyo3(get)]
    pub stoi: f32,
    /// MCD score (lower is better)
    #[pyo3(get)]
    pub mcd: f32,
    /// MSD score (lower is better)
    #[pyo3(get)]
    pub msd: f32,
    /// Overall quality score (0.0-1.0, higher is better)
    #[pyo3(get)]
    pub overall_score: f32,
    /// Confidence score (0.0-1.0)
    #[pyo3(get)]
    pub confidence: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQualityResult {
    /// Convert to Python dictionary for DataFrame integration
    pub fn to_dict(&self) -> PyResult<HashMap<String, f32>> {
        let mut result = HashMap::new();
        result.insert("pesq".to_string(), self.pesq);
        result.insert("stoi".to_string(), self.stoi);
        result.insert("mcd".to_string(), self.mcd);
        result.insert("msd".to_string(), self.msd);
        result.insert("overall_score".to_string(), self.overall_score);
        result.insert("confidence".to_string(), self.confidence);
        Ok(result)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "QualityResult(pesq={:.3}, stoi={:.3}, mcd={:.3}, msd={:.3}, overall={:.3}, confidence={:.3})",
            self.pesq, self.stoi, self.mcd, self.msd, self.overall_score, self.confidence
        )
    }
}

/// Python wrapper for statistical test results
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyStatisticalResult {
    /// Test statistic value
    #[pyo3(get)]
    pub statistic: f32,
    /// P-value
    #[pyo3(get)]
    pub p_value: f32,
    /// Effect size
    #[pyo3(get)]
    pub effect_size: f32,
    /// Confidence interval lower bound
    #[pyo3(get)]
    pub ci_lower: f32,
    /// Confidence interval upper bound
    #[pyo3(get)]
    pub ci_upper: f32,
    /// Degrees of freedom
    #[pyo3(get)]
    pub degrees_of_freedom: i32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStatisticalResult {
    /// Check if result is statistically significant (p < 0.05)
    pub fn is_significant(&self) -> bool {
        self.p_value < 0.05
    }

    /// Convert to Python dictionary
    pub fn to_dict(&self) -> PyResult<HashMap<String, f32>> {
        let mut result = HashMap::new();
        result.insert("statistic".to_string(), self.statistic);
        result.insert("p_value".to_string(), self.p_value);
        result.insert("effect_size".to_string(), self.effect_size);
        result.insert("ci_lower".to_string(), self.ci_lower);
        result.insert("ci_upper".to_string(), self.ci_upper);
        result.insert(
            "degrees_of_freedom".to_string(),
            self.degrees_of_freedom as f32,
        );
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "StatisticalResult(statistic={:.3}, p_value={:.3}, effect_size={:.3}, ci=[{:.3}, {:.3}])",
            self.statistic, self.p_value, self.effect_size, self.ci_lower, self.ci_upper
        )
    }
}

/// Python wrapper for pronunciation evaluation results
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyPronunciationResult {
    /// Overall pronunciation score (0.0-1.0)
    #[pyo3(get)]
    pub overall_score: f32,
    /// Phoneme accuracy score
    #[pyo3(get)]
    pub phoneme_accuracy: f32,
    /// Fluency score
    #[pyo3(get)]
    pub fluency_score: f32,
    /// Prosody score
    #[pyo3(get)]
    pub prosody_score: f32,
    /// Confidence score
    #[pyo3(get)]
    pub confidence: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPronunciationResult {
    /// Convert to Python dictionary for DataFrame integration
    pub fn to_dict(&self) -> PyResult<HashMap<String, f32>> {
        let mut result = HashMap::new();
        result.insert("overall_score".to_string(), self.overall_score);
        result.insert("phoneme_accuracy".to_string(), self.phoneme_accuracy);
        result.insert("fluency_score".to_string(), self.fluency_score);
        result.insert("prosody_score".to_string(), self.prosody_score);
        result.insert("confidence".to_string(), self.confidence);
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "PronunciationResult(overall={:.3}, phoneme={:.3}, fluency={:.3}, prosody={:.3})",
            self.overall_score, self.phoneme_accuracy, self.fluency_score, self.prosody_score
        )
    }
}

/// Python wrapper for quality evaluator
#[cfg(feature = "python")]
#[pyclass]
pub struct PyQualityEvaluator {
    /// Internal quality evaluator
    evaluator: Option<QualityEvaluator>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQualityEvaluator {
    /// Create a new quality evaluator
    #[new]
    pub fn new() -> Self {
        Self { evaluator: None }
    }

    /// Initialize the evaluator (call this after creation)
    pub fn initialize(&mut self) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create async runtime: {}",
                e
            ))
        })?;

        let evaluator = rt
            .block_on(async { QualityEvaluator::new().await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to initialize evaluator: {}",
                    e
                ))
            })?;

        self.evaluator = Some(evaluator);
        Ok(())
    }

    /// Evaluate audio quality using NumPy arrays
    pub fn evaluate(
        &self,
        reference: PyReadonlyArray1<f32>,
        degraded: PyReadonlyArray1<f32>,
        sample_rate: u32,
    ) -> PyResult<PyQualityResult> {
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Evaluator not initialized. Call initialize() first.",
            )
        })?;

        let ref_array = reference.as_array();
        let deg_array = degraded.as_array();

        // Convert to Vec<f32>
        let ref_samples: Vec<f32> = ref_array.to_vec();
        let deg_samples: Vec<f32> = deg_array.to_vec();

        // Create AudioBuffer instances
        let ref_audio = AudioBuffer::mono(ref_samples, sample_rate);
        let deg_audio = AudioBuffer::mono(deg_samples, sample_rate);

        // Use actual quality evaluation
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create async runtime: {}",
                e
            ))
        })?;

        let quality_score = rt
            .block_on(async {
                evaluator
                    .evaluate_quality(&deg_audio, Some(&ref_audio), None)
                    .await
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Quality evaluation failed: {}",
                    e
                ))
            })?;

        // Extract individual metric scores from component_scores
        let pesq = quality_score
            .component_scores
            .get("PESQ")
            .copied()
            .unwrap_or(0.0);
        let stoi = quality_score
            .component_scores
            .get("STOI")
            .copied()
            .unwrap_or(0.0);
        let mcd = quality_score
            .component_scores
            .get("MCD")
            .copied()
            .unwrap_or(0.0);
        let msd = quality_score
            .component_scores
            .get("MSD")
            .copied()
            .unwrap_or(0.0);

        Ok(PyQualityResult {
            pesq,
            stoi,
            mcd,
            msd,
            overall_score: quality_score.overall_score,
            confidence: quality_score.confidence,
        })
    }

    /// Evaluate quality without reference (no-reference evaluation)
    pub fn evaluate_no_reference(
        &self,
        audio: PyReadonlyArray1<f32>,
        sample_rate: u32,
    ) -> PyResult<PyQualityResult> {
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Evaluator not initialized. Call initialize() first.",
            )
        })?;

        let audio_array = audio.as_array();
        let samples: Vec<f32> = audio_array.to_vec();
        let audio_buffer = AudioBuffer::mono(samples, sample_rate);

        // Use actual quality evaluation (no reference)
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create async runtime: {}",
                e
            ))
        })?;

        let quality_score = rt
            .block_on(async { evaluator.evaluate_quality(&audio_buffer, None, None).await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Quality evaluation failed: {}",
                    e
                ))
            })?;

        // Extract individual metric scores from component_scores
        // Note: For no-reference evaluation, some metrics may not be available
        let pesq = quality_score
            .component_scores
            .get("PESQ")
            .copied()
            .unwrap_or(0.0);
        let stoi = quality_score
            .component_scores
            .get("STOI")
            .copied()
            .unwrap_or(0.0);
        let mcd = quality_score
            .component_scores
            .get("MCD")
            .copied()
            .unwrap_or(0.0);
        let msd = quality_score
            .component_scores
            .get("MSD")
            .copied()
            .unwrap_or(0.0);

        Ok(PyQualityResult {
            pesq,
            stoi,
            mcd,
            msd,
            overall_score: quality_score.overall_score,
            confidence: quality_score.confidence,
        })
    }

    /// Batch evaluation for multiple audio pairs
    pub fn evaluate_batch(
        &self,
        references: Vec<PyReadonlyArray1<f32>>,
        degraded_samples: Vec<PyReadonlyArray1<f32>>,
        sample_rate: u32,
    ) -> PyResult<Vec<PyQualityResult>> {
        let evaluator = self.evaluator.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Evaluator not initialized. Call initialize() first.",
            )
        })?;

        if references.len() != degraded_samples.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Reference and degraded arrays must have the same length",
            ));
        }

        // Prepare audio buffer pairs for batch evaluation
        let mut audio_pairs = Vec::new();
        for (ref_array, deg_array) in references.iter().zip(degraded_samples.iter()) {
            let ref_samples: Vec<f32> = ref_array.as_array().to_vec();
            let deg_samples: Vec<f32> = deg_array.as_array().to_vec();

            let ref_audio = AudioBuffer::mono(ref_samples, sample_rate);
            let deg_audio = AudioBuffer::mono(deg_samples, sample_rate);

            audio_pairs.push((deg_audio, Some(ref_audio)));
        }

        // Use actual batch quality evaluation
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create async runtime: {}",
                e
            ))
        })?;

        let quality_scores = rt
            .block_on(async { evaluator.evaluate_quality_batch(&audio_pairs, None).await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch quality evaluation failed: {}",
                    e
                ))
            })?;

        // Convert results to PyQualityResult
        let mut results = Vec::new();
        for quality_score in quality_scores {
            let pesq = quality_score
                .component_scores
                .get("PESQ")
                .copied()
                .unwrap_or(0.0);
            let stoi = quality_score
                .component_scores
                .get("STOI")
                .copied()
                .unwrap_or(0.0);
            let mcd = quality_score
                .component_scores
                .get("MCD")
                .copied()
                .unwrap_or(0.0);
            let msd = quality_score
                .component_scores
                .get("MSD")
                .copied()
                .unwrap_or(0.0);

            results.push(PyQualityResult {
                pesq,
                stoi,
                mcd,
                msd,
                overall_score: quality_score.overall_score,
                confidence: quality_score.confidence,
            });
        }

        Ok(results)
    }
}

/// Python wrapper for statistical analyzer
#[cfg(feature = "python")]
#[pyclass]
pub struct PyStatisticalAnalyzer {
    analyzer: StatisticalAnalyzer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStatisticalAnalyzer {
    /// Create a new statistical analyzer
    #[new]
    pub fn new() -> Self {
        Self {
            analyzer: StatisticalAnalyzer::new(),
        }
    }

    /// Perform paired t-test
    pub fn paired_t_test(
        &self,
        group1: Vec<f32>,
        group2: Vec<f32>,
    ) -> PyResult<PyStatisticalResult> {
        let result = self
            .analyzer
            .paired_t_test(&group1, &group2, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        let (ci_lower, ci_upper) = result.confidence_interval.unwrap_or((0.0, 0.0));

        Ok(PyStatisticalResult {
            statistic: result.test_statistic as f32,
            p_value: result.p_value as f32,
            effect_size: result.effect_size.unwrap_or(0.0) as f32,
            ci_lower: ci_lower as f32,
            ci_upper: ci_upper as f32,
            degrees_of_freedom: result.degrees_of_freedom.map(|df| df as i32).unwrap_or(-1),
        })
    }

    /// Perform correlation test
    pub fn correlation_test(&self, x: Vec<f32>, y: Vec<f32>) -> PyResult<PyStatisticalResult> {
        let correlation_analyzer = crate::statistical::correlation::CorrelationAnalyzer::new();
        let result = correlation_analyzer
            .pearson_correlation(&x, &y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        let (ci_lower, ci_upper) = result.confidence_interval;

        Ok(PyStatisticalResult {
            statistic: result.test_statistic,
            p_value: result.p_value,
            effect_size: Some(result.coefficient).unwrap_or(0.0),
            ci_lower,
            ci_upper,
            degrees_of_freedom: result.degrees_freedom as i32,
        })
    }

    /// Calculate descriptive statistics (simplified)
    pub fn descriptive_stats(&self, data: Vec<f32>) -> PyResult<HashMap<String, f32>> {
        if data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data cannot be empty",
            ));
        }

        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        let mut result = HashMap::new();
        result.insert("mean".to_string(), mean);
        result.insert("std_dev".to_string(), std_dev);
        result.insert("variance".to_string(), variance);
        result.insert("min".to_string(), min);
        result.insert("max".to_string(), max);
        result.insert("median".to_string(), median);

        Ok(result)
    }
}

/// Python wrapper for pronunciation evaluator (simplified)
#[cfg(feature = "python")]
#[pyclass]
pub struct PyPronunciationEvaluator;

#[cfg(feature = "python")]
#[pymethods]
impl PyPronunciationEvaluator {
    /// Create a new pronunciation evaluator
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Evaluate pronunciation from audio and reference text
    pub fn evaluate(
        &self,
        audio: PyReadonlyArray1<f32>,
        reference_text: String,
        sample_rate: u32,
    ) -> PyResult<PyPronunciationResult> {
        let audio_array = audio.as_array();
        let samples: Vec<f32> = audio_array.to_vec();
        let _audio_buffer = AudioBuffer::mono(samples, sample_rate);

        // Placeholder implementation
        let _text_len = reference_text.len() as f32;

        Ok(PyPronunciationResult {
            overall_score: 0.85,
            phoneme_accuracy: 0.82,
            fluency_score: 0.88,
            prosody_score: 0.75,
            confidence: 0.9,
        })
    }
}

/// Utility functions for Python integration
#[cfg(feature = "python")]
#[pyfunction]
pub fn create_sine_wave(frequency: f32, duration: f32, sample_rate: u32) -> PyResult<Vec<f32>> {
    let samples_count = (duration * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(samples_count);

    for i in 0..samples_count {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(sample);
    }

    Ok(samples)
}

/// Add white noise to audio signal
#[cfg(feature = "python")]
#[pyfunction]
pub fn add_noise(audio: PyReadonlyArray1<f32>, noise_level: f32) -> PyResult<Vec<f32>> {
    use rand::Rng;
    let audio_array = audio.as_array();
    let mut rng = rand::thread_rng();

    let noisy_samples: Vec<f32> = audio_array
        .iter()
        .map(|&sample| {
            let noise = rng.gen_range(-noise_level..noise_level);
            sample + noise
        })
        .collect();

    Ok(noisy_samples)
}

/// Calculate SNR between two signals
#[cfg(feature = "python")]
#[pyfunction]
pub fn calculate_snr(signal: PyReadonlyArray1<f32>, noise: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let signal_array = signal.as_array();
    let noise_array = noise.as_array();

    if signal_array.len() != noise_array.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal and noise arrays must have the same length",
        ));
    }

    let signal_power: f32 = signal_array.iter().map(|&x| x * x).sum();
    let noise_power: f32 = noise_array.iter().map(|&x| x * x).sum();

    if noise_power == 0.0 {
        return Ok(f32::INFINITY);
    }

    let snr_linear = signal_power / noise_power;
    let snr_db = 10.0 * snr_linear.log10();

    Ok(snr_db)
}

/// Initialize the Python module
#[cfg(feature = "python")]
#[pymodule]
fn voirs_evaluation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<PyQualityEvaluator>()?;
    m.add_class::<PyStatisticalAnalyzer>()?;
    m.add_class::<PyPronunciationEvaluator>()?;
    m.add_class::<PyQualityResult>()?;
    m.add_class::<PyStatisticalResult>()?;
    m.add_class::<PyPronunciationResult>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(create_sine_wave, m)?)?;
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_snr, m)?)?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "VoiRS Team")?;
    m.add("__description__", "VoiRS evaluation system Python bindings")?;

    Ok(())
}

// Re-export for easier access when python feature is not enabled
#[cfg(not(feature = "python"))]
pub fn python_bindings_not_available() {
    eprintln!("Python bindings are not available. Build with --features python to enable.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_compilation() {
        // Test that the module compiles correctly
        #[cfg(feature = "python")]
        {
            // Python feature tests would go here
            // For now, just test that the structs can be created
            let _evaluator = PyQualityEvaluator::new();
            let _analyzer = PyStatisticalAnalyzer::new();
            let _pron_evaluator = PyPronunciationEvaluator::new();
        }

        #[cfg(not(feature = "python"))]
        {
            python_bindings_not_available();
        }
    }
}
