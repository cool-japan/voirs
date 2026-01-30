//! Audio analyzer for Python bindings

use super::common::*;

#[cfg(feature = "numpy")]
#[pyclass]
pub struct PyAudioAnalyzer;

#[cfg(feature = "numpy")]
#[pymethods]
impl PyAudioAnalyzer {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute RMS energy of audio signal
    #[staticmethod]
    fn rms_energy<'py>(py: Python<'py>, audio: PyReadonlyArray1<f32>) -> PyResult<f32> {
        let samples = audio.as_array();
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / samples.len() as f32).sqrt();
        Ok(rms)
    }

    /// Find silence regions in audio
    #[staticmethod]
    fn find_silence<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
        threshold: f32,
        min_duration: usize,
    ) -> PyResult<PyObject> {
        let samples = audio.as_array();
        let mut silence_regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start = 0;

        for (i, &sample) in samples.iter().enumerate() {
            let is_silent = sample.abs() < threshold;

            if is_silent && !in_silence {
                silence_start = i;
                in_silence = true;
            } else if !is_silent && in_silence {
                let duration = i - silence_start;
                if duration >= min_duration {
                    silence_regions.push([silence_start, i]);
                }
                in_silence = false;
            }
        }

        // Handle silence at the end
        if in_silence {
            let duration = samples.len() - silence_start;
            if duration >= min_duration {
                silence_regions.push([silence_start, samples.len()]);
            }
        }

        // Convert Vec<[usize; 2]> to Vec<Vec<usize>> for from_vec2
        let silence_vecs: Vec<Vec<usize>> = silence_regions
            .into_iter()
            .map(|[start, end]| vec![start, end])
            .collect();
        let array = PyArray2::from_vec2(py, &silence_vecs)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))?;
        Ok(array.unbind().into())
    }

    /// Compute zero crossing rate
    #[staticmethod]
    fn zero_crossing_rate<'py>(py: Python<'py>, audio: PyReadonlyArray1<f32>) -> PyResult<f32> {
        let samples = audio.as_array();
        if samples.len() < 2 {
            return Ok(0.0);
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        Ok(crossings as f32 / (samples.len() - 1) as f32)
    }

    /// Compute spectral centroid (brightness measure)
    #[staticmethod]
    fn spectral_centroid<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
        sample_rate: u32,
    ) -> PyResult<f32> {
        let samples = audio.as_array();
        let n = samples.len();

        // Simple spectral centroid calculation (placeholder for real FFT)
        let mut magnitude_sum = 0.0f32;
        let mut weighted_sum = 0.0f32;

        for (i, &sample) in samples.iter().enumerate() {
            let magnitude = sample.abs();
            let freq = (i as f32 * sample_rate as f32) / (n as f32);

            magnitude_sum += magnitude;
            weighted_sum += magnitude * freq;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }
}
