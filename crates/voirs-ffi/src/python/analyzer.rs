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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        Ok(array.into_any())
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
    ///
    /// The spectral centroid is the magnitude-weighted mean of the frequency
    /// components, i.e. `centroid = Σ(f_k · |X_k|) / Σ|X_k|`, where `X_k` is the
    /// `k`-th bin of the real FFT of the (Hann-windowed) signal and
    /// `f_k = k · sample_rate / n` is the frequency of that bin in Hz. The result
    /// is a perceptual "brightness" measure: signals dominated by high-frequency
    /// content yield a larger centroid than low-frequency-dominant signals.
    #[staticmethod]
    fn spectral_centroid<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
        sample_rate: u32,
    ) -> PyResult<f32> {
        let samples = audio.as_array();
        let slice = samples.as_slice().ok_or_else(|| {
            PyRuntimeError::new_err("Audio array must be contiguous for spectral analysis")
        })?;
        spectral_centroid_core(slice, sample_rate)
            .map_err(|e| PyRuntimeError::new_err(format!("RFFT failed: {}", e)))
    }
}

/// Pure-numeric spectral centroid, independent of the Python runtime.
///
/// Applies a Hann window, takes the real FFT (`scirs2_fft::rfft`), and returns
/// the magnitude-weighted mean frequency `Σ(f_k · |X_k|) / Σ|X_k|` in Hz, with
/// `f_k = k · sample_rate / n`. Returns `0.0` for signals shorter than two
/// samples or with zero total magnitude.
#[cfg(feature = "numpy")]
fn spectral_centroid_core(samples: &[f32], sample_rate: u32) -> scirs2_fft::FFTResult<f32> {
    let n = samples.len();

    // A spectral centroid is undefined for fewer than two samples (no
    // resolvable frequency bins beyond DC).
    if n < 2 {
        return Ok(0.0);
    }

    // Apply a Hann window to reduce spectral leakage, then take the real FFT.
    // `rfft` is evaluated in f64 for precision; the one-sided spectrum has
    // `n / 2 + 1` complex bins.
    let denom = (n - 1) as f64;
    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            sample as f64 * hann
        })
        .collect();

    let spectrum = scirs2_fft::rfft(&windowed, None)?;

    // centroid = Σ(f_k · |X_k|) / Σ|X_k| with f_k = k · sample_rate / n.
    let bin_hz = sample_rate as f64 / n as f64;
    let mut magnitude_sum = 0.0f64;
    let mut weighted_sum = 0.0f64;
    for (k, bin) in spectrum.iter().enumerate() {
        let magnitude = bin.norm();
        let freq = k as f64 * bin_hz;
        magnitude_sum += magnitude;
        weighted_sum += magnitude * freq;
    }

    if magnitude_sum > 0.0 {
        Ok((weighted_sum / magnitude_sum) as f32)
    } else {
        Ok(0.0)
    }
}

#[cfg(all(test, feature = "numpy"))]
mod tests {
    use super::*;

    /// Generate `n` samples of a sine wave at `freq` Hz sampled at `sample_rate`.
    fn sine(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_spectral_centroid_low_below_high() {
        let sample_rate = 16_000;
        let n = 2048;
        // 200 Hz dominant vs 6000 Hz dominant.
        let low = sine(200.0, sample_rate, n);
        let high = sine(6000.0, sample_rate, n);

        let centroid_low = spectral_centroid_core(&low, sample_rate).expect("low centroid");
        let centroid_high = spectral_centroid_core(&high, sample_rate).expect("high centroid");

        assert!(
            centroid_low < centroid_high,
            "low-frequency centroid ({centroid_low}) should be below high-frequency centroid ({centroid_high})"
        );
        // Each centroid should sit reasonably near its tone's frequency.
        assert!(
            centroid_low < 1500.0,
            "low centroid too high: {centroid_low}"
        );
        assert!(
            centroid_high > 4000.0,
            "high centroid too low: {centroid_high}"
        );
    }

    #[test]
    fn test_spectral_centroid_degenerate_inputs() {
        // Fewer than two samples => 0.0.
        assert_eq!(spectral_centroid_core(&[], 16_000).unwrap(), 0.0);
        assert_eq!(spectral_centroid_core(&[0.5], 16_000).unwrap(), 0.0);
        // All-zero signal => 0.0 (no magnitude).
        assert_eq!(spectral_centroid_core(&[0.0; 64], 16_000).unwrap(), 0.0);
    }
}
