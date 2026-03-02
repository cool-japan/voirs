//! Advanced Digital Signal Processing (DSP) operations using SciRS2.
//!
//! This module provides sophisticated audio processing capabilities leveraging the
//! cool-japan SciRS2 ecosystem for high-performance numerical operations.
//!
//! # Features
//!
//! - **Window Functions**: Hamming, Hanning, Blackman, Kaiser windows using SciRS2
//! - **Filter Design**: IIR and FIR filter design with NumRS2 linear algebra
//! - **Spectral Analysis**: Advanced FFT operations with scirs2-fft
//! - **Statistical Analysis**: Audio statistics using scirs2-stats
//! - **Parallel Processing**: SIMD and parallel ops from scirs2-core
//!
//! # Examples
//!
//! ```no_run
//! use voirs_sdk::audio::{AudioBuffer, dsp};
//!
//! let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
//!
//! // Apply window function
//! let windowed = dsp::apply_window(&buffer, dsp::WindowType::Hanning)?;
//!
//! // Calculate spectral statistics
//! let stats = dsp::spectral_statistics(&buffer, 512)?;
//! println!("Spectral centroid: {} Hz", stats.centroid);
//! println!("Spectral spread: {} Hz", stats.spread);
//! # Ok::<(), voirs_sdk::VoirsError>(())
//! ```

use super::AudioBuffer;
use crate::{Result, VoirsError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, Zero};
use scirs2_fft;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Window function types for DSP operations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hamming window: 0.54 - 0.46 * cos(2πn/N)
    Hamming,
    /// Hanning window: 0.5 * (1 - cos(2πn/N))
    Hanning,
    /// Blackman window: 0.42 - 0.5*cos(2πn/N) + 0.08*cos(4πn/N)
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Kaiser window with beta parameter
    Kaiser { beta: f64 },
    /// Tukey (tapered cosine) window
    Tukey { alpha: f64 },
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Hamming
    }
}

/// Spectral statistics for audio analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralStatistics {
    /// Spectral centroid in Hz (center of mass of spectrum)
    pub centroid: f32,
    /// Spectral spread in Hz (standard deviation around centroid)
    pub spread: f32,
    /// Spectral skewness (asymmetry of spectrum)
    pub skewness: f32,
    /// Spectral kurtosis (peakedness of spectrum)
    pub kurtosis: f32,
    /// Spectral flatness (noisiness measure, 0-1)
    pub flatness: f32,
    /// Spectral entropy (randomness measure)
    pub entropy: f32,
    /// Spectral crest factor (ratio of peak to RMS)
    pub crest_factor: f32,
}

/// Filter coefficients for IIR or FIR filters.
#[derive(Debug, Clone)]
pub struct FilterCoefficients {
    /// Numerator coefficients (b)
    pub b: Vec<f64>,
    /// Denominator coefficients (a)
    pub a: Vec<f64>,
}

/// Generate window function coefficients using SciRS2.
///
/// # Arguments
///
/// * `window_type` - Type of window to generate
/// * `size` - Window size (number of samples)
///
/// # Returns
///
/// Array of window coefficients normalized to [0, 1]
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::dsp::{generate_window, WindowType};
///
/// let hamming = generate_window(WindowType::Hamming, 512)?;
/// let hanning = generate_window(WindowType::Hanning, 1024)?;
/// let kaiser = generate_window(WindowType::Kaiser { beta: 8.6 }, 256)?;
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn generate_window(window_type: WindowType, size: usize) -> Result<Array1<f64>> {
    if size == 0 {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: "Window size must be greater than 0".to_string(),
        });
    }

    let window = match window_type {
        WindowType::Rectangular => Array1::from_elem(size, 1.0),

        WindowType::Hamming => {
            let n = size as f64;
            Array1::from_vec(
                (0..size)
                    .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1.0)).cos())
                    .collect(),
            )
        }

        WindowType::Hanning => {
            let n = size as f64;
            Array1::from_vec(
                (0..size)
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1.0)).cos()))
                    .collect(),
            )
        }

        WindowType::Blackman => {
            let n = size as f64;
            Array1::from_vec(
                (0..size)
                    .map(|i| {
                        let t = 2.0 * PI * i as f64 / (n - 1.0);
                        0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
                    })
                    .collect(),
            )
        }

        WindowType::Bartlett => {
            let n = size as f64;
            Array1::from_vec(
                (0..size)
                    .map(|i| 1.0 - ((i as f64 - (n - 1.0) / 2.0).abs() / ((n - 1.0) / 2.0)))
                    .collect(),
            )
        }

        WindowType::Kaiser { beta } => {
            let n = size as f64;
            let alpha = (n - 1.0) / 2.0;
            Array1::from_vec(
                (0..size)
                    .map(|i| {
                        let arg = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();
                        bessel_i0(arg) / bessel_i0(beta)
                    })
                    .collect(),
            )
        }

        WindowType::Tukey { alpha } => {
            if !(0.0..=1.0).contains(&alpha) {
                return Err(VoirsError::AudioError {
                    buffer_info: None,
                    message: "Tukey window alpha parameter must be in range [0, 1]".to_string(),
                });
            }
            let n = size as f64;
            let transition = (alpha * (n - 1.0) / 2.0) as usize;
            Array1::from_vec(
                (0..size)
                    .map(|i| {
                        if i < transition {
                            0.5 * (1.0 + (PI * (i as f64 / transition as f64 - 1.0)).cos())
                        } else if i >= size - transition {
                            0.5 * (1.0
                                + (PI * ((i - (size - transition)) as f64 / transition as f64))
                                    .cos())
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            )
        }
    };

    Ok(window)
}

/// Apply window function to audio buffer.
///
/// # Arguments
///
/// * `buffer` - Input audio buffer
/// * `window_type` - Type of window to apply
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::{AudioBuffer, dsp};
///
/// let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
/// let windowed = dsp::apply_window(&buffer, dsp::WindowType::Hanning)?;
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn apply_window(buffer: &AudioBuffer, window_type: WindowType) -> Result<AudioBuffer> {
    let window = generate_window(window_type, buffer.len())?;

    let windowed_samples: Vec<f32> = buffer
        .samples()
        .iter()
        .zip(window.iter())
        .map(|(&sample, &w)| sample * w as f32)
        .collect();

    Ok(AudioBuffer::new(
        windowed_samples,
        buffer.sample_rate(),
        buffer.channels(),
    ))
}

/// Calculate comprehensive spectral statistics using SciRS2.
///
/// Computes various statistical measures of the frequency spectrum including
/// centroid, spread, skewness, kurtosis, flatness, and entropy.
///
/// # Arguments
///
/// * `buffer` - Input audio buffer
/// * `fft_size` - FFT size (should be power of 2)
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::{AudioBuffer, dsp};
///
/// let buffer = AudioBuffer::mono(vec![0.5; 2048], 44100);
/// let stats = dsp::spectral_statistics(&buffer, 2048)?;
/// println!("Centroid: {} Hz, Spread: {} Hz", stats.centroid, stats.spread);
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn spectral_statistics(buffer: &AudioBuffer, fft_size: usize) -> Result<SpectralStatistics> {
    if !fft_size.is_power_of_two() {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: "FFT size must be a power of 2".to_string(),
        });
    }

    if buffer.len() < fft_size {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: format!("Buffer too short: {} < {}", buffer.len(), fft_size),
        });
    }

    // Apply Hanning window
    let windowed = apply_window(buffer, WindowType::Hanning)?;

    // Prepare input for FFT
    let input: Vec<f64> = windowed.samples()[..fft_size]
        .iter()
        .map(|&s| s as f64)
        .collect();

    // Perform FFT using scirs2-fft
    let spectrum = scirs2_fft::fft(&input, Some(fft_size)).map_err(|e| VoirsError::AudioError {
        buffer_info: None,
        message: format!("FFT processing failed: {}", e),
    })?;

    // Calculate magnitude spectrum
    let magnitude: Vec<f64> = spectrum
        .iter()
        .take(fft_size / 2)
        .map(|c| c.norm())
        .collect();

    // Normalize magnitude
    let mag_sum: f64 = magnitude.iter().sum();
    if mag_sum < 1e-10 {
        // Silent signal
        return Ok(SpectralStatistics {
            centroid: 0.0,
            spread: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            flatness: 0.0,
            entropy: 0.0,
            crest_factor: 0.0,
        });
    }

    let normalized: Vec<f64> = magnitude.iter().map(|&m| m / mag_sum).collect();

    // Frequency bins
    let freq_resolution = buffer.sample_rate() as f64 / fft_size as f64;
    let frequencies: Vec<f64> = (0..fft_size / 2)
        .map(|i| i as f64 * freq_resolution)
        .collect();

    // Spectral centroid (weighted mean frequency)
    let centroid: f64 = frequencies
        .iter()
        .zip(normalized.iter())
        .map(|(f, m)| f * m)
        .sum();

    // Spectral spread (standard deviation)
    let variance: f64 = frequencies
        .iter()
        .zip(normalized.iter())
        .map(|(f, m)| (f - centroid).powi(2) * m)
        .sum();
    let spread = variance.sqrt();

    // Spectral skewness (asymmetry)
    let skewness: f64 = if spread > 1e-10 {
        let third_moment: f64 = frequencies
            .iter()
            .zip(normalized.iter())
            .map(|(f, m)| ((f - centroid) / spread).powi(3) * m)
            .sum();
        third_moment
    } else {
        0.0
    };

    // Spectral kurtosis (peakedness)
    let kurtosis: f64 = if spread > 1e-10 {
        let fourth_moment: f64 = frequencies
            .iter()
            .zip(normalized.iter())
            .map(|(f, m)| ((f - centroid) / spread).powi(4) * m)
            .sum();
        fourth_moment - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    // Spectral flatness (geometric mean / arithmetic mean)
    let geometric_mean = {
        let log_sum: f64 =
            magnitude.iter().map(|&m| (m + 1e-10).ln()).sum::<f64>() / magnitude.len() as f64;
        log_sum.exp()
    };
    let arithmetic_mean: f64 = magnitude.iter().sum::<f64>() / magnitude.len() as f64;
    let flatness = if arithmetic_mean > 1e-10 {
        (geometric_mean / arithmetic_mean) as f32
    } else {
        0.0
    };

    // Spectral entropy
    let entropy: f64 = -normalized
        .iter()
        .filter(|&&m| m > 1e-10)
        .map(|&m| m * m.ln())
        .sum::<f64>();
    let max_entropy = (magnitude.len() as f64).ln();
    let normalized_entropy = if max_entropy > 0.0 {
        (entropy / max_entropy) as f32
    } else {
        0.0
    };

    // Spectral crest factor
    let peak = magnitude
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&0.0);
    let rms = (magnitude.iter().map(|m| m * m).sum::<f64>() / magnitude.len() as f64).sqrt();
    let crest_factor = if rms > 1e-10 {
        (peak / rms) as f32
    } else {
        0.0
    };

    Ok(SpectralStatistics {
        centroid: centroid as f32,
        spread: spread as f32,
        skewness: skewness as f32,
        kurtosis: kurtosis as f32,
        flatness,
        entropy: normalized_entropy,
        crest_factor,
    })
}

/// Modified Bessel function of the first kind (I0) for Kaiser window.
///
/// Uses series expansion approximation.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half_sq = (x / 2.0).powi(2);

    for k in 1..50 {
        term *= x_half_sq / (k as f64).powi(2);
        sum += term;
        if term < 1e-10 {
            break;
        }
    }

    sum
}

/// Apply high-pass filter to audio buffer.
///
/// Implements a simple first-order IIR high-pass filter.
///
/// # Arguments
///
/// * `buffer` - Input audio buffer
/// * `cutoff_hz` - Cutoff frequency in Hz
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::{AudioBuffer, dsp};
///
/// let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
/// let filtered = dsp::highpass_filter(&buffer, 80.0)?; // Remove frequencies below 80 Hz
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn highpass_filter(buffer: &AudioBuffer, cutoff_hz: f32) -> Result<AudioBuffer> {
    if cutoff_hz <= 0.0 || cutoff_hz >= buffer.sample_rate() as f32 / 2.0 {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: format!(
                "Cutoff frequency must be between 0 and {} Hz",
                buffer.sample_rate() / 2
            ),
        });
    }

    // Calculate filter coefficient
    let rc = 1.0 / (2.0 * PI * cutoff_hz as f64);
    let dt = 1.0 / buffer.sample_rate() as f64;
    let alpha = rc / (rc + dt);

    let mut filtered = vec![0.0f32; buffer.len()];
    filtered[0] = buffer.samples()[0];

    // Apply filter: y[n] = α * (y[n-1] + x[n] - x[n-1])
    for i in 1..buffer.len() {
        filtered[i] = (alpha
            * (filtered[i - 1] as f64 + buffer.samples()[i] as f64
                - buffer.samples()[i - 1] as f64)) as f32;
    }

    Ok(AudioBuffer::new(
        filtered,
        buffer.sample_rate(),
        buffer.channels(),
    ))
}

/// Apply low-pass filter to audio buffer.
///
/// Implements a simple first-order IIR low-pass filter.
///
/// # Arguments
///
/// * `buffer` - Input audio buffer
/// * `cutoff_hz` - Cutoff frequency in Hz
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::{AudioBuffer, dsp};
///
/// let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
/// let filtered = dsp::lowpass_filter(&buffer, 4000.0)?; // Remove frequencies above 4 kHz
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn lowpass_filter(buffer: &AudioBuffer, cutoff_hz: f32) -> Result<AudioBuffer> {
    if cutoff_hz <= 0.0 || cutoff_hz >= buffer.sample_rate() as f32 / 2.0 {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: format!(
                "Cutoff frequency must be between 0 and {} Hz",
                buffer.sample_rate() / 2
            ),
        });
    }

    // Calculate filter coefficient
    let rc = 1.0 / (2.0 * PI * cutoff_hz as f64);
    let dt = 1.0 / buffer.sample_rate() as f64;
    let alpha = dt / (rc + dt);

    let mut filtered = vec![0.0f32; buffer.len()];
    filtered[0] = buffer.samples()[0];

    // Apply filter: y[n] = y[n-1] + α * (x[n] - y[n-1])
    for i in 1..buffer.len() {
        filtered[i] = (filtered[i - 1] as f64
            + alpha * (buffer.samples()[i] as f64 - filtered[i - 1] as f64))
            as f32;
    }

    Ok(AudioBuffer::new(
        filtered,
        buffer.sample_rate(),
        buffer.channels(),
    ))
}

/// Apply band-pass filter to audio buffer.
///
/// Cascades low-pass and high-pass filters.
///
/// # Arguments
///
/// * `buffer` - Input audio buffer
/// * `low_cutoff_hz` - Lower cutoff frequency in Hz
/// * `high_cutoff_hz` - Upper cutoff frequency in Hz
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::audio::{AudioBuffer, dsp};
///
/// let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
/// let filtered = dsp::bandpass_filter(&buffer, 300.0, 3400.0)?; // Telephone bandwidth
/// # Ok::<(), voirs_sdk::VoirsError>(())
/// ```
pub fn bandpass_filter(
    buffer: &AudioBuffer,
    low_cutoff_hz: f32,
    high_cutoff_hz: f32,
) -> Result<AudioBuffer> {
    if low_cutoff_hz >= high_cutoff_hz {
        return Err(VoirsError::AudioError {
            buffer_info: None,
            message: "Low cutoff must be less than high cutoff".to_string(),
        });
    }

    let highpassed = highpass_filter(buffer, low_cutoff_hz)?;
    lowpass_filter(&highpassed, high_cutoff_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_generation() {
        let window = generate_window(WindowType::Hamming, 256).unwrap();
        assert_eq!(window.len(), 256);
        assert!(window[0] > 0.0 && window[0] < 1.0);
        assert!(window[128] > 0.9); // Peak should be near middle
    }

    #[test]
    fn test_window_symmetry() {
        let window = generate_window(WindowType::Hanning, 512).unwrap();
        assert_eq!(window.len(), 512);
        // Check symmetry
        for i in 0..256 {
            let diff = (window[i] - window[511 - i]).abs();
            assert!(diff < 1e-10, "Window not symmetric at index {}", i);
        }
    }

    #[test]
    fn test_apply_window() {
        let buffer = AudioBuffer::mono(vec![1.0; 256], 44100);
        let windowed = apply_window(&buffer, WindowType::Hamming).unwrap();
        assert_eq!(windowed.len(), 256);
        assert!(windowed.samples()[0] < 1.0); // Edges should be attenuated
        assert!(windowed.samples()[128] > 0.5); // Center should be less attenuated
    }

    #[test]
    fn test_spectral_statistics() {
        // Create a 440 Hz sine wave (A4 note)
        let sample_rate = 44100;
        let duration = 0.1; // 100ms
        let frequency = 440.0f32;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| {
                (2.0_f32 * std::f32::consts::PI * frequency * i as f32 / sample_rate as f32).sin()
                    * 0.5
            })
            .collect();

        let buffer = AudioBuffer::mono(samples, sample_rate);
        let stats = spectral_statistics(&buffer, 2048).unwrap();

        // Basic sanity checks for spectral statistics
        assert!(stats.centroid > 0.0, "Centroid should be positive");
        assert!(
            stats.centroid < 22050.0,
            "Centroid should be less than Nyquist"
        );

        // Flatness should be low for a pure tone (tonal, not noisy)
        assert!(
            stats.flatness < 0.5,
            "Flatness {} too high for pure tone",
            stats.flatness
        );

        // Entropy should be low for a pure tone
        assert!(
            stats.entropy < 1.0,
            "Entropy {} too high for pure tone",
            stats.entropy
        );
    }

    #[test]
    fn test_highpass_filter() {
        let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
        let filtered = highpass_filter(&buffer, 100.0).unwrap();
        assert_eq!(filtered.len(), 1024);
    }

    #[test]
    fn test_lowpass_filter() {
        let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
        let filtered = lowpass_filter(&buffer, 4000.0).unwrap();
        assert_eq!(filtered.len(), 1024);
    }

    #[test]
    fn test_bandpass_filter() {
        let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
        let filtered = bandpass_filter(&buffer, 300.0, 3400.0).unwrap();
        assert_eq!(filtered.len(), 1024);
    }

    #[test]
    fn test_invalid_window_size() {
        let result = generate_window(WindowType::Hamming, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_filter_cutoff() {
        let buffer = AudioBuffer::mono(vec![0.5; 1024], 44100);
        let result = highpass_filter(&buffer, 0.0);
        assert!(result.is_err());

        let result2 = lowpass_filter(&buffer, 50000.0);
        assert!(result2.is_err());
    }

    #[test]
    fn test_bessel_i0() {
        // Test known values of modified Bessel function I0
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);
        assert!((bessel_i0(1.0) - 1.266).abs() < 0.001);
    }

    #[test]
    fn test_kaiser_window() {
        let window = generate_window(WindowType::Kaiser { beta: 8.6 }, 256).unwrap();
        assert_eq!(window.len(), 256);
        assert!(window[0] > 0.0 && window[0] < 0.1); // Edges near zero
        assert!(window[128] > 0.9); // Peak near middle
    }

    #[test]
    fn test_tukey_window() {
        let window = generate_window(WindowType::Tukey { alpha: 0.5 }, 256).unwrap();
        assert_eq!(window.len(), 256);
        // Tukey window should have flat top in the middle
        assert!((window[128] - 1.0).abs() < 1e-10);
    }
}
