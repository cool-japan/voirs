//! Mel spectrogram computation and processing
//!
//! This module provides comprehensive mel spectrogram functionality including
//! computation from audio, tensor operations, and various utility functions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, AcousticError, MelSpectrogram};

pub mod computation;
pub mod ops;
pub mod utils;

pub use computation::*;
pub use ops::*;
pub use utils::*;

/// Mel spectrogram parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelParams {
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// Number of mel filter banks
    pub n_mels: u32,
    /// FFT window size
    pub n_fft: u32,
    /// Hop length between frames
    pub hop_length: u32,
    /// Window length
    pub win_length: u32,
    /// Minimum frequency for mel filters
    pub fmin: f32,
    /// Maximum frequency for mel filters
    pub fmax: Option<f32>,
    /// Window function type
    pub window: WindowType,
    /// Power for power spectrogram
    pub power: f32,
    /// Whether to use log scale
    pub log: bool,
    /// Log offset to avoid log(0)
    pub log_offset: f32,
    /// Normalization method
    pub norm: Option<MelNormalization>,
}

impl MelParams {
    /// Create new mel parameters with defaults
    pub fn new(sample_rate: u32, n_mels: u32) -> Self {
        Self {
            sample_rate,
            n_mels,
            n_fft: 2048,
            hop_length: 256,
            win_length: 1024,
            fmin: 0.0,
            fmax: None,
            window: WindowType::Hann,
            power: 2.0,
            log: true,
            log_offset: 1e-10,
            norm: Some(MelNormalization::Slaney),
        }
    }
    
    /// Create standard 22kHz parameters
    pub fn standard_22khz() -> Self {
        Self::new(22050, 80)
    }
    
    /// Create standard 16kHz parameters
    pub fn standard_16khz() -> Self {
        let mut params = Self::new(16000, 80);
        params.n_fft = 1024;
        params.hop_length = 256;
        params.win_length = 1024;
        params.fmax = Some(8000.0);
        params
    }
    
    /// Create high-resolution parameters
    pub fn high_resolution(sample_rate: u32) -> Self {
        let mut params = Self::new(sample_rate, 128);
        params.n_fft = 4096;
        params.hop_length = 512;
        params.win_length = 2048;
        params
    }
    
    /// Validate mel parameters
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(AcousticError::ConfigError("Sample rate must be > 0".to_string()));
        }
        if self.n_mels == 0 {
            return Err(AcousticError::ConfigError("Number of mel filters must be > 0".to_string()));
        }
        if self.n_fft == 0 {
            return Err(AcousticError::ConfigError("FFT size must be > 0".to_string()));
        }
        if self.hop_length == 0 {
            return Err(AcousticError::ConfigError("Hop length must be > 0".to_string()));
        }
        if self.win_length == 0 {
            return Err(AcousticError::ConfigError("Window length must be > 0".to_string()));
        }
        if self.win_length > self.n_fft {
            return Err(AcousticError::ConfigError("Window length must be <= FFT size".to_string()));
        }
        if self.fmin < 0.0 {
            return Err(AcousticError::ConfigError("Minimum frequency must be >= 0".to_string()));
        }
        if let Some(fmax) = self.fmax {
            if fmax <= self.fmin {
                return Err(AcousticError::ConfigError("Maximum frequency must be > minimum frequency".to_string()));
            }
            if fmax > self.sample_rate as f32 / 2.0 {
                return Err(AcousticError::ConfigError("Maximum frequency must be <= Nyquist frequency".to_string()));
            }
        }
        if self.power <= 0.0 {
            return Err(AcousticError::ConfigError("Power must be > 0".to_string()));
        }
        if self.log_offset <= 0.0 {
            return Err(AcousticError::ConfigError("Log offset must be > 0".to_string()));
        }
        Ok(())
    }
    
    /// Get effective maximum frequency
    pub fn effective_fmax(&self) -> f32 {
        self.fmax.unwrap_or(self.sample_rate as f32 / 2.0)
    }
    
    /// Get number of frequency bins
    pub fn n_freqs(&self) -> u32 {
        self.n_fft / 2 + 1
    }
    
    /// Convert time to frame index
    pub fn time_to_frame(&self, time_seconds: f32) -> u32 {
        ((time_seconds * self.sample_rate as f32) / self.hop_length as f32).round() as u32
    }
    
    /// Convert frame index to time
    pub fn frame_to_time(&self, frame: u32) -> f32 {
        (frame * self.hop_length) as f32 / self.sample_rate as f32
    }
}

impl Default for MelParams {
    fn default() -> Self {
        Self::standard_22khz()
    }
}

/// Window functions for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowType {
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett window
    Bartlett,
    /// Rectangular window
    Rectangular,
}

impl WindowType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            WindowType::Hann => "hann",
            WindowType::Hamming => "hamming",
            WindowType::Blackman => "blackman",
            WindowType::Bartlett => "bartlett",
            WindowType::Rectangular => "rectangular",
        }
    }
    
    /// Generate window function
    pub fn generate(&self, length: usize) -> Vec<f32> {
        match self {
            WindowType::Hann => hann_window(length),
            WindowType::Hamming => hamming_window(length),
            WindowType::Blackman => blackman_window(length),
            WindowType::Bartlett => bartlett_window(length),
            WindowType::Rectangular => vec![1.0; length],
        }
    }
}

/// Mel filter bank normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MelNormalization {
    /// Slaney normalization (default in librosa)
    Slaney,
    /// No normalization
    None,
    /// Unit area normalization
    UnitArea,
}

impl MelNormalization {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            MelNormalization::Slaney => "slaney",
            MelNormalization::None => "none",
            MelNormalization::UnitArea => "unit_area",
        }
    }
}

/// Mel spectrogram metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelMetadata {
    /// Mel parameters used for computation
    pub params: MelParams,
    /// Duration in seconds
    pub duration_seconds: f32,
    /// Number of frames
    pub n_frames: u32,
    /// Spectral statistics
    pub stats: Option<MelStats>,
    /// Additional metadata
    pub extra: HashMap<String, serde_json::Value>,
}

impl MelMetadata {
    /// Create new mel metadata
    pub fn new(params: MelParams, n_frames: u32) -> Self {
        let duration_seconds = params.frame_to_time(n_frames);
        Self {
            params,
            duration_seconds,
            n_frames,
            stats: None,
            extra: HashMap::new(),
        }
    }
    
    /// Add spectral statistics
    pub fn with_stats(mut self, stats: MelStats) -> Self {
        self.stats = Some(stats);
        self
    }
    
    /// Add extra metadata
    pub fn with_extra<K: Into<String>, V: Into<serde_json::Value>>(mut self, key: K, value: V) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Spectral statistics for mel spectrograms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelStats {
    /// Mean values per mel channel
    pub mean: Vec<f32>,
    /// Standard deviation per mel channel
    pub std: Vec<f32>,
    /// Minimum values per mel channel
    pub min: Vec<f32>,
    /// Maximum values per mel channel
    pub max: Vec<f32>,
    /// Global statistics
    pub global: GlobalStats,
}

impl MelStats {
    /// Compute statistics from mel spectrogram
    pub fn compute(mel: &MelSpectrogram) -> Result<Self> {
        if mel.data.is_empty() {
            return Err(AcousticError::InputError("Empty mel spectrogram".to_string()));
        }
        
        let n_mels = mel.n_mels;
        let mut mean = vec![0.0; n_mels];
        let mut std = vec![0.0; n_mels];
        let mut min = vec![f32::INFINITY; n_mels];
        let mut max = vec![f32::NEG_INFINITY; n_mels];
        
        // Compute per-channel statistics
        for (i, channel) in mel.data.iter().enumerate() {
            if i >= n_mels {
                break;
            }
            
            let sum: f32 = channel.iter().sum();
            let count = channel.len() as f32;
            mean[i] = sum / count;
            
            let variance: f32 = channel.iter()
                .map(|&x| (x - mean[i]).powi(2))
                .sum::<f32>() / count;
            std[i] = variance.sqrt();
            
            min[i] = channel.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            max[i] = channel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        }
        
        // Compute global statistics
        let all_values: Vec<f32> = mel.data.iter().flatten().copied().collect();
        let global_mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
        let global_variance = all_values.iter()
            .map(|&x| (x - global_mean).powi(2))
            .sum::<f32>() / all_values.len() as f32;
        let global_std = global_variance.sqrt();
        let global_min = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let global_max = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let global = GlobalStats {
            mean: global_mean,
            std: global_std,
            min: global_min,
            max: global_max,
            energy: all_values.iter().map(|&x| x.powi(2)).sum::<f32>(),
            rms: (all_values.iter().map(|&x| x.powi(2)).sum::<f32>() / all_values.len() as f32).sqrt(),
        };
        
        Ok(Self {
            mean,
            std,
            min,
            max,
            global,
        })
    }
}

/// Global spectral statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStats {
    /// Global mean
    pub mean: f32,
    /// Global standard deviation
    pub std: f32,
    /// Global minimum
    pub min: f32,
    /// Global maximum
    pub max: f32,
    /// Total energy
    pub energy: f32,
    /// Root mean square
    pub rms: f32,
}

// Window function implementations
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let factor = 2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32;
            0.5 * (1.0 - factor.cos())
        })
        .collect()
}

fn hamming_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let factor = 2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32;
            0.54 - 0.46 * factor.cos()
        })
        .collect()
}

fn blackman_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let factor = 2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32;
            0.42 - 0.5 * factor.cos() + 0.08 * (2.0 * factor).cos()
        })
        .collect()
}

fn bartlett_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let n = length - 1;
            if i <= n / 2 {
                2.0 * i as f32 / n as f32
            } else {
                2.0 - 2.0 * i as f32 / n as f32
            }
        })
        .collect()
}

/// Convert frequency to mel scale
pub fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filter bank
pub fn create_mel_filterbank(
    n_mels: u32,
    n_freqs: u32,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
    norm: Option<MelNormalization>,
) -> Result<Vec<Vec<f32>>> {
    if n_mels == 0 {
        return Err(AcousticError::InputError("Number of mel filters must be > 0".to_string()));
    }
    if n_freqs == 0 {
        return Err(AcousticError::InputError("Number of frequency bins must be > 0".to_string()));
    }
    if fmin >= fmax {
        return Err(AcousticError::InputError("fmin must be < fmax".to_string()));
    }
    
    // Convert frequency range to mel scale
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    
    // Create equally spaced points in mel scale
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    
    // Convert back to frequency
    let freq_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();
    
    // Convert frequencies to FFT bin numbers
    let bin_points: Vec<f32> = freq_points
        .iter()
        .map(|&freq| freq * (n_freqs - 1) as f32 * 2.0 / sample_rate as f32)
        .collect();
    
    let mut filterbank = vec![vec![0.0; n_freqs as usize]; n_mels as usize];
    
    for i in 0..n_mels as usize {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];
        
        for j in 0..n_freqs as usize {
            let freq_bin = j as f32;
            
            if freq_bin >= left && freq_bin <= center {
                filterbank[i][j] = (freq_bin - left) / (center - left);
            } else if freq_bin >= center && freq_bin <= right {
                filterbank[i][j] = (right - freq_bin) / (right - center);
            }
        }
        
        // Apply normalization
        if let Some(norm_type) = norm {
            match norm_type {
                MelNormalization::Slaney => {
                    // Slaney normalization: divide by width of mel filter
                    let width = freq_points[i + 2] - freq_points[i];
                    if width > 0.0 {
                        for value in &mut filterbank[i] {
                            *value /= width;
                        }
                    }
                }
                MelNormalization::UnitArea => {
                    // Unit area normalization
                    let area: f32 = filterbank[i].iter().sum();
                    if area > 0.0 {
                        for value in &mut filterbank[i] {
                            *value /= area;
                        }
                    }
                }
                MelNormalization::None => {
                    // No normalization
                }
            }
        }
    }
    
    Ok(filterbank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_params_validation() {
        let params = MelParams::standard_22khz();
        assert!(params.validate().is_ok());
        
        let mut params = MelParams::standard_22khz();
        params.sample_rate = 0;
        assert!(params.validate().is_err());
        
        params.sample_rate = 22050;
        params.win_length = 4096;
        params.n_fft = 2048;
        assert!(params.validate().is_err());
        
        params.win_length = 1024;
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_mel_params_conversions() {
        let params = MelParams::standard_22khz();
        
        let frame = params.time_to_frame(1.0);
        let time = params.frame_to_time(frame);
        assert!((time - 1.0).abs() < 0.01);
        
        assert_eq!(params.n_freqs(), 1025);
        assert_eq!(params.effective_fmax(), 11025.0);
    }

    #[test]
    fn test_window_functions() {
        let length = 1024;
        
        let hann = WindowType::Hann.generate(length);
        assert_eq!(hann.len(), length);
        assert!((hann[0] - 0.0).abs() < 1e-6);
        assert!((hann[length - 1] - 0.0).abs() < 1e-6);
        
        let rect = WindowType::Rectangular.generate(length);
        assert_eq!(rect.len(), length);
        assert!(rect.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_mel_scale_conversion() {
        let freq = 1000.0;
        let mel = hz_to_mel(freq);
        let freq_back = mel_to_hz(mel);
        assert!((freq - freq_back).abs() < 1e-3);
        
        assert!(hz_to_mel(0.0) >= 0.0);
        assert!(mel_to_hz(0.0) >= 0.0);
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let filterbank = create_mel_filterbank(
            80, 1025, 22050, 0.0, 11025.0, Some(MelNormalization::Slaney)
        ).unwrap();
        
        assert_eq!(filterbank.len(), 80);
        assert_eq!(filterbank[0].len(), 1025);
        
        // Check that filters are non-negative
        for filter in &filterbank {
            for &value in filter {
                assert!(value >= 0.0);
            }
        }
    }

    #[test]
    fn test_mel_filterbank_validation() {
        let result = create_mel_filterbank(0, 1025, 22050, 0.0, 11025.0, None);
        assert!(result.is_err());
        
        let result = create_mel_filterbank(80, 0, 22050, 0.0, 11025.0, None);
        assert!(result.is_err());
        
        let result = create_mel_filterbank(80, 1025, 22050, 1000.0, 500.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mel_stats_computation() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let mel = MelSpectrogram::new(data, 22050, 256);
        
        let stats = MelStats::compute(&mel).unwrap();
        assert_eq!(stats.mean.len(), 2);
        assert_eq!(stats.std.len(), 2);
        assert_eq!(stats.min.len(), 2);
        assert_eq!(stats.max.len(), 2);
        
        assert!((stats.mean[0] - 2.0).abs() < 1e-6);
        assert!((stats.mean[1] - 5.0).abs() < 1e-6);
        assert!((stats.global.mean - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_window_type_string() {
        assert_eq!(WindowType::Hann.as_str(), "hann");
        assert_eq!(WindowType::Hamming.as_str(), "hamming");
        assert_eq!(WindowType::Blackman.as_str(), "blackman");
        assert_eq!(WindowType::Bartlett.as_str(), "bartlett");
        assert_eq!(WindowType::Rectangular.as_str(), "rectangular");
    }

    #[test]
    fn test_mel_normalization_string() {
        assert_eq!(MelNormalization::Slaney.as_str(), "slaney");
        assert_eq!(MelNormalization::None.as_str(), "none");
        assert_eq!(MelNormalization::UnitArea.as_str(), "unit_area");
    }
}