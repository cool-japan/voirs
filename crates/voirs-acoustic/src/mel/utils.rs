//! Mel spectrogram utilities and format conversions
//!
//! This module provides utility functions for mel spectrograms including
//! format conversions, validation, visualization helpers, and quality metrics.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::{Result, AcousticError, MelSpectrogram};
use super::{MelParams, MelStats, MelMetadata};

/// Mel spectrogram utilities
pub struct MelUtils;

impl MelUtils {
    /// Convert mel spectrogram to ndarray format
    #[cfg(feature = "ndarray")]
    pub fn to_ndarray(mel: &MelSpectrogram) -> Result<ndarray::Array2<f32>> {
        use ndarray::Array2;
        
        let mut array = Array2::zeros((mel.n_mels, mel.n_frames));
        
        for (i, channel) in mel.data.iter().enumerate() {
            for (j, &value) in channel.iter().enumerate() {
                array[[i, j]] = value;
            }
        }
        
        Ok(array)
    }
    
    /// Create mel spectrogram from ndarray
    #[cfg(feature = "ndarray")]
    pub fn from_ndarray(
        array: &ndarray::Array2<f32>,
        sample_rate: u32,
        hop_length: u32,
    ) -> Result<MelSpectrogram> {
        let (n_mels, n_frames) = array.dim();
        let mut data = vec![vec![0.0; n_frames]; n_mels];
        
        for i in 0..n_mels {
            for j in 0..n_frames {
                data[i][j] = array[[i, j]];
            }
        }
        
        Ok(MelSpectrogram::new(data, sample_rate, hop_length))
    }
    
    /// Convert mel spectrogram to flat vector (row-major order)
    pub fn to_flat_vector(mel: &MelSpectrogram) -> Vec<f32> {
        let mut flat = Vec::with_capacity(mel.n_mels * mel.n_frames);
        
        for channel in &mel.data {
            flat.extend_from_slice(channel);
        }
        
        flat
    }
    
    /// Create mel spectrogram from flat vector (row-major order)
    pub fn from_flat_vector(
        flat: &[f32],
        n_mels: usize,
        n_frames: usize,
        sample_rate: u32,
        hop_length: u32,
    ) -> Result<MelSpectrogram> {
        if flat.len() != n_mels * n_frames {
            return Err(AcousticError::InputError("Flat vector size mismatch".to_string()));
        }
        
        let mut data = vec![vec![0.0; n_frames]; n_mels];
        
        for (i, chunk) in flat.chunks(n_frames).enumerate() {
            if i < n_mels {
                data[i].copy_from_slice(chunk);
            }
        }
        
        Ok(MelSpectrogram::new(data, sample_rate, hop_length))
    }
    
    /// Save mel spectrogram to binary format
    pub fn save_binary<P: AsRef<Path>>(mel: &MelSpectrogram, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        writer.write_all(&(mel.n_mels as u32).to_le_bytes())
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writer.write_all(&(mel.n_frames as u32).to_le_bytes())
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writer.write_all(&mel.sample_rate.to_le_bytes())
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writer.write_all(&mel.hop_length.to_le_bytes())
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        
        // Write data
        for channel in &mel.data {
            for &value in channel {
                writer.write_all(&value.to_le_bytes())
                    .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Load mel spectrogram from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<MelSpectrogram> {
        use std::io::Read;
        
        let file = File::open(path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)
            .map_err(|e| AcousticError::ModelError(format!("Read error: {}", e)))?;
        let n_mels = u32::from_le_bytes(buffer) as usize;
        
        reader.read_exact(&mut buffer)
            .map_err(|e| AcousticError::ModelError(format!("Read error: {}", e)))?;
        let n_frames = u32::from_le_bytes(buffer) as usize;
        
        reader.read_exact(&mut buffer)
            .map_err(|e| AcousticError::ModelError(format!("Read error: {}", e)))?;
        let sample_rate = u32::from_le_bytes(buffer);
        
        reader.read_exact(&mut buffer)
            .map_err(|e| AcousticError::ModelError(format!("Read error: {}", e)))?;
        let hop_length = u32::from_le_bytes(buffer);
        
        // Read data
        let mut data = vec![vec![0.0; n_frames]; n_mels];
        for channel in &mut data {
            for value in channel {
                reader.read_exact(&mut buffer)
                    .map_err(|e| AcousticError::ModelError(format!("Read error: {}", e)))?;
                *value = f32::from_le_bytes(buffer);
            }
        }
        
        Ok(MelSpectrogram::new(data, sample_rate, hop_length))
    }
    
    /// Save mel spectrogram as CSV
    pub fn save_csv<P: AsRef<Path>>(mel: &MelSpectrogram, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        writeln!(writer, "# n_mels: {}", mel.n_mels)
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writeln!(writer, "# n_frames: {}", mel.n_frames)
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writeln!(writer, "# sample_rate: {}", mel.sample_rate)
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        writeln!(writer, "# hop_length: {}", mel.hop_length)
            .map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        
        // Write data (each row is a mel channel, each column is a time frame)
        for channel in &mel.data {
            for (i, &value) in channel.iter().enumerate() {
                if i > 0 {
                    write!(writer, ",").map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
                }
                write!(writer, "{:.6}", value).map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
            }
            writeln!(writer).map_err(|e| AcousticError::ModelError(format!("Write error: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Validate mel spectrogram for common issues
    pub fn validate(mel: &MelSpectrogram) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        
        // Check dimensions
        if mel.n_mels == 0 {
            report.errors.push("Number of mel channels is zero".to_string());
        }
        if mel.n_frames == 0 {
            report.errors.push("Number of frames is zero".to_string());
        }
        if mel.data.len() != mel.n_mels {
            report.errors.push("Data length doesn't match n_mels".to_string());
        }
        
        // Check data consistency
        let expected_frame_count = mel.n_frames;
        for (i, channel) in mel.data.iter().enumerate() {
            if channel.len() != expected_frame_count {
                report.errors.push(format!("Channel {} has {} frames, expected {}", i, channel.len(), expected_frame_count));
            }
        }
        
        // Check for NaN and infinite values
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut negative_count = 0;
        
        for channel in &mel.data {
            for &value in channel {
                if value.is_nan() {
                    nan_count += 1;
                } else if value.is_infinite() {
                    inf_count += 1;
                } else if value < 0.0 {
                    negative_count += 1;
                }
            }
        }
        
        if nan_count > 0 {
            report.warnings.push(format!("Found {} NaN values", nan_count));
        }
        if inf_count > 0 {
            report.warnings.push(format!("Found {} infinite values", inf_count));
        }
        if negative_count > 0 {
            report.warnings.push(format!("Found {} negative values (unusual for mel spectrograms)", negative_count));
        }
        
        // Check audio parameters
        if mel.sample_rate == 0 {
            report.errors.push("Sample rate is zero".to_string());
        }
        if mel.hop_length == 0 {
            report.errors.push("Hop length is zero".to_string());
        }
        
        // Check for silent channels
        for (i, channel) in mel.data.iter().enumerate() {
            let max_val = channel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            if max_val <= 0.0 {
                report.warnings.push(format!("Channel {} appears to be silent", i));
            }
        }
        
        // Check duration reasonableness
        let duration = mel.duration();
        if duration <= 0.0 {
            report.errors.push("Duration is zero or negative".to_string());
        } else if duration > 3600.0 {
            report.warnings.push(format!("Duration is very long: {:.2} seconds", duration));
        } else if duration < 0.1 {
            report.warnings.push(format!("Duration is very short: {:.3} seconds", duration));
        }
        
        Ok(report)
    }
    
    /// Compute quality metrics for mel spectrogram
    pub fn compute_quality_metrics(mel: &MelSpectrogram) -> Result<QualityMetrics> {
        let stats = MelStats::compute(mel)?;
        
        // Spectral centroid (center of mass of spectrum)
        let mut spectral_centroids = Vec::new();
        for frame_idx in 0..mel.n_frames {
            let mut weighted_sum = 0.0;
            let mut total_energy = 0.0;
            
            for mel_idx in 0..mel.n_mels {
                if frame_idx < mel.data[mel_idx].len() {
                    let energy = mel.data[mel_idx][frame_idx];
                    weighted_sum += (mel_idx as f32) * energy;
                    total_energy += energy;
                }
            }
            
            if total_energy > 0.0 {
                spectral_centroids.push(weighted_sum / total_energy);
            } else {
                spectral_centroids.push(0.0);
            }
        }
        
        let mean_spectral_centroid = spectral_centroids.iter().sum::<f32>() / spectral_centroids.len() as f32;
        
        // Spectral bandwidth
        let mut spectral_bandwidths = Vec::new();
        for (frame_idx, &centroid) in spectral_centroids.iter().enumerate() {
            let mut weighted_variance = 0.0;
            let mut total_energy = 0.0;
            
            for mel_idx in 0..mel.n_mels {
                if frame_idx < mel.data[mel_idx].len() {
                    let energy = mel.data[mel_idx][frame_idx];
                    weighted_variance += ((mel_idx as f32) - centroid).powi(2) * energy;
                    total_energy += energy;
                }
            }
            
            if total_energy > 0.0 {
                spectral_bandwidths.push((weighted_variance / total_energy).sqrt());
            } else {
                spectral_bandwidths.push(0.0);
            }
        }
        
        let mean_spectral_bandwidth = spectral_bandwidths.iter().sum::<f32>() / spectral_bandwidths.len() as f32;
        
        // Zero crossing rate equivalent (energy transitions)
        let mut energy_transitions = 0;
        for channel in &mel.data {
            for i in 1..channel.len() {
                if (channel[i] > 0.0 && channel[i-1] <= 0.0) || 
                   (channel[i] <= 0.0 && channel[i-1] > 0.0) {
                    energy_transitions += 1;
                }
            }
        }
        let zcr_equivalent = energy_transitions as f32 / (mel.n_mels * mel.n_frames.saturating_sub(1)) as f32;
        
        // Spectral rolloff (frequency below which 85% of energy is contained)
        let mut spectral_rolloffs = Vec::new();
        for frame_idx in 0..mel.n_frames {
            let mut cumulative_energy = 0.0;
            let mut total_energy: f32 = (0..mel.n_mels)
                .map(|mel_idx| if frame_idx < mel.data[mel_idx].len() { mel.data[mel_idx][frame_idx] } else { 0.0 })
                .sum();
            
            let threshold = total_energy * 0.85;
            let mut rolloff_bin = mel.n_mels;
            
            for mel_idx in 0..mel.n_mels {
                if frame_idx < mel.data[mel_idx].len() {
                    cumulative_energy += mel.data[mel_idx][frame_idx];
                    if cumulative_energy >= threshold {
                        rolloff_bin = mel_idx;
                        break;
                    }
                }
            }
            
            spectral_rolloffs.push(rolloff_bin as f32);
        }
        
        let mean_spectral_rolloff = spectral_rolloffs.iter().sum::<f32>() / spectral_rolloffs.len() as f32;
        
        // Signal-to-noise ratio estimate
        let signal_power = stats.global.energy;
        let noise_estimate = stats.global.min * (mel.n_mels * mel.n_frames) as f32;
        let snr_estimate = if noise_estimate > 0.0 {
            10.0 * (signal_power / noise_estimate).log10()
        } else {
            f32::INFINITY
        };
        
        // Dynamic range
        let dynamic_range = stats.global.max - stats.global.min;
        
        Ok(QualityMetrics {
            spectral_centroid: mean_spectral_centroid,
            spectral_bandwidth: mean_spectral_bandwidth,
            spectral_rolloff: mean_spectral_rolloff,
            zero_crossing_rate: zcr_equivalent,
            signal_to_noise_ratio: snr_estimate,
            dynamic_range,
            rms_energy: stats.global.rms,
        })
    }
    
    /// Compare two mel spectrograms and compute similarity metrics
    pub fn compare_spectrograms(mel1: &MelSpectrogram, mel2: &MelSpectrogram) -> Result<SimilarityMetrics> {
        if mel1.n_mels != mel2.n_mels {
            return Err(AcousticError::InputError("Mel spectrograms must have same number of mel channels".to_string()));
        }
        
        let min_frames = mel1.n_frames.min(mel2.n_frames);
        
        // Mean Squared Error
        let mut mse = 0.0;
        let mut mae = 0.0;
        let mut count = 0;
        
        for mel_idx in 0..mel1.n_mels {
            for frame_idx in 0..min_frames {
                if frame_idx < mel1.data[mel_idx].len() && frame_idx < mel2.data[mel_idx].len() {
                    let diff = mel1.data[mel_idx][frame_idx] - mel2.data[mel_idx][frame_idx];
                    mse += diff * diff;
                    mae += diff.abs();
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            mse /= count as f32;
            mae /= count as f32;
        }
        
        let rmse = mse.sqrt();
        
        // Peak Signal-to-Noise Ratio
        let max_val = mel1.data.iter()
            .flatten()
            .chain(mel2.data.iter().flatten())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let psnr = if mse > 0.0 {
            20.0 * max_val.log10() - 10.0 * mse.log10()
        } else {
            f32::INFINITY
        };
        
        // Structural Similarity Index (simplified)
        let stats1 = MelStats::compute(mel1)?;
        let stats2 = MelStats::compute(mel2)?;
        
        let mean1 = stats1.global.mean;
        let mean2 = stats2.global.mean;
        let var1 = stats1.global.std * stats1.global.std;
        let var2 = stats2.global.std * stats2.global.std;
        
        // Compute covariance
        let mut covariance = 0.0;
        count = 0;
        for mel_idx in 0..mel1.n_mels {
            for frame_idx in 0..min_frames {
                if frame_idx < mel1.data[mel_idx].len() && frame_idx < mel2.data[mel_idx].len() {
                    covariance += (mel1.data[mel_idx][frame_idx] - mean1) * 
                                 (mel2.data[mel_idx][frame_idx] - mean2);
                    count += 1;
                }
            }
        }
        if count > 0 {
            covariance /= count as f32;
        }
        
        // SSIM computation
        let c1 = 0.01 * max_val * 0.01 * max_val;
        let c2 = 0.03 * max_val * 0.03 * max_val;
        
        let ssim = ((2.0 * mean1 * mean2 + c1) * (2.0 * covariance + c2)) /
                   ((mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2));
        
        // Correlation coefficient
        let correlation = if stats1.global.std > 0.0 && stats2.global.std > 0.0 {
            covariance / (stats1.global.std * stats2.global.std)
        } else {
            0.0
        };
        
        Ok(SimilarityMetrics {
            mse,
            rmse,
            mae,
            psnr,
            ssim,
            correlation,
        })
    }
    
    /// Create visualization data for mel spectrogram (for plotting)
    pub fn create_visualization_data(mel: &MelSpectrogram) -> VisualizationData {
        let time_axis: Vec<f32> = (0..mel.n_frames)
            .map(|i| i as f32 * mel.hop_length as f32 / mel.sample_rate as f32)
            .collect();
        
        let mel_axis: Vec<f32> = (0..mel.n_mels)
            .map(|i| i as f32)
            .collect();
        
        VisualizationData {
            time_axis,
            mel_axis,
            data: mel.data.clone(),
            sample_rate: mel.sample_rate,
            hop_length: mel.hop_length,
        }
    }
}

/// Validation report for mel spectrograms
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Quality metrics for mel spectrograms
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Mean spectral centroid
    pub spectral_centroid: f32,
    /// Mean spectral bandwidth
    pub spectral_bandwidth: f32,
    /// Mean spectral rolloff
    pub spectral_rolloff: f32,
    /// Zero crossing rate equivalent
    pub zero_crossing_rate: f32,
    /// Signal-to-noise ratio estimate
    pub signal_to_noise_ratio: f32,
    /// Dynamic range
    pub dynamic_range: f32,
    /// RMS energy
    pub rms_energy: f32,
}

/// Similarity metrics between two mel spectrograms
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Peak Signal-to-Noise Ratio
    pub psnr: f32,
    /// Structural Similarity Index
    pub ssim: f32,
    /// Correlation coefficient
    pub correlation: f32,
}

/// Visualization data for mel spectrograms
#[derive(Debug, Clone)]
pub struct VisualizationData {
    /// Time axis (seconds)
    pub time_axis: Vec<f32>,
    /// Mel axis (mel bin indices)
    pub mel_axis: Vec<f32>,
    /// Mel spectrogram data
    pub data: Vec<Vec<f32>>,
    /// Sample rate
    pub sample_rate: u32,
    /// Hop length
    pub hop_length: u32,
}

impl VisualizationData {
    /// Get data at specific time and mel indices
    pub fn get_value(&self, mel_idx: usize, frame_idx: usize) -> Option<f32> {
        self.data.get(mel_idx)?.get(frame_idx).copied()
    }
    
    /// Get time range
    pub fn time_range(&self) -> (f32, f32) {
        let min_time = self.time_axis.first().copied().unwrap_or(0.0);
        let max_time = self.time_axis.last().copied().unwrap_or(0.0);
        (min_time, max_time)
    }
    
    /// Get mel range
    pub fn mel_range(&self) -> (f32, f32) {
        let min_mel = self.mel_axis.first().copied().unwrap_or(0.0);
        let max_mel = self.mel_axis.last().copied().unwrap_or(0.0);
        (min_mel, max_mel)
    }
    
    /// Get value range
    pub fn value_range(&self) -> (f32, f32) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for channel in &self.data {
            for &value in channel {
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }
        
        (min_val, max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_mel() -> MelSpectrogram {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![0.5, 1.0, 1.5, 2.0],
        ];
        MelSpectrogram::new(data, 22050, 256)
    }

    #[test]
    fn test_to_flat_vector() {
        let mel = create_test_mel();
        let flat = MelUtils::to_flat_vector(&mel);
        
        assert_eq!(flat.len(), mel.n_mels * mel.n_frames);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[4], 2.0);
        assert_eq!(flat[8], 0.5);
    }

    #[test]
    fn test_from_flat_vector() {
        let original = create_test_mel();
        let flat = MelUtils::to_flat_vector(&original);
        let reconstructed = MelUtils::from_flat_vector(
            &flat, 
            original.n_mels, 
            original.n_frames, 
            original.sample_rate, 
            original.hop_length
        ).unwrap();
        
        assert_eq!(reconstructed.n_mels, original.n_mels);
        assert_eq!(reconstructed.n_frames, original.n_frames);
        assert_eq!(reconstructed.data, original.data);
    }

    #[test]
    fn test_validate() {
        let mel = create_test_mel();
        let report = MelUtils::validate(&mel).unwrap();
        
        assert!(report.is_valid());
        // Note: There might be warnings about very short duration, which is expected for test data
    }

    #[test]
    fn test_validate_invalid() {
        // Create invalid mel spectrogram
        let mut mel = create_test_mel();
        mel.n_mels = 0;
        
        let report = MelUtils::validate(&mel).unwrap();
        assert!(!report.is_valid());
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_quality_metrics() {
        let mel = create_test_mel();
        let metrics = MelUtils::compute_quality_metrics(&mel).unwrap();
        
        assert!(metrics.spectral_centroid >= 0.0);
        assert!(metrics.spectral_bandwidth >= 0.0);
        assert!(metrics.rms_energy > 0.0);
        assert!(metrics.dynamic_range > 0.0);
    }

    #[test]
    fn test_compare_spectrograms() {
        let mel1 = create_test_mel();
        let mel2 = create_test_mel();
        
        let similarity = MelUtils::compare_spectrograms(&mel1, &mel2).unwrap();
        
        // Identical spectrograms should have perfect similarity
        assert!((similarity.mse).abs() < 1e-6);
        assert!((similarity.correlation - 1.0).abs() < 1e-6);
        assert!(similarity.psnr.is_infinite() || similarity.psnr > 100.0);
    }

    #[test]
    fn test_compare_different_spectrograms() {
        let mel1 = create_test_mel();
        let mut mel2 = create_test_mel();
        
        // Modify mel2 to make it different in a way that changes correlation
        for (i, channel) in mel2.data.iter_mut().enumerate() {
            for (j, value) in channel.iter_mut().enumerate() {
                *value = if (i + j) % 2 == 0 { *value * 2.0 } else { *value * 0.5 };
            }
        }
        
        let similarity = MelUtils::compare_spectrograms(&mel1, &mel2).unwrap();
        
        // Different spectrograms should have non-zero error
        assert!(similarity.mse > 0.0);
        assert!(similarity.mae > 0.0);
        assert!(similarity.correlation < 0.99); // Allow for some floating point precision issues
    }

    #[test]
    fn test_visualization_data() {
        let mel = create_test_mel();
        let viz_data = MelUtils::create_visualization_data(&mel);
        
        assert_eq!(viz_data.time_axis.len(), mel.n_frames);
        assert_eq!(viz_data.mel_axis.len(), mel.n_mels);
        assert_eq!(viz_data.data, mel.data);
        
        let (min_time, max_time) = viz_data.time_range();
        assert!(max_time > min_time);
        
        let (min_mel, max_mel) = viz_data.mel_range();
        assert!(max_mel > min_mel);
        
        let (min_val, max_val) = viz_data.value_range();
        assert!(max_val > min_val);
    }

    #[test]
    fn test_visualization_data_access() {
        let mel = create_test_mel();
        let viz_data = MelUtils::create_visualization_data(&mel);
        
        assert_eq!(viz_data.get_value(0, 0), Some(1.0));
        assert_eq!(viz_data.get_value(1, 1), Some(4.0));
        assert_eq!(viz_data.get_value(2, 3), Some(2.0));
        assert_eq!(viz_data.get_value(10, 0), None); // Out of bounds
    }

    #[test]
    fn test_binary_save_load() {
        let original = create_test_mel();
        let temp_path = std::env::temp_dir().join("test_mel.bin");
        
        // Save
        MelUtils::save_binary(&original, &temp_path).unwrap();
        
        // Load
        let loaded = MelUtils::load_binary(&temp_path).unwrap();
        
        // Compare
        assert_eq!(loaded.n_mels, original.n_mels);
        assert_eq!(loaded.n_frames, original.n_frames);
        assert_eq!(loaded.sample_rate, original.sample_rate);
        assert_eq!(loaded.hop_length, original.hop_length);
        assert_eq!(loaded.data, original.data);
        
        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_csv_save() {
        let mel = create_test_mel();
        let temp_path = std::env::temp_dir().join("test_mel.csv");
        
        // Save
        MelUtils::save_csv(&mel, &temp_path).unwrap();
        
        // Check file exists
        assert!(temp_path.exists());
        
        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_flat_vector_error() {
        let flat = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = MelUtils::from_flat_vector(&flat, 2, 2, 22050, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_incompatible_spectrograms() {
        let mel1 = create_test_mel();
        let mut mel2 = create_test_mel();
        mel2.data.push(vec![1.0; mel2.n_frames]); // Add extra channel
        mel2.n_mels += 1;
        
        let result = MelUtils::compare_spectrograms(&mel1, &mel2);
        assert!(result.is_err());
    }
}