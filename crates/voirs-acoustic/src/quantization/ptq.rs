//! Post-training quantization (PTQ) implementation
//!
//! This module provides post-training quantization techniques that can be applied
//! to already trained acoustic models without requiring retraining.

use std::collections::HashMap;

use super::{QuantizationConfig, QuantizationParams, QuantizationPrecision};
use crate::{AcousticError, Result};

/// Post-training quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtqMethod {
    /// MinMax quantization using min/max values
    MinMax,
    /// Percentile-based quantization (reduces outlier impact)
    Percentile,
    /// K-means clustering for optimal quantization levels
    KMeans,
    /// Entropy-based calibration
    Entropy,
}

/// Post-training quantization configuration
#[derive(Debug, Clone)]
pub struct PtqConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// PTQ-specific method
    pub method: PtqMethod,
    /// Percentile for outlier clipping (used with Percentile method)
    pub percentile: f32,
    /// Number of clusters for K-means (used with KMeans method)
    pub num_clusters: usize,
    /// Batch size for calibration
    pub batch_size: usize,
}

impl Default for PtqConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            method: PtqMethod::MinMax,
            percentile: 99.9,  // 99.9th percentile
            num_clusters: 256, // For 8-bit quantization
            batch_size: 32,
        }
    }
}

/// Post-training quantizer
pub struct PostTrainingQuantizer {
    /// Configuration
    config: PtqConfig,
    /// Collected statistics per layer
    layer_stats: HashMap<String, LayerStatistics>,
}

/// Statistics collected for a layer during calibration
#[derive(Debug, Clone)]
pub struct LayerStatistics {
    /// All observed values (for histogram-based methods)
    values: Vec<f32>,
    /// Running minimum
    min_val: f32,
    /// Running maximum
    max_val: f32,
    /// Sample count
    sample_count: usize,
}

impl LayerStatistics {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            sample_count: 0,
        }
    }

    fn update(&mut self, values: &[f32]) {
        for &val in values {
            if val.is_finite() {
                self.values.push(val);
                self.min_val = self.min_val.min(val);
                self.max_val = self.max_val.max(val);
                self.sample_count += 1;
            }
        }
    }

    fn percentile(&self, p: f32) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((p / 100.0) * (sorted.len() - 1) as f32) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl PostTrainingQuantizer {
    /// Create new post-training quantizer
    pub fn new(config: PtqConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
        }
    }

    /// Add calibration data for a layer
    pub fn add_calibration_data(&mut self, layer_name: &str, data: &[f32]) -> Result<()> {
        let stats = self
            .layer_stats
            .entry(layer_name.to_string())
            .or_insert_with(LayerStatistics::new);
        stats.update(data);
        Ok(())
    }

    /// Calculate quantization parameters using the specified method
    pub fn calculate_quantization_params(&self, layer_name: &str) -> Result<QuantizationParams> {
        let stats = self.layer_stats.get(layer_name).ok_or_else(|| {
            AcousticError::Processing(format!("No statistics found for layer: {layer_name}"))
        })?;

        match self.config.method {
            PtqMethod::MinMax => self.calculate_minmax_params(stats),
            PtqMethod::Percentile => self.calculate_percentile_params(stats),
            PtqMethod::KMeans => self.calculate_kmeans_params(stats),
            PtqMethod::Entropy => self.calculate_entropy_params(stats),
        }
    }

    /// MinMax quantization parameters
    fn calculate_minmax_params(&self, stats: &LayerStatistics) -> Result<QuantizationParams> {
        let (qmin, qmax) = self.get_quantization_range();

        if self.config.base.symmetric {
            let abs_max = stats.max_val.abs().max(stats.min_val.abs());
            let scale = abs_max / (qmax as f32);
            Ok(QuantizationParams::symmetric(scale, qmin, qmax))
        } else {
            let scale = (stats.max_val - stats.min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (stats.min_val / scale).round() as i32;
            Ok(QuantizationParams::asymmetric(
                scale, zero_point, qmin, qmax,
            ))
        }
    }

    /// Percentile-based quantization parameters
    fn calculate_percentile_params(&self, stats: &LayerStatistics) -> Result<QuantizationParams> {
        let (qmin, qmax) = self.get_quantization_range();

        let min_percentile = 100.0 - self.config.percentile;
        let min_val = stats.percentile(min_percentile);
        let max_val = stats.percentile(self.config.percentile);

        if self.config.base.symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / (qmax as f32);
            Ok(QuantizationParams::symmetric(scale, qmin, qmax))
        } else {
            let scale = (max_val - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            Ok(QuantizationParams::asymmetric(
                scale, zero_point, qmin, qmax,
            ))
        }
    }

    /// K-means clustering for quantization levels
    fn calculate_kmeans_params(&self, stats: &LayerStatistics) -> Result<QuantizationParams> {
        // Simplified K-means implementation
        let (qmin, qmax) = self.get_quantization_range();
        let num_levels = (qmax - qmin + 1) as usize;

        if stats.values.is_empty() {
            return Err(AcousticError::Processing(
                "No data for K-means clustering".to_string(),
            ));
        }

        // Initialize centroids uniformly across the range
        let min_val = stats.min_val;
        let max_val = stats.max_val;
        let step = (max_val - min_val) / (num_levels - 1) as f32;

        let mut centroids: Vec<f32> = (0..num_levels).map(|i| min_val + i as f32 * step).collect();

        // Run simplified K-means for a few iterations
        for _ in 0..10 {
            let mut new_centroids = vec![0.0; num_levels];
            let mut counts = vec![0; num_levels];

            // Assign points to nearest centroids
            for &value in &stats.values {
                let mut best_idx = 0;
                let mut best_dist = (value - centroids[0]).abs();

                for (i, &centroid) in centroids.iter().enumerate() {
                    let dist = (value - centroid).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }

                new_centroids[best_idx] += value;
                counts[best_idx] += 1;
            }

            // Update centroids
            for i in 0..num_levels {
                if counts[i] > 0 {
                    centroids[i] = new_centroids[i] / counts[i] as f32;
                }
            }
        }

        // Use the centroids to determine quantization scale
        let scale = (centroids[num_levels - 1] - centroids[0]) / (qmax - qmin) as f32;
        let zero_point = qmin - (centroids[0] / scale).round() as i32;

        Ok(QuantizationParams::asymmetric(
            scale, zero_point, qmin, qmax,
        ))
    }

    /// Entropy-based calibration
    fn calculate_entropy_params(&self, stats: &LayerStatistics) -> Result<QuantizationParams> {
        // Simplified entropy-based method
        // In practice, this would involve more sophisticated histogram analysis
        let (qmin, qmax) = self.get_quantization_range();

        // Create histogram
        let num_bins = 1024;
        let min_val = stats.min_val;
        let max_val = stats.max_val;
        let bin_width = (max_val - min_val) / num_bins as f32;

        let mut histogram = vec![0u32; num_bins];
        for &value in &stats.values {
            let bin = ((value - min_val) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }

        // Find optimal threshold using KL divergence (simplified)
        let mut best_threshold = max_val;
        let mut best_kl_div = f32::INFINITY;

        for threshold_bin in (num_bins / 2)..num_bins {
            let threshold = min_val + threshold_bin as f32 * bin_width;
            let kl_div = self.calculate_kl_divergence(&histogram, threshold_bin);

            if kl_div < best_kl_div {
                best_kl_div = kl_div;
                best_threshold = threshold;
            }
        }

        // Use the optimal threshold
        if self.config.base.symmetric {
            let abs_max = best_threshold.abs().max(min_val.abs());
            let scale = abs_max / (qmax as f32);
            Ok(QuantizationParams::symmetric(scale, qmin, qmax))
        } else {
            let scale = (best_threshold - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            Ok(QuantizationParams::asymmetric(
                scale, zero_point, qmin, qmax,
            ))
        }
    }

    /// Calculate simplified KL divergence for threshold selection
    fn calculate_kl_divergence(&self, histogram: &[u32], threshold_bin: usize) -> f32 {
        let total_count: u32 = histogram.iter().sum();
        if total_count == 0 {
            return f32::INFINITY;
        }

        let mut kl_div = 0.0;
        let num_quant_bins = self.get_quantization_range().1 - self.get_quantization_range().0 + 1;

        for (i, &hist_val) in histogram.iter().enumerate().take(threshold_bin) {
            let original_prob = hist_val as f32 / total_count as f32;
            if original_prob > 0.0 {
                let _quant_bin = (i * num_quant_bins as usize) / threshold_bin;
                let quant_prob = 1.0 / num_quant_bins as f32; // Uniform quantization
                kl_div += original_prob * (original_prob / quant_prob).ln();
            }
        }

        kl_div
    }

    /// Get quantization range for current precision
    fn get_quantization_range(&self) -> (i32, i32) {
        match self.config.base.precision {
            QuantizationPrecision::Int8 => (-128, 127),
            QuantizationPrecision::Int16 => (-32768, 32767),
            QuantizationPrecision::Int4 => (-8, 7),
            QuantizationPrecision::Mixed => (-128, 127), // Default to Int8
        }
    }

    /// Quantize all layers using calibrated parameters
    pub fn quantize_model(&self) -> Result<HashMap<String, QuantizationParams>> {
        let mut quantized_params = HashMap::new();

        for layer_name in self.layer_stats.keys() {
            if self.config.base.skip_layers.contains(layer_name) {
                continue;
            }

            let params = self.calculate_quantization_params(layer_name)?;
            quantized_params.insert(layer_name.clone(), params);
        }

        Ok(quantized_params)
    }

    /// Get calibration statistics for a layer
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerStatistics> {
        self.layer_stats.get(layer_name)
    }

    /// Get number of calibrated layers
    pub fn num_calibrated_layers(&self) -> usize {
        self.layer_stats.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_statistics() {
        let mut stats = LayerStatistics::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        stats.update(&data);

        assert_eq!(stats.min_val, 1.0);
        assert_eq!(stats.max_val, 5.0);
        assert_eq!(stats.sample_count, 5);
        assert_eq!(stats.percentile(50.0), 3.0); // Median
    }

    #[test]
    fn test_ptq_minmax() {
        let config = PtqConfig::default();
        let mut quantizer = PostTrainingQuantizer::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        quantizer.add_calibration_data("layer1", &data).unwrap();

        let params = quantizer.calculate_quantization_params("layer1").unwrap();
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_ptq_percentile() {
        let config = PtqConfig {
            method: PtqMethod::Percentile,
            percentile: 95.0,
            ..Default::default()
        };

        let mut quantizer = PostTrainingQuantizer::new(config);

        // Add data with outliers
        let mut data = vec![1.0; 95];
        data.extend(vec![100.0; 5]); // 5% outliers

        quantizer.add_calibration_data("layer1", &data).unwrap();

        let params = quantizer.calculate_quantization_params("layer1").unwrap();
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_ptq_kmeans() {
        let config = PtqConfig {
            method: PtqMethod::KMeans,
            ..Default::default()
        };

        let mut quantizer = PostTrainingQuantizer::new(config);

        let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2];
        quantizer.add_calibration_data("layer1", &data).unwrap();

        let params = quantizer.calculate_quantization_params("layer1").unwrap();
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quantize_model() {
        let config = PtqConfig::default();
        let mut quantizer = PostTrainingQuantizer::new(config);

        quantizer
            .add_calibration_data("layer1", &[1.0, 2.0, 3.0])
            .unwrap();
        quantizer
            .add_calibration_data("layer2", &[4.0, 5.0, 6.0])
            .unwrap();

        let quantized = quantizer.quantize_model().unwrap();
        assert_eq!(quantized.len(), 2);
        assert!(quantized.contains_key("layer1"));
        assert!(quantized.contains_key("layer2"));
    }
}
