//! Calibration utilities for quantization
//!
//! This module provides calibration datasets and utilities for determining
//! optimal quantization parameters for neural acoustic models.

use crate::{AcousticError, Result};
use std::collections::HashMap;

/// Calibration dataset for quantization parameter estimation
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    /// Layer name to sample data mapping
    pub data: HashMap<String, Vec<f32>>,
    /// Number of calibration samples
    pub sample_count: usize,
}

impl CalibrationDataset {
    /// Create new calibration dataset
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            sample_count: 0,
        }
    }

    /// Add calibration samples for a layer
    pub fn add_samples(&mut self, layer_name: String, samples: Vec<f32>) {
        self.data.insert(layer_name, samples);
        self.sample_count += 1;
    }

    /// Get samples for a layer
    pub fn get_samples(&self, layer_name: &str) -> Option<&Vec<f32>> {
        self.data.get(layer_name)
    }

    /// Check if dataset has samples for a layer
    pub fn has_layer(&self, layer_name: &str) -> bool {
        self.data.contains_key(layer_name)
    }

    /// Get all layer names
    pub fn layer_names(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for CalibrationDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibration utilities
pub mod utils {
    use super::*;

    /// Generate synthetic calibration data for testing
    pub fn generate_synthetic_calibration_data(
        layer_names: &[&str],
        sample_count: usize,
    ) -> CalibrationDataset {
        let mut dataset = CalibrationDataset::new();

        for &layer_name in layer_names {
            let samples: Vec<f32> = (0..sample_count)
                .map(|i| {
                    // Generate synthetic data with some realistic distribution
                    let x = (i as f32) / (sample_count as f32) * 2.0 * std::f32::consts::PI;
                    x.sin() * 0.5 + (x * 2.0).cos() * 0.3
                })
                .collect();

            dataset.add_samples(layer_name.to_string(), samples);
        }

        dataset
    }

    /// Validate calibration dataset
    pub fn validate_calibration_dataset(dataset: &CalibrationDataset) -> Result<()> {
        if dataset.data.is_empty() {
            return Err(AcousticError::Processing(
                "Calibration dataset is empty".to_string(),
            ));
        }

        for (layer_name, samples) in &dataset.data {
            if samples.is_empty() {
                return Err(AcousticError::Processing(format!(
                    "No calibration samples for layer: {layer_name}"
                )));
            }

            // Check for invalid values
            for &sample in samples {
                if !sample.is_finite() {
                    return Err(AcousticError::Processing(format!(
                        "Invalid calibration sample in layer: {layer_name}"
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_dataset_creation() {
        let mut dataset = CalibrationDataset::new();
        let samples = vec![1.0, 2.0, 3.0];

        dataset.add_samples("layer1".to_string(), samples.clone());

        assert!(dataset.has_layer("layer1"));
        assert_eq!(dataset.get_samples("layer1"), Some(&samples));
        assert_eq!(dataset.layer_names(), vec!["layer1"]);
    }

    #[test]
    fn test_synthetic_calibration_data() {
        let layer_names = vec!["layer1", "layer2"];
        let dataset = utils::generate_synthetic_calibration_data(&layer_names, 100);

        assert_eq!(dataset.layer_names().len(), 2);
        assert!(dataset.has_layer("layer1"));
        assert!(dataset.has_layer("layer2"));

        let samples = dataset.get_samples("layer1").unwrap();
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_dataset_validation() {
        let mut dataset = CalibrationDataset::new();
        dataset.add_samples("layer1".to_string(), vec![1.0, 2.0, 3.0]);

        assert!(utils::validate_calibration_dataset(&dataset).is_ok());

        // Test empty dataset
        let empty_dataset = CalibrationDataset::new();
        assert!(utils::validate_calibration_dataset(&empty_dataset).is_err());
    }
}
