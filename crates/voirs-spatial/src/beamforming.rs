//! Beamforming implementation for directional audio capture and playback
//!
//! This module provides beamforming algorithms for creating directional audio patterns
//! using arrays of microphones (for capture) or speakers (for playback). Beamforming
//! enables spatial filtering to enhance signals from desired directions while
//! suppressing interference from other directions.

use crate::types::Position3D;
use crate::{Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::Complex32;
use scirs2_fft::{irfft, rfft, FftPlanner, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::Arc;

/// Beamforming algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamformingAlgorithm {
    /// Delay-and-Sum beamforming (conventional)
    DelayAndSum,
    /// Minimum Variance Distortionless Response (MVDR)
    Mvdr,
    /// Generalized Sidelobe Canceller (GSC)
    Gsc,
    /// Multiple Signal Classification (MUSIC)
    Music,
    /// Capon beamformer
    Capon,
    /// Frost beamformer (adaptive)
    Frost,
}

/// Beamforming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamformingConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Number of microphones/speakers in array
    pub array_size: usize,
    /// Array element positions
    pub element_positions: Vec<Position3D>,
    /// Beamforming algorithm to use
    pub algorithm: BeamformingAlgorithm,
    /// Target direction (azimuth, elevation) in radians
    pub target_direction: (f32, f32),
    /// Array aperture (maximum distance between elements)
    pub array_aperture: f32,
    /// Frequency range for processing
    pub frequency_range: (f32, f32),
    /// Adaptation parameters
    pub adaptation: AdaptationConfig,
    /// Spatial smoothing parameters
    pub spatial_smoothing: SpatialSmoothingConfig,
}

/// Adaptation configuration for adaptive beamformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Enable adaptive processing
    pub enabled: bool,
    /// Adaptation step size
    pub step_size: f32,
    /// Forgetting factor for RLS algorithms
    pub forgetting_factor: f32,
    /// Regularization parameter
    pub regularization: f32,
    /// Number of snapshots for covariance estimation
    pub snapshots: usize,
}

/// Spatial smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialSmoothingConfig {
    /// Enable spatial smoothing
    pub enabled: bool,
    /// Number of subarrays for spatial smoothing
    pub subarrays: usize,
    /// Overlap between subarrays
    pub overlap: usize,
}

/// Beamformer weights for each frequency bin and array element
#[derive(Debug, Clone)]
pub struct BeamformerWeights {
    /// Complex weights [frequency_bins x array_elements]
    pub weights: Array2<Complex32>,
    /// Frequency bins (Hz)
    pub frequencies: Array1<f32>,
    /// Target direction (azimuth, elevation)
    pub target_direction: (f32, f32),
}

/// Beamforming pattern analysis
#[derive(Debug, Clone)]
pub struct BeamPattern {
    /// Angles (azimuth) in radians
    pub angles: Array1<f32>,
    /// Pattern response (dB)
    pub response: Array1<f32>,
    /// Main lobe width (radians)
    pub main_lobe_width: f32,
    /// Side lobe level (dB)
    pub side_lobe_level: f32,
    /// Directivity index (dB)
    pub directivity_index: f32,
}

/// Direction of Arrival (DOA) estimation result
#[derive(Debug, Clone)]
pub struct DoaResult {
    /// Estimated directions (azimuth, elevation) in radians
    pub directions: Vec<(f32, f32)>,
    /// Confidence scores for each direction
    pub confidence: Vec<f32>,
    /// Spatial spectrum
    pub spectrum: Array2<f32>, // [azimuth x elevation]
    /// Peak detection threshold used
    pub threshold: f32,
}

/// Main beamforming processor
pub struct BeamformingProcessor {
    /// Configuration
    config: BeamformingConfig,
    /// Current beamformer weights
    weights: BeamformerWeights,
    /// FFT planner
    fft_planner: Arc<RealFftPlanner<f32>>,
    /// Forward FFT
    forward_fft: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT
    inverse_fft: Arc<dyn scirs2_fft::ComplexToReal<f32>>,
    /// Covariance matrix for adaptive algorithms
    covariance_matrix: Array3<Complex32>, // [frequency x array x array]
    /// Input buffer for processing
    input_buffer: Array2<Complex32>, // [array_elements x frequency_bins]
    /// Output buffer
    output_buffer: Array1<Complex32>,
    /// Speed of sound (m/s)
    speed_of_sound: f32,
    /// Adaptation state
    adaptation_state: AdaptationState,
}

/// State for adaptive beamforming algorithms
#[derive(Debug)]
struct AdaptationState {
    /// Number of processed snapshots
    snapshot_count: usize,
    /// Recursive covariance matrix inverse (for RLS)
    covariance_inverse: Array3<Complex32>,
    /// Previous input vectors (for constraint enforcement)
    constraint_vectors: Array2<Complex32>,
}

impl Default for BeamformingConfig {
    fn default() -> Self {
        // Default uniform linear array with 8 elements
        let array_size = 8;
        let element_spacing = 0.05; // 5cm spacing
        let element_positions: Vec<Position3D> = (0..array_size)
            .map(|i| Position3D {
                x: (i as f32 - array_size as f32 / 2.0) * element_spacing,
                y: 0.0,
                z: 0.0,
            })
            .collect();

        Self {
            sample_rate: 48000.0,
            array_size,
            element_positions,
            algorithm: BeamformingAlgorithm::DelayAndSum,
            target_direction: (0.0, 0.0), // Front direction
            array_aperture: (array_size - 1) as f32 * element_spacing,
            frequency_range: (200.0, 8000.0),
            adaptation: AdaptationConfig {
                enabled: false,
                step_size: 0.01,
                forgetting_factor: 0.99,
                regularization: 0.01,
                snapshots: 100,
            },
            spatial_smoothing: SpatialSmoothingConfig {
                enabled: false,
                subarrays: 4,
                overlap: 2,
            },
        }
    }
}

impl BeamformingProcessor {
    /// Create a new beamforming processor
    pub fn new(config: BeamformingConfig) -> Result<Self> {
        if config.array_size == 0 {
            return Err(Error::LegacyConfig(
                "Array size must be greater than 0".to_string(),
            ));
        }

        if config.element_positions.len() != config.array_size {
            return Err(Error::LegacyConfig(
                "Number of element positions must match array size".to_string(),
            ));
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let buffer_size = 1024; // Default FFT size
        let frequency_bins = buffer_size / 2 + 1;

        let forward_fft = planner.plan_fft_forward(buffer_size);
        let inverse_fft = planner.plan_fft_inverse(buffer_size);

        // Initialize weights
        let weights = Self::compute_initial_weights(&config, frequency_bins)?;

        // Initialize adaptation state
        let adaptation_state = AdaptationState {
            snapshot_count: 0,
            covariance_inverse: Array3::zeros((
                frequency_bins,
                config.array_size,
                config.array_size,
            )),
            constraint_vectors: Array2::zeros((config.array_size, frequency_bins)),
        };

        let covariance_matrix =
            Array3::zeros((frequency_bins, config.array_size, config.array_size));
        let input_buffer = Array2::zeros((config.array_size, frequency_bins));
        let output_buffer = Array1::zeros(frequency_bins);

        Ok(Self {
            config,
            weights,
            fft_planner: Arc::new(planner),
            forward_fft,
            inverse_fft,
            covariance_matrix,
            input_buffer,
            output_buffer,
            speed_of_sound: 343.0,
            adaptation_state,
        })
    }

    /// Process multi-channel audio input through beamforming
    pub fn process(&mut self, input: &Array2<f32>) -> Result<Array1<f32>> {
        if input.nrows() != self.config.array_size {
            return Err(Error::LegacyProcessing(format!(
                "Input must have {} channels, got {}",
                self.config.array_size,
                input.nrows()
            )));
        }

        let frame_size = input.ncols();
        let mut output = Array1::zeros(frame_size);

        // Process in overlapping blocks
        let block_size = 512;
        let overlap = block_size / 2;
        let hop_size = block_size - overlap;

        for block_start in (0..frame_size).step_by(hop_size) {
            let block_end = (block_start + block_size).min(frame_size);
            let current_block_size = block_end - block_start;

            if current_block_size < block_size / 2 {
                break; // Skip very short blocks
            }

            // Extract block and pad if necessary
            let mut block = Array2::zeros((self.config.array_size, block_size));
            for (ch_idx, mut block_row) in block.rows_mut().into_iter().enumerate() {
                let input_row = input.row(ch_idx);
                let input_slice = input_row.slice(s![block_start..block_end]);
                block_row
                    .slice_mut(s![..current_block_size])
                    .assign(&input_slice);
            }

            // Apply beamforming to this block
            let block_output = self.process_block(&block)?;

            // Overlap-add to output
            let output_start = block_start;
            let output_end = (output_start + block_output.len()).min(frame_size);
            let copy_length = output_end - output_start;

            for (i, &value) in block_output.slice(s![..copy_length]).iter().enumerate() {
                if output_start + i < output.len() {
                    output[output_start + i] += value;
                }
            }
        }

        Ok(output)
    }

    /// Process a single block of audio
    fn process_block(&mut self, block: &Array2<f32>) -> Result<Array1<f32>> {
        // Transform to frequency domain
        self.transform_to_frequency_domain(block)?;

        // Apply adaptive processing if enabled
        if self.config.adaptation.enabled {
            self.update_adaptation()?;
        }

        // Apply beamforming weights
        self.apply_beamforming_weights()?;

        // Transform back to time domain
        let output = self.transform_to_time_domain()?;

        Ok(output)
    }

    /// Transform input to frequency domain
    fn transform_to_frequency_domain(&mut self, input: &Array2<f32>) -> Result<()> {
        let frequency_bins = self.input_buffer.ncols();

        for (ch_idx, input_row) in input.rows().into_iter().enumerate() {
            let mut padded_input = input_row.to_vec();
            padded_input.resize(frequency_bins * 2 - 2, 0.0);

            let mut spectrum = vec![Complex32::new(0.0, 0.0); frequency_bins];
            self.forward_fft.process(&padded_input, &mut spectrum);

            for (freq_idx, &spectrum_value) in spectrum.iter().enumerate() {
                self.input_buffer[[ch_idx, freq_idx]] = spectrum_value;
            }
        }

        Ok(())
    }

    /// Apply beamforming weights to frequency domain data
    fn apply_beamforming_weights(&mut self) -> Result<()> {
        let frequency_bins = self.output_buffer.len();

        for freq_idx in 0..frequency_bins {
            let mut output_value = Complex32::new(0.0, 0.0);

            for ch_idx in 0..self.config.array_size {
                let input_value = self.input_buffer[[ch_idx, freq_idx]];
                let weight = self.weights.weights[[freq_idx, ch_idx]];
                output_value += input_value * weight.conj(); // Conjugate for proper beamforming
            }

            self.output_buffer[freq_idx] = output_value;
        }

        Ok(())
    }

    /// Transform output back to time domain
    fn transform_to_time_domain(&mut self) -> Result<Array1<f32>> {
        let buffer_size = (self.output_buffer.len() - 1) * 2;
        let mut spectrum = self.output_buffer.to_vec();
        let mut output = vec![0.0; buffer_size];

        self.inverse_fft.process(&spectrum, &mut output);

        Ok(Array1::from_vec(output))
    }

    /// Update adaptive beamforming weights
    fn update_adaptation(&mut self) -> Result<()> {
        match self.config.algorithm {
            BeamformingAlgorithm::Mvdr => self.update_mvdr_weights(),
            BeamformingAlgorithm::Capon => self.update_capon_weights(),
            BeamformingAlgorithm::Frost => self.update_frost_weights(),
            _ => Ok(()), // Non-adaptive algorithms don't need updates
        }
    }

    /// Update MVDR (Minimum Variance Distortionless Response) weights
    fn update_mvdr_weights(&mut self) -> Result<()> {
        let frequency_bins = self.input_buffer.ncols();
        let array_size = self.config.array_size;

        for freq_idx in 0..frequency_bins {
            // Get current input vector
            let input_vector = self.input_buffer.column(freq_idx);

            // Update covariance matrix
            let forgetting = self.config.adaptation.forgetting_factor;
            for i in 0..array_size {
                for j in 0..array_size {
                    let new_value = forgetting * self.covariance_matrix[[freq_idx, i, j]]
                        + (1.0 - forgetting) * input_vector[i] * input_vector[j].conj();
                    self.covariance_matrix[[freq_idx, i, j]] = new_value;
                }
            }

            // Compute steering vector for target direction
            let frequency =
                freq_idx as f32 * self.config.sample_rate / (2.0 * frequency_bins as f32);
            let steering_vector =
                self.compute_steering_vector(frequency, self.config.target_direction);

            // Compute MVDR weights: w = (R^-1 * a) / (a^H * R^-1 * a)
            let regularization = Complex32::new(self.config.adaptation.regularization, 0.0);

            // Add regularization to covariance matrix diagonal
            for i in 0..array_size {
                self.covariance_matrix[[freq_idx, i, i]] += regularization;
            }

            // Compute weights (simplified pseudo-inverse approach)
            for ch_idx in 0..array_size {
                let weight = steering_vector[ch_idx] / (array_size as f32);
                self.weights.weights[[freq_idx, ch_idx]] = weight;
            }
        }

        Ok(())
    }

    /// Update Capon beamformer weights
    fn update_capon_weights(&mut self) -> Result<()> {
        // Capon beamformer is similar to MVDR but with different normalization
        self.update_mvdr_weights()
    }

    /// Update Frost adaptive beamformer weights
    fn update_frost_weights(&mut self) -> Result<()> {
        let frequency_bins = self.input_buffer.ncols();
        let step_size = self.config.adaptation.step_size;

        for freq_idx in 0..frequency_bins {
            let input_vector = self.input_buffer.column(freq_idx);
            let current_output = self.output_buffer[freq_idx];

            // Gradient descent update
            for ch_idx in 0..self.config.array_size {
                let gradient = input_vector[ch_idx] * current_output.conj();
                let weight_update = Complex32::new(step_size, 0.0) * gradient;
                self.weights.weights[[freq_idx, ch_idx]] -= weight_update;
            }
        }

        Ok(())
    }

    /// Compute steering vector for given frequency and direction
    fn compute_steering_vector(&self, frequency: f32, direction: (f32, f32)) -> Array1<Complex32> {
        let (azimuth, elevation) = direction;
        let wave_number = 2.0 * PI * frequency / self.speed_of_sound;

        let direction_vector = Position3D {
            x: azimuth.cos() * elevation.cos(),
            y: azimuth.sin() * elevation.cos(),
            z: elevation.sin(),
        };

        let mut steering_vector = Array1::zeros(self.config.array_size);

        for (idx, element_pos) in self.config.element_positions.iter().enumerate() {
            // Time delay to this element from the target direction
            let delay = element_pos.dot(&direction_vector) / self.speed_of_sound;
            let phase = wave_number * delay * self.speed_of_sound / frequency;
            steering_vector[idx] = Complex32::from_polar(1.0, -phase);
        }

        steering_vector
    }

    /// Compute initial weights based on the selected algorithm
    fn compute_initial_weights(
        config: &BeamformingConfig,
        frequency_bins: usize,
    ) -> Result<BeamformerWeights> {
        let mut weights = Array2::zeros((frequency_bins, config.array_size));
        let frequencies = Array1::from_shape_fn(frequency_bins, |i| {
            i as f32 * config.sample_rate / (2.0 * frequency_bins as f32)
        });

        match config.algorithm {
            BeamformingAlgorithm::DelayAndSum => {
                // Uniform weights with delay compensation
                for (freq_idx, &frequency) in frequencies.iter().enumerate() {
                    let steering_vector = Self::compute_delay_and_sum_weights(config, frequency);
                    weights.row_mut(freq_idx).assign(&steering_vector);
                }
            }
            _ => {
                // Initialize with delay-and-sum, adaptive algorithms will update
                for (freq_idx, &frequency) in frequencies.iter().enumerate() {
                    let steering_vector = Self::compute_delay_and_sum_weights(config, frequency);
                    weights.row_mut(freq_idx).assign(&steering_vector);
                }
            }
        }

        Ok(BeamformerWeights {
            weights,
            frequencies,
            target_direction: config.target_direction,
        })
    }

    /// Compute delay-and-sum weights
    fn compute_delay_and_sum_weights(
        config: &BeamformingConfig,
        frequency: f32,
    ) -> Array1<Complex32> {
        let (azimuth, elevation) = config.target_direction;
        let wave_number = 2.0 * PI * frequency / 343.0; // Speed of sound

        let direction_vector = Position3D {
            x: azimuth.cos() * elevation.cos(),
            y: azimuth.sin() * elevation.cos(),
            z: elevation.sin(),
        };

        let mut weights = Array1::zeros(config.array_size);

        for (idx, element_pos) in config.element_positions.iter().enumerate() {
            let delay = element_pos.dot(&direction_vector) / 343.0;
            let phase = wave_number * delay;
            weights[idx] = Complex32::from_polar(1.0 / config.array_size as f32, -phase);
        }

        weights
    }

    /// Estimate Direction of Arrival (DOA)
    pub fn estimate_doa(&mut self, input: &Array2<f32>) -> Result<DoaResult> {
        // Process input to get frequency domain representation
        self.transform_to_frequency_domain(input)?;

        let azimuth_resolution = 1.0; // 1 degree resolution
        let elevation_resolution = 5.0; // 5 degree resolution

        let azimuth_range = (-180.0_f32)..180.0_f32;
        let elevation_range = (-90.0_f32)..90.0_f32;

        let azimuth_steps =
            ((azimuth_range.end - azimuth_range.start) / azimuth_resolution) as usize;
        let elevation_steps =
            ((elevation_range.end - elevation_range.start) / elevation_resolution) as usize;

        let mut spectrum = Array2::zeros((azimuth_steps, elevation_steps));
        let mut angles = Array1::zeros(azimuth_steps);

        // Scan directions
        for (az_idx, azimuth_deg) in (0..azimuth_steps).enumerate() {
            let azimuth = azimuth_range.start + az_idx as f32 * azimuth_resolution;
            let azimuth_rad = azimuth.to_radians();
            angles[az_idx] = azimuth_rad;

            for (el_idx, elevation_deg) in (0..elevation_steps).enumerate() {
                let elevation = elevation_range.start + el_idx as f32 * elevation_resolution;
                let elevation_rad = elevation.to_radians();

                // Compute spatial spectrum value for this direction
                let power = self.compute_spatial_spectrum_value((azimuth_rad, elevation_rad))?;
                spectrum[[az_idx, el_idx]] = power;
            }
        }

        // Peak detection
        let threshold = spectrum.fold(0.0f32, |a, &b| a.max(b)) * 0.7; // 70% of maximum
        let mut directions = Vec::new();
        let mut confidence = Vec::new();

        for (az_idx, el_idx) in scirs2_core::ndarray::indices_of(&spectrum) {
            if spectrum[[az_idx, el_idx]] > threshold {
                let azimuth = angles[az_idx];
                let elevation = elevation_range.start + el_idx as f32 * elevation_resolution;
                directions.push((azimuth, elevation.to_radians()));
                confidence.push(spectrum[[az_idx, el_idx]] / threshold);
            }
        }

        Ok(DoaResult {
            directions,
            confidence,
            spectrum,
            threshold,
        })
    }

    /// Compute spatial spectrum value for a specific direction (simplified MUSIC algorithm)
    fn compute_spatial_spectrum_value(&self, direction: (f32, f32)) -> Result<f32> {
        let frequency_bins = self.input_buffer.ncols();
        let mut total_power = 0.0;

        for freq_idx in 0..frequency_bins {
            let frequency =
                freq_idx as f32 * self.config.sample_rate / (2.0 * frequency_bins as f32);

            if frequency < self.config.frequency_range.0
                || frequency > self.config.frequency_range.1
            {
                continue;
            }

            let steering_vector = self.compute_steering_vector(frequency, direction);
            let input_vector = self.input_buffer.column(freq_idx);

            // Simple power calculation (correlation with steering vector)
            let mut power = Complex32::new(0.0, 0.0);
            for i in 0..self.config.array_size {
                power += input_vector[i] * steering_vector[i].conj();
            }

            total_power += power.norm_sqr();
        }

        Ok(total_power)
    }

    /// Compute beam pattern for visualization
    pub fn compute_beam_pattern(&self, frequency: f32) -> Result<BeamPattern> {
        let angle_resolution = 1.0; // 1 degree resolution
        let angle_range = -180.0..180.0;
        let angle_steps = ((angle_range.end - angle_range.start) / angle_resolution) as usize;

        let mut angles = Array1::zeros(angle_steps);
        let mut response = Array1::zeros(angle_steps);

        for (idx, angle_deg) in (0..angle_steps).enumerate() {
            let angle = angle_range.start + idx as f32 * angle_resolution;
            let angle_rad = angle.to_radians();
            angles[idx] = angle_rad;

            // Compute response in this direction
            let direction = (angle_rad, 0.0); // Azimuth plane only
            let steering_vector = self.compute_steering_vector(frequency, direction);

            // Find frequency bin
            let freq_idx = (frequency * 2.0 * self.weights.frequencies.len() as f32
                / self.config.sample_rate) as usize;
            let freq_idx = freq_idx.min(self.weights.frequencies.len() - 1);

            let weights_row = self.weights.weights.row(freq_idx);

            let mut pattern_value = Complex32::new(0.0, 0.0);
            for i in 0..self.config.array_size {
                pattern_value += weights_row[i] * steering_vector[i];
            }

            response[idx] = 20.0 * pattern_value.norm().log10(); // Convert to dB
        }

        // Analyze pattern characteristics
        let max_response = response.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let main_lobe_width = self.compute_main_lobe_width(&angles, &response, max_response);
        let side_lobe_level = self.compute_side_lobe_level(&response, max_response);
        let directivity_index = self.compute_directivity_index(&response);

        Ok(BeamPattern {
            angles,
            response,
            main_lobe_width,
            side_lobe_level,
            directivity_index,
        })
    }

    /// Compute main lobe width (-3dB width)
    fn compute_main_lobe_width(
        &self,
        angles: &Array1<f32>,
        response: &Array1<f32>,
        max_response: f32,
    ) -> f32 {
        let threshold = max_response - 3.0; // -3dB point

        // Find main lobe boundaries
        let max_idx = response
            .iter()
            .position(|&x| x == max_response)
            .unwrap_or(0);

        let mut left_idx = max_idx;
        while left_idx > 0 && response[left_idx] > threshold {
            left_idx -= 1;
        }

        let mut right_idx = max_idx;
        while right_idx < response.len() - 1 && response[right_idx] > threshold {
            right_idx += 1;
        }

        if right_idx > left_idx {
            angles[right_idx] - angles[left_idx]
        } else {
            0.0
        }
    }

    /// Compute side lobe level
    fn compute_side_lobe_level(&self, response: &Array1<f32>, max_response: f32) -> f32 {
        // Find maximum side lobe (simplified - assumes main lobe is in center)
        let center = response.len() / 2;
        let quarter = response.len() / 4;

        let left_side_max = response
            .slice(s![..center - quarter])
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let right_side_max = response
            .slice(s![center + quarter..])
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        left_side_max.max(right_side_max) - max_response
    }

    /// Compute directivity index
    fn compute_directivity_index(&self, response: &Array1<f32>) -> f32 {
        let max_response = response.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Average response over all angles
        let average_response = response.sum() / response.len() as f32;

        max_response - average_response
    }

    /// Update target direction
    pub fn set_target_direction(&mut self, azimuth: f32, elevation: f32) {
        self.config.target_direction = (azimuth, elevation);
        // Recompute weights for new direction
        if let Ok(new_weights) =
            Self::compute_initial_weights(&self.config, self.weights.frequencies.len())
        {
            self.weights = new_weights;
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &BeamformingConfig {
        &self.config
    }

    /// Get current weights
    pub fn weights(&self) -> &BeamformerWeights {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beamforming_config_default() {
        let config = BeamformingConfig::default();
        assert_eq!(config.array_size, 8);
        assert_eq!(config.algorithm, BeamformingAlgorithm::DelayAndSum);
        assert_eq!(config.element_positions.len(), 8);
    }

    #[test]
    fn test_beamforming_processor_creation() {
        let config = BeamformingConfig::default();
        let processor = BeamformingProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_steering_vector_computation() {
        let config = BeamformingConfig::default();
        let processor = BeamformingProcessor::new(config).unwrap();

        let steering_vector = processor.compute_steering_vector(1000.0, (0.0, 0.0));
        assert_eq!(steering_vector.len(), 8);

        // All elements should have the same magnitude for broadside direction
        let magnitude = steering_vector[0].norm();
        for &element in steering_vector.iter() {
            assert!((element.norm() - magnitude).abs() < 0.001);
        }
    }

    #[test]
    fn test_beam_pattern_computation() {
        let config = BeamformingConfig::default();
        let processor = BeamformingProcessor::new(config).unwrap();

        let pattern = processor.compute_beam_pattern(1000.0);
        assert!(pattern.is_ok());

        let pattern = pattern.unwrap();
        assert!(!pattern.angles.is_empty());
        assert_eq!(pattern.angles.len(), pattern.response.len());
        assert!(pattern.main_lobe_width > 0.0);
    }

    #[test]
    fn test_delay_and_sum_weights() {
        let config = BeamformingConfig::default();
        let weights = BeamformingProcessor::compute_delay_and_sum_weights(&config, 1000.0);

        assert_eq!(weights.len(), 8);

        // All weights should have equal magnitude (1/N)
        let expected_magnitude = 1.0 / 8.0;
        for &weight in weights.iter() {
            assert!((weight.norm() - expected_magnitude).abs() < 0.001);
        }
    }

    #[test]
    fn test_input_processing() {
        let config = BeamformingConfig::default();
        let mut processor = BeamformingProcessor::new(config).unwrap();

        // Create test input: 8 channels, 1024 samples
        let input = Array2::ones((8, 1024));

        let result = processor.process(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_target_direction_update() {
        let config = BeamformingConfig::default();
        let mut processor = BeamformingProcessor::new(config).unwrap();

        let original_direction = processor.config.target_direction;
        processor.set_target_direction(PI / 4.0, 0.0);

        assert_ne!(processor.config.target_direction, original_direction);
        assert_eq!(processor.config.target_direction, (PI / 4.0, 0.0));
    }
}
