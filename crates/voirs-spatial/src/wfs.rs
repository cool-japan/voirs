//! Wave Field Synthesis (WFS) implementation for advanced spatial audio reproduction
//!
//! This module provides Wave Field Synthesis capabilities for creating virtual sound fields
//! using arrays of loudspeakers. WFS enables reproduction of spatial audio with high
//! localization accuracy and extended listening area compared to traditional stereo systems.

use crate::types::Position3D;
use crate::{Error, Result};
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;

/// Wave Field Synthesis processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfsConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Number of speakers in the WFS array
    pub speaker_count: usize,
    /// Array geometry type
    pub array_geometry: ArrayGeometry,
    /// Speaker positions in 3D space
    pub speaker_positions: Vec<Position3D>,
    /// Maximum processing distance (meters)
    pub max_distance: f32,
    /// Frequency range for WFS processing (Hz)
    pub frequency_range: (f32, f32),
    /// Reference distance for amplitude scaling (meters)
    pub reference_distance: f32,
    /// Pre-emphasis filter parameters
    pub pre_emphasis: PreEmphasisConfig,
    /// Spatial aliasing compensation
    pub aliasing_compensation: bool,
}

/// WFS array geometry types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArrayGeometry {
    /// Linear array of speakers
    Linear,
    /// Circular array of speakers
    Circular,
    /// Rectangular array of speakers
    Rectangular,
    /// Custom arrangement
    Custom,
}

/// Pre-emphasis filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreEmphasisConfig {
    /// Enable pre-emphasis filtering
    pub enabled: bool,
    /// High-pass cutoff frequency (Hz)
    pub cutoff_frequency: f32,
    /// Filter order
    pub filter_order: usize,
}

/// Virtual sound source for WFS reproduction
#[derive(Debug, Clone)]
pub struct WfsSource {
    /// Unique identifier
    pub id: String,
    /// 3D position
    pub position: Position3D,
    /// Audio signal
    pub audio_data: Array1<f32>,
    /// Source type
    pub source_type: WfsSourceType,
    /// Gain factor
    pub gain: f32,
    /// Distance from reference point
    pub distance: f32,
}

/// WFS source types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WfsSourceType {
    /// Point source
    Point,
    /// Plane wave
    PlaneWave,
    /// Extended source
    Extended,
}

/// WFS driving function for a single speaker
#[derive(Debug, Clone)]
pub struct WfsDrivingFunction {
    /// Speaker index
    pub speaker_index: usize,
    /// Complex frequency response
    pub frequency_response: Array1<Complex32>,
    /// Delay in samples
    pub delay_samples: f32,
    /// Amplitude scaling factor
    pub amplitude: f32,
}

/// Main Wave Field Synthesis processor
pub struct WfsProcessor {
    /// Configuration
    config: WfsConfig,
    /// FFT planner for frequency domain processing
    fft_planner: Arc<RealFftPlanner<f32>>,
    /// Forward FFT
    forward_fft: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT
    inverse_fft: Arc<dyn realfft::ComplexToReal<f32>>,
    /// Speaker driving functions cache
    driving_functions_cache: HashMap<String, Vec<WfsDrivingFunction>>,
    /// Processing buffers
    frequency_buffer: Array2<Complex32>,
    time_buffer: Array2<f32>,
    /// Speed of sound (m/s)
    speed_of_sound: f32,
}

impl Default for WfsConfig {
    fn default() -> Self {
        // Default linear array configuration
        let speaker_count = 16;
        let speaker_spacing = 0.2; // 20cm spacing
        let speaker_positions: Vec<Position3D> = (0..speaker_count)
            .map(|i| Position3D {
                x: (i as f32 - speaker_count as f32 / 2.0) * speaker_spacing,
                y: 0.0,
                z: 0.0,
            })
            .collect();

        Self {
            sample_rate: 48000.0,
            speaker_count,
            array_geometry: ArrayGeometry::Linear,
            speaker_positions,
            max_distance: 10.0,
            frequency_range: (20.0, 20000.0),
            reference_distance: 1.0,
            pre_emphasis: PreEmphasisConfig {
                enabled: true,
                cutoff_frequency: 100.0,
                filter_order: 2,
            },
            aliasing_compensation: true,
        }
    }
}

impl WfsProcessor {
    /// Create a new WFS processor
    pub fn new(config: WfsConfig) -> Result<Self> {
        if config.speaker_count == 0 {
            return Err(Error::LegacyConfig(
                "Speaker count must be greater than 0".to_string(),
            ));
        }

        if config.speaker_positions.len() != config.speaker_count {
            return Err(Error::LegacyConfig(
                "Number of speaker positions must match speaker count".to_string(),
            ));
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let buffer_size = 1024; // Default FFT size

        let forward_fft = planner.plan_fft_forward(buffer_size);
        let inverse_fft = planner.plan_fft_inverse(buffer_size);

        let frequency_buffer = Array2::zeros((config.speaker_count, buffer_size / 2 + 1));
        let time_buffer = Array2::zeros((config.speaker_count, buffer_size));

        Ok(Self {
            config,
            fft_planner: Arc::new(planner),
            forward_fft,
            inverse_fft,
            driving_functions_cache: HashMap::new(),
            frequency_buffer,
            time_buffer,
            speed_of_sound: 343.0, // Standard speed of sound at 20°C
        })
    }

    /// Process a virtual source using WFS
    pub fn process_source(&mut self, source: &WfsSource) -> Result<Array2<f32>> {
        let driving_functions = self.compute_driving_functions(source)?;
        self.apply_driving_functions(&driving_functions, &source.audio_data)
    }

    /// Compute WFS driving functions for a source
    fn compute_driving_functions(&mut self, source: &WfsSource) -> Result<Vec<WfsDrivingFunction>> {
        // Check cache first
        if let Some(cached) = self.driving_functions_cache.get(&source.id) {
            return Ok(cached.clone());
        }

        let mut driving_functions = Vec::with_capacity(self.config.speaker_count);

        for (speaker_idx, speaker_pos) in self.config.speaker_positions.iter().enumerate() {
            let driving_function = match source.source_type {
                WfsSourceType::Point => {
                    self.compute_point_source_driving_function(source, speaker_pos, speaker_idx)?
                }
                WfsSourceType::PlaneWave => {
                    self.compute_plane_wave_driving_function(source, speaker_pos, speaker_idx)?
                }
                WfsSourceType::Extended => {
                    self.compute_extended_source_driving_function(source, speaker_pos, speaker_idx)?
                }
            };
            driving_functions.push(driving_function);
        }

        // Cache the result
        self.driving_functions_cache
            .insert(source.id.clone(), driving_functions.clone());

        Ok(driving_functions)
    }

    /// Compute driving function for point source
    fn compute_point_source_driving_function(
        &self,
        source: &WfsSource,
        speaker_pos: &Position3D,
        speaker_idx: usize,
    ) -> Result<WfsDrivingFunction> {
        // Distance from source to speaker
        let distance = source.position.distance_to(speaker_pos);

        // Delay calculation
        let delay_time = distance / self.speed_of_sound;
        let delay_samples = delay_time * self.config.sample_rate;

        // Amplitude calculation with distance attenuation
        let amplitude = source.gain * (self.config.reference_distance / distance).sqrt();

        // Frequency response (simplified - can be extended with directivity patterns)
        let buffer_size = self.frequency_buffer.ncols();
        let mut frequency_response = Array1::zeros(buffer_size);

        // Apply frequency-dependent processing
        for (freq_idx, response) in frequency_response.iter_mut().enumerate() {
            let frequency = freq_idx as f32 * self.config.sample_rate / (2.0 * buffer_size as f32);

            if frequency >= self.config.frequency_range.0
                && frequency <= self.config.frequency_range.1
            {
                // Basic WFS frequency response
                let omega = 2.0 * PI * frequency;
                let wave_number = omega / self.speed_of_sound;

                // Phase shift due to distance
                let phase = -wave_number * distance;
                *response = Complex32::from_polar(amplitude, phase);

                // Apply pre-emphasis if enabled
                if self.config.pre_emphasis.enabled {
                    let pre_emphasis_gain = self.compute_pre_emphasis_gain(frequency);
                    *response *= pre_emphasis_gain;
                }
            }
        }

        Ok(WfsDrivingFunction {
            speaker_index: speaker_idx,
            frequency_response,
            delay_samples,
            amplitude,
        })
    }

    /// Compute driving function for plane wave
    fn compute_plane_wave_driving_function(
        &self,
        source: &WfsSource,
        speaker_pos: &Position3D,
        speaker_idx: usize,
    ) -> Result<WfsDrivingFunction> {
        // For plane waves, the delay is based on the projection of speaker position
        // onto the wave direction
        let wave_direction = source.position.normalized();
        let projection = speaker_pos.dot(&wave_direction);

        let delay_time = projection / self.speed_of_sound;
        let delay_samples = delay_time * self.config.sample_rate;

        // Constant amplitude for plane waves
        let amplitude = source.gain;

        // Frequency response for plane wave
        let buffer_size = self.frequency_buffer.ncols();
        let mut frequency_response = Array1::zeros(buffer_size);

        for (freq_idx, response) in frequency_response.iter_mut().enumerate() {
            let frequency = freq_idx as f32 * self.config.sample_rate / (2.0 * buffer_size as f32);

            if frequency >= self.config.frequency_range.0
                && frequency <= self.config.frequency_range.1
            {
                let omega = 2.0 * PI * frequency;
                let wave_number = omega / self.speed_of_sound;
                let phase = -wave_number * projection;

                *response = Complex32::from_polar(amplitude, phase);
            }
        }

        Ok(WfsDrivingFunction {
            speaker_index: speaker_idx,
            frequency_response,
            delay_samples,
            amplitude,
        })
    }

    /// Compute driving function for extended source (simplified implementation)
    fn compute_extended_source_driving_function(
        &self,
        source: &WfsSource,
        speaker_pos: &Position3D,
        speaker_idx: usize,
    ) -> Result<WfsDrivingFunction> {
        // For extended sources, use point source approximation
        // In a full implementation, this would integrate over the source extent
        self.compute_point_source_driving_function(source, speaker_pos, speaker_idx)
    }

    /// Compute pre-emphasis filter gain
    fn compute_pre_emphasis_gain(&self, frequency: f32) -> f32 {
        if !self.config.pre_emphasis.enabled
            || frequency < self.config.pre_emphasis.cutoff_frequency
        {
            return 1.0;
        }

        // Simple high-pass filter response
        let normalized_freq = frequency / self.config.pre_emphasis.cutoff_frequency;
        normalized_freq.sqrt() // Square root frequency response
    }

    /// Apply driving functions to generate speaker signals
    fn apply_driving_functions(
        &mut self,
        driving_functions: &[WfsDrivingFunction],
        audio_data: &Array1<f32>,
    ) -> Result<Array2<f32>> {
        let output_length = audio_data.len();
        let mut output = Array2::zeros((self.config.speaker_count, output_length));

        // Process each speaker channel
        for (speaker_idx, driving_function) in driving_functions.iter().enumerate() {
            let delayed_signal = self.apply_delay_and_amplitude(
                audio_data,
                driving_function.delay_samples,
                driving_function.amplitude,
            )?;

            // Apply frequency domain filtering if needed
            let processed_signal = if self.should_apply_frequency_processing(driving_function) {
                self.apply_frequency_response(
                    &delayed_signal,
                    &driving_function.frequency_response,
                )?
            } else {
                delayed_signal
            };

            // Copy to output
            let output_length = output_length.min(processed_signal.len());
            output
                .row_mut(speaker_idx)
                .slice_mut(ndarray::s![..output_length])
                .assign(&processed_signal.slice(ndarray::s![..output_length]));
        }

        Ok(output)
    }

    /// Apply delay and amplitude scaling to signal
    fn apply_delay_and_amplitude(
        &self,
        signal: &Array1<f32>,
        delay_samples: f32,
        amplitude: f32,
    ) -> Result<Array1<f32>> {
        let signal_length = signal.len();
        let delay_int = delay_samples.floor() as isize;
        let delay_frac = delay_samples - delay_int as f32;

        let mut output = Array1::zeros(signal_length);

        // Apply integer delay
        if delay_int >= 0 {
            let start_idx = delay_int as usize;
            if start_idx < signal_length {
                let copy_length = signal_length - start_idx;
                output
                    .slice_mut(ndarray::s![start_idx..])
                    .assign(&signal.slice(ndarray::s![..copy_length]));
            }
        }

        // Apply fractional delay using linear interpolation
        if delay_frac > 0.001 {
            for i in 1..signal_length {
                output[i] = output[i] * (1.0 - delay_frac) + output[i - 1] * delay_frac;
            }
        }

        // Apply amplitude scaling
        output *= amplitude;

        Ok(output)
    }

    /// Check if frequency domain processing should be applied
    fn should_apply_frequency_processing(&self, driving_function: &WfsDrivingFunction) -> bool {
        // Apply frequency processing if the response is not flat
        driving_function
            .frequency_response
            .iter()
            .any(|&response| (response.norm() - 1.0).abs() > 0.1 || response.arg().abs() > 0.1)
    }

    /// Apply frequency response using FFT
    fn apply_frequency_response(
        &mut self,
        signal: &Array1<f32>,
        frequency_response: &Array1<Complex32>,
    ) -> Result<Array1<f32>> {
        let buffer_size = self.frequency_buffer.ncols() * 2 - 2;
        let mut padded_signal = Array1::zeros(buffer_size);

        // Copy signal to padded buffer
        let copy_length = signal.len().min(buffer_size);
        padded_signal
            .slice_mut(ndarray::s![..copy_length])
            .assign(&signal.slice(ndarray::s![..copy_length]));

        // Transform to frequency domain
        let mut spectrum = Array1::zeros(frequency_response.len());
        self.forward_fft
            .process(
                padded_signal.as_slice_mut().unwrap(),
                spectrum.as_slice_mut().unwrap(),
            )
            .map_err(|e| Error::LegacyProcessing(format!("FFT error: {e:?}")))?;

        // Apply frequency response
        for (spectrum_bin, &response) in spectrum.iter_mut().zip(frequency_response.iter()) {
            *spectrum_bin *= response;
        }

        // Transform back to time domain
        let mut result = Array1::zeros(buffer_size);
        self.inverse_fft
            .process(
                spectrum.as_slice_mut().unwrap(),
                result.as_slice_mut().unwrap(),
            )
            .map_err(|e| Error::LegacyProcessing(format!("IFFT error: {e:?}")))?;

        // Return original length
        Ok(result.slice(ndarray::s![..signal.len()]).to_owned())
    }

    /// Update source position (invalidates cache for that source)
    pub fn update_source_position(&mut self, source_id: &str, new_position: Position3D) {
        self.driving_functions_cache.remove(source_id);
    }

    /// Clear all cached driving functions
    pub fn clear_cache(&mut self) {
        self.driving_functions_cache.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &WfsConfig {
        &self.config
    }

    /// Set speed of sound (for different environmental conditions)
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        if speed > 0.0 {
            self.speed_of_sound = speed;
            self.clear_cache(); // Invalidate cache since speed affects calculations
        }
    }
}

/// WFS array builder for different geometries
pub struct WfsArrayBuilder {
    geometry: ArrayGeometry,
    speaker_count: usize,
    dimensions: (f32, f32, f32), // width, height, depth
}

impl WfsArrayBuilder {
    /// Create a new array builder
    pub fn new(geometry: ArrayGeometry) -> Self {
        Self {
            geometry,
            speaker_count: 16,
            dimensions: (3.0, 0.0, 0.0), // 3m wide linear array by default
        }
    }

    /// Set the number of speakers
    pub fn speaker_count(mut self, count: usize) -> Self {
        self.speaker_count = count;
        self
    }

    /// Set array dimensions
    pub fn dimensions(mut self, width: f32, height: f32, depth: f32) -> Self {
        self.dimensions = (width, height, depth);
        self
    }

    /// Build speaker positions based on geometry
    pub fn build_positions(self) -> Vec<Position3D> {
        match self.geometry {
            ArrayGeometry::Linear => self.build_linear_array(),
            ArrayGeometry::Circular => self.build_circular_array(),
            ArrayGeometry::Rectangular => self.build_rectangular_array(),
            ArrayGeometry::Custom => vec![], // User must provide custom positions
        }
    }

    fn build_linear_array(&self) -> Vec<Position3D> {
        let spacing = self.dimensions.0 / (self.speaker_count - 1) as f32;
        let start_x = -self.dimensions.0 / 2.0;

        (0..self.speaker_count)
            .map(|i| Position3D {
                x: start_x + i as f32 * spacing,
                y: 0.0,
                z: 0.0,
            })
            .collect()
    }

    fn build_circular_array(&self) -> Vec<Position3D> {
        let radius = self.dimensions.0 / 2.0;
        let angle_step = 2.0 * PI / self.speaker_count as f32;

        (0..self.speaker_count)
            .map(|i| {
                let angle = i as f32 * angle_step;
                Position3D {
                    x: radius * angle.cos(),
                    y: radius * angle.sin(),
                    z: 0.0,
                }
            })
            .collect()
    }

    fn build_rectangular_array(&self) -> Vec<Position3D> {
        // Simple rectangular grid
        let cols = (self.speaker_count as f32).sqrt().ceil() as usize;
        let rows = (self.speaker_count + cols - 1) / cols;

        let x_spacing = self.dimensions.0 / (cols - 1) as f32;
        let y_spacing = self.dimensions.1 / (rows - 1) as f32;

        let start_x = -self.dimensions.0 / 2.0;
        let start_y = -self.dimensions.1 / 2.0;

        let mut positions = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                if positions.len() < self.speaker_count {
                    positions.push(Position3D {
                        x: start_x + col as f32 * x_spacing,
                        y: start_y + row as f32 * y_spacing,
                        z: 0.0,
                    });
                }
            }
        }
        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wfs_config_default() {
        let config = WfsConfig::default();
        assert_eq!(config.speaker_count, 16);
        assert_eq!(config.array_geometry, ArrayGeometry::Linear);
        assert_eq!(config.speaker_positions.len(), 16);
    }

    #[test]
    fn test_wfs_processor_creation() {
        let config = WfsConfig::default();
        let processor = WfsProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_array_builder_linear() {
        let positions = WfsArrayBuilder::new(ArrayGeometry::Linear)
            .speaker_count(8)
            .dimensions(2.0, 0.0, 0.0)
            .build_positions();

        assert_eq!(positions.len(), 8);
        assert_eq!(positions[0].x, -1.0);
        assert_eq!(positions[7].x, 1.0);
    }

    #[test]
    fn test_array_builder_circular() {
        let positions = WfsArrayBuilder::new(ArrayGeometry::Circular)
            .speaker_count(4)
            .dimensions(2.0, 0.0, 0.0) // diameter = 2.0, so radius = 1.0
            .build_positions();

        assert_eq!(positions.len(), 4);
        // First speaker should be at (1, 0, 0)
        assert!((positions[0].x - 1.0).abs() < 0.001);
        assert!(positions[0].y.abs() < 0.001);
    }

    #[test]
    fn test_wfs_source_creation() {
        let source = WfsSource {
            id: "test_source".to_string(),
            position: Position3D {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            audio_data: Array1::zeros(1024),
            source_type: WfsSourceType::Point,
            gain: 1.0,
            distance: 1.0,
        };

        assert_eq!(source.id, "test_source");
        assert_eq!(source.source_type, WfsSourceType::Point);
    }

    #[test]
    fn test_processor_source_processing() {
        let config = WfsConfig::default();
        let mut processor = WfsProcessor::new(config).unwrap();

        let source = WfsSource {
            id: "test".to_string(),
            position: Position3D {
                x: 2.0,
                y: 0.0,
                z: 0.0,
            },
            audio_data: Array1::ones(512),
            source_type: WfsSourceType::Point,
            gain: 1.0,
            distance: 2.0,
        };

        let result = processor.process_source(&source);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.nrows(), 16); // 16 speakers
        assert_eq!(output.ncols(), 512); // Same length as input
    }
}
