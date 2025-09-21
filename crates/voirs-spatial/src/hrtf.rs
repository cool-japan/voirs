//! HRTF (Head-Related Transfer Function) processing for spatial audio

pub mod ai_personalization;
pub mod database;

use crate::types::Position3D;
pub use database::{
    DatabaseConfig, DatabaseStatistics, HrtfDatabaseManager, HrtfMeasurement as DbHrtfMeasurement,
    HrtfPosition, InterpolationMethod as DbInterpolationMethod, PersonalizedHrtf, StorageFormat,
};
use ndarray::Array1;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// HRTF processor for binaural audio rendering
pub struct HrtfProcessor {
    /// HRTF database
    database: Arc<HrtfDatabase>,
    /// FFT planner for convolution
    #[allow(dead_code)]
    fft_planner: RealFftPlanner<f32>,
    /// Convolution buffer size
    #[allow(dead_code)]
    buffer_size: usize,
    /// Overlap-add state for left channel
    #[allow(dead_code)]
    overlap_left: Array1<f32>,
    /// Overlap-add state for right channel
    #[allow(dead_code)]
    overlap_right: Array1<f32>,
    /// Processing configuration
    config: HrtfConfig,
}

/// HRTF database containing impulse responses
#[derive(Clone)]
pub struct HrtfDatabase {
    /// Database metadata
    metadata: HrtfMetadata,
    /// Left ear impulse responses indexed by (azimuth, elevation)
    left_responses: HashMap<(i32, i32), Array1<f32>>,
    /// Right ear impulse responses indexed by (azimuth, elevation)
    right_responses: HashMap<(i32, i32), Array1<f32>>,
    /// Frequency responses (optional, for analysis)
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    frequency_responses: Option<HashMap<(i32, i32), (Array1<f32>, Array1<f32>)>>,
    /// Distance-dependent responses (if available)
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    distance_responses: Option<HashMap<(i32, i32, u32), (Array1<f32>, Array1<f32>)>>,
}

/// HRTF configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// HRIR length
    pub hrir_length: usize,
    /// Crossfade time for smooth transitions
    pub crossfade_time: f32,
    /// Distance modeling enabled
    pub enable_distance_modeling: bool,
    /// Interpolation method
    pub interpolation_method: InterpolationMethod,
    /// Head circumference for personalization (in cm)
    pub head_circumference: Option<f32>,
    /// Near-field distance threshold (meters)
    pub near_field_distance: f32,
    /// Far-field distance threshold (meters)
    pub far_field_distance: f32,
    /// Enable air absorption
    pub enable_air_absorption: bool,
    /// Temperature for air absorption (Celsius)
    pub temperature: f32,
    /// Humidity for air absorption (relative, 0.0-1.0)
    pub humidity: f32,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
}

/// HRTF database metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfMetadata {
    /// Database name/source
    pub name: String,
    /// Sample rate of recordings
    pub sample_rate: u32,
    /// Length of impulse responses
    pub hrir_length: usize,
    /// Available azimuth angles (degrees)
    pub azimuth_angles: Vec<i32>,
    /// Available elevation angles (degrees)
    pub elevation_angles: Vec<i32>,
    /// Available distances (if any)
    pub distances: Option<Vec<f32>>,
    /// Subject information
    pub subject_info: Option<SubjectInfo>,
}

/// Subject information for personalized HRTF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectInfo {
    /// Head circumference in cm
    pub head_circumference: f32,
    /// Head width in cm
    pub head_width: f32,
    /// Head height in cm
    pub head_height: f32,
    /// Ear height in cm
    pub ear_height: f32,
    /// Shoulder width in cm
    pub shoulder_width: f32,
}

/// Interpolation methods for HRTF
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Nearest neighbor (fastest)
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Spherical interpolation
    Spherical,
    /// Weighted interpolation based on distance
    Weighted,
}

/// Spherical coordinates for HRTF lookup
#[derive(Debug, Clone, Copy)]
pub struct SphericalCoordinates {
    /// Azimuth angle in degrees (-180 to 180)
    pub azimuth: f32,
    /// Elevation angle in degrees (-90 to 90)
    pub elevation: f32,
    /// Distance in meters
    pub distance: f32,
}

/// HRTF measurement point
#[derive(Debug, Clone)]
pub struct HrtfMeasurement {
    /// Spherical coordinates
    pub coordinates: SphericalCoordinates,
    /// Left ear impulse response
    pub left_hrir: Array1<f32>,
    /// Right ear impulse response
    pub right_hrir: Array1<f32>,
}

impl HrtfProcessor {
    /// Create new HRTF processor
    pub async fn new(database_path: Option<PathBuf>) -> crate::Result<Self> {
        let database = if let Some(path) = database_path {
            HrtfDatabase::load_from_file(&path).await?
        } else {
            HrtfDatabase::load_default().await?
        };

        let config = HrtfConfig::default();
        let buffer_size = config.hrir_length * 2; // For overlap-add

        Ok(Self {
            database: Arc::new(database),
            fft_planner: RealFftPlanner::new(),
            buffer_size,
            overlap_left: Array1::zeros(buffer_size),
            overlap_right: Array1::zeros(buffer_size),
            config,
        })
    }

    /// Create new HRTF processor with default database
    pub async fn new_default() -> crate::Result<Self> {
        Self::new(None).await
    }

    /// Create HRTF processor with custom configuration
    pub async fn with_config(
        database_path: Option<PathBuf>,
        config: HrtfConfig,
    ) -> crate::Result<Self> {
        let database = if let Some(path) = database_path {
            HrtfDatabase::load_from_file(&path).await?
        } else {
            HrtfDatabase::load_default().await?
        };

        let buffer_size = config.hrir_length * 2;

        Ok(Self {
            database: Arc::new(database),
            fft_planner: RealFftPlanner::new(),
            buffer_size,
            overlap_left: Array1::zeros(buffer_size),
            overlap_right: Array1::zeros(buffer_size),
            config,
        })
    }

    /// Process audio with HRTF for a specific position
    pub async fn process_position(
        &self,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        position: &Position3D,
    ) -> crate::Result<()> {
        // Convert Cartesian to spherical coordinates
        let spherical = self.cartesian_to_spherical(position);

        // Get HRTF for this position (with distance modeling if enabled)
        let (mut left_hrir, mut right_hrir) = self.get_hrtf(&spherical)?;

        // Apply distance modeling if enabled
        if self.config.enable_distance_modeling {
            self.apply_distance_modeling(&mut left_hrir, &mut right_hrir, &spherical)?;
        }

        // Apply air absorption if enabled
        if self.config.enable_air_absorption && spherical.distance > self.config.near_field_distance
        {
            self.apply_air_absorption(&mut left_hrir, &mut right_hrir, spherical.distance)?;
        }

        // Convolve input with processed HRIRs
        self.convolve_hrtf(input, &left_hrir, &right_hrir, left_output, right_output)?;

        Ok(())
    }

    /// Process audio with smooth position interpolation
    pub async fn process_position_smooth(
        &self,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        start_position: &Position3D,
        end_position: &Position3D,
        progress: f32,
    ) -> crate::Result<()> {
        // Interpolate position
        let current_position = Position3D::new(
            start_position.x * (1.0 - progress) + end_position.x * progress,
            start_position.y * (1.0 - progress) + end_position.y * progress,
            start_position.z * (1.0 - progress) + end_position.z * progress,
        );

        self.process_position(input, left_output, right_output, &current_position)
            .await
    }

    /// Process audio stream with overlap-add for real-time applications
    pub async fn process_realtime_chunk(
        &mut self,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        position: &Position3D,
    ) -> crate::Result<()> {
        let chunk_size = input.len();
        let hrir_len = self.config.hrir_length;

        // Ensure output buffers are correct size
        if left_output.len() != chunk_size || right_output.len() != chunk_size {
            return Err(crate::Error::processing(
                "Output buffer size must match input chunk size",
            ));
        }

        // Convert position to spherical coordinates
        let spherical = self.cartesian_to_spherical(position);

        // Get HRTF for this position
        let (left_hrir, right_hrir) = self.get_hrtf(&spherical)?;

        // Prepare extended buffers for convolution
        let conv_len = chunk_size + hrir_len - 1;
        let mut left_conv = Array1::zeros(conv_len);
        let mut right_conv = Array1::zeros(conv_len);

        // Perform convolution
        self.convolve_hrtf(
            input,
            &left_hrir,
            &right_hrir,
            &mut left_conv,
            &mut right_conv,
        )?;

        // Overlap-add with previous tail
        for i in 0..chunk_size {
            left_output[i] = left_conv[i] + self.overlap_left[i];
            right_output[i] = right_conv[i] + self.overlap_right[i];
        }

        // Store tail for next chunk
        self.overlap_left.fill(0.0);
        self.overlap_right.fill(0.0);

        let tail_start = chunk_size;
        let tail_len = (conv_len - chunk_size).min(self.overlap_left.len());

        for i in 0..tail_len {
            self.overlap_left[i] = left_conv[tail_start + i];
            self.overlap_right[i] = right_conv[tail_start + i];
        }

        Ok(())
    }

    /// Reset overlap-add buffers (call when starting new stream)
    pub fn reset_buffers(&mut self) {
        self.overlap_left.fill(0.0);
        self.overlap_right.fill(0.0);
    }

    /// Process multiple positions with crossfading between them
    pub async fn process_crossfade(
        &self,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        positions: &[(Position3D, f32)], // (position, weight)
    ) -> crate::Result<()> {
        left_output.fill(0.0);
        right_output.fill(0.0);

        let mut temp_left = Array1::zeros(input.len());
        let mut temp_right = Array1::zeros(input.len());

        for (position, weight) in positions {
            // Process audio for this position
            self.process_position(input, &mut temp_left, &mut temp_right, position)
                .await?;

            // Add weighted contribution
            for i in 0..left_output.len() {
                left_output[i] += temp_left[i] * weight;
                right_output[i] += temp_right[i] * weight;
            }
        }

        Ok(())
    }

    /// Get HRTF for spherical coordinates
    fn get_hrtf(&self, coords: &SphericalCoordinates) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        match self.config.interpolation_method {
            InterpolationMethod::Nearest => self.get_nearest_hrtf(coords),
            InterpolationMethod::Bilinear => self.get_bilinear_hrtf(coords),
            InterpolationMethod::Spherical => self.get_spherical_hrtf(coords),
            InterpolationMethod::Weighted => self.get_weighted_hrtf(coords),
        }
    }

    /// Get nearest neighbor HRTF
    fn get_nearest_hrtf(
        &self,
        coords: &SphericalCoordinates,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        let azimuth = coords.azimuth.round() as i32;
        let elevation = coords.elevation.round() as i32;

        // Find closest available angles
        let closest_azimuth =
            self.find_closest_angle(azimuth, &self.database.metadata.azimuth_angles);
        let closest_elevation =
            self.find_closest_angle(elevation, &self.database.metadata.elevation_angles);

        let key = (closest_azimuth, closest_elevation);

        let left_hrir = self.database.left_responses.get(&key).ok_or_else(|| {
            crate::Error::LegacyHrtf(format!(
                "No HRTF found for angles ({closest_azimuth}, {closest_elevation})"
            ))
        })?;
        let right_hrir = self.database.right_responses.get(&key).ok_or_else(|| {
            crate::Error::LegacyHrtf(format!(
                "No HRTF found for angles ({closest_azimuth}, {closest_elevation})"
            ))
        })?;

        Ok((left_hrir.clone(), right_hrir.clone()))
    }

    /// Get bilinear interpolated HRTF
    fn get_bilinear_hrtf(
        &self,
        coords: &SphericalCoordinates,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        // Find surrounding angles
        let az_low = self.find_lower_angle(
            coords.azimuth as i32,
            &self.database.metadata.azimuth_angles,
        );
        let az_high = self.find_higher_angle(
            coords.azimuth as i32,
            &self.database.metadata.azimuth_angles,
        );
        let el_low = self.find_lower_angle(
            coords.elevation as i32,
            &self.database.metadata.elevation_angles,
        );
        let el_high = self.find_higher_angle(
            coords.elevation as i32,
            &self.database.metadata.elevation_angles,
        );

        // Get four corner HRTFs
        let hrtf_00 = self.get_hrtf_at_angles(az_low, el_low)?;
        let hrtf_01 = self.get_hrtf_at_angles(az_low, el_high)?;
        let hrtf_10 = self.get_hrtf_at_angles(az_high, el_low)?;
        let hrtf_11 = self.get_hrtf_at_angles(az_high, el_high)?;

        // Calculate interpolation weights
        let az_weight = if az_high != az_low {
            (coords.azimuth - az_low as f32) / (az_high - az_low) as f32
        } else {
            0.0
        };
        let el_weight = if el_high != el_low {
            (coords.elevation - el_low as f32) / (el_high - el_low) as f32
        } else {
            0.0
        };

        // Bilinear interpolation
        let left_hrir = self.interpolate_hrtf(&[
            (&hrtf_00.0, (1.0 - az_weight) * (1.0 - el_weight)),
            (&hrtf_01.0, (1.0 - az_weight) * el_weight),
            (&hrtf_10.0, az_weight * (1.0 - el_weight)),
            (&hrtf_11.0, az_weight * el_weight),
        ]);

        let right_hrir = self.interpolate_hrtf(&[
            (&hrtf_00.1, (1.0 - az_weight) * (1.0 - el_weight)),
            (&hrtf_01.1, (1.0 - az_weight) * el_weight),
            (&hrtf_10.1, az_weight * (1.0 - el_weight)),
            (&hrtf_11.1, az_weight * el_weight),
        ]);

        Ok((left_hrir, right_hrir))
    }

    /// Get spherical interpolated HRTF using great circle distances
    fn get_spherical_hrtf(
        &self,
        coords: &SphericalCoordinates,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        let mut left_sum = Array1::zeros(self.config.hrir_length);
        let mut right_sum = Array1::zeros(self.config.hrir_length);
        let mut weight_sum = 0.0;

        // Find the 4 nearest neighbors on the sphere
        let mut nearest_points = Vec::new();

        for (&(az, el), left_hrir) in &self.database.left_responses {
            let right_hrir = self.database.right_responses.get(&(az, el)).unwrap();

            // Calculate great circle distance (spherical distance)
            let angular_distance = self.calculate_angular_distance(
                coords.azimuth,
                coords.elevation,
                az as f32,
                el as f32,
            );

            nearest_points.push((angular_distance, left_hrir, right_hrir));
        }

        // Sort by distance and take the 4 nearest
        nearest_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nearest_points.truncate(4);

        // Use inverse distance weighting with spherical consideration
        for (distance, left_hrir, right_hrir) in nearest_points {
            let weight = if distance < 0.01 {
                1.0 // Very close, use full weight
            } else {
                // Inverse distance weighting with spherical correction
                1.0 / (distance.to_radians().sin() + 0.001)
            };

            weight_sum += weight;

            for i in 0..left_sum.len().min(left_hrir.len()) {
                left_sum[i] += left_hrir[i] * weight;
                right_sum[i] += right_hrir[i] * weight;
            }
        }

        // Normalize
        if weight_sum > 0.0 {
            left_sum /= weight_sum;
            right_sum /= weight_sum;
        }

        Ok((left_sum, right_sum))
    }

    /// Get weighted interpolated HRTF
    fn get_weighted_hrtf(
        &self,
        coords: &SphericalCoordinates,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        let mut left_sum = Array1::zeros(self.config.hrir_length);
        let mut right_sum = Array1::zeros(self.config.hrir_length);
        let mut weight_sum = 0.0;

        // Find nearby measurements and weight by distance
        for (&(az, el), left_hrir) in &self.database.left_responses {
            let right_hrir = self.database.right_responses.get(&(az, el)).unwrap();

            // Calculate angular distance
            let angular_distance = self.calculate_angular_distance(
                coords.azimuth,
                coords.elevation,
                az as f32,
                el as f32,
            );

            if angular_distance < 30.0 {
                // Only use nearby measurements
                let weight = 1.0 / (1.0 + angular_distance);
                weight_sum += weight;

                for i in 0..left_sum.len() {
                    left_sum[i] += left_hrir[i] * weight;
                    right_sum[i] += right_hrir[i] * weight;
                }
            }
        }

        if weight_sum > 0.0 {
            left_sum /= weight_sum;
            right_sum /= weight_sum;
        }

        Ok((left_sum, right_sum))
    }

    /// Convolve input with HRTF using overlap-add FFT convolution
    fn convolve_hrtf(
        &self,
        input: &Array1<f32>,
        left_hrir: &Array1<f32>,
        right_hrir: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
    ) -> crate::Result<()> {
        let input_len = input.len();
        let hrir_len = left_hrir.len();

        // For small inputs or HRIRs, use time-domain convolution
        if input_len < 64 || hrir_len < 64 {
            return self.convolve_time_domain(
                input,
                left_hrir,
                right_hrir,
                left_output,
                right_output,
            );
        }

        // Use frequency-domain convolution for better performance
        self.convolve_frequency_domain(input, left_hrir, right_hrir, left_output, right_output)
    }

    /// Time-domain convolution for small inputs
    fn convolve_time_domain(
        &self,
        input: &Array1<f32>,
        left_hrir: &Array1<f32>,
        right_hrir: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
    ) -> crate::Result<()> {
        left_output.fill(0.0);
        right_output.fill(0.0);

        for (i, &sample) in input.iter().enumerate() {
            for (j, &hrir_sample) in left_hrir.iter().enumerate() {
                if i + j < left_output.len() {
                    left_output[i + j] += sample * hrir_sample;
                }
            }
            for (j, &hrir_sample) in right_hrir.iter().enumerate() {
                if i + j < right_output.len() {
                    right_output[i + j] += sample * hrir_sample;
                }
            }
        }

        Ok(())
    }

    /// Frequency-domain convolution using FFT
    fn convolve_frequency_domain(
        &self,
        input: &Array1<f32>,
        left_hrir: &Array1<f32>,
        right_hrir: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
    ) -> crate::Result<()> {
        let input_len = input.len();
        let hrir_len = left_hrir.len();
        let conv_len = input_len + hrir_len - 1;

        // Find next power of 2 for FFT
        let fft_len = conv_len.next_power_of_two();

        // Prepare padded signals
        let mut input_padded = vec![0.0; fft_len];
        let mut left_hrir_padded = vec![0.0; fft_len];
        let mut right_hrir_padded = vec![0.0; fft_len];

        input_padded[..input_len].copy_from_slice(input.as_slice().unwrap());
        left_hrir_padded[..hrir_len].copy_from_slice(left_hrir.as_slice().unwrap());
        right_hrir_padded[..hrir_len].copy_from_slice(right_hrir.as_slice().unwrap());

        // Create FFT planner
        let mut planner = realfft::RealFftPlanner::new();
        let fft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);

        // FFT of input and HRIRs
        let mut input_spectrum = fft.make_output_vec();
        let mut left_hrir_spectrum = fft.make_output_vec();
        let mut right_hrir_spectrum = fft.make_output_vec();

        fft.process(&mut input_padded, &mut input_spectrum)
            .map_err(|e| crate::Error::LegacyProcessing(format!("FFT error: {e}")))?;

        fft.process(&mut left_hrir_padded, &mut left_hrir_spectrum)
            .map_err(|e| crate::Error::LegacyProcessing(format!("FFT error: {e}")))?;

        fft.process(&mut right_hrir_padded, &mut right_hrir_spectrum)
            .map_err(|e| crate::Error::LegacyProcessing(format!("FFT error: {e}")))?;

        // Complex multiplication in frequency domain
        let mut left_result_spectrum =
            vec![num_complex::Complex::new(0.0, 0.0); input_spectrum.len()];
        let mut right_result_spectrum =
            vec![num_complex::Complex::new(0.0, 0.0); input_spectrum.len()];

        for i in 0..input_spectrum.len() {
            left_result_spectrum[i] = input_spectrum[i] * left_hrir_spectrum[i];
            right_result_spectrum[i] = input_spectrum[i] * right_hrir_spectrum[i];
        }

        // IFFT back to time domain
        let mut left_result_time = ifft.make_output_vec();
        let mut right_result_time = ifft.make_output_vec();

        ifft.process(&mut left_result_spectrum, &mut left_result_time)
            .map_err(|e| crate::Error::LegacyProcessing(format!("IFFT error: {e}")))?;

        ifft.process(&mut right_result_spectrum, &mut right_result_time)
            .map_err(|e| crate::Error::LegacyProcessing(format!("IFFT error: {e}")))?;

        // Normalize and copy to output (FFT scaling)
        let scale = 1.0 / fft_len as f32;
        let output_len = left_output.len().min(conv_len);

        for i in 0..output_len {
            left_output[i] = left_result_time[i] * scale;
            right_output[i] = right_result_time[i] * scale;
        }

        Ok(())
    }

    /// Convert Cartesian to spherical coordinates
    fn cartesian_to_spherical(&self, position: &Position3D) -> SphericalCoordinates {
        let distance =
            (position.x * position.x + position.y * position.y + position.z * position.z).sqrt();

        let azimuth = if distance > 0.0 {
            position.z.atan2(position.x).to_degrees()
        } else {
            0.0
        };

        let elevation = if distance > 0.0 {
            (position.y / distance).asin().to_degrees()
        } else {
            0.0
        };

        SphericalCoordinates {
            azimuth,
            elevation,
            distance,
        }
    }

    /// Find closest angle from available angles
    fn find_closest_angle(&self, target: i32, available: &[i32]) -> i32 {
        available
            .iter()
            .min_by_key(|&&angle| (angle - target).abs())
            .copied()
            .unwrap_or(0)
    }

    /// Find lower angle
    fn find_lower_angle(&self, target: i32, available: &[i32]) -> i32 {
        available
            .iter()
            .filter(|&&angle| angle <= target)
            .max()
            .copied()
            .unwrap_or(*available.first().unwrap_or(&0))
    }

    /// Find higher angle
    fn find_higher_angle(&self, target: i32, available: &[i32]) -> i32 {
        available
            .iter()
            .filter(|&&angle| angle >= target)
            .min()
            .copied()
            .unwrap_or(*available.last().unwrap_or(&0))
    }

    /// Get HRTF at specific angles
    fn get_hrtf_at_angles(
        &self,
        azimuth: i32,
        elevation: i32,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        let key = (azimuth, elevation);
        let left = self.database.left_responses.get(&key).ok_or_else(|| {
            crate::Error::LegacyHrtf(format!("No HRTF for angles ({azimuth}, {elevation})"))
        })?;
        let right = self.database.right_responses.get(&key).ok_or_else(|| {
            crate::Error::LegacyHrtf(format!("No HRTF for angles ({azimuth}, {elevation})"))
        })?;
        Ok((left.clone(), right.clone()))
    }

    /// Interpolate HRTF from weighted samples
    fn interpolate_hrtf(&self, weighted_hrirs: &[(&Array1<f32>, f32)]) -> Array1<f32> {
        let mut result = Array1::zeros(self.config.hrir_length);

        for (hrir, weight) in weighted_hrirs {
            for i in 0..result.len().min(hrir.len()) {
                result[i] += hrir[i] * weight;
            }
        }

        result
    }

    /// Calculate angular distance between two points
    fn calculate_angular_distance(&self, az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
        let az1_rad = az1.to_radians();
        let el1_rad = el1.to_radians();
        let az2_rad = az2.to_radians();
        let el2_rad = el2.to_radians();

        // Spherical distance formula
        let cos_distance = el1_rad.sin() * el2_rad.sin()
            + el1_rad.cos() * el2_rad.cos() * (az1_rad - az2_rad).cos();

        cos_distance.clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Apply distance modeling to HRIRs
    fn apply_distance_modeling(
        &self,
        left_hrir: &mut Array1<f32>,
        right_hrir: &mut Array1<f32>,
        coords: &SphericalCoordinates,
    ) -> crate::Result<()> {
        let distance = coords.distance.max(0.01); // Minimum distance to avoid division by zero

        // Distance attenuation (inverse square law)
        let attenuation = 1.0 / distance;

        // Near-field effects (head shadowing and proximity effects)
        let near_field_factor = if distance < self.config.near_field_distance {
            // Enhance proximity effect for close sources
            let proximity_boost = (self.config.near_field_distance / distance).powf(0.3);
            proximity_boost.min(3.0) // Limit boost to avoid excessive amplification
        } else {
            1.0
        };

        // Far-field approximation (simple attenuation)
        let far_field_factor = if distance > self.config.far_field_distance {
            // Apply additional high-frequency attenuation for very distant sources
            0.8 + 0.2 * (self.config.far_field_distance / distance)
        } else {
            1.0
        };

        // Apply combined distance effects
        let total_gain = attenuation * near_field_factor * far_field_factor;

        // Scale HRIRs
        left_hrir.mapv_inplace(|x| x * total_gain);
        right_hrir.mapv_inplace(|x| x * total_gain);

        // Add distance-dependent delay for very close sources
        if distance < self.config.near_field_distance {
            self.apply_proximity_delay(left_hrir, right_hrir, coords)?;
        }

        Ok(())
    }

    /// Apply air absorption based on distance and atmospheric conditions
    fn apply_air_absorption(
        &self,
        left_hrir: &mut Array1<f32>,
        right_hrir: &mut Array1<f32>,
        distance: f32,
    ) -> crate::Result<()> {
        // Calculate frequency-dependent absorption coefficients
        // Based on ISO 9613-1 standard for atmospheric absorption

        let temp_celsius = self.config.temperature;
        let relative_humidity = self.config.humidity;

        // Convert to Kelvin and calculate atmospheric parameters
        let temp_kelvin = temp_celsius + 273.15;
        let temp_ratio = temp_kelvin / 293.15; // Reference temperature 20°C

        // Calculate oxygen and nitrogen absorption
        let h_rel = relative_humidity * (101.325 * temp_ratio.powf(-5.0241));

        // Simplified high-frequency attenuation for air absorption
        // In practice, this would be applied in frequency domain
        let distance_factor = (-distance / 100.0).exp(); // Simple exponential decay
        let temp_factor = temp_ratio.powf(-0.1);
        let humidity_factor = 1.0 - relative_humidity * 0.1;

        let absorption_factor = distance_factor * temp_factor * humidity_factor;

        // Apply frequency-dependent attenuation (simplified)
        // Higher frequencies are attenuated more
        for i in 0..left_hrir.len() {
            let freq_weight = if i as f32 / left_hrir.len() as f32 > 0.5 {
                // Attenuate high frequencies more
                absorption_factor.powf(1.0 + i as f32 / left_hrir.len() as f32)
            } else {
                absorption_factor
            };

            left_hrir[i] *= freq_weight;
            right_hrir[i] *= freq_weight;
        }

        Ok(())
    }

    /// Apply proximity delay effects for very close sources
    fn apply_proximity_delay(
        &self,
        left_hrir: &mut Array1<f32>,
        right_hrir: &mut Array1<f32>,
        coords: &SphericalCoordinates,
    ) -> crate::Result<()> {
        let distance = coords.distance;
        let azimuth_rad = coords.azimuth.to_radians();

        // Calculate inter-aural time difference (ITD) enhancement for close sources
        let head_radius =
            self.config.head_circumference.unwrap_or(57.0) / (2.0 * std::f32::consts::PI);
        let sound_speed = 343.0; // m/s at 20°C

        // Enhanced ITD calculation for near field
        let itd_samples = if distance < head_radius * 2.0 {
            // Use enhanced ITD model for very close sources
            let enhanced_itd =
                (head_radius * azimuth_rad.sin() * (1.0 + azimuth_rad.cos())) / sound_speed;
            (enhanced_itd * self.config.sample_rate as f32) as usize
        } else {
            0
        };

        // Apply fractional delay if needed (simplified integer delay for now)
        if itd_samples > 0 && azimuth_rad.abs() > 0.1 {
            let delay_samples = itd_samples.min(left_hrir.len() / 4);

            if azimuth_rad > 0.0 {
                // Source on the right, delay left ear
                self.apply_delay(left_hrir, delay_samples);
            } else {
                // Source on the left, delay right ear
                self.apply_delay(right_hrir, delay_samples);
            }
        }

        Ok(())
    }

    /// Apply delay to HRIR
    fn apply_delay(&self, hrir: &mut Array1<f32>, delay_samples: usize) {
        if delay_samples == 0 || delay_samples >= hrir.len() {
            return;
        }

        // Shift samples to apply delay
        let original = hrir.clone();
        hrir.fill(0.0);

        for i in 0..(hrir.len() - delay_samples) {
            hrir[i + delay_samples] = original[i];
        }
    }
}

impl HrtfDatabase {
    /// Load HRTF database from file
    pub async fn load_from_file(path: &std::path::Path) -> crate::Result<Self> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        match extension.as_deref() {
            Some("sofa") => Self::load_sofa_file(path).await,
            Some("json") => Self::load_json_file(path).await,
            Some("bin") | Some("hrtf") => Self::load_binary_file(path).await,
            _ => {
                tracing::warn!("Unknown HRTF file format, using default database");
                Self::load_default().await
            }
        }
    }

    /// Load HRTF from SOFA (Spatially Oriented Format for Acoustics) file
    async fn load_sofa_file(path: &std::path::Path) -> crate::Result<Self> {
        tracing::info!("Loading SOFA HRTF file: {:?}", path);

        // For this implementation, we'll handle a simplified text-based SOFA format
        // Real SOFA files are HDF5-based, but this provides basic compatibility
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| crate::Error::hrtf(&format!("Failed to read SOFA file: {e}")))?;

        let mut metadata = HrtfMetadata {
            name: "SOFA HRTF Database".to_string(),
            sample_rate: 44100,
            hrir_length: 512,
            azimuth_angles: Vec::new(),
            elevation_angles: Vec::new(),
            distances: Some(vec![1.0]),
            subject_info: Some(SubjectInfo {
                head_circumference: 56.0,
                head_width: 15.0,
                head_height: 20.0,
                ear_height: 10.0,
                shoulder_width: 40.0,
            }),
        };

        let mut left_responses = HashMap::new();
        let mut right_responses = HashMap::new();
        let mut current_section = "";
        let mut current_measurement: Option<(i32, i32)> = None;
        let mut left_hrir_data = Vec::new();
        let mut right_hrir_data = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse SOFA-like sections
            if line.starts_with('[') && line.ends_with(']') {
                current_section = &line[1..line.len() - 1];
                continue;
            }

            match current_section {
                "GLOBAL" => {
                    if let Some((key, value)) = line.split_once('=') {
                        let key = key.trim();
                        let value = value.trim();
                        match key {
                            "Data.SamplingRate" => {
                                if let Ok(rate) = value.parse::<u32>() {
                                    metadata.sample_rate = rate;
                                }
                            }
                            "Data.IRLength" => {
                                if let Ok(length) = value.parse::<usize>() {
                                    metadata.hrir_length = length;
                                }
                            }
                            "GLOBAL:DatabaseName" => {
                                metadata.name = value.to_string();
                            }
                            "GLOBAL:ListenerShortName" => {
                                metadata.subject_info = Some(SubjectInfo {
                                    head_circumference: 56.0,
                                    head_width: 15.0,
                                    head_height: 20.0,
                                    ear_height: 10.0,
                                    shoulder_width: 40.0,
                                });
                            }
                            _ => {}
                        }
                    }
                }
                "POSITION" => {
                    if line.starts_with("SourcePosition") {
                        // Parse source position: SourcePosition=azimuth,elevation,distance
                        if let Some((_, coords)) = line.split_once('=') {
                            let parts: Vec<&str> = coords.split(',').collect();
                            if parts.len() >= 3 {
                                if let (Ok(azimuth), Ok(elevation), Ok(distance)) = (
                                    parts[0].trim().parse::<f32>(),
                                    parts[1].trim().parse::<f32>(),
                                    parts[2].trim().parse::<f32>(),
                                ) {
                                    current_measurement = Some((azimuth as i32, elevation as i32));
                                    metadata.distances = Some(vec![distance]);
                                }
                            }
                        }
                    }
                }
                "DATA_IR" => {
                    if let Some((azimuth, elevation)) = current_measurement {
                        if let Some(data_str) = line.strip_prefix("L:") {
                            // Left channel HRIR data
                            left_hrir_data = data_str
                                .split_whitespace()
                                .filter_map(|s| s.parse::<f32>().ok())
                                .collect();
                        } else if let Some(data_str) = line.strip_prefix("R:") {
                            // Right channel HRIR data
                            right_hrir_data = data_str
                                .split_whitespace()
                                .filter_map(|s| s.parse::<f32>().ok())
                                .collect();
                        }

                        // If both channels are loaded, store the measurement
                        if !left_hrir_data.is_empty() && !right_hrir_data.is_empty() {
                            left_responses
                                .insert((azimuth, elevation), Array1::from(left_hrir_data.clone()));
                            right_responses.insert(
                                (azimuth, elevation),
                                Array1::from(right_hrir_data.clone()),
                            );

                            if !metadata.azimuth_angles.contains(&azimuth) {
                                metadata.azimuth_angles.push(azimuth);
                            }
                            if !metadata.elevation_angles.contains(&elevation) {
                                metadata.elevation_angles.push(elevation);
                            }

                            left_hrir_data.clear();
                            right_hrir_data.clear();
                            current_measurement = None;
                        }
                    }
                }
                _ => {}
            }
        }

        metadata.azimuth_angles.sort();
        metadata.elevation_angles.sort();

        // If no valid measurements found, use enhanced default
        if left_responses.is_empty() || right_responses.is_empty() {
            tracing::warn!("No valid HRTF measurements found in SOFA file, using enhanced default");
            return Self::load_enhanced_default().await;
        }

        tracing::info!(
            "Successfully loaded {} HRTF measurements from SOFA file",
            left_responses.len()
        );

        Ok(Self {
            metadata,
            left_responses,
            right_responses,
            frequency_responses: None,
            distance_responses: None,
        })
    }

    /// Load HRTF from JSON file
    async fn load_json_file(path: &std::path::Path) -> crate::Result<Self> {
        tracing::info!("Loading JSON HRTF file: {:?}", path);

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| crate::Error::hrtf(&format!("Failed to read JSON file: {e}")))?;

        let json_data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| crate::Error::hrtf(&format!("Failed to parse JSON: {e}")))?;

        // Parse metadata
        let metadata = if let Some(meta) = json_data.get("metadata") {
            HrtfMetadata {
                name: meta
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("JSON HRTF Database")
                    .to_string(),
                sample_rate: meta
                    .get("sample_rate")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(44100) as u32,
                hrir_length: meta
                    .get("hrir_length")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(512) as usize,
                azimuth_angles: meta
                    .get("azimuth_angles")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_i64().map(|i| i as i32))
                            .collect()
                    })
                    .unwrap_or_else(|| (-180..=180).step_by(15).collect()),
                elevation_angles: meta
                    .get("elevation_angles")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_i64().map(|i| i as i32))
                            .collect()
                    })
                    .unwrap_or_else(|| (-40..=90).step_by(10).collect()),
                distances: meta
                    .get("distance")
                    .and_then(|v| v.as_f64())
                    .map(|d| vec![d as f32]),
                subject_info: meta.get("subject_id").and_then(|v| v.as_str()).map(|_| {
                    SubjectInfo {
                        head_circumference: 56.0,
                        head_width: 15.0,
                        head_height: 20.0,
                        ear_height: 10.0,
                        shoulder_width: 40.0,
                    }
                }),
            }
        } else {
            return Self::load_enhanced_default().await;
        };

        // Parse measurements
        let mut left_responses = HashMap::new();
        let mut right_responses = HashMap::new();

        if let Some(measurements) = json_data.get("measurements").and_then(|v| v.as_array()) {
            for measurement in measurements {
                let azimuth = measurement
                    .get("azimuth")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let elevation = measurement
                    .get("elevation")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;

                // Parse left HRIR
                if let Some(left_hrir) = measurement.get("left_hrir").and_then(|v| v.as_array()) {
                    let left_data: Vec<f32> = left_hrir
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    if !left_data.is_empty() {
                        left_responses.insert((azimuth, elevation), Array1::from(left_data));
                    }
                }

                // Parse right HRIR
                if let Some(right_hrir) = measurement.get("right_hrir").and_then(|v| v.as_array()) {
                    let right_data: Vec<f32> = right_hrir
                        .iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect();
                    if !right_data.is_empty() {
                        right_responses.insert((azimuth, elevation), Array1::from(right_data));
                    }
                }
            }
        }

        // If no valid measurements found, use enhanced default
        if left_responses.is_empty() || right_responses.is_empty() {
            tracing::warn!("No valid HRTF measurements found in JSON file, using enhanced default");
            return Self::load_enhanced_default().await;
        }

        tracing::info!(
            "Successfully loaded {} HRTF measurements from JSON",
            left_responses.len()
        );

        Ok(Self {
            metadata,
            left_responses,
            right_responses,
            frequency_responses: None,
            distance_responses: None,
        })
    }

    /// Load HRTF from binary file
    async fn load_binary_file(path: &std::path::Path) -> crate::Result<Self> {
        tracing::info!("Loading binary HRTF file: {:?}", path);

        let data = tokio::fs::read(path)
            .await
            .map_err(|e| crate::Error::hrtf(&format!("Failed to read binary file: {e}")))?;

        if data.len() < 32 {
            return Err(crate::Error::hrtf(
                "Binary file too small to contain valid HRTF data",
            ));
        }

        let mut cursor = 0;

        // Parse binary header (simple format)
        // Magic number (4 bytes): "HRTF"
        let magic = &data[cursor..cursor + 4];
        if magic != b"HRTF" {
            return Err(crate::Error::hrtf("Invalid binary HRTF file format"));
        }
        cursor += 4;

        // Version (4 bytes)
        let version = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        cursor += 4;
        if version != 1 {
            return Err(crate::Error::hrtf(&format!(
                "Unsupported HRTF binary version: {version}"
            )));
        }

        // Sample rate (4 bytes)
        let sample_rate = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        cursor += 4;

        // HRIR length (4 bytes)
        let hrir_length = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Number of measurements (4 bytes)
        let measurement_count = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Distance (4 bytes)
        let distance = f32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        cursor += 4;

        // Subject ID length (4 bytes) + subject ID
        let subject_id_len = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        if cursor + subject_id_len > data.len() {
            return Err(crate::Error::hrtf(
                "Invalid subject ID length in binary file",
            ));
        }

        let subject_id =
            String::from_utf8_lossy(&data[cursor..cursor + subject_id_len]).to_string();
        cursor += subject_id_len;

        // Parse measurements
        let mut left_responses = HashMap::new();
        let mut right_responses = HashMap::new();
        let mut azimuth_angles = Vec::new();
        let mut elevation_angles = Vec::new();

        for _ in 0..measurement_count {
            if cursor + 8 + (hrir_length * 8) > data.len() {
                return Err(crate::Error::hrtf("Insufficient data for measurement"));
            }

            // Azimuth (4 bytes)
            let azimuth = i32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            // Elevation (4 bytes)
            let elevation = i32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            // Left HRIR (hrir_length * 4 bytes)
            let mut left_hrir = Vec::with_capacity(hrir_length);
            for _ in 0..hrir_length {
                let sample = f32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]);
                left_hrir.push(sample);
                cursor += 4;
            }

            // Right HRIR (hrir_length * 4 bytes)
            let mut right_hrir = Vec::with_capacity(hrir_length);
            for _ in 0..hrir_length {
                let sample = f32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]);
                right_hrir.push(sample);
                cursor += 4;
            }

            left_responses.insert((azimuth, elevation), Array1::from(left_hrir));
            right_responses.insert((azimuth, elevation), Array1::from(right_hrir));

            if !azimuth_angles.contains(&azimuth) {
                azimuth_angles.push(azimuth);
            }
            if !elevation_angles.contains(&elevation) {
                elevation_angles.push(elevation);
            }
        }

        azimuth_angles.sort();
        elevation_angles.sort();

        if left_responses.is_empty() || right_responses.is_empty() {
            return Err(crate::Error::hrtf(
                "No valid HRTF measurements found in binary file",
            ));
        }

        let metadata = HrtfMetadata {
            name: format!("Binary HRTF Database ({subject_id})"),
            sample_rate,
            hrir_length,
            azimuth_angles,
            elevation_angles,
            distances: Some(vec![distance]),
            subject_info: Some(SubjectInfo {
                head_circumference: 56.0,
                head_width: 15.0,
                head_height: 20.0,
                ear_height: 10.0,
                shoulder_width: 40.0,
            }),
        };

        tracing::info!(
            "Successfully loaded {} HRTF measurements from binary file",
            left_responses.len()
        );

        Ok(Self {
            metadata,
            left_responses,
            right_responses,
            frequency_responses: None,
            distance_responses: None,
        })
    }

    /// Create enhanced default HRTF database with higher quality measurements
    async fn load_enhanced_default() -> crate::Result<Self> {
        let metadata = HrtfMetadata {
            name: "Enhanced Default HRTF".to_string(),
            sample_rate: 44100,
            hrir_length: 512, // Longer impulse responses for better quality
            azimuth_angles: (-180..=180).step_by(5).collect(), // Higher resolution
            elevation_angles: (-90..=90).step_by(5).collect(), // Higher resolution
            distances: Some(vec![0.2, 0.5, 1.0, 2.0, 5.0]), // Multiple distance measurements
            subject_info: Some(SubjectInfo {
                head_circumference: 57.0,
                head_width: 15.5,
                head_height: 24.0,
                ear_height: 12.0,
                shoulder_width: 45.0,
            }),
        };

        let mut left_responses = HashMap::new();
        let mut right_responses = HashMap::new();
        let mut distance_responses = HashMap::new();

        // Generate enhanced HRTFs with multiple distances
        for &azimuth in &metadata.azimuth_angles {
            for &elevation in &metadata.elevation_angles {
                for &distance in metadata.distances.as_ref().unwrap() {
                    let (left_hrir, right_hrir) = Self::generate_enhanced_hrtf(
                        azimuth,
                        elevation,
                        distance,
                        metadata.hrir_length,
                    );

                    // Store default distance (1.0m)
                    if (distance - 1.0).abs() < 0.1 {
                        left_responses.insert((azimuth, elevation), left_hrir.clone());
                        right_responses.insert((azimuth, elevation), right_hrir.clone());
                    }

                    // Store distance-specific responses
                    let distance_key = (distance * 100.0) as u32; // Store as cm
                    distance_responses
                        .insert((azimuth, elevation, distance_key), (left_hrir, right_hrir));
                }
            }
        }

        Ok(Self {
            metadata,
            left_responses,
            right_responses,
            frequency_responses: None,
            distance_responses: Some(distance_responses),
        })
    }

    /// Load default HRTF database
    pub async fn load_default() -> crate::Result<Self> {
        // Create a simple default HRTF database
        let metadata = HrtfMetadata {
            name: "Default HRTF".to_string(),
            sample_rate: 44100,
            hrir_length: 256,
            azimuth_angles: (-180..=180).step_by(15).collect(),
            elevation_angles: (-90..=90).step_by(15).collect(),
            distances: None,
            subject_info: None,
        };

        let mut left_responses = HashMap::new();
        let mut right_responses = HashMap::new();

        // Generate simple HRTFs (in practice, these would be measured)
        for &azimuth in &metadata.azimuth_angles {
            for &elevation in &metadata.elevation_angles {
                let (left_hrir, right_hrir) =
                    Self::generate_simple_hrtf(azimuth, elevation, metadata.hrir_length);
                left_responses.insert((azimuth, elevation), left_hrir);
                right_responses.insert((azimuth, elevation), right_hrir);
            }
        }

        Ok(Self {
            metadata,
            left_responses,
            right_responses,
            frequency_responses: None,
            distance_responses: None,
        })
    }

    /// Generate simple HRTF for testing
    fn generate_simple_hrtf(
        azimuth: i32,
        _elevation: i32,
        length: usize,
    ) -> (Array1<f32>, Array1<f32>) {
        let mut left_hrir = Array1::zeros(length);
        let mut right_hrir = Array1::zeros(length);

        // Very simple HRTF model - just delay and attenuation based on angle
        let _azimuth_rad = (azimuth as f32).to_radians();

        // Left ear delay (negative azimuth = sound from left)
        let left_delay = if azimuth < 0 {
            0
        } else {
            (azimuth as f32 / 180.0 * 10.0) as usize
        };
        let left_gain = 1.0 - (azimuth as f32).abs() / 180.0 * 0.3;

        // Right ear delay
        let right_delay = if azimuth > 0 {
            0
        } else {
            ((-azimuth) as f32 / 180.0 * 10.0) as usize
        };
        let right_gain = 1.0 - (azimuth as f32).abs() / 180.0 * 0.3;

        // Set impulse responses
        if left_delay < length {
            left_hrir[left_delay] = left_gain;
        }
        if right_delay < length {
            right_hrir[right_delay] = right_gain;
        }

        (left_hrir, right_hrir)
    }

    /// Generate enhanced HRTF with distance modeling
    fn generate_enhanced_hrtf(
        azimuth: i32,
        elevation: i32,
        distance: f32,
        length: usize,
    ) -> (Array1<f32>, Array1<f32>) {
        let sample_rate = 44100.0; // Use standard sample rate for calculations
        let mut left_hrir = Array1::zeros(length);
        let mut right_hrir = Array1::zeros(length);

        // More sophisticated HRTF model with distance modeling
        let azimuth_rad = (azimuth as f32).to_radians();
        let elevation_rad = (elevation as f32).to_radians();

        // Head model parameters
        let head_radius = 0.09; // Average head radius in meters

        // Distance attenuation (inverse square law)
        let distance_attenuation = 1.0 / (distance + 0.01);

        // ITD (Interaural Time Difference) calculation using Woodworth model
        let itd = if azimuth_rad.abs() <= std::f32::consts::PI / 2.0 {
            // For azimuth <= 90 degrees
            (head_radius / 343.0) * (azimuth_rad + azimuth_rad.sin()) * sample_rate
        } else {
            // For azimuth > 90 degrees
            (head_radius / 343.0) * (std::f32::consts::PI / 2.0 + azimuth_rad.sin()) * sample_rate
        };

        // Convert ITD to sample delay
        let left_delay = if azimuth >= 0 {
            (itd / 2.0) as usize
        } else {
            0
        };
        let right_delay = if azimuth < 0 {
            (-itd / 2.0) as usize
        } else {
            0
        };

        // ILD (Interaural Level Difference) - frequency-dependent attenuation
        let frequency_factor = 1.0; // Simplified - in practice this would be frequency-dependent
        let shadow_attenuation = if azimuth_rad.abs() > std::f32::consts::PI / 2.0 {
            0.3
        } else {
            0.0
        };

        let left_gain = distance_attenuation
            * frequency_factor
            * (1.0 - if azimuth > 0 { shadow_attenuation } else { 0.0 });
        let right_gain = distance_attenuation
            * frequency_factor
            * (1.0 - if azimuth < 0 { shadow_attenuation } else { 0.0 });

        // Elevation effects (simplified pinna filtering)
        let elevation_gain = (1.0 + 0.2 * elevation_rad.sin()).clamp(0.5, 1.5);

        // Near-field effects for close distances
        let near_field_boost = if distance < 0.5 {
            1.0 + (0.5 - distance) * 0.5
        } else {
            1.0
        };

        // Generate impulse response with multiple components
        let primary_delay = (distance / 343.0 * sample_rate) as usize;

        // Primary arrival
        if primary_delay + left_delay < length {
            left_hrir[primary_delay + left_delay] = left_gain * elevation_gain * near_field_boost;
        }
        if primary_delay + right_delay < length {
            right_hrir[primary_delay + right_delay] =
                right_gain * elevation_gain * near_field_boost;
        }

        // Add some early reflections for realism (simplified)
        let reflection_delay = primary_delay + (0.002 * sample_rate) as usize;
        if reflection_delay < length {
            let reflection_gain = 0.1 * distance_attenuation;
            if reflection_delay + left_delay < length {
                left_hrir[reflection_delay + left_delay] += reflection_gain;
            }
            if reflection_delay + right_delay < length {
                right_hrir[reflection_delay + right_delay] += reflection_gain;
            }
        }

        // Apply a simple smoothing window to make it more realistic
        let window_size = (length / 8).min(32);
        for i in 0..window_size {
            let window_val =
                0.5 * (1.0 - ((i as f32) / (window_size as f32) * std::f32::consts::PI).cos());
            if i < left_hrir.len() {
                left_hrir[i] *= window_val;
            }
            if i < right_hrir.len() {
                right_hrir[i] *= window_val;
            }
        }

        (left_hrir, right_hrir)
    }

    /// Get metadata
    pub fn metadata(&self) -> &HrtfMetadata {
        &self.metadata
    }

    /// Get available positions
    pub fn available_positions(&self) -> Vec<SphericalCoordinates> {
        let mut positions = Vec::new();
        for &azimuth in &self.metadata.azimuth_angles {
            for &elevation in &self.metadata.elevation_angles {
                positions.push(SphericalCoordinates {
                    azimuth: azimuth as f32,
                    elevation: elevation as f32,
                    distance: 1.0, // Default distance
                });
            }
        }
        positions
    }
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            hrir_length: 256,
            crossfade_time: 0.01, // 10ms
            enable_distance_modeling: true,
            interpolation_method: InterpolationMethod::Bilinear,
            head_circumference: None,
            near_field_distance: 0.2, // 20cm - near field starts
            far_field_distance: 10.0, // 10m - far field approximation
            enable_air_absorption: true,
            temperature: 20.0, // 20°C
            humidity: 0.5,     // 50% relative humidity
            enable_simd: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hrtf_processor_creation() {
        let processor = HrtfProcessor::new(None).await;
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_hrtf_database_loading() {
        let database = HrtfDatabase::load_default().await;
        assert!(database.is_ok());

        let db = database.unwrap();
        assert!(!db.left_responses.is_empty());
        assert!(!db.right_responses.is_empty());
        assert_eq!(db.left_responses.len(), db.right_responses.len());
    }

    #[tokio::test]
    async fn test_cartesian_to_spherical() {
        let processor = HrtfProcessor::new(None).await.unwrap();

        // Test point to the right
        let pos = Position3D::new(1.0, 0.0, 0.0);
        let spherical = processor.cartesian_to_spherical(&pos);
        assert!((spherical.azimuth - 0.0).abs() < 0.1);
        assert!((spherical.elevation - 0.0).abs() < 0.1);

        // Test point to the front
        let pos = Position3D::new(0.0, 0.0, 1.0);
        let spherical = processor.cartesian_to_spherical(&pos);
        assert!((spherical.azimuth - 90.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_hrtf_processing() {
        let processor = HrtfProcessor::new(None).await.unwrap();
        let input = Array1::from_vec(vec![1.0, 0.5, -0.5, -1.0]);
        let mut left_output = Array1::zeros(input.len());
        let mut right_output = Array1::zeros(input.len());

        let position = Position3D::new(1.0, 0.0, 0.0);
        let result = processor
            .process_position(&input, &mut left_output, &mut right_output, &position)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_realtime_processing() {
        let mut processor = HrtfProcessor::new(None).await.unwrap();
        let chunk_size = 64;
        let input = Array1::from_vec(vec![0.1; chunk_size]);
        let mut left_output = Array1::zeros(chunk_size);
        let mut right_output = Array1::zeros(chunk_size);

        let position = Position3D::new(1.0, 0.0, 0.0);

        // Process first chunk
        let result1 = processor
            .process_realtime_chunk(&input, &mut left_output, &mut right_output, &position)
            .await;
        assert!(result1.is_ok());

        // Process second chunk (should use overlap-add)
        let result2 = processor
            .process_realtime_chunk(&input, &mut left_output, &mut right_output, &position)
            .await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_interpolation_methods() {
        // Test different interpolation methods
        let configs = [
            (InterpolationMethod::Nearest, "Nearest"),
            (InterpolationMethod::Bilinear, "Bilinear"),
            (InterpolationMethod::Spherical, "Spherical"),
            (InterpolationMethod::Weighted, "Weighted"),
        ];

        for (method, name) in configs {
            let config = HrtfConfig {
                interpolation_method: method,
                ..Default::default()
            };

            let processor = HrtfProcessor::with_config(None, config).await.unwrap();
            let coords = SphericalCoordinates {
                azimuth: 45.0,
                elevation: 15.0,
                distance: 1.0,
            };

            let result = processor.get_hrtf(&coords);
            assert!(result.is_ok(), "Failed interpolation method: {}", name);
        }
    }

    #[tokio::test]
    async fn test_crossfade_processing() {
        let processor = HrtfProcessor::new(None).await.unwrap();
        let input = Array1::from_vec(vec![1.0, 0.5, -0.5, -1.0]);
        let mut left_output = Array1::zeros(input.len());
        let mut right_output = Array1::zeros(input.len());

        let positions = vec![
            (Position3D::new(1.0, 0.0, 0.0), 0.7),  // 70% weight
            (Position3D::new(-1.0, 0.0, 0.0), 0.3), // 30% weight
        ];

        let result = processor
            .process_crossfade(&input, &mut left_output, &mut right_output, &positions)
            .await;
        assert!(result.is_ok());

        // Check that output is not all zeros
        let left_sum: f32 = left_output.iter().map(|x| x.abs()).sum();
        let right_sum: f32 = right_output.iter().map(|x| x.abs()).sum();
        assert!(left_sum > 0.0);
        assert!(right_sum > 0.0);
    }

    #[tokio::test]
    async fn test_frequency_domain_convolution() {
        let processor = HrtfProcessor::new(None).await.unwrap();

        // Create input large enough to trigger frequency-domain convolution
        let input = Array1::from_vec(vec![0.1; 128]);
        let hrir_len = processor.config.hrir_length;
        let left_hrir = Array1::from_vec(vec![1.0; hrir_len]);
        let right_hrir = Array1::from_vec(vec![0.8; hrir_len]);

        let mut left_output = Array1::zeros(input.len());
        let mut right_output = Array1::zeros(input.len());

        let result = processor.convolve_hrtf(
            &input,
            &left_hrir,
            &right_hrir,
            &mut left_output,
            &mut right_output,
        );
        assert!(result.is_ok());

        // Check that convolution produced non-zero output
        let left_energy: f32 = left_output.iter().map(|x| x * x).sum();
        let right_energy: f32 = right_output.iter().map(|x| x * x).sum();
        assert!(left_energy > 0.0);
        assert!(right_energy > 0.0);
    }
}
