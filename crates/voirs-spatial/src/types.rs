//! Core types for spatial audio processing

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 3D position in space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    /// X coordinate (left-right)
    pub x: f32,
    /// Y coordinate (up-down)
    pub y: f32,
    /// Z coordinate (front-back)
    pub z: f32,
}

impl Position3D {
    /// Create new position
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Distance to another position
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate dot product with another position vector
    pub fn dot(&self, other: &Position3D) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Get normalized vector (unit vector)
    pub fn normalized(&self) -> Position3D {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if length > 0.0 {
            Position3D::new(self.x / length, self.y / length, self.z / length)
        } else {
            *self
        }
    }

    /// Get magnitude/length of the vector
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Cross product with another vector
    pub fn cross(&self, other: &Position3D) -> Position3D {
        Position3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Vector addition
    pub fn add(&self, other: &Position3D) -> Position3D {
        Position3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Vector subtraction
    pub fn sub(&self, other: &Position3D) -> Position3D {
        Position3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Position3D {
        Position3D::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    /// Linear interpolation between two positions
    pub fn lerp(&self, other: &Position3D, t: f32) -> Position3D {
        Position3D::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }
}

/// SIMD-optimized spatial calculations
pub struct SIMDSpatialOps;

impl SIMDSpatialOps {
    /// Calculate distances from one position to multiple positions using SIMD
    #[cfg(target_feature = "sse2")]
    pub fn distances_simd(from: Position3D, positions: &[Position3D]) -> Vec<f32> {
        let mut distances = Vec::with_capacity(positions.len());

        unsafe {
            let from_x = _mm_set1_ps(from.x);
            let from_y = _mm_set1_ps(from.y);
            let from_z = _mm_set1_ps(from.z);

            let mut i = 0;
            while i + 4 <= positions.len() {
                // Load 4 positions at once
                let x_vals = _mm_set_ps(
                    positions[i + 3].x,
                    positions[i + 2].x,
                    positions[i + 1].x,
                    positions[i].x,
                );
                let y_vals = _mm_set_ps(
                    positions[i + 3].y,
                    positions[i + 2].y,
                    positions[i + 1].y,
                    positions[i].y,
                );
                let z_vals = _mm_set_ps(
                    positions[i + 3].z,
                    positions[i + 2].z,
                    positions[i + 1].z,
                    positions[i].z,
                );

                // Calculate differences
                let dx = _mm_sub_ps(x_vals, from_x);
                let dy = _mm_sub_ps(y_vals, from_y);
                let dz = _mm_sub_ps(z_vals, from_z);

                // Square the differences
                let dx2 = _mm_mul_ps(dx, dx);
                let dy2 = _mm_mul_ps(dy, dy);
                let dz2 = _mm_mul_ps(dz, dz);

                // Sum and take square root
                let sum = _mm_add_ps(_mm_add_ps(dx2, dy2), dz2);
                let sqrt_vals = _mm_sqrt_ps(sum);

                // Extract results
                let mut result = [0.0f32; 4];
                _mm_storeu_ps(result.as_mut_ptr(), sqrt_vals);

                for j in 0..4 {
                    distances.push(result[j]);
                }

                i += 4;
            }

            // Handle remaining positions
            for pos in &positions[i..] {
                distances.push(from.distance_to(pos));
            }
        }

        distances
    }

    /// Fallback non-SIMD version for compatibility
    pub fn distances_fallback(from: Position3D, positions: &[Position3D]) -> Vec<f32> {
        positions.iter().map(|pos| from.distance_to(pos)).collect()
    }

    /// Automatic distance calculation with SIMD when available
    pub fn distances(from: Position3D, positions: &[Position3D]) -> Vec<f32> {
        #[cfg(all(target_feature = "sse2", target_arch = "x86_64"))]
        {
            Self::distances_simd(from, positions)
        }
        #[cfg(not(all(target_feature = "sse2", target_arch = "x86_64")))]
        {
            Self::distances_fallback(from, positions)
        }
    }

    /// SIMD-optimized vector normalization for multiple positions
    #[cfg(target_feature = "sse2")]
    pub fn normalize_batch_simd(positions: &mut [Position3D]) {
        unsafe {
            let mut i = 0;
            while i + 4 <= positions.len() {
                // Load 4 positions
                let x_vals = _mm_set_ps(
                    positions[i + 3].x,
                    positions[i + 2].x,
                    positions[i + 1].x,
                    positions[i].x,
                );
                let y_vals = _mm_set_ps(
                    positions[i + 3].y,
                    positions[i + 2].y,
                    positions[i + 1].y,
                    positions[i].y,
                );
                let z_vals = _mm_set_ps(
                    positions[i + 3].z,
                    positions[i + 2].z,
                    positions[i + 1].z,
                    positions[i].z,
                );

                // Calculate magnitude squared
                let x2 = _mm_mul_ps(x_vals, x_vals);
                let y2 = _mm_mul_ps(y_vals, y_vals);
                let z2 = _mm_mul_ps(z_vals, z_vals);
                let mag2 = _mm_add_ps(_mm_add_ps(x2, y2), z2);

                // Calculate reciprocal square root (1/sqrt(mag2))
                let rsqrt = _mm_rsqrt_ps(mag2);

                // Refine the approximation (Newton-Raphson iteration)
                let half = _mm_set1_ps(0.5);
                let three = _mm_set1_ps(3.0);
                let rsqrt_refined = _mm_mul_ps(
                    rsqrt,
                    _mm_sub_ps(three, _mm_mul_ps(_mm_mul_ps(mag2, rsqrt), rsqrt)),
                );
                let inv_mag = _mm_mul_ps(half, rsqrt_refined);

                // Normalize the vectors
                let norm_x = _mm_mul_ps(x_vals, inv_mag);
                let norm_y = _mm_mul_ps(y_vals, inv_mag);
                let norm_z = _mm_mul_ps(z_vals, inv_mag);

                // Store results back
                let mut x_result = [0.0f32; 4];
                let mut y_result = [0.0f32; 4];
                let mut z_result = [0.0f32; 4];

                _mm_storeu_ps(x_result.as_mut_ptr(), norm_x);
                _mm_storeu_ps(y_result.as_mut_ptr(), norm_y);
                _mm_storeu_ps(z_result.as_mut_ptr(), norm_z);

                for j in 0..4 {
                    positions[i + j].x = x_result[j];
                    positions[i + j].y = y_result[j];
                    positions[i + j].z = z_result[j];
                }

                i += 4;
            }

            // Handle remaining positions
            for pos in &mut positions[i..] {
                *pos = pos.normalized();
            }
        }
    }

    /// Fallback batch normalization
    pub fn normalize_batch_fallback(positions: &mut [Position3D]) {
        for pos in positions {
            *pos = pos.normalized();
        }
    }

    /// Automatic batch normalization
    pub fn normalize_batch(positions: &mut [Position3D]) {
        #[cfg(all(target_feature = "sse2", target_arch = "x86_64"))]
        {
            Self::normalize_batch_simd(positions);
        }
        #[cfg(not(all(target_feature = "sse2", target_arch = "x86_64")))]
        {
            Self::normalize_batch_fallback(positions);
        }
    }

    /// SIMD-optimized dot product calculations
    #[cfg(target_feature = "sse2")]
    pub fn dot_products_simd(a_positions: &[Position3D], b_positions: &[Position3D]) -> Vec<f32> {
        assert_eq!(a_positions.len(), b_positions.len());
        let mut results = Vec::with_capacity(a_positions.len());

        unsafe {
            let mut i = 0;
            while i + 4 <= a_positions.len() {
                // Load positions A
                let ax = _mm_set_ps(
                    a_positions[i + 3].x,
                    a_positions[i + 2].x,
                    a_positions[i + 1].x,
                    a_positions[i].x,
                );
                let ay = _mm_set_ps(
                    a_positions[i + 3].y,
                    a_positions[i + 2].y,
                    a_positions[i + 1].y,
                    a_positions[i].y,
                );
                let az = _mm_set_ps(
                    a_positions[i + 3].z,
                    a_positions[i + 2].z,
                    a_positions[i + 1].z,
                    a_positions[i].z,
                );

                // Load positions B
                let bx = _mm_set_ps(
                    b_positions[i + 3].x,
                    b_positions[i + 2].x,
                    b_positions[i + 1].x,
                    b_positions[i].x,
                );
                let by = _mm_set_ps(
                    b_positions[i + 3].y,
                    b_positions[i + 2].y,
                    b_positions[i + 1].y,
                    b_positions[i].y,
                );
                let bz = _mm_set_ps(
                    b_positions[i + 3].z,
                    b_positions[i + 2].z,
                    b_positions[i + 1].z,
                    b_positions[i].z,
                );

                // Calculate dot products
                let xx = _mm_mul_ps(ax, bx);
                let yy = _mm_mul_ps(ay, by);
                let zz = _mm_mul_ps(az, bz);
                let dots = _mm_add_ps(_mm_add_ps(xx, yy), zz);

                // Extract results
                let mut result = [0.0f32; 4];
                _mm_storeu_ps(result.as_mut_ptr(), dots);

                for j in 0..4 {
                    results.push(result[j]);
                }

                i += 4;
            }

            // Handle remaining pairs
            for j in i..a_positions.len() {
                results.push(a_positions[j].dot(&b_positions[j]));
            }
        }

        results
    }

    /// Fallback dot product calculation
    pub fn dot_products_fallback(
        a_positions: &[Position3D],
        b_positions: &[Position3D],
    ) -> Vec<f32> {
        assert_eq!(a_positions.len(), b_positions.len());
        a_positions
            .iter()
            .zip(b_positions.iter())
            .map(|(a, b)| a.dot(b))
            .collect()
    }

    /// Automatic dot product calculation
    pub fn dot_products(a_positions: &[Position3D], b_positions: &[Position3D]) -> Vec<f32> {
        #[cfg(all(target_feature = "sse2", target_arch = "x86_64"))]
        {
            Self::dot_products_simd(a_positions, b_positions)
        }
        #[cfg(not(all(target_feature = "sse2", target_arch = "x86_64")))]
        {
            Self::dot_products_fallback(a_positions, b_positions)
        }
    }
}

impl Default for Position3D {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Audio channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioChannel {
    /// Mono audio
    Mono,
    /// Stereo audio (left/right)
    Stereo,
    /// Binaural audio for headphones
    Binaural,
    /// 5.1 surround sound
    Surround5_1,
    /// 7.1 surround sound
    Surround7_1,
}

/// Binaural audio with left and right channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinauraAudio {
    /// Left channel samples
    pub left: Vec<f32>,
    /// Right channel samples
    pub right: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

impl BinauraAudio {
    /// Create new binaural audio
    pub fn new(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            left,
            right,
            sample_rate,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.left.len() as f32 / self.sample_rate as f32
    }
}

/// Spatial effect types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialEffect {
    /// HRTF-based binaural rendering
    Hrtf,
    /// Room reverberation
    Reverb,
    /// Distance attenuation
    DistanceAttenuation,
    /// Doppler effect
    Doppler,
    /// Air absorption
    AirAbsorption,
}

/// Request for spatial audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialRequest {
    /// Request ID
    pub id: String,
    /// Input audio (mono)
    pub audio: Vec<f32>,
    /// Input sample rate
    pub sample_rate: u32,
    /// Sound source position
    pub source_position: Position3D,
    /// Listener position
    pub listener_position: Position3D,
    /// Listener orientation (yaw, pitch, roll in radians)
    pub listener_orientation: (f32, f32, f32),
    /// Effects to apply
    pub effects: Vec<SpatialEffect>,
    /// Processing parameters
    pub parameters: std::collections::HashMap<String, f32>,
}

impl SpatialRequest {
    /// Create new spatial request
    pub fn new(
        id: String,
        audio: Vec<f32>,
        sample_rate: u32,
        source_position: Position3D,
        listener_position: Position3D,
    ) -> Self {
        Self {
            id,
            audio,
            sample_rate,
            source_position,
            listener_position,
            listener_orientation: (0.0, 0.0, 0.0),
            effects: vec![SpatialEffect::Hrtf],
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Validate the request
    pub fn validate(&self) -> crate::Result<()> {
        if self.audio.is_empty() {
            return Err(crate::Error::LegacyValidation(
                "Audio cannot be empty".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(crate::Error::LegacyValidation(
                "Sample rate must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of spatial audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialResult {
    /// Request ID
    pub request_id: String,
    /// Processed binaural audio
    pub audio: BinauraAudio,
    /// Processing time
    pub processing_time: Duration,
    /// Applied effects
    pub applied_effects: Vec<SpatialEffect>,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

impl SpatialResult {
    /// Create successful result
    pub fn success(
        request_id: String,
        audio: BinauraAudio,
        processing_time: Duration,
        applied_effects: Vec<SpatialEffect>,
    ) -> Self {
        Self {
            request_id,
            audio,
            processing_time,
            applied_effects,
            success: true,
            error_message: None,
        }
    }

    /// Create failed result
    pub fn failure(request_id: String, error_message: String, processing_time: Duration) -> Self {
        Self {
            request_id,
            audio: BinauraAudio::new(Vec::new(), Vec::new(), 0),
            processing_time,
            applied_effects: Vec::new(),
            success: false,
            error_message: Some(error_message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let pos1 = Position3D::new(0.0, 0.0, 0.0);
        let pos2 = Position3D::new(3.0, 4.0, 0.0);
        assert_eq!(pos1.distance_to(&pos2), 5.0);
    }

    #[test]
    fn test_binaural_audio_duration() {
        let audio = BinauraAudio::new(vec![0.0; 44100], vec![0.0; 44100], 44100);
        assert_eq!(audio.duration(), 1.0);
    }

    #[test]
    fn test_spatial_request_validation() {
        let request = SpatialRequest::new(
            "test".to_string(),
            vec![0.1, 0.2, 0.3],
            44100,
            Position3D::default(),
            Position3D::default(),
        );
        assert!(request.validate().is_ok());

        let empty_request = SpatialRequest::new(
            "test".to_string(),
            Vec::new(),
            44100,
            Position3D::default(),
            Position3D::default(),
        );
        assert!(empty_request.validate().is_err());
    }

    #[test]
    fn test_position3d_additional_methods() {
        let pos1 = Position3D::new(1.0, 2.0, 3.0);
        let pos2 = Position3D::new(4.0, 5.0, 6.0);

        // Test vector addition
        let sum = pos1.add(&pos2);
        assert_eq!(sum, Position3D::new(5.0, 7.0, 9.0));

        // Test vector subtraction
        let diff = pos2.sub(&pos1);
        assert_eq!(diff, Position3D::new(3.0, 3.0, 3.0));

        // Test scalar multiplication
        let scaled = pos1.scale(2.0);
        assert_eq!(scaled, Position3D::new(2.0, 4.0, 6.0));

        // Test dot product
        let dot = pos1.dot(&pos2);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        // Test magnitude
        let pos = Position3D::new(3.0, 4.0, 0.0);
        assert_eq!(pos.magnitude(), 5.0);

        // Test normalization
        let normalized = pos.normalized();
        assert!((normalized.magnitude() - 1.0).abs() < 0.001);

        // Test linear interpolation
        let pos_a = Position3D::new(0.0, 0.0, 0.0);
        let pos_b = Position3D::new(10.0, 10.0, 10.0);
        let mid = pos_a.lerp(&pos_b, 0.5);
        assert_eq!(mid, Position3D::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_simd_distance_fallback() {
        let from = Position3D::new(0.0, 0.0, 0.0);
        let positions = vec![
            Position3D::new(1.0, 0.0, 0.0),
            Position3D::new(0.0, 1.0, 0.0),
            Position3D::new(0.0, 0.0, 1.0),
            Position3D::new(3.0, 4.0, 0.0),
        ];

        let distances = SIMDSpatialOps::distances_fallback(from, &positions);

        assert_eq!(distances.len(), 4);
        assert!((distances[0] - 1.0).abs() < 0.001);
        assert!((distances[1] - 1.0).abs() < 0.001);
        assert!((distances[2] - 1.0).abs() < 0.001);
        assert!((distances[3] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_distance_automatic() {
        let from = Position3D::new(0.0, 0.0, 0.0);
        let positions = vec![
            Position3D::new(1.0, 0.0, 0.0),
            Position3D::new(0.0, 1.0, 0.0),
            Position3D::new(0.0, 0.0, 1.0),
            Position3D::new(3.0, 4.0, 0.0),
        ];

        let distances = SIMDSpatialOps::distances(from, &positions);

        assert_eq!(distances.len(), 4);
        assert!((distances[0] - 1.0).abs() < 0.001);
        assert!((distances[1] - 1.0).abs() < 0.001);
        assert!((distances[2] - 1.0).abs() < 0.001);
        assert!((distances[3] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_normalize_batch_fallback() {
        let mut positions = vec![
            Position3D::new(3.0, 4.0, 0.0),
            Position3D::new(5.0, 12.0, 0.0),
            Position3D::new(8.0, 15.0, 0.0),
        ];

        SIMDSpatialOps::normalize_batch_fallback(&mut positions);

        for pos in &positions {
            assert!((pos.magnitude() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_normalize_batch_automatic() {
        let mut positions = vec![
            Position3D::new(3.0, 4.0, 0.0),
            Position3D::new(5.0, 12.0, 0.0),
            Position3D::new(8.0, 15.0, 0.0),
            Position3D::new(1.0, 2.0, 3.0),
            Position3D::new(4.0, 5.0, 6.0),
        ];

        SIMDSpatialOps::normalize_batch(&mut positions);

        for pos in &positions {
            assert!((pos.magnitude() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_dot_products_fallback() {
        let a_positions = vec![
            Position3D::new(1.0, 2.0, 3.0),
            Position3D::new(4.0, 5.0, 6.0),
            Position3D::new(7.0, 8.0, 9.0),
        ];
        let b_positions = vec![
            Position3D::new(1.0, 1.0, 1.0),
            Position3D::new(2.0, 2.0, 2.0),
            Position3D::new(3.0, 3.0, 3.0),
        ];

        let dots = SIMDSpatialOps::dot_products_fallback(&a_positions, &b_positions);

        assert_eq!(dots.len(), 3);
        assert_eq!(dots[0], 6.0); // 1*1 + 2*1 + 3*1 = 6
        assert_eq!(dots[1], 30.0); // 4*2 + 5*2 + 6*2 = 30
        assert_eq!(dots[2], 72.0); // 7*3 + 8*3 + 9*3 = 72
    }

    #[test]
    fn test_simd_dot_products_automatic() {
        let a_positions = vec![
            Position3D::new(1.0, 2.0, 3.0),
            Position3D::new(4.0, 5.0, 6.0),
            Position3D::new(7.0, 8.0, 9.0),
            Position3D::new(10.0, 11.0, 12.0),
        ];
        let b_positions = vec![
            Position3D::new(1.0, 1.0, 1.0),
            Position3D::new(2.0, 2.0, 2.0),
            Position3D::new(3.0, 3.0, 3.0),
            Position3D::new(4.0, 4.0, 4.0),
        ];

        let dots = SIMDSpatialOps::dot_products(&a_positions, &b_positions);

        assert_eq!(dots.len(), 4);
        assert_eq!(dots[0], 6.0); // 1*1 + 2*1 + 3*1 = 6
        assert_eq!(dots[1], 30.0); // 4*2 + 5*2 + 6*2 = 30
        assert_eq!(dots[2], 72.0); // 7*3 + 8*3 + 9*3 = 72
        assert_eq!(dots[3], 132.0); // 10*4 + 11*4 + 12*4 = 132
    }
}
