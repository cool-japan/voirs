//! Utility functions for spatial audio processing

use crate::types::{BinauraAudio, Position3D};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::Complex;
use std::f32::consts::PI;

/// Audio format conversion utilities
pub struct AudioConverter;

/// Mathematical utilities for spatial calculations
pub struct SpatialMath;

/// Signal processing utilities
pub struct SignalProcessor {
    /// Window functions cache
    window_cache: std::collections::HashMap<usize, Array1<f32>>,
}

/// Interpolation utilities
pub struct Interpolator;

/// Coordinate system conversion utilities
pub struct CoordinateConverter;

/// Performance monitoring utilities
pub struct PerformanceMonitor {
    /// Processing times
    processing_times: Vec<std::time::Duration>,
    /// Memory usage tracking
    memory_usage: Vec<usize>,
    /// Frame statistics
    frame_stats: FrameStatistics,
}

/// Frame processing statistics
#[derive(Debug, Clone, Default)]
pub struct FrameStatistics {
    /// Total frames processed
    pub total_frames: u64,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Average processing time
    pub avg_processing_time: std::time::Duration,
    /// Peak processing time
    pub peak_processing_time: std::time::Duration,
    /// CPU usage percentage
    pub cpu_usage: f32,
}

/// Audio quality metrics
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Total harmonic distortion percentage
    pub thd_percent: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Frequency response flatness
    pub frequency_flatness: f32,
    /// Stereo imaging quality
    pub stereo_imaging: f32,
}

impl AudioConverter {
    /// Convert mono to binaural audio
    pub fn mono_to_binaural(mono: &Array1<f32>, sample_rate: u32) -> BinauraAudio {
        BinauraAudio::new(mono.to_vec(), mono.to_vec(), sample_rate)
    }

    /// Convert stereo to binaural audio
    pub fn stereo_to_binaural(
        left: &Array1<f32>,
        right: &Array1<f32>,
        sample_rate: u32,
    ) -> BinauraAudio {
        BinauraAudio::new(left.to_vec(), right.to_vec(), sample_rate)
    }

    /// Mix binaural audio to mono
    pub fn binaural_to_mono(binaural: &BinauraAudio) -> Array1<f32> {
        let len = binaural.left.len().min(binaural.right.len());
        let mut mono = Array1::zeros(len);

        for i in 0..len {
            mono[i] = (binaural.left[i] + binaural.right[i]) * 0.5;
        }

        mono
    }

    /// Resample audio to target sample rate
    pub fn resample(input: &Array1<f32>, input_rate: u32, target_rate: u32) -> Array1<f32> {
        if input_rate == target_rate {
            return input.clone();
        }

        let ratio = target_rate as f32 / input_rate as f32;
        let output_len = (input.len() as f32 * ratio) as usize;
        let mut output = Array1::zeros(output_len);

        // Simple linear interpolation resampling
        for i in 0..output_len {
            let src_index = i as f32 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = src_index.ceil() as usize;

            if src_index_ceil < input.len() {
                let frac = src_index - src_index_floor as f32;
                output[i] = input[src_index_floor] * (1.0 - frac) + input[src_index_ceil] * frac;
            } else if src_index_floor < input.len() {
                output[i] = input[src_index_floor];
            }
        }

        output
    }

    /// Normalize audio to prevent clipping
    pub fn normalize(audio: &mut Array1<f32>, target_level: f32) {
        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        if max_amplitude > 0.0 {
            let scale = target_level / max_amplitude;
            audio.mapv_inplace(|x| x * scale);
        }
    }

    /// Apply fade in/out to audio
    pub fn apply_fade(audio: &mut Array1<f32>, fade_in_samples: usize, fade_out_samples: usize) {
        let len = audio.len();

        // Fade in
        for i in 0..fade_in_samples.min(len) {
            let factor = i as f32 / fade_in_samples as f32;
            audio[i] *= factor;
        }

        // Fade out
        if fade_out_samples > 0 {
            let fade_start = len.saturating_sub(fade_out_samples);
            for i in fade_start..len {
                let factor = (len - i) as f32 / fade_out_samples as f32;
                audio[i] *= factor;
            }
        }
    }
}

impl SpatialMath {
    /// Calculate distance between two 3D points
    pub fn distance_3d(p1: &Position3D, p2: &Position3D) -> f32 {
        p1.distance_to(p2)
    }

    /// Calculate angle between two vectors
    pub fn angle_between_vectors(v1: &Position3D, v2: &Position3D) -> f32 {
        let dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        let magnitude1 = (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z).sqrt();
        let magnitude2 = (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z).sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return 0.0;
        }

        let cos_angle = dot_product / (magnitude1 * magnitude2);
        cos_angle.clamp(-1.0, 1.0).acos()
    }

    /// Normalize a 3D vector
    pub fn normalize_vector(v: &Position3D) -> Position3D {
        let magnitude = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        if magnitude == 0.0 {
            return Position3D::new(0.0, 0.0, 0.0);
        }
        Position3D::new(v.x / magnitude, v.y / magnitude, v.z / magnitude)
    }

    /// Cross product of two 3D vectors
    pub fn cross_product(v1: &Position3D, v2: &Position3D) -> Position3D {
        Position3D::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
    }

    /// Dot product of two 3D vectors
    pub fn dot_product(v1: &Position3D, v2: &Position3D) -> f32 {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    /// Convert Cartesian to spherical coordinates
    pub fn cartesian_to_spherical(pos: &Position3D) -> (f32, f32, f32) {
        let distance = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
        let azimuth = pos.z.atan2(pos.x).to_degrees();
        let elevation = if distance > 0.0 {
            (pos.y / distance).asin().to_degrees()
        } else {
            0.0
        };
        (azimuth, elevation, distance)
    }

    /// Convert spherical to Cartesian coordinates
    pub fn spherical_to_cartesian(azimuth: f32, elevation: f32, distance: f32) -> Position3D {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        Position3D::new(
            distance * el_rad.cos() * az_rad.cos(),
            distance * el_rad.sin(),
            distance * el_rad.cos() * az_rad.sin(),
        )
    }

    /// Calculate reflection vector for a surface
    pub fn reflect_vector(incident: &Position3D, normal: &Position3D) -> Position3D {
        let normalized_normal = Self::normalize_vector(normal);
        let dot = Self::dot_product(incident, &normalized_normal);

        Position3D::new(
            incident.x - 2.0 * dot * normalized_normal.x,
            incident.y - 2.0 * dot * normalized_normal.y,
            incident.z - 2.0 * dot * normalized_normal.z,
        )
    }

    /// Calculate attenuation based on distance
    pub fn distance_attenuation(
        distance: f32,
        reference_distance: f32,
        rolloff_factor: f32,
    ) -> f32 {
        if distance <= reference_distance {
            return 1.0;
        }
        (reference_distance / distance).powf(rolloff_factor)
    }

    /// Calculate Doppler shift factor
    pub fn doppler_factor(
        source_velocity: &Position3D,
        listener_velocity: &Position3D,
        source_to_listener: &Position3D,
        speed_of_sound: f32,
    ) -> f32 {
        let direction = Self::normalize_vector(source_to_listener);
        let source_radial_velocity = Self::dot_product(source_velocity, &direction);
        let listener_radial_velocity = Self::dot_product(listener_velocity, &direction);

        (speed_of_sound + listener_radial_velocity) / (speed_of_sound + source_radial_velocity)
    }
}

impl SignalProcessor {
    /// Create new signal processor
    pub fn new() -> Self {
        Self {
            window_cache: std::collections::HashMap::new(),
        }
    }

    /// Apply window function to signal
    pub fn apply_window(&mut self, signal: &mut Array1<f32>, window_type: WindowType) {
        let len = signal.len();
        let window = self.get_window(len, window_type);

        for i in 0..len {
            signal[i] *= window[i];
        }
    }

    /// Get or create window function
    fn get_window(&mut self, size: usize, window_type: WindowType) -> &Array1<f32> {
        let key = size * 1000 + window_type as usize; // Simple hash

        self.window_cache
            .entry(key)
            .or_insert_with(|| Self::create_window(size, window_type))
    }

    /// Create window function
    fn create_window(size: usize, window_type: WindowType) -> Array1<f32> {
        let mut window = Array1::zeros(size);

        match window_type {
            WindowType::Hann => {
                for i in 0..size {
                    window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos());
                }
            }
            WindowType::Hamming => {
                for i in 0..size {
                    window[i] = 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos();
                }
            }
            WindowType::Blackman => {
                for i in 0..size {
                    let n = i as f32;
                    let n_max = (size - 1) as f32;
                    window[i] = 0.42 - 0.5 * (2.0 * PI * n / n_max).cos()
                        + 0.08 * (4.0 * PI * n / n_max).cos();
                }
            }
            WindowType::Rectangular => {
                window.fill(1.0);
            }
        }

        window
    }

    /// Perform FFT on signal
    pub fn fft(&self, signal: &Array1<f32>) -> Vec<Complex<f32>> {
        // Convert to Complex<f64> for scirs2_fft
        let input: Vec<scirs2_core::Complex<f64>> = signal
            .iter()
            .map(|&x| scirs2_core::Complex::new(x as f64, 0.0))
            .collect();

        // Perform FFT
        let output = scirs2_fft::fft(&input, None)
            .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); input.len()]);

        // Convert back to Complex<f32>
        output
            .into_iter()
            .map(|c| Complex::new(c.re as f32, c.im as f32))
            .collect()
    }

    /// Perform inverse FFT
    pub fn ifft(&self, spectrum: &[Complex<f32>]) -> Array1<f32> {
        // Convert to Complex<f64> for scirs2_fft
        let input: Vec<scirs2_core::Complex<f64>> = spectrum
            .iter()
            .map(|&c| scirs2_core::Complex::new(c.re as f64, c.im as f64))
            .collect();

        // Perform IFFT
        let output = scirs2_fft::ifft(&input, None)
            .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); input.len()]);

        // Extract real part and convert to f32
        Array1::from_vec(output.iter().map(|c| c.re as f32).collect())
    }

    /// Calculate magnitude spectrum
    pub fn magnitude_spectrum(spectrum: &[Complex<f32>]) -> Array1<f32> {
        Array1::from_iter(spectrum.iter().map(|c| c.norm()))
    }

    /// Calculate phase spectrum
    pub fn phase_spectrum(spectrum: &[Complex<f32>]) -> Array1<f32> {
        Array1::from_iter(spectrum.iter().map(|c| c.arg()))
    }

    /// Apply low-pass filter
    pub fn low_pass_filter(
        &self,
        signal: &Array1<f32>,
        cutoff: f32,
        sample_rate: f32,
    ) -> Array1<f32> {
        // Simple first-order low-pass filter
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        let mut output = Array1::zeros(signal.len());
        output[0] = signal[0] * alpha;

        for i in 1..signal.len() {
            output[i] = output[i - 1] + alpha * (signal[i] - output[i - 1]);
        }

        output
    }

    /// Apply high-pass filter
    pub fn high_pass_filter(
        &self,
        signal: &Array1<f32>,
        cutoff: f32,
        sample_rate: f32,
    ) -> Array1<f32> {
        // Simple first-order high-pass filter
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        let mut output = Array1::zeros(signal.len());
        output[0] = signal[0];

        for i in 1..signal.len() {
            output[i] = alpha * (output[i - 1] + signal[i] - signal[i - 1]);
        }

        output
    }
}

impl Default for SignalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular window (no tapering)
    Rectangular = 0,
    /// Hann window (cosine squared tapering)
    Hann = 1,
    /// Hamming window (raised cosine tapering)
    Hamming = 2,
    /// Blackman window (three-term cosine tapering)
    Blackman = 3,
}

impl Interpolator {
    /// Linear interpolation between two values
    pub fn linear(a: f32, b: f32, t: f32) -> f32 {
        a * (1.0 - t) + b * t
    }

    /// Cubic interpolation (Hermite spline)
    pub fn cubic(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;

        let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let c = -0.5 * p0 + 0.5 * p2;
        let d = p1;

        a * t3 + b * t2 + c * t + d
    }

    /// Spherical linear interpolation (SLERP)
    pub fn slerp(start: &Position3D, end: &Position3D, t: f32) -> Position3D {
        let dot = SpatialMath::dot_product(start, end);
        let dot_clamped = dot.clamp(-1.0, 1.0);
        let theta = dot_clamped.acos();

        if theta.sin().abs() < 1e-6 {
            // Vectors are nearly parallel, use linear interpolation
            return Position3D::new(
                Self::linear(start.x, end.x, t),
                Self::linear(start.y, end.y, t),
                Self::linear(start.z, end.z, t),
            );
        }

        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Position3D::new(
            a * start.x + b * end.x,
            a * start.y + b * end.y,
            a * start.z + b * end.z,
        )
    }

    /// Interpolate array of values
    pub fn interpolate_array(values: &[f32], position: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        if position <= 0.0 {
            return values[0];
        }

        if position >= values.len() as f32 - 1.0 {
            return values[values.len() - 1];
        }

        let index = position.floor() as usize;
        let frac = position - index as f32;

        if index + 1 < values.len() {
            Self::linear(values[index], values[index + 1], frac)
        } else {
            values[index]
        }
    }
}

impl CoordinateConverter {
    /// Convert from left-handed to right-handed coordinate system
    pub fn left_to_right_handed(pos: &Position3D) -> Position3D {
        Position3D::new(pos.x, pos.y, -pos.z)
    }

    /// Convert from right-handed to left-handed coordinate system
    pub fn right_to_left_handed(pos: &Position3D) -> Position3D {
        Position3D::new(pos.x, pos.y, -pos.z)
    }

    /// Transform position by rotation matrix
    pub fn rotate_position(pos: &Position3D, rotation_matrix: &Array2<f32>) -> Position3D {
        if rotation_matrix.shape() != [3, 3] {
            return *pos; // Invalid rotation matrix
        }

        Position3D::new(
            rotation_matrix[[0, 0]] * pos.x
                + rotation_matrix[[0, 1]] * pos.y
                + rotation_matrix[[0, 2]] * pos.z,
            rotation_matrix[[1, 0]] * pos.x
                + rotation_matrix[[1, 1]] * pos.y
                + rotation_matrix[[1, 2]] * pos.z,
            rotation_matrix[[2, 0]] * pos.x
                + rotation_matrix[[2, 1]] * pos.y
                + rotation_matrix[[2, 2]] * pos.z,
        )
    }

    /// Create rotation matrix from Euler angles (yaw, pitch, roll)
    pub fn euler_to_rotation_matrix(yaw: f32, pitch: f32, roll: f32) -> Array2<f32> {
        let cy = yaw.cos();
        let sy = yaw.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cr = roll.cos();
        let sr = roll.sin();

        let mut matrix = Array2::zeros((3, 3));
        matrix[[0, 0]] = cy * cp;
        matrix[[0, 1]] = cy * sp * sr - sy * cr;
        matrix[[0, 2]] = cy * sp * cr + sy * sr;
        matrix[[1, 0]] = sy * cp;
        matrix[[1, 1]] = sy * sp * sr + cy * cr;
        matrix[[1, 2]] = sy * sp * cr - cy * sr;
        matrix[[2, 0]] = -sp;
        matrix[[2, 1]] = cp * sr;
        matrix[[2, 2]] = cp * cr;

        matrix
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            processing_times: Vec::new(),
            memory_usage: Vec::new(),
            frame_stats: FrameStatistics::default(),
        }
    }

    /// Record processing time
    pub fn record_processing_time(&mut self, time: std::time::Duration) {
        self.processing_times.push(time);

        // Update statistics
        self.frame_stats.total_frames += 1;

        if time > self.frame_stats.peak_processing_time {
            self.frame_stats.peak_processing_time = time;
        }

        // Calculate moving average
        let recent_times: Vec<_> = self.processing_times.iter().rev().take(100).collect();

        if !recent_times.is_empty() {
            let sum: std::time::Duration = recent_times.iter().map(|&&t| t).sum();
            self.frame_stats.avg_processing_time = sum / recent_times.len() as u32;
        }

        // Limit history size
        if self.processing_times.len() > 1000 {
            self.processing_times.remove(0);
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage.push(bytes);

        if self.memory_usage.len() > 1000 {
            self.memory_usage.remove(0);
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &FrameStatistics {
        &self.frame_stats
    }

    /// Check if performance is within acceptable limits
    pub fn is_performance_acceptable(&self, max_latency: std::time::Duration) -> bool {
        self.frame_stats.avg_processing_time < max_latency
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for audio quality analysis
impl AudioQualityMetrics {
    /// Calculate audio quality metrics
    pub fn analyze(audio: &Array1<f32>, sample_rate: u32) -> Self {
        let snr_db = Self::calculate_snr(audio);
        let thd_percent = Self::calculate_thd(audio, sample_rate);
        let dynamic_range_db = Self::calculate_dynamic_range(audio);

        Self {
            snr_db,
            thd_percent,
            dynamic_range_db,
            frequency_flatness: 0.9, // Placeholder
            stereo_imaging: 0.8,     // Placeholder
        }
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(audio: &Array1<f32>) -> f32 {
        let signal_power: f32 = audio.iter().map(|&x| x * x).sum();
        let noise_power = signal_power * 0.01; // Simplified noise estimation

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0 // Very high SNR
        }
    }

    /// Calculate total harmonic distortion
    fn calculate_thd(_audio: &Array1<f32>, _sample_rate: u32) -> f32 {
        // Simplified THD calculation
        // In practice, this would require FFT analysis of harmonics
        0.1 // 0.1% THD placeholder
    }

    /// Calculate dynamic range
    fn calculate_dynamic_range(audio: &Array1<f32>) -> f32 {
        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        let noise_floor = 0.001; // Simplified noise floor

        if noise_floor > 0.0 {
            20.0 * (max_amplitude / noise_floor).log10()
        } else {
            120.0 // Very high dynamic range
        }
    }
}

/// Benchmark utilities for performance testing
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Benchmark a spatial processing function
    pub fn benchmark_spatial_processing<F>(
        name: &str,
        iterations: usize,
        mut process_fn: F,
    ) -> std::time::Duration
    where
        F: FnMut(),
    {
        let start = std::time::Instant::now();

        for _ in 0..iterations {
            process_fn();
        }

        let duration = start.elapsed();
        println!(
            "Benchmark {}: {:?} total, {:?} per iteration",
            name,
            duration,
            duration / iterations as u32
        );

        duration
    }

    /// Create test signal for benchmarking
    pub fn create_test_signal(length: usize, frequency: f32, sample_rate: f32) -> Array1<f32> {
        Array1::from_iter(
            (0..length).map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin()),
        )
    }

    /// Generate white noise for testing
    pub fn generate_white_noise(length: usize) -> Array1<f32> {
        let mut rng = thread_rng();
        Array1::from_iter((0..length).map(|_| rng.gen::<f32>() * 2.0 - 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_converter() {
        let mono = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let binaural = AudioConverter::mono_to_binaural(&mono, 44100);

        assert_eq!(binaural.left.len(), mono.len());
        assert_eq!(binaural.right.len(), mono.len());
        assert_eq!(binaural.sample_rate, 44100);
    }

    #[test]
    fn test_spatial_math() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);

        let distance = SpatialMath::distance_3d(&p1, &p2);
        assert_eq!(distance, 5.0);

        let (azimuth, elevation, _) = SpatialMath::cartesian_to_spherical(&p2);
        assert!((azimuth - 0.0).abs() < 0.1);
        assert!((elevation - 53.13).abs() < 0.1);
    }

    #[test]
    fn test_interpolation() {
        let result = Interpolator::linear(0.0, 10.0, 0.5);
        assert_eq!(result, 5.0);

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let interpolated = Interpolator::interpolate_array(&values, 1.5);
        assert_eq!(interpolated, 2.5);
    }

    #[test]
    fn test_signal_processor() {
        let mut processor = SignalProcessor::new();
        let mut signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        processor.apply_window(&mut signal, WindowType::Hann);
        assert!(signal.iter().all(|&x| x <= 4.0)); // Should be attenuated
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        monitor.record_processing_time(std::time::Duration::from_millis(10));
        monitor.record_processing_time(std::time::Duration::from_millis(15));

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_frames, 2);
        assert!(stats.avg_processing_time > std::time::Duration::ZERO);
    }

    #[test]
    fn test_coordinate_conversion() {
        let pos = Position3D::new(1.0, 2.0, 3.0);
        let converted = CoordinateConverter::left_to_right_handed(&pos);
        assert_eq!(converted.z, -3.0);

        let rotation_matrix = CoordinateConverter::euler_to_rotation_matrix(0.0, 0.0, 0.0);
        assert_eq!(rotation_matrix.shape(), [3, 3]);
    }

    #[test]
    fn test_audio_quality_metrics() {
        let test_signal = BenchmarkUtils::create_test_signal(1000, 440.0, 44100.0);
        let metrics = AudioQualityMetrics::analyze(&test_signal, 44100);

        assert!(metrics.snr_db > 0.0);
        assert!(metrics.dynamic_range_db > 0.0);
    }
}
