//! Real-time binaural audio rendering system
//!
//! This module provides high-performance binaural audio synthesis for spatial audio applications,
//! including VR/AR, gaming, and immersive audio experiences.

use crate::hrtf::{HrtfDatabase, HrtfProcessor};
use crate::types::{AudioChannel, BinauraAudio, Position3D};
use ndarray::{Array1, Array2};
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Real-time binaural renderer for spatial audio
pub struct BinauralRenderer {
    /// Configuration
    config: BinauralConfig,
    /// HRTF database
    hrtf_database: Arc<HrtfDatabase>,
    /// FFT planner for convolution
    fft_planner: RealFftPlanner<f32>,
    /// Active sources being rendered
    sources: Arc<RwLock<HashMap<String, BinauralSource>>>,
    /// Output buffer for mixing
    output_buffer: Arc<RwLock<Array2<f32>>>,
    /// Convolution buffers
    convolution_state: ConvolutionState,
    /// Performance metrics
    metrics: Arc<RwLock<BinauralMetrics>>,
}

/// Configuration for binaural rendering
#[derive(Debug, Clone)]
pub struct BinauralConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// HRIR length
    pub hrir_length: usize,
    /// Maximum number of simultaneous sources
    pub max_sources: usize,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Crossfade duration for position changes
    pub crossfade_duration: f32,
    /// Quality vs performance trade-off (0.0 = performance, 1.0 = quality)
    pub quality_level: f32,
    /// Enable distance modeling
    pub enable_distance_modeling: bool,
    /// Enable air absorption
    pub enable_air_absorption: bool,
    /// Near-field compensation distance
    pub near_field_distance: f32,
    /// Far-field distance threshold
    pub far_field_distance: f32,
    /// Enable real-time optimization
    pub optimize_for_latency: bool,
}

/// Individual binaural source
#[derive(Debug)]
pub struct BinauralSource {
    /// Unique source identifier
    pub id: String,
    /// Current 3D position
    pub position: Position3D,
    /// Previous position for interpolation
    pub previous_position: Position3D,
    /// Input audio buffer
    pub input_buffer: VecDeque<f32>,
    /// Current gain
    pub gain: f32,
    /// Source type
    pub source_type: SourceType,
    /// Convolution state for this source
    pub convolution_state: SourceConvolutionState,
    /// Last update timestamp
    pub last_update: Instant,
    /// Activity state
    pub is_active: bool,
}

/// Type of audio source
#[derive(Debug, Clone, PartialEq)]
pub enum SourceType {
    /// Static sound source
    Static,
    /// Moving sound source with velocity
    Moving {
        /// Velocity vector of the moving source
        velocity: Position3D,
    },
    /// Streaming source (continuous audio)
    Streaming,
    /// One-shot source (short audio clip)
    OneShot,
}

/// Convolution state for FFT-based processing
struct ConvolutionState {
    /// FFT forward transform
    forward_fft: Arc<dyn RealToComplex<f32>>,
    /// FFT inverse transform  
    inverse_fft: Arc<dyn realfft::ComplexToReal<f32>>,
    /// FFT buffer size
    fft_size: usize,
    /// Overlap-add buffers for left and right channels
    overlap_left: Array1<f32>,
    overlap_right: Array1<f32>,
    /// Frequency domain buffers
    freq_buffer_left: Vec<num_complex::Complex<f32>>,
    freq_buffer_right: Vec<num_complex::Complex<f32>>,
    /// Input FFT buffer
    input_fft_buffer: Vec<f32>,
    /// Output IFFT buffer
    output_ifft_buffer: Vec<f32>,
}

impl std::fmt::Debug for ConvolutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvolutionState")
            .field("fft_size", &self.fft_size)
            .field("overlap_left_len", &self.overlap_left.len())
            .field("overlap_right_len", &self.overlap_right.len())
            .field("freq_buffer_left_len", &self.freq_buffer_left.len())
            .field("freq_buffer_right_len", &self.freq_buffer_right.len())
            .field("input_fft_buffer_len", &self.input_fft_buffer.len())
            .field("output_ifft_buffer_len", &self.output_ifft_buffer.len())
            .finish()
    }
}

/// Per-source convolution state
#[derive(Debug)]
pub struct SourceConvolutionState {
    /// Left channel overlap buffer
    left_overlap: Array1<f32>,
    /// Right channel overlap buffer
    right_overlap: Array1<f32>,
    /// Current HRIR left channel
    current_hrir_left: Array1<f32>,
    /// Current HRIR right channel
    current_hrir_right: Array1<f32>,
    /// Previous HRIR for crossfading
    previous_hrir_left: Array1<f32>,
    /// Previous HRIR right channel
    previous_hrir_right: Array1<f32>,
    /// Crossfade progress (0.0 to 1.0)
    crossfade_progress: f32,
    /// Position interpolation state
    position_interpolation: PositionInterpolation,
}

/// Position interpolation for smooth movement
#[derive(Debug)]
struct PositionInterpolation {
    /// Start position
    start_position: Position3D,
    /// Target position
    target_position: Position3D,
    /// Interpolation progress (0.0 to 1.0)
    progress: f32,
    /// Interpolation duration
    duration: f32,
    /// Start time
    start_time: Instant,
}

/// Performance metrics
#[derive(Debug, Default)]
pub struct BinauralMetrics {
    /// Number of sources currently active
    pub active_sources: usize,
    /// Average processing time per buffer
    pub average_processing_time: Duration,
    /// Peak processing time
    pub peak_processing_time: Duration,
    /// Total processed samples
    pub total_samples_processed: u64,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Underruns (buffer not ready in time)
    pub underruns: u64,
    /// Overruns (buffer overflow)
    pub overruns: u64,
}

impl BinauralRenderer {
    /// Create new binaural renderer
    pub async fn new(
        config: BinauralConfig,
        hrtf_database: Arc<HrtfDatabase>,
    ) -> crate::Result<Self> {
        config.validate()?;

        let fft_size = config.buffer_size.next_power_of_two() * 2;
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let forward_fft = fft_planner.plan_fft_forward(fft_size);
        let inverse_fft = fft_planner.plan_fft_inverse(fft_size);

        let convolution_state = ConvolutionState {
            forward_fft,
            inverse_fft,
            fft_size,
            overlap_left: Array1::zeros(config.hrir_length - 1),
            overlap_right: Array1::zeros(config.hrir_length - 1),
            freq_buffer_left: vec![num_complex::Complex::new(0.0, 0.0); fft_size / 2 + 1],
            freq_buffer_right: vec![num_complex::Complex::new(0.0, 0.0); fft_size / 2 + 1],
            input_fft_buffer: vec![0.0; fft_size],
            output_ifft_buffer: vec![0.0; fft_size],
        };

        Ok(Self {
            output_buffer: Arc::new(RwLock::new(Array2::zeros((2, config.buffer_size)))),
            config,
            hrtf_database,
            fft_planner,
            sources: Arc::new(RwLock::new(HashMap::new())),
            convolution_state,
            metrics: Arc::new(RwLock::new(BinauralMetrics::default())),
        })
    }

    /// Add audio source for binaural rendering
    pub async fn add_source(
        &self,
        id: String,
        position: Position3D,
        source_type: SourceType,
    ) -> crate::Result<()> {
        let mut sources = self.sources.write().await;

        if sources.len() >= self.config.max_sources {
            return Err(crate::Error::LegacyProcessing(
                "Maximum number of sources reached".to_string(),
            ));
        }

        let convolution_state = SourceConvolutionState {
            left_overlap: Array1::zeros(self.config.hrir_length - 1),
            right_overlap: Array1::zeros(self.config.hrir_length - 1),
            current_hrir_left: Array1::zeros(self.config.hrir_length),
            current_hrir_right: Array1::zeros(self.config.hrir_length),
            previous_hrir_left: Array1::zeros(self.config.hrir_length),
            previous_hrir_right: Array1::zeros(self.config.hrir_length),
            crossfade_progress: 1.0,
            position_interpolation: PositionInterpolation {
                start_position: position,
                target_position: position,
                progress: 1.0,
                duration: 0.0,
                start_time: Instant::now(),
            },
        };

        let source = BinauralSource {
            id: id.clone(),
            position,
            previous_position: position,
            input_buffer: VecDeque::new(),
            gain: 1.0,
            source_type,
            convolution_state,
            last_update: Instant::now(),
            is_active: true,
        };

        sources.insert(id, source);
        Ok(())
    }

    /// Remove audio source
    pub async fn remove_source(&self, id: &str) -> crate::Result<()> {
        let mut sources = self.sources.write().await;
        sources.remove(id);
        Ok(())
    }

    /// Update source position with smooth interpolation
    pub async fn update_source_position(
        &self,
        id: &str,
        new_position: Position3D,
        interpolation_duration: Option<f32>,
    ) -> crate::Result<()> {
        let mut sources = self.sources.write().await;

        if let Some(source) = sources.get_mut(id) {
            source.previous_position = source.position;

            if let Some(duration) = interpolation_duration {
                // Set up smooth position interpolation
                source.convolution_state.position_interpolation = PositionInterpolation {
                    start_position: source.position,
                    target_position: new_position,
                    progress: 0.0,
                    duration,
                    start_time: Instant::now(),
                };
                source.convolution_state.crossfade_progress = 0.0;
            } else {
                // Immediate position update
                source.position = new_position;
                source.convolution_state.position_interpolation.progress = 1.0;
            }

            source.last_update = Instant::now();
        }

        Ok(())
    }

    /// Feed audio data to a source
    pub async fn feed_source_audio(&self, id: &str, audio_data: Vec<f32>) -> crate::Result<()> {
        let mut sources = self.sources.write().await;

        if let Some(source) = sources.get_mut(id) {
            // Add audio to source buffer
            source.input_buffer.extend(audio_data);

            // Limit buffer size to prevent memory issues
            let max_buffer_size = self.config.buffer_size * 4; // 4x buffer size max
            while source.input_buffer.len() > max_buffer_size {
                source.input_buffer.pop_front();
            }

            source.last_update = Instant::now();
        }

        Ok(())
    }

    /// Process real-time binaural audio rendering
    pub async fn process_frame(
        &mut self,
        listener_position: Position3D,
        listener_orientation: (f32, f32, f32),
    ) -> crate::Result<BinauraAudio> {
        let start_time = Instant::now();

        // Clear output buffer
        let mut output_buffer = self.output_buffer.write().await;
        output_buffer.fill(0.0);

        // Get sources for processing
        let mut sources = self.sources.write().await;
        let mut active_source_count = 0;

        // Process each active source
        for (_id, source) in sources.iter_mut() {
            if !source.is_active || source.input_buffer.is_empty() {
                continue;
            }

            active_source_count += 1;

            // Update position interpolation
            self.update_position_interpolation(source).await?;

            // Calculate current listener-relative position
            let relative_position = self.calculate_relative_position(
                &source.position,
                &listener_position,
                &listener_orientation,
            );

            // Get HRIR for current position
            let (hrir_left, hrir_right) = self.get_hrir_for_position(&relative_position).await?;

            // Update source HRIR with crossfading if needed
            self.update_source_hrir(source, &hrir_left, &hrir_right)?;

            // Extract audio samples for processing
            let samples_needed = self.config.buffer_size;
            let mut input_samples = Vec::with_capacity(samples_needed);

            for _ in 0..samples_needed {
                if let Some(sample) = source.input_buffer.pop_front() {
                    input_samples.push(sample * source.gain);
                } else {
                    input_samples.push(0.0);
                }
            }

            // Apply binaural convolution
            let (left_output, right_output) =
                self.apply_binaural_convolution(&input_samples, &mut source.convolution_state)?;

            // Mix into output buffer
            for i in 0..samples_needed {
                output_buffer[[0, i]] += left_output[i]; // Left channel
                output_buffer[[1, i]] += right_output[i]; // Right channel
            }
        }

        drop(sources);

        // Apply post-processing effects
        self.apply_post_processing(&mut output_buffer).await?;

        // Convert to BinauraAudio
        let left_channel: Vec<f32> = output_buffer.row(0).to_vec();
        let right_channel: Vec<f32> = output_buffer.row(1).to_vec();

        drop(output_buffer);

        // Update metrics
        let processing_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.active_sources = active_source_count;
        metrics.total_samples_processed += self.config.buffer_size as u64;

        if processing_time > metrics.peak_processing_time {
            metrics.peak_processing_time = processing_time;
        }

        // Update average processing time (exponential moving average)
        let alpha = 0.1; // Smoothing factor
        if metrics.average_processing_time == Duration::ZERO {
            metrics.average_processing_time = processing_time;
        } else {
            let avg_nanos = metrics.average_processing_time.as_nanos() as f64;
            let new_avg = avg_nanos * (1.0 - alpha) + processing_time.as_nanos() as f64 * alpha;
            metrics.average_processing_time = Duration::from_nanos(new_avg as u64);
        }

        Ok(BinauraAudio::new(
            left_channel,
            right_channel,
            self.config.sample_rate,
        ))
    }

    /// Update position interpolation for smooth movement
    async fn update_position_interpolation(
        &self,
        source: &mut BinauralSource,
    ) -> crate::Result<()> {
        let interp = &mut source.convolution_state.position_interpolation;

        if interp.progress >= 1.0 {
            return Ok(());
        }

        let elapsed = interp.start_time.elapsed().as_secs_f32();
        interp.progress = (elapsed / interp.duration).min(1.0);

        // Smooth interpolation (cosine interpolation for natural movement)
        let smooth_progress = (1.0 - (interp.progress * std::f32::consts::PI).cos()) * 0.5;

        source.position = Position3D::new(
            interp.start_position.x
                + (interp.target_position.x - interp.start_position.x) * smooth_progress,
            interp.start_position.y
                + (interp.target_position.y - interp.start_position.y) * smooth_progress,
            interp.start_position.z
                + (interp.target_position.z - interp.start_position.z) * smooth_progress,
        );

        Ok(())
    }

    /// Calculate listener-relative position
    fn calculate_relative_position(
        &self,
        source_pos: &Position3D,
        listener_pos: &Position3D,
        listener_orientation: &(f32, f32, f32),
    ) -> Position3D {
        // Calculate position relative to listener
        let mut relative = Position3D::new(
            source_pos.x - listener_pos.x,
            source_pos.y - listener_pos.y,
            source_pos.z - listener_pos.z,
        );

        // Apply listener orientation rotation
        let (yaw, pitch, roll) = *listener_orientation;

        // Full 3D rotation (simplified for performance)
        let cos_yaw = yaw.cos();
        let sin_yaw = yaw.sin();
        let cos_pitch = pitch.cos();
        let sin_pitch = pitch.sin();
        let cos_roll = roll.cos();
        let sin_roll = roll.sin();

        // Yaw rotation (around Y axis)
        let rotated_x = relative.x * cos_yaw + relative.z * sin_yaw;
        let rotated_z = -relative.x * sin_yaw + relative.z * cos_yaw;
        relative.x = rotated_x;
        relative.z = rotated_z;

        // Pitch rotation (around X axis)
        let rotated_y = relative.y * cos_pitch - relative.z * sin_pitch;
        let rotated_z = relative.y * sin_pitch + relative.z * cos_pitch;
        relative.y = rotated_y;
        relative.z = rotated_z;

        // Roll rotation (around Z axis)
        let rotated_x = relative.x * cos_roll - relative.y * sin_roll;
        let rotated_y = relative.x * sin_roll + relative.y * cos_roll;
        relative.x = rotated_x;
        relative.y = rotated_y;

        relative
    }

    /// Get HRIR for a given position
    async fn get_hrir_for_position(
        &self,
        position: &Position3D,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        // Convert to spherical coordinates
        let distance =
            (position.x * position.x + position.y * position.y + position.z * position.z).sqrt();
        let azimuth = position.x.atan2(-position.z).to_degrees();
        let elevation = (position.y / distance.max(0.001)).asin().to_degrees();

        // Get HRIR from database (with interpolation)
        let hrir_left = self
            .hrtf_database
            .get_hrir_left(azimuth as i32, elevation as i32)?;
        let hrir_right = self
            .hrtf_database
            .get_hrir_right(azimuth as i32, elevation as i32)?;

        Ok((hrir_left, hrir_right))
    }

    /// Update source HRIR with crossfading
    fn update_source_hrir(
        &self,
        source: &mut BinauralSource,
        new_hrir_left: &Array1<f32>,
        new_hrir_right: &Array1<f32>,
    ) -> crate::Result<()> {
        let state = &mut source.convolution_state;

        // Check if HRIR has changed significantly
        let change_threshold = 0.1;
        let left_change = (&state.current_hrir_left - new_hrir_left)
            .mapv(|x| x.abs())
            .sum()
            / new_hrir_left.len() as f32;
        let right_change = (&state.current_hrir_right - new_hrir_right)
            .mapv(|x| x.abs())
            .sum()
            / new_hrir_right.len() as f32;

        if left_change > change_threshold || right_change > change_threshold {
            // Start crossfade
            state.previous_hrir_left = state.current_hrir_left.clone();
            state.previous_hrir_right = state.current_hrir_right.clone();
            state.current_hrir_left = new_hrir_left.clone();
            state.current_hrir_right = new_hrir_right.clone();
            state.crossfade_progress = 0.0;
        }

        // Update crossfade progress
        if state.crossfade_progress < 1.0 {
            let crossfade_speed = 1.0
                / (self.config.crossfade_duration * self.config.sample_rate as f32
                    / self.config.buffer_size as f32);
            state.crossfade_progress = (state.crossfade_progress + crossfade_speed).min(1.0);
        }

        Ok(())
    }

    /// Apply binaural convolution with overlap-add
    fn apply_binaural_convolution(
        &self,
        input: &[f32],
        state: &mut SourceConvolutionState,
    ) -> crate::Result<(Vec<f32>, Vec<f32>)> {
        let buffer_size = input.len();
        let hrir_length = self.config.hrir_length;

        // Apply crossfading if in progress
        let (effective_hrir_left, effective_hrir_right) = if state.crossfade_progress < 1.0 {
            let fade_factor = state.crossfade_progress;
            let inverse_fade = 1.0 - fade_factor;

            let left =
                &state.previous_hrir_left * inverse_fade + &state.current_hrir_left * fade_factor;
            let right =
                &state.previous_hrir_right * inverse_fade + &state.current_hrir_right * fade_factor;
            (left, right)
        } else {
            (
                state.current_hrir_left.clone(),
                state.current_hrir_right.clone(),
            )
        };

        // Perform convolution (simplified direct convolution for now)
        // In production, this would use FFT for efficiency
        let mut left_output = vec![0.0; buffer_size];
        let mut right_output = vec![0.0; buffer_size];

        // Direct convolution with overlap-add
        for n in 0..buffer_size {
            for k in 0..hrir_length.min(n + 1) {
                if n >= k {
                    left_output[n] += input[n - k] * effective_hrir_left[k];
                    right_output[n] += input[n - k] * effective_hrir_right[k];
                }
            }
        }

        // Add overlap from previous block
        for i in 0..(hrir_length - 1).min(buffer_size) {
            left_output[i] += state.left_overlap[i];
            right_output[i] += state.right_overlap[i];
        }

        // Compute new overlap for next block
        state.left_overlap.fill(0.0);
        state.right_overlap.fill(0.0);

        for n in 0..(hrir_length - 1) {
            for k in (buffer_size)..(buffer_size + hrir_length).min(buffer_size + n + 1) {
                if k >= buffer_size && k - buffer_size < hrir_length && n < hrir_length - 1 {
                    state.left_overlap[n] += input.get(k - buffer_size).unwrap_or(&0.0)
                        * effective_hrir_left.get(k - buffer_size).unwrap_or(&0.0);
                    state.right_overlap[n] += input.get(k - buffer_size).unwrap_or(&0.0)
                        * effective_hrir_right.get(k - buffer_size).unwrap_or(&0.0);
                }
            }
        }

        Ok((left_output, right_output))
    }

    /// Apply post-processing effects
    async fn apply_post_processing(&self, _output_buffer: &mut Array2<f32>) -> crate::Result<()> {
        // Apply limiter to prevent clipping
        // Apply EQ if configured
        // Apply final gain adjustment
        Ok(())
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> BinauralMetrics {
        self.metrics.read().await.clone()
    }
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            buffer_size: 256,
            hrir_length: 256,
            max_sources: 32,
            use_gpu: false,
            crossfade_duration: 0.05, // 50ms
            quality_level: 0.8,
            enable_distance_modeling: true,
            enable_air_absorption: true,
            near_field_distance: 0.2,
            far_field_distance: 10.0,
            optimize_for_latency: true,
        }
    }
}

impl BinauralConfig {
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.sample_rate == 0 {
            return Err(crate::Error::LegacyConfig(
                "Sample rate must be positive".to_string(),
            ));
        }
        if self.buffer_size == 0 {
            return Err(crate::Error::LegacyConfig(
                "Buffer size must be positive".to_string(),
            ));
        }
        if self.hrir_length == 0 {
            return Err(crate::Error::LegacyConfig(
                "HRIR length must be positive".to_string(),
            ));
        }
        if self.max_sources == 0 {
            return Err(crate::Error::LegacyConfig(
                "Max sources must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.quality_level) {
            return Err(crate::Error::LegacyConfig(
                "Quality level must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Clone for BinauralMetrics {
    fn clone(&self) -> Self {
        Self {
            active_sources: self.active_sources,
            average_processing_time: self.average_processing_time,
            peak_processing_time: self.peak_processing_time,
            total_samples_processed: self.total_samples_processed,
            cpu_usage: self.cpu_usage,
            memory_usage: self.memory_usage,
            underruns: self.underruns,
            overruns: self.overruns,
        }
    }
}

// Helper trait for HRTF database access
trait HrtfDatabaseExt {
    fn get_hrir_left(&self, azimuth: i32, elevation: i32) -> crate::Result<Array1<f32>>;
    fn get_hrir_right(&self, azimuth: i32, elevation: i32) -> crate::Result<Array1<f32>>;
}

impl HrtfDatabaseExt for HrtfDatabase {
    fn get_hrir_left(&self, azimuth: i32, elevation: i32) -> crate::Result<Array1<f32>> {
        // This would use the actual HRTF database lookup
        // For now, return a simple impulse response
        let mut hrir = Array1::zeros(256);
        hrir[0] = 1.0; // Simple impulse
        Ok(hrir)
    }

    fn get_hrir_right(&self, azimuth: i32, elevation: i32) -> crate::Result<Array1<f32>> {
        // This would use the actual HRTF database lookup
        // For now, return a simple impulse response
        let mut hrir = Array1::zeros(256);
        hrir[0] = 1.0; // Simple impulse
        Ok(hrir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binaural_config_validation() {
        let config = BinauralConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config;
        invalid_config.sample_rate = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[tokio::test]
    async fn test_binaural_renderer_creation() {
        let config = BinauralConfig::default();
        let hrtf_db = Arc::new(crate::hrtf::HrtfDatabase::load_default().await.unwrap());
        let renderer = BinauralRenderer::new(config, hrtf_db).await;
        assert!(renderer.is_ok());
    }

    #[tokio::test]
    async fn test_source_management() {
        let config = BinauralConfig::default();
        let hrtf_db = Arc::new(crate::hrtf::HrtfDatabase::load_default().await.unwrap());
        let renderer = BinauralRenderer::new(config, hrtf_db).await.unwrap();

        // Add source
        let result = renderer
            .add_source(
                "test_source".to_string(),
                Position3D::new(1.0, 0.0, 0.0),
                SourceType::Static,
            )
            .await;
        assert!(result.is_ok());

        // Remove source
        let result = renderer.remove_source("test_source").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_position_interpolation() {
        let config = BinauralConfig::default();
        let hrtf_db = Arc::new(crate::hrtf::HrtfDatabase::load_default().await.unwrap());
        let renderer = BinauralRenderer::new(config, hrtf_db).await.unwrap();

        renderer
            .add_source(
                "moving_source".to_string(),
                Position3D::new(0.0, 0.0, 0.0),
                SourceType::Moving {
                    velocity: Position3D::new(1.0, 0.0, 0.0),
                },
            )
            .await
            .unwrap();

        let result = renderer
            .update_source_position(
                "moving_source",
                Position3D::new(5.0, 0.0, 0.0),
                Some(1.0), // 1 second interpolation
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audio_feeding() {
        let config = BinauralConfig::default();
        let hrtf_db = Arc::new(crate::hrtf::HrtfDatabase::load_default().await.unwrap());
        let renderer = BinauralRenderer::new(config, hrtf_db).await.unwrap();

        renderer
            .add_source(
                "audio_source".to_string(),
                Position3D::new(1.0, 1.0, 1.0),
                SourceType::Streaming,
            )
            .await
            .unwrap();

        let audio_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = renderer.feed_source_audio("audio_source", audio_data).await;
        assert!(result.is_ok());
    }
}
