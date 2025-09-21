//! Core spatial audio processing functionality

use crate::config::SpatialConfig;
use crate::hrtf::HrtfProcessor;
use crate::memory::{MemoryConfig, MemoryManager};
use crate::position::{Listener, SoundSource};
use crate::room::RoomSimulator;
use crate::types::{BinauraAudio, Position3D, SpatialEffect, SpatialRequest, SpatialResult};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Main spatial audio processor
pub struct SpatialProcessor {
    /// Configuration
    config: SpatialConfig,
    /// HRTF processor
    hrtf_processor: Arc<RwLock<HrtfProcessor>>,
    /// Room simulator
    room_simulator: Arc<RwLock<RoomSimulator>>,
    /// Active sound sources
    sound_sources: Arc<RwLock<HashMap<String, SoundSource>>>,
    /// Listener
    listener: Arc<RwLock<Listener>>,
    /// Processing state
    processing_state: ProcessingState,
    /// Memory manager for optimization
    memory_manager: Arc<MemoryManager>,
}

/// Builder for SpatialProcessor
pub struct SpatialProcessorBuilder {
    config: Option<SpatialConfig>,
    hrtf_database_path: Option<std::path::PathBuf>,
    memory_config: Option<MemoryConfig>,
}

/// Internal processing state
#[derive(Debug)]
struct ProcessingState {
    /// Sample buffers
    #[allow(dead_code)]
    input_buffer: Array1<f32>,
    #[allow(dead_code)]
    output_buffer: Array2<f32>,
    /// Processing statistics
    processed_frames: u64,
    total_processing_time: Duration,
}

/// Distance attenuation model
#[derive(Debug, Clone, Copy)]
pub enum AttenuationModel {
    /// Linear attenuation
    Linear,
    /// Inverse distance law
    InverseDistance,
    /// Inverse square law
    InverseSquare,
    /// Custom exponential model
    Exponential(f32),
}

/// Doppler effect processor
#[allow(dead_code)]
struct DopplerProcessor {
    /// Previous positions for velocity calculation
    previous_positions: HashMap<String, (Position3D, Instant)>,
    /// Speed of sound
    speed_of_sound: f32,
    /// Sample rate
    sample_rate: f32,
}

impl SpatialProcessor {
    /// Create new spatial processor with configuration
    pub async fn new(config: SpatialConfig) -> crate::Result<Self> {
        Self::with_memory_config(config, MemoryConfig::default()).await
    }

    /// Create new spatial processor with spatial and memory configurations
    pub async fn with_memory_config(
        config: SpatialConfig,
        memory_config: MemoryConfig,
    ) -> crate::Result<Self> {
        config.validate()?;

        let hrtf_processor = Arc::new(RwLock::new(
            HrtfProcessor::new(config.hrtf_database_path.clone()).await?,
        ));

        let room_simulator = Arc::new(RwLock::new(RoomSimulator::new(
            config.room_dimensions,
            config.reverb_time,
        )?));

        let processing_state = ProcessingState {
            input_buffer: Array1::zeros(config.buffer_size),
            output_buffer: Array2::zeros((2, config.buffer_size)),
            processed_frames: 0,
            total_processing_time: Duration::ZERO,
        };

        // Initialize memory manager with provided configuration
        let memory_manager = Arc::new(MemoryManager::new(memory_config));

        Ok(Self {
            config,
            hrtf_processor,
            room_simulator,
            sound_sources: Arc::new(RwLock::new(HashMap::new())),
            listener: Arc::new(RwLock::new(Listener::default())),
            processing_state,
            memory_manager,
        })
    }

    /// Process spatial audio request
    pub async fn process_request(
        &mut self,
        request: SpatialRequest,
    ) -> crate::Result<SpatialResult> {
        let start_time = Instant::now();

        // Validate request
        request.validate()?;

        // Store needed values before moving request.audio
        let source_position = request.source_position;
        let effects = request.effects.clone();
        let request_id = request.id.clone();
        let sample_rate = request.sample_rate;

        // Convert input audio to Array1
        let input_audio = Array1::from_vec(request.audio);

        // Initialize output channels
        let mut left_channel = Array1::zeros(input_audio.len());
        let mut right_channel = Array1::zeros(input_audio.len());

        // Get listener information
        let listener = self.listener.read().await;
        let listener_position = listener.position();
        let listener_orientation = listener.orientation();
        drop(listener);

        // Calculate relative position
        let relative_position = self.calculate_relative_position(
            &source_position,
            &listener_position,
            &listener_orientation,
        );

        // Apply requested effects
        for effect in &effects {
            match effect {
                SpatialEffect::Hrtf => {
                    self.apply_hrtf(
                        &input_audio,
                        &mut left_channel,
                        &mut right_channel,
                        &relative_position,
                    )
                    .await?;
                }
                SpatialEffect::DistanceAttenuation => {
                    let distance = source_position.distance_to(&listener_position);
                    let attenuation = self.calculate_distance_attenuation(distance);
                    left_channel *= attenuation;
                    right_channel *= attenuation;
                }
                SpatialEffect::Reverb => {
                    self.apply_reverb(&mut left_channel, &mut right_channel, &source_position)
                        .await?;
                }
                SpatialEffect::Doppler => {
                    self.apply_doppler_effect(
                        &mut left_channel,
                        &mut right_channel,
                        &source_position,
                    )
                    .await?;
                }
                SpatialEffect::AirAbsorption => {
                    let distance = source_position.distance_to(&listener_position);
                    self.apply_air_absorption(&mut left_channel, &mut right_channel, distance);
                }
            }
        }

        // Create result
        let processing_time = start_time.elapsed();
        self.processing_state.processed_frames += 1;
        self.processing_state.total_processing_time += processing_time;

        let binaural_audio =
            BinauraAudio::new(left_channel.to_vec(), right_channel.to_vec(), sample_rate);

        Ok(SpatialResult::success(
            request_id,
            binaural_audio,
            processing_time,
            effects,
        ))
    }

    /// Add sound source
    pub async fn add_sound_source(&self, id: String, source: SoundSource) {
        let mut sources = self.sound_sources.write().await;
        sources.insert(id, source);
    }

    /// Remove sound source
    pub async fn remove_sound_source(&self, id: &str) {
        let mut sources = self.sound_sources.write().await;
        sources.remove(id);
    }

    /// Update listener position and orientation
    pub async fn update_listener(&self, position: Position3D, orientation: (f32, f32, f32)) {
        let mut listener = self.listener.write().await;
        listener.set_position(position);
        listener.set_orientation(orientation);
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            processed_frames: self.processing_state.processed_frames,
            total_processing_time: self.processing_state.total_processing_time,
            average_processing_time: if self.processing_state.processed_frames > 0 {
                self.processing_state.total_processing_time
                    / self.processing_state.processed_frames as u32
            } else {
                Duration::ZERO
            },
        }
    }

    /// Calculate relative position from listener perspective
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
        let (yaw, _pitch, _roll) = *listener_orientation;

        // Simplified rotation (yaw only for now)
        let cos_yaw = yaw.cos();
        let sin_yaw = yaw.sin();

        let rotated_x = relative.x * cos_yaw + relative.z * sin_yaw;
        let rotated_z = -relative.x * sin_yaw + relative.z * cos_yaw;

        relative.x = rotated_x;
        relative.z = rotated_z;

        relative
    }

    /// Apply HRTF processing
    async fn apply_hrtf(
        &self,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        position: &Position3D,
    ) -> crate::Result<()> {
        let hrtf_processor = self.hrtf_processor.read().await;
        hrtf_processor
            .process_position(input, left_output, right_output, position)
            .await
    }

    /// Apply reverb effect
    async fn apply_reverb(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        source_position: &Position3D,
    ) -> crate::Result<()> {
        let room_simulator = self.room_simulator.read().await;
        room_simulator
            .process_reverb(left_channel, right_channel, source_position)
            .await
    }

    /// Apply Doppler effect
    async fn apply_doppler_effect(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        _source_position: &Position3D,
    ) -> crate::Result<()> {
        // Simplified Doppler effect implementation
        // In a real implementation, this would require velocity tracking
        let doppler_factor = 1.0; // Placeholder

        *left_channel *= doppler_factor;
        *right_channel *= doppler_factor;

        Ok(())
    }

    /// Calculate distance attenuation
    fn calculate_distance_attenuation(&self, distance: f32) -> f32 {
        if !self.config.enable_distance_attenuation {
            return 1.0;
        }

        if distance <= 1.0 {
            return 1.0;
        }

        if distance >= self.config.max_distance {
            return 0.0;
        }

        // Inverse distance law with minimum distance
        1.0 / distance.max(1.0)
    }

    /// Apply air absorption
    fn apply_air_absorption(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        distance: f32,
    ) {
        if !self.config.enable_air_absorption {
            return;
        }

        // Simplified air absorption model
        // Higher frequencies are absorbed more
        let absorption_factor = (-0.01 * distance).exp();

        *left_channel *= absorption_factor;
        *right_channel *= absorption_factor;
    }

    /// Get memory manager for advanced memory operations
    pub fn memory_manager(&self) -> &Arc<MemoryManager> {
        &self.memory_manager
    }

    /// Get memory usage statistics
    pub async fn memory_stats(&self) -> crate::memory::MemoryStatistics {
        self.memory_manager.get_memory_stats().await
    }

    /// Check for memory pressure and clean up if needed
    pub async fn optimize_memory(&self) -> bool {
        self.memory_manager.check_memory_pressure().await
    }
}

impl SpatialProcessorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: None,
            hrtf_database_path: None,
            memory_config: None,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: SpatialConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set HRTF database path
    pub fn hrtf_database_path(mut self, path: std::path::PathBuf) -> Self {
        self.hrtf_database_path = Some(path);
        self
    }

    /// Set memory configuration
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Build the spatial processor
    pub async fn build(self) -> crate::Result<SpatialProcessor> {
        let mut config = self.config.unwrap_or_default();

        if let Some(path) = self.hrtf_database_path {
            config.hrtf_database_path = Some(path);
        }

        let memory_config = self.memory_config.unwrap_or_default();
        SpatialProcessor::with_memory_config(config, memory_config).await
    }
}

impl Default for SpatialProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Total processed frames
    pub processed_frames: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per frame
    pub average_processing_time: Duration,
}

impl DopplerProcessor {
    /// Create new Doppler processor
    #[allow(dead_code)]
    fn new(speed_of_sound: f32, sample_rate: f32) -> Self {
        Self {
            previous_positions: HashMap::new(),
            speed_of_sound,
            sample_rate,
        }
    }

    /// Calculate Doppler factor for a moving source
    #[allow(dead_code)]
    fn calculate_doppler_factor(
        &mut self,
        source_id: &str,
        current_position: Position3D,
        listener_position: Position3D,
    ) -> f32 {
        let now = Instant::now();

        if let Some((prev_pos, prev_time)) = self.previous_positions.get(source_id) {
            let time_diff = now.duration_since(*prev_time).as_secs_f32();
            if time_diff > 0.0 {
                // Calculate velocity towards listener
                let prev_distance = prev_pos.distance_to(&listener_position);
                let curr_distance = current_position.distance_to(&listener_position);
                let radial_velocity = (curr_distance - prev_distance) / time_diff;

                // Apply Doppler formula
                let factor = self.speed_of_sound / (self.speed_of_sound + radial_velocity);

                self.previous_positions
                    .insert(source_id.to_string(), (current_position, now));
                return factor.clamp(0.5, 2.0); // Limit extreme values
            }
        }

        self.previous_positions
            .insert(source_id.to_string(), (current_position, now));
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spatial_processor_creation() {
        let config = SpatialConfig::default();
        let result = SpatialProcessor::new(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_spatial_processor_builder() {
        let processor = SpatialProcessorBuilder::new()
            .config(SpatialConfig::default())
            .build()
            .await;
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_distance_attenuation() {
        let config = SpatialConfig::default();
        let processor = SpatialProcessor {
            config: config.clone(),
            hrtf_processor: Arc::new(RwLock::new(HrtfProcessor::new(None).await.unwrap())),
            room_simulator: Arc::new(RwLock::new(
                RoomSimulator::new((10.0, 8.0, 3.0), 1.2).unwrap(),
            )),
            sound_sources: Arc::new(RwLock::new(HashMap::new())),
            listener: Arc::new(RwLock::new(Listener::default())),
            processing_state: ProcessingState {
                input_buffer: Array1::zeros(1024),
                output_buffer: Array2::zeros((2, 1024)),
                processed_frames: 0,
                total_processing_time: Duration::ZERO,
            },
            memory_manager: Arc::new(MemoryManager::new(MemoryConfig::default())),
        };

        // Test distance attenuation
        assert_eq!(processor.calculate_distance_attenuation(1.0), 1.0);
        assert!(processor.calculate_distance_attenuation(2.0) < 1.0);
        assert!(
            processor.calculate_distance_attenuation(10.0)
                < processor.calculate_distance_attenuation(5.0)
        );
    }

    #[tokio::test]
    async fn test_relative_position_calculation() {
        let config = SpatialConfig::default();
        let processor = SpatialProcessor {
            config,
            hrtf_processor: Arc::new(RwLock::new(HrtfProcessor::new(None).await.unwrap())),
            room_simulator: Arc::new(RwLock::new(
                RoomSimulator::new((10.0, 8.0, 3.0), 1.2).unwrap(),
            )),
            sound_sources: Arc::new(RwLock::new(HashMap::new())),
            listener: Arc::new(RwLock::new(Listener::default())),
            processing_state: ProcessingState {
                input_buffer: Array1::zeros(1024),
                output_buffer: Array2::zeros((2, 1024)),
                processed_frames: 0,
                total_processing_time: Duration::ZERO,
            },
            memory_manager: Arc::new(MemoryManager::new(MemoryConfig::default())),
        };

        let source_pos = Position3D::new(5.0, 0.0, 0.0);
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        let listener_orientation = (0.0, 0.0, 0.0);

        let relative_pos = processor.calculate_relative_position(
            &source_pos,
            &listener_pos,
            &listener_orientation,
        );

        assert_eq!(relative_pos.x, 5.0);
        assert_eq!(relative_pos.y, 0.0);
        assert_eq!(relative_pos.z, 0.0);
    }
}
