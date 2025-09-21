//! 3D Spatial Audio integration for VoiRS SDK

use crate::{Result, VoirsError};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 3D position in space
#[derive(Debug, Clone, PartialEq)]
pub struct Position3D {
    /// X coordinate (left-right)
    pub x: f32,
    /// Y coordinate (up-down)
    pub y: f32,
    /// Z coordinate (front-back)
    pub z: f32,
}

/// 3D orientation (rotation)
#[derive(Debug, Clone, PartialEq)]
pub struct Orientation3D {
    /// Yaw (rotation around Y-axis) in radians
    pub yaw: f32,
    /// Pitch (rotation around X-axis) in radians
    pub pitch: f32,
    /// Roll (rotation around Z-axis) in radians
    pub roll: f32,
}

/// 3D velocity for Doppler effect
#[derive(Debug, Clone, PartialEq)]
pub struct Velocity3D {
    /// X velocity component
    pub x: f32,
    /// Y velocity component
    pub y: f32,
    /// Z velocity component
    pub z: f32,
}

/// Audio source in 3D space
#[derive(Debug, Clone)]
pub struct AudioSource3D {
    /// Source position
    pub position: Position3D,
    /// Source orientation
    pub orientation: Orientation3D,
    /// Source velocity (for Doppler effect)
    pub velocity: Velocity3D,
    /// Source volume (0.0-1.0)
    pub volume: f32,
    /// Source directivity pattern
    pub directivity: DirectivityPattern,
}

/// Audio listener in 3D space
#[derive(Debug, Clone)]
pub struct AudioListener3D {
    /// Listener position
    pub position: Position3D,
    /// Listener orientation
    pub orientation: Orientation3D,
    /// Listener velocity (for Doppler effect)
    pub velocity: Velocity3D,
}

/// Directivity pattern for audio sources
#[derive(Debug, Clone, PartialEq)]
pub enum DirectivityPattern {
    /// Omnidirectional (equal in all directions)
    Omnidirectional,
    /// Cardioid (heart-shaped, directional)
    Cardioid,
    /// Bidirectional (figure-8 pattern)
    Bidirectional,
    /// Custom pattern with angular coefficients
    Custom(Vec<f32>),
}

/// Room acoustics parameters
#[derive(Debug, Clone)]
pub struct RoomAcoustics {
    /// Room dimensions
    pub dimensions: Position3D,
    /// Absorption coefficients for different frequencies
    pub absorption: FrequencyResponse,
    /// Reverberation time (RT60) in seconds
    pub reverb_time: f32,
    /// Early reflections delay in milliseconds
    pub early_reflections_delay: f32,
    /// Diffusion coefficient (0.0-1.0)
    pub diffusion: f32,
}

/// Frequency response curve
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Frequency bands in Hz
    pub frequencies: Vec<f32>,
    /// Corresponding response values
    pub responses: Vec<f32>,
}

/// HRTF (Head-Related Transfer Function) configuration
#[derive(Debug, Clone)]
pub struct HrtfConfig {
    /// HRTF dataset to use
    pub dataset: HrtfDataset,
    /// Head circumference in cm (for personalization)
    pub head_circumference: f32,
    /// Interpupillary distance in cm
    pub interpupillary_distance: f32,
    /// Enable crossfeed for better stereo imaging
    pub crossfeed_enabled: bool,
    /// Crossfeed strength (0.0-1.0)
    pub crossfeed_strength: f32,
}

/// HRTF dataset options
#[derive(Debug, Clone, PartialEq)]
pub enum HrtfDataset {
    /// Generic HRTF dataset
    Generic,
    /// MIT KEMAR dataset
    Kemar,
    /// CIPIC database
    Cipic,
    /// Custom dataset
    Custom(String),
}

/// Binaural rendering configuration
#[derive(Debug, Clone)]
pub struct BinauralConfig {
    /// HRTF configuration
    pub hrtf: HrtfConfig,
    /// Enable dynamic range compression
    pub compression_enabled: bool,
    /// Compression ratio (1.0 = no compression)
    pub compression_ratio: f32,
    /// Output sample rate
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
}

/// 3D spatial audio configuration
#[derive(Debug, Clone)]
pub struct SpatialAudioConfig {
    /// Enable spatial audio processing
    pub enabled: bool,
    /// Binaural rendering configuration
    pub binaural: BinauralConfig,
    /// Room acoustics simulation
    pub room_acoustics: Option<RoomAcoustics>,
    /// Speed of sound in m/s
    pub speed_of_sound: f32,
    /// Doppler effect strength (0.0-1.0)
    pub doppler_strength: f32,
    /// Distance attenuation model
    pub distance_model: DistanceModel,
    /// Maximum processing distance
    pub max_distance: f32,
}

/// Distance attenuation models
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceModel {
    /// No attenuation
    None,
    /// Linear attenuation
    Linear,
    /// Inverse distance law
    Inverse,
    /// Inverse square law
    InverseSquare,
}

/// Spatial audio processing result
#[derive(Debug, Clone)]
pub struct SpatialAudioResult {
    /// Processed binaural audio
    pub audio: crate::audio::AudioBuffer,
    /// Applied spatial configuration
    pub config: SpatialAudioConfig,
    /// Processing statistics
    pub stats: SpatialAudioStats,
}

/// Processing statistics for spatial audio
#[derive(Debug, Clone)]
pub struct SpatialAudioStats {
    /// Number of active sources
    pub active_sources: usize,
    /// Processing latency in ms
    pub latency_ms: f32,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// HRTF interpolation quality
    pub hrtf_quality: f32,
}

/// SDK-integrated spatial audio controller
#[derive(Debug, Clone)]
pub struct SpatialAudioController {
    /// Internal spatial audio processor configuration
    config: Arc<RwLock<SpatialAudioConfig>>,
    /// Active audio sources
    sources: Arc<RwLock<Vec<AudioSource3D>>>,
    /// Audio listener
    listener: Arc<RwLock<AudioListener3D>>,
}

impl SpatialAudioController {
    /// Create new spatial audio controller
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(SpatialAudioConfig::default())),
            sources: Arc::new(RwLock::new(Vec::new())),
            listener: Arc::new(RwLock::new(AudioListener3D::default())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: SpatialAudioConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            sources: Arc::new(RwLock::new(Vec::new())),
            listener: Arc::new(RwLock::new(AudioListener3D::default())),
        })
    }

    /// Set listener position and orientation
    pub async fn set_listener(
        &self,
        position: Position3D,
        orientation: Orientation3D,
    ) -> Result<()> {
        let mut listener = self.listener.write().await;
        listener.position = position;
        listener.orientation = orientation;
        Ok(())
    }

    /// Add audio source at 3D position
    pub async fn add_source(&self, source: AudioSource3D) -> Result<usize> {
        let mut sources = self.sources.write().await;
        sources.push(source);
        Ok(sources.len() - 1) // Return source ID
    }

    /// Update source position
    pub async fn update_source_position(
        &self,
        source_id: usize,
        position: Position3D,
    ) -> Result<()> {
        let mut sources = self.sources.write().await;
        if let Some(source) = sources.get_mut(source_id) {
            source.position = position;
            Ok(())
        } else {
            Err(VoirsError::ConfigError {
                field: "source_id".to_string(),
                message: format!("Source ID {} not found", source_id),
            })
        }
    }

    /// Remove audio source
    pub async fn remove_source(&self, source_id: usize) -> Result<()> {
        let mut sources = self.sources.write().await;
        if source_id < sources.len() {
            sources.remove(source_id);
            Ok(())
        } else {
            Err(VoirsError::ConfigError {
                field: "source_id".to_string(),
                message: format!("Source ID {} not found", source_id),
            })
        }
    }

    /// Process audio with spatial effects
    pub async fn process_spatial_audio(
        &self,
        audio: &crate::audio::AudioBuffer,
    ) -> Result<SpatialAudioResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "spatial".to_string(),
                message: "Spatial audio processing is disabled".to_string(),
            });
        }

        let sources = self.sources.read().await;
        let listener = self.listener.read().await;

        // Mock implementation - in reality would use advanced 3D audio processing
        let processed_audio = self
            .render_binaural_audio(audio, &sources, &listener, &config)
            .await?;

        Ok(SpatialAudioResult {
            audio: processed_audio,
            config: config.clone(),
            stats: SpatialAudioStats {
                active_sources: sources.len(),
                latency_ms: 12.0,
                cpu_usage: 15.0,
                hrtf_quality: 0.92,
            },
        })
    }

    /// Set room acoustics parameters
    pub async fn set_room_acoustics(&self, room: RoomAcoustics) -> Result<()> {
        let mut config = self.config.write().await;
        config.room_acoustics = Some(room);
        Ok(())
    }

    /// Set HRTF configuration
    pub async fn set_hrtf_config(&self, hrtf: HrtfConfig) -> Result<()> {
        let mut config = self.config.write().await;
        config.binaural.hrtf = hrtf;
        Ok(())
    }

    /// Enable or disable spatial audio
    pub async fn set_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if spatial audio is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// Get current configuration
    pub async fn get_config(&self) -> SpatialAudioConfig {
        self.config.read().await.clone()
    }

    /// Get active sources
    pub async fn get_sources(&self) -> Vec<AudioSource3D> {
        self.sources.read().await.clone()
    }

    /// Get listener configuration
    pub async fn get_listener(&self) -> AudioListener3D {
        self.listener.read().await.clone()
    }

    /// Apply spatial audio preset
    pub async fn apply_preset(&self, preset_name: &str) -> Result<()> {
        let config = match preset_name {
            "gaming" => SpatialAudioConfig {
                enabled: true,
                binaural: BinauralConfig {
                    hrtf: HrtfConfig {
                        dataset: HrtfDataset::Generic,
                        head_circumference: 56.0,
                        interpupillary_distance: 6.4,
                        crossfeed_enabled: false,
                        crossfeed_strength: 0.0,
                    },
                    compression_enabled: true,
                    compression_ratio: 3.0,
                    sample_rate: 44100,
                    buffer_size: 512,
                },
                room_acoustics: None,
                speed_of_sound: 343.0,
                doppler_strength: 0.8,
                distance_model: DistanceModel::InverseSquare,
                max_distance: 100.0,
            },
            "cinema" => SpatialAudioConfig {
                enabled: true,
                binaural: BinauralConfig {
                    hrtf: HrtfConfig {
                        dataset: HrtfDataset::Kemar,
                        head_circumference: 57.0,
                        interpupillary_distance: 6.5,
                        crossfeed_enabled: true,
                        crossfeed_strength: 0.3,
                    },
                    compression_enabled: false,
                    compression_ratio: 1.0,
                    sample_rate: 48000,
                    buffer_size: 1024,
                },
                room_acoustics: Some(RoomAcoustics::default()),
                speed_of_sound: 343.0,
                doppler_strength: 0.6,
                distance_model: DistanceModel::Inverse,
                max_distance: 50.0,
            },
            "vr" => SpatialAudioConfig {
                enabled: true,
                binaural: BinauralConfig {
                    hrtf: HrtfConfig {
                        dataset: HrtfDataset::Cipic,
                        head_circumference: 55.0,
                        interpupillary_distance: 6.2,
                        crossfeed_enabled: false,
                        crossfeed_strength: 0.0,
                    },
                    compression_enabled: true,
                    compression_ratio: 2.0,
                    sample_rate: 48000,
                    buffer_size: 256,
                },
                room_acoustics: Some(RoomAcoustics::default()),
                speed_of_sound: 343.0,
                doppler_strength: 1.0,
                distance_model: DistanceModel::InverseSquare,
                max_distance: 200.0,
            },
            _ => {
                return Err(VoirsError::ConfigError {
                    field: "preset".to_string(),
                    message: format!("Unknown spatial audio preset: {}", preset_name),
                })
            }
        };

        {
            let mut config_guard = self.config.write().await;
            *config_guard = config;
        }

        Ok(())
    }

    /// List available presets
    pub fn list_presets(&self) -> Vec<String> {
        vec!["gaming".to_string(), "cinema".to_string(), "vr".to_string()]
    }

    // Private helper methods

    /// Render binaural audio from sources
    async fn render_binaural_audio(
        &self,
        audio: &crate::audio::AudioBuffer,
        sources: &[AudioSource3D],
        listener: &AudioListener3D,
        config: &SpatialAudioConfig,
    ) -> Result<crate::audio::AudioBuffer> {
        // Mock implementation - in reality would use HRTF convolution
        let mut processed_samples = audio.samples().to_vec();

        // Apply distance attenuation and directivity
        for source in sources {
            let distance = self.calculate_distance(&source.position, &listener.position);
            let attenuation = self.calculate_attenuation(distance, &config.distance_model);

            // Apply attenuation to all samples
            for sample in &mut processed_samples {
                *sample *= attenuation * source.volume;
            }
        }

        // Apply HRTF processing (mock)
        self.apply_hrtf_processing(&mut processed_samples, &config.binaural.hrtf)
            .await?;

        Ok(crate::audio::AudioBuffer::mono(
            processed_samples,
            audio.sample_rate(),
        ))
    }

    /// Calculate distance between two points
    fn calculate_distance(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate attenuation based on distance model
    fn calculate_attenuation(&self, distance: f32, model: &DistanceModel) -> f32 {
        match model {
            DistanceModel::None => 1.0,
            DistanceModel::Linear => (1.0 - distance / 100.0).max(0.0),
            DistanceModel::Inverse => 1.0 / (1.0 + distance),
            DistanceModel::InverseSquare => 1.0 / (1.0 + distance * distance),
        }
    }

    /// Apply HRTF processing (mock implementation)
    async fn apply_hrtf_processing(&self, samples: &mut [f32], _hrtf: &HrtfConfig) -> Result<()> {
        // Mock HRTF processing - in reality would use complex convolution
        for sample in samples {
            *sample *= 0.9; // Slight attenuation to simulate processing
        }
        Ok(())
    }
}

/// Builder for spatial audio controller configuration
#[derive(Debug, Clone)]
pub struct SpatialAudioControllerBuilder {
    config: SpatialAudioConfig,
}

impl SpatialAudioControllerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SpatialAudioConfig::default(),
        }
    }

    /// Enable or disable spatial audio
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set HRTF configuration
    pub fn hrtf_config(mut self, hrtf: HrtfConfig) -> Self {
        self.config.binaural.hrtf = hrtf;
        self
    }

    /// Set room acoustics
    pub fn room_acoustics(mut self, room: RoomAcoustics) -> Self {
        self.config.room_acoustics = Some(room);
        self
    }

    /// Set distance model
    pub fn distance_model(mut self, model: DistanceModel) -> Self {
        self.config.distance_model = model;
        self
    }

    /// Set Doppler effect strength
    pub fn doppler_strength(mut self, strength: f32) -> Self {
        self.config.doppler_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set maximum processing distance
    pub fn max_distance(mut self, distance: f32) -> Self {
        self.config.max_distance = distance;
        self
    }

    /// Build the spatial audio controller
    pub async fn build(self) -> Result<SpatialAudioController> {
        let controller = SpatialAudioController::with_config(self.config).await?;
        Ok(controller)
    }
}

impl Default for SpatialAudioControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SpatialAudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            binaural: BinauralConfig::default(),
            room_acoustics: None,
            speed_of_sound: 343.0,
            doppler_strength: 0.5,
            distance_model: DistanceModel::InverseSquare,
            max_distance: 100.0,
        }
    }
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            hrtf: HrtfConfig::default(),
            compression_enabled: false,
            compression_ratio: 1.0,
            sample_rate: 44100,
            buffer_size: 512,
        }
    }
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            dataset: HrtfDataset::Generic,
            head_circumference: 56.0,
            interpupillary_distance: 6.4,
            crossfeed_enabled: false,
            crossfeed_strength: 0.0,
        }
    }
}

impl Default for RoomAcoustics {
    fn default() -> Self {
        Self {
            dimensions: Position3D {
                x: 10.0,
                y: 3.0,
                z: 8.0,
            },
            absorption: FrequencyResponse {
                frequencies: vec![125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
                responses: vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            },
            reverb_time: 0.5,
            early_reflections_delay: 20.0,
            diffusion: 0.7,
        }
    }
}

impl Default for Position3D {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for Orientation3D {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }
}

impl Default for Velocity3D {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for AudioListener3D {
    fn default() -> Self {
        Self {
            position: Position3D::default(),
            orientation: Orientation3D::default(),
            velocity: Velocity3D::default(),
        }
    }
}

impl Position3D {
    /// Create new position
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Origin position
    pub fn origin() -> Self {
        Self::default()
    }
}

impl Orientation3D {
    /// Create new orientation
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self { yaw, pitch, roll }
    }

    /// Identity orientation
    pub fn identity() -> Self {
        Self::default()
    }
}

impl AudioSource3D {
    /// Create new audio source
    pub fn new(position: Position3D, volume: f32) -> Self {
        Self {
            position,
            orientation: Orientation3D::default(),
            velocity: Velocity3D::default(),
            volume: volume.clamp(0.0, 1.0),
            directivity: DirectivityPattern::Omnidirectional,
        }
    }

    /// Set directivity pattern
    pub fn with_directivity(mut self, pattern: DirectivityPattern) -> Self {
        self.directivity = pattern;
        self
    }

    /// Set velocity for Doppler effect
    pub fn with_velocity(mut self, velocity: Velocity3D) -> Self {
        self.velocity = velocity;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spatial_audio_controller_creation() {
        let controller = SpatialAudioController::new().await.unwrap();
        assert!(controller.is_enabled().await);
    }

    #[tokio::test]
    async fn test_listener_positioning() {
        let controller = SpatialAudioController::new().await.unwrap();
        let position = Position3D::new(1.0, 2.0, 3.0);
        let orientation = Orientation3D::new(0.5, 0.0, 0.0);

        controller
            .set_listener(position.clone(), orientation.clone())
            .await
            .unwrap();

        let listener = controller.get_listener().await;
        assert_eq!(listener.position, position);
        assert_eq!(listener.orientation, orientation);
    }

    #[tokio::test]
    async fn test_source_management() {
        let controller = SpatialAudioController::new().await.unwrap();
        let source = AudioSource3D::new(Position3D::new(5.0, 0.0, 0.0), 0.8);

        let source_id = controller.add_source(source).await.unwrap();
        assert_eq!(source_id, 0);

        let sources = controller.get_sources().await;
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].volume, 0.8);

        controller.remove_source(source_id).await.unwrap();
        let sources = controller.get_sources().await;
        assert_eq!(sources.len(), 0);
    }

    #[tokio::test]
    async fn test_preset_application() {
        let controller = SpatialAudioController::new().await.unwrap();
        controller.apply_preset("gaming").await.unwrap();

        let config = controller.get_config().await;
        assert_eq!(config.distance_model, DistanceModel::InverseSquare);
        assert_eq!(config.doppler_strength, 0.8);
    }

    #[tokio::test]
    async fn test_spatial_audio_builder() {
        let controller = SpatialAudioControllerBuilder::new()
            .enabled(true)
            .doppler_strength(0.7)
            .max_distance(150.0)
            .distance_model(DistanceModel::Inverse)
            .build()
            .await
            .unwrap();

        assert!(controller.is_enabled().await);
        let config = controller.get_config().await;
        assert_eq!(config.doppler_strength, 0.7);
        assert_eq!(config.max_distance, 150.0);
        assert_eq!(config.distance_model, DistanceModel::Inverse);
    }

    #[tokio::test]
    async fn test_position_distance_calculation() {
        let controller = SpatialAudioController::new().await.unwrap();
        let pos1 = Position3D::new(0.0, 0.0, 0.0);
        let pos2 = Position3D::new(3.0, 4.0, 0.0);

        let distance = controller.calculate_distance(&pos1, &pos2);
        assert!((distance - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_room_acoustics() {
        let controller = SpatialAudioController::new().await.unwrap();
        let room = RoomAcoustics::default();

        controller.set_room_acoustics(room.clone()).await.unwrap();

        let config = controller.get_config().await;
        assert!(config.room_acoustics.is_some());
        assert_eq!(config.room_acoustics.unwrap().reverb_time, 0.5);
    }

    #[tokio::test]
    async fn test_preset_listing() {
        let controller = SpatialAudioController::new().await.unwrap();
        let presets = controller.list_presets();
        assert!(presets.contains(&"gaming".to_string()));
        assert!(presets.contains(&"cinema".to_string()));
        assert!(presets.contains(&"vr".to_string()));
    }
}
