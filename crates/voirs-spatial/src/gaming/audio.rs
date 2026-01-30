//! Gaming audio manager and core audio processing

use super::config::GamingConfig;
use super::types::{
    AttenuationCurve, FrameTimer, GameAudioSource, GameAudioSourceData, GamingMetrics,
    PlaybackState,
};
use super::types::{AttenuationSettings, AudioCategory, GameEngine};
use crate::config::SpatialConfig;
use crate::core::SpatialProcessor;
use crate::position::SoundSource;
use crate::types::Position3D;
use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Gaming audio manager
pub struct GamingAudioManager {
    /// Spatial processor
    pub(super) processor: Arc<Mutex<SpatialProcessor>>,
    /// Configuration
    pub(super) config: GamingConfig,
    /// Active audio sources
    pub(super) sources: Arc<Mutex<HashMap<u32, GameAudioSourceData>>>,
    /// Next source ID
    pub(super) next_source_id: Arc<Mutex<u32>>,
    /// Performance metrics
    pub(super) metrics: Arc<Mutex<GamingMetrics>>,
    /// Frame timing
    pub(super) frame_timer: Arc<Mutex<FrameTimer>>,
}

impl GamingAudioManager {
    /// Create a new gaming audio manager
    pub async fn new(config: GamingConfig) -> Result<Self> {
        let spatial_config = SpatialConfig::default();
        let processor = SpatialProcessor::new(spatial_config).await?;

        Ok(Self {
            processor: Arc::new(Mutex::new(processor)),
            config,
            sources: Arc::new(Mutex::new(HashMap::new())),
            next_source_id: Arc::new(Mutex::new(1)),
            metrics: Arc::new(Mutex::new(GamingMetrics::default())),
            frame_timer: Arc::new(Mutex::new(FrameTimer::default())),
        })
    }

    /// Initialize the gaming audio system
    pub async fn initialize(&mut self) -> Result<()> {
        // Engine-specific initialization
        match self.config.engine {
            GameEngine::Unity => self.initialize_unity().await?,
            GameEngine::Unreal => self.initialize_unreal().await?,
            GameEngine::Godot => self.initialize_godot().await?,
            GameEngine::Custom => self.initialize_custom().await?,
            GameEngine::WebEngine => self.initialize_web().await?,
            GameEngine::Console => self.initialize_console().await?,
            GameEngine::PlayStation => self.initialize_console().await?,
            GameEngine::Xbox => self.initialize_console().await?,
            GameEngine::NintendoSwitch => self.initialize_console().await?,
        }

        Ok(())
    }

    /// Create a new audio source
    pub fn create_source(
        &self,
        category: AudioCategory,
        priority: u8,
        position: Position3D,
    ) -> Result<GameAudioSource> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;
        let mut next_id = self
            .next_source_id
            .lock()
            .map_err(|_| Error::LegacyAudio("ID lock poisoned".to_string()))?;

        let id = *next_id;
        *next_id += 1;

        let handle = GameAudioSource {
            id,
            category,
            priority,
        };

        let spatial_source = SoundSource::new_point(format!("game_source_{id}"), position);

        let source_data = GameAudioSourceData {
            handle,
            spatial_source,
            audio_data: Vec::new(),
            state: PlaybackState::Stopped,
            volume: 1.0,
            looping: false,
            attenuation: AttenuationSettings::default(),
        };

        sources.insert(id, source_data);

        Ok(handle)
    }

    /// Set audio data for a source
    pub fn set_audio_data(&self, source: GameAudioSource, audio_data: Vec<f32>) -> Result<()> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;

        if let Some(source_data) = sources.get_mut(&source.id) {
            source_data.audio_data = audio_data;
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!(
                "Source {id} not found",
                id = source.id
            )))
        }
    }

    /// Play an audio source
    pub fn play_source(&self, source: GameAudioSource) -> Result<()> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;

        if let Some(source_data) = sources.get_mut(&source.id) {
            source_data.state = PlaybackState::Playing;
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!(
                "Source {id} not found",
                id = source.id
            )))
        }
    }

    /// Stop an audio source
    pub fn stop_source(&self, source: GameAudioSource) -> Result<()> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;

        if let Some(source_data) = sources.get_mut(&source.id) {
            source_data.state = PlaybackState::Stopped;
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!(
                "Source {id} not found",
                id = source.id
            )))
        }
    }

    /// Update source position
    pub fn update_source_position(
        &self,
        source: GameAudioSource,
        position: Position3D,
    ) -> Result<()> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;

        if let Some(source_data) = sources.get_mut(&source.id) {
            source_data.spatial_source.set_position(position);
            Ok(())
        } else {
            Err(Error::LegacyAudio(format!(
                "Source {id} not found",
                id = source.id
            )))
        }
    }

    /// Update listener position
    pub fn update_listener(
        &self,
        position: Position3D,
        orientation: (f32, f32, f32),
    ) -> Result<()> {
        let processor = self
            .processor
            .lock()
            .map_err(|_| Error::LegacyAudio("Processor lock poisoned".to_string()))?;

        // Update listener position and orientation
        // This would be implemented when the processor API is extended
        Ok(())
    }

    /// Process audio for current frame
    pub fn process_frame(&self, output_buffer: &mut [f32]) -> Result<()> {
        let frame_start = Instant::now();

        // Update frame timing
        {
            let mut timer = self
                .frame_timer
                .lock()
                .map_err(|_| Error::LegacyAudio("Timer lock poisoned".to_string()))?;
            timer.frame_count += 1;

            let frame_time = frame_start.duration_since(timer.last_frame).as_secs_f32();
            timer.last_frame = frame_start;

            // Update FPS calculation
            if frame_start.duration_since(timer.fps_timer).as_secs_f32() >= 1.0 {
                timer.fps_accumulator = timer.frame_count as f32;
                timer.frame_count = 0;
                timer.fps_timer = frame_start;
            }
        }

        // Clear output buffer
        output_buffer.fill(0.0);

        // Process all active sources
        let sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;
        let mut active_count = 0;

        for source_data in sources.values() {
            if source_data.state == PlaybackState::Playing && !source_data.audio_data.is_empty() {
                active_count += 1;

                // Simple mixing (placeholder for real spatial processing)
                let volume = source_data.volume * self.calculate_distance_attenuation(source_data);
                for (i, &sample) in source_data.audio_data.iter().enumerate() {
                    if i >= output_buffer.len() {
                        break;
                    }
                    output_buffer[i] += sample * volume;
                }
            }
        }

        // Update metrics
        {
            let mut metrics = self
                .metrics
                .lock()
                .map_err(|_| Error::LegacyAudio("Metrics lock poisoned".to_string()))?;
            metrics.audio_time_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
            metrics.active_sources = active_count;

            let timer = self
                .frame_timer
                .lock()
                .map_err(|_| Error::LegacyAudio("Timer lock poisoned".to_string()))?;
            metrics.fps = timer.fps_accumulator;
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> Result<GamingMetrics> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| Error::LegacyAudio("Metrics lock poisoned".to_string()))?;
        Ok(metrics.clone())
    }

    /// Remove an audio source
    pub fn remove_source(&self, source: GameAudioSource) -> Result<()> {
        let mut sources = self
            .sources
            .lock()
            .map_err(|_| Error::LegacyAudio("Sources lock poisoned".to_string()))?;
        sources.remove(&source.id);
        Ok(())
    }

    /// Calculate distance attenuation for a source
    pub(super) fn calculate_distance_attenuation(&self, source_data: &GameAudioSourceData) -> f32 {
        // Simple distance calculation (placeholder)
        let distance = 1.0; // Would calculate from listener and source positions
        let settings = &source_data.attenuation;

        if distance <= settings.min_distance {
            return 1.0;
        }

        if distance >= settings.max_distance {
            return 0.0;
        }

        let normalized_distance =
            (distance - settings.min_distance) / (settings.max_distance - settings.min_distance);

        (match settings.curve {
            AttenuationCurve::Linear => 1.0 - normalized_distance,
            AttenuationCurve::Logarithmic => 1.0 - normalized_distance.ln().abs(),
            AttenuationCurve::Exponential => (1.0 - normalized_distance).powf(2.0),
            AttenuationCurve::Custom => 1.0 - normalized_distance, // Custom implementation
        }) * settings.factor
    }

    // Engine-specific initialization methods (stubs for non-Unity/Unreal engines)
    pub(super) async fn initialize_godot(&self) -> Result<()> {
        // Godot-specific initialization
        Ok(())
    }

    pub(super) async fn initialize_custom(&self) -> Result<()> {
        // Custom engine initialization
        Ok(())
    }

    pub(super) async fn initialize_web(&self) -> Result<()> {
        // Web engine initialization
        Ok(())
    }

    pub(super) async fn initialize_console(&self) -> Result<()> {
        // Console-specific initialization
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaming::c_api::*;
    use std::ffi::CString;

    #[tokio::test]
    async fn test_gaming_manager_creation() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_audio_source_creation() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let position = Position3D::new(1.0, 0.0, 0.0);
        let source = manager.create_source(AudioCategory::Sfx, 128, position);
        assert!(source.is_ok());

        let source = source.unwrap();
        assert_eq!(source.category, AudioCategory::Sfx);
        assert_eq!(source.priority, 128);
    }

    #[tokio::test]
    async fn test_source_playback_control() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let position = Position3D::new(1.0, 0.0, 0.0);
        let source = manager
            .create_source(AudioCategory::Sfx, 128, position)
            .unwrap();

        // Set audio data
        let audio_data = vec![0.5; 1024];
        assert!(manager.set_audio_data(source, audio_data).is_ok());

        // Play source
        assert!(manager.play_source(source).is_ok());

        // Stop source
        assert!(manager.stop_source(source).is_ok());
    }

    #[tokio::test]
    async fn test_position_updates() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let position = Position3D::new(1.0, 0.0, 0.0);
        let source = manager
            .create_source(AudioCategory::Sfx, 128, position)
            .unwrap();

        let new_position = Position3D::new(2.0, 1.0, 0.0);
        assert!(manager.update_source_position(source, new_position).is_ok());

        let listener_position = Position3D::new(0.0, 0.0, 0.0);
        assert!(manager
            .update_listener(listener_position, (0.0, 0.0, 1.0))
            .is_ok());
    }

    #[tokio::test]
    async fn test_audio_processing() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let position = Position3D::new(1.0, 0.0, 0.0);
        let source = manager
            .create_source(AudioCategory::Sfx, 128, position)
            .unwrap();

        let audio_data = vec![0.5; 1024];
        manager.set_audio_data(source, audio_data).unwrap();
        manager.play_source(source).unwrap();

        let mut output_buffer = vec![0.0; 1024];
        assert!(manager.process_frame(&mut output_buffer).is_ok());

        // Check that audio was mixed into output
        let has_audio = output_buffer.iter().any(|&x| x != 0.0);
        assert!(has_audio);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let metrics = manager.get_metrics();
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.active_sources, 0);
    }

    #[tokio::test]
    async fn test_attenuation_curves() {
        let config = GamingConfig::default();
        let manager = GamingAudioManager::new(config).await.unwrap();

        let position = Position3D::new(1.0, 0.0, 0.0);
        let source = manager
            .create_source(AudioCategory::Sfx, 128, position)
            .unwrap();

        // Test distance attenuation calculation
        let sources = manager
            .sources
            .lock()
            .expect("Failed to lock sources in test");
        if let Some(source_data) = sources.get(&source.id) {
            let attenuation = manager.calculate_distance_attenuation(source_data);
            assert!(attenuation >= 0.0 && attenuation <= 1.0);
        }
    }

    #[test]
    fn test_c_api_functions() {
        // Test C API function signatures (basic validation)
        let config = GamingConfig::default();
        let config_json =
            serde_json::to_string(&config).expect("Failed to serialize config in test");
        let config_cstring = CString::new(config_json).expect("Failed to create CString in test");

        unsafe {
            let manager = voirs_gaming_create_manager(config_cstring.as_ptr());
            assert!(!manager.is_null());

            let source_id = voirs_gaming_create_source(manager, 4, 128, 1.0, 0.0, 0.0);
            assert!(source_id >= 0);

            let result = voirs_gaming_update_listener(manager, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            assert_eq!(result, 0);

            voirs_gaming_destroy_manager(manager);
        }
    }

    #[tokio::test]
    async fn test_unity_initialization() {
        let mut config = GamingConfig::default();
        config.engine = GameEngine::Unity;
        config.target_fps = 60;
        config.memory_budget_mb = 512;

        let manager = GamingAudioManager::new(config).await.unwrap();

        // Test Unity-specific initialization
        let result = manager.initialize_unity().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_unreal_initialization() {
        let mut config = GamingConfig::default();
        config.engine = GameEngine::Unreal;
        config.target_fps = 120;
        config.memory_budget_mb = 1024;

        let manager = GamingAudioManager::new(config).await.unwrap();

        // Test Unreal-specific initialization
        let result = manager.initialize_unreal().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_unity_c_api_functions() {
        let config = GamingConfig {
            engine: GameEngine::Unity,
            ..GamingConfig::default()
        };
        let config_json =
            serde_json::to_string(&config).expect("Failed to serialize Unity config in test");
        let config_cstring =
            CString::new(config_json).expect("Failed to create Unity CString in test");

        unsafe {
            let manager = voirs_gaming_create_manager(config_cstring.as_ptr());
            assert!(!manager.is_null());

            // Test Unity-specific initialization
            let unity_config = r#"{"unity_version": "2023.1", "audio_mixer": "MainMixer"}"#;
            let unity_config_cstring =
                CString::new(unity_config).expect("Failed to create Unity config CString in test");
            let result = voirs_unity_initialize_manager(manager, unity_config_cstring.as_ptr());
            assert_eq!(result, 0);

            // Test Unity AudioSource creation
            let source_id = voirs_unity_create_audiosource(
                manager, 12345, // GameObject ID
                67890, // AudioClip ID
                1.0, 0.0, 0.0, // Position
                0.8, // Volume
                1.0, // Pitch
            );
            assert!(source_id >= 0);

            // Test Unity AudioListener transform
            let result = voirs_unity_set_audio_listener_transform(
                manager, 0.0, 0.0, 0.0, // Position
                0.0, 0.0, 0.0, 1.0, // Quaternion rotation
            );
            assert_eq!(result, 0);

            voirs_gaming_destroy_manager(manager);
        }
    }

    #[test]
    fn test_unreal_c_api_functions() {
        let config = GamingConfig {
            engine: GameEngine::Unreal,
            ..GamingConfig::default()
        };
        let config_json =
            serde_json::to_string(&config).expect("Failed to serialize Unreal config in test");
        let config_cstring =
            CString::new(config_json).expect("Failed to create Unreal CString in test");

        unsafe {
            let manager = voirs_gaming_create_manager(config_cstring.as_ptr());
            assert!(!manager.is_null());

            // Test Unreal-specific initialization
            let unreal_config = r#"{"unreal_version": "5.3", "audio_engine": "default"}"#;
            let unreal_config_cstring = CString::new(unreal_config)
                .expect("Failed to create Unreal config CString in test");
            let result = voirs_unreal_initialize_manager(manager, unreal_config_cstring.as_ptr());
            assert_eq!(result, 0);

            // Test Unreal AudioComponent creation
            let source_id = voirs_unreal_create_audio_component(
                manager, 54321, // Actor ID
                98765, // SoundWave ID
                100.0, 50.0, 0.0,    // Location (Unreal units)
                0.9,    // Volume multiplier
                1.2,    // Pitch multiplier
                1000.0, // Attenuation distance
            );
            assert!(source_id >= 0);

            // Test Unreal AudioListener transform
            let result = voirs_unreal_set_audio_listener_transform(
                manager, 0.0, 0.0, 100.0, // Location
                0.0, 90.0, 0.0, // Rotation (pitch, yaw, roll)
            );
            assert_eq!(result, 0);

            // Test location update
            let result = voirs_unreal_set_audio_component_location(
                manager, source_id, 200.0, 100.0, 50.0, // New location
            );
            assert_eq!(result, 0);

            voirs_gaming_destroy_manager(manager);
        }
    }

    #[test]
    fn test_engine_agnostic_functions() {
        let config = GamingConfig::default();
        let config_json =
            serde_json::to_string(&config).expect("Failed to serialize config in test");
        let config_cstring = CString::new(config_json).expect("Failed to create CString in test");

        unsafe {
            let manager = voirs_gaming_create_manager(config_cstring.as_ptr());
            assert!(!manager.is_null());

            let source_id = voirs_gaming_create_source(manager, 4, 128, 1.0, 0.0, 0.0);
            assert!(source_id >= 0);

            // Test attenuation settings
            let result = voirs_gaming_set_source_attenuation(
                manager, source_id, 1.0,   // min_distance
                100.0, // max_distance
                1,     // logarithmic curve
                1.0,   // attenuation_factor
            );
            assert_eq!(result, 0);

            // Test performance metrics
            let result = voirs_gaming_get_performance_metrics(manager, std::ptr::null_mut());
            assert_eq!(result, -1); // Should return -1 for null pointer

            voirs_gaming_destroy_manager(manager);
        }
    }

    #[tokio::test]
    async fn test_gaming_engine_selection() {
        // Test different engine configurations
        let engines = [
            GameEngine::Unity,
            GameEngine::Unreal,
            GameEngine::Godot,
            GameEngine::Custom,
        ];

        for engine in engines {
            let config = GamingConfig {
                engine,
                target_fps: 60,
                ..GamingConfig::default()
            };

            let mut manager = GamingAudioManager::new(config).await.unwrap();
            let result = manager.initialize().await;
            assert!(result.is_ok());
        }
    }
}
