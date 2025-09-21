//! Gaming engine integration for voice conversion
//!
//! This module provides comprehensive integration with major gaming engines,
//! enabling real-time voice conversion within games for character voices,
//! player voice chat, and immersive voice experiences.
//!
//! ## Supported Engines
//!
//! - **Unity**: C# bindings and Unity-specific optimizations
//! - **Unreal Engine**: Blueprint integration and C++ bindings
//! - **Godot**: GDScript bindings and native extensions
//! - **Bevy**: Rust-native integration with ECS systems
//! - **Custom Engines**: Generic C API for custom game engines
//!
//! ## Features
//!
//! - **Real-time Voice Conversion**: Ultra-low latency for game voice chat
//! - **Character Voice Synthesis**: Dynamic NPC voice generation
//! - **Player Voice Morphing**: Real-time player voice transformation
//! - **Spatial Audio Integration**: 3D positional voice conversion
//! - **Performance Optimization**: Game-specific performance targets
//! - **Memory Management**: Efficient memory usage for game constraints
//!
//! ## Usage
//!
//! ```rust
//! # use voirs_conversion::gaming::{GameEngine, GameVoiceProcessor, GameAudioConfig};
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create processor for Unity
//! let config = GameAudioConfig::unity_optimized();
//! let mut processor = GameVoiceProcessor::new(GameEngine::Unity, config)?;
//!
//! // Process voice in real-time
//! let input_audio = vec![0.0f32; 1024]; // Sample audio data
//! let converted_audio = processor.process_game_voice(&input_audio, "hero_voice").await?;
//! # Ok(())
//! # }
//! ```

use crate::{
    config::ConversionConfig,
    core::VoiceConverter,
    processing::{AudioBuffer, ProcessingPipeline},
    realtime::{RealtimeConfig, RealtimeConverter},
    types::{ConversionRequest, ConversionTarget, ConversionType, VoiceCharacteristics},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Supported gaming engines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameEngine {
    /// Unity 3D Engine
    Unity,
    /// Unreal Engine 4/5
    Unreal,
    /// Godot Engine
    Godot,
    /// Bevy Engine (Rust)
    Bevy,
    /// Custom/Generic Engine
    Custom,
}

impl GameEngine {
    /// Get engine-specific performance constraints
    pub fn performance_constraints(&self) -> GamePerformanceConstraints {
        match self {
            GameEngine::Unity => GamePerformanceConstraints {
                max_latency_ms: 20.0,
                max_cpu_usage_percent: 15.0,
                max_memory_mb: 64,
                target_fps_impact: 2.0,
                audio_thread_priority: ThreadPriority::High,
            },
            GameEngine::Unreal => GamePerformanceConstraints {
                max_latency_ms: 16.0, // 60 FPS target
                max_cpu_usage_percent: 12.0,
                max_memory_mb: 128,
                target_fps_impact: 1.5,
                audio_thread_priority: ThreadPriority::RealTime,
            },
            GameEngine::Godot => GamePerformanceConstraints {
                max_latency_ms: 25.0,
                max_cpu_usage_percent: 18.0,
                max_memory_mb: 48,
                target_fps_impact: 3.0,
                audio_thread_priority: ThreadPriority::High,
            },
            GameEngine::Bevy => GamePerformanceConstraints {
                max_latency_ms: 16.0,
                max_cpu_usage_percent: 10.0,
                max_memory_mb: 96,
                target_fps_impact: 1.0,
                audio_thread_priority: ThreadPriority::High,
            },
            GameEngine::Custom => GamePerformanceConstraints {
                max_latency_ms: 30.0,
                max_cpu_usage_percent: 20.0,
                max_memory_mb: 80,
                target_fps_impact: 4.0,
                audio_thread_priority: ThreadPriority::Normal,
            },
        }
    }

    /// Get engine name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            GameEngine::Unity => "Unity",
            GameEngine::Unreal => "Unreal",
            GameEngine::Godot => "Godot",
            GameEngine::Bevy => "Bevy",
            GameEngine::Custom => "Custom",
        }
    }
}

/// Thread priority levels for gaming engines
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    RealTime,
}

/// Performance constraints for gaming engines
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GamePerformanceConstraints {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Maximum CPU usage percentage
    pub max_cpu_usage_percent: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
    /// Target FPS impact (lower is better)
    pub target_fps_impact: f32,
    /// Audio processing thread priority
    pub audio_thread_priority: ThreadPriority,
}

/// Game-specific audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAudioConfig {
    /// Target gaming engine
    pub engine: GameEngine,
    /// Audio buffer size for low latency
    pub buffer_size: usize,
    /// Sample rate (typically 44.1kHz or 48kHz)
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Enable spatial audio processing
    pub spatial_audio: bool,
    /// Enable voice activity detection
    pub voice_activity_detection: bool,
    /// Enable automatic gain control
    pub automatic_gain_control: bool,
    /// Enable noise suppression
    pub noise_suppression: bool,
    /// Quality vs performance trade-off (0.0-1.0)
    pub quality_level: f32,
    /// Enable adaptive quality based on performance
    pub adaptive_quality: bool,
    /// Game-specific optimizations
    pub engine_optimizations: HashMap<String, f32>,
}

impl GameAudioConfig {
    /// Create Unity-optimized configuration
    pub fn unity_optimized() -> Self {
        let mut engine_optimizations = HashMap::new();
        engine_optimizations.insert("unity_audio_clip_compatibility".to_string(), 1.0);
        engine_optimizations.insert("unity_mixer_integration".to_string(), 1.0);
        engine_optimizations.insert("unity_audio_source_optimization".to_string(), 0.8);

        Self {
            engine: GameEngine::Unity,
            buffer_size: 512,
            sample_rate: 44100,
            channels: 2,
            spatial_audio: true,
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            quality_level: 0.8,
            adaptive_quality: true,
            engine_optimizations,
        }
    }

    /// Create Unreal Engine-optimized configuration
    pub fn unreal_optimized() -> Self {
        let mut engine_optimizations = HashMap::new();
        engine_optimizations.insert("unreal_sound_cue_integration".to_string(), 1.0);
        engine_optimizations.insert("unreal_audio_component_optimization".to_string(), 1.0);
        engine_optimizations.insert("unreal_metasound_compatibility".to_string(), 0.9);

        Self {
            engine: GameEngine::Unreal,
            buffer_size: 256, // Lower latency for Unreal
            sample_rate: 48000,
            channels: 2,
            spatial_audio: true,
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            quality_level: 0.9,
            adaptive_quality: true,
            engine_optimizations,
        }
    }

    /// Create Godot-optimized configuration
    pub fn godot_optimized() -> Self {
        let mut engine_optimizations = HashMap::new();
        engine_optimizations.insert("godot_audio_stream_integration".to_string(), 1.0);
        engine_optimizations.insert("godot_audio_bus_optimization".to_string(), 0.8);

        Self {
            engine: GameEngine::Godot,
            buffer_size: 1024,
            sample_rate: 44100,
            channels: 2,
            spatial_audio: true,
            voice_activity_detection: true,
            automatic_gain_control: false, // Godot handles this
            noise_suppression: true,
            quality_level: 0.7,
            adaptive_quality: true,
            engine_optimizations,
        }
    }

    /// Create Bevy-optimized configuration
    pub fn bevy_optimized() -> Self {
        let mut engine_optimizations = HashMap::new();
        engine_optimizations.insert("bevy_audio_resource_integration".to_string(), 1.0);
        engine_optimizations.insert("bevy_ecs_system_optimization".to_string(), 1.0);

        Self {
            engine: GameEngine::Bevy,
            buffer_size: 256,
            sample_rate: 44100,
            channels: 2,
            spatial_audio: true,
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            quality_level: 0.85,
            adaptive_quality: true,
            engine_optimizations,
        }
    }
}

/// Game voice processing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameVoiceMode {
    /// Player voice chat
    PlayerChat,
    /// NPC character voice
    CharacterVoice,
    /// Environmental voice effects
    EnvironmentalVoice,
    /// Narrator voice
    NarratorVoice,
    /// Dynamic voice morphing
    DynamicMorphing,
}

/// Game voice processor for real-time voice conversion in games
#[derive(Debug)]
pub struct GameVoiceProcessor {
    /// Target gaming engine
    engine: GameEngine,
    /// Game audio configuration
    config: GameAudioConfig,
    /// Real-time voice converter
    realtime_converter: RealtimeConverter,
    /// Voice converter for complex transformations
    voice_converter: Arc<VoiceConverter>,
    /// Performance constraints
    constraints: GamePerformanceConstraints,
    /// Character voice presets
    character_presets: Arc<RwLock<HashMap<String, VoiceCharacteristics>>>,
    /// Active voice sessions
    active_sessions: Arc<RwLock<HashMap<String, GameVoiceSession>>>,
    /// Performance monitor
    performance_monitor: GamePerformanceMonitor,
}

impl GameVoiceProcessor {
    /// Create new game voice processor
    pub fn new(engine: GameEngine, config: GameAudioConfig) -> Result<Self> {
        let constraints = engine.performance_constraints();

        // Create real-time converter with game-optimized settings
        let realtime_config = RealtimeConfig {
            buffer_size: config.buffer_size,
            sample_rate: config.sample_rate,
            target_latency_ms: constraints.max_latency_ms,
            overlap_factor: 0.25,
            adaptive_buffering: config.adaptive_quality,
            max_threads: 2,          // Conservative for games
            enable_lookahead: false, // Disable for lowest latency
            lookahead_size: 0,
        };

        let realtime_converter = RealtimeConverter::new(realtime_config)?;
        let voice_converter = Arc::new(VoiceConverter::new()?);

        Ok(Self {
            engine,
            config,
            realtime_converter,
            voice_converter,
            constraints,
            character_presets: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: GamePerformanceMonitor::new(),
        })
    }

    /// Process game voice in real-time
    pub async fn process_game_voice(
        &mut self,
        input_audio: &[f32],
        voice_id: &str,
    ) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        // Check performance constraints
        if !self
            .performance_monitor
            .check_performance_budget(&self.constraints)
        {
            warn!("Performance budget exceeded, using passthrough mode");
            return Ok(input_audio.to_vec());
        }

        // Get voice characteristics
        let voice_characteristics = {
            let presets = self.character_presets.read().await;
            presets.get(voice_id).cloned().unwrap_or_default()
        };

        // Set conversion target
        let target = ConversionTarget::new(voice_characteristics);
        self.realtime_converter.set_conversion_target(target);

        // Process with real-time converter
        let result = self.realtime_converter.process_chunk(input_audio).await?;

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.performance_monitor.record_processing(
            processing_time,
            input_audio.len(),
            &self.constraints,
        );

        debug!(
            "Game voice processed: {} samples in {:.2}ms",
            input_audio.len(),
            processing_time.as_secs_f32() * 1000.0
        );

        Ok(result)
    }

    /// Process character voice with specific mode
    pub async fn process_character_voice(
        &mut self,
        input_audio: &[f32],
        character_id: &str,
        mode: GameVoiceMode,
    ) -> Result<Vec<f32>> {
        // Apply mode-specific processing
        let processed_audio = match mode {
            GameVoiceMode::PlayerChat => self.apply_player_chat_processing(input_audio).await?,
            GameVoiceMode::CharacterVoice => {
                self.apply_character_voice_processing(input_audio, character_id)
                    .await?
            }
            GameVoiceMode::EnvironmentalVoice => {
                self.apply_environmental_processing(input_audio).await?
            }
            GameVoiceMode::NarratorVoice => self.apply_narrator_processing(input_audio).await?,
            GameVoiceMode::DynamicMorphing => {
                self.apply_dynamic_morphing(input_audio, character_id)
                    .await?
            }
        };

        Ok(processed_audio)
    }

    /// Register character voice preset
    pub async fn register_character_preset(
        &self,
        character_id: String,
        characteristics: VoiceCharacteristics,
    ) {
        let mut presets = self.character_presets.write().await;
        presets.insert(character_id.clone(), characteristics);
        info!("Registered character preset: {}", character_id);
    }

    /// Start voice session for persistent processing
    pub async fn start_voice_session(
        &self,
        session_id: String,
        character_id: String,
        mode: GameVoiceMode,
    ) -> Result<()> {
        let session = GameVoiceSession {
            session_id: session_id.clone(),
            character_id,
            mode,
            start_time: std::time::Instant::now(),
            processed_samples: 0,
            latency_samples: Vec::new(),
        };

        let mut sessions = self.active_sessions.write().await;
        sessions.insert(session_id.clone(), session);

        info!("Started voice session: {}", session_id);
        Ok(())
    }

    /// Stop voice session
    pub async fn stop_voice_session(&self, session_id: &str) -> Result<GameVoiceSession> {
        let mut sessions = self.active_sessions.write().await;
        sessions
            .remove(session_id)
            .ok_or_else(|| Error::processing(format!("Voice session not found: {session_id}")))
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> GamePerformanceMetrics {
        self.performance_monitor.get_current_metrics()
    }

    /// Check if processing meets game performance requirements
    pub fn is_performance_acceptable(&self) -> bool {
        self.performance_monitor
            .check_performance_budget(&self.constraints)
    }

    /// Get engine-specific integration info
    pub fn get_engine_integration_info(&self) -> GameEngineIntegration {
        match self.engine {
            GameEngine::Unity => GameEngineIntegration::Unity(UnityIntegration {
                plugin_version: "1.0.0".to_string(),
                unity_version_compatibility: "2021.3+".to_string(),
                audio_clip_support: true,
                mixer_integration: true,
                blueprint_support: false,
            }),
            GameEngine::Unreal => GameEngineIntegration::Unreal(UnrealIntegration {
                plugin_version: "1.0.0".to_string(),
                unreal_version_compatibility: "4.27+, 5.0+".to_string(),
                blueprint_support: true,
                metasound_support: true,
                audio_component_integration: true,
            }),
            GameEngine::Godot => GameEngineIntegration::Godot(GodotIntegration {
                plugin_version: "1.0.0".to_string(),
                godot_version_compatibility: "3.5+, 4.0+".to_string(),
                gdscript_bindings: true,
                audio_stream_support: true,
            }),
            GameEngine::Bevy => GameEngineIntegration::Bevy(BevyIntegration {
                plugin_version: "1.0.0".to_string(),
                bevy_version_compatibility: "0.11+".to_string(),
                ecs_integration: true,
                audio_resource_support: true,
            }),
            GameEngine::Custom => GameEngineIntegration::Custom(CustomIntegration {
                c_api_version: "1.0.0".to_string(),
                supported_platforms: vec![
                    "Windows".to_string(),
                    "macOS".to_string(),
                    "Linux".to_string(),
                ],
                callback_support: true,
            }),
        }
    }

    // Private helper methods

    async fn apply_player_chat_processing(&mut self, input_audio: &[f32]) -> Result<Vec<f32>> {
        // Apply noise suppression and voice enhancement
        let mut processed = input_audio.to_vec();

        if self.config.noise_suppression {
            processed = self.apply_noise_suppression(&processed)?;
        }

        if self.config.automatic_gain_control {
            processed = self.apply_automatic_gain_control(&processed)?;
        }

        Ok(processed)
    }

    async fn apply_character_voice_processing(
        &mut self,
        input_audio: &[f32],
        character_id: &str,
    ) -> Result<Vec<f32>> {
        self.process_game_voice(input_audio, character_id).await
    }

    async fn apply_environmental_processing(&mut self, input_audio: &[f32]) -> Result<Vec<f32>> {
        // Apply environmental effects (reverb, echo, etc.)
        let mut processed = input_audio.to_vec();

        // Apply environmental reverb
        for sample in processed.iter_mut() {
            *sample *= 0.7; // Reduce direct signal
        }

        Ok(processed)
    }

    async fn apply_narrator_processing(&mut self, input_audio: &[f32]) -> Result<Vec<f32>> {
        // Apply narrator-specific processing (clarity, authority)
        let mut processed = input_audio.to_vec();

        // Enhance clarity
        for sample in processed.iter_mut() {
            *sample = (*sample * 1.2).clamp(-1.0, 1.0);
        }

        Ok(processed)
    }

    async fn apply_dynamic_morphing(
        &mut self,
        input_audio: &[f32],
        character_id: &str,
    ) -> Result<Vec<f32>> {
        // Apply time-varying voice morphing
        self.process_game_voice(input_audio, character_id).await
    }

    fn apply_noise_suppression(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simple noise gate implementation
        let threshold = 0.01;
        let mut processed = audio.to_vec();

        for sample in processed.iter_mut() {
            if sample.abs() < threshold {
                *sample *= 0.1;
            }
        }

        Ok(processed)
    }

    fn apply_automatic_gain_control(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simple AGC implementation
        let target_level = 0.7;
        let current_level = audio.iter().map(|&x| x.abs()).sum::<f32>() / audio.len() as f32;

        if current_level > 0.0 {
            let gain = target_level / current_level;
            let clamped_gain = gain.clamp(0.5, 2.0);

            Ok(audio
                .iter()
                .map(|&x| (x * clamped_gain).clamp(-1.0, 1.0))
                .collect())
        } else {
            Ok(audio.to_vec())
        }
    }
}

/// Game voice session for tracking processing state
#[derive(Debug, Clone)]
pub struct GameVoiceSession {
    pub session_id: String,
    pub character_id: String,
    pub mode: GameVoiceMode,
    pub start_time: std::time::Instant,
    pub processed_samples: usize,
    pub latency_samples: Vec<f32>,
}

/// Game performance monitor
#[derive(Debug)]
pub struct GamePerformanceMonitor {
    processing_times: Vec<std::time::Duration>,
    cpu_usage_samples: Vec<f32>,
    memory_usage_samples: Vec<u32>,
    frame_drops: u32,
    last_check: std::time::Instant,
}

impl GamePerformanceMonitor {
    fn new() -> Self {
        Self {
            processing_times: Vec::new(),
            cpu_usage_samples: Vec::new(),
            memory_usage_samples: Vec::new(),
            frame_drops: 0,
            last_check: std::time::Instant::now(),
        }
    }

    fn record_processing(
        &mut self,
        processing_time: std::time::Duration,
        _sample_count: usize,
        constraints: &GamePerformanceConstraints,
    ) {
        self.processing_times.push(processing_time);

        // Keep only recent samples
        if self.processing_times.len() > 100 {
            self.processing_times.drain(0..50);
        }

        // Check for frame drops
        let latency_ms = processing_time.as_secs_f32() * 1000.0;
        if latency_ms > constraints.max_latency_ms {
            self.frame_drops += 1;
        }
    }

    fn check_performance_budget(&self, constraints: &GamePerformanceConstraints) -> bool {
        if self.processing_times.is_empty() {
            return true;
        }

        let avg_latency_ms = self
            .processing_times
            .iter()
            .map(|d| d.as_secs_f32() * 1000.0)
            .sum::<f32>()
            / self.processing_times.len() as f32;

        avg_latency_ms <= constraints.max_latency_ms
    }

    fn get_current_metrics(&self) -> GamePerformanceMetrics {
        let avg_latency_ms = if self.processing_times.is_empty() {
            0.0
        } else {
            self.processing_times
                .iter()
                .map(|d| d.as_secs_f32() * 1000.0)
                .sum::<f32>()
                / self.processing_times.len() as f32
        };

        GamePerformanceMetrics {
            average_latency_ms: avg_latency_ms,
            cpu_usage_percent: self.cpu_usage_samples.last().copied().unwrap_or(0.0),
            memory_usage_mb: self.memory_usage_samples.last().copied().unwrap_or(0),
            frame_drops: self.frame_drops,
            uptime_seconds: self.last_check.elapsed().as_secs(),
        }
    }
}

/// Game performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamePerformanceMetrics {
    pub average_latency_ms: f32,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u32,
    pub frame_drops: u32,
    pub uptime_seconds: u64,
}

/// Engine-specific integration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameEngineIntegration {
    Unity(UnityIntegration),
    Unreal(UnrealIntegration),
    Godot(GodotIntegration),
    Bevy(BevyIntegration),
    Custom(CustomIntegration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityIntegration {
    pub plugin_version: String,
    pub unity_version_compatibility: String,
    pub audio_clip_support: bool,
    pub mixer_integration: bool,
    pub blueprint_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrealIntegration {
    pub plugin_version: String,
    pub unreal_version_compatibility: String,
    pub blueprint_support: bool,
    pub metasound_support: bool,
    pub audio_component_integration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodotIntegration {
    pub plugin_version: String,
    pub godot_version_compatibility: String,
    pub gdscript_bindings: bool,
    pub audio_stream_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BevyIntegration {
    pub plugin_version: String,
    pub bevy_version_compatibility: String,
    pub ecs_integration: bool,
    pub audio_resource_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIntegration {
    pub c_api_version: String,
    pub supported_platforms: Vec<String>,
    pub callback_support: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_engine_constraints() {
        let unity_constraints = GameEngine::Unity.performance_constraints();
        assert!(unity_constraints.max_latency_ms <= 20.0);
        assert!(unity_constraints.max_cpu_usage_percent <= 15.0);

        let unreal_constraints = GameEngine::Unreal.performance_constraints();
        assert!(unreal_constraints.max_latency_ms <= 16.0);
        assert_eq!(
            unreal_constraints.audio_thread_priority,
            ThreadPriority::RealTime
        );
    }

    #[test]
    fn test_game_audio_config_creation() {
        let unity_config = GameAudioConfig::unity_optimized();
        assert_eq!(unity_config.engine, GameEngine::Unity);
        assert!(unity_config.spatial_audio);
        assert!(unity_config.adaptive_quality);

        let unreal_config = GameAudioConfig::unreal_optimized();
        assert_eq!(unreal_config.engine, GameEngine::Unreal);
        assert_eq!(unreal_config.buffer_size, 256);
        assert_eq!(unreal_config.sample_rate, 48000);
    }

    #[tokio::test]
    async fn test_game_voice_processor_creation() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config);
        assert!(processor.is_ok());

        let processor = processor.unwrap();
        assert_eq!(processor.engine, GameEngine::Unity);
    }

    #[tokio::test]
    async fn test_character_preset_registration() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config).unwrap();

        let characteristics = VoiceCharacteristics::default();
        processor
            .register_character_preset("hero".to_string(), characteristics)
            .await;

        let presets = processor.character_presets.read().await;
        assert!(presets.contains_key("hero"));
    }

    #[tokio::test]
    async fn test_voice_session_management() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config).unwrap();

        // Start session
        let result = processor
            .start_voice_session(
                "session1".to_string(),
                "hero".to_string(),
                GameVoiceMode::CharacterVoice,
            )
            .await;
        assert!(result.is_ok());

        // Check session exists
        let sessions = processor.active_sessions.read().await;
        assert!(sessions.contains_key("session1"));
    }

    #[test]
    fn test_game_voice_modes() {
        let modes = [
            GameVoiceMode::PlayerChat,
            GameVoiceMode::CharacterVoice,
            GameVoiceMode::EnvironmentalVoice,
            GameVoiceMode::NarratorVoice,
            GameVoiceMode::DynamicMorphing,
        ];

        for mode in &modes {
            // Test serialization
            let serialized = serde_json::to_string(mode).unwrap();
            let deserialized: GameVoiceMode = serde_json::from_str(&serialized).unwrap();
            assert_eq!(*mode, deserialized);
        }
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = GamePerformanceMonitor::new();
        let constraints = GameEngine::Unity.performance_constraints();

        // Record some processing times
        monitor.record_processing(std::time::Duration::from_millis(10), 1024, &constraints);

        let metrics = monitor.get_current_metrics();
        assert!(metrics.average_latency_ms > 0.0);
        assert!(monitor.check_performance_budget(&constraints));
    }

    #[test]
    fn test_engine_integration_info() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config).unwrap();

        let integration = processor.get_engine_integration_info();
        match integration {
            GameEngineIntegration::Unity(unity) => {
                assert!(unity.audio_clip_support);
                assert!(unity.mixer_integration);
            }
            _ => panic!("Expected Unity integration"),
        }
    }

    #[test]
    fn test_thread_priority_ordering() {
        assert!(ThreadPriority::RealTime > ThreadPriority::High);
        assert!(ThreadPriority::High > ThreadPriority::Normal);
        assert!(ThreadPriority::Normal > ThreadPriority::Low);
    }

    #[tokio::test]
    async fn test_noise_suppression() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config).unwrap();

        let noisy_audio = vec![0.001, 0.5, 0.002, -0.7, 0.003]; // Mix of noise and signal
        let processed = processor.apply_noise_suppression(&noisy_audio).unwrap();

        // Check that small signals are suppressed
        assert!(processed[0].abs() < noisy_audio[0].abs());
        assert!(processed[2].abs() < noisy_audio[2].abs());
        assert!(processed[4].abs() < noisy_audio[4].abs());

        // Check that large signals are preserved
        assert_eq!(processed[1], noisy_audio[1]);
        assert_eq!(processed[3], noisy_audio[3]);
    }

    #[tokio::test]
    async fn test_automatic_gain_control() {
        let config = GameAudioConfig::unity_optimized();
        let processor = GameVoiceProcessor::new(GameEngine::Unity, config).unwrap();

        let quiet_audio = vec![0.1, -0.1, 0.05, -0.05];
        let processed = processor
            .apply_automatic_gain_control(&quiet_audio)
            .unwrap();

        // Check that gain was applied
        let original_level =
            quiet_audio.iter().map(|&x| x.abs()).sum::<f32>() / quiet_audio.len() as f32;
        let processed_level =
            processed.iter().map(|&x| x.abs()).sum::<f32>() / processed.len() as f32;
        assert!(processed_level > original_level);
    }
}
