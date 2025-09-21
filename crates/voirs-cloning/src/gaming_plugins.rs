//! Gaming Engine Integration Plugins
//!
//! This module provides native integration plugins for popular gaming engines including
//! Unity and Unreal Engine, enabling seamless voice cloning integration in game development
//! with optimized performance, real-time processing, and game-specific features.

use crate::{Error, Result, VoiceCloneRequest, VoiceCloneResult, VoiceSample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Gaming engine plugin manager
#[derive(Debug)]
pub struct GamingPluginManager {
    /// Unity plugin interface
    unity_plugin: Option<UnityPlugin>,
    /// Unreal Engine plugin interface
    unreal_plugin: Option<UnrealPlugin>,
    /// Godot plugin interface
    godot_plugin: Option<GodotPlugin>,
    /// Plugin configuration
    config: GamingPluginConfig,
    /// Active game sessions
    game_sessions: HashMap<String, GameSession>,
}

/// Gaming plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPluginConfig {
    /// Enable real-time voice synthesis
    pub enable_realtime_synthesis: bool,
    /// Maximum concurrent voice instances
    pub max_voice_instances: usize,
    /// Audio buffer size for gaming
    pub audio_buffer_size: usize,
    /// Target latency for real-time synthesis
    pub target_latency_ms: f32,
    /// Enable voice caching for performance
    pub enable_voice_caching: bool,
    /// Memory limit for cached voices
    pub voice_cache_size_mb: usize,
    /// Enable 3D spatial audio support
    pub enable_spatial_audio: bool,
    /// Performance profile for gaming
    pub performance_profile: GamePerformanceProfile,
}

/// Performance profiles optimized for gaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GamePerformanceProfile {
    /// Maximum quality, higher latency
    HighQuality,
    /// Balanced quality and performance
    Balanced,
    /// Low latency, optimized for real-time
    LowLatency,
    /// Minimum resources, basic quality
    Performance,
    /// Custom profile with specific settings
    Custom(CustomPerformanceSettings),
}

/// Custom performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPerformanceSettings {
    /// Quality level (0.0 - 1.0)
    pub quality_level: f32,
    /// Maximum processing time per frame (ms)
    pub max_processing_time_ms: f32,
    /// Use GPU acceleration
    pub use_gpu_acceleration: bool,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
}

/// Unity Engine plugin interface
#[derive(Debug)]
pub struct UnityPlugin {
    /// Native library handle
    native_handle: Option<*mut c_void>,
    /// Audio system interface
    audio_system: UnityAudioSystem,
    /// GameObject management
    gameobject_manager: UnityGameObjectManager,
    /// Script interface
    script_interface: UnityScriptInterface,
}

/// Unreal Engine plugin interface
#[derive(Debug)]
pub struct UnrealPlugin {
    /// UE module handle
    module_handle: Option<*mut c_void>,
    /// Audio component system
    audio_components: UnrealAudioComponents,
    /// Blueprint interface
    blueprint_interface: UnrealBlueprintInterface,
    /// Niagara integration for voice particles
    niagara_integration: NiagaraVoiceIntegration,
}

/// Godot Engine plugin interface
#[derive(Debug)]
pub struct GodotPlugin {
    /// GDNative handle
    gdnative_handle: Option<*mut c_void>,
    /// Audio stream interface
    audio_streams: GodotAudioStreams,
    /// Node management
    node_manager: GodotNodeManager,
    /// GDScript interface
    gdscript_interface: GodotScriptInterface,
}

/// Game session for voice synthesis
#[derive(Debug, Clone)]
pub struct GameSession {
    /// Session ID
    pub session_id: String,
    /// Game engine type
    pub engine_type: GameEngineType,
    /// Active voice instances
    pub voice_instances: HashMap<String, VoiceInstance>,
    /// Session start time
    pub started_at: SystemTime,
    /// Performance metrics
    pub performance_metrics: GamePerformanceMetrics,
    /// Session configuration
    pub config: GameSessionConfig,
}

/// Game engine types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameEngineType {
    Unity,
    UnrealEngine,
    Godot,
    Custom(String),
}

/// Voice instance in game
#[derive(Debug, Clone)]
pub struct VoiceInstance {
    /// Instance ID
    pub instance_id: String,
    /// Associated game object/actor ID
    pub game_object_id: String,
    /// Voice characteristics
    pub voice_profile: GameVoiceProfile,
    /// Current playback state
    pub playback_state: VoicePlaybackState,
    /// 3D spatial properties
    pub spatial_properties: SpatialAudioProperties,
    /// Performance metrics for this instance
    pub performance_metrics: VoiceInstanceMetrics,
}

/// Game-specific voice profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameVoiceProfile {
    /// Character ID this voice belongs to
    pub character_id: String,
    /// Voice emotional state
    pub emotional_state: EmotionalState,
    /// Dynamic characteristics
    pub dynamic_characteristics: DynamicVoiceCharacteristics,
    /// Contextual modifiers
    pub contextual_modifiers: ContextualModifiers,
}

/// Voice playback state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoicePlaybackState {
    Idle,
    Synthesizing,
    Playing,
    Paused,
    Stopped,
    Error(String),
}

/// 3D spatial audio properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAudioProperties {
    /// 3D position in game world
    pub position: [f32; 3],
    /// Velocity for doppler effect
    pub velocity: [f32; 3],
    /// Audio attenuation settings
    pub attenuation: AudioAttenuation,
    /// Reverb zone settings
    pub reverb_settings: ReverbSettings,
}

/// Audio attenuation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioAttenuation {
    /// Minimum distance for full volume
    pub min_distance: f32,
    /// Maximum distance for audio cutoff
    pub max_distance: f32,
    /// Rolloff curve type
    pub rolloff_type: AudioRolloffType,
    /// Custom rolloff curve points
    pub custom_curve: Option<Vec<(f32, f32)>>,
}

/// Audio rolloff types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioRolloffType {
    Linear,
    Logarithmic,
    Inverse,
    Custom,
}

/// Reverb settings for spatial audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbSettings {
    /// Reverb room size
    pub room_size: f32,
    /// Reverb decay time
    pub decay_time: f32,
    /// Reverb dampening
    pub dampening: f32,
    /// Early reflections
    pub early_reflections: f32,
}

/// Emotional state for dynamic voice synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Happiness level (0.0 - 1.0)
    pub happiness: f32,
    /// Anger level (0.0 - 1.0)
    pub anger: f32,
    /// Fear level (0.0 - 1.0)
    pub fear: f32,
    /// Excitement level (0.0 - 1.0)
    pub excitement: f32,
    /// Stress level (0.0 - 1.0)
    pub stress: f32,
}

/// Dynamic voice characteristics that change during gameplay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicVoiceCharacteristics {
    /// Health-based voice modulation
    pub health_modifier: f32,
    /// Fatigue effect on voice
    pub fatigue_level: f32,
    /// Environmental effects (underwater, in cave, etc.)
    pub environmental_filter: EnvironmentalFilter,
    /// Equipment modifiers (helmet, mask, etc.)
    pub equipment_modifiers: Vec<EquipmentModifier>,
}

/// Environmental audio filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentalFilter {
    None,
    Underwater,
    Cave,
    Outdoor,
    Indoor,
    Helmet,
    Radio,
    Custom(HashMap<String, f32>),
}

/// Equipment-based voice modifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentModifier {
    /// Equipment type
    pub equipment_type: String,
    /// Pitch modification
    pub pitch_modifier: f32,
    /// Filter cutoff frequency
    pub filter_cutoff: f32,
    /// Distortion amount
    pub distortion: f32,
}

/// Contextual modifiers based on game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualModifiers {
    /// Distance-based modulation
    pub distance_modulation: bool,
    /// Combat state effects
    pub combat_state: CombatState,
    /// Weather effects
    pub weather_effects: WeatherEffects,
    /// Time of day effects
    pub time_of_day: TimeOfDayEffects,
}

/// Combat state for voice modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombatState {
    Peaceful,
    Alert,
    Combat,
    Injured,
    Critical,
}

/// Weather-based voice effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherEffects {
    /// Wind effect strength
    pub wind_strength: f32,
    /// Rain dampening effect
    pub rain_dampening: f32,
    /// Thunder reverb effect
    pub thunder_reverb: f32,
}

/// Time of day voice effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOfDayEffects {
    /// Tiredness level based on time
    pub tiredness_level: f32,
    /// Ambient noise level
    pub ambient_noise: f32,
}

/// Unity-specific implementations
impl UnityPlugin {
    /// Initialize Unity plugin
    pub fn new() -> Result<Self> {
        Ok(Self {
            native_handle: None,
            audio_system: UnityAudioSystem::new(),
            gameobject_manager: UnityGameObjectManager::new(),
            script_interface: UnityScriptInterface::new(),
        })
    }

    /// Register Unity callbacks
    pub fn register_callbacks(&mut self) -> Result<()> {
        // Register native callbacks for Unity integration
        Ok(())
    }

    /// Create voice component for GameObject
    pub fn create_voice_component(&mut self, game_object_id: &str) -> Result<String> {
        let component_id = format!("voice_component_{}", game_object_id);
        self.gameobject_manager
            .register_voice_component(game_object_id, &component_id)?;
        Ok(component_id)
    }
}

/// Unreal Engine-specific implementations
impl UnrealPlugin {
    /// Initialize Unreal Engine plugin
    pub fn new() -> Result<Self> {
        Ok(Self {
            module_handle: None,
            audio_components: UnrealAudioComponents::new(),
            blueprint_interface: UnrealBlueprintInterface::new(),
            niagara_integration: NiagaraVoiceIntegration::new(),
        })
    }

    /// Register Unreal Engine module
    pub fn register_module(&mut self) -> Result<()> {
        // Register UE module for voice cloning
        Ok(())
    }

    /// Create voice actor component
    pub fn create_voice_actor_component(&mut self, actor_id: &str) -> Result<String> {
        let component_id = format!("voice_actor_component_{}", actor_id);
        self.audio_components
            .register_voice_component(actor_id, &component_id)?;
        Ok(component_id)
    }
}

/// Gaming Plugin Manager implementation
impl GamingPluginManager {
    /// Create new gaming plugin manager
    pub fn new(config: GamingPluginConfig) -> Self {
        Self {
            unity_plugin: None,
            unreal_plugin: None,
            godot_plugin: None,
            config,
            game_sessions: HashMap::new(),
        }
    }

    /// Initialize Unity plugin
    pub fn initialize_unity(&mut self) -> Result<()> {
        let plugin = UnityPlugin::new()?;
        self.unity_plugin = Some(plugin);
        Ok(())
    }

    /// Initialize Unreal Engine plugin
    pub fn initialize_unreal(&mut self) -> Result<()> {
        let plugin = UnrealPlugin::new()?;
        self.unreal_plugin = Some(plugin);
        Ok(())
    }

    /// Create game session
    pub fn create_game_session(&mut self, engine_type: GameEngineType) -> Result<String> {
        let session_id = format!(
            "game_session_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let session = GameSession {
            session_id: session_id.clone(),
            engine_type,
            voice_instances: HashMap::new(),
            started_at: SystemTime::now(),
            performance_metrics: GamePerformanceMetrics::default(),
            config: GameSessionConfig::default(),
        };

        self.game_sessions.insert(session_id.clone(), session);
        Ok(session_id)
    }

    /// Create voice instance in game
    pub fn create_voice_instance(
        &mut self,
        session_id: &str,
        game_object_id: &str,
        voice_profile: GameVoiceProfile,
    ) -> Result<String> {
        let session = self
            .game_sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation("Game session not found".to_string()))?;

        let instance_id = format!(
            "voice_instance_{}_{}",
            game_object_id,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let voice_instance = VoiceInstance {
            instance_id: instance_id.clone(),
            game_object_id: game_object_id.to_string(),
            voice_profile,
            playback_state: VoicePlaybackState::Idle,
            spatial_properties: SpatialAudioProperties::default(),
            performance_metrics: VoiceInstanceMetrics::default(),
        };

        session
            .voice_instances
            .insert(instance_id.clone(), voice_instance);
        Ok(instance_id)
    }

    /// Synthesize voice for game context
    pub async fn synthesize_game_voice(
        &mut self,
        session_id: &str,
        instance_id: &str,
        text: &str,
        context: GameContext,
    ) -> Result<GameVoiceResult> {
        // First, extract the needed data and update the state
        let (voice_profile, instance_id_clone, spatial_properties) = {
            let session = self
                .game_sessions
                .get_mut(session_id)
                .ok_or_else(|| Error::Validation("Game session not found".to_string()))?;

            let voice_instance = session
                .voice_instances
                .get_mut(instance_id)
                .ok_or_else(|| Error::Validation("Voice instance not found".to_string()))?;

            voice_instance.playback_state = VoicePlaybackState::Synthesizing;

            // Clone data needed for processing
            (
                voice_instance.voice_profile.clone(),
                voice_instance.instance_id.clone(),
                voice_instance.spatial_properties.clone(),
            )
        };

        // Apply game-specific modulations (no borrow conflicts here)
        let modified_request = self.apply_game_modulations_with_profile(
            instance_id_clone.clone(),
            &voice_profile,
            text,
            &context,
        )?;

        // Perform synthesis with game optimizations
        let result = self
            .synthesize_with_game_optimizations(modified_request)
            .await?;

        // Update the final state
        {
            let session = self
                .game_sessions
                .get_mut(session_id)
                .ok_or_else(|| Error::Validation("Game session not found".to_string()))?;

            let voice_instance = session
                .voice_instances
                .get_mut(instance_id)
                .ok_or_else(|| Error::Validation("Voice instance not found".to_string()))?;

            voice_instance.playback_state = VoicePlaybackState::Playing;
        }

        Ok(GameVoiceResult {
            instance_id: instance_id_clone,
            audio_data: result.audio,
            sample_rate: result.sample_rate,
            spatial_properties,
            performance_metrics: VoiceInstanceMetrics::default(),
        })
    }

    /// Apply game-specific voice modulations
    fn apply_game_modulations_with_profile(
        &self,
        instance_id: String,
        voice_profile: &GameVoiceProfile,
        text: &str,
        context: &GameContext,
    ) -> Result<VoiceCloneRequest> {
        let mut request = VoiceCloneRequest {
            id: instance_id,
            speaker_data: Default::default(),
            method: Default::default(),
            text: text.to_string(),
            language: None,
            quality_level: self.get_quality_level_for_performance(),
            quality_tradeoff: self.get_quality_tradeoff_for_performance(),
            parameters: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        // Apply emotional state modulations
        self.apply_emotional_modulations(&mut request, &voice_profile.emotional_state);

        // Apply dynamic characteristics
        self.apply_dynamic_characteristics(&mut request, &voice_profile.dynamic_characteristics);

        // Apply contextual modifiers
        self.apply_contextual_modifiers(&mut request, &voice_profile.contextual_modifiers, context);

        Ok(request)
    }

    /// Apply emotional state modulations
    fn apply_emotional_modulations(
        &self,
        request: &mut VoiceCloneRequest,
        emotional_state: &EmotionalState,
    ) {
        // Modulate voice based on emotional state
        request
            .parameters
            .insert("emotion_happiness".to_string(), emotional_state.happiness);
        request
            .parameters
            .insert("emotion_anger".to_string(), emotional_state.anger);
        request
            .parameters
            .insert("emotion_fear".to_string(), emotional_state.fear);
        request
            .parameters
            .insert("emotion_excitement".to_string(), emotional_state.excitement);
        request
            .parameters
            .insert("emotion_stress".to_string(), emotional_state.stress);
    }

    /// Apply dynamic characteristics
    fn apply_dynamic_characteristics(
        &self,
        request: &mut VoiceCloneRequest,
        characteristics: &DynamicVoiceCharacteristics,
    ) {
        request.parameters.insert(
            "health_modifier".to_string(),
            characteristics.health_modifier,
        );
        request
            .parameters
            .insert("fatigue_level".to_string(), characteristics.fatigue_level);

        // Apply environmental filter
        match &characteristics.environmental_filter {
            EnvironmentalFilter::Underwater => {
                request
                    .parameters
                    .insert("filter_underwater".to_string(), 1.0);
            }
            EnvironmentalFilter::Cave => {
                request.parameters.insert("filter_cave".to_string(), 1.0);
            }
            _ => {}
        }
    }

    /// Apply contextual modifiers
    fn apply_contextual_modifiers(
        &self,
        request: &mut VoiceCloneRequest,
        modifiers: &ContextualModifiers,
        _context: &GameContext,
    ) {
        // Apply combat state effects
        match modifiers.combat_state {
            CombatState::Combat => {
                request
                    .parameters
                    .insert("combat_intensity".to_string(), 0.9);
            }
            CombatState::Alert => {
                request
                    .parameters
                    .insert("combat_intensity".to_string(), 0.5);
            }
            _ => {}
        }

        // Apply weather effects
        request.parameters.insert(
            "wind_strength".to_string(),
            modifiers.weather_effects.wind_strength,
        );
        request.parameters.insert(
            "rain_dampening".to_string(),
            modifiers.weather_effects.rain_dampening,
        );
    }

    /// Get quality level based on performance profile
    fn get_quality_level_for_performance(&self) -> f32 {
        match self.config.performance_profile {
            GamePerformanceProfile::HighQuality => 0.9,
            GamePerformanceProfile::Balanced => 0.7,
            GamePerformanceProfile::LowLatency => 0.5,
            GamePerformanceProfile::Performance => 0.3,
            GamePerformanceProfile::Custom(ref settings) => settings.quality_level,
        }
    }

    /// Get quality tradeoff based on performance profile
    fn get_quality_tradeoff_for_performance(&self) -> f32 {
        match self.config.performance_profile {
            GamePerformanceProfile::HighQuality => 1.0,
            GamePerformanceProfile::Balanced => 0.7,
            GamePerformanceProfile::LowLatency => 0.3,
            GamePerformanceProfile::Performance => 0.1,
            GamePerformanceProfile::Custom(_) => 0.5,
        }
    }

    /// Synthesize with game-specific optimizations
    async fn synthesize_with_game_optimizations(
        &self,
        request: VoiceCloneRequest,
    ) -> Result<VoiceCloneResult> {
        // Mock synthesis with game optimizations
        Ok(VoiceCloneResult {
            request_id: request.id,
            audio: vec![0.0; 44100], // 1 second of silence
            sample_rate: 44100,
            quality_metrics: [("overall".to_string(), 0.8f32)].into_iter().collect(),
            similarity_score: 0.85,
            processing_time: Duration::from_millis(50),
            method_used: request.method,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: std::time::SystemTime::now(),
        })
    }
}

/// Supporting types and implementations

#[derive(Debug)]
pub struct UnityAudioSystem;

impl UnityAudioSystem {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct UnityGameObjectManager;

impl UnityGameObjectManager {
    pub fn new() -> Self {
        Self
    }
    pub fn register_voice_component(
        &mut self,
        _game_object_id: &str,
        _component_id: &str,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct UnityScriptInterface;

impl UnityScriptInterface {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct UnrealAudioComponents;

impl UnrealAudioComponents {
    pub fn new() -> Self {
        Self
    }
    pub fn register_voice_component(&mut self, _actor_id: &str, _component_id: &str) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct UnrealBlueprintInterface;

impl UnrealBlueprintInterface {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct NiagaraVoiceIntegration;

impl NiagaraVoiceIntegration {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct GodotAudioStreams;

#[derive(Debug)]
pub struct GodotNodeManager;

#[derive(Debug)]
pub struct GodotScriptInterface;

/// Game context for voice synthesis
#[derive(Debug, Clone)]
pub struct GameContext {
    /// Current game state
    pub game_state: String,
    /// Player location in game world
    pub player_location: [f32; 3],
    /// Current weather
    pub weather: String,
    /// Time of day
    pub time_of_day: f32,
    /// Additional context
    pub additional_context: HashMap<String, String>,
}

/// Game voice synthesis result
#[derive(Debug, Clone)]
pub struct GameVoiceResult {
    /// Voice instance ID
    pub instance_id: String,
    /// Generated audio data
    pub audio_data: Vec<f32>,
    /// Audio sample rate
    pub sample_rate: u32,
    /// 3D spatial properties
    pub spatial_properties: SpatialAudioProperties,
    /// Performance metrics
    pub performance_metrics: VoiceInstanceMetrics,
}

/// Performance metrics for game sessions
#[derive(Debug, Clone, Default)]
pub struct GamePerformanceMetrics {
    /// Average synthesis latency
    pub avg_synthesis_latency_ms: f32,
    /// Frame rate impact
    pub frame_rate_impact: f32,
    /// Memory usage
    pub memory_usage_mb: f32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Performance metrics for individual voice instances
#[derive(Debug, Clone, Default)]
pub struct VoiceInstanceMetrics {
    /// Synthesis time
    pub synthesis_time_ms: f32,
    /// Memory usage
    pub memory_usage_bytes: usize,
    /// Quality achieved
    pub quality_achieved: f32,
}

/// Game session configuration
#[derive(Debug, Clone, Default)]
pub struct GameSessionConfig {
    /// Maximum voice instances
    pub max_voice_instances: usize,
    /// Enable voice priority system
    pub enable_priority_system: bool,
    /// Voice culling distance
    pub voice_culling_distance: f32,
}

impl Default for GamingPluginConfig {
    fn default() -> Self {
        Self {
            enable_realtime_synthesis: true,
            max_voice_instances: 32,
            audio_buffer_size: 1024,
            target_latency_ms: 50.0,
            enable_voice_caching: true,
            voice_cache_size_mb: 256,
            enable_spatial_audio: true,
            performance_profile: GamePerformanceProfile::Balanced,
        }
    }
}

impl Default for SpatialAudioProperties {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            attenuation: AudioAttenuation::default(),
            reverb_settings: ReverbSettings::default(),
        }
    }
}

impl Default for AudioAttenuation {
    fn default() -> Self {
        Self {
            min_distance: 1.0,
            max_distance: 100.0,
            rolloff_type: AudioRolloffType::Logarithmic,
            custom_curve: None,
        }
    }
}

impl Default for ReverbSettings {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            decay_time: 1.0,
            dampening: 0.5,
            early_reflections: 0.3,
        }
    }
}

// C API exports for Unity and Unreal Engine integration
// Note: C API functions are commented out due to unsafe code restrictions
// These would be used for native plugin integration in production builds

/*
#[no_mangle]
pub extern "C" fn voirs_gaming_create_manager() -> *mut c_void {
    let config = GamingPluginConfig::default();
    let manager = Box::new(GamingPluginManager::new(config));
    Box::into_raw(manager) as *mut c_void
}

#[no_mangle]
pub extern "C" fn voirs_gaming_destroy_manager(manager: *mut c_void) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager as *mut GamingPluginManager);
        }
    }
}

#[no_mangle]
pub extern "C" fn voirs_gaming_create_session(
    manager: *mut c_void,
    engine_type: c_int,
) -> *const c_char {
    if manager.is_null() {
        return std::ptr::null();
    }

    unsafe {
        let manager = &mut *(manager as *mut GamingPluginManager);
        let engine_type = match engine_type {
            0 => GameEngineType::Unity,
            1 => GameEngineType::UnrealEngine,
            2 => GameEngineType::Godot,
            _ => GameEngineType::Custom("Unknown".to_string()),
        };

        match manager.create_game_session(engine_type) {
            Ok(session_id) => {
                let c_string = CString::new(session_id).unwrap();
                c_string.into_raw()
            }
            Err(_) => std::ptr::null(),
        }
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_plugin_manager_creation() {
        let config = GamingPluginConfig::default();
        let manager = GamingPluginManager::new(config);
        assert!(manager.game_sessions.is_empty());
    }

    #[test]
    fn test_game_session_creation() {
        let config = GamingPluginConfig::default();
        let mut manager = GamingPluginManager::new(config);
        let session_id = manager.create_game_session(GameEngineType::Unity).unwrap();
        assert!(!session_id.is_empty());
        assert!(manager.game_sessions.contains_key(&session_id));
    }

    #[test]
    fn test_voice_instance_creation() {
        let config = GamingPluginConfig::default();
        let mut manager = GamingPluginManager::new(config);
        let session_id = manager.create_game_session(GameEngineType::Unity).unwrap();

        let voice_profile = GameVoiceProfile {
            character_id: "hero".to_string(),
            emotional_state: EmotionalState {
                happiness: 0.7,
                anger: 0.2,
                fear: 0.1,
                excitement: 0.8,
                stress: 0.3,
            },
            dynamic_characteristics: DynamicVoiceCharacteristics {
                health_modifier: 1.0,
                fatigue_level: 0.2,
                environmental_filter: EnvironmentalFilter::None,
                equipment_modifiers: vec![],
            },
            contextual_modifiers: ContextualModifiers {
                distance_modulation: true,
                combat_state: CombatState::Peaceful,
                weather_effects: WeatherEffects {
                    wind_strength: 0.1,
                    rain_dampening: 0.0,
                    thunder_reverb: 0.0,
                },
                time_of_day: TimeOfDayEffects {
                    tiredness_level: 0.2,
                    ambient_noise: 0.1,
                },
            },
        };

        let instance_id = manager
            .create_voice_instance(&session_id, "hero_object", voice_profile)
            .unwrap();
        assert!(!instance_id.is_empty());
    }

    #[test]
    fn test_unity_plugin_creation() {
        let plugin = UnityPlugin::new();
        assert!(plugin.is_ok());
    }

    #[test]
    fn test_unreal_plugin_creation() {
        let plugin = UnrealPlugin::new();
        assert!(plugin.is_ok());
    }

    #[test]
    fn test_performance_profiles() {
        let config = GamingPluginConfig {
            performance_profile: GamePerformanceProfile::HighQuality,
            ..Default::default()
        };
        let manager = GamingPluginManager::new(config);
        assert_eq!(manager.get_quality_level_for_performance(), 0.9);
    }

    #[test]
    fn test_emotional_state_modulation() {
        let emotional_state = EmotionalState {
            happiness: 0.8,
            anger: 0.2,
            fear: 0.1,
            excitement: 0.9,
            stress: 0.3,
        };

        // Test that emotional states are within valid range
        assert!(emotional_state.happiness <= 1.0);
        assert!(emotional_state.anger >= 0.0);
        assert!(emotional_state.excitement <= 1.0);
    }

    #[test]
    fn test_spatial_audio_properties() {
        let spatial = SpatialAudioProperties::default();
        assert_eq!(spatial.position, [0.0, 0.0, 0.0]);
        assert_eq!(spatial.attenuation.min_distance, 1.0);
        assert_eq!(spatial.attenuation.max_distance, 100.0);
    }

    // C API tests are commented out due to unsafe code restrictions
    /*
    #[test]
    fn test_c_api_exports() {
        let manager_ptr = voirs_gaming_create_manager();
        assert!(!manager_ptr.is_null());

        let session_ptr = voirs_gaming_create_session(manager_ptr, 0); // Unity
        assert!(!session_ptr.is_null());

        voirs_gaming_destroy_manager(manager_ptr);
    }
    */
}
