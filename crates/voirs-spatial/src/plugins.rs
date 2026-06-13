//! Plugin System for VoiRS Spatial Audio
//!
//! This module provides an extensible plugin architecture for the spatial audio system,
//! allowing third-party developers to create custom spatial processing effects,
//! HRTF implementations, room simulation algorithms, and audio processing pipelines.

use crate::types::{AudioChannel, BinauraAudio, Position3D};
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Plugin trait that all spatial audio plugins must implement
#[async_trait]
pub trait SpatialPlugin: Send + Sync + Debug {
    /// Get the plugin name
    fn name(&self) -> &str;

    /// Get the plugin version
    fn version(&self) -> &str;

    /// Get the plugin description
    fn description(&self) -> &str;

    /// Get the plugin author
    fn author(&self) -> &str;

    /// Get the plugin capabilities
    fn capabilities(&self) -> PluginCapabilities;

    /// Initialize the plugin
    async fn initialize(&mut self, config: PluginConfig) -> Result<()>;

    /// Process audio with spatial effects
    async fn process_audio(
        &self,
        audio: &[f32],
        listener_position: Position3D,
        source_position: Position3D,
        context: &ProcessingContext,
    ) -> Result<Vec<f32>>;

    /// Process binaural audio
    async fn process_binaural(
        &self,
        audio: &BinauraAudio,
        context: &ProcessingContext,
    ) -> Result<BinauraAudio> {
        // Default implementation just returns the input unchanged
        Ok(audio.clone())
    }

    /// Update plugin parameters
    async fn update_parameters(&mut self, parameters: PluginParameters) -> Result<()>;

    /// Get current plugin state
    fn get_state(&self) -> PluginState;

    /// Cleanup plugin resources
    async fn cleanup(&mut self) -> Result<()>;

    /// Cast to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Cast to mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Plugin capabilities bitmask
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Can process mono audio
    pub supports_mono: bool,
    /// Can process stereo audio
    pub supports_stereo: bool,
    /// Can process multi-channel audio
    pub supports_multichannel: bool,
    /// Can process binaural audio
    pub supports_binaural: bool,
    /// Can process real-time streams
    pub supports_realtime: bool,
    /// Can process batch audio
    pub supports_batch: bool,
    /// Has configurable parameters
    pub has_parameters: bool,
    /// Supports state serialization
    pub supports_serialization: bool,
    /// Requires GPU acceleration
    pub requires_gpu: bool,
    /// Supports 3D positioning
    pub supports_3d_positioning: bool,
    /// Supports HRTF processing
    pub supports_hrtf: bool,
    /// Supports room simulation
    pub supports_room_simulation: bool,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            supports_mono: true,
            supports_stereo: true,
            supports_multichannel: false,
            supports_binaural: false,
            supports_realtime: true,
            supports_batch: true,
            has_parameters: false,
            supports_serialization: false,
            requires_gpu: false,
            supports_3d_positioning: false,
            supports_hrtf: false,
            supports_room_simulation: false,
        }
    }
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin-specific configuration parameters
    pub parameters: HashMap<String, PluginParameter>,
    /// Sample rate for audio processing
    pub sample_rate: f32,
    /// Buffer size for audio processing
    pub buffer_size: usize,
    /// Number of audio channels
    pub channels: usize,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
    /// Real-time processing mode
    pub realtime_mode: bool,
    /// Quality level (0.0 = lowest, 1.0 = highest)
    pub quality_level: f32,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            sample_rate: 44100.0,
            buffer_size: 1024,
            channels: 2,
            use_gpu: false,
            realtime_mode: true,
            quality_level: 0.8,
        }
    }
}

/// Plugin parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginParameter {
    /// Boolean parameter
    Bool(bool),
    /// Integer parameter
    Int(i32),
    /// Float parameter
    Float(f32),
    /// String parameter
    String(String),
    /// Array of floats
    FloatArray(Vec<f32>),
    /// Nested parameters
    Object(HashMap<String, PluginParameter>),
}

/// Plugin parameters collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginParameters {
    /// Parameter map
    pub parameters: HashMap<String, PluginParameter>,
}

impl PluginParameters {
    /// Create new empty parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a boolean parameter
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.parameters
            .insert(key.to_string(), PluginParameter::Bool(value));
    }

    /// Set an integer parameter
    pub fn set_int(&mut self, key: &str, value: i32) {
        self.parameters
            .insert(key.to_string(), PluginParameter::Int(value));
    }

    /// Set a float parameter
    pub fn set_float(&mut self, key: &str, value: f32) {
        self.parameters
            .insert(key.to_string(), PluginParameter::Float(value));
    }

    /// Set a string parameter
    pub fn set_string(&mut self, key: &str, value: String) {
        self.parameters
            .insert(key.to_string(), PluginParameter::String(value));
    }

    /// Get a boolean parameter
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.parameters.get(key)? {
            PluginParameter::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Get an integer parameter
    pub fn get_int(&self, key: &str) -> Option<i32> {
        match self.parameters.get(key)? {
            PluginParameter::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// Get a float parameter
    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.parameters.get(key)? {
            PluginParameter::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Get a string parameter
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.parameters.get(key)? {
            PluginParameter::String(value) => Some(value),
            _ => None,
        }
    }
}

/// Processing context for plugins
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Current sample rate
    pub sample_rate: f32,
    /// Current buffer size
    pub buffer_size: usize,
    /// Number of channels
    pub channels: usize,
    /// Processing timestamp
    pub timestamp: std::time::Instant,
    /// Quality level (0.0 = lowest, 1.0 = highest)
    pub quality_level: f32,
    /// Real-time processing mode
    pub realtime_mode: bool,
    /// Additional context data
    pub context_data: HashMap<String, PluginParameter>,
}

impl Default for ProcessingContext {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            buffer_size: 1024,
            channels: 2,
            timestamp: std::time::Instant::now(),
            quality_level: 0.8,
            realtime_mode: true,
            context_data: HashMap::new(),
        }
    }
}

/// Plugin state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginState {
    /// Plugin is uninitialized
    Uninitialized,
    /// Plugin is initialized and ready
    Ready,
    /// Plugin is currently processing
    Processing,
    /// Plugin is paused
    Paused,
    /// Plugin has an error
    Error(String),
    /// Plugin is being cleaned up
    Cleanup,
}

/// Plugin manager for loading and managing spatial audio plugins
#[derive(Debug)]
pub struct PluginManager {
    /// Loaded plugins
    plugins: Arc<RwLock<HashMap<String, Box<dyn SpatialPlugin>>>>,
    /// Plugin configurations
    configs: Arc<RwLock<HashMap<String, PluginConfig>>>,
    /// Processing chains
    chains: Arc<RwLock<HashMap<String, ProcessingChain>>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            chains: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a plugin
    pub async fn register_plugin(
        &self,
        plugin: Box<dyn SpatialPlugin>,
        config: PluginConfig,
    ) -> Result<()> {
        let name = plugin.name().to_string();

        // Store plugin configuration
        {
            let mut configs = self
                .configs
                .write()
                .map_err(|_| Error::LegacyAudio("Plugin config lock poisoned".to_string()))?;
            configs.insert(name.clone(), config);
        }

        // Store plugin
        {
            let mut plugins = self
                .plugins
                .write()
                .map_err(|_| Error::LegacyAudio("Plugin lock poisoned".to_string()))?;
            plugins.insert(name, plugin);
        }

        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, name: &str) -> Result<()> {
        // Cleanup plugin
        let plugin_to_cleanup = {
            let mut plugins = self
                .plugins
                .write()
                .map_err(|_| Error::LegacyAudio("Plugin lock poisoned".to_string()))?;
            plugins.remove(name)
        };

        if let Some(mut plugin) = plugin_to_cleanup {
            plugin.cleanup().await?;
        }

        // Remove configuration
        {
            let mut configs = self
                .configs
                .write()
                .map_err(|_| Error::LegacyAudio("Plugin config lock poisoned".to_string()))?;
            configs.remove(name);
        }

        Ok(())
    }

    /// Get plugin names
    pub fn get_plugin_names(&self) -> Vec<String> {
        match self.plugins.read() {
            Ok(plugins) => plugins.keys().cloned().collect(),
            Err(_) => {
                tracing::warn!("Plugin lock poisoned, returning empty list");
                Vec::new()
            }
        }
    }

    /// Check if plugin exists
    pub fn has_plugin(&self, name: &str) -> bool {
        match self.plugins.read() {
            Ok(plugins) => plugins.contains_key(name),
            Err(_) => {
                tracing::warn!("Plugin lock poisoned, assuming plugin doesn't exist");
                false
            }
        }
    }

    /// Process audio through a specific plugin
    #[allow(clippy::await_holding_lock)]
    pub async fn process_with_plugin(
        &self,
        plugin_name: &str,
        audio: &[f32],
        listener_position: Position3D,
        source_position: Position3D,
        context: &ProcessingContext,
    ) -> Result<Vec<f32>> {
        let plugins = self
            .plugins
            .read()
            .map_err(|_| Error::LegacyAudio("Plugin lock poisoned".to_string()))?;
        let plugin = plugins
            .get(plugin_name)
            .ok_or_else(|| Error::LegacyAudio(format!("Plugin {plugin_name} not found")))?;

        plugin
            .process_audio(audio, listener_position, source_position, context)
            .await
    }

    /// Process audio through a processing chain
    #[allow(clippy::await_holding_lock)]
    pub async fn process_with_chain(
        &self,
        chain_name: &str,
        audio: &[f32],
        listener_position: Position3D,
        source_position: Position3D,
        context: &ProcessingContext,
    ) -> Result<Vec<f32>> {
        let chains = self
            .chains
            .read()
            .map_err(|_| Error::LegacyAudio("Chain lock poisoned".to_string()))?;
        let chain = chains.get(chain_name).ok_or_else(|| {
            Error::LegacyAudio(format!("Processing chain {chain_name} not found"))
        })?;

        self.process_chain(chain, audio, listener_position, source_position, context)
            .await
    }

    /// Create a processing chain
    pub async fn create_chain(&self, name: &str, plugin_names: Vec<String>) -> Result<()> {
        let chain = ProcessingChain {
            name: name.to_string(),
            plugins: plugin_names,
            enabled: true,
        };

        let mut chains = self
            .chains
            .write()
            .map_err(|_| Error::LegacyAudio("Chain lock poisoned".to_string()))?;
        chains.insert(name.to_string(), chain);

        Ok(())
    }

    /// Remove a processing chain
    pub async fn remove_chain(&self, name: &str) -> Result<()> {
        let mut chains = self
            .chains
            .write()
            .map_err(|_| Error::LegacyAudio("Chain lock poisoned".to_string()))?;
        chains.remove(name);
        Ok(())
    }

    /// Process through a processing chain
    async fn process_chain(
        &self,
        chain: &ProcessingChain,
        mut audio: &[f32],
        listener_position: Position3D,
        source_position: Position3D,
        context: &ProcessingContext,
    ) -> Result<Vec<f32>> {
        if !chain.enabled {
            return Ok(audio.to_vec());
        }

        let mut result = audio.to_vec();

        for plugin_name in &chain.plugins {
            result = self
                .process_with_plugin(
                    plugin_name,
                    &result,
                    listener_position,
                    source_position,
                    context,
                )
                .await?;
        }

        Ok(result)
    }

    /// Get plugin capabilities
    pub fn get_plugin_capabilities(&self, name: &str) -> Option<PluginCapabilities> {
        match self.plugins.read() {
            Ok(plugins) => plugins.get(name).map(|plugin| plugin.capabilities()),
            Err(_) => {
                tracing::warn!("Plugin lock poisoned, returning None for capabilities");
                None
            }
        }
    }

    /// Update plugin parameters
    #[allow(clippy::await_holding_lock)]
    pub async fn update_plugin_parameters(
        &self,
        plugin_name: &str,
        parameters: PluginParameters,
    ) -> Result<()> {
        let mut plugins = self
            .plugins
            .write()
            .map_err(|_| Error::LegacyAudio("Plugin lock poisoned".to_string()))?;
        let plugin = plugins
            .get_mut(plugin_name)
            .ok_or_else(|| Error::LegacyAudio(format!("Plugin {plugin_name} not found")))?;

        plugin.update_parameters(parameters).await
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing chain definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingChain {
    /// Chain name
    pub name: String,
    /// Ordered list of plugin names
    pub plugins: Vec<String>,
    /// Whether the chain is enabled
    pub enabled: bool,
}

// ---------------------------------------------------------------------------
// Freeverb-style algorithmic reverberator
// ---------------------------------------------------------------------------

/// Comb-filter delay-line tunings (samples at 44.1 kHz) from the classic Freeverb.
const FREEVERB_COMB_TUNINGS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
/// All-pass delay-line tunings (samples at 44.1 kHz).
const FREEVERB_ALLPASS_TUNINGS: [usize; 4] = [556, 441, 341, 225];
/// Sample rate the tunings above were chosen for.
const FREEVERB_REFERENCE_SAMPLE_RATE: f32 = 44_100.0;
/// Input gain applied before the comb bank to keep the wet path well-conditioned.
const FREEVERB_FIXED_GAIN: f32 = 0.015;
/// Maps `room_size` ∈ [0, 1] onto comb feedback ∈ [0.7, 0.98] (always < 1 → stable).
const FREEVERB_ROOM_SCALE: f32 = 0.28;
/// Feedback offset paired with [`FREEVERB_ROOM_SCALE`].
const FREEVERB_ROOM_OFFSET: f32 = 0.7;
/// Maps `damping` ∈ [0, 1] onto the comb low-pass coefficient ∈ [0, 0.4].
const FREEVERB_DAMP_SCALE: f32 = 0.4;
/// Fixed all-pass feedback coefficient.
const FREEVERB_ALLPASS_FEEDBACK: f32 = 0.5;

/// Low-pass-damped feedback comb filter (one of Freeverb's parallel combs).
#[derive(Debug, Clone)]
struct FreeverbComb {
    /// Ring buffer holding the delay line.
    buffer: Vec<f32>,
    /// Current read/write position in the ring buffer.
    index: usize,
    /// One-pole low-pass state in the feedback path.
    filter_store: f32,
    /// Recirculation gain (< 1 for stability).
    feedback: f32,
    /// Low-pass coefficient on the delayed sample.
    damp1: f32,
    /// Low-pass coefficient on the previous filter state (`1 - damp1`).
    damp2: f32,
}

impl FreeverbComb {
    fn new(size: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            index: 0,
            filter_store: 0.0,
            feedback,
            damp1: damp,
            damp2: 1.0 - damp,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];
        // One-pole low-pass in the feedback path → frequency-dependent decay.
        self.filter_store = output * self.damp2 + self.filter_store * self.damp1;
        self.buffer[self.index] = input + self.filter_store * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

/// Schroeder all-pass section used to build echo density on the output path.
#[derive(Debug, Clone)]
struct FreeverbAllpass {
    /// Ring buffer holding the delay line.
    buffer: Vec<f32>,
    /// Current read/write position in the ring buffer.
    index: usize,
    /// Recirculation gain.
    feedback: f32,
}

impl FreeverbAllpass {
    fn new(size: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            index: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buffer[self.index];
        let output = buffered - input;
        self.buffer[self.index] = input + buffered * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

/// Compact mono Freeverb: eight parallel damped comb filters summed into four
/// series all-pass sections. Feedback is held strictly below 1, so the impulse
/// response is finite-energy and always decays.
#[derive(Debug, Clone)]
struct Freeverb {
    /// Parallel comb-filter bank.
    combs: Vec<FreeverbComb>,
    /// Series all-pass diffusion chain.
    allpasses: Vec<FreeverbAllpass>,
}

impl Freeverb {
    fn new(room_size: f32, damping: f32, sample_rate: f32) -> Self {
        let feedback = room_size.clamp(0.0, 1.0) * FREEVERB_ROOM_SCALE + FREEVERB_ROOM_OFFSET;
        let damp = damping.clamp(0.0, 1.0) * FREEVERB_DAMP_SCALE;
        // Scale the reference tunings to the actual sample rate.
        let sr_scale = (sample_rate / FREEVERB_REFERENCE_SAMPLE_RATE).max(0.1);

        let combs = FREEVERB_COMB_TUNINGS
            .iter()
            .map(|&tuning| {
                let size = (tuning as f32 * sr_scale).round() as usize;
                FreeverbComb::new(size, feedback, damp)
            })
            .collect();
        let allpasses = FREEVERB_ALLPASS_TUNINGS
            .iter()
            .map(|&tuning| {
                let size = (tuning as f32 * sr_scale).round() as usize;
                FreeverbAllpass::new(size, FREEVERB_ALLPASS_FEEDBACK)
            })
            .collect();

        Self { combs, allpasses }
    }

    /// Process one input sample and return the wet (reverberated) sample.
    fn process(&mut self, input: f32) -> f32 {
        let scaled = input * FREEVERB_FIXED_GAIN;
        // Parallel comb bank.
        let mut wet = 0.0f32;
        for comb in &mut self.combs {
            wet += comb.process(scaled);
        }
        // Series all-pass diffusion.
        for allpass in &mut self.allpasses {
            wet = allpass.process(wet);
        }
        wet
    }
}

/// Example reverb plugin implementation
#[derive(Debug)]
pub struct ReverbPlugin {
    name: String,
    version: String,
    room_size: f32,
    damping: f32,
    wet_level: f32,
    dry_level: f32,
    state: PluginState,
}

impl Default for ReverbPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ReverbPlugin {
    /// Create a new reverb plugin
    pub fn new() -> Self {
        Self {
            name: "Spatial Reverb".to_string(),
            version: "1.0.0".to_string(),
            room_size: 0.5,
            damping: 0.5,
            wet_level: 0.3,
            dry_level: 0.7,
            state: PluginState::Uninitialized,
        }
    }
}

#[async_trait]
impl SpatialPlugin for ReverbPlugin {
    fn name(&self) -> &str {
        &self.name
    }
    fn version(&self) -> &str {
        &self.version
    }
    fn description(&self) -> &str {
        "Spatial reverb effect for room simulation"
    }
    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            supports_mono: true,
            supports_stereo: true,
            supports_multichannel: true,
            supports_binaural: true,
            supports_realtime: true,
            supports_batch: true,
            has_parameters: true,
            supports_serialization: true,
            requires_gpu: false,
            supports_3d_positioning: true,
            supports_hrtf: false,
            supports_room_simulation: true,
        }
    }

    async fn initialize(&mut self, config: PluginConfig) -> Result<()> {
        // Initialize reverb parameters from config
        if let Some(PluginParameter::Float(size)) = config.parameters.get("room_size") {
            self.room_size = *size;
        }
        if let Some(PluginParameter::Float(damping)) = config.parameters.get("damping") {
            self.damping = *damping;
        }
        if let Some(PluginParameter::Float(wet)) = config.parameters.get("wet_level") {
            self.wet_level = *wet;
        }
        if let Some(PluginParameter::Float(dry)) = config.parameters.get("dry_level") {
            self.dry_level = *dry;
        }

        self.state = PluginState::Ready;
        Ok(())
    }

    async fn process_audio(
        &self,
        audio: &[f32],
        listener_position: Position3D,
        source_position: Position3D,
        context: &ProcessingContext,
    ) -> Result<Vec<f32>> {
        if matches!(self.state, PluginState::Error(_)) {
            return Err(Error::LegacyAudio("Plugin is in error state".to_string()));
        }

        // Reverb send rises with distance (more reflected vs. direct energy).
        let distance = listener_position.distance_to(&source_position);
        let reverb_scale = (distance / 10.0).min(1.0);

        // A Freeverb tank parameterised by the plugin's room size and damping.
        // It is instantiated per call because `process_audio` borrows `&self`; the
        // tank carries no cross-buffer state in this example plugin.
        let mut reverb = Freeverb::new(self.room_size, self.damping, context.sample_rate);

        let mut output = Vec::with_capacity(audio.len());
        for &sample in audio {
            let wet = reverb.process(sample);
            let dry = sample * self.dry_level;
            output.push(dry + wet * self.wet_level * reverb_scale);
        }

        Ok(output)
    }

    async fn update_parameters(&mut self, parameters: PluginParameters) -> Result<()> {
        if let Some(size) = parameters.get_float("room_size") {
            self.room_size = size.clamp(0.0, 1.0);
        }
        if let Some(damping) = parameters.get_float("damping") {
            self.damping = damping.clamp(0.0, 1.0);
        }
        if let Some(wet) = parameters.get_float("wet_level") {
            self.wet_level = wet.clamp(0.0, 1.0);
        }
        if let Some(dry) = parameters.get_float("dry_level") {
            self.dry_level = dry.clamp(0.0, 1.0);
        }
        Ok(())
    }

    fn get_state(&self) -> PluginState {
        self.state.clone()
    }

    async fn cleanup(&mut self) -> Result<()> {
        self.state = PluginState::Cleanup;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.get_plugin_names().len(), 0);
    }

    #[test]
    async fn test_plugin_registration() {
        let manager = PluginManager::new();
        let plugin = Box::new(ReverbPlugin::new());
        let config = PluginConfig::default();

        manager.register_plugin(plugin, config).await.unwrap();
        assert_eq!(manager.get_plugin_names().len(), 1);
        assert!(manager.has_plugin("Spatial Reverb"));
    }

    #[test]
    async fn test_plugin_capabilities() {
        let plugin = ReverbPlugin::new();
        let caps = plugin.capabilities();

        assert!(caps.supports_mono);
        assert!(caps.supports_stereo);
        assert!(caps.supports_room_simulation);
        assert!(caps.has_parameters);
    }

    #[test]
    async fn test_plugin_parameters() {
        let mut params = PluginParameters::new();
        params.set_float("room_size", 0.8);
        params.set_bool("enabled", true);
        params.set_string("preset", "Hall".to_string());

        assert_eq!(params.get_float("room_size"), Some(0.8));
        assert_eq!(params.get_bool("enabled"), Some(true));
        assert_eq!(params.get_string("preset"), Some("Hall"));
    }

    #[test]
    async fn test_plugin_audio_processing() {
        let mut plugin = ReverbPlugin::new();
        let config = PluginConfig::default();
        plugin.initialize(config).await.unwrap();

        let audio = vec![0.5; 1024];
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        let source_pos = Position3D::new(1.0, 0.0, 0.0);
        let context = ProcessingContext::default();

        let result = plugin
            .process_audio(&audio, listener_pos, source_pos, &context)
            .await
            .unwrap();
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    async fn test_processing_chain() {
        let manager = PluginManager::new();

        // Register a plugin
        let plugin = Box::new(ReverbPlugin::new());
        let config = PluginConfig::default();
        manager.register_plugin(plugin, config).await.unwrap();

        // Create a processing chain
        manager
            .create_chain("test_chain", vec!["Spatial Reverb".to_string()])
            .await
            .unwrap();

        // Process audio through the chain
        let audio = vec![0.5; 1024];
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        let source_pos = Position3D::new(1.0, 0.0, 0.0);
        let context = ProcessingContext::default();

        let result = manager
            .process_with_chain("test_chain", &audio, listener_pos, source_pos, &context)
            .await
            .unwrap();
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    async fn test_plugin_cleanup() {
        let manager = PluginManager::new();
        let plugin = Box::new(ReverbPlugin::new());
        let config = PluginConfig::default();

        manager.register_plugin(plugin, config).await.unwrap();
        assert!(manager.has_plugin("Spatial Reverb"));

        manager.unregister_plugin("Spatial Reverb").await.unwrap();
        assert!(!manager.has_plugin("Spatial Reverb"));
    }

    #[test]
    async fn test_reverb_impulse_produces_decaying_bounded_tail() {
        let mut plugin = ReverbPlugin::new();
        plugin.initialize(PluginConfig::default()).await.unwrap();

        let len = 30_000;
        let mut audio = vec![0.0f32; len];
        audio[0] = 1.0; // unit impulse

        let listener = Position3D::new(0.0, 0.0, 0.0);
        // Far source → reverb send saturates (distance ≥ 10 gives full wet level).
        let source = Position3D::new(12.0, 0.0, 0.0);
        let context = ProcessingContext::default();

        let output = plugin
            .process_audio(&audio, listener, source, &context)
            .await
            .unwrap();
        assert_eq!(output.len(), len);

        // Every sample must be finite and bounded (feedback < 1 guarantees this).
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "non-finite output at {i}");
            assert!(s.abs() < 4.0, "reverb blew up at {i}: {s}");
        }

        // A reverberant tail must exist beyond the dry impulse at index 0.
        let tail_energy: f32 = output[1..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.0, "no reverb tail: {tail_energy}");

        // The tail must decay: an early window holds more energy than a later one.
        let window_energy = |start: usize, end: usize| -> f32 {
            output[start..end.min(len)].iter().map(|s| s * s).sum()
        };
        let early = window_energy(2_000, 7_000);
        let late = window_energy(22_000, 27_000);
        assert!(early > 0.0, "no early tail energy: {early}");
        assert!(
            early > late,
            "tail should decay: early={early}, late={late}"
        );
    }
}
