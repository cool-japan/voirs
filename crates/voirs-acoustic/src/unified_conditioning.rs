//! Unified conditioning interface for all synthesis features.
//!
//! This module provides a single, consistent interface for all conditioning features
//! including emotion, speaker characteristics, prosody, and style controls.

use crate::conditioning::{ConditionalConfig, MultiFeatureConditionalNetwork};
use crate::parallel_attention::{EmotionAttentionConfig, EmotionAwareMultiHeadAttention};
use crate::prosody::{EmotionProsodyModifier, ProsodyConfig};
use crate::speaker::emotion::{
    EmotionConfig, EmotionPreprocessor, EmotionValidator, EmotionVector,
};
use crate::Result;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unified conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConditioningConfig {
    /// Emotion conditioning configuration
    pub emotion: EmotionConditioningConfig,
    /// Speaker conditioning configuration
    pub speaker: SpeakerConditioningConfig,
    /// Prosody conditioning configuration
    pub prosody: ProsodyConditioningConfig,
    /// Style conditioning configuration
    pub style: StyleConditioningConfig,
    /// Attention conditioning configuration
    pub attention: AttentionConditioningConfig,
    /// Neural conditioning configuration
    pub neural: NeuralConditioningConfig,
    /// Feature priorities (higher values = higher priority)
    pub feature_priorities: HashMap<String, f32>,
}

impl Default for UnifiedConditioningConfig {
    fn default() -> Self {
        let mut feature_priorities = HashMap::new();
        feature_priorities.insert("emotion".to_string(), 1.0);
        feature_priorities.insert("speaker".to_string(), 0.8);
        feature_priorities.insert("prosody".to_string(), 0.9);
        feature_priorities.insert("style".to_string(), 0.7);
        feature_priorities.insert("attention".to_string(), 0.6);

        Self {
            emotion: EmotionConditioningConfig::default(),
            speaker: SpeakerConditioningConfig::default(),
            prosody: ProsodyConditioningConfig::default(),
            style: StyleConditioningConfig::default(),
            attention: AttentionConditioningConfig::default(),
            neural: NeuralConditioningConfig::default(),
            feature_priorities,
        }
    }
}

/// Emotion-specific conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConditioningConfig {
    /// Enable emotion conditioning
    pub enabled: bool,
    /// Emotion vector dimension
    pub emotion_dim: usize,
    /// Emotion intensity scaling
    pub intensity_scale: f32,
    /// Emotion interpolation smoothing
    pub interpolation_smoothing: f32,
    /// Validate emotion parameters
    pub validate_parameters: bool,
}

impl Default for EmotionConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            emotion_dim: 256,
            intensity_scale: 1.0,
            interpolation_smoothing: 0.1,
            validate_parameters: true,
        }
    }
}

/// Speaker-specific conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConditioningConfig {
    /// Enable speaker conditioning
    pub enabled: bool,
    /// Speaker embedding dimension
    pub speaker_dim: usize,
    /// Speaker adaptation strength
    pub adaptation_strength: f32,
    /// Enable multi-speaker support
    pub multi_speaker: bool,
}

impl Default for SpeakerConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            speaker_dim: 256,
            adaptation_strength: 1.0,
            multi_speaker: true,
        }
    }
}

/// Prosody-specific conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConditioningConfig {
    /// Enable prosody conditioning
    pub enabled: bool,
    /// Prosody strength
    pub strength: f32,
    /// Enable emotion-aware prosody
    pub emotion_aware: bool,
    /// Prosody variation amount
    pub variation: f32,
}

impl Default for ProsodyConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 1.0,
            emotion_aware: true,
            variation: 0.1,
        }
    }
}

/// Style-specific conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConditioningConfig {
    /// Enable style conditioning
    pub enabled: bool,
    /// Style vector dimension
    pub style_dim: usize,
    /// Style transfer strength
    pub transfer_strength: f32,
    /// Enable style interpolation
    pub interpolation: bool,
}

impl Default for StyleConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            style_dim: 128,
            transfer_strength: 0.8,
            interpolation: true,
        }
    }
}

/// Attention-specific conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConditioningConfig {
    /// Enable attention conditioning
    pub enabled: bool,
    /// Attention conditioning strength
    pub strength: f32,
    /// Enable emotion-aware attention
    pub emotion_aware: bool,
    /// Attention head scaling
    pub head_scaling: bool,
}

impl Default for AttentionConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 1.0,
            emotion_aware: true,
            head_scaling: true,
        }
    }
}

/// Neural conditioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConditioningConfig {
    /// Enable neural conditioning layers
    pub enabled: bool,
    /// Hidden dimension for conditioning layers
    pub hidden_dim: usize,
    /// Number of conditioning layers
    pub num_layers: usize,
    /// Conditioning layer dropout
    pub dropout: f32,
    /// Enable residual connections
    pub residual: bool,
}

impl Default for NeuralConditioningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 512,
            num_layers: 2,
            dropout: 0.1,
            residual: true,
        }
    }
}

/// Unified conditioning state
#[derive(Debug, Clone)]
pub struct ConditioningState {
    /// Current emotion state
    pub emotion: Option<EmotionVector>,
    /// Current speaker embedding
    pub speaker: Option<Tensor>,
    /// Current prosody configuration
    pub prosody: Option<ProsodyConfig>,
    /// Current style vector
    pub style: Option<Tensor>,
    /// Feature strengths
    pub feature_strengths: HashMap<String, f32>,
    /// Timestamp for state updates
    pub timestamp: std::time::Instant,
}

impl Default for ConditioningState {
    fn default() -> Self {
        Self {
            emotion: None,
            speaker: None,
            prosody: None,
            style: None,
            feature_strengths: HashMap::new(),
            timestamp: std::time::Instant::now(),
        }
    }
}

/// Unified conditioning interface
pub struct UnifiedConditioning {
    /// Configuration
    config: UnifiedConditioningConfig,
    /// Emotion components
    emotion_validator: EmotionValidator,
    emotion_preprocessor: EmotionPreprocessor,
    emotion_prosody_modifier: EmotionProsodyModifier,
    /// Neural conditioning layers
    neural_conditioning: Option<MultiFeatureConditionalNetwork>,
    /// Emotion-aware attention
    emotion_attention: Option<EmotionAwareMultiHeadAttention>,
    /// Current conditioning state
    current_state: std::sync::Mutex<ConditioningState>,
    /// Advanced caching system
    #[allow(dead_code)]
    cache: ConditioningCache,
    /// Performance optimizer
    #[allow(dead_code)]
    performance_optimizer: PerformanceOptimizer,
    /// Device
    device: Device,
}

impl UnifiedConditioning {
    /// Create new unified conditioning interface
    pub fn new(config: UnifiedConditioningConfig, device: Device, vs: &VarBuilder) -> Result<Self> {
        // Initialize emotion components
        let emotion_validator = EmotionValidator::new();
        let emotion_preprocessor = EmotionPreprocessor::new();
        let emotion_prosody_modifier = EmotionProsodyModifier::new(ProsodyConfig::default());

        // Initialize neural conditioning if enabled
        let neural_conditioning = if config.neural.enabled {
            let neural_config = ConditionalConfig {
                hidden_dim: config.neural.hidden_dim,
                condition_dim: config.emotion.emotion_dim,
                num_layers: config.neural.num_layers,
                dropout: config.neural.dropout,
                residual: config.neural.residual,
                ..Default::default()
            };

            Some(MultiFeatureConditionalNetwork::new(
                neural_config,
                vec![
                    "emotion".to_string(),
                    "speaker".to_string(),
                    "style".to_string(),
                ],
                device.clone(),
                &vs.pp("neural_conditioning"),
            )?)
        } else {
            None
        };

        // Initialize emotion-aware attention if enabled
        let emotion_attention = if config.attention.enabled && config.attention.emotion_aware {
            let attention_config = EmotionAttentionConfig {
                emotion_dim: config.emotion.emotion_dim,
                emotion_head_scaling: config.attention.head_scaling,
                ..Default::default()
            };

            Some(EmotionAwareMultiHeadAttention::new(
                attention_config,
                device.clone(),
                &vs.pp("emotion_attention"),
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            emotion_validator,
            emotion_preprocessor,
            emotion_prosody_modifier,
            neural_conditioning,
            emotion_attention,
            current_state: std::sync::Mutex::new(ConditioningState::default()),
            cache: ConditioningCache::new(3600, 1000), // 1 hour expiration, 1000 max items
            performance_optimizer: PerformanceOptimizer::default(),
            device,
        })
    }

    /// Set emotion conditioning
    pub fn set_emotion(&self, emotion: EmotionConfig) -> Result<()> {
        // Validate emotion if enabled
        if self.config.emotion.validate_parameters {
            self.emotion_validator.validate(&emotion)?;
        }

        // Preprocess emotion
        let processed_emotion = self.emotion_preprocessor.preprocess_for_neural(&emotion)?;

        // Create emotion vector
        let emotion_vector = EmotionVector::new(processed_emotion, self.config.emotion.emotion_dim);

        // Update current state
        if let Ok(mut state) = self.current_state.lock() {
            state.emotion = Some(emotion_vector.clone());
            state.timestamp = std::time::Instant::now();

            // Update feature strength based on emotion intensity
            let intensity = emotion.intensity.as_f32() * self.config.emotion.intensity_scale;
            state
                .feature_strengths
                .insert("emotion".to_string(), intensity);
        }

        // Update emotion-aware attention if available
        if let Some(ref attention) = self.emotion_attention {
            attention.set_emotion(emotion_vector)?;
        }

        Ok(())
    }

    /// Set speaker conditioning
    pub fn set_speaker(&self, speaker_embedding: Tensor) -> Result<()> {
        if let Ok(mut state) = self.current_state.lock() {
            state.speaker = Some(speaker_embedding);
            state.timestamp = std::time::Instant::now();

            // Update feature strength
            let strength = self.config.speaker.adaptation_strength;
            state
                .feature_strengths
                .insert("speaker".to_string(), strength);
        }

        Ok(())
    }

    /// Set prosody conditioning
    pub fn set_prosody(&self, prosody: ProsodyConfig) -> Result<()> {
        if let Ok(mut state) = self.current_state.lock() {
            state.prosody = Some(prosody);
            state.timestamp = std::time::Instant::now();

            // Update feature strength
            let strength = self.config.prosody.strength;
            state
                .feature_strengths
                .insert("prosody".to_string(), strength);
        }

        Ok(())
    }

    /// Set style conditioning
    pub fn set_style(&self, style_vector: Tensor) -> Result<()> {
        if let Ok(mut state) = self.current_state.lock() {
            state.style = Some(style_vector);
            state.timestamp = std::time::Instant::now();

            // Update feature strength
            let strength = self.config.style.transfer_strength;
            state
                .feature_strengths
                .insert("style".to_string(), strength);
        }

        Ok(())
    }

    /// Apply unified conditioning to input tensor
    pub fn apply_conditioning(&self, input: &Tensor) -> CandleResult<Tensor> {
        let state = self.current_state.lock().unwrap();

        // Start with input tensor
        let mut conditioned = input.clone();

        // Apply neural conditioning if available
        if let Some(ref neural) = self.neural_conditioning {
            let mut feature_conditions = HashMap::new();

            // Add emotion conditioning
            if let Some(ref emotion) = state.emotion {
                let emotion_tensor =
                    Tensor::from_slice(emotion.as_slice(), emotion.dimension, &self.device)?;
                feature_conditions.insert("emotion".to_string(), emotion_tensor);
            }

            // Add speaker conditioning
            if let Some(ref speaker) = state.speaker {
                feature_conditions.insert("speaker".to_string(), speaker.clone());
            }

            // Add style conditioning
            if let Some(ref style) = state.style {
                feature_conditions.insert("style".to_string(), style.clone());
            }

            // Apply neural conditioning
            if !feature_conditions.is_empty() {
                conditioned = neural.forward(&conditioned, &feature_conditions)?;
            }
        }

        Ok(conditioned)
    }

    /// Apply emotion-aware attention
    pub fn apply_emotion_attention(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        if let Some(ref attention) = self.emotion_attention {
            attention.forward(input, attention_mask)
        } else {
            Ok(input.clone())
        }
    }

    /// Generate emotion-conditioned prosody
    pub fn generate_emotion_prosody(&mut self) -> Result<Option<ProsodyConfig>> {
        let state = self.current_state.lock().unwrap();

        if let Some(ref emotion) = state.emotion {
            let prosody = self
                .emotion_prosody_modifier
                .apply_emotion(emotion.config.clone())?;
            Ok(Some(prosody))
        } else {
            Ok(None)
        }
    }

    /// Get current conditioning state
    pub fn get_state(&self) -> ConditioningState {
        self.current_state.lock().unwrap().clone()
    }

    /// Clear all conditioning
    pub fn clear_conditioning(&self) {
        if let Ok(mut state) = self.current_state.lock() {
            *state = ConditioningState::default();
        }

        // Clear emotion attention if available
        if let Some(ref attention) = self.emotion_attention {
            attention.clear_emotion();
        }
    }

    /// Get feature priorities
    pub fn get_feature_priorities(&self) -> &HashMap<String, f32> {
        &self.config.feature_priorities
    }

    /// Set feature priority
    pub fn set_feature_priority(&mut self, feature: String, priority: f32) {
        self.config
            .feature_priorities
            .insert(feature, priority.clamp(0.0, 2.0));
    }

    /// Get conditioning statistics
    pub fn get_statistics(&self) -> ConditioningStatistics {
        let state = self.current_state.lock().unwrap();

        ConditioningStatistics {
            active_features: state.feature_strengths.len(),
            total_strength: state.feature_strengths.values().sum(),
            emotion_active: state.emotion.is_some(),
            speaker_active: state.speaker.is_some(),
            prosody_active: state.prosody.is_some(),
            style_active: state.style.is_some(),
            last_update: state.timestamp,
        }
    }

    /// Real-time parameter adjustment capabilities
    pub fn adjust_parameter_realtime(
        &mut self,
        parameter: RealtimeParameter,
        value: f32,
    ) -> Result<()> {
        match parameter {
            RealtimeParameter::EmotionIntensity => {
                self.config.emotion.intensity_scale = value.clamp(0.0, 2.0);
                // Update current state if emotion is active
                if let Ok(mut state) = self.current_state.lock() {
                    if let Some(ref mut emotion) = state.emotion {
                        let new_intensity =
                            emotion.config.intensity.as_f32() * self.config.emotion.intensity_scale;
                        state
                            .feature_strengths
                            .insert("emotion".to_string(), new_intensity);
                    }
                }
            }
            RealtimeParameter::SpeakerAdaptation => {
                self.config.speaker.adaptation_strength = value.clamp(0.0, 2.0);
                if let Ok(mut state) = self.current_state.lock() {
                    if state.speaker.is_some() {
                        state.feature_strengths.insert("speaker".to_string(), value);
                    }
                }
            }
            RealtimeParameter::ProsodyStrength => {
                self.config.prosody.strength = value.clamp(0.0, 2.0);
                if let Ok(mut state) = self.current_state.lock() {
                    if state.prosody.is_some() {
                        state.feature_strengths.insert("prosody".to_string(), value);
                    }
                }
            }
            RealtimeParameter::StyleTransfer => {
                self.config.style.transfer_strength = value.clamp(0.0, 2.0);
                if let Ok(mut state) = self.current_state.lock() {
                    if state.style.is_some() {
                        state.feature_strengths.insert("style".to_string(), value);
                    }
                }
            }
            RealtimeParameter::AttentionStrength => {
                self.config.attention.strength = value.clamp(0.0, 2.0);
            }
            RealtimeParameter::InterpolationSmoothing => {
                self.config.emotion.interpolation_smoothing = value.clamp(0.0, 1.0);
            }
        }

        // Update timestamp
        if let Ok(mut state) = self.current_state.lock() {
            state.timestamp = std::time::Instant::now();
        }

        Ok(())
    }

    /// Batch adjust multiple parameters for real-time control
    pub fn adjust_parameters_batch(
        &mut self,
        adjustments: &[(RealtimeParameter, f32)],
    ) -> Result<()> {
        for &(param, value) in adjustments {
            self.adjust_parameter_realtime(param, value)?;
        }
        Ok(())
    }

    /// Smooth parameter interpolation for real-time adjustment
    pub fn interpolate_parameter(
        &mut self,
        parameter: RealtimeParameter,
        target_value: f32,
        steps: usize,
    ) -> Result<Vec<f32>> {
        let current_value = match parameter {
            RealtimeParameter::EmotionIntensity => self.config.emotion.intensity_scale,
            RealtimeParameter::SpeakerAdaptation => self.config.speaker.adaptation_strength,
            RealtimeParameter::ProsodyStrength => self.config.prosody.strength,
            RealtimeParameter::StyleTransfer => self.config.style.transfer_strength,
            RealtimeParameter::AttentionStrength => self.config.attention.strength,
            RealtimeParameter::InterpolationSmoothing => {
                self.config.emotion.interpolation_smoothing
            }
        };

        let mut interpolation_values = Vec::with_capacity(steps);
        let step_size = (target_value - current_value) / steps as f32;

        for i in 0..steps {
            let interpolated_value = current_value + step_size * (i + 1) as f32;
            interpolation_values.push(interpolated_value);
        }

        Ok(interpolation_values)
    }

    /// Create conditioning preset for specific use case
    pub fn create_preset(preset_type: ConditioningPreset) -> UnifiedConditioningConfig {
        match preset_type {
            ConditioningPreset::Expressive => {
                let mut config = UnifiedConditioningConfig::default();
                config.emotion.intensity_scale = 1.2;
                config.prosody.strength = 1.1;
                config.attention.strength = 1.1;
                config.neural.num_layers = 3;
                config
            }
            ConditioningPreset::Natural => {
                let mut config = UnifiedConditioningConfig::default();
                config.emotion.intensity_scale = 0.8;
                config.prosody.variation = 0.15;
                config.style.transfer_strength = 0.6;
                config
            }
            ConditioningPreset::Subtle => {
                let mut config = UnifiedConditioningConfig::default();
                config.emotion.intensity_scale = 0.5;
                config.prosody.strength = 0.8;
                config.attention.strength = 0.7;
                config.neural.dropout = 0.2;
                config
            }
            ConditioningPreset::Dramatic => {
                let mut config = UnifiedConditioningConfig::default();
                config.emotion.intensity_scale = 1.5;
                config.prosody.strength = 1.3;
                config.attention.strength = 1.2;
                config.style.transfer_strength = 1.0;
                config
            }
        }
    }
}

/// Conditioning statistics
#[derive(Debug, Clone)]
pub struct ConditioningStatistics {
    /// Number of active features
    pub active_features: usize,
    /// Total conditioning strength
    pub total_strength: f32,
    /// Whether emotion is active
    pub emotion_active: bool,
    /// Whether speaker is active
    pub speaker_active: bool,
    /// Whether prosody is active
    pub prosody_active: bool,
    /// Whether style is active
    pub style_active: bool,
    /// Last update timestamp
    pub last_update: std::time::Instant,
}

/// Conditioning presets
#[derive(Debug, Clone, Copy)]
pub enum ConditioningPreset {
    /// Expressive synthesis with strong emotion
    Expressive,
    /// Natural synthesis with balanced features
    Natural,
    /// Subtle synthesis with light conditioning
    Subtle,
    /// Dramatic synthesis with strong conditioning
    Dramatic,
}

/// Real-time adjustable parameters
#[derive(Debug, Clone, Copy)]
pub enum RealtimeParameter {
    /// Emotion intensity scaling
    EmotionIntensity,
    /// Speaker adaptation strength
    SpeakerAdaptation,
    /// Prosody strength
    ProsodyStrength,
    /// Style transfer strength
    StyleTransfer,
    /// Attention conditioning strength
    AttentionStrength,
    /// Interpolation smoothing factor
    InterpolationSmoothing,
}

/// Performance optimizer for conditioning operations
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Optimization level (0.0 = quality, 1.0 = speed)
    optimization_level: f32,
    /// Memory usage tracking
    #[allow(dead_code)]
    memory_usage: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Processing time tracking
    processing_times: std::sync::RwLock<Vec<f64>>,
    /// Auto-optimization enabled
    #[allow(dead_code)]
    auto_optimize: bool,
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(optimization_level: f32) -> Self {
        Self {
            optimization_level: optimization_level.clamp(0.0, 1.0),
            memory_usage: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            processing_times: std::sync::RwLock::new(Vec::new()),
            auto_optimize: true,
        }
    }

    /// Get current optimization level
    pub fn optimization_level(&self) -> f32 {
        self.optimization_level
    }

    /// Set optimization level
    pub fn set_optimization_level(&mut self, level: f32) {
        self.optimization_level = level.clamp(0.0, 1.0);
    }

    /// Record processing time
    pub fn record_processing_time(&self, time_ms: f64) {
        if let Ok(mut times) = self.processing_times.write() {
            times.push(time_ms);
            // Keep only last 100 measurements
            if times.len() > 100 {
                times.remove(0);
            }
        }
    }

    /// Get average processing time
    pub fn average_processing_time(&self) -> f64 {
        if let Ok(times) = self.processing_times.read() {
            if times.is_empty() {
                0.0
            } else {
                times.iter().sum::<f64>() / times.len() as f64
            }
        } else {
            0.0
        }
    }
}

impl Default for PerformanceOptimizer {
    fn default() -> Self {
        Self::new(0.5) // Balanced optimization
    }
}

/// Advanced caching for conditioning features
#[derive(Debug)]
pub struct ConditioningCache {
    /// Emotion vector cache
    emotion_cache: std::sync::RwLock<HashMap<String, (EmotionVector, std::time::Instant)>>,
    /// Speaker embedding cache
    speaker_cache: std::sync::RwLock<HashMap<String, (Tensor, std::time::Instant)>>,
    /// Prosody configuration cache
    prosody_cache: std::sync::RwLock<HashMap<String, (ProsodyConfig, std::time::Instant)>>,
    /// Style vector cache
    style_cache: std::sync::RwLock<HashMap<String, (Tensor, std::time::Instant)>>,
    /// Cache expiration time (in seconds)
    cache_expiration: u64,
    /// Maximum cache size per feature type
    max_cache_size: usize,
}

impl ConditioningCache {
    /// Create new conditioning cache
    pub fn new(cache_expiration: u64, max_cache_size: usize) -> Self {
        Self {
            emotion_cache: std::sync::RwLock::new(HashMap::new()),
            speaker_cache: std::sync::RwLock::new(HashMap::new()),
            prosody_cache: std::sync::RwLock::new(HashMap::new()),
            style_cache: std::sync::RwLock::new(HashMap::new()),
            cache_expiration,
            max_cache_size,
        }
    }

    /// Cache emotion vector
    pub fn cache_emotion(&self, key: String, emotion: EmotionVector) -> Result<()> {
        let mut cache = self.emotion_cache.write().unwrap();

        // Remove expired entries
        let now = std::time::Instant::now();
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_secs() < self.cache_expiration
        });

        // Enforce cache size limit
        while cache.len() >= self.max_cache_size {
            // Remove oldest entry
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            } else {
                break;
            }
        }

        cache.insert(key, (emotion, now));
        Ok(())
    }

    /// Get cached emotion vector
    pub fn get_cached_emotion(&self, key: &str) -> Option<EmotionVector> {
        let cache = self.emotion_cache.read().unwrap();
        let now = std::time::Instant::now();

        if let Some((emotion, timestamp)) = cache.get(key) {
            if now.duration_since(*timestamp).as_secs() < self.cache_expiration {
                return Some(emotion.clone());
            }
        }

        None
    }

    /// Cache speaker embedding
    pub fn cache_speaker(&self, key: String, speaker: Tensor) -> Result<()> {
        let mut cache = self.speaker_cache.write().unwrap();

        // Remove expired entries
        let now = std::time::Instant::now();
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_secs() < self.cache_expiration
        });

        // Enforce cache size limit
        while cache.len() >= self.max_cache_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            } else {
                break;
            }
        }

        cache.insert(key, (speaker, now));
        Ok(())
    }

    /// Get cached speaker embedding
    pub fn get_cached_speaker(&self, key: &str) -> Option<Tensor> {
        let cache = self.speaker_cache.read().unwrap();
        let now = std::time::Instant::now();

        if let Some((speaker, timestamp)) = cache.get(key) {
            if now.duration_since(*timestamp).as_secs() < self.cache_expiration {
                return Some(speaker.clone());
            }
        }

        None
    }

    /// Cache prosody configuration
    pub fn cache_prosody(&self, key: String, prosody: ProsodyConfig) -> Result<()> {
        let mut cache = self.prosody_cache.write().unwrap();

        // Remove expired entries
        let now = std::time::Instant::now();
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_secs() < self.cache_expiration
        });

        // Enforce cache size limit
        while cache.len() >= self.max_cache_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            } else {
                break;
            }
        }

        cache.insert(key, (prosody, now));
        Ok(())
    }

    /// Get cached prosody configuration
    pub fn get_cached_prosody(&self, key: &str) -> Option<ProsodyConfig> {
        let cache = self.prosody_cache.read().unwrap();
        let now = std::time::Instant::now();

        if let Some((prosody, timestamp)) = cache.get(key) {
            if now.duration_since(*timestamp).as_secs() < self.cache_expiration {
                return Some(prosody.clone());
            }
        }

        None
    }

    /// Cache style vector
    pub fn cache_style(&self, key: String, style: Tensor) -> Result<()> {
        let mut cache = self.style_cache.write().unwrap();

        // Remove expired entries
        let now = std::time::Instant::now();
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_secs() < self.cache_expiration
        });

        // Enforce cache size limit
        while cache.len() >= self.max_cache_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            } else {
                break;
            }
        }

        cache.insert(key, (style, now));
        Ok(())
    }

    /// Get cached style vector
    pub fn get_cached_style(&self, key: &str) -> Option<Tensor> {
        let cache = self.style_cache.read().unwrap();
        let now = std::time::Instant::now();

        if let Some((style, timestamp)) = cache.get(key) {
            if now.duration_since(*timestamp).as_secs() < self.cache_expiration {
                return Some(style.clone());
            }
        }

        None
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.emotion_cache.write().unwrap().clear();
        self.speaker_cache.write().unwrap().clear();
        self.prosody_cache.write().unwrap().clear();
        self.style_cache.write().unwrap().clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let emotion_count = self.emotion_cache.read().unwrap().len();
        let speaker_count = self.speaker_cache.read().unwrap().len();
        let prosody_count = self.prosody_cache.read().unwrap().len();
        let style_count = self.style_cache.read().unwrap().len();

        CacheStatistics {
            emotion_entries: emotion_count,
            speaker_entries: speaker_count,
            prosody_entries: prosody_count,
            style_entries: style_count,
            total_entries: emotion_count + speaker_count + prosody_count + style_count,
            max_size_per_type: self.max_cache_size,
            expiration_time: self.cache_expiration,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Number of emotion cache entries
    pub emotion_entries: usize,
    /// Number of speaker cache entries
    pub speaker_entries: usize,
    /// Number of prosody cache entries
    pub prosody_entries: usize,
    /// Number of style cache entries
    pub style_entries: usize,
    /// Total cache entries
    pub total_entries: usize,
    /// Maximum cache size per feature type
    pub max_size_per_type: usize,
    /// Cache expiration time in seconds
    pub expiration_time: u64,
}

/// Unified conditioning builder for easy configuration
pub struct UnifiedConditioningBuilder {
    config: UnifiedConditioningConfig,
}

impl UnifiedConditioningBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: UnifiedConditioningConfig::default(),
        }
    }

    /// Enable/disable emotion conditioning
    pub fn with_emotion(mut self, enabled: bool) -> Self {
        self.config.emotion.enabled = enabled;
        self
    }

    /// Set emotion intensity scale
    pub fn with_emotion_intensity(mut self, scale: f32) -> Self {
        self.config.emotion.intensity_scale = scale;
        self
    }

    /// Enable/disable speaker conditioning
    pub fn with_speaker(mut self, enabled: bool) -> Self {
        self.config.speaker.enabled = enabled;
        self
    }

    /// Enable/disable prosody conditioning
    pub fn with_prosody(mut self, enabled: bool) -> Self {
        self.config.prosody.enabled = enabled;
        self
    }

    /// Enable/disable style conditioning
    pub fn with_style(mut self, enabled: bool) -> Self {
        self.config.style.enabled = enabled;
        self
    }

    /// Set neural conditioning layers
    pub fn with_neural_layers(mut self, num_layers: usize) -> Self {
        self.config.neural.num_layers = num_layers;
        self
    }

    /// Set feature priority
    pub fn with_feature_priority(mut self, feature: String, priority: f32) -> Self {
        self.config
            .feature_priorities
            .insert(feature, priority.clamp(0.0, 2.0));
        self
    }

    /// Build configuration
    pub fn build(self) -> UnifiedConditioningConfig {
        self.config
    }
}

impl Default for UnifiedConditioningBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Type alias for compatibility with test code
pub type UnifiedConditioningProcessor = UnifiedConditioningBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    #[test]
    fn test_unified_conditioning_config_default() {
        let config = UnifiedConditioningConfig::default();
        assert!(config.emotion.enabled);
        assert!(config.speaker.enabled);
        assert!(config.prosody.enabled);
        assert!(config.style.enabled);
        assert_eq!(config.feature_priorities.len(), 5);
    }

    #[test]
    fn test_conditioning_state_default() {
        let state = ConditioningState::default();
        assert!(state.emotion.is_none());
        assert!(state.speaker.is_none());
        assert!(state.prosody.is_none());
        assert!(state.style.is_none());
        assert_eq!(state.feature_strengths.len(), 0);
    }

    #[test]
    fn test_conditioning_presets() {
        let expressive = UnifiedConditioning::create_preset(ConditioningPreset::Expressive);
        assert_eq!(expressive.emotion.intensity_scale, 1.2);

        let natural = UnifiedConditioning::create_preset(ConditioningPreset::Natural);
        assert_eq!(natural.emotion.intensity_scale, 0.8);

        let subtle = UnifiedConditioning::create_preset(ConditioningPreset::Subtle);
        assert_eq!(subtle.emotion.intensity_scale, 0.5);

        let dramatic = UnifiedConditioning::create_preset(ConditioningPreset::Dramatic);
        assert_eq!(dramatic.emotion.intensity_scale, 1.5);
    }

    #[test]
    fn test_conditioning_builder() {
        let config = UnifiedConditioningBuilder::new()
            .with_emotion(true)
            .with_emotion_intensity(1.5)
            .with_speaker(false)
            .with_neural_layers(3)
            .with_feature_priority("emotion".to_string(), 1.5)
            .build();

        assert!(config.emotion.enabled);
        assert_eq!(config.emotion.intensity_scale, 1.5);
        assert!(!config.speaker.enabled);
        assert_eq!(config.neural.num_layers, 3);
        assert_eq!(config.feature_priorities.get("emotion"), Some(&1.5));
    }

    #[tokio::test]
    async fn test_unified_conditioning_creation() {
        let device = Device::Cpu;
        let vs = VarBuilder::zeros(candle_core::DType::F32, &device);
        let config = UnifiedConditioningConfig::default();

        let conditioning = UnifiedConditioning::new(config, device, &vs);
        assert!(conditioning.is_ok());
    }

    #[test]
    fn test_conditioning_statistics() {
        let stats = ConditioningStatistics {
            active_features: 3,
            total_strength: 2.5,
            emotion_active: true,
            speaker_active: true,
            prosody_active: false,
            style_active: true,
            last_update: std::time::Instant::now(),
        };

        assert_eq!(stats.active_features, 3);
        assert_eq!(stats.total_strength, 2.5);
        assert!(stats.emotion_active);
        assert!(stats.speaker_active);
        assert!(!stats.prosody_active);
        assert!(stats.style_active);
    }
}
