//! # Haptic Integration for Spatial Audio
//!
//! This module provides tactile feedback integration with spatial audio systems,
//! enabling immersive multi-sensory experiences that combine audio positioning
//! with tactile sensations.

use crate::{types::AudioChannel, Position3D, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Haptic device interface for tactile feedback
pub trait HapticDevice: Send + Sync {
    /// Send a haptic pattern to the device
    fn send_pattern(&mut self, pattern: &HapticPattern) -> Result<()>;

    /// Stop all haptic feedback
    fn stop(&mut self) -> Result<()>;

    /// Check if the device is connected and ready
    fn is_ready(&self) -> bool;

    /// Get device capabilities
    fn capabilities(&self) -> HapticCapabilities;

    /// Get device identifier
    fn device_id(&self) -> String;
}

/// Haptic feedback pattern with timing and intensity control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticPattern {
    /// Pattern identifier
    pub id: String,

    /// Pattern name
    pub name: String,

    /// Pattern elements with timing
    pub elements: Vec<HapticElement>,

    /// Total pattern duration
    pub duration: Duration,

    /// Whether pattern should loop
    pub looping: bool,

    /// Pattern priority (higher numbers take precedence)
    pub priority: u8,

    /// Spatial positioning for the haptic effect
    pub spatial_position: Option<Position3D>,
}

/// Individual haptic element within a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticElement {
    /// Start time within the pattern
    pub start_time: Duration,

    /// Element duration
    pub duration: Duration,

    /// Haptic effect type
    pub effect_type: HapticEffectType,

    /// Intensity (0.0 to 1.0)
    pub intensity: f32,

    /// Frequency for vibration effects (Hz)
    pub frequency: Option<f32>,

    /// Spatial attenuation based on audio source distance
    pub spatial_attenuation: f32,
}

/// Types of haptic effects
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HapticEffectType {
    /// Simple vibration
    Vibration,

    /// Pulsed vibration
    Pulse,

    /// Continuous buzz
    Buzz,

    /// Sharp click/tap
    Click,

    /// Rumble effect
    Rumble,

    /// Directional force
    DirectionalForce,

    /// Temperature change
    Thermal,

    /// Custom effect
    Custom(String),
}

/// Haptic device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticCapabilities {
    /// Maximum simultaneous effects
    pub max_concurrent_effects: usize,

    /// Supported effect types
    pub supported_effects: Vec<HapticEffectType>,

    /// Intensity resolution (steps between 0.0 and 1.0)
    pub intensity_resolution: u32,

    /// Frequency range for vibration (Hz)
    pub frequency_range: Option<(f32, f32)>,

    /// Spatial positioning support
    pub spatial_support: bool,

    /// Latency compensation in milliseconds
    pub latency_compensation: f32,

    /// Device-specific features
    pub features: HashMap<String, String>,
}

/// Configuration for haptic-audio integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticAudioConfig {
    /// Enable haptic feedback
    pub enabled: bool,

    /// Master haptic intensity multiplier
    pub master_intensity: f32,

    /// Distance-based attenuation curve
    pub distance_attenuation: DistanceAttenuation,

    /// Audio-to-haptic mapping settings
    pub audio_mapping: AudioHapticMapping,

    /// Synchronization settings
    pub sync_settings: SyncSettings,

    /// Pattern preferences
    pub pattern_preferences: PatternPreferences,
}

/// Distance-based attenuation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceAttenuation {
    /// Minimum distance for full intensity
    pub min_distance: f32,

    /// Maximum distance for zero intensity
    pub max_distance: f32,

    /// Attenuation curve type
    pub curve_type: AttenuationCurve,

    /// Custom curve parameters
    pub curve_parameters: Vec<f32>,
}

/// Attenuation curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttenuationCurve {
    /// Linear attenuation
    Linear,

    /// Inverse square law (realistic)
    InverseSquare,

    /// Exponential decay
    Exponential,

    /// Logarithmic curve
    Logarithmic,

    /// Custom curve with parameters
    Custom,
}

/// Audio-to-haptic mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioHapticMapping {
    /// Bass frequency haptic mapping
    pub bass_mapping: FrequencyMapping,

    /// Mid-range frequency haptic mapping
    pub mid_mapping: FrequencyMapping,

    /// High frequency haptic mapping
    pub high_mapping: FrequencyMapping,

    /// Transient detection settings
    pub transient_settings: TransientSettings,

    /// Rhythm extraction settings
    pub rhythm_settings: RhythmSettings,
}

/// Frequency band to haptic mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyMapping {
    /// Frequency range (Hz)
    pub frequency_range: (f32, f32),

    /// Haptic effect type for this range
    pub effect_type: HapticEffectType,

    /// Intensity scaling factor
    pub intensity_scale: f32,

    /// Frequency to haptic frequency mapping
    pub frequency_scale: f32,

    /// Threshold for activation
    pub activation_threshold: f32,
}

/// Transient detection for haptic triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransientSettings {
    /// Enable transient detection
    pub enabled: bool,

    /// Detection sensitivity
    pub sensitivity: f32,

    /// Minimum time between transients (ms)
    pub min_interval: f32,

    /// Haptic effect for transients
    pub transient_effect: HapticEffectType,
}

/// Rhythm extraction for haptic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmSettings {
    /// Enable rhythm-based haptics
    pub enabled: bool,

    /// Tempo detection range (BPM)
    pub tempo_range: (f32, f32),

    /// Beat emphasis intensity
    pub beat_emphasis: f32,

    /// Downbeat special effect
    pub downbeat_effect: Option<HapticEffectType>,
}

/// Synchronization settings for audio-haptic alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSettings {
    /// Audio-haptic latency compensation (ms)
    pub latency_compensation: f32,

    /// Synchronization tolerance (ms)
    pub sync_tolerance: f32,

    /// Prediction lookahead (ms)
    pub prediction_lookahead: f32,

    /// Buffer size for processing
    pub buffer_size: usize,
}

/// Pattern preference settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPreferences {
    /// Preferred pattern styles
    pub preferred_styles: Vec<PatternStyle>,

    /// Pattern complexity level (0-10)
    pub complexity_level: u8,

    /// Adaptive pattern learning
    pub adaptive_learning: bool,

    /// User comfort settings
    pub comfort_settings: HapticComfortSettings,
}

/// Haptic pattern style categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternStyle {
    /// Minimal, subtle feedback
    Subtle,

    /// Moderate intensity
    Moderate,

    /// Strong, pronounced effects
    Intense,

    /// Musical rhythm-based
    Musical,

    /// Environmental effects
    Environmental,

    /// Narrative/storytelling effects
    Narrative,
}

/// User comfort and safety settings for haptic feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticComfortSettings {
    /// Maximum session duration (minutes)
    pub max_session_duration: u32,

    /// Rest interval frequency (minutes)
    pub rest_interval: u32,

    /// Intensity fade during long sessions
    pub session_fade: bool,

    /// Accessibility accommodations
    pub accessibility: HapticAccessibilitySettings,
}

/// Accessibility settings for haptic feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticAccessibilitySettings {
    /// Support for users with limited tactile sensitivity
    pub enhanced_intensity: bool,

    /// Visual feedback substitution
    pub visual_substitution: bool,

    /// Simplified pattern mode
    pub simplified_patterns: bool,

    /// One-handed operation support
    pub one_handed_mode: bool,
}

/// Main haptic integration processor
pub struct HapticAudioProcessor {
    /// Configuration
    config: HapticAudioConfig,

    /// Connected haptic devices
    devices: Arc<RwLock<HashMap<String, Box<dyn HapticDevice>>>>,

    /// Active patterns
    active_patterns: Arc<RwLock<HashMap<String, ActivePattern>>>,

    /// Audio analysis state
    audio_analyzer: AudioAnalyzer,

    /// Pattern library
    pattern_library: PatternLibrary,

    /// Synchronization state
    sync_state: SyncState,

    /// Performance metrics
    metrics: HapticMetrics,
}

/// Active pattern tracking
#[derive(Debug)]
struct ActivePattern {
    /// Pattern definition
    pattern: HapticPattern,

    /// Start time
    start_time: Instant,

    /// Current element index
    current_element: usize,

    /// Associated audio source
    audio_source_id: Option<String>,

    /// Current spatial position
    current_position: Option<Position3D>,

    /// Intensity scaling factor
    intensity_scale: f32,
}

/// Audio analysis for haptic generation
struct AudioAnalyzer {
    /// FFT analysis window
    fft_window: Vec<f32>,

    /// Previous audio frame for comparison
    previous_frame: Vec<f32>,

    /// Beat detection state
    beat_detector: BeatDetector,

    /// Transient detection state
    transient_detector: TransientDetector,

    /// Frequency analysis bins
    frequency_bins: Vec<f32>,
}

/// Beat detection for rhythm-based haptics
struct BeatDetector {
    /// Energy history for beat detection
    energy_history: Vec<f32>,

    /// Current tempo estimate (BPM)
    current_tempo: f32,

    /// Beat phase tracking
    beat_phase: f32,

    /// Last beat time
    last_beat: Instant,
}

/// Transient detection for haptic triggers
struct TransientDetector {
    /// Spectral flux history
    flux_history: Vec<Vec<f32>>,

    /// Detection threshold
    threshold: f32,

    /// Last transient time
    last_transient: Instant,
}

/// Pattern library management
struct PatternLibrary {
    /// Built-in patterns
    builtin_patterns: HashMap<String, HapticPattern>,

    /// User-created patterns
    user_patterns: HashMap<String, HapticPattern>,

    /// Pattern usage statistics
    usage_stats: HashMap<String, PatternUsageStats>,
}

/// Pattern usage statistics for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternUsageStats {
    /// Number of times used
    usage_count: u32,

    /// Average user rating
    average_rating: f32,

    /// Contexts where pattern was effective
    effective_contexts: Vec<String>,

    /// Last used timestamp (as seconds since epoch)
    #[serde(skip)]
    last_used: Option<Instant>,
}

/// Synchronization state management
struct SyncState {
    /// Audio timeline reference
    audio_timeline: Instant,

    /// Haptic timeline reference
    haptic_timeline: Instant,

    /// Measured latency offset
    measured_latency: Duration,

    /// Prediction buffer
    prediction_buffer: Vec<PredictedHapticEvent>,
}

/// Predicted haptic event for synchronization
#[derive(Debug)]
struct PredictedHapticEvent {
    /// Event timestamp
    timestamp: Instant,

    /// Haptic pattern to trigger
    pattern_id: String,

    /// Spatial position
    position: Option<Position3D>,

    /// Intensity scaling
    intensity: f32,
}

/// Performance metrics for haptic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticMetrics {
    /// Processing latency (ms)
    pub processing_latency: f32,

    /// Synchronization accuracy (ms RMS error)
    pub sync_accuracy: f32,

    /// Active patterns count
    pub active_patterns: usize,

    /// Device utilization percentage
    pub device_utilization: f32,

    /// Pattern cache hit rate
    pub cache_hit_rate: f32,

    /// User satisfaction rating
    pub user_satisfaction: f32,

    /// System resource usage
    pub resource_usage: ResourceUsage,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage (MB)
    pub memory_usage: f32,

    /// Pattern library size
    pub pattern_library_size: usize,

    /// Active device count
    pub active_devices: usize,
}

impl Default for HapticAudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            master_intensity: 0.7,
            distance_attenuation: DistanceAttenuation {
                min_distance: 0.5,
                max_distance: 10.0,
                curve_type: AttenuationCurve::InverseSquare,
                curve_parameters: vec![],
            },
            audio_mapping: AudioHapticMapping {
                bass_mapping: FrequencyMapping {
                    frequency_range: (20.0, 250.0),
                    effect_type: HapticEffectType::Rumble,
                    intensity_scale: 1.2,
                    frequency_scale: 0.1,
                    activation_threshold: 0.3,
                },
                mid_mapping: FrequencyMapping {
                    frequency_range: (250.0, 4000.0),
                    effect_type: HapticEffectType::Vibration,
                    intensity_scale: 1.0,
                    frequency_scale: 0.5,
                    activation_threshold: 0.2,
                },
                high_mapping: FrequencyMapping {
                    frequency_range: (4000.0, 20000.0),
                    effect_type: HapticEffectType::Click,
                    intensity_scale: 0.8,
                    frequency_scale: 1.0,
                    activation_threshold: 0.4,
                },
                transient_settings: TransientSettings {
                    enabled: true,
                    sensitivity: 0.7,
                    min_interval: 50.0,
                    transient_effect: HapticEffectType::Click,
                },
                rhythm_settings: RhythmSettings {
                    enabled: true,
                    tempo_range: (60.0, 180.0),
                    beat_emphasis: 1.5,
                    downbeat_effect: Some(HapticEffectType::Pulse),
                },
            },
            sync_settings: SyncSettings {
                latency_compensation: 15.0,
                sync_tolerance: 5.0,
                prediction_lookahead: 100.0,
                buffer_size: 1024,
            },
            pattern_preferences: PatternPreferences {
                preferred_styles: vec![PatternStyle::Moderate, PatternStyle::Musical],
                complexity_level: 5,
                adaptive_learning: true,
                comfort_settings: HapticComfortSettings {
                    max_session_duration: 60,
                    rest_interval: 15,
                    session_fade: true,
                    accessibility: HapticAccessibilitySettings {
                        enhanced_intensity: false,
                        visual_substitution: false,
                        simplified_patterns: false,
                        one_handed_mode: false,
                    },
                },
            },
        }
    }
}

impl HapticAudioProcessor {
    /// Create new haptic audio processor
    pub fn new(config: HapticAudioConfig) -> Self {
        Self {
            config,
            devices: Arc::new(RwLock::new(HashMap::new())),
            active_patterns: Arc::new(RwLock::new(HashMap::new())),
            audio_analyzer: AudioAnalyzer::new(),
            pattern_library: PatternLibrary::new(),
            sync_state: SyncState::new(),
            metrics: HapticMetrics::default(),
        }
    }

    /// Add haptic device
    pub fn add_device(&mut self, device: Box<dyn HapticDevice>) -> Result<()> {
        let device_id = device.device_id();
        let mut devices = self.devices.write().unwrap();
        devices.insert(device_id, device);
        Ok(())
    }

    /// Remove haptic device
    pub fn remove_device(&mut self, device_id: &str) -> Result<()> {
        let mut devices = self.devices.write().unwrap();
        devices.remove(device_id);
        Ok(())
    }

    /// Process audio frame and generate haptic feedback
    pub fn process_audio_frame(
        &mut self,
        audio_samples: &[f32],
        audio_channel_type: AudioChannel,
        spatial_positions: &[(String, Position3D)],
        listener_position: Position3D,
    ) -> Result<()> {
        // Analyze audio for haptic generation
        let haptic_events = self.audio_analyzer.analyze_frame(
            audio_samples,
            audio_channel_type,
            &self.config.audio_mapping,
        )?;

        // Process spatial positioning
        let spatial_haptic_events =
            self.apply_spatial_processing(haptic_events, spatial_positions, listener_position)?;

        // Generate haptic patterns
        for event in spatial_haptic_events {
            self.trigger_haptic_event(event)?;
        }

        // Update active patterns
        self.update_active_patterns()?;

        // Update metrics
        self.update_metrics();

        Ok(())
    }

    /// Manually trigger haptic pattern
    pub fn trigger_pattern(
        &mut self,
        pattern_id: &str,
        position: Option<Position3D>,
        intensity_scale: f32,
    ) -> Result<()> {
        if let Some(pattern) = self.pattern_library.get_pattern(pattern_id) {
            let active_pattern = ActivePattern {
                pattern: pattern.clone(),
                start_time: Instant::now(),
                current_element: 0,
                audio_source_id: None,
                current_position: position,
                intensity_scale,
            };

            let mut active_patterns = self.active_patterns.write().unwrap();
            active_patterns.insert(pattern_id.to_string(), active_pattern);
        }

        Ok(())
    }

    /// Stop all haptic feedback
    pub fn stop_all(&mut self) -> Result<()> {
        // Stop all devices
        let mut devices = self.devices.write().unwrap();
        for device in devices.values_mut() {
            device.stop()?;
        }

        // Clear active patterns
        let mut active_patterns = self.active_patterns.write().unwrap();
        active_patterns.clear();

        Ok(())
    }

    /// Get current metrics
    pub fn metrics(&self) -> &HapticMetrics {
        &self.metrics
    }

    /// Update configuration
    pub fn update_config(&mut self, config: HapticAudioConfig) {
        self.config = config;
    }

    // Private helper methods

    fn apply_spatial_processing(
        &self,
        events: Vec<HapticEvent>,
        spatial_positions: &[(String, Position3D)],
        listener_position: Position3D,
    ) -> Result<Vec<SpatialHapticEvent>> {
        let mut spatial_events = Vec::new();

        for event in events {
            // Find corresponding spatial position
            if let Some((_, source_position)) = spatial_positions
                .iter()
                .find(|(id, _)| id == &event.source_id)
            {
                // Calculate distance and attenuation
                let distance = calculate_distance(listener_position, *source_position);
                let attenuation = self.calculate_distance_attenuation(distance);

                let spatial_event = SpatialHapticEvent {
                    base_event: event,
                    position: *source_position,
                    distance,
                    attenuation,
                };

                spatial_events.push(spatial_event);
            }
        }

        Ok(spatial_events)
    }

    fn calculate_distance_attenuation(&self, distance: f32) -> f32 {
        let attenuation = &self.config.distance_attenuation;

        if distance <= attenuation.min_distance {
            return 1.0;
        }

        if distance >= attenuation.max_distance {
            return 0.0;
        }

        let normalized_distance = (distance - attenuation.min_distance)
            / (attenuation.max_distance - attenuation.min_distance);

        match attenuation.curve_type {
            AttenuationCurve::Linear => 1.0 - normalized_distance,
            AttenuationCurve::InverseSquare => {
                1.0 / (1.0 + normalized_distance * normalized_distance)
            }
            AttenuationCurve::Exponential => (-normalized_distance * 2.0).exp(),
            AttenuationCurve::Logarithmic => (1.0 - normalized_distance).ln().abs().min(1.0),
            AttenuationCurve::Custom => {
                // Custom curve implementation would go here
                1.0 - normalized_distance
            }
        }
    }

    fn trigger_haptic_event(&mut self, event: SpatialHapticEvent) -> Result<()> {
        // Find appropriate pattern for the event
        let pattern = self.pattern_library.select_pattern_for_event(&event)?;

        // Apply spatial and intensity scaling
        let mut scaled_pattern = pattern;
        for element in &mut scaled_pattern.elements {
            element.intensity *= event.attenuation * self.config.master_intensity;
            element.spatial_attenuation = event.attenuation;
        }

        // Set spatial position
        scaled_pattern.spatial_position = Some(event.position);

        // Send to devices
        let devices = self.devices.read().unwrap();
        for device in devices.values() {
            if device.is_ready() {
                // Create device-specific pattern based on capabilities
                let device_pattern =
                    self.adapt_pattern_for_device(&scaled_pattern, device.capabilities())?;
                // Note: would need to make device mutable or use interior mutability
            }
        }

        Ok(())
    }

    fn adapt_pattern_for_device(
        &self,
        pattern: &HapticPattern,
        capabilities: HapticCapabilities,
    ) -> Result<HapticPattern> {
        let mut adapted_pattern = pattern.clone();

        // Filter unsupported effects
        adapted_pattern.elements.retain(|element| {
            capabilities
                .supported_effects
                .contains(&element.effect_type)
        });

        // Adjust intensity resolution
        for element in &mut adapted_pattern.elements {
            let resolution_step = 1.0 / capabilities.intensity_resolution as f32;
            element.intensity = (element.intensity / resolution_step).round() * resolution_step;
        }

        // Limit concurrent effects
        if adapted_pattern.elements.len() > capabilities.max_concurrent_effects {
            adapted_pattern
                .elements
                .truncate(capabilities.max_concurrent_effects);
        }

        Ok(adapted_pattern)
    }

    fn update_active_patterns(&mut self) -> Result<()> {
        let mut active_patterns = self.active_patterns.write().unwrap();
        let current_time = Instant::now();

        // Remove completed patterns
        active_patterns.retain(|_, pattern| {
            let elapsed = current_time.duration_since(pattern.start_time);
            elapsed < pattern.pattern.duration || pattern.pattern.looping
        });

        // Update pattern states
        for pattern in active_patterns.values_mut() {
            let elapsed = current_time.duration_since(pattern.start_time);

            // Update current element index
            while pattern.current_element < pattern.pattern.elements.len() {
                let element = &pattern.pattern.elements[pattern.current_element];
                if elapsed >= element.start_time {
                    pattern.current_element += 1;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn update_metrics(&mut self) {
        let active_patterns = self.active_patterns.read().unwrap();
        let devices = self.devices.read().unwrap();

        self.metrics.active_patterns = active_patterns.len();
        self.metrics.resource_usage.active_devices = devices.len();
        self.metrics.resource_usage.pattern_library_size = self.pattern_library.size();

        // Update other metrics (would be implemented with actual measurements)
        self.metrics.processing_latency = 5.0; // Placeholder
        self.metrics.sync_accuracy = 2.0; // Placeholder
        self.metrics.device_utilization = 65.0; // Placeholder
        self.metrics.cache_hit_rate = 85.0; // Placeholder
    }
}

// Helper structs for internal processing

#[derive(Debug)]
struct HapticEvent {
    source_id: String,
    effect_type: HapticEffectType,
    intensity: f32,
    frequency: Option<f32>,
    timestamp: Instant,
}

#[derive(Debug)]
struct SpatialHapticEvent {
    base_event: HapticEvent,
    position: Position3D,
    distance: f32,
    attenuation: f32,
}

impl Default for HapticMetrics {
    fn default() -> Self {
        Self {
            processing_latency: 0.0,
            sync_accuracy: 0.0,
            active_patterns: 0,
            device_utilization: 0.0,
            cache_hit_rate: 0.0,
            user_satisfaction: 5.0,
            resource_usage: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                pattern_library_size: 0,
                active_devices: 0,
            },
        }
    }
}

impl AudioAnalyzer {
    fn new() -> Self {
        Self {
            fft_window: vec![0.0; 1024],
            previous_frame: vec![0.0; 1024],
            beat_detector: BeatDetector::new(),
            transient_detector: TransientDetector::new(),
            frequency_bins: vec![0.0; 512],
        }
    }

    fn analyze_frame(
        &mut self,
        audio_samples: &[f32],
        audio_channel_type: AudioChannel,
        mapping: &AudioHapticMapping,
    ) -> Result<Vec<HapticEvent>> {
        let mut events = Vec::new();

        // Perform FFT analysis
        self.perform_fft_analysis(audio_samples)?;

        // Detect transients
        if mapping.transient_settings.enabled {
            if let Some(transient) = self.transient_detector.detect(&self.frequency_bins)? {
                events.push(HapticEvent {
                    source_id: format!("audio_{audio_channel_type:?}"),
                    effect_type: mapping.transient_settings.transient_effect.clone(),
                    intensity: transient.intensity,
                    frequency: None,
                    timestamp: Instant::now(),
                });
            }
        }

        // Detect beats
        if mapping.rhythm_settings.enabled {
            if let Some(beat) = self.beat_detector.detect(&self.frequency_bins)? {
                let effect_type = if beat.is_downbeat {
                    mapping
                        .rhythm_settings
                        .downbeat_effect
                        .clone()
                        .unwrap_or(HapticEffectType::Pulse)
                } else {
                    HapticEffectType::Pulse
                };

                events.push(HapticEvent {
                    source_id: format!("audio_{audio_channel_type:?}"),
                    effect_type,
                    intensity: beat.intensity * mapping.rhythm_settings.beat_emphasis,
                    frequency: Some(beat.tempo / 60.0), // Convert BPM to Hz
                    timestamp: Instant::now(),
                });
            }
        }

        // Analyze frequency bands
        self.analyze_frequency_bands(mapping, &audio_channel_type, &mut events)?;

        Ok(events)
    }

    fn perform_fft_analysis(&mut self, samples: &[f32]) -> Result<()> {
        // Copy samples to FFT window (simplified implementation)
        let copy_len = samples.len().min(self.fft_window.len());
        self.fft_window[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Perform FFT (simplified - would use proper FFT library)
        for (i, bin) in self.frequency_bins.iter_mut().enumerate() {
            *bin = self.fft_window[i * 2].abs(); // Simplified magnitude calculation
        }

        Ok(())
    }

    fn analyze_frequency_bands(
        &self,
        mapping: &AudioHapticMapping,
        audio_channel_type: &AudioChannel,
        events: &mut Vec<HapticEvent>,
    ) -> Result<()> {
        let mappings = [
            &mapping.bass_mapping,
            &mapping.mid_mapping,
            &mapping.high_mapping,
        ];

        for frequency_mapping in mappings {
            let band_energy = self.calculate_band_energy(&frequency_mapping.frequency_range);

            if band_energy > frequency_mapping.activation_threshold {
                events.push(HapticEvent {
                    source_id: format!("audio_{audio_channel_type:?}"),
                    effect_type: frequency_mapping.effect_type.clone(),
                    intensity: band_energy * frequency_mapping.intensity_scale,
                    frequency: Some(
                        (frequency_mapping.frequency_range.0 + frequency_mapping.frequency_range.1)
                            / 2.0
                            * frequency_mapping.frequency_scale,
                    ),
                    timestamp: Instant::now(),
                });
            }
        }

        Ok(())
    }

    fn calculate_band_energy(&self, frequency_range: &(f32, f32)) -> f32 {
        // Simplified band energy calculation
        let start_bin = (frequency_range.0 / 20000.0 * self.frequency_bins.len() as f32) as usize;
        let end_bin = (frequency_range.1 / 20000.0 * self.frequency_bins.len() as f32) as usize;

        self.frequency_bins[start_bin..end_bin.min(self.frequency_bins.len())]
            .iter()
            .sum::<f32>()
            / (end_bin - start_bin) as f32
    }
}

impl BeatDetector {
    fn new() -> Self {
        Self {
            energy_history: Vec::with_capacity(100),
            current_tempo: 120.0,
            beat_phase: 0.0,
            last_beat: Instant::now(),
        }
    }

    fn detect(&mut self, frequency_bins: &[f32]) -> Result<Option<BeatEvent>> {
        // Calculate current energy
        let current_energy = frequency_bins.iter().sum::<f32>() / frequency_bins.len() as f32;
        self.energy_history.push(current_energy);

        if self.energy_history.len() > 100 {
            self.energy_history.remove(0);
        }

        // Simple beat detection based on energy peaks
        if self.energy_history.len() >= 3 {
            let len = self.energy_history.len();
            let current = self.energy_history[len - 1];
            let previous = self.energy_history[len - 2];
            let prev_prev = self.energy_history[len - 3];

            // Detect peak
            if current > previous && previous > prev_prev && current > 0.5 {
                let now = Instant::now();
                let interval = now.duration_since(self.last_beat).as_secs_f32();

                if interval > 0.3 {
                    // Minimum 200 BPM
                    self.last_beat = now;
                    self.current_tempo = 60.0 / interval;

                    return Ok(Some(BeatEvent {
                        intensity: current,
                        tempo: self.current_tempo,
                        is_downbeat: self.beat_phase < 0.1, // First beat in measure
                    }));
                }
            }
        }

        Ok(None)
    }
}

impl TransientDetector {
    fn new() -> Self {
        Self {
            flux_history: Vec::with_capacity(10),
            threshold: 0.1,
            last_transient: Instant::now(),
        }
    }

    fn detect(&mut self, frequency_bins: &[f32]) -> Result<Option<TransientEvent>> {
        // Calculate spectral flux (rate of change in spectrum)
        if !self.flux_history.is_empty() {
            let previous_bins = &self.flux_history[self.flux_history.len() - 1];
            let flux: f32 = frequency_bins
                .iter()
                .zip(previous_bins.iter())
                .map(|(current, previous)| (current - previous).max(0.0))
                .sum();

            if flux > self.threshold {
                let now = Instant::now();
                let interval = now.duration_since(self.last_transient).as_millis() as f32;

                if interval > 50.0 {
                    // Minimum 50ms between transients
                    self.last_transient = now;
                    return Ok(Some(TransientEvent {
                        intensity: flux.min(1.0),
                        flux_value: flux,
                    }));
                }
            }
        }

        // Store current bins for next comparison
        self.flux_history.push(frequency_bins.to_vec());
        if self.flux_history.len() > 10 {
            self.flux_history.remove(0);
        }

        Ok(None)
    }
}

impl PatternLibrary {
    fn new() -> Self {
        let mut builtin_patterns = HashMap::new();

        // Add some built-in patterns
        builtin_patterns.insert(
            "bass_rumble".to_string(),
            Self::create_bass_rumble_pattern(),
        );
        builtin_patterns.insert("click_feedback".to_string(), Self::create_click_pattern());
        builtin_patterns.insert("heartbeat".to_string(), Self::create_heartbeat_pattern());

        Self {
            builtin_patterns,
            user_patterns: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }

    fn get_pattern(&self, pattern_id: &str) -> Option<&HapticPattern> {
        self.builtin_patterns
            .get(pattern_id)
            .or_else(|| self.user_patterns.get(pattern_id))
    }

    fn select_pattern_for_event(&self, event: &SpatialHapticEvent) -> Result<HapticPattern> {
        // Simple pattern selection logic
        let pattern_id = match event.base_event.effect_type {
            HapticEffectType::Rumble => "bass_rumble",
            HapticEffectType::Click => "click_feedback",
            HapticEffectType::Pulse => "heartbeat",
            _ => "click_feedback",
        };

        self.get_pattern(pattern_id)
            .cloned()
            .ok_or_else(|| crate::Error::LegacyProcessing("Pattern not found".to_string()))
    }

    fn size(&self) -> usize {
        self.builtin_patterns.len() + self.user_patterns.len()
    }

    fn create_bass_rumble_pattern() -> HapticPattern {
        HapticPattern {
            id: "bass_rumble".to_string(),
            name: "Bass Rumble".to_string(),
            elements: vec![HapticElement {
                start_time: Duration::from_millis(0),
                duration: Duration::from_millis(200),
                effect_type: HapticEffectType::Rumble,
                intensity: 0.8,
                frequency: Some(40.0),
                spatial_attenuation: 1.0,
            }],
            duration: Duration::from_millis(200),
            looping: false,
            priority: 5,
            spatial_position: None,
        }
    }

    fn create_click_pattern() -> HapticPattern {
        HapticPattern {
            id: "click_feedback".to_string(),
            name: "Click Feedback".to_string(),
            elements: vec![HapticElement {
                start_time: Duration::from_millis(0),
                duration: Duration::from_millis(50),
                effect_type: HapticEffectType::Click,
                intensity: 1.0,
                frequency: None,
                spatial_attenuation: 1.0,
            }],
            duration: Duration::from_millis(50),
            looping: false,
            priority: 8,
            spatial_position: None,
        }
    }

    fn create_heartbeat_pattern() -> HapticPattern {
        HapticPattern {
            id: "heartbeat".to_string(),
            name: "Heartbeat".to_string(),
            elements: vec![
                HapticElement {
                    start_time: Duration::from_millis(0),
                    duration: Duration::from_millis(100),
                    effect_type: HapticEffectType::Pulse,
                    intensity: 0.9,
                    frequency: Some(2.0),
                    spatial_attenuation: 1.0,
                },
                HapticElement {
                    start_time: Duration::from_millis(150),
                    duration: Duration::from_millis(80),
                    effect_type: HapticEffectType::Pulse,
                    intensity: 0.7,
                    frequency: Some(2.0),
                    spatial_attenuation: 1.0,
                },
            ],
            duration: Duration::from_millis(800),
            looping: true,
            priority: 3,
            spatial_position: None,
        }
    }
}

impl SyncState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            audio_timeline: now,
            haptic_timeline: now,
            measured_latency: Duration::from_millis(15),
            prediction_buffer: Vec::new(),
        }
    }
}

// Helper structs for audio analysis

#[derive(Debug)]
struct BeatEvent {
    intensity: f32,
    tempo: f32,
    is_downbeat: bool,
}

#[derive(Debug)]
struct TransientEvent {
    intensity: f32,
    flux_value: f32,
}

// Utility functions

fn calculate_distance(pos1: Position3D, pos2: Position3D) -> f32 {
    let dx = pos1.x - pos2.x;
    let dy = pos1.y - pos2.y;
    let dz = pos1.z - pos2.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haptic_config_creation() {
        let config = HapticAudioConfig::default();
        assert!(config.enabled);
        assert_eq!(config.master_intensity, 0.7);
    }

    #[test]
    fn test_pattern_creation() {
        let pattern = HapticPattern {
            id: "test".to_string(),
            name: "Test Pattern".to_string(),
            elements: vec![],
            duration: Duration::from_millis(100),
            looping: false,
            priority: 5,
            spatial_position: None,
        };
        assert_eq!(pattern.id, "test");
        assert_eq!(pattern.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_distance_calculation() {
        let pos1 = Position3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let pos2 = Position3D {
            x: 3.0,
            y: 4.0,
            z: 0.0,
        };
        let distance = calculate_distance(pos1, pos2);
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_haptic_processor_creation() {
        let config = HapticAudioConfig::default();
        let processor = HapticAudioProcessor::new(config);
        assert_eq!(processor.metrics().active_patterns, 0);
    }

    #[test]
    fn test_pattern_library() {
        let library = PatternLibrary::new();
        assert!(library.get_pattern("bass_rumble").is_some());
        assert!(library.get_pattern("click_feedback").is_some());
        assert!(library.get_pattern("heartbeat").is_some());
        assert!(library.get_pattern("nonexistent").is_none());
    }

    #[test]
    fn test_distance_attenuation() {
        let config = HapticAudioConfig::default();
        let processor = HapticAudioProcessor::new(config);

        // Test linear attenuation
        let attenuation_close = processor.calculate_distance_attenuation(1.0);
        let attenuation_far = processor.calculate_distance_attenuation(8.0);

        assert!(attenuation_close > attenuation_far);
        assert!(attenuation_close <= 1.0);
        assert!(attenuation_far >= 0.0);
    }

    #[test]
    fn test_effect_type_serialization() {
        let effect = HapticEffectType::Vibration;
        let serialized = serde_json::to_string(&effect).unwrap();
        let deserialized: HapticEffectType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(effect, deserialized);
    }

    #[test]
    fn test_capabilities_structure() {
        let capabilities = HapticCapabilities {
            max_concurrent_effects: 4,
            supported_effects: vec![HapticEffectType::Vibration, HapticEffectType::Click],
            intensity_resolution: 256,
            frequency_range: Some((10.0, 1000.0)),
            spatial_support: true,
            latency_compensation: 15.0,
            features: HashMap::new(),
        };

        assert_eq!(capabilities.max_concurrent_effects, 4);
        assert!(capabilities.spatial_support);
        assert_eq!(capabilities.supported_effects.len(), 2);
    }
}
