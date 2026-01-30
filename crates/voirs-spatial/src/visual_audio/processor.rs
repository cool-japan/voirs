//! Main visual audio processor implementation

use super::analyzer::VisualAudioAnalyzer;
use super::config::VisualAudioConfig;
use super::effects::VisualEffectLibrary;
use super::mapping::ScalingCurve;
use super::sync::VisualSyncState;
use super::types::{
    DirectionZone, SpatialVisualEvent, VisualAudioMetrics, VisualDisplay, VisualEffect, VisualEvent,
};
use crate::{types::AudioChannel, Position3D, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Main visual audio integration processor
pub struct VisualAudioProcessor {
    /// Configuration
    config: VisualAudioConfig,

    /// Connected visual displays
    displays: Arc<RwLock<HashMap<String, Box<dyn VisualDisplay>>>>,

    /// Active visual effects
    active_effects: Arc<RwLock<HashMap<String, ActiveVisualEffect>>>,

    /// Audio analysis for visual generation
    audio_analyzer: VisualAudioAnalyzer,

    /// Effect library
    effect_library: VisualEffectLibrary,

    /// Synchronization state
    sync_state: VisualSyncState,

    /// Performance metrics
    metrics: VisualAudioMetrics,
}

/// Active visual effect tracking
#[derive(Debug)]
struct ActiveVisualEffect {
    /// Effect definition
    effect: VisualEffect,

    /// Start time
    start_time: Instant,

    /// Current element index
    current_element: usize,

    /// Associated audio source
    audio_source_id: Option<String>,

    /// Current 3D position
    current_position: Position3D,

    /// Intensity scaling factor
    intensity_scale: f32,

    /// Distance from listener
    distance: f32,
}

impl VisualAudioProcessor {
    /// Create new visual audio processor
    pub fn new(config: VisualAudioConfig) -> Self {
        Self {
            config,
            displays: Arc::new(RwLock::new(HashMap::new())),
            active_effects: Arc::new(RwLock::new(HashMap::new())),
            audio_analyzer: VisualAudioAnalyzer::new(),
            effect_library: VisualEffectLibrary::new(),
            sync_state: VisualSyncState::new(),
            metrics: VisualAudioMetrics::default(),
        }
    }

    /// Add visual display
    pub fn add_display(&mut self, display: Box<dyn VisualDisplay>) -> Result<()> {
        let display_id = display.display_id();
        let mut displays = self.displays.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on displays: {}",
                e
            ))
        })?;
        displays.insert(display_id, display);
        Ok(())
    }

    /// Remove visual display
    pub fn remove_display(&mut self, display_id: &str) -> Result<()> {
        let mut displays = self.displays.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on displays: {}",
                e
            ))
        })?;
        displays.remove(display_id);
        Ok(())
    }

    /// Process audio frame and generate visual effects
    pub fn process_audio_frame(
        &mut self,
        audio_samples: &[f32],
        audio_channel_type: AudioChannel,
        spatial_positions: &[(String, Position3D)],
        listener_position: Position3D,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Analyze audio for visual generation
        let visual_events = self.audio_analyzer.analyze_frame(
            audio_samples,
            audio_channel_type,
            &self.config.audio_mapping,
        )?;

        // Process spatial positioning for visual effects
        let spatial_visual_events =
            self.apply_spatial_processing(visual_events, spatial_positions, listener_position)?;

        // Generate and trigger visual effects
        for event in spatial_visual_events {
            self.trigger_visual_event(event)?;
        }

        // Update active effects
        self.update_active_effects()?;

        // Render effects to displays
        self.render_to_displays()?;

        // Update metrics
        self.update_metrics();

        Ok(())
    }

    /// Manually trigger visual effect
    pub fn trigger_effect(
        &mut self,
        effect_id: &str,
        position: Position3D,
        intensity_scale: f32,
    ) -> Result<()> {
        if let Some(effect) = self.effect_library.get_effect(effect_id) {
            let active_effect = ActiveVisualEffect {
                effect: effect.clone(),
                start_time: Instant::now(),
                current_element: 0,
                audio_source_id: None,
                current_position: position,
                intensity_scale,
                distance: calculate_distance(
                    position,
                    Position3D {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
            };

            let mut active_effects = self.active_effects.write().map_err(|e| {
                crate::Error::LegacyProcessing(format!(
                    "Failed to acquire write lock on active_effects: {}",
                    e
                ))
            })?;
            active_effects.insert(effect_id.to_string(), active_effect);
        }

        Ok(())
    }

    /// Clear all visual effects
    pub fn clear_all_effects(&mut self) -> Result<()> {
        // Clear active effects
        let mut active_effects = self.active_effects.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on active_effects: {}",
                e
            ))
        })?;
        active_effects.clear();

        // Clear all displays
        let mut displays = self.displays.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on displays: {}",
                e
            ))
        })?;
        for display in displays.values_mut() {
            display.clear_all()?;
        }

        Ok(())
    }

    /// Get current metrics
    pub fn metrics(&self) -> &VisualAudioMetrics {
        &self.metrics
    }

    /// Update configuration
    pub fn update_config(&mut self, config: VisualAudioConfig) {
        self.config = config;
    }

    // Private helper methods

    fn apply_spatial_processing(
        &self,
        events: Vec<VisualEvent>,
        spatial_positions: &[(String, Position3D)],
        listener_position: Position3D,
    ) -> Result<Vec<SpatialVisualEvent>> {
        let mut spatial_events = Vec::new();

        for event in events {
            // Find corresponding spatial position or use default
            let source_position = spatial_positions
                .iter()
                .find(|(id, _)| id == &event.source_id)
                .map(|(_, pos)| *pos)
                .unwrap_or(Position3D {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });

            // Calculate distance and attenuation
            let distance = calculate_distance(listener_position, source_position);
            let attenuation = self.calculate_visual_distance_attenuation(distance);

            let spatial_event = SpatialVisualEvent {
                base_event: event,
                position: source_position,
                distance,
                attenuation,
                direction_zone: self.calculate_direction_zone(listener_position, source_position),
            };

            spatial_events.push(spatial_event);
        }

        Ok(spatial_events)
    }

    pub(crate) fn calculate_visual_distance_attenuation(&self, distance: f32) -> f32 {
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
            ScalingCurve::Linear => 1.0 - normalized_distance,
            ScalingCurve::Logarithmic => (1.0 - normalized_distance).ln().abs().min(1.0),
            ScalingCurve::Exponential => (-normalized_distance * 2.0).exp(),
            ScalingCurve::Power(p) => (1.0 - normalized_distance).powf(p),
            ScalingCurve::Custom => 1.0 - normalized_distance, // Fallback
        }
    }

    pub(crate) fn calculate_direction_zone(
        &self,
        listener_pos: Position3D,
        source_pos: Position3D,
    ) -> DirectionZone {
        let dx = source_pos.x - listener_pos.x;
        let dy = source_pos.y - listener_pos.y;
        let dz = source_pos.z - listener_pos.z;

        // Calculate azimuth angle (Y is forward, X is right)
        let azimuth = dx.atan2(dy).to_degrees();
        let normalized_azimuth = if azimuth < 0.0 {
            azimuth + 360.0
        } else {
            azimuth
        };

        // Check elevation first
        let elevation = dz.atan2((dx * dx + dy * dy).sqrt()).to_degrees();
        if elevation > 45.0 {
            return DirectionZone::Above;
        } else if elevation < -45.0 {
            return DirectionZone::Below;
        }

        // Determine horizontal zone
        match normalized_azimuth {
            a if a >= 315.0 || a < 45.0 => DirectionZone::Front,
            a if a >= 45.0 && a < 135.0 => DirectionZone::Right,
            a if a >= 135.0 && a < 225.0 => DirectionZone::Back,
            a if a >= 225.0 && a < 315.0 => DirectionZone::Left,
            _ => DirectionZone::Front,
        }
    }

    fn trigger_visual_event(&mut self, event: SpatialVisualEvent) -> Result<()> {
        // Select appropriate visual effect for the event
        let effect = self.effect_library.select_effect_for_event(&event)?;

        // Apply spatial and intensity scaling
        let mut scaled_effect = effect;
        for element in &mut scaled_effect.elements {
            element.intensity *= event.attenuation * self.config.master_intensity;
            element.distance_attenuation = event.attenuation;

            // Apply directional color coding if enabled
            if self.config.audio_mapping.directional_cues.enabled {
                if let Some(direction_color) = self
                    .config
                    .audio_mapping
                    .directional_cues
                    .direction_colors
                    .get(&event.direction_zone)
                {
                    // Blend with original color
                    element.color.r = (element.color.r + direction_color.r) * 0.5;
                    element.color.g = (element.color.g + direction_color.g) * 0.5;
                    element.color.b = (element.color.b + direction_color.b) * 0.5;
                }
            }
        }

        // Set spatial position
        scaled_effect.position = event.position;

        // Create active effect
        let active_effect = ActiveVisualEffect {
            effect: scaled_effect,
            start_time: Instant::now(),
            current_element: 0,
            audio_source_id: Some(event.base_event.source_id.clone()),
            current_position: event.position,
            intensity_scale: event.attenuation * self.config.master_intensity,
            distance: event.distance,
        };

        // Add to active effects
        let mut active_effects = self.active_effects.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on active_effects: {}",
                e
            ))
        })?;
        let effect_id = format!(
            "{}_{}",
            event.base_event.source_id,
            active_effect.start_time.elapsed().as_millis()
        );
        active_effects.insert(effect_id, active_effect);

        Ok(())
    }

    fn update_active_effects(&mut self) -> Result<()> {
        let mut active_effects = self.active_effects.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on active_effects: {}",
                e
            ))
        })?;
        let current_time = Instant::now();

        // Remove completed effects
        active_effects.retain(|_, effect| {
            let elapsed = current_time.duration_since(effect.start_time);
            elapsed < effect.effect.duration || effect.effect.looping
        });

        // Update effect states
        for effect in active_effects.values_mut() {
            let elapsed = current_time.duration_since(effect.start_time);

            // Update current element index
            while effect.current_element < effect.effect.elements.len() {
                let element = &effect.effect.elements[effect.current_element];
                if elapsed >= element.start_time {
                    effect.current_element += 1;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn render_to_displays(&mut self) -> Result<()> {
        let active_effects = self.active_effects.read().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire read lock on active_effects: {}",
                e
            ))
        })?;
        let mut displays = self.displays.write().map_err(|e| {
            crate::Error::LegacyProcessing(format!(
                "Failed to acquire write lock on displays: {}",
                e
            ))
        })?;

        // Render to each display
        for display in displays.values_mut() {
            if !display.is_ready() {
                continue;
            }

            // Clear previous frame
            display.clear_all()?;

            // Render active effects
            for effect in active_effects.values() {
                display.render_effect(&effect.effect)?;
            }

            // Update display
            display.update()?;
        }

        Ok(())
    }

    fn update_metrics(&mut self) {
        if let Ok(active_effects) = self.active_effects.read() {
            if let Ok(displays) = self.displays.read() {
                self.metrics.active_effects = active_effects.len();
                self.metrics.resource_usage.active_displays = displays.len();
                self.metrics.resource_usage.effect_library_size = self.effect_library.size();

                // Update other metrics (would be implemented with actual measurements)
                self.metrics.processing_latency = 8.0; // Placeholder
                self.metrics.sync_accuracy = 3.0; // Placeholder
                self.metrics.frame_rate = 60.0; // Placeholder
                self.metrics.gpu_utilization = 45.0; // Placeholder
                self.metrics.cache_hit_rate = 90.0; // Placeholder
            }
        }
    }
}

// Utility functions

/// Calculate distance between two 3D positions
pub(crate) fn calculate_distance(pos1: Position3D, pos2: Position3D) -> f32 {
    let dx = pos1.x - pos2.x;
    let dy = pos1.y - pos2.y;
    let dz = pos1.z - pos2.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}
