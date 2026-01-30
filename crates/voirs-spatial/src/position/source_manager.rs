//! Spatial source management and Doppler processing

use super::occlusion::{OcclusionDetector, OcclusionResult};
use super::prediction::{MotionPredictor, MotionSnapshot};
use super::spatial_grid::SpatialGrid;
use super::types::SoundSource;
use crate::types::Position3D;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Dynamic source manager for spatial audio
pub struct SpatialSourceManager {
    /// Active sound sources
    pub sources: HashMap<String, SoundSource>,
    /// Spatial awareness grid for optimization
    spatial_grid: SpatialGrid,
    /// Occlusion detector
    occlusion_detector: OcclusionDetector,
    /// Maximum number of concurrent sources
    max_sources: usize,
    /// Distance-based culling threshold
    pub culling_distance: f32,
    /// Update frequency for spatial calculations
    update_frequency: f32,
}

/// Doppler effect processor for moving sources and listeners
#[derive(Debug, Clone)]
pub struct DopplerProcessor {
    /// Speed of sound (m/s)
    speed_of_sound: f32,
    /// Sample rate
    sample_rate: f32,
    /// Maximum Doppler shift factor (safety limit)
    max_doppler_factor: f32,
    /// Smoothing factor for Doppler transitions
    smoothing_factor: f32,
}

/// Dynamic source manager for handling moving sources
#[derive(Debug, Clone)]
pub struct DynamicSourceManager {
    /// Dynamic sources
    sources: HashMap<String, DynamicSource>,
    /// Doppler processor
    doppler_processor: DopplerProcessor,
    /// Motion predictor
    motion_predictor: MotionPredictor,
}

/// Dynamic source with motion tracking and prediction
#[derive(Debug, Clone)]
pub struct DynamicSource {
    /// Base source
    pub base_source: SoundSource,
    /// Velocity tracking
    pub velocity: Position3D,
    /// Acceleration tracking
    pub acceleration: Position3D,
    /// Motion history for prediction
    pub motion_history: VecDeque<MotionSnapshot>,
    /// Current Doppler factor
    pub doppler_factor: f32,
    /// Smoothed Doppler factor
    pub smoothed_doppler_factor: f32,
    /// Last Doppler update time
    pub last_doppler_update: Option<Instant>,
}

impl SpatialSourceManager {
    /// Create new spatial source manager
    pub fn new(bounds: (Position3D, Position3D), cell_size: f32) -> Self {
        Self {
            sources: HashMap::new(),
            spatial_grid: SpatialGrid::new(bounds, cell_size),
            occlusion_detector: OcclusionDetector::new(),
            max_sources: 64,
            culling_distance: 100.0,
            update_frequency: 60.0,
        }
    }

    /// Add source to manager
    pub fn add_source(&mut self, source: SoundSource) -> crate::Result<()> {
        if self.sources.len() >= self.max_sources {
            return Err(crate::Error::LegacyPosition(
                "Maximum sources exceeded".to_string(),
            ));
        }

        let source_id = source.id.clone();
        let position = source.position();

        // Add to spatial grid
        self.spatial_grid.add_source(&source_id, position);

        // Add to sources
        self.sources.insert(source_id, source);

        Ok(())
    }

    /// Remove source from manager
    pub fn remove_source(&mut self, source_id: &str) -> Option<SoundSource> {
        if let Some(source) = self.sources.remove(source_id) {
            self.spatial_grid.remove_source(source_id);
            Some(source)
        } else {
            None
        }
    }

    /// Update source position
    pub fn update_source_position(
        &mut self,
        source_id: &str,
        position: Position3D,
    ) -> crate::Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            let old_position = source.position();
            source.set_position(position);

            // Update spatial grid
            self.spatial_grid
                .move_source(source_id, old_position, position);

            Ok(())
        } else {
            Err(crate::Error::LegacyPosition(format!(
                "Source not found: {source_id}"
            )))
        }
    }

    /// Get nearby sources for listener
    pub fn get_nearby_sources(
        &self,
        listener_position: Position3D,
        radius: f32,
    ) -> Vec<&SoundSource> {
        let nearby_ids = self.spatial_grid.query_sphere(listener_position, radius);
        nearby_ids
            .iter()
            .filter_map(|id| self.sources.get(id))
            .collect()
    }

    /// Check occlusion for source
    pub fn check_occlusion(
        &self,
        source_position: Position3D,
        listener_position: Position3D,
    ) -> OcclusionResult {
        self.occlusion_detector
            .check_occlusion(source_position, listener_position)
    }

    /// Cull distant sources
    pub fn cull_distant_sources(&mut self, listener_position: Position3D) {
        let culling_distance_sq = self.culling_distance * self.culling_distance;
        let distant_sources: Vec<String> = self
            .sources
            .iter()
            .filter(|(_, source)| {
                let distance_sq = listener_position.distance_to(&source.position()).powi(2);
                distance_sq > culling_distance_sq
            })
            .map(|(id, _)| id.clone())
            .collect();

        for source_id in distant_sources {
            self.remove_source(&source_id);
        }
    }

    /// Get all active sources
    pub fn get_active_sources(&self) -> Vec<&SoundSource> {
        self.sources.values().filter(|s| s.is_active()).collect()
    }
}

impl DopplerProcessor {
    /// Create new Doppler processor
    pub fn new(sample_rate: f32) -> Self {
        Self {
            speed_of_sound: 343.0, // m/s at 20°C
            sample_rate,
            max_doppler_factor: 2.0, // Safety limit to prevent extreme shifts
            smoothing_factor: 0.95,  // Smooth Doppler transitions
        }
    }

    /// Create Doppler processor with custom speed of sound
    pub fn with_speed_of_sound(sample_rate: f32, speed_of_sound: f32) -> Self {
        Self {
            speed_of_sound,
            sample_rate,
            max_doppler_factor: 2.0,
            smoothing_factor: 0.95,
        }
    }

    /// Calculate Doppler factor for source and listener
    pub fn calculate_doppler_factor(
        &self,
        source_position: Position3D,
        source_velocity: Position3D,
        listener_position: Position3D,
        listener_velocity: Position3D,
    ) -> f32 {
        // Vector from source to listener
        let source_to_listener = Position3D::new(
            listener_position.x - source_position.x,
            listener_position.y - source_position.y,
            listener_position.z - source_position.z,
        );

        let distance = source_to_listener.magnitude();
        if distance < 0.001 {
            return 1.0; // Avoid division by zero
        }

        // Unit vector from source to listener
        let direction = source_to_listener.normalized();

        // Radial velocities (positive means approaching for source, receding for listener)
        let source_radial_velocity = source_velocity.dot(&direction); // Positive = towards listener
        let listener_radial_velocity = -listener_velocity.dot(&direction); // Positive = towards source

        // Classic Doppler formula: f' = f * (v + vr) / (v - vs)
        // where vr is listener radial velocity, vs is source radial velocity
        // Positive source velocity means approaching (should increase frequency)
        let numerator = self.speed_of_sound + listener_radial_velocity;
        let denominator = self.speed_of_sound - source_radial_velocity;

        if denominator.abs() < 0.001 {
            return 1.0; // Avoid division by zero
        }

        let doppler_factor = numerator / denominator;

        // Clamp to reasonable limits
        doppler_factor.clamp(1.0 / self.max_doppler_factor, self.max_doppler_factor)
    }

    /// Apply Doppler effect to audio buffer
    pub fn process_doppler_effect(
        &self,
        input: &[f32],
        output: &mut [f32],
        doppler_factor: f32,
    ) -> crate::Result<()> {
        if input.len() != output.len() {
            return Err(crate::Error::LegacyProcessing(
                "Input and output buffers must have the same length".to_string(),
            ));
        }

        if (doppler_factor - 1.0).abs() < 0.001 {
            // No significant Doppler effect, just copy
            output.copy_from_slice(input);
            return Ok(());
        }

        // Simple pitch shifting using linear interpolation
        // In production, would use more sophisticated algorithms like PSOLA or phase vocoder
        let pitch_ratio = doppler_factor;
        let mut read_pos = 0.0;

        for i in 0..output.len() {
            let read_index = read_pos as usize;
            let read_frac = read_pos - read_index as f32;

            if read_index + 1 < input.len() {
                // Linear interpolation
                let sample1 = input[read_index];
                let sample2 = input[read_index + 1];
                output[i] = sample1 + read_frac * (sample2 - sample1);
            } else if read_index < input.len() {
                output[i] = input[read_index];
            } else {
                output[i] = 0.0;
            }

            read_pos += pitch_ratio;
            if read_pos >= input.len() as f32 {
                // Fill remaining samples with zero
                output[i + 1..].fill(0.0);
                break;
            }
        }

        Ok(())
    }

    /// Smooth Doppler factor transitions
    pub fn smooth_doppler_factor(&self, current: f32, target: f32) -> f32 {
        current * self.smoothing_factor + target * (1.0 - self.smoothing_factor)
    }

    /// Set speed of sound (useful for different atmospheric conditions)
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.speed_of_sound = speed;
    }

    /// Get current speed of sound
    pub fn speed_of_sound(&self) -> f32 {
        self.speed_of_sound
    }
}

impl DynamicSourceManager {
    /// Create new dynamic source manager
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sources: HashMap::new(),
            doppler_processor: DopplerProcessor::new(sample_rate),
            motion_predictor: MotionPredictor::new(),
        }
    }

    /// Add dynamic source
    pub fn add_source(&mut self, source: SoundSource) -> crate::Result<()> {
        let dynamic_source = DynamicSource::new(source);
        self.sources
            .insert(dynamic_source.base_source.id.clone(), dynamic_source);
        Ok(())
    }

    /// Remove dynamic source
    pub fn remove_source(&mut self, source_id: &str) -> Option<DynamicSource> {
        self.sources.remove(source_id)
    }

    /// Update source motion
    pub fn update_source_motion(
        &mut self,
        source_id: &str,
        position: Position3D,
        velocity: Position3D,
        acceleration: Position3D,
    ) -> crate::Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.update_motion(position, velocity, acceleration);
            Ok(())
        } else {
            Err(crate::Error::LegacyPosition(format!(
                "Source '{source_id}' not found"
            )))
        }
    }

    /// Process all dynamic sources with Doppler effects
    pub async fn process_dynamic_sources(
        &mut self,
        listener_position: Position3D,
        listener_velocity: Position3D,
    ) -> crate::Result<()> {
        for source in self.sources.values_mut() {
            // Update Doppler factor
            let doppler_factor = self.doppler_processor.calculate_doppler_factor(
                source.base_source.position(),
                source.velocity,
                listener_position,
                listener_velocity,
            );

            // Smooth the Doppler factor
            source.smoothed_doppler_factor = self
                .doppler_processor
                .smooth_doppler_factor(source.smoothed_doppler_factor, doppler_factor);

            source.doppler_factor = doppler_factor;
            source.last_doppler_update = Some(Instant::now());
        }

        Ok(())
    }

    /// Get all dynamic sources
    pub fn sources(&self) -> &HashMap<String, DynamicSource> {
        &self.sources
    }

    /// Get dynamic source by ID
    pub fn get_source(&self, source_id: &str) -> Option<&DynamicSource> {
        self.sources.get(source_id)
    }

    /// Get mutable dynamic source by ID
    pub fn get_source_mut(&mut self, source_id: &str) -> Option<&mut DynamicSource> {
        self.sources.get_mut(source_id)
    }

    /// Predict source positions for latency compensation
    pub fn predict_source_positions(
        &self,
        prediction_time: Duration,
    ) -> HashMap<String, Position3D> {
        let mut predictions = HashMap::new();

        for (id, source) in &self.sources {
            if let Some(predicted_pos) = self
                .motion_predictor
                .predict_position(&source.motion_history, prediction_time)
            {
                predictions.insert(id.clone(), predicted_pos);
            }
        }

        predictions
    }
}

impl DynamicSource {
    /// Create new dynamic source
    pub fn new(base_source: SoundSource) -> Self {
        Self {
            base_source,
            velocity: Position3D::default(),
            acceleration: Position3D::default(),
            motion_history: VecDeque::with_capacity(100),
            doppler_factor: 1.0,
            smoothed_doppler_factor: 1.0,
            last_doppler_update: None,
        }
    }

    /// Update motion parameters
    pub fn update_motion(
        &mut self,
        position: Position3D,
        velocity: Position3D,
        acceleration: Position3D,
    ) {
        self.velocity = velocity;
        self.acceleration = acceleration;

        // Update position in base source
        self.base_source.set_position(position);

        // Add to motion history
        let snapshot = MotionSnapshot {
            position,
            velocity,
            acceleration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        };

        self.motion_history.push_back(snapshot);

        // Keep history size manageable
        while self.motion_history.len() > 100 {
            self.motion_history.pop_front();
        }
    }

    /// Get predicted position after given time
    pub fn predict_position(&self, time_delta: Duration) -> Position3D {
        let dt = time_delta.as_secs_f32();
        let current_pos = self.base_source.position();

        // Simple kinematic prediction: pos = pos0 + v*t + 0.5*a*t^2
        Position3D::new(
            current_pos.x + self.velocity.x * dt + 0.5 * self.acceleration.x * dt * dt,
            current_pos.y + self.velocity.y * dt + 0.5 * self.acceleration.y * dt * dt,
            current_pos.z + self.velocity.z * dt + 0.5 * self.acceleration.z * dt * dt,
        )
    }

    /// Check if source is moving significantly
    pub fn is_moving(&self) -> bool {
        self.velocity.magnitude() > 0.1 // m/s threshold
    }
}
