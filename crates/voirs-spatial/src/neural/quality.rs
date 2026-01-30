//! Quality control and adaptation for neural spatial audio

/// Adaptive quality controller for real-time performance
pub struct AdaptiveQualityController {
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Current quality level (0.0-1.0)
    pub current_quality: f32,
    quality_history: Vec<f32>,
    latency_history: Vec<f32>,
    adaptation_rate: f32,
}

impl AdaptiveQualityController {
    /// Create a new adaptive quality controller
    pub fn new(target_latency_ms: f32) -> Self {
        Self {
            target_latency_ms,
            current_quality: 0.8,
            quality_history: Vec::new(),
            latency_history: Vec::new(),
            adaptation_rate: 0.1,
        }
    }

    /// Update the controller with new latency measurement
    pub fn update(&mut self, latency_ms: f32) {
        self.latency_history.push(latency_ms);
        if self.latency_history.len() > 10 {
            self.latency_history.remove(0);
        }

        // Calculate average latency over recent frames
        let avg_latency =
            self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32;

        // Adjust quality based on latency performance
        if avg_latency > self.target_latency_ms * 1.2 {
            // Latency too high, reduce quality
            self.current_quality = (self.current_quality - self.adaptation_rate).max(0.1);
        } else if avg_latency < self.target_latency_ms * 0.8 {
            // Latency good, can increase quality
            self.current_quality = (self.current_quality + self.adaptation_rate * 0.5).min(1.0);
        }

        self.quality_history.push(self.current_quality);
        if self.quality_history.len() > 10 {
            self.quality_history.remove(0);
        }
    }

    /// Get the current quality level
    pub fn get_quality(&self) -> f32 {
        self.current_quality
    }
}
