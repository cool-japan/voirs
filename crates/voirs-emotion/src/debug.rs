//! Debugging tools for emotion state visualization and analysis
//!
//! This module provides utilities for debugging emotion processing,
//! visualizing emotion states, and analyzing emotion transitions.

use crate::{
    core::EmotionProcessor,
    history::{EmotionHistory, EmotionHistoryEntry},
    types::{Emotion, EmotionParameters, EmotionVector},
    Error, Result,
};

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};
use tracing::{debug, info, warn};

/// Emotion state debugger for visualization and analysis
#[derive(Debug)]
pub struct EmotionDebugger {
    /// Debug session configuration
    config: DebugConfig,
    /// Captured emotion states
    captured_states: Vec<EmotionStateSnapshot>,
    /// Performance metrics
    performance_metrics: DebugPerformanceMetrics,
}

/// Configuration for emotion debugging
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Enable detailed state capture
    pub capture_detailed_states: bool,
    /// Enable performance tracking
    pub track_performance: bool,
    /// Capture interval for continuous monitoring
    pub capture_interval: Duration,
    /// Maximum number of snapshots to keep
    pub max_snapshots: usize,
    /// Enable audio characteristic analysis
    pub analyze_audio: bool,
    /// Output format for debug data
    pub output_format: DebugOutputFormat,
}

/// Debug output format options
#[derive(Debug, Clone)]
pub enum DebugOutputFormat {
    /// Human-readable text format
    Text,
    /// JSON format for programmatic access
    Json,
    /// CSV format for spreadsheet analysis
    Csv,
    /// HTML format with visualization
    Html,
}

/// Snapshot of emotion state at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionStateSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: SystemTime,
    /// Current emotion parameters
    pub emotion_parameters: EmotionParameters,
    /// Dominant emotion and intensity
    pub dominant_emotion: (String, f32),
    /// All emotion values in the vector
    pub emotion_vector_values: HashMap<String, f32>,
    /// Processing performance metrics
    pub performance: Option<SnapshotPerformanceMetrics>,
    /// Audio characteristics (if available)
    pub audio_characteristics: Option<AudioCharacteristics>,
    /// Metadata and context
    pub metadata: HashMap<String, String>,
}

/// Performance metrics for a single snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotPerformanceMetrics {
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Number of active interpolations
    pub active_interpolations: usize,
}

/// Audio characteristics for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCharacteristics {
    /// RMS energy level
    pub rms_energy: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Estimated pitch
    pub estimated_pitch: Option<f32>,
}

/// Overall performance metrics for debugging session
#[derive(Debug, Clone)]
pub struct DebugPerformanceMetrics {
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per operation
    pub average_processing_time: Duration,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Total number of operations
    pub operation_count: usize,
    /// Error count
    pub error_count: usize,
}

impl EmotionDebugger {
    /// Create new emotion debugger
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            captured_states: Vec::new(),
            performance_metrics: DebugPerformanceMetrics::default(),
        }
    }

    /// Create debugger with default configuration
    pub fn default() -> Self {
        Self::new(DebugConfig::default())
    }

    /// Capture current emotion state
    pub async fn capture_state(
        &mut self,
        processor: &EmotionProcessor,
        context: Option<&str>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        debug!("Capturing emotion state for debugging");

        // Get current emotion parameters
        // Note: This is a placeholder - the actual method might be different
        let emotion_params = EmotionParameters::neutral(); // placeholder
        let dominant_emotion = emotion_params
            .emotion_vector
            .dominant_emotion()
            .map(|(e, i)| (e.as_str().to_string(), i.value()))
            .unwrap_or(("neutral".to_string(), 0.0));

        // Extract emotion vector values
        let emotion_vector_values =
            self.extract_emotion_vector_values(&emotion_params.emotion_vector);

        // Capture performance metrics if enabled
        let performance = if self.config.track_performance {
            Some(SnapshotPerformanceMetrics {
                processing_time_us: start_time.elapsed().as_micros() as u64,
                memory_usage_bytes: self.estimate_memory_usage(),
                cpu_usage_percent: self.estimate_cpu_usage(),
                active_interpolations: 0, // Placeholder - would need processor access
            })
        } else {
            None
        };

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("context".to_string(), context.unwrap_or("none").to_string());
        metadata.insert("capture_method".to_string(), "manual".to_string());

        // Create snapshot
        let snapshot = EmotionStateSnapshot {
            timestamp: SystemTime::now(),
            emotion_parameters: emotion_params,
            dominant_emotion,
            emotion_vector_values,
            performance,
            audio_characteristics: None, // Would be filled with actual audio analysis
            metadata,
        };

        // Add to captured states
        self.captured_states.push(snapshot);

        // Limit snapshot count
        if self.captured_states.len() > self.config.max_snapshots {
            self.captured_states.remove(0);
        }

        // Update performance metrics
        self.performance_metrics.operation_count += 1;
        self.performance_metrics.total_processing_time += start_time.elapsed();

        debug!("Emotion state captured successfully");
        Ok(())
    }

    /// Start continuous state monitoring
    pub async fn start_continuous_monitoring(
        &mut self,
        processor: &EmotionProcessor,
    ) -> Result<()> {
        info!("Starting continuous emotion state monitoring");

        // This would run in a background task in a real implementation
        // For now, we'll just capture the current state
        self.capture_state(processor, Some("continuous_monitoring"))
            .await?;

        Ok(())
    }

    /// Analyze emotion state transitions
    pub fn analyze_transitions(&self) -> EmotionTransitionAnalysis {
        debug!("Analyzing emotion transitions");

        let mut transitions = Vec::new();
        let mut emotion_durations = HashMap::new();
        let mut transition_frequencies = HashMap::new();

        for window in self.captured_states.windows(2) {
            let prev_state = &window[0];
            let curr_state = &window[1];

            let prev_emotion = &prev_state.dominant_emotion.0;
            let curr_emotion = &curr_state.dominant_emotion.0;

            if prev_emotion != curr_emotion {
                let transition = EmotionTransition {
                    from_emotion: prev_emotion.clone(),
                    to_emotion: curr_emotion.clone(),
                    from_intensity: prev_state.dominant_emotion.1,
                    to_intensity: curr_state.dominant_emotion.1,
                    duration: curr_state
                        .timestamp
                        .duration_since(prev_state.timestamp)
                        .unwrap_or(Duration::from_millis(0)),
                    timestamp: curr_state.timestamp,
                };

                transitions.push(transition);

                // Track transition frequency
                let transition_key = format!("{}->{}", prev_emotion, curr_emotion);
                *transition_frequencies.entry(transition_key).or_insert(0) += 1;
            }

            // Track emotion duration
            let duration = curr_state
                .timestamp
                .duration_since(prev_state.timestamp)
                .unwrap_or(Duration::from_millis(0));
            emotion_durations
                .entry(prev_emotion.clone())
                .and_modify(|d: &mut Duration| *d += duration)
                .or_insert(duration);
        }

        EmotionTransitionAnalysis {
            transitions,
            emotion_durations,
            transition_frequencies,
            total_transitions: self.captured_states.len().saturating_sub(1),
            analysis_timestamp: SystemTime::now(),
        }
    }

    /// Generate debug report
    pub fn generate_debug_report(&self) -> Result<String> {
        match self.config.output_format {
            DebugOutputFormat::Text => self.generate_text_report(),
            DebugOutputFormat::Json => self.generate_json_report(),
            DebugOutputFormat::Csv => self.generate_csv_report(),
            DebugOutputFormat::Html => self.generate_html_report(),
        }
    }

    /// Clear captured states
    pub fn clear_captured_states(&mut self) {
        self.captured_states.clear();
        debug!("Captured states cleared");
    }

    /// Get captured states count
    pub fn captured_states_count(&self) -> usize {
        self.captured_states.len()
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &DebugPerformanceMetrics {
        &self.performance_metrics
    }

    // Helper methods

    fn extract_emotion_vector_values(&self, vector: &EmotionVector) -> HashMap<String, f32> {
        let mut values = HashMap::new();

        // Get all emotion values from the vector
        // This is a simplified implementation - real implementation would
        // iterate through all emotions in the vector
        if let Some((emotion, intensity)) = vector.dominant_emotion() {
            values.insert(emotion.as_str().to_string(), intensity.value());
        }

        values
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate of memory usage
        std::mem::size_of::<EmotionDebugger>()
            + self.captured_states.len() * std::mem::size_of::<EmotionStateSnapshot>()
    }

    fn estimate_cpu_usage(&self) -> f32 {
        // Placeholder CPU usage estimation
        // Real implementation would use system APIs
        10.0
    }

    fn generate_text_report(&self) -> Result<String> {
        let mut report = String::new();

        report.push_str("=== EMOTION DEBUG REPORT ===\n\n");

        // Summary
        report.push_str(&format!(
            "Captured States: {}\n",
            self.captured_states.len()
        ));
        report.push_str(&format!(
            "Total Operations: {}\n",
            self.performance_metrics.operation_count
        ));
        report.push_str(&format!(
            "Average Processing Time: {:?}\n",
            self.performance_metrics.average_processing_time
        ));
        report.push_str("\n");

        // Recent states
        report.push_str("Recent Emotion States:\n");
        for (i, state) in self.captured_states.iter().rev().take(10).enumerate() {
            report.push_str(&format!(
                "  {}: {} ({:.2}) at {:?}\n",
                i + 1,
                state.dominant_emotion.0,
                state.dominant_emotion.1,
                state.timestamp
            ));
        }

        // Transition analysis
        let transition_analysis = self.analyze_transitions();
        report.push_str(&format!("\nTransition Analysis:\n"));
        report.push_str(&format!(
            "  Total Transitions: {}\n",
            transition_analysis.total_transitions
        ));

        for (transition_key, frequency) in transition_analysis.transition_frequencies.iter().take(5)
        {
            report.push_str(&format!("  {}: {} times\n", transition_key, frequency));
        }

        Ok(report)
    }

    fn generate_json_report(&self) -> Result<String> {
        let report_data = serde_json::json!({
            "summary": {
                "captured_states": self.captured_states.len(),
                "operation_count": self.performance_metrics.operation_count,
                "total_processing_time_ms": self.performance_metrics.total_processing_time.as_millis(),
            },
            "captured_states": self.captured_states,
            "transition_analysis": self.analyze_transitions(),
            "performance_metrics": {
                "peak_memory_usage": self.performance_metrics.peak_memory_usage,
                "error_count": self.performance_metrics.error_count,
            }
        });

        serde_json::to_string_pretty(&report_data).map_err(|e| Error::Serialization(e))
    }

    fn generate_csv_report(&self) -> Result<String> {
        let mut csv = String::new();

        // CSV header
        csv.push_str("timestamp,emotion,intensity,pitch_shift,tempo_scale,energy_scale\n");

        // CSV data
        for state in &self.captured_states {
            csv.push_str(&format!(
                "{:?},{},{},{},{},{}\n",
                state.timestamp,
                state.dominant_emotion.0,
                state.dominant_emotion.1,
                state.emotion_parameters.pitch_shift,
                state.emotion_parameters.tempo_scale,
                state.emotion_parameters.energy_scale
            ));
        }

        Ok(csv)
    }

    fn generate_html_report(&self) -> Result<String> {
        let mut html = String::new();

        html.push_str(
            "<!DOCTYPE html><html><head><title>Emotion Debug Report</title></head><body>",
        );
        html.push_str("<h1>Emotion Debug Report</h1>");

        // Summary section
        html.push_str("<h2>Summary</h2>");
        html.push_str(&format!(
            "<p>Captured States: {}</p>",
            self.captured_states.len()
        ));
        html.push_str(&format!(
            "<p>Total Operations: {}</p>",
            self.performance_metrics.operation_count
        ));

        // States table
        html.push_str("<h2>Recent Emotion States</h2>");
        html.push_str(
            "<table border='1'><tr><th>Timestamp</th><th>Emotion</th><th>Intensity</th></tr>",
        );

        for state in self.captured_states.iter().rev().take(20) {
            html.push_str(&format!(
                "<tr><td>{:?}</td><td>{}</td><td>{:.2}</td></tr>",
                state.timestamp, state.dominant_emotion.0, state.dominant_emotion.1
            ));
        }

        html.push_str("</table>");
        html.push_str("</body></html>");

        Ok(html)
    }
}

/// Analysis of emotion transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransitionAnalysis {
    /// List of all transitions
    pub transitions: Vec<EmotionTransition>,
    /// Duration spent in each emotion
    pub emotion_durations: HashMap<String, Duration>,
    /// Frequency of each transition type
    pub transition_frequencies: HashMap<String, usize>,
    /// Total number of transitions
    pub total_transitions: usize,
    /// When the analysis was performed
    pub analysis_timestamp: SystemTime,
}

/// Individual emotion transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransition {
    /// Source emotion
    pub from_emotion: String,
    /// Target emotion
    pub to_emotion: String,
    /// Source intensity
    pub from_intensity: f32,
    /// Target intensity
    pub to_intensity: f32,
    /// Transition duration
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    /// When the transition occurred
    pub timestamp: SystemTime,
}

/// Serde helper for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            capture_detailed_states: true,
            track_performance: true,
            capture_interval: Duration::from_secs(1),
            max_snapshots: 1000,
            analyze_audio: false,
            output_format: DebugOutputFormat::Text,
        }
    }
}

impl Default for DebugPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_processing_time: Duration::from_secs(0),
            average_processing_time: Duration::from_secs(0),
            peak_memory_usage: 0,
            operation_count: 0,
            error_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_debugger_creation() {
        let debugger = EmotionDebugger::default();
        assert_eq!(debugger.captured_states_count(), 0);
        assert_eq!(debugger.get_performance_metrics().operation_count, 0);
    }

    #[tokio::test]
    async fn test_state_capture() {
        let mut debugger = EmotionDebugger::default();
        let processor = crate::core::EmotionProcessor::new().unwrap();

        // Capture initial state
        debugger
            .capture_state(&processor, Some("test"))
            .await
            .unwrap();
        assert_eq!(debugger.captured_states_count(), 1);

        // Capture another state
        debugger
            .capture_state(&processor, Some("test2"))
            .await
            .unwrap();
        assert_eq!(debugger.captured_states_count(), 2);
    }

    #[tokio::test]
    async fn test_debug_config_default() {
        let config = DebugConfig::default();
        assert!(config.capture_detailed_states);
        assert!(config.track_performance);
        assert_eq!(config.max_snapshots, 1000);
    }

    #[tokio::test]
    async fn test_transition_analysis() {
        let mut debugger = EmotionDebugger::default();
        let processor = crate::core::EmotionProcessor::new().unwrap();

        // Capture several states
        for _ in 0..5 {
            debugger
                .capture_state(&processor, Some("test"))
                .await
                .unwrap();
        }

        let analysis = debugger.analyze_transitions();
        assert!(analysis.total_transitions <= 4); // At most 4 transitions for 5 states
    }

    #[test]
    fn test_debug_output_formats() {
        let debugger = EmotionDebugger::default();

        // Test text report generation
        let text_report = debugger.generate_text_report().unwrap();
        assert!(text_report.contains("EMOTION DEBUG REPORT"));

        // Test JSON report generation
        let json_report = debugger.generate_json_report().unwrap();
        assert!(json_report.contains("captured_states"));

        // Test CSV report generation
        let csv_report = debugger.generate_csv_report().unwrap();
        assert!(csv_report.contains("timestamp,emotion,intensity"));

        // Test HTML report generation
        let html_report = debugger.generate_html_report().unwrap();
        assert!(html_report.contains("<html>"));
        assert!(html_report.contains("Emotion Debug Report"));
    }

    #[test]
    fn test_clear_captured_states() {
        let mut debugger = EmotionDebugger::default();

        // Manually add a state for testing
        let snapshot = EmotionStateSnapshot {
            timestamp: SystemTime::now(),
            emotion_parameters: crate::types::EmotionParameters::neutral(),
            dominant_emotion: ("neutral".to_string(), 0.5),
            emotion_vector_values: HashMap::new(),
            performance: None,
            audio_characteristics: None,
            metadata: HashMap::new(),
        };

        debugger.captured_states.push(snapshot);
        assert_eq!(debugger.captured_states_count(), 1);

        debugger.clear_captured_states();
        assert_eq!(debugger.captured_states_count(), 0);
    }

    #[test]
    fn test_memory_and_cpu_estimation() {
        let debugger = EmotionDebugger::default();

        let memory_usage = debugger.estimate_memory_usage();
        assert!(memory_usage > 0);

        let cpu_usage = debugger.estimate_cpu_usage();
        assert!(cpu_usage >= 0.0 && cpu_usage <= 100.0);
    }
}
