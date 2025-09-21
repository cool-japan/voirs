//! Core active learning types and interfaces
//!
//! This module contains the core types for active learning including
//! selection results, annotation results, and the main active learner interface.

use crate::{DatasetSample, QualityMetrics, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Active selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSelectionResult {
    /// Selected sample indices
    pub selected_indices: Vec<usize>,
    /// Uncertainty scores
    pub uncertainty_scores: Vec<f32>,
    /// Diversity scores
    pub diversity_scores: Vec<f32>,
    /// Combined scores
    pub combined_scores: Vec<f32>,
    /// Selection metadata
    pub metadata: HashMap<String, String>,
}

/// Annotation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationResult {
    /// Sample ID
    pub sample_id: String,
    /// Annotator ID
    pub annotator_id: String,
    /// Quality metrics annotation
    pub quality_annotation: QualityMetrics,
    /// Text corrections
    pub text_corrections: Option<String>,
    /// Audio quality issues
    pub audio_issues: Vec<AudioIssue>,
    /// Confidence in annotation
    pub confidence: f32,
    /// Annotation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional notes
    pub notes: Option<String>,
}

/// Audio quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioIssue {
    Noise,
    Clipping,
    LowVolume,
    HighVolume,
    Distortion,
    Echo,
    Reverb,
    Artifacts,
    SpeechQuality,
    Other(String),
}

/// Active learner interface
#[async_trait::async_trait]
pub trait ActiveLearner: Send + Sync {
    /// Select samples for annotation using active learning
    async fn select_samples(
        &self,
        unlabeled_samples: &[DatasetSample],
        labeled_samples: &[DatasetSample],
        batch_size: usize,
    ) -> Result<ActiveSelectionResult>;

    /// Calculate uncertainty scores for samples
    async fn calculate_uncertainty(&self, samples: &[DatasetSample]) -> Result<Vec<f32>>;

    /// Calculate diversity scores for samples
    async fn calculate_diversity(
        &self,
        samples: &[DatasetSample],
        reference_samples: &[DatasetSample],
    ) -> Result<Vec<f32>>;

    /// Update model with new annotations
    async fn update_model(&mut self, annotations: &[AnnotationResult]) -> Result<()>;

    /// Get annotation interface
    async fn get_annotation_interface(&self) -> Result<Box<dyn AnnotationInterface + Send>>;

    /// Process human feedback
    async fn process_feedback(&mut self, feedback: &[AnnotationResult]) -> Result<()>;

    /// Generate annotation statistics
    async fn get_annotation_statistics(&self) -> Result<AnnotationStatistics>;
}

/// Annotation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationStatistics {
    /// Total annotations
    pub total_annotations: usize,
    /// Annotations per annotator
    pub annotations_per_annotator: HashMap<String, usize>,
    /// Average annotation time
    pub avg_annotation_time: f32,
    /// Inter-annotator agreement
    pub inter_annotator_agreement: f32,
    /// Quality distribution
    pub quality_distribution: HashMap<String, f32>,
    /// Common issues
    pub common_issues: Vec<(AudioIssue, usize)>,
}

/// Annotation interface trait
#[async_trait::async_trait]
pub trait AnnotationInterface: Send + Sync {
    /// Start the annotation interface
    async fn start(&mut self) -> Result<()>;

    /// Stop the annotation interface
    async fn stop(&mut self) -> Result<()>;

    /// Present a sample for annotation
    async fn present_sample(&mut self, sample: &DatasetSample) -> Result<AnnotationResult>;

    /// Show feedback to the annotator
    async fn show_feedback(&mut self, feedback: &str) -> Result<()>;

    /// Get interface statistics
    async fn get_statistics(&self) -> Result<HashMap<String, f32>>;
}

impl ActiveSelectionResult {
    /// Create a new selection result
    pub fn new(
        selected_indices: Vec<usize>,
        uncertainty_scores: Vec<f32>,
        diversity_scores: Vec<f32>,
        combined_scores: Vec<f32>,
    ) -> Self {
        Self {
            selected_indices,
            uncertainty_scores,
            diversity_scores,
            combined_scores,
            metadata: HashMap::new(),
        }
    }

    /// Get the number of selected samples
    pub fn num_selected(&self) -> usize {
        self.selected_indices.len()
    }

    /// Get the top-k selected indices
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        self.selected_indices.iter().take(k).cloned().collect()
    }

    /// Get average uncertainty score
    pub fn average_uncertainty(&self) -> f32 {
        if self.uncertainty_scores.is_empty() {
            0.0
        } else {
            self.uncertainty_scores.iter().sum::<f32>() / self.uncertainty_scores.len() as f32
        }
    }

    /// Get average diversity score
    pub fn average_diversity(&self) -> f32 {
        if self.diversity_scores.is_empty() {
            0.0
        } else {
            self.diversity_scores.iter().sum::<f32>() / self.diversity_scores.len() as f32
        }
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Check if selection is valid
    pub fn is_valid(&self) -> bool {
        let len = self.selected_indices.len();
        len == self.uncertainty_scores.len()
            && len == self.diversity_scores.len()
            && len == self.combined_scores.len()
    }
}

impl AnnotationResult {
    /// Create a new annotation result
    pub fn new(
        sample_id: String,
        annotator_id: String,
        quality_annotation: QualityMetrics,
    ) -> Self {
        Self {
            sample_id,
            annotator_id,
            quality_annotation,
            text_corrections: None,
            audio_issues: vec![],
            confidence: 1.0,
            timestamp: chrono::Utc::now(),
            notes: None,
        }
    }

    /// Check if the annotation has high confidence
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > 0.8
    }

    /// Check if there are audio quality issues
    pub fn has_audio_issues(&self) -> bool {
        !self.audio_issues.is_empty()
    }

    /// Check if text was corrected
    pub fn has_text_corrections(&self) -> bool {
        self.text_corrections.is_some()
    }

    /// Add an audio issue
    pub fn add_audio_issue(&mut self, issue: AudioIssue) {
        self.audio_issues.push(issue);
    }

    /// Set text correction
    pub fn set_text_correction(&mut self, correction: String) {
        self.text_corrections = Some(correction);
    }

    /// Set notes
    pub fn set_notes(&mut self, notes: String) {
        self.notes = Some(notes);
    }

    /// Get annotation age in seconds
    pub fn age_seconds(&self) -> i64 {
        (chrono::Utc::now() - self.timestamp).num_seconds()
    }

    /// Check if annotation is recent (within last hour)
    pub fn is_recent(&self) -> bool {
        self.age_seconds() < 3600
    }
}

impl AudioIssue {
    /// Get the severity of the issue (0.0 = minor, 1.0 = critical)
    pub fn severity(&self) -> f32 {
        match self {
            AudioIssue::Noise => 0.6,
            AudioIssue::Clipping => 0.9,
            AudioIssue::LowVolume => 0.4,
            AudioIssue::HighVolume => 0.7,
            AudioIssue::Distortion => 0.8,
            AudioIssue::Echo => 0.5,
            AudioIssue::Reverb => 0.3,
            AudioIssue::Artifacts => 0.7,
            AudioIssue::SpeechQuality => 0.8,
            AudioIssue::Other(_) => 0.5,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &str {
        match self {
            AudioIssue::Noise => "Background noise present",
            AudioIssue::Clipping => "Audio clipping detected",
            AudioIssue::LowVolume => "Volume too low",
            AudioIssue::HighVolume => "Volume too high",
            AudioIssue::Distortion => "Audio distortion",
            AudioIssue::Echo => "Echo present",
            AudioIssue::Reverb => "Reverb present",
            AudioIssue::Artifacts => "Audio artifacts",
            AudioIssue::SpeechQuality => "Poor speech quality",
            AudioIssue::Other(desc) => desc,
        }
    }

    /// Check if this is a critical issue
    pub fn is_critical(&self) -> bool {
        self.severity() > 0.8
    }

    /// Get recommended action
    pub fn recommended_action(&self) -> &str {
        match self {
            AudioIssue::Noise => "Apply noise reduction",
            AudioIssue::Clipping => "Re-record or normalize",
            AudioIssue::LowVolume => "Increase gain",
            AudioIssue::HighVolume => "Reduce gain",
            AudioIssue::Distortion => "Check input levels",
            AudioIssue::Echo => "Apply echo cancellation",
            AudioIssue::Reverb => "Record in treated room",
            AudioIssue::Artifacts => "Check recording equipment",
            AudioIssue::SpeechQuality => "Improve recording conditions",
            AudioIssue::Other(_) => "Manual review required",
        }
    }
}

impl Default for AnnotationStatistics {
    fn default() -> Self {
        Self {
            total_annotations: 0,
            annotations_per_annotator: HashMap::new(),
            avg_annotation_time: 0.0,
            inter_annotator_agreement: 0.0,
            quality_distribution: HashMap::new(),
            common_issues: vec![],
        }
    }
}

impl AnnotationStatistics {
    /// Create new annotation statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an annotation to the statistics
    pub fn add_annotation(&mut self, result: &AnnotationResult, annotation_time: f32) {
        self.total_annotations += 1;

        // Update annotator statistics
        *self
            .annotations_per_annotator
            .entry(result.annotator_id.clone())
            .or_insert(0) += 1;

        // Update average annotation time
        let total_time =
            self.avg_annotation_time * (self.total_annotations - 1) as f32 + annotation_time;
        self.avg_annotation_time = total_time / self.total_annotations as f32;

        // Update quality distribution
        let quality_level = self.categorize_quality(&result.quality_annotation);
        *self
            .quality_distribution
            .entry(quality_level)
            .or_insert(0.0) += 1.0;

        // Update common issues
        for issue in &result.audio_issues {
            if let Some(entry) = self
                .common_issues
                .iter_mut()
                .find(|(i, _)| std::mem::discriminant(i) == std::mem::discriminant(issue))
            {
                entry.1 += 1;
            } else {
                self.common_issues.push((issue.clone(), 1));
            }
        }

        // Sort common issues by frequency
        self.common_issues.sort_by(|a, b| b.1.cmp(&a.1));
    }

    /// Get the number of unique annotators
    pub fn num_annotators(&self) -> usize {
        self.annotations_per_annotator.len()
    }

    /// Get the most active annotator
    pub fn most_active_annotator(&self) -> Option<(&String, &usize)> {
        self.annotations_per_annotator
            .iter()
            .max_by_key(|(_, count)| *count)
    }

    /// Get the most common issue
    pub fn most_common_issue(&self) -> Option<&(AudioIssue, usize)> {
        self.common_issues.first()
    }

    /// Calculate annotation rate (annotations per hour)
    pub fn annotation_rate(&self) -> f32 {
        if self.avg_annotation_time > 0.0 {
            3600.0 / self.avg_annotation_time
        } else {
            0.0
        }
    }

    /// Get quality distribution as percentages
    pub fn quality_percentages(&self) -> HashMap<String, f32> {
        let total = self.total_annotations as f32;
        if total == 0.0 {
            return HashMap::new();
        }

        self.quality_distribution
            .iter()
            .map(|(k, v)| (k.clone(), v / total * 100.0))
            .collect()
    }

    fn categorize_quality(&self, quality: &QualityMetrics) -> String {
        if let Some(overall) = quality.overall_quality {
            if overall >= 0.8 {
                "high".to_string()
            } else if overall >= 0.6 {
                "medium".to_string()
            } else {
                "low".to_string()
            }
        } else {
            "unknown".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_selection_result() {
        let indices = vec![0, 2, 4];
        let uncertainty = vec![0.8, 0.6, 0.9];
        let diversity = vec![0.5, 0.7, 0.4];
        let combined = vec![0.65, 0.65, 0.65];

        let result = ActiveSelectionResult::new(indices, uncertainty, diversity, combined);

        assert!(result.is_valid());
        assert_eq!(result.num_selected(), 3);
        assert_eq!(result.top_k(2), vec![0, 2]);
        assert_eq!(result.average_uncertainty(), 0.7666667);
        assert_eq!(result.average_diversity(), 0.53333336);
    }

    #[test]
    fn test_annotation_result() {
        let quality = QualityMetrics::default();
        let mut result =
            AnnotationResult::new("sample_1".to_string(), "annotator_1".to_string(), quality);

        assert!(!result.has_audio_issues());
        assert!(!result.has_text_corrections());
        assert!(result.is_high_confidence());
        assert!(result.is_recent());

        result.add_audio_issue(AudioIssue::Noise);
        result.set_text_correction("Corrected text".to_string());
        result.set_notes("Test notes".to_string());

        assert!(result.has_audio_issues());
        assert!(result.has_text_corrections());
    }

    #[test]
    fn test_audio_issue() {
        let clipping = AudioIssue::Clipping;
        let noise = AudioIssue::Noise;
        let reverb = AudioIssue::Reverb;

        assert!(clipping.is_critical());
        assert!(!noise.is_critical());
        assert!(!reverb.is_critical());

        assert_eq!(clipping.description(), "Audio clipping detected");
        assert_eq!(clipping.recommended_action(), "Re-record or normalize");
        assert_eq!(clipping.severity(), 0.9);
    }

    #[test]
    fn test_annotation_statistics() {
        let mut stats = AnnotationStatistics::new();

        let quality = QualityMetrics {
            overall_quality: Some(0.8),
            ..Default::default()
        };

        let mut result =
            AnnotationResult::new("sample_1".to_string(), "annotator_1".to_string(), quality);
        result.add_audio_issue(AudioIssue::Noise);

        stats.add_annotation(&result, 30.0);

        assert_eq!(stats.total_annotations, 1);
        assert_eq!(stats.num_annotators(), 1);
        assert_eq!(stats.avg_annotation_time, 30.0);
        assert_eq!(stats.annotation_rate(), 120.0);
        assert_eq!(stats.most_common_issue().unwrap().1, 1);
    }
}
