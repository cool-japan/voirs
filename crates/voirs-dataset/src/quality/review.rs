//! Manual quality review tools for dataset curation
//! 
//! This module provides tools for manual review and annotation of audio samples,
//! including interactive browsing, quality scoring, and batch approval workflows.

use crate::{DatasetSample, DatasetError, Result, QualityMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Review status for individual samples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReviewStatus {
    /// Not yet reviewed
    Pending,
    /// Approved for inclusion
    Approved,
    /// Rejected due to quality issues
    Rejected,
    /// Needs further review
    NeedsReview,
    /// Conditionally approved with notes
    Conditional,
}

impl Default for ReviewStatus {
    fn default() -> Self {
        ReviewStatus::Pending
    }
}

/// Review annotation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewAnnotation {
    /// Sample ID
    pub sample_id: String,
    /// Review status
    pub status: ReviewStatus,
    /// Quality score (0.0 - 10.0)
    pub quality_score: f32,
    /// Reviewer comments
    pub comments: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Specific quality issues identified
    pub issues: Vec<QualityIssue>,
    /// Reviewer ID
    pub reviewer_id: String,
    /// Review timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Confidence in the review (0.0 - 1.0)
    pub confidence: f32,
}

impl ReviewAnnotation {
    /// Create new review annotation
    pub fn new(sample_id: String, reviewer_id: String) -> Self {
        Self {
            sample_id,
            status: ReviewStatus::Pending,
            quality_score: 5.0,
            comments: String::new(),
            tags: Vec::new(),
            issues: Vec::new(),
            reviewer_id,
            timestamp: chrono::Utc::now(),
            confidence: 1.0,
        }
    }

    /// Set review status with timestamp update
    pub fn set_status(&mut self, status: ReviewStatus) {
        self.status = status;
        self.timestamp = chrono::Utc::now();
    }

    /// Add quality issue
    pub fn add_issue(&mut self, issue: QualityIssue) {
        self.issues.push(issue);
        self.timestamp = chrono::Utc::now();
    }

    /// Add tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
            self.timestamp = chrono::Utc::now();
        }
    }

    /// Check if approved
    pub fn is_approved(&self) -> bool {
        matches!(self.status, ReviewStatus::Approved | ReviewStatus::Conditional)
    }

    /// Check if rejected
    pub fn is_rejected(&self) -> bool {
        self.status == ReviewStatus::Rejected
    }
}

/// Specific quality issues that can be identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssue {
    /// Audio quality problems
    AudioClipping,
    AudioNoise,
    AudioDistortion,
    AudioTooQuiet,
    AudioTooLoud,
    
    /// Text-related issues
    TextMismatch,
    TextErrors,
    PronunciationIssues,
    
    /// Content issues
    InappropriateContent,
    IncompleteUtterance,
    BackgroundNoise,
    
    /// Technical issues
    FileCorruption,
    FormatIssues,
    
    /// Custom issue with description
    Custom(String),
}

/// Review session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSessionConfig {
    /// Reviewer identification
    pub reviewer_id: String,
    /// Number of samples to review in this session
    pub batch_size: usize,
    /// Minimum quality score for approval
    pub approval_threshold: f32,
    /// Whether to randomize sample order
    pub randomize_order: bool,
    /// Focus on specific quality issues
    pub focus_areas: Vec<QualityIssue>,
    /// Auto-advance after review
    pub auto_advance: bool,
}

impl Default for ReviewSessionConfig {
    fn default() -> Self {
        Self {
            reviewer_id: "default-reviewer".to_string(),
            batch_size: 50,
            approval_threshold: 6.0,
            randomize_order: true,
            focus_areas: Vec::new(),
            auto_advance: false,
        }
    }
}

/// Review session state
#[derive(Debug)]
pub struct ReviewSession {
    /// Configuration
    config: ReviewSessionConfig,
    /// Current sample queue
    sample_queue: VecDeque<String>,
    /// Current sample index
    current_index: usize,
    /// Review annotations
    annotations: HashMap<String, ReviewAnnotation>,
    /// Session start time
    start_time: chrono::DateTime<chrono::Utc>,
    /// Reviewed samples count
    reviewed_count: usize,
}

impl ReviewSession {
    /// Create new review session
    pub fn new(config: ReviewSessionConfig, sample_ids: Vec<String>) -> Self {
        let mut sample_queue: VecDeque<String> = sample_ids.into_iter().collect();
        
        if config.randomize_order {
            use rand::seq::SliceRandom;
            let mut vec: Vec<String> = sample_queue.into_iter().collect();
            vec.shuffle(&mut rand::thread_rng());
            sample_queue = vec.into_iter().collect();
        }

        Self {
            config,
            sample_queue,
            current_index: 0,
            annotations: HashMap::new(),
            start_time: chrono::Utc::now(),
            reviewed_count: 0,
        }
    }

    /// Get current sample ID
    pub fn current_sample(&self) -> Option<&String> {
        self.sample_queue.get(self.current_index)
    }

    /// Move to next sample
    pub fn next_sample(&mut self) -> Option<&String> {
        if self.current_index + 1 < self.sample_queue.len() {
            self.current_index += 1;
            self.current_sample()
        } else {
            None
        }
    }

    /// Move to previous sample
    pub fn previous_sample(&mut self) -> Option<&String> {
        if self.current_index > 0 {
            self.current_index -= 1;
            self.current_sample()
        } else {
            None
        }
    }

    /// Add review annotation
    pub fn add_annotation(&mut self, annotation: ReviewAnnotation) {
        self.annotations.insert(annotation.sample_id.clone(), annotation);
        self.reviewed_count += 1;
        
        if self.config.auto_advance {
            self.next_sample();
        }
    }

    /// Get session progress
    pub fn progress(&self) -> ReviewProgress {
        ReviewProgress {
            total_samples: self.sample_queue.len(),
            reviewed_samples: self.reviewed_count,
            current_index: self.current_index,
            approved_count: self.annotations.values()
                .filter(|a| a.is_approved())
                .count(),
            rejected_count: self.annotations.values()
                .filter(|a| a.is_rejected())
                .count(),
            session_duration: chrono::Utc::now() - self.start_time,
        }
    }

    /// Check if session is complete
    pub fn is_complete(&self) -> bool {
        self.reviewed_count >= self.config.batch_size || 
        self.current_index >= self.sample_queue.len()
    }

    /// Get all annotations
    pub fn annotations(&self) -> &HashMap<String, ReviewAnnotation> {
        &self.annotations
    }
}

/// Review progress information
#[derive(Debug, Clone)]
pub struct ReviewProgress {
    pub total_samples: usize,
    pub reviewed_samples: usize,
    pub current_index: usize,
    pub approved_count: usize,
    pub rejected_count: usize,
    pub session_duration: chrono::Duration,
}

impl ReviewProgress {
    /// Get completion percentage
    pub fn completion_percent(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            (self.reviewed_samples as f32 / self.total_samples as f32) * 100.0
        }
    }

    /// Get approval rate
    pub fn approval_rate(&self) -> f32 {
        if self.reviewed_samples == 0 {
            0.0
        } else {
            (self.approved_count as f32 / self.reviewed_samples as f32) * 100.0
        }
    }

    /// Get rejection rate
    pub fn rejection_rate(&self) -> f32 {
        if self.reviewed_samples == 0 {
            0.0
        } else {
            (self.rejected_count as f32 / self.reviewed_samples as f32) * 100.0
        }
    }

    /// Get review rate (samples per hour)
    pub fn review_rate(&self) -> f32 {
        let hours = self.session_duration.num_seconds() as f32 / 3600.0;
        if hours > 0.0 {
            self.reviewed_samples as f32 / hours
        } else {
            0.0
        }
    }
}

/// Quality reviewer for managing review sessions and annotations
pub struct QualityReviewer {
    /// Review data storage path
    storage_path: PathBuf,
    /// Current review session
    current_session: Option<ReviewSession>,
    /// Loaded annotations
    annotations: HashMap<String, ReviewAnnotation>,
}

impl QualityReviewer {
    /// Create new quality reviewer
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Self {
        Self {
            storage_path: storage_path.as_ref().to_path_buf(),
            current_session: None,
            annotations: HashMap::new(),
        }
    }

    /// Start new review session
    pub async fn start_session(
        &mut self, 
        config: ReviewSessionConfig, 
        samples: &[DatasetSample]
    ) -> Result<()> {
        let sample_ids: Vec<String> = samples.iter()
            .map(|s| s.id.clone())
            .collect();
            
        self.current_session = Some(ReviewSession::new(config, sample_ids));
        Ok(())
    }

    /// Get current session
    pub fn current_session(&self) -> Option<&ReviewSession> {
        self.current_session.as_ref()
    }

    /// Get mutable current session
    pub fn current_session_mut(&mut self) -> Option<&mut ReviewSession> {
        self.current_session.as_mut()
    }

    /// Review current sample
    pub fn review_sample(
        &mut self, 
        status: ReviewStatus, 
        quality_score: f32,
        comments: String,
        issues: Vec<QualityIssue>
    ) -> Result<()> {
        let session = self.current_session.as_mut()
            .ok_or_else(|| DatasetError::ConfigError("No active review session".to_string()))?;

        let sample_id = session.current_sample()
            .ok_or_else(|| DatasetError::ConfigError("No current sample".to_string()))?
            .clone();

        let mut annotation = ReviewAnnotation::new(sample_id, session.config.reviewer_id.clone());
        annotation.set_status(status);
        annotation.quality_score = quality_score;
        annotation.comments = comments;
        annotation.issues = issues;

        session.add_annotation(annotation.clone());
        self.annotations.insert(annotation.sample_id.clone(), annotation);

        Ok(())
    }

    /// Save annotations to storage
    pub async fn save_annotations(&self) -> Result<()> {
        let annotations_path = self.storage_path.join("annotations.json");
        
        // Ensure directory exists
        if let Some(parent) = annotations_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let json = serde_json::to_string_pretty(&self.annotations)?;
        fs::write(annotations_path, json).await?;

        Ok(())
    }

    /// Load annotations from storage
    pub async fn load_annotations(&mut self) -> Result<()> {
        let annotations_path = self.storage_path.join("annotations.json");
        
        if annotations_path.exists() {
            let json = fs::read_to_string(annotations_path).await?;
            self.annotations = serde_json::from_str(&json)?;
        }

        Ok(())
    }

    /// Generate review report
    pub fn generate_report(&self) -> ReviewReport {
        let total_annotations = self.annotations.len();
        let approved = self.annotations.values()
            .filter(|a| a.is_approved())
            .count();
        let rejected = self.annotations.values()
            .filter(|a| a.is_rejected())
            .count();
        let pending = self.annotations.values()
            .filter(|a| a.status == ReviewStatus::Pending)
            .count();

        let quality_scores: Vec<f32> = self.annotations.values()
            .map(|a| a.quality_score)
            .collect();

        let avg_quality = if !quality_scores.is_empty() {
            quality_scores.iter().sum::<f32>() / quality_scores.len() as f32
        } else {
            0.0
        };

        // Issue statistics
        let mut issue_counts = HashMap::new();
        for annotation in self.annotations.values() {
            for issue in &annotation.issues {
                let issue_name = match issue {
                    QualityIssue::AudioClipping => "Audio Clipping".to_string(),
                    QualityIssue::AudioNoise => "Audio Noise".to_string(),
                    QualityIssue::AudioDistortion => "Audio Distortion".to_string(),
                    QualityIssue::AudioTooQuiet => "Audio Too Quiet".to_string(),
                    QualityIssue::AudioTooLoud => "Audio Too Loud".to_string(),
                    QualityIssue::TextMismatch => "Text Mismatch".to_string(),
                    QualityIssue::TextErrors => "Text Errors".to_string(),
                    QualityIssue::PronunciationIssues => "Pronunciation Issues".to_string(),
                    QualityIssue::InappropriateContent => "Inappropriate Content".to_string(),
                    QualityIssue::IncompleteUtterance => "Incomplete Utterance".to_string(),
                    QualityIssue::BackgroundNoise => "Background Noise".to_string(),
                    QualityIssue::FileCorruption => "File Corruption".to_string(),
                    QualityIssue::FormatIssues => "Format Issues".to_string(),
                    QualityIssue::Custom(desc) => desc.clone(),
                };
                *issue_counts.entry(issue_name).or_insert(0) += 1;
            }
        }

        ReviewReport {
            total_samples: total_annotations,
            approved_samples: approved,
            rejected_samples: rejected,
            pending_samples: pending,
            average_quality_score: avg_quality,
            issue_distribution: issue_counts,
            annotations: self.annotations.clone(),
        }
    }

    /// Get samples by status
    pub fn samples_by_status(&self, status: ReviewStatus) -> Vec<&ReviewAnnotation> {
        self.annotations.values()
            .filter(|a| a.status == status)
            .collect()
    }

    /// Get annotations for specific sample
    pub fn get_annotation(&self, sample_id: &str) -> Option<&ReviewAnnotation> {
        self.annotations.get(sample_id)
    }

    /// Bulk approve samples meeting criteria
    pub async fn bulk_approve(&mut self, min_quality_score: f32) -> Result<usize> {
        let mut approved_count = 0;
        
        for annotation in self.annotations.values_mut() {
            if annotation.status == ReviewStatus::Pending && 
               annotation.quality_score >= min_quality_score {
                annotation.set_status(ReviewStatus::Approved);
                approved_count += 1;
            }
        }

        self.save_annotations().await?;
        Ok(approved_count)
    }

    /// Export approved samples list
    pub async fn export_approved_list<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let approved_ids: Vec<String> = self.annotations.values()
            .filter(|a| a.is_approved())
            .map(|a| a.sample_id.clone())
            .collect();

        let json = serde_json::to_string_pretty(&approved_ids)?;
        fs::write(output_path, json).await?;

        Ok(())
    }
}

/// Review report with statistics and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewReport {
    pub total_samples: usize,
    pub approved_samples: usize,
    pub rejected_samples: usize,
    pub pending_samples: usize,
    pub average_quality_score: f32,
    pub issue_distribution: HashMap<String, usize>,
    pub annotations: HashMap<String, ReviewAnnotation>,
}

impl ReviewReport {
    /// Get approval percentage
    pub fn approval_rate(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            (self.approved_samples as f32 / self.total_samples as f32) * 100.0
        }
    }

    /// Get rejection percentage
    pub fn rejection_rate(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            (self.rejected_samples as f32 / self.total_samples as f32) * 100.0
        }
    }

    /// Get most common issues
    pub fn top_issues(&self, limit: usize) -> Vec<(String, usize)> {
        let mut issues: Vec<_> = self.issue_distribution.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        issues.sort_by(|a, b| b.1.cmp(&a.1));
        issues.into_iter().take(limit).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode};
    use tempfile::TempDir;

    #[test]
    fn test_review_annotation_creation() {
        let annotation = ReviewAnnotation::new(
            "test-001".to_string(),
            "reviewer-1".to_string()
        );

        assert_eq!(annotation.sample_id, "test-001");
        assert_eq!(annotation.reviewer_id, "reviewer-1");
        assert_eq!(annotation.status, ReviewStatus::Pending);
        assert_eq!(annotation.quality_score, 5.0);
        assert!(!annotation.is_approved());
        assert!(!annotation.is_rejected());
    }

    #[test]
    fn test_review_annotation_status_changes() {
        let mut annotation = ReviewAnnotation::new(
            "test-001".to_string(),
            "reviewer-1".to_string()
        );

        annotation.set_status(ReviewStatus::Approved);
        assert!(annotation.is_approved());
        assert!(!annotation.is_rejected());

        annotation.set_status(ReviewStatus::Rejected);
        assert!(!annotation.is_approved());
        assert!(annotation.is_rejected());
    }

    #[test]
    fn test_review_session_creation() {
        let config = ReviewSessionConfig::default();
        let sample_ids = vec!["001".to_string(), "002".to_string(), "003".to_string()];
        
        let session = ReviewSession::new(config, sample_ids);
        
        assert_eq!(session.sample_queue.len(), 3);
        assert_eq!(session.current_index, 0);
        assert_eq!(session.reviewed_count, 0);
        assert!(session.current_sample().is_some());
    }

    #[test]
    fn test_review_session_navigation() {
        let config = ReviewSessionConfig {
            randomize_order: false,
            ..Default::default()
        };
        let sample_ids = vec!["001".to_string(), "002".to_string(), "003".to_string()];
        
        let mut session = ReviewSession::new(config, sample_ids);
        
        assert_eq!(session.current_sample().unwrap(), "001");
        
        session.next_sample();
        assert_eq!(session.current_sample().unwrap(), "002");
        
        session.previous_sample();
        assert_eq!(session.current_sample().unwrap(), "001");
    }

    #[test]
    fn test_review_progress() {
        let config = ReviewSessionConfig::default();
        let sample_ids = vec!["001".to_string(), "002".to_string(), "003".to_string()];
        
        let mut session = ReviewSession::new(config, sample_ids);
        
        let progress = session.progress();
        assert_eq!(progress.total_samples, 3);
        assert_eq!(progress.reviewed_samples, 0);
        assert_eq!(progress.completion_percent(), 0.0);

        // Add annotation
        let annotation = ReviewAnnotation::new("001".to_string(), "reviewer-1".to_string());
        session.add_annotation(annotation);
        
        let progress = session.progress();
        assert_eq!(progress.reviewed_samples, 1);
        assert!((progress.completion_percent() - 33.33).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_quality_reviewer_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let mut reviewer = QualityReviewer::new(temp_dir.path());

        // Create test samples
        let samples = vec![
            crate::DatasetSample::new(
                "test-001".to_string(),
                "Hello world".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
            crate::DatasetSample::new(
                "test-002".to_string(),
                "Another sample".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
        ];

        // Start review session
        let config = ReviewSessionConfig {
            randomize_order: false,  // Ensure deterministic order for testing
            ..Default::default()
        };
        reviewer.start_session(config, &samples).await.unwrap();

        // Review first sample
        reviewer.review_sample(
            ReviewStatus::Approved,
            8.0,
            "Good quality".to_string(),
            vec![]
        ).unwrap();

        // Check session progress
        let session = reviewer.current_session().unwrap();
        let progress = session.progress();
        assert_eq!(progress.reviewed_samples, 1);
        assert_eq!(progress.approved_count, 1);

        // Save and load annotations
        reviewer.save_annotations().await.unwrap();
        
        let mut reviewer2 = QualityReviewer::new(temp_dir.path());
        reviewer2.load_annotations().await.unwrap();
        
        assert_eq!(reviewer2.annotations.len(), 1);
        assert!(reviewer2.get_annotation("test-001").unwrap().is_approved());
    }

    #[test]
    fn test_review_report_generation() {
        let mut annotations = HashMap::new();
        
        let mut ann1 = ReviewAnnotation::new("001".to_string(), "reviewer-1".to_string());
        ann1.set_status(ReviewStatus::Approved);
        ann1.quality_score = 8.0;
        
        let mut ann2 = ReviewAnnotation::new("002".to_string(), "reviewer-1".to_string());
        ann2.set_status(ReviewStatus::Rejected);
        ann2.quality_score = 3.0;
        ann2.add_issue(QualityIssue::AudioNoise);
        
        annotations.insert(ann1.sample_id.clone(), ann1);
        annotations.insert(ann2.sample_id.clone(), ann2);

        let reviewer = QualityReviewer {
            storage_path: PathBuf::new(),
            current_session: None,
            annotations,
        };

        let report = reviewer.generate_report();
        assert_eq!(report.total_samples, 2);
        assert_eq!(report.approved_samples, 1);
        assert_eq!(report.rejected_samples, 1);
        assert_eq!(report.approval_rate(), 50.0);
        assert_eq!(report.average_quality_score, 5.5);
        assert_eq!(report.issue_distribution.get("Audio Noise"), Some(&1));
    }
}
