//! Dataset deduplication with perceptual similarity detection
//!
//! Provides tools for detecting and removing duplicate or highly similar samples from datasets:
//! - Exact text match detection
//! - Perceptual audio similarity detection
//! - Configurable similarity thresholds
//! - Detailed deduplication reports

use crate::{
    audio::fingerprint::{AudioFingerprint, FingerprintConfig, FingerprintIndex},
    DatasetError, DatasetSample, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Strategy for selecting which sample to keep among duplicates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateSelectionStrategy {
    /// Keep the first occurrence
    KeepFirst,
    /// Keep the last occurrence
    KeepLast,
    /// Keep the highest quality sample
    KeepHighestQuality,
    /// Keep the sample with longest audio duration
    KeepLongest,
    /// Keep the sample with shortest audio duration
    KeepShortest,
}

/// Configuration for deduplication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    /// Enable exact text matching
    pub check_exact_text: bool,
    /// Enable audio fingerprint similarity detection
    pub check_audio_similarity: bool,
    /// Similarity threshold for audio matching (0.0-1.0)
    pub audio_similarity_threshold: f32,
    /// Strategy for selecting which duplicate to keep
    pub selection_strategy: DuplicateSelectionStrategy,
    /// Fingerprint configuration for audio similarity
    pub fingerprint_config: FingerprintConfig,
    /// Case-sensitive text matching
    pub case_sensitive_text: bool,
    /// Normalize whitespace in text matching
    pub normalize_whitespace: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            check_exact_text: true,
            check_audio_similarity: true,
            audio_similarity_threshold: 0.95,
            selection_strategy: DuplicateSelectionStrategy::KeepHighestQuality,
            fingerprint_config: FingerprintConfig::for_speech(),
            case_sensitive_text: false,
            normalize_whitespace: true,
        }
    }
}

impl DeduplicationConfig {
    /// Create configuration for text-only deduplication
    pub fn text_only() -> Self {
        Self {
            check_exact_text: true,
            check_audio_similarity: false,
            ..Default::default()
        }
    }

    /// Create configuration for audio-only deduplication
    pub fn audio_only() -> Self {
        Self {
            check_exact_text: false,
            check_audio_similarity: true,
            ..Default::default()
        }
    }

    /// Create configuration for strict deduplication (high threshold)
    pub fn strict() -> Self {
        Self {
            audio_similarity_threshold: 0.98,
            ..Default::default()
        }
    }

    /// Create configuration for relaxed deduplication (lower threshold)
    pub fn relaxed() -> Self {
        Self {
            audio_similarity_threshold: 0.90,
            ..Default::default()
        }
    }
}

/// Group of duplicate samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Sample IDs in this duplicate group
    pub sample_ids: Vec<String>,
    /// Similarity scores between samples (if audio similarity was checked)
    pub similarity_scores: Vec<f32>,
    /// The ID of the sample that will be kept
    pub kept_sample_id: String,
    /// The IDs of samples that will be removed
    pub removed_sample_ids: Vec<String>,
    /// Reason for duplicate detection
    pub detection_reason: String,
}

/// Report of deduplication process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationReport {
    /// Total number of input samples
    pub total_input: usize,
    /// Number of unique samples
    pub unique_samples: usize,
    /// Number of duplicate samples removed
    pub duplicates_removed: usize,
    /// Duplicate groups found
    pub duplicate_groups: Vec<DuplicateGroup>,
    /// Statistics about duplicate types
    pub stats: DeduplicationStats,
}

/// Statistics about types of duplicates found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationStats {
    /// Number of exact text matches
    pub exact_text_matches: usize,
    /// Number of audio similarity matches
    pub audio_similarity_matches: usize,
    /// Average similarity score of audio duplicates
    pub avg_audio_similarity: f32,
}

impl DeduplicationReport {
    /// Print human-readable summary
    pub fn print(&self) {
        println!("=== Deduplication Report ===");
        println!("Total input samples: {}", self.total_input);
        println!("Unique samples: {}", self.unique_samples);
        println!("Duplicates removed: {}", self.duplicates_removed);
        println!(
            "Deduplication rate: {:.2}%",
            (self.duplicates_removed as f32 / self.total_input as f32) * 100.0
        );
        println!("\nDuplicate Statistics:");
        println!("  Exact text matches: {}", self.stats.exact_text_matches);
        println!(
            "  Audio similarity matches: {}",
            self.stats.audio_similarity_matches
        );
        if self.stats.audio_similarity_matches > 0 {
            println!(
                "  Average audio similarity: {:.3}",
                self.stats.avg_audio_similarity
            );
        }
        println!("\nDuplicate Groups: {}", self.duplicate_groups.len());
        for (idx, group) in self.duplicate_groups.iter().enumerate() {
            println!(
                "  Group {}: {} samples (keeping: {}, reason: {})",
                idx + 1,
                group.sample_ids.len(),
                group.kept_sample_id,
                group.detection_reason
            );
        }
    }
}

/// Dataset deduplicator
pub struct Deduplicator {
    config: DeduplicationConfig,
}

impl Deduplicator {
    /// Create new deduplicator with given configuration
    pub fn new(config: DeduplicationConfig) -> Self {
        Self { config }
    }

    /// Deduplicate dataset and return unique samples with report
    pub fn deduplicate(
        &self,
        samples: Vec<DatasetSample>,
    ) -> Result<(Vec<DatasetSample>, DeduplicationReport)> {
        info!("Starting deduplication of {} samples", samples.len());

        let total_input = samples.len();
        let mut duplicate_groups = Vec::new();
        let mut exact_text_matches = 0;
        let mut audio_similarity_matches = 0;
        let mut audio_similarity_sum = 0.0;

        // Track which samples to remove
        let mut removed_indices = HashSet::new();

        // Phase 1: Detect exact text duplicates
        if self.config.check_exact_text {
            debug!("Phase 1: Detecting exact text duplicates");
            let text_groups = self.find_text_duplicates(&samples);

            for group_indices in text_groups.values() {
                if group_indices.len() > 1 {
                    exact_text_matches += group_indices.len() - 1;

                    let kept_idx = self.select_best_sample(&samples, group_indices);
                    let mut removed = Vec::new();

                    for &idx in group_indices {
                        if idx != kept_idx {
                            removed_indices.insert(idx);
                            removed.push(samples[idx].id.clone());
                        }
                    }

                    duplicate_groups.push(DuplicateGroup {
                        sample_ids: group_indices
                            .iter()
                            .map(|&i| samples[i].id.clone())
                            .collect(),
                        similarity_scores: vec![1.0; group_indices.len()],
                        kept_sample_id: samples[kept_idx].id.clone(),
                        removed_sample_ids: removed,
                        detection_reason: "Exact text match".to_string(),
                    });
                }
            }
        }

        // Phase 2: Detect audio similarity duplicates
        if self.config.check_audio_similarity {
            debug!("Phase 2: Detecting audio similarity duplicates");

            // Build fingerprint index for remaining samples
            let mut fp_index = FingerprintIndex::new(self.config.fingerprint_config.clone());
            let mut id_to_idx = HashMap::new();

            for (idx, sample) in samples.iter().enumerate() {
                if !removed_indices.contains(&idx) {
                    if let Ok(()) = fp_index.add(idx.to_string(), &sample.audio) {
                        id_to_idx.insert(idx.to_string(), idx);
                    } else {
                        warn!(
                            "Failed to create fingerprint for sample {}: {}",
                            sample.id, "Audio processing error"
                        );
                    }
                }
            }

            // Find audio similarity duplicates
            let audio_groups = self.find_audio_duplicates(&fp_index, &id_to_idx);

            for (group_indices, avg_similarity) in audio_groups {
                if group_indices.len() > 1 {
                    audio_similarity_matches += group_indices.len() - 1;
                    audio_similarity_sum += avg_similarity;

                    let kept_idx = self.select_best_sample(&samples, &group_indices);
                    let mut removed = Vec::new();

                    for &idx in &group_indices {
                        if idx != kept_idx {
                            removed_indices.insert(idx);
                            removed.push(samples[idx].id.clone());
                        }
                    }

                    duplicate_groups.push(DuplicateGroup {
                        sample_ids: group_indices
                            .iter()
                            .map(|&i| samples[i].id.clone())
                            .collect(),
                        similarity_scores: vec![avg_similarity; group_indices.len()],
                        kept_sample_id: samples[kept_idx].id.clone(),
                        removed_sample_ids: removed,
                        detection_reason: format!("Audio similarity ({:.3})", avg_similarity),
                    });
                }
            }
        }

        // Reconstruct dataset with only unique samples
        let unique_samples: Vec<DatasetSample> = samples
            .into_iter()
            .enumerate()
            .filter_map(|(idx, sample)| {
                if !removed_indices.contains(&idx) {
                    Some(sample)
                } else {
                    None
                }
            })
            .collect();

        let avg_audio_similarity = if audio_similarity_matches > 0 {
            audio_similarity_sum / audio_similarity_matches as f32
        } else {
            0.0
        };

        let report = DeduplicationReport {
            total_input,
            unique_samples: unique_samples.len(),
            duplicates_removed: total_input - unique_samples.len(),
            duplicate_groups,
            stats: DeduplicationStats {
                exact_text_matches,
                audio_similarity_matches,
                avg_audio_similarity,
            },
        };

        info!(
            "Deduplication complete: {} unique samples, {} duplicates removed",
            report.unique_samples, report.duplicates_removed
        );

        Ok((unique_samples, report))
    }

    /// Find text duplicate groups
    fn find_text_duplicates(&self, samples: &[DatasetSample]) -> HashMap<String, Vec<usize>> {
        let mut text_map: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, sample) in samples.iter().enumerate() {
            let normalized_text = self.normalize_text(&sample.text);
            text_map.entry(normalized_text).or_default().push(idx);
        }

        // Filter to only groups with duplicates
        text_map.retain(|_, indices| indices.len() > 1);

        text_map
    }

    /// Normalize text for comparison
    fn normalize_text(&self, text: &str) -> String {
        let mut normalized = text.to_string();

        if !self.config.case_sensitive_text {
            normalized = normalized.to_lowercase();
        }

        if self.config.normalize_whitespace {
            normalized = normalized
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ");
        }

        normalized
    }

    /// Find audio similarity duplicate groups
    fn find_audio_duplicates(
        &self,
        fp_index: &FingerprintIndex,
        id_to_idx: &HashMap<String, usize>,
    ) -> Vec<(Vec<usize>, f32)> {
        let mut groups = Vec::new();
        let mut processed = HashSet::new();

        for (id_str, &idx) in id_to_idx {
            if processed.contains(&idx) {
                continue;
            }

            // Get fingerprint for this sample
            if let Some(fp) = fp_index.get_fingerprint(id_str) {
                // Find similar samples
                let similar = fp_index.find_similar(fp, self.config.audio_similarity_threshold);

                if similar.len() > 1 {
                    let mut group_indices = Vec::new();
                    let mut similarity_sum = 0.0;

                    for (similar_id, similarity) in similar {
                        if let Some(&similar_idx) = id_to_idx.get(&similar_id) {
                            group_indices.push(similar_idx);
                            similarity_sum += similarity;
                            processed.insert(similar_idx);
                        }
                    }

                    if group_indices.len() > 1 {
                        let avg_similarity = similarity_sum / group_indices.len() as f32;
                        groups.push((group_indices, avg_similarity));
                    }
                }
            }
        }

        groups
    }

    /// Select best sample from a group based on configured strategy
    fn select_best_sample(&self, samples: &[DatasetSample], indices: &[usize]) -> usize {
        if indices.is_empty() {
            return 0;
        }

        match self.config.selection_strategy {
            DuplicateSelectionStrategy::KeepFirst => indices[0],
            DuplicateSelectionStrategy::KeepLast => *indices.last().unwrap_or(&indices[0]),
            DuplicateSelectionStrategy::KeepHighestQuality => {
                let mut best_idx = indices[0];
                let mut best_quality = samples[best_idx].quality.overall_quality.unwrap_or(0.0);

                for &idx in indices.iter().skip(1) {
                    let quality = samples[idx].quality.overall_quality.unwrap_or(0.0);
                    if quality > best_quality {
                        best_quality = quality;
                        best_idx = idx;
                    }
                }

                best_idx
            }
            DuplicateSelectionStrategy::KeepLongest => {
                let mut best_idx = indices[0];
                let mut best_duration = samples[best_idx].duration();

                for &idx in indices.iter().skip(1) {
                    let duration = samples[idx].duration();
                    if duration > best_duration {
                        best_duration = duration;
                        best_idx = idx;
                    }
                }

                best_idx
            }
            DuplicateSelectionStrategy::KeepShortest => {
                let mut best_idx = indices[0];
                let mut best_duration = samples[best_idx].duration();

                for &idx in indices.iter().skip(1) {
                    let duration = samples[idx].duration();
                    if duration < best_duration {
                        best_duration = duration;
                        best_idx = idx;
                    }
                }

                best_idx
            }
        }
    }
}

impl Default for Deduplicator {
    fn default() -> Self {
        Self::new(DeduplicationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, QualityMetrics};

    fn create_test_sample(id: &str, text: &str, quality: f32, duration: f32) -> DatasetSample {
        let sample_rate = 22050;
        let num_samples = (duration * sample_rate as f32) as usize;
        let samples = vec![0.1; num_samples];
        let audio = AudioData::new(samples, sample_rate, 1);

        DatasetSample::new(id.to_string(), text.to_string(), audio, LanguageCode::EnUs)
            .with_quality(QualityMetrics {
                overall_quality: Some(quality),
                snr: Some(20.0),
                clipping: Some(0.01),
                dynamic_range: Some(40.0),
                spectral_quality: Some(0.8),
            })
    }

    #[test]
    fn test_text_deduplication() {
        let deduplicator = Deduplicator::new(DeduplicationConfig::text_only());

        let samples = vec![
            create_test_sample("001", "Hello world", 0.9, 1.0),
            create_test_sample("002", "Hello world", 0.85, 1.0), // Duplicate
            create_test_sample("003", "Goodbye world", 0.9, 1.0),
        ];

        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 2);
        assert_eq!(report.duplicates_removed, 1);
        assert_eq!(report.stats.exact_text_matches, 1);
    }

    #[test]
    fn test_keep_highest_quality() {
        let config = DeduplicationConfig {
            selection_strategy: DuplicateSelectionStrategy::KeepHighestQuality,
            ..DeduplicationConfig::text_only()
        };

        let deduplicator = Deduplicator::new(config);

        let samples = vec![
            create_test_sample("001", "Same text", 0.7, 1.0),
            create_test_sample("002", "Same text", 0.95, 1.0), // Highest quality
            create_test_sample("003", "Same text", 0.8, 1.0),
        ];

        let (unique, _) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0].id, "002"); // Kept the highest quality
    }

    #[test]
    fn test_no_duplicates() {
        // Use text-only deduplication to avoid audio similarity false positives
        let deduplicator = Deduplicator::new(DeduplicationConfig::text_only());

        let samples = vec![
            create_test_sample("001", "First sample", 0.9, 1.0),
            create_test_sample("002", "Second sample", 0.9, 1.0),
            create_test_sample("003", "Third sample", 0.9, 1.0),
        ];

        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 3);
        assert_eq!(report.duplicates_removed, 0);
    }

    #[test]
    fn test_case_insensitive() {
        let config = DeduplicationConfig {
            case_sensitive_text: false,
            ..DeduplicationConfig::text_only()
        };

        let deduplicator = Deduplicator::new(config);

        let samples = vec![
            create_test_sample("001", "Hello World", 0.9, 1.0),
            create_test_sample("002", "hello world", 0.9, 1.0), // Same (case-insensitive)
        ];

        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 1);
        assert_eq!(report.duplicates_removed, 1);
    }

    #[test]
    fn test_whitespace_normalization() {
        let config = DeduplicationConfig {
            normalize_whitespace: true,
            ..DeduplicationConfig::text_only()
        };

        let deduplicator = Deduplicator::new(config);

        let samples = vec![
            create_test_sample("001", "Hello  world", 0.9, 1.0),
            create_test_sample("002", "Hello    world", 0.9, 1.0), // Same (normalized)
        ];

        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 1);
        assert_eq!(report.duplicates_removed, 1);
    }

    #[test]
    fn test_keep_first_strategy() {
        let config = DeduplicationConfig {
            selection_strategy: DuplicateSelectionStrategy::KeepFirst,
            ..DeduplicationConfig::text_only()
        };

        let deduplicator = Deduplicator::new(config);

        let samples = vec![
            create_test_sample("001", "Duplicate", 0.9, 1.0),
            create_test_sample("002", "Duplicate", 0.95, 1.0),
        ];

        let (unique, _) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0].id, "001"); // Kept first
    }

    #[test]
    fn test_keep_last_strategy() {
        let config = DeduplicationConfig {
            selection_strategy: DuplicateSelectionStrategy::KeepLast,
            ..DeduplicationConfig::text_only()
        };

        let deduplicator = Deduplicator::new(config);

        let samples = vec![
            create_test_sample("001", "Duplicate", 0.9, 1.0),
            create_test_sample("002", "Duplicate", 0.95, 1.0),
        ];

        let (unique, _) = deduplicator.deduplicate(samples).unwrap();

        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0].id, "002"); // Kept last
    }
}
