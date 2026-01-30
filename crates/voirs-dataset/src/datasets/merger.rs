//! Dataset merger and combiner utilities
//!
//! Provides functionality for intelligently combining multiple datasets with:
//! - Conflict resolution strategies
//! - Deduplication options
//! - Metadata preservation
//! - Quality-based filtering
//! - Balanced sampling across sources

use crate::{
    DatasetError, DatasetItem, DatasetSample, LanguageCode, QualityMetrics, Result, SpeakerInfo,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Strategy for resolving ID conflicts when merging datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdConflictStrategy {
    /// Rename conflicting IDs by appending dataset source prefix
    RenameWithPrefix,
    /// Skip samples with conflicting IDs
    SkipDuplicates,
    /// Keep the first occurrence
    KeepFirst,
    /// Keep the highest quality sample
    KeepHighestQuality,
    /// Fail on conflicts
    Error,
}

/// Strategy for handling speaker information conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeakerMergeStrategy {
    /// Preserve original speaker IDs (may cause conflicts)
    PreserveOriginal,
    /// Prefix speaker IDs with dataset name
    PrefixWithDataset,
    /// Merge speakers with same ID
    MergeSameId,
    /// Renumber all speakers sequentially
    Renumber,
}

/// Strategy for quality filtering during merge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityFilterStrategy {
    /// No quality filtering
    None,
    /// Filter out samples below minimum overall quality score
    MinimumQuality { threshold: f32 },
    /// Filter out samples below minimum SNR
    MinimumSnr { threshold: f32 },
    /// Combined quality criteria
    Combined {
        min_quality: Option<f32>,
        min_snr: Option<f32>,
        max_clipping: Option<f32>,
    },
}

/// Configuration for dataset merging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Strategy for resolving ID conflicts
    pub id_conflict_strategy: IdConflictStrategy,
    /// Strategy for handling speaker information
    pub speaker_strategy: SpeakerMergeStrategy,
    /// Quality filtering strategy
    pub quality_filter: QualityFilterStrategy,
    /// Whether to deduplicate identical text entries
    pub deduplicate_text: bool,
    /// Whether to preserve source dataset information in metadata
    pub preserve_source_info: bool,
    /// Maximum samples per dataset (for balanced sampling)
    pub max_samples_per_dataset: Option<usize>,
    /// Languages to include (None = all)
    pub language_filter: Option<Vec<LanguageCode>>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            id_conflict_strategy: IdConflictStrategy::RenameWithPrefix,
            speaker_strategy: SpeakerMergeStrategy::PrefixWithDataset,
            quality_filter: QualityFilterStrategy::None,
            deduplicate_text: false,
            preserve_source_info: true,
            max_samples_per_dataset: None,
            language_filter: None,
        }
    }
}

impl MergeConfig {
    /// Create configuration with quality filtering
    pub fn with_quality_filter(min_quality: f32) -> Self {
        Self {
            quality_filter: QualityFilterStrategy::MinimumQuality {
                threshold: min_quality,
            },
            ..Default::default()
        }
    }

    /// Create configuration for balanced merging
    pub fn balanced(max_samples_per_dataset: usize) -> Self {
        Self {
            max_samples_per_dataset: Some(max_samples_per_dataset),
            ..Default::default()
        }
    }

    /// Create configuration for language-specific merging
    pub fn language_filter(languages: Vec<LanguageCode>) -> Self {
        Self {
            language_filter: Some(languages),
            ..Default::default()
        }
    }
}

/// Dataset merger for combining multiple datasets
pub struct DatasetMerger {
    config: MergeConfig,
    merged_samples: Vec<DatasetSample>,
    source_stats: HashMap<String, SourceStats>,
    id_map: HashMap<String, String>,
    speaker_map: HashMap<String, String>,
    text_hashes: HashSet<u64>,
}

/// Statistics for a source dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStats {
    /// Name of source dataset
    pub name: String,
    /// Number of samples from this source
    pub sample_count: usize,
    /// Number of samples filtered out
    pub filtered_count: usize,
    /// Number of duplicate samples skipped
    pub duplicate_count: usize,
    /// Average quality of samples from this source
    pub average_quality: f32,
}

impl DatasetMerger {
    /// Create a new dataset merger with given configuration
    pub fn new(config: MergeConfig) -> Self {
        Self {
            config,
            merged_samples: Vec::new(),
            source_stats: HashMap::new(),
            id_map: HashMap::new(),
            speaker_map: HashMap::new(),
            text_hashes: HashSet::new(),
        }
    }

    /// Add a dataset to be merged
    pub fn add_dataset(
        &mut self,
        name: impl Into<String>,
        samples: Vec<DatasetSample>,
    ) -> Result<()> {
        let name = name.into();
        info!("Adding dataset '{}' with {} samples", name, samples.len());

        let mut stats = SourceStats {
            name: name.clone(),
            sample_count: 0,
            filtered_count: 0,
            duplicate_count: 0,
            average_quality: 0.0,
        };

        let mut quality_sum = 0.0;
        let mut quality_count = 0;

        // Apply max samples limit if configured
        let samples_to_process = if let Some(max_samples) = self.config.max_samples_per_dataset {
            if samples.len() > max_samples {
                info!(
                    "Limiting dataset '{}' from {} to {} samples",
                    name,
                    samples.len(),
                    max_samples
                );
                samples.into_iter().take(max_samples).collect()
            } else {
                samples
            }
        } else {
            samples
        };

        for mut sample in samples_to_process {
            // Apply language filter
            if let Some(ref lang_filter) = self.config.language_filter {
                if !lang_filter.contains(&sample.language) {
                    stats.filtered_count += 1;
                    continue;
                }
            }

            // Apply quality filter
            if !self.meets_quality_criteria(&sample) {
                stats.filtered_count += 1;
                continue;
            }

            // Deduplicate by text hash if configured
            if self.config.deduplicate_text {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                sample.text.hash(&mut hasher);
                let text_hash = hasher.finish();

                if self.text_hashes.contains(&text_hash) {
                    stats.duplicate_count += 1;
                    debug!("Skipping duplicate text: {}", sample.text);
                    continue;
                }
                self.text_hashes.insert(text_hash);
            }

            // Handle ID conflicts
            let new_id = self.resolve_id_conflict(&name, &sample.id, &sample)?;
            if new_id.is_empty() {
                // Sample was skipped due to conflict resolution
                stats.duplicate_count += 1;
                continue;
            }
            sample.id = new_id;

            // Handle speaker merging
            if let Some(ref mut speaker_info) = sample.speaker {
                let new_speaker_id = self.resolve_speaker_id(&name, &speaker_info.id);
                speaker_info.id = new_speaker_id;
            }

            // Preserve source information in metadata if configured
            if self.config.preserve_source_info {
                sample.metadata.insert(
                    "source_dataset".to_string(),
                    serde_json::Value::String(name.clone()),
                );
            }

            // Track quality statistics
            if let Some(quality) = sample.quality.overall_quality {
                quality_sum += quality;
                quality_count += 1;
            }

            self.merged_samples.push(sample);
            stats.sample_count += 1;
        }

        // Calculate average quality
        if quality_count > 0 {
            stats.average_quality = quality_sum / quality_count as f32;
        }

        info!(
            "Dataset '{}': {} samples added, {} filtered, {} duplicates",
            name, stats.sample_count, stats.filtered_count, stats.duplicate_count
        );

        self.source_stats.insert(name, stats);
        Ok(())
    }

    /// Check if sample meets quality criteria
    fn meets_quality_criteria(&self, sample: &DatasetSample) -> bool {
        match &self.config.quality_filter {
            QualityFilterStrategy::None => true,
            QualityFilterStrategy::MinimumQuality { threshold } => {
                sample.quality.overall_quality.unwrap_or(0.0) >= *threshold
            }
            QualityFilterStrategy::MinimumSnr { threshold } => {
                sample.quality.snr.unwrap_or(0.0) >= *threshold
            }
            QualityFilterStrategy::Combined {
                min_quality,
                min_snr,
                max_clipping,
            } => {
                let quality_ok = min_quality
                    .map(|threshold| sample.quality.overall_quality.unwrap_or(0.0) >= threshold)
                    .unwrap_or(true);

                let snr_ok = min_snr
                    .map(|threshold| sample.quality.snr.unwrap_or(0.0) >= threshold)
                    .unwrap_or(true);

                let clipping_ok = max_clipping
                    .map(|threshold| sample.quality.clipping.unwrap_or(1.0) <= threshold)
                    .unwrap_or(true);

                quality_ok && snr_ok && clipping_ok
            }
        }
    }

    /// Resolve ID conflict according to configured strategy
    fn resolve_id_conflict(
        &mut self,
        dataset_name: &str,
        id: &str,
        sample: &DatasetSample,
    ) -> Result<String> {
        match self.config.id_conflict_strategy {
            IdConflictStrategy::RenameWithPrefix => {
                let new_id = format!("{}_{}", dataset_name, id);
                if self.id_map.contains_key(&new_id) {
                    // Even with prefix, there's a conflict - make it unique
                    let mut counter = 1;
                    loop {
                        let candidate = format!("{}_{}_{}", dataset_name, id, counter);
                        if !self.id_map.contains_key(&candidate) {
                            self.id_map
                                .insert(candidate.clone(), dataset_name.to_string());
                            return Ok(candidate);
                        }
                        counter += 1;
                    }
                }
                self.id_map.insert(new_id.clone(), dataset_name.to_string());
                Ok(new_id)
            }
            IdConflictStrategy::SkipDuplicates => {
                if self.id_map.contains_key(id) {
                    debug!("Skipping duplicate ID: {}", id);
                    Ok(String::new()) // Signal to skip this sample
                } else {
                    self.id_map.insert(id.to_string(), dataset_name.to_string());
                    Ok(id.to_string())
                }
            }
            IdConflictStrategy::KeepFirst => {
                if self.id_map.contains_key(id) {
                    debug!("Keeping first occurrence of ID: {}", id);
                    Ok(String::new()) // Skip this sample
                } else {
                    self.id_map.insert(id.to_string(), dataset_name.to_string());
                    Ok(id.to_string())
                }
            }
            IdConflictStrategy::KeepHighestQuality => {
                if let Some(existing_dataset) = self.id_map.get(id) {
                    // Find existing sample with this ID
                    if let Some(existing_idx) = self.merged_samples.iter().position(|s| s.id == id)
                    {
                        let existing_quality = self.merged_samples[existing_idx]
                            .quality
                            .overall_quality
                            .unwrap_or(0.0);
                        let new_quality = sample.quality.overall_quality.unwrap_or(0.0);

                        if new_quality > existing_quality {
                            // Replace existing sample
                            warn!(
                                "Replacing sample {} from {} (quality: {}) with sample from {} (quality: {})",
                                id, existing_dataset, existing_quality, dataset_name, new_quality
                            );
                            self.merged_samples.remove(existing_idx);
                            self.id_map.insert(id.to_string(), dataset_name.to_string());
                            Ok(id.to_string())
                        } else {
                            debug!("Keeping higher quality sample for ID: {}", id);
                            Ok(String::new()) // Skip this sample
                        }
                    } else {
                        // Shouldn't happen, but handle gracefully
                        Ok(id.to_string())
                    }
                } else {
                    self.id_map.insert(id.to_string(), dataset_name.to_string());
                    Ok(id.to_string())
                }
            }
            IdConflictStrategy::Error => {
                if self.id_map.contains_key(id) {
                    Err(DatasetError::ConfigError(format!(
                        "ID conflict detected for '{}' in dataset '{}'",
                        id, dataset_name
                    )))
                } else {
                    self.id_map.insert(id.to_string(), dataset_name.to_string());
                    Ok(id.to_string())
                }
            }
        }
    }

    /// Resolve speaker ID according to configured strategy
    fn resolve_speaker_id(&mut self, dataset_name: &str, speaker_id: &str) -> String {
        match self.config.speaker_strategy {
            SpeakerMergeStrategy::PreserveOriginal => speaker_id.to_string(),
            SpeakerMergeStrategy::PrefixWithDataset => {
                format!("{}_{}", dataset_name, speaker_id)
            }
            SpeakerMergeStrategy::MergeSameId => speaker_id.to_string(),
            SpeakerMergeStrategy::Renumber => {
                if let Some(mapped_id) = self.speaker_map.get(speaker_id) {
                    mapped_id.clone()
                } else {
                    let new_id = format!("speaker_{:04}", self.speaker_map.len() + 1);
                    self.speaker_map
                        .insert(speaker_id.to_string(), new_id.clone());
                    new_id
                }
            }
        }
    }

    /// Get the merged dataset samples
    pub fn into_samples(self) -> Vec<DatasetSample> {
        info!(
            "Merge complete: {} total samples from {} source datasets",
            self.merged_samples.len(),
            self.source_stats.len()
        );
        self.merged_samples
    }

    /// Get statistics for all source datasets
    pub fn source_statistics(&self) -> &HashMap<String, SourceStats> {
        &self.source_stats
    }

    /// Get merged samples as reference
    pub fn samples(&self) -> &[DatasetSample] {
        &self.merged_samples
    }

    /// Get total sample count
    pub fn sample_count(&self) -> usize {
        self.merged_samples.len()
    }

    /// Get merge summary report
    pub fn summary(&self) -> MergeSummary {
        let total_samples = self.merged_samples.len();
        let total_filtered: usize = self.source_stats.values().map(|s| s.filtered_count).sum();
        let total_duplicates: usize = self.source_stats.values().map(|s| s.duplicate_count).sum();

        // Calculate language distribution
        let mut language_distribution = HashMap::new();
        for sample in &self.merged_samples {
            *language_distribution.entry(sample.language).or_insert(0) += 1;
        }

        // Calculate speaker distribution
        let mut speaker_distribution = HashMap::new();
        for sample in &self.merged_samples {
            if let Some(ref speaker) = sample.speaker {
                *speaker_distribution.entry(speaker.id.clone()).or_insert(0) += 1;
            }
        }

        // Calculate average quality
        let quality_samples: Vec<f32> = self
            .merged_samples
            .iter()
            .filter_map(|s| s.quality.overall_quality)
            .collect();
        let average_quality = if !quality_samples.is_empty() {
            quality_samples.iter().sum::<f32>() / quality_samples.len() as f32
        } else {
            0.0
        };

        MergeSummary {
            total_samples,
            total_filtered,
            total_duplicates,
            source_count: self.source_stats.len(),
            language_distribution,
            speaker_count: speaker_distribution.len(),
            average_quality,
            source_stats: self.source_stats.clone(),
        }
    }
}

impl Default for DatasetMerger {
    fn default() -> Self {
        Self::new(MergeConfig::default())
    }
}

/// Summary of merge operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSummary {
    /// Total samples in merged dataset
    pub total_samples: usize,
    /// Total samples filtered out
    pub total_filtered: usize,
    /// Total duplicate samples skipped
    pub total_duplicates: usize,
    /// Number of source datasets
    pub source_count: usize,
    /// Distribution of samples by language
    pub language_distribution: HashMap<LanguageCode, usize>,
    /// Number of unique speakers
    pub speaker_count: usize,
    /// Average quality score
    pub average_quality: f32,
    /// Statistics per source dataset
    pub source_stats: HashMap<String, SourceStats>,
}

impl MergeSummary {
    /// Print human-readable summary
    pub fn print(&self) {
        println!("=== Dataset Merge Summary ===");
        println!("Total samples: {}", self.total_samples);
        println!("Filtered out: {}", self.total_filtered);
        println!("Duplicates skipped: {}", self.total_duplicates);
        println!("Source datasets: {}", self.source_count);
        println!("Unique speakers: {}", self.speaker_count);
        println!("Average quality: {:.3}", self.average_quality);
        println!("\nLanguage Distribution:");
        for (lang, count) in &self.language_distribution {
            println!("  {:?}: {}", lang, count);
        }
        println!("\nSource Statistics:");
        for (name, stats) in &self.source_stats {
            println!(
                "  {}: {} samples (avg quality: {:.3})",
                name, stats.sample_count, stats.average_quality
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode};

    fn create_test_sample(id: &str, text: &str, lang: LanguageCode, quality: f32) -> DatasetSample {
        let audio = AudioData::silence(1.0, 22050, 1);
        DatasetSample::new(id.to_string(), text.to_string(), audio, lang).with_quality(
            QualityMetrics {
                overall_quality: Some(quality),
                snr: Some(20.0),
                clipping: Some(0.01),
                dynamic_range: Some(40.0),
                spectral_quality: Some(0.8),
            },
        )
    }

    #[test]
    fn test_merger_basic() {
        let mut merger = DatasetMerger::default();

        let samples1 = vec![
            create_test_sample("001", "Hello world", LanguageCode::EnUs, 0.9),
            create_test_sample("002", "Good morning", LanguageCode::EnUs, 0.85),
        ];

        let samples2 = vec![
            create_test_sample("003", "こんにちは", LanguageCode::Ja, 0.88),
            create_test_sample("004", "おはようございます", LanguageCode::Ja, 0.92),
        ];

        merger.add_dataset("dataset1", samples1).unwrap();
        merger.add_dataset("dataset2", samples2).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn test_id_conflict_rename() {
        let config = MergeConfig {
            id_conflict_strategy: IdConflictStrategy::RenameWithPrefix,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let samples1 = vec![create_test_sample(
            "001",
            "Sample 1",
            LanguageCode::EnUs,
            0.9,
        )];
        let samples2 = vec![create_test_sample(
            "001",
            "Sample 2",
            LanguageCode::EnUs,
            0.85,
        )];

        merger.add_dataset("ds1", samples1).unwrap();
        merger.add_dataset("ds2", samples2).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 2);
        assert!(merged[0].id.starts_with("ds1_"));
        assert!(merged[1].id.starts_with("ds2_"));
    }

    #[test]
    fn test_quality_filter() {
        let config = MergeConfig::with_quality_filter(0.9);
        let mut merger = DatasetMerger::new(config);

        let samples = vec![
            create_test_sample("001", "High quality", LanguageCode::EnUs, 0.95),
            create_test_sample("002", "Low quality", LanguageCode::EnUs, 0.7),
            create_test_sample("003", "Medium quality", LanguageCode::EnUs, 0.85),
        ];

        merger.add_dataset("test", samples).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 1); // Only high quality sample
        assert_eq!(merged[0].id, "test_001");
    }

    #[test]
    fn test_language_filter() {
        let config = MergeConfig::language_filter(vec![LanguageCode::EnUs]);
        let mut merger = DatasetMerger::new(config);

        let samples = vec![
            create_test_sample("001", "English", LanguageCode::EnUs, 0.9),
            create_test_sample("002", "日本語", LanguageCode::Ja, 0.9),
            create_test_sample("003", "English again", LanguageCode::EnUs, 0.9),
        ];

        merger.add_dataset("test", samples).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 2); // Only English samples
    }

    #[test]
    fn test_balanced_merge() {
        let config = MergeConfig::balanced(2);
        let mut merger = DatasetMerger::new(config);

        let samples1 = vec![
            create_test_sample("001", "Sample 1", LanguageCode::EnUs, 0.9),
            create_test_sample("002", "Sample 2", LanguageCode::EnUs, 0.9),
            create_test_sample("003", "Sample 3", LanguageCode::EnUs, 0.9),
        ];

        let samples2 = vec![
            create_test_sample("004", "Sample 4", LanguageCode::Ja, 0.9),
            create_test_sample("005", "Sample 5", LanguageCode::Ja, 0.9),
        ];

        merger.add_dataset("large", samples1).unwrap();
        merger.add_dataset("small", samples2).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 4); // 2 from each dataset
    }

    #[test]
    fn test_deduplicate_text() {
        let config = MergeConfig {
            deduplicate_text: true,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let samples = vec![
            create_test_sample("001", "Duplicate text", LanguageCode::EnUs, 0.9),
            create_test_sample("002", "Duplicate text", LanguageCode::EnUs, 0.85),
            create_test_sample("003", "Unique text", LanguageCode::EnUs, 0.9),
        ];

        merger.add_dataset("test", samples).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 2); // Two unique texts
    }

    #[test]
    fn test_merge_summary() {
        let mut merger = DatasetMerger::default();

        let samples = vec![
            create_test_sample("001", "Sample 1", LanguageCode::EnUs, 0.9),
            create_test_sample("002", "Sample 2", LanguageCode::Ja, 0.85),
        ];

        merger.add_dataset("test", samples).unwrap();

        let summary = merger.summary();
        assert_eq!(summary.total_samples, 2);
        assert_eq!(summary.source_count, 1);
        assert_eq!(summary.language_distribution.len(), 2);
        assert!(summary.average_quality > 0.0);
    }

    #[test]
    fn test_keep_highest_quality() {
        let config = MergeConfig {
            id_conflict_strategy: IdConflictStrategy::KeepHighestQuality,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let samples1 = vec![create_test_sample(
            "001",
            "Low quality",
            LanguageCode::EnUs,
            0.7,
        )];
        let samples2 = vec![create_test_sample(
            "001",
            "High quality",
            LanguageCode::EnUs,
            0.95,
        )];

        merger.add_dataset("ds1", samples1).unwrap();
        merger.add_dataset("ds2", samples2).unwrap();

        let merged = merger.into_samples();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "High quality");
        assert_eq!(merged[0].quality.overall_quality.unwrap(), 0.95);
    }
}
