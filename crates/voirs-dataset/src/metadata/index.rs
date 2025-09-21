//! Dataset indexing system for fast sample lookup
//!
//! Provides efficient multi-field indexing with query optimization
//! and persistence capabilities.

use crate::{DatasetError, Result};
use ordered_float;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Dataset index for fast sample lookup
#[derive(Debug)]
pub struct DatasetIndex {
    /// Primary index by sample ID
    primary_index: RwLock<HashMap<String, IndexEntry>>,
    /// Secondary indexes for various fields
    secondary_indexes: RwLock<HashMap<String, BTreeMap<String, HashSet<String>>>>,
    /// Range indexes for numerical fields
    range_indexes: RwLock<HashMap<String, BTreeMap<OrderedValue, HashSet<String>>>>,
    /// Index configuration
    config: IndexConfig,
    /// Index metadata
    metadata: RwLock<IndexMetadata>,
}

/// Index entry containing sample metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Sample ID
    pub id: String,
    /// Audio file path
    pub audio_path: PathBuf,
    /// Text content
    pub text: String,
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Language code
    pub language: String,
    /// Speaker ID
    pub speaker_id: Option<String>,
    /// Quality score
    pub quality_score: Option<f32>,
    /// Indexed fields
    pub indexed_fields: HashMap<String, IndexValue>,
}

/// Value types for indexing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<String>),
}

/// Ordered value for range indexing
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderedValue {
    Number(ordered_float::OrderedFloat<f64>),
    String(String),
}

/// Index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Total indexed samples
    pub total_samples: usize,
    /// Index version
    pub version: String,
    /// Indexed fields
    pub indexed_fields: Vec<String>,
}

/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Fields to index
    pub indexed_fields: Vec<String>,
    /// Enable range indexing for numerical fields
    pub enable_range_indexing: bool,
    /// Maximum entries per index
    pub max_entries: Option<usize>,
    /// Cache frequently accessed entries
    pub enable_caching: bool,
    /// Auto-optimize index structure
    pub auto_optimize: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            indexed_fields: vec![
                "language".to_string(),
                "speaker_id".to_string(),
                "duration".to_string(),
                "quality_score".to_string(),
            ],
            enable_range_indexing: true,
            max_entries: None,
            enable_caching: true,
            auto_optimize: false,
        }
    }
}

/// Query builder for searching the index
#[derive(Debug, Clone, Default)]
pub struct IndexQuery {
    /// Field equality filters
    pub equals: HashMap<String, IndexValue>,
    /// Range filters (field -> (min, max))
    pub ranges: HashMap<String, (Option<f64>, Option<f64>)>,
    /// Text search filters
    pub text_contains: HashMap<String, String>,
    /// Array contains filters
    pub array_contains: HashMap<String, String>,
    /// Result limit
    pub limit: Option<usize>,
    /// Result offset
    pub offset: Option<usize>,
    /// Sort field and direction
    pub sort_by: Option<(String, SortDirection)>,
}

/// Sort direction
#[derive(Debug, Clone)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Index builder for creating dataset indexes
pub struct IndexBuilder {
    config: IndexConfig,
}

impl IndexBuilder {
    /// Create a new index builder
    pub fn new() -> Self {
        Self {
            config: IndexConfig::default(),
        }
    }

    /// Create index builder with custom configuration
    pub fn with_config(config: IndexConfig) -> Self {
        Self { config }
    }

    /// Build index from dataset samples
    pub async fn build_from_samples(
        &self,
        samples: Vec<crate::DatasetSample>,
    ) -> Result<DatasetIndex> {
        let index = DatasetIndex::new(self.config.clone());

        for sample in samples {
            index.add_sample(&sample).await?;
        }

        if self.config.auto_optimize {
            index.optimize().await?;
        }

        Ok(index)
    }

    /// Set indexed fields
    pub fn indexed_fields(mut self, fields: Vec<String>) -> Self {
        self.config.indexed_fields = fields;
        self
    }

    /// Enable range indexing
    pub fn enable_range_indexing(mut self, enable: bool) -> Self {
        self.config.enable_range_indexing = enable;
        self
    }

    /// Set maximum entries
    pub fn max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = Some(max);
        self
    }
}

impl DatasetIndex {
    /// Create a new empty index
    pub fn new(config: IndexConfig) -> Self {
        Self {
            primary_index: RwLock::new(HashMap::new()),
            secondary_indexes: RwLock::new(HashMap::new()),
            range_indexes: RwLock::new(HashMap::new()),
            config,
            metadata: RwLock::new(IndexMetadata {
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                total_samples: 0,
                version: "1.0.0".to_string(),
                indexed_fields: Vec::new(),
            }),
        }
    }

    /// Add a sample to the index
    pub async fn add_sample(&self, sample: &crate::DatasetSample) -> Result<()> {
        let entry = self.create_index_entry(sample)?;
        let sample_id = entry.id.clone();

        // Add to primary index
        {
            let mut primary = self
                .primary_index
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
            primary.insert(sample_id.clone(), entry.clone());
        }

        // Add to secondary indexes
        {
            let mut secondary = self
                .secondary_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for (field, value) in &entry.indexed_fields {
                if !self.config.indexed_fields.contains(field) {
                    continue;
                }

                let field_index = secondary.entry(field.clone()).or_insert_with(BTreeMap::new);

                match value {
                    IndexValue::String(s) => {
                        field_index
                            .entry(s.clone())
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    IndexValue::Array(arr) => {
                        for item in arr {
                            field_index
                                .entry(item.clone())
                                .or_insert_with(HashSet::new)
                                .insert(sample_id.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Add to range indexes
        if self.config.enable_range_indexing {
            let mut range = self
                .range_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for (field, value) in &entry.indexed_fields {
                if !self.config.indexed_fields.contains(field) {
                    continue;
                }

                match value {
                    IndexValue::Number(n) => {
                        let field_index = range.entry(field.clone()).or_insert_with(BTreeMap::new);
                        let ordered_value = OrderedValue::Number(ordered_float::OrderedFloat(*n));
                        field_index
                            .entry(ordered_value)
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    IndexValue::String(s) => {
                        let field_index = range.entry(field.clone()).or_insert_with(BTreeMap::new);
                        let ordered_value = OrderedValue::String(s.clone());
                        field_index
                            .entry(ordered_value)
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    _ => {}
                }
            }
        }

        // Update metadata
        {
            let mut metadata = self
                .metadata
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
            metadata.updated_at = chrono::Utc::now();
            metadata.total_samples += 1;

            for field in self.config.indexed_fields.iter() {
                if !metadata.indexed_fields.contains(field) {
                    metadata.indexed_fields.push(field.clone());
                }
            }
        }

        Ok(())
    }

    /// Query the index
    pub async fn query(&self, query: &IndexQuery) -> Result<Vec<IndexEntry>> {
        let mut candidate_ids: Option<HashSet<String>> = None;

        // Apply equality filters
        for (field, value) in &query.equals {
            let field_candidates = self.get_candidates_for_field(field, value)?;
            candidate_ids = match candidate_ids {
                Some(existing) => Some(existing.intersection(&field_candidates).cloned().collect()),
                None => Some(field_candidates),
            };
        }

        // Apply range filters
        for (field, (min, max)) in &query.ranges {
            let range_candidates = self.get_range_candidates(field, *min, *max)?;
            candidate_ids = match candidate_ids {
                Some(existing) => Some(existing.intersection(&range_candidates).cloned().collect()),
                None => Some(range_candidates),
            };
        }

        // Apply text contains filters
        for (field, search_text) in &query.text_contains {
            let text_candidates = self.get_text_candidates(field, search_text)?;
            candidate_ids = match candidate_ids {
                Some(existing) => Some(existing.intersection(&text_candidates).cloned().collect()),
                None => Some(text_candidates),
            };
        }

        // If no filters, get all entries
        let final_ids = candidate_ids.unwrap_or_else(|| {
            self.primary_index
                .read()
                .map(|primary| primary.keys().cloned().collect())
                .unwrap_or_default()
        });

        // Retrieve entries
        let mut results = Vec::new();
        {
            let primary = self
                .primary_index
                .read()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for id in final_ids {
                if let Some(entry) = primary.get(&id) {
                    results.push(entry.clone());
                }
            }
        }

        // Apply sorting
        if let Some((sort_field, direction)) = &query.sort_by {
            results.sort_by(|a, b| {
                let ord = self.compare_entries(a, b, sort_field);
                match direction {
                    SortDirection::Ascending => ord,
                    SortDirection::Descending => ord.reverse(),
                }
            });
        }

        // Apply pagination
        if let Some(offset) = query.offset {
            results = results.into_iter().skip(offset).collect();
        }

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get sample by ID
    pub async fn get_by_id(&self, id: &str) -> Result<Option<IndexEntry>> {
        let primary = self
            .primary_index
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
        Ok(primary.get(id).cloned())
    }

    /// Remove sample from index
    pub async fn remove_sample(&self, id: &str) -> Result<bool> {
        let entry = {
            let mut primary = self
                .primary_index
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
            primary.remove(id)
        };

        if let Some(entry) = entry {
            // Remove from secondary indexes
            {
                let mut secondary = self
                    .secondary_indexes
                    .write()
                    .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

                for (field, value) in &entry.indexed_fields {
                    if let Some(field_index) = secondary.get_mut(field) {
                        match value {
                            IndexValue::String(s) => {
                                if let Some(id_set) = field_index.get_mut(s) {
                                    id_set.remove(id);
                                    if id_set.is_empty() {
                                        field_index.remove(s);
                                    }
                                }
                            }
                            IndexValue::Array(arr) => {
                                for item in arr {
                                    if let Some(id_set) = field_index.get_mut(item) {
                                        id_set.remove(id);
                                        if id_set.is_empty() {
                                            field_index.remove(item);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Remove from range indexes
            {
                let mut range = self
                    .range_indexes
                    .write()
                    .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

                for (field, value) in &entry.indexed_fields {
                    if let Some(field_index) = range.get_mut(field) {
                        let ordered_value = match value {
                            IndexValue::Number(n) => {
                                Some(OrderedValue::Number(ordered_float::OrderedFloat(*n)))
                            }
                            IndexValue::String(s) => Some(OrderedValue::String(s.clone())),
                            _ => None,
                        };

                        if let Some(ov) = ordered_value {
                            if let Some(id_set) = field_index.get_mut(&ov) {
                                id_set.remove(id);
                                if id_set.is_empty() {
                                    field_index.remove(&ov);
                                }
                            }
                        }
                    }
                }
            }

            // Update metadata
            {
                let mut metadata = self
                    .metadata
                    .write()
                    .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
                metadata.updated_at = chrono::Utc::now();
                metadata.total_samples = metadata.total_samples.saturating_sub(1);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Optimize index structure
    pub async fn optimize(&self) -> Result<()> {
        // Remove empty entries from secondary indexes
        {
            let mut secondary = self
                .secondary_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for field_index in secondary.values_mut() {
                field_index.retain(|_, id_set| !id_set.is_empty());
            }
        }

        // Remove empty entries from range indexes
        {
            let mut range = self
                .range_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for field_index in range.values_mut() {
                field_index.retain(|_, id_set| !id_set.is_empty());
            }
        }

        Ok(())
    }

    /// Save index to file
    pub async fn save(&self, path: &Path) -> Result<()> {
        let serializable_index = SerializableIndex {
            primary_index: self
                .primary_index
                .read()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?
                .clone(),
            metadata: self
                .metadata
                .read()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?
                .clone(),
        };

        let file = File::create(path).map_err(DatasetError::IoError)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &serializable_index)
            .map_err(|e| DatasetError::FormatError(format!("Index serialization failed: {e}")))?;

        Ok(())
    }

    /// Load index from file
    pub async fn load(path: &Path, config: IndexConfig) -> Result<Self> {
        let file = File::open(path).map_err(DatasetError::IoError)?;
        let reader = BufReader::new(file);

        let serializable_index: SerializableIndex = serde_json::from_reader(reader)
            .map_err(|e| DatasetError::FormatError(format!("Index deserialization failed: {e}")))?;

        let index = Self::new(config);

        // Restore primary index
        {
            let mut primary = index
                .primary_index
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
            *primary = serializable_index.primary_index;
        }

        // Rebuild secondary and range indexes
        {
            let primary = index
                .primary_index
                .read()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for entry in primary.values() {
                index.rebuild_indexes_for_entry(entry)?;
            }
        }

        // Restore metadata
        {
            let mut metadata = index
                .metadata
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;
            *metadata = serializable_index.metadata;
        }

        Ok(index)
    }

    /// Get index statistics
    pub async fn get_statistics(&self) -> Result<IndexStatistics> {
        let metadata = self
            .metadata
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

        let secondary_count = self
            .secondary_indexes
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?
            .len();

        let range_count = self
            .range_indexes
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?
            .len();

        Ok(IndexStatistics {
            total_samples: metadata.total_samples,
            indexed_fields: metadata.indexed_fields.len(),
            secondary_indexes: secondary_count,
            range_indexes: range_count,
            created_at: metadata.created_at,
            updated_at: metadata.updated_at,
        })
    }

    // Helper methods

    fn create_index_entry(&self, sample: &crate::DatasetSample) -> Result<IndexEntry> {
        let mut indexed_fields = HashMap::new();

        // Add standard fields
        indexed_fields.insert(
            "language".to_string(),
            IndexValue::String(sample.language.as_str().to_string()),
        );
        indexed_fields.insert(
            "duration".to_string(),
            IndexValue::Number(sample.audio.duration() as f64),
        );
        indexed_fields.insert(
            "sample_rate".to_string(),
            IndexValue::Number(sample.audio.sample_rate() as f64),
        );

        if let Some(speaker) = &sample.speaker {
            indexed_fields.insert(
                "speaker_id".to_string(),
                IndexValue::String(speaker.id.clone()),
            );
        }

        indexed_fields.insert(
            "quality_score".to_string(),
            IndexValue::Number(sample.quality.overall_quality.unwrap_or(0.0) as f64),
        );

        Ok(IndexEntry {
            id: sample.id.clone(),
            audio_path: PathBuf::from(format!("{}.wav", sample.id)),
            text: sample.text.clone(),
            duration: sample.audio.duration() as f64,
            sample_rate: sample.audio.sample_rate(),
            language: sample.language.as_str().to_string(),
            speaker_id: sample.speaker.as_ref().map(|s| s.id.clone()),
            quality_score: sample.quality.overall_quality,
            indexed_fields,
        })
    }

    fn get_candidates_for_field(&self, field: &str, value: &IndexValue) -> Result<HashSet<String>> {
        let secondary = self
            .secondary_indexes
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

        if let Some(field_index) = secondary.get(field) {
            match value {
                IndexValue::String(s) => Ok(field_index.get(s).cloned().unwrap_or_default()),
                _ => Ok(HashSet::new()),
            }
        } else {
            Ok(HashSet::new())
        }
    }

    fn get_range_candidates(
        &self,
        field: &str,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Result<HashSet<String>> {
        let range = self
            .range_indexes
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

        if let Some(field_index) = range.get(field) {
            let mut candidates = HashSet::new();

            for (ordered_value, id_set) in field_index.iter() {
                let value = match ordered_value {
                    OrderedValue::Number(n) => n.into_inner(),
                    OrderedValue::String(_) => continue,
                };

                let in_range = match (min, max) {
                    (Some(min_val), Some(max_val)) => value >= min_val && value <= max_val,
                    (Some(min_val), None) => value >= min_val,
                    (None, Some(max_val)) => value <= max_val,
                    (None, None) => true,
                };

                if in_range {
                    candidates.extend(id_set.iter().cloned());
                }
            }

            Ok(candidates)
        } else {
            Ok(HashSet::new())
        }
    }

    fn get_text_candidates(&self, field: &str, search_text: &str) -> Result<HashSet<String>> {
        let primary = self
            .primary_index
            .read()
            .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

        let mut candidates = HashSet::new();
        let search_lower = search_text.to_lowercase();

        for entry in primary.values() {
            let text = match field {
                "text" => &entry.text,
                _ => continue,
            };

            if text.to_lowercase().contains(&search_lower) {
                candidates.insert(entry.id.clone());
            }
        }

        Ok(candidates)
    }

    fn compare_entries(&self, a: &IndexEntry, b: &IndexEntry, field: &str) -> std::cmp::Ordering {
        match field {
            "id" => a.id.cmp(&b.id),
            "duration" => a
                .duration
                .partial_cmp(&b.duration)
                .unwrap_or(std::cmp::Ordering::Equal),
            "sample_rate" => a.sample_rate.cmp(&b.sample_rate),
            "language" => a.language.cmp(&b.language),
            "quality_score" => a
                .quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        }
    }

    fn rebuild_indexes_for_entry(&self, entry: &IndexEntry) -> Result<()> {
        let sample_id = &entry.id;

        // Rebuild secondary indexes
        {
            let mut secondary = self
                .secondary_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for (field, value) in &entry.indexed_fields {
                if !self.config.indexed_fields.contains(field) {
                    continue;
                }

                let field_index = secondary.entry(field.clone()).or_insert_with(BTreeMap::new);

                match value {
                    IndexValue::String(s) => {
                        field_index
                            .entry(s.clone())
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    IndexValue::Array(arr) => {
                        for item in arr {
                            field_index
                                .entry(item.clone())
                                .or_insert_with(HashSet::new)
                                .insert(sample_id.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Rebuild range indexes
        if self.config.enable_range_indexing {
            let mut range = self
                .range_indexes
                .write()
                .map_err(|_| DatasetError::ConfigError("Index lock poisoned".to_string()))?;

            for (field, value) in &entry.indexed_fields {
                if !self.config.indexed_fields.contains(field) {
                    continue;
                }

                match value {
                    IndexValue::Number(n) => {
                        let field_index = range.entry(field.clone()).or_insert_with(BTreeMap::new);
                        let ordered_value = OrderedValue::Number(ordered_float::OrderedFloat(*n));
                        field_index
                            .entry(ordered_value)
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    IndexValue::String(s) => {
                        let field_index = range.entry(field.clone()).or_insert_with(BTreeMap::new);
                        let ordered_value = OrderedValue::String(s.clone());
                        field_index
                            .entry(ordered_value)
                            .or_insert_with(HashSet::new)
                            .insert(sample_id.clone());
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

/// Serializable version of the index for persistence
#[derive(Serialize, Deserialize)]
struct SerializableIndex {
    primary_index: HashMap<String, IndexEntry>,
    metadata: IndexMetadata,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_samples: usize,
    pub indexed_fields: usize,
    pub secondary_indexes: usize,
    pub range_indexes: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_index_creation() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..5 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let builder = IndexBuilder::new();
        let index = builder.build_from_samples(samples).await.unwrap();

        let stats = index.get_statistics().await.unwrap();
        assert_eq!(stats.total_samples, 5);
        assert!(stats.indexed_fields > 0);
    }

    #[tokio::test]
    async fn test_index_query() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..dataset.len().min(10) {
            samples.push(dataset.get(i).await.unwrap());
        }

        let builder = IndexBuilder::new();
        let index = builder.build_from_samples(samples).await.unwrap();

        // Test simple query
        let mut query = IndexQuery::default();
        query.equals.insert(
            "language".to_string(),
            IndexValue::String("en-US".to_string()),
        );

        let results = index.query(&query).await.unwrap();
        assert!(!results.is_empty());

        // Test range query
        let mut range_query = IndexQuery::default();
        range_query
            .ranges
            .insert("duration".to_string(), (Some(1.0), Some(10.0)));

        let range_results = index.query(&range_query).await.unwrap();
        assert!(!range_results.is_empty());
    }

    #[tokio::test]
    async fn test_index_persistence() {
        let dataset = DummyDataset::small();

        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(dataset.get(i).await.unwrap());
        }

        let builder = IndexBuilder::new();
        let index = builder.build_from_samples(samples).await.unwrap();

        let temp_path = std::env::temp_dir().join("test_index.json");
        index.save(&temp_path).await.unwrap();

        assert!(temp_path.exists());

        let loaded_index = DatasetIndex::load(&temp_path, IndexConfig::default())
            .await
            .unwrap();
        let stats = loaded_index.get_statistics().await.unwrap();
        assert_eq!(stats.total_samples, 3);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_index_operations() {
        let dataset = DummyDataset::small();

        let sample = dataset.get(0).await.unwrap();

        let builder = IndexBuilder::new();
        let index = builder.build_from_samples(vec![]).await.unwrap();

        // Test add sample
        index.add_sample(&sample).await.unwrap();

        let stats = index.get_statistics().await.unwrap();
        assert_eq!(stats.total_samples, 1);

        // Test get by ID
        let retrieved = index.get_by_id(&sample.id).await.unwrap();
        assert!(retrieved.is_some());

        // Test remove sample
        let removed = index.remove_sample(&sample.id).await.unwrap();
        assert!(removed);

        let stats_after = index.get_statistics().await.unwrap();
        assert_eq!(stats_after.total_samples, 0);
    }
}
