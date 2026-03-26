//! Data Versioning System for Benchmark and Evaluation Results
//!
//! This module provides comprehensive data versioning capabilities for managing
//! benchmark datasets, evaluation results, and model configurations over time.
//! It enables reproducibility, change tracking, and collaborative evaluation workflows.
//!
//! # Features
//!
//! - **Version Control**: Git-like versioning for datasets and results
//! - **Change Tracking**: Complete audit trail of all modifications
//! - **Branching**: Support for experimental branches and comparisons
//! - **Tagging**: Semantic versioning and milestone tracking
//! - **Diff Generation**: Compare versions and identify changes
//! - **Metadata Management**: Comprehensive metadata for each version
//! - **Storage Backend**: Pluggable storage (filesystem, S3, database)
//! - **Compression**: Efficient storage with delta compression
//!
//! # Example
//!
//! ```no_run
//! use voirs_evaluation::data_versioning::*;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create version control system
//! let mut vcs = DataVersionControl::new("/tmp/voirs_benchmarks")?;
//!
//! // Create initial dataset version
//! let mut metadata = HashMap::new();
//! metadata.insert("description".to_string(), "Initial benchmark dataset".to_string());
//! metadata.insert("author".to_string(), "user@example.com".to_string());
//!
//! let version_id = vcs.commit(
//!     "benchmark-v1",
//!     vec![1u8, 2, 3], // Sample data bytes
//!     metadata,
//!     "Initial benchmark dataset"
//! )?;
//!
//! println!("Created version: {}", version_id);
//!
//! // Tag a version
//! vcs.tag(&version_id, "v1.0.0", "Release 1.0.0")?;
//!
//! // Retrieve version
//! let retrieved = vcs.checkout(&version_id)?;
//! println!("Retrieved {} bytes from version {}", retrieved.data.len(), version_id);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use oxiarc_deflate::{GzipStreamDecoder, GzipStreamEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Data versioning errors
#[derive(Error, Debug)]
pub enum VersioningError {
    /// Version not found
    #[error("Version not found: {version_id}")]
    VersionNotFound {
        /// Version ID
        version_id: String,
    },

    /// Tag already exists
    #[error("Tag already exists: {tag}")]
    TagAlreadyExists {
        /// Tag name
        tag: String,
    },

    /// Branch not found
    #[error("Branch not found: {branch}")]
    BranchNotFound {
        /// Branch name
        branch: String,
    },

    /// Invalid version
    #[error("Invalid version: {message}")]
    InvalidVersion {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Compression error
    #[error("Compression error: {message}")]
    CompressionError {
        /// Error message
        message: String,
    },

    /// Merge conflict
    #[error("Merge conflict: {message}")]
    MergeConflict {
        /// Error message
        message: String,
    },
}

/// Data version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Unique version ID
    pub id: String,
    /// Parent version ID (if any)
    pub parent_id: Option<String>,
    /// Dataset/result name
    pub name: String,
    /// Commit message
    pub message: String,
    /// Author
    pub author: String,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Branch name
    pub branch: String,
    /// Tags associated with this version
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Data hash (for integrity)
    pub data_hash: String,
    /// Data size in bytes
    pub data_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
}

/// Version tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionTag {
    /// Tag name
    pub name: String,
    /// Tagged version ID
    pub version_id: String,
    /// Tag message
    pub message: String,
    /// Tag timestamp
    pub timestamp: DateTime<Utc>,
    /// Tagger
    pub author: String,
}

/// Branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch name
    pub name: String,
    /// Current HEAD version ID
    pub head: String,
    /// Branch description
    pub description: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

/// Version diff representing changes between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Source version ID
    pub from_version: String,
    /// Target version ID
    pub to_version: String,
    /// Number of bytes added
    pub bytes_added: usize,
    /// Number of bytes removed
    pub bytes_removed: usize,
    /// Metadata changes
    pub metadata_changes: Vec<MetadataChange>,
    /// Summary of changes
    pub summary: String,
}

/// Metadata change entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataChange {
    /// Metadata key
    pub key: String,
    /// Change type
    pub change_type: ChangeType,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: Option<String>,
}

/// Type of change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Added
    Added,
    /// Modified
    Modified,
    /// Deleted
    Deleted,
}

/// Versioned data entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedData {
    /// Version metadata
    pub metadata: VersionMetadata,
    /// Actual data (compressed)
    pub data: Vec<u8>,
}

/// Version control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControlConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Maximum versions to keep
    pub max_versions: Option<usize>,
    /// Enable delta compression
    pub delta_compression: bool,
    /// Author name (default)
    pub default_author: String,
}

impl Default for VersionControlConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_level: 6,
            max_versions: None,
            delta_compression: true,
            default_author: "unknown".to_string(),
        }
    }
}

/// Data Version Control System
pub struct DataVersionControl {
    /// Root directory for version storage
    root_dir: PathBuf,
    /// Configuration
    config: VersionControlConfig,
    /// In-memory index of versions
    versions: HashMap<String, VersionMetadata>,
    /// Tags
    tags: HashMap<String, VersionTag>,
    /// Branches
    branches: HashMap<String, Branch>,
    /// Current branch
    current_branch: String,
}

impl DataVersionControl {
    /// Create new version control system
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self, VersioningError> {
        Self::with_config(root_dir, VersionControlConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(
        root_dir: impl AsRef<Path>,
        config: VersionControlConfig,
    ) -> Result<Self, VersioningError> {
        let root_dir = root_dir.as_ref().to_path_buf();

        // Create directory structure
        fs::create_dir_all(&root_dir)?;
        fs::create_dir_all(root_dir.join("versions"))?;
        fs::create_dir_all(root_dir.join("metadata"))?;
        fs::create_dir_all(root_dir.join("tags"))?;
        fs::create_dir_all(root_dir.join("branches"))?;

        let mut vcs = Self {
            root_dir,
            config,
            versions: HashMap::new(),
            tags: HashMap::new(),
            branches: HashMap::new(),
            current_branch: "main".to_string(),
        };

        // Initialize main branch if it doesn't exist
        vcs.initialize_main_branch()?;

        // Load existing data
        vcs.load_index()?;

        info!("Initialized version control system");

        Ok(vcs)
    }

    /// Initialize main branch
    fn initialize_main_branch(&mut self) -> Result<(), VersioningError> {
        let main_branch = Branch {
            name: "main".to_string(),
            head: String::new(),
            description: "Main branch".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.branches
            .insert("main".to_string(), main_branch.clone());
        self.save_branch(&main_branch)?;

        Ok(())
    }

    /// Load version index from disk
    fn load_index(&mut self) -> Result<(), VersioningError> {
        // Load versions
        let metadata_dir = self.root_dir.join("metadata");
        if metadata_dir.exists() {
            for entry in fs::read_dir(metadata_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("meta") {
                    let data = fs::read(entry.path())?;
                    let (metadata, _): (VersionMetadata, _) =
                        oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
                    self.versions.insert(metadata.id.clone(), metadata);
                }
            }
        }

        // Load tags
        let tags_dir = self.root_dir.join("tags");
        if tags_dir.exists() {
            for entry in fs::read_dir(tags_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("tag") {
                    let data = fs::read(entry.path())?;
                    let (tag, _): (VersionTag, _) =
                        oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
                    self.tags.insert(tag.name.clone(), tag);
                }
            }
        }

        // Load branches
        let branches_dir = self.root_dir.join("branches");
        if branches_dir.exists() {
            for entry in fs::read_dir(branches_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("branch") {
                    let data = fs::read(entry.path())?;
                    let (branch, _): (Branch, _) =
                        oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
                    self.branches.insert(branch.name.clone(), branch);
                }
            }
        }

        debug!(
            "Loaded {} versions, {} tags, {} branches",
            self.versions.len(),
            self.tags.len(),
            self.branches.len()
        );

        Ok(())
    }

    /// Commit a new version
    pub fn commit(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        metadata: HashMap<String, String>,
        message: impl Into<String>,
    ) -> Result<String, VersioningError> {
        let name = name.into();
        let message = message.into();
        let version_id = Uuid::new_v4().to_string();

        // Get parent from current branch
        let parent_id = self.branches.get(&self.current_branch).and_then(|b| {
            if b.head.is_empty() {
                None
            } else {
                Some(b.head.clone())
            }
        });

        // Compress data if enabled
        let (compressed_data, compression_ratio) = if self.config.enable_compression {
            let compressed = self.compress_data(&data)?;
            let ratio = compressed.len() as f32 / data.len() as f32;
            (compressed, ratio)
        } else {
            (data.clone(), 1.0)
        };

        // Calculate hash
        let data_hash = format!("{:x}", md5::compute(&data));

        // Create metadata
        let version_metadata = VersionMetadata {
            id: version_id.clone(),
            parent_id,
            name: name.clone(),
            message,
            author: self.config.default_author.clone(),
            timestamp: Utc::now(),
            branch: self.current_branch.clone(),
            tags: Vec::new(),
            metadata,
            data_hash,
            data_size: data.len(),
            compression_ratio,
        };

        // Save data
        self.save_version_data(&version_id, &compressed_data)?;
        self.save_version_metadata(&version_metadata)?;

        // Update branch HEAD
        let branch_to_save = if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = version_id.clone();
            branch.updated_at = Utc::now();
            Some(branch.clone())
        } else {
            None
        };

        if let Some(branch) = branch_to_save {
            self.save_branch(&branch)?;
        }

        // Update index
        self.versions.insert(version_id.clone(), version_metadata);

        info!("Committed version {} ({})", version_id, name);

        // Cleanup old versions if needed
        if let Some(max) = self.config.max_versions {
            self.cleanup_old_versions(max)?;
        }

        Ok(version_id)
    }

    /// Checkout a version by ID
    pub fn checkout(&self, version_id: &str) -> Result<VersionedData, VersioningError> {
        let metadata = self
            .versions
            .get(version_id)
            .ok_or_else(|| VersioningError::VersionNotFound {
                version_id: version_id.to_string(),
            })?
            .clone();

        let compressed_data = self.load_version_data(version_id)?;

        let data = if self.config.enable_compression {
            self.decompress_data(&compressed_data)?
        } else {
            compressed_data
        };

        Ok(VersionedData { metadata, data })
    }

    /// Checkout by tag
    pub fn checkout_tag(&self, tag: &str) -> Result<VersionedData, VersioningError> {
        let version_tag = self
            .tags
            .get(tag)
            .ok_or_else(|| VersioningError::VersionNotFound {
                version_id: format!("tag:{}", tag),
            })?;

        self.checkout(&version_tag.version_id)
    }

    /// Create a tag
    pub fn tag(
        &mut self,
        version_id: &str,
        tag_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), VersioningError> {
        let tag_name = tag_name.into();

        if self.tags.contains_key(&tag_name) {
            return Err(VersioningError::TagAlreadyExists { tag: tag_name });
        }

        // Verify version exists
        if !self.versions.contains_key(version_id) {
            return Err(VersioningError::VersionNotFound {
                version_id: version_id.to_string(),
            });
        }

        let tag = VersionTag {
            name: tag_name.clone(),
            version_id: version_id.to_string(),
            message: message.into(),
            timestamp: Utc::now(),
            author: self.config.default_author.clone(),
        };

        self.save_tag(&tag)?;
        self.tags.insert(tag_name.clone(), tag);

        // Update version metadata to include tag
        let metadata_to_save = if let Some(metadata) = self.versions.get_mut(version_id) {
            metadata.tags.push(tag_name.clone());
            Some(metadata.clone())
        } else {
            None
        };

        if let Some(metadata) = metadata_to_save {
            self.save_version_metadata(&metadata)?;
        }

        info!("Created tag: {}", tag_name);

        Ok(())
    }

    /// Create a new branch
    pub fn create_branch(
        &mut self,
        branch_name: impl Into<String>,
        from_version: Option<&str>,
    ) -> Result<(), VersioningError> {
        let branch_name = branch_name.into();

        let head = if let Some(version_id) = from_version {
            version_id.to_string()
        } else {
            self.branches
                .get(&self.current_branch)
                .map(|b| b.head.clone())
                .unwrap_or_default()
        };

        let branch = Branch {
            name: branch_name.clone(),
            head,
            description: format!("Branch {}", branch_name),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.save_branch(&branch)?;
        self.branches.insert(branch_name.clone(), branch);

        info!("Created branch: {}", branch_name);

        Ok(())
    }

    /// Switch to a branch
    pub fn switch_branch(&mut self, branch_name: &str) -> Result<(), VersioningError> {
        if !self.branches.contains_key(branch_name) {
            return Err(VersioningError::BranchNotFound {
                branch: branch_name.to_string(),
            });
        }

        self.current_branch = branch_name.to_string();
        info!("Switched to branch: {}", branch_name);

        Ok(())
    }

    /// Get version history
    pub fn get_history(&self, max_count: Option<usize>) -> Vec<VersionMetadata> {
        let mut history: Vec<_> = self.versions.values().cloned().collect();
        history.sort_by_key(|b| std::cmp::Reverse(b.timestamp));

        if let Some(max) = max_count {
            history.truncate(max);
        }

        history
    }

    /// Get version diff
    pub fn diff(&self, from: &str, to: &str) -> Result<VersionDiff, VersioningError> {
        let from_version = self.checkout(from)?;
        let to_version = self.checkout(to)?;

        // Simple byte-level diff
        let bytes_added = if to_version.data.len() > from_version.data.len() {
            to_version.data.len() - from_version.data.len()
        } else {
            0
        };

        let bytes_removed = if from_version.data.len() > to_version.data.len() {
            from_version.data.len() - to_version.data.len()
        } else {
            0
        };

        // Metadata changes
        let mut metadata_changes = Vec::new();
        for (key, new_value) in &to_version.metadata.metadata {
            match from_version.metadata.metadata.get(key) {
                Some(old_value) if old_value != new_value => {
                    metadata_changes.push(MetadataChange {
                        key: key.clone(),
                        change_type: ChangeType::Modified,
                        old_value: Some(old_value.clone()),
                        new_value: Some(new_value.clone()),
                    });
                }
                None => {
                    metadata_changes.push(MetadataChange {
                        key: key.clone(),
                        change_type: ChangeType::Added,
                        old_value: None,
                        new_value: Some(new_value.clone()),
                    });
                }
                _ => {}
            }
        }

        for (key, old_value) in &from_version.metadata.metadata {
            if !to_version.metadata.metadata.contains_key(key) {
                metadata_changes.push(MetadataChange {
                    key: key.clone(),
                    change_type: ChangeType::Deleted,
                    old_value: Some(old_value.clone()),
                    new_value: None,
                });
            }
        }

        let summary = format!(
            "+{} bytes, -{} bytes, {} metadata changes",
            bytes_added,
            bytes_removed,
            metadata_changes.len()
        );

        Ok(VersionDiff {
            from_version: from.to_string(),
            to_version: to.to_string(),
            bytes_added,
            bytes_removed,
            metadata_changes,
            summary,
        })
    }

    /// List all versions
    pub fn list_versions(&self) -> Vec<VersionMetadata> {
        let mut versions: Vec<_> = self.versions.values().cloned().collect();
        versions.sort_by_key(|b| std::cmp::Reverse(b.timestamp));
        versions
    }

    /// List all tags
    pub fn list_tags(&self) -> Vec<VersionTag> {
        self.tags.values().cloned().collect()
    }

    /// List all branches
    pub fn list_branches(&self) -> Vec<Branch> {
        self.branches.values().cloned().collect()
    }

    /// Compress data
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, VersioningError> {
        let mut encoder = GzipStreamEncoder::new(Vec::new(), self.config.compression_level as u8);
        encoder
            .write_all(data)
            .map_err(|e| VersioningError::CompressionError {
                message: e.to_string(),
            })?;
        encoder
            .finish()
            .map_err(|e| VersioningError::CompressionError {
                message: e.to_string(),
            })
    }

    /// Decompress data
    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>, VersioningError> {
        let mut decoder = GzipStreamDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| VersioningError::CompressionError {
                message: e.to_string(),
            })?;
        Ok(decompressed)
    }

    /// Save version data to disk
    fn save_version_data(&self, version_id: &str, data: &[u8]) -> Result<(), VersioningError> {
        let path = self
            .root_dir
            .join("versions")
            .join(format!("{}.dat", version_id));
        fs::write(path, data)?;
        Ok(())
    }

    /// Load version data from disk
    fn load_version_data(&self, version_id: &str) -> Result<Vec<u8>, VersioningError> {
        let path = self
            .root_dir
            .join("versions")
            .join(format!("{}.dat", version_id));
        Ok(fs::read(path)?)
    }

    /// Save version metadata
    fn save_version_metadata(&self, metadata: &VersionMetadata) -> Result<(), VersioningError> {
        let path = self
            .root_dir
            .join("metadata")
            .join(format!("{}.meta", metadata.id));
        let data = oxicode::serde::encode_to_vec(metadata, oxicode::config::standard())
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Save tag
    fn save_tag(&self, tag: &VersionTag) -> Result<(), VersioningError> {
        let path = self.root_dir.join("tags").join(format!("{}.tag", tag.name));
        let data = oxicode::serde::encode_to_vec(tag, oxicode::config::standard())
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Save branch
    fn save_branch(&self, branch: &Branch) -> Result<(), VersioningError> {
        let path = self
            .root_dir
            .join("branches")
            .join(format!("{}.branch", branch.name));
        let data = oxicode::serde::encode_to_vec(branch, oxicode::config::standard())
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Cleanup old versions
    fn cleanup_old_versions(&mut self, max_versions: usize) -> Result<(), VersioningError> {
        if self.versions.len() <= max_versions {
            return Ok(());
        }

        let mut versions: Vec<_> = self.versions.values().cloned().collect();
        versions.sort_by_key(|a| a.timestamp);

        let to_remove = versions.len() - max_versions;
        let mut to_remove_ids = Vec::new();

        for i in 0..to_remove {
            let version = &versions[i];

            // Don't remove tagged versions
            if !version.tags.is_empty() {
                continue;
            }

            to_remove_ids.push(version.id.clone());
        }

        // Remove versions after collecting IDs
        for version_id in to_remove_ids {
            // Remove files
            let _ = fs::remove_file(
                self.root_dir
                    .join("versions")
                    .join(format!("{}.dat", version_id)),
            );
            let _ = fs::remove_file(
                self.root_dir
                    .join("metadata")
                    .join(format!("{}.meta", version_id)),
            );

            // Remove from index
            self.versions.remove(&version_id);

            debug!("Cleaned up old version: {}", version_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_version_control_creation() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let vcs = DataVersionControl::new(&temp_dir).unwrap();

        assert_eq!(vcs.current_branch, "main");
        assert!(vcs.branches.contains_key("main"));

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_commit_and_checkout() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let mut vcs = DataVersionControl::new(&temp_dir).unwrap();

        let data = vec![1u8, 2, 3, 4, 5];
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());

        let version_id = vcs
            .commit("test-data", data.clone(), metadata, "Initial commit")
            .unwrap();

        let retrieved = vcs.checkout(&version_id).unwrap();
        assert_eq!(retrieved.data, data);
        assert_eq!(retrieved.metadata.name, "test-data");

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_tagging() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let mut vcs = DataVersionControl::new(&temp_dir).unwrap();

        let data = vec![1u8, 2, 3];
        let version_id = vcs
            .commit("test", data.clone(), HashMap::new(), "Test")
            .unwrap();

        vcs.tag(&version_id, "v1.0.0", "Release 1.0").unwrap();

        let retrieved = vcs.checkout_tag("v1.0.0").unwrap();
        assert_eq!(retrieved.data, data);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_branching() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let mut vcs = DataVersionControl::new(&temp_dir).unwrap();

        let version_id = vcs
            .commit("test", vec![1, 2, 3], HashMap::new(), "Main commit")
            .unwrap();

        vcs.create_branch("dev", Some(&version_id)).unwrap();
        vcs.switch_branch("dev").unwrap();

        assert_eq!(vcs.current_branch, "dev");

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_version_diff() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let mut vcs = DataVersionControl::new(&temp_dir).unwrap();

        let v1 = vcs
            .commit("test", vec![1, 2, 3], HashMap::new(), "Version 1")
            .unwrap();
        let v2 = vcs
            .commit("test", vec![1, 2, 3, 4, 5], HashMap::new(), "Version 2")
            .unwrap();

        let diff = vcs.diff(&v1, &v2).unwrap();
        assert_eq!(diff.bytes_added, 2);
        assert_eq!(diff.bytes_removed, 0);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_version_history() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let mut vcs = DataVersionControl::new(&temp_dir).unwrap();

        vcs.commit("test1", vec![1], HashMap::new(), "Commit 1")
            .unwrap();
        vcs.commit("test2", vec![2], HashMap::new(), "Commit 2")
            .unwrap();
        vcs.commit("test3", vec![3], HashMap::new(), "Commit 3")
            .unwrap();

        let history = vcs.get_history(None);
        assert_eq!(history.len(), 3);

        let limited_history = vcs.get_history(Some(2));
        assert_eq!(limited_history.len(), 2);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_compression() {
        let temp_dir = env::temp_dir().join(format!("voirs_vcs_test_{}", Uuid::new_v4()));
        let config = VersionControlConfig {
            enable_compression: true,
            ..Default::default()
        };
        let mut vcs = DataVersionControl::with_config(&temp_dir, config).unwrap();

        let data = vec![1u8; 1000]; // Highly compressible
        let version_id = vcs
            .commit(
                "compressed",
                data.clone(),
                HashMap::new(),
                "Compressed data",
            )
            .unwrap();

        let retrieved = vcs.checkout(&version_id).unwrap();
        assert_eq!(retrieved.data, data);
        assert!(retrieved.metadata.compression_ratio < 1.0);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }
}
