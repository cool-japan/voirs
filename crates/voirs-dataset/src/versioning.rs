//! Dataset versioning and checksum verification
//!
//! This module provides tools for dataset versioning, integrity checking, and reproducibility.
//! It supports:
//! - Dataset version tracking with semantic versioning
//! - File integrity verification using checksums
//! - Dataset manifests with complete metadata
//! - Reproducibility through deterministic ordering and hashing

use crate::{DatasetError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Dataset version following semantic versioning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DatasetVersion {
    /// Major version (incompatible changes)
    pub major: u32,
    /// Minor version (backward-compatible additions)
    pub minor: u32,
    /// Patch version (backward-compatible bug fixes)
    pub patch: u32,
    /// Optional pre-release identifier
    pub pre_release: Option<String>,
}

impl DatasetVersion {
    /// Create new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
        }
    }

    /// Create version with pre-release identifier
    pub fn with_pre_release(major: u32, minor: u32, patch: u32, pre_release: String) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: Some(pre_release),
        }
    }

    /// Parse version from string (e.g., "1.2.3" or "1.2.3-beta.1")
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        let version_parts: Vec<&str> = parts[0].split('.').collect();

        if version_parts.len() != 3 {
            return Err(DatasetError::ValidationError(format!(
                "Invalid version format: {}",
                s
            )));
        }

        let major = version_parts[0]
            .parse()
            .map_err(|_| DatasetError::ValidationError("Invalid major version".to_string()))?;
        let minor = version_parts[1]
            .parse()
            .map_err(|_| DatasetError::ValidationError("Invalid minor version".to_string()))?;
        let patch = version_parts[2]
            .parse()
            .map_err(|_| DatasetError::ValidationError("Invalid patch version".to_string()))?;

        let pre_release = if parts.len() > 1 {
            Some(parts[1..].join("-"))
        } else {
            None
        };

        Ok(Self {
            major,
            minor,
            patch,
            pre_release,
        })
    }

    /// Convert to string representation
    pub fn as_string(&self) -> String {
        let base = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if let Some(pre) = &self.pre_release {
            format!("{}-{}", base, pre)
        } else {
            base
        }
    }

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &DatasetVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

/// File checksum information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChecksum {
    /// Relative file path
    pub path: PathBuf,
    /// MD5 checksum
    pub md5: String,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub modified: DateTime<Utc>,
}

impl FileChecksum {
    /// Calculate checksum for a file
    pub fn calculate<P: AsRef<Path>>(path: P, base_path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let base_ref = base_path.as_ref();
        let file = File::open(path_ref)?;
        let mut reader = BufReader::new(file);

        let mut hasher = md5::Context::new();
        let mut buffer = [0u8; 8192];

        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.consume(&buffer[..count]);
        }

        let hash = hasher.compute();
        let md5 = format!("{:x}", hash);

        let metadata = std::fs::metadata(path_ref)?;
        let size = metadata.len();
        let modified = metadata.modified()?.into();

        let relative_path = path_ref
            .strip_prefix(base_ref)
            .unwrap_or(path_ref)
            .to_path_buf();

        Ok(Self {
            path: relative_path,
            md5,
            size,
            modified,
        })
    }

    /// Verify checksum matches current file
    pub fn verify<P: AsRef<Path>>(&self, base_path: P) -> Result<bool> {
        let base_ref = base_path.as_ref();
        let full_path = base_ref.join(&self.path);
        let current = Self::calculate(&full_path, &base_ref.to_path_buf())?;

        Ok(current.md5 == self.md5)
    }
}

/// Dataset manifest with complete metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: DatasetVersion,
    /// Creation timestamp
    pub created: DateTime<Utc>,
    /// Creator information
    pub creator: Option<String>,
    /// Dataset description
    pub description: Option<String>,
    /// License information
    pub license: Option<String>,
    /// File checksums
    pub files: Vec<FileChecksum>,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Total number of samples
    pub total_samples: usize,
    /// Total duration in seconds
    pub total_duration: f64,
    /// Total size in bytes
    pub total_size: u64,
    /// Number of speakers (if applicable)
    pub num_speakers: Option<usize>,
    /// Sample rate(s)
    pub sample_rates: Vec<u32>,
    /// Languages
    pub languages: Vec<String>,
}

impl DatasetManifest {
    /// Create new manifest
    pub fn new(name: String, version: DatasetVersion) -> Self {
        Self {
            name,
            version,
            created: Utc::now(),
            creator: None,
            description: None,
            license: None,
            files: Vec::new(),
            statistics: DatasetStatistics {
                total_samples: 0,
                total_duration: 0.0,
                total_size: 0,
                num_speakers: None,
                sample_rates: Vec::new(),
                languages: Vec::new(),
            },
            metadata: HashMap::new(),
        }
    }

    /// Add file to manifest
    pub fn add_file(&mut self, checksum: FileChecksum) {
        self.statistics.total_size += checksum.size;
        self.files.push(checksum);
    }

    /// Save manifest to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Load manifest from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let manifest = serde_json::from_reader(file)?;
        Ok(manifest)
    }

    /// Verify all files in manifest
    pub fn verify_all<P: AsRef<Path>>(&self, base_path: P) -> Result<VerificationReport> {
        let mut report = VerificationReport::new();

        for file in &self.files {
            match file.verify(&base_path) {
                Ok(true) => report.verified_files += 1,
                Ok(false) => {
                    report.failed_files += 1;
                    report
                        .errors
                        .push(format!("Checksum mismatch: {}", file.path.display()));
                }
                Err(e) => {
                    report.missing_files += 1;
                    report
                        .errors
                        .push(format!("Error verifying {}: {}", file.path.display(), e));
                }
            }
        }

        Ok(report)
    }

    /// Calculate manifest hash for reproducibility
    pub fn calculate_hash(&self) -> String {
        let mut hasher = md5::Context::new();

        // Hash files in sorted order for deterministic result
        let mut sorted_files = self.files.clone();
        sorted_files.sort_by(|a, b| a.path.cmp(&b.path));

        for file in sorted_files {
            hasher.consume(file.path.to_string_lossy().as_bytes());
            hasher.consume(file.md5.as_bytes());
        }

        let hash = hasher.compute();
        format!("{:x}", hash)
    }
}

/// Verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Number of verified files
    pub verified_files: usize,
    /// Number of failed verifications
    pub failed_files: usize,
    /// Number of missing files
    pub missing_files: usize,
    /// Error messages
    pub errors: Vec<String>,
}

impl VerificationReport {
    /// Create new report
    pub fn new() -> Self {
        Self {
            verified_files: 0,
            failed_files: 0,
            missing_files: 0,
            errors: Vec::new(),
        }
    }

    /// Check if verification passed
    pub fn is_valid(&self) -> bool {
        self.failed_files == 0 && self.missing_files == 0
    }

    /// Get total files checked
    pub fn total_files(&self) -> usize {
        self.verified_files + self.failed_files + self.missing_files
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset versioning builder
pub struct DatasetVersioningBuilder {
    name: String,
    version: DatasetVersion,
    base_path: PathBuf,
    manifest: DatasetManifest,
}

impl DatasetVersioningBuilder {
    /// Create new builder
    pub fn new<P: AsRef<Path>>(name: String, version: DatasetVersion, base_path: P) -> Self {
        let manifest = DatasetManifest::new(name.clone(), version.clone());

        Self {
            name,
            version,
            base_path: base_path.as_ref().to_path_buf(),
            manifest,
        }
    }

    /// Set creator
    pub fn creator(mut self, creator: String) -> Self {
        self.manifest.creator = Some(creator);
        self
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.manifest.description = Some(description);
        self
    }

    /// Set license
    pub fn license(mut self, license: String) -> Self {
        self.manifest.license = Some(license);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.manifest.metadata.insert(key, value);
        self
    }

    /// Scan directory and add all files
    pub fn scan_directory(mut self, pattern: Option<&str>) -> Result<Self> {
        for entry in WalkDir::new(&self.base_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                let path = entry.path();

                // Check pattern if specified
                if let Some(pat) = pattern {
                    if !path.to_string_lossy().contains(pat) {
                        continue;
                    }
                }

                let checksum = FileChecksum::calculate(path, &self.base_path)?;
                self.manifest.add_file(checksum);
            }
        }

        Ok(self)
    }

    /// Add specific file
    pub fn add_file<P: AsRef<Path>>(mut self, file_path: P) -> Result<Self> {
        let full_path = self.base_path.join(file_path.as_ref());
        let checksum = FileChecksum::calculate(&full_path, &self.base_path)?;
        self.manifest.add_file(checksum);
        Ok(self)
    }

    /// Update statistics
    pub fn statistics(mut self, stats: DatasetStatistics) -> Self {
        self.manifest.statistics = stats;
        self
    }

    /// Build manifest
    pub fn build(self) -> DatasetManifest {
        self.manifest
    }

    /// Build and save manifest
    pub fn build_and_save<P: AsRef<Path>>(self, output_path: P) -> Result<DatasetManifest> {
        let manifest = self.build();
        manifest.save(output_path)?;
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_version_new() {
        let version = DatasetVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert!(version.pre_release.is_none());
    }

    #[test]
    fn test_version_with_pre_release() {
        let version = DatasetVersion::with_pre_release(1, 2, 3, "beta.1".to_string());
        assert_eq!(version.pre_release, Some("beta.1".to_string()));
    }

    #[test]
    fn test_version_parse() {
        let version = DatasetVersion::parse("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        let version = DatasetVersion::parse("2.0.1-alpha").unwrap();
        assert_eq!(version.major, 2);
        assert_eq!(version.pre_release, Some("alpha".to_string()));
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(DatasetVersion::parse("1.2").is_err());
        assert!(DatasetVersion::parse("invalid").is_err());
        assert!(DatasetVersion::parse("1.a.3").is_err());
    }

    #[test]
    fn test_version_to_string() {
        let version = DatasetVersion::new(1, 2, 3);
        assert_eq!(version.as_string(), "1.2.3");

        let version = DatasetVersion::with_pre_release(1, 2, 3, "beta".to_string());
        assert_eq!(version.as_string(), "1.2.3-beta");
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = DatasetVersion::new(1, 2, 0);
        let v2 = DatasetVersion::new(1, 3, 0);
        let v3 = DatasetVersion::new(2, 0, 0);

        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
        assert!(!v3.is_compatible_with(&v1));
    }

    #[test]
    fn test_file_checksum_calculate() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Hello, world!").unwrap();
        drop(file);

        let checksum = FileChecksum::calculate(&file_path, &temp_dir.path().to_path_buf()).unwrap();

        assert_eq!(checksum.path, PathBuf::from("test.txt"));
        assert_eq!(checksum.size, 13);
        assert!(!checksum.md5.is_empty());
    }

    #[test]
    fn test_file_checksum_verify() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Hello, world!").unwrap();
        drop(file);

        let checksum = FileChecksum::calculate(&file_path, &temp_dir.path().to_path_buf()).unwrap();

        // Should verify successfully
        assert!(checksum.verify(temp_dir.path()).unwrap());

        // Modify file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Modified content").unwrap();
        drop(file);

        // Should fail verification
        assert!(!checksum.verify(temp_dir.path()).unwrap());
    }

    #[test]
    fn test_manifest_new() {
        let version = DatasetVersion::new(1, 0, 0);
        let manifest = DatasetManifest::new("test_dataset".to_string(), version.clone());

        assert_eq!(manifest.name, "test_dataset");
        assert_eq!(manifest.version, version);
        assert!(manifest.files.is_empty());
    }

    #[test]
    fn test_manifest_add_file() {
        let version = DatasetVersion::new(1, 0, 0);
        let mut manifest = DatasetManifest::new("test".to_string(), version);

        let checksum = FileChecksum {
            path: PathBuf::from("file.txt"),
            md5: "abc123".to_string(),
            size: 100,
            modified: Utc::now(),
        };

        manifest.add_file(checksum);

        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.statistics.total_size, 100);
    }

    #[test]
    fn test_manifest_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");

        let version = DatasetVersion::new(1, 0, 0);
        let manifest = DatasetManifest::new("test".to_string(), version);

        // Save
        manifest.save(&manifest_path).unwrap();

        // Load
        let loaded = DatasetManifest::load(&manifest_path).unwrap();

        assert_eq!(loaded.name, manifest.name);
        assert_eq!(loaded.version, manifest.version);
    }

    #[test]
    fn test_manifest_calculate_hash() {
        let version = DatasetVersion::new(1, 0, 0);
        let mut manifest = DatasetManifest::new("test".to_string(), version);

        let checksum = FileChecksum {
            path: PathBuf::from("file.txt"),
            md5: "abc123".to_string(),
            size: 100,
            modified: Utc::now(),
        };

        manifest.add_file(checksum);

        let hash = manifest.calculate_hash();
        assert!(!hash.is_empty());

        // Same hash for same files
        let hash2 = manifest.calculate_hash();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_verification_report() {
        let mut report = VerificationReport::new();

        assert_eq!(report.verified_files, 0);
        assert!(report.is_valid());

        report.verified_files = 5;
        assert!(report.is_valid());

        report.failed_files = 1;
        assert!(!report.is_valid());

        assert_eq!(report.total_files(), 6);
    }

    #[test]
    fn test_builder() {
        let temp_dir = TempDir::new().unwrap();
        let version = DatasetVersion::new(1, 0, 0);

        let manifest = DatasetVersioningBuilder::new("test".to_string(), version, temp_dir.path())
            .creator("Test Creator".to_string())
            .description("Test dataset".to_string())
            .license("MIT".to_string())
            .metadata("key".to_string(), "value".to_string())
            .build();

        assert_eq!(manifest.creator, Some("Test Creator".to_string()));
        assert_eq!(manifest.description, Some("Test dataset".to_string()));
        assert_eq!(manifest.license, Some("MIT".to_string()));
        assert_eq!(manifest.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_builder_scan_directory() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        for i in 0..3 {
            let file_path = temp_dir.path().join(format!("test_{}.txt", i));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(format!("Content {}", i).as_bytes()).unwrap();
        }

        let version = DatasetVersion::new(1, 0, 0);
        let manifest = DatasetVersioningBuilder::new("test".to_string(), version, temp_dir.path())
            .scan_directory(None)
            .unwrap()
            .build();

        assert_eq!(manifest.files.len(), 3);
        assert!(manifest.statistics.total_size > 0);
    }

    #[test]
    fn test_builder_scan_with_pattern() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        File::create(temp_dir.path().join("test.txt"))
            .unwrap()
            .write_all(b"test")
            .unwrap();
        File::create(temp_dir.path().join("data.csv"))
            .unwrap()
            .write_all(b"data")
            .unwrap();

        let version = DatasetVersion::new(1, 0, 0);
        let manifest = DatasetVersioningBuilder::new("test".to_string(), version, temp_dir.path())
            .scan_directory(Some(".txt"))
            .unwrap()
            .build();

        // Should only include .txt file
        assert_eq!(manifest.files.len(), 1);
        assert!(manifest.files[0].path.to_string_lossy().contains(".txt"));
    }
}
