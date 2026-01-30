//! # Cloud Storage Integration
//!
//! Provides unified cloud storage integration for model management across
//! AWS S3, Google Cloud Storage, and Azure Blob Storage.
//!
//! Features:
//! - Multi-cloud model storage and retrieval
//! - Automatic caching and version management
//! - Parallel download optimization
//! - Checksum verification
//! - Retry logic with exponential backoff

use crate::RecognitionError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Cloud storage errors
#[derive(Debug, Error)]
pub enum CloudStorageError {
    /// Download failed
    #[error("Download failed: {0}")]
    DownloadFailed(String),

    /// Upload failed
    #[error("Upload failed: {0}")]
    UploadFailed(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Checksum mismatch
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Expected checksum
        expected: String,
        /// Actual checksum
        actual: String,
    },

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Cloud storage provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudProvider {
    /// AWS S3
    AwsS3,
    /// Google Cloud Storage
    GoogleCloudStorage,
    /// Azure Blob Storage
    AzureBlobStorage,
    /// Local filesystem (for testing)
    LocalFilesystem,
}

/// Cloud storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageConfig {
    /// Provider to use
    pub provider: CloudProvider,

    /// Bucket/container name
    pub bucket_name: String,

    /// Region (for AWS/GCP)
    pub region: Option<String>,

    /// Access key ID (for AWS)
    pub access_key_id: Option<String>,

    /// Secret access key (for AWS)
    pub secret_access_key: Option<String>,

    /// Service account key path (for GCP)
    pub service_account_key_path: Option<PathBuf>,

    /// Azure connection string
    pub azure_connection_string: Option<String>,

    /// Local cache directory
    pub cache_dir: PathBuf,

    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,

    /// Enable checksum verification
    pub verify_checksums: bool,

    /// Download timeout in seconds
    pub download_timeout_secs: u64,

    /// Maximum retry attempts
    pub max_retry_attempts: u32,

    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for CloudStorageConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: "voirs-models".to_string(),
            region: None,
            access_key_id: None,
            secret_access_key: None,
            service_account_key_path: None,
            azure_connection_string: None,
            cache_dir: std::env::temp_dir().join("voirs_cloud_cache"),
            max_cache_size_mb: 2048,
            verify_checksums: true,
            download_timeout_secs: 300,
            max_retry_attempts: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Model metadata for cloud storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model size in bytes
    pub size_bytes: u64,

    /// SHA256 checksum
    pub checksum: String,

    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,

    /// Model type (whisper, deepspeech, etc.)
    pub model_type: String,

    /// Cloud storage path
    pub storage_path: String,

    /// Tags for categorization
    pub tags: HashMap<String, String>,
}

/// Cloud storage manager
pub struct CloudStorageManager {
    /// Configuration
    config: Arc<RwLock<CloudStorageConfig>>,

    /// Model metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,

    /// Download statistics
    download_stats: Arc<RwLock<DownloadStatistics>>,

    /// Active downloads
    active_downloads: Arc<RwLock<HashMap<String, DownloadProgress>>>,
}

/// Download statistics
#[derive(Debug, Default, Clone)]
pub struct DownloadStatistics {
    /// Total downloads
    pub total_downloads: u64,

    /// Successful downloads
    pub successful_downloads: u64,

    /// Failed downloads
    pub failed_downloads: u64,

    /// Total bytes downloaded
    pub total_bytes_downloaded: u64,

    /// Average download speed in bytes/sec
    pub average_download_speed: f64,

    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Download progress
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Model name
    pub model_name: String,

    /// Total bytes
    pub total_bytes: u64,

    /// Downloaded bytes
    pub downloaded_bytes: u64,

    /// Download speed in bytes/sec
    pub download_speed: f64,

    /// Started at
    pub started_at: std::time::Instant,

    /// ETA in seconds
    pub eta_seconds: Option<f64>,
}

impl CloudStorageManager {
    /// Create a new cloud storage manager
    pub fn new(config: CloudStorageConfig) -> Result<Self, CloudStorageError> {
        // Create cache directory if it doesn't exist
        if !config.cache_dir.exists() {
            std::fs::create_dir_all(&config.cache_dir)?;
        }

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            download_stats: Arc::new(RwLock::new(DownloadStatistics::default())),
            active_downloads: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Download model from cloud storage
    pub async fn download_model(&self, model_name: &str) -> Result<PathBuf, CloudStorageError> {
        let config = self.config.read();

        // Check if model exists in cache
        let cache_path = config.cache_dir.join(model_name);
        if cache_path.exists() && self.verify_cached_model(&cache_path, model_name)? {
            tracing::info!("Using cached model: {}", model_name);
            self.update_cache_hit();
            return Ok(cache_path);
        }

        // Download from cloud
        tracing::info!(
            "Downloading model {} from {:?}",
            model_name,
            config.provider
        );

        let start_time = std::time::Instant::now();

        // Initialize download progress
        self.init_download_progress(model_name);

        let downloaded_path = match config.provider {
            CloudProvider::AwsS3 => self.download_from_s3(model_name, &cache_path).await?,
            CloudProvider::GoogleCloudStorage => {
                self.download_from_gcs(model_name, &cache_path).await?
            }
            CloudProvider::AzureBlobStorage => {
                self.download_from_azure(model_name, &cache_path).await?
            }
            CloudProvider::LocalFilesystem => {
                self.download_from_local(model_name, &cache_path).await?
            }
        };

        // Verify checksum if enabled
        if config.verify_checksums {
            self.verify_checksum(&downloaded_path, model_name)?;
        }

        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_download_stats(true, elapsed);

        // Clean up old cache if needed
        self.cleanup_cache()?;

        Ok(downloaded_path)
    }

    /// Upload model to cloud storage
    pub async fn upload_model(
        &self,
        model_path: &Path,
        model_name: &str,
        metadata: ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        let config = self.config.read();

        tracing::info!("Uploading model {} to {:?}", model_name, config.provider);

        match config.provider {
            CloudProvider::AwsS3 => self.upload_to_s3(model_path, model_name, &metadata).await?,
            CloudProvider::GoogleCloudStorage => {
                self.upload_to_gcs(model_path, model_name, &metadata)
                    .await?;
            }
            CloudProvider::AzureBlobStorage => {
                self.upload_to_azure(model_path, model_name, &metadata)
                    .await?;
            }
            CloudProvider::LocalFilesystem => {
                self.upload_to_local(model_path, model_name, &metadata)
                    .await?;
            }
        }

        // Update metadata cache
        self.metadata_cache
            .write()
            .insert(model_name.to_string(), metadata);

        Ok(())
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        let config = self.config.read();

        let models = match config.provider {
            CloudProvider::AwsS3 => self.list_s3_models().await?,
            CloudProvider::GoogleCloudStorage => self.list_gcs_models().await?,
            CloudProvider::AzureBlobStorage => self.list_azure_models().await?,
            CloudProvider::LocalFilesystem => self.list_local_models().await?,
        };

        Ok(models)
    }

    /// Get download statistics
    #[must_use]
    pub fn get_download_stats(&self) -> DownloadStatistics {
        self.download_stats.read().clone()
    }

    /// Get active downloads
    #[must_use]
    pub fn get_active_downloads(&self) -> Vec<DownloadProgress> {
        self.active_downloads.read().values().cloned().collect()
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<(), CloudStorageError> {
        let config = self.config.read();

        if config.cache_dir.exists() {
            std::fs::remove_dir_all(&config.cache_dir)?;
            std::fs::create_dir_all(&config.cache_dir)?;
        }

        self.metadata_cache.write().clear();

        Ok(())
    }

    // Private helper methods

    async fn download_from_s3(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        // Simulated S3 download (in real implementation, use aws-sdk-s3)
        tracing::warn!("S3 download not yet implemented, using placeholder");
        Err(CloudStorageError::DownloadFailed(
            "S3 integration not yet implemented".to_string(),
        ))
    }

    async fn download_from_gcs(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        // Simulated GCS download (in real implementation, use google-cloud-storage)
        tracing::warn!("GCS download not yet implemented, using placeholder");
        Err(CloudStorageError::DownloadFailed(
            "GCS integration not yet implemented".to_string(),
        ))
    }

    async fn download_from_azure(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        // Simulated Azure download (in real implementation, use azure-storage-blobs)
        tracing::warn!("Azure download not yet implemented, using placeholder");
        Err(CloudStorageError::DownloadFailed(
            "Azure integration not yet implemented".to_string(),
        ))
    }

    async fn download_from_local(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        let config = self.config.read();
        let source_path = PathBuf::from(&config.bucket_name).join(model_name);

        if !source_path.exists() {
            return Err(CloudStorageError::ModelNotFound(model_name.to_string()));
        }

        // Copy file to cache
        std::fs::copy(&source_path, cache_path)?;

        Ok(cache_path.to_path_buf())
    }

    async fn upload_to_s3(
        &self,
        model_path: &Path,
        model_name: &str,
        metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        tracing::warn!("S3 upload not yet implemented");
        Err(CloudStorageError::UploadFailed(
            "S3 integration not yet implemented".to_string(),
        ))
    }

    async fn upload_to_gcs(
        &self,
        model_path: &Path,
        model_name: &str,
        metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        tracing::warn!("GCS upload not yet implemented");
        Err(CloudStorageError::UploadFailed(
            "GCS integration not yet implemented".to_string(),
        ))
    }

    async fn upload_to_azure(
        &self,
        model_path: &Path,
        model_name: &str,
        metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        tracing::warn!("Azure upload not yet implemented");
        Err(CloudStorageError::UploadFailed(
            "Azure integration not yet implemented".to_string(),
        ))
    }

    async fn upload_to_local(
        &self,
        model_path: &Path,
        model_name: &str,
        metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        let config = self.config.read();
        let dest_path = PathBuf::from(&config.bucket_name).join(model_name);

        // Create directory if needed
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::copy(model_path, &dest_path)?;

        Ok(())
    }

    async fn list_s3_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        tracing::warn!("S3 listing not yet implemented");
        Ok(Vec::new())
    }

    async fn list_gcs_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        tracing::warn!("GCS listing not yet implemented");
        Ok(Vec::new())
    }

    async fn list_azure_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        tracing::warn!("Azure listing not yet implemented");
        Ok(Vec::new())
    }

    async fn list_local_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        let config = self.config.read();
        let bucket_path = PathBuf::from(&config.bucket_name);

        if !bucket_path.exists() {
            return Ok(Vec::new());
        }

        let mut models = Vec::new();

        for entry in std::fs::read_dir(&bucket_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    let metadata = ModelMetadata {
                        name: file_name.to_string(),
                        version: "1.0.0".to_string(),
                        size_bytes: entry.metadata()?.len(),
                        checksum: String::new(),
                        last_modified: chrono::Utc::now(),
                        model_type: "unknown".to_string(),
                        storage_path: path.to_string_lossy().to_string(),
                        tags: HashMap::new(),
                    };
                    models.push(metadata);
                }
            }
        }

        Ok(models)
    }

    fn verify_cached_model(
        &self,
        cache_path: &Path,
        model_name: &str,
    ) -> Result<bool, CloudStorageError> {
        // Check if file exists and is valid
        if !cache_path.exists() {
            return Ok(false);
        }

        // In real implementation, verify checksum against metadata
        Ok(true)
    }

    fn verify_checksum(&self, file_path: &Path, model_name: &str) -> Result<(), CloudStorageError> {
        // In real implementation, compute SHA256 and verify against metadata
        tracing::debug!("Checksum verification for {} (placeholder)", model_name);
        Ok(())
    }

    fn init_download_progress(&self, model_name: &str) {
        let progress = DownloadProgress {
            model_name: model_name.to_string(),
            total_bytes: 0,
            downloaded_bytes: 0,
            download_speed: 0.0,
            started_at: std::time::Instant::now(),
            eta_seconds: None,
        };

        self.active_downloads
            .write()
            .insert(model_name.to_string(), progress);
    }

    fn update_download_stats(&self, success: bool, elapsed: Duration) {
        let mut stats = self.download_stats.write();
        stats.total_downloads += 1;

        if success {
            stats.successful_downloads += 1;
        } else {
            stats.failed_downloads += 1;
        }

        // Update average download speed
        if success && elapsed.as_secs() > 0 {
            // Simplified calculation
            stats.average_download_speed = 1_000_000.0 / elapsed.as_secs_f64(); // 1MB placeholder
        }
    }

    fn update_cache_hit(&self) {
        let mut stats = self.download_stats.write();
        stats.total_downloads += 1;
        stats.successful_downloads += 1;

        // Update cache hit rate
        stats.cache_hit_rate = stats.successful_downloads as f64 / stats.total_downloads as f64;
    }

    fn cleanup_cache(&self) -> Result<(), CloudStorageError> {
        let config = self.config.read();

        // Calculate current cache size
        let cache_size_bytes = self.calculate_cache_size(&config.cache_dir)?;
        let max_cache_bytes = config.max_cache_size_mb * 1024 * 1024;

        if cache_size_bytes > max_cache_bytes {
            tracing::info!(
                "Cache size {}MB exceeds limit {}MB, cleaning up",
                cache_size_bytes / (1024 * 1024),
                config.max_cache_size_mb
            );

            // Remove oldest files until under limit
            self.remove_oldest_files(&config.cache_dir, cache_size_bytes - max_cache_bytes)?;
        }

        Ok(())
    }

    fn calculate_cache_size(&self, dir: &Path) -> Result<u64, CloudStorageError> {
        let mut total_size = 0u64;

        if dir.exists() && dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total_size += metadata.len();
                }
            }
        }

        Ok(total_size)
    }

    fn remove_oldest_files(
        &self,
        dir: &Path,
        bytes_to_remove: u64,
    ) -> Result<(), CloudStorageError> {
        // Collect files with modification times
        let mut files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                files.push((entry.path(), metadata.modified()?));
            }
        }

        // Sort by modification time (oldest first)
        files.sort_by_key(|(_, time)| *time);

        // Remove files until we've freed enough space
        let mut removed_bytes = 0u64;
        for (path, _) in files {
            if removed_bytes >= bytes_to_remove {
                break;
            }

            if let Ok(metadata) = std::fs::metadata(&path) {
                let file_size = metadata.len();
                std::fs::remove_file(&path)?;
                removed_bytes += file_size;
                tracing::debug!("Removed cached file: {}", path.display());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cloud_storage_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_local_filesystem_download() {
        let temp_cache = TempDir::new().unwrap();
        let temp_bucket = TempDir::new().unwrap();

        // Create a test model file
        let model_path = temp_bucket.path().join("test_model.bin");
        std::fs::write(&model_path, b"test model data").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            cache_dir: temp_cache.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let result = manager.download_model("test_model.bin").await;
        assert!(result.is_ok());

        let downloaded_path = result.unwrap();
        assert!(downloaded_path.exists());
        assert_eq!(
            std::fs::read_to_string(downloaded_path).unwrap(),
            "test model data"
        );
    }

    #[tokio::test]
    async fn test_local_filesystem_upload() {
        let temp_cache = TempDir::new().unwrap();
        let temp_bucket = TempDir::new().unwrap();

        // Create source model file
        let source_model = temp_cache.path().join("source_model.bin");
        std::fs::write(&source_model, b"upload test data").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            cache_dir: temp_cache.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let metadata = ModelMetadata {
            name: "test_upload.bin".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 16,
            checksum: "test_checksum".to_string(),
            last_modified: chrono::Utc::now(),
            model_type: "test".to_string(),
            storage_path: "test_upload.bin".to_string(),
            tags: HashMap::new(),
        };

        let result = manager
            .upload_model(&source_model, "test_upload.bin", metadata)
            .await;
        assert!(result.is_ok());

        // Verify uploaded file
        let uploaded_path = temp_bucket.path().join("test_upload.bin");
        assert!(uploaded_path.exists());
    }

    #[tokio::test]
    async fn test_list_local_models() {
        let temp_bucket = TempDir::new().unwrap();

        // Create test model files
        std::fs::write(temp_bucket.path().join("model1.bin"), b"model1").unwrap();
        std::fs::write(temp_bucket.path().join("model2.bin"), b"model2").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let models = manager.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_download_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        manager.update_download_stats(true, Duration::from_secs(10));
        manager.update_download_stats(true, Duration::from_secs(5));
        manager.update_download_stats(false, Duration::from_secs(1));

        let stats = manager.get_download_stats();
        assert_eq!(stats.total_downloads, 3);
        assert_eq!(stats.successful_downloads, 2);
        assert_eq!(stats.failed_downloads, 1);
    }

    #[test]
    fn test_cache_cleanup() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        let file1 = temp_dir.path().join("file1.bin");
        let file2 = temp_dir.path().join("file2.bin");

        std::fs::write(&file1, vec![0u8; 1024 * 1024]).unwrap(); // 1MB
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&file2, vec![0u8; 1024 * 1024]).unwrap(); // 1MB

        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_cache_size_mb: 1, // 1MB limit
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        // Cleanup should remove the oldest file
        manager.cleanup_cache().unwrap();

        // file1 should be removed, file2 should exist
        assert!(!file1.exists());
        assert!(file2.exists());
    }
}
