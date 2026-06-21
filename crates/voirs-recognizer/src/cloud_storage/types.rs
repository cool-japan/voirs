use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
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
    pub(super) config: Arc<RwLock<CloudStorageConfig>>,

    /// Model metadata cache
    pub(super) metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,

    /// Download statistics
    pub(super) download_stats: Arc<RwLock<DownloadStatistics>>,

    /// Active downloads
    pub(super) active_downloads: Arc<RwLock<HashMap<String, DownloadProgress>>>,
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
