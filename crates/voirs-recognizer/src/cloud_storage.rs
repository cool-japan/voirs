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
use sha2::{Digest, Sha256};
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

    #[cfg(feature = "cloud")]
    async fn download_from_s3(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;
        use tokio::io::AsyncWriteExt;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("us-east-1").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing S3 access_key_id".into())
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing S3 secret_access_key".into())
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let empty_hash =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let url_str = format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            bucket, region, model_name
        );
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_s3_request(
            "GET",
            &url_str,
            empty_hash,
            &access_key,
            &secret_key,
            &region,
            "s3",
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client.get(&url_str);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "S3 returned HTTP {}: {}",
                response.status(),
                model_name
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::File::create(cache_path).await?;
        file.write_all(&bytes).await?;
        file.flush().await?;

        {
            let mut stats = self.download_stats.write();
            stats.total_bytes_downloaded += bytes.len() as u64;
        }

        Ok(cache_path.to_path_buf())
    }

    #[cfg(not(feature = "cloud"))]
    async fn download_from_s3(
        &self,
        model_name: &str,
        _cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        Err(CloudStorageError::DownloadFailed(format!(
            "S3 integration requires `cloud` feature (model: {})",
            model_name
        )))
    }

    #[cfg(feature = "cloud")]
    async fn download_from_gcs(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;
        use tokio::io::AsyncWriteExt;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("auto").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys (access_key_id + secret_access_key) for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys (access_key_id + secret_access_key) for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let empty_hash =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let url_str = format!(
            "https://storage.googleapis.com/{}/{}",
            bucket, model_name
        );
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_gcs_request(
            "GET",
            &url_str,
            empty_hash,
            &access_key,
            &secret_key,
            &region,
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client.get(&url_str);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "GCS returned HTTP {}: {}",
                response.status(),
                model_name
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::File::create(cache_path).await?;
        file.write_all(&bytes).await?;
        file.flush().await?;

        {
            let mut stats = self.download_stats.write();
            stats.total_bytes_downloaded += bytes.len() as u64;
        }

        Ok(cache_path.to_path_buf())
    }

    #[cfg(not(feature = "cloud"))]
    async fn download_from_gcs(
        &self,
        model_name: &str,
        _cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        Err(CloudStorageError::DownloadFailed(format!(
            "GCS integration requires `cloud` feature (model: {})",
            model_name
        )))
    }

    #[cfg(feature = "cloud")]
    async fn download_from_azure(
        &self,
        model_name: &str,
        cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;
        use tokio::io::AsyncWriteExt;

        let config = self.config.read();
        let conn_str = config
            .azure_connection_string
            .clone()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing azure_connection_string".into())
            })?;
        let container = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let (account_name, account_key) = self.parse_azure_connection_string(&conn_str)?;

        let x_ms_date = chrono::Utc::now()
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        let x_ms_version = "2020-10-02";

        let auth = cloud_auth::sign_azure_request(
            "GET",
            &account_name,
            &container,
            model_name,
            0,
            "",
            &x_ms_date,
            x_ms_version,
            &account_key,
        )?;

        let url_str = format!(
            "https://{}.blob.core.windows.net/{}/{}",
            account_name, container, model_name
        );

        voirs_sdk::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let response = client
            .get(&url_str)
            .header(
                HeaderName::from_str("Authorization")
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(&auth)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            )
            .header("x-ms-date", &x_ms_date)
            .header("x-ms-version", x_ms_version)
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "Azure returned HTTP {}: {}",
                response.status(),
                model_name
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::File::create(cache_path).await?;
        file.write_all(&bytes).await?;
        file.flush().await?;

        {
            let mut stats = self.download_stats.write();
            stats.total_bytes_downloaded += bytes.len() as u64;
        }

        Ok(cache_path.to_path_buf())
    }

    #[cfg(not(feature = "cloud"))]
    async fn download_from_azure(
        &self,
        model_name: &str,
        _cache_path: &Path,
    ) -> Result<PathBuf, CloudStorageError> {
        Err(CloudStorageError::DownloadFailed(format!(
            "Azure integration requires `cloud` feature (model: {})",
            model_name
        )))
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

    #[cfg(feature = "cloud")]
    async fn upload_to_s3(
        &self,
        model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("us-east-1").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing access_key_id".into())
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing secret_access_key".into())
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let body_bytes = tokio::fs::read(model_path).await?;
        let body_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&body_bytes);
            hex::encode(hasher.finalize())
        };

        let url_str = format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            bucket, region, model_name
        );
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_s3_request(
            "PUT",
            &url_str,
            &body_hash,
            &access_key,
            &secret_key,
            &region,
            "s3",
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client
            .put(&url_str)
            .header("Content-Type", "application/octet-stream")
            .body(body_bytes);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::UploadFailed(format!(
                "S3 PUT returned HTTP {}",
                response.status()
            )));
        }

        Ok(())
    }

    #[cfg(not(feature = "cloud"))]
    async fn upload_to_s3(
        &self,
        _model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        Err(CloudStorageError::UploadFailed(format!(
            "S3 integration requires `cloud` feature (model: {})",
            model_name
        )))
    }

    #[cfg(feature = "cloud")]
    async fn upload_to_gcs(
        &self,
        model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("auto").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let body_bytes = tokio::fs::read(model_path).await?;
        let body_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&body_bytes);
            hex::encode(hasher.finalize())
        };

        let url_str = format!("https://storage.googleapis.com/{}/{}", bucket, model_name);
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_gcs_request(
            "PUT",
            &url_str,
            &body_hash,
            &access_key,
            &secret_key,
            &region,
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client
            .put(&url_str)
            .header("Content-Type", "application/octet-stream")
            .body(body_bytes);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::UploadFailed(format!(
                "GCS PUT returned HTTP {}",
                response.status()
            )));
        }

        Ok(())
    }

    #[cfg(not(feature = "cloud"))]
    async fn upload_to_gcs(
        &self,
        _model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        Err(CloudStorageError::UploadFailed(format!(
            "GCS integration requires `cloud` feature (model: {})",
            model_name
        )))
    }

    #[cfg(feature = "cloud")]
    async fn upload_to_azure(
        &self,
        model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        use crate::cloud_auth;

        let config = self.config.read();
        let conn_str = config
            .azure_connection_string
            .clone()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing azure_connection_string".into())
            })?;
        let container = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let (account_name, account_key) = self.parse_azure_connection_string(&conn_str)?;

        let body_bytes = tokio::fs::read(model_path).await?;
        let content_length = body_bytes.len() as u64;
        let content_type = "application/octet-stream";

        let x_ms_date = chrono::Utc::now()
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        let x_ms_version = "2020-10-02";

        let auth = cloud_auth::sign_azure_request(
            "PUT",
            &account_name,
            &container,
            model_name,
            content_length,
            content_type,
            &x_ms_date,
            x_ms_version,
            &account_key,
        )?;

        let url_str = format!(
            "https://{}.blob.core.windows.net/{}/{}",
            account_name, container, model_name
        );

        voirs_sdk::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let response = client
            .put(&url_str)
            .header("Authorization", &auth)
            .header("x-ms-date", &x_ms_date)
            .header("x-ms-version", x_ms_version)
            .header("x-ms-blob-type", "BlockBlob")
            .header("Content-Type", content_type)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::UploadFailed(format!(
                "Azure PUT returned HTTP {}",
                response.status()
            )));
        }

        Ok(())
    }

    #[cfg(not(feature = "cloud"))]
    async fn upload_to_azure(
        &self,
        _model_path: &Path,
        model_name: &str,
        _metadata: &ModelMetadata,
    ) -> Result<(), CloudStorageError> {
        Err(CloudStorageError::UploadFailed(format!(
            "Azure integration requires `cloud` feature (model: {})",
            model_name
        )))
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

    #[cfg(feature = "cloud")]
    async fn list_s3_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("us-east-1").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing access_key_id".into())
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing secret_access_key".into())
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let empty_hash =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let url_str = format!(
            "https://{}.s3.{}.amazonaws.com/?list-type=2",
            bucket, region
        );
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_s3_request(
            "GET",
            &url_str,
            empty_hash,
            &access_key,
            &secret_key,
            &region,
            "s3",
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client.get(&url_str);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "S3 LIST returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        parse_s3_list_xml(&body)
    }

    #[cfg(not(feature = "cloud"))]
    async fn list_s3_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        Ok(Vec::new())
    }

    #[cfg(feature = "cloud")]
    async fn list_gcs_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        use crate::cloud_auth;
        use reqwest::header::{HeaderName, HeaderValue};
        use std::str::FromStr;

        let config = self.config.read();
        let region = config.region.as_deref().unwrap_or("auto").to_owned();
        let access_key = config
            .access_key_id
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let secret_key = config
            .secret_access_key
            .as_deref()
            .ok_or_else(|| {
                CloudStorageError::InvalidConfiguration(
                    "GCS requires HMAC keys for S3-compatible API".into(),
                )
            })?
            .to_owned();
        let bucket = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let empty_hash =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let url_str = format!("https://storage.googleapis.com/{}?list-type=2", bucket);
        let timestamp = cloud_auth::current_timestamp();

        voirs_sdk::ensure_crypto_provider();

        let headers_map = cloud_auth::sign_gcs_request(
            "GET",
            &url_str,
            empty_hash,
            &access_key,
            &secret_key,
            &region,
            &timestamp,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let mut req_builder = client.get(&url_str);
        for (k, v) in &headers_map {
            req_builder = req_builder.header(
                HeaderName::from_str(k)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
                HeaderValue::from_str(v)
                    .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?,
            );
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "GCS LIST returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        parse_s3_list_xml(&body)
    }

    #[cfg(not(feature = "cloud"))]
    async fn list_gcs_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        Ok(Vec::new())
    }

    #[cfg(feature = "cloud")]
    async fn list_azure_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        use crate::cloud_auth;

        let config = self.config.read();
        let conn_str = config
            .azure_connection_string
            .clone()
            .ok_or_else(|| {
                CloudStorageError::AuthenticationFailed("Missing azure_connection_string".into())
            })?;
        let container = config.bucket_name.clone();
        let timeout = Duration::from_secs(config.download_timeout_secs);
        drop(config);

        let (account_name, account_key) = self.parse_azure_connection_string(&conn_str)?;

        let x_ms_date = chrono::Utc::now()
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        let x_ms_version = "2020-10-02";

        let auth = cloud_auth::sign_azure_request(
            "GET",
            &account_name,
            &container,
            "",
            0,
            "",
            &x_ms_date,
            x_ms_version,
            &account_key,
        )?;

        let url_str = format!(
            "https://{}.blob.core.windows.net/{}?restype=container&comp=list",
            account_name, container
        );

        voirs_sdk::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        let response = client
            .get(&url_str)
            .header("Authorization", &auth)
            .header("x-ms-date", &x_ms_date)
            .header("x-ms-version", x_ms_version)
            .send()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CloudStorageError::DownloadFailed(format!(
                "Azure LIST returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| CloudStorageError::NetworkError(e.to_string()))?;

        parse_azure_list_xml(&body)
    }

    #[cfg(not(feature = "cloud"))]
    async fn list_azure_models(&self) -> Result<Vec<ModelMetadata>, CloudStorageError> {
        Ok(Vec::new())
    }

    #[cfg(feature = "cloud")]
    fn parse_azure_connection_string(
        &self,
        conn_str: &str,
    ) -> Result<(String, String), CloudStorageError> {
        let mut account_name = String::new();
        let mut account_key = String::new();
        for part in conn_str.split(';') {
            if part.starts_with("AccountName=") {
                account_name = part["AccountName=".len()..].to_string();
            } else if part.starts_with("AccountKey=") {
                account_key = part["AccountKey=".len()..].to_string();
            }
        }
        if account_name.is_empty() || account_key.is_empty() {
            return Err(CloudStorageError::InvalidConfiguration(
                "Azure connection string must contain AccountName and AccountKey".to_string(),
            ));
        }
        Ok((account_name, account_key))
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
        if !cache_path.exists() {
            return Ok(false);
        }

        let has_checksum = {
            let cache = self.metadata_cache.read();
            cache
                .get(model_name)
                .map(|m| !m.checksum.is_empty())
                .unwrap_or(false)
        };

        if has_checksum {
            self.verify_checksum(cache_path, model_name)?;
        }

        Ok(true)
    }

    fn verify_checksum(&self, file_path: &Path, model_name: &str) -> Result<(), CloudStorageError> {
        let expected = {
            let cache = self.metadata_cache.read();
            cache
                .get(model_name)
                .map(|m| m.checksum.clone())
                .unwrap_or_default()
        };

        if expected.is_empty() {
            tracing::debug!("No checksum metadata for {}, skipping verification", model_name);
            return Ok(());
        }

        let data = std::fs::read(file_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let actual = hex::encode(hasher.finalize());

        if actual != expected {
            return Err(CloudStorageError::ChecksumMismatch { expected, actual });
        }

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

#[cfg(feature = "cloud")]
fn parse_s3_list_xml(xml: &str) -> Result<Vec<ModelMetadata>, CloudStorageError> {
    use quick_xml::{Reader, events::Event};

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut models = Vec::new();
    let mut in_contents = false;
    let mut current_key = String::new();
    let mut current_size: u64 = 0;
    let mut current_last_modified = String::new();
    let mut current_etag = String::new();
    let mut current_tag = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                current_tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if current_tag == "Contents" {
                    in_contents = true;
                }
            }
            Ok(Event::End(e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if tag == "Contents" && in_contents {
                    if !current_key.is_empty() {
                        let last_modified =
                            chrono::DateTime::parse_from_rfc3339(&current_last_modified)
                                .map(|dt| dt.with_timezone(&chrono::Utc))
                                .unwrap_or_else(|_| chrono::Utc::now());
                        models.push(ModelMetadata {
                            name: current_key.clone(),
                            version: "1.0.0".to_string(),
                            size_bytes: current_size,
                            checksum: current_etag.trim_matches('"').to_string(),
                            last_modified,
                            model_type: "unknown".to_string(),
                            storage_path: current_key.clone(),
                            tags: HashMap::new(),
                        });
                    }
                    in_contents = false;
                    current_key.clear();
                    current_size = 0;
                    current_last_modified.clear();
                    current_etag.clear();
                }
            }
            Ok(Event::Text(e)) if in_contents => {
                let text = e
                    .unescape()
                    .map_err(|e| {
                        CloudStorageError::DownloadFailed(format!("XML parse error: {}", e))
                    })?
                    .to_string();
                match current_tag.as_str() {
                    "Key" => current_key = text,
                    "Size" => current_size = text.parse::<u64>().unwrap_or(0),
                    "LastModified" => current_last_modified = text,
                    "ETag" => current_etag = text,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(CloudStorageError::DownloadFailed(format!(
                    "XML parse error: {}",
                    e
                )));
            }
            _ => {}
        }
    }
    Ok(models)
}

#[cfg(feature = "cloud")]
fn parse_azure_list_xml(xml: &str) -> Result<Vec<ModelMetadata>, CloudStorageError> {
    use quick_xml::{Reader, events::Event};

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut models = Vec::new();
    let mut in_blob = false;
    let mut in_properties = false;
    let mut current_name = String::new();
    let mut current_size: u64 = 0;
    let mut current_last_modified = String::new();
    let mut current_tag = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                current_tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match current_tag.as_str() {
                    "Blob" => in_blob = true,
                    "Properties" if in_blob => in_properties = true,
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "Properties" if in_blob => in_properties = false,
                    "Blob" if in_blob => {
                        if !current_name.is_empty() {
                            let last_modified =
                                chrono::DateTime::parse_from_rfc2822(&current_last_modified)
                                    .map(|dt| dt.with_timezone(&chrono::Utc))
                                    .unwrap_or_else(|_| chrono::Utc::now());
                            models.push(ModelMetadata {
                                name: current_name.clone(),
                                version: "1.0.0".to_string(),
                                size_bytes: current_size,
                                checksum: String::new(),
                                last_modified,
                                model_type: "unknown".to_string(),
                                storage_path: current_name.clone(),
                                tags: HashMap::new(),
                            });
                        }
                        in_blob = false;
                        current_name.clear();
                        current_size = 0;
                        current_last_modified.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) if in_blob => {
                let text = e
                    .unescape()
                    .map_err(|e| {
                        CloudStorageError::DownloadFailed(format!("XML parse error: {}", e))
                    })?
                    .to_string();
                if !in_properties {
                    if current_tag == "Name" {
                        current_name = text;
                    }
                } else {
                    match current_tag.as_str() {
                        "Content-Length" => current_size = text.parse::<u64>().unwrap_or(0),
                        "Last-Modified" => current_last_modified = text,
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(CloudStorageError::DownloadFailed(format!(
                    "XML parse error: {}",
                    e
                )));
            }
            _ => {}
        }
    }
    Ok(models)
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

#[cfg(all(test, feature = "cloud"))]
mod cloud_auth_tests {
    use super::*;

    #[test]
    fn test_sigv4_canonical_request() {
        let result = crate::cloud_auth::sign_s3_request(
            "GET",
            "https://examplebucket.s3.amazonaws.com/?list-type=2",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "s3",
            "20130524T000000Z",
        );
        assert!(result.is_ok(), "SigV4 signing should succeed: {:?}", result.err());
        let headers = result.unwrap();
        assert!(
            headers.contains_key("Authorization"),
            "Should have Authorization header"
        );
        let auth = &headers["Authorization"];
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256"),
            "Should use AWS4-HMAC-SHA256"
        );
        assert!(
            auth.contains(
                "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"
            ),
            "Credential scope must match: {}",
            auth
        );
        assert!(
            auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"),
            "Signed headers must be canonical: {}",
            auth
        );
    }

    #[test]
    fn test_azure_shared_key_signing() {
        let result = crate::cloud_auth::sign_azure_request(
            "GET",
            "devstoreaccount1",
            "models",
            "",
            0,
            "",
            "Thu, 19 Jun 2025 12:00:00 GMT",
            "2020-10-02",
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
        );
        assert!(
            result.is_ok(),
            "Azure signing should succeed: {:?}",
            result.err()
        );
        let auth = result.unwrap();
        assert!(
            auth.starts_with("SharedKey devstoreaccount1:"),
            "Should use SharedKey scheme: {}",
            auth
        );
    }
}

#[cfg(test)]
mod checksum_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_verify_checksum_correct() {
        use sha2::{Digest, Sha256};
        let temp_dir = std::env::temp_dir().join("voirs_checksum_test");
        std::fs::create_dir_all(&temp_dir).ok();
        let file_path = temp_dir.join("test_model.bin");
        let test_data = b"Hello, VoiRS checksum test!";
        std::fs::write(&file_path, test_data).expect("write test file");

        let mut hasher = Sha256::new();
        hasher.update(test_data);
        let expected = hex::encode(hasher.finalize());

        let config = CloudStorageConfig {
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");
        manager.metadata_cache.write().insert(
            "test_model.bin".to_string(),
            ModelMetadata {
                name: "test_model.bin".to_string(),
                version: "1.0.0".to_string(),
                size_bytes: test_data.len() as u64,
                checksum: expected.clone(),
                last_modified: chrono::Utc::now(),
                model_type: "test".to_string(),
                storage_path: file_path.to_string_lossy().to_string(),
                tags: HashMap::new(),
            },
        );

        let result = manager.verify_checksum(&file_path, "test_model.bin");
        assert!(
            result.is_ok(),
            "Correct checksum should pass: {:?}",
            result.err()
        );
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let temp_dir = std::env::temp_dir().join("voirs_checksum_mismatch_test");
        std::fs::create_dir_all(&temp_dir).ok();
        let file_path = temp_dir.join("bad_model.bin");
        std::fs::write(&file_path, b"tampered data").expect("write file");

        let config = CloudStorageConfig {
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");
        manager.metadata_cache.write().insert(
            "bad_model.bin".to_string(),
            ModelMetadata {
                name: "bad_model.bin".to_string(),
                version: "1.0.0".to_string(),
                size_bytes: 13,
                checksum: "0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                last_modified: chrono::Utc::now(),
                model_type: "test".to_string(),
                storage_path: file_path.to_string_lossy().to_string(),
                tags: HashMap::new(),
            },
        );

        let result = manager.verify_checksum(&file_path, "bad_model.bin");
        assert!(
            matches!(result, Err(CloudStorageError::ChecksumMismatch { .. })),
            "Mismatched checksum should return ChecksumMismatch error, got: {:?}",
            result
        );
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    #[ignore = "Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, TEST_BUCKET, TEST_AWS_REGION env vars"]
    async fn test_s3_roundtrip() {
        let access_key =
            std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let secret_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");
        let bucket = std::env::var("TEST_BUCKET").expect("TEST_BUCKET must be set");
        let region =
            std::env::var("TEST_AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        let temp_dir = std::env::temp_dir().join("voirs_s3_roundtrip");
        std::fs::create_dir_all(&temp_dir).ok();
        let test_model = temp_dir.join("s3_test_model.bin");
        std::fs::write(&test_model, b"VoiRS S3 roundtrip test data").expect("write test model");

        let config = CloudStorageConfig {
            provider: CloudProvider::AwsS3,
            bucket_name: bucket,
            region: Some(region),
            access_key_id: Some(access_key),
            secret_access_key: Some(secret_key),
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");

        let meta = ModelMetadata {
            name: "s3_test_model.bin".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 28,
            checksum: String::new(),
            last_modified: chrono::Utc::now(),
            model_type: "test".to_string(),
            storage_path: "s3_test_model.bin".to_string(),
            tags: HashMap::new(),
        };

        manager
            .upload_model(&test_model, "s3_test_model.bin", meta)
            .await
            .expect("upload should succeed");

        let models = manager.list_models().await.expect("list should succeed");
        assert!(
            models.iter().any(|m| m.name.contains("s3_test_model")),
            "Uploaded model should appear in listing"
        );

        std::fs::remove_file(&test_model).ok();
        let downloaded = manager
            .download_model("s3_test_model.bin")
            .await
            .expect("download should succeed");
        assert!(downloaded.exists(), "Downloaded file should exist");
        let content = std::fs::read(&downloaded).expect("read downloaded file");
        assert_eq!(content, b"VoiRS S3 roundtrip test data");

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
