use super::s3_client::{AwsCredentials, S3Client};
use super::*;
use chrono::{DateTime, Utc};
use oxiarc_deflate::{GzipStreamDecoder, GzipStreamEncoder};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::sync::Mutex;

#[cfg(feature = "cloud")]
use oxiarc_zstd;

/// Which real remote backend (if any) [`VoirsCloudStorage`] was able to
/// resolve from its [`CloudConfig`] at construction time.
///
/// Resolved once (see [`resolve_cloud_backend`]) so every "cloud" operation
/// either genuinely talks to a real S3(-compatible) endpoint over HTTPS, or
/// fails with the exact same clear reason — never silently falling back to
/// writing local files while claiming to be cloud storage.
enum CloudBackend {
    /// A real, SigV4-signed S3(-compatible) client (AWS S3, or any
    /// S3-compatible endpoint configured via `CloudCredentials::endpoint`,
    /// e.g. MinIO/Cloudflare R2/on-prem gateways).
    S3(S3Client),
    /// No real backend could be resolved (missing credentials/bucket, or an
    /// unimplemented provider such as Azure/GCP). Every operation against
    /// this variant fails with `reason`.
    Unsupported(String),
}

/// Resolve `config` into a real network backend, or a clear reason why none
/// is available. Never returns a "fake" backend.
fn resolve_cloud_backend(config: &CloudConfig) -> CloudBackend {
    let bucket = config.storage_config.bucket_name.trim();
    if bucket.is_empty() {
        return CloudBackend::Unsupported(
            "cloud storage is not configured: storage_config.bucket_name is empty".to_string(),
        );
    }

    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(client) => client,
        Err(e) => return CloudBackend::Unsupported(format!("failed to build HTTP client: {e}")),
    };

    let credentials_present = !config.credentials.access_key.trim().is_empty()
        && !config.credentials.secret_key.trim().is_empty();
    let custom_endpoint = config
        .credentials
        .endpoint
        .as_deref()
        .map(str::trim)
        .filter(|e| !e.is_empty());

    match &config.provider {
        CloudProvider::AWS => {
            if !credentials_present {
                return CloudBackend::Unsupported(
                    "AWS credentials are not configured (CloudCredentials.access_key/secret_key \
                     are empty)"
                        .to_string(),
                );
            }
            let credentials = AwsCredentials {
                region: config.region.clone(),
                access_key_id: config.credentials.access_key.clone(),
                secret_access_key: config.credentials.secret_key.clone(),
                session_token: config.credentials.token.clone(),
            };
            match custom_endpoint {
                // An explicit endpoint override (e.g. a regional/VPC S3
                // endpoint, or an S3-compatible service) uses path-style
                // addressing rather than assuming `*.amazonaws.com`.
                Some(endpoint) => {
                    CloudBackend::S3(S3Client::new_path_style(http, endpoint, bucket, credentials))
                }
                None => CloudBackend::S3(S3Client::new_aws(http, bucket, credentials)),
            }
        }
        CloudProvider::Custom(name) => {
            let Some(endpoint) = custom_endpoint else {
                return CloudBackend::Unsupported(format!(
                    "custom cloud provider '{name}' requires CloudCredentials.endpoint to be \
                     set to an S3-compatible endpoint URL"
                ));
            };
            if !credentials_present {
                return CloudBackend::Unsupported(format!(
                    "custom cloud provider '{name}' requires CloudCredentials.access_key and \
                     secret_key"
                ));
            }
            let credentials = AwsCredentials {
                region: config.region.clone(),
                access_key_id: config.credentials.access_key.clone(),
                secret_access_key: config.credentials.secret_key.clone(),
                session_token: config.credentials.token.clone(),
            };
            CloudBackend::S3(S3Client::new_path_style(
                http,
                endpoint,
                bucket,
                credentials,
            ))
        }
        CloudProvider::Azure => CloudBackend::Unsupported(
            "Azure Blob Storage is not implemented in this build; use CloudProvider::AWS or \
             CloudProvider::Custom with an S3-compatible endpoint instead"
                .to_string(),
        ),
        CloudProvider::GCP => CloudBackend::Unsupported(
            "Google Cloud Storage is not implemented in this build; use CloudProvider::AWS or \
             CloudProvider::Custom with an S3-compatible endpoint instead"
                .to_string(),
        ),
    }
}

/// Cloud storage implementation for VoiRS models.
///
/// The local disk cache (`local_cache`) is a genuine, always-real on-disk
/// cache — that part was never fabricated. What used to be fabricated was
/// the "cloud" half: `upload_model`/`download_model`/`delete_model` queue a
/// [`SyncOperation`] that a background task executes via internal
/// `upload_model_to_cloud`/`download_model_from_cloud`/`delete_model_from_cloud`
/// helpers, which now perform real SigV4-signed S3(-compatible) HTTP
/// requests through `cloud_backend` instead of mirroring to a local
/// directory.
pub struct VoirsCloudStorage {
    config: CloudConfig,
    local_cache: Arc<Mutex<LocalCache>>,
    sync_manager: Arc<SyncManager>,
    backup_manager: Arc<BackupManager>,
    version_manager: Arc<VersionManager>,
    /// Real remote backend resolved once from `config` (see
    /// [`resolve_cloud_backend`]); shared into the background sync task.
    cloud_backend: Arc<CloudBackend>,
}

struct LocalCache {
    models: BTreeMap<String, CachedModel>,
    cache_dir: PathBuf,
    max_size_bytes: u64,
    current_size_bytes: AtomicU64,
}

struct CachedModel {
    metadata: ModelMetadata,
    local_path: PathBuf,
    last_accessed: DateTime<Utc>,
    is_dirty: bool,
}

struct SyncManager {
    sync_queue: Arc<Mutex<Vec<SyncOperation>>>,
    sync_status: Arc<Mutex<SyncStatus>>,
}

struct BackupManager {
    backup_storage: Arc<dyn BackupStorage>,
    backup_schedule: BackupSchedule,
}

struct VersionManager {
    versions: Arc<Mutex<BTreeMap<String, Vec<ModelVersion>>>>,
    current_versions: Arc<Mutex<BTreeMap<String, String>>>,
}

/// Type of synchronization operation
#[derive(Debug, Clone)]
pub enum SyncOperation {
    /// Upload operation
    Upload(String),
    /// Download operation
    Download(String),
    /// Delete operation
    Delete(String),
    /// Verify operation
    Verify(String),
}

/// Status of cloud synchronization operations
#[derive(Debug, Clone)]
pub struct SyncStatus {
    /// Whether synchronization is currently in progress
    pub in_progress: bool,
    /// Timestamp of the last successful sync
    pub last_sync: Option<DateTime<Utc>>,
    /// Number of pending sync operations
    pub pending_operations: usize,
    /// List of errors encountered during sync
    pub errors: Vec<SyncError>,
    /// Number of models synchronized
    pub models_synced: u32,
    /// Number of models updated
    pub models_updated: u32,
    /// Number of models deleted
    pub models_deleted: u32,
}

/// Error that occurred during synchronization
#[derive(Debug, Clone)]
pub struct SyncError {
    /// The operation that failed
    pub operation: SyncOperation,
    /// Error message
    pub error: String,
    /// When the error occurred
    pub timestamp: DateTime<Utc>,
    /// Number of retry attempts
    pub retry_count: u32,
}

#[derive(Debug, Clone)]
struct BackupSchedule {
    enabled: bool,
    interval_hours: u32,
    retention_days: u32,
    incremental: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelVersion {
    id: String,
    version: String,
    checksum: String,
    size_bytes: u64,
    created_at: DateTime<Utc>,
    changes: Vec<String>,
    parent_version: Option<String>,
}

#[async_trait::async_trait]
trait BackupStorage: Send + Sync {
    async fn store_backup(&self, backup: &BackupData) -> Result<String>;
    async fn retrieve_backup(&self, backup_id: &str) -> Result<BackupData>;
    async fn list_backups(&self) -> Result<Vec<BackupInfo>>;
    async fn delete_backup(&self, backup_id: &str) -> Result<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackupData {
    id: String,
    models: Vec<ModelMetadata>,
    data: Vec<u8>,
    compression: CompressionType,
    encryption: Option<EncryptionInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum CompressionType {
    None,
    Gzip,
    Zstd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncryptionInfo {
    algorithm: String,
    key_id: String,
    nonce: Vec<u8>,
}

impl VoirsCloudStorage {
    pub async fn new(config: CloudConfig, cache_dir: PathBuf) -> Result<Self> {
        // Ensure cache directory exists
        fs::create_dir_all(&cache_dir).await.map_err(|e| {
            VoirsError::config_error(format!("Failed to create cache directory: {}", e))
        })?;

        let local_cache = Arc::new(Mutex::new(LocalCache {
            models: BTreeMap::new(),
            cache_dir: cache_dir.clone(),
            max_size_bytes: 10_000_000_000, // 10GB default
            current_size_bytes: AtomicU64::new(0),
        }));

        let sync_manager = Arc::new(SyncManager {
            sync_queue: Arc::new(Mutex::new(Vec::new())),
            sync_status: Arc::new(Mutex::new(SyncStatus {
                in_progress: false,
                last_sync: None,
                pending_operations: 0,
                errors: Vec::new(),
                models_synced: 0,
                models_updated: 0,
                models_deleted: 0,
            })),
        });

        let backup_manager = Arc::new(BackupManager {
            backup_storage: Arc::new(LocalBackupStorage::new(cache_dir.join("backups"))),
            backup_schedule: BackupSchedule {
                enabled: config.storage_config.backup_retention_days > 0,
                interval_hours: 24,
                retention_days: config.storage_config.backup_retention_days,
                incremental: true,
            },
        });

        let version_manager = Arc::new(VersionManager {
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            current_versions: Arc::new(Mutex::new(BTreeMap::new())),
        });

        let cloud_backend = Arc::new(resolve_cloud_backend(&config));

        let storage = Self {
            config,
            local_cache,
            sync_manager,
            backup_manager,
            version_manager,
            cloud_backend,
        };

        // Initialize cache from existing files
        storage.initialize_cache().await?;

        Ok(storage)
    }

    async fn initialize_cache(&self) -> Result<()> {
        let cache_dir = {
            let cache = self.local_cache.lock().await;
            cache.cache_dir.clone()
        };

        if !cache_dir.exists() {
            return Ok(());
        }

        let mut total_size = 0u64;
        let mut models = BTreeMap::new();

        let mut entries = fs::read_dir(&cache_dir).await.map_err(|e| {
            VoirsError::config_error(format!("Failed to read cache directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            VoirsError::config_error(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "model") {
                if let Ok(metadata) = self.load_model_metadata(&path).await {
                    let file_size = entry
                        .metadata()
                        .await
                        .map_err(|e| {
                            VoirsError::config_error(format!("Failed to get file metadata: {}", e))
                        })?
                        .len();

                    total_size += file_size;

                    models.insert(
                        metadata.id.clone(),
                        CachedModel {
                            metadata,
                            local_path: path,
                            last_accessed: Utc::now(),
                            is_dirty: false,
                        },
                    );
                }
            }
        }

        let mut cache = self.local_cache.lock().await;
        cache.models = models;
        cache
            .current_size_bytes
            .store(total_size, Ordering::Relaxed);

        Ok(())
    }

    async fn load_model_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        let metadata_path = path.with_extension("metadata");
        let metadata_content = fs::read_to_string(&metadata_path)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to read metadata: {}", e)))?;

        serde_json::from_str(&metadata_content)
            .map_err(|e| VoirsError::config_error(format!("Failed to parse metadata: {}", e)))
    }

    async fn save_model_metadata(&self, model: &CachedModel) -> Result<()> {
        let metadata_path = model.local_path.with_extension("metadata");
        let metadata_content = serde_json::to_string_pretty(&model.metadata).map_err(|e| {
            VoirsError::config_error(format!("Failed to serialize metadata: {}", e))
        })?;

        fs::write(&metadata_path, metadata_content)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to write metadata: {}", e)))
    }

    fn calculate_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Get preferred compression type based on configuration and features
    fn get_compression_type(&self) -> CompressionType {
        if !self.config.storage_config.compression {
            return CompressionType::None;
        }

        // Prefer Zstd if available, otherwise fall back to Gzip
        if cfg!(feature = "cloud") {
            CompressionType::Zstd
        } else {
            CompressionType::Gzip
        }
    }

    fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
        Self::compress_data_with_type(data, CompressionType::Gzip)
    }

    fn decompress_data(compressed_data: &[u8]) -> Result<Vec<u8>> {
        Self::decompress_data_with_type(compressed_data, CompressionType::Gzip)
    }

    fn compress_data_with_type(data: &[u8], compression_type: CompressionType) -> Result<Vec<u8>> {
        match compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                let mut encoder = GzipStreamEncoder::new(Vec::new(), 6);
                encoder.write_all(data).map_err(|e| {
                    VoirsError::config_error(format!("Failed to compress data with Gzip: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    VoirsError::config_error(format!("Failed to finish Gzip compression: {}", e))
                })
            }
            #[cfg(feature = "cloud")]
            CompressionType::Zstd => {
                oxiarc_zstd::encode_all(data, 3) // Compression level 3 (balanced)
                    .map_err(|e| {
                        VoirsError::config_error(format!(
                            "Failed to compress data with Zstd: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "cloud"))]
            CompressionType::Zstd => Err(VoirsError::config_error(
                "Zstd compression not available - compile with 'cloud' feature".to_string(),
            )),
        }
    }

    fn decompress_data_with_type(
        compressed_data: &[u8],
        compression_type: CompressionType,
    ) -> Result<Vec<u8>> {
        match compression_type {
            CompressionType::None => Ok(compressed_data.to_vec()),
            CompressionType::Gzip => {
                let mut decoder = GzipStreamDecoder::new(compressed_data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    VoirsError::config_error(format!("Failed to decompress Gzip data: {}", e))
                })?;
                Ok(decompressed)
            }
            #[cfg(feature = "cloud")]
            CompressionType::Zstd => oxiarc_zstd::decode_all(compressed_data).map_err(|e| {
                VoirsError::config_error(format!("Failed to decompress Zstd data: {}", e))
            }),
            #[cfg(not(feature = "cloud"))]
            CompressionType::Zstd => Err(VoirsError::config_error(
                "Zstd decompression not available - compile with 'cloud' feature".to_string(),
            )),
        }
    }

    async fn ensure_cache_space(&self, required_bytes: u64) -> Result<()> {
        let mut cache = self.local_cache.lock().await;
        let current_size = cache.current_size_bytes.load(Ordering::Relaxed);

        if current_size + required_bytes <= cache.max_size_bytes {
            return Ok(());
        }

        // Sort models by last accessed time and remove oldest ones
        let mut models_by_access: Vec<_> = cache
            .models
            .iter()
            .map(|(id, model)| (id.clone(), model.last_accessed))
            .collect();

        models_by_access.sort_by_key(|a| a.1);

        let mut freed_bytes = 0u64;
        let mut to_remove = Vec::new();

        for (model_id, _) in models_by_access {
            if current_size - freed_bytes + required_bytes <= cache.max_size_bytes {
                break;
            }

            if let Some(model) = cache.models.get(&model_id) {
                if let Ok(metadata) = fs::metadata(&model.local_path).await {
                    freed_bytes += metadata.len();
                    to_remove.push(model_id);
                }
            }
        }

        // Remove selected models
        for model_id in to_remove {
            if let Some(model) = cache.models.remove(&model_id) {
                let _ = fs::remove_file(&model.local_path).await;
                let _ = fs::remove_file(model.local_path.with_extension("metadata")).await;
            }
        }

        cache
            .current_size_bytes
            .store(current_size - freed_bytes, Ordering::Relaxed);
        Ok(())
    }

    pub async fn get_sync_status(&self) -> Result<SyncStatus> {
        let status = self.sync_manager.sync_status.lock().await;
        Ok(status.clone())
    }

    pub async fn start_sync(&self) -> Result<()> {
        let mut status = self.sync_manager.sync_status.lock().await;
        if status.in_progress {
            return Ok(());
        }

        status.in_progress = true;
        drop(status);

        // Start background sync task
        let sync_manager = self.sync_manager.clone();
        let local_cache = self.local_cache.clone();
        let cloud_backend = Arc::clone(&self.cloud_backend);

        tokio::spawn(async move {
            let _ = Self::run_sync_process(sync_manager, local_cache, cloud_backend).await;
        });

        Ok(())
    }

    async fn run_sync_process(
        sync_manager: Arc<SyncManager>,
        local_cache: Arc<Mutex<LocalCache>>,
        cloud_backend: Arc<CloudBackend>,
    ) -> Result<()> {
        let operations = {
            let mut queue = sync_manager.sync_queue.lock().await;
            let ops = queue.clone();
            queue.clear();
            ops
        };

        let mut errors = Vec::new();

        for operation in operations {
            match Self::execute_sync_operation(&operation, &local_cache, &cloud_backend).await {
                Ok(_) => {}
                Err(e) => {
                    errors.push(SyncError {
                        operation,
                        error: e.to_string(),
                        timestamp: Utc::now(),
                        retry_count: 0,
                    });
                }
            }
        }

        let mut status = sync_manager.sync_status.lock().await;
        status.in_progress = false;
        status.last_sync = Some(Utc::now());
        status.pending_operations = 0;
        status.errors = errors;

        Ok(())
    }

    /// Resolve the real S3(-compatible) client for `cloud_backend`, or a
    /// clear `Err` explaining why no real backend is available for `op`.
    fn require_s3<'a>(
        cloud_backend: &'a CloudBackend,
        op: &str,
        model_id: &str,
    ) -> Result<&'a S3Client> {
        match cloud_backend {
            CloudBackend::S3(client) => Ok(client),
            CloudBackend::Unsupported(reason) => Err(VoirsError::config_error(format!(
                "cannot {op} model '{model_id}' to/from cloud storage: {reason}"
            ))),
        }
    }

    async fn execute_sync_operation(
        operation: &SyncOperation,
        local_cache: &Arc<Mutex<LocalCache>>,
        cloud_backend: &CloudBackend,
    ) -> Result<()> {
        match operation {
            SyncOperation::Upload(model_id) => {
                tracing::info!("Uploading model: {}", model_id);
                Self::upload_model_to_cloud(model_id, local_cache, cloud_backend).await
            }
            SyncOperation::Download(model_id) => {
                tracing::info!("Downloading model: {}", model_id);
                Self::download_model_from_cloud(model_id, local_cache, cloud_backend).await
            }
            SyncOperation::Delete(model_id) => {
                tracing::info!("Deleting model: {}", model_id);
                Self::delete_model_from_cloud(model_id, cloud_backend).await
            }
            SyncOperation::Verify(model_id) => {
                tracing::info!("Verifying model: {}", model_id);
                Self::verify_model_checksum(model_id, local_cache).await
            }
        }
    }

    /// Really upload the model's bytes and metadata to `cloud_backend` as two
    /// S3 objects (`{model_id}.model`, `{model_id}.metadata`) via a genuine
    /// SigV4-signed HTTP PUT — no local-file mirroring.
    async fn upload_model_to_cloud(
        model_id: &str,
        local_cache: &Arc<Mutex<LocalCache>>,
        cloud_backend: &CloudBackend,
    ) -> Result<()> {
        let client = Self::require_s3(cloud_backend, "upload", model_id)?;

        let (data, metadata) = {
            let cache = local_cache.lock().await;
            let Some(model) = cache.models.get(model_id) else {
                return Err(VoirsError::config_error(format!(
                    "Model {model_id} not found in local cache"
                )));
            };
            let data = fs::read(&model.local_path).await.map_err(|e| {
                VoirsError::config_error(format!("Failed to read model file: {}", e))
            })?;
            (data, model.metadata.clone())
        };

        let checksum = Self::calculate_checksum(&data);
        let compressed_data = Self::compress_data(&data)?;

        client
            .put_object(
                &format!("{model_id}.model"),
                compressed_data,
                "application/octet-stream",
            )
            .await?;

        let metadata_json = serde_json::to_vec(&metadata).map_err(|e| {
            VoirsError::config_error(format!("Failed to serialize metadata: {}", e))
        })?;
        client
            .put_object(
                &format!("{model_id}.metadata"),
                metadata_json,
                "application/json",
            )
            .await?;

        tracing::info!(
            "Successfully uploaded model {} to cloud (checksum: {})",
            model_id,
            checksum
        );
        Ok(())
    }

    /// Really download the model's bytes and metadata from `cloud_backend`
    /// via genuine SigV4-signed HTTP GET requests, then verify the checksum
    /// and populate the local cache — no local-file mirroring.
    async fn download_model_from_cloud(
        model_id: &str,
        local_cache: &Arc<Mutex<LocalCache>>,
        cloud_backend: &CloudBackend,
    ) -> Result<()> {
        let client = Self::require_s3(cloud_backend, "download", model_id)?;

        let compressed_data = client.get_object(&format!("{model_id}.model")).await?;
        let metadata_json = client.get_object(&format!("{model_id}.metadata")).await?;
        let metadata: ModelMetadata = serde_json::from_slice(&metadata_json)
            .map_err(|e| VoirsError::config_error(format!("Failed to parse metadata: {}", e)))?;

        // Decompress data
        let data = Self::decompress_data(&compressed_data)?;

        // Verify checksum
        let calculated_checksum = Self::calculate_checksum(&data);
        if calculated_checksum != metadata.checksum {
            return Err(VoirsError::config_error(format!(
                "Checksum verification failed for model {}: expected {}, got {}",
                model_id, metadata.checksum, calculated_checksum
            )));
        }

        // Save to local cache
        let cache_dir = { local_cache.lock().await.cache_dir.clone() };
        let local_path = cache_dir.join(format!("{}.model", model_id));
        fs::write(&local_path, &data)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to write local model: {}", e)))?;

        // Update cache
        let mut cache = local_cache.lock().await;
        cache.models.insert(
            model_id.to_string(),
            CachedModel {
                metadata,
                local_path,
                last_accessed: Utc::now(),
                is_dirty: false,
            },
        );

        tracing::info!(
            "Successfully downloaded model {} from cloud (checksum: {})",
            model_id,
            calculated_checksum
        );
        Ok(())
    }

    /// Really delete both cloud objects for `model_id` via genuine
    /// SigV4-signed HTTP DELETE requests.
    async fn delete_model_from_cloud(model_id: &str, cloud_backend: &CloudBackend) -> Result<()> {
        let client = Self::require_s3(cloud_backend, "delete", model_id)?;

        client.delete_object(&format!("{model_id}.model")).await?;
        client
            .delete_object(&format!("{model_id}.metadata"))
            .await?;

        tracing::info!(
            "Successfully deleted model {} from cloud storage",
            model_id
        );
        Ok(())
    }

    async fn verify_model_checksum(
        model_id: &str,
        local_cache: &Arc<Mutex<LocalCache>>,
    ) -> Result<()> {
        let cache = local_cache.lock().await;
        if let Some(model) = cache.models.get(model_id) {
            if model.local_path.exists() {
                let data = fs::read(&model.local_path).await.map_err(|e| {
                    VoirsError::config_error(format!("Failed to read model file: {}", e))
                })?;

                let calculated_checksum = Self::calculate_checksum(&data);
                if calculated_checksum == model.metadata.checksum {
                    tracing::info!("Checksum verification passed for model {}", model_id);
                    Ok(())
                } else {
                    Err(VoirsError::config_error(format!(
                        "Checksum verification failed for model {}: expected {}, got {}",
                        model_id, model.metadata.checksum, calculated_checksum
                    )))
                }
            } else {
                Err(VoirsError::config_error(format!(
                    "Model file {} not found locally",
                    model_id
                )))
            }
        } else {
            Err(VoirsError::config_error(format!(
                "Model {} not found in cache",
                model_id
            )))
        }
    }
}

#[async_trait::async_trait]
impl CloudStorage for VoirsCloudStorage {
    async fn upload_model(&self, model_id: &str, data: &[u8]) -> Result<String> {
        self.ensure_cache_space(data.len() as u64).await?;

        let checksum = Self::calculate_checksum(data);
        let compression_type = self.get_compression_type();
        let compressed_data = Self::compress_data_with_type(data, compression_type)?;

        let metadata = ModelMetadata {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: "1.0.0".to_string(),
            size_bytes: data.len() as u64,
            checksum: checksum.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            tags: HashMap::new(),
        };

        let cache_dir = {
            let cache = self.local_cache.lock().await;
            cache.cache_dir.clone()
        };

        let file_path = cache_dir.join(format!("{}.model", model_id));
        fs::write(&file_path, &compressed_data)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to write model file: {}", e)))?;

        let cached_model = CachedModel {
            metadata: metadata.clone(),
            local_path: file_path,
            last_accessed: Utc::now(),
            is_dirty: true,
        };

        self.save_model_metadata(&cached_model).await?;

        let mut cache = self.local_cache.lock().await;
        cache
            .current_size_bytes
            .fetch_add(compressed_data.len() as u64, Ordering::Relaxed);
        cache.models.insert(model_id.to_string(), cached_model);

        // Queue for cloud upload
        let mut queue = self.sync_manager.sync_queue.lock().await;
        queue.push(SyncOperation::Upload(model_id.to_string()));

        Ok(checksum)
    }

    async fn download_model(&self, model_id: &str) -> Result<Vec<u8>> {
        // Check local cache first
        let cache = self.local_cache.lock().await;
        if let Some(model) = cache.models.get(model_id) {
            let compressed_data = fs::read(&model.local_path).await.map_err(|e| {
                VoirsError::config_error(format!("Failed to read cached model: {}", e))
            })?;

            let compression_type = self.get_compression_type();
            let data = Self::decompress_data_with_type(&compressed_data, compression_type)?;

            // Update access time
            drop(cache);
            let mut cache = self.local_cache.lock().await;
            if let Some(model) = cache.models.get_mut(model_id) {
                model.last_accessed = Utc::now();
            }

            return Ok(data);
        }
        drop(cache);

        // Try to download from cloud storage
        tracing::info!(
            "Model {} not in local cache, attempting cloud download",
            model_id
        );

        // Attempt cloud download
        match Self::download_model_from_cloud(model_id, &self.local_cache, &self.cloud_backend)
            .await
        {
            Ok(()) => {
                // Successfully downloaded, now retrieve from cache
                let cache = self.local_cache.lock().await;
                if let Some(model) = cache.models.get(model_id) {
                    let data = fs::read(&model.local_path).await.map_err(|e| {
                        VoirsError::config_error(format!("Failed to read downloaded model: {}", e))
                    })?;

                    let compression_type = self.get_compression_type();
                    Self::decompress_data_with_type(&data, compression_type)
                } else {
                    Err(VoirsError::config_error(format!(
                        "Model {} not found after download",
                        model_id
                    )))
                }
            }
            Err(_) => {
                // Cloud download failed, model not available
                Err(VoirsError::config_error(format!(
                    "Model {} not found in cache or cloud",
                    model_id
                )))
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelMetadata>> {
        let cache = self.local_cache.lock().await;
        Ok(cache
            .models
            .values()
            .map(|model| model.metadata.clone())
            .collect())
    }

    async fn delete_model(&self, model_id: &str) -> Result<()> {
        let mut cache = self.local_cache.lock().await;
        if let Some(model) = cache.models.remove(model_id) {
            let _ = fs::remove_file(&model.local_path).await;
            let _ = fs::remove_file(model.local_path.with_extension("metadata")).await;

            if let Ok(metadata) = fs::metadata(&model.local_path).await {
                cache
                    .current_size_bytes
                    .fetch_sub(metadata.len(), Ordering::Relaxed);
            }
        }

        // Queue for cloud deletion
        let mut queue = self.sync_manager.sync_queue.lock().await;
        queue.push(SyncOperation::Delete(model_id.to_string()));

        Ok(())
    }

    async fn sync_models(&self) -> Result<SyncReport> {
        let start_time = Utc::now();
        self.start_sync().await?;

        // Wait for sync to complete (with timeout)
        let timeout = tokio::time::Duration::from_secs(300); // 5 minutes
        let deadline = tokio::time::Instant::now() + timeout;

        while tokio::time::Instant::now() < deadline {
            let status = self.get_sync_status().await?;
            if !status.in_progress {
                let end_time = Utc::now();
                return Ok(SyncReport {
                    models_synced: status.models_synced,
                    models_updated: status.models_updated,
                    models_deleted: status.models_deleted,
                    sync_duration: end_time - start_time,
                    errors: status.errors.into_iter().map(|e| e.error).collect(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Err(VoirsError::config_error(
            "Sync operation timed out".to_string(),
        ))
    }

    async fn create_backup(&self, backup_id: &str) -> Result<BackupInfo> {
        let models = self.list_models().await?;
        let mut backup_data = Vec::new();

        // Collect all model data
        for model in &models {
            if let Ok(data) = self.download_model(&model.id).await {
                backup_data.extend_from_slice(&data);
            }
        }

        // Use preferred compression type based on configuration
        let compression_type = self.get_compression_type();
        let compressed_backup = Self::compress_data_with_type(&backup_data, compression_type)?;
        let checksum = Self::calculate_checksum(&compressed_backup);

        let backup = BackupData {
            id: backup_id.to_string(),
            models: models.clone(),
            data: compressed_backup.clone(),
            compression: compression_type,
            encryption: None,
        };

        self.backup_manager
            .backup_storage
            .store_backup(&backup)
            .await?;

        Ok(BackupInfo {
            id: backup_id.to_string(),
            name: format!("Backup {}", backup_id),
            size_bytes: compressed_backup.len() as u64,
            created_at: Utc::now(),
            models_count: models.len() as u32,
            checksum,
        })
    }

    async fn restore_backup(&self, backup_id: &str) -> Result<()> {
        let backup = self
            .backup_manager
            .backup_storage
            .retrieve_backup(backup_id)
            .await?;

        let decompressed_data = Self::decompress_data_with_type(&backup.data, backup.compression)?;

        // Implement proper restoration logic
        let mut restored_count = 0;

        for model_metadata in &backup.models {
            // Extract model data from backup
            let model_start = restored_count * (decompressed_data.len() / backup.models.len());
            let model_end = (restored_count + 1) * (decompressed_data.len() / backup.models.len());

            if model_end <= decompressed_data.len() {
                let model_data = &decompressed_data[model_start..model_end];

                // Save model to cache
                let model_path = {
                    let cache = self.local_cache.lock().await;
                    cache.cache_dir.join(format!("{}.model", model_metadata.id))
                };

                fs::write(&model_path, model_data).await.map_err(|e| {
                    VoirsError::config_error(format!(
                        "Failed to restore model {}: {}",
                        model_metadata.id, e
                    ))
                })?;

                // Add to cache
                let mut cache = self.local_cache.lock().await;
                cache.models.insert(
                    model_metadata.id.clone(),
                    CachedModel {
                        metadata: model_metadata.clone(),
                        local_path: model_path,
                        last_accessed: Utc::now(),
                        is_dirty: false,
                    },
                );

                restored_count += 1;
                tracing::debug!("Restored model: {}", model_metadata.id);
            }
        }

        tracing::info!(
            "Successfully restored backup {} with {} models",
            backup_id,
            restored_count
        );
        Ok(())
    }
}

struct LocalBackupStorage {
    backup_dir: PathBuf,
}

impl LocalBackupStorage {
    fn new(backup_dir: PathBuf) -> Self {
        Self { backup_dir }
    }
}

#[async_trait::async_trait]
impl BackupStorage for LocalBackupStorage {
    async fn store_backup(&self, backup: &BackupData) -> Result<String> {
        fs::create_dir_all(&self.backup_dir).await.map_err(|e| {
            VoirsError::config_error(format!("Failed to create backup directory: {}", e))
        })?;

        let backup_path = self.backup_dir.join(format!("{}.backup", backup.id));
        let backup_content = serde_json::to_vec(backup)
            .map_err(|e| VoirsError::config_error(format!("Failed to serialize backup: {}", e)))?;

        fs::write(&backup_path, backup_content)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to write backup: {}", e)))?;

        Ok(backup.id.clone())
    }

    async fn retrieve_backup(&self, backup_id: &str) -> Result<BackupData> {
        let backup_path = self.backup_dir.join(format!("{}.backup", backup_id));
        let backup_content = fs::read(&backup_path)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to read backup: {}", e)))?;

        serde_json::from_slice(&backup_content)
            .map_err(|e| VoirsError::config_error(format!("Failed to deserialize backup: {}", e)))
    }

    async fn list_backups(&self) -> Result<Vec<BackupInfo>> {
        let mut backups = Vec::new();

        if !self.backup_dir.exists() {
            return Ok(backups);
        }

        let mut entries = fs::read_dir(&self.backup_dir).await.map_err(|e| {
            VoirsError::config_error(format!("Failed to read backup directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            VoirsError::config_error(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "backup") {
                // Get file stem as string
                let backup_id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string());

                if let Some(id) = backup_id {
                    if let Ok(backup) = self.retrieve_backup(&id).await {
                        // Get creation time from file metadata if available
                        let created_at = if let Ok(metadata) = fs::metadata(&path).await {
                            if let Ok(created) = metadata.created() {
                                DateTime::<Utc>::from(created)
                            } else {
                                Utc::now()
                            }
                        } else {
                            Utc::now()
                        };

                        backups.push(BackupInfo {
                            id: backup.id,
                            name: "Backup".to_string(),
                            size_bytes: backup.data.len() as u64,
                            created_at,
                            models_count: backup.models.len() as u32,
                            checksum: Self::calculate_backup_checksum(&backup.data),
                        });
                    }
                }
            }
        }

        Ok(backups)
    }

    async fn delete_backup(&self, backup_id: &str) -> Result<()> {
        let backup_path = self.backup_dir.join(format!("{}.backup", backup_id));
        fs::remove_file(&backup_path)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to delete backup: {}", e)))
    }
}

impl LocalBackupStorage {
    fn calculate_backup_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cloud_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudConfig::default();

        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf()).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_model_upload_download() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudConfig::default();
        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let test_data = b"test model data";
        let model_id = "test_model";

        // Upload model
        let checksum = storage.upload_model(model_id, test_data).await.unwrap();
        assert!(!checksum.is_empty());

        // Download model
        let downloaded_data = storage.download_model(model_id).await.unwrap();
        assert_eq!(test_data, downloaded_data.as_slice());
    }

    #[tokio::test]
    async fn test_model_listing() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudConfig::default();
        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Upload multiple models
        storage.upload_model("model1", b"data1").await.unwrap();
        storage.upload_model("model2", b"data2").await.unwrap();

        // List models
        let models = storage.list_models().await.unwrap();
        assert_eq!(models.len(), 2);

        let model_ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        assert!(model_ids.contains(&"model1".to_string()));
        assert!(model_ids.contains(&"model2".to_string()));
    }

    #[tokio::test]
    async fn test_backup_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudConfig::default();
        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Upload a model
        storage
            .upload_model("test_model", b"test data")
            .await
            .unwrap();

        // Create backup
        let backup_info = storage.create_backup("test_backup").await.unwrap();
        assert_eq!(backup_info.id, "test_backup");
        assert_eq!(backup_info.models_count, 1);
    }

    #[test]
    fn test_checksum_calculation() {
        let data = b"test data";
        let checksum = VoirsCloudStorage::calculate_checksum(data);
        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // SHA256 produces 64-char hex string
    }

    #[test]
    fn test_compression_decompression() {
        let data = b"test data for compression";
        let compressed = VoirsCloudStorage::compress_data(data).unwrap();
        let decompressed = VoirsCloudStorage::decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_gzip_compression_decompression() {
        let data = b"test data for gzip compression";
        let compressed =
            VoirsCloudStorage::compress_data_with_type(data, CompressionType::Gzip).unwrap();
        let decompressed =
            VoirsCloudStorage::decompress_data_with_type(&compressed, CompressionType::Gzip)
                .unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[cfg(feature = "cloud")]
    #[test]
    fn test_zstd_compression_decompression() {
        // Create larger, repetitive data that compresses well
        let data_str =
            "test data for zstd compression - this should compress well with zstd. ".repeat(100);
        let data = data_str.as_bytes();
        let compressed =
            VoirsCloudStorage::compress_data_with_type(data, CompressionType::Zstd).unwrap();
        let decompressed =
            VoirsCloudStorage::decompress_data_with_type(&compressed, CompressionType::Zstd)
                .unwrap();
        assert_eq!(data, decompressed.as_slice());

        // Zstd should achieve significant compression on repetitive data
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_no_compression() {
        let data = b"test data without compression";
        let compressed =
            VoirsCloudStorage::compress_data_with_type(data, CompressionType::None).unwrap();
        let decompressed =
            VoirsCloudStorage::decompress_data_with_type(&compressed, CompressionType::None)
                .unwrap();
        assert_eq!(data, decompressed.as_slice());
        assert_eq!(data.len(), compressed.len());
    }

    #[tokio::test]
    async fn test_compression_type_selection() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = CloudConfig::default();

        // Test with compression disabled
        config.storage_config.compression = false;
        let storage = VoirsCloudStorage::new(config.clone(), temp_dir.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(storage.get_compression_type(), CompressionType::None);

        // Test with compression enabled
        config.storage_config.compression = true;
        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        #[cfg(feature = "cloud")]
        assert_eq!(storage.get_compression_type(), CompressionType::Zstd);

        #[cfg(not(feature = "cloud"))]
        assert_eq!(storage.get_compression_type(), CompressionType::Gzip);
    }

    // ---- Real cloud-backend resolution & S3(-compatible) round trips -----
    //
    // Direct regression coverage for the fabrication bug: `VoirsCloudStorage`
    // used to silently mirror "cloud" uploads/downloads/deletes to a local
    // `cloud_mirror` directory regardless of `CloudConfig`, and
    // `delete_model_from_cloud` did not delete anything at all. These tests
    // prove real network requests are attempted and real bytes move over an
    // actual TCP socket, using a hand-rolled loopback HTTP server (not a
    // canned-response mock).

    #[test]
    fn test_resolve_cloud_backend_without_credentials_is_honestly_unsupported() {
        let config = CloudConfig::default(); // empty access_key/secret_key
        match resolve_cloud_backend(&config) {
            CloudBackend::Unsupported(reason) => {
                assert!(reason.to_lowercase().contains("credentials"));
            }
            CloudBackend::S3(_) => panic!("must not resolve to a real backend without credentials"),
        }
    }

    #[test]
    fn test_resolve_cloud_backend_azure_and_gcp_are_honestly_unsupported() {
        for provider in [CloudProvider::Azure, CloudProvider::GCP] {
            let mut config = CloudConfig::default();
            config.provider = provider;
            config.credentials.access_key = "key".to_string();
            config.credentials.secret_key = "secret".to_string();
            match resolve_cloud_backend(&config) {
                CloudBackend::Unsupported(_) => {}
                CloudBackend::S3(_) => panic!("Azure/GCP must not fabricate a working S3 backend"),
            }
        }
    }

    #[test]
    fn test_resolve_cloud_backend_empty_bucket_is_unsupported() {
        let mut config = CloudConfig::default();
        config.storage_config.bucket_name = String::new();
        config.credentials.access_key = "key".to_string();
        config.credentials.secret_key = "secret".to_string();
        assert!(matches!(
            resolve_cloud_backend(&config),
            CloudBackend::Unsupported(_)
        ));
    }

    #[test]
    fn test_resolve_cloud_backend_custom_endpoint_with_credentials_resolves_to_real_s3() {
        let mut config = CloudConfig::default();
        config.provider = CloudProvider::Custom("minio".to_string());
        config.credentials.access_key = "key".to_string();
        config.credentials.secret_key = "secret".to_string();
        config.credentials.endpoint = Some("http://127.0.0.1:9000".to_string());
        assert!(matches!(
            resolve_cloud_backend(&config),
            CloudBackend::S3(_)
        ));
    }

    #[tokio::test]
    async fn test_upload_download_delete_model_to_cloud_round_trip_real_bytes_over_loopback_http()
    {
        let server = spawn_mock_object_server().await;
        let temp_dir = TempDir::new().unwrap();

        let mut config = CloudConfig::default();
        config.provider = CloudProvider::Custom("loopback-test".to_string());
        config.storage_config.bucket_name = "test-bucket".to_string();
        config.credentials.access_key = "AKIATESTACCESSKEY".to_string();
        config.credentials.secret_key = "test/secret/access/key".to_string();
        config.credentials.endpoint = Some(format!("http://{}", server.addr));

        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();
        assert!(
            matches!(*storage.cloud_backend, CloudBackend::S3(_)),
            "loopback config with real credentials must resolve to a real S3 backend"
        );

        let model_id = "cloud_round_trip_model";
        let data = b"real model weight bytes, not a fabricated placeholder".to_vec();

        // Populate the local cache the same way the public API does.
        storage.upload_model(model_id, &data).await.unwrap();

        // Directly drive the real cloud-upload helper (bypassing the
        // background sync task purely for test determinism - this is the
        // exact function `execute_sync_operation` calls in production).
        VoirsCloudStorage::upload_model_to_cloud(
            model_id,
            &storage.local_cache,
            &storage.cloud_backend,
        )
        .await
        .unwrap();

        // The mock server really received both objects over a real socket.
        {
            let store = server.store.lock().unwrap();
            assert!(store.contains_key(&format!("{model_id}.model")));
            assert!(store.contains_key(&format!("{model_id}.metadata")));
        }

        // Evict from the local cache to force `download_model` down the real
        // cloud-download path rather than serving from local disk.
        {
            let mut cache = storage.local_cache.lock().await;
            cache.models.remove(model_id);
        }
        let downloaded = storage.download_model(model_id).await.unwrap();
        assert_eq!(
            downloaded, data,
            "downloaded bytes must match the originally uploaded bytes exactly"
        );

        VoirsCloudStorage::delete_model_from_cloud(model_id, &storage.cloud_backend)
            .await
            .unwrap();
        {
            let store = server.store.lock().unwrap();
            assert!(
                !store.contains_key(&format!("{model_id}.model")),
                "delete must really remove the object, unlike the old always-succeeds no-op"
            );
            assert!(!store.contains_key(&format!("{model_id}.metadata")));
        }
    }

    #[tokio::test]
    async fn test_cloud_upload_without_credentials_fails_closed_not_fabricated_success() {
        // Direct regression test: the old implementation always returned
        // `Ok(())` from the cloud-sync helpers regardless of configuration.
        let temp_dir = TempDir::new().unwrap();
        let config = CloudConfig::default(); // no credentials configured
        let storage = VoirsCloudStorage::new(config, temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let model_id = "unconfigured_model";
        storage.upload_model(model_id, b"data").await.unwrap();

        let result = VoirsCloudStorage::upload_model_to_cloud(
            model_id,
            &storage.local_cache,
            &storage.cloud_backend,
        )
        .await;
        assert!(
            result.is_err(),
            "must not report success when no real cloud backend is configured"
        );

        let result =
            VoirsCloudStorage::delete_model_from_cloud(model_id, &storage.cloud_backend).await;
        assert!(
            result.is_err(),
            "delete must also fail closed rather than silently no-op-succeeding"
        );
    }

    // ---- Minimal real loopback HTTP server for the tests above -----------

    struct MockObjectServer {
        addr: std::net::SocketAddr,
        store: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>>,
    }

    async fn spawn_mock_object_server() -> MockObjectServer {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};

        crate::ensure_crypto_provider();

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock object server listener");
        let addr = listener.local_addr().expect("local addr");
        let store: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>> =
            std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let store_for_task = store.clone();

        async fn read_request(
            socket: &mut TcpStream,
        ) -> (String, String, std::collections::HashMap<String, String>, Vec<u8>) {
            let mut buf: Vec<u8> = Vec::new();
            let mut tmp = [0u8; 8192];
            let header_len = loop {
                let n = socket.read(&mut tmp).await.expect("socket read failed");
                assert!(n > 0, "connection closed before headers were complete");
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let header_text = String::from_utf8_lossy(&buf[..header_len]).to_string();
            let mut lines = header_text.split("\r\n");
            let request_line = lines.next().unwrap_or_default();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();

            let mut headers = std::collections::HashMap::new();
            for line in lines {
                if let Some((k, v)) = line.split_once(':') {
                    headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                }
            }
            let content_length: usize = headers
                .get("content-length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut body = buf[header_len..].to_vec();
            while body.len() < content_length {
                let n = socket.read(&mut tmp).await.expect("socket read failed");
                assert!(n > 0, "connection closed before body was complete");
                body.extend_from_slice(&tmp[..n]);
            }
            body.truncate(content_length);
            (method, path, headers, body)
        }

        async fn write_raw(socket: &mut TcpStream, head: &str, body: &[u8]) {
            socket.write_all(head.as_bytes()).await.expect("write head");
            socket.write_all(body).await.expect("write body");
            socket.flush().await.expect("flush");
        }

        tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let store = store_for_task.clone();
                tokio::spawn(async move {
                    let (method, path, headers, body) = read_request(&mut socket).await;
                    assert!(
                        headers
                            .get("authorization")
                            .is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256 Credential=")),
                        "expected a real SigV4 Authorization header, got: {headers:?}"
                    );
                    // Path-style: "/{bucket}/{key}".
                    let key = path
                        .splitn(3, '/')
                        .nth(2)
                        .unwrap_or_default()
                        .to_string();
                    match method.as_str() {
                        "PUT" => {
                            store.lock().expect("store lock").insert(key, body);
                            write_raw(
                                &mut socket,
                                "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                                b"",
                            )
                            .await;
                        }
                        "GET" => {
                            let found = store.lock().expect("store lock").get(&key).cloned();
                            match found {
                                Some(data) => {
                                    let head = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                        data.len()
                                    );
                                    write_raw(&mut socket, &head, &data).await;
                                }
                                None => {
                                    write_raw(
                                        &mut socket,
                                        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                                        b"",
                                    )
                                    .await;
                                }
                            }
                        }
                        "DELETE" => {
                            store.lock().expect("store lock").remove(&key);
                            write_raw(
                                &mut socket,
                                "HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                                b"",
                            )
                            .await;
                        }
                        _ => {
                            write_raw(
                                &mut socket,
                                "HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                                b"",
                            )
                            .await;
                        }
                    }
                });
            }
        });

        MockObjectServer { addr, store }
    }
}
