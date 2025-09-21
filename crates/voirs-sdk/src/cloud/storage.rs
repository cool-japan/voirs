use super::*;
use chrono::{DateTime, Utc};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::sync::Mutex;

#[cfg(feature = "cloud")]
use zstd;

/// Cloud storage implementation for VoiRS models
pub struct VoirsCloudStorage {
    config: CloudConfig,
    local_cache: Arc<Mutex<LocalCache>>,
    sync_manager: Arc<SyncManager>,
    backup_manager: Arc<BackupManager>,
    version_manager: Arc<VersionManager>,
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

#[derive(Debug, Clone)]
enum SyncOperation {
    Upload(String),
    Download(String),
    Delete(String),
    Verify(String),
}

#[derive(Debug, Clone)]
struct SyncStatus {
    in_progress: bool,
    last_sync: Option<DateTime<Utc>>,
    pending_operations: usize,
    errors: Vec<SyncError>,
    models_synced: u32,
    models_updated: u32,
    models_deleted: u32,
}

#[derive(Debug, Clone)]
struct SyncError {
    operation: SyncOperation,
    error: String,
    timestamp: DateTime<Utc>,
    retry_count: u32,
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

        let storage = Self {
            config,
            local_cache,
            sync_manager,
            backup_manager,
            version_manager,
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
            if path.extension().map_or(false, |ext| ext == "model") {
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
        format!("{:x}", hasher.finalize())
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
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(data).map_err(|e| {
                    VoirsError::config_error(format!("Failed to compress data with Gzip: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    VoirsError::config_error(format!("Failed to finish Gzip compression: {}", e))
                })
            }
            #[cfg(feature = "cloud")]
            CompressionType::Zstd => {
                zstd::encode_all(data, 3) // Compression level 3 (balanced)
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
                let mut decoder = GzDecoder::new(compressed_data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    VoirsError::config_error(format!("Failed to decompress Gzip data: {}", e))
                })?;
                Ok(decompressed)
            }
            #[cfg(feature = "cloud")]
            CompressionType::Zstd => zstd::decode_all(compressed_data).map_err(|e| {
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

        models_by_access.sort_by(|a, b| a.1.cmp(&b.1));

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

        tokio::spawn(async move {
            let _ = Self::run_sync_process(sync_manager, local_cache).await;
        });

        Ok(())
    }

    async fn run_sync_process(
        sync_manager: Arc<SyncManager>,
        local_cache: Arc<Mutex<LocalCache>>,
    ) -> Result<()> {
        let operations = {
            let mut queue = sync_manager.sync_queue.lock().await;
            let ops = queue.clone();
            queue.clear();
            ops
        };

        let mut errors = Vec::new();

        for operation in operations {
            match Self::execute_sync_operation(&operation, &local_cache).await {
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

    async fn execute_sync_operation(
        operation: &SyncOperation,
        local_cache: &Arc<Mutex<LocalCache>>,
    ) -> Result<()> {
        match operation {
            SyncOperation::Upload(model_id) => {
                tracing::info!("Uploading model: {}", model_id);
                Self::upload_model_to_cloud(model_id, local_cache).await
            }
            SyncOperation::Download(model_id) => {
                tracing::info!("Downloading model: {}", model_id);
                Self::download_model_from_cloud(model_id, local_cache).await
            }
            SyncOperation::Delete(model_id) => {
                tracing::info!("Deleting model: {}", model_id);
                Self::delete_model_from_cloud(model_id).await
            }
            SyncOperation::Verify(model_id) => {
                tracing::info!("Verifying model: {}", model_id);
                Self::verify_model_checksum(model_id, local_cache).await
            }
        }
    }

    async fn upload_model_to_cloud(
        model_id: &str,
        local_cache: &Arc<Mutex<LocalCache>>,
    ) -> Result<()> {
        let cache = local_cache.lock().await;
        if let Some(model) = cache.models.get(model_id) {
            let data = fs::read(&model.local_path).await.map_err(|e| {
                VoirsError::config_error(format!("Failed to read model file: {}", e))
            })?;

            // Calculate checksum for verification
            let checksum = Self::calculate_checksum(&data);

            // Compress data if enabled
            let compressed_data = Self::compress_data(&data)?;

            // Simulate cloud upload with local storage for now
            let cloud_path = cache
                .cache_dir
                .join("cloud_mirror")
                .join(format!("{}.cloud", model_id));
            fs::create_dir_all(cloud_path.parent().unwrap())
                .await
                .map_err(|e| {
                    VoirsError::config_error(format!(
                        "Failed to create cloud mirror directory: {}",
                        e
                    ))
                })?;

            // Write compressed data and metadata
            fs::write(&cloud_path, &compressed_data)
                .await
                .map_err(|e| {
                    VoirsError::config_error(format!("Failed to write cloud data: {}", e))
                })?;

            let metadata_path = cloud_path.with_extension("metadata");
            let metadata_json = serde_json::to_string_pretty(&model.metadata).map_err(|e| {
                VoirsError::config_error(format!("Failed to serialize metadata: {}", e))
            })?;
            fs::write(&metadata_path, metadata_json)
                .await
                .map_err(|e| {
                    VoirsError::config_error(format!("Failed to write metadata: {}", e))
                })?;

            tracing::info!(
                "Successfully uploaded model {} to cloud (checksum: {})",
                model_id,
                checksum
            );
            Ok(())
        } else {
            Err(VoirsError::config_error(format!(
                "Model {} not found in local cache",
                model_id
            )))
        }
    }

    async fn download_model_from_cloud(
        model_id: &str,
        local_cache: &Arc<Mutex<LocalCache>>,
    ) -> Result<()> {
        let cache_dir = {
            let cache = local_cache.lock().await;
            cache.cache_dir.clone()
        };

        // Simulate cloud download from local mirror
        let cloud_path = cache_dir
            .join("cloud_mirror")
            .join(format!("{}.cloud", model_id));
        let metadata_path = cloud_path.with_extension("metadata");

        if !cloud_path.exists() {
            return Err(VoirsError::config_error(format!(
                "Model {} not found in cloud storage",
                model_id
            )));
        }

        // Read compressed data and metadata
        let compressed_data = fs::read(&cloud_path)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to read cloud data: {}", e)))?;

        let metadata_json = fs::read_to_string(&metadata_path)
            .await
            .map_err(|e| VoirsError::config_error(format!("Failed to read metadata: {}", e)))?;

        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
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

    async fn delete_model_from_cloud(model_id: &str) -> Result<()> {
        // For the local mirror simulation, we would delete the cloud files
        // In a real implementation, this would call the cloud provider's delete API

        tracing::info!("Marking model {} for deletion from cloud", model_id);

        // Simulate deletion by marking as deleted (in real implementation, call cloud API)
        // For now, we'll just log the operation since it's a placeholder
        tracing::warn!(
            "Cloud deletion is simulated - model {} marked for removal",
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
        match Self::download_model_from_cloud(model_id, &self.local_cache).await {
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
            if path.extension().map_or(false, |ext| ext == "backup") {
                if let Ok(backup) = self
                    .retrieve_backup(path.file_stem().unwrap().to_str().unwrap())
                    .await
                {
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
                        name: format!("Backup"),
                        size_bytes: backup.data.len() as u64,
                        created_at,
                        models_count: backup.models.len() as u32,
                        checksum: Self::calculate_backup_checksum(&backup.data),
                    });
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
        format!("{:x}", hasher.finalize())
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
        let data = b"test data for zstd compression - this should compress well with zstd";
        let compressed =
            VoirsCloudStorage::compress_data_with_type(data, CompressionType::Zstd).unwrap();
        let decompressed =
            VoirsCloudStorage::decompress_data_with_type(&compressed, CompressionType::Zstd)
                .unwrap();
        assert_eq!(data, decompressed.as_slice());

        // Zstd should achieve some compression
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
}
