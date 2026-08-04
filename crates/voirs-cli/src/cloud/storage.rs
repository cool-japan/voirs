// Cloud storage integration for VoiRS model and data synchronization
use crate::cloud::azure_backend::{AzureBackend, AzureConfig};
use crate::cloud::error::CloudStorageError;
use crate::cloud::gcp_backend::{GcpBackend, GcpConfig};
use crate::cloud::s3_backend::{S3Backend, S3Config};
use aes_gcm::{
    aead::{Aead, Generate, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::Result;
use hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Real S3 single-`PUT` uploads are limited to 5 GiB by the S3 API itself;
/// beyond that, S3 requires the multipart upload protocol, which this
/// client does not implement. Enforced up front so an oversized upload
/// fails with a clear, typed error instead of an opaque HTTP failure deep
/// inside the transport layer.
const MAX_SINGLE_PUT_BYTES: u64 = 5 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageConfig {
    pub provider: StorageProvider,
    pub bucket_name: String,
    pub region: String,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub endpoint: Option<String>,
    pub encryption_enabled: bool,
    pub compression_enabled: bool,
    pub sync_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageProvider {
    AWS,
    Azure,
    GoogleCloud,
    MinIO,
    S3Compatible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncableItem {
    pub local_path: PathBuf,
    pub remote_path: String,
    pub last_modified: u64,
    pub checksum: String,
    pub size_bytes: u64,
    pub sync_priority: SyncPriority,
    pub sync_direction: SyncDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncDirection {
    Upload,
    Download,
    Bidirectional,
}

/// Options controlling one [`CloudStorageManager::sync_with_options`]
/// call. Every field here corresponds to a real, user-facing `voirs
/// cloud sync` CLI flag (`--force`, `--directory`, `--dry-run`) and is
/// actually consulted -- there is no flag here that is accepted, printed,
/// and then dropped.
#[derive(Debug, Clone, Default)]
pub struct SyncOptions<'a> {
    /// When `true`, transfer every in-scope item regardless of whether
    /// timestamp-based staleness checks would normally skip it.
    pub force: bool,
    /// When `Some(dir)`, only manifest items whose `local_path` resolves
    /// under `dir` participate in this sync; all other items are left
    /// completely untouched.
    pub directory: Option<&'a Path>,
    /// When `true`, perform every "would this item transfer" decision but
    /// skip the actual upload/download I/O, the manifest write, and the
    /// `last_sync_timestamp` update -- so a dry run is guaranteed to have
    /// zero observable side effects.
    pub dry_run: bool,
}

/// `true` if `path` is `directory` itself or lexically nested under it.
/// Comparison is purely lexical (component-wise prefix match on
/// `path.components()`), not filesystem-canonicalizing: `SyncableItem`
/// paths recorded in the manifest may point at files that no longer exist
/// (e.g. already deleted locally, pending download), so canonicalizing
/// via `fs::canonicalize` would spuriously fail for exactly the paths a
/// download-direction sync needs to match.
fn path_is_within(path: &Path, directory: &Path) -> bool {
    let mut path_components = path.components();
    for dir_component in directory.components() {
        match path_components.next() {
            Some(path_component) if path_component == dir_component => {}
            _ => return false,
        }
    }
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    pub version: u32,
    pub last_sync_timestamp: u64,
    pub items: Vec<SyncableItem>,
    pub total_size_bytes: u64,
    pub checksum: String,
}

pub struct CloudStorageManager {
    config: CloudStorageConfig,
    local_cache_dir: PathBuf,
    sync_manifest: SyncManifest,
    pending_uploads: Vec<SyncableItem>,
    pending_downloads: Vec<SyncableItem>,
    /// Shared HTTP client for every real cloud-provider backend
    /// (`S3Backend`/`AzureBackend`/`GcpBackend`). `reqwest::Client` is
    /// cheap to clone (it wraps an `Arc`), so one instance is built here
    /// and handed to each backend as it is constructed.
    http_client: reqwest::Client,
}

impl CloudStorageManager {
    pub fn new(config: CloudStorageConfig, cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;

        let manifest_path = cache_dir.join("sync_manifest.json");
        let sync_manifest = if manifest_path.exists() {
            let content = std::fs::read_to_string(&manifest_path)?;
            serde_json::from_str(&content)?
        } else {
            SyncManifest::new()
        };

        // Install the pure-Rust rustls CryptoProvider before any TLS
        // handshake (reqwest is built with `rustls-no-provider`).
        // Once-guarded; safe to call repeatedly.
        voirs_acoustic::hub::ensure_crypto_provider();
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        Ok(Self {
            config,
            local_cache_dir: cache_dir,
            sync_manifest,
            pending_uploads: Vec::new(),
            pending_downloads: Vec::new(),
            http_client,
        })
    }

    /// Add a file or directory to the synchronization list
    pub async fn add_to_sync(
        &mut self,
        local_path: PathBuf,
        remote_path: String,
        direction: SyncDirection,
    ) -> Result<()> {
        let metadata = fs::metadata(&local_path).await?;
        let last_modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let checksum = self.calculate_file_checksum(&local_path).await?;

        let item = SyncableItem {
            local_path,
            remote_path,
            last_modified,
            checksum,
            size_bytes: metadata.len(),
            sync_priority: SyncPriority::Normal,
            sync_direction: direction,
        };

        self.sync_manifest.items.push(item);
        self.save_manifest().await?;

        Ok(())
    }

    /// Perform a full synchronization based on the current manifest, with
    /// no filtering, no forcing, and real (non-dry-run) network I/O.
    /// Equivalent to `sync_with_options(&SyncOptions::default())`.
    pub async fn sync(&mut self) -> Result<SyncResult> {
        self.sync_with_options(&SyncOptions::default()).await
    }

    /// Perform synchronization based on the current manifest, honoring
    /// `options`:
    /// - `options.directory`: only items whose `local_path` resolves
    ///   under this directory are considered; everything else is left
    ///   untouched (not even counted as skipped -- it was never in scope
    ///   for this invocation).
    /// - `options.force`: bypass the "is this file already up to date"
    ///   check (`should_upload`/`should_download`/`determine_sync_direction`)
    ///   and transfer every in-scope item regardless of timestamps.
    /// - `options.dry_run`: perform every check that decides *whether* an
    ///   item would be transferred, but skip the actual network I/O, the
    ///   `last_sync_timestamp` update, and the manifest write -- so a dry
    ///   run has zero observable side effects, matching what the CLI's
    ///   "no actual changes will be made" message promises.
    pub async fn sync_with_options(&mut self, options: &SyncOptions<'_>) -> Result<SyncResult> {
        let mut result = SyncResult::new();

        // Process all items in the manifest
        for item in &self.sync_manifest.items {
            if let Some(directory) = options.directory {
                if !path_is_within(&item.local_path, directory) {
                    continue;
                }
            }

            match item.sync_direction {
                SyncDirection::Upload => {
                    if options.force || self.should_upload(item).await? {
                        if options.dry_run {
                            result.uploaded_files += 1;
                        } else {
                            match self.upload_file(item).await {
                                Ok(_) => result.uploaded_files += 1,
                                Err(e) => {
                                    result.failed_uploads += 1;
                                    result.errors.push(format!(
                                        "Upload failed for {}: {}",
                                        item.local_path.display(),
                                        e
                                    ));
                                }
                            }
                        }
                    }
                }
                SyncDirection::Download => {
                    if options.force || self.should_download(item).await? {
                        if options.dry_run {
                            result.downloaded_files += 1;
                        } else {
                            match self.download_file(item).await {
                                Ok(_) => result.downloaded_files += 1,
                                Err(e) => {
                                    result.failed_downloads += 1;
                                    result.errors.push(format!(
                                        "Download failed for {}: {}",
                                        item.remote_path, e
                                    ));
                                }
                            }
                        }
                    }
                }
                SyncDirection::Bidirectional => {
                    // Determine sync direction based on timestamps, unless
                    // forced: a forced bidirectional item uploads if the
                    // local copy exists (it is the source of truth) and
                    // downloads otherwise, mirroring
                    // `determine_sync_direction`'s own tie-break.
                    let sync_direction = if options.force {
                        Some(if item.local_path.exists() {
                            SyncDirection::Upload
                        } else {
                            SyncDirection::Download
                        })
                    } else {
                        self.determine_sync_direction(item).await?
                    };
                    match sync_direction {
                        Some(SyncDirection::Upload) => {
                            if options.dry_run {
                                result.uploaded_files += 1;
                            } else {
                                match self.upload_file(item).await {
                                    Ok(_) => result.uploaded_files += 1,
                                    Err(e) => {
                                        result.failed_uploads += 1;
                                        result.errors.push(format!(
                                            "Upload failed for {}: {}",
                                            item.local_path.display(),
                                            e
                                        ));
                                    }
                                }
                            }
                        }
                        Some(SyncDirection::Download) => {
                            if options.dry_run {
                                result.downloaded_files += 1;
                            } else {
                                match self.download_file(item).await {
                                    Ok(_) => result.downloaded_files += 1,
                                    Err(e) => {
                                        result.failed_downloads += 1;
                                        result.errors.push(format!(
                                            "Download failed for {}: {}",
                                            item.remote_path, e
                                        ));
                                    }
                                }
                            }
                        }
                        _ => {
                            // Files are in sync, no action needed
                            result.skipped_files += 1;
                        }
                    }
                }
            }
        }

        if !options.dry_run {
            // Update sync timestamp
            self.sync_manifest.last_sync_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();

            self.save_manifest().await?;
        }

        Ok(result)
    }

    /// Upload a specific file to cloud storage
    async fn upload_file(&self, item: &SyncableItem) -> Result<()> {
        tracing::info!(
            "Uploading {} to {}",
            item.local_path.display(),
            item.remote_path
        );

        // Validate that the local file exists
        if !item.local_path.exists() {
            return Err(anyhow::anyhow!(
                "Local file does not exist: {}",
                item.local_path.display()
            ));
        }

        // Read the file content
        let file_content = fs::read(&item.local_path).await?;

        // Verify file integrity using checksum
        let file_checksum = calculate_file_checksum(&file_content);
        if file_checksum != item.checksum {
            return Err(anyhow::anyhow!("File checksum mismatch during upload"));
        }

        // Compress the file if compression is enabled
        let upload_content = if self.config.compression_enabled {
            self.compress_data(&file_content).await?
        } else {
            file_content
        };

        // Encrypt the file if encryption is enabled
        let final_content = if self.config.encryption_enabled {
            self.encrypt_data(&upload_content).await?
        } else {
            upload_content
        };

        // Perform the actual upload based on provider
        match self.config.provider {
            StorageProvider::AWS => {
                self.upload_to_aws(&item.remote_path, &final_content)
                    .await?
            }
            StorageProvider::Azure => {
                self.upload_to_azure(&item.remote_path, &final_content)
                    .await?
            }
            StorageProvider::GoogleCloud => {
                self.upload_to_gcp(&item.remote_path, &final_content)
                    .await?
            }
            StorageProvider::MinIO | StorageProvider::S3Compatible => {
                self.upload_to_s3_compatible(&item.remote_path, &final_content)
                    .await?
            }
        }

        tracing::info!(
            "Successfully uploaded {} ({} bytes) to {}",
            item.local_path.display(),
            item.size_bytes,
            item.remote_path
        );

        Ok(())
    }

    /// Download a specific file from cloud storage
    async fn download_file(&self, item: &SyncableItem) -> Result<()> {
        tracing::info!(
            "Downloading {} to {}",
            item.remote_path,
            item.local_path.display()
        );

        // Ensure local directory exists
        if let Some(parent) = item.local_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Download the file content based on provider
        let downloaded_content = match self.config.provider {
            StorageProvider::AWS => self.download_from_aws(&item.remote_path).await?,
            StorageProvider::Azure => self.download_from_azure(&item.remote_path).await?,
            StorageProvider::GoogleCloud => self.download_from_gcp(&item.remote_path).await?,
            StorageProvider::MinIO | StorageProvider::S3Compatible => {
                self.download_from_s3_compatible(&item.remote_path).await?
            }
        };

        // Decrypt the file if encryption is enabled
        let decrypted_content = if self.config.encryption_enabled {
            self.decrypt_data(&downloaded_content).await?
        } else {
            downloaded_content
        };

        // Decompress the file if compression is enabled
        let final_content = if self.config.compression_enabled {
            self.decompress_data(&decrypted_content).await?
        } else {
            decrypted_content
        };

        // Verify file integrity using checksum
        let file_checksum = calculate_file_checksum(&final_content);
        if file_checksum != item.checksum {
            return Err(anyhow::anyhow!("File checksum mismatch during download"));
        }

        // Write the file to local storage
        fs::write(&item.local_path, &final_content).await?;

        // Update file metadata to match the remote version
        let metadata = fs::metadata(&item.local_path).await?;
        if metadata.len() != item.size_bytes {
            return Err(anyhow::anyhow!("Downloaded file size mismatch"));
        }

        tracing::info!(
            "Successfully downloaded {} ({} bytes) to {}",
            item.remote_path,
            item.size_bytes,
            item.local_path.display()
        );

        Ok(())
    }

    /// Check if a file should be uploaded
    async fn should_upload(&self, item: &SyncableItem) -> Result<bool> {
        // Check if local file exists and is newer than last sync
        if !item.local_path.exists() {
            return Ok(false);
        }

        let metadata = fs::metadata(&item.local_path).await?;
        let last_modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        // Upload if file was modified since last sync
        Ok(last_modified > self.sync_manifest.last_sync_timestamp)
    }

    /// Check if a file should be downloaded
    async fn should_download(&self, item: &SyncableItem) -> Result<bool> {
        // This would check remote file timestamp
        // For now, we'll just check if local file doesn't exist
        Ok(!item.local_path.exists())
    }

    /// Determine sync direction for bidirectional items
    async fn determine_sync_direction(&self, item: &SyncableItem) -> Result<Option<SyncDirection>> {
        if !item.local_path.exists() {
            return Ok(Some(SyncDirection::Download));
        }

        // In a real implementation, this would compare local and remote timestamps
        // For now, we'll prioritize upload if local file is newer
        let metadata = fs::metadata(&item.local_path).await?;
        let last_modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        if last_modified > self.sync_manifest.last_sync_timestamp {
            Ok(Some(SyncDirection::Upload))
        } else {
            Ok(None) // Files are in sync
        }
    }

    /// Calculate SHA256 checksum of a file
    async fn calculate_file_checksum(&self, path: &Path) -> Result<String> {
        let content = fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Save the sync manifest to disk
    async fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.local_cache_dir.join("sync_manifest.json");
        let content = serde_json::to_string_pretty(&self.sync_manifest)?;
        fs::write(manifest_path, content).await?;
        Ok(())
    }

    /// Get storage usage statistics
    pub async fn get_storage_stats(&self) -> Result<StorageStats> {
        let total_size: u64 = self
            .sync_manifest
            .items
            .iter()
            .map(|item| item.size_bytes)
            .sum();

        let local_files = self
            .sync_manifest
            .items
            .iter()
            .filter(|item| item.local_path.exists())
            .count();

        Ok(StorageStats {
            total_files: self.sync_manifest.items.len(),
            local_files,
            total_size_bytes: total_size,
            last_sync_timestamp: self.sync_manifest.last_sync_timestamp,
            cache_directory: self.local_cache_dir.clone(),
        })
    }

    /// Delete manifest items (and their local files) older than
    /// `max_age_days`. Equivalent to
    /// `cleanup_cache_with_options(max_age_days, false)`.
    pub async fn cleanup_cache(&mut self, max_age_days: u32) -> Result<CleanupResult> {
        self.cleanup_cache_with_options(max_age_days, false).await
    }

    /// Delete manifest items (and their local files) older than
    /// `max_age_days`. When `dry_run` is `true`, computes and reports
    /// exactly what *would* be removed -- including probing real file
    /// metadata to compute `freed_bytes` -- but deletes nothing and does
    /// not touch the manifest, so the CLI's "no files will actually be
    /// deleted" promise for `--dry-run` is actually true.
    pub async fn cleanup_cache_with_options(
        &mut self,
        max_age_days: u32,
        dry_run: bool,
    ) -> Result<CleanupResult> {
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
            - (max_age_days as u64 * 24 * 60 * 60);

        let mut removed_files = 0;
        let mut freed_bytes = 0u64;
        let mut errors = Vec::new();

        if dry_run {
            for item in &self.sync_manifest.items {
                if item.last_modified < cutoff_time && item.local_path.exists() {
                    removed_files += 1;
                    freed_bytes += item.size_bytes;
                }
            }
            return Ok(CleanupResult {
                removed_files,
                freed_bytes,
                errors,
            });
        }

        // Remove old items from manifest
        self.sync_manifest.items.retain(|item| {
            if item.last_modified < cutoff_time {
                if item.local_path.exists() {
                    match std::fs::remove_file(&item.local_path) {
                        Ok(_) => {
                            removed_files += 1;
                            freed_bytes += item.size_bytes;
                        }
                        Err(e) => {
                            errors.push(format!(
                                "Failed to remove {}: {}",
                                item.local_path.display(),
                                e
                            ));
                        }
                    }
                }
                false // Remove from manifest
            } else {
                true // Keep in manifest
            }
        });

        self.save_manifest().await?;

        Ok(CleanupResult {
            removed_files,
            freed_bytes,
            errors,
        })
    }

    /// Build a real, signed S3(-compatible) client for `provider_name`
    /// (used in error/log messages). `require_endpoint` should be `true`
    /// for MinIO/S3-compatible providers, which have no meaningful default
    /// host and therefore cannot proceed without an explicit endpoint URL.
    ///
    /// # Errors
    /// Returns [`CloudStorageError::NotConfigured`] -- never a fabricated
    /// client -- if credentials, bucket, or (when required) endpoint are
    /// missing.
    fn s3_backend(&self, provider_name: &str, require_endpoint: bool) -> Result<S3Backend> {
        let access_key = non_empty(&self.config.access_key).ok_or_else(|| {
            CloudStorageError::NotConfigured {
                provider: provider_name.to_string(),
                detail: "no access key configured (set AWS_ACCESS_KEY_ID, or storage.access_key in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
        })?;
        let secret_key = non_empty(&self.config.secret_key).ok_or_else(|| {
            CloudStorageError::NotConfigured {
                provider: provider_name.to_string(),
                detail: "no secret key configured (set AWS_SECRET_ACCESS_KEY, or storage.secret_key in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
        })?;
        if self.config.bucket_name.trim().is_empty() {
            return Err(CloudStorageError::NotConfigured {
                provider: provider_name.to_string(),
                detail: "no bucket_name configured in storage settings".to_string(),
            }
            .into());
        }
        if require_endpoint && non_empty(&self.config.endpoint).is_none() {
            return Err(CloudStorageError::NotConfigured {
                provider: provider_name.to_string(),
                detail: "no endpoint configured; S3-compatible storage (MinIO, R2, ...) requires an explicit endpoint URL (set VOIRS_S3_ENDPOINT, or storage.endpoint in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
            .into());
        }

        Ok(S3Backend::new(
            self.http_client.clone(),
            S3Config {
                access_key,
                secret_key,
                region: self.config.region.clone(),
                bucket: self.config.bucket_name.clone(),
                endpoint: non_empty(&self.config.endpoint),
            },
        ))
    }

    fn azure_backend(&self) -> Result<AzureBackend> {
        let account = non_empty(&self.config.access_key).ok_or_else(|| {
            CloudStorageError::NotConfigured {
                provider: "Azure Blob Storage".to_string(),
                detail: "no storage account name configured (set AZURE_STORAGE_ACCOUNT, or storage.access_key in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
        })?;
        let account_key = non_empty(&self.config.secret_key).ok_or_else(|| {
            CloudStorageError::NotConfigured {
                provider: "Azure Blob Storage".to_string(),
                detail: "no storage account key configured (set AZURE_STORAGE_KEY, or storage.secret_key in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
        })?;
        if self.config.bucket_name.trim().is_empty() {
            return Err(CloudStorageError::NotConfigured {
                provider: "Azure Blob Storage".to_string(),
                detail: "no container name configured (storage.bucket_name is used as the Azure container name)".to_string(),
            }
            .into());
        }

        Ok(AzureBackend::new(
            self.http_client.clone(),
            AzureConfig {
                account,
                account_key,
                container: self.config.bucket_name.clone(),
            },
        ))
    }

    fn gcp_backend(&self) -> Result<GcpBackend> {
        let token = non_empty(&self.config.secret_key).ok_or_else(|| {
            CloudStorageError::NotConfigured {
                provider: "Google Cloud Storage".to_string(),
                detail: "no OAuth2 bearer token configured (set GOOGLE_OAUTH_TOKEN, or storage.secret_key in ~/.config/voirs/cloud_config.toml)".to_string(),
            }
        })?;
        if self.config.bucket_name.trim().is_empty() {
            return Err(CloudStorageError::NotConfigured {
                provider: "Google Cloud Storage".to_string(),
                detail: "no bucket_name configured in storage settings".to_string(),
            }
            .into());
        }

        Ok(GcpBackend::new(
            self.http_client.clone(),
            GcpConfig {
                bucket: self.config.bucket_name.clone(),
                token,
            },
        ))
    }

    /// Upload to AWS S3: issues a real, SigV4-signed HTTPS `PUT` request.
    /// Returns `Ok(())` only if the object actually landed in the bucket.
    async fn upload_to_aws(&self, remote_path: &str, content: &[u8]) -> Result<()> {
        reject_oversized(content)?;
        let backend = self.s3_backend("AWS S3", false)?;
        tracing::debug!(
            "Uploading to AWS S3: s3://{}/{} ({} bytes)",
            self.config.bucket_name,
            remote_path,
            content.len()
        );
        backend.put_object(remote_path, content.to_vec()).await?;
        Ok(())
    }

    /// Download from AWS S3: issues a real, SigV4-signed HTTPS `GET`
    /// request and returns the exact bytes the server sent.
    async fn download_from_aws(&self, remote_path: &str) -> Result<Vec<u8>> {
        let backend = self.s3_backend("AWS S3", false)?;
        tracing::debug!(
            "Downloading from AWS S3: s3://{}/{}",
            self.config.bucket_name,
            remote_path
        );
        Ok(backend.get_object(remote_path).await?)
    }

    /// Upload to Azure Blob Storage: issues a real, Shared-Key-signed
    /// HTTPS `PUT Blob` request.
    async fn upload_to_azure(&self, remote_path: &str, content: &[u8]) -> Result<()> {
        let backend = self.azure_backend()?;
        tracing::debug!(
            "Uploading to Azure Blob Storage: {}/{} ({} bytes)",
            self.config.bucket_name,
            remote_path,
            content.len()
        );
        backend
            .put_blob(remote_path, content.to_vec(), "application/octet-stream")
            .await?;
        Ok(())
    }

    /// Download from Azure Blob Storage: issues a real, Shared-Key-signed
    /// HTTPS `GET Blob` request.
    async fn download_from_azure(&self, remote_path: &str) -> Result<Vec<u8>> {
        let backend = self.azure_backend()?;
        tracing::debug!(
            "Downloading from Azure Blob Storage: {}/{}",
            self.config.bucket_name,
            remote_path
        );
        Ok(backend.get_blob(remote_path).await?)
    }

    /// Upload to Google Cloud Storage: issues a real, bearer-authenticated
    /// HTTPS request against the GCS JSON API. Fails closed with
    /// [`CloudStorageError::NotConfigured`] if no OAuth2 token is set --
    /// never fabricates success.
    async fn upload_to_gcp(&self, remote_path: &str, content: &[u8]) -> Result<()> {
        let backend = self.gcp_backend()?;
        tracing::debug!(
            "Uploading to Google Cloud Storage: {}/{} ({} bytes)",
            self.config.bucket_name,
            remote_path,
            content.len()
        );
        backend
            .put_object(remote_path, content.to_vec(), "application/octet-stream")
            .await?;
        Ok(())
    }

    /// Download from Google Cloud Storage: issues a real,
    /// bearer-authenticated HTTPS request against the GCS JSON API.
    async fn download_from_gcp(&self, remote_path: &str) -> Result<Vec<u8>> {
        let backend = self.gcp_backend()?;
        tracing::debug!(
            "Downloading from Google Cloud Storage: {}/{}",
            self.config.bucket_name,
            remote_path
        );
        Ok(backend.get_object(remote_path).await?)
    }

    /// Upload to S3-compatible storage (MinIO, Cloudflare R2, ...): issues
    /// a real, SigV4-signed HTTP(S) `PUT` request against the configured
    /// `endpoint`, path-style.
    async fn upload_to_s3_compatible(&self, remote_path: &str, content: &[u8]) -> Result<()> {
        reject_oversized(content)?;
        let backend = self.s3_backend("S3-compatible storage", true)?;
        tracing::debug!(
            "Uploading to S3-compatible storage: {}/{} ({} bytes)",
            self.config.bucket_name,
            remote_path,
            content.len()
        );
        backend.put_object(remote_path, content.to_vec()).await?;
        Ok(())
    }

    /// Download from S3-compatible storage (MinIO, Cloudflare R2, ...):
    /// issues a real, SigV4-signed HTTP(S) `GET` request.
    async fn download_from_s3_compatible(&self, remote_path: &str) -> Result<Vec<u8>> {
        let backend = self.s3_backend("S3-compatible storage", true)?;
        tracing::debug!(
            "Downloading from S3-compatible storage: {}/{}",
            self.config.bucket_name,
            remote_path
        );
        Ok(backend.get_object(remote_path).await?)
    }

    /// Compress data using gzip
    async fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use oxiarc_deflate::GzipStreamEncoder;
        use std::io::Write;

        let mut encoder = GzipStreamEncoder::new(Vec::new(), 6);
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;

        tracing::debug!(
            "Compressed {} bytes to {} bytes",
            data.len(),
            compressed.len()
        );

        Ok(compressed)
    }

    /// Decompress data using gzip
    async fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use oxiarc_deflate::GzipStreamDecoder;
        use std::io::Read;

        let mut decoder = GzipStreamDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;

        tracing::debug!(
            "Decompressed {} bytes to {} bytes",
            data.len(),
            decompressed.len()
        );

        Ok(decompressed)
    }

    /// Encrypt data using AES-256-GCM
    async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Get 256-bit encryption key
        let key = self.get_encryption_key().await?;

        // Ensure key is exactly 32 bytes for AES-256
        let key_bytes: [u8; 32] = if key.len() >= 32 {
            key[..32]
                .try_into()
                .expect("slice of 32 bytes fits into [u8; 32]")
        } else {
            // Derive 32-byte key using SHA-256
            let mut hasher = Sha256::new();
            hasher.update(&key);
            hasher.finalize().into()
        };

        // Create cipher instance
        let cipher = Aes256Gcm::new(&key_bytes.into());

        // Generate random 96-bit nonce (12 bytes)
        let nonce_bytes = <[u8; 12]>::generate();
        let nonce: &Nonce<_> = (&nonce_bytes).into();

        // Encrypt data (GCM automatically adds authentication tag)
        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("AES-GCM encryption failed: {}", e))?;

        // Format: [nonce (12 bytes)] + [ciphertext + tag (16 bytes)]
        let mut encrypted = Vec::with_capacity(12 + ciphertext.len());
        encrypted.extend_from_slice(&nonce_bytes);
        encrypted.extend_from_slice(&ciphertext);

        tracing::debug!(
            "Encrypted {} bytes to {} bytes (including nonce and tag)",
            data.len(),
            encrypted.len()
        );

        Ok(encrypted)
    }

    /// Decrypt data using AES-256-GCM
    async fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Ensure we have at least nonce (12 bytes) + tag (16 bytes)
        if data.len() < 28 {
            anyhow::bail!(
                "Invalid encrypted data: too short (need at least 28 bytes, got {})",
                data.len()
            );
        }

        // Get 256-bit encryption key
        let key = self.get_encryption_key().await?;

        // Ensure key is exactly 32 bytes for AES-256
        let key_bytes: [u8; 32] = if key.len() >= 32 {
            key[..32]
                .try_into()
                .expect("slice of 32 bytes fits into [u8; 32]")
        } else {
            // Derive 32-byte key using SHA-256
            let mut hasher = Sha256::new();
            hasher.update(&key);
            hasher.finalize().into()
        };

        // Create cipher instance
        let cipher = Aes256Gcm::new(&key_bytes.into());

        // Extract nonce (first 12 bytes)
        let nonce: &Nonce<_> = data[..12]
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid nonce length"))?;

        // Extract ciphertext (remaining bytes include authentication tag)
        let ciphertext = &data[12..];

        // Decrypt and verify authentication tag
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("AES-GCM decryption failed: {}", e))?;

        tracing::debug!(
            "Decrypted {} bytes to {} bytes",
            data.len(),
            plaintext.len()
        );

        Ok(plaintext)
    }

    /// Get encryption key from configuration or environment
    async fn get_encryption_key(&self) -> Result<Vec<u8>> {
        // Priority order for key sources:
        // 1. VOIRS_ENCRYPTION_KEY environment variable (highest priority)
        // 2. Key from cloud provider's KMS (if configured)
        // 3. Key from config file
        // 4. Derive from access credentials (fallback)

        // Check environment variable first
        if let Ok(key_str) = std::env::var("VOIRS_ENCRYPTION_KEY") {
            if key_str.len() >= 32 {
                tracing::debug!("Using encryption key from VOIRS_ENCRYPTION_KEY environment");
                return Ok(key_str.as_bytes().to_vec());
            } else {
                tracing::warn!(
                    "VOIRS_ENCRYPTION_KEY is too short ({} bytes), deriving with SHA-256",
                    key_str.len()
                );
                let mut hasher = Sha256::new();
                hasher.update(key_str.as_bytes());
                return Ok(hasher.finalize().to_vec());
            }
        }

        // Check config file key
        if let Ok(key_str) = std::env::var("VOIRS_CONFIG_ENCRYPTION_KEY") {
            tracing::debug!("Using encryption key from config file");
            let mut hasher = Sha256::new();
            hasher.update(key_str.as_bytes());
            return Ok(hasher.finalize().to_vec());
        }

        // Fallback: derive key from access credentials
        // This ensures encryption works even without explicit key configuration
        // but credentials must remain consistent for decryption to work
        let key_material = format!(
            "{:?}:{}:{}",
            self.config.provider,
            self.config.bucket_name,
            self.config
                .access_key
                .as_ref()
                .unwrap_or(&"voirs-default".to_string())
        );

        tracing::warn!(
            "No explicit encryption key configured, deriving from credentials (secure but requires consistent config)"
        );

        let mut hasher = Sha256::new();
        hasher.update(key_material.as_bytes());
        // Add salt for additional security
        hasher.update(b"voirs-cloud-storage-encryption-v1");

        Ok(hasher.finalize().to_vec())
    }
}

/// `Some(trimmed)` if `value` is `Some` and non-blank after trimming,
/// `None` otherwise. Used to treat an empty-string credential the same as
/// an absent one, so `CloudStorageConfig { access_key: Some(String::new()), .. }`
/// fails closed exactly like `access_key: None` rather than being handed
/// to a signer as a valid (but empty) key.
fn non_empty(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Reject uploads larger than what a single `PUT` can carry on real S3
/// (5 GiB); multipart upload is not implemented by this client. Returns a
/// clear, typed error instead of letting an oversized request fail deep
/// inside the transport layer with an opaque HTTP error.
fn reject_oversized(content: &[u8]) -> Result<()> {
    if content.len() as u64 > MAX_SINGLE_PUT_BYTES {
        return Err(CloudStorageError::Unsupported {
            provider: "S3".to_string(),
            detail: format!(
                "object is {} bytes, which exceeds the {} byte single-PUT limit; multipart upload is not implemented",
                content.len(),
                MAX_SINGLE_PUT_BYTES
            ),
        }
        .into());
    }
    Ok(())
}

/// Calculate SHA256 checksum of data
fn calculate_file_checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub uploaded_files: u32,
    pub downloaded_files: u32,
    pub skipped_files: u32,
    pub failed_uploads: u32,
    pub failed_downloads: u32,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_files: usize,
    pub local_files: usize,
    pub total_size_bytes: u64,
    pub last_sync_timestamp: u64,
    pub cache_directory: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    pub removed_files: u32,
    pub freed_bytes: u64,
    pub errors: Vec<String>,
}

impl SyncManifest {
    fn new() -> Self {
        Self {
            version: 1,
            last_sync_timestamp: 0,
            items: Vec::new(),
            total_size_bytes: 0,
            checksum: String::new(),
        }
    }
}

impl SyncResult {
    fn new() -> Self {
        Self {
            uploaded_files: 0,
            downloaded_files: 0,
            skipped_files: 0,
            failed_uploads: 0,
            failed_downloads: 0,
            errors: Vec::new(),
        }
    }
}

impl Default for CloudStorageConfig {
    fn default() -> Self {
        Self {
            provider: StorageProvider::S3Compatible,
            bucket_name: "voirs-models".to_string(),
            region: "us-west-1".to_string(),
            access_key: None,
            secret_key: None,
            endpoint: None,
            encryption_enabled: true,
            compression_enabled: true,
            sync_interval_seconds: 3600, // 1 hour
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_storage_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig::default();

        let manager = CloudStorageManager::new(config, temp_dir.path().to_path_buf());
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_add_to_sync() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig::default();
        let mut manager = CloudStorageManager::new(config, temp_dir.path().to_path_buf()).unwrap();

        // Create a test file
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").await.unwrap();

        let result = manager
            .add_to_sync(
                test_file,
                "remote/test.txt".to_string(),
                SyncDirection::Upload,
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(manager.sync_manifest.items.len(), 1);
    }

    #[tokio::test]
    async fn test_storage_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig::default();
        let manager = CloudStorageManager::new(config, temp_dir.path().to_path_buf()).unwrap();

        let stats = manager.get_storage_stats().await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.local_files, 0);
    }

    #[test]
    fn test_sync_direction_serialization() {
        let direction = SyncDirection::Bidirectional;
        let serialized = serde_json::to_string(&direction);
        assert!(serialized.is_ok());

        let deserialized: Result<SyncDirection, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }

    fn unconfigured_config(provider: StorageProvider) -> CloudStorageConfig {
        CloudStorageConfig {
            provider,
            bucket_name: "voirs-test".to_string(),
            region: "us-east-1".to_string(),
            access_key: None,
            secret_key: None,
            endpoint: None,
            encryption_enabled: false,
            compression_enabled: false,
            sync_interval_seconds: 300,
        }
    }

    /// Regression test for the fabrication this module used to contain:
    /// `upload_to_aws` used to `tokio::time::sleep` and return `Ok(())`
    /// unconditionally, regardless of whether any credentials existed.
    /// With no credentials configured, `sync()` must report a real,
    /// explanatory failure -- never silently report success.
    #[tokio::test]
    async fn upload_fails_closed_without_credentials_no_fabricated_success() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.bin");
        fs::write(&test_file, b"local model bytes").await.unwrap();

        let config = unconfigured_config(StorageProvider::AWS);
        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager
            .add_to_sync(
                test_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();

        let result = manager
            .sync()
            .await
            .expect("sync() itself must not error; per-item failures live in SyncResult");

        assert_eq!(result.uploaded_files, 0, "no credentials => no upload");
        assert_eq!(result.failed_uploads, 1);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.to_lowercase().contains("not configured")
                    || e.to_lowercase().contains("access key")),
            "error must clearly explain missing credentials, got: {:?}",
            result.errors
        );
    }

    /// Same regression, S3-compatible/MinIO path: with credentials but no
    /// `endpoint`, this must fail closed too (path-style addressing has no
    /// sensible default host).
    #[tokio::test]
    async fn s3_compatible_upload_fails_closed_without_endpoint() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.bin");
        fs::write(&test_file, b"local model bytes").await.unwrap();

        let mut config = unconfigured_config(StorageProvider::S3Compatible);
        config.access_key = Some("ak".to_string());
        config.secret_key = Some("sk".to_string());
        // endpoint deliberately left None.

        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager
            .add_to_sync(
                test_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();

        let result = manager.sync().await.expect("sync() itself must not error");
        assert_eq!(result.uploaded_files, 0);
        assert_eq!(result.failed_uploads, 1);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.to_lowercase().contains("endpoint")),
            "error must mention the missing endpoint, got: {:?}",
            result.errors
        );
    }

    /// End-to-end regression test for the upload fabrication: runs the
    /// *real* `sync()` pipeline (manifest -> `upload_to_s3_compatible` ->
    /// `S3Backend::put_object`) against a bare loopback TCP listener
    /// standing in for an S3-compatible server, and asserts the server
    /// receives the exact real file bytes -- proving nothing is
    /// `tokio::time::sleep`-and-`Ok`-faked anymore.
    #[tokio::test]
    async fn upload_delivers_real_bytes_to_mock_s3_compatible_server() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || -> Vec<u8> {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().expect("accept");
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();

            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut chunk).expect("read");
                assert!(n > 0, "connection closed before headers were complete");
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let content_length: usize = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())
                        .flatten()
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let n = stream.read(&mut chunk).expect("read body");
                assert!(n > 0, "connection closed before body was complete");
                buf.extend_from_slice(&chunk[..n]);
            }
            let body = buf[header_end..header_end + content_length].to_vec();

            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
            let _ = stream.flush();
            body
        });

        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.bin");
        let real_bytes = b"these are the real local file bytes, not a placeholder".to_vec();
        fs::write(&test_file, &real_bytes).await.unwrap();

        let mut config = unconfigured_config(StorageProvider::S3Compatible);
        config.access_key = Some("test-access-key".to_string());
        config.secret_key = Some("test-secret-key".to_string());
        config.endpoint = Some(format!("http://127.0.0.1:{port}"));

        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager
            .add_to_sync(
                test_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), manager.sync())
            .await
            .expect("sync must not hang")
            .expect("sync must succeed");

        let received_body = tokio::task::spawn_blocking(move || server.join().unwrap())
            .await
            .unwrap();

        assert_eq!(result.uploaded_files, 1, "errors: {:?}", result.errors);
        assert_eq!(
            received_body, real_bytes,
            "server must receive the exact real file bytes"
        );
    }

    /// End-to-end regression test for the download fabrication: the old
    /// `s3_compatible_get_object` returned the literal string
    /// `"S3-compatible content for {path}"` regardless of what was asked
    /// for. This drives the real `sync()` pipeline against a mock server
    /// that serves known real bytes and asserts they land on disk
    /// unchanged.
    #[tokio::test]
    async fn download_delivers_real_bytes_from_mock_s3_compatible_server() {
        let expected_bytes =
            b"real object bytes served by the mock S3-compatible endpoint".to_vec();
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(&expected_bytes);
            hex::encode(hasher.finalize())
        };

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        let body_for_server = expected_bytes.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().expect("accept");
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).expect("read request");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body_for_server.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body_for_server).unwrap();
            let _ = stream.flush();
        });

        let temp_dir = TempDir::new().unwrap();
        let download_path = temp_dir.path().join("downloaded.bin");

        let mut config = unconfigured_config(StorageProvider::S3Compatible);
        config.access_key = Some("test-access-key".to_string());
        config.secret_key = Some("test-secret-key".to_string());
        config.endpoint = Some(format!("http://127.0.0.1:{port}"));

        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager.sync_manifest.items.push(SyncableItem {
            local_path: download_path.clone(),
            remote_path: "models/real.bin".to_string(),
            last_modified: 0,
            checksum,
            size_bytes: expected_bytes.len() as u64,
            sync_priority: SyncPriority::Normal,
            sync_direction: SyncDirection::Download,
        });

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), manager.sync())
            .await
            .expect("sync must not hang")
            .expect("sync must succeed");

        tokio::task::spawn_blocking(move || server.join().unwrap())
            .await
            .unwrap();

        assert_eq!(result.downloaded_files, 1, "errors: {:?}", result.errors);
        let on_disk = fs::read(&download_path).await.unwrap();
        assert_eq!(
            on_disk, expected_bytes,
            "downloaded file must contain the real server bytes"
        );
        assert!(
            !String::from_utf8_lossy(&on_disk).contains("content for"),
            "must not contain the old fabricated placeholder string"
        );
    }

    /// Regression test for the "dry_run is printed but ignored" finding:
    /// with `dry_run: true`, no network request may reach the server (the
    /// mock listener below is never even connected to -- if `sync()`
    /// accidentally attempted the real upload, the test would hang until
    /// the outer timeout and fail), the manifest's `last_sync_timestamp`
    /// must stay at its initial value, and the local file must be
    /// untouched.
    #[tokio::test]
    async fn dry_run_upload_performs_no_network_io_and_no_manifest_write() {
        // Bind a listener but deliberately never `accept()` on it: if
        // sync_with_options were to attempt the real upload despite
        // dry_run=true, the connection would hang and the test's timeout
        // would catch it.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();

        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.bin");
        fs::write(&test_file, b"local model bytes").await.unwrap();

        let mut config = unconfigured_config(StorageProvider::S3Compatible);
        config.access_key = Some("test-access-key".to_string());
        config.secret_key = Some("test-secret-key".to_string());
        config.endpoint = Some(format!("http://127.0.0.1:{port}"));

        let cache_dir = temp_dir.path().join("cache");
        let mut manager = CloudStorageManager::new(config, cache_dir.clone()).unwrap();
        manager
            .add_to_sync(
                test_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();
        assert_eq!(manager.sync_manifest.last_sync_timestamp, 0);

        // add_to_sync() legitimately writes the manifest (recording the
        // newly-added item) -- capture its content here so the assertion
        // below can prove the dry run made *no further* write, rather
        // than incorrectly asserting the file never exists at all.
        let manifest_path = cache_dir.join("sync_manifest.json");
        let manifest_after_add = fs::read_to_string(&manifest_path)
            .await
            .expect("manifest written by add_to_sync");

        let options = SyncOptions {
            dry_run: true,
            ..Default::default()
        };
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            manager.sync_with_options(&options),
        )
        .await
        .expect("dry run must not hang waiting on network I/O")
        .expect("dry run must succeed");

        assert_eq!(
            result.uploaded_files, 1,
            "dry run must still report what *would* be uploaded"
        );
        assert_eq!(
            manager.sync_manifest.last_sync_timestamp, 0,
            "dry run must not update last_sync_timestamp"
        );

        let manifest_after_dry_run = fs::read_to_string(&manifest_path)
            .await
            .expect("manifest file must still exist from add_to_sync");
        assert_eq!(
            manifest_after_dry_run, manifest_after_add,
            "dry run must not write the sync manifest to disk again"
        );

        drop(listener); // never accepted a connection
    }

    /// Regression test for the "--directory is printed but ignored"
    /// finding: an item outside the requested directory must be left
    /// completely alone (not uploaded, not counted as skipped either --
    /// it was out of scope).
    #[tokio::test]
    async fn directory_option_filters_items_outside_the_requested_directory() {
        let temp_dir = TempDir::new().unwrap();
        let in_scope_dir = temp_dir.path().join("in_scope");
        let out_of_scope_dir = temp_dir.path().join("out_of_scope");
        fs::create_dir_all(&in_scope_dir).await.unwrap();
        fs::create_dir_all(&out_of_scope_dir).await.unwrap();

        let in_scope_file = in_scope_dir.join("model.bin");
        let out_of_scope_file = out_of_scope_dir.join("other.bin");
        fs::write(&in_scope_file, b"in scope").await.unwrap();
        fs::write(&out_of_scope_file, b"out of scope")
            .await
            .unwrap();

        let config = unconfigured_config(StorageProvider::AWS); // no credentials: any real
                                                                // attempt to upload the
                                                                // in-scope file will fail
                                                                // closed, which is fine --
                                                                // we only assert the
                                                                // out-of-scope file was
                                                                // never even considered.
        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager
            .add_to_sync(
                in_scope_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();
        manager
            .add_to_sync(
                out_of_scope_file,
                "models/other.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();
        assert_eq!(manager.sync_manifest.items.len(), 2);

        let options = SyncOptions {
            directory: Some(in_scope_dir.as_path()),
            ..Default::default()
        };
        let result = manager.sync_with_options(&options).await.unwrap();

        // Only the in-scope item was attempted at all (and failed closed,
        // since no credentials are configured) -- the out-of-scope item
        // contributes to neither uploaded_files, failed_uploads, nor
        // skipped_files.
        assert_eq!(result.uploaded_files, 0);
        assert_eq!(result.failed_uploads, 1);
        assert_eq!(result.skipped_files, 0);
        assert_eq!(
            result.errors.len(),
            1,
            "exactly one item (the in-scope one) should have been attempted"
        );
    }

    /// Regression test for the "--force is printed but ignored" finding:
    /// an already-up-to-date item (local file unmodified since last sync)
    /// is normally skipped by `should_upload`'s staleness check; `force`
    /// must bypass that check and attempt the transfer anyway.
    #[tokio::test]
    async fn force_option_bypasses_staleness_check() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("model.bin");
        fs::write(&test_file, b"local model bytes").await.unwrap();

        let config = unconfigured_config(StorageProvider::AWS); // no credentials -> fails
                                                                // closed either way; this
                                                                // test only asserts whether
                                                                // the *attempt* happens.
        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager
            .add_to_sync(
                test_file,
                "models/model.bin".to_string(),
                SyncDirection::Upload,
            )
            .await
            .unwrap();

        // Simulate "already synced": last_sync_timestamp far in the
        // future relative to the file's actual mtime, so should_upload()
        // would normally return false.
        manager.sync_manifest.last_sync_timestamp = u64::MAX / 2;

        // Without force: should_upload() says "not modified since last
        // sync" -> zero attempts, zero failures.
        let no_force = manager
            .sync_with_options(&SyncOptions::default())
            .await
            .unwrap();
        assert_eq!(no_force.uploaded_files, 0);
        assert_eq!(no_force.failed_uploads, 0);

        // With force: the item is attempted (and fails closed on missing
        // credentials, proving it was genuinely attempted rather than
        // skipped).
        let forced = manager
            .sync_with_options(&SyncOptions {
                force: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(forced.uploaded_files, 0);
        assert_eq!(forced.failed_uploads, 1);
    }

    /// Regression test for the "cleanup --dry-run is printed but ignored"
    /// finding: with `dry_run: true`, the file must not be deleted and
    /// the manifest item must remain.
    #[tokio::test]
    async fn cleanup_cache_dry_run_deletes_nothing() {
        let temp_dir = TempDir::new().unwrap();
        let stale_file = temp_dir.path().join("stale.bin");
        fs::write(&stale_file, b"stale bytes").await.unwrap();

        let config = unconfigured_config(StorageProvider::AWS);
        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager.sync_manifest.items.push(SyncableItem {
            local_path: stale_file.clone(),
            remote_path: "models/stale.bin".to_string(),
            last_modified: 0, // definitely older than any max_age_days cutoff
            checksum: String::new(),
            size_bytes: 11,
            sync_priority: SyncPriority::Normal,
            sync_direction: SyncDirection::Upload,
        });

        let result = manager.cleanup_cache_with_options(1, true).await.unwrap();

        assert_eq!(
            result.removed_files, 1,
            "dry run must still report what *would* be removed"
        );
        assert_eq!(result.freed_bytes, 11);
        assert!(
            stale_file.exists(),
            "dry run must not actually delete the file"
        );
        assert_eq!(
            manager.sync_manifest.items.len(),
            1,
            "dry run must not modify the manifest"
        );
    }

    /// Companion test: without dry_run, the same setup really does delete
    /// the file and shrink the manifest -- proving the dry-run test above
    /// isn't passing merely because deletion was already broken.
    #[tokio::test]
    async fn cleanup_cache_without_dry_run_really_deletes() {
        let temp_dir = TempDir::new().unwrap();
        let stale_file = temp_dir.path().join("stale.bin");
        fs::write(&stale_file, b"stale bytes").await.unwrap();

        let config = unconfigured_config(StorageProvider::AWS);
        let mut manager = CloudStorageManager::new(config, temp_dir.path().join("cache")).unwrap();
        manager.sync_manifest.items.push(SyncableItem {
            local_path: stale_file.clone(),
            remote_path: "models/stale.bin".to_string(),
            last_modified: 0,
            checksum: String::new(),
            size_bytes: 11,
            sync_priority: SyncPriority::Normal,
            sync_direction: SyncDirection::Upload,
        });

        let result = manager.cleanup_cache_with_options(1, false).await.unwrap();

        assert_eq!(result.removed_files, 1);
        assert!(!stale_file.exists(), "file must actually be deleted");
        assert_eq!(manager.sync_manifest.items.len(), 0);
    }

    #[test]
    fn path_is_within_matches_nested_paths_and_rejects_others() {
        assert!(path_is_within(
            Path::new("/a/b/c/file.bin"),
            Path::new("/a/b")
        ));
        assert!(path_is_within(Path::new("/a/b"), Path::new("/a/b")));
        assert!(!path_is_within(
            Path::new("/a/other/file.bin"),
            Path::new("/a/b")
        ));
        assert!(!path_is_within(Path::new("/a"), Path::new("/a/b")));
    }
}
