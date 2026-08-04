//! Cloud storage integration for voirs-dataset
//!
//! Provides real, network-backed object storage for dataset hosting:
//! [`CloudProvider::AWS`] is implemented against the actual S3 REST API,
//! signed with a hand-rolled, Pure-Rust AWS Signature Version 4 (SigV4)
//! implementation (see the private `sigv4` and `s3` submodules) built on
//! `reqwest` + `hmac`/`sha2`. No AWS SDK, no OpenSSL, no vendored C signing
//! library.
//!
//! [`CloudProvider::GCP`] and [`CloudProvider::Azure`] are not implemented:
//! GCS requires an OAuth2 service-account JWT (RS256) token exchange and
//! Azure requires its own Shared Key signing scheme, and this build performs
//! neither. Operations against those providers return a clear, typed
//! [`DatasetError::CloudStorage`] explaining exactly that, rather than
//! fabricating success.

mod s3;
mod sigv4;

use crate::{DatasetError, DatasetSample, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::io::AsyncRead;

/// Cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon Web Services S3
    AWS {
        /// AWS region
        region: String,
        /// Access key ID
        access_key_id: String,
        /// Secret access key
        secret_access_key: String,
        /// Optional session token
        session_token: Option<String>,
    },
    /// Google Cloud Storage
    GCP {
        /// Project ID
        project_id: String,
        /// Service account key path
        service_account_key: String,
        /// Optional location
        location: Option<String>,
    },
    /// Azure Blob Storage
    Azure {
        /// Storage account name
        account_name: String,
        /// Storage account key
        account_key: String,
        /// Optional container name
        container: Option<String>,
    },
}

/// Cloud storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageConfig {
    /// Cloud provider configuration
    pub provider: CloudProvider,
    /// Default bucket/container name
    pub bucket: String,
    /// Base path for dataset objects
    pub base_path: String,
    /// Enable compression for uploads
    pub compression: bool,
    /// Enable encryption at rest
    pub encryption: bool,
    /// Default chunk size for multipart uploads (in bytes)
    pub chunk_size: usize,
    /// Maximum concurrent uploads
    pub max_concurrent_uploads: usize,
    /// Timeout for operations (in seconds)
    pub timeout_seconds: u64,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Retry configuration for cloud operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Base delay between retries (in milliseconds)
    pub base_delay_ms: u64,
    /// Maximum delay between retries (in milliseconds)
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for CloudStorageConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::AWS {
                region: String::from("us-east-1"),
                access_key_id: String::from(""),
                secret_access_key: String::from(""),
                session_token: None,
            },
            bucket: String::from(""),
            base_path: String::from("datasets"),
            compression: true,
            encryption: true,
            chunk_size: 8 * 1024 * 1024, // 8MB
            max_concurrent_uploads: 4,
            timeout_seconds: 300,
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Cloud storage interface
#[async_trait::async_trait]
pub trait CloudStorage: Send + Sync {
    /// Upload a dataset to cloud storage
    async fn upload_dataset(&self, dataset_name: &str, samples: &[DatasetSample])
        -> Result<String>;

    /// Download a dataset from cloud storage
    async fn download_dataset(&self, dataset_name: &str) -> Result<Vec<DatasetSample>>;

    /// Stream a dataset from cloud storage
    async fn stream_dataset(&self, dataset_name: &str) -> Result<Box<dyn AsyncRead + Unpin>>;

    /// Upload a single file to cloud storage
    async fn upload_file(&self, local_path: &Path, remote_path: &str) -> Result<String>;

    /// Download a single file from cloud storage
    async fn download_file(&self, remote_path: &str, local_path: &Path) -> Result<()>;

    /// List objects in a bucket/container
    async fn list_objects(&self, prefix: &str) -> Result<Vec<String>>;

    /// Delete an object from cloud storage
    async fn delete_object(&self, path: &str) -> Result<()>;

    /// Get object metadata
    async fn get_metadata(&self, path: &str) -> Result<ObjectMetadata>;

    /// Check if an object exists
    async fn object_exists(&self, path: &str) -> Result<bool>;

    /// Generate a presigned URL for direct access
    async fn generate_presigned_url(&self, path: &str, expiry_seconds: u64) -> Result<String>;
}

/// Object metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object size in bytes
    pub size: u64,
    /// Content type/MIME type
    pub content_type: String,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
    /// ETag/checksum
    pub etag: String,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Which real client (if any) backs a configured provider. Resolved once at
/// construction time so per-call operations neither re-read credentials nor
/// silently swallow a resolution failure -- every operation on an
/// `Unsupported` provider fails with the same clear reason.
enum ProviderClient {
    Aws(s3::S3Client),
    Unsupported(String),
}

/// Cloud storage implementation.
///
/// [`CloudStorageImpl::new`] never fails merely because credentials are
/// missing or the provider is GCP/Azure -- construction always succeeds (as
/// callers may reasonably build a `CloudStorageImpl` before credentials are
/// available), but every trait operation on such an instance fails closed
/// with a clear, typed [`DatasetError`] explaining exactly what is missing.
pub struct CloudStorageImpl {
    config: CloudStorageConfig,
    provider_client: ProviderClient,
}

impl CloudStorageImpl {
    /// Create a new cloud storage instance
    pub fn new(config: CloudStorageConfig) -> Result<Self> {
        // `reqwest` is built with `rustls-no-provider`; a default crypto
        // provider must be installed before any TLS handshake.
        crate::tls::ensure_crypto_provider();

        let provider_client = match &config.provider {
            CloudProvider::AWS { .. } => match s3::resolve_aws_credentials(&config.provider) {
                Ok(creds) => {
                    if config.bucket.is_empty() {
                        ProviderClient::Unsupported("S3 bucket name is not configured".to_string())
                    } else {
                        let http = reqwest::Client::builder()
                            .timeout(std::time::Duration::from_secs(
                                config.timeout_seconds.max(1),
                            ))
                            .build()
                            .map_err(|e| {
                                DatasetError::CloudStorage(format!(
                                    "failed to build HTTP client: {e}"
                                ))
                            })?;
                        ProviderClient::Aws(s3::S3Client::new(
                            http,
                            config.bucket.clone(),
                            creds,
                            &config.retry_config,
                        ))
                    }
                }
                Err(e) => ProviderClient::Unsupported(e.to_string()),
            },
            CloudProvider::GCP { .. } | CloudProvider::Azure { .. } => {
                ProviderClient::Unsupported(s3::unsupported_provider_reason(&config.provider))
            }
        };

        Ok(Self {
            config,
            provider_client,
        })
    }

    /// Test-only constructor: injects an already-built [`s3::S3Client`]
    /// (typically pointed at a local loopback server) directly, bypassing
    /// normal credential resolution and HTTP client construction entirely.
    #[cfg(test)]
    fn from_test_s3_client(config: CloudStorageConfig, client: s3::S3Client) -> Self {
        Self {
            config,
            provider_client: ProviderClient::Aws(client),
        }
    }

    fn s3_client(&self) -> Result<&s3::S3Client> {
        match &self.provider_client {
            ProviderClient::Aws(client) => Ok(client),
            ProviderClient::Unsupported(reason) => Err(DatasetError::CloudStorage(reason.clone())),
        }
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        if self.config.bucket.is_empty() {
            return Err(DatasetError::Configuration(String::from(
                "Bucket name cannot be empty",
            )));
        }

        match &self.config.provider {
            CloudProvider::AWS {
                access_key_id,
                secret_access_key,
                ..
            } => {
                if access_key_id.is_empty() || secret_access_key.is_empty() {
                    return Err(DatasetError::Configuration(String::from(
                        "AWS credentials cannot be empty",
                    )));
                }
            }
            CloudProvider::GCP {
                project_id,
                service_account_key,
                ..
            } => {
                if project_id.is_empty() || service_account_key.is_empty() {
                    return Err(DatasetError::Configuration(String::from(
                        "GCP credentials cannot be empty",
                    )));
                }
            }
            CloudProvider::Azure {
                account_name,
                account_key,
                ..
            } => {
                if account_name.is_empty() || account_key.is_empty() {
                    return Err(DatasetError::Configuration(String::from(
                        "Azure credentials cannot be empty",
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get full object path
    fn get_object_path(&self, relative_path: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_path.trim_end_matches('/'),
            relative_path
        )
    }

    /// Calculate exponential backoff delay (shares its formula with the S3
    /// client's own internal per-request retry loop; see
    /// [`s3::backoff_delay_ms`]).
    fn calculate_backoff_delay(&self, attempt: usize) -> u64 {
        s3::backoff_delay_ms(&self.config.retry_config, attempt)
    }
}

/// Encode `audio` as an in-memory 16-bit PCM WAV file.
fn encode_wav_bytes(audio: &crate::AudioData) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: audio.channels() as u16,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &sample in audio.samples() {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Best-effort content type from a remote path's extension.
fn guess_content_type(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("json") => "application/json",
        Some("wav") => "audio/wav",
        Some("txt") => "text/plain",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    }
}

#[async_trait::async_trait]
impl CloudStorage for CloudStorageImpl {
    async fn upload_dataset(
        &self,
        dataset_name: &str,
        samples: &[DatasetSample],
    ) -> Result<String> {
        let client = self.s3_client()?;

        let manifest_json = serde_json::to_vec_pretty(samples)?;
        let manifest_key = self.get_object_path(&format!("{dataset_name}/manifest.json"));
        client
            .put_object(&manifest_key, manifest_json, "application/json")
            .await?;

        // Also upload each sample's audio as a standalone WAV object, so the
        // dataset can be fetched per-sample (e.g. browsing the bucket)
        // rather than only via the manifest.
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(
            self.config.max_concurrent_uploads.max(1),
        ));
        let mut upload_tasks = Vec::with_capacity(samples.len());
        for sample in samples {
            let sem = semaphore.clone();
            let client = client.clone();
            let audio_key =
                self.get_object_path(&format!("{dataset_name}/audio/{}.wav", sample.id));
            let wav_bytes = encode_wav_bytes(&sample.audio)?;
            upload_tasks.push(async move {
                let _permit = sem.acquire().await.map_err(|e| {
                    DatasetError::CloudStorage(format!("upload semaphore closed: {e}"))
                })?;
                client.put_object(&audio_key, wav_bytes, "audio/wav").await
            });
        }
        futures::future::try_join_all(upload_tasks).await?;

        Ok(client.object_url(&manifest_key))
    }

    async fn download_dataset(&self, dataset_name: &str) -> Result<Vec<DatasetSample>> {
        let client = self.s3_client()?;
        let manifest_key = self.get_object_path(&format!("{dataset_name}/manifest.json"));
        // The manifest already contains every sample's full, lossless audio
        // (serde-serialized `AudioData`), so re-parsing it alone
        // reconstructs the complete dataset -- no need to separately
        // re-fetch and lossily re-decode the standalone per-sample WAV
        // files `upload_dataset` also writes for external browsing.
        let bytes = client.get_object(&manifest_key).await?;
        let samples: Vec<DatasetSample> = serde_json::from_slice(&bytes)?;
        Ok(samples)
    }

    async fn stream_dataset(&self, dataset_name: &str) -> Result<Box<dyn AsyncRead + Unpin>> {
        use futures::StreamExt;

        let client = self.s3_client()?;
        let manifest_key = self.get_object_path(&format!("{dataset_name}/manifest.json"));
        let response = client.get_object_response(&manifest_key).await?;
        let stream = response
            .bytes_stream()
            .map(|item| item.map_err(std::io::Error::other));
        Ok(Box::new(tokio_util::io::StreamReader::new(stream)))
    }

    async fn upload_file(&self, local_path: &Path, remote_path: &str) -> Result<String> {
        let full_path = self.get_object_path(remote_path);
        let client = self.s3_client()?;
        let bytes = tokio::fs::read(local_path)
            .await
            .map_err(DatasetError::IoError)?;
        let content_type = guess_content_type(remote_path);
        client.put_object(&full_path, bytes, content_type).await?;
        Ok(client.object_url(&full_path))
    }

    async fn download_file(&self, remote_path: &str, local_path: &Path) -> Result<()> {
        let full_path = self.get_object_path(remote_path);
        let client = self.s3_client()?;
        let bytes = client.get_object(&full_path).await?;
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(DatasetError::IoError)?;
        }
        tokio::fs::write(local_path, bytes)
            .await
            .map_err(DatasetError::IoError)
    }

    async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.get_object_path(prefix);
        self.s3_client()?.list_objects(&full_prefix).await
    }

    async fn delete_object(&self, path: &str) -> Result<()> {
        let full_path = self.get_object_path(path);
        self.s3_client()?.delete_object(&full_path).await
    }

    async fn get_metadata(&self, path: &str) -> Result<ObjectMetadata> {
        let full_path = self.get_object_path(path);
        self.s3_client()?
            .head_object(&full_path)
            .await?
            .ok_or_else(|| DatasetError::CloudStorage(format!("object not found: {full_path}")))
    }

    async fn object_exists(&self, path: &str) -> Result<bool> {
        let full_path = self.get_object_path(path);
        Ok(self.s3_client()?.head_object(&full_path).await?.is_some())
    }

    async fn generate_presigned_url(&self, path: &str, expiry_seconds: u64) -> Result<String> {
        let full_path = self.get_object_path(path);
        self.s3_client()?
            .presigned_get_url(&full_path, expiry_seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, QualityMetrics};
    use std::collections::HashMap as StdHashMap;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn test_cloud_storage_config_default() {
        let config = CloudStorageConfig::default();
        assert_eq!(config.base_path, "datasets");
        assert!(config.compression);
        assert!(config.encryption);
        assert_eq!(config.chunk_size, 8 * 1024 * 1024);
        assert_eq!(config.max_concurrent_uploads, 4);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.base_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 30_000);
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[tokio::test]
    async fn test_cloud_storage_creation() {
        let config = CloudStorageConfig::default();
        let storage = CloudStorageImpl::new(config);
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_object_path_generation() {
        let config = CloudStorageConfig::default();
        let storage = CloudStorageImpl::new(config).unwrap();

        let path = storage.get_object_path("test/file.txt");
        assert_eq!(path, "datasets/test/file.txt");
    }

    #[tokio::test]
    async fn test_backoff_delay_calculation() {
        let config = CloudStorageConfig::default();
        let storage = CloudStorageImpl::new(config).unwrap();

        let delay_0 = storage.calculate_backoff_delay(0);
        let delay_1 = storage.calculate_backoff_delay(1);
        let delay_2 = storage.calculate_backoff_delay(2);

        assert_eq!(delay_0, 100);
        assert_eq!(delay_1, 200);
        assert_eq!(delay_2, 400);
    }

    /// Regression test: an unconfigured (empty-credential) AWS provider, and
    /// the unimplemented GCP/Azure providers, must fail every real operation
    /// with a clear typed error -- never a fabricated success.
    #[tokio::test]
    async fn test_unconfigured_and_unimplemented_providers_fail_closed() {
        for provider in [
            CloudProvider::AWS {
                region: String::new(),
                access_key_id: String::new(),
                secret_access_key: String::new(),
                session_token: None,
            },
            CloudProvider::GCP {
                project_id: "proj".to_string(),
                service_account_key: "{}".to_string(),
                location: None,
            },
            CloudProvider::Azure {
                account_name: "acct".to_string(),
                account_key: "key".to_string(),
                container: None,
            },
        ] {
            let config = CloudStorageConfig {
                provider,
                bucket: "some-bucket".to_string(),
                ..CloudStorageConfig::default()
            };
            let storage = CloudStorageImpl::new(config).unwrap();
            let result = storage.list_objects("prefix").await;
            assert!(result.is_err(), "expected a typed error, got {result:?}");
        }
    }

    #[test]
    fn test_encode_wav_bytes_round_trips_through_hound() {
        let samples = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.25];
        let audio = AudioData::new(samples.clone(), 22050, 1);
        let wav_bytes = encode_wav_bytes(&audio).unwrap();
        assert!(!wav_bytes.is_empty());

        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav_bytes)).unwrap();
        assert_eq!(reader.spec().sample_rate, 22050);
        assert_eq!(reader.spec().channels, 1);
        let decoded: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        assert_eq!(decoded.len(), samples.len());
        for (original, round_tripped) in samples.iter().zip(decoded.iter()) {
            assert!(
                (original - round_tripped).abs() < 1e-3,
                "expected {original}, got {round_tripped}"
            );
        }
    }

    fn make_sample(id: &str, text: &str, samples: Vec<f32>) -> DatasetSample {
        DatasetSample::new(
            id.to_string(),
            text.to_string(),
            AudioData::new(samples, 16000, 1),
            LanguageCode::EnUs,
        )
        .with_quality(QualityMetrics {
            snr: None,
            clipping: None,
            dynamic_range: None,
            spectral_quality: None,
            overall_quality: None,
        })
    }

    // ---- Minimal loopback HTTP server used to prove real byte movement ----
    //
    // This is not a mock in the "returns canned data" sense: it is a real
    // TCP server that `S3Client` connects to over real sockets, sending a
    // fully SigV4-signed HTTP/1.1 request that this server parses by hand
    // and answers according to what it actually received.

    struct MockS3Server {
        addr: std::net::SocketAddr,
        store: Arc<Mutex<StdHashMap<String, Vec<u8>>>>,
    }

    async fn read_one_request(
        socket: &mut TcpStream,
    ) -> (String, String, StdHashMap<String, String>, Vec<u8>) {
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

        let mut headers = StdHashMap::new();
        for line in lines {
            if line.is_empty() {
                continue;
            }
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

    async fn spawn_mock_s3_server() -> MockS3Server {
        // Every caller of this helper immediately constructs its own
        // `reqwest::Client`, which (even for plain-HTTP loopback requests)
        // requires a rustls `CryptoProvider` to already be installed.
        crate::tls::ensure_crypto_provider();

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock S3 listener");
        let addr = listener.local_addr().expect("local addr");
        let store: Arc<Mutex<StdHashMap<String, Vec<u8>>>> =
            Arc::new(Mutex::new(StdHashMap::new()));
        let store_for_task = store.clone();

        tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let store = store_for_task.clone();
                tokio::spawn(async move {
                    let (method, path, _headers, body) = read_one_request(&mut socket).await;
                    // `path` is `/<key>` or `/?<query>` (ListObjectsV2);
                    // requests carrying signed authorization headers
                    // demonstrate the client actually built and sent them,
                    // even though this test server does not itself verify
                    // the signature (it merely proves it arrived and that
                    // the wire bytes match what the client intended).
                    let path_only = path.split('?').next().unwrap_or("");
                    let key = path_only.trim_start_matches('/').to_string();
                    match method.as_str() {
                        "PUT" => {
                            store.lock().expect("store lock").insert(key, body);
                            write_raw(
                                &mut socket,
                                "HTTP/1.1 200 OK\r\ncontent-length: 0\r\netag: \"mock\"\r\nconnection: close\r\n\r\n",
                                b"",
                            )
                            .await;
                        }
                        "GET" => {
                            let found = store.lock().expect("store lock").get(&key).cloned();
                            match found {
                                Some(data) => {
                                    let head = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: application/octet-stream\r\nconnection: close\r\n\r\n",
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
                        "HEAD" => {
                            let found = store.lock().expect("store lock").get(&key).cloned();
                            match found {
                                Some(data) => {
                                    let head = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: application/octet-stream\r\netag: \"mock-etag\"\r\nlast-modified: Fri, 24 May 2013 00:00:00 GMT\r\nconnection: close\r\n\r\n",
                                        data.len()
                                    );
                                    write_raw(&mut socket, &head, b"").await;
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

        MockS3Server { addr, store }
    }

    fn test_credentials() -> s3::AwsCredentials {
        s3::AwsCredentials {
            region: "us-east-1".to_string(),
            access_key_id: "AKIATESTACCESSKEY".to_string(),
            secret_access_key: "test/secret/access/key".to_string(),
            session_token: None,
        }
    }

    #[tokio::test]
    async fn s3_client_round_trips_real_bytes_over_loopback_http() {
        let server = spawn_mock_s3_server().await;
        let client = s3::S3Client::for_test(
            reqwest::Client::new(),
            server.addr,
            test_credentials(),
            &RetryConfig::default(),
        );

        let key = "datasets/demo/manifest.json";
        let payload = b"{\"hello\":\"world\",\"n\":42}".to_vec();

        // Nothing exists yet -- proves this is a real request/response
        // cycle, not a canned "found" answer.
        assert!(client.head_object(key).await.unwrap().is_none());
        assert!(client.get_object(key).await.is_err());

        client
            .put_object(key, payload.clone(), "application/json")
            .await
            .unwrap();

        // The mock server received these exact bytes over a real TCP
        // socket -- not a simulated / fabricated success.
        assert_eq!(
            server.store.lock().unwrap().get(key),
            Some(&payload),
            "server did not receive the exact uploaded bytes"
        );

        let fetched = client.get_object(key).await.unwrap();
        assert_eq!(
            fetched, payload,
            "downloaded bytes must match uploaded bytes exactly"
        );

        let meta = client.head_object(key).await.unwrap().unwrap();
        assert_eq!(meta.size, payload.len() as u64);

        client.delete_object(key).await.unwrap();
        assert!(client.head_object(key).await.unwrap().is_none());
        assert!(client.get_object(key).await.is_err());
    }

    #[tokio::test]
    async fn s3_client_different_keys_produce_different_stored_bytes() {
        // Regression guard against any accidental constant/canned response:
        // uploading different content under different keys must be
        // retrievable independently and distinctly.
        let server = spawn_mock_s3_server().await;
        let client = s3::S3Client::for_test(
            reqwest::Client::new(),
            server.addr,
            test_credentials(),
            &RetryConfig::default(),
        );

        client
            .put_object("a.txt", b"first payload".to_vec(), "text/plain")
            .await
            .unwrap();
        client
            .put_object(
                "b.txt",
                b"a completely different second payload".to_vec(),
                "text/plain",
            )
            .await
            .unwrap();

        let a = client.get_object("a.txt").await.unwrap();
        let b = client.get_object("b.txt").await.unwrap();
        assert_eq!(a, b"first payload".to_vec());
        assert_eq!(b, b"a completely different second payload".to_vec());
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn cloud_storage_impl_upload_and_download_dataset_round_trip_real_bytes() {
        let server = spawn_mock_s3_server().await;
        let client = s3::S3Client::for_test(
            reqwest::Client::new(),
            server.addr,
            test_credentials(),
            &RetryConfig::default(),
        );
        let storage = CloudStorageImpl::from_test_s3_client(CloudStorageConfig::default(), client);

        let samples = vec![
            make_sample("s1", "hello world", vec![0.1, 0.2, -0.1, -0.2]),
            make_sample("s2", "goodbye world", vec![0.5, -0.5, 0.0]),
        ];

        let url = storage.upload_dataset("demo", &samples).await.unwrap();
        assert!(url.contains("manifest.json"));

        // The manifest object was really written to the (mock) store.
        let manifest_key = "datasets/demo/manifest.json";
        assert!(server.store.lock().unwrap().contains_key(manifest_key));
        // So were the per-sample audio objects.
        assert!(server
            .store
            .lock()
            .unwrap()
            .contains_key("datasets/demo/audio/s1.wav"));
        assert!(server
            .store
            .lock()
            .unwrap()
            .contains_key("datasets/demo/audio/s2.wav"));

        let downloaded = storage.download_dataset("demo").await.unwrap();
        assert_eq!(downloaded.len(), 2);
        assert_eq!(downloaded[0].id, "s1");
        assert_eq!(downloaded[0].text, "hello world");
        assert_eq!(downloaded[1].id, "s2");
        assert_eq!(downloaded[1].text, "goodbye world");
        // Audio samples survive the JSON manifest round trip exactly
        // (unlike the lossy 16-bit WAV side files).
        assert_eq!(downloaded[0].audio.samples(), samples[0].audio.samples());
        assert_eq!(downloaded[1].audio.samples(), samples[1].audio.samples());
    }

    #[tokio::test]
    async fn cloud_storage_impl_object_lifecycle_uses_real_requests() {
        let server = spawn_mock_s3_server().await;
        let client = s3::S3Client::for_test(
            reqwest::Client::new(),
            server.addr,
            test_credentials(),
            &RetryConfig::default(),
        );
        let storage = CloudStorageImpl::from_test_s3_client(CloudStorageConfig::default(), client);

        assert!(!storage.object_exists("thing.bin").await.unwrap());

        // Write a small real temp file so `upload_file` reads real bytes
        // from disk, not a synthetic in-memory buffer.
        let upload_path =
            std::env::temp_dir().join(format!("voirs_cloud_test_{}.bin", std::process::id()));
        std::fs::write(&upload_path, b"local file contents").unwrap();

        let url = storage
            .upload_file(&upload_path, "thing.bin")
            .await
            .unwrap();
        let _ = std::fs::remove_file(&upload_path);
        assert!(url.contains("thing.bin"));

        assert!(storage.object_exists("thing.bin").await.unwrap());
        let meta = storage.get_metadata("thing.bin").await.unwrap();
        assert_eq!(meta.size, b"local file contents".len() as u64);

        let download_dir = std::env::temp_dir();
        let download_path =
            download_dir.join(format!("voirs_cloud_test_dl_{}.bin", std::process::id()));
        storage
            .download_file("thing.bin", &download_path)
            .await
            .unwrap();
        let downloaded = std::fs::read(&download_path).unwrap();
        assert_eq!(downloaded, b"local file contents");
        let _ = std::fs::remove_file(&download_path);

        storage.delete_object("thing.bin").await.unwrap();
        assert!(!storage.object_exists("thing.bin").await.unwrap());
    }
}
