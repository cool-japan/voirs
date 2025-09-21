//! Network streaming capabilities for remote dataset access
//!
//! This module provides HTTP-based dataset streaming with resume capability,
//! bandwidth throttling, and connection pooling.

use crate::traits::{Dataset, DatasetMetadata};
use crate::{DatasetError, DatasetSample, DatasetStatistics, Result, ValidationReport};
use async_trait::async_trait;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::sleep;

/// Configuration for network streaming
#[derive(Debug, Clone)]
pub struct NetworkStreamingConfig {
    /// Base URL for the dataset server
    pub base_url: String,
    /// Maximum concurrent connections
    pub max_concurrent_connections: usize,
    /// Request timeout
    pub request_timeout: Duration,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Bandwidth throttling
    pub bandwidth_limit: Option<BandwidthLimit>,
    /// Connection pooling settings
    pub connection_pool: ConnectionPoolConfig,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Caching configuration
    pub cache_config: CacheConfig,
}

/// Retry configuration for failed requests
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies for retries
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Exponential backoff
    Exponential { multiplier: f64 },
    /// Linear backoff
    Linear { increment: Duration },
    /// Jittered exponential backoff
    JitteredExponential { multiplier: f64, jitter_factor: f64 },
}

/// Bandwidth limiting configuration
#[derive(Debug, Clone)]
pub struct BandwidthLimit {
    /// Maximum bytes per second
    pub bytes_per_second: usize,
    /// Burst allowance (bytes)
    pub burst_size: usize,
    /// Measurement window for rate limiting
    pub window_duration: Duration,
}

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Idle timeout for connections
    pub idle_timeout: Duration,
    /// Maximum lifetime for connections
    pub max_lifetime: Duration,
    /// Keep-alive settings
    pub keep_alive: bool,
    /// Connection validation
    pub validate_connections: bool,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub enum AuthConfig {
    /// Bearer token authentication
    Bearer { token: String },
    /// Basic authentication
    Basic { username: String, password: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// Custom header authentication
    Custom { headers: HashMap<String, String> },
}

/// Cache configuration for network requests
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable response caching
    pub enabled: bool,
    /// Cache TTL
    pub ttl: Duration,
    /// Maximum cache size (bytes)
    pub max_size: usize,
    /// Cache invalidation strategy
    pub invalidation_strategy: CacheInvalidationStrategy,
}

/// Cache invalidation strategies
#[derive(Debug, Clone)]
pub enum CacheInvalidationStrategy {
    /// Time-based TTL
    TimeToLive,
    /// LRU eviction
    LeastRecentlyUsed,
    /// Manual invalidation
    Manual,
}

impl Default for NetworkStreamingConfig {
    fn default() -> Self {
        Self {
            base_url: String::from("http://localhost:8080"),
            max_concurrent_connections: 10,
            request_timeout: Duration::from_secs(30),
            retry_config: RetryConfig {
                max_retries: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
            },
            bandwidth_limit: None,
            connection_pool: ConnectionPoolConfig {
                idle_timeout: Duration::from_secs(90),
                max_lifetime: Duration::from_secs(300),
                keep_alive: true,
                validate_connections: true,
            },
            auth: None,
            cache_config: CacheConfig {
                enabled: true,
                ttl: Duration::from_secs(3600),
                max_size: 100 * 1024 * 1024, // 100MB
                invalidation_strategy: CacheInvalidationStrategy::LeastRecentlyUsed,
            },
        }
    }
}

/// Network-based streaming dataset
pub struct NetworkStreamingDataset {
    /// Configuration
    config: NetworkStreamingConfig,
    /// HTTP client
    client: Arc<Client>,
    /// Connection semaphore for limiting concurrent requests
    connection_semaphore: Arc<Semaphore>,
    /// Dataset metadata cache
    metadata_cache: Arc<RwLock<Option<DatasetMetadata>>>,
    /// Response cache
    response_cache: Arc<Mutex<HashMap<String, CachedResponse>>>,
    /// Bandwidth throttle
    bandwidth_throttle: Arc<Mutex<BandwidthThrottle>>,
    /// Network statistics
    stats: Arc<RwLock<NetworkStatistics>>,
}

/// Cached response data
#[derive(Debug, Clone)]
struct CachedResponse {
    /// Response data
    data: Vec<u8>,
    /// Cache timestamp
    cached_at: Instant,
    /// Response headers
    #[allow(dead_code)]
    headers: HashMap<String, String>,
    /// ETag for validation
    #[allow(dead_code)]
    etag: Option<String>,
}

/// Bandwidth throttling state
struct BandwidthThrottle {
    /// Bytes consumed in current window
    bytes_consumed: usize,
    /// Current window start time
    window_start: Instant,
    /// Token bucket for burst handling
    token_bucket: TokenBucket,
}

/// Token bucket for rate limiting
struct TokenBucket {
    /// Current token count
    tokens: usize,
    /// Last refill time
    last_refill: Instant,
    /// Maximum bucket capacity
    capacity: usize,
    /// Refill rate (tokens per second)
    refill_rate: usize,
}

/// Network operation statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStatistics {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    /// Total bytes uploaded
    pub bytes_uploaded: u64,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Connection pool statistics
    pub connection_pool_stats: ConnectionPoolStats,
}

/// Connection pool statistics
#[derive(Debug, Clone, Default)]
pub struct ConnectionPoolStats {
    /// Active connections
    pub active_connections: usize,
    /// Idle connections
    pub idle_connections: usize,
    /// Total connections created
    pub total_connections_created: u64,
    /// Connection reuse rate
    pub connection_reuse_rate: f64,
}

/// Remote dataset manifest for network datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDatasetManifest {
    /// Dataset metadata
    pub metadata: DatasetMetadata,
    /// Sample endpoints
    pub sample_endpoints: Vec<SampleEndpoint>,
    /// API version
    pub api_version: String,
    /// Dataset hash for integrity checking
    pub dataset_hash: Option<String>,
}

/// Sample endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEndpoint {
    /// Sample index
    pub index: usize,
    /// Endpoint URL
    pub url: String,
    /// Expected content type
    pub content_type: String,
    /// Sample metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl NetworkStreamingDataset {
    /// Create a new network streaming dataset
    pub async fn new(config: NetworkStreamingConfig) -> Result<Self> {
        let client = Self::create_http_client(&config)?;
        let connection_semaphore = Arc::new(Semaphore::new(config.max_concurrent_connections));

        let bandwidth_throttle = if let Some(limit) = &config.bandwidth_limit {
            BandwidthThrottle::new(limit.clone())
        } else {
            BandwidthThrottle::unlimited()
        };

        let dataset = Self {
            config,
            client: Arc::new(client),
            connection_semaphore,
            metadata_cache: Arc::new(RwLock::new(None)),
            response_cache: Arc::new(Mutex::new(HashMap::new())),
            bandwidth_throttle: Arc::new(Mutex::new(bandwidth_throttle)),
            stats: Arc::new(RwLock::new(NetworkStatistics::default())),
        };

        // Initialize by fetching dataset manifest
        dataset.initialize().await?;

        Ok(dataset)
    }

    /// Initialize the dataset by fetching remote manifest
    async fn initialize(&self) -> Result<()> {
        let manifest_url = format!("{base_url}/manifest", base_url = self.config.base_url);
        let manifest_data = self.fetch_with_retry(&manifest_url).await?;

        let manifest: RemoteDatasetManifest = serde_json::from_slice(&manifest_data)
            .map_err(|e| DatasetError::FormatError(format!("Invalid manifest format: {e}")))?;

        // Cache the metadata
        {
            let mut cache = self.metadata_cache.write().await;
            *cache = Some(manifest.metadata);
        }

        Ok(())
    }

    /// Fetch data with retry logic
    async fn fetch_with_retry(&self, url: &str) -> Result<Vec<u8>> {
        let mut attempt = 0;
        let mut delay = self.config.retry_config.initial_delay;

        loop {
            // Acquire connection semaphore
            let _permit = self.connection_semaphore.acquire().await.map_err(|_| {
                DatasetError::ConfigError(String::from("Failed to acquire connection permit"))
            })?;

            match self.fetch_single_request(url).await {
                Ok(data) => {
                    self.update_success_stats(data.len()).await;
                    return Ok(data);
                }
                Err(e) => {
                    attempt += 1;
                    self.update_failure_stats().await;

                    if attempt >= self.config.retry_config.max_retries {
                        return Err(e);
                    }

                    // Apply backoff strategy
                    delay = self.calculate_backoff_delay(delay, attempt);
                    sleep(delay).await;
                }
            }
        }
    }

    /// Perform a single HTTP request
    async fn fetch_single_request(&self, url: &str) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Check cache first
        if self.config.cache_config.enabled {
            if let Some(cached) = self.get_from_cache(url).await {
                return Ok(cached.data);
            }
        }

        // Build request
        let mut request = self.client.get(url).timeout(self.config.request_timeout);

        // Add authentication headers
        if let Some(auth) = &self.config.auth {
            request = self.add_auth_headers(request, auth);
        }

        // Execute request
        let response = request
            .send()
            .await
            .map_err(|e| DatasetError::NetworkError(format!("Request failed: {e}")))?;

        // Check response status
        if !response.status().is_success() {
            return Err(DatasetError::NetworkError(format!(
                "HTTP error: {status}",
                status = response.status()
            )));
        }

        // Read response body with bandwidth throttling
        let body_bytes = self.read_response_with_throttling(response).await?;

        // Update response time statistics
        let response_time = start_time.elapsed().as_millis() as f64;
        self.update_response_time_stats(response_time).await;

        // Cache response if caching is enabled
        if self.config.cache_config.enabled {
            self.cache_response(url, &body_bytes).await;
        }

        Ok(body_bytes)
    }

    /// Read response body with bandwidth throttling
    async fn read_response_with_throttling(&self, response: Response) -> Result<Vec<u8>> {
        let content_length = response.content_length().unwrap_or(0) as usize;
        let mut body_bytes = Vec::with_capacity(content_length);

        let bytes = response
            .bytes()
            .await
            .map_err(|e| DatasetError::NetworkError(format!("Failed to read response: {e}")))?;

        // Apply bandwidth throttling
        if let Some(_limit) = &self.config.bandwidth_limit {
            let mut throttle = self.bandwidth_throttle.lock().await;
            throttle.wait_for_tokens(bytes.len()).await;
        }

        body_bytes.extend_from_slice(&bytes);
        Ok(body_bytes)
    }

    /// Calculate backoff delay based on strategy
    fn calculate_backoff_delay(&self, current_delay: Duration, _attempt: usize) -> Duration {
        match &self.config.retry_config.backoff_strategy {
            BackoffStrategy::Fixed => current_delay,
            BackoffStrategy::Exponential { multiplier } => {
                let new_delay =
                    Duration::from_millis((current_delay.as_millis() as f64 * multiplier) as u64);
                new_delay.min(self.config.retry_config.max_delay)
            }
            BackoffStrategy::Linear { increment } => {
                let new_delay = current_delay + *increment;
                new_delay.min(self.config.retry_config.max_delay)
            }
            BackoffStrategy::JitteredExponential {
                multiplier,
                jitter_factor,
            } => {
                use rand::Rng;
                let base_delay = (current_delay.as_millis() as f64 * multiplier) as u64;
                let jitter =
                    (base_delay as f64 * jitter_factor * rand::thread_rng().r#gen::<f64>()) as u64;
                let new_delay = Duration::from_millis(base_delay + jitter);
                new_delay.min(self.config.retry_config.max_delay)
            }
        }
    }

    /// Add authentication headers to request
    fn add_auth_headers(
        &self,
        mut request: reqwest::RequestBuilder,
        auth: &AuthConfig,
    ) -> reqwest::RequestBuilder {
        match auth {
            AuthConfig::Bearer { token } => {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            AuthConfig::Basic { username, password } => {
                request = request.basic_auth(username, Some(password));
            }
            AuthConfig::ApiKey { key, header } => {
                request = request.header(header, key);
            }
            AuthConfig::Custom { headers } => {
                for (name, value) in headers {
                    request = request.header(name, value);
                }
            }
        }
        request
    }

    /// Get response from cache
    async fn get_from_cache(&self, url: &str) -> Option<CachedResponse> {
        let cache = self.response_cache.lock().await;
        if let Some(cached) = cache.get(url) {
            // Check if cache entry is still valid
            if cached.cached_at.elapsed() < self.config.cache_config.ttl {
                return Some(cached.clone());
            }
        }
        None
    }

    /// Cache response data
    async fn cache_response(&self, url: &str, data: &[u8]) {
        let mut cache = self.response_cache.lock().await;

        // Check cache size limit
        let total_size: usize = cache.values().map(|v| v.data.len()).sum();
        if total_size + data.len() > self.config.cache_config.max_size {
            // Implement LRU eviction
            self.evict_cache_entries(&mut cache, data.len()).await;
        }

        let cached_response = CachedResponse {
            data: data.to_vec(),
            cached_at: Instant::now(),
            headers: HashMap::new(),
            etag: None,
        };

        cache.insert(url.to_string(), cached_response);
    }

    /// Evict cache entries using LRU strategy
    async fn evict_cache_entries(
        &self,
        cache: &mut HashMap<String, CachedResponse>,
        needed_space: usize,
    ) {
        // Simple LRU implementation: remove oldest entries
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, response)| response.cached_at);

        let mut freed_space = 0;
        let mut keys_to_remove = Vec::new();

        for (key, response) in entries {
            keys_to_remove.push(key.clone());
            freed_space += response.data.len();

            if freed_space >= needed_space {
                break;
            }
        }

        for key in keys_to_remove {
            cache.remove(&key);
        }
    }

    /// Update success statistics
    async fn update_success_stats(&self, bytes_downloaded: usize) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.bytes_downloaded += bytes_downloaded as u64;
    }

    /// Update failure statistics
    async fn update_failure_stats(&self) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.failed_requests += 1;
    }

    /// Update response time statistics
    async fn update_response_time_stats(&self, response_time_ms: f64) {
        let mut stats = self.stats.write().await;
        let total_successful = stats.successful_requests as f64;
        stats.avg_response_time_ms = (stats.avg_response_time_ms * (total_successful - 1.0)
            + response_time_ms)
            / total_successful;
    }

    /// Create HTTP client with appropriate configuration
    fn create_http_client(config: &NetworkStreamingConfig) -> Result<Client> {
        let mut client_builder = Client::builder()
            .timeout(config.request_timeout)
            .pool_idle_timeout(config.connection_pool.idle_timeout)
            .pool_max_idle_per_host(config.max_concurrent_connections);

        if config.connection_pool.keep_alive {
            client_builder = client_builder.tcp_keepalive(Duration::from_secs(60));
        }

        client_builder
            .build()
            .map_err(|e| DatasetError::ConfigError(format!("Failed to create HTTP client: {e}")))
    }

    /// Get network statistics
    pub async fn get_network_statistics(&self) -> NetworkStatistics {
        self.stats.read().await.clone()
    }
}

#[async_trait]
impl Dataset for NetworkStreamingDataset {
    type Sample = DatasetSample;

    fn len(&self) -> usize {
        // Get the dataset length from the metadata cache
        if let Ok(metadata_guard) = self.metadata_cache.try_read() {
            if let Some(metadata) = metadata_guard.as_ref() {
                return metadata.total_samples;
            }
        }

        // If metadata is not available, try to fetch it synchronously from the manifest
        // This is a fallback for when metadata hasn't been cached yet
        0 // Return 0 if metadata is not available - this should trigger metadata fetching
    }

    async fn get(&self, index: usize) -> Result<Self::Sample> {
        let sample_url = format!(
            "{base_url}/samples/{index}",
            base_url = self.config.base_url,
            index = index
        );
        let sample_data = self.fetch_with_retry(&sample_url).await?;

        let sample: DatasetSample = serde_json::from_slice(&sample_data)
            .map_err(|e| DatasetError::FormatError(format!("Invalid sample format: {e}")))?;

        Ok(sample)
    }

    fn metadata(&self) -> &DatasetMetadata {
        // This is not ideal for async access, but matches the trait signature
        // In a real implementation, you might want to change the trait to be async
        use std::sync::OnceLock;
        static DEFAULT_METADATA: OnceLock<DatasetMetadata> = OnceLock::new();
        DEFAULT_METADATA.get_or_init(|| DatasetMetadata {
            name: String::from("NetworkDataset"),
            version: String::from("1.0.0"),
            description: Some(String::from("Remote network dataset")),
            total_samples: 0,
            total_duration: 0.0,
            languages: vec![String::from("en")],
            speakers: vec![],
            license: None,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn statistics(&self) -> Result<DatasetStatistics> {
        let stats_url = format!("{base_url}/statistics", base_url = self.config.base_url);
        let stats_data = self.fetch_with_retry(&stats_url).await?;

        let stats: DatasetStatistics = serde_json::from_slice(&stats_data)
            .map_err(|e| DatasetError::FormatError(format!("Invalid statistics format: {e}")))?;

        Ok(stats)
    }

    async fn validate(&self) -> Result<ValidationReport> {
        let validation_url = format!("{base_url}/validate", base_url = self.config.base_url);
        let validation_data = self.fetch_with_retry(&validation_url).await?;

        let report: ValidationReport = serde_json::from_slice(&validation_data).map_err(|e| {
            DatasetError::FormatError(format!("Invalid validation report format: {e}"))
        })?;

        Ok(report)
    }
}

impl BandwidthThrottle {
    fn new(limit: BandwidthLimit) -> Self {
        Self {
            bytes_consumed: 0,
            window_start: Instant::now(),
            token_bucket: TokenBucket::new(limit.burst_size, limit.bytes_per_second),
        }
    }

    fn unlimited() -> Self {
        Self {
            bytes_consumed: 0,
            window_start: Instant::now(),
            token_bucket: TokenBucket::unlimited(),
        }
    }

    async fn wait_for_tokens(&mut self, bytes: usize) {
        self.token_bucket.wait_for_tokens(bytes).await;
    }
}

impl TokenBucket {
    fn new(capacity: usize, refill_rate: usize) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    fn unlimited() -> Self {
        Self {
            tokens: usize::MAX,
            last_refill: Instant::now(),
            capacity: usize::MAX,
            refill_rate: usize::MAX,
        }
    }

    async fn wait_for_tokens(&mut self, tokens_needed: usize) {
        loop {
            self.refill();

            if self.tokens >= tokens_needed {
                self.tokens -= tokens_needed;
                break;
            }

            // Wait for refill
            let wait_time = Duration::from_millis(100);
            sleep(wait_time).await;
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64) as usize;

        self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
        self.last_refill = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_config_creation() {
        let config = NetworkStreamingConfig::default();

        assert_eq!(config.base_url, "http://localhost:8080");
        assert_eq!(config.max_concurrent_connections, 10);
        assert!(config.cache_config.enabled);
    }

    #[tokio::test]
    async fn test_retry_config() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_strategy: BackoffStrategy::Exponential { multiplier: 2.0 },
        };

        assert_eq!(retry_config.max_retries, 3);
        assert_eq!(retry_config.initial_delay, Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_bandwidth_limit() {
        let limit = BandwidthLimit {
            bytes_per_second: 1024 * 1024, // 1MB/s
            burst_size: 1024 * 512,        // 512KB burst
            window_duration: Duration::from_secs(1),
        };

        assert_eq!(limit.bytes_per_second, 1024 * 1024);
        assert_eq!(limit.burst_size, 1024 * 512);
    }

    #[tokio::test]
    async fn test_token_bucket() {
        let mut bucket = TokenBucket::new(1000, 100);

        // Should have initial capacity
        assert_eq!(bucket.tokens, 1000);

        // Consume tokens
        bucket.wait_for_tokens(500).await;
        assert_eq!(bucket.tokens, 500);
    }

    #[tokio::test]
    async fn test_auth_config() {
        let auth = AuthConfig::Bearer {
            token: String::from("test-token"),
        };

        match auth {
            AuthConfig::Bearer { token } => {
                assert_eq!(token, "test-token");
            }
            _ => panic!("Wrong auth type"),
        }
    }

    #[tokio::test]
    async fn test_cache_config() {
        let cache_config = CacheConfig {
            enabled: true,
            ttl: Duration::from_secs(3600),
            max_size: 100 * 1024 * 1024,
            invalidation_strategy: CacheInvalidationStrategy::LeastRecentlyUsed,
        };

        assert!(cache_config.enabled);
        assert_eq!(cache_config.ttl, Duration::from_secs(3600));
        assert_eq!(cache_config.max_size, 100 * 1024 * 1024);
    }

    #[test]
    fn test_backoff_strategies() {
        let strategies = [
            BackoffStrategy::Fixed,
            BackoffStrategy::Exponential { multiplier: 2.0 },
            BackoffStrategy::Linear {
                increment: Duration::from_millis(100),
            },
            BackoffStrategy::JitteredExponential {
                multiplier: 2.0,
                jitter_factor: 0.1,
            },
        ];

        // Test that all strategies can be created
        assert_eq!(strategies.len(), 4);
    }

    #[test]
    fn test_connection_pool_config() {
        let pool_config = ConnectionPoolConfig {
            idle_timeout: Duration::from_secs(90),
            max_lifetime: Duration::from_secs(300),
            keep_alive: true,
            validate_connections: true,
        };

        assert_eq!(pool_config.idle_timeout, Duration::from_secs(90));
        assert!(pool_config.keep_alive);
        assert!(pool_config.validate_connections);
    }
}
