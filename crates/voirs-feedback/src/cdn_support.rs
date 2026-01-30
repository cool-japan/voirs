//! CDN Support for Global Content Delivery
//!
//! Provides Content Delivery Network integration for optimizing global delivery
//! of audio files, media assets, and static resources with intelligent caching,
//! geo-routing, and edge computing capabilities.

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

/// CDN errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum CdnError {
    /// Upload failed
    #[error("CDN upload failed: {message}")]
    UploadFailed { message: String },

    /// Download failed
    #[error("CDN download failed: {message}")]
    DownloadFailed { message: String },

    /// Invalidation failed
    #[error("Cache invalidation failed: {message}")]
    InvalidationFailed { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Provider error
    #[error("CDN provider error: {provider} - {message}")]
    ProviderError { provider: String, message: String },
}

/// Result type for CDN operations
pub type CdnResult<T> = Result<T, CdnError>;

/// CDN provider types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum CdnProvider {
    /// Amazon `CloudFront`
    CloudFront,
    /// Cloudflare
    Cloudflare,
    /// Akamai
    Akamai,
    /// Fastly
    Fastly,
    /// Azure CDN
    AzureCdn,
    /// Google Cloud CDN
    GoogleCdn,
    /// Custom CDN
    Custom { name: String },
}

/// Geographic region for edge locations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash, Eq)]
#[allow(missing_docs)]
pub enum GeoRegion {
    /// North America
    NorthAmerica,
    /// South America
    SouthAmerica,
    /// Europe
    Europe,
    /// Asia Pacific
    AsiaPacific,
    /// Middle East
    MiddleEast,
    /// Africa
    Africa,
    /// Custom region
    Custom { name: String },
}

/// CDN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    /// CDN provider
    pub provider: CdnProvider,
    /// Distribution domain
    pub distribution_domain: String,
    /// Origin domain
    pub origin_domain: String,
    /// Cache TTL (time to live) in seconds
    pub cache_ttl: u64,
    /// Enable compression
    pub compression_enabled: bool,
    /// Enable HTTPS
    pub https_enabled: bool,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
    /// Geographic restrictions
    pub geo_restrictions: Option<GeoRestrictions>,
    /// Price class / edge locations
    pub price_class: PriceClass,
}

/// Geographic restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoRestrictions {
    /// Restriction type
    pub restriction_type: RestrictionType,
    /// List of country codes
    pub locations: Vec<String>,
}

/// Restriction type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum RestrictionType {
    /// Whitelist - only allow these locations
    Whitelist,
    /// Blacklist - block these locations
    Blacklist,
}

/// Price class for edge location selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum PriceClass {
    /// All edge locations (highest cost)
    All,
    /// Most edge locations (mid-tier cost)
    Standard,
    /// Minimal edge locations (lowest cost)
    Basic,
}

/// Asset metadata for CDN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnAsset {
    /// Asset ID
    pub asset_id: Uuid,
    /// Asset key/path
    pub key: String,
    /// Content type
    pub content_type: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// CDN URL
    pub cdn_url: String,
    /// Origin URL
    pub origin_url: String,
    /// Upload timestamp
    pub uploaded_at: DateTime<Utc>,
    /// Last access timestamp
    pub last_accessed: Option<DateTime<Utc>>,
    /// Access count
    pub access_count: u64,
    /// Cache status
    pub cache_status: CacheStatus,
    /// Metadata tags
    pub tags: HashMap<String, String>,
}

/// Cache status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum CacheStatus {
    /// Cached at edge
    Hit,
    /// Not cached, fetched from origin
    Miss,
    /// Cache expired
    Expired,
    /// Cache invalidated
    Invalidated,
    /// Pending caching
    Pending,
}

/// Cache invalidation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationRequest {
    /// Request ID
    pub request_id: Uuid,
    /// Paths to invalidate
    pub paths: Vec<String>,
    /// Requested timestamp
    pub requested_at: DateTime<Utc>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Status
    pub status: InvalidationStatus,
}

/// Invalidation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InvalidationStatus {
    /// Pending execution
    Pending,
    /// In progress
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed
    Failed { reason: String },
}

/// CDN analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnAnalytics {
    /// Total requests
    pub total_requests: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Bandwidth used in bytes
    pub bandwidth_bytes: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f32,
    /// Requests by region
    pub requests_by_region: HashMap<GeoRegion, u64>,
    /// Top assets
    pub top_assets: Vec<(String, u64)>,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
}

/// Edge location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLocation {
    /// Location ID
    pub location_id: String,
    /// Geographic region
    pub region: GeoRegion,
    /// City
    pub city: String,
    /// Country code
    pub country_code: String,
    /// Active status
    pub active: bool,
    /// Latency in milliseconds
    pub latency_ms: f32,
    /// Load percentage (0.0 to 1.0)
    pub load: f32,
}

/// CDN manager
pub struct CdnManager {
    /// Configuration
    config: Arc<RwLock<CdnConfig>>,
    /// Assets registry
    assets: Arc<RwLock<HashMap<String, CdnAsset>>>,
    /// Invalidation requests
    invalidations: Arc<RwLock<Vec<InvalidationRequest>>>,
    /// Analytics data
    analytics: Arc<RwLock<CdnAnalytics>>,
    /// Edge locations
    edge_locations: Arc<RwLock<Vec<EdgeLocation>>>,
}

impl CdnManager {
    /// Create new CDN manager
    #[must_use]
    pub fn new(config: CdnConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            assets: Arc::new(RwLock::new(HashMap::new())),
            invalidations: Arc::new(RwLock::new(Vec::new())),
            analytics: Arc::new(RwLock::new(Self::default_analytics())),
            edge_locations: Arc::new(RwLock::new(Self::default_edge_locations())),
        }
    }

    /// Upload asset to CDN
    pub async fn upload_asset(
        &self,
        key: String,
        content: Vec<u8>,
        content_type: String,
        tags: HashMap<String, String>,
    ) -> CdnResult<CdnAsset> {
        let config = self.config.read().await;

        // Simulate upload (in real implementation, would use provider SDK)
        let cdn_url = format!("https://{}/{}", config.distribution_domain, key);
        let origin_url = format!("https://{}/{}", config.origin_domain, key);

        let asset = CdnAsset {
            asset_id: Uuid::new_v4(),
            key: key.clone(),
            content_type,
            size_bytes: content.len() as u64,
            cdn_url,
            origin_url,
            uploaded_at: Utc::now(),
            last_accessed: None,
            access_count: 0,
            cache_status: CacheStatus::Pending,
            tags,
        };

        let mut assets = self.assets.write().await;
        assets.insert(key, asset.clone());

        Ok(asset)
    }

    /// Get asset URL from CDN
    pub async fn get_asset_url(&self, key: &str) -> CdnResult<String> {
        let mut assets = self.assets.write().await;
        let asset = assets
            .get_mut(key)
            .ok_or_else(|| CdnError::DownloadFailed {
                message: format!("Asset not found: {key}"),
            })?;

        // Update access stats
        asset.last_accessed = Some(Utc::now());
        asset.access_count += 1;

        // Update analytics
        let mut analytics = self.analytics.write().await;
        analytics.total_requests += 1;

        Ok(asset.cdn_url.clone())
    }

    /// Invalidate cache for specific paths
    pub async fn invalidate_cache(&self, paths: Vec<String>) -> CdnResult<InvalidationRequest> {
        let request = InvalidationRequest {
            request_id: Uuid::new_v4(),
            paths: paths.clone(),
            requested_at: Utc::now(),
            completed_at: None,
            status: InvalidationStatus::Pending,
        };

        // Update asset cache status
        let mut assets = self.assets.write().await;
        for path in &paths {
            if let Some(asset) = assets.get_mut(path) {
                asset.cache_status = CacheStatus::Invalidated;
            }
        }

        let mut invalidations = self.invalidations.write().await;
        invalidations.push(request.clone());

        Ok(request)
    }

    /// Get asset metadata
    pub async fn get_asset(&self, key: &str) -> Option<CdnAsset> {
        let assets = self.assets.read().await;
        assets.get(key).cloned()
    }

    /// List all assets
    pub async fn list_assets(&self) -> Vec<CdnAsset> {
        let assets = self.assets.read().await;
        assets.values().cloned().collect()
    }

    /// Delete asset from CDN
    pub async fn delete_asset(&self, key: &str) -> CdnResult<()> {
        let mut assets = self.assets.write().await;
        assets.remove(key).ok_or_else(|| CdnError::DownloadFailed {
            message: format!("Asset not found: {key}"),
        })?;

        Ok(())
    }

    /// Get CDN analytics
    pub async fn get_analytics(&self) -> CdnAnalytics {
        self.analytics.read().await.clone()
    }

    /// Get edge locations
    pub async fn get_edge_locations(&self) -> Vec<EdgeLocation> {
        self.edge_locations.read().await.clone()
    }

    /// Find nearest edge location
    pub async fn find_nearest_edge(&self, region: GeoRegion) -> Option<EdgeLocation> {
        let locations = self.edge_locations.read().await;
        locations
            .iter()
            .filter(|loc| loc.active && loc.region == region)
            .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap())
            .cloned()
    }

    /// Update cache hit rate
    pub async fn update_cache_hit_rate(&self, is_hit: bool) {
        let mut analytics = self.analytics.write().await;
        let total = analytics.total_requests;

        if total > 0 {
            let hits = (analytics.cache_hit_rate * total as f32) as u64;
            let new_hits = if is_hit { hits + 1 } else { hits };
            analytics.cache_hit_rate = new_hits as f32 / (total + 1) as f32;
        } else {
            analytics.cache_hit_rate = if is_hit { 1.0 } else { 0.0 };
        }
    }

    /// Generate CDN report
    pub async fn generate_report(&self) -> CdnReport {
        let config = self.config.read().await;
        let assets = self.assets.read().await;
        let analytics = self.analytics.read().await;
        let invalidations = self.invalidations.read().await;

        let total_size_bytes: u64 = assets.values().map(|a| a.size_bytes).sum();
        let pending_invalidations = invalidations
            .iter()
            .filter(|inv| {
                matches!(
                    inv.status,
                    InvalidationStatus::Pending | InvalidationStatus::InProgress
                )
            })
            .count();

        CdnReport {
            provider: config.provider.clone(),
            total_assets: assets.len(),
            total_size_bytes,
            cache_hit_rate: analytics.cache_hit_rate,
            total_requests: analytics.total_requests,
            bandwidth_bytes: analytics.bandwidth_bytes,
            avg_response_time_ms: analytics.avg_response_time_ms,
            pending_invalidations,
            active_edge_locations: self.count_active_edges().await,
        }
    }

    /// Purge old assets
    pub async fn purge_old_assets(&self, older_than: ChronoDuration) -> usize {
        let mut assets = self.assets.write().await;
        let cutoff = Utc::now() - older_than;
        let original_count = assets.len();

        assets.retain(|_, asset| {
            let last_access = asset.last_accessed.unwrap_or(asset.uploaded_at);
            last_access > cutoff
        });

        original_count - assets.len()
    }

    async fn count_active_edges(&self) -> usize {
        let edges = self.edge_locations.read().await;
        edges.iter().filter(|e| e.active).count()
    }

    fn default_analytics() -> CdnAnalytics {
        CdnAnalytics {
            total_requests: 0,
            cache_hit_rate: 0.0,
            bandwidth_bytes: 0,
            avg_response_time_ms: 0.0,
            requests_by_region: HashMap::new(),
            top_assets: Vec::new(),
            error_rate: 0.0,
        }
    }

    fn default_edge_locations() -> Vec<EdgeLocation> {
        vec![
            EdgeLocation {
                location_id: "use1".to_string(),
                region: GeoRegion::NorthAmerica,
                city: "Virginia".to_string(),
                country_code: "US".to_string(),
                active: true,
                latency_ms: 10.0,
                load: 0.5,
            },
            EdgeLocation {
                location_id: "euw1".to_string(),
                region: GeoRegion::Europe,
                city: "London".to_string(),
                country_code: "GB".to_string(),
                active: true,
                latency_ms: 15.0,
                load: 0.4,
            },
            EdgeLocation {
                location_id: "apse1".to_string(),
                region: GeoRegion::AsiaPacific,
                city: "Singapore".to_string(),
                country_code: "SG".to_string(),
                active: true,
                latency_ms: 20.0,
                load: 0.6,
            },
        ]
    }
}

/// CDN report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnReport {
    /// CDN provider
    pub provider: CdnProvider,
    /// Total assets
    pub total_assets: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Total requests
    pub total_requests: u64,
    /// Bandwidth used
    pub bandwidth_bytes: u64,
    /// Average response time
    pub avg_response_time_ms: f32,
    /// Pending invalidations
    pub pending_invalidations: usize,
    /// Active edge locations
    pub active_edge_locations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CdnConfig {
        CdnConfig {
            provider: CdnProvider::CloudFront,
            distribution_domain: "d123.cloudfront.net".to_string(),
            origin_domain: "origin.example.com".to_string(),
            cache_ttl: 86400,
            compression_enabled: true,
            https_enabled: true,
            custom_headers: HashMap::new(),
            geo_restrictions: None,
            price_class: PriceClass::Standard,
        }
    }

    #[tokio::test]
    async fn test_upload_asset() {
        let manager = CdnManager::new(create_test_config());

        let asset = manager
            .upload_asset(
                "audio/test.wav".to_string(),
                vec![1, 2, 3, 4, 5],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        assert_eq!(asset.key, "audio/test.wav");
        assert_eq!(asset.content_type, "audio/wav");
        assert_eq!(asset.size_bytes, 5);
        assert!(asset.cdn_url.contains("d123.cloudfront.net"));
    }

    #[tokio::test]
    async fn test_get_asset_url() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test.wav".to_string(),
                vec![1, 2, 3],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let url = manager.get_asset_url("test.wav").await.unwrap();
        assert!(url.contains("test.wav"));

        // Check access count was incremented
        let asset = manager.get_asset("test.wav").await.unwrap();
        assert_eq!(asset.access_count, 1);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test.wav".to_string(),
                vec![1, 2, 3],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let request = manager
            .invalidate_cache(vec!["test.wav".to_string()])
            .await
            .unwrap();

        assert_eq!(request.paths.len(), 1);
        assert_eq!(request.status, InvalidationStatus::Pending);

        // Check asset cache status was updated
        let asset = manager.get_asset("test.wav").await.unwrap();
        assert_eq!(asset.cache_status, CacheStatus::Invalidated);
    }

    #[tokio::test]
    async fn test_delete_asset() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test.wav".to_string(),
                vec![1, 2, 3],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        manager.delete_asset("test.wav").await.unwrap();

        assert!(manager.get_asset("test.wav").await.is_none());
    }

    #[tokio::test]
    async fn test_find_nearest_edge() {
        let manager = CdnManager::new(create_test_config());

        let edge = manager
            .find_nearest_edge(GeoRegion::NorthAmerica)
            .await
            .unwrap();

        assert_eq!(edge.region, GeoRegion::NorthAmerica);
        assert_eq!(edge.country_code, "US");
    }

    #[tokio::test]
    async fn test_analytics() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test1.wav".to_string(),
                vec![1, 2, 3],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        manager.get_asset_url("test1.wav").await.unwrap();

        let analytics = manager.get_analytics().await;
        assert_eq!(analytics.total_requests, 1);
    }

    #[tokio::test]
    async fn test_generate_report() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test.wav".to_string(),
                vec![1, 2, 3, 4, 5],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let report = manager.generate_report().await;
        assert_eq!(report.total_assets, 1);
        assert_eq!(report.total_size_bytes, 5);
        assert!(matches!(report.provider, CdnProvider::CloudFront));
    }

    #[tokio::test]
    async fn test_purge_old_assets() {
        let manager = CdnManager::new(create_test_config());

        manager
            .upload_asset(
                "test.wav".to_string(),
                vec![1, 2, 3],
                "audio/wav".to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();

        // Should not purge assets uploaded just now
        let purged = manager.purge_old_assets(ChronoDuration::days(1)).await;
        assert_eq!(purged, 0);

        // Should purge assets older than 0 seconds
        let purged = manager.purge_old_assets(ChronoDuration::seconds(0)).await;
        assert_eq!(purged, 1);
    }
}
