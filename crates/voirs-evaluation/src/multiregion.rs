// Copyright (c) 2024 VoiRS Contributors
// Licensed under the MIT License

//! Multi-Region Deployment Configuration for Global Distribution
//!
//! This module extends the Kubernetes deployment capabilities to support multi-region
//! deployments with geographic distribution, failover, and data sovereignty compliance.
//!
//! # Features
//!
//! - **Geographic Distribution**: Deploy across multiple cloud regions globally
//! - **Active-Active/Active-Passive**: Flexible deployment patterns for high availability
//! - **Geo-Routing**: Intelligent traffic routing based on user location
//! - **Data Sovereignty**: Compliance with regional data residency requirements
//! - **Cross-Region Failover**: Automatic failover between regions
//! - **Replication Strategy**: Configurable data replication across regions
//! - **Latency Optimization**: Route users to nearest region for best performance
//! - **Cost Optimization**: Balance between performance and infrastructure costs
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │              Global Load Balancer                        │
//! │         (GeoDNS / Traffic Manager)                       │
//! └──────────────────────────────────────────────────────────┘
//!          │                  │                  │
//!          ▼                  ▼                  ▼
//! ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
//! │  US-West-2   │   │  EU-Central  │   │  AP-Southeast│
//! │  (Primary)   │   │  (Secondary) │   │  (Secondary) │
//! ├──────────────┤   ├──────────────┤   ├──────────────┤
//! │ Kubernetes   │   │ Kubernetes   │   │ Kubernetes   │
//! │ Cluster      │←─→│ Cluster      │←─→│ Cluster      │
//! │              │   │              │   │              │
//! │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │
//! │ │ Workers  │ │   │ │ Workers  │ │   │ │ Workers  │ │
//! │ │ (HPA)    │ │   │ │ (HPA)    │ │   │ │ (HPA)    │ │
//! │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │
//! └──────────────┘   └──────────────┘   └──────────────┘
//!       │ Replication   │ Replication   │
//!       └───────────────┴───────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```rust
//! use voirs_evaluation::multiregion::{
//!     MultiRegionConfig, Region, DeploymentPattern, GeoRoutingStrategy
//! };
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure multi-region deployment
//! let config = MultiRegionConfig::builder()
//!     .add_region(Region::UsWest2, true)  // Primary
//!     .add_region(Region::EuCentral1, false)  // Secondary
//!     .add_region(Region::ApSoutheast1, false)  // Secondary
//!     .deployment_pattern(DeploymentPattern::ActiveActive)
//!     .geo_routing_strategy(GeoRoutingStrategy::LatencyBased)
//!     .enable_cross_region_replication(true)
//!     .build()?;
//!
//! println!("Configured {} regions", config.regions.len());
//! println!("Primary region: {:?}", config.get_primary_region());
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Multi-region deployment errors
#[derive(Debug, Error)]
pub enum MultiRegionError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Region error: {0}")]
    RegionError(String),

    #[error("Replication error: {0}")]
    ReplicationError(String),

    #[error("Failover error: {0}")]
    FailoverError(String),

    #[error("No primary region configured")]
    NoPrimaryRegion,
}

pub type Result<T> = std::result::Result<T, MultiRegionError>;

/// Cloud regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    // AWS Regions
    UsEast1,
    UsEast2,
    UsWest1,
    UsWest2,
    CaCentral1,
    EuWest1,
    EuWest2,
    EuCentral1,
    EuNorth1,
    ApSouth1,
    ApNortheast1,
    ApNortheast2,
    ApSoutheast1,
    ApSoutheast2,
    SaEast1,

    // Azure Regions
    EastUs,
    WestUs,
    NorthEurope,
    WestEurope,
    SoutheastAsia,
    EastAsia,

    // GCP Regions
    UsEast4,
    UsWest4,
    EuropeWest1,
    AsiaEast1,
    AsiaSoutheast1,
}

impl Region {
    /// Get region name
    pub fn name(&self) -> &str {
        match self {
            Region::UsEast1 => "us-east-1",
            Region::UsEast2 => "us-east-2",
            Region::UsWest1 => "us-west-1",
            Region::UsWest2 => "us-west-2",
            Region::CaCentral1 => "ca-central-1",
            Region::EuWest1 => "eu-west-1",
            Region::EuWest2 => "eu-west-2",
            Region::EuCentral1 => "eu-central-1",
            Region::EuNorth1 => "eu-north-1",
            Region::ApSouth1 => "ap-south-1",
            Region::ApNortheast1 => "ap-northeast-1",
            Region::ApNortheast2 => "ap-northeast-2",
            Region::ApSoutheast1 => "ap-southeast-1",
            Region::ApSoutheast2 => "ap-southeast-2",
            Region::SaEast1 => "sa-east-1",
            Region::EastUs => "eastus",
            Region::WestUs => "westus",
            Region::NorthEurope => "northeurope",
            Region::WestEurope => "westeurope",
            Region::SoutheastAsia => "southeastasia",
            Region::EastAsia => "eastasia",
            Region::UsEast4 => "us-east4",
            Region::UsWest4 => "us-west4",
            Region::EuropeWest1 => "europe-west1",
            Region::AsiaEast1 => "asia-east1",
            Region::AsiaSoutheast1 => "asia-southeast1",
        }
    }

    /// Get geographic location
    pub fn geographic_location(&self) -> GeographicLocation {
        match self {
            Region::UsEast1
            | Region::UsEast2
            | Region::UsWest1
            | Region::UsWest2
            | Region::EastUs
            | Region::WestUs
            | Region::UsEast4
            | Region::UsWest4 => GeographicLocation::NorthAmerica,
            Region::CaCentral1 => GeographicLocation::NorthAmerica,
            Region::EuWest1
            | Region::EuWest2
            | Region::EuCentral1
            | Region::EuNorth1
            | Region::NorthEurope
            | Region::WestEurope
            | Region::EuropeWest1 => GeographicLocation::Europe,
            Region::ApSouth1 => GeographicLocation::SouthAsia,
            Region::ApNortheast1 | Region::ApNortheast2 => GeographicLocation::EastAsia,
            Region::ApSoutheast1
            | Region::ApSoutheast2
            | Region::SoutheastAsia
            | Region::EastAsia
            | Region::AsiaEast1
            | Region::AsiaSoutheast1 => GeographicLocation::SoutheastAsia,
            Region::SaEast1 => GeographicLocation::SouthAmerica,
        }
    }

    /// Get approximate latency to another region (milliseconds)
    pub fn estimated_latency_to(&self, other: &Region) -> f64 {
        let same_location = self.geographic_location() == other.geographic_location();

        if self == other {
            1.0 // Same region
        } else if same_location {
            20.0 // Same geographic location
        } else {
            // Cross-region latency estimates
            match (self.geographic_location(), other.geographic_location()) {
                (GeographicLocation::NorthAmerica, GeographicLocation::Europe) => 100.0,
                (GeographicLocation::NorthAmerica, GeographicLocation::EastAsia) => 150.0,
                (GeographicLocation::Europe, GeographicLocation::EastAsia) => 200.0,
                _ => 120.0,
            }
        }
    }
}

/// Geographic locations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeographicLocation {
    NorthAmerica,
    SouthAmerica,
    Europe,
    EastAsia,
    SoutheastAsia,
    SouthAsia,
    MiddleEast,
    Africa,
    Oceania,
}

/// Deployment patterns for multi-region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentPattern {
    /// All regions serve traffic simultaneously
    ActiveActive,

    /// One primary region, others serve only on failover
    ActivePassive,

    /// Primary handles writes, secondaries handle reads
    PrimarySecondary,

    /// Traffic distributed based on geographic proximity
    GeoDistributed,
}

/// Geo-routing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeoRoutingStrategy {
    /// Route to geographically nearest region
    Geographic,

    /// Route based on measured latency
    LatencyBased,

    /// Route based on current load
    LoadBalanced,

    /// Route to region with best performance score
    PerformanceBased,

    /// Round-robin across regions
    RoundRobin,
}

/// Replication strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication (strong consistency)
    Synchronous,

    /// Asynchronous replication (eventual consistency)
    Asynchronous,

    /// Semi-synchronous (one sync, others async)
    SemiSynchronous,

    /// No cross-region replication
    None,
}

/// Region configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConfig {
    /// Region identifier
    pub region: Region,

    /// Whether this is the primary region
    pub is_primary: bool,

    /// Region weight for load balancing (0.0 to 1.0)
    pub weight: f64,

    /// Region enabled
    pub enabled: bool,

    /// Kubernetes cluster endpoint
    pub cluster_endpoint: Option<String>,

    /// Maximum capacity (requests per second)
    pub max_capacity: u32,

    /// Current utilization percentage
    pub current_utilization: f64,

    /// Health status
    pub healthy: bool,

    /// Data sovereignty restrictions
    pub data_sovereignty: Vec<String>,
}

impl RegionConfig {
    pub fn new(region: Region, is_primary: bool) -> Self {
        Self {
            region,
            is_primary,
            weight: 1.0,
            enabled: true,
            cluster_endpoint: None,
            max_capacity: 1000,
            current_utilization: 0.0,
            healthy: true,
            data_sovereignty: vec![],
        }
    }
}

/// Multi-region configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRegionConfig {
    /// Regions in deployment
    pub regions: HashMap<Region, RegionConfig>,

    /// Deployment pattern
    pub deployment_pattern: DeploymentPattern,

    /// Geo-routing strategy
    pub geo_routing_strategy: GeoRoutingStrategy,

    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,

    /// Enable automatic failover
    pub auto_failover_enabled: bool,

    /// Failover threshold (health check failures)
    pub failover_threshold: u32,

    /// Enable cross-region replication
    pub cross_region_replication: bool,

    /// Replication lag tolerance (seconds)
    pub max_replication_lag_secs: u64,

    /// Enable data sovereignty enforcement
    pub data_sovereignty_enabled: bool,

    /// Global load balancer endpoint
    pub global_endpoint: Option<String>,
}

impl Default for MultiRegionConfig {
    fn default() -> Self {
        let mut regions = HashMap::new();
        regions.insert(Region::UsWest2, RegionConfig::new(Region::UsWest2, true));

        Self {
            regions,
            deployment_pattern: DeploymentPattern::ActivePassive,
            geo_routing_strategy: GeoRoutingStrategy::LatencyBased,
            replication_strategy: ReplicationStrategy::Asynchronous,
            auto_failover_enabled: true,
            failover_threshold: 3,
            cross_region_replication: true,
            max_replication_lag_secs: 30,
            data_sovereignty_enabled: false,
            global_endpoint: None,
        }
    }
}

impl MultiRegionConfig {
    pub fn builder() -> MultiRegionConfigBuilder {
        MultiRegionConfigBuilder::default()
    }

    /// Get primary region
    pub fn get_primary_region(&self) -> Option<&RegionConfig> {
        self.regions.values().find(|r| r.is_primary)
    }

    /// Get healthy regions
    pub fn get_healthy_regions(&self) -> Vec<&RegionConfig> {
        self.regions
            .values()
            .filter(|r| r.healthy && r.enabled)
            .collect()
    }

    /// Route request to optimal region
    pub fn route_request(&self, client_region: Option<Region>) -> Option<Region> {
        let healthy_regions = self.get_healthy_regions();

        if healthy_regions.is_empty() {
            return None;
        }

        match self.geo_routing_strategy {
            GeoRoutingStrategy::Geographic | GeoRoutingStrategy::LatencyBased => {
                if let Some(client_reg) = client_region {
                    // Find region with lowest latency
                    healthy_regions
                        .iter()
                        .min_by(|a, b| {
                            let latency_a = client_reg.estimated_latency_to(&a.region);
                            let latency_b = client_reg.estimated_latency_to(&b.region);
                            latency_a
                                .partial_cmp(&latency_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|r| r.region)
                } else {
                    // Default to primary
                    self.get_primary_region().map(|r| r.region)
                }
            }
            GeoRoutingStrategy::LoadBalanced => {
                // Route to least utilized region
                healthy_regions
                    .iter()
                    .min_by(|a, b| {
                        a.current_utilization
                            .partial_cmp(&b.current_utilization)
                            .expect("value should be present")
                    })
                    .map(|r| r.region)
            }
            GeoRoutingStrategy::PerformanceBased => {
                // Route to region with best capacity/utilization ratio
                healthy_regions
                    .iter()
                    .max_by(|a, b| {
                        let score_a = a.max_capacity as f64 / (a.current_utilization + 1.0);
                        let score_b = b.max_capacity as f64 / (b.current_utilization + 1.0);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.region)
            }
            GeoRoutingStrategy::RoundRobin => {
                // Simple: return first healthy region
                healthy_regions.first().map(|r| r.region)
            }
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.regions.is_empty() {
            return Err(MultiRegionError::ConfigError(
                "At least one region must be configured".to_string(),
            ));
        }

        let primary_count = self.regions.values().filter(|r| r.is_primary).count();
        if primary_count == 0 {
            return Err(MultiRegionError::NoPrimaryRegion);
        }

        if primary_count > 1 && self.deployment_pattern == DeploymentPattern::ActivePassive {
            return Err(MultiRegionError::ConfigError(
                "Active-Passive pattern requires exactly one primary region".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for multi-region configuration
#[derive(Debug, Default)]
pub struct MultiRegionConfigBuilder {
    regions: HashMap<Region, RegionConfig>,
    deployment_pattern: Option<DeploymentPattern>,
    geo_routing_strategy: Option<GeoRoutingStrategy>,
    replication_strategy: Option<ReplicationStrategy>,
    auto_failover_enabled: bool,
    failover_threshold: u32,
    cross_region_replication: bool,
    max_replication_lag_secs: u64,
    data_sovereignty_enabled: bool,
    global_endpoint: Option<String>,
}

impl MultiRegionConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_region(mut self, region: Region, is_primary: bool) -> Self {
        self.regions
            .insert(region, RegionConfig::new(region, is_primary));
        self
    }

    pub fn deployment_pattern(mut self, pattern: DeploymentPattern) -> Self {
        self.deployment_pattern = Some(pattern);
        self
    }

    pub fn geo_routing_strategy(mut self, strategy: GeoRoutingStrategy) -> Self {
        self.geo_routing_strategy = Some(strategy);
        self
    }

    pub fn replication_strategy(mut self, strategy: ReplicationStrategy) -> Self {
        self.replication_strategy = Some(strategy);
        self
    }

    pub fn enable_auto_failover(mut self, enable: bool) -> Self {
        self.auto_failover_enabled = enable;
        self
    }

    pub fn failover_threshold(mut self, threshold: u32) -> Self {
        self.failover_threshold = threshold;
        self
    }

    pub fn enable_cross_region_replication(mut self, enable: bool) -> Self {
        self.cross_region_replication = enable;
        self
    }

    pub fn max_replication_lag(mut self, seconds: u64) -> Self {
        self.max_replication_lag_secs = seconds;
        self
    }

    pub fn enable_data_sovereignty(mut self, enable: bool) -> Self {
        self.data_sovereignty_enabled = enable;
        self
    }

    pub fn global_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.global_endpoint = Some(endpoint.into());
        self
    }

    pub fn build(self) -> Result<MultiRegionConfig> {
        let config = MultiRegionConfig {
            regions: self.regions,
            deployment_pattern: self
                .deployment_pattern
                .unwrap_or(DeploymentPattern::ActivePassive),
            geo_routing_strategy: self
                .geo_routing_strategy
                .unwrap_or(GeoRoutingStrategy::LatencyBased),
            replication_strategy: self
                .replication_strategy
                .unwrap_or(ReplicationStrategy::Asynchronous),
            auto_failover_enabled: self.auto_failover_enabled,
            failover_threshold: self.failover_threshold,
            cross_region_replication: self.cross_region_replication,
            max_replication_lag_secs: self.max_replication_lag_secs,
            data_sovereignty_enabled: self.data_sovereignty_enabled,
            global_endpoint: self.global_endpoint,
        };

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_names() {
        assert_eq!(Region::UsWest2.name(), "us-west-2");
        assert_eq!(Region::EuCentral1.name(), "eu-central-1");
        assert_eq!(Region::ApSoutheast1.name(), "ap-southeast-1");
    }

    #[test]
    fn test_geographic_location() {
        assert_eq!(
            Region::UsWest2.geographic_location(),
            GeographicLocation::NorthAmerica
        );
        assert_eq!(
            Region::EuCentral1.geographic_location(),
            GeographicLocation::Europe
        );
        assert_eq!(
            Region::ApSoutheast1.geographic_location(),
            GeographicLocation::SoutheastAsia
        );
    }

    #[test]
    fn test_estimated_latency() {
        assert_eq!(Region::UsWest2.estimated_latency_to(&Region::UsWest2), 1.0);
        assert_eq!(Region::UsWest2.estimated_latency_to(&Region::UsEast1), 20.0);
        assert_eq!(
            Region::UsWest2.estimated_latency_to(&Region::EuCentral1),
            100.0
        );
    }

    #[test]
    fn test_default_config() {
        let config = MultiRegionConfig::default();
        assert_eq!(config.regions.len(), 1);
        assert!(config.auto_failover_enabled);
        assert!(config.cross_region_replication);
    }

    #[test]
    fn test_builder() {
        let config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, false)
            .deployment_pattern(DeploymentPattern::ActiveActive)
            .geo_routing_strategy(GeoRoutingStrategy::LatencyBased)
            .build()
            .unwrap();

        assert_eq!(config.regions.len(), 2);
        assert_eq!(config.deployment_pattern, DeploymentPattern::ActiveActive);
        assert_eq!(
            config.geo_routing_strategy,
            GeoRoutingStrategy::LatencyBased
        );
    }

    #[test]
    fn test_get_primary_region() {
        let config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, false)
            .build()
            .unwrap();

        let primary = config.get_primary_region().unwrap();
        assert_eq!(primary.region, Region::UsWest2);
        assert!(primary.is_primary);
    }

    #[test]
    fn test_no_primary_region_error() {
        let result = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, false)
            .add_region(Region::EuCentral1, false)
            .build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MultiRegionError::NoPrimaryRegion
        ));
    }

    #[test]
    fn test_route_request_latency_based() {
        let config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, false)
            .add_region(Region::ApSoutheast1, false)
            .geo_routing_strategy(GeoRoutingStrategy::LatencyBased)
            .build()
            .unwrap();

        // Client in US should route to UsWest2
        let routed = config.route_request(Some(Region::UsEast1));
        assert_eq!(routed, Some(Region::UsWest2));

        // Client in Europe should route to EuCentral1
        let routed = config.route_request(Some(Region::EuWest1));
        assert_eq!(routed, Some(Region::EuCentral1));
    }

    #[test]
    fn test_healthy_regions_filter() {
        let mut config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, false)
            .build()
            .unwrap();

        // Mark one region unhealthy
        config.regions.get_mut(&Region::EuCentral1).unwrap().healthy = false;

        let healthy = config.get_healthy_regions();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].region, Region::UsWest2);
    }

    #[test]
    fn test_validation() {
        let config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multiple_primaries_active_passive_error() {
        let result = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, true)
            .deployment_pattern(DeploymentPattern::ActivePassive)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_active_active_allows_multiple_primaries() {
        let config = MultiRegionConfig::builder()
            .add_region(Region::UsWest2, true)
            .add_region(Region::EuCentral1, true)
            .deployment_pattern(DeploymentPattern::ActiveActive)
            .build()
            .unwrap();

        let primaries: Vec<_> = config.regions.values().filter(|r| r.is_primary).collect();

        assert_eq!(primaries.len(), 2);
    }
}
