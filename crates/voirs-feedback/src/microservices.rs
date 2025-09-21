//! Microservices architecture framework for VoiRS feedback system
//!
//! This module provides a framework for organizing the system into independent,
//! scalable microservices with service discovery, health monitoring, and
//! inter-service communication.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Microservice errors
#[derive(Error, Debug)]
pub enum MicroserviceError {
    /// Service not found
    #[error("Service {service_name} not found")]
    ServiceNotFound {
        /// Service name
        service_name: String,
    },

    /// Service unavailable
    #[error("Service {service_name} is unavailable")]
    ServiceUnavailable {
        /// Service name
        service_name: String,
    },

    /// Communication error
    #[error("Communication error: {message}")]
    CommunicationError {
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },

    /// Service startup error
    #[error("Service startup failed: {message}")]
    StartupError {
        /// Error message
        message: String,
    },
}

/// Result type for microservice operations
pub type MicroserviceResult<T> = Result<T, MicroserviceError>;

/// Service type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    /// Gateway service (API gateway, routing)
    Gateway,
    /// Authentication and authorization service
    Auth,
    /// User management service
    UserManagement,
    /// Feedback processing service
    FeedbackProcessing,
    /// Real-time audio processing service
    AudioProcessing,
    /// Analytics and reporting service
    Analytics,
    /// Notification service
    Notification,
    /// File storage service
    Storage,
    /// Configuration management service
    Configuration,
    /// Monitoring and logging service
    Monitoring,
    /// Custom service type
    Custom(String),
}

/// Service status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServiceStatus {
    /// Service is starting up
    Starting,
    /// Service is healthy and ready
    Healthy,
    /// Service has degraded performance
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Service is shutting down
    Stopping,
    /// Service is stopped
    Stopped,
}

/// Service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    /// Service unique identifier
    pub service_id: Uuid,
    /// Service name
    pub service_name: String,
    /// Service type
    pub service_type: ServiceType,
    /// Service version
    pub version: String,
    /// Service endpoints
    pub endpoints: Vec<ServiceEndpoint>,
    /// Service status
    pub status: ServiceStatus,
    /// Service health metrics
    pub health: ServiceHealth,
    /// Service configuration
    pub config: ServiceConfig,
    /// Service metadata
    pub metadata: HashMap<String, String>,
    /// Service registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
}

/// Service endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Endpoint name
    pub name: String,
    /// Protocol (HTTP, gRPC, WebSocket, etc.)
    pub protocol: String,
    /// Host address
    pub host: String,
    /// Port number
    pub port: u16,
    /// Endpoint path
    pub path: String,
    /// Whether endpoint uses TLS
    pub secure: bool,
    /// Endpoint tags
    pub tags: Vec<String>,
}

/// Service health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Disk utilization (0.0 to 1.0)
    pub disk_utilization: f32,
    /// Response time in milliseconds
    pub response_time_ms: f32,
    /// Request rate (requests per second)
    pub request_rate: f32,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Active connections count
    pub active_connections: u32,
    /// Health check score (0.0 to 1.0)
    pub health_score: f32,
}

/// Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Maximum memory limit in bytes
    pub max_memory_bytes: u64,
    /// Maximum CPU cores
    pub max_cpu_cores: f32,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum concurrent requests
    pub max_concurrent_requests: u32,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Feature flags
    pub features: HashMap<String, bool>,
}

/// Service discovery interface
#[async_trait]
pub trait ServiceDiscovery: Send + Sync {
    /// Register a service
    async fn register_service(&self, service: ServiceInfo) -> MicroserviceResult<()>;

    /// Deregister a service
    async fn deregister_service(&self, service_id: Uuid) -> MicroserviceResult<()>;

    /// Discover services by type
    async fn discover_services(
        &self,
        service_type: ServiceType,
    ) -> MicroserviceResult<Vec<ServiceInfo>>;

    /// Find a specific service by name
    async fn find_service(&self, service_name: &str) -> MicroserviceResult<Option<ServiceInfo>>;

    /// Update service health
    async fn update_service_health(
        &self,
        service_id: Uuid,
        health: ServiceHealth,
    ) -> MicroserviceResult<()>;

    /// Get all registered services
    async fn get_all_services(&self) -> MicroserviceResult<Vec<ServiceInfo>>;
}

/// Service communication interface
#[async_trait]
pub trait ServiceCommunication: Send + Sync {
    /// Send a request to another service
    async fn send_request(
        &self,
        target_service: &str,
        endpoint: &str,
        payload: Vec<u8>,
    ) -> MicroserviceResult<Vec<u8>>;

    /// Send a broadcast message to all services of a type
    async fn broadcast_message(
        &self,
        service_type: ServiceType,
        message: ServiceMessage,
    ) -> MicroserviceResult<()>;

    /// Subscribe to messages from other services
    async fn subscribe_to_messages(
        &self,
        message_types: Vec<String>,
    ) -> MicroserviceResult<broadcast::Receiver<ServiceMessage>>;

    /// Publish a message to subscribers
    async fn publish_message(&self, message: ServiceMessage) -> MicroserviceResult<()>;
}

/// Inter-service message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMessage {
    /// Message ID
    pub message_id: Uuid,
    /// Source service ID
    pub source_service_id: Uuid,
    /// Message type
    pub message_type: String,
    /// Message payload
    pub payload: Vec<u8>,
    /// Message metadata
    pub metadata: HashMap<String, String>,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
    /// Message TTL in seconds
    pub ttl_seconds: Option<u64>,
}

/// Service registry implementation
pub struct ServiceRegistry {
    /// Registered services
    services: Arc<RwLock<HashMap<Uuid, ServiceInfo>>>,
    /// Service type index
    type_index: Arc<RwLock<HashMap<ServiceType, Vec<Uuid>>>>,
    /// Name index
    name_index: Arc<RwLock<HashMap<String, Uuid>>>,
    /// Health check interval
    health_check_interval: Duration,
}

impl ServiceRegistry {
    /// Create a new service registry
    pub fn new(health_check_interval: Duration) -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
            name_index: Arc::new(RwLock::new(HashMap::new())),
            health_check_interval,
        }
    }

    /// Start health checking background task
    pub async fn start_health_monitoring(&self) -> MicroserviceResult<()> {
        let services = self.services.clone();
        let health_check_interval = self.health_check_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_check_interval);

            loop {
                interval.tick().await;

                let mut services_guard = services.write().await;
                let mut unhealthy_services = Vec::new();

                for (service_id, service) in services_guard.iter_mut() {
                    let time_since_heartbeat = Utc::now()
                        .signed_duration_since(service.last_heartbeat)
                        .to_std()
                        .unwrap_or(Duration::ZERO);

                    if time_since_heartbeat > health_check_interval * 3 {
                        service.status = ServiceStatus::Unhealthy;
                        unhealthy_services.push(*service_id);
                    }
                }

                // Remove unhealthy services after a grace period
                for service_id in unhealthy_services {
                    if let Some(service) = services_guard.get(&service_id) {
                        let time_since_heartbeat = Utc::now()
                            .signed_duration_since(service.last_heartbeat)
                            .to_std()
                            .unwrap_or(Duration::ZERO);

                        if time_since_heartbeat > health_check_interval * 10 {
                            log::warn!("Removing unhealthy service: {}", service.service_name);
                            services_guard.remove(&service_id);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Get healthy services of a specific type
    pub async fn get_healthy_services(
        &self,
        service_type: ServiceType,
    ) -> MicroserviceResult<Vec<ServiceInfo>> {
        let services = self.services.read().await;
        let type_index = self.type_index.read().await;

        let service_ids = type_index.get(&service_type).cloned().unwrap_or_default();
        let mut healthy_services = Vec::new();

        for service_id in service_ids {
            if let Some(service) = services.get(&service_id) {
                if service.status == ServiceStatus::Healthy {
                    healthy_services.push(service.clone());
                }
            }
        }

        Ok(healthy_services)
    }
}

#[async_trait]
impl ServiceDiscovery for ServiceRegistry {
    async fn register_service(&self, mut service: ServiceInfo) -> MicroserviceResult<()> {
        service.registered_at = Utc::now();
        service.last_heartbeat = Utc::now();

        let service_id = service.service_id;
        let service_type = service.service_type.clone();
        let service_name = service.service_name.clone();

        // Update services
        self.services.write().await.insert(service_id, service);

        // Update type index
        let mut type_index = self.type_index.write().await;
        type_index
            .entry(service_type)
            .or_insert_with(Vec::new)
            .push(service_id);

        // Update name index
        self.name_index
            .write()
            .await
            .insert(service_name, service_id);

        log::info!("Service registered: {}", service_id);
        Ok(())
    }

    async fn deregister_service(&self, service_id: Uuid) -> MicroserviceResult<()> {
        let mut services = self.services.write().await;

        if let Some(service) = services.remove(&service_id) {
            // Remove from type index
            let mut type_index = self.type_index.write().await;
            if let Some(service_list) = type_index.get_mut(&service.service_type) {
                service_list.retain(|&id| id != service_id);
                if service_list.is_empty() {
                    type_index.remove(&service.service_type);
                }
            }

            // Remove from name index
            self.name_index.write().await.remove(&service.service_name);

            log::info!("Service deregistered: {}", service_id);
        }

        Ok(())
    }

    async fn discover_services(
        &self,
        service_type: ServiceType,
    ) -> MicroserviceResult<Vec<ServiceInfo>> {
        let services = self.services.read().await;
        let type_index = self.type_index.read().await;

        let service_ids = type_index.get(&service_type).cloned().unwrap_or_default();
        let mut discovered_services = Vec::new();

        for service_id in service_ids {
            if let Some(service) = services.get(&service_id) {
                discovered_services.push(service.clone());
            }
        }

        Ok(discovered_services)
    }

    async fn find_service(&self, service_name: &str) -> MicroserviceResult<Option<ServiceInfo>> {
        let services = self.services.read().await;
        let name_index = self.name_index.read().await;

        if let Some(&service_id) = name_index.get(service_name) {
            Ok(services.get(&service_id).cloned())
        } else {
            Ok(None)
        }
    }

    async fn update_service_health(
        &self,
        service_id: Uuid,
        health: ServiceHealth,
    ) -> MicroserviceResult<()> {
        let mut services = self.services.write().await;

        if let Some(service) = services.get_mut(&service_id) {
            service.health = health;
            service.last_heartbeat = Utc::now();

            // Update status based on health score
            service.status = if service.health.health_score >= 0.8 {
                ServiceStatus::Healthy
            } else if service.health.health_score >= 0.5 {
                ServiceStatus::Degraded
            } else {
                ServiceStatus::Unhealthy
            };
        }

        Ok(())
    }

    async fn get_all_services(&self) -> MicroserviceResult<Vec<ServiceInfo>> {
        let services = self.services.read().await;
        Ok(services.values().cloned().collect())
    }
}

/// Service communication manager
pub struct ServiceCommunicationManager {
    /// Service discovery reference
    service_discovery: Arc<dyn ServiceDiscovery>,
    /// Message broadcaster
    message_broadcaster: broadcast::Sender<ServiceMessage>,
    /// HTTP client for service-to-service communication
    http_client: reqwest::Client,
}

impl ServiceCommunicationManager {
    /// Create a new service communication manager
    pub fn new(service_discovery: Arc<dyn ServiceDiscovery>) -> Self {
        let (tx, _) = broadcast::channel(1000); // Buffer up to 1000 messages

        Self {
            service_discovery,
            message_broadcaster: tx,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ServiceCommunication for ServiceCommunicationManager {
    async fn send_request(
        &self,
        target_service: &str,
        endpoint: &str,
        payload: Vec<u8>,
    ) -> MicroserviceResult<Vec<u8>> {
        // Find the target service
        let service = self
            .service_discovery
            .find_service(target_service)
            .await?
            .ok_or_else(|| MicroserviceError::ServiceNotFound {
                service_name: target_service.to_string(),
            })?;

        if service.status != ServiceStatus::Healthy {
            return Err(MicroserviceError::ServiceUnavailable {
                service_name: target_service.to_string(),
            });
        }

        // Find HTTP endpoint
        let http_endpoint = service
            .endpoints
            .iter()
            .find(|ep| ep.protocol == "HTTP" || ep.protocol == "HTTPS")
            .ok_or_else(|| MicroserviceError::CommunicationError {
                message: "No HTTP endpoint found for service".to_string(),
            })?;

        // Build URL
        let scheme = if http_endpoint.secure {
            "https"
        } else {
            "http"
        };
        let url = format!(
            "{}://{}:{}/{}",
            scheme,
            http_endpoint.host,
            http_endpoint.port,
            endpoint.trim_start_matches('/')
        );

        // Send request
        let response = self
            .http_client
            .post(&url)
            .body(payload)
            .send()
            .await
            .map_err(|e| MicroserviceError::CommunicationError {
                message: format!("HTTP request failed: {}", e),
            })?;

        let response_bytes =
            response
                .bytes()
                .await
                .map_err(|e| MicroserviceError::CommunicationError {
                    message: format!("Failed to read response: {}", e),
                })?;

        Ok(response_bytes.to_vec())
    }

    async fn broadcast_message(
        &self,
        service_type: ServiceType,
        message: ServiceMessage,
    ) -> MicroserviceResult<()> {
        let services = self
            .service_discovery
            .discover_services(service_type)
            .await?;

        for service in services {
            if service.status == ServiceStatus::Healthy {
                // Send message to each healthy service
                if let Err(e) = self.message_broadcaster.send(message.clone()) {
                    log::warn!(
                        "Failed to broadcast message to {}: {}",
                        service.service_name,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    async fn subscribe_to_messages(
        &self,
        _message_types: Vec<String>,
    ) -> MicroserviceResult<broadcast::Receiver<ServiceMessage>> {
        Ok(self.message_broadcaster.subscribe())
    }

    async fn publish_message(&self, message: ServiceMessage) -> MicroserviceResult<()> {
        self.message_broadcaster.send(message).map_err(|e| {
            MicroserviceError::CommunicationError {
                message: format!("Failed to publish message: {}", e),
            }
        })?;

        Ok(())
    }
}

/// Microservice management framework
pub struct MicroserviceFramework {
    /// Service registry
    pub service_registry: Arc<ServiceRegistry>,
    /// Service communication manager
    pub communication_manager: Arc<ServiceCommunicationManager>,
    /// Framework configuration
    config: MicroserviceConfig,
}

/// Microservice framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroserviceConfig {
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Service timeout in seconds
    pub service_timeout_seconds: u64,
    /// Maximum retries for service calls
    pub max_retries: u32,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

/// Load balancing strategies for microservices
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Random selection
    Random,
    /// Least connections
    LeastConnections,
    /// Health-based selection
    HealthBased,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: u32,
    /// Time window for failure counting in seconds
    pub time_window_seconds: u64,
    /// Recovery timeout in seconds
    pub recovery_timeout_seconds: u64,
    /// Half-open request count
    pub half_open_requests: u32,
}

impl Default for MicroserviceConfig {
    fn default() -> Self {
        Self {
            health_check_interval_seconds: 30,
            service_timeout_seconds: 30,
            max_retries: 3,
            load_balancing: LoadBalancingStrategy::HealthBased,
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 5,
                time_window_seconds: 60,
                recovery_timeout_seconds: 60,
                half_open_requests: 3,
            },
        }
    }
}

impl MicroserviceFramework {
    /// Create a new microservice framework
    pub async fn new(config: MicroserviceConfig) -> MicroserviceResult<Self> {
        let health_check_interval = Duration::from_secs(config.health_check_interval_seconds);
        let service_registry = Arc::new(ServiceRegistry::new(health_check_interval));

        // Start health monitoring
        service_registry.start_health_monitoring().await?;

        let communication_manager = Arc::new(ServiceCommunicationManager::new(
            service_registry.clone() as Arc<dyn ServiceDiscovery>,
        ));

        Ok(Self {
            service_registry,
            communication_manager,
            config,
        })
    }

    /// Create a new service instance
    pub fn create_service(
        &self,
        service_name: String,
        service_type: ServiceType,
        version: String,
        endpoints: Vec<ServiceEndpoint>,
        config: ServiceConfig,
    ) -> ServiceInfo {
        ServiceInfo {
            service_id: Uuid::new_v4(),
            service_name,
            service_type,
            version,
            endpoints,
            status: ServiceStatus::Starting,
            health: ServiceHealth {
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                disk_utilization: 0.0,
                response_time_ms: 0.0,
                request_rate: 0.0,
                error_rate: 0.0,
                active_connections: 0,
                health_score: 1.0,
            },
            config,
            metadata: HashMap::new(),
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
        }
    }

    /// Get framework statistics
    pub async fn get_framework_stats(&self) -> MicroserviceResult<FrameworkStats> {
        let all_services = self.service_registry.get_all_services().await?;

        let mut service_counts = HashMap::new();
        let mut total_healthy = 0;
        let mut total_unhealthy = 0;

        for service in &all_services {
            let count = service_counts
                .entry(service.service_type.clone())
                .or_insert(0);
            *count += 1;

            match service.status {
                ServiceStatus::Healthy => total_healthy += 1,
                ServiceStatus::Unhealthy => total_unhealthy += 1,
                _ => {}
            }
        }

        Ok(FrameworkStats {
            total_services: all_services.len(),
            healthy_services: total_healthy,
            unhealthy_services: total_unhealthy,
            service_counts,
            framework_uptime: Duration::from_secs(0), // Would track actual uptime
        })
    }
}

/// Framework statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkStats {
    /// Total number of registered services
    pub total_services: usize,
    /// Number of healthy services
    pub healthy_services: usize,
    /// Number of unhealthy services
    pub unhealthy_services: usize,
    /// Service count by type
    pub service_counts: HashMap<ServiceType, usize>,
    /// Framework uptime
    pub framework_uptime: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registry_creation() {
        let registry = ServiceRegistry::new(Duration::from_secs(30));
        let services = registry.get_all_services().await.unwrap();
        assert_eq!(services.len(), 0);
    }

    #[tokio::test]
    async fn test_service_registration() {
        let registry = ServiceRegistry::new(Duration::from_secs(30));

        let service = ServiceInfo {
            service_id: Uuid::new_v4(),
            service_name: "test-service".to_string(),
            service_type: ServiceType::FeedbackProcessing,
            version: "1.0.0".to_string(),
            endpoints: vec![],
            status: ServiceStatus::Healthy,
            health: ServiceHealth {
                cpu_utilization: 0.5,
                memory_utilization: 0.6,
                disk_utilization: 0.3,
                response_time_ms: 50.0,
                request_rate: 100.0,
                error_rate: 0.01,
                active_connections: 10,
                health_score: 0.9,
            },
            config: ServiceConfig {
                max_memory_bytes: 1_000_000_000,
                max_cpu_cores: 2.0,
                health_check_interval_seconds: 30,
                request_timeout_seconds: 30,
                max_concurrent_requests: 100,
                environment: HashMap::new(),
                features: HashMap::new(),
            },
            metadata: HashMap::new(),
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
        };

        registry.register_service(service.clone()).await.unwrap();

        let found_service = registry.find_service("test-service").await.unwrap();
        assert!(found_service.is_some());
        assert_eq!(found_service.unwrap().service_id, service.service_id);
    }

    #[tokio::test]
    async fn test_service_discovery_by_type() {
        let registry = ServiceRegistry::new(Duration::from_secs(30));

        let service1 = ServiceInfo {
            service_id: Uuid::new_v4(),
            service_name: "feedback-service-1".to_string(),
            service_type: ServiceType::FeedbackProcessing,
            version: "1.0.0".to_string(),
            endpoints: vec![],
            status: ServiceStatus::Healthy,
            health: ServiceHealth {
                cpu_utilization: 0.5,
                memory_utilization: 0.6,
                disk_utilization: 0.3,
                response_time_ms: 50.0,
                request_rate: 100.0,
                error_rate: 0.01,
                active_connections: 10,
                health_score: 0.9,
            },
            config: ServiceConfig {
                max_memory_bytes: 1_000_000_000,
                max_cpu_cores: 2.0,
                health_check_interval_seconds: 30,
                request_timeout_seconds: 30,
                max_concurrent_requests: 100,
                environment: HashMap::new(),
                features: HashMap::new(),
            },
            metadata: HashMap::new(),
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
        };

        let service2 = ServiceInfo {
            service_id: Uuid::new_v4(),
            service_name: "audio-service-1".to_string(),
            service_type: ServiceType::AudioProcessing,
            version: "1.0.0".to_string(),
            endpoints: vec![],
            status: ServiceStatus::Healthy,
            health: ServiceHealth {
                cpu_utilization: 0.4,
                memory_utilization: 0.5,
                disk_utilization: 0.2,
                response_time_ms: 30.0,
                request_rate: 200.0,
                error_rate: 0.005,
                active_connections: 20,
                health_score: 0.95,
            },
            config: ServiceConfig {
                max_memory_bytes: 2_000_000_000,
                max_cpu_cores: 4.0,
                health_check_interval_seconds: 30,
                request_timeout_seconds: 30,
                max_concurrent_requests: 200,
                environment: HashMap::new(),
                features: HashMap::new(),
            },
            metadata: HashMap::new(),
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
        };

        registry.register_service(service1).await.unwrap();
        registry.register_service(service2).await.unwrap();

        let feedback_services = registry
            .discover_services(ServiceType::FeedbackProcessing)
            .await
            .unwrap();
        assert_eq!(feedback_services.len(), 1);
        assert_eq!(feedback_services[0].service_name, "feedback-service-1");

        let audio_services = registry
            .discover_services(ServiceType::AudioProcessing)
            .await
            .unwrap();
        assert_eq!(audio_services.len(), 1);
        assert_eq!(audio_services[0].service_name, "audio-service-1");
    }

    #[tokio::test]
    async fn test_microservice_framework_creation() {
        let config = MicroserviceConfig::default();
        let framework = MicroserviceFramework::new(config).await.unwrap();

        let stats = framework.get_framework_stats().await.unwrap();
        assert_eq!(stats.total_services, 0);
        assert_eq!(stats.healthy_services, 0);
    }

    #[test]
    fn test_service_endpoint_creation() {
        let endpoint = ServiceEndpoint {
            name: "api".to_string(),
            protocol: "HTTP".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            path: "/api/v1".to_string(),
            secure: false,
            tags: vec!["public".to_string()],
        };

        assert_eq!(endpoint.name, "api");
        assert_eq!(endpoint.port, 8080);
        assert!(!endpoint.secure);
    }
}
