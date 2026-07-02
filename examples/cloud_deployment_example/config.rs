/// Comprehensive Cloud Deployment Example for VoiRS
///
/// This example demonstrates enterprise-scale cloud deployment of VoiRS
/// across multiple cloud platforms including AWS, Azure, Google Cloud,
/// and Kubernetes orchestration for handling millions of synthesis requests.
///
/// Features Demonstrated:
/// - AWS deployment with ECS, Lambda, and S3 integration
/// - Azure deployment with Container Instances and Blob Storage
/// - Google Cloud deployment with Cloud Run and Cloud Storage
/// - Kubernetes orchestration with auto-scaling
/// - Load balancing and traffic distribution
/// - Multi-region deployment for global availability
/// - CDN integration for audio delivery
/// - Monitoring, logging, and observability
/// - Cost optimization and resource management
/// - Disaster recovery and high availability

#[derive(Debug, Clone)]
pub struct CloudDeploymentConfig {
    pub cloud_provider: CloudProvider,
    pub deployment_model: DeploymentModel,
    pub scaling_config: ScalingConfig,
    pub storage_config: StorageConfig,
    pub network_config: NetworkConfig,
    pub monitoring_config: MonitoringConfig,
    pub security_config: SecurityConfig,
    pub cost_optimization: CostOptimizationConfig,
}

#[derive(Debug, Clone, Copy)]
pub enum CloudProvider {
    AWS,
    Azure,
    GoogleCloud,
    Kubernetes,
    MultiCloud,
}

#[derive(Debug, Clone)]
pub enum DeploymentModel {
    Serverless {
        function_memory_mb: u32,
        timeout_seconds: u32,
        concurrent_executions: u32,
    },
    Containers {
        cpu_cores: f32,
        memory_gb: u32,
        replicas: u32,
    },
    VirtualMachines {
        instance_type: String,
        instance_count: u32,
    },
    Hybrid {
        serverless_ratio: f32,
        container_ratio: f32,
    },
}

#[derive(Debug, Clone)]
pub struct ScalingConfig {
    pub min_instances: u32,
    pub max_instances: u32,
    pub target_cpu_utilization: f32,
    pub target_memory_utilization: f32,
    pub scale_up_cooldown_seconds: u32,
    pub scale_down_cooldown_seconds: u32,
    pub requests_per_second_threshold: u32,
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub audio_storage: AudioStorageConfig,
    pub model_storage: ModelStorageConfig,
    pub cache_storage: CacheStorageConfig,
    pub backup_config: BackupConfig,
}

#[derive(Debug, Clone)]
pub struct AudioStorageConfig {
    pub storage_type: StorageType,
    pub cdn_enabled: bool,
    pub compression_enabled: bool,
    pub retention_days: u32,
}

#[derive(Debug, Clone)]
pub struct ModelStorageConfig {
    pub storage_type: StorageType,
    pub versioning_enabled: bool,
    pub encryption_at_rest: bool,
    pub global_replication: bool,
}

#[derive(Debug, Clone)]
pub struct CacheStorageConfig {
    pub cache_type: CacheType,
    pub cache_size_gb: u32,
    pub ttl_seconds: u32,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageType {
    S3,           // AWS S3
    BlobStorage,  // Azure Blob Storage
    CloudStorage, // Google Cloud Storage
    MinIO,        // Self-hosted
    Distributed,  // Multi-cloud
}

#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    Redis,
    ElastiCache,
    MemoryStore,
    InMemory,
}

#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
}

#[derive(Debug, Clone)]
pub struct BackupConfig {
    pub enabled: bool,
    pub frequency_hours: u32,
    pub retention_days: u32,
    pub cross_region_backup: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub load_balancer: LoadBalancerConfig,
    pub cdn_config: CDNConfig,
    pub ssl_config: SSLConfig,
    pub regions: Vec<CloudRegion>,
}

#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check_interval_seconds: u32,
    pub unhealthy_threshold: u32,
    pub timeout_seconds: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    IPHash,
    GeographicProximity,
}

#[derive(Debug, Clone)]
pub struct CDNConfig {
    pub enabled: bool,
    pub cache_ttl_seconds: u32,
    pub edge_locations: Vec<String>,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct SSLConfig {
    pub enabled: bool,
    pub certificate_source: CertificateSource,
    pub min_tls_version: String,
}

#[derive(Debug, Clone, Copy)]
pub enum CertificateSource {
    LetsEncrypt,
    CloudProvider,
    Custom,
}

#[derive(Debug, Clone)]
pub struct CloudRegion {
    pub region_id: String,
    pub primary: bool,
    pub traffic_ratio: f32,
    pub disaster_recovery: bool,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub logging_level: LogLevel,
    pub alerting_config: AlertingConfig,
    pub observability_tools: Vec<ObservabilityTool>,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct AlertingConfig {
    pub email_alerts: bool,
    pub slack_webhook: Option<String>,
    pub pagerduty_enabled: bool,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub error_rate_percent: f32,
    pub response_time_ms: u32,
    pub cpu_utilization_percent: f32,
    pub memory_utilization_percent: f32,
    pub disk_usage_percent: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum ObservabilityTool {
    Prometheus,
    Grafana,
    CloudWatch,
    DataDog,
    NewRelic,
    Jaeger,
}

#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub authentication: AuthenticationConfig,
    pub authorization: AuthorizationConfig,
    pub encryption: EncryptionConfig,
    pub network_security: NetworkSecurityConfig,
}

#[derive(Debug, Clone)]
pub struct AuthenticationConfig {
    pub method: AuthenticationMethod,
    pub api_keys_enabled: bool,
    pub jwt_enabled: bool,
    pub oauth2_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum AuthenticationMethod {
    APIKey,
    JWT,
    OAuth2,
    IAM,
    SAML,
}

#[derive(Debug, Clone)]
pub struct AuthorizationConfig {
    pub rbac_enabled: bool,
    pub rate_limiting: RateLimitingConfig,
    pub ip_whitelist: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RateLimitingConfig {
    pub requests_per_minute: u32,
    pub burst_capacity: u32,
    pub per_user_limit: u32,
}

#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
    pub key_management: KeyManagementConfig,
}

#[derive(Debug, Clone)]
pub struct KeyManagementConfig {
    pub service: KeyManagementService,
    pub key_rotation_days: u32,
    pub hsm_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum KeyManagementService {
    CloudKMS,
    AWSKms,
    AzureKeyVault,
    HashiCorpVault,
    Custom,
}

#[derive(Debug, Clone)]
pub struct NetworkSecurityConfig {
    pub vpc_enabled: bool,
    pub firewall_rules: Vec<FirewallRule>,
    pub ddos_protection: bool,
    pub waf_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct FirewallRule {
    pub name: String,
    pub direction: TrafficDirection,
    pub protocol: String,
    pub port_range: String,
    pub source_cidrs: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum TrafficDirection {
    Ingress,
    Egress,
}

#[derive(Debug, Clone)]
pub struct CostOptimizationConfig {
    pub spot_instances_enabled: bool,
    pub reserved_instances_ratio: f32,
    pub auto_scaling_aggressive: bool,
    pub storage_lifecycle_policies: Vec<StorageLifecyclePolicy>,
    pub cost_alerts_enabled: bool,
    pub budget_limit_usd: f32,
}

#[derive(Debug, Clone)]
pub struct StorageLifecyclePolicy {
    pub name: String,
    pub transition_days: u32,
    pub storage_class: StorageClass,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageClass {
    Standard,
    InfrequentAccess,
    Archive,
    DeepArchive,
}
