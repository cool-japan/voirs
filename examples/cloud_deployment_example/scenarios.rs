use crate::config::*;

/// Create example deployment configurations for different scenarios
pub fn create_deployment_scenarios() -> Vec<(String, CloudDeploymentConfig)> {
    vec![
        // Startup/MVP scenario
        (
            "startup_mvp".to_string(),
            CloudDeploymentConfig {
                cloud_provider: CloudProvider::AWS,
                deployment_model: DeploymentModel::Serverless {
                    function_memory_mb: 1024,
                    timeout_seconds: 30,
                    concurrent_executions: 100,
                },
                scaling_config: ScalingConfig {
                    min_instances: 1,
                    max_instances: 10,
                    target_cpu_utilization: 70.0,
                    target_memory_utilization: 80.0,
                    scale_up_cooldown_seconds: 300,
                    scale_down_cooldown_seconds: 300,
                    requests_per_second_threshold: 10,
                },
                storage_config: StorageConfig {
                    audio_storage: AudioStorageConfig {
                        storage_type: StorageType::S3,
                        cdn_enabled: false,
                        compression_enabled: true,
                        retention_days: 30,
                    },
                    model_storage: ModelStorageConfig {
                        storage_type: StorageType::S3,
                        versioning_enabled: false,
                        encryption_at_rest: false,
                        global_replication: false,
                    },
                    cache_storage: CacheStorageConfig {
                        cache_type: CacheType::InMemory,
                        cache_size_gb: 1,
                        ttl_seconds: 3600,
                        eviction_policy: EvictionPolicy::LRU,
                    },
                    backup_config: BackupConfig {
                        enabled: false,
                        frequency_hours: 24,
                        retention_days: 7,
                        cross_region_backup: false,
                    },
                },
                network_config: NetworkConfig {
                    load_balancer: LoadBalancerConfig {
                        algorithm: LoadBalancingAlgorithm::RoundRobin,
                        health_check_interval_seconds: 30,
                        unhealthy_threshold: 3,
                        timeout_seconds: 30,
                    },
                    cdn_config: CDNConfig {
                        enabled: false,
                        cache_ttl_seconds: 3600,
                        edge_locations: vec!["us-east-1".to_string()],
                        compression_enabled: true,
                    },
                    ssl_config: SSLConfig {
                        enabled: true,
                        certificate_source: CertificateSource::LetsEncrypt,
                        min_tls_version: "1.2".to_string(),
                    },
                    regions: vec![CloudRegion {
                        region_id: "us-east-1".to_string(),
                        primary: true,
                        traffic_ratio: 1.0,
                        disaster_recovery: false,
                    }],
                },
                monitoring_config: MonitoringConfig {
                    metrics_enabled: true,
                    logging_level: LogLevel::Info,
                    alerting_config: AlertingConfig {
                        email_alerts: true,
                        slack_webhook: None,
                        pagerduty_enabled: false,
                        alert_thresholds: AlertThresholds {
                            error_rate_percent: 5.0,
                            response_time_ms: 5000,
                            cpu_utilization_percent: 80.0,
                            memory_utilization_percent: 85.0,
                            disk_usage_percent: 90.0,
                        },
                    },
                    observability_tools: vec![ObservabilityTool::CloudWatch],
                },
                security_config: SecurityConfig {
                    authentication: AuthenticationConfig {
                        method: AuthenticationMethod::APIKey,
                        api_keys_enabled: true,
                        jwt_enabled: false,
                        oauth2_enabled: false,
                    },
                    authorization: AuthorizationConfig {
                        rbac_enabled: false,
                        rate_limiting: RateLimitingConfig {
                            requests_per_minute: 100,
                            burst_capacity: 50,
                            per_user_limit: 10,
                        },
                        ip_whitelist: vec![],
                    },
                    encryption: EncryptionConfig {
                        encryption_at_rest: false,
                        encryption_in_transit: true,
                        key_management: KeyManagementConfig {
                            service: KeyManagementService::CloudKMS,
                            key_rotation_days: 90,
                            hsm_enabled: false,
                        },
                    },
                    network_security: NetworkSecurityConfig {
                        vpc_enabled: true,
                        firewall_rules: vec![],
                        ddos_protection: false,
                        waf_enabled: false,
                    },
                },
                cost_optimization: CostOptimizationConfig {
                    spot_instances_enabled: false,
                    reserved_instances_ratio: 0.0,
                    auto_scaling_aggressive: false,
                    storage_lifecycle_policies: vec![],
                    cost_alerts_enabled: true,
                    budget_limit_usd: 500.0,
                },
            },
        ),
        // Enterprise scenario
        (
            "enterprise_production".to_string(),
            CloudDeploymentConfig {
                cloud_provider: CloudProvider::MultiCloud,
                deployment_model: DeploymentModel::Containers {
                    cpu_cores: 2.0,
                    memory_gb: 8,
                    replicas: 10,
                },
                scaling_config: ScalingConfig {
                    min_instances: 5,
                    max_instances: 100,
                    target_cpu_utilization: 60.0,
                    target_memory_utilization: 70.0,
                    scale_up_cooldown_seconds: 180,
                    scale_down_cooldown_seconds: 600,
                    requests_per_second_threshold: 1000,
                },
                storage_config: StorageConfig {
                    audio_storage: AudioStorageConfig {
                        storage_type: StorageType::Distributed,
                        cdn_enabled: true,
                        compression_enabled: true,
                        retention_days: 365,
                    },
                    model_storage: ModelStorageConfig {
                        storage_type: StorageType::Distributed,
                        versioning_enabled: true,
                        encryption_at_rest: true,
                        global_replication: true,
                    },
                    cache_storage: CacheStorageConfig {
                        cache_type: CacheType::Redis,
                        cache_size_gb: 50,
                        ttl_seconds: 7200,
                        eviction_policy: EvictionPolicy::LRU,
                    },
                    backup_config: BackupConfig {
                        enabled: true,
                        frequency_hours: 6,
                        retention_days: 90,
                        cross_region_backup: true,
                    },
                },
                network_config: NetworkConfig {
                    load_balancer: LoadBalancerConfig {
                        algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
                        health_check_interval_seconds: 15,
                        unhealthy_threshold: 2,
                        timeout_seconds: 10,
                    },
                    cdn_config: CDNConfig {
                        enabled: true,
                        cache_ttl_seconds: 86400,
                        edge_locations: vec![
                            "us-east-1".to_string(),
                            "us-west-2".to_string(),
                            "eu-west-1".to_string(),
                            "ap-southeast-1".to_string(),
                        ],
                        compression_enabled: true,
                    },
                    ssl_config: SSLConfig {
                        enabled: true,
                        certificate_source: CertificateSource::Custom,
                        min_tls_version: "1.3".to_string(),
                    },
                    regions: vec![
                        CloudRegion {
                            region_id: "us-east-1".to_string(),
                            primary: true,
                            traffic_ratio: 0.4,
                            disaster_recovery: false,
                        },
                        CloudRegion {
                            region_id: "us-west-2".to_string(),
                            primary: false,
                            traffic_ratio: 0.3,
                            disaster_recovery: true,
                        },
                        CloudRegion {
                            region_id: "eu-west-1".to_string(),
                            primary: false,
                            traffic_ratio: 0.2,
                            disaster_recovery: false,
                        },
                        CloudRegion {
                            region_id: "ap-southeast-1".to_string(),
                            primary: false,
                            traffic_ratio: 0.1,
                            disaster_recovery: false,
                        },
                    ],
                },
                monitoring_config: MonitoringConfig {
                    metrics_enabled: true,
                    logging_level: LogLevel::Warning,
                    alerting_config: AlertingConfig {
                        email_alerts: true,
                        slack_webhook: Some("https://hooks.slack.com/services/...".to_string()),
                        pagerduty_enabled: true,
                        alert_thresholds: AlertThresholds {
                            error_rate_percent: 1.0,
                            response_time_ms: 1000,
                            cpu_utilization_percent: 70.0,
                            memory_utilization_percent: 80.0,
                            disk_usage_percent: 85.0,
                        },
                    },
                    observability_tools: vec![
                        ObservabilityTool::Prometheus,
                        ObservabilityTool::Grafana,
                        ObservabilityTool::DataDog,
                        ObservabilityTool::Jaeger,
                    ],
                },
                security_config: SecurityConfig {
                    authentication: AuthenticationConfig {
                        method: AuthenticationMethod::OAuth2,
                        api_keys_enabled: true,
                        jwt_enabled: true,
                        oauth2_enabled: true,
                    },
                    authorization: AuthorizationConfig {
                        rbac_enabled: true,
                        rate_limiting: RateLimitingConfig {
                            requests_per_minute: 10000,
                            burst_capacity: 5000,
                            per_user_limit: 1000,
                        },
                        ip_whitelist: vec![],
                    },
                    encryption: EncryptionConfig {
                        encryption_at_rest: true,
                        encryption_in_transit: true,
                        key_management: KeyManagementConfig {
                            service: KeyManagementService::HashiCorpVault,
                            key_rotation_days: 30,
                            hsm_enabled: true,
                        },
                    },
                    network_security: NetworkSecurityConfig {
                        vpc_enabled: true,
                        firewall_rules: vec![FirewallRule {
                            name: "allow-https".to_string(),
                            direction: TrafficDirection::Ingress,
                            protocol: "TCP".to_string(),
                            port_range: "443".to_string(),
                            source_cidrs: vec!["0.0.0.0/0".to_string()],
                        }],
                        ddos_protection: true,
                        waf_enabled: true,
                    },
                },
                cost_optimization: CostOptimizationConfig {
                    spot_instances_enabled: true,
                    reserved_instances_ratio: 0.6,
                    auto_scaling_aggressive: true,
                    storage_lifecycle_policies: vec![
                        StorageLifecyclePolicy {
                            name: "archive-old-audio".to_string(),
                            transition_days: 30,
                            storage_class: StorageClass::InfrequentAccess,
                        },
                        StorageLifecyclePolicy {
                            name: "deep-archive".to_string(),
                            transition_days: 365,
                            storage_class: StorageClass::DeepArchive,
                        },
                    ],
                    cost_alerts_enabled: true,
                    budget_limit_usd: 50000.0,
                },
            },
        ),
    ]
}
