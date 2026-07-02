use std::time::{Instant, SystemTime};

use crate::config::*;
use crate::infrastructure_support::*;
use crate::monitoring::*;
use crate::scenarios::*;
use crate::service_types::*;

#[test]
fn test_cloud_config_creation() {
    let scenarios = create_deployment_scenarios();
    assert_eq!(scenarios.len(), 2);

    let (name, config) = &scenarios[0];
    assert_eq!(name, "startup_mvp");
    assert!(matches!(config.cloud_provider, CloudProvider::AWS));
}

#[test]
fn test_cloud_service_creation() {
    let scenarios = create_deployment_scenarios();
    let config = scenarios[0].1.clone();

    let service = CloudVoiRSService::new(config);
    assert!(service.is_ok());
}

#[test]
fn test_load_balancer() {
    let config = LoadBalancerConfig {
        algorithm: LoadBalancingAlgorithm::RoundRobin,
        health_check_interval_seconds: 30,
        unhealthy_threshold: 3,
        timeout_seconds: 30,
    };

    let load_balancer = LoadBalancer::new(&config);
    assert!(load_balancer.is_ok());
}

#[test]
fn test_storage_manager() {
    let config = StorageConfig {
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
    };

    let storage_manager = StorageManager::new(&config);
    assert!(storage_manager.is_ok());
}

#[test]
fn test_synthesis_request() {
    let request = SynthesisRequest {
        id: 1,
        text: "Test".to_string(),
        voice_id: "test_voice".to_string(),
        priority: RequestPriority::Normal,
        region: "us-east-1".to_string(),
        client_id: "test_client".to_string(),
        callback_url: None,
        timestamp: SystemTime::now(),
        timeout_seconds: 30,
        quality: AudioQuality::Standard,
    };

    assert_eq!(request.id, 1);
    assert_eq!(request.text, "Test");
}

#[test]
fn test_priority_ordering() {
    let high = RequestPriority::Critical;
    let low = RequestPriority::Low;

    assert!(high > low);
    assert!(low < high);
}

#[test]
fn test_infrastructure_code_generation() {
    let scenarios = create_deployment_scenarios();
    let config = scenarios[0].1.clone();
    let service = CloudVoiRSService::new(config).unwrap();

    let infra_code = service.generate_infrastructure_code();

    assert!(infra_code
        .aws_cloudformation
        .contains("AWSTemplateFormatVersion"));
    assert!(infra_code.azure_arm.contains("Microsoft.ContainerInstance"));
    assert!(infra_code.gcp_deployment_manager.contains("gcp-types"));
    assert!(infra_code.kubernetes_yaml.contains("apiVersion"));
    assert!(infra_code.terraform.contains("terraform"));
    assert!(infra_code.docker_compose.contains("version"));
}

#[test]
fn test_auto_scaler() {
    let config = ScalingConfig {
        min_instances: 2,
        max_instances: 10,
        target_cpu_utilization: 70.0,
        target_memory_utilization: 80.0,
        scale_up_cooldown_seconds: 300,
        scale_down_cooldown_seconds: 300,
        requests_per_second_threshold: 100,
    };

    let scaler = AutoScaler::new(&config);
    assert!(scaler.is_ok());
}

#[test]
fn test_service_instance() {
    let instance = ServiceInstance {
        id: "test-instance".to_string(),
        region: "us-east-1".to_string(),
        status: InstanceStatus::Healthy,
        cpu_usage: 50.0,
        memory_usage: 60.0,
        request_count: 100,
        last_health_check: Instant::now(),
        created_at: Instant::now(),
    };

    assert_eq!(instance.id, "test-instance");
    assert!(matches!(instance.status, InstanceStatus::Healthy));
}

#[test]
fn test_cloud_monitor() {
    let config = MonitoringConfig {
        metrics_enabled: true,
        logging_level: LogLevel::Info,
        alerting_config: AlertingConfig {
            email_alerts: true,
            slack_webhook: None,
            pagerduty_enabled: false,
            alert_thresholds: AlertThresholds {
                error_rate_percent: 5.0,
                response_time_ms: 1000,
                cpu_utilization_percent: 80.0,
                memory_utilization_percent: 85.0,
                disk_usage_percent: 90.0,
            },
        },
        observability_tools: vec![ObservabilityTool::CloudWatch],
    };

    let monitor = CloudMonitor::new(&config);
    assert!(monitor.is_ok());
}
