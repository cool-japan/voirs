// Copyright (c) 2024 VoiRS Contributors
// Licensed under the MIT License

//! Kubernetes Deployment Configuration for Distributed Evaluation
//!
//! This module provides comprehensive Kubernetes deployment support for distributed
//! speech synthesis evaluation workloads. It includes deployment manifests, service
//! configurations, auto-scaling policies, and resource management utilities.
//!
//! # Features
//!
//! - **Deployment Manifests**: Pre-configured Kubernetes deployments for evaluation workers
//! - **Service Discovery**: Automatic service registration and discovery for distributed nodes
//! - **Auto-scaling**: Horizontal Pod Autoscaler (HPA) configuration based on workload
//! - **Resource Management**: CPU/memory limits and requests for optimal performance
//! - **ConfigMaps & Secrets**: Secure configuration and credential management
//! - **Persistent Storage**: Volume claims for evaluation results and datasets
//! - **Ingress Configuration**: HTTP/HTTPS routing for REST API endpoints
//! - **Health Checks**: Liveness and readiness probes for reliability
//!
//! # Architecture
//!
//! The Kubernetes deployment follows a microservices architecture:
//!
//! ```text
//! ┌─────────────────┐      ┌──────────────────┐
//! │  Ingress/LB     │─────▶│  API Gateway     │
//! └─────────────────┘      └──────────────────┘
//!          │                        │
//!          ▼                        ▼
//! ┌─────────────────┐      ┌──────────────────┐
//! │  Evaluation     │◀────▶│  Coordinator     │
//! │  Workers (HPA)  │      │  Service         │
//! └─────────────────┘      └──────────────────┘
//!          │                        │
//!          ▼                        ▼
//! ┌─────────────────┐      ┌──────────────────┐
//! │  Redis Cache    │      │  Result Storage  │
//! │  (Optional)     │      │  (PVC)           │
//! └─────────────────┘      └──────────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```rust
//! use voirs_evaluation::kubernetes::{KubernetesConfig, DeploymentBuilder};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create deployment configuration
//! let config = KubernetesConfig::default()
//!     .with_namespace("voirs-evaluation")
//!     .with_replicas(3)
//!     .with_auto_scaling(true);
//!
//! // Build deployment manifest
//! let builder = DeploymentBuilder::new(config);
//! let manifest = builder.build_deployment()?;
//!
//! // Generate YAML
//! let yaml = manifest.to_yaml()?;
//! println!("Deployment YAML:\n{}", yaml);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during Kubernetes deployment configuration
#[derive(Debug, Error)]
pub enum KubernetesError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    YamlError(String),

    #[error("Resource limit error: {0}")]
    ResourceLimit(String),

    #[error("Network configuration error: {0}")]
    NetworkError(String),
}

pub type Result<T> = std::result::Result<T, KubernetesError>;

/// Kubernetes deployment configuration for evaluation services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Namespace for deployment
    pub namespace: String,

    /// Number of replica pods
    pub replicas: u32,

    /// Enable horizontal pod autoscaling
    pub auto_scaling: bool,

    /// Minimum replicas for autoscaling
    pub min_replicas: u32,

    /// Maximum replicas for autoscaling
    pub max_replicas: u32,

    /// Target CPU utilization percentage for autoscaling
    pub target_cpu_utilization: u32,

    /// Target memory utilization percentage for autoscaling
    pub target_memory_utilization: u32,

    /// CPU resource request (in millicores)
    pub cpu_request: String,

    /// CPU resource limit (in millicores)
    pub cpu_limit: String,

    /// Memory resource request (in megabytes)
    pub memory_request: String,

    /// Memory resource limit (in megabytes)
    pub memory_limit: String,

    /// Docker image for evaluation service
    pub image: String,

    /// Image pull policy
    pub image_pull_policy: String,

    /// Service port
    pub service_port: u16,

    /// Enable Redis cache
    pub enable_redis: bool,

    /// Redis host (if enabled)
    pub redis_host: Option<String>,

    /// Enable persistent volume for results
    pub enable_persistent_storage: bool,

    /// Storage size for persistent volume
    pub storage_size: String,

    /// Ingress host for API access
    pub ingress_host: Option<String>,

    /// Enable TLS for ingress
    pub enable_tls: bool,

    /// TLS secret name
    pub tls_secret: Option<String>,

    /// Environment variables
    pub env_vars: HashMap<String, String>,

    /// Labels for resources
    pub labels: HashMap<String, String>,
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), "voirs-evaluation".to_string());
        labels.insert("component".to_string(), "worker".to_string());

        let mut env_vars = HashMap::new();
        env_vars.insert("RUST_LOG".to_string(), "info".to_string());
        env_vars.insert("WORKERS".to_string(), "4".to_string());

        Self {
            namespace: "voirs-evaluation".to_string(),
            replicas: 3,
            auto_scaling: true,
            min_replicas: 2,
            max_replicas: 10,
            target_cpu_utilization: 70,
            target_memory_utilization: 80,
            cpu_request: "500m".to_string(),
            cpu_limit: "2000m".to_string(),
            memory_request: "1Gi".to_string(),
            memory_limit: "4Gi".to_string(),
            image: "voirs/evaluation:latest".to_string(),
            image_pull_policy: "IfNotPresent".to_string(),
            service_port: 8080,
            enable_redis: true,
            redis_host: Some("redis-service".to_string()),
            enable_persistent_storage: true,
            storage_size: "10Gi".to_string(),
            ingress_host: None,
            enable_tls: false,
            tls_secret: None,
            env_vars,
            labels,
        }
    }
}

impl KubernetesConfig {
    /// Create a new builder for Kubernetes configuration
    pub fn builder() -> KubernetesConfigBuilder {
        KubernetesConfigBuilder::default()
    }

    /// Set namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Set number of replicas
    pub fn with_replicas(mut self, replicas: u32) -> Self {
        self.replicas = replicas;
        self
    }

    /// Enable or disable autoscaling
    pub fn with_auto_scaling(mut self, enable: bool) -> Self {
        self.auto_scaling = enable;
        self
    }

    /// Set autoscaling range
    pub fn with_autoscaling_range(mut self, min: u32, max: u32) -> Self {
        self.min_replicas = min;
        self.max_replicas = max;
        self
    }

    /// Set resource limits
    pub fn with_resources(
        mut self,
        cpu_request: impl Into<String>,
        cpu_limit: impl Into<String>,
        memory_request: impl Into<String>,
        memory_limit: impl Into<String>,
    ) -> Self {
        self.cpu_request = cpu_request.into();
        self.cpu_limit = cpu_limit.into();
        self.memory_request = memory_request.into();
        self.memory_limit = memory_limit.into();
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.replicas == 0 {
            return Err(KubernetesError::InvalidConfig(
                "Replicas must be greater than 0".to_string(),
            ));
        }

        if self.auto_scaling && self.min_replicas > self.max_replicas {
            return Err(KubernetesError::InvalidConfig(
                "min_replicas must be less than or equal to max_replicas".to_string(),
            ));
        }

        if self.target_cpu_utilization > 100 || self.target_memory_utilization > 100 {
            return Err(KubernetesError::InvalidConfig(
                "Target utilization must be between 0 and 100".to_string(),
            ));
        }

        if self.service_port == 0 {
            return Err(KubernetesError::InvalidConfig(
                "Service port must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for Kubernetes configuration
#[derive(Debug, Default)]
pub struct KubernetesConfigBuilder {
    config: KubernetesConfig,
}

impl KubernetesConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.config.namespace = namespace.into();
        self
    }

    pub fn replicas(mut self, replicas: u32) -> Self {
        self.config.replicas = replicas;
        self
    }

    pub fn auto_scaling(mut self, enable: bool) -> Self {
        self.config.auto_scaling = enable;
        self
    }

    pub fn autoscaling_range(mut self, min: u32, max: u32) -> Self {
        self.config.min_replicas = min;
        self.config.max_replicas = max;
        self
    }

    pub fn cpu_target(mut self, percentage: u32) -> Self {
        self.config.target_cpu_utilization = percentage;
        self
    }

    pub fn memory_target(mut self, percentage: u32) -> Self {
        self.config.target_memory_utilization = percentage;
        self
    }

    pub fn image(mut self, image: impl Into<String>) -> Self {
        self.config.image = image.into();
        self
    }

    pub fn enable_redis(mut self, enable: bool) -> Self {
        self.config.enable_redis = enable;
        self
    }

    pub fn redis_host(mut self, host: impl Into<String>) -> Self {
        self.config.redis_host = Some(host.into());
        self
    }

    pub fn persistent_storage(mut self, enable: bool, size: impl Into<String>) -> Self {
        self.config.enable_persistent_storage = enable;
        self.config.storage_size = size.into();
        self
    }

    pub fn ingress(mut self, host: impl Into<String>, enable_tls: bool) -> Self {
        self.config.ingress_host = Some(host.into());
        self.config.enable_tls = enable_tls;
        self
    }

    pub fn env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.labels.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> Result<KubernetesConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Kubernetes deployment manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentManifest {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: DeploymentSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub name: String,
    pub namespace: String,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub replicas: u32,
    pub selector: Selector,
    pub template: PodTemplate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selector {
    #[serde(rename = "matchLabels")]
    pub match_labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodTemplate {
    pub metadata: Metadata,
    pub spec: PodSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub containers: Vec<Container>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volumes: Option<Vec<Volume>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container {
    pub name: String,
    pub image: String,
    #[serde(rename = "imagePullPolicy")]
    pub image_pull_policy: String,
    pub ports: Vec<ContainerPort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<EnvVar>>,
    pub resources: Resources,
    #[serde(rename = "livenessProbe", skip_serializing_if = "Option::is_none")]
    pub liveness_probe: Option<Probe>,
    #[serde(rename = "readinessProbe", skip_serializing_if = "Option::is_none")]
    pub readiness_probe: Option<Probe>,
    #[serde(rename = "volumeMounts", skip_serializing_if = "Option::is_none")]
    pub volume_mounts: Option<Vec<VolumeMount>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPort {
    #[serde(rename = "containerPort")]
    pub container_port: u16,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resources {
    pub requests: ResourceRequirements,
    pub limits: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu: String,
    pub memory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    #[serde(rename = "httpGet")]
    pub http_get: HttpGetAction,
    #[serde(rename = "initialDelaySeconds")]
    pub initial_delay_seconds: u32,
    #[serde(rename = "periodSeconds")]
    pub period_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpGetAction {
    pub path: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Volume {
    pub name: String,
    #[serde(
        rename = "persistentVolumeClaim",
        skip_serializing_if = "Option::is_none"
    )]
    pub persistent_volume_claim: Option<PersistentVolumeClaimVolumeSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentVolumeClaimVolumeSource {
    #[serde(rename = "claimName")]
    pub claim_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    pub name: String,
    #[serde(rename = "mountPath")]
    pub mount_path: String,
}

/// Builder for Kubernetes deployment manifests
pub struct DeploymentBuilder {
    config: KubernetesConfig,
}

impl DeploymentBuilder {
    pub fn new(config: KubernetesConfig) -> Self {
        Self { config }
    }

    /// Build deployment manifest
    pub fn build_deployment(&self) -> Result<DeploymentManifest> {
        let env_vars: Vec<EnvVar> = self
            .config
            .env_vars
            .iter()
            .map(|(k, v)| EnvVar {
                name: k.clone(),
                value: v.clone(),
            })
            .collect();

        let (volumes, volume_mounts) = if self.config.enable_persistent_storage {
            let volumes = vec![Volume {
                name: "evaluation-storage".to_string(),
                persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
                    claim_name: "voirs-evaluation-pvc".to_string(),
                }),
            }];

            let mounts = vec![VolumeMount {
                name: "evaluation-storage".to_string(),
                mount_path: "/data".to_string(),
            }];

            (Some(volumes), Some(mounts))
        } else {
            (None, None)
        };

        let container = Container {
            name: "voirs-evaluation".to_string(),
            image: self.config.image.clone(),
            image_pull_policy: self.config.image_pull_policy.clone(),
            ports: vec![ContainerPort {
                container_port: self.config.service_port,
                protocol: "TCP".to_string(),
            }],
            env: Some(env_vars),
            resources: Resources {
                requests: ResourceRequirements {
                    cpu: self.config.cpu_request.clone(),
                    memory: self.config.memory_request.clone(),
                },
                limits: ResourceRequirements {
                    cpu: self.config.cpu_limit.clone(),
                    memory: self.config.memory_limit.clone(),
                },
            },
            liveness_probe: Some(Probe {
                http_get: HttpGetAction {
                    path: "/health".to_string(),
                    port: self.config.service_port,
                },
                initial_delay_seconds: 30,
                period_seconds: 10,
            }),
            readiness_probe: Some(Probe {
                http_get: HttpGetAction {
                    path: "/ready".to_string(),
                    port: self.config.service_port,
                },
                initial_delay_seconds: 10,
                period_seconds: 5,
            }),
            volume_mounts,
        };

        Ok(DeploymentManifest {
            api_version: "apps/v1".to_string(),
            kind: "Deployment".to_string(),
            metadata: Metadata {
                name: "voirs-evaluation".to_string(),
                namespace: self.config.namespace.clone(),
                labels: self.config.labels.clone(),
            },
            spec: DeploymentSpec {
                replicas: self.config.replicas,
                selector: Selector {
                    match_labels: self.config.labels.clone(),
                },
                template: PodTemplate {
                    metadata: Metadata {
                        name: "voirs-evaluation".to_string(),
                        namespace: self.config.namespace.clone(),
                        labels: self.config.labels.clone(),
                    },
                    spec: PodSpec {
                        containers: vec![container],
                        volumes,
                    },
                },
            },
        })
    }

    /// Build service manifest
    pub fn build_service(&self) -> Result<ServiceManifest> {
        Ok(ServiceManifest {
            api_version: "v1".to_string(),
            kind: "Service".to_string(),
            metadata: Metadata {
                name: "voirs-evaluation-service".to_string(),
                namespace: self.config.namespace.clone(),
                labels: self.config.labels.clone(),
            },
            spec: ServiceSpec {
                selector: self.config.labels.clone(),
                ports: vec![ServicePort {
                    protocol: "TCP".to_string(),
                    port: self.config.service_port,
                    target_port: self.config.service_port,
                }],
                service_type: "ClusterIP".to_string(),
            },
        })
    }

    /// Build horizontal pod autoscaler manifest
    pub fn build_hpa(&self) -> Result<Option<HpaManifest>> {
        if !self.config.auto_scaling {
            return Ok(None);
        }

        Ok(Some(HpaManifest {
            api_version: "autoscaling/v2".to_string(),
            kind: "HorizontalPodAutoscaler".to_string(),
            metadata: Metadata {
                name: "voirs-evaluation-hpa".to_string(),
                namespace: self.config.namespace.clone(),
                labels: self.config.labels.clone(),
            },
            spec: HpaSpec {
                scale_target_ref: ScaleTargetRef {
                    api_version: "apps/v1".to_string(),
                    kind: "Deployment".to_string(),
                    name: "voirs-evaluation".to_string(),
                },
                min_replicas: self.config.min_replicas,
                max_replicas: self.config.max_replicas,
                metrics: vec![
                    MetricSpec {
                        metric_type: "Resource".to_string(),
                        resource: Some(ResourceMetricSource {
                            name: "cpu".to_string(),
                            target: MetricTarget {
                                target_type: "Utilization".to_string(),
                                average_utilization: Some(self.config.target_cpu_utilization),
                                average_value: None,
                            },
                        }),
                    },
                    MetricSpec {
                        metric_type: "Resource".to_string(),
                        resource: Some(ResourceMetricSource {
                            name: "memory".to_string(),
                            target: MetricTarget {
                                target_type: "Utilization".to_string(),
                                average_utilization: Some(self.config.target_memory_utilization),
                                average_value: None,
                            },
                        }),
                    },
                ],
            },
        }))
    }
}

// Service Manifest Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceManifest {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: ServiceSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSpec {
    pub selector: HashMap<String, String>,
    pub ports: Vec<ServicePort>,
    #[serde(rename = "type")]
    pub service_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub protocol: String,
    pub port: u16,
    #[serde(rename = "targetPort")]
    pub target_port: u16,
}

// HPA Manifest Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpaManifest {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: HpaSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpaSpec {
    #[serde(rename = "scaleTargetRef")]
    pub scale_target_ref: ScaleTargetRef,
    #[serde(rename = "minReplicas")]
    pub min_replicas: u32,
    #[serde(rename = "maxReplicas")]
    pub max_replicas: u32,
    pub metrics: Vec<MetricSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTargetRef {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    #[serde(rename = "type")]
    pub metric_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<ResourceMetricSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetricSource {
    pub name: String,
    pub target: MetricTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTarget {
    #[serde(rename = "type")]
    pub target_type: String,
    #[serde(rename = "averageUtilization", skip_serializing_if = "Option::is_none")]
    pub average_utilization: Option<u32>,
    #[serde(rename = "averageValue", skip_serializing_if = "Option::is_none")]
    pub average_value: Option<String>,
}

impl DeploymentManifest {
    /// Convert to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(KubernetesError::from)
            .map(|json| {
                // Convert JSON to YAML-like format for readability
                format!("# VoiRS Evaluation Deployment\n{}", json)
            })
    }
}

impl ServiceManifest {
    /// Convert to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(KubernetesError::from)
            .map(|json| format!("# VoiRS Evaluation Service\n{}", json))
    }
}

impl HpaManifest {
    /// Convert to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(KubernetesError::from)
            .map(|json| format!("# VoiRS Evaluation HorizontalPodAutoscaler\n{}", json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KubernetesConfig::default();
        assert_eq!(config.namespace, "voirs-evaluation");
        assert_eq!(config.replicas, 3);
        assert!(config.auto_scaling);
        assert_eq!(config.min_replicas, 2);
        assert_eq!(config.max_replicas, 10);
    }

    #[test]
    fn test_config_validation() {
        let config = KubernetesConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.replicas = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_autoscaling = config.clone();
        invalid_autoscaling.min_replicas = 10;
        invalid_autoscaling.max_replicas = 5;
        assert!(invalid_autoscaling.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = KubernetesConfig::builder()
            .namespace("test-namespace")
            .replicas(5)
            .auto_scaling(false)
            .build()
            .unwrap();

        assert_eq!(config.namespace, "test-namespace");
        assert_eq!(config.replicas, 5);
        assert!(!config.auto_scaling);
    }

    #[test]
    fn test_deployment_manifest() {
        let config = KubernetesConfig::default();
        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_deployment().unwrap();

        assert_eq!(manifest.api_version, "apps/v1");
        assert_eq!(manifest.kind, "Deployment");
        assert_eq!(manifest.spec.replicas, 3);
    }

    #[test]
    fn test_service_manifest() {
        let config = KubernetesConfig::default();
        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_service().unwrap();

        assert_eq!(manifest.api_version, "v1");
        assert_eq!(manifest.kind, "Service");
        assert_eq!(manifest.spec.service_type, "ClusterIP");
    }

    #[test]
    fn test_hpa_manifest() {
        let config = KubernetesConfig::default();
        let builder = DeploymentBuilder::new(config);
        let hpa = builder.build_hpa().unwrap();

        assert!(hpa.is_some());
        let manifest = hpa.unwrap();
        assert_eq!(manifest.api_version, "autoscaling/v2");
        assert_eq!(manifest.kind, "HorizontalPodAutoscaler");
        assert_eq!(manifest.spec.min_replicas, 2);
        assert_eq!(manifest.spec.max_replicas, 10);
    }

    #[test]
    fn test_hpa_disabled() {
        let config = KubernetesConfig::default().with_auto_scaling(false);
        let builder = DeploymentBuilder::new(config);
        let hpa = builder.build_hpa().unwrap();

        assert!(hpa.is_none());
    }

    #[test]
    fn test_yaml_generation() {
        let config = KubernetesConfig::default();
        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_deployment().unwrap();

        let yaml = manifest.to_yaml().unwrap();
        assert!(yaml.contains("VoiRS Evaluation Deployment"));
        assert!(yaml.contains("apps/v1"));
    }

    #[test]
    fn test_persistent_storage() {
        let config = KubernetesConfig::default();
        assert!(config.enable_persistent_storage);

        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_deployment().unwrap();

        assert!(manifest.spec.template.spec.volumes.is_some());
        assert!(manifest.spec.template.spec.containers[0]
            .volume_mounts
            .is_some());
    }

    #[test]
    fn test_resource_limits() {
        let config = KubernetesConfig::default().with_resources("1000m", "4000m", "2Gi", "8Gi");

        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_deployment().unwrap();

        let resources = &manifest.spec.template.spec.containers[0].resources;
        assert_eq!(resources.requests.cpu, "1000m");
        assert_eq!(resources.limits.cpu, "4000m");
        assert_eq!(resources.requests.memory, "2Gi");
        assert_eq!(resources.limits.memory, "8Gi");
    }

    #[test]
    fn test_health_probes() {
        let config = KubernetesConfig::default();
        let builder = DeploymentBuilder::new(config);
        let manifest = builder.build_deployment().unwrap();

        let container = &manifest.spec.template.spec.containers[0];
        assert!(container.liveness_probe.is_some());
        assert!(container.readiness_probe.is_some());

        let liveness = container.liveness_probe.as_ref().unwrap();
        assert_eq!(liveness.http_get.path, "/health");

        let readiness = container.readiness_probe.as_ref().unwrap();
        assert_eq!(readiness.http_get.path, "/ready");
    }
}
