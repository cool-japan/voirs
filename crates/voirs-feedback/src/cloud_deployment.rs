//! Cloud deployment and orchestration for `VoiRS` feedback microservices
//!
//! This module provides comprehensive cloud deployment functionality including
//! container orchestration, service mesh integration, auto-scaling, and
//! cloud-native monitoring capabilities.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::process::Command;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

// Note: Using local types instead of microservices module types for compatibility

/// Cloud deployment errors
#[derive(Error, Debug)]
pub enum CloudDeploymentError {
    /// Deployment failed
    #[error("Deployment failed: {message}")]
    DeploymentFailed {
        /// Error message
        message: String,
    },

    /// Container orchestration error
    #[error("Container orchestration error: {details}")]
    OrchestrationError {
        /// Error details
        details: String,
    },

    /// Scaling operation failed
    #[error("Scaling operation failed: {operation} - {reason}")]
    ScalingFailed {
        /// Scaling operation
        operation: String,
        /// Failure reason
        reason: String,
    },

    /// Service mesh configuration error
    #[error("Service mesh configuration error: {service} - {error}")]
    ServiceMeshError {
        /// Service name
        service: String,
        /// Error description
        error: String,
    },

    /// Cloud provider API error
    #[error("Cloud provider API error: {provider} - {message}")]
    CloudProviderError {
        /// Cloud provider name
        provider: String,
        /// Error message
        message: String,
    },
}

/// Result type for cloud deployment operations
pub type CloudDeploymentResult<T> = Result<T, CloudDeploymentError>;

/// Supported cloud providers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon Web Services
    AWS,
    /// Google Cloud Platform
    GCP,
    /// Microsoft Azure
    Azure,
    /// Kubernetes (provider-agnostic)
    Kubernetes,
    /// Docker Swarm
    DockerSwarm,
    /// Local development
    Local,
}

/// Container runtime types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContainerRuntime {
    /// Docker
    Docker,
    /// Containerd
    Containerd,
    /// CRI-O
    CriO,
}

/// Deployment strategy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Rolling update deployment
    RollingUpdate {
        /// Maximum unavailable replicas during update
        max_unavailable: u32,
        /// Maximum surge replicas during update
        max_surge: u32,
    },
    /// Blue-green deployment
    BlueGreen,
    /// Canary deployment
    Canary {
        /// Percentage of traffic for canary
        traffic_percentage: f32,
        /// Canary evaluation duration
        evaluation_duration: Duration,
    },
    /// Recreate deployment (downtime)
    Recreate,
}

/// Service deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Service name
    pub service_name: String,
    /// Container image
    pub image: String,
    /// Image tag
    pub tag: String,
    /// Number of replicas
    pub replicas: u32,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Port configurations
    pub ports: Vec<PortConfig>,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Service mesh configuration
    pub service_mesh: Option<ServiceMeshConfig>,
    /// Auto-scaling configuration
    pub auto_scaling: Option<AutoScalingConfig>,
}

/// Resource requirements and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU requests (in millicores)
    pub cpu_request: u32,
    /// CPU limits (in millicores)
    pub cpu_limit: u32,
    /// Memory requests (in MiB)
    pub memory_request: u32,
    /// Memory limits (in MiB)
    pub memory_limit: u32,
    /// Storage requests (in GiB)
    pub storage_request: Option<u32>,
    /// GPU requirements
    pub gpu_request: Option<u32>,
}

/// Port configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConfig {
    /// Port name
    pub name: String,
    /// Container port
    pub container_port: u16,
    /// Service port
    pub service_port: u16,
    /// Protocol (TCP, UDP, HTTP, GRPC)
    pub protocol: String,
    /// Load balancer configuration
    pub load_balancer: Option<LoadBalancerConfig>,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Session affinity
    pub session_affinity: bool,
    /// Health check path
    pub health_check_path: String,
    /// Timeout settings
    pub timeout: Duration,
}

/// Load balancing algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round robin
    WeightedRoundRobin,
    /// IP hash
    IpHash,
    /// Least response time
    LeastResponseTime,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check path
    pub path: String,
    /// Health check port
    pub port: u16,
    /// Initial delay before health checks
    pub initial_delay: Duration,
    /// Interval between health checks
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Success threshold
    pub success_threshold: u32,
    /// Failure threshold
    pub failure_threshold: u32,
}

/// Service mesh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    /// Service mesh type (Istio, Linkerd, Consul Connect)
    pub mesh_type: String,
    /// Traffic policies
    pub traffic_policies: Vec<TrafficPolicy>,
    /// Security policies
    pub security_policies: Vec<SecurityPolicy>,
    /// Observability configuration
    pub observability: ObservabilityConfig,
}

/// Traffic policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPolicy {
    /// Policy name
    pub name: String,
    /// Source services
    pub sources: Vec<String>,
    /// Destination services
    pub destinations: Vec<String>,
    /// Traffic weight (0.0 to 1.0)
    pub weight: f32,
    /// Timeout configuration
    pub timeout: Option<Duration>,
    /// Retry configuration
    pub retry: Option<RetryPolicy>,
}

/// Security policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// mTLS configuration
    pub mtls_mode: MtlsMode,
    /// Authorization rules
    pub authorization_rules: Vec<AuthorizationRule>,
    /// Rate limiting
    pub rate_limit: Option<RateLimitConfig>,
}

/// mTLS modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MtlsMode {
    /// Strict mTLS required
    Strict,
    /// Permissive (mTLS optional)
    Permissive,
    /// Disabled
    Disabled,
}

/// Authorization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRule {
    /// Rule name
    pub name: String,
    /// Allowed principals
    pub principals: Vec<String>,
    /// Allowed operations
    pub operations: Vec<String>,
    /// Conditions
    pub conditions: HashMap<String, String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second limit
    pub requests_per_second: u32,
    /// Burst limit
    pub burst_limit: u32,
    /// Rate limit window
    pub window: Duration,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Retry timeout
    pub per_try_timeout: Duration,
    /// Retry backoff
    pub backoff: Duration,
    /// Retryable status codes
    pub retryable_status_codes: Vec<u16>,
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Metrics collection enabled
    pub metrics_enabled: bool,
    /// Tracing enabled
    pub tracing_enabled: bool,
    /// Access logging enabled
    pub access_logs_enabled: bool,
    /// Custom metrics
    pub custom_metrics: Vec<String>,
    /// Trace sampling rate (0.0 to 1.0)
    pub trace_sampling_rate: f32,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum number of replicas
    pub min_replicas: u32,
    /// Maximum number of replicas
    pub max_replicas: u32,
    /// Target CPU utilization (0.0 to 1.0)
    pub target_cpu_utilization: f32,
    /// Target memory utilization (0.0 to 1.0)
    pub target_memory_utilization: f32,
    /// Scale up cooldown period
    pub scale_up_cooldown: Duration,
    /// Scale down cooldown period
    pub scale_down_cooldown: Duration,
    /// Custom metrics for scaling
    pub custom_metrics: Vec<ScalingMetric>,
}

/// Custom scaling metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetric {
    /// Metric name
    pub name: String,
    /// Target value
    pub target_value: f64,
    /// Metric source
    pub source: MetricSource,
}

/// Metric source types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricSource {
    /// Pod-level metrics
    Pod,
    /// External metrics (e.g., queue length)
    External,
    /// Resource metrics (CPU, memory)
    Resource,
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment in progress
    Deploying,
    /// Deployment successful
    Deployed,
    /// Deployment failed
    Failed,
    /// Deployment updating
    Updating,
    /// Deployment scaling
    Scaling,
    /// Deployment terminating
    Terminating,
}

/// Deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    /// Deployment ID
    pub deployment_id: Uuid,
    /// Service name
    pub service_name: String,
    /// Cloud provider
    pub cloud_provider: CloudProvider,
    /// Deployment configuration
    pub config: DeploymentConfig,
    /// Current status
    pub status: DeploymentStatus,
    /// Current replica count
    pub current_replicas: u32,
    /// Ready replica count
    pub ready_replicas: u32,
    /// Deployment events
    pub events: Vec<DeploymentEvent>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Deployment event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: String,
    /// Event message
    pub message: String,
    /// Event severity
    pub severity: EventSeverity,
}

/// Event severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Informational event
    Info,
    /// Warning event
    Warning,
    /// Error event
    Error,
    /// Critical event
    Critical,
}

/// Cloud deployment orchestrator trait
#[async_trait]
pub trait CloudOrchestrator: Send + Sync {
    /// Deploy a service
    async fn deploy_service(
        &self,
        config: DeploymentConfig,
    ) -> CloudDeploymentResult<DeploymentInfo>;

    /// Update a deployment
    async fn update_deployment(
        &self,
        deployment_id: Uuid,
        config: DeploymentConfig,
    ) -> CloudDeploymentResult<()>;

    /// Scale a deployment
    async fn scale_deployment(
        &self,
        deployment_id: Uuid,
        replicas: u32,
    ) -> CloudDeploymentResult<()>;

    /// Delete a deployment
    async fn delete_deployment(&self, deployment_id: Uuid) -> CloudDeploymentResult<()>;

    /// Get deployment status
    async fn get_deployment_status(
        &self,
        deployment_id: Uuid,
    ) -> CloudDeploymentResult<DeploymentInfo>;

    /// List all deployments
    async fn list_deployments(&self) -> CloudDeploymentResult<Vec<DeploymentInfo>>;

    /// Get deployment logs
    async fn get_deployment_logs(
        &self,
        deployment_id: Uuid,
        tail_lines: Option<u32>,
    ) -> CloudDeploymentResult<Vec<String>>;

    /// Execute command in deployment
    async fn exec_command(
        &self,
        deployment_id: Uuid,
        pod_name: String,
        command: Vec<String>,
    ) -> CloudDeploymentResult<String>;
}

/// Kubernetes orchestrator implementation.
///
/// Talks to a real cluster through the `kubectl` binary on `PATH` (using
/// whatever `KUBECONFIG`/current-context is active, or the explicit
/// namespace/context set via [`Self::with_namespace`]/
/// [`Self::with_kubeconfig_context`]). There is no bundled fake backend: if
/// `kubectl` is missing or no cluster is reachable, every operation returns
/// a typed [`CloudDeploymentError`] instead of a fabricated success.
pub struct KubernetesOrchestrator {
    /// Deployment registry, kept in sync with the real cluster by
    /// background `kubectl get` polling after every apply/scale.
    deployments: Arc<RwLock<HashMap<Uuid, DeploymentInfo>>>,
    /// Cloud provider label (informational only: `kubectl` talks to
    /// whichever cluster the active kubeconfig points at, whether that is
    /// EKS/GKE/AKS or a local cluster).
    provider: CloudProvider,
    /// Optional `--namespace` to pass to every `kubectl` invocation.
    namespace: Option<String>,
    /// Optional `--context` to pass to every `kubectl` invocation.
    kubeconfig_context: Option<String>,
}

impl KubernetesOrchestrator {
    /// Create a new Kubernetes orchestrator targeting the current
    /// kubeconfig context's default namespace.
    #[must_use]
    pub fn new(provider: CloudProvider) -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
            provider,
            namespace: None,
            kubeconfig_context: None,
        }
    }

    /// Target a specific Kubernetes namespace for all `kubectl` calls.
    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Target a specific kubeconfig context for all `kubectl` calls.
    #[must_use]
    pub fn with_kubeconfig_context(mut self, context: impl Into<String>) -> Self {
        self.kubeconfig_context = Some(context.into());
        self
    }

    /// Build a `kubectl` command with the configured namespace/context
    /// already applied.
    fn kubectl(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new("kubectl");
        cmd.args(args);
        if let Some(namespace) = &self.namespace {
            cmd.arg("--namespace").arg(namespace);
        }
        if let Some(context) = &self.kubeconfig_context {
            cmd.arg("--context").arg(context);
        }
        cmd
    }

    /// Run a `kubectl` command, mapping a missing binary and a non-zero
    /// exit to distinct, honest error messages.
    async fn run_kubectl(&self, args: &[&str]) -> CloudDeploymentResult<std::process::Output> {
        let output = self
            .kubectl(args)
            .output()
            .await
            .map_err(|e| kubectl_spawn_error(&e))?;

        if !output.status.success() {
            return Err(CloudDeploymentError::OrchestrationError {
                details: format!(
                    "kubectl {} failed (exit {:?}): {}",
                    args.join(" "),
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        Ok(output)
    }

    /// Apply a manifest file to the real cluster via `kubectl apply -f`.
    async fn kubectl_apply(&self, manifest_path: &std::path::Path) -> CloudDeploymentResult<()> {
        let path = manifest_path.to_string_lossy();
        self.run_kubectl(&["apply", "-f", &path]).await?;
        Ok(())
    }

    /// Generate a Kubernetes `Deployment` + `Service` YAML manifest that
    /// actually reflects every configured field (environment variables,
    /// health-check probes, and deployment strategy), not just name/image.
    async fn generate_manifest(&self, config: &DeploymentConfig) -> CloudDeploymentResult<String> {
        let deployment = build_deployment_manifest(config);
        let service = build_service_manifest(config);

        let deployment_yaml = serde_yaml::to_string(&deployment).map_err(|e| {
            CloudDeploymentError::OrchestrationError {
                details: format!("failed to serialize Deployment manifest: {e}"),
            }
        })?;
        let service_yaml = serde_yaml::to_string(&service).map_err(|e| {
            CloudDeploymentError::OrchestrationError {
                details: format!("failed to serialize Service manifest: {e}"),
            }
        })?;

        Ok(format!("{deployment_yaml}---\n{service_yaml}"))
    }

    /// Write a manifest to a real file under the OS temp directory and
    /// return its path, so callers (and `kubectl apply -f`) can use it.
    async fn write_manifest_file(
        &self,
        deployment_id: Uuid,
        suffix: &str,
        manifest: &str,
    ) -> CloudDeploymentResult<std::path::PathBuf> {
        let path = std::env::temp_dir().join(format!("voirs-deploy-{deployment_id}{suffix}.yaml"));
        tokio::fs::write(&path, manifest).await.map_err(|e| {
            CloudDeploymentError::OrchestrationError {
                details: format!("failed to write manifest to {}: {e}", path.display()),
            }
        })?;
        Ok(path)
    }

    /// Background task: repeatedly `kubectl get deployment -o json` the
    /// real cluster and mirror the real `replicas`/`readyReplicas` counts
    /// into the local registry, flipping to `Deployed` only once the
    /// cluster itself reports enough ready replicas (or `Failed` after a
    /// bounded timeout). This replaces the previous fixed `sleep(5s)` +
    /// unconditional status flip with an honest reflection of real cluster
    /// state.
    async fn poll_deployment_status(
        deployments: Arc<RwLock<HashMap<Uuid, DeploymentInfo>>>,
        deployment_id: Uuid,
        service_name: String,
        namespace: Option<String>,
        kubeconfig_context: Option<String>,
        poll_interval: Duration,
        timeout: Duration,
    ) {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await; // first tick fires immediately; consume it

        loop {
            if tokio::time::Instant::now() >= deadline {
                if let Some(deployment) = deployments.write().await.get_mut(&deployment_id) {
                    if deployment.status != DeploymentStatus::Deployed {
                        deployment.status = DeploymentStatus::Failed;
                        deployment.updated_at = Utc::now();
                        deployment.events.push(DeploymentEvent {
                            timestamp: Utc::now(),
                            event_type: "Deployment".to_string(),
                            message: format!(
                                "Timed out after {}s waiting for kubectl to report readiness",
                                timeout.as_secs()
                            ),
                            severity: EventSeverity::Error,
                        });
                    }
                }
                return;
            }

            let mut cmd = Command::new("kubectl");
            cmd.arg("get")
                .arg("deployment")
                .arg(&service_name)
                .arg("-o")
                .arg("json");
            if let Some(namespace) = &namespace {
                cmd.arg("--namespace").arg(namespace);
            }
            if let Some(context) = &kubeconfig_context {
                cmd.arg("--context").arg(context);
            }

            if let Ok(output) = cmd.output().await {
                if output.status.success() {
                    if let Ok(status) = serde_json::from_slice::<serde_json::Value>(&output.stdout)
                    {
                        let current_replicas =
                            status["status"]["replicas"].as_u64().unwrap_or(0) as u32;
                        let ready_replicas =
                            status["status"]["readyReplicas"].as_u64().unwrap_or(0) as u32;
                        let desired_replicas =
                            status["spec"]["replicas"].as_u64().unwrap_or(0) as u32;

                        let mut guard = deployments.write().await;
                        let Some(deployment) = guard.get_mut(&deployment_id) else {
                            return; // Deployment was deleted locally; stop polling.
                        };
                        deployment.current_replicas = current_replicas;
                        deployment.ready_replicas = ready_replicas;
                        deployment.updated_at = Utc::now();

                        if desired_replicas > 0 && ready_replicas >= desired_replicas {
                            deployment.status = DeploymentStatus::Deployed;
                            deployment.events.push(DeploymentEvent {
                                timestamp: Utc::now(),
                                event_type: "Deployment".to_string(),
                                message: format!(
                                    "kubectl reports {ready_replicas}/{desired_replicas} replicas ready"
                                ),
                                severity: EventSeverity::Info,
                            });
                            return;
                        }
                    }
                }
                // Non-success (e.g. not yet visible to the API server) or
                // unparsable output: keep polling until the deadline.
            }

            ticker.tick().await;
        }
    }

    /// Spawn the real status-polling task for a deployment.
    fn spawn_status_poll(&self, deployment_id: Uuid, service_name: String) {
        let deployments = self.deployments.clone();
        let namespace = self.namespace.clone();
        let context = self.kubeconfig_context.clone();
        tokio::spawn(Self::poll_deployment_status(
            deployments,
            deployment_id,
            service_name,
            namespace,
            context,
            Duration::from_secs(3),
            Duration::from_secs(300),
        ));
    }
}

#[async_trait]
impl CloudOrchestrator for KubernetesOrchestrator {
    async fn deploy_service(
        &self,
        config: DeploymentConfig,
    ) -> CloudDeploymentResult<DeploymentInfo> {
        let deployment_id = Uuid::new_v4();
        let now = Utc::now();

        // Generate the real manifest and write it to a real file.
        let manifest = self.generate_manifest(&config).await?;
        let manifest_path = self
            .write_manifest_file(deployment_id, "", &manifest)
            .await?;

        // Actually apply it to the cluster. No cluster/kubectl => Err here,
        // and nothing is ever inserted into the registry claiming success.
        self.kubectl_apply(&manifest_path).await?;

        let deployment_info = DeploymentInfo {
            deployment_id,
            service_name: config.service_name.clone(),
            cloud_provider: self.provider.clone(),
            config: config.clone(),
            status: DeploymentStatus::Deploying,
            current_replicas: 0,
            ready_replicas: 0,
            events: vec![DeploymentEvent {
                timestamp: now,
                event_type: "Deployment".to_string(),
                message: format!(
                    "kubectl apply succeeded for {}; waiting for readiness",
                    config.service_name
                ),
                severity: EventSeverity::Info,
            }],
            created_at: now,
            updated_at: now,
        };

        self.deployments
            .write()
            .await
            .insert(deployment_id, deployment_info.clone());

        // Real background polling of the real cluster (replaces the old
        // fixed sleep(5s) + unconditional status flip).
        self.spawn_status_poll(deployment_id, config.service_name);

        Ok(deployment_info)
    }

    async fn update_deployment(
        &self,
        deployment_id: Uuid,
        config: DeploymentConfig,
    ) -> CloudDeploymentResult<()> {
        {
            let deployments = self.deployments.read().await;
            if !deployments.contains_key(&deployment_id) {
                return Err(CloudDeploymentError::DeploymentFailed {
                    message: format!("Deployment {deployment_id} not found"),
                });
            }
        }

        let manifest = self.generate_manifest(&config).await?;
        let manifest_path = self
            .write_manifest_file(deployment_id, "-update", &manifest)
            .await?;
        self.kubectl_apply(&manifest_path).await?;

        let service_name = config.service_name.clone();
        {
            let mut deployments = self.deployments.write().await;
            if let Some(deployment) = deployments.get_mut(&deployment_id) {
                deployment.config = config;
                deployment.status = DeploymentStatus::Updating;
                deployment.updated_at = Utc::now();
                deployment.events.push(DeploymentEvent {
                    timestamp: Utc::now(),
                    event_type: "Update".to_string(),
                    message: "kubectl apply succeeded for update; waiting for readiness"
                        .to_string(),
                    severity: EventSeverity::Info,
                });
            }
        }

        self.spawn_status_poll(deployment_id, service_name);
        Ok(())
    }

    async fn scale_deployment(
        &self,
        deployment_id: Uuid,
        replicas: u32,
    ) -> CloudDeploymentResult<()> {
        let service_name = {
            let deployments = self.deployments.read().await;
            deployments
                .get(&deployment_id)
                .map(|d| d.service_name.clone())
                .ok_or_else(|| CloudDeploymentError::ScalingFailed {
                    operation: "scale".to_string(),
                    reason: format!("Deployment {deployment_id} not found"),
                })?
        };

        let replicas_arg = format!("--replicas={replicas}");
        self.run_kubectl(&["scale", "deployment", &service_name, &replicas_arg])
            .await
            .map_err(|e| CloudDeploymentError::ScalingFailed {
                operation: "scale".to_string(),
                reason: e.to_string(),
            })?;

        {
            let mut deployments = self.deployments.write().await;
            if let Some(deployment) = deployments.get_mut(&deployment_id) {
                deployment.config.replicas = replicas;
                deployment.status = DeploymentStatus::Scaling;
                deployment.updated_at = Utc::now();
                deployment.events.push(DeploymentEvent {
                    timestamp: Utc::now(),
                    event_type: "Scale".to_string(),
                    message: format!("kubectl scale succeeded, target {replicas} replicas"),
                    severity: EventSeverity::Info,
                });
            }
        }

        self.spawn_status_poll(deployment_id, service_name);
        Ok(())
    }

    async fn delete_deployment(&self, deployment_id: Uuid) -> CloudDeploymentResult<()> {
        let service_name = {
            let deployments = self.deployments.read().await;
            deployments
                .get(&deployment_id)
                .map(|d| d.service_name.clone())
                .ok_or_else(|| CloudDeploymentError::DeploymentFailed {
                    message: format!("Deployment {deployment_id} not found"),
                })?
        };

        self.run_kubectl(&["delete", "deployment", &service_name, "--ignore-not-found"])
            .await?;

        self.deployments.write().await.remove(&deployment_id);
        Ok(())
    }

    async fn get_deployment_status(
        &self,
        deployment_id: Uuid,
    ) -> CloudDeploymentResult<DeploymentInfo> {
        let deployments = self.deployments.read().await;

        deployments.get(&deployment_id).cloned().ok_or_else(|| {
            CloudDeploymentError::DeploymentFailed {
                message: format!("Deployment {deployment_id} not found"),
            }
        })
    }

    async fn list_deployments(&self) -> CloudDeploymentResult<Vec<DeploymentInfo>> {
        let deployments = self.deployments.read().await;
        Ok(deployments.values().cloned().collect())
    }

    async fn get_deployment_logs(
        &self,
        deployment_id: Uuid,
        tail_lines: Option<u32>,
    ) -> CloudDeploymentResult<Vec<String>> {
        let service_name = {
            let deployments = self.deployments.read().await;
            deployments
                .get(&deployment_id)
                .map(|d| d.service_name.clone())
                .ok_or_else(|| CloudDeploymentError::DeploymentFailed {
                    message: format!("Deployment {deployment_id} not found"),
                })?
        };

        let workload = format!("deployment/{service_name}");
        let tail_arg = tail_lines.unwrap_or(100).to_string();
        let output = self
            .run_kubectl(&["logs", &workload, "--tail", &tail_arg])
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect())
    }

    async fn exec_command(
        &self,
        deployment_id: Uuid,
        pod_name: String,
        command: Vec<String>,
    ) -> CloudDeploymentResult<String> {
        {
            let deployments = self.deployments.read().await;
            if !deployments.contains_key(&deployment_id) {
                return Err(CloudDeploymentError::DeploymentFailed {
                    message: format!("Deployment {deployment_id} not found"),
                });
            }
        }
        if command.is_empty() {
            return Err(CloudDeploymentError::OrchestrationError {
                details: "exec_command requires a non-empty command".to_string(),
            });
        }

        let mut cmd = self.kubectl(&["exec", &pod_name, "--"]);
        cmd.args(&command);
        let output = cmd.output().await.map_err(|e| kubectl_spawn_error(&e))?;

        if !output.status.success() {
            return Err(CloudDeploymentError::OrchestrationError {
                details: format!(
                    "kubectl exec failed (exit {:?}): {}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

/// Map a `kubectl` spawn failure to a typed error, distinguishing "the
/// binary isn't installed" from other OS-level launch failures.
fn kubectl_spawn_error(e: &std::io::Error) -> CloudDeploymentError {
    if e.kind() == std::io::ErrorKind::NotFound {
        CloudDeploymentError::OrchestrationError {
            details: "kubectl binary not found in PATH; install kubectl to deploy for real"
                .to_string(),
        }
    } else {
        CloudDeploymentError::OrchestrationError {
            details: format!("failed to launch kubectl: {e}"),
        }
    }
}

/// Build the `Deployment` document, wiring in environment variables, the
/// configured health-check as readiness/liveness probes, and the
/// deployment strategy (fields the previous manifest generator ignored).
fn build_deployment_manifest(config: &DeploymentConfig) -> serde_json::Value {
    let container_port = config.ports.first().map_or(8080, |p| p.container_port);

    let env: Vec<serde_json::Value> = {
        let mut vars: Vec<_> = config.environment.iter().collect();
        vars.sort_by_key(|&(k, _)| k); // deterministic manifest output
        vars.into_iter()
            .map(|(k, v)| serde_json::json!({ "name": k, "value": v }))
            .collect()
    };

    let strategy = match &config.strategy {
        DeploymentStrategy::RollingUpdate {
            max_unavailable,
            max_surge,
        } => serde_json::json!({
            "type": "RollingUpdate",
            "rollingUpdate": { "maxUnavailable": max_unavailable, "maxSurge": max_surge },
        }),
        DeploymentStrategy::Recreate => serde_json::json!({ "type": "Recreate" }),
        // Vanilla Kubernetes Deployments have no native blue-green concept;
        // approximate with a max-surge-only rolling update (old pods stay
        // up until every new pod is ready) rather than silently ignoring
        // the requested strategy.
        DeploymentStrategy::BlueGreen => serde_json::json!({
            "type": "RollingUpdate",
            "rollingUpdate": { "maxUnavailable": 0, "maxSurge": "100%" },
        }),
        // Likewise, native Canary requires a service mesh or a controller
        // like Argo Rollouts; approximate the requested traffic percentage
        // as the fraction of pods surged during a rolling update.
        DeploymentStrategy::Canary {
            traffic_percentage, ..
        } => serde_json::json!({
            "type": "RollingUpdate",
            "rollingUpdate": {
                "maxUnavailable": 0,
                "maxSurge": format!("{}%", traffic_percentage.clamp(1.0, 100.0).round() as i64),
            },
        }),
    };

    let probe = serde_json::json!({
        "httpGet": {
            "path": config.health_check.path,
            "port": config.health_check.port,
        },
        "initialDelaySeconds": config.health_check.initial_delay.as_secs(),
        "periodSeconds": config.health_check.interval.as_secs().max(1),
        "timeoutSeconds": config.health_check.timeout.as_secs().max(1),
        "successThreshold": config.health_check.success_threshold.max(1),
        "failureThreshold": config.health_check.failure_threshold.max(1),
    });

    serde_json::json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": config.service_name,
            "labels": { "app": config.service_name },
        },
        "spec": {
            "replicas": config.replicas,
            "strategy": strategy,
            "selector": { "matchLabels": { "app": config.service_name } },
            "template": {
                "metadata": { "labels": { "app": config.service_name } },
                "spec": {
                    "containers": [{
                        "name": config.service_name,
                        "image": format!("{}:{}", config.image, config.tag),
                        "ports": [{ "containerPort": container_port }],
                        "env": env,
                        "resources": {
                            "requests": {
                                "cpu": format!("{}m", config.resources.cpu_request),
                                "memory": format!("{}Mi", config.resources.memory_request),
                            },
                            "limits": {
                                "cpu": format!("{}m", config.resources.cpu_limit),
                                "memory": format!("{}Mi", config.resources.memory_limit),
                            },
                        },
                        "readinessProbe": probe,
                        "livenessProbe": probe,
                    }],
                },
            },
        },
    })
}

/// Build the `Service` document exposing the deployment.
fn build_service_manifest(config: &DeploymentConfig) -> serde_json::Value {
    let service_port = config.ports.first().map_or(8080, |p| p.service_port);
    let container_port = config.ports.first().map_or(8080, |p| p.container_port);

    serde_json::json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": { "name": format!("{}-service", config.service_name) },
        "spec": {
            "selector": { "app": config.service_name },
            "ports": [{ "port": service_port, "targetPort": container_port }],
            "type": "ClusterIP",
        },
    })
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_request: 100,    // 100m CPU
            cpu_limit: 500,      // 500m CPU
            memory_request: 128, // 128 MiB
            memory_limit: 512,   // 512 MiB
            storage_request: None,
            gpu_request: None,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            path: "/health".to_string(),
            port: 8080,
            initial_delay: Duration::from_secs(30),
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            success_threshold: 1,
            failure_threshold: 3,
        }
    }
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 10,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scale_up_cooldown: Duration::from_secs(300), // 5 minutes
            scale_down_cooldown: Duration::from_secs(600), // 10 minutes
            custom_metrics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config(name: &str) -> DeploymentConfig {
        let mut environment = HashMap::new();
        environment.insert("VOIRS_ENV".to_string(), "test".to_string());

        DeploymentConfig {
            service_name: name.to_string(),
            image: "nginx".to_string(),
            tag: "latest".to_string(),
            replicas: 2,
            resources: ResourceRequirements::default(),
            environment,
            ports: vec![PortConfig {
                name: "http".to_string(),
                container_port: 8080,
                service_port: 80,
                protocol: "HTTP".to_string(),
                load_balancer: None,
            }],
            health_check: HealthCheckConfig::default(),
            strategy: DeploymentStrategy::RollingUpdate {
                max_unavailable: 1,
                max_surge: 1,
            },
            service_mesh: None,
            auto_scaling: None,
        }
    }

    /// Whether the `kubectl` binary is on `PATH` at all.
    async fn kubectl_available() -> bool {
        Command::new("kubectl")
            .arg("version")
            .arg("--client")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Whether `kubectl` can reach a real, live cluster.
    async fn cluster_reachable() -> bool {
        Command::new("kubectl")
            .arg("cluster-info")
            .arg("--request-timeout=2s")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_kubernetes_orchestrator_creation() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);
        let deployments = orchestrator.list_deployments().await.unwrap();
        assert!(deployments.is_empty());
    }

    #[test]
    fn test_kubectl_spawn_error_distinguishes_missing_binary() {
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        let err = kubectl_spawn_error(&not_found);
        assert!(
            matches!(err, CloudDeploymentError::OrchestrationError { ref details } if details.contains("not found"))
        );

        let other = std::io::Error::other("permission denied");
        let err = kubectl_spawn_error(&other);
        assert!(
            matches!(err, CloudDeploymentError::OrchestrationError { ref details } if details.contains("permission denied"))
        );
    }

    #[test]
    fn test_manifest_generation_uses_all_config_fields() {
        let mut config = make_test_config("manifest-test");
        config
            .environment
            .insert("FEATURE_FLAG".to_string(), "enabled".to_string());
        config.health_check.path = "/healthz".to_string();
        config.health_check.port = 9090;

        let deployment_doc = build_deployment_manifest(&config);
        let deployment_yaml = serde_yaml::to_string(&deployment_doc).unwrap();

        // Real values from the config must actually appear in the manifest
        // (previously `environment`, `health_check`, and `strategy` were
        // silently ignored).
        assert!(deployment_yaml.contains("manifest-test"));
        assert!(deployment_yaml.contains("nginx:latest"));
        assert!(deployment_yaml.contains("FEATURE_FLAG"));
        assert!(deployment_yaml.contains("enabled"));
        assert!(deployment_yaml.contains("/healthz"));
        assert!(deployment_yaml.contains("9090"));
        assert!(deployment_yaml.contains("RollingUpdate"));
        assert!(deployment_yaml.contains("maxSurge"));

        let service_doc = build_service_manifest(&config);
        let service_yaml = serde_yaml::to_string(&service_doc).unwrap();
        assert!(service_yaml.contains("manifest-test-service"));
        assert!(service_yaml.contains("8080"));

        // A different config must produce genuinely different output.
        let other_config = make_test_config("other-service");
        let other_yaml = serde_yaml::to_string(&build_deployment_manifest(&other_config)).unwrap();
        assert_ne!(deployment_yaml, other_yaml);
        assert!(other_yaml.contains("other-service"));
        assert!(!other_yaml.contains("manifest-test"));
    }

    #[test]
    fn test_manifest_strategy_variants_are_distinguishable() {
        let mut recreate_config = make_test_config("recreate-svc");
        recreate_config.strategy = DeploymentStrategy::Recreate;
        let recreate_yaml =
            serde_yaml::to_string(&build_deployment_manifest(&recreate_config)).unwrap();
        assert!(recreate_yaml.contains("Recreate"));
        assert!(!recreate_yaml.contains("RollingUpdate"));

        let mut canary_config = make_test_config("canary-svc");
        canary_config.strategy = DeploymentStrategy::Canary {
            traffic_percentage: 25.0,
            evaluation_duration: Duration::from_secs(60),
        };
        let canary_yaml =
            serde_yaml::to_string(&build_deployment_manifest(&canary_config)).unwrap();
        assert!(canary_yaml.contains("25%"));
    }

    #[tokio::test]
    async fn test_generate_manifest_writes_and_returns_real_yaml() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);
        let config = make_test_config("generate-manifest-svc");

        let manifest = orchestrator.generate_manifest(&config).await.unwrap();
        assert!(manifest.contains("generate-manifest-svc"));
        assert!(manifest.contains("kind: Deployment"));
        assert!(manifest.contains("kind: Service"));

        // Real file I/O: written to a real path under the OS temp dir.
        let deployment_id = Uuid::new_v4();
        let path = orchestrator
            .write_manifest_file(deployment_id, "-test", &manifest)
            .await
            .unwrap();
        assert!(path.starts_with(std::env::temp_dir()));
        let on_disk = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(on_disk, manifest);
        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_deployment_config_creation() {
        let config = DeploymentConfig {
            service_name: "test-service".to_string(),
            image: "nginx".to_string(),
            tag: "latest".to_string(),
            replicas: 3,
            resources: ResourceRequirements::default(),
            environment: HashMap::new(),
            ports: vec![PortConfig {
                name: "http".to_string(),
                container_port: 8080,
                service_port: 80,
                protocol: "HTTP".to_string(),
                load_balancer: None,
            }],
            health_check: HealthCheckConfig::default(),
            strategy: DeploymentStrategy::RollingUpdate {
                max_unavailable: 1,
                max_surge: 1,
            },
            service_mesh: None,
            auto_scaling: Some(AutoScalingConfig::default()),
        };

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.replicas, 3);
        assert!(config.auto_scaling.is_some());
    }

    /// `deploy_service` must never fabricate success: on a machine with no
    /// `kubectl` binary, or one where `kubectl` cannot reach a cluster
    /// (this sandbox: binary present, no cluster configured), it must
    /// return a real, typed error instead of a fake `Deploying`/`Deployed`
    /// registry entry. On a machine with a genuinely reachable cluster,
    /// it must actually create the resource and eventually observe it
    /// become ready via real `kubectl get` polling.
    #[tokio::test]
    async fn test_service_deployment_reflects_real_cluster_state() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);
        let config = make_test_config("voirs-test-deployment");

        if !kubectl_available().await {
            let err = orchestrator.deploy_service(config).await.unwrap_err();
            assert!(matches!(
                err,
                CloudDeploymentError::OrchestrationError { .. }
            ));
            return;
        }

        if !cluster_reachable().await {
            let err = orchestrator.deploy_service(config).await.unwrap_err();
            assert!(matches!(
                err,
                CloudDeploymentError::OrchestrationError { .. }
            ));
            // Nothing should have been registered on a failed apply.
            assert!(orchestrator.list_deployments().await.unwrap().is_empty());
            return;
        }

        // A real cluster is reachable: exercise the full real path.
        let deployment_info = orchestrator.deploy_service(config).await.unwrap();
        assert_eq!(deployment_info.service_name, "voirs-test-deployment");
        assert_eq!(deployment_info.status, DeploymentStatus::Deploying);

        let mut final_status = deployment_info.status.clone();
        for _ in 0..15 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let updated = orchestrator
                .get_deployment_status(deployment_info.deployment_id)
                .await
                .unwrap();
            final_status = updated.status;
            if final_status == DeploymentStatus::Deployed {
                break;
            }
        }
        assert_eq!(final_status, DeploymentStatus::Deployed);

        let _ = orchestrator
            .delete_deployment(deployment_info.deployment_id)
            .await;
    }

    #[tokio::test]
    async fn test_deployment_scaling_fails_closed_without_cluster() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);

        // Scaling a deployment id that was never actually created must
        // fail with a real "not found" error, never silently succeed.
        let result = orchestrator.scale_deployment(Uuid::new_v4(), 5).await;
        assert!(matches!(
            result,
            Err(CloudDeploymentError::ScalingFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_deployment_logs_and_exec_require_a_tracked_deployment() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);
        let random_id = Uuid::new_v4();

        let logs_result = orchestrator.get_deployment_logs(random_id, Some(10)).await;
        assert!(matches!(
            logs_result,
            Err(CloudDeploymentError::DeploymentFailed { .. })
        ));

        let exec_result = orchestrator
            .exec_command(random_id, "pod-x".to_string(), vec!["echo".to_string()])
            .await;
        assert!(matches!(
            exec_result,
            Err(CloudDeploymentError::DeploymentFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_deployment_deletion_of_unknown_id_fails_closed() {
        let orchestrator = KubernetesOrchestrator::new(CloudProvider::Kubernetes);

        let status_result = orchestrator.get_deployment_status(Uuid::new_v4()).await;
        assert!(status_result.is_err());

        let delete_result = orchestrator.delete_deployment(Uuid::new_v4()).await;
        assert!(delete_result.is_err());
    }

    #[tokio::test]
    async fn test_resource_requirements_defaults() {
        let resources = ResourceRequirements::default();
        assert_eq!(resources.cpu_request, 100);
        assert_eq!(resources.cpu_limit, 500);
        assert_eq!(resources.memory_request, 128);
        assert_eq!(resources.memory_limit, 512);
    }

    #[tokio::test]
    async fn test_auto_scaling_config_defaults() {
        let auto_scaling = AutoScalingConfig::default();
        assert_eq!(auto_scaling.min_replicas, 1);
        assert_eq!(auto_scaling.max_replicas, 10);
        assert_eq!(auto_scaling.target_cpu_utilization, 0.7);
        assert_eq!(auto_scaling.target_memory_utilization, 0.8);
    }
}
