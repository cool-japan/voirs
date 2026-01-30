//! Cloud Deployment and Distributed Synthesis
//!
//! This module provides infrastructure for deploying singing synthesis workloads
//! to cloud environments with distributed processing capabilities.

use crate::score::{KeySignature, Mode, MusicalScore, Note, TimeSignature};
use crate::types::{NoteEvent, SingingRequest, SingingResponse, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Cloud deployment manager
///
/// Manages distributed synthesis across multiple cloud nodes
#[derive(Debug)]
pub struct CloudDeploymentManager {
    /// Configuration
    config: CloudConfig,
    /// Active worker nodes
    workers: Arc<RwLock<Vec<WorkerNode>>>,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Request queue
    request_queue: Arc<RwLock<Vec<SynthesisJob>>>,
}

/// Cloud deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Cloud provider (e.g., "aws", "gcp", "azure", "local")
    pub provider: String,
    /// Region for deployment
    pub region: String,
    /// Maximum number of worker nodes
    pub max_workers: usize,
    /// Minimum number of worker nodes
    pub min_workers: usize,
    /// Enable auto-scaling
    pub auto_scaling: bool,
    /// Target CPU utilization for auto-scaling (0.0-1.0)
    pub target_cpu_utilization: f32,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Synthesis quality tier
    pub quality_tier: QualityTier,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            provider: "local".to_string(),
            region: "us-east-1".to_string(),
            max_workers: 10,
            min_workers: 1,
            auto_scaling: true,
            target_cpu_utilization: 0.7,
            enable_load_balancing: true,
            quality_tier: QualityTier::Standard,
        }
    }
}

/// Quality tier for synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTier {
    /// Economic tier - faster, lower quality
    Economy,
    /// Standard tier - balanced quality/speed
    Standard,
    /// Premium tier - highest quality
    Premium,
}

/// Worker node in cloud cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique worker ID
    pub id: String,
    /// Worker status
    pub status: WorkerStatus,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f32,
    /// Number of active jobs
    pub active_jobs: usize,
    /// Worker capacity
    pub capacity: usize,
    /// Endpoint URL
    pub endpoint: String,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is available
    Available,
    /// Worker is busy
    Busy,
    /// Worker is offline
    Offline,
    /// Worker is initializing
    Initializing,
}

/// Synthesis job in distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisJob {
    /// Job ID
    pub id: String,
    /// Synthesis request
    pub request: SingingRequest,
    /// Job priority (higher = more priority)
    pub priority: u32,
    /// Assigned worker ID
    pub assigned_worker: Option<String>,
    /// Job status
    pub status: JobStatus,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Started timestamp
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Completed timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is queued
    Queued,
    /// Job is running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
}

/// Load balancer for distributing jobs
#[derive(Debug)]
pub struct LoadBalancer {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded worker first
    LeastLoaded,
    /// Random assignment
    Random,
    /// Priority-based assignment
    Priority,
}

impl CloudDeploymentManager {
    /// Create new cloud deployment manager
    pub fn new(config: CloudConfig) -> Self {
        Self {
            config: config.clone(),
            workers: Arc::new(RwLock::new(Vec::new())),
            load_balancer: LoadBalancer::new(LoadBalancingStrategy::LeastLoaded),
            request_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize cloud deployment
    ///
    /// Sets up initial worker nodes and prepares for synthesis
    pub async fn initialize(&mut self) -> crate::Result<()> {
        // Initialize minimum number of workers
        for i in 0..self.config.min_workers {
            let worker = WorkerNode {
                id: format!("worker-{}", i),
                status: WorkerStatus::Initializing,
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                active_jobs: 0,
                capacity: 10,
                endpoint: format!("http://worker-{}.cloud.local:8080", i),
            };

            self.workers.write().await.push(worker);
        }

        // Mark workers as available
        for worker in self.workers.write().await.iter_mut() {
            worker.status = WorkerStatus::Available;
        }

        Ok(())
    }

    /// Perform health check on all workers
    ///
    /// # Returns
    /// Health check results for each worker
    pub async fn health_check(&self) -> Vec<WorkerHealthStatus> {
        let workers = self.workers.read().await;
        let mut health_statuses = Vec::with_capacity(workers.len());

        for worker in workers.iter() {
            let health = WorkerHealthStatus {
                worker_id: worker.id.clone(),
                is_healthy: worker.status != WorkerStatus::Offline,
                cpu_utilization: worker.cpu_utilization,
                memory_utilization: worker.memory_utilization,
                active_jobs: worker.active_jobs,
                response_time_ms: Self::simulate_response_time(worker),
                last_heartbeat: chrono::Utc::now(),
            };
            health_statuses.push(health);
        }

        health_statuses
    }

    /// Simulate response time based on worker load
    fn simulate_response_time(worker: &WorkerNode) -> f64 {
        // Base response time + load-dependent delay
        50.0 + (worker.cpu_utilization * 200.0) as f64
    }

    /// Cancel a running job
    pub async fn cancel_job(&mut self, job_id: &str) -> crate::Result<bool> {
        let mut queue = self.request_queue.write().await;

        if let Some(job) = queue.iter_mut().find(|j| j.id == job_id) {
            if job.status == JobStatus::Queued || job.status == JobStatus::Running {
                job.status = JobStatus::Cancelled;

                // Release worker resources
                if let Some(worker_id) = &job.assigned_worker {
                    let mut workers = self.workers.write().await;
                    if let Some(worker) = workers.iter_mut().find(|w| &w.id == worker_id) {
                        worker.active_jobs = worker.active_jobs.saturating_sub(1);
                        worker.cpu_utilization =
                            (worker.active_jobs as f32 / worker.capacity as f32).min(1.0);
                        if worker.status == WorkerStatus::Busy
                            && worker.active_jobs < worker.capacity
                        {
                            worker.status = WorkerStatus::Available;
                        }
                    }
                }

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get detailed metrics for monitoring
    pub async fn get_detailed_metrics(&self) -> DetailedMetrics {
        let workers = self.workers.read().await;
        let queue = self.request_queue.read().await;

        let total_capacity: usize = workers.iter().map(|w| w.capacity).sum();
        let total_active_jobs: usize = workers.iter().map(|w| w.active_jobs).sum();
        let utilization_percentage = if total_capacity > 0 {
            (total_active_jobs as f32 / total_capacity as f32) * 100.0
        } else {
            0.0
        };

        let completed_jobs = queue
            .iter()
            .filter(|j| j.status == JobStatus::Completed)
            .count();
        let failed_jobs = queue
            .iter()
            .filter(|j| j.status == JobStatus::Failed)
            .count();
        let success_rate = if completed_jobs + failed_jobs > 0 {
            (completed_jobs as f32 / (completed_jobs + failed_jobs) as f32) * 100.0
        } else {
            100.0
        };

        // Calculate average job duration
        let completed_job_durations: Vec<f64> = queue
            .iter()
            .filter(|j| {
                j.status == JobStatus::Completed
                    && j.started_at.is_some()
                    && j.completed_at.is_some()
            })
            .filter_map(|j| {
                let duration = j.completed_at?.signed_duration_since(j.started_at?);
                Some(duration.num_milliseconds() as f64)
            })
            .collect();

        let avg_job_duration_ms = if !completed_job_durations.is_empty() {
            completed_job_durations.iter().sum::<f64>() / completed_job_durations.len() as f64
        } else {
            0.0
        };

        DetailedMetrics {
            total_workers: workers.len(),
            healthy_workers: workers
                .iter()
                .filter(|w| w.status != WorkerStatus::Offline)
                .count(),
            total_capacity,
            cluster_utilization_percentage: utilization_percentage,
            queued_jobs: queue
                .iter()
                .filter(|j| j.status == JobStatus::Queued)
                .count(),
            running_jobs: queue
                .iter()
                .filter(|j| j.status == JobStatus::Running)
                .count(),
            completed_jobs,
            failed_jobs,
            success_rate_percentage: success_rate,
            average_job_duration_ms: avg_job_duration_ms,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Simulate job completion (for testing/demo purposes)
    pub async fn simulate_job_completion(&mut self, job_id: &str) -> crate::Result<()> {
        let mut queue = self.request_queue.write().await;

        if let Some(job) = queue.iter_mut().find(|j| j.id == job_id) {
            if job.status == JobStatus::Running {
                job.status = JobStatus::Completed;
                job.completed_at = Some(chrono::Utc::now());

                // Release worker resources
                if let Some(worker_id) = &job.assigned_worker {
                    let mut workers = self.workers.write().await;
                    if let Some(worker) = workers.iter_mut().find(|w| &w.id == worker_id) {
                        worker.active_jobs = worker.active_jobs.saturating_sub(1);
                        worker.cpu_utilization =
                            (worker.active_jobs as f32 / worker.capacity as f32).min(1.0);
                        if worker.status == WorkerStatus::Busy
                            && worker.active_jobs < worker.capacity
                        {
                            worker.status = WorkerStatus::Available;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Submit synthesis job to cloud cluster
    ///
    /// # Arguments
    /// * `request` - Singing synthesis request
    /// * `priority` - Job priority (0-100)
    ///
    /// # Returns
    /// Job ID for tracking
    pub async fn submit_job(
        &mut self,
        request: SingingRequest,
        priority: u32,
    ) -> crate::Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();

        let job = SynthesisJob {
            id: job_id.clone(),
            request,
            priority,
            assigned_worker: None,
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
        };

        // Add to queue
        self.request_queue.write().await.push(job);

        // Trigger job scheduling
        self.schedule_jobs().await?;

        Ok(job_id)
    }

    /// Schedule queued jobs to available workers
    async fn schedule_jobs(&mut self) -> crate::Result<()> {
        let mut queue = self.request_queue.write().await;
        let mut workers = self.workers.write().await;

        // Sort queue by priority (highest first)
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));

        for job in queue.iter_mut() {
            if job.status != JobStatus::Queued {
                continue;
            }

            // Find available worker using load balancer
            if let Some(worker_idx) = self
                .load_balancer
                .select_worker(&workers, &self.config)
                .await
            {
                let worker = &mut workers[worker_idx];

                // Assign job to worker
                job.assigned_worker = Some(worker.id.clone());
                job.status = JobStatus::Running;
                job.started_at = Some(chrono::Utc::now());

                worker.active_jobs += 1;
                worker.cpu_utilization =
                    (worker.active_jobs as f32 / worker.capacity as f32).min(1.0);

                if worker.active_jobs >= worker.capacity {
                    worker.status = WorkerStatus::Busy;
                }
            }
        }

        // Drop locks before auto-scaling
        drop(queue);
        let workers_snapshot: Vec<WorkerNode> = workers.iter().cloned().collect();
        drop(workers);

        // Check if auto-scaling is needed
        if self.config.auto_scaling {
            self.check_auto_scaling(&workers_snapshot).await?;
        }

        Ok(())
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        let queue = self.request_queue.read().await;
        queue.iter().find(|j| j.id == job_id).map(|j| j.status)
    }

    /// Get synthesis result
    pub async fn get_result(&self, job_id: &str) -> crate::Result<Option<SingingResponse>> {
        let queue = self.request_queue.read().await;

        if let Some(job) = queue.iter().find(|j| j.id == job_id) {
            if job.status == JobStatus::Completed {
                // In production, this would fetch the actual result from worker
                // For now, return a placeholder response
                Ok(Some(SingingResponse::default()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Check and perform auto-scaling
    async fn check_auto_scaling(&mut self, workers: &[WorkerNode]) -> crate::Result<()> {
        let avg_cpu = workers.iter().map(|w| w.cpu_utilization).sum::<f32>() / workers.len() as f32;

        if avg_cpu > self.config.target_cpu_utilization && workers.len() < self.config.max_workers {
            // Scale up
            tracing::info!("Auto-scaling up: CPU utilization {:.2}", avg_cpu);
        } else if avg_cpu < self.config.target_cpu_utilization * 0.5
            && workers.len() > self.config.min_workers
        {
            // Scale down
            tracing::info!("Auto-scaling down: CPU utilization {:.2}", avg_cpu);
        }

        Ok(())
    }

    /// Get cluster statistics
    pub async fn get_cluster_stats(&self) -> ClusterStats {
        let workers = self.workers.read().await;
        let queue = self.request_queue.read().await;

        ClusterStats {
            total_workers: workers.len(),
            available_workers: workers
                .iter()
                .filter(|w| w.status == WorkerStatus::Available)
                .count(),
            busy_workers: workers
                .iter()
                .filter(|w| w.status == WorkerStatus::Busy)
                .count(),
            queued_jobs: queue
                .iter()
                .filter(|j| j.status == JobStatus::Queued)
                .count(),
            running_jobs: queue
                .iter()
                .filter(|j| j.status == JobStatus::Running)
                .count(),
            completed_jobs: queue
                .iter()
                .filter(|j| j.status == JobStatus::Completed)
                .count(),
            average_cpu_utilization: workers.iter().map(|w| w.cpu_utilization).sum::<f32>()
                / workers.len().max(1) as f32,
            average_memory_utilization: workers.iter().map(|w| w.memory_utilization).sum::<f32>()
                / workers.len().max(1) as f32,
        }
    }
}

/// Cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Total number of workers
    pub total_workers: usize,
    /// Number of available workers
    pub available_workers: usize,
    /// Number of busy workers
    pub busy_workers: usize,
    /// Number of queued jobs
    pub queued_jobs: usize,
    /// Number of running jobs
    pub running_jobs: usize,
    /// Number of completed jobs
    pub completed_jobs: usize,
    /// Average CPU utilization across cluster
    pub average_cpu_utilization: f32,
    /// Average memory utilization across cluster
    pub average_memory_utilization: f32,
}

/// Worker health status for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealthStatus {
    /// Worker ID
    pub worker_id: String,
    /// Is worker healthy
    pub is_healthy: bool,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f32,
    /// Number of active jobs
    pub active_jobs: usize,
    /// Response time in milliseconds
    pub response_time_ms: f64,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Detailed metrics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMetrics {
    /// Total number of workers
    pub total_workers: usize,
    /// Number of healthy workers
    pub healthy_workers: usize,
    /// Total cluster capacity
    pub total_capacity: usize,
    /// Cluster utilization percentage
    pub cluster_utilization_percentage: f32,
    /// Number of queued jobs
    pub queued_jobs: usize,
    /// Number of running jobs
    pub running_jobs: usize,
    /// Number of completed jobs
    pub completed_jobs: usize,
    /// Number of failed jobs
    pub failed_jobs: usize,
    /// Success rate percentage
    pub success_rate_percentage: f32,
    /// Average job duration in milliseconds
    pub average_job_duration_ms: f64,
    /// Timestamp of metrics collection
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self { strategy }
    }

    /// Select worker for job assignment
    async fn select_worker(&self, workers: &[WorkerNode], _config: &CloudConfig) -> Option<usize> {
        match self.strategy {
            LoadBalancingStrategy::LeastLoaded => {
                // Find worker with lowest utilization
                workers
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.status == WorkerStatus::Available)
                    .min_by(|(_, a), (_, b)| {
                        a.cpu_utilization.partial_cmp(&b.cpu_utilization).unwrap()
                    })
                    .map(|(idx, _)| idx)
            }
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (first available)
                workers
                    .iter()
                    .enumerate()
                    .find(|(_, w)| w.status == WorkerStatus::Available)
                    .map(|(idx, _)| idx)
            }
            LoadBalancingStrategy::Random => {
                // Random available worker
                use scirs2_core::random::Rng;
                let mut rng = scirs2_core::random::thread_rng();

                let available: Vec<usize> = workers
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.status == WorkerStatus::Available)
                    .map(|(idx, _)| idx)
                    .collect();

                if available.is_empty() {
                    None
                } else {
                    let idx = rng.gen_range(0..available.len());
                    Some(available[idx])
                }
            }
            LoadBalancingStrategy::Priority => {
                // Priority-based (use least loaded strategy)
                workers
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.status == WorkerStatus::Available)
                    .min_by(|(_, a), (_, b)| {
                        a.cpu_utilization.partial_cmp(&b.cpu_utilization).unwrap()
                    })
                    .map(|(idx, _)| idx)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NoteEvent, VoiceCharacteristics};

    #[tokio::test]
    async fn test_cloud_manager_creation() {
        let config = CloudConfig::default();
        let manager = CloudDeploymentManager::new(config);

        assert_eq!(manager.config.provider, "local");
    }

    #[tokio::test]
    async fn test_cloud_initialization() {
        let config = CloudConfig {
            min_workers: 3,
            ..Default::default()
        };

        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let workers = manager.workers.read().await;
        assert_eq!(workers.len(), 3);
        assert!(workers.iter().all(|w| w.status == WorkerStatus::Available));
    }

    #[tokio::test]
    async fn test_job_submission() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let score = MusicalScore {
            title: "Test Song".to_string(),
            composer: "Test Composer".to_string(),
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            tempo: 120.0,
            notes: vec![],
            lyrics: None,
            metadata: HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let request = SingingRequest {
            score,
            voice: VoiceCharacteristics::default(),
            technique: crate::techniques::SingingTechnique::default(),
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality: crate::types::QualitySettings::default(),
        };

        let job_id = manager.submit_job(request, 10).await.unwrap();

        assert!(!job_id.is_empty());

        // Check job status
        let status = manager.get_job_status(&job_id).await;
        assert!(status.is_some());
    }

    #[tokio::test]
    async fn test_cluster_stats() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let stats = manager.get_cluster_stats().await;

        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.available_workers, 1);
        assert_eq!(stats.queued_jobs, 0);
    }

    #[test]
    fn test_quality_tier() {
        let economy = QualityTier::Economy;
        let standard = QualityTier::Standard;
        let premium = QualityTier::Premium;

        assert_ne!(economy, standard);
        assert_ne!(standard, premium);
    }

    #[test]
    fn test_worker_status() {
        let status = WorkerStatus::Available;
        assert_eq!(status, WorkerStatus::Available);

        let status2 = WorkerStatus::Busy;
        assert_ne!(status, status2);
    }

    #[test]
    fn test_job_status() {
        let status = JobStatus::Queued;
        assert_eq!(status, JobStatus::Queued);

        let status2 = JobStatus::Running;
        assert_ne!(status, status2);
    }

    #[tokio::test]
    async fn test_load_balancer_least_loaded() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);

        let workers = vec![
            WorkerNode {
                id: "w1".to_string(),
                status: WorkerStatus::Available,
                cpu_utilization: 0.5,
                memory_utilization: 0.4,
                active_jobs: 5,
                capacity: 10,
                endpoint: "http://w1".to_string(),
            },
            WorkerNode {
                id: "w2".to_string(),
                status: WorkerStatus::Available,
                cpu_utilization: 0.2,
                memory_utilization: 0.3,
                active_jobs: 2,
                capacity: 10,
                endpoint: "http://w2".to_string(),
            },
        ];

        let config = CloudConfig::default();
        let selected = lb.select_worker(&workers, &config).await;

        // Should select worker with lowest utilization (w2)
        assert_eq!(selected, Some(1));
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let health_statuses = manager.health_check().await;

        assert_eq!(health_statuses.len(), 1);
        assert!(health_statuses[0].is_healthy);
        assert!(health_statuses[0].response_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let score = MusicalScore {
            title: "Test Song".to_string(),
            composer: "Test Composer".to_string(),
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            tempo: 120.0,
            notes: vec![],
            lyrics: None,
            metadata: HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let request = SingingRequest {
            score,
            voice: VoiceCharacteristics::default(),
            technique: crate::techniques::SingingTechnique::default(),
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality: crate::types::QualitySettings::default(),
        };

        let job_id = manager.submit_job(request, 10).await.unwrap();

        // Cancel the job
        let cancelled = manager.cancel_job(&job_id).await.unwrap();
        assert!(cancelled);

        // Check job status
        let status = manager.get_job_status(&job_id).await;
        assert_eq!(status, Some(JobStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_detailed_metrics() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let metrics = manager.get_detailed_metrics().await;

        assert_eq!(metrics.total_workers, 1);
        assert_eq!(metrics.healthy_workers, 1);
        assert!(metrics.total_capacity > 0);
        assert_eq!(metrics.cluster_utilization_percentage, 0.0);
        assert_eq!(metrics.success_rate_percentage, 100.0);
    }

    #[tokio::test]
    async fn test_job_simulation() {
        let config = CloudConfig::default();
        let mut manager = CloudDeploymentManager::new(config);
        manager.initialize().await.unwrap();

        let score = MusicalScore {
            title: "Test Song".to_string(),
            composer: "Test Composer".to_string(),
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            tempo: 120.0,
            notes: vec![],
            lyrics: None,
            metadata: HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let request = SingingRequest {
            score,
            voice: VoiceCharacteristics::default(),
            technique: crate::techniques::SingingTechnique::default(),
            effects: vec![],
            sample_rate: 44100,
            target_duration: None,
            quality: crate::types::QualitySettings::default(),
        };

        let job_id = manager.submit_job(request, 10).await.unwrap();

        // Simulate job completion
        manager.simulate_job_completion(&job_id).await.unwrap();

        // Check job status
        let status = manager.get_job_status(&job_id).await;
        assert_eq!(status, Some(JobStatus::Completed));

        // Check metrics
        let metrics = manager.get_detailed_metrics().await;
        assert_eq!(metrics.completed_jobs, 1);
    }
}
