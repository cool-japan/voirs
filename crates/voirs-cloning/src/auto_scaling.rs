//! Auto-scaling System for Dynamic GPU Resource Allocation
//!
//! This module provides intelligent auto-scaling capabilities that dynamically allocate
//! and deallocate GPU resources based on workload demand, performance requirements,
//! and cost optimization. It works in conjunction with the load balancing system.

use crate::{
    gpu_acceleration::{GpuAccelerationConfig, GpuAccelerator, GpuDeviceType},
    load_balancing::{GpuDeviceInfo, GpuLoadBalancer, LoadBalancingStats},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Auto-scaling strategies for resource allocation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AutoScalingStrategy {
    /// Conservative scaling - slower response, lower cost
    Conservative,
    /// Balanced scaling - moderate response time and cost
    Balanced,
    /// Aggressive scaling - fast response, higher cost
    Aggressive,
    /// Predictive scaling based on usage patterns
    Predictive,
    /// Custom strategy with user-defined parameters
    Custom {
        scale_up_threshold: f32,
        scale_down_threshold: f32,
        cooldown_period_secs: u64,
        prediction_weight: f32,
    },
}

impl Default for AutoScalingStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Scaling triggers for auto-scaling decisions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalingTrigger {
    /// CPU/GPU utilization threshold
    Utilization { threshold: f32, duration_secs: u64 },
    /// Memory usage threshold
    Memory { threshold: f32, duration_secs: u64 },
    /// Queue length threshold
    QueueLength {
        threshold: usize,
        duration_secs: u64,
    },
    /// Response time threshold
    Latency {
        threshold: Duration,
        duration_secs: u64,
    },
    /// Throughput threshold (operations per second)
    Throughput { threshold: f32, duration_secs: u64 },
    /// Predicted demand based on historical patterns
    PredictedDemand { confidence_threshold: f32 },
    /// Custom metric threshold
    Custom {
        metric_name: String,
        threshold: f32,
        duration_secs: u64,
    },
}

/// Auto-scaling configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Auto-scaling strategy
    pub strategy: AutoScalingStrategy,
    /// Minimum number of GPU instances
    pub min_instances: usize,
    /// Maximum number of GPU instances
    pub max_instances: usize,
    /// Enable auto-scaling
    pub enabled: bool,
    /// Scale-up triggers
    pub scale_up_triggers: Vec<ScalingTrigger>,
    /// Scale-down triggers
    pub scale_down_triggers: Vec<ScalingTrigger>,
    /// Cooldown period after scaling operations (seconds)
    pub cooldown_period_secs: u64,
    /// Monitoring interval (seconds)
    pub monitoring_interval_secs: u64,
    /// Enable predictive scaling
    pub enable_predictive_scaling: bool,
    /// Historical data window for predictions (hours)
    pub prediction_window_hours: u64,
    /// Cost optimization priority (0.0 = performance, 1.0 = cost)
    pub cost_optimization_weight: f32,
    /// Enable pre-emptive scaling
    pub enable_preemptive_scaling: bool,
    /// Time ahead to scale for predicted demand (seconds)
    pub preemptive_scale_ahead_secs: u64,
    /// Maximum scaling events per hour
    pub max_scaling_events_per_hour: usize,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            strategy: AutoScalingStrategy::default(),
            min_instances: 1,
            max_instances: 8,
            enabled: true,
            scale_up_triggers: vec![
                ScalingTrigger::Utilization {
                    threshold: 0.8,
                    duration_secs: 300,
                },
                ScalingTrigger::QueueLength {
                    threshold: 20,
                    duration_secs: 180,
                },
                ScalingTrigger::Latency {
                    threshold: Duration::from_secs(5),
                    duration_secs: 240,
                },
            ],
            scale_down_triggers: vec![
                ScalingTrigger::Utilization {
                    threshold: 0.3,
                    duration_secs: 600,
                },
                ScalingTrigger::QueueLength {
                    threshold: 5,
                    duration_secs: 450,
                },
            ],
            cooldown_period_secs: 300,
            monitoring_interval_secs: 60,
            enable_predictive_scaling: true,
            prediction_window_hours: 24,
            cost_optimization_weight: 0.3,
            enable_preemptive_scaling: false,
            preemptive_scale_ahead_secs: 600,
            max_scaling_events_per_hour: 10,
        }
    }
}

/// Scaling decision and reasoning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalingDecision {
    /// Decision timestamp
    pub timestamp: SystemTime,
    /// Scaling action
    pub action: ScalingAction,
    /// Number of instances to change
    pub instance_delta: i32,
    /// Target instance count after scaling
    pub target_instances: usize,
    /// Triggers that caused this decision
    pub triggered_by: Vec<ScalingTrigger>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Expected impact
    pub expected_impact: ExpectedImpact,
    /// Cost impact
    pub cost_impact: CostImpact,
}

/// Scaling actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up (add resources)
    ScaleUp,
    /// Scale down (remove resources)
    ScaleDown,
    /// No action needed
    NoAction,
    /// Scale up preemptively based on prediction
    PreemptiveScaleUp,
    /// Scale down based on low predicted demand
    PreemptiveScaleDown,
}

/// Expected impact of scaling decision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedImpact {
    /// Expected utilization change
    pub utilization_change: f32,
    /// Expected latency change
    pub latency_change: Duration,
    /// Expected throughput change (ops/sec)
    pub throughput_change: f32,
    /// Expected queue length change
    pub queue_length_change: i32,
    /// Confidence in predictions (0.0 to 1.0)
    pub confidence: f32,
}

/// Cost impact analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostImpact {
    /// Cost change per hour
    pub hourly_cost_change: f32,
    /// ROI score based on performance/cost
    pub roi_score: f32,
    /// Efficiency score
    pub efficiency_score: f32,
}

/// GPU instance information for scaling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalableGpuInstance {
    /// Instance ID
    pub instance_id: String,
    /// Device ID
    pub device_id: usize,
    /// Device type
    pub device_type: GpuDeviceType,
    /// Instance state
    pub state: InstanceState,
    /// Launch time
    pub launched_at: SystemTime,
    /// Last activity time
    pub last_activity: SystemTime,
    /// Cost per hour
    pub hourly_cost: f32,
    /// Performance tier
    pub performance_tier: PerformanceTier,
    /// Health status
    pub health_status: InstanceHealth,
}

/// GPU instance state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstanceState {
    /// Instance is initializing
    Initializing,
    /// Instance is running and available
    Running,
    /// Instance is terminating
    Terminating,
    /// Instance is stopped
    Stopped,
    /// Instance has failed
    Failed,
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerformanceTier {
    /// Basic performance GPU
    Basic,
    /// Standard performance GPU
    Standard,
    /// High performance GPU
    High,
    /// Premium performance GPU
    Premium,
}

/// Instance health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstanceHealth {
    /// Instance is healthy
    Healthy,
    /// Instance has warnings
    Warning,
    /// Instance is unhealthy
    Unhealthy,
    /// Health status unknown
    Unknown,
}

/// Workload prediction based on historical data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadPrediction {
    /// Prediction timestamp
    pub timestamp: SystemTime,
    /// Time range for prediction
    pub prediction_window: Duration,
    /// Predicted utilization (0.0 to 1.0)
    pub predicted_utilization: f32,
    /// Predicted queue length
    pub predicted_queue_length: usize,
    /// Predicted throughput (ops/sec)
    pub predicted_throughput: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Historical data points used
    pub data_points: usize,
    /// Seasonal patterns detected
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

/// Seasonal pattern in workload
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f32,
    /// Pattern period
    pub period: Duration,
    /// Pattern phase offset
    pub phase_offset: Duration,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Daily pattern (24-hour cycle)
    Daily,
    /// Weekly pattern (7-day cycle)
    Weekly,
    /// Monthly pattern
    Monthly,
    /// Custom pattern
    Custom,
}

/// Auto-scaling statistics and metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoScalingStats {
    /// Total scaling events
    pub total_scaling_events: u64,
    /// Scale-up events
    pub scale_up_events: u64,
    /// Scale-down events
    pub scale_down_events: u64,
    /// Average instance count
    pub avg_instance_count: f32,
    /// Peak instance count
    pub peak_instance_count: usize,
    /// Minimum instance count
    pub min_instance_count: usize,
    /// Total cost over time period
    pub total_cost: f32,
    /// Cost per operation
    pub cost_per_operation: f32,
    /// Average scaling response time
    pub avg_scaling_response_time: Duration,
    /// Prediction accuracy (0.0 to 1.0)
    pub prediction_accuracy: f32,
    /// Resource utilization efficiency (0.0 to 1.0)
    pub utilization_efficiency: f32,
    /// Scaling decision accuracy (0.0 to 1.0)
    pub decision_accuracy: f32,
}

/// Historical metrics for pattern analysis
#[derive(Debug, Clone)]
struct MetricsSnapshot {
    timestamp: SystemTime,
    utilization: f32,
    queue_length: usize,
    latency: Duration,
    throughput: f32,
    instance_count: usize,
    cost: f32,
}

/// Auto-scaling engine
pub struct AutoScaler {
    /// Configuration
    config: AutoScalingConfig,
    /// Current GPU instances
    instances: Arc<RwLock<HashMap<String, ScalableGpuInstance>>>,
    /// Load balancer reference
    load_balancer: Option<Arc<GpuLoadBalancer>>,
    /// Scaling history
    scaling_history: Arc<RwLock<VecDeque<ScalingDecision>>>,
    /// Metrics history for prediction
    metrics_history: Arc<RwLock<VecDeque<MetricsSnapshot>>>,
    /// Auto-scaling statistics
    stats: Arc<RwLock<AutoScalingStats>>,
    /// Last scaling time
    last_scaling_time: Arc<RwLock<SystemTime>>,
    /// Monitoring active flag
    monitoring_active: Arc<RwLock<bool>>,
    /// Scaling events counter for rate limiting
    scaling_events_count: Arc<RwLock<VecDeque<SystemTime>>>,
}

impl AutoScaler {
    /// Create new auto-scaler
    pub fn new(config: AutoScalingConfig) -> Self {
        Self {
            config,
            instances: Arc::new(RwLock::new(HashMap::new())),
            load_balancer: None,
            scaling_history: Arc::new(RwLock::new(VecDeque::new())),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(AutoScalingStats::default())),
            last_scaling_time: Arc::new(RwLock::new(SystemTime::now())),
            monitoring_active: Arc::new(RwLock::new(false)),
            scaling_events_count: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(AutoScalingConfig::default())
    }

    /// Set load balancer reference for integration
    pub fn set_load_balancer(&mut self, load_balancer: Arc<GpuLoadBalancer>) {
        self.load_balancer = Some(load_balancer);
    }

    /// Start auto-scaling monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        {
            let mut active = self.monitoring_active.write().unwrap();
            *active = true;
        }

        // Initialize minimum instances
        self.ensure_minimum_instances().await?;

        // Start monitoring loop
        self.start_monitoring_loop().await;

        tracing::info!("Auto-scaling monitoring started");
        Ok(())
    }

    /// Stop auto-scaling monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        {
            let mut active = self.monitoring_active.write().unwrap();
            *active = false;
        }

        tracing::info!("Auto-scaling monitoring stopped");
        Ok(())
    }

    /// Start monitoring loop
    async fn start_monitoring_loop(&self) {
        let instances = Arc::clone(&self.instances);
        let load_balancer = self.load_balancer.clone();
        let scaling_history = Arc::clone(&self.scaling_history);
        let metrics_history = Arc::clone(&self.metrics_history);
        let stats = Arc::clone(&self.stats);
        let last_scaling_time = Arc::clone(&self.last_scaling_time);
        let monitoring_active = Arc::clone(&self.monitoring_active);
        let scaling_events_count = Arc::clone(&self.scaling_events_count);
        let config = self.config.clone();

        tokio::spawn(async move {
            let monitoring_interval = Duration::from_secs(config.monitoring_interval_secs);

            while *monitoring_active.read().unwrap() {
                tokio::time::sleep(monitoring_interval).await;

                // Collect current metrics
                if let Some(ref lb) = load_balancer {
                    let lb_stats = lb.get_statistics();
                    let device_info = lb.get_device_info();

                    let snapshot = Self::create_metrics_snapshot(&lb_stats, &device_info);

                    // Store metrics history
                    {
                        let mut history = metrics_history.write().unwrap();
                        history.push_back(snapshot.clone());

                        // Keep only recent history
                        let max_history_size = (config.prediction_window_hours * 60) as usize; // One sample per minute
                        while history.len() > max_history_size {
                            history.pop_front();
                        }
                    }

                    // Check if scaling decision is needed
                    if let Ok(decision) = Self::evaluate_scaling_decision(
                        &config,
                        &snapshot,
                        &metrics_history,
                        &last_scaling_time,
                        &scaling_events_count,
                    )
                    .await
                    {
                        if decision.action != ScalingAction::NoAction {
                            // Execute scaling decision
                            if let Ok(_) = Self::execute_scaling_decision(
                                &config,
                                &decision,
                                &instances,
                                &load_balancer,
                            )
                            .await
                            {
                                // Record scaling decision
                                {
                                    let mut history = scaling_history.write().unwrap();
                                    history.push_back(decision.clone());

                                    // Keep only recent history
                                    while history.len() > 100 {
                                        history.pop_front();
                                    }
                                }

                                // Update statistics
                                Self::update_scaling_stats(&stats, &decision).await;

                                // Update last scaling time
                                {
                                    let mut last_time = last_scaling_time.write().unwrap();
                                    *last_time = SystemTime::now();
                                }

                                tracing::info!("Executed scaling decision: {:?}", decision.action);
                            }
                        }
                    }
                }
            }
        });
    }

    /// Ensure minimum number of instances are running
    async fn ensure_minimum_instances(&self) -> Result<()> {
        let current_count = self.instances.read().unwrap().len();

        if current_count < self.config.min_instances {
            let instances_needed = self.config.min_instances - current_count;

            for _ in 0..instances_needed {
                self.launch_instance().await?;
            }

            tracing::info!(
                "Launched {} instances to meet minimum requirement",
                instances_needed
            );
        }

        Ok(())
    }

    /// Launch new GPU instance
    async fn launch_instance(&self) -> Result<String> {
        let instance_id = uuid::Uuid::new_v4().to_string();
        let device_id = self.get_next_available_device_id();

        // Create GPU accelerator for new instance
        let gpu_config = GpuAccelerationConfig {
            enabled: true,
            device_id,
            auto_fallback: false,
            ..Default::default()
        };

        let _accelerator = GpuAccelerator::new(gpu_config)?;

        let instance = ScalableGpuInstance {
            instance_id: instance_id.clone(),
            device_id,
            device_type: GpuDeviceType::Cuda, // Default to CUDA
            state: InstanceState::Initializing,
            launched_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            hourly_cost: self.calculate_instance_cost(PerformanceTier::Standard),
            performance_tier: PerformanceTier::Standard,
            health_status: InstanceHealth::Healthy,
        };

        {
            let mut instances = self.instances.write().unwrap();
            instances.insert(instance_id.clone(), instance);
        }

        // Simulate initialization time
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Update instance state to running
        {
            let mut instances = self.instances.write().unwrap();
            if let Some(instance) = instances.get_mut(&instance_id) {
                instance.state = InstanceState::Running;
            }
        }

        tracing::info!("Launched GPU instance: {}", instance_id);
        Ok(instance_id)
    }

    /// Terminate GPU instance
    async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        {
            let mut instances = self.instances.write().unwrap();
            if let Some(instance) = instances.get_mut(instance_id) {
                instance.state = InstanceState::Terminating;
            }
        }

        // Simulate termination time
        tokio::time::sleep(Duration::from_secs(2)).await;

        {
            let mut instances = self.instances.write().unwrap();
            instances.remove(instance_id);
        }

        tracing::info!("Terminated GPU instance: {}", instance_id);
        Ok(())
    }

    /// Get next available device ID
    fn get_next_available_device_id(&self) -> usize {
        let instances = self.instances.read().unwrap();
        let used_devices: std::collections::HashSet<usize> = instances
            .values()
            .map(|instance| instance.device_id)
            .collect();

        for device_id in 0..self.config.max_instances {
            if !used_devices.contains(&device_id) {
                return device_id;
            }
        }

        0 // Fallback
    }

    /// Calculate instance cost based on performance tier
    fn calculate_instance_cost(&self, tier: PerformanceTier) -> f32 {
        match tier {
            PerformanceTier::Basic => 1.0,
            PerformanceTier::Standard => 2.5,
            PerformanceTier::High => 5.0,
            PerformanceTier::Premium => 10.0,
        }
    }

    /// Create metrics snapshot from current state
    fn create_metrics_snapshot(
        lb_stats: &LoadBalancingStats,
        device_info: &HashMap<usize, GpuDeviceInfo>,
    ) -> MetricsSnapshot {
        let avg_utilization = if !device_info.is_empty() {
            device_info
                .values()
                .map(|info| info.utilization)
                .sum::<f32>()
                / device_info.len() as f32
        } else {
            0.0
        };

        let total_queue_length = device_info.values().map(|info| info.queue_size).sum();

        let avg_latency = if !device_info.is_empty() {
            let total_nanos: u64 = device_info
                .values()
                .map(|info| info.avg_latency.as_nanos() as u64)
                .sum();
            Duration::from_nanos(total_nanos / device_info.len() as u64)
        } else {
            Duration::from_millis(0)
        };

        let throughput = if lb_stats.total_operations > 0 {
            lb_stats.total_operations as f32 / 3600.0 // Ops per hour to ops per second (simplified)
        } else {
            0.0
        };

        MetricsSnapshot {
            timestamp: SystemTime::now(),
            utilization: avg_utilization,
            queue_length: total_queue_length,
            latency: avg_latency,
            throughput,
            instance_count: device_info.len(),
            cost: device_info.len() as f32 * 2.5, // Simplified cost calculation
        }
    }

    /// Evaluate whether scaling decision is needed
    async fn evaluate_scaling_decision(
        config: &AutoScalingConfig,
        current_snapshot: &MetricsSnapshot,
        metrics_history: &Arc<RwLock<VecDeque<MetricsSnapshot>>>,
        last_scaling_time: &Arc<RwLock<SystemTime>>,
        scaling_events_count: &Arc<RwLock<VecDeque<SystemTime>>>,
    ) -> Result<ScalingDecision> {
        // Check cooldown period
        let last_scaling = *last_scaling_time.read().unwrap();
        let cooldown_duration = Duration::from_secs(config.cooldown_period_secs);

        if SystemTime::now()
            .duration_since(last_scaling)
            .unwrap_or(Duration::from_secs(0))
            < cooldown_duration
        {
            return Ok(ScalingDecision {
                timestamp: SystemTime::now(),
                action: ScalingAction::NoAction,
                instance_delta: 0,
                target_instances: current_snapshot.instance_count,
                triggered_by: vec![],
                confidence: 1.0,
                expected_impact: ExpectedImpact::default(),
                cost_impact: CostImpact::default(),
            });
        }

        // Check rate limiting
        {
            let mut events = scaling_events_count.write().unwrap();
            let one_hour_ago = SystemTime::now() - Duration::from_secs(3600);

            // Remove old events
            while let Some(&front_time) = events.front() {
                if front_time < one_hour_ago {
                    events.pop_front();
                } else {
                    break;
                }
            }

            if events.len() >= config.max_scaling_events_per_hour {
                return Ok(ScalingDecision {
                    timestamp: SystemTime::now(),
                    action: ScalingAction::NoAction,
                    instance_delta: 0,
                    target_instances: current_snapshot.instance_count,
                    triggered_by: vec![],
                    confidence: 1.0,
                    expected_impact: ExpectedImpact::default(),
                    cost_impact: CostImpact::default(),
                });
            }
        }

        // Evaluate scale-up triggers
        let mut triggered_scale_up = Vec::new();
        for trigger in &config.scale_up_triggers {
            if Self::evaluate_trigger(trigger, current_snapshot, metrics_history) {
                triggered_scale_up.push(trigger.clone());
            }
        }

        // Evaluate scale-down triggers
        let mut triggered_scale_down = Vec::new();
        for trigger in &config.scale_down_triggers {
            if Self::evaluate_trigger(trigger, current_snapshot, metrics_history) {
                triggered_scale_down.push(trigger.clone());
            }
        }

        // Determine scaling action
        let (action, instance_delta, triggered_by) = if !triggered_scale_up.is_empty()
            && current_snapshot.instance_count < config.max_instances
        {
            (ScalingAction::ScaleUp, 1, triggered_scale_up)
        } else if !triggered_scale_down.is_empty()
            && current_snapshot.instance_count > config.min_instances
        {
            (ScalingAction::ScaleDown, -1, triggered_scale_down)
        } else {
            (ScalingAction::NoAction, 0, vec![])
        };

        let target_instances = (current_snapshot.instance_count as i32 + instance_delta)
            .max(config.min_instances as i32)
            .min(config.max_instances as i32) as usize;

        // Calculate expected impact and cost
        let expected_impact = Self::calculate_expected_impact(&action, current_snapshot);
        let cost_impact = Self::calculate_cost_impact(&action, current_snapshot);

        Ok(ScalingDecision {
            timestamp: SystemTime::now(),
            action,
            instance_delta,
            target_instances,
            triggered_by,
            confidence: 0.8, // Simplified confidence calculation
            expected_impact,
            cost_impact,
        })
    }

    /// Evaluate individual scaling trigger
    fn evaluate_trigger(
        trigger: &ScalingTrigger,
        current_snapshot: &MetricsSnapshot,
        metrics_history: &Arc<RwLock<VecDeque<MetricsSnapshot>>>,
    ) -> bool {
        match trigger {
            ScalingTrigger::Utilization {
                threshold,
                duration_secs,
            } => {
                let history = metrics_history.read().unwrap();
                let cutoff_time = SystemTime::now() - Duration::from_secs(*duration_secs);

                let sustained = history
                    .iter()
                    .filter(|snapshot| snapshot.timestamp > cutoff_time)
                    .all(|snapshot| snapshot.utilization > *threshold);

                sustained && current_snapshot.utilization > *threshold
            }
            ScalingTrigger::Memory {
                threshold: _,
                duration_secs: _,
            } => {
                // Simplified memory check
                false // Would need actual memory metrics
            }
            ScalingTrigger::QueueLength {
                threshold,
                duration_secs,
            } => {
                let history = metrics_history.read().unwrap();
                let cutoff_time = SystemTime::now() - Duration::from_secs(*duration_secs);

                let sustained = history
                    .iter()
                    .filter(|snapshot| snapshot.timestamp > cutoff_time)
                    .all(|snapshot| snapshot.queue_length > *threshold);

                sustained && current_snapshot.queue_length > *threshold
            }
            ScalingTrigger::Latency {
                threshold,
                duration_secs,
            } => {
                let history = metrics_history.read().unwrap();
                let cutoff_time = SystemTime::now() - Duration::from_secs(*duration_secs);

                let sustained = history
                    .iter()
                    .filter(|snapshot| snapshot.timestamp > cutoff_time)
                    .all(|snapshot| snapshot.latency > *threshold);

                sustained && current_snapshot.latency > *threshold
            }
            ScalingTrigger::Throughput {
                threshold,
                duration_secs,
            } => {
                let history = metrics_history.read().unwrap();
                let cutoff_time = SystemTime::now() - Duration::from_secs(*duration_secs);

                let sustained = history
                    .iter()
                    .filter(|snapshot| snapshot.timestamp > cutoff_time)
                    .all(|snapshot| snapshot.throughput < *threshold);

                sustained && current_snapshot.throughput < *threshold
            }
            ScalingTrigger::PredictedDemand {
                confidence_threshold: _,
            } => {
                // Would implement predictive scaling based on historical patterns
                false
            }
            ScalingTrigger::Custom {
                metric_name: _,
                threshold: _,
                duration_secs: _,
            } => {
                // Would implement custom metric evaluation
                false
            }
        }
    }

    /// Calculate expected impact of scaling action
    fn calculate_expected_impact(
        action: &ScalingAction,
        current_snapshot: &MetricsSnapshot,
    ) -> ExpectedImpact {
        match action {
            ScalingAction::ScaleUp | ScalingAction::PreemptiveScaleUp => {
                ExpectedImpact {
                    utilization_change: -0.2, // Expect utilization to decrease
                    latency_change: Duration::from_millis(0)
                        .saturating_sub(Duration::from_millis(500)), // Expect latency to improve
                    throughput_change: current_snapshot.throughput * 0.3, // Expect 30% throughput increase
                    queue_length_change: -(current_snapshot.queue_length as i32 / 2), // Expect queue to reduce
                    confidence: 0.7,
                }
            }
            ScalingAction::ScaleDown | ScalingAction::PreemptiveScaleDown => {
                ExpectedImpact {
                    utilization_change: 0.1, // Expect utilization to increase slightly
                    latency_change: Duration::from_millis(200), // Expect latency to increase slightly
                    throughput_change: -(current_snapshot.throughput * 0.1), // Expect slight throughput decrease
                    queue_length_change: 2, // Expect queue to increase slightly
                    confidence: 0.6,
                }
            }
            ScalingAction::NoAction => ExpectedImpact::default(),
        }
    }

    /// Calculate cost impact of scaling action
    fn calculate_cost_impact(
        action: &ScalingAction,
        current_snapshot: &MetricsSnapshot,
    ) -> CostImpact {
        let base_cost_per_instance = 2.5; // Per hour

        match action {
            ScalingAction::ScaleUp | ScalingAction::PreemptiveScaleUp => {
                CostImpact {
                    hourly_cost_change: base_cost_per_instance,
                    roi_score: 0.8, // Good ROI if demand justifies it
                    efficiency_score: 0.7,
                }
            }
            ScalingAction::ScaleDown | ScalingAction::PreemptiveScaleDown => {
                CostImpact {
                    hourly_cost_change: -base_cost_per_instance,
                    roi_score: 0.9, // High ROI for cost savings
                    efficiency_score: 0.8,
                }
            }
            ScalingAction::NoAction => CostImpact::default(),
        }
    }

    /// Execute scaling decision
    async fn execute_scaling_decision(
        config: &AutoScalingConfig,
        decision: &ScalingDecision,
        instances: &Arc<RwLock<HashMap<String, ScalableGpuInstance>>>,
        _load_balancer: &Option<Arc<GpuLoadBalancer>>,
    ) -> Result<()> {
        match decision.action {
            ScalingAction::ScaleUp | ScalingAction::PreemptiveScaleUp => {
                // Launch new instances
                for _ in 0..decision.instance_delta {
                    // In a real implementation, this would launch actual GPU instances
                    let instance_id = uuid::Uuid::new_v4().to_string();
                    let device_id = instances.read().unwrap().len();

                    let instance = ScalableGpuInstance {
                        instance_id: instance_id.clone(),
                        device_id,
                        device_type: GpuDeviceType::Cuda,
                        state: InstanceState::Running,
                        launched_at: SystemTime::now(),
                        last_activity: SystemTime::now(),
                        hourly_cost: 2.5,
                        performance_tier: PerformanceTier::Standard,
                        health_status: InstanceHealth::Healthy,
                    };

                    instances.write().unwrap().insert(instance_id, instance);
                }
            }
            ScalingAction::ScaleDown | ScalingAction::PreemptiveScaleDown => {
                // Terminate instances
                let instances_to_terminate: Vec<String> = {
                    let instances_lock = instances.read().unwrap();
                    instances_lock
                        .values()
                        .filter(|instance| instance.state == InstanceState::Running)
                        .take(decision.instance_delta.abs() as usize)
                        .map(|instance| instance.instance_id.clone())
                        .collect()
                };

                for instance_id in instances_to_terminate {
                    instances.write().unwrap().remove(&instance_id);
                }
            }
            ScalingAction::NoAction => {
                // No action needed
            }
        }

        Ok(())
    }

    /// Update scaling statistics
    async fn update_scaling_stats(
        stats: &Arc<RwLock<AutoScalingStats>>,
        decision: &ScalingDecision,
    ) {
        let mut stats_lock = stats.write().unwrap();

        stats_lock.total_scaling_events += 1;

        match decision.action {
            ScalingAction::ScaleUp | ScalingAction::PreemptiveScaleUp => {
                stats_lock.scale_up_events += 1;
            }
            ScalingAction::ScaleDown | ScalingAction::PreemptiveScaleDown => {
                stats_lock.scale_down_events += 1;
            }
            ScalingAction::NoAction => {}
        }

        // Update other metrics (simplified)
        stats_lock.avg_instance_count = decision.target_instances as f32;
        stats_lock.peak_instance_count = stats_lock
            .peak_instance_count
            .max(decision.target_instances);
        stats_lock.decision_accuracy = 0.85; // Would be calculated based on actual outcomes
    }

    /// Get current auto-scaling statistics
    pub fn get_statistics(&self) -> AutoScalingStats {
        self.stats.read().unwrap().clone()
    }

    /// Get current instances
    pub fn get_instances(&self) -> HashMap<String, ScalableGpuInstance> {
        self.instances.read().unwrap().clone()
    }

    /// Get scaling history
    pub fn get_scaling_history(&self) -> Vec<ScalingDecision> {
        self.scaling_history
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Generate workload prediction
    pub fn predict_workload(&self, prediction_window: Duration) -> WorkloadPrediction {
        let history = self.metrics_history.read().unwrap();

        if history.is_empty() {
            return WorkloadPrediction {
                timestamp: SystemTime::now(),
                prediction_window,
                predicted_utilization: 0.5,
                predicted_queue_length: 0,
                predicted_throughput: 0.0,
                confidence: 0.1,
                data_points: 0,
                seasonal_patterns: vec![],
            };
        }

        // Simple prediction based on recent average
        let recent_samples: Vec<_> = history.iter().rev().take(10).collect();

        let avg_utilization =
            recent_samples.iter().map(|s| s.utilization).sum::<f32>() / recent_samples.len() as f32;
        let avg_queue_length =
            recent_samples.iter().map(|s| s.queue_length).sum::<usize>() / recent_samples.len();
        let avg_throughput =
            recent_samples.iter().map(|s| s.throughput).sum::<f32>() / recent_samples.len() as f32;

        WorkloadPrediction {
            timestamp: SystemTime::now(),
            prediction_window,
            predicted_utilization: avg_utilization,
            predicted_queue_length: avg_queue_length,
            predicted_throughput: avg_throughput,
            confidence: 0.6,
            data_points: recent_samples.len(),
            seasonal_patterns: vec![], // Would detect patterns in full implementation
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: AutoScalingConfig) {
        self.config = new_config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &AutoScalingConfig {
        &self.config
    }
}

impl Default for ExpectedImpact {
    fn default() -> Self {
        Self {
            utilization_change: 0.0,
            latency_change: Duration::from_millis(0),
            throughput_change: 0.0,
            queue_length_change: 0,
            confidence: 0.5,
        }
    }
}

impl Default for CostImpact {
    fn default() -> Self {
        Self {
            hourly_cost_change: 0.0,
            roi_score: 0.5,
            efficiency_score: 0.5,
        }
    }
}

impl Default for AutoScalingStats {
    fn default() -> Self {
        Self {
            total_scaling_events: 0,
            scale_up_events: 0,
            scale_down_events: 0,
            avg_instance_count: 1.0,
            peak_instance_count: 1,
            min_instance_count: 1,
            total_cost: 0.0,
            cost_per_operation: 0.0,
            avg_scaling_response_time: Duration::from_secs(0),
            prediction_accuracy: 0.5,
            utilization_efficiency: 0.5,
            decision_accuracy: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_scaling_config() {
        let config = AutoScalingConfig::default();

        assert_eq!(config.strategy, AutoScalingStrategy::Balanced);
        assert_eq!(config.min_instances, 1);
        assert_eq!(config.max_instances, 8);
        assert!(config.enabled);
        assert!(!config.scale_up_triggers.is_empty());
        assert!(!config.scale_down_triggers.is_empty());
    }

    #[test]
    fn test_scaling_triggers() {
        let utilization_trigger = ScalingTrigger::Utilization {
            threshold: 0.8,
            duration_secs: 300,
        };

        match utilization_trigger {
            ScalingTrigger::Utilization {
                threshold,
                duration_secs,
            } => {
                assert_eq!(threshold, 0.8);
                assert_eq!(duration_secs, 300);
            }
            _ => panic!("Expected utilization trigger"),
        }
    }

    #[test]
    fn test_auto_scaler_creation() {
        let auto_scaler = AutoScaler::new_default();
        assert_eq!(auto_scaler.config.strategy, AutoScalingStrategy::Balanced);
        assert!(auto_scaler.instances.read().unwrap().is_empty());
    }

    #[test]
    fn test_performance_tier_cost() {
        let auto_scaler = AutoScaler::new_default();

        assert_eq!(
            auto_scaler.calculate_instance_cost(PerformanceTier::Basic),
            1.0
        );
        assert_eq!(
            auto_scaler.calculate_instance_cost(PerformanceTier::Standard),
            2.5
        );
        assert_eq!(
            auto_scaler.calculate_instance_cost(PerformanceTier::High),
            5.0
        );
        assert_eq!(
            auto_scaler.calculate_instance_cost(PerformanceTier::Premium),
            10.0
        );
    }

    #[test]
    fn test_scaling_decision() {
        let decision = ScalingDecision {
            timestamp: SystemTime::now(),
            action: ScalingAction::ScaleUp,
            instance_delta: 1,
            target_instances: 3,
            triggered_by: vec![ScalingTrigger::Utilization {
                threshold: 0.8,
                duration_secs: 300,
            }],
            confidence: 0.8,
            expected_impact: ExpectedImpact::default(),
            cost_impact: CostImpact::default(),
        };

        assert_eq!(decision.action, ScalingAction::ScaleUp);
        assert_eq!(decision.instance_delta, 1);
        assert_eq!(decision.target_instances, 3);
        assert_eq!(decision.confidence, 0.8);
    }

    #[test]
    fn test_scalable_gpu_instance() {
        let instance = ScalableGpuInstance {
            instance_id: "test-instance".to_string(),
            device_id: 0,
            device_type: GpuDeviceType::Cuda,
            state: InstanceState::Running,
            launched_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            hourly_cost: 2.5,
            performance_tier: PerformanceTier::Standard,
            health_status: InstanceHealth::Healthy,
        };

        assert_eq!(instance.instance_id, "test-instance");
        assert_eq!(instance.device_id, 0);
        assert_eq!(instance.state, InstanceState::Running);
        assert_eq!(instance.hourly_cost, 2.5);
    }

    #[test]
    fn test_workload_prediction() {
        let prediction = WorkloadPrediction {
            timestamp: SystemTime::now(),
            prediction_window: Duration::from_secs(3600), // 1 hour
            predicted_utilization: 0.7,
            predicted_queue_length: 15,
            predicted_throughput: 100.0,
            confidence: 0.8,
            data_points: 50,
            seasonal_patterns: vec![],
        };

        assert_eq!(prediction.predicted_utilization, 0.7);
        assert_eq!(prediction.predicted_queue_length, 15);
        assert_eq!(prediction.predicted_throughput, 100.0);
        assert_eq!(prediction.confidence, 0.8);
    }
}
