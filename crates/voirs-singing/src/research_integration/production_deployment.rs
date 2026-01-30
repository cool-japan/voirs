// Production Deployment Features - Phase 5 Feature
//
// Health checks, hot-reloading, A/B testing, and deployment management
// for production singing synthesis systems.

use crate::Error;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Component health check result
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub component: String,
    pub status: HealthStatus,
    pub message: String,
    pub last_check: SystemTime,
    pub response_time: Duration,
}

/// Health check system
#[derive(Debug, Clone)]
pub struct HealthChecker {
    checks: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    thresholds: HealthCheckThresholds,
}

#[derive(Debug, Clone)]
pub struct HealthCheckThresholds {
    pub response_time_warning: Duration,
    pub response_time_critical: Duration,
    pub error_rate_warning: f64,
    pub error_rate_critical: f64,
}

impl Default for HealthCheckThresholds {
    fn default() -> Self {
        Self {
            response_time_warning: Duration::from_millis(500),
            response_time_critical: Duration::from_secs(2),
            error_rate_warning: 0.05,  // 5%
            error_rate_critical: 0.10, // 10%
        }
    }
}

impl HealthChecker {
    pub fn new(thresholds: HealthCheckThresholds) -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            thresholds,
        }
    }

    /// Register a component for health checking
    pub fn register_component(&self, component: &str) -> Result<(), Error> {
        let mut checks = self
            .checks
            .write()
            .map_err(|_| Error::Processing("Failed to acquire checks lock".into()))?;

        checks.insert(
            component.to_string(),
            ComponentHealth {
                component: component.to_string(),
                status: HealthStatus::Healthy,
                message: "Component registered".to_string(),
                last_check: SystemTime::now(),
                response_time: Duration::ZERO,
            },
        );

        Ok(())
    }

    /// Update component health
    pub fn update_health(
        &self,
        component: &str,
        status: HealthStatus,
        message: &str,
        response_time: Duration,
    ) -> Result<(), Error> {
        let mut checks = self
            .checks
            .write()
            .map_err(|_| Error::Processing("Failed to acquire checks lock".into()))?;

        if let Some(health) = checks.get_mut(component) {
            health.status = status;
            health.message = message.to_string();
            health.last_check = SystemTime::now();
            health.response_time = response_time;
        }

        Ok(())
    }

    /// Perform health check on component
    pub async fn check_component<F>(&self, component: &str, check_fn: F) -> Result<(), Error>
    where
        F: FnOnce() -> Result<(), Error>,
    {
        let start = Instant::now();

        let (status, message) = match check_fn() {
            Ok(()) => {
                let elapsed = start.elapsed();
                if elapsed > self.thresholds.response_time_critical {
                    (
                        HealthStatus::Unhealthy,
                        format!("Response time critical: {:?}", elapsed),
                    )
                } else if elapsed > self.thresholds.response_time_warning {
                    (
                        HealthStatus::Degraded,
                        format!("Response time warning: {:?}", elapsed),
                    )
                } else {
                    (HealthStatus::Healthy, "Component healthy".to_string())
                }
            }
            Err(e) => (HealthStatus::Unhealthy, format!("Check failed: {}", e)),
        };

        self.update_health(component, status, &message, start.elapsed())
    }

    /// Get overall system health
    pub fn overall_health(&self) -> Result<HealthStatus, Error> {
        let checks = self
            .checks
            .read()
            .map_err(|_| Error::Processing("Failed to read checks".into()))?;

        if checks.values().any(|h| h.status == HealthStatus::Unhealthy) {
            Ok(HealthStatus::Unhealthy)
        } else if checks.values().any(|h| h.status == HealthStatus::Degraded) {
            Ok(HealthStatus::Degraded)
        } else {
            Ok(HealthStatus::Healthy)
        }
    }

    /// Get all component health status
    pub fn get_all_health(&self) -> Result<Vec<ComponentHealth>, Error> {
        let checks = self
            .checks
            .read()
            .map_err(|_| Error::Processing("Failed to read checks".into()))?;
        Ok(checks.values().cloned().collect())
    }

    /// Get specific component health
    pub fn get_health(&self, component: &str) -> Result<Option<ComponentHealth>, Error> {
        let checks = self
            .checks
            .read()
            .map_err(|_| Error::Processing("Failed to read checks".into()))?;
        Ok(checks.get(component).cloned())
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new(HealthCheckThresholds::default())
    }
}

/// Hot-reloading configuration manager
#[derive(Debug, Clone)]
pub struct HotReloader {
    configs: Arc<RwLock<HashMap<String, ConfigEntry>>>,
    reload_listeners: Arc<RwLock<Vec<String>>>,
}

#[derive(Debug, Clone)]
struct ConfigEntry {
    key: String,
    value: String,
    version: u64,
    last_updated: SystemTime,
}

impl HotReloader {
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            reload_listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set configuration value
    pub fn set_config(&self, key: &str, value: &str) -> Result<(), Error> {
        let mut configs = self
            .configs
            .write()
            .map_err(|_| Error::Processing("Failed to acquire configs lock".into()))?;

        let version = configs.get(key).map(|e| e.version + 1).unwrap_or(1);

        configs.insert(
            key.to_string(),
            ConfigEntry {
                key: key.to_string(),
                value: value.to_string(),
                version,
                last_updated: SystemTime::now(),
            },
        );

        Ok(())
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Result<Option<String>, Error> {
        let configs = self
            .configs
            .read()
            .map_err(|_| Error::Processing("Failed to read configs".into()))?;

        Ok(configs.get(key).map(|e| e.value.clone()))
    }

    /// Get configuration version
    pub fn get_version(&self, key: &str) -> Result<Option<u64>, Error> {
        let configs = self
            .configs
            .read()
            .map_err(|_| Error::Processing("Failed to read configs".into()))?;

        Ok(configs.get(key).map(|e| e.version))
    }

    /// Reload all configurations
    pub fn reload_all(&self) -> Result<Vec<String>, Error> {
        let configs = self
            .configs
            .read()
            .map_err(|_| Error::Processing("Failed to read configs".into()))?;

        Ok(configs.keys().cloned().collect())
    }

    /// Register reload listener
    pub fn register_listener(&self, listener: &str) -> Result<(), Error> {
        let mut listeners = self
            .reload_listeners
            .write()
            .map_err(|_| Error::Processing("Failed to acquire listeners lock".into()))?;

        listeners.push(listener.to_string());
        Ok(())
    }

    /// Get all configurations
    pub fn get_all_configs(&self) -> Result<HashMap<String, String>, Error> {
        let configs = self
            .configs
            .read()
            .map_err(|_| Error::Processing("Failed to read configs".into()))?;

        Ok(configs
            .iter()
            .map(|(k, v)| (k.clone(), v.value.clone()))
            .collect())
    }
}

impl Default for HotReloader {
    fn default() -> Self {
        Self::new()
    }
}

/// A/B testing framework
#[derive(Debug, Clone)]
pub struct ABTestManager {
    experiments: Arc<RwLock<HashMap<String, Experiment>>>,
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub variants: Vec<Variant>,
    pub traffic_split: HashMap<String, f64>,
    pub metrics: ExperimentMetrics,
    pub start_time: SystemTime,
    pub status: ExperimentStatus,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub id: String,
    pub name: String,
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ExperimentMetrics {
    pub total_assignments: usize,
    pub variant_assignments: HashMap<String, usize>,
    pub conversions: HashMap<String, usize>,
    pub conversion_rates: HashMap<String, f64>,
}

impl Default for ExperimentMetrics {
    fn default() -> Self {
        Self {
            total_assignments: 0,
            variant_assignments: HashMap::new(),
            conversions: HashMap::new(),
            conversion_rates: HashMap::new(),
        }
    }
}

/// Status of an A/B test experiment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Experiment is actively running and accepting traffic
    Running,
    /// Experiment is temporarily paused
    Paused,
    /// Experiment has completed and no longer accepts traffic
    Completed,
}

impl ABTestManager {
    /// Create a new A/B test manager
    pub fn new() -> Self {
        Self {
            experiments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(
        &self,
        id: &str,
        name: &str,
        variants: Vec<Variant>,
        traffic_split: HashMap<String, f64>,
    ) -> Result<(), Error> {
        let mut experiments = self
            .experiments
            .write()
            .map_err(|_| Error::Processing("Failed to acquire experiments lock".into()))?;

        // Validate traffic split sums to 1.0
        let total_traffic: f64 = traffic_split.values().sum();
        if (total_traffic - 1.0).abs() > 0.01 {
            return Err(Error::Processing(format!(
                "Traffic split must sum to 1.0, got {}",
                total_traffic
            )));
        }

        experiments.insert(
            id.to_string(),
            Experiment {
                id: id.to_string(),
                name: name.to_string(),
                variants,
                traffic_split,
                metrics: ExperimentMetrics::default(),
                start_time: SystemTime::now(),
                status: ExperimentStatus::Running,
            },
        );

        Ok(())
    }

    /// Assign user to variant
    pub fn assign_variant(&self, experiment_id: &str, user_hash: f64) -> Result<String, Error> {
        let mut experiments = self
            .experiments
            .write()
            .map_err(|_| Error::Processing("Failed to acquire experiments lock".into()))?;

        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| Error::Processing(format!("Experiment {} not found", experiment_id)))?;

        if experiment.status != ExperimentStatus::Running {
            return Err(Error::Processing("Experiment not running".into()));
        }

        // Deterministic assignment based on user hash
        let mut cumulative = 0.0;
        let mut assigned_variant = None;

        for (variant_id, traffic) in &experiment.traffic_split {
            cumulative += traffic;
            if user_hash <= cumulative {
                assigned_variant = Some(variant_id.clone());
                break;
            }
        }

        let variant_id =
            assigned_variant.ok_or_else(|| Error::Processing("Failed to assign variant".into()))?;

        // Update metrics
        experiment.metrics.total_assignments += 1;
        *experiment
            .metrics
            .variant_assignments
            .entry(variant_id.clone())
            .or_insert(0) += 1;

        Ok(variant_id)
    }

    /// Record conversion for variant
    pub fn record_conversion(&self, experiment_id: &str, variant_id: &str) -> Result<(), Error> {
        let mut experiments = self
            .experiments
            .write()
            .map_err(|_| Error::Processing("Failed to acquire experiments lock".into()))?;

        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| Error::Processing(format!("Experiment {} not found", experiment_id)))?;

        *experiment
            .metrics
            .conversions
            .entry(variant_id.to_string())
            .or_insert(0) += 1;

        // Update conversion rate
        let conversions = experiment
            .metrics
            .conversions
            .get(variant_id)
            .copied()
            .unwrap_or(0);
        let assignments = experiment
            .metrics
            .variant_assignments
            .get(variant_id)
            .copied()
            .unwrap_or(1);
        let rate = conversions as f64 / assignments as f64;

        experiment
            .metrics
            .conversion_rates
            .insert(variant_id.to_string(), rate);

        Ok(())
    }

    /// Get experiment results
    pub fn get_results(&self, experiment_id: &str) -> Result<ExperimentResults, Error> {
        let experiments = self
            .experiments
            .read()
            .map_err(|_| Error::Processing("Failed to read experiments".into()))?;

        let experiment = experiments
            .get(experiment_id)
            .ok_or_else(|| Error::Processing(format!("Experiment {} not found", experiment_id)))?;

        Ok(ExperimentResults {
            experiment_id: experiment.id.clone(),
            experiment_name: experiment.name.clone(),
            status: experiment.status,
            total_assignments: experiment.metrics.total_assignments,
            variant_results: experiment
                .variants
                .iter()
                .map(|v| {
                    let assignments = experiment
                        .metrics
                        .variant_assignments
                        .get(&v.id)
                        .copied()
                        .unwrap_or(0);
                    let conversions = experiment
                        .metrics
                        .conversions
                        .get(&v.id)
                        .copied()
                        .unwrap_or(0);
                    let rate = experiment
                        .metrics
                        .conversion_rates
                        .get(&v.id)
                        .copied()
                        .unwrap_or(0.0);

                    VariantResult {
                        variant_id: v.id.clone(),
                        variant_name: v.name.clone(),
                        assignments,
                        conversions,
                        conversion_rate: rate,
                    }
                })
                .collect(),
        })
    }

    /// Pause experiment
    pub fn pause_experiment(&self, experiment_id: &str) -> Result<(), Error> {
        let mut experiments = self
            .experiments
            .write()
            .map_err(|_| Error::Processing("Failed to acquire experiments lock".into()))?;

        if let Some(experiment) = experiments.get_mut(experiment_id) {
            experiment.status = ExperimentStatus::Paused;
        }

        Ok(())
    }

    /// Complete experiment
    pub fn complete_experiment(&self, experiment_id: &str) -> Result<(), Error> {
        let mut experiments = self
            .experiments
            .write()
            .map_err(|_| Error::Processing("Failed to acquire experiments lock".into()))?;

        if let Some(experiment) = experiments.get_mut(experiment_id) {
            experiment.status = ExperimentStatus::Completed;
        }

        Ok(())
    }

    /// Get all experiments
    pub fn get_all_experiments(&self) -> Result<Vec<String>, Error> {
        let experiments = self
            .experiments
            .read()
            .map_err(|_| Error::Processing("Failed to read experiments".into()))?;

        Ok(experiments.keys().cloned().collect())
    }
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Results of an A/B test experiment
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    /// Unique identifier for the experiment
    pub experiment_id: String,
    /// Human-readable name of the experiment
    pub experiment_name: String,
    /// Current status of the experiment
    pub status: ExperimentStatus,
    /// Total number of user assignments across all variants
    pub total_assignments: usize,
    /// Results for each variant in the experiment
    pub variant_results: Vec<VariantResult>,
}

/// Results for a single variant in an experiment
#[derive(Debug, Clone)]
pub struct VariantResult {
    /// Unique identifier for the variant
    pub variant_id: String,
    /// Human-readable name of the variant
    pub variant_name: String,
    /// Number of users assigned to this variant
    pub assignments: usize,
    /// Number of conversions for this variant
    pub conversions: usize,
    /// Conversion rate (conversions / assignments)
    pub conversion_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_checker_registration() {
        let checker = HealthChecker::default();

        checker.register_component("database").unwrap();
        checker.register_component("cache").unwrap();

        let all_health = checker.get_all_health().unwrap();
        assert_eq!(all_health.len(), 2);
    }

    #[test]
    fn test_health_checker_update() {
        let checker = HealthChecker::default();

        checker.register_component("api").unwrap();
        checker
            .update_health(
                "api",
                HealthStatus::Degraded,
                "High latency",
                Duration::from_millis(600),
            )
            .unwrap();

        let health = checker.get_health("api").unwrap().unwrap();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.message, "High latency");
    }

    #[tokio::test]
    async fn test_health_checker_component_check() {
        let checker = HealthChecker::default();

        checker.register_component("service").unwrap();

        checker.check_component("service", || Ok(())).await.unwrap();

        let health = checker.get_health("service").unwrap().unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_checker_overall_status() {
        let checker = HealthChecker::default();

        checker.register_component("comp1").unwrap();
        checker.register_component("comp2").unwrap();

        checker
            .update_health(
                "comp1",
                HealthStatus::Healthy,
                "OK",
                Duration::from_millis(10),
            )
            .unwrap();
        checker
            .update_health(
                "comp2",
                HealthStatus::Degraded,
                "Slow",
                Duration::from_millis(500),
            )
            .unwrap();

        assert_eq!(checker.overall_health().unwrap(), HealthStatus::Degraded);
    }

    #[test]
    fn test_hot_reloader_config() {
        let reloader = HotReloader::new();

        reloader.set_config("max_voices", "8").unwrap();
        reloader.set_config("quality", "high").unwrap();

        assert_eq!(
            reloader.get_config("max_voices").unwrap(),
            Some("8".to_string())
        );
        assert_eq!(reloader.get_version("max_voices").unwrap(), Some(1));
    }

    #[test]
    fn test_hot_reloader_versioning() {
        let reloader = HotReloader::new();

        reloader.set_config("param", "value1").unwrap();
        assert_eq!(reloader.get_version("param").unwrap(), Some(1));

        reloader.set_config("param", "value2").unwrap();
        assert_eq!(reloader.get_version("param").unwrap(), Some(2));

        assert_eq!(
            reloader.get_config("param").unwrap(),
            Some("value2".to_string())
        );
    }

    #[test]
    fn test_hot_reloader_listeners() {
        let reloader = HotReloader::new();

        reloader.register_listener("synthesis_engine").unwrap();
        reloader.register_listener("quality_monitor").unwrap();

        reloader.set_config("test", "value").unwrap();

        let configs = reloader.reload_all().unwrap();
        assert!(configs.contains(&"test".to_string()));
    }

    #[test]
    fn test_ab_test_create_experiment() {
        let manager = ABTestManager::new();

        let variants = vec![
            Variant {
                id: "control".to_string(),
                name: "Control".to_string(),
                config: HashMap::new(),
            },
            Variant {
                id: "treatment".to_string(),
                name: "Treatment".to_string(),
                config: HashMap::new(),
            },
        ];

        let mut traffic_split = HashMap::new();
        traffic_split.insert("control".to_string(), 0.5);
        traffic_split.insert("treatment".to_string(), 0.5);

        manager
            .create_experiment("exp1", "Quality Test", variants, traffic_split)
            .unwrap();

        let experiments = manager.get_all_experiments().unwrap();
        assert_eq!(experiments.len(), 1);
    }

    #[test]
    fn test_ab_test_variant_assignment() {
        let manager = ABTestManager::new();

        let variants = vec![
            Variant {
                id: "A".to_string(),
                name: "Variant A".to_string(),
                config: HashMap::new(),
            },
            Variant {
                id: "B".to_string(),
                name: "Variant B".to_string(),
                config: HashMap::new(),
            },
        ];

        let mut traffic_split = HashMap::new();
        traffic_split.insert("A".to_string(), 0.5);
        traffic_split.insert("B".to_string(), 0.5);

        manager
            .create_experiment("exp1", "Test", variants, traffic_split)
            .unwrap();

        // Test that both hash values get assigned to valid variants
        let variant1 = manager.assign_variant("exp1", 0.3).unwrap();
        assert!(
            variant1 == "A" || variant1 == "B",
            "Expected A or B, got {}",
            variant1
        );

        let variant2 = manager.assign_variant("exp1", 0.7).unwrap();
        assert!(
            variant2 == "A" || variant2 == "B",
            "Expected A or B, got {}",
            variant2
        );

        // Verify deterministic assignment - same hash should give same variant
        let variant1_again = manager.assign_variant("exp1", 0.3).unwrap();
        assert_eq!(
            variant1, variant1_again,
            "Assignment should be deterministic"
        );
    }

    #[test]
    fn test_ab_test_conversion_tracking() {
        let manager = ABTestManager::new();

        let variants = vec![Variant {
            id: "A".to_string(),
            name: "A".to_string(),
            config: HashMap::new(),
        }];

        let mut traffic_split = HashMap::new();
        traffic_split.insert("A".to_string(), 1.0);

        manager
            .create_experiment("exp1", "Test", variants, traffic_split)
            .unwrap();

        manager.assign_variant("exp1", 0.5).unwrap();
        manager.assign_variant("exp1", 0.6).unwrap();
        manager.record_conversion("exp1", "A").unwrap();

        let results = manager.get_results("exp1").unwrap();
        assert_eq!(results.total_assignments, 2);
        assert_eq!(results.variant_results[0].conversions, 1);
        assert_eq!(results.variant_results[0].conversion_rate, 0.5);
    }

    #[test]
    fn test_ab_test_experiment_lifecycle() {
        let manager = ABTestManager::new();

        let variants = vec![Variant {
            id: "A".to_string(),
            name: "A".to_string(),
            config: HashMap::new(),
        }];

        let mut traffic_split = HashMap::new();
        traffic_split.insert("A".to_string(), 1.0);

        manager
            .create_experiment("exp1", "Test", variants, traffic_split)
            .unwrap();

        let results = manager.get_results("exp1").unwrap();
        assert_eq!(results.status, ExperimentStatus::Running);

        manager.pause_experiment("exp1").unwrap();
        let results = manager.get_results("exp1").unwrap();
        assert_eq!(results.status, ExperimentStatus::Paused);

        manager.complete_experiment("exp1").unwrap();
        let results = manager.get_results("exp1").unwrap();
        assert_eq!(results.status, ExperimentStatus::Completed);
    }
}
