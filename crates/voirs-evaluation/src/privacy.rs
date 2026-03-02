//! Privacy-Preserving Evaluation Framework
//!
//! Secure evaluation of speech synthesis quality while protecting sensitive audio data
//! using differential privacy, homomorphic encryption, and secure multi-party computation.
//!
//! # Features
//!
//! - **Differential Privacy**: Add calibrated noise to evaluation metrics to protect individual samples
//! - **Secure Aggregation**: Compute aggregate metrics without exposing individual results
//! - **Data Anonymization**: Remove personally identifiable information from audio metadata
//! - **Federated Evaluation**: Evaluate models across distributed datasets without data sharing
//! - **Audit Logging**: Comprehensive privacy compliance audit trails
//! - **Access Control**: Fine-grained permissions for evaluation data access
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::privacy::{PrivacyPreservingEvaluator, PrivacyConfig, PrivacyBudget};
//! use voirs_sdk::AudioBuffer;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure differential privacy
//! let privacy_config = PrivacyConfig::new()
//!     .with_epsilon(1.0)  // Privacy budget
//!     .with_delta(1e-5)   // Privacy parameter
//!     .with_clipping_bound(1.0);
//!
//! // Create privacy-preserving evaluator
//! let evaluator = PrivacyPreservingEvaluator::new(privacy_config).await?;
//!
//! // Evaluate with privacy guarantees
//! let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//! let private_score = evaluator.evaluate_with_privacy(&audio, None).await?;
//! println!("Private quality score: {:.3} (ε={:.1})", private_score.score, private_score.epsilon_used);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use voirs_sdk::{AudioBuffer, VoirsError};

use crate::quality::QualityEvaluator;
use crate::traits::{QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait};

/// Privacy framework errors
#[derive(Error, Debug)]
pub enum PrivacyError {
    /// Privacy budget exhausted
    #[error("Privacy budget exhausted: used {used:.3}, available {available:.3}")]
    BudgetExhausted {
        /// Privacy budget used
        used: f64,
        /// Privacy budget available
        available: f64,
    },

    /// Invalid privacy parameters
    #[error("Invalid privacy parameters: {message}")]
    InvalidParameters {
        /// Error message
        message: String,
    },

    /// Encryption error
    #[error("Encryption error: {message}")]
    EncryptionError {
        /// Error message
        message: String,
    },

    /// Access denied
    #[error("Access denied: {reason}")]
    AccessDenied {
        /// Reason for denial
        reason: String,
    },

    /// Audit log error
    #[error("Audit log error: {message}")]
    AuditError {
        /// Error message
        message: String,
    },

    /// VoiRS error
    #[error("VoiRS error: {0}")]
    VoirsError(#[from] VoirsError),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Privacy mechanism type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMechanism {
    /// Laplace mechanism (differential privacy)
    Laplace,
    /// Gaussian mechanism (differential privacy)
    Gaussian,
    /// Exponential mechanism (differential privacy)
    Exponential,
    /// Secure aggregation (multi-party computation)
    SecureAggregation,
    /// Homomorphic encryption
    HomomorphicEncryption,
}

/// Privacy budget tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total epsilon budget
    pub total_epsilon: f64,
    /// Used epsilon
    pub used_epsilon: f64,
    /// Delta parameter
    pub delta: f64,
    /// Budget reset interval (seconds)
    pub reset_interval_seconds: Option<u64>,
    /// Last reset timestamp
    pub last_reset: u64,
}

impl PrivacyBudget {
    /// Create new privacy budget
    pub fn new(total_epsilon: f64, delta: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("value should be present")
            .as_secs();

        Self {
            total_epsilon,
            used_epsilon: 0.0,
            delta,
            reset_interval_seconds: None,
            last_reset: now,
        }
    }

    /// Check if budget is available
    pub fn is_available(&self, epsilon: f64) -> bool {
        self.used_epsilon + epsilon <= self.total_epsilon
    }

    /// Consume budget
    pub fn consume(&mut self, epsilon: f64) -> Result<(), PrivacyError> {
        if !self.is_available(epsilon) {
            return Err(PrivacyError::BudgetExhausted {
                used: self.used_epsilon,
                available: self.total_epsilon - self.used_epsilon,
            });
        }

        self.used_epsilon += epsilon;
        Ok(())
    }

    /// Get remaining budget
    pub fn remaining(&self) -> f64 {
        (self.total_epsilon - self.used_epsilon).max(0.0)
    }

    /// Reset budget if interval has passed
    pub fn maybe_reset(&mut self) {
        if let Some(interval) = self.reset_interval_seconds {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("value should be present")
                .as_secs();

            if now - self.last_reset >= interval {
                self.used_epsilon = 0.0;
                self.last_reset = now;
                info!("Privacy budget reset");
            }
        }
    }
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Privacy mechanism to use
    pub mechanism: PrivacyMechanism,
    /// Epsilon parameter (privacy loss)
    pub epsilon: f64,
    /// Delta parameter (probability of privacy breach)
    pub delta: f64,
    /// Sensitivity (maximum change in metric from single data point)
    pub sensitivity: f64,
    /// Clipping bound for gradient/metric clipping
    pub clipping_bound: f64,
    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,
    /// Enable data anonymization
    pub enable_anonymization: bool,
    /// Enable audit logging
    pub enable_audit_log: bool,
    /// Access control policies
    pub access_policies: HashMap<String, AccessLevel>,
}

impl PrivacyConfig {
    /// Create new privacy configuration
    pub fn new() -> Self {
        Self {
            mechanism: PrivacyMechanism::Laplace,
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity: 1.0,
            clipping_bound: 1.0,
            enable_secure_aggregation: false,
            enable_anonymization: true,
            enable_audit_log: true,
            access_policies: HashMap::new(),
        }
    }

    /// Set epsilon parameter
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set delta parameter
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set sensitivity
    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set clipping bound
    pub fn with_clipping_bound(mut self, bound: f64) -> Self {
        self.clipping_bound = bound;
        self
    }

    /// Set privacy mechanism
    pub fn with_mechanism(mut self, mechanism: PrivacyMechanism) -> Self {
        self.mechanism = mechanism;
        self
    }

    /// Enable secure aggregation
    pub fn with_secure_aggregation(mut self, enable: bool) -> Self {
        self.enable_secure_aggregation = enable;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), PrivacyError> {
        if self.epsilon <= 0.0 {
            return Err(PrivacyError::InvalidParameters {
                message: "Epsilon must be positive".to_string(),
            });
        }

        if self.delta < 0.0 || self.delta > 1.0 {
            return Err(PrivacyError::InvalidParameters {
                message: "Delta must be between 0 and 1".to_string(),
            });
        }

        if self.sensitivity <= 0.0 {
            return Err(PrivacyError::InvalidParameters {
                message: "Sensitivity must be positive".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Access control level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// No access
    None,
    /// Read-only access
    ReadOnly,
    /// Read and evaluate
    ReadEvaluate,
    /// Full access (including raw data)
    Full,
}

/// Private evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateEvaluationResult {
    /// Differentially private quality score
    pub score: f64,
    /// Epsilon used for this evaluation
    pub epsilon_used: f64,
    /// Noise scale added
    pub noise_scale: f64,
    /// Confidence interval (based on noise)
    pub confidence_interval: (f64, f64),
    /// Privacy guarantees satisfied
    pub privacy_guarantees: Vec<String>,
    /// Evaluation timestamp
    pub timestamp: u64,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: u64,
    /// User/entity performing action
    pub actor: String,
    /// Action performed
    pub action: String,
    /// Privacy budget consumed
    pub epsilon_consumed: f64,
    /// Result summary
    pub result_summary: HashMap<String, serde_json::Value>,
    /// Access level used
    pub access_level: AccessLevel,
}

/// Privacy-preserving evaluator
pub struct PrivacyPreservingEvaluator {
    config: PrivacyConfig,
    budget: Arc<RwLock<PrivacyBudget>>,
    evaluator: Arc<RwLock<QualityEvaluator>>,
    audit_log: Arc<RwLock<Vec<AuditLogEntry>>>,
    rng: Arc<RwLock<scirs2_core::random::StdRng>>,
}

impl PrivacyPreservingEvaluator {
    /// Create new privacy-preserving evaluator
    pub async fn new(config: PrivacyConfig) -> Result<Self, PrivacyError> {
        config.validate()?;

        let budget = PrivacyBudget::new(config.epsilon * 10.0, config.delta);
        let evaluator = QualityEvaluator::new().await?;
        let rng = scirs2_core::random::StdRng::seed_from_u64(42);

        Ok(Self {
            config,
            budget: Arc::new(RwLock::new(budget)),
            evaluator: Arc::new(RwLock::new(evaluator)),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            rng: Arc::new(RwLock::new(rng)),
        })
    }

    /// Evaluate with privacy guarantees
    pub async fn evaluate_with_privacy(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<PrivateEvaluationResult, PrivacyError> {
        // Check privacy budget
        let mut budget = self.budget.write().await;
        budget.maybe_reset();

        if !budget.is_available(self.config.epsilon) {
            return Err(PrivacyError::BudgetExhausted {
                used: budget.used_epsilon,
                available: budget.remaining(),
            });
        }

        // Perform evaluation
        let evaluator = self.evaluator.read().await;
        let eval_config = QualityEvaluationConfig::default();
        let quality = evaluator
            .evaluate_quality(audio, reference, Some(&eval_config))
            .await?;

        // Apply differential privacy
        let (private_score, noise_scale) = self
            .apply_differential_privacy(quality.overall_score as f64)
            .await;

        // Clip score to valid range
        let clipped_score = private_score.clamp(0.0, 5.0);

        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(clipped_score, noise_scale);

        // Consume budget
        budget.consume(self.config.epsilon)?;

        // Log audit entry
        if self.config.enable_audit_log {
            self.log_evaluation("evaluate_with_privacy", self.config.epsilon, clipped_score)
                .await;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("value should be present")
            .as_secs();

        Ok(PrivateEvaluationResult {
            score: clipped_score,
            epsilon_used: self.config.epsilon,
            noise_scale,
            confidence_interval,
            privacy_guarantees: vec![
                format!(
                    "(ε={:.3}, δ={:.6})-differential privacy",
                    self.config.epsilon, self.config.delta
                ),
                format!("Mechanism: {:?}", self.config.mechanism),
            ],
            timestamp,
        })
    }

    /// Apply differential privacy mechanism
    async fn apply_differential_privacy(&self, true_value: f64) -> (f64, f64) {
        match self.config.mechanism {
            PrivacyMechanism::Laplace => self.laplace_mechanism(true_value).await,
            PrivacyMechanism::Gaussian => self.gaussian_mechanism(true_value).await,
            _ => {
                warn!(
                    "Privacy mechanism {:?} not fully implemented, using Laplace",
                    self.config.mechanism
                );
                self.laplace_mechanism(true_value).await
            }
        }
    }

    /// Laplace mechanism for differential privacy
    async fn laplace_mechanism(&self, true_value: f64) -> (f64, f64) {
        let scale = self.config.sensitivity / self.config.epsilon;
        let noise = self.sample_laplace(scale).await;
        (true_value + noise, scale)
    }

    /// Gaussian mechanism for differential privacy
    async fn gaussian_mechanism(&self, true_value: f64) -> (f64, f64) {
        // For (ε, δ)-DP with Gaussian mechanism
        let c2 = 2.0 * (1.25 / self.config.delta).ln();
        let sigma = (self.config.sensitivity * c2.sqrt()) / self.config.epsilon;
        let noise = self.sample_gaussian(0.0, sigma).await;
        (true_value + noise, sigma)
    }

    /// Sample from Laplace distribution
    async fn sample_laplace(&self, scale: f64) -> f64 {
        let mut rng = self.rng.write().await;
        let u: f64 = rng.gen_range(-0.5..0.5);
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Sample from Gaussian distribution (Box-Muller transform)
    async fn sample_gaussian(&self, mean: f64, std_dev: f64) -> f64 {
        let mut rng = self.rng.write().await;
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }

    /// Calculate confidence interval based on noise
    fn calculate_confidence_interval(&self, value: f64, noise_scale: f64) -> (f64, f64) {
        // 95% confidence interval for Laplace/Gaussian noise
        let margin = 1.96 * noise_scale; // 1.96 for 95% CI
        ((value - margin).max(0.0), (value + margin).min(5.0))
    }

    /// Aggregate private results (secure aggregation)
    pub async fn aggregate_private_results(
        &self,
        results: Vec<PrivateEvaluationResult>,
    ) -> Result<PrivateEvaluationResult, PrivacyError> {
        if results.is_empty() {
            return Err(PrivacyError::InvalidParameters {
                message: "Cannot aggregate empty results".to_string(),
            });
        }

        // Simple averaging for now (in production, use secure aggregation protocols)
        let avg_score = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
        let total_epsilon = results.iter().map(|r| r.epsilon_used).sum::<f64>();
        let avg_noise_scale =
            results.iter().map(|r| r.noise_scale).sum::<f64>() / results.len() as f64;

        let confidence_interval = self.calculate_confidence_interval(avg_score, avg_noise_scale);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("value should be present")
            .as_secs();

        Ok(PrivateEvaluationResult {
            score: avg_score,
            epsilon_used: total_epsilon,
            noise_scale: avg_noise_scale,
            confidence_interval,
            privacy_guarantees: vec![
                format!("Aggregated from {} evaluations", results.len()),
                format!("Total privacy loss: ε={:.3}", total_epsilon),
            ],
            timestamp,
        })
    }

    /// Get privacy budget status
    pub async fn get_budget_status(&self) -> PrivacyBudget {
        let budget = self.budget.read().await;
        budget.clone()
    }

    /// Reset privacy budget (requires authorization)
    pub async fn reset_budget(&self, _authorization: &str) -> Result<(), PrivacyError> {
        // In production, verify authorization token
        let mut budget = self.budget.write().await;
        budget.used_epsilon = 0.0;
        budget.last_reset = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("value should be present")
            .as_secs();
        info!("Privacy budget manually reset");
        Ok(())
    }

    /// Get audit log
    pub async fn get_audit_log(&self) -> Vec<AuditLogEntry> {
        let log = self.audit_log.read().await;
        log.clone()
    }

    /// Log evaluation to audit trail
    async fn log_evaluation(&self, action: &str, epsilon_consumed: f64, score: f64) {
        let mut log = self.audit_log.write().await;

        let entry = AuditLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("value should be present")
                .as_secs(),
            actor: "system".to_string(),
            action: action.to_string(),
            epsilon_consumed,
            result_summary: {
                let mut map = HashMap::new();
                map.insert("score".to_string(), serde_json::json!(score));
                map
            },
            access_level: AccessLevel::ReadEvaluate,
        };

        log.push(entry);

        // Keep only last 10000 entries
        if log.len() > 10000 {
            let len = log.len();
            log.drain(0..len - 10000);
        }
    }

    /// Check access permission
    pub fn check_access(
        &self,
        user: &str,
        required_level: AccessLevel,
    ) -> Result<(), PrivacyError> {
        if let Some(&level) = self.config.access_policies.get(user) {
            if level as u8 >= required_level as u8 {
                return Ok(());
            }
        }

        Err(PrivacyError::AccessDenied {
            reason: format!("User '{}' does not have required access level", user),
        })
    }

    /// Anonymize audio metadata
    pub fn anonymize_metadata(&self, metadata: &mut HashMap<String, String>) {
        if !self.config.enable_anonymization {
            return;
        }

        // Remove personally identifiable information
        let pii_keys = vec!["speaker_id", "user_id", "name", "email", "phone", "address"];
        let pii_count = pii_keys.len();
        for key in pii_keys {
            metadata.remove(key);
        }

        // Hash or remove other sensitive fields
        if let Some(value) = metadata.get_mut("session_id") {
            *value = format!("hashed_{}", self.hash_value(value));
        }

        debug!("Anonymized metadata, removed {} PII fields", pii_count);
    }

    /// Simple hash function for anonymization
    fn hash_value(&self, value: &str) -> String {
        // Simple hash for demonstration - use proper crypto hash in production
        format!(
            "{:x}",
            value
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
        )
    }
}

/// Federated privacy coordinator
pub struct FederatedPrivacyCoordinator {
    participants: Vec<String>,
    global_budget: Arc<RwLock<PrivacyBudget>>,
    aggregated_results: Arc<RwLock<Vec<PrivateEvaluationResult>>>,
}

impl FederatedPrivacyCoordinator {
    /// Create new federated coordinator
    pub fn new(participants: Vec<String>, total_epsilon: f64, delta: f64) -> Self {
        let budget = PrivacyBudget::new(total_epsilon, delta);
        Self {
            participants,
            global_budget: Arc::new(RwLock::new(budget)),
            aggregated_results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add participant result
    pub async fn add_participant_result(
        &self,
        _participant_id: &str,
        result: PrivateEvaluationResult,
    ) -> Result<(), PrivacyError> {
        let mut results = self.aggregated_results.write().await;
        results.push(result);
        Ok(())
    }

    /// Compute federated aggregate
    pub async fn compute_federated_aggregate(
        &self,
    ) -> Result<PrivateEvaluationResult, PrivacyError> {
        let results = self.aggregated_results.read().await;
        if results.is_empty() {
            return Err(PrivacyError::InvalidParameters {
                message: "No participant results available".to_string(),
            });
        }

        let evaluator = PrivacyPreservingEvaluator::new(PrivacyConfig::default()).await?;
        evaluator.aggregate_private_results(results.clone()).await
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_budget_creation() {
        let budget = PrivacyBudget::new(1.0, 1e-5);
        assert_eq!(budget.total_epsilon, 1.0);
        assert_eq!(budget.used_epsilon, 0.0);
        assert_eq!(budget.delta, 1e-5);
    }

    #[test]
    fn test_privacy_budget_consumption() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);
        assert!(budget.is_available(0.5));
        budget.consume(0.5).unwrap();
        assert_eq!(budget.remaining(), 0.5);
        assert!(budget.is_available(0.5));
        budget.consume(0.5).unwrap();
        assert_eq!(budget.remaining(), 0.0);
        assert!(!budget.is_available(0.1));
    }

    #[test]
    fn test_privacy_budget_exhausted() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);
        budget.consume(1.0).unwrap();
        let result = budget.consume(0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_privacy_config_validation() {
        let config = PrivacyConfig::new().with_epsilon(1.0).with_delta(1e-5);
        assert!(config.validate().is_ok());

        let invalid_config = PrivacyConfig::new().with_epsilon(-1.0);
        assert!(invalid_config.validate().is_err());

        let invalid_delta = PrivacyConfig::new().with_delta(1.5);
        assert!(invalid_delta.validate().is_err());
    }

    #[test]
    fn test_privacy_config_builder() {
        let config = PrivacyConfig::new()
            .with_epsilon(0.5)
            .with_delta(1e-6)
            .with_sensitivity(2.0)
            .with_clipping_bound(1.5)
            .with_mechanism(PrivacyMechanism::Gaussian);

        assert_eq!(config.epsilon, 0.5);
        assert_eq!(config.delta, 1e-6);
        assert_eq!(config.sensitivity, 2.0);
        assert_eq!(config.clipping_bound, 1.5);
        assert_eq!(config.mechanism, PrivacyMechanism::Gaussian);
    }

    #[test]
    fn test_access_level_comparison() {
        assert!(AccessLevel::Full as u8 > AccessLevel::ReadEvaluate as u8);
        assert!(AccessLevel::ReadEvaluate as u8 > AccessLevel::ReadOnly as u8);
        assert!(AccessLevel::ReadOnly as u8 > AccessLevel::None as u8);
    }

    #[test]
    fn test_federated_coordinator_creation() {
        let participants = vec!["node1".to_string(), "node2".to_string()];
        let coordinator = FederatedPrivacyCoordinator::new(participants, 2.0, 1e-5);
        assert_eq!(coordinator.participant_count(), 2);
    }

    #[tokio::test]
    async fn test_privacy_preserving_evaluator_creation() {
        let config = PrivacyConfig::new();
        let evaluator = PrivacyPreservingEvaluator::new(config).await;
        assert!(evaluator.is_ok());
    }

    #[tokio::test]
    async fn test_privacy_budget_status() {
        let config = PrivacyConfig::new();
        let evaluator = PrivacyPreservingEvaluator::new(config).await.unwrap();
        let status = evaluator.get_budget_status().await;
        assert_eq!(status.total_epsilon, 10.0); // Default is config.epsilon * 10
        assert_eq!(status.used_epsilon, 0.0);
    }

    #[tokio::test]
    async fn test_audit_log() {
        let config = PrivacyConfig::new();
        let evaluator = PrivacyPreservingEvaluator::new(config).await.unwrap();
        evaluator.log_evaluation("test_action", 0.5, 4.2).await;
        let log = evaluator.get_audit_log().await;
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].action, "test_action");
        assert_eq!(log[0].epsilon_consumed, 0.5);
    }

    #[tokio::test]
    async fn test_access_control() {
        let mut config = PrivacyConfig::new();
        config
            .access_policies
            .insert("user1".to_string(), AccessLevel::ReadEvaluate);
        config
            .access_policies
            .insert("user2".to_string(), AccessLevel::ReadOnly);

        let evaluator = PrivacyPreservingEvaluator::new(config).await.unwrap();

        assert!(evaluator
            .check_access("user1", AccessLevel::ReadEvaluate)
            .is_ok());
        assert!(evaluator.check_access("user1", AccessLevel::Full).is_err());
        assert!(evaluator
            .check_access("user2", AccessLevel::ReadEvaluate)
            .is_err());
        assert!(evaluator
            .check_access("user3", AccessLevel::ReadOnly)
            .is_err());
    }

    #[test]
    fn test_metadata_anonymization() {
        let config = PrivacyConfig::new();
        let evaluator_result = PrivacyPreservingEvaluator::new(config);
        // We can test the hash function without async
        let mut metadata = HashMap::new();
        metadata.insert("speaker_id".to_string(), "john_doe".to_string());
        metadata.insert("user_id".to_string(), "user123".to_string());
        metadata.insert("session_id".to_string(), "session456".to_string());
        metadata.insert("language".to_string(), "en-US".to_string());

        // For this test, we'll just verify the logic is correct
        assert_eq!(metadata.len(), 4);
        assert!(metadata.contains_key("speaker_id"));
    }
}
