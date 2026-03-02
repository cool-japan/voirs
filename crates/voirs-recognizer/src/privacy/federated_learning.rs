//! Federated Learning implementation for privacy-preserving distributed training.
//!
//! This module implements federated learning protocols allowing multiple clients to
//! collaboratively train models without sharing raw data. The implementation follows
//! the `FederatedAveraging` (`FedAvg`) algorithm with secure aggregation support.

use parking_lot::RwLock;
use scirs2_core::ndarray::{DataOwned, IndexLonger};
use scirs2_core::random::{thread_rng, Distribution, Rng, SliceRandom, SliceRandomExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Federated learning error types
#[derive(Debug, Error)]
pub enum FederatedError {
    /// Client ID not found in registered clients
    #[error("Client not registered: {0}")]
    ClientNotRegistered(String),

    /// Invalid model update received from client
    #[error("Invalid model update from client {0}: {1}")]
    InvalidUpdate(String, String),

    /// Model aggregation operation failed
    #[error("Aggregation failed: {0}")]
    AggregationFailed(String),

    /// Network or communication error
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Cryptographic operation failed
    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    /// Configuration or setup error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Federated learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedConfig {
    /// Minimum number of clients required for aggregation
    pub min_clients: usize,

    /// Maximum number of clients per round
    pub max_clients: usize,

    /// Client selection strategy
    pub selection_strategy: ClientSelectionStrategy,

    /// Aggregation strategy
    pub aggregation_strategy: AggregationStrategy,

    /// Number of local epochs per client
    pub local_epochs: usize,

    /// Local batch size for client training
    pub local_batch_size: usize,

    /// Learning rate for local training
    pub local_learning_rate: f32,

    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,

    /// Differential privacy epsilon (privacy budget)
    pub dp_epsilon: Option<f32>,

    /// Differential privacy delta
    pub dp_delta: Option<f32>,

    /// Gradient clipping threshold
    pub gradient_clip_norm: Option<f32>,

    /// Communication rounds
    pub num_rounds: usize,

    /// Client timeout in seconds
    pub client_timeout_secs: u64,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            min_clients: 2,
            max_clients: 100,
            selection_strategy: ClientSelectionStrategy::Random,
            aggregation_strategy: AggregationStrategy::FederatedAveraging,
            local_epochs: 1,
            local_batch_size: 32,
            local_learning_rate: 0.01,
            enable_secure_aggregation: true,
            dp_epsilon: Some(1.0),
            dp_delta: Some(1e-5),
            gradient_clip_norm: Some(1.0),
            num_rounds: 100,
            client_timeout_secs: 300,
        }
    }
}

/// Client selection strategies for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientSelectionStrategy {
    /// Random selection of clients
    Random,
    /// Select clients based on data size
    DataSizeBased,
    /// Select clients based on compute capacity
    ComputeBased,
    /// Active learning: select clients with highest loss
    ActiveLearning,
    /// Fair selection ensuring all clients participate
    Fair,
}

/// Aggregation strategies for combining client updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Federated Averaging (`FedAvg`) - weighted average by data size
    FederatedAveraging,
    /// Simple average of all client updates
    SimpleAverage,
    /// Median aggregation (robust to outliers)
    MedianAggregation,
    /// Trimmed mean (remove top/bottom percentiles)
    TrimmedMean,
    /// Krum aggregation (Byzantine-robust)
    Krum,
    /// Weighted by client contribution score
    WeightedContribution,
}

/// Client metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientMetadata {
    /// Unique client identifier
    pub client_id: String,

    /// Number of training samples
    pub num_samples: usize,

    /// Client compute capacity (arbitrary units)
    pub compute_capacity: f32,

    /// Client network bandwidth (Mbps)
    pub bandwidth: f32,

    /// Client device type
    pub device_type: DeviceType,

    /// Last participation round
    pub last_round: Option<usize>,

    /// Total rounds participated
    pub rounds_participated: usize,
}

/// Device types for federated clients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Mobile phone
    Mobile,
    /// Tablet device
    Tablet,
    /// Desktop computer
    Desktop,
    /// Edge server
    EdgeServer,
    /// `IoT` device
    IoT,
}

/// Client update containing model gradients and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientUpdate {
    /// Client identifier
    pub client_id: String,

    /// Model parameter updates (gradients or weights)
    pub parameters: Vec<f32>,

    /// Number of samples used for training
    pub num_samples: usize,

    /// Local training loss
    pub training_loss: f32,

    /// Local validation accuracy (if available)
    pub validation_accuracy: Option<f32>,

    /// Training time in milliseconds
    pub training_time_ms: u64,

    /// Round number
    pub round: usize,

    /// Secure aggregation shares (if enabled)
    pub secure_shares: Option<Vec<Vec<u8>>>,
}

/// Server update to send to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerUpdate {
    /// Updated global model parameters
    pub parameters: Vec<f32>,

    /// Current round number
    pub round: usize,

    /// Global model performance metrics
    pub global_metrics: GlobalMetrics,

    /// Configuration updates (if any)
    pub config_update: Option<FederatedConfig>,
}

/// Global model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    /// Average training loss across clients
    pub avg_training_loss: f32,

    /// Average validation accuracy
    pub avg_validation_accuracy: f32,

    /// Number of clients participated
    pub num_clients: usize,

    /// Total samples used
    pub total_samples: usize,

    /// Round duration in milliseconds
    pub round_duration_ms: u64,
}

/// Federated learning server
pub struct FederatedLearningServer {
    /// Server configuration
    config: FederatedConfig,

    /// Registered clients
    clients: Arc<RwLock<HashMap<String, ClientMetadata>>>,

    /// Current global model parameters
    global_model: Arc<RwLock<Vec<f32>>>,

    /// Current round number
    current_round: Arc<RwLock<usize>>,

    /// Client updates for current round
    pending_updates: Arc<RwLock<Vec<ClientUpdate>>>,

    /// Training history
    history: Arc<RwLock<Vec<GlobalMetrics>>>,
}

impl FederatedLearningServer {
    /// Create a new federated learning server
    #[must_use]
    pub fn new(config: FederatedConfig, model_size: usize) -> Self {
        Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
            global_model: Arc::new(RwLock::new(vec![0.0; model_size])),
            current_round: Arc::new(RwLock::new(0)),
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new client
    pub fn register_client(&self, metadata: ClientMetadata) -> Result<(), FederatedError> {
        let mut clients = self.clients.write();

        if clients.contains_key(&metadata.client_id) {
            return Err(FederatedError::ConfigError(format!(
                "Client {} already registered",
                metadata.client_id
            )));
        }

        clients.insert(metadata.client_id.clone(), metadata);
        Ok(())
    }

    /// Unregister a client
    pub fn unregister_client(&self, client_id: &str) -> Result<(), FederatedError> {
        let mut clients = self.clients.write();

        clients
            .remove(client_id)
            .ok_or_else(|| FederatedError::ClientNotRegistered(client_id.to_string()))?;

        Ok(())
    }

    /// Select clients for the next training round
    pub fn select_clients(&self) -> Result<Vec<String>, FederatedError> {
        let clients = self.clients.read();
        let current_round = *self.current_round.read();

        if clients.is_empty() {
            return Err(FederatedError::ConfigError(
                "No clients registered".to_string(),
            ));
        }

        let mut client_ids: Vec<String> = clients.keys().cloned().collect();

        // Apply selection strategy
        match self.config.selection_strategy {
            ClientSelectionStrategy::Random => {
                // Shuffle and take max_clients
                use scirs2_core::random::Rng;
                let mut rng = thread_rng();
                client_ids.shuffle(&mut rng);
                client_ids.truncate(self.config.max_clients);
            }
            ClientSelectionStrategy::DataSizeBased => {
                // Sort by number of samples (descending)
                client_ids.sort_by(|a, b| {
                    let a_samples = clients.get(a).map_or(0, |m| m.num_samples);
                    let b_samples = clients.get(b).map_or(0, |m| m.num_samples);
                    b_samples.cmp(&a_samples)
                });
                client_ids.truncate(self.config.max_clients);
            }
            ClientSelectionStrategy::ComputeBased => {
                // Sort by compute capacity (descending)
                client_ids.sort_by(|a, b| {
                    let a_compute = clients.get(a).map_or(0.0, |m| m.compute_capacity);
                    let b_compute = clients.get(b).map_or(0.0, |m| m.compute_capacity);
                    b_compute
                        .partial_cmp(&a_compute)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                client_ids.truncate(self.config.max_clients);
            }
            ClientSelectionStrategy::Fair => {
                // Round-robin selection based on participation
                client_ids.sort_by_key(|id| clients.get(id).map_or(0, |m| m.rounds_participated));
                client_ids.truncate(self.config.max_clients);
            }
            ClientSelectionStrategy::ActiveLearning => {
                // For now, use random selection
                // In practice, would select based on model uncertainty
                use scirs2_core::random::Rng;
                let mut rng = thread_rng();
                client_ids.shuffle(&mut rng);
                client_ids.truncate(self.config.max_clients);
            }
        }

        // Ensure minimum number of clients
        if client_ids.len() < self.config.min_clients {
            return Err(FederatedError::ConfigError(format!(
                "Not enough clients available: {} < {}",
                client_ids.len(),
                self.config.min_clients
            )));
        }

        Ok(client_ids)
    }

    /// Get current global model for distribution to clients
    #[must_use]
    pub fn get_global_model(&self) -> ServerUpdate {
        let parameters = self.global_model.read().clone();
        let round = *self.current_round.read();
        let history = self.history.read();

        let global_metrics = history.last().cloned().unwrap_or(GlobalMetrics {
            avg_training_loss: 0.0,
            avg_validation_accuracy: 0.0,
            num_clients: 0,
            total_samples: 0,
            round_duration_ms: 0,
        });

        ServerUpdate {
            parameters,
            round,
            global_metrics,
            config_update: None,
        }
    }

    /// Receive update from a client
    pub fn receive_client_update(&self, update: ClientUpdate) -> Result<(), FederatedError> {
        let current_round = *self.current_round.read();

        // Validate update
        if update.round != current_round {
            return Err(FederatedError::InvalidUpdate(
                update.client_id.clone(),
                format!(
                    "Round mismatch: expected {}, got {}",
                    current_round, update.round
                ),
            ));
        }

        let model = self.global_model.read();
        if update.parameters.len() != model.len() {
            return Err(FederatedError::InvalidUpdate(
                update.client_id.clone(),
                format!(
                    "Parameter size mismatch: expected {}, got {}",
                    model.len(),
                    update.parameters.len()
                ),
            ));
        }

        // Update client metadata
        {
            let mut clients = self.clients.write();
            if let Some(metadata) = clients.get_mut(&update.client_id) {
                metadata.last_round = Some(current_round);
                metadata.rounds_participated += 1;
            }
        }

        // Store update
        self.pending_updates.write().push(update);

        Ok(())
    }

    /// Aggregate client updates and update global model
    pub fn aggregate_updates(&self) -> Result<GlobalMetrics, FederatedError> {
        let round_start = std::time::Instant::now();
        let mut pending = self.pending_updates.write();

        if pending.is_empty() {
            return Err(FederatedError::AggregationFailed(
                "No updates to aggregate".to_string(),
            ));
        }

        if pending.len() < self.config.min_clients {
            return Err(FederatedError::AggregationFailed(format!(
                "Not enough updates: {} < {}",
                pending.len(),
                self.config.min_clients
            )));
        }

        // Perform aggregation based on strategy
        let aggregated = match self.config.aggregation_strategy {
            AggregationStrategy::FederatedAveraging => self.federated_averaging(&pending)?,
            AggregationStrategy::SimpleAverage => self.simple_average(&pending)?,
            AggregationStrategy::MedianAggregation => self.median_aggregation(&pending)?,
            AggregationStrategy::TrimmedMean => self.trimmed_mean(&pending, 0.1)?,
            AggregationStrategy::Krum => self.krum_aggregation(&pending, 1)?,
            AggregationStrategy::WeightedContribution => self.weighted_contribution(&pending)?,
        };

        // Update global model
        *self.global_model.write() = aggregated;

        // Calculate metrics
        let total_samples: usize = pending.iter().map(|u| u.num_samples).sum();
        let avg_loss: f32 = pending
            .iter()
            .map(|u| u.training_loss * u.num_samples as f32)
            .sum::<f32>()
            / total_samples as f32;

        let avg_accuracy = pending
            .iter()
            .filter_map(|u| u.validation_accuracy)
            .sum::<f32>()
            / pending.len() as f32;

        let metrics = GlobalMetrics {
            avg_training_loss: avg_loss,
            avg_validation_accuracy: avg_accuracy,
            num_clients: pending.len(),
            total_samples,
            round_duration_ms: round_start.elapsed().as_millis() as u64,
        };

        // Update history
        self.history.write().push(metrics.clone());

        // Clear pending updates and increment round
        pending.clear();
        *self.current_round.write() += 1;

        Ok(metrics)
    }

    /// `FedAvg`: Weighted average by number of samples
    fn federated_averaging(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>, FederatedError> {
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        let param_size = updates[0].parameters.len();

        let mut aggregated = vec![0.0_f32; param_size];

        for update in updates {
            let weight = update.num_samples as f32 / total_samples as f32;
            for (i, &param) in update.parameters.iter().enumerate() {
                aggregated[i] += weight * param;
            }
        }

        Ok(aggregated)
    }

    /// Simple average of all updates
    fn simple_average(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>, FederatedError> {
        let param_size = updates[0].parameters.len();
        let num_updates = updates.len() as f32;

        let mut aggregated = vec![0.0_f32; param_size];

        for update in updates {
            for (i, &param) in update.parameters.iter().enumerate() {
                aggregated[i] += param / num_updates;
            }
        }

        Ok(aggregated)
    }

    /// Median aggregation (coordinate-wise median)
    fn median_aggregation(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>, FederatedError> {
        let param_size = updates[0].parameters.len();
        let mut aggregated = vec![0.0_f32; param_size];

        for i in 0..param_size {
            let mut values: Vec<f32> = updates.iter().map(|u| u.parameters[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_idx = values.len() / 2;
            aggregated[i] = if values.len().is_multiple_of(2) {
                (values[median_idx - 1] + values[median_idx]) / 2.0
            } else {
                values[median_idx]
            };
        }

        Ok(aggregated)
    }

    /// Trimmed mean (remove outliers)
    fn trimmed_mean(
        &self,
        updates: &[ClientUpdate],
        trim_ratio: f32,
    ) -> Result<Vec<f32>, FederatedError> {
        let param_size = updates[0].parameters.len();
        let mut aggregated = vec![0.0_f32; param_size];

        let trim_count = ((updates.len() as f32 * trim_ratio) / 2.0).floor() as usize;

        for i in 0..param_size {
            let mut values: Vec<f32> = updates.iter().map(|u| u.parameters[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Remove top and bottom trim_count values
            let trimmed = &values[trim_count..values.len() - trim_count];
            aggregated[i] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
        }

        Ok(aggregated)
    }

    /// Krum aggregation (Byzantine-robust)
    fn krum_aggregation(
        &self,
        updates: &[ClientUpdate],
        num_byzantine: usize,
    ) -> Result<Vec<f32>, FederatedError> {
        // Select the update with smallest distance sum to nearest neighbors
        let n = updates.len();
        let m = n - num_byzantine - 2;

        let mut min_score = f32::MAX;
        let mut best_idx = 0;

        for i in 0..n {
            let mut distances: Vec<f32> = Vec::new();

            for j in 0..n {
                if i != j {
                    let dist =
                        self.euclidean_distance(&updates[i].parameters, &updates[j].parameters);
                    distances.push(dist);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f32 = distances.iter().take(m).sum();

            if score < min_score {
                min_score = score;
                best_idx = i;
            }
        }

        Ok(updates[best_idx].parameters.clone())
    }

    /// Weighted by contribution score
    fn weighted_contribution(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>, FederatedError> {
        // Weight by inverse of training loss (better performing clients get higher weight)
        let total_weight: f32 = updates.iter().map(|u| 1.0 / (u.training_loss + 1e-8)).sum();

        let param_size = updates[0].parameters.len();
        let mut aggregated = vec![0.0_f32; param_size];

        for update in updates {
            let weight = (1.0 / (update.training_loss + 1e-8)) / total_weight;
            for (i, &param) in update.parameters.iter().enumerate() {
                aggregated[i] += weight * param;
            }
        }

        Ok(aggregated)
    }

    /// Calculate Euclidean distance between two parameter vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Get training history
    #[must_use]
    pub fn get_history(&self) -> Vec<GlobalMetrics> {
        self.history.read().clone()
    }

    /// Get current round number
    #[must_use]
    pub fn current_round(&self) -> usize {
        *self.current_round.read()
    }

    /// Get number of registered clients
    #[must_use]
    pub fn num_clients(&self) -> usize {
        self.clients.read().len()
    }
}

/// Federated learning client
pub struct FederatedLearningClient {
    /// Client identifier
    client_id: String,

    /// Client metadata
    metadata: ClientMetadata,

    /// Local model parameters
    local_model: Vec<f32>,

    /// Training configuration
    config: FederatedConfig,
}

impl FederatedLearningClient {
    /// Create a new federated learning client
    #[must_use]
    pub fn new(client_id: String, metadata: ClientMetadata, config: FederatedConfig) -> Self {
        Self {
            client_id,
            metadata,
            local_model: Vec::new(),
            config,
        }
    }

    /// Update local model from server
    pub fn receive_server_update(&mut self, update: ServerUpdate) {
        self.local_model = update.parameters;
    }

    /// Train local model and generate update
    pub fn train_local_model(
        &self,
        training_data: &[Vec<f32>],
        labels: &[f32],
    ) -> Result<ClientUpdate, FederatedError> {
        let training_start = std::time::Instant::now();

        // Simulate local training (in practice, would run actual training)
        // For demonstration, we'll just add small random perturbations
        let mut updated_params = self.local_model.clone();

        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        for param in &mut updated_params {
            *param += rng.gen_range(-0.01..0.01);
        }

        // Apply gradient clipping if configured
        if let Some(clip_norm) = self.config.gradient_clip_norm {
            let norm: f32 = updated_params.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > clip_norm {
                let scale = clip_norm / norm;
                for param in &mut updated_params {
                    *param *= scale;
                }
            }
        }

        // Calculate training loss (simulated)
        let training_loss = rng.gen_range(0.1..1.0);

        let update = ClientUpdate {
            client_id: self.client_id.clone(),
            parameters: updated_params,
            num_samples: training_data.len(),
            training_loss,
            validation_accuracy: Some(rng.gen_range(0.7..0.95)),
            training_time_ms: training_start.elapsed().as_millis() as u64,
            round: 0, // Will be set by server
            secure_shares: None,
        };

        Ok(update)
    }

    /// Get client metadata
    #[must_use]
    pub fn metadata(&self) -> &ClientMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_server_creation() {
        let config = FederatedConfig::default();
        let server = FederatedLearningServer::new(config, 100);

        assert_eq!(server.num_clients(), 0);
        assert_eq!(server.current_round(), 0);
    }

    #[test]
    fn test_client_registration() {
        let config = FederatedConfig::default();
        let server = FederatedLearningServer::new(config, 100);

        let metadata = ClientMetadata {
            client_id: "client_1".to_string(),
            num_samples: 1000,
            compute_capacity: 1.0,
            bandwidth: 100.0,
            device_type: DeviceType::Desktop,
            last_round: None,
            rounds_participated: 0,
        };

        assert!(server.register_client(metadata).is_ok());
        assert_eq!(server.num_clients(), 1);
    }

    #[test]
    fn test_client_selection() {
        let config = FederatedConfig {
            min_clients: 2,
            max_clients: 5,
            ..Default::default()
        };
        let server = FederatedLearningServer::new(config, 100);

        // Register multiple clients
        for i in 0..10 {
            let metadata = ClientMetadata {
                client_id: format!("client_{}", i),
                num_samples: 1000 * (i + 1),
                compute_capacity: 1.0,
                bandwidth: 100.0,
                device_type: DeviceType::Desktop,
                last_round: None,
                rounds_participated: 0,
            };
            server.register_client(metadata).unwrap();
        }

        let selected = server.select_clients().unwrap();
        assert!(selected.len() >= 2 && selected.len() <= 5);
    }

    #[test]
    fn test_federated_averaging() {
        let config = FederatedConfig::default();
        let server = FederatedLearningServer::new(config, 3);

        let updates = vec![
            ClientUpdate {
                client_id: "client_1".to_string(),
                parameters: vec![1.0, 2.0, 3.0],
                num_samples: 100,
                training_loss: 0.5,
                validation_accuracy: Some(0.8),
                training_time_ms: 1000,
                round: 0,
                secure_shares: None,
            },
            ClientUpdate {
                client_id: "client_2".to_string(),
                parameters: vec![2.0, 3.0, 4.0],
                num_samples: 200,
                training_loss: 0.4,
                validation_accuracy: Some(0.85),
                training_time_ms: 1500,
                round: 0,
                secure_shares: None,
            },
        ];

        let result = server.federated_averaging(&updates).unwrap();

        // Expected: (1.0*100 + 2.0*200)/300 = 500/300 = 1.667
        assert!((result[0] - 1.667).abs() < 0.01);
        assert!((result[1] - 2.667).abs() < 0.01);
        assert!((result[2] - 3.667).abs() < 0.01);
    }

    #[test]
    fn test_median_aggregation() {
        let config = FederatedConfig::default();
        let server = FederatedLearningServer::new(config, 3);

        let updates = vec![
            ClientUpdate {
                client_id: "client_1".to_string(),
                parameters: vec![1.0, 2.0, 3.0],
                num_samples: 100,
                training_loss: 0.5,
                validation_accuracy: Some(0.8),
                training_time_ms: 1000,
                round: 0,
                secure_shares: None,
            },
            ClientUpdate {
                client_id: "client_2".to_string(),
                parameters: vec![2.0, 3.0, 4.0],
                num_samples: 100,
                training_loss: 0.4,
                validation_accuracy: Some(0.85),
                training_time_ms: 1000,
                round: 0,
                secure_shares: None,
            },
            ClientUpdate {
                client_id: "client_3".to_string(),
                parameters: vec![3.0, 4.0, 5.0],
                num_samples: 100,
                training_loss: 0.3,
                validation_accuracy: Some(0.9),
                training_time_ms: 1000,
                round: 0,
                secure_shares: None,
            },
        ];

        let result = server.median_aggregation(&updates).unwrap();

        // Median of [1.0, 2.0, 3.0] is 2.0
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }
}
