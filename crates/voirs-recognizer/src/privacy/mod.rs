//! Privacy-preserving techniques for speech recognition.
//!
//! This module provides state-of-the-art privacy-preserving mechanisms for ASR systems,
//! including federated learning, differential privacy, and encrypted inference.
//!
//! # Features
//!
//! - **Federated Learning**: Train models across distributed clients without centralizing data
//! - **Differential Privacy**: Add calibrated noise to protect individual privacy
//! - **Encrypted Inference**: Perform predictions on encrypted data using homomorphic encryption
//!
//! # Examples
//!
//! ```no_run
//! use voirs_recognizer::privacy::{DifferentialPrivacy, DPConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Apply differential privacy to training data (epsilon=1.0, delta=1e-5)
//! let config = DPConfig { epsilon: 1.0, delta: 1e-5, ..DPConfig::default() };
//! let dp = DifferentialPrivacy::new(config)?;
//! # Ok(())
//! # }
//! ```

pub mod differential_privacy;
pub mod encrypted_inference;
pub mod federated_learning;

pub use federated_learning::{
    AggregationStrategy, ClientUpdate, FederatedConfig, FederatedError, FederatedLearningClient,
    FederatedLearningServer, ServerUpdate,
};

pub use differential_privacy::{
    DPConfig, DPError, DPMechanism, DifferentialPrivacy, GaussianMechanism, LaplaceMechanism,
    NoiseDistribution, PrivacyBudget,
};

pub use encrypted_inference::{
    EncryptedInference, EncryptionError, EncryptionScheme, HomomorphicEncryption,
    SecureInferenceConfig,
};
