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
//! ```rust,no_run
//! use voirs_recognizer::privacy::{FederatedLearningServer, DifferentialPrivacy};
//!
//! // Create federated learning server
//! let server = FederatedLearningServer::new(/* config */);
//!
//! // Apply differential privacy to training data
//! let dp = DifferentialPrivacy::new(epsilon: 1.0, delta: 1e-5);
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
