//! External integrations for voirs-dataset
//!
//! This module provides integrations with various external services and platforms,
//! including cloud storage providers, version control systems, and MLOps platforms.

pub mod cloud;
pub mod git;
pub mod mlops;

// Re-export commonly used types
pub use cloud::{CloudProvider, CloudStorage, CloudStorageConfig};
pub use git::{GitConfig, GitLFS, GitRepository};
pub use mlops::{MLOpsConfig, MLOpsIntegration, MLOpsProvider};
