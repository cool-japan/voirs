//! Cloud integration and distributed processing for VoiRS.
//!
//! This module provides cloud-based functionality for VoiRS CLI, enabling
//! distributed synthesis, model synchronization, and collaborative workflows.
//! It supports various cloud storage providers and distributed processing
//! architectures for scalable text-to-speech synthesis.
//!
//! ## Features
//!
//! - **Cloud Storage**: Synchronize models, configurations, and audio files
//! - **Distributed Processing**: Scale synthesis across multiple cloud instances
//! - **API Integration**: Connect to external TTS and translation services
//! - **Load Balancing**: Distribute synthesis requests efficiently
//! - **Collaborative Workflows**: Share voices and configurations across teams
//!
//! ## Modules
//!
//! - [`api`]: External API integrations and service connections
//! - [`distributed`]: Distributed processing and load balancing
//! - [`storage`]: Cloud storage synchronization and backup
//!
//! ## Example
//!
//! ```rust,no_run
//! use voirs_cli::cloud::storage::{CloudStorageManager, CloudStorageConfig, SyncDirection};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = CloudStorageConfig::default();
//! let cache_dir = PathBuf::from("/tmp/voirs_cache");
//! let mut storage = CloudStorageManager::new(config, cache_dir)?;
//! storage.add_to_sync(
//!     PathBuf::from("model.bin"),
//!     "models/model.bin".to_string(),
//!     SyncDirection::Upload
//! ).await?;
//! let _result = storage.sync().await?;
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod azure_backend;
pub mod distributed;
pub mod error;
pub mod gcp_backend;
pub mod s3_backend;
pub mod sigv4;
pub mod storage;

pub use api::*;
pub use distributed::*;
pub use error::CloudStorageError;
pub use storage::*;
