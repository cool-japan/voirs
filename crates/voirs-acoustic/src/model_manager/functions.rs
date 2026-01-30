//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::backends::{create_default_loader, AcousticModelLoader};
use crate::config::{ModelArchitecture, ModelConfig};
use crate::{AcousticError, AcousticModel, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manager::types::{CacheStats, ModelManager, ModelRegistry, PipelineSettings};
    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::default();
        assert!(registry.models.is_empty());
        assert!(registry.pipelines.is_empty());
        assert_eq!(registry.metadata.version, "1.0.0");
    }
    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new().await;
        assert!(manager.is_ok());
        let manager = manager.unwrap();
        assert!(manager.registry().models.is_empty());
    }
    #[test]
    fn test_pipeline_settings_default() {
        let settings = PipelineSettings::default();
        assert_eq!(settings.real_time_factor, 0.3);
        assert_eq!(settings.quality_level, 0.8);
        assert!(settings.supports_streaming);
    }
    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            loaded_models: 5,
            cache_size_mb: 512,
        };
        assert_eq!(stats.loaded_models, 5);
        assert_eq!(stats.cache_size_mb, 512);
    }
}
