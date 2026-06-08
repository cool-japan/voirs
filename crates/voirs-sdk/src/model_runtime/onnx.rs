//! OxiONNX session wrapper with profiling, optimization, and convenience APIs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use oxionnx::{ModelInfo, NodeProfile, OptLevel, Session, Tensor};

use crate::error::VoirsError;

/// Configuration for creating an ONNX session.
#[derive(Debug, Clone)]
pub struct OnnxSessionConfig {
    /// Optimization level for graph optimization passes.
    pub opt_level: OptLevel,
    /// Whether to enable per-node profiling during inference.
    pub enable_profiling: bool,
    /// Whether to enable memory pool for buffer reuse.
    pub enable_memory_pool: bool,
}

impl Default for OnnxSessionConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
        }
    }
}

/// Thread-safe wrapper around an OxiONNX Session with profiling and convenience APIs.
pub struct OnnxSession {
    session: Arc<RwLock<Session>>,
    model_path: Option<PathBuf>,
    config: OnnxSessionConfig,
}

impl OnnxSession {
    /// Load an ONNX model from a file path with the given configuration.
    pub fn from_file(path: &Path, config: &OnnxSessionConfig) -> Result<Self, VoirsError> {
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);

        if config.enable_profiling {
            builder = builder.with_profiling();
        }

        let session = builder.load(path).map_err(|e| {
            VoirsError::ModelLoadError(format!(
                "Failed to load ONNX model '{}': {}",
                path.display(),
                e
            ))
        })?;

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            model_path: Some(path.to_path_buf()),
            config: config.clone(),
        })
    }

    /// Load an ONNX model from raw bytes with the given configuration.
    pub fn from_bytes(bytes: &[u8], config: &OnnxSessionConfig) -> Result<Self, VoirsError> {
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);

        if config.enable_profiling {
            builder = builder.with_profiling();
        }

        let session = builder.load_from_bytes(bytes).map_err(|e| {
            VoirsError::ModelLoadError(format!("Failed to load ONNX model from bytes: {}", e))
        })?;

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            model_path: None,
            config: config.clone(),
        })
    }

    /// Run inference with the given named inputs.
    pub fn run(
        &self,
        inputs: &HashMap<&str, Tensor>,
    ) -> Result<HashMap<String, Tensor>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        session
            .run(inputs)
            .map_err(|e| VoirsError::InferenceError(format!("ONNX inference failed: {}", e)))
    }

    /// Convenience: run inference with a single named input.
    pub fn run_one(
        &self,
        name: &str,
        input: Tensor,
    ) -> Result<HashMap<String, Tensor>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        session
            .run_one(name, input)
            .map_err(|e| VoirsError::InferenceError(format!("ONNX inference failed: {}", e)))
    }

    /// Return the model's graph input names.
    pub fn input_names(&self) -> Result<Vec<String>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.input_names().to_vec())
    }

    /// Return the model's graph output names.
    pub fn output_names(&self) -> Result<Vec<String>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.output_names().to_vec())
    }

    /// Return summary information about the loaded model.
    pub fn model_info(&self) -> Result<ModelInfo, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.model_info())
    }

    /// Return profiling results from previous inference runs.
    /// Returns None if profiling was not enabled.
    pub fn profiling_results(&self) -> Result<Option<Vec<NodeProfile>>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.profiling_results())
    }

    /// Generate a profiling summary from collected profiling data.
    pub fn profiling_summary(&self) -> Result<Option<ProfilingSummary>, VoirsError> {
        let profiles = self.profiling_results()?;
        Ok(profiles.map(|p| ProfilingSummary::from_node_profiles(&p)))
    }

    /// Export the computation graph as a DOT (Graphviz) string.
    pub fn export_dot(&self) -> Result<String, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.export_dot())
    }

    /// Return estimated peak memory usage in bytes for intermediate tensors.
    pub fn estimated_memory_bytes(&self) -> Result<Option<usize>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session.estimated_memory_bytes())
    }

    /// Return the model file path, if loaded from a file.
    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    /// Return the configuration used to create this session.
    pub fn config(&self) -> &OnnxSessionConfig {
        &self.config
    }

    /// Get a clone of the inner `Arc<RwLock<Session>>` for sharing.
    pub fn inner(&self) -> Arc<RwLock<Session>> {
        Arc::clone(&self.session)
    }

    /// Return a reference to the model's weight tensors (name and shape pairs).
    pub fn weights(&self) -> Result<Vec<(String, Vec<usize>)>, VoirsError> {
        let session = self.session.read().map_err(|e| {
            VoirsError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;
        Ok(session
            .weights()
            .iter()
            .map(|(k, v)| (k.clone(), v.shape.clone()))
            .collect())
    }
}

impl Clone for OnnxSession {
    fn clone(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            model_path: self.model_path.clone(),
            config: self.config.clone(),
        }
    }
}

/// Aggregated profiling summary from inference runs.
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    /// Total duration across all profiled nodes.
    pub total_duration: Duration,
    /// Top nodes sorted by execution time (descending).
    pub top_nodes: Vec<(String, Duration)>,
    /// Aggregate duration per operator type.
    pub op_type_durations: HashMap<String, Duration>,
    /// Node with the longest execution time.
    pub bottleneck_node: Option<String>,
}

impl ProfilingSummary {
    /// Create a summary from a list of node profiles.
    pub fn from_node_profiles(profiles: &[NodeProfile]) -> Self {
        let total_duration: Duration = profiles.iter().map(|p| p.duration).sum();

        let mut op_type_durations: HashMap<String, Duration> = HashMap::new();
        for profile in profiles {
            let entry = op_type_durations
                .entry(profile.op_type.clone())
                .or_insert(Duration::ZERO);
            *entry += profile.duration;
        }

        let mut top_nodes: Vec<(String, Duration)> = profiles
            .iter()
            .map(|p| (p.node_name.clone(), p.duration))
            .collect();
        top_nodes.sort_by_key(|b| std::cmp::Reverse(b.1));

        let bottleneck_node = top_nodes.first().map(|(name, _)| name.clone());

        // Keep top 20
        top_nodes.truncate(20);

        Self {
            total_duration,
            top_nodes,
            op_type_durations,
            bottleneck_node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_session_config_default() {
        let config = OnnxSessionConfig::default();
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
    }

    #[test]
    fn test_profiling_summary_empty() {
        let summary = ProfilingSummary::from_node_profiles(&[]);
        assert_eq!(summary.total_duration, Duration::ZERO);
        assert!(summary.top_nodes.is_empty());
        assert!(summary.op_type_durations.is_empty());
        assert!(summary.bottleneck_node.is_none());
    }

    #[test]
    fn test_profiling_summary_from_profiles() {
        let profiles = vec![
            NodeProfile {
                node_name: "matmul_1".to_string(),
                op_type: "MatMul".to_string(),
                duration: Duration::from_millis(10),
                output_shapes: vec![vec![1, 512]],
            },
            NodeProfile {
                node_name: "relu_1".to_string(),
                op_type: "Relu".to_string(),
                duration: Duration::from_millis(2),
                output_shapes: vec![vec![1, 512]],
            },
            NodeProfile {
                node_name: "matmul_2".to_string(),
                op_type: "MatMul".to_string(),
                duration: Duration::from_millis(8),
                output_shapes: vec![vec![1, 256]],
            },
        ];

        let summary = ProfilingSummary::from_node_profiles(&profiles);
        assert_eq!(summary.total_duration, Duration::from_millis(20));
        assert_eq!(summary.bottleneck_node.as_deref(), Some("matmul_1"));
        assert_eq!(summary.top_nodes.len(), 3);
        assert_eq!(summary.top_nodes[0].0, "matmul_1");
        // MatMul total: 18ms
        assert_eq!(
            *summary
                .op_type_durations
                .get("MatMul")
                .expect("MatMul should exist"),
            Duration::from_millis(18)
        );
    }
}
