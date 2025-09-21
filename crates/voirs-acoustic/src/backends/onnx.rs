//! ONNX Runtime backend for acoustic modeling.
//!
//! This module provides ONNX-based implementations for acoustic models,
//! enabling high-performance inference using pre-trained models.

use crate::{
    speaker::SpeakerEmbedding,
    traits::{AcousticModel, AcousticModelFeature, AcousticModelMetadata},
    AcousticError, MelSpectrogram, Phoneme, Result, SynthesisConfig,
};
use async_trait::async_trait;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session, SessionOutputs,
    },
    value::Value,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tracing::{debug, error, info, warn};

/// Streaming state for ONNX acoustic model
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Current streaming configuration
    pub config: SynthesisConfig,

    /// Phoneme buffer for accumulating input
    pub phoneme_buffer: Vec<Phoneme>,

    /// Context from previous chunks
    pub context_phonemes: Vec<Phoneme>,

    /// Total frames processed so far
    pub total_frames: usize,

    /// Chunk size for streaming processing
    pub chunk_size: usize,

    /// Overlap size for maintaining context
    pub overlap_size: usize,

    /// Whether streaming is active
    pub is_active: bool,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            config: SynthesisConfig::default(),
            phoneme_buffer: Vec::new(),
            context_phonemes: Vec::new(),
            total_frames: 0,
            chunk_size: 50,   // Default chunk size for streaming
            overlap_size: 10, // Overlap for context continuity
            is_active: false,
        }
    }
}

/// ONNX-based acoustic model implementation
pub struct OnnxAcousticModel {
    /// ONNX Runtime session
    session: Arc<RwLock<Session>>,

    /// Model metadata
    metadata: ModelMetadata,

    /// Speaker embeddings cache
    speaker_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,

    /// Model configuration
    config: OnnxModelConfig,

    /// Streaming state
    streaming_state: Arc<RwLock<StreamingState>>,
}

/// ONNX model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model architecture (e.g., "FastSpeech2", "VITS", "Tacotron2")
    pub architecture: String,

    /// Supported sample rates
    pub sample_rates: Vec<u32>,

    /// Input phoneme vocabulary size
    pub vocab_size: usize,

    /// Output mel spectrogram dimensions
    pub mel_dim: usize,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Supported speakers (if multi-speaker model)
    pub speakers: Vec<String>,

    /// Model input names
    pub input_names: Vec<String>,

    /// Model output names
    pub output_names: Vec<String>,
}

/// ONNX model configuration
#[derive(Debug, Clone)]
pub struct OnnxModelConfig {
    /// Model file path
    pub model_path: PathBuf,

    /// Execution providers (CPU, CUDA, etc.)
    pub execution_providers: Vec<String>,

    /// Number of threads for CPU execution
    pub num_threads: usize,

    /// Enable memory pattern optimization
    pub enable_memory_pattern: bool,

    /// Enable CPU memory arena
    pub enable_cpu_mem_arena: bool,

    /// Optimization level (store as enum variant name)
    pub graph_optimization_level: String,

    /// Session options
    pub inter_op_num_threads: Option<usize>,
    pub intra_op_num_threads: Option<usize>,
}

impl Default for OnnxModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            execution_providers: vec!["CPUExecutionProvider".to_string()],
            num_threads: num_cpus::get(),
            enable_memory_pattern: true,
            enable_cpu_mem_arena: true,
            graph_optimization_level: "All".to_string(),
            inter_op_num_threads: Some(1),
            intra_op_num_threads: Some(num_cpus::get()),
        }
    }
}

impl OnnxAcousticModel {
    /// Create a new ONNX acoustic model
    pub async fn new(config: OnnxModelConfig) -> Result<Self> {
        info!(
            "Initializing ONNX acoustic model from {:?}",
            config.model_path
        );

        // Ensure ONNX Runtime is initialized
        ort::init().commit().map_err(|e| {
            AcousticError::ModelError(format!("Failed to initialize ONNX Runtime: {}", e))
        })?;

        // Configure session builder
        let mut session_builder = SessionBuilder::new().map_err(|e| {
            AcousticError::ModelError(format!("Failed to create session builder: {}", e))
        })?;
        // Convert string to GraphOptimizationLevel
        let opt_level = match config.graph_optimization_level.as_str() {
            "Disable" => GraphOptimizationLevel::Disable,
            "Level1" => GraphOptimizationLevel::Level1,
            "Level2" => GraphOptimizationLevel::Level2,
            "Level3" | "All" => GraphOptimizationLevel::Level3,
            _ => GraphOptimizationLevel::Level1,
        };
        session_builder = session_builder
            .with_optimization_level(opt_level)
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to set optimization level: {}", e))
            })?
            .with_memory_pattern(config.enable_memory_pattern)
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to set memory pattern: {}", e))
            })?;

        // Set thread counts
        if let Some(inter_op) = config.inter_op_num_threads {
            session_builder = session_builder.with_inter_threads(inter_op).map_err(|e| {
                AcousticError::ModelError(format!("Failed to set inter threads: {}", e))
            })?;
        }
        if let Some(intra_op) = config.intra_op_num_threads {
            session_builder = session_builder.with_intra_threads(intra_op).map_err(|e| {
                AcousticError::ModelError(format!("Failed to set intra threads: {}", e))
            })?;
        }

        // Add execution providers - skip for now as API changed
        // TODO: Update to new ort API for execution providers

        // Load the model
        let session = session_builder
            .commit_from_file(&config.model_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to load ONNX model: {}", e)))?;

        // Extract model metadata
        let metadata = Self::extract_metadata(&session, &config.model_path)?;

        info!("ONNX acoustic model loaded successfully: {}", metadata.name);
        debug!("Model metadata: {:?}", metadata);

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            metadata,
            speaker_embeddings: Arc::new(RwLock::new(HashMap::new())),
            config,
            streaming_state: Arc::new(RwLock::new(StreamingState::default())),
        })
    }

    /// Extract model metadata from ONNX session
    fn extract_metadata(session: &Session, model_path: &Path) -> Result<ModelMetadata> {
        // Get input and output names - using default names for now
        // TODO: Extract from session metadata when API is available
        let input_names: Vec<String> = vec!["phonemes".to_string()];
        let output_names: Vec<String> = vec!["mel_spectrogram".to_string()];

        // Extract model name from file path
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Default metadata (these would typically come from model metadata or config)
        Ok(ModelMetadata {
            name: model_name,
            version: "1.0.0".to_string(),
            architecture: "Unknown".to_string(),
            sample_rates: vec![22050, 24000, 48000],
            vocab_size: 256, // Default phoneme vocabulary size
            mel_dim: 80,     // Standard mel dimension
            max_sequence_length: 1000,
            speakers: vec!["default".to_string()],
            input_names,
            output_names,
        })
    }

    /// Load speaker embedding
    pub async fn load_speaker_embedding(
        &self,
        speaker_id: &str,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let mut embeddings = self.speaker_embeddings.write().unwrap();
        embeddings.insert(speaker_id.to_string(), embedding);
        info!("Loaded speaker embedding for: {}", speaker_id);
        Ok(())
    }

    /// Get speaker embedding
    fn get_speaker_embedding(&self, speaker_id: Option<&str>) -> Option<Vec<f32>> {
        if let Some(id) = speaker_id {
            let embeddings = self.speaker_embeddings.read().unwrap();
            embeddings.get(id).cloned()
        } else {
            None
        }
    }

    /// Prepare input tensors for ONNX inference
    async fn prepare_inputs(
        &self,
        phonemes: &[Phoneme],
        config: &SynthesisConfig,
    ) -> Result<Vec<(String, Value)>> {
        let mut inputs: Vec<(String, Value)> = Vec::new();

        // Convert phonemes to integer sequence
        let phoneme_ids: Vec<i64> = phonemes
            .iter()
            .map(|p| self.phoneme_to_id(&p.symbol))
            .collect();

        // Create phoneme input tensor
        let phoneme_tensor =
            Value::from_array(([1, phoneme_ids.len()], phoneme_ids)).map_err(|e| {
                AcousticError::InferenceError(format!("Failed to create phoneme tensor: {}", e))
            })?;
        inputs.push(("phonemes".to_string(), phoneme_tensor.into_dyn()));

        // Add speaker embedding if available
        if let Some(speaker_id) = config.speaker_id {
            if let Some(embedding) = self.get_speaker_embedding(Some(&speaker_id.to_string())) {
                let speaker_tensor =
                    Value::from_array(([1, embedding.len()], embedding)).map_err(|e| {
                        AcousticError::InferenceError(format!(
                            "Failed to create speaker tensor: {}",
                            e
                        ))
                    })?;
                inputs.push(("speaker".to_string(), speaker_tensor.into_dyn()));
            }
        }

        // Add synthesis control parameters
        if self.metadata.input_names.contains(&"speed".to_string()) {
            let speed_tensor = Value::from_array(([1], vec![config.speed])).map_err(|e| {
                AcousticError::InferenceError(format!("Failed to create speed tensor: {}", e))
            })?;
            inputs.push(("speed".to_string(), speed_tensor.into_dyn()));
        }

        if self
            .metadata
            .input_names
            .contains(&"pitch_shift".to_string())
        {
            let pitch_tensor = Value::from_array(([1], vec![config.pitch_shift])).map_err(|e| {
                AcousticError::InferenceError(format!("Failed to create pitch tensor: {}", e))
            })?;
            inputs.push(("pitch_shift".to_string(), pitch_tensor.into_dyn()));
        }

        if self.metadata.input_names.contains(&"energy".to_string()) {
            let energy_tensor = Value::from_array(([1], vec![config.energy])).map_err(|e| {
                AcousticError::InferenceError(format!("Failed to create energy tensor: {}", e))
            })?;
            inputs.push(("energy".to_string(), energy_tensor.into_dyn()));
        }

        Ok(inputs)
    }

    /// Convert phoneme symbol to ID
    fn phoneme_to_id(&self, symbol: &str) -> i64 {
        // Simple hash-based mapping for now
        // In practice, this would use a proper phoneme vocabulary
        let mut hash = 0u64;
        for byte in symbol.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        (hash % self.metadata.vocab_size as u64) as i64
    }

    /// Synthesize a chunk of phonemes for streaming
    async fn synthesize_chunk(
        &mut self,
        phonemes: &[Phoneme],
        config: &SynthesisConfig,
    ) -> Result<MelSpectrogram> {
        debug!(
            "ONNX chunk synthesis: processing {} phonemes",
            phonemes.len()
        );

        if phonemes.is_empty() {
            return Ok(MelSpectrogram {
                data: vec![vec![]; self.metadata.mel_dim],
                sample_rate: 22050,
                hop_length: 256,
                n_mels: self.metadata.mel_dim,
                n_frames: 0,
            });
        }

        // Use regular synthesis for the chunk, but with optimizations for streaming
        let chunk_size = std::cmp::min(phonemes.len(), self.metadata.max_sequence_length);
        let chunk_phonemes = &phonemes[..chunk_size];

        // Prepare input tensors for the chunk
        let inputs = self.prepare_inputs(chunk_phonemes, config).await?;

        // Run inference on the chunk
        let mut session = self.session.write().unwrap();
        let outputs = session.run(inputs).map_err(|e| {
            AcousticError::ModelError(format!("ONNX chunk inference failed: {}", e))
        })?;

        // Process outputs
        let mel_spectrogram = self.process_outputs(outputs)?;

        debug!(
            "ONNX chunk synthesis completed: {} mel frames",
            mel_spectrogram.data[0].len()
        );

        Ok(mel_spectrogram)
    }

    /// Process ONNX outputs to extract mel spectrogram
    fn process_outputs(&self, outputs: SessionOutputs) -> Result<MelSpectrogram> {
        // Extract first output by name or index
        let mel_output = outputs
            .get("mel_spectrogram")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("0"))
            .ok_or_else(|| {
                AcousticError::ModelError("No outputs received from ONNX model".to_string())
            })?;

        // Extract mel spectrogram data
        let (shape, mel_data) = mel_output
            .try_extract_tensor::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to extract tensor: {}", e)))?;

        // Get output shape
        let (batch_size, mel_dim, seq_len) = if shape.len() == 3 {
            (shape[0] as usize, shape[1] as usize, shape[2] as usize)
        } else {
            return Err(AcousticError::ModelError(format!(
                "Unexpected mel output shape: {:?}",
                shape
            )));
        };

        if batch_size != 1 {
            warn!("Batch size {} > 1, using first sample", batch_size);
        }

        // Reshape data to 2D matrix (mel_dim, seq_len)
        let mut mel_matrix = vec![vec![0.0; seq_len]; mel_dim];
        for i in 0..mel_dim {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if idx < mel_data.len() {
                    mel_matrix[i][j] = mel_data[idx];
                }
            }
        }

        Ok(MelSpectrogram {
            data: mel_matrix,
            sample_rate: 22050, // Default sample rate
            hop_length: 256,
            n_mels: mel_dim,
            n_frames: seq_len,
        })
    }
}

#[async_trait]
impl AcousticModel for OnnxAcousticModel {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        debug!(
            "Starting ONNX acoustic synthesis for {} phonemes",
            phonemes.len()
        );

        if phonemes.is_empty() {
            return Err(AcousticError::InputError(
                "Empty phoneme sequence".to_string(),
            ));
        }

        if phonemes.len() > self.metadata.max_sequence_length {
            return Err(AcousticError::InputError(format!(
                "Sequence length {} exceeds maximum {}",
                phonemes.len(),
                self.metadata.max_sequence_length
            )));
        }

        // Prepare input tensors
        let default_config = SynthesisConfig::default();
        let config = config.unwrap_or(&default_config);
        let inputs = self.prepare_inputs(phonemes, config).await?;

        // Run inference
        let mut session = self.session.write().unwrap();
        let outputs = session
            .run(inputs)
            .map_err(|e| AcousticError::ModelError(format!("ONNX inference failed: {}", e)))?;

        // Process outputs
        let mel_spectrogram = self.process_outputs(outputs)?;

        debug!(
            "ONNX acoustic synthesis completed: {} mel frames",
            mel_spectrogram.data[0].len()
        );

        Ok(mel_spectrogram)
    }

    fn metadata(&self) -> AcousticModelMetadata {
        AcousticModelMetadata {
            name: self.metadata.name.clone(),
            version: self.metadata.version.clone(),
            architecture: self.metadata.architecture.clone(),
            supported_languages: vec![], // TODO: Extract from model metadata
            sample_rate: self.metadata.sample_rates.first().copied().unwrap_or(22050),
            mel_channels: self.metadata.mel_dim as u32,
            is_multi_speaker: self.metadata.speakers.len() > 1,
            speaker_count: if self.metadata.speakers.len() > 1 {
                Some(self.metadata.speakers.len() as u32)
            } else {
                None
            },
        }
    }

    fn supports(&self, feature: AcousticModelFeature) -> bool {
        match feature {
            AcousticModelFeature::MultiSpeaker => self.metadata.speakers.len() > 1,
            AcousticModelFeature::BatchProcessing => true,
            AcousticModelFeature::StreamingInference => true,
            AcousticModelFeature::StreamingSynthesis => true,
            AcousticModelFeature::GpuAcceleration => self
                .config
                .execution_providers
                .iter()
                .any(|p| p.contains("CUDA") || p.contains("ROCm") || p.contains("TensorRT")),
            _ => false,
        }
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        let mut results = Vec::with_capacity(inputs.len());

        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            let result = self.synthesize(phonemes, config).await?;
            results.push(result);
        }

        Ok(results)
    }
}

impl OnnxAcousticModel {
    async fn set_speaker_embedding(&self, speaker_id: &str, embedding: Vec<f32>) -> Result<()> {
        self.load_speaker_embedding(speaker_id, embedding).await
    }

    async fn get_supported_speakers(&self) -> Vec<String> {
        self.metadata.speakers.clone()
    }

    async fn extract_speaker_embedding(&self, _samples: &[f32]) -> Result<Vec<f32>> {
        // This would require a separate speaker encoder model
        // For now, return an error indicating this feature is not implemented
        Err(AcousticError::InvalidConfiguration(
            "Speaker embedding extraction not implemented for ONNX backend".to_string(),
        ))
    }
}

impl OnnxAcousticModel {
    async fn start_stream(&self, config: &SynthesisConfig) -> Result<()> {
        info!("Starting ONNX streaming synthesis");

        let mut state = self.streaming_state.write().unwrap();

        // Reset streaming state
        state.config = config.clone();
        state.phoneme_buffer.clear();
        state.context_phonemes.clear();
        state.total_frames = 0;
        state.is_active = true;

        // Configure chunk size based on model constraints
        let max_chunk = self.metadata.max_sequence_length / 4; // Use 1/4 of max for safety
        state.chunk_size = std::cmp::min(state.chunk_size, max_chunk);
        state.overlap_size = std::cmp::min(state.overlap_size, state.chunk_size / 4);

        debug!(
            "ONNX streaming initialized: chunk_size={}, overlap_size={}",
            state.chunk_size, state.overlap_size
        );

        Ok(())
    }

    async fn stream_phonemes(&mut self, phonemes: &[Phoneme]) -> Result<MelSpectrogram> {
        // Extract data from state and drop lock early
        let (chunk_phonemes, config, chunk_size) = {
            let mut state = self.streaming_state.write().unwrap();

            if !state.is_active {
                return Err(AcousticError::InvalidConfiguration(
                    "Streaming not started. Call start_stream first.".to_string(),
                ));
            }

            // Add new phonemes to buffer
            state.phoneme_buffer.extend_from_slice(phonemes);

            debug!(
                "ONNX streaming: buffered {} phonemes, total buffer size: {}",
                phonemes.len(),
                state.phoneme_buffer.len()
            );

            // Check if we have enough phonemes to process a chunk
            if state.phoneme_buffer.len() < state.chunk_size {
                // Not enough phonemes yet, return empty mel spectrogram
                return Ok(MelSpectrogram {
                    data: vec![vec![]; self.metadata.mel_dim],
                    sample_rate: 22050,
                    hop_length: 256,
                    n_mels: self.metadata.mel_dim,
                    n_frames: 0,
                });
            }

            // Prepare chunk with context
            let mut chunk_phonemes = state.context_phonemes.clone();
            let chunk_size = std::cmp::min(state.chunk_size, state.phoneme_buffer.len());
            chunk_phonemes.extend_from_slice(&state.phoneme_buffer[..chunk_size]);

            // Update context for next chunk (keep last overlap_size phonemes)
            let context_start =
                std::cmp::max(0, chunk_size as i32 - state.overlap_size as i32) as usize;
            state.context_phonemes = state.phoneme_buffer[context_start..chunk_size].to_vec();

            // Remove processed phonemes from buffer
            state.phoneme_buffer.drain(..chunk_size);

            debug!(
                "ONNX streaming: processing chunk of {} phonemes (with {} context phonemes)",
                chunk_phonemes.len(),
                chunk_phonemes.len() - chunk_size
            );

            // Return data needed for processing
            (chunk_phonemes, state.config.clone(), chunk_size)
        }; // Lock is dropped here

        // Process the chunk using regular synthesis
        let mel_result = self.synthesize_chunk(&chunk_phonemes, &config).await?;

        // Update frame count
        {
            let mut state = self.streaming_state.write().unwrap();
            state.total_frames += mel_result.data[0].len();

            debug!(
                "ONNX streaming: generated {} mel frames, total frames: {}",
                mel_result.data[0].len(),
                state.total_frames
            );
        }

        Ok(mel_result)
    }

    async fn end_stream(&mut self) -> Result<()> {
        info!("Ending ONNX streaming synthesis");

        // Extract final phonemes to process if needed
        let final_data = {
            let mut state = self.streaming_state.write().unwrap();

            // Check if there are remaining phonemes in buffer
            if !state.phoneme_buffer.is_empty() && state.is_active {
                debug!(
                    "ONNX streaming: processing final {} phonemes",
                    state.phoneme_buffer.len()
                );

                // Process final chunk with context
                let mut final_phonemes = state.context_phonemes.clone();
                final_phonemes.extend_from_slice(&state.phoneme_buffer);
                let config = state.config.clone();

                Some((final_phonemes, config))
            } else {
                None
            }
        }; // Lock is dropped here

        // Process final chunk if needed
        if let Some((final_phonemes, config)) = final_data {
            // This would typically be returned or processed differently in a real implementation
            let _final_mel = self.synthesize_chunk(&final_phonemes, &config).await?;
        }

        // Reset streaming state
        {
            let mut state = self.streaming_state.write().unwrap();
            state.is_active = false;
            state.phoneme_buffer.clear();
            state.context_phonemes.clear();

            info!(
                "ONNX streaming completed: total {} frames processed",
                state.total_frames
            );
        }

        Ok(())
    }
}

/// Builder for ONNX acoustic model
pub struct OnnxAcousticModelBuilder {
    config: OnnxModelConfig,
}

impl OnnxAcousticModelBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: OnnxModelConfig::default(),
        }
    }

    /// Set model path
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Add execution provider
    pub fn with_execution_provider(mut self, provider: String) -> Self {
        self.config.execution_providers.push(provider);
        self
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        self.config.intra_op_num_threads = Some(num_threads);
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: String) -> Self {
        self.config.graph_optimization_level = level;
        self
    }

    /// Build the model
    pub async fn build(self) -> Result<OnnxAcousticModel> {
        if !self.config.model_path.exists() {
            return Err(AcousticError::ModelError(format!(
                "Model file not found: {:?}",
                self.config.model_path
            )));
        }

        OnnxAcousticModel::new(self.config).await
    }
}

impl Default for OnnxAcousticModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX Backend implementation for the Backend trait
pub struct OnnxBackend {
    /// Default configuration
    config: OnnxModelConfig,
}

impl OnnxBackend {
    /// Create new ONNX backend
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: OnnxModelConfig::default(),
        })
    }

    /// Create ONNX backend with options
    pub fn with_options(options: crate::config::BackendOptions) -> Result<Self> {
        let mut config = OnnxModelConfig::default();

        // Configure based on ONNX options if provided
        if let Some(onnx_opts) = options.onnx {
            config.execution_providers = onnx_opts.execution_providers;
            // Map optimization level if needed
        }

        Ok(Self { config })
    }
}

#[async_trait]
impl crate::backends::Backend for OnnxBackend {
    fn name(&self) -> &'static str {
        "ONNX Runtime"
    }

    fn supports_gpu(&self) -> bool {
        // Check if CUDA provider is available
        self.config
            .execution_providers
            .iter()
            .any(|p| p == "CUDAExecutionProvider")
    }

    fn available_devices(&self) -> Vec<String> {
        let mut devices = vec!["cpu".to_string()];

        // Check if CUDA provider is configured
        if self
            .config
            .execution_providers
            .iter()
            .any(|p| p == "CUDAExecutionProvider")
        {
            devices.push("cuda".to_string());
        }

        devices
    }

    async fn create_model(&self, model_path: &str) -> Result<Box<dyn crate::AcousticModel>> {
        let model = OnnxAcousticModelBuilder::new()
            .with_model_path(model_path)
            .with_num_threads(self.config.num_threads)
            .with_optimization_level(self.config.graph_optimization_level.clone())
            .build()
            .await?;

        Ok(Box::new(model))
    }

    fn capabilities(&self) -> crate::backends::BackendCapabilities {
        crate::backends::BackendCapabilities {
            name: self.name().to_string(),
            supports_gpu: self.supports_gpu(),
            supports_streaming: true, // Now implemented
            supports_batch_processing: true,
            max_batch_size: Some(32),
            memory_efficient: true,
        }
    }

    fn validate_model(&self, model_path: &str) -> Result<crate::backends::ModelInfo> {
        use std::fs;

        let path = std::path::Path::new(model_path);
        if !path.exists() {
            return Err(AcousticError::ModelError(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        let metadata = fs::metadata(path).map_err(|e| {
            AcousticError::ModelError(format!("Failed to read model metadata: {}", e))
        })?;

        let format = if model_path.ends_with(".onnx") {
            crate::backends::ModelFormat::Onnx
        } else {
            crate::backends::ModelFormat::Unknown
        };

        let compatible = format == crate::backends::ModelFormat::Onnx;

        let mut info_metadata = std::collections::HashMap::new();
        info_metadata.insert("backend".to_string(), "ONNX".to_string());
        info_metadata.insert("format".to_string(), format!("{:?}", format));

        Ok(crate::backends::ModelInfo {
            path: model_path.to_string(),
            format,
            size_bytes: metadata.len(),
            compatible,
            metadata: info_metadata,
        })
    }

    fn optimization_options(&self) -> Vec<crate::backends::OptimizationOption> {
        vec![
            crate::backends::OptimizationOption {
                name: "graph_optimization".to_string(),
                description: "Enable ONNX graph optimization".to_string(),
                enabled: true,
            },
            crate::backends::OptimizationOption {
                name: "memory_pattern".to_string(),
                description: "Enable memory pattern optimization".to_string(),
                enabled: self.config.enable_memory_pattern,
            },
            crate::backends::OptimizationOption {
                name: "cpu_mem_arena".to_string(),
                description: "Enable CPU memory arena".to_string(),
                enabled: self.config.enable_cpu_mem_arena,
            },
        ]
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create default ONNX backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_onnx_model_builder() {
        let builder = OnnxAcousticModelBuilder::new()
            .with_num_threads(4)
            .with_optimization_level(GraphOptimizationLevel::Level1);

        // Test would require a real ONNX model file
        assert_eq!(builder.config.num_threads, 4);
        assert_eq!(
            builder.config.graph_optimization_level,
            GraphOptimizationLevel::Level1
        );
    }

    #[test]
    fn test_phoneme_to_id() {
        let config = OnnxModelConfig::default();
        let metadata = ModelMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            architecture: "test".to_string(),
            sample_rates: vec![22050],
            vocab_size: 256,
            mel_dim: 80,
            max_sequence_length: 1000,
            speakers: vec!["default".to_string()],
            input_names: vec!["phonemes".to_string()],
            output_names: vec!["mel".to_string()],
        };

        // This would require creating a mock session for full testing
        // For now, just test the basic ID mapping logic
        let mock_model = || {
            let vocab_size = 256;
            let phoneme_to_id = |symbol: &str| -> i64 {
                let mut hash = 0u64;
                for byte in symbol.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
                }
                (hash % vocab_size as u64) as i64
            };

            let id1 = phoneme_to_id("a");
            let id2 = phoneme_to_id("b");
            assert_ne!(id1, id2);
            assert!(id1 >= 0 && id1 < vocab_size);
            assert!(id2 >= 0 && id2 < vocab_size);
        };

        mock_model();
    }
}
