//! OxiONNX backend for acoustic modeling.
//!
//! This module provides ONNX-based implementations for acoustic models,
//! enabling high-performance inference using pre-trained models via oxionnx.

use crate::{
    speaker::SpeakerEmbedding,
    traits::{AcousticModel, AcousticModelFeature, AcousticModelMetadata},
    AcousticError, MelSpectrogram, Phoneme, Result, SynthesisConfig,
};
use async_trait::async_trait;
use oxionnx::{OnnxError, OptLevel, Session, SessionBuilder, Tensor};
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
    /// OxiONNX session
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

    /// Number of threads for CPU execution
    pub num_threads: usize,

    /// Enable memory pattern optimization
    pub enable_memory_pattern: bool,

    /// Enable CPU memory arena
    pub enable_cpu_mem_arena: bool,

    /// Optimization level for ONNX graph optimization (default: All)
    pub opt_level: Option<OptLevel>,

    /// Enable per-node profiling during inference
    pub enable_profiling: Option<bool>,

    /// Enable memory pool for buffer reuse
    pub enable_memory_pool: Option<bool>,
}

impl Default for OnnxModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            num_threads: num_cpus::get(),
            enable_memory_pattern: true,
            enable_cpu_mem_arena: true,
            opt_level: None,
            enable_profiling: None,
            enable_memory_pool: None,
        }
    }
}

impl OnnxAcousticModel {
    /// Create a new ONNX acoustic model
    pub async fn new(config: OnnxModelConfig) -> Result<Self> {
        info!(
            "Initializing OxiONNX acoustic model from {:?}",
            config.model_path
        );

        // Load the model via oxionnx SessionBuilder
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level.unwrap_or(OptLevel::All))
            .with_memory_pool(config.enable_memory_pool.unwrap_or(false));
        if config.enable_profiling.unwrap_or(false) {
            builder = builder.with_profiling();
        }
        let session = builder
            .load(&config.model_path)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to load ONNX model: {}", e),
            })?;

        // Extract model metadata
        let metadata = Self::extract_metadata(&session, &config.model_path)?;

        info!(
            "OxiONNX acoustic model loaded successfully: {}",
            metadata.name
        );
        debug!("Model metadata: {:?}", metadata);

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            metadata,
            speaker_embeddings: Arc::new(RwLock::new(HashMap::new())),
            config,
            streaming_state: Arc::new(RwLock::new(StreamingState::default())),
        })
    }

    /// Extract model metadata from oxionnx session
    fn extract_metadata(session: &Session, model_path: &Path) -> Result<ModelMetadata> {
        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        debug!("ONNX model inputs: {:?}", input_names);
        debug!("ONNX model outputs: {:?}", output_names);

        // Extract model name from file path
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // oxionnx does not have a metadata() method, use defaults
        let architecture = "ONNX".to_string();
        let version = "1.0.0".to_string();

        // Standard mel dimension
        let mel_dim = 80;

        debug!("Using mel dimension: {}", mel_dim);

        Ok(ModelMetadata {
            name: model_name,
            version,
            architecture,
            sample_rates: vec![22050, 24000, 48000],
            vocab_size: 256,
            mel_dim,
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
        let mut embeddings =
            self.speaker_embeddings
                .write()
                .map_err(|_| AcousticError::ModelError {
                    message: "Speaker embeddings RwLock poisoned".to_string(),
                })?;
        embeddings.insert(speaker_id.to_string(), embedding);
        info!("Loaded speaker embedding for: {}", speaker_id);
        Ok(())
    }

    /// Get speaker embedding
    fn get_speaker_embedding(&self, speaker_id: Option<&str>) -> Option<Vec<f32>> {
        if let Some(id) = speaker_id {
            let embeddings = self.speaker_embeddings.read().ok()?;
            embeddings.get(id).cloned()
        } else {
            None
        }
    }

    /// Prepare input tensors for oxionnx inference
    async fn prepare_inputs(
        &self,
        phonemes: &[Phoneme],
        config: &SynthesisConfig,
    ) -> Result<HashMap<String, Tensor>> {
        let mut inputs: HashMap<String, Tensor> = HashMap::new();

        // Convert phonemes to integer sequence, then to f32 (oxionnx uses f32 tensors)
        let phoneme_ids: Vec<f32> = phonemes
            .iter()
            .map(|p| self.phoneme_to_id(&p.symbol) as f32)
            .collect();

        // Create phoneme input tensor with shape [1, seq_len]
        let seq_len = phoneme_ids.len();
        let phoneme_tensor = Tensor::new(phoneme_ids, vec![1, seq_len]);
        inputs.insert("phonemes".to_string(), phoneme_tensor);

        // Add speaker embedding if available
        if let Some(speaker_id) = config.speaker_id {
            if let Some(embedding) = self.get_speaker_embedding(Some(&speaker_id.to_string())) {
                let emb_len = embedding.len();
                let speaker_tensor = Tensor::new(embedding, vec![1, emb_len]);
                inputs.insert("speaker".to_string(), speaker_tensor);
            }
        }

        // Add synthesis control parameters
        if self.metadata.input_names.contains(&"speed".to_string()) {
            let speed_tensor = Tensor::new(vec![config.speed], vec![1]);
            inputs.insert("speed".to_string(), speed_tensor);
        }

        if self
            .metadata
            .input_names
            .contains(&"pitch_shift".to_string())
        {
            let pitch_tensor = Tensor::new(vec![config.pitch_shift], vec![1]);
            inputs.insert("pitch_shift".to_string(), pitch_tensor);
        }

        if self.metadata.input_names.contains(&"energy".to_string()) {
            let energy_tensor = Tensor::new(vec![config.energy], vec![1]);
            inputs.insert("energy".to_string(), energy_tensor);
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
        let ref_inputs: HashMap<&str, Tensor> = inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Run inference on the chunk
        let session = self.session.read().map_err(|_| AcousticError::ModelError {
            message: "Session RwLock poisoned".to_string(),
        })?;
        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| AcousticError::ModelError {
                message: format!("ONNX chunk inference failed: {}", e),
            })?;

        // Process outputs
        let mel_spectrogram = self.process_outputs(&outputs)?;

        debug!(
            "ONNX chunk synthesis completed: {} mel frames",
            mel_spectrogram.n_frames
        );

        Ok(mel_spectrogram)
    }

    /// Process oxionnx outputs to extract mel spectrogram
    fn process_outputs(&self, outputs: &HashMap<String, Tensor>) -> Result<MelSpectrogram> {
        // Extract first output by name
        let mel_output = outputs
            .get("mel_spectrogram")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("0"))
            .ok_or_else(|| AcousticError::ModelError {
                message: "No outputs received from ONNX model".to_string(),
            })?;

        // Access tensor data and shape directly
        let mel_data = &mel_output.data;
        let shape = &mel_output.shape;

        // Get output shape: expect [batch_size, mel_dim, seq_len]
        let (_batch_size, mel_dim, seq_len) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            return Err(AcousticError::ModelError {
                message: format!("Unexpected mel output shape: {:?}", shape),
            });
        };

        if _batch_size != 1 {
            warn!("Batch size {} > 1, using first sample", _batch_size);
        }

        // Reshape data to 2D matrix (mel_dim, seq_len)
        let mut mel_matrix = vec![vec![0.0; seq_len]; mel_dim];
        for (i, row) in mel_matrix.iter_mut().enumerate().take(mel_dim) {
            for (j, cell) in row.iter_mut().enumerate().take(seq_len) {
                let idx = i * seq_len + j;
                if idx < mel_data.len() {
                    *cell = mel_data[idx];
                }
            }
        }

        Ok(MelSpectrogram {
            data: mel_matrix,
            sample_rate: 22050,
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
            return Err(AcousticError::InputError {
                message: "Empty phoneme sequence".to_string(),
            });
        }

        if phonemes.len() > self.metadata.max_sequence_length {
            return Err(AcousticError::InputError {
                message: format!(
                    "Sequence length {} exceeds maximum {}",
                    phonemes.len(),
                    self.metadata.max_sequence_length
                ),
            });
        }

        // Prepare input tensors
        let default_config = SynthesisConfig::default();
        let config = config.unwrap_or(&default_config);
        let inputs = self.prepare_inputs(phonemes, config).await?;
        let ref_inputs: HashMap<&str, Tensor> = inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Run inference
        let session = self.session.read().map_err(|_| AcousticError::ModelError {
            message: "Session RwLock poisoned".to_string(),
        })?;
        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| AcousticError::ModelError {
                message: format!("ONNX inference failed: {}", e),
            })?;

        // Process outputs
        let mel_spectrogram = self.process_outputs(&outputs)?;

        debug!(
            "ONNX acoustic synthesis completed: {} mel frames",
            mel_spectrogram.n_frames
        );

        Ok(mel_spectrogram)
    }

    fn metadata(&self) -> AcousticModelMetadata {
        // Infer supported languages from model architecture/name
        let supported_languages = Self::infer_supported_languages(&self.metadata);

        AcousticModelMetadata {
            name: self.metadata.name.clone(),
            version: self.metadata.version.clone(),
            architecture: self.metadata.architecture.clone(),
            supported_languages,
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
            AcousticModelFeature::GpuAcceleration => {
                // oxionnx supports GPU via the gpu feature at compile time
                cfg!(feature = "gpu")
            }
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
    /// Infer supported languages from model metadata
    fn infer_supported_languages(metadata: &ModelMetadata) -> Vec<crate::LanguageCode> {
        use crate::LanguageCode;

        let name_lower = metadata.name.to_lowercase();
        let arch_lower = metadata.architecture.to_lowercase();

        let mut languages = Vec::new();

        if name_lower.contains("en") || arch_lower.contains("english") {
            languages.push(LanguageCode::EnUs);
        }
        if name_lower.contains("ja") || name_lower.contains("jp") || arch_lower.contains("japanese")
        {
            languages.push(LanguageCode::JaJp);
        }
        if name_lower.contains("zh") || name_lower.contains("cn") || arch_lower.contains("chinese")
        {
            languages.push(LanguageCode::ZhCn);
        }
        if name_lower.contains("ko") || name_lower.contains("kr") || arch_lower.contains("korean") {
            languages.push(LanguageCode::KoKr);
        }
        if name_lower.contains("de") || arch_lower.contains("german") {
            languages.push(LanguageCode::DeDe);
        }
        if name_lower.contains("fr") || arch_lower.contains("french") {
            languages.push(LanguageCode::FrFr);
        }
        if name_lower.contains("es") || arch_lower.contains("spanish") {
            languages.push(LanguageCode::EsEs);
        }
        if name_lower.contains("it") || arch_lower.contains("italian") {
            languages.push(LanguageCode::ItIt);
        }

        if (languages.is_empty()
            || name_lower.contains("multilingual")
            || name_lower.contains("multi"))
            && !languages.contains(&LanguageCode::EnUs)
        {
            languages.push(LanguageCode::EnUs);
        }

        debug!("Inferred supported languages: {:?}", languages);
        languages
    }

    async fn set_speaker_embedding(&self, speaker_id: &str, embedding: Vec<f32>) -> Result<()> {
        self.load_speaker_embedding(speaker_id, embedding).await
    }

    async fn get_supported_speakers(&self) -> Vec<String> {
        self.metadata.speakers.clone()
    }

    async fn extract_speaker_embedding(&self, _samples: &[f32]) -> Result<Vec<f32>> {
        Err(AcousticError::ConfigError {
            message: "Speaker embedding extraction not implemented for ONNX backend".to_string(),
        })
    }
}

impl OnnxAcousticModel {
    async fn start_stream(&self, config: &SynthesisConfig) -> Result<()> {
        info!("Starting ONNX streaming synthesis");

        let mut state = self
            .streaming_state
            .write()
            .map_err(|_| AcousticError::ModelError {
                message: "Streaming state RwLock poisoned".to_string(),
            })?;

        state.config = config.clone();
        state.phoneme_buffer.clear();
        state.context_phonemes.clear();
        state.total_frames = 0;
        state.is_active = true;

        let max_chunk = self.metadata.max_sequence_length / 4;
        state.chunk_size = std::cmp::min(state.chunk_size, max_chunk);
        state.overlap_size = std::cmp::min(state.overlap_size, state.chunk_size / 4);

        debug!(
            "ONNX streaming initialized: chunk_size={}, overlap_size={}",
            state.chunk_size, state.overlap_size
        );

        Ok(())
    }

    async fn stream_phonemes(&mut self, phonemes: &[Phoneme]) -> Result<MelSpectrogram> {
        let (chunk_phonemes, config, _chunk_size) = {
            let mut state =
                self.streaming_state
                    .write()
                    .map_err(|_| AcousticError::ModelError {
                        message: "Streaming state RwLock poisoned".to_string(),
                    })?;

            if !state.is_active {
                return Err(AcousticError::ConfigError {
                    message: "Streaming not started. Call start_stream first.".to_string(),
                });
            }

            state.phoneme_buffer.extend_from_slice(phonemes);

            debug!(
                "ONNX streaming: buffered {} phonemes, total buffer size: {}",
                phonemes.len(),
                state.phoneme_buffer.len()
            );

            if state.phoneme_buffer.len() < state.chunk_size {
                return Ok(MelSpectrogram {
                    data: vec![vec![]; self.metadata.mel_dim],
                    sample_rate: 22050,
                    hop_length: 256,
                    n_mels: self.metadata.mel_dim,
                    n_frames: 0,
                });
            }

            let mut chunk_phonemes = state.context_phonemes.clone();
            let chunk_size = std::cmp::min(state.chunk_size, state.phoneme_buffer.len());
            chunk_phonemes.extend_from_slice(&state.phoneme_buffer[..chunk_size]);

            let context_start =
                std::cmp::max(0, chunk_size as i32 - state.overlap_size as i32) as usize;
            state.context_phonemes = state.phoneme_buffer[context_start..chunk_size].to_vec();

            state.phoneme_buffer.drain(..chunk_size);

            debug!(
                "ONNX streaming: processing chunk of {} phonemes (with {} context phonemes)",
                chunk_phonemes.len(),
                chunk_phonemes.len() - chunk_size
            );

            (chunk_phonemes, state.config.clone(), chunk_size)
        };

        let mel_result = self.synthesize_chunk(&chunk_phonemes, &config).await?;

        {
            let mut state =
                self.streaming_state
                    .write()
                    .map_err(|_| AcousticError::ModelError {
                        message: "Streaming state RwLock poisoned".to_string(),
                    })?;
            state.total_frames += mel_result.n_frames;

            debug!(
                "ONNX streaming: generated {} mel frames, total frames: {}",
                mel_result.n_frames, state.total_frames
            );
        }

        Ok(mel_result)
    }

    async fn end_stream(&mut self) -> Result<()> {
        info!("Ending ONNX streaming synthesis");

        let final_data = {
            let mut state =
                self.streaming_state
                    .write()
                    .map_err(|_| AcousticError::ModelError {
                        message: "Streaming state RwLock poisoned".to_string(),
                    })?;

            if !state.phoneme_buffer.is_empty() && state.is_active {
                debug!(
                    "ONNX streaming: processing final {} phonemes",
                    state.phoneme_buffer.len()
                );

                let mut final_phonemes = state.context_phonemes.clone();
                final_phonemes.extend_from_slice(&state.phoneme_buffer);
                let config = state.config.clone();

                Some((final_phonemes, config))
            } else {
                None
            }
        };

        if let Some((final_phonemes, config)) = final_data {
            let _final_mel = self.synthesize_chunk(&final_phonemes, &config).await?;
        }

        {
            let mut state =
                self.streaming_state
                    .write()
                    .map_err(|_| AcousticError::ModelError {
                        message: "Streaming state RwLock poisoned".to_string(),
                    })?;
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

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        self
    }

    /// Build the model
    pub async fn build(self) -> Result<OnnxAcousticModel> {
        if !self.config.model_path.exists() {
            return Err(AcousticError::ModelError {
                message: format!("Model file not found: {:?}", self.config.model_path),
            });
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
        let config = OnnxModelConfig::default();

        // OnnxOptions from config are noted but oxionnx does not support
        // execution_providers / graph_optimization_level / thread settings.
        // We still accept the options struct but use defaults.
        if let Some(ref _onnx_opts) = options.onnx {
            debug!("ONNX options provided; oxionnx uses defaults for execution/threading");
        }

        Ok(Self { config })
    }
}

#[async_trait]
impl crate::backends::Backend for OnnxBackend {
    fn name(&self) -> &'static str {
        "OxiONNX"
    }

    fn supports_gpu(&self) -> bool {
        cfg!(feature = "gpu")
    }

    fn available_devices(&self) -> Vec<String> {
        let mut devices = vec!["cpu".to_string()];

        if cfg!(feature = "gpu") {
            devices.push("gpu".to_string());
        }

        devices
    }

    async fn create_model(&self, model_path: &str) -> Result<Box<dyn crate::AcousticModel>> {
        let model = OnnxAcousticModelBuilder::new()
            .with_model_path(model_path)
            .with_num_threads(self.config.num_threads)
            .build()
            .await?;

        Ok(Box::new(model))
    }

    fn capabilities(&self) -> crate::backends::BackendCapabilities {
        crate::backends::BackendCapabilities {
            name: self.name().to_string(),
            supports_gpu: self.supports_gpu(),
            supports_streaming: true,
            supports_batch_processing: true,
            max_batch_size: Some(32),
            memory_efficient: true,
        }
    }

    fn validate_model(&self, model_path: &str) -> Result<crate::backends::ModelInfo> {
        use std::fs;

        let path = std::path::Path::new(model_path);
        if !path.exists() {
            return Err(AcousticError::ModelError {
                message: format!("Model file not found: {}", model_path),
            });
        }

        let metadata = fs::metadata(path).map_err(|e| AcousticError::ModelError {
            message: format!("Failed to read model metadata: {}", e),
        })?;

        let format = if model_path.ends_with(".onnx") {
            crate::backends::ModelFormat::Onnx
        } else {
            crate::backends::ModelFormat::Unknown
        };

        let compatible = format == crate::backends::ModelFormat::Onnx;

        let mut info_metadata = std::collections::HashMap::new();
        info_metadata.insert("backend".to_string(), "OxiONNX".to_string());
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
        let builder = OnnxAcousticModelBuilder::new().with_num_threads(4);

        assert_eq!(builder.config.num_threads, 4);
    }

    #[test]
    fn test_phoneme_to_id() {
        let vocab_size: usize = 256;
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
        assert!(id1 >= 0 && id1 < vocab_size as i64);
        assert!(id2 >= 0 && id2 < vocab_size as i64);
    }

    #[test]
    fn test_onnx_model_config_default() {
        let config = OnnxModelConfig::default();
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.num_threads > 0);
        assert!(config.enable_memory_pattern);
        assert!(config.enable_cpu_mem_arena);
    }

    #[test]
    fn test_streaming_state_default() {
        let state = StreamingState::default();
        assert_eq!(state.chunk_size, 50);
        assert_eq!(state.overlap_size, 10);
        assert!(!state.is_active);
        assert_eq!(state.total_frames, 0);
        assert!(state.phoneme_buffer.is_empty());
        assert!(state.context_phonemes.is_empty());
    }
}
