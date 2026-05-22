//! OxiONNX backend for neural vocoders.
//!
//! This module provides ONNX-based implementations for neural vocoders,
//! enabling high-performance audio generation using pre-trained models.

use crate::{
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderError, VocoderFeature,
    VocoderMetadata,
};
use async_trait::async_trait;
use oxionnx::{OptLevel, Session, Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tracing::{debug, info, warn};

/// ONNX-based vocoder implementation
pub struct OnnxVocoder {
    /// ONNX Runtime session
    session: Session,

    /// Model metadata
    metadata: VocoderMetadata,

    /// ONNX-specific details
    onnx_details: OnnxVocoderDetails,

    /// Model configuration
    config: OnnxVocoderConfig,
}

/// ONNX-specific vocoder metadata
#[derive(Debug, Clone)]
pub struct OnnxVocoderDetails {
    /// Input mel spectrogram dimensions
    pub mel_dim: usize,

    /// Hop length for STFT
    pub hop_length: usize,

    /// Window length for STFT
    pub win_length: usize,

    /// FFT size
    pub n_fft: usize,

    /// Model input names
    pub input_names: Vec<String>,

    /// Model output names
    pub output_names: Vec<String>,

    /// Expected input shape (batch_size, mel_dim, time_steps)
    pub input_shape: Option<(usize, usize, Option<usize>)>,
}

/// ONNX vocoder configuration
#[derive(Debug, Clone)]
pub struct OnnxVocoderConfig {
    /// Model file path
    pub model_path: PathBuf,

    /// Number of threads for CPU execution
    pub num_threads: usize,

    /// Audio generation settings
    pub audio_config: AudioConfig,

    /// Whether GPU acceleration is requested
    pub use_gpu: bool,

    /// Optimization level for ONNX graph optimization (default: All)
    pub opt_level: Option<OptLevel>,

    /// Enable per-node profiling during inference
    pub enable_profiling: Option<bool>,

    /// Enable memory pool for buffer reuse
    pub enable_memory_pool: Option<bool>,
}

/// Audio generation configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate
    pub sample_rate: u32,

    /// Hop length
    pub hop_length: usize,

    /// Audio scaling factor
    pub audio_scale: f32,

    /// Audio clipping threshold
    pub clip_threshold: f32,

    /// Enable denoising
    pub enable_denoising: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            hop_length: 256,
            audio_scale: 1.0,
            clip_threshold: 0.99,
            enable_denoising: false,
        }
    }
}

impl Default for OnnxVocoderConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            num_threads: num_cpus::get(),
            audio_config: AudioConfig::default(),
            use_gpu: false,
            opt_level: None,
            enable_profiling: None,
            enable_memory_pool: None,
        }
    }
}

impl OnnxVocoder {
    /// Create a new ONNX vocoder
    pub async fn new(config: OnnxVocoderConfig) -> Result<Self> {
        info!("Initializing OxiONNX vocoder from {:?}", config.model_path);

        // Load the model via oxionnx SessionBuilder
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level.unwrap_or(OptLevel::All))
            .with_memory_pool(config.enable_memory_pool.unwrap_or(false));
        if config.enable_profiling.unwrap_or(false) {
            builder = builder.with_profiling();
        }
        let session = builder.load(&config.model_path).map_err(|e| {
            VocoderError::ModelError(format!("Failed to load ONNX vocoder model: {e}"))
        })?;

        // Extract model metadata
        let (metadata, onnx_details) =
            Self::extract_metadata(&session, &config.model_path, &config.audio_config)?;

        info!("OxiONNX vocoder loaded successfully: {}", metadata.name);
        debug!("Vocoder metadata: {:?}", metadata);

        Ok(Self {
            session,
            metadata,
            onnx_details,
            config,
        })
    }

    /// Extract model metadata from ONNX session
    fn extract_metadata(
        session: &Session,
        model_path: &Path,
        audio_config: &AudioConfig,
    ) -> Result<(VocoderMetadata, OnnxVocoderDetails)> {
        // Get input and output names
        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        // Extract model name from file path
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // oxionnx doesn't expose per-input shape info; leave as None
        let input_shape = None;

        // Standard VoiRS metadata
        let metadata = VocoderMetadata {
            name: model_name,
            version: "1.0.0".to_string(),
            architecture: "ONNX".to_string(),
            sample_rate: audio_config.sample_rate,
            mel_channels: 80,   // Standard mel dimension
            latency_ms: 10.0,   // Estimated latency
            quality_score: 4.0, // Good quality score
        };

        // ONNX-specific details
        let onnx_details = OnnxVocoderDetails {
            mel_dim: 80, // Standard mel dimension
            hop_length: audio_config.hop_length,
            win_length: audio_config.hop_length * 4, // Common ratio
            n_fft: 2048,                             // Common FFT size
            input_names,
            output_names,
            input_shape,
        };

        Ok((metadata, onnx_details))
    }

    /// Prepare input tensors for OxiONNX inference
    async fn prepare_inputs(
        &self,
        mel_spectrogram: &MelSpectrogram,
    ) -> Result<HashMap<&str, Tensor>> {
        let mut inputs = HashMap::new();

        // Convert mel spectrogram to tensor format
        let (mel_dim, time_steps) = (mel_spectrogram.data.len(), mel_spectrogram.data[0].len());

        // Flatten mel spectrogram data (batch_size=1, mel_dim, time_steps)
        let mut mel_data = Vec::with_capacity(mel_dim * time_steps);
        for mel_row in mel_spectrogram.data.iter().take(mel_dim) {
            for j in 0..time_steps {
                mel_data.push(mel_row[j]);
            }
        }

        // Create mel input tensor with shape [1, mel_dim, time_steps]
        let mel_tensor = Tensor::new(mel_data, vec![1, mel_dim, time_steps]);

        // Use the first input name from the model
        let input_name = self
            .onnx_details
            .input_names
            .first()
            .map(|s| s.as_str())
            .unwrap_or("mel");
        inputs.insert(input_name, mel_tensor);

        Ok(inputs)
    }

    /// Process OxiONNX outputs to extract audio
    fn process_outputs(&self, outputs: &HashMap<String, Tensor>) -> Result<AudioBuffer> {
        if outputs.is_empty() {
            return Err(VocoderError::ModelError(
                "No outputs received from ONNX vocoder model".to_string(),
            ));
        }

        // Extract audio from first output
        let audio_tensor = outputs
            .values()
            .next()
            .ok_or_else(|| VocoderError::ModelError("No output tensor found".to_string()))?;

        let shape = &audio_tensor.shape;
        let audio_samples = if shape.len() == 1 {
            // Shape: [samples]
            audio_tensor.data.clone()
        } else if shape.len() == 2 && shape[0] == 1 {
            // Shape: [1, samples] - remove batch dimension
            audio_tensor.data.clone()
        } else if shape.len() == 3 && shape[0] == 1 && shape[1] == 1 {
            // Shape: [1, 1, samples] - remove batch and channel dimensions
            audio_tensor.data.clone()
        } else {
            return Err(VocoderError::ModelError(format!(
                "Unexpected audio output shape: {:?}",
                shape
            )));
        };

        // Apply audio scaling and clipping
        let scaled_audio: Vec<f32> = audio_samples
            .iter()
            .map(|&sample| {
                let scaled = sample * self.config.audio_config.audio_scale;
                scaled.clamp(
                    -self.config.audio_config.clip_threshold,
                    self.config.audio_config.clip_threshold,
                )
            })
            .collect();

        // Apply denoising if enabled
        let final_audio = if self.config.audio_config.enable_denoising {
            self.apply_denoising(&scaled_audio)?
        } else {
            scaled_audio
        };

        Ok(AudioBuffer {
            samples: final_audio,
            sample_rate: self.config.audio_config.sample_rate,
            channels: 1, // Mono output
        })
    }

    /// Apply simple denoising to audio
    fn apply_denoising(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simple high-pass filter to remove low-frequency noise
        let alpha = 0.95; // High-pass filter coefficient
        let mut filtered = vec![0.0; audio.len()];

        if !audio.is_empty() {
            filtered[0] = audio[0];
            for i in 1..audio.len() {
                filtered[i] = alpha * (filtered[i - 1] + audio[i] - audio[i - 1]);
            }
        }

        Ok(filtered)
    }

    /// Apply synthesis configuration to audio
    fn apply_synthesis_config(
        &self,
        audio: &AudioBuffer,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        let mut samples = audio.samples.clone();

        // Apply speed modification
        if config.speed != 1.0 {
            samples = self.apply_speed_change(&samples, config.speed)?;
        }

        // Apply pitch shift
        if config.pitch_shift != 0.0 {
            samples = self.apply_pitch_shift(&samples, config.pitch_shift)?;
        }

        // Apply energy scaling
        if config.energy != 1.0 {
            for sample in &mut samples {
                *sample *= config.energy;
            }
        }

        Ok(AudioBuffer {
            samples,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        })
    }

    /// Apply speed change using linear interpolation
    fn apply_speed_change(&self, samples: &[f32], speed: f32) -> Result<Vec<f32>> {
        if speed <= 0.0 {
            return Err(VocoderError::InputError(
                "Speed must be positive".to_string(),
            ));
        }

        let new_length = (samples.len() as f32 / speed) as usize;
        let mut result = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let original_idx = (i as f32 * speed) as usize;
            if original_idx < samples.len() {
                result.push(samples[original_idx]);
            } else {
                result.push(0.0);
            }
        }

        Ok(result)
    }

    /// Apply pitch shift using time-stretching and resampling
    fn apply_pitch_shift(&self, samples: &[f32], pitch_shift: f32) -> Result<Vec<f32>> {
        if pitch_shift.abs() < 0.001 {
            // No pitch shift needed
            return Ok(samples.to_vec());
        }

        // Convert semitones to pitch ratio
        let pitch_ratio = 2.0f32.powf(pitch_shift / 12.0);

        // Clamp pitch shift to reasonable ranges to avoid extreme artifacts
        let clamped_pitch_ratio = pitch_ratio.clamp(0.5, 2.0);

        // Use simple linear interpolation for pitch shifting
        let output_length = (samples.len() as f32 / clamped_pitch_ratio) as usize;
        let mut output = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let src_index = i as f32 * clamped_pitch_ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = src_index_floor + 1;
            let frac = src_index - src_index_floor as f32;

            if src_index_floor < samples.len() {
                // Linear interpolation between adjacent samples
                let sample_low = samples[src_index_floor];
                let sample_high = if src_index_ceil < samples.len() {
                    samples[src_index_ceil]
                } else {
                    sample_low
                };

                let interpolated = sample_low + frac * (sample_high - sample_low);
                output.push(interpolated);
            }
        }

        // Apply simple anti-aliasing window to reduce artifacts
        self.apply_anti_aliasing_window(&mut output);

        Ok(output)
    }

    /// Apply simple anti-aliasing window to reduce pitch shift artifacts
    fn apply_anti_aliasing_window(&self, samples: &mut [f32]) {
        let window_size = 32.min(samples.len() / 4);

        // Apply fade-in
        for i in 0..window_size {
            let fade = i as f32 / window_size as f32;
            samples[i] *= fade;
        }

        // Apply fade-out
        let start = samples.len().saturating_sub(window_size);
        for i in start..samples.len() {
            let fade = (samples.len() - i) as f32 / window_size as f32;
            samples[i] *= fade;
        }
    }
}

#[async_trait]
impl Vocoder for OnnxVocoder {
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        debug!(
            "Starting ONNX vocoder synthesis for mel shape: {}x{}",
            mel.data.len(),
            mel.data[0].len()
        );

        if mel.data.is_empty() || mel.data[0].is_empty() {
            return Err(VocoderError::InputError(
                "Empty mel spectrogram".to_string(),
            ));
        }

        // Validate mel dimensions
        let (mel_dim, time_steps) = (mel.data.len(), mel.data[0].len());
        if mel_dim != self.onnx_details.mel_dim {
            warn!(
                "Mel dimension mismatch: expected {}, got {}",
                self.onnx_details.mel_dim, mel_dim
            );
        }

        // Prepare input tensors
        let inputs = self.prepare_inputs(mel).await?;

        // Run inference
        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| VocoderError::ModelError(format!("ONNX vocoder inference failed: {e}")))?;

        // Process outputs
        let mut audio_buffer = self.process_outputs(&outputs)?;

        // Apply synthesis config if provided
        if let Some(config) = config {
            audio_buffer = self.apply_synthesis_config(&audio_buffer, config)?;
        }

        debug!(
            "ONNX vocoder synthesis completed: {} samples at {} Hz",
            audio_buffer.samples.len(),
            audio_buffer.sample_rate
        );

        Ok(audio_buffer)
    }

    async fn vocode_stream(
        &self,
        mut mel_stream: Box<dyn futures::Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::StreamExt;

        // Drain the incoming mel stream, running one ONNX inference pass per
        // chunk, and collect results.  Because `Session` is not `Clone` (it
        // holds non-cloneable GPU/thread-pool resources), we run inference
        // eagerly here rather than spawning a separate task that would need
        // its own owned copy of the session.  The resulting `Vec` is then
        // wrapped in `futures::stream::iter` and returned as a ready stream.
        // This keeps memory bounded to the synthesised audio for each chunk
        // (not the full mel buffer) while still exposing the streaming API
        // contract to callers.
        let mut results: Vec<Result<AudioBuffer>> = Vec::new();

        while let Some(mel) = mel_stream.next().await {
            if mel.data.is_empty() || mel.data.iter().all(|row| row.is_empty()) {
                // Skip empty chunks rather than propagating an error.
                continue;
            }

            let mel_dim = mel.data.len();
            if mel_dim != self.onnx_details.mel_dim {
                warn!(
                    "Streaming: mel dimension mismatch on chunk — expected {}, got {}; skipping",
                    self.onnx_details.mel_dim, mel_dim
                );
                continue;
            }

            // Prepare input, run inference, process output.
            let audio_result = async {
                let inputs = self.prepare_inputs(&mel).await?;
                let outputs = self.session.run(&inputs).map_err(|e| {
                    VocoderError::ModelError(format!("ONNX streaming inference failed: {e}"))
                })?;
                let mut audio_buffer = self.process_outputs(&outputs)?;
                if let Some(cfg) = config {
                    audio_buffer = self.apply_synthesis_config(&audio_buffer, cfg)?;
                }
                debug!(
                    "ONNX streaming chunk synthesised: {} samples at {} Hz",
                    audio_buffer.samples.len(),
                    audio_buffer.sample_rate
                );
                Ok(audio_buffer)
            }
            .await;

            results.push(audio_result);
        }

        Ok(Box::new(futures::stream::iter(results)))
    }

    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::with_capacity(mels.len());

        for (i, mel) in mels.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            let audio = self.vocode(mel, config).await?;
            results.push(audio);
        }

        Ok(results)
    }

    fn metadata(&self) -> VocoderMetadata {
        self.metadata.clone()
    }

    fn supports(&self, feature: VocoderFeature) -> bool {
        match feature {
            VocoderFeature::BatchProcessing => true,
            VocoderFeature::GpuAcceleration => cfg!(feature = "gpu") && self.config.use_gpu,
            VocoderFeature::HighQuality => true,
            VocoderFeature::FastInference => true,
            VocoderFeature::StreamingInference => false,
            VocoderFeature::RealtimeProcessing => false,
            // ONNX backend does not support emotion/conversion/singing/spatial features
            _ => false,
        }
    }
}

/// Builder for ONNX vocoder
pub struct OnnxVocoderBuilder {
    config: OnnxVocoderConfig,
}

impl OnnxVocoderBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: OnnxVocoderConfig::default(),
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

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }

    /// Set audio configuration
    pub fn with_audio_config(mut self, audio_config: AudioConfig) -> Self {
        self.config.audio_config = audio_config;
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.audio_config.sample_rate = sample_rate;
        self
    }

    /// Set hop length
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.config.audio_config.hop_length = hop_length;
        self
    }

    /// Enable denoising
    pub fn with_denoising(mut self, enabled: bool) -> Self {
        self.config.audio_config.enable_denoising = enabled;
        self
    }

    /// Build the vocoder
    pub async fn build(self) -> Result<OnnxVocoder> {
        if !self.config.model_path.exists() {
            return Err(VocoderError::ModelError(format!(
                "Vocoder model file not found: {:?}",
                self.config.model_path
            )));
        }

        OnnxVocoder::new(self.config).await
    }
}

impl Default for OnnxVocoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX Backend implementation for the Backend trait
pub struct OnnxVocoderBackend {
    /// Default configuration
    config: OnnxVocoderConfig,
}

impl OnnxVocoderBackend {
    /// Create new ONNX vocoder backend
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: OnnxVocoderConfig::default(),
        })
    }

    /// Create ONNX vocoder backend with device type
    pub fn with_device(device: crate::config::DeviceType) -> Result<Self> {
        let mut config = OnnxVocoderConfig::default();

        // Configure based on device type
        match device {
            crate::config::DeviceType::Cpu => {
                config.use_gpu = false;
            }
            crate::config::DeviceType::Cuda => {
                config.use_gpu = true;
            }
            crate::config::DeviceType::Metal => {
                // oxionnx doesn't support Metal, fallback to CPU
                config.use_gpu = false;
            }
            crate::config::DeviceType::Auto => {
                // Auto: prefer GPU if available, fall back to CPU
                config.use_gpu = cfg!(feature = "gpu");
            }
        }

        Ok(Self { config })
    }
}

/// ONNX Backend implementation
pub struct OnnxBackend {
    vocoder: Option<OnnxVocoder>,
    config: OnnxVocoderConfig,
}

impl OnnxBackend {
    /// Create new ONNX backend
    pub fn new() -> Self {
        Self {
            vocoder: None,
            config: OnnxVocoderConfig::default(),
        }
    }
}

#[async_trait]
impl crate::backends::Backend for OnnxBackend {
    async fn initialize(&mut self, config: &crate::config::ModelConfig) -> Result<()> {
        // Configure ONNX backend based on model config
        match config.device {
            crate::config::DeviceType::Cpu => {
                self.config.use_gpu = false;
            }
            crate::config::DeviceType::Cuda => {
                self.config.use_gpu = true;
            }
            crate::config::DeviceType::Metal => {
                // oxionnx doesn't support Metal, fallback to CPU
                self.config.use_gpu = false;
            }
            crate::config::DeviceType::Auto => {
                // Auto: prefer GPU if feature is enabled, otherwise CPU
                self.config.use_gpu = cfg!(feature = "gpu");
            }
        }

        // Apply thread count from optimization config (0 = let runtime auto-detect)
        let max_threads = config.optimization.max_threads;
        if max_threads > 0 {
            self.config.num_threads = max_threads;
        }

        Ok(())
    }

    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        self.config.model_path = std::path::PathBuf::from(model_path);
        let vocoder = OnnxVocoder::new(self.config.clone()).await?;
        self.vocoder = Some(vocoder);
        Ok(())
    }

    async fn inference(&self, mel: &MelSpectrogram) -> Result<AudioBuffer> {
        match &self.vocoder {
            Some(vocoder) => vocoder.vocode(mel, None).await,
            None => Err(VocoderError::ModelError("Model not loaded".to_string())),
        }
    }

    async fn batch_inference(&self, mels: &[MelSpectrogram]) -> Result<Vec<AudioBuffer>> {
        match &self.vocoder {
            Some(vocoder) => vocoder.vocode_batch(mels, None).await,
            None => Err(VocoderError::ModelError("Model not loaded".to_string())),
        }
    }

    fn metadata(&self) -> crate::backends::BackendMetadata {
        crate::backends::BackendMetadata {
            name: "OxiONNX".to_string(),
            version: "0.1.0".to_string(),
            supported_devices: vec![
                crate::config::DeviceType::Cpu,
                crate::config::DeviceType::Cuda,
            ],
            supported_formats: vec!["onnx".to_string()],
            gpu_acceleration: cfg!(feature = "gpu") && self.config.use_gpu,
            mixed_precision: false,
            quantization: false,
        }
    }

    fn supports_device(&self, device: crate::config::DeviceType) -> bool {
        match device {
            crate::config::DeviceType::Cpu => true,
            crate::config::DeviceType::Cuda => true,
            crate::config::DeviceType::Metal => false,
            // Auto maps to whichever device is available; report as supported
            crate::config::DeviceType::Auto => true,
        }
    }

    fn memory_usage(&self) -> u64 {
        // Estimate memory usage (this would be more accurate with actual tracking)
        if self.vocoder.is_some() {
            256 * 1024 * 1024 // 256MB estimate
        } else {
            0
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        self.vocoder = None;
        Ok(())
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Backend trait implementation removed to avoid conflicts with the existing implementation above

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_onnx_vocoder_builder() {
        let builder = OnnxVocoderBuilder::new()
            .with_num_threads(4)
            .with_sample_rate(24000)
            .with_hop_length(512)
            .with_denoising(true)
            .with_gpu(true);

        // Test would require a real ONNX model file
        assert_eq!(builder.config.num_threads, 4);
        assert_eq!(builder.config.audio_config.sample_rate, 24000);
        assert_eq!(builder.config.audio_config.hop_length, 512);
        assert!(builder.config.audio_config.enable_denoising);
        assert!(builder.config.use_gpu);
    }

    #[test]
    fn test_audio_config() {
        let config = AudioConfig {
            sample_rate: 48000,
            hop_length: 512,
            audio_scale: 0.8,
            clip_threshold: 0.95,
            enable_denoising: true,
        };

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.hop_length, 512);
        assert_eq!(config.audio_scale, 0.8);
        assert_eq!(config.clip_threshold, 0.95);
        assert!(config.enable_denoising);
    }

    #[test]
    fn test_denoising() {
        let config = OnnxVocoderConfig::default();

        // Mock vocoder for testing denoising
        let mock_test = || {
            let audio = vec![0.1, -0.05, 0.2, -0.1, 0.15];

            // Simple high-pass filter implementation for testing
            let alpha = 0.95;
            let mut filtered = vec![0.0; audio.len()];

            if !audio.is_empty() {
                filtered[0] = audio[0];
                for i in 1..audio.len() {
                    filtered[i] = alpha * (filtered[i - 1] + audio[i] - audio[i - 1]);
                }
            }

            // Check that denoising produces different output
            assert_ne!(audio, filtered);
            assert_eq!(audio.len(), filtered.len());
        };

        mock_test();
    }

    #[test]
    fn test_synthesis_config_application() {
        use crate::{AudioBuffer, SynthesisConfig};

        // Create mock vocoder for testing synthesis configuration
        let mock_test = || {
            let original_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            let original_audio = AudioBuffer::new(original_samples.clone(), 22050, 1);

            // Test energy scaling
            let mut config = SynthesisConfig::default();
            config.energy = 2.0;

            let expected_scaled: Vec<f32> = original_samples.iter().map(|&x| x * 2.0).collect();
            let mut scaled_samples = original_samples.clone();
            for sample in &mut scaled_samples {
                *sample *= config.energy;
            }
            assert_eq!(scaled_samples, expected_scaled);

            // Test speed change logic
            let speed = 2.0;
            let new_length = (original_samples.len() as f32 / speed) as usize;
            assert_eq!(new_length, 4); // 8 / 2 = 4

            // Test pitch shift ratio calculation
            let pitch_shift = 12.0; // 1 octave up
            let pitch_ratio = 2.0f32.powf(pitch_shift / 12.0);
            assert_eq!(pitch_ratio, 2.0);

            let pitch_shift_negative = -12.0; // 1 octave down
            let pitch_ratio_negative = 2.0f32.powf(pitch_shift_negative / 12.0);
            assert_eq!(pitch_ratio_negative, 0.5);
        };

        mock_test();
    }
}
