//! VITS (Variational Inference Text-to-Speech) model implementation
//!
//! This module contains the complete VITS architecture including:
//! - Text encoder (transformer-based)
//! - Posterior encoder (CNN-based)
//! - Normalizing flows
//! - Decoder/generator
//! - Duration predictor

use async_trait::async_trait;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    AcousticModel, AcousticModelFeature, AcousticModelMetadata, 
    LanguageCode, MelSpectrogram, Phoneme, Result, SynthesisConfig, AcousticError,
};

#[cfg(feature = "candle")]
use crate::backends::candle::CandleTensorOps;

pub mod text_encoder;
pub mod posterior;
pub mod flows;
pub mod decoder;
pub mod duration;

// Re-export main components
pub use text_encoder::{TextEncoder, TextEncoderConfig, PhonemeEmbedding};

// Re-export implemented components
pub use posterior::{PosteriorEncoder, PosteriorConfig};
pub use duration::{DurationPredictor, DurationConfig};
pub use flows::{NormalizingFlows, FlowConfig};
pub use decoder::{Decoder, DecoderConfig};

/// VITS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitsConfig {
    /// Text encoder configuration
    pub text_encoder: TextEncoderConfig,
    /// Posterior encoder configuration
    pub posterior_encoder: PosteriorConfig,
    /// Duration predictor configuration
    pub duration_predictor: DurationConfig,
    /// Normalizing flows configuration
    pub flows: FlowConfig,
    /// Decoder configuration
    pub decoder: DecoderConfig,
    
    /// Sample rate for audio generation
    pub sample_rate: u32,
    /// Number of mel spectrogram channels
    pub mel_channels: usize,
    /// Whether the model supports multiple speakers
    pub multi_speaker: bool,
    /// Number of speakers (if multi-speaker)
    pub speaker_count: Option<usize>,
}

impl Default for VitsConfig {
    fn default() -> Self {
        Self {
            text_encoder: TextEncoderConfig::default(),
            posterior_encoder: PosteriorConfig::default(),
            duration_predictor: DurationConfig::default(),
            flows: FlowConfig::default(),
            decoder: DecoderConfig::default(),
            sample_rate: 22050,
            mel_channels: 80,
            multi_speaker: false,
            speaker_count: None,
        }
    }
}

/// VITS model implementation
pub struct VitsModel {
    config: VitsConfig,
    text_encoder: Arc<TextEncoder>,
    posterior_encoder: Arc<PosteriorEncoder>,
    duration_predictor: Arc<DurationPredictor>,
    flows: Arc<std::sync::Mutex<NormalizingFlows>>,
    decoder: Arc<Decoder>,
    device: Device,
}

impl VitsModel {
    /// Create new VITS model with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(VitsConfig::default())
    }
    
    /// Create VITS model with custom configuration
    pub fn with_config(config: VitsConfig) -> Result<Self> {
        let device = Device::Cpu; // TODO: Support GPU device selection
        
        let text_encoder = Arc::new(
            TextEncoder::new(config.text_encoder.clone(), device.clone())?
        );
        
        let posterior_encoder = Arc::new(
            PosteriorEncoder::new(config.posterior_encoder.clone(), device.clone())?
        );
        
        let duration_predictor = Arc::new(
            DurationPredictor::new(config.duration_predictor.clone(), device.clone())?
        );
        
        let flows = Arc::new(std::sync::Mutex::new(
            NormalizingFlows::new(config.flows.clone(), device.clone())?
        ));
        
        let decoder = Arc::new(
            Decoder::new(config.decoder.clone(), device.clone())?
        );
        
        Ok(Self {
            config,
            text_encoder,
            posterior_encoder,
            duration_predictor,
            flows,
            decoder,
            device,
        })
    }
    
    /// Get model configuration
    pub fn config(&self) -> &VitsConfig {
        &self.config
    }
    
    /// Get text encoder
    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }
    
    /// Get posterior encoder
    pub fn posterior_encoder(&self) -> &PosteriorEncoder {
        &self.posterior_encoder
    }
    
    /// Get duration predictor
    pub fn duration_predictor(&self) -> &DurationPredictor {
        &self.duration_predictor
    }
    
    /// Get normalizing flows
    pub fn flows(&self) -> &Arc<std::sync::Mutex<NormalizingFlows>> {
        &self.flows
    }
    
    /// Get decoder
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }
    
    /// Set device for inference
    pub fn with_device(mut self, device: Device) -> Result<Self> {
        self.device = device.clone();
        
        // Move all components to new device
        self.text_encoder = Arc::new(
            TextEncoder::new(self.config.text_encoder.clone(), device.clone())?
        );
        
        self.posterior_encoder = Arc::new(
            PosteriorEncoder::new(self.config.posterior_encoder.clone(), device.clone())?
        );
        
        self.duration_predictor = Arc::new(
            DurationPredictor::new(self.config.duration_predictor.clone(), device.clone())?
        );
        
        self.flows = Arc::new(std::sync::Mutex::new(
            NormalizingFlows::new(self.config.flows.clone(), device.clone())?
        ));
        
        self.decoder = Arc::new(
            Decoder::new(self.config.decoder.clone(), device)?
        );
        
        Ok(self)
    }
    
    /// Generate prior latent representation from text encoding and durations
    fn generate_prior(&self, text_encoding: &candle_core::Tensor, durations: &[f32]) -> Result<candle_core::Tensor> {
        use candle_core::{Tensor, DType};
        
        let (batch_size, text_dim, seq_len) = text_encoding.dims3()
            .map_err(|e| AcousticError::ModelError(format!("Invalid text encoding shape: {}", e)))?;
        
        if durations.len() != seq_len {
            return Err(AcousticError::InputError(
                format!("Duration length {} doesn't match sequence length {}", durations.len(), seq_len)
            ));
        }
        
        // Calculate total frames from durations
        let total_frames = durations.iter().map(|&d| d as usize).sum::<usize>().max(1);
        
        // Use mel_channels as the latent dimension (matching flows configuration)
        let latent_dim = self.config.mel_channels;
        
        tracing::debug!(
            "Generating prior: text_encoding [{}, {}, {}] -> prior [{}, {}, {}]",
            batch_size, text_dim, seq_len, batch_size, latent_dim, total_frames
        );
        tracing::debug!("Durations: {:?}", durations);
        tracing::debug!("Total frames: {}", total_frames);
        
        // For now, create a simple prior tensor with correct shape
        // TODO: Implement proper prior generation from text encoding
        let prior = Tensor::randn(0.0f32, 1.0f32, (batch_size, latent_dim, total_frames), &self.device)
            .map_err(|e| AcousticError::ModelError(format!("Failed to create prior tensor: {}", e)))?;
        
        tracing::debug!("Generated prior shape: {:?}", prior.dims());
        Ok(prior)
    }
}

impl Default for VitsModel {
    fn default() -> Self {
        Self::new().expect("Failed to create default VITS model")
    }
}

#[async_trait]
impl AcousticModel for VitsModel {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        if phonemes.is_empty() {
            return Err(AcousticError::InputError("Empty phoneme sequence".to_string()));
        }
        
        tracing::info!("VITS: Starting synthesis for {} phonemes", phonemes.len());
        
        // Step 1: Text encoding
        let text_encoding_raw = self.text_encoder.forward(phonemes, None)?;
        tracing::debug!("VITS: Text encoding raw shape: {:?}", text_encoding_raw.shape());
        
        // Transpose to match expected format [batch, features, sequence]
        let text_encoding = text_encoding_raw.transpose(1, 2)
            .map_err(|e| AcousticError::ModelError(format!("Failed to transpose text encoding: {}", e)))?;
        tracing::debug!("VITS: Text encoding transposed shape: {:?}", text_encoding.shape());
        
        // Step 2: Duration prediction using neural predictor
        let durations = self.duration_predictor.predict_phoneme_durations(phonemes)?;
        tracing::debug!("VITS: Predicted durations: {:?}", durations);
        
        // Step 3: Generate latent representation (prior) from text encoding
        let z_prior = self.generate_prior(&text_encoding, &durations)?;
        tracing::debug!("VITS: Generated prior shape: {:?}", z_prior.dims());
        
        // Step 4: Apply normalizing flows
        let (z_flow, log_det) = {
            let mut flows = self.flows.lock().unwrap();
            flows.forward(&z_prior)?
        };
        tracing::debug!("VITS: Flow output shape: {:?}, log_det: {:?}", z_flow.dims(), log_det.dims());
        
        // Step 5: Decode to mel spectrogram using neural decoder
        tracing::info!("VITS: Using neural decoder to generate mel spectrogram");
        
        let mel_tensor = self.decoder.forward(&z_flow)
            .map_err(|e| AcousticError::ModelError(format!("Decoder forward failed: {}", e)))?;
        
        tracing::debug!("VITS: Decoder output shape: {:?}", mel_tensor.dims());
        
        // Convert tensor to MelSpectrogram format
        let hop_length = 256;
        let mel = tensor_to_mel_spectrogram(&mel_tensor, self.config.sample_rate, hop_length)?;
        
        tracing::info!("VITS: Synthesis complete, generated mel spectrogram: {}x{}", 
                      mel.n_mels, mel.n_frames);
        
        Ok(mel)
    }
    
    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            let mel = self.synthesize(phonemes, config).await?;
            results.push(mel);
        }
        
        Ok(results)
    }
    
    fn metadata(&self) -> AcousticModelMetadata {
        AcousticModelMetadata {
            name: "VITS".to_string(),
            version: "1.0.0".to_string(),
            architecture: "VITS".to_string(),
            supported_languages: vec![LanguageCode::EnUs, LanguageCode::EnGb, LanguageCode::Ja],
            sample_rate: self.config.sample_rate,
            mel_channels: self.config.mel_channels as u32,
            is_multi_speaker: self.config.multi_speaker,
            speaker_count: self.config.speaker_count.map(|c| c as u32),
        }
    }
    
    fn supports(&self, feature: AcousticModelFeature) -> bool {
        match feature {
            AcousticModelFeature::MultiSpeaker => self.config.multi_speaker,
            AcousticModelFeature::BatchProcessing => true,
            AcousticModelFeature::GpuAcceleration => true,
            AcousticModelFeature::StreamingInference => false, // TODO: Implement
            AcousticModelFeature::StreamingSynthesis => false, // TODO: Implement
            AcousticModelFeature::ProsodyControl => false, // TODO: Implement
            AcousticModelFeature::StyleTransfer => false, // TODO: Implement
            AcousticModelFeature::VoiceCloning => false, // TODO: Implement
            AcousticModelFeature::RealTimeInference => false, // TODO: Optimize
        }
    }
}

/// Get phoneme-specific acoustic characteristics
/// Returns (fundamental_frequency, formant_frequencies, energy_level)
fn get_phoneme_characteristics(phoneme: &str) -> (f32, Vec<f32>, f32) {
    match phoneme {
        // Vowels - have clear formant structure
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW" | "OY" | "UH" | "UW" => {
            let (f1, f2, f3) = match phoneme {
                "AA" => (730.0, 1090.0, 2440.0), // father
                "AE" => (660.0, 1720.0, 2410.0), // cat
                "AH" => (520.0, 1190.0, 2390.0), // but
                "AO" => (570.0, 840.0, 2410.0),  // thought
                "EH" => (530.0, 1840.0, 2480.0), // bed
                "ER" => (490.0, 1350.0, 1690.0), // bird
                "EY" => (400.0, 2000.0, 2550.0), // bait
                "IH" => (390.0, 1990.0, 2550.0), // bit
                "IY" => (270.0, 2290.0, 3010.0), // beat
                "OW" => (490.0, 910.0, 2200.0),  // boat
                "UH" => (440.0, 1020.0, 2240.0), // book
                "UW" => (300.0, 870.0, 2240.0),  // boot
                _ => (500.0, 1500.0, 2500.0),    // default vowel
            };
            (150.0, vec![f1, f2, f3], 0.8) // High energy for vowels
        },
        
        // Fricatives - high frequency energy
        "F" | "TH" | "S" | "SH" | "HH" | "V" | "DH" | "Z" | "ZH" => {
            let center_freq = match phoneme {
                "S" => 7000.0,
                "SH" => 4000.0,
                "F" | "TH" => 6000.0,
                "HH" => 3000.0,
                _ => 5000.0,
            };
            (0.0, vec![center_freq], 0.6) // No fundamental, moderate energy
        },
        
        // Stops - brief bursts
        "P" | "B" | "T" | "D" | "K" | "G" => {
            let burst_freq = match phoneme {
                "P" | "B" => 1500.0,
                "T" | "D" => 4000.0,
                "K" | "G" => 2500.0,
                _ => 2000.0,
            };
            (0.0, vec![burst_freq], 0.5) // No fundamental, brief energy
        },
        
        // Nasals - low frequency resonance
        "M" | "N" | "NG" => {
            let nasal_freq = match phoneme {
                "M" => 1000.0,
                "N" => 1500.0,
                "NG" => 1200.0,
                _ => 1200.0,
            };
            (120.0, vec![nasal_freq, 2500.0], 0.7) // Low fundamental, good energy
        },
        
        // Liquids - formant-like structure
        "L" | "R" => {
            let formants = match phoneme {
                "L" => vec![350.0, 1200.0, 2900.0],
                "R" => vec![350.0, 1200.0, 1700.0],
                _ => vec![400.0, 1200.0, 2800.0],
            };
            (140.0, formants, 0.75)
        },
        
        // Glides - vowel-like but transitional
        "W" | "Y" => {
            let formants = match phoneme {
                "W" => vec![300.0, 600.0, 2200.0],
                "Y" => vec![300.0, 2200.0, 3000.0],
                _ => vec![300.0, 1400.0, 2600.0],
            };
            (140.0, formants, 0.6)
        },
        
        // Affricates
        "CH" | "JH" => {
            (0.0, vec![2500.0, 4000.0], 0.5)
        },
        
        // Silence and special tokens
        " " | "<pad>" | "<unk>" | "<bos>" | "<eos>" => {
            (0.0, vec![], 0.0)
        },
        
        // Default for unknown phonemes
        _ => (130.0, vec![500.0, 1500.0, 2500.0], 0.5),
    }
}

/// Convert decoder output tensor to MelSpectrogram
fn tensor_to_mel_spectrogram(
    tensor: &candle_core::Tensor,
    sample_rate: u32,
    hop_length: u32,
) -> Result<MelSpectrogram> {
    use candle_core::Tensor;
    
    // Get tensor dimensions [batch_size, n_mel_channels, n_frames]
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(AcousticError::ModelError(
            format!("Expected 3D tensor [batch, mel_channels, frames], got {:?}", dims)
        ));
    }
    
    let (_batch_size, n_mel_channels, n_frames) = tensor.dims3()
        .map_err(|e| AcousticError::ModelError(format!("Failed to get tensor dimensions: {}", e)))?;
    
    // For now, take the first batch
    let mel_tensor = tensor.get(0)
        .map_err(|e| AcousticError::ModelError(format!("Failed to get first batch: {}", e)))?;
    
    // Convert to Vec<Vec<f32>> format
    let mut data = Vec::with_capacity(n_mel_channels);
    
    for mel_idx in 0..n_mel_channels {
        let mel_channel = mel_tensor.get(mel_idx)
            .map_err(|e| AcousticError::ModelError(format!("Failed to get mel channel {}: {}", mel_idx, e)))?;
        
        let channel_data = mel_channel.to_vec1::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to convert channel {} to vec: {}", mel_idx, e)))?;
        
        data.push(channel_data);
    }
    
    let mel = MelSpectrogram::new(data, sample_rate, hop_length);
    
    tracing::debug!(
        "Converted tensor {:?} to MelSpectrogram: {}x{} frames",
        dims, mel.n_mels, mel.n_frames
    );
    
    Ok(mel)
}

/// Convert mel bin index to frequency in Hz
fn mel_idx_to_frequency(mel_idx: usize, total_mel_bins: usize) -> f32 {
    // Mel scale conversion: mel = 2595 * log10(1 + freq/700)
    // Inverse: freq = 700 * (10^(mel/2595) - 1)
    
    let mel_max = 2595.0 * (1.0_f32 + 8000.0 / 700.0).log10(); // Max mel for 8kHz
    let mel_value = (mel_idx as f32 / total_mel_bins as f32) * mel_max;
    
    700.0 * (10.0_f32.powf(mel_value / 2595.0) - 1.0)
}