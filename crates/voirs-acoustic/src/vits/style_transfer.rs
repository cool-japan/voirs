//! Style transfer module for VITS
//!
//! Provides voice style adaptation capabilities including:
//! - Style embedding extraction from reference audio
//! - Style-conditioned synthesis
//! - Style encoder network
//! - Optional adversarial training for style transfer

use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};

use crate::{AcousticError, Result};

use super::utils::LinearLayer;

/// Style transfer configuration
#[derive(Debug, Clone)]
pub struct StyleTransferConfig {
    /// Number of style embedding dimensions
    pub style_dim: usize,
    /// Learning rate for style adaptation
    pub adaptation_rate: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Use adversarial training for style transfer
    pub use_adversarial: bool,
}

impl Default for StyleTransferConfig {
    fn default() -> Self {
        Self {
            style_dim: 256,
            adaptation_rate: 0.001,
            adaptation_steps: 100,
            use_adversarial: true,
        }
    }
}

/// Style transfer module for VITS
pub struct StyleTransfer {
    config: StyleTransferConfig,
    /// Style encoder network
    style_encoder: Arc<StyleEncoder>,
    /// Style discriminator for adversarial training
    #[allow(dead_code)]
    style_discriminator: Option<Arc<StyleDiscriminator>>,
    /// Device for computation
    device: Device,
    /// Current style embeddings cache
    style_cache: Arc<Mutex<std::collections::HashMap<String, Tensor>>>,
}

impl StyleTransfer {
    /// Create new style transfer module
    pub fn new(config: StyleTransferConfig, device: Device) -> Result<Self> {
        let style_encoder = Arc::new(StyleEncoder::new(config.style_dim, device.clone())?);

        let style_discriminator = if config.use_adversarial {
            Some(Arc::new(StyleDiscriminator::new(
                config.style_dim,
                device.clone(),
            )?))
        } else {
            None
        };

        Ok(Self {
            config,
            style_encoder,
            style_discriminator,
            device,
            style_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        })
    }

    /// Extract style embedding from reference audio
    pub fn extract_style(&self, reference_audio: &Tensor) -> Result<Tensor> {
        // Extract mel-spectrogram from reference audio
        let mel_spec = self.audio_to_mel(reference_audio)?;

        // Extract style embedding using style encoder
        let style_embedding = self.style_encoder.encode(&mel_spec)?;

        Ok(style_embedding)
    }

    /// Transfer style from reference to target synthesis
    pub fn transfer_style(
        &self,
        phoneme_sequence: &[String],
        reference_style: &Tensor,
        target_speaker_id: Option<usize>,
    ) -> Result<Tensor> {
        // Adapt phoneme embeddings with style information
        let style_adapted_phonemes =
            self.adapt_phonemes_with_style(phoneme_sequence, reference_style)?;

        // Apply speaker conditioning if provided
        let conditioned_phonemes = if let Some(speaker_id) = target_speaker_id {
            self.apply_speaker_conditioning(&style_adapted_phonemes, speaker_id)?
        } else {
            style_adapted_phonemes
        };

        Ok(conditioned_phonemes)
    }

    /// Adapt phoneme embeddings with style information
    fn adapt_phonemes_with_style(&self, phonemes: &[String], style: &Tensor) -> Result<Tensor> {
        // Convert phonemes to embeddings
        let phoneme_embeddings = self.phonemes_to_embeddings(phonemes)?;

        // Broadcast style to match phoneme sequence length
        let style_broadcast = style.broadcast_as(phoneme_embeddings.shape())?;

        // Combine phoneme and style embeddings
        let scale_tensor = Tensor::new(&[0.5f32], style_broadcast.device())?;
        let styled = (style_broadcast * scale_tensor)?;
        let combined = (phoneme_embeddings + styled)?;

        Ok(combined)
    }

    /// Apply speaker conditioning to style-adapted phonemes
    fn apply_speaker_conditioning(&self, phonemes: &Tensor, speaker_id: usize) -> Result<Tensor> {
        // Create speaker embedding
        let speaker_embedding = self.create_speaker_embedding(speaker_id)?;

        // Apply speaker conditioning
        let speaker_broadcast = speaker_embedding.broadcast_as(phonemes.shape())?;
        let conditioned = (phonemes + speaker_broadcast)?;

        Ok(conditioned)
    }

    /// Convert reference audio to a log-mel spectrogram tensor.
    ///
    /// Runs a real STFT → power-spectrum → mel-filterbank → log pipeline by
    /// reusing the crate's [`MelComputer`](crate::mel::MelComputer) (80-band,
    /// 22.05 kHz, matching the [`StyleEncoder`]'s 80-dim input). The PCM input
    /// arrives as a `Tensor` of arbitrary rank, so it is flattened to a 1-D
    /// sample buffer; the result is returned as a `[n_frames, n_mels]` tensor
    /// (feature axis last) so the encoder's `Linear(80 -> ...)` layers consume
    /// it directly.
    fn audio_to_mel(&self, audio: &Tensor) -> Result<Tensor> {
        use crate::mel::{MelComputer, MelParams};

        // Flatten whatever shape the reference audio arrives in to a 1-D buffer.
        let samples = audio
            .flatten_all()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to flatten reference audio: {e}"),
            })?
            .to_vec1::<f32>()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Failed to read reference audio samples: {e}"),
            })?;

        if samples.is_empty() {
            return Err(AcousticError::InputError {
                message: "Reference audio is empty".to_string(),
            });
        }

        // 80-band, 22.05 kHz log-mel matching the StyleEncoder's input width.
        let params = MelParams::standard_22khz();
        let n_mels = params.n_mels as usize;
        let computer = MelComputer::new(params)?;

        // `MelSpectrogram::data` is laid out as [n_mels, n_frames].
        let mel = computer.compute(&samples)?;
        let n_frames = mel.n_frames;

        // Transpose to [n_frames, n_mels] so the mel (feature) axis is last.
        let mut frame_major = vec![0.0f32; n_frames * n_mels];
        for (mel_idx, channel) in mel.data.iter().enumerate().take(n_mels) {
            for (frame_idx, &value) in channel.iter().enumerate().take(n_frames) {
                frame_major[frame_idx * n_mels + mel_idx] = value;
            }
        }

        let mel_tensor =
            Tensor::from_vec(frame_major, (n_frames, n_mels), &self.device).map_err(|e| {
                AcousticError::ProcessingError {
                    message: format!("Failed to build mel tensor: {e}"),
                }
            })?;

        Ok(mel_tensor)
    }

    /// Convert phonemes to embeddings
    fn phonemes_to_embeddings(&self, phonemes: &[String]) -> Result<Tensor> {
        // Simplified phoneme to embedding conversion
        let embedding_dim = self.config.style_dim;
        let seq_len = phonemes.len();

        let embeddings = Tensor::randn(0f32, 1f32, &[seq_len, embedding_dim], &self.device)?;
        Ok(embeddings)
    }

    /// Create speaker embedding
    fn create_speaker_embedding(&self, _speaker_id: usize) -> Result<Tensor> {
        // Simplified speaker embedding creation
        let speaker_embedding = Tensor::randn(0f32, 1f32, &[self.config.style_dim], &self.device)?;
        Ok(speaker_embedding)
    }

    /// Cache style embedding with identifier
    pub fn cache_style(&self, style_id: String, style_embedding: Tensor) -> Result<()> {
        let mut cache = self
            .style_cache
            .lock()
            .map_err(|_| AcousticError::ProcessingError {
                message: "Failed to lock style cache".to_string(),
            })?;
        cache.insert(style_id, style_embedding);
        Ok(())
    }

    /// Retrieve cached style embedding
    pub fn get_cached_style(&self, style_id: &str) -> Result<Option<Tensor>> {
        let cache = self
            .style_cache
            .lock()
            .map_err(|_| AcousticError::ProcessingError {
                message: "Failed to lock style cache".to_string(),
            })?;
        Ok(cache.get(style_id).cloned())
    }
}

/// Style encoder network
pub(crate) struct StyleEncoder {
    layers: Vec<LinearLayer>,
    #[allow(dead_code)]
    device: Device,
}

impl StyleEncoder {
    pub(crate) fn new(style_dim: usize, device: Device) -> Result<Self> {
        let layers = vec![
            LinearLayer::new(80, 512, device.clone())?, // Mel-spec input
            LinearLayer::new(512, 256, device.clone())?,
            LinearLayer::new(256, style_dim, device.clone())?,
        ];

        Ok(Self { layers, device })
    }

    pub(crate) fn encode(&self, mel_spec: &Tensor) -> Result<Tensor> {
        let mut x = mel_spec.clone();

        // Apply layers with ReLU activation
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        // Global average pooling over time dimension
        x = x.mean(1)?;

        Ok(x)
    }
}

/// Style discriminator for adversarial training
pub(crate) struct StyleDiscriminator {
    #[allow(dead_code)]
    layers: Vec<LinearLayer>,
    #[allow(dead_code)]
    device: Device,
}

impl StyleDiscriminator {
    pub(crate) fn new(style_dim: usize, device: Device) -> Result<Self> {
        let layers = vec![
            LinearLayer::new(style_dim, 256, device.clone())?,
            LinearLayer::new(256, 128, device.clone())?,
            LinearLayer::new(128, 1, device.clone())?, // Binary classification
        ];

        Ok(Self { layers, device })
    }

    #[allow(dead_code)]
    fn discriminate(&self, style_embedding: &Tensor) -> Result<Tensor> {
        let mut x = style_embedding.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        // Binary classification output (simplified without sigmoid)
        // In real implementation, would use sigmoid or softmax

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// `audio_to_mel` of a pure tone must yield a `[n_frames, n_mels]` log-mel
    /// tensor with its energy concentrated in the low mel bands (a 440 Hz tone
    /// sits well below the 11 kHz Nyquist band).
    #[test]
    fn test_audio_to_mel_tone() {
        let device = Device::Cpu;
        let style_transfer =
            StyleTransfer::new(StyleTransferConfig::default(), device.clone()).unwrap();

        // 0.5 s of a 440 Hz sine at 22.05 kHz.
        let sample_rate = 22050.0f32;
        let frequency = 440.0f32;
        let n_samples = 11025usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin())
            .collect();
        let audio = Tensor::from_vec(samples, (n_samples,), &device).unwrap();

        let mel = style_transfer.audio_to_mel(&audio).unwrap();

        // Shape must be [n_frames, n_mels] with the standard 80 mel bands.
        let dims = mel.dims();
        assert_eq!(dims.len(), 2, "expected 2-D mel tensor, got {dims:?}");
        let n_frames = dims[0];
        let n_mels = dims[1];
        assert_eq!(n_mels, 80);
        assert!(n_frames > 1, "expected multiple frames, got {n_frames}");

        // Energy (de-logged) must favour the low bands over the high bands.
        let data = mel.to_vec2::<f32>().unwrap(); // [n_frames][n_mels]
        let mut low_energy = 0.0f32;
        let mut high_energy = 0.0f32;
        for frame in &data {
            for (bin, &value) in frame.iter().enumerate() {
                if bin < 20 {
                    low_energy += value.exp();
                } else if bin >= 60 {
                    high_energy += value.exp();
                }
            }
        }
        assert!(
            low_energy > high_energy,
            "expected more low-band energy for a 440 Hz tone: low={low_energy}, high={high_energy}"
        );
    }

    /// An empty reference audio tensor must be rejected rather than silently
    /// producing a degenerate mel.
    #[test]
    fn test_audio_to_mel_empty_errors() {
        let device = Device::Cpu;
        let style_transfer =
            StyleTransfer::new(StyleTransferConfig::default(), device.clone()).unwrap();

        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        assert!(style_transfer.audio_to_mel(&empty).is_err());
    }
}
