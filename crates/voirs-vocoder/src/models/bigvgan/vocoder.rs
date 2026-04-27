//! `Vocoder` trait implementation for [`BigVGANInference`].
//!
//! Bridges the low-level BigVGAN generator API (tensor in / tensor out) with the
//! high-level [`Vocoder`] trait used by the VoiRS pipeline.

#[cfg(feature = "candle")]
use super::inference::BigVGANInference;

#[cfg(feature = "candle")]
use crate::{
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderError, VocoderFeature,
    VocoderMetadata,
};

#[cfg(feature = "candle")]
use async_trait::async_trait;

#[cfg(feature = "candle")]
use candle_core::{DType, Tensor};

#[cfg(feature = "candle")]
#[async_trait]
impl Vocoder for BigVGANInference {
    /// Convert a [`MelSpectrogram`] to an [`AudioBuffer`] using BigVGAN.
    ///
    /// The mel data is expected in row-major order: `data[mel_channel][time_frame]`.
    /// The tensor passed to the generator has shape `[1, n_mels, n_frames]`.
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        _config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        // Flatten Vec<Vec<f32>> (outer = mel channels, inner = time frames)
        // into a contiguous buffer in row-major (C) order for shape [n_mels, n_frames].
        let mut flat: Vec<f32> = Vec::with_capacity(mel.n_mels * mel.n_frames);
        for mel_row in &mel.data {
            flat.extend_from_slice(mel_row);
        }

        let mel_tensor = Tensor::from_vec(flat, vec![1, mel.n_mels, mel.n_frames], self.device())
            .map_err(|e| {
            VocoderError::ModelError(format!("BigVGAN: failed to build mel tensor: {}", e))
        })?;

        // Ensure tensor dtype is F32 (Tensor::from_vec with f32 should already be F32)
        let mel_tensor = mel_tensor
            .to_dtype(DType::F32)
            .map_err(|e| VocoderError::ModelError(format!("BigVGAN: dtype cast error: {}", e)))?;

        // Generator forward pass → [1, 1, samples]
        let output = self.generate(&mel_tensor)?;

        // Extract the flat sample vector
        let samples = output
            .squeeze(0)
            .map_err(|e| VocoderError::ModelError(format!("BigVGAN: squeeze(0) failed: {}", e)))?
            .squeeze(0)
            .map_err(|e| {
                VocoderError::ModelError(format!("BigVGAN: squeeze(0) second call failed: {}", e))
            })?
            .to_vec1::<f32>()
            .map_err(|e| VocoderError::ModelError(format!("BigVGAN: to_vec1 failed: {}", e)))?;

        Ok(AudioBuffer::new(samples, self.config().sample_rate, 1))
    }

    /// Streaming vocoding — spawns a Tokio task that drains the mel stream and
    /// sends converted [`AudioBuffer`]s over an unbounded channel.
    async fn vocode_stream(
        &self,
        mut mel_stream: Box<dyn futures::Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::StreamExt;
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        // Clone self so the spawned task can own it
        let vocoder = self.clone();

        // Clone config (if provided) so it lives in the task
        let config_owned: Option<SynthesisConfig> = config.cloned();

        let (tx, rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            while let Some(mel) = mel_stream.next().await {
                let result = vocoder.vocode(&mel, config_owned.as_ref()).await;
                if tx.send(result).is_err() {
                    // Receiver dropped — stop streaming
                    break;
                }
            }
        });

        Ok(Box::new(UnboundedReceiverStream::new(rx)))
    }

    /// Batch vocoding — sequential loop over each mel spectrogram.
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::with_capacity(mels.len());
        for (idx, mel) in mels.iter().enumerate() {
            let cfg = configs.and_then(|c| c.get(idx));
            let audio = self.vocode(mel, cfg).await?;
            results.push(audio);
        }
        Ok(results)
    }

    fn metadata(&self) -> VocoderMetadata {
        VocoderMetadata {
            name: "BigVGAN".to_string(),
            version: "0.1.0".to_string(),
            architecture: "BigVGAN+Snake".to_string(),
            sample_rate: self.config().sample_rate,
            mel_channels: self.config().num_mels as u32,
            latency_ms: 80.0,
            quality_score: 4.6,
        }
    }

    fn supports(&self, feature: VocoderFeature) -> bool {
        matches!(
            feature,
            VocoderFeature::BatchProcessing
                | VocoderFeature::HighQuality
                | VocoderFeature::GpuAcceleration
                | VocoderFeature::FastInference
        )
    }
}

#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;
    use crate::models::bigvgan::config::BigVGANConfig;
    use candle_core::Device;

    /// Helper that constructs a simple [`MelSpectrogram`] of the given size.
    fn make_mel(n_mels: usize, n_frames: usize, sample_rate: u32) -> MelSpectrogram {
        MelSpectrogram::new(vec![vec![0.0f32; n_frames]; n_mels], sample_rate, 256)
    }

    #[tokio::test]
    async fn test_vocode_produces_audio() {
        let config = BigVGANConfig::fast_24khz();
        let n_mels = config.num_mels;
        let sample_rate = config.sample_rate;
        let device = Device::Cpu;

        let inference = BigVGANInference::new(config, device).unwrap();
        let mel = make_mel(n_mels, 16, sample_rate);

        let result = inference.vocode(&mel, None).await;
        assert!(result.is_ok(), "vocode failed: {:?}", result.err());

        let audio = result.unwrap();
        assert!(!audio.is_empty(), "expected non-empty audio output");
    }

    #[test]
    fn test_load_weights_rejects_no_match() {
        use safetensors::serialize_to_file;
        use std::collections::HashMap;

        let config = BigVGANConfig::fast_24khz();
        let device = Device::Cpu;
        let mut inference = BigVGANInference::new(config, device).unwrap();

        // Build a minimal safetensors file whose tensor name will not match
        // any registered VarMap variable.
        let data_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let shape: Vec<usize> = vec![4usize];
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &data_bytes)
                .unwrap();

        let mut metadata_map: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
        metadata_map.insert("nonexistent.bias".to_string(), view);

        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("bigvgan_test_no_match.safetensors");

        serialize_to_file(metadata_map, None, tmp_path.as_path())
            .expect("safetensors serialize_to_file failed");

        let result = inference.load_weights(&tmp_path);
        assert!(
            result.is_err(),
            "expected Err when no weights match VarMap, got Ok"
        );

        // Clean up
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[tokio::test]
    async fn test_metadata_values() {
        let config = BigVGANConfig::fast_24khz();
        let device = Device::Cpu;
        let inference = BigVGANInference::new(config.clone(), device).unwrap();

        let meta = inference.metadata();
        assert_eq!(meta.name, "BigVGAN");
        assert_eq!(meta.sample_rate, config.sample_rate);
        assert_eq!(meta.mel_channels, config.num_mels as u32);
    }

    #[tokio::test]
    async fn test_supports_features() {
        let config = BigVGANConfig::fast_24khz();
        let device = Device::Cpu;
        let inference = BigVGANInference::new(config, device).unwrap();

        assert!(inference.supports(VocoderFeature::BatchProcessing));
        assert!(inference.supports(VocoderFeature::HighQuality));
        assert!(inference.supports(VocoderFeature::GpuAcceleration));
        assert!(inference.supports(VocoderFeature::FastInference));
        assert!(!inference.supports(VocoderFeature::SingingVoice));
    }
}
