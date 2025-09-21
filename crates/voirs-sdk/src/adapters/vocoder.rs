//! Vocoder adapter for integrating voirs-vocoder with the SDK.

use crate::traits::{
    Vocoder as SdkVocoder, VocoderFeature as SdkVocoderFeature,
    VocoderMetadata as SdkVocoderMetadata,
};
use crate::types::MelSpectrogram as SdkMelSpectrogram;
use crate::AudioBuffer as SdkAudioBuffer;
use crate::{Result, VoirsError};
use async_trait::async_trait;
use futures::Stream;
use std::sync::Arc;

/// Adapter that bridges voirs-vocoder components to SDK Vocoder trait
pub struct VocoderAdapter {
    inner: Arc<dyn voirs_vocoder::Vocoder>,
}

impl VocoderAdapter {
    /// Create new vocoder adapter wrapping a voirs-vocoder implementation
    pub fn new(vocoder: Arc<dyn voirs_vocoder::Vocoder>) -> Self {
        Self { inner: vocoder }
    }

    /// Convert SDK MelSpectrogram to voirs-vocoder MelSpectrogram
    fn convert_mel_spectrogram_to_vocoder(
        mel: &SdkMelSpectrogram,
    ) -> voirs_vocoder::MelSpectrogram {
        voirs_vocoder::MelSpectrogram {
            data: mel.data.clone(),
            sample_rate: mel.sample_rate,
            hop_length: mel.hop_length,
            n_mels: mel.n_mels as usize,
            n_frames: mel.n_frames as usize,
        }
    }

    /// Convert voirs-vocoder AudioBuffer to SDK AudioBuffer
    fn convert_audio_buffer_from_vocoder(audio: voirs_vocoder::AudioBuffer) -> SdkAudioBuffer {
        SdkAudioBuffer::new(
            audio.samples().to_vec(),
            audio.sample_rate(),
            audio.channels(),
        )
    }

    /// Convert SDK SynthesisConfig to voirs-vocoder SynthesisConfig
    fn convert_synthesis_config_to_vocoder(
        config: &Option<&crate::types::SynthesisConfig>,
    ) -> Option<voirs_vocoder::SynthesisConfig> {
        config.map(|cfg| {
            voirs_vocoder::SynthesisConfig {
                speed: cfg.speaking_rate,
                pitch_shift: cfg.pitch_shift,
                energy: 10.0_f32.powf(cfg.volume_gain / 20.0), // Convert dB to linear
                speaker_id: None, // SDK doesn't have speaker_id in SynthesisConfig
                seed: cfg.seed,
            }
        })
    }

    /// Convert voirs-vocoder VocoderMetadata to SDK VocoderMetadata
    fn convert_metadata_from_vocoder(
        metadata: voirs_vocoder::VocoderMetadata,
    ) -> SdkVocoderMetadata {
        SdkVocoderMetadata {
            name: metadata.name,
            version: metadata.version,
            architecture: metadata.architecture,
            sample_rate: metadata.sample_rate,
            mel_channels: metadata.mel_channels,
            latency_ms: metadata.latency_ms,
            quality_score: metadata.quality_score,
        }
    }

    /// Convert SDK VocoderFeature to voirs-vocoder VocoderFeature
    fn convert_feature_to_vocoder(feature: SdkVocoderFeature) -> voirs_vocoder::VocoderFeature {
        match feature {
            SdkVocoderFeature::StreamingInference => {
                voirs_vocoder::VocoderFeature::StreamingInference
            }
            SdkVocoderFeature::BatchProcessing => voirs_vocoder::VocoderFeature::BatchProcessing,
            SdkVocoderFeature::GpuAcceleration => voirs_vocoder::VocoderFeature::GpuAcceleration,
            SdkVocoderFeature::MultiSampleRate => voirs_vocoder::VocoderFeature::RealtimeProcessing, // Map to closest feature
            SdkVocoderFeature::EnhancementFilters => voirs_vocoder::VocoderFeature::HighQuality,
            SdkVocoderFeature::RealtimeProcessing => {
                voirs_vocoder::VocoderFeature::RealtimeProcessing
            }
        }
    }
}

#[async_trait]
impl SdkVocoder for VocoderAdapter {
    async fn vocode(
        &self,
        mel: &SdkMelSpectrogram,
        config: Option<&crate::types::SynthesisConfig>,
    ) -> Result<SdkAudioBuffer> {
        let vocoder_mel = Self::convert_mel_spectrogram_to_vocoder(mel);
        let vocoder_config = Self::convert_synthesis_config_to_vocoder(&config);

        match self
            .inner
            .vocode(&vocoder_mel, vocoder_config.as_ref())
            .await
        {
            Ok(audio) => Ok(Self::convert_audio_buffer_from_vocoder(audio)),
            Err(err) => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Vocoder,
                message: format!("Vocoder synthesis failed: {err}"),
                source: Some(Box::new(err)),
            }),
        }
    }

    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn Stream<Item = SdkMelSpectrogram> + Send + Unpin>,
        config: Option<&crate::types::SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<SdkAudioBuffer>> + Send + Unpin>> {
        let vocoder_config = Self::convert_synthesis_config_to_vocoder(&config);

        // Create a stream that converts SDK MelSpectrogram to vocoder MelSpectrogram
        let vocoder_mel_stream = Box::pin(async_stream::stream! {
            let mut pinned_stream = mel_stream;
            while let Some(mel) = futures::StreamExt::next(&mut pinned_stream).await {
                yield Self::convert_mel_spectrogram_to_vocoder(&mel);
            }
        });

        match self
            .inner
            .vocode_stream(Box::new(vocoder_mel_stream), vocoder_config.as_ref())
            .await
        {
            Ok(stream) => {
                // Convert the resulting stream back to SDK types
                let result_stream = Box::pin(async_stream::stream! {
                    let mut pinned_stream = stream;
                    while let Some(result) = futures::StreamExt::next(&mut pinned_stream).await {
                        match result {
                            Ok(audio) => yield Ok(Self::convert_audio_buffer_from_vocoder(audio)),
                            Err(err) => yield Err(VoirsError::ModelError {
                                model_type: crate::error::types::ModelType::Vocoder,
                                message: format!("Vocoder stream synthesis failed: {err}"),
                                source: Some(Box::new(err)),
                            }),
                        }
                    }
                });
                Ok(Box::new(result_stream))
            }
            Err(err) => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Vocoder,
                message: format!("Vocoder stream setup failed: {err}"),
                source: Some(Box::new(err)),
            }),
        }
    }

    async fn vocode_batch(
        &self,
        mels: &[SdkMelSpectrogram],
        configs: Option<&[crate::types::SynthesisConfig]>,
    ) -> Result<Vec<SdkAudioBuffer>> {
        let vocoder_mels: Vec<_> = mels
            .iter()
            .map(Self::convert_mel_spectrogram_to_vocoder)
            .collect();

        let vocoder_configs: Option<Vec<_>> = configs.map(|cfgs| {
            cfgs.iter()
                .map(|cfg| Self::convert_synthesis_config_to_vocoder(&Some(cfg)).unwrap())
                .collect()
        });

        match self
            .inner
            .vocode_batch(&vocoder_mels, vocoder_configs.as_deref())
            .await
        {
            Ok(audios) => Ok(audios
                .into_iter()
                .map(Self::convert_audio_buffer_from_vocoder)
                .collect()),
            Err(err) => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Vocoder,
                message: format!("Vocoder batch synthesis failed: {err}"),
                source: Some(Box::new(err)),
            }),
        }
    }

    fn metadata(&self) -> SdkVocoderMetadata {
        Self::convert_metadata_from_vocoder(self.inner.metadata())
    }

    fn supports(&self, feature: SdkVocoderFeature) -> bool {
        let vocoder_feature = Self::convert_feature_to_vocoder(feature);
        self.inner.supports(vocoder_feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_vocoder::HiFiGanVocoder;

    #[tokio::test]
    async fn test_vocoder_adapter_creation() {
        let mut hifigan = HiFiGanVocoder::new();
        // Initialize inference for testing
        hifigan.initialize_inference_for_testing().unwrap();
        let adapter = VocoderAdapter::new(Arc::new(hifigan));

        // Test that adapter can be created and basic functionality works
        let metadata = adapter.metadata();
        assert!(!metadata.name.is_empty());
        assert!(metadata.sample_rate > 0);
        assert!(metadata.mel_channels > 0);
    }

    #[tokio::test]
    async fn test_vocoder_adapter_synthesis() {
        let mut hifigan = HiFiGanVocoder::new();
        // Initialize inference for testing
        hifigan.initialize_inference_for_testing().unwrap();
        let adapter = VocoderAdapter::new(Arc::new(hifigan));

        // Create a test mel spectrogram
        let mel = SdkMelSpectrogram {
            data: vec![vec![0.1, 0.2, 0.3]; 80], // 80 mel bins with 3 time frames
            sample_rate: 22050,
            hop_length: 256,
            n_mels: 80,
            n_frames: 3,
        };

        let result = adapter.vocode(&mel, None).await;
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert!(!audio.samples().is_empty());
        assert!(audio.sample_rate() > 0);
        assert!(audio.channels() > 0);
    }

    #[tokio::test]
    async fn test_vocoder_adapter_batch_synthesis() {
        let mut hifigan = HiFiGanVocoder::new();
        // Initialize inference for testing
        hifigan.initialize_inference_for_testing().unwrap();
        let adapter = VocoderAdapter::new(Arc::new(hifigan));

        // Create test mel spectrograms
        let mels = vec![
            SdkMelSpectrogram {
                data: vec![vec![0.1, 0.2, 0.3]; 80], // 80 mel bins with 3 time frames
                sample_rate: 22050,
                hop_length: 256,
                n_mels: 80,
                n_frames: 3,
            },
            SdkMelSpectrogram {
                data: vec![vec![0.4, 0.5, 0.6]; 80], // 80 mel bins with 3 time frames
                sample_rate: 22050,
                hop_length: 256,
                n_mels: 80,
                n_frames: 3,
            },
        ];

        let result = adapter.vocode_batch(&mels, None).await;
        assert!(result.is_ok());

        let audios = result.unwrap();
        assert_eq!(audios.len(), 2);

        for audio in audios {
            assert!(!audio.samples().is_empty());
            assert!(audio.sample_rate() > 0);
            assert!(audio.channels() > 0);
        }
    }
}
