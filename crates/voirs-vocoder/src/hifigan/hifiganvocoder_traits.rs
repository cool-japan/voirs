//! # HiFiGanVocoder - Trait Implementations
//!
//! This module contains trait implementations for `HiFiGanVocoder`.
//!
//! ## Implemented Traits
//!
//! - `Vocoder`
//! - `Default`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HiFiGanVocoder, StreamingVocoder};
use crate::effects::EffectPresets;
use crate::models::hifigan::HiFiGanVariant;
use crate::{
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderFeature, VocoderMetadata,
};
use async_trait::async_trait;
use futures::Stream;

#[async_trait]
impl Vocoder for HiFiGanVocoder {
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "candle")]
        {
            if let Some(inference) = &self.inference {
                let (mut audio, _stats) = inference.infer(mel, config).await?;
                self.apply_basic_post_processing(&mut audio);
                self.apply_emotion_processing(&mut audio);
                self.apply_voice_conversion_processing(&mut audio)?;
                self.apply_unified_conditioning_processing(&mut audio)?;
                Ok(audio)
            } else {
                Err(crate::VocoderError::ModelError(
                    "HiFi-GAN inference not initialized. Call initialize_inference() first."
                        .to_string(),
                ))
            }
        }
        #[cfg(not(feature = "candle"))]
        {
            tracing::info!(
                "HiFi-GAN inference requires Candle feature. Using basic vocoder fallback."
            );
            let mut audio = self.synthesize_from_mel_fallback(mel, config)?;
            self.apply_basic_post_processing(&mut audio);
            self.apply_emotion_processing(&mut audio);
            self.apply_voice_conversion_processing(&mut audio)?;
            self.apply_unified_conditioning_processing(&mut audio)?;
            Ok(audio)
        }
    }
    async fn vocode_stream(
        &self,
        mut mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::StreamExt;
        let mut streaming_vocoder = StreamingVocoder::new(self.clone(), config.cloned())?;
        let mut results = Vec::new();
        while let Some(mel) = mel_stream.next().await {
            match streaming_vocoder.process_chunk(&mel).await {
                Ok(Some(audio)) => results.push(Ok(audio)),
                Ok(None) => {}
                Err(e) => results.push(Err(e)),
            }
        }
        if let Ok(Some(audio)) = streaming_vocoder.flush().await {
            results.push(Ok(audio));
        }
        let audio_stream = futures::stream::iter(results);
        Ok(Box::new(audio_stream))
    }
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::new();
        for (i, mel) in mels.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            results.push(self.vocode(mel, config).await?);
        }
        Ok(results)
    }
    fn metadata(&self) -> VocoderMetadata {
        self.metadata.clone()
    }
    fn supports(&self, feature: VocoderFeature) -> bool {
        match feature {
            VocoderFeature::StreamingInference => true,
            VocoderFeature::BatchProcessing => true,
            VocoderFeature::GpuAcceleration => cfg!(feature = "candle"),
            VocoderFeature::HighQuality => true,
            VocoderFeature::RealtimeProcessing => {
                matches!(self.config.variant, HiFiGanVariant::V2 | HiFiGanVariant::V3)
            }
            VocoderFeature::FastInference => {
                matches!(self.config.variant, HiFiGanVariant::V1 | HiFiGanVariant::V2)
            }
            VocoderFeature::EmotionConditioning => true,
            VocoderFeature::VoiceConversion => true,
            VocoderFeature::AgeTransformation => true,
            VocoderFeature::GenderTransformation => true,
            VocoderFeature::VoiceMorphing => true,
            VocoderFeature::SingingVoice => true,
            VocoderFeature::SpatialAudio => true,
            VocoderFeature::Base => true,
            VocoderFeature::Emotion => true,
            VocoderFeature::Singing => true,
            VocoderFeature::Spatial => true,
        }
    }
}

impl Default for HiFiGanVocoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HiFiGanVocoder {
    fn clone(&self) -> Self {
        let mut new_vocoder = HiFiGanVocoder::with_config(self.config.clone());
        new_vocoder.effect_chain = EffectPresets::speech_enhancement(self.config.sample_rate);
        new_vocoder
            .effect_chain
            .set_bypass(self.effect_chain.is_bypassed());
        new_vocoder
            .effect_chain
            .set_wet_dry_mix(self.effect_chain.get_wet_dry_mix());
        new_vocoder
            .effect_chain
            .set_output_gain(self.effect_chain.get_output_gain());
        new_vocoder
    }
}
