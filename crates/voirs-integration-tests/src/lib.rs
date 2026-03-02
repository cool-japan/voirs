//! Integration test helpers for VoiRS.
//!
//! This crate provides bridge implementations and helpers used by integration tests.

use std::sync::Arc;

pub use voirs_acoustic as acoustic;
pub use voirs_g2p as g2p;
pub use voirs_sdk as sdk;
pub use voirs_vocoder as vocoder;

// Re-export SDK types for use in integration tests
// Note: pub use also brings items into scope for use within this module
pub use voirs_sdk::{
    AcousticModel, AudioBuffer, AudioFormat, G2p, LanguageCode, MelSpectrogram, Phoneme,
    QualityLevel, Result, SynthesisConfig, Vocoder, VoirsError, VoirsPipeline,
    VoirsPipelineBuilder,
};

// Backend enums for convenient API
#[derive(Debug, Clone, Copy)]
pub enum G2pBackend {
    Phonetisaurus,
    Neural,
    RuleBased,
}

#[derive(Debug, Clone, Copy)]
pub enum AcousticBackend {
    Vits,
    FastSpeech2,
}

#[derive(Debug, Clone, Copy)]
pub enum VocoderBackend {
    HifiGan,
    DiffWave,
    WaveGlow,
}

// Helper functions for creating backend components
pub fn create_g2p(backend: G2pBackend) -> Arc<dyn G2p> {
    use voirs_sdk::pipeline::DummyG2p;

    match backend {
        G2pBackend::Phonetisaurus => {
            let neural_g2p = g2p::backends::neural::NeuralG2pBackend::new(
                g2p::backends::neural::LstmConfig::default(),
            )
            .unwrap_or_else(|_| g2p::backends::neural::NeuralG2pBackend::default());
            Arc::new(G2pBridge::new(neural_g2p))
        }
        G2pBackend::Neural => {
            match g2p::backends::neural::NeuralG2pBackend::new(
                g2p::backends::neural::LstmConfig::default(),
            ) {
                Ok(neural_g2p) => Arc::new(G2pBridge::new(neural_g2p)),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to initialize NeuralG2p: {e}. Falling back to DummyG2p."
                    );
                    Arc::new(DummyG2p::new())
                }
            }
        }
        G2pBackend::RuleBased => {
            match g2p::rules::EnglishRuleG2p::new() {
                Ok(rule_g2p) => Arc::new(G2pBridge::new(rule_g2p)),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize EnglishRuleG2p: {e}. Falling back to DummyG2p.");
                    Arc::new(DummyG2p::new())
                }
            }
        }
    }
}

pub fn create_acoustic(backend: AcousticBackend) -> Arc<dyn AcousticModel> {
    use voirs_sdk::pipeline::DummyAcoustic;

    match backend {
        AcousticBackend::Vits => {
            match acoustic::vits::VitsModel::new() {
                Ok(vits_model) => Arc::new(AcousticBridge::new(vits_model)),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize VITS model: {e}. Falling back to DummyAcoustic.");
                    Arc::new(DummyAcoustic::new())
                }
            }
        }
        AcousticBackend::FastSpeech2 => {
            let fastspeech2_model = acoustic::fastspeech::FastSpeech2Model::new();
            Arc::new(AcousticBridge::new(fastspeech2_model))
        }
    }
}

pub fn create_vocoder(backend: VocoderBackend) -> Arc<dyn Vocoder> {
    use voirs_sdk::pipeline::DummyVocoder;

    match backend {
        VocoderBackend::HifiGan => {
            let mut hifigan_vocoder = vocoder::hifigan::HiFiGanVocoder::new();
            if let Err(e) = hifigan_vocoder.initialize_inference_for_testing() {
                eprintln!("Warning: Failed to initialize HiFi-GAN inference: {e}. Falling back to DummyVocoder.");
                Arc::new(DummyVocoder::new())
            } else {
                Arc::new(VocoderBridge::new(hifigan_vocoder))
            }
        }
        VocoderBackend::DiffWave => {
            match vocoder::models::diffwave::DiffWaveVocoder::with_default_config() {
                Ok(diffwave_vocoder) => Arc::new(VocoderBridge::new(diffwave_vocoder)),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize DiffWave vocoder: {e}. Falling back to DummyVocoder.");
                    Arc::new(DummyVocoder::new())
                }
            }
        }
        VocoderBackend::WaveGlow => {
            match vocoder::waveglow::WaveGlowVocoder::new(
                vocoder::waveglow::WaveGlowConfig::default(),
            ) {
                Ok(waveglow_vocoder) => Arc::new(VocoderBridge::new(waveglow_vocoder)),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize WaveGlow vocoder: {e}. Falling back to DummyVocoder.");
                    Arc::new(DummyVocoder::new())
                }
            }
        }
    }
}

/// Prelude module for easy imports in integration tests
pub mod prelude {
    pub use crate::{
        create_acoustic, create_g2p, create_vocoder, AcousticBackend, G2pBackend, VocoderBackend,
    };
    pub use async_trait::async_trait;
    pub use voirs_sdk::{
        AcousticModel, AudioBuffer, AudioFormat, G2p, LanguageCode, MelSpectrogram, Phoneme,
        QualityLevel, Result, SynthesisConfig, Vocoder, VoirsError, VoirsPipeline,
        VoirsPipelineBuilder,
    };
}

// Bridge implementations

/// Bridge for connecting voirs-g2p types to SDK G2p trait
pub struct G2pBridge<T> {
    inner: T,
}

impl<T> G2pBridge<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<T> G2p for G2pBridge<T>
where
    T: voirs_g2p::G2p + Send + Sync,
{
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        let g2p_lang = lang.map(convert_language_code_to_g2p);
        let g2p_result =
            self.inner
                .to_phonemes(text, g2p_lang)
                .await
                .map_err(|e| VoirsError::G2pError {
                    text: text.to_string(),
                    message: format!("G2P error: {e}"),
                    language: lang.map(|l| l.as_str().to_string()),
                })?;

        let sdk_phonemes = g2p_result.into_iter().map(convert_phoneme_to_sdk).collect();
        Ok(sdk_phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.inner
            .supported_languages()
            .into_iter()
            .map(convert_language_code_to_sdk)
            .collect()
    }

    fn metadata(&self) -> voirs_sdk::traits::G2pMetadata {
        let g2p_meta = self.inner.metadata();
        voirs_sdk::traits::G2pMetadata {
            name: g2p_meta.name,
            version: g2p_meta.version,
            description: g2p_meta.description,
            supported_languages: g2p_meta
                .supported_languages
                .into_iter()
                .map(convert_language_code_to_sdk)
                .collect(),
            accuracy_scores: g2p_meta
                .accuracy_scores
                .into_iter()
                .map(|(k, v)| (convert_language_code_to_sdk(k), v))
                .collect(),
        }
    }
}

fn convert_language_code_to_g2p(lang: LanguageCode) -> voirs_g2p::LanguageCode {
    match lang {
        LanguageCode::EnUs => voirs_g2p::LanguageCode::EnUs,
        LanguageCode::EnGb => voirs_g2p::LanguageCode::EnUs,
        LanguageCode::DeDe => voirs_g2p::LanguageCode::De,
        LanguageCode::FrFr => voirs_g2p::LanguageCode::Fr,
        LanguageCode::EsEs | LanguageCode::EsMx => voirs_g2p::LanguageCode::Es,
        LanguageCode::JaJp => voirs_g2p::LanguageCode::Ja,
        LanguageCode::KoKr => voirs_g2p::LanguageCode::Ko,
        LanguageCode::ZhCn => voirs_g2p::LanguageCode::EnUs,
        _ => voirs_g2p::LanguageCode::EnUs,
    }
}

fn convert_language_code_to_sdk(lang: voirs_g2p::LanguageCode) -> LanguageCode {
    match lang {
        voirs_g2p::LanguageCode::EnUs => LanguageCode::EnUs,
        voirs_g2p::LanguageCode::EnGb => LanguageCode::EnGb,
        voirs_g2p::LanguageCode::De => LanguageCode::DeDe,
        voirs_g2p::LanguageCode::Fr => LanguageCode::FrFr,
        voirs_g2p::LanguageCode::Es => LanguageCode::EsEs,
        voirs_g2p::LanguageCode::Ja => LanguageCode::JaJp,
        voirs_g2p::LanguageCode::Ko => LanguageCode::KoKr,
        voirs_g2p::LanguageCode::ZhCn => LanguageCode::ZhCn,
        voirs_g2p::LanguageCode::It => LanguageCode::ItIt,
        voirs_g2p::LanguageCode::Pt => LanguageCode::PtBr,
        voirs_g2p::LanguageCode::Ru => LanguageCode::RuRu,
        voirs_g2p::LanguageCode::Ar => LanguageCode::EnUs,
    }
}

fn convert_phoneme_to_sdk(phoneme: voirs_g2p::Phoneme) -> Phoneme {
    Phoneme {
        symbol: phoneme.symbol.clone(),
        ipa_symbol: phoneme.symbol,
        stress: phoneme.stress,
        syllable_position: convert_syllable_position_to_sdk(phoneme.syllable_position),
        duration_ms: phoneme.duration_ms,
        confidence: phoneme.confidence,
    }
}

fn convert_syllable_position_to_sdk(
    pos: voirs_g2p::SyllablePosition,
) -> voirs_sdk::types::SyllablePosition {
    match pos {
        voirs_g2p::SyllablePosition::Onset => voirs_sdk::types::SyllablePosition::Onset,
        voirs_g2p::SyllablePosition::Nucleus => voirs_sdk::types::SyllablePosition::Nucleus,
        voirs_g2p::SyllablePosition::Coda => voirs_sdk::types::SyllablePosition::Coda,
        voirs_g2p::SyllablePosition::Final => voirs_sdk::types::SyllablePosition::Coda,
        voirs_g2p::SyllablePosition::Standalone => voirs_sdk::types::SyllablePosition::Unknown,
    }
}

/// Bridge for connecting voirs-acoustic types to SDK AcousticModel trait
pub struct AcousticBridge<T> {
    inner: T,
}

impl<T> AcousticBridge<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<T> AcousticModel for AcousticBridge<T>
where
    T: acoustic::traits::AcousticModel + Send + Sync,
{
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        let acoustic_phonemes = convert_phonemes_to_acoustic_batch(phonemes);
        let acoustic_config = config.map(convert_synthesis_config_to_acoustic);

        let acoustic_result = self
            .inner
            .synthesize(&acoustic_phonemes, acoustic_config.as_ref())
            .await
            .map_err(|e| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Acoustic,
                message: format!("Acoustic synthesis error: {e}"),
                source: None,
            })?;

        Ok(convert_mel_spectrogram_to_sdk(acoustic_result))
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        let acoustic_inputs: Vec<Vec<acoustic::Phoneme>> = inputs
            .iter()
            .map(|phonemes| convert_phonemes_to_acoustic_batch(phonemes))
            .collect();

        let acoustic_inputs_refs: Vec<&[acoustic::Phoneme]> =
            acoustic_inputs.iter().map(|v| v.as_slice()).collect();

        let acoustic_configs = configs.map(|configs| {
            configs
                .iter()
                .map(convert_synthesis_config_to_acoustic)
                .collect::<Vec<_>>()
        });

        let acoustic_result = self
            .inner
            .synthesize_batch(&acoustic_inputs_refs, acoustic_configs.as_deref())
            .await
            .map_err(|e| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Acoustic,
                message: format!("Acoustic batch synthesis error: {e}"),
                source: None,
            })?;

        Ok(acoustic_result
            .into_iter()
            .map(convert_mel_spectrogram_to_sdk)
            .collect())
    }

    fn metadata(&self) -> voirs_sdk::traits::AcousticModelMetadata {
        let acoustic_meta = self.inner.metadata();
        voirs_sdk::traits::AcousticModelMetadata {
            name: acoustic_meta.name,
            version: acoustic_meta.version,
            architecture: acoustic_meta.architecture,
            supported_languages: acoustic_meta
                .supported_languages
                .into_iter()
                .map(convert_acoustic_language_code_to_sdk)
                .collect(),
            sample_rate: acoustic_meta.sample_rate,
            mel_channels: acoustic_meta.mel_channels,
            is_multi_speaker: acoustic_meta.is_multi_speaker,
            speaker_count: acoustic_meta.speaker_count,
        }
    }

    fn supports(&self, feature: voirs_sdk::traits::AcousticModelFeature) -> bool {
        let acoustic_feature = convert_acoustic_feature_to_crate(feature);
        self.inner.supports(acoustic_feature)
    }
}

fn convert_phoneme_to_acoustic(phoneme: &Phoneme) -> acoustic::Phoneme {
    acoustic::Phoneme {
        symbol: phoneme.symbol.clone(),
        features: None,
        duration: phoneme.duration_ms.map(|d| d / 1000.0),
    }
}

fn convert_phonemes_to_acoustic_batch(phonemes: &[Phoneme]) -> Vec<acoustic::Phoneme> {
    phonemes.iter().map(convert_phoneme_to_acoustic).collect()
}

fn convert_synthesis_config_to_acoustic(config: &SynthesisConfig) -> acoustic::SynthesisConfig {
    acoustic::SynthesisConfig {
        speed: config.speaking_rate,
        pitch_shift: config.pitch_shift,
        energy: config.volume_gain,
        speaker_id: None,
        seed: config.seed,
        emotion: None,
        voice_style: None,
    }
}

fn convert_mel_spectrogram_to_sdk(mel: acoustic::MelSpectrogram) -> MelSpectrogram {
    MelSpectrogram {
        data: mel.data,
        n_mels: mel.n_mels as u32,
        n_frames: mel.n_frames as u32,
        sample_rate: mel.sample_rate,
        hop_length: mel.hop_length,
    }
}

fn convert_acoustic_language_code_to_sdk(lang: acoustic::LanguageCode) -> LanguageCode {
    match lang {
        acoustic::LanguageCode::EnUs => LanguageCode::EnUs,
        acoustic::LanguageCode::EnGb => LanguageCode::EnGb,
        acoustic::LanguageCode::JaJp => LanguageCode::JaJp,
        acoustic::LanguageCode::ZhCn => LanguageCode::ZhCn,
        acoustic::LanguageCode::KoKr => LanguageCode::KoKr,
        acoustic::LanguageCode::DeDe => LanguageCode::DeDe,
        acoustic::LanguageCode::FrFr => LanguageCode::FrFr,
        acoustic::LanguageCode::EsEs => LanguageCode::EsEs,
        acoustic::LanguageCode::ItIt => LanguageCode::ItIt,
        acoustic::LanguageCode::PtBr => LanguageCode::PtBr,
        acoustic::LanguageCode::PtPt => LanguageCode::PtBr,
        acoustic::LanguageCode::RuRu => LanguageCode::RuRu,
        acoustic::LanguageCode::NlNl => LanguageCode::NlNl,
        acoustic::LanguageCode::PlPl => LanguageCode::EnUs,
        acoustic::LanguageCode::SvSe => LanguageCode::SvSe,
        acoustic::LanguageCode::TrTr => LanguageCode::EnUs,
        acoustic::LanguageCode::ArSa => LanguageCode::EnUs,
        acoustic::LanguageCode::HiIn => LanguageCode::EnUs,
        acoustic::LanguageCode::ViVn => LanguageCode::EnUs,
        acoustic::LanguageCode::ThTh => LanguageCode::EnUs,
        acoustic::LanguageCode::IdId => LanguageCode::EnUs,
        acoustic::LanguageCode::FiFi => LanguageCode::EnUs,
        acoustic::LanguageCode::CsCz => LanguageCode::EnUs,
        acoustic::LanguageCode::ElGr => LanguageCode::EnUs,
        acoustic::LanguageCode::HeIl => LanguageCode::EnUs,
        acoustic::LanguageCode::NoNo => LanguageCode::NoNo,
        acoustic::LanguageCode::DaDk => LanguageCode::DaDk,
        acoustic::LanguageCode::MsMy => LanguageCode::EnUs,
    }
}

fn convert_acoustic_feature_to_crate(
    feature: voirs_sdk::traits::AcousticModelFeature,
) -> acoustic::traits::AcousticModelFeature {
    match feature {
        voirs_sdk::traits::AcousticModelFeature::MultiSpeaker => {
            acoustic::traits::AcousticModelFeature::MultiSpeaker
        }
        voirs_sdk::traits::AcousticModelFeature::EmotionControl => {
            acoustic::traits::AcousticModelFeature::EmotionControl
        }
        voirs_sdk::traits::AcousticModelFeature::StreamingInference => {
            acoustic::traits::AcousticModelFeature::StreamingInference
        }
        voirs_sdk::traits::AcousticModelFeature::StreamingSynthesis => {
            acoustic::traits::AcousticModelFeature::StreamingSynthesis
        }
        voirs_sdk::traits::AcousticModelFeature::BatchProcessing => {
            acoustic::traits::AcousticModelFeature::BatchProcessing
        }
        voirs_sdk::traits::AcousticModelFeature::ProsodyControl => {
            acoustic::traits::AcousticModelFeature::ProsodyControl
        }
        voirs_sdk::traits::AcousticModelFeature::StyleTransfer => {
            acoustic::traits::AcousticModelFeature::StyleTransfer
        }
        voirs_sdk::traits::AcousticModelFeature::GpuAcceleration => {
            acoustic::traits::AcousticModelFeature::GpuAcceleration
        }
        voirs_sdk::traits::AcousticModelFeature::VoiceCloning => {
            acoustic::traits::AcousticModelFeature::VoiceCloning
        }
        voirs_sdk::traits::AcousticModelFeature::RealTimeInference => {
            acoustic::traits::AcousticModelFeature::RealTimeInference
        }
    }
}

/// Bridge for connecting voirs-vocoder types to SDK Vocoder trait
pub struct VocoderBridge<T> {
    inner: T,
}

impl<T> VocoderBridge<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<T> Vocoder for VocoderBridge<T>
where
    T: vocoder::Vocoder + Send + Sync,
{
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        let vocoder_mel = convert_mel_spectrogram_to_vocoder(mel);
        let vocoder_config = config.map(convert_synthesis_config_to_vocoder);

        let vocoder_result = self
            .inner
            .vocode(&vocoder_mel, vocoder_config.as_ref())
            .await
            .map_err(|e| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Vocoder,
                message: format!("Vocoder synthesis error: {e}"),
                source: None,
            })?;

        Ok(convert_audio_buffer_to_sdk(vocoder_result))
    }

    async fn vocode_stream(
        &self,
        mel_stream: Box<dyn futures::Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::StreamExt;

        let config_clone = config.cloned();
        let vocoder_mels = mel_stream.map(|mel| convert_mel_spectrogram_to_vocoder(&mel));
        let vocoder_config = config_clone.map(|c| convert_synthesis_config_to_vocoder(&c));

        let vocoder_stream = self
            .inner
            .vocode_stream(Box::new(vocoder_mels), vocoder_config.as_ref())
            .await
            .map_err(|e| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Vocoder,
                message: format!("Vocoder streaming failed: {e}"),
                source: None,
            })?;

        let audio_stream = vocoder_stream.map(|result| {
            result
                .map(convert_audio_buffer_to_sdk)
                .map_err(|e| VoirsError::ModelError {
                    model_type: voirs_sdk::error::ModelType::Vocoder,
                    message: format!("Vocoder streaming error: {e}"),
                    source: None,
                })
        });

        Ok(Box::new(audio_stream))
    }

    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let vocoder_mels: Vec<vocoder::MelSpectrogram> = mels
            .iter()
            .map(convert_mel_spectrogram_to_vocoder)
            .collect();

        let vocoder_configs = configs.map(|configs| {
            configs
                .iter()
                .map(convert_synthesis_config_to_vocoder)
                .collect::<Vec<_>>()
        });

        let vocoder_result = self
            .inner
            .vocode_batch(&vocoder_mels, vocoder_configs.as_deref())
            .await
            .map_err(|e| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::Vocoder,
                message: format!("Vocoder batch synthesis error: {e}"),
                source: None,
            })?;

        Ok(vocoder_result
            .into_iter()
            .map(convert_audio_buffer_to_sdk)
            .collect())
    }

    fn metadata(&self) -> voirs_sdk::traits::VocoderMetadata {
        let vocoder_meta = self.inner.metadata();
        voirs_sdk::traits::VocoderMetadata {
            name: vocoder_meta.name,
            version: vocoder_meta.version,
            architecture: vocoder_meta.architecture,
            sample_rate: vocoder_meta.sample_rate,
            mel_channels: vocoder_meta.mel_channels,
            latency_ms: vocoder_meta.latency_ms,
            quality_score: vocoder_meta.quality_score,
        }
    }

    fn supports(&self, feature: voirs_sdk::traits::VocoderFeature) -> bool {
        let vocoder_feature = convert_vocoder_feature_to_crate(feature);
        self.inner.supports(vocoder_feature)
    }
}

fn convert_mel_spectrogram_to_vocoder(mel: &MelSpectrogram) -> vocoder::MelSpectrogram {
    let mut data = Vec::with_capacity(mel.data.len());
    for row in &mel.data {
        data.push(row.clone());
    }
    vocoder::MelSpectrogram {
        data,
        n_mels: mel.n_mels as usize,
        n_frames: mel.n_frames as usize,
        sample_rate: mel.sample_rate,
        hop_length: mel.hop_length,
    }
}

fn convert_synthesis_config_to_vocoder(config: &SynthesisConfig) -> vocoder::SynthesisConfig {
    vocoder::SynthesisConfig {
        speed: config.speaking_rate,
        pitch_shift: config.pitch_shift,
        energy: config.volume_gain,
        speaker_id: None,
        seed: config.seed,
    }
}

fn convert_audio_buffer_to_sdk(audio: vocoder::AudioBuffer) -> AudioBuffer {
    AudioBuffer::new(
        audio.samples().to_vec(),
        audio.sample_rate(),
        audio.channels(),
    )
}

fn convert_vocoder_feature_to_crate(
    feature: voirs_sdk::traits::VocoderFeature,
) -> vocoder::VocoderFeature {
    match feature {
        voirs_sdk::traits::VocoderFeature::StreamingInference => {
            vocoder::VocoderFeature::StreamingInference
        }
        voirs_sdk::traits::VocoderFeature::BatchProcessing => {
            vocoder::VocoderFeature::BatchProcessing
        }
        voirs_sdk::traits::VocoderFeature::GpuAcceleration => {
            vocoder::VocoderFeature::GpuAcceleration
        }
        voirs_sdk::traits::VocoderFeature::RealtimeProcessing => {
            vocoder::VocoderFeature::RealtimeProcessing
        }
        voirs_sdk::traits::VocoderFeature::MultiSampleRate => vocoder::VocoderFeature::HighQuality,
        voirs_sdk::traits::VocoderFeature::EnhancementFilters => {
            vocoder::VocoderFeature::HighQuality
        }
    }
}
