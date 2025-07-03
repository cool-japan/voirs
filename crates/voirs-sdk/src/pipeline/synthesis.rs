//! Synthesis orchestration and processing.

use crate::{
    audio::AudioBuffer,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, SynthesisConfig},
    VoirsError,
};
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Synthesis orchestrator
#[derive(Clone)]
pub struct SynthesisOrchestrator {
    g2p: Arc<dyn G2p>,
    acoustic: Arc<dyn AcousticModel>,
    vocoder: Arc<dyn Vocoder>,
}

impl SynthesisOrchestrator {
    /// Create new synthesis orchestrator
    pub fn new(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
    ) -> Self {
        Self {
            g2p,
            acoustic,
            vocoder,
        }
    }

    /// Synthesize text to audio with full pipeline
    pub async fn synthesize(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        let start_time = Instant::now();
        info!("Starting synthesis for text: {}", text);

        // Monitor memory usage
        let memory_monitor = MemoryMonitor::new();

        // Step 1: Text to phonemes (G2P)
        let phonemes = self.text_to_phonemes(text, config).await?;
        memory_monitor.check_memory_usage("after G2P")?;

        // Step 2: Phonemes to mel spectrogram (Acoustic Model)
        let mel = self.phonemes_to_mel(&phonemes, config).await?;
        memory_monitor.check_memory_usage("after acoustic model")?;

        // Step 3: Mel spectrogram to audio (Vocoder)
        let mut audio = self.mel_to_audio(&mel, config).await?;
        memory_monitor.check_memory_usage("after vocoder")?;

        // Step 4: Apply post-processing
        self.apply_post_processing(&mut audio, config).await?;

        let duration = start_time.elapsed();
        info!(
            "Synthesis complete: {:.2}s audio generated in {:.2}s",
            audio.duration(),
            duration.as_secs_f64()
        );

        Ok(audio)
    }

    /// Synthesize SSML markup
    pub async fn synthesize_ssml(
        &self,
        ssml: &str,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        info!("Synthesizing SSML markup");

        // Parse SSML to extract text and processing instructions
        let parsed = self.parse_ssml(ssml)?;
        
        // For now, just synthesize the text content
        // TODO: Implement proper SSML processing
        warn!("SSML synthesis not yet fully implemented, processing text content only");
        self.synthesize(&parsed.text, config).await
    }

    /// Stream synthesis for long texts
    pub async fn synthesize_stream(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<impl futures::Stream<Item = Result<AudioBuffer>>> {
        info!("Starting streaming synthesis for long text");

        // Split text into chunks for streaming
        let chunks = self.split_text_for_streaming(text, config)?;
        
        // Create streaming pipeline
        let g2p = Arc::clone(&self.g2p);
        let acoustic = Arc::clone(&self.acoustic);
        let vocoder = Arc::clone(&self.vocoder);
        let config = config.clone();

        let stream = futures::stream::iter(chunks)
            .map(move |chunk| {
                let g2p = Arc::clone(&g2p);
                let acoustic = Arc::clone(&acoustic);
                let vocoder = Arc::clone(&vocoder);
                let config = config.clone();
                
                async move {
                    let orchestrator = SynthesisOrchestrator::new(g2p, acoustic, vocoder);
                    orchestrator.synthesize(&chunk, &config).await
                }
            })
            .buffer_unordered(4); // Process up to 4 chunks concurrently

        Ok(stream)
    }

    /// Convert text to phonemes
    async fn text_to_phonemes(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<Vec<crate::types::Phoneme>> {
        let start_time = Instant::now();
        debug!("Converting text to phonemes");

        let phonemes = self.g2p.to_phonemes(text, Some(config.language))
            .await
            .map_err(|e| VoirsError::synthesis_failed(text, e))?;

        let duration = start_time.elapsed();
        debug!(
            "G2P conversion complete: {} phonemes in {:.2}ms",
            phonemes.len(),
            duration.as_millis()
        );

        Ok(phonemes)
    }

    /// Convert phonemes to mel spectrogram
    async fn phonemes_to_mel(
        &self,
        phonemes: &[crate::types::Phoneme],
        config: &SynthesisConfig,
    ) -> Result<crate::types::MelSpectrogram> {
        let start_time = Instant::now();
        debug!("Converting phonemes to mel spectrogram");

        let mel = self.acoustic.synthesize(phonemes, Some(config))
            .await
            .map_err(|e| VoirsError::SynthesisFailed {
                text: format!("{} phonemes", phonemes.len()),
                text_length: phonemes.len(),
                stage: crate::error::types::SynthesisStage::AcousticModeling,
                cause: e.into(),
            })?;

        let duration = start_time.elapsed();
        debug!(
            "Acoustic model synthesis complete: {}x{} mel in {:.2}ms",
            mel.n_mels,
            mel.n_frames,
            duration.as_millis()
        );

        Ok(mel)
    }

    /// Convert mel spectrogram to audio
    async fn mel_to_audio(
        &self,
        mel: &crate::types::MelSpectrogram,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        let start_time = Instant::now();
        debug!("Converting mel spectrogram to audio");

        let audio = self.vocoder.vocode(mel, Some(config))
            .await
            .map_err(|e| VoirsError::SynthesisFailed {
                text: format!("{}x{} mel", mel.n_mels, mel.n_frames),
                text_length: mel.n_frames as usize,
                stage: crate::error::types::SynthesisStage::Vocoding,
                cause: e.into(),
            })?;

        let duration = start_time.elapsed();
        debug!(
            "Vocoder synthesis complete: {:.2}s audio in {:.2}ms",
            audio.duration(),
            duration.as_millis()
        );

        Ok(audio)
    }

    /// Apply post-processing to audio
    async fn apply_post_processing(
        &self,
        audio: &mut AudioBuffer,
        config: &SynthesisConfig,
    ) -> Result<()> {
        debug!("Applying post-processing");

        // Apply volume gain
        if config.volume_gain != 0.0 {
            audio.apply_gain(config.volume_gain)?;
        }

        // Apply enhancement if enabled
        if config.enable_enhancement {
            self.apply_enhancement(audio, config).await?;
        }

        // Apply effects if specified
        if !config.effects.is_empty() {
            self.apply_effects(audio, config).await?;
        }

        Ok(())
    }

    /// Apply audio enhancement
    async fn apply_enhancement(
        &self,
        audio: &mut AudioBuffer,
        _config: &SynthesisConfig,
    ) -> Result<()> {
        debug!("Applying audio enhancement");

        // TODO: Implement proper audio enhancement
        // For now, just apply basic normalization
        audio.normalize(0.95)?;

        Ok(())
    }

    /// Apply audio effects
    async fn apply_effects(
        &self,
        audio: &mut AudioBuffer,
        config: &SynthesisConfig,
    ) -> Result<()> {
        debug!("Applying audio effects: {:?}", config.effects);

        // TODO: Implement audio effects processing
        // For now, just log the effects that would be applied
        for effect in &config.effects {
            debug!("Would apply effect: {:?}", effect);
        }

        Ok(())
    }

    /// Parse SSML markup
    fn parse_ssml(&self, ssml: &str) -> Result<SsmlParseResult> {
        // TODO: Implement proper SSML parsing
        // For now, just strip tags and extract text
        let text = self.strip_ssml_tags(ssml);
        Ok(SsmlParseResult {
            text,
            instructions: Vec::new(),
        })
    }

    /// Strip SSML tags (simple implementation)
    fn strip_ssml_tags(&self, ssml: &str) -> String {
        ssml.chars()
            .fold((String::new(), false), |(mut result, in_tag), ch| {
                match ch {
                    '<' => (result, true),
                    '>' => (result, false),
                    _ if !in_tag => {
                        result.push(ch);
                        (result, in_tag)
                    }
                    _ => (result, in_tag),
                }
            })
            .0
            .trim()
            .to_string()
    }

    /// Split text into chunks for streaming
    fn split_text_for_streaming(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<Vec<String>> {
        let chunk_size = config.streaming_chunk_size.unwrap_or(100);
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let chunks: Vec<String> = words
            .chunks(chunk_size)
            .map(|chunk| chunk.join(" "))
            .collect();

        debug!("Split text into {} chunks", chunks.len());
        Ok(chunks)
    }
}

/// SSML parsing result
#[derive(Debug)]
struct SsmlParseResult {
    text: String,
    instructions: Vec<SsmlInstruction>,
}

/// SSML instruction
#[derive(Debug)]
struct SsmlInstruction {
    tag: String,
    attributes: std::collections::HashMap<String, String>,
    position: usize,
}

/// Memory usage monitor
struct MemoryMonitor {
    start_memory: usize,
}

impl MemoryMonitor {
    fn new() -> Self {
        Self {
            start_memory: Self::get_memory_usage(),
        }
    }

    fn check_memory_usage(&self, stage: &str) -> Result<()> {
        let current_memory = Self::get_memory_usage();
        let memory_increase = current_memory.saturating_sub(self.start_memory);
        
        debug!("Memory usage {} - Current: {} bytes (+{})", stage, current_memory, memory_increase);
        
        // Check if memory usage is getting too high
        if memory_increase > 1_000_000_000 { // 1GB
            warn!("High memory usage detected: {} bytes", memory_increase);
        }
        
        Ok(())
    }

    fn get_memory_usage() -> usize {
        // TODO: Implement actual memory usage tracking
        // For now, return a dummy value
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synthesis_orchestrator() {
        let g2p = Arc::new(crate::pipeline::DummyG2p::new());
        let acoustic = Arc::new(crate::pipeline::DummyAcoustic::new());
        let vocoder = Arc::new(crate::pipeline::DummyVocoder::new());
        
        let orchestrator = SynthesisOrchestrator::new(g2p, acoustic, vocoder);
        let config = SynthesisConfig::default();
        
        let result = orchestrator.synthesize("Hello, world!", &config).await;
        assert!(result.is_ok());
        
        let audio = result.unwrap();
        assert!(audio.duration() > 0.0);
    }

    #[tokio::test]
    async fn test_ssml_synthesis() {
        let g2p = Arc::new(crate::pipeline::DummyG2p::new());
        let acoustic = Arc::new(crate::pipeline::DummyAcoustic::new());
        let vocoder = Arc::new(crate::pipeline::DummyVocoder::new());
        
        let orchestrator = SynthesisOrchestrator::new(g2p, acoustic, vocoder);
        let config = SynthesisConfig::default();
        
        let ssml = "<speak>Hello, <break time='1s'/> world!</speak>";
        let result = orchestrator.synthesize_ssml(ssml, &config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_streaming_synthesis() {
        let g2p = Arc::new(crate::pipeline::DummyG2p::new());
        let acoustic = Arc::new(crate::pipeline::DummyAcoustic::new());
        let vocoder = Arc::new(crate::pipeline::DummyVocoder::new());
        
        let orchestrator = SynthesisOrchestrator::new(g2p, acoustic, vocoder);
        let config = SynthesisConfig::default();
        
        let text = "This is a long text that will be split into chunks for streaming synthesis.";
        let stream = orchestrator.synthesize_stream(text, &config).await;
        assert!(stream.is_ok());
    }
}