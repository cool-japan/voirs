//! Real-time synthesis engine for interactive mode
//!
//! Handles:
//! - Real-time text-to-speech synthesis
//! - Immediate audio playback
//! - Voice switching during session
//! - Audio parameter adjustments

use crate::audio::playback::{AudioData, AudioPlayer, PlaybackConfig};
use crate::error::{Result, VoirsCliError};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Real-time synthesis engine
pub struct SynthesisEngine {
    /// VoiRS SDK pipeline for synthesis
    pipeline: Option<Arc<RwLock<voirs_sdk::VoirsPipeline>>>,

    /// Audio player for immediate playback
    audio_player: AudioPlayer,

    /// Current synthesis parameters
    current_speed: f32,
    current_pitch: f32,
    current_volume: f32,

    /// Available voices cache
    available_voices: Vec<String>,

    /// Current voice
    current_voice: Option<String>,
}

impl SynthesisEngine {
    /// Create a new synthesis engine
    pub async fn new() -> Result<Self> {
        // Initialize audio player with default config
        let config = PlaybackConfig::default();
        let audio_player = AudioPlayer::new(config).map_err(|e| {
            VoirsCliError::AudioError(format!("Failed to initialize audio player: {}", e))
        })?;

        // Load available voices
        let available_voices = Self::load_available_voices().await?;

        Ok(Self {
            pipeline: None,
            audio_player,
            current_speed: 1.0,
            current_pitch: 0.0,
            current_volume: 1.0,
            available_voices,
            current_voice: None,
        })
    }

    /// Load available voices from the system's real voice registry.
    ///
    /// Queries `voirs_sdk::voice::VoiceRegistry` -- the same registry
    /// `DefaultVoiceManager` resolves voices against when the SDK actually
    /// builds a pipeline (see `create_pipeline` below) -- rather than an
    /// independent hardcoded list. A hardcoded list here previously drifted
    /// from the SDK's real default voice IDs, so selecting any of the
    /// offered voices failed with "Voice not found" the moment
    /// `create_pipeline` tried to build a real pipeline for it.
    async fn load_available_voices() -> Result<Vec<String>> {
        let registry = voirs_sdk::voice::VoiceRegistry::new();
        let mut voices: Vec<String> = registry
            .list_voices()
            .into_iter()
            .map(|voice| voice.id.clone())
            .collect();
        voices.sort();
        Ok(voices)
    }

    /// Get list of available voices
    pub async fn list_voices(&self) -> Result<Vec<String>> {
        Ok(self.available_voices.clone())
    }

    /// Set the current voice
    pub async fn set_voice(&mut self, voice: &str) -> Result<()> {
        // Validate voice exists
        if !self.available_voices.contains(&voice.to_string()) {
            return Err(VoirsCliError::VoiceError(format!(
                "Voice '{}' not found. Available voices: {}",
                voice,
                self.available_voices.join(", ")
            )));
        }

        // Initialize pipeline if needed
        if self.pipeline.is_none() {
            self.pipeline = Some(Arc::new(RwLock::new(self.create_pipeline(voice).await?)));
        } else {
            // Switch voice in existing pipeline
            if let Some(ref pipeline) = self.pipeline {
                let mut pipeline_guard = pipeline.write().await;
                pipeline_guard.set_voice(voice).await.map_err(|e| {
                    VoirsCliError::SynthesisError(format!("Failed to set voice: {}", e))
                })?;
            }
        }

        self.current_voice = Some(voice.to_string());
        println!("✓ Voice set to: {}", voice);

        Ok(())
    }

    /// Create a new VoiRS pipeline
    async fn create_pipeline(&self, voice: &str) -> Result<voirs_sdk::VoirsPipeline> {
        // Create pipeline using VoiRS SDK builder
        let pipeline = voirs_sdk::VoirsPipeline::builder()
            .with_quality(voirs_sdk::QualityLevel::High)
            .with_voice(voice)
            .build()
            .await
            .map_err(|e| {
                VoirsCliError::SynthesisError(format!(
                    "Failed to create VoiRS pipeline for voice '{}': {}",
                    voice, e
                ))
            })?;

        Ok(pipeline)
    }

    /// Synthesize text to audio
    pub async fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if self.pipeline.is_none() {
            return Err(VoirsCliError::SynthesisError(
                "No voice selected. Use ':voice <voice_name>' to set a voice.".to_string(),
            ));
        }

        // Use the VoiRS pipeline for real synthesis
        if let Some(pipeline) = &self.pipeline {
            // Build synthesis configuration
            let mut config = voirs_sdk::types::SynthesisConfig::default();
            config.speaking_rate = self.current_speed;
            config.pitch_shift = self.current_pitch;
            config.volume_gain = self.current_volume;

            // Perform synthesis
            let pipeline_guard = pipeline.read().await;
            match pipeline_guard.synthesize_with_config(text, &config).await {
                Ok(audio_buffer) => {
                    // Convert AudioBuffer to Vec<f32>
                    Ok(audio_buffer.samples().to_vec())
                }
                Err(e) => {
                    // Propagate the real synthesis failure to the caller instead of
                    // silently substituting fabricated audio (e.g. a sine-wave beep).
                    // The REPL loop (see `shell.rs::run`) catches `Err` and prints it
                    // to the user via `print_error`, so this is visible, not silent.
                    tracing::warn!("Synthesis failed: {}", e);
                    Err(VoirsCliError::SynthesisError(format!(
                        "Failed to synthesize \"{}\": {}",
                        text, e
                    )))
                }
            }
        } else {
            Err(VoirsCliError::SynthesisError(
                "No pipeline available for synthesis".to_string(),
            ))
        }
    }

    /// Play audio data
    pub async fn play_audio(&mut self, audio_data: &[f32]) -> Result<()> {
        // Convert f32 samples to i16 for AudioData
        let samples_i16: Vec<i16> = audio_data
            .iter()
            .map(|&sample| (sample * i16::MAX as f32) as i16)
            .collect();

        let audio_data = AudioData {
            samples: samples_i16,
            sample_rate: 22050,
            channels: 1,
        };

        self.audio_player
            .play(&audio_data)
            .await
            .map_err(|e| VoirsCliError::AudioError(format!("Failed to play audio: {}", e)))?;

        Ok(())
    }

    /// Set synthesis speed
    pub async fn set_speed(&mut self, speed: f32) -> Result<()> {
        self.current_speed = speed.clamp(0.1, 3.0);

        // Apply to pipeline if available
        if let Some(ref pipeline) = self.pipeline {
            // In a real implementation, this would configure the pipeline
            println!("✓ Speed set to: {:.1}x", self.current_speed);
        }

        Ok(())
    }

    /// Set synthesis pitch
    pub async fn set_pitch(&mut self, pitch: f32) -> Result<()> {
        self.current_pitch = pitch.clamp(-12.0, 12.0);

        // Apply to pipeline if available
        if let Some(ref pipeline) = self.pipeline {
            // In a real implementation, this would configure the pipeline
            println!("✓ Pitch set to: {:.1} semitones", self.current_pitch);
        }

        Ok(())
    }

    /// Set synthesis volume
    pub async fn set_volume(&mut self, volume: f32) -> Result<()> {
        self.current_volume = volume.clamp(0.0, 2.0);

        // Apply to audio player
        self.audio_player
            .set_volume(self.current_volume)
            .map_err(|e| VoirsCliError::AudioError(format!("Failed to set volume: {}", e)))?;

        println!("✓ Volume set to: {:.1}", self.current_volume);

        Ok(())
    }

    /// Get current synthesis parameters
    pub fn current_params(&self) -> (f32, f32, f32) {
        (self.current_speed, self.current_pitch, self.current_volume)
    }

    /// Get current voice
    pub fn current_voice(&self) -> Option<&str> {
        self.current_voice.as_deref()
    }

    /// Check if synthesis engine is ready
    pub fn is_ready(&self) -> bool {
        self.pipeline.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_never_reintroduces_the_sine_wave_fallback() {
        // Regression test for the fabricated-audio finding: on a real
        // synthesis failure, `synthesize()` must propagate a real `Err`,
        // never silently substitute a 440Hz sine-wave beep as if it were
        // the requested speech. Guard against the exact fabrication
        // reappearing.
        //
        // Only the *production* code above `#[cfg(test)]` is scanned, so
        // this self-inspecting check never trips over its own assertion
        // strings (which would otherwise always "find" themselves).
        let full_source = include_str!("synthesis.rs");
        let production_code = full_source
            .split("#[cfg(test)]")
            .next()
            .expect("this file has a #[cfg(test)] section");

        assert!(
            !production_code.contains("440.0"),
            "the fabricated 440Hz sine-wave fallback must not be reintroduced"
        );
        assert!(
            !production_code
                .to_lowercase()
                .contains("fallback to simple sine"),
            "the fabricated audio fallback must not be reintroduced"
        );
        // The only success return path for `synthesize()`'s body must be
        // built from the real pipeline's own output -- never `Ok(samples)`
        // constructed from anything else (e.g. a generated waveform).
        assert_eq!(
            production_code
                .matches("Ok(audio_buffer.samples().to_vec())")
                .count(),
            1,
            "synthesize() must have exactly one success path: the real pipeline's own output"
        );
    }

    #[tokio::test]
    async fn synthesize_without_a_selected_voice_returns_a_real_error() {
        // The one failure path reachable without needing real model
        // weights: no voice has been selected yet, so `pipeline` is
        // `None`. This must return `Err`, never `Ok` with placeholder
        // samples.
        let engine = match SynthesisEngine::new().await {
            Ok(engine) => engine,
            Err(_) => {
                // No default audio output device in this environment
                // (e.g. headless CI) -- `AudioPlayer::new()` fails before
                // synthesis logic is even reachable, so there is nothing
                // further to exercise here.
                return;
            }
        };
        assert!(!engine.is_ready(), "no voice has been selected yet");

        let result = engine.synthesize("hello").await;
        assert!(
            result.is_err(),
            "synthesize() with no voice selected must return Err, not fabricated audio"
        );
    }
}
