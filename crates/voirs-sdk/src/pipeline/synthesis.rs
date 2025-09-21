//! Synthesis orchestration and processing.

use crate::{
    audio::AudioBuffer,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::SynthesisConfig,
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
    test_mode: bool,
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
            test_mode: false,
        }
    }

    /// Create new synthesis orchestrator with test mode
    pub fn with_test_mode(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        test_mode: bool,
    ) -> Self {
        Self {
            g2p,
            acoustic,
            vocoder,
            test_mode,
        }
    }

    /// Synthesize text to audio with full pipeline
    pub async fn synthesize(&self, text: &str, config: &SynthesisConfig) -> Result<AudioBuffer> {
        let start_time = Instant::now();
        info!("Starting synthesis for text: {}", text);

        // Monitor memory usage
        let memory_monitor = MemoryMonitor::new(self.test_mode);

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
        info!("Synthesizing SSML markup with enhanced processing");

        // Parse SSML to extract text and processing instructions
        let parsed = self.parse_ssml(ssml)?;

        // Process SSML instructions to modify synthesis config
        let enhanced_config = self.apply_ssml_instructions(&parsed.instructions, config)?;

        // Synthesize text with enhanced configuration
        let mut audio = self.synthesize(&parsed.text, &enhanced_config).await?;

        // Apply SSML-specific post-processing
        self.apply_ssml_post_processing(&mut audio, &parsed.instructions, config)
            .await?;

        info!(
            "SSML synthesis complete with {} instructions processed",
            parsed.instructions.len()
        );

        Ok(audio)
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
        let test_mode = self.test_mode;
        let config = config.clone();

        let stream = futures::stream::iter(chunks)
            .map(move |chunk| {
                let g2p = Arc::clone(&g2p);
                let acoustic = Arc::clone(&acoustic);
                let vocoder = Arc::clone(&vocoder);
                let config = config.clone();

                async move {
                    let orchestrator =
                        SynthesisOrchestrator::with_test_mode(g2p, acoustic, vocoder, test_mode);
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

        let phonemes = self
            .g2p
            .to_phonemes(text, Some(config.language))
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

        let mel = self
            .acoustic
            .synthesize(phonemes, Some(config))
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

        let audio = self.vocoder.vocode(mel, Some(config)).await.map_err(|e| {
            VoirsError::SynthesisFailed {
                text: format!("{}x{} mel", mel.n_mels, mel.n_frames),
                text_length: mel.n_frames as usize,
                stage: crate::error::types::SynthesisStage::Vocoding,
                cause: e.into(),
            }
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
        config: &SynthesisConfig,
    ) -> Result<()> {
        debug!("Applying audio enhancement");

        if !config.enable_enhancement {
            return Ok(());
        }

        // Apply normalization
        audio.normalize(0.95)?;

        // Apply volume gain
        if config.volume_gain != 0.0 {
            audio.apply_gain(config.volume_gain)?;
        }

        // Apply speaking rate adjustment (time-stretching)
        if config.speaking_rate != 1.0 {
            self.apply_time_stretch(audio, config.speaking_rate)?;
        }

        // Apply pitch shifting
        if config.pitch_shift != 0.0 {
            self.apply_pitch_shift(audio, config.pitch_shift)?;
        }

        debug!("Audio enhancement complete");
        Ok(())
    }

    /// Apply audio effects
    async fn apply_effects(&self, audio: &mut AudioBuffer, config: &SynthesisConfig) -> Result<()> {
        debug!("Applying audio effects: {:?}", config.effects);

        use crate::types::AudioEffect;

        for effect in &config.effects {
            match effect {
                AudioEffect::Reverb {
                    room_size,
                    damping,
                    wet_level,
                } => {
                    debug!(
                        "Applying reverb: room_size={}, damping={}, wet_level={}",
                        room_size, damping, wet_level
                    );
                    self.apply_reverb(audio, *room_size, *damping, *wet_level)?;
                }
                AudioEffect::Delay {
                    delay_time,
                    feedback,
                    wet_level,
                } => {
                    debug!(
                        "Applying delay: delay_time={}, feedback={}, wet_level={}",
                        delay_time, feedback, wet_level
                    );
                    self.apply_delay(audio, *delay_time, *feedback, *wet_level)?;
                }
                AudioEffect::Equalizer {
                    low_gain,
                    mid_gain,
                    high_gain,
                } => {
                    debug!(
                        "Applying equalizer: low_gain={}, mid_gain={}, high_gain={}",
                        low_gain, mid_gain, high_gain
                    );
                    self.apply_equalizer(audio, *low_gain, *mid_gain, *high_gain)?;
                }
                AudioEffect::Compressor {
                    threshold,
                    ratio,
                    attack,
                    release,
                } => {
                    debug!(
                        "Applying compressor: threshold={}, ratio={}, attack={}, release={}",
                        threshold, ratio, attack, release
                    );
                    self.apply_compressor(audio, *threshold, *ratio, *attack, *release)?;
                }
            }
        }

        debug!("Audio effects processing complete");
        Ok(())
    }

    /// Apply time stretching for speaking rate adjustment
    fn apply_time_stretch(&self, audio: &mut AudioBuffer, speaking_rate: f32) -> Result<()> {
        debug!("Applying time stretch with rate: {}", speaking_rate);

        // Use AudioBuffer's existing time_stretch method
        if speaking_rate != 1.0 {
            let stretched = audio.time_stretch(1.0 / speaking_rate)?;
            *audio = stretched;
        }

        Ok(())
    }

    /// Apply pitch shifting
    fn apply_pitch_shift(&self, audio: &mut AudioBuffer, pitch_shift: f32) -> Result<()> {
        debug!("Applying pitch shift: {} semitones", pitch_shift);

        // Use AudioBuffer's existing pitch_shift method
        if pitch_shift != 0.0 {
            let shifted = audio.pitch_shift(pitch_shift)?;
            *audio = shifted;
        }

        Ok(())
    }

    /// Apply reverb effect
    fn apply_reverb(
        &self,
        audio: &mut AudioBuffer,
        room_size: f32,
        damping: f32,
        wet_level: f32,
    ) -> Result<()> {
        debug!(
            "Applying reverb: room_size={}, damping={}, wet_level={}",
            room_size, damping, wet_level
        );

        // Use AudioBuffer's existing reverb method
        audio.reverb(room_size, damping, wet_level)?;

        Ok(())
    }

    /// Apply delay effect
    fn apply_delay(
        &self,
        audio: &mut AudioBuffer,
        delay_time: f32,
        feedback: f32,
        wet_level: f32,
    ) -> Result<()> {
        debug!(
            "Applying delay: delay_time={}, feedback={}, wet_level={}",
            delay_time, feedback, wet_level
        );

        let sample_rate = audio.sample_rate() as f32;
        let delay_samples = (delay_time * sample_rate) as usize;
        let samples = audio.samples_mut();

        if delay_samples > 0 && delay_samples < samples.len() {
            let mut delayed_samples = vec![0.0; samples.len()];

            for i in delay_samples..samples.len() {
                let delayed_value = samples[i - delay_samples] * feedback;
                delayed_samples[i] = delayed_value;

                // Add feedback
                if i + delay_samples < samples.len() {
                    delayed_samples[i + delay_samples] += delayed_value * feedback;
                }
            }

            // Mix dry and wet signals
            for i in 0..samples.len() {
                samples[i] = samples[i] * (1.0 - wet_level) + delayed_samples[i] * wet_level;
            }
        }

        Ok(())
    }

    /// Apply equalizer effect
    fn apply_equalizer(
        &self,
        audio: &mut AudioBuffer,
        low_gain: f32,
        mid_gain: f32,
        high_gain: f32,
    ) -> Result<()> {
        debug!(
            "Applying equalizer: low_gain={}, mid_gain={}, high_gain={}",
            low_gain, mid_gain, high_gain
        );

        // Basic 3-band EQ implementation
        // For production use, consider using proper filter coefficients
        let samples = audio.samples_mut();

        // Simple gain adjustment based on frequency bands
        // This is a simplified implementation - real EQ would use proper filters
        for sample in samples.iter_mut() {
            // Apply frequency-dependent gain (simplified)
            let low_component = *sample * 0.3 * low_gain;
            let mid_component = *sample * 0.4 * mid_gain;
            let high_component = *sample * 0.3 * high_gain;

            *sample = low_component + mid_component + high_component;
        }

        Ok(())
    }

    /// Apply compressor effect
    fn apply_compressor(
        &self,
        audio: &mut AudioBuffer,
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
    ) -> Result<()> {
        debug!(
            "Applying compressor: threshold={}, ratio={}, attack={}, release={}",
            threshold, ratio, attack, release
        );

        let sample_rate = audio.sample_rate() as f32;
        let samples = audio.samples_mut();
        let attack_coeff = (-1.0 / (attack * sample_rate)).exp();
        let release_coeff = (-1.0 / (release * sample_rate)).exp();

        let mut envelope = 0.0;

        for sample in samples.iter_mut() {
            let input_level = sample.abs();

            // Envelope follower
            if input_level > envelope {
                envelope += (input_level - envelope) * (1.0 - attack_coeff);
            } else {
                envelope += (input_level - envelope) * (1.0 - release_coeff);
            }

            // Compression
            if envelope > threshold {
                let excess = envelope - threshold;
                let compressed_excess = excess / ratio;
                let gain_reduction = (threshold + compressed_excess) / envelope;
                *sample *= gain_reduction;
            }
        }

        Ok(())
    }

    /// Apply SSML instructions to modify synthesis configuration
    fn apply_ssml_instructions(
        &self,
        instructions: &[SsmlInstruction],
        base_config: &SynthesisConfig,
    ) -> Result<SynthesisConfig> {
        let mut config = base_config.clone();

        for instruction in instructions {
            match instruction.tag.as_str() {
                "prosody" => {
                    self.apply_prosody_instruction(&mut config, &instruction.attributes)?;
                }
                "voice" => {
                    self.apply_voice_instruction(&mut config, &instruction.attributes)?;
                }
                "emphasis" => {
                    self.apply_emphasis_instruction(&mut config, &instruction.attributes)?;
                }
                "audio" => {
                    // Audio tags will be handled in post-processing
                    debug!("Audio tag found, will be processed in post-processing");
                }
                "break" => {
                    // Break tags will be handled in post-processing
                    debug!("Break tag found, will be processed in post-processing");
                }
                _ => {
                    debug!("Unsupported SSML tag: {}", instruction.tag);
                }
            }
        }

        Ok(config)
    }

    /// Apply SSML-specific post-processing to audio
    async fn apply_ssml_post_processing(
        &self,
        audio: &mut AudioBuffer,
        instructions: &[SsmlInstruction],
        config: &SynthesisConfig,
    ) -> Result<()> {
        for instruction in instructions {
            match instruction.tag.as_str() {
                "break" => {
                    self.apply_break_instruction(audio, &instruction.attributes, config)
                        .await?;
                }
                "audio" => {
                    // For audio insertion, we would need to load and mix external audio
                    warn!("Audio tag insertion not yet implemented");
                }
                _ => {
                    // Other tags were handled in config modification
                }
            }
        }

        Ok(())
    }

    /// Apply prosody instruction to modify synthesis parameters
    fn apply_prosody_instruction(
        &self,
        config: &mut SynthesisConfig,
        attributes: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        if let Some(rate) = attributes.get("rate") {
            match rate.as_str() {
                "x-slow" => config.speaking_rate = 0.5,
                "slow" => config.speaking_rate = 0.75,
                "medium" => config.speaking_rate = 1.0,
                "fast" => config.speaking_rate = 1.25,
                "x-fast" => config.speaking_rate = 1.5,
                _ => {
                    // Try parsing as percentage or decimal
                    if rate.ends_with('%') {
                        if let Ok(percent) = rate.trim_end_matches('%').parse::<f32>() {
                            config.speaking_rate = percent / 100.0;
                        }
                    } else if let Ok(value) = rate.parse::<f32>() {
                        config.speaking_rate = value;
                    }
                }
            }
        }

        if let Some(pitch) = attributes.get("pitch") {
            match pitch.as_str() {
                "x-low" => config.pitch_shift = -12.0,
                "low" => config.pitch_shift = -6.0,
                "medium" => config.pitch_shift = 0.0,
                "high" => config.pitch_shift = 6.0,
                "x-high" => config.pitch_shift = 12.0,
                _ => {
                    // Try parsing as Hz or semitones
                    if pitch.ends_with("Hz") {
                        // Convert Hz to semitones (relative to some base frequency)
                        warn!("Hz pitch specification not fully implemented");
                    } else if let Ok(value) = pitch.parse::<f32>() {
                        config.pitch_shift = value;
                    }
                }
            }
        }

        if let Some(volume) = attributes.get("volume") {
            match volume.as_str() {
                "silent" => config.volume_gain = -60.0,
                "x-soft" => config.volume_gain = -20.0,
                "soft" => config.volume_gain = -10.0,
                "medium" => config.volume_gain = 0.0,
                "loud" => config.volume_gain = 6.0,
                "x-loud" => config.volume_gain = 12.0,
                _ => {
                    if volume.ends_with("dB") {
                        if let Ok(db) = volume.trim_end_matches("dB").parse::<f32>() {
                            config.volume_gain = db;
                        }
                    } else if let Ok(value) = volume.parse::<f32>() {
                        config.volume_gain = 20.0 * value.log10();
                    }
                }
            }
        }

        debug!(
            "Applied prosody: rate={}, pitch={}, volume={}",
            config.speaking_rate, config.pitch_shift, config.volume_gain
        );

        Ok(())
    }

    /// Apply voice instruction to change voice settings
    fn apply_voice_instruction(
        &self,
        _config: &mut SynthesisConfig,
        attributes: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        if let Some(name) = attributes.get("name") {
            // Note: Voice switching would need to be handled at pipeline level
            debug!(
                "Voice change requested to: {} (requires pipeline-level handling)",
                name
            );
        }

        if let Some(gender) = attributes.get("gender") {
            debug!(
                "Voice gender requested: {} (requires voice switching)",
                gender
            );
        }

        if let Some(age) = attributes.get("age") {
            debug!("Voice age requested: {} (requires voice switching)", age);
        }

        Ok(())
    }

    /// Apply emphasis instruction
    fn apply_emphasis_instruction(
        &self,
        config: &mut SynthesisConfig,
        attributes: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let level = attributes
            .get("level")
            .map(|s| s.as_str())
            .unwrap_or("moderate");

        match level {
            "strong" => {
                config.volume_gain += 3.0;
                config.speaking_rate *= 0.9; // Slightly slower for emphasis
            }
            "moderate" => {
                config.volume_gain += 1.5;
                config.speaking_rate *= 0.95;
            }
            "reduced" => {
                config.volume_gain -= 1.5;
                config.speaking_rate *= 1.05;
            }
            _ => {
                debug!("Unknown emphasis level: {}", level);
            }
        }

        debug!("Applied emphasis level: {}", level);
        Ok(())
    }

    /// Apply break instruction to insert pauses
    async fn apply_break_instruction(
        &self,
        audio: &mut AudioBuffer,
        attributes: &std::collections::HashMap<String, String>,
        _config: &SynthesisConfig,
    ) -> Result<()> {
        let duration = if let Some(time) = attributes.get("time") {
            self.parse_time_duration(time)?
        } else if let Some(strength) = attributes.get("strength") {
            match strength.as_str() {
                "none" => 0.0,
                "x-weak" => 0.1,
                "weak" => 0.25,
                "medium" => 0.5,
                "strong" => 1.0,
                "x-strong" => 2.0,
                _ => 0.5,
            }
        } else {
            0.5 // Default break
        };

        if duration > 0.0 {
            // Create silence buffer and append to audio
            let silence = AudioBuffer::silence(duration, audio.sample_rate(), audio.channels());
            audio.append(&silence)?;

            debug!("Added break of {:.2}s", duration);
        }

        Ok(())
    }

    /// Parse time duration from SSML time strings
    fn parse_time_duration(&self, time_str: &str) -> Result<f32> {
        if time_str.ends_with("ms") {
            time_str
                .trim_end_matches("ms")
                .parse::<f32>()
                .map(|ms| ms / 1000.0)
                .map_err(|_| VoirsError::config_error(format!("Invalid time duration: {time_str}")))
        } else if time_str.ends_with('s') {
            time_str
                .trim_end_matches('s')
                .parse::<f32>()
                .map_err(|_| VoirsError::config_error(format!("Invalid time duration: {time_str}")))
        } else {
            // Assume seconds if no unit
            time_str
                .parse::<f32>()
                .map_err(|_| VoirsError::config_error(format!("Invalid time duration: {time_str}")))
        }
    }

    /// Parse SSML markup
    fn parse_ssml(&self, ssml: &str) -> Result<SsmlParseResult> {
        // Enhanced SSML parsing with full instruction extraction support
        debug!("Parsing SSML markup with {} characters", ssml.len());

        let mut text = String::new();
        let mut instructions = Vec::new();
        let mut chars = ssml.chars().peekable();
        let mut position = 0;

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Parse SSML tag
                if let Some(instruction) = self.parse_ssml_tag(&mut chars, position)? {
                    instructions.push(instruction);
                }
            } else if ch != '\n' && ch != '\r' && ch != '\t' {
                // Add regular text content (skip whitespace control characters)
                text.push(ch);
                position += 1;
            } else if ch == ' ' {
                // Preserve spaces
                text.push(ch);
                position += 1;
            }
        }

        debug!(
            "SSML parsing complete: {} instructions, {} characters of text",
            instructions.len(),
            text.len()
        );

        Ok(SsmlParseResult { text, instructions })
    }

    /// Parse individual SSML tag
    fn parse_ssml_tag(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        position: usize,
    ) -> Result<Option<SsmlInstruction>> {
        let mut tag_content = String::new();

        // Read until closing >
        while let Some(&ch) = chars.peek() {
            if ch == '>' {
                chars.next(); // consume the >
                break;
            }
            tag_content.push(chars.next().unwrap());
        }

        if tag_content.is_empty() {
            return Ok(None);
        }

        // Skip closing tags
        if tag_content.starts_with('/') {
            return Ok(None);
        }

        // Parse tag name and attributes
        let parts: Vec<&str> = tag_content.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(None);
        }

        let tag_name = parts[0].to_lowercase();
        let mut attributes = std::collections::HashMap::new();

        // Parse attributes (simple key="value" or key=value format)
        for part in parts.iter().skip(1) {
            if let Some(eq_pos) = part.find('=') {
                let key = part[..eq_pos].to_string();
                let mut value = part[eq_pos + 1..].to_string();

                // Handle self-closing tags by removing trailing "/" if present
                if value.ends_with('/') {
                    value.pop();
                }

                // Remove quotes
                let value = value.trim_matches('"').trim_matches('\'').to_string();
                attributes.insert(key, value);
            }
        }

        debug!(
            "Parsed SSML tag: {} with {} attributes at position {}",
            tag_name,
            attributes.len(),
            position
        );

        Ok(Some(SsmlInstruction {
            tag: tag_name,
            attributes,
            position,
        }))
    }

    /// Strip SSML tags (simple implementation)
    #[allow(dead_code)]
    fn strip_ssml_tags(&self, ssml: &str) -> String {
        ssml.chars()
            .fold(
                (String::new(), false),
                |(mut result, in_tag), ch| match ch {
                    '<' => (result, true),
                    '>' => (result, false),
                    _ if !in_tag => {
                        result.push(ch);
                        (result, in_tag)
                    }
                    _ => (result, in_tag),
                },
            )
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
    #[allow(dead_code)]
    instructions: Vec<SsmlInstruction>,
}

/// SSML instruction
#[derive(Debug)]
struct SsmlInstruction {
    #[allow(dead_code)]
    tag: String,
    #[allow(dead_code)]
    attributes: std::collections::HashMap<String, String>,
    #[allow(dead_code)]
    position: usize,
}

/// Memory usage monitor
struct MemoryMonitor {
    start_memory: usize,
    test_mode: bool,
}

impl MemoryMonitor {
    fn new(test_mode: bool) -> Self {
        Self {
            start_memory: Self::get_memory_usage(test_mode),
            test_mode,
        }
    }

    fn check_memory_usage(&self, stage: &str) -> Result<()> {
        let current_memory = Self::get_memory_usage(self.test_mode);
        let memory_increase = current_memory.saturating_sub(self.start_memory);

        debug!(
            "Memory usage {} - Current: {} bytes (+{})",
            stage, current_memory, memory_increase
        );

        // Check if memory usage is getting too high
        if memory_increase > 1_000_000_000 {
            // 1GB
            warn!("High memory usage detected: {} bytes", memory_increase);
        }

        Ok(())
    }

    fn get_memory_usage(test_mode: bool) -> usize {
        // Skip expensive system calls in test mode
        if test_mode {
            debug!("Skipping memory usage tracking in test mode");
            return 0;
        }

        // Get current process memory usage in bytes
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = output_str.trim().parse::<usize>() {
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use Windows API to get process memory information
            use std::mem;
            use std::ptr;

            #[allow(non_snake_case)]
            #[repr(C)]
            struct PROCESS_MEMORY_COUNTERS {
                cb: u32,
                PageFaultCount: u32,
                PeakWorkingSetSize: usize,
                WorkingSetSize: usize,
                QuotaPeakPagedPoolUsage: usize,
                QuotaPagedPoolUsage: usize,
                QuotaPeakNonPagedPoolUsage: usize,
                QuotaNonPagedPoolUsage: usize,
                PagefileUsage: usize,
                PeakPagefileUsage: usize,
            }

            extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn GetProcessMemoryInfo(
                    process: *mut std::ffi::c_void,
                    memory_counters: *mut PROCESS_MEMORY_COUNTERS,
                    cb: u32,
                ) -> i32;
            }

            unsafe {
                let mut memory_counters: PROCESS_MEMORY_COUNTERS = mem::zeroed();
                memory_counters.cb = mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

                let process_handle = GetCurrentProcess();
                let success = GetProcessMemoryInfo(
                    process_handle,
                    &mut memory_counters,
                    mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                );

                if success != 0 {
                    return memory_counters.WorkingSetSize;
                } else {
                    debug!("Failed to get Windows process memory info, falling back to 0");
                }
            }
        }

        // Fallback for unsupported platforms or if system calls fail
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

        let orchestrator = SynthesisOrchestrator::with_test_mode(g2p, acoustic, vocoder, true);
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

        let orchestrator = SynthesisOrchestrator::with_test_mode(g2p, acoustic, vocoder, true);
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

        let orchestrator = SynthesisOrchestrator::with_test_mode(g2p, acoustic, vocoder, true);
        let config = SynthesisConfig::default();

        let text = "This is a long text that will be split into chunks for streaming synthesis.";
        let stream = orchestrator.synthesize_stream(text, &config).await;
        assert!(stream.is_ok());
    }
}
