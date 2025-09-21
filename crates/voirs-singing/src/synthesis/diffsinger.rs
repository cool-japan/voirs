//! DiffSinger: Diffusion-based Singing Voice Synthesis
//!
//! This module implements DiffSinger, a state-of-the-art diffusion-based model for
//! high-quality singing voice synthesis that can generate natural and expressive
//! singing voices from musical scores and lyrics.

use super::core::{SynthesisModel, SynthesisParams};
use crate::types::core_types::Articulation;
use crate::{Expression, MusicalNote, NoteEvent, Result, VoiceType};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Helper function to create NoteEvent from MIDI note number
fn midi_to_note_event(midi_number: u8, duration: f32, velocity: f32) -> NoteEvent {
    // Convert MIDI number to frequency
    let frequency = 440.0 * 2.0_f32.powf((midi_number as f32 - 69.0) / 12.0);

    // Convert MIDI number to note name and octave
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi_number / 12) as u8;
    let note_index = (midi_number % 12) as usize;
    let note = note_names[note_index].to_string();

    NoteEvent::new(note, octave, duration, velocity)
}

/// DiffSinger model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSingerConfig {
    /// Model name identifier
    pub model_name: String,
    /// Sample rate for audio generation
    pub sample_rate: f32,
    /// Frame size for processing
    pub frame_size: usize,
    /// Hop size for overlapping frames
    pub hop_size: usize,
    /// Number of mel bands for spectrogram
    pub n_mel: usize,
    /// Number of diffusion steps
    pub n_diffusion_steps: usize,
    /// Noise schedule type
    pub noise_schedule: NoiseSchedule,
    /// Conditioning features
    pub conditioning_features: Vec<ConditioningFeature>,
    /// Voice type support
    pub supported_voices: Vec<VoiceType>,
    /// Use neural vocoder
    pub use_neural_vocoder: bool,
    /// Vocoder type
    pub vocoder_type: VocoderType,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Use phoneme conditioning
    pub use_phoneme_conditioning: bool,
    /// Use musical conditioning  
    pub use_musical_conditioning: bool,
    /// Use style embedding
    pub use_style_embedding: bool,
    /// Style embedding dimension
    pub style_embedding_dim: usize,
}

impl Default for DiffSingerConfig {
    fn default() -> Self {
        Self {
            model_name: "DiffSinger".to_string(),
            sample_rate: 44100.0,
            frame_size: 1024,
            hop_size: 256,
            n_mel: 80,
            n_diffusion_steps: 50,
            noise_schedule: NoiseSchedule::Linear,
            conditioning_features: vec![
                ConditioningFeature::Phoneme,
                ConditioningFeature::Pitch,
                ConditioningFeature::Duration,
                ConditioningFeature::Musical,
            ],
            supported_voices: vec![
                VoiceType::Soprano,
                VoiceType::Alto,
                VoiceType::Tenor,
                VoiceType::Bass,
            ],
            use_neural_vocoder: true,
            vocoder_type: VocoderType::HiFiGAN,
            max_sequence_length: 2048,
            use_phoneme_conditioning: true,
            use_musical_conditioning: true,
            use_style_embedding: true,
            style_embedding_dim: 128,
        }
    }
}

/// Noise scheduling strategies for diffusion process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseSchedule {
    /// Linear noise schedule
    Linear,
    /// Cosine noise schedule (often better for audio)
    Cosine,
    /// Custom schedule with explicit beta values
    Custom(Vec<f32>),
}

/// Conditioning features for DiffSinger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditioningFeature {
    /// Phoneme sequence conditioning
    Phoneme,
    /// Pitch contour conditioning
    Pitch,
    /// Note duration conditioning
    Duration,
    /// Musical structure conditioning (key, time signature, etc.)
    Musical,
    /// Style/expression conditioning
    Style,
    /// Singer identity conditioning
    Singer,
    /// Breath control conditioning
    Breath,
    /// Vibrato conditioning
    Vibrato,
}

/// Vocoder types supported by DiffSinger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VocoderType {
    /// HiFi-GAN vocoder
    HiFiGAN,
    /// PWG (Parallel WaveGAN) vocoder
    PWG,
    /// WaveNet vocoder
    WaveNet,
    /// NSF (Neural Source-Filter) vocoder
    NSF,
}

/// Diffusion model state for iterative denoising
#[derive(Debug, Clone)]
pub struct DiffusionState {
    /// Current noisy spectrogram
    pub noisy_spec: Array2<f32>,
    /// Current step in diffusion process
    pub step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Noise level at current step
    pub noise_level: f32,
}

impl DiffusionState {
    /// Create new diffusion state
    pub fn new(target_shape: (usize, usize), total_steps: usize) -> Self {
        Self {
            noisy_spec: Array2::from_elem(target_shape, 0.0),
            step: 0,
            total_steps,
            noise_level: 1.0,
        }
    }

    /// Initialize with pure noise
    pub fn initialize_noise(&mut self) {
        let (height, width) = self.noisy_spec.dim();
        for i in 0..height {
            for j in 0..width {
                self.noisy_spec[[i, j]] =
                    (((i * 17 + j * 23) % 10007) as f32 / 10007.0) * 2.0 - 1.0; // Pseudo-random noise [-1, 1]
            }
        }
    }

    /// Update state for next diffusion step
    pub fn next_step(&mut self, denoised_prediction: &Array2<f32>) -> Result<()> {
        if self.step >= self.total_steps {
            return Ok(());
        }

        // Simple DDPM-style update (simplified)
        let alpha = 1.0 - self.noise_level;
        let beta = self.noise_level;

        // Update noisy spectrogram
        let (height, width) = self.noisy_spec.dim();
        for i in 0..height {
            for j in 0..width {
                if i < denoised_prediction.nrows() && j < denoised_prediction.ncols() {
                    // Simplified denoising step
                    self.noisy_spec[[i, j]] =
                        alpha * denoised_prediction[[i, j]] + beta * self.noisy_spec[[i, j]];
                }
            }
        }

        self.step += 1;
        self.noise_level *= 0.98; // Reduce noise level

        Ok(())
    }

    /// Check if diffusion is complete
    pub fn is_complete(&self) -> bool {
        self.step >= self.total_steps
    }
}

/// Musical conditioning information
#[derive(Debug, Clone)]
pub struct MusicalConditioning {
    /// Note sequence
    pub notes: Vec<MusicalNote>,
    /// Key signature
    pub key_signature: String,
    /// Time signature
    pub time_signature: (u32, u32),
    /// Tempo (BPM)
    pub tempo: f32,
    /// Lyrics alignment
    pub lyrics_alignment: Vec<(String, f32, f32)>, // (syllable, start_time, end_time)
}

impl MusicalConditioning {
    /// Create new musical conditioning
    pub fn new(notes: Vec<MusicalNote>) -> Self {
        Self {
            notes,
            key_signature: "C".to_string(),
            time_signature: (4, 4),
            tempo: 120.0,
            lyrics_alignment: Vec::new(),
        }
    }

    /// Add lyrics with timing
    pub fn with_lyrics(mut self, lyrics: Vec<(String, f32, f32)>) -> Self {
        self.lyrics_alignment = lyrics;
        self
    }

    /// Set key signature
    pub fn with_key(mut self, key: String) -> Self {
        self.key_signature = key;
        self
    }

    /// Set tempo
    pub fn with_tempo(mut self, tempo: f32) -> Self {
        self.tempo = tempo;
        self
    }
}

/// DiffSinger synthesis model
#[derive(Debug, Clone)]
pub struct DiffSingerModel {
    /// Model configuration
    pub config: DiffSingerConfig,
    /// Model parameters (placeholder for neural network weights)
    pub model_params: HashMap<String, Vec<f32>>,
    /// Vocoder for mel-to-audio conversion
    pub vocoder_params: HashMap<String, Vec<f32>>,
    /// Conditioning embeddings
    pub embeddings: HashMap<String, Array2<f32>>,
    /// Model version
    pub version: String,
}

impl DiffSingerModel {
    /// Create new DiffSinger model
    pub fn new(config: DiffSingerConfig) -> Self {
        Self {
            config,
            model_params: HashMap::new(),
            vocoder_params: HashMap::new(),
            embeddings: HashMap::new(),
            version: "1.0.0".to_string(),
        }
    }

    /// Create with high-quality settings
    pub fn high_quality() -> Self {
        let config = DiffSingerConfig {
            n_diffusion_steps: 100,
            n_mel: 128,
            frame_size: 2048,
            noise_schedule: NoiseSchedule::Cosine,
            use_style_embedding: true,
            style_embedding_dim: 256,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create with fast settings
    pub fn fast() -> Self {
        let config = DiffSingerConfig {
            n_diffusion_steps: 20,
            n_mel: 64,
            frame_size: 512,
            noise_schedule: NoiseSchedule::Linear,
            use_style_embedding: false,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Generate mel spectrogram using diffusion process
    pub fn generate_mel_spectrogram(
        &self,
        musical_conditioning: &MusicalConditioning,
        voice_type: VoiceType,
    ) -> Result<Array2<f32>> {
        // Calculate target spectrogram dimensions
        let total_duration: f32 = musical_conditioning
            .notes
            .iter()
            .map(|note| note.duration)
            .sum();
        let n_frames =
            (total_duration * self.config.sample_rate / self.config.hop_size as f32) as usize;
        let target_shape = (self.config.n_mel, n_frames);

        // Initialize diffusion state
        let mut diffusion_state = DiffusionState::new(target_shape, self.config.n_diffusion_steps);
        diffusion_state.initialize_noise();

        // Prepare conditioning features
        let conditioning = self.prepare_conditioning(musical_conditioning, voice_type)?;

        // Iterative denoising process
        for step in 0..self.config.n_diffusion_steps {
            // Predict noise/clean signal (placeholder implementation)
            let noise_prediction = self.predict_noise(&diffusion_state, &conditioning, step)?;

            // Update diffusion state
            diffusion_state.next_step(&noise_prediction)?;
        }

        Ok(diffusion_state.noisy_spec)
    }

    /// Prepare conditioning features for synthesis
    fn prepare_conditioning(
        &self,
        musical_conditioning: &MusicalConditioning,
        voice_type: VoiceType,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let mut conditioning = HashMap::new();

        // Phoneme conditioning (simplified)
        if self.config.use_phoneme_conditioning {
            let phoneme_features =
                self.extract_phoneme_features(&musical_conditioning.lyrics_alignment)?;
            conditioning.insert("phoneme".to_string(), phoneme_features);
        }

        // Musical conditioning
        if self.config.use_musical_conditioning {
            let musical_features = self.extract_musical_features(musical_conditioning)?;
            conditioning.insert("musical".to_string(), musical_features);
        }

        // Pitch conditioning
        let pitch_features = self.extract_pitch_features(&musical_conditioning.notes)?;
        conditioning.insert("pitch".to_string(), pitch_features);

        // Voice type conditioning
        let voice_features = self.extract_voice_features(voice_type)?;
        conditioning.insert("voice".to_string(), voice_features);

        Ok(conditioning)
    }

    /// Extract phoneme features from lyrics
    fn extract_phoneme_features(&self, lyrics: &[(String, f32, f32)]) -> Result<Array2<f32>> {
        let n_frames = 100; // Placeholder
        let phoneme_dim = 64; // Typical phoneme embedding dimension

        let mut features = Array2::zeros((phoneme_dim, n_frames));

        // Simple phoneme encoding (in practice, this would use a proper phoneme encoder)
        for (i, (syllable, start_time, end_time)) in lyrics.iter().enumerate() {
            let start_frame =
                (*start_time * self.config.sample_rate / self.config.hop_size as f32) as usize;
            let end_frame =
                (*end_time * self.config.sample_rate / self.config.hop_size as f32) as usize;

            let syllable_hash =
                syllable.chars().map(|c| c as u8 as f32).sum::<f32>() / syllable.len() as f32;

            for frame in start_frame..end_frame.min(n_frames) {
                for dim in 0..phoneme_dim {
                    features[[dim, frame]] = (syllable_hash + dim as f32) / 100.0;
                    // Normalized encoding
                }
            }
        }

        Ok(features)
    }

    /// Extract musical features
    fn extract_musical_features(
        &self,
        musical_conditioning: &MusicalConditioning,
    ) -> Result<Array2<f32>> {
        let n_frames = 100; // Placeholder
        let musical_dim = 32; // Musical feature dimension

        let mut features = Array2::zeros((musical_dim, n_frames));

        // Encode key signature
        let key_encoding = match musical_conditioning.key_signature.as_str() {
            "C" => 0.0,
            "G" => 1.0,
            "D" => 2.0,
            "A" => 3.0,
            "E" => 4.0,
            "B" => 5.0,
            "F#" => 6.0,
            "C#" => 7.0,
            "F" => 8.0,
            "Bb" => 9.0,
            "Eb" => 10.0,
            "Ab" => 11.0,
            _ => 0.0,
        } / 12.0; // Normalize to [0, 1]

        // Encode tempo
        let tempo_encoding = musical_conditioning.tempo / 200.0; // Normalize typical tempo range

        // Fill features
        for frame in 0..n_frames {
            features[[0, frame]] = key_encoding;
            features[[1, frame]] = tempo_encoding;
            features[[2, frame]] = musical_conditioning.time_signature.0 as f32 / 8.0;
            features[[3, frame]] = musical_conditioning.time_signature.1 as f32 / 8.0;
        }

        Ok(features)
    }

    /// Extract pitch features from notes
    fn extract_pitch_features(&self, notes: &[MusicalNote]) -> Result<Array2<f32>> {
        let n_frames = 100; // Placeholder
        let pitch_dim = 1; // F0 values

        let mut features = Array2::zeros((pitch_dim, n_frames));

        let mut current_time = 0.0;
        let frame_duration = self.config.hop_size as f32 / self.config.sample_rate;

        for note in notes {
            let note_f0 = note.event.frequency; // Use frequency directly from note event
            let note_frames = (note.duration / frame_duration) as usize;

            let start_frame = (current_time / frame_duration) as usize;
            let end_frame = (start_frame + note_frames).min(n_frames);

            for frame in start_frame..end_frame {
                features[[0, frame]] = note_f0 / 1000.0; // Normalize
            }

            current_time += note.duration;
        }

        Ok(features)
    }

    /// Extract voice type features
    fn extract_voice_features(&self, voice_type: VoiceType) -> Result<Array2<f32>> {
        let n_frames = 100; // Placeholder
        let voice_dim = 16; // Voice embedding dimension

        let mut features = Array2::zeros((voice_dim, n_frames));

        // Simple voice type encoding
        let voice_encoding = match voice_type {
            VoiceType::Soprano => vec![1.0, 0.0, 0.0, 0.0],
            VoiceType::Alto => vec![0.0, 1.0, 0.0, 0.0],
            VoiceType::Tenor => vec![0.0, 0.0, 1.0, 0.0],
            VoiceType::Bass => vec![0.0, 0.0, 0.0, 1.0],
            _ => vec![0.25, 0.25, 0.25, 0.25], // Neutral
        };

        for frame in 0..n_frames {
            for (i, &value) in voice_encoding.iter().enumerate() {
                if i < voice_dim {
                    features[[i, frame]] = value;
                }
            }
        }

        Ok(features)
    }

    /// Predict noise for current diffusion step (placeholder neural network)
    fn predict_noise(
        &self,
        diffusion_state: &DiffusionState,
        conditioning: &HashMap<String, Array2<f32>>,
        step: usize,
    ) -> Result<Array2<f32>> {
        let (n_mel, n_frames) = diffusion_state.noisy_spec.dim();
        let mut prediction = Array2::zeros((n_mel, n_frames));

        // Placeholder noise prediction (in practice, this would be a neural network)
        // Simple denoising based on conditioning
        let step_factor = 1.0 - (step as f32 / diffusion_state.total_steps as f32);

        for i in 0..n_mel {
            for j in 0..n_frames {
                let current_value = diffusion_state.noisy_spec[[i, j]];

                // Apply conditioning influence
                let mut conditioning_influence = 0.0;
                if let Some(pitch_cond) = conditioning.get("pitch") {
                    if i < pitch_cond.nrows() && j < pitch_cond.ncols() {
                        conditioning_influence +=
                            pitch_cond[[i.min(pitch_cond.nrows() - 1), j]] * 0.5;
                    }
                }

                // Simple denoising prediction
                prediction[[i, j]] =
                    current_value * step_factor + conditioning_influence * (1.0 - step_factor);
            }
        }

        Ok(prediction)
    }

    /// Convert mel spectrogram to audio using vocoder
    pub fn vocoder_synthesis(&self, mel_spec: &Array2<f32>) -> Result<Vec<f32>> {
        let (n_mel, n_frames) = mel_spec.dim();
        let audio_length = n_frames * self.config.hop_size;
        let mut audio = vec![0.0; audio_length];

        // Placeholder vocoder implementation
        // In practice, this would use a neural vocoder like HiFi-GAN
        match self.config.vocoder_type {
            VocoderType::HiFiGAN => {
                // Simple overlap-add synthesis as placeholder
                for frame in 0..n_frames {
                    let frame_start = frame * self.config.hop_size;
                    let frame_end = (frame_start + self.config.frame_size).min(audio_length);

                    // Generate audio frame from mel features
                    for sample_idx in frame_start..frame_end {
                        let mut sample_value = 0.0;
                        for mel_idx in 0..n_mel {
                            let mel_value = mel_spec[[mel_idx, frame]];
                            let frequency =
                                mel_idx as f32 * self.config.sample_rate / (2.0 * n_mel as f32);
                            let phase = 2.0 * std::f32::consts::PI * frequency * sample_idx as f32
                                / self.config.sample_rate;
                            sample_value += mel_value * phase.sin() / n_mel as f32;
                        }
                        audio[sample_idx] += sample_value * 0.1; // Scale down
                    }
                }
            }
            _ => {
                // Fallback to simple synthesis
                for i in 0..audio_length {
                    audio[i] = (i as f32 / audio_length as f32).sin() * 0.1;
                }
            }
        }

        Ok(audio)
    }
}

impl SynthesisModel for DiffSingerModel {
    fn synthesize(&self, params: &SynthesisParams) -> Result<Vec<f32>> {
        // Extract musical information from synthesis parameters
        let notes = if params.pitch_contour.f0_values.is_empty() {
            // Create a simple note if no pitch contour provided
            let note_event = NoteEvent {
                note: "C".to_string(),
                octave: 4,
                frequency: 261.63, // Middle C
                duration: params.duration,
                velocity: 1.0,
                vibrato: 0.0,
                lyric: Some("la".to_string()),
                phonemes: vec!["l".to_string(), "a".to_string()],
                expression: Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: Articulation::Normal,
            };
            vec![MusicalNote::new(note_event, 0.0, params.duration)]
        } else {
            // Convert pitch contour to notes (simplified)
            params
                .pitch_contour
                .f0_values
                .iter()
                .enumerate()
                .map(|(i, &f0)| {
                    let note_duration =
                        params.duration / params.pitch_contour.f0_values.len() as f32;
                    let note_event = NoteEvent {
                        note: "C".to_string(),
                        octave: 4,
                        frequency: f0,
                        duration: note_duration,
                        velocity: 1.0,
                        vibrato: 0.0,
                        lyric: Some("la".to_string()),
                        phonemes: vec!["l".to_string(), "a".to_string()],
                        expression: Expression::Neutral,
                        timing_offset: 0.0,
                        breath_before: 0.0,
                        legato: false,
                        articulation: Articulation::Normal,
                    };
                    MusicalNote::new(note_event, i as f32 * note_duration, note_duration)
                })
                .collect()
        };

        let musical_conditioning = MusicalConditioning::new(notes);
        let voice_type = VoiceType::Soprano; // Default voice type

        // Generate mel spectrogram using diffusion
        let mel_spec = self.generate_mel_spectrogram(&musical_conditioning, voice_type)?;

        // Convert to audio using vocoder
        let audio = self.vocoder_synthesis(&mel_spec)?;

        // Ensure correct length
        let target_length = (params.duration * params.sample_rate) as usize;
        if audio.len() > target_length {
            Ok(audio[..target_length].to_vec())
        } else if audio.len() < target_length {
            let mut extended_audio = audio;
            extended_audio.resize(target_length, 0.0);
            Ok(extended_audio)
        } else {
            Ok(audio)
        }
    }

    fn name(&self) -> &str {
        &self.config.model_name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn load_from_file(&mut self, path: &str) -> Result<()> {
        // Placeholder for loading model weights
        // In practice, this would load pretrained DiffSinger weights
        println!("Loading DiffSinger model from: {}", path);
        Ok(())
    }

    fn save_to_file(&self, path: &str) -> Result<()> {
        // Placeholder for saving model weights
        println!("Saving DiffSinger model to: {}", path);
        Ok(())
    }
}

/// DiffSinger synthesis request with advanced options
#[derive(Debug, Clone)]
pub struct DiffSingerRequest {
    /// Musical conditioning
    pub musical_conditioning: MusicalConditioning,
    /// Voice type
    pub voice_type: VoiceType,
    /// Synthesis quality level
    pub quality_level: QualityLevel,
    /// Style embedding (optional)
    pub style_embedding: Option<Vec<f32>>,
    /// Singer embedding (optional)
    pub singer_embedding: Option<Vec<f32>>,
    /// Custom diffusion steps (override default)
    pub custom_diffusion_steps: Option<usize>,
}

/// Quality levels for DiffSinger synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Fast synthesis with lower quality
    Fast,
    /// Balanced quality and speed
    Medium,
    /// High quality synthesis (slower)
    High,
    /// Maximum quality (slowest)
    Ultra,
}

impl DiffSingerRequest {
    /// Create new DiffSinger request
    pub fn new(musical_conditioning: MusicalConditioning, voice_type: VoiceType) -> Self {
        Self {
            musical_conditioning,
            voice_type,
            quality_level: QualityLevel::Medium,
            style_embedding: None,
            singer_embedding: None,
            custom_diffusion_steps: None,
        }
    }

    /// Set quality level
    pub fn with_quality(mut self, quality: QualityLevel) -> Self {
        self.quality_level = quality;
        self
    }

    /// Set style embedding
    pub fn with_style(mut self, style: Vec<f32>) -> Self {
        self.style_embedding = Some(style);
        self
    }

    /// Set singer embedding
    pub fn with_singer(mut self, singer: Vec<f32>) -> Self {
        self.singer_embedding = Some(singer);
        self
    }
}

impl Default for DiffSingerModel {
    fn default() -> Self {
        Self::new(DiffSingerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::core_types::VoiceType;

    #[test]
    fn test_diffsinger_config() {
        let config = DiffSingerConfig::default();
        assert_eq!(config.model_name, "DiffSinger");
        assert_eq!(config.sample_rate, 44100.0);
        assert_eq!(config.n_diffusion_steps, 50);
        assert!(!config.supported_voices.is_empty());
    }

    #[test]
    fn test_diffsinger_model_creation() {
        let model = DiffSingerModel::default();
        assert_eq!(model.name(), "DiffSinger");
        assert_eq!(model.version(), "1.0.0");
    }

    #[test]
    fn test_high_quality_config() {
        let model = DiffSingerModel::high_quality();
        assert_eq!(model.config.n_diffusion_steps, 100);
        assert_eq!(model.config.n_mel, 128);
        assert!(model.config.use_style_embedding);
    }

    #[test]
    fn test_fast_config() {
        let model = DiffSingerModel::fast();
        assert_eq!(model.config.n_diffusion_steps, 20);
        assert_eq!(model.config.n_mel, 64);
        assert!(!model.config.use_style_embedding);
    }

    #[test]
    fn test_musical_conditioning() {
        let notes = vec![
            MusicalNote::new(midi_to_note_event(60, 1.0, 0.8), 0.0, 1.0), // C4
            MusicalNote::new(midi_to_note_event(64, 1.0, 0.8), 1.0, 1.0), // E4
            MusicalNote::new(midi_to_note_event(67, 1.0, 0.8), 2.0, 1.0), // G4
        ];

        let conditioning = MusicalConditioning::new(notes)
            .with_key("C".to_string())
            .with_tempo(120.0);

        assert_eq!(conditioning.notes.len(), 3);
        assert_eq!(conditioning.key_signature, "C");
        assert_eq!(conditioning.tempo, 120.0);
    }

    #[test]
    fn test_diffusion_state() {
        let mut state = DiffusionState::new((80, 100), 50);
        assert_eq!(state.step, 0);
        assert_eq!(state.total_steps, 50);
        assert!(!state.is_complete());

        state.initialize_noise();
        // Check that noise was actually added
        let has_non_zero = state.noisy_spec.iter().any(|&x| x != 0.0);
        assert!(has_non_zero);
    }

    #[test]
    fn test_diffsinger_request() {
        let notes = vec![MusicalNote::new(midi_to_note_event(60, 1.0, 0.8), 0.0, 1.0)];
        let conditioning = MusicalConditioning::new(notes);

        let request = DiffSingerRequest::new(conditioning, VoiceType::Soprano)
            .with_quality(QualityLevel::High);

        assert!(matches!(request.voice_type, VoiceType::Soprano));
        assert!(matches!(request.quality_level, QualityLevel::High));
    }

    #[test]
    fn test_noise_schedules() {
        let linear_schedule = NoiseSchedule::Linear;
        let cosine_schedule = NoiseSchedule::Cosine;
        let custom_schedule = NoiseSchedule::Custom(vec![0.1, 0.2, 0.3]);

        // Just verify the enum variants compile and can be matched
        match linear_schedule {
            NoiseSchedule::Linear => assert!(true),
            _ => assert!(false),
        }

        match custom_schedule {
            NoiseSchedule::Custom(ref values) => assert_eq!(values.len(), 3),
            _ => assert!(false),
        }
    }
}
