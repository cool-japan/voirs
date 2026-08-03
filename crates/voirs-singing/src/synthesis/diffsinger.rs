//! DiffSinger: Diffusion-based Singing Voice Synthesis
//!
//! This module implements the DDPM (denoising diffusion probabilistic model)
//! structure behind DiffSinger: a real noise schedule (see
//! [`compute_alpha_bar`]) and a real iterative reverse-process loop (see
//! [`DiffusionState::next_step`]).
//!
//! The noise-*prediction* step has two selectable backends (see
//! [`DiffSingerConfig::use_neural_denoiser`]):
//! - [`DiffusionDenoiser`]: a real, trainable/loadable candle neural network.
//!   This crate ships no pretrained weights, so it must be loaded explicitly
//!   via [`DiffSingerModel::load_from_file`] before use; without loaded
//!   weights it fails closed rather than fabricating a prediction.
//! - `predict_noise_analytical_fallback`: an explicit, non-neural DSP
//!   approximation that reconstructs the noise estimate algebraically from
//!   the conditioning features. This is the default, since it requires no
//!   pretrained weights, but it must never be presented as "diffusion
//!   synthesis" on its own - it is a deterministic fallback the caller opts
//!   into (or, today, gets by default in the absence of real weights).
//!
//! The vocoder stage has an analogous split (see
//! [`DiffSingerConfig::use_neural_vocoder`]): a real neural vocoder
//! (HiFi-GAN/PWG/...) is not implemented in this crate yet, so enabling it
//! fails closed; the default is an explicit additive-sine DSP fallback.

use super::core::{SynthesisModel, SynthesisParams};
use crate::types::core_types::Articulation;
use crate::{Error, Expression, MusicalNote, NoteEvent, Result, VoiceType};
use candle_core::{Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use scirs2_core::ndarray::Array2;
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
    let octave = midi_number / 12;
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
    /// Real, effective selector: `true` requires a loaded [`DiffusionDenoiser`]
    /// (via [`DiffSingerModel::load_from_file`]) and neural vocoder weights;
    /// synthesis fails closed (a typed [`crate::Error::Model`]) rather than
    /// fabricating output if the corresponding weights aren't loaded.
    /// Defaults to `false` since this crate ships no pretrained weights, so
    /// synthesis instead uses the explicit analytical DSP fallbacks.
    pub use_neural_vocoder: bool,
    /// Vocoder architecture a `true` `use_neural_vocoder` would target (not
    /// yet implemented locally - see module docs).
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
    /// Real, effective selector for the diffusion noise-prediction step:
    /// `true` requires a [`DiffusionDenoiser`] loaded via
    /// [`DiffSingerModel::load_from_file`] and fails closed without one;
    /// `false` (the default) uses the explicit, non-neural
    /// `predict_noise_analytical_fallback` DSP approximation.
    pub use_neural_denoiser: bool,
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
            use_neural_vocoder: false,
            vocoder_type: VocoderType::HiFiGAN,
            max_sequence_length: 2048,
            use_phoneme_conditioning: true,
            use_musical_conditioning: true,
            use_style_embedding: true,
            style_embedding_dim: 128,
            use_neural_denoiser: false,
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

/// Real, trainable/loadable neural noise-prediction network `ε_θ(x_t, t, c)`.
///
/// Given a noisy mel spectrogram, the conditioning features, and the current
/// diffusion timestep, predicts the noise added at that step. Every frame is
/// processed independently through a shared 3-layer MLP (a pointwise
/// conditional denoiser): `linear3(relu(linear2(relu(linear1(x)))))`, where
/// `x` concatenates the noisy mel frame, a fixed-size conditioning vector
/// (see [`conditioning_frame_vector`]), and a sinusoidal timestep embedding
/// (see [`timestep_embedding`]).
///
/// This is a genuine candle neural network - weights are real
/// [`candle_core::Var`]s backed by a [`VarMap`], trainable, and persisted in
/// safetensors format via [`Self::save`]/[`Self::load`] - not an analytical
/// shortcut. This crate ships no pretrained weights for it, so
/// [`DiffSingerModel::load_from_file`] must be called with a real
/// safetensors file before it can be used; see the module docs for the
/// (default) non-neural fallback used otherwise.
#[derive(Debug, Clone)]
pub struct DiffusionDenoiser {
    n_mel: usize,
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
    device: Device,
    varmap: VarMap,
}

impl DiffusionDenoiser {
    /// Conditioning vector dimensions, matching `extract_pitch_features`
    /// (1), `extract_musical_features` (32), `extract_phoneme_features`
    /// (64), and `extract_voice_features` (16).
    const PITCH_DIM: usize = 1;
    const MUSICAL_DIM: usize = 32;
    const PHONEME_DIM: usize = 64;
    const VOICE_DIM: usize = 16;
    /// Sinusoidal timestep embedding dimension.
    const TIMESTEP_EMBED_DIM: usize = 16;
    /// Hidden layer width.
    const HIDDEN_DIM: usize = 128;

    /// Total conditioning-vector width (see [`conditioning_frame_vector`]).
    const fn conditioning_dim() -> usize {
        Self::PITCH_DIM + Self::MUSICAL_DIM + Self::PHONEME_DIM + Self::VOICE_DIM
    }

    /// Build a fresh (randomly-initialized, untrained) denoiser for `n_mel`
    /// mel bands. Real weights must be loaded via [`Self::load`] before the
    /// network's predictions are meaningful.
    fn new(n_mel: usize, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let input_dim = n_mel + Self::conditioning_dim() + Self::TIMESTEP_EMBED_DIM;

        let linear1 = candle_nn::linear(input_dim, Self::HIDDEN_DIM, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(Self::HIDDEN_DIM, Self::HIDDEN_DIM, vb.pp("linear2"))?;
        let linear3 = candle_nn::linear(Self::HIDDEN_DIM, n_mel, vb.pp("linear3"))?;

        Ok(Self {
            n_mel,
            linear1,
            linear2,
            linear3,
            device,
            varmap,
        })
    }

    /// Real neural forward pass: predicts noise for every frame of
    /// `noisy_mel` at once, conditioned on `conditioning` and the diffusion
    /// `step`.
    fn forward(
        &self,
        noisy_mel: &Array2<f32>,
        conditioning: &HashMap<String, Array2<f32>>,
        step: usize,
        total_steps: usize,
    ) -> Result<Array2<f32>> {
        let (n_mel, n_frames) = noisy_mel.dim();
        if n_mel != self.n_mel {
            return Err(Error::Model(format!(
                "DiffusionDenoiser was built for {} mel bands but received {n_mel}",
                self.n_mel
            )));
        }

        let embed = timestep_embedding(step, total_steps, Self::TIMESTEP_EMBED_DIM);
        let input_dim = n_mel + Self::conditioning_dim() + Self::TIMESTEP_EMBED_DIM;
        let mut input_data = Vec::with_capacity(n_frames * input_dim);
        for frame in 0..n_frames {
            for mel_bin in 0..n_mel {
                input_data.push(noisy_mel[[mel_bin, frame]]);
            }
            input_data.extend_from_slice(&conditioning_frame_vector(conditioning, frame));
            input_data.extend_from_slice(&embed);
        }

        let input_tensor = Tensor::from_vec(input_data, (n_frames, input_dim), &self.device)?;
        let hidden = self.linear1.forward(&input_tensor)?.relu()?;
        let hidden = self.linear2.forward(&hidden)?.relu()?;
        let output = self.linear3.forward(&hidden)?; // (n_frames, n_mel)

        let output_data = output.to_vec2::<f32>()?;
        let mut result = Array2::<f32>::zeros((n_mel, n_frames));
        for (frame, row) in output_data.iter().enumerate() {
            for (mel_bin, &value) in row.iter().enumerate() {
                result[[mel_bin, frame]] = value;
            }
        }
        Ok(result)
    }

    /// Persist the real weight tensors in safetensors format.
    fn save(&self, path: &str) -> Result<()> {
        self.varmap
            .save(path)
            .map_err(|e| Error::Model(format!("Failed to save DiffusionDenoiser weights to {path}: {e}")))
    }

    /// Load real weight tensors from a safetensors file, failing closed
    /// (a typed [`Error::Model`]) if the file doesn't exist or its tensors
    /// don't match this architecture's shapes - never silently keeping
    /// untrained construction-time weights while reporting success.
    fn load(n_mel: usize, device: Device, path: &str) -> Result<Self> {
        if !std::path::Path::new(path).exists() {
            return Err(Error::Model(format!(
                "DiffSinger neural denoiser weights not available at {path}"
            )));
        }
        let mut denoiser = Self::new(n_mel, device)?;
        denoiser.varmap.load(path).map_err(|e| {
            Error::Model(format!(
                "Failed to load DiffusionDenoiser weights from {path}: {e}"
            ))
        })?;
        Ok(denoiser)
    }
}

/// Build a fixed-size conditioning vector for one frame
/// (`DiffusionDenoiser::conditioning_dim()` wide: pitch + musical + phoneme +
/// voice), zero-filling any conditioning type that's absent (e.g. disabled
/// via config) or shorter than expected, rather than failing or fabricating
/// a nonzero value.
fn conditioning_frame_vector(conditioning: &HashMap<String, Array2<f32>>, frame: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(DiffusionDenoiser::conditioning_dim());
    for (key, dim) in [
        ("pitch", DiffusionDenoiser::PITCH_DIM),
        ("musical", DiffusionDenoiser::MUSICAL_DIM),
        ("phoneme", DiffusionDenoiser::PHONEME_DIM),
        ("voice", DiffusionDenoiser::VOICE_DIM),
    ] {
        let array = conditioning.get(key);
        for row in 0..dim {
            let value = array
                .filter(|a| row < a.nrows() && frame < a.ncols())
                .map(|a| a[[row, frame]])
                .unwrap_or(0.0);
            vector.push(value);
        }
    }
    vector
}

/// Standard sinusoidal timestep embedding (as used in the original DDPM /
/// Transformer positional encoding), giving the denoiser network a smooth,
/// distinguishable representation of how noisy the current diffusion step
/// is expected to be.
fn timestep_embedding(step: usize, total_steps: usize, dim: usize) -> Vec<f32> {
    let t = step as f32 / total_steps.max(1) as f32;
    (0..dim)
        .map(|i| {
            let pair_index = (i / 2) as f32;
            let freq = 10000f32.powf(-2.0 * pair_index / dim as f32);
            if i % 2 == 0 {
                (t * freq).sin()
            } else {
                (t * freq).cos()
            }
        })
        .collect()
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

    /// Advance the diffusion state by one denoising step.
    ///
    /// Given the predicted noise ε̂ returned by `predict_noise`, we apply the
    /// standard DDPM reverse-process update in the "predict x_0" form:
    ///
    ///   x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε̂) / sqrt(ᾱ_t)
    ///   x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0 + sqrt(1-ᾱ_{t-1}) * ε̂
    ///
    /// This is deterministic (no added stochastic noise), which is appropriate
    /// for inference.  When `step == 0` the update writes x̂_0 directly.
    pub fn next_step(&mut self, noise_prediction: &Array2<f32>) -> Result<()> {
        if self.step >= self.total_steps {
            return Ok(());
        }

        let t = self.step;
        let total = self.total_steps;

        let alpha_bar_t = compute_alpha_bar(t, total);
        let sqrt_alpha_bar_t = alpha_bar_t.sqrt().max(1e-6);
        let sqrt_one_minus_alpha_bar_t = (1.0_f32 - alpha_bar_t).sqrt().max(1e-6);

        // ᾱ_{t-1}: when t == 0 the previous step has ᾱ = 1 (pure signal).
        let alpha_bar_prev = if t == 0 {
            1.0_f32
        } else {
            compute_alpha_bar(t - 1, total)
        };
        let sqrt_alpha_bar_prev = alpha_bar_prev.sqrt();
        let sqrt_one_minus_alpha_bar_prev = (1.0_f32 - alpha_bar_prev).sqrt();

        let (height, width) = self.noisy_spec.dim();
        let pred_rows = noise_prediction.nrows();
        let pred_cols = noise_prediction.ncols();

        for i in 0..height {
            for j in 0..width {
                let x_t = self.noisy_spec[[i, j]];
                let eps_hat = if i < pred_rows && j < pred_cols {
                    noise_prediction[[i, j]]
                } else {
                    0.0
                };

                // Estimate clean signal from noisy sample and predicted noise.
                let x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_hat) / sqrt_alpha_bar_t;

                // Reconstruct x_{t-1} using the "predict x_0" parameterization.
                self.noisy_spec[[i, j]] =
                    sqrt_alpha_bar_prev * x0_hat + sqrt_one_minus_alpha_bar_prev * eps_hat;
            }
        }

        self.step += 1;
        // Update noise_level to reflect ᾱ after the step advance.
        let new_t = self.step;
        self.noise_level = if new_t < total {
            1.0 - compute_alpha_bar(new_t, total)
        } else {
            0.0
        };

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

/// Compute the cumulative product of (1 - β_t) for t = 0..=step using a linear beta schedule.
///
/// The linear schedule interpolates β from `beta_start` (1e-4) to `beta_end` (0.02)
/// over `total_steps` steps, matching the original DDPM paper.
///
/// Returns ᾱ_step = ∏_{s=0}^{step} (1 - β_s)
fn compute_alpha_bar(step: usize, total_steps: usize) -> f32 {
    const BETA_START: f32 = 1e-4;
    const BETA_END: f32 = 0.02;

    // Guard against degenerate schedules
    let denom = (total_steps.saturating_sub(1).max(1)) as f32;

    (0..=step)
        .map(|s| {
            let beta_s = BETA_START + (BETA_END - BETA_START) * (s as f32 / denom);
            1.0_f32 - beta_s
        })
        .product::<f32>()
}

/// DiffSinger synthesis model
#[derive(Debug, Clone)]
pub struct DiffSingerModel {
    /// Model configuration
    pub config: DiffSingerConfig,
    /// Real, loadable noise-prediction network (see [`DiffusionDenoiser`]).
    /// `None` until [`Self::load_from_file`] is called with real weights;
    /// [`Self::predict_noise`] fails closed rather than using this when it's
    /// absent and `config.use_neural_denoiser` is `true`.
    pub neural_denoiser: Option<DiffusionDenoiser>,
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
            neural_denoiser: None,
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

    /// Predict noise ε_θ(x_t, t, c) for the current diffusion step.
    ///
    /// Dispatches to a real, loaded [`DiffusionDenoiser`] when
    /// `config.use_neural_denoiser` is `true` (failing closed if none is
    /// loaded), otherwise uses the explicit, non-neural
    /// [`Self::predict_noise_analytical_fallback`].
    fn predict_noise(
        &self,
        diffusion_state: &DiffusionState,
        conditioning: &HashMap<String, Array2<f32>>,
        step: usize,
    ) -> Result<Array2<f32>> {
        if self.config.use_neural_denoiser {
            let denoiser = self.neural_denoiser.as_ref().ok_or_else(|| {
                Error::Model(
                    "use_neural_denoiser is enabled but no DiffusionDenoiser weights are \
                     loaded; call `load_from_file` with a real safetensors file, or set \
                     `use_neural_denoiser = false` to use the analytical DSP fallback"
                        .to_string(),
                )
            })?;
            return denoiser.forward(
                &diffusion_state.noisy_spec,
                conditioning,
                step,
                diffusion_state.total_steps,
            );
        }
        self.predict_noise_analytical_fallback(diffusion_state, conditioning, step)
    }

    /// Explicit, non-neural analytical DSP fallback for noise prediction.
    ///
    /// Without a trained neural network we treat the conditioning features as
    /// the "target" clean signal x̂_0 and derive the noise estimate analytically
    /// from the DDPM forward-process equation:
    ///
    ///   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    ///
    /// Rearranging for ε:
    ///
    ///   ε_estimate = (x_t - sqrt(ᾱ_t) * x̂_0) / sqrt(1-ᾱ_t)
    ///
    /// The conditioning-based target mel x̂_0 is composed from pitch (primary)
    /// and musical (secondary, scaled 0.5) features.  Any missing conditioning
    /// channels default to zero, which results in a noise estimate that simply
    /// drives the noisy spectrogram toward silence — a safe fallback.
    ///
    /// This is **not** a trained neural network and must never be presented
    /// as "DiffSinger diffusion synthesis" on its own - see the module docs.
    fn predict_noise_analytical_fallback(
        &self,
        diffusion_state: &DiffusionState,
        conditioning: &HashMap<String, Array2<f32>>,
        step: usize,
    ) -> Result<Array2<f32>> {
        let (n_mel, n_frames) = diffusion_state.noisy_spec.dim();

        // ── Step 1: build the conditioning-based target mel x̂_0 ──────────────
        let mut target = Array2::<f32>::zeros((n_mel, n_frames));

        // Pitch conditioning is the primary structural guide.
        if let Some(pitch_cond) = conditioning.get("pitch") {
            let rows = n_mel.min(pitch_cond.nrows());
            let cols = n_frames.min(pitch_cond.ncols());
            for i in 0..rows {
                for j in 0..cols {
                    target[[i, j]] += pitch_cond[[i, j]];
                }
            }
        }

        // Musical conditioning provides harmonic context at half weight.
        if let Some(musical_cond) = conditioning.get("musical") {
            const MUSICAL_SCALE: f32 = 0.5;
            let rows = n_mel.min(musical_cond.nrows());
            let cols = n_frames.min(musical_cond.ncols());
            for i in 0..rows {
                for j in 0..cols {
                    target[[i, j]] += musical_cond[[i, j]] * MUSICAL_SCALE;
                }
            }
        }

        // ── Step 2: compute ᾱ_t from the linear beta schedule ────────────────
        let alpha_bar_t = compute_alpha_bar(step, diffusion_state.total_steps);
        let sqrt_alpha_bar = alpha_bar_t.sqrt();
        // Clamp denominator away from zero to avoid division-by-zero at t=0.
        let sqrt_one_minus_alpha_bar = (1.0_f32 - alpha_bar_t).sqrt().max(1e-6);

        // ── Step 3: ε_estimate = (x_t - sqrt(ᾱ_t) * x̂_0) / sqrt(1-ᾱ_t) ────
        let mut noise_estimate = Array2::<f32>::zeros((n_mel, n_frames));
        for i in 0..n_mel {
            for j in 0..n_frames {
                noise_estimate[[i, j]] = (diffusion_state.noisy_spec[[i, j]]
                    - sqrt_alpha_bar * target[[i, j]])
                    / sqrt_one_minus_alpha_bar;
            }
        }

        Ok(noise_estimate)
    }

    /// Convert mel spectrogram to audio.
    ///
    /// Dispatches to a real trained neural vocoder when
    /// `config.use_neural_vocoder` is `true` - not implemented locally in
    /// this crate yet, so this fails closed rather than fabricating audio -
    /// otherwise uses the explicit, non-neural
    /// [`Self::vocoder_synthesis_analytical_fallback`].
    pub fn vocoder_synthesis(&self, mel_spec: &Array2<f32>) -> Result<Vec<f32>> {
        if self.config.use_neural_vocoder {
            return Err(Error::Model(format!(
                "use_neural_vocoder is enabled but this crate does not ship a trained neural \
                 vocoder integration for {:?} yet; set `use_neural_vocoder = false` to use the \
                 analytical sine-additive DSP fallback",
                self.config.vocoder_type
            )));
        }
        self.vocoder_synthesis_analytical_fallback(mel_spec)
    }

    /// Explicit, non-neural additive-sine DSP fallback for mel-to-audio
    /// conversion. This is **not** a trained neural vocoder (HiFi-GAN or
    /// otherwise) and must never be presented as one - see the module docs.
    fn vocoder_synthesis_analytical_fallback(&self, mel_spec: &Array2<f32>) -> Result<Vec<f32>> {
        let (n_mel, n_frames) = mel_spec.dim();
        let audio_length = n_frames * self.config.hop_size;
        let mut audio = vec![0.0; audio_length];

        match self.config.vocoder_type {
            VocoderType::HiFiGAN => {
                // Simple overlap-add synthesis as placeholder
                for frame in 0..n_frames {
                    let frame_start = frame * self.config.hop_size;
                    let frame_end = (frame_start + self.config.frame_size).min(audio_length);

                    // Generate audio frame from mel features
                    for (offset, sample) in audio[frame_start..frame_end].iter_mut().enumerate() {
                        let sample_idx = frame_start + offset;
                        let mut sample_value = 0.0;
                        for mel_idx in 0..n_mel {
                            let mel_value = mel_spec[[mel_idx, frame]];
                            let frequency =
                                mel_idx as f32 * self.config.sample_rate / (2.0 * n_mel as f32);
                            let phase = 2.0 * std::f32::consts::PI * frequency * sample_idx as f32
                                / self.config.sample_rate;
                            sample_value += mel_value * phase.sin() / n_mel as f32;
                        }
                        *sample += sample_value * 0.1; // Scale down
                    }
                }
            }
            _ => {
                // Fallback to simple synthesis
                for (i, sample) in audio.iter_mut().enumerate().take(audio_length) {
                    *sample = (i as f32 / audio_length as f32).sin() * 0.1;
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

    /// Load real [`DiffusionDenoiser`] weights from a safetensors file at
    /// `path`, enabling the neural noise-prediction path (see
    /// [`DiffSingerConfig::use_neural_denoiser`]).
    ///
    /// Fails closed with a typed [`Error::Model`] if `path` doesn't exist or
    /// its tensors don't match the expected architecture - never silently
    /// "succeeds" without reading real weight data.
    fn load_from_file(&mut self, path: &str) -> Result<()> {
        let denoiser = DiffusionDenoiser::load(self.config.n_mel, Device::Cpu, path)?;
        self.neural_denoiser = Some(denoiser);
        Ok(())
    }

    /// Save the currently-loaded [`DiffusionDenoiser`]'s real weights to a
    /// safetensors file at `path`.
    ///
    /// Fails with a typed [`Error::Model`] if no neural denoiser is loaded -
    /// there is nothing real to save, so this never reports a fake success.
    fn save_to_file(&self, path: &str) -> Result<()> {
        let denoiser = self.neural_denoiser.as_ref().ok_or_else(|| {
            Error::Model(
                "No neural denoiser is loaded to save; call `load_from_file` with real weights \
                 first"
                    .to_string(),
            )
        })?;
        denoiser.save(path)
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
            NoiseSchedule::Linear => {}
            _ => panic!("Expected Linear schedule"),
        }

        // cosine_schedule is intentionally unused; keep it to verify the
        // variant is constructible without warnings.
        let _ = cosine_schedule;

        match custom_schedule {
            NoiseSchedule::Custom(ref values) => assert_eq!(values.len(), 3),
            _ => panic!("Expected Custom schedule"),
        }
    }

    // ── New DDPM-related tests ─────────────────────────────────────────────

    /// predict_noise must return a tensor with the same (n_mel, n_frames) shape
    /// as the noisy spectrogram stored in the DiffusionState.
    #[test]
    fn test_predict_noise_output_shape() {
        let model = DiffSingerModel::default();
        let n_mel = 80;
        let n_frames = 60;

        let mut state = DiffusionState::new((n_mel, n_frames), 50);
        state.initialize_noise();

        // Build minimal conditioning maps with different shapes to verify
        // boundary clamping is handled inside predict_noise.
        let mut conditioning: HashMap<String, Array2<f32>> = HashMap::new();
        conditioning.insert("pitch".to_string(), Array2::<f32>::zeros((1, n_frames)));
        conditioning.insert("musical".to_string(), Array2::<f32>::zeros((8, n_frames)));

        let result = model.predict_noise(&state, &conditioning, 5).unwrap();
        assert_eq!(result.dim(), (n_mel, n_frames));
    }

    /// predict_noise must never produce NaN values regardless of the
    /// conditioning input or the diffusion step.
    #[test]
    fn test_predict_noise_no_nan() {
        let model = DiffSingerModel::default();
        let n_mel = 40;
        let n_frames = 30;

        let mut state = DiffusionState::new((n_mel, n_frames), 50);
        state.initialize_noise();

        let mut conditioning: HashMap<String, Array2<f32>> = HashMap::new();
        conditioning.insert("pitch".to_string(), Array2::<f32>::ones((n_mel, n_frames)));
        conditioning.insert(
            "musical".to_string(),
            Array2::<f32>::from_elem((n_mel, n_frames), 0.3),
        );

        // Test across multiple steps, including the boundary step 0.
        for &step in &[0usize, 1, 25, 49] {
            let result = model.predict_noise(&state, &conditioning, step).unwrap();
            for &v in result.iter() {
                assert!(
                    v.is_finite(),
                    "NaN or Inf detected at step {step}: value = {v}"
                );
            }
        }
    }

    /// Verify the linear beta schedule properties:
    /// - At step 0 (very first), ᾱ is close to 1 (almost no noise has been added).
    /// - At step T-1 (last step), ᾱ is close to 0 (nearly all signal is noise).
    #[test]
    fn test_alpha_bar_schedule() {
        let total_steps = 1000_usize;

        let alpha_bar_first = compute_alpha_bar(0, total_steps);
        // β_0 ≈ 1e-4, so ᾱ_0 = 1 - 1e-4 ≈ 0.9999
        assert!(
            alpha_bar_first > 0.99,
            "ᾱ at step 0 should be close to 1, got {alpha_bar_first}"
        );

        let alpha_bar_last = compute_alpha_bar(total_steps - 1, total_steps);
        // After 1000 multiplications by (1-β_t), the product approaches 0.
        assert!(
            alpha_bar_last < 0.05,
            "ᾱ at step T-1 should be close to 0, got {alpha_bar_last}"
        );

        // Monotonically decreasing property.
        let alpha_bar_mid = compute_alpha_bar(total_steps / 2, total_steps);
        assert!(
            alpha_bar_mid < alpha_bar_first,
            "ᾱ should decrease: first={alpha_bar_first} mid={alpha_bar_mid}"
        );
        assert!(
            alpha_bar_last < alpha_bar_mid,
            "ᾱ should decrease: mid={alpha_bar_mid} last={alpha_bar_last}"
        );
    }

    /// For two distinct conditioning inputs, predict_noise must return
    /// different outputs (the function must be sensitive to conditioning).
    #[test]
    fn test_predict_noise_not_constant() {
        let model = DiffSingerModel::default();
        let n_mel = 20;
        let n_frames = 15;

        let mut state = DiffusionState::new((n_mel, n_frames), 50);
        state.initialize_noise();

        // Conditioning set A: pitch all zeros.
        let mut cond_a: HashMap<String, Array2<f32>> = HashMap::new();
        cond_a.insert("pitch".to_string(), Array2::<f32>::zeros((n_mel, n_frames)));

        // Conditioning set B: pitch all ones (significantly different).
        let mut cond_b: HashMap<String, Array2<f32>> = HashMap::new();
        cond_b.insert("pitch".to_string(), Array2::<f32>::ones((n_mel, n_frames)));

        let result_a = model.predict_noise(&state, &cond_a, 10).unwrap();
        let result_b = model.predict_noise(&state, &cond_b, 10).unwrap();

        // At least one element must differ between the two predictions.
        let any_different = result_a
            .iter()
            .zip(result_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            any_different,
            "predict_noise returned identical outputs for different conditioning inputs"
        );
    }
}
