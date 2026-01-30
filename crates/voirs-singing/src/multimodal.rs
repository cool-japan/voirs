//! Multimodal Singing Synthesis - Audio-Visual Integration
//!
//! This module provides synchronization between audio synthesis and visual parameters
//! for creating realistic singing avatars, virtual performers, and lip-synced animations.
//!
//! # Features
//!
//! - **Phoneme-to-Viseme Mapping**: Convert phonemes to facial animation parameters
//! - **Emotion-Driven Facial Animation**: Synchronize emotional expressions with singing
//! - **Timeline Synchronization**: Precise alignment between audio and visual frames
//! - **Blend Shape Support**: Generate standard blend shape weights for 3D avatars
//! - **Performance Capture Integration**: Real-time visual parameter generation
//!
//! # Example
//!
//! ```rust,no_run
//! use voirs_singing::multimodal::{MultimodalSynthesizer, VisualConfig, PhonemeTiming, SimpleNote};
//! use std::path::Path;
//!
//! # async fn example() -> voirs_singing::Result<()> {
//! let config = VisualConfig::default();
//! let mut synthesizer = MultimodalSynthesizer::new(config)?;
//!
//! let notes = vec![
//!     SimpleNote { pitch: 60, duration: 1.0, velocity: 100, onset_time: 0.0, lyric: None }
//! ];
//! let phonemes = vec![
//!     PhonemeTiming { phoneme: "a".to_string(), start_time: 0.0, duration: 0.3 },
//!     PhonemeTiming { phoneme: "i".to_string(), start_time: 0.3, duration: 0.4 },
//!     PhonemeTiming { phoneme: "u".to_string(), start_time: 0.7, duration: 0.3 },
//! ];
//!
//! // Generate synchronized audio-visual parameters
//! let result = synthesizer.synthesize_multimodal(&notes, &phonemes).await?;
//!
//! // Export visual animation
//! synthesizer.export_animation(&result.visual_timeline, Path::new("output.json"))?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::emotion_transfer::EmotionVector;
use crate::MusicalNote;

/// Simple musical note representation for visual synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleNote {
    /// MIDI pitch value
    pub pitch: u8,
    /// Duration in seconds
    pub duration: f32,
    /// Velocity (0-127)
    pub velocity: u8,
    /// Onset time in seconds
    pub onset_time: f32,
    /// Optional lyric
    pub lyric: Option<String>,
}

impl SimpleNote {
    /// Convert from MusicalNote
    pub fn from_musical_note(note: &MusicalNote, sample_rate: f32) -> Self {
        // Extract pitch from frequency (A4 = 440Hz = MIDI 69)
        let frequency = note.event.frequency;
        let pitch = (69.0 + 12.0 * (frequency / 440.0).log2()).round() as u8;

        Self {
            pitch,
            duration: note.duration,
            velocity: (note.event.velocity * 127.0) as u8,
            onset_time: note.start_time,
            lyric: note.event.lyric.clone(),
        }
    }
}

/// Configuration for visual parameter generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConfig {
    /// Frame rate for visual timeline (typically 30 or 60 fps)
    pub frame_rate: f32,

    /// Enable advanced facial micro-expressions
    pub enable_micro_expressions: bool,

    /// Blend shape smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
    pub smoothing_factor: f32,

    /// Enable eye blink generation
    pub enable_eye_blinks: bool,

    /// Blink frequency in blinks per minute
    pub blink_frequency: f32,

    /// Enable head motion generation
    pub enable_head_motion: bool,

    /// Intensity of automatic head movements (0.0 to 1.0)
    pub head_motion_intensity: f32,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            frame_rate: 60.0,
            enable_micro_expressions: true,
            smoothing_factor: 0.3,
            enable_eye_blinks: true,
            blink_frequency: 17.0, // Average human blink rate
            enable_head_motion: true,
            head_motion_intensity: 0.5,
        }
    }
}

/// Viseme representation for lip sync
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Viseme {
    /// Silence (mouth closed)
    Silence,
    /// /a/ sound (open mouth)
    A,
    /// /e/ sound
    E,
    /// /i/ sound (wide smile)
    I,
    /// /o/ sound (rounded lips)
    O,
    /// /u/ sound (pursed lips)
    U,
    /// Consonants: /p/, /b/, /m/ (lips together)
    BilabialStop,
    /// Consonants: /f/, /v/ (lip-teeth contact)
    Labiodental,
    /// Consonants: /th/ (tongue-teeth)
    Dental,
    /// Consonants: /t/, /d/, /n/, /l/
    Alveolar,
    /// Consonants: /k/, /g/
    Velar,
    /// Consonants: /r/
    Rhotic,
    /// Consonants: /s/, /z/
    Sibilant,
}

impl Viseme {
    /// Convert phoneme string to viseme
    pub fn from_phoneme(phoneme: &str) -> Self {
        match phoneme.to_lowercase().as_str() {
            "a" | "aa" | "ah" => Viseme::A,
            "e" | "eh" | "ey" => Viseme::E,
            "i" | "ih" | "iy" => Viseme::I,
            "o" | "oh" | "ow" => Viseme::O,
            "u" | "uh" | "uw" => Viseme::U,
            "p" | "b" | "m" => Viseme::BilabialStop,
            "f" | "v" => Viseme::Labiodental,
            "th" | "dh" => Viseme::Dental,
            "t" | "d" | "n" | "l" => Viseme::Alveolar,
            "k" | "g" | "ng" => Viseme::Velar,
            "r" => Viseme::Rhotic,
            "s" | "z" | "sh" | "zh" => Viseme::Sibilant,
            "sil" | "pau" => Viseme::Silence,
            _ => Viseme::Silence,
        }
    }
}

/// Blend shape weights for facial animation (ARKit-compatible)
///
/// All values range from 0.0 (no deformation) to 1.0 (full deformation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendShapeWeights {
    // Mouth shapes
    /// Jaw open amount (0.0 = closed, 1.0 = fully open)
    pub jaw_open: f32,
    /// Mouth close tightness (0.0 = relaxed, 1.0 = pressed together)
    pub mouth_close: f32,
    /// Mouth funnel/pucker outward (O shape)
    pub mouth_funnel: f32,
    /// Lips pucker inward (kiss shape)
    pub mouth_pucker: f32,
    /// Left side smile intensity
    pub mouth_smile_left: f32,
    /// Right side smile intensity
    pub mouth_smile_right: f32,
    /// Left side frown intensity
    pub mouth_frown_left: f32,
    /// Right side frown intensity
    pub mouth_frown_right: f32,
    /// Left side mouth stretch (wide smile)
    pub mouth_stretch_left: f32,
    /// Right side mouth stretch (wide smile)
    pub mouth_stretch_right: f32,

    // Eye shapes
    /// Left eye blink (0.0 = open, 1.0 = closed)
    pub eye_blink_left: f32,
    /// Right eye blink (0.0 = open, 1.0 = closed)
    pub eye_blink_right: f32,
    /// Left eye wide open intensity
    pub eye_wide_left: f32,
    /// Right eye wide open intensity
    pub eye_wide_right: f32,

    // Eyebrow shapes
    /// Inner eyebrow raise (both sides together)
    pub brow_inner_up: f32,
    /// Left outer eyebrow raise
    pub brow_outer_up_left: f32,
    /// Right outer eyebrow raise
    pub brow_outer_up_right: f32,
    /// Left eyebrow lower/furrow
    pub brow_down_left: f32,
    /// Right eyebrow lower/furrow
    pub brow_down_right: f32,

    // Cheek shapes
    /// Cheek puff (blow air into cheeks)
    pub cheek_puff: f32,
    /// Left cheek squint/raise
    pub cheek_squint_left: f32,
    /// Right cheek squint/raise
    pub cheek_squint_right: f32,
}

impl Default for BlendShapeWeights {
    fn default() -> Self {
        Self {
            jaw_open: 0.0,
            mouth_close: 1.0,
            mouth_funnel: 0.0,
            mouth_pucker: 0.0,
            mouth_smile_left: 0.0,
            mouth_smile_right: 0.0,
            mouth_frown_left: 0.0,
            mouth_frown_right: 0.0,
            mouth_stretch_left: 0.0,
            mouth_stretch_right: 0.0,
            eye_blink_left: 0.0,
            eye_blink_right: 0.0,
            eye_wide_left: 0.0,
            eye_wide_right: 0.0,
            brow_inner_up: 0.0,
            brow_outer_up_left: 0.0,
            brow_outer_up_right: 0.0,
            brow_down_left: 0.0,
            brow_down_right: 0.0,
            cheek_puff: 0.0,
            cheek_squint_left: 0.0,
            cheek_squint_right: 0.0,
        }
    }
}

impl BlendShapeWeights {
    /// Generate blend shapes from viseme
    pub fn from_viseme(viseme: Viseme) -> Self {
        let mut weights = Self::default();

        match viseme {
            Viseme::Silence => {
                weights.mouth_close = 1.0;
            }
            Viseme::A => {
                weights.jaw_open = 0.8;
                weights.mouth_close = 0.0;
            }
            Viseme::E => {
                weights.jaw_open = 0.4;
                weights.mouth_smile_left = 0.5;
                weights.mouth_smile_right = 0.5;
                weights.mouth_close = 0.0;
            }
            Viseme::I => {
                weights.jaw_open = 0.2;
                weights.mouth_smile_left = 0.8;
                weights.mouth_smile_right = 0.8;
                weights.mouth_stretch_left = 0.5;
                weights.mouth_stretch_right = 0.5;
                weights.mouth_close = 0.0;
            }
            Viseme::O => {
                weights.jaw_open = 0.6;
                weights.mouth_funnel = 0.7;
                weights.mouth_close = 0.0;
            }
            Viseme::U => {
                weights.jaw_open = 0.3;
                weights.mouth_pucker = 0.8;
                weights.mouth_funnel = 0.4;
                weights.mouth_close = 0.0;
            }
            Viseme::BilabialStop => {
                weights.mouth_close = 1.0;
                weights.jaw_open = 0.0;
            }
            Viseme::Labiodental => {
                weights.jaw_open = 0.2;
                weights.mouth_close = 0.3;
            }
            Viseme::Dental => {
                weights.jaw_open = 0.3;
                weights.mouth_close = 0.1;
            }
            Viseme::Alveolar => {
                weights.jaw_open = 0.3;
                weights.mouth_close = 0.2;
            }
            Viseme::Velar => {
                weights.jaw_open = 0.4;
                weights.mouth_close = 0.0;
            }
            Viseme::Rhotic => {
                weights.jaw_open = 0.35;
                weights.mouth_funnel = 0.3;
                weights.mouth_close = 0.0;
            }
            Viseme::Sibilant => {
                weights.jaw_open = 0.2;
                weights.mouth_smile_left = 0.3;
                weights.mouth_smile_right = 0.3;
                weights.mouth_close = 0.0;
            }
        }

        weights
    }

    /// Apply emotion modulation to blend shapes
    pub fn apply_emotion(&mut self, emotion: &EmotionVector) {
        // Valence affects smile/frown
        if emotion.valence > 0.0 {
            let smile_strength = emotion.valence * emotion.intensity;
            self.mouth_smile_left += smile_strength * 0.5;
            self.mouth_smile_right += smile_strength * 0.5;
            self.cheek_squint_left += smile_strength * 0.3;
            self.cheek_squint_right += smile_strength * 0.3;
        } else {
            let frown_strength = (-emotion.valence) * emotion.intensity;
            self.mouth_frown_left += frown_strength * 0.4;
            self.mouth_frown_right += frown_strength * 0.4;
        }

        // Arousal affects eye openness and eyebrows
        if emotion.arousal > 0.5 {
            let excitement = (emotion.arousal - 0.5) * 2.0 * emotion.intensity;
            self.eye_wide_left += excitement * 0.4;
            self.eye_wide_right += excitement * 0.4;
            self.brow_inner_up += excitement * 0.3;
            self.brow_outer_up_left += excitement * 0.3;
            self.brow_outer_up_right += excitement * 0.3;
        }

        // Dominance affects jaw tension
        if emotion.dominance > 0.5 {
            let tension = (emotion.dominance - 0.5) * 2.0 * emotion.intensity;
            self.jaw_open *= 1.0 - (tension * 0.3);
        }

        // Clamp all values to valid range
        self.clamp_weights();
    }

    /// Clamp all weights to [0.0, 1.0] range
    fn clamp_weights(&mut self) {
        self.jaw_open = self.jaw_open.clamp(0.0, 1.0);
        self.mouth_close = self.mouth_close.clamp(0.0, 1.0);
        self.mouth_funnel = self.mouth_funnel.clamp(0.0, 1.0);
        self.mouth_pucker = self.mouth_pucker.clamp(0.0, 1.0);
        self.mouth_smile_left = self.mouth_smile_left.clamp(0.0, 1.0);
        self.mouth_smile_right = self.mouth_smile_right.clamp(0.0, 1.0);
        self.mouth_frown_left = self.mouth_frown_left.clamp(0.0, 1.0);
        self.mouth_frown_right = self.mouth_frown_right.clamp(0.0, 1.0);
        self.mouth_stretch_left = self.mouth_stretch_left.clamp(0.0, 1.0);
        self.mouth_stretch_right = self.mouth_stretch_right.clamp(0.0, 1.0);
        self.eye_blink_left = self.eye_blink_left.clamp(0.0, 1.0);
        self.eye_blink_right = self.eye_blink_right.clamp(0.0, 1.0);
        self.eye_wide_left = self.eye_wide_left.clamp(0.0, 1.0);
        self.eye_wide_right = self.eye_wide_right.clamp(0.0, 1.0);
        self.brow_inner_up = self.brow_inner_up.clamp(0.0, 1.0);
        self.brow_outer_up_left = self.brow_outer_up_left.clamp(0.0, 1.0);
        self.brow_outer_up_right = self.brow_outer_up_right.clamp(0.0, 1.0);
        self.brow_down_left = self.brow_down_left.clamp(0.0, 1.0);
        self.brow_down_right = self.brow_down_right.clamp(0.0, 1.0);
        self.cheek_puff = self.cheek_puff.clamp(0.0, 1.0);
        self.cheek_squint_left = self.cheek_squint_left.clamp(0.0, 1.0);
        self.cheek_squint_right = self.cheek_squint_right.clamp(0.0, 1.0);
    }

    /// Interpolate between two blend shape sets
    pub fn lerp(&self, other: &Self, alpha: f32) -> Self {
        let t = alpha.clamp(0.0, 1.0);
        Self {
            jaw_open: self.jaw_open + (other.jaw_open - self.jaw_open) * t,
            mouth_close: self.mouth_close + (other.mouth_close - self.mouth_close) * t,
            mouth_funnel: self.mouth_funnel + (other.mouth_funnel - self.mouth_funnel) * t,
            mouth_pucker: self.mouth_pucker + (other.mouth_pucker - self.mouth_pucker) * t,
            mouth_smile_left: self.mouth_smile_left
                + (other.mouth_smile_left - self.mouth_smile_left) * t,
            mouth_smile_right: self.mouth_smile_right
                + (other.mouth_smile_right - self.mouth_smile_right) * t,
            mouth_frown_left: self.mouth_frown_left
                + (other.mouth_frown_left - self.mouth_frown_left) * t,
            mouth_frown_right: self.mouth_frown_right
                + (other.mouth_frown_right - self.mouth_frown_right) * t,
            mouth_stretch_left: self.mouth_stretch_left
                + (other.mouth_stretch_left - self.mouth_stretch_left) * t,
            mouth_stretch_right: self.mouth_stretch_right
                + (other.mouth_stretch_right - self.mouth_stretch_right) * t,
            eye_blink_left: self.eye_blink_left + (other.eye_blink_left - self.eye_blink_left) * t,
            eye_blink_right: self.eye_blink_right
                + (other.eye_blink_right - self.eye_blink_right) * t,
            eye_wide_left: self.eye_wide_left + (other.eye_wide_left - self.eye_wide_left) * t,
            eye_wide_right: self.eye_wide_right + (other.eye_wide_right - self.eye_wide_right) * t,
            brow_inner_up: self.brow_inner_up + (other.brow_inner_up - self.brow_inner_up) * t,
            brow_outer_up_left: self.brow_outer_up_left
                + (other.brow_outer_up_left - self.brow_outer_up_left) * t,
            brow_outer_up_right: self.brow_outer_up_right
                + (other.brow_outer_up_right - self.brow_outer_up_right) * t,
            brow_down_left: self.brow_down_left + (other.brow_down_left - self.brow_down_left) * t,
            brow_down_right: self.brow_down_right
                + (other.brow_down_right - self.brow_down_right) * t,
            cheek_puff: self.cheek_puff + (other.cheek_puff - self.cheek_puff) * t,
            cheek_squint_left: self.cheek_squint_left
                + (other.cheek_squint_left - self.cheek_squint_left) * t,
            cheek_squint_right: self.cheek_squint_right
                + (other.cheek_squint_right - self.cheek_squint_right) * t,
        }
    }
}

/// Head motion parameters for natural animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMotion {
    /// Rotation around X axis (pitch: up/down)
    pub pitch: f32,
    /// Rotation around Y axis (yaw: left/right)
    pub yaw: f32,
    /// Rotation around Z axis (roll: tilt)
    pub roll: f32,
}

impl Default for HeadMotion {
    fn default() -> Self {
        Self {
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
        }
    }
}

impl HeadMotion {
    /// Generate subtle head motion from musical features
    pub fn from_musical_context(note: &SimpleNote, time: f32, intensity: f32) -> Self {
        use std::f32::consts::PI;

        // Pitch motion follows melodic contour
        let pitch_motion = (note.pitch as f32 - 60.0) * 0.001 * intensity;

        // Yaw motion has slow periodic variation
        let yaw_motion = (time * 0.5).sin() * 2.0 * intensity;

        // Roll motion follows rhythm
        let roll_motion = (time * note.duration * 2.0 * PI).sin() * 1.0 * intensity;

        Self {
            pitch: pitch_motion.clamp(-5.0, 5.0),
            yaw: yaw_motion.clamp(-10.0, 10.0),
            roll: roll_motion.clamp(-3.0, 3.0),
        }
    }
}

/// Visual frame with all animation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFrame {
    /// Timestamp in seconds
    pub timestamp: f32,

    /// Blend shape weights for this frame
    pub blend_shapes: BlendShapeWeights,

    /// Head motion parameters
    pub head_motion: HeadMotion,

    /// Current viseme being displayed
    pub viseme: Viseme,
}

/// Timeline of visual frames synchronized with audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualTimeline {
    /// Frame rate (frames per second)
    pub frame_rate: f32,

    /// Sequence of visual frames
    pub frames: Vec<VisualFrame>,

    /// Total duration in seconds
    pub duration: f32,
}

impl VisualTimeline {
    /// Create new empty timeline
    pub fn new(frame_rate: f32) -> Self {
        Self {
            frame_rate,
            frames: Vec::new(),
            duration: 0.0,
        }
    }

    /// Get frame at specific timestamp using interpolation
    pub fn get_frame_at(&self, timestamp: f32) -> Option<VisualFrame> {
        if self.frames.is_empty() || timestamp < 0.0 || timestamp > self.duration {
            return None;
        }

        // Find frames surrounding the timestamp
        let mut prev_index = 0;
        let mut next_index = 0;

        for (i, frame) in self.frames.iter().enumerate() {
            if frame.timestamp <= timestamp {
                prev_index = i;
            }
            if frame.timestamp >= timestamp {
                next_index = i;
                break;
            }
        }

        // If we're exactly on a frame timestamp
        if prev_index == next_index {
            return Some(self.frames[prev_index].clone());
        }

        // Interpolate between frames
        let prev_frame = &self.frames[prev_index];
        let next_frame = &self.frames[next_index];
        let alpha =
            (timestamp - prev_frame.timestamp) / (next_frame.timestamp - prev_frame.timestamp);

        Some(VisualFrame {
            timestamp,
            blend_shapes: prev_frame
                .blend_shapes
                .lerp(&next_frame.blend_shapes, alpha),
            head_motion: HeadMotion {
                pitch: prev_frame.head_motion.pitch
                    + (next_frame.head_motion.pitch - prev_frame.head_motion.pitch) * alpha,
                yaw: prev_frame.head_motion.yaw
                    + (next_frame.head_motion.yaw - prev_frame.head_motion.yaw) * alpha,
                roll: prev_frame.head_motion.roll
                    + (next_frame.head_motion.roll - prev_frame.head_motion.roll) * alpha,
            },
            viseme: prev_frame.viseme,
        })
    }
}

/// Result of multimodal synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalResult {
    /// Audio samples
    pub audio: Vec<f32>,

    /// Visual animation timeline
    pub visual_timeline: VisualTimeline,

    /// Sample rate for audio
    pub sample_rate: u32,
}

/// Phoneme timing information
#[derive(Debug, Clone)]
pub struct PhonemeTiming {
    /// Phoneme string
    pub phoneme: String,

    /// Start time in seconds
    pub start_time: f32,

    /// Duration in seconds
    pub duration: f32,
}

/// Multimodal synthesizer for audio-visual singing
pub struct MultimodalSynthesizer {
    config: VisualConfig,
    blink_timer: f32,
}

impl MultimodalSynthesizer {
    /// Create new multimodal synthesizer
    pub fn new(config: VisualConfig) -> crate::Result<Self> {
        Ok(Self {
            config,
            blink_timer: 0.0,
        })
    }

    /// Synthesize multimodal output from notes and phonemes
    pub async fn synthesize_multimodal(
        &mut self,
        notes: &[SimpleNote],
        phoneme_timings: &[PhonemeTiming],
    ) -> crate::Result<MultimodalResult> {
        // Calculate total duration
        let total_duration = phoneme_timings
            .iter()
            .map(|pt| pt.start_time + pt.duration)
            .fold(0.0f32, |a, b| a.max(b));

        // Generate visual timeline
        let visual_timeline =
            self.generate_visual_timeline(notes, phoneme_timings, total_duration)?;

        // Generate placeholder audio (in real implementation, this would call actual synthesis)
        let sample_rate = 22050;
        let audio_samples = (total_duration * sample_rate as f32) as usize;
        let audio = vec![0.0; audio_samples];

        Ok(MultimodalResult {
            audio,
            visual_timeline,
            sample_rate,
        })
    }

    /// Generate visual timeline from phoneme timings
    fn generate_visual_timeline(
        &mut self,
        notes: &[SimpleNote],
        phoneme_timings: &[PhonemeTiming],
        duration: f32,
    ) -> crate::Result<VisualTimeline> {
        let mut timeline = VisualTimeline::new(self.config.frame_rate);
        timeline.duration = duration;

        let frame_duration = 1.0 / self.config.frame_rate;
        let num_frames = (duration * self.config.frame_rate).ceil() as usize;

        let mut current_time = 0.0;
        let mut prev_blend_shapes = BlendShapeWeights::default();

        for _ in 0..num_frames {
            // Find current phoneme
            let current_phoneme = phoneme_timings.iter().find(|pt| {
                current_time >= pt.start_time && current_time < pt.start_time + pt.duration
            });

            // Get viseme for current phoneme
            let viseme = current_phoneme
                .map(|pt| Viseme::from_phoneme(&pt.phoneme))
                .unwrap_or(Viseme::Silence);

            // Generate blend shapes from viseme
            let mut blend_shapes = BlendShapeWeights::from_viseme(viseme);

            // Apply emotion if available (placeholder - would integrate with emotion_transfer)
            let neutral_emotion = EmotionVector::neutral();
            blend_shapes.apply_emotion(&neutral_emotion);

            // Apply smoothing
            if self.config.smoothing_factor > 0.0 {
                blend_shapes =
                    prev_blend_shapes.lerp(&blend_shapes, 1.0 - self.config.smoothing_factor);
            }

            // Generate eye blinks
            if self.config.enable_eye_blinks {
                self.apply_eye_blinks(&mut blend_shapes, current_time);
            }

            // Generate head motion
            let head_motion = if self.config.enable_head_motion {
                let default_note = SimpleNote {
                    pitch: 60,
                    duration: 0.5,
                    velocity: 80,
                    onset_time: 0.0,
                    lyric: None,
                };
                let note = notes.first().unwrap_or(&default_note);
                HeadMotion::from_musical_context(
                    note,
                    current_time,
                    self.config.head_motion_intensity,
                )
            } else {
                HeadMotion::default()
            };

            timeline.frames.push(VisualFrame {
                timestamp: current_time,
                blend_shapes: blend_shapes.clone(),
                head_motion,
                viseme,
            });

            prev_blend_shapes = blend_shapes;
            current_time += frame_duration;
        }

        Ok(timeline)
    }

    /// Apply automatic eye blinks at natural intervals
    fn apply_eye_blinks(&mut self, blend_shapes: &mut BlendShapeWeights, current_time: f32) {
        let blink_interval = 60.0 / self.config.blink_frequency; // Convert to seconds

        // Check if it's time for a blink
        if current_time - self.blink_timer >= blink_interval {
            self.blink_timer = current_time;
        }

        // Apply blink animation (lasts ~0.15 seconds)
        let time_since_blink = current_time - self.blink_timer;
        if time_since_blink < 0.15 {
            // Blink curve (quick close, quick open)
            let blink_progress = time_since_blink / 0.15;
            let blink_curve = if blink_progress < 0.5 {
                blink_progress * 2.0
            } else {
                2.0 - (blink_progress * 2.0)
            };

            blend_shapes.eye_blink_left = blink_curve;
            blend_shapes.eye_blink_right = blink_curve;
        }
    }

    /// Export animation timeline to JSON file
    pub fn export_animation(&self, timeline: &VisualTimeline, path: &Path) -> crate::Result<()> {
        let json = serde_json::to_string_pretty(timeline).map_err(crate::Error::Serialization)?;

        std::fs::write(path, json).map_err(crate::Error::Io)?;

        Ok(())
    }

    /// Import animation timeline from JSON file
    pub fn import_animation(&self, path: &Path) -> crate::Result<VisualTimeline> {
        let json = std::fs::read_to_string(path).map_err(crate::Error::Io)?;

        let timeline = serde_json::from_str(&json).map_err(crate::Error::Serialization)?;

        Ok(timeline)
    }

    /// Generate phoneme timings from simple notes (simplified version)
    pub fn generate_phoneme_timings(&self, notes: &[SimpleNote]) -> Vec<PhonemeTiming> {
        let mut timings = Vec::new();
        let mut current_time = 0.0;

        for note in notes {
            // Use lyric if available, otherwise use default vowel
            let phoneme = note.lyric.as_deref().unwrap_or("a").to_string();

            timings.push(PhonemeTiming {
                phoneme,
                start_time: current_time,
                duration: note.duration,
            });

            current_time += note.duration;
        }

        timings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viseme_from_phoneme() {
        assert_eq!(Viseme::from_phoneme("a"), Viseme::A);
        assert_eq!(Viseme::from_phoneme("i"), Viseme::I);
        assert_eq!(Viseme::from_phoneme("p"), Viseme::BilabialStop);
        assert_eq!(Viseme::from_phoneme("s"), Viseme::Sibilant);
        assert_eq!(Viseme::from_phoneme("sil"), Viseme::Silence);
    }

    #[test]
    fn test_blend_shapes_from_viseme() {
        let shapes_a = BlendShapeWeights::from_viseme(Viseme::A);
        assert!(shapes_a.jaw_open > 0.5);
        assert_eq!(shapes_a.mouth_close, 0.0);

        let shapes_i = BlendShapeWeights::from_viseme(Viseme::I);
        assert!(shapes_i.mouth_smile_left > 0.5);
        assert!(shapes_i.mouth_smile_right > 0.5);

        let shapes_silence = BlendShapeWeights::from_viseme(Viseme::Silence);
        assert_eq!(shapes_silence.mouth_close, 1.0);
    }

    #[test]
    fn test_blend_shapes_emotion_modulation() {
        let mut shapes = BlendShapeWeights::from_viseme(Viseme::A);
        let happy = EmotionVector::happy();

        shapes.apply_emotion(&happy);

        // Happy emotion should increase smile
        assert!(shapes.mouth_smile_left > 0.0);
        assert!(shapes.mouth_smile_right > 0.0);

        // All weights should be in valid range
        assert!(shapes.jaw_open >= 0.0 && shapes.jaw_open <= 1.0);
        assert!(shapes.mouth_smile_left >= 0.0 && shapes.mouth_smile_left <= 1.0);
    }

    #[test]
    fn test_blend_shapes_interpolation() {
        let shapes_a = BlendShapeWeights::from_viseme(Viseme::A);
        let shapes_i = BlendShapeWeights::from_viseme(Viseme::I);

        let interpolated = shapes_a.lerp(&shapes_i, 0.5);

        // Interpolated values should be between the two extremes
        assert!(interpolated.jaw_open > shapes_i.jaw_open);
        assert!(interpolated.jaw_open < shapes_a.jaw_open);
        assert!(interpolated.mouth_smile_left > shapes_a.mouth_smile_left);
        assert!(interpolated.mouth_smile_left < shapes_i.mouth_smile_left);
    }

    #[test]
    fn test_head_motion_generation() {
        let note = SimpleNote {
            pitch: 60,
            duration: 0.5,
            velocity: 80,
            onset_time: 0.0,
            lyric: Some("test".to_string()),
        };

        let motion = HeadMotion::from_musical_context(&note, 1.0, 0.5);

        // Motion values should be within reasonable ranges
        assert!(motion.pitch.abs() <= 5.0);
        assert!(motion.yaw.abs() <= 10.0);
        assert!(motion.roll.abs() <= 3.0);
    }

    #[tokio::test]
    async fn test_multimodal_synthesis() {
        let config = VisualConfig::default();
        let mut synthesizer = MultimodalSynthesizer::new(config).unwrap();

        let notes = vec![
            SimpleNote {
                pitch: 60,
                duration: 0.5,
                velocity: 80,
                onset_time: 0.0,
                lyric: Some("a".to_string()),
            },
            SimpleNote {
                pitch: 64,
                duration: 0.5,
                velocity: 80,
                onset_time: 0.5,
                lyric: Some("i".to_string()),
            },
        ];

        let phoneme_timings = vec![
            PhonemeTiming {
                phoneme: "a".to_string(),
                start_time: 0.0,
                duration: 0.5,
            },
            PhonemeTiming {
                phoneme: "i".to_string(),
                start_time: 0.5,
                duration: 0.5,
            },
        ];

        let result = synthesizer
            .synthesize_multimodal(&notes, &phoneme_timings)
            .await
            .unwrap();

        assert_eq!(result.visual_timeline.frame_rate, 60.0);
        assert!(!result.visual_timeline.frames.is_empty());
        assert_eq!(result.sample_rate, 22050);
    }

    #[test]
    fn test_visual_timeline_frame_interpolation() {
        let mut timeline = VisualTimeline::new(30.0);
        timeline.duration = 1.0;

        // Add two frames
        timeline.frames.push(VisualFrame {
            timestamp: 0.0,
            blend_shapes: BlendShapeWeights::from_viseme(Viseme::A),
            head_motion: HeadMotion::default(),
            viseme: Viseme::A,
        });

        timeline.frames.push(VisualFrame {
            timestamp: 0.5,
            blend_shapes: BlendShapeWeights::from_viseme(Viseme::I),
            head_motion: HeadMotion::default(),
            viseme: Viseme::I,
        });

        // Get interpolated frame at midpoint
        let frame = timeline.get_frame_at(0.25).unwrap();

        // Should be between the two original frames
        assert!(frame.blend_shapes.jaw_open < timeline.frames[0].blend_shapes.jaw_open);
        assert!(frame.blend_shapes.jaw_open > timeline.frames[1].blend_shapes.jaw_open);
    }

    #[test]
    fn test_phoneme_timing_generation() {
        let config = VisualConfig::default();
        let synthesizer = MultimodalSynthesizer::new(config).unwrap();

        let notes = vec![
            SimpleNote {
                pitch: 60,
                duration: 0.5,
                velocity: 80,
                onset_time: 0.0,
                lyric: Some("hello".to_string()),
            },
            SimpleNote {
                pitch: 64,
                duration: 0.3,
                velocity: 80,
                onset_time: 0.5,
                lyric: Some("world".to_string()),
            },
        ];

        let timings = synthesizer.generate_phoneme_timings(&notes);

        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].phoneme, "hello");
        assert_eq!(timings[0].start_time, 0.0);
        assert_eq!(timings[0].duration, 0.5);
        assert_eq!(timings[1].phoneme, "world");
        assert_eq!(timings[1].start_time, 0.5);
        assert_eq!(timings[1].duration, 0.3);
    }

    #[tokio::test]
    async fn test_export_import_animation() {
        use std::env;

        let config = VisualConfig::default();
        let mut synthesizer = MultimodalSynthesizer::new(config).unwrap();

        let notes = vec![SimpleNote {
            pitch: 60,
            duration: 0.5,
            velocity: 80,
            onset_time: 0.0,
            lyric: Some("test".to_string()),
        }];

        let phoneme_timings = vec![PhonemeTiming {
            phoneme: "a".to_string(),
            start_time: 0.0,
            duration: 0.5,
        }];

        let result = synthesizer
            .synthesize_multimodal(&notes, &phoneme_timings)
            .await
            .unwrap();

        // Export to temporary file
        let temp_dir = env::temp_dir();
        let export_path = temp_dir.join("test_animation.json");

        synthesizer
            .export_animation(&result.visual_timeline, &export_path)
            .unwrap();

        // Import back
        let imported = synthesizer.import_animation(&export_path).unwrap();

        assert_eq!(imported.frame_rate, result.visual_timeline.frame_rate);
        assert_eq!(imported.frames.len(), result.visual_timeline.frames.len());

        // Cleanup
        std::fs::remove_file(export_path).ok();
    }

    #[test]
    fn test_visual_config_defaults() {
        let config = VisualConfig::default();

        assert_eq!(config.frame_rate, 60.0);
        assert!(config.enable_micro_expressions);
        assert!(config.enable_eye_blinks);
        assert!(config.enable_head_motion);
        assert!(config.blink_frequency > 0.0);
        assert!(config.smoothing_factor >= 0.0 && config.smoothing_factor <= 1.0);
    }

    #[test]
    fn test_blend_shape_weight_clamping() {
        let mut shapes = BlendShapeWeights::default();

        // Set values outside valid range
        shapes.jaw_open = 1.5;
        shapes.mouth_smile_left = -0.5;

        shapes.clamp_weights();

        // Should be clamped to [0.0, 1.0]
        assert_eq!(shapes.jaw_open, 1.0);
        assert_eq!(shapes.mouth_smile_left, 0.0);
    }
}
