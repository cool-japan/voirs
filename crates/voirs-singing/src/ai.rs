//! # AI-Driven Features for Singing Voice Synthesis
//!
//! This module provides AI-powered features including style transfer, automatic harmonization,
//! improvisation assistance, and emotion recognition for singing voice synthesis.

use crate::{
    styles::{MusicalStyle, StyleCharacteristics},
    types::{NoteEvent, VoiceCharacteristics},
    Error, Result,
};
use candle_core::Device;
use std::collections::HashMap;

/// Expression features for emotion recognition
#[derive(Debug, Clone)]
pub struct ExpressionFeatures {
    /// Volume level (0.0-1.0)
    pub volume: f32,
    /// Vibrato rate in Hz
    pub vibrato_rate: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
    /// Tremolo rate in Hz
    pub tremolo_rate: f32,
    /// Tremolo depth (0.0-1.0)
    pub tremolo_depth: f32,
}

impl Default for ExpressionFeatures {
    fn default() -> Self {
        Self {
            volume: 0.7,
            vibrato_rate: 5.0,
            vibrato_depth: 0.3,
            tremolo_rate: 3.0,
            tremolo_depth: 0.2,
        }
    }
}

/// AI-powered style transfer for singing voices
#[derive(Debug, Clone)]
pub struct StyleTransfer {
    device: Device,
    style_embeddings: HashMap<String, StyleEmbedding>,
    transfer_strength: f32,
    preserve_characteristics: Vec<String>,
}

/// Style embedding for neural style transfer
#[derive(Debug, Clone)]
pub struct StyleEmbedding {
    /// Neural embedding representing style characteristics
    pub embedding: Vec<f32>,
    /// Associated musical style
    pub style: MusicalStyle,
    /// Confidence score of the embedding
    pub confidence: f32,
    /// Metadata about the style
    pub metadata: StyleMetadata,
}

/// Metadata associated with a style
#[derive(Debug, Clone)]
pub struct StyleMetadata {
    /// Genre classification
    pub genre: String,
    /// Era or time period
    pub era: String,
    /// Cultural origin
    pub culture: String,
    /// Typical vocal techniques
    pub techniques: Vec<String>,
    /// Characteristic tempo range
    pub tempo_range: (f32, f32),
    /// Common key signatures
    pub common_keys: Vec<String>,
}

/// Configuration for style transfer operations
#[derive(Debug, Clone)]
pub struct StyleTransferConfig {
    /// Strength of style transfer (0.0 = no transfer, 1.0 = full transfer)
    pub transfer_strength: f32,
    /// Preserve original pitch characteristics
    pub preserve_pitch: bool,
    /// Preserve original timing characteristics
    pub preserve_timing: bool,
    /// Preserve original dynamics characteristics
    pub preserve_dynamics: bool,
    /// Smooth transitions between styles
    pub smooth_transitions: bool,
    /// Target style confidence threshold
    pub confidence_threshold: f32,
}

/// Result of style transfer operation
#[derive(Debug, Clone)]
pub struct StyleTransferResult {
    /// Transformed voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Confidence in the transfer quality
    pub transfer_confidence: f32,
    /// Applied style characteristics
    pub applied_style: StyleCharacteristics,
    /// Transfer quality metrics
    pub quality_metrics: TransferQualityMetrics,
}

/// Quality metrics for style transfer
#[derive(Debug, Clone)]
pub struct TransferQualityMetrics {
    /// Style consistency score (0.0-1.0)
    pub consistency: f32,
    /// Natural sound preservation score (0.0-1.0)
    pub naturalness: f32,
    /// Original characteristics preservation score (0.0-1.0)
    pub preservation: f32,
    /// Overall transfer quality score (0.0-1.0)
    pub overall_quality: f32,
}

impl Default for StyleTransferConfig {
    fn default() -> Self {
        Self {
            transfer_strength: 0.7,
            preserve_pitch: true,
            preserve_timing: false,
            preserve_dynamics: false,
            smooth_transitions: true,
            confidence_threshold: 0.8,
        }
    }
}

impl StyleTransfer {
    /// Create a new style transfer engine
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            style_embeddings: HashMap::new(),
            transfer_strength: 0.7,
            preserve_characteristics: vec![String::from("pitch")],
        })
    }

    /// Add a style embedding to the transfer engine
    pub fn add_style_embedding(&mut self, name: String, embedding: StyleEmbedding) {
        self.style_embeddings.insert(name, embedding);
    }

    /// Extract style embedding from voice characteristics
    pub fn extract_style_embedding(
        &self,
        voice_chars: &VoiceCharacteristics,
        style: &MusicalStyle,
    ) -> Result<StyleEmbedding> {
        // Extract style features from voice characteristics
        let features = vec![
            // Voice type features
            match voice_chars.voice_type {
                crate::types::VoiceType::Soprano => 1.0,
                crate::types::VoiceType::Alto => 0.8,
                crate::types::VoiceType::Tenor => 0.6,
                crate::types::VoiceType::Bass => 0.4,
                crate::types::VoiceType::MezzoSoprano => 0.9,
                crate::types::VoiceType::Baritone => 0.5,
            },
            // Pitch range features
            voice_chars.range.0 / 1000.0, // Normalize frequency
            voice_chars.range.1 / 1000.0,
            // Timbre characteristics from HashMap
            voice_chars.timbre.get("brightness").copied().unwrap_or(0.5),
            voice_chars.timbre.get("warmth").copied().unwrap_or(0.5),
            voice_chars
                .timbre
                .get("breathiness")
                .copied()
                .unwrap_or(0.3),
            voice_chars.timbre.get("roughness").copied().unwrap_or(0.2),
            // Style-specific features from voice characteristics
            voice_chars.vibrato_frequency,
            voice_chars.vibrato_depth,
            voice_chars
                .timbre
                .get("breathiness")
                .copied()
                .unwrap_or(0.3),
            voice_chars.vocal_power,
        ];

        // Create metadata
        let metadata = StyleMetadata {
            genre: style.name.clone(),
            era: String::from("Contemporary"),
            culture: style
                .cultural_variants
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| String::from("Western")),
            techniques: vec![String::from("vibrato"), String::from("legato")],
            tempo_range: (60.0, 180.0), // Default tempo range
            common_keys: vec![String::from("C"), String::from("G"), String::from("F")],
        };

        Ok(StyleEmbedding {
            embedding: features,
            style: style.clone(),
            confidence: 0.85,
            metadata,
        })
    }

    /// Transfer style between voice characteristics
    pub fn transfer_style(
        &self,
        source_voice: &VoiceCharacteristics,
        target_style: &str,
        config: &StyleTransferConfig,
    ) -> Result<StyleTransferResult> {
        let target_embedding = self
            .style_embeddings
            .get(target_style)
            .ok_or_else(|| Error::Processing(format!("Style '{target_style}' not found")))?;

        if target_embedding.confidence < config.confidence_threshold {
            return Err(Error::Processing(format!(
                "Target style confidence {confidence} below threshold {threshold}",
                confidence = target_embedding.confidence,
                threshold = config.confidence_threshold
            )));
        }

        // Create transferred voice characteristics
        let mut transferred_voice = source_voice.clone();

        // Apply style transfer based on configuration
        if !config.preserve_pitch {
            // Adjust pitch characteristics based on target style
            let pitch_adjustment = self.calculate_pitch_adjustment(target_embedding)?;
            transferred_voice.range.0 *= pitch_adjustment;
            transferred_voice.range.1 *= pitch_adjustment;
        }

        if !config.preserve_dynamics {
            // Adjust dynamic characteristics
            transferred_voice.vocal_power *= config.transfer_strength;
        }

        // Blend timbre characteristics
        let blend_factor = config.transfer_strength;
        let brightness = transferred_voice
            .timbre
            .get("brightness")
            .copied()
            .unwrap_or(0.5);
        let warmth = transferred_voice
            .timbre
            .get("warmth")
            .copied()
            .unwrap_or(0.5);
        let breathiness = transferred_voice
            .timbre
            .get("breathiness")
            .copied()
            .unwrap_or(0.3);

        transferred_voice.timbre.insert(
            String::from("brightness"),
            lerp(brightness, 0.7, blend_factor),
        );
        transferred_voice
            .timbre
            .insert(String::from("warmth"), lerp(warmth, 0.6, blend_factor));
        transferred_voice.timbre.insert(
            String::from("breathiness"),
            lerp(breathiness, 0.4, blend_factor),
        );

        // Calculate quality metrics
        let quality_metrics = self.calculate_transfer_quality(
            source_voice,
            &transferred_voice,
            target_embedding,
            config,
        )?;

        // Create applied style characteristics (simplified)
        let applied_style = target_embedding.style.characteristics.clone();

        Ok(StyleTransferResult {
            voice_characteristics: transferred_voice,
            transfer_confidence: target_embedding.confidence * config.transfer_strength,
            applied_style,
            quality_metrics,
        })
    }

    /// Calculate pitch adjustment factor based on target style
    fn calculate_pitch_adjustment(&self, target_embedding: &StyleEmbedding) -> Result<f32> {
        // Simple heuristic based on style characteristics
        let base_adjustment = match target_embedding.metadata.genre.as_str() {
            "Classical" => 1.05, // Slightly higher pitch
            "Jazz" => 0.98,      // Slightly lower pitch
            "Pop" => 1.02,       // Moderate adjustment
            "Folk" => 0.95,      // Lower pitch for earthier sound
            _ => 1.0,            // No adjustment
        };

        Ok(base_adjustment)
    }

    /// Calculate transfer quality metrics
    fn calculate_transfer_quality(
        &self,
        source: &VoiceCharacteristics,
        transferred: &VoiceCharacteristics,
        target_embedding: &StyleEmbedding,
        config: &StyleTransferConfig,
    ) -> Result<TransferQualityMetrics> {
        // Consistency: How well the transferred voice matches the target style
        let consistency = self.calculate_style_consistency(transferred, target_embedding)?;

        // Naturalness: How natural the transferred voice sounds
        let naturalness = self.calculate_naturalness(transferred)?;

        // Preservation: How well original characteristics are preserved
        let preservation = self.calculate_preservation(source, transferred, config)?;

        // Overall quality: Weighted combination
        let overall_quality =
            (consistency * 0.4 + naturalness * 0.3 + preservation * 0.3).clamp(0.0, 1.0);

        Ok(TransferQualityMetrics {
            consistency,
            naturalness,
            preservation,
            overall_quality,
        })
    }

    /// Calculate style consistency score
    fn calculate_style_consistency(
        &self,
        voice: &VoiceCharacteristics,
        _target_embedding: &StyleEmbedding,
    ) -> Result<f32> {
        // Simplified consistency calculation based on voice characteristics
        let brightness = voice.timbre.get("brightness").copied().unwrap_or(0.5);
        let warmth = voice.timbre.get("warmth").copied().unwrap_or(0.5);
        let breathiness = voice.timbre.get("breathiness").copied().unwrap_or(0.3);

        // Simple heuristic: consistency based on how "natural" the values are
        let consistency = (brightness + warmth + (1.0 - breathiness)) / 3.0;

        Ok(consistency.clamp(0.0, 1.0))
    }

    /// Calculate naturalness score
    fn calculate_naturalness(&self, voice: &VoiceCharacteristics) -> Result<f32> {
        // Simple heuristics for naturalness
        let mut naturalness = 1.0;

        let brightness = voice.timbre.get("brightness").copied().unwrap_or(0.5);
        let warmth = voice.timbre.get("warmth").copied().unwrap_or(0.5);
        let breathiness = voice.timbre.get("breathiness").copied().unwrap_or(0.3);
        let roughness = voice.timbre.get("roughness").copied().unwrap_or(0.2);

        // Check for extreme values that might sound unnatural
        if brightness > 0.95 || brightness < 0.05 {
            naturalness *= 0.8;
        }
        if warmth > 0.95 || warmth < 0.05 {
            naturalness *= 0.8;
        }
        if breathiness > 0.9 {
            naturalness *= 0.7; // Too much breathiness sounds unnatural
        }
        if roughness > 0.8 {
            naturalness *= 0.6; // Too much roughness sounds harsh
        }

        Ok(naturalness)
    }

    /// Calculate preservation score
    fn calculate_preservation(
        &self,
        source: &VoiceCharacteristics,
        transferred: &VoiceCharacteristics,
        config: &StyleTransferConfig,
    ) -> Result<f32> {
        let mut preservation = 1.0;

        // Calculate differences in preserved characteristics
        if config.preserve_pitch {
            let pitch_diff = ((source.range.0 - transferred.range.0).abs()
                + (source.range.1 - transferred.range.1).abs())
                / 2.0;
            preservation *= (1.0 - pitch_diff / 1000.0).max(0.0); // Normalize by 1000Hz
        }

        if config.preserve_dynamics {
            let dynamic_diff = (source.vocal_power - transferred.vocal_power).abs();
            preservation *= (1.0 - dynamic_diff).max(0.0);
        }

        Ok(preservation)
    }

    /// Blend style characteristics (simplified)
    fn blend_style_characteristics(
        &self,
        target_style: &StyleCharacteristics,
        _blend_factor: f32,
    ) -> StyleCharacteristics {
        // Return a simplified copy for now
        target_style.clone()
    }

    /// Get available styles
    pub fn available_styles(&self) -> Vec<String> {
        self.style_embeddings.keys().cloned().collect()
    }

    /// Get style embedding information
    pub fn get_style_info(&self, style_name: &str) -> Option<&StyleEmbedding> {
        self.style_embeddings.get(style_name)
    }
}

/// Linear interpolation helper function
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Automatic Harmonization System
#[derive(Debug, Clone)]
pub struct AutoHarmonizer {
    device: Device,
    harmony_models: HashMap<String, HarmonyModel>,
    default_rules: HarmonyRules,
}

/// Neural model for harmony generation
#[derive(Debug, Clone)]
pub struct HarmonyModel {
    /// Model weights for harmony prediction
    pub weights: Vec<f32>,
    /// Supported harmony types
    pub harmony_types: Vec<String>,
    /// Model confidence
    pub confidence: f32,
}

/// Rules for automatic harmonization
#[derive(Debug, Clone)]
pub struct HarmonyRules {
    /// Preferred chord progressions
    pub chord_progressions: Vec<Vec<String>>,
    /// Voice leading rules
    pub voice_leading: VoiceLeadingRules,
    /// Harmonic rhythm preferences
    pub harmonic_rhythm: HarmonicRhythm,
}

/// Voice leading rules for harmony
#[derive(Debug, Clone)]
pub struct VoiceLeadingRules {
    /// Maximum interval between adjacent notes
    pub max_interval: f32,
    /// Prefer stepwise motion
    pub prefer_stepwise: bool,
    /// Avoid parallel fifths/octaves
    pub avoid_parallels: bool,
}

/// Harmonic rhythm preferences
#[derive(Debug, Clone)]
pub struct HarmonicRhythm {
    /// Chord change frequency (chords per measure)
    pub chord_frequency: f32,
    /// Syncopation amount
    pub syncopation: f32,
    /// Rhythmic complexity
    pub complexity: f32,
}

impl Default for HarmonyRules {
    fn default() -> Self {
        Self {
            chord_progressions: vec![
                vec![
                    String::from("C"),
                    String::from("F"),
                    String::from("G"),
                    String::from("C"),
                ],
                vec![
                    String::from("Am"),
                    String::from("F"),
                    String::from("C"),
                    String::from("G"),
                ],
                vec![
                    String::from("C"),
                    String::from("Am"),
                    String::from("F"),
                    String::from("G"),
                ],
            ],
            voice_leading: VoiceLeadingRules {
                max_interval: 7.0, // Perfect fifth
                prefer_stepwise: true,
                avoid_parallels: true,
            },
            harmonic_rhythm: HarmonicRhythm {
                chord_frequency: 1.0, // One chord per measure
                syncopation: 0.2,
                complexity: 0.5,
            },
        }
    }
}

impl AutoHarmonizer {
    /// Create a new automatic harmonizer
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            harmony_models: HashMap::new(),
            default_rules: HarmonyRules::default(),
        })
    }

    /// Generate harmony for a melody
    pub fn generate_harmony(
        &self,
        melody: &[NoteEvent],
        style: &str,
    ) -> Result<Vec<Vec<NoteEvent>>> {
        // Use default harmony generation for now
        self.generate_basic_harmony(melody)
    }

    /// Generate basic harmony using traditional rules
    fn generate_basic_harmony(&self, melody: &[NoteEvent]) -> Result<Vec<Vec<NoteEvent>>> {
        let mut harmony_parts = vec![Vec::new(); 3]; // Alto, Tenor, Bass

        for note in melody {
            // Generate chord tones based on the melody note
            let chord_tones = self.generate_chord_tones(note)?;

            // Distribute chord tones to harmony parts
            for (i, tone) in chord_tones.iter().enumerate() {
                if i < harmony_parts.len() {
                    harmony_parts[i].push(tone.clone());
                }
            }
        }

        Ok(harmony_parts)
    }

    /// Generate chord tones for a melody note
    fn generate_chord_tones(&self, melody_note: &NoteEvent) -> Result<Vec<NoteEvent>> {
        let mut chord_tones = Vec::new();

        // Simple triad generation (root, third, fifth)
        let root_freq = melody_note.frequency;
        let third_freq = root_freq * 1.25992; // Major third
        let fifth_freq = root_freq * 1.49831; // Perfect fifth

        // Create harmony notes
        let alto_note = NoteEvent {
            note: melody_note.note.clone(),
            octave: melody_note.octave,
            frequency: third_freq,
            duration: melody_note.duration,
            velocity: melody_note.velocity * 0.8,
            vibrato: melody_note.vibrato,
            lyric: melody_note.lyric.clone(),
            phonemes: melody_note.phonemes.clone(),
            expression: melody_note.expression,
            timing_offset: melody_note.timing_offset,
            breath_before: melody_note.breath_before,
            legato: melody_note.legato,
            articulation: melody_note.articulation,
        };

        let tenor_note = NoteEvent {
            note: melody_note.note.clone(),
            octave: melody_note.octave.saturating_sub(1),
            frequency: root_freq * 0.75, // Lower octave
            duration: melody_note.duration,
            velocity: melody_note.velocity * 0.7,
            vibrato: melody_note.vibrato,
            lyric: melody_note.lyric.clone(),
            phonemes: melody_note.phonemes.clone(),
            expression: melody_note.expression,
            timing_offset: melody_note.timing_offset,
            breath_before: melody_note.breath_before,
            legato: melody_note.legato,
            articulation: melody_note.articulation,
        };

        let bass_note = NoteEvent {
            note: melody_note.note.clone(),
            octave: melody_note.octave.saturating_sub(2),
            frequency: root_freq * 0.5, // Two octaves down
            duration: melody_note.duration,
            velocity: melody_note.velocity * 0.9,
            vibrato: melody_note.vibrato,
            lyric: melody_note.lyric.clone(),
            phonemes: melody_note.phonemes.clone(),
            expression: melody_note.expression,
            timing_offset: melody_note.timing_offset,
            breath_before: melody_note.breath_before,
            legato: melody_note.legato,
            articulation: melody_note.articulation,
        };

        chord_tones.push(alto_note);
        chord_tones.push(tenor_note);
        chord_tones.push(bass_note);

        Ok(chord_tones)
    }
}

/// AI-powered improvisation assistant
#[derive(Debug, Clone)]
pub struct ImprovisationAssistant {
    device: Device,
    style_models: HashMap<String, ImprovisationModel>,
    creativity_level: f32,
}

/// Model for improvisation generation
#[derive(Debug, Clone)]
pub struct ImprovisationModel {
    /// Pattern library for improvisation
    pub patterns: Vec<ImprovisationPattern>,
    /// Style preferences
    pub style_weights: Vec<f32>,
    /// Complexity level
    pub complexity: f32,
}

/// Improvisation pattern
#[derive(Debug, Clone)]
pub struct ImprovisationPattern {
    /// Pattern name
    pub name: String,
    /// Note intervals
    pub intervals: Vec<i32>,
    /// Rhythmic pattern
    pub rhythm: Vec<f32>,
    /// Usage weight
    pub weight: f32,
}

impl ImprovisationAssistant {
    /// Create a new improvisation assistant
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            style_models: HashMap::new(),
            creativity_level: 0.7,
        })
    }

    /// Generate improvisation variations
    pub fn generate_variations(
        &self,
        base_melody: &[NoteEvent],
        style: &str,
        count: usize,
    ) -> Result<Vec<Vec<NoteEvent>>> {
        let mut variations = Vec::new();

        for i in 0..count {
            let variation = self.create_variation(base_melody, i as f32 / count as f32)?;
            variations.push(variation);
        }

        Ok(variations)
    }

    /// Create a single variation
    fn create_variation(
        &self,
        base_melody: &[NoteEvent],
        variation_amount: f32,
    ) -> Result<Vec<NoteEvent>> {
        let mut variation = Vec::new();

        for note in base_melody {
            let mut varied_note = note.clone();

            // Apply pitch variation by adjusting frequency
            let pitch_variation = (variation_amount * self.creativity_level * 2.0 - 1.0) * 3.0; // ±3 semitones max
            varied_note.frequency *= 2.0_f32.powf(pitch_variation / 12.0);

            // Apply timing variation
            let timing_variation = (variation_amount * self.creativity_level * 0.2 - 0.1) + 1.0; // ±10% timing
            varied_note.duration *= timing_variation;

            variation.push(varied_note);
        }

        Ok(variation)
    }
}

/// Emotion recognition system for singing
#[derive(Debug, Clone)]
pub struct EmotionRecognizer {
    device: Device,
    emotion_models: HashMap<String, EmotionModel>,
    confidence_threshold: f32,
}

/// Model for emotion recognition
#[derive(Debug, Clone)]
pub struct EmotionModel {
    /// Feature weights for emotion classification
    pub feature_weights: Vec<f32>,
    /// Supported emotions
    pub emotions: Vec<String>,
    /// Model accuracy
    pub accuracy: f32,
}

/// Recognized emotion information
#[derive(Debug, Clone)]
pub struct EmotionResult {
    /// Primary emotion
    pub primary_emotion: String,
    /// Confidence in primary emotion
    pub confidence: f32,
    /// Secondary emotions with their confidences
    pub secondary_emotions: Vec<(String, f32)>,
    /// Arousal level (0.0 = calm, 1.0 = excited)
    pub arousal: f32,
    /// Valence level (0.0 = negative, 1.0 = positive)
    pub valence: f32,
}

impl EmotionRecognizer {
    /// Create a new emotion recognizer
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            emotion_models: HashMap::new(),
            confidence_threshold: 0.7,
        })
    }

    /// Recognize emotion from voice characteristics and expression features
    pub fn recognize_emotion(
        &self,
        voice_chars: &VoiceCharacteristics,
        expression_features: &ExpressionFeatures,
    ) -> Result<EmotionResult> {
        // Extract features for emotion recognition
        let features = self.extract_emotion_features(voice_chars, expression_features)?;

        // Simple heuristic-based emotion recognition
        let (primary_emotion, confidence) = self.classify_emotion(&features)?;

        // Determine arousal and valence
        let arousal = self.calculate_arousal(&features);
        let valence = self.calculate_valence(&features);

        Ok(EmotionResult {
            primary_emotion,
            confidence,
            secondary_emotions: vec![(String::from("neutral"), 1.0 - confidence)],
            arousal,
            valence,
        })
    }

    /// Extract features for emotion recognition
    fn extract_emotion_features(
        &self,
        voice_chars: &VoiceCharacteristics,
        expression_features: &ExpressionFeatures,
    ) -> Result<Vec<f32>> {
        // Voice characteristics features + Expression features
        let features = vec![
            voice_chars.timbre.get("brightness").copied().unwrap_or(0.5),
            voice_chars.timbre.get("warmth").copied().unwrap_or(0.5),
            voice_chars
                .timbre
                .get("breathiness")
                .copied()
                .unwrap_or(0.3),
            voice_chars.timbre.get("roughness").copied().unwrap_or(0.2),
            voice_chars.vocal_power,
            expression_features.volume,
            expression_features.vibrato_rate,
            expression_features.vibrato_depth,
            expression_features.tremolo_rate,
            expression_features.tremolo_depth,
        ];

        Ok(features)
    }

    /// Classify emotion from features
    fn classify_emotion(&self, features: &[f32]) -> Result<(String, f32)> {
        // Simple heuristic classification
        let brightness = features[0];
        let warmth = features[1];
        let breathiness = features[2];
        let volume = features[5];
        let vibrato_depth = features[7];

        let emotion = if brightness > 0.7 && volume > 0.7 {
            "joy"
        } else if warmth > 0.8 && vibrato_depth < 0.3 {
            "calm"
        } else if breathiness > 0.6 && volume < 0.5 {
            "sadness"
        } else if brightness < 0.3 && volume > 0.8 {
            "anger"
        } else {
            "neutral"
        };

        // Calculate confidence based on how extreme the features are
        let extremeness =
            features.iter().map(|&f| (f - 0.5).abs() * 2.0).sum::<f32>() / features.len() as f32;
        let confidence = extremeness.min(1.0);

        Ok((emotion.to_string(), confidence))
    }

    /// Calculate arousal level
    fn calculate_arousal(&self, features: &[f32]) -> f32 {
        let volume = features[5];
        let vibrato_rate = features[6];
        let tremolo_rate = features[8];

        // High volume, fast vibrato/tremolo indicate high arousal
        (volume + vibrato_rate * 0.5 + tremolo_rate * 0.5).min(1.0)
    }

    /// Calculate valence level
    fn calculate_valence(&self, features: &[f32]) -> f32 {
        let brightness = features[0];
        let warmth = features[1];
        let breathiness = features[2];

        // Bright and warm voices tend to be positive
        // Breathy voices can be either positive (intimate) or negative (sad)
        let positive_indicators = brightness * 0.6 + warmth * 0.4;
        let negative_indicators = breathiness * 0.3;

        (positive_indicators - negative_indicators + 1.0) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceType;

    #[test]
    fn test_style_transfer_creation() {
        let device = Device::Cpu;
        let style_transfer = StyleTransfer::new(device).unwrap();
        assert_eq!(style_transfer.style_embeddings.len(), 0);
        assert_eq!(style_transfer.transfer_strength, 0.7);
    }

    #[test]
    fn test_style_embedding_extraction() {
        let device = Device::Cpu;
        let style_transfer = StyleTransfer::new(device).unwrap();

        let mut timbre = std::collections::HashMap::new();
        timbre.insert(String::from("brightness"), 0.7);
        timbre.insert(String::from("warmth"), 0.6);
        timbre.insert(String::from("breathiness"), 0.3);
        timbre.insert(String::from("roughness"), 0.2);

        let voice_chars = VoiceCharacteristics {
            voice_type: VoiceType::Soprano,
            range: (200.0, 800.0),
            f0_mean: 440.0,
            f0_std: 50.0,
            vibrato_frequency: 6.0,
            vibrato_depth: 0.3,
            breath_capacity: 8.0,
            vocal_power: 0.8,
            resonance: std::collections::HashMap::new(),
            timbre,
        };

        let style = MusicalStyle::classical();
        let embedding = style_transfer
            .extract_style_embedding(&voice_chars, &style)
            .unwrap();

        assert!(!embedding.embedding.is_empty());
        assert_eq!(embedding.style.name, "Classical");
        assert!(embedding.confidence > 0.0);
    }

    #[test]
    fn test_auto_harmonizer_creation() {
        let device = Device::Cpu;
        let harmonizer = AutoHarmonizer::new(device).unwrap();
        assert_eq!(harmonizer.harmony_models.len(), 0);
        assert_eq!(harmonizer.default_rules.chord_progressions.len(), 3);
    }

    #[test]
    fn test_chord_tone_generation() {
        let device = Device::Cpu;
        let harmonizer = AutoHarmonizer::new(device).unwrap();

        let melody_note = NoteEvent {
            note: "A".to_string(),
            octave: 4,
            frequency: 440.0, // A4
            duration: 1.0,
            velocity: 0.8,
            vibrato: 0.5,
            lyric: None,
            phonemes: Vec::new(),
            expression: crate::types::Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: crate::types::Articulation::Normal,
        };

        let chord_tones = harmonizer.generate_chord_tones(&melody_note).unwrap();
        assert_eq!(chord_tones.len(), 3); // Alto, Tenor, Bass

        // Check that frequencies are different
        let frequencies: Vec<f32> = chord_tones.iter().map(|n| n.frequency).collect();
        assert!(frequencies[0] != frequencies[1]);
        assert!(frequencies[1] != frequencies[2]);
    }

    #[test]
    fn test_improvisation_assistant_creation() {
        let device = Device::Cpu;
        let assistant = ImprovisationAssistant::new(device).unwrap();
        assert_eq!(assistant.style_models.len(), 0);
        assert_eq!(assistant.creativity_level, 0.7);
    }

    #[test]
    fn test_variation_generation() {
        let device = Device::Cpu;
        let assistant = ImprovisationAssistant::new(device).unwrap();

        let base_melody = vec![
            NoteEvent {
                note: String::from("A"),
                octave: 4,
                frequency: 440.0,
                duration: 1.0,
                velocity: 0.8,
                vibrato: 0.5,
                lyric: None,
                phonemes: Vec::new(),
                expression: crate::types::Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: crate::types::Articulation::Normal,
            },
            NoteEvent {
                note: String::from("B"),
                octave: 4,
                frequency: 494.0,
                duration: 1.0,
                velocity: 0.7,
                vibrato: 0.5,
                lyric: None,
                phonemes: Vec::new(),
                expression: crate::types::Expression::Neutral,
                timing_offset: 1.0,
                breath_before: 0.0,
                legato: false,
                articulation: crate::types::Articulation::Normal,
            },
        ];

        let variations = assistant
            .generate_variations(&base_melody, "jazz", 3)
            .unwrap();
        assert_eq!(variations.len(), 3);
        assert_eq!(variations[0].len(), 2);

        // Check that variations are actually different
        let original_freq = base_melody[0].frequency;
        let varied_freq = variations[1][0].frequency;
        assert_ne!(original_freq, varied_freq);
    }

    #[test]
    fn test_emotion_recognizer_creation() {
        let device = Device::Cpu;
        let recognizer = EmotionRecognizer::new(device).unwrap();
        assert_eq!(recognizer.emotion_models.len(), 0);
        assert_eq!(recognizer.confidence_threshold, 0.7);
    }

    #[test]
    fn test_emotion_recognition() {
        let device = Device::Cpu;
        let recognizer = EmotionRecognizer::new(device).unwrap();

        let mut timbre = std::collections::HashMap::new();
        timbre.insert(String::from("brightness"), 0.8); // High brightness
        timbre.insert(String::from("warmth"), 0.7);
        timbre.insert(String::from("breathiness"), 0.2);
        timbre.insert(String::from("roughness"), 0.1);

        let voice_chars = VoiceCharacteristics {
            voice_type: VoiceType::Soprano,
            range: (200.0, 800.0),
            f0_mean: 440.0,
            f0_std: 50.0,
            vibrato_frequency: 6.0,
            vibrato_depth: 0.3,
            breath_capacity: 8.0,
            vocal_power: 0.8,
            resonance: std::collections::HashMap::new(),
            timbre,
        };

        let expression_features = ExpressionFeatures {
            volume: 0.8,
            vibrato_rate: 5.0,
            vibrato_depth: 0.3,
            tremolo_rate: 3.0,
            tremolo_depth: 0.2,
        };

        let emotion_result = recognizer
            .recognize_emotion(&voice_chars, &expression_features)
            .unwrap();

        assert_eq!(emotion_result.primary_emotion, "joy"); // High brightness + volume = joy
        assert!(emotion_result.confidence > 0.0);
        assert!(emotion_result.arousal > 0.5); // High volume should indicate high arousal
        assert!(emotion_result.valence > 0.5); // Brightness should indicate positive valence
    }

    #[test]
    fn test_style_transfer_config_default() {
        let config = StyleTransferConfig::default();
        assert_eq!(config.transfer_strength, 0.7);
        assert!(config.preserve_pitch);
        assert!(!config.preserve_timing);
        assert!(config.smooth_transitions);
        assert_eq!(config.confidence_threshold, 0.8);
    }

    #[test]
    fn test_harmony_rules_default() {
        let rules = HarmonyRules::default();
        assert_eq!(rules.chord_progressions.len(), 3);
        assert!(rules.voice_leading.prefer_stepwise);
        assert!(rules.voice_leading.avoid_parallels);
        assert_eq!(rules.harmonic_rhythm.chord_frequency, 1.0);
    }
}
