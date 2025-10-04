//! Multi-voice harmony synthesis and management

use crate::config::SingingConfig;
use crate::score::MusicalScore;
use crate::synthesis::SynthesisEngine;
use crate::techniques::SingingTechnique;
use crate::types::{
    Articulation, Expression, NoteEvent, QualitySettings, SingingRequest, VoiceCharacteristics,
    VoiceType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Harmony arrangement types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HarmonyType {
    /// Simple parallel harmony (3rds, 5ths, etc.)
    Parallel,
    /// Traditional four-part harmony (SATB)
    FourPart,
    /// Jazz chord voicing
    Jazz,
    /// Close harmony (barbershop style)
    Close,
    /// Open harmony (wider intervals)
    Open,
    /// Custom harmony arrangement
    Custom,
}

/// Voice part in a harmony arrangement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePart {
    /// Voice identifier
    pub id: String,
    /// Voice characteristics
    pub voice: VoiceCharacteristics,
    /// Musical score for this voice part
    pub score: MusicalScore,
    /// Volume level (0.0-1.0)
    pub volume: f32,
    /// Pan position (-1.0 to 1.0, left to right)
    pub pan: f32,
    /// Priority in mixing (higher = more prominent)
    pub priority: u8,
    /// Effects specific to this voice
    pub effects: Vec<String>,
}

/// Multi-voice harmony arrangement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyArrangement {
    /// Arrangement name
    pub name: String,
    /// Harmony type
    pub harmony_type: HarmonyType,
    /// Voice parts in the arrangement
    pub voice_parts: Vec<VoicePart>,
    /// Master volume (0.0-1.0)
    pub master_volume: f32,
    /// Stereo width (0.0-1.0)
    pub stereo_width: f32,
    /// Reverb level (0.0-1.0)
    pub reverb: f32,
    /// Tempo in BPM
    pub tempo: f32,
    /// Key signature
    pub key_signature: String,
    /// Time signature
    pub time_signature: (u8, u8),
}

/// Multi-voice synthesis engine
pub struct MultiVoiceSynthesizer {
    /// Individual synthesis engines for each voice
    engines: HashMap<String, Arc<RwLock<SynthesisEngine>>>,
    /// Current harmony arrangement
    arrangement: Option<HarmonyArrangement>,
    /// Audio mixing buffer
    mix_buffer: Vec<Vec<f32>>,
    /// Sample rate
    sample_rate: u32,
}

impl MultiVoiceSynthesizer {
    /// Create new multi-voice synthesizer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            engines: HashMap::new(),
            arrangement: None,
            mix_buffer: Vec::new(),
            sample_rate,
        }
    }

    /// Set harmony arrangement
    ///
    /// Configures the multi-voice synthesizer with a harmony arrangement and initializes
    /// synthesis engines for each voice part. If engines already exist for voice parts,
    /// they are reused; otherwise, new engines are created with default configuration.
    ///
    /// # Arguments
    ///
    /// * `arrangement` - The harmony arrangement containing voice parts, mixing settings, and musical parameters
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the arrangement was set successfully.
    ///
    /// # Errors
    ///
    /// This method currently does not produce errors but returns a Result for future extensibility.
    pub async fn set_arrangement(
        &mut self,
        arrangement: HarmonyArrangement,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize synthesis engines for each voice part
        for voice_part in &arrangement.voice_parts {
            if !self.engines.contains_key(&voice_part.id) {
                let config = SingingConfig::default();
                let engine = Arc::new(RwLock::new(SynthesisEngine::new(config)));
                self.engines.insert(voice_part.id.clone(), engine);
            }
        }

        self.arrangement = Some(arrangement);
        Ok(())
    }

    /// Generate harmony arrangement from a lead melody
    ///
    /// Automatically creates a multi-voice harmony arrangement based on the specified harmony type.
    /// The method generates appropriate harmony lines (e.g., thirds, fifths, chord tones) for each
    /// voice type, with proper stereo positioning, volume balancing, and effect assignments.
    ///
    /// # Arguments
    ///
    /// * `lead_melody` - The primary melody to harmonize
    /// * `harmony_type` - The style of harmony to generate (parallel, four-part SATB, jazz, close, open, or custom)
    /// * `voice_types` - Ordered list of voice types to use in the arrangement (first is typically the lead)
    ///
    /// # Returns
    ///
    /// Returns a complete `HarmonyArrangement` with generated voice parts, mixing parameters, and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `HarmonyType::Custom` is specified (requires manual arrangement)
    /// - Internal harmony generation fails
    pub fn generate_harmony(
        &self,
        lead_melody: &MusicalScore,
        harmony_type: HarmonyType,
        voice_types: &[VoiceType],
    ) -> Result<HarmonyArrangement, Box<dyn std::error::Error>> {
        let voice_parts = match harmony_type {
            HarmonyType::FourPart => self.generate_four_part_harmony(lead_melody, voice_types)?,
            HarmonyType::Parallel => self.generate_parallel_harmony(lead_melody, voice_types)?,
            HarmonyType::Jazz => self.generate_jazz_harmony(lead_melody, voice_types)?,
            HarmonyType::Close => self.generate_close_harmony(lead_melody, voice_types)?,
            HarmonyType::Open => self.generate_open_harmony(lead_melody, voice_types)?,
            HarmonyType::Custom => return Err("Custom harmony requires manual arrangement".into()),
        };

        Ok(HarmonyArrangement {
            name: format!("{:?} Harmony", harmony_type),
            harmony_type,
            voice_parts,
            master_volume: 1.0,
            stereo_width: 0.8,
            reverb: 0.3,
            tempo: 120.0,
            key_signature: String::from("C"),
            time_signature: (4, 4),
        })
    }

    /// Synthesize multi-voice harmony
    ///
    /// Renders all voice parts in the current harmony arrangement and mixes them into a single
    /// audio output. Each voice part is synthesized independently using its assigned synthesis
    /// engine, then mixed with volume, panning, and normalization applied.
    ///
    /// # Returns
    ///
    /// Returns a mono audio buffer containing the mixed harmony output at the configured sample rate.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No harmony arrangement has been set (call `set_arrangement` first)
    /// - Individual voice synthesis fails
    /// - Audio mixing fails
    pub async fn synthesize(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let arrangement = self
            .arrangement
            .as_ref()
            .ok_or("No harmony arrangement set")?;

        let mut voice_outputs = HashMap::new();
        let mut max_length = 0;

        // Synthesize each voice part
        for voice_part in &arrangement.voice_parts {
            if let Some(engine) = self.engines.get(&voice_part.id) {
                let mut engine_lock = engine.write().await;

                // Create a singing request for this voice part
                let request = SingingRequest {
                    score: voice_part.score.clone(),
                    voice: voice_part.voice.clone(),
                    technique: SingingTechnique::default(),
                    effects: voice_part.effects.clone(),
                    sample_rate: self.sample_rate,
                    target_duration: None,
                    quality: QualitySettings::default(),
                };

                let synthesis_result = engine_lock.synthesize(request).await?;
                max_length = max_length.max(synthesis_result.audio.len());
                voice_outputs.insert(voice_part.id.clone(), synthesis_result.audio);
            }
        }

        // Mix voice parts
        let mixed_audio = self.mix_voices(&voice_outputs, arrangement, max_length)?;

        Ok(mixed_audio)
    }

    /// Mix multiple voice outputs
    fn mix_voices(
        &self,
        voice_outputs: &HashMap<String, Vec<f32>>,
        arrangement: &HarmonyArrangement,
        length: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut mixed = vec![0.0; length];

        for voice_part in &arrangement.voice_parts {
            if let Some(audio) = voice_outputs.get(&voice_part.id) {
                // Apply voice-specific volume and panning
                let volume = voice_part.volume * arrangement.master_volume;
                let pan = voice_part.pan;

                for (i, &sample) in audio.iter().enumerate() {
                    if i < mixed.len() {
                        // Simple stereo mixing (for mono output, ignore pan)
                        mixed[i] += sample * volume;
                    }
                }
            }
        }

        // Normalize to prevent clipping
        let max_amplitude = mixed.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_amplitude > 0.8 {
            let normalization_factor = 0.8 / max_amplitude;
            for sample in &mut mixed {
                *sample *= normalization_factor;
            }
        }

        Ok(mixed)
    }

    /// Generate traditional four-part harmony (SATB)
    fn generate_four_part_harmony(
        &self,
        lead_melody: &MusicalScore,
        voice_types: &[VoiceType],
    ) -> Result<Vec<VoicePart>, Box<dyn std::error::Error>> {
        let mut voice_parts = Vec::new();

        // Soprano (lead melody)
        if voice_types.contains(&VoiceType::Soprano) {
            let soprano_voice = VoiceCharacteristics::for_voice_type(VoiceType::Soprano);
            voice_parts.push(VoicePart {
                id: String::from("soprano"),
                voice: soprano_voice,
                score: lead_melody.clone(),
                volume: 1.0,
                pan: -0.3,
                priority: 10,
                effects: vec![String::from("slight_reverb")],
            });
        }

        // Alto (harmony below soprano)
        if voice_types.contains(&VoiceType::Alto) {
            let alto_voice = VoiceCharacteristics::for_voice_type(VoiceType::Alto);
            let alto_score = self.generate_harmony_line(lead_melody, -3, -7)?; // 3rd or 5th below
            voice_parts.push(VoicePart {
                id: String::from("alto"),
                voice: alto_voice,
                score: alto_score,
                volume: 0.9,
                pan: -0.1,
                priority: 8,
                effects: vec![String::from("warm_reverb")],
            });
        }

        // Tenor (middle harmony)
        if voice_types.contains(&VoiceType::Tenor) {
            let tenor_voice = VoiceCharacteristics::for_voice_type(VoiceType::Tenor);
            let tenor_score = self.generate_harmony_line(lead_melody, -8, -12)?; // Octave with variations
            voice_parts.push(VoicePart {
                id: String::from("tenor"),
                voice: tenor_voice,
                score: tenor_score,
                volume: 0.8,
                pan: 0.1,
                priority: 7,
                effects: vec![String::from("warm_reverb")],
            });
        }

        // Bass (root notes and bass line)
        if voice_types.contains(&VoiceType::Bass) {
            let bass_voice = VoiceCharacteristics::for_voice_type(VoiceType::Bass);
            let bass_score = self.generate_bass_line(lead_melody)?;
            voice_parts.push(VoicePart {
                id: String::from("bass"),
                voice: bass_voice,
                score: bass_score,
                volume: 0.9,
                pan: 0.3,
                priority: 9,
                effects: vec![String::from("deep_reverb")],
            });
        }

        Ok(voice_parts)
    }

    /// Generate parallel harmony (3rds, 5ths, etc.)
    fn generate_parallel_harmony(
        &self,
        lead_melody: &MusicalScore,
        voice_types: &[VoiceType],
    ) -> Result<Vec<VoicePart>, Box<dyn std::error::Error>> {
        let mut voice_parts = Vec::new();

        // Lead voice
        if let Some(&first_voice_type) = voice_types.first() {
            let lead_voice = VoiceCharacteristics::for_voice_type(first_voice_type);
            voice_parts.push(VoicePart {
                id: String::from("lead"),
                voice: lead_voice,
                score: lead_melody.clone(),
                volume: 1.0,
                pan: 0.0,
                priority: 10,
                effects: vec![String::from("lead_reverb")],
            });
        }

        // Parallel harmonies
        for (i, &voice_type) in voice_types.iter().skip(1).enumerate() {
            let interval = if i == 0 {
                -3
            } else if i == 1 {
                -5
            } else {
                -7
            }; // 3rd, 5th, 7th
            let harmony_voice = VoiceCharacteristics::for_voice_type(voice_type);
            let harmony_score = self.generate_harmony_line(lead_melody, interval, interval)?;

            voice_parts.push(VoicePart {
                id: format!("harmony_{num}", num = i + 1),
                voice: harmony_voice,
                score: harmony_score,
                volume: 0.8 - (i as f32 * 0.1),
                pan: if i % 2 == 0 { -0.5 } else { 0.5 },
                priority: 8 - i as u8,
                effects: vec![String::from("harmony_reverb")],
            });
        }

        Ok(voice_parts)
    }

    /// Generate jazz harmony with extended chords
    fn generate_jazz_harmony(
        &self,
        lead_melody: &MusicalScore,
        voice_types: &[VoiceType],
    ) -> Result<Vec<VoicePart>, Box<dyn std::error::Error>> {
        // Jazz harmony uses more complex chord voicings
        let mut voice_parts = Vec::new();

        // Lead melody
        if let Some(&first_voice_type) = voice_types.first() {
            let lead_voice = VoiceCharacteristics::for_voice_type(first_voice_type);
            voice_parts.push(VoicePart {
                id: String::from("lead"),
                voice: lead_voice,
                score: lead_melody.clone(),
                volume: 1.0,
                pan: 0.0,
                priority: 10,
                effects: vec![String::from("jazz_reverb")],
            });
        }

        // Jazz harmony intervals (7ths, 9ths, 11ths)
        let jazz_intervals = [-2, -4, -6, -9, -11]; // Various jazz intervals
        for (i, &voice_type) in voice_types.iter().skip(1).enumerate() {
            if let Some(&interval) = jazz_intervals.get(i) {
                let harmony_voice = VoiceCharacteristics::for_voice_type(voice_type);
                let harmony_score =
                    self.generate_harmony_line(lead_melody, interval, interval - 2)?;

                voice_parts.push(VoicePart {
                    id: format!("jazz_harmony_{num}", num = i + 1),
                    voice: harmony_voice,
                    score: harmony_score,
                    volume: 0.7,
                    pan: (i as f32 - 2.0) * 0.3, // Spread across stereo field
                    priority: 7,
                    effects: vec![String::from("jazz_chorus"), String::from("jazz_reverb")],
                });
            }
        }

        Ok(voice_parts)
    }

    /// Generate close harmony (tight intervals)
    fn generate_close_harmony(
        &self,
        lead_melody: &MusicalScore,
        voice_types: &[VoiceType],
    ) -> Result<Vec<VoicePart>, Box<dyn std::error::Error>> {
        let mut voice_parts = Vec::new();

        // Lead voice
        if let Some(&first_voice_type) = voice_types.first() {
            let lead_voice = VoiceCharacteristics::for_voice_type(first_voice_type);
            voice_parts.push(VoicePart {
                id: String::from("lead"),
                voice: lead_voice,
                score: lead_melody.clone(),
                volume: 1.0,
                pan: 0.0,
                priority: 10,
                effects: vec![String::from("close_harmony_reverb")],
            });
        }

        // Close harmony intervals (2nds, 3rds, 4ths)
        let close_intervals = [-1, -2, -3, -4]; // Tight intervals
        for (i, &voice_type) in voice_types.iter().skip(1).enumerate() {
            if let Some(&interval) = close_intervals.get(i) {
                let harmony_voice = VoiceCharacteristics::for_voice_type(voice_type);
                let harmony_score = self.generate_harmony_line(lead_melody, interval, interval)?;

                voice_parts.push(VoicePart {
                    id: format!("close_harmony_{num}", num = i + 1),
                    voice: harmony_voice,
                    score: harmony_score,
                    volume: 0.9,
                    pan: (i as f32 - 1.5) * 0.2, // Tight stereo spread
                    priority: 8,
                    effects: vec![String::from("close_harmony_reverb")],
                });
            }
        }

        Ok(voice_parts)
    }

    /// Generate open harmony (wide intervals)
    fn generate_open_harmony(
        &self,
        lead_melody: &MusicalScore,
        voice_types: &[VoiceType],
    ) -> Result<Vec<VoicePart>, Box<dyn std::error::Error>> {
        let mut voice_parts = Vec::new();

        // Lead voice
        if let Some(&first_voice_type) = voice_types.first() {
            let lead_voice = VoiceCharacteristics::for_voice_type(first_voice_type);
            voice_parts.push(VoicePart {
                id: String::from("lead"),
                voice: lead_voice,
                score: lead_melody.clone(),
                volume: 1.0,
                pan: 0.0,
                priority: 10,
                effects: vec![String::from("open_harmony_reverb")],
            });
        }

        // Open harmony intervals (octaves, wide 5ths, etc.)
        let open_intervals = [-12, -7, -5, -17]; // Wide intervals
        for (i, &voice_type) in voice_types.iter().skip(1).enumerate() {
            if let Some(&interval) = open_intervals.get(i) {
                let harmony_voice = VoiceCharacteristics::for_voice_type(voice_type);
                let harmony_score = self.generate_harmony_line(lead_melody, interval, interval)?;

                voice_parts.push(VoicePart {
                    id: format!("open_harmony_{num}", num = i + 1),
                    voice: harmony_voice,
                    score: harmony_score,
                    volume: 0.8,
                    pan: (i as f32 - 1.5) * 0.6, // Wide stereo spread
                    priority: 7,
                    effects: vec![String::from("open_harmony_reverb")],
                });
            }
        }

        Ok(voice_parts)
    }

    /// Generate a harmony line at specified intervals
    fn generate_harmony_line(
        &self,
        lead_melody: &MusicalScore,
        primary_interval: i8,
        secondary_interval: i8,
    ) -> Result<MusicalScore, Box<dyn std::error::Error>> {
        let mut harmony_notes = Vec::new();

        for note in &lead_melody.notes {
            // Create harmony note by transposing the original
            let harmony_event = self.transpose_note(&note.event, primary_interval)?;
            let harmony_note =
                crate::score::MusicalNote::new(harmony_event, note.start_time, note.duration);
            harmony_notes.push(harmony_note);
        }

        Ok(MusicalScore {
            title: format!("Harmony Line ({:+} semitones)", primary_interval),
            composer: String::from("Generated"),
            tempo: lead_melody.tempo,
            time_signature: lead_melody.time_signature,
            key_signature: lead_melody.key_signature,
            notes: harmony_notes,
            lyrics: None, // Harmony lines typically don't have separate lyrics
            metadata: std::collections::HashMap::new(),
            duration: lead_melody.duration,
            sections: Vec::new(),
            markers: Vec::new(),
            breath_marks: Vec::new(),
            dynamics: Vec::new(),
            expressions: Vec::new(),
        })
    }

    /// Generate bass line from lead melody
    fn generate_bass_line(
        &self,
        lead_melody: &MusicalScore,
    ) -> Result<MusicalScore, Box<dyn std::error::Error>> {
        let mut bass_notes = Vec::new();

        for note in &lead_melody.notes {
            // Create bass note (typically root or fifth)
            let bass_event = self.generate_bass_note(&note.event)?;
            let bass_note = crate::score::MusicalNote::new(
                bass_event,
                note.start_time,
                note.duration * 2.0, // Bass notes often longer
            );
            bass_notes.push(bass_note);
        }

        Ok(MusicalScore {
            title: String::from("Bass Line"),
            composer: String::from("Generated"),
            tempo: lead_melody.tempo,
            time_signature: lead_melody.time_signature,
            key_signature: lead_melody.key_signature,
            notes: bass_notes,
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: lead_melody.duration,
            sections: Vec::new(),
            markers: Vec::new(),
            breath_marks: Vec::new(),
            dynamics: Vec::new(),
            expressions: Vec::new(),
        })
    }

    /// Transpose a note by semitones
    fn transpose_note(
        &self,
        note: &NoteEvent,
        semitones: i8,
    ) -> Result<NoteEvent, Box<dyn std::error::Error>> {
        let new_frequency = note.frequency * 2_f32.powf(semitones as f32 / 12.0);

        // Calculate new note name and octave
        let (new_note, new_octave) = self.frequency_to_note(new_frequency);

        Ok(NoteEvent {
            note: new_note,
            octave: new_octave,
            frequency: new_frequency,
            duration: note.duration,
            velocity: note.velocity * 0.8, // Slightly lower velocity for harmony
            vibrato: note.vibrato * 0.7,   // Less vibrato for harmony
            lyric: None,                   // Harmony lines don't need lyrics
            phonemes: vec![String::from("ah")], // Default vowel sound
            expression: Expression::Neutral, // More neutral expression for harmony
            timing_offset: note.timing_offset,
            breath_before: note.breath_before * 0.5, // Less breath for harmony
            legato: note.legato,
            articulation: note.articulation,
        })
    }

    /// Generate bass note from melody note
    fn generate_bass_note(
        &self,
        note: &NoteEvent,
    ) -> Result<NoteEvent, Box<dyn std::error::Error>> {
        // Bass notes are typically 1-2 octaves lower and follow chord roots
        let bass_frequency = note.frequency / 4.0; // Two octaves down
        let (bass_note, bass_octave) = self.frequency_to_note(bass_frequency);

        Ok(NoteEvent {
            note: bass_note,
            octave: bass_octave,
            frequency: bass_frequency,
            duration: note.duration * 2.0, // Bass notes often longer
            velocity: 0.9,                 // Strong bass presence
            vibrato: 0.2,                  // Minimal vibrato for bass
            lyric: None,
            phonemes: vec![String::from("oh")], // Deep vowel sound
            expression: Expression::Neutral,
            timing_offset: note.timing_offset,
            breath_before: 0.0,                 // No breath for sustained bass
            legato: true,                       // Bass lines often legato
            articulation: Articulation::Legato, // Bass typically legato
        })
    }

    /// Convert frequency to note name and octave
    fn frequency_to_note(&self, frequency: f32) -> (String, u8) {
        let a4_frequency = 440.0;
        let semitones_from_a4 = (12.0 * (frequency / a4_frequency).log2()).round() as i32;

        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let note_index = ((semitones_from_a4 + 9) % 12 + 12) % 12; // A4 is index 9
        let octave = ((semitones_from_a4 + 9) / 12 + 4).max(0) as u8;

        (note_names[note_index as usize].to_string(), octave)
    }
}

impl VoiceCharacteristics {
    /// Create voice characteristics for a specific voice type
    ///
    /// Generates appropriate vocal parameters (range, formants, timbre) for standard voice classifications.
    /// Each voice type has predefined frequency ranges, mean fundamental frequency (F0), formant frequencies,
    /// and timbral characteristics that match typical human vocal characteristics.
    ///
    /// # Arguments
    ///
    /// * `voice_type` - The vocal classification (Soprano, Alto, Tenor, Bass, etc.)
    ///
    /// # Returns
    ///
    /// Returns a `VoiceCharacteristics` instance configured with appropriate parameters for the voice type.
    ///
    /// # Voice Type Ranges
    ///
    /// - Soprano: C4-C6 (261-1047 Hz)
    /// - MezzoSoprano: A3-A5 (220-880 Hz)
    /// - Alto: F3-F5 (175-698 Hz)
    /// - Tenor: D3-D5 (147-587 Hz)
    /// - Baritone: A2-A4 (110-440 Hz)
    /// - Bass: E2-E4 (82-330 Hz)
    pub fn for_voice_type(voice_type: VoiceType) -> Self {
        let (range, f0_mean) = match voice_type {
            VoiceType::Soprano => ((261.0, 1047.0), 523.0), // C4 to C6, average A4
            VoiceType::MezzoSoprano => ((220.0, 880.0), 440.0), // A3 to A5, average A4
            VoiceType::Alto => ((175.0, 698.0), 349.0),     // F3 to F5, average F4
            VoiceType::Tenor => ((147.0, 587.0), 294.0),    // D3 to D5, average D4
            VoiceType::Baritone => ((110.0, 440.0), 220.0), // A2 to A4, average A3
            VoiceType::Bass => ((82.0, 330.0), 165.0),      // E2 to E4, average E3
        };

        let mut resonance = HashMap::new();
        resonance.insert(String::from("formant_f1"), f0_mean * 1.5);
        resonance.insert(String::from("formant_f2"), f0_mean * 3.0);
        resonance.insert(String::from("formant_f3"), f0_mean * 5.0);

        let mut timbre = HashMap::new();
        timbre.insert(String::from("brightness"), 0.7);
        timbre.insert(String::from("warmth"), 0.8);
        timbre.insert(String::from("breathiness"), 0.3);

        Self {
            voice_type,
            range,
            f0_mean,
            f0_std: f0_mean * 0.1,
            vibrato_frequency: 5.5,
            vibrato_depth: 0.15,
            breath_capacity: 15.0,
            vocal_power: 0.8,
            resonance,
            timbre,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::MusicalNote;
    use std::time::Duration;

    fn create_test_score() -> MusicalScore {
        use crate::score::*;
        use std::time::Duration;

        let note_events = vec![
            NoteEvent {
                note: String::from("C"),
                octave: 4,
                frequency: 261.63,
                duration: 1.0,
                velocity: 0.8,
                vibrato: 0.2,
                lyric: Some(String::from("Test")),
                phonemes: vec![
                    String::from("t"),
                    String::from("e"),
                    String::from("s"),
                    String::from("t"),
                ],
                expression: Expression::Happy,
                timing_offset: 0.0,
                breath_before: 0.1,
                legato: false,
                articulation: Articulation::Normal,
            },
            NoteEvent {
                note: String::from("E"),
                octave: 4,
                frequency: 329.63,
                duration: 1.0,
                velocity: 0.8,
                vibrato: 0.2,
                lyric: Some(String::from("Note")),
                phonemes: vec![String::from("n"), String::from("o"), String::from("t")],
                expression: Expression::Happy,
                timing_offset: 1.0,
                breath_before: 0.0,
                legato: true,
                articulation: Articulation::Legato,
            },
        ];

        let notes = note_events
            .into_iter()
            .enumerate()
            .map(|(i, event)| MusicalNote::new(event, i as f32, 1.0))
            .collect();

        let lyrics = Lyrics {
            lines: vec![String::from("Test Note")],
            syllables: Vec::new(),
            language: String::from("en"),
            phonemes: None,
        };

        MusicalScore {
            title: String::from("Test Song"),
            composer: String::from("Test Composer"),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature::default(),
            notes,
            lyrics: Some(lyrics),
            metadata: std::collections::HashMap::new(),
            duration: Duration::from_secs(2),
            sections: Vec::new(),
            markers: Vec::new(),
            breath_marks: Vec::new(),
            dynamics: Vec::new(),
            expressions: Vec::new(),
        }
    }

    #[test]
    fn test_multi_voice_synthesizer_creation() {
        let synthesizer = MultiVoiceSynthesizer::new(44100);
        assert_eq!(synthesizer.sample_rate, 44100);
        assert!(synthesizer.arrangement.is_none());
        assert!(synthesizer.engines.is_empty());
    }

    #[test]
    fn test_harmony_generation_four_part() {
        let synthesizer = MultiVoiceSynthesizer::new(44100);
        let lead_melody = create_test_score();
        let voice_types = vec![
            VoiceType::Soprano,
            VoiceType::Alto,
            VoiceType::Tenor,
            VoiceType::Bass,
        ];

        let harmony = synthesizer
            .generate_harmony(&lead_melody, HarmonyType::FourPart, &voice_types)
            .unwrap();

        assert_eq!(harmony.harmony_type, HarmonyType::FourPart);
        assert_eq!(harmony.voice_parts.len(), 4);

        // Check that all voice types are represented
        let part_ids: Vec<&String> = harmony.voice_parts.iter().map(|p| &p.id).collect();
        assert!(part_ids.contains(&&String::from("soprano")));
        assert!(part_ids.contains(&&String::from("alto")));
        assert!(part_ids.contains(&&String::from("tenor")));
        assert!(part_ids.contains(&&String::from("bass")));
    }

    #[test]
    fn test_harmony_generation_parallel() {
        let synthesizer = MultiVoiceSynthesizer::new(44100);
        let lead_melody = create_test_score();
        let voice_types = vec![VoiceType::Soprano, VoiceType::Alto, VoiceType::Tenor];

        let harmony = synthesizer
            .generate_harmony(&lead_melody, HarmonyType::Parallel, &voice_types)
            .unwrap();

        assert_eq!(harmony.harmony_type, HarmonyType::Parallel);
        assert_eq!(harmony.voice_parts.len(), 3);

        // Lead voice should be first
        assert_eq!(harmony.voice_parts[0].id, "lead");
        assert_eq!(harmony.voice_parts[0].priority, 10);
    }

    #[test]
    fn test_voice_characteristics_for_voice_type() {
        let soprano_voice = VoiceCharacteristics::for_voice_type(VoiceType::Soprano);
        assert_eq!(soprano_voice.voice_type, VoiceType::Soprano);
        assert_eq!(soprano_voice.range, (261.0, 1047.0));
        assert_eq!(soprano_voice.f0_mean, 523.0);

        let bass_voice = VoiceCharacteristics::for_voice_type(VoiceType::Bass);
        assert_eq!(bass_voice.voice_type, VoiceType::Bass);
        assert_eq!(bass_voice.range, (82.0, 330.0));
        assert_eq!(bass_voice.f0_mean, 165.0);
    }

    #[test]
    fn test_transpose_note() {
        let synthesizer = MultiVoiceSynthesizer::new(44100);
        let original_note = NoteEvent {
            note: "C".to_string(),
            octave: 4,
            frequency: 261.63,
            duration: 1.0,
            velocity: 0.8,
            vibrato: 0.2,
            lyric: Some("Test".to_string()),
            phonemes: vec![String::from("t")],
            expression: Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.1,
            legato: false,
            articulation: Articulation::Normal,
        };

        // Transpose up by 3 semitones (minor third)
        let transposed_note = synthesizer.transpose_note(&original_note, 3).unwrap();

        // Should be approximately D#4 (~311.13 Hz) - C4 + 3 semitones
        assert!((transposed_note.frequency - 311.13).abs() < 1.0);
        assert_eq!(transposed_note.note, "D#");
        assert_eq!(transposed_note.octave, 4);

        // Harmony notes should have adjusted properties
        assert_eq!(transposed_note.velocity, original_note.velocity * 0.8);
        assert_eq!(transposed_note.vibrato, original_note.vibrato * 0.7);
        assert!(transposed_note.lyric.is_none());
    }

    #[test]
    fn test_frequency_to_note() {
        let synthesizer = MultiVoiceSynthesizer::new(44100);

        // Test A4 = 440 Hz
        let (note, octave) = synthesizer.frequency_to_note(440.0);
        assert_eq!(note, "A");
        assert_eq!(octave, 4);

        // Test C4 = 261.63 Hz
        let (note, octave) = synthesizer.frequency_to_note(261.63);
        assert_eq!(note, "C");
        assert_eq!(octave, 4);

        // Test C5 = 523.25 Hz
        let (note, octave) = synthesizer.frequency_to_note(523.25);
        assert_eq!(note, "C");
        assert_eq!(octave, 5);
    }

    #[tokio::test]
    async fn test_set_arrangement() {
        let mut synthesizer = MultiVoiceSynthesizer::new(44100);
        let lead_melody = create_test_score();
        let voice_types = vec![VoiceType::Soprano, VoiceType::Alto];

        let harmony = synthesizer
            .generate_harmony(&lead_melody, HarmonyType::Parallel, &voice_types)
            .unwrap();

        let result = synthesizer.set_arrangement(harmony).await;
        assert!(result.is_ok());
        assert!(synthesizer.arrangement.is_some());
        assert_eq!(synthesizer.engines.len(), 2);
    }

    #[test]
    fn test_harmony_type_enum() {
        // Test that all harmony types can be created
        let harmony_types = vec![
            HarmonyType::Parallel,
            HarmonyType::FourPart,
            HarmonyType::Jazz,
            HarmonyType::Close,
            HarmonyType::Open,
            HarmonyType::Custom,
        ];

        for harmony_type in harmony_types {
            // Should be able to serialize/deserialize
            let serialized = serde_json::to_string(&harmony_type).unwrap();
            let deserialized: HarmonyType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(harmony_type, deserialized);
        }
    }
}
