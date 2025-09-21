//! Singing voice synthesis functionality for acoustic models.

use crate::{Phoneme, Result, SynthesisConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Musical note representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalNote {
    /// Note name (e.g., "C", "C#", "D", etc.)
    pub name: String,
    /// Octave number (e.g., 4 for middle C)
    pub octave: u8,
    /// Frequency in Hz
    pub frequency: f32,
    /// Duration in beats
    pub duration: f32,
    /// Phoneme to sing
    pub phoneme: Option<Phoneme>,
    /// Lyrics syllable
    pub lyrics: Option<String>,
    /// Dynamics (volume) marking
    pub dynamics: Option<DynamicsMarking>,
    /// Articulation marking
    pub articulation: Option<ArticulationMarking>,
}

impl MusicalNote {
    /// Create a new musical note
    pub fn new(name: String, octave: u8, duration: f32) -> Self {
        let frequency = Self::calculate_frequency(&name, octave);
        Self {
            name,
            octave,
            frequency,
            duration,
            phoneme: None,
            lyrics: None,
            dynamics: None,
            articulation: None,
        }
    }

    /// Calculate frequency from note name and octave
    fn calculate_frequency(name: &str, octave: u8) -> f32 {
        // A4 = 440 Hz reference
        let a4_freq = 440.0;

        // Note positions relative to A (A=0, A#=1, B=2, C=3, etc.)
        let note_positions = [
            ("C", -9),
            ("C#", -8),
            ("Db", -8),
            ("D", -7),
            ("D#", -6),
            ("Eb", -6),
            ("E", -5),
            ("F", -4),
            ("F#", -3),
            ("Gb", -3),
            ("G", -2),
            ("G#", -1),
            ("Ab", -1),
            ("A", 0),
            ("A#", 1),
            ("Bb", 1),
            ("B", 2),
        ];

        let position = note_positions
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, p)| *p)
            .unwrap_or(0);

        // Calculate semitones from A4
        let semitones = (octave as i32 - 4) * 12 + position;

        // Convert to frequency using equal temperament
        a4_freq * (2.0_f32).powf(semitones as f32 / 12.0)
    }

    /// Set phoneme for singing
    pub fn with_phoneme(mut self, phoneme: Phoneme) -> Self {
        self.phoneme = Some(phoneme);
        self
    }

    /// Set lyrics syllable
    pub fn with_lyrics(mut self, lyrics: String) -> Self {
        self.lyrics = Some(lyrics);
        self
    }

    /// Set dynamics marking
    pub fn with_dynamics(mut self, dynamics: DynamicsMarking) -> Self {
        self.dynamics = Some(dynamics);
        self
    }

    /// Set articulation marking
    pub fn with_articulation(mut self, articulation: ArticulationMarking) -> Self {
        self.articulation = Some(articulation);
        self
    }
}

/// Dynamics markings for musical expression
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DynamicsMarking {
    /// Pianississimo (very very soft)
    Ppp,
    /// Pianissimo (very soft)
    Pp,
    /// Piano (soft)
    P,
    /// Mezzo-piano (medium soft)
    Mp,
    /// Mezzo-forte (medium loud)
    Mf,
    /// Forte (loud)
    F,
    /// Fortissimo (very loud)
    Ff,
    /// Fortississimo (very very loud)
    Fff,
    /// Crescendo (gradually louder)
    Crescendo,
    /// Diminuendo (gradually softer)
    Diminuendo,
}

impl DynamicsMarking {
    /// Get volume scaling factor (0.0 to 1.0)
    pub fn volume_scale(&self) -> f32 {
        match self {
            DynamicsMarking::Ppp => 0.1,
            DynamicsMarking::Pp => 0.2,
            DynamicsMarking::P => 0.3,
            DynamicsMarking::Mp => 0.5,
            DynamicsMarking::Mf => 0.7,
            DynamicsMarking::F => 0.8,
            DynamicsMarking::Ff => 0.9,
            DynamicsMarking::Fff => 1.0,
            DynamicsMarking::Crescendo => 0.6,  // Default midpoint
            DynamicsMarking::Diminuendo => 0.6, // Default midpoint
        }
    }
}

/// Articulation markings for musical expression
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ArticulationMarking {
    /// Staccato (short and detached)
    Staccato,
    /// Legato (smooth and connected)
    Legato,
    /// Marcato (stressed)
    Marcato,
    /// Tenuto (held)
    Tenuto,
    /// Accent (emphasized)
    Accent,
    /// Slur (smooth connection between notes)
    Slur,
    /// Tie (same pitch connected)
    Tie,
}

impl ArticulationMarking {
    /// Get duration scaling factor
    pub fn duration_scale(&self) -> f32 {
        match self {
            ArticulationMarking::Staccato => 0.5,
            ArticulationMarking::Legato => 1.0,
            ArticulationMarking::Marcato => 0.8,
            ArticulationMarking::Tenuto => 1.0,
            ArticulationMarking::Accent => 0.9,
            ArticulationMarking::Slur => 1.0,
            ArticulationMarking::Tie => 1.0,
        }
    }
}

/// Musical phrase containing multiple notes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalPhrase {
    /// Notes in the phrase
    pub notes: Vec<MusicalNote>,
    /// Tempo in BPM (beats per minute)
    pub tempo: f32,
    /// Time signature numerator
    pub time_signature_numerator: u8,
    /// Time signature denominator
    pub time_signature_denominator: u8,
    /// Key signature
    pub key_signature: Option<KeySignature>,
    /// Breath marks
    pub breath_marks: Vec<f32>, // Positions in beats
}

impl MusicalPhrase {
    /// Create a new musical phrase
    pub fn new(tempo: f32) -> Self {
        Self {
            notes: Vec::new(),
            tempo,
            time_signature_numerator: 4,
            time_signature_denominator: 4,
            key_signature: None,
            breath_marks: Vec::new(),
        }
    }

    /// Add a note to the phrase
    pub fn add_note(&mut self, note: MusicalNote) {
        self.notes.push(note);
    }

    /// Set time signature
    pub fn with_time_signature(mut self, numerator: u8, denominator: u8) -> Self {
        self.time_signature_numerator = numerator;
        self.time_signature_denominator = denominator;
        self
    }

    /// Set key signature
    pub fn with_key_signature(mut self, key: KeySignature) -> Self {
        self.key_signature = Some(key);
        self
    }

    /// Add breath mark at position
    pub fn add_breath_mark(&mut self, position: f32) {
        self.breath_marks.push(position);
        self.breath_marks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    /// Get total duration in seconds
    pub fn total_duration(&self) -> f32 {
        let beats_per_second = self.tempo / 60.0;
        let total_beats: f32 = self.notes.iter().map(|n| n.duration).sum();
        total_beats / beats_per_second
    }

    /// Get phrase range (lowest and highest frequencies)
    pub fn frequency_range(&self) -> (f32, f32) {
        let frequencies: Vec<f32> = self.notes.iter().map(|n| n.frequency).collect();
        let min_freq = frequencies.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_freq = frequencies.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        (min_freq, max_freq)
    }
}

/// Key signature for musical phrases
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KeySignature {
    /// C major / A minor
    CMajor,
    /// G major / E minor (1 sharp)
    GMajor,
    /// D major / B minor (2 sharps)
    DMajor,
    /// A major / F# minor (3 sharps)
    AMajor,
    /// E major / C# minor (4 sharps)
    EMajor,
    /// B major / G# minor (5 sharps)
    BMajor,
    /// F# major / D# minor (6 sharps)
    FSharpMajor,
    /// C# major / A# minor (7 sharps)
    CSharpMajor,
    /// F major / D minor (1 flat)
    FMajor,
    /// Bb major / G minor (2 flats)
    BbMajor,
    /// Eb major / C minor (3 flats)
    EbMajor,
    /// Ab major / F minor (4 flats)
    AbMajor,
    /// Db major / Bb minor (5 flats)
    DbMajor,
    /// Gb major / Eb minor (6 flats)
    GbMajor,
    /// Cb major / Ab minor (7 flats)
    CbMajor,
}

/// Vibrato configuration for singing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingVibratoConfig {
    /// Vibrato rate in Hz (typically 4-8 Hz)
    pub rate: f32,
    /// Vibrato depth in cents (typically 50-200 cents)
    pub depth: f32,
    /// Vibrato onset delay in seconds
    pub onset_delay: f32,
    /// Vibrato fade-in duration in seconds
    pub fade_in_duration: f32,
    /// Whether vibrato is enabled
    pub enabled: bool,
}

impl Default for SingingVibratoConfig {
    fn default() -> Self {
        Self {
            rate: 6.0,
            depth: 100.0,
            onset_delay: 0.3,
            fade_in_duration: 0.2,
            enabled: true,
        }
    }
}

/// Breath control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathControlConfig {
    /// Breath support strength (0.0-1.0)
    pub support_strength: f32,
    /// Breath flow rate (0.0-1.0)
    pub flow_rate: f32,
    /// Breath capacity (0.0-1.0)
    pub capacity: f32,
    /// Breath pause duration in seconds
    pub pause_duration: f32,
    /// Whether to add breath sounds
    pub add_breath_sounds: bool,
}

impl Default for BreathControlConfig {
    fn default() -> Self {
        Self {
            support_strength: 0.8,
            flow_rate: 0.7,
            capacity: 0.9,
            pause_duration: 0.3,
            add_breath_sounds: true,
        }
    }
}

/// Singing technique configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingTechnique {
    /// Voice type (soprano, alto, tenor, bass)
    pub voice_type: VoiceType,
    /// Vocal register (chest, head, mix)
    pub vocal_register: VocalRegister,
    /// Vibrato configuration
    pub vibrato: SingingVibratoConfig,
    /// Breath control configuration
    pub breath_control: BreathControlConfig,
    /// Resonance configuration
    pub resonance: ResonanceConfig,
    /// Formant adjustment
    pub formant_adjustment: FormantAdjustment,
}

impl Default for SingingTechnique {
    fn default() -> Self {
        Self {
            voice_type: VoiceType::Soprano,
            vocal_register: VocalRegister::Mix,
            vibrato: SingingVibratoConfig::default(),
            breath_control: BreathControlConfig::default(),
            resonance: ResonanceConfig::default(),
            formant_adjustment: FormantAdjustment::default(),
        }
    }
}

/// Voice type classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoiceType {
    /// Soprano (high female voice)
    Soprano,
    /// Mezzo-soprano (medium female voice)
    MezzoSoprano,
    /// Alto (low female voice)
    Alto,
    /// Countertenor (high male voice)
    Countertenor,
    /// Tenor (high male voice)
    Tenor,
    /// Baritone (medium male voice)
    Baritone,
    /// Bass (low male voice)
    Bass,
}

impl VoiceType {
    /// Get typical frequency range for voice type
    pub fn frequency_range(&self) -> (f32, f32) {
        match self {
            VoiceType::Soprano => (261.6, 1046.5),     // C4 to C6
            VoiceType::MezzoSoprano => (220.0, 880.0), // A3 to A5
            VoiceType::Alto => (196.0, 698.5),         // G3 to F5
            VoiceType::Countertenor => (196.0, 698.5), // G3 to F5
            VoiceType::Tenor => (146.8, 523.3),        // D3 to C5
            VoiceType::Baritone => (110.0, 392.0),     // A2 to G4
            VoiceType::Bass => (82.4, 311.1),          // E2 to D#4
        }
    }

    /// Get formant adjustment for voice type
    pub fn formant_adjustment(&self) -> FormantAdjustment {
        match self {
            VoiceType::Soprano => FormantAdjustment {
                f1_adjustment: 1.1,
                f2_adjustment: 1.15,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::MezzoSoprano => FormantAdjustment {
                f1_adjustment: 1.05,
                f2_adjustment: 1.1,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::Alto => FormantAdjustment {
                f1_adjustment: 1.0,
                f2_adjustment: 1.05,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::Countertenor => FormantAdjustment {
                f1_adjustment: 0.95,
                f2_adjustment: 1.0,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::Tenor => FormantAdjustment {
                f1_adjustment: 0.9,
                f2_adjustment: 0.95,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::Baritone => FormantAdjustment {
                f1_adjustment: 0.85,
                f2_adjustment: 0.9,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
            VoiceType::Bass => FormantAdjustment {
                f1_adjustment: 0.8,
                f2_adjustment: 0.85,
                f3_adjustment: 1.0,
                f4_adjustment: 1.0,
            },
        }
    }
}

/// Vocal register classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VocalRegister {
    /// Chest voice (lower register)
    Chest,
    /// Head voice (upper register)
    Head,
    /// Mixed voice (blend of chest and head)
    Mix,
    /// Falsetto (breathy upper register)
    Falsetto,
}

impl VocalRegister {
    /// Get resonance adjustment for vocal register
    pub fn resonance_adjustment(&self) -> f32 {
        match self {
            VocalRegister::Chest => 0.8,
            VocalRegister::Head => 1.2,
            VocalRegister::Mix => 1.0,
            VocalRegister::Falsetto => 0.6,
        }
    }
}

/// Resonance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceConfig {
    /// Chest resonance factor (0.0-1.0)
    pub chest_resonance: f32,
    /// Head resonance factor (0.0-1.0)
    pub head_resonance: f32,
    /// Nasal resonance factor (0.0-1.0)
    pub nasal_resonance: f32,
    /// Oral cavity size factor (0.0-2.0)
    pub oral_cavity_size: f32,
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            chest_resonance: 0.7,
            head_resonance: 0.8,
            nasal_resonance: 0.2,
            oral_cavity_size: 1.0,
        }
    }
}

/// Formant frequency adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantAdjustment {
    /// F1 (first formant) adjustment factor
    pub f1_adjustment: f32,
    /// F2 (second formant) adjustment factor
    pub f2_adjustment: f32,
    /// F3 (third formant) adjustment factor
    pub f3_adjustment: f32,
    /// F4 (fourth formant) adjustment factor
    pub f4_adjustment: f32,
}

impl Default for FormantAdjustment {
    fn default() -> Self {
        Self {
            f1_adjustment: 1.0,
            f2_adjustment: 1.0,
            f3_adjustment: 1.0,
            f4_adjustment: 1.0,
        }
    }
}

/// Singing-specific synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingConfig {
    /// Base synthesis configuration
    pub base_config: SynthesisConfig,
    /// Singing technique configuration
    pub technique: SingingTechnique,
    /// Musical phrase
    pub phrase: MusicalPhrase,
    /// Pitch bend range in semitones
    pub pitch_bend_range: f32,
    /// Portamento (glide) between notes
    pub portamento_enabled: bool,
    /// Portamento duration in seconds
    pub portamento_duration: f32,
    /// Lyric timing alignment
    pub lyric_timing_strict: bool,
}

impl Default for SingingConfig {
    fn default() -> Self {
        Self {
            base_config: SynthesisConfig::default(),
            technique: SingingTechnique::default(),
            phrase: MusicalPhrase::new(120.0),
            pitch_bend_range: 2.0,
            portamento_enabled: true,
            portamento_duration: 0.1,
            lyric_timing_strict: false,
        }
    }
}

/// Singing voice synthesizer
pub struct SingingVoiceSynthesizer {
    /// Synthesizer configuration
    config: SingingConfig,
    /// Pitch contour cache
    pitch_contour_cache: HashMap<String, Vec<f32>>,
    /// Timing cache
    timing_cache: HashMap<String, Vec<f32>>,
}

impl SingingVoiceSynthesizer {
    /// Create a new singing voice synthesizer
    pub fn new(config: SingingConfig) -> Self {
        Self {
            config,
            pitch_contour_cache: HashMap::new(),
            timing_cache: HashMap::new(),
        }
    }

    /// Generate pitch contour for a musical phrase
    pub fn generate_pitch_contour(&mut self, phrase: &MusicalPhrase) -> Result<Vec<f32>> {
        let cache_key = format!("{phrase:?}");

        if let Some(cached) = self.pitch_contour_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut contour = Vec::new();
        let beats_per_second = phrase.tempo / 60.0;
        let samples_per_second = 100.0; // 100 Hz sampling rate for pitch contour

        for note in &phrase.notes {
            let duration_seconds = note.duration / beats_per_second;
            let num_samples = (duration_seconds * samples_per_second) as usize;

            // Generate pitch contour for this note
            let mut note_contour = self.generate_note_pitch_contour(note, num_samples)?;

            // Apply vibrato if enabled
            if self.config.technique.vibrato.enabled {
                self.apply_vibrato(
                    &mut note_contour,
                    &self.config.technique.vibrato,
                    duration_seconds,
                )?;
            }

            contour.extend(note_contour);
        }

        // Apply portamento between notes
        if self.config.portamento_enabled {
            self.apply_portamento(&mut contour, phrase)?;
        }

        // Cache the result
        self.pitch_contour_cache.insert(cache_key, contour.clone());

        Ok(contour)
    }

    /// Generate pitch contour for a single note
    fn generate_note_pitch_contour(
        &self,
        note: &MusicalNote,
        num_samples: usize,
    ) -> Result<Vec<f32>> {
        let mut contour = vec![note.frequency; num_samples];

        // Apply slight pitch variations for naturalness
        let mut pitch_variation = 0.0;
        for (i, sample) in contour.iter_mut().enumerate() {
            let t = i as f32 / num_samples as f32;

            // Add subtle pitch drift
            pitch_variation += (t * 0.1 - 0.05) * 0.01;
            *sample *= 1.0 + pitch_variation;

            // Add attack and release curves
            let attack_factor = if t < 0.1 { t * 10.0 } else { 1.0 };
            let release_factor = if t > 0.9 { (1.0 - t) * 10.0 } else { 1.0 };
            let envelope = attack_factor * release_factor;

            *sample *= envelope.max(0.1);
        }

        Ok(contour)
    }

    /// Apply vibrato to pitch contour
    fn apply_vibrato(
        &self,
        contour: &mut [f32],
        vibrato: &SingingVibratoConfig,
        duration: f32,
    ) -> Result<()> {
        let samples_per_second = contour.len() as f32 / duration;

        for (i, sample) in contour.iter_mut().enumerate() {
            let t = i as f32 / samples_per_second;

            // Apply vibrato onset delay
            if t < vibrato.onset_delay {
                continue;
            }

            // Apply vibrato fade-in
            let fade_in_factor = if t < vibrato.onset_delay + vibrato.fade_in_duration {
                (t - vibrato.onset_delay) / vibrato.fade_in_duration
            } else {
                1.0
            };

            // Generate vibrato modulation
            let vibrato_phase =
                2.0 * std::f32::consts::PI * vibrato.rate * (t - vibrato.onset_delay);
            let vibrato_factor =
                1.0 + (vibrato.depth / 1200.0) * vibrato_phase.sin() * fade_in_factor;

            *sample *= vibrato_factor;
        }

        Ok(())
    }

    /// Apply portamento between notes
    fn apply_portamento(&self, contour: &mut [f32], phrase: &MusicalPhrase) -> Result<()> {
        // Check if we have at least 2 notes for portamento
        if phrase.notes.len() < 2 {
            return Ok(());
        }

        let beats_per_second = phrase.tempo / 60.0;
        let samples_per_second = 100.0;

        let mut current_pos = 0;

        for i in 0..phrase.notes.len() - 1 {
            let current_note = &phrase.notes[i];
            let next_note = &phrase.notes[i + 1];

            let current_duration = current_note.duration / beats_per_second;
            let current_samples = (current_duration * samples_per_second) as usize;

            // Calculate portamento region
            let portamento_samples =
                (self.config.portamento_duration * samples_per_second) as usize;
            let portamento_start = current_pos + current_samples - portamento_samples / 2;
            let portamento_end = current_pos + current_samples + portamento_samples / 2;

            // Apply portamento
            for j in portamento_start..portamento_end.min(contour.len()) {
                let progress = (j - portamento_start) as f32 / portamento_samples as f32;
                let interpolated_freq =
                    current_note.frequency * (1.0 - progress) + next_note.frequency * progress;

                if j < contour.len() {
                    contour[j] = interpolated_freq;
                }
            }

            current_pos += current_samples;
        }

        Ok(())
    }

    /// Generate musical note timing
    pub fn generate_note_timing(&mut self, phrase: &MusicalPhrase) -> Result<Vec<f32>> {
        let cache_key = format!("{phrase:?}");

        if let Some(cached) = self.timing_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut timing = Vec::new();
        let beats_per_second = phrase.tempo / 60.0;
        let mut current_time = 0.0;

        for note in &phrase.notes {
            let duration_seconds = note.duration / beats_per_second;

            // Apply articulation timing modifications
            let adjusted_duration = if let Some(articulation) = note.articulation {
                duration_seconds * articulation.duration_scale()
            } else {
                duration_seconds
            };

            timing.push(current_time);
            current_time += adjusted_duration;

            // Add breath marks
            for &breath_pos in &phrase.breath_marks {
                let breath_time = breath_pos / beats_per_second;
                if breath_time > current_time - adjusted_duration && breath_time < current_time {
                    // Add breath pause
                    current_time += self.config.technique.breath_control.pause_duration;
                }
            }
        }

        // Cache the result
        self.timing_cache.insert(cache_key, timing.clone());

        Ok(timing)
    }

    /// Convert musical phrase to phoneme sequence with timing
    pub fn phrase_to_phonemes(&self, phrase: &MusicalPhrase) -> Result<Vec<(Phoneme, f32)>> {
        let mut phonemes = Vec::new();
        let beats_per_second = phrase.tempo / 60.0;

        for note in &phrase.notes {
            let duration_seconds = note.duration / beats_per_second;

            // Use provided phoneme or generate from lyrics
            let phoneme = if let Some(ref phoneme) = note.phoneme {
                phoneme.clone()
            } else if let Some(ref lyrics) = note.lyrics {
                // Simple lyrics-to-phoneme mapping (in real implementation, use G2P)
                self.lyrics_to_phoneme(lyrics)?
            } else {
                // Default vowel phoneme for humming
                Phoneme::new("a")
            };

            // Apply articulation timing
            let adjusted_duration = if let Some(articulation) = note.articulation {
                duration_seconds * articulation.duration_scale()
            } else {
                duration_seconds
            };

            phonemes.push((phoneme, adjusted_duration));
        }

        Ok(phonemes)
    }

    /// Simple lyrics to phoneme conversion
    fn lyrics_to_phoneme(&self, lyrics: &str) -> Result<Phoneme> {
        // This is a simplified implementation
        // In a real system, this would use a proper G2P system
        let phoneme_map = [
            ("a", "a"),
            ("e", "e"),
            ("i", "i"),
            ("o", "o"),
            ("u", "u"),
            ("la", "l a"),
            ("le", "l e"),
            ("li", "l i"),
            ("lo", "l o"),
            ("lu", "l u"),
            ("ma", "m a"),
            ("me", "m e"),
            ("mi", "m i"),
            ("mo", "m o"),
            ("mu", "m u"),
            ("na", "n a"),
            ("ne", "n e"),
            ("ni", "n i"),
            ("no", "n o"),
            ("nu", "n u"),
            ("da", "d a"),
            ("de", "d e"),
            ("di", "d i"),
            ("do", "d o"),
            ("du", "d u"),
            ("ta", "t a"),
            ("te", "t e"),
            ("ti", "t i"),
            ("to", "t o"),
            ("tu", "t u"),
            ("ka", "k a"),
            ("ke", "k e"),
            ("ki", "k i"),
            ("ko", "k o"),
            ("ku", "k u"),
            ("ga", "g a"),
            ("ge", "g e"),
            ("gi", "g i"),
            ("go", "g o"),
            ("gu", "g u"),
            ("sa", "s a"),
            ("se", "s e"),
            ("si", "s i"),
            ("so", "s o"),
            ("su", "s u"),
            ("za", "z a"),
            ("ze", "z e"),
            ("zi", "z i"),
            ("zo", "z o"),
            ("zu", "z u"),
            ("ha", "h a"),
            ("he", "h e"),
            ("hi", "h i"),
            ("ho", "h o"),
            ("hu", "h u"),
            ("ba", "b a"),
            ("be", "b e"),
            ("bi", "b i"),
            ("bo", "b o"),
            ("bu", "b u"),
            ("pa", "p a"),
            ("pe", "p e"),
            ("pi", "p i"),
            ("po", "p o"),
            ("pu", "p u"),
            ("fa", "f a"),
            ("fe", "f e"),
            ("fi", "f i"),
            ("fo", "f o"),
            ("fu", "f u"),
            ("va", "v a"),
            ("ve", "v e"),
            ("vi", "v i"),
            ("vo", "v o"),
            ("vu", "v u"),
            ("ya", "j a"),
            ("ye", "j e"),
            ("yi", "j i"),
            ("yo", "j o"),
            ("yu", "j u"),
            ("ra", "r a"),
            ("re", "r e"),
            ("ri", "r i"),
            ("ro", "r o"),
            ("ru", "r u"),
            ("wa", "w a"),
            ("we", "w e"),
            ("wi", "w i"),
            ("wo", "w o"),
            ("wu", "w u"),
        ];

        let normalized = lyrics.to_lowercase();
        let phoneme_symbol = phoneme_map
            .iter()
            .find(|(syllable, _)| *syllable == normalized)
            .map(|(_, phoneme)| *phoneme)
            .unwrap_or("a"); // Default to 'a' vowel

        Ok(Phoneme::new(phoneme_symbol))
    }

    /// Apply singing-specific prosody modifications
    pub fn apply_singing_prosody(
        &self,
        phonemes: &mut [Phoneme],
        phrase: &MusicalPhrase,
    ) -> Result<()> {
        let beats_per_second = phrase.tempo / 60.0;

        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            if i < phrase.notes.len() {
                let note = &phrase.notes[i];
                let duration_seconds = note.duration / beats_per_second;

                // Set phoneme duration based on musical timing
                phoneme.duration = Some(duration_seconds);

                // Add musical features to phoneme
                if let Some(ref mut features) = phoneme.features {
                    features.insert("frequency".to_string(), note.frequency.to_string());
                    features.insert("note_name".to_string(), note.name.clone());
                    features.insert("octave".to_string(), note.octave.to_string());

                    if let Some(dynamics) = note.dynamics {
                        features.insert("dynamics".to_string(), format!("{dynamics:?}"));
                        features.insert(
                            "volume_scale".to_string(),
                            dynamics.volume_scale().to_string(),
                        );
                    }

                    if let Some(articulation) = note.articulation {
                        features.insert("articulation".to_string(), format!("{articulation:?}"));
                        features.insert(
                            "duration_scale".to_string(),
                            articulation.duration_scale().to_string(),
                        );
                    }
                } else {
                    let mut features = HashMap::new();
                    features.insert("frequency".to_string(), note.frequency.to_string());
                    features.insert("note_name".to_string(), note.name.clone());
                    features.insert("octave".to_string(), note.octave.to_string());
                    phoneme.features = Some(features);
                }
            }
        }

        Ok(())
    }

    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.pitch_contour_cache.clear();
        self.timing_cache.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &SingingConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SingingConfig) {
        self.config = config;
        self.clear_caches(); // Clear caches when config changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_musical_note_creation() {
        let note = MusicalNote::new("C".to_string(), 4, 1.0);
        assert_eq!(note.name, "C");
        assert_eq!(note.octave, 4);
        assert_eq!(note.duration, 1.0);
        assert!((note.frequency - 261.6).abs() < 1.0); // C4 is approximately 261.6 Hz
    }

    #[test]
    fn test_musical_note_frequency_calculation() {
        let a4 = MusicalNote::new("A".to_string(), 4, 1.0);
        assert!((a4.frequency - 440.0).abs() < 0.1); // A4 = 440 Hz

        let c4 = MusicalNote::new("C".to_string(), 4, 1.0);
        assert!((c4.frequency - 261.6).abs() < 1.0); // C4 ≈ 261.6 Hz
    }

    #[test]
    fn test_musical_note_with_phoneme() {
        let note = MusicalNote::new("C".to_string(), 4, 1.0)
            .with_phoneme(Phoneme::new("a"))
            .with_lyrics("la".to_string());

        assert!(note.phoneme.is_some());
        assert_eq!(note.phoneme.unwrap().symbol, "a");
        assert_eq!(note.lyrics, Some("la".to_string()));
    }

    #[test]
    fn test_dynamics_marking_volume_scale() {
        assert_eq!(DynamicsMarking::Ppp.volume_scale(), 0.1);
        assert_eq!(DynamicsMarking::Mf.volume_scale(), 0.7);
        assert_eq!(DynamicsMarking::Fff.volume_scale(), 1.0);
    }

    #[test]
    fn test_articulation_marking_duration_scale() {
        assert_eq!(ArticulationMarking::Staccato.duration_scale(), 0.5);
        assert_eq!(ArticulationMarking::Legato.duration_scale(), 1.0);
        assert_eq!(ArticulationMarking::Marcato.duration_scale(), 0.8);
    }

    #[test]
    fn test_musical_phrase_creation() {
        let phrase = MusicalPhrase::new(120.0);
        assert_eq!(phrase.tempo, 120.0);
        assert_eq!(phrase.time_signature_numerator, 4);
        assert_eq!(phrase.time_signature_denominator, 4);
        assert!(phrase.notes.is_empty());
    }

    #[test]
    fn test_musical_phrase_add_note() {
        let mut phrase = MusicalPhrase::new(120.0);
        let note = MusicalNote::new("C".to_string(), 4, 1.0);
        phrase.add_note(note);

        assert_eq!(phrase.notes.len(), 1);
        assert_eq!(phrase.notes[0].name, "C");
    }

    #[test]
    fn test_musical_phrase_total_duration() {
        let mut phrase = MusicalPhrase::new(120.0); // 120 BPM = 2 beats per second
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0)); // 1 beat
        phrase.add_note(MusicalNote::new("D".to_string(), 4, 2.0)); // 2 beats

        // Total: 3 beats at 2 beats/second = 1.5 seconds
        assert!((phrase.total_duration() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_musical_phrase_frequency_range() {
        let mut phrase = MusicalPhrase::new(120.0);
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0)); // ~261.6 Hz
        phrase.add_note(MusicalNote::new("G".to_string(), 4, 1.0)); // ~392.0 Hz

        let (min_freq, max_freq) = phrase.frequency_range();
        assert!(min_freq < 300.0);
        assert!(max_freq > 350.0);
    }

    #[test]
    fn test_voice_type_frequency_range() {
        let soprano_range = VoiceType::Soprano.frequency_range();
        let bass_range = VoiceType::Bass.frequency_range();

        assert!(soprano_range.0 > bass_range.0); // Soprano min > Bass min
        assert!(soprano_range.1 > bass_range.1); // Soprano max > Bass max
    }

    #[test]
    fn test_vocal_register_resonance_adjustment() {
        assert_eq!(VocalRegister::Chest.resonance_adjustment(), 0.8);
        assert_eq!(VocalRegister::Head.resonance_adjustment(), 1.2);
        assert_eq!(VocalRegister::Mix.resonance_adjustment(), 1.0);
    }

    #[test]
    fn test_vibrato_config_default() {
        let vibrato = SingingVibratoConfig::default();
        assert_eq!(vibrato.rate, 6.0);
        assert_eq!(vibrato.depth, 100.0);
        assert_eq!(vibrato.onset_delay, 0.3);
        assert!(vibrato.enabled);
    }

    #[test]
    fn test_breath_control_config_default() {
        let breath = BreathControlConfig::default();
        assert_eq!(breath.support_strength, 0.8);
        assert_eq!(breath.flow_rate, 0.7);
        assert_eq!(breath.pause_duration, 0.3);
        assert!(breath.add_breath_sounds);
    }

    #[test]
    fn test_singing_config_default() {
        let config = SingingConfig::default();
        assert_eq!(config.phrase.tempo, 120.0);
        assert_eq!(config.pitch_bend_range, 2.0);
        assert!(config.portamento_enabled);
        assert_eq!(config.portamento_duration, 0.1);
    }

    #[test]
    fn test_singing_voice_synthesizer_creation() {
        let config = SingingConfig::default();
        let synthesizer = SingingVoiceSynthesizer::new(config);

        assert_eq!(synthesizer.config.phrase.tempo, 120.0);
        assert!(synthesizer.pitch_contour_cache.is_empty());
        assert!(synthesizer.timing_cache.is_empty());
    }

    #[test]
    fn test_generate_pitch_contour() {
        let mut phrase = MusicalPhrase::new(120.0);
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0));
        phrase.add_note(MusicalNote::new("D".to_string(), 4, 1.0));

        let config = SingingConfig {
            phrase: phrase.clone(),
            ..Default::default()
        };

        let mut synthesizer = SingingVoiceSynthesizer::new(config);
        let contour = synthesizer.generate_pitch_contour(&phrase).unwrap();

        assert!(!contour.is_empty());
        assert!(contour.iter().all(|&f| f > 0.0)); // All frequencies should be positive
    }

    #[test]
    fn test_generate_note_timing() {
        let mut phrase = MusicalPhrase::new(120.0);
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0));
        phrase.add_note(MusicalNote::new("D".to_string(), 4, 0.5));

        let config = SingingConfig {
            phrase: phrase.clone(),
            ..Default::default()
        };

        let mut synthesizer = SingingVoiceSynthesizer::new(config);
        let timing = synthesizer.generate_note_timing(&phrase).unwrap();

        assert_eq!(timing.len(), 2);
        assert_eq!(timing[0], 0.0); // First note starts at 0
        assert!(timing[1] > 0.0); // Second note starts later
    }

    #[test]
    fn test_phrase_to_phonemes() {
        let mut phrase = MusicalPhrase::new(120.0);
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0).with_phoneme(Phoneme::new("a")));
        phrase.add_note(MusicalNote::new("D".to_string(), 4, 0.5).with_lyrics("la".to_string()));

        let config = SingingConfig {
            phrase: phrase.clone(),
            ..Default::default()
        };

        let synthesizer = SingingVoiceSynthesizer::new(config);
        let phonemes = synthesizer.phrase_to_phonemes(&phrase).unwrap();

        assert_eq!(phonemes.len(), 2);
        assert_eq!(phonemes[0].0.symbol, "a");
        assert!(phonemes[0].1 > 0.0); // Duration should be positive
    }

    #[test]
    fn test_lyrics_to_phoneme() {
        let config = SingingConfig::default();
        let synthesizer = SingingVoiceSynthesizer::new(config);

        let phoneme = synthesizer.lyrics_to_phoneme("la").unwrap();
        assert_eq!(phoneme.symbol, "l a");

        let phoneme2 = synthesizer.lyrics_to_phoneme("unknown").unwrap();
        assert_eq!(phoneme2.symbol, "a"); // Default fallback
    }

    #[test]
    fn test_apply_singing_prosody() {
        let mut phrase = MusicalPhrase::new(120.0);
        phrase.add_note(MusicalNote::new("C".to_string(), 4, 1.0));

        let config = SingingConfig {
            phrase: phrase.clone(),
            ..Default::default()
        };

        let synthesizer = SingingVoiceSynthesizer::new(config);
        let mut phonemes = vec![Phoneme::new("a")];

        synthesizer
            .apply_singing_prosody(&mut phonemes, &phrase)
            .unwrap();

        assert!(phonemes[0].duration.is_some());
        assert!(phonemes[0].features.is_some());

        let features = phonemes[0].features.as_ref().unwrap();
        assert!(features.contains_key("frequency"));
        assert!(features.contains_key("note_name"));
    }

    #[test]
    fn test_clear_caches() {
        let config = SingingConfig::default();
        let mut synthesizer = SingingVoiceSynthesizer::new(config);

        // Add something to cache
        let phrase = MusicalPhrase::new(120.0);
        let _ = synthesizer.generate_pitch_contour(&phrase);

        // Clear caches
        synthesizer.clear_caches();

        assert!(synthesizer.pitch_contour_cache.is_empty());
        assert!(synthesizer.timing_cache.is_empty());
    }
}
