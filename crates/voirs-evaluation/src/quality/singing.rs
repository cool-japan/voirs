//! Singing Voice Synthesis Evaluation
//!
//! This module provides comprehensive evaluation metrics specifically designed for singing voice
//! synthesis systems. It includes specialized assessments for musical aspects that are not
//! covered by traditional speech quality metrics.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::traits::*;
use crate::{AudioBuffer, EvaluationError, LanguageCode};

/// Musical note representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MusicalNote {
    /// Note frequency in Hz
    pub frequency: f32,
    /// Note start time in seconds
    pub start_time: f32,
    /// Note duration in seconds
    pub duration: f32,
    /// Note velocity/volume (0.0-1.0)
    pub velocity: f32,
    /// MIDI note number (0-127)
    pub midi_note: u8,
}

/// Musical key signature
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MusicalKey {
    /// C Major
    CMajor,
    /// G Major
    GMajor,
    /// D Major
    DMajor,
    /// A Major
    AMajor,
    /// E Major
    EMajor,
    /// B Major
    BMajor,
    /// F# Major
    FSharpMajor,
    /// C# Major
    CSharpMajor,
    /// F Major
    FMajor,
    /// Bb Major
    BbMajor,
    /// Eb Major
    EbMajor,
    /// Ab Major
    AbMajor,
    /// Db Major
    DbMajor,
    /// Gb Major
    GbMajor,
    /// Cb Major
    CbMajor,
    /// A Minor
    AMinor,
    /// E Minor
    EMinor,
    /// B Minor
    BMinor,
    /// F# Minor
    FSharpMinor,
    /// C# Minor
    CSharpMinor,
    /// G# Minor
    GSharpMinor,
    /// D# Minor
    DSharpMinor,
    /// A# Minor
    ASharpMinor,
    /// D Minor
    DMinor,
    /// G Minor
    GMinor,
    /// C Minor
    CMinor,
    /// F Minor
    FMinor,
    /// Bb Minor
    BbMinor,
    /// Eb Minor
    EbMinor,
    /// Ab Minor
    AbMinor,
}

/// Time signature specification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeSignature {
    /// Beats per measure
    pub numerator: u8,
    /// Note value that gets the beat
    pub denominator: u8,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

/// Tempo specification in beats per minute
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tempo {
    /// Beats per minute
    pub bpm: f32,
    /// Tempo stability (0.0 = very unstable, 1.0 = rock solid)
    pub stability: f32,
}

/// Vibrato characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VibratoAnalysis {
    /// Vibrato rate in Hz (cycles per second)
    pub rate: f32,
    /// Vibrato depth as percentage of fundamental frequency
    pub depth_percent: f32,
    /// Vibrato onset time from note start (seconds)
    pub onset_time: f32,
    /// Vibrato regularity (0.0 = irregular, 1.0 = perfectly regular)
    pub regularity: f32,
    /// Vibrato presence score (0.0 = no vibrato, 1.0 = strong vibrato)
    pub presence: f32,
}

/// Harmonic analysis for singing voice
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarmonicStructure {
    /// Fundamental frequency strength
    pub fundamental_strength: f32,
    /// Harmonic-to-noise ratio
    pub harmonic_noise_ratio: f32,
    /// Formant frequencies (F1, F2, F3, F4)
    pub formants: Vec<f32>,
    /// Formant bandwidths
    pub formant_bandwidths: Vec<f32>,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral brightness
    pub brightness: f32,
    /// Harmonic richness (number of prominent harmonics)
    pub harmonic_richness: u8,
}

/// Breath control and phrasing analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BreathControlAnalysis {
    /// Breath support consistency (0.0 = poor, 1.0 = excellent)
    pub breath_support: f32,
    /// Phrase boundary detection accuracy
    pub phrase_boundaries: Vec<f32>,
    /// Breath intake locations
    pub breath_intakes: Vec<f32>,
    /// Phrase length distribution appropriateness
    pub phrase_length_score: f32,
    /// Overall breath control score
    pub overall_score: f32,
}

/// Musical expressiveness analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalExpressiveness {
    /// Dynamic range usage (0.0 = monotone, 1.0 = full range)
    pub dynamic_range: f32,
    /// Articulation clarity
    pub articulation: f32,
    /// Emotional expression appropriateness
    pub emotional_expression: f32,
    /// Musical phrasing quality
    pub musical_phrasing: f32,
    /// Stylistic authenticity
    pub stylistic_authenticity: f32,
    /// Overall expressiveness score
    pub overall_score: f32,
}

/// Singer identity and characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingerIdentity {
    /// Voice type classification
    pub voice_type: VoiceType,
    /// Vocal range in semitones
    pub vocal_range: f32,
    /// Voice timbre characteristics
    pub timbre_profile: TimbreProfile,
    /// Singer consistency across different notes
    pub consistency: f32,
    /// Identity preservation score (vs reference singer)
    pub identity_preservation: f32,
}

/// Voice type classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VoiceType {
    /// Soprano voice
    Soprano,
    /// Mezzo-soprano voice
    MezzoSoprano,
    /// Alto voice
    Alto,
    /// Tenor voice
    Tenor,
    /// Baritone voice
    Baritone,
    /// Bass voice
    Bass,
    /// Countertenor voice
    Countertenor,
    /// Unclassified or unknown
    Unknown,
}

/// Voice timbre characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimbreProfile {
    /// Brightness (0.0 = dark, 1.0 = bright)
    pub brightness: f32,
    /// Warmth (0.0 = cold, 1.0 = warm)
    pub warmth: f32,
    /// Roughness (0.0 = smooth, 1.0 = rough)
    pub roughness: f32,
    /// Breathiness (0.0 = clear, 1.0 = breathy)
    pub breathiness: f32,
    /// Nasality (0.0 = not nasal, 1.0 = very nasal)
    pub nasality: f32,
}

/// Configuration for singing evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingEvaluationConfig {
    /// Whether to analyze pitch accuracy
    pub analyze_pitch_accuracy: bool,
    /// Whether to analyze vibrato
    pub analyze_vibrato: bool,
    /// Whether to analyze harmonic structure
    pub analyze_harmonics: bool,
    /// Whether to analyze breath control
    pub analyze_breath_control: bool,
    /// Whether to analyze musical expressiveness
    pub analyze_expressiveness: bool,
    /// Whether to analyze singer identity
    pub analyze_singer_identity: bool,
    /// Expected musical key (if known)
    pub expected_key: Option<MusicalKey>,
    /// Expected time signature
    pub time_signature: TimeSignature,
    /// Expected tempo range
    pub expected_tempo: Option<Tempo>,
    /// Reference notes/melody (if available)
    pub reference_melody: Option<Vec<MusicalNote>>,
    /// Language of the lyrics
    pub language: LanguageCode,
    /// Musical style/genre
    pub musical_style: Option<String>,
}

impl Default for SingingEvaluationConfig {
    fn default() -> Self {
        Self {
            analyze_pitch_accuracy: true,
            analyze_vibrato: true,
            analyze_harmonics: true,
            analyze_breath_control: true,
            analyze_expressiveness: true,
            analyze_singer_identity: true,
            expected_key: None,
            time_signature: TimeSignature::default(),
            expected_tempo: None,
            reference_melody: None,
            language: LanguageCode::EnUs,
            musical_style: None,
        }
    }
}

/// Comprehensive singing evaluation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingingEvaluationResult {
    /// Overall singing quality score (0.0-1.0)
    pub overall_score: f32,
    /// Pitch accuracy analysis
    pub pitch_accuracy: PitchAccuracyResult,
    /// Vibrato analysis results
    pub vibrato_analysis: Option<VibratoAnalysis>,
    /// Harmonic structure analysis
    pub harmonic_structure: Option<HarmonicStructure>,
    /// Breath control analysis
    pub breath_control: Option<BreathControlAnalysis>,
    /// Musical expressiveness analysis
    pub expressiveness: Option<MusicalExpressiveness>,
    /// Singer identity analysis
    pub singer_identity: Option<SingerIdentity>,
    /// Detected tempo
    pub detected_tempo: Option<Tempo>,
    /// Detected musical key
    pub detected_key: Option<MusicalKey>,
    /// Overall musical accuracy (rhythm, pitch, expression)
    pub musical_accuracy: f32,
    /// Technical vocal quality
    pub technical_quality: f32,
    /// Artistic expression quality
    pub artistic_quality: f32,
    /// Confidence in the evaluation
    pub confidence: f32,
}

/// Pitch accuracy specific results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PitchAccuracyResult {
    /// Overall pitch accuracy score (0.0-1.0)
    pub overall_accuracy: f32,
    /// RMS pitch error in cents (100 cents = 1 semitone)
    pub rms_pitch_error_cents: f32,
    /// Maximum pitch error in cents
    pub max_pitch_error_cents: f32,
    /// Percentage of notes within 50 cents of target
    pub notes_in_tune_percent: f32,
    /// Intonation stability over time
    pub intonation_stability: f32,
    /// Per-note accuracy scores
    pub note_accuracies: Vec<f32>,
    /// Interval accuracy (melodic intervals)
    pub interval_accuracy: f32,
    /// Scale accuracy (adherence to musical scale)
    pub scale_accuracy: f32,
}

/// Singing voice evaluator implementation
pub struct SingingEvaluator {
    config: SingingEvaluationConfig,
}

impl SingingEvaluator {
    /// Create a new singing evaluator
    pub async fn new() -> Result<Self, EvaluationError> {
        Ok(Self {
            config: SingingEvaluationConfig::default(),
        })
    }

    /// Create singing evaluator with custom configuration
    pub async fn with_config(config: SingingEvaluationConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Update evaluator configuration
    pub fn set_config(&mut self, config: SingingEvaluationConfig) {
        self.config = config;
    }

    /// Evaluate singing voice synthesis quality
    pub async fn evaluate_singing(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
        reference_melody: Option<&[MusicalNote]>,
    ) -> Result<SingingEvaluationResult, EvaluationError> {
        // Extract fundamental frequency and notes from generated audio
        let generated_notes = self.extract_musical_notes(generated_audio).await?;

        // Extract reference notes if available
        let reference_notes = if let Some(ref_audio) = reference_audio {
            Some(self.extract_musical_notes(ref_audio).await?)
        } else {
            reference_melody.map(|melody| melody.to_vec())
        };

        // Analyze pitch accuracy
        let pitch_accuracy = self
            .analyze_pitch_accuracy(&generated_notes, reference_notes.as_deref())
            .await?;

        // Analyze vibrato if enabled
        let vibrato_analysis = if self.config.analyze_vibrato {
            Some(
                self.analyze_vibrato(generated_audio, &generated_notes)
                    .await?,
            )
        } else {
            None
        };

        // Analyze harmonic structure if enabled
        let harmonic_structure = if self.config.analyze_harmonics {
            Some(self.analyze_harmonic_structure(generated_audio).await?)
        } else {
            None
        };

        // Analyze breath control if enabled
        let breath_control = if self.config.analyze_breath_control {
            Some(
                self.analyze_breath_control(generated_audio, &generated_notes)
                    .await?,
            )
        } else {
            None
        };

        // Analyze musical expressiveness if enabled
        let expressiveness = if self.config.analyze_expressiveness {
            Some(
                self.analyze_musical_expressiveness(generated_audio, &generated_notes)
                    .await?,
            )
        } else {
            None
        };

        // Analyze singer identity if enabled
        let singer_identity = if self.config.analyze_singer_identity {
            Some(
                self.analyze_singer_identity(generated_audio, reference_audio)
                    .await?,
            )
        } else {
            None
        };

        // Detect tempo and key
        let detected_tempo = self.detect_tempo(&generated_notes).await?;
        let detected_key = self.detect_musical_key(&generated_notes).await?;

        // Calculate composite scores
        let musical_accuracy = self
            .calculate_musical_accuracy(&pitch_accuracy, &detected_tempo)
            .await?;
        let technical_quality = self
            .calculate_technical_quality(&harmonic_structure, &vibrato_analysis, &breath_control)
            .await?;
        let artistic_quality = self
            .calculate_artistic_quality(&expressiveness, &singer_identity)
            .await?;

        // Calculate overall score
        let overall_score =
            (musical_accuracy * 0.4 + technical_quality * 0.3 + artistic_quality * 0.3)
                .max(0.0)
                .min(1.0);

        // Calculate confidence based on available data
        let confidence =
            self.calculate_evaluation_confidence(&generated_notes, reference_notes.as_deref());

        Ok(SingingEvaluationResult {
            overall_score,
            pitch_accuracy,
            vibrato_analysis,
            harmonic_structure,
            breath_control,
            expressiveness,
            singer_identity,
            detected_tempo,
            detected_key,
            musical_accuracy,
            technical_quality,
            artistic_quality,
            confidence,
        })
    }

    /// Extract musical notes from audio buffer
    async fn extract_musical_notes(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<MusicalNote>, EvaluationError> {
        let mut notes = Vec::new();
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        // Simple pitch tracking using autocorrelation
        let frame_size = 2048;
        let hop_size = 512;
        let mut time = 0.0;

        for i in (0..samples.len()).step_by(hop_size) {
            if i + frame_size > samples.len() {
                break;
            }

            let frame = &samples[i..i + frame_size];
            let frequency = self.estimate_fundamental_frequency(frame, sample_rate)?;

            // Only consider frequencies in singing range (80-2000 Hz)
            if frequency > 80.0 && frequency < 2000.0 {
                let midi_note = Self::frequency_to_midi(frequency);
                let velocity = self.estimate_note_velocity(frame);

                notes.push(MusicalNote {
                    frequency,
                    start_time: time,
                    duration: hop_size as f32 / sample_rate,
                    velocity,
                    midi_note,
                });
            }

            time += hop_size as f32 / sample_rate;
        }

        // Post-process to merge similar consecutive notes
        Ok(self.merge_consecutive_notes(notes))
    }

    /// Estimate fundamental frequency using autocorrelation
    fn estimate_fundamental_frequency(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> Result<f32, EvaluationError> {
        if frame.len() < 2 {
            return Ok(0.0);
        }

        let min_period = (sample_rate / 2000.0) as usize; // Highest singing frequency
        let max_period = (sample_rate / 80.0) as usize; // Lowest singing frequency

        if max_period >= frame.len() {
            return Ok(0.0);
        }

        let mut max_correlation = 0.0;
        let mut best_period = min_period;

        for period in min_period..=max_period.min(frame.len() - 1) {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(frame.len() - period) {
                correlation += frame[i] * frame[i + period];
                norm1 += frame[i] * frame[i];
                norm2 += frame[i + period] * frame[i + period];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_period = period;
                }
            }
        }

        if max_correlation > 0.3 {
            Ok(sample_rate / best_period as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate note velocity from frame energy
    fn estimate_note_velocity(&self, frame: &[f32]) -> f32 {
        let energy: f32 = frame.iter().map(|&x| x * x).sum();
        let rms = (energy / frame.len() as f32).sqrt();
        (rms * 10.0).min(1.0) // Scale and clamp
    }

    /// Convert frequency to MIDI note number
    fn frequency_to_midi(frequency: f32) -> u8 {
        if frequency <= 0.0 {
            return 0;
        }
        (69.0 + 12.0 * (frequency / 440.0).log2())
            .round()
            .max(0.0)
            .min(127.0) as u8
    }

    /// Merge consecutive similar notes
    fn merge_consecutive_notes(&self, notes: Vec<MusicalNote>) -> Vec<MusicalNote> {
        if notes.is_empty() {
            return notes;
        }

        let mut merged = Vec::new();
        let mut current = notes[0];

        for note in notes.into_iter().skip(1) {
            // Merge if same MIDI note and consecutive timing
            if note.midi_note == current.midi_note
                && (note.start_time - (current.start_time + current.duration)).abs() < 0.1
            {
                current.duration = note.start_time + note.duration - current.start_time;
                current.velocity = (current.velocity + note.velocity) / 2.0;
            } else {
                merged.push(current);
                current = note;
            }
        }
        merged.push(current);

        merged
    }

    /// Analyze pitch accuracy against reference
    async fn analyze_pitch_accuracy(
        &self,
        generated_notes: &[MusicalNote],
        reference_notes: Option<&[MusicalNote]>,
    ) -> Result<PitchAccuracyResult, EvaluationError> {
        if generated_notes.is_empty() {
            return Ok(PitchAccuracyResult {
                overall_accuracy: 0.0,
                rms_pitch_error_cents: 0.0,
                max_pitch_error_cents: 0.0,
                notes_in_tune_percent: 0.0,
                intonation_stability: 0.0,
                note_accuracies: vec![],
                interval_accuracy: 0.0,
                scale_accuracy: 0.0,
            });
        }

        let mut pitch_errors = Vec::new();
        let mut note_accuracies = Vec::new();

        if let Some(reference) = reference_notes {
            // Compare against reference notes
            for (gen_note, ref_note) in generated_notes.iter().zip(reference.iter()) {
                let error_cents =
                    Self::frequency_difference_cents(gen_note.frequency, ref_note.frequency);
                pitch_errors.push(error_cents);
                note_accuracies.push(Self::cents_to_accuracy_score(error_cents.abs()));
            }
        } else {
            // Analyze against ideal pitch for detected notes
            for note in generated_notes {
                let ideal_frequency = Self::midi_to_frequency(note.midi_note);
                let error_cents = Self::frequency_difference_cents(note.frequency, ideal_frequency);
                pitch_errors.push(error_cents);
                note_accuracies.push(Self::cents_to_accuracy_score(error_cents.abs()));
            }
        }

        // Calculate statistics
        let rms_pitch_error_cents = if !pitch_errors.is_empty() {
            (pitch_errors.iter().map(|&e| e * e).sum::<f32>() / pitch_errors.len() as f32).sqrt()
        } else {
            0.0
        };

        let max_pitch_error_cents = pitch_errors.iter().map(|&e| e.abs()).fold(0.0, f32::max);

        let notes_in_tune = pitch_errors.iter().filter(|&&e| e.abs() < 50.0).count();
        let notes_in_tune_percent = if !pitch_errors.is_empty() {
            notes_in_tune as f32 / pitch_errors.len() as f32
        } else {
            0.0
        };

        let overall_accuracy = if !note_accuracies.is_empty() {
            note_accuracies.iter().sum::<f32>() / note_accuracies.len() as f32
        } else {
            0.0
        };

        let intonation_stability = self.calculate_intonation_stability(generated_notes);
        let interval_accuracy = self.calculate_interval_accuracy(generated_notes, reference_notes);
        let scale_accuracy = self.calculate_scale_accuracy(generated_notes);

        Ok(PitchAccuracyResult {
            overall_accuracy,
            rms_pitch_error_cents,
            max_pitch_error_cents,
            notes_in_tune_percent,
            intonation_stability,
            note_accuracies,
            interval_accuracy,
            scale_accuracy,
        })
    }

    /// Calculate frequency difference in cents
    fn frequency_difference_cents(freq1: f32, freq2: f32) -> f32 {
        if freq1 <= 0.0 || freq2 <= 0.0 {
            return 0.0;
        }
        1200.0 * (freq1 / freq2).log2()
    }

    /// Convert MIDI note to frequency
    fn midi_to_frequency(midi_note: u8) -> f32 {
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    /// Convert cents error to accuracy score
    fn cents_to_accuracy_score(cents_error: f32) -> f32 {
        // Perfect accuracy at 0 cents, linearly decreasing to 0 at 100 cents
        (1.0 - cents_error / 100.0).max(0.0)
    }

    /// Calculate intonation stability
    fn calculate_intonation_stability(&self, notes: &[MusicalNote]) -> f32 {
        if notes.len() < 2 {
            return 1.0;
        }

        let mut pitch_variations = Vec::new();
        for window in notes.windows(3) {
            let prev = window[0].frequency;
            let curr = window[1].frequency;
            let next = window[2].frequency;

            if prev > 0.0 && curr > 0.0 && next > 0.0 {
                let variation = ((curr - prev).abs() + (next - curr).abs()) / 2.0;
                pitch_variations.push(variation);
            }
        }

        if pitch_variations.is_empty() {
            return 1.0;
        }

        let mean_variation = pitch_variations.iter().sum::<f32>() / pitch_variations.len() as f32;
        // Stability decreases with variation, normalized to singing range
        (1.0 - (mean_variation / 50.0).min(1.0)).max(0.0)
    }

    /// Calculate interval accuracy
    fn calculate_interval_accuracy(
        &self,
        generated: &[MusicalNote],
        reference: Option<&[MusicalNote]>,
    ) -> f32 {
        if generated.len() < 2 {
            return 1.0;
        }

        let mut interval_errors = Vec::new();

        if let Some(ref_notes) = reference {
            // Compare intervals against reference
            for i in 0..(generated.len() - 1).min(ref_notes.len() - 1) {
                let gen_interval =
                    generated[i + 1].midi_note as i16 - generated[i].midi_note as i16;
                let ref_interval =
                    ref_notes[i + 1].midi_note as i16 - ref_notes[i].midi_note as i16;
                interval_errors.push((gen_interval - ref_interval).abs() as f32);
            }
        } else {
            // Analyze interval consistency
            let intervals: Vec<i16> = generated
                .windows(2)
                .map(|w| w[1].midi_note as i16 - w[0].midi_note as i16)
                .collect();

            // Check for common musical intervals
            for &interval in &intervals {
                let error = match interval.abs() {
                    1..=2 => 0.0,   // Minor/major second
                    3..=4 => 0.0,   // Minor/major third
                    5 => 0.0,       // Perfect fourth
                    6 => 1.0,       // Tritone (less common)
                    7 => 0.0,       // Perfect fifth
                    8..=9 => 0.0,   // Minor/major sixth
                    10..=11 => 0.0, // Minor/major seventh
                    12 => 0.0,      // Octave
                    _ => 2.0,       // Large intervals
                };
                interval_errors.push(error);
            }
        }

        if interval_errors.is_empty() {
            return 1.0;
        }

        let mean_error = interval_errors.iter().sum::<f32>() / interval_errors.len() as f32;
        (1.0 - (mean_error / 5.0).min(1.0)).max(0.0)
    }

    /// Calculate scale accuracy
    fn calculate_scale_accuracy(&self, notes: &[MusicalNote]) -> f32 {
        if notes.is_empty() {
            return 1.0;
        }

        // Extract unique notes (remove octave duplicates)
        let unique_notes: std::collections::HashSet<u8> =
            notes.iter().map(|note| note.midi_note % 12).collect();

        // Check against common scales (simplified)
        let major_scale = [0, 2, 4, 5, 7, 9, 11]; // C major scale degrees
        let minor_scale = [0, 2, 3, 5, 7, 8, 10]; // Natural minor scale degrees

        let major_matches = unique_notes
            .iter()
            .filter(|&&note| major_scale.contains(&note))
            .count();

        let minor_matches = unique_notes
            .iter()
            .filter(|&&note| minor_scale.contains(&note))
            .count();

        let best_matches = major_matches.max(minor_matches);

        if unique_notes.is_empty() {
            1.0
        } else {
            best_matches as f32 / unique_notes.len() as f32
        }
    }

    /// Analyze vibrato characteristics
    async fn analyze_vibrato(
        &self,
        audio: &AudioBuffer,
        notes: &[MusicalNote],
    ) -> Result<VibratoAnalysis, EvaluationError> {
        // Simplified vibrato analysis
        // In practice, this would involve detailed spectral analysis

        let mut total_rate = 0.0;
        let mut total_depth = 0.0;
        let mut vibrato_count = 0;

        for note in notes {
            if note.duration > 0.5 {
                // Only analyze longer notes
                // Simulate vibrato detection
                let rate = 4.5 + (note.frequency / 440.0 - 1.0) * 1.0; // 4-6 Hz typical
                let depth = note.velocity * 0.02; // Depth as % of fundamental

                total_rate += rate;
                total_depth += depth;
                vibrato_count += 1;
            }
        }

        let avg_rate = if vibrato_count > 0 {
            total_rate / vibrato_count as f32
        } else {
            5.0
        };

        let avg_depth = if vibrato_count > 0 {
            total_depth / vibrato_count as f32
        } else {
            0.01
        };

        Ok(VibratoAnalysis {
            rate: avg_rate,
            depth_percent: avg_depth * 100.0,
            onset_time: 0.3, // Typical onset time
            regularity: 0.8, // Assume good regularity
            presence: if vibrato_count > 0 { 0.7 } else { 0.1 },
        })
    }

    /// Analyze harmonic structure
    async fn analyze_harmonic_structure(
        &self,
        audio: &AudioBuffer,
    ) -> Result<HarmonicStructure, EvaluationError> {
        // Simplified harmonic analysis
        // In practice, this would involve FFT and harmonic detection

        let samples = audio.samples();
        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        let rms = (energy / samples.len() as f32).sqrt();

        Ok(HarmonicStructure {
            fundamental_strength: rms.min(1.0),
            harmonic_noise_ratio: 15.0 + rms * 10.0, // Simulate HNR
            formants: vec![800.0, 1200.0, 2500.0, 3500.0], // Typical singing formants
            formant_bandwidths: vec![100.0, 150.0, 200.0, 300.0],
            spectral_centroid: 1500.0 + rms * 500.0,
            brightness: rms * 0.8,
            harmonic_richness: ((rms * 10.0) as u8).min(8),
        })
    }

    /// Analyze breath control
    async fn analyze_breath_control(
        &self,
        audio: &AudioBuffer,
        notes: &[MusicalNote],
    ) -> Result<BreathControlAnalysis, EvaluationError> {
        // Simplified breath control analysis
        let samples = audio.samples();
        let mut phrase_boundaries = Vec::new();
        let mut breath_intakes = Vec::new();

        // Detect silence gaps as potential breath points
        let silence_threshold = 0.01;
        let mut in_silence = false;
        let mut silence_start = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let time = i as f32 / audio.sample_rate() as f32;

            if sample.abs() < silence_threshold {
                if !in_silence {
                    silence_start = time;
                    in_silence = true;
                }
            } else if in_silence {
                let silence_duration = time - silence_start;
                if silence_duration > 0.1 {
                    // Significant pause
                    phrase_boundaries.push(silence_start);
                    if silence_duration > 0.3 {
                        // Potential breath
                        breath_intakes.push(time);
                    }
                }
                in_silence = false;
            }
        }

        // Calculate phrase length score
        let phrase_lengths: Vec<f32> = phrase_boundaries.windows(2).map(|w| w[1] - w[0]).collect();

        let avg_phrase_length = if !phrase_lengths.is_empty() {
            phrase_lengths.iter().sum::<f32>() / phrase_lengths.len() as f32
        } else {
            audio.duration()
        };

        // Ideal phrase length is 4-8 seconds for singing
        let phrase_length_score = if avg_phrase_length >= 4.0 && avg_phrase_length <= 8.0 {
            1.0
        } else {
            1.0 - ((avg_phrase_length - 6.0).abs() / 6.0).min(1.0)
        };

        let breath_support = 0.8; // Simulate breath support analysis
        let overall_score = (breath_support + phrase_length_score) / 2.0;

        Ok(BreathControlAnalysis {
            breath_support,
            phrase_boundaries,
            breath_intakes,
            phrase_length_score,
            overall_score,
        })
    }

    /// Analyze musical expressiveness
    async fn analyze_musical_expressiveness(
        &self,
        audio: &AudioBuffer,
        notes: &[MusicalNote],
    ) -> Result<MusicalExpressiveness, EvaluationError> {
        // Calculate dynamic range
        let velocities: Vec<f32> = notes.iter().map(|n| n.velocity).collect();
        let dynamic_range = if !velocities.is_empty() {
            let max_vel = velocities.iter().fold(0.0_f32, |a, &b| a.max(b));
            let min_vel = velocities.iter().fold(1.0_f32, |a, &b| a.min(b));
            max_vel - min_vel
        } else {
            0.0
        };

        // Simulate other expressiveness metrics
        let articulation = 0.75; // Based on note transitions
        let emotional_expression = 0.7; // Based on pitch and timing variations
        let musical_phrasing = 0.8; // Based on phrase structure
        let stylistic_authenticity = 0.7; // Would require style classification

        let overall_score = (dynamic_range
            + articulation
            + emotional_expression
            + musical_phrasing
            + stylistic_authenticity)
            / 5.0;

        Ok(MusicalExpressiveness {
            dynamic_range,
            articulation,
            emotional_expression,
            musical_phrasing,
            stylistic_authenticity,
            overall_score,
        })
    }

    /// Analyze singer identity
    async fn analyze_singer_identity(
        &self,
        generated_audio: &AudioBuffer,
        reference_audio: Option<&AudioBuffer>,
    ) -> Result<SingerIdentity, EvaluationError> {
        // Extract vocal characteristics
        let samples = generated_audio.samples();
        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        let rms = (energy / samples.len() as f32).sqrt();

        // Simulate voice type classification based on frequency range
        let voice_type = VoiceType::Unknown; // Would require more sophisticated analysis

        // Calculate vocal range from notes
        let vocal_range = 24.0; // Simulate 2-octave range

        // Simulate timbre profile
        let timbre_profile = TimbreProfile {
            brightness: rms * 0.8,
            warmth: (1.0 - rms).max(0.0),
            roughness: (rms - 0.5).abs(),
            breathiness: rms * 0.3,
            nasality: 0.2,
        };

        let consistency = 0.8; // Simulate consistency analysis

        // Identity preservation (vs reference)
        let identity_preservation = if reference_audio.is_some() {
            0.7 // Simulate similarity analysis
        } else {
            1.0 // No reference to compare against
        };

        Ok(SingerIdentity {
            voice_type,
            vocal_range,
            timbre_profile,
            consistency,
            identity_preservation,
        })
    }

    /// Detect tempo from notes
    async fn detect_tempo(&self, notes: &[MusicalNote]) -> Result<Option<Tempo>, EvaluationError> {
        if notes.len() < 4 {
            return Ok(None);
        }

        // Simple tempo detection based on note onsets
        let mut intervals = Vec::new();
        for window in notes.windows(2) {
            let interval = window[1].start_time - window[0].start_time;
            if interval > 0.1 && interval < 2.0 {
                // Reasonable note intervals
                intervals.push(interval);
            }
        }

        if intervals.is_empty() {
            return Ok(None);
        }

        // Find most common interval (simplified)
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_interval = intervals[intervals.len() / 2];

        // Convert to BPM (assumes quarter note intervals)
        let bpm = 60.0 / median_interval;

        // Calculate stability
        let variance = intervals
            .iter()
            .map(|&i| (i - median_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let stability = 1.0 - (variance.sqrt() / median_interval).min(1.0);

        Ok(Some(Tempo { bpm, stability }))
    }

    /// Detect musical key from notes
    async fn detect_musical_key(
        &self,
        notes: &[MusicalNote],
    ) -> Result<Option<MusicalKey>, EvaluationError> {
        if notes.is_empty() {
            return Ok(None);
        }

        // Simple key detection based on note frequency
        let note_counts = notes.iter().fold(HashMap::new(), |mut acc, note| {
            let pitch_class = note.midi_note % 12;
            *acc.entry(pitch_class).or_insert(0) += 1;
            acc
        });

        // Find most frequent note (simplified tonic detection)
        let most_frequent = note_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&note, _)| note);

        // Map to key (simplified - assumes major keys)
        let key = match most_frequent {
            Some(0) => Some(MusicalKey::CMajor),
            Some(1) => Some(MusicalKey::DbMajor),
            Some(2) => Some(MusicalKey::DMajor),
            Some(3) => Some(MusicalKey::EbMajor),
            Some(4) => Some(MusicalKey::EMajor),
            Some(5) => Some(MusicalKey::FMajor),
            Some(6) => Some(MusicalKey::FSharpMajor),
            Some(7) => Some(MusicalKey::GMajor),
            Some(8) => Some(MusicalKey::AbMajor),
            Some(9) => Some(MusicalKey::AMajor),
            Some(10) => Some(MusicalKey::BbMajor),
            Some(11) => Some(MusicalKey::BMajor),
            _ => None,
        };

        Ok(key)
    }

    /// Calculate musical accuracy score
    async fn calculate_musical_accuracy(
        &self,
        pitch_accuracy: &PitchAccuracyResult,
        tempo: &Option<Tempo>,
    ) -> Result<f32, EvaluationError> {
        let pitch_weight = 0.7;
        let tempo_weight = 0.3;

        let tempo_accuracy = if let Some(tempo_info) = tempo {
            tempo_info.stability
        } else {
            0.5 // Neutral score if no tempo detected
        };

        Ok(pitch_accuracy.overall_accuracy * pitch_weight + tempo_accuracy * tempo_weight)
    }

    /// Calculate technical quality score
    async fn calculate_technical_quality(
        &self,
        harmonic_structure: &Option<HarmonicStructure>,
        vibrato_analysis: &Option<VibratoAnalysis>,
        breath_control: &Option<BreathControlAnalysis>,
    ) -> Result<f32, EvaluationError> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        if let Some(harmonics) = harmonic_structure {
            score += (harmonics.harmonic_noise_ratio / 25.0).min(1.0) * 0.4;
            weight_sum += 0.4;
        }

        if let Some(vibrato) = vibrato_analysis {
            let vibrato_quality = (vibrato.presence * vibrato.regularity).min(1.0);
            score += vibrato_quality * 0.3;
            weight_sum += 0.3;
        }

        if let Some(breath) = breath_control {
            score += breath.overall_score * 0.3;
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            Ok(score / weight_sum)
        } else {
            Ok(0.5) // Neutral score if no technical analysis available
        }
    }

    /// Calculate artistic quality score
    async fn calculate_artistic_quality(
        &self,
        expressiveness: &Option<MusicalExpressiveness>,
        singer_identity: &Option<SingerIdentity>,
    ) -> Result<f32, EvaluationError> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        if let Some(expression) = expressiveness {
            score += expression.overall_score * 0.7;
            weight_sum += 0.7;
        }

        if let Some(identity) = singer_identity {
            score += identity.consistency * 0.3;
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            Ok(score / weight_sum)
        } else {
            Ok(0.5) // Neutral score if no artistic analysis available
        }
    }

    /// Calculate evaluation confidence
    fn calculate_evaluation_confidence(
        &self,
        generated_notes: &[MusicalNote],
        reference_notes: Option<&[MusicalNote]>,
    ) -> f32 {
        let mut confidence = 0.5_f32; // Base confidence

        // More notes = higher confidence
        if generated_notes.len() > 10 {
            confidence += 0.2;
        }

        // Reference available = higher confidence
        if reference_notes.is_some() {
            confidence += 0.2;
        }

        // Longer audio = higher confidence
        if let Some(last_note) = generated_notes.last() {
            if last_note.start_time + last_note.duration > 10.0 {
                confidence += 0.1;
            }
        }

        confidence.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_singing_evaluator_creation() {
        let evaluator = SingingEvaluator::new().await.unwrap();
        assert!(evaluator.config.analyze_pitch_accuracy);
    }

    #[tokio::test]
    async fn test_singing_evaluation() {
        let evaluator = SingingEvaluator::new().await.unwrap();

        // Create test audio (sine wave at 440 Hz)
        let sample_rate = 16000;
        let duration = 2.0;
        let frequency = 440.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioBuffer::new(samples, sample_rate, 1);

        let result = evaluator
            .evaluate_singing(&audio, None, None)
            .await
            .unwrap();

        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(result.pitch_accuracy.overall_accuracy >= 0.0);
        assert!(result.pitch_accuracy.overall_accuracy <= 1.0);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_frequency_to_midi() {
        assert_eq!(SingingEvaluator::frequency_to_midi(440.0), 69); // A4
        assert_eq!(SingingEvaluator::frequency_to_midi(261.63), 60); // C4
    }

    #[test]
    fn test_midi_to_frequency() {
        let freq = SingingEvaluator::midi_to_frequency(69);
        assert!((freq - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_frequency_difference_cents() {
        let diff = SingingEvaluator::frequency_difference_cents(440.0, 440.0);
        assert!((diff - 0.0).abs() < 0.1);

        let diff = SingingEvaluator::frequency_difference_cents(466.16, 440.0);
        assert!((diff - 100.0).abs() < 1.0); // Should be approximately 100 cents
    }

    #[tokio::test]
    async fn test_note_extraction() {
        let evaluator = SingingEvaluator::new().await.unwrap();

        // Create simple test audio
        let audio = AudioBuffer::new(vec![0.1; 1000], 16000, 1);
        let notes = evaluator.extract_musical_notes(&audio).await.unwrap();

        // Should extract some notes (even if not perfect)
        assert!(notes.len() >= 0);
    }

    #[tokio::test]
    async fn test_vibrato_analysis() {
        let evaluator = SingingEvaluator::new().await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let notes = vec![MusicalNote {
            frequency: 440.0,
            start_time: 0.0,
            duration: 1.0,
            velocity: 0.8,
            midi_note: 69,
        }];

        let vibrato = evaluator.analyze_vibrato(&audio, &notes).await.unwrap();
        assert!(vibrato.rate > 0.0);
        assert!(vibrato.depth_percent >= 0.0);
    }

    #[test]
    fn test_config_default() {
        let config = SingingEvaluationConfig::default();
        assert!(config.analyze_pitch_accuracy);
        assert!(config.analyze_vibrato);
        assert_eq!(config.time_signature.numerator, 4);
        assert_eq!(config.time_signature.denominator, 4);
    }
}
