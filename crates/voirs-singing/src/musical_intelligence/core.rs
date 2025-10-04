//! Core musical intelligence system

use super::chord_recognition::ChordRecognizer;
use super::key_detection::KeyDetector;
use super::rhythm_analysis::RhythmAnalyzer;
use super::scale_analysis::ScaleAnalyzer;
use super::types::{ChordResult, KeyResult, MusicalAnalysis, RhythmResult, ScaleResult};
use crate::score::MusicalScore;
use crate::types::NoteEvent;
use crate::{Error, Result};
use std::collections::HashMap;

/// Musical intelligence system for analysis and recognition
#[derive(Debug, Clone)]
pub struct MusicalIntelligence {
    /// Chord recognition subsystem for detecting harmonic structures
    chord_recognizer: ChordRecognizer,
    /// Key detection subsystem using Krumhansl-Schmuckler algorithm
    key_detector: KeyDetector,
    /// Scale analysis subsystem for identifying melodic patterns
    scale_analyzer: ScaleAnalyzer,
    /// Rhythm analysis subsystem for tempo and groove detection
    rhythm_analyzer: RhythmAnalyzer,
}

impl MusicalIntelligence {
    /// Create a new musical intelligence system
    pub fn new() -> Self {
        Self {
            chord_recognizer: ChordRecognizer::new(),
            key_detector: KeyDetector::new(),
            scale_analyzer: ScaleAnalyzer::new(),
            rhythm_analyzer: RhythmAnalyzer::new(),
        }
    }

    /// Analyze a musical score comprehensively
    ///
    /// # Arguments
    ///
    /// * `score` - Musical score containing notes, tempo, and time signature information
    ///
    /// # Returns
    ///
    /// Comprehensive musical analysis including chord progression, key, scales, and rhythm
    ///
    /// # Errors
    ///
    /// Returns error if note extraction fails or any analysis component encounters an error
    pub async fn analyze_score(&self, score: &MusicalScore) -> Result<MusicalAnalysis> {
        // Extract note events from score
        let note_events = self.extract_note_events(score)?;

        // Perform individual analyses
        let chord_analysis = self.chord_recognizer.analyze_chords(&note_events).await?;
        let key_analysis = self.key_detector.detect_key(&note_events).await?;
        let scale_analysis = self.scale_analyzer.analyze_scales(&note_events).await?;
        let rhythm_analysis = self.rhythm_analyzer.analyze_rhythm(score).await?;

        // Calculate overall confidence
        let overall_confidence = self.calculate_overall_confidence(
            &chord_analysis,
            &key_analysis,
            &scale_analysis,
            &rhythm_analysis,
        );

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("analysis_version".to_string(), "1.0".to_string());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());

        Ok(MusicalAnalysis {
            chord_analysis,
            key_analysis,
            scale_analysis,
            rhythm_analysis,
            overall_confidence,
            metadata,
        })
    }

    /// Analyze audio samples for musical content
    ///
    /// # Arguments
    ///
    /// * `audio_samples` - Raw audio samples as f32 values
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
    ///
    /// # Returns
    ///
    /// Musical analysis extracted from audio including detected notes, chords, key, and rhythm
    ///
    /// # Errors
    ///
    /// Returns error if audio-to-note conversion fails or analysis components encounter errors
    pub async fn analyze_audio(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<MusicalAnalysis> {
        // Convert audio to note events (simplified)
        let note_events = self.audio_to_note_events(audio_samples, sample_rate)?;

        // Perform analyses on extracted notes
        let chord_analysis = self.chord_recognizer.analyze_chords(&note_events).await?;
        let key_analysis = self.key_detector.detect_key(&note_events).await?;
        let scale_analysis = self.scale_analyzer.analyze_scales(&note_events).await?;

        // For audio, use different rhythm analysis
        let rhythm_analysis = self
            .rhythm_analyzer
            .analyze_audio_rhythm(audio_samples, sample_rate)
            .await?;

        let overall_confidence = self.calculate_overall_confidence(
            &chord_analysis,
            &key_analysis,
            &scale_analysis,
            &rhythm_analysis,
        );

        let mut metadata = HashMap::new();
        metadata.insert("analysis_type".to_string(), "audio".to_string());
        metadata.insert("sample_rate".to_string(), sample_rate.to_string());
        metadata.insert(
            "duration".to_string(),
            (audio_samples.len() as f32 / sample_rate as f32).to_string(),
        );

        Ok(MusicalAnalysis {
            chord_analysis,
            key_analysis,
            scale_analysis,
            rhythm_analysis,
            overall_confidence,
            metadata,
        })
    }

    /// Extract note events from musical score
    ///
    /// # Arguments
    ///
    /// * `score` - Musical score to extract note events from
    ///
    /// # Returns
    ///
    /// Vector of note events with timing, pitch, and expression information
    ///
    /// # Errors
    ///
    /// Returns error if MIDI note conversion fails
    fn extract_note_events(&self, score: &MusicalScore) -> Result<Vec<NoteEvent>> {
        let mut note_events = Vec::new();
        let mut current_time = 0.0;

        for note in &score.notes {
            let note_event = NoteEvent {
                note: self.midi_to_note_name(note.event.frequency as u8)?,
                octave: self.frequency_to_octave(note.event.frequency),
                frequency: note.event.frequency,
                duration: note.duration,
                velocity: note.event.velocity,
                vibrato: note.event.vibrato,
                lyric: note.event.lyric.clone(),
                phonemes: note.event.phonemes.clone(),
                expression: note.event.expression,
                timing_offset: current_time,
                breath_before: note.event.breath_before,
                legato: note.event.legato,
                articulation: note.event.articulation,
            };

            note_events.push(note_event);
            current_time += note.duration;
        }

        Ok(note_events)
    }

    /// Convert MIDI note number to note name
    ///
    /// # Arguments
    ///
    /// * `midi_note` - MIDI note number (0-127)
    ///
    /// # Returns
    ///
    /// Note name (C, C#, D, etc.)
    ///
    /// # Errors
    ///
    /// Returns error if MIDI note is invalid
    fn midi_to_note_name(&self, midi_note: u8) -> Result<String> {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let note_index = (midi_note % 12) as usize;

        if note_index < note_names.len() {
            Ok(note_names[note_index].to_string())
        } else {
            Err(Error::Processing(format!(
                "Invalid MIDI note: {}",
                midi_note
            )))
        }
    }

    /// Convert frequency to octave number
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Octave number (0-10, where 4 is middle octave with A4=440Hz)
    fn frequency_to_octave(&self, frequency: f32) -> u8 {
        if frequency <= 0.0 {
            return 4; // Default octave
        }

        let a4_freq = 440.0;
        let octave = (frequency / a4_freq).log2() + 4.0;
        octave.round().clamp(0.0, 10.0) as u8
    }

    /// Convert audio samples to note events (simplified pitch detection)
    ///
    /// # Arguments
    ///
    /// * `audio_samples` - Audio samples as f32 values
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Vector of detected note events from audio
    ///
    /// # Errors
    ///
    /// Returns error if frequency-to-note conversion fails
    fn audio_to_note_events(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<NoteEvent>> {
        // This is a simplified implementation
        // In practice, this would involve complex pitch detection and onset detection
        let mut note_events = Vec::new();

        let frame_size = sample_rate as usize / 20; // 50ms frames
        let mut current_time = 0.0;

        for chunk in audio_samples.chunks(frame_size) {
            if chunk.len() < frame_size {
                continue;
            }

            // Estimate fundamental frequency using autocorrelation
            let frequency = self.estimate_fundamental_frequency(chunk, sample_rate as f32);

            if frequency > 80.0 && frequency < 2000.0 {
                // Valid singing range
                let note_event = NoteEvent {
                    note: self.frequency_to_note_name(frequency)?,
                    octave: self.frequency_to_octave(frequency),
                    frequency,
                    duration: frame_size as f32 / sample_rate as f32,
                    velocity: self.estimate_velocity(chunk),
                    vibrato: 0.0, // Would need more analysis
                    lyric: None,
                    phonemes: Vec::new(),
                    expression: crate::types::Expression::Neutral,
                    timing_offset: current_time,
                    breath_before: 0.0,
                    legato: false,
                    articulation: crate::types::Articulation::Normal,
                };

                note_events.push(note_event);
            }

            current_time += frame_size as f32 / sample_rate as f32;
        }

        Ok(note_events)
    }

    /// Estimate fundamental frequency using autocorrelation
    ///
    /// # Arguments
    ///
    /// * `audio_frame` - Audio frame samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Estimated fundamental frequency in Hz (or 0.0 if no pitch detected)
    fn estimate_fundamental_frequency(&self, audio_frame: &[f32], sample_rate: f32) -> f32 {
        let min_period = (sample_rate / 800.0) as usize; // 800 Hz max
        let max_period = (sample_rate / 80.0) as usize; // 80 Hz min

        let mut best_correlation = 0.0;
        let mut best_period = 0;

        for period in min_period..max_period.min(audio_frame.len() / 2) {
            let mut correlation = 0.0;
            for i in 0..audio_frame.len() - period {
                correlation += audio_frame[i] * audio_frame[i + period];
            }

            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        if best_period > 0 && best_correlation > 0.1 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// Convert frequency to note name
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Note name (C, C#, D, etc.)
    ///
    /// # Errors
    ///
    /// Returns error only in exceptional cases (uses fallback)
    fn frequency_to_note_name(&self, frequency: f32) -> Result<String> {
        let a4_freq = 440.0;
        let semitones_from_a4 = 12.0 * (frequency / a4_freq).log2();
        let note_index = ((semitones_from_a4.round() as i32 + 9) % 12) as usize; // +9 to start from C

        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        if note_index < note_names.len() {
            Ok(note_names[note_index].to_string())
        } else {
            Ok("C".to_string()) // Default fallback
        }
    }

    /// Estimate velocity from audio frame using RMS energy
    ///
    /// # Arguments
    ///
    /// * `audio_frame` - Audio frame samples
    ///
    /// # Returns
    ///
    /// Velocity value normalized to 0.0-1.0 range
    fn estimate_velocity(&self, audio_frame: &[f32]) -> f32 {
        let rms: f32 = audio_frame.iter().map(|&x| x * x).sum::<f32>() / audio_frame.len() as f32;

        (rms.sqrt() * 2.0).clamp(0.0, 1.0) // Normalize to 0-1
    }

    /// Calculate overall confidence from individual analyses
    ///
    /// # Arguments
    ///
    /// * `chord_analysis` - Chord recognition results
    /// * `key_analysis` - Key detection result
    /// * `scale_analysis` - Scale analysis results
    /// * `rhythm_analysis` - Rhythm analysis result
    ///
    /// # Returns
    ///
    /// Weighted average confidence score (0.0-1.0) with weights: key 30%, chord 30%, scale 20%, rhythm 20%
    fn calculate_overall_confidence(
        &self,
        chord_analysis: &[ChordResult],
        key_analysis: &KeyResult,
        scale_analysis: &[ScaleResult],
        rhythm_analysis: &RhythmResult,
    ) -> f32 {
        let chord_confidence = if chord_analysis.is_empty() {
            0.5
        } else {
            chord_analysis.iter().map(|c| c.confidence).sum::<f32>() / chord_analysis.len() as f32
        };

        let scale_confidence = if scale_analysis.is_empty() {
            0.5
        } else {
            scale_analysis.iter().map(|s| s.confidence).sum::<f32>() / scale_analysis.len() as f32
        };

        // Weighted average: key 30%, chord 30%, scale 20%, rhythm 20%
        (key_analysis.confidence * 0.3
            + chord_confidence * 0.3
            + scale_confidence * 0.2
            + rhythm_analysis.confidence * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Get musical intelligence capabilities
    ///
    /// # Returns
    ///
    /// List of available analysis capabilities as human-readable strings
    pub fn capabilities(&self) -> Vec<String> {
        vec![
            "Chord Recognition".to_string(),
            "Key Detection".to_string(),
            "Scale Analysis".to_string(),
            "Rhythm Analysis".to_string(),
            "Audio Analysis".to_string(),
            "Score Analysis".to_string(),
        ]
    }
}

impl Default for MusicalIntelligence {
    fn default() -> Self {
        Self::new()
    }
}
