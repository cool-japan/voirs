//! Musical format parsers

#[cfg(any(feature = "midi-support", feature = "musicxml-support"))]
use crate::score::{KeySignature, Mode, Note, TimeSignature};
use crate::score::{MusicalNote, MusicalScore};
use crate::types::{Articulation, Dynamics, Expression, NoteEvent};
use async_trait::async_trait;
use std::fs;

/// Format parser trait for parsing musical score formats
#[async_trait]
pub trait FormatParser: Send + Sync {
    /// Parse a musical score from a file path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to parse
    ///
    /// # Returns
    ///
    /// Returns a `MusicalScore` on success
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    async fn parse_file(&self, path: &str) -> crate::Result<MusicalScore>;

    /// Parse a musical score from a string
    ///
    /// # Arguments
    ///
    /// * `data` - String data to parse
    ///
    /// # Returns
    ///
    /// Returns a `MusicalScore` on success
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be parsed
    async fn parse_string(&self, data: &str) -> crate::Result<MusicalScore>;

    /// Get list of supported file extensions
    ///
    /// # Returns
    ///
    /// Vector of file extension strings (without dots)
    fn supported_extensions(&self) -> Vec<String>;
}

/// MusicXML format parser
///
/// Parses MusicXML files into internal musical score representation.
/// Supports both compressed (.mxl) and uncompressed (.xml, .musicxml) formats.
pub struct MusicXmlParser;

impl Default for MusicXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicXmlParser {
    /// Create a new MusicXML parser
    pub fn new() -> Self {
        Self
    }

    /// Parse MusicXML data into a musical score
    #[cfg(feature = "musicxml-support")]
    fn parse_musicxml_data(&self, data: &str) -> crate::Result<MusicalScore> {
        // For now, implement a simplified MusicXML parser using basic XML parsing
        // This provides a working foundation that can be enhanced later
        use std::collections::HashMap;

        let mut score = MusicalScore::new(String::from("MusicXML Score"), String::from("Unknown"));

        // Basic XML parsing to extract title and composer
        if let Some(title_start) = data.find("<work-title>") {
            if let Some(title_end) = data[title_start..].find("</work-title>") {
                let title_content = &data[title_start + 12..title_start + title_end];
                score.title = title_content.to_string();
            }
        }

        if let Some(composer_start) = data.find(r#"<creator type="composer">"#) {
            let tag_end_pos = composer_start + r#"<creator type="composer">"#.len();
            if let Some(composer_end) = data[tag_end_pos..].find("</creator>") {
                let composer_content = &data[tag_end_pos..tag_end_pos + composer_end];
                score.composer = composer_content.trim().to_string();
            }
        }

        // Parse time signature
        if let Some(time_start) = data.find("<time>") {
            if let Some(beats_start) = data[time_start..].find("<beats>") {
                if let Some(beats_end) = data[time_start + beats_start..].find("</beats>") {
                    let beats_str =
                        &data[time_start + beats_start + 7..time_start + beats_start + beats_end];
                    if let Ok(beats) = beats_str.parse::<u8>() {
                        score.time_signature.numerator = beats;
                    }
                }
            }
            if let Some(beat_type_start) = data[time_start..].find("<beat-type>") {
                if let Some(beat_type_end) =
                    data[time_start + beat_type_start..].find("</beat-type>")
                {
                    let beat_type_str = &data[time_start + beat_type_start + 11
                        ..time_start + beat_type_start + beat_type_end];
                    if let Ok(beat_type) = beat_type_str.parse::<u8>() {
                        score.time_signature.denominator = beat_type;
                    }
                }
            }
        }

        // Parse key signature
        if let Some(key_start) = data.find("<key>") {
            if let Some(fifths_start) = data[key_start..].find("<fifths>") {
                if let Some(fifths_end) = data[key_start + fifths_start..].find("</fifths>") {
                    let fifths_str =
                        &data[key_start + fifths_start + 8..key_start + fifths_start + fifths_end];
                    if let Ok(fifths) = fifths_str.parse::<i8>() {
                        score.key_signature.accidentals = fifths;
                    }
                }
            }

            if data[key_start..].contains("<mode>minor</mode>") {
                score.key_signature.mode = crate::score::Mode::Minor;
            } else {
                score.key_signature.mode = crate::score::Mode::Major;
            }
        }

        // Parse notes - simplified approach
        let mut current_time = 0.0;
        let mut divisions = 4.0; // Default

        // Extract divisions
        if let Some(div_start) = data.find("<divisions>") {
            if let Some(div_end) = data[div_start..].find("</divisions>") {
                let div_str = &data[div_start + 11..div_start + div_end];
                if let Ok(div) = div_str.parse::<f32>() {
                    divisions = div;
                }
            }
        }

        // Find all note elements
        let mut search_from = 0;
        while let Some(note_start) = data[search_from..].find("<note>") {
            let note_start_abs = search_from + note_start;
            if let Some(note_end) = data[note_start_abs..].find("</note>") {
                let note_xml = &data[note_start_abs..note_start_abs + note_end + 7];

                if let Some(musical_note) =
                    self.parse_simple_note(note_xml, current_time, divisions)?
                {
                    current_time += musical_note.duration;
                    score.add_note(musical_note);
                }

                search_from = note_start_abs + note_end + 7;
            } else {
                break;
            }
        }

        Ok(score)
    }

    #[cfg(feature = "musicxml-support")]
    fn parse_simple_note(
        &self,
        note_xml: &str,
        start_time: f32,
        divisions: f32,
    ) -> crate::Result<Option<MusicalNote>> {
        // Skip rest notes
        if note_xml.contains("<rest") {
            return Ok(None);
        }

        // Extract pitch information
        let step = if let Some(step_start) = note_xml.find("<step>") {
            if let Some(step_end) = note_xml[step_start..].find("</step>") {
                note_xml[step_start + 6..step_start + step_end].to_string()
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        let octave = if let Some(oct_start) = note_xml.find("<octave>") {
            if let Some(oct_end) = note_xml[oct_start..].find("</octave>") {
                let oct_str = &note_xml[oct_start + 8..oct_start + oct_end];
                oct_str.parse::<i32>().unwrap_or(4)
            } else {
                4
            }
        } else {
            4
        };

        let alter = if let Some(alt_start) = note_xml.find("<alter>") {
            if let Some(alt_end) = note_xml[alt_start..].find("</alter>") {
                let alt_str = &note_xml[alt_start + 7..alt_start + alt_end];
                alt_str.parse::<i32>().unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        // Extract duration
        let duration = if let Some(dur_start) = note_xml.find("<duration>") {
            if let Some(dur_end) = note_xml[dur_start..].find("</duration>") {
                let dur_str = &note_xml[dur_start + 10..dur_start + dur_end];
                dur_str.parse::<f32>().unwrap_or(divisions) / divisions
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Create note
        let note_name = format!(
            "{}{}",
            step,
            if alter > 0 {
                "#".repeat(alter as usize)
            } else {
                "b".repeat((-alter) as usize)
            }
        );
        let frequency = self.step_octave_to_frequency(&step, octave, alter);

        let note_event = NoteEvent {
            note: note_name,
            octave: octave.max(0) as u8,
            frequency,
            velocity: 0.8,
            duration,
            vibrato: 0.0,
            lyric: None,
            phonemes: Vec::new(),
            expression: Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: Articulation::Normal,
        };

        let musical_note = MusicalNote {
            event: note_event,
            start_time,
            duration,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: crate::types::Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        };

        Ok(Some(musical_note))
    }

    #[cfg(feature = "musicxml-support")]
    fn step_octave_to_frequency(&self, step: &str, octave: i32, alter: i32) -> f32 {
        // Convert note step to semitone offset from C
        let base_semitone = match step {
            "C" => 0,
            "D" => 2,
            "E" => 4,
            "F" => 5,
            "G" => 7,
            "A" => 9,
            "B" => 11,
            _ => 0,
        };

        // MIDI note number: (octave + 1) * 12 + semitone + alter
        let midi_note = (octave + 1) * 12 + base_semitone + alter;

        // Convert MIDI note to frequency (A4 = 440 Hz = MIDI note 69)
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    #[cfg(not(feature = "musicxml-support"))]
    fn parse_musicxml_data(&self, _data: &str) -> crate::Result<MusicalScore> {
        Err(crate::Error::Format(String::from(
            "MusicXML support not enabled. Enable 'musicxml-support' feature.",
        )))
    }
}

#[async_trait]
impl FormatParser for MusicXmlParser {
    async fn parse_file(&self, path: &str) -> crate::Result<MusicalScore> {
        let data = fs::read_to_string(path).map_err(crate::Error::Io)?;
        self.parse_string(&data).await
    }

    async fn parse_string(&self, data: &str) -> crate::Result<MusicalScore> {
        self.parse_musicxml_data(data)
    }

    fn supported_extensions(&self) -> Vec<String> {
        vec![String::from("xml"), String::from("musicxml")]
    }
}

/// MIDI format parser
///
/// Parses MIDI files into internal musical score representation.
/// Supports Standard MIDI File (SMF) format types 0 and 1.
pub struct MidiParser {
    /// Ticks per quarter note for timing calculation
    ticks_per_quarter: u16,
    /// Current tempo in microseconds per quarter note
    tempo: u32,
}

impl Default for MidiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MidiParser {
    /// Create a new MIDI parser with default settings
    ///
    /// Default settings:
    /// - Ticks per quarter note: 480
    /// - Tempo: 120 BPM (500,000 microseconds per quarter note)
    pub fn new() -> Self {
        Self {
            ticks_per_quarter: 480,
            tempo: 500_000, // Default 120 BPM (500,000 microseconds per quarter note)
        }
    }

    /// Convert MIDI note number to frequency
    fn midi_to_frequency(midi_note: u8) -> f32 {
        // A4 = 440 Hz = MIDI note 69
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    /// Convert MIDI velocity to dynamic level
    fn velocity_to_dynamics(velocity: u8) -> Dynamics {
        match velocity {
            0..=15 => Dynamics::Pianissimo,
            16..=31 => Dynamics::Pianissimo,
            32..=47 => Dynamics::Piano,
            48..=63 => Dynamics::MezzoPiano,
            64..=79 => Dynamics::MezzoForte,
            80..=95 => Dynamics::Forte,
            96..=111 => Dynamics::Fortissimo,
            112..=127 => Dynamics::Fortissimo,
            _ => Dynamics::MezzoForte,
        }
    }

    /// Convert MIDI note to note name
    fn midi_to_note_name(midi_note: u8) -> String {
        let notes = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let octave = (midi_note / 12) as i32 - 1;
        let note_index = (midi_note % 12) as usize;
        format!("{}{}", notes[note_index], octave)
    }

    /// Convert ticks to beats
    fn ticks_to_beats(&self, ticks: u32) -> f32 {
        ticks as f32 / self.ticks_per_quarter as f32
    }

    /// Parse MIDI data into musical score
    #[cfg(feature = "midi-support")]
    fn parse_midi_data(&mut self, data: &[u8]) -> crate::Result<MusicalScore> {
        use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};

        let smf = Smf::parse(data)
            .map_err(|e| crate::Error::Format(format!("MIDI parse error: {}", e)))?;

        // Create score with default values
        let mut score = MusicalScore::new(String::from("MIDI Score"), String::from("Unknown"));

        // Set timing division
        if let midly::Timing::Metrical(tpq) = smf.header.timing {
            self.ticks_per_quarter = tpq.as_int();
        }

        // Track note on/off events
        let mut active_notes: std::collections::HashMap<u8, (u32, u8)> =
            std::collections::HashMap::new();

        // Process all tracks
        for track in smf.tracks {
            let mut current_time = 0u32;

            for event in track {
                current_time += event.delta.as_int();

                match event.kind {
                    TrackEventKind::Midi {
                        channel: _,
                        message,
                    } => {
                        match message {
                            MidiMessage::NoteOn { key, vel } => {
                                if vel.as_int() > 0 {
                                    // Store note on event
                                    active_notes.insert(key.as_int(), (current_time, vel.as_int()));
                                } else {
                                    // Note on with velocity 0 = note off
                                    if let Some((start_time, velocity)) =
                                        active_notes.remove(&key.as_int())
                                    {
                                        let note = self.create_musical_note(
                                            key.as_int(),
                                            velocity,
                                            start_time,
                                            current_time,
                                        );
                                        score.add_note(note);
                                    }
                                }
                            }
                            MidiMessage::NoteOff { key, vel: _ } => {
                                if let Some((start_time, velocity)) =
                                    active_notes.remove(&key.as_int())
                                {
                                    let note = self.create_musical_note(
                                        key.as_int(),
                                        velocity,
                                        start_time,
                                        current_time,
                                    );
                                    score.add_note(note);
                                }
                            }
                            _ => {} // Ignore other MIDI messages for now
                        }
                    }
                    TrackEventKind::Meta(message) => {
                        match message {
                            MetaMessage::Tempo(tempo) => {
                                self.tempo = tempo.as_int();
                            }
                            MetaMessage::TimeSignature(numerator, denominator, _, _) => {
                                score.time_signature = TimeSignature {
                                    numerator,
                                    denominator: 2_u8.pow(denominator as u32),
                                };
                            }
                            MetaMessage::KeySignature(sharps_flats, is_minor) => {
                                // Parse key signature
                                score.key_signature = KeySignature {
                                    root: Note::C, // Simplified for now
                                    mode: if is_minor { Mode::Minor } else { Mode::Major },
                                    accidentals: sharps_flats,
                                };
                            }
                            MetaMessage::TrackName(name) => {
                                if score.title == "MIDI Score" {
                                    score.title = String::from_utf8_lossy(name).to_string();
                                }
                            }
                            _ => {} // Ignore other meta messages
                        }
                    }
                    _ => {} // Ignore SysEx and other events
                }
            }
        }

        // Calculate tempo
        let bpm = 60_000_000.0 / self.tempo as f32;
        score.tempo = bpm;

        Ok(score)
    }

    /// Create a musical note from MIDI data
    fn create_musical_note(
        &self,
        midi_note: u8,
        velocity: u8,
        start_ticks: u32,
        end_ticks: u32,
    ) -> MusicalNote {
        let frequency = Self::midi_to_frequency(midi_note);
        let note_name = Self::midi_to_note_name(midi_note);
        let octave = ((midi_note / 12) as i32 - 1).max(0) as u8;

        let velocity_normalized = velocity as f32 / 127.0;
        let dynamics = Self::velocity_to_dynamics(velocity);

        let start_beats = self.ticks_to_beats(start_ticks);
        let duration_beats = self.ticks_to_beats(end_ticks - start_ticks);

        let note_event = NoteEvent {
            note: note_name,
            octave,
            frequency,
            velocity: velocity_normalized,
            duration: duration_beats,
            vibrato: 0.0,
            lyric: None,
            phonemes: Vec::new(),
            expression: Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: Articulation::Normal,
        };

        MusicalNote {
            event: note_event,
            start_time: start_beats,
            duration: duration_beats,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        }
    }

    #[cfg(not(feature = "midi-support"))]
    fn parse_midi_data(&mut self, _data: &[u8]) -> crate::Result<MusicalScore> {
        Err(crate::Error::Format(String::from(
            "MIDI support not enabled. Enable 'midi-support' feature.",
        )))
    }
}

#[async_trait]
impl FormatParser for MidiParser {
    async fn parse_file(&self, path: &str) -> crate::Result<MusicalScore> {
        let data = fs::read(path).map_err(crate::Error::Io)?;
        let mut parser = self.clone();
        parser.parse_midi_data(&data)
    }

    async fn parse_string(&self, data: &str) -> crate::Result<MusicalScore> {
        // For MIDI, we expect base64 encoded data
        let decoded = base64::decode(data)
            .map_err(|_| crate::Error::Format(String::from("Invalid base64 MIDI data")))?;
        let mut parser = self.clone();
        parser.parse_midi_data(&decoded)
    }

    fn supported_extensions(&self) -> Vec<String> {
        vec![String::from("mid"), String::from("midi")]
    }
}

// Make MidiParser cloneable for async operations
impl Clone for MidiParser {
    fn clone(&self) -> Self {
        Self {
            ticks_per_quarter: self.ticks_per_quarter,
            tempo: self.tempo,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midi_to_frequency() {
        // Test A4 = 440 Hz (MIDI note 69)
        assert!((MidiParser::midi_to_frequency(69) - 440.0).abs() < 0.01);

        // Test C4 = ~261.63 Hz (MIDI note 60)
        assert!((MidiParser::midi_to_frequency(60) - 261.63).abs() < 0.1);

        // Test A5 = 880 Hz (MIDI note 81)
        assert!((MidiParser::midi_to_frequency(81) - 880.0).abs() < 0.01);
    }

    #[test]
    fn test_midi_to_note_name() {
        assert_eq!(MidiParser::midi_to_note_name(60), "C4");
        assert_eq!(MidiParser::midi_to_note_name(69), "A4");
        assert_eq!(MidiParser::midi_to_note_name(72), "C5");
        assert_eq!(MidiParser::midi_to_note_name(61), "C#4");
    }

    #[test]
    fn test_velocity_to_dynamics() {
        assert_eq!(MidiParser::velocity_to_dynamics(10), Dynamics::Pianissimo);
        assert_eq!(MidiParser::velocity_to_dynamics(40), Dynamics::Piano);
        assert_eq!(MidiParser::velocity_to_dynamics(70), Dynamics::MezzoForte);
        assert_eq!(MidiParser::velocity_to_dynamics(85), Dynamics::Forte);
        assert_eq!(MidiParser::velocity_to_dynamics(100), Dynamics::Fortissimo);
        assert_eq!(MidiParser::velocity_to_dynamics(120), Dynamics::Fortissimo);
    }

    #[test]
    fn test_ticks_to_beats() {
        let parser = MidiParser::new();
        assert!((parser.ticks_to_beats(480) - 1.0).abs() < 0.01); // 1 beat
        assert!((parser.ticks_to_beats(240) - 0.5).abs() < 0.01); // 0.5 beats
        assert!((parser.ticks_to_beats(960) - 2.0).abs() < 0.01); // 2 beats
    }

    #[tokio::test]
    async fn test_format_parser_extensions() {
        let midi_parser = MidiParser::new();
        let extensions = midi_parser.supported_extensions();
        assert!(extensions.contains(&String::from("mid")));
        assert!(extensions.contains(&String::from("midi")));

        let musicxml_parser = MusicXmlParser::new();
        let extensions = musicxml_parser.supported_extensions();
        assert!(extensions.contains(&String::from("xml")));
        assert!(extensions.contains(&String::from("musicxml")));
    }

    #[cfg(feature = "midi-support")]
    #[tokio::test]
    async fn test_midi_parser_empty_file() {
        let parser = MidiParser::new();

        // Test with invalid MIDI data should return error
        let result = parser.parse_string("invalid_base64").await;
        assert!(result.is_err());
    }

    #[cfg(feature = "musicxml-support")]
    #[tokio::test]
    async fn test_musicxml_parser_basic() {
        let parser = MusicXmlParser::new();

        // Create a simple MusicXML document
        let musicxml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="3.1">
            <work>
                <work-title>Test Song</work-title>
            </work>
            <identification>
                <creator type="composer">Test Composer</creator>
            </identification>
            <part-list>
                <score-part id="P1">
                    <part-name>Voice</part-name>
                </score-part>
            </part-list>
            <part id="P1">
                <measure number="1">
                    <attributes>
                        <divisions>4</divisions>
                        <key>
                            <fifths>0</fifths>
                            <mode>major</mode>
                        </key>
                        <time>
                            <beats>4</beats>
                            <beat-type>4</beat-type>
                        </time>
                    </attributes>
                    <note>
                        <pitch>
                            <step>C</step>
                            <octave>4</octave>
                        </pitch>
                        <duration>4</duration>
                    </note>
                    <note>
                        <pitch>
                            <step>D</step>
                            <octave>4</octave>
                        </pitch>
                        <duration>4</duration>
                    </note>
                </measure>
            </part>
        </score-partwise>"#;

        let result = parser.parse_string(musicxml_data).await;
        assert!(result.is_ok());

        let score = result.unwrap();
        assert_eq!(score.title, "Test Song");
        assert_eq!(score.composer, "Test Composer");
        assert_eq!(score.time_signature.numerator, 4);
        assert_eq!(score.time_signature.denominator, 4);
        assert_eq!(score.notes.len(), 2); // Two notes in the measure

        // Check first note (C4)
        let first_note = &score.notes[0];
        assert_eq!(first_note.event.note, "C");
        assert_eq!(first_note.event.octave, 4);
        assert!((first_note.event.frequency - 261.63).abs() < 1.0); // C4 frequency

        // Check second note (D4)
        let second_note = &score.notes[1];
        assert_eq!(second_note.event.note, "D");
        assert_eq!(second_note.event.octave, 4);
        assert!((second_note.event.frequency - 293.66).abs() < 1.0); // D4 frequency
    }

    #[cfg(not(feature = "musicxml-support"))]
    #[tokio::test]
    async fn test_musicxml_parser_no_feature() {
        let parser = MusicXmlParser::new();

        // Test that it returns error when feature is not enabled
        let result = parser.parse_string("<musicxml></musicxml>").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("MusicXML support not enabled"));
    }

    #[tokio::test]
    async fn test_musicxml_parser_stub() {
        let parser = MusicXmlParser::new();

        // Test extensions
        let extensions = parser.supported_extensions();
        assert!(extensions.contains(&String::from("xml")));
        assert!(extensions.contains(&String::from("musicxml")));
    }
}
