//! Musical score rendering for visual notation display
//!
//! This module provides functionality to render musical scores to various visual formats
//! including SVG, ASCII art, and other notation representations.

use crate::score::{KeySignature, Mode, MusicalNote, MusicalScore, TimeSignature};

/// Score rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Width of the output
    pub width: u32,
    /// Height of the output  
    pub height: u32,
    /// Staff line spacing
    pub staff_spacing: f32,
    /// Note size
    pub note_size: f32,
    /// Show clef
    pub show_clef: bool,
    /// Show time signature
    pub show_time_signature: bool,
    /// Show key signature
    pub show_key_signature: bool,
    /// Measures per line
    pub measures_per_line: u32,
    /// Font family for text
    pub font_family: String,
    /// Font size
    pub font_size: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            staff_spacing: 10.0,
            note_size: 8.0,
            show_clef: true,
            show_time_signature: true,
            show_key_signature: true,
            measures_per_line: 4,
            font_family: "Arial".to_string(),
            font_size: 12.0,
        }
    }
}

/// Rendering format
#[derive(Debug, Clone, Copy)]
pub enum RenderFormat {
    /// SVG vector format
    Svg,
    /// ASCII art text format
    Ascii,
    /// Simple text representation
    Text,
}

/// Position on the staff
#[derive(Debug, Clone, Copy)]
pub struct StaffPosition {
    /// X coordinate
    pub x: f32,
    /// Y coordinate (staff line position)
    pub y: f32,
}

/// Musical score renderer
#[derive(Debug)]
pub struct ScoreRenderer {
    config: RenderConfig,
}

impl ScoreRenderer {
    /// Create a new score renderer
    pub fn new(config: RenderConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self {
            config: RenderConfig::default(),
        }
    }

    /// Render a musical score to the specified format
    pub fn render(&self, score: &MusicalScore, format: RenderFormat) -> crate::Result<String> {
        match format {
            RenderFormat::Svg => self.render_svg(score),
            RenderFormat::Ascii => self.render_ascii(score),
            RenderFormat::Text => self.render_text(score),
        }
    }

    /// Render score to SVG format
    fn render_svg(&self, score: &MusicalScore) -> crate::Result<String> {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));
        svg.push('\n');

        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="30" font-family="{}" font-size="{}" font-weight="bold" text-anchor="middle">{}</text>"#,
            self.config.width / 2, self.config.font_family, self.config.font_size + 4.0, score.title
        ));
        svg.push('\n');

        // Composer
        svg.push_str(&format!(
            r#"<text x="{}" y="50" font-family="{}" font-size="{}" text-anchor="middle">{}</text>"#,
            self.config.width / 2,
            self.config.font_family,
            self.config.font_size,
            score.composer
        ));
        svg.push('\n');

        // Staff lines
        let staff_y = 80.0;
        self.render_staff_lines(&mut svg, staff_y);

        // Clef
        let mut current_x = 40.0;
        if self.config.show_clef {
            self.render_treble_clef(&mut svg, current_x, staff_y);
            current_x += 30.0;
        }

        // Key signature
        if self.config.show_key_signature {
            current_x =
                self.render_key_signature(&mut svg, &score.key_signature, current_x, staff_y);
            current_x += 20.0;
        }

        // Time signature
        if self.config.show_time_signature {
            current_x =
                self.render_time_signature(&mut svg, &score.time_signature, current_x, staff_y);
            current_x += 30.0;
        }

        // Notes
        self.render_notes(&mut svg, &score.notes, current_x, staff_y)?;

        // SVG footer
        svg.push_str("</svg>");

        Ok(svg)
    }

    /// Render staff lines
    fn render_staff_lines(&self, svg: &mut String, y: f32) {
        for i in 0..5 {
            let line_y = y + (i as f32 * self.config.staff_spacing);
            svg.push_str(&format!(
                r#"<line x1="20" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
                line_y,
                self.config.width - 20,
                line_y
            ));
            svg.push('\n');
        }
    }

    /// Render treble clef
    fn render_treble_clef(&self, svg: &mut String, x: f32, staff_y: f32) {
        // Simplified treble clef representation
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="serif" font-size="24" text-anchor="middle">𝄞</text>"#,
            x, staff_y + self.config.staff_spacing * 2.0
        ));
        svg.push('\n');
    }

    /// Render key signature
    fn render_key_signature(
        &self,
        svg: &mut String,
        key_sig: &KeySignature,
        x: f32,
        staff_y: f32,
    ) -> f32 {
        let mut current_x = x;

        // Render sharps or flats based on accidentals
        for i in 0..key_sig.accidentals.abs() {
            let symbol = if key_sig.accidentals > 0 {
                "♯"
            } else {
                "♭"
            };
            let positions = if key_sig.accidentals > 0 {
                // Sharp positions (F, C, G, D, A, E, B)
                vec![3.0, 1.0, 3.5, 1.5, 3.0, 1.0, 3.5]
            } else {
                // Flat positions (B, E, A, D, G, C, F)
                vec![1.5, 2.5, 0.5, 2.0, 1.0, 2.5, 0.5]
            };

            if (i as usize) < positions.len() {
                let symbol_y = staff_y + positions[i as usize] * self.config.staff_spacing;
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="serif" font-size="16" text-anchor="middle">{}</text>"#,
                    current_x, symbol_y, symbol
                ));
                svg.push('\n');
                current_x += 12.0;
            }
        }

        current_x
    }

    /// Render time signature
    fn render_time_signature(
        &self,
        svg: &mut String,
        time_sig: &TimeSignature,
        x: f32,
        staff_y: f32,
    ) -> f32 {
        // Numerator
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="serif" font-size="18" font-weight="bold" text-anchor="middle">{}</text>"#,
            x, staff_y + self.config.staff_spacing, time_sig.numerator
        ));
        svg.push('\n');

        // Denominator
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="serif" font-size="18" font-weight="bold" text-anchor="middle">{}</text>"#,
            x, staff_y + self.config.staff_spacing * 3.0, time_sig.denominator
        ));
        svg.push('\n');

        x + 20.0
    }

    /// Render notes
    fn render_notes(
        &self,
        svg: &mut String,
        notes: &[MusicalNote],
        start_x: f32,
        staff_y: f32,
    ) -> crate::Result<()> {
        let mut current_x = start_x;
        let note_spacing = 40.0;

        for note in notes {
            let position = self.calculate_note_position(&note.event.note, note.event.octave);
            let note_y = staff_y + position * (self.config.staff_spacing / 2.0);

            // Note head
            svg.push_str(&format!(
                r#"<ellipse cx="{}" cy="{}" rx="{}" ry="{}" fill="black"/>"#,
                current_x,
                note_y,
                self.config.note_size / 2.0,
                self.config.note_size / 3.0
            ));
            svg.push('\n');

            // Stem
            let stem_height = self.config.staff_spacing * 3.0;
            let stem_up = position > 4.0; // Stem direction based on position
            let stem_y1 = note_y;
            let stem_y2 = if stem_up {
                note_y - stem_height
            } else {
                note_y + stem_height
            };
            let stem_x = if stem_up {
                current_x + self.config.note_size / 2.0
            } else {
                current_x - self.config.note_size / 2.0
            };

            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1.5"/>"#,
                stem_x, stem_y1, stem_x, stem_y2
            ));
            svg.push('\n');

            // Ledger lines for notes outside staff
            if position < 0.0 || position > 8.0 {
                let ledger_lines = self.calculate_ledger_lines(position);
                for ledger_pos in ledger_lines {
                    let ledger_y = staff_y + ledger_pos * (self.config.staff_spacing / 2.0);
                    svg.push_str(&format!(
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
                        current_x - 8.0, ledger_y, current_x + 8.0, ledger_y
                    ));
                    svg.push('\n');
                }
            }

            // Accidentals
            if note.event.note.contains('#') || note.event.note.contains('b') {
                let accidental = if note.event.note.contains('#') {
                    "♯"
                } else {
                    "♭"
                };
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="serif" font-size="14" text-anchor="middle">{}</text>"#,
                    current_x - 15.0, note_y + 4.0, accidental
                ));
                svg.push('\n');
            }

            current_x += note_spacing;
        }

        Ok(())
    }

    /// Calculate note position on staff (0 = bottom line E4, 8 = top line F5)
    fn calculate_note_position(&self, note: &str, octave: u8) -> f32 {
        let base_note = note.chars().next().unwrap_or('C');

        // Calculate semitone offset from C
        let semitone_from_c = match base_note {
            'C' => 0,
            'D' => 2,
            'E' => 4,
            'F' => 5,
            'G' => 7,
            'A' => 9,
            'B' => 11,
            _ => 0,
        };

        // Calculate MIDI note number
        let midi_note = (octave as i32 + 1) * 12 + semitone_from_c;

        // E4 (MIDI 64) is the bottom staff line (position 0)
        let e4_midi = 64;
        let position_offset = (midi_note - e4_midi) as f32;

        // Each semitone is 0.5 staff positions (since there are 2 semitones per staff position on average)
        position_offset * 0.5
    }

    /// Calculate ledger lines needed for notes outside the staff
    fn calculate_ledger_lines(&self, position: f32) -> Vec<f32> {
        let mut ledger_lines = Vec::new();

        if position < 0.0 {
            let mut line_pos = -2.0;
            while line_pos >= position {
                ledger_lines.push(line_pos);
                line_pos -= 2.0;
            }
        } else if position > 8.0 {
            let mut line_pos = 10.0;
            while line_pos <= position {
                ledger_lines.push(line_pos);
                line_pos += 2.0;
            }
        }

        ledger_lines
    }

    /// Render score to ASCII art format
    fn render_ascii(&self, score: &MusicalScore) -> crate::Result<String> {
        let mut output = String::new();

        // Title and composer
        output.push_str(&format!("{title}\n", title = score.title));
        output.push_str(&format!("by {composer}\n\n", composer = score.composer));

        // Key and time signature
        output.push_str(&format!(
            "Key: {} {}\n",
            self.key_signature_to_string(&score.key_signature),
            if score.key_signature.mode == Mode::Major {
                "Major"
            } else {
                "Minor"
            }
        ));
        output.push_str(&format!(
            "Time: {}/{}\n",
            score.time_signature.numerator, score.time_signature.denominator
        ));
        output.push_str(&format!("Tempo: {tempo} BPM\n\n", tempo = score.tempo));

        // ASCII staff
        output.push_str("     ┌─────────────────────────────────────────────────────────┐\n");
        output.push_str("  F  │─────────────────────────────────────────────────────────│\n");
        output.push_str("  D  │─────────────────────────────────────────────────────────│\n");
        output.push_str("  B  │─────────────────────────────────────────────────────────│\n");
        output.push_str("  G  │─────────────────────────────────────────────────────────│\n");
        output.push_str("  E  │─────────────────────────────────────────────────────────│\n");
        output.push_str("     └─────────────────────────────────────────────────────────┘\n\n");

        // Notes as text
        output.push_str("Notes:\n");
        for (i, note) in score.notes.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} ({}Hz) - Duration: {:.2}s\n",
                i + 1,
                self.format_note_name(&note.event.note, note.event.octave),
                note.event.frequency,
                note.duration
            ));
        }

        Ok(output)
    }

    /// Render score to simple text format
    fn render_text(&self, score: &MusicalScore) -> crate::Result<String> {
        let mut output = String::new();

        // Score information
        output.push_str(&format!("Title: {}\n", score.title));
        output.push_str(&format!("Composer: {}\n", score.composer));
        output.push_str(&format!(
            "Key Signature: {} {}\n",
            self.key_signature_to_string(&score.key_signature),
            if score.key_signature.mode == Mode::Major {
                "Major"
            } else {
                "Minor"
            }
        ));
        output.push_str(&format!(
            "Time Signature: {}/{}\n",
            score.time_signature.numerator, score.time_signature.denominator
        ));
        output.push_str(&format!("Tempo: {tempo} BPM\n", tempo = score.tempo));
        output.push_str(&format!(
            "Duration: {:.2} seconds\n",
            score.duration.as_secs_f32()
        ));
        output.push_str(&format!("Number of notes: {}\n\n", score.notes.len()));

        // Notes
        output.push_str("Musical Notes:\n");
        output.push_str(&"─".repeat(60));
        output.push('\n');

        for (i, note) in score.notes.iter().enumerate() {
            output.push_str(&format!(
                "{:3}. {:>4} | {:>7.2}Hz | {:>6.2}s | {:>4.2}s | {:>10} | {:>12}\n",
                i + 1,
                self.format_note_name(&note.event.note, note.event.octave),
                note.event.frequency,
                note.start_time,
                note.duration,
                format!("{:?}", note.dynamics),
                format!("{:?}", note.articulation)
            ));
        }

        Ok(output)
    }

    /// Convert key signature to string representation
    fn key_signature_to_string(&self, key_sig: &KeySignature) -> String {
        match key_sig.accidentals {
            0 => "C".to_string(),
            1 => "G".to_string(),
            2 => "D".to_string(),
            3 => "A".to_string(),
            4 => "E".to_string(),
            5 => "B".to_string(),
            6 => "F#".to_string(),
            7 => "C#".to_string(),
            -1 => "F".to_string(),
            -2 => "Bb".to_string(),
            -3 => "Eb".to_string(),
            -4 => "Ab".to_string(),
            -5 => "Db".to_string(),
            -6 => "Gb".to_string(),
            -7 => "Cb".to_string(),
            _ => "C".to_string(),
        }
    }

    /// Format note name with octave
    fn format_note_name(&self, note: &str, octave: u8) -> String {
        format!("{}{}", note, octave)
    }
}

/// Score renderer builder for fluent configuration
#[derive(Debug)]
pub struct ScoreRendererBuilder {
    config: RenderConfig,
}

impl ScoreRendererBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: RenderConfig::default(),
        }
    }

    /// Set the output dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set staff spacing
    pub fn staff_spacing(mut self, spacing: f32) -> Self {
        self.config.staff_spacing = spacing;
        self
    }

    /// Set note size
    pub fn note_size(mut self, size: f32) -> Self {
        self.config.note_size = size;
        self
    }

    /// Configure what elements to show
    pub fn show_elements(mut self, clef: bool, time_sig: bool, key_sig: bool) -> Self {
        self.config.show_clef = clef;
        self.config.show_time_signature = time_sig;
        self.config.show_key_signature = key_sig;
        self
    }

    /// Set measures per line
    pub fn measures_per_line(mut self, measures: u32) -> Self {
        self.config.measures_per_line = measures;
        self
    }

    /// Set font properties
    pub fn font(mut self, family: String, size: f32) -> Self {
        self.config.font_family = family;
        self.config.font_size = size;
        self
    }

    /// Build the renderer
    pub fn build(self) -> ScoreRenderer {
        ScoreRenderer::new(self.config)
    }
}

impl Default for ScoreRendererBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{KeySignature, Mode, MusicalNote, MusicalScore, Note, TimeSignature};
    use crate::types::{Articulation, Dynamics, Expression, NoteEvent};

    fn create_test_score() -> MusicalScore {
        let mut score = MusicalScore::new("Test Song".to_string(), "Test Composer".to_string());

        // Add a simple C major scale
        let notes = vec!["C", "D", "E", "F", "G", "A", "B", "C"];
        let frequencies = vec![261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25];

        for (i, (note_name, freq)) in notes.iter().zip(frequencies.iter()).enumerate() {
            let note_event = NoteEvent {
                note: note_name.to_string(),
                octave: if i == 7 { 5 } else { 4 },
                frequency: *freq,
                velocity: 0.8,
                duration: 1.0,
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
                start_time: i as f32,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: Vec::new(),
                chord: None,
            };

            score.add_note(musical_note);
        }

        score
    }

    #[test]
    fn test_score_renderer_creation() {
        let renderer = ScoreRenderer::default();
        assert_eq!(renderer.config.width, 800);
        assert_eq!(renderer.config.height, 600);
        assert!(renderer.config.show_clef);
    }

    #[test]
    fn test_score_renderer_builder() {
        let renderer = ScoreRendererBuilder::new()
            .dimensions(1000, 800)
            .staff_spacing(12.0)
            .note_size(10.0)
            .show_elements(true, true, false)
            .measures_per_line(6)
            .font("Times".to_string(), 14.0)
            .build();

        assert_eq!(renderer.config.width, 1000);
        assert_eq!(renderer.config.height, 800);
        assert_eq!(renderer.config.staff_spacing, 12.0);
        assert_eq!(renderer.config.note_size, 10.0);
        assert!(renderer.config.show_clef);
        assert!(renderer.config.show_time_signature);
        assert!(!renderer.config.show_key_signature);
        assert_eq!(renderer.config.measures_per_line, 6);
        assert_eq!(renderer.config.font_family, "Times");
        assert_eq!(renderer.config.font_size, 14.0);
    }

    #[test]
    fn test_svg_rendering() {
        let renderer = ScoreRenderer::default();
        let score = create_test_score();

        let result = renderer.render(&score, RenderFormat::Svg);
        assert!(result.is_ok());

        let svg = result.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Test Song"));
        assert!(svg.contains("Test Composer"));
        assert!(svg.contains("𝄞")); // Treble clef
    }

    #[test]
    fn test_ascii_rendering() {
        let renderer = ScoreRenderer::default();
        let score = create_test_score();

        let result = renderer.render(&score, RenderFormat::Ascii);
        assert!(result.is_ok());

        let ascii = result.unwrap();
        assert!(ascii.contains("Test Song"));
        assert!(ascii.contains("Test Composer"));
        assert!(ascii.contains("Key: C Major"));
        assert!(ascii.contains("Time: 4/4"));
        assert!(ascii.contains("Notes:"));
    }

    #[test]
    fn test_text_rendering() {
        let renderer = ScoreRenderer::default();
        let score = create_test_score();

        let result = renderer.render(&score, RenderFormat::Text);
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(text.contains("Title: Test Song"));
        assert!(text.contains("Composer: Test Composer"));
        assert!(text.contains("Key Signature: C Major"));
        assert!(text.contains("Time Signature: 4/4"));
        assert!(text.contains("Musical Notes:"));
        assert!(text.contains("C4"));
        assert!(text.contains("261.63Hz"));
    }

    #[test]
    fn test_note_position_calculation() {
        let renderer = ScoreRenderer::default();

        // Test middle C (C4) position - should be below staff
        let c4_position = renderer.calculate_note_position("C", 4);
        assert!((c4_position - (-2.0)).abs() < 0.1);

        // Test E4 (bottom staff line)
        let e4_position = renderer.calculate_note_position("E", 4);
        assert!((e4_position - 0.0).abs() < 0.1);

        // Test G4 (second staff line)
        let g4_position = renderer.calculate_note_position("G", 4);
        assert!((g4_position - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_key_signature_conversion() {
        let renderer = ScoreRenderer::default();

        let c_major = KeySignature {
            root: Note::C,
            mode: Mode::Major,
            accidentals: 0,
        };
        assert_eq!(renderer.key_signature_to_string(&c_major), "C");

        let g_major = KeySignature {
            root: Note::G,
            mode: Mode::Major,
            accidentals: 1,
        };
        assert_eq!(renderer.key_signature_to_string(&g_major), "G");

        let f_major = KeySignature {
            root: Note::F,
            mode: Mode::Major,
            accidentals: -1,
        };
        assert_eq!(renderer.key_signature_to_string(&f_major), "F");
    }

    #[test]
    fn test_ledger_lines_calculation() {
        let renderer = ScoreRenderer::default();

        // Test note above staff (needs ledger lines)
        let high_lines = renderer.calculate_ledger_lines(12.0);
        assert!(!high_lines.is_empty());
        assert!(high_lines.contains(&10.0));

        // Test note below staff (needs ledger lines)
        let low_lines = renderer.calculate_ledger_lines(-4.0);
        assert!(!low_lines.is_empty());
        assert!(low_lines.contains(&-2.0));

        // Test note on staff (no ledger lines needed)
        let staff_lines = renderer.calculate_ledger_lines(4.0);
        assert!(staff_lines.is_empty());
    }

    #[test]
    fn test_render_format_coverage() {
        let renderer = ScoreRenderer::default();
        let score = create_test_score();

        // Test all render formats
        assert!(renderer.render(&score, RenderFormat::Svg).is_ok());
        assert!(renderer.render(&score, RenderFormat::Ascii).is_ok());
        assert!(renderer.render(&score, RenderFormat::Text).is_ok());
    }

    #[test]
    fn test_empty_score_rendering() {
        let renderer = ScoreRenderer::default();
        let empty_score = MusicalScore::new("Empty".to_string(), "None".to_string());

        // Should handle empty scores gracefully
        assert!(renderer.render(&empty_score, RenderFormat::Svg).is_ok());
        assert!(renderer.render(&empty_score, RenderFormat::Ascii).is_ok());
        assert!(renderer.render(&empty_score, RenderFormat::Text).is_ok());
    }
}
