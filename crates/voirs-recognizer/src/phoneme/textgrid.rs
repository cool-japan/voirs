//! Real Praat `TextGrid` parsing.
//!
//! `TextGrid` is the annotation format Praat and the Montreal Forced Aligner write.
//! This module parses both of its serialisations — the verbose "long" form and the
//! bare "short" form — into typed tiers and intervals, so that alignment results come
//! from the aligner's real output instead of being invented.
//!
//! # Format note
//!
//! Both serialisations carry exactly the same values in exactly the same order; they
//! differ only in whether each value is preceded by a `key = ` label and wrapped in
//! structural `item [n]:` markers. The parser therefore reduces a file to its ordered
//! value stream first and then reads that stream, which makes it agnostic to the form.
//!
//! ```
//! use voirs_recognizer::phoneme::textgrid::TextGrid;
//!
//! let grid = TextGrid::parse(
//!     r#"File type = "ooTextFile"
//! Object class = "TextGrid"
//!
//! xmin = 0
//! xmax = 1.0
//! tiers? <exists>
//! size = 1
//! item []:
//!     item [1]:
//!         class = "IntervalTier"
//!         name = "words"
//!         xmin = 0
//!         xmax = 1.0
//!         intervals: size = 1
//!         intervals [1]:
//!             xmin = 0
//!             xmax = 1.0
//!             text = "hello"
//! "#,
//! )
//! .expect("valid TextGrid");
//!
//! let words = grid.tier("words").expect("words tier");
//! assert_eq!(words.intervals[0].text, "hello");
//! ```

use std::fmt;

/// Failure modes of [`TextGrid::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextGridError {
    /// The file does not identify itself as a `TextGrid`.
    NotATextGrid,
    /// A quoted string was opened but never closed.
    UnterminatedString {
        /// 1-based line number where the string starts.
        line: usize,
    },
    /// The value stream ended while a field was still expected.
    UnexpectedEnd {
        /// Name of the field that was being read.
        field: &'static str,
    },
    /// A value could not be interpreted as the expected kind.
    Malformed {
        /// Name of the field that was being read.
        field: &'static str,
        /// The offending value.
        value: String,
    },
}

impl fmt::Display for TextGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotATextGrid => {
                write!(f, "input does not declare `Object class = \"TextGrid\"`")
            }
            Self::UnterminatedString { line } => {
                write!(f, "unterminated quoted string starting on line {line}")
            }
            Self::UnexpectedEnd { field } => {
                write!(f, "TextGrid ended while reading `{field}`")
            }
            Self::Malformed { field, value } => {
                write!(f, "`{field}` has malformed value {value:?}")
            }
        }
    }
}

impl std::error::Error for TextGridError {}

/// One labelled time span on an interval tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    /// Start time in seconds.
    pub xmin: f64,
    /// End time in seconds.
    pub xmax: f64,
    /// Label text; empty for the silence intervals aligners emit between words.
    pub text: String,
}

impl Interval {
    /// Duration of the interval in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        (self.xmax - self.xmin).max(0.0)
    }

    /// Whether the label is empty or one of the conventional silence symbols.
    #[must_use]
    pub fn is_silence(&self) -> bool {
        is_silence_label(&self.text)
    }
}

/// Whether a `TextGrid` label denotes silence rather than speech.
///
/// Covers the empty label plus the silence/short-pause/unknown symbols the Montreal
/// Forced Aligner and Praat conventionally emit.
#[must_use]
pub fn is_silence_label(label: &str) -> bool {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return true;
    }
    matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "sil" | "sp" | "spn" | "silence" | "<eps>" | "#" | "pau"
    )
}

/// The two tier kinds Praat defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierKind {
    /// A tier of labelled time spans.
    Interval,
    /// A tier of labelled instants; each point is stored as a zero-length interval.
    Point,
}

/// A single named tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Tier {
    /// Tier name, e.g. `words` or `phones`.
    pub name: String,
    /// Whether the tier holds spans or instants.
    pub kind: TierKind,
    /// Tier start time in seconds.
    pub xmin: f64,
    /// Tier end time in seconds.
    pub xmax: f64,
    /// The tier's entries, in file order.
    pub intervals: Vec<Interval>,
}

impl Tier {
    /// Entries whose label is not silence.
    pub fn speech_intervals(&self) -> impl Iterator<Item = &Interval> {
        self.intervals.iter().filter(|i| !i.is_silence())
    }
}

/// A parsed `TextGrid` file.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGrid {
    /// File start time in seconds.
    pub xmin: f64,
    /// File end time in seconds.
    pub xmax: f64,
    /// All tiers, in file order.
    pub tiers: Vec<Tier>,
}

impl TextGrid {
    /// Parse a `TextGrid` in either the long or the short serialisation.
    ///
    /// # Errors
    /// Returns [`TextGridError`] when the input is not a `TextGrid`, contains an
    /// unterminated string, ends early, or holds a value of the wrong kind.
    pub fn parse(input: &str) -> Result<Self, TextGridError> {
        if !input.contains("\"TextGrid\"") {
            return Err(TextGridError::NotATextGrid);
        }

        let values = value_stream(input)?;
        let mut cursor = Cursor::new(&values);

        let xmin = cursor.number("xmin")?;
        let xmax = cursor.number("xmax")?;
        let tier_count = cursor.count("size")?;

        let mut tiers = Vec::with_capacity(tier_count.min(1024));
        for _ in 0..tier_count {
            let class = cursor.text("class")?;
            let kind = match class.as_str() {
                "IntervalTier" => TierKind::Interval,
                "TextTier" => TierKind::Point,
                _ => {
                    return Err(TextGridError::Malformed {
                        field: "class",
                        value: class,
                    })
                }
            };
            let name = cursor.text("name")?;
            let tier_xmin = cursor.number("tier xmin")?;
            let tier_xmax = cursor.number("tier xmax")?;
            let entry_count = cursor.count("intervals: size")?;

            let mut intervals = Vec::with_capacity(entry_count.min(1 << 20));
            for _ in 0..entry_count {
                match kind {
                    TierKind::Interval => {
                        let interval_xmin = cursor.number("interval xmin")?;
                        let interval_xmax = cursor.number("interval xmax")?;
                        let text = cursor.text("text")?;
                        intervals.push(Interval {
                            xmin: interval_xmin,
                            xmax: interval_xmax,
                            text,
                        });
                    }
                    TierKind::Point => {
                        let time = cursor.number("point number")?;
                        let text = cursor.text("mark")?;
                        intervals.push(Interval {
                            xmin: time,
                            xmax: time,
                            text,
                        });
                    }
                }
            }

            tiers.push(Tier {
                name,
                kind,
                xmin: tier_xmin,
                xmax: tier_xmax,
                intervals,
            });
        }

        Ok(Self { xmin, xmax, tiers })
    }

    /// The first tier whose name matches `name`, case-insensitively.
    #[must_use]
    pub fn tier(&self, name: &str) -> Option<&Tier> {
        self.tiers
            .iter()
            .find(|tier| tier.name.eq_ignore_ascii_case(name))
    }

    /// The first tier matching any of `names`, in preference order.
    #[must_use]
    pub fn tier_any(&self, names: &[&str]) -> Option<&Tier> {
        names.iter().find_map(|name| self.tier(name))
    }

    /// Total duration covered by the file, in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        (self.xmax - self.xmin).max(0.0)
    }
}

/// A borrowed position in the extracted value stream.
struct Cursor<'a> {
    values: &'a [String],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn new(values: &'a [String]) -> Self {
        Self { values, index: 0 }
    }

    fn next_value(&mut self, field: &'static str) -> Result<&'a str, TextGridError> {
        let value = self
            .values
            .get(self.index)
            .ok_or(TextGridError::UnexpectedEnd { field })?;
        self.index += 1;
        Ok(value.as_str())
    }

    fn number(&mut self, field: &'static str) -> Result<f64, TextGridError> {
        let raw = self.next_value(field)?;
        raw.trim()
            .parse::<f64>()
            .map_err(|_| TextGridError::Malformed {
                field,
                value: raw.to_string(),
            })
    }

    fn count(&mut self, field: &'static str) -> Result<usize, TextGridError> {
        let raw = self.next_value(field)?;
        raw.trim()
            .parse::<usize>()
            .map_err(|_| TextGridError::Malformed {
                field,
                value: raw.to_string(),
            })
    }

    fn text(&mut self, field: &'static str) -> Result<String, TextGridError> {
        Ok(self.next_value(field)?.to_string())
    }
}

/// Reduce a `TextGrid` to its ordered stream of values, discarding key labels and
/// structural markers so that the long and short forms become identical.
fn value_stream(input: &str) -> Result<Vec<String>, TextGridError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut values = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        let line_number = index + 1;
        let line = lines[index].trim();
        index += 1;

        if line.is_empty() || is_structural(line) {
            continue;
        }

        // In the long form every value is written as `key = value`, and keys never
        // contain `=`, so the first `=` is always the separator. The short form has no
        // key at all, in which case the whole line is the value.
        let payload = match line.find('=') {
            Some(position) => line[position + 1..].trim(),
            None => line,
        };

        if let Some(rest) = payload.strip_prefix('"') {
            let (text, terminated) = unescape_quoted(rest);
            if terminated {
                values.push(text);
                continue;
            }
            // Praat permits labels containing newlines; keep consuming lines until the
            // closing quote is found.
            let mut accumulated = text;
            let mut closed = false;
            while index < lines.len() {
                let continuation = lines[index];
                index += 1;
                accumulated.push('\n');
                let (chunk, terminated) = unescape_quoted(continuation);
                accumulated.push_str(&chunk);
                if terminated {
                    closed = true;
                    break;
                }
            }
            if !closed {
                return Err(TextGridError::UnterminatedString { line: line_number });
            }
            values.push(accumulated);
        } else {
            values.push(payload.to_string());
        }
    }

    Ok(values)
}

/// Whether a line is a pure structural marker carrying no value.
fn is_structural(line: &str) -> bool {
    if line.starts_with("File type") || line.starts_with("Object class") {
        return true;
    }
    if line.starts_with("tiers?") || line == "<exists>" || line == "<absent>" {
        return true;
    }
    // `item []:`, `item [3]:`, `intervals [12]:`, `points [1]:`
    let Some(bracket) = line.find('[') else {
        return false;
    };
    let head = line[..bracket].trim();
    if !matches!(head, "item" | "intervals" | "points") {
        return false;
    }
    let Some(close) = line.find(']') else {
        return false;
    };
    close > bracket
        && line[close + 1..].trim() == ":"
        && line[bracket + 1..close].chars().all(|c| c.is_ascii_digit())
}

/// Read the body of a quoted Praat string, resolving the `""` escape for a literal
/// quote. Returns the decoded text and whether the closing quote was found.
fn unescape_quoted(rest: &str) -> (String, bool) {
    let mut out = String::with_capacity(rest.len());
    let mut chars = rest.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if chars.peek() == Some(&'"') {
                chars.next();
                out.push('"');
            } else {
                return (out, true);
            }
        } else {
            out.push(ch);
        }
    }
    (out, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_FORM: &str = r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.25
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.25
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.6
            text = "hello"
        intervals [3]:
            xmin = 0.6
            xmax = 1.25
            text = "world"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.25
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.35
            text = "HH"
        intervals [3]:
            xmin = 0.35
            xmax = 0.6
            text = "AH0"
"#;

    const SHORT_FORM: &str = r#"File type = "ooTextFile"
Object class = "TextGrid"

0
1.25
<exists>
1
"IntervalTier"
"words"
0
1.25
2
0
0.6
"hello"
0.6
1.25
"world"
"#;

    #[test]
    fn parses_long_form_with_two_tiers() {
        let grid = TextGrid::parse(LONG_FORM).expect("long form must parse");
        assert!((grid.xmax - 1.25).abs() < 1e-9);
        assert_eq!(grid.tiers.len(), 2);

        let words = grid.tier("words").expect("words tier");
        assert_eq!(words.kind, TierKind::Interval);
        assert_eq!(words.intervals.len(), 3);
        assert_eq!(words.intervals[1].text, "hello");
        assert!((words.intervals[1].xmin - 0.1).abs() < 1e-9);
        assert!((words.intervals[1].duration() - 0.5).abs() < 1e-9);
        assert!(words.intervals[0].is_silence());

        let spoken: Vec<&str> = words
            .speech_intervals()
            .map(|i| i.text.as_str())
            .collect();
        assert_eq!(spoken, vec!["hello", "world"]);

        let phones = grid.tier("PHONES").expect("case-insensitive tier lookup");
        assert_eq!(phones.intervals[2].text, "AH0");
        assert!(phones.intervals[0].is_silence(), "`sil` counts as silence");
    }

    #[test]
    fn parses_short_form_identically() {
        let grid = TextGrid::parse(SHORT_FORM).expect("short form must parse");
        assert_eq!(grid.tiers.len(), 1);
        let words = &grid.tiers[0];
        assert_eq!(words.name, "words");
        assert_eq!(words.intervals.len(), 2);
        assert_eq!(words.intervals[0].text, "hello");
        assert_eq!(words.intervals[1].text, "world");
        assert!((grid.duration() - 1.25).abs() < 1e-9);
    }

    #[test]
    fn tier_any_prefers_earlier_names() {
        let grid = TextGrid::parse(LONG_FORM).expect("parse");
        let tier = grid
            .tier_any(&["phone", "phones"])
            .expect("falls through to `phones`");
        assert_eq!(tier.name, "phones");
        assert!(grid.tier_any(&["nope"]).is_none());
    }

    #[test]
    fn rejects_non_textgrid_input() {
        assert_eq!(
            TextGrid::parse("File type = \"ooTextFile\"\nObject class = \"Sound\"\n"),
            Err(TextGridError::NotATextGrid)
        );
    }

    #[test]
    fn reports_truncated_files() {
        let truncated = "Object class = \"TextGrid\"\nxmin = 0\nxmax = 1\n";
        match TextGrid::parse(truncated) {
            Err(TextGridError::UnexpectedEnd { field }) => assert_eq!(field, "size"),
            other => panic!("expected UnexpectedEnd, got {other:?}"),
        }
    }

    #[test]
    fn reports_malformed_numbers() {
        let bad = "Object class = \"TextGrid\"\nxmin = zero\n";
        match TextGrid::parse(bad) {
            Err(TextGridError::Malformed { field, value }) => {
                assert_eq!(field, "xmin");
                assert_eq!(value, "zero");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_tier_class() {
        let bad = "Object class = \"TextGrid\"\n0\n1\n1\n\"WeirdTier\"\n\"x\"\n";
        match TextGrid::parse(bad) {
            Err(TextGridError::Malformed { field, value }) => {
                assert_eq!(field, "class");
                assert_eq!(value, "WeirdTier");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn handles_labels_containing_equals_and_quotes() {
        let grid = TextGrid::parse(
            "Object class = \"TextGrid\"\n\
             xmin = 0\nxmax = 1\nsize = 1\n\
             class = \"IntervalTier\"\nname = \"words\"\nxmin = 0\nxmax = 1\n\
             intervals: size = 2\n\
             xmin = 0\nxmax = 0.5\ntext = \"a = b\"\n\
             xmin = 0.5\nxmax = 1\ntext = \"say \"\"hi\"\"\"\n",
        )
        .expect("parse");
        let words = &grid.tiers[0];
        assert_eq!(words.intervals[0].text, "a = b");
        assert_eq!(words.intervals[1].text, "say \"hi\"");
    }

    #[test]
    fn parses_point_tiers_as_zero_length_intervals() {
        let grid = TextGrid::parse(
            "Object class = \"TextGrid\"\n0\n2\n<exists>\n1\n\"TextTier\"\n\"marks\"\n0\n2\n1\n\
             0.75\n\"beep\"\n",
        )
        .expect("parse");
        let tier = grid.tier("marks").expect("marks tier");
        assert_eq!(tier.kind, TierKind::Point);
        assert_eq!(tier.intervals.len(), 1);
        assert!((tier.intervals[0].xmin - 0.75).abs() < 1e-9);
        assert_eq!(tier.intervals[0].duration(), 0.0);
        assert_eq!(tier.intervals[0].text, "beep");
    }

    #[test]
    fn silence_labels_are_recognised() {
        for label in ["", "  ", "sil", "SIL", "sp", "spn", "<eps>", "#"] {
            assert!(is_silence_label(label), "{label:?} should be silence");
        }
        for label in ["hello", "HH", "AH0"] {
            assert!(!is_silence_label(label), "{label:?} should be speech");
        }
    }

    #[test]
    fn reports_unterminated_strings() {
        let bad = "Object class = \"TextGrid\"\n0\n1\n1\n\"IntervalTier\"\n\"unclosed\n";
        match TextGrid::parse(bad) {
            Err(TextGridError::UnterminatedString { line }) => assert_eq!(line, 6),
            other => panic!("expected UnterminatedString, got {other:?}"),
        }
    }
}
