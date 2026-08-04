//! Real invocation of the Montreal Forced Aligner command-line tool.
//!
//! The Montreal Forced Aligner (MFA) is a separate Kaldi/Python program, not a Rust
//! library. VoiRS therefore drives the real `mfa` executable as a subprocess: it probes
//! for the binary, asks it which acoustic models and dictionaries are really installed,
//! writes a real corpus to a temporary directory, runs a real alignment, and reads the
//! `TextGrid` files MFA writes.
//!
//! Every function here fails closed with a typed [`RecognitionError`] when MFA is not
//! installed or when a run fails — nothing is ever simulated or substituted.

use crate::RecognitionError;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::Duration;

/// Kinds of asset the `mfa model` subcommand manages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// A trained acoustic model.
    Acoustic,
    /// A pronunciation dictionary.
    Dictionary,
    /// A grapheme-to-phoneme model.
    G2p,
    /// A language model.
    LanguageModel,
}

impl ModelKind {
    /// The token the `mfa model` subcommand uses for this kind.
    #[must_use]
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Acoustic => "acoustic",
            Self::Dictionary => "dictionary",
            Self::G2p => "g2p",
            Self::LanguageModel => "language_model",
        }
    }
}

/// A handle to a real, verified `mfa` executable.
#[derive(Debug, Clone)]
pub struct MfaCli {
    executable: PathBuf,
    version: String,
}

impl MfaCli {
    /// Locate the `mfa` executable and record the version it really reports.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the executable is absent from
    /// `PATH`, cannot be run, or exits with a failure status.
    pub fn discover(executable: &str) -> Result<Self, RecognitionError> {
        let output = std::process::Command::new(executable)
            .arg("version")
            .output()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    not_installed_error(executable)
                } else {
                    RecognitionError::ModelLoadError {
                        message: format!("Failed to run `{executable} version`: {e}"),
                        source: Some(Box::new(e)),
                    }
                }
            })?;

        if !output.status.success() {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "`{executable} version` exited with {}: {}",
                    output.status,
                    stderr_snippet(&output)
                ),
                source: None,
            });
        }

        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let version = if version.is_empty() {
            String::from_utf8_lossy(&output.stderr).trim().to_string()
        } else {
            version
        };

        Ok(Self {
            executable: PathBuf::from(executable),
            version,
        })
    }

    /// The version string the installed aligner really printed.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The resolved executable path.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Ask MFA which assets of `kind` are really installed.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the subcommand cannot be run or
    /// exits with a failure status.
    pub fn list_models(&self, kind: ModelKind) -> Result<Vec<String>, RecognitionError> {
        let output = self.run(&[
            OsStr::new("model"),
            OsStr::new("list"),
            OsStr::new(kind.as_arg()),
        ])?;
        Ok(parse_model_list(&String::from_utf8_lossy(&output.stdout)))
    }

    /// Download an asset through MFA's own downloader.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the download subcommand fails.
    pub fn download_model(&self, kind: ModelKind, name: &str) -> Result<(), RecognitionError> {
        self.run(&[
            OsStr::new("model"),
            OsStr::new("download"),
            OsStr::new(kind.as_arg()),
            OsStr::new(name),
        ])?;
        Ok(())
    }

    /// Run `mfa align` over a prepared corpus and return the output directory.
    ///
    /// `dictionary` and `acoustic_model` may be either installed asset names or
    /// filesystem paths; MFA accepts both.
    ///
    /// # Errors
    /// Returns [`RecognitionError::PhonemeRecognitionError`] when the aligner exits with
    /// a failure status, quoting its real stderr.
    pub fn align(&self, request: &AlignRequest<'_>) -> Result<(), RecognitionError> {
        let jobs = request.num_jobs.max(1).to_string();
        let beam = format_finite(request.beam_width, 10.0);
        let retry_beam = format_finite(request.retry_beam, 40.0);

        let mut args: Vec<&OsStr> = vec![
            OsStr::new("align"),
            OsStr::new("--clean"),
            OsStr::new("--no_use_mp"),
            OsStr::new("--num_jobs"),
            OsStr::new(jobs.as_str()),
            OsStr::new("--beam"),
            OsStr::new(beam.as_str()),
            OsStr::new("--retry_beam"),
            OsStr::new(retry_beam.as_str()),
            OsStr::new("--output_format"),
            OsStr::new("long_textgrid"),
        ];
        if !request.cleanup {
            args.push(OsStr::new("--debug"));
        }
        args.push(request.corpus_dir.as_os_str());
        args.push(request.dictionary.as_ref());
        args.push(request.acoustic_model.as_ref());
        args.push(request.output_dir.as_os_str());

        let output = self.spawn(&args).map_err(|e| e.into_alignment_error())?;
        if !output.status.success() {
            return Err(RecognitionError::PhonemeRecognitionError {
                message: format!(
                    "Montreal Forced Aligner failed with {}: {}",
                    output.status,
                    stderr_snippet(&output)
                ),
                source: None,
            });
        }
        Ok(())
    }

    /// Train a new acoustic model from a real corpus.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when training exits with a failure
    /// status.
    pub fn train(
        &self,
        corpus_dir: &Path,
        dictionary: &OsStr,
        output_model: &Path,
        num_jobs: usize,
    ) -> Result<(), RecognitionError> {
        let jobs = num_jobs.max(1).to_string();
        self.run(&[
            OsStr::new("train"),
            OsStr::new("--clean"),
            OsStr::new("--num_jobs"),
            OsStr::new(jobs.as_str()),
            corpus_dir.as_os_str(),
            dictionary,
            output_model.as_os_str(),
        ])?;
        Ok(())
    }

    fn run(&self, args: &[&OsStr]) -> Result<Output, RecognitionError> {
        let output = self.spawn(args).map_err(SpawnError::into_load_error)?;
        if !output.status.success() {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "`{} {}` exited with {}: {}",
                    self.executable.display(),
                    render_args(args),
                    output.status,
                    stderr_snippet(&output)
                ),
                source: None,
            });
        }
        Ok(output)
    }

    fn spawn(&self, args: &[&OsStr]) -> Result<Output, SpawnError> {
        std::process::Command::new(&self.executable)
            .args(args)
            .output()
            .map_err(|e| SpawnError {
                executable: self.executable.display().to_string(),
                rendered_args: render_args(args),
                source: e,
            })
    }
}

/// Everything one `mfa align` invocation needs.
#[derive(Debug)]
pub struct AlignRequest<'a> {
    /// Directory holding the `.wav`/`.lab` pairs to align.
    pub corpus_dir: &'a Path,
    /// Installed dictionary name or path to a dictionary file.
    pub dictionary: &'a OsStr,
    /// Installed acoustic model name or path to a model file.
    pub acoustic_model: &'a OsStr,
    /// Directory MFA should write `TextGrid` files into.
    pub output_dir: &'a Path,
    /// Parallel job count.
    pub num_jobs: usize,
    /// Decoding beam width.
    pub beam_width: f32,
    /// Wider beam used when the first pass fails.
    pub retry_beam: f32,
    /// Whether MFA should clean its temporary state.
    pub cleanup: bool,
}

struct SpawnError {
    executable: String,
    rendered_args: String,
    source: std::io::Error,
}

impl SpawnError {
    fn into_load_error(self) -> RecognitionError {
        if self.source.kind() == std::io::ErrorKind::NotFound {
            return not_installed_error(&self.executable);
        }
        RecognitionError::ModelLoadError {
            message: format!(
                "Failed to run `{} {}`: {}",
                self.executable, self.rendered_args, self.source
            ),
            source: Some(Box::new(self.source)),
        }
    }

    fn into_alignment_error(self) -> RecognitionError {
        if self.source.kind() == std::io::ErrorKind::NotFound {
            return not_installed_error(&self.executable);
        }
        RecognitionError::PhonemeRecognitionError {
            message: format!(
                "Failed to run `{} {}`: {}",
                self.executable, self.rendered_args, self.source
            ),
            source: Some(Box::new(self.source)),
        }
    }
}

/// The typed error every entry point returns when MFA is simply not installed.
#[must_use]
pub fn not_installed_error(executable: &str) -> RecognitionError {
    RecognitionError::ModelLoadError {
        message: format!(
            "Montreal Forced Aligner executable '{executable}' was not found on PATH. MFA is a \
             separate Kaldi-based program; install it (for example `conda install -c conda-forge \
             montreal-forced-aligner`) and make sure `{executable} version` runs, or use \
             ForcedAlignModel, which aligns with VoiRS's own MFCC + DTW implementation and needs \
             no external tools."
        ),
        source: None,
    }
}

/// Parse the asset names out of `mfa model list <kind>` output.
///
/// MFA prints a short banner and then one asset per line, sometimes as a bulleted or
/// numbered list. Blank lines, banner lines and decorations are discarded.
#[must_use]
pub fn parse_model_list(stdout: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in stdout.lines() {
        let mut candidate = line.trim();
        if candidate.is_empty() {
            continue;
        }
        // Drop list decorations such as "- name", "* name" or "1. name".
        candidate = candidate
            .trim_start_matches(['-', '*', '+', '\u{2022}'])
            .trim_start();
        if let Some((prefix, rest)) = candidate.split_once('.') {
            if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                candidate = rest.trim_start();
            }
        }
        if candidate.is_empty() {
            continue;
        }
        // Banner and informational lines are prose, never bare asset names.
        let looks_like_prose = candidate.contains(' ')
            || candidate.contains(':')
            || candidate.starts_with('=')
            || candidate.starts_with('#');
        if looks_like_prose {
            continue;
        }
        names.push(candidate.to_string());
    }
    names
}

/// Format a duration for log messages without pulling in a formatting crate.
#[must_use]
pub fn format_duration(duration: Duration) -> String {
    format!("{:.2}s", duration.as_secs_f64())
}

fn format_finite(value: f32, fallback: f32) -> String {
    let effective = if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    };
    format!("{effective:.0}")
}

fn render_args(args: &[&OsStr]) -> String {
    args.iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

fn stderr_snippet(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trimmed = stderr.trim();
    let source = if trimmed.is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        trimmed.to_string()
    };
    if source.is_empty() {
        return "(no diagnostic output)".to_string();
    }
    // Keep messages bounded; MFA can emit very long tracebacks.
    let mut snippet: String = source.chars().take(600).collect();
    if source.chars().count() > 600 {
        snippet.push_str(" ...");
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_list_parsing_keeps_only_asset_names() {
        let stdout = "\
Available acoustic models:
============================
english_us_arpa
english_mfa
  - german_mfa
  2. french_mfa

Use `mfa model download acoustic <name>` to install more.
";
        let names = parse_model_list(stdout);
        assert_eq!(
            names,
            vec!["english_us_arpa", "english_mfa", "german_mfa", "french_mfa"]
        );
    }

    #[test]
    fn model_list_parsing_handles_empty_output() {
        assert!(parse_model_list("").is_empty());
        assert!(parse_model_list("\n\n   \n").is_empty());
    }

    #[test]
    fn missing_executable_is_reported_as_typed_error() {
        // A name that cannot exist on PATH.
        let err = MfaCli::discover("voirs-definitely-not-a-real-mfa-binary").unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("was not found on PATH"), "got: {message}");
                assert!(message.contains("ForcedAlignModel"), "got: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    #[test]
    fn model_kind_arguments_match_the_cli() {
        assert_eq!(ModelKind::Acoustic.as_arg(), "acoustic");
        assert_eq!(ModelKind::Dictionary.as_arg(), "dictionary");
        assert_eq!(ModelKind::G2p.as_arg(), "g2p");
        assert_eq!(ModelKind::LanguageModel.as_arg(), "language_model");
    }

    #[test]
    fn beam_values_fall_back_when_not_finite() {
        assert_eq!(format_finite(f32::NAN, 10.0), "10");
        assert_eq!(format_finite(-1.0, 10.0), "10");
        assert_eq!(format_finite(0.0, 40.0), "40");
        assert_eq!(format_finite(12.4, 10.0), "12");
    }

    #[test]
    fn duration_formatting_is_stable() {
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.50s");
    }
}
