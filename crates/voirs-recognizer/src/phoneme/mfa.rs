//! Montreal Forced Aligner (MFA) integration.
//!
//! MFA is a separate Kaldi/Python program, so this module does not reimplement it — it
//! drives the real `mfa` executable. Every operation is a real subprocess run against a
//! real corpus written to a temporary directory, and every alignment comes from parsing
//! the real `TextGrid` files MFA emits.
//!
//! When MFA is not installed, or the requested acoustic model or dictionary is not
//! really available, every entry point fails closed with a typed
//! [`RecognitionError`] naming what is missing. Nothing is simulated and no alignment is
//! ever invented.
//!
//! For an aligner that needs no external tools, use
//! [`ForcedAlignModel`](super::forced_align::ForcedAlignModel), which implements MFCC
//! extraction plus DTW alignment in pure Rust.

use super::mfa_cli::{self, AlignRequest, MfaCli, ModelKind};
use super::textgrid::{is_silence_label, Interval, TextGrid, Tier};
use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Frame length used when scoring how well an aligned interval matches the audio.
const SCORE_FRAME_SAMPLES: usize = 400; // 25 ms at 16 kHz

/// Montreal Forced Alignment model, backed by the real `mfa` command-line tool.
pub struct MFAModel {
    /// Model configuration
    config: MFAConfig,
    /// Model state
    state: Arc<RwLock<MFAState>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: PhonemeRecognizerMetadata,
}

/// MFA model configuration
#[derive(Debug, Clone)]
pub struct MFAConfig {
    /// Installed acoustic-model name, or a path to a model file
    pub model: String,
    /// Installed dictionary name, or a path to a dictionary file
    pub dictionary: String,
    /// Explicit acoustic model path, overriding `model` when set
    pub acoustic_model_path: Option<String>,
    /// Number of jobs for parallel processing (`mfa align --num_jobs`)
    pub num_jobs: usize,
    /// Pass `mfa align --clean`, so MFA clears its temporary state before each run
    pub cleanup: bool,
    /// Beam width for alignment (`mfa align --beam`)
    pub beam_width: f32,
    /// Retry beam for alignment (`mfa align --retry_beam`)
    pub retry_beam: f32,
    /// Include phone alignment in the returned result
    pub include_phone_alignment: bool,
    /// Include word alignment in the returned result
    pub include_word_alignment: bool,
    /// Name or path of the `mfa` executable
    pub executable: String,
    /// Automatically run `mfa model download` for assets that are not installed
    pub auto_download: bool,
}

impl Default for MFAConfig {
    fn default() -> Self {
        Self {
            model: "english_us_arpa".to_string(),
            dictionary: "english_us_arpa".to_string(),
            acoustic_model_path: None,
            num_jobs: num_cpus::get(),
            cleanup: true,
            beam_width: 10.0,
            retry_beam: 40.0,
            include_phone_alignment: true,
            include_word_alignment: true,
            executable: "mfa".to_string(),
            auto_download: false,
        }
    }
}

/// Internal state for MFA model
struct MFAState {
    /// Verified handle to the real `mfa` executable, once discovered
    cli: Option<MfaCli>,
    /// Acoustic-model identifier
    model: String,
    /// Dictionary identifier
    dictionary: String,
    /// Acoustic model names MFA really reported as installed
    available_models: Vec<String>,
    /// Dictionary names MFA really reported as installed
    available_dictionaries: Vec<String>,
    /// How long discovery and validation really took
    load_time: Option<Duration>,
    /// Alignment count
    alignment_count: usize,
    /// Total alignment time
    total_alignment_time: Duration,
}

/// Information about an installed MFA acoustic model.
///
/// `mfa model list` reports names only, so that is all this carries. Size, architecture
/// and training-corpus fields were removed rather than filled with invented values;
/// inspect the model file yourself if you need more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MFAModelInfo {
    /// Model name as MFA reports it
    pub name: String,
    /// Language inferred from the model name, when the name identifies one
    pub language: Option<LanguageCode>,
}

/// Information about an installed MFA pronunciation dictionary.
///
/// As with [`MFAModelInfo`], only what MFA really reports is carried. Use
/// [`MFAModel::validate_dictionary`] with a path to count a dictionary's real entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MFADictionaryInfo {
    /// Dictionary name as MFA reports it
    pub name: String,
    /// Language inferred from the dictionary name, when the name identifies one
    pub language: Option<LanguageCode>,
}

impl MFAState {
    fn new(model: String, dictionary: String) -> Self {
        Self {
            cli: None,
            model,
            dictionary,
            available_models: Vec::new(),
            available_dictionaries: Vec::new(),
            load_time: None,
            alignment_count: 0,
            total_alignment_time: Duration::ZERO,
        }
    }
}

impl MFAModel {
    /// Create a new MFA model handle.
    ///
    /// Construction does not touch the filesystem; the real `mfa` executable is probed
    /// lazily on the first alignment so that building a handle never blocks.
    ///
    /// # Errors
    /// Currently infallible, but returns `Result` so that future validation can be added
    /// without a breaking change.
    pub async fn new(
        model: String,
        dictionary: String,
        acoustic_model_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        let config = MFAConfig {
            model: model.clone(),
            dictionary: dictionary.clone(),
            acoustic_model_path,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create with custom configuration.
    ///
    /// # Errors
    /// Currently infallible; see [`MFAModel::new`].
    pub async fn with_config(config: MFAConfig) -> Result<Self, RecognitionError> {
        let supported_languages = Self::get_supported_languages(&config.model, &config.dictionary);

        let metadata = PhonemeRecognizerMetadata {
            name: format!("Montreal Forced Alignment ({})", config.model),
            // Filled in with the real reported version once the executable is probed.
            version: "unknown (mfa not yet probed)".to_string(),
            description: "Montreal Forced Alignment, driven as a real subprocess. Requires the \
                          `mfa` executable and its acoustic model/dictionary to be installed."
                .to_string(),
            supported_languages: supported_languages.clone(),
            alignment_methods: vec![AlignmentMethod::Forced],
            // 0.0 == not measured by VoiRS. Alignment accuracy depends entirely on the
            // installed acoustic model and the audio, so no figure is asserted here.
            alignment_accuracy: 0.0,
            supported_features: vec![
                PhonemeRecognitionFeature::WordAlignment,
                PhonemeRecognitionFeature::CustomPronunciation,
                PhonemeRecognitionFeature::MultiLanguage,
                PhonemeRecognitionFeature::ConfidenceScoring,
            ],
        };

        let state = Arc::new(RwLock::new(MFAState::new(
            config.model.clone(),
            config.dictionary.clone(),
        )));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Probe for the real executable and validate the configured assets.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when `mfa` is not installed or when
    /// the configured acoustic model or dictionary is not really available.
    async fn ensure_loaded(&self) -> Result<MfaCli, RecognitionError> {
        {
            let state = self.state.read().await;
            if let Some(cli) = &state.cli {
                return Ok(cli.clone());
            }
        }

        let start_time = Instant::now();
        let executable = self.config.executable.clone();
        let auto_download = self.config.auto_download;
        let model = self.config.model.clone();
        let dictionary = self.config.dictionary.clone();
        let model_is_path = self
            .acoustic_model_arg()
            .is_some_and(|arg| is_existing_path(&arg));
        let dictionary_is_path = is_existing_path(&OsString::from(&dictionary));

        // The MFA CLI is blocking, so keep it off the async runtime's worker threads.
        let probed = tokio::task::spawn_blocking(move || -> Result<Probe, RecognitionError> {
            let cli = MfaCli::discover(&executable)?;
            let acoustic = cli.list_models(ModelKind::Acoustic)?;
            let dictionaries = cli.list_models(ModelKind::Dictionary)?;

            // Pre-validation is a courtesy that turns a long MFA failure into a short,
            // precise message. It is skipped when `mfa model list` printed nothing this
            // parser recognised, because an unfamiliar output layout must not block a
            // model that really is installed — `mfa align` remains the authority.
            if !acoustic.is_empty() && !model_is_path && !acoustic.iter().any(|name| name == &model)
            {
                if auto_download {
                    cli.download_model(ModelKind::Acoustic, &model)?;
                } else {
                    return Err(missing_asset_error("acoustic model", &model, &acoustic));
                }
            }
            if !dictionaries.is_empty()
                && !dictionary_is_path
                && !dictionaries.iter().any(|name| name == &dictionary)
            {
                if auto_download {
                    cli.download_model(ModelKind::Dictionary, &dictionary)?;
                } else {
                    return Err(missing_asset_error(
                        "dictionary",
                        &dictionary,
                        &dictionaries,
                    ));
                }
            }

            Ok(Probe {
                cli,
                acoustic,
                dictionaries,
            })
        })
        .await
        .map_err(|e| RecognitionError::ModelLoadError {
            message: format!("MFA discovery task failed: {e}"),
            source: Some(Box::new(e)),
        })??;

        let mut state = self.state.write().await;
        state.available_models = probed.acoustic;
        state.available_dictionaries = probed.dictionaries;
        state.load_time = Some(start_time.elapsed());
        state.cli = Some(probed.cli.clone());

        tracing::info!(
            "Montreal Forced Aligner {} ready in {}",
            probed.cli.version(),
            mfa_cli::format_duration(start_time.elapsed())
        );

        Ok(probed.cli)
    }

    /// The acoustic-model argument handed to MFA: the explicit path when configured,
    /// otherwise the installed model name.
    fn acoustic_model_arg(&self) -> Option<OsString> {
        self.config
            .acoustic_model_path
            .as_ref()
            .map(OsString::from)
            .or_else(|| Some(OsString::from(&self.config.model)))
    }

    /// Get supported languages based on model and dictionary names.
    fn get_supported_languages(model: &str, dictionary: &str) -> Vec<LanguageCode> {
        let joined = format!("{} {}", model.to_lowercase(), dictionary.to_lowercase());
        if joined.contains("english") {
            vec![LanguageCode::EnUs, LanguageCode::EnGb]
        } else if joined.contains("german") {
            vec![LanguageCode::DeDe]
        } else if joined.contains("french") {
            vec![LanguageCode::FrFr]
        } else if joined.contains("spanish") {
            vec![LanguageCode::EsEs, LanguageCode::EsMx]
        } else if joined.contains("japanese") {
            vec![LanguageCode::JaJp]
        } else if joined.contains("mandarin") || joined.contains("chinese") {
            vec![LanguageCode::ZhCn]
        } else if joined.contains("korean") {
            vec![LanguageCode::KoKr]
        } else {
            vec![LanguageCode::EnUs]
        }
    }

    /// Language inferred from an asset name, or `None` when the name says nothing.
    fn infer_language(name: &str) -> Option<LanguageCode> {
        let lowered = name.to_lowercase();
        if lowered.contains("english") {
            Some(LanguageCode::EnUs)
        } else if lowered.contains("german") {
            Some(LanguageCode::DeDe)
        } else if lowered.contains("french") {
            Some(LanguageCode::FrFr)
        } else if lowered.contains("spanish") {
            Some(LanguageCode::EsEs)
        } else if lowered.contains("japanese") {
            Some(LanguageCode::JaJp)
        } else if lowered.contains("mandarin") || lowered.contains("chinese") {
            Some(LanguageCode::ZhCn)
        } else if lowered.contains("korean") {
            Some(LanguageCode::KoKr)
        } else {
            None
        }
    }

    /// Run a real alignment and parse MFA's real `TextGrid` output.
    async fn run_alignment(
        &self,
        audio: &AudioBuffer,
        transcript: &str,
        dictionary_override: Option<PathBuf>,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        if transcript.split_whitespace().next().is_none() {
            return Err(RecognitionError::InvalidInput {
                message: "Cannot align an empty transcript".to_string(),
            });
        }

        let cli = self.ensure_loaded().await?;
        let start_time = Instant::now();

        let language = config.map_or(LanguageCode::EnUs, |c| c.language);
        if !self.supported_languages.contains(&language) {
            return Err(RecognitionError::FeatureNotSupported {
                feature: format!(
                    "Language {language:?} with MFA assets '{}' / '{}'",
                    self.config.model, self.config.dictionary
                ),
            });
        }

        let processed = self.prepare_audio_for_mfa(audio)?;
        let total_duration = duration_seconds(&processed);

        let workspace = tempfile::Builder::new()
            .prefix("voirs-mfa-")
            .tempdir()
            .map_err(|e| RecognitionError::PhonemeRecognitionError {
                message: format!("Failed to create MFA workspace: {e}"),
                source: Some(Box::new(e)),
            })?;
        let corpus_dir = workspace.path().join("corpus");
        let output_dir = workspace.path().join("aligned");
        std::fs::create_dir_all(&corpus_dir).map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Failed to create MFA corpus directory: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let utterance = "utterance";
        write_wav(&corpus_dir.join(format!("{utterance}.wav")), &processed)?;
        std::fs::write(
            corpus_dir.join(format!("{utterance}.lab")),
            transcript.trim(),
        )
        .map_err(|e| RecognitionError::PhonemeRecognitionError {
            message: format!("Failed to write MFA transcript file: {e}"),
            source: Some(Box::new(e)),
        })?;

        let dictionary_arg = dictionary_override
            .as_ref()
            .map_or_else(|| OsString::from(&self.config.dictionary), OsString::from);
        let acoustic_arg =
            self.acoustic_model_arg()
                .ok_or_else(|| RecognitionError::ConfigurationError {
                    message: "No MFA acoustic model configured".to_string(),
                })?;

        let num_jobs = self.config.num_jobs;
        let beam_width = self.config.beam_width;
        let retry_beam = self.config.retry_beam;
        let cleanup = self.config.cleanup;
        let corpus_for_task = corpus_dir.clone();
        let output_for_task = output_dir.clone();

        tokio::task::spawn_blocking(move || {
            cli.align(&AlignRequest {
                corpus_dir: &corpus_for_task,
                dictionary: &dictionary_arg,
                acoustic_model: &acoustic_arg,
                output_dir: &output_for_task,
                num_jobs,
                beam_width,
                retry_beam,
                cleanup,
            })
        })
        .await
        .map_err(|e| RecognitionError::PhonemeRecognitionError {
            message: format!("MFA alignment task failed: {e}"),
            source: Some(Box::new(e)),
        })??;

        let grid_text = read_textgrid(&output_dir, utterance)?;
        let grid =
            TextGrid::parse(&grid_text).map_err(|e| RecognitionError::PhonemeRecognitionError {
                message: format!("Failed to parse the TextGrid MFA produced: {e}"),
                source: Some(Box::new(e)),
            })?;

        let alignment = self.alignment_from_grid(&grid, &processed, total_duration)?;

        let mut state = self.state.write().await;
        state.alignment_count += 1;
        state.total_alignment_time += start_time.elapsed();

        Ok(alignment)
    }

    /// Convert MFA's real tiers into VoiRS alignment types.
    fn alignment_from_grid(
        &self,
        grid: &TextGrid,
        audio: &AudioBuffer,
        total_duration: f32,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        let phone_tier = grid
            .tier_any(&["phones", "phone", "phonemes"])
            .ok_or_else(|| RecognitionError::PhonemeRecognitionError {
                message: format!(
                    "MFA's TextGrid has no phone tier (found: {})",
                    tier_names(grid)
                ),
                source: None,
            })?;

        let energy = FrameEnergy::analyse(audio);

        let phonemes: Vec<AlignedPhoneme> = phone_tier
            .intervals
            .iter()
            .filter(|interval| !interval.is_silence())
            .map(|interval| aligned_phoneme(interval, &energy))
            .collect();

        let word_alignments = if self.config.include_word_alignment {
            grid.tier_any(&["words", "word"])
                .map(|tier| word_alignments(tier, &phonemes, &energy))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let alignment_confidence = mean_confidence(&phonemes);
        let phonemes = if self.config.include_phone_alignment {
            phonemes
        } else {
            Vec::new()
        };

        Ok(PhonemeAlignment {
            phonemes,
            total_duration,
            alignment_confidence,
            word_alignments,
        })
    }

    /// Resample and downmix the audio into the 16 kHz mono form MFA expects.
    fn prepare_audio_for_mfa(&self, audio: &AudioBuffer) -> Result<AudioBuffer, RecognitionError> {
        super::super::asr::utils::preprocess_audio(audio).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("Failed to preprocess audio for MFA: {e}"),
                source: Some(Box::new(e)),
            }
        })
    }

    /// Get model statistics.
    pub async fn get_stats(&self) -> MFAStats {
        let state = self.state.read().await;
        MFAStats {
            alignment_count: state.alignment_count,
            total_alignment_time: state.total_alignment_time,
            average_alignment_time: if state.alignment_count > 0 {
                state.total_alignment_time
                    / u32::try_from(state.alignment_count).unwrap_or(u32::MAX)
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            model: state.model.clone(),
            dictionary: state.dictionary.clone(),
            available_models: state.available_models.len(),
            available_dictionaries: state.available_dictionaries.len(),
            mfa_version: state.cli.as_ref().map(|cli| cli.version().to_string()),
        }
    }

    /// List the acoustic models MFA really reports as installed.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when MFA is not installed.
    pub async fn list_available_models(&self) -> Result<Vec<MFAModelInfo>, RecognitionError> {
        self.ensure_loaded().await?;
        let state = self.state.read().await;
        Ok(state
            .available_models
            .iter()
            .map(|name| MFAModelInfo {
                language: Self::infer_language(name),
                name: name.clone(),
            })
            .collect())
    }

    /// List the dictionaries MFA really reports as installed.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when MFA is not installed.
    pub async fn list_available_dictionaries(
        &self,
    ) -> Result<Vec<MFADictionaryInfo>, RecognitionError> {
        self.ensure_loaded().await?;
        let state = self.state.read().await;
        Ok(state
            .available_dictionaries
            .iter()
            .map(|name| MFADictionaryInfo {
                language: Self::infer_language(name),
                name: name.clone(),
            })
            .collect())
    }

    /// Train a real acoustic model with `mfa train`.
    ///
    /// `training_data_path` must be a real MFA corpus directory (paired audio and
    /// transcript files); `output_path` is where the trained `.zip` model is written.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when MFA is missing, the corpus does
    /// not exist, or training exits with a failure status.
    pub async fn train_custom_model(
        &self,
        training_data_path: &str,
        output_path: &str,
    ) -> Result<(), RecognitionError> {
        let corpus = PathBuf::from(training_data_path);
        if !corpus.is_dir() {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "MFA training corpus not found at {}: expected a directory of paired audio \
                     and transcript files",
                    corpus.display()
                ),
                source: None,
            });
        }

        let cli = self.ensure_loaded().await?;
        let dictionary = OsString::from(&self.config.dictionary);
        let output = PathBuf::from(output_path);
        let num_jobs = self.config.num_jobs;

        tracing::info!("Training MFA acoustic model from {}", corpus.display());
        tokio::task::spawn_blocking(move || cli.train(&corpus, &dictionary, &output, num_jobs))
            .await
            .map_err(|e| RecognitionError::ModelLoadError {
                message: format!("MFA training task failed: {e}"),
                source: Some(Box::new(e)),
            })?
    }

    /// Parse a real MFA pronunciation dictionary from disk.
    ///
    /// Returns every entry the file really contains, keyed by the upper-cased word.
    /// Words with several pronunciations keep the first one listed.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the file is missing, unreadable,
    /// or contains no usable entries.
    pub async fn validate_dictionary(
        &self,
        dictionary_path: &str,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        let path = PathBuf::from(dictionary_path);
        tokio::task::spawn_blocking(move || parse_dictionary_file(&path))
            .await
            .map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Dictionary validation task failed: {e}"),
                source: Some(Box::new(e)),
            })?
    }
}

struct Probe {
    cli: MfaCli,
    acoustic: Vec<String>,
    dictionaries: Vec<String>,
}

/// MFA model statistics
#[derive(Debug, Clone)]
pub struct MFAStats {
    /// Total number of alignments performed
    pub alignment_count: usize,
    /// Total alignment time
    pub total_alignment_time: Duration,
    /// Average alignment time
    pub average_alignment_time: Duration,
    /// How long probing and validating the installation really took
    pub load_time: Option<Duration>,
    /// Current acoustic model identifier
    pub model: String,
    /// Current dictionary identifier
    pub dictionary: String,
    /// Number of acoustic models MFA reported
    pub available_models: usize,
    /// Number of dictionaries MFA reported
    pub available_dictionaries: usize,
    /// Version string the installed aligner reported, once probed
    pub mfa_version: Option<String>,
}

#[async_trait]
impl PhonemeRecognizer for MFAModel {
    /// Reference-free phoneme recognition is not something MFA can do.
    ///
    /// MFA is a *forced* aligner: it needs a transcript. Rather than invent a phoneme
    /// sequence, this method fails closed. Use [`PhonemeRecognizer::align_text`] with the
    /// spoken text, or run an ASR model first and align its transcript.
    async fn recognize_phonemes(
        &self,
        _audio: &AudioBuffer,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<Phoneme>> {
        Err(RecognitionError::FeatureNotSupported {
            feature: "reference-free phoneme recognition with MFA: the Montreal Forced Aligner \
                      requires a transcript. Call align_text() with the spoken text, or \
                      transcribe with an ASR model first."
                .to_string(),
        }
        .into())
    }

    /// Align a known phoneme sequence by driving MFA with a generated dictionary in
    /// which every phoneme symbol is its own single-phone "word".
    async fn align_phonemes(
        &self,
        audio: &AudioBuffer,
        expected: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        if expected.is_empty() {
            return Err(RecognitionError::InvalidInput {
                message: "Cannot align an empty phoneme sequence".to_string(),
            }
            .into());
        }

        let symbols: Vec<String> = expected
            .iter()
            .map(|phoneme| sanitise_symbol(&phoneme.symbol))
            .collect();
        if symbols.iter().any(String::is_empty) {
            return Err(RecognitionError::InvalidInput {
                message: "Phoneme symbols must contain at least one non-whitespace character"
                    .to_string(),
            }
            .into());
        }

        // A phone-as-word dictionary lets the real aligner place each expected phone.
        let dictionary_dir = tempfile::Builder::new()
            .prefix("voirs-mfa-dict-")
            .tempdir()
            .map_err(|e| RecognitionError::PhonemeRecognitionError {
                message: format!("Failed to create MFA dictionary workspace: {e}"),
                source: Some(Box::new(e)),
            })?;
        let dictionary_path = dictionary_dir.path().join("phones.dict");
        let mut seen: Vec<&String> = Vec::new();
        let mut body = String::new();
        for symbol in &symbols {
            if !seen.contains(&symbol) {
                seen.push(symbol);
                body.push_str(symbol);
                body.push('\t');
                body.push_str(symbol);
                body.push('\n');
            }
        }
        std::fs::write(&dictionary_path, body).map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Failed to write MFA phone dictionary: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let transcript = symbols.join(" ");
        let alignment = self
            .run_alignment(audio, &transcript, Some(dictionary_path), config)
            .await?;
        Ok(alignment)
    }

    async fn align_text(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        self.run_alignment(audio, text, None, config)
            .await
            .map_err(Into::into)
    }

    fn metadata(&self) -> PhonemeRecognizerMetadata {
        self.metadata.clone()
    }

    fn supports_feature(&self, feature: PhonemeRecognitionFeature) -> bool {
        self.metadata.supported_features.contains(&feature)
    }
}

impl Clone for MFAModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            supported_languages: self.supported_languages.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Per-frame RMS energy of the aligned audio, used to score how plausible each interval
/// is against the signal it claims to cover.
struct FrameEnergy {
    /// RMS per fixed-length frame.
    frames: Vec<f32>,
    /// Seconds covered by one frame.
    seconds_per_frame: f32,
    /// Loudest frame in the clip, used to normalise.
    peak: f32,
    /// Quietest frame in the clip, used as the noise floor.
    floor: f32,
}

impl FrameEnergy {
    fn analyse(audio: &AudioBuffer) -> Self {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate().max(1);
        #[allow(clippy::cast_precision_loss)]
        let seconds_per_frame = SCORE_FRAME_SAMPLES as f32 / sample_rate as f32;

        let frames: Vec<f32> = samples
            .chunks(SCORE_FRAME_SAMPLES)
            .map(|chunk| {
                #[allow(clippy::cast_precision_loss)]
                let n = chunk.len().max(1) as f32;
                (chunk.iter().map(|s| s * s).sum::<f32>() / n).sqrt()
            })
            .collect();

        let peak = frames.iter().copied().fold(0.0_f32, f32::max);
        let floor = frames.iter().copied().fold(f32::INFINITY, f32::min);
        let floor = if floor.is_finite() { floor } else { 0.0 };

        Self {
            frames,
            seconds_per_frame,
            peak,
            floor,
        }
    }

    /// Fraction of the clip's dynamic range occupied by `[start, end)`, in `[0, 1]`.
    ///
    /// A value near 1 means the span really is loud relative to the rest of the clip;
    /// near 0 means it is at the noise floor.
    fn activity(&self, start: f32, end: f32) -> Option<f32> {
        if self.frames.is_empty() || self.seconds_per_frame <= 0.0 {
            return None;
        }
        let span = self.peak - self.floor;
        if span <= f32::EPSILON {
            return None;
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (start.max(0.0) / self.seconds_per_frame).floor() as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let last = (end.max(0.0) / self.seconds_per_frame).ceil() as usize;
        let last = last.min(self.frames.len());
        if first >= last {
            return None;
        }

        let slice = &self.frames[first..last];
        #[allow(clippy::cast_precision_loss)]
        let mean = slice.iter().sum::<f32>() / slice.len() as f32;
        Some(((mean - self.floor) / span).clamp(0.0, 1.0))
    }

    /// How well a label of the given kind agrees with the audio under it.
    ///
    /// Speech labels score high where the signal is energetic; silence labels score high
    /// where it is not. Returns `None` when the clip carries no usable dynamic range, so
    /// callers can report "not measured" rather than a made-up number.
    fn agreement(&self, label: &str, start: f32, end: f32) -> Option<f32> {
        let activity = self.activity(start, end)?;
        let score = if is_silence_label(label) {
            1.0 - activity
        } else {
            activity
        };
        Some(score.clamp(0.0, 1.0))
    }
}

fn aligned_phoneme(interval: &Interval, energy: &FrameEnergy) -> AlignedPhoneme {
    #[allow(clippy::cast_possible_truncation)]
    let start_time = interval.xmin as f32;
    #[allow(clippy::cast_possible_truncation)]
    let end_time = interval.xmax as f32;
    #[allow(clippy::cast_possible_truncation)]
    let duration_ms = (interval.duration() * 1000.0) as f32;

    // MFA's TextGrid carries no posterior, so confidence is measured from the audio:
    // it is the agreement between the label's kind and the energy under the interval.
    // `None` (no usable dynamic range) is reported as 0.0 == not measured.
    let confidence = energy
        .agreement(&interval.text, start_time, end_time)
        .unwrap_or(0.0);

    AlignedPhoneme {
        phoneme: Phoneme {
            symbol: interval.text.clone(),
            ipa_symbol: interval.text.clone(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
            duration_ms: Some(duration_ms),
            confidence,
        },
        start_time,
        end_time,
        confidence,
    }
}

fn word_alignments(
    tier: &Tier,
    phonemes: &[AlignedPhoneme],
    energy: &FrameEnergy,
) -> Vec<WordAlignment> {
    tier.intervals
        .iter()
        .filter(|interval| !interval.is_silence())
        .map(|interval| {
            #[allow(clippy::cast_possible_truncation)]
            let start_time = interval.xmin as f32;
            #[allow(clippy::cast_possible_truncation)]
            let end_time = interval.xmax as f32;

            // A phone belongs to the word whose span contains its midpoint.
            let contained: Vec<AlignedPhoneme> = phonemes
                .iter()
                .filter(|phoneme| {
                    let midpoint = (phoneme.start_time + phoneme.end_time) / 2.0;
                    midpoint >= start_time && midpoint < end_time
                })
                .cloned()
                .collect();

            let confidence = if contained.is_empty() {
                energy
                    .agreement(&interval.text, start_time, end_time)
                    .unwrap_or(0.0)
            } else {
                mean_confidence(&contained)
            };

            WordAlignment {
                word: interval.text.clone(),
                start_time,
                end_time,
                phonemes: contained,
                confidence,
            }
        })
        .collect()
}

fn mean_confidence(phonemes: &[AlignedPhoneme]) -> f32 {
    if phonemes.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let count = phonemes.len() as f32;
    phonemes.iter().map(|p| p.confidence).sum::<f32>() / count
}

fn tier_names(grid: &TextGrid) -> String {
    if grid.tiers.is_empty() {
        return "none".to_string();
    }
    grid.tiers
        .iter()
        .map(|tier| tier.name.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

fn duration_seconds(audio: &AudioBuffer) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let samples = audio.samples().len() as f32;
    #[allow(clippy::cast_precision_loss)]
    let rate = audio.sample_rate().max(1) as f32;
    samples / rate
}

fn is_existing_path(value: &OsString) -> bool {
    Path::new(value).exists()
}

/// Strip whitespace from a phoneme symbol so it can be used as a dictionary key.
fn sanitise_symbol(symbol: &str) -> String {
    symbol.split_whitespace().collect::<Vec<_>>().join("_")
}

/// Find the `TextGrid` MFA wrote for `utterance`, searching nested speaker directories.
fn read_textgrid(output_dir: &Path, utterance: &str) -> Result<String, RecognitionError> {
    let direct = output_dir.join(format!("{utterance}.TextGrid"));
    if direct.is_file() {
        return std::fs::read_to_string(&direct).map_err(|e| read_error(&direct, &e));
    }

    // MFA groups output by speaker; walk one level down looking for any TextGrid.
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(inner) = std::fs::read_dir(&path) {
                    for candidate in inner.flatten() {
                        let candidate = candidate.path();
                        if candidate.extension().is_some_and(|ext| ext == "TextGrid") {
                            return std::fs::read_to_string(&candidate)
                                .map_err(|e| read_error(&candidate, &e));
                        }
                    }
                }
            } else if path.extension().is_some_and(|ext| ext == "TextGrid") {
                return std::fs::read_to_string(&path).map_err(|e| read_error(&path, &e));
            }
        }
    }

    Err(RecognitionError::PhonemeRecognitionError {
        message: format!(
            "MFA reported success but wrote no TextGrid under {}. The utterance was most likely \
             skipped because its transcript contains out-of-vocabulary words.",
            output_dir.display()
        ),
        source: None,
    })
}

fn read_error(path: &Path, error: &std::io::Error) -> RecognitionError {
    RecognitionError::PhonemeRecognitionError {
        message: format!("Failed to read {}: {error}", path.display()),
        source: None,
    }
}

/// Write a 16-bit PCM WAV file, the input format MFA expects.
fn write_wav(path: &Path, audio: &AudioBuffer) -> Result<(), RecognitionError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| write_error(path, &e.to_string()))?;
    for sample in audio.samples() {
        #[allow(clippy::cast_possible_truncation)]
        let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        writer
            .write_sample(value)
            .map_err(|e| write_error(path, &e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| write_error(path, &e.to_string()))
}

fn write_error(path: &Path, detail: &str) -> RecognitionError {
    RecognitionError::PhonemeRecognitionError {
        message: format!("Failed to write {}: {detail}", path.display()),
        source: None,
    }
}

/// Parse a real MFA pronunciation dictionary.
///
/// Each line is `word<TAB or spaces>phone phone ...`; comment lines starting with `#`
/// and blank lines are skipped. Optional probability columns (a bare float immediately
/// after the word, as MFA 2.x writes) are dropped.
fn parse_dictionary_file(path: &Path) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
    if !path.is_file() {
        return Err(RecognitionError::ModelLoadError {
            message: format!("Dictionary file not found: {}", path.display()),
            source: None,
        });
    }

    let text = std::fs::read_to_string(path).map_err(|e| RecognitionError::ModelLoadError {
        message: format!("Failed to read dictionary {}: {e}", path.display()),
        source: Some(Box::new(e)),
    })?;

    let mut entries: HashMap<String, Vec<String>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(word) = fields.next() else {
            continue;
        };
        let mut phones: Vec<String> = fields.map(str::to_string).collect();
        // MFA 2.x may write up to four numeric probability columns before the phones.
        while phones
            .first()
            .is_some_and(|field| field.parse::<f64>().is_ok())
        {
            phones.remove(0);
        }
        if phones.is_empty() {
            continue;
        }
        entries.entry(word.to_uppercase()).or_insert(phones);
    }

    if entries.is_empty() {
        return Err(RecognitionError::ModelLoadError {
            message: format!(
                "Dictionary {} contains no usable `word phone...` entries",
                path.display()
            ),
            source: None,
        });
    }

    Ok(entries)
}

fn missing_asset_error(kind: &str, name: &str, available: &[String]) -> RecognitionError {
    let listing = if available.is_empty() {
        "none are installed".to_string()
    } else {
        format!("installed: {}", available.join(", "))
    };
    RecognitionError::ModelLoadError {
        message: format!(
            "MFA {kind} '{name}' is not available ({listing}). Install it with \
             `mfa model download {} {name}`, pass a filesystem path instead, or set \
             MFAConfig::auto_download.",
            if kind.starts_with("acoustic") {
                "acoustic"
            } else {
                "dictionary"
            }
        ),
        source: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use voirs_sdk::AudioBuffer;

    /// A configuration pointing at an executable name that cannot exist, so that every
    /// test exercises the fail-closed path without needing MFA installed.
    fn offline_config() -> MFAConfig {
        MFAConfig {
            executable: "voirs-nonexistent-mfa-binary".to_string(),
            ..Default::default()
        }
    }

    async fn offline_model() -> MFAModel {
        MFAModel::with_config(offline_config())
            .await
            .expect("constructing a handle must not touch the filesystem")
    }

    /// A two-tier grid: one spoken word covering two phones, then silence.
    const TWO_TIER_GRID: &str = "Object class = \"TextGrid\"\n\
             xmin = 0\nxmax = 1.0\nsize = 2\n\
             class = \"IntervalTier\"\nname = \"words\"\nxmin = 0\nxmax = 1.0\n\
             intervals: size = 2\n\
             xmin = 0\nxmax = 0.5\ntext = \"hi\"\n\
             xmin = 0.5\nxmax = 1.0\ntext = \"\"\n\
             class = \"IntervalTier\"\nname = \"phones\"\nxmin = 0\nxmax = 1.0\n\
             intervals: size = 3\n\
             xmin = 0\nxmax = 0.25\ntext = \"HH\"\n\
             xmin = 0.25\nxmax = 0.5\ntext = \"AY1\"\n\
             xmin = 0.5\nxmax = 1.0\ntext = \"sil\"\n";

    fn blank_metadata() -> PhonemeRecognizerMetadata {
        PhonemeRecognizerMetadata {
            name: String::new(),
            version: String::new(),
            description: String::new(),
            supported_languages: vec![],
            alignment_methods: vec![],
            alignment_accuracy: 0.0,
            supported_features: vec![],
        }
    }

    fn speech_then_silence() -> AudioBuffer {
        // 0.5 s of a loud tone followed by 0.5 s of near-silence, at 16 kHz.
        let mut samples = Vec::with_capacity(16000);
        for i in 0..8000 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 16000.0;
            samples.push((t * 220.0 * std::f32::consts::TAU).sin() * 0.8);
        }
        samples.extend(std::iter::repeat_n(0.0, 8000));
        AudioBuffer::new(samples, 16000, 1)
    }

    #[tokio::test]
    async fn model_creation_does_not_require_mfa() {
        let model = offline_model().await;
        assert!(model.metadata.name.contains("Montreal Forced Alignment"));
        assert!(model.supported_languages.contains(&LanguageCode::EnUs));
        // No accuracy is asserted, because none has been measured.
        assert_eq!(model.metadata.alignment_accuracy, 0.0);
    }

    #[tokio::test]
    async fn align_text_fails_closed_when_mfa_is_absent() {
        let model = offline_model().await;
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let err = model
            .align_text(&audio, "hello world", None)
            .await
            .expect_err("must not fabricate an alignment without MFA installed");
        let rendered = err.to_string();
        assert!(
            rendered.contains("was not found on PATH"),
            "unexpected error: {rendered}"
        );
        assert!(
            rendered.contains("ForcedAlignModel"),
            "the error should point at the pure-Rust alternative: {rendered}"
        );
    }

    #[tokio::test]
    async fn align_phonemes_fails_closed_when_mfa_is_absent() {
        let model = offline_model().await;
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let expected = vec![Phoneme {
            symbol: "HH".to_string(),
            ipa_symbol: "HH".to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Onset,
            duration_ms: None,
            confidence: 1.0,
        }];

        let err = model
            .align_phonemes(&audio, &expected, None)
            .await
            .expect_err("must fail closed");
        assert!(err.to_string().contains("was not found on PATH"));
    }

    #[tokio::test]
    async fn empty_inputs_are_rejected_before_touching_mfa() {
        let model = offline_model().await;
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let err = model.align_text(&audio, "   ", None).await.unwrap_err();
        assert!(err.to_string().contains("empty transcript"), "{err}");

        let err = model.align_phonemes(&audio, &[], None).await.unwrap_err();
        assert!(err.to_string().contains("empty phoneme sequence"), "{err}");
    }

    #[tokio::test]
    async fn reference_free_recognition_is_refused() {
        let model = offline_model().await;
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let err = model
            .recognize_phonemes(&audio, None)
            .await
            .expect_err("MFA cannot recognise phonemes without a transcript");
        assert!(
            err.to_string().contains("requires a transcript"),
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn listing_assets_fails_closed_when_mfa_is_absent() {
        let model = offline_model().await;
        assert!(model.list_available_models().await.is_err());
        assert!(model.list_available_dictionaries().await.is_err());
    }

    #[tokio::test]
    async fn training_rejects_a_missing_corpus_before_probing() {
        let model = offline_model().await;
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("absent-corpus");

        let err = model
            .train_custom_model(
                missing.to_str().expect("utf-8 path"),
                dir.path().join("out.zip").to_str().expect("utf-8 path"),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("training corpus not found"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn statistics_start_empty_and_report_no_version() {
        let model = offline_model().await;
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 0);
        assert_eq!(stats.available_models, 0);
        assert!(stats.mfa_version.is_none());
        assert!(stats.load_time.is_none());
    }

    #[tokio::test]
    async fn language_support_follows_the_configured_assets() {
        let english = MFAModel::with_config(MFAConfig {
            model: "english_us_arpa".to_string(),
            dictionary: "english_us_arpa".to_string(),
            ..offline_config()
        })
        .await
        .expect("construct");
        assert!(english.supported_languages.contains(&LanguageCode::EnUs));

        let german = MFAModel::with_config(MFAConfig {
            model: "german_mfa".to_string(),
            dictionary: "german_mfa".to_string(),
            ..offline_config()
        })
        .await
        .expect("construct");
        assert!(german.supported_languages.contains(&LanguageCode::DeDe));
        assert!(!german.supported_languages.contains(&LanguageCode::EnUs));

        let french = MFAModel::with_config(MFAConfig {
            model: "french_mfa".to_string(),
            dictionary: "french_mfa".to_string(),
            ..offline_config()
        })
        .await
        .expect("construct");
        assert!(french.supported_languages.contains(&LanguageCode::FrFr));
    }

    #[tokio::test]
    async fn advertised_features_match_what_mfa_can_do() {
        let model = offline_model().await;
        assert!(model.supports_feature(PhonemeRecognitionFeature::WordAlignment));
        assert!(model.supports_feature(PhonemeRecognitionFeature::MultiLanguage));
        assert!(model.supports_feature(PhonemeRecognitionFeature::ConfidenceScoring));
        // Pronunciation assessment was advertised but never implemented.
        assert!(!model.supports_feature(PhonemeRecognitionFeature::PronunciationAssessment));
    }

    #[tokio::test]
    async fn dictionary_validation_parses_a_real_file() {
        let model = offline_model().await;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.dict");
        let mut file = std::fs::File::create(&path).expect("create");
        writeln!(file, "# a comment").expect("write");
        writeln!(file, "hello\tHH AH0 L OW1").expect("write");
        writeln!(file, "world  W ER1 L D").expect("write");
        writeln!(file, "probable 0.5 P R AA1 B").expect("write");
        writeln!(file).expect("write");
        drop(file);

        let entries = model
            .validate_dictionary(path.to_str().expect("utf-8 path"))
            .await
            .expect("real dictionary must parse");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries["HELLO"], vec!["HH", "AH0", "L", "OW1"]);
        assert_eq!(entries["WORLD"], vec!["W", "ER1", "L", "D"]);
        // The probability column is dropped, not treated as a phone.
        assert_eq!(entries["PROBABLE"], vec!["P", "R", "AA1", "B"]);
    }

    #[tokio::test]
    async fn dictionary_validation_rejects_missing_and_empty_files() {
        let model = offline_model().await;
        let dir = tempfile::tempdir().expect("tempdir");

        let missing = dir.path().join("absent.dict");
        let err = model
            .validate_dictionary(missing.to_str().expect("utf-8 path"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");

        let empty = dir.path().join("empty.dict");
        std::fs::write(&empty, "# only comments\n\n").expect("write");
        let err = model
            .validate_dictionary(empty.to_str().expect("utf-8 path"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no usable"), "{err}");
    }

    #[test]
    fn frame_energy_tracks_the_real_signal() {
        let audio = speech_then_silence();
        let energy = FrameEnergy::analyse(&audio);

        let loud = energy.activity(0.0, 0.5).expect("dynamic range exists");
        let quiet = energy.activity(0.5, 1.0).expect("dynamic range exists");
        assert!(
            loud > quiet + 0.5,
            "the tone half must be far more active than the silent half: {loud} vs {quiet}"
        );

        // Agreement flips with the label kind, using the same measurement.
        let speech_on_speech = energy.agreement("HH", 0.0, 0.5).expect("measured");
        let silence_on_speech = energy.agreement("sil", 0.0, 0.5).expect("measured");
        assert!((speech_on_speech + silence_on_speech - 1.0).abs() < 1e-5);
        assert!(speech_on_speech > 0.8, "{speech_on_speech}");

        let silence_on_silence = energy.agreement("", 0.5, 1.0).expect("measured");
        assert!(silence_on_silence > 0.8, "{silence_on_silence}");
    }

    #[test]
    fn flat_audio_reports_confidence_as_not_measured() {
        // A constant signal has no dynamic range, so no claim can be made.
        let audio = AudioBuffer::new(vec![0.25; 16000], 16000, 1);
        let energy = FrameEnergy::analyse(&audio);
        assert!(energy.activity(0.0, 1.0).is_none());
        assert!(energy.agreement("HH", 0.0, 1.0).is_none());
    }

    #[test]
    fn alignment_from_grid_uses_the_real_intervals() {
        let audio = speech_then_silence();
        let grid = TextGrid::parse(
            "Object class = \"TextGrid\"\n\
             xmin = 0\nxmax = 1.0\nsize = 2\n\
             class = \"IntervalTier\"\nname = \"words\"\nxmin = 0\nxmax = 1.0\n\
             intervals: size = 2\n\
             xmin = 0\nxmax = 0.5\ntext = \"hi\"\n\
             xmin = 0.5\nxmax = 1.0\ntext = \"\"\n\
             class = \"IntervalTier\"\nname = \"phones\"\nxmin = 0\nxmax = 1.0\n\
             intervals: size = 3\n\
             xmin = 0\nxmax = 0.25\ntext = \"HH\"\n\
             xmin = 0.25\nxmax = 0.5\ntext = \"AY1\"\n\
             xmin = 0.5\nxmax = 1.0\ntext = \"sil\"\n",
        )
        .expect("fixture parses");

        let config = MFAConfig::default();
        let model = MFAModel {
            config: config.clone(),
            state: Arc::new(RwLock::new(MFAState::new(config.model, config.dictionary))),
            supported_languages: vec![LanguageCode::EnUs],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        let alignment = model
            .alignment_from_grid(&grid, &audio, 1.0)
            .expect("conversion succeeds");

        // Silence intervals are dropped; the real labels and times survive.
        assert_eq!(alignment.phonemes.len(), 2);
        assert_eq!(alignment.phonemes[0].phoneme.symbol, "HH");
        assert!((alignment.phonemes[0].start_time - 0.0).abs() < 1e-6);
        assert!((alignment.phonemes[1].end_time - 0.5).abs() < 1e-6);
        assert!((alignment.phonemes[0].phoneme.duration_ms.unwrap_or(0.0) - 250.0).abs() < 1e-3);

        // Both phones fall inside the single spoken word.
        assert_eq!(alignment.word_alignments.len(), 1);
        assert_eq!(alignment.word_alignments[0].word, "hi");
        assert_eq!(alignment.word_alignments[0].phonemes.len(), 2);

        // Confidence is measured from the audio, and this fixture places both phones on
        // the energetic half of the clip.
        assert!(
            alignment.alignment_confidence > 0.5,
            "confidence should reflect real energy agreement: {}",
            alignment.alignment_confidence
        );
    }

    #[test]
    fn alignment_confidence_varies_with_the_audio() {
        let grid_text = "Object class = \"TextGrid\"\n\
             xmin = 0\nxmax = 1.0\nsize = 1\n\
             class = \"IntervalTier\"\nname = \"phones\"\nxmin = 0\nxmax = 1.0\n\
             intervals: size = 1\n\
             xmin = 0\nxmax = 0.5\ntext = \"HH\"\n";
        let grid = TextGrid::parse(grid_text).expect("fixture parses");

        let config = MFAConfig::default();
        let make_model = || MFAModel {
            config: config.clone(),
            state: Arc::new(RwLock::new(MFAState::new(
                config.model.clone(),
                config.dictionary.clone(),
            ))),
            supported_languages: vec![LanguageCode::EnUs],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        // Energy in the first half: the "HH" label agrees with the signal.
        let matching = make_model()
            .alignment_from_grid(&grid, &speech_then_silence(), 1.0)
            .expect("convert");

        // The same grid over audio whose energy is in the *second* half disagrees.
        let mut flipped: Vec<f32> = vec![0.0; 8000];
        for i in 0..8000 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / 16000.0;
            flipped.push((t * 220.0 * std::f32::consts::TAU).sin() * 0.8);
        }
        let mismatching = make_model()
            .alignment_from_grid(&grid, &AudioBuffer::new(flipped, 16000, 1), 1.0)
            .expect("convert");

        assert!(
            matching.alignment_confidence > mismatching.alignment_confidence + 0.4,
            "confidence must respond to the real audio: {} vs {}",
            matching.alignment_confidence,
            mismatching.alignment_confidence
        );
    }

    /// Both inclusion flags must really change the returned result, and turning phone
    /// alignment off must not silently strip the phones nested inside each word.
    #[test]
    fn inclusion_flags_really_shape_the_result() {
        let audio = speech_then_silence();
        let grid = TextGrid::parse(TWO_TIER_GRID).expect("fixture parses");

        let build = |include_phone_alignment: bool, include_word_alignment: bool| {
            let config = MFAConfig {
                include_phone_alignment,
                include_word_alignment,
                ..Default::default()
            };
            let model = MFAModel {
                state: Arc::new(RwLock::new(MFAState::new(
                    config.model.clone(),
                    config.dictionary.clone(),
                ))),
                config,
                supported_languages: vec![LanguageCode::EnUs],
                metadata: blank_metadata(),
            };
            model
                .alignment_from_grid(&grid, &audio, 1.0)
                .expect("conversion succeeds")
        };

        let both = build(true, true);
        assert_eq!(both.phonemes.len(), 2);
        assert_eq!(both.word_alignments.len(), 1);
        assert_eq!(both.word_alignments[0].phonemes.len(), 2);

        let words_only = build(false, true);
        assert!(words_only.phonemes.is_empty(), "phone list must be dropped");
        assert_eq!(words_only.word_alignments.len(), 1);
        assert_eq!(
            words_only.word_alignments[0].phonemes.len(),
            2,
            "phones nested in words must survive `include_phone_alignment: false`"
        );
        // The confidence is still measured from every phone, not from the empty list.
        assert!((words_only.alignment_confidence - both.alignment_confidence).abs() < 1e-6);

        let phones_only = build(true, false);
        assert_eq!(phones_only.phonemes.len(), 2);
        assert!(phones_only.word_alignments.is_empty());

        let neither = build(false, false);
        assert!(neither.phonemes.is_empty());
        assert!(neither.word_alignments.is_empty());
    }

    #[test]
    fn missing_phone_tier_is_reported() {
        let grid = TextGrid::parse(
            "Object class = \"TextGrid\"\n0\n1\n1\n\"IntervalTier\"\n\"words\"\n0\n1\n1\n\
             0\n1\n\"hi\"\n",
        )
        .expect("parse");

        let config = MFAConfig::default();
        let model = MFAModel {
            config: config.clone(),
            state: Arc::new(RwLock::new(MFAState::new(config.model, config.dictionary))),
            supported_languages: vec![LanguageCode::EnUs],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let err = model.alignment_from_grid(&grid, &audio, 1.0).unwrap_err();
        assert!(err.to_string().contains("no phone tier"), "{err}");
    }

    #[test]
    fn wav_round_trips_through_hound() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("utterance.wav");
        let audio = AudioBuffer::new(vec![0.0, 0.5, -0.5, 1.0], 16000, 1);
        write_wav(&path, &audio).expect("write");

        let mut reader = hound::WavReader::open(&path).expect("read back");
        assert_eq!(reader.spec().sample_rate, 16000);
        assert_eq!(reader.spec().channels, 1);
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], 0);
        assert!(
            samples[3] > 32_000,
            "full-scale sample survived: {samples:?}"
        );
    }

    #[test]
    fn symbols_are_sanitised_for_dictionary_use() {
        assert_eq!(sanitise_symbol("HH"), "HH");
        assert_eq!(sanitise_symbol(" AH 0 "), "AH_0");
        assert_eq!(sanitise_symbol("   "), "");
    }

    #[test]
    fn language_inference_reports_none_when_unknown() {
        assert_eq!(
            MFAModel::infer_language("english_us_arpa"),
            Some(LanguageCode::EnUs)
        );
        assert_eq!(
            MFAModel::infer_language("german_mfa"),
            Some(LanguageCode::DeDe)
        );
        assert_eq!(MFAModel::infer_language("my_custom_model"), None);
    }

    #[test]
    fn missing_textgrid_is_reported_rather_than_invented() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = read_textgrid(dir.path(), "utterance").unwrap_err();
        assert!(err.to_string().contains("wrote no TextGrid"), "{err}");
    }

    #[test]
    fn textgrid_is_found_in_speaker_subdirectories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let speaker = dir.path().join("speaker-1");
        std::fs::create_dir_all(&speaker).expect("mkdir");
        std::fs::write(speaker.join("utterance.TextGrid"), "grid contents").expect("write");

        let found = read_textgrid(dir.path(), "utterance").expect("found nested grid");
        assert_eq!(found, "grid contents");
    }
}
