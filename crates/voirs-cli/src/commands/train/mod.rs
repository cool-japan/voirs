//! Training command implementations
//!
//! This module provides CLI commands for training VoiRS models:
//! - Vocoder training (HiFi-GAN, DiffWave)
//! - Acoustic model training (VITS, FastSpeech2)
//! - G2P model training
//! - Training progress monitoring and visualization

pub mod acoustic;
pub mod data_loader;
pub mod g2p;
pub mod progress;
pub mod vocoder;

use crate::error::{CliError, Result};
use crate::GlobalOptions;
use clap::Subcommand;
use std::path::{Path, PathBuf};

/// Learning rate scheduler names accepted by `--lr-scheduler`. Single source of
/// truth shared between the `--help` text below and the up-front validation in
/// [`resolve_training_config`]: an unrecognized name is a hard configuration
/// error (fail closed) rather than a silent fallback to `"none"`.
pub(crate) const VALID_LR_SCHEDULERS: &[&str] = &[
    "none",
    "step",
    "cosine",
    "exponential",
    "onecycle",
    "plateau",
];

/// Training subcommands
#[derive(Debug, Clone, Subcommand)]
pub enum TrainCommands {
    /// Train vocoder model (HiFi-GAN, DiffWave)
    Vocoder {
        /// Model type (hifigan, diffwave)
        #[arg(long, default_value = "diffwave")]
        model_type: String,

        /// Training data directory
        #[arg(long)]
        data: PathBuf,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints/vocoder")]
        output: PathBuf,

        /// Training config file (TOML/JSON). Supplies defaults for the
        /// scheduler/early-stopping/checkpoint-cadence flags below; any of those
        /// flags given explicitly on the command line always takes precedence
        /// over the value in this file. Core run parameters (--epochs,
        /// --batch-size, --lr, --data, --output, --resume, --gpu) are always
        /// CLI-only and are not read from this file.
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Number of epochs
        #[arg(long, default_value = "1000")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "16")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "0.0002")]
        lr: f64,

        /// Learning rate scheduler: none, step, cosine, exponential, onecycle,
        /// plateau (default: none, or the config file's value if --config is
        /// given and omits this flag)
        #[arg(long)]
        lr_scheduler: Option<String>,

        /// LR scheduler step size in epochs: the decay interval for `step`, or
        /// the number of epochs without validation improvement before `plateau`
        /// applies one decay (default: 100)
        #[arg(long)]
        lr_step_size: Option<usize>,

        /// LR scheduler decay factor, applied per interval for `step`,
        /// `exponential`, and `plateau` (default: 0.1)
        #[arg(long)]
        lr_gamma: Option<f64>,

        /// Enable early stopping (also enabled if the config file sets
        /// early-stopping = true; this flag can only turn it on, never off)
        #[arg(long)]
        early_stopping: bool,

        /// Early stopping patience, in epochs without validation improvement
        /// (default: 50)
        #[arg(long)]
        patience: Option<usize>,

        /// Minimum validation-loss improvement to reset patience / save a new
        /// best checkpoint (default: 0.0001)
        #[arg(long)]
        min_delta: Option<f64>,

        /// Validation frequency, in epochs (default: 5)
        #[arg(long)]
        val_frequency: Option<usize>,

        /// Linear LR warmup over the first N optimizer steps (default: 0,
        /// disabled)
        #[arg(long)]
        warmup_steps: Option<usize>,

        /// Gradient clipping max global L2 norm, 0 disables clipping (default:
        /// 1.0)
        #[arg(long)]
        grad_clip: Option<f64>,

        /// Save checkpoint every N epochs (default: 10)
        #[arg(long)]
        save_frequency: Option<usize>,

        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<PathBuf>,

        /// Use GPU if available
        #[arg(long)]
        gpu: bool,
    },

    /// Train acoustic model (VITS, FastSpeech2) [currently fails closed: the
    /// underlying trainers do not yet perform real gradient-based training, see
    /// the error message for details]
    Acoustic {
        /// Model type (vits, fastspeech2)
        #[arg(long, default_value = "vits")]
        model_type: String,

        /// Training data directory
        #[arg(long)]
        data: PathBuf,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints/acoustic")]
        output: PathBuf,

        /// Training config file (TOML/JSON)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Number of epochs
        #[arg(long, default_value = "500")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        lr: f64,

        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<PathBuf>,

        /// Use GPU if available
        #[arg(long)]
        gpu: bool,
    },

    /// Train G2P model [currently fails closed: the underlying LstmTrainer does
    /// not yet perform real gradient-based training, see the error message for
    /// details]
    G2p {
        /// Language code (en, ja, etc.)
        #[arg(long, default_value = "en")]
        language: String,

        /// Dictionary file (pronunciation dictionary)
        #[arg(long)]
        dictionary: PathBuf,

        /// Output model path
        #[arg(short, long, default_value = "models/g2p.safetensors")]
        output: PathBuf,

        /// Training config file (TOML/JSON)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Number of epochs
        #[arg(long, default_value = "100")]
        epochs: usize,

        /// Learning rate
        #[arg(long, default_value = "0.001")]
        lr: f64,
    },
}

/// Training configuration for advanced options
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub lr_scheduler: String,
    pub lr_step_size: usize,
    pub lr_gamma: f64,
    pub early_stopping: bool,
    pub patience: usize,
    pub min_delta: f64,
    pub val_frequency: usize,
    pub warmup_steps: usize,
    pub grad_clip: f64,
    pub save_frequency: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr_scheduler: "none".to_string(),
            lr_step_size: 100,
            lr_gamma: 0.1,
            early_stopping: false,
            patience: 50,
            min_delta: 0.0001,
            val_frequency: 5,
            warmup_steps: 0,
            grad_clip: 1.0,
            save_frequency: 10,
        }
    }
}

/// Optional on-disk overrides for [`TrainingConfig`], loaded from the file
/// passed via `--config`. Every field is optional so a config file only needs
/// to specify the hyperparameters it actually wants to change; fields left
/// unset fall through to whatever the CLI didn't already provide, and finally
/// to `TrainingConfig::default()`. Deliberately does not cover `epochs`,
/// `batch_size`, `lr`, `data`, `output`, `resume`, or `gpu`: those define the
/// invocation itself and stay CLI-only.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingConfigFile {
    #[serde(alias = "lr-scheduler")]
    lr_scheduler: Option<String>,
    #[serde(alias = "lr-step-size")]
    lr_step_size: Option<usize>,
    #[serde(alias = "lr-gamma")]
    lr_gamma: Option<f64>,
    #[serde(alias = "early-stopping")]
    early_stopping: Option<bool>,
    patience: Option<usize>,
    #[serde(alias = "min-delta")]
    min_delta: Option<f64>,
    #[serde(alias = "val-frequency")]
    val_frequency: Option<usize>,
    #[serde(alias = "warmup-steps")]
    warmup_steps: Option<usize>,
    #[serde(alias = "grad-clip")]
    grad_clip: Option<f64>,
    #[serde(alias = "save-frequency")]
    save_frequency: Option<usize>,
}

/// Read and parse a `--config` file into a [`TrainingConfigFile`]. Format is
/// selected from the file extension (`.toml` or `.json`); anything else, or a
/// file that fails to read/parse, is a hard error -- a config file the user
/// explicitly asked for is never silently skipped.
fn load_training_config_file(path: &Path) -> Result<TrainingConfigFile> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::config(format!(
            "Failed to read training config file {}: {}",
            path.display(),
            e
        ))
    })?;

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => toml::from_str(&content).map_err(|e| {
            CliError::config(format!(
                "Invalid TOML in training config file {}: {}",
                path.display(),
                e
            ))
        }),
        Some("json") => serde_json::from_str(&content).map_err(|e| {
            CliError::config(format!(
                "Invalid JSON in training config file {}: {}",
                path.display(),
                e
            ))
        }),
        other => Err(CliError::config(format!(
            "Unsupported training config file extension {:?} for {}. Supported: .toml, .json",
            other,
            path.display()
        ))),
    }
}

/// Resolve the final [`TrainingConfig`] from three layers, highest priority
/// first: (1) CLI flags explicitly given (the `Some(_)` case of each
/// `Option<T>` parameter -- `early_stopping` is a bare flag, so its "explicit"
/// state is just `true`), (2) the `--config` file if one was given, (3)
/// [`TrainingConfig::default()`]. Also validates `lr_scheduler` against
/// [`VALID_LR_SCHEDULERS`] up front so an unrecognized scheduler name fails
/// closed here rather than silently behaving like `"none"` deep inside the
/// training loop.
#[allow(clippy::too_many_arguments)]
fn resolve_training_config(
    config_path: Option<&Path>,
    lr_scheduler: Option<String>,
    lr_step_size: Option<usize>,
    lr_gamma: Option<f64>,
    early_stopping: bool,
    patience: Option<usize>,
    min_delta: Option<f64>,
    val_frequency: Option<usize>,
    warmup_steps: Option<usize>,
    grad_clip: Option<f64>,
    save_frequency: Option<usize>,
) -> Result<TrainingConfig> {
    let file_cfg = match config_path {
        Some(path) => load_training_config_file(path)?,
        None => TrainingConfigFile::default(),
    };
    let default = TrainingConfig::default();

    let resolved = TrainingConfig {
        lr_scheduler: lr_scheduler
            .or(file_cfg.lr_scheduler)
            .unwrap_or(default.lr_scheduler),
        lr_step_size: lr_step_size
            .or(file_cfg.lr_step_size)
            .unwrap_or(default.lr_step_size),
        lr_gamma: lr_gamma.or(file_cfg.lr_gamma).unwrap_or(default.lr_gamma),
        early_stopping: early_stopping || file_cfg.early_stopping.unwrap_or(false),
        patience: patience.or(file_cfg.patience).unwrap_or(default.patience),
        min_delta: min_delta
            .or(file_cfg.min_delta)
            .unwrap_or(default.min_delta),
        val_frequency: val_frequency
            .or(file_cfg.val_frequency)
            .unwrap_or(default.val_frequency),
        warmup_steps: warmup_steps
            .or(file_cfg.warmup_steps)
            .unwrap_or(default.warmup_steps),
        grad_clip: grad_clip
            .or(file_cfg.grad_clip)
            .unwrap_or(default.grad_clip),
        save_frequency: save_frequency
            .or(file_cfg.save_frequency)
            .unwrap_or(default.save_frequency),
    };

    if !VALID_LR_SCHEDULERS.contains(&resolved.lr_scheduler.as_str()) {
        return Err(CliError::config(format!(
            "Unknown --lr-scheduler '{}'. Supported values: {}",
            resolved.lr_scheduler,
            VALID_LR_SCHEDULERS.join(", ")
        )));
    }

    // The training loop uses these as divisors (`epoch % val_frequency`,
    // `epoch % save_frequency`, and -- for the "step"/"plateau" schedulers --
    // `epoch / lr_step_size`). A resolved value of 0 (whether from a CLI flag
    // or, now that --config is real, a config-file value) would panic deep
    // inside the training loop; reject it here instead, at the single choke
    // point where every other config value is already validated.
    if resolved.val_frequency == 0 {
        return Err(CliError::config(
            "--val-frequency must be greater than 0 (used as a divisor: epoch % val_frequency)",
        ));
    }
    if resolved.save_frequency == 0 {
        return Err(CliError::config(
            "--save-frequency must be greater than 0 (used as a divisor: epoch % save_frequency)",
        ));
    }
    if matches!(resolved.lr_scheduler.as_str(), "step" | "plateau") && resolved.lr_step_size == 0 {
        return Err(CliError::config(format!(
            "--lr-step-size must be greater than 0 when --lr-scheduler is '{}' (used as a divisor)",
            resolved.lr_scheduler
        )));
    }

    Ok(resolved)
}

/// Execute training command
pub async fn execute_train_command(command: TrainCommands, global: &GlobalOptions) -> Result<()> {
    match command {
        TrainCommands::Vocoder {
            model_type,
            data,
            output,
            config,
            epochs,
            batch_size,
            lr,
            lr_scheduler,
            lr_step_size,
            lr_gamma,
            early_stopping,
            patience,
            min_delta,
            val_frequency,
            warmup_steps,
            grad_clip,
            save_frequency,
            resume,
            gpu,
        } => {
            let training_config = resolve_training_config(
                config.as_deref(),
                lr_scheduler,
                lr_step_size,
                lr_gamma,
                early_stopping,
                patience,
                min_delta,
                val_frequency,
                warmup_steps,
                grad_clip,
                save_frequency,
            )?;

            let args = vocoder::VocoderTrainingArgs {
                model_type,
                data,
                output,
                config,
                epochs,
                batch_size,
                lr,
                resume,
                use_gpu: gpu || global.gpu,
                training_config,
            };

            vocoder::run_train_vocoder(args, global).await
        }
        TrainCommands::Acoustic {
            model_type,
            data,
            output,
            config,
            epochs,
            batch_size,
            lr,
            resume,
            gpu,
        } => {
            let args = acoustic::AcousticModelTrainingArgs {
                model_type: model_type.clone(),
                data: data.clone(),
                output: output.clone(),
                config: config.clone(),
                epochs,
                batch_size,
                lr,
                resume: resume.clone(),
                use_gpu: gpu || global.gpu,
            };
            acoustic::run_train_acoustic(args, global).await
        }
        TrainCommands::G2p {
            language,
            dictionary,
            output,
            config,
            epochs,
            lr,
        } => g2p::run_train_g2p(language, dictionary, output, config, epochs, lr, global).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_training_config_defaults_when_nothing_given() {
        let cfg = resolve_training_config(
            None, None, None, None, false, None, None, None, None, None, None,
        )
        .expect("defaults should always resolve");
        let default = TrainingConfig::default();
        assert_eq!(cfg.lr_scheduler, default.lr_scheduler);
        assert_eq!(cfg.patience, default.patience);
        assert_eq!(cfg.grad_clip, default.grad_clip);
        assert!(!cfg.early_stopping);
    }

    #[test]
    fn test_resolve_training_config_rejects_unknown_scheduler() {
        let err = resolve_training_config(
            None,
            Some("not-a-real-scheduler".to_string()),
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect_err("an unrecognized scheduler name must fail closed");
        assert!(err.to_string().contains("Unknown --lr-scheduler"));
    }

    /// Config-file values are used only when the CLI did not already specify
    /// the flag.
    #[test]
    fn test_resolve_training_config_uses_file_value_when_cli_omits_flag() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.toml");
        std::fs::write(&config_path, "lr_scheduler = \"cosine\"\npatience = 7\n")
            .expect("failed to write config file");

        let cfg = resolve_training_config(
            Some(&config_path),
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("valid config file should resolve");

        assert_eq!(cfg.lr_scheduler, "cosine");
        assert_eq!(cfg.patience, 7);
        // Fields the file didn't set still fall back to the built-in default.
        assert_eq!(cfg.grad_clip, TrainingConfig::default().grad_clip);
    }

    /// CLI flags always win over the config file, even when both specify the
    /// same setting.
    #[test]
    fn test_resolve_training_config_cli_flag_overrides_file() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.toml");
        std::fs::write(&config_path, "lr_scheduler = \"cosine\"\npatience = 7\n")
            .expect("failed to write config file");

        let cfg = resolve_training_config(
            Some(&config_path),
            Some("step".to_string()),
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("valid config file should resolve");

        // CLI explicitly set lr_scheduler, so it wins over the file's "cosine".
        assert_eq!(cfg.lr_scheduler, "step");
        // CLI left patience unset, so the file's value is used.
        assert_eq!(cfg.patience, 7);
    }

    /// A malformed config file must fail closed (typed error), never be
    /// silently ignored.
    #[test]
    fn test_resolve_training_config_rejects_malformed_file() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.toml");
        std::fs::write(&config_path, "this is not valid toml {{{")
            .expect("failed to write config file");

        let result = resolve_training_config(
            Some(&config_path),
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err(), "malformed config file must be rejected");
    }

    /// An unsupported config file extension must fail closed rather than
    /// silently being skipped (the whole point of fixing defect 3).
    #[test]
    fn test_resolve_training_config_rejects_unsupported_extension() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.yaml");
        std::fs::write(&config_path, "lr_scheduler: cosine\n")
            .expect("failed to write config file");

        let result = resolve_training_config(
            Some(&config_path),
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let err = result.expect_err("unsupported extension must be rejected");
        assert!(err.to_string().contains("Unsupported training config"));
    }

    /// A missing config file must fail closed rather than silently falling
    /// back to defaults.
    #[test]
    fn test_resolve_training_config_rejects_missing_file() {
        let missing = PathBuf::from("/this/path/does/not/exist/train.toml");
        let result = resolve_training_config(
            Some(&missing),
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err(), "a missing config file must be rejected");
    }

    /// JSON config files are supported too, and the kebab-case alias (matching
    /// the CLI flag spelling) works alongside the canonical snake_case name.
    #[test]
    fn test_resolve_training_config_supports_json_and_kebab_alias() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.json");
        std::fs::write(
            &config_path,
            r#"{"lr-scheduler": "onecycle", "grad_clip": 2.5}"#,
        )
        .expect("failed to write config file");

        let cfg = resolve_training_config(
            Some(&config_path),
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("valid JSON config file should resolve");

        assert_eq!(cfg.lr_scheduler, "onecycle");
        assert_eq!(cfg.grad_clip, 2.5);
    }

    /// The bare `--early-stopping` CLI flag can only turn early stopping on; it
    /// never turns off a config file's early_stopping = true.
    #[test]
    fn test_resolve_training_config_early_stopping_is_cli_or_file() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp.path().join("train.toml");
        std::fs::write(&config_path, "early_stopping = true\n")
            .expect("failed to write config file");

        let cfg = resolve_training_config(
            Some(&config_path),
            None,
            None,
            None,
            false, // CLI flag not given
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("valid config file should resolve");
        assert!(
            cfg.early_stopping,
            "config file's early_stopping=true must take effect when the CLI flag is absent"
        );
    }

    /// `--val-frequency 0` would panic on `epoch % val_frequency` deep inside
    /// the training loop; it must be rejected here instead, whether it came
    /// from the CLI or (now that --config is real) a config file.
    #[test]
    fn test_resolve_training_config_rejects_zero_val_frequency() {
        let err = resolve_training_config(
            None,
            None,
            None,
            None,
            false,
            None,
            None,
            Some(0),
            None,
            None,
            None,
        )
        .expect_err("val_frequency=0 must be rejected");
        assert!(err.to_string().contains("--val-frequency"));
    }

    /// Same guard for `--save-frequency`, which divides `epoch %
    /// save_frequency`.
    #[test]
    fn test_resolve_training_config_rejects_zero_save_frequency() {
        let err = resolve_training_config(
            None,
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            Some(0),
        )
        .expect_err("save_frequency=0 must be rejected");
        assert!(err.to_string().contains("--save-frequency"));
    }

    /// `--lr-step-size 0` only matters for schedulers that actually divide by
    /// it ("step" and "plateau"); it must be rejected for those but left
    /// alone for schedulers that never use it as a divisor (e.g. "cosine").
    #[test]
    fn test_resolve_training_config_rejects_zero_step_size_only_for_step_and_plateau() {
        let err = resolve_training_config(
            None,
            Some("step".to_string()),
            Some(0),
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect_err("lr_step_size=0 with scheduler=step must be rejected");
        assert!(err.to_string().contains("--lr-step-size"));

        let err = resolve_training_config(
            None,
            Some("plateau".to_string()),
            Some(0),
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect_err("lr_step_size=0 with scheduler=plateau must be rejected");
        assert!(err.to_string().contains("--lr-step-size"));

        let cfg = resolve_training_config(
            None,
            Some("cosine".to_string()),
            Some(0),
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect(
            "lr_step_size=0 with scheduler=cosine is harmless (never divided by) and must resolve",
        );
        assert_eq!(cfg.lr_step_size, 0);
    }
}
