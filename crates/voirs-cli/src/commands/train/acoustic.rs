//! Acoustic model training command implementation
//!
//! Provides CLI interface for training acoustic models (VITS, FastSpeech2).
//!
//! # Current status: fails closed
//!
//! Neither underlying trainer in `voirs-acoustic` performs real gradient-based
//! training yet (see the `VITS_TRAINING_BLOCKED_REASON` and
//! `FASTSPEECH2_TRAINING_BLOCKED_REASON` constants below for the exact evidence).
//! Running the old training loop against real data would still produce an
//! untrained, randomly initialized checkpoint dressed up with a "training
//! completed successfully" banner. Rather than fabricate that success, this
//! command validates its inputs and then refuses to run, with a diagnostic
//! explaining exactly what is missing. This guard should be removed once
//! `voirs-acoustic` implements real backward passes / optimizer steps for both
//! trainers.

use crate::error::CliError;
use crate::GlobalOptions;
use std::path::{Path, PathBuf};
use voirs_sdk::Result;

/// Arguments for acoustic model training
///
/// Consolidates all training parameters to improve maintainability.
///
/// # Example
///
/// ```no_run
/// # use voirs_cli::commands::train::acoustic::AcousticModelTrainingArgs;
/// # use std::path::PathBuf;
/// let args = AcousticModelTrainingArgs {
///     model_type: "vits".to_string(),
///     data: PathBuf::from("./data/train"),
///     output: PathBuf::from("./models/output"),
///     config: None,
///     epochs: 100,
///     batch_size: 16,
///     lr: 0.0002,
///     resume: None,
///     use_gpu: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AcousticModelTrainingArgs {
    /// Model type ("vits" or "fastspeech2")
    pub model_type: String,
    /// Training data directory path
    pub data: PathBuf,
    /// Output directory for trained models
    pub output: PathBuf,
    /// Optional model configuration file path
    pub config: Option<PathBuf>,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate
    pub lr: f64,
    /// Optional checkpoint path to resume from
    pub resume: Option<PathBuf>,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

/// Run acoustic model training
///
/// # Arguments
///
/// * `args` - Training configuration and parameters
/// * `global` - Global CLI options
pub async fn run_train_acoustic(
    args: AcousticModelTrainingArgs,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║          🎤 VoiRS Acoustic Model Training                 ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Model type:    {:<40} ║", args.model_type);
        println!("║ Data path:     {:<40} ║", truncate_path(&args.data, 40));
        println!("║ Output path:   {:<40} ║", truncate_path(&args.output, 40));
        println!("║ Epochs:        {:<40} ║", args.epochs);
        println!("║ Batch size:    {:<40} ║", args.batch_size);
        println!("║ Learning rate: {:<40} ║", args.lr);
        println!(
            "║ GPU enabled:   {:<40} ║",
            if args.use_gpu { "Yes" } else { "No" }
        );
        if let Some(ref resume_path) = args.resume {
            println!("║ Resume from:   {:<40} ║", truncate_path(resume_path, 40));
        }
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
    }

    // Validate input
    if !args.data.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Training data directory not found: {}",
            args.data.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    match args.model_type.as_str() {
        "vits" => {
            train_vits(
                AcousticTrainingArgs {
                    data: args.data,
                    output: args.output,
                    config: args.config,
                    epochs: args.epochs,
                    batch_size: args.batch_size,
                    lr: args.lr,
                    resume: args.resume,
                    use_gpu: args.use_gpu,
                },
                global,
            )
            .await
        }
        "fastspeech2" => {
            train_fastspeech2(
                AcousticTrainingArgs {
                    data: args.data,
                    output: args.output,
                    config: args.config,
                    epochs: args.epochs,
                    batch_size: args.batch_size,
                    lr: args.lr,
                    resume: args.resume,
                    use_gpu: args.use_gpu,
                },
                global,
            )
            .await
        }
        _ => Err(voirs_sdk::VoirsError::config_error(format!(
            "Unsupported acoustic model type: {}. Supported: vits, fastspeech2",
            args.model_type
        ))),
    }
}

/// Configuration for acoustic model training
struct AcousticTrainingArgs {
    data: PathBuf,
    output: PathBuf,
    config: Option<PathBuf>,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    resume: Option<PathBuf>,
    use_gpu: bool,
}

/// Why VITS training refuses to run.
///
/// Evidence, all in `crates/voirs-acoustic/src/vits/trainer.rs`:
/// - `VitsTrainer::train_step` (~L452) never calls a backward pass or optimizer
///   step, so model weights never change no matter how much data is fed in or how
///   many epochs run.
/// - `PeriodDiscriminator::simulate_conv` (~L174) and `ScaleDiscriminator::
///   simulate_conv` (~L312) return `Tensor::zeros(..)`, ignoring their input
///   entirely, so every discriminator/adversarial/feature-matching loss term is a
///   constant regardless of whether the audio is real or generated.
/// - `calculate_kl_divergence_loss` (~L659) and `calculate_duration_loss` (~L666)
///   return `fastrand`-generated numbers unrelated to the model or its input.
/// - `validate_step` (~L765) is entirely `fastrand`-based.
/// - `save_checkpoint` (~L783) writes a JSON metadata stub to the `.safetensors`
///   path; no tensor weights are ever serialized.
const VITS_TRAINING_BLOCKED_REASON: &str =
    "VitsTrainer::train_step never performs a backward pass or optimizer step \
     (crates/voirs-acoustic/src/vits/trainer.rs), so model weights never change; its \
     GAN discriminators simulate convolutions with Tensor::zeros(..) regardless of the \
     input audio (PeriodDiscriminator/ScaleDiscriminator::simulate_conv), making \
     discriminator/adversarial/feature-matching losses input-independent constants; the \
     KL-divergence and duration loss terms are fastrand-generated numbers unrelated to \
     the model; validate_step returns fastrand-based metrics; and save_checkpoint writes \
     JSON metadata rather than real tensor weights to the .safetensors path. Running this \
     loop would silently hand back an untrained, randomly initialized model presented as \
     a completed training run, so this command refuses to run until voirs-acoustic \
     implements real gradient-based training for VITS";

/// Why FastSpeech2 training refuses to run.
///
/// Evidence, all in `crates/voirs-acoustic/src/fastspeech2_trainer.rs`:
/// - `FastSpeech2Trainer::train_step` never calls a backward pass or optimizer
///   step, so model weights never change.
/// - `FastSpeech2Encoder::forward` (~L86-99) maps every phoneme to the same
///   constant id (`.map(|_| 1u32)`) regardless of its symbol, discarding all
///   phoneme identity before it ever reaches the model.
/// - The variance adaptor predicts duration/pitch/energy from an all-zero
///   `dummy_features` placeholder (~L674) instead of the real encoder output.
/// - `validate_step` (~L840) is entirely `fastrand`-based.
/// - `save_checkpoint` (~L860) writes a JSON metadata stub to the `.safetensors`
///   path; no tensor weights are ever serialized.
const FASTSPEECH2_TRAINING_BLOCKED_REASON: &str =
    "FastSpeech2Trainer::train_step never performs a backward pass or optimizer step \
     (crates/voirs-acoustic/src/fastspeech2_trainer.rs), so model weights never change; \
     FastSpeech2Encoder::forward maps every phoneme to the same constant id regardless of \
     its symbol (`.map(|_| 1u32)`), discarding all phoneme identity before it reaches the \
     model; the variance adaptor predicts duration/pitch/energy from an all-zero \
     placeholder feature vector instead of real encoder output; validate_step returns \
     fastrand-based metrics; and save_checkpoint writes JSON metadata rather than real \
     tensor weights to the .safetensors path. Running this loop would silently hand back \
     an untrained, content-blind model presented as a completed training run, so this \
     command refuses to run until voirs-acoustic implements real gradient-based training \
     for FastSpeech2";

/// Refuse to run acoustic model training and explain exactly why.
///
/// The current `voirs-acoustic` trainers never perform a real gradient-based
/// training step (see `reason`), so any checkpoint they "save" is in fact an
/// untouched, randomly initialized model. Printing a success banner and writing
/// that checkpoint would be fabricated output, so this fails closed with a typed,
/// diagnostic error instead of running the loop.
fn fail_closed_acoustic_training(
    model_name: &str,
    args: AcousticTrainingArgs,
    reason: &str,
) -> Result<()> {
    let AcousticTrainingArgs {
        data,
        output,
        config: _config,
        epochs,
        batch_size,
        lr,
        resume: _resume,
        use_gpu,
    } = args;

    Err(CliError::NotImplemented(format!(
        "{model_name} training requested (data={}, output={}, epochs={epochs}, \
         batch_size={batch_size}, lr={lr}, gpu={use_gpu}) but refused: {reason}",
        data.display(),
        output.display(),
    ))
    .into())
}

async fn train_vits(args: AcousticTrainingArgs, _global: &GlobalOptions) -> Result<()> {
    fail_closed_acoustic_training("VITS", args, VITS_TRAINING_BLOCKED_REASON)
}

async fn train_fastspeech2(args: AcousticTrainingArgs, _global: &GlobalOptions) -> Result<()> {
    fail_closed_acoustic_training("FastSpeech2", args, FASTSPEECH2_TRAINING_BLOCKED_REASON)
}

fn truncate_path(path: &Path, max_len: usize) -> String {
    let path_str = path.display().to_string();
    if path_str.len() <= max_len {
        path_str
    } else {
        format!("...{}", &path_str[path_str.len() - (max_len - 3)..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_path() {
        let path = PathBuf::from("/very/long/path/to/some/directory/file.txt");
        let truncated = truncate_path(&path, 20);
        assert!(truncated.len() <= 20);
        assert!(truncated.starts_with("..."));
    }

    fn test_global_options(quiet: bool) -> GlobalOptions {
        GlobalOptions {
            config: None,
            verbose: 0,
            quiet,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        }
    }

    /// VITS training must fail closed (never fabricate a "trained" checkpoint) and
    /// must explain why, rather than silently writing an untrained model.
    #[tokio::test]
    async fn test_vits_training_fails_closed_without_fabricating_a_checkpoint() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let data_dir = temp.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("failed to create data dir");
        let output_dir = temp.path().join("out");

        let args = AcousticModelTrainingArgs {
            model_type: "vits".to_string(),
            data: data_dir,
            output: output_dir.clone(),
            config: None,
            epochs: 1,
            batch_size: 2,
            lr: 0.0002,
            resume: None,
            use_gpu: false,
        };

        let result = run_train_acoustic(args, &test_global_options(true)).await;
        assert!(
            result.is_err(),
            "VITS training must fail closed instead of fabricating a trained model"
        );
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("backward pass"),
            "error should explain the real blocker (no backward pass), got: {message}"
        );

        // No checkpoint should ever be fabricated for an untrained model.
        let entries: Vec<_> = std::fs::read_dir(&output_dir)
            .expect("output dir should exist")
            .collect();
        assert!(
            entries.is_empty(),
            "must not write any checkpoint for a model that was never actually trained"
        );
    }

    /// Same guarantee for FastSpeech2: fails closed with a model-specific reason.
    #[tokio::test]
    async fn test_fastspeech2_training_fails_closed_without_fabricating_a_checkpoint() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let data_dir = temp.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("failed to create data dir");
        let output_dir = temp.path().join("out");

        let args = AcousticModelTrainingArgs {
            model_type: "fastspeech2".to_string(),
            data: data_dir,
            output: output_dir.clone(),
            config: None,
            epochs: 1,
            batch_size: 2,
            lr: 0.0002,
            resume: None,
            use_gpu: false,
        };

        let result = run_train_acoustic(args, &test_global_options(true)).await;
        assert!(
            result.is_err(),
            "FastSpeech2 training must fail closed instead of fabricating a trained model"
        );
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("phoneme"),
            "error should explain the real blocker (content-blind phoneme ids), got: {message}"
        );
        assert!(
            entries_is_empty(&output_dir),
            "must not write any checkpoint for a model that was never actually trained"
        );
    }

    fn entries_is_empty(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .map(|mut it| it.next().is_none())
            .unwrap_or(true)
    }

    /// An unsupported model type must still be rejected before reaching either
    /// fail-closed path (existing, unrelated validation behavior).
    #[tokio::test]
    async fn test_unsupported_model_type_is_rejected() {
        let temp = tempfile::tempdir().expect("failed to create temp dir");
        let data_dir = temp.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("failed to create data dir");

        let args = AcousticModelTrainingArgs {
            model_type: "not-a-real-model".to_string(),
            data: data_dir,
            output: temp.path().join("out"),
            config: None,
            epochs: 1,
            batch_size: 2,
            lr: 0.0002,
            resume: None,
            use_gpu: false,
        };

        let result = run_train_acoustic(args, &test_global_options(true)).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported acoustic model type"));
    }
}
