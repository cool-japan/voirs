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

use crate::GlobalOptions;
use clap::Subcommand;
use std::path::PathBuf;
use voirs_sdk::Result;

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

        /// Training config file (TOML/JSON)
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

        /// Learning rate scheduler (none, step, cosine, exponential, onecycle)
        #[arg(long, default_value = "none")]
        lr_scheduler: String,

        /// LR scheduler step size (for step scheduler)
        #[arg(long, default_value = "100")]
        lr_step_size: usize,

        /// LR scheduler gamma (decay factor)
        #[arg(long, default_value = "0.1")]
        lr_gamma: f64,

        /// Enable early stopping
        #[arg(long)]
        early_stopping: bool,

        /// Early stopping patience (epochs)
        #[arg(long, default_value = "50")]
        patience: usize,

        /// Early stopping minimum delta
        #[arg(long, default_value = "0.0001")]
        min_delta: f64,

        /// Validation frequency (epochs)
        #[arg(long, default_value = "5")]
        val_frequency: usize,

        /// Warmup steps
        #[arg(long, default_value = "0")]
        warmup_steps: usize,

        /// Gradient clipping value (0 = disabled)
        #[arg(long, default_value = "1.0")]
        grad_clip: f64,

        /// Save checkpoint every N epochs
        #[arg(long, default_value = "10")]
        save_frequency: usize,

        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<PathBuf>,

        /// Use GPU if available
        #[arg(long)]
        gpu: bool,
    },

    /// Train acoustic model (VITS, FastSpeech2)
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

    /// Train G2P model
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
            let training_config = TrainingConfig {
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
            };

            vocoder::run_train_vocoder(
                model_type,
                data,
                output,
                config,
                epochs,
                batch_size,
                lr,
                resume,
                gpu || global.gpu,
                training_config,
                global,
            )
            .await
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
            acoustic::run_train_acoustic(
                model_type,
                data,
                output,
                config,
                epochs,
                batch_size,
                lr,
                resume,
                gpu || global.gpu,
                global,
            )
            .await
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
