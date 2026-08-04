//! Vocoder training command implementation
//!
//! Provides CLI interface for training vocoder models (HiFi-GAN, DiffWave).

use super::progress::{
    EpochMetrics, ResourceUsage, TrainingMetrics, TrainingProgress, TrainingStats,
};
use crate::error::{CliError, Result};
use crate::GlobalOptions;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{optim::AdamW, Optimizer, VarBuilder, VarMap};
use std::path::{Path, PathBuf};
use std::time::Instant;
use voirs_vocoder::models::diffwave::diffusion::DiffWave;

/// Configuration for vocoder training operations
///
/// This struct consolidates all parameters needed for training vocoder models
/// (DiffWave, HiFi-GAN) to reduce function signature complexity and improve
/// maintainability.
///
/// # Examples
///
/// ```no_run
/// use voirs_cli::commands::train::vocoder::VocoderTrainingArgs;
/// use voirs_cli::commands::train::TrainingConfig;
/// use std::path::PathBuf;
///
/// let args = VocoderTrainingArgs {
///     model_type: "diffwave".to_string(),
///     data: PathBuf::from("./training_data"),
///     output: PathBuf::from("./checkpoints"),
///     config: None,
///     epochs: 100,
///     batch_size: 32,
///     lr: 0.0002,
///     resume: None,
///     use_gpu: true,
///     training_config: TrainingConfig::default(),
/// };
/// ```
pub struct VocoderTrainingArgs {
    /// Type of vocoder model to train ("diffwave" or "hifigan")
    pub model_type: String,
    /// Path to training data directory containing audio/mel pairs
    pub data: PathBuf,
    /// Output directory for checkpoints and logs
    pub output: PathBuf,
    /// Optional path to model configuration file
    pub config: Option<PathBuf>,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate for optimizer
    pub lr: f64,
    /// Optional path to checkpoint for resuming training
    pub resume: Option<PathBuf>,
    /// Whether to use GPU acceleration (CUDA/Metal)
    pub use_gpu: bool,
    /// Advanced training configuration (scheduler, early stopping, etc.)
    pub training_config: super::TrainingConfig,
}

/// Run vocoder training
pub async fn run_train_vocoder(args: VocoderTrainingArgs, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║          🎵 VoiRS Vocoder Training                        ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Model type:    {:<40} ║", args.model_type);
        println!("║ Data path:     {:<40} ║", truncate_path(&args.data, 40));
        println!("║ Output path:   {:<40} ║", truncate_path(&args.output, 40));
        println!("║ Epochs:        {:<40} ║", args.epochs);
        println!("║ Batch size:    {:<40} ║", args.batch_size);
        println!("║ Learning rate: {:<40} ║", args.lr);
        println!(
            "║ LR scheduler:  {:<40} ║",
            args.training_config.lr_scheduler
        );
        if let Some(ref config_path) = args.config {
            println!("║ Config file:   {:<40} ║", truncate_path(config_path, 40));
        }
        if args.training_config.early_stopping {
            println!(
                "║ Early stopping: {} (patience: {})                   ║",
                if args.training_config.early_stopping {
                    "Yes"
                } else {
                    "No"
                },
                args.training_config.patience
            );
        }
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
        return Err(CliError::config(format!(
            "Training data directory not found: {}\n\
             \n\
             The directory should contain:\n\
             - Audio files (.wav, .flac) or\n\
             - Mel spectrogram files (.npy, .pt) or\n\
             - Audio-mel pairs in a structured format\n\
             \n\
             Please ensure the path is correct and the directory exists.",
            args.data.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    match args.model_type.as_str() {
        "diffwave" => train_diffwave(args, global).await,
        "hifigan" => train_hifigan(args, global).await,
        _ => Err(CliError::config(format!(
            "Unsupported vocoder model type: '{}'\n\
             \n\
             Supported model types:\n\
             - diffwave: DiffWave probabilistic vocoder (high quality, slower)\n\
             - hifigan:  HiFi-GAN neural vocoder (fast, good quality)\n\
             \n\
             Usage: voirs train vocoder --model-type diffwave|hifigan ...",
            args.model_type
        ))),
    }
}

async fn train_diffwave(args: VocoderTrainingArgs, global: &GlobalOptions) -> Result<()> {
    use super::data_loader::VocoderDataLoader;
    use candle_nn::VarMap;
    use voirs_vocoder::models::diffwave::diffusion::DiffWaveConfig;

    if !global.quiet {
        println!("🔧 Initializing DiffWave training...\n");
    }

    // Setup device
    let device = if args.use_gpu {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => {
                    if !global.quiet {
                        println!("✓ Using Metal GPU (Apple Silicon)\n");
                    }
                    d
                }
                Err(_) => {
                    if !global.quiet {
                        println!("⚠️  Metal GPU not available, falling back to CPU\n");
                    }
                    Device::Cpu
                }
            }
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            match Device::new_cuda(0) {
                Ok(d) => {
                    if !global.quiet {
                        println!("✓ Using CUDA GPU\n");
                    }
                    d
                }
                Err(_) => {
                    if !global.quiet {
                        println!("⚠️  CUDA GPU not available, falling back to CPU\n");
                    }
                    Device::Cpu
                }
            }
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            if !global.quiet {
                println!("⚠️  GPU requested but not compiled with GPU support, using CPU\n");
            }
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Load dataset
    if !global.quiet {
        println!("📚 Loading dataset from {:?}...", args.data);
    }

    let mut data_loader = VocoderDataLoader::load(&args.data).await?;

    if !global.quiet {
        println!("   ✓ Loaded {} audio samples\n", data_loader.len());
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Create model with VarMap for training
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model_config = DiffWaveConfig::default();

    if !global.quiet {
        println!("🔨 Creating DiffWave model...");
    }

    let model = DiffWave::new(model_config, device.clone(), vb).map_err(|e| {
        CliError::config(format!(
            "Failed to create DiffWave model: {}\n\
             \n\
             Possible causes:\n\
             - Insufficient GPU/CPU memory\n\
             - Incompatible device configuration\n\
             - Missing model dependencies\n\
             \n\
             Try: Use --no-gpu flag or reduce batch size",
            e
        ))
    })?;

    // Resume from checkpoint, if requested. Must happen after the model is
    // constructed (so every parameter name is already registered in `varmap`)
    // and before the optimizer is created.
    let mut start_epoch = 0usize;
    if let Some(ref resume_path) = args.resume {
        start_epoch = resume_from_checkpoint(&mut varmap, resume_path, "DiffWave", global.quiet)?;
    }

    // Create optimizer
    let params = varmap.all_vars();
    // A second, independent snapshot of the same underlying parameter storage
    // (Var is a cheap Arc-like handle), used for gradient-norm computation and
    // clipping without fighting the optimizer for ownership of `params`.
    let train_vars = params.clone();
    let mut optimizer = AdamW::new_lr(params, args.lr).map_err(|e| {
        CliError::config(format!(
            "Failed to create AdamW optimizer: {}\n\
             \n\
             This may indicate:\n\
             - Invalid learning rate (try 0.0001 to 0.001)\n\
             - Model parameters not properly initialized\n\
             \n\
             Current learning rate: {}",
            e, args.lr
        ))
    })?;

    // Calculate batches per epoch
    let batches_per_epoch = data_loader.len().div_ceil(args.batch_size);

    if !global.quiet {
        println!("✅ Training setup complete!\n");
        println!("📊 Model Information:");
        println!("   Parameters: {}", model.num_parameters());
        println!("   Device: {:?}", device);
        println!("   Batches per epoch: {}", batches_per_epoch);
        println!("\n🚀 Starting training with real DiffWave model...\n");
    }

    // Create progress tracker
    let mut progress = TrainingProgress::new(args.epochs, batches_per_epoch, !global.quiet);

    // Training statistics
    let start_time = Instant::now();
    let mut total_steps = 0;
    let mut best_val_loss = f64::MAX;
    // Epoch at which validation loss last improved. Early-stopping patience is
    // measured as real epochs elapsed since this point (see below), matching
    // `--patience`'s own "(epochs)" help text regardless of `--val-frequency`.
    // Initialized to `start_epoch` so a `--resume`d run doesn't inherit an
    // artificially stale improvement point from before the process restarted.
    let mut last_improvement_epoch = start_epoch;

    // Tracks genuine training-step failures across the whole run (never reset
    // between epochs). Crossing the threshold aborts the run with a real error
    // instead of silently continuing on fabricated data.
    let mut error_count: usize = 0;
    // Last real (non-fabricated) batch loss/gradient-norm, used only to keep the
    // live progress display continuous across a transient failure; never fed
    // into any aggregate/reported statistic.
    let mut last_known_batch_loss: Option<f64> = None;
    let mut last_known_grad_norm: Option<f64> = None;
    // Real per-epoch statistics from the most recently completed epoch, used for
    // the final summary and checkpoint instead of hardcoded literals.
    let mut last_completed_epoch: Option<usize> = None;
    let mut last_avg_epoch_loss: f64 = 0.0;
    let mut last_val_loss: Option<f64> = None;

    // Calculate total warmup steps (if warmup_steps > 0, treat as absolute steps)
    let warmup_steps = args.training_config.warmup_steps;

    // Training loop
    for epoch in start_epoch..args.epochs {
        progress.start_epoch(epoch, batches_per_epoch);

        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut successful_batches: u64 = 0;

        // Reset data loader for new epoch
        data_loader.reset();

        // Batch loop
        for batch_idx in 0..batches_per_epoch {
            let batch_start = Instant::now();

            // Load real batch data
            let batch_data = data_loader.get_batch(args.batch_size)?;

            // Convert batch to tensors
            let (audio_tensors, mel_tensors) = convert_batch_to_tensors(&batch_data, args.use_gpu)
                .map_err(|e| CliError::config(format!("Tensor conversion failed: {}", e)))?;

            // Real training step with DiffWave model
            if epoch == 0 && batch_idx == 0 && !global.quiet {
                println!("   🔬 Attempting real DiffWave forward pass...");
            }

            let (batch_loss, grad_norm) = match train_step_real(
                &model,
                &mut optimizer,
                &train_vars,
                &audio_tensors,
                &mel_tensors,
                &device,
                args.training_config.grad_clip,
            ) {
                Ok((loss, norm)) => {
                    // Log first batch to confirm real training is working
                    if epoch == 0 && batch_idx == 0 && !global.quiet {
                        println!("   ✅ Real forward pass SUCCESS! Loss: {:.6}", loss);
                    }
                    epoch_loss += loss;
                    successful_batches += 1;
                    last_known_batch_loss = Some(loss);
                    last_known_grad_norm = Some(norm);
                    (loss, norm)
                }
                Err(e) => {
                    error_count += 1;
                    if !global.quiet {
                        eprintln!(
                            "⚠️  DiffWave training step {}/{} failed: {}",
                            epoch + 1,
                            batch_idx + 1,
                            e
                        );
                    }
                    if should_abort_on_errors(error_count, batches_per_epoch) {
                        return Err(CliError::config(format!(
                            "Too many DiffWave training-step failures ({error_count} failed \
                             out of {} batches attempted), aborting instead of fabricating \
                             training progress. Last error: {e}",
                            total_steps + 1
                        )));
                    }
                    // Do not fabricate a loss/grad-norm value: reuse the last
                    // real measurement for the live display only, or NaN if no
                    // batch has ever succeeded yet. Never added to epoch_loss.
                    (
                        last_known_batch_loss.unwrap_or(f64::NAN),
                        last_known_grad_norm.unwrap_or(f64::NAN),
                    )
                }
            };
            total_steps += 1;

            // Apply warmup to learning rate (overrides scheduler during warmup
            // phase) and push it into the real optimizer -- not just the
            // display -- via candle-nn's `Optimizer::set_learning_rate`.
            if warmup_steps > 0 && total_steps <= warmup_steps {
                // Linear warmup: gradually increase from 0 to target lr
                let warmup_lr = args.lr * (total_steps as f64 / warmup_steps as f64);
                optimizer.set_learning_rate(warmup_lr);

                if total_steps % 100 == 0 && !global.quiet {
                    println!(
                        "   🔥 Warmup: step {}/{}, lr: {:.6}",
                        total_steps,
                        warmup_steps,
                        optimizer.learning_rate()
                    );
                }
            }

            // Calculate samples per second
            let batch_duration = batch_start.elapsed().as_secs_f64();
            let samples_per_sec = (batch_data.len() as f64) / batch_duration;

            // Update progress
            progress.update_batch(batch_idx, batch_loss, samples_per_sec);

            // Update metrics every 10 batches
            if batch_idx % 10 == 0 {
                let metrics = TrainingMetrics {
                    loss: batch_loss,
                    // Read back from the optimizer itself so the displayed
                    // value can never drift from the rate actually applied.
                    learning_rate: optimizer.learning_rate(),
                    grad_norm: Some(grad_norm),
                };
                progress.update_metrics(&metrics);

                // Update resources
                let resources = ResourceUsage::current();
                progress.update_resources(&resources);
            }

            progress.finish_batch();
        }

        // Calculate epoch metrics: divide by batches that actually succeeded, not
        // by batches_per_epoch, so a failed batch (excluded above) cannot dilute
        // the reported average toward a falsely lower loss.
        let avg_epoch_loss = epoch_loss / successful_batches.max(1) as f64;

        // Perform validation at specified frequency
        let val_loss = if epoch % args.training_config.val_frequency == 0 {
            // Use 10% of data for validation (or minimum 32 samples)
            let val_samples = (data_loader.len() / 10).max(32);
            Some(
                run_validation(
                    &model,
                    &mut data_loader,
                    args.batch_size,
                    &device,
                    val_samples,
                )
                .await,
            )
        } else {
            None
        };

        // Record real per-epoch statistics for the final summary/checkpoint.
        last_completed_epoch = Some(epoch);
        last_avg_epoch_loss = avg_epoch_loss;
        if val_loss.is_some() {
            last_val_loss = val_loss;
        }

        // Update best validation loss and check early stopping. Patience is
        // counted in real epochs elapsed since the last improvement (not in
        // validation *events*), so it matches the flag's documented
        // "(epochs)" semantics regardless of --val-frequency.
        if let Some(vl) = val_loss {
            let improved = vl < (best_val_loss - args.training_config.min_delta);

            if improved {
                best_val_loss = vl;
                last_improvement_epoch = epoch;

                // Save best checkpoint
                if !global.quiet {
                    println!("\n💾 New best model saved (val_loss: {:.4})", vl);
                }
                save_checkpoint(
                    &args.output,
                    "best_model",
                    epoch,
                    avg_epoch_loss,
                    Some(vl),
                    "DiffWave",
                    &varmap,
                )
                .await?;
            } else if args.training_config.early_stopping {
                let epochs_without_improvement = epoch.saturating_sub(last_improvement_epoch);
                if epochs_without_improvement >= args.training_config.patience {
                    if !global.quiet {
                        println!(
                            "\n⚠️  Early stopping triggered after {} epochs without improvement",
                            epochs_without_improvement
                        );
                    }
                    break;
                }
            }
        }

        let epoch_metrics = EpochMetrics {
            epoch,
            train_loss: avg_epoch_loss,
            val_loss,
            duration: epoch_start.elapsed(),
        };

        progress.finish_epoch(&epoch_metrics);

        // Apply learning rate scheduler (only after warmup is complete) and
        // push it into the real optimizer, not just the display.
        if args.training_config.lr_scheduler != "none" && total_steps > warmup_steps {
            let scheduled_lr = apply_lr_scheduler(
                &args.training_config.lr_scheduler,
                args.lr,
                epoch,
                args.training_config.lr_step_size,
                args.training_config.lr_gamma,
                args.epochs,
                epoch.saturating_sub(last_improvement_epoch),
            );
            optimizer.set_learning_rate(scheduled_lr);

            if !global.quiet && epoch % 10 == 0 {
                println!("   📊 Learning rate: {:.6}", optimizer.learning_rate());
            }
        } else if total_steps <= warmup_steps && !global.quiet && epoch % 10 == 0 {
            println!(
                "   🔥 Still in warmup phase (step {}/{})",
                total_steps, warmup_steps
            );
        }

        // Save checkpoint at specified frequency
        if epoch % args.training_config.save_frequency == 0 {
            save_checkpoint(
                &args.output,
                &format!("epoch_{}", epoch),
                epoch,
                avg_epoch_loss,
                val_loss,
                "DiffWave",
                &varmap,
            )
            .await?;
            if !global.quiet {
                println!("\n💾 Checkpoint saved: epoch_{}.safetensors", epoch);
            }
        }
    }

    // Finish training
    let total_duration = start_time.elapsed();

    // Save final model and print the summary only if at least one epoch actually
    // completed; otherwise there is nothing real to report or persist.
    match last_completed_epoch {
        Some(final_epoch) => {
            save_checkpoint(
                &args.output,
                "final_model",
                final_epoch,
                last_avg_epoch_loss,
                last_val_loss,
                "DiffWave",
                &varmap,
            )
            .await?;

            progress.finish("✅ Training completed successfully!");

            if !global.quiet {
                let stats = TrainingStats {
                    total_duration,
                    epochs_completed: final_epoch + 1,
                    total_steps,
                    final_train_loss: last_avg_epoch_loss,
                    final_val_loss: last_val_loss,
                    best_val_loss: Some(best_val_loss),
                    avg_samples_per_sec: (total_steps * args.batch_size) as f64
                        / total_duration.as_secs_f64(),
                };
                progress.print_summary(&stats);

                println!("\n📊 Model outputs:");
                println!(
                    "   - Final model: {}/final_model.safetensors",
                    args.output.display()
                );
                println!(
                    "   - Best model:  {}/best_model.safetensors",
                    args.output.display()
                );
            }
        }
        None => {
            progress.finish("⚠️  No epochs were completed; nothing was trained.");
            if !global.quiet {
                println!(
                    "\n⚠️  Training requested {} epochs but none completed; no checkpoint \
                     was saved.",
                    args.epochs
                );
            }
        }
    }

    Ok(())
}

async fn train_hifigan(args: VocoderTrainingArgs, global: &GlobalOptions) -> Result<()> {
    use super::data_loader::VocoderDataLoader;
    use candle_nn::VarMap;
    use voirs_vocoder::models::hifigan::{
        generator::HiFiGanGenerator, HiFiGanConfig, HiFiGanVariant,
    };

    if !global.quiet {
        println!("🔧 Initializing HiFi-GAN training...\n");
    }

    // Setup device
    let device = if args.use_gpu {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => {
                    if !global.quiet {
                        println!("✓ Using Metal GPU (Apple Silicon)\n");
                    }
                    d
                }
                Err(_) => {
                    if !global.quiet {
                        println!("⚠️  Metal GPU not available, falling back to CPU\n");
                    }
                    Device::Cpu
                }
            }
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            match Device::new_cuda(0) {
                Ok(d) => {
                    if !global.quiet {
                        println!("✓ Using CUDA GPU\n");
                    }
                    d
                }
                Err(_) => {
                    if !global.quiet {
                        println!("⚠️  CUDA GPU not available, falling back to CPU\n");
                    }
                    Device::Cpu
                }
            }
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            if !global.quiet {
                println!("⚠️  GPU requested but not compiled with GPU support, using CPU\n");
            }
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Load dataset
    if !global.quiet {
        println!("📚 Loading dataset from {:?}...", args.data);
    }

    let mut data_loader = VocoderDataLoader::load(&args.data).await?;

    if !global.quiet {
        println!("   ✓ Loaded {} audio samples\n", data_loader.len());
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Create model with VarMap for training (using V2 variant for balance of speed/quality)
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model_config = HiFiGanVariant::V2.default_config();

    if !global.quiet {
        println!("🔨 Creating HiFi-GAN V2 generator...");
    }

    let model = HiFiGanGenerator::new(model_config.clone(), vb)
        .map_err(|e| CliError::config(format!("Failed to create model: {}", e)))?;

    // Resume from checkpoint, if requested. Must happen after the model is
    // constructed (so every parameter name is already registered in `varmap`)
    // and before the optimizer is created.
    let mut start_epoch = 0usize;
    if let Some(ref resume_path) = args.resume {
        start_epoch = resume_from_checkpoint(&mut varmap, resume_path, "HiFiGan", global.quiet)?;
    }

    // Create optimizer
    let params = varmap.all_vars();
    // A second, independent snapshot of the same underlying parameter storage
    // (Var is a cheap Arc-like handle), used for gradient-norm computation and
    // clipping without fighting the optimizer for ownership of `params`.
    let train_vars = params.clone();
    let mut optimizer = AdamW::new_lr(params, args.lr)
        .map_err(|e| CliError::config(format!("Failed to create optimizer: {}", e)))?;

    // Calculate batches per epoch
    let batches_per_epoch = data_loader.len().div_ceil(args.batch_size);

    if !global.quiet {
        println!("✅ Training setup complete!\n");
        println!("📊 Model Information:");
        println!("   Variant: HiFi-GAN V2");
        println!("   Upsampling factor: {}x", model.total_upsampling_factor());
        println!("   Device: {:?}", device);
        println!("   Batches per epoch: {}", batches_per_epoch);
        println!("\n🚀 Starting HiFi-GAN generator training...\n");
        println!("   Note: This trains the generator with reconstruction loss.");
        println!(
            "   For full GAN training with discriminators, use a dedicated training script.\n"
        );
    }

    // Create progress tracker
    let mut progress = TrainingProgress::new(args.epochs, batches_per_epoch, !global.quiet);

    // Training statistics
    let start_time = Instant::now();
    let mut total_steps = 0;
    let mut best_val_loss = f64::MAX;
    // Epoch at which validation loss last improved. Early-stopping patience is
    // measured as real epochs elapsed since this point (see below), matching
    // `--patience`'s own "(epochs)" help text regardless of `--val-frequency`.
    // Initialized to `start_epoch` so a `--resume`d run doesn't inherit an
    // artificially stale improvement point from before the process restarted.
    let mut last_improvement_epoch = start_epoch;

    // Tracks genuine training-step failures across the whole run (never reset
    // between epochs). Crossing the threshold aborts the run with a real error
    // instead of silently continuing on fabricated data.
    let mut error_count: usize = 0;
    // Last real (non-fabricated) batch loss/gradient-norm, used only to keep the
    // live progress display continuous across a transient failure; never fed
    // into any aggregate/reported statistic.
    let mut last_known_batch_loss: Option<f64> = None;
    let mut last_known_grad_norm: Option<f64> = None;
    // Real per-epoch statistics from the most recently completed epoch, used for
    // the final summary and checkpoint instead of hardcoded literals.
    let mut last_completed_epoch: Option<usize> = None;
    let mut last_avg_epoch_loss: f64 = 0.0;
    let mut last_val_loss: Option<f64> = None;

    // Calculate total warmup steps
    let warmup_steps = args.training_config.warmup_steps;

    // Training loop
    for epoch in start_epoch..args.epochs {
        progress.start_epoch(epoch, batches_per_epoch);

        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut successful_batches: u64 = 0;

        // Reset data loader for new epoch
        data_loader.reset();

        // Batch loop
        for batch_idx in 0..batches_per_epoch {
            let batch_start = Instant::now();

            // Load batch data
            let batch_data = data_loader.get_batch(args.batch_size)?;

            // Convert batch to tensors
            let (audio_tensors, mel_tensors) = convert_batch_to_tensors(&batch_data, args.use_gpu)
                .map_err(|e| CliError::config(format!("Tensor conversion failed: {}", e)))?;

            // Training step: Generator reconstruction loss, with the same real
            // gradient-clipping mechanism used for DiffWave (see
            // `backward_step_with_grad_norm`).
            let (batch_loss, grad_norm) = match train_hifigan_step(
                &model,
                &mut optimizer,
                &train_vars,
                &audio_tensors,
                &mel_tensors,
                args.training_config.grad_clip,
            ) {
                Ok((loss, norm)) => {
                    epoch_loss += loss;
                    successful_batches += 1;
                    last_known_batch_loss = Some(loss);
                    last_known_grad_norm = Some(norm);
                    (loss, norm)
                }
                Err(e) => {
                    error_count += 1;
                    if !global.quiet {
                        eprintln!(
                            "⚠️  HiFi-GAN training step {}/{} failed: {}",
                            epoch + 1,
                            batch_idx + 1,
                            e
                        );
                    }
                    if should_abort_on_errors(error_count, batches_per_epoch) {
                        return Err(CliError::config(format!(
                            "Too many HiFi-GAN training-step failures ({error_count} failed \
                             out of {} batches attempted), aborting instead of fabricating \
                             training progress. Last error: {e}",
                            total_steps + 1
                        )));
                    }
                    // Do not fabricate a loss/grad-norm value: reuse the last
                    // real measurement for the live display only, or NaN if no
                    // batch has ever succeeded yet. Never added to epoch_loss.
                    (
                        last_known_batch_loss.unwrap_or(f64::NAN),
                        last_known_grad_norm.unwrap_or(f64::NAN),
                    )
                }
            };

            total_steps += 1;

            // Apply warmup to learning rate and push it into the real
            // optimizer, not just the display.
            if warmup_steps > 0 && total_steps <= warmup_steps {
                let warmup_lr = args.lr * (total_steps as f64 / warmup_steps as f64);
                optimizer.set_learning_rate(warmup_lr);
                if total_steps % 100 == 0 && !global.quiet {
                    println!(
                        "   🔥 Warmup: step {}/{}, lr: {:.6}",
                        total_steps,
                        warmup_steps,
                        optimizer.learning_rate()
                    );
                }
            }

            // Calculate samples per second
            let batch_duration = batch_start.elapsed().as_secs_f64();
            let samples_per_sec = (batch_data.len() as f64) / batch_duration;

            // Update progress
            progress.update_batch(batch_idx, batch_loss, samples_per_sec);

            // Update metrics every 10 batches
            if batch_idx % 10 == 0 {
                let metrics = TrainingMetrics {
                    loss: batch_loss,
                    // Read back from the optimizer itself so the displayed
                    // value can never drift from the rate actually applied.
                    learning_rate: optimizer.learning_rate(),
                    grad_norm: Some(grad_norm),
                };
                progress.update_metrics(&metrics);

                let resources = ResourceUsage::current();
                progress.update_resources(&resources);
            }

            progress.finish_batch();
        }

        // Calculate epoch metrics: divide by batches that actually succeeded, not
        // by batches_per_epoch, so a failed batch (excluded above) cannot dilute
        // the reported average toward a falsely lower loss.
        let avg_epoch_loss = epoch_loss / successful_batches.max(1) as f64;

        // Perform validation at specified frequency (HiFi-GAN specific)
        let val_loss = if epoch % args.training_config.val_frequency == 0 {
            // Use 10% of data for validation (or minimum 32 samples)
            let val_samples = (data_loader.len() / 10).max(32);
            Some(
                run_validation_hifigan(
                    &model,
                    &mut data_loader,
                    args.batch_size,
                    &device,
                    val_samples,
                )
                .await,
            )
        } else {
            None
        };

        // Record real per-epoch statistics for the final summary/checkpoint.
        last_completed_epoch = Some(epoch);
        last_avg_epoch_loss = avg_epoch_loss;
        if val_loss.is_some() {
            last_val_loss = val_loss;
        }

        // Update best validation loss and check early stopping. Patience is
        // counted in real epochs elapsed since the last improvement (not in
        // validation *events*), so it matches the flag's documented
        // "(epochs)" semantics regardless of --val-frequency.
        if let Some(vl) = val_loss {
            let improved = vl < (best_val_loss - args.training_config.min_delta);

            if improved {
                best_val_loss = vl;
                last_improvement_epoch = epoch;

                if !global.quiet {
                    println!("\n💾 New best model saved (val_loss: {:.4})", vl);
                }
                save_checkpoint(
                    &args.output,
                    "best_model",
                    epoch,
                    avg_epoch_loss,
                    Some(vl),
                    "HiFiGan",
                    &varmap,
                )
                .await?;
            } else if args.training_config.early_stopping {
                let epochs_without_improvement = epoch.saturating_sub(last_improvement_epoch);
                if epochs_without_improvement >= args.training_config.patience {
                    if !global.quiet {
                        println!(
                            "\n⚠️  Early stopping triggered after {} epochs without improvement",
                            epochs_without_improvement
                        );
                    }
                    break;
                }
            }
        }

        let epoch_metrics = EpochMetrics {
            epoch,
            train_loss: avg_epoch_loss,
            val_loss,
            duration: epoch_start.elapsed(),
        };

        progress.finish_epoch(&epoch_metrics);

        // Apply learning rate scheduler (only after warmup) and push it into
        // the real optimizer, not just the display.
        if args.training_config.lr_scheduler != "none" && total_steps > warmup_steps {
            let scheduled_lr = apply_lr_scheduler(
                &args.training_config.lr_scheduler,
                args.lr,
                epoch,
                args.training_config.lr_step_size,
                args.training_config.lr_gamma,
                args.epochs,
                epoch.saturating_sub(last_improvement_epoch),
            );
            optimizer.set_learning_rate(scheduled_lr);

            if !global.quiet && epoch % 10 == 0 {
                println!("   📊 Learning rate: {:.6}", optimizer.learning_rate());
            }
        }

        // Save checkpoint at specified frequency
        if epoch % args.training_config.save_frequency == 0 {
            save_checkpoint(
                &args.output,
                &format!("epoch_{}", epoch),
                epoch,
                avg_epoch_loss,
                val_loss,
                "HiFiGan",
                &varmap,
            )
            .await?;
            if !global.quiet {
                println!("\n💾 Checkpoint saved: epoch_{}.safetensors", epoch);
            }
        }
    }

    // Finish training
    let total_duration = start_time.elapsed();

    // Save final model and print the summary only if at least one epoch actually
    // completed; otherwise there is nothing real to report or persist.
    match last_completed_epoch {
        Some(final_epoch) => {
            save_checkpoint(
                &args.output,
                "final_model",
                final_epoch,
                last_avg_epoch_loss,
                last_val_loss,
                "HiFiGan",
                &varmap,
            )
            .await?;

            progress.finish("✅ HiFi-GAN generator training completed successfully!");

            if !global.quiet {
                let stats = TrainingStats {
                    total_duration,
                    epochs_completed: final_epoch + 1,
                    total_steps,
                    final_train_loss: last_avg_epoch_loss,
                    final_val_loss: last_val_loss,
                    best_val_loss: Some(best_val_loss),
                    avg_samples_per_sec: (total_steps * args.batch_size) as f64
                        / total_duration.as_secs_f64(),
                };
                progress.print_summary(&stats);

                println!("\n📊 Model outputs:");
                println!(
                    "   - Final model: {}/final_model.safetensors",
                    args.output.display()
                );
                println!(
                    "   - Best model:  {}/best_model.safetensors",
                    args.output.display()
                );
            }
        }
        None => {
            progress.finish("⚠️  No epochs were completed; nothing was trained.");
            if !global.quiet {
                println!(
                    "\n⚠️  Training requested {} epochs but none completed; no checkpoint \
                     was saved.",
                    args.epochs
                );
            }
        }
    }

    Ok(())
}

// Helper functions for training

/// Convert VocoderBatch to Candle tensors
fn convert_batch_to_tensors(
    batch: &super::data_loader::VocoderBatch,
    use_gpu: bool,
) -> std::result::Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let device = if use_gpu {
        // Try Metal first (macOS), then CUDA, then fallback to CPU
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            eprintln!("⚠️  GPU requested but neither Metal nor CUDA features enabled, using CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    // Convert audio Vec<Vec<f32>> to Tensor
    // Shape: (batch_size, max_audio_len)
    let max_audio_len = batch.audio.iter().map(|a| a.len()).max().unwrap_or(0);
    let batch_size = batch.audio.len();

    let mut audio_data = vec![0.0f32; batch_size * max_audio_len];
    for (i, audio) in batch.audio.iter().enumerate() {
        for (j, &sample) in audio.iter().enumerate() {
            audio_data[i * max_audio_len + j] = sample;
        }
    }

    let audio_tensor = Tensor::from_slice(&audio_data, (batch_size, max_audio_len), &device)?;

    // Convert mel Vec<Vec<Vec<f32>>> to Tensor
    // Shape: (batch_size, mel_channels, max_frames)
    let max_frames = batch.mels.iter().map(|m| m.len()).max().unwrap_or(0);
    let mel_channels = if batch.mels.is_empty() || batch.mels[0].is_empty() {
        80
    } else {
        batch.mels[0][0].len()
    };

    let mut mel_data = vec![0.0f32; batch_size * mel_channels * max_frames];
    for (i, mel) in batch.mels.iter().enumerate() {
        for (t, frame) in mel.iter().enumerate() {
            for (c, &value) in frame.iter().enumerate() {
                mel_data[i * mel_channels * max_frames + c * max_frames + t] = value;
            }
        }
    }

    let mel_tensor =
        Tensor::from_slice(&mel_data, (batch_size, mel_channels, max_frames), &device)?;

    Ok((audio_tensor, mel_tensor))
}

/// HiFi-GAN training step (generator-only with reconstruction loss). Returns
/// `(loss, real gradient global L2 norm)`.
fn train_hifigan_step(
    model: &voirs_vocoder::models::hifigan::generator::HiFiGanGenerator,
    optimizer: &mut AdamW,
    vars: &[Var],
    audio: &Tensor,
    mel: &Tensor,
    grad_clip: f64,
) -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
    // Forward pass: generate audio from mel spectrogram
    let generated_audio = model.forward(mel)?;

    // Reshape audio target to match generated shape
    // generated: (batch, 1, samples), target: (batch, samples) -> (batch, 1, samples)
    let target_audio = audio.unsqueeze(1)?;

    // Compute reconstruction loss (L1 + L2 combined)
    // L1 loss: mean(|generated - target|)
    let l1_diff = (generated_audio.sub(&target_audio))?.abs()?;
    let l1_loss = l1_diff.mean_all()?;

    // L2 loss: mean((generated - target)^2)
    let l2_diff = (generated_audio.sub(&target_audio))?;
    let l2_loss = l2_diff.sqr()?.mean_all()?;

    // Combined loss: 0.45 * L1 + 0.55 * L2 (typical for vocoders)
    let l1_weight = 0.45;
    let l2_weight = 0.55;
    let total_loss = (l1_loss.affine(l1_weight, 0.0)? + l2_loss.affine(l2_weight, 0.0)?)?;

    let loss_value = total_loss.to_vec0::<f32>()? as f64;

    // Backward pass, real global-norm gradient clipping (the same mechanism
    // used for DiffWave below), then the optimizer step.
    let grad_norm = backward_step_with_grad_norm(optimizer, vars, &total_loss, grad_clip)?;

    Ok((loss_value, grad_norm))
}

/// Real training step with DiffWave model. Returns `(loss, real gradient
/// global L2 norm)`.
fn train_step_real(
    model: &DiffWave,
    optimizer: &mut AdamW,
    vars: &[Var],
    audio: &Tensor,
    mel: &Tensor,
    device: &Device,
    grad_clip: f64,
) -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
    let batch_size = audio.dims()[0];

    // Generate random timesteps for diffusion (0 to 999)
    let timesteps: Vec<u32> = (0..batch_size).map(|_| fastrand::u32(0..1000)).collect();
    let timesteps = Tensor::from_vec(timesteps, (batch_size,), device)?;

    // Forward pass: get predicted noise and actual noise
    let (predicted_noise, actual_noise) = model.forward_with_target(audio, mel, &timesteps)?;

    // Compute MSE/L2 loss
    // Loss = mean((predicted_noise - actual_noise)^2)
    let diff = (predicted_noise - actual_noise)?;
    let loss_tensor = diff.sqr()?.mean_all()?;
    let loss_value = loss_tensor.to_vec0::<f32>()? as f64;

    // Backward pass, real global-norm gradient clipping, then the optimizer
    // step. See `backward_step_with_grad_norm` for the shared mechanism.
    let grad_norm = backward_step_with_grad_norm(optimizer, vars, &loss_tensor, grad_clip)?;

    Ok((loss_value, grad_norm))
}

/// Run a real backward pass for `loss`, compute the real global L2 norm of the
/// resulting gradients across `vars`, optionally rescale them so the norm
/// does not exceed `grad_clip` (disabled when `grad_clip <= 0.0`), then apply
/// the optimizer step. Returns the real, pre-clipping gradient norm -- the
/// value worth displaying, since it reflects what the model actually produced
/// before any clamping.
///
/// This is the single mechanism shared by both DiffWave's and HiFi-GAN's
/// training steps, replacing an earlier loss-scaling approximation that only
/// applied to DiffWave and was a complete no-op for HiFi-GAN. Standard
/// global-norm clipping (the same algorithm as
/// `torch.nn.utils.clip_grad_norm_`) is more correct than loss scaling and
/// applies identically to both models.
fn backward_step_with_grad_norm(
    optimizer: &mut AdamW,
    vars: &[Var],
    loss: &Tensor,
    grad_clip: f64,
) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    let mut grads = loss.backward()?;
    let grad_norm = clip_gradients(&mut grads, vars, grad_clip)?;
    optimizer.step(&grads)?;
    Ok(grad_norm)
}

/// Compute the real global L2 norm of the gradients in `grads` for `vars`
/// (accumulated on-device, with a single host sync at the end rather than one
/// per parameter tensor), and -- when `grad_clip > 0.0` and the norm exceeds
/// it -- rescale every gradient in place so the resulting norm equals
/// `grad_clip`. Always returns the norm as measured *before* any rescaling.
fn clip_gradients(
    grads: &mut GradStore,
    vars: &[Var],
    grad_clip: f64,
) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    let mut sum_sq: Option<Tensor> = None;
    for var in vars {
        if !var.dtype().is_float() {
            continue;
        }
        if let Some(g) = grads.get(var.as_tensor()) {
            let sq = g.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                Some(acc) => (acc + sq)?,
                None => sq,
            });
        }
    }

    let grad_norm = match sum_sq {
        Some(acc) => (acc.to_vec0::<f32>()? as f64).sqrt(),
        None => 0.0,
    };

    if grad_clip > 0.0 && grad_norm > grad_clip {
        let scale = grad_clip / (grad_norm + 1e-6);
        for var in vars {
            if !var.dtype().is_float() {
                continue;
            }
            if let Some(g) = grads.get(var.as_tensor()) {
                let scaled = (g * scale)?;
                grads.insert(var.as_tensor(), scaled);
            }
        }
    }

    Ok(grad_norm)
}

/// Save checkpoint to file
///
/// `val_loss` is `None` when no validation ran for this checkpoint (e.g. this
/// epoch fell outside `val_frequency`); the metadata records that honestly
/// (`null`) rather than substituting a fabricated number.
async fn save_checkpoint(
    output_dir: &Path,
    name: &str,
    epoch: usize,
    train_loss: f64,
    val_loss: Option<f64>,
    model_type: &str,
    varmap: &VarMap,
) -> Result<()> {
    use safetensors::tensor::{Dtype, SafeTensors};
    use serde_json::json;
    use std::collections::HashMap;

    let checkpoint_path = output_dir.join(format!("{}.safetensors", name));
    let val_loss_str = val_loss
        .map(|v| format!("{:.6}", v))
        .unwrap_or_else(|| "null".to_string());

    // Create checkpoint metadata
    let mut metadata = HashMap::new();
    metadata.insert("epoch".to_string(), epoch.to_string());
    metadata.insert("train_loss".to_string(), format!("{:.6}", train_loss));
    metadata.insert("val_loss".to_string(), val_loss_str.clone());
    metadata.insert(
        "timestamp".to_string(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs()
            .to_string(),
    );

    // Extract real model parameters from VarMap
    let mut tensors = Vec::new();

    // Scope the lock to ensure it's dropped before any await points
    {
        let varmap_data = varmap.data().lock().expect("lock should not be poisoned");
        for (name, var) in varmap_data.iter() {
            let tensor = var.as_tensor();
            let shape: Vec<usize> = tensor.dims().to_vec();

            // Convert tensor to Vec<f32>
            let data: Vec<f32> = tensor
                .flatten_all()
                .map_err(|e| CliError::config(format!("Failed to flatten tensor: {}", e)))?
                .to_vec1()
                .map_err(|e| CliError::config(format!("Failed to convert tensor to vec: {}", e)))?;

            tensors.push((name.clone(), (data, shape)));
        }
    } // Lock is automatically dropped here

    // Create SafeTensors format manually
    // SafeTensors format: [8 bytes header size][JSON header][tensor data]
    let mut safetensors_data = Vec::new();

    // Build header JSON
    let mut header = serde_json::Map::new();

    // Add metadata
    header.insert(
        "__metadata__".to_string(),
        json!({
            "epoch": epoch.to_string(),
            "train_loss": format!("{:.6}", train_loss),
            "val_loss": val_loss_str,
            "model_type": model_type,
        }),
    );

    // Add tensor information and collect data
    let mut tensor_data = Vec::new();
    let mut current_offset = 0usize;

    for (name, (data, shape)) in &tensors {
        let num_elements: usize = shape.iter().product();
        let data_size = num_elements * std::mem::size_of::<f32>();

        header.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + data_size]
            }),
        );

        // Convert f32 vec to bytes
        for &val in data {
            tensor_data.extend_from_slice(&val.to_le_bytes());
        }

        current_offset += data_size;
    }

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header)?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Write SafeTensors format: [header_len (8 bytes)][header JSON][tensor data]
    safetensors_data.extend_from_slice(&header_len.to_le_bytes());
    safetensors_data.extend_from_slice(header_bytes);
    safetensors_data.extend_from_slice(&tensor_data);

    // Write safetensors file
    tokio::fs::write(&checkpoint_path, &safetensors_data).await?;

    // Also save human-readable metadata
    let metadata_json = json!({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs(),
        "model_type": model_type,
        "tensors": tensors.iter().map(|(name, (_, shape))| {
            json!({
                "name": name,
                "shape": shape
            })
        }).collect::<Vec<_>>(),
    });

    let metadata_path = output_dir.join(format!("{}.json", name));
    tokio::fs::write(
        &metadata_path,
        serde_json::to_string_pretty(&metadata_json)?,
    )
    .await?;

    Ok(())
}

/// Perform real validation on validation dataset
///
/// Takes a subset of the data for validation and runs forward pass only (no optimization)
/// to calculate validation loss. This provides a true measure of generalization.
async fn run_validation(
    model: &DiffWave,
    data_loader: &mut super::data_loader::VocoderDataLoader,
    batch_size: usize,
    device: &Device,
    val_samples: usize,
) -> f64 {
    // Use a portion of data for validation (don't overlap with training batches)
    let val_batches = val_samples.div_ceil(batch_size);
    let mut total_val_loss = 0.0;
    let mut val_batch_count = 0;

    // Save current position in data loader
    let current_position = data_loader.current_index();

    // Perform validation on separate samples
    for _ in 0..val_batches {
        // Get validation batch
        if let Ok(batch_data) = data_loader.get_batch(batch_size) {
            // Convert to tensors
            if let Ok((audio_tensors, mel_tensors)) =
                convert_batch_to_tensors(&batch_data, device.is_cuda() || device.is_metal())
            {
                // Forward pass only (no backward/optimizer)
                if let Ok(loss) = validate_step_real(model, &audio_tensors, &mel_tensors, device) {
                    total_val_loss += loss;
                    val_batch_count += 1;
                }
            }
        }
    }

    // Restore data loader position for continued training
    data_loader.set_index(current_position);

    // Return average validation loss, or fallback if no valid batches
    if val_batch_count > 0 {
        total_val_loss / val_batch_count as f64
    } else {
        // Fallback: return a high loss indicating validation failed
        1.0
    }
}

/// Validation step: forward pass only without optimization (DiffWave)
fn validate_step_real(
    model: &DiffWave,
    audio: &Tensor,
    mel: &Tensor,
    device: &Device,
) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    let batch_size = audio.dims()[0];

    // Generate random timesteps for diffusion (0 to 999)
    let timesteps: Vec<u32> = (0..batch_size).map(|_| fastrand::u32(0..1000)).collect();
    let timesteps = Tensor::from_vec(timesteps, (batch_size,), device)?;

    // Forward pass only (no gradient computation needed)
    let (predicted_noise, actual_noise) = model.forward_with_target(audio, mel, &timesteps)?;

    // Compute MSE/L2 loss
    let diff = (predicted_noise - actual_noise)?;
    let loss_tensor = diff.sqr()?.mean_all()?;
    let loss_value = loss_tensor.to_vec0::<f32>()? as f64;

    Ok(loss_value)
}

/// Perform real validation for HiFi-GAN model
async fn run_validation_hifigan(
    model: &voirs_vocoder::models::hifigan::generator::HiFiGanGenerator,
    data_loader: &mut super::data_loader::VocoderDataLoader,
    batch_size: usize,
    device: &Device,
    val_samples: usize,
) -> f64 {
    // Use a portion of data for validation
    let val_batches = val_samples.div_ceil(batch_size);
    let mut total_val_loss = 0.0;
    let mut val_batch_count = 0;

    // Save current position in data loader
    let current_position = data_loader.current_index();

    // Perform validation on separate samples
    for _ in 0..val_batches {
        // Get validation batch
        if let Ok(batch_data) = data_loader.get_batch(batch_size) {
            // Convert to tensors
            if let Ok((audio_tensors, mel_tensors)) =
                convert_batch_to_tensors(&batch_data, device.is_cuda() || device.is_metal())
            {
                // Forward pass only (no backward/optimizer)
                if let Ok(loss) = validate_step_hifigan(model, &audio_tensors, &mel_tensors) {
                    total_val_loss += loss;
                    val_batch_count += 1;
                }
            }
        }
    }

    // Restore data loader position for continued training
    data_loader.set_index(current_position);

    // Return average validation loss, or fallback if no valid batches
    if val_batch_count > 0 {
        total_val_loss / val_batch_count as f64
    } else {
        1.0 // Fallback: return a high loss indicating validation failed
    }
}

/// Validation step: forward pass only without optimization (HiFi-GAN)
fn validate_step_hifigan(
    model: &voirs_vocoder::models::hifigan::generator::HiFiGanGenerator,
    audio: &Tensor,
    mel: &Tensor,
) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    // Forward pass: generate audio from mel spectrogram
    let generated_audio = model.forward(mel)?;

    // Reshape audio target to match generated shape
    let target_audio = audio.unsqueeze(1)?;

    // Compute reconstruction loss (L1 + L2 combined)
    let l1_diff = (generated_audio.sub(&target_audio))?.abs()?;
    let l1_loss = l1_diff.mean_all()?;

    let l2_diff = (generated_audio.sub(&target_audio))?;
    let l2_loss = l2_diff.sqr()?.mean_all()?;

    // Combined loss: 0.45 * L1 + 0.55 * L2
    let l1_weight = 0.45;
    let l2_weight = 0.55;
    let total_loss = (l1_loss.affine(l1_weight, 0.0)? + l2_loss.affine(l2_weight, 0.0)?)?;

    let loss_value = total_loss.to_vec0::<f32>()? as f64;

    Ok(loss_value)
}

fn truncate_path(path: &Path, max_len: usize) -> String {
    let path_str = path.display().to_string();
    if path_str.len() <= max_len {
        path_str
    } else {
        format!("...{}", &path_str[path_str.len() - (max_len - 3)..])
    }
}

/// Whether accumulated training-step failures are severe enough to abort the run
/// rather than continue. Mirrors the acoustic-training abort threshold (more than
/// half of a single epoch's batches worth of failures): past that point the run
/// can no longer be trusted, so it fails with a real error instead of silently
/// substituting fabricated loss values forever.
fn should_abort_on_errors(error_count: usize, batches_per_epoch: usize) -> bool {
    error_count > batches_per_epoch / 2
}

/// Resume training from a checkpoint written by `save_checkpoint`.
///
/// Must be called after the model has been constructed (so every parameter
/// name is already registered in `varmap`) and before the optimizer is
/// created. Loads real weights via `VarMap::load`; fails closed with a typed
/// error if the checkpoint file is missing, or if its tensor names/shapes
/// don't match this model (a partial or mismatched load is never accepted as
/// success -- `VarMap::load` itself errors out in that case).
///
/// `save_checkpoint` never persisted optimizer state (AdamW's per-parameter
/// moment buffers), only weights, so that state always restarts fresh. The
/// epoch counter is best-effort recovered from the checkpoint's `<name>.json`
/// sidecar (written alongside every checkpoint); if that sidecar is missing
/// or unparseable, the epoch counter honestly restarts at 0 too, and this is
/// logged rather than silently assumed.
///
/// Returns the epoch to resume training from.
fn resume_from_checkpoint(
    varmap: &mut VarMap,
    resume_path: &Path,
    model_name: &str,
    quiet: bool,
) -> Result<usize> {
    if !resume_path.exists() {
        return Err(CliError::config(format!(
            "Resume checkpoint not found: {}\n\n\
             Expected a .safetensors checkpoint file written by a previous \
             `voirs train vocoder --model-type {}` run.",
            resume_path.display(),
            model_name.to_lowercase()
        )));
    }

    varmap.load(resume_path).map_err(|e| {
        CliError::config(format!(
            "Failed to load resume checkpoint {}: {}\n\n\
             The checkpoint is missing, corrupted, or its tensor names/shapes do \
             not match a {model_name} model. Refusing to silently start training \
             from scratch.",
            resume_path.display(),
            e
        ))
    })?;

    let start_epoch = match resume_epoch_from_sidecar(resume_path) {
        Some(saved_epoch) => {
            // `saved_epoch` (0-indexed) is the last epoch the checkpoint
            // completed; resume with the next one.
            let next_epoch = saved_epoch + 1;
            if !quiet {
                println!(
                    "   ✓ Restored model weights from {}; resuming at epoch {} \
                     (optimizer state -- AdamW moment buffers -- is not persisted \
                     by this checkpoint format, so it restarts fresh)\n",
                    resume_path.display(),
                    // 1-indexed for display, matching the rest of the UI
                    // (e.g. progress.start_epoch's "Epoch {epoch+1}/{total}").
                    next_epoch + 1
                );
            }
            next_epoch
        }
        None => {
            if !quiet {
                println!(
                    "   ✓ Restored model weights from {} (no epoch metadata sidecar \
                     found alongside it; epoch counter restarts at 0, optimizer \
                     state restarts fresh)\n",
                    resume_path.display()
                );
            }
            0
        }
    };

    Ok(start_epoch)
}

/// Best-effort recovery of the epoch number a checkpoint was saved at, from
/// the `<name>.json` sidecar file `save_checkpoint` writes alongside every
/// `.safetensors` checkpoint. Returns `None` (never a fabricated number) when
/// the sidecar is missing or its `epoch` field can't be parsed.
fn resume_epoch_from_sidecar(checkpoint_path: &Path) -> Option<usize> {
    let json_path = checkpoint_path.with_extension("json");
    let content = std::fs::read_to_string(json_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&content).ok()?;
    value.get("epoch")?.as_u64().map(|e| e as usize)
}

/// Apply learning rate scheduler.
///
/// `epochs_since_improvement` is only used by the `"plateau"` variant (real
/// epochs elapsed since validation loss last improved, as tracked by the
/// caller's early-stopping bookkeeping); every other scheduler ignores it.
fn apply_lr_scheduler(
    scheduler_type: &str,
    initial_lr: f64,
    epoch: usize,
    step_size: usize,
    gamma: f64,
    total_epochs: usize,
    epochs_since_improvement: usize,
) -> f64 {
    match scheduler_type {
        "step" => {
            // StepLR: Multiply LR by gamma every step_size epochs
            let decay_factor = (epoch / step_size) as f64;
            initial_lr * gamma.powf(decay_factor)
        }
        "exponential" => {
            // ExponentialLR: Multiply LR by gamma every epoch
            initial_lr * gamma.powf(epoch as f64)
        }
        "cosine" => {
            // CosineAnnealingLR: Cosine annealing schedule
            let min_lr = initial_lr * 0.01; // Minimum learning rate
            min_lr
                + (initial_lr - min_lr)
                    * (1.0 + (std::f64::consts::PI * epoch as f64 / total_epochs as f64).cos())
                    / 2.0
        }
        "onecycle" => {
            // OneCycleLR: Increase then decrease
            let pct = epoch as f64 / total_epochs as f64;
            if pct < 0.5 {
                // Increasing phase
                initial_lr * (1.0 + pct * 2.0)
            } else {
                // Decreasing phase
                initial_lr * (3.0 - pct * 2.0)
            }
        }
        "plateau" => {
            // ReduceLROnPlateau-style: decay by `gamma` for every full
            // `step_size` epochs validation loss has gone without improving.
            // Purely a function of stagnation duration (not absolute epoch),
            // so it stays flat at `initial_lr` immediately after any
            // improvement and only starts decaying once a full window has
            // passed without one.
            if step_size == 0 || epochs_since_improvement == 0 {
                initial_lr
            } else {
                let decay_steps = (epochs_since_improvement / step_size) as f64;
                initial_lr * gamma.powf(decay_steps)
            }
        }
        _ => initial_lr,
    }
}

#[cfg(test)]
mod tests;
