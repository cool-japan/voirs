//! Checkpoint management commands
//!
//! Provides utilities for inspecting, managing, and converting model checkpoints.

use crate::GlobalOptions;
use clap::Subcommand;
use safetensors::SafeTensors;
use std::path::PathBuf;
use voirs_sdk::Result;

/// Checkpoint management subcommands
#[derive(Debug, Clone, Subcommand)]
pub enum CheckpointCommands {
    /// Inspect checkpoint file and show metadata
    Inspect {
        /// Path to checkpoint file (.safetensors)
        #[arg(value_name = "FILE")]
        checkpoint: PathBuf,

        /// Show detailed tensor information
        #[arg(long)]
        verbose: bool,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List all checkpoints in directory
    List {
        /// Directory to search for checkpoints
        #[arg(value_name = "DIR", default_value = "checkpoints")]
        directory: PathBuf,

        /// Sort by (name, epoch, loss, size, date)
        #[arg(long, default_value = "epoch")]
        sort_by: String,

        /// Show only best N checkpoints
        #[arg(long)]
        top: Option<usize>,
    },

    /// Compare two checkpoints
    Compare {
        /// First checkpoint file
        #[arg(value_name = "FILE1")]
        checkpoint1: PathBuf,

        /// Second checkpoint file
        #[arg(value_name = "FILE2")]
        checkpoint2: PathBuf,

        /// Show parameter differences
        #[arg(long)]
        diff_params: bool,
    },

    /// Convert checkpoint format
    Convert {
        /// Input checkpoint file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output checkpoint file
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,

        /// Input format (auto, safetensors, pytorch, onnx)
        #[arg(long, default_value = "auto")]
        input_format: String,

        /// Output format (safetensors, pytorch, onnx)
        #[arg(long, default_value = "safetensors")]
        output_format: String,
    },

    /// Prune checkpoints (keep only best/latest)
    Prune {
        /// Directory containing checkpoints
        #[arg(value_name = "DIR")]
        directory: PathBuf,

        /// Keep best N checkpoints by validation loss
        #[arg(long)]
        keep_best: Option<usize>,

        /// Keep latest N checkpoints
        #[arg(long)]
        keep_latest: Option<usize>,

        /// Dry run (don't actually delete)
        #[arg(long)]
        dry_run: bool,
    },
}

/// Execute checkpoint management command
pub async fn execute_checkpoint_command(
    command: CheckpointCommands,
    global: &GlobalOptions,
) -> Result<()> {
    match command {
        CheckpointCommands::Inspect {
            checkpoint,
            verbose,
            format,
        } => inspect_checkpoint(&checkpoint, verbose, &format, global).await,
        CheckpointCommands::List {
            directory,
            sort_by,
            top,
        } => list_checkpoints(&directory, &sort_by, top, global).await,
        CheckpointCommands::Compare {
            checkpoint1,
            checkpoint2,
            diff_params,
        } => compare_checkpoints(&checkpoint1, &checkpoint2, diff_params, global).await,
        CheckpointCommands::Convert {
            input,
            output,
            input_format,
            output_format,
        } => convert_checkpoint(&input, &output, &input_format, &output_format, global).await,
        CheckpointCommands::Prune {
            directory,
            keep_best,
            keep_latest,
            dry_run,
        } => prune_checkpoints(&directory, keep_best, keep_latest, dry_run, global).await,
    }
}

/// Inspect a checkpoint file
async fn inspect_checkpoint(
    checkpoint_path: &PathBuf,
    verbose: bool,
    format: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !checkpoint_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Checkpoint file not found: {}",
            checkpoint_path.display()
        )));
    }

    // Read checkpoint file
    let data = tokio::fs::read(checkpoint_path).await?;
    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to parse checkpoint: {}", e))
    })?;

    // Try to load metadata from companion .json file
    let json_path = checkpoint_path.with_extension("json");
    let metadata = if json_path.exists() {
        tokio::fs::read_to_string(&json_path)
            .await
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };

    if format == "json" {
        output_json_format(&tensors, checkpoint_path, verbose, metadata.as_ref())?;
    } else {
        output_text_format(&tensors, checkpoint_path, verbose, global, metadata.as_ref())?;
    }

    Ok(())
}

/// Output checkpoint info in text format
fn output_text_format(
    tensors: &SafeTensors,
    checkpoint_path: &PathBuf,
    verbose: bool,
    global: &GlobalOptions,
    metadata: Option<&serde_json::Value>,
) -> Result<()> {
    if !global.quiet {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              Checkpoint Inspection                       ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ File: {:<50} ║", truncate_str(&checkpoint_path.display().to_string(), 50));

        // Display metadata if available
        if let Some(meta_val) = metadata {
            if let Some(obj) = meta_val.as_object() {
                for (key, value) in obj {
                    if key != "tensors" {
                        // Skip tensors array in metadata
                        let value_str = match value {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            _ => value.to_string(),
                        };
                        println!("║ {}: {:<47} ║", key, truncate_str(&value_str, 47 - key.len()));
                    }
                }
            }
        }

        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Tensors: {:<47} ║", tensors.names().len());

        // Calculate total parameters
        let mut total_params: usize = 0;
        let mut total_size: usize = 0;

        for name in tensors.names() {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                let params: usize = shape.iter().product();
                total_params += params;
                total_size += tensor.data().len();
            }
        }

        println!("║ Total parameters: {:<38} ║", format_number(total_params));
        println!(
            "║ Total size: {:<44} ║",
            format_bytes(total_size)
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        if verbose {
            println!("\n📊 Tensor Details:\n");
            println!("{:<50} {:>15} {:>15}", "Name", "Shape", "Parameters");
            println!("{}", "─".repeat(82));

            for name in tensors.names() {
                if let Ok(tensor) = tensors.tensor(name) {
                    let shape = tensor.shape();
                    let params: usize = shape.iter().product();
                    let shape_str = format!("{:?}", shape);

                    println!(
                        "{:<50} {:>15} {:>15}",
                        truncate_str(name, 50),
                        truncate_str(&shape_str, 15),
                        format_number(params)
                    );
                }
            }
            println!();
        }
    }

    Ok(())
}

/// Output checkpoint info in JSON format
fn output_json_format(
    tensors: &SafeTensors,
    checkpoint_path: &PathBuf,
    verbose: bool,
    metadata: Option<&serde_json::Value>,
) -> Result<()> {
    use serde_json::json;

    let mut tensor_info = Vec::new();
    let mut total_params: usize = 0;

    for name in tensors.names() {
        if let Ok(tensor) = tensors.tensor(name) {
            let shape: Vec<usize> = tensor.shape().to_vec();
            let params: usize = shape.iter().product();
            total_params += params;

            if verbose {
                tensor_info.push(json!({
                    "name": name,
                    "shape": shape,
                    "parameters": params,
                    "dtype": "F32",
                }));
            }
        }
    }

    let output = json!({
        "file": checkpoint_path.display().to_string(),
        "num_tensors": tensors.names().len(),
        "total_parameters": total_params,
        "metadata": metadata,
        "tensors": if verbose { Some(tensor_info) } else { None },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

/// List checkpoints in a directory
async fn list_checkpoints(
    directory: &PathBuf,
    sort_by: &str,
    top: Option<usize>,
    global: &GlobalOptions,
) -> Result<()> {
    if !directory.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Directory not found: {}",
            directory.display()
        )));
    }

    let mut checkpoints = Vec::new();

    // Read all .safetensors files
    let mut entries = tokio::fs::read_dir(directory).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            if let Ok(metadata) = entry.metadata().await {
                // Try to read checkpoint metadata from companion .json file
                let json_path = path.with_extension("json");
                let mut epoch = 0;
                let mut train_loss = 0.0;
                let mut val_loss = 0.0;

                if json_path.exists() {
                    if let Ok(meta_str) = tokio::fs::read_to_string(&json_path).await {
                        if let Ok(meta_json) =
                            serde_json::from_str::<serde_json::Value>(&meta_str)
                        {
                            if let Some(obj) = meta_json.as_object() {
                                epoch = obj
                                    .get("epoch")
                                    .and_then(|v| {
                                        v.as_u64()
                                            .map(|n| n as usize)
                                            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                                    })
                                    .unwrap_or(0);
                                train_loss = obj
                                    .get("train_loss")
                                    .and_then(|v| {
                                        v.as_f64().or_else(|| {
                                            v.as_str().and_then(|s| s.parse().ok())
                                        })
                                    })
                                    .unwrap_or(0.0);
                                val_loss = obj
                                    .get("val_loss")
                                    .and_then(|v| {
                                        v.as_f64().or_else(|| {
                                            v.as_str().and_then(|s| s.parse().ok())
                                        })
                                    })
                                    .unwrap_or(0.0);
                            }
                        }
                    }
                }

                if let Ok(data) = tokio::fs::read(&path).await {
                    if SafeTensors::deserialize(&data).is_ok() {

                        checkpoints.push(CheckpointInfo {
                            path: path.clone(),
                            name: path.file_name().unwrap().to_string_lossy().to_string(),
                            epoch,
                            train_loss,
                            val_loss,
                            size: metadata.len(),
                            modified: metadata
                                .modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                        });
                    }
                }
            }
        }
    }

    // Sort checkpoints
    match sort_by {
        "name" => checkpoints.sort_by(|a, b| a.name.cmp(&b.name)),
        "epoch" => checkpoints.sort_by(|a, b| b.epoch.cmp(&a.epoch)),
        "loss" => checkpoints.sort_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap()),
        "size" => checkpoints.sort_by(|a, b| b.size.cmp(&a.size)),
        "date" => checkpoints.sort_by(|a, b| b.modified.cmp(&a.modified)),
        _ => {}
    }

    // Limit to top N
    if let Some(n) = top {
        checkpoints.truncate(n);
    }

    if !global.quiet {
        println!("\n📁 Checkpoints in {}:\n", directory.display());
        println!(
            "{:<35} {:>8} {:>12} {:>12} {:>10}",
            "Name", "Epoch", "Train Loss", "Val Loss", "Size"
        );
        println!("{}", "─".repeat(82));

        for ckpt in &checkpoints {
            println!(
                "{:<35} {:>8} {:>12.6} {:>12.6} {:>10}",
                truncate_str(&ckpt.name, 35),
                ckpt.epoch,
                ckpt.train_loss,
                ckpt.val_loss,
                format_bytes(ckpt.size as usize)
            );
        }

        println!("\nTotal: {} checkpoints\n", checkpoints.len());
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CheckpointInfo {
    path: PathBuf,
    name: String,
    epoch: usize,
    train_loss: f64,
    val_loss: f64,
    size: u64,
    modified: u64,
}

/// Compare two checkpoints
async fn compare_checkpoints(
    checkpoint1: &PathBuf,
    checkpoint2: &PathBuf,
    diff_params: bool,
    global: &GlobalOptions,
) -> Result<()> {
    let data1 = tokio::fs::read(checkpoint1).await?;
    let data2 = tokio::fs::read(checkpoint2).await?;

    let tensors1 = SafeTensors::deserialize(&data1).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to parse checkpoint 1: {}", e))
    })?;

    let tensors2 = SafeTensors::deserialize(&data2).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to parse checkpoint 2: {}", e))
    })?;

    // Try to load metadata from companion .json files
    let json_path1 = checkpoint1.with_extension("json");
    let json_path2 = checkpoint2.with_extension("json");

    let meta1 = if json_path1.exists() {
        tokio::fs::read_to_string(&json_path1)
            .await
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };

    let meta2 = if json_path2.exists() {
        tokio::fs::read_to_string(&json_path2)
            .await
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };

    if !global.quiet {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              Checkpoint Comparison                       ║");
        println!("╠══════════════════════════════════════════════════════════╣");

        // Compare metadata
        if let (Some(m1), Some(m2)) = (meta1.as_ref(), meta2.as_ref()) {
            if let (Some(o1), Some(o2)) = (m1.as_object(), m2.as_object()) {
                println!("║ {:<25} {:<12} {:<15} ║", "Metric", "Checkpoint 1", "Checkpoint 2");
                println!("╠══════════════════════════════════════════════════════════╣");

                for key in o1.keys() {
                    if key != "tensors" {
                        // Skip tensors array
                        if let (Some(v1), Some(v2)) = (o1.get(key), o2.get(key)) {
                            let s1 = match v1 {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Number(n) => n.to_string(),
                                _ => v1.to_string(),
                            };
                            let s2 = match v2 {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Number(n) => n.to_string(),
                                _ => v2.to_string(),
                            };

                            println!(
                                "║ {:<25} {:<12} {:<15} ║",
                                truncate_str(key, 25),
                                truncate_str(&s1, 12),
                                truncate_str(&s2, 15)
                            );
                        }
                    }
                }
            }
        }

        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Tensors in checkpoint 1: {:<31} ║", tensors1.names().len());
        println!("║ Tensors in checkpoint 2: {:<31} ║", tensors2.names().len());
        println!("╚══════════════════════════════════════════════════════════╝\n");

        if diff_params {
            // Show parameter differences
            let names1: std::collections::HashSet<String> =
                tensors1.names().iter().map(|s| s.to_string()).collect();
            let names2: std::collections::HashSet<String> =
                tensors2.names().iter().map(|s| s.to_string()).collect();

            let only_in_1: Vec<_> = names1.difference(&names2).collect();
            let only_in_2: Vec<_> = names2.difference(&names1).collect();

            if !only_in_1.is_empty() {
                println!("⚠️  Tensors only in checkpoint 1:");
                for name in only_in_1 {
                    println!("   - {}", name);
                }
                println!();
            }

            if !only_in_2.is_empty() {
                println!("⚠️  Tensors only in checkpoint 2:");
                for name in only_in_2 {
                    println!("   - {}", name);
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Convert checkpoint format
async fn convert_checkpoint(
    input: &PathBuf,
    output: &PathBuf,
    input_format: &str,
    output_format: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !input.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Input checkpoint not found: {}",
            input.display()
        )));
    }

    // Auto-detect input format if needed
    let detected_input_format = if input_format == "auto" {
        match input.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => "safetensors",
            Some("pt") | Some("pth") => "pytorch",
            Some("onnx") => "onnx",
            _ => {
                return Err(voirs_sdk::VoirsError::config_error(
                    "Could not auto-detect input format. Please specify --input-format",
                ));
            }
        }
    } else {
        input_format
    };

    if !global.quiet {
        println!("\n🔄 Checkpoint Conversion:");
        println!("   Input:  {} ({})", input.display(), detected_input_format);
        println!("   Output: {} ({})", output.display(), output_format);
        println!();
    }

    // Handle conversion based on format pair
    match (detected_input_format, output_format) {
        ("safetensors", "safetensors") => {
            convert_safetensors_to_safetensors(input, output, global).await
        }
        ("safetensors", "pytorch") => {
            Err(voirs_sdk::VoirsError::config_error(
                "SafeTensors to PyTorch conversion not yet implemented. Consider using Python: \
                import safetensors.torch; safetensors.torch.save_file(tensors, 'output.pt')",
            ))
        }
        ("safetensors", "onnx") => {
            Err(voirs_sdk::VoirsError::config_error(
                "SafeTensors to ONNX conversion not supported. ONNX requires model architecture definition.",
            ))
        }
        ("pytorch", "safetensors") => {
            Err(voirs_sdk::VoirsError::config_error(
                "PyTorch to SafeTensors conversion not yet implemented. Consider using Python: \
                import safetensors.torch; safetensors.torch.save_file(torch.load('input.pt'), 'output.safetensors')",
            ))
        }
        ("pytorch", "pytorch") => {
            // Simple copy with potential re-serialization
            tokio::fs::copy(input, output).await?;
            if !global.quiet {
                println!("✅ Checkpoint copied successfully");
            }
            Ok(())
        }
        ("onnx", _) => {
            Err(voirs_sdk::VoirsError::config_error(
                "ONNX checkpoint conversion not supported. ONNX models are runtime-optimized formats.",
            ))
        }
        _ => {
            Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported conversion: {} to {}",
                detected_input_format, output_format
            )))
        }
    }
}

/// Convert SafeTensors to SafeTensors (with potential metadata updates)
async fn convert_safetensors_to_safetensors(
    input: &PathBuf,
    output: &PathBuf,
    global: &GlobalOptions,
) -> Result<()> {
    // Read input checkpoint
    let data = tokio::fs::read(input).await?;
    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to parse input checkpoint: {}", e))
    })?;

    // Read metadata if available
    let json_path = input.with_extension("json");
    let metadata = if json_path.exists() {
        tokio::fs::read_to_string(&json_path)
            .await
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };

    // For now, just copy the file and metadata
    tokio::fs::copy(input, output).await?;

    if let Some(ref meta) = metadata {
        let output_json = output.with_extension("json");
        tokio::fs::write(&output_json, serde_json::to_string_pretty(meta)?).await?;
    }

    if !global.quiet {
        println!("✅ SafeTensors checkpoint converted successfully");
        println!("   Tensors: {}", tensors.names().len());

        if metadata.is_some() {
            println!("   Metadata copied: {}", output.with_extension("json").display());
        }
    }

    Ok(())
}

/// Prune checkpoints
async fn prune_checkpoints(
    directory: &PathBuf,
    keep_best: Option<usize>,
    keep_latest: Option<usize>,
    dry_run: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !directory.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Directory not found: {}",
            directory.display()
        )));
    }

    if keep_best.is_none() && keep_latest.is_none() {
        return Err(voirs_sdk::VoirsError::config_error(
            "Must specify at least one of --keep-best or --keep-latest",
        ));
    }

    // Collect all checkpoints
    let mut checkpoints = Vec::new();
    let mut entries = tokio::fs::read_dir(directory).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            if let Ok(metadata) = entry.metadata().await {
                let json_path = path.with_extension("json");
                let mut epoch = 0;
                let mut train_loss = 0.0;
                let mut val_loss = f64::MAX;

                if json_path.exists() {
                    if let Ok(meta_str) = tokio::fs::read_to_string(&json_path).await {
                        if let Ok(meta_json) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                            if let Some(obj) = meta_json.as_object() {
                                epoch = obj.get("epoch")
                                    .and_then(|v| v.as_u64().map(|n| n as usize)
                                        .or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                                    .unwrap_or(0);
                                train_loss = obj.get("train_loss")
                                    .and_then(|v| v.as_f64()
                                        .or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                                    .unwrap_or(0.0);
                                val_loss = obj.get("val_loss")
                                    .and_then(|v| v.as_f64()
                                        .or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                                    .unwrap_or(f64::MAX);
                            }
                        }
                    }
                }

                if let Ok(data) = tokio::fs::read(&path).await {
                    if SafeTensors::deserialize(&data).is_ok() {
                        checkpoints.push(CheckpointInfo {
                            path: path.clone(),
                            name: path.file_name().unwrap().to_string_lossy().to_string(),
                            epoch,
                            train_loss,
                            val_loss,
                            size: metadata.len(),
                            modified: metadata.modified().ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                        });
                    }
                }
            }
        }
    }

    if checkpoints.is_empty() {
        if !global.quiet {
            println!("No checkpoints found in {}", directory.display());
        }
        return Ok(());
    }

    let mut to_delete = Vec::new();

    // Determine which checkpoints to keep
    if let Some(n) = keep_best {
        // Sort by validation loss (ascending - lower is better)
        let mut sorted = checkpoints.clone();
        sorted.sort_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap_or(std::cmp::Ordering::Equal));

        // Keep best N, mark rest for deletion
        let to_keep: std::collections::HashSet<_> = sorted.iter()
            .take(n)
            .map(|c| c.path.clone())
            .collect();

        for ckpt in &checkpoints {
            if !to_keep.contains(&ckpt.path) {
                to_delete.push(ckpt.clone());
            }
        }
    }

    if let Some(n) = keep_latest {
        // Sort by modification time (descending - newer first)
        let mut sorted = checkpoints.clone();
        sorted.sort_by(|a, b| b.modified.cmp(&a.modified));

        // Keep latest N
        let to_keep: std::collections::HashSet<_> = sorted.iter()
            .take(n)
            .map(|c| c.path.clone())
            .collect();

        // Only delete if not already marked and not in keep set
        for ckpt in &checkpoints {
            if !to_keep.contains(&ckpt.path) && !to_delete.iter().any(|d| d.path == ckpt.path) {
                to_delete.push(ckpt.clone());
            }
        }
    }

    if to_delete.is_empty() {
        if !global.quiet {
            println!("✅ No checkpoints need to be pruned");
        }
        return Ok(());
    }

    if !global.quiet {
        println!("\n🗑️  Checkpoint Pruning:\n");
        println!("Total checkpoints: {}", checkpoints.len());
        println!("To delete: {}", to_delete.len());

        if dry_run {
            println!("\n⚠️  DRY RUN - No files will be deleted\n");
        }

        println!("\nCheckpoints to be deleted:");
        println!("{:<35} {:>8} {:>12} {:>10}", "Name", "Epoch", "Val Loss", "Size");
        println!("{}", "─".repeat(70));

        for ckpt in &to_delete {
            println!(
                "{:<35} {:>8} {:>12.6} {:>10}",
                truncate_str(&ckpt.name, 35),
                ckpt.epoch,
                if ckpt.val_loss == f64::MAX { 0.0 } else { ckpt.val_loss },
                format_bytes(ckpt.size as usize)
            );
        }
        println!();
    }

    if !dry_run {
        let mut deleted_count = 0;
        for ckpt in &to_delete {
            // Delete .safetensors file
            if let Err(e) = tokio::fs::remove_file(&ckpt.path).await {
                if !global.quiet {
                    eprintln!("⚠️  Failed to delete {}: {}", ckpt.name, e);
                }
            } else {
                deleted_count += 1;

                // Also delete companion .json file if exists
                let json_path = ckpt.path.with_extension("json");
                if json_path.exists() {
                    let _ = tokio::fs::remove_file(&json_path).await;
                }
            }
        }

        if !global.quiet {
            println!("✅ Deleted {} checkpoint(s)", deleted_count);
        }
    }

    Ok(())
}

// Helper functions

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}
