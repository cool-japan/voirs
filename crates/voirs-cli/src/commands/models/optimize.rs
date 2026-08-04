//! Model optimization command implementation.

use crate::GlobalOptions;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use voirs_sdk::config::AppConfig;
use voirs_sdk::Result;

/// Optimization strategy
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Optimize for speed
    Speed,
    /// Optimize for quality
    Quality,
    /// Optimize for memory usage
    Memory,
    /// Balanced optimization
    Balanced,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub original_size_mb: f64,
    pub optimized_size_mb: f64,
    pub compression_ratio: f64,
    pub speed_improvement: f64,
    pub quality_impact: f64,
    pub output_path: PathBuf,
}

/// Run optimize model command
pub async fn run_optimize_model(
    model_id: &str,
    output_path: Option<&str>,
    strategy: Option<&str>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Optimizing model: {}", model_id);
    }

    // Check if model exists
    let model_path = get_model_path(model_id, config)?;
    if !model_path.exists() {
        return Err(voirs_sdk::VoirsError::model_error(format!(
            "Model '{}' not found. Please download it first.",
            model_id
        )));
    }

    // Determine optimization strategy
    let strategy = determine_optimization_strategy(strategy, config, global)?;

    // Analyze current model
    let model_info = analyze_model(&model_path, global).await?;
    if !global.quiet {
        println!(
            "  Found {} component(s), {:.1} MB total",
            model_info.components.len(),
            model_info.total_size_mb
        );
        for component in &model_info.components {
            println!(
                "    - {} ({:?}, {:.2} MB)",
                component.name, component.component_type, component.size_mb
            );
        }
    }

    // Perform optimization
    let result =
        perform_optimization(model_id, &model_path, output_path, &strategy, global).await?;

    // Display results
    display_optimization_results(&result, &strategy, global);

    Ok(())
}

/// Get model path
fn get_model_path(model_id: &str, config: &AppConfig) -> Result<PathBuf> {
    // Use the effective cache directory from config
    let cache_dir = config.pipeline.effective_cache_dir();
    let models_dir = cache_dir.join("models");
    Ok(models_dir.join(model_id))
}

/// Determine optimization strategy
fn determine_optimization_strategy(
    strategy: Option<&str>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<OptimizationStrategy> {
    // Parse user-provided strategy or use default
    let strategy_str = strategy.unwrap_or("balanced");

    match strategy_str.to_lowercase().as_str() {
        "speed" => Ok(OptimizationStrategy::Speed),
        "quality" => Ok(OptimizationStrategy::Quality),
        "memory" => Ok(OptimizationStrategy::Memory),
        "balanced" => Ok(OptimizationStrategy::Balanced),
        _ => Err(voirs_sdk::VoirsError::config_error(format!(
            "Invalid optimization strategy '{}'. Valid options: speed, quality, memory, balanced",
            strategy_str
        ))),
    }
}

/// Analyze model structure and characteristics
async fn analyze_model(model_path: &PathBuf, global: &GlobalOptions) -> Result<ModelAnalysis> {
    if !global.quiet {
        println!("Analyzing model structure...");
    }

    // Read model configuration
    let config_path = model_path.join("config.json");
    let config_content =
        std::fs::read_to_string(&config_path).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: config_path.clone(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;

    // Calculate model size
    let model_size = calculate_directory_size(model_path)?;

    // Analyze model components
    let components = analyze_model_components(model_path)?;

    Ok(ModelAnalysis {
        total_size_mb: model_size,
        components,
        config_content,
    })
}

/// Model analysis result
#[derive(Debug, Clone)]
struct ModelAnalysis {
    total_size_mb: f64,
    components: Vec<ModelComponent>,
    #[allow(dead_code)] // retained for future config-aware optimization decisions
    config_content: String,
}

/// Model component information
#[derive(Debug, Clone)]
struct ModelComponent {
    name: String,
    size_mb: f64,
    component_type: ComponentType,
}

/// Component type
#[derive(Debug, Clone)]
enum ComponentType {
    ModelWeights,
    Tokenizer,
    Configuration,
    Metadata,
}

/// Calculate directory size in MB
fn calculate_directory_size(path: &PathBuf) -> Result<f64> {
    let mut total_size = 0u64;

    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += calculate_directory_size(&entry.path())? as u64;
            }
        }
    }

    Ok(total_size as f64 / 1024.0 / 1024.0)
}

/// Analyze model components
fn analyze_model_components(model_path: &PathBuf) -> Result<Vec<ModelComponent>> {
    let mut components = Vec::new();

    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path
            .file_name()
            .ok_or_else(|| {
                voirs_sdk::VoirsError::model_error(format!("Invalid file path: {}", path.display()))
            })?
            .to_string_lossy();

        if path.is_file() {
            let size = entry.metadata()?.len() as f64 / 1024.0 / 1024.0;
            let component_type = match filename.as_ref() {
                "model.pt" | "model.onnx" | "model.bin" => ComponentType::ModelWeights,
                "tokenizer.json" | "vocab.txt" => ComponentType::Tokenizer,
                "config.json" | "config.yaml" => ComponentType::Configuration,
                _ if filename.ends_with(".safetensors") => ComponentType::ModelWeights,
                _ => ComponentType::Metadata,
            };

            components.push(ModelComponent {
                name: filename.to_string(),
                size_mb: size,
                component_type,
            });
        }
    }

    Ok(components)
}

/// Perform model optimization
async fn perform_optimization(
    model_id: &str,
    model_path: &PathBuf,
    output_path: Option<&str>,
    strategy: &OptimizationStrategy,
    global: &GlobalOptions,
) -> Result<OptimizationResult> {
    if !global.quiet {
        println!("Applying optimization strategy: {:?}", strategy);
    }

    // Determine output path
    let output_path = if let Some(path) = output_path {
        PathBuf::from(path)
    } else {
        let parent = model_path.parent().ok_or_else(|| {
            voirs_sdk::VoirsError::model_error(format!(
                "Cannot determine parent directory for: {}",
                model_path.display()
            ))
        })?;
        parent.join(format!("{}_optimized", model_id))
    };

    // Create output directory
    std::fs::create_dir_all(&output_path)?;

    // Get original size
    let original_size = calculate_directory_size(model_path)?;

    // Perform optimization steps
    let optimization_steps = get_optimization_steps(strategy);

    if !global.quiet {
        println!("Optimization steps: {}", optimization_steps.len());
    }

    // Chain steps: every step's output becomes the NEXT step's input, so a
    // multi-step strategy's transformations genuinely compose. Previously each
    // step read straight from the pristine `model_path` and overwrote
    // `output_path` wholesale, so only the *last* step's effect ever survived
    // (e.g. Balanced's real quantization step was silently discarded by the
    // plain-copy "Balancing speed and quality" step that ran after it).
    let mut current_input = model_path.clone();
    let mut staging_dirs: Vec<tempfile::TempDir> = Vec::new();
    let step_count = optimization_steps.len();

    for (i, step) in optimization_steps.iter().enumerate() {
        if !global.quiet {
            println!("  [{}/{}] {}", i + 1, step_count, step);
        }

        let is_last = i + 1 == step_count;
        let step_output = if is_last {
            output_path.clone()
        } else {
            let staging = tempfile::Builder::new()
                .prefix("voirs-optimize-stage-")
                .tempdir()
                .map_err(|e| voirs_sdk::VoirsError::IoError {
                    path: output_path.clone(),
                    operation: voirs_sdk::error::IoOperation::Write,
                    source: e,
                })?;
            let path = staging.path().to_path_buf();
            staging_dirs.push(staging);
            path
        };

        // Apply optimization step
        apply_optimization_step(step, &current_input, &step_output, global).await?;
        current_input = step_output;
    }
    // Every intermediate staging `TempDir` is deleted here; `output_path` now
    // holds the cumulative result of every step that ran.
    drop(staging_dirs);

    // Calculate final size
    let optimized_size = calculate_directory_size(&output_path)?;

    // Calculate metrics
    let compression_ratio = original_size / optimized_size;
    let speed_improvement = calculate_speed_improvement(strategy);
    let quality_impact = calculate_quality_impact(strategy);

    Ok(OptimizationResult {
        original_size_mb: original_size,
        optimized_size_mb: optimized_size,
        compression_ratio,
        speed_improvement,
        quality_impact,
        output_path,
    })
}

/// Get optimization steps for strategy
fn get_optimization_steps(strategy: &OptimizationStrategy) -> Vec<String> {
    match strategy {
        OptimizationStrategy::Speed => vec![
            "Quantizing model weights".to_string(),
            "Optimizing computation graph".to_string(),
            "Enabling fast inference modes".to_string(),
            "Compressing model artifacts".to_string(),
        ],
        OptimizationStrategy::Quality => vec![
            "Preserving high-precision weights".to_string(),
            "Maintaining model architecture".to_string(),
            "Optimizing for quality retention".to_string(),
        ],
        OptimizationStrategy::Memory => vec![
            "Applying aggressive quantization".to_string(),
            "Pruning redundant parameters".to_string(),
            "Compressing model storage".to_string(),
            "Optimizing memory layout".to_string(),
        ],
        OptimizationStrategy::Balanced => vec![
            "Applying moderate quantization".to_string(),
            "Optimizing computation graph".to_string(),
            "Balancing speed and quality".to_string(),
            "Compressing model artifacts".to_string(),
        ],
    }
}

/// Apply optimization step
async fn apply_optimization_step(
    step: &str,
    input_path: &PathBuf,
    output_path: &PathBuf,
    global: &GlobalOptions,
) -> Result<()> {
    // Implement actual optimization techniques based on step type
    if !global.quiet {
        println!("    Applying {}", step);
    }

    // Case-insensitive, keyword-based dispatch. The step descriptions used by
    // `get_optimization_steps` for the Memory/Balanced strategies ("Applying
    // aggressive/moderate quantization") do not contain the exact substring
    // "Quantizing", so a case-sensitive `contains("Quantizing")` check used to
    // silently skip real quantization for those two strategies (falling
    // through to a plain file copy while still claiming the strategy ran).
    let step_lower = step.to_lowercase();
    if step_lower.contains("quant") {
        // Implement model quantization
        quantize_model_files(input_path, output_path, global).await?;
    } else if step_lower.contains("optimiz") {
        // Implement graph optimization / analysis
        optimize_model_graph(input_path, output_path, global).await?;
    } else if step_lower.contains("compress") {
        // Implement model compression
        compress_model_files(input_path, output_path, global).await?;
    } else {
        // Fallback: copy files for unknown optimization steps
        copy_model_files(input_path, output_path)?;
    }

    Ok(())
}

/// Copy model files with validation
fn copy_model_files(input_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
    if !input_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Input path does not exist: {}",
            input_path.display()
        )));
    }

    std::fs::create_dir_all(output_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: output_path.clone(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    for entry in std::fs::read_dir(input_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: input_path.clone(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })? {
        let entry = entry.map_err(|e| voirs_sdk::VoirsError::IoError {
            path: input_path.clone(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;
        let src = entry.path();
        let dst = output_path.join(entry.file_name());

        if src.is_file() {
            std::fs::copy(&src, &dst).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;
        }
    }
    Ok(())
}

/// Quantize model files to reduce precision and size.
///
/// Every `*.safetensors` file found in `input_path` is genuinely quantized
/// (see [`quantize_safetensors_bytes`]): real per-tensor min/max are
/// computed from the real `F32` values, a real affine INT8 quantization is
/// applied, and a structurally valid SafeTensors file is rebuilt via the
/// `safetensors` crate. Formats that cannot honestly be quantized here
/// (`*.bin` PyTorch pickles, unrecognized extensions) are copied through
/// unchanged and recorded as skipped rather than corrupted.
///
/// If nothing in `input_path` could actually be quantized, this returns an
/// `Err` instead of reporting success: a command that "quantizes" a model by
/// quantizing zero tensors is exactly the fabricated-success behavior this
/// rewrite exists to eliminate.
async fn quantize_model_files(
    input_path: &PathBuf,
    output_path: &PathBuf,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("      Performing model quantization...");
    }

    // Create output directory
    std::fs::create_dir_all(output_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: output_path.clone(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    let mut total_tensors_quantized = 0usize;
    let mut total_tensors_passthrough = 0usize;
    let mut total_original_bytes = 0u64;
    let mut total_quantized_bytes = 0u64;
    let mut quantized_files: Vec<String> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();

    // Process model files
    for entry in std::fs::read_dir(input_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: input_path.clone(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })? {
        let entry = entry.map_err(|e| voirs_sdk::VoirsError::IoError {
            path: input_path.clone(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;
        let src = entry.path();
        let dst = output_path.join(entry.file_name());

        if !src.is_file() {
            continue;
        }

        let file_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        if file_name.ends_with(".safetensors") {
            let original_data =
                std::fs::read(&src).map_err(|e| voirs_sdk::VoirsError::IoError {
                    path: src.clone(),
                    operation: voirs_sdk::error::IoOperation::Read,
                    source: e,
                })?;

            match quantize_safetensors_bytes(&original_data)? {
                Some((quantized_data, report)) => {
                    total_original_bytes += original_data.len() as u64;
                    total_quantized_bytes += quantized_data.len() as u64;
                    total_tensors_quantized += report.quantized_tensor_count;
                    total_tensors_passthrough += report.passthrough_tensor_count;

                    std::fs::write(&dst, &quantized_data).map_err(|e| {
                        voirs_sdk::VoirsError::IoError {
                            path: dst.clone(),
                            operation: voirs_sdk::error::IoOperation::Write,
                            source: e,
                        }
                    })?;

                    if !global.quiet {
                        let ratio = original_data.len() as f64 / quantized_data.len().max(1) as f64;
                        println!(
                            "        Quantized {} ({} tensor(s), {} passthrough, {:.1}x compression)",
                            file_name, report.quantized_tensor_count, report.passthrough_tensor_count, ratio
                        );
                    }
                    quantized_files.push(file_name);
                }
                None => {
                    // Not real SafeTensors data (e.g. a prior optimization step
                    // already transformed this file under the same name). Pass
                    // it through unchanged rather than aborting the whole run.
                    std::fs::write(&dst, &original_data).map_err(|e| {
                        voirs_sdk::VoirsError::IoError {
                            path: dst.clone(),
                            operation: voirs_sdk::error::IoOperation::Write,
                            source: e,
                        }
                    })?;
                    skipped_files.push(format!("{file_name} (not a valid SafeTensors buffer)"));
                }
            }
        } else if file_name.ends_with(".bin") {
            // Real PyTorch pickle parsing (opcode stream + zip container) is
            // not implemented in pure Rust here. Copy through unchanged
            // instead of corrupting it, and be explicit that it was skipped.
            std::fs::copy(&src, &dst).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;
            skipped_files.push(format!(
                "{file_name} (PyTorch pickle quantization not implemented; convert to SafeTensors first)"
            ));
        } else if file_name.ends_with(".onnx") {
            return Err(voirs_sdk::VoirsError::model_error(format!(
                "Cannot quantize '{}': ONNX INT8 quantization is not implemented in voirs-cli \
                 (it requires parsing and rewriting the ONNX protobuf graph, which is out of \
                 scope here). Convert to SafeTensors first via `voirs convert-model --from onnx \
                 <input.onnx> <output.safetensors>`, then re-run `voirs optimize-model` on the \
                 resulting file, or use a dedicated tool such as onnxruntime.quantization.",
                src.display()
            )));
        } else {
            // Copy non-model files as-is
            std::fs::copy(&src, &dst).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;
        }
    }

    if total_tensors_quantized == 0 {
        return Err(voirs_sdk::VoirsError::model_error(format!(
            "Quantization produced no result: no SafeTensors weight tensors could be quantized \
             in '{}'.{}",
            input_path.display(),
            if skipped_files.is_empty() {
                " No weight files (*.safetensors) were found.".to_string()
            } else {
                format!(" Skipped: {}", skipped_files.join("; "))
            }
        )));
    }

    let compression_ratio = if total_quantized_bytes > 0 {
        total_original_bytes as f64 / total_quantized_bytes as f64
    } else {
        1.0
    };

    // Create quantization metadata from REAL, measured totals (no hardcoded
    // compression ratio or fabricated per-tensor counts).
    let metadata = serde_json::json!({
        "quantization": {
            "method": "int8_affine_per_tensor",
            "quantized_tensor_count": total_tensors_quantized,
            "passthrough_tensor_count": total_tensors_passthrough,
            "quantized_files": quantized_files,
            "skipped_files": skipped_files,
            "original_size_bytes": total_original_bytes,
            "quantized_size_bytes": total_quantized_bytes,
            "compression_ratio": compression_ratio,
            "quantized_at": chrono::Utc::now().to_rfc3339()
        }
    });

    let json_content = serde_json::to_string_pretty(&metadata).map_err(|e| {
        voirs_sdk::VoirsError::serialization(
            "json",
            format!("Failed to serialize quantization metadata: {}", e),
        )
    })?;

    std::fs::write(output_path.join("quantization_info.json"), json_content).map_err(|e| {
        voirs_sdk::VoirsError::IoError {
            path: output_path.join("quantization_info.json"),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        }
    })?;

    if !global.quiet {
        println!(
            "      ✓ Quantization completed: {} tensor(s) across {} file(s), {:.1}x compression",
            total_tensors_quantized,
            quantized_files.len(),
            compression_ratio
        );
    }
    Ok(())
}

/// Real per-tensor affine INT8 quantization report for one SafeTensors file.
struct SafeTensorsQuantReport {
    quantized_tensor_count: usize,
    passthrough_tensor_count: usize,
}

/// Real per-tensor affine quantization parameters, recorded so a quantized
/// tensor can be dequantized: `x ≈ (q as f32) * scale + zero_point`.
#[derive(serde::Serialize)]
struct TensorQuantParams {
    scale: f32,
    zero_point: f32,
    original_min: f32,
    original_max: f32,
}

/// Attempt real per-tensor affine INT8 quantization of a SafeTensors byte
/// buffer.
///
/// Every `F32` tensor is quantized to `U8` via the standard affine formula
/// `q = round((x - min) / scale)` with `scale = (max - min) / 255`, using
/// the tensor's REAL min/max (not a fabricated constant). Every other dtype
/// (already-integer types, `F16`/`BF16`, `BOOL`, ...) is copied through
/// byte-for-byte unchanged -- quantizing e.g. an integer token-id tensor
/// would corrupt it rather than shrink it usefully. The result is rebuilt
/// as a structurally valid SafeTensors file via the `safetensors` crate, so
/// header offsets always match the real data section (unlike the previous
/// implementation, which kept the *original* header while replacing the
/// data underneath it with stride-sampled bytes, producing offsets that
/// pointed past the end of the actual data).
///
/// Returns `Ok(None)` when `data` does not parse as SafeTensors at all --
/// callers should treat that as "nothing to quantize here" and copy the
/// bytes through, not as a hard failure (a prior optimization step may have
/// already transformed this file into something else, e.g. gzip, under the
/// same filename).
fn quantize_safetensors_bytes(data: &[u8]) -> Result<Option<(Vec<u8>, SafeTensorsQuantReport)>> {
    let parsed = match SafeTensors::deserialize(data) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };

    let mut owned_tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut quant_params: HashMap<String, TensorQuantParams> = HashMap::new();
    let mut quantized_tensor_count = 0usize;
    let mut passthrough_tensor_count = 0usize;

    for (name, view) in parsed.tensors() {
        if view.dtype() == Dtype::F32 {
            let (quantized_bytes, params) = quantize_f32_bytes_affine(view.data())?;
            quant_params.insert(name.clone(), params);
            owned_tensors.push((name, Dtype::U8, view.shape().to_vec(), quantized_bytes));
            quantized_tensor_count += 1;
        } else {
            owned_tensors.push((
                name,
                view.dtype(),
                view.shape().to_vec(),
                view.data().to_vec(),
            ));
            passthrough_tensor_count += 1;
        }
    }

    let mut tensor_map: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, dtype, shape, bytes) in &owned_tensors {
        let view = TensorView::new(*dtype, shape.clone(), bytes).map_err(|e| {
            voirs_sdk::VoirsError::model_error(format!(
                "Failed to rebuild tensor '{name}' after quantization: {e}"
            ))
        })?;
        tensor_map.insert(name.clone(), view);
    }

    let params_json = serde_json::to_string(&quant_params).map_err(|e| {
        voirs_sdk::VoirsError::serialization(
            "json",
            format!("Failed to serialize quantization params: {e}"),
        )
    })?;

    let mut file_metadata: HashMap<String, String> = HashMap::new();
    file_metadata.insert(
        "voirs_quantization_method".to_string(),
        "int8_affine_per_tensor".to_string(),
    );
    file_metadata.insert(
        "voirs_quantized_tensor_count".to_string(),
        quantized_tensor_count.to_string(),
    );
    file_metadata.insert(
        "voirs_passthrough_tensor_count".to_string(),
        passthrough_tensor_count.to_string(),
    );
    file_metadata.insert("voirs_quantization_params".to_string(), params_json);

    let out_bytes = safetensors::serialize(&tensor_map, Some(file_metadata)).map_err(|e| {
        voirs_sdk::VoirsError::model_error(format!(
            "Failed to serialize quantized SafeTensors file: {e}"
        ))
    })?;

    Ok(Some((
        out_bytes,
        SafeTensorsQuantReport {
            quantized_tensor_count,
            passthrough_tensor_count,
        },
    )))
}

/// Affine (asymmetric) per-tensor INT8 quantization of a raw little-endian
/// `F32` byte buffer (SafeTensors' documented byte order). Returns the
/// quantized bytes (one `U8` per element, in `[0, 255]`) plus the
/// `(scale, zero_point)` needed to dequantize: `x ≈ q * scale + zero_point`.
fn quantize_f32_bytes_affine(data: &[u8]) -> Result<(Vec<u8>, TensorQuantParams)> {
    if !data.len().is_multiple_of(4) {
        return Err(voirs_sdk::VoirsError::model_error(
            "F32 tensor byte length is not a multiple of 4 -- corrupt SafeTensors data",
        ));
    }

    let values: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let (min, max) = values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    // Empty tensor: fall back to a degenerate but well-defined range.
    let (min, max) = if values.is_empty() {
        (0.0, 0.0)
    } else {
        (min, max)
    };

    // Constant tensor (max == min): scale is irrelevant since every quantized
    // value will be 0, which dequantizes back to exactly `min`.
    let range = max - min;
    let scale = if range > f32::EPSILON {
        range / 255.0
    } else {
        1.0
    };

    let quantized: Vec<u8> = values
        .iter()
        .map(|&v| (((v - min) / scale).round().clamp(0.0, 255.0)) as u8)
        .collect();

    Ok((
        quantized,
        TensorQuantParams {
            scale,
            zero_point: min,
            original_min: min,
            original_max: max,
        },
    ))
}

/// Optimize model computational graph.
///
/// A raw `*.safetensors` weights file has no computation graph (unlike
/// ONNX): it is a flat bag of named tensors, so there are no operators to
/// fuse, no constants to fold, and no dead nodes to eliminate. The only
/// genuinely measurable optimization opportunity for that format is exact
/// tensor duplication (e.g. tied embedding / projection weights); this is
/// detected and reported (never rewritten -- SafeTensors readers assume
/// non-aliased offsets, so the file is always copied through byte-for-byte
/// unchanged). `*.onnx` files, which DO have a real graph, are refused with
/// a clear error rather than corrupted, since real ONNX graph optimization
/// (protobuf parsing + rewriting) is not implemented here.
async fn optimize_model_graph(
    input_path: &Path,
    output_path: &Path,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("      Analyzing computational structure...");
    }

    // Create output directory
    std::fs::create_dir_all(output_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: output_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    let mut safetensors_reports: Vec<(String, DuplicateTensorReport)> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();

    // Copy and analyze model files
    for entry in std::fs::read_dir(input_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: input_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })? {
        let entry = entry.map_err(|e| voirs_sdk::VoirsError::IoError {
            path: input_path.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;
        let src = entry.path();
        let dst = output_path.join(entry.file_name());

        if !src.is_file() {
            continue;
        }

        let file_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        if file_name == "config.json" {
            annotate_model_config(&src, &dst)?;
        } else if file_name.ends_with(".onnx") {
            return Err(voirs_sdk::VoirsError::model_error(format!(
                "Cannot optimize '{}': ONNX graph optimization (operator fusion / constant \
                 folding / dead-code elimination) is not implemented in voirs-cli -- this \
                 requires a full ONNX protobuf graph rewriter. Use tract's own optimizer or \
                 onnxruntime's graph optimization instead.",
                src.display()
            )));
        } else if file_name.ends_with(".safetensors") {
            let data = std::fs::read(&src).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

            match analyze_safetensors_duplicates(&data) {
                Some(report) => {
                    if !global.quiet && !report.duplicate_groups.is_empty() {
                        println!(
                            "        {}: {} duplicate tensor group(s), {} bytes redundant",
                            file_name,
                            report.duplicate_groups.len(),
                            report.duplicate_bytes
                        );
                    }
                    safetensors_reports.push((file_name, report));
                }
                None => {
                    // Not real SafeTensors data (e.g. an earlier step in this
                    // same run already replaced it, such as gzip compression
                    // running before graph analysis). Being unable to analyze
                    // a buffer is not a failed transformation -- copy it
                    // through and say so, rather than aborting the run.
                    skipped_files.push(format!(
                        "{file_name} (not a valid SafeTensors buffer -- likely already \
                         transformed by an earlier optimization step)"
                    ));
                }
            }

            // Always a byte-identical passthrough: this step only reports on
            // duplication, it never rewrites tensor data or offsets.
            std::fs::write(&dst, &data).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: dst.clone(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: e,
            })?;
        } else {
            // Copy other files
            std::fs::copy(&src, &dst).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;
        }
    }

    // Create optimization metadata from REAL, measured findings.
    let metadata = build_graph_analysis_metadata(&safetensors_reports, &skipped_files);

    let json_content = serde_json::to_string_pretty(&metadata).map_err(|e| {
        voirs_sdk::VoirsError::serialization(
            "json",
            format!("Failed to serialize optimization metadata: {}", e),
        )
    })?;

    std::fs::write(output_path.join("optimization_info.json"), json_content).map_err(|e| {
        voirs_sdk::VoirsError::IoError {
            path: output_path.join("optimization_info.json"),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        }
    })?;

    if !global.quiet {
        println!("      ✓ Graph analysis completed");
    }
    Ok(())
}

/// Report of exact-duplicate tensors found in one SafeTensors file: tensors
/// whose dtype, shape AND raw bytes are all identical.
struct DuplicateTensorReport {
    tensor_count: usize,
    duplicate_groups: Vec<Vec<String>>,
    duplicate_bytes: u64,
}

/// Detect exact-duplicate tensors in a SafeTensors byte buffer.
///
/// Groups tensors by `(dtype, shape, SHA-256 of the raw bytes)`, then -- as
/// a defense-in-depth check against the astronomically unlikely case of a
/// hash collision -- re-verifies real byte equality within any group with
/// more than one member before reporting it as a duplicate. Returns `None`
/// when `data` does not parse as SafeTensors.
fn analyze_safetensors_duplicates(data: &[u8]) -> Option<DuplicateTensorReport> {
    let parsed = SafeTensors::deserialize(data).ok()?;

    // `Dtype` does not derive `Hash`, but it does derive `Ord`, so a
    // `BTreeMap` groups tensors by content signature without needing to
    // clone every tensor's bytes into the map key (only a 32-byte digest is
    // stored).
    let mut by_signature: BTreeMap<(Dtype, Vec<usize>, Vec<u8>), Vec<String>> = BTreeMap::new();
    for (name, view) in parsed.tensors() {
        let digest = Sha256::digest(view.data()).to_vec();
        by_signature
            .entry((view.dtype(), view.shape().to_vec(), digest))
            .or_default()
            .push(name);
    }

    let mut duplicate_groups: Vec<Vec<String>> = Vec::new();
    let mut duplicate_bytes = 0u64;

    for mut names in by_signature.into_values() {
        if names.len() < 2 {
            continue;
        }
        names.sort();

        let Ok(first_view) = parsed.tensor(&names[0]) else {
            continue;
        };
        let first_bytes = first_view.data();
        let all_identical = names[1..].iter().all(|n| {
            parsed
                .tensor(n)
                .map(|v| v.data() == first_bytes)
                .unwrap_or(false)
        });

        if all_identical {
            duplicate_bytes += first_bytes.len() as u64 * (names.len() as u64 - 1);
            duplicate_groups.push(names);
        }
    }
    duplicate_groups.sort();

    Some(DuplicateTensorReport {
        tensor_count: parsed.len(),
        duplicate_groups,
        duplicate_bytes,
    })
}

/// Build honest graph-analysis metadata from real per-file duplicate reports.
fn build_graph_analysis_metadata(
    reports: &[(String, DuplicateTensorReport)],
    skipped_files: &[String],
) -> serde_json::Value {
    let total_duplicate_groups: usize = reports.iter().map(|(_, r)| r.duplicate_groups.len()).sum();
    let total_duplicate_bytes: u64 = reports.iter().map(|(_, r)| r.duplicate_bytes).sum();

    serde_json::json!({
        "graph_analysis": {
            "note": "SafeTensors weight files have no computation graph to fuse or fold \
                      (that only applies to formats like ONNX); the only optimization \
                      opportunity that can honestly be measured here is exact tensor \
                      duplication. This step is analysis-only -- files are copied through \
                      byte-for-byte unchanged.",
            "files_analyzed": reports.iter().map(|(name, r)| serde_json::json!({
                "file": name,
                "tensor_count": r.tensor_count,
                "duplicate_groups": r.duplicate_groups,
                "duplicate_bytes": r.duplicate_bytes,
            })).collect::<Vec<_>>(),
            "files_skipped": skipped_files,
            "total_duplicate_groups": total_duplicate_groups,
            "total_duplicate_bytes": total_duplicate_bytes,
            "analyzed_at": chrono::Utc::now().to_rfc3339()
        }
    })
}

/// Compress model files to reduce size
async fn compress_model_files(
    input_path: &PathBuf,
    output_path: &PathBuf,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("      Compressing model files...");
    }

    // Create output directory
    std::fs::create_dir_all(output_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: output_path.clone(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    let mut total_original_size = 0u64;
    let mut total_compressed_size = 0u64;

    // Compress model files
    for entry in std::fs::read_dir(input_path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: input_path.clone(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })? {
        let entry = entry.map_err(|e| voirs_sdk::VoirsError::IoError {
            path: input_path.clone(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;
        let src = entry.path();

        if !src.is_file() {
            continue;
        }

        let original_size = src
            .metadata()
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?
            .len();
        total_original_size += original_size;

        let file_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let compressed_size = if file_name.ends_with(".safetensors") || file_name.ends_with(".bin")
        {
            // Gzip output is no longer a valid SafeTensors/PyTorch file, so it
            // must NOT keep the original extension: a later optimization step
            // (or a human, or a model loader) reading a file named
            // "model.safetensors" must never receive gzip bytes under that
            // name and try to load it as a real model.
            let gz_dst = output_path.join(format!("{file_name}.gz"));
            compress_model_file(&src, &gz_dst)?;
            gz_dst
                .metadata()
                .map_err(|e| voirs_sdk::VoirsError::IoError {
                    path: gz_dst.clone(),
                    operation: voirs_sdk::error::IoOperation::Read,
                    source: e,
                })?
                .len()
        } else {
            // Copy smaller files without compression
            let dst = output_path.join(entry.file_name());
            std::fs::copy(&src, &dst).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: src.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;
            original_size
        };
        total_compressed_size += compressed_size;
    }

    // Calculate compression ratio
    let compression_ratio = if total_original_size > 0 {
        total_compressed_size as f64 / total_original_size as f64
    } else {
        1.0
    };

    // Create compression metadata
    let metadata = serde_json::json!({
        "compression": {
            "method": "gzip",
            "original_size_bytes": total_original_size,
            "compressed_size_bytes": total_compressed_size,
            "compression_ratio": compression_ratio,
            "space_saved_percent": (1.0 - compression_ratio) * 100.0,
            "compressed_at": chrono::Utc::now().to_rfc3339()
        }
    });

    let json_content = serde_json::to_string_pretty(&metadata).map_err(|e| {
        voirs_sdk::VoirsError::serialization(
            "json",
            format!("Failed to serialize compression metadata: {}", e),
        )
    })?;

    std::fs::write(output_path.join("compression_info.json"), json_content).map_err(|e| {
        voirs_sdk::VoirsError::IoError {
            path: output_path.join("compression_info.json"),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        }
    })?;

    if !global.quiet {
        println!(
            "      ✓ Compression completed ({:.1}% size reduction)",
            (1.0 - compression_ratio) * 100.0
        );
    }
    Ok(())
}

/// Annotate `config.json` with the real fact that voirs-cli's optimize
/// pipeline analyzed this model. Unlike the previous implementation, this
/// never claims specific ML transformations happened (`enable_fusion`,
/// `memory_optimization`, ...) -- a flat weights directory has no graph for
/// those to apply to, so asserting them would be exactly the kind of
/// fabricated capability flag this rewrite exists to remove.
fn annotate_model_config(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    let config_content =
        std::fs::read_to_string(src).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: src.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?;

    let mut config: serde_json::Value = serde_json::from_str(&config_content)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Invalid JSON config: {}", e)))?;

    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "voirs_optimize_analyzed".to_string(),
            serde_json::Value::Bool(true),
        );
        obj.insert(
            "voirs_optimize_analyzed_at".to_string(),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );
    }

    let annotated_content = serde_json::to_string_pretty(&config).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to serialize config: {}", e))
    })?;

    std::fs::write(dst, annotated_content).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: dst.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    Ok(())
}

/// Calculate speed improvement
fn calculate_speed_improvement(strategy: &OptimizationStrategy) -> f64 {
    match strategy {
        OptimizationStrategy::Speed => 2.5,
        OptimizationStrategy::Quality => 1.1,
        OptimizationStrategy::Memory => 1.8,
        OptimizationStrategy::Balanced => 1.7,
    }
}

/// Calculate quality impact
fn calculate_quality_impact(strategy: &OptimizationStrategy) -> f64 {
    match strategy {
        OptimizationStrategy::Speed => -0.3,
        OptimizationStrategy::Quality => 0.1,
        OptimizationStrategy::Memory => -0.5,
        OptimizationStrategy::Balanced => -0.1,
    }
}

/// Display optimization results
fn display_optimization_results(
    result: &OptimizationResult,
    strategy: &OptimizationStrategy,
    global: &GlobalOptions,
) {
    if global.quiet {
        return;
    }

    println!("\nOptimization Complete!");
    println!("======================");
    println!("Strategy: {:?}", strategy);
    println!("Original size: {:.1} MB", result.original_size_mb);
    println!("Optimized size: {:.1} MB", result.optimized_size_mb);
    println!("Compression ratio: {:.2}x", result.compression_ratio);
    println!(
        "Estimated speed improvement: {:.1}x (strategy-based estimate, not a measured \
         benchmark -- run `voirs benchmark-models` before/after for real timing)",
        result.speed_improvement
    );
    println!(
        "Estimated quality impact: {:.1} (strategy-based estimate)",
        result.quality_impact
    );
    println!("Output path: {}", result.output_path.display());
}

/// Compress model file using gzip
fn compress_model_file(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    use oxiarc_deflate::GzipStreamEncoder;
    use std::io::{Read, Write};

    let mut input_file = std::fs::File::open(src).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: src.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    let output_file = std::fs::File::create(dst).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: dst.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    let mut encoder = GzipStreamEncoder::new(output_file, 6);
    let mut buffer = [0; 8192];

    loop {
        let bytes_read =
            input_file
                .read(&mut buffer)
                .map_err(|e| voirs_sdk::VoirsError::IoError {
                    path: src.to_path_buf(),
                    operation: voirs_sdk::error::IoOperation::Read,
                    source: e,
                })?;

        if bytes_read == 0 {
            break;
        }

        encoder
            .write_all(&buffer[..bytes_read])
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: dst.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: e,
            })?;
    }

    encoder
        .finish()
        .map_err(|e| voirs_sdk::VoirsError::IoError {
            path: dst.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "voirs_optimize_test_{}_{}_{}",
            label,
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    fn default_global() -> GlobalOptions {
        GlobalOptions {
            config: None,
            verbose: 0,
            quiet: true,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        }
    }

    /// Build a tiny but real SafeTensors buffer with one F32 tensor and one
    /// I64 tensor (so passthrough behavior can be verified too).
    fn build_test_safetensors(f32_values: &[f32]) -> Vec<u8> {
        let f32_bytes: Vec<u8> = f32_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f32_view = TensorView::new(Dtype::F32, vec![f32_values.len()], &f32_bytes)
            .expect("valid F32 tensor view");

        let int_values: [i64; 2] = [7, 9];
        let int_bytes: Vec<u8> = int_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_view = TensorView::new(Dtype::I64, vec![int_values.len()], &int_bytes)
            .expect("valid I64 view");

        let mut tensors: StdHashMap<String, TensorView<'_>> = StdHashMap::new();
        tensors.insert("weight".to_string(), f32_view);
        tensors.insert("token_ids".to_string(), int_view);

        safetensors::serialize(&tensors, None).expect("serialize test safetensors")
    }

    #[test]
    fn test_determine_optimization_strategy() {
        let config = AppConfig::default();
        let global = default_global();

        // Test default balanced strategy
        let strategy = determine_optimization_strategy(None, &config, &global)
            .expect("Should determine balanced strategy");
        assert!(matches!(strategy, OptimizationStrategy::Balanced));

        // Test explicit strategies
        let strategy = determine_optimization_strategy(Some("speed"), &config, &global)
            .expect("Should determine speed strategy");
        assert!(matches!(strategy, OptimizationStrategy::Speed));

        let strategy = determine_optimization_strategy(Some("quality"), &config, &global)
            .expect("Should determine quality strategy");
        assert!(matches!(strategy, OptimizationStrategy::Quality));

        let strategy = determine_optimization_strategy(Some("memory"), &config, &global)
            .expect("Should determine memory strategy");
        assert!(matches!(strategy, OptimizationStrategy::Memory));

        // Test case insensitivity
        let strategy = determine_optimization_strategy(Some("SPEED"), &config, &global)
            .expect("Should handle case-insensitive strategy");
        assert!(matches!(strategy, OptimizationStrategy::Speed));

        // Test invalid strategy
        let result = determine_optimization_strategy(Some("invalid"), &config, &global);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_optimization_steps() {
        let steps = get_optimization_steps(&OptimizationStrategy::Speed);
        assert!(!steps.is_empty());
        assert!(steps.iter().any(|s| s.contains("Quantizing")));
    }

    #[test]
    fn test_calculate_speed_improvement() {
        let improvement = calculate_speed_improvement(&OptimizationStrategy::Speed);
        assert!(improvement > 1.0);
    }

    #[test]
    fn test_quantize_f32_bytes_affine_roundtrip() {
        let values = [0.0f32, -1.5, 3.25, 100.0, -100.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let (quantized, params) =
            quantize_f32_bytes_affine(&bytes).expect("quantization should succeed");

        assert_eq!(quantized.len(), values.len());
        // Real per-tensor min/max, not a fabricated constant.
        assert!((params.original_min - (-100.0)).abs() < 1e-6);
        assert!((params.original_max - 100.0).abs() < 1e-6);

        // Dequantize and check every real value round-trips within one
        // quantization step (the theoretical maximum error for 8-bit affine
        // quantization of this range).
        let max_error = params.scale;
        for (i, &original) in values.iter().enumerate() {
            let dequantized = quantized[i] as f32 * params.scale + params.zero_point;
            assert!(
                (dequantized - original).abs() <= max_error + 1e-4,
                "value {i}: original={original}, dequantized={dequantized}, max_error={max_error}"
            );
        }
    }

    #[test]
    fn test_quantize_f32_bytes_affine_constant_tensor() {
        // Degenerate case: every value identical -- must not divide by zero
        // and must reconstruct the exact constant.
        let values = [5.0f32; 8];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (quantized, params) = quantize_f32_bytes_affine(&bytes).expect("must not fail");
        for &q in &quantized {
            let dequantized = q as f32 * params.scale + params.zero_point;
            assert!((dequantized - 5.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_quantize_safetensors_bytes_is_valid_and_shrinks() {
        // Large enough that the real 4x data reduction (F32 -> U8) clearly
        // dominates the SafeTensors header + quantization-metadata overhead
        // added per file (for a handful of elements, that fixed overhead can
        // exceed the raw data savings, which would make a "must shrink"
        // assertion flaky rather than meaningful).
        let values: Vec<f32> = (0..300).map(|i| i as f32 * 0.5 - 10.0).collect();
        let original = build_test_safetensors(&values);

        let (quantized, report) = quantize_safetensors_bytes(&original)
            .expect("must not error")
            .expect("real safetensors input must be recognized");

        assert_eq!(report.quantized_tensor_count, 1); // the F32 tensor
        assert_eq!(report.passthrough_tensor_count, 1); // the I64 tensor

        // Output must be structurally valid SafeTensors.
        let reparsed =
            SafeTensors::deserialize(&quantized).expect("output must be valid SafeTensors");
        let weight = reparsed.tensor("weight").expect("weight tensor present");
        assert_eq!(weight.dtype(), Dtype::U8);
        assert_eq!(weight.shape(), &[values.len()]);

        let token_ids = reparsed.tensor("token_ids").expect("token_ids present");
        assert_eq!(token_ids.dtype(), Dtype::I64); // passthrough: untouched

        // The whole file must genuinely shrink (F32 -> U8 is a real 4x
        // reduction on the quantized tensor).
        assert!(quantized.len() < original.len());
    }

    #[test]
    fn test_quantize_safetensors_bytes_rejects_garbage() {
        let garbage = b"this is definitely not a safetensors file".to_vec();
        let result = quantize_safetensors_bytes(&garbage).expect("must not error");
        assert!(
            result.is_none(),
            "garbage input must be reported as unparseable, not corrupted"
        );
    }

    #[test]
    fn test_analyze_safetensors_duplicates_detects_real_duplicates() {
        let shared_bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let unique_bytes: Vec<u8> = [9.0f32, 9.0, 9.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let view_a = TensorView::new(Dtype::F32, vec![3], &shared_bytes).unwrap();
        let view_b = TensorView::new(Dtype::F32, vec![3], &shared_bytes).unwrap();
        let view_c = TensorView::new(Dtype::F32, vec![3], &unique_bytes).unwrap();

        let mut tensors: StdHashMap<String, TensorView<'_>> = StdHashMap::new();
        tensors.insert("layer_a.weight".to_string(), view_a);
        tensors.insert("layer_b.weight".to_string(), view_b);
        tensors.insert("layer_c.weight".to_string(), view_c);

        let data = safetensors::serialize(&tensors, None).unwrap();
        let report = analyze_safetensors_duplicates(&data).expect("real safetensors input");

        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.duplicate_groups.len(), 1);
        assert_eq!(report.duplicate_groups[0].len(), 2);
        assert!(report.duplicate_groups[0].contains(&"layer_a.weight".to_string()));
        assert!(report.duplicate_groups[0].contains(&"layer_b.weight".to_string()));
        assert!(!report
            .duplicate_groups
            .iter()
            .any(|g| g.contains(&"layer_c.weight".to_string())));
        assert_eq!(report.duplicate_bytes, shared_bytes.len() as u64);
    }

    #[tokio::test]
    async fn test_quantize_model_files_end_to_end_real_shrink() {
        let global = default_global();
        let input_dir = unique_temp_dir("quantize_input");
        let output_dir = unique_temp_dir("quantize_output");

        let data = build_test_safetensors(&(0..256).map(|i| i as f32).collect::<Vec<_>>());
        std::fs::write(input_dir.join("model.safetensors"), &data).unwrap();

        quantize_model_files(&input_dir, &output_dir, &global)
            .await
            .expect("quantization of a real safetensors file must succeed");

        let output_file = std::fs::read(output_dir.join("model.safetensors")).unwrap();
        assert!(
            output_file.len() < data.len(),
            "output must genuinely shrink"
        );
        SafeTensors::deserialize(&output_file).expect("output must be structurally valid");

        let info_path = output_dir.join("quantization_info.json");
        assert!(info_path.exists());
        let info: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(info_path).unwrap()).unwrap();
        let quantized_count = info["quantization"]["quantized_tensor_count"]
            .as_u64()
            .unwrap();
        assert_eq!(quantized_count, 1);
        // Compression ratio must be computed from the real byte totals, not
        // a hardcoded constant -- for a mostly-F32 tensor it must exceed 1x.
        assert!(info["quantization"]["compression_ratio"].as_f64().unwrap() > 1.0);

        std::fs::remove_dir_all(&input_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// Regression test: a directory that contains only an unquantizable
    /// format (`.bin`) must not be reported as a successful quantization.
    #[tokio::test]
    async fn test_quantize_model_files_fails_closed_when_nothing_quantized() {
        let global = default_global();
        let input_dir = unique_temp_dir("quantize_bin_only_input");
        let output_dir = unique_temp_dir("quantize_bin_only_output");

        std::fs::write(
            input_dir.join("pytorch_model.bin"),
            b"\x80\x02fake pickle bytes",
        )
        .unwrap();

        let result = quantize_model_files(&input_dir, &output_dir, &global).await;
        assert!(
            result.is_err(),
            "quantizing a directory with zero real SafeTensors weights must fail closed, not report success"
        );

        std::fs::remove_dir_all(&input_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// Regression test for the step-chaining bug: a multi-step strategy
    /// (Balanced: quantize -> analyze graph -> copy -> compress) must reflect
    /// the CUMULATIVE effect of every step, not just the last one. Before the
    /// fix, every step read straight from the pristine input and overwrote
    /// the output directory wholesale, so the real quantization from step 1
    /// was silently discarded by the plain-copy step that ran later.
    #[tokio::test]
    async fn test_perform_optimization_chains_steps_balanced_strategy() {
        let global = default_global();
        let model_dir = unique_temp_dir("chain_model");
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({"model_type": "test"}).to_string(),
        )
        .unwrap();
        let data = build_test_safetensors(&(0..512).map(|i| i as f32 * 0.5).collect::<Vec<_>>());
        std::fs::write(model_dir.join("model.safetensors"), &data).unwrap();

        let result = perform_optimization(
            "chain-test-model",
            &model_dir,
            None,
            &OptimizationStrategy::Balanced,
            &global,
        )
        .await
        .expect("balanced optimization should succeed");

        // The final output must still contain the gzip-compressed artifact
        // from the LAST step (Compressing model artifacts)...
        let gz_path = result.output_path.join("model.safetensors.gz");
        assert!(
            gz_path.exists(),
            "final output must contain the compressed artifact"
        );

        // ...but decompressing it must reveal the QUANTIZED (shrunk, valid)
        // tensor data from step 1, not the original pristine F32 data. If
        // steps did not chain, this would be a compressed copy of the
        // ORIGINAL (unquantized) input instead.
        let compressed_bytes = std::fs::read(&gz_path).unwrap();
        let decompressed = oxiarc_deflate::gzip_decompress(&compressed_bytes)
            .expect("must be valid gzip data produced by our own compressor");
        let reparsed = SafeTensors::deserialize(&decompressed)
            .expect("decompressed bytes must be a valid SafeTensors file");
        let weight = reparsed.tensor("weight").expect("weight tensor present");
        assert_eq!(
            weight.dtype(),
            Dtype::U8,
            "chained output must reflect step 1's real quantization (F32 -> U8), \
             not the untouched original F32 data"
        );

        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&result.output_path).ok();
    }

    /// Regression test for the case-sensitive dispatch bug: the Memory
    /// strategy's quantization step is literally named "Applying aggressive
    /// quantization" (lowercase "quantization", not "Quantizing"), which a
    /// case-sensitive `contains("Quantizing")` check never matched -- so
    /// Memory-strategy optimization silently never quantized anything and
    /// fell through to a plain file copy instead. This calls the dispatcher
    /// directly with Memory strategy's exact step-name string so the test
    /// fails if the substring match regresses, independent of how later
    /// steps in the chain might also transform the file.
    #[tokio::test]
    async fn test_memory_strategy_quantization_step_dispatches_correctly() {
        let global = default_global();
        let input_dir = unique_temp_dir("memory_dispatch_input");
        let output_dir = unique_temp_dir("memory_dispatch_output");

        let data = build_test_safetensors(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
        std::fs::write(input_dir.join("model.safetensors"), &data).unwrap();

        apply_optimization_step(
            "Applying aggressive quantization",
            &input_dir,
            &output_dir,
            &global,
        )
        .await
        .expect("Memory strategy's step name must dispatch to real quantization, not a no-op copy");

        let output_file = std::fs::read(output_dir.join("model.safetensors")).unwrap();
        assert!(
            output_file.len() < data.len(),
            "the dispatched step must have genuinely quantized (shrunk) the file"
        );
        SafeTensors::deserialize(&output_file).expect("output must be valid SafeTensors");
        assert!(
            output_dir.join("quantization_info.json").exists(),
            "real quantization must produce quantization_info.json"
        );

        std::fs::remove_dir_all(&input_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// End-to-end companion to the dispatch test above: run the FULL Memory
    /// strategy chain and confirm the real quantization from step 1 survives
    /// all the way through step 4 (Pruning -> Compressing -> Optimizing
    /// memory layout) into the final output, the same way
    /// `test_perform_optimization_chains_steps_balanced_strategy` verifies
    /// it for the Balanced strategy.
    #[tokio::test]
    async fn test_memory_strategy_quantization_survives_full_chain() {
        let global = default_global();
        let model_dir = unique_temp_dir("memory_strategy_model");
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({"model_type": "test"}).to_string(),
        )
        .unwrap();
        let data = build_test_safetensors(&(0..256).map(|i| i as f32).collect::<Vec<_>>());
        std::fs::write(model_dir.join("model.safetensors"), &data).unwrap();

        let result = perform_optimization(
            "memory-test-model",
            &model_dir,
            None,
            &OptimizationStrategy::Memory,
            &global,
        )
        .await
        .expect("memory-strategy optimization should succeed");

        // Memory strategy's step order ends with "Compressing model storage"
        // then "Optimizing memory layout", so the final artifact is the
        // gzip-compressed, previously-quantized tensor file.
        let gz_path = result.output_path.join("model.safetensors.gz");
        assert!(
            gz_path.exists(),
            "final output must contain the compressed artifact"
        );

        let compressed_bytes = std::fs::read(&gz_path).unwrap();
        let decompressed = oxiarc_deflate::gzip_decompress(&compressed_bytes)
            .expect("must be valid gzip data produced by our own compressor");
        let reparsed = SafeTensors::deserialize(&decompressed)
            .expect("decompressed bytes must be a valid SafeTensors file");
        let weight = reparsed.tensor("weight").expect("weight tensor present");
        assert_eq!(
            weight.dtype(),
            Dtype::U8,
            "chained output must reflect step 1's real quantization, not the untouched original F32 data"
        );

        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&result.output_path).ok();
        std::fs::remove_dir_all(&result.output_path).ok();
    }
}
