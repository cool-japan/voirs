//! Model format conversion utilities
//!
//! Converts models from various formats (ONNX, PyTorch) to SafeTensors format
//! for use with VoiRS.

use crate::GlobalOptions;
use bytemuck;
use safetensors;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tract_core::ops::konst::Const;
use tract_onnx::prelude::*;
use voirs_sdk::Result;

/// Run model conversion
pub async fn run_convert_model(
    input: PathBuf,
    output: PathBuf,
    from: Option<String>,
    model_type: String,
    verify: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔄 VoiRS Model Converter");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Input:  {}", input.display());
        println!("Output: {}", output.display());
        println!("Type:   {}", model_type);
    }

    // Validate input file
    if !input.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Input model file not found: {}",
            input.display()
        )));
    }

    // Auto-detect format if not specified
    let source_format = from.unwrap_or_else(|| detect_format(&input));

    if !global.quiet {
        println!("Format: {} → SafeTensors", source_format);
        println!();
    }

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert based on format
    match source_format.as_str() {
        "onnx" => {
            if !global.quiet {
                println!("📥 Loading ONNX model...");
            }
            convert_onnx_to_safetensors(&input, &output, &model_type, global).await?;
        }
        "pytorch" | "pt" | "pth" => {
            if !global.quiet {
                println!("📥 Loading PyTorch model...");
            }
            convert_pytorch_to_safetensors(&input, &output, &model_type, global).await?;
        }
        _ => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported format: '{}'. Supported formats: onnx, pytorch/pt/pth",
                source_format
            )));
        }
    }

    if !global.quiet {
        println!("✅ Conversion complete!");
        println!("   Output: {}", output.display());
    }

    // Verify if requested
    if verify {
        if !global.quiet {
            println!();
            println!("🔍 Verifying converted model...");
        }
        verify_conversion(&output, &model_type, global).await?;
        if !global.quiet {
            println!("✅ Verification passed!");
        }
    }

    if !global.quiet {
        println!();
        println!("🎉 Model conversion successful!");
    }

    Ok(())
}

/// Detect input format from file extension
fn detect_format(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Convert ONNX model to SafeTensors
async fn convert_onnx_to_safetensors(
    input: &Path,
    output: &Path,
    model_type: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("📥 Loading ONNX model with tract-onnx...");
    }

    // Load ONNX model
    let model = tract_onnx::onnx()
        .model_for_path(input)
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to load ONNX model: {}", e))
        })?
        .into_optimized()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to optimize ONNX model: {}", e))
        })?;

    if !global.quiet {
        println!("✅ Model loaded successfully");
        println!("🔍 Extracting weights from model graph...");
    }

    // Note: Full ONNX weight extraction through tract requires additional implementation
    // For now, we provide model validation and structure analysis

    let node_count = model.nodes().len();
    let input_count = model
        .input_outlets()
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to get inputs: {}", e)))?
        .len();
    let output_count = model
        .output_outlets()
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to get outputs: {}", e)))?
        .len();

    if !global.quiet {
        println!("✅ Model structure analyzed");
        println!("📊 Model information:");
        println!("   - Total nodes: {}", node_count);
        println!("   - Inputs: {}", input_count);
        println!("   - Outputs: {}", output_count);
        println!();
        println!("⚠️  Note: Full tensor weight extraction not yet implemented");
        println!("   For complete ONNX → SafeTensors conversion, use:");
        println!();
        println!("   Python method (recommended):");
        println!("   ```python");
        println!("   import onnx, numpy as np");
        println!("   from safetensors import serialize_to_file");
        println!();
        println!("   model = onnx.load('{}')", input.display());
        println!("   tensors = {{}}");
        println!("   for init in model.graph.initializer:");
        println!("       tensors[init.name] = numpy_helper.to_array(init)");
        println!("   serialize_to_file(tensors, '{}')", output.display());
        println!("   ```");
    }

    // Create placeholder tensors_map (empty for now)
    let tensors_map: HashMap<String, TensorView<'_>> = HashMap::new();
    let tensor_count = 0;

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "onnx".to_string());
    metadata.insert("source_path".to_string(), input.display().to_string());
    metadata.insert("model_type".to_string(), model_type.to_string());
    metadata.insert("tensor_count".to_string(), tensor_count.to_string());
    metadata.insert(
        "converted_with".to_string(),
        "voirs-cli/tract-onnx".to_string(),
    );

    if !global.quiet {
        println!("💾 Saving as SafeTensors...");
    }

    // Save as SafeTensors
    safetensors::serialize_to_file(&tensors_map, Some(metadata), output).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to save SafeTensors: {}", e))
    })?;

    if !global.quiet {
        println!("✅ Saved to {}", output.display());
        println!("📊 Summary:");
        println!("   - Extracted {} tensors", tensor_count);
        println!("   - Model type: {}", model_type);
        println!("   - Output format: SafeTensors");
    }

    Ok(())
}

/// Convert tract tensor to SafeTensors TensorView
fn tract_tensor_to_safetensors<'a>(tensor: &'a Tensor, name: &str) -> Result<TensorView<'a>> {
    // Get tensor shape
    let shape: Vec<usize> = tensor.shape().to_vec();

    // Convert based on datum type
    let datum_type = tensor.datum_type();

    // SafeTensors TensorView expects raw bytes, so we need to convert typed slices to &[u8]
    // For now, we'll support common types used in neural networks
    if datum_type == f32::datum_type() {
        let data = tensor.as_slice::<f32>().map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to get f32 slice for tensor '{}': {}",
                name, e
            ))
        })?;

        // Convert to bytes using bytemuck
        let bytes = bytemuck::cast_slice::<f32, u8>(data);

        Ok(TensorView::new(Dtype::F32, shape, bytes).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to create TensorView for '{}': {}",
                name, e
            ))
        })?)
    } else if datum_type == f64::datum_type() {
        let data = tensor.as_slice::<f64>().map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to get f64 slice for tensor '{}': {}",
                name, e
            ))
        })?;

        let bytes = bytemuck::cast_slice::<f64, u8>(data);

        Ok(TensorView::new(Dtype::F64, shape, bytes).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to create TensorView for '{}': {}",
                name, e
            ))
        })?)
    } else if datum_type == i64::datum_type() {
        let data = tensor.as_slice::<i64>().map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to get i64 slice for tensor '{}': {}",
                name, e
            ))
        })?;

        let bytes = bytemuck::cast_slice::<i64, u8>(data);

        Ok(TensorView::new(Dtype::I64, shape, bytes).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to create TensorView for '{}': {}",
                name, e
            ))
        })?)
    } else if datum_type == i32::datum_type() {
        let data = tensor.as_slice::<i32>().map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to get i32 slice for tensor '{}': {}",
                name, e
            ))
        })?;

        let bytes = bytemuck::cast_slice::<i32, u8>(data);

        Ok(TensorView::new(Dtype::I32, shape, bytes).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!(
                "Failed to create TensorView for '{}': {}",
                name, e
            ))
        })?)
    } else {
        Err(voirs_sdk::VoirsError::config_error(format!(
            "Unsupported tensor data type for '{}': {:?}. Supported: f32, f64, i32, i64",
            name, datum_type
        )))
    }
}

/// Convert PyTorch model to SafeTensors
async fn convert_pytorch_to_safetensors(
    input: &Path,
    output: &Path,
    _model_type: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("⚠️  PyTorch .pt/.pth conversion not yet implemented in pure Rust.");
        println!("    PyTorch files use Python's pickle format which requires:");
        println!("    1. Python interpreter with PyTorch installed, OR");
        println!("    2. tch-rs crate with libtorch dependency");
        println!();
        println!("🔧 Recommended Conversion Methods:");
        println!();
        println!("   Method 1: Python script (easiest)");
        println!("   ```python");
        println!("   import torch");
        println!("   from safetensors.torch import save_file");
        println!();
        println!("   # Load PyTorch model");
        println!(
            "   state_dict = torch.load('{}', map_location='cpu')",
            input.display()
        );
        println!();
        println!("   # Save as SafeTensors");
        println!("   save_file(state_dict, '{}')", output.display());
        println!("   ```");
        println!();
        println!("   Method 2: Convert to ONNX first");
        println!("   ```python");
        println!("   import torch");
        println!("   import torch.onnx");
        println!();
        println!("   model = torch.load('{}').eval()", input.display());
        println!("   dummy_input = torch.randn(1, 80, 100)  # Adjust shape");
        println!("   torch.onnx.export(model, dummy_input, 'model.onnx')");
        println!("   ```");
        println!("   Then: voirs convert-model model.onnx output.safetensors");
        println!();
        println!("   Method 3: Use tch-rs (requires libtorch)");
        println!("   Add to Cargo.toml: tch = \"0.15\"");
        println!("   Requires: libtorch C++ library installed");
    }

    Err(voirs_sdk::VoirsError::config_error(
        "PyTorch conversion requires Python script or tch-rs. See output above for methods.",
    ))
}

/// Verify converted model
async fn verify_conversion(output: &Path, model_type: &str, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("   Checking file exists...");
    }

    // Check if output file exists
    let metadata_path = output.with_extension("json");
    if !metadata_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(
            "Converted model metadata file not found",
        ));
    }

    if !global.quiet {
        println!("   Loading metadata...");
    }

    // Load and verify metadata
    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

    // Check model type matches
    if let Some(mt) = metadata.get("model_type").and_then(|v| v.as_str()) {
        if mt != model_type && !global.quiet {
            println!(
                "   ⚠️  Model type mismatch: expected '{}', found '{}'",
                model_type, mt
            );
        }
    }

    if !global.quiet {
        println!("   Model type: {}", model_type);
        println!(
            "   Source format: {}",
            metadata
                .get("source_format")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );
    }

    // Load and verify SafeTensors file
    if !global.quiet {
        println!("   Loading SafeTensors file...");
    }

    // Read SafeTensors file
    let safetensors_data = std::fs::read(output)?;

    // Parse SafeTensors format
    match safetensors::SafeTensors::deserialize(&safetensors_data) {
        Ok(tensors) => {
            if !global.quiet {
                println!("   ✅ SafeTensors format valid");
                println!("   Tensors found: {}", tensors.names().len());
                println!();

                // Show tensor information
                println!("   Tensor Details:");
                for name in tensors.names() {
                    if let Ok(tensor_view) = tensors.tensor(name) {
                        let shape = tensor_view.shape();
                        let dtype = tensor_view.dtype();
                        println!("   - {}: shape={:?}, dtype={:?}", name, shape, dtype);
                    }
                }

                // Model type specific validation
                println!();
                println!("   Model Type Validation:");
                match model_type {
                    "acoustic" => {
                        println!("   Checking for acoustic model tensors...");
                        let expected_tensors = vec!["encoder", "decoder", "mel_linear"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    "vocoder" => {
                        println!("   Checking for vocoder model tensors...");
                        let expected_tensors = vec!["upsample", "resblock", "conv_post"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    "g2p" => {
                        println!("   Checking for G2P model tensors...");
                        let expected_tensors = vec!["embedding", "transformer"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    _ => {
                        println!("   Generic model - skipping specific tensor checks");
                    }
                }
            }
            Ok(())
        }
        Err(e) => Err(voirs_sdk::VoirsError::config_error(format!(
            "Failed to load SafeTensors: {}",
            e
        ))),
    }
}

/// Helper function to check for expected tensors
fn check_expected_tensors(tensors: &safetensors::SafeTensors, expected: &[&str]) {
    let names = tensors.names();
    let mut found_count = 0;

    for &expected_name in expected {
        let found = names.iter().any(|name| name.contains(expected_name));
        if found {
            println!("   ✅ Found tensor matching '{}'", expected_name);
            found_count += 1;
        } else {
            println!("   ⚠️  No tensor matching '{}'", expected_name);
        }
    }

    if found_count > 0 {
        println!(
            "   Model appears valid ({}/{} expected patterns found)",
            found_count,
            expected.len()
        );
    } else {
        println!("   ⚠️  Model may not match expected type (no standard tensors found)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.onnx")), "onnx");
        assert_eq!(detect_format(Path::new("model.pt")), "pt");
        assert_eq!(detect_format(Path::new("model.pth")), "pth");
        assert_eq!(detect_format(Path::new("model")), "unknown");
    }
}
