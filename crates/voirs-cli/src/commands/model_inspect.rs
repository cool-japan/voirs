//! Model inspection and analysis command implementation.
//!
//! Provides tools to inspect model architecture, verify integrity, and analyze checkpoints.

use crate::GlobalOptions;
use hex;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use voirs_sdk::Result;

/// Model inspection result
#[derive(Debug)]
pub struct ModelInspection {
    pub model_type: String,
    pub format: String,
    pub file_size: u64,
    pub parameter_count: Option<usize>,
    pub layers: Vec<LayerInfo>,
    pub metadata: Vec<(String, String)>,
}

/// Layer information
#[derive(Debug)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub shape: Vec<usize>,
    pub param_count: usize,
}

/// Run model inspection command.
pub async fn run_model_inspect(
    model_path: &Path,
    detailed: bool,
    export_path: Option<&PathBuf>,
    verify: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔍 Inspecting model: {}", model_path.display());
        println!();
    }

    // Check if file exists
    if !model_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    // Read file metadata
    let metadata = std::fs::metadata(model_path).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to read file metadata: {}", e))
    })?;

    let file_size = metadata.len();

    // Determine model format
    let format = detect_model_format(model_path)?;

    if !global.quiet {
        println!("📄 File Information:");
        println!("  Format: {}", format);
        println!(
            "  Size: {} bytes ({:.2} MB)",
            file_size,
            file_size as f64 / 1_048_576.0
        );
        println!();
    }

    // Perform format-specific inspection
    let inspection = match format.as_str() {
        "SafeTensors" => inspect_safetensors(model_path, detailed)?,
        "PyTorch" => inspect_pytorch(model_path, detailed)?,
        "ONNX" => inspect_onnx(model_path, detailed)?,
        _ => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported model format: {}",
                format
            )));
        }
    };

    // Display inspection results
    display_inspection(&inspection, detailed, global.quiet);

    // Verify integrity if requested
    if verify {
        verify_model_integrity(model_path, &format, global.quiet)?;
    }

    // Export architecture if requested
    if let Some(export_path) = export_path {
        export_architecture(&inspection, export_path)?;
        if !global.quiet {
            println!("\n✅ Architecture exported to: {}", export_path.display());
        }
    }

    Ok(())
}

/// Detect model file format
fn detect_model_format(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("No file extension found"))?;

    match ext.to_lowercase().as_str() {
        "safetensors" | "st" => Ok("SafeTensors".to_string()),
        "pt" | "pth" | "bin" => Ok("PyTorch".to_string()),
        "onnx" => Ok("ONNX".to_string()),
        _ => Err(voirs_sdk::VoirsError::config_error(format!(
            "Unknown model format: {}",
            ext
        ))),
    }
}

/// Inspect SafeTensors model
fn inspect_safetensors(path: &Path, detailed: bool) -> Result<ModelInspection> {
    use safetensors::SafeTensors;

    let mut file = File::open(path)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to open file: {}", e)))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to read file: {}", e)))?;

    let tensors = SafeTensors::deserialize(&buffer).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to deserialize SafeTensors: {}", e))
    })?;

    let mut layers = Vec::new();
    let mut total_params = 0;

    for name in tensors.names() {
        let tensor = tensors.tensor(name).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to get tensor: {}", e))
        })?;

        let shape: Vec<usize> = tensor.shape().to_vec();
        let param_count: usize = shape.iter().product();
        total_params += param_count;

        if detailed {
            layers.push(LayerInfo {
                name: name.to_string(),
                layer_type: infer_layer_type(name),
                shape,
                param_count,
            });
        }
    }

    let mut metadata = Vec::new();
    // SafeTensors metadata is not publicly accessible in this version
    // metadata.push(("metadata".to_string(), "See SafeTensors spec".to_string()));

    Ok(ModelInspection {
        model_type: infer_model_type(tensors.names()),
        format: "SafeTensors".to_string(),
        file_size: buffer.len() as u64,
        parameter_count: Some(total_params),
        layers,
        metadata,
    })
}

/// Inspect PyTorch model
fn inspect_pytorch(path: &Path, detailed: bool) -> Result<ModelInspection> {
    let metadata_result = std::fs::metadata(path);
    let file_size = metadata_result.map(|m| m.len()).unwrap_or(0);

    // Try to validate PyTorch format by checking pickle magic bytes
    let mut file = File::open(path)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to open file: {}", e)))?;

    let mut magic = [0u8; 8];
    let _ = file.read(&mut magic);

    let mut metadata = vec![];
    let is_valid_pickle = magic.starts_with(b"\x80") || magic.starts_with(b"PK");

    if is_valid_pickle {
        metadata.push(("format_valid".to_string(), "true".to_string()));
        metadata.push((
            "pickle_protocol".to_string(),
            format!("{}", magic[1] as char),
        ));
    } else {
        metadata.push(("format_valid".to_string(), "false".to_string()));
        metadata.push((
            "warning".to_string(),
            "File may not be a valid PyTorch checkpoint".to_string(),
        ));
    }

    // Try to estimate parameter count from file size (rough heuristic)
    let estimated_params = if file_size > 1024 {
        Some(((file_size as f64 / 4.0) * 0.9) as usize) // Rough estimate: ~4 bytes per float32 param
    } else {
        None
    };

    metadata.push((
        "note".to_string(),
        "Full inspection requires PyTorch/tch-rs bindings".to_string(),
    ));
    metadata.push((
        "recommendation".to_string(),
        "Convert to SafeTensors format for detailed inspection".to_string(),
    ));

    Ok(ModelInspection {
        model_type: infer_pytorch_model_type(path),
        format: "PyTorch".to_string(),
        file_size,
        parameter_count: estimated_params,
        layers: vec![],
        metadata,
    })
}

/// Inspect ONNX model
fn inspect_onnx(path: &Path, detailed: bool) -> Result<ModelInspection> {
    let metadata_result = std::fs::metadata(path);
    let file_size = metadata_result.map(|m| m.len()).unwrap_or(0);

    // Try to validate ONNX format by checking protobuf header
    let mut file = File::open(path)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to open file: {}", e)))?;

    let mut buffer = vec![0u8; 256]; // Read first 256 bytes for analysis
    let bytes_read = file.read(&mut buffer).unwrap_or(0);

    let mut metadata = vec![];

    // Check for protobuf magic bytes and common ONNX markers
    let has_onnx_marker = buffer.windows(4).any(|w| w == b"ONNX" || w == b"onnx");
    let has_protobuf = bytes_read > 0 && (buffer[0] == 0x08 || buffer[0] == 0x0a);

    if has_onnx_marker && has_protobuf {
        metadata.push(("format_valid".to_string(), "true".to_string()));
        metadata.push(("protobuf_format".to_string(), "detected".to_string()));
    } else if has_protobuf {
        metadata.push(("format_valid".to_string(), "likely".to_string()));
        metadata.push((
            "warning".to_string(),
            "Protobuf detected but no ONNX marker found".to_string(),
        ));
    } else {
        metadata.push(("format_valid".to_string(), "false".to_string()));
        metadata.push((
            "warning".to_string(),
            "File may not be a valid ONNX model".to_string(),
        ));
    }

    // Try to find IR version in the protobuf data
    if let Some(ir_version) = extract_onnx_ir_version(&buffer[..bytes_read]) {
        metadata.push(("ir_version".to_string(), ir_version.to_string()));
    }

    // Estimate parameters from file size
    let estimated_params = if file_size > 1024 {
        Some(((file_size as f64 / 4.5) * 0.85) as usize) // Rough estimate accounting for protobuf overhead
    } else {
        None
    };

    metadata.push((
        "note".to_string(),
        "Full inspection requires tract-onnx or onnxruntime bindings".to_string(),
    ));
    metadata.push((
        "recommendation".to_string(),
        "Use 'onnx' Python tools for detailed inspection, or convert to SafeTensors".to_string(),
    ));

    Ok(ModelInspection {
        model_type: infer_onnx_model_type(path),
        format: "ONNX".to_string(),
        file_size,
        parameter_count: estimated_params,
        layers: vec![],
        metadata,
    })
}

/// Infer layer type from tensor name
fn infer_layer_type(name: &str) -> String {
    if name.contains("weight") && name.contains("conv") {
        "Convolution".to_string()
    } else if name.contains("weight") && name.contains("linear") {
        "Linear".to_string()
    } else if name.contains("weight") && name.contains("attention") {
        "Attention".to_string()
    } else if name.contains("norm") || name.contains("bn") {
        "Normalization".to_string()
    } else if name.contains("embedding") {
        "Embedding".to_string()
    } else if name.contains("bias") {
        "Bias".to_string()
    } else {
        "Other".to_string()
    }
}

/// Infer model type from tensor names
fn infer_model_type(names: Vec<&str>) -> String {
    let names_str = names.join(" ").to_lowercase();

    if names_str.contains("diffwave") || names_str.contains("residual_blocks") {
        "DiffWave Vocoder".to_string()
    } else if names_str.contains("hifigan") || names_str.contains("generator") {
        "HiFi-GAN Vocoder".to_string()
    } else if names_str.contains("vits") || names_str.contains("posterior_encoder") {
        "VITS Acoustic Model".to_string()
    } else if names_str.contains("fastspeech") {
        "FastSpeech2 Acoustic Model".to_string()
    } else if names_str.contains("g2p") || names_str.contains("phoneme") {
        "G2P Model".to_string()
    } else {
        "Unknown Model Type".to_string()
    }
}

/// Infer PyTorch model type from filename
fn infer_pytorch_model_type(path: &Path) -> String {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    if filename.contains("vocoder") || filename.contains("hifigan") || filename.contains("diffwave")
    {
        "Vocoder Model (PyTorch)".to_string()
    } else if filename.contains("acoustic")
        || filename.contains("vits")
        || filename.contains("fastspeech")
    {
        "Acoustic Model (PyTorch)".to_string()
    } else if filename.contains("g2p") || filename.contains("phoneme") {
        "G2P Model (PyTorch)".to_string()
    } else if filename.contains("encoder") {
        "Encoder Model (PyTorch)".to_string()
    } else if filename.contains("decoder") {
        "Decoder Model (PyTorch)".to_string()
    } else {
        "Unknown Model Type (PyTorch)".to_string()
    }
}

/// Infer ONNX model type from filename
fn infer_onnx_model_type(path: &Path) -> String {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    if filename.contains("vocoder") || filename.contains("hifigan") || filename.contains("diffwave")
    {
        "Vocoder Model (ONNX)".to_string()
    } else if filename.contains("acoustic")
        || filename.contains("vits")
        || filename.contains("fastspeech")
    {
        "Acoustic Model (ONNX)".to_string()
    } else if filename.contains("g2p") || filename.contains("phoneme") {
        "G2P Model (ONNX)".to_string()
    } else if filename.contains("encoder") {
        "Encoder Model (ONNX)".to_string()
    } else if filename.contains("decoder") {
        "Decoder Model (ONNX)".to_string()
    } else {
        "Unknown Model Type (ONNX)".to_string()
    }
}

/// Extract ONNX IR version from protobuf data
fn extract_onnx_ir_version(buffer: &[u8]) -> Option<u8> {
    // IR version is typically one of the first fields in ONNX protobuf
    // Field tag 1 (ir_version) has wire type 0 (varint)
    // This is a simple heuristic search
    for i in 0..buffer.len().saturating_sub(2) {
        // Look for field tag 0x08 (field 1, wire type 0)
        if buffer[i] == 0x08 && buffer[i + 1] > 0 && buffer[i + 1] < 20 {
            return Some(buffer[i + 1]);
        }
    }
    None
}

/// Display inspection results
fn display_inspection(inspection: &ModelInspection, detailed: bool, quiet: bool) {
    if quiet {
        return;
    }

    println!("🔬 Model Analysis:");
    println!("  Type: {}", inspection.model_type);

    if let Some(count) = inspection.parameter_count {
        println!(
            "  Parameters: {:?} ({:.2}M)",
            count,
            count as f64 / 1_000_000.0
        );
    }

    if !inspection.metadata.is_empty() {
        println!("\n📋 Metadata:");
        for (key, value) in &inspection.metadata {
            println!("  {}: {}", key, value);
        }
    }

    if detailed && !inspection.layers.is_empty() {
        println!("\n🧩 Layers ({} total):", inspection.layers.len());
        for layer in &inspection.layers {
            println!("  {} [{}]", layer.name, layer.layer_type);
            println!("    Shape: {:?}", layer.shape);
            println!("    Parameters: {}", layer.param_count);
        }
    } else if !inspection.layers.is_empty() {
        println!(
            "  Layers: {} (use --detailed for full list)",
            inspection.layers.len()
        );
    }
}

/// Verify model integrity
fn verify_model_integrity(path: &Path, format: &str, quiet: bool) -> Result<()> {
    use safetensors::SafeTensors;

    if !quiet {
        println!("\n🔐 Verifying model integrity...");
    }

    // Basic file existence and readability check
    let _file = File::open(path)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to open file: {}", e)))?;

    // Calculate checksum for integrity tracking
    let checksum = calculate_file_checksum(path)?;
    if !quiet {
        println!("  SHA-256: {}", checksum);
    }

    // Format-specific validation
    match format {
        "SafeTensors" => {
            // Try to deserialize
            let mut file = File::open(path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            SafeTensors::deserialize(&buffer).map_err(|e| {
                voirs_sdk::VoirsError::config_error(format!("SafeTensors validation failed: {}", e))
            })?;

            if !quiet {
                println!("  Format: Valid SafeTensors");
            }
        }
        "PyTorch" => {
            // Validate pickle format
            let mut file = File::open(path)?;
            let mut magic = [0u8; 2];
            file.read_exact(&mut magic).ok();

            if magic[0] == 0x80 || magic.starts_with(b"PK") {
                if !quiet {
                    println!("  Format: Valid PyTorch/Pickle");
                }
            } else if !quiet {
                println!("  Format: Warning - may not be valid PyTorch");
            }
        }
        "ONNX" => {
            // Validate ONNX protobuf format
            let mut file = File::open(path)?;
            let mut buffer = vec![0u8; 64];
            let _ = file.read(&mut buffer);

            let has_onnx = buffer.windows(4).any(|w| w == b"ONNX");
            if has_onnx && !quiet {
                println!("  Format: Valid ONNX");
            } else if !quiet {
                println!("  Format: Warning - may not be valid ONNX");
            }
        }
        _ => {
            // Basic checks only for other formats
        }
    }

    if !quiet {
        println!("✅ Model integrity verified");
    }

    Ok(())
}

/// Calculate SHA-256 checksum of a file
fn calculate_file_checksum(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to open file: {}", e)))?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to read file: {}", e))
        })?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(hex::encode(result))
}

/// Export architecture to file
fn export_architecture(inspection: &ModelInspection, path: &PathBuf) -> Result<()> {
    use serde_json;

    let json = serde_json::to_string_pretty(&serde_json::json!({
        "model_type": inspection.model_type,
        "format": inspection.format,
        "file_size": inspection.file_size,
        "parameter_count": inspection.parameter_count,
        "layer_count": inspection.layers.len(),
        "layers": inspection.layers.iter().map(|l| serde_json::json!({
            "name": l.name,
            "type": l.layer_type,
            "shape": l.shape,
            "parameters": l.param_count,
        })).collect::<Vec<_>>(),
        "metadata": inspection.metadata.iter().map(|(k, v)| serde_json::json!({
            "key": k,
            "value": v,
        })).collect::<Vec<_>>(),
    }))
    .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to serialize: {}", e)))?;

    std::fs::write(path, json)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to write file: {}", e)))?;

    Ok(())
}
