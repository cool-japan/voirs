//! Vocoder inference command - convert mel spectrograms to audio

use crate::GlobalOptions;
use candle_core::{Device, Tensor};
use std::path::Path;
use voirs_sdk::Result;
use voirs_vocoder::models::diffwave::{DiffWave, SamplingMethod};

/// Run vocoder inference: mel spectrogram → audio waveform
///
/// # Arguments
/// * `checkpoint` - Path to SafeTensors checkpoint file
/// * `mel_path` - Path to mel spectrogram file (optional, will use dummy if None)
/// * `output` - Output audio file path
/// * `steps` - Number of diffusion steps for sampling
/// * `global` - Global CLI options
pub async fn run_vocoder_inference(
    checkpoint: &Path,
    mel_path: Option<&Path>,
    output: &Path,
    steps: usize,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🎵 VoiRS Vocoder Inference");
        println!("═══════════════════════════════════════");
        println!("Checkpoint: {}", checkpoint.display());
        if let Some(mel) = mel_path {
            println!("Mel spec:   {}", mel.display());
        } else {
            println!("Mel spec:   <generating dummy>");
        }
        println!("Output:     {}", output.display());
        println!("Steps:      {}", steps);
        println!("═══════════════════════════════════════\n");
    }

    // Determine device
    let device = if global.gpu {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "cuda"))]
        {
            if !global.quiet {
                println!("⚠️  GPU requested but CUDA not available, using CPU");
            }
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    if !global.quiet {
        println!("📦 Loading DiffWave model from checkpoint...");
    }

    // Load DiffWave model from SafeTensors
    let model = DiffWave::load_from_safetensors(checkpoint, device.clone()).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to load DiffWave model: {}", e))
    })?;

    if !global.quiet {
        println!("✓ Model loaded successfully");
        println!("  Parameters: {}", model.num_parameters());
        println!();
    }

    // Load or generate mel spectrogram
    let mel_tensor = if let Some(mel_file) = mel_path {
        if !global.quiet {
            println!("📊 Loading mel spectrogram from file...");
        }
        load_mel_spectrogram(mel_file, &device)?
    } else {
        if !global.quiet {
            println!("📊 Generating dummy mel spectrogram...");
        }
        generate_dummy_mel_spectrogram(&device)?
    };

    if !global.quiet {
        println!("✓ Mel spectrogram ready");
        println!("  Shape: {:?}", mel_tensor.dims());
        println!();
    }

    // Run inference
    if !global.quiet {
        println!("🔄 Running vocoder inference...");
        println!("  Sampling method: DDIM");
        println!("  Diffusion steps: {}", steps);
    }

    let sampling_method = SamplingMethod::DDIM { steps, eta: 0.0 };
    let audio_tensor = model
        .inference(&mel_tensor, sampling_method)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Inference failed: {}", e)))?;

    if !global.quiet {
        println!("✓ Inference complete");
        println!("  Audio shape: {:?}", audio_tensor.dims());
        println!();
    }

    // Save audio
    if !global.quiet {
        println!("💾 Saving audio to {}...", output.display());
    }

    save_audio_tensor(&audio_tensor, output, 22050)?;

    if !global.quiet {
        println!("✅ Vocoder inference complete!");
        println!("  Output: {}", output.display());
    }

    Ok(())
}

/// Load mel spectrogram from file
///
/// Supports multiple formats:
/// - NumPy (.npy): Native parser for NumPy binary format
/// - SafeTensors (.safetensors): Uses safetensors crate
/// - PyTorch (.pt, .pth): Requires conversion (see error message for guidance)
fn load_mel_spectrogram(path: &Path, device: &Device) -> Result<Tensor> {
    // Check file extension and load appropriately
    match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => load_numpy_file(path, device),
        Some("pt") | Some("pth") => load_pytorch_file(path, device),
        Some("safetensors") => load_safetensors_file(path, device),
        _ => Err(voirs_sdk::VoirsError::UnsupportedFileFormat {
            path: path.to_path_buf(),
            format: path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_string(),
        }),
    }
}

/// Load NumPy file (.npy)
fn load_numpy_file(path: &Path, device: &Device) -> Result<Tensor> {
    // Read entire file
    let data = std::fs::read(path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    // Parse NumPy .npy format manually
    // Format: Magic (6 bytes) + Version (2 bytes) + Header Len (2/4 bytes) + Header (JSON-like dict) + Data

    // Check magic number: b'\x93NUMPY'
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(voirs_sdk::VoirsError::config_error(
            "Invalid NumPy file: magic number mismatch",
        ));
    }

    let major_version = data[6];
    let minor_version = data[7];

    if major_version != 1 && major_version != 2 {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Unsupported NumPy version: {}.{}",
            major_version, minor_version
        )));
    }

    // Read header length (little-endian)
    let header_len = if major_version == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };

    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_end = header_start + header_len;

    if data.len() < header_end {
        return Err(voirs_sdk::VoirsError::config_error(
            "Invalid NumPy file: truncated header",
        ));
    }

    // Parse header (Python dict-like string)
    let header_str = std::str::from_utf8(&data[header_start..header_end]).map_err(|_| {
        voirs_sdk::VoirsError::config_error("Invalid NumPy header: not UTF-8")
    })?;

    // Extract shape from header (format: 'shape': (dim0, dim1, ...), )
    let shape = parse_numpy_shape(header_str)?;

    // Extract dtype (we only support float32 for now)
    let dtype = parse_numpy_dtype(header_str)?;
    if dtype != "f4" && dtype != "<f4" && dtype != "float32" {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Unsupported NumPy dtype: {}. Only float32 is supported.",
            dtype
        )));
    }

    // Read data
    let data_start = header_end;
    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * 4; // f32 = 4 bytes

    if data.len() < data_start + expected_bytes {
        return Err(voirs_sdk::VoirsError::config_error(
            "Invalid NumPy file: insufficient data",
        ));
    }

    // Convert bytes to f32 (little-endian)
    let f32_data: Vec<f32> = data[data_start..data_start + expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Create tensor
    let tensor = Tensor::from_vec(f32_data, shape.as_slice(), device).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to create tensor from NumPy data: {}", e))
    })?;

    Ok(tensor)
}

/// Parse shape from NumPy header
fn parse_numpy_shape(header: &str) -> Result<Vec<usize>> {
    // Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (80, 100), }
    // Extract shape tuple
    let shape_start = header.find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy header missing 'shape' field"))?;

    let shape_str = &header[shape_start..];
    let tuple_start = shape_str.find('(')
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy shape malformed"))?;
    let tuple_end = shape_str.find(')')
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy shape malformed"))?;

    let tuple_content = &shape_str[tuple_start + 1..tuple_end];

    if tuple_content.trim().is_empty() {
        // Scalar array
        return Ok(vec![1]);
    }

    // Parse dimensions
    let dims: Result<Vec<usize>> = tuple_content
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| voirs_sdk::VoirsError::config_error(format!("Invalid dimension: {}", s)))
        })
        .collect();

    dims
}

/// Parse dtype from NumPy header
fn parse_numpy_dtype(header: &str) -> Result<String> {
    // Extract descr field
    let descr_start = header.find("'descr':")
        .or_else(|| header.find("\"descr\":"))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy header missing 'descr' field"))?;

    let descr_str = &header[descr_start..];

    // Find the value (between quotes)
    let value_start = descr_str.find('\'')
        .or_else(|| descr_str.find('"'))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy descr malformed"))?;

    let value_str = &descr_str[value_start + 1..];
    let value_end = value_str.find('\'')
        .or_else(|| value_str.find('"'))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy descr malformed"))?;

    Ok(value_str[..value_end].to_string())
}

/// Load PyTorch file (.pt, .pth)
fn load_pytorch_file(path: &Path, _device: &Device) -> Result<Tensor> {
    // PyTorch .pt files use Python's pickle format, which is complex to parse in pure Rust
    // For now, we provide helpful guidance for users

    Err(voirs_sdk::VoirsError::config_error(format!(
        "PyTorch .pt file loading requires Python interop or conversion.\n\
        \n\
        Alternatives:\n\
        1. Convert to NumPy: python -c \"import torch, numpy as np; np.save('output.npy', torch.load('{}').numpy())\"\n\
        2. Convert to SafeTensors: Use safetensors.torch.save_file() in Python\n\
        3. Use ONNX format: Export model to ONNX and use --input-format onnx\n\
        \n\
        For native PyTorch support, compile with 'tch-rs' feature (requires libtorch).",
        path.display()
    )))
}

/// Load SafeTensors file
fn load_safetensors_file(path: &Path, device: &Device) -> Result<Tensor> {
    use safetensors::SafeTensors;

    let data = std::fs::read(path).map_err(|e| voirs_sdk::VoirsError::IoError {
        path: path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to load SafeTensors: {}", e))
    })?;

    // Assume the first tensor is the mel spectrogram
    let names = tensors.names();
    let tensor_name = names
        .first()
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("No tensors found in file"))?;

    let tensor_view = tensors
        .tensor(tensor_name)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to get tensor: {}", e)))?;

    let shape: Vec<usize> = tensor_view.shape().to_vec();
    let data = tensor_view.data();

    // Convert bytes to f32
    let f32_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let tensor = Tensor::from_vec(f32_data, shape.as_slice(), device).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to create tensor: {}", e))
    })?;
    Ok(tensor)
}

/// Generate dummy mel spectrogram for testing
fn generate_dummy_mel_spectrogram(device: &Device) -> Result<Tensor> {
    // Create a dummy mel spectrogram: [batch=1, mel_channels=80, time=100]
    let batch_size = 1;
    let mel_channels = 80;
    let time_frames = 100;

    // Generate random values (in practice, this would be from an acoustic model)
    let data: Vec<f32> = (0..(batch_size * mel_channels * time_frames))
        .map(|_| fastrand::f32() * 2.0 - 1.0) // Random values between -1 and 1
        .collect();

    let tensor =
        Tensor::from_vec(data, (batch_size, mel_channels, time_frames), device).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to create tensor: {}", e))
        })?;
    Ok(tensor)
}

/// Save audio tensor to WAV file
fn save_audio_tensor(tensor: &Tensor, output: &Path, sample_rate: u32) -> Result<()> {
    use hound::{WavSpec, WavWriter};

    // Extract audio data from tensor
    let audio_data: Vec<f32> = tensor
        .flatten_all()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to flatten tensor: {}", e))
        })?
        .to_vec1()
        .map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to convert tensor to vec: {}", e))
        })?;

    // Create WAV spec
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Create WAV writer
    let mut writer =
        WavWriter::create(output, spec).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: output.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: std::io::Error::new(std::io::ErrorKind::Other, e),
        })?;

    // Write samples (convert f32 to i16)
    for &sample in &audio_data {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: output.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: std::io::Error::new(std::io::ErrorKind::Other, e),
            })?;
    }

    writer
        .finalize()
        .map_err(|e| voirs_sdk::VoirsError::IoError {
            path: output.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: std::io::Error::new(std::io::ErrorKind::Other, e),
        })?;

    Ok(())
}
