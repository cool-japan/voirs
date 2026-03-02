//! Vocoder inference command - convert mel spectrograms to audio

use crate::GlobalOptions;
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use voirs_sdk::Result;
use voirs_vocoder::models::diffwave::{DiffWave, SamplingMethod};

/// Configuration for vocoder inference operations
///
/// Consolidates parameters for converting mel spectrograms to audio waveforms
/// using trained vocoder models. Supports both single-file and batch processing.
///
/// # Examples
///
/// Single file inference:
/// ```no_run
/// use voirs_cli::commands::vocoder_inference::VocoderInferenceConfig;
/// use std::path::Path;
///
/// let config = VocoderInferenceConfig {
///     checkpoint: Path::new("./checkpoints/vocoder.safetensors"),
///     mel_path: Some(Path::new("./input.mel")),
///     output: Path::new("./output.wav"),
///     steps: 50,
///     quality: Some("balanced"),
///     batch_input: None,
///     batch_output: None,
///     show_metrics: false,
/// };
/// ```
///
/// Batch processing:
/// ```no_run
/// use voirs_cli::commands::vocoder_inference::VocoderInferenceConfig;
/// use std::path::{Path, PathBuf};
///
/// let config = VocoderInferenceConfig {
///     checkpoint: Path::new("./checkpoints/vocoder.safetensors"),
///     mel_path: None,
///     output: Path::new("./output_dir"),
///     steps: 50,
///     quality: Some("high"),
///     batch_input: Some(&PathBuf::from("./mel_dir")),
///     batch_output: Some(&PathBuf::from("./audio_dir")),
///     show_metrics: true,
/// };
/// ```
#[derive(Debug)]
pub struct VocoderInferenceConfig<'a> {
    /// Path to trained vocoder checkpoint file
    pub checkpoint: &'a Path,
    /// Optional path to input mel spectrogram file (single file mode)
    pub mel_path: Option<&'a Path>,
    /// Output path for generated audio file or batch directory
    pub output: &'a Path,
    /// Number of diffusion steps for generation (higher = better quality, slower)
    pub steps: usize,
    /// Quality preset: "fast" (20 steps), "balanced" (50 steps), or "high" (100 steps)
    pub quality: Option<&'a str>,
    /// Optional directory for batch input (batch mode)
    pub batch_input: Option<&'a PathBuf>,
    /// Optional directory for batch output (batch mode)
    pub batch_output: Option<&'a PathBuf>,
    /// Whether to display performance metrics after inference
    pub show_metrics: bool,
}

/// Quality preset for vocoder inference
#[derive(Debug, Clone, Copy)]
enum QualityPreset {
    Fast,     // 20 steps, faster generation
    Balanced, // 50 steps, balance of speed and quality
    High,     // 100 steps, best quality
}

impl QualityPreset {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fast" => Ok(Self::Fast),
            "balanced" => Ok(Self::Balanced),
            "high" => Ok(Self::High),
            _ => Err(voirs_sdk::VoirsError::config_error(format!(
                "Invalid quality preset: {}. Use 'fast', 'balanced', or 'high'",
                s
            ))),
        }
    }

    fn steps(&self) -> usize {
        match self {
            Self::Fast => 20,
            Self::Balanced => 50,
            Self::High => 100,
        }
    }
}

/// Run vocoder inference: mel spectrogram → audio waveform
///
/// # Arguments
/// * `config` - Vocoder inference configuration
/// * `global` - Global CLI options
pub async fn run_vocoder_inference(
    config: VocoderInferenceConfig<'_>,
    global: &GlobalOptions,
) -> Result<()> {
    // Check for batch mode
    if config.batch_input.is_some() || config.batch_output.is_some() {
        if config.batch_input.is_none() || config.batch_output.is_none() {
            return Err(voirs_sdk::VoirsError::config_error(
                "Batch mode requires both --batch-input and --batch-output",
            ));
        }
        return run_batch_inference(
            config.checkpoint,
            config.batch_input.expect("checked is_none above"),
            config.batch_output.expect("checked is_none above"),
            config.steps,
            config.quality,
            config.show_metrics,
            global,
        )
        .await;
    }

    // Single file mode
    run_single_inference(
        config.checkpoint,
        config.mel_path,
        config.output,
        config.steps,
        config.quality,
        config.show_metrics,
        global,
    )
    .await
}

/// Run single file inference
async fn run_single_inference(
    checkpoint: &Path,
    mel_path: Option<&Path>,
    output: &Path,
    mut steps: usize,
    quality: Option<&str>,
    show_metrics: bool,
    global: &GlobalOptions,
) -> Result<()> {
    // Apply quality preset if specified
    if let Some(quality_str) = quality {
        let preset = QualityPreset::from_str(quality_str)?;
        steps = preset.steps();
        if !global.quiet {
            println!("Using quality preset: {:?} ({} steps)", preset, steps);
        }
    }
    use std::time::Instant;
    let total_start = Instant::now();

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

    let total_time = total_start.elapsed();

    if !global.quiet {
        println!("✅ Vocoder inference complete!");
        println!("  Output: {}", output.display());
    }

    // Show metrics if requested
    if show_metrics {
        println!();
        println!("╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌");
        println!("Performance Metrics:");
        println!("╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌");
        println!("Total time:        {:.3}s", total_time.as_secs_f64());
        if let Ok(dims) = audio_tensor.dims3() {
            let (_, _, samples) = dims;
            let duration_sec = samples as f64 / 22050.0;
            let rtf = total_time.as_secs_f64() / duration_sec;
            println!("Audio duration:    {:.2}s", duration_sec);
            println!("Real-time factor:  {:.3}x", rtf);
        }
        println!("╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌");
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
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| voirs_sdk::VoirsError::config_error("Invalid NumPy header: not UTF-8"))?;

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
        voirs_sdk::VoirsError::config_error(format!(
            "Failed to create tensor from NumPy data: {}",
            e
        ))
    })?;

    Ok(tensor)
}

/// Parse shape from NumPy header
fn parse_numpy_shape(header: &str) -> Result<Vec<usize>> {
    // Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (80, 100), }
    // Extract shape tuple
    let shape_start = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy header missing 'shape' field"))?;

    let shape_str = &header[shape_start..];
    let tuple_start = shape_str
        .find('(')
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy shape malformed"))?;
    let tuple_end = shape_str
        .find(')')
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
            s.trim().parse::<usize>().map_err(|_| {
                voirs_sdk::VoirsError::config_error(format!("Invalid dimension: {}", s))
            })
        })
        .collect();

    dims
}

/// Parse dtype from NumPy header
fn parse_numpy_dtype(header: &str) -> Result<String> {
    // Extract descr field
    let descr_start = header
        .find("'descr':")
        .or_else(|| header.find("\"descr\":"))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy header missing 'descr' field"))?;

    let descr_str = &header[descr_start..];

    // Find the value (between quotes)
    let value_start = descr_str
        .find('\'')
        .or_else(|| descr_str.find('"'))
        .ok_or_else(|| voirs_sdk::VoirsError::config_error("NumPy descr malformed"))?;

    let value_str = &descr_str[value_start + 1..];
    let value_end = value_str
        .find('\'')
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
            source: std::io::Error::other(e),
        })?;

    // Write samples (convert f32 to i16)
    for &sample in &audio_data {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: output.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: std::io::Error::other(e),
            })?;
    }

    writer
        .finalize()
        .map_err(|e| voirs_sdk::VoirsError::IoError {
            path: output.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: std::io::Error::other(e),
        })?;

    Ok(())
}

/// Run batch inference on directory of mel spectrograms
async fn run_batch_inference(
    checkpoint: &Path,
    input_dir: &Path,
    output_dir: &Path,
    mut steps: usize,
    quality: Option<&str>,
    show_metrics: bool,
    global: &GlobalOptions,
) -> Result<()> {
    use std::time::Instant;

    // Apply quality preset
    if let Some(quality_str) = quality {
        let preset = QualityPreset::from_str(quality_str)?;
        steps = preset.steps();
    }

    if !global.quiet {
        println!("🎵 VoiRS Batch Vocoder Inference");
        println!("═══════════════════════════════════════");
        println!("Checkpoint:  {}", checkpoint.display());
        println!("Input dir:   {}", input_dir.display());
        println!("Output dir:  {}", output_dir.display());
        println!("Steps:       {}", steps);
        if let Some(q) = quality {
            println!("Quality:     {}", q);
        }
        println!("═══════════════════════════════════════\n");
    }

    // Validate input directory
    if !input_dir.is_dir() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Input directory not found: {}",
            input_dir.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Find all mel spectrogram files
    let mel_files: Vec<_> = std::fs::read_dir(input_dir)
        .map_err(|e| voirs_sdk::VoirsError::IoError {
            path: input_dir.to_path_buf(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: e,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|ext| matches!(ext, "npy" | "safetensors" | "pt" | "pth"))
                .unwrap_or(false)
        })
        .collect();

    if mel_files.is_empty() {
        return Err(voirs_sdk::VoirsError::config_error(
            "No mel spectrogram files found in input directory",
        ));
    }

    if !global.quiet {
        println!("Found {} mel spectrogram files", mel_files.len());
        println!();
    }

    // Load model once
    let device = if global.gpu {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    let model = DiffWave::load_from_safetensors(checkpoint, device.clone())?;

    // Performance tracking
    let mut total_time = 0.0;
    let mut successful = 0;
    let mut failed = 0;
    let batch_start = Instant::now();

    // Process each file
    for (idx, mel_file) in mel_files.iter().enumerate() {
        let file_start = Instant::now();

        let output_name = mel_file
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("output");
        let output_path = output_dir.join(format!("{}.wav", output_name));

        if !global.quiet {
            println!(
                "[{}/{}] Processing {}...",
                idx + 1,
                mel_files.len(),
                mel_file.display()
            );
        }

        // Process file
        let result =
            process_single_mel(&model, mel_file, &output_path, steps, &device, global).await;

        let file_time = file_start.elapsed().as_secs_f64();
        total_time += file_time;

        match result {
            Ok(_) => {
                successful += 1;
                if !global.quiet {
                    println!("  ✓ Complete in {:.2}s", file_time);
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ Failed: {}", e);
            }
        }
    }

    let total_elapsed = batch_start.elapsed().as_secs_f64();

    // Display results
    if !global.quiet || show_metrics {
        println!();
        println!("╔═══════════════════════════════════════╗");
        println!("║       Batch Inference Complete        ║");
        println!("╠═══════════════════════════════════════╣");
        println!("║ Total files:    {:<21} ║", mel_files.len());
        println!("║ Successful:     {:<21} ║", successful);
        println!("║ Failed:         {:<21} ║", failed);
        println!("║ Total time:     {:<18.2}s ║", total_elapsed);
        println!(
            "║ Avg time/file:  {:<18.2}s ║",
            total_time / mel_files.len() as f64
        );
        if successful > 0 {
            println!(
                "║ Throughput:     {:<18.2}/s ║",
                successful as f64 / total_elapsed
            );
        }
        println!("╚═══════════════════════════════════════╝");
    }

    Ok(())
}

/// Process a single mel spectrogram file
async fn process_single_mel(
    model: &DiffWave,
    mel_path: &Path,
    output_path: &Path,
    steps: usize,
    device: &Device,
    _global: &GlobalOptions,
) -> Result<()> {
    // Load mel spectrogram
    let mel_tensor = load_mel_spectrogram(mel_path, device)?;

    // Run inference
    let sampling_method = SamplingMethod::DDIM { steps, eta: 0.0 };
    let audio_tensor = model
        .inference(&mel_tensor, sampling_method)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Inference failed: {}", e)))?;

    // Save audio
    save_audio_tensor(&audio_tensor, output_path, 22050)?;

    Ok(())
}
