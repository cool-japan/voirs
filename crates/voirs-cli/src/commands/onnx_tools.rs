//! ONNX model tools: inspect, profile, graph visualization, and info.
//!
//! Provides CLI subcommands for working with ONNX models using the oxionnx
//! pure-Rust inference engine. All commands are feature-gated behind the `onnx` feature.

use std::path::PathBuf;

use clap::Subcommand;

/// ONNX model tool subcommands.
#[derive(Debug, Subcommand)]
pub enum OnnxCommand {
    /// Inspect an ONNX model: node count, parameters, op histogram
    Inspect {
        /// Path to the ONNX model file
        model_path: PathBuf,
        /// Show detailed per-operator breakdown
        #[arg(long, short)]
        detailed: bool,
    },
    /// Profile an ONNX model: run inference with dummy data and show per-node timing
    Profile {
        /// Path to the ONNX model file
        model_path: PathBuf,
        /// Number of warm-up runs before profiling
        #[arg(long, default_value = "1")]
        warmup: usize,
        /// Number of profiling runs
        #[arg(long, default_value = "1")]
        runs: usize,
    },
    /// Export computation graph as Graphviz DOT format
    Dot {
        /// Path to the ONNX model file
        model_path: PathBuf,
        /// Output file path (default: stdout)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Show model input/output names and shapes
    Info {
        /// Path to the ONNX model file
        model_path: PathBuf,
    },
}

/// Handle ONNX tool subcommands.
///
/// When the `onnx` feature is enabled, this delegates to the concrete
/// oxionnx-based implementations. Otherwise it returns an error.
#[cfg(feature = "onnx")]
pub fn handle_onnx_command(cmd: &OnnxCommand) -> anyhow::Result<()> {
    match cmd {
        OnnxCommand::Inspect {
            model_path,
            detailed,
        } => handle_inspect(model_path, *detailed),
        OnnxCommand::Profile {
            model_path,
            warmup,
            runs,
        } => handle_profile(model_path, *warmup, *runs),
        OnnxCommand::Dot { model_path, output } => handle_dot(model_path, output.as_deref()),
        OnnxCommand::Info { model_path } => handle_info(model_path),
    }
}

#[cfg(not(feature = "onnx"))]
pub fn handle_onnx_command(_cmd: &OnnxCommand) -> anyhow::Result<()> {
    anyhow::bail!("ONNX tools require the 'onnx' feature. Build with --features onnx")
}

// ---------------------------------------------------------------------------
// Feature-gated implementations
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
fn handle_inspect(model_path: &std::path::Path, detailed: bool) -> anyhow::Result<()> {
    use oxionnx::{OptLevel, Session, SessionBuilder};

    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .load(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    let info = session.model_info();

    println!("=== ONNX Model Inspection ===");
    println!("File: {}", model_path.display());
    println!(
        "Size: {:.2} MB",
        std::fs::metadata(model_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    );
    println!("Nodes: {}", info.node_count);
    println!(
        "Parameters: {} ({:.2} M)",
        info.parameter_count,
        info.parameter_count as f64 / 1_000_000.0
    );
    println!(
        "Weight memory: {:.2} MB",
        info.weight_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Inputs: {:?}", session.input_names());
    println!("Outputs: {:?}", session.output_names());

    if detailed {
        println!("\n--- Operator Histogram ---");
        let mut ops: Vec<_> = info.op_histogram.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        for (op, count) in &ops {
            println!("  {:<24} {}", op, count);
        }
    }

    Ok(())
}

#[cfg(feature = "onnx")]
fn handle_profile(model_path: &std::path::Path, warmup: usize, runs: usize) -> anyhow::Result<()> {
    use std::collections::HashMap;

    use oxionnx::{OptLevel, Session, SessionBuilder, Tensor};

    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .with_profiling()
        .load(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    // Build dummy inputs — each input gets a small 1x100 tensor of zeros.
    let input_names = session.input_names().to_vec();
    let build_inputs = || -> HashMap<&str, Tensor> {
        let mut inputs = HashMap::new();
        for name in &input_names {
            let tensor = Tensor::new(vec![0.0f32; 100], vec![1, 100]);
            inputs.insert(name.as_str(), tensor);
        }
        inputs
    };

    // Warm-up runs
    for _ in 0..warmup {
        let inputs = build_inputs();
        let _ = session.run(&inputs);
    }

    // Profiling runs
    for i in 0..runs {
        let inputs = build_inputs();
        let _ = session.run(&inputs);

        if let Some(profiles) = session.profiling_results() {
            println!("=== Profile Run {} ===", i + 1);
            let total: std::time::Duration = profiles.iter().map(|p| p.duration).sum();
            println!("Total inference time: {:?}", total);

            // Top 10 slowest nodes
            let mut sorted = profiles.clone();
            sorted.sort_by_key(|p| std::cmp::Reverse(p.duration));
            println!("\nTop 10 slowest nodes:");
            for (idx, p) in sorted.iter().take(10).enumerate() {
                let pct = if total.as_nanos() > 0 {
                    p.duration.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "  {}. {} ({}) - {:?} ({:.1}%)",
                    idx + 1,
                    p.node_name,
                    p.op_type,
                    p.duration,
                    pct
                );
            }

            // Aggregate by op type
            let mut op_times: HashMap<String, std::time::Duration> = HashMap::new();
            for p in &profiles {
                *op_times.entry(p.op_type.clone()).or_default() += p.duration;
            }
            let mut op_sorted: Vec<_> = op_times.iter().collect();
            op_sorted.sort_by(|a, b| b.1.cmp(a.1));
            println!("\nTime by operator type:");
            for (op, dur) in &op_sorted {
                let pct = if total.as_nanos() > 0 {
                    dur.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
                } else {
                    0.0
                };
                println!("  {:<24} {:?} ({:.1}%)", op, dur, pct);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "onnx")]
fn handle_dot(
    model_path: &std::path::Path,
    output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    use oxionnx::{OptLevel, SessionBuilder};

    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .load(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    let dot = session.export_dot();

    if let Some(output_path) = output {
        std::fs::write(output_path, &dot)
            .map_err(|e| anyhow::anyhow!("Failed to write DOT file: {}", e))?;
        println!("DOT graph written to {}", output_path.display());
    } else {
        println!("{}", dot);
    }

    Ok(())
}

#[cfg(feature = "onnx")]
fn handle_info(model_path: &std::path::Path) -> anyhow::Result<()> {
    use oxionnx::{OptLevel, SessionBuilder};

    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    // Load with memory pool to get estimated memory
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .load(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    let info = session.model_info();

    println!("=== ONNX Model Info ===");
    println!("File: {}", model_path.display());
    println!(
        "Size: {:.2} MB",
        std::fs::metadata(model_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    );

    println!("\nInputs:");
    for name in session.input_names() {
        println!("  - {}", name);
    }
    println!("\nOutputs:");
    for name in session.output_names() {
        println!("  - {}", name);
    }

    println!("\nModel Statistics:");
    println!("  Nodes: {}", info.node_count);
    println!(
        "  Parameters: {} ({:.2} M)",
        info.parameter_count,
        info.parameter_count as f64 / 1_000_000.0
    );
    println!(
        "  Weight memory: {:.2} MB",
        info.weight_bytes as f64 / (1024.0 * 1024.0)
    );

    if let Some(mem) = session.estimated_memory_bytes() {
        println!(
            "  Estimated peak memory: {:.2} MB",
            mem as f64 / (1024.0 * 1024.0)
        );
    }

    // Print op histogram summary (top 5)
    if !info.op_histogram.is_empty() {
        let mut ops: Vec<_> = info.op_histogram.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        println!("\nTop operators:");
        for (op, count) in ops.iter().take(5) {
            println!("  {:<24} {}", op, count);
        }
        if ops.len() > 5 {
            println!("  ... and {} more operator types", ops.len() - 5);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_command_variants() {
        // Verify the enum variants exist and can be constructed
        let _inspect = OnnxCommand::Inspect {
            model_path: PathBuf::from("test.onnx"),
            detailed: true,
        };
        let _profile = OnnxCommand::Profile {
            model_path: PathBuf::from("test.onnx"),
            warmup: 2,
            runs: 3,
        };
        let _dot = OnnxCommand::Dot {
            model_path: PathBuf::from("test.onnx"),
            output: Some(PathBuf::from("graph.dot")),
        };
        let _info = OnnxCommand::Info {
            model_path: PathBuf::from("test.onnx"),
        };
    }

    #[test]
    fn test_missing_model_file() {
        let cmd = OnnxCommand::Info {
            model_path: PathBuf::from("/nonexistent/path/model.onnx"),
        };
        let result = handle_onnx_command(&cmd);
        assert!(result.is_err());
    }
}
