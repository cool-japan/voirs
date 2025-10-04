//! Model management commands.
//!
//! This module provides commands for managing TTS models, including
//! listing, downloading, benchmarking, and optimizing models.

use crate::GlobalOptions;
use voirs_sdk::config::AppConfig;
use voirs_sdk::Result;

pub mod benchmark;
pub mod download;
pub mod list;
pub mod optimize;
pub mod safetensors_support;

/// List available models
pub async fn run_list_models(
    backend: Option<&str>,
    detailed: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    list::run_list_models(backend, detailed, config, global).await
}

/// Download a model
pub async fn run_download_model(
    model_id: &str,
    force: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    download::run_download_model(model_id, force, config, global).await
}

/// Benchmark models
pub async fn run_benchmark_models(
    model_ids: &[String],
    iterations: u32,
    include_accuracy: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    benchmark::run_benchmark_models(model_ids, iterations, include_accuracy, config, global).await
}

/// Optimize model for current hardware
pub async fn run_optimize_model(
    model_id: &str,
    output_path: Option<&str>,
    strategy: Option<&str>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    optimize::run_optimize_model(model_id, output_path, strategy, config, global).await
}
