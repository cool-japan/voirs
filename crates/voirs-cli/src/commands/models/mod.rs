//! Model management commands.
//!
//! This module provides commands for managing TTS models, including
//! listing, downloading, benchmarking, and optimizing models.

use voirs::config::AppConfig;
use voirs::error::Result;
use crate::GlobalOptions;

pub mod list;
pub mod download;
pub mod benchmark;
pub mod optimize;

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
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    benchmark::run_benchmark_models(model_ids, iterations, config, global).await
}

/// Optimize model for current hardware
pub async fn run_optimize_model(
    model_id: &str,
    output_path: Option<&str>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    optimize::run_optimize_model(model_id, output_path, config, global).await
}