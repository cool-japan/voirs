//! Parallel processing for batch operations.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use voirs::config::AppConfig;
use voirs::error::Result;
use voirs::types::{AudioFormat, SynthesisConfig};
use voirs::VoirsPipeline;
use indicatif::{ProgressBar, ProgressStyle};
use crate::GlobalOptions;
use super::{BatchConfig, files::BatchInput};

/// Result of processing a single batch item
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Input that was processed
    pub input: BatchInput,
    /// Whether processing succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Output file path if successful
    pub output_path: Option<std::path::PathBuf>,
    /// Processing time
    pub duration: Duration,
    /// Generated audio duration
    pub audio_duration: Option<f32>,
}

/// Statistics for batch processing
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    /// Total items processed
    pub total_items: usize,
    /// Successfully processed items
    pub successful_items: usize,
    /// Failed items
    pub failed_items: usize,
    /// Total processing time
    pub total_time: Duration,
    /// Average processing time per item
    pub avg_time_per_item: Duration,
    /// Total audio duration generated
    pub total_audio_duration: f32,
    /// Items processed per second
    pub throughput: f32,
}

/// Process multiple inputs in parallel
pub async fn process_inputs_parallel(
    inputs: &[BatchInput],
    batch_config: &BatchConfig,
    app_config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if inputs.is_empty() {
        if !global.quiet {
            println!("No inputs to process");
        }
        return Ok(());
    }
    
    let start_time = Instant::now();
    
    // Create progress bar
    let progress_bar = if !global.quiet {
        let pb = ProgressBar::new(inputs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        Some(pb)
    } else {
        None
    };
    
    // Create channels for results
    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<ProcessingResult>();
    
    // Create semaphore to limit concurrent workers
    let semaphore = Arc::new(Semaphore::new(batch_config.workers));
    
    // Build pipeline once for all workers
    let pipeline = Arc::new(
        VoirsPipeline::builder()
            .with_quality(batch_config.quality)
            .with_gpu_acceleration(app_config.pipeline.use_gpu || global.gpu)
            .build()
            .await?
    );
    
    // Spawn worker tasks
    let mut handles = Vec::new();
    
    for (index, input) in inputs.iter().enumerate() {
        let semaphore = semaphore.clone();
        let pipeline = pipeline.clone();
        let batch_config = batch_config.clone();
        let input = input.clone();
        let result_tx = result_tx.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            let result = process_single_input(input, index, &pipeline, &batch_config).await;
            let _ = result_tx.send(result);
        });
        
        handles.push(handle);
    }
    
    // Drop the original sender so the receiver knows when all workers are done
    drop(result_tx);
    
    // Collect results
    let mut results = Vec::new();
    let mut successful_count = 0;
    let mut failed_count = 0;
    let mut total_audio_duration = 0.0;
    
    while let Some(result) = result_rx.recv().await {
        if result.success {
            successful_count += 1;
            if let Some(duration) = result.audio_duration {
                total_audio_duration += duration;
            }
        } else {
            failed_count += 1;
            if !global.quiet {
                if let Some(error) = &result.error {
                    tracing::warn!("Failed to process '{}': {}", result.input.id, error);
                }
            }
        }
        
        results.push(result);
        
        // Update progress
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message(format!("✓ {} ✗ {}", successful_count, failed_count));
        }
    }
    
    // Wait for all workers to complete
    for handle in handles {
        let _ = handle.await;
    }
    
    if let Some(pb) = &progress_bar {
        pb.finish_with_message("Processing complete");
    }
    
    // Calculate and display statistics
    let total_time = start_time.elapsed();
    let statistics = BatchStatistics {
        total_items: results.len(),
        successful_items: successful_count,
        failed_items: failed_count,
        total_time,
        avg_time_per_item: if results.len() > 0 {
            total_time / results.len() as u32
        } else {
            Duration::from_secs(0)
        },
        total_audio_duration,
        throughput: if total_time.as_secs_f32() > 0.0 {
            successful_count as f32 / total_time.as_secs_f32()
        } else {
            0.0
        },
    };
    
    display_statistics(&statistics, global);
    
    // Handle failed items
    if failed_count > 0 && !global.quiet {
        println!("\nFailed items:");
        for result in &results {
            if !result.success {
                println!("  - {}: {}", result.input.id, result.error.as_deref().unwrap_or("Unknown error"));
            }
        }
    }
    
    Ok(())
}

/// Process a single input item
async fn process_single_input(
    input: BatchInput,
    index: usize,
    pipeline: &VoirsPipeline,
    batch_config: &BatchConfig,
) -> ProcessingResult {
    let start_time = Instant::now();
    
    // Create synthesis config with overrides from input
    let synth_config = SynthesisConfig {
        speaking_rate: input.rate.unwrap_or(batch_config.speaking_rate),
        pitch_shift: input.pitch.unwrap_or(batch_config.pitch),
        volume_gain: input.volume.unwrap_or(batch_config.volume),
        quality: batch_config.quality,
        ..Default::default()
    };
    
    // Attempt synthesis
    match pipeline.synthesize_with_config(&input.text, &synth_config).await {
        Ok(audio) => {
            // Generate output filename
            let format = AudioFormat::Wav; // TODO: Make configurable
            let filename = super::files::generate_output_filename(&input, index, format);
            let output_path = batch_config.output_dir.join(filename);
            
            // Save audio file
            match audio.save(&output_path, format) {
                Ok(_) => {
                    ProcessingResult {
                        input,
                        success: true,
                        error: None,
                        output_path: Some(output_path),
                        duration: start_time.elapsed(),
                        audio_duration: Some(audio.duration()),
                    }
                }
                Err(e) => {
                    ProcessingResult {
                        input,
                        success: false,
                        error: Some(format!("Failed to save audio: {}", e)),
                        output_path: None,
                        duration: start_time.elapsed(),
                        audio_duration: None,
                    }
                }
            }
        }
        Err(e) => {
            ProcessingResult {
                input,
                success: false,
                error: Some(format!("Synthesis failed: {}", e)),
                output_path: None,
                duration: start_time.elapsed(),
                audio_duration: None,
            }
        }
    }
}

/// Display batch processing statistics
fn display_statistics(stats: &BatchStatistics, global: &GlobalOptions) {
    if global.quiet {
        return;
    }
    
    println!("\nBatch Processing Statistics:");
    println!("============================");
    println!("Total items: {}", stats.total_items);
    println!("Successful: {} ({:.1}%)", 
             stats.successful_items, 
             (stats.successful_items as f32 / stats.total_items as f32) * 100.0);
    println!("Failed: {} ({:.1}%)", 
             stats.failed_items,
             (stats.failed_items as f32 / stats.total_items as f32) * 100.0);
    println!("Total time: {:.2}s", stats.total_time.as_secs_f32());
    println!("Average time per item: {:.2}s", stats.avg_time_per_item.as_secs_f32());
    println!("Total audio generated: {:.2}s", stats.total_audio_duration);
    println!("Throughput: {:.2} items/second", stats.throughput);
    
    if stats.total_audio_duration > 0.0 {
        let real_time_factor = stats.total_time.as_secs_f32() / stats.total_audio_duration;
        println!("Real-time factor: {:.2}x", real_time_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_batch_statistics_calculation() {
        let stats = BatchStatistics {
            total_items: 100,
            successful_items: 95,
            failed_items: 5,
            total_time: Duration::from_secs(60),
            avg_time_per_item: Duration::from_millis(600),
            total_audio_duration: 120.0,
            throughput: 1.58,
        };
        
        assert_eq!(stats.total_items, 100);
        assert_eq!(stats.successful_items, 95);
        assert_eq!(stats.failed_items, 5);
        assert_eq!(stats.throughput, 1.58);
    }
    
    #[test]
    fn test_processing_result_creation() {
        let input = BatchInput {
            id: "test".to_string(),
            text: "Test text".to_string(),
            filename: None,
            voice: None,
            rate: None,
            pitch: None,
            volume: None,
            metadata: HashMap::new(),
        };
        
        let result = ProcessingResult {
            input: input.clone(),
            success: true,
            error: None,
            output_path: Some(std::path::PathBuf::from("/tmp/output.wav")),
            duration: Duration::from_millis(500),
            audio_duration: Some(2.5),
        };
        
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(result.output_path.is_some());
        assert_eq!(result.input.id, "test");
    }
}