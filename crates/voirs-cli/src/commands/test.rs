//! Test command implementation.

use voirs::{config::AppConfig, error::Result, VoirsPipeline};
use crate::GlobalOptions;

/// Run test command
pub async fn run_test(
    text: &str,
    play: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    println!("Testing VoiRS synthesis pipeline...");
    
    let pipeline = VoirsPipeline::builder()
        .with_gpu_acceleration(config.pipeline.use_gpu || global.gpu)
        .build()
        .await?;
    
    let audio = pipeline.synthesize(text).await?;
    
    println!("✓ Synthesis successful");
    println!("  Duration: {:.2}s", audio.duration());
    println!("  Sample rate: {}Hz", audio.sample_rate());
    println!("  Channels: {}", audio.channels());
    
    if play {
        // TODO: Implement audio playback
        println!("Playing audio... (not yet implemented)");
        audio.play()?;
    } else {
        let temp_file = std::env::temp_dir().join("voirs_test.wav");
        audio.save_wav(&temp_file)?;
        println!("✓ Audio saved to: {}", temp_file.display());
    }
    
    Ok(())
}