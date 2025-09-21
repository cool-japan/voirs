//! Test command implementation.

use crate::GlobalOptions;
use voirs_sdk::{config::AppConfig, error::Result, VoirsPipeline};

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
        println!("Playing audio...");
        audio.play()?;
        println!("✓ Audio playback completed");
    } else {
        let temp_file = std::env::temp_dir().join("voirs_test.wav");
        audio.save_wav(&temp_file)?;
        println!("✓ Audio saved to: {}", temp_file.display());
    }

    Ok(())
}
