//! Batch Synthesis Example - VoiRS Text-to-Speech
//!
//! Demonstrates synthesizing multiple texts in sequence using the VoiRS pipeline.
//! Outputs are written to a temporary batch directory.

use anyhow::Result;
use tokio::fs;
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let pipeline = VoirsPipelineBuilder::new()
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build pipeline: {e}"))?;

    let texts = [
        "This is the first sentence to synthesize.",
        "Here's another sentence with different content.",
        "And finally, a third sentence to complete our batch.",
    ];

    fs::create_dir_all("batch_output").await?;

    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {}: {}", i + 1, text);

        let audio = pipeline
            .synthesize(text)
            .await
            .map_err(|e| anyhow::anyhow!("Synthesis failed for text {}: {e}", i + 1))?;

        let output_path = std::path::PathBuf::from(format!("batch_output/output_{:02}.wav", i + 1));

        audio
            .save_wav(&output_path)
            .map_err(|e| anyhow::anyhow!("Failed to save {}: {e}", output_path.display()))?;

        println!("Saved: {}", output_path.display());
    }

    println!("Batch synthesis complete!");
    Ok(())
}
