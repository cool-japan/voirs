use anyhow::Result;
use std::path::Path;
use tokio::fs;
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let pipeline = VoirsPipeline::builder()
        .with_g2p(G2pBackend::Phonetisaurus)
        .with_acoustic(AcousticBackend::Vits)
        .with_vocoder(VocoderBackend::HifiGan)
        .with_speaker_id("female_1")
        .build()
        .await?;

    let texts = vec![
        "This is the first sentence to synthesize.",
        "Here's another sentence with different content.",
        "And finally, a third sentence to complete our batch.",
    ];

    fs::create_dir_all("batch_output").await?;

    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {}: {}", i + 1, text);
        
        let audio = pipeline.synthesize(text).await?;
        let output_path = format!("batch_output/output_{:02}.wav", i + 1);
        
        audio.save_wav(&output_path)?;
        println!("Saved: {}", output_path);
    }

    println!("Batch synthesis complete!");
    Ok(())
}