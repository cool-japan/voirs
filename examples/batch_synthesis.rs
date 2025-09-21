use anyhow::Result;
use tokio::fs;
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await?;

    let texts = [
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
        println!("Saved: {output_path}");
    }

    println!("Batch synthesis complete!");
    Ok(())
}
