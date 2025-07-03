use anyhow::Result;
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let pipeline = VoirsPipeline::builder()
        .with_g2p(G2pBackend::Phonetisaurus)
        .with_acoustic(AcousticBackend::Vits)
        .with_vocoder(VocoderBackend::HifiGan)
        .build()
        .await?;

    let text = "Hello, world! This is VoiRS speaking in pure Rust.";
    
    println!("Synthesizing: {}", text);
    
    let audio = pipeline.synthesize(text).await?;
    
    audio.save_wav("output.wav")?;
    
    println!("Synthesis complete! Output saved to output.wav");
    println!("Sample rate: {} Hz", audio.sample_rate());
    println!("Duration: {:.2} seconds", audio.duration());
    
    Ok(())
}