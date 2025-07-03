use anyhow::Result;
use voirs::prelude::*;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let pipeline = VoirsPipeline::builder()
        .with_g2p(G2pBackend::Phonetisaurus)
        .with_acoustic(AcousticBackend::Vits)
        .with_vocoder(VocoderBackend::HifiGan)
        .with_streaming(true)
        .build()
        .await?;

    let text = "This is a demonstration of streaming speech synthesis. \
                Each chunk of audio will be generated as soon as possible, \
                allowing for real-time playback with minimal latency.";

    println!("Starting streaming synthesis...");

    let mut stream = pipeline.synthesize_stream(text).await?;
    let mut chunk_count = 0;

    while let Some(audio_chunk) = stream.next().await {
        let chunk = audio_chunk?;
        chunk_count += 1;
        
        println!("Received chunk {}: {:.3}s of audio", 
                chunk_count, chunk.duration());
        
        let output_path = format!("stream_chunk_{:02}.wav", chunk_count);
        chunk.save_wav(&output_path)?;
        
        sleep(Duration::from_millis(100)).await;
    }

    println!("Streaming synthesis complete! Generated {} chunks", chunk_count);
    Ok(())
}