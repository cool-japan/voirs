use anyhow::Result;
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let pipeline = VoirsPipeline::builder()
        .with_g2p(G2pBackend::Phonetisaurus)
        .with_acoustic(AcousticBackend::Vits)
        .with_vocoder(VocoderBackend::HifiGan)
        .with_ssml_support(true)
        .build()
        .await?;

    let ssml_text = r#"
        <speak>
            <p>Welcome to <emphasis level="strong">VoiRS</emphasis>, 
            the pure Rust speech synthesis framework.</p>
            
            <break time="1s"/>
            
            <p>This example demonstrates <prosody rate="slow">slow speech</prosody>, 
            <prosody rate="fast">fast speech</prosody>, and 
            <prosody pitch="high">high pitch</prosody>.</p>
            
            <p>You can also specify different voices:
            <voice name="female_calm">This is a calm female voice.</voice>
            <voice name="male_energetic">And this is an energetic male voice!</voice>
            </p>
        </speak>
    "#;

    println!("Synthesizing SSML content...");
    
    let audio = pipeline.synthesize_ssml(ssml_text).await?;
    
    audio.save_wav("ssml_output.wav")?;
    
    println!("SSML synthesis complete!");
    println!("Output saved to ssml_output.wav");
    println!("Duration: {:.2} seconds", audio.duration());
    
    Ok(())
}