//! Simple demonstration of voirs-acoustic synthesis capabilities
//!
//! This example shows basic usage of VITS and FastSpeech2 models.

use tracing::info;
use voirs_acoustic::{fastspeech::FastSpeech2Model, AcousticModel, Phoneme, Result, VitsModel};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🎵 Starting VoiRS Acoustic Synthesis Simple Demo");

    // Test input phonemes (English: "Hello")
    let phonemes = vec![
        Phoneme::new("h"),
        Phoneme::new("ɛ"),
        Phoneme::new("l"),
        Phoneme::new("oʊ"),
    ];

    // Demo 1: Basic VITS synthesis
    info!("📢 Demo 1: Basic VITS Synthesis");
    demo_basic_vits_synthesis(&phonemes).await?;

    // Demo 2: FastSpeech2 synthesis
    info!("📢 Demo 2: FastSpeech2 Synthesis");
    demo_fastspeech2_synthesis(&phonemes).await?;

    // Demo 3: Batch synthesis
    info!("📢 Demo 3: Batch Synthesis");
    demo_batch_synthesis(&phonemes).await?;

    info!("✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic VITS synthesis
async fn demo_basic_vits_synthesis(phonemes: &[Phoneme]) -> Result<()> {
    let model = VitsModel::new()?;
    let result = model.synthesize(phonemes, None).await?;

    info!(
        "✅ VITS synthesis completed: {} mel frames, {} mel bins",
        result.n_frames, result.n_mels
    );
    Ok(())
}

/// Demo 2: FastSpeech2 synthesis
async fn demo_fastspeech2_synthesis(phonemes: &[Phoneme]) -> Result<()> {
    let model = FastSpeech2Model::new();
    let result = model.synthesize(phonemes, None).await?;

    info!(
        "✅ FastSpeech2 synthesis completed: {} mel frames, {} mel bins",
        result.n_frames, result.n_mels
    );
    Ok(())
}

/// Demo 3: Batch synthesis
async fn demo_batch_synthesis(phonemes: &[Phoneme]) -> Result<()> {
    let model = FastSpeech2Model::new();

    // Create batch of different phoneme sequences
    let seq1 = vec![Phoneme::new("g"), Phoneme::new("ʊ"), Phoneme::new("d")];
    let seq2 = vec![Phoneme::new("b"), Phoneme::new("aɪ")];
    let batch_inputs = vec![phonemes, &seq1, &seq2];

    let results = model.synthesize_batch(&batch_inputs, None).await?;

    info!(
        "✅ Batch synthesis completed: {} sequences processed",
        results.len()
    );
    for (i, result) in results.iter().enumerate() {
        info!("   Sequence {}: {} frames", i + 1, result.n_frames);
    }

    Ok(())
}

/// Helper function to demonstrate model metadata
fn _demonstrate_model_metadata(model: &dyn AcousticModel) {
    let metadata = model.metadata();
    info!("Model: {}", metadata.name);
    info!("Version: {}", metadata.version);
    info!("Architecture: {:?}", metadata.architecture);
    info!("Supported languages: {:?}", metadata.supported_languages);
}
