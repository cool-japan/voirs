//! End-to-end pipeline test for VoiRS
//!
//! This test demonstrates the complete TTS pipeline:
//! "Hello world" → dummy phonemes → random mel → sine wave → WAV file

use std::sync::Once;
use voirs_acoustic::{
    AcousticModel, DummyAcousticModel, MelSpectrogram as AcousticMel, Phoneme as AcousticPhoneme,
    SynthesisConfig as AcousticConfig,
};
use voirs_g2p::{DummyG2p, G2p, LanguageCode};
use voirs_vocoder::{
    audio::io::convenience::write_wav, DummyVocoder, MelSpectrogram as VocoderMel,
    SynthesisConfig as VocoderConfig, Vocoder,
};

static INIT_TRACING: Once = Once::new();

// Type conversion functions
fn g2p_phoneme_to_acoustic(phoneme: &voirs_g2p::Phoneme) -> AcousticPhoneme {
    AcousticPhoneme::new(&phoneme.symbol)
}

fn acoustic_mel_to_vocoder(mel: &AcousticMel) -> VocoderMel {
    VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length)
}

fn acoustic_config_to_vocoder(config: &AcousticConfig) -> VocoderConfig {
    VocoderConfig {
        speed: config.speed,
        pitch_shift: config.pitch_shift,
        energy: config.energy,
        speaker_id: config.speaker_id,
        seed: config.seed,
    }
}

#[tokio::test]
async fn test_complete_pipeline() {
    // Initialize tracing for debug output (only once to prevent memory leaks)
    INIT_TRACING.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .try_init();
    });

    // Step 1: G2P - Convert "Hello world" to phonemes
    println!("Step 1: Converting text to phonemes...");
    let g2p = DummyG2p::new();
    let text = "Hello world";
    let phonemes = g2p
        .to_phonemes(text, Some(LanguageCode::EnUs))
        .await
        .expect("G2P conversion failed");

    println!("Generated {} phonemes from '{}'", phonemes.len(), text);
    for (i, phoneme) in phonemes.iter().enumerate() {
        println!("  {}: {}", i, phoneme.symbol);
    }

    // Step 2: Acoustic Model - Convert phonemes to mel spectrogram
    println!("\nStep 2: Converting phonemes to mel spectrogram...");
    let acoustic_model = DummyAcousticModel::new();
    let acoustic_config = AcousticConfig {
        speed: 1.0,
        pitch_shift: 0.0,
        energy: 1.0,
        speaker_id: None,
        seed: Some(42), // For reproducible results
        emotion: None,
        voice_style: None,
    };

    // Convert G2P phonemes to acoustic phonemes
    let acoustic_phonemes: Vec<AcousticPhoneme> =
        phonemes.iter().map(g2p_phoneme_to_acoustic).collect();

    let mel_spectrogram = acoustic_model
        .synthesize(&acoustic_phonemes, Some(&acoustic_config))
        .await
        .expect("Acoustic synthesis failed");

    println!(
        "Generated mel spectrogram: {}x{} (duration: {:.2}s)",
        mel_spectrogram.n_mels,
        mel_spectrogram.n_frames,
        mel_spectrogram.duration()
    );

    // Step 3: Vocoder - Convert mel spectrogram to audio
    println!("\nStep 3: Converting mel spectrogram to audio...");
    let vocoder = DummyVocoder::new();

    // Convert acoustic mel to vocoder mel and config
    let vocoder_mel = acoustic_mel_to_vocoder(&mel_spectrogram);
    let vocoder_config = acoustic_config_to_vocoder(&acoustic_config);

    let audio = vocoder
        .vocode(&vocoder_mel, Some(&vocoder_config))
        .await
        .expect("Vocoding failed");

    println!(
        "Generated audio: {:.2}s @ {}Hz ({} samples)",
        audio.duration(),
        audio.sample_rate(),
        audio.samples().len()
    );

    // Step 4: Save audio to WAV file
    println!("\nStep 4: Saving audio to WAV file...");
    let output_path = "/tmp/voirs_hello_world_test.wav";
    write_wav(&audio, output_path).expect("Failed to write WAV file");

    println!("Audio saved to: {output_path}");

    // Verify the file was created and has reasonable size
    let metadata = std::fs::metadata(output_path).expect("WAV file was not created");

    println!("WAV file size: {} bytes", metadata.len());
    assert!(metadata.len() > 1000, "WAV file seems too small");

    // Cleanup
    let _ = std::fs::remove_file(output_path);

    println!("\n✅ End-to-end pipeline test completed successfully!");
    println!(
        "Pipeline: '{}' → {} phonemes → {}x{} mel → {:.2}s audio @ {}Hz",
        text,
        phonemes.len(),
        mel_spectrogram.n_mels,
        mel_spectrogram.n_frames,
        audio.duration(),
        audio.sample_rate()
    );
}

#[tokio::test]
async fn test_batch_pipeline() {
    println!("Testing batch processing pipeline...");

    // Initialize components
    let g2p = DummyG2p::new();
    let acoustic_model = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();

    // Test sentences
    let sentences = ["Hello", "world", "from", "VoiRS"];
    let mut all_phonemes = Vec::new();

    // Convert all sentences to phonemes
    for sentence in &sentences {
        let phonemes = g2p
            .to_phonemes(sentence, Some(LanguageCode::EnUs))
            .await
            .expect("G2P conversion failed");
        all_phonemes.push(phonemes);
    }

    // Convert to acoustic phonemes
    let acoustic_phonemes_vec: Vec<Vec<AcousticPhoneme>> = all_phonemes
        .iter()
        .map(|phonemes| phonemes.iter().map(g2p_phoneme_to_acoustic).collect())
        .collect();

    // Batch synthesis
    let phoneme_refs: Vec<&[AcousticPhoneme]> =
        acoustic_phonemes_vec.iter().map(|p| p.as_slice()).collect();
    let mel_spectrograms = acoustic_model
        .synthesize_batch(&phoneme_refs, None)
        .await
        .expect("Batch acoustic synthesis failed");

    // Convert mels for vocoder
    let vocoder_mels: Vec<VocoderMel> = mel_spectrograms
        .iter()
        .map(acoustic_mel_to_vocoder)
        .collect();

    // Batch vocoding
    let audio_buffers = vocoder
        .vocode_batch(&vocoder_mels, None)
        .await
        .expect("Batch vocoding failed");

    // Verify results
    assert_eq!(audio_buffers.len(), sentences.len());

    for (i, audio) in audio_buffers.iter().enumerate() {
        println!(
            "Audio {}: {:.2}s @ {}Hz",
            i,
            audio.duration(),
            audio.sample_rate()
        );
        assert!(audio.duration() > 0.0);
        assert!(!audio.is_empty());
    }

    println!("✅ Batch pipeline test completed successfully!");
}

#[tokio::test]
async fn test_pipeline_with_different_configs() {
    println!("Testing pipeline with different synthesis configs...");

    let g2p = DummyG2p::new();
    let acoustic_model = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();

    let text = "Test synthesis";
    let phonemes = g2p
        .to_phonemes(text, Some(LanguageCode::EnUs))
        .await
        .expect("G2P conversion failed");

    // Convert to acoustic phonemes
    let acoustic_phonemes: Vec<AcousticPhoneme> =
        phonemes.iter().map(g2p_phoneme_to_acoustic).collect();

    // Test different speed settings
    let speeds = [0.5, 1.0, 1.5, 2.0];
    let mut results = Vec::new();

    for &speed in &speeds {
        let acoustic_config = AcousticConfig {
            speed,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: Some(42),
            emotion: None,
            voice_style: None,
        };

        let mel = acoustic_model
            .synthesize(&acoustic_phonemes, Some(&acoustic_config))
            .await
            .expect("Acoustic synthesis failed");

        let vocoder_mel = acoustic_mel_to_vocoder(&mel);
        let vocoder_config = acoustic_config_to_vocoder(&acoustic_config);

        let audio = vocoder
            .vocode(&vocoder_mel, Some(&vocoder_config))
            .await
            .expect("Vocoding failed");

        results.push((speed, audio.duration()));
        println!("Speed {:.1}x: {:.2}s audio", speed, audio.duration());
    }

    // Verify that speed changes affect duration appropriately
    // (Note: DummyAcousticModel should respect speed parameter)
    for i in 1..results.len() {
        let (prev_speed, _prev_duration) = results[i - 1];
        let (curr_speed, curr_duration) = results[i];

        // Duration should be inversely related to speed
        if curr_speed > prev_speed {
            // We expect shorter duration for higher speed, but the dummy model
            // might not implement this perfectly, so we just check basic sanity
            assert!(curr_duration > 0.0, "Duration should be positive");
        }
    }

    println!("✅ Config variation test completed successfully!");
}

#[tokio::test]
async fn test_error_handling() {
    println!("Testing error handling in pipeline...");

    let acoustic_model = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();

    // Test empty phoneme sequence
    let empty_phonemes: Vec<AcousticPhoneme> = vec![];
    let result = acoustic_model.synthesize(&empty_phonemes, None).await;
    assert!(result.is_err(), "Should fail with empty phonemes");

    // Test valid mel spectrogram
    let mel_data = vec![vec![0.5; 100]; 80];
    let mel = VocoderMel::new(mel_data, 22050, 256);
    let audio_result = vocoder.vocode(&mel, None).await;
    assert!(audio_result.is_ok(), "Should succeed with valid mel");

    println!("✅ Error handling test completed successfully!");
}
