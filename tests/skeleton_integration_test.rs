//! Skeleton integration test for VoiRS MVP pipeline
//! Tests the basic flow: text -> phonemes -> mel -> audio -> WAV file

use voirs_g2p::{DummyG2p, G2p, LanguageCode};
use voirs_acoustic::{DummyAcousticModel, AcousticModel, Phoneme};
use voirs_acoustic::SynthesisConfig as AcousticSynthesisConfig;
use voirs_vocoder::{DummyVocoder, Vocoder, SynthesisConfig as VocoderSynthesisConfig};
use voirs_vocoder::audio::io::convenience;

#[tokio::test]
async fn test_skeleton_end_to_end_pipeline() {
    // Initialize logging
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    println!("🚀 Starting skeleton end-to-end pipeline test");

    // Test text
    let text = "Hello world";
    println!("📝 Input text: '{}'", text);

    // Step 1: G2P - Text to Phonemes
    let g2p = DummyG2p::new();
    let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await.unwrap();
    
    println!("🔤 G2P Result: {} phonemes", phonemes.len());
    for (i, phoneme) in phonemes.iter().enumerate() {
        println!("   {}: {}", i, phoneme.symbol);
    }

    // Step 2: Acoustic Model - Phonemes to Mel Spectrogram
    let acoustic_model = DummyAcousticModel::new();
    let acoustic_config = AcousticSynthesisConfig::default();
    let acoustic_mel = acoustic_model.synthesize(&phonemes, Some(&acoustic_config)).await.unwrap();
    
    // Convert acoustic mel to vocoder mel format
    let mel_spectrogram = voirs_vocoder::MelSpectrogram::new(
        acoustic_mel.data.clone(),
        acoustic_mel.sample_rate,
        acoustic_mel.hop_length,
    );
    
    println!("🎵 Acoustic Result: {}x{} mel spectrogram", mel_spectrogram.n_mels, mel_spectrogram.n_frames);
    println!("   Duration: {:.3}s", mel_spectrogram.duration());
    println!("   Sample rate: {} Hz", mel_spectrogram.sample_rate);

    // Step 3: Vocoder - Mel Spectrogram to Audio
    let vocoder = DummyVocoder::new();
    let vocoder_config = VocoderSynthesisConfig::default();
    let audio_buffer = vocoder.vocode(&mel_spectrogram, Some(&vocoder_config)).await.unwrap();
    
    println!("🔊 Vocoder Result: {:.3}s audio", audio_buffer.duration());
    println!("   Sample rate: {} Hz", audio_buffer.sample_rate());
    println!("   Channels: {}", audio_buffer.channels());
    println!("   Samples: {}", audio_buffer.samples().len());

    // Step 4: Validate end-to-end flow
    assert!(!phonemes.is_empty(), "Should generate phonemes from text");
    assert!(mel_spectrogram.n_frames > 0, "Should generate mel frames");
    assert!(mel_spectrogram.n_mels > 0, "Should generate mel channels");
    assert!(audio_buffer.duration() > 0.0, "Should generate audio with duration");
    assert!(!audio_buffer.samples().is_empty(), "Should generate audio samples");

    // Verify audio quality (basic checks)
    let samples = audio_buffer.samples();
    let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    
    println!("📊 Audio Stats:");
    println!("   Peak amplitude: {:.3}", peak);
    println!("   RMS level: {:.3}", rms);
    
    assert!(peak > 0.0, "Audio should have non-zero amplitude");
    assert!(peak <= 1.0, "Audio should not clip");
    assert!(rms > 0.0, "Audio should have energy");

    // Step 5: Save audio to WAV file
    let wav_path = "/tmp/voirs_skeleton_test.wav";
    convenience::write_wav(&audio_buffer, wav_path).unwrap();
    
    println!("💾 WAV Output: Saved to {}", wav_path);
    
    // Verify WAV file was created and is readable
    let loaded_audio = convenience::read_wav(wav_path).unwrap();
    assert_eq!(loaded_audio.sample_rate(), audio_buffer.sample_rate());
    assert_eq!(loaded_audio.samples().len(), audio_buffer.samples().len());

    println!("✅ Skeleton end-to-end pipeline test PASSED!");
    println!("   '{}' -> {} phonemes -> {}x{} mel -> {:.3}s audio -> WAV file", 
             text, phonemes.len(), mel_spectrogram.n_mels, 
             mel_spectrogram.n_frames, audio_buffer.duration());
}

#[tokio::test]
async fn test_skeleton_batch_processing() {
    println!("🚀 Starting skeleton batch processing test");

    let texts = vec!["Hello", "world", "test"];
    let g2p = DummyG2p::new();
    let acoustic_model = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();
    let acoustic_config = AcousticSynthesisConfig::default();
    let vocoder_config = VocoderSynthesisConfig::default();

    let mut total_audio_duration = 0.0;

    for (i, text) in texts.iter().enumerate() {
        println!("📝 Processing text {}: '{}'", i + 1, text);

        // Process each text through the pipeline
        let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await.unwrap();
        let acoustic_mel = acoustic_model.synthesize(&phonemes, Some(&acoustic_config)).await.unwrap();
        let mel = voirs_vocoder::MelSpectrogram::new(
            acoustic_mel.data.clone(),
            acoustic_mel.sample_rate,
            acoustic_mel.hop_length,
        );
        let audio = vocoder.vocode(&mel, Some(&vocoder_config)).await.unwrap();

        total_audio_duration += audio.duration();
        
        println!("   Result: {} phonemes -> {}x{} mel -> {:.3}s audio", 
                 phonemes.len(), mel.n_mels, mel.n_frames, audio.duration());
    }

    println!("✅ Batch processing test PASSED!");
    println!("   Total audio generated: {:.3}s", total_audio_duration);
}

#[tokio::test]
async fn test_skeleton_component_metadata() {
    println!("🚀 Starting skeleton component metadata test");

    // Test G2P metadata
    let g2p = DummyG2p::new();
    let g2p_metadata = g2p.metadata();
    println!("🔤 G2P Metadata:");
    println!("   Name: {}", g2p_metadata.name);
    println!("   Version: {}", g2p_metadata.version);
    println!("   Languages: {:?}", g2p_metadata.supported_languages);

    // Test Acoustic Model metadata
    let acoustic = DummyAcousticModel::new();
    let acoustic_metadata = acoustic.metadata();
    println!("🎵 Acoustic Metadata:");
    println!("   Name: {}", acoustic_metadata.name);
    println!("   Architecture: {}", acoustic_metadata.architecture);
    println!("   Sample Rate: {} Hz", acoustic_metadata.sample_rate);
    println!("   Mel Channels: {}", acoustic_metadata.mel_channels);

    // Test Vocoder metadata
    let vocoder = DummyVocoder::new();
    let vocoder_metadata = vocoder.metadata();
    println!("🔊 Vocoder Metadata:");
    println!("   Name: {}", vocoder_metadata.name);
    println!("   Architecture: {}", vocoder_metadata.architecture);
    println!("   Sample Rate: {} Hz", vocoder_metadata.sample_rate);
    println!("   Latency: {:.1}ms", vocoder_metadata.latency_ms);
    println!("   Quality Score: {:.1}/5", vocoder_metadata.quality_score);

    println!("✅ Component metadata test PASSED!");
}

#[tokio::test]
async fn test_skeleton_synthesis_config_effects() {
    println!("🚀 Starting skeleton synthesis config effects test");

    let g2p = DummyG2p::new();
    let acoustic = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();
    let text = "Test config";

    // Test different synthesis configurations
    let acoustic_configs = vec![
        ("Normal", AcousticSynthesisConfig { speed: 1.0, ..Default::default() }),
        ("Fast", AcousticSynthesisConfig { speed: 1.5, ..Default::default() }),
        ("Slow", AcousticSynthesisConfig { speed: 0.7, ..Default::default() }),
        ("High Energy", AcousticSynthesisConfig { energy: 1.3, ..Default::default() }),
    ];
    let vocoder_config = VocoderSynthesisConfig::default();

    for (name, acoustic_config) in acoustic_configs {
        println!("⚙️  Testing config: {}", name);
        
        let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await.unwrap();
        let acoustic_mel = acoustic.synthesize(&phonemes, Some(&acoustic_config)).await.unwrap();
        let mel = voirs_vocoder::MelSpectrogram::new(
            acoustic_mel.data.clone(),
            acoustic_mel.sample_rate,
            acoustic_mel.hop_length,
        );
        let audio = vocoder.vocode(&mel, Some(&vocoder_config)).await.unwrap();

        println!("   Result: {:.3}s audio (speed: {:.1}x, energy: {:.1}x)", 
                 audio.duration(), acoustic_config.speed, acoustic_config.energy);
    }

    println!("✅ Synthesis config effects test PASSED!");
}

#[tokio::test]
async fn test_skeleton_wav_file_output() {
    println!("🚀 Starting skeleton WAV file output test");

    let g2p = DummyG2p::new();
    let acoustic = DummyAcousticModel::new();
    let vocoder = DummyVocoder::new();
    let text = "WAV output test";

    // Generate audio
    let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await.unwrap();
    let acoustic_mel = acoustic.synthesize(&phonemes, Some(&AcousticSynthesisConfig::default())).await.unwrap();
    let mel = voirs_vocoder::MelSpectrogram::new(
        acoustic_mel.data.clone(),
        acoustic_mel.sample_rate,
        acoustic_mel.hop_length,
    );
    let audio = vocoder.vocode(&mel, Some(&VocoderSynthesisConfig::default())).await.unwrap();

    // Test different WAV output formats
    let outputs = vec![
        ("/tmp/voirs_test_16bit.wav", "16-bit WAV"),
        ("/tmp/voirs_test_24bit.wav", "24-bit WAV"),
        ("/tmp/voirs_test_float.wav", "32-bit float WAV"),
    ];

    for (path, description) in outputs {
        println!("💾 Testing {}", description);
        
        // Write file based on description
        match description {
            "16-bit WAV" => convenience::write_wav(&audio, path).unwrap(),
            "24-bit WAV" => convenience::write_wav_hq(&audio, path).unwrap(),
            "32-bit float WAV" => convenience::write_wav_float(&audio, path).unwrap(),
            _ => convenience::write_wav(&audio, path).unwrap(),
        }
        
        // Verify file exists and can be read
        let loaded_audio = convenience::read_wav(path).unwrap();
        
        println!("   ✓ Saved {:.3}s audio to {}", loaded_audio.duration(), path);
        assert_eq!(loaded_audio.sample_rate(), audio.sample_rate());
        assert!(loaded_audio.duration() > 0.0);
    }

    println!("✅ WAV file output test PASSED!");
}