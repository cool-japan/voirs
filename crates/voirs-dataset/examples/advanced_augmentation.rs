//! Advanced Audio Augmentation Example
//!
//! This example demonstrates how to use the modern augmentation techniques:
//! - SpecAugment (time/frequency masking for spectrograms)
//! - Codec Simulation (telephone, mobile, VoIP quality degradation)
//! - MixUp (mixing audio samples for improved generalization)
//!
//! These techniques are essential for training robust speech synthesis models.

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use voirs_dataset::augmentation::codec::{
    BatchCodecSimulator, CodecConfig, CodecSimulator, CodecType,
};
use voirs_dataset::augmentation::mixup::{
    BatchMixUpAugmentor, MixUpAugmentor, MixUpConfig, MixingStrategy,
};
use voirs_dataset::augmentation::specaugment::{BatchSpecAugment, SpecAugment, SpecAugmentConfig};
use voirs_dataset::{AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo};

/// Create example audio samples
fn create_sample_audio(frequency: f32, duration: f32) -> AudioData {
    let sample_rate = 22050;
    let num_samples = (sample_rate as f32 * duration) as usize;

    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    AudioData::new(samples, sample_rate, 1)
}

/// Create a dataset sample
fn create_dataset_sample(id: &str, audio: AudioData) -> DatasetSample {
    DatasetSample {
        id: id.to_string(),
        audio,
        text: format!("Example text for {}", id),
        speaker: Some(SpeakerInfo {
            id: "speaker1".to_string(),
            name: Some("Example Speaker".to_string()),
            gender: Some("F".to_string()),
            age: Some(30),
            accent: Some("neutral".to_string()),
            metadata: HashMap::new(),
        }),
        language: LanguageCode::EnUs,
        quality: QualityMetrics {
            snr: Some(35.0),
            clipping: Some(0.0),
            dynamic_range: Some(65.0),
            spectral_quality: Some(0.95),
            overall_quality: Some(0.9),
        },
        phonemes: None,
        metadata: HashMap::new(),
    }
}

/// Example 1: SpecAugment for mel spectrogram augmentation
fn example_specaugment() {
    println!("\n=== Example 1: SpecAugment ===\n");

    // Create a mock mel spectrogram (80 mel bins × 500 time steps)
    let spectrogram = Array2::from_shape_fn((80, 500), |(f, t)| {
        // Simulate mel spectrogram values
        ((f as f32 * 0.05 + t as f32 * 0.02).sin().abs() * 10.0).exp()
    });

    println!("Original spectrogram shape: {:?}", spectrogram.shape());
    println!(
        "Original spectrogram mean: {:.4}",
        spectrogram.mean().unwrap()
    );

    // Apply light augmentation (good for small datasets)
    let augmenter_light = SpecAugment::new(SpecAugmentConfig::light());
    let augmented_light = augmenter_light.augment_spectrogram(&spectrogram);
    println!("\nLight augmentation applied:");
    println!("  - Time masks: 1, width up to 70 steps");
    println!("  - Freq masks: 1, width up to 15 bins");
    println!("  - Result mean: {:.4}", augmented_light.mean().unwrap());

    // Apply strong augmentation (good for large datasets)
    let augmenter_strong = SpecAugment::new(SpecAugmentConfig::strong());
    let augmented_strong = augmenter_strong.augment_spectrogram(&spectrogram);
    println!("\nStrong augmentation applied:");
    println!("  - Time masks: 2, width up to 100 steps");
    println!("  - Freq masks: 2, width up to 27 bins");
    println!("  - Time warping: enabled");
    println!("  - Result mean: {:.4}", augmented_strong.mean().unwrap());

    // Apply adaptive augmentation (adjusts based on difficulty)
    let augmenter_adaptive = SpecAugment::new(SpecAugmentConfig::default());
    let easy_sample = augmenter_adaptive.augment_adaptive(&spectrogram, 0.2);
    let hard_sample = augmenter_adaptive.augment_adaptive(&spectrogram, 0.8);
    println!("\nAdaptive augmentation:");
    println!(
        "  - Easy sample (difficulty=0.2) mean: {:.4}",
        easy_sample.mean().unwrap()
    );
    println!(
        "  - Hard sample (difficulty=0.8) mean: {:.4}",
        hard_sample.mean().unwrap()
    );

    // Batch processing
    let batch_augmenter = BatchSpecAugment::new(SpecAugmentConfig::default());
    let batch = vec![spectrogram.clone(), spectrogram.clone(), spectrogram];
    let augmented_batch = batch_augmenter.augment_batch(&batch);
    println!("\nBatch augmentation:");
    println!(
        "  - Processed {} spectrograms independently",
        augmented_batch.len()
    );
}

/// Example 2: Codec Simulation for robustness training
fn example_codec_simulation() {
    println!("\n=== Example 2: Codec Simulation ===\n");

    // Create high-quality audio
    let audio = create_sample_audio(440.0, 2.0);
    println!("Original audio:");
    println!("  - Sample rate: {} Hz", audio.sample_rate());
    println!("  - Duration: {:.2} seconds", audio.duration());
    println!("  - RMS: {:.4}", audio.rms().unwrap());

    // Simulate telephone quality
    println!("\n1. Telephone Quality Simulation:");
    let telephone_config = CodecConfig::telephone();
    let telephone_sim = CodecSimulator::new(telephone_config);
    let telephone_audio = telephone_sim.simulate(&audio).unwrap();
    println!("  - Codec: G.711 μ-law (8-bit, 8kHz)");
    println!("  - Bandwidth: 3.4 kHz");
    println!("  - Packet loss: 1%");
    println!("  - Jitter: 2%");
    println!("  - Result RMS: {:.4}", telephone_audio.rms().unwrap());

    // Simulate mobile phone quality
    println!("\n2. Mobile Phone Quality Simulation:");
    let mobile_config = CodecConfig::mobile();
    let mobile_sim = CodecSimulator::new(mobile_config);
    let mobile_audio = mobile_sim.simulate(&audio).unwrap();
    println!("  - Codec: AMR-NB (12.2 kbps)");
    println!("  - Bandwidth: 3.4 kHz");
    println!("  - Packet loss: 2%");
    println!("  - Jitter: 5%");
    println!("  - Result RMS: {:.4}", mobile_audio.rms().unwrap());

    // Simulate VoIP quality
    println!("\n3. VoIP Quality Simulation:");
    let voip_config = CodecConfig::voip();
    let voip_sim = CodecSimulator::new(voip_config);
    let voip_audio = voip_sim.simulate(&audio).unwrap();
    println!("  - Codec: Opus (32 kbps)");
    println!("  - Bandwidth: 6 kHz");
    println!("  - Packet loss: 1%");
    println!("  - Jitter: 3%");
    println!("  - Result RMS: {:.4}", voip_audio.rms().unwrap());

    // Custom codec configuration
    println!("\n4. Custom Codec Configuration:");
    let custom_config = CodecConfig {
        codec_type: CodecType::Mp3Low,
        packet_loss_rate: 0.05,
        jitter_rate: 0.03,
        add_quantization_noise: true,
        add_pre_emphasis: false,
        bit_error_rate: 0.01,
    };
    let custom_sim = CodecSimulator::new(custom_config);
    let custom_audio = custom_sim.simulate(&audio).unwrap();
    println!("  - Codec: MP3 128kbps");
    println!("  - Packet loss: 5%");
    println!("  - Bit error rate: 1%");
    println!("  - Result RMS: {:.4}", custom_audio.rms().unwrap());

    // Batch simulation with multiple codecs
    println!("\n5. Batch Codec Simulation:");
    let batch_sim = BatchCodecSimulator::common_codecs();
    let simulated_batch = batch_sim.simulate_all(&audio).unwrap();
    println!(
        "  - Simulated {} different codec conditions",
        simulated_batch.len()
    );
    for (i, sim_audio) in simulated_batch.iter().enumerate() {
        println!("    Codec {}: RMS = {:.4}", i + 1, sim_audio.rms().unwrap());
    }
}

/// Example 3: MixUp for data augmentation
fn example_mixup() {
    println!("\n=== Example 3: MixUp Augmentation ===\n");

    // Create two different audio samples
    let audio1 = create_sample_audio(440.0, 1.0);
    let audio2 = create_sample_audio(880.0, 1.0);
    let sample1 = create_dataset_sample("sample1", audio1);
    let sample2 = create_dataset_sample("sample2", audio2);

    println!("Original samples:");
    println!(
        "  - Sample 1: {} Hz tone, RMS = {:.4}",
        440.0,
        sample1.audio.rms().unwrap()
    );
    println!(
        "  - Sample 2: {} Hz tone, RMS = {:.4}",
        880.0,
        sample2.audio.rms().unwrap()
    );

    // Standard MixUp
    println!("\n1. Standard MixUp:");
    let mixup_augmenter = MixUpAugmentor::new(MixUpConfig::default());
    let mixed_standard = mixup_augmenter.mix_samples(&sample1, &sample2).unwrap();
    println!("  - Strategy: Linear interpolation");
    println!("  - Mixed sample ID: {}", mixed_standard.id);
    println!("  - Mixed RMS: {:.4}", mixed_standard.audio.rms().unwrap());
    println!("  - Mixed text: {}", mixed_standard.text);

    // Balanced MixUp (more uniform mixing)
    println!("\n2. Balanced MixUp:");
    let balanced_augmenter = MixUpAugmentor::new(MixUpConfig::balanced());
    let mixed_balanced = balanced_augmenter.mix_samples(&sample1, &sample2).unwrap();
    println!("  - Alpha: 1.0 (uniform distribution)");
    println!("  - Lambda range: [0.3, 0.7]");
    println!("  - Mixed RMS: {:.4}", mixed_balanced.audio.rms().unwrap());

    // CutMix (segment replacement)
    println!("\n3. CutMix:");
    let cutmix_config = MixUpConfig {
        strategy: MixingStrategy::CutMix,
        ..Default::default()
    };
    let cutmix_augmenter = MixUpAugmentor::new(cutmix_config);
    let mixed_cutmix = cutmix_augmenter.mix_samples(&sample1, &sample2).unwrap();
    println!("  - Strategy: Replace random segments");
    println!("  - Mixed sample ID: {}", mixed_cutmix.id);
    println!("  - Mixed RMS: {:.4}", mixed_cutmix.audio.rms().unwrap());

    // TimeMix (time-varying mixing)
    println!("\n4. TimeMix:");
    let timemix_config = MixUpConfig {
        strategy: MixingStrategy::TimeMix,
        ..Default::default()
    };
    let timemix_augmenter = MixUpAugmentor::new(timemix_config);
    let mixed_timemix = timemix_augmenter.mix_samples(&sample1, &sample2).unwrap();
    println!("  - Strategy: Time-varying mixing ratio");
    println!("  - Mixed RMS: {:.4}", mixed_timemix.audio.rms().unwrap());

    // Adaptive MixUp (based on sample characteristics)
    println!("\n5. Adaptive MixUp:");
    let adaptive_config = MixUpConfig {
        strategy: MixingStrategy::AdaptiveMix,
        ..Default::default()
    };
    let adaptive_augmenter = MixUpAugmentor::new(adaptive_config);
    let mixed_adaptive = adaptive_augmenter.mix_samples(&sample1, &sample2).unwrap();
    println!("  - Strategy: Adaptive based on RMS levels");
    println!("  - Mixed RMS: {:.4}", mixed_adaptive.audio.rms().unwrap());

    // Batch MixUp
    println!("\n6. Batch MixUp:");
    let samples = vec![
        create_dataset_sample("s1", create_sample_audio(440.0, 0.5)),
        create_dataset_sample("s2", create_sample_audio(880.0, 0.5)),
        create_dataset_sample("s3", create_sample_audio(1320.0, 0.5)),
        create_dataset_sample("s4", create_sample_audio(1760.0, 0.5)),
    ];
    let batch_augmenter = BatchMixUpAugmentor::new(MixUpConfig::default());
    let mixed_batch = batch_augmenter.mix_batch(&samples).unwrap();
    println!("  - Processed {} samples", mixed_batch.len());
    println!("  - Each sample mixed with a random other sample");
}

/// Example 4: Combined augmentation pipeline
fn example_combined_pipeline() {
    println!("\n=== Example 4: Combined Augmentation Pipeline ===\n");

    // Create original sample
    let audio = create_sample_audio(440.0, 2.0);
    let sample = create_dataset_sample("original", audio);

    println!("Starting with high-quality audio:");
    println!("  - RMS: {:.4}", sample.audio.rms().unwrap());
    println!("  - Duration: {:.2}s", sample.audio.duration());

    // Pipeline Step 1: Codec simulation for robustness
    println!("\nStep 1: Apply codec simulation (mobile quality)");
    let codec_sim = CodecSimulator::new(CodecConfig::mobile());
    let degraded_audio = codec_sim.simulate(&sample.audio).unwrap();
    let mut augmented_sample = sample.clone();
    augmented_sample.audio = degraded_audio;
    println!("  - Simulated mobile phone codec");
    println!(
        "  - RMS after codec: {:.4}",
        augmented_sample.audio.rms().unwrap()
    );

    // Pipeline Step 2: MixUp for generalization
    println!("\nStep 2: Apply MixUp with another sample");
    let other_audio = create_sample_audio(880.0, 2.0);
    let other_sample = create_dataset_sample("other", other_audio);
    let mixup = MixUpAugmentor::new(MixUpConfig::balanced());
    let mixed_sample = mixup.mix_samples(&augmented_sample, &other_sample).unwrap();
    println!("  - Mixed with another sample");
    println!(
        "  - RMS after mixing: {:.4}",
        mixed_sample.audio.rms().unwrap()
    );

    // Summary
    println!("\nPipeline summary:");
    println!("  - Original RMS: {:.4}", sample.audio.rms().unwrap());
    println!("  - Final RMS: {:.4}", mixed_sample.audio.rms().unwrap());
    println!("  - Augmentation factor: 1 → multiple variants possible");
    println!("  - Ready for model training!");
}

/// Example 5: Practical training dataset augmentation
fn example_training_augmentation() {
    println!("\n=== Example 5: Training Dataset Augmentation ===\n");

    // Simulate a small training dataset
    let training_samples = vec![
        create_dataset_sample("train_001", create_sample_audio(440.0, 1.0)),
        create_dataset_sample("train_002", create_sample_audio(494.0, 1.0)),
        create_dataset_sample("train_003", create_sample_audio(523.0, 1.0)),
    ];

    println!(
        "Original training dataset: {} samples",
        training_samples.len()
    );

    // Apply augmentation to expand dataset
    let mut augmented_dataset = training_samples.clone();

    // Add codec-augmented versions
    println!("\nAdding codec-augmented versions...");
    let codec_configs = vec![
        CodecConfig::telephone(),
        CodecConfig::mobile(),
        CodecConfig::voip(),
    ];

    for sample in &training_samples {
        for (i, config) in codec_configs.iter().enumerate() {
            let simulator = CodecSimulator::new(config.clone());
            let augmented_audio = simulator.simulate(&sample.audio).unwrap();
            let mut aug_sample = sample.clone();
            aug_sample.id = format!("{}_codec_{}", sample.id, i);
            aug_sample.audio = augmented_audio;
            augmented_dataset.push(aug_sample);
        }
    }

    println!(
        "  - After codec augmentation: {} samples",
        augmented_dataset.len()
    );

    // Add MixUp versions
    println!("\nAdding MixUp versions...");
    let mixup = MixUpAugmentor::new(MixUpConfig::default());
    let mut mixup_samples = Vec::new();

    for i in 0..training_samples.len() {
        for j in (i + 1)..training_samples.len() {
            let mixed = mixup
                .mix_samples(&training_samples[i], &training_samples[j])
                .unwrap();
            mixup_samples.push(mixed);
        }
    }

    augmented_dataset.extend(mixup_samples);
    println!("  - After MixUp: {} samples", augmented_dataset.len());

    // Final statistics
    println!("\nFinal augmented dataset:");
    println!("  - Original samples: {}", training_samples.len());
    println!("  - Augmented samples: {}", augmented_dataset.len());
    println!(
        "  - Augmentation factor: {:.1}x",
        augmented_dataset.len() as f32 / training_samples.len() as f32
    );
    println!("\nAugmentation breakdown:");
    println!("  - Original: {} samples", training_samples.len());
    println!("  - Codec variants: {} samples", training_samples.len() * 3);
    println!(
        "  - MixUp combinations: {} samples",
        training_samples.len() * (training_samples.len() - 1) / 2
    );
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       VoiRS Advanced Audio Augmentation Examples          ║");
    println!("║                                                            ║");
    println!("║  Modern techniques for robust speech synthesis models     ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Run all examples
    example_specaugment();
    example_codec_simulation();
    example_mixup();
    example_combined_pipeline();
    example_training_augmentation();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                     Examples Complete                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Key Takeaways:");
    println!("  1. SpecAugment: Best for spectrogram-based models (ASR/TTS)");
    println!("  2. Codec Simulation: Essential for real-world robustness");
    println!("  3. MixUp: Improves generalization and reduces overfitting");
    println!("  4. Combined Pipeline: Use multiple techniques together");
    println!("  5. Training Augmentation: Can expand datasets by 5-10x");
    println!("\nFor production use, tune augmentation strength based on:");
    println!("  - Dataset size (more augmentation for smaller datasets)");
    println!("  - Model capacity (stronger augmentation for larger models)");
    println!("  - Target application (adjust codec simulation to match)");
}
