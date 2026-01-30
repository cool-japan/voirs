//! # State-of-the-Art Research Models Demo
//!
//! This example demonstrates the latest research models integrated into VoiRS,
//! including Diffusion Transformers, Neural Codecs, Flow-Matching, Score-Based Models,
//! and Consistency Models.

use voirs_singing::prelude::*;
use voirs_singing::{
    ConditioningType, DistillationSchedule, IntegrationMethod, NoiseSchedule, SamplingMethod,
};
use voirs_singing::{ConsistencyModelConfig, ScoreBasedConfig};
use voirs_singing::{DiffusionTransformerConfig, FlowMatchingConfig, NeuralCodecConfig};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS Research Models Integration Demo ===\n");

    // Prepare conditioning signal
    let conditioning = vec![0.5; 1000];
    let length = 2000;

    // 1. Diffusion Transformer Demo
    println!("1. Diffusion Transformer Model:");
    let dit_config = DiffusionTransformerConfig {
        model_dim: 512,
        num_layers: 6,
        num_heads: 8,
        ff_dim: 2048,
        timesteps: 100,
        noise_schedule: NoiseSchedule::Cosine,
        conditioning: ConditioningType::CrossAttention,
    };
    let dit_model = DiffusionTransformer::new(dit_config);
    let dit_info = dit_model.get_info();

    println!("   Model Configuration:");
    println!("     Dimension: {}", dit_info.model_dim);
    println!("     Layers: {}", dit_info.num_layers);
    println!("     Attention Heads: {}", dit_info.num_heads);
    println!("     Timesteps: {}", dit_info.timesteps);
    println!("     Noise Schedule: {:?}", dit_info.noise_schedule);
    println!("     Parameters: {}M", dit_info.param_count / 1_000_000);

    println!("   Generating audio with {} denoising steps...", 20);
    let dit_audio = dit_model.generate(&conditioning, 20).await?;
    println!("   ✓ Generated {} audio samples\n", dit_audio.len());

    // 2. Neural Codec Language Model Demo
    println!("2. Neural Codec Language Model:");
    let codec_config = NeuralCodecConfig {
        codebook_size: 1024,
        num_codebooks: 8,
        embedding_dim: 128,
        downsample_factor: 320,
        bandwidth: 6.0,
    };
    let codec_model = NeuralCodecLanguageModel::new(codec_config);
    let codec_stats = codec_model.get_stats();

    println!("   Codec Configuration:");
    println!("     Codebook Size: {}", codec_stats.codebook_size);
    println!("     Number of Codebooks: {}", codec_stats.num_codebooks);
    println!(
        "     Compression Ratio: {:.1}x",
        codec_stats.compression_ratio
    );
    println!("     Bitrate: {:.1} kbps", codec_stats.bitrate);
    println!("     Bandwidth: {:.1} kHz", codec_stats.bandwidth);

    let audio = vec![0.5; 16000];
    println!("   Encoding audio to discrete tokens...");
    let tokens = codec_model.encode(&audio).await?;
    println!(
        "   ✓ Encoded {} frames with {} codebooks",
        tokens.tokens.len(),
        tokens.num_codebooks
    );

    println!("   Decoding tokens back to audio...");
    let reconstructed = codec_model.decode(&tokens).await?;
    println!("   ✓ Reconstructed {} audio samples\n", reconstructed.len());

    // 3. Flow-Matching Synthesis Demo
    println!("3. Flow-Matching Synthesis:");
    let flow_methods = [
        ("Euler", IntegrationMethod::Euler),
        ("Heun", IntegrationMethod::Heun),
        ("Runge-Kutta 4", IntegrationMethod::RungeKutta4),
    ];

    for (name, method) in &flow_methods {
        let flow_config = FlowMatchingConfig {
            model_dim: 512,
            flow_steps: 50,
            integration_method: *method,
            conditional: true,
        };
        let flow_synth = FlowMatchingSynthesizer::new(flow_config);

        println!("   {} Method:", name);
        let start = std::time::Instant::now();
        let flow_audio = flow_synth.generate(&conditioning, length).await?;
        let elapsed = start.elapsed();
        println!(
            "     Generated {} samples in {:?}",
            flow_audio.len(),
            elapsed
        );
    }
    println!();

    // 4. Score-Based Generative Model Demo
    println!("4. Score-Based Generative Model:");
    let score_config = ScoreBasedConfig {
        model_dim: 512,
        num_scales: 100,
        sigma_min: 0.01,
        sigma_max: 50.0,
        sampling_method: SamplingMethod::AnnealedLangevin,
    };

    println!("   Model Configuration:");
    println!("     Noise Scales: {}", score_config.num_scales);
    println!("     σ_min: {}", score_config.sigma_min);
    println!("     σ_max: {}", score_config.sigma_max);
    println!("     Sampling: {:?}", score_config.sampling_method);

    let score_model = ScoreBasedModel::new(score_config);

    println!("   Generating with annealed Langevin dynamics...");
    let start = std::time::Instant::now();
    let score_audio = score_model.generate(&conditioning, length).await?;
    let elapsed = start.elapsed();
    println!(
        "   ✓ Generated {} samples in {:?}\n",
        score_audio.len(),
        elapsed
    );

    // 5. Consistency Model Demo
    println!("5. Consistency Model:");
    let consistency_config = ConsistencyModelConfig {
        model_dim: 512,
        training_steps: 1000,
        epsilon: 0.002,
        distillation_schedule: DistillationSchedule::Cosine,
    };
    let consistency_model = ConsistencyModel::new(consistency_config);

    println!("   Single-Step Generation:");
    let start = std::time::Instant::now();
    let single_step = consistency_model.generate(&conditioning, length).await?;
    let single_elapsed = start.elapsed();
    println!(
        "     Generated {} samples in {:?}",
        single_step.len(),
        single_elapsed
    );

    println!("   Multi-Step Generation (5 steps):");
    let start = std::time::Instant::now();
    let multi_step = consistency_model
        .generate_multistep(&conditioning, length, 5)
        .await?;
    let multi_elapsed = start.elapsed();
    println!(
        "     Generated {} samples in {:?}",
        multi_step.len(),
        multi_elapsed
    );
    println!(
        "     Speedup over single-step: {:.2}x\n",
        multi_elapsed.as_secs_f32() / single_elapsed.as_secs_f32()
    );

    // 6. Comparison Summary
    println!("6. Model Comparison Summary:");
    println!("   ┌─────────────────────┬───────────────┬──────────────────┐");
    println!("   │ Model               │ Generation    │ Key Feature      │");
    println!("   ├─────────────────────┼───────────────┼──────────────────┤");
    println!("   │ Diffusion Trans.    │ Multi-step    │ High Quality     │");
    println!("   │ Neural Codec LM     │ Discrete      │ Compression      │");
    println!("   │ Flow-Matching       │ ODE-based     │ Flexible         │");
    println!("   │ Score-Based         │ Iterative     │ Probabilistic    │");
    println!("   │ Consistency         │ Single-step   │ Fast Inference   │");
    println!("   └─────────────────────┴───────────────┴──────────────────┘");

    println!("\n=== Demo Complete ===");
    println!("VoiRS integrates cutting-edge research models for high-quality,");
    println!("efficient, and flexible singing synthesis.");

    Ok(())
}
