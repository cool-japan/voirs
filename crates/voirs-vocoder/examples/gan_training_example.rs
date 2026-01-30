//! Example demonstrating GAN loss functions for neural vocoder training.
//!
//! This example shows how to use the adversarial and feature matching losses
//! for training GAN-based vocoders like HiFi-GAN, BigVGAN, and UnivNet.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example gan_training_example --features candle
//! ```

use scirs2_core::ndarray::prelude::*;
use voirs_vocoder::loss::gan::{
    AdversarialLoss, AdversarialLossConfig, AdversarialLossType, CombinedDiscriminatorLoss,
    CombinedDiscriminatorLossConfig, FeatureMatchingLoss, FeatureMatchingLossConfig,
    MultiPeriodDiscriminatorLoss, MultiPeriodDiscriminatorLossConfig, MultiScaleDiscriminatorLoss,
    MultiScaleDiscriminatorLossConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GAN Loss Functions for Neural Vocoder Training ===\n");

    // Example 1: Basic Adversarial Loss
    println!("1. Basic Adversarial Loss (Least Squares)");
    println!("   Used in HiFi-GAN for stable training\n");
    demonstrate_basic_adversarial_loss()?;

    // Example 2: Hinge Loss
    println!("\n2. Hinge Loss");
    println!("   Used in BigVGAN for improved gradient flow\n");
    demonstrate_hinge_loss()?;

    // Example 3: Feature Matching Loss
    println!("\n3. Feature Matching Loss");
    println!("   Encourages generator to match intermediate features\n");
    demonstrate_feature_matching_loss()?;

    // Example 4: Multi-Scale Discriminator Loss
    println!("\n4. Multi-Scale Discriminator Loss (HiFi-GAN)");
    println!("   Combines adversarial and feature matching across multiple scales\n");
    demonstrate_multi_scale_loss()?;

    // Example 5: Multi-Period Discriminator Loss (NEW!)
    println!("\n5. Multi-Period Discriminator Loss (UnivNet)");
    println!("   Captures periodic patterns at different periods\n");
    demonstrate_multi_period_loss()?;

    // Example 6: Combined Multi-Scale + Multi-Period Loss (NEW!)
    println!("\n6. Combined Multi-Scale + Multi-Period Loss");
    println!("   State-of-the-art: Uses both scale and period discriminators\n");
    demonstrate_combined_loss()?;

    // Example 7: Complete Training Loop Simulation
    println!("\n7. Simulated Training Loop");
    println!("   Shows typical usage in a training iteration\n");
    simulate_training_loop()?;

    println!("\n=== All Examples Completed Successfully ===");
    Ok(())
}

/// Demonstrate basic adversarial loss with least squares
fn demonstrate_basic_adversarial_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::LeastSquares,
        weight: 1.0,
    };
    let adv_loss = AdversarialLoss::new(config);

    // Simulate discriminator outputs for generated audio
    // Values close to 1.0 mean discriminator thinks it's real
    let disc_fake_outputs = vec![
        Array1::from_vec(vec![0.8, 0.75, 0.82, 0.79]), // Good generator
    ];

    let gen_loss = adv_loss.generator_loss(&disc_fake_outputs)?;
    println!("   Generator Loss: {:.6}", gen_loss);
    println!("   (Lower is better - generator fooling discriminator)");

    // Simulate discriminator training
    let disc_real_outputs = vec![
        Array1::from_vec(vec![0.9, 0.95, 0.88, 0.92]), // Real audio should be close to 1.0
    ];
    let disc_fake_outputs = vec![
        Array1::from_vec(vec![0.2, 0.15, 0.25, 0.18]), // Fake audio should be close to 0.0
    ];

    let disc_loss = adv_loss.discriminator_loss(&disc_real_outputs, &disc_fake_outputs)?;
    println!("   Discriminator Loss: {:.6}", disc_loss);
    println!("   (Lower is better - discriminator correctly classifying)");

    Ok(())
}

/// Demonstrate hinge loss (used in BigVGAN)
fn demonstrate_hinge_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = AdversarialLossConfig {
        loss_type: AdversarialLossType::Hinge,
        weight: 1.0,
    };
    let adv_loss = AdversarialLoss::new(config);

    let disc_fake_outputs = vec![Array1::from_vec(vec![0.6, 0.7, 0.65])];
    let gen_loss = adv_loss.generator_loss(&disc_fake_outputs)?;
    println!("   Generator Loss (Hinge): {:.6}", gen_loss);
    println!("   (Hinge loss is typically negative for generator)");

    let disc_real_outputs = vec![Array1::from_vec(vec![0.9, 0.95, 0.85])];
    let disc_fake_outputs = vec![Array1::from_vec(vec![0.2, 0.15, 0.25])];

    let disc_loss = adv_loss.discriminator_loss(&disc_real_outputs, &disc_fake_outputs)?;
    println!("   Discriminator Loss (Hinge): {:.6}", disc_loss);

    Ok(())
}

/// Demonstrate feature matching loss
fn demonstrate_feature_matching_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = FeatureMatchingLossConfig {
        weight: 10.0, // Typically weighted higher
        num_layers: 3,
    };
    let fm_loss = FeatureMatchingLoss::new(config);

    // Simulate intermediate discriminator features
    // In practice, these come from discriminator layer outputs
    let real_features = vec![
        // Layer 1: 4 timesteps × 8 features
        Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32 * 0.1).collect())?,
        // Layer 2: 4 timesteps × 16 features
        Array2::<f32>::from_shape_vec((4, 16), (0..64).map(|x| x as f32 * 0.05).collect())?,
        // Layer 3: 4 timesteps × 32 features
        Array2::<f32>::from_shape_vec((4, 32), (0..128).map(|x| x as f32 * 0.02).collect())?,
    ];

    // Generated features are slightly different
    let fake_features = vec![
        Array2::<f32>::from_shape_vec((4, 8), (0..32).map(|x| x as f32 * 0.11).collect())?,
        Array2::<f32>::from_shape_vec((4, 16), (0..64).map(|x| x as f32 * 0.048).collect())?,
        Array2::<f32>::from_shape_vec((4, 32), (0..128).map(|x| x as f32 * 0.021).collect())?,
    ];

    let loss = fm_loss.compute(&real_features, &fake_features)?;
    println!("   Feature Matching Loss: {:.6}", loss);
    println!("   (Encourages generator to produce similar feature representations)");

    // Demonstrate weighted feature matching
    let layer_weights = vec![0.5, 1.0, 1.5]; // Weight later layers more
    let weighted_loss = fm_loss.compute_weighted(&real_features, &fake_features, &layer_weights)?;
    println!("   Weighted Feature Matching Loss: {:.6}", weighted_loss);
    println!("   (With higher weights on deeper layers)");

    Ok(())
}

/// Demonstrate multi-scale discriminator loss
fn demonstrate_multi_scale_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = MultiScaleDiscriminatorLossConfig {
        num_scales: 3, // Typically 3 scales: original, 2x downsampled, 4x downsampled
        adversarial_config: AdversarialLossConfig {
            loss_type: AdversarialLossType::LeastSquares,
            weight: 1.0,
        },
        feature_matching_config: Some(FeatureMatchingLossConfig {
            weight: 10.0,
            num_layers: 4,
        }),
    };

    let msd_loss = MultiScaleDiscriminatorLoss::new(config);

    // Simulate discriminator outputs at 3 scales
    let disc_fake_outputs = vec![
        // Scale 1 (original resolution)
        vec![Array1::from_vec(vec![0.8, 0.75, 0.82])],
        // Scale 2 (2x downsampled)
        vec![Array1::from_vec(vec![0.77, 0.80])],
        // Scale 3 (4x downsampled)
        vec![Array1::from_vec(vec![0.85])],
    ];

    // Without feature matching
    let breakdown = msd_loss.generator_loss(&disc_fake_outputs, None, None)?;
    println!("   Multi-Scale Generator Loss:");
    println!("{}", breakdown.summary());

    // With feature matching
    let real_features = vec![
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9],
        )?],
    ];

    let fake_features = vec![
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05],
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15],
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 4),
            vec![0.95, 1.95, 2.95, 3.95, 4.95, 5.95, 6.95, 7.95],
        )?],
    ];

    let breakdown_with_fm = msd_loss.generator_loss(
        &disc_fake_outputs,
        Some(&real_features),
        Some(&fake_features),
    )?;
    println!("\n   With Feature Matching:");
    println!("{}", breakdown_with_fm.summary());

    Ok(())
}

/// Demonstrate multi-period discriminator loss (UnivNet)
fn demonstrate_multi_period_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = MultiPeriodDiscriminatorLossConfig {
        periods: vec![2, 3, 5, 7, 11], // Standard UnivNet periods
        ..Default::default()
    };
    let mpd_loss = MultiPeriodDiscriminatorLoss::new(config);

    println!("   UnivNet uses discriminators at 5 different periods:");
    println!("   Periods: [2, 3, 5, 7, 11]");
    println!("   Each captures different periodic patterns in audio\n");

    // Simulate discriminator outputs for each period
    let disc_fake_outputs = vec![
        vec![Array1::from_vec(vec![0.75, 0.72, 0.78])], // Period 2
        vec![Array1::from_vec(vec![0.77, 0.73, 0.76])], // Period 3
        vec![Array1::from_vec(vec![0.80, 0.74, 0.79])], // Period 5
        vec![Array1::from_vec(vec![0.76, 0.78, 0.75])], // Period 7
        vec![Array1::from_vec(vec![0.79, 0.77, 0.81])], // Period 11
    ];

    let breakdown = mpd_loss.generator_loss(&disc_fake_outputs, None, None)?;
    println!("   Multi-Period Generator Loss:");
    println!("{}", breakdown.summary());

    // With feature matching
    println!("\n   With feature matching (recommended):");
    let real_features = vec![
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.1).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.11).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.09).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.12).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.10).collect(),
        )?],
    ];

    let fake_features = vec![
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.105).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.115).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.095).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.125).collect(),
        )?],
        vec![Array2::<f32>::from_shape_vec(
            (2, 8),
            (0..16).map(|x| x as f32 * 0.105).collect(),
        )?],
    ];

    let breakdown_fm = mpd_loss.generator_loss(
        &disc_fake_outputs,
        Some(&real_features),
        Some(&fake_features),
    )?;
    println!("{}", breakdown_fm.summary());

    Ok(())
}

/// Demonstrate combined multi-scale + multi-period loss
fn demonstrate_combined_loss() -> Result<(), Box<dyn std::error::Error>> {
    let config = CombinedDiscriminatorLossConfig {
        scale_weight: 1.0,
        period_weight: 1.0,
        ..Default::default()
    };
    let combined_loss = CombinedDiscriminatorLoss::new(config);

    println!("   State-of-the-art vocoders use BOTH:");
    println!("   • Multi-Scale: Captures frequency content at different resolutions");
    println!("   • Multi-Period: Captures periodic patterns at different periods\n");

    // Multi-scale outputs (3 scales)
    let scale_fake_outputs = vec![
        vec![Array1::from_vec(vec![0.80, 0.75, 0.82])],
        vec![Array1::from_vec(vec![0.77, 0.81])],
        vec![Array1::from_vec(vec![0.79])],
    ];

    // Multi-period outputs (5 periods)
    let period_fake_outputs = vec![
        vec![Array1::from_vec(vec![0.75, 0.72])],
        vec![Array1::from_vec(vec![0.78, 0.76])],
        vec![Array1::from_vec(vec![0.80, 0.77])],
        vec![Array1::from_vec(vec![0.73, 0.79])],
        vec![Array1::from_vec(vec![0.74, 0.81])],
    ];

    let breakdown = combined_loss.generator_loss(
        &scale_fake_outputs,
        &period_fake_outputs,
        None,
        None,
        None,
        None,
    )?;

    println!("   Combined Loss Breakdown:");
    println!("{}", breakdown.summary());

    println!("\n   Benefits of combined approach:");
    println!("   ✓ Better capture of audio structure");
    println!("   ✓ Improved perceptual quality");
    println!("   ✓ Faster convergence during training");
    println!("   ✓ More robust to different audio types");

    Ok(())
}

/// Simulate a typical training loop iteration
fn simulate_training_loop() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Simulating one training iteration...\n");

    // Setup losses
    let msd_config = MultiScaleDiscriminatorLossConfig::default();
    let msd_loss = MultiScaleDiscriminatorLoss::new(msd_config);

    // Simulate forward pass through discriminator
    println!("   Step 1: Forward pass through discriminators");
    let disc_real_outputs = vec![
        vec![Array1::from_vec(vec![0.92, 0.88, 0.95])],
        vec![Array1::from_vec(vec![0.90, 0.93])],
        vec![Array1::from_vec(vec![0.89])],
    ];

    let disc_fake_outputs = vec![
        vec![Array1::from_vec(vec![0.18, 0.22, 0.15])],
        vec![Array1::from_vec(vec![0.20, 0.17])],
        vec![Array1::from_vec(vec![0.19])],
    ];

    println!("           ✓ Real audio discriminator scores: high (0.88-0.95)");
    println!("           ✓ Fake audio discriminator scores: low (0.15-0.22)");

    // Step 2: Compute discriminator loss
    println!("\n   Step 2: Compute discriminator loss");
    let disc_loss = msd_loss.discriminator_loss(&disc_real_outputs, &disc_fake_outputs)?;
    println!("           Discriminator Loss: {:.6}", disc_loss);
    println!("           → Update discriminator weights (backprop)");

    // Step 3: Compute generator loss
    println!("\n   Step 3: Generate new samples and compute generator loss");
    let disc_fake_outputs_gen = vec![
        vec![Array1::from_vec(vec![0.75, 0.78, 0.72])],
        vec![Array1::from_vec(vec![0.76, 0.74])],
        vec![Array1::from_vec(vec![0.77])],
    ];

    println!("           New discriminator scores: moderate (0.72-0.78)");
    println!("           (Generator getting better at fooling discriminator)");

    let gen_breakdown = msd_loss.generator_loss(&disc_fake_outputs_gen, None, None)?;
    println!("\n           {}", gen_breakdown.summary());
    println!("           → Update generator weights (backprop)");

    println!("\n   Training iteration complete!");
    println!("   In practice, repeat for thousands of iterations...");

    Ok(())
}
