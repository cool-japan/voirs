//! # Advanced VoiRS Emotion Control Examples
//!
//! This file contains comprehensive examples demonstrating advanced features
//! of the VoiRS emotion control system including custom emotions, machine learning,
//! cross-cultural adaptation, and GPU acceleration.

use voirs_emotion::prelude::*;
use tokio::time::{Duration, Instant};
use std::collections::HashMap;

/// Example 1: Custom Emotion Creation and Usage
/// 
/// This example demonstrates how to create custom emotions with specific
/// characteristics and use them in the emotion processing pipeline.
#[tokio::main]
async fn example_1_custom_emotions() -> Result<()> {
    println!("=== Custom Emotion Creation Example ===");
    
    // Create a custom emotion with specific dimensional characteristics
    let nostalgic = CustomEmotionBuilder::new("nostalgic")
        .description("A bittersweet longing for the past")
        .dimensions(-0.2, -0.3, -0.1) // Valence, Arousal, Dominance
        .prosody(0.9, 0.8, 0.7) // Pitch, tempo, energy scaling
        .voice_quality(0.3, 0.1, -0.2, 0.2) // Breathiness, roughness, brightness, resonance
        .tags(["memory", "bittersweet", "contemplative"])
        .cultural_context("Western")
        .build()?;
    
    // Create another custom emotion for contrast
    let euphoric = CustomEmotionBuilder::new("euphoric")
        .description("Intense joy and exhilaration")
        .dimensions(0.9, 0.8, 0.7) // High positive valence, high arousal, high dominance
        .prosody(1.3, 1.2, 1.4) // Higher pitch, faster tempo, more energy
        .voice_quality(-0.1, 0.0, 0.4, 0.3) // Less breathiness, bright, resonant
        .tags(["intense", "positive", "energetic"])
        .cultural_context("Universal")
        .build()?;
    
    // Register custom emotions
    let mut registry = CustomEmotionRegistry::new();
    registry.register(nostalgic)?;
    registry.register(euphoric)?;
    
    // Create emotion processor with custom registry
    let processor = EmotionProcessor::builder()
        .config(
            EmotionConfig::builder()
                .enabled(true)
                .transition_smoothing(0.8)
                .build()?
        )
        .custom_registry(registry)
        .build()?;
    
    // Use custom emotions in processing
    processor.set_custom_emotion("nostalgic", Some(0.7)).await?;
    let state1 = processor.get_current_state().await;
    println!("Nostalgic emotion state: {:?}", state1.current.emotion_vector);
    
    processor.set_custom_emotion("euphoric", Some(0.9)).await?;
    let state2 = processor.get_current_state().await;
    println!("Euphoric emotion state: {:?}", state2.current.emotion_vector);
    
    // Mix custom and built-in emotions
    let mut emotion_mix = HashMap::new();
    emotion_mix.insert(Emotion::Custom("nostalgic".to_string()), 0.6);
    emotion_mix.insert(Emotion::Happy, 0.4);
    
    processor.set_emotion_mix(emotion_mix).await?;
    let mixed_state = processor.get_current_state().await;
    println!("Mixed emotion state: {:?}", mixed_state.current.emotion_vector);
    
    Ok(())
}

/// Example 2: Machine Learning-Based Emotion Personalization
///
/// This example shows how to use the emotion learning system to personalize
/// emotional expression based on user feedback and preferences.
#[tokio::main]
async fn example_2_emotion_learning() -> Result<()> {
    println!("=== Emotion Learning System Example ===");
    
    // Configure emotion learning system
    let learning_config = EmotionLearningConfig {
        enable_gpu_training: true,
        learning_rate: 0.001,
        batch_size: 32,
        max_training_iterations: 1000,
        convergence_threshold: 0.01,
        feature_dimensions: 64,
        hidden_layers: vec![128, 64, 32],
        dropout_rate: 0.1,
        validation_split: 0.2,
    };
    
    let learner = EmotionLearner::new(learning_config)?;
    
    // Simulate user feedback for different emotions and contexts
    let contexts = ["conversation", "presentation", "storytelling", "casual"];
    let emotions = [Emotion::Happy, Emotion::Sad, Emotion::Excited, Emotion::Calm];
    
    for (i, context) in contexts.iter().enumerate() {
        for (j, emotion) in emotions.iter().enumerate() {
            // Simulate varied user feedback
            let satisfaction = 0.3 + (i + j) as f32 * 0.1;
            let naturalness = 0.4 + (i * 2 + j) as f32 * 0.05;
            let intensity_rating = 0.5 + (j as f32 * 0.15);
            let authenticity = 0.6 + (i as f32 * 0.08);
            let appropriateness = 0.7 + ((i + j) as f32 * 0.03);
            
            let feedback = EmotionFeedback {
                user_id: "demo_user".to_string(),
                emotion: emotion.clone(),
                intensity: EmotionIntensity::new(0.7),
                context: context.to_string(),
                satisfaction,
                detailed_ratings: FeedbackRatings {
                    naturalness,
                    intensity: intensity_rating,
                    authenticity,
                    appropriateness,
                },
                timestamp: Instant::now(),
                session_id: format!("session_{}", i),
            };
            
            learner.add_feedback(feedback).await?;
        }
    }
    
    // Train the personalization model
    println!("Training personalization model...");
    learner.train_personalization_model("demo_user").await?;
    
    // Get personalized emotion parameters
    let base_emotion = EmotionParameters {
        arousal: 0.7,
        valence: 0.8,
        dominance: 0.6,
        intensity: EmotionIntensity::new(0.8),
        pitch_shift: 1.1,
        tempo_scale: 1.0,
        energy_scale: 1.2,
        breathiness: 0.1,
        roughness: 0.0,
        brightness: 0.3,
        resonance: 0.2,
    };
    
    let personalized = learner
        .get_personalized_emotion("demo_user", &base_emotion, "conversation")
        .await?;
    
    println!("Original emotion parameters: {:?}", base_emotion);
    println!("Personalized emotion parameters: {:?}", personalized);
    
    // Export user profile for persistence
    let profile = learner.export_profile("demo_user").await?;
    if let Some(profile) = profile {
        println!("User preference profile created with {} feedback entries", 
                 profile.learned_biases.len());
    }
    
    Ok(())
}

/// Example 3: Cross-Cultural Emotion Adaptation
///
/// Demonstrates how the cultural adaptation system modifies emotional expression
/// based on cultural context and social hierarchy.
#[tokio::main]
async fn example_3_cultural_adaptation() -> Result<()> {
    println!("=== Cross-Cultural Emotion Adaptation Example ===");
    
    let processor = EmotionProcessor::new()?;
    
    // Test emotion expression in different cultural contexts
    let cultures = ["japanese", "western", "arabic"];
    let base_emotion = Emotion::Happy;
    let base_intensity = Some(0.8);
    
    for culture in &cultures {
        println!("\n--- Testing in {} culture ---", culture);
        
        // Set cultural context
        processor.set_cultural_context(culture).await?;
        
        // Test different social contexts
        let social_contexts = [
            (SocialContext::Formal, Some(SocialHierarchy::Superior)),
            (SocialContext::Informal, None),
            (SocialContext::Professional, Some(SocialHierarchy::Peer)),
            (SocialContext::Personal, None),
        ];
        
        for (social_context, hierarchy) in &social_contexts {
            processor.set_emotion_with_cultural_context(
                base_emotion.clone(),
                base_intensity,
                social_context.clone(),
                hierarchy.clone(),
            ).await?;
            
            let state = processor.get_current_state().await;
            let actual_intensity = state
                .current
                .emotion_vector
                .emotions
                .get(&base_emotion)
                .unwrap_or(&0.0);
            
            println!(
                "  {:?} context with {:?} hierarchy: intensity {:.2} (from {:.2})",
                social_context,
                hierarchy.as_ref().unwrap_or(&SocialHierarchy::Peer),
                actual_intensity,
                base_intensity.unwrap_or(0.0)
            );
        }
    }
    
    // Demonstrate cultural emotion mapping differences
    processor.set_cultural_context("japanese").await?;
    processor.set_emotion_with_cultural_context(
        Emotion::Angry,
        Some(0.8),
        SocialContext::Formal,
        Some(SocialHierarchy::Superior),
    ).await?;
    
    let japanese_anger = processor.get_current_state().await;
    
    processor.set_cultural_context("western").await?;
    processor.set_emotion_with_cultural_context(
        Emotion::Angry,
        Some(0.8),
        SocialContext::Formal,
        Some(SocialHierarchy::Superior),
    ).await?;
    
    let western_anger = processor.get_current_state().await;
    
    println!("\nAnger expression comparison:");
    println!("Japanese formal context: {:?}", japanese_anger.current.emotion_vector);
    println!("Western formal context: {:?}", western_anger.current.emotion_vector);
    
    Ok(())
}

/// Example 4: Comprehensive Emotion History and Analysis
///
/// Shows how to track and analyze emotion history for insights and patterns.
#[tokio::main]
async fn example_4_emotion_history() -> Result<()> {
    println!("=== Emotion History and Analysis Example ===");
    
    // Configure comprehensive history tracking
    let history_config = EmotionHistoryConfig {
        max_entries: 1000,
        max_age: Duration::from_secs(24 * 60 * 60), // 24 hours
        track_duration: true,
        min_interval: Duration::from_millis(100),
        enable_compression: true,
        compression_rate: 10,
    };
    
    let processor = EmotionProcessor::builder()
        .config(
            EmotionConfig::builder()
                .enabled(true)
                .history_config(history_config)
                .build()?
        )
        .build()?;
    
    // Simulate a conversation with varying emotions
    let conversation_emotions = [
        (Emotion::Neutral, 0.5, "Starting conversation"),
        (Emotion::Happy, 0.7, "Greeting friend"),
        (Emotion::Excited, 0.8, "Sharing good news"),
        (Emotion::Surprised, 0.6, "Unexpected response"),
        (Emotion::Thoughtful, 0.5, "Considering options"),
        (Emotion::Sad, 0.4, "Discussing problem"),
        (Emotion::Empathetic, 0.7, "Offering support"),
        (Emotion::Hopeful, 0.6, "Suggesting solution"),
        (Emotion::Happy, 0.8, "Resolution achieved"),
        (Emotion::Content, 0.7, "Ending on positive note"),
    ];
    
    for (emotion, intensity, context) in &conversation_emotions {
        processor.set_emotion(emotion.clone(), Some(*intensity)).await?;
        processor.add_to_history_with_context(context).await?;
        
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    
    // Analyze emotion history
    println!("Total emotion entries: {}", processor.get_history_count().await);
    
    let stats = processor.get_history_stats().await;
    println!("Emotion distribution: {:?}", stats.emotion_distribution);
    println!("Average intensity: {:.2}", stats.average_intensity);
    println!("Session duration: {:?}", stats.total_duration);
    
    // Detect emotion patterns
    let patterns = processor.get_emotion_patterns().await;
    println!("\nDetected emotion patterns:");
    for pattern in patterns.iter().take(5) {
        println!("  Pattern: {:?} -> {:?} (confidence: {:.2})",
                 pattern.sequence[0], 
                 pattern.sequence.last().unwrap_or(&Emotion::Neutral),
                 pattern.confidence);
    }
    
    // Analyze emotion transitions
    let transitions = processor.get_emotion_transitions().await;
    println!("\nEmotion transitions:");
    for transition in transitions.iter().take(5) {
        println!("  {} -> {} (duration: {:.1}s, frequency: {})",
                 format!("{:?}", transition.from_emotion),
                 format!("{:?}", transition.to_emotion),
                 transition.average_duration.as_secs_f32(),
                 transition.frequency);
    }
    
    // Export history for external analysis
    let json_export = processor.export_history_json().await?;
    println!("\nHistory exported: {} characters", json_export.len());
    
    Ok(())
}

/// Example 5: GPU-Accelerated Emotion Processing
///
/// Demonstrates high-performance emotion processing using GPU acceleration.
#[tokio::main]
async fn example_5_gpu_acceleration() -> Result<()> {
    println!("=== GPU-Accelerated Emotion Processing Example ===");
    
    // Create GPU-enabled processor
    let config = EmotionConfig::builder()
        .use_gpu(true)
        .enabled(true)
        .build()?;
    
    let processor = EmotionProcessor::with_config(config)?;
    
    // Check GPU availability
    if processor.is_gpu_enabled() {
        println!("GPU acceleration is enabled!");
        
        // Benchmark GPU vs CPU performance
        if let Some((gpu_time, cpu_time)) = processor.benchmark_gpu_performance(1000) {
            println!("Performance comparison for 1000 operations:");
            println!("  GPU time: {:.2}ms", gpu_time);
            println!("  CPU time: {:.2}ms", cpu_time);
            println!("  Speedup: {:.1}x", cpu_time / gpu_time);
        }
    } else {
        println!("GPU not available, using CPU fallback");
    }
    
    // Process large audio buffer with emotion effects
    let large_audio = vec![0.1; 44100]; // 1 second of audio at 44.1kHz
    
    processor.set_emotion(Emotion::Excited, Some(0.9)).await?;
    
    let start = Instant::now();
    let processed_audio = processor.process_audio(&large_audio).await?;
    let processing_time = start.elapsed();
    
    println!("Processed {} samples in {:.2}ms ({} enabled)",
             processed_audio.len(),
             processing_time.as_secs_f64() * 1000.0,
             if processor.is_gpu_enabled() { "GPU" } else { "CPU" });
    
    // Demonstrate different GPU-accelerated effects
    let emotions_and_effects = [
        (Emotion::Happy, "High energy, bright tone"),
        (Emotion::Sad, "Low energy, darker tone"),
        (Emotion::Angry, "Distorted, aggressive tone"),
        (Emotion::Calm, "Smooth, relaxed tone"),
    ];
    
    for (emotion, description) in &emotions_and_effects {
        processor.set_emotion(emotion.clone(), Some(0.8)).await?;
        let start = Instant::now();
        let _processed = processor.process_audio(&large_audio).await?;
        let time = start.elapsed();
        
        println!("  {}: {:.2}ms - {}", 
                 format!("{:?}", emotion), 
                 time.as_secs_f64() * 1000.0,
                 description);
    }
    
    Ok(())
}

/// Example 6: Natural Variation Generation
///
/// Shows how to add realistic micro-variations to emotional expression.
#[tokio::main]
async fn example_6_natural_variation() -> Result<()> {
    println!("=== Natural Variation Generation Example ===");
    
    // Configure natural variation system
    let variation_config = NaturalVariationConfig {
        base_variation_intensity: 0.2, // 20% base variation
        temporal_frequency: 0.8,
        enable_prosodic_variation: true,
        enable_voice_quality_variation: true,
        enable_breathing_patterns: true,
        emotion_scaling: {
            let mut scaling = HashMap::new();
            scaling.insert("Happy".to_string(), 1.4);
            scaling.insert("Excited".to_string(), 1.6);
            scaling.insert("Calm".to_string(), 0.6);
            scaling.insert("Sad".to_string(), 0.8);
            scaling
        },
        speaker_characteristics_influence: 0.4,
        random_seed: Some(42), // For reproducible results
        smoothing_factor: 0.7,
    };
    
    // Create speaker characteristics for different speaker types
    let speakers = [
        ("young_energetic", SpeakerCharacteristics {
            age: 25.0,
            gender: 0.8, // More feminine characteristics
            expressiveness: 0.9,
            stability: 0.6,
            base_pitch: 180.0,
            pitch_range: 80.0,
        }),
        ("mature_professional", SpeakerCharacteristics {
            age: 45.0,
            gender: 0.2, // More masculine characteristics
            expressiveness: 0.6,
            stability: 0.9,
            base_pitch: 120.0,
            pitch_range: 40.0,
        }),
        ("elderly_gentle", SpeakerCharacteristics {
            age: 70.0,
            gender: 0.5, // Neutral
            expressiveness: 0.7,
            stability: 0.8,
            base_pitch: 140.0,
            pitch_range: 30.0,
        }),
    ];
    
    for (speaker_name, characteristics) in &speakers {
        println!("\n--- {} Speaker ---", speaker_name);
        
        let mut generator = NaturalVariationGenerator::new(
            variation_config.clone(), 
            characteristics.clone()
        );
        
        // Generate variations for different emotions
        let emotions = [Emotion::Happy, Emotion::Sad, Emotion::Excited, Emotion::Calm];
        
        for emotion in &emotions {
            let variation = generator.generate_variation(
                emotion.clone(),
                EmotionIntensity::new(0.8),
                Duration::from_secs(2),
            )?;
            
            println!("  {:?} emotion variation:", emotion);
            println!("    Applied patterns: {}", variation.applied_patterns.len());
            
            // Show energy scaling differences
            let energy_patterns: Vec<_> = variation.applied_patterns
                .iter()
                .filter(|p| matches!(p.variation_type, VariationType::Prosodic))
                .take(3)
                .collect();
                
            for pattern in energy_patterns {
                println!("    - {}: amplitude {:.2}, frequency {:.1}Hz",
                         pattern.name,
                         pattern.amplitude,
                         pattern.frequency);
            }
        }
        
        // Get variation statistics
        let stats = generator.get_variation_statistics();
        println!("  Statistics: {} total patterns, avg intensity {:.2}",
                 stats.total_patterns_applied,
                 stats.average_variation_intensity);
    }
    
    Ok(())
}

/// Example 7: A/B Testing and Quality Validation
///
/// Demonstrates the A/B testing framework for comparing emotion processing approaches.
#[tokio::main]
async fn example_7_ab_testing() -> Result<()> {
    println!("=== A/B Testing and Quality Validation Example ===");
    
    // Configure A/B testing
    let ab_config = ABTestConfig {
        max_comparisons: 100,
        confidence_level: 0.95,
        min_effect_size: 0.1,
        max_duration: Duration::from_secs(3600), // 1 hour
        randomization_seed: Some(42),
    };
    
    let mut ab_manager = ABTestManager::new(ab_config);
    
    // Create test variants for different emotion processing approaches
    let control_variant = ABTestVariant {
        variant_id: "control".to_string(),
        name: "Standard Emotion Processing".to_string(),
        description: "Current emotion processing with standard parameters".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("smoothing".to_string(), "0.7".to_string());
            params.insert("intensity_scaling".to_string(), "1.0".to_string());
            params.insert("gpu_enabled".to_string(), "false".to_string());
            params
        },
        allocation_weight: 0.5,
    };
    
    let treatment_variant = ABTestVariant {
        variant_id: "enhanced".to_string(),
        name: "Enhanced Emotion Processing".to_string(),
        description: "Improved processing with GPU acceleration and natural variation".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("smoothing".to_string(), "0.9".to_string());
            params.insert("intensity_scaling".to_string(), "1.2".to_string());
            params.insert("gpu_enabled".to_string(), "true".to_string());
            params.insert("natural_variation".to_string(), "true".to_string());
            params
        },
        allocation_weight: 0.5,
    };
    
    ab_manager.add_variant(control_variant)?;
    ab_manager.add_variant(treatment_variant)?;
    
    // Simulate A/B test comparisons
    for i in 0..50 {
        if let Some((variant_a, variant_b)) = ab_manager.next_comparison_pair() {
            // Simulate quality scores (in real usage, these would come from 
            // perceptual evaluation or automated quality metrics)
            let score_a = 0.6 + (i as f32 * 0.008) + (scirs2_core::random::random::<f32>() * 0.2);
            let score_b = 0.7 + (i as f32 * 0.004) + (scirs2_core::random::random::<f32>() * 0.15);
            
            let comparison = ABComparison {
                variant_a_id: variant_a.variant_id.clone(),
                variant_b_id: variant_b.variant_id.clone(),
                quality_score_a: score_a,
                quality_score_b: score_b,
                preference: if score_a > score_b { 
                    variant_a.variant_id.clone() 
                } else { 
                    variant_b.variant_id.clone() 
                },
                confidence: (score_a - score_b).abs() + 0.1,
                evaluation_time: Duration::from_millis(200 + (scirs2_core::random::random::<u64>() % 300)),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("evaluator".to_string(), format!("evaluator_{}", i % 5));
                    meta.insert("emotion".to_string(), "happy".to_string());
                    meta
                },
            };
            
            ab_manager.add_comparison(comparison)?;
        }
        
        if ab_manager.has_significant_result() {
            break;
        }
    }
    
    // Analyze results
    let stats = ab_manager.calculate_statistics()?;
    println!("A/B Test Results:");
    println!("  Total comparisons: {}", stats.total_comparisons);
    println!("  Statistical significance: {}", stats.is_significant);
    println!("  Confidence level: {:.2}%", stats.confidence_level * 100.0);
    
    if let Some(winner) = &stats.winning_variant {
        println!("  Winning variant: {} (p-value: {:.4})", 
                 winner, 
                 stats.p_value.unwrap_or(1.0));
        println!("  Effect size: {:.3}", stats.effect_size.unwrap_or(0.0));
    } else {
        println!("  No statistically significant winner found");
    }
    
    // Generate recommendations
    if let Some(recommendation) = ab_manager.get_recommendation() {
        println!("  Recommendation: {}", recommendation);
    }
    
    // Set up perceptual validation study
    let validation_config = PerceptualValidationConfig {
        max_evaluations: 200,
        max_evaluators: 10,
        max_duration: Duration::from_secs(7200), // 2 hours
        randomize_order: true,
        require_training: true,
        evaluation_criteria: EvaluationCriteria {
            naturalness_weight: 0.3,
            appropriateness_weight: 0.25,
            quality_weight: 0.25,
            preference_weight: 0.2,
        },
    };
    
    let mut validation_study = PerceptualValidationStudy::new(validation_config);
    
    // Simulate some perceptual evaluations
    for i in 0..20 {
        let evaluation = PerceptualEvaluation {
            evaluator_id: format!("evaluator_{}", i % 5),
            emotion: Emotion::Happy,
            intensity: EmotionIntensity::new(0.8),
            perceived_emotion: Emotion::Happy, // Would be determined by human evaluator
            perceived_intensity: EmotionIntensity::new(0.75 + (scirs2_core::random::random::<f32>() * 0.2)),
            naturalness_score: 0.7 + (scirs2_core::random::random::<f32>() * 0.25),
            appropriateness_score: 0.8 + (scirs2_core::random::random::<f32>() * 0.15),
            overall_quality: 0.75 + (scirs2_core::random::random::<f32>() * 0.2),
            confidence: 0.8 + (scirs2_core::random::random::<f32>() * 0.15),
            evaluation_time: Duration::from_millis(2000 + (scirs2_core::random::random::<u64>() % 3000)),
            comments: Some(format!("Evaluation {} - emotion seems natural", i)),
        };
        
        validation_study.add_evaluation(evaluation)?;
    }
    
    let validation_stats = validation_study.calculate_statistics()?;
    println!("\nPerceptual Validation Results:");
    println!("  Total evaluations: {}", validation_stats.total_evaluations);
    println!("  Recognition accuracy: {:.1}%", validation_stats.recognition_accuracy * 100.0);
    println!("  Average naturalness: {:.2}", validation_stats.average_naturalness);
    println!("  Inter-evaluator agreement: {:.3}", validation_stats.inter_evaluator_agreement);
    println!("  Composite quality score: {:.2}", validation_stats.composite_score);
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎭 VoiRS Emotion Control System - Advanced Examples\n");
    
    // Run all examples
    example_1_custom_emotions().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_2_emotion_learning().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_3_cultural_adaptation().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_4_emotion_history().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_5_gpu_acceleration().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_6_natural_variation().await?;
    println!("\n{}\n", "=".repeat(60));
    
    example_7_ab_testing().await?;
    
    println!("\n🎉 All advanced examples completed successfully!");
    
    Ok(())
}