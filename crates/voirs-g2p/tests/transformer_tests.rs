//! Comprehensive tests for transformer G2P components
//!
//! Tests the neural transformer architecture including:
//! - Multi-head attention mechanisms
//! - Positional encoding
//! - Layer normalization
//! - Transformer encoder/decoder layers
//! - Complete TransformerG2P model
//! - Advanced sampling strategies
//! - Label smoothing
//! - Learning rate scheduling

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use voirs_g2p::backends::neural::core::{
    ALiBiPositionBias, LabelSmoothing, LayerNorm, LearningRateScheduler, MultiHeadAttention,
    PositionalEncoding, RotaryPositionEmbedding, SamplingStrategy, SchedulerType,
    SwiGLUFeedForward, TransformerDecoderLayer, TransformerEncoderLayer, TransformerG2P,
};

/// Helper function to create a VarBuilder for testing
fn create_test_var_builder() -> VarBuilder<'static> {
    let device = Device::Cpu;
    VarBuilder::zeros(candle_core::DType::F32, &device)
}

#[test]
fn test_positional_encoding_creation() {
    let device = Device::Cpu;
    let max_seq_len = 100;
    let hidden_size = 512;

    let pos_enc = PositionalEncoding::new(max_seq_len, hidden_size, &device);
    assert!(pos_enc.is_ok(), "Failed to create positional encoding");
}

#[test]
fn test_positional_encoding_forward() {
    let device = Device::Cpu;
    let max_seq_len = 100;
    let hidden_size = 512;
    let batch_size = 2;
    let seq_len = 10;

    let pos_enc = PositionalEncoding::new(max_seq_len, hidden_size, &device).unwrap();

    // Create dummy input: (batch, seq_len, hidden_size)
    let input = Tensor::zeros(
        &[batch_size, seq_len, hidden_size],
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let output = pos_enc.forward(&input);
    assert!(output.is_ok(), "Positional encoding forward failed");

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_positional_encoding_exceeds_max_length() {
    let device = Device::Cpu;
    let max_seq_len = 10;
    let hidden_size = 512;
    let seq_len = 20; // Exceeds max_seq_len

    let pos_enc = PositionalEncoding::new(max_seq_len, hidden_size, &device).unwrap();

    let input =
        Tensor::zeros(&[1, seq_len, hidden_size], candle_core::DType::F32, &device).unwrap();

    let output = pos_enc.forward(&input);
    assert!(
        output.is_err(),
        "Should fail when sequence length exceeds maximum"
    );
}

#[test]
fn test_layer_norm_creation() {
    let vb = create_test_var_builder();
    let normalized_shape = 512;
    let eps = 1e-5;

    let layer_norm = LayerNorm::new(normalized_shape, eps, vb);
    assert!(layer_norm.is_ok(), "Failed to create layer normalization");
}

#[test]
fn test_layer_norm_forward() {
    let vb = create_test_var_builder();
    let normalized_shape = 512;
    let eps = 1e-5;
    let device = Device::Cpu;

    let layer_norm = LayerNorm::new(normalized_shape, eps, vb).unwrap();

    // Create random input
    let input = Tensor::randn(0.0f32, 1.0, &[2, 10, normalized_shape], &device).unwrap();

    let output = layer_norm.forward(&input);
    assert!(output.is_ok(), "Layer norm forward failed");

    let output = output.unwrap();
    assert_eq!(output.dims(), input.dims());
}

#[test]
fn test_multi_head_attention_creation() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb);
    assert!(mha.is_ok(), "Failed to create multi-head attention");
}

#[test]
fn test_multi_head_attention_invalid_dims() {
    let vb = create_test_var_builder();
    let hidden_size = 513; // Not divisible by num_heads
    let num_heads = 8;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb);
    assert!(
        mha.is_err(),
        "Should fail when hidden_size is not divisible by num_heads"
    );
}

#[test]
fn test_multi_head_attention_forward() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;
    let device = Device::Cpu;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb).unwrap();

    let batch_size = 2;
    let seq_len = 10;

    // Create dummy query, key, value
    let query = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();
    let key = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();
    let value = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

    let output = mha.forward(&query, &key, &value, None);
    if let Err(e) = &output {
        eprintln!("Multi-head attention forward error: {:?}", e);
    }
    assert!(
        output.is_ok(),
        "Multi-head attention forward failed: {:?}",
        output.err()
    );

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_transformer_encoder_layer_creation() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;
    let ff_dim = 2048;
    let dropout = 0.1;

    let encoder = TransformerEncoderLayer::new(hidden_size, num_heads, ff_dim, dropout, vb);
    assert!(
        encoder.is_ok(),
        "Failed to create transformer encoder layer"
    );
}

#[test]
fn test_transformer_encoder_layer_forward() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;
    let ff_dim = 2048;
    let dropout = 0.1;
    let device = Device::Cpu;

    let encoder =
        TransformerEncoderLayer::new(hidden_size, num_heads, ff_dim, dropout, vb).unwrap();

    let batch_size = 2;
    let seq_len = 10;

    let input = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

    let output = encoder.forward(&input, None);
    assert!(output.is_ok(), "Transformer encoder forward failed");

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_transformer_decoder_layer_creation() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;
    let ff_dim = 2048;
    let dropout = 0.1;

    let decoder = TransformerDecoderLayer::new(hidden_size, num_heads, ff_dim, dropout, vb);
    assert!(
        decoder.is_ok(),
        "Failed to create transformer decoder layer"
    );
}

#[test]
fn test_transformer_decoder_layer_forward() {
    let vb = create_test_var_builder();
    let hidden_size = 512;
    let num_heads = 8;
    let ff_dim = 2048;
    let dropout = 0.1;
    let device = Device::Cpu;

    let decoder =
        TransformerDecoderLayer::new(hidden_size, num_heads, ff_dim, dropout, vb).unwrap();

    let batch_size = 2;
    let tgt_len = 8;
    let src_len = 10;

    let target = Tensor::randn(0.0f32, 1.0, &[batch_size, tgt_len, hidden_size], &device).unwrap();
    let encoder_output =
        Tensor::randn(0.0f32, 1.0, &[batch_size, src_len, hidden_size], &device).unwrap();

    let output = decoder.forward(&target, &encoder_output, None, None);
    if let Err(e) = &output {
        eprintln!("Transformer decoder forward error: {:?}", e);
    }
    assert!(
        output.is_ok(),
        "Transformer decoder forward failed: {:?}",
        output.err()
    );

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, tgt_len, hidden_size]);
}

#[test]
fn test_transformer_g2p_creation() {
    let vb = create_test_var_builder();
    let grapheme_vocab_size = 100;
    let phoneme_vocab_size = 50;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 100;
    let dropout = 0.1;

    let transformer = TransformerG2P::new(
        grapheme_vocab_size,
        phoneme_vocab_size,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        max_seq_len,
        dropout,
        vb,
    );

    assert!(transformer.is_ok(), "Failed to create TransformerG2P model");
}

#[test]
fn test_transformer_g2p_encode() {
    let vb = create_test_var_builder();
    let grapheme_vocab_size = 100;
    let phoneme_vocab_size = 50;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 100;
    let dropout = 0.1;
    let device = Device::Cpu;

    let transformer = TransformerG2P::new(
        grapheme_vocab_size,
        phoneme_vocab_size,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        max_seq_len,
        dropout,
        vb,
    )
    .unwrap();

    let batch_size = 2;
    let seq_len = 10;

    // Create dummy grapheme IDs
    let grapheme_ids = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, seq_len, grapheme_vocab_size],
        &device,
    )
    .unwrap();

    let encoded = transformer.encode(&grapheme_ids, None);
    assert!(encoded.is_ok(), "Transformer encode failed");

    let encoded = encoded.unwrap();
    assert_eq!(encoded.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_transformer_g2p_decode() {
    let vb = create_test_var_builder();
    let grapheme_vocab_size = 100;
    let phoneme_vocab_size = 50;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 100;
    let dropout = 0.1;
    let device = Device::Cpu;

    let transformer = TransformerG2P::new(
        grapheme_vocab_size,
        phoneme_vocab_size,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        max_seq_len,
        dropout,
        vb,
    )
    .unwrap();

    let batch_size = 2;
    let tgt_len = 8;
    let src_len = 10;

    let phoneme_ids = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, tgt_len, phoneme_vocab_size],
        &device,
    )
    .unwrap();
    let encoder_output =
        Tensor::randn(0.0f32, 1.0, &[batch_size, src_len, hidden_size], &device).unwrap();

    let decoded = transformer.decode(&phoneme_ids, &encoder_output, None, None);
    assert!(decoded.is_ok(), "Transformer decode failed");

    let decoded = decoded.unwrap();
    assert_eq!(decoded.dims(), &[batch_size, tgt_len, phoneme_vocab_size]);
}

#[test]
fn test_transformer_g2p_forward() {
    let vb = create_test_var_builder();
    let grapheme_vocab_size = 100;
    let phoneme_vocab_size = 50;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 100;
    let dropout = 0.1;
    let device = Device::Cpu;

    let transformer = TransformerG2P::new(
        grapheme_vocab_size,
        phoneme_vocab_size,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        max_seq_len,
        dropout,
        vb,
    )
    .unwrap();

    let batch_size = 2;
    let src_len = 10;
    let tgt_len = 8;

    let grapheme_ids = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, src_len, grapheme_vocab_size],
        &device,
    )
    .unwrap();
    let phoneme_ids = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, tgt_len, phoneme_vocab_size],
        &device,
    )
    .unwrap();

    let output = transformer.forward(&grapheme_ids, &phoneme_ids, None, None, None);
    assert!(output.is_ok(), "Transformer forward failed");

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, tgt_len, phoneme_vocab_size]);
}

#[test]
fn test_transformer_g2p_causal_mask() {
    let vb = create_test_var_builder();
    let grapheme_vocab_size = 100;
    let phoneme_vocab_size = 50;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 100;
    let dropout = 0.1;

    let transformer = TransformerG2P::new(
        grapheme_vocab_size,
        phoneme_vocab_size,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        max_seq_len,
        dropout,
        vb,
    )
    .unwrap();

    let seq_len = 5;
    let mask = transformer.generate_causal_mask(seq_len);
    assert!(mask.is_ok(), "Failed to generate causal mask");

    let mask = mask.unwrap();
    assert_eq!(mask.dims(), &[seq_len, seq_len]);

    // Verify mask structure: lower triangle should be 0, upper triangle should be -inf
    let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j > i {
                assert!(
                    mask_data[idx].is_infinite() && mask_data[idx].is_sign_negative(),
                    "Upper triangle should be -inf at ({}, {})",
                    i,
                    j
                );
            } else {
                assert_eq!(
                    mask_data[idx], 0.0,
                    "Lower triangle should be 0 at ({}, {})",
                    i, j
                );
            }
        }
    }
}

#[test]
fn test_sampling_strategy_greedy() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::greedy();

    // Greedy should have temperature 0.0
    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5], 5, &device).unwrap();
    let result = strategy.sample(&logits, &[]);

    assert!(result.is_ok(), "Greedy sampling failed");
    let sampled = result.unwrap();

    // Greedy should always pick the highest logit (index 2)
    assert_eq!(sampled, 2, "Greedy sampling should pick the highest logit");
}

#[test]
fn test_sampling_strategy_temperature() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::new(0.8);

    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5], 5, &device).unwrap();
    let result = strategy.sample(&logits, &[]);

    assert!(result.is_ok(), "Temperature sampling failed");
    let sampled = result.unwrap();

    // Should return a valid index
    assert!(sampled < 5, "Sampled index out of bounds");
}

#[test]
fn test_sampling_strategy_top_k() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::new(1.0).with_top_k(3);

    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5], 5, &device).unwrap();
    let result = strategy.sample(&logits, &[]);

    assert!(result.is_ok(), "Top-k sampling failed");
    let sampled = result.unwrap();

    // Should return a valid index
    assert!(sampled < 5, "Sampled index out of bounds");
}

#[test]
fn test_sampling_strategy_top_p() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::new(1.0).with_top_p(0.9);

    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5], 5, &device).unwrap();
    let result = strategy.sample(&logits, &[]);

    assert!(result.is_ok(), "Top-p sampling failed");
    let sampled = result.unwrap();

    // Should return a valid index
    assert!(sampled < 5, "Sampled index out of bounds");
}

#[test]
fn test_sampling_strategy_repetition_penalty() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::new(1.0).with_repetition_penalty(1.2);

    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5], 5, &device).unwrap();
    let previous = vec![2]; // Previously generated token 2

    let result = strategy.sample(&logits, &previous);

    assert!(result.is_ok(), "Repetition penalty sampling failed");
    let sampled = result.unwrap();

    // Should return a valid index
    assert!(sampled < 5, "Sampled index out of bounds");
}

#[test]
fn test_sampling_strategy_combined() {
    let device = Device::Cpu;
    let strategy = SamplingStrategy::new(0.9)
        .with_top_k(5)
        .with_top_p(0.95)
        .with_repetition_penalty(1.1);

    let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.5, 0.5, 2.5, 1.8], 7, &device).unwrap();
    let previous = vec![1, 3];

    let result = strategy.sample(&logits, &previous);

    assert!(result.is_ok(), "Combined sampling failed");
    let sampled = result.unwrap();

    // Should return a valid index
    assert!(sampled < 7, "Sampled index out of bounds");
}

#[test]
fn test_label_smoothing_creation() {
    let vocab_size = 100;
    let smoothing = 0.1;

    // Note: LabelSmoothing::new takes (smoothing, vocab_size) in that order
    let label_smoothing = LabelSmoothing::new(smoothing, vocab_size);

    // Test by using smooth_labels method
    let target_ids = vec![0];
    let smoothed = label_smoothing.smooth_labels(&target_ids);

    // Verify we got a result
    assert_eq!(smoothed.len(), 1);
    assert_eq!(smoothed[0].len(), vocab_size);
}

#[test]
fn test_label_smoothing_smooth_labels() {
    let vocab_size = 10;
    let smoothing = 0.1;
    let label_smoothing = LabelSmoothing::new(smoothing, vocab_size);

    let target_ids = vec![0, 1, 2];
    let smoothed = label_smoothing.smooth_labels(&target_ids);

    assert_eq!(smoothed.len(), 3, "Should have 3 smoothed distributions");

    // Check each distribution
    for (i, distribution) in smoothed.iter().enumerate() {
        assert_eq!(
            distribution.len(),
            vocab_size,
            "Distribution should have vocab_size elements"
        );

        // Sum should be approximately 1.0
        let sum: f32 = distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Distribution should sum to 1.0");

        // Target class should have higher probability
        let target_id = target_ids[i];
        let confidence = 1.0 - smoothing;
        let smoothing_value = smoothing / (vocab_size as f32 - 1.0);

        assert!(
            (distribution[target_id] - confidence).abs() < 1e-4,
            "Target class should have confidence probability"
        );

        // Other classes should have smoothing value
        for (j, &prob) in distribution.iter().enumerate() {
            if j != target_id {
                assert!(
                    (prob - smoothing_value).abs() < 1e-4,
                    "Non-target classes should have smoothing value"
                );
            }
        }
    }
}

#[test]
fn test_learning_rate_scheduler_linear() {
    let base_lr = 0.001;
    let warmup_steps = 1000;
    let total_steps = 10000;

    let scheduler =
        LearningRateScheduler::new(base_lr, warmup_steps, total_steps, SchedulerType::Linear);

    // Step 0 should return 0.0
    assert_eq!(scheduler.get_lr(0), 0.0);

    // Warmup phase: linear increase
    let lr_half_warmup = scheduler.get_lr(warmup_steps / 2);
    assert!(
        (lr_half_warmup - base_lr * 0.5).abs() < 1e-6,
        "LR at half warmup should be half of base_lr"
    );

    // At end of warmup
    let lr_end_warmup = scheduler.get_lr(warmup_steps);
    assert!(
        (lr_end_warmup - base_lr).abs() < 1e-6,
        "LR at end of warmup should be base_lr"
    );

    // Decay phase: linear decrease
    let lr_middle = scheduler.get_lr((total_steps + warmup_steps) / 2);
    assert!(
        lr_middle < base_lr && lr_middle > 0.0,
        "LR in decay phase should be between 0 and base_lr"
    );

    // At total steps, should be close to 0
    let lr_end = scheduler.get_lr(total_steps);
    assert!(lr_end < base_lr * 0.1, "LR at end should be close to 0");
}

#[test]
fn test_learning_rate_scheduler_cosine() {
    let base_lr = 0.001;
    let warmup_steps = 1000;
    let total_steps = 10000;

    let scheduler =
        LearningRateScheduler::new(base_lr, warmup_steps, total_steps, SchedulerType::Cosine);

    // Warmup phase should be same as linear
    let lr_half_warmup = scheduler.get_lr(warmup_steps / 2);
    assert!(
        (lr_half_warmup - base_lr * 0.5).abs() < 1e-6,
        "LR at half warmup should be half of base_lr"
    );

    // Decay phase: cosine annealing
    let lr_middle = scheduler.get_lr((total_steps + warmup_steps) / 2);
    assert!(
        lr_middle < base_lr && lr_middle > 0.0,
        "LR in decay phase should be between 0 and base_lr"
    );

    // Cosine should provide smoother decay than linear, approaching but not necessarily > 0
    let lr_end = scheduler.get_lr(total_steps);
    assert!(
        lr_end >= 0.0 && lr_end < base_lr * 0.1,
        "Cosine decay should approach 0 (got lr_end = {})",
        lr_end
    );
}

#[test]
fn test_learning_rate_scheduler_transformer() {
    let base_lr = 1.0; // For transformer schedule, base_lr is often 1.0
    let warmup_steps = 4000;
    let total_steps = 100000;

    let scheduler = LearningRateScheduler::new(
        base_lr,
        warmup_steps,
        total_steps,
        SchedulerType::Transformer,
    );

    // LR should increase during warmup
    let lr_early = scheduler.get_lr(100);
    let lr_mid_warmup = scheduler.get_lr(2000);
    let lr_end_warmup = scheduler.get_lr(warmup_steps);

    assert!(lr_mid_warmup > lr_early, "LR should increase during warmup");
    assert!(
        lr_end_warmup > lr_mid_warmup,
        "LR should continue increasing"
    );

    // After warmup, LR should decrease
    let lr_after_warmup = scheduler.get_lr(warmup_steps * 2);
    assert!(
        lr_after_warmup < lr_end_warmup,
        "LR should decrease after warmup"
    );
}

#[test]
fn test_learning_rate_scheduler_warmup_consistency() {
    let base_lr = 0.001;
    let warmup_steps = 1000;
    let total_steps = 10000;

    // All scheduler types should have consistent warmup behavior
    let linear =
        LearningRateScheduler::new(base_lr, warmup_steps, total_steps, SchedulerType::Linear);
    let cosine =
        LearningRateScheduler::new(base_lr, warmup_steps, total_steps, SchedulerType::Cosine);

    for step in &[0, 250, 500, 750, 1000] {
        let lr_linear = linear.get_lr(*step);
        let lr_cosine = cosine.get_lr(*step);

        if *step <= warmup_steps {
            assert!(
                (lr_linear - lr_cosine).abs() < 1e-6,
                "Linear and cosine should have same warmup at step {}",
                step
            );
        }
    }
}

// ============================================================================
// Modern Transformer Components Tests (RoPE, SwiGLU, ALiBi)
// ============================================================================

/// Test RoPE initialization and basic properties
#[test]
fn test_rope_initialization() {
    let device = Device::Cpu;
    let dim = 64;
    let max_seq_len = 128;
    let base = 10000.0;

    let rope = RotaryPositionEmbedding::new(dim, max_seq_len, base, &device);
    assert!(rope.is_ok(), "Failed to create RoPE");
}

/// Test RoPE forward pass
#[test]
fn test_rope_forward() {
    let device = Device::Cpu;
    let dim = 64;
    let max_seq_len = 128;
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 16;

    let rope = RotaryPositionEmbedding::new(dim, max_seq_len, 10000.0, &device).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, seq_len, dim), &device).unwrap();

    let output = rope.apply_rotary_embedding(&input, 0);
    assert!(output.is_ok(), "Failed to apply RoPE");

    let output = output.unwrap();
    assert_eq!(output.dims(), input.dims());
}

/// Test RoPE with different offsets
#[test]
fn test_rope_with_offset() {
    let device = Device::Cpu;
    let dim = 64;
    let max_seq_len = 128;
    let seq_len = 10;

    let rope = RotaryPositionEmbedding::new(dim, max_seq_len, 10000.0, &device).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (1, 4, seq_len, dim), &device).unwrap();

    let output1 = rope.apply_rotary_embedding(&input, 0).unwrap();
    let output2 = rope.apply_rotary_embedding(&input, 10).unwrap();

    let vec1 = output1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vec2 = output2.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let diff_count = vec1
        .iter()
        .zip(vec2.iter())
        .filter(|(&a, &b)| (a - b).abs() > 1e-5)
        .count();
    assert!(
        diff_count > 0,
        "Different offsets should produce different rotations"
    );
}

/// Test RoPE with various dimensions
#[test]
fn test_rope_various_dimensions() {
    let device = Device::Cpu;
    let max_seq_len = 128;

    for dim in &[32, 64, 128, 256] {
        let rope = RotaryPositionEmbedding::new(*dim, max_seq_len, 10000.0, &device).unwrap();
        let input = Tensor::randn(0.0f32, 1.0, (1, 4, 10, *dim), &device).unwrap();
        let output = rope.apply_rotary_embedding(&input, 0).unwrap();
        assert_eq!(output.dims(), input.dims());
    }
}

/// Test SwiGLU initialization
#[test]
fn test_swiglu_initialization() {
    let device = Device::Cpu;
    let hidden_size = 256;
    let ff_dim = 512;
    let dropout = 0.1;

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let swiglu = SwiGLUFeedForward::new(hidden_size, ff_dim, dropout, vb.pp("swiglu"));
    assert!(swiglu.is_ok(), "Failed to create SwiGLU");
}

/// Test SwiGLU forward pass
#[test]
fn test_swiglu_forward() {
    let device = Device::Cpu;
    let batch_size = 2;
    let seq_len = 10;
    let hidden_size = 256;
    let ff_dim = 512;

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let swiglu = SwiGLUFeedForward::new(hidden_size, ff_dim, 0.1, vb.pp("swiglu")).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();

    let output = swiglu.forward(&input);
    assert!(output.is_ok(), "Forward pass failed");

    let output = output.unwrap();
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

/// Test SwiGLU with various hidden sizes
#[test]
fn test_swiglu_various_sizes() {
    let device = Device::Cpu;

    for hidden_size in &[128, 256, 512, 1024] {
        let ff_dim = hidden_size * 2;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let swiglu = SwiGLUFeedForward::new(*hidden_size, ff_dim, 0.1, vb.pp("swiglu")).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (2, 8, *hidden_size), &device).unwrap();
        let output = swiglu.forward(&input).unwrap();
        assert_eq!(output.dims(), input.dims());
    }
}

/// Test ALiBi initialization
#[test]
fn test_alibi_initialization() {
    let device = Device::Cpu;
    let num_heads = 8;
    let max_seq_len = 256;

    let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device);
    assert!(alibi.is_ok(), "Failed to create ALiBi");
}

/// Test ALiBi bias matrix generation
#[test]
fn test_alibi_get_bias() {
    let device = Device::Cpu;
    let num_heads = 4;
    let max_seq_len = 128;
    let seq_len = 16;

    let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();
    let bias = alibi.get_bias(seq_len);
    assert!(bias.is_ok(), "Failed to get bias");

    let bias = bias.unwrap();
    assert_eq!(bias.dims(), &[num_heads, seq_len, seq_len]);
}

/// Test ALiBi bias properties: diagonal should be zero
#[test]
fn test_alibi_diagonal_zero() {
    let device = Device::Cpu;
    let num_heads = 8;
    let max_seq_len = 128;
    let seq_len = 32;

    let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();
    let bias = alibi.get_bias(seq_len).unwrap();

    let bias_vec = bias.to_vec3::<f32>().unwrap();

    for (h, head_bias) in bias_vec.iter().enumerate().take(num_heads) {
        for (i, row) in head_bias.iter().enumerate().take(seq_len) {
            assert!(
                row[i].abs() < 1e-6,
                "Diagonal should be zero at head={}, pos={}",
                h,
                i
            );
        }
    }
}

/// Test ALiBi apply bias to attention scores
#[test]
fn test_alibi_apply_bias() {
    let device = Device::Cpu;
    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 16;
    let max_seq_len = 128;

    let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();

    let attention_scores = Tensor::zeros(
        (batch_size, num_heads, seq_len, seq_len),
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let biased_scores = alibi.apply_bias(&attention_scores);
    assert!(biased_scores.is_ok(), "Failed to apply bias");

    let biased_scores = biased_scores.unwrap();
    assert_eq!(biased_scores.dims(), attention_scores.dims());
}

/// Test ALiBi with power-of-two heads
#[test]
fn test_alibi_power_of_two_heads() {
    let device = Device::Cpu;
    let max_seq_len = 128;

    for num_heads in &[2, 4, 8, 16, 32] {
        let alibi = ALiBiPositionBias::new(*num_heads, max_seq_len, &device).unwrap();
        let bias = alibi.get_bias(16).unwrap();
        assert_eq!(bias.dims(), &[*num_heads, 16, 16]);
    }
}

/// Integration test: RoPE + MultiHeadAttention simulation
#[test]
fn test_rope_attention_integration() {
    let device = Device::Cpu;
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 16;
    let head_dim = 64;
    let max_seq_len = 128;

    let rope = RotaryPositionEmbedding::new(head_dim, max_seq_len, 10000.0, &device).unwrap();

    let query = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();
    let key = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();

    let query_rot = rope.apply_rotary_embedding(&query, 0).unwrap();
    let key_rot = rope.apply_rotary_embedding(&key, 0).unwrap();

    assert_eq!(query_rot.dims(), query.dims());
    assert_eq!(key_rot.dims(), key.dims());

    let key_t = key_rot.transpose(2, 3).unwrap();
    let scores = query_rot.matmul(&key_t).unwrap();

    assert_eq!(scores.dims(), &[batch_size, num_heads, seq_len, seq_len]);
}

/// Integration test: SwiGLU in transformer layer
#[test]
fn test_swiglu_transformer_integration() {
    let device = Device::Cpu;
    let batch_size = 4;
    let seq_len = 20;
    let hidden_size = 256;
    let ff_dim = (hidden_size * 8) / 3;

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let ffn = SwiGLUFeedForward::new(hidden_size, ff_dim, 0.1, vb.pp("ffn")).unwrap();

    let attn_output =
        Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let ffn_output = ffn.forward(&attn_output).unwrap();
    let layer_output = (&attn_output + &ffn_output).unwrap();

    assert_eq!(layer_output.dims(), &[batch_size, seq_len, hidden_size]);
}

/// Integration test: ALiBi + Attention computation
#[test]
fn test_alibi_attention_integration() {
    let device = Device::Cpu;
    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 12;
    let max_seq_len = 128;

    let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();

    let raw_scores = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, seq_len),
        &device,
    )
    .unwrap();

    let biased_scores = alibi.apply_bias(&raw_scores).unwrap();
    let attention_weights = candle_nn::ops::softmax_last_dim(&biased_scores).unwrap();

    let weights_sum = attention_weights
        .sum_keepdim(attention_weights.dims().len() - 1)
        .unwrap();
    let weights_sum_vec = weights_sum.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for &sum_val in &weights_sum_vec {
        assert!(
            (sum_val - 1.0).abs() < 1e-4,
            "Attention weights should sum to 1.0, got {}",
            sum_val
        );
    }
}
