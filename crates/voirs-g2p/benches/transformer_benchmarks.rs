//! Performance benchmarks for transformer components
//!
//! Benchmarks the performance of transformer-based neural G2P components including:
//! - Multi-head attention mechanisms
//! - Positional encoding
//! - Layer normalization
//! - Transformer encoder/decoder layers
//! - Complete TransformerG2P model
//! - Sampling strategies

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use voirs_g2p::backends::neural::core::{
    ALiBiPositionBias, LayerNorm, MultiHeadAttention, PositionalEncoding, RotaryPositionEmbedding,
    SamplingStrategy, SwiGLUFeedForward, TransformerDecoderLayer, TransformerEncoderLayer,
    TransformerG2P,
};

/// Helper function to create a VarBuilder for benchmarking
fn create_var_builder() -> VarBuilder<'static> {
    let device = Device::Cpu;
    VarBuilder::zeros(candle_core::DType::F32, &device)
}

/// Benchmark positional encoding forward pass
fn benchmark_positional_encoding(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("positional_encoding");
    group.measurement_time(Duration::from_secs(5));

    for &seq_len in &[10, 50, 100, 256, 512] {
        let hidden_size = 512;
        let max_seq_len = 1024;
        let batch_size = 2;

        let pos_enc = PositionalEncoding::new(max_seq_len, hidden_size, &device).unwrap();
        let input = Tensor::zeros(
            &[batch_size, seq_len, hidden_size],
            candle_core::DType::F32,
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &(pos_enc, input),
            |b, (pos_enc, input)| {
                b.iter(|| {
                    let result = pos_enc.forward(black_box(input));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark layer normalization forward pass
fn benchmark_layer_norm(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("layer_norm");
    group.measurement_time(Duration::from_secs(5));

    for &hidden_size in &[256, 512, 768, 1024] {
        let batch_size = 2;
        let seq_len = 50;

        let layer_norm =
            LayerNorm::new(hidden_size, 1e-5, vb.pp(format!("ln_{}", hidden_size))).unwrap();
        let input =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", hidden_size),
            &(layer_norm, input),
            |b, (layer_norm, input)| {
                b.iter(|| {
                    let result = layer_norm.forward(black_box(input));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-head attention forward pass
fn benchmark_multi_head_attention(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("multi_head_attention");
    group.measurement_time(Duration::from_secs(10));

    for &seq_len in &[10, 50, 100, 256] {
        let hidden_size = 512;
        let num_heads = 8;
        let batch_size = 2;

        let mha =
            MultiHeadAttention::new(hidden_size, num_heads, vb.pp(format!("mha_{}", seq_len)))
                .unwrap();
        let query =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();
        let key = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();
        let value =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &(mha, query, key, value),
            |b, (mha, query, key, value)| {
                b.iter(|| {
                    let result =
                        mha.forward(black_box(query), black_box(key), black_box(value), None);
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark transformer encoder layer forward pass
fn benchmark_transformer_encoder_layer(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("transformer_encoder_layer");
    group.measurement_time(Duration::from_secs(10));

    for &seq_len in &[10, 50, 100] {
        let hidden_size = 512;
        let num_heads = 8;
        let ff_dim = 2048;
        let dropout = 0.1;
        let batch_size = 2;

        let encoder = TransformerEncoderLayer::new(
            hidden_size,
            num_heads,
            ff_dim,
            dropout,
            vb.pp(format!("enc_{}", seq_len)),
        )
        .unwrap();
        let input =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &(encoder, input),
            |b, (encoder, input)| {
                b.iter(|| {
                    let result = encoder.forward(black_box(input), None);
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark transformer decoder layer forward pass
fn benchmark_transformer_decoder_layer(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("transformer_decoder_layer");
    group.measurement_time(Duration::from_secs(10));

    for &tgt_len in &[10, 50, 100] {
        let hidden_size = 512;
        let num_heads = 8;
        let ff_dim = 2048;
        let dropout = 0.1;
        let batch_size = 2;
        let src_len = 50; // Fixed source length

        let decoder = TransformerDecoderLayer::new(
            hidden_size,
            num_heads,
            ff_dim,
            dropout,
            vb.pp(format!("dec_{}", tgt_len)),
        )
        .unwrap();
        let target =
            Tensor::randn(0.0f32, 1.0, &[batch_size, tgt_len, hidden_size], &device).unwrap();
        let encoder_output =
            Tensor::randn(0.0f32, 1.0, &[batch_size, src_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", tgt_len),
            &(decoder, target, encoder_output),
            |b, (decoder, target, encoder_output)| {
                b.iter(|| {
                    let result =
                        decoder.forward(black_box(target), black_box(encoder_output), None, None);
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complete TransformerG2P model
fn benchmark_transformer_g2p_model(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("transformer_g2p_model");
    group.measurement_time(Duration::from_secs(15));

    // Smaller model for faster benchmarking
    let grapheme_vocab_size = 50;
    let phoneme_vocab_size = 40;
    let hidden_size = 256;
    let num_heads = 4;
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let ff_dim = 1024;
    let max_seq_len = 200;
    let dropout = 0.1;

    for &seq_len in &[10, 20, 30] {
        // Create a new transformer for each benchmark to avoid ownership issues
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
            vb.pp(format!("transformer_{}", seq_len)),
        )
        .unwrap();

        let batch_size = 2;
        let grapheme_ids = Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, seq_len, grapheme_vocab_size],
            &device,
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::new("encode", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let result = transformer.encode(black_box(&grapheme_ids), None);
                black_box(result.unwrap())
            });
        });
    }

    // Separate loop for decode benchmarks
    for &seq_len in &[10, 20, 30] {
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
            vb.pp(format!("transformer_dec_{}", seq_len)),
        )
        .unwrap();

        let batch_size = 2;
        let grapheme_ids = Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, seq_len, grapheme_vocab_size],
            &device,
        )
        .unwrap();
        let phoneme_ids = Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, seq_len, phoneme_vocab_size],
            &device,
        )
        .unwrap();

        // Pre-encode for decode benchmark
        let encoder_output = transformer.encode(&grapheme_ids, None).unwrap();

        group.bench_with_input(BenchmarkId::new("decode", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let result = transformer.decode(
                    black_box(&phoneme_ids),
                    black_box(&encoder_output),
                    None,
                    None,
                );
                black_box(result.unwrap())
            });
        });
    }

    // Separate loop for full forward benchmarks
    for &seq_len in &[10, 20, 30] {
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
            vb.pp(format!("transformer_fwd_{}", seq_len)),
        )
        .unwrap();

        let batch_size = 2;
        let grapheme_ids = Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, seq_len, grapheme_vocab_size],
            &device,
        )
        .unwrap();
        let phoneme_ids = Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, seq_len, phoneme_vocab_size],
            &device,
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::new("forward", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let result = transformer.forward(
                    black_box(&grapheme_ids),
                    black_box(&phoneme_ids),
                    None,
                    None,
                    None,
                );
                black_box(result.unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark sampling strategies
fn benchmark_sampling_strategies(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("sampling_strategies");
    group.measurement_time(Duration::from_secs(5));

    for &vocab_size in &[50, 100, 1000, 10000] {
        let logits = Tensor::randn(0.0f32, 1.0, vocab_size, &device).unwrap();

        // Greedy sampling
        let greedy_strategy = SamplingStrategy::greedy();
        group.bench_with_input(
            BenchmarkId::new("greedy", vocab_size),
            &(greedy_strategy, logits.clone()),
            |b, (strategy, logits)| {
                b.iter(|| {
                    let result = strategy.sample(black_box(logits), &[]);
                    black_box(result.unwrap())
                });
            },
        );

        // Temperature sampling
        let temp_strategy = SamplingStrategy::new(0.8);
        group.bench_with_input(
            BenchmarkId::new("temperature", vocab_size),
            &(temp_strategy, logits.clone()),
            |b, (strategy, logits)| {
                b.iter(|| {
                    let result = strategy.sample(black_box(logits), &[]);
                    black_box(result.unwrap())
                });
            },
        );

        // Top-k sampling
        let topk_strategy = SamplingStrategy::new(1.0).with_top_k(50);
        group.bench_with_input(
            BenchmarkId::new("top_k", vocab_size),
            &(topk_strategy, logits.clone()),
            |b, (strategy, logits)| {
                b.iter(|| {
                    let result = strategy.sample(black_box(logits), &[]);
                    black_box(result.unwrap())
                });
            },
        );

        // Top-p (nucleus) sampling
        let topp_strategy = SamplingStrategy::new(1.0).with_top_p(0.9);
        group.bench_with_input(
            BenchmarkId::new("top_p", vocab_size),
            &(topp_strategy, logits.clone()),
            |b, (strategy, logits)| {
                b.iter(|| {
                    let result = strategy.sample(black_box(logits), &[]);
                    black_box(result.unwrap())
                });
            },
        );

        // Combined sampling with repetition penalty
        let combined_strategy = SamplingStrategy::new(0.9)
            .with_top_k(50)
            .with_top_p(0.95)
            .with_repetition_penalty(1.2);
        let previous_tokens = vec![5, 10, 15, 20]; // Some previously generated tokens
        group.bench_with_input(
            BenchmarkId::new("combined", vocab_size),
            &(combined_strategy, logits.clone(), previous_tokens),
            |b, (strategy, logits, previous_tokens)| {
                b.iter(|| {
                    let result = strategy.sample(black_box(logits), black_box(previous_tokens));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Rotary Position Embedding (RoPE)
fn benchmark_rope(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("rotary_position_embedding");
    group.measurement_time(Duration::from_secs(5));

    for &(head_dim, seq_len) in &[(64, 16), (64, 64), (128, 16), (128, 64), (64, 256)] {
        let max_seq_len = 512;
        let batch_size = 2;
        let num_heads = 8;

        let rope = RotaryPositionEmbedding::new(head_dim, max_seq_len, 10000.0, &device).unwrap();
        let input = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", format!("dim{}_len{}", head_dim, seq_len)),
            &(rope, input),
            |b, (rope, input)| {
                b.iter(|| {
                    let result = rope.apply_rotary_embedding(black_box(input), 0);
                    black_box(result.unwrap())
                });
            },
        );
    }

    // Benchmark with different offsets (cached KV scenario)
    let head_dim = 64;
    let seq_len = 16;
    let max_seq_len = 512;
    let batch_size = 2;
    let num_heads = 8;

    let rope = RotaryPositionEmbedding::new(head_dim, max_seq_len, 10000.0, &device).unwrap();
    let input = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();

    for &offset in &[0, 16, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::new("offset", offset), &offset, |b, offset| {
            b.iter(|| {
                let result = rope.apply_rotary_embedding(black_box(&input), *offset);
                black_box(result.unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark SwiGLU feedforward network
fn benchmark_swiglu(c: &mut Criterion) {
    let vb = create_var_builder();
    let device = Device::Cpu;
    let mut group = c.benchmark_group("swiglu_feedforward");
    group.measurement_time(Duration::from_secs(5));

    for &(hidden_size, ff_dim) in &[
        (256, 512),
        (256, 683), // 256 * 8/3 (SwiGLU recommendation)
        (512, 1024),
        (512, 1365), // 512 * 8/3
        (768, 2048),
    ] {
        let batch_size = 2;
        let seq_len = 50;

        let swiglu = SwiGLUFeedForward::new(
            hidden_size,
            ff_dim,
            0.1,
            vb.pp(format!("swiglu_{}_{}", hidden_size, ff_dim)),
        )
        .unwrap();
        let input =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", format!("h{}_ff{}", hidden_size, ff_dim)),
            &(swiglu, input),
            |b, (swiglu, input)| {
                b.iter(|| {
                    let result = swiglu.forward(black_box(input));
                    black_box(result.unwrap())
                });
            },
        );
    }

    // Benchmark comparison with different batch sizes and sequence lengths
    let hidden_size = 512;
    let ff_dim = 1365; // SwiGLU recommended

    for &(batch_size, seq_len) in &[(1, 10), (2, 50), (4, 100), (8, 200)] {
        let swiglu = SwiGLUFeedForward::new(
            hidden_size,
            ff_dim,
            0.1,
            vb.pp(format!("swiglu_batch{}_seq{}", batch_size, seq_len)),
        )
        .unwrap();
        let input =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("batch_seq", format!("b{}_s{}", batch_size, seq_len)),
            &(swiglu, input),
            |b, (swiglu, input)| {
                b.iter(|| {
                    let result = swiglu.forward(black_box(input));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ALiBi position bias
fn benchmark_alibi(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("alibi_position_bias");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark bias generation for different configurations
    for &(num_heads, seq_len) in &[(4, 16), (8, 32), (16, 64), (32, 128), (8, 256)] {
        let max_seq_len = 512;

        let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("get_bias", format!("h{}_l{}", num_heads, seq_len)),
            &(alibi, seq_len),
            |b, (alibi, seq_len)| {
                b.iter(|| {
                    let result = alibi.get_bias(black_box(*seq_len));
                    black_box(result.unwrap())
                });
            },
        );
    }

    // Benchmark apply_bias with attention scores
    for &(num_heads, seq_len) in &[(4, 16), (8, 64), (16, 128)] {
        let max_seq_len = 512;
        let batch_size = 2;

        let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();
        let attention_scores = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, seq_len),
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("apply_bias", format!("h{}_l{}", num_heads, seq_len)),
            &(alibi, attention_scores),
            |b, (alibi, attention_scores)| {
                b.iter(|| {
                    let result = alibi.apply_bias(black_box(attention_scores));
                    black_box(result.unwrap())
                });
            },
        );
    }

    // Benchmark with varying number of heads (power of 2)
    let seq_len = 64;
    let max_seq_len = 512;
    let batch_size = 2;

    for &num_heads in &[2, 4, 8, 16, 32, 64] {
        let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();
        let attention_scores = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, seq_len),
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("heads_scaling", num_heads),
            &(alibi, attention_scores),
            |b, (alibi, attention_scores)| {
                b.iter(|| {
                    let result = alibi.apply_bias(black_box(attention_scores));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparison: Traditional positional encoding vs RoPE vs ALiBi
fn benchmark_position_encoding_comparison(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("position_encoding_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &seq_len in &[16, 64, 128, 256] {
        let hidden_size = 512;
        let head_dim = 64;
        let num_heads = 8;
        let max_seq_len = 512;
        let batch_size = 2;

        // Traditional positional encoding
        let pos_enc = PositionalEncoding::new(max_seq_len, hidden_size, &device).unwrap();
        let input_trad =
            Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("traditional", seq_len),
            &(pos_enc, input_trad),
            |b, (pos_enc, input)| {
                b.iter(|| {
                    let result = pos_enc.forward(black_box(input));
                    black_box(result.unwrap())
                });
            },
        );

        // RoPE
        let rope = RotaryPositionEmbedding::new(head_dim, max_seq_len, 10000.0, &device).unwrap();
        let input_rope = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("rope", seq_len),
            &(rope, input_rope),
            |b, (rope, input)| {
                b.iter(|| {
                    let result = rope.apply_rotary_embedding(black_box(input), 0);
                    black_box(result.unwrap())
                });
            },
        );

        // ALiBi
        let alibi = ALiBiPositionBias::new(num_heads, max_seq_len, &device).unwrap();
        let attention_scores = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, seq_len),
            &device,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("alibi", seq_len),
            &(alibi, attention_scores),
            |b, (alibi, attention_scores)| {
                b.iter(|| {
                    let result = alibi.apply_bias(black_box(attention_scores));
                    black_box(result.unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_positional_encoding,
    benchmark_layer_norm,
    benchmark_multi_head_attention,
    benchmark_transformer_encoder_layer,
    benchmark_transformer_decoder_layer,
    benchmark_transformer_g2p_model,
    benchmark_sampling_strategies,
    benchmark_rope,
    benchmark_swiglu,
    benchmark_alibi,
    benchmark_position_encoding_comparison,
);

criterion_main!(benches);
