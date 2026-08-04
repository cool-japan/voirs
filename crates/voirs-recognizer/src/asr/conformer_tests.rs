//! Tests for the Conformer ASR model.
//!
//! These verify the two properties the implementation must hold: the forward passes
//! really consume the stored parameters (so the output changes when the parameters or
//! the input change), and no transcript is produced from untrained parameters.

use super::*;
use crate::traits::ASRFeature;
use std::io::Write;
use std::path::{Path, PathBuf};
use voirs_sdk::AudioBuffer;

/// A deliberately tiny configuration so the real O(n²·d) attention and the O(d·h)
/// feed-forward stay fast in a unit test.
fn tiny_config() -> ConformerConfig {
    ConformerConfig {
        num_blocks: 1,
        encoder_dim: 8,
        attention_heads: 2,
        feed_forward_dim: 16,
        conv_kernel_size: 3,
        dropout_rate: 0.0,
        input_dim: 4,
        vocab_size: 6,
        max_seq_length: 512,
        use_relative_positional_encoding: false,
        macaron_style: true,
        conv_activation: ActivationType::Swish,
    }
}

/// Every tensor the tiny configuration requires, with its exact shape.
fn tiny_layout(config: &ConformerConfig) -> Vec<(String, Vec<usize>)> {
    let dim = config.encoder_dim;
    let hidden = config.feed_forward_dim;
    let mut layout = vec![
        (
            "input_projection.weight".to_string(),
            vec![dim, config.input_dim],
        ),
        (
            "output_projection.weight".to_string(),
            vec![config.vocab_size, dim],
        ),
    ];

    for index in 0..config.num_blocks {
        let p = format!("blocks.{index}");
        layout.extend([
            (format!("{p}.attention.query.weight"), vec![dim, dim]),
            (format!("{p}.attention.key.weight"), vec![dim, dim]),
            (format!("{p}.attention.value.weight"), vec![dim, dim]),
            (format!("{p}.attention.output.weight"), vec![dim, dim]),
            (format!("{p}.conv.pointwise1.weight"), vec![dim * 2, dim]),
            (
                format!("{p}.conv.depthwise.weight"),
                vec![dim, config.conv_kernel_size],
            ),
            (format!("{p}.conv.pointwise2.weight"), vec![dim, dim]),
            (format!("{p}.conv.norm.weight"), vec![dim]),
            (format!("{p}.conv.norm.bias"), vec![dim]),
        ]);
        for slot in ["ff1", "ff2"] {
            layout.extend([
                (format!("{p}.{slot}.linear1.weight"), vec![hidden, dim]),
                (format!("{p}.{slot}.linear1.bias"), vec![hidden]),
                (format!("{p}.{slot}.linear2.weight"), vec![dim, hidden]),
                (format!("{p}.{slot}.linear2.bias"), vec![dim]),
            ]);
        }
        for norm in ["layer_norm1", "layer_norm2", "layer_norm3", "layer_norm4"] {
            layout.extend([
                (format!("{p}.{norm}.weight"), vec![dim]),
                (format!("{p}.{norm}.bias"), vec![dim]),
            ]);
        }
    }

    layout
}

/// Write a real safetensors checkpoint whose values are a deterministic function of
/// `seed`, so two different seeds give two genuinely different models.
fn write_checkpoint(dir: &Path, config: &ConformerConfig, seed: u32) -> PathBuf {
    let layout = tiny_layout(config);

    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    let mut counter = seed;

    for (name, shape) in &layout {
        let elements: usize = shape.iter().product();
        let start = payload.len() as u64;
        for _ in 0..elements {
            // Small deterministic pseudo-random values in roughly [-0.5, 0.5].
            counter = counter.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = f32::from(((counter >> 16) & 0x3FF) as u16) / 1024.0 - 0.5;
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let end = payload.len() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [start, end] }),
        );
    }

    let header_bytes = serde_json::to_vec(&header).unwrap();
    let path = dir.join(format!("conformer-{seed}.safetensors"));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload).unwrap();
    file.flush().unwrap();
    path
}

/// A 1-second 16 kHz sine at `freq` Hz.
fn tone(freq: f32, seconds: f32) -> AudioBuffer {
    let count = (16_000.0 * seconds) as usize;
    let samples: Vec<f32> = (0..count)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 16_000.0).sin() * 0.5)
        .collect();
    AudioBuffer::new(samples, 16_000, 1)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_conformer_creation() {
    let model = ConformerModel::new().await.unwrap();
    assert_eq!(model.config.num_blocks, 16);
    assert_eq!(model.config.encoder_dim, 512);
    assert_eq!(model.config.attention_heads, 8);
    assert_eq!(model.weight_source(), &ConformerWeightSource::RandomInit);
}

#[tokio::test]
async fn test_conformer_config() {
    let config = ConformerConfig {
        num_blocks: 12,
        encoder_dim: 256,
        attention_heads: 4,
        ..Default::default()
    };

    let model = ConformerModel::with_config(config).await.unwrap();
    assert_eq!(model.config.num_blocks, 12);
    assert_eq!(model.config.encoder_dim, 256);
    assert_eq!(model.config.attention_heads, 4);
}

#[tokio::test]
async fn test_conformer_supported_languages() {
    let model = ConformerModel::with_config(tiny_config()).await.unwrap();
    let languages = model.supported_languages();

    assert!(!languages.is_empty());
    assert!(languages.contains(&LanguageCode::EnUs));
    assert!(languages.contains(&LanguageCode::JaJp));
    assert!(languages.contains(&LanguageCode::ZhCn));
}

// ---------------------------------------------------------------------------
// Fail-closed behaviour on untrained parameters
// ---------------------------------------------------------------------------

/// Regression test: an untrained Conformer must refuse to transcribe instead of
/// decoding noise into a plausible-looking transcript.
#[tokio::test]
async fn test_untrained_conformer_refuses_to_transcribe() {
    let model = ConformerModel::with_config(tiny_config()).await.unwrap();
    let audio = tone(440.0, 0.2);

    let err = match model.transcribe(&audio, None).await {
        Ok(t) => panic!("untrained Conformer produced a transcript: {t:?}"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("randomly initialised"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_untrained_conformer_refuses_to_stream() {
    use futures::stream;

    let model = ConformerModel::with_config(tiny_config()).await.unwrap();
    let audio_stream: AudioStream = Box::pin(stream::iter(vec![tone(440.0, 0.2)]));

    assert!(model
        .transcribe_streaming(audio_stream, None)
        .await
        .is_err());
}

/// An untrained model must not advertise capabilities it refuses to perform.
#[tokio::test]
async fn test_untrained_conformer_advertises_nothing() {
    let model = ConformerModel::with_config(tiny_config()).await.unwrap();

    assert!(model.metadata().supported_features.is_empty());
    assert!(!model.supports_feature(ASRFeature::StreamingInference));
    assert!(!model.supports_feature(ASRFeature::WordTimestamps));
    assert!(!model.supports_feature(ASRFeature::LanguageDetection));
}

// ---------------------------------------------------------------------------
// Honest metadata
// ---------------------------------------------------------------------------

/// Regression test for the fabricated `wer_benchmarks: 0.05`, `model_size_mb: 512.0`
/// and `inference_speed: 1.5` constants.
#[tokio::test]
async fn test_conformer_metadata_is_derived_not_asserted() {
    let model = ConformerModel::with_config(tiny_config()).await.unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.name, "Conformer");
    assert_eq!(metadata.architecture, "Conformer");
    assert!(
        metadata.wer_benchmarks.is_empty(),
        "WER must not be fabricated"
    );
    assert_eq!(metadata.inference_speed, 0.0, "speed must not be asserted");

    // Size must equal the real parameter count times four bytes.
    let expected = (model.parameter_count() * 4) as f32 / (1024.0 * 1024.0);
    assert!((metadata.model_size_mb - expected).abs() < f32::EPSILON);
    assert!(metadata.model_size_mb > 0.0);

    // A bigger architecture must report a bigger size.
    let bigger = ConformerModel::with_config(ConformerConfig {
        encoder_dim: 16,
        ..tiny_config()
    })
    .await
    .unwrap();
    assert!(bigger.metadata().model_size_mb > metadata.model_size_mb);
}

// ---------------------------------------------------------------------------
// Real weight loading
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_from_checkpoint_loads_real_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let path = write_checkpoint(dir.path(), &config, 7);

    let model = ConformerModel::from_checkpoint(&path, config.clone())
        .await
        .unwrap();

    match model.weight_source() {
        ConformerWeightSource::Checkpoint {
            path: loaded,
            parameter_count,
        } => {
            assert_eq!(loaded, &path);
            assert_eq!(*parameter_count, model.parameter_count());
        }
        other => panic!("expected Checkpoint, got {other:?}"),
    }

    // The loaded values must be the file's values, not the random initialisation.
    let expected: usize = tiny_layout(&config)
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum();
    assert_eq!(model.parameter_count(), expected);
}

#[tokio::test]
async fn test_from_checkpoint_rejects_shape_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_checkpoint(dir.path(), &tiny_config(), 1);

    // Same file, different architecture: every shape check must fail.
    let wrong = ConformerConfig {
        encoder_dim: 16,
        ..tiny_config()
    };
    let err = match ConformerModel::from_checkpoint(&path, wrong).await {
        Ok(_) => panic!("a mismatched checkpoint must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("expected"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_from_checkpoint_rejects_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        ConformerModel::from_checkpoint(dir.path().join("absent.safetensors"), tiny_config())
            .await
            .is_err()
    );
}

/// A trained model transcribes; its metadata names the checkpoint it came from.
#[tokio::test]
async fn test_trained_conformer_transcribes() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let path = write_checkpoint(dir.path(), &config, 11);
    let model = ConformerModel::from_checkpoint(&path, config)
        .await
        .unwrap();

    let result = model.transcribe(&tone(440.0, 0.2), None).await.unwrap();
    assert!(result.processing_duration.is_some());
    assert!((0.0..=1.0).contains(&result.confidence));
    assert!(
        model.metadata().description.contains("loaded from"),
        "description should name the checkpoint"
    );
    assert!(model.supports_feature(ASRFeature::StreamingInference));
}

// ---------------------------------------------------------------------------
// The forward passes really use the parameters
// ---------------------------------------------------------------------------

/// The core regression test for the old `x * 0.95` / `x * scale * 0.98` placeholders:
/// two models that differ **only** in their parameters must produce different encoder
/// output for identical input.
#[tokio::test]
async fn test_forward_output_depends_on_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let a =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 3), config.clone())
            .await
            .unwrap();
    let b =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 4), config.clone())
            .await
            .unwrap();

    let features = vec![vec![0.3_f32, -0.1, 0.7, 0.2]; 6];
    let out_a = a.forward(features.clone()).await.unwrap();
    let out_b = b.forward(features).await.unwrap();

    assert_eq!(out_a.len(), out_b.len());
    let differs = out_a
        .iter()
        .zip(out_b.iter())
        .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    assert!(differs, "forward pass ignored the model parameters");
}

/// The forward pass must also depend on the input, and must not be a per-element
/// rescale of it (which is what the placeholders effectively were).
#[tokio::test]
async fn test_forward_output_depends_on_input() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let model =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 5), config.clone())
            .await
            .unwrap();

    let first = vec![vec![0.3_f32, -0.1, 0.7, 0.2]; 6];
    let mut second = first.clone();
    second[2] = vec![-0.9, 0.4, 0.1, -0.6];

    let out_first = model.forward(first).await.unwrap();
    let out_second = model.forward(second).await.unwrap();

    let differs = out_first
        .iter()
        .zip(out_second.iter())
        .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    assert!(differs, "forward pass ignored the input");
}

/// Attention must mix information across time: changing one frame must change the
/// output at other frames. A per-element scalar transform cannot do this.
#[tokio::test]
async fn test_attention_mixes_across_time() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let model =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 9), config.clone())
            .await
            .unwrap();

    let block = &model.blocks[0];
    let base = vec![vec![0.2_f32; config.encoder_dim]; 5];
    let mut perturbed = base.clone();
    perturbed[4] = vec![1.5_f32; config.encoder_dim];

    let out_base = model
        .apply_multi_head_attention(&base, &block.attention)
        .await
        .unwrap();
    let out_perturbed = model
        .apply_multi_head_attention(&perturbed, &block.attention)
        .await
        .unwrap();

    // Frame 0 must react to a change made only at frame 4.
    let frame0_changed = out_base[0]
        .iter()
        .zip(out_perturbed[0].iter())
        .any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(frame0_changed, "attention did not mix across time steps");
}

/// The feed-forward network must apply its own matrices and biases, and must scale
/// the macaron half-step by exactly one half.
#[tokio::test]
async fn test_feed_forward_uses_weights_and_scale() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let model =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 13), config.clone())
            .await
            .unwrap();

    let ff = &model.blocks[0].feed_forward_1;
    let input = vec![vec![0.4_f32; config.encoder_dim]; 3];

    let full = model.apply_feed_forward(&input, ff, 1.0).await.unwrap();
    let half = model.apply_feed_forward(&input, ff, 0.5).await.unwrap();

    for (row_full, row_half) in full.iter().zip(half.iter()) {
        for (a, b) in row_full.iter().zip(row_half.iter()) {
            assert!(
                (a * 0.5 - b).abs() < 1e-5,
                "scale was not applied: {a} vs {b}"
            );
        }
    }

    // The result must not be a rescale of the input (the old placeholder behaviour).
    let is_rescale = full.iter().zip(input.iter()).all(|(o, i)| {
        o.iter()
            .zip(i.iter())
            .all(|(x, y)| (x - y * 0.98).abs() < 1e-5)
    });
    assert!(!is_rescale, "feed-forward is still a scalar rescale");
}

/// The convolution module must use its own depthwise kernel, so a change to the
/// kernel changes the output.
#[tokio::test]
async fn test_convolution_uses_its_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_config();
    let mut model =
        ConformerModel::from_checkpoint(write_checkpoint(dir.path(), &config, 17), config.clone())
            .await
            .unwrap();

    let input: Vec<Vec<f32>> = (0..6)
        .map(|t| {
            (0..config.encoder_dim)
                .map(|c| (t + c) as f32 * 0.05)
                .collect()
        })
        .collect();

    let before = model
        .apply_convolution_module(&input, &model.blocks[0].convolution)
        .await
        .unwrap();

    // Flip the sign of one channel's kernel.
    for tap in &mut model.blocks[0].convolution.depthwise_conv_weights[0] {
        *tap = -*tap - 1.0;
    }

    let after = model
        .apply_convolution_module(&input, &model.blocks[0].convolution)
        .await
        .unwrap();

    let differs = before
        .iter()
        .zip(after.iter())
        .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    assert!(differs, "convolution ignored its depthwise kernel");
}

/// The depthwise kernel must be allocated with the configured number of taps.
#[tokio::test]
async fn test_depthwise_kernel_matches_configured_size() {
    for kernel_size in [3_usize, 7, 15] {
        let config = ConformerConfig {
            conv_kernel_size: kernel_size,
            ..tiny_config()
        };
        let model = ConformerModel::with_config(config).await.unwrap();
        for kernel in &model.blocks[0].convolution.depthwise_conv_weights {
            assert_eq!(kernel.len(), kernel_size);
        }
    }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// The log-mel front end must be a real spectral transform: a low tone and a high
/// tone must light up different mel bands.
#[tokio::test]
async fn test_mel_spectrogram_is_frequency_selective() {
    let config = ConformerConfig {
        input_dim: 16,
        ..tiny_config()
    };
    let model = ConformerModel::with_config(config).await.unwrap();

    let low = model.extract_features(&tone(300.0, 0.2)).await.unwrap();
    let high = model.extract_features(&tone(6000.0, 0.2)).await.unwrap();

    assert!(!low.is_empty() && !high.is_empty());

    let peak = |frames: &Vec<Vec<f32>>| -> usize {
        let n_mels = frames[0].len();
        let mut sums = vec![0.0_f32; n_mels];
        for frame in frames {
            for (bin, value) in frame.iter().enumerate() {
                sums[bin] += *value;
            }
        }
        sums.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    };

    let low_peak = peak(&low);
    let high_peak = peak(&high);
    assert!(
        low_peak < high_peak,
        "a 300 Hz tone peaked at mel bin {low_peak} and a 6 kHz tone at {high_peak}"
    );
}

#[tokio::test]
async fn test_mel_filterbank_is_normalised_and_ordered() {
    let filters = mel_filterbank(8, 65, 16_000.0);
    assert_eq!(filters.len(), 8);

    // Each triangle must have some support and a peak of at most 1.
    for filter in &filters {
        assert_eq!(filter.len(), 65);
        let peak = filter.iter().copied().fold(0.0_f32, f32::max);
        assert!(peak > 0.0 && peak <= 1.0 + 1e-6, "bad peak {peak}");
    }

    // Centres must increase monotonically with the filter index.
    let centre = |filter: &Vec<f32>| {
        filter
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    };
    for pair in filters.windows(2) {
        assert!(centre(&pair[0]) <= centre(&pair[1]));
    }
}

#[test]
fn test_softmax_is_a_distribution() {
    let mut scores = vec![1.0_f32, 2.0, 3.0];
    softmax_in_place(&mut scores);
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(scores[2] > scores[1] && scores[1] > scores[0]);

    // Large values must not overflow to NaN.
    let mut extreme = vec![1e30_f32, 1.0];
    softmax_in_place(&mut extreme);
    assert!(extreme.iter().all(|v| v.is_finite()));
    assert!((extreme.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_matmul_rows_rejects_shape_mismatch() {
    let input = vec![vec![1.0_f32, 2.0, 3.0]];
    let weights = vec![vec![1.0_f32, 2.0]]; // wrong column count
    assert!(matmul_rows(&input, &weights).is_err());

    let good = vec![vec![1.0_f32, 0.0, -1.0], vec![0.0, 1.0, 0.0]];
    let out = matmul_rows(&input, &good).unwrap();
    assert_eq!(out, vec![vec![-2.0, 2.0]]);
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_factory_functions() {
    let model1 = create_conformer_asr().await;
    assert!(model1.is_ok());

    let config = tiny_config();
    let model2 = create_conformer_asr_with_config(config).await;
    assert!(model2.is_ok());
}
