//! Integration tests for BatchConverter.
//!
//! All tests use a CPU-only, lightweight [`VoiceConverter`] so they run
//! without GPU hardware or model files.

use std::sync::Arc;

use futures::stream;
use voirs_conversion::{
    BatchConfig, BatchConverter, ConversionConfig, ConversionRequest, ConversionTarget,
    ConversionType, VoiceCharacteristics, VoiceConverter,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a minimal, CPU-only [`VoiceConverter`] wrapped in `Arc`.
fn make_test_converter() -> Arc<VoiceConverter> {
    let config = ConversionConfig {
        use_gpu: false,
        quality_level: 0.5,
        buffer_size: 512,
        output_sample_rate: 22050,
        batch_size: 4,
        enable_async_processing: true,
        ..ConversionConfig::default()
    };
    Arc::new(VoiceConverter::with_config(config).expect("test converter should construct"))
}

/// Build a valid [`ConversionRequest`] with a 0.1-second 440 Hz sine wave
/// and `ConversionType::PassThrough` (the cheapest non-trivial path).
fn make_valid_request(id: &str) -> ConversionRequest {
    let sample_rate: u32 = 22050;
    let num_samples = (sample_rate as f64 * 0.1) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();
    let target = ConversionTarget::new(VoiceCharacteristics::default());
    ConversionRequest::new(
        id.to_string(),
        samples,
        sample_rate,
        ConversionType::PassThrough,
        target,
    )
}

/// Build a [`ConversionRequest`] that is guaranteed to fail validation
/// because `source_audio` is empty.
fn make_invalid_request(id: &str) -> ConversionRequest {
    let target = ConversionTarget::new(VoiceCharacteristics::default());
    ConversionRequest::new(
        id.to_string(),
        vec![], // empty audio — fails ConversionRequest::validate()
        22050,
        ConversionType::PassThrough,
        target,
    )
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// 1. All valid requests succeed; no failures reported.
#[tokio::test]
async fn test_batch_all_succeed() {
    let conv = make_test_converter();
    let cfg = BatchConfig {
        max_concurrency: 2,
        fail_fast: false,
        preserve_order: true,
    };
    let batch = BatchConverter::new(conv, cfg);

    let requests: Vec<ConversionRequest> = (0..4)
        .map(|i| make_valid_request(&format!("req-{i}")))
        .collect();

    let result = batch.convert_batch(requests).await;

    assert_eq!(result.successes.len(), 4, "expected 4 successes");
    assert!(result.failures.is_empty(), "expected no failures");
    assert!(result.total_duration_ms < 60_000, "batch took too long");
}

/// 2. When `preserve_order = true`, the `successes` vector is in ascending
///    index order regardless of the order completions arrived.
#[tokio::test]
async fn test_batch_preserves_order() {
    let conv = make_test_converter();
    let cfg = BatchConfig {
        max_concurrency: 3,
        fail_fast: false,
        preserve_order: true,
    };
    let batch = BatchConverter::new(conv, cfg);

    let requests: Vec<ConversionRequest> = (0..6)
        .map(|i| make_valid_request(&format!("ord-{i}")))
        .collect();

    let result = batch.convert_batch(requests).await;

    assert!(result.failures.is_empty(), "no failures expected");
    assert_eq!(result.successes.len(), 6);

    // Indices must be strictly ascending.
    let indices: Vec<usize> = result.successes.iter().map(|(i, _)| *i).collect();
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    assert_eq!(
        indices, sorted,
        "successes are not in ascending index order"
    );

    // Sanity: indices 0..6 should all appear.
    for expected_idx in 0..6_usize {
        assert!(
            indices.contains(&expected_idx),
            "index {expected_idx} missing from results"
        );
    }
}

/// 3. With `fail_fast = true`, the batch stops collecting results as soon as
///    the first failure is encountered, leaving subsequent requests unprocessed.
///
///    We submit 1 bad request followed by 9 good ones.  Because the bad request
///    arrives first and `fail_fast` is set, `failures.len()` must be ≥ 1 and
///    the total number of processed items must be less than 10.
#[tokio::test]
async fn test_batch_fail_fast_stops_early() {
    let conv = make_test_converter();
    let cfg = BatchConfig {
        max_concurrency: 1, // serial execution so the bad request is seen first
        fail_fast: true,
        preserve_order: false,
    };
    let batch = BatchConverter::new(conv, cfg);

    // First request is invalid; the rest are valid.
    let mut requests = vec![make_invalid_request("bad-0")];
    for i in 0..9 {
        requests.push(make_valid_request(&format!("ok-{i}")));
    }

    let result = batch.convert_batch(requests).await;

    assert!(
        !result.failures.is_empty(),
        "expected at least one failure with fail_fast"
    );

    let total_processed = result.successes.len() + result.failures.len();
    assert!(
        total_processed < 10,
        "fail_fast should have stopped early; processed {total_processed} items"
    );
}

/// 4. Without `fail_fast`, all requests are attempted and the counts match
///    the number of valid and invalid requests supplied.
#[tokio::test]
async fn test_batch_collects_all_errors() {
    let conv = make_test_converter();
    let cfg = BatchConfig {
        max_concurrency: 2,
        fail_fast: false,
        preserve_order: false,
    };
    let batch = BatchConverter::new(conv, cfg);

    // 2 valid, 3 invalid (interleaved).
    let requests = vec![
        make_valid_request("ok-0"),
        make_invalid_request("bad-0"),
        make_valid_request("ok-1"),
        make_invalid_request("bad-1"),
        make_invalid_request("bad-2"),
    ];

    let result = batch.convert_batch(requests).await;

    assert_eq!(
        result.successes.len(),
        2,
        "expected exactly 2 successes, got {}",
        result.successes.len()
    );
    assert_eq!(
        result.failures.len(),
        3,
        "expected exactly 3 failures, got {}",
        result.failures.len()
    );
}

/// 5. Streaming API: items are emitted in completion order; all valid requests
///    eventually appear in the stream output.
#[tokio::test]
async fn test_convert_stream_collects_all() {
    use futures::StreamExt;

    let conv = make_test_converter();
    let cfg = BatchConfig {
        max_concurrency: 2,
        fail_fast: false,
        preserve_order: false,
    };
    let batch = BatchConverter::new(conv, cfg);

    let requests: Vec<ConversionRequest> = (0..5)
        .map(|i| make_valid_request(&format!("stream-{i}")))
        .collect();

    let req_stream = stream::iter(requests);
    let output: Vec<(usize, voirs_conversion::Result<_>)> =
        batch.convert_stream(req_stream).collect().await;

    assert_eq!(output.len(), 5, "stream should emit exactly 5 items");

    let failures: Vec<_> = output.iter().filter(|(_, r)| r.is_err()).collect();
    assert!(
        failures.is_empty(),
        "stream should have no failures, got {failures:?}"
    );

    // All 5 indices 0..5 should be present.
    let mut indices: Vec<usize> = output.iter().map(|(i, _)| *i).collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}
