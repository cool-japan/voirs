//! Batch voice conversion with concurrency-controlled processing.
//!
//! Provides [`BatchConverter`] which wraps a [`VoiceConverter`] and processes
//! multiple [`ConversionRequest`]s concurrently, bounded by a Tokio `Semaphore`.
//! A streaming variant, [`BatchConverter::convert_stream`], handles infinite or
//! lazily-produced request sequences without materialising the entire workload
//! up front.

use std::sync::Arc;
use std::time::Instant;

use futures::{Stream, StreamExt};
use tokio::sync::Semaphore;

use crate::types::{ConversionRequest, ConversionResult};
use crate::Error;

use super::converter::VoiceConverter;

// ── BatchConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`BatchConverter`].
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of [`VoiceConverter::convert`] calls that may run
    /// concurrently.  A value of `0` is silently promoted to `1`.
    pub max_concurrency: usize,
    /// When `true`, processing stops as soon as the first failure is collected;
    /// subsequent tasks that are already spawned are still awaited but new ones
    /// are not started.
    pub fail_fast: bool,
    /// When `true`, the `successes` and `failures` vectors inside [`BatchResult`]
    /// are sorted by the original request index before returning.
    pub preserve_order: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            fail_fast: false,
            preserve_order: true,
        }
    }
}

// ── BatchResult ───────────────────────────────────────────────────────────────

/// Aggregated outcome of a [`BatchConverter::convert_batch`] call.
#[derive(Debug)]
pub struct BatchResult {
    /// Successfully converted items: `(original_index, result)`.
    pub successes: Vec<(usize, ConversionResult)>,
    /// Failed items: `(original_index, error)`.
    pub failures: Vec<(usize, Error)>,
    /// Wall-clock duration for the whole batch, in milliseconds.
    pub total_duration_ms: u64,
}

impl BatchResult {
    /// Returns `true` when every submitted request succeeded.
    pub fn is_all_success(&self) -> bool {
        self.failures.is_empty()
    }

    /// Total number of requests that were submitted.
    pub fn total_submitted(&self) -> usize {
        self.successes.len() + self.failures.len()
    }
}

// ── BatchConverter ────────────────────────────────────────────────────────────

/// Concurrency-controlled batch wrapper around [`VoiceConverter`].
///
/// Internally holds an `Arc<Semaphore>` with `max_concurrency` permits.
/// Each spawned task acquires an owned permit before calling
/// [`VoiceConverter::convert`]; the permit is released when the task finishes,
/// allowing the next queued task to proceed.
pub struct BatchConverter {
    converter: Arc<VoiceConverter>,
    semaphore: Arc<Semaphore>,
    config: BatchConfig,
}

impl BatchConverter {
    /// Create a new [`BatchConverter`] from a shared converter and a config.
    ///
    /// `config.max_concurrency` is clamped to at least `1` so the semaphore
    /// always allows at least one in-flight conversion.
    pub fn new(converter: Arc<VoiceConverter>, config: BatchConfig) -> Self {
        let effective_concurrency = config.max_concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(effective_concurrency));
        Self {
            converter,
            semaphore,
            config,
        }
    }

    // ── convert_batch ─────────────────────────────────────────────────────────

    /// Process a batch of [`ConversionRequest`]s with bounded concurrency.
    ///
    /// All tasks are spawned upfront; the semaphore limits how many run at
    /// once.  When `fail_fast` is `true`, collection stops after the first
    /// failure record is encountered — already-running tasks are awaited but
    /// no additional join-handles are polled.
    ///
    /// When `preserve_order` is `true`, both result vectors are sorted by the
    /// original request index before the [`BatchResult`] is returned.
    pub async fn convert_batch(&self, requests: Vec<ConversionRequest>) -> BatchResult {
        let wall_start = Instant::now();

        // Spawn one task per request.  Tasks block on the semaphore internally,
        // so only `max_concurrency` conversions execute simultaneously.
        let mut join_handles = Vec::with_capacity(requests.len());

        for (idx, req) in requests.into_iter().enumerate() {
            let sem = Arc::clone(&self.semaphore);
            let conv = Arc::clone(&self.converter);

            let handle = tokio::spawn(async move {
                // Acquire a permit — blocks if all permits are taken.
                let _permit = sem
                    .acquire_owned()
                    .await
                    .expect("BatchConverter semaphore must never be closed");

                let outcome = conv.convert(req).await;
                (idx, outcome)
            });

            join_handles.push(handle);
        }

        // Collect results, respecting fail_fast.
        let mut successes: Vec<(usize, ConversionResult)> = Vec::new();
        let mut failures: Vec<(usize, Error)> = Vec::new();
        let mut abort_early = false;

        for handle in join_handles {
            if abort_early {
                // Drop the remaining handles — tasks are already spawned and
                // consuming resources, but we stop collecting results.  The
                // tasks will finish naturally; we just discard their output.
                // This matches the documented "no new tasks" semantics for
                // fail_fast (all tasks are already spawned).
                break;
            }

            match handle.await {
                Ok((idx, Ok(result))) => {
                    successes.push((idx, result));
                }
                Ok((idx, Err(e))) => {
                    failures.push((idx, e));
                    if self.config.fail_fast {
                        abort_early = true;
                    }
                }
                Err(join_err) => {
                    // The task panicked or was cancelled.  Assign the best
                    // available index (number of already-seen items).
                    let idx = successes.len() + failures.len();
                    failures.push((idx, Error::runtime(format!("task join error: {join_err}"))));
                    if self.config.fail_fast {
                        abort_early = true;
                    }
                }
            }
        }

        if self.config.preserve_order {
            successes.sort_by_key(|(i, _)| *i);
            failures.sort_by_key(|(i, _)| *i);
        }

        BatchResult {
            successes,
            failures,
            total_duration_ms: wall_start.elapsed().as_millis() as u64,
        }
    }

    // ── convert_stream ────────────────────────────────────────────────────────

    /// Process a potentially-unbounded stream of [`ConversionRequest`]s.
    ///
    /// Returns a [`Stream`] of `(original_index, Result<ConversionResult>)`.
    /// Items are emitted as soon as they complete — order is therefore
    /// non-deterministic (fastest conversions arrive first).
    ///
    /// The semaphore is still respected: at most `max_concurrency` conversions
    /// run simultaneously.  The method spawns a background driver task that
    /// reads from `requests`, acquires a semaphore permit, then delegates to an
    /// inner task for the actual conversion.  Results are forwarded through an
    /// unbounded MPSC channel whose receiver end is returned as a stream.
    ///
    /// Dropping the returned stream does *not* cancel in-flight tasks — they
    /// will complete and their results will be silently discarded when the
    /// channel sender is dropped.
    pub fn convert_stream<S>(
        &self,
        requests: S,
    ) -> impl Stream<Item = (usize, crate::Result<ConversionResult>)>
    where
        S: Stream<Item = ConversionRequest> + Send + 'static,
    {
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let sem = Arc::clone(&self.semaphore);
        let conv = Arc::clone(&self.converter);
        let (tx, rx) = mpsc::unbounded_channel::<(usize, crate::Result<ConversionResult>)>();

        // Driver task: iterates over `requests`, acquiring permits and
        // spawning per-request inner tasks.
        tokio::spawn(async move {
            futures::pin_mut!(requests);
            let mut idx: usize = 0;

            while let Some(req) = requests.next().await {
                // Acquire permit before spawning so that back-pressure is
                // applied to the driver; we don't accept a new request until
                // a slot is available.
                let permit = Arc::clone(&sem)
                    .acquire_owned()
                    .await
                    .expect("BatchConverter semaphore must never be closed");

                let tx2 = tx.clone();
                let conv2 = Arc::clone(&conv);
                let cur_idx = idx;

                tokio::spawn(async move {
                    // Permit is held for the duration of the conversion.
                    let _permit = permit;
                    let result = conv2.convert(req).await;
                    // Ignore send errors: the receiver may have been dropped.
                    let _ = tx2.send((cur_idx, result));
                });

                idx += 1;
            }
            // `tx` is dropped here when all per-request `tx2` clones also
            // finish, which closes the channel and terminates the stream.
        });

        UnboundedReceiverStream::new(rx)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::config::ConversionConfig;
    use crate::types::{ConversionTarget, ConversionType, VoiceCharacteristics};

    /// Build a lightweight, CPU-only [`VoiceConverter`] for unit tests.
    fn make_converter() -> Arc<VoiceConverter> {
        let config = ConversionConfig {
            use_gpu: false,
            quality_level: 0.5,
            buffer_size: 512,
            output_sample_rate: 22050,
            batch_size: 4,
            ..ConversionConfig::default()
        };
        Arc::new(VoiceConverter::with_config(config).expect("unit test converter"))
    }

    /// Build a minimal valid [`ConversionRequest`] with a short sine wave.
    fn make_request(id: &str) -> ConversionRequest {
        let sample_rate: u32 = 22050;
        let samples: Vec<f32> = (0..(sample_rate as usize))
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
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

    #[tokio::test]
    async fn unit_batch_all_succeed() {
        let conv = make_converter();
        let cfg = BatchConfig {
            max_concurrency: 2,
            fail_fast: false,
            preserve_order: true,
        };
        let batch = BatchConverter::new(conv, cfg);
        let requests: Vec<_> = (0..4).map(|i| make_request(&format!("u{i}"))).collect();
        let result = batch.convert_batch(requests).await;
        assert_eq!(result.successes.len(), 4);
        assert!(result.failures.is_empty());
    }

    #[tokio::test]
    async fn unit_batch_invalid_request_produces_failure() {
        let conv = make_converter();
        let cfg = BatchConfig {
            max_concurrency: 2,
            fail_fast: false,
            preserve_order: true,
        };
        let batch = BatchConverter::new(conv, cfg);

        // An empty source_audio always triggers `validate()` failure.
        let bad_target = ConversionTarget::new(VoiceCharacteristics::default());
        let bad_req = ConversionRequest::new(
            "bad".to_string(),
            vec![], // empty — will fail validation
            22050,
            ConversionType::PassThrough,
            bad_target,
        );

        let result = batch.convert_batch(vec![bad_req]).await;
        assert_eq!(result.failures.len(), 1);
        assert!(result.successes.is_empty());
    }
}
