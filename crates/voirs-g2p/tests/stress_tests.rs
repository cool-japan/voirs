use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use voirs_g2p::rules::EnglishRuleG2p;
use voirs_g2p::{DummyG2p, G2p, LanguageCode};

#[derive(Debug)]
struct StressTestResults {
    total_operations: usize,
    successful_operations: usize,
    failed_operations: usize,
    total_duration: Duration,
    average_latency: Duration,
    max_latency: Duration,
    min_latency: Duration,
    operations_per_second: f64,
}

impl StressTestResults {
    fn new(
        total_operations: usize,
        successful_operations: usize,
        failed_operations: usize,
        total_duration: Duration,
        latencies: Vec<Duration>,
    ) -> Self {
        let average_latency = if !latencies.is_empty() {
            Duration::from_nanos(
                (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / latencies.len() as u128)
                    .try_into()
                    .unwrap_or(0),
            )
        } else {
            Duration::ZERO
        };

        let max_latency = latencies.iter().max().copied().unwrap_or(Duration::ZERO);
        let min_latency = latencies.iter().min().copied().unwrap_or(Duration::ZERO);

        let operations_per_second = if total_duration.as_secs_f64() > 0.0 {
            successful_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            total_operations,
            successful_operations,
            failed_operations,
            total_duration,
            average_latency,
            max_latency,
            min_latency,
            operations_per_second,
        }
    }
}

async fn run_concurrent_stress_test<G>(
    g2p: Arc<G>,
    concurrent_requests: usize,
    requests_per_worker: usize,
    test_input: &str,
) -> StressTestResults
where
    G: G2p + Send + Sync + 'static,
{
    let mut join_set = JoinSet::new();
    let start_time = Instant::now();
    let test_input = test_input.to_string();

    // Spawn concurrent workers
    for worker_id in 0..concurrent_requests {
        let g2p_clone = Arc::clone(&g2p);
        let input_clone = test_input.clone();

        join_set.spawn(async move {
            let mut latencies = Vec::new();
            let mut successful = 0;
            let mut failed = 0;

            for request_id in 0..requests_per_worker {
                let request_start = Instant::now();

                match g2p_clone
                    .to_phonemes(&input_clone, Some(LanguageCode::EnUs))
                    .await
                {
                    Ok(_) => {
                        successful += 1;
                        latencies.push(request_start.elapsed());
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("Worker {worker_id} request {request_id} failed: {e:?}");
                    }
                }

                // Small delay to avoid overwhelming the system
                tokio::time::sleep(Duration::from_millis(1)).await;
            }

            (successful, failed, latencies)
        });
    }

    // Collect results from all workers
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut all_latencies = Vec::new();

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((successful, failed, latencies)) => {
                total_successful += successful;
                total_failed += failed;
                all_latencies.extend(latencies);
            }
            Err(e) => {
                eprintln!("Worker task failed: {e:?}");
                total_failed += requests_per_worker;
            }
        }
    }

    let total_duration = start_time.elapsed();
    let total_operations = concurrent_requests * requests_per_worker;

    StressTestResults::new(
        total_operations,
        total_successful,
        total_failed,
        total_duration,
        all_latencies,
    )
}

async fn run_rate_limited_stress_test<G>(
    g2p: Arc<G>,
    max_concurrent: usize,
    total_requests: usize,
    test_input: &str,
) -> StressTestResults
where
    G: G2p + Send + Sync + 'static,
{
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let mut join_set = JoinSet::new();
    let start_time = Instant::now();
    let test_input = test_input.to_string();

    // Spawn all requests with rate limiting
    for request_id in 0..total_requests {
        let g2p_clone = Arc::clone(&g2p);
        let input_clone = test_input.clone();
        let semaphore_clone = Arc::clone(&semaphore);

        join_set.spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();
            let request_start = Instant::now();

            let result = g2p_clone
                .to_phonemes(&input_clone, Some(LanguageCode::EnUs))
                .await;
            let latency = request_start.elapsed();

            (request_id, result.is_ok(), latency)
        });
    }

    // Collect results
    let mut successful = 0;
    let mut failed = 0;
    let mut latencies = Vec::new();

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((_, success, latency)) => {
                if success {
                    successful += 1;
                } else {
                    failed += 1;
                }
                latencies.push(latency);
            }
            Err(e) => {
                eprintln!("Request task failed: {e:?}");
                failed += 1;
            }
        }
    }

    let total_duration = start_time.elapsed();

    StressTestResults::new(
        total_requests,
        successful,
        failed,
        total_duration,
        latencies,
    )
}

#[tokio::test]
async fn test_dummy_g2p_concurrent_stress() {
    let g2p = Arc::new(DummyG2p::new());
    let test_input = "Concurrent stress test for dummy G2P implementation";

    println!("Running concurrent stress test for DummyG2p...");

    let results = run_concurrent_stress_test(
        g2p, 10,  // 10 concurrent workers
        100, // 100 requests per worker = 1000 total requests
        test_input,
    )
    .await;

    println!("DummyG2p Concurrent Stress Test Results:");
    println!("  Total operations: {}", results.total_operations);
    println!("  Successful: {}", results.successful_operations);
    println!("  Failed: {}", results.failed_operations);
    println!(
        "  Success rate: {:.2}%",
        (results.successful_operations as f64 / results.total_operations as f64) * 100.0
    );
    println!("  Total duration: {:?}", results.total_duration);
    println!(
        "  Operations per second: {:.2}",
        results.operations_per_second
    );
    println!("  Average latency: {:?}", results.average_latency);
    println!("  Min latency: {:?}", results.min_latency);
    println!("  Max latency: {:?}", results.max_latency);

    // Assertions for stress test quality
    assert!(
        results.successful_operations > results.total_operations * 95 / 100,
        "Success rate should be > 95%"
    );
    assert!(
        results.operations_per_second > 100.0,
        "Should handle > 100 operations per second"
    );
    assert!(
        results.max_latency < Duration::from_millis(500),
        "Max latency should be < 500ms"
    );
}

#[tokio::test]
async fn test_english_rule_g2p_concurrent_stress() {
    let g2p_result = EnglishRuleG2p::new();
    assert!(g2p_result.is_ok(), "Failed to create EnglishRuleG2p");

    let g2p = Arc::new(g2p_result.unwrap());
    let test_input = "Concurrent stress test for English rule-based G2P";

    println!("Running concurrent stress test for EnglishRuleG2p...");

    let results = run_concurrent_stress_test(
        g2p, 5,  // 5 concurrent workers (lower due to potentially higher overhead)
        50, // 50 requests per worker = 250 total requests
        test_input,
    )
    .await;

    println!("EnglishRuleG2p Concurrent Stress Test Results:");
    println!("  Total operations: {}", results.total_operations);
    println!("  Successful: {}", results.successful_operations);
    println!("  Failed: {}", results.failed_operations);
    println!(
        "  Success rate: {:.2}%",
        (results.successful_operations as f64 / results.total_operations as f64) * 100.0
    );
    println!("  Total duration: {:?}", results.total_duration);
    println!(
        "  Operations per second: {:.2}",
        results.operations_per_second
    );
    println!("  Average latency: {:?}", results.average_latency);
    println!("  Min latency: {:?}", results.min_latency);
    println!("  Max latency: {:?}", results.max_latency);

    // Assertions for stress test quality
    assert!(
        results.successful_operations > results.total_operations * 90 / 100,
        "Success rate should be > 90%"
    );
    assert!(
        results.operations_per_second > 10.0,
        "Should handle > 10 operations per second"
    );
    assert!(
        results.max_latency < Duration::from_secs(5),
        "Max latency should be < 5 seconds"
    );
}

#[tokio::test]
async fn test_rate_limited_stress() {
    let g2p = Arc::new(DummyG2p::new());
    let test_input = "Rate limited stress test";

    println!("Running rate-limited stress test...");

    let results = run_rate_limited_stress_test(
        g2p, 20,  // Max 20 concurrent requests
        500, // 500 total requests
        test_input,
    )
    .await;

    println!("Rate-Limited Stress Test Results:");
    println!("  Total operations: {}", results.total_operations);
    println!("  Successful: {}", results.successful_operations);
    println!("  Failed: {}", results.failed_operations);
    println!(
        "  Success rate: {:.2}%",
        (results.successful_operations as f64 / results.total_operations as f64) * 100.0
    );
    println!("  Total duration: {:?}", results.total_duration);
    println!(
        "  Operations per second: {:.2}",
        results.operations_per_second
    );
    println!("  Average latency: {:?}", results.average_latency);

    // Assertions
    assert!(
        results.successful_operations > results.total_operations * 95 / 100,
        "Success rate should be > 95%"
    );
    assert!(
        results.operations_per_second > 50.0,
        "Should handle > 50 operations per second with rate limiting"
    );
}

#[tokio::test]
async fn test_long_duration_stress() {
    let g2p = Arc::new(DummyG2p::new());
    let test_input = "Long duration stress test";

    println!("Running long duration stress test (30 seconds)...");

    let mut join_set = JoinSet::new();
    let start_time = Instant::now();
    let test_duration = Duration::from_secs(30);

    // Spawn continuous workers
    for worker_id in 0..5 {
        let g2p_clone = Arc::clone(&g2p);
        let input_clone = test_input.to_string();

        join_set.spawn(async move {
            let mut requests = 0;
            let mut failures = 0;
            let worker_start = Instant::now();

            while worker_start.elapsed() < test_duration {
                match g2p_clone
                    .to_phonemes(&input_clone, Some(LanguageCode::EnUs))
                    .await
                {
                    Ok(_) => requests += 1,
                    Err(_) => failures += 1,
                }

                // Small delay between requests
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            (worker_id, requests, failures)
        });
    }

    // Collect results
    let mut total_requests = 0;
    let mut total_failures = 0;

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((worker_id, requests, failures)) => {
                println!("Worker {worker_id}: {requests} requests, {failures} failures");
                total_requests += requests;
                total_failures += failures;
            }
            Err(e) => {
                eprintln!("Worker task failed: {e:?}");
            }
        }
    }

    let actual_duration = start_time.elapsed();
    let success_rate = if total_requests + total_failures > 0 {
        (total_requests as f64 / (total_requests + total_failures) as f64) * 100.0
    } else {
        0.0
    };

    println!("Long Duration Stress Test Results:");
    println!("  Test duration: {actual_duration:?}");
    println!("  Total requests: {total_requests}");
    println!("  Total failures: {total_failures}");
    println!("  Success rate: {success_rate:.2}%");
    println!(
        "  Requests per second: {:.2}",
        total_requests as f64 / actual_duration.as_secs_f64()
    );

    // Assertions
    assert!(
        success_rate > 95.0,
        "Success rate should be > 95% during long duration test"
    );
    assert!(
        total_requests > 100,
        "Should complete > 100 requests in 30 seconds"
    );
    assert!(
        actual_duration >= test_duration
            && actual_duration < test_duration + Duration::from_secs(5),
        "Test should run for approximately the expected duration"
    );
}

#[tokio::test]
async fn test_memory_stability_under_stress() {
    let g2p = Arc::new(DummyG2p::new());
    let test_input = "Memory stability stress test";

    println!("Running memory stability stress test...");

    // Run multiple rounds of stress testing to check for memory leaks
    let mut all_durations = Vec::new();

    for round in 0..5 {
        println!("  Round {} of 5", round + 1);

        let start = Instant::now();
        let results = run_concurrent_stress_test(
            Arc::clone(&g2p),
            8,  // 8 concurrent workers
            25, // 25 requests per worker = 200 total requests per round
            test_input,
        )
        .await;

        let round_duration = start.elapsed();
        all_durations.push(round_duration);

        println!(
            "    Round {} completed in {:?}, {} successful operations",
            round + 1,
            round_duration,
            results.successful_operations
        );

        // Small delay between rounds
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Analyze performance consistency across rounds
    let avg_duration = Duration::from_nanos(
        (all_durations.iter().map(|d| d.as_nanos()).sum::<u128>() / all_durations.len() as u128)
            .try_into()
            .unwrap_or(0),
    );
    let max_duration = all_durations.iter().max().unwrap();
    let min_duration = all_durations.iter().min().unwrap();

    println!("Memory Stability Test Results:");
    println!("  Average round duration: {avg_duration:?}");
    println!("  Min round duration: {min_duration:?}");
    println!("  Max round duration: {max_duration:?}");
    println!(
        "  Duration variance: {:?}",
        max_duration.saturating_sub(*min_duration)
    );

    // Check for performance degradation (potential memory leak indicator)
    let performance_variance = max_duration.as_secs_f64() / min_duration.as_secs_f64();
    assert!(
        performance_variance < 2.0,
        "Performance variance too high ({performance_variance:.2}x), potential memory leak"
    );
}

#[tokio::test]
async fn test_error_handling_under_stress() {
    let g2p = Arc::new(DummyG2p::new());

    println!("Running error handling stress test...");

    // Test with various potentially problematic inputs
    let very_long_input = "a".repeat(10000);
    let problem_inputs = [
        "",                       // Empty string
        " ",                      // Whitespace only
        very_long_input.as_str(), // Very long input
        "!@#$%^&*()",             // Special characters only
        "\u{200B}test\u{200B}",   // Zero-width spaces
        "test\x00null",           // Null bytes
    ];

    let mut join_set = JoinSet::new();

    for (input_id, input) in problem_inputs.iter().enumerate() {
        for worker_id in 0..3 {
            let g2p_clone = Arc::clone(&g2p);
            let input_clone = input.to_string();

            join_set.spawn(async move {
                let mut results = Vec::new();

                for _ in 0..10 {
                    let start = Instant::now();
                    let result = g2p_clone
                        .to_phonemes(&input_clone, Some(LanguageCode::EnUs))
                        .await;
                    let duration = start.elapsed();

                    results.push((result.is_ok(), duration));
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }

                (input_id, worker_id, results)
            });
        }
    }

    // Collect and analyze results
    let mut total_operations = 0;
    let mut successful_operations = 0;
    let mut max_latency = Duration::ZERO;

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((input_id, worker_id, results)) => {
                let successes = results.iter().filter(|(success, _)| *success).count();
                let max_worker_latency = results
                    .iter()
                    .map(|(_, duration)| *duration)
                    .max()
                    .unwrap_or(Duration::ZERO);

                println!(
                    "  Input {} Worker {}: {}/{} successful, max latency {:?}",
                    input_id,
                    worker_id,
                    successes,
                    results.len(),
                    max_worker_latency
                );

                total_operations += results.len();
                successful_operations += successes;
                max_latency = max_latency.max(max_worker_latency);
            }
            Err(e) => {
                eprintln!("Error handling test task failed: {e:?}");
            }
        }
    }

    println!("Error Handling Stress Test Results:");
    println!("  Total operations: {total_operations}");
    println!("  Successful operations: {successful_operations}");
    println!(
        "  Success rate: {:.2}%",
        (successful_operations as f64 / total_operations as f64) * 100.0
    );
    println!("  Max latency: {max_latency:?}");

    // The system should handle problematic inputs gracefully
    assert!(
        max_latency < Duration::from_secs(1),
        "Even problematic inputs should be handled quickly"
    );
    // We don't require 100% success rate for problematic inputs, but the system shouldn't crash
    assert!(
        total_operations > 0,
        "Should have completed some operations"
    );
}
