use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::runtime::Runtime;
use voirs_g2p::{DummyG2p, G2p, LanguageCode};

// Custom allocator for memory tracking
struct TrackingAllocator {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    peak_memory: AtomicUsize,
    current_memory: AtomicUsize,
}

static TRACKING_ALLOCATOR: TrackingAllocator = TrackingAllocator {
    allocations: AtomicUsize::new(0),
    deallocations: AtomicUsize::new(0),
    peak_memory: AtomicUsize::new(0),
    current_memory: AtomicUsize::new(0),
};

impl TrackingAllocator {
    fn reset_stats(&self) {
        self.allocations.store(0, Ordering::SeqCst);
        self.deallocations.store(0, Ordering::SeqCst);
        self.peak_memory.store(0, Ordering::SeqCst);
        self.current_memory.store(0, Ordering::SeqCst);
    }

    fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            allocations: self.allocations.load(Ordering::SeqCst),
            deallocations: self.deallocations.load(Ordering::SeqCst),
            peak_memory: self.peak_memory.load(Ordering::SeqCst),
            current_memory: self.current_memory.load(Ordering::SeqCst),
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            self.allocations.fetch_add(1, Ordering::SeqCst);
            let current = self
                .current_memory
                .fetch_add(layout.size(), Ordering::SeqCst)
                + layout.size();

            // Update peak memory if current exceeds it
            let mut peak = self.peak_memory.load(Ordering::SeqCst);
            while current > peak {
                match self.peak_memory.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(new_peak) => peak = new_peak,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.deallocations.fetch_add(1, Ordering::SeqCst);
        self.current_memory
            .fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[derive(Debug, Clone)]
struct MemoryStats {
    allocations: usize,
    deallocations: usize,
    peak_memory: usize,
    #[allow(dead_code)]
    current_memory: usize,
}

impl MemoryStats {
    fn memory_efficiency(&self) -> f64 {
        if self.allocations == 0 {
            return 1.0;
        }
        self.deallocations as f64 / self.allocations as f64
    }

    fn peak_memory_mb(&self) -> f64 {
        self.peak_memory as f64 / (1024.0 * 1024.0)
    }
}

// Memory profile results storage
static MEMORY_RESULTS: Mutex<Vec<(String, MemoryStats)>> = Mutex::new(Vec::new());

fn measure_memory_usage<F, R>(name: &str, mut operation: F) -> R
where
    F: FnMut() -> R,
{
    // Reset tracking stats
    TRACKING_ALLOCATOR.reset_stats();

    // Perform operation
    let result = operation();

    // Collect stats
    let stats = TRACKING_ALLOCATOR.get_stats();

    // Store results
    if let Ok(mut results) = MEMORY_RESULTS.lock() {
        results.push((name.to_string(), stats.clone()));
    }

    println!(
        "Memory usage for {}: {} allocs, {} deallocs, {:.2} MB peak, {:.1}% efficiency",
        name,
        stats.allocations,
        stats.deallocations,
        stats.peak_memory_mb(),
        stats.memory_efficiency() * 100.0
    );

    result
}

fn benchmark_dummy_g2p_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();

    let mut group = c.benchmark_group("dummy_g2p_memory");
    group.measurement_time(Duration::from_secs(10));

    // Test memory usage with different text sizes
    let large_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(10);
    let very_large_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);

    let test_cases = vec![
        ("small", "hello"),
        ("medium", "The quick brown fox jumps over the lazy dog"),
        ("large", large_text.as_str()),
        ("very_large", very_large_text.as_str()),
    ];

    for (size_name, text) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("text_size", size_name),
            &text,
            |b, text| {
                b.iter_custom(|iters| {
                    let name = format!("dummy_g2p_{size_name}");
                    measure_memory_usage(&name, || {
                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            rt.block_on(async {
                                let result = g2p
                                    .to_phonemes(black_box(text), Some(LanguageCode::EnUs))
                                    .await;
                                black_box(result.unwrap());
                            });
                        }
                        start.elapsed()
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_repeated_operations_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();

    let mut group = c.benchmark_group("repeated_operations_memory");
    group.measurement_time(Duration::from_secs(15));

    let test_text = "Memory test for repeated operations";
    let iteration_counts = vec![10, 100, 1000];

    for iterations in iteration_counts {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                b.iter_custom(|_| {
                    let name = format!("repeated_ops_{iterations}_iterations");
                    measure_memory_usage(&name, || {
                        let start = std::time::Instant::now();
                        rt.block_on(async {
                            for _ in 0..iterations {
                                let result = g2p
                                    .to_phonemes(black_box(test_text), Some(LanguageCode::EnUs))
                                    .await;
                                black_box(result.unwrap());
                            }
                        });
                        start.elapsed()
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_memory_usage");
    group.measurement_time(Duration::from_secs(20));

    let test_text = "Concurrent memory test";
    let thread_counts = vec![1, 2, 4, 8];

    for thread_count in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter_custom(|_| {
                    let name = format!("concurrent_{thread_count}_threads");
                    measure_memory_usage(&name, || {
                        let start = std::time::Instant::now();
                        rt.block_on(async {
                            let mut handles = Vec::new();

                            for _ in 0..thread_count {
                                let g2p = DummyG2p::new();
                                let handle = tokio::spawn(async move {
                                    for _ in 0..50 {
                                        let result = g2p
                                            .to_phonemes(
                                                black_box(test_text),
                                                Some(LanguageCode::EnUs),
                                            )
                                            .await;
                                        black_box(result.unwrap());
                                    }
                                });
                                handles.push(handle);
                            }

                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                        start.elapsed()
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_leaks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_leak_detection");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let test_text = "Memory leak detection test";

    group.bench_function("leak_test", |b| {
        b.iter_custom(|_| {
            measure_memory_usage("leak_detection", || {
                let start = std::time::Instant::now();

                // Create and destroy many G2P instances
                rt.block_on(async {
                    for _ in 0..1000 {
                        let g2p = DummyG2p::new();
                        let result = g2p
                            .to_phonemes(black_box(test_text), Some(LanguageCode::EnUs))
                            .await;
                        black_box(result.unwrap());
                        // G2P should be dropped here
                    }
                });

                start.elapsed()
            })
        });
    });

    group.finish();
}

fn benchmark_large_input_memory(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let g2p = DummyG2p::new();

    let mut group = c.benchmark_group("large_input_memory");
    group.measurement_time(Duration::from_secs(15));

    // Test with progressively larger inputs
    let base_text = "The quick brown fox jumps over the lazy dog. ";
    let sizes = vec![1, 10, 100, 1000];

    for size in sizes {
        let large_text = base_text.repeat(size);
        let text_size_kb = large_text.len() as f64 / 1024.0;

        group.bench_with_input(
            BenchmarkId::new("size_kb", format!("{text_size_kb:.1}")),
            &large_text,
            |b, text| {
                b.iter_custom(|iters| {
                    let name = format!("large_input_{text_size_kb:.1}kb");
                    measure_memory_usage(&name, || {
                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            rt.block_on(async {
                                let result = g2p
                                    .to_phonemes(black_box(text), Some(LanguageCode::EnUs))
                                    .await;
                                black_box(result.unwrap());
                            });
                        }
                        start.elapsed()
                    })
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn report_memory_summary() {
    if let Ok(results) = MEMORY_RESULTS.lock() {
        println!("\n=== Memory Profiling Summary ===");

        let mut max_peak: f64 = 0.0;
        let mut min_efficiency: f64 = 100.0;
        let mut total_allocations = 0;

        for (name, stats) in results.iter() {
            let peak_mb = stats.peak_memory_mb();
            let efficiency = stats.memory_efficiency() * 100.0;

            println!(
                "{name}: {peak_mb:.2} MB peak, {efficiency:.1}% efficiency, {} allocs",
                stats.allocations
            );

            max_peak = max_peak.max(peak_mb);
            min_efficiency = min_efficiency.min(efficiency);
            total_allocations += stats.allocations;
        }

        println!("\nOverall Statistics:");
        println!("  Maximum peak memory: {max_peak:.2} MB");
        println!("  Minimum efficiency: {min_efficiency:.1}%");
        println!("  Total allocations across all tests: {total_allocations}");

        // Memory usage recommendations
        if max_peak > 100.0 {
            println!("\nWARNING: Peak memory usage exceeds 100 MB - consider optimization");
        }
        if min_efficiency < 95.0 {
            println!("WARNING: Memory efficiency below 95% - potential memory leaks detected");
        }

        println!("=== End Memory Summary ===\n");
    }
}

// Custom criterion configuration for memory benchmarks
fn memory_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(20)
        .warm_up_time(Duration::from_secs(3))
}

criterion_group!(
    name = memory_benches;
    config = memory_criterion();
    targets =
        benchmark_dummy_g2p_memory,
        benchmark_repeated_operations_memory,
        benchmark_concurrent_memory_usage,
        benchmark_memory_leaks,
        benchmark_large_input_memory,
);

criterion_main!(memory_benches);

// Run memory summary when benchmark completes
#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::TRACKING_ALLOCATOR;

    #[test]
    fn test_memory_tracking() {
        TRACKING_ALLOCATOR.reset_stats();

        // Perform some allocations
        let _vec: Vec<u8> = vec![0; 1024];
        let _string = String::from("test");

        let stats = TRACKING_ALLOCATOR.get_stats();
        assert!(stats.allocations > 0);
        assert!(stats.peak_memory > 0);

        drop(_vec);
        drop(_string);

        // Note: deallocations might not be immediate due to allocator behavior
        println!("Memory tracking test: {stats:?}");
    }
}
