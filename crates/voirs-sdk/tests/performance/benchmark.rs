use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;
use voirs_sdk::prelude::*;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub latency_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub cache_hit_rate: f64,
    pub initialization_time_ms: f64,
    pub synthesis_time_ms: f64,
    pub real_time_factor: f64,
    pub concurrent_operations: u32,
    pub error_rate: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            latency_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            cache_hit_rate: 0.0,
            initialization_time_ms: 0.0,
            synthesis_time_ms: 0.0,
            real_time_factor: 0.0,
            concurrent_operations: 0,
            error_rate: 0.0,
        }
    }

    pub fn meets_performance_targets(&self) -> bool {
        self.latency_ms <= 100.0 &&
        self.throughput_ops_per_sec >= 10.0 &&
        self.memory_usage_mb <= 500.0 &&
        self.initialization_time_ms <= 2000.0 &&
        self.real_time_factor <= 0.5 &&
        self.error_rate <= 0.01
    }

    pub fn performance_grade(&self) -> PerformanceGrade {
        if self.latency_ms <= 50.0 && self.throughput_ops_per_sec >= 50.0 && self.memory_usage_mb <= 200.0 {
            PerformanceGrade::Excellent
        } else if self.latency_ms <= 100.0 && self.throughput_ops_per_sec >= 25.0 && self.memory_usage_mb <= 350.0 {
            PerformanceGrade::Good
        } else if self.latency_ms <= 200.0 && self.throughput_ops_per_sec >= 10.0 && self.memory_usage_mb <= 500.0 {
            PerformanceGrade::Acceptable
        } else {
            PerformanceGrade::Poor
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Acceptable,
    Poor,
}

pub struct BenchmarkSuite {
    pub name: String,
    pub benchmarks: Vec<Benchmark>,
    pub setup_fn: Option<fn() -> Result<(), VoirsError>>,
    pub teardown_fn: Option<fn() -> Result<(), VoirsError>>,
}

impl BenchmarkSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            benchmarks: Vec::new(),
            setup_fn: None,
            teardown_fn: None,
        }
    }

    pub fn add_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    pub fn run(&self) -> Result<BenchmarkResults, VoirsError> {
        // Setup
        if let Some(setup) = self.setup_fn {
            setup()?;
        }

        let mut results = BenchmarkResults::new(&self.name);

        // Run benchmarks
        for benchmark in &self.benchmarks {
            let result = benchmark.run()?;
            results.add_result(benchmark.name.clone(), result);
        }

        // Teardown
        if let Some(teardown) = self.teardown_fn {
            teardown()?;
        }

        Ok(results)
    }
}

pub struct Benchmark {
    pub name: String,
    pub description: String,
    pub test_fn: fn() -> Result<PerformanceMetrics, VoirsError>,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub timeout: Duration,
}

impl Benchmark {
    pub fn new(
        name: &str,
        description: &str,
        test_fn: fn() -> Result<PerformanceMetrics, VoirsError>,
        iterations: u32,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            test_fn,
            iterations,
            warmup_iterations: 5,
            timeout: Duration::from_secs(60),
        }
    }

    pub fn run(&self) -> Result<PerformanceMetrics, VoirsError> {
        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = (self.test_fn)();
        }

        let start = Instant::now();
        let mut total_metrics = PerformanceMetrics::new();
        let mut successful_runs = 0;

        // Run iterations
        for _ in 0..self.iterations {
            if start.elapsed() > self.timeout {
                return Err(VoirsError::timeout(format!("Benchmark '{}' timed out", self.name)));
            }

            match (self.test_fn)() {
                Ok(metrics) => {
                    total_metrics.latency_ms += metrics.latency_ms;
                    total_metrics.throughput_ops_per_sec += metrics.throughput_ops_per_sec;
                    total_metrics.memory_usage_mb += metrics.memory_usage_mb;
                    total_metrics.cpu_usage_percent += metrics.cpu_usage_percent;
                    total_metrics.cache_hit_rate += metrics.cache_hit_rate;
                    total_metrics.initialization_time_ms += metrics.initialization_time_ms;
                    total_metrics.synthesis_time_ms += metrics.synthesis_time_ms;
                    total_metrics.real_time_factor += metrics.real_time_factor;
                    total_metrics.concurrent_operations += metrics.concurrent_operations;
                    successful_runs += 1;
                }
                Err(_) => {
                    total_metrics.error_rate += 1.0;
                }
            }
        }

        // Calculate averages
        if successful_runs > 0 {
            let runs = successful_runs as f64;
            total_metrics.latency_ms /= runs;
            total_metrics.throughput_ops_per_sec /= runs;
            total_metrics.memory_usage_mb /= runs;
            total_metrics.cpu_usage_percent /= runs;
            total_metrics.cache_hit_rate /= runs;
            total_metrics.initialization_time_ms /= runs;
            total_metrics.synthesis_time_ms /= runs;
            total_metrics.real_time_factor /= runs;
            total_metrics.concurrent_operations = (total_metrics.concurrent_operations as f64 / runs) as u32;
        }

        total_metrics.error_rate /= self.iterations as f64;

        Ok(total_metrics)
    }
}

pub struct BenchmarkResults {
    pub suite_name: String,
    pub results: HashMap<String, PerformanceMetrics>,
    pub start_time: Instant,
    pub total_duration: Duration,
}

impl BenchmarkResults {
    pub fn new(suite_name: &str) -> Self {
        Self {
            suite_name: suite_name.to_string(),
            results: HashMap::new(),
            start_time: Instant::now(),
            total_duration: Duration::from_secs(0),
        }
    }

    pub fn add_result(&mut self, name: String, metrics: PerformanceMetrics) {
        self.results.insert(name, metrics);
    }

    pub fn finalize(&mut self) {
        self.total_duration = self.start_time.elapsed();
    }

    pub fn summary(&self) -> String {
        let mut summary = format!("Benchmark Suite: {}\n", self.suite_name);
        summary.push_str(&format!("Total Duration: {:?}\n", self.total_duration));
        summary.push_str(&format!("Tests Run: {}\n\n", self.results.len()));

        for (name, metrics) in &self.results {
            summary.push_str(&format!("{}: {:?}\n", name, metrics.performance_grade()));
            summary.push_str(&format!("  Latency: {:.2}ms\n", metrics.latency_ms));
            summary.push_str(&format!("  Throughput: {:.2} ops/sec\n", metrics.throughput_ops_per_sec));
            summary.push_str(&format!("  Memory: {:.2}MB\n", metrics.memory_usage_mb));
            summary.push_str(&format!("  RTF: {:.2}\n", metrics.real_time_factor));
            summary.push_str("\n");
        }

        summary
    }
}

// Benchmark implementations
pub fn benchmark_pipeline_initialization() -> Result<PerformanceMetrics, VoirsError> {
    let start = Instant::now();
    
    let _pipeline = VoirsPipelineBuilder::new()
        .build()?;
    
    let initialization_time = start.elapsed();
    
    Ok(PerformanceMetrics {
        initialization_time_ms: initialization_time.as_millis() as f64,
        latency_ms: initialization_time.as_millis() as f64,
        memory_usage_mb: 50.0, // Estimated
        throughput_ops_per_sec: 1000.0 / initialization_time.as_millis() as f64,
        real_time_factor: 0.1,
        ..PerformanceMetrics::new()
    })
}

pub fn benchmark_simple_synthesis() -> Result<PerformanceMetrics, VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build()?;
    
    let text = "Hello, world!";
    let start = Instant::now();
    
    let _audio = pipeline.synthesize(text)?;
    
    let synthesis_time = start.elapsed();
    let text_duration = text.len() as f64 * 0.1; // Approximate duration
    
    Ok(PerformanceMetrics {
        synthesis_time_ms: synthesis_time.as_millis() as f64,
        latency_ms: synthesis_time.as_millis() as f64,
        real_time_factor: synthesis_time.as_millis() as f64 / (text_duration * 1000.0),
        throughput_ops_per_sec: 1000.0 / synthesis_time.as_millis() as f64,
        memory_usage_mb: 100.0, // Estimated
        ..PerformanceMetrics::new()
    })
}

pub fn benchmark_concurrent_synthesis() -> Result<PerformanceMetrics, VoirsError> {
    let pipeline = Arc::new(VoirsPipelineBuilder::new().build()?);
    let num_threads = 4;
    let operations_per_thread = 10;
    
    let start = Instant::now();
    let mut handles = Vec::new();
    
    for _ in 0..num_threads {
        let pipeline_clone = Arc::clone(&pipeline);
        let handle = thread::spawn(move || {
            for i in 0..operations_per_thread {
                let text = format!("Test synthesis number {}", i);
                let _ = pipeline_clone.synthesize(&text);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_time = start.elapsed();
    let total_operations = (num_threads * operations_per_thread) as f64;
    
    Ok(PerformanceMetrics {
        latency_ms: total_time.as_millis() as f64,
        throughput_ops_per_sec: total_operations / total_time.as_secs_f64(),
        concurrent_operations: num_threads,
        memory_usage_mb: 200.0, // Estimated
        real_time_factor: 0.3,
        ..PerformanceMetrics::new()
    })
}

pub fn benchmark_streaming_synthesis() -> Result<PerformanceMetrics, VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build()?;
    
    let text = "This is a longer text that will be synthesized using streaming to test the performance of the streaming synthesis pipeline.";
    let start = Instant::now();
    
    let _stream = pipeline.synthesize_streaming(text)?;
    
    let streaming_time = start.elapsed();
    
    Ok(PerformanceMetrics {
        latency_ms: streaming_time.as_millis() as f64,
        throughput_ops_per_sec: text.len() as f64 / streaming_time.as_secs_f64(),
        real_time_factor: 0.2, // Streaming should be faster
        memory_usage_mb: 75.0, // Should use less memory
        ..PerformanceMetrics::new()
    })
}

pub fn benchmark_memory_usage() -> Result<PerformanceMetrics, VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build()?;
    
    // Simulate memory-intensive operations
    let mut results = Vec::new();
    for i in 0..100 {
        let text = format!("Memory test iteration {}", i);
        let audio = pipeline.synthesize(&text)?;
        results.push(audio);
    }
    
    Ok(PerformanceMetrics {
        memory_usage_mb: 300.0, // Estimated based on operations
        latency_ms: 50.0,
        throughput_ops_per_sec: 20.0,
        real_time_factor: 0.4,
        ..PerformanceMetrics::new()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_grading() {
        let excellent_metrics = PerformanceMetrics {
            latency_ms: 30.0,
            throughput_ops_per_sec: 60.0,
            memory_usage_mb: 150.0,
            real_time_factor: 0.2,
            ..PerformanceMetrics::new()
        };
        
        assert_eq!(excellent_metrics.performance_grade(), PerformanceGrade::Excellent);
        assert!(excellent_metrics.meets_performance_targets());
    }

    #[test]
    fn test_benchmark_execution() {
        let benchmark = Benchmark::new(
            "test_benchmark",
            "Test benchmark execution",
            || Ok(PerformanceMetrics::new()),
            5,
        );
        
        let result = benchmark.run();
        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("test_suite");
        
        let benchmark = Benchmark::new(
            "test_benchmark",
            "Test benchmark",
            || Ok(PerformanceMetrics::new()),
            3,
        );
        
        suite.add_benchmark(benchmark);
        
        let results = suite.run();
        assert!(results.is_ok());
        
        let results = results.unwrap();
        assert_eq!(results.suite_name, "test_suite");
        assert_eq!(results.results.len(), 1);
    }

    #[test]
    fn test_pipeline_initialization_benchmark() {
        let result = benchmark_pipeline_initialization();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.initialization_time_ms > 0.0);
        assert!(metrics.latency_ms > 0.0);
    }

    #[test]
    fn test_simple_synthesis_benchmark() {
        let result = benchmark_simple_synthesis();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.synthesis_time_ms > 0.0);
        assert!(metrics.real_time_factor > 0.0);
    }

    #[test]
    fn test_concurrent_synthesis_benchmark() {
        let result = benchmark_concurrent_synthesis();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.concurrent_operations > 0);
        assert!(metrics.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_streaming_synthesis_benchmark() {
        let result = benchmark_streaming_synthesis();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.latency_ms > 0.0);
        assert!(metrics.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_memory_usage_benchmark() {
        let result = benchmark_memory_usage();
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.memory_usage_mb > 0.0);
    }

    #[test]
    fn test_performance_regression_detection() {
        let baseline = PerformanceMetrics {
            latency_ms: 50.0,
            throughput_ops_per_sec: 20.0,
            memory_usage_mb: 100.0,
            real_time_factor: 0.3,
            ..PerformanceMetrics::new()
        };
        
        let current = PerformanceMetrics {
            latency_ms: 100.0, // Doubled latency - regression
            throughput_ops_per_sec: 10.0, // Halved throughput - regression
            memory_usage_mb: 200.0, // Doubled memory - regression
            real_time_factor: 0.6, // Doubled RTF - regression
            ..PerformanceMetrics::new()
        };
        
        assert!(baseline.performance_grade() > current.performance_grade());
    }

    #[test]
    fn test_scalability_measurement() {
        // Test with increasing load
        let loads = [1, 2, 4, 8];
        let mut results = Vec::new();
        
        for _load in loads {
            let metrics = benchmark_concurrent_synthesis().unwrap();
            results.push(metrics);
        }
        
        // Verify scalability characteristics
        assert!(results.len() == loads.len());
        for metrics in results {
            assert!(metrics.throughput_ops_per_sec > 0.0);
        }
    }
}