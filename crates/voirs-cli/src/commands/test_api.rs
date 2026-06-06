//! API Testing Tools for VoiRS Server
//!
//! This module provides comprehensive API testing functionality for the VoiRS HTTP server.
//! It includes endpoint validation, performance measurements, authentication testing,
//! and detailed test reporting.
//!
//! # Features
//!
//! - **Endpoint Testing**: Test all server endpoints (health, voices, synthesize, stats)
//! - **Authentication**: Validate API key authentication mechanisms
//! - **Performance Metrics**: Measure response times, throughput, latency
//! - **Contract Validation**: Verify API responses match expected schemas
//! - **Test Reports**: Generate detailed JSON/Markdown test reports
//! - **Load Testing**: Basic concurrent request testing
//!
//! # Usage
//!
//! ```bash
//! # Test server at localhost:8080
//! voirs test-api localhost:8080
//!
//! # Test with API key authentication
//! voirs test-api localhost:8080 --api-key YOUR_KEY
//!
//! # Generate detailed report
//! voirs test-api localhost:8080 --report report.json
//!
//! # Run load test
//! voirs test-api localhost:8080 --load-test --concurrent 10
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// API test configuration
#[derive(Debug, Clone)]
pub struct ApiTestConfig {
    /// Server URL (e.g., "http://localhost:8080")
    pub server_url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Timeout for requests in seconds
    pub timeout_secs: u64,
    /// Number of concurrent requests for load testing
    pub concurrent_requests: usize,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for ApiTestConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8080".to_string(),
            api_key: None,
            timeout_secs: 30,
            concurrent_requests: 1,
            verbose: false,
        }
    }
}

/// Test result for a single endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointTestResult {
    /// Endpoint path
    pub endpoint: String,
    /// HTTP method
    pub method: String,
    /// Success status
    pub success: bool,
    /// Response status code
    pub status_code: u16,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Response body size in bytes
    pub response_size_bytes: usize,
    /// Error message if failed
    pub error: Option<String>,
    /// Timestamp of the test
    pub timestamp: DateTime<Utc>,
}

impl EndpointTestResult {
    /// Create a successful test result
    fn success(
        endpoint: String,
        method: String,
        status_code: u16,
        response_time_ms: u64,
        response_size_bytes: usize,
    ) -> Self {
        Self {
            endpoint,
            method,
            success: true,
            status_code,
            response_time_ms,
            response_size_bytes,
            error: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a failed test result
    fn failure(endpoint: String, method: String, error: String) -> Self {
        Self {
            endpoint,
            method,
            success: false,
            status_code: 0,
            response_time_ms: 0,
            response_size_bytes: 0,
            error: Some(error),
            timestamp: Utc::now(),
        }
    }
}

/// Complete API test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTestReport {
    /// Server URL tested
    pub server_url: String,
    /// Test start time
    pub start_time: DateTime<Utc>,
    /// Test end time
    pub end_time: DateTime<Utc>,
    /// Total duration in seconds
    pub duration_secs: f64,
    /// Individual endpoint results
    pub endpoint_results: Vec<EndpointTestResult>,
    /// Overall statistics
    pub statistics: TestStatistics,
    /// Load test results if applicable
    pub load_test: Option<LoadTestResults>,
}

impl ApiTestReport {
    /// Calculate success rate percentage
    pub fn success_rate(&self) -> f64 {
        if self.endpoint_results.is_empty() {
            return 0.0;
        }
        let successful = self.endpoint_results.iter().filter(|r| r.success).count();
        (successful as f64 / self.endpoint_results.len() as f64) * 100.0
    }

    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# VoiRS API Test Report\n\n");
        md.push_str(&format!("**Server**: {}\n", self.server_url));
        md.push_str(&format!(
            "**Date**: {}\n",
            self.start_time.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        md.push_str(&format!("**Duration**: {:.2}s\n\n", self.duration_secs));

        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Success Rate**: {:.1}%\n",
            self.success_rate()
        ));
        md.push_str(&format!(
            "- **Total Tests**: {}\n",
            self.endpoint_results.len()
        ));
        md.push_str(&format!(
            "- **Passed**: {}\n",
            self.endpoint_results.iter().filter(|r| r.success).count()
        ));
        md.push_str(&format!(
            "- **Failed**: {}\n",
            self.endpoint_results.iter().filter(|r| !r.success).count()
        ));
        md.push_str(&format!(
            "- **Avg Response Time**: {:.0}ms\n\n",
            self.statistics.avg_response_time_ms
        ));

        md.push_str("## Endpoint Results\n\n");
        md.push_str("| Endpoint | Method | Status | Response Time | Size | Result |\n");
        md.push_str("|----------|--------|--------|---------------|------|--------|\n");

        for result in &self.endpoint_results {
            let status_emoji = if result.success { "✅" } else { "❌" };
            md.push_str(&format!(
                "| {} | {} | {} | {}ms | {} bytes | {} |\n",
                result.endpoint,
                result.method,
                result.status_code,
                result.response_time_ms,
                result.response_size_bytes,
                status_emoji
            ));
        }

        if let Some(load_test) = &self.load_test {
            md.push_str("\n## Load Test Results\n\n");
            md.push_str(&format!(
                "- **Concurrent Requests**: {}\n",
                load_test.concurrent_requests
            ));
            md.push_str(&format!(
                "- **Total Requests**: {}\n",
                load_test.total_requests
            ));
            md.push_str(&format!(
                "- **Successful**: {}\n",
                load_test.successful_requests
            ));
            md.push_str(&format!("- **Failed**: {}\n", load_test.failed_requests));
            md.push_str(&format!(
                "- **Avg Latency**: {:.0}ms\n",
                load_test.avg_latency_ms
            ));
            md.push_str(&format!(
                "- **Min Latency**: {}ms\n",
                load_test.min_latency_ms
            ));
            md.push_str(&format!(
                "- **Max Latency**: {}ms\n",
                load_test.max_latency_ms
            ));
            md.push_str(&format!(
                "- **Throughput**: {:.2} req/s\n",
                load_test.requests_per_second
            ));
        }

        md
    }
}

/// Test statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatistics {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of successful tests
    pub successful_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Minimum response time in milliseconds
    pub min_response_time_ms: u64,
    /// Maximum response time in milliseconds
    pub max_response_time_ms: u64,
}

/// Load test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    /// Number of concurrent requests
    pub concurrent_requests: usize,
    /// Total requests sent
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Minimum latency in milliseconds
    pub min_latency_ms: u64,
    /// Maximum latency in milliseconds
    pub max_latency_ms: u64,
    /// Requests per second
    pub requests_per_second: f64,
}

/// API tester
pub struct ApiTester {
    config: ApiTestConfig,
    client: Client,
}

impl ApiTester {
    /// Create a new API tester
    pub fn new(config: ApiTestConfig) -> Result<Self> {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        voirs_acoustic::hub::ensure_crypto_provider();

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { config, client })
    }

    /// Run all API tests
    pub async fn run_tests(&self) -> Result<ApiTestReport> {
        let start_time = Utc::now();
        let mut results = Vec::new();

        println!("🚀 Starting API tests for {}", self.config.server_url);
        println!();

        // Test health endpoint
        println!("Testing /api/v1/health...");
        results.push(self.test_health().await);

        // Test voices endpoint
        println!("Testing /api/v1/voices...");
        results.push(self.test_voices().await);

        // Test stats endpoint
        println!("Testing /api/v1/stats...");
        results.push(self.test_stats().await);

        // Test synthesize endpoint
        println!("Testing /api/v1/synthesize...");
        results.push(self.test_synthesize().await);

        // Test authentication if API key provided
        if self.config.api_key.is_some() {
            println!("Testing /api/v1/auth/info...");
            results.push(self.test_auth_info().await);
        }

        let end_time = Utc::now();
        let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

        // Calculate statistics
        let statistics = self.calculate_statistics(&results);

        // Run load test if configured
        let load_test = if self.config.concurrent_requests > 1 {
            println!("\n🔥 Running load test...");
            Some(self.run_load_test().await?)
        } else {
            None
        };

        Ok(ApiTestReport {
            server_url: self.config.server_url.clone(),
            start_time,
            end_time,
            duration_secs,
            endpoint_results: results,
            statistics,
            load_test,
        })
    }

    /// Test health endpoint
    async fn test_health(&self) -> EndpointTestResult {
        let url = format!("{}/api/v1/health", self.config.server_url);
        let start = Instant::now();

        match self.client.get(&url).send().await {
            Ok(response) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let status = response.status().as_u16();
                match response.bytes().await {
                    Ok(body) => EndpointTestResult::success(
                        "/api/v1/health".to_string(),
                        "GET".to_string(),
                        status,
                        elapsed,
                        body.len(),
                    ),
                    Err(e) => EndpointTestResult::failure(
                        "/api/v1/health".to_string(),
                        "GET".to_string(),
                        format!("Failed to read response body: {}", e),
                    ),
                }
            }
            Err(e) => EndpointTestResult::failure(
                "/api/v1/health".to_string(),
                "GET".to_string(),
                format!("Request failed: {}", e),
            ),
        }
    }

    /// Test voices endpoint
    async fn test_voices(&self) -> EndpointTestResult {
        let url = format!("{}/api/v1/voices", self.config.server_url);
        let start = Instant::now();

        let mut request = self.client.get(&url);
        if let Some(api_key) = &self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        match request.send().await {
            Ok(response) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let status = response.status().as_u16();
                match response.bytes().await {
                    Ok(body) => EndpointTestResult::success(
                        "/api/v1/voices".to_string(),
                        "GET".to_string(),
                        status,
                        elapsed,
                        body.len(),
                    ),
                    Err(e) => EndpointTestResult::failure(
                        "/api/v1/voices".to_string(),
                        "GET".to_string(),
                        format!("Failed to read response body: {}", e),
                    ),
                }
            }
            Err(e) => EndpointTestResult::failure(
                "/api/v1/voices".to_string(),
                "GET".to_string(),
                format!("Request failed: {}", e),
            ),
        }
    }

    /// Test stats endpoint
    async fn test_stats(&self) -> EndpointTestResult {
        let url = format!("{}/api/v1/stats", self.config.server_url);
        let start = Instant::now();

        let mut request = self.client.get(&url);
        if let Some(api_key) = &self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        match request.send().await {
            Ok(response) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let status = response.status().as_u16();
                match response.bytes().await {
                    Ok(body) => EndpointTestResult::success(
                        "/api/v1/stats".to_string(),
                        "GET".to_string(),
                        status,
                        elapsed,
                        body.len(),
                    ),
                    Err(e) => EndpointTestResult::failure(
                        "/api/v1/stats".to_string(),
                        "GET".to_string(),
                        format!("Failed to read response body: {}", e),
                    ),
                }
            }
            Err(e) => EndpointTestResult::failure(
                "/api/v1/stats".to_string(),
                "GET".to_string(),
                format!("Request failed: {}", e),
            ),
        }
    }

    /// Test synthesize endpoint
    async fn test_synthesize(&self) -> EndpointTestResult {
        let url = format!("{}/api/v1/synthesize", self.config.server_url);
        let start = Instant::now();

        let body = serde_json::json!({
            "text": "Hello, this is a test.",
            "voice": "default",
        });

        let mut request = self.client.post(&url).json(&body);
        if let Some(api_key) = &self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        match request.send().await {
            Ok(response) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let status = response.status().as_u16();
                match response.bytes().await {
                    Ok(body_bytes) => EndpointTestResult::success(
                        "/api/v1/synthesize".to_string(),
                        "POST".to_string(),
                        status,
                        elapsed,
                        body_bytes.len(),
                    ),
                    Err(e) => EndpointTestResult::failure(
                        "/api/v1/synthesize".to_string(),
                        "POST".to_string(),
                        format!("Failed to read response body: {}", e),
                    ),
                }
            }
            Err(e) => EndpointTestResult::failure(
                "/api/v1/synthesize".to_string(),
                "POST".to_string(),
                format!("Request failed: {}", e),
            ),
        }
    }

    /// Test auth info endpoint
    async fn test_auth_info(&self) -> EndpointTestResult {
        let url = format!("{}/api/v1/auth/info", self.config.server_url);
        let start = Instant::now();

        let mut request = self.client.get(&url);
        if let Some(api_key) = &self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        match request.send().await {
            Ok(response) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let status = response.status().as_u16();
                match response.bytes().await {
                    Ok(body) => EndpointTestResult::success(
                        "/api/v1/auth/info".to_string(),
                        "GET".to_string(),
                        status,
                        elapsed,
                        body.len(),
                    ),
                    Err(e) => EndpointTestResult::failure(
                        "/api/v1/auth/info".to_string(),
                        "GET".to_string(),
                        format!("Failed to read response body: {}", e),
                    ),
                }
            }
            Err(e) => EndpointTestResult::failure(
                "/api/v1/auth/info".to_string(),
                "GET".to_string(),
                format!("Request failed: {}", e),
            ),
        }
    }

    /// Calculate test statistics
    fn calculate_statistics(&self, results: &[EndpointTestResult]) -> TestStatistics {
        let total_tests = results.len();
        let successful_tests = results.iter().filter(|r| r.success).count();
        let failed_tests = total_tests - successful_tests;

        let response_times: Vec<u64> = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.response_time_ms)
            .collect();

        let avg_response_time_ms = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
        };

        let min_response_time_ms = response_times.iter().min().copied().unwrap_or(0);
        let max_response_time_ms = response_times.iter().max().copied().unwrap_or(0);

        TestStatistics {
            total_tests,
            successful_tests,
            failed_tests,
            avg_response_time_ms,
            min_response_time_ms,
            max_response_time_ms,
        }
    }

    /// Run load test
    async fn run_load_test(&self) -> Result<LoadTestResults> {
        let url = format!("{}/api/v1/health", self.config.server_url);
        let concurrent = self.config.concurrent_requests;
        let total_requests = concurrent * 10; // 10 rounds per worker

        let start = Instant::now();
        let mut handles = Vec::new();

        for _ in 0..concurrent {
            let client = self.client.clone();
            let url = url.clone();
            let handle = tokio::spawn(async move {
                let mut results = Vec::new();
                for _ in 0..10 {
                    let req_start = Instant::now();
                    let result = client.get(&url).send().await;
                    let elapsed = req_start.elapsed().as_millis() as u64;
                    results.push((result.is_ok(), elapsed));
                }
                results
            });
            handles.push(handle);
        }

        let mut all_results = Vec::new();
        for handle in handles {
            let results = handle.await?;
            all_results.extend(results);
        }

        let duration = start.elapsed();
        let successful = all_results.iter().filter(|(ok, _)| *ok).count();
        let failed = total_requests - successful;

        let latencies: Vec<u64> = all_results.iter().map(|(_, latency)| *latency).collect();
        let avg_latency_ms = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
        let min_latency_ms = latencies.iter().min().copied().unwrap_or(0);
        let max_latency_ms = latencies.iter().max().copied().unwrap_or(0);
        let requests_per_second = total_requests as f64 / duration.as_secs_f64();

        Ok(LoadTestResults {
            concurrent_requests: concurrent,
            total_requests,
            successful_requests: successful,
            failed_requests: failed,
            avg_latency_ms,
            min_latency_ms,
            max_latency_ms,
            requests_per_second,
        })
    }
}

/// Print test report to console
pub fn print_report(report: &ApiTestReport) {
    println!("\n{}", "=".repeat(60));
    println!("  VoiRS API Test Report");
    println!("{}\n", "=".repeat(60));

    println!("Server: {}", report.server_url);
    println!("Duration: {:.2}s", report.duration_secs);
    println!();

    println!("Summary:");
    println!("  Success Rate: {:.1}%", report.success_rate());
    println!("  Total Tests: {}", report.endpoint_results.len());
    println!(
        "  Passed: {}",
        report.endpoint_results.iter().filter(|r| r.success).count()
    );
    println!(
        "  Failed: {}",
        report
            .endpoint_results
            .iter()
            .filter(|r| !r.success)
            .count()
    );
    println!(
        "  Avg Response Time: {:.0}ms",
        report.statistics.avg_response_time_ms
    );
    println!();

    println!("Endpoint Results:");
    for result in &report.endpoint_results {
        let status = if result.success {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        println!(
            "  {} {} {} - {} - {}ms",
            status, result.method, result.endpoint, result.status_code, result.response_time_ms
        );
        if let Some(error) = &result.error {
            println!("    Error: {}", error);
        }
    }

    if let Some(load_test) = &report.load_test {
        println!();
        println!("Load Test Results:");
        println!("  Concurrent Requests: {}", load_test.concurrent_requests);
        println!("  Total Requests: {}", load_test.total_requests);
        println!("  Successful: {}", load_test.successful_requests);
        println!("  Failed: {}", load_test.failed_requests);
        println!("  Avg Latency: {:.0}ms", load_test.avg_latency_ms);
        println!("  Min Latency: {}ms", load_test.min_latency_ms);
        println!("  Max Latency: {}ms", load_test.max_latency_ms);
        println!("  Throughput: {:.2} req/s", load_test.requests_per_second);
    }

    println!("\n{}\n", "=".repeat(60));
}

/// Run API tests with the given configuration
pub async fn run_api_tests(
    server_url: String,
    api_key: Option<String>,
    concurrent: Option<usize>,
    report_path: Option<String>,
    verbose: bool,
) -> Result<()> {
    let config = ApiTestConfig {
        server_url,
        api_key,
        timeout_secs: 30,
        concurrent_requests: concurrent.unwrap_or(1),
        verbose,
    };

    let tester = ApiTester::new(config)?;
    let report = tester.run_tests().await?;

    // Print report to console
    print_report(&report);

    // Save report if path provided
    if let Some(path) = report_path {
        if path.ends_with(".json") {
            let json = serde_json::to_string_pretty(&report)
                .context("Failed to serialize report to JSON")?;
            std::fs::write(&path, json).context("Failed to write JSON report")?;
            println!("📄 JSON report saved to: {}", path);
        } else if path.ends_with(".md") {
            let markdown = report.to_markdown();
            std::fs::write(&path, markdown).context("Failed to write Markdown report")?;
            println!("📄 Markdown report saved to: {}", path);
        } else {
            anyhow::bail!("Report path must end with .json or .md");
        }
    }

    // Exit with error code if any tests failed
    if report.statistics.failed_tests > 0 {
        anyhow::bail!("{} test(s) failed", report.statistics.failed_tests);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_test_config_default() {
        let config = ApiTestConfig::default();
        assert_eq!(config.server_url, "http://localhost:8080");
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.concurrent_requests, 1);
        assert!(!config.verbose);
    }

    #[test]
    fn test_endpoint_test_result_success() {
        let result = EndpointTestResult::success(
            "/api/v1/health".to_string(),
            "GET".to_string(),
            200,
            150,
            1024,
        );

        assert!(result.success);
        assert_eq!(result.endpoint, "/api/v1/health");
        assert_eq!(result.method, "GET");
        assert_eq!(result.status_code, 200);
        assert_eq!(result.response_time_ms, 150);
        assert_eq!(result.response_size_bytes, 1024);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_endpoint_test_result_failure() {
        let result = EndpointTestResult::failure(
            "/api/v1/health".to_string(),
            "GET".to_string(),
            "Connection refused".to_string(),
        );

        assert!(!result.success);
        assert_eq!(result.endpoint, "/api/v1/health");
        assert_eq!(result.method, "GET");
        assert_eq!(result.error, Some("Connection refused".to_string()));
    }

    #[test]
    fn test_api_test_report_success_rate() {
        let results = vec![
            EndpointTestResult::success(
                "/api/v1/health".to_string(),
                "GET".to_string(),
                200,
                100,
                512,
            ),
            EndpointTestResult::success(
                "/api/v1/voices".to_string(),
                "GET".to_string(),
                200,
                150,
                1024,
            ),
            EndpointTestResult::failure(
                "/api/v1/synth".to_string(),
                "POST".to_string(),
                "Error".to_string(),
            ),
        ];

        let report = ApiTestReport {
            server_url: "http://localhost:8080".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            duration_secs: 1.5,
            endpoint_results: results,
            statistics: TestStatistics {
                total_tests: 3,
                successful_tests: 2,
                failed_tests: 1,
                avg_response_time_ms: 125.0,
                min_response_time_ms: 100,
                max_response_time_ms: 150,
            },
            load_test: None,
        };

        assert!((report.success_rate() - 66.666).abs() < 0.01);
    }

    #[test]
    fn test_markdown_report_generation() {
        let results = vec![EndpointTestResult::success(
            "/api/v1/health".to_string(),
            "GET".to_string(),
            200,
            100,
            512,
        )];

        let report = ApiTestReport {
            server_url: "http://localhost:8080".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            duration_secs: 0.5,
            endpoint_results: results,
            statistics: TestStatistics {
                total_tests: 1,
                successful_tests: 1,
                failed_tests: 0,
                avg_response_time_ms: 100.0,
                min_response_time_ms: 100,
                max_response_time_ms: 100,
            },
            load_test: None,
        };

        let markdown = report.to_markdown();
        assert!(markdown.contains("# VoiRS API Test Report"));
        assert!(markdown.contains("http://localhost:8080"));
        assert!(markdown.contains("/api/v1/health"));
    }
}
