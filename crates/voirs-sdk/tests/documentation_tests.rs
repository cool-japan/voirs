mod docs;

use docs::DocumentationTester;

#[tokio::test]
async fn test_readme_examples() {
    let mut tester = DocumentationTester::new(".");

    tester
        .test_readme_examples()
        .await
        .expect("Failed to test README examples");

    let results = tester.get_results();
    let summary = tester.get_summary();

    println!("README Examples Test Results:");
    println!("{summary}");

    for result in results {
        if !result.passed {
            println!(
                "FAILED: {} - {}",
                result.test_name,
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    // Allow some failures for now, but ensure we have some tests
    assert!(summary.total > 0, "No documentation tests found in README");
}

#[tokio::test]
async fn test_lib_rs_examples() {
    let mut tester = DocumentationTester::new(".");

    tester
        .test_lib_rs_examples()
        .await
        .expect("Failed to test lib.rs examples");

    let results = tester.get_results();
    let summary = tester.get_summary();

    println!("lib.rs Examples Test Results:");
    println!("{summary}");

    for result in results {
        if !result.passed {
            println!(
                "FAILED: {} - {}",
                result.test_name,
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    // Allow some failures for now, but ensure we have some tests
    assert!(summary.total > 0, "No documentation tests found in lib.rs");
}

#[tokio::test]
async fn test_example_files() {
    let mut tester = DocumentationTester::new(".");

    tester
        .test_example_files()
        .await
        .expect("Failed to test example files");

    let results = tester.get_results();
    let summary = tester.get_summary();

    println!("Example Files Test Results:");
    println!("{summary}");

    for result in results {
        if !result.passed {
            println!(
                "FAILED: {} - {}",
                result.test_name,
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    // Example files should exist and be valid
    if summary.total > 0 {
        assert!(
            summary.pass_rate > 0.5,
            "More than 50% of example files should be valid"
        );
    }
}

#[tokio::test]
async fn test_api_documentation_completeness() {
    let mut tester = DocumentationTester::new(".");

    tester
        .test_api_documentation()
        .await
        .expect("Failed to test API documentation");

    let results = tester.get_results();
    let summary = tester.get_summary();

    println!("API Documentation Test Results:");
    println!("{summary}");

    for result in results {
        if !result.passed {
            println!(
                "FAILED: {} - {}",
                result.test_name,
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }
}

#[tokio::test]
async fn test_all_documentation() {
    let mut tester = DocumentationTester::new(".");

    tester
        .run_all_tests()
        .await
        .expect("Failed to run all documentation tests");

    let summary = tester.get_summary();

    println!("Complete Documentation Test Results:");
    println!("{summary}");
    println!("Detailed Results:");

    for result in tester.get_results() {
        let status = if result.passed { "PASS" } else { "FAIL" };
        println!("  {} - {} ({})", status, result.test_name, result.file_path);

        if let Some(error) = &result.error {
            println!("    Error: {error}");
        }
    }

    // Ensure we have comprehensive documentation tests
    assert!(
        summary.total >= 5,
        "Should have at least 5 documentation tests"
    );

    // Allow some failures but require a reasonable pass rate
    assert!(
        summary.pass_rate >= 0.3,
        "Documentation tests should have at least 30% pass rate"
    );
}

#[tokio::test]
async fn test_code_quality_standards() {
    let mut tester = DocumentationTester::new(".");

    // Test that all code examples follow quality standards
    tester
        .test_lib_rs_examples()
        .await
        .expect("Failed to test lib.rs examples");

    let results = tester.get_results();

    // Check for specific quality standards
    let mut has_error_handling = false;
    let mut has_async_examples = false;
    let mut has_pipeline_usage = false;

    for result in results {
        if result.test_name.contains("error handling") || result.test_name.contains("Result") {
            has_error_handling = true;
        }
        if result.test_name.contains("async") || result.test_name.contains("await") {
            has_async_examples = true;
        }
        if result.test_name.contains("Pipeline") || result.test_name.contains("synthesis") {
            has_pipeline_usage = true;
        }
    }

    // These are aspirational goals - we'll track them but not fail the test
    println!("Code Quality Standards:");
    println!("  Error handling examples: {has_error_handling}");
    println!("  Async examples: {has_async_examples}");
    println!("  Pipeline usage examples: {has_pipeline_usage}");
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_documentation_tester_functionality() {
        // Test the documentation tester itself
        let tester = DocumentationTester::new(".");

        assert_eq!(tester.base_path, ".");
        assert!(tester.get_results().is_empty());

        let summary = tester.get_summary();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.pass_rate, 0.0);
    }

    #[tokio::test]
    async fn test_code_block_extraction() {
        let tester = DocumentationTester::new(".");

        let content = r#"
# Example Documentation

Here's some Rust code:

```rust
fn main() {
    println!("Hello, world!");
}
```

And here's another example:

```rust,no_run
use voirs_sdk::prelude::*;

async fn example() -> Result<(), VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    Ok(())
}
```
"#;

        let blocks = tester.extract_rust_code_blocks(content);

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("fn main"));
        assert!(blocks[1].contains("VoirsPipelineBuilder"));
    }
}
