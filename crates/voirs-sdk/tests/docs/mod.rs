use regex::Regex;
use std::fs;
use std::path::Path;
use voirs_sdk::prelude::*;

pub struct DocumentationTester {
    pub base_path: String,
    pub results: Vec<TestResult>,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub file_path: String,
    pub test_name: String,
    pub passed: bool,
    pub error: Option<String>,
}

impl DocumentationTester {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            results: Vec::new(),
        }
    }

    pub async fn run_all_tests(&mut self) -> Result<()> {
        self.test_readme_examples().await?;
        self.test_lib_rs_examples().await?;
        self.test_module_documentation().await?;
        self.test_example_files().await?;
        self.test_api_documentation().await?;

        Ok(())
    }

    pub async fn test_readme_examples(&mut self) -> Result<()> {
        let readme_path = Path::new(&self.base_path).join("README.md");

        if !readme_path.exists() {
            self.results.push(TestResult {
                file_path: readme_path.to_string_lossy().to_string(),
                test_name: "README existence".to_string(),
                passed: false,
                error: Some("README.md not found".to_string()),
            });
            return Ok(());
        }

        let content = fs::read_to_string(&readme_path)
            .map_err(|e| VoirsError::config_error(format!("Failed to read README.md: {e}")))?;

        let code_blocks = self.extract_rust_code_blocks(&content);

        for (index, code_block) in code_blocks.iter().enumerate() {
            let test_name = format!("README code block {}", index + 1);

            match self.validate_code_block(code_block).await {
                Ok(()) => {
                    self.results.push(TestResult {
                        file_path: readme_path.to_string_lossy().to_string(),
                        test_name,
                        passed: true,
                        error: None,
                    });
                }
                Err(e) => {
                    self.results.push(TestResult {
                        file_path: readme_path.to_string_lossy().to_string(),
                        test_name,
                        passed: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(())
    }

    pub async fn test_lib_rs_examples(&mut self) -> Result<()> {
        let lib_path = Path::new(&self.base_path).join("src/lib.rs");

        let content = fs::read_to_string(&lib_path)
            .map_err(|e| VoirsError::config_error(format!("Failed to read lib.rs: {e}")))?;

        let code_blocks = self.extract_rust_code_blocks(&content);

        for (index, code_block) in code_blocks.iter().enumerate() {
            let test_name = format!("lib.rs code block {}", index + 1);

            match self.validate_code_block(code_block).await {
                Ok(()) => {
                    self.results.push(TestResult {
                        file_path: lib_path.to_string_lossy().to_string(),
                        test_name,
                        passed: true,
                        error: None,
                    });
                }
                Err(e) => {
                    self.results.push(TestResult {
                        file_path: lib_path.to_string_lossy().to_string(),
                        test_name,
                        passed: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(())
    }

    pub async fn test_module_documentation(&mut self) -> Result<()> {
        let src_path = Path::new(&self.base_path).join("src");
        let rust_files = Self::find_rust_files(&src_path)?;

        for file_path in rust_files {
            let content = fs::read_to_string(&file_path).map_err(|e| {
                VoirsError::config_error(format!("Failed to read {}: {}", file_path.display(), e))
            })?;

            let code_blocks = self.extract_rust_code_blocks(&content);

            for (index, code_block) in code_blocks.iter().enumerate() {
                let test_name = format!("{} code block {}", file_path.display(), index + 1);

                match self.validate_code_block(code_block).await {
                    Ok(()) => {
                        self.results.push(TestResult {
                            file_path: file_path.to_string_lossy().to_string(),
                            test_name,
                            passed: true,
                            error: None,
                        });
                    }
                    Err(e) => {
                        self.results.push(TestResult {
                            file_path: file_path.to_string_lossy().to_string(),
                            test_name,
                            passed: false,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn test_example_files(&mut self) -> Result<()> {
        let examples_path = Path::new(&self.base_path).join("examples");

        if !examples_path.exists() {
            self.results.push(TestResult {
                file_path: examples_path.to_string_lossy().to_string(),
                test_name: "Examples directory existence".to_string(),
                passed: false,
                error: Some("Examples directory not found".to_string()),
            });
            return Ok(());
        }

        let example_files = Self::find_rust_files(&examples_path)?;

        for file_path in example_files {
            let test_name = format!("Example file: {}", file_path.display());

            match self.validate_example_file(&file_path).await {
                Ok(()) => {
                    self.results.push(TestResult {
                        file_path: file_path.to_string_lossy().to_string(),
                        test_name,
                        passed: true,
                        error: None,
                    });
                }
                Err(e) => {
                    self.results.push(TestResult {
                        file_path: file_path.to_string_lossy().to_string(),
                        test_name,
                        passed: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(())
    }

    pub async fn test_api_documentation(&mut self) -> Result<()> {
        let test_name = "API documentation completeness".to_string();

        match self.validate_api_documentation().await {
            Ok(()) => {
                self.results.push(TestResult {
                    file_path: "API documentation".to_string(),
                    test_name,
                    passed: true,
                    error: None,
                });
            }
            Err(e) => {
                self.results.push(TestResult {
                    file_path: "API documentation".to_string(),
                    test_name,
                    passed: false,
                    error: Some(e.to_string()),
                });
            }
        }

        Ok(())
    }

    pub fn extract_rust_code_blocks(&self, content: &str) -> Vec<String> {
        let mut blocks = Vec::new();

        // Pattern for all rust code blocks (with DOTALL flag for multiline)
        // This matches ```rust, ```rust,no_run, etc.
        let rust_code_regex = Regex::new(r"(?s)```rust(?:,\s*no_run)?\s*\n(.*?)\n```").unwrap();

        // Pattern for documentation comment code blocks (//! ```)
        let doc_code_block_regex =
            Regex::new(r"//!\s*```(?:rust,)?(?:\s*no_run)?\n((?://!.*\n)*?)//!\s*```").unwrap();

        // Extract regular markdown code blocks
        for cap in rust_code_regex.captures_iter(content) {
            if let Some(code) = cap.get(1) {
                blocks.push(code.as_str().to_string());
            }
        }

        // Extract documentation comment code blocks and clean them up
        for cap in doc_code_block_regex.captures_iter(content) {
            if let Some(code) = cap.get(1) {
                // Remove the "//! " prefix from each line
                let cleaned_code = code
                    .as_str()
                    .lines()
                    .map(|line| {
                        if let Some(stripped) = line.strip_prefix("//! ") {
                            stripped
                        } else if let Some(stripped) = line.strip_prefix("//!") {
                            stripped
                        } else {
                            line
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                blocks.push(cleaned_code);
            }
        }

        blocks
    }

    pub async fn validate_code_block(&self, code: &str) -> Result<()> {
        // Basic syntax validation
        if code.trim().is_empty() {
            return Err(VoirsError::config_error("Empty code block".to_string()));
        }

        // Check for basic Rust syntax patterns
        if !code.contains("fn ") && !code.contains("let ") && !code.contains("use ") {
            return Err(VoirsError::config_error(
                "Code block doesn't appear to be valid Rust".to_string(),
            ));
        }

        // Check for proper VoiRS SDK usage
        if (code.contains("voirs_sdk") || code.contains("VoirsPipeline"))
            && !code.contains("use voirs_sdk")
            && !code.contains("use crate::")
        {
            return Err(VoirsError::config_error(
                "Missing proper imports for VoiRS SDK".to_string(),
            ));
        }

        // Check for async/await patterns
        if (code.contains("async fn") || code.contains(".await"))
            && !code.contains("tokio::main")
            && !code.contains("async fn")
        {
            return Err(VoirsError::config_error(
                "Async code should have proper runtime setup".to_string(),
            ));
        }

        Ok(())
    }

    async fn validate_example_file(&self, file_path: &Path) -> Result<()> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| VoirsError::config_error(format!("Failed to read example file: {e}")))?;

        // Check for basic structure
        if !content.contains("fn main") {
            return Err(VoirsError::config_error(
                "Example file missing main function".to_string(),
            ));
        }

        // Check for proper imports
        if !content.contains("use voirs_sdk") {
            return Err(VoirsError::config_error(
                "Example file missing VoiRS SDK imports".to_string(),
            ));
        }

        // Check for error handling
        if !content.contains("Result") && !content.contains("?") {
            return Err(VoirsError::config_error(
                "Example file should demonstrate error handling".to_string(),
            ));
        }

        Ok(())
    }

    async fn validate_api_documentation(&self) -> Result<()> {
        // Check that all public APIs are documented
        let lib_path = Path::new(&self.base_path).join("src/lib.rs");
        let content = fs::read_to_string(&lib_path)
            .map_err(|e| VoirsError::config_error(format!("Failed to read lib.rs: {e}")))?;

        let public_items = self.find_public_items(&content);
        let documented_items = self.find_documented_items(&content);

        for item in &public_items {
            if !documented_items.contains(item) {
                return Err(VoirsError::config_error(format!(
                    "Public item '{item}' is not documented"
                )));
            }
        }

        Ok(())
    }

    fn find_public_items(&self, content: &str) -> Vec<String> {
        let mut items = Vec::new();
        let pub_regex =
            Regex::new(r"pub\s+(?:fn|struct|enum|trait|mod|type|const|static)\s+(\w+)").unwrap();

        for cap in pub_regex.captures_iter(content) {
            if let Some(item) = cap.get(1) {
                items.push(item.as_str().to_string());
            }
        }

        items
    }

    fn find_documented_items(&self, content: &str) -> Vec<String> {
        let mut items = Vec::new();
        let doc_regex = Regex::new(r"///.*\n(?:.*\n)*?\s*(?:pub\s+)?(?:fn|struct|enum|trait|mod|type|const|static)\s+(\w+)").unwrap();

        for cap in doc_regex.captures_iter(content) {
            if let Some(item) = cap.get(1) {
                items.push(item.as_str().to_string());
            }
        }

        items
    }

    fn find_rust_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        if dir.is_dir() {
            for entry in fs::read_dir(dir)
                .map_err(|e| VoirsError::config_error(format!("Failed to read directory: {e}")))?
            {
                let entry = entry.map_err(|e| {
                    VoirsError::config_error(format!("Failed to read directory entry: {e}"))
                })?;
                let path = entry.path();

                if path.is_dir() {
                    files.extend(Self::find_rust_files(&path)?);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    files.push(path);
                }
            }
        }

        Ok(files)
    }

    pub fn get_results(&self) -> &[TestResult] {
        &self.results
    }

    pub fn get_summary(&self) -> TestSummary {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        TestSummary {
            total,
            passed,
            failed,
            pass_rate: if total > 0 {
                passed as f64 / total as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
}

impl std::fmt::Display for TestSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Documentation Tests: {}/{} passed ({:.1}%)",
            self.passed,
            self.total,
            self.pass_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_documentation_tester_creation() {
        let tester = DocumentationTester::new(".");
        assert_eq!(tester.base_path, ".");
        assert!(tester.results.is_empty());
    }

    #[tokio::test]
    async fn test_code_block_extraction() {
        let content = r#"
Some text here.

```rust
fn main() {
    println!("Hello, world!");
}
```

More text.

```rust,no_run
use voirs_sdk::prelude::*;

async fn example() -> Result<(), VoirsError> {
    let pipeline = VoirsPipelineBuilder::new().build().await?;
    Ok(())
}
```
"#;

        let tester = DocumentationTester::new(".");
        let blocks = tester.extract_rust_code_blocks(content);

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("fn main"));
        assert!(blocks[1].contains("VoirsPipelineBuilder"));
    }

    #[tokio::test]
    async fn test_code_block_validation() {
        let tester = DocumentationTester::new(".");

        // Valid code block
        let valid_code = r#"
use voirs_sdk::prelude::*;

fn main() {
    println!("Hello, world!");
}
"#;

        assert!(tester.validate_code_block(valid_code).await.is_ok());

        // Invalid code block (empty)
        let invalid_code = "";
        assert!(tester.validate_code_block(invalid_code).await.is_err());
    }

    #[tokio::test]
    async fn test_summary_display() {
        let summary = TestSummary {
            total: 10,
            passed: 8,
            failed: 2,
            pass_rate: 0.8,
        };

        let display = format!("{summary}");
        assert!(display.contains("8/10 passed"));
        assert!(display.contains("80.0%"));
    }
}
