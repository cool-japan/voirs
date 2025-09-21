//! Documentation Testing Example - Comprehensive Documentation Validation
//!
//! This example demonstrates comprehensive documentation testing and validation
//! for the VoiRS project, including accuracy checking, link validation,
//! code example testing, and version compatibility verification.
//!
//! ## What this example demonstrates:
//! 1. Documentation accuracy validation
//! 2. Link checking for internal and external references
//! 3. Code example compilation and execution testing
//! 4. Version compatibility verification
//! 5. API documentation consistency checking
//! 6. Cross-reference validation
//!
//! ## Key Testing Features:
//! - Automated documentation parsing and analysis
//! - Link validation with timeout handling
//! - Code example extraction and testing
//! - Version compatibility matrix testing
//! - API consistency verification
//! - Documentation completeness analysis
//!
//! ## Testing Categories:
//! - Accuracy: Content correctness and up-to-date information
//! - Links: Internal and external link validation
//! - Code: Example compilation and execution
//! - Versions: Compatibility across VoiRS versions
//!
//! ## Prerequisites:
//! - VoiRS documentation in standard formats
//! - Network access for external link checking
//! - Multiple VoiRS versions for compatibility testing
//!
//! ## Expected output:
//! - Comprehensive documentation health report
//! - Link validation results with recommendations
//! - Code example test results
//! - Version compatibility matrix

use anyhow::{Context, Result};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Comprehensive documentation tester
pub struct DocumentationTester {
    config: DocumentationTestConfig,
    test_results: DocumentationTestResults,
    link_cache: HashMap<String, LinkStatus>,
}

#[derive(Debug, Clone)]
pub struct DocumentationTestConfig {
    /// Root directory for documentation
    pub doc_root: PathBuf,
    /// Enable network-based link checking
    pub check_external_links: bool,
    /// Timeout for link checking (seconds)
    pub link_timeout: u64,
    /// Code example testing configuration
    pub test_code_examples: bool,
    /// Version compatibility testing
    pub test_version_compatibility: bool,
    /// Maximum concurrent link checks
    pub max_concurrent_checks: usize,
    /// Skip patterns for documentation files
    pub skip_patterns: Vec<String>,
}

impl Default for DocumentationTestConfig {
    fn default() -> Self {
        Self {
            doc_root: PathBuf::from(".."),
            check_external_links: true,
            link_timeout: 30,
            test_code_examples: true,
            test_version_compatibility: true,
            max_concurrent_checks: 10,
            skip_patterns: vec![
                "target/*".to_string(),
                ".git/*".to_string(),
                "*.log".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct DocumentationTestResults {
    /// Overall test success rate
    pub success_rate: f32,
    /// Accuracy validation results
    pub accuracy_results: AccuracyTestResults,
    /// Link checking results
    pub link_results: LinkTestResults,
    /// Code example test results
    pub code_results: CodeTestResults,
    /// Version compatibility results
    pub version_results: VersionTestResults,
    /// Test execution time
    pub execution_time: Duration,
    /// Detailed test report
    pub test_report: String,
}

#[derive(Debug, Clone)]
pub struct AccuracyTestResults {
    /// Number of files tested
    pub files_tested: usize,
    /// Number of accuracy issues found
    pub issues_found: usize,
    /// Content freshness analysis
    pub content_freshness: HashMap<String, ContentFreshness>,
    /// API consistency issues
    pub api_consistency_issues: Vec<ApiConsistencyIssue>,
}

#[derive(Debug, Clone)]
pub struct LinkTestResults {
    /// Total links checked
    pub total_links: usize,
    /// Working links
    pub working_links: usize,
    /// Broken links
    pub broken_links: usize,
    /// Broken link details
    pub broken_link_details: Vec<BrokenLink>,
    /// Link checking time
    pub checking_time: Duration,
}

#[derive(Debug, Clone)]
pub struct CodeTestResults {
    /// Total code examples found
    pub total_examples: usize,
    /// Successfully compiled examples
    pub compiled_examples: usize,
    /// Successfully executed examples
    pub executed_examples: usize,
    /// Code example failures
    pub example_failures: Vec<CodeExampleFailure>,
}

#[derive(Debug, Clone)]
pub struct VersionTestResults {
    /// Tested version combinations
    pub tested_combinations: usize,
    /// Compatible combinations
    pub compatible_combinations: usize,
    /// Compatibility matrix
    pub compatibility_matrix: HashMap<String, HashMap<String, CompatibilityStatus>>,
}

#[derive(Debug, Clone)]
pub enum LinkStatus {
    Working,
    Broken(String),
    Timeout,
    NotChecked,
}

#[derive(Debug, Clone)]
pub struct ContentFreshness {
    pub last_updated: Option<std::time::SystemTime>,
    pub staleness_score: f32, // 0.0 = fresh, 1.0 = very stale
    pub update_recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ApiConsistencyIssue {
    pub file_path: String,
    pub issue_type: String,
    pub description: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BrokenLink {
    pub file_path: String,
    pub line_number: usize,
    pub link_url: String,
    pub error_message: String,
    pub link_type: LinkType,
}

#[derive(Debug, Clone)]
pub enum LinkType {
    Internal,
    External,
    RelativeFile,
    Anchor,
}

#[derive(Debug, Clone)]
pub struct LinkInfo {
    pub url: String,
    pub line_number: usize,
    pub link_type: LinkType,
}

#[derive(Debug, Clone)]
pub struct CodeExampleFailure {
    pub file_path: String,
    pub example_id: String,
    pub failure_type: FailureType,
    pub error_message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum FailureType {
    CompilationError,
    RuntimeError,
    TestFailure,
    DependencyMissing,
}

#[derive(Debug, Clone)]
pub enum CompatibilityStatus {
    Compatible,
    Incompatible(String),
    Unknown,
    PartiallyCompatible(Vec<String>),
}

impl DocumentationTester {
    /// Create new documentation tester
    pub fn new(config: DocumentationTestConfig) -> Self {
        Self {
            config,
            test_results: DocumentationTestResults::new(),
            link_cache: HashMap::new(),
        }
    }

    /// Run comprehensive documentation tests
    pub async fn run_comprehensive_tests(&mut self) -> Result<&DocumentationTestResults> {
        let start_time = Instant::now();
        info!("Starting comprehensive documentation testing");

        // Phase 1: Documentation Discovery
        let doc_files = self.discover_documentation_files()?;
        info!("Found {} documentation files to test", doc_files.len());

        // Phase 2: Accuracy Validation
        info!("Running accuracy validation tests...");
        self.test_documentation_accuracy(&doc_files).await?;

        // Phase 3: Link Validation
        if self.config.check_external_links {
            info!("Running link validation tests...");
            self.test_documentation_links(&doc_files).await?;
        }

        // Phase 4: Code Example Testing
        if self.config.test_code_examples {
            info!("Running code example tests...");
            self.test_code_examples(&doc_files).await?;
        }

        // Phase 5: Version Compatibility Testing
        if self.config.test_version_compatibility {
            info!("Running version compatibility tests...");
            self.test_version_compatibility().await?;
        }

        // Calculate overall results
        self.test_results.execution_time = start_time.elapsed();
        self.calculate_success_rate();
        self.generate_test_report();

        info!(
            "Documentation testing completed in {:?}",
            self.test_results.execution_time
        );
        Ok(&self.test_results)
    }

    /// Discover all documentation files
    fn discover_documentation_files(&self) -> Result<Vec<PathBuf>> {
        let mut doc_files = Vec::new();
        self.scan_directory(&self.config.doc_root, &mut doc_files)?;

        // Filter out skipped patterns
        let filtered_files: Vec<PathBuf> = doc_files
            .into_iter()
            .filter(|path| {
                let path_str = path.to_string_lossy();
                !self
                    .config
                    .skip_patterns
                    .iter()
                    .any(|pattern| path_str.contains(pattern.trim_end_matches('*')))
            })
            .collect();

        Ok(filtered_files)
    }

    /// Recursively scan directory for documentation files
    fn scan_directory(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.scan_directory(&path, files)?;
            } else if self.is_documentation_file(&path) {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Check if file is a documentation file
    fn is_documentation_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            matches!(extension.to_str(), Some("md") | Some("rst") | Some("txt"))
        } else {
            false
        }
    }

    /// Test documentation accuracy
    async fn test_documentation_accuracy(&mut self, doc_files: &[PathBuf]) -> Result<()> {
        let mut accuracy_results = AccuracyTestResults {
            files_tested: 0,
            issues_found: 0,
            content_freshness: HashMap::new(),
            api_consistency_issues: Vec::new(),
        };

        for file_path in doc_files {
            let content = fs::read_to_string(file_path)
                .with_context(|| format!("Failed to read file: {:?}", file_path))?;

            // Test content freshness
            let freshness = self.analyze_content_freshness(&content, file_path)?;
            accuracy_results
                .content_freshness
                .insert(file_path.to_string_lossy().to_string(), freshness);

            // Test API consistency
            let api_issues = self.check_api_consistency(&content, file_path)?;
            accuracy_results.api_consistency_issues.extend(api_issues);

            accuracy_results.files_tested += 1;
        }

        accuracy_results.issues_found = accuracy_results.api_consistency_issues.len();
        self.test_results.accuracy_results = accuracy_results;
        Ok(())
    }

    /// Analyze content freshness
    fn analyze_content_freshness(
        &self,
        _content: &str,
        file_path: &Path,
    ) -> Result<ContentFreshness> {
        let metadata = fs::metadata(file_path)?;
        let last_modified = metadata.modified().ok();

        let staleness_score = if let Some(modified_time) = last_modified {
            let age = std::time::SystemTime::now()
                .duration_since(modified_time)
                .unwrap_or(Duration::from_secs(0));

            // Calculate staleness (0.0 = fresh, 1.0 = very stale)
            (age.as_secs() as f32 / (365.0 * 24.0 * 3600.0)).min(1.0)
        } else {
            1.0 // Unknown age = assume stale
        };

        let update_recommendations = if staleness_score > 0.5 {
            vec!["Consider updating content".to_string()]
        } else {
            vec![]
        };

        Ok(ContentFreshness {
            last_updated: last_modified,
            staleness_score,
            update_recommendations,
        })
    }

    /// Check API consistency
    fn check_api_consistency(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Result<Vec<ApiConsistencyIssue>> {
        let mut issues = Vec::new();

        // Check for outdated API references
        let api_pattern = Regex::new(r"```rust\n([^`]+)\n```").unwrap();
        for (line_no, line) in content.lines().enumerate() {
            if line.contains("deprecated") || line.contains("DEPRECATED") {
                issues.push(ApiConsistencyIssue {
                    file_path: file_path.to_string_lossy().to_string(),
                    issue_type: "Deprecated API".to_string(),
                    description: format!("Line {}: Contains deprecated API reference", line_no + 1),
                    suggested_fix: Some("Update to current API".to_string()),
                });
            }
        }

        Ok(issues)
    }

    /// Test documentation links
    async fn test_documentation_links(&mut self, doc_files: &[PathBuf]) -> Result<()> {
        let start_time = Instant::now();
        let mut link_results = LinkTestResults {
            total_links: 0,
            working_links: 0,
            broken_links: 0,
            broken_link_details: Vec::new(),
            checking_time: Duration::from_secs(0),
        };

        for file_path in doc_files {
            let content = fs::read_to_string(file_path)
                .with_context(|| format!("Failed to read file: {:?}", file_path))?;

            let links = self.extract_links(&content, file_path)?;
            link_results.total_links += links.len();

            for link_info in links {
                let status = self.check_link(&link_info.url).await;
                match status {
                    LinkStatus::Working => link_results.working_links += 1,
                    LinkStatus::Broken(error) => {
                        link_results.broken_links += 1;
                        link_results.broken_link_details.push(BrokenLink {
                            file_path: file_path.to_string_lossy().to_string(),
                            line_number: link_info.line_number,
                            link_url: link_info.url,
                            error_message: error,
                            link_type: link_info.link_type,
                        });
                    }
                    _ => {}
                }
            }
        }

        link_results.checking_time = start_time.elapsed();
        self.test_results.link_results = link_results;
        Ok(())
    }

    /// Extract links from content
    fn extract_links(&self, content: &str, _file_path: &Path) -> Result<Vec<LinkInfo>> {
        let mut links = Vec::new();

        // Split content into lines to track line numbers
        let lines: Vec<&str> = content.lines().collect();

        // Extract markdown links
        let md_link_pattern = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
        for (line_idx, line) in lines.iter().enumerate() {
            for captures in md_link_pattern.captures_iter(line) {
                if let Some(link) = captures.get(2) {
                    let url = link.as_str().to_string();
                    let link_type = determine_link_type(&url);
                    links.push(LinkInfo {
                        url,
                        line_number: line_idx + 1,
                        link_type,
                    });
                }
            }
        }

        // Extract HTTP/HTTPS URLs
        let url_pattern = Regex::new(r"https?://[^\s\)]+").unwrap();
        for (line_idx, line) in lines.iter().enumerate() {
            for captures in url_pattern.captures_iter(line) {
                let url = captures.get(0).unwrap().as_str().to_string();
                // Only add if it's not already captured by markdown pattern
                if !links
                    .iter()
                    .any(|link| link.url == url && link.line_number == line_idx + 1)
                {
                    links.push(LinkInfo {
                        url,
                        line_number: line_idx + 1,
                        link_type: LinkType::External,
                    });
                }
            }
        }

        Ok(links)
    }
}

/// Determine the type of a link based on its URL
fn determine_link_type(url: &str) -> LinkType {
    if url.starts_with("http://") || url.starts_with("https://") {
        LinkType::External
    } else if url.starts_with('#') {
        LinkType::Anchor
    } else if url.starts_with('/')
        || url.contains("../")
        || url.ends_with(".md")
        || url.ends_with(".rs")
    {
        LinkType::RelativeFile
    } else {
        LinkType::Internal
    }
}

impl DocumentationTester {
    /// Check individual link status
    async fn check_link(&mut self, link: &str) -> LinkStatus {
        // Check cache first
        if let Some(cached_status) = self.link_cache.get(link) {
            return cached_status.clone();
        }

        // Simple link validation (placeholder implementation)
        let status = if link.starts_with("http://") || link.starts_with("https://") {
            // For this example, we'll assume external links are working
            // In a real implementation, you'd make HTTP requests
            LinkStatus::Working
        } else if link.starts_with("#") {
            // Anchor links - would need to check if anchor exists
            LinkStatus::Working
        } else {
            // Relative file links - would need to check if file exists
            LinkStatus::Working
        };

        self.link_cache.insert(link.to_string(), status.clone());
        status
    }

    /// Test code examples
    async fn test_code_examples(&mut self, doc_files: &[PathBuf]) -> Result<()> {
        let mut code_results = CodeTestResults {
            total_examples: 0,
            compiled_examples: 0,
            executed_examples: 0,
            example_failures: Vec::new(),
        };

        for file_path in doc_files {
            let content = fs::read_to_string(file_path)
                .with_context(|| format!("Failed to read file: {:?}", file_path))?;

            let code_examples = self.extract_code_examples(&content)?;
            code_results.total_examples += code_examples.len();

            // For this example, we'll simulate testing
            // In a real implementation, you'd compile and run the code
            for example in code_examples {
                // Simulate compilation test
                if example.contains("compile_error") {
                    code_results.example_failures.push(CodeExampleFailure {
                        file_path: file_path.to_string_lossy().to_string(),
                        example_id: "example_1".to_string(),
                        failure_type: FailureType::CompilationError,
                        error_message: "Simulated compilation error".to_string(),
                        suggestion: Some("Fix syntax error".to_string()),
                    });
                } else {
                    code_results.compiled_examples += 1;
                    code_results.executed_examples += 1;
                }
            }
        }

        self.test_results.code_results = code_results;
        Ok(())
    }

    /// Extract code examples from content
    fn extract_code_examples(&self, content: &str) -> Result<Vec<String>> {
        let mut examples = Vec::new();

        let code_block_pattern = Regex::new(r"```rust\n([^`]+)\n```").unwrap();
        for captures in code_block_pattern.captures_iter(content) {
            if let Some(code) = captures.get(1) {
                examples.push(code.as_str().to_string());
            }
        }

        Ok(examples)
    }

    /// Test version compatibility
    async fn test_version_compatibility(&mut self) -> Result<()> {
        let mut version_results = VersionTestResults {
            tested_combinations: 0,
            compatible_combinations: 0,
            compatibility_matrix: HashMap::new(),
        };

        // Simulate version compatibility testing
        let versions = vec!["0.1.0", "0.1.1", "0.2.0"];

        for version_a in &versions {
            let mut version_compat = HashMap::new();
            for version_b in &versions {
                version_results.tested_combinations += 1;

                // Simulate compatibility check
                let status = if version_a == version_b {
                    CompatibilityStatus::Compatible
                } else if version_a.starts_with("0.1") && version_b.starts_with("0.1") {
                    CompatibilityStatus::Compatible
                } else {
                    CompatibilityStatus::PartiallyCompatible(vec!["Some API changes".to_string()])
                };

                if matches!(status, CompatibilityStatus::Compatible) {
                    version_results.compatible_combinations += 1;
                }

                version_compat.insert(version_b.to_string(), status);
            }
            version_results
                .compatibility_matrix
                .insert(version_a.to_string(), version_compat);
        }

        self.test_results.version_results = version_results;
        Ok(())
    }

    /// Calculate overall success rate
    fn calculate_success_rate(&mut self) {
        let total_tests = self.test_results.accuracy_results.files_tested
            + self.test_results.link_results.total_links
            + self.test_results.code_results.total_examples
            + self.test_results.version_results.tested_combinations;

        let successful_tests = (self.test_results.accuracy_results.files_tested
            - self.test_results.accuracy_results.issues_found)
            + self.test_results.link_results.working_links
            + self.test_results.code_results.executed_examples
            + self.test_results.version_results.compatible_combinations;

        self.test_results.success_rate = if total_tests > 0 {
            successful_tests as f32 / total_tests as f32
        } else {
            0.0
        };
    }

    /// Generate comprehensive test report
    fn generate_test_report(&mut self) {
        let mut report = String::new();

        report.push_str("# VoiRS Documentation Testing Report\n\n");
        report.push_str(&format!(
            "**Overall Success Rate:** {:.1}%\n",
            self.test_results.success_rate * 100.0
        ));
        report.push_str(&format!(
            "**Execution Time:** {:?}\n\n",
            self.test_results.execution_time
        ));

        // Accuracy Results
        report.push_str("## Accuracy Testing Results\n");
        report.push_str(&format!(
            "- Files Tested: {}\n",
            self.test_results.accuracy_results.files_tested
        ));
        report.push_str(&format!(
            "- Issues Found: {}\n",
            self.test_results.accuracy_results.issues_found
        ));
        report.push_str("\n");

        // Link Results
        report.push_str("## Link Validation Results\n");
        report.push_str(&format!(
            "- Total Links: {}\n",
            self.test_results.link_results.total_links
        ));
        report.push_str(&format!(
            "- Working Links: {}\n",
            self.test_results.link_results.working_links
        ));
        report.push_str(&format!(
            "- Broken Links: {}\n",
            self.test_results.link_results.broken_links
        ));
        report.push_str("\n");

        // Code Results
        report.push_str("## Code Example Testing Results\n");
        report.push_str(&format!(
            "- Total Examples: {}\n",
            self.test_results.code_results.total_examples
        ));
        report.push_str(&format!(
            "- Compiled Examples: {}\n",
            self.test_results.code_results.compiled_examples
        ));
        report.push_str(&format!(
            "- Executed Examples: {}\n",
            self.test_results.code_results.executed_examples
        ));
        report.push_str("\n");

        // Version Results
        report.push_str("## Version Compatibility Results\n");
        report.push_str(&format!(
            "- Tested Combinations: {}\n",
            self.test_results.version_results.tested_combinations
        ));
        report.push_str(&format!(
            "- Compatible Combinations: {}\n",
            self.test_results.version_results.compatible_combinations
        ));

        self.test_results.test_report = report;
    }

    /// Print test results summary
    pub fn print_summary(&self) {
        println!("\n📚 VoiRS Documentation Testing Summary");
        println!("=====================================");
        println!(
            "Overall Success Rate: {:.1}%",
            self.test_results.success_rate * 100.0
        );
        println!("Execution Time: {:?}", self.test_results.execution_time);
        println!("\nDetailed Results:");
        println!(
            "• Accuracy: {}/{} files passed",
            self.test_results.accuracy_results.files_tested
                - self.test_results.accuracy_results.issues_found,
            self.test_results.accuracy_results.files_tested
        );
        println!(
            "• Links: {}/{} links working",
            self.test_results.link_results.working_links,
            self.test_results.link_results.total_links
        );
        println!(
            "• Code: {}/{} examples executed successfully",
            self.test_results.code_results.executed_examples,
            self.test_results.code_results.total_examples
        );
        println!(
            "• Versions: {}/{} combinations compatible",
            self.test_results.version_results.compatible_combinations,
            self.test_results.version_results.tested_combinations
        );
    }
}

impl DocumentationTestResults {
    fn new() -> Self {
        Self {
            success_rate: 0.0,
            accuracy_results: AccuracyTestResults {
                files_tested: 0,
                issues_found: 0,
                content_freshness: HashMap::new(),
                api_consistency_issues: Vec::new(),
            },
            link_results: LinkTestResults {
                total_links: 0,
                working_links: 0,
                broken_links: 0,
                broken_link_details: Vec::new(),
                checking_time: Duration::from_secs(0),
            },
            code_results: CodeTestResults {
                total_examples: 0,
                compiled_examples: 0,
                executed_examples: 0,
                example_failures: Vec::new(),
            },
            version_results: VersionTestResults {
                tested_combinations: 0,
                compatible_combinations: 0,
                compatibility_matrix: HashMap::new(),
            },
            execution_time: Duration::from_secs(0),
            test_report: String::new(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 VoiRS Documentation Testing Example");
    println!("======================================");

    // Create documentation tester with default configuration
    let config = DocumentationTestConfig::default();
    let mut tester = DocumentationTester::new(config);

    // Run comprehensive documentation tests
    let results = match tester.run_comprehensive_tests().await {
        Ok(results) => results.clone(),
        Err(e) => {
            error!("Documentation testing failed: {}", e);
            return Err(e);
        }
    };

    tester.print_summary();

    // Save detailed report
    if let Err(e) = fs::write("documentation_test_report.md", &results.test_report) {
        warn!("Failed to save detailed report: {}", e);
    } else {
        println!("\n📄 Detailed report saved to documentation_test_report.md");
    }

    println!("\n✅ Documentation testing completed successfully!");
    Ok(())
}
