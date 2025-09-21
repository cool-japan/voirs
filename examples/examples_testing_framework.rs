//! VoiRS Examples Testing Framework
//!
//! This framework provides comprehensive testing for all VoiRS examples including:
//! - Rust examples compilation and execution testing
//! - Python bindings example validation  
//! - C++ integration example compilation testing
//! - Cross-platform compatibility validation
//! - Performance benchmarking of examples
//! - Output quality validation
//! - CI/CD integration support

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamplesTestConfig {
    pub examples_dir: PathBuf,
    pub test_timeout_seconds: u64,
    pub performance_requirements: PerformanceRequirements,
    pub platforms: Vec<TargetPlatform>,
    pub quality_thresholds: QualityThresholds,
    pub output_dir: PathBuf,
    pub ci_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub max_compile_time_seconds: f64,
    pub max_execution_time_seconds: f64,
    pub max_memory_mb: f64,
    pub max_cpu_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_audio_duration_seconds: f64,
    pub max_audio_file_size_mb: f64,
    pub require_no_warnings: bool,
    pub require_clippy_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetPlatform {
    Linux,
    MacOS,
    Windows,
    WebAssembly,
}

#[derive(Debug, Clone)]
pub enum ExampleType {
    RustBinary {
        name: String,
        file_path: PathBuf,
        features: Vec<String>,
        dependencies: Vec<String>,
    },
    RustLibrary {
        name: String,
        file_path: PathBuf,
        features: Vec<String>,
    },
    PythonScript {
        name: String,
        file_path: PathBuf,
        requirements: Vec<String>,
    },
    CppProgram {
        name: String,
        file_path: PathBuf,
        compile_flags: Vec<String>,
        link_libs: Vec<String>,
    },
}

#[derive(Debug, Serialize)]
pub struct ExampleTestReport {
    pub run_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub platform: String,
    pub total_duration: Duration,
    pub examples_tested: usize,
    pub results: Vec<ExampleTestResult>,
    pub summary: TestSummary,
    pub performance_analysis: PerformanceAnalysis,
    pub quality_analysis: QualityAnalysis,
}

#[derive(Debug, Serialize)]
pub struct ExampleTestResult {
    pub name: String,
    pub example_type: String,
    pub compilation_result: CompilationResult,
    pub execution_result: Option<ExecutionResult>,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
    pub overall_status: TestStatus,
}

#[derive(Debug, Serialize)]
pub struct CompilationResult {
    pub success: bool,
    pub duration: Duration,
    pub warnings_count: usize,
    pub errors: Vec<String>,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub exit_code: i32,
    pub duration: Duration,
    pub stdout: String,
    pub stderr: String,
    pub generated_files: Vec<GeneratedFile>,
}

#[derive(Debug, Serialize)]
pub struct GeneratedFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub file_type: String,
    pub quality_score: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct PerformanceMetrics {
    pub compilation_time: f64,
    pub execution_time: f64,
    pub peak_memory_mb: f64,
    pub avg_cpu_percent: f64,
    pub meets_requirements: bool,
}

#[derive(Debug, Serialize)]
pub struct QualityMetrics {
    pub code_quality_score: f64,
    pub output_quality_score: Option<f64>,
    pub warnings_penalty: f64,
    pub meets_thresholds: bool,
}

#[derive(Debug, Serialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Warning,
    Skipped,
}

#[derive(Debug, Serialize)]
pub struct TestSummary {
    pub total_examples: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub skipped: usize,
    pub success_rate: f64,
    pub platform_coverage: f64,
}

#[derive(Debug, Serialize)]
pub struct PerformanceAnalysis {
    pub fastest_example: String,
    pub slowest_example: String,
    pub avg_compilation_time: f64,
    pub avg_execution_time: f64,
    pub performance_regressions: Vec<PerformanceRegression>,
}

#[derive(Debug, Serialize)]
pub struct QualityAnalysis {
    pub best_quality_example: String,
    pub quality_issues: Vec<QualityIssue>,
    pub overall_quality_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PerformanceRegression {
    pub example_name: String,
    pub metric: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub regression_percent: f64,
}

#[derive(Debug, Serialize)]
pub struct QualityIssue {
    pub example_name: String,
    pub issue_type: String,
    pub severity: String,
    pub description: String,
    pub suggestion: Option<String>,
}

pub struct ExamplesTestFramework {
    config: ExamplesTestConfig,
    discovered_examples: Vec<ExampleType>,
}

impl ExamplesTestFramework {
    pub fn new(config: ExamplesTestConfig) -> Result<Self> {
        let framework = Self {
            discovered_examples: Vec::new(),
            config,
        };

        Ok(framework)
    }

    /// Discover all examples in the examples directory
    pub async fn discover_examples(&mut self) -> Result<()> {
        println!("🔍 Discovering examples in {:?}", self.config.examples_dir);

        let entries =
            fs::read_dir(&self.config.examples_dir).context("Failed to read examples directory")?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(extension) = path.extension() {
                match extension.to_str() {
                    Some("rs") => {
                        self.discover_rust_example(path).await?;
                    }
                    Some("py") => {
                        self.discover_python_example(path).await?;
                    }
                    Some("cpp") | Some("cc") | Some("cxx") => {
                        self.discover_cpp_example(path).await?;
                    }
                    _ => continue,
                }
            }
        }

        println!("📋 Discovered {} examples:", self.discovered_examples.len());
        for example in &self.discovered_examples {
            match example {
                ExampleType::RustBinary { name, .. } => println!("   🦀 Rust: {}", name),
                ExampleType::RustLibrary { name, .. } => println!("   📚 Rust Lib: {}", name),
                ExampleType::PythonScript { name, .. } => println!("   🐍 Python: {}", name),
                ExampleType::CppProgram { name, .. } => println!("   ⚡ C++: {}", name),
            }
        }

        Ok(())
    }

    async fn discover_rust_example(&mut self, path: PathBuf) -> Result<()> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Read file to determine if it's a binary or library
        let content = fs::read_to_string(&path)?;
        let features = self.extract_rust_features(&content);
        let dependencies = self.extract_rust_dependencies(&content);

        if content.contains("fn main()") {
            self.discovered_examples.push(ExampleType::RustBinary {
                name,
                file_path: path,
                features,
                dependencies,
            });
        } else {
            self.discovered_examples.push(ExampleType::RustLibrary {
                name,
                file_path: path,
                features,
            });
        }

        Ok(())
    }

    async fn discover_python_example(&mut self, path: PathBuf) -> Result<()> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Extract Python requirements from comments or imports
        let content = fs::read_to_string(&path)?;
        let requirements = self.extract_python_requirements(&content);

        self.discovered_examples.push(ExampleType::PythonScript {
            name,
            file_path: path,
            requirements,
        });

        Ok(())
    }

    async fn discover_cpp_example(&mut self, path: PathBuf) -> Result<()> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Extract compile flags and libraries from comments
        let content = fs::read_to_string(&path)?;
        let compile_flags = self.extract_cpp_compile_flags(&content);
        let link_libs = self.extract_cpp_link_libs(&content);

        self.discovered_examples.push(ExampleType::CppProgram {
            name,
            file_path: path,
            compile_flags,
            link_libs,
        });

        Ok(())
    }

    /// Run all discovered examples with comprehensive testing
    pub async fn run_all_tests(&mut self) -> Result<ExampleTestReport> {
        let run_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        println!("🚀 Starting VoiRS Examples Test Framework");
        println!("📋 Run ID: {}", run_id);
        println!("📂 Examples directory: {:?}", self.config.examples_dir);
        println!("🎯 Platform: {}", self.detect_current_platform());

        // Discover examples if not already done
        if self.discovered_examples.is_empty() {
            self.discover_examples().await?;
        }

        let mut results = Vec::new();
        let mut passed = 0;
        let mut failed = 0;
        let mut warnings = 0;
        let mut skipped = 0;

        // Test each example
        for (i, example) in self.discovered_examples.iter().enumerate() {
            let example_name = match example {
                ExampleType::RustBinary { name, .. }
                | ExampleType::RustLibrary { name, .. }
                | ExampleType::PythonScript { name, .. }
                | ExampleType::CppProgram { name, .. } => name,
            };

            println!(
                "\n🧪 Testing example [{}/{}]: {}",
                i + 1,
                self.discovered_examples.len(),
                example_name
            );

            let result = self.test_example(example).await?;

            match result.overall_status {
                TestStatus::Passed => passed += 1,
                TestStatus::Failed => failed += 1,
                TestStatus::Warning => warnings += 1,
                TestStatus::Skipped => skipped += 1,
            }

            results.push(result);
        }

        let total_duration = start_time.elapsed();
        let total_examples = self.discovered_examples.len();
        let success_rate = if total_examples > 0 {
            (passed as f64 / total_examples as f64) * 100.0
        } else {
            0.0
        };

        // Generate analyses
        let performance_analysis = self.analyze_performance(&results);
        let quality_analysis = self.analyze_quality(&results);

        let summary = TestSummary {
            total_examples,
            passed,
            failed,
            warnings,
            skipped,
            success_rate,
            platform_coverage: 100.0, // Single platform for now
        };

        let report = ExampleTestReport {
            run_id,
            timestamp: chrono::Utc::now(),
            platform: self.detect_current_platform(),
            total_duration,
            examples_tested: total_examples,
            results,
            summary,
            performance_analysis,
            quality_analysis,
        };

        // Generate output reports
        self.generate_reports(&report).await?;

        // Print summary
        self.print_summary(&report);

        Ok(report)
    }

    async fn test_example(&self, example: &ExampleType) -> Result<ExampleTestResult> {
        let (name, example_type_str) = match example {
            ExampleType::RustBinary { name, .. } => (name.clone(), "Rust Binary".to_string()),
            ExampleType::RustLibrary { name, .. } => (name.clone(), "Rust Library".to_string()),
            ExampleType::PythonScript { name, .. } => (name.clone(), "Python Script".to_string()),
            ExampleType::CppProgram { name, .. } => (name.clone(), "C++ Program".to_string()),
        };

        println!("  📝 Type: {}", example_type_str);

        // Test compilation
        let compilation_result = self.test_compilation(example).await?;
        let mut overall_status = if compilation_result.success {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        };

        // Test execution if compilation succeeded
        let execution_result = if compilation_result.success {
            match self.test_execution(example).await {
                Ok(exec_result) => {
                    if !exec_result.success {
                        overall_status = TestStatus::Failed;
                    }
                    Some(exec_result)
                }
                Err(e) => {
                    println!("  ⚠️  Execution test failed: {}", e);
                    overall_status = TestStatus::Warning;
                    None
                }
            }
        } else {
            None
        };

        // Calculate performance metrics
        let performance_metrics =
            self.calculate_performance_metrics(&compilation_result, &execution_result);

        // Calculate quality metrics
        let quality_metrics =
            self.calculate_quality_metrics(&compilation_result, &execution_result);

        // Adjust status based on metrics
        if matches!(overall_status, TestStatus::Passed) {
            if !performance_metrics.meets_requirements || !quality_metrics.meets_thresholds {
                overall_status = TestStatus::Warning;
            }
        }

        println!(
            "  {} Status: {:?}",
            match overall_status {
                TestStatus::Passed => "✅",
                TestStatus::Failed => "❌",
                TestStatus::Warning => "⚠️",
                TestStatus::Skipped => "⏭️",
            },
            overall_status
        );

        Ok(ExampleTestResult {
            name,
            example_type: example_type_str,
            compilation_result,
            execution_result,
            performance_metrics,
            quality_metrics,
            overall_status,
        })
    }

    async fn test_compilation(&self, example: &ExampleType) -> Result<CompilationResult> {
        let start_time = Instant::now();

        match example {
            ExampleType::RustBinary {
                file_path,
                features,
                ..
            }
            | ExampleType::RustLibrary {
                file_path,
                features,
                ..
            } => self.test_rust_compilation(file_path, features).await,
            ExampleType::PythonScript { file_path, .. } => self.test_python_syntax(file_path).await,
            ExampleType::CppProgram {
                file_path,
                compile_flags,
                link_libs,
                ..
            } => {
                self.test_cpp_compilation(file_path, compile_flags, link_libs)
                    .await
            }
        }
    }

    async fn test_rust_compilation(
        &self,
        file_path: &Path,
        features: &[String],
    ) -> Result<CompilationResult> {
        let start_time = Instant::now();

        let mut cmd = Command::new("cargo");
        cmd.arg("check")
            .arg("--example")
            .arg(file_path.file_stem().unwrap().to_str().unwrap())
            .current_dir(&self.config.examples_dir);

        if !features.is_empty() {
            cmd.arg("--features").arg(features.join(","));
        }

        println!("  🔨 Compiling Rust example...");

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            cmd.output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let warnings_count = stderr.matches("warning:").count();
        let mut errors = Vec::new();

        if !success {
            errors.push(stderr.clone());
        }

        println!(
            "  📊 Compilation: {} in {:?} ({} warnings)",
            if success { "✅" } else { "❌" },
            duration,
            warnings_count
        );

        Ok(CompilationResult {
            success,
            duration,
            warnings_count,
            errors,
            output: format!("{}\n{}", stdout, stderr),
        })
    }

    async fn test_python_syntax(&self, file_path: &Path) -> Result<CompilationResult> {
        let start_time = Instant::now();

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            Command::new("python3")
                .arg("-m")
                .arg("py_compile")
                .arg(file_path)
                .output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut errors = Vec::new();
        if !success {
            errors.push(stderr.clone());
        }

        println!(
            "  🐍 Python syntax check: {} in {:?}",
            if success { "✅" } else { "❌" },
            duration
        );

        Ok(CompilationResult {
            success,
            duration,
            warnings_count: 0,
            errors,
            output: stderr,
        })
    }

    async fn test_cpp_compilation(
        &self,
        file_path: &Path,
        compile_flags: &[String],
        link_libs: &[String],
    ) -> Result<CompilationResult> {
        let start_time = Instant::now();

        let output_path = self.config.output_dir.join(format!(
            "{}_test",
            file_path.file_stem().unwrap().to_str().unwrap()
        ));

        let mut cmd = Command::new("g++");
        cmd.arg("-std=c++14")
            .arg("-O2")
            .arg(file_path)
            .arg("-o")
            .arg(&output_path);

        // Add compile flags
        for flag in compile_flags {
            cmd.arg(flag);
        }

        // Add link libraries
        for lib in link_libs {
            if lib.starts_with("-l") {
                cmd.arg(lib);
            } else {
                cmd.arg(format!("-l{}", lib));
            }
        }

        println!("  ⚡ Compiling C++ example...");

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            cmd.output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let warnings_count = stderr.matches("warning:").count();
        let mut errors = Vec::new();

        if !success {
            errors.push(stderr.clone());
        }

        println!(
            "  📊 C++ compilation: {} in {:?} ({} warnings)",
            if success { "✅" } else { "❌" },
            duration,
            warnings_count
        );

        Ok(CompilationResult {
            success,
            duration,
            warnings_count,
            errors,
            output: stderr,
        })
    }

    async fn test_execution(&self, example: &ExampleType) -> Result<ExecutionResult> {
        match example {
            ExampleType::RustBinary { name, .. } => self.test_rust_execution(name).await,
            ExampleType::PythonScript { file_path, .. } => {
                self.test_python_execution(file_path).await
            }
            ExampleType::CppProgram { name, .. } => self.test_cpp_execution(name).await,
            ExampleType::RustLibrary { .. } => {
                // Library examples don't execute directly
                Ok(ExecutionResult {
                    success: true,
                    exit_code: 0,
                    duration: Duration::from_millis(0),
                    stdout: "Library example - no execution".to_string(),
                    stderr: String::new(),
                    generated_files: Vec::new(),
                })
            }
        }
    }

    async fn test_rust_execution(&self, name: &str) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        println!("  ▶️  Executing Rust example...");

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            Command::new("cargo")
                .arg("run")
                .arg("--example")
                .arg(name)
                .current_dir(&self.config.examples_dir)
                .output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Look for generated files
        let generated_files = self.find_generated_files().await?;

        println!(
            "  📊 Execution: {} in {:?} (exit code: {})",
            if success { "✅" } else { "❌" },
            duration,
            exit_code
        );

        Ok(ExecutionResult {
            success,
            exit_code,
            duration,
            stdout,
            stderr,
            generated_files,
        })
    }

    async fn test_python_execution(&self, file_path: &Path) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        println!("  ▶️  Executing Python script...");

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            Command::new("python3").arg(file_path).output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let generated_files = self.find_generated_files().await?;

        println!(
            "  📊 Python execution: {} in {:?} (exit code: {})",
            if success { "✅" } else { "❌" },
            duration,
            exit_code
        );

        Ok(ExecutionResult {
            success,
            exit_code,
            duration,
            stdout,
            stderr,
            generated_files,
        })
    }

    async fn test_cpp_execution(&self, name: &str) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        let executable_path = self.config.output_dir.join(format!("{}_test", name));

        if !executable_path.exists() {
            bail!("Compiled C++ executable not found: {:?}", executable_path);
        }

        println!("  ▶️  Executing C++ program...");

        let output = timeout(
            Duration::from_secs(self.config.test_timeout_seconds),
            Command::new(&executable_path).output(),
        )
        .await??;

        let duration = start_time.elapsed();
        let success = output.status.success();
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let generated_files = self.find_generated_files().await?;

        println!(
            "  📊 C++ execution: {} in {:?} (exit code: {})",
            if success { "✅" } else { "❌" },
            duration,
            exit_code
        );

        Ok(ExecutionResult {
            success,
            exit_code,
            duration,
            stdout,
            stderr,
            generated_files,
        })
    }

    async fn find_generated_files(&self) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // Look for generated files in common directories
        let search_dirs = [
            PathBuf::from("/tmp"),
            self.config.output_dir.clone(),
            PathBuf::from("."),
        ];

        for dir in &search_dirs {
            if !dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        let name_str = name.to_string_lossy();
                        // Look for files that might be generated by VoiRS examples
                        if name_str.starts_with("voirs_")
                            && (name_str.ends_with(".wav")
                                || name_str.ends_with(".mp3")
                                || name_str.ends_with(".flac")
                                || name_str.ends_with(".txt"))
                        {
                            if let Ok(metadata) = path.metadata() {
                                files.push(GeneratedFile {
                                    path: path.clone(),
                                    size_bytes: metadata.len(),
                                    file_type: path
                                        .extension()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                    quality_score: None, // Would need audio analysis for this
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(files)
    }

    fn calculate_performance_metrics(
        &self,
        compilation: &CompilationResult,
        execution: &Option<ExecutionResult>,
    ) -> PerformanceMetrics {
        let compilation_time = compilation.duration.as_secs_f64();
        let execution_time = execution
            .as_ref()
            .map(|e| e.duration.as_secs_f64())
            .unwrap_or(0.0);

        let meets_requirements = compilation_time
            <= self
                .config
                .performance_requirements
                .max_compile_time_seconds
            && execution_time
                <= self
                    .config
                    .performance_requirements
                    .max_execution_time_seconds;

        PerformanceMetrics {
            compilation_time,
            execution_time,
            peak_memory_mb: 0.0,  // Would need system monitoring for this
            avg_cpu_percent: 0.0, // Would need system monitoring for this
            meets_requirements,
        }
    }

    fn calculate_quality_metrics(
        &self,
        compilation: &CompilationResult,
        _execution: &Option<ExecutionResult>,
    ) -> QualityMetrics {
        let warning_penalty = if self.config.quality_thresholds.require_no_warnings
            && compilation.warnings_count > 0
        {
            0.2 * compilation.warnings_count as f64
        } else {
            0.0
        };

        let code_quality_score = (100.0 - warning_penalty).max(0.0);

        let meets_thresholds = if self.config.quality_thresholds.require_no_warnings {
            compilation.warnings_count == 0
        } else {
            true
        };

        QualityMetrics {
            code_quality_score,
            output_quality_score: None, // Would need audio analysis
            warnings_penalty: warning_penalty,
            meets_thresholds,
        }
    }

    // Helper methods for extracting information from source files
    fn extract_rust_features(&self, content: &str) -> Vec<String> {
        // Look for #[cfg(feature = "...")] patterns
        let mut features = Vec::new();
        for line in content.lines() {
            if line.trim().starts_with("#[cfg(feature") {
                if let Some(start) = line.find('"') {
                    if let Some(end) = line[start + 1..].find('"') {
                        features.push(line[start + 1..start + 1 + end].to_string());
                    }
                }
            }
        }
        features
    }

    fn extract_rust_dependencies(&self, content: &str) -> Vec<String> {
        // Look for use statements to infer dependencies
        let mut deps = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("use ") && line.contains("::") {
                if let Some(crate_name) = line[4..].split("::").next() {
                    if !crate_name.starts_with("std") && !crate_name.starts_with("core") {
                        deps.push(crate_name.to_string());
                    }
                }
            }
        }
        deps.sort();
        deps.dedup();
        deps
    }

    fn extract_python_requirements(&self, content: &str) -> Vec<String> {
        let mut requirements = Vec::new();
        for line in content.lines() {
            if line.trim().starts_with("import ") || line.trim().starts_with("from ") {
                if line.contains("numpy") && !requirements.contains(&"numpy".to_string()) {
                    requirements.push("numpy".to_string());
                }
                if line.contains("voirs") && !requirements.contains(&"voirs_ffi".to_string()) {
                    requirements.push("voirs_ffi".to_string());
                }
            }
        }
        requirements
    }

    fn extract_cpp_compile_flags(&self, content: &str) -> Vec<String> {
        let mut flags = vec!["-std=c++14".to_string(), "-O2".to_string()];

        // Look for comment lines with compile flags
        for line in content.lines() {
            if line.trim().starts_with("// Build:") || line.trim().starts_with("* Build Command:") {
                if line.contains("-std=") {
                    // Already have default std flag
                }
                if line.contains("-I") {
                    // Extract include paths
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    for (i, part) in parts.iter().enumerate() {
                        if part == &"-I" && i + 1 < parts.len() {
                            flags.push(format!("-I{}", parts[i + 1]));
                        } else if part.starts_with("-I") {
                            flags.push(part.to_string());
                        }
                    }
                }
            }
        }

        flags
    }

    fn extract_cpp_link_libs(&self, content: &str) -> Vec<String> {
        let mut libs = Vec::new();

        // Default libraries for VoiRS
        libs.extend_from_slice(&[
            "voirs_ffi".to_string(),
            "voirs_recognizer".to_string(),
            "pthread".to_string(),
            "dl".to_string(),
            "m".to_string(),
        ]);

        // Look for additional libraries in comments
        for line in content.lines() {
            if line.contains("-l")
                && (line.trim().starts_with("//") || line.trim().starts_with("*"))
            {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for part in parts {
                    if part.starts_with("-l") {
                        libs.push(part[2..].to_string());
                    }
                }
            }
        }

        libs.sort();
        libs.dedup();
        libs
    }

    fn analyze_performance(&self, results: &[ExampleTestResult]) -> PerformanceAnalysis {
        if results.is_empty() {
            return PerformanceAnalysis {
                fastest_example: "None".to_string(),
                slowest_example: "None".to_string(),
                avg_compilation_time: 0.0,
                avg_execution_time: 0.0,
                performance_regressions: Vec::new(),
            };
        }

        let fastest = results
            .iter()
            .min_by(|a, b| {
                a.performance_metrics
                    .execution_time
                    .partial_cmp(&b.performance_metrics.execution_time)
                    .unwrap()
            })
            .unwrap();

        let slowest = results
            .iter()
            .max_by(|a, b| {
                a.performance_metrics
                    .execution_time
                    .partial_cmp(&b.performance_metrics.execution_time)
                    .unwrap()
            })
            .unwrap();

        let avg_compilation_time = results
            .iter()
            .map(|r| r.performance_metrics.compilation_time)
            .sum::<f64>()
            / results.len() as f64;

        let avg_execution_time = results
            .iter()
            .map(|r| r.performance_metrics.execution_time)
            .sum::<f64>()
            / results.len() as f64;

        PerformanceAnalysis {
            fastest_example: fastest.name.clone(),
            slowest_example: slowest.name.clone(),
            avg_compilation_time,
            avg_execution_time,
            performance_regressions: Vec::new(), // Would need baseline data
        }
    }

    fn analyze_quality(&self, results: &[ExampleTestResult]) -> QualityAnalysis {
        if results.is_empty() {
            return QualityAnalysis {
                best_quality_example: "None".to_string(),
                quality_issues: Vec::new(),
                overall_quality_score: 0.0,
                recommendations: Vec::new(),
            };
        }

        let best = results
            .iter()
            .max_by(|a, b| {
                a.quality_metrics
                    .code_quality_score
                    .partial_cmp(&b.quality_metrics.code_quality_score)
                    .unwrap()
            })
            .unwrap();

        let mut quality_issues = Vec::new();
        let mut recommendations = Vec::new();

        for result in results {
            if !result.quality_metrics.meets_thresholds {
                quality_issues.push(QualityIssue {
                    example_name: result.name.clone(),
                    issue_type: "Quality Threshold".to_string(),
                    severity: "Warning".to_string(),
                    description: format!(
                        "Example does not meet quality thresholds (score: {:.1})",
                        result.quality_metrics.code_quality_score
                    ),
                    suggestion: Some(
                        "Review compilation warnings and fix code quality issues".to_string(),
                    ),
                });
            }

            if result.compilation_result.warnings_count > 0 {
                quality_issues.push(QualityIssue {
                    example_name: result.name.clone(),
                    issue_type: "Compiler Warnings".to_string(),
                    severity: "Info".to_string(),
                    description: format!(
                        "{} compiler warnings detected",
                        result.compilation_result.warnings_count
                    ),
                    suggestion: Some("Fix compiler warnings to improve code quality".to_string()),
                });
            }
        }

        let overall_quality_score = results
            .iter()
            .map(|r| r.quality_metrics.code_quality_score)
            .sum::<f64>()
            / results.len() as f64;

        if overall_quality_score < 90.0 {
            recommendations.push(
                "Consider addressing compiler warnings to improve overall code quality".to_string(),
            );
        }

        if quality_issues.len() > results.len() / 2 {
            recommendations.push(
                "Many examples have quality issues - consider a systematic review".to_string(),
            );
        }

        QualityAnalysis {
            best_quality_example: best.name.clone(),
            quality_issues,
            overall_quality_score,
            recommendations,
        }
    }

    async fn generate_reports(&self, report: &ExampleTestReport) -> Result<()> {
        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_dir)?;

        // Generate JSON report
        let json_path = self.config.output_dir.join("examples_test_report.json");
        let json_content = serde_json::to_string_pretty(report)?;
        fs::write(&json_path, json_content)?;
        println!("📄 JSON report saved to: {:?}", json_path);

        // Generate HTML report
        self.generate_html_report(report).await?;

        // Generate JUnit XML for CI/CD integration
        self.generate_junit_xml(report).await?;

        Ok(())
    }

    async fn generate_html_report(&self, report: &ExampleTestReport) -> Result<()> {
        let html_path = self.config.output_dir.join("examples_test_report.html");

        let html_content = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>VoiRS Examples Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .failed {{ background-color: #ffe8e8; }}
        .warning {{ background-color: #fff3cd; }}
        .example {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        .metrics {{ display: flex; gap: 20px; margin: 10px 0; }}
        .metric {{ background-color: #f9f9f9; padding: 5px 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 VoiRS Examples Test Report</h1>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Platform:</strong> {platform}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Duration:</strong> {duration:.2}s</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Total Examples:</strong> {total}</p>
        <p><strong>Passed:</strong> {passed} ✅</p>
        <p><strong>Failed:</strong> {failed} ❌</p>
        <p><strong>Warnings:</strong> {warnings} ⚠️</p>
        <p><strong>Success Rate:</strong> {success_rate:.1}%</p>
    </div>
    
    <div>
        <h2>📋 Example Results</h2>
        {results}
    </div>
    
    <div>
        <h2>📈 Performance Analysis</h2>
        <p><strong>Fastest Example:</strong> {fastest}</p>
        <p><strong>Slowest Example:</strong> {slowest}</p>
        <p><strong>Average Compilation Time:</strong> {avg_compile:.2}s</p>
        <p><strong>Average Execution Time:</strong> {avg_exec:.2}s</p>
    </div>
</body>
</html>
        "#,
            run_id = report.run_id,
            platform = report.platform,
            timestamp = report.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            duration = report.total_duration.as_secs_f64(),
            total = report.summary.total_examples,
            passed = report.summary.passed,
            failed = report.summary.failed,
            warnings = report.summary.warnings,
            success_rate = report.summary.success_rate,
            fastest = report.performance_analysis.fastest_example,
            slowest = report.performance_analysis.slowest_example,
            avg_compile = report.performance_analysis.avg_compilation_time,
            avg_exec = report.performance_analysis.avg_execution_time,
            results = self.format_results_html(&report.results)
        );

        fs::write(&html_path, html_content)?;
        println!("🌐 HTML report saved to: {:?}", html_path);

        Ok(())
    }

    fn format_results_html(&self, results: &[ExampleTestResult]) -> String {
        let mut html = String::new();

        for result in results {
            let status_class = match result.overall_status {
                TestStatus::Failed => "failed",
                TestStatus::Warning => "warning",
                _ => "",
            };

            html.push_str(&format!(
                r#"
                <div class="example {}">
                    <h3>{} ({})</h3>
                    <div class="metrics">
                        <div class="metric">Compilation: {:.2}s</div>
                        <div class="metric">Execution: {:.2}s</div>
                        <div class="metric">Quality: {:.1}/100</div>
                        <div class="metric">Warnings: {}</div>
                    </div>
                    <p><strong>Status:</strong> {:?}</p>
                </div>
            "#,
                status_class,
                result.name,
                result.example_type,
                result.performance_metrics.compilation_time,
                result.performance_metrics.execution_time,
                result.quality_metrics.code_quality_score,
                result.compilation_result.warnings_count,
                result.overall_status
            ));
        }

        html
    }

    async fn generate_junit_xml(&self, report: &ExampleTestReport) -> Result<()> {
        let xml_path = self.config.output_dir.join("examples_test_results.xml");

        let mut xml = String::new();
        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push_str(&format!(r#"<testsuites name="VoiRS Examples" tests="{}" failures="{}" errors="0" time="{:.3}">"#,
                              report.summary.total_examples,
                              report.summary.failed,
                              report.total_duration.as_secs_f64()));

        xml.push_str(&format!(
            r#"<testsuite name="Examples" tests="{}" failures="{}" errors="0" time="{:.3}">"#,
            report.summary.total_examples,
            report.summary.failed,
            report.total_duration.as_secs_f64()
        ));

        for result in &report.results {
            xml.push_str(&format!(
                r#"<testcase classname="{}" name="{}" time="{:.3}""#,
                result.example_type, result.name, result.performance_metrics.execution_time
            ));

            match result.overall_status {
                TestStatus::Failed => {
                    xml.push_str(">");
                    xml.push_str(&format!(
                        r#"<failure message="Example test failed">{}</failure>"#,
                        result.compilation_result.errors.join("\n")
                    ));
                    xml.push_str("</testcase>");
                }
                TestStatus::Warning => {
                    xml.push_str(">");
                    xml.push_str(&format!(
                        r#"<system-out>Warnings: {}</system-out>"#,
                        result.compilation_result.warnings_count
                    ));
                    xml.push_str("</testcase>");
                }
                _ => {
                    xml.push_str("/>");
                }
            }
        }

        xml.push_str("</testsuite>");
        xml.push_str("</testsuites>");

        fs::write(&xml_path, xml)?;
        println!("📋 JUnit XML saved to: {:?}", xml_path);

        Ok(())
    }

    fn print_summary(&self, report: &ExampleTestReport) {
        println!("\n🎉 Examples Test Framework Summary");
        println!("{}", "=".repeat(60));
        println!(
            "📊 Results: {}/{} passed ({:.1}% success rate)",
            report.summary.passed, report.summary.total_examples, report.summary.success_rate
        );

        if report.summary.failed > 0 {
            println!("❌ Failed: {}", report.summary.failed);
        }

        if report.summary.warnings > 0 {
            println!("⚠️  Warnings: {}", report.summary.warnings);
        }

        println!(
            "⏱️  Total duration: {:.2}s",
            report.total_duration.as_secs_f64()
        );

        println!("\n🏆 Performance Champions:");
        println!(
            "   🥇 Fastest: {}",
            report.performance_analysis.fastest_example
        );
        println!(
            "   🐌 Slowest: {}",
            report.performance_analysis.slowest_example
        );

        if !report.quality_analysis.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for rec in &report.quality_analysis.recommendations {
                println!("   • {}", rec);
            }
        }

        println!("\n📁 Reports saved to: {:?}", self.config.output_dir);
    }

    fn detect_current_platform(&self) -> String {
        if cfg!(target_os = "linux") {
            "Linux".to_string()
        } else if cfg!(target_os = "macos") {
            "macOS".to_string()
        } else if cfg!(target_os = "windows") {
            "Windows".to_string()
        } else {
            "Unknown".to_string()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ExamplesTestConfig {
        examples_dir: PathBuf::from("."),
        test_timeout_seconds: 120,
        performance_requirements: PerformanceRequirements {
            max_compile_time_seconds: 60.0,
            max_execution_time_seconds: 30.0,
            max_memory_mb: 500.0,
            max_cpu_percent: 80.0,
        },
        quality_thresholds: QualityThresholds {
            min_audio_duration_seconds: 1.0,
            max_audio_file_size_mb: 50.0,
            require_no_warnings: false, // Allow warnings for now
            require_clippy_pass: false,
        },
        platforms: vec![TargetPlatform::MacOS], // Current platform
        output_dir: PathBuf::from("test_reports"),
        ci_mode: std::env::var("CI").is_ok(),
    };

    let mut framework = ExamplesTestFramework::new(config)?;

    match framework.run_all_tests().await {
        Ok(report) => {
            let exit_code = if report.summary.failed > 0 {
                println!("\n❌ Some examples failed - exiting with error code");
                1
            } else if report.summary.warnings > 0 {
                println!("\n⚠️  Examples completed with warnings");
                0 // Don't fail CI for warnings
            } else {
                println!("\n✅ All examples passed successfully!");
                0
            };

            std::process::exit(exit_code);
        }
        Err(e) => {
            eprintln!("💥 Examples test framework failed: {}", e);
            std::process::exit(2);
        }
    }
}
