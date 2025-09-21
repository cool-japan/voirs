//! R Statistical Analysis Integration
//!
//! This module provides integration with the R statistical computing environment,
//! enabling advanced statistical analysis capabilities that leverage R's extensive
//! statistical library ecosystem.
//!
//! ## Features
//!
//! - R session management and script execution
//! - Data exchange between Rust and R (vectors, matrices, data frames)
//! - Advanced statistical tests and modeling
//! - Time series analysis and forecasting
//! - Machine learning integration with R packages
//! - Plotting and visualization capabilities
//!
//! ## Usage
//!
//! ```rust
//! use voirs_evaluation::r_integration::{RSession, RDataFrame};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create R session
//! let mut r_session = RSession::new().await?;
//!
//! // Perform statistical analysis
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let result = r_session.t_test(&data, Some(3.0)).await?;
//! println!("T-test p-value: {}", result.p_value);
//!
//! # Ok(())
//! # }
//! ```

use crate::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// R session for executing statistical analysis
#[derive(Debug)]
pub struct RSession {
    /// Working directory for R session
    working_dir: std::path::PathBuf,
    /// Environment variables for R session
    env_vars: HashMap<String, String>,
    /// R executable path
    r_executable: String,
    /// Session ID for temporary files
    session_id: String,
}

/// R statistical test result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RTestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Confidence interval (lower, upper)
    pub confidence_interval: Option<(f64, f64)>,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<f64>,
    /// Test method name
    pub method: String,
    /// Alternative hypothesis
    pub alternative: String,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// R data frame representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RDataFrame {
    /// Column names
    pub columns: Vec<String>,
    /// Data rows (each row is a vector of values)
    pub data: Vec<Vec<RValue>>,
}

impl RDataFrame {
    /// Create a new empty data frame
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Create a data frame with specified columns
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self {
            columns,
            data: Vec::new(),
        }
    }

    /// Add a column to the data frame
    pub fn add_column(&mut self, name: String, values: Vec<RValue>) -> Result<(), EvaluationError> {
        if !self.data.is_empty() && values.len() != self.data.len() {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Column length {} doesn't match existing data length {}",
                    values.len(),
                    self.data.len()
                ),
            });
        }

        self.columns.push(name);

        if self.data.is_empty() {
            // Initialize rows if this is the first column
            for value in values {
                self.data.push(vec![value]);
            }
        } else {
            // Add values to existing rows
            for (i, value) in values.into_iter().enumerate() {
                if i < self.data.len() {
                    self.data[i].push(value);
                }
            }
        }

        Ok(())
    }

    /// Add a row to the data frame
    pub fn add_row(&mut self, row: Vec<RValue>) -> Result<(), EvaluationError> {
        if row.len() != self.columns.len() {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Row length {} doesn't match column count {}",
                    row.len(),
                    self.columns.len()
                ),
            });
        }
        self.data.push(row);
        Ok(())
    }

    /// Get a column by name
    pub fn get_column(&self, name: &str) -> Option<Vec<RValue>> {
        if let Some(col_index) = self.columns.iter().position(|c| c == name) {
            Some(self.data.iter().map(|row| row[col_index].clone()).collect())
        } else {
            None
        }
    }

    /// Get a row by index
    pub fn get_row(&self, index: usize) -> Option<&Vec<RValue>> {
        self.data.get(index)
    }

    /// Filter rows based on a predicate
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Vec<RValue>) -> bool,
    {
        let filtered_data: Vec<Vec<RValue>> = self
            .data
            .iter()
            .filter(|row| predicate(row))
            .cloned()
            .collect();

        Self {
            columns: self.columns.clone(),
            data: filtered_data,
        }
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Check if the data frame is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get column names
    pub fn column_names(&self) -> &Vec<String> {
        &self.columns
    }
}

impl Default for RDataFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// R value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RValue {
    /// Numeric value
    Numeric(f64),
    /// Integer value
    Integer(i64),
    /// String value
    String(String),
    /// Boolean value
    Logical(bool),
    /// Missing value (NA)
    NA,
}

/// R linear model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLinearModel {
    /// Model coefficients
    pub coefficients: HashMap<String, f64>,
    /// Coefficient standard errors
    pub std_errors: HashMap<String, f64>,
    /// T-values for coefficients
    pub t_values: HashMap<String, f64>,
    /// P-values for coefficients
    pub p_values: HashMap<String, f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Adjusted R-squared value
    pub adjusted_r_squared: f64,
    /// F-statistic
    pub f_statistic: f64,
    /// F-statistic p-value
    pub f_p_value: f64,
    /// Residual standard error
    pub residual_std_error: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: (i32, i32),
}

/// R ANOVA result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAnovaResult {
    /// Source of variation
    pub sources: Vec<String>,
    /// Degrees of freedom
    pub df: Vec<i32>,
    /// Sum of squares
    pub sum_squares: Vec<f64>,
    /// Mean squares
    pub mean_squares: Vec<f64>,
    /// F-statistics
    pub f_statistics: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
}

/// R time series analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTimeSeriesResult {
    /// Time series data
    pub data: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
    /// Forecasted values
    pub forecast: Option<Vec<f64>>,
    /// Forecast intervals
    pub forecast_intervals: Option<Vec<(f64, f64)>>,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model diagnostics
    pub diagnostics: HashMap<String, f64>,
}

/// Logistic regression model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLogisticModel {
    /// Coefficient names
    pub coefficients: Vec<String>,
    /// Coefficient estimates
    pub estimates: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// AIC value
    pub aic: f64,
    /// Deviance
    pub deviance: f64,
    /// Null deviance
    pub null_deviance: f64,
}

/// Random forest model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RRandomForestModel {
    /// Number of trees
    pub ntree: i32,
    /// Number of variables randomly sampled as candidates at each split
    pub mtry: i32,
    /// Out-of-bag error rate
    pub oob_error: f64,
    /// Variable importance scores
    pub variable_importance: Vec<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
}

/// Generalized Additive Model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGamModel {
    /// Deviance explained
    pub deviance_explained: f64,
    /// R-squared value
    pub r_squared: f64,
    /// AIC value
    pub aic: f64,
    /// BIC value
    pub bic: f64,
    /// GCV score
    pub gcv_score: f64,
    /// Smooth terms
    pub smooth_terms: Vec<String>,
}

/// ARIMA model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RArimaModel {
    /// AIC value
    pub aic: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Coefficient names
    pub coefficients: Vec<String>,
    /// Coefficient estimates
    pub estimates: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Residual variance
    pub residual_variance: f64,
}

/// Principal Component Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPcaResult {
    /// Variance explained by each component
    pub variance_explained: Vec<f64>,
    /// Cumulative variance explained
    pub cumulative_variance: Vec<f64>,
    /// Component names
    pub component_names: Vec<String>,
    /// Number of components
    pub n_components: i32,
}

/// K-means clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RKmeansResult {
    /// Cluster centers
    pub centers: Vec<f64>,
    /// Size of each cluster
    pub cluster_sizes: Vec<i32>,
    /// Within-cluster sum of squares for each cluster
    pub within_ss: Vec<f64>,
    /// Total within-cluster sum of squares
    pub total_within_ss: f64,
    /// Between-cluster sum of squares
    pub between_ss: f64,
    /// Total sum of squares
    pub total_ss: f64,
    /// Number of clusters
    pub k: i32,
}

/// Survival analysis model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSurvivalModel {
    /// Concordance index
    pub concordance: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC value
    pub aic: f64,
    /// Coefficient names
    pub coefficients: Vec<String>,
    /// Coefficient estimates
    pub estimates: Vec<f64>,
    /// Hazard ratios
    pub hazard_ratios: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
}

impl RSession {
    /// Create a new R session
    pub async fn new() -> EvaluationResult<Self> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let working_dir = std::env::temp_dir().join(format!("voirs_r_session_{}", session_id));

        // Create working directory
        tokio::fs::create_dir_all(&working_dir).await.map_err(|e| {
            EvaluationError::ProcessingError {
                message: format!("Failed to create R working directory: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Find R executable
        let r_executable = Self::find_r_executable()?;

        let mut env_vars = HashMap::new();
        env_vars.insert(
            "R_LIBS_USER".to_string(),
            working_dir.to_string_lossy().to_string(),
        );

        info!("Created R session with ID: {}", session_id);

        Ok(Self {
            working_dir,
            env_vars,
            r_executable,
            session_id,
        })
    }

    /// Find R executable on the system
    fn find_r_executable() -> EvaluationResult<String> {
        // Try common R executable names and paths
        let candidates = vec![
            "R",
            "Rscript",
            "/usr/bin/R",
            "/usr/local/bin/R",
            "/opt/R/bin/R",
            // Windows paths
            "C:\\Program Files\\R\\R-4.3.0\\bin\\R.exe",
            "C:\\Program Files\\R\\R-4.2.0\\bin\\R.exe",
            "C:\\Program Files\\R\\R-4.1.0\\bin\\R.exe",
        ];

        for candidate in candidates {
            if let Ok(output) = std::process::Command::new(candidate)
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    info!("Found R executable: {}", candidate);
                    return Ok(candidate.to_string());
                }
            }
        }

        Err(EvaluationError::ConfigurationError {
            message: "R executable not found. Please install R and ensure it's in PATH."
                .to_string(),
        }
        .into())
    }

    /// Execute R script and return output
    pub async fn execute_script(&self, script: &str) -> EvaluationResult<String> {
        let script_file = self
            .working_dir
            .join(format!("script_{}.R", uuid::Uuid::new_v4()));

        // Write script to file
        tokio::fs::write(&script_file, script).await.map_err(|e| {
            EvaluationError::ProcessingError {
                message: format!("Failed to write R script: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Execute R script
        let mut cmd = Command::new(&self.r_executable);
        cmd.arg("--slave")
            .arg("--no-restore")
            .arg("--file")
            .arg(&script_file)
            .current_dir(&self.working_dir);

        // Set environment variables
        for (key, value) in &self.env_vars {
            cmd.env(key, value);
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to execute R script: {}", e),
                source: Some(Box::new(e)),
            })?;

        // Clean up script file
        let _ = tokio::fs::remove_file(&script_file).await;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(EvaluationError::ProcessingError {
                message: format!("R script execution failed: {}", stderr),
                source: None,
            }
            .into());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Perform t-test using R
    pub async fn t_test(&mut self, data: &[f64], mu: Option<f64>) -> EvaluationResult<RTestResult> {
        let data_str = format!(
            "c({})",
            data.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mu_str = mu.map(|m| format!(", mu = {}", m)).unwrap_or_default();

        let script = format!(
            r#"
            data <- {}
            result <- t.test(data{})
            
            cat("statistic:", result$statistic, "\n")
            cat("p_value:", result$p.value, "\n")
            if (!is.null(result$conf.int)) {{
                cat("conf_lower:", result$conf.int[1], "\n")
                cat("conf_upper:", result$conf.int[2], "\n")
            }}
            if (!is.null(result$parameter)) {{
                cat("df:", result$parameter, "\n")
            }}
            cat("method:", result$method, "\n")
            cat("alternative:", result$alternative, "\n")
            "#,
            data_str, mu_str
        );

        let output = self.execute_script(&script).await?;
        self.parse_test_result(&output)
    }

    /// Perform Wilcoxon test using R
    pub async fn wilcox_test(
        &mut self,
        x: &[f64],
        y: Option<&[f64]>,
    ) -> EvaluationResult<RTestResult> {
        let x_str = format!(
            "c({})",
            x.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let script = if let Some(y_data) = y {
            let y_str = format!(
                "c({})",
                y_data
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            format!(
                r#"
                x <- {}
                y <- {}
                result <- wilcox.test(x, y)
                
                cat("statistic:", result$statistic, "\n")
                cat("p_value:", result$p.value, "\n")
                cat("method:", result$method, "\n")
                cat("alternative:", result$alternative, "\n")
                "#,
                x_str, y_str
            )
        } else {
            format!(
                r#"
                x <- {}
                result <- wilcox.test(x)
                
                cat("statistic:", result$statistic, "\n")
                cat("p_value:", result$p.value, "\n")
                cat("method:", result$method, "\n")
                cat("alternative:", result$alternative, "\n")
                "#,
                x_str
            )
        };

        let output = self.execute_script(&script).await?;
        self.parse_test_result(&output)
    }

    /// Perform linear regression using R
    pub async fn linear_regression(
        &mut self,
        x: &[f64],
        y: &[f64],
    ) -> EvaluationResult<RLinearModel> {
        if x.len() != y.len() {
            return Err(EvaluationError::InvalidInput {
                message: "x and y vectors must have the same length".to_string(),
            }
            .into());
        }

        let x_str = format!(
            "c({})",
            x.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let y_str = format!(
            "c({})",
            y.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let script = format!(
            r#"
            x <- {}
            y <- {}
            model <- lm(y ~ x)
            summary_model <- summary(model)
            
            cat("intercept_coef:", summary_model$coefficients[1,1], "\n")
            cat("intercept_stderr:", summary_model$coefficients[1,2], "\n")
            cat("intercept_tvalue:", summary_model$coefficients[1,3], "\n")
            cat("intercept_pvalue:", summary_model$coefficients[1,4], "\n")
            
            cat("x_coef:", summary_model$coefficients[2,1], "\n")
            cat("x_stderr:", summary_model$coefficients[2,2], "\n")
            cat("x_tvalue:", summary_model$coefficients[2,3], "\n")
            cat("x_pvalue:", summary_model$coefficients[2,4], "\n")
            
            cat("r_squared:", summary_model$r.squared, "\n")
            cat("adj_r_squared:", summary_model$adj.r.squared, "\n")
            cat("f_statistic:", summary_model$fstatistic[1], "\n")
            cat("f_df1:", summary_model$fstatistic[2], "\n")
            cat("f_df2:", summary_model$fstatistic[3], "\n")
            cat("residual_std_error:", summary_model$sigma, "\n")
            
            # Calculate F p-value
            f_pvalue <- pf(summary_model$fstatistic[1], summary_model$fstatistic[2], summary_model$fstatistic[3], lower.tail = FALSE)
            cat("f_pvalue:", f_pvalue, "\n")
            "#,
            x_str, y_str
        );

        let output = self.execute_script(&script).await?;
        self.parse_linear_model(&output)
    }

    /// Perform ANOVA using R
    pub async fn anova(&mut self, groups: &[Vec<f64>]) -> EvaluationResult<RAnovaResult> {
        if groups.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "At least one group is required for ANOVA".to_string(),
            }
            .into());
        }

        // Create data frame for ANOVA
        let mut data_values = Vec::new();
        let mut group_labels = Vec::new();

        for (i, group) in groups.iter().enumerate() {
            for &value in group {
                data_values.push(value);
                group_labels.push(format!("Group{}", i + 1));
            }
        }

        let values_str = format!(
            "c({})",
            data_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let groups_str = format!(
            "c({})",
            group_labels
                .iter()
                .map(|g| format!("\"{}\"", g))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let script = format!(
            r#"
            values <- {}
            groups <- factor({})
            
            model <- aov(values ~ groups)
            anova_result <- anova(model)
            
            cat("sources: groups,Residuals\n")
            cat("df:", anova_result$Df[1], ",", anova_result$Df[2], "\n")
            cat("sum_squares:", anova_result$'Sum Sq'[1], ",", anova_result$'Sum Sq'[2], "\n")
            cat("mean_squares:", anova_result$'Mean Sq'[1], ",", anova_result$'Mean Sq'[2], "\n")
            cat("f_statistics:", anova_result$'F value'[1], ",NA\n")
            cat("p_values:", anova_result$'Pr(>F)'[1], ",NA\n")
            "#,
            values_str, groups_str
        );

        let output = self.execute_script(&script).await?;
        self.parse_anova_result(&output)
    }

    /// Perform correlation analysis using R
    pub async fn correlation(
        &mut self,
        x: &[f64],
        y: &[f64],
        method: &str,
    ) -> EvaluationResult<RTestResult> {
        if x.len() != y.len() {
            return Err(EvaluationError::InvalidInput {
                message: "x and y vectors must have the same length".to_string(),
            }
            .into());
        }

        let valid_methods = ["pearson", "spearman", "kendall"];
        if !valid_methods.contains(&method) {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Invalid correlation method: {}. Use pearson, spearman, or kendall",
                    method
                ),
            }
            .into());
        }

        let x_str = format!(
            "c({})",
            x.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let y_str = format!(
            "c({})",
            y.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let script = format!(
            r#"
            x <- {}
            y <- {}
            result <- cor.test(x, y, method = "{}")
            
            cat("statistic:", result$statistic, "\n")
            cat("p_value:", result$p.value, "\n")
            if (!is.null(result$conf.int)) {{
                cat("conf_lower:", result$conf.int[1], "\n")
                cat("conf_upper:", result$conf.int[2], "\n")
            }}
            if (!is.null(result$parameter)) {{
                cat("df:", result$parameter, "\n")
            }}
            cat("method:", result$method, "\n")
            cat("alternative:", result$alternative, "\n")
            cat("estimate:", result$estimate, "\n")
            "#,
            x_str, y_str, method
        );

        let output = self.execute_script(&script).await?;
        self.parse_test_result(&output)
    }

    /// Install R package if not already installed
    pub async fn install_package(&self, package: &str) -> EvaluationResult<()> {
        let script = format!(
            r#"
            package_name <- "{}"
            if (!require(package_name, character.only = TRUE)) {{
                install.packages(package_name, repos = "https://cran.r-project.org/")
                library(package_name, character.only = TRUE)
            }}
            cat("Package", package_name, "is available\n")
            "#,
            package
        );

        let output = self.execute_script(&script).await?;
        debug!("Package installation output: {}", output);
        Ok(())
    }

    /// Parse R test result from output
    fn parse_test_result(&self, output: &str) -> EvaluationResult<RTestResult> {
        let mut statistic = 0.0;
        let mut p_value = 1.0;
        let mut confidence_interval = None;
        let mut degrees_of_freedom = None;
        let mut method = "Unknown".to_string();
        let mut alternative = "two.sided".to_string();
        let mut parameters = HashMap::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "statistic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            statistic = val;
                        }
                    }
                    "p_value" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            p_value = val;
                        }
                    }
                    "conf_lower" => {
                        if let Ok(lower) = parts[1].parse::<f64>() {
                            if let Some((_, upper)) = confidence_interval {
                                confidence_interval = Some((lower, upper));
                            } else {
                                confidence_interval = Some((lower, 0.0));
                            }
                        }
                    }
                    "conf_upper" => {
                        if let Ok(upper) = parts[1].parse::<f64>() {
                            if let Some((lower, _)) = confidence_interval {
                                confidence_interval = Some((lower, upper));
                            } else {
                                confidence_interval = Some((0.0, upper));
                            }
                        }
                    }
                    "df" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            degrees_of_freedom = Some(val);
                        }
                    }
                    "method" => {
                        method = parts[1..].join(" ");
                    }
                    "alternative" => {
                        alternative = parts[1].to_string();
                    }
                    "estimate" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            parameters.insert("estimate".to_string(), val);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RTestResult {
            statistic,
            p_value,
            confidence_interval,
            degrees_of_freedom,
            method,
            alternative,
            parameters,
        })
    }

    /// Parse linear model result from output
    fn parse_linear_model(&self, output: &str) -> EvaluationResult<RLinearModel> {
        let mut coefficients = HashMap::new();
        let mut std_errors = HashMap::new();
        let mut t_values = HashMap::new();
        let mut p_values = HashMap::new();
        let mut r_squared = 0.0;
        let mut adjusted_r_squared = 0.0;
        let mut f_statistic = 0.0;
        let mut f_p_value = 1.0;
        let mut residual_std_error = 0.0;
        let mut f_df1 = 0;
        let mut f_df2 = 0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "intercept_coef" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            coefficients.insert("(Intercept)".to_string(), val);
                        }
                    }
                    "intercept_stderr" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            std_errors.insert("(Intercept)".to_string(), val);
                        }
                    }
                    "intercept_tvalue" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            t_values.insert("(Intercept)".to_string(), val);
                        }
                    }
                    "intercept_pvalue" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            p_values.insert("(Intercept)".to_string(), val);
                        }
                    }
                    "x_coef" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            coefficients.insert("x".to_string(), val);
                        }
                    }
                    "x_stderr" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            std_errors.insert("x".to_string(), val);
                        }
                    }
                    "x_tvalue" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            t_values.insert("x".to_string(), val);
                        }
                    }
                    "x_pvalue" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            p_values.insert("x".to_string(), val);
                        }
                    }
                    "r_squared" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            r_squared = val;
                        }
                    }
                    "adj_r_squared" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            adjusted_r_squared = val;
                        }
                    }
                    "f_statistic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            f_statistic = val;
                        }
                    }
                    "f_df1" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            f_df1 = val;
                        }
                    }
                    "f_df2" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            f_df2 = val;
                        }
                    }
                    "f_pvalue" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            f_p_value = val;
                        }
                    }
                    "residual_std_error" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            residual_std_error = val;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RLinearModel {
            coefficients,
            std_errors,
            t_values,
            p_values,
            r_squared,
            adjusted_r_squared,
            f_statistic,
            f_p_value,
            residual_std_error,
            degrees_of_freedom: (f_df1, f_df2),
        })
    }

    /// Parse ANOVA result from output
    fn parse_anova_result(&self, output: &str) -> EvaluationResult<RAnovaResult> {
        let mut sources = Vec::new();
        let mut df = Vec::new();
        let mut sum_squares = Vec::new();
        let mut mean_squares = Vec::new();
        let mut f_statistics = Vec::new();
        let mut p_values = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "sources" => {
                        sources = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "df" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<i32>() {
                                df.push(val);
                            }
                        }
                    }
                    "sum_squares" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                sum_squares.push(val);
                            }
                        }
                    }
                    "mean_squares" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                mean_squares.push(val);
                            }
                        }
                    }
                    "f_statistics" => {
                        for part in parts[1].split(',') {
                            if part == "NA" {
                                f_statistics.push(f64::NAN);
                            } else if let Ok(val) = part.parse::<f64>() {
                                f_statistics.push(val);
                            }
                        }
                    }
                    "p_values" => {
                        for part in parts[1].split(',') {
                            if part == "NA" {
                                p_values.push(f64::NAN);
                            } else if let Ok(val) = part.parse::<f64>() {
                                p_values.push(val);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RAnovaResult {
            sources,
            df,
            sum_squares,
            mean_squares,
            f_statistics,
            p_values,
        })
    }

    /// Create a data frame in R from RDataFrame
    pub async fn create_dataframe(&mut self, name: &str, df: &RDataFrame) -> EvaluationResult<()> {
        if df.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Cannot create empty data frame".to_string(),
            }
            .into());
        }

        let r_df_str = utils::rust_to_r_dataframe(df);
        let script = format!("{} <- {}", name, r_df_str);

        let output = self.execute_script(&script).await?;
        debug!("Created data frame '{}': {}", name, output);
        Ok(())
    }

    /// Read a data frame from R
    pub async fn read_dataframe(&mut self, name: &str) -> EvaluationResult<RDataFrame> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            df <- {}
            cat("columns:", paste(names(df), collapse=","), "\n")
            cat("nrows:", nrow(df), "\n")
            cat("ncols:", ncol(df), "\n")
            
            for(i in 1:nrow(df)) {{
                row_values <- c()
                for(j in 1:ncol(df)) {{
                    val <- df[i, j]
                    if(is.na(val)) {{
                        row_values <- c(row_values, "NA")
                    }} else if(is.numeric(val)) {{
                        row_values <- c(row_values, paste0("n:", val))
                    }} else if(is.logical(val)) {{
                        row_values <- c(row_values, paste0("l:", val))
                    }} else {{
                        row_values <- c(row_values, paste0("s:", val))
                    }}
                }}
                cat("row", i, ":", paste(row_values, collapse=","), "\n")
            }}
            "#,
            name, name, name
        );

        let output = self.execute_script(&script).await?;
        self.parse_dataframe_result(&output)
    }

    /// Write a data frame to CSV file through R
    pub async fn write_dataframe_csv(
        &mut self,
        df_name: &str,
        file_path: &str,
    ) -> EvaluationResult<()> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            write.csv({}, "{}", row.names=FALSE)
            cat("Data frame written to: {}")
            "#,
            df_name, df_name, df_name, file_path, file_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Written data frame to CSV: {}", output);
        Ok(())
    }

    /// Read a data frame from CSV file through R
    pub async fn read_dataframe_csv(&mut self, file_path: &str) -> EvaluationResult<RDataFrame> {
        let script = format!(
            r#"
            df <- read.csv("{}")
            cat("columns:", paste(names(df), collapse=","), "\n")
            cat("nrows:", nrow(df), "\n")
            cat("ncols:", ncol(df), "\n")
            
            for(i in 1:nrow(df)) {{
                row_values <- c()
                for(j in 1:ncol(df)) {{
                    val <- df[i, j]
                    if(is.na(val)) {{
                        row_values <- c(row_values, "NA")
                    }} else if(is.numeric(val)) {{
                        row_values <- c(row_values, paste0("n:", val))
                    }} else if(is.logical(val)) {{
                        row_values <- c(row_values, paste0("l:", val))
                    }} else {{
                        row_values <- c(row_values, paste0("s:", val))
                    }}
                }}
                cat("row", i, ":", paste(row_values, collapse=","), "\n")
            }}
            "#,
            file_path
        );

        let output = self.execute_script(&script).await?;
        self.parse_dataframe_result(&output)
    }

    /// Filter data frame in R
    pub async fn filter_dataframe(
        &mut self,
        df_name: &str,
        condition: &str,
        result_name: &str,
    ) -> EvaluationResult<()> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            {} <- {}[{}, ]
            cat("Filtered data frame created:", "{}")
            "#,
            df_name, df_name, result_name, df_name, condition, result_name
        );

        let output = self.execute_script(&script).await?;
        debug!("Filtered data frame: {}", output);
        Ok(())
    }

    /// Get summary statistics for a data frame
    pub async fn dataframe_summary(&mut self, df_name: &str) -> EvaluationResult<String> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            summary({})
            "#,
            df_name, df_name, df_name
        );

        self.execute_script(&script).await
    }

    /// Parse data frame result from R output
    fn parse_dataframe_result(&self, output: &str) -> EvaluationResult<RDataFrame> {
        let mut columns = Vec::new();
        let mut data = Vec::new();
        let mut nrows = 0;
        let mut ncols = 0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "columns" => {
                        columns = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "nrows" => {
                        if let Ok(val) = parts[1].parse::<usize>() {
                            nrows = val;
                        }
                    }
                    "ncols" => {
                        if let Ok(val) = parts[1].parse::<usize>() {
                            ncols = val;
                        }
                    }
                    part if part.starts_with("row") => {
                        if parts.len() >= 3 {
                            let row_data_str = parts[2..].join(" ");
                            let mut row = Vec::new();

                            for value_str in row_data_str.split(',') {
                                let value = match value_str.trim() {
                                    "NA" => RValue::NA,
                                    s if s.starts_with("n:") => {
                                        if let Ok(val) = s[2..].parse::<f64>() {
                                            RValue::Numeric(val)
                                        } else {
                                            RValue::NA
                                        }
                                    }
                                    s if s.starts_with("l:") => {
                                        match s[2..].to_lowercase().as_str() {
                                            "true" => RValue::Logical(true),
                                            "false" => RValue::Logical(false),
                                            _ => RValue::NA,
                                        }
                                    }
                                    s if s.starts_with("s:") => RValue::String(s[2..].to_string()),
                                    s => RValue::String(s.to_string()),
                                };
                                row.push(value);
                            }
                            data.push(row);
                        }
                    }
                    _ => {}
                }
            }
        }

        if columns.is_empty() || data.is_empty() {
            return Err(EvaluationError::ProcessingError {
                message: "Failed to parse data frame from R output".to_string(),
                source: None,
            }
            .into());
        }

        Ok(RDataFrame { columns, data })
    }

    /// Create a scatter plot using ggplot2
    pub async fn ggplot_scatter(
        &mut self,
        df_name: &str,
        x_col: &str,
        y_col: &str,
        color_col: Option<&str>,
        output_path: &str,
    ) -> EvaluationResult<()> {
        // Ensure ggplot2 is available
        self.install_package("ggplot2").await?;

        let color_mapping = if let Some(col) = color_col {
            format!(", color = {}", col)
        } else {
            String::new()
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {}, y = {}{})) +
                 geom_point() +
                 theme_minimal() +
                 labs(title = "Scatter Plot: {} vs {}")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Scatter plot saved to: {}")
            "#,
            df_name,
            df_name,
            df_name,
            x_col,
            y_col,
            color_mapping,
            x_col,
            y_col,
            output_path,
            output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created scatter plot: {}", output);
        Ok(())
    }

    /// Create a line plot using ggplot2
    pub async fn ggplot_line(
        &mut self,
        df_name: &str,
        x_col: &str,
        y_col: &str,
        group_col: Option<&str>,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let group_mapping = if let Some(col) = group_col {
            format!(", group = {}, color = {}", col, col)
        } else {
            String::new()
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {}, y = {}{})) +
                 geom_line() +
                 geom_point() +
                 theme_minimal() +
                 labs(title = "Line Plot: {} over {}")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Line plot saved to: {}")
            "#,
            df_name,
            df_name,
            df_name,
            x_col,
            y_col,
            group_mapping,
            y_col,
            x_col,
            output_path,
            output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created line plot: {}", output);
        Ok(())
    }

    /// Create a histogram using ggplot2
    pub async fn ggplot_histogram(
        &mut self,
        df_name: &str,
        x_col: &str,
        bins: Option<i32>,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let bins_param = if let Some(b) = bins {
            format!(", bins = {}", b)
        } else {
            ", bins = 30".to_string()
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {})) +
                 geom_histogram(alpha = 0.7{}) +
                 theme_minimal() +
                 labs(title = "Histogram of {}", x = "{}", y = "Frequency")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Histogram saved to: {}")
            "#,
            df_name, df_name, df_name, x_col, bins_param, x_col, x_col, output_path, output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created histogram: {}", output);
        Ok(())
    }

    /// Create a box plot using ggplot2
    pub async fn ggplot_boxplot(
        &mut self,
        df_name: &str,
        x_col: Option<&str>,
        y_col: &str,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let (x_mapping, geom_box) = if let Some(x) = x_col {
            (format!(", x = {}", x), "geom_boxplot()".to_string())
        } else {
            (
                "".to_string(),
                format!("geom_boxplot(aes(x = \"\", y = {})) + coord_flip()", y_col),
            )
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(y = {}{})) +
                 {} +
                 theme_minimal() +
                 labs(title = "Box Plot of {}")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Box plot saved to: {}")
            "#,
            df_name, df_name, df_name, y_col, x_mapping, geom_box, y_col, output_path, output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created box plot: {}", output);
        Ok(())
    }

    /// Create a bar plot using ggplot2
    pub async fn ggplot_barplot(
        &mut self,
        df_name: &str,
        x_col: &str,
        y_col: Option<&str>,
        fill_col: Option<&str>,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let (y_mapping, geom_bar) = if let Some(y) = y_col {
            (format!(", y = {}", y), "geom_col()".to_string())
        } else {
            ("".to_string(), "geom_bar()".to_string())
        };

        let fill_mapping = if let Some(fill) = fill_col {
            format!(", fill = {}", fill)
        } else {
            String::new()
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {}{}{})) +
                 {} +
                 theme_minimal() +
                 labs(title = "Bar Plot of {}")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Bar plot saved to: {}")
            "#,
            df_name,
            df_name,
            df_name,
            x_col,
            y_mapping,
            fill_mapping,
            geom_bar,
            x_col,
            output_path,
            output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created bar plot: {}", output);
        Ok(())
    }

    /// Create a density plot using ggplot2
    pub async fn ggplot_density(
        &mut self,
        df_name: &str,
        x_col: &str,
        group_col: Option<&str>,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let group_mapping = if let Some(col) = group_col {
            format!(", fill = {}, color = {}", col, col)
        } else {
            String::new()
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {}{})) +
                 geom_density(alpha = 0.7) +
                 theme_minimal() +
                 labs(title = "Density Plot of {}", x = "{}", y = "Density")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Density plot saved to: {}")
            "#,
            df_name, df_name, df_name, x_col, group_mapping, x_col, x_col, output_path, output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created density plot: {}", output);
        Ok(())
    }

    /// Create a correlation heatmap using ggplot2
    pub async fn ggplot_correlation_heatmap(
        &mut self,
        df_name: &str,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let script = format!(
            r#"
            library(ggplot2)
            library(reshape2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            # Select only numeric columns
            numeric_cols <- sapply({}, is.numeric)
            if (sum(numeric_cols) < 2) {{
                stop("Need at least 2 numeric columns for correlation heatmap")
            }}
            
            numeric_df <- {}[, numeric_cols, drop = FALSE]
            cor_matrix <- cor(numeric_df, use = "complete.obs")
            cor_melted <- melt(cor_matrix)
            
            p <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
                 geom_tile() +
                 scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                                    midpoint = 0, limit = c(-1, 1)) +
                 theme_minimal() +
                 theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                 labs(title = "Correlation Heatmap", x = "", y = "", fill = "Correlation")
            
            ggsave("{}", plot = p, width = 8, height = 6, dpi = 300)
            cat("Correlation heatmap saved to: {}")
            "#,
            df_name, df_name, df_name, df_name, output_path, output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created correlation heatmap: {}", output);
        Ok(())
    }

    /// Create a faceted plot using ggplot2
    pub async fn ggplot_faceted_plot(
        &mut self,
        df_name: &str,
        x_col: &str,
        y_col: &str,
        facet_col: &str,
        plot_type: &str,
        output_path: &str,
    ) -> EvaluationResult<()> {
        self.install_package("ggplot2").await?;

        let geom_layer = match plot_type {
            "point" => "geom_point()",
            "line" => "geom_line()",
            "bar" => "geom_col()",
            _ => "geom_point()", // default
        };

        let script = format!(
            r#"
            library(ggplot2)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            p <- ggplot({}, aes(x = {}, y = {})) +
                 {} +
                 facet_wrap(~ {}) +
                 theme_minimal() +
                 labs(title = "Faceted {} Plot: {} vs {} by {}")
            
            ggsave("{}", plot = p, width = 10, height = 8, dpi = 300)
            cat("Faceted plot saved to: {}")
            "#,
            df_name,
            df_name,
            df_name,
            x_col,
            y_col,
            geom_layer,
            facet_col,
            plot_type,
            x_col,
            y_col,
            facet_col,
            output_path,
            output_path
        );

        let output = self.execute_script(&script).await?;
        debug!("Created faceted plot: {}", output);
        Ok(())
    }

    /// Fit a logistic regression model
    pub async fn logistic_regression(
        &mut self,
        df_name: &str,
        formula: &str,
    ) -> EvaluationResult<RLogisticModel> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            model <- glm({}, data = {}, family = binomial)
            summary_model <- summary(model)
            
            cat("coefficients:", paste(names(coef(model)), collapse=","), "\n")
            cat("estimates:", paste(coef(model), collapse=","), "\n")
            cat("std_errors:", paste(summary_model$coefficients[, "Std. Error"], collapse=","), "\n")
            cat("z_values:", paste(summary_model$coefficients[, "z value"], collapse=","), "\n") 
            cat("p_values:", paste(summary_model$coefficients[, "Pr(>|z|)"], collapse=","), "\n")
            cat("aic:", summary_model$aic, "\n")
            cat("deviance:", summary_model$deviance, "\n")
            cat("null_deviance:", summary_model$null.deviance, "\n")
            "#,
            df_name, df_name, formula, df_name
        );

        let output = self.execute_script(&script).await?;
        self.parse_logistic_model(&output)
    }

    /// Fit a random forest model using randomForest package
    pub async fn random_forest(
        &mut self,
        df_name: &str,
        formula: &str,
        n_trees: Option<i32>,
    ) -> EvaluationResult<RRandomForestModel> {
        self.install_package("randomForest").await?;

        let ntree_param = n_trees.unwrap_or(500);

        let script = format!(
            r#"
            library(randomForest)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            model <- randomForest({}, data = {}, ntree = {}, importance = TRUE)
            
            cat("ntree:", model$ntree, "\n")
            cat("mtry:", model$mtry, "\n")
            cat("oob_error:", tail(model$err.rate[, "OOB"], 1), "\n")
            cat("variable_importance:", paste(importance(model)[, "MeanDecreaseGini"], collapse=","), "\n")
            cat("variable_names:", paste(rownames(importance(model)), collapse=","), "\n")
            "#,
            df_name, df_name, formula, df_name, ntree_param
        );

        let output = self.execute_script(&script).await?;
        self.parse_random_forest_model(&output)
    }

    /// Fit a Generalized Additive Model using mgcv package
    pub async fn gam_model(&mut self, df_name: &str, formula: &str) -> EvaluationResult<RGamModel> {
        self.install_package("mgcv").await?;

        let script = format!(
            r#"
            library(mgcv)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            model <- gam({}, data = {})
            summary_model <- summary(model)
            
            cat("deviance_explained:", summary_model$dev.expl, "\n")
            cat("r_squared:", summary_model$r.sq, "\n")
            cat("aic:", AIC(model), "\n")
            cat("bic:", BIC(model), "\n")
            cat("gcv_score:", model$gcv.ubre, "\n")
            cat("smooth_terms:", paste(names(model$smooth), collapse=","), "\n")
            "#,
            df_name, df_name, formula, df_name
        );

        let output = self.execute_script(&script).await?;
        self.parse_gam_model(&output)
    }

    /// Fit an ARIMA time series model
    pub async fn arima_model(
        &mut self,
        series_name: &str,
        order: (i32, i32, i32),
    ) -> EvaluationResult<RArimaModel> {
        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Time series '{}' does not exist")
            }}
            
            model <- arima({}, order = c({}, {}, {}))
            
            cat("aic:", model$aic, "\n")
            cat("log_likelihood:", model$loglik, "\n")
            cat("coefficients:", paste(names(coef(model)), collapse=","), "\n")
            cat("estimates:", paste(coef(model), collapse=","), "\n")
            cat("std_errors:", paste(sqrt(diag(model$var.coef)), collapse=","), "\n")
            cat("residual_variance:", model$sigma2, "\n")
            "#,
            series_name, series_name, series_name, order.0, order.1, order.2
        );

        let output = self.execute_script(&script).await?;
        self.parse_arima_model(&output)
    }

    /// Perform principal component analysis
    pub async fn pca_analysis(
        &mut self,
        df_name: &str,
        scale: bool,
    ) -> EvaluationResult<RPcaResult> {
        let scale_param = if scale { "TRUE" } else { "FALSE" };

        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            # Select only numeric columns
            numeric_cols <- sapply({}, is.numeric)
            if (sum(numeric_cols) < 2) {{
                stop("Need at least 2 numeric columns for PCA")
            }}
            
            numeric_df <- {}[, numeric_cols, drop = FALSE]
            pca_result <- prcomp(numeric_df, scale. = {})
            
            cat("variance_explained:", paste(pca_result$sdev^2 / sum(pca_result$sdev^2), collapse=","), "\n")
            cat("cumulative_variance:", paste(cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2), collapse=","), "\n")
            cat("component_names:", paste(colnames(pca_result$rotation), collapse=","), "\n")
            cat("n_components:", ncol(pca_result$rotation), "\n")
            "#,
            df_name, df_name, df_name, df_name, scale_param
        );

        let output = self.execute_script(&script).await?;
        self.parse_pca_result(&output)
    }

    /// Perform k-means clustering
    pub async fn kmeans_clustering(
        &mut self,
        df_name: &str,
        k: i32,
        n_start: Option<i32>,
    ) -> EvaluationResult<RKmeansResult> {
        let nstart_param = n_start.unwrap_or(25);

        let script = format!(
            r#"
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            # Select only numeric columns
            numeric_cols <- sapply({}, is.numeric)
            if (sum(numeric_cols) < 2) {{
                stop("Need at least 2 numeric columns for k-means")
            }}
            
            numeric_df <- {}[, numeric_cols, drop = FALSE]
            kmeans_result <- kmeans(numeric_df, centers = {}, nstart = {})
            
            cat("centers:", paste(as.vector(t(kmeans_result$centers)), collapse=","), "\n")
            cat("cluster_sizes:", paste(kmeans_result$size, collapse=","), "\n")
            cat("within_ss:", paste(kmeans_result$withinss, collapse=","), "\n")
            cat("total_within_ss:", kmeans_result$tot.withinss, "\n")
            cat("between_ss:", kmeans_result$betweenss, "\n")
            cat("total_ss:", kmeans_result$totss, "\n")
            cat("k:", length(kmeans_result$size), "\n")
            "#,
            df_name, df_name, df_name, df_name, k, nstart_param
        );

        let output = self.execute_script(&script).await?;
        self.parse_kmeans_result(&output)
    }

    /// Perform survival analysis using survival package
    pub async fn survival_analysis(
        &mut self,
        df_name: &str,
        time_col: &str,
        event_col: &str,
        covariates: &[&str],
    ) -> EvaluationResult<RSurvivalModel> {
        self.install_package("survival").await?;

        let covariates_str = if covariates.is_empty() {
            "1".to_string()
        } else {
            covariates.join(" + ")
        };

        let formula = format!("Surv({}, {}) ~ {}", time_col, event_col, covariates_str);

        let script = format!(
            r#"
            library(survival)
            if (!exists("{}")) {{
                stop("Data frame '{}' does not exist")
            }}
            
            model <- coxph({}, data = {})
            summary_model <- summary(model)
            
            cat("concordance:", summary_model$concordance[1], "\n")
            cat("log_likelihood:", model$loglik[2], "\n")
            cat("aic:", AIC(model), "\n")
            cat("coefficients:", paste(names(coef(model)), collapse=","), "\n")
            cat("estimates:", paste(coef(model), collapse=","), "\n")
            cat("hazard_ratios:", paste(exp(coef(model)), collapse=","), "\n")
            cat("p_values:", paste(summary_model$coefficients[, "Pr(>|z|)"], collapse=","), "\n")
            "#,
            df_name, df_name, formula, df_name
        );

        let output = self.execute_script(&script).await?;
        self.parse_survival_model(&output)
    }

    /// Parse logistic regression model results
    fn parse_logistic_model(&self, output: &str) -> EvaluationResult<RLogisticModel> {
        let mut coefficients = Vec::new();
        let mut estimates = Vec::new();
        let mut std_errors = Vec::new();
        let mut z_values = Vec::new();
        let mut p_values = Vec::new();
        let mut aic = 0.0;
        let mut deviance = 0.0;
        let mut null_deviance = 0.0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "coefficients" => {
                        coefficients = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "estimates" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                estimates.push(val);
                            }
                        }
                    }
                    "std_errors" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                std_errors.push(val);
                            }
                        }
                    }
                    "z_values" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                z_values.push(val);
                            }
                        }
                    }
                    "p_values" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                p_values.push(val);
                            }
                        }
                    }
                    "aic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            aic = val;
                        }
                    }
                    "deviance" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            deviance = val;
                        }
                    }
                    "null_deviance" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            null_deviance = val;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RLogisticModel {
            coefficients,
            estimates,
            std_errors,
            z_values,
            p_values,
            aic,
            deviance,
            null_deviance,
        })
    }

    /// Parse random forest model results
    fn parse_random_forest_model(&self, output: &str) -> EvaluationResult<RRandomForestModel> {
        let mut ntree = 0;
        let mut mtry = 0;
        let mut oob_error = 0.0;
        let mut variable_importance = Vec::new();
        let mut variable_names = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "ntree" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            ntree = val;
                        }
                    }
                    "mtry" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            mtry = val;
                        }
                    }
                    "oob_error" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            oob_error = val;
                        }
                    }
                    "variable_importance" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                variable_importance.push(val);
                            }
                        }
                    }
                    "variable_names" => {
                        variable_names = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    _ => {}
                }
            }
        }

        Ok(RRandomForestModel {
            ntree,
            mtry,
            oob_error,
            variable_importance,
            variable_names,
        })
    }

    /// Parse GAM model results
    fn parse_gam_model(&self, output: &str) -> EvaluationResult<RGamModel> {
        let mut deviance_explained = 0.0;
        let mut r_squared = 0.0;
        let mut aic = 0.0;
        let mut bic = 0.0;
        let mut gcv_score = 0.0;
        let mut smooth_terms = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "deviance_explained" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            deviance_explained = val;
                        }
                    }
                    "r_squared" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            r_squared = val;
                        }
                    }
                    "aic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            aic = val;
                        }
                    }
                    "bic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            bic = val;
                        }
                    }
                    "gcv_score" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            gcv_score = val;
                        }
                    }
                    "smooth_terms" => {
                        smooth_terms = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    _ => {}
                }
            }
        }

        Ok(RGamModel {
            deviance_explained,
            r_squared,
            aic,
            bic,
            gcv_score,
            smooth_terms,
        })
    }

    /// Parse ARIMA model results
    fn parse_arima_model(&self, output: &str) -> EvaluationResult<RArimaModel> {
        let mut aic = 0.0;
        let mut log_likelihood = 0.0;
        let mut coefficients = Vec::new();
        let mut estimates = Vec::new();
        let mut std_errors = Vec::new();
        let mut residual_variance = 0.0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "aic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            aic = val;
                        }
                    }
                    "log_likelihood" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            log_likelihood = val;
                        }
                    }
                    "coefficients" => {
                        coefficients = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "estimates" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                estimates.push(val);
                            }
                        }
                    }
                    "std_errors" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                std_errors.push(val);
                            }
                        }
                    }
                    "residual_variance" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            residual_variance = val;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RArimaModel {
            aic,
            log_likelihood,
            coefficients,
            estimates,
            std_errors,
            residual_variance,
        })
    }

    /// Parse PCA results
    fn parse_pca_result(&self, output: &str) -> EvaluationResult<RPcaResult> {
        let mut variance_explained = Vec::new();
        let mut cumulative_variance = Vec::new();
        let mut component_names = Vec::new();
        let mut n_components = 0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "variance_explained" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                variance_explained.push(val);
                            }
                        }
                    }
                    "cumulative_variance" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                cumulative_variance.push(val);
                            }
                        }
                    }
                    "component_names" => {
                        component_names = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "n_components" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            n_components = val;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RPcaResult {
            variance_explained,
            cumulative_variance,
            component_names,
            n_components,
        })
    }

    /// Parse k-means clustering results
    fn parse_kmeans_result(&self, output: &str) -> EvaluationResult<RKmeansResult> {
        let mut centers = Vec::new();
        let mut cluster_sizes = Vec::new();
        let mut within_ss = Vec::new();
        let mut total_within_ss = 0.0;
        let mut between_ss = 0.0;
        let mut total_ss = 0.0;
        let mut k = 0;

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "centers" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                centers.push(val);
                            }
                        }
                    }
                    "cluster_sizes" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<i32>() {
                                cluster_sizes.push(val);
                            }
                        }
                    }
                    "within_ss" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                within_ss.push(val);
                            }
                        }
                    }
                    "total_within_ss" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            total_within_ss = val;
                        }
                    }
                    "between_ss" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            between_ss = val;
                        }
                    }
                    "total_ss" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            total_ss = val;
                        }
                    }
                    "k" => {
                        if let Ok(val) = parts[1].parse::<i32>() {
                            k = val;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RKmeansResult {
            centers,
            cluster_sizes,
            within_ss,
            total_within_ss,
            between_ss,
            total_ss,
            k,
        })
    }

    /// Parse survival analysis model results
    fn parse_survival_model(&self, output: &str) -> EvaluationResult<RSurvivalModel> {
        let mut concordance = 0.0;
        let mut log_likelihood = 0.0;
        let mut aic = 0.0;
        let mut coefficients = Vec::new();
        let mut estimates = Vec::new();
        let mut hazard_ratios = Vec::new();
        let mut p_values = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[0].trim_end_matches(':') {
                    "concordance" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            concordance = val;
                        }
                    }
                    "log_likelihood" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            log_likelihood = val;
                        }
                    }
                    "aic" => {
                        if let Ok(val) = parts[1].parse::<f64>() {
                            aic = val;
                        }
                    }
                    "coefficients" => {
                        coefficients = parts[1].split(',').map(|s| s.to_string()).collect();
                    }
                    "estimates" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                estimates.push(val);
                            }
                        }
                    }
                    "hazard_ratios" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                hazard_ratios.push(val);
                            }
                        }
                    }
                    "p_values" => {
                        for part in parts[1].split(',') {
                            if let Ok(val) = part.parse::<f64>() {
                                p_values.push(val);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(RSurvivalModel {
            concordance,
            log_likelihood,
            aic,
            coefficients,
            estimates,
            hazard_ratios,
            p_values,
        })
    }
}

impl Drop for RSession {
    fn drop(&mut self) {
        // Clean up working directory
        if self.working_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.working_dir);
        }
        debug!("Cleaned up R session: {}", self.session_id);
    }
}

/// Utility functions for R integration
pub mod utils {
    use super::*;

    /// Convert Rust vector to R vector string
    pub fn rust_to_r_vector(data: &[f64]) -> String {
        format!(
            "c({})",
            data.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    /// Convert Rust data to R data frame string
    pub fn rust_to_r_dataframe(df: &RDataFrame) -> String {
        let mut result = String::from("data.frame(");

        for (i, column) in df.columns.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }

            let column_data: Vec<String> = df
                .data
                .iter()
                .map(|row| match &row[i] {
                    RValue::Numeric(n) => n.to_string(),
                    RValue::Integer(i) => i.to_string(),
                    RValue::String(s) => format!("\"{}\"", s),
                    RValue::Logical(b) => {
                        if *b {
                            "TRUE".to_string()
                        } else {
                            "FALSE".to_string()
                        }
                    }
                    RValue::NA => "NA".to_string(),
                })
                .collect();

            result.push_str(&format!("{} = c({})", column, column_data.join(", ")));
        }

        result.push(')');
        result
    }

    /// Check if R is available on the system
    pub fn is_r_available() -> bool {
        RSession::find_r_executable().is_ok()
    }

    /// Get R version information
    pub async fn get_r_version() -> EvaluationResult<String> {
        let output = tokio::process::Command::new("R")
            .arg("--version")
            .output()
            .await
            .map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to get R version: {}", e),
                source: Some(Box::new(e)),
            })?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(EvaluationError::ProcessingError {
                message: "Failed to get R version".to_string(),
                source: None,
            }
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_r_session_creation() {
        if utils::is_r_available() {
            let session = RSession::new().await;
            assert!(session.is_ok());
        }
    }

    #[tokio::test]
    async fn test_simple_r_script() {
        if utils::is_r_available() {
            let session = RSession::new().await.unwrap();
            let result = session.execute_script("cat('Hello from R')").await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_t_test() {
        if utils::is_r_available() {
            let mut session = RSession::new().await.unwrap();
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let result = session.t_test(&data, Some(3.0)).await;
            assert!(result.is_ok());

            let test_result = result.unwrap();
            assert!(test_result.p_value >= 0.0 && test_result.p_value <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_correlation() {
        if utils::is_r_available() {
            let mut session = RSession::new().await.unwrap();
            let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
            let result = session.correlation(&x, &y, "pearson").await;
            assert!(result.is_ok());

            let corr_result = result.unwrap();
            assert!(corr_result.p_value >= 0.0 && corr_result.p_value <= 1.0);
        }
    }

    #[test]
    fn test_utils() {
        let data = vec![1.0, 2.0, 3.0];
        let r_vector = utils::rust_to_r_vector(&data);
        assert_eq!(r_vector, "c(1, 2, 3)");

        let df = RDataFrame {
            columns: vec!["x".to_string(), "y".to_string()],
            data: vec![
                vec![RValue::Numeric(1.0), RValue::Numeric(2.0)],
                vec![RValue::Numeric(3.0), RValue::Numeric(4.0)],
            ],
        };
        let r_df = utils::rust_to_r_dataframe(&df);
        assert!(r_df.contains("data.frame"));
        assert!(r_df.contains("x = c(1, 3)"));
        assert!(r_df.contains("y = c(2, 4)"));
    }
}
