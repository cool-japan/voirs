//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{EvaluationError, EvaluationResult};
use tokio::process::Command;

use super::rdataframe_and_models::RDataFrame;
use super::rsession::{RSession, RValue};

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
