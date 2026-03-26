//! RDataFrame and remaining data model types for R integration

use crate::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::rsession::RValue;

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
            for value in values {
                self.data.push(vec![value]);
            }
        } else {
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
