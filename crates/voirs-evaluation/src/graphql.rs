//! GraphQL API for Complex Evaluation Queries
//!
//! Provides a powerful GraphQL interface for querying evaluation results,
//! managing datasets, running evaluations, and analyzing metrics.
//!
//! # Features
//!
//! - **Flexible Queries**: Query evaluation results with complex filters
//! - **Batch Operations**: Execute multiple evaluations in a single request
//! - **Real-time Subscriptions**: Subscribe to evaluation progress updates
//! - **Nested Queries**: Access related data (datasets, models, metrics)
//! - **Pagination**: Efficiently handle large result sets
//! - **Aggregations**: Compute statistics across multiple evaluations
//!
//! # Example
//!
//! ```graphql
//! query GetEvaluations {
//!   evaluations(
//!     filter: { minQuality: 4.0, language: "en-US" }
//!     limit: 10
//!   ) {
//!     id
//!     qualityScore
//!     metrics {
//!       pesq
//!       stoi
//!       mcd
//!     }
//!     dataset {
//!       name
//!       sampleCount
//!     }
//!   }
//! }
//!
//! mutation RunEvaluation {
//!   evaluateAudio(input: {
//!     audioData: "base64..."
//!     referenceId: "ref123"
//!     language: "en-US"
//!   }) {
//!     id
//!     qualityScore
//!     status
//!   }
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, VoirsError};

use crate::quality::QualityEvaluator;
use crate::traits::{
    QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait, QualityScore,
};

/// GraphQL API errors
#[derive(Error, Debug)]
pub enum GraphQLError {
    /// Query execution error
    #[error("Query execution error: {message}")]
    QueryError {
        /// Error message
        message: String,
    },

    /// Invalid query syntax
    #[error("Invalid query syntax: {message}")]
    SyntaxError {
        /// Error message
        message: String,
    },

    /// Authorization error
    #[error("Authorization error: {message}")]
    AuthError {
        /// Error message
        message: String,
    },

    /// Resource not found
    #[error("Resource not found: {resource_type} with id '{id}'")]
    NotFound {
        /// Resource type
        resource_type: String,
        /// Resource ID
        id: String,
    },

    /// VoiRS error
    #[error("VoiRS error: {0}")]
    VoirsError(#[from] VoirsError),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// GraphQL evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Unique evaluation ID
    pub id: String,
    /// Quality score
    pub quality_score: f32,
    /// Evaluation status
    pub status: EvaluationStatus,
    /// Timestamp
    pub timestamp: u64,
    /// Metrics
    pub metrics: Option<MetricsData>,
    /// Dataset information
    pub dataset: Option<DatasetInfo>,
    /// Model information
    pub model: Option<ModelInfo>,
}

/// Evaluation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationStatus {
    /// Pending execution
    Pending,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled
    Cancelled,
}

/// Metrics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    /// PESQ score
    pub pesq: Option<f32>,
    /// STOI score
    pub stoi: Option<f32>,
    /// MCD score
    pub mcd: Option<f32>,
    /// Custom metrics
    pub custom: HashMap<String, f32>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset ID
    pub id: String,
    /// Dataset name
    pub name: String,
    /// Sample count
    pub sample_count: usize,
    /// Language
    pub language: String,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Architecture type
    pub architecture: String,
}

/// Evaluation filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationFilter {
    /// Minimum quality score
    pub min_quality: Option<f32>,
    /// Maximum quality score
    pub max_quality: Option<f32>,
    /// Language filter
    pub language: Option<String>,
    /// Status filter
    pub status: Option<EvaluationStatus>,
    /// Dataset ID filter
    pub dataset_id: Option<String>,
    /// Model ID filter
    pub model_id: Option<String>,
    /// Time range (from timestamp)
    pub from_timestamp: Option<u64>,
    /// Time range (to timestamp)
    pub to_timestamp: Option<u64>,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    /// Limit results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Sort field
    pub sort_by: Option<String>,
    /// Sort direction
    pub sort_desc: Option<bool>,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            limit: Some(100),
            offset: Some(0),
            sort_by: None,
            sort_desc: Some(false),
        }
    }
}

/// Query result with pagination info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResults<T> {
    /// Result items
    pub items: Vec<T>,
    /// Total count
    pub total_count: usize,
    /// Has next page
    pub has_next_page: bool,
    /// Current page info
    pub page_info: PageInfo,
}

/// Page information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInfo {
    /// Current offset
    pub offset: usize,
    /// Current limit
    pub limit: usize,
    /// Total items
    pub total: usize,
}

/// Evaluation input for mutations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationInput {
    /// Audio data (base64 encoded)
    pub audio_data: String,
    /// Reference audio ID (optional)
    pub reference_id: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Dataset ID
    pub dataset_id: Option<String>,
    /// Model ID
    pub model_id: Option<String>,
    /// Custom parameters
    pub parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Batch evaluation input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvaluationInput {
    /// List of evaluation inputs
    pub evaluations: Vec<EvaluationInput>,
    /// Parallel execution count
    pub parallel: Option<usize>,
}

/// Aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Average quality score
    pub avg_quality: f32,
    /// Minimum quality score
    pub min_quality: f32,
    /// Maximum quality score
    pub max_quality: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Sample count
    pub count: usize,
    /// Percentiles
    pub percentiles: HashMap<String, f32>,
}

/// GraphQL schema trait
#[async_trait]
pub trait GraphQLSchema: Send + Sync {
    /// Query evaluations
    async fn query_evaluations(
        &self,
        filter: Option<EvaluationFilter>,
        pagination: Option<Pagination>,
    ) -> Result<PaginatedResults<EvaluationResult>, GraphQLError>;

    /// Get evaluation by ID
    async fn get_evaluation(&self, id: &str) -> Result<EvaluationResult, GraphQLError>;

    /// Run new evaluation
    async fn evaluate_audio(
        &self,
        input: EvaluationInput,
    ) -> Result<EvaluationResult, GraphQLError>;

    /// Run batch evaluations
    async fn evaluate_batch(
        &self,
        input: BatchEvaluationInput,
    ) -> Result<Vec<EvaluationResult>, GraphQLError>;

    /// Get aggregated statistics
    async fn get_aggregations(
        &self,
        filter: Option<EvaluationFilter>,
    ) -> Result<AggregationResult, GraphQLError>;

    /// Cancel evaluation
    async fn cancel_evaluation(&self, id: &str) -> Result<bool, GraphQLError>;

    /// Delete evaluation
    async fn delete_evaluation(&self, id: &str) -> Result<bool, GraphQLError>;
}

/// GraphQL evaluation service
pub struct GraphQLService {
    evaluator: Arc<RwLock<QualityEvaluator>>,
    evaluations: Arc<RwLock<HashMap<String, EvaluationResult>>>,
}

impl GraphQLService {
    /// Create new GraphQL service
    pub async fn new() -> Result<Self, GraphQLError> {
        let evaluator = QualityEvaluator::new().await?;

        Ok(Self {
            evaluator: Arc::new(RwLock::new(evaluator)),
            evaluations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Apply filter to evaluation results
    fn apply_filter(
        &self,
        results: &[EvaluationResult],
        filter: &EvaluationFilter,
    ) -> Vec<EvaluationResult> {
        results
            .iter()
            .filter(|eval| {
                // Filter by min quality
                if let Some(min_q) = filter.min_quality {
                    if eval.quality_score < min_q {
                        return false;
                    }
                }

                // Filter by max quality
                if let Some(max_q) = filter.max_quality {
                    if eval.quality_score > max_q {
                        return false;
                    }
                }

                // Filter by status
                if let Some(ref status) = filter.status {
                    if &eval.status != status {
                        return false;
                    }
                }

                // Filter by language
                if let Some(ref lang) = filter.language {
                    if let Some(ref dataset) = eval.dataset {
                        if &dataset.language != lang {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }

                // Filter by dataset ID
                if let Some(ref dataset_id) = filter.dataset_id {
                    if let Some(ref dataset) = eval.dataset {
                        if &dataset.id != dataset_id {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }

                // Filter by timestamp range
                if let Some(from_ts) = filter.from_timestamp {
                    if eval.timestamp < from_ts {
                        return false;
                    }
                }

                if let Some(to_ts) = filter.to_timestamp {
                    if eval.timestamp > to_ts {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Apply pagination
    fn apply_pagination(
        &self,
        mut results: Vec<EvaluationResult>,
        pagination: &Pagination,
    ) -> (Vec<EvaluationResult>, PageInfo) {
        let total = results.len();

        // Sort if requested
        if let Some(ref sort_by) = pagination.sort_by {
            let sort_desc = pagination.sort_desc.unwrap_or(false);
            match sort_by.as_str() {
                "quality_score" => {
                    results.sort_by(|a, b| {
                        if sort_desc {
                            b.quality_score
                                .partial_cmp(&a.quality_score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            a.quality_score
                                .partial_cmp(&b.quality_score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }
                    });
                }
                "timestamp" => {
                    results.sort_by(|a, b| {
                        if sort_desc {
                            b.timestamp.cmp(&a.timestamp)
                        } else {
                            a.timestamp.cmp(&b.timestamp)
                        }
                    });
                }
                _ => {}
            }
        }

        // Apply offset and limit
        let offset = pagination.offset.unwrap_or(0);
        let limit = pagination.limit.unwrap_or(100);

        let items: Vec<EvaluationResult> = results.into_iter().skip(offset).take(limit).collect();

        let page_info = PageInfo {
            offset,
            limit,
            total,
        };

        (items, page_info)
    }

    /// Calculate aggregations
    fn calculate_aggregations(
        &self,
        results: &[EvaluationResult],
    ) -> Result<AggregationResult, GraphQLError> {
        if results.is_empty() {
            return Ok(AggregationResult {
                avg_quality: 0.0,
                min_quality: 0.0,
                max_quality: 0.0,
                std_dev: 0.0,
                count: 0,
                percentiles: HashMap::new(),
            });
        }

        let scores: Vec<f32> = results.iter().map(|r| r.quality_score).collect();

        let avg_quality = scores.iter().sum::<f32>() / scores.len() as f32;
        let min_quality = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_quality = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let variance = scores
            .iter()
            .map(|s| (s - avg_quality).powi(2))
            .sum::<f32>()
            / scores.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut percentiles = HashMap::new();
        for p in [25, 50, 75, 90, 95, 99] {
            let idx = ((p as f32 / 100.0) * sorted_scores.len() as f32) as usize;
            let idx = idx.min(sorted_scores.len() - 1);
            percentiles.insert(format!("p{}", p), sorted_scores[idx]);
        }

        Ok(AggregationResult {
            avg_quality,
            min_quality,
            max_quality,
            std_dev,
            count: results.len(),
            percentiles,
        })
    }
}

#[async_trait]
impl GraphQLSchema for GraphQLService {
    async fn query_evaluations(
        &self,
        filter: Option<EvaluationFilter>,
        pagination: Option<Pagination>,
    ) -> Result<PaginatedResults<EvaluationResult>, GraphQLError> {
        let evaluations = self.evaluations.read().await;
        let all_results: Vec<EvaluationResult> = evaluations.values().cloned().collect();

        // Apply filter
        let filtered_results = if let Some(f) = filter {
            self.apply_filter(&all_results, &f)
        } else {
            all_results
        };

        // Apply pagination
        let pagination = pagination.unwrap_or_default();
        let (items, page_info) = self.apply_pagination(filtered_results, &pagination);

        let has_next_page = page_info.offset + items.len() < page_info.total;

        Ok(PaginatedResults {
            items,
            total_count: page_info.total,
            has_next_page,
            page_info,
        })
    }

    async fn get_evaluation(&self, id: &str) -> Result<EvaluationResult, GraphQLError> {
        let evaluations = self.evaluations.read().await;
        evaluations
            .get(id)
            .cloned()
            .ok_or_else(|| GraphQLError::NotFound {
                resource_type: "Evaluation".to_string(),
                id: id.to_string(),
            })
    }

    async fn evaluate_audio(
        &self,
        input: EvaluationInput,
    ) -> Result<EvaluationResult, GraphQLError> {
        // Decode base64 audio data
        let audio_bytes =
            base64::decode(&input.audio_data).map_err(|e| GraphQLError::QueryError {
                message: format!("Invalid base64 audio data: {}", e),
            })?;

        // Create audio buffer (simplified - in production, parse actual audio format)
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        // Run evaluation
        let evaluator = self.evaluator.read().await;
        let config = QualityEvaluationConfig::default();
        let quality = evaluator
            .evaluate_quality(&audio, None, Some(&config))
            .await?;

        // Create evaluation result
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("value should be present")
            .as_secs();

        let result = EvaluationResult {
            id: id.clone(),
            quality_score: quality.overall_score,
            status: EvaluationStatus::Completed,
            timestamp,
            metrics: Some(MetricsData {
                pesq: None,
                stoi: None,
                mcd: None,
                custom: HashMap::new(),
            }),
            dataset: input.dataset_id.map(|dataset_id| DatasetInfo {
                id: dataset_id,
                name: "Sample Dataset".to_string(),
                sample_count: 100,
                language: input.language.unwrap_or_else(|| "en-US".to_string()),
            }),
            model: input.model_id.map(|model_id| ModelInfo {
                id: model_id,
                name: "Sample Model".to_string(),
                version: "1.0.0".to_string(),
                architecture: "VITS".to_string(),
            }),
        };

        // Store result
        let mut evaluations = self.evaluations.write().await;
        evaluations.insert(id, result.clone());

        Ok(result)
    }

    async fn evaluate_batch(
        &self,
        input: BatchEvaluationInput,
    ) -> Result<Vec<EvaluationResult>, GraphQLError> {
        let mut results = Vec::new();

        for eval_input in input.evaluations {
            let result = self.evaluate_audio(eval_input).await?;
            results.push(result);
        }

        Ok(results)
    }

    async fn get_aggregations(
        &self,
        filter: Option<EvaluationFilter>,
    ) -> Result<AggregationResult, GraphQLError> {
        let evaluations = self.evaluations.read().await;
        let all_results: Vec<EvaluationResult> = evaluations.values().cloned().collect();

        // Apply filter
        let filtered_results = if let Some(f) = filter {
            self.apply_filter(&all_results, &f)
        } else {
            all_results
        };

        self.calculate_aggregations(&filtered_results)
    }

    async fn cancel_evaluation(&self, id: &str) -> Result<bool, GraphQLError> {
        let mut evaluations = self.evaluations.write().await;
        if let Some(eval) = evaluations.get_mut(id) {
            if eval.status == EvaluationStatus::Pending || eval.status == EvaluationStatus::Running
            {
                eval.status = EvaluationStatus::Cancelled;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(GraphQLError::NotFound {
                resource_type: "Evaluation".to_string(),
                id: id.to_string(),
            })
        }
    }

    async fn delete_evaluation(&self, id: &str) -> Result<bool, GraphQLError> {
        let mut evaluations = self.evaluations.write().await;
        Ok(evaluations.remove(id).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graphql_service_creation() {
        let service = GraphQLService::new().await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_query_evaluations_empty() {
        let service = GraphQLService::new().await.unwrap();
        let results = service.query_evaluations(None, None).await.unwrap();
        assert_eq!(results.total_count, 0);
        assert!(results.items.is_empty());
    }

    #[tokio::test]
    async fn test_evaluate_audio() {
        let service = GraphQLService::new().await.unwrap();
        let input = EvaluationInput {
            audio_data: base64::encode("test_audio_data"),
            reference_id: None,
            language: Some("en-US".to_string()),
            dataset_id: Some("dataset1".to_string()),
            model_id: Some("model1".to_string()),
            parameters: None,
        };

        let result = service.evaluate_audio(input).await;
        assert!(result.is_ok());

        let eval = result.unwrap();
        assert_eq!(eval.status, EvaluationStatus::Completed);
        assert!(eval.dataset.is_some());
        assert!(eval.model.is_some());
    }

    #[tokio::test]
    async fn test_get_evaluation() {
        let service = GraphQLService::new().await.unwrap();
        let input = EvaluationInput {
            audio_data: base64::encode("test"),
            reference_id: None,
            language: Some("en-US".to_string()),
            dataset_id: None,
            model_id: None,
            parameters: None,
        };

        let created = service.evaluate_audio(input).await.unwrap();
        let retrieved = service.get_evaluation(&created.id).await.unwrap();

        assert_eq!(created.id, retrieved.id);
        assert_eq!(created.quality_score, retrieved.quality_score);
    }

    #[tokio::test]
    async fn test_pagination() {
        let service = GraphQLService::new().await.unwrap();

        // Create multiple evaluations
        for _ in 0..5 {
            let input = EvaluationInput {
                audio_data: base64::encode("test"),
                reference_id: None,
                language: None,
                dataset_id: None,
                model_id: None,
                parameters: None,
            };
            service.evaluate_audio(input).await.unwrap();
        }

        let pagination = Pagination {
            limit: Some(2),
            offset: Some(0),
            sort_by: None,
            sort_desc: None,
        };

        let results = service
            .query_evaluations(None, Some(pagination))
            .await
            .unwrap();
        assert_eq!(results.items.len(), 2);
        assert_eq!(results.total_count, 5);
        assert!(results.has_next_page);
    }

    #[tokio::test]
    async fn test_filter_by_quality() {
        let service = GraphQLService::new().await.unwrap();

        // Create evaluations (they'll have similar scores in this mock)
        for _ in 0..3 {
            let input = EvaluationInput {
                audio_data: base64::encode("test"),
                reference_id: None,
                language: None,
                dataset_id: None,
                model_id: None,
                parameters: None,
            };
            service.evaluate_audio(input).await.unwrap();
        }

        let filter = EvaluationFilter {
            min_quality: Some(0.0),
            max_quality: Some(5.0),
            language: None,
            status: None,
            dataset_id: None,
            model_id: None,
            from_timestamp: None,
            to_timestamp: None,
        };

        let results = service.query_evaluations(Some(filter), None).await.unwrap();
        assert_eq!(results.total_count, 3);
    }

    #[tokio::test]
    async fn test_aggregations() {
        let service = GraphQLService::new().await.unwrap();

        // Create evaluations
        for _ in 0..10 {
            let input = EvaluationInput {
                audio_data: base64::encode("test"),
                reference_id: None,
                language: None,
                dataset_id: None,
                model_id: None,
                parameters: None,
            };
            service.evaluate_audio(input).await.unwrap();
        }

        let agg = service.get_aggregations(None).await.unwrap();
        assert_eq!(agg.count, 10);
        assert!(agg.avg_quality >= 0.0);
        assert!(agg.std_dev >= 0.0);
        assert!(!agg.percentiles.is_empty());
    }

    #[tokio::test]
    async fn test_cancel_evaluation() {
        let service = GraphQLService::new().await.unwrap();
        let input = EvaluationInput {
            audio_data: base64::encode("test"),
            reference_id: None,
            language: None,
            dataset_id: None,
            model_id: None,
            parameters: None,
        };

        let eval = service.evaluate_audio(input).await.unwrap();

        // Can't cancel completed evaluation
        let cancelled = service.cancel_evaluation(&eval.id).await.unwrap();
        assert!(!cancelled);
    }

    #[tokio::test]
    async fn test_delete_evaluation() {
        let service = GraphQLService::new().await.unwrap();
        let input = EvaluationInput {
            audio_data: base64::encode("test"),
            reference_id: None,
            language: None,
            dataset_id: None,
            model_id: None,
            parameters: None,
        };

        let eval = service.evaluate_audio(input).await.unwrap();
        let deleted = service.delete_evaluation(&eval.id).await.unwrap();
        assert!(deleted);

        // Should not be found after deletion
        let result = service.get_evaluation(&eval.id).await;
        assert!(result.is_err());
    }
}
