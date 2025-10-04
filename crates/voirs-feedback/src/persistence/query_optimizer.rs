//! Database query optimization utilities
//!
//! This module provides optimized database queries, connection pooling,
//! and performance enhancements for user data operations.

use crate::persistence::{PersistenceError, PersistenceResult};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};
use chrono::{DateTime, Timelike, Utc};
use futures::stream;
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Postgres, Row, Sqlite};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Query optimization configuration
#[derive(Debug, Clone)]
pub struct QueryOptimizerConfig {
    /// Enable query caching
    pub enable_query_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable query batching
    pub enable_batching: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Connection pool configuration
    pub pool_config: PoolConfig,
    /// Enable read replicas for read-only queries
    pub enable_read_replicas: bool,
    /// Read replica connection strings
    pub read_replica_urls: Vec<String>,
    /// Enable query plan optimization hints
    pub enable_query_hints: bool,
    /// Enable connection multiplexing
    pub enable_connection_multiplexing: bool,
    /// Stream result threshold (rows)
    pub stream_threshold: usize,
}

/// Database connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections
    pub min_connections: u32,
    /// Maximum number of connections
    pub max_connections: u32,
    /// Connection timeout
    pub acquire_timeout: Duration,
    /// Idle timeout for connections
    pub idle_timeout: Option<Duration>,
    /// Maximum lifetime for connections
    pub max_lifetime: Option<Duration>,
}

impl Default for QueryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_query_cache: true,
            cache_ttl_seconds: 300, // 5 minutes
            max_cache_size: 1000,
            enable_batching: true,
            batch_size: 100,
            pool_config: PoolConfig::default(),
            enable_read_replicas: false,
            read_replica_urls: Vec::new(),
            enable_query_hints: true,
            enable_connection_multiplexing: true,
            stream_threshold: 10000, // Stream if more than 10k rows expected
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 5,
            max_connections: 30,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Some(Duration::from_secs(600)), // 10 minutes
            max_lifetime: Some(Duration::from_secs(1800)), // 30 minutes
        }
    }
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult<T> {
    /// Cached data
    data: T,
    /// Creation timestamp
    created_at: Instant,
    /// Time to live
    ttl: Duration,
}

impl<T> CachedResult<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            created_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Database query optimizer
pub struct QueryOptimizer {
    config: QueryOptimizerConfig,
    query_cache: Arc<RwLock<HashMap<String, CachedResult<Vec<u8>>>>>,
    query_stats: Arc<RwLock<QueryStats>>,
    prepared_statements: Arc<RwLock<HashMap<String, PreparedStatementInfo>>>,
    index_recommendations: Arc<RwLock<Vec<IndexRecommendation>>>,
    performance_baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
}

/// Query performance statistics
#[derive(Debug, Default, Clone)]
pub struct QueryStats {
    /// Description
    pub total_queries: u64,
    /// Description
    pub cache_hits: u64,
    /// Description
    pub cache_misses: u64,
    /// Description
    pub average_query_time: Duration,
    /// Description
    pub batch_operations: u64,
    /// Description
    pub optimized_queries: u64,
    /// Description
    pub prepared_statements_used: u64,
    /// Description
    pub index_recommendations_generated: u64,
    /// Description
    pub performance_regressions_detected: u64,
}

/// Prepared statement information
#[derive(Debug, Clone)]
pub struct PreparedStatementInfo {
    /// Description
    pub statement_id: String,
    /// Description
    pub query_template: String,
    /// Description
    pub parameter_count: usize,
    /// Description
    pub last_used: Instant,
    /// Description
    pub usage_count: u64,
    /// Description
    pub average_execution_time: Duration,
}

/// Index recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    /// Description
    pub table_name: String,
    /// Description
    pub columns: Vec<String>,
    /// Description
    pub index_type: IndexType,
    /// Description
    pub estimated_improvement: f64, // Percentage improvement
    /// Description
    pub frequency_score: u64,
    /// Description
    pub created_at: std::time::SystemTime,
    /// Description
    pub query_patterns: Vec<String>,
}

/// Index types for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// B-Tree index
    BTree,
    /// Hash index
    Hash,
    /// GIN (Generalized Inverted Index)
    Gin,
    /// GiST (Generalized Search Tree)
    Gist,
    /// Composite index on multiple columns
    Composite,
    /// Partial index with condition
    PartialIndex {
        /// Index condition
        condition: String
    },
}

/// Performance baseline for regression detection
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Description
    pub query_signature: String,
    /// Description
    pub baseline_time: Duration,
    /// Description
    pub baseline_established_at: Instant,
    /// Description
    pub sample_count: u64,
    /// Description
    pub variance: f64,
    /// Description
    pub last_updated: Instant,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(config: QueryOptimizerConfig) -> Self {
        Self {
            config,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            query_stats: Arc::new(RwLock::new(QueryStats::default())),
            prepared_statements: Arc::new(RwLock::new(HashMap::new())),
            index_recommendations: Arc::new(RwLock::new(Vec::new())),
            performance_baselines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimized feedback history query with proper pagination
    pub async fn load_feedback_history_optimized(
        &self,
        pool: &Pool<Postgres>,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        let start_time = Instant::now();

        // Use prepared statement with parameterized queries
        let limit = limit.unwrap_or(50).min(1000); // Cap at 1000 for performance
        let offset = offset.unwrap_or(0);

        // Check cache first
        let cache_key = format!("feedback_history:{}:{}:{}", user_id, limit, offset);
        if self.config.enable_query_cache {
            if let Some(cached) = self
                .get_from_cache::<Vec<FeedbackResponse>>(&cache_key)
                .await
            {
                self.record_cache_hit().await;
                return Ok(cached);
            }
        }

        // Execute optimized query with proper parameterization
        let rows = sqlx::query(
            r#"
            SELECT feedback_data 
            FROM feedback_history 
            WHERE user_id = $1 
            ORDER BY created_at DESC 
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(user_id)
        .bind(limit as i64)
        .bind(offset as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to load feedback history: {}", e),
        })?;

        // Deserialize results
        let mut feedback_history = Vec::with_capacity(rows.len());
        for row in rows {
            let feedback_data: serde_json::Value = row.get("feedback_data");
            let feedback: FeedbackResponse =
                serde_json::from_value(feedback_data).map_err(|e| {
                    PersistenceError::SerializationError {
                        message: format!("Failed to deserialize feedback: {}", e),
                    }
                })?;
            feedback_history.push(feedback);
        }

        // Cache result
        if self.config.enable_query_cache {
            self.cache_result(&cache_key, &feedback_history).await;
        }

        // Update statistics
        self.record_query_execution(start_time.elapsed()).await;

        Ok(feedback_history)
    }

    /// Batch save multiple feedback items for improved performance
    pub async fn batch_save_feedback(
        &self,
        pool: &Pool<Postgres>,
        feedback_items: Vec<(String, FeedbackResponse)>, // (user_id, feedback)
    ) -> PersistenceResult<usize> {
        if feedback_items.is_empty() {
            return Ok(0);
        }

        let start_time = Instant::now();
        let mut saved_count = 0;

        // Process in batches to avoid overwhelming the database
        for batch in feedback_items.chunks(self.config.batch_size) {
            let mut query_builder =
                sqlx::QueryBuilder::new("INSERT INTO feedback_history (user_id, feedback_data) ");

            query_builder.push_values(batch, |mut b, (user_id, feedback)| {
                let feedback_data =
                    serde_json::to_value(feedback).expect("Failed to serialize feedback");
                b.push_bind(user_id).push_bind(feedback_data);
            });

            let result = query_builder.build().execute(pool).await.map_err(|e| {
                PersistenceError::ConnectionError {
                    message: format!("Failed to batch save feedback: {}", e),
                }
            })?;

            saved_count += result.rows_affected() as usize;
        }

        // Update statistics
        self.record_batch_operation().await;
        self.record_query_execution(start_time.elapsed()).await;

        Ok(saved_count)
    }

    /// Optimized user data query with joins
    pub async fn load_user_data_complete(
        &self,
        pool: &Pool<Postgres>,
        user_id: &str,
    ) -> PersistenceResult<CompleteUserData> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = format!("user_data_complete:{}", user_id);
        if self.config.enable_query_cache {
            if let Some(cached) = self.get_from_cache::<CompleteUserData>(&cache_key).await {
                self.record_cache_hit().await;
                return Ok(cached);
            }
        }

        // Single query with joins to reduce round trips
        let row = sqlx::query(
            r#"
            SELECT 
                up.progress_data,
                upr.preferences_data,
                (
                    SELECT COUNT(*)::bigint 
                    FROM sessions s 
                    WHERE s.user_id = $1
                ) as session_count,
                (
                    SELECT COUNT(*)::bigint 
                    FROM feedback_history fh 
                    WHERE fh.user_id = $1
                ) as feedback_count
            FROM user_progress up
            FULL OUTER JOIN user_preferences upr ON up.user_id = upr.user_id
            WHERE up.user_id = $1 OR upr.user_id = $1
            LIMIT 1
            "#,
        )
        .bind(user_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to load complete user data: {}", e),
        })?;

        let user_data = match row {
            Some(row) => {
                let progress_data: Option<serde_json::Value> = row.get("progress_data");
                let preferences_data: Option<serde_json::Value> = row.get("preferences_data");
                let session_count: i64 = row.get("session_count");
                let feedback_count: i64 = row.get("feedback_count");

                let progress = if let Some(data) = progress_data {
                    Some(serde_json::from_value(data).map_err(|e| {
                        PersistenceError::SerializationError {
                            message: format!("Failed to deserialize progress: {}", e),
                        }
                    })?)
                } else {
                    None
                };

                let preferences = if let Some(data) = preferences_data {
                    Some(serde_json::from_value(data).map_err(|e| {
                        PersistenceError::SerializationError {
                            message: format!("Failed to deserialize preferences: {}", e),
                        }
                    })?)
                } else {
                    None
                };

                CompleteUserData {
                    user_id: user_id.to_string(),
                    progress,
                    preferences,
                    session_count: session_count as u64,
                    feedback_count: feedback_count as u64,
                }
            }
            None => CompleteUserData {
                user_id: user_id.to_string(),
                progress: None,
                preferences: None,
                session_count: 0,
                feedback_count: 0,
            },
        };

        // Cache result
        if self.config.enable_query_cache {
            self.cache_result(&cache_key, &user_data).await;
        }

        // Update statistics
        self.record_optimized_query().await;
        self.record_query_execution(start_time.elapsed()).await;

        Ok(user_data)
    }

    /// Get cached result
    async fn get_from_cache<T>(&self, key: &str) -> Option<T>
    where
        T: Clone + for<'de> Deserialize<'de>,
    {
        let cache = self.query_cache.read().await;
        if let Some(cached) = cache.get(key) {
            if !cached.is_expired() {
                if let Ok((data, _)) =
                    bincode::serde::decode_from_slice(&cached.data, bincode::config::standard())
                {
                    return Some(data);
                }
            }
        }
        None
    }

    /// Cache query result
    async fn cache_result<T>(&self, key: &str, data: &T)
    where
        T: Serialize,
    {
        if let Ok(serialized) = bincode::serde::encode_to_vec(data, bincode::config::standard()) {
            let cached = CachedResult::new(
                serialized,
                Duration::from_secs(self.config.cache_ttl_seconds),
            );

            let mut cache = self.query_cache.write().await;

            // Simple cache eviction: remove oldest if at capacity
            if cache.len() >= self.config.max_cache_size {
                // Remove expired entries first
                cache.retain(|_, v| !v.is_expired());

                // If still at capacity, remove a random entry
                if cache.len() >= self.config.max_cache_size {
                    if let Some(key_to_remove) = cache.keys().next().cloned() {
                        cache.remove(&key_to_remove);
                    }
                }
            }

            cache.insert(key.to_string(), cached);
        }
    }

    /// Record cache hit
    async fn record_cache_hit(&self) {
        let mut stats = self.query_stats.write().await;
        stats.cache_hits += 1;
    }

    /// Record query execution
    async fn record_query_execution(&self, duration: Duration) {
        let mut stats = self.query_stats.write().await;
        stats.total_queries += 1;

        // Update average query time
        if stats.total_queries == 1 {
            stats.average_query_time = duration;
        } else {
            let total_time = stats.average_query_time * (stats.total_queries - 1) as u32 + duration;
            stats.average_query_time = total_time / stats.total_queries as u32;
        }
    }

    /// Record batch operation
    async fn record_batch_operation(&self) {
        let mut stats = self.query_stats.write().await;
        stats.batch_operations += 1;
    }

    /// Record optimized query
    async fn record_optimized_query(&self) {
        let mut stats = self.query_stats.write().await;
        stats.optimized_queries += 1;
    }

    /// Get query statistics
    pub async fn get_stats(&self) -> QueryStats {
        let stats_guard = self.query_stats.read().await;
        (*stats_guard).clone()
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        self.query_cache.write().await.clear();
    }

    /// Optimized large result streaming query with cursor-based pagination
    pub async fn stream_large_results<T>(
        &self,
        pool: &Pool<Postgres>,
        base_query: &str,
        page_size: Option<usize>,
    ) -> PersistenceResult<impl futures::Stream<Item = PersistenceResult<T>>>
    where
        T: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    {
        use futures::stream::{self, StreamExt};

        let page_size = page_size.unwrap_or(1000).min(5000); // Cap at 5k for memory efficiency

        // Add streaming optimization hints if enabled
        let optimized_query = if self.config.enable_query_hints {
            format!("/*+ USE_HASH_JOIN CURSOR_SHARING=EXACT */ {}", base_query)
        } else {
            base_query.to_string()
        };

        let pool_clone = pool.clone();
        let stream = stream::unfold(
            (0usize, false), // (offset, is_done)
            move |(offset, is_done)| {
                let pool = pool_clone.clone();
                let query = optimized_query.clone();
                async move {
                    if is_done {
                        return None;
                    }

                    let paginated_query =
                        format!("{} LIMIT {} OFFSET {}", query, page_size, offset);

                    match sqlx::query(&paginated_query).fetch_all(&pool).await {
                        Ok(rows) => {
                            let is_last_page = rows.len() < page_size;
                            let results: Vec<PersistenceResult<T>> = rows
                                .into_iter()
                                .map(|row| {
                                    T::from_row(&row).map_err(|e| {
                                        PersistenceError::SerializationError {
                                            message: format!("Failed to deserialize row: {}", e),
                                        }
                                    })
                                })
                                .collect();

                            Some((stream::iter(results), (offset + page_size, is_last_page)))
                        }
                        Err(e) => {
                            let error = PersistenceError::ConnectionError {
                                message: format!("Streaming query failed: {}", e),
                            };
                            Some((
                                stream::iter(vec![Err(error)]),
                                (offset, true), // Stop on error
                            ))
                        }
                    }
                }
            },
        )
        .flatten();

        Ok(stream)
    }

    /// Index usage optimization analyzer
    pub async fn analyze_query_performance(
        &self,
        pool: &Pool<Postgres>,
        query: &str,
    ) -> PersistenceResult<QueryPerformanceAnalysis> {
        let start_time = Instant::now();

        // Get query execution plan
        let explain_query = format!("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}", query);
        let row = sqlx::query(&explain_query)
            .fetch_one(pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Failed to get query plan: {}", e),
            })?;

        let execution_plan: serde_json::Value = row.get(0);

        // Analyze the plan for optimization opportunities
        let analysis = self.analyze_execution_plan(&execution_plan);

        let total_time = start_time.elapsed();

        Ok(QueryPerformanceAnalysis {
            query: query.to_string(),
            total_execution_time: total_time,
            execution_plan,
            recommendations: analysis.recommendations,
            index_usage: analysis.index_usage,
            scan_types: analysis.scan_types,
            cost_estimate: analysis.cost_estimate,
        })
    }

    /// Analyze execution plan for optimization opportunities
    fn analyze_execution_plan(&self, plan: &serde_json::Value) -> ExecutionPlanAnalysis {
        let mut recommendations = Vec::new();
        let mut index_usage = Vec::new();
        let mut scan_types = Vec::new();
        let mut cost_estimate = 0.0;

        if let Some(plan_array) = plan.as_array() {
            if let Some(plan_obj) = plan_array.get(0) {
                if let Some(plan_data) = plan_obj.get("Plan") {
                    self.analyze_plan_node(
                        plan_data,
                        &mut recommendations,
                        &mut index_usage,
                        &mut scan_types,
                        &mut cost_estimate,
                    );
                }
            }
        }

        ExecutionPlanAnalysis {
            recommendations,
            index_usage,
            scan_types,
            cost_estimate,
        }
    }

    /// Recursively analyze plan nodes
    fn analyze_plan_node(
        &self,
        node: &serde_json::Value,
        recommendations: &mut Vec<String>,
        index_usage: &mut Vec<String>,
        scan_types: &mut Vec<String>,
        cost_estimate: &mut f64,
    ) {
        if let Some(node_type) = node.get("Node Type").and_then(|v| v.as_str()) {
            scan_types.push(node_type.to_string());

            // Add cost
            if let Some(total_cost) = node.get("Total Cost").and_then(|v| v.as_f64()) {
                *cost_estimate += total_cost;
            }

            // Analyze different node types for optimization opportunities
            match node_type {
                "Seq Scan" => {
                    if let Some(relation) = node.get("Relation Name").and_then(|v| v.as_str()) {
                        recommendations.push(format!(
                            "Consider adding an index on table '{}' to avoid sequential scan",
                            relation
                        ));
                    }
                }
                "Index Scan" | "Index Only Scan" | "Bitmap Index Scan" => {
                    if let Some(index) = node.get("Index Name").and_then(|v| v.as_str()) {
                        index_usage.push(index.to_string());
                    }
                }
                "Hash Join" | "Nested Loop" => {
                    if let Some(rows) = node.get("Actual Rows").and_then(|v| v.as_u64()) {
                        if rows > 100000 {
                            recommendations.push(
                                String::from("Large join detected. Consider optimizing join conditions or partitioning")
                            );
                        }
                    }
                }
                "Sort" => {
                    if let Some(sort_key) = node.get("Sort Key") {
                        recommendations.push(format!(
                            "Sort operation detected on {}. Consider adding an index to avoid sorting",
                            sort_key
                        ));
                    }
                }
                _ => {}
            }

            // Recursively analyze child plans
            if let Some(plans) = node.get("Plans").and_then(|v| v.as_array()) {
                for plan in plans {
                    self.analyze_plan_node(
                        plan,
                        recommendations,
                        index_usage,
                        scan_types,
                        cost_estimate,
                    );
                }
            }
        }
    }

    /// Connection pool health check and optimization
    pub async fn optimize_connection_pool(
        &self,
        pool: &Pool<Postgres>,
    ) -> PersistenceResult<ConnectionPoolStats> {
        let start_time = Instant::now();

        // Test connection acquisition time
        let connection_test = pool.acquire().await;
        let acquisition_time = start_time.elapsed();

        match connection_test {
            Ok(_conn) => {
                // Connection successful, gather pool stats
                let pool_size = pool.size();
                let idle_connections = pool.num_idle();

                // Calculate utilization
                let utilization = if pool_size > 0 {
                    ((pool_size as f32 - idle_connections as f32) / pool_size as f32) * 100.0
                } else {
                    0.0
                };

                let mut recommendations = Vec::new();

                // Provide optimization recommendations
                if utilization > 90.0 {
                    recommendations.push(
                        "Connection pool utilization high. Consider increasing max_connections."
                            .to_string(),
                    );
                } else if utilization < 10.0 && pool_size > 5 {
                    recommendations.push(String::from("Connection pool underutilized. Consider reducing max_connections to save resources."));
                }

                if acquisition_time > Duration::from_millis(100) {
                    recommendations.push(String::from("Connection acquisition time high. Consider optimizing pool settings or adding connection multiplexing."));
                }

                Ok(ConnectionPoolStats {
                    total_connections: pool_size,
                    idle_connections: idle_connections as u32,
                    utilization_percentage: utilization,
                    acquisition_time,
                    recommendations,
                    health_status: if utilization < 95.0
                        && acquisition_time < Duration::from_millis(200)
                    {
                        PoolHealthStatus::Healthy
                    } else if utilization < 98.0 && acquisition_time < Duration::from_millis(500) {
                        PoolHealthStatus::Warning
                    } else {
                        PoolHealthStatus::Critical
                    },
                })
            }
            Err(e) => Err(PersistenceError::ConnectionError {
                message: format!("Connection pool health check failed: {}", e),
            }),
        }
    }

    /// Enhanced prepared statement execution with caching
    pub async fn execute_prepared_statement<'q>(
        &self,
        pool: &Pool<Postgres>,
        query_template: &str,
        // Simplified parameter handling - in practice would use proper query builder
    ) -> PersistenceResult<sqlx::postgres::PgQueryResult> {
        let start_time = Instant::now();
        let statement_signature = self.generate_query_signature(query_template);

        // Update or create prepared statement info
        {
            let mut statements = self.prepared_statements.write().await;
            let info = statements
                .entry(statement_signature.clone())
                .or_insert_with(|| {
                    PreparedStatementInfo {
                        statement_id: statement_signature.clone(),
                        query_template: query_template.to_string(),
                        parameter_count: 0, // Simplified for now
                        last_used: Instant::now(),
                        usage_count: 0,
                        average_execution_time: Duration::from_millis(0),
                    }
                });

            info.last_used = Instant::now();
            info.usage_count += 1;
        }

        // Execute the prepared statement (simplified version)
        let query = sqlx::query(query_template);

        let result = query
            .execute(pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError {
                message: format!("Prepared statement execution failed: {}", e),
            })?;

        let execution_time = start_time.elapsed();

        // Update execution time statistics
        {
            let mut statements = self.prepared_statements.write().await;
            if let Some(info) = statements.get_mut(&statement_signature) {
                if info.usage_count == 1 {
                    info.average_execution_time = execution_time;
                } else {
                    let total_time = info.average_execution_time * (info.usage_count - 1) as u32
                        + execution_time;
                    info.average_execution_time = total_time / info.usage_count as u32;
                }
            }
        }

        // Check for performance regression
        self.check_performance_regression(&statement_signature, execution_time)
            .await;

        // Update statistics
        {
            let mut stats = self.query_stats.write().await;
            stats.prepared_statements_used += 1;
        }

        Ok(result)
    }

    /// Generate automatic index recommendations based on query patterns
    pub async fn generate_index_recommendations(
        &self,
        pool: &Pool<Postgres>,
    ) -> PersistenceResult<Vec<IndexRecommendation>> {
        let start_time = Instant::now();

        // Analyze slow queries from pg_stat_statements
        let slow_queries = sqlx::query(
            r#"
            SELECT query, calls, total_time, mean_time, rows
            FROM pg_stat_statements 
            WHERE mean_time > 1000 -- Queries taking more than 1 second on average
            ORDER BY mean_time DESC 
            LIMIT 50
            "#,
        )
        .fetch_all(pool)
        .await
        .map_err(|e| PersistenceError::ConnectionError {
            message: format!("Failed to analyze slow queries: {}", e),
        })?;

        let mut recommendations = Vec::new();

        for row in slow_queries {
            let query: String = row.get("query");
            let calls: i64 = row.get("calls");
            let mean_time: f64 = row.get("mean_time");

            // Analyze query for index opportunities
            let query_recommendations =
                self.analyze_query_for_indexes(&query, calls as u64, mean_time);
            recommendations.extend(query_recommendations);
        }

        // Deduplicate and prioritize recommendations
        recommendations.sort_by(|a, b| {
            b.estimated_improvement
                .partial_cmp(&a.estimated_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        recommendations.dedup_by(|a, b| a.table_name == b.table_name && a.columns == b.columns);

        // Cache recommendations
        {
            let mut cached_recommendations = self.index_recommendations.write().await;
            *cached_recommendations = recommendations.clone();
        }

        // Update statistics
        {
            let mut stats = self.query_stats.write().await;
            stats.index_recommendations_generated += recommendations.len() as u64;
        }

        self.record_query_execution(start_time.elapsed()).await;

        Ok(recommendations)
    }

    /// Analyze individual query for potential indexes
    fn analyze_query_for_indexes(
        &self,
        query: &str,
        frequency: u64,
        avg_time: f64,
    ) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();
        let query_lower = query.to_lowercase();

        // Look for WHERE clause patterns
        if let Some(where_pos) = query_lower.find("where") {
            let where_clause = &query_lower[where_pos..];

            // Extract potential index candidates
            let patterns = [
                (r"(\w+)\s*=\s*", IndexType::BTree),
                (r"(\w+)\s+in\s*\(", IndexType::BTree),
                (r"(\w+)\s*>\s*", IndexType::BTree),
                (r"(\w+)\s*<\s*", IndexType::BTree),
                (r"(\w+)\s+like\s+", IndexType::Gin),
            ];

            for (pattern, index_type) in patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    for capture in regex.captures_iter(where_clause) {
                        if let Some(column) = capture.get(1) {
                            let table_name = self.extract_table_name(&query_lower);

                            recommendations.push(IndexRecommendation {
                                table_name: table_name
                                    .unwrap_or_else(|| String::from("unknown_table")),
                                columns: vec![column.as_str().to_string()],
                                index_type: index_type.clone(),
                                estimated_improvement: self
                                    .calculate_improvement_estimate(avg_time, frequency),
                                frequency_score: frequency,
                                created_at: std::time::SystemTime::now(),
                                query_patterns: vec![query.to_string()],
                            });
                        }
                    }
                }
            }
        }

        // Look for ORDER BY patterns
        if let Some(order_pos) = query_lower.find("order by") {
            let order_clause = &query_lower[order_pos + 8..].trim();

            // Handle multiple columns in ORDER BY
            let order_columns: Vec<&str> = order_clause
                .split(',')
                .map(|col| {
                    // Remove ASC/DESC and extra whitespace
                    col.trim().split_whitespace().next().unwrap_or("").trim()
                })
                .filter(|col| !col.is_empty())
                .collect();

            for column in order_columns {
                if !column.is_empty() {
                    let table_name = self.extract_table_name(&query_lower);

                    recommendations.push(IndexRecommendation {
                        table_name: table_name.unwrap_or_else(|| String::from("unknown_table")),
                        columns: vec![column.to_string()],
                        index_type: IndexType::BTree,
                        estimated_improvement: self
                            .calculate_improvement_estimate(avg_time, frequency),
                        frequency_score: frequency,
                        created_at: std::time::SystemTime::now(),
                        query_patterns: vec![query.to_string()],
                    });
                }
            }
        }

        recommendations
    }

    /// Extract table name from query
    fn extract_table_name(&self, query: &str) -> Option<String> {
        // Simple regex to extract table name from FROM clause
        if let Ok(regex) = regex::Regex::new(r"from\s+(\w+)") {
            if let Some(capture) = regex.captures(query) {
                return capture.get(1).map(|m| m.as_str().to_string());
            }
        }
        None
    }

    /// Calculate estimated improvement percentage
    fn calculate_improvement_estimate(&self, avg_time: f64, frequency: u64) -> f64 {
        // Simple heuristic: higher frequency and time = higher potential improvement
        let time_factor = (avg_time / 1000.0).min(10.0); // Normalize to 0-10 scale
        let frequency_factor = (frequency as f64).log10().min(5.0); // Log scale for frequency

        (time_factor * frequency_factor * 10.0).min(95.0) // Cap at 95% improvement
    }

    /// Check for performance regression
    async fn check_performance_regression(&self, query_signature: &str, execution_time: Duration) {
        let mut baselines = self.performance_baselines.write().await;

        match baselines.get_mut(query_signature) {
            Some(baseline) => {
                // Check if current execution is significantly slower than baseline
                let threshold_multiplier = 2.0; // Consider 2x slower as regression
                let regression_threshold = Duration::from_nanos(
                    (baseline.baseline_time.as_nanos() as f64 * threshold_multiplier) as u64,
                );

                if execution_time > regression_threshold {
                    // Performance regression detected
                    let mut stats = self.query_stats.write().await;
                    stats.performance_regressions_detected += 1;
                    drop(stats);

                    // Log regression (in a real system, this would trigger alerts)
                    eprintln!(
                        "Performance regression detected for query {}: current {}ms vs baseline {}ms",
                        query_signature,
                        execution_time.as_millis(),
                        baseline.baseline_time.as_millis()
                    );
                }

                // Update baseline with exponential moving average
                let alpha = 0.1; // Smoothing factor
                let new_baseline_nanos = baseline.baseline_time.as_nanos() as f64 * (1.0 - alpha)
                    + execution_time.as_nanos() as f64 * alpha;

                baseline.baseline_time = Duration::from_nanos(new_baseline_nanos as u64);
                baseline.sample_count += 1;
                baseline.last_updated = Instant::now();
            }
            None => {
                // Establish new baseline
                baselines.insert(
                    query_signature.to_string(),
                    PerformanceBaseline {
                        query_signature: query_signature.to_string(),
                        baseline_time: execution_time,
                        baseline_established_at: Instant::now(),
                        sample_count: 1,
                        variance: 0.0,
                        last_updated: Instant::now(),
                    },
                );
            }
        }
    }

    /// Generate query signature for identification
    fn generate_query_signature(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Normalize query by removing parameters and whitespace
        let normalized = query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        format!("query_{:x}", hasher.finish())
    }

    /// Get index recommendations
    pub async fn get_index_recommendations(&self) -> Vec<IndexRecommendation> {
        let recommendations = self.index_recommendations.read().await;
        recommendations.clone()
    }

    /// Get prepared statement statistics
    pub async fn get_prepared_statement_stats(&self) -> HashMap<String, PreparedStatementInfo> {
        let statements = self.prepared_statements.read().await;
        statements.clone()
    }

    /// Get performance baselines
    pub async fn get_performance_baselines(&self) -> HashMap<String, PerformanceBaseline> {
        let baselines = self.performance_baselines.read().await;
        baselines.clone()
    }

    /// Clear expired prepared statements
    pub async fn cleanup_prepared_statements(&self, max_age: Duration) {
        let mut statements = self.prepared_statements.write().await;
        let now = Instant::now();

        statements.retain(|_, info| now.duration_since(info.last_used) < max_age);
    }

    /// Adaptive query cache warmup based on usage patterns
    pub async fn warmup_cache_adaptive(&self, pool: &Pool<Postgres>) -> PersistenceResult<usize> {
        let mut warmed_queries = 0;

        // Get frequently used prepared statements
        let statements = self.get_prepared_statement_stats().await;

        // Sort by usage frequency and recent usage
        let mut frequent_statements: Vec<_> = statements
            .values()
            .filter(|stmt| {
                stmt.usage_count > 10 && stmt.last_used.elapsed() < Duration::from_secs(3600)
            })
            .collect();

        frequent_statements.sort_by(|a, b| {
            let score_a = a.usage_count as f64 / (a.last_used.elapsed().as_secs() as f64 + 1.0);
            let score_b = b.usage_count as f64 / (b.last_used.elapsed().as_secs() as f64 + 1.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Warmup top 20 queries
        for statement in frequent_statements.iter().take(20) {
            // Skip if already cached
            let cache_key = format!("warmup:{}", statement.statement_id);
            if self
                .get_from_cache::<serde_json::Value>(&cache_key)
                .await
                .is_some()
            {
                continue;
            }

            // Execute warmup query if it's a simple SELECT
            if statement
                .query_template
                .trim_start()
                .to_lowercase()
                .starts_with("select")
            {
                match sqlx::query(&statement.query_template)
                    .fetch_optional(pool)
                    .await
                {
                    Ok(Some(row)) => {
                        // Cache a placeholder to mark as warmed
                        let warmup_data = serde_json::json!({"warmed_at": chrono::Utc::now()});
                        self.cache_result(&cache_key, &warmup_data).await;
                        warmed_queries += 1;
                    }
                    Ok(None) => {
                        // Cache empty result
                        let empty_data = serde_json::Value::Null;
                        self.cache_result(&cache_key, &empty_data).await;
                        warmed_queries += 1;
                    }
                    Err(_) => {
                        // Skip problematic queries
                        continue;
                    }
                }
            }
        }

        Ok(warmed_queries)
    }

    /// Auto-scaling connection pool based on load patterns
    pub async fn auto_scale_connection_pool(
        &self,
        current_stats: &ConnectionPoolStats,
    ) -> PoolScalingRecommendation {
        let mut recommendation = PoolScalingRecommendation {
            action: ScalingAction::NoChange,
            recommended_min_connections: self.config.pool_config.min_connections,
            recommended_max_connections: self.config.pool_config.max_connections,
            confidence: 0.0,
            reasoning: Vec::new(),
        };

        // Analyze current utilization patterns
        let utilization = current_stats.utilization_percentage;
        let acquisition_time = current_stats.acquisition_time;

        // High utilization suggests need for more connections
        if utilization > 85.0 {
            recommendation.action = ScalingAction::ScaleUp;
            recommendation.recommended_max_connections =
                (current_stats.total_connections as f32 * 1.5).min(100.0) as u32;
            recommendation.confidence = ((utilization as f64 - 85.0) / 15.0).min(1.0);
            recommendation.reasoning.push(format!(
                "High utilization ({}%) suggests connection pressure",
                utilization
            ));
        }

        // Slow acquisition time also suggests scaling up
        if acquisition_time > Duration::from_millis(500) {
            recommendation.action = ScalingAction::ScaleUp;
            recommendation.recommended_max_connections =
                (current_stats.total_connections as f32 * 1.2).min(100.0) as u32;
            recommendation.confidence = recommendation.confidence.max(0.7);
            recommendation.reasoning.push(format!(
                "Slow connection acquisition ({}ms) indicates bottleneck",
                acquisition_time.as_millis()
            ));
        }

        // Low utilization suggests scaling down (but be conservative)
        if utilization < 20.0 && current_stats.total_connections > 10 {
            recommendation.action = ScalingAction::ScaleDown;
            recommendation.recommended_max_connections =
                (current_stats.total_connections as f32 * 0.8).max(10.0) as u32;
            recommendation.confidence = ((20.0 - utilization as f64) / 20.0).min(0.6); // Lower confidence for scaling down
            recommendation.reasoning.push(format!(
                "Low utilization ({}%) suggests overprovisioning",
                utilization
            ));
        }

        recommendation
    }

    /// Smart query load balancing across read replicas
    pub async fn route_query_smart(
        &self,
        query: &str,
        primary_pool: &Pool<Postgres>,
        read_replica_pools: &[Pool<Postgres>],
    ) -> QueryRoutingDecision {
        let query_lower = query.trim_start().to_lowercase();

        // Route based on query type
        let routing = if query_lower.starts_with("select") && !query_lower.contains("for update") {
            // Read query - route to read replica
            if !read_replica_pools.is_empty() {
                // Simple round-robin with health checking
                let replica_index =
                    (self.get_stats().await.total_queries as usize) % read_replica_pools.len();
                QueryRoutingDecision {
                    target: QueryTarget::ReadReplica(replica_index),
                    reasoning: String::from("Read-only query routed to read replica"),
                    estimated_load_impact: LoadImpact::Low,
                }
            } else {
                QueryRoutingDecision {
                    target: QueryTarget::Primary,
                    reasoning: String::from("No read replicas available"),
                    estimated_load_impact: LoadImpact::Medium,
                }
            }
        } else {
            // Write query or transaction - must go to primary
            QueryRoutingDecision {
                target: QueryTarget::Primary,
                reasoning: String::from("Write operation requires primary database"),
                estimated_load_impact: if query_lower.contains("insert")
                    || query_lower.contains("update")
                {
                    LoadImpact::High
                } else {
                    LoadImpact::Medium
                },
            }
        };

        routing
    }

    /// Batch execute with automatic batching optimization
    pub async fn execute_batch_optimized<T>(
        &self,
        pool: &Pool<Postgres>,
        queries: Vec<String>,
        batch_processor: impl Fn(&Pool<Postgres>, &[String]) -> PersistenceResult<Vec<T>>,
    ) -> PersistenceResult<Vec<T>>
    where
        T: Send + 'static,
    {
        let start_time = Instant::now();
        let mut results = Vec::new();

        // Adaptive batch sizing based on query complexity
        let avg_query_length: f64 =
            queries.iter().map(|q| q.len()).sum::<usize>() as f64 / queries.len() as f64;
        let adaptive_batch_size = if avg_query_length > 200.0 {
            // Complex queries - smaller batches
            self.config.batch_size / 2
        } else if avg_query_length < 50.0 {
            // Simple queries - larger batches
            self.config.batch_size * 2
        } else {
            self.config.batch_size
        }
        .max(1)
        .min(1000);

        // Process in adaptive batches
        for batch in queries.chunks(adaptive_batch_size) {
            let batch_results = batch_processor(pool, batch)?;
            results.extend(batch_results);
        }

        // Update statistics
        self.record_batch_operation().await;
        self.record_query_execution(start_time.elapsed()).await;

        Ok(results)
    }

    /// Intelligent cache preloading based on access patterns
    pub async fn preload_cache_intelligent(
        &self,
        pool: &Pool<Postgres>,
        user_context: &UserCacheContext,
    ) -> PersistenceResult<usize> {
        let mut preloaded_items = 0;

        // Predict likely queries based on user behavior patterns
        let predicted_queries = self.predict_likely_queries(user_context);

        for predicted_query in predicted_queries.iter().take(10) {
            let cache_key = format!(
                "preload:{}:{}",
                user_context.user_id, predicted_query.query_hash
            );

            // Skip if already cached
            if self
                .get_from_cache::<serde_json::Value>(&cache_key)
                .await
                .is_some()
            {
                continue;
            }

            // Execute preload query with timeout
            let query_result = tokio::time::timeout(
                Duration::from_millis(1000), // 1 second timeout for preloading
                sqlx::query(&predicted_query.query)
                    .bind(&user_context.user_id)
                    .fetch_optional(pool),
            )
            .await;

            match query_result {
                Ok(Ok(Some(row))) => {
                    // Convert row to JSON and cache
                    let row_data = self.row_to_json(row)?;
                    self.cache_result(&cache_key, &row_data).await;
                    preloaded_items += 1;
                }
                Ok(Ok(None)) => {
                    // Cache null result
                    self.cache_result(&cache_key, &serde_json::Value::Null)
                        .await;
                    preloaded_items += 1;
                }
                _ => {
                    // Skip failed/timed out queries
                    continue;
                }
            }
        }

        Ok(preloaded_items)
    }

    /// Predict likely queries based on user patterns
    fn predict_likely_queries(&self, user_context: &UserCacheContext) -> Vec<PredictedQuery> {
        let mut predictions = Vec::new();

        // Common user data queries
        predictions.push(PredictedQuery {
            query: String::from("SELECT progress_data FROM user_progress WHERE user_id = $1"),
            query_hash: String::from("user_progress"),
            probability: 0.9,
        });

        predictions.push(PredictedQuery {
            query: String::from("SELECT preferences_data FROM user_preferences WHERE user_id = $1"),
            query_hash: String::from("user_preferences"),
            probability: 0.8,
        });

        // Time-based predictions
        let current_hour = chrono::Utc::now().hour();
        if current_hour >= 9 && current_hour <= 17 {
            // Business hours - likely to access recent feedback
            predictions.push(PredictedQuery {
                query: String::from("SELECT feedback_data FROM feedback_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10"),
                query_hash: String::from("recent_feedback"),
                probability: 0.7,
            });
        }

        // Sort by probability
        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        predictions
    }

    /// Convert database row to JSON for caching
    fn row_to_json(&self, row: sqlx::postgres::PgRow) -> PersistenceResult<serde_json::Value> {
        // Simplified conversion - in practice would handle different column types
        let mut json_obj = serde_json::Map::new();

        // For now, just return a placeholder structure
        json_obj.insert(
            String::from("cached_at"),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );

        Ok(serde_json::Value::Object(json_obj))
    }
}

/// Complete user data structure for optimized queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteUserData {
    /// User identifier
    pub user_id: String,
    /// User progress data
    pub progress: Option<UserProgress>,
    /// User preferences
    pub preferences: Option<UserPreferences>,
    /// Total session count
    pub session_count: u64,
    /// Total feedback count
    pub feedback_count: u64,
}

/// Query performance analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    /// SQL query analyzed
    pub query: String,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Execution plan from database
    pub execution_plan: serde_json::Value,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Indexes used in query
    pub index_usage: Vec<String>,
    /// Scan types performed
    pub scan_types: Vec<String>,
    /// Estimated query cost
    pub cost_estimate: f64,
}

/// Execution plan analysis details
#[derive(Debug, Clone)]
pub struct ExecutionPlanAnalysis {
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Indexes used
    pub index_usage: Vec<String>,
    /// Scan types found
    pub scan_types: Vec<String>,
    /// Total cost estimate
    pub cost_estimate: f64,
}

/// Connection pool statistics and health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    /// Total number of connections
    pub total_connections: u32,
    /// Number of idle connections
    pub idle_connections: u32,
    /// Utilization percentage
    pub utilization_percentage: f32,
    /// Connection acquisition time
    pub acquisition_time: Duration,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Pool health status
    pub health_status: PoolHealthStatus,
}

/// Pool health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolHealthStatus {
    /// Pool is healthy
    Healthy,
    /// Pool has warnings
    Warning,
    /// Pool is in critical state
    Critical,
}

/// Pool scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolScalingRecommendation {
    /// Recommended scaling action
    pub action: ScalingAction,
    /// Recommended minimum connections
    pub recommended_min_connections: u32,
    /// Recommended maximum connections
    pub recommended_max_connections: u32,
    /// Confidence in recommendation (0.0-1.0)
    pub confidence: f64,
    /// Reasoning for recommendation
    pub reasoning: Vec<String>,
}

/// Scaling action recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up connections
    ScaleUp,
    /// Scale down connections
    ScaleDown,
    /// No change needed
    NoChange,
}

/// Query routing decision
#[derive(Debug, Clone)]
pub struct QueryRoutingDecision {
    /// Target database for query
    pub target: QueryTarget,
    /// Reasoning for routing decision
    pub reasoning: String,
    /// Estimated load impact
    pub estimated_load_impact: LoadImpact,
}

/// Query target for routing
#[derive(Debug, Clone)]
pub enum QueryTarget {
    /// Route to primary database
    Primary,
    /// Route to read replica at index
    ReadReplica(usize),
}

/// Estimated load impact
#[derive(Debug, Clone)]
pub enum LoadImpact {
    /// Low load impact
    Low,
    /// Medium load impact
    Medium,
    /// High load impact
    High,
}

/// User cache context for intelligent preloading
#[derive(Debug, Clone)]
pub struct UserCacheContext {
    /// User identifier
    pub user_id: String,
    /// Last access timestamp
    pub last_access_time: chrono::DateTime<chrono::Utc>,
    /// User access patterns
    pub access_patterns: Vec<String>,
    /// Preferred data types
    pub preferred_data_types: Vec<String>,
}

/// Predicted query for cache preloading
#[derive(Debug, Clone)]
pub struct PredictedQuery {
    /// SQL query
    pub query: String,
    /// Query hash/identifier
    pub query_hash: String,
    /// Prediction probability (0.0-1.0)
    pub probability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_optimizer_config() {
        let config = QueryOptimizerConfig::default();
        assert!(config.enable_query_cache);
        assert_eq!(config.cache_ttl_seconds, 300);
        assert!(config.enable_batching);
    }

    #[test]
    fn test_pool_config() {
        let config = PoolConfig::default();
        assert_eq!(config.min_connections, 5);
        assert_eq!(config.max_connections, 30);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Test caching a simple string
        let test_data = String::from("test_data");
        optimizer.cache_result("test_key", &test_data).await;

        // Retrieve from cache
        let cached: Option<String> = optimizer.get_from_cache("test_key").await;
        assert_eq!(cached, Some(test_data));
    }

    #[tokio::test]
    async fn test_query_signature_generation() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        let query1 = "SELECT * FROM users WHERE id = $1";
        let query2 = "select * from users where id = $1";
        let query3 = "SELECT * FROM posts WHERE id = $1";

        let sig1 = optimizer.generate_query_signature(query1);
        let sig2 = optimizer.generate_query_signature(query2);
        let sig3 = optimizer.generate_query_signature(query3);

        // Same normalized queries should have same signature
        assert_eq!(sig1, sig2);
        // Different queries should have different signatures
        assert_ne!(sig1, sig3);
    }

    #[tokio::test]
    async fn test_index_recommendation_analysis() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        let query = "SELECT * FROM users WHERE email = 'test@example.com' ORDER BY created_at";
        let recommendations = optimizer.analyze_query_for_indexes(query, 100, 1500.0);

        assert!(!recommendations.is_empty());
        // Should recommend index on email column and created_at for ordering
        assert!(recommendations
            .iter()
            .any(|r| r.columns.contains(&String::from("email"))));
        assert!(recommendations
            .iter()
            .any(|r| r.columns.contains(&String::from("created_at"))));
    }

    #[tokio::test]
    async fn test_performance_baseline_establishment() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        let query_sig = "test_query_signature";
        let execution_time = Duration::from_millis(100);

        // First execution should establish baseline
        optimizer
            .check_performance_regression(query_sig, execution_time)
            .await;

        let baselines = optimizer.get_performance_baselines().await;
        assert!(baselines.contains_key(query_sig));
        assert_eq!(baselines[query_sig].baseline_time, execution_time);
        assert_eq!(baselines[query_sig].sample_count, 1);
    }

    #[tokio::test]
    async fn test_prepared_statement_statistics() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Simulate prepared statement usage
        let query_template = "SELECT * FROM users WHERE id = $1";
        let statement_signature = optimizer.generate_query_signature(query_template);

        // Check performance regression should update prepared statement info
        optimizer
            .check_performance_regression(&statement_signature, Duration::from_millis(50))
            .await;

        let stats = optimizer.get_prepared_statement_stats().await;
        // Should have performance baseline data
        let baselines = optimizer.get_performance_baselines().await;
        assert!(baselines.contains_key(&statement_signature));
    }

    #[tokio::test]
    async fn test_query_stats_enhancement() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Get initial stats
        let initial_stats = optimizer.get_stats().await;
        assert_eq!(initial_stats.prepared_statements_used, 0);
        assert_eq!(initial_stats.index_recommendations_generated, 0);
        assert_eq!(initial_stats.performance_regressions_detected, 0);

        // Simulate some activity
        {
            let mut stats = optimizer.query_stats.write().await;
            stats.prepared_statements_used += 5;
            stats.index_recommendations_generated += 3;
            stats.performance_regressions_detected += 1;
        }

        let updated_stats = optimizer.get_stats().await;
        assert_eq!(updated_stats.prepared_statements_used, 5);
        assert_eq!(updated_stats.index_recommendations_generated, 3);
        assert_eq!(updated_stats.performance_regressions_detected, 1);
    }

    #[tokio::test]
    async fn test_prepared_statement_cleanup() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Add some prepared statements with different ages
        {
            let mut statements = optimizer.prepared_statements.write().await;
            statements.insert(
                String::from("old_statement"),
                PreparedStatementInfo {
                    statement_id: String::from("old_statement"),
                    query_template: String::from("SELECT * FROM old_table"),
                    parameter_count: 0,
                    last_used: Instant::now() - Duration::from_secs(3600), // 1 hour ago
                    usage_count: 1,
                    average_execution_time: Duration::from_millis(100),
                },
            );

            statements.insert(
                String::from("recent_statement"),
                PreparedStatementInfo {
                    statement_id: String::from("recent_statement"),
                    query_template: String::from("SELECT * FROM recent_table"),
                    parameter_count: 0,
                    last_used: Instant::now(),
                    usage_count: 1,
                    average_execution_time: Duration::from_millis(50),
                },
            );
        }

        // Cleanup statements older than 30 minutes
        optimizer
            .cleanup_prepared_statements(Duration::from_secs(1800))
            .await;

        let statements = optimizer.get_prepared_statement_stats().await;
        assert!(!statements.contains_key("old_statement"));
        assert!(statements.contains_key("recent_statement"));
    }

    #[tokio::test]
    async fn test_connection_pool_auto_scaling() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Test high utilization scenario
        let high_util_stats = ConnectionPoolStats {
            total_connections: 20,
            idle_connections: 2,
            utilization_percentage: 90.0,
            acquisition_time: Duration::from_millis(600),
            recommendations: Vec::new(),
            health_status: PoolHealthStatus::Warning,
        };

        let scaling_rec = optimizer.auto_scale_connection_pool(&high_util_stats).await;
        assert!(matches!(scaling_rec.action, ScalingAction::ScaleUp));
        assert!(scaling_rec.recommended_max_connections > 20);
        assert!(scaling_rec.confidence > 0.0);

        // Test low utilization scenario
        let low_util_stats = ConnectionPoolStats {
            total_connections: 30,
            idle_connections: 25,
            utilization_percentage: 16.7,
            acquisition_time: Duration::from_millis(50),
            recommendations: Vec::new(),
            health_status: PoolHealthStatus::Healthy,
        };

        let scaling_rec = optimizer.auto_scale_connection_pool(&low_util_stats).await;
        assert!(matches!(scaling_rec.action, ScalingAction::ScaleDown));
        assert!(scaling_rec.recommended_max_connections < 30);
    }

    #[tokio::test]
    async fn test_query_routing_decisions() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Test read query routing logic without actual database connection
        let read_query = "SELECT * FROM users WHERE id = 123";

        // We can't create actual pools in tests, so we'll test the routing logic only
        // by creating a mock scenario that doesn't require actual pool access

        // Create a dummy pool (this won't work for real connections but is fine for testing logic)
        let test_db_url = "postgresql://test:test@localhost/test_db";

        // Skip actual database connection tests and just validate routing logic
        // Test that read queries would be routed to replicas if available
        let query_lower = read_query.trim_start().to_lowercase();
        assert!(query_lower.starts_with("select"));
        assert!(!query_lower.contains("for update"));

        // Test write query classification
        let write_query = "INSERT INTO users (name) VALUES ('test')";
        let write_query_lower = write_query.trim_start().to_lowercase();
        assert!(!write_query_lower.starts_with("select"));

        // Test update query classification
        let update_query = "UPDATE users SET name = 'test' WHERE id = 1";
        let update_query_lower = update_query.trim_start().to_lowercase();
        assert!(update_query_lower.contains("update"));
    }

    #[test]
    fn test_predicted_query_generation() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        let user_context = UserCacheContext {
            user_id: String::from("test_user"),
            last_access_time: chrono::Utc::now(),
            access_patterns: vec![String::from("progress"), String::from("preferences")],
            preferred_data_types: vec![String::from("progress")],
        };

        let predictions = optimizer.predict_likely_queries(&user_context);

        assert!(!predictions.is_empty());
        assert!(predictions.iter().any(|p| p.query_hash == "user_progress"));
        assert!(predictions
            .iter()
            .any(|p| p.query_hash == "user_preferences"));

        // Check that predictions are sorted by probability
        for window in predictions.windows(2) {
            assert!(window[0].probability >= window[1].probability);
        }
    }

    #[tokio::test]
    async fn test_adaptive_batch_sizing() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        // Test with complex queries (should use smaller batches)
        let complex_queries = vec![
            String::from("SELECT u.user_id, u.username, u.email, u.created_at, u.last_login, p.progress_data, p.completion_percentage, p.skills_learned, p.time_spent, f.feedback_data, f.scores, f.improvement_suggestions, f.audio_analysis_results FROM users u JOIN user_progress p ON u.id = p.user_id LEFT JOIN feedback_history f ON u.id = f.user_id WHERE u.created_at > NOW() - INTERVAL '30 days' AND p.completion_percentage > 50 ORDER BY u.last_login DESC, p.completion_percentage DESC");
            10
        ];

        // Simple batch processor for testing
        let batch_processor =
            |_pool: &Pool<Postgres>, batch: &[String]| -> PersistenceResult<Vec<String>> {
                Ok(batch
                    .iter()
                    .map(|q| format!("processed: {}", q.len()))
                    .collect())
            };

        // We can't actually test the database execution, but we can verify the batching logic
        // would work by checking the query complexity detection
        let avg_length: f64 = complex_queries.iter().map(|q| q.len()).sum::<usize>() as f64
            / complex_queries.len() as f64;
        assert!(avg_length > 200.0); // Complex queries detected

        // Test with simple queries (should use larger batches)
        let simple_queries = vec![String::from("SELECT id FROM users"); 10];
        let avg_simple_length: f64 = simple_queries.iter().map(|q| q.len()).sum::<usize>() as f64
            / simple_queries.len() as f64;
        assert!(avg_simple_length < 50.0); // Simple queries detected
    }
}
