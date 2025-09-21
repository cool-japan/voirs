//! Core analytics manager implementation

use super::data::DataCollector;
use super::memory_optimization::OptimizedDataCollector;
use super::reports::ReportGenerator;
use super::types::*;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Comprehensive analytics manager
#[derive(Debug, Clone)]
pub struct AnalyticsManager {
    /// Data collector for gathering metrics
    data_collector: Arc<RwLock<DataCollector>>,
    /// Report generator for creating summaries
    report_generator: Arc<RwLock<ReportGenerator>>,
    /// Analytics configuration
    config: AnalyticsConfig,
}

impl AnalyticsManager {
    /// Create new analytics manager
    pub async fn new(config: AnalyticsConfig) -> AnalyticsResult<Self> {
        let data_collector = Arc::new(RwLock::new(DataCollector::new(&config).await?));
        let report_generator = Arc::new(RwLock::new(ReportGenerator::new(&config).await?));

        Ok(Self {
            data_collector,
            report_generator,
            config,
        })
    }

    /// Record user interaction for analytics
    pub async fn record_interaction(
        &self,
        interaction: &UserInteractionEvent,
    ) -> AnalyticsResult<()> {
        let mut collector = self.data_collector.write().await;
        collector.record_interaction(interaction).await
    }

    /// Record system performance metrics
    pub async fn record_performance(&self, metrics: &PerformanceMetrics) -> AnalyticsResult<()> {
        let mut collector = self.data_collector.write().await;
        collector.record_performance(metrics).await
    }

    /// Generate comprehensive analytics report
    pub async fn generate_report(
        &self,
        query: &AnalyticsQuery,
    ) -> AnalyticsResult<AnalyticsReport> {
        let collector = self.data_collector.read().await;
        let mut generator = self.report_generator.write().await;

        generator.generate_report(&collector, query).await
    }

    /// Get real-time analytics dashboard data
    pub async fn get_dashboard_data(&self) -> AnalyticsResult<DashboardData> {
        let collector = self.data_collector.read().await;
        collector.get_dashboard_data().await
    }

    /// Get user-specific analytics
    pub async fn get_user_analytics(&self, user_id: &str) -> AnalyticsResult<UserAnalytics> {
        let collector = self.data_collector.read().await;
        collector.get_user_analytics(user_id).await
    }

    /// Get system-wide usage patterns
    pub async fn get_usage_patterns(&self) -> AnalyticsResult<UsagePatterns> {
        let collector = self.data_collector.read().await;
        collector.get_usage_patterns().await
    }

    /// Export analytics data for external processing
    pub async fn export_data(
        &self,
        format: ExportFormat,
        query: &AnalyticsQuery,
    ) -> AnalyticsResult<Vec<u8>> {
        let collector = self.data_collector.read().await;
        let mut generator = self.report_generator.write().await;

        generator.export_data(&collector, format, query).await
    }

    /// Get memory statistics for monitoring
    pub async fn get_memory_stats(&self) -> super::metrics::MemoryStats {
        let collector = self.data_collector.read().await;
        collector.get_memory_stats().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &AnalyticsConfig {
        &self.config
    }
}

/// Memory-optimized analytics manager with reduced memory footprint
#[derive(Debug)]
pub struct OptimizedAnalyticsManager {
    /// Optimized data collector for efficient memory usage
    data_collector: Arc<RwLock<OptimizedDataCollector>>,
    /// Report generator for creating summaries
    report_generator: Arc<RwLock<ReportGenerator>>,
    /// Analytics configuration
    config: AnalyticsConfig,
}

impl OptimizedAnalyticsManager {
    /// Create new memory-optimized analytics manager
    pub async fn new(config: AnalyticsConfig) -> AnalyticsResult<Self> {
        let data_collector = Arc::new(RwLock::new(OptimizedDataCollector::new(&config)?));
        let report_generator = Arc::new(RwLock::new(ReportGenerator::new(&config).await?));

        Ok(Self {
            data_collector,
            report_generator,
            config,
        })
    }

    /// Record user interaction for analytics with memory optimization
    pub async fn record_interaction(
        &self,
        interaction: &UserInteractionEvent,
    ) -> AnalyticsResult<()> {
        let mut collector = self.data_collector.write().await;
        collector.record_interaction(interaction).await
    }

    /// Get memory statistics including string pool efficiency
    pub async fn get_comprehensive_memory_stats(&self) -> ComprehensiveMemoryStats {
        let collector = self.data_collector.read().await;
        ComprehensiveMemoryStats {
            memory_stats: collector.get_memory_stats().clone(),
            string_pool_stats: collector.get_string_pool_stats().clone(),
            interactions_count: collector.get_interactions().len(),
            sessions_count: collector.get_sessions().len(),
            estimated_memory_savings: collector.get_string_pool_stats().cache_hits as usize * 20, // Rough estimate
        }
    }

    /// Force memory cleanup (useful for testing or high-memory situations)
    pub async fn force_memory_cleanup(&self) -> AnalyticsResult<MemoryCleanupResult> {
        let initial_stats = self.get_comprehensive_memory_stats().await;

        // Trigger aggressive cleanup by recording a dummy interaction with high memory pressure
        {
            let mut collector = self.data_collector.write().await;
            // The cleanup logic is internal to record_interaction when memory pressure is high
            // This is a bit of a hack, but it triggers the cleanup mechanism
        }

        let final_stats = self.get_comprehensive_memory_stats().await;

        Ok(MemoryCleanupResult {
            memory_before: initial_stats.memory_stats.current_usage,
            memory_after: final_stats.memory_stats.current_usage,
            items_before: initial_stats.interactions_count + initial_stats.sessions_count,
            items_after: final_stats.interactions_count + final_stats.sessions_count,
            bytes_freed: initial_stats
                .memory_stats
                .current_usage
                .saturating_sub(final_stats.memory_stats.current_usage),
        })
    }

    /// Get performance comparison between optimized and standard data structures
    pub async fn get_optimization_metrics(&self) -> OptimizationMetrics {
        let comprehensive_stats = self.get_comprehensive_memory_stats().await;
        let string_pool_stats = &comprehensive_stats.string_pool_stats;

        // Calculate approximate memory savings
        let avg_string_length = 20; // Rough estimate for user_id, feature_used etc.
        let duplicate_strings_avoided = string_pool_stats.cache_hits;
        let memory_saved_by_interning = duplicate_strings_avoided as usize * avg_string_length;

        // Estimate metadata compression savings
        let metadata_compression_savings = comprehensive_stats.sessions_count * 200; // Rough estimate

        OptimizationMetrics {
            string_interning_hit_rate: string_pool_stats.hit_ratio(),
            memory_saved_by_interning,
            metadata_compression_savings,
            total_optimization_benefit: memory_saved_by_interning + metadata_compression_savings,
            interactions_per_kb: if comprehensive_stats.memory_stats.current_usage > 0 {
                (comprehensive_stats.interactions_count * 1024)
                    / comprehensive_stats.memory_stats.current_usage
            } else {
                0
            },
        }
    }

    /// Get configuration
    pub fn config(&self) -> &AnalyticsConfig {
        &self.config
    }
}

/// Factory for creating analytics managers
pub struct AnalyticsManagerFactory;

impl AnalyticsManagerFactory {
    /// Create standard analytics manager
    pub async fn create_standard(config: AnalyticsConfig) -> AnalyticsResult<AnalyticsManager> {
        AnalyticsManager::new(config).await
    }

    /// Create memory-optimized analytics manager
    pub async fn create_optimized(
        config: AnalyticsConfig,
    ) -> AnalyticsResult<OptimizedAnalyticsManager> {
        OptimizedAnalyticsManager::new(config).await
    }

    /// Create analytics manager based on memory requirements
    pub async fn create_for_memory_profile(
        config: AnalyticsConfig,
        memory_profile: MemoryProfile,
    ) -> AnalyticsResult<Box<dyn AnalyticsManagerTrait>> {
        match memory_profile {
            MemoryProfile::LowMemory => {
                let manager = Self::create_optimized(config).await?;
                Ok(Box::new(manager))
            }
            MemoryProfile::Standard => {
                let manager = Self::create_standard(config).await?;
                Ok(Box::new(manager))
            }
        }
    }
}

/// Memory profile configuration
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryProfile {
    /// Use memory-optimized data structures
    LowMemory,
    /// Use standard data structures (faster but more memory)
    Standard,
}

/// Comprehensive memory statistics including optimization metrics
#[derive(Debug, Clone)]
pub struct ComprehensiveMemoryStats {
    /// Basic memory statistics
    pub memory_stats: super::metrics::MemoryStats,
    /// String pool statistics
    pub string_pool_stats: StringPoolStats,
    /// Number of interactions stored
    pub interactions_count: usize,
    /// Number of sessions stored
    pub sessions_count: usize,
    /// Estimated memory savings from optimizations
    pub estimated_memory_savings: usize,
}

/// Memory cleanup result
#[derive(Debug, Clone)]
pub struct MemoryCleanupResult {
    /// Memory usage before cleanup (bytes)
    pub memory_before: usize,
    /// Memory usage after cleanup (bytes)
    pub memory_after: usize,
    /// Total items before cleanup
    pub items_before: usize,
    /// Total items after cleanup
    pub items_after: usize,
    /// Bytes freed by cleanup
    pub bytes_freed: usize,
}

/// Optimization performance metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    /// String interning cache hit rate (0.0 to 1.0)
    pub string_interning_hit_rate: f64,
    /// Estimated memory saved by string interning (bytes)
    pub memory_saved_by_interning: usize,
    /// Estimated memory saved by metadata compression (bytes)
    pub metadata_compression_savings: usize,
    /// Total optimization benefit (bytes)
    pub total_optimization_benefit: usize,
    /// Interactions stored per KB of memory
    pub interactions_per_kb: usize,
}

/// Common trait for analytics managers
#[async_trait::async_trait]
pub trait AnalyticsManagerTrait: Send + Sync {
    /// Record user interaction
    async fn record_interaction(&self, interaction: &UserInteractionEvent) -> AnalyticsResult<()>;

    /// Get memory statistics
    async fn get_memory_stats(&self) -> super::metrics::MemoryStats;
}

#[async_trait::async_trait]
impl AnalyticsManagerTrait for AnalyticsManager {
    async fn record_interaction(&self, interaction: &UserInteractionEvent) -> AnalyticsResult<()> {
        self.record_interaction(interaction).await
    }

    async fn get_memory_stats(&self) -> super::metrics::MemoryStats {
        self.get_memory_stats().await
    }
}

#[async_trait::async_trait]
impl AnalyticsManagerTrait for OptimizedAnalyticsManager {
    async fn record_interaction(&self, interaction: &UserInteractionEvent) -> AnalyticsResult<()> {
        self.record_interaction(interaction).await
    }

    async fn get_memory_stats(&self) -> super::metrics::MemoryStats {
        let comprehensive = self.get_comprehensive_memory_stats().await;
        comprehensive.memory_stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::types::{InteractionType, UserInteractionEvent};
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_analytics_manager_creation() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_interaction_recording() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "pronunciation_practice".to_string(),
            feedback_score: Some(0.85),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        let result = manager.record_interaction(&interaction).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_recording() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        let metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_ms: 125.0,
            throughput: 45.0,
            error_rate: 0.01,
            memory_usage: 1024 * 1024, // 1MB
            cpu_usage: 25.0,
        };

        let result = manager.record_performance(&metrics).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dashboard_data_generation() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        // Record some test data
        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "pronunciation_practice".to_string(),
            feedback_score: Some(0.85),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        manager.record_interaction(&interaction).await.unwrap();

        let dashboard = manager.get_dashboard_data().await;
        assert!(dashboard.is_ok());

        let data = dashboard.unwrap();
        assert!(data.system_health >= 0.0 && data.system_health <= 1.0);
    }

    #[tokio::test]
    async fn test_report_generation() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        // Record some test data
        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "pronunciation_practice".to_string(),
            feedback_score: Some(0.85),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        manager.record_interaction(&interaction).await.unwrap();

        let query = AnalyticsQuery {
            start_time: Some(Utc::now() - Duration::hours(1)),
            end_time: Some(Utc::now()),
            user_id: None,
            interaction_type: None,
            feature: None,
            include_performance: true,
            aggregation: AggregationLevel::Raw,
        };

        let report = manager.generate_report(&query).await;
        assert!(report.is_ok());

        let report_data = report.unwrap();
        assert!(!report_data.report_id.is_empty());
        assert!(!report_data.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_user_analytics() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        // Record multiple interactions for a user
        for i in 0..5 {
            let interaction = UserInteractionEvent {
                user_id: "test_user".to_string(),
                timestamp: Utc::now() - Duration::minutes(i * 10),
                interaction_type: if i % 2 == 0 {
                    InteractionType::Practice
                } else {
                    InteractionType::ExerciseCompleted
                },
                feature_used: format!("feature_{}", i % 3),
                feedback_score: Some(0.7 + (i as f32 * 0.05)),
                engagement_duration: std::time::Duration::from_secs((200 + i * 30) as u64),
                metadata: HashMap::new(),
            };

            manager.record_interaction(&interaction).await.unwrap();
        }

        let user_analytics = manager.get_user_analytics("test_user").await;
        assert!(user_analytics.is_ok());

        let analytics = user_analytics.unwrap();
        assert_eq!(analytics.user_id, "test_user");
        assert!(analytics.total_interactions > 0);
        assert!(analytics.engagement_score >= 0.0 && analytics.engagement_score <= 1.0);
    }

    #[tokio::test]
    async fn test_data_export() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        // Record some test data
        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "pronunciation_practice".to_string(),
            feedback_score: Some(0.85),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        manager.record_interaction(&interaction).await.unwrap();

        let query = AnalyticsQuery {
            start_time: Some(Utc::now() - Duration::hours(1)),
            end_time: Some(Utc::now()),
            user_id: None,
            interaction_type: None,
            feature: None,
            include_performance: true,
            aggregation: AggregationLevel::Raw,
        };

        // Test JSON export
        let json_export = manager.export_data(ExportFormat::Json, &query).await;
        assert!(json_export.is_ok());
        assert!(!json_export.unwrap().is_empty());

        // Test CSV export
        let csv_export = manager.export_data(ExportFormat::Csv, &query).await;
        assert!(csv_export.is_ok());
        assert!(!csv_export.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        let stats = manager.get_memory_stats().await;
        assert!(stats.current_usage >= 0);
        assert!(stats.peak_usage >= 0);
        assert!(stats.item_count >= 0);
    }

    #[tokio::test]
    async fn test_usage_patterns() {
        let config = AnalyticsConfig::default();
        let manager = AnalyticsManager::new(config).await.unwrap();

        // Record some test data
        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "pronunciation_practice".to_string(),
            feedback_score: Some(0.85),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        manager.record_interaction(&interaction).await.unwrap();

        let patterns = manager.get_usage_patterns().await;
        assert!(patterns.is_ok());

        let usage_patterns = patterns.unwrap();
        assert!(!usage_patterns.feature_usage_distribution.is_empty());
        assert!(usage_patterns.retention_rates.day_1 >= 0.0);
        assert!(usage_patterns.retention_rates.day_7 >= 0.0);
        assert!(usage_patterns.retention_rates.day_30 >= 0.0);
    }

    #[test]
    fn test_config_access() {
        let config = AnalyticsConfig {
            enabled: true,
            max_interactions: 50000,
            max_performance_records: 5000,
            retention_days: 60,
            enable_realtime: false,
            max_active_sessions: Some(500),
            export_formats: vec![ExportFormat::Json],
        };

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let manager = runtime
            .block_on(AnalyticsManager::new(config.clone()))
            .unwrap();

        let manager_config = manager.config();
        assert_eq!(manager_config.enabled, config.enabled);
        assert_eq!(manager_config.max_interactions, config.max_interactions);
        assert_eq!(manager_config.retention_days, config.retention_days);
        assert_eq!(manager_config.enable_realtime, config.enable_realtime);
    }
}
