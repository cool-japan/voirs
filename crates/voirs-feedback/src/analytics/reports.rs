//! Report generation functionality for analytics

use super::data::DataCollector;
use super::types::*;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Report generator for creating analytics summaries
#[derive(Debug)]
pub struct ReportGenerator {
    config: AnalyticsConfig,
}

impl ReportGenerator {
    /// Create new report generator
    pub async fn new(config: &AnalyticsConfig) -> AnalyticsResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Generate comprehensive analytics report
    pub async fn generate_report(
        &mut self,
        collector: &DataCollector,
        query: &AnalyticsQuery,
    ) -> AnalyticsResult<AnalyticsReport> {
        let start_time = query
            .start_time
            .unwrap_or_else(|| Utc::now() - Duration::days(30));
        let end_time = query.end_time.unwrap_or_else(Utc::now);

        // Filter data by time range
        let filtered_interactions = collector
            .get_interactions()
            .iter()
            .filter(|i| i.timestamp >= start_time && i.timestamp <= end_time)
            .collect::<Vec<_>>();

        let filtered_performance = collector
            .get_performance_history()
            .iter()
            .filter(|p| p.timestamp >= start_time && p.timestamp <= end_time)
            .collect::<Vec<_>>();

        // Generate report sections
        let executive_summary =
            self.generate_executive_summary(&filtered_interactions, &filtered_performance);
        let user_engagement = self.generate_user_engagement_report(&filtered_interactions);
        let performance_analysis = self.generate_performance_analysis(&filtered_performance);
        let feature_usage = self.generate_feature_usage_report(&filtered_interactions);
        let recommendations =
            self.generate_recommendations(&filtered_interactions, &filtered_performance);

        Ok(AnalyticsReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            generated_at: Utc::now(),
            time_range: TimeRange {
                start: start_time,
                end: end_time,
            },
            executive_summary,
            user_engagement,
            performance_analysis,
            feature_usage,
            recommendations,
        })
    }

    /// Export analytics data
    pub async fn export_data(
        &mut self,
        collector: &DataCollector,
        format: ExportFormat,
        query: &AnalyticsQuery,
    ) -> AnalyticsResult<Vec<u8>> {
        match format {
            ExportFormat::Json => {
                let report = self.generate_report(collector, query).await?;
                serde_json::to_vec(&report).map_err(|e| AnalyticsError::ReportGenerationError {
                    message: format!("JSON serialization failed: {}", e),
                })
            }
            ExportFormat::Csv => {
                // Generate CSV export
                let mut csv_data = String::new();
                csv_data.push_str("timestamp,user_id,interaction_type,feature_used,feedback_score,engagement_duration\n");

                for interaction in collector.get_interactions() {
                    csv_data.push_str(&format!(
                        "{},{},{:?},{},{},{}\n",
                        interaction.timestamp.to_rfc3339(),
                        interaction.user_id,
                        interaction.interaction_type,
                        interaction.feature_used,
                        interaction.feedback_score.unwrap_or(0.0),
                        interaction.engagement_duration.as_secs()
                    ));
                }

                Ok(csv_data.into_bytes())
            }
        }
    }

    // Helper methods for report generation

    fn generate_executive_summary(
        &self,
        interactions: &[&UserInteractionEvent],
        performance: &[&PerformanceMetrics],
    ) -> ExecutiveSummary {
        let total_interactions = interactions.len();
        let unique_users = interactions
            .iter()
            .map(|i| &i.user_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let avg_performance = if performance.is_empty() {
            0.0
        } else {
            performance.iter().map(|p| p.latency_ms).sum::<f32>() / performance.len() as f32
        };

        let user_satisfaction = if interactions.is_empty() {
            0.0
        } else {
            interactions
                .iter()
                .filter_map(|i| i.feedback_score)
                .sum::<f32>()
                / interactions.len() as f32
        };

        ExecutiveSummary {
            total_interactions,
            unique_users,
            average_performance_ms: avg_performance,
            user_satisfaction_score: user_satisfaction,
            key_insights: vec![
                "Strong user engagement in core features".to_string(),
                "Performance metrics within acceptable range".to_string(),
                "Consistent growth in user adoption".to_string(),
            ],
        }
    }

    fn generate_user_engagement_report(
        &self,
        interactions: &[&UserInteractionEvent],
    ) -> UserEngagementReport {
        let total_sessions = interactions.len();
        let avg_session_duration = if interactions.is_empty() {
            chrono::Duration::zero()
        } else {
            let total_duration = interactions
                .iter()
                .map(|i| i.engagement_duration.as_secs() as i64)
                .sum::<i64>();
            Duration::seconds(total_duration / interactions.len() as i64)
        };

        let completion_rate = if total_sessions == 0 {
            0.0
        } else {
            interactions
                .iter()
                .filter(|i| i.interaction_type == InteractionType::ExerciseCompleted)
                .count() as f32
                / total_sessions as f32
        };

        UserEngagementReport {
            total_sessions,
            average_session_duration: avg_session_duration,
            completion_rate,
            most_popular_features: self.get_popular_features(interactions),
            engagement_trends: self.calculate_engagement_trends(interactions),
        }
    }

    fn generate_performance_analysis(
        &self,
        performance: &[&PerformanceMetrics],
    ) -> PerformanceAnalysis {
        if performance.is_empty() {
            return PerformanceAnalysis {
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                error_rate: 0.0,
                throughput_rps: 0.0,
                performance_trends: Vec::new(),
            };
        }

        let mut latencies = performance.iter().map(|p| p.latency_ms).collect::<Vec<_>>();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
        let p95_index = (latencies.len() as f32 * 0.95) as usize;
        let p95_latency = latencies.get(p95_index).copied().unwrap_or(0.0);

        let avg_error_rate =
            performance.iter().map(|p| p.error_rate).sum::<f32>() / performance.len() as f32;
        let avg_throughput =
            performance.iter().map(|p| p.throughput).sum::<f32>() / performance.len() as f32;

        PerformanceAnalysis {
            average_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            error_rate: avg_error_rate,
            throughput_rps: avg_throughput,
            performance_trends: self.calculate_performance_trends(performance),
        }
    }

    fn generate_feature_usage_report(
        &self,
        interactions: &[&UserInteractionEvent],
    ) -> FeatureUsageReport {
        let mut usage_counts = HashMap::new();

        for interaction in interactions {
            *usage_counts
                .entry(interaction.feature_used.clone())
                .or_insert(0) += 1;
        }

        let total_usage = usage_counts.values().sum::<u32>() as f32;
        let usage_percentages = usage_counts
            .iter()
            .map(|(feature, count)| (feature.clone(), (*count as f32 / total_usage) * 100.0))
            .collect();

        FeatureUsageReport {
            total_feature_interactions: total_usage as u32,
            usage_by_feature: usage_percentages,
            trending_features: self.identify_trending_features(interactions),
        }
    }

    fn generate_recommendations(
        &self,
        interactions: &[&UserInteractionEvent],
        performance: &[&PerformanceMetrics],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if !performance.is_empty() {
            let avg_latency =
                performance.iter().map(|p| p.latency_ms).sum::<f32>() / performance.len() as f32;
            if avg_latency > 200.0 {
                recommendations
                    .push("Consider optimizing system performance to reduce latency".to_string());
            }
        }

        // User engagement recommendations
        if !interactions.is_empty() {
            let avg_engagement = interactions
                .iter()
                .map(|i| i.engagement_duration.as_secs() as f32)
                .sum::<f32>()
                / interactions.len() as f32;

            if avg_engagement < 120.0 {
                recommendations
                    .push("Implement features to increase user engagement duration".to_string());
            }
        }

        // Feature usage recommendations
        let feature_usage = self.analyze_feature_usage(interactions);
        if feature_usage.len() > 5 {
            recommendations
                .push("Consider highlighting underutilized features to users".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("System is performing well with good user engagement".to_string());
        }

        recommendations
    }

    fn get_popular_features(&self, interactions: &[&UserInteractionEvent]) -> Vec<String> {
        let mut usage_counts = HashMap::new();

        for interaction in interactions {
            *usage_counts
                .entry(interaction.feature_used.clone())
                .or_insert(0) += 1;
        }

        let mut features: Vec<_> = usage_counts.into_iter().collect();
        features.sort_by(|a, b| b.1.cmp(&a.1));

        features
            .into_iter()
            .take(5)
            .map(|(feature, _)| feature)
            .collect()
    }

    fn calculate_engagement_trends(
        &self,
        interactions: &[&UserInteractionEvent],
    ) -> Vec<TrendPoint> {
        // Group interactions by day and calculate engagement trends
        let mut daily_engagement: HashMap<i64, Vec<&UserInteractionEvent>> = HashMap::new();

        for interaction in interactions {
            let day = interaction
                .timestamp
                .date_naive()
                .and_time(chrono::NaiveTime::MIN)
                .and_utc()
                .timestamp();
            daily_engagement
                .entry(day)
                .or_insert_with(Vec::new)
                .push(interaction);
        }

        let mut trends = Vec::new();
        for (day, day_interactions) in daily_engagement {
            let avg_duration = day_interactions
                .iter()
                .map(|i| i.engagement_duration.as_secs() as f32)
                .sum::<f32>()
                / day_interactions.len() as f32;

            trends.push(TrendPoint {
                timestamp: DateTime::from_timestamp(day, 0).unwrap_or_else(Utc::now),
                latency: avg_duration,
                throughput: day_interactions.len() as f32,
                error_rate: 0.0,
            });
        }

        trends.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        trends
    }

    fn calculate_performance_trends(&self, performance: &[&PerformanceMetrics]) -> Vec<TrendPoint> {
        // Group performance data by hour
        let mut hourly_performance: HashMap<i64, Vec<&PerformanceMetrics>> = HashMap::new();

        for metric in performance {
            let hour = metric.timestamp.timestamp() / 3600;
            hourly_performance
                .entry(hour)
                .or_insert_with(Vec::new)
                .push(metric);
        }

        let mut trends = Vec::new();
        for (hour, metrics) in hourly_performance {
            let avg_latency =
                metrics.iter().map(|m| m.latency_ms).sum::<f32>() / metrics.len() as f32;
            let avg_throughput =
                metrics.iter().map(|m| m.throughput).sum::<f32>() / metrics.len() as f32;
            let avg_error_rate =
                metrics.iter().map(|m| m.error_rate).sum::<f32>() / metrics.len() as f32;

            trends.push(TrendPoint {
                timestamp: DateTime::from_timestamp(hour * 3600, 0).unwrap_or_else(Utc::now),
                latency: avg_latency,
                throughput: avg_throughput,
                error_rate: avg_error_rate,
            });
        }

        trends.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        trends
    }

    fn identify_trending_features(&self, interactions: &[&UserInteractionEvent]) -> Vec<String> {
        // Simple trending calculation based on recent vs historical usage
        let now = Utc::now();
        let recent_cutoff = now - Duration::days(7);

        let recent_usage = interactions
            .iter()
            .filter(|i| i.timestamp > recent_cutoff)
            .fold(HashMap::new(), |mut acc, interaction| {
                *acc.entry(interaction.feature_used.clone()).or_insert(0) += 1;
                acc
            });

        let historical_usage = interactions
            .iter()
            .filter(|i| i.timestamp <= recent_cutoff)
            .fold(HashMap::new(), |mut acc, interaction| {
                *acc.entry(interaction.feature_used.clone()).or_insert(0) += 1;
                acc
            });

        let mut trending = Vec::new();
        for (feature, recent_count) in recent_usage {
            let historical_count = historical_usage.get(&feature).copied().unwrap_or(0);
            let growth_rate = if historical_count > 0 {
                (recent_count as f32 / historical_count as f32) - 1.0
            } else {
                1.0 // New features are trending
            };

            if growth_rate > 0.2 {
                // 20% growth threshold
                trending.push(feature);
            }
        }

        trending
    }

    fn analyze_feature_usage(
        &self,
        interactions: &[&UserInteractionEvent],
    ) -> HashMap<String, u32> {
        let mut usage = HashMap::new();

        for interaction in interactions {
            *usage.entry(interaction.feature_used.clone()).or_insert(0) += 1;
        }

        usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::types::{InteractionType, PerformanceMetrics};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_report_generator_creation() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator::new(&config).await;
        assert!(generator.is_ok());
    }

    #[test]
    fn test_executive_summary_generation() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator { config };

        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "test_feature".to_string(),
            feedback_score: Some(0.8),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        let interactions = vec![&interaction];
        let performance = vec![];

        let summary = generator.generate_executive_summary(&interactions, &performance);
        assert_eq!(summary.total_interactions, 1);
        assert_eq!(summary.unique_users, 1);
        assert_eq!(summary.user_satisfaction_score, 0.8);
    }

    #[test]
    fn test_user_engagement_report_generation() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator { config };

        let interaction = UserInteractionEvent {
            user_id: "test_user".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::ExerciseCompleted,
            feature_used: "test_feature".to_string(),
            feedback_score: Some(0.8),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        let interactions = vec![&interaction];
        let report = generator.generate_user_engagement_report(&interactions);

        assert_eq!(report.total_sessions, 1);
        assert_eq!(report.completion_rate, 1.0); // 100% completion rate
        assert!(report
            .most_popular_features
            .contains(&"test_feature".to_string()));
    }

    #[test]
    fn test_performance_analysis_generation() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator { config };

        let metric = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_ms: 100.0,
            throughput: 50.0,
            error_rate: 0.01,
            memory_usage: 1024,
            cpu_usage: 25.0,
        };

        let performance = vec![&metric];
        let analysis = generator.generate_performance_analysis(&performance);

        assert_eq!(analysis.average_latency_ms, 100.0);
        assert_eq!(analysis.error_rate, 0.01);
        assert_eq!(analysis.throughput_rps, 50.0);
    }

    #[test]
    fn test_feature_usage_report_generation() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator { config };

        let interaction1 = UserInteractionEvent {
            user_id: "user1".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "feature_a".to_string(),
            feedback_score: Some(0.8),
            engagement_duration: std::time::Duration::from_secs(300),
            metadata: HashMap::new(),
        };

        let interaction2 = UserInteractionEvent {
            user_id: "user2".to_string(),
            timestamp: Utc::now(),
            interaction_type: InteractionType::Practice,
            feature_used: "feature_a".to_string(),
            feedback_score: Some(0.9),
            engagement_duration: std::time::Duration::from_secs(200),
            metadata: HashMap::new(),
        };

        let interactions = vec![&interaction1, &interaction2];
        let report = generator.generate_feature_usage_report(&interactions);

        assert_eq!(report.total_feature_interactions, 2);
        assert!(report.usage_by_feature.contains_key("feature_a"));
        assert_eq!(report.usage_by_feature["feature_a"], 100.0); // 100% usage
    }

    #[test]
    fn test_popular_features_extraction() {
        let config = AnalyticsConfig::default();
        let generator = ReportGenerator { config };

        let interactions = vec![
            UserInteractionEvent {
                user_id: "user1".to_string(),
                timestamp: Utc::now(),
                interaction_type: InteractionType::Practice,
                feature_used: "popular_feature".to_string(),
                feedback_score: Some(0.8),
                engagement_duration: std::time::Duration::from_secs(300),
                metadata: HashMap::new(),
            },
            UserInteractionEvent {
                user_id: "user2".to_string(),
                timestamp: Utc::now(),
                interaction_type: InteractionType::Practice,
                feature_used: "popular_feature".to_string(),
                feedback_score: Some(0.9),
                engagement_duration: std::time::Duration::from_secs(200),
                metadata: HashMap::new(),
            },
            UserInteractionEvent {
                user_id: "user3".to_string(),
                timestamp: Utc::now(),
                interaction_type: InteractionType::Practice,
                feature_used: "less_popular_feature".to_string(),
                feedback_score: Some(0.7),
                engagement_duration: std::time::Duration::from_secs(150),
                metadata: HashMap::new(),
            },
        ];

        let interaction_refs: Vec<_> = interactions.iter().collect();
        let popular = generator.get_popular_features(&interaction_refs);

        assert_eq!(popular[0], "popular_feature");
        assert_eq!(popular[1], "less_popular_feature");
    }
}
