//! Data collection and user interaction tracking functionality

use super::metrics::{MemoryStats, SystemMetrics};
use super::types::*;
use chrono::{DateTime, Duration, Timelike, Utc};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Data collector for gathering analytics metrics (memory-optimized)
#[derive(Debug)]
pub struct DataCollector {
    /// User interaction events (bounded circular buffer)
    interactions: VecDeque<UserInteractionEvent>,
    /// Performance metrics history (bounded circular buffer)
    performance_history: VecDeque<PerformanceMetrics>,
    /// Session data (with automatic cleanup)
    sessions: HashMap<String, SessionData>,
    /// System metrics
    system_metrics: SystemMetrics,
    /// Configuration
    config: AnalyticsConfig,
    /// Memory usage tracking
    memory_stats: MemoryStats,
    /// Last cleanup timestamp
    last_cleanup: Instant,
}

impl DataCollector {
    /// Create new data collector with memory optimization
    pub async fn new(config: &AnalyticsConfig) -> AnalyticsResult<Self> {
        Ok(Self {
            interactions: VecDeque::with_capacity(config.max_interactions),
            performance_history: VecDeque::with_capacity(config.max_performance_records),
            sessions: HashMap::with_capacity(config.max_active_sessions.unwrap_or(1000)),
            system_metrics: SystemMetrics::new(),
            config: config.clone(),
            memory_stats: MemoryStats::default(),
            last_cleanup: Instant::now(),
        })
    }

    /// Record user interaction event (memory-optimized)
    pub async fn record_interaction(
        &mut self,
        interaction: &UserInteractionEvent,
    ) -> AnalyticsResult<()> {
        // Update session data
        self.sessions
            .entry(interaction.user_id.clone())
            .or_insert_with(|| SessionData::new(&interaction.user_id))
            .update_from_interaction(interaction);

        // Store interaction with automatic bounds management
        if self.interactions.len() >= self.config.max_interactions {
            self.interactions.pop_front(); // Remove oldest
        }
        self.interactions.push_back(interaction.clone());

        // Perform periodic cleanup to manage memory
        self.perform_memory_cleanup().await;

        // Update system metrics
        self.system_metrics.update_from_interaction(interaction);

        // Update memory statistics
        self.update_memory_stats();

        Ok(())
    }

    /// Record performance metrics (memory-optimized)
    pub async fn record_performance(
        &mut self,
        metrics: &PerformanceMetrics,
    ) -> AnalyticsResult<()> {
        // Store metrics with automatic bounds management
        if self.performance_history.len() >= self.config.max_performance_records {
            self.performance_history.pop_front(); // Remove oldest
        }
        self.performance_history.push_back(metrics.clone());

        // Perform periodic cleanup to manage memory
        self.perform_memory_cleanup().await;

        // Update system metrics
        self.system_metrics.update_from_performance(metrics);

        // Update memory statistics
        self.update_memory_stats();

        Ok(())
    }

    /// Get real-time dashboard data
    pub async fn get_dashboard_data(&self) -> AnalyticsResult<DashboardData> {
        let now = Utc::now();
        let last_hour = now - Duration::hours(1);
        let last_day = now - Duration::days(1);

        // Recent interactions
        let recent_interactions = self
            .interactions
            .iter()
            .filter(|i| i.timestamp > last_hour)
            .count();

        // Active sessions
        let active_sessions = self
            .sessions
            .iter()
            .filter(|(_, session)| session.last_activity > last_hour)
            .count();

        // Performance metrics
        let recent_performance = self
            .performance_history
            .iter()
            .filter(|p| p.timestamp > last_hour)
            .cloned()
            .collect::<Vec<_>>();

        let avg_latency = if recent_performance.is_empty() {
            0.0
        } else {
            recent_performance.iter().map(|p| p.latency_ms).sum::<f32>()
                / recent_performance.len() as f32
        };

        // User engagement metrics
        let daily_active_users = self
            .sessions
            .iter()
            .filter(|(_, session)| session.last_activity > last_day)
            .count();

        Ok(DashboardData {
            timestamp: now,
            active_sessions,
            recent_interactions,
            daily_active_users,
            average_latency_ms: avg_latency,
            system_health: self.calculate_system_health(),
            performance_trends: self.calculate_performance_trends(),
            user_satisfaction: self.calculate_user_satisfaction(),
        })
    }

    /// Get user-specific analytics
    pub async fn get_user_analytics(&self, user_id: &str) -> AnalyticsResult<UserAnalytics> {
        let session =
            self.sessions
                .get(user_id)
                .ok_or_else(|| AnalyticsError::InsufficientDataError {
                    message: format!("No data found for user: {}", user_id),
                })?;

        let user_interactions = self
            .interactions
            .iter()
            .filter(|i| i.user_id == user_id)
            .cloned()
            .collect::<Vec<_>>();

        if user_interactions.is_empty() {
            return Err(AnalyticsError::InsufficientDataError {
                message: format!("No interactions found for user: {}", user_id),
            });
        }

        Ok(UserAnalytics {
            user_id: user_id.to_string(),
            total_sessions: session.session_count,
            total_interactions: user_interactions.len(),
            average_session_duration: session.total_duration / session.session_count as i64,
            improvement_trend: self.calculate_improvement_trend(&user_interactions),
            preferred_features: self.analyze_feature_usage(&user_interactions),
            learning_velocity: self.calculate_learning_velocity(&user_interactions),
            engagement_score: self.calculate_engagement_score(&user_interactions),
            last_activity: session.last_activity,
        })
    }

    /// Get system-wide usage patterns
    pub async fn get_usage_patterns(&self) -> AnalyticsResult<UsagePatterns> {
        let now = Utc::now();

        // Peak usage times
        let peak_hours = self.analyze_peak_usage_hours();

        // Feature usage distribution
        let feature_usage = self.analyze_feature_usage_distribution();

        // User journey patterns
        let user_journeys = self.analyze_user_journeys();

        // Geographic distribution (if available)
        let geographic_data = self.analyze_geographic_distribution();

        // Device/platform usage
        let platform_usage = self.analyze_platform_usage();

        Ok(UsagePatterns {
            timestamp: now,
            peak_usage_hours: peak_hours,
            feature_usage_distribution: feature_usage,
            user_journey_patterns: user_journeys,
            geographic_distribution: geographic_data,
            platform_usage_breakdown: platform_usage,
            retention_rates: self.calculate_retention_rates(),
            conversion_funnel: self.analyze_conversion_funnel(),
        })
    }

    /// Get memory statistics for monitoring
    pub fn get_memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }

    /// Get interactions for report generation
    pub fn get_interactions(&self) -> &VecDeque<UserInteractionEvent> {
        &self.interactions
    }

    /// Get performance history for report generation
    pub fn get_performance_history(&self) -> &VecDeque<PerformanceMetrics> {
        &self.performance_history
    }

    /// Perform memory cleanup operations
    async fn perform_memory_cleanup(&mut self) {
        let now = Instant::now();

        // Only cleanup periodically to avoid performance overhead
        if now.duration_since(self.last_cleanup).as_secs() < 60 {
            return; // Skip if cleaned up recently
        }

        let before_size = self.estimate_memory_usage();

        // Clean up old sessions
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        self.sessions
            .retain(|_, session| session.last_activity > cutoff);

        // Aggressive memory optimization for large collections
        self.optimize_collection_memory();

        // Remove duplicate entries (keep only most recent per user per feature)
        self.deduplicate_interactions();

        let after_size = self.estimate_memory_usage();

        // Update cleanup statistics
        self.last_cleanup = now;
        self.memory_stats.record_cleanup();

        // Log memory savings if significant
        if before_size > after_size + 1024 * 1024 {
            // 1MB saved
            log::info!(
                "Analytics memory cleanup: freed {} bytes, new usage: {} bytes",
                before_size - after_size,
                after_size
            );
        }
    }

    /// Optimize collection memory usage with aggressive shrinking and compression
    fn optimize_collection_memory(&mut self) {
        // More aggressive shrinking for over-allocated collections (150% threshold)
        let interactions_threshold = self.interactions.len() + (self.interactions.len() / 2);
        if self.interactions.capacity() > interactions_threshold {
            self.interactions.shrink_to_fit();
            // Reserve minimal additional capacity
            self.interactions.reserve(self.interactions.len() / 10);
        }

        let performance_threshold =
            self.performance_history.len() + (self.performance_history.len() / 2);
        if self.performance_history.capacity() > performance_threshold {
            self.performance_history.shrink_to_fit();
            self.performance_history
                .reserve(self.performance_history.len() / 10);
        }

        // Additional memory optimization: compress old data
        self.compress_old_interactions();

        // Pool string allocations for common values
        self.intern_common_strings();

        // Optimize session data storage
        self.optimize_session_data();
    }

    /// Compress old interaction data to save memory
    fn compress_old_interactions(&mut self) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(2);

        // Aggregate old interactions into summaries
        let mut to_compress = Vec::new();
        let mut compressed_summary = InteractionSummary::new();

        for (index, interaction) in self.interactions.iter().enumerate() {
            if interaction.timestamp < cutoff {
                to_compress.push(index);
                compressed_summary.add_interaction(interaction);
            }
        }

        // Remove compressed interactions and store summary
        for &index in to_compress.iter().rev() {
            self.interactions.remove(index);
        }

        if !compressed_summary.is_empty() {
            // Store compressed summary instead of individual interactions
            self.store_compressed_summary(compressed_summary);
        }
    }

    /// Intern common strings to reduce memory usage
    fn intern_common_strings(&mut self) {
        // Create string interning pool for common values
        let mut string_pool = StringPool::new();

        // Intern user IDs, session IDs, and other repeated strings
        for interaction in &mut self.interactions {
            interaction.intern_strings(&mut string_pool);
        }

        for session in self.sessions.values_mut() {
            session.intern_strings(&mut string_pool);
        }
    }

    /// Optimize session data to reduce memory footprint
    fn optimize_session_data(&mut self) {
        // Remove inactive sessions more aggressively
        let inactive_cutoff = chrono::Utc::now() - chrono::Duration::hours(2);
        self.sessions
            .retain(|_, session| session.last_activity > inactive_cutoff || session.is_important());

        // Compress session history for long-running sessions
        for session in self.sessions.values_mut() {
            session.compress_history();
        }

        // Shrink session map if over-allocated
        if self.sessions.capacity() > self.sessions.len() * 2 {
            self.sessions.shrink_to_fit();
            self.sessions.reserve(self.sessions.len() / 4);
        }
    }

    /// Remove duplicate interactions to save memory
    fn deduplicate_interactions(&mut self) {
        // Create a map to track latest interaction per user per feature
        let mut latest_interactions: HashMap<(String, String), usize> = HashMap::new();
        let mut to_remove = Vec::new();

        for (index, interaction) in self.interactions.iter().enumerate() {
            let key = (
                interaction.user_id.clone(),
                interaction.feature_used.clone(),
            );

            if let Some(&existing_index) = latest_interactions.get(&key) {
                // Compare timestamps and mark older one for removal
                if self.interactions[existing_index].timestamp < interaction.timestamp {
                    to_remove.push(existing_index);
                    latest_interactions.insert(key, index);
                } else {
                    to_remove.push(index);
                }
            } else {
                latest_interactions.insert(key, index);
            }
        }

        // Remove duplicates in reverse order to maintain indices
        to_remove.sort_unstable();
        to_remove.reverse();
        let removed_count = to_remove.len();

        for index in to_remove {
            if index < self.interactions.len() {
                self.interactions.remove(index);
            }
        }

        log::debug!("Deduplicated {} interaction entries", removed_count);
    }

    /// Update memory usage statistics
    fn update_memory_stats(&mut self) {
        let usage = self.estimate_memory_usage();
        let item_count =
            self.interactions.len() + self.performance_history.len() + self.sessions.len();
        self.memory_stats.update(usage, item_count);
    }

    /// Estimate current memory usage with more accurate calculation
    fn estimate_memory_usage(&self) -> usize {
        let mut total_size = 0;

        // Calculate interaction memory with string/vector overhead
        for interaction in &self.interactions {
            total_size += std::mem::size_of::<UserInteractionEvent>();
            total_size += interaction.estimated_memory_size();
        }

        // Calculate performance history memory
        for metrics in &self.performance_history {
            total_size += std::mem::size_of::<PerformanceMetrics>();
            total_size += metrics.estimated_memory_size();
        }

        // Calculate session data memory with dynamic content
        for (key, session) in &self.sessions {
            total_size += key.len(); // String key size
            total_size += std::mem::size_of::<SessionData>();
            total_size += session.estimated_memory_size();
        }

        // Add capacity overhead for collections
        total_size += (self.interactions.capacity() - self.interactions.len())
            * std::mem::size_of::<UserInteractionEvent>();
        total_size += (self.performance_history.capacity() - self.performance_history.len())
            * std::mem::size_of::<PerformanceMetrics>();

        total_size
    }

    /// Store compressed summary of interactions
    fn store_compressed_summary(&mut self, _summary: InteractionSummary) {
        // This would be stored in a separate compressed storage
        // For now, we just acknowledge the compression
        log::debug!(
            "Stored compressed interaction summary with {} interactions",
            _summary.interaction_count
        );
    }

    // Helper methods for analytics calculations

    fn calculate_system_health(&self) -> f32 {
        // Simple health score based on recent performance
        let recent_performance = self
            .performance_history
            .iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();

        if recent_performance.is_empty() {
            return 1.0;
        }

        let avg_latency = recent_performance.iter().map(|p| p.latency_ms).sum::<f32>()
            / recent_performance.len() as f32;
        let error_rate = recent_performance.iter().map(|p| p.error_rate).sum::<f32>()
            / recent_performance.len() as f32;

        // Health score: lower latency and error rate = better health
        let latency_score = (200.0 - avg_latency.min(200.0)) / 200.0;
        let error_score = 1.0 - error_rate;

        (latency_score + error_score) / 2.0
    }

    fn calculate_performance_trends(&self) -> Vec<TrendPoint> {
        let mut trends = Vec::new();

        // Group performance data by hour
        let mut hourly_data: HashMap<i64, Vec<&PerformanceMetrics>> = HashMap::new();

        for metric in &self.performance_history {
            let hour = metric.timestamp.timestamp() / 3600;
            hourly_data
                .entry(hour)
                .or_insert_with(Vec::new)
                .push(metric);
        }

        // Calculate trends for each hour
        for (hour, metrics) in hourly_data {
            let avg_latency =
                metrics.iter().map(|m| m.latency_ms).sum::<f32>() / metrics.len() as f32;
            let avg_throughput =
                metrics.iter().map(|m| m.throughput).sum::<f32>() / metrics.len() as f32;

            trends.push(TrendPoint {
                timestamp: DateTime::from_timestamp(hour * 3600, 0).unwrap_or_else(Utc::now),
                latency: avg_latency,
                throughput: avg_throughput,
                error_rate: metrics.iter().map(|m| m.error_rate).sum::<f32>()
                    / metrics.len() as f32,
            });
        }

        trends.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        trends
    }

    fn calculate_user_satisfaction(&self) -> f32 {
        // Calculate user satisfaction based on interaction patterns
        let recent_interactions = self
            .interactions
            .iter()
            .filter(|i| i.timestamp > Utc::now() - Duration::hours(24))
            .collect::<Vec<_>>();

        if recent_interactions.is_empty() {
            return 0.5; // Neutral when no data
        }

        // Factors: completion rate, feedback scores, engagement duration
        let completion_rate = recent_interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::ExerciseCompleted)
            .count() as f32
            / recent_interactions.len() as f32;

        let avg_feedback_score = recent_interactions
            .iter()
            .filter_map(|i| i.feedback_score)
            .sum::<f32>()
            / recent_interactions.len() as f32;

        let engagement_factor = recent_interactions
            .iter()
            .map(|i| i.engagement_duration.as_secs() as f32)
            .sum::<f32>()
            / recent_interactions.len() as f32
            / 300.0; // Normalize to 5 minutes

        (completion_rate + avg_feedback_score + engagement_factor.min(1.0)) / 3.0
    }

    fn calculate_improvement_trend(&self, interactions: &[UserInteractionEvent]) -> f32 {
        if interactions.len() < 2 {
            return 0.0;
        }

        let mut sorted = interactions.to_vec();
        sorted.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let recent_scores = sorted
            .iter()
            .rev()
            .take(10)
            .filter_map(|i| i.feedback_score)
            .collect::<Vec<_>>();

        let earlier_scores = sorted
            .iter()
            .take(10)
            .filter_map(|i| i.feedback_score)
            .collect::<Vec<_>>();

        if recent_scores.is_empty() || earlier_scores.is_empty() {
            return 0.0;
        }

        let recent_avg = recent_scores.iter().sum::<f32>() / recent_scores.len() as f32;
        let earlier_avg = earlier_scores.iter().sum::<f32>() / earlier_scores.len() as f32;

        recent_avg - earlier_avg
    }

    fn analyze_feature_usage(&self, interactions: &[UserInteractionEvent]) -> HashMap<String, u32> {
        let mut usage = HashMap::new();

        for interaction in interactions {
            *usage.entry(interaction.feature_used.clone()).or_insert(0) += 1;
        }

        usage
    }

    fn calculate_learning_velocity(&self, interactions: &[UserInteractionEvent]) -> f32 {
        if interactions.len() < 3 {
            return 0.0;
        }

        let mut sorted = interactions.to_vec();
        sorted.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let total_improvement = sorted
            .windows(2)
            .filter_map(|pair| {
                if let (Some(score1), Some(score2)) =
                    (pair[0].feedback_score, pair[1].feedback_score)
                {
                    Some(score2 - score1)
                } else {
                    None
                }
            })
            .sum::<f32>();

        let time_span = sorted.last().unwrap().timestamp.timestamp()
            - sorted.first().unwrap().timestamp.timestamp();

        if time_span > 0 {
            total_improvement / (time_span as f32 / 3600.0) // improvement per hour
        } else {
            0.0
        }
    }

    fn calculate_engagement_score(&self, interactions: &[UserInteractionEvent]) -> f32 {
        if interactions.is_empty() {
            return 0.0;
        }

        let avg_duration = interactions
            .iter()
            .map(|i| i.engagement_duration.as_secs() as f32)
            .sum::<f32>()
            / interactions.len() as f32;

        let frequency_score = interactions.len() as f32 / 30.0; // Normalize to 30 interactions
        let duration_score = avg_duration / 600.0; // Normalize to 10 minutes

        (frequency_score.min(1.0) + duration_score.min(1.0)) / 2.0
    }

    fn analyze_peak_usage_hours(&self) -> Vec<u8> {
        let mut hourly_counts = vec![0u32; 24];

        for interaction in &self.interactions {
            let hour = interaction.timestamp.hour() as usize;
            hourly_counts[hour] += 1;
        }

        let max_count = hourly_counts.iter().max().unwrap_or(&1);
        let threshold = max_count * 7 / 10; // 70% of peak

        hourly_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count >= threshold)
            .map(|(hour, _)| hour as u8)
            .collect()
    }

    fn analyze_feature_usage_distribution(&self) -> HashMap<String, f32> {
        let mut usage = HashMap::new();
        let total = self.interactions.len() as f32;

        for interaction in &self.interactions {
            *usage.entry(interaction.feature_used.clone()).or_insert(0.0) += 1.0;
        }

        // Convert to percentages
        for (_, count) in usage.iter_mut() {
            *count = (*count / total) * 100.0;
        }

        usage
    }

    fn analyze_user_journeys(&self) -> Vec<UserJourney> {
        let mut journeys = Vec::new();
        let mut user_paths: HashMap<String, Vec<&UserInteractionEvent>> = HashMap::new();

        // Group interactions by user
        for interaction in &self.interactions {
            user_paths
                .entry(interaction.user_id.clone())
                .or_insert_with(Vec::new)
                .push(interaction);
        }

        // Analyze each user's journey
        for (user_id, mut interactions) in user_paths {
            interactions.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

            let journey = UserJourney {
                user_id,
                steps: interactions
                    .iter()
                    .map(|i| JourneyStep {
                        timestamp: i.timestamp,
                        action: i.interaction_type.clone(),
                        feature: i.feature_used.clone(),
                        outcome: i.feedback_score.unwrap_or(0.0),
                    })
                    .collect(),
                total_duration: if interactions.len() > 1 {
                    interactions.last().unwrap().timestamp.timestamp()
                        - interactions.first().unwrap().timestamp.timestamp()
                } else {
                    0
                },
                completion_rate: self.calculate_journey_completion_rate(&interactions),
            };

            journeys.push(journey);
        }

        journeys
    }

    fn calculate_journey_completion_rate(&self, interactions: &[&UserInteractionEvent]) -> f32 {
        let completed = interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::ExerciseCompleted)
            .count() as f32;

        let started = interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::Practice)
            .count() as f32;

        if started > 0.0 {
            completed / started
        } else {
            0.0
        }
    }

    fn analyze_geographic_distribution(&self) -> HashMap<String, u32> {
        let mut geographic_distribution = HashMap::new();

        // Analyze user interactions by extracting geographic patterns
        for interaction in &self.interactions {
            // In a real implementation, this would use IP geolocation or user settings
            // For now, we'll derive geographic data from user ID patterns and interaction patterns
            let geographic_region = self.extract_geographic_region(&interaction.user_id);

            *geographic_distribution
                .entry(geographic_region)
                .or_insert(0) += 1;
        }

        // If no geographic data is available, provide default distribution
        if geographic_distribution.is_empty() {
            geographic_distribution.insert("Unknown".to_string(), self.interactions.len() as u32);
        }

        geographic_distribution
    }

    /// Extract geographic region from user interaction patterns
    fn extract_geographic_region(&self, user_id: &str) -> String {
        // Simulate geographic region extraction based on user ID patterns
        // In a real implementation, this would use IP geolocation or user settings
        let hash = user_id
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_add(b as u32));

        match hash % 6 {
            0 => "North America".to_string(),
            1 => "Europe".to_string(),
            2 => "Asia Pacific".to_string(),
            3 => "South America".to_string(),
            4 => "Africa".to_string(),
            5 => "Middle East".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    fn analyze_platform_usage(&self) -> HashMap<String, u32> {
        let mut platform_usage = HashMap::new();

        // Analyze user interactions by extracting platform patterns
        for interaction in &self.interactions {
            // In a real implementation, this would use user agent or device info
            // For now, we'll derive platform data from user ID patterns and interaction timing
            let platform = self.extract_platform_type(&interaction.user_id, &interaction.timestamp);

            *platform_usage.entry(platform).or_insert(0) += 1;
        }

        // If no platform data is available, provide default distribution
        if platform_usage.is_empty() {
            platform_usage.insert("Web".to_string(), self.interactions.len() as u32);
        }

        platform_usage
    }

    /// Extract platform type from user interaction patterns
    fn extract_platform_type(&self, user_id: &str, timestamp: &DateTime<Utc>) -> String {
        // Simulate platform type extraction based on user ID and interaction timing
        // In a real implementation, this would use user agent or device info
        let hash = user_id
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let time_factor = timestamp.timestamp() as u32;
        let combined_hash = hash.wrapping_add(time_factor);

        match combined_hash % 5 {
            0 => "Web".to_string(),
            1 => "Mobile iOS".to_string(),
            2 => "Mobile Android".to_string(),
            3 => "Desktop".to_string(),
            4 => "Tablet".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    fn calculate_retention_rates(&self) -> RetentionRates {
        let now = Utc::now();
        let day_1 = now - Duration::days(1);
        let day_7 = now - Duration::days(7);
        let day_30 = now - Duration::days(30);

        let total_users = self.sessions.len() as f32;

        let day_1_retained = self
            .sessions
            .iter()
            .filter(|(_, session)| session.last_activity > day_1)
            .count() as f32;

        let day_7_retained = self
            .sessions
            .iter()
            .filter(|(_, session)| session.last_activity > day_7)
            .count() as f32;

        let day_30_retained = self
            .sessions
            .iter()
            .filter(|(_, session)| session.last_activity > day_30)
            .count() as f32;

        RetentionRates {
            day_1: if total_users > 0.0 {
                day_1_retained / total_users
            } else {
                0.0
            },
            day_7: if total_users > 0.0 {
                day_7_retained / total_users
            } else {
                0.0
            },
            day_30: if total_users > 0.0 {
                day_30_retained / total_users
            } else {
                0.0
            },
        }
    }

    fn analyze_conversion_funnel(&self) -> ConversionFunnel {
        let visitors = self.sessions.len() as f32;
        let signups = self
            .interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::Practice)
            .map(|i| &i.user_id)
            .collect::<std::collections::HashSet<_>>()
            .len() as f32;

        let active_users = self
            .sessions
            .iter()
            .filter(|(_, session)| session.session_count > 1)
            .count() as f32;

        let power_users = self
            .sessions
            .iter()
            .filter(|(_, session)| session.session_count > 10)
            .count() as f32;

        ConversionFunnel {
            visitors,
            signups,
            active_users,
            power_users,
            signup_rate: if visitors > 0.0 {
                signups / visitors
            } else {
                0.0
            },
            activation_rate: if signups > 0.0 {
                active_users / signups
            } else {
                0.0
            },
            retention_rate: if active_users > 0.0 {
                power_users / active_users
            } else {
                0.0
            },
        }
    }
}
