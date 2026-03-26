//! Batch scheduling strategies for optimal resource utilization.

use super::*;

/// Scheduling strategy for batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// First-in-first-out (simple queue)
    FIFO,

    /// Priority-based scheduling
    PriorityBased,

    /// Load-balanced distribution across workers
    LoadBalanced,

    /// Shortest-job-first scheduling
    ShortestFirst,

    /// Adaptive scheduling based on system load
    Adaptive,
}

impl Default for SchedulingStrategy {
    fn default() -> Self {
        Self::LoadBalanced
    }
}

/// Batch scheduler for organizing and distributing synthesis requests.
pub struct BatchScheduler {
    strategy: SchedulingStrategy,
    processed_count: usize,
    total_processing_time: Duration,
}

impl BatchScheduler {
    /// Create a new batch scheduler with the specified strategy.
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            processed_count: 0,
            total_processing_time: Duration::from_secs(0),
        }
    }

    /// Schedule a batch of requests according to the strategy.
    ///
    /// Returns a vector of batches, where each batch should be processed concurrently.
    pub fn schedule(
        &mut self,
        mut requests: Vec<BatchRequest>,
        config: &BatchConfig,
    ) -> VoirsResult<Vec<Vec<BatchRequest>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Apply scheduling strategy
        match self.strategy {
            SchedulingStrategy::FIFO => {
                // No reordering needed
            }
            SchedulingStrategy::PriorityBased => {
                // Sort by priority (descending)
                requests.sort_by_key(|b| std::cmp::Reverse(b.priority));
            }
            SchedulingStrategy::LoadBalanced => {
                // Distribute evenly across workers
                self.apply_load_balancing(&mut requests, config);
            }
            SchedulingStrategy::ShortestFirst => {
                // Sort by estimated processing time (text length as proxy)
                requests.sort_by_key(|r| r.text.len());
            }
            SchedulingStrategy::Adaptive => {
                // Combine priority and load balancing
                self.apply_adaptive_scheduling(&mut requests, config);
            }
        }

        // Split into batches based on max_batch_size
        let batches = self.split_into_batches(requests, config.max_batch_size);

        debug!(
            "Scheduled {} batches using {:?} strategy",
            batches.len(),
            self.strategy
        );

        Ok(batches)
    }

    /// Split requests into batches of maximum size.
    fn split_into_batches(
        &self,
        requests: Vec<BatchRequest>,
        max_batch_size: usize,
    ) -> Vec<Vec<BatchRequest>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        for request in requests {
            current_batch.push(request);

            if current_batch.len() >= max_batch_size {
                batches.push(std::mem::take(&mut current_batch));
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    /// Apply load balancing to distribute requests evenly.
    fn apply_load_balancing(&self, requests: &mut [BatchRequest], config: &BatchConfig) {
        // Round-robin distribution with priority consideration
        requests.sort_by(|a, b| {
            // Primary: priority (descending)
            match b.priority.cmp(&a.priority) {
                std::cmp::Ordering::Equal => {
                    // Secondary: text length (for load balancing)
                    a.text.len().cmp(&b.text.len())
                }
                other => other,
            }
        });
    }

    /// Apply adaptive scheduling based on system state.
    fn apply_adaptive_scheduling(&mut self, requests: &mut [BatchRequest], config: &BatchConfig) {
        // Calculate average processing time if available
        let avg_time_ms = if self.processed_count > 0 {
            self.total_processing_time.as_millis() / self.processed_count as u128
        } else {
            100 // Default estimate
        };

        // Sort by combined score: priority + estimated time
        requests.sort_by(|a, b| {
            let score_a = self.calculate_adaptive_score(a, avg_time_ms);
            let score_b = self.calculate_adaptive_score(b, avg_time_ms);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Calculate adaptive scheduling score for a request.
    fn calculate_adaptive_score(&self, request: &BatchRequest, avg_time_ms: u128) -> f64 {
        // Estimate processing time based on text length
        let estimated_time_ms = (request.text.len() as f64 * 10.0).max(50.0);

        // Combine priority and time estimate
        let priority_weight = 0.6;
        let time_weight = 0.4;

        let priority_score = request.priority as f64;
        let time_score = 1.0 / (estimated_time_ms / avg_time_ms as f64).max(0.1);

        priority_score * priority_weight + time_score * time_weight
    }

    /// Update scheduler statistics after processing.
    pub fn update_stats(&mut self, processing_time: Duration, request_count: usize) {
        self.processed_count += request_count;
        self.total_processing_time += processing_time;
    }

    /// Get current strategy.
    pub fn strategy(&self) -> SchedulingStrategy {
        self.strategy
    }

    /// Set scheduling strategy.
    pub fn set_strategy(&mut self, strategy: SchedulingStrategy) {
        self.strategy = strategy;
    }

    /// Get processed request count.
    pub fn processed_count(&self) -> usize {
        self.processed_count
    }

    /// Get total processing time.
    pub fn total_processing_time(&self) -> Duration {
        self.total_processing_time
    }

    /// Reset scheduler statistics.
    pub fn reset_stats(&mut self) {
        self.processed_count = 0;
        self.total_processing_time = Duration::from_secs(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = BatchScheduler::new(SchedulingStrategy::FIFO);
        assert_eq!(scheduler.strategy(), SchedulingStrategy::FIFO);
        assert_eq!(scheduler.processed_count(), 0);
    }

    #[test]
    fn test_fifo_scheduling() {
        let mut scheduler = BatchScheduler::new(SchedulingStrategy::FIFO);
        let config = BatchConfig::default();

        let requests = vec![
            BatchRequest::new("First", None),
            BatchRequest::new("Second", None),
            BatchRequest::new("Third", None),
        ];

        let batches = scheduler.schedule(requests, &config).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = BatchScheduler::new(SchedulingStrategy::PriorityBased);
        let config = BatchConfig::default();

        let requests = vec![
            BatchRequest::new("Low", None).with_priority(1),
            BatchRequest::new("High", None).with_priority(10),
            BatchRequest::new("Medium", None).with_priority(5),
        ];

        let batches = scheduler.schedule(requests, &config).unwrap();
        assert!(!batches.is_empty());

        // First batch should have high priority request first
        let first_batch = &batches[0];
        assert_eq!(first_batch[0].priority, 10);
    }

    #[test]
    fn test_batch_splitting() {
        let mut scheduler = BatchScheduler::new(SchedulingStrategy::FIFO);
        let mut config = BatchConfig::default();
        config.max_batch_size = 2;

        let requests = vec![
            BatchRequest::new("1", None),
            BatchRequest::new("2", None),
            BatchRequest::new("3", None),
            BatchRequest::new("4", None),
            BatchRequest::new("5", None),
        ];

        let batches = scheduler.schedule(requests, &config).unwrap();
        assert_eq!(batches.len(), 3); // 2 + 2 + 1
    }

    #[test]
    fn test_stats_update() {
        let mut scheduler = BatchScheduler::new(SchedulingStrategy::FIFO);

        scheduler.update_stats(Duration::from_millis(500), 10);
        assert_eq!(scheduler.processed_count(), 10);
        assert!(scheduler.total_processing_time() > Duration::from_secs(0));

        scheduler.reset_stats();
        assert_eq!(scheduler.processed_count(), 0);
    }

    #[test]
    fn test_strategy_change() {
        let mut scheduler = BatchScheduler::new(SchedulingStrategy::FIFO);
        assert_eq!(scheduler.strategy(), SchedulingStrategy::FIFO);

        scheduler.set_strategy(SchedulingStrategy::PriorityBased);
        assert_eq!(scheduler.strategy(), SchedulingStrategy::PriorityBased);
    }
}
