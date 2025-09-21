use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub id: String,
    pub title: String,
    pub category: PracticeCategory,
    pub difficulty_level: DifficultyLevel,
    pub description: String,
    pub context: String,
    pub problem_statement: String,
    pub solution: String,
    pub code_example: Option<String>,
    pub benefits: Vec<String>,
    pub trade_offs: Vec<String>,
    pub when_to_use: Vec<String>,
    pub when_not_to_use: Vec<String>,
    pub related_practices: Vec<String>,
    pub performance_impact: PerformanceImpact,
    pub security_considerations: Vec<String>,
    pub testing_recommendations: Vec<String>,
    pub monitoring_suggestions: Vec<String>,
    pub common_mistakes: Vec<String>,
    pub troubleshooting_tips: Vec<String>,
    pub references: Vec<String>,
    pub examples: Vec<String>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub verified: bool,
    pub community_rating: f32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PracticeCategory {
    Architecture,
    Performance,
    Security,
    Integration,
    Testing,
    Monitoring,
    Deployment,
    CodeQuality,
    ErrorHandling,
    Configuration,
    Scaling,
    Optimization,
    Maintenance,
    DataManagement,
    UserExperience,
    Accessibility,
    Compliance,
    Documentation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    pub cpu_impact: ImpactLevel,
    pub memory_impact: ImpactLevel,
    pub latency_impact: ImpactLevel,
    pub throughput_impact: ImpactLevel,
    pub scalability_impact: ImpactLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImpactLevel {
    Negligible,
    Low,
    Medium,
    High,
    Critical,
}

pub struct BestPracticesGuide {
    pub practices: HashMap<String, BestPractice>,
    pub category_index: HashMap<PracticeCategory, Vec<String>>,
    pub difficulty_index: HashMap<DifficultyLevel, Vec<String>>,
    pub tag_index: HashMap<String, Vec<String>>,
    pub featured_practices: Vec<String>,
    pub community_favorites: Vec<String>,
}

impl BestPracticesGuide {
    pub fn new() -> Self {
        Self {
            practices: HashMap::new(),
            category_index: HashMap::new(),
            difficulty_index: HashMap::new(),
            tag_index: HashMap::new(),
            featured_practices: Vec::new(),
            community_favorites: Vec::new(),
        }
    }

    pub fn add_practice(&mut self, practice: BestPractice) {
        let id = practice.id.clone();

        self.category_index
            .entry(practice.category.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.difficulty_index
            .entry(practice.difficulty_level.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        for tag in &practice.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(id.clone());
        }

        if practice.community_rating >= 4.5 {
            self.community_favorites.push(id.clone());
        }

        self.practices.insert(id, practice);
    }

    pub fn get_by_category(&self, category: &PracticeCategory) -> Vec<&BestPractice> {
        if let Some(ids) = self.category_index.get(category) {
            ids.iter().filter_map(|id| self.practices.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_by_difficulty(&self, difficulty: &DifficultyLevel) -> Vec<&BestPractice> {
        if let Some(ids) = self.difficulty_index.get(difficulty) {
            ids.iter().filter_map(|id| self.practices.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn search_by_tag(&self, tag: &str) -> Vec<&BestPractice> {
        if let Some(ids) = self.tag_index.get(tag) {
            ids.iter().filter_map(|id| self.practices.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn search(&self, query: &str) -> Vec<&BestPractice> {
        let query = query.to_lowercase();
        self.practices
            .values()
            .filter(|practice| {
                practice.title.to_lowercase().contains(&query)
                    || practice.description.to_lowercase().contains(&query)
                    || practice.problem_statement.to_lowercase().contains(&query)
                    || practice.solution.to_lowercase().contains(&query)
                    || practice
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect()
    }

    pub fn get_community_favorites(&self) -> Vec<&BestPractice> {
        self.community_favorites
            .iter()
            .filter_map(|id| self.practices.get(id))
            .collect()
    }

    pub fn get_related_practices(&self, practice_id: &str) -> Vec<&BestPractice> {
        if let Some(practice) = self.practices.get(practice_id) {
            practice
                .related_practices
                .iter()
                .filter_map(|id| self.practices.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }
}

fn create_essential_best_practices() -> Vec<BestPractice> {
    vec![
        BestPractice {
            id: "async_synthesis_patterns".to_string(),
            title: "Asynchronous Synthesis Patterns".to_string(),
            category: PracticeCategory::Performance,
            difficulty_level: DifficultyLevel::Intermediate,
            description: "Implement asynchronous synthesis patterns to prevent blocking operations and improve application responsiveness".to_string(),
            context: "In production applications, synthesis operations can take significant time and should not block the main thread or user interface".to_string(),
            problem_statement: "Synchronous synthesis calls can freeze user interfaces and reduce application throughput".to_string(),
            solution: "Use async/await patterns with proper task scheduling and progress reporting".to_string(),
            code_example: Some(r#"
use tokio::time::{timeout, Duration};
use voirs::prelude::*;

pub struct AsyncSynthesizer {
    synthesizer: VoirsSynthesizer,
    task_queue: tokio::sync::mpsc::Sender<SynthesisTask>,
}

impl AsyncSynthesizer {
    pub async fn synthesize_async(&self, text: &str) -> Result<AudioBuffer, VoirsError> {
        // Set timeout to prevent hanging
        let synthesis_future = self.synthesizer.synthesize(text);
        
        match timeout(Duration::from_secs(30), synthesis_future).await {
            Ok(result) => result,
            Err(_) => Err(VoirsError::Timeout("Synthesis timed out after 30 seconds".to_string())),
        }
    }

    pub async fn batch_synthesize(&self, texts: Vec<String>) -> Vec<Result<AudioBuffer, VoirsError>> {
        // Process in parallel with controlled concurrency
        let semaphore = tokio::sync::Semaphore::new(4); // Limit to 4 concurrent tasks
        let tasks: Vec<_> = texts.into_iter().map(|text| {
            let synthesizer = self.synthesizer.clone();
            let permit = semaphore.clone();
            
            tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                synthesizer.synthesize(&text).await
            })
        }).collect();
        
        // Wait for all tasks to complete
        let mut results = Vec::new();
        for task in tasks {
            results.push(task.await.unwrap_or_else(|_| Err(VoirsError::TaskCancelled)));
        }
        
        results
    }
}
"#.to_string()),
            benefits: vec![
                "Improved application responsiveness".to_string(),
                "Better resource utilization".to_string(),
                "Scalable concurrent processing".to_string(),
                "Timeout protection against hanging operations".to_string(),
            ],
            trade_offs: vec![
                "Increased code complexity".to_string(),
                "Memory overhead for task management".to_string(),
                "Need for error handling across async boundaries".to_string(),
            ],
            when_to_use: vec![
                "UI applications requiring responsiveness".to_string(),
                "High-throughput server applications".to_string(),
                "Batch processing scenarios".to_string(),
                "Long-running synthesis operations".to_string(),
            ],
            when_not_to_use: vec![
                "Simple command-line tools".to_string(),
                "Single-shot synthesis scripts".to_string(),
                "When latency is more important than throughput".to_string(),
            ],
            related_practices: vec!["error_handling_strategies".to_string(), "resource_management".to_string()],
            performance_impact: PerformanceImpact {
                cpu_impact: ImpactLevel::Low,
                memory_impact: ImpactLevel::Medium,
                latency_impact: ImpactLevel::Low,
                throughput_impact: ImpactLevel::High,
                scalability_impact: ImpactLevel::High,
            },
            security_considerations: vec![
                "Validate input before queuing tasks".to_string(),
                "Implement rate limiting for async operations".to_string(),
                "Monitor for resource exhaustion attacks".to_string(),
            ],
            testing_recommendations: vec![
                "Test with various concurrency levels".to_string(),
                "Verify timeout behavior".to_string(),
                "Test error propagation across async boundaries".to_string(),
                "Load test with realistic workloads".to_string(),
            ],
            monitoring_suggestions: vec![
                "Track task queue depth".to_string(),
                "Monitor async task completion rates".to_string(),
                "Alert on timeout frequency".to_string(),
                "Measure end-to-end synthesis latency".to_string(),
            ],
            common_mistakes: vec![
                "Not setting timeouts for synthesis operations".to_string(),
                "Creating too many concurrent tasks".to_string(),
                "Not handling task cancellation properly".to_string(),
                "Blocking async context with synchronous operations".to_string(),
            ],
            troubleshooting_tips: vec![
                "Use tokio-console for async debugging".to_string(),
                "Monitor memory usage under high concurrency".to_string(),
                "Check for deadlocks in complex async patterns".to_string(),
                "Verify proper cleanup of cancelled tasks".to_string(),
            ],
            references: vec![
                "Tokio async book: https://tokio.rs/tokio/tutorial".to_string(),
                "Rust async patterns guide".to_string(),
            ],
            examples: vec!["batch_synthesis.rs".to_string(), "streaming_synthesis.rs".to_string()],
            last_updated: chrono::Utc::now(),
            verified: true,
            community_rating: 4.7,
            tags: vec!["async".to_string(), "performance".to_string(), "concurrency".to_string()],
        },

        BestPractice {
            id: "voice_caching_strategies".to_string(),
            title: "Voice Model Caching Strategies".to_string(),
            category: PracticeCategory::Performance,
            difficulty_level: DifficultyLevel::Advanced,
            description: "Implement efficient caching strategies for voice models to reduce latency and improve performance".to_string(),
            context: "Voice models can be large and expensive to load, making caching essential for production performance".to_string(),
            problem_statement: "Loading voice models on every synthesis request leads to high latency and poor user experience".to_string(),
            solution: "Implement multi-level caching with intelligent eviction policies and preloading strategies".to_string(),
            code_example: Some(r#"
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::time::{Duration, Instant};

pub struct VoiceModelCache {
    models: Arc<RwLock<HashMap<String, CachedModel>>>,
    max_memory_mb: usize,
    ttl: Duration,
    preload_popular: bool,
}

#[derive(Clone)]
struct CachedModel {
    model: VoiceModel,
    last_accessed: Instant,
    access_count: u32,
    memory_size_mb: usize,
}

impl VoiceModelCache {
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            max_memory_mb,
            ttl: Duration::from_hours(1),
            preload_popular: true,
        }
    }

    pub async fn get_model(&self, voice_id: &str) -> Result<VoiceModel, VoirsError> {
        // Try cache first
        if let Some(model) = self.get_from_cache(voice_id) {
            return Ok(model);
        }

        // Load model if not cached
        let model = self.load_model(voice_id).await?;
        self.cache_model(voice_id, model.clone()).await?;
        
        Ok(model)
    }

    fn get_from_cache(&self, voice_id: &str) -> Option<VoiceModel> {
        let mut models = self.models.write().unwrap();
        
        if let Some(cached) = models.get_mut(voice_id) {
            // Check TTL
            if cached.last_accessed.elapsed() < self.ttl {
                cached.last_accessed = Instant::now();
                cached.access_count += 1;
                return Some(cached.model.clone());
            } else {
                // Remove expired model
                models.remove(voice_id);
            }
        }
        
        None
    }

    async fn cache_model(&self, voice_id: &str, model: VoiceModel) -> Result<(), VoirsError> {
        let model_size = model.memory_size_mb();
        
        // Check if we need to evict models
        self.ensure_memory_capacity(model_size).await?;
        
        let cached_model = CachedModel {
            model,
            last_accessed: Instant::now(),
            access_count: 1,
            memory_size_mb: model_size,
        };
        
        let mut models = self.models.write().unwrap();
        models.insert(voice_id.to_string(), cached_model);
        
        Ok(())
    }

    async fn ensure_memory_capacity(&self, required_mb: usize) -> Result<(), VoirsError> {
        let mut models = self.models.write().unwrap();
        let current_usage: usize = models.values().map(|m| m.memory_size_mb).sum();
        
        if current_usage + required_mb <= self.max_memory_mb {
            return Ok(());
        }

        // Evict least recently used models
        let mut model_list: Vec<_> = models.iter().collect();
        model_list.sort_by_key(|(_, model)| model.last_accessed);
        
        let mut freed_memory = 0;
        let mut to_remove = Vec::new();
        
        for (voice_id, model) in model_list {
            if freed_memory >= required_mb {
                break;
            }
            freed_memory += model.memory_size_mb;
            to_remove.push(voice_id.clone());
        }
        
        for voice_id in to_remove {
            models.remove(&voice_id);
        }
        
        Ok(())
    }

    pub async fn preload_popular_models(&self) -> Result<(), VoirsError> {
        if !self.preload_popular {
            return Ok();
        }

        let popular_voices = vec!["default", "premium", "natural"];
        
        for voice_id in popular_voices {
            if self.get_from_cache(voice_id).is_none() {
                let _ = self.get_model(voice_id).await;
            }
        }
        
        Ok(())
    }
}
"#.to_string()),
            benefits: vec![
                "Dramatically reduced synthesis latency".to_string(),
                "Lower memory pressure through intelligent eviction".to_string(),
                "Improved user experience with preloading".to_string(),
                "Better resource utilization".to_string(),
            ],
            trade_offs: vec![
                "Increased memory usage".to_string(),
                "Code complexity for cache management".to_string(),
                "Potential for stale model issues".to_string(),
            ],
            when_to_use: vec![
                "Production applications with repeated voice usage".to_string(),
                "High-latency model loading scenarios".to_string(),
                "Memory-constrained environments".to_string(),
                "Applications with predictable voice patterns".to_string(),
            ],
            when_not_to_use: vec![
                "Single-use synthesis scenarios".to_string(),
                "When model sizes are very small".to_string(),
                "Development and testing environments".to_string(),
            ],
            related_practices: vec!["async_synthesis_patterns".to_string(), "resource_management".to_string()],
            performance_impact: PerformanceImpact {
                cpu_impact: ImpactLevel::Medium,
                memory_impact: ImpactLevel::High,
                latency_impact: ImpactLevel::Critical,
                throughput_impact: ImpactLevel::High,
                scalability_impact: ImpactLevel::High,
            },
            security_considerations: vec![
                "Sanitize voice IDs used as cache keys".to_string(),
                "Implement access controls for cached models".to_string(),
                "Monitor for cache poisoning attempts".to_string(),
            ],
            testing_recommendations: vec![
                "Test cache hit/miss scenarios".to_string(),
                "Verify eviction policies under memory pressure".to_string(),
                "Test TTL expiration behavior".to_string(),
                "Load test with realistic access patterns".to_string(),
            ],
            monitoring_suggestions: vec![
                "Track cache hit/miss ratios".to_string(),
                "Monitor memory usage by cache".to_string(),
                "Alert on cache eviction frequency".to_string(),
                "Measure model loading times".to_string(),
            ],
            common_mistakes: vec![
                "Not considering model update scenarios".to_string(),
                "Ignoring memory constraints".to_string(),
                "Using inappropriate eviction policies".to_string(),
                "Not handling concurrent access properly".to_string(),
            ],
            troubleshooting_tips: vec![
                "Monitor cache statistics for optimization".to_string(),
                "Use memory profilers to verify cache behavior".to_string(),
                "Check for memory leaks in cache implementation".to_string(),
                "Verify thread safety under concurrent access".to_string(),
            ],
            references: vec![
                "Cache replacement algorithms overview".to_string(),
                "Rust memory management best practices".to_string(),
            ],
            examples: vec!["production_pipeline_example.rs".to_string(), "performance_benchmarking.rs".to_string()],
            last_updated: chrono::Utc::now(),
            verified: true,
            community_rating: 4.8,
            tags: vec!["caching".to_string(), "performance".to_string(), "memory".to_string()],
        },

        BestPractice {
            id: "error_handling_strategies".to_string(),
            title: "Comprehensive Error Handling Strategies".to_string(),
            category: PracticeCategory::ErrorHandling,
            difficulty_level: DifficultyLevel::Intermediate,
            description: "Implement robust error handling patterns that provide meaningful feedback and graceful degradation".to_string(),
            context: "Voice synthesis can fail for various reasons including network issues, model problems, and invalid input".to_string(),
            problem_statement: "Poor error handling leads to application crashes and poor user experience".to_string(),
            solution: "Implement structured error types with recovery strategies and user-friendly error messages".to_string(),
            code_example: Some(r#"
use thiserror::Error;
use std::fmt;

#[derive(Error, Debug)]
pub enum VoirsSynthesisError {
    #[error("Invalid input text: {message}")]
    InvalidInput { message: String },
    
    #[error("Model loading failed: {model_id} - {reason}")]
    ModelLoadError { model_id: String, reason: String },
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("Synthesis timeout after {seconds} seconds")]
    TimeoutError { seconds: u64 },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { field: String, message: String },
}

pub struct ErrorHandler {
    retry_strategy: RetryStrategy,
    fallback_voice: Option<String>,
    user_friendly_messages: bool,
}

impl ErrorHandler {
    pub async fn handle_synthesis_error(
        &self, 
        error: VoirsSynthesisError,
        context: &SynthesisContext
    ) -> Result<RecoveryAction, VoirsSynthesisError> {
        match error {
            VoirsSynthesisError::NetworkError(_) => {
                if self.retry_strategy.should_retry(&error, context.attempt_count) {
                    Ok(RecoveryAction::Retry { 
                        delay: self.retry_strategy.next_delay(context.attempt_count) 
                    })
                } else {
                    Ok(RecoveryAction::UseFallback { 
                        voice_id: self.fallback_voice.clone() 
                    })
                }
            },
            
            VoirsSynthesisError::ModelLoadError { model_id, .. } => {
                if let Some(fallback) = &self.fallback_voice {
                    Ok(RecoveryAction::UseFallback { 
                        voice_id: Some(fallback.clone()) 
                    })
                } else {
                    Err(error)
                }
            },
            
            VoirsSynthesisError::InvalidInput { .. } => {
                Ok(RecoveryAction::SanitizeInput)
            },
            
            VoirsSynthesisError::TimeoutError { .. } => {
                Ok(RecoveryAction::ReduceQuality)
            },
            
            _ => Err(error),
        }
    }

    pub fn to_user_message(&self, error: &VoirsSynthesisError) -> String {
        if !self.user_friendly_messages {
            return error.to_string();
        }

        match error {
            VoirsSynthesisError::InvalidInput { .. } => {
                "Please check your text input and try again.".to_string()
            },
            VoirsSynthesisError::NetworkError(_) => {
                "Connection issue detected. Please check your internet connection.".to_string()
            },
            VoirsSynthesisError::ModelLoadError { .. } => {
                "Voice model temporarily unavailable. Using default voice.".to_string()
            },
            VoirsSynthesisError::TimeoutError { .. } => {
                "Processing is taking longer than expected. Trying with faster settings.".to_string()
            },
            VoirsSynthesisError::ResourceExhausted { .. } => {
                "System is currently busy. Please try again in a moment.".to_string()
            },
            _ => "An unexpected error occurred. Please try again.".to_string(),
        }
    }
}

pub enum RecoveryAction {
    Retry { delay: std::time::Duration },
    UseFallback { voice_id: Option<String> },
    SanitizeInput,
    ReduceQuality,
    Fail,
}

pub struct RetryStrategy {
    max_retries: u32,
    base_delay: std::time::Duration,
    exponential_backoff: bool,
}

impl RetryStrategy {
    pub fn should_retry(&self, error: &VoirsSynthesisError, attempt: u32) -> bool {
        if attempt >= self.max_retries {
            return false;
        }

        matches!(error, 
            VoirsSynthesisError::NetworkError(_) |
            VoirsSynthesisError::TimeoutError { .. } |
            VoirsSynthesisError::ResourceExhausted { .. }
        )
    }

    pub fn next_delay(&self, attempt: u32) -> std::time::Duration {
        if self.exponential_backoff {
            self.base_delay * 2_u32.pow(attempt)
        } else {
            self.base_delay
        }
    }
}
"#.to_string()),
            benefits: vec![
                "Improved application reliability".to_string(),
                "Better user experience with meaningful messages".to_string(),
                "Graceful degradation instead of crashes".to_string(),
                "Easier debugging and maintenance".to_string(),
            ],
            trade_offs: vec![
                "Increased code complexity".to_string(),
                "Additional error handling overhead".to_string(),
                "Need for comprehensive error testing".to_string(),
            ],
            when_to_use: vec![
                "Production applications".to_string(),
                "User-facing applications".to_string(),
                "Systems requiring high reliability".to_string(),
                "Applications with network dependencies".to_string(),
            ],
            when_not_to_use: vec![
                "Simple internal scripts".to_string(),
                "Prototype development".to_string(),
                "When fail-fast behavior is preferred".to_string(),
            ],
            related_practices: vec!["async_synthesis_patterns".to_string(), "resource_management".to_string()],
            performance_impact: PerformanceImpact {
                cpu_impact: ImpactLevel::Low,
                memory_impact: ImpactLevel::Low,
                latency_impact: ImpactLevel::Medium,
                throughput_impact: ImpactLevel::Medium,
                scalability_impact: ImpactLevel::High,
            },
            security_considerations: vec![
                "Don't expose internal error details to users".to_string(),
                "Log security-relevant errors appropriately".to_string(),
                "Prevent error-based information disclosure".to_string(),
            ],
            testing_recommendations: vec![
                "Test all error paths and recovery strategies".to_string(),
                "Verify error message quality and clarity".to_string(),
                "Test retry logic with various failure scenarios".to_string(),
                "Validate fallback behavior".to_string(),
            ],
            monitoring_suggestions: vec![
                "Track error rates by type".to_string(),
                "Monitor retry success rates".to_string(),
                "Alert on unusual error patterns".to_string(),
                "Measure fallback usage frequency".to_string(),
            ],
            common_mistakes: vec![
                "Exposing internal error details to users".to_string(),
                "Not implementing appropriate retry strategies".to_string(),
                "Ignoring error context information".to_string(),
                "Failing to test error scenarios".to_string(),
            ],
            troubleshooting_tips: vec![
                "Use structured logging for error analysis".to_string(),
                "Implement error correlation IDs".to_string(),
                "Test error scenarios in isolation".to_string(),
                "Monitor error patterns for system issues".to_string(),
            ],
            references: vec![
                "Rust error handling guide".to_string(),
                "thiserror crate documentation".to_string(),
            ],
            examples: vec!["robust_error_handling_patterns.rs".to_string(), "production_pipeline_example.rs".to_string()],
            last_updated: chrono::Utc::now(),
            verified: true,
            community_rating: 4.9,
            tags: vec!["error-handling".to_string(), "reliability".to_string(), "user-experience".to_string()],
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 VoiRS Best Practices Guide");
    println!("============================");

    let mut guide = BestPracticesGuide::new();

    println!("\n📋 Loading essential best practices...");
    let essential_practices = create_essential_best_practices();
    for practice in essential_practices {
        guide.add_practice(practice);
    }

    println!("✅ Loaded {} best practices", guide.practices.len());

    println!("\n🏆 Community Favorites (Rating ≥ 4.5):");
    let favorites = guide.get_community_favorites();
    for practice in favorites {
        println!(
            "   ⭐ {} ({:.1}/5.0)",
            practice.title, practice.community_rating
        );
        println!("      📂 Category: {:?}", practice.category);
        println!("      🎯 Level: {:?}", practice.difficulty_level);
        println!("      💡 {}", practice.description);
        println!();
    }

    println!("\n🚀 Performance Best Practices:");
    let performance_practices = guide.get_by_category(&PracticeCategory::Performance);
    for practice in performance_practices {
        println!(
            "   ⚡ {} ({:?} level)",
            practice.title, practice.difficulty_level
        );
        println!("      🎯 Problem: {}", practice.problem_statement);
        println!("      ✅ Solution: {}", practice.solution);

        println!("      📊 Performance Impact:");
        println!(
            "         CPU: {:?}, Memory: {:?}, Latency: {:?}",
            practice.performance_impact.cpu_impact,
            practice.performance_impact.memory_impact,
            practice.performance_impact.latency_impact
        );

        if !practice.benefits.is_empty() {
            println!("      💰 Benefits:");
            for benefit in &practice.benefits {
                println!("         • {}", benefit);
            }
        }

        if !practice.common_mistakes.is_empty() {
            println!("      ⚠️  Common Mistakes:");
            for mistake in practice.common_mistakes.iter().take(2) {
                println!("         • {}", mistake);
            }
        }
        println!();
    }

    println!("\n🛡️ Error Handling Best Practices:");
    let error_practices = guide.get_by_category(&PracticeCategory::ErrorHandling);
    for practice in error_practices {
        println!(
            "   🔧 {} ({:?} level)",
            practice.title, practice.difficulty_level
        );
        println!("      📝 Context: {}", practice.context);

        if !practice.when_to_use.is_empty() {
            println!("      ✅ When to use:");
            for usage in &practice.when_to_use {
                println!("         • {}", usage);
            }
        }

        if !practice.security_considerations.is_empty() {
            println!("      🔒 Security considerations:");
            for consideration in &practice.security_considerations {
                println!("         • {}", consideration);
            }
        }
        println!();
    }

    println!("\n🔍 Search Example: 'async'");
    let async_practices = guide.search("async");
    for practice in async_practices {
        println!("   🔎 {} - {:?}", practice.title, practice.category);
        if let Some(code) = &practice.code_example {
            let lines: Vec<&str> = code.lines().take(5).collect();
            println!("      💻 Code preview:");
            for line in lines {
                if !line.trim().is_empty() {
                    println!("         {}", line);
                }
            }
            println!("         ... (truncated)");
        }
        println!();
    }

    println!("\n🏷️ Tag-based Search: 'performance'");
    let performance_tagged = guide.search_by_tag("performance");
    for practice in performance_tagged {
        println!(
            "   🏷️  {} - Tags: {}",
            practice.title,
            practice.tags.join(", ")
        );
    }

    println!("\n📊 Practice Statistics by Category:");
    let mut category_counts: HashMap<PracticeCategory, usize> = HashMap::new();
    for practice in guide.practices.values() {
        *category_counts
            .entry(practice.category.clone())
            .or_insert(0) += 1;
    }

    for (category, count) in category_counts {
        println!("   📂 {:?}: {}", category, count);
    }

    println!("\n📊 Practice Statistics by Difficulty:");
    let mut difficulty_counts: HashMap<DifficultyLevel, usize> = HashMap::new();
    for practice in guide.practices.values() {
        *difficulty_counts
            .entry(practice.difficulty_level.clone())
            .or_insert(0) += 1;
    }

    let difficulty_order = vec![
        DifficultyLevel::Beginner,
        DifficultyLevel::Intermediate,
        DifficultyLevel::Advanced,
        DifficultyLevel::Expert,
    ];

    for difficulty in difficulty_order {
        if let Some(count) = difficulty_counts.get(&difficulty) {
            println!("   🎯 {:?}: {}", difficulty, count);
        }
    }

    println!("\n💾 Exporting best practices guide...");
    let export_data = serde_json::to_string_pretty(&guide.practices)?;
    let export_path = Path::new("/tmp/best_practices_guide_export.json");
    fs::write(export_path, export_data).await?;
    println!(
        "✅ Best practices guide exported to: {}",
        export_path.display()
    );

    println!("\n🎉 Best Practices Guide Demo Complete!");
    println!("\nKey Takeaways:");
    println!("• Async patterns are essential for production performance");
    println!("• Caching strategies dramatically improve latency");
    println!("• Comprehensive error handling improves reliability");
    println!("• Always consider security implications");
    println!("• Monitor and test all implementations thoroughly");
    println!("\nFor complete implementation details, refer to the code examples!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_practices_guide_creation() {
        let guide = BestPracticesGuide::new();
        assert_eq!(guide.practices.len(), 0);
        assert_eq!(guide.community_favorites.len(), 0);
    }

    #[test]
    fn test_practice_categorization() {
        let mut guide = BestPracticesGuide::new();
        let practices = create_essential_best_practices();

        for practice in practices {
            guide.add_practice(practice);
        }

        let performance_practices = guide.get_by_category(&PracticeCategory::Performance);
        assert!(!performance_practices.is_empty());

        let error_practices = guide.get_by_category(&PracticeCategory::ErrorHandling);
        assert!(!error_practices.is_empty());
    }

    #[test]
    fn test_community_favorites() {
        let mut guide = BestPracticesGuide::new();
        let practices = create_essential_best_practices();

        for practice in practices {
            guide.add_practice(practice);
        }

        let favorites = guide.get_community_favorites();
        assert!(!favorites.is_empty());

        for favorite in favorites {
            assert!(favorite.community_rating >= 4.5);
        }
    }

    #[test]
    fn test_search_functionality() {
        let mut guide = BestPracticesGuide::new();
        let practices = create_essential_best_practices();

        for practice in practices {
            guide.add_practice(practice);
        }

        let async_results = guide.search("async");
        assert!(!async_results.is_empty());

        let performance_results = guide.search_by_tag("performance");
        assert!(!performance_results.is_empty());
    }

    #[test]
    fn test_difficulty_filtering() {
        let mut guide = BestPracticesGuide::new();
        let practices = create_essential_best_practices();

        for practice in practices {
            guide.add_practice(practice);
        }

        let intermediate_practices = guide.get_by_difficulty(&DifficultyLevel::Intermediate);
        assert!(!intermediate_practices.is_empty());

        for practice in intermediate_practices {
            assert!(matches!(
                practice.difficulty_level,
                DifficultyLevel::Intermediate
            ));
        }
    }
}
