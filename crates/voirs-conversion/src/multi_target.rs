//! Multi-target voice conversion functionality

use crate::{
    core::VoiceConverter,
    types::{
        ConversionRequest, ConversionResult, ConversionTarget, ConversionType, VoiceCharacteristics,
    },
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Multi-target conversion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTargetConversionRequest {
    /// Request ID
    pub id: String,
    /// Source audio data
    pub source_audio: Vec<f32>,
    /// Source sample rate
    pub source_sample_rate: u32,
    /// Conversion type to apply to all targets
    pub conversion_type: ConversionType,
    /// Multiple conversion targets
    pub targets: Vec<NamedTarget>,
    /// Real-time processing flag
    pub realtime: bool,
    /// Quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Processing parameters
    pub parameters: HashMap<String, f32>,
    /// Request timestamp
    pub timestamp: SystemTime,
}

impl MultiTargetConversionRequest {
    /// Create new multi-target conversion request
    pub fn new(
        id: String,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        conversion_type: ConversionType,
        targets: Vec<NamedTarget>,
    ) -> Self {
        Self {
            id,
            source_audio,
            source_sample_rate,
            conversion_type,
            targets,
            realtime: false,
            quality_level: 0.8,
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Enable real-time processing
    pub fn with_realtime(mut self, realtime: bool) -> Self {
        self.realtime = realtime;
        self
    }

    /// Set quality level
    pub fn with_quality_level(mut self, level: f32) -> Self {
        self.quality_level = level.clamp(0.0, 1.0);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Add a target
    pub fn add_target(mut self, target: NamedTarget) -> Self {
        self.targets.push(target);
        self
    }

    /// Validate the request
    pub fn validate(&self) -> Result<()> {
        if self.source_audio.is_empty() {
            return Err(Error::validation(
                "Source audio cannot be empty".to_string(),
            ));
        }

        if self.source_sample_rate == 0 {
            return Err(Error::validation(
                "Source sample rate must be positive".to_string(),
            ));
        }

        if self.targets.is_empty() {
            return Err(Error::validation(
                "At least one target must be specified".to_string(),
            ));
        }

        if self.targets.len() > 10 {
            return Err(Error::validation(
                "Maximum 10 targets supported for multi-target conversion".to_string(),
            ));
        }

        if self.realtime && !self.conversion_type.supports_realtime() {
            return Err(Error::validation(format!(
                "Conversion type {:?} does not support real-time processing",
                self.conversion_type
            )));
        }

        // Validate each target
        for (i, target) in self.targets.iter().enumerate() {
            if target.name.is_empty() {
                return Err(Error::validation(format!(
                    "Target {i} must have a non-empty name"
                )));
            }
        }

        // Check for duplicate target names
        let mut names = std::collections::HashSet::new();
        for target in &self.targets {
            if !names.insert(&target.name) {
                return Err(Error::validation(format!(
                    "Duplicate target name: {}",
                    target.name
                )));
            }
        }

        Ok(())
    }

    /// Get source duration in seconds
    pub fn source_duration(&self) -> f32 {
        self.source_audio.len() as f32 / self.source_sample_rate as f32
    }
}

/// Named conversion target for multi-target processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedTarget {
    /// Target name/identifier
    pub name: String,
    /// Conversion target specification
    pub target: ConversionTarget,
    /// Priority for processing (higher = processed first)
    pub priority: i32,
    /// Custom parameters for this specific target
    pub custom_params: HashMap<String, f32>,
}

impl NamedTarget {
    /// Create new named target
    pub fn new(name: String, target: ConversionTarget) -> Self {
        Self {
            name,
            target,
            priority: 0,
            custom_params: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, key: String, value: f32) -> Self {
        self.custom_params.insert(key, value);
        self
    }
}

/// Multi-target conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTargetConversionResult {
    /// Request ID this result corresponds to
    pub request_id: String,
    /// Results for each target (keyed by target name)
    pub target_results: HashMap<String, ConversionResult>,
    /// Overall processing time
    pub total_processing_time: Duration,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Result timestamp
    pub timestamp: SystemTime,
    /// Processing statistics
    pub stats: MultiTargetProcessingStats,
}

impl MultiTargetConversionResult {
    /// Create successful result
    pub fn success(
        request_id: String,
        target_results: HashMap<String, ConversionResult>,
        total_processing_time: Duration,
        stats: MultiTargetProcessingStats,
    ) -> Self {
        Self {
            request_id,
            target_results,
            total_processing_time,
            success: true,
            error_message: None,
            timestamp: SystemTime::now(),
            stats,
        }
    }

    /// Create failed result
    pub fn failure(request_id: String, error_message: String) -> Self {
        Self {
            request_id,
            target_results: HashMap::new(),
            total_processing_time: Duration::from_millis(0),
            success: false,
            error_message: Some(error_message),
            timestamp: SystemTime::now(),
            stats: MultiTargetProcessingStats::default(),
        }
    }

    /// Get result for specific target
    pub fn get_target_result(&self, target_name: &str) -> Option<&ConversionResult> {
        self.target_results.get(target_name)
    }

    /// Get all successful target results
    pub fn successful_results(&self) -> HashMap<String, &ConversionResult> {
        self.target_results
            .iter()
            .filter(|(_, result)| result.success)
            .map(|(name, result)| (name.clone(), result))
            .collect()
    }

    /// Get all failed target results
    pub fn failed_results(&self) -> HashMap<String, &ConversionResult> {
        self.target_results
            .iter()
            .filter(|(_, result)| !result.success)
            .map(|(name, result)| (name.clone(), result))
            .collect()
    }
}

/// Processing statistics for multi-target conversion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiTargetProcessingStats {
    /// Number of targets processed
    pub targets_processed: usize,
    /// Number of successful conversions
    pub successful_conversions: usize,
    /// Number of failed conversions
    pub failed_conversions: usize,
    /// Average processing time per target
    pub average_processing_time: Duration,
    /// Maximum processing time among targets
    pub max_processing_time: Duration,
    /// Minimum processing time among targets
    pub min_processing_time: Duration,
    /// Whether parallel processing was used
    pub parallel_processing: bool,
    /// Memory usage during processing (in bytes)
    pub peak_memory_usage: usize,
}

/// Multi-target voice converter
#[derive(Debug)]
pub struct MultiTargetConverter {
    /// Underlying voice converter
    converter: Arc<VoiceConverter>,
    /// Processing mode configuration
    processing_mode: ProcessingMode,
    /// Maximum concurrent targets
    max_concurrent_targets: usize,
    /// Processing statistics
    stats: Arc<RwLock<MultiTargetProcessingStats>>,
}

/// Processing mode for multi-target conversion
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingMode {
    /// Process targets sequentially (lower memory usage)
    Sequential,
    /// Process targets in parallel (faster but higher memory usage)
    Parallel,
    /// Automatically choose based on number of targets and system resources
    Adaptive,
}

impl MultiTargetConverter {
    /// Create new multi-target converter
    pub fn new(converter: VoiceConverter) -> Self {
        Self {
            converter: Arc::new(converter),
            processing_mode: ProcessingMode::Adaptive,
            max_concurrent_targets: 4,
            stats: Arc::new(RwLock::new(MultiTargetProcessingStats::default())),
        }
    }

    /// Create with custom processing mode
    pub fn with_processing_mode(mut self, mode: ProcessingMode) -> Self {
        self.processing_mode = mode;
        self
    }

    /// Set maximum concurrent targets for parallel processing
    pub fn with_max_concurrent_targets(mut self, max: usize) -> Self {
        self.max_concurrent_targets = max.clamp(1, 16); // Clamp between 1 and 16
        self
    }

    /// Convert audio to multiple targets
    pub async fn convert_multi_target(
        &self,
        request: MultiTargetConversionRequest,
    ) -> Result<MultiTargetConversionResult> {
        request.validate()?;

        let start_time = std::time::Instant::now();
        info!(
            "Starting multi-target conversion for request: {} with {} targets",
            request.id,
            request.targets.len()
        );

        // Determine processing mode
        let processing_mode = self.determine_processing_mode(&request);
        let use_parallel = matches!(processing_mode, ProcessingMode::Parallel);

        // Initialize statistics
        let mut stats = MultiTargetProcessingStats {
            targets_processed: request.targets.len(),
            parallel_processing: use_parallel,
            ..Default::default()
        };

        // Sort targets by priority (higher priority first)
        let mut sorted_targets = request.targets.clone();
        sorted_targets.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Process targets based on mode
        let target_results = if use_parallel {
            self.process_targets_parallel(&request, &sorted_targets)
                .await?
        } else {
            self.process_targets_sequential(&request, &sorted_targets)
                .await?
        };

        let total_processing_time = start_time.elapsed();

        // Calculate statistics
        let successful_count = target_results.values().filter(|r| r.success).count();
        let failed_count = target_results.len() - successful_count;

        let processing_times: Vec<Duration> =
            target_results.values().map(|r| r.processing_time).collect();

        stats.successful_conversions = successful_count;
        stats.failed_conversions = failed_count;
        stats.average_processing_time = if !processing_times.is_empty() {
            let total_nanos = processing_times.iter().map(|d| d.as_nanos()).sum::<u128>();
            let avg_nanos = total_nanos / processing_times.len() as u128;
            Duration::from_nanos(avg_nanos.min(u64::MAX as u128) as u64)
        } else {
            Duration::from_millis(0)
        };
        stats.max_processing_time = processing_times.iter().max().copied().unwrap_or_default();
        stats.min_processing_time = processing_times.iter().min().copied().unwrap_or_default();

        // Estimate memory usage
        stats.peak_memory_usage = self.estimate_memory_usage(&request, use_parallel);

        // Update global statistics
        {
            let mut global_stats = self.stats.write().await;
            *global_stats = stats.clone();
        }

        info!(
            "Multi-target conversion completed for request: {} in {:?} - {}/{} targets successful",
            request.id,
            total_processing_time,
            successful_count,
            request.targets.len()
        );

        Ok(MultiTargetConversionResult::success(
            request.id,
            target_results,
            total_processing_time,
            stats,
        ))
    }

    /// Determine optimal processing mode
    fn determine_processing_mode(&self, request: &MultiTargetConversionRequest) -> ProcessingMode {
        match self.processing_mode {
            ProcessingMode::Sequential => ProcessingMode::Sequential,
            ProcessingMode::Parallel => ProcessingMode::Parallel,
            ProcessingMode::Adaptive => {
                // Use heuristics to decide
                let target_count = request.targets.len();
                let audio_duration = request.source_duration();
                let is_realtime = request.realtime;

                if is_realtime || target_count <= 2 || audio_duration > 30.0 {
                    // Use sequential for real-time, few targets, or long audio
                    ProcessingMode::Sequential
                } else {
                    // Use parallel for multiple targets with shorter audio
                    ProcessingMode::Parallel
                }
            }
        }
    }

    /// Process targets in parallel
    async fn process_targets_parallel(
        &self,
        request: &MultiTargetConversionRequest,
        targets: &[NamedTarget],
    ) -> Result<HashMap<String, ConversionResult>> {
        debug!("Processing {} targets in parallel", targets.len());

        let mut target_results = HashMap::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent_targets));

        // Create futures for each target
        let mut handles = Vec::new();
        for target in targets {
            let converter = Arc::clone(&self.converter);
            let target = target.clone();
            let request = request.clone();
            let semaphore = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let conversion_request = Self::create_single_conversion_request(&request, &target);
                let result = converter.convert(conversion_request).await;
                (target.name.clone(), result)
            });

            handles.push(handle);
        }

        // Wait for all conversions to complete
        for handle in handles {
            match handle.await {
                Ok((target_name, result)) => match result {
                    Ok(conversion_result) => {
                        target_results.insert(target_name, conversion_result);
                    }
                    Err(e) => {
                        warn!("Conversion failed for target {}: {}", target_name, e);
                        // Create a failed result
                        let failed_result = ConversionResult {
                            request_id: request.id.clone(),
                            converted_audio: vec![],
                            output_sample_rate: request.source_sample_rate,
                            quality_metrics: HashMap::new(),
                            artifacts: None,
                            objective_quality: None,
                            processing_time: Duration::from_millis(0),
                            conversion_type: request.conversion_type.clone(),
                            success: false,
                            error_message: Some(e.to_string()),
                            timestamp: SystemTime::now(),
                        };
                        target_results.insert(target_name, failed_result);
                    }
                },
                Err(e) => {
                    warn!("Task failed for target: {}", e);
                }
            }
        }

        Ok(target_results)
    }

    /// Process targets sequentially
    async fn process_targets_sequential(
        &self,
        request: &MultiTargetConversionRequest,
        targets: &[NamedTarget],
    ) -> Result<HashMap<String, ConversionResult>> {
        debug!("Processing {} targets sequentially", targets.len());

        let mut target_results = HashMap::new();

        for target in targets {
            let conversion_request = Self::create_single_conversion_request(request, target);

            match self.converter.convert(conversion_request).await {
                Ok(result) => {
                    target_results.insert(target.name.clone(), result);
                }
                Err(e) => {
                    warn!("Conversion failed for target {}: {}", target.name, e);
                    // Create a failed result
                    let failed_result = ConversionResult {
                        request_id: request.id.clone(),
                        converted_audio: vec![],
                        output_sample_rate: request.source_sample_rate,
                        quality_metrics: HashMap::new(),
                        artifacts: None,
                        objective_quality: None,
                        processing_time: Duration::from_millis(0),
                        conversion_type: request.conversion_type.clone(),
                        success: false,
                        error_message: Some(e.to_string()),
                        timestamp: SystemTime::now(),
                    };
                    target_results.insert(target.name.clone(), failed_result);
                }
            }
        }

        Ok(target_results)
    }

    /// Create single conversion request from multi-target request and named target
    fn create_single_conversion_request(
        request: &MultiTargetConversionRequest,
        named_target: &NamedTarget,
    ) -> ConversionRequest {
        let mut single_request = ConversionRequest::new(
            format!("{}_{}", request.id, named_target.name),
            request.source_audio.clone(),
            request.source_sample_rate,
            request.conversion_type.clone(),
            named_target.target.clone(),
        )
        .with_realtime(request.realtime)
        .with_quality_level(request.quality_level);

        // Add global parameters
        for (key, value) in &request.parameters {
            single_request = single_request.with_parameter(key.clone(), *value);
        }

        // Add target-specific parameters (override global ones)
        for (key, value) in &named_target.custom_params {
            single_request = single_request.with_parameter(key.clone(), *value);
        }

        single_request
    }

    /// Estimate memory usage for the conversion
    fn estimate_memory_usage(
        &self,
        request: &MultiTargetConversionRequest,
        parallel: bool,
    ) -> usize {
        let audio_size = request.source_audio.len() * std::mem::size_of::<f32>();
        let base_memory = audio_size * 2; // Input + output audio

        if parallel {
            base_memory * request.targets.len() // Each target needs its own memory
        } else {
            base_memory * 2 // Sequential processing needs less memory
        }
    }

    /// Get processing statistics
    pub async fn get_stats(&self) -> MultiTargetProcessingStats {
        self.stats.read().await.clone()
    }

    /// Reset processing statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = MultiTargetProcessingStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::ConversionConfig,
        types::{AgeGroup, ConversionTarget, Gender, VoiceCharacteristics},
    };

    fn create_test_converter() -> MultiTargetConverter {
        let config = ConversionConfig::default();
        let converter = VoiceConverter::with_config(config).unwrap();
        MultiTargetConverter::new(converter)
    }

    fn create_test_audio() -> Vec<f32> {
        vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05]
    }

    #[tokio::test]
    async fn test_multi_target_conversion_request_creation() {
        let audio = create_test_audio();
        let target1 = NamedTarget::new(
            "target1".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_gender(Gender::Male)),
        );
        let target2 = NamedTarget::new(
            "target2".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_age(AgeGroup::Senior)),
        );

        let request = MultiTargetConversionRequest::new(
            "test_multi".to_string(),
            audio,
            22050,
            ConversionType::GenderTransformation,
            vec![target1, target2],
        );

        assert_eq!(request.targets.len(), 2);
        assert_eq!(
            request.conversion_type,
            ConversionType::GenderTransformation
        );
        assert!(request.validate().is_ok());
    }

    #[tokio::test]
    async fn test_multi_target_conversion_validation() {
        let audio = create_test_audio();

        // Test empty targets
        let empty_request = MultiTargetConversionRequest::new(
            "test_empty".to_string(),
            audio.clone(),
            22050,
            ConversionType::PitchShift,
            vec![],
        );
        assert!(empty_request.validate().is_err());

        // Test too many targets
        let mut many_targets = Vec::new();
        for i in 0..12 {
            many_targets.push(NamedTarget::new(
                format!("target_{}", i),
                ConversionTarget::new(VoiceCharacteristics::default()),
            ));
        }
        let many_request = MultiTargetConversionRequest::new(
            "test_many".to_string(),
            audio.clone(),
            22050,
            ConversionType::PitchShift,
            many_targets,
        );
        assert!(many_request.validate().is_err());

        // Test duplicate target names
        let target1 = NamedTarget::new(
            "same_name".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );
        let target2 = NamedTarget::new(
            "same_name".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );
        let duplicate_request = MultiTargetConversionRequest::new(
            "test_duplicate".to_string(),
            audio,
            22050,
            ConversionType::PitchShift,
            vec![target1, target2],
        );
        assert!(duplicate_request.validate().is_err());
    }

    #[tokio::test]
    async fn test_named_target_creation() {
        let characteristics = VoiceCharacteristics::for_gender(Gender::Female);
        let target = ConversionTarget::new(characteristics);

        let named_target = NamedTarget::new("female_voice".to_string(), target)
            .with_priority(5)
            .with_custom_param("strength".to_string(), 0.8);

        assert_eq!(named_target.name, "female_voice");
        assert_eq!(named_target.priority, 5);
        assert_eq!(named_target.custom_params.get("strength"), Some(&0.8));
    }

    #[tokio::test]
    async fn test_multi_target_converter_sequential() {
        let converter = create_test_converter().with_processing_mode(ProcessingMode::Sequential);

        let audio = create_test_audio();
        let target1 = NamedTarget::new(
            "pitch_high".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_gender(Gender::Female)),
        );
        let target2 = NamedTarget::new(
            "pitch_low".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_gender(Gender::Male)),
        );

        let request = MultiTargetConversionRequest::new(
            "test_sequential".to_string(),
            audio,
            22050,
            ConversionType::GenderTransformation,
            vec![target1, target2],
        );

        let result = converter.convert_multi_target(request).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.target_results.len(), 2);
        assert!(result.target_results.contains_key("pitch_high"));
        assert!(result.target_results.contains_key("pitch_low"));
        assert!(!result.stats.parallel_processing);
    }

    #[tokio::test]
    async fn test_multi_target_converter_parallel() {
        let converter = create_test_converter()
            .with_processing_mode(ProcessingMode::Parallel)
            .with_max_concurrent_targets(2);

        let audio = create_test_audio();
        let target1 = NamedTarget::new(
            "speed_fast".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );
        let target2 = NamedTarget::new(
            "speed_slow".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );

        let request = MultiTargetConversionRequest::new(
            "test_parallel".to_string(),
            audio,
            22050,
            ConversionType::SpeedTransformation,
            vec![target1, target2],
        );

        let result = converter.convert_multi_target(request).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.target_results.len(), 2);
        assert!(result.stats.parallel_processing);
    }

    #[tokio::test]
    async fn test_target_priority_ordering() {
        let converter = create_test_converter().with_processing_mode(ProcessingMode::Sequential);

        let audio = create_test_audio();
        let target1 = NamedTarget::new(
            "low_priority".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        )
        .with_priority(1);

        let target2 = NamedTarget::new(
            "high_priority".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        )
        .with_priority(10);

        let request = MultiTargetConversionRequest::new(
            "test_priority".to_string(),
            audio,
            22050,
            ConversionType::PitchShift,
            vec![target1, target2], // Add low priority first
        );

        let result = converter.convert_multi_target(request).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.target_results.len(), 2);

        // Both targets should succeed regardless of order
        assert!(result.target_results.get("low_priority").unwrap().success);
        assert!(result.target_results.get("high_priority").unwrap().success);
    }

    #[tokio::test]
    async fn test_adaptive_processing_mode() {
        let converter = create_test_converter().with_processing_mode(ProcessingMode::Adaptive);

        let audio = create_test_audio();

        // Test with few targets (should use sequential)
        let target1 = NamedTarget::new(
            "target1".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );
        let request_few = MultiTargetConversionRequest::new(
            "test_adaptive_few".to_string(),
            audio.clone(),
            22050,
            ConversionType::PitchShift,
            vec![target1],
        );

        let result = converter.convert_multi_target(request_few).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.stats.parallel_processing); // Should use sequential

        // Test with multiple targets (should use parallel)
        let targets: Vec<NamedTarget> = (0..4)
            .map(|i| {
                NamedTarget::new(
                    format!("target_{}", i),
                    ConversionTarget::new(VoiceCharacteristics::default()),
                )
            })
            .collect();

        let request_many = MultiTargetConversionRequest::new(
            "test_adaptive_many".to_string(),
            audio,
            22050,
            ConversionType::PitchShift,
            targets,
        );

        let result = converter.convert_multi_target(request_many).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.stats.parallel_processing); // Should use parallel
    }

    #[tokio::test]
    async fn test_conversion_result_filtering() {
        let converter = create_test_converter();

        let audio = create_test_audio();
        let target1 = NamedTarget::new(
            "valid_target".to_string(),
            ConversionTarget::new(VoiceCharacteristics::default()),
        );

        let request = MultiTargetConversionRequest::new(
            "test_filtering".to_string(),
            audio,
            22050,
            ConversionType::PitchShift,
            vec![target1],
        );

        let result = converter.convert_multi_target(request).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);

        let successful = result.successful_results();
        let failed = result.failed_results();

        assert_eq!(successful.len(), 1);
        assert_eq!(failed.len(), 0);
        assert!(successful.contains_key("valid_target"));
    }

    #[tokio::test]
    async fn test_converter_statistics() {
        let converter = create_test_converter();

        // Get initial stats
        let initial_stats = converter.get_stats().await;
        assert_eq!(initial_stats.targets_processed, 0);

        let audio = create_test_audio();
        let targets: Vec<NamedTarget> = (0..3)
            .map(|i| {
                NamedTarget::new(
                    format!("target_{}", i),
                    ConversionTarget::new(VoiceCharacteristics::default()),
                )
            })
            .collect();

        let request = MultiTargetConversionRequest::new(
            "test_stats".to_string(),
            audio,
            22050,
            ConversionType::PitchShift,
            targets,
        );

        let _result = converter.convert_multi_target(request).await.unwrap();

        // Check updated stats
        let final_stats = converter.get_stats().await;
        assert_eq!(final_stats.targets_processed, 3);
        assert_eq!(final_stats.successful_conversions, 3);
        assert_eq!(final_stats.failed_conversions, 0);
        assert!(final_stats.average_processing_time > Duration::from_millis(0));
    }
}
