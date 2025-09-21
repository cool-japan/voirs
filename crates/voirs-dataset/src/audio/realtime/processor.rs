//! Real-time processor implementation and trait

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::buffer::{BufferConfig, RealTimeBuffer};
use super::interactive::InteractiveConfig;
use super::latency::LatencyConfig;
use super::monitoring::{Alert, AlertSeverity, AlertType, QualityMonitoringConfig};
use super::optimization::RealTimeOptimization;
use super::processing::{ProcessingConfig, ProcessingStage};
use super::quality::QualityAssessment;
use crate::Result;

/// Real-time processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealTimeConfig {
    /// Buffer configuration
    pub buffer_config: BufferConfig,
    /// Processing configuration
    pub processing_config: ProcessingConfig,
    /// Quality monitoring configuration
    pub quality_monitoring: QualityMonitoringConfig,
    /// Latency configuration
    pub latency_config: LatencyConfig,
    /// Interactive processing configuration
    pub interactive_config: InteractiveConfig,
    /// Performance optimization settings
    pub optimization: RealTimeOptimization,
}

/// Real-time processing statistics
#[derive(Debug, Clone, Serialize)]
pub struct RealTimeStatistics {
    /// Processing latency
    #[serde(with = "duration_serde")]
    pub processing_latency: Duration,
    /// Buffer utilization
    pub buffer_utilization: f32,
    /// CPU usage
    pub cpu_usage: f32,
    /// Memory usage
    pub memory_usage: usize,
    /// Throughput
    pub throughput: f32,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Error rate
    pub error_rate: f32,
    /// Timestamp
    #[serde(skip)]
    pub timestamp: Instant,
}

/// Real-time processing result
#[derive(Debug, Clone, Serialize)]
pub struct RealTimeResult {
    /// Processed audio data
    pub audio_data: Vec<f32>,
    /// Processing statistics
    pub statistics: RealTimeStatistics,
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
    /// Alerts
    pub alerts: Vec<Alert>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Real-time processor trait
#[async_trait]
pub trait RealTimeProcessor: Send + Sync {
    /// Start real-time processing
    async fn start(&mut self) -> Result<()>;

    /// Stop real-time processing
    async fn stop(&mut self) -> Result<()>;

    /// Process audio buffer
    async fn process_buffer(&mut self, buffer: &RealTimeBuffer) -> Result<RealTimeResult>;

    /// Get processing statistics
    async fn get_statistics(&self) -> Result<RealTimeStatistics>;

    /// Get quality assessment
    async fn get_quality_assessment(&self) -> Result<QualityAssessment>;

    /// Set configuration
    async fn set_config(&mut self, config: RealTimeConfig) -> Result<()>;

    /// Get configuration
    async fn get_config(&self) -> Result<RealTimeConfig>;
}

/// Default real-time processor implementation
pub struct DefaultRealTimeProcessor {
    config: RealTimeConfig,
    #[allow(dead_code)]
    buffer: Arc<Mutex<RealTimeBuffer>>,
    statistics: Arc<RwLock<RealTimeStatistics>>,
    is_running: Arc<Mutex<bool>>,
}

impl DefaultRealTimeProcessor {
    /// Create a new real-time processor
    pub fn new(config: RealTimeConfig) -> Self {
        let buffer = RealTimeBuffer::new(
            config.buffer_config.buffer_size,
            config.buffer_config.sample_rate,
            config.buffer_config.channels,
        );

        let statistics = RealTimeStatistics {
            processing_latency: Duration::from_millis(0),
            buffer_utilization: 0.0,
            cpu_usage: 0.0,
            memory_usage: 0,
            throughput: 0.0,
            quality_metrics: HashMap::new(),
            error_rate: 0.0,
            timestamp: Instant::now(),
        };

        Self {
            config,
            buffer: Arc::new(Mutex::new(buffer)),
            statistics: Arc::new(RwLock::new(statistics)),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(RealTimeConfig::default())
    }

    /// Process audio in real-time
    async fn process_audio(&mut self, audio_data: &[f32]) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // Apply processing chain
        let mut processed_data = audio_data.to_vec();

        for stage in &self.config.processing_config.processing_chain {
            processed_data = self.apply_processing_stage(stage, &processed_data)?;
        }

        // Update statistics
        let processing_latency = start_time.elapsed();
        let mut stats = self.statistics.write().await;
        stats.processing_latency = processing_latency;
        stats.timestamp = Instant::now();

        Ok(processed_data)
    }

    /// Apply processing stage
    fn apply_processing_stage(&self, stage: &ProcessingStage, data: &[f32]) -> Result<Vec<f32>> {
        match stage {
            ProcessingStage::PreProcessing {
                noise_reduction,
                gain_control,
                high_pass_filter: _,
                low_pass_filter: _,
            } => {
                let mut result = data.to_vec();

                if *noise_reduction {
                    // Simple noise reduction implementation
                    for sample in &mut result {
                        if sample.abs() < 0.01 {
                            *sample *= 0.5;
                        }
                    }
                }

                if *gain_control {
                    // Simple gain control implementation
                    let max_amplitude = result.iter().map(|x| x.abs()).fold(0.0, f32::max);
                    if max_amplitude > 0.8 {
                        let gain = 0.8 / max_amplitude;
                        for sample in &mut result {
                            *sample *= gain;
                        }
                    }
                }

                Ok(result)
            }
            ProcessingStage::Analysis { .. } => {
                // For analysis, just return the data unchanged
                Ok(data.to_vec())
            }
            ProcessingStage::Enhancement { .. } => {
                // Simple enhancement implementation
                Ok(data.to_vec())
            }
            ProcessingStage::PostProcessing { normalization, .. } => {
                let mut result = data.to_vec();

                if *normalization {
                    let max_amplitude = result.iter().map(|x| x.abs()).fold(0.0, f32::max);
                    if max_amplitude > 0.0 {
                        let gain = 1.0 / max_amplitude;
                        for sample in &mut result {
                            *sample *= gain;
                        }
                    }
                }

                Ok(result)
            }
        }
    }
}

#[async_trait]
impl RealTimeProcessor for DefaultRealTimeProcessor {
    async fn start(&mut self) -> Result<()> {
        let mut is_running = self.is_running.lock().unwrap();
        *is_running = true;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        let mut is_running = self.is_running.lock().unwrap();
        *is_running = false;
        Ok(())
    }

    async fn process_buffer(&mut self, buffer: &RealTimeBuffer) -> Result<RealTimeResult> {
        let audio_data: Vec<f32> = buffer.data.iter().copied().collect();
        let processed_data = self.process_audio(&audio_data).await?;

        let statistics = self.get_statistics().await?;
        let quality_assessment = self.get_quality_assessment().await?;

        // Generate alerts if needed
        let mut alerts = Vec::new();
        if statistics.processing_latency > self.config.latency_config.max_latency {
            alerts.push(Alert::new(
                "latency_warning".to_string(),
                AlertType::Performance,
                AlertSeverity::Medium,
                format!(
                    "Processing latency ({:?}) exceeds threshold ({:?})",
                    statistics.processing_latency, self.config.latency_config.max_latency
                ),
            ));
        }

        Ok(RealTimeResult {
            audio_data: processed_data,
            statistics,
            quality_assessment,
            alerts,
            metadata: HashMap::new(),
        })
    }

    async fn get_statistics(&self) -> Result<RealTimeStatistics> {
        let stats = self.statistics.read().await;
        Ok(stats.clone())
    }

    async fn get_quality_assessment(&self) -> Result<QualityAssessment> {
        let mut assessment = QualityAssessment::new();
        assessment.overall_score = 0.8; // Placeholder score
        assessment.confidence = 0.9;
        Ok(assessment)
    }

    async fn set_config(&mut self, config: RealTimeConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    async fn get_config(&self) -> Result<RealTimeConfig> {
        Ok(self.config.clone())
    }
}

impl Default for DefaultRealTimeProcessor {
    fn default() -> Self {
        Self::with_default_config()
    }
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}
