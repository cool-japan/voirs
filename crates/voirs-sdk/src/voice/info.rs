//! Voice information and metadata utilities.

use crate::types::{
    Gender, LanguageCode, ModelConfig, QualityLevel, SpeakingStyle, VoiceCharacteristics,
    VoiceConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Detailed voice information with computed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Basic voice configuration
    pub config: VoiceConfig,

    /// Computed voice metrics
    pub metrics: VoiceMetrics,

    /// Voice compatibility information
    pub compatibility: VoiceCompatibility,

    /// Model file information
    pub model_info: ModelInfo,

    /// Usage statistics
    pub usage_stats: Option<VoiceUsageStats>,
}

impl VoiceInfo {
    /// Create voice info from configuration
    pub fn from_config(config: VoiceConfig) -> Self {
        let metrics = VoiceMetrics::from_voice(&config);
        let compatibility = VoiceCompatibility::from_voice(&config);
        let model_info = ModelInfo::from_config(&config.model_config);

        Self {
            config,
            metrics,
            compatibility,
            model_info,
            usage_stats: None,
        }
    }

    /// Get voice ID
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Get voice name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get voice language
    pub fn language(&self) -> LanguageCode {
        self.config.language
    }

    /// Get voice characteristics
    pub fn characteristics(&self) -> &VoiceCharacteristics {
        &self.config.characteristics
    }

    /// Check if voice supports specific feature
    pub fn supports_feature(&self, feature: VoiceFeature) -> bool {
        match feature {
            VoiceFeature::EmotionSupport => self.config.characteristics.emotion_support,
            VoiceFeature::GpuAcceleration => {
                self.config.model_config.device_requirements.gpu_support
            }
            VoiceFeature::LowMemory => {
                self.config.model_config.device_requirements.min_memory_mb <= 512
            }
            VoiceFeature::HighQuality => matches!(
                self.config.characteristics.quality,
                QualityLevel::High | QualityLevel::Ultra
            ),
        }
    }

    /// Get voice summary
    pub fn summary(&self) -> VoiceSummary {
        VoiceSummary {
            id: self.config.id.clone(),
            name: self.config.name.clone(),
            language: self.config.language,
            gender: self.config.characteristics.gender,
            style: self.config.characteristics.style,
            quality: self.config.characteristics.quality,
            emotion_support: self.config.characteristics.emotion_support,
            memory_requirement: self.config.model_config.device_requirements.min_memory_mb,
            gpu_support: self.config.model_config.device_requirements.gpu_support,
        }
    }

    /// Export voice info as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import voice info from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Voice metrics and computed properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMetrics {
    /// Estimated quality score (0.0 - 1.0)
    pub quality_score: f32,

    /// Estimated naturalness score (0.0 - 1.0)
    pub naturalness_score: f32,

    /// Performance score (0.0 - 1.0, higher is faster)
    pub performance_score: f32,

    /// Memory efficiency score (0.0 - 1.0)
    pub memory_efficiency: f32,

    /// Overall rating (0.0 - 1.0)
    pub overall_rating: f32,

    /// Complexity level
    pub complexity: VoiceComplexity,
}

impl VoiceMetrics {
    /// Compute metrics from voice configuration
    pub fn from_voice(voice: &VoiceConfig) -> Self {
        let quality_score = Self::compute_quality_score(&voice.characteristics);
        let naturalness_score = Self::compute_naturalness_score(&voice.characteristics);
        let performance_score = Self::compute_performance_score(&voice.model_config);
        let memory_efficiency = Self::compute_memory_efficiency(&voice.model_config);
        let complexity = Self::determine_complexity(voice);

        let overall_rating =
            (quality_score + naturalness_score + performance_score + memory_efficiency) / 4.0;

        Self {
            quality_score,
            naturalness_score,
            performance_score,
            memory_efficiency,
            overall_rating,
            complexity,
        }
    }

    fn compute_quality_score(characteristics: &VoiceCharacteristics) -> f32 {
        match characteristics.quality {
            QualityLevel::Low => 0.25,
            QualityLevel::Medium => 0.5,
            QualityLevel::High => 0.75,
            QualityLevel::Ultra => 1.0,
        }
    }

    fn compute_naturalness_score(characteristics: &VoiceCharacteristics) -> f32 {
        let mut score: f32 = 0.6; // Base score

        // Emotion support increases naturalness
        if characteristics.emotion_support {
            score += 0.2;
        }

        // Certain styles are more natural
        match characteristics.style {
            SpeakingStyle::Neutral | SpeakingStyle::Calm => score += 0.15,
            SpeakingStyle::Conversational => score += 0.2,
            SpeakingStyle::News | SpeakingStyle::Formal => score += 0.05,
            _ => {}
        }

        score.min(1.0)
    }

    fn compute_performance_score(model_config: &ModelConfig) -> f32 {
        let base_score: f32 = if model_config.device_requirements.gpu_support {
            0.8 // GPU acceleration gives better performance
        } else {
            0.5
        };

        // Lower memory requirement = better performance score
        let memory_factor: f32 = if model_config.device_requirements.min_memory_mb <= 512 {
            1.0
        } else if model_config.device_requirements.min_memory_mb <= 1024 {
            0.8
        } else {
            0.6
        };

        (base_score * memory_factor).min(1.0)
    }

    fn compute_memory_efficiency(model_config: &ModelConfig) -> f32 {
        // Inverse relationship with memory requirement
        let memory_mb = model_config.device_requirements.min_memory_mb as f32;
        (2048.0 - memory_mb.min(2048.0)) / 2048.0
    }

    fn determine_complexity(voice: &VoiceConfig) -> VoiceComplexity {
        let memory_mb = voice.model_config.device_requirements.min_memory_mb;
        let has_emotion = voice.characteristics.emotion_support;
        let high_quality = matches!(
            voice.characteristics.quality,
            QualityLevel::High | QualityLevel::Ultra
        );

        match (memory_mb, has_emotion, high_quality) {
            (mb, _, _) if mb > 1536 => VoiceComplexity::High,
            (mb, true, true) if mb > 768 => VoiceComplexity::High,
            (mb, _, true) if mb > 512 => VoiceComplexity::Medium,
            (mb, true, _) if mb > 512 => VoiceComplexity::Medium,
            _ => VoiceComplexity::Low,
        }
    }
}

/// Voice complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceComplexity {
    /// Simple voice, low resource requirements
    Low,
    /// Moderate complexity and resource usage
    Medium,
    /// Complex voice with high resource requirements
    High,
}

impl fmt::Display for VoiceComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
        }
    }
}

/// Voice compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCompatibility {
    /// Supported platforms
    pub platforms: Vec<String>,

    /// Required compute capabilities
    pub compute_capabilities: Vec<String>,

    /// Compatible languages
    pub compatible_languages: Vec<LanguageCode>,

    /// Minimum system requirements
    pub min_requirements: SystemRequirements,

    /// Recommended system requirements
    pub recommended_requirements: SystemRequirements,
}

impl VoiceCompatibility {
    /// Determine compatibility from voice configuration
    pub fn from_voice(voice: &VoiceConfig) -> Self {
        let platforms = vec![
            "linux".to_string(),
            "macos".to_string(),
            "windows".to_string(),
        ];

        let compute_capabilities = voice
            .model_config
            .device_requirements
            .compute_capabilities
            .clone();

        // Compatible languages based on voice characteristics
        let compatible_languages = vec![voice.language];

        let min_requirements = SystemRequirements {
            memory_mb: voice.model_config.device_requirements.min_memory_mb,
            storage_mb: 1024, // Estimated
            cpu_cores: 1,
            gpu_memory_mb: if voice.model_config.device_requirements.gpu_support {
                Some(512)
            } else {
                None
            },
        };

        let recommended_requirements = SystemRequirements {
            memory_mb: voice.model_config.device_requirements.min_memory_mb * 2,
            storage_mb: 2048,
            cpu_cores: 4,
            gpu_memory_mb: if voice.model_config.device_requirements.gpu_support {
                Some(2048)
            } else {
                None
            },
        };

        Self {
            platforms,
            compute_capabilities,
            compatible_languages,
            min_requirements,
            recommended_requirements,
        }
    }
}

/// System requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemRequirements {
    /// Required memory in MB
    pub memory_mb: u32,

    /// Required storage in MB
    pub storage_mb: u32,

    /// Minimum CPU cores
    pub cpu_cores: u32,

    /// Required GPU memory in MB (if applicable)
    pub gpu_memory_mb: Option<u32>,
}

/// Model file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// G2P model file info
    pub g2p_model: Option<ModelFileInfo>,

    /// Acoustic model file info
    pub acoustic_model: ModelFileInfo,

    /// Vocoder model file info
    pub vocoder_model: ModelFileInfo,

    /// Total estimated size in MB
    pub total_size_mb: u32,

    /// Model format
    pub format: String,
}

impl ModelInfo {
    /// Create model info from configuration
    pub fn from_config(config: &ModelConfig) -> Self {
        let g2p_model = config
            .g2p_model
            .as_ref()
            .map(|path| ModelFileInfo::from_path(path));

        let acoustic_model = ModelFileInfo::from_path(&config.acoustic_model);
        let vocoder_model = ModelFileInfo::from_path(&config.vocoder_model);

        let total_size_mb = g2p_model.as_ref().map(|m| m.estimated_size_mb).unwrap_or(0)
            + acoustic_model.estimated_size_mb
            + vocoder_model.estimated_size_mb;

        Self {
            g2p_model,
            acoustic_model,
            vocoder_model,
            total_size_mb,
            format: format!("{:?}", config.format),
        }
    }
}

/// Individual model file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileInfo {
    /// File path
    pub path: String,

    /// Estimated file size in MB
    pub estimated_size_mb: u32,

    /// Model type
    pub model_type: String,
}

impl ModelFileInfo {
    /// Create model file info from path
    pub fn from_path(path: &str) -> Self {
        let model_type = if path.contains("g2p") {
            "G2P"
        } else if path.contains("acoustic") {
            "Acoustic"
        } else if path.contains("vocoder") {
            "Vocoder"
        } else {
            "Unknown"
        }
        .to_string();

        // Estimate size based on model type
        let estimated_size_mb = match model_type.as_str() {
            "G2P" => 50,
            "Acoustic" => 200,
            "Vocoder" => 100,
            _ => 150,
        };

        Self {
            path: path.to_string(),
            estimated_size_mb,
            model_type,
        }
    }
}

/// Voice usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceUsageStats {
    /// Number of times voice has been used
    pub usage_count: u64,

    /// Total synthesis time in seconds
    pub total_synthesis_time: f64,

    /// Average synthesis time per request
    pub avg_synthesis_time: f64,

    /// Last used timestamp
    pub last_used: Option<std::time::SystemTime>,

    /// Most common text lengths
    pub common_text_lengths: Vec<usize>,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for voice usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average real-time factor (lower is better)
    pub avg_rtf: f32,

    /// Peak memory usage in MB
    pub peak_memory_mb: u32,

    /// Average memory usage in MB
    pub avg_memory_mb: u32,

    /// Error rate (0.0 - 1.0)
    pub error_rate: f32,
}

/// Voice feature enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceFeature {
    /// Supports emotion in synthesis
    EmotionSupport,
    /// Supports GPU acceleration
    GpuAcceleration,
    /// Low memory usage
    LowMemory,
    /// High quality output
    HighQuality,
}

/// Compact voice summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSummary {
    /// Voice ID
    pub id: String,

    /// Voice name
    pub name: String,

    /// Language
    pub language: LanguageCode,

    /// Gender (if specified)
    pub gender: Option<Gender>,

    /// Speaking style
    pub style: SpeakingStyle,

    /// Quality level
    pub quality: QualityLevel,

    /// Emotion support
    pub emotion_support: bool,

    /// Memory requirement in MB
    pub memory_requirement: u32,

    /// GPU support
    pub gpu_support: bool,
}

impl fmt::Display for VoiceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) - {:?} {:?} {:?} [{}MB{}]",
            self.name,
            self.id,
            self.language,
            self.gender
                .map(|g| format!("{g:?}"))
                .unwrap_or_else(|| "Unknown".to_string()),
            self.quality,
            self.memory_requirement,
            if self.emotion_support {
                ", Emotion"
            } else {
                ""
            }
        )
    }
}

/// Voice comparison utility
pub struct VoiceComparator;

impl VoiceComparator {
    /// Compare two voices and return a comparison result
    pub fn compare(voice1: &VoiceInfo, voice2: &VoiceInfo) -> VoiceComparison {
        VoiceComparison {
            voice1_id: voice1.id().to_string(),
            voice2_id: voice2.id().to_string(),
            quality_diff: voice1.metrics.quality_score - voice2.metrics.quality_score,
            performance_diff: voice1.metrics.performance_score - voice2.metrics.performance_score,
            memory_diff: voice1.config.model_config.device_requirements.min_memory_mb as i32
                - voice2.config.model_config.device_requirements.min_memory_mb as i32,
            features_diff: Self::compare_features(voice1, voice2),
        }
    }

    /// Find best voice for specific criteria
    pub fn find_best_voice<'a>(
        voices: &'a [VoiceInfo],
        criteria: &VoiceSelectionCriteria,
    ) -> Option<&'a VoiceInfo> {
        voices
            .iter()
            .filter(|voice| Self::matches_criteria(voice, criteria))
            .max_by(|a, b| {
                Self::score_voice(a, criteria)
                    .partial_cmp(&Self::score_voice(b, criteria))
                    .unwrap()
            })
    }

    fn compare_features(
        voice1: &VoiceInfo,
        voice2: &VoiceInfo,
    ) -> HashMap<String, FeatureComparison> {
        let mut features = HashMap::new();

        features.insert(
            "emotion_support".to_string(),
            FeatureComparison {
                voice1_has: voice1.config.characteristics.emotion_support,
                voice2_has: voice2.config.characteristics.emotion_support,
            },
        );

        features.insert(
            "gpu_support".to_string(),
            FeatureComparison {
                voice1_has: voice1.config.model_config.device_requirements.gpu_support,
                voice2_has: voice2.config.model_config.device_requirements.gpu_support,
            },
        );

        features
    }

    fn matches_criteria(voice: &VoiceInfo, criteria: &VoiceSelectionCriteria) -> bool {
        if let Some(max_memory) = criteria.max_memory_mb {
            if voice.config.model_config.device_requirements.min_memory_mb > max_memory {
                return false;
            }
        }

        if let Some(min_quality) = criteria.min_quality_score {
            if voice.metrics.quality_score < min_quality {
                return false;
            }
        }

        if let Some(require_emotion) = criteria.require_emotion_support {
            if voice.config.characteristics.emotion_support != require_emotion {
                return false;
            }
        }

        true
    }

    fn score_voice(voice: &VoiceInfo, criteria: &VoiceSelectionCriteria) -> f32 {
        let mut score = voice.metrics.overall_rating;

        // Apply criteria-specific weights
        if criteria.prioritize_quality {
            score += voice.metrics.quality_score * 0.3;
        }

        if criteria.prioritize_performance {
            score += voice.metrics.performance_score * 0.3;
        }

        if criteria.prioritize_memory_efficiency {
            score += voice.metrics.memory_efficiency * 0.2;
        }

        score
    }
}

/// Voice comparison result
#[derive(Debug, Clone)]
pub struct VoiceComparison {
    pub voice1_id: String,
    pub voice2_id: String,
    pub quality_diff: f32,
    pub performance_diff: f32,
    pub memory_diff: i32,
    pub features_diff: HashMap<String, FeatureComparison>,
}

/// Feature comparison for two voices
#[derive(Debug, Clone)]
pub struct FeatureComparison {
    pub voice1_has: bool,
    pub voice2_has: bool,
}

/// Voice selection criteria for finding best voice
#[derive(Debug, Clone, Default)]
pub struct VoiceSelectionCriteria {
    pub max_memory_mb: Option<u32>,
    pub min_quality_score: Option<f32>,
    pub require_emotion_support: Option<bool>,
    pub prioritize_quality: bool,
    pub prioritize_performance: bool,
    pub prioritize_memory_efficiency: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn create_test_voice() -> VoiceConfig {
        VoiceConfig {
            id: "test-voice".to_string(),
            name: "Test Voice".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::High,
            },
            model_config: ModelConfig {
                g2p_model: Some("g2p.bin".to_string()),
                acoustic_model: "acoustic.bin".to_string(),
                vocoder_model: "vocoder.bin".to_string(),
                format: ModelFormat::Candle,
                device_requirements: DeviceRequirements {
                    min_memory_mb: 1024,
                    gpu_support: true,
                    compute_capabilities: vec!["cpu".to_string(), "cuda".to_string()],
                },
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_voice_info_creation() {
        let voice_config = create_test_voice();
        let voice_info = VoiceInfo::from_config(voice_config);

        assert_eq!(voice_info.id(), "test-voice");
        assert_eq!(voice_info.language(), LanguageCode::EnUs);
        assert!(voice_info.supports_feature(VoiceFeature::EmotionSupport));
        assert!(voice_info.supports_feature(VoiceFeature::GpuAcceleration));
        assert!(voice_info.supports_feature(VoiceFeature::HighQuality));
    }

    #[test]
    fn test_voice_metrics() {
        let voice_config = create_test_voice();
        let metrics = VoiceMetrics::from_voice(&voice_config);

        assert!(metrics.quality_score > 0.0);
        assert!(metrics.naturalness_score > 0.0);
        assert!(metrics.overall_rating > 0.0);
        assert_eq!(metrics.complexity, VoiceComplexity::High);
    }

    #[test]
    fn test_voice_summary() {
        let voice_config = create_test_voice();
        let voice_info = VoiceInfo::from_config(voice_config);
        let summary = voice_info.summary();

        assert_eq!(summary.id, "test-voice");
        assert_eq!(summary.name, "Test Voice");
        assert_eq!(summary.language, LanguageCode::EnUs);
        assert_eq!(summary.gender, Some(Gender::Female));
        assert!(summary.emotion_support);
    }

    #[test]
    fn test_voice_comparison() {
        let voice1 = VoiceInfo::from_config(create_test_voice());

        let mut voice2_config = create_test_voice();
        voice2_config.id = "test-voice-2".to_string();
        voice2_config.characteristics.quality = QualityLevel::Medium;
        voice2_config.model_config.device_requirements.min_memory_mb = 512;
        let voice2 = VoiceInfo::from_config(voice2_config);

        let comparison = VoiceComparator::compare(&voice1, &voice2);

        assert!(comparison.quality_diff > 0.0); // voice1 has higher quality
        assert!(comparison.memory_diff > 0); // voice1 uses more memory
    }

    #[test]
    fn test_voice_selection() {
        let voice1 = VoiceInfo::from_config(create_test_voice());

        let mut voice2_config = create_test_voice();
        voice2_config.id = "test-voice-2".to_string();
        voice2_config.characteristics.emotion_support = false;
        let voice2 = VoiceInfo::from_config(voice2_config);

        let voices = vec![voice1, voice2];

        let criteria = VoiceSelectionCriteria {
            require_emotion_support: Some(true),
            ..Default::default()
        };

        let best = VoiceComparator::find_best_voice(&voices, &criteria);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id(), "test-voice");
    }

    #[test]
    fn test_json_serialization() {
        let voice_config = create_test_voice();
        let voice_info = VoiceInfo::from_config(voice_config);

        let json = voice_info.to_json().unwrap();
        assert!(json.contains("test-voice"));

        let restored = VoiceInfo::from_json(&json).unwrap();
        assert_eq!(restored.id(), voice_info.id());
    }

    #[test]
    fn test_model_info() {
        let voice_config = create_test_voice();
        let model_info = ModelInfo::from_config(&voice_config.model_config);

        assert!(model_info.g2p_model.is_some());
        assert_eq!(model_info.acoustic_model.model_type, "Acoustic");
        assert_eq!(model_info.vocoder_model.model_type, "Vocoder");
        assert!(model_info.total_size_mb > 0);
    }
}
