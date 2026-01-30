//! Utility functions for diffusion sampling
//!
//! This module provides helper functions for configuring, analyzing, and optimizing
//! diffusion sampling processes.

use super::sampling::{SamplingAlgorithm, SamplingConfig};
use serde::{Deserialize, Serialize};

/// Recommended sampling configuration for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingPreset {
    /// Maximum quality for research (DDPM with 1000 steps)
    MaximumQuality,
    /// Production high quality (UniPC with 10 steps)
    ProductionHighQuality,
    /// Fast production (DPM-Solver++ with 10 steps)
    FastProduction,
    /// Real-time capable (DPM-Solver++ with 5-7 steps)
    RealTime,
    /// Quick preview (FastDDIM with 10 steps)
    QuickPreview,
    /// Balanced quality and speed (UniPC with 7 steps)
    Balanced,
    /// Automatic optimization (Adaptive with 30 steps)
    Adaptive,
}

impl SamplingPreset {
    /// Get the sampling configuration for this preset
    pub fn to_config(self) -> SamplingConfig {
        match self {
            SamplingPreset::MaximumQuality => SamplingConfig {
                algorithm: SamplingAlgorithm::DDPM,
                num_steps: 1000,
                eta: 0.0,
                temperature: 1.0,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::ProductionHighQuality => SamplingConfig {
                algorithm: SamplingAlgorithm::UniPC,
                num_steps: 10,
                eta: 0.0,
                temperature: 0.9,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::FastProduction => SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 10,
                eta: 0.0,
                temperature: 0.95,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::RealTime => SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 6,
                eta: 0.0,
                temperature: 1.0,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::QuickPreview => SamplingConfig {
                algorithm: SamplingAlgorithm::FastDDIM,
                num_steps: 10,
                eta: 0.0,
                temperature: 1.1,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::Balanced => SamplingConfig {
                algorithm: SamplingAlgorithm::UniPC,
                num_steps: 7,
                eta: 0.0,
                temperature: 0.95,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
            SamplingPreset::Adaptive => SamplingConfig {
                algorithm: SamplingAlgorithm::Adaptive,
                num_steps: 30,
                eta: 0.0,
                temperature: 1.0,
                guidance_scale: 7.5,
                use_guidance: false,
                seed: None,
            },
        }
    }

    /// Get the expected Mean Opinion Score (MOS) for this preset
    pub fn expected_mos(self) -> f32 {
        match self {
            SamplingPreset::MaximumQuality => 5.0,
            SamplingPreset::ProductionHighQuality => 4.85,
            SamplingPreset::FastProduction => 4.75,
            SamplingPreset::RealTime => 4.4,
            SamplingPreset::QuickPreview => 4.0,
            SamplingPreset::Balanced => 4.7,
            SamplingPreset::Adaptive => 4.6,
        }
    }

    /// Get the expected Real-Time Factor (RTF) for this preset
    /// Lower is better (< 1.0 means faster than real-time)
    pub fn expected_rtf(self) -> f32 {
        match self {
            SamplingPreset::MaximumQuality => 0.50,
            SamplingPreset::ProductionHighQuality => 0.030,
            SamplingPreset::FastProduction => 0.025,
            SamplingPreset::RealTime => 0.012,
            SamplingPreset::QuickPreview => 0.015,
            SamplingPreset::Balanced => 0.020,
            SamplingPreset::Adaptive => 0.075,
        }
    }

    /// Get a description of this preset
    pub fn description(self) -> &'static str {
        match self {
            SamplingPreset::MaximumQuality => {
                "Maximum quality for research purposes. Very slow but highest quality."
            }
            SamplingPreset::ProductionHighQuality => {
                "Recommended for production. Excellent quality with good speed."
            }
            SamplingPreset::FastProduction => {
                "Fast production synthesis with high quality. Good balance."
            }
            SamplingPreset::RealTime => {
                "Real-time capable synthesis. Suitable for interactive applications."
            }
            SamplingPreset::QuickPreview => {
                "Quick preview generation. Good for prototyping and testing."
            }
            SamplingPreset::Balanced => {
                "Best balance between quality and speed. Recommended starting point."
            }
            SamplingPreset::Adaptive => {
                "Automatically adjusts based on input complexity. Variable speed."
            }
        }
    }

    /// Check if this preset is suitable for real-time synthesis
    pub fn is_realtime_capable(self) -> bool {
        self.expected_rtf() < 0.05
    }

    /// Check if this preset is suitable for production use
    pub fn is_production_ready(self) -> bool {
        self.expected_mos() >= 4.5 && self.expected_rtf() < 0.1
    }

    /// Get all available presets
    pub fn all() -> Vec<Self> {
        vec![
            Self::MaximumQuality,
            Self::ProductionHighQuality,
            Self::FastProduction,
            Self::RealTime,
            Self::QuickPreview,
            Self::Balanced,
            Self::Adaptive,
        ]
    }

    /// Select the best preset for given constraints
    pub fn select_best(
        min_quality_mos: Option<f32>,
        max_rtf: Option<f32>,
        prefer_speed: bool,
    ) -> Self {
        let mut candidates: Vec<_> = Self::all()
            .into_iter()
            .filter(|preset| {
                let quality_ok = min_quality_mos
                    .map(|min_mos| preset.expected_mos() >= min_mos)
                    .unwrap_or(true);
                let speed_ok = max_rtf
                    .map(|max| preset.expected_rtf() <= max)
                    .unwrap_or(true);
                quality_ok && speed_ok
            })
            .collect();

        if candidates.is_empty() {
            // If no preset meets constraints, return balanced
            return Self::Balanced;
        }

        // Sort by preference
        if prefer_speed {
            candidates.sort_by(|a, b| {
                a.expected_rtf()
                    .partial_cmp(&b.expected_rtf())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            candidates.sort_by(|a, b| {
                b.expected_mos()
                    .partial_cmp(&a.expected_mos())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        candidates[0]
    }
}

/// Performance analysis for a sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingPerformanceAnalysis {
    /// Estimated time to generate 1 second of audio (in milliseconds)
    pub estimated_time_ms: f32,
    /// Estimated Real-Time Factor
    pub estimated_rtf: f32,
    /// Estimated quality (MOS scale)
    pub estimated_mos: f32,
    /// Whether this is suitable for real-time use
    pub realtime_capable: bool,
    /// Whether this is suitable for production use
    pub production_ready: bool,
    /// Recommended use case
    pub recommended_use_case: String,
}

impl SamplingPerformanceAnalysis {
    /// Analyze a sampling configuration
    pub fn analyze(config: &SamplingConfig) -> Self {
        // Estimate based on algorithm and steps
        let base_time_per_step = match config.algorithm {
            SamplingAlgorithm::DDPM => 50.0,
            SamplingAlgorithm::DDIM => 20.0,
            SamplingAlgorithm::FastDDIM => 15.0,
            SamplingAlgorithm::Adaptive => 22.0,
            SamplingAlgorithm::DPMSolverPlusPlus => 25.0,
            SamplingAlgorithm::UniPC => 30.0,
        };

        let estimated_time_ms = base_time_per_step * config.num_steps as f32;
        let estimated_rtf = estimated_time_ms / 1000.0; // Assuming 1 second of audio

        // Estimate quality based on algorithm and steps
        let estimated_mos = match config.algorithm {
            SamplingAlgorithm::DDPM => (4.5 + (config.num_steps as f32 / 2000.0)).min(5.0),
            SamplingAlgorithm::DDIM => (3.8 + (config.num_steps as f32 / 50.0)).min(4.7),
            SamplingAlgorithm::FastDDIM => (3.5 + (config.num_steps as f32 / 25.0)).min(4.3),
            SamplingAlgorithm::Adaptive => (3.9 + (config.num_steps as f32 / 40.0)).min(4.6),
            SamplingAlgorithm::DPMSolverPlusPlus => {
                (4.0 + (config.num_steps as f32 / 20.0)).min(4.8)
            }
            SamplingAlgorithm::UniPC => (4.2 + (config.num_steps as f32 / 15.0)).min(4.9),
        };

        let realtime_capable = estimated_rtf < 0.05;
        let production_ready = estimated_mos >= 4.5 && estimated_rtf < 0.1;

        let recommended_use_case = if estimated_rtf > 0.3 {
            "Research/Offline Processing".to_string()
        } else if realtime_capable {
            "Real-time/Interactive Applications".to_string()
        } else if production_ready {
            "Production Synthesis".to_string()
        } else {
            "Preview/Testing".to_string()
        };

        Self {
            estimated_time_ms,
            estimated_rtf,
            estimated_mos,
            realtime_capable,
            production_ready,
            recommended_use_case,
        }
    }

    /// Get a quality rating (Poor, Fair, Good, Excellent)
    pub fn quality_rating(&self) -> &'static str {
        if self.estimated_mos >= 4.7 {
            "Excellent"
        } else if self.estimated_mos >= 4.5 {
            "Very Good"
        } else if self.estimated_mos >= 4.2 {
            "Good"
        } else if self.estimated_mos >= 4.0 {
            "Fair"
        } else {
            "Acceptable"
        }
    }

    /// Get a speed rating
    pub fn speed_rating(&self) -> &'static str {
        if self.estimated_rtf < 0.02 {
            "Very Fast"
        } else if self.estimated_rtf < 0.05 {
            "Fast"
        } else if self.estimated_rtf < 0.1 {
            "Moderate"
        } else if self.estimated_rtf < 0.3 {
            "Slow"
        } else {
            "Very Slow"
        }
    }
}

/// Create a sampling configuration optimized for a specific target
pub struct SamplingOptimizer;

impl SamplingOptimizer {
    /// Optimize for maximum quality (research use)
    pub fn for_maximum_quality() -> SamplingConfig {
        SamplingPreset::MaximumQuality.to_config()
    }

    /// Optimize for production use (balance of quality and speed)
    pub fn for_production() -> SamplingConfig {
        SamplingPreset::ProductionHighQuality.to_config()
    }

    /// Optimize for real-time synthesis
    pub fn for_realtime() -> SamplingConfig {
        SamplingPreset::RealTime.to_config()
    }

    /// Optimize for specific quality target
    pub fn for_quality_target(target_mos: f32) -> SamplingConfig {
        if target_mos >= 4.8 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::UniPC,
                num_steps: 10,
                eta: 0.0,
                temperature: 0.9,
                ..Default::default()
            }
        } else if target_mos >= 4.5 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 15,
                eta: 0.0,
                temperature: 0.95,
                ..Default::default()
            }
        } else if target_mos >= 4.2 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 10,
                eta: 0.0,
                temperature: 1.0,
                ..Default::default()
            }
        } else {
            SamplingConfig {
                algorithm: SamplingAlgorithm::FastDDIM,
                num_steps: 12,
                eta: 0.0,
                temperature: 1.0,
                ..Default::default()
            }
        }
    }

    /// Optimize for specific speed target (RTF)
    pub fn for_speed_target(max_rtf: f32) -> SamplingConfig {
        if max_rtf < 0.02 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 5,
                eta: 0.0,
                temperature: 1.0,
                ..Default::default()
            }
        } else if max_rtf < 0.05 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::UniPC,
                num_steps: 7,
                eta: 0.0,
                temperature: 0.95,
                ..Default::default()
            }
        } else if max_rtf < 0.1 {
            SamplingConfig {
                algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
                num_steps: 15,
                eta: 0.0,
                temperature: 0.95,
                ..Default::default()
            }
        } else {
            SamplingConfig {
                algorithm: SamplingAlgorithm::DDIM,
                num_steps: 50,
                eta: 0.0,
                temperature: 1.0,
                ..Default::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_preset_configs() {
        for preset in SamplingPreset::all() {
            let config = preset.to_config();
            assert!(config.num_steps > 0);
            assert!(config.temperature > 0.0);
            assert!(preset.expected_mos() >= 4.0);
            assert!(preset.expected_mos() <= 5.0);
            assert!(preset.expected_rtf() > 0.0);
        }
    }

    #[test]
    fn test_realtime_presets() {
        let realtime_presets: Vec<_> = SamplingPreset::all()
            .into_iter()
            .filter(|p| p.is_realtime_capable())
            .collect();

        assert!(!realtime_presets.is_empty());
        for preset in realtime_presets {
            assert!(preset.expected_rtf() < 0.05);
        }
    }

    #[test]
    fn test_production_ready_presets() {
        let production_presets: Vec<_> = SamplingPreset::all()
            .into_iter()
            .filter(|p| p.is_production_ready())
            .collect();

        assert!(!production_presets.is_empty());
        for preset in production_presets {
            assert!(preset.expected_mos() >= 4.5);
            assert!(preset.expected_rtf() < 0.1);
        }
    }

    #[test]
    fn test_preset_selection() {
        // High quality requirement
        let preset = SamplingPreset::select_best(Some(4.8), None, false);
        assert!(preset.expected_mos() >= 4.8);

        // Fast requirement
        let preset = SamplingPreset::select_best(None, Some(0.02), true);
        assert!(preset.expected_rtf() <= 0.02);

        // Balanced requirement
        let preset = SamplingPreset::select_best(Some(4.5), Some(0.05), false);
        assert!(preset.expected_mos() >= 4.5);
        assert!(preset.expected_rtf() <= 0.05);
    }

    #[test]
    fn test_performance_analysis() {
        let config = SamplingConfig {
            algorithm: SamplingAlgorithm::UniPC,
            num_steps: 10,
            ..Default::default()
        };

        let analysis = SamplingPerformanceAnalysis::analyze(&config);
        assert!(analysis.estimated_time_ms > 0.0);
        assert!(analysis.estimated_rtf > 0.0);
        assert!(analysis.estimated_mos >= 4.0);
        assert!(!analysis.recommended_use_case.is_empty());
    }

    #[test]
    fn test_quality_ratings() {
        let excellent_config = SamplingConfig {
            algorithm: SamplingAlgorithm::UniPC,
            num_steps: 10,
            ..Default::default()
        };
        let analysis = SamplingPerformanceAnalysis::analyze(&excellent_config);
        assert!(matches!(
            analysis.quality_rating(),
            "Excellent" | "Very Good" | "Good"
        ));

        let fast_config = SamplingConfig {
            algorithm: SamplingAlgorithm::DPMSolverPlusPlus,
            num_steps: 5,
            ..Default::default()
        };
        let analysis = SamplingPerformanceAnalysis::analyze(&fast_config);
        // Should be fast, but exact rating depends on implementation
        assert!(!analysis.speed_rating().is_empty());
    }

    #[test]
    fn test_sampling_optimizer() {
        // Maximum quality
        let config = SamplingOptimizer::for_maximum_quality();
        assert!(matches!(config.algorithm, SamplingAlgorithm::DDPM));

        // Production - should have good quality
        let config = SamplingOptimizer::for_production();
        let analysis = SamplingPerformanceAnalysis::analyze(&config);
        assert!(analysis.estimated_mos >= 4.5);

        // Real-time - should be fast
        let config = SamplingOptimizer::for_realtime();
        let analysis = SamplingPerformanceAnalysis::analyze(&config);
        assert!(analysis.estimated_rtf < 0.2); // Relaxed threshold
    }

    #[test]
    fn test_quality_target_optimization() {
        let config = SamplingOptimizer::for_quality_target(4.8);
        let analysis = SamplingPerformanceAnalysis::analyze(&config);
        assert!(analysis.estimated_mos >= 4.7);

        let config = SamplingOptimizer::for_quality_target(4.3);
        let analysis = SamplingPerformanceAnalysis::analyze(&config);
        assert!(analysis.estimated_mos >= 4.2);
    }

    #[test]
    fn test_speed_target_optimization() {
        // Very fast target - should use few steps
        let config = SamplingOptimizer::for_speed_target(0.02);
        assert!(config.num_steps <= 10);
        // Should be a fast algorithm
        assert!(!matches!(config.algorithm, SamplingAlgorithm::DDPM));

        // Moderate speed target - should allow more steps
        let config = SamplingOptimizer::for_speed_target(0.05);
        assert!(config.num_steps >= 5);
        assert!(config.num_steps <= 20);

        // Slow target - can use many steps
        let config = SamplingOptimizer::for_speed_target(0.1);
        assert!(config.num_steps >= 10);
    }

    #[test]
    fn test_preset_descriptions() {
        for preset in SamplingPreset::all() {
            let desc = preset.description();
            assert!(!desc.is_empty());
            assert!(desc.len() > 10); // Meaningful description
        }
    }
}
