//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::traits::{EvaluationResult, QualityScore};
use async_trait::async_trait;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::types::{
    AgeGroup, AiNaturalnessMetrics, CulturalBackground, ExperienceLevel, Gender, HearingAbility,
    HumanAiNaturalnessCorrelationConfig, HumanAiNaturalnessCorrelationEvaluator,
    HumanAiNaturalnessCorrelationResult, HumanNaturalnessRating, RaterDemographics,
};

/// Human-AI naturalness correlation evaluation trait
#[async_trait]
pub trait HumanAiNaturalnessCorrelationEvaluationTrait {
    /// Evaluate human-AI naturalness correlation
    async fn evaluate_human_ai_naturalness_correlation(
        &mut self,
        audio_id: &str,
        audio: &AudioBuffer,
        human_ratings: &[HumanNaturalnessRating],
        language: LanguageCode,
    ) -> EvaluationResult<HumanAiNaturalnessCorrelationResult>;
    /// Get cached human ratings
    fn get_cached_human_ratings(&self, audio_id: &str) -> Option<&Vec<HumanNaturalnessRating>>;
    /// Get cached AI metrics
    fn get_cached_ai_metrics(&self, audio_id: &str) -> Option<&AiNaturalnessMetrics>;
    /// Clear cache
    fn clear_cache(&mut self);
}
