//! Main audio quality researcher implementation

use crate::Error;
use std::collections::HashMap;

use super::config::ResearchConfig;
use super::neural_model::NeuralQualityModel;
use super::types::ComprehensiveQualityAnalysis;

/// Main audio quality researcher
pub struct AudioQualityResearcher {
    /// Research configuration
    pub(crate) config: ResearchConfig,
    /// Neural quality model
    pub(crate) neural_model: NeuralQualityModel,
    /// Analysis cache for performance
    pub(crate) analysis_cache: HashMap<String, ComprehensiveQualityAnalysis>,
    /// Statistics tracking
    pub(crate) analysis_count: usize,
}

impl AudioQualityResearcher {
    /// Create a new audio quality researcher
    pub fn new(config: ResearchConfig) -> Result<Self, Error> {
        let neural_model = NeuralQualityModel::default();

        Ok(Self {
            config,
            neural_model,
            analysis_cache: HashMap::new(),
            analysis_count: 0,
        })
    }

    /// Get analysis statistics
    pub fn get_analysis_count(&self) -> usize {
        self.analysis_count
    }

    /// Clear analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }
}
