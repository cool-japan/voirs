//! VoiRS synthesis pipeline implementation.
//!
//! This module provides a modular pipeline architecture with:
//! - Component initialization and management
//! - Synthesis orchestration
//! - State management and synchronization

use super::{init, synthesis, state};

use crate::{
    audio::AudioBuffer,
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, SynthesisConfig, VoiceConfig},
    VoirsError,
};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::RwLock;

use init::PipelineInitializer;
use synthesis::SynthesisOrchestrator;
use state::{PipelineStateManager, PipelineState, ComponentType, ComponentState, ComponentStates};

// Re-export types for external use
pub use state::{PipelineState as PublicPipelineState, ComponentType as PublicComponentType, ComponentState as PublicComponentState, ComponentStates as PublicComponentStates};

/// Main VoiRS synthesis pipeline
#[derive(Clone)]
pub struct VoirsPipeline {
    /// Synthesis orchestrator
    orchestrator: SynthesisOrchestrator,
    
    /// State manager
    state_manager: PipelineStateManager,
    
    /// Pipeline configuration
    config: Arc<RwLock<PipelineConfig>>,
    
    /// Current voice configuration
    current_voice: Arc<RwLock<Option<VoiceConfig>>>,
}

impl VoirsPipeline {
    /// Create a new pipeline builder
    pub fn builder() -> super::VoirsPipelineBuilder {
        super::VoirsPipelineBuilder::new()
    }

    /// Create pipeline with components
    pub fn new(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: PipelineConfig,
    ) -> Self {
        let orchestrator = SynthesisOrchestrator::new(g2p, acoustic, vocoder);
        let state_manager = PipelineStateManager::new(config.clone());
        
        Self {
            orchestrator,
            state_manager,
            config: Arc::new(RwLock::new(config)),
            current_voice: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize pipeline from builder
    pub async fn from_builder(builder: super::VoirsPipelineBuilder) -> Result<Self> {
        let config = builder.get_config();
        let initializer = PipelineInitializer::new(config.clone());
        
        // Initialize components
        let (g2p, acoustic, vocoder) = initializer.initialize_components().await?;
        
        // Create pipeline
        let mut pipeline = Self::new(g2p, acoustic, vocoder, config);
        
        // Set voice if specified
        if let Some(voice_id) = builder.get_voice_id() {
            pipeline.set_voice(&voice_id).await?;
        }
        
        // Update state to ready
        pipeline.state_manager.set_state(PipelineState::Ready).await?;
        
        Ok(pipeline)
    }

    /// Synthesize text to audio
    pub async fn synthesize(&self, text: &str) -> Result<AudioBuffer> {
        self.synthesize_with_config(text, &SynthesisConfig::default()).await
    }

    /// Synthesize with custom configuration
    pub async fn synthesize_with_config(
        &self,
        text: &str,
        config: &SynthesisConfig,
    ) -> Result<AudioBuffer> {
        // Check pipeline state
        if self.state_manager.get_state().await != PipelineState::Ready {
            return Err(VoirsError::PipelineNotReady);
        }
        
        // Set state to busy
        self.state_manager.set_state(PipelineState::Busy).await?;
        
        // Perform synthesis
        let result = self.orchestrator.synthesize(text, config).await;
        
        // Reset state to ready
        match result {
            Ok(_) => {
                self.state_manager.set_state(PipelineState::Ready).await?;
            }
            Err(_) => {
                self.state_manager.set_state(PipelineState::Error).await?;
            }
        }
        
        result
    }

    /// Synthesize SSML markup
    pub async fn synthesize_ssml(&self, ssml: &str) -> Result<AudioBuffer> {
        let config = SynthesisConfig::default();
        
        // Check pipeline state
        if self.state_manager.get_state().await != PipelineState::Ready {
            return Err(VoirsError::PipelineNotReady);
        }
        
        // Set state to busy
        self.state_manager.set_state(PipelineState::Busy).await?;
        
        // Perform synthesis
        let result = self.orchestrator.synthesize_ssml(ssml, &config).await;
        
        // Reset state to ready
        match result {
            Ok(_) => {
                self.state_manager.set_state(PipelineState::Ready).await?;
            }
            Err(_) => {
                self.state_manager.set_state(PipelineState::Error).await?;
            }
        }
        
        result
    }

    /// Stream synthesis for long texts
    pub async fn synthesize_stream(
        self: Arc<Self>,
        text: &str,
    ) -> Result<impl futures::Stream<Item = Result<AudioBuffer>>> {
        // Check pipeline state
        if self.state_manager.get_state().await != PipelineState::Ready {
            return Err(VoirsError::PipelineNotReady);
        }
        
        let config = SynthesisConfig::default();
        self.orchestrator.synthesize_stream(text, &config).await
    }

    /// Change voice during runtime
    pub async fn set_voice(&self, voice_id: &str) -> Result<()> {
        // Create voice configuration
        let voice_config = VoiceConfig {
            id: voice_id.to_string(),
            name: voice_id.to_string(),
            language: LanguageCode::EnUs, // Default
            characteristics: Default::default(),
            model_config: Default::default(),
            metadata: Default::default(),
        };
        
        // Update state manager
        self.state_manager.set_current_voice(Some(voice_config.clone())).await?;
        
        // Update internal voice
        let mut current = self.current_voice.write().await;
        *current = Some(voice_config);
        
        Ok(())
    }

    /// Get current voice information
    pub async fn current_voice(&self) -> Option<VoiceConfig> {
        self.current_voice.read().await.clone()
    }

    /// List available voices
    pub async fn list_voices(&self) -> Result<Vec<VoiceConfig>> {
        // TODO: Implement voice discovery
        Ok(vec![
            VoiceConfig {
                id: "en-US-female-calm".to_string(),
                name: "English US Female Calm".to_string(),
                language: LanguageCode::EnUs,
                characteristics: Default::default(),
                model_config: Default::default(),
                metadata: Default::default(),
            },
            VoiceConfig {
                id: "en-US-male-news".to_string(),
                name: "English US Male News".to_string(),
                language: LanguageCode::EnUs,
                characteristics: Default::default(),
                model_config: Default::default(),
                metadata: Default::default(),
            },
        ])
    }

    /// Get pipeline state
    pub async fn get_state(&self) -> PipelineState {
        self.state_manager.get_state().await
    }

    /// Get pipeline configuration
    pub async fn get_config(&self) -> PipelineConfig {
        self.config.read().await.clone()
    }

    /// Update pipeline configuration
    pub async fn update_config(&self, new_config: PipelineConfig) -> Result<()> {
        // Update state manager
        self.state_manager.update_config(new_config.clone()).await?;
        
        // Update internal config
        let mut config = self.config.write().await;
        *config = new_config;
        
        Ok(())
    }

    /// Get component states
    pub async fn get_component_states(&self) -> ComponentStates {
        self.state_manager.get_component_states().await
    }

    /// Synchronize all components
    pub async fn synchronize_components(&self) -> Result<()> {
        self.state_manager.synchronize_components().await
    }

    /// Cleanup pipeline resources
    pub async fn cleanup(&self) -> Result<()> {
        self.state_manager.cleanup().await
    }

    /// Set pipeline state to ready
    pub async fn set_ready(&self) -> Result<()> {
        self.state_manager.set_state(PipelineState::Ready).await
    }
}

