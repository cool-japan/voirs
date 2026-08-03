//! VoiRS synthesis pipeline implementation.
//!
//! This module provides a modular pipeline architecture with:
//! - Component initialization and management
//! - Synthesis orchestration
//! - State management and synchronization

use super::{init, state, synthesis};

use crate::{
    audio::AudioBuffer,
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder, VoiceManager},
    types::{SynthesisConfig, VoiceConfig},
    voice::DefaultVoiceManager,
    VoirsError,
};
use std::sync::Arc;
use tokio::sync::RwLock;

use init::{ComponentOverrides, PipelineInitializer};
use state::{ComponentStates, PipelineState, PipelineStateManager};
use synthesis::SynthesisOrchestrator;

// Re-export types for external use
pub use state::{
    ComponentState as PublicComponentState, ComponentStates as PublicComponentStates,
    ComponentType as PublicComponentType, PipelineState as PublicPipelineState,
};

/// The three synthesis components currently installed in a pipeline.
///
/// Held behind an `RwLock` so that [`VoirsPipeline::set_voice`] can genuinely
/// swap them when the active voice changes.
#[derive(Clone)]
struct PipelineComponents {
    g2p: Arc<dyn G2p>,
    acoustic: Arc<dyn AcousticModel>,
    vocoder: Arc<dyn Vocoder>,
}

/// Main VoiRS synthesis pipeline
#[derive(Clone)]
pub struct VoirsPipeline {
    /// Currently installed synthesis components
    components: Arc<RwLock<PipelineComponents>>,

    /// Whether stub components are in use (explicit test mode)
    test_mode: bool,

    /// State manager
    state_manager: PipelineStateManager,

    /// Pipeline configuration
    config: Arc<RwLock<PipelineConfig>>,

    /// Current voice configuration
    current_voice: Arc<RwLock<Option<VoiceConfig>>>,

    /// Voice registry / manager used to resolve voice IDs
    voice_manager: Arc<RwLock<DefaultVoiceManager>>,
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
        Self::with_test_mode(g2p, acoustic, vocoder, config, false)
    }

    /// Create pipeline with components and test mode
    pub fn with_test_mode(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: PipelineConfig,
        test_mode: bool,
    ) -> Self {
        let mut voice_manager = DefaultVoiceManager::new(config.effective_cache_dir());
        voice_manager.set_test_mode(test_mode);

        Self::with_components(
            g2p,
            acoustic,
            vocoder,
            config,
            test_mode,
            Arc::new(RwLock::new(voice_manager)),
        )
    }

    /// Create pipeline with an explicit voice manager
    fn with_components(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: PipelineConfig,
        test_mode: bool,
        voice_manager: Arc<RwLock<DefaultVoiceManager>>,
    ) -> Self {
        let state_manager = PipelineStateManager::with_test_mode(config.clone(), test_mode);

        Self {
            components: Arc::new(RwLock::new(PipelineComponents {
                g2p,
                acoustic,
                vocoder,
            })),
            test_mode,
            state_manager,
            config: Arc::new(RwLock::new(config)),
            current_voice: Arc::new(RwLock::new(None)),
            voice_manager,
        }
    }

    /// Initialize pipeline from builder
    pub async fn from_builder(builder: super::VoirsPipelineBuilder) -> Result<Self> {
        Self::from_builder_core(&builder).await
    }

    /// Initialize pipeline from builder core (for advanced features)
    ///
    /// This is the single real component-resolution path of the SDK: it runs
    /// [`PipelineInitializer`], honoring any custom components injected through
    /// `with_g2p` / `with_acoustic_model` / `with_vocoder`, and honoring the
    /// explicit test-mode opt-in.
    pub async fn from_builder_core(builder: &super::VoirsPipelineBuilder) -> Result<Self> {
        let mut config = builder.get_config();
        let test_mode = builder.get_test_mode();

        // Reuse the caller-provided voice manager when present so that custom
        // voice registries are visible to `set_voice`.
        let voice_manager = match builder.get_voice_manager() {
            Some(manager) => {
                if test_mode {
                    manager.write().await.set_test_mode(true);
                }
                manager
            }
            None => {
                let mut manager = DefaultVoiceManager::new(config.effective_cache_dir());
                manager.set_test_mode(test_mode);
                Arc::new(RwLock::new(manager))
            }
        };

        // Resolve the requested voice up front so that the components are loaded
        // for that voice directly (language and per-voice model files) instead of
        // being loaded once and then thrown away.
        let selected_voice = match builder.get_voice_id() {
            Some(voice_id) => {
                let manager = voice_manager.read().await;
                let fallback_language = config.default_synthesis.language;
                let voice = resolve_voice_in(&manager, &voice_id, fallback_language).await?;
                apply_voice_to_config(&mut config, &voice);
                Some(voice)
            }
            None => None,
        };

        let initializer = PipelineInitializer::new(config.clone());

        let overrides = ComponentOverrides {
            g2p: builder.get_custom_g2p(),
            acoustic: builder.get_custom_acoustic(),
            vocoder: builder.get_custom_vocoder(),
        };

        // Initialize components
        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(overrides, test_mode)
            .await?;

        // Create pipeline with test mode
        let pipeline = Self::with_components(
            g2p,
            acoustic,
            vocoder,
            config,
            test_mode,
            voice_manager.clone(),
        );

        // Record the already-loaded voice without reloading the components again.
        if let Some(voice) = selected_voice {
            pipeline
                .state_manager
                .set_current_voice(Some(voice.clone()))
                .await?;
            voice_manager
                .write()
                .await
                .set_current_voice(Some(voice.id.clone()))?;
            *pipeline.current_voice.write().await = Some(voice);
        }

        // Update state to ready
        pipeline
            .state_manager
            .set_state(PipelineState::Ready)
            .await?;

        Ok(pipeline)
    }

    /// Build a synthesis orchestrator over the currently installed components
    async fn orchestrator(&self) -> SynthesisOrchestrator {
        let components = self.components.read().await;
        SynthesisOrchestrator::with_test_mode(
            components.g2p.clone(),
            components.acoustic.clone(),
            components.vocoder.clone(),
            self.test_mode,
        )
    }

    /// Get the currently installed G2P component
    pub(crate) async fn g2p(&self) -> Arc<dyn G2p> {
        self.components.read().await.g2p.clone()
    }

    /// Get the currently installed acoustic model
    pub(crate) async fn acoustic(&self) -> Arc<dyn AcousticModel> {
        self.components.read().await.acoustic.clone()
    }

    /// Get the currently installed vocoder
    pub(crate) async fn vocoder(&self) -> Arc<dyn Vocoder> {
        self.components.read().await.vocoder.clone()
    }

    /// Synthesize text to audio
    pub async fn synthesize(&self, text: &str) -> Result<AudioBuffer> {
        self.synthesize_with_config(text, &SynthesisConfig::default())
            .await
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
        let result = self.orchestrator().await.synthesize(text, config).await;

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
        let result = self
            .orchestrator()
            .await
            .synthesize_ssml(ssml, &config)
            .await;

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
    ) -> Result<impl futures::Stream<Item = Result<AudioBuffer>> + 'static> {
        // Check pipeline state
        if self.state_manager.get_state().await != PipelineState::Ready {
            return Err(VoirsError::PipelineNotReady);
        }

        let config = SynthesisConfig::default();
        self.orchestrator()
            .await
            .synthesize_stream(text, &config)
            .await
    }

    /// Change voice during runtime.
    ///
    /// The voice ID is resolved against the voice registry; unknown IDs fail with
    /// [`VoirsError::VoiceNotFound`]. When the resolved voice differs from the
    /// active one, the affected synthesis components are genuinely reloaded (the
    /// G2P backend follows the voice's language, and the acoustic/vocoder models
    /// are re-resolved from the voice's model configuration), so subsequent
    /// synthesis really uses the new voice.
    ///
    /// The literal ID `"default"` is an alias for the registry's default voice for
    /// the currently configured language.
    pub async fn set_voice(&self, voice_id: &str) -> Result<()> {
        let voice_config = self.resolve_voice(voice_id).await?;

        // Reload the components affected by the voice change.
        self.reload_components_for_voice(&voice_config).await?;

        // Update state manager
        self.state_manager
            .set_current_voice(Some(voice_config.clone()))
            .await?;

        // Track the active voice on the voice manager as well
        {
            let mut manager = self.voice_manager.write().await;
            manager.set_current_voice(Some(voice_config.id.clone()))?;
        }

        // Update internal voice
        let mut current = self.current_voice.write().await;
        *current = Some(voice_config);

        Ok(())
    }

    /// Resolve a voice ID against the registry
    async fn resolve_voice(&self, voice_id: &str) -> Result<VoiceConfig> {
        let fallback_language = self.config.read().await.default_synthesis.language;
        let manager = self.voice_manager.read().await;
        resolve_voice_in(&manager, voice_id, fallback_language).await
    }

    /// Reload the synthesis components for a newly selected voice.
    ///
    /// Components are rebuilt only when the voice actually changes what would be
    /// loaded (its language or its model files); otherwise the already-installed
    /// components remain correct and are kept.
    async fn reload_components_for_voice(&self, voice: &VoiceConfig) -> Result<()> {
        let current = self.config.read().await.clone();

        // Derive a configuration that targets the voice's language and models.
        let mut config = current.clone();
        apply_voice_to_config(&mut config, voice);

        if self.test_mode {
            // Stub components stay installed; only the resolved configuration and
            // the recorded voice change. This keeps explicit test mode fast and
            // never pretends real weights were loaded.
            *self.config.write().await = config;
            return Ok(());
        }

        if config == current {
            // Nothing that affects component loading changed.
            return Ok(());
        }

        let initializer = PipelineInitializer::new(config.clone());
        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(ComponentOverrides::default(), false)
            .await?;

        {
            let mut components = self.components.write().await;
            *components = PipelineComponents {
                g2p,
                acoustic,
                vocoder,
            };
        }
        *self.config.write().await = config;

        tracing::info!(
            "Reloaded pipeline components for voice '{}' (language {:?})",
            voice.id,
            voice.language
        );

        Ok(())
    }

    /// Get current voice information
    pub async fn current_voice(&self) -> Option<VoiceConfig> {
        self.current_voice.read().await.clone()
    }

    /// List available voices
    pub async fn list_voices(&self) -> Result<Vec<VoiceConfig>> {
        let manager = self.voice_manager.read().await;
        let mut voices = manager.list_voices().await?;

        // Mark voices as available/unavailable based on local model files
        for voice in &mut voices {
            if manager.is_voice_available(&voice.id) {
                voice
                    .metadata
                    .insert("status".to_string(), "available".to_string());
                voice
                    .metadata
                    .insert("location".to_string(), "local".to_string());
            } else {
                voice
                    .metadata
                    .insert("status".to_string(), "downloadable".to_string());
                voice
                    .metadata
                    .insert("location".to_string(), "remote".to_string());
            }
        }

        // Sort voices by language and name for consistent ordering
        voices.sort_by(|a, b| a.language.cmp(&b.language).then(a.name.cmp(&b.name)));

        tracing::debug!("Discovered {} voices", voices.len());
        Ok(voices)
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

/// Resolve a voice ID against a voice registry.
///
/// Unknown IDs produce [`VoirsError::VoiceNotFound`] listing the registered voices.
/// The literal ID `default` (case-insensitive) resolves to the registry default
/// for `fallback_language`.
async fn resolve_voice_in(
    manager: &DefaultVoiceManager,
    voice_id: &str,
    fallback_language: crate::types::LanguageCode,
) -> Result<VoiceConfig> {
    if let Some(voice) = manager.get_voice(voice_id).await? {
        return Ok(voice);
    }

    if voice_id.eq_ignore_ascii_case("default") {
        if let Some(default_id) = manager.default_voice_for_language(fallback_language) {
            if let Some(voice) = manager.get_voice(&default_id).await? {
                return Ok(voice);
            }
        }
    }

    let available = manager
        .list_voices()
        .await?
        .into_iter()
        .map(|voice| voice.id)
        .collect::<Vec<_>>();

    Err(VoirsError::voice_not_found(voice_id.to_string(), available))
}

/// Apply a resolved voice to a pipeline configuration: its language becomes the
/// synthesis language, and its model files (when they exist on disk) become the
/// local-path overrides used by the component loaders.
fn apply_voice_to_config(config: &mut PipelineConfig, voice: &VoiceConfig) {
    config.language_code = Some(voice.language);
    config.default_synthesis.language = voice.language;

    let models_dir = config.effective_cache_dir();
    let voice_dir = models_dir.join("voices").join(&voice.id);

    let acoustic_name = config
        .acoustic_model
        .clone()
        .unwrap_or_else(|| "candle".to_string());
    if let Some(path) = resolve_voice_model_path(
        &models_dir,
        &voice_dir,
        &voice.model_config.acoustic_model,
        "acoustic_model",
    ) {
        set_local_path_override(config, acoustic_name, path);
    }

    let vocoder_name = config
        .vocoder_model
        .clone()
        .unwrap_or_else(|| "hifigan".to_string());
    if let Some(path) = resolve_voice_model_path(
        &models_dir,
        &voice_dir,
        &voice.model_config.vocoder_model,
        "vocoder_model",
    ) {
        set_local_path_override(config, vocoder_name, path);
    }
}

/// Install a local-path override for a named model in the configuration
fn set_local_path_override(
    config: &mut PipelineConfig,
    model_name: String,
    path: std::path::PathBuf,
) {
    let entry = config
        .model_loading
        .model_overrides
        .entry(model_name)
        .or_insert_with(|| crate::config::ModelOverride {
            url: None,
            checksum: None,
            local_path: None,
            priority: None,
        });
    entry.local_path = Some(path);
}

/// Resolve a voice's model file to an existing path on disk, if any.
///
/// Candidates, in order: the declared path if absolute, the per-voice download
/// directory written by the voice downloader (`<cache>/voices/<id>/<stem>.*`), and
/// the declared path relative to the models directory.
fn resolve_voice_model_path(
    models_dir: &std::path::Path,
    voice_dir: &std::path::Path,
    declared_path: &str,
    downloaded_stem: &str,
) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(declared_path);
    if direct.is_absolute() && direct.exists() {
        return Some(direct.to_path_buf());
    }

    for extension in ["safetensors", "bin"] {
        let candidate = voice_dir.join(format!("{downloaded_stem}.{extension}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let joined = models_dir.join(declared_path);
    if joined.exists() {
        return Some(joined);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LanguageCode;

    fn stub_pipeline() -> VoirsPipeline {
        VoirsPipeline::with_test_mode(
            Arc::new(crate::pipeline::DummyG2p::new()),
            Arc::new(crate::pipeline::DummyAcoustic::new()),
            Arc::new(crate::pipeline::DummyVocoder::new()),
            PipelineConfig::default(),
            true,
        )
    }

    #[tokio::test]
    async fn test_set_voice_rejects_unknown_ids() {
        let pipeline = stub_pipeline();

        let result = pipeline.set_voice("definitely-not-a-real-voice").await;
        assert!(result.is_err(), "unknown voice IDs must not succeed");
        match result {
            Err(VoirsError::VoiceNotFound { voice, .. }) => {
                assert_eq!(voice, "definitely-not-a-real-voice");
            }
            other => panic!("expected VoiceNotFound, got {other:?}"),
        }

        // A rejected switch must not become the current voice.
        assert!(pipeline.current_voice().await.is_none());
    }

    #[tokio::test]
    async fn test_set_voice_resolves_registry_entry() {
        let pipeline = stub_pipeline();

        pipeline.set_voice("ja-JP-female-neutral").await.unwrap();

        let current = pipeline.current_voice().await.expect("voice set");
        assert_eq!(current.id, "ja-JP-female-neutral");
        // The real registry entry (not a fabricated placeholder) must be used.
        assert_eq!(current.language, LanguageCode::JaJp);
        assert_eq!(current.name, "Japanese Female Neutral");

        // The pipeline configuration follows the resolved voice's language.
        let config = pipeline.get_config().await;
        assert_eq!(config.language_code, Some(LanguageCode::JaJp));
        assert_eq!(config.default_synthesis.language, LanguageCode::JaJp);
    }

    #[tokio::test]
    async fn test_set_voice_default_alias() {
        let pipeline = stub_pipeline();

        pipeline.set_voice("default").await.unwrap();

        let current = pipeline.current_voice().await.expect("voice set");
        assert_eq!(current.language, LanguageCode::EnUs);
        assert_ne!(current.id, "default");
    }
}
