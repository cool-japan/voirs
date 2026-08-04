//! Pipeline state management and synchronization.

use crate::{config::PipelineConfig, error::Result, types::VoiceConfig, VoirsError};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Pipeline state manager
#[derive(Clone)]
pub struct PipelineStateManager {
    /// Current pipeline state
    state: Arc<RwLock<PipelineState>>,

    /// Configuration state
    config: Arc<RwLock<PipelineConfig>>,

    /// Current voice configuration
    current_voice: Arc<RwLock<Option<VoiceConfig>>>,

    /// Component states
    component_states: Arc<RwLock<ComponentStates>>,

    /// Test mode flag
    test_mode: bool,
}

impl PipelineStateManager {
    /// Create new state manager
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(PipelineState::Initializing)),
            config: Arc::new(RwLock::new(config)),
            current_voice: Arc::new(RwLock::new(None)),
            component_states: Arc::new(RwLock::new(ComponentStates::default())),
            test_mode: cfg!(test),
        }
    }

    /// Create new state manager with test mode
    pub fn with_test_mode(config: PipelineConfig, test_mode: bool) -> Self {
        Self {
            state: Arc::new(RwLock::new(PipelineState::Initializing)),
            config: Arc::new(RwLock::new(config)),
            current_voice: Arc::new(RwLock::new(None)),
            component_states: Arc::new(RwLock::new(ComponentStates::default())),
            test_mode,
        }
    }

    /// Get current pipeline state
    pub async fn get_state(&self) -> PipelineState {
        *self.state.read().await
    }

    /// Set pipeline state
    pub async fn set_state(&self, new_state: PipelineState) -> Result<()> {
        let mut state = self.state.write().await;
        let old_state = *state;

        // Validate state transition
        if !self.is_valid_state_transition(old_state, new_state) {
            return Err(VoirsError::InvalidStateTransition {
                from: format!("{old_state:?}"),
                to: format!("{new_state:?}"),
                reason: "Invalid state transition".to_string(),
            });
        }

        *state = new_state;
        info!("Pipeline state changed: {:?} -> {:?}", old_state, new_state);

        // Handle state change effects
        self.handle_state_change(old_state, new_state).await?;

        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> PipelineConfig {
        self.config.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config(&self, new_config: PipelineConfig) -> Result<()> {
        let mut config = self.config.write().await;
        let old_config = config.clone();

        // Validate configuration update
        self.validate_config_update(&old_config, &new_config)
            .await?;

        *config = new_config;
        info!("Pipeline configuration updated");

        // Handle configuration change effects
        self.handle_config_change(&old_config, &config).await?;

        Ok(())
    }

    /// Get current voice
    pub async fn get_current_voice(&self) -> Option<VoiceConfig> {
        self.current_voice.read().await.clone()
    }

    /// Set current voice
    pub async fn set_current_voice(&self, voice: Option<VoiceConfig>) -> Result<()> {
        let mut current_voice = self.current_voice.write().await;
        let old_voice = current_voice.clone();

        *current_voice = voice.clone();

        if let Some(voice) = &voice {
            info!("Voice changed to: {} ({})", voice.name, voice.id);
        } else {
            info!("Voice cleared");
        }

        // Handle voice change effects
        self.handle_voice_change(&old_voice, &voice).await?;

        Ok(())
    }

    /// Get component states
    pub async fn get_component_states(&self) -> ComponentStates {
        self.component_states.read().await.clone()
    }

    /// Update component state
    pub async fn update_component_state(
        &self,
        component: ComponentType,
        state: ComponentState,
    ) -> Result<()> {
        let mut states = self.component_states.write().await;

        match component {
            ComponentType::G2p => states.g2p = state,
            ComponentType::Acoustic => states.acoustic = state,
            ComponentType::Vocoder => states.vocoder = state,
        }

        debug!("Component {:?} state updated to {:?}", component, state);

        // Check if all components are ready
        if states.all_ready() {
            drop(states); // Release lock before calling set_state
            if self.get_state().await == PipelineState::Initializing {
                self.set_state(PipelineState::Ready).await?;
            }
        }

        Ok(())
    }

    /// Synchronize all component states
    pub async fn synchronize_components(&self) -> Result<()> {
        // Skip expensive synchronization in test mode
        if self.test_mode {
            debug!("Skipping component synchronization in test mode");
            return Ok(());
        }

        info!("Synchronizing component states");

        let states = self.component_states.read().await;

        // Check component health
        if !states.all_healthy() {
            warn!("Some components are not healthy");
            return Err(VoirsError::ComponentSynchronizationFailed {
                component: "pipeline".to_string(),
                reason: "Unhealthy components detected".to_string(),
            });
        }

        // Synchronize component configurations
        self.synchronize_component_configs().await?;

        info!("Component synchronization complete");
        Ok(())
    }

    /// Cleanup resources
    pub async fn cleanup(&self) -> Result<()> {
        info!("Cleaning up pipeline resources");

        // Set state to shutting down
        self.set_state(PipelineState::ShuttingDown).await?;

        // Clear current voice
        self.set_current_voice(None).await?;

        // Reset component states
        let mut states = self.component_states.write().await;
        *states = ComponentStates::default();

        // Set final state
        self.set_state(PipelineState::Shutdown).await?;

        info!("Pipeline cleanup complete");
        Ok(())
    }

    /// Check if state transition is valid
    fn is_valid_state_transition(&self, from: PipelineState, to: PipelineState) -> bool {
        use PipelineState::*;

        match (from, to) {
            (Initializing, Ready) => true,
            (Initializing, Error) => true,
            (Ready, Busy) => true,
            (Ready, Error) => true,
            (Ready, ShuttingDown) => true,
            (Busy, Ready) => true,
            (Busy, Error) => true,
            (Error, Ready) => true,
            (Error, ShuttingDown) => true,
            (ShuttingDown, Shutdown) => true,
            (_, _) => false,
        }
    }

    /// Handle state change effects
    async fn handle_state_change(
        &self,
        old_state: PipelineState,
        new_state: PipelineState,
    ) -> Result<()> {
        use PipelineState::*;

        match (old_state, new_state) {
            (Initializing, Ready) => {
                info!("Pipeline is now ready for synthesis");
            }
            (Ready, Busy) => {
                debug!("Pipeline is now busy with synthesis");
            }
            (Busy, Ready) => {
                debug!("Pipeline synthesis complete, back to ready");
            }
            (_, Error) => {
                warn!("Pipeline entered error state");
            }
            (_, ShuttingDown) => {
                info!("Pipeline is shutting down");
            }
            (ShuttingDown, Shutdown) => {
                info!("Pipeline shutdown complete");
            }
            _ => {}
        }

        Ok(())
    }

    /// Validate configuration update
    async fn validate_config_update(
        &self,
        _old_config: &PipelineConfig,
        new_config: &PipelineConfig,
    ) -> Result<()> {
        // Validate device availability
        if !self.is_device_available(&new_config.device) {
            return Err(VoirsError::InvalidConfiguration {
                field: "device".to_string(),
                value: new_config.device.clone(),
                reason: "Device not available".to_string(),
                valid_values: None,
            });
        }

        // Validate cache directory
        if let Some(cache_dir) = &new_config.cache_dir {
            if !cache_dir.exists() {
                return Err(VoirsError::InvalidConfiguration {
                    field: "cache_dir".to_string(),
                    value: cache_dir.display().to_string(),
                    reason: "Cache directory does not exist".to_string(),
                    valid_values: None,
                });
            }
        }

        Ok(())
    }

    /// Handle configuration change effects
    async fn handle_config_change(
        &self,
        old_config: &PipelineConfig,
        new_config: &PipelineConfig,
    ) -> Result<()> {
        let mut needs_reinitialization = false;

        // Check if device changed
        if old_config.device != new_config.device {
            info!(
                "Device changed: {} -> {}",
                old_config.device, new_config.device
            );
            needs_reinitialization = true;
        }

        // Check if GPU setting changed
        if old_config.use_gpu != new_config.use_gpu {
            info!(
                "GPU setting changed: {} -> {}",
                old_config.use_gpu, new_config.use_gpu
            );
            needs_reinitialization = true;
        }

        // Check if cache directory changed
        if old_config.cache_dir != new_config.cache_dir {
            info!("Cache directory changed, triggering reinitialization");
            needs_reinitialization = true;
        }

        // Check if model configuration changed
        if old_config.model_loading != new_config.model_loading {
            info!("Model configuration changed, triggering reinitialization");
            needs_reinitialization = true;
        }

        // Trigger component reinitialization if needed
        if needs_reinitialization {
            self.trigger_component_reinitialization().await?;
        }

        Ok(())
    }

    /// Trigger component reinitialization
    async fn trigger_component_reinitialization(&self) -> Result<()> {
        info!("Triggering component reinitialization");

        // Set pipeline state to initializing
        self.set_state(PipelineState::Initializing).await?;

        // Reset all component states to uninitialized
        {
            let mut states = self.component_states.write().await;
            states.g2p = ComponentState::Uninitialized;
            states.acoustic = ComponentState::Uninitialized;
            states.vocoder = ComponentState::Uninitialized;
            states.last_update = SystemTime::now();
        }

        // Components would be reinitialized by the pipeline initialization process
        // This would typically be handled by the pipeline builder or initializer
        info!("Component reinitialization triggered - components will be reloaded");

        Ok(())
    }

    /// Handle voice change effects
    async fn handle_voice_change(
        &self,
        old_voice: &Option<VoiceConfig>,
        new_voice: &Option<VoiceConfig>,
    ) -> Result<()> {
        match (old_voice, new_voice) {
            (None, Some(voice)) => {
                info!("Voice set to: {}", voice.name);
            }
            (Some(old), Some(new)) if old.id != new.id => {
                info!("Voice changed: {} -> {}", old.name, new.name);
            }
            (Some(_), None) => {
                info!("Voice cleared");
            }
            _ => {}
        }

        Ok(())
    }

    /// Synchronize component configurations
    async fn synchronize_component_configs(&self) -> Result<()> {
        let config = self.config.read().await;

        debug!(
            "Synchronizing component configurations with device: {}",
            config.device
        );

        // Synchronize G2P component configuration
        if let Err(e) = self.synchronize_g2p_config(&config).await {
            warn!("Failed to synchronize G2P configuration: {}", e);
        }

        // Synchronize acoustic model configuration
        if let Err(e) = self.synchronize_acoustic_config(&config).await {
            warn!("Failed to synchronize acoustic configuration: {}", e);
        }

        // Synchronize vocoder configuration
        if let Err(e) = self.synchronize_vocoder_config(&config).await {
            warn!("Failed to synchronize vocoder configuration: {}", e);
        }

        debug!("Component configuration synchronization completed");
        Ok(())
    }

    /// Synchronize G2P component configuration
    async fn synchronize_g2p_config(&self, config: &PipelineConfig) -> Result<()> {
        debug!("Synchronizing G2P configuration");

        // Update G2P component state to loading
        self.update_component_state(ComponentType::G2p, ComponentState::Loading)
            .await?;

        // Simulate configuration synchronization (skip in test mode)
        if !self.test_mode {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Update G2P component state to ready
        self.update_component_state(ComponentType::G2p, ComponentState::Ready)
            .await?;

        debug!(
            "G2P configuration synchronized for device: {}",
            config.device
        );
        Ok(())
    }

    /// Synchronize acoustic model configuration
    async fn synchronize_acoustic_config(&self, config: &PipelineConfig) -> Result<()> {
        debug!("Synchronizing acoustic model configuration");

        // Update acoustic component state to loading
        self.update_component_state(ComponentType::Acoustic, ComponentState::Loading)
            .await?;

        // Simulate configuration synchronization (skip in test mode)
        if !self.test_mode {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Update acoustic component state to ready
        self.update_component_state(ComponentType::Acoustic, ComponentState::Ready)
            .await?;

        debug!(
            "Acoustic model configuration synchronized for device: {}",
            config.device
        );
        Ok(())
    }

    /// Synchronize vocoder configuration
    async fn synchronize_vocoder_config(&self, config: &PipelineConfig) -> Result<()> {
        debug!("Synchronizing vocoder configuration");

        // Update vocoder component state to loading
        self.update_component_state(ComponentType::Vocoder, ComponentState::Loading)
            .await?;

        // Simulate configuration synchronization (skip in test mode)
        if !self.test_mode {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Update vocoder component state to ready
        self.update_component_state(ComponentType::Vocoder, ComponentState::Ready)
            .await?;

        debug!(
            "Vocoder configuration synchronized for device: {}",
            config.device
        );
        Ok(())
    }

    /// Check if device is available
    fn is_device_available(&self, device: &str) -> bool {
        match device {
            "cpu" => true,
            "cuda" => self.is_cuda_available(),
            "metal" => self.is_metal_available(),
            "opencl" => self.is_opencl_available(),
            _ => false,
        }
    }

    /// Check if CUDA is available
    ///
    /// The underlying `nvidia-smi` probe is executed at most once per
    /// process (cached in a [`OnceLock`]) and under a hard timeout, so this
    /// can never hang the caller even if the subprocess itself does - see
    /// [`crate::process_probe::run_with_timeout`].
    fn is_cuda_available(&self) -> bool {
        // Skip expensive system calls in test mode
        if self.test_mode {
            debug!("Skipping CUDA availability check in test mode");
            return false;
        }

        static CUDA_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *CUDA_AVAILABLE.get_or_init(probe_cuda_available)
    }

    /// Check if Metal is available (macOS)
    ///
    /// Cached and timeout-guarded for the same reason as
    /// [`Self::is_cuda_available`]: the `system_profiler` probe it used to
    /// run unguarded is known to occasionally take a very long time (or hang)
    /// on macOS.
    fn is_metal_available(&self) -> bool {
        // Skip expensive system calls in test mode
        if self.test_mode {
            debug!("Skipping Metal availability check in test mode");
            return false;
        }

        static METAL_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *METAL_AVAILABLE.get_or_init(probe_metal_available)
    }

    /// Check if OpenCL is available
    ///
    /// Cached and timeout-guarded for the same reason as
    /// [`Self::is_cuda_available`].
    fn is_opencl_available(&self) -> bool {
        // Skip expensive system calls in test mode
        if self.test_mode {
            debug!("Skipping OpenCL availability check in test mode");
            return false;
        }

        static OPENCL_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *OPENCL_AVAILABLE.get_or_init(probe_opencl_available)
    }
}

/// Probe CUDA availability by running `nvidia-smi` under a hard timeout.
/// Called at most once per process; the result is cached by the caller.
fn probe_cuda_available() -> bool {
    let mut command = std::process::Command::new("nvidia-smi");
    match crate::process_probe::run_with_timeout(
        &mut command,
        crate::process_probe::DEFAULT_PROBE_TIMEOUT,
    ) {
        Ok(Some(output)) => output.status.success(),
        Ok(None) => {
            warn!("nvidia-smi did not respond within the probe timeout; assuming CUDA unavailable");
            false
        }
        Err(_) => false,
    }
}

/// Probe Metal availability by running `system_profiler` under a hard
/// timeout. Called at most once per process; the result is cached by the
/// caller.
fn probe_metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Metal is available on macOS with Apple Silicon or discrete GPUs
        let mut command = std::process::Command::new("system_profiler");
        command.args(["SPDisplaysDataType", "-detailLevel", "mini"]);
        match crate::process_probe::run_with_timeout(
            &mut command,
            crate::process_probe::DEFAULT_PROBE_TIMEOUT,
        ) {
            Ok(Some(output)) => {
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains("Metal")
            }
            Ok(None) => {
                warn!(
                    "system_profiler did not respond within the probe timeout; \
                     assuming Metal unavailable"
                );
                false
            }
            Err(_) => false,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Probe OpenCL availability by running `clinfo` under a hard timeout, with a
/// filesystem-based fallback if `clinfo` cannot be spawned at all. Called at
/// most once per process; the result is cached by the caller.
fn probe_opencl_available() -> bool {
    let mut command = std::process::Command::new("clinfo");
    match crate::process_probe::run_with_timeout(
        &mut command,
        crate::process_probe::DEFAULT_PROBE_TIMEOUT,
    ) {
        Ok(Some(output)) => output.status.success(),
        Ok(None) => {
            warn!("clinfo did not respond within the probe timeout; assuming OpenCL unavailable");
            false
        }
        Err(_) => {
            // Fallback: check for OpenCL library existence
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1").exists()
                    || std::path::Path::new("/usr/lib/libOpenCL.so.1").exists()
            }
            #[cfg(target_os = "macos")]
            {
                std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
            }
            #[cfg(target_os = "windows")]
            {
                std::path::Path::new("C:\\Windows\\System32\\OpenCL.dll").exists()
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                false
            }
        }
    }
}

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    /// Pipeline is initializing
    Initializing,
    /// Pipeline is ready for synthesis
    Ready,
    /// Pipeline is busy with synthesis
    Busy,
    /// Pipeline is in error state
    Error,
    /// Pipeline is shutting down
    ShuttingDown,
    /// Pipeline is shutdown
    Shutdown,
}

/// Component type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentType {
    G2p,
    Acoustic,
    Vocoder,
}

/// Component state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentState {
    /// Component is uninitialized
    Uninitialized,
    /// Component is loading
    Loading,
    /// Component is ready
    Ready,
    /// Component is busy
    Busy,
    /// Component is in error state
    Error,
    /// Component is shutting down
    ShuttingDown,
}

/// Component states
#[derive(Debug, Clone)]
pub struct ComponentStates {
    pub g2p: ComponentState,
    pub acoustic: ComponentState,
    pub vocoder: ComponentState,
    pub last_update: SystemTime,
}

impl Default for ComponentStates {
    fn default() -> Self {
        Self {
            g2p: ComponentState::Uninitialized,
            acoustic: ComponentState::Uninitialized,
            vocoder: ComponentState::Uninitialized,
            last_update: SystemTime::now(),
        }
    }
}

impl ComponentStates {
    /// Check if all components are ready
    pub fn all_ready(&self) -> bool {
        matches!(
            (self.g2p, self.acoustic, self.vocoder),
            (
                ComponentState::Ready,
                ComponentState::Ready,
                ComponentState::Ready
            )
        )
    }

    /// Check if all components are healthy
    pub fn all_healthy(&self) -> bool {
        !matches!(
            (self.g2p, self.acoustic, self.vocoder),
            (ComponentState::Error, _, _)
                | (_, ComponentState::Error, _)
                | (_, _, ComponentState::Error)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_manager_creation() {
        let config = PipelineConfig::default();
        let state_manager = PipelineStateManager::new(config);

        assert_eq!(state_manager.get_state().await, PipelineState::Initializing);
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let config = PipelineConfig::default();
        let state_manager = PipelineStateManager::new(config);

        // Valid transition
        let result = state_manager.set_state(PipelineState::Ready).await;
        assert!(result.is_ok());
        assert_eq!(state_manager.get_state().await, PipelineState::Ready);

        // Invalid transition
        let result = state_manager.set_state(PipelineState::Shutdown).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_component_state_management() {
        let config = PipelineConfig::default();
        let state_manager = PipelineStateManager::new(config);

        // Update component states
        state_manager
            .update_component_state(ComponentType::G2p, ComponentState::Ready)
            .await
            .unwrap();
        state_manager
            .update_component_state(ComponentType::Acoustic, ComponentState::Ready)
            .await
            .unwrap();
        state_manager
            .update_component_state(ComponentType::Vocoder, ComponentState::Ready)
            .await
            .unwrap();

        // Check that pipeline state was updated
        assert_eq!(state_manager.get_state().await, PipelineState::Ready);

        let states = state_manager.get_component_states().await;
        assert!(states.all_ready());
    }

    #[tokio::test]
    async fn test_voice_management() {
        let config = PipelineConfig::default();
        let state_manager = PipelineStateManager::new(config);

        // Set voice
        let voice = VoiceConfig {
            id: "test-voice".to_string(),
            name: "Test Voice".to_string(),
            language: crate::types::LanguageCode::EnUs,
            characteristics: Default::default(),
            model_config: Default::default(),
            metadata: Default::default(),
        };

        state_manager
            .set_current_voice(Some(voice.clone()))
            .await
            .unwrap();

        let current = state_manager.get_current_voice().await;
        assert!(current.is_some());
        assert_eq!(current.unwrap().id, "test-voice");
    }

    #[tokio::test]
    async fn test_cleanup() {
        let config = PipelineConfig::default();
        let state_manager = PipelineStateManager::new(config);

        // Set to ready state first
        state_manager.set_state(PipelineState::Ready).await.unwrap();

        // Cleanup
        state_manager.cleanup().await.unwrap();

        assert_eq!(state_manager.get_state().await, PipelineState::Shutdown);
        assert!(state_manager.get_current_voice().await.is_none());
    }

    /// Regression test for a hang that used to be reachable through
    /// `VoirsPipelineBuilder::build()` -> `update_config` ->
    /// `validate_config_update` -> `is_device_available`: the device-probe
    /// helpers shelled out to `nvidia-smi` / `system_profiler` / `clinfo`
    /// with no timeout, so a slow or hung subprocess could block the calling
    /// thread indefinitely.
    ///
    /// `PipelineStateManager` always runs with `test_mode = cfg!(test)`
    /// inside `#[cfg(test)]`, which makes the public `is_*_available`
    /// wrappers short-circuit before ever touching a subprocess. To actually
    /// exercise the timeout-guarded probes, this test calls the private
    /// `probe_*_available` free functions directly (bypassing the test-mode
    /// gate) on background threads and requires each to report back within a
    /// generous bound, proving the underlying `Command` calls can never hang
    /// the caller even when the tool is missing, slow, or wedged.
    #[test]
    fn device_probe_functions_return_within_bounded_time() {
        assert_probe_returns_within_bound("cuda", probe_cuda_available);
        assert_probe_returns_within_bound("metal", probe_metal_available);
        assert_probe_returns_within_bound("opencl", probe_opencl_available);
    }

    /// Run `probe` on a background thread and panic unless it reports back
    /// within 10 seconds.
    fn assert_probe_returns_within_bound(name: &str, probe: fn() -> bool) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Ignore send errors: if the receiver already timed out and was
            // dropped, there is nothing left to report to.
            let _ = tx.send(probe());
        });

        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(_available) => {}
            Err(_) => panic!(
                "{name} availability probe did not return within the 10s bound; \
                 the subprocess timeout guard has regressed"
            ),
        }
    }
}
