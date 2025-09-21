//! # Browser Extension Integration
//!
//! This module provides browser extension integration capabilities for Chrome,
//! Firefox, Safari, and other browsers. It supports real-time pronunciation help,
//! web page content coaching, video call enhancement, and online learning platform
//! integration for the VoiRS feedback system.

use crate::realtime::types::RealtimeConfig;
use crate::traits::{FeedbackSession, UserProgress};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::{Duration, SystemTime};

/// Browser extension error types
#[derive(Debug, Clone)]
pub enum BrowserExtensionError {
    ExtensionNotInstalled(String),
    BrowserNotSupported(String),
    PermissionDenied(String),
    CommunicationError(String),
    ScriptInjectionFailed(String),
    ContentSecurityPolicyError(String),
    ManifestError(String),
    StorageError(String),
    NetworkError(String),
    ConfigurationError(String),
    ApiLimitExceeded,
    UnauthorizedAccess,
}

impl fmt::Display for BrowserExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrowserExtensionError::ExtensionNotInstalled(browser) => {
                write!(f, "Extension not installed in {}", browser)
            }
            BrowserExtensionError::BrowserNotSupported(browser) => {
                write!(f, "Browser not supported: {}", browser)
            }
            BrowserExtensionError::PermissionDenied(permission) => {
                write!(f, "Permission denied: {}", permission)
            }
            BrowserExtensionError::CommunicationError(msg) => {
                write!(f, "Communication error: {}", msg)
            }
            BrowserExtensionError::ScriptInjectionFailed(msg) => {
                write!(f, "Script injection failed: {}", msg)
            }
            BrowserExtensionError::ContentSecurityPolicyError(msg) => {
                write!(f, "Content Security Policy error: {}", msg)
            }
            BrowserExtensionError::ManifestError(msg) => write!(f, "Manifest error: {}", msg),
            BrowserExtensionError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            BrowserExtensionError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            BrowserExtensionError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            BrowserExtensionError::ApiLimitExceeded => write!(f, "API limit exceeded"),
            BrowserExtensionError::UnauthorizedAccess => write!(f, "Unauthorized access"),
        }
    }
}

impl Error for BrowserExtensionError {}

/// Supported browsers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrowserType {
    Chrome,
    Firefox,
    Safari,
    Edge,
    Opera,
    Brave,
    Custom(String),
}

impl fmt::Display for BrowserType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrowserType::Chrome => write!(f, "Google Chrome"),
            BrowserType::Firefox => write!(f, "Mozilla Firefox"),
            BrowserType::Safari => write!(f, "Safari"),
            BrowserType::Edge => write!(f, "Microsoft Edge"),
            BrowserType::Opera => write!(f, "Opera"),
            BrowserType::Brave => write!(f, "Brave"),
            BrowserType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Extension manifest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub manifest_version: u32,
    pub permissions: Vec<ExtensionPermission>,
    pub host_permissions: Vec<String>,
    pub content_scripts: Vec<ContentScript>,
    pub background: Option<BackgroundScript>,
    pub action: Option<BrowserAction>,
    pub options_page: Option<String>,
    pub web_accessible_resources: Vec<WebAccessibleResource>,
}

/// Extension permissions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtensionPermission {
    ActiveTab,
    Storage,
    Scripting,
    Notifications,
    AudioCapture,
    VideoCapture,
    TabCapture,
    DesktopCapture,
    Identity,
    Cookies,
    History,
    BookMarks,
    Tabs,
    ContextMenus,
    Background,
    OfflineDocument,
    UnlimitedStorage,
    Custom(String),
}

/// Content script configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentScript {
    pub matches: Vec<String>,
    pub js: Vec<String>,
    pub css: Vec<String>,
    pub run_at: RunAt,
    pub all_frames: bool,
    pub match_about_blank: bool,
}

/// Script execution timing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RunAt {
    DocumentStart,
    DocumentEnd,
    DocumentIdle,
}

/// Background script configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundScript {
    pub service_worker: Option<String>,
    pub scripts: Vec<String>,
    pub persistent: bool,
}

/// Browser action configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserAction {
    pub default_popup: Option<String>,
    pub default_title: String,
    pub default_icon: HashMap<String, String>,
}

/// Web accessible resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAccessibleResource {
    pub resources: Vec<String>,
    pub matches: Vec<String>,
    pub use_dynamic_url: bool,
}

/// Extension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfig {
    pub browser: BrowserType,
    pub extension_id: String,
    pub manifest: ExtensionManifest,
    pub api_endpoint: String,
    pub realtime_enabled: bool,
    pub auto_inject: bool,
    pub privacy_settings: PrivacySettings,
    pub ui_settings: UISettings,
}

/// Privacy settings for the extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    pub collect_browsing_data: bool,
    pub store_audio_locally: bool,
    pub send_anonymized_metrics: bool,
    pub respect_do_not_track: bool,
    pub incognito_mode_support: bool,
}

/// UI settings for the extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UISettings {
    pub show_pronunciation_hints: bool,
    pub highlight_difficult_words: bool,
    pub enable_tooltips: bool,
    pub floating_feedback_panel: bool,
    pub theme: ExtensionTheme,
    pub position: UIPosition,
    pub opacity: f32,
}

/// Extension theme
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtensionTheme {
    Light,
    Dark,
    Auto,
    Custom(String),
}

/// UI positioning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UIPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
    Custom { x: i32, y: i32 },
}

/// Page analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageAnalysis {
    pub url: String,
    pub title: String,
    pub text_content: String,
    pub difficult_words: Vec<DifficultWord>,
    pub learning_opportunities: Vec<LearningOpportunity>,
    pub pronunciation_hints: Vec<PronunciationHint>,
    pub estimated_reading_level: f32,
    pub language_detected: String,
    pub analysis_timestamp: SystemTime,
}

/// Difficult word identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultWord {
    pub word: String,
    pub difficulty_score: f32,
    pub position: TextPosition,
    pub phonetic_transcription: String,
    pub definition: Option<String>,
    pub usage_examples: Vec<String>,
    pub audio_url: Option<String>,
}

/// Text position in the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPosition {
    pub start_offset: usize,
    pub end_offset: usize,
    pub element_selector: String,
    pub element_type: String,
}

/// Learning opportunity on the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningOpportunity {
    pub opportunity_type: OpportunityType,
    pub description: String,
    pub action_required: String,
    pub estimated_benefit: f32,
    pub difficulty_level: f32,
    pub position: Option<TextPosition>,
}

/// Types of learning opportunities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpportunityType {
    PronunciationPractice,
    VocabularyExpansion,
    ReadingComprehension,
    ListeningPractice,
    SpeakingPractice,
    GrammarReview,
}

/// Pronunciation hint for specific text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationHint {
    pub text: String,
    pub phonetic: String,
    pub audio_url: Option<String>,
    pub position: TextPosition,
    pub confidence: f32,
    pub alternatives: Vec<String>,
}

/// Web page integration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPageIntegration {
    pub injected_scripts: Vec<String>,
    pub modified_elements: Vec<String>,
    pub created_overlays: Vec<UIOverlay>,
    pub event_listeners: Vec<EventListener>,
    pub storage_keys: Vec<String>,
}

/// UI overlay for the extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIOverlay {
    pub overlay_id: String,
    pub overlay_type: OverlayType,
    pub position: UIPosition,
    pub size: Size,
    pub content: String,
    pub visible: bool,
    pub z_index: i32,
}

/// Types of UI overlays
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OverlayType {
    FeedbackPanel,
    PronunciationTooltip,
    ProgressBar,
    NotificationBanner,
    SettingsPanel,
    HelpDialog,
}

/// Size specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

/// Event listener configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventListener {
    pub event_type: String,
    pub selector: String,
    pub handler_function: String,
    pub capture: bool,
}

/// Browser extension manager
pub struct BrowserExtensionManager {
    config: ExtensionConfig,
    realtime_config: RealtimeConfig,
    installed_extensions: HashMap<BrowserType, ExtensionInfo>,
    active_sessions: HashMap<String, ExtensionSession>,
}

/// Information about an installed extension
#[derive(Debug, Clone)]
struct ExtensionInfo {
    extension_id: String,
    version: String,
    enabled: bool,
    permissions: Vec<ExtensionPermission>,
    install_date: SystemTime,
}

/// Active extension session
#[derive(Debug, Clone)]
struct ExtensionSession {
    session_id: String,
    tab_id: String,
    url: String,
    start_time: SystemTime,
    page_analysis: Option<PageAnalysis>,
    integration: Option<WebPageIntegration>,
    feedback_count: u32,
}

impl BrowserExtensionManager {
    /// Create a new browser extension manager
    pub fn new(config: ExtensionConfig, realtime_config: RealtimeConfig) -> Self {
        Self {
            config,
            realtime_config,
            installed_extensions: HashMap::new(),
            active_sessions: HashMap::new(),
        }
    }

    /// Install extension for a specific browser
    pub async fn install_extension(
        &mut self,
        browser: &BrowserType,
    ) -> Result<String, BrowserExtensionError> {
        match browser {
            BrowserType::Chrome => self.install_chrome_extension().await,
            BrowserType::Firefox => self.install_firefox_extension().await,
            BrowserType::Safari => self.install_safari_extension().await,
            BrowserType::Edge => self.install_edge_extension().await,
            BrowserType::Opera => self.install_opera_extension().await,
            BrowserType::Brave => self.install_brave_extension().await,
            BrowserType::Custom(name) => self.install_custom_extension(name).await,
        }
    }

    /// Check if extension is installed and active
    pub fn is_extension_active(&self, browser: &BrowserType) -> bool {
        self.installed_extensions
            .get(browser)
            .map(|info| info.enabled)
            .unwrap_or(false)
    }

    /// Analyze web page content for learning opportunities
    pub async fn analyze_page(
        &self,
        url: &str,
        content: &str,
    ) -> Result<PageAnalysis, BrowserExtensionError> {
        let difficult_words = self.identify_difficult_words(content).await?;
        let learning_opportunities = self
            .identify_learning_opportunities(content, &difficult_words)
            .await?;
        let pronunciation_hints = self.generate_pronunciation_hints(content).await?;

        Ok(PageAnalysis {
            url: url.to_string(),
            title: self.extract_page_title(content),
            text_content: content.to_string(),
            difficult_words,
            learning_opportunities,
            pronunciation_hints,
            estimated_reading_level: self.calculate_reading_level(content),
            language_detected: self.detect_language(content),
            analysis_timestamp: SystemTime::now(),
        })
    }

    /// Inject VoiRS functionality into a web page
    pub async fn inject_into_page(
        &mut self,
        tab_id: &str,
        url: &str,
        page_analysis: &PageAnalysis,
    ) -> Result<WebPageIntegration, BrowserExtensionError> {
        let session_id = format!(
            "session_{}_{}",
            tab_id,
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        // Create extension session
        let session = ExtensionSession {
            session_id: session_id.clone(),
            tab_id: tab_id.to_string(),
            url: url.to_string(),
            start_time: SystemTime::now(),
            page_analysis: Some(page_analysis.clone()),
            integration: None,
            feedback_count: 0,
        };

        self.active_sessions.insert(session_id.clone(), session);

        // Inject scripts and create UI overlays
        let integration = self.create_page_integration(page_analysis).await?;

        // Update session with integration info
        if let Some(session) = self.active_sessions.get_mut(&session_id) {
            session.integration = Some(integration.clone());
        }

        Ok(integration)
    }

    /// Provide real-time pronunciation feedback
    pub async fn provide_realtime_feedback(
        &mut self,
        session_id: &str,
        text: &str,
        audio_data: Option<&[f32]>,
    ) -> Result<Vec<PronunciationHint>, BrowserExtensionError> {
        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.feedback_count += 1;

            // Generate pronunciation hints for the text
            let hints = self.generate_pronunciation_hints(text).await?;

            // If audio data is provided, analyze it for feedback
            if let Some(audio) = audio_data {
                let _audio_feedback = self.analyze_audio_pronunciation(audio).await?;
                // Could enhance hints based on audio analysis
            }

            Ok(hints)
        } else {
            Err(BrowserExtensionError::CommunicationError(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Remove extension functionality from a page
    pub async fn cleanup_page(&mut self, session_id: &str) -> Result<(), BrowserExtensionError> {
        if let Some(session) = self.active_sessions.remove(session_id) {
            // Clean up injected scripts and UI overlays
            self.remove_page_integration(&session).await?;
            Ok(())
        } else {
            Err(BrowserExtensionError::CommunicationError(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Get extension statistics
    pub fn get_extension_stats(&self) -> ExtensionStats {
        let total_sessions = self.active_sessions.len();
        let total_feedback = self
            .active_sessions
            .values()
            .map(|s| s.feedback_count)
            .sum();

        ExtensionStats {
            active_sessions: total_sessions as u32,
            total_feedback_provided: total_feedback,
            installed_browsers: self.installed_extensions.len() as u32,
            uptime_seconds: 0, // Would track actual uptime
        }
    }

    // Browser-specific installation methods
    async fn install_chrome_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-chrome-extension-id";

        // Simulate Chrome Web Store installation
        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Chrome, info);
        Ok(extension_id.to_string())
    }

    async fn install_firefox_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-firefox-addon-id";

        // Simulate Firefox Add-ons installation
        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Firefox, info);
        Ok(extension_id.to_string())
    }

    async fn install_safari_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-safari-extension-id";

        // Simulate Safari App Store installation
        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Safari, info);
        Ok(extension_id.to_string())
    }

    async fn install_edge_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-edge-extension-id";

        // Simulate Microsoft Edge Add-ons installation
        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Edge, info);
        Ok(extension_id.to_string())
    }

    async fn install_opera_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-opera-extension-id";

        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Opera, info);
        Ok(extension_id.to_string())
    }

    async fn install_brave_extension(&mut self) -> Result<String, BrowserExtensionError> {
        let extension_id = "voirs-brave-extension-id";

        let info = ExtensionInfo {
            extension_id: extension_id.to_string(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions.insert(BrowserType::Brave, info);
        Ok(extension_id.to_string())
    }

    async fn install_custom_extension(
        &mut self,
        browser_name: &str,
    ) -> Result<String, BrowserExtensionError> {
        let extension_id = format!("voirs-{}-extension-id", browser_name.to_lowercase());

        let info = ExtensionInfo {
            extension_id: extension_id.clone(),
            version: self.config.manifest.version.clone(),
            enabled: true,
            permissions: self.config.manifest.permissions.clone(),
            install_date: SystemTime::now(),
        };

        self.installed_extensions
            .insert(BrowserType::Custom(browser_name.to_string()), info);
        Ok(extension_id)
    }

    // Content analysis methods
    async fn identify_difficult_words(
        &self,
        content: &str,
    ) -> Result<Vec<DifficultWord>, BrowserExtensionError> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut difficult_words = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase();

            if clean_word.len() > 7 || self.is_technically_complex(&clean_word) {
                difficult_words.push(DifficultWord {
                    word: clean_word.clone(),
                    difficulty_score: self.calculate_word_difficulty(&clean_word),
                    position: TextPosition {
                        start_offset: i * 5, // Simplified calculation
                        end_offset: i * 5 + word.len(),
                        element_selector: "body".to_string(),
                        element_type: "text".to_string(),
                    },
                    phonetic_transcription: self.get_phonetic_transcription(&clean_word),
                    definition: Some(format!("Definition of {}", clean_word)),
                    usage_examples: vec![format!("Example usage of {}", clean_word)],
                    audio_url: Some(format!("https://api.voirs.com/audio/{}", clean_word)),
                });
            }
        }

        Ok(difficult_words)
    }

    async fn identify_learning_opportunities(
        &self,
        content: &str,
        difficult_words: &[DifficultWord],
    ) -> Result<Vec<LearningOpportunity>, BrowserExtensionError> {
        let mut opportunities = Vec::new();

        // Add pronunciation practice opportunities
        if !difficult_words.is_empty() {
            opportunities.push(LearningOpportunity {
                opportunity_type: OpportunityType::PronunciationPractice,
                description: format!(
                    "Practice pronouncing {} difficult words on this page",
                    difficult_words.len()
                ),
                action_required: "Click on highlighted words to hear pronunciation".to_string(),
                estimated_benefit: 75.0,
                difficulty_level: 60.0,
                position: None,
            });
        }

        // Add vocabulary expansion opportunity
        let word_count = content.split_whitespace().count();
        if word_count > 100 {
            opportunities.push(LearningOpportunity {
                opportunity_type: OpportunityType::VocabularyExpansion,
                description: "Learn new vocabulary from this content".to_string(),
                action_required: "Enable vocabulary highlighting in settings".to_string(),
                estimated_benefit: 65.0,
                difficulty_level: 45.0,
                position: None,
            });
        }

        // Add reading comprehension opportunity
        if word_count > 200 {
            opportunities.push(LearningOpportunity {
                opportunity_type: OpportunityType::ReadingComprehension,
                description: "Practice reading comprehension with this article".to_string(),
                action_required: "Read aloud and use pronunciation feedback".to_string(),
                estimated_benefit: 80.0,
                difficulty_level: 55.0,
                position: None,
            });
        }

        Ok(opportunities)
    }

    async fn generate_pronunciation_hints(
        &self,
        content: &str,
    ) -> Result<Vec<PronunciationHint>, BrowserExtensionError> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut hints = Vec::new();

        for (i, word) in words.iter().enumerate().take(10) {
            // Limit to first 10 words for demo
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase();

            if clean_word.len() > 3 {
                hints.push(PronunciationHint {
                    text: clean_word.clone(),
                    phonetic: self.get_phonetic_transcription(&clean_word),
                    audio_url: Some(format!("https://api.voirs.com/audio/{}", clean_word)),
                    position: TextPosition {
                        start_offset: i * 5,
                        end_offset: i * 5 + word.len(),
                        element_selector: "body".to_string(),
                        element_type: "text".to_string(),
                    },
                    confidence: 0.85,
                    alternatives: vec![
                        format!("{}-alt1", clean_word),
                        format!("{}-alt2", clean_word),
                    ],
                });
            }
        }

        Ok(hints)
    }

    async fn create_page_integration(
        &self,
        page_analysis: &PageAnalysis,
    ) -> Result<WebPageIntegration, BrowserExtensionError> {
        let mut overlays = Vec::new();
        let mut event_listeners = Vec::new();

        // Create feedback panel overlay
        overlays.push(UIOverlay {
            overlay_id: "voirs-feedback-panel".to_string(),
            overlay_type: OverlayType::FeedbackPanel,
            position: self.config.ui_settings.position.clone(),
            size: Size {
                width: 300,
                height: 400,
            },
            content: "VoiRS Pronunciation Helper".to_string(),
            visible: true,
            z_index: 10000,
        });

        // Create pronunciation tooltips for difficult words
        for word in &page_analysis.difficult_words {
            overlays.push(UIOverlay {
                overlay_id: format!("voirs-tooltip-{}", word.word),
                overlay_type: OverlayType::PronunciationTooltip,
                position: UIPosition::Custom { x: 0, y: 0 }, // Would be positioned relative to word
                size: Size {
                    width: 200,
                    height: 100,
                },
                content: format!("🔊 {}: {}", word.word, word.phonetic_transcription),
                visible: false,
                z_index: 10001,
            });
        }

        // Add event listeners for interactions
        event_listeners.push(EventListener {
            event_type: "click".to_string(),
            selector: ".voirs-difficult-word".to_string(),
            handler_function: "showPronunciationTooltip".to_string(),
            capture: false,
        });

        event_listeners.push(EventListener {
            event_type: "mouseover".to_string(),
            selector: ".voirs-difficult-word".to_string(),
            handler_function: "preloadAudio".to_string(),
            capture: false,
        });

        Ok(WebPageIntegration {
            injected_scripts: vec![
                "voirs-content-script.js".to_string(),
                "voirs-pronunciation-helper.js".to_string(),
                "voirs-ui-overlay.js".to_string(),
            ],
            modified_elements: page_analysis
                .difficult_words
                .iter()
                .map(|w| format!("word-{}", w.word))
                .collect(),
            created_overlays: overlays,
            event_listeners,
            storage_keys: vec![
                "voirs-user-preferences".to_string(),
                "voirs-session-data".to_string(),
                "voirs-pronunciation-cache".to_string(),
            ],
        })
    }

    async fn remove_page_integration(
        &self,
        session: &ExtensionSession,
    ) -> Result<(), BrowserExtensionError> {
        // Remove all injected content and event listeners
        // This would involve sending messages to the content script to clean up
        Ok(())
    }

    async fn analyze_audio_pronunciation(
        &self,
        audio_data: &[f32],
    ) -> Result<AudioFeedback, BrowserExtensionError> {
        // Simple audio analysis for demonstration
        let volume_level =
            audio_data.iter().map(|&x| x.abs()).sum::<f32>() / audio_data.len() as f32;

        Ok(AudioFeedback {
            clarity_score: volume_level * 100.0,
            volume_level,
            speech_rate: 150.0, // words per minute
            confidence: 0.8,
            suggestions: vec![
                "Speak more clearly".to_string(),
                "Adjust volume level".to_string(),
            ],
        })
    }

    // Utility methods
    fn extract_page_title(&self, content: &str) -> String {
        // Simple title extraction - in real implementation would parse HTML
        if content.len() > 50 {
            format!("{}...", &content[..50])
        } else {
            content.to_string()
        }
    }

    fn calculate_reading_level(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        let sentence_count = content.split(['.', '!', '?']).count();

        if sentence_count == 0 {
            return 5.0;
        }

        // Simplified Flesch Reading Ease approximation
        let avg_sentence_length = word_count as f32 / sentence_count as f32;
        let complexity_factor = if avg_sentence_length > 20.0 { 2.0 } else { 1.0 };

        5.0 + complexity_factor * (avg_sentence_length / 10.0)
    }

    fn detect_language(&self, content: &str) -> String {
        // Simple language detection - in real implementation would use proper language detection
        if content.contains("the") || content.contains("and") || content.contains("or") {
            "en".to_string()
        } else {
            "unknown".to_string()
        }
    }

    fn is_technically_complex(&self, word: &str) -> bool {
        // Check if word is technically complex
        word.contains("tion")
            || word.contains("sion")
            || word.contains("phy")
            || word.contains("ology")
    }

    fn calculate_word_difficulty(&self, word: &str) -> f32 {
        let base_difficulty = word.len() as f32 * 5.0;
        let complexity_bonus = if self.is_technically_complex(word) {
            20.0
        } else {
            0.0
        };

        (base_difficulty + complexity_bonus).min(100.0)
    }

    fn get_phonetic_transcription(&self, word: &str) -> String {
        // Simplified phonetic transcription - in real implementation would use proper IPA conversion
        format!("/{}~/", word.replace("th", "θ").replace("sh", "ʃ"))
    }
}

/// Extension statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionStats {
    pub active_sessions: u32,
    pub total_feedback_provided: u32,
    pub installed_browsers: u32,
    pub uptime_seconds: u32,
}

/// Audio feedback result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeedback {
    pub clarity_score: f32,
    pub volume_level: f32,
    pub speech_rate: f32,
    pub confidence: f32,
    pub suggestions: Vec<String>,
}

impl Default for ExtensionConfig {
    fn default() -> Self {
        Self {
            browser: BrowserType::Chrome,
            extension_id: String::new(),
            manifest: ExtensionManifest {
                name: "VoiRS Pronunciation Helper".to_string(),
                version: "1.0.0".to_string(),
                description: "Real-time pronunciation feedback and learning assistance".to_string(),
                manifest_version: 3,
                permissions: vec![
                    ExtensionPermission::ActiveTab,
                    ExtensionPermission::Storage,
                    ExtensionPermission::AudioCapture,
                    ExtensionPermission::Scripting,
                ],
                host_permissions: vec!["<all_urls>".to_string()],
                content_scripts: vec![ContentScript {
                    matches: vec!["<all_urls>".to_string()],
                    js: vec!["content-script.js".to_string()],
                    css: vec!["content-style.css".to_string()],
                    run_at: RunAt::DocumentIdle,
                    all_frames: false,
                    match_about_blank: false,
                }],
                background: Some(BackgroundScript {
                    service_worker: Some("background.js".to_string()),
                    scripts: vec![],
                    persistent: false,
                }),
                action: Some(BrowserAction {
                    default_popup: Some("popup.html".to_string()),
                    default_title: "VoiRS Helper".to_string(),
                    default_icon: HashMap::from([
                        ("16".to_string(), "icons/icon16.png".to_string()),
                        ("48".to_string(), "icons/icon48.png".to_string()),
                        ("128".to_string(), "icons/icon128.png".to_string()),
                    ]),
                }),
                options_page: Some("options.html".to_string()),
                web_accessible_resources: vec![WebAccessibleResource {
                    resources: vec!["audio/*".to_string(), "images/*".to_string()],
                    matches: vec!["<all_urls>".to_string()],
                    use_dynamic_url: false,
                }],
            },
            api_endpoint: "https://api.voirs.com".to_string(),
            realtime_enabled: true,
            auto_inject: true,
            privacy_settings: PrivacySettings {
                collect_browsing_data: false,
                store_audio_locally: true,
                send_anonymized_metrics: true,
                respect_do_not_track: true,
                incognito_mode_support: false,
            },
            ui_settings: UISettings {
                show_pronunciation_hints: true,
                highlight_difficult_words: true,
                enable_tooltips: true,
                floating_feedback_panel: true,
                theme: ExtensionTheme::Auto,
                position: UIPosition::TopRight,
                opacity: 0.9,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_config_default() {
        let config = ExtensionConfig::default();
        assert_eq!(config.browser, BrowserType::Chrome);
        assert_eq!(config.manifest.name, "VoiRS Pronunciation Helper");
        assert_eq!(config.manifest.manifest_version, 3);
        assert!(config.realtime_enabled);
    }

    #[test]
    fn test_browser_type_display() {
        assert_eq!(BrowserType::Chrome.to_string(), "Google Chrome");
        assert_eq!(BrowserType::Firefox.to_string(), "Mozilla Firefox");
        assert_eq!(
            BrowserType::Custom("Test".to_string()).to_string(),
            "Custom: Test"
        );
    }

    #[tokio::test]
    async fn test_extension_manager_creation() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);
        // Manager should be created successfully
    }

    #[tokio::test]
    async fn test_extension_installation() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let mut manager = BrowserExtensionManager::new(config, realtime_config);

        let result = manager.install_extension(&BrowserType::Chrome).await;
        assert!(result.is_ok());

        let extension_id = result.unwrap();
        assert!(!extension_id.is_empty());
        assert!(manager.is_extension_active(&BrowserType::Chrome));
    }

    #[tokio::test]
    async fn test_page_analysis() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);

        let content = "This is a comprehensive examination of pronunciation difficulties in the technological field.";
        let analysis = manager
            .analyze_page("https://example.com", content)
            .await
            .unwrap();

        assert_eq!(analysis.url, "https://example.com");
        assert!(!analysis.difficult_words.is_empty());
        assert!(!analysis.learning_opportunities.is_empty());
        assert!(!analysis.pronunciation_hints.is_empty());
    }

    #[tokio::test]
    async fn test_difficult_word_identification() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);

        let content =
            "The comprehensive technological implementation requires systematic organization.";
        let difficult_words = manager.identify_difficult_words(content).await.unwrap();

        assert!(!difficult_words.is_empty());

        // Check if long/complex words are identified
        let has_comprehensive = difficult_words
            .iter()
            .any(|w| w.word.contains("comprehensive"));
        let has_technological = difficult_words
            .iter()
            .any(|w| w.word.contains("technological"));
        let has_implementation = difficult_words
            .iter()
            .any(|w| w.word.contains("implementation"));

        assert!(has_comprehensive || has_technological || has_implementation);
    }

    #[tokio::test]
    async fn test_pronunciation_hints_generation() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);

        let content = "Hello world this is a test";
        let hints = manager.generate_pronunciation_hints(content).await.unwrap();

        assert!(!hints.is_empty());

        for hint in &hints {
            assert!(!hint.text.is_empty());
            assert!(!hint.phonetic.is_empty());
            assert!(hint.confidence > 0.0);
        }
    }

    #[test]
    fn test_reading_level_calculation() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);

        let simple_content = "This is simple. It has short words.";
        let complex_content = "This comprehensive analysis demonstrates the intricate relationships between multifaceted technological implementations and their corresponding organizational implications.";

        let simple_level = manager.calculate_reading_level(simple_content);
        let complex_level = manager.calculate_reading_level(complex_content);

        assert!(complex_level > simple_level);
    }

    #[test]
    fn test_extension_stats() {
        let config = ExtensionConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = BrowserExtensionManager::new(config, realtime_config);

        let stats = manager.get_extension_stats();
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_feedback_provided, 0);
    }

    #[test]
    fn test_privacy_settings() {
        let config = ExtensionConfig::default();
        assert!(!config.privacy_settings.collect_browsing_data);
        assert!(config.privacy_settings.store_audio_locally);
        assert!(config.privacy_settings.respect_do_not_track);
    }

    #[test]
    fn test_ui_settings() {
        let config = ExtensionConfig::default();
        assert!(config.ui_settings.show_pronunciation_hints);
        assert!(config.ui_settings.highlight_difficult_words);
        assert_eq!(config.ui_settings.theme, ExtensionTheme::Auto);
        assert_eq!(config.ui_settings.position, UIPosition::TopRight);
    }
}
