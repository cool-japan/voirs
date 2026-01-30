//! Accessibility Support Module
//!
//! Provides WCAG 2.1 AA/AAA compliance features, screen reader support,
//! keyboard navigation, and inclusive design utilities for the `VoiRS` feedback system.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Accessibility errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum AccessibilityError {
    /// Validation failed
    #[error("Accessibility validation failed: {message}")]
    ValidationFailed { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Feature not supported
    #[error("Feature not supported: {feature}")]
    FeatureNotSupported { feature: String },
}

/// Result type for accessibility operations
pub type AccessibilityResult<T> = Result<T, AccessibilityError>;

/// WCAG 2.1 conformance level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WcagLevel {
    /// Level A (minimum)
    A,
    /// Level AA (standard)
    AA,
    /// Level AAA (enhanced)
    AAA,
}

/// Accessibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityConfig {
    /// Target WCAG conformance level
    pub wcag_level: WcagLevel,
    /// Enable screen reader support
    pub screen_reader_enabled: bool,
    /// Enable keyboard navigation
    pub keyboard_navigation_enabled: bool,
    /// Enable high contrast mode
    pub high_contrast_enabled: bool,
    /// Enable reduced motion
    pub reduced_motion_enabled: bool,
    /// Text size multiplier (1.0 = normal)
    pub text_size_multiplier: f32,
    /// Enable focus indicators
    pub focus_indicators_enabled: bool,
    /// Enable audio descriptions
    pub audio_descriptions_enabled: bool,
    /// Enable captions
    pub captions_enabled: bool,
    /// Enable sign language interpretation
    pub sign_language_enabled: bool,
    /// Color blindness mode
    pub color_blindness_mode: Option<ColorBlindnessMode>,
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            wcag_level: WcagLevel::AA,
            screen_reader_enabled: true,
            keyboard_navigation_enabled: true,
            high_contrast_enabled: false,
            reduced_motion_enabled: false,
            text_size_multiplier: 1.0,
            focus_indicators_enabled: true,
            audio_descriptions_enabled: false,
            captions_enabled: false,
            sign_language_enabled: false,
            color_blindness_mode: None,
        }
    }
}

/// Color blindness adaptation modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColorBlindnessMode {
    /// Protanopia (red-blind)
    Protanopia,
    /// Deuteranopia (green-blind)
    Deuteranopia,
    /// Tritanopia (blue-blind)
    Tritanopia,
    /// Achromatopsia (total color blindness)
    Achromatopsia,
}

/// Screen reader announcement priority
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AnnouncementPriority {
    /// Polite - wait for user to finish
    Polite,
    /// Assertive - interrupt current speech
    Assertive,
    /// Off - no announcement
    Off,
}

/// Screen reader announcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenReaderAnnouncement {
    /// Announcement ID
    pub id: String,
    /// Text to announce
    pub text: String,
    /// Priority level
    pub priority: AnnouncementPriority,
    /// ARIA live region type
    pub live_region: AriaLiveRegion,
    /// Language code
    pub language: Option<String>,
}

/// ARIA live region types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AriaLiveRegion {
    /// Polite updates
    Polite,
    /// Assertive updates
    Assertive,
    /// No automatic updates
    Off,
}

/// Keyboard shortcut
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyboardShortcut {
    /// Shortcut ID
    pub id: String,
    /// Key combination (e.g., "Ctrl+S", "Alt+F")
    pub key_combo: String,
    /// Description
    pub description: String,
    /// Category
    pub category: ShortcutCategory,
    /// Enabled state
    pub enabled: bool,
}

/// Keyboard shortcut categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShortcutCategory {
    /// Navigation shortcuts
    Navigation,
    /// Editing shortcuts
    Editing,
    /// Playback control
    Playback,
    /// System control
    System,
    /// Custom shortcuts
    Custom,
}

/// Focus navigation support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusNavigation {
    /// Current focus element ID
    pub current_focus: Option<String>,
    /// Focus history
    pub focus_history: Vec<String>,
    /// Tab order
    pub tab_order: Vec<String>,
    /// Focus trap enabled
    pub trap_enabled: bool,
}

impl FocusNavigation {
    /// Create new focus navigation
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_focus: None,
            focus_history: Vec::new(),
            tab_order: Vec::new(),
            trap_enabled: false,
        }
    }

    /// Move focus to next element
    pub fn focus_next(&mut self) -> Option<String> {
        if self.tab_order.is_empty() {
            return None;
        }

        let next_index = if let Some(current) = &self.current_focus {
            (self
                .tab_order
                .iter()
                .position(|id| id == current)
                .unwrap_or(0)
                + 1)
                % self.tab_order.len()
        } else {
            0
        };

        let next_id = self.tab_order[next_index].clone();
        self.set_focus(next_id.clone());
        Some(next_id)
    }

    /// Move focus to previous element
    pub fn focus_previous(&mut self) -> Option<String> {
        if self.tab_order.is_empty() {
            return None;
        }

        let prev_index = if let Some(current) = &self.current_focus {
            let current_index = self
                .tab_order
                .iter()
                .position(|id| id == current)
                .unwrap_or(0);
            if current_index == 0 {
                self.tab_order.len() - 1
            } else {
                current_index - 1
            }
        } else {
            self.tab_order.len() - 1
        };

        let prev_id = self.tab_order[prev_index].clone();
        self.set_focus(prev_id.clone());
        Some(prev_id)
    }

    /// Set focus to specific element
    pub fn set_focus(&mut self, element_id: String) {
        if let Some(current) = &self.current_focus {
            self.focus_history.push(current.clone());
        }
        self.current_focus = Some(element_id);
    }

    /// Return to previous focus
    pub fn restore_focus(&mut self) -> Option<String> {
        self.focus_history.pop().inspect(|prev_id| {
            self.current_focus = Some(prev_id.clone());
        })
    }
}

impl Default for FocusNavigation {
    fn default() -> Self {
        Self::new()
    }
}

/// Color contrast checker
pub struct ColorContrastChecker {
    /// Minimum contrast ratio for AA
    aa_normal_ratio: f32,
    /// Minimum contrast ratio for AA large text
    aa_large_ratio: f32,
    /// Minimum contrast ratio for AAA
    aaa_normal_ratio: f32,
    /// Minimum contrast ratio for AAA large text
    aaa_large_ratio: f32,
}

impl ColorContrastChecker {
    /// Create new color contrast checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            aa_normal_ratio: 4.5,
            aa_large_ratio: 3.0,
            aaa_normal_ratio: 7.0,
            aaa_large_ratio: 4.5,
        }
    }

    /// Check contrast ratio between two colors
    #[must_use]
    pub fn check_contrast(
        &self,
        foreground: Color,
        background: Color,
        is_large_text: bool,
        level: WcagLevel,
    ) -> bool {
        let ratio = self.calculate_contrast_ratio(foreground, background);

        match level {
            WcagLevel::A => true, // Level A doesn't have contrast requirements
            WcagLevel::AA => {
                if is_large_text {
                    ratio >= self.aa_large_ratio
                } else {
                    ratio >= self.aa_normal_ratio
                }
            }
            WcagLevel::AAA => {
                if is_large_text {
                    ratio >= self.aaa_large_ratio
                } else {
                    ratio >= self.aaa_normal_ratio
                }
            }
        }
    }

    /// Calculate contrast ratio
    fn calculate_contrast_ratio(&self, foreground: Color, background: Color) -> f32 {
        let l1 = self.relative_luminance(foreground);
        let l2 = self.relative_luminance(background);

        let lighter = l1.max(l2);
        let darker = l1.min(l2);

        (lighter + 0.05) / (darker + 0.05)
    }

    /// Calculate relative luminance
    fn relative_luminance(&self, color: Color) -> f32 {
        let r = self.linearize_channel(color.r);
        let g = self.linearize_channel(color.g);
        let b = self.linearize_channel(color.b);

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Linearize RGB channel
    fn linearize_channel(&self, channel: u8) -> f32 {
        let value = f32::from(channel) / 255.0;
        if value <= 0.03928 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }
}

impl Default for ColorContrastChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// RGB color
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    /// Red (0-255)
    pub r: u8,
    /// Green (0-255)
    pub g: u8,
    /// Blue (0-255)
    pub b: u8,
}

/// Accessibility manager
pub struct AccessibilityManager {
    /// Configuration
    config: Arc<RwLock<AccessibilityConfig>>,
    /// Screen reader announcements queue
    announcements: Arc<RwLock<Vec<ScreenReaderAnnouncement>>>,
    /// Keyboard shortcuts
    shortcuts: Arc<RwLock<HashMap<String, KeyboardShortcut>>>,
    /// Focus navigation
    focus_navigation: Arc<RwLock<FocusNavigation>>,
    /// Color contrast checker
    contrast_checker: ColorContrastChecker,
    /// User preferences
    user_preferences: Arc<RwLock<HashMap<String, AccessibilityConfig>>>,
}

impl AccessibilityManager {
    /// Create new accessibility manager
    #[must_use]
    pub fn new(config: AccessibilityConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            announcements: Arc::new(RwLock::new(Vec::new())),
            shortcuts: Arc::new(RwLock::new(Self::default_shortcuts())),
            focus_navigation: Arc::new(RwLock::new(FocusNavigation::new())),
            contrast_checker: ColorContrastChecker::new(),
            user_preferences: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Announce to screen reader
    pub async fn announce(&self, text: String, priority: AnnouncementPriority) {
        let config = self.config.read().await;
        if !config.screen_reader_enabled {
            return;
        }
        drop(config);

        let announcement = ScreenReaderAnnouncement {
            id: uuid::Uuid::new_v4().to_string(),
            text,
            priority,
            live_region: match priority {
                AnnouncementPriority::Polite => AriaLiveRegion::Polite,
                AnnouncementPriority::Assertive => AriaLiveRegion::Assertive,
                AnnouncementPriority::Off => AriaLiveRegion::Off,
            },
            language: None,
        };

        let mut announcements = self.announcements.write().await;
        announcements.push(announcement);
    }

    /// Get pending announcements
    pub async fn get_announcements(&self) -> Vec<ScreenReaderAnnouncement> {
        let mut announcements = self.announcements.write().await;
        let result = announcements.clone();
        announcements.clear();
        result
    }

    /// Register keyboard shortcut
    pub async fn register_shortcut(&self, shortcut: KeyboardShortcut) {
        let mut shortcuts = self.shortcuts.write().await;
        shortcuts.insert(shortcut.id.clone(), shortcut);
    }

    /// Get keyboard shortcuts by category
    pub async fn get_shortcuts_by_category(
        &self,
        category: ShortcutCategory,
    ) -> Vec<KeyboardShortcut> {
        let shortcuts = self.shortcuts.read().await;
        shortcuts
            .values()
            .filter(|s| s.category == category)
            .cloned()
            .collect()
    }

    /// Handle keyboard event
    pub async fn handle_keyboard_event(&self, key_combo: &str) -> Option<String> {
        let config = self.config.read().await;
        if !config.keyboard_navigation_enabled {
            return None;
        }
        drop(config);

        let shortcuts = self.shortcuts.read().await;
        shortcuts
            .values()
            .find(|s| s.enabled && s.key_combo == key_combo)
            .map(|s| s.id.clone())
    }

    /// Move focus to next element
    pub async fn focus_next(&self) -> Option<String> {
        let mut nav = self.focus_navigation.write().await;
        nav.focus_next()
    }

    /// Move focus to previous element
    pub async fn focus_previous(&self) -> Option<String> {
        let mut nav = self.focus_navigation.write().await;
        nav.focus_previous()
    }

    /// Set tab order
    pub async fn set_tab_order(&self, element_ids: Vec<String>) {
        let mut nav = self.focus_navigation.write().await;
        nav.tab_order = element_ids;
    }

    /// Check color contrast compliance
    #[must_use]
    pub fn check_color_contrast(
        &self,
        foreground: Color,
        background: Color,
        is_large_text: bool,
        level: WcagLevel,
    ) -> bool {
        self.contrast_checker
            .check_contrast(foreground, background, is_large_text, level)
    }

    /// Update configuration
    pub async fn update_config(&self, config: AccessibilityConfig) {
        let mut current_config = self.config.write().await;
        *current_config = config;
    }

    /// Get current configuration
    pub async fn get_config(&self) -> AccessibilityConfig {
        self.config.read().await.clone()
    }

    /// Save user preferences
    pub async fn save_user_preferences(&self, user_id: String, config: AccessibilityConfig) {
        let mut prefs = self.user_preferences.write().await;
        prefs.insert(user_id, config);
    }

    /// Load user preferences
    pub async fn load_user_preferences(&self, user_id: &str) -> Option<AccessibilityConfig> {
        let prefs = self.user_preferences.read().await;
        prefs.get(user_id).cloned()
    }

    /// Generate accessibility report
    pub async fn generate_report(&self) -> AccessibilityReport {
        let config = self.config.read().await;
        let shortcuts = self.shortcuts.read().await;
        let announcements = self.announcements.read().await;

        AccessibilityReport {
            wcag_level: config.wcag_level.clone(),
            features_enabled: vec![
                ("Screen Reader", config.screen_reader_enabled),
                ("Keyboard Navigation", config.keyboard_navigation_enabled),
                ("High Contrast", config.high_contrast_enabled),
                ("Reduced Motion", config.reduced_motion_enabled),
                ("Focus Indicators", config.focus_indicators_enabled),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
            registered_shortcuts: shortcuts.len(),
            pending_announcements: announcements.len(),
            text_size_multiplier: config.text_size_multiplier,
        }
    }

    fn default_shortcuts() -> HashMap<String, KeyboardShortcut> {
        let mut shortcuts = HashMap::new();

        // Navigation shortcuts
        shortcuts.insert(
            "nav_next".to_string(),
            KeyboardShortcut {
                id: "nav_next".to_string(),
                key_combo: "Tab".to_string(),
                description: "Move to next element".to_string(),
                category: ShortcutCategory::Navigation,
                enabled: true,
            },
        );

        shortcuts.insert(
            "nav_prev".to_string(),
            KeyboardShortcut {
                id: "nav_prev".to_string(),
                key_combo: "Shift+Tab".to_string(),
                description: "Move to previous element".to_string(),
                category: ShortcutCategory::Navigation,
                enabled: true,
            },
        );

        shortcuts.insert(
            "nav_home".to_string(),
            KeyboardShortcut {
                id: "nav_home".to_string(),
                key_combo: "Home".to_string(),
                description: "Go to beginning".to_string(),
                category: ShortcutCategory::Navigation,
                enabled: true,
            },
        );

        shortcuts.insert(
            "nav_end".to_string(),
            KeyboardShortcut {
                id: "nav_end".to_string(),
                key_combo: "End".to_string(),
                description: "Go to end".to_string(),
                category: ShortcutCategory::Navigation,
                enabled: true,
            },
        );

        // Playback shortcuts
        shortcuts.insert(
            "play_pause".to_string(),
            KeyboardShortcut {
                id: "play_pause".to_string(),
                key_combo: "Space".to_string(),
                description: "Play/Pause".to_string(),
                category: ShortcutCategory::Playback,
                enabled: true,
            },
        );

        shortcuts.insert(
            "stop".to_string(),
            KeyboardShortcut {
                id: "stop".to_string(),
                key_combo: "Escape".to_string(),
                description: "Stop playback".to_string(),
                category: ShortcutCategory::Playback,
                enabled: true,
            },
        );

        // System shortcuts
        shortcuts.insert(
            "help".to_string(),
            KeyboardShortcut {
                id: "help".to_string(),
                key_combo: "F1".to_string(),
                description: "Show help".to_string(),
                category: ShortcutCategory::System,
                enabled: true,
            },
        );

        shortcuts
    }
}

/// Accessibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityReport {
    /// WCAG level
    pub wcag_level: WcagLevel,
    /// Enabled features
    pub features_enabled: HashMap<String, bool>,
    /// Number of registered shortcuts
    pub registered_shortcuts: usize,
    /// Pending announcements
    pub pending_announcements: usize,
    /// Text size multiplier
    pub text_size_multiplier: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_navigation() {
        let mut nav = FocusNavigation::new();
        nav.tab_order = vec!["btn1".to_string(), "btn2".to_string(), "btn3".to_string()];

        // Test next navigation
        assert_eq!(nav.focus_next(), Some("btn1".to_string()));
        assert_eq!(nav.focus_next(), Some("btn2".to_string()));
        assert_eq!(nav.focus_next(), Some("btn3".to_string()));
        assert_eq!(nav.focus_next(), Some("btn1".to_string())); // Wrap around

        // Test previous navigation
        assert_eq!(nav.focus_previous(), Some("btn3".to_string()));
        assert_eq!(nav.focus_previous(), Some("btn2".to_string()));
    }

    #[test]
    fn test_color_contrast_aa() {
        let checker = ColorContrastChecker::new();

        // Black on white - should pass AA
        let black = Color { r: 0, g: 0, b: 0 };
        let white = Color {
            r: 255,
            g: 255,
            b: 255,
        };
        assert!(checker.check_contrast(black, white, false, WcagLevel::AA));

        // Gray on white - might not pass AA for normal text
        let gray = Color {
            r: 150,
            g: 150,
            b: 150,
        };
        assert!(!checker.check_contrast(gray, white, false, WcagLevel::AA));

        // Darker gray should pass AA for large text (needs at least 3:1 ratio)
        let dark_gray = Color {
            r: 118,
            g: 118,
            b: 118,
        };
        assert!(checker.check_contrast(dark_gray, white, true, WcagLevel::AA));
    }

    #[tokio::test]
    async fn test_screen_reader_announcements() {
        let config = AccessibilityConfig::default();
        let manager = AccessibilityManager::new(config);

        manager
            .announce(
                "Test announcement".to_string(),
                AnnouncementPriority::Polite,
            )
            .await;

        let announcements = manager.get_announcements().await;
        assert_eq!(announcements.len(), 1);
        assert_eq!(announcements[0].text, "Test announcement");
        assert_eq!(announcements[0].priority, AnnouncementPriority::Polite);

        // Second call should return empty
        let announcements2 = manager.get_announcements().await;
        assert_eq!(announcements2.len(), 0);
    }

    #[tokio::test]
    async fn test_keyboard_shortcuts() {
        let config = AccessibilityConfig::default();
        let manager = AccessibilityManager::new(config);

        let shortcuts = manager
            .get_shortcuts_by_category(ShortcutCategory::Navigation)
            .await;
        assert!(!shortcuts.is_empty());

        // Test handling keyboard event
        let shortcut_id = manager.handle_keyboard_event("Tab").await;
        assert_eq!(shortcut_id, Some("nav_next".to_string()));
    }

    #[tokio::test]
    async fn test_user_preferences() {
        let config = AccessibilityConfig::default();
        let manager = AccessibilityManager::new(config);

        let mut user_config = AccessibilityConfig::default();
        user_config.high_contrast_enabled = true;
        user_config.text_size_multiplier = 1.5;

        manager
            .save_user_preferences("user123".to_string(), user_config.clone())
            .await;

        let loaded_config = manager.load_user_preferences("user123").await.unwrap();
        assert!(loaded_config.high_contrast_enabled);
        assert_eq!(loaded_config.text_size_multiplier, 1.5);
    }

    #[tokio::test]
    async fn test_accessibility_report() {
        let config = AccessibilityConfig::default();
        let manager = AccessibilityManager::new(config);

        manager
            .announce("Test".to_string(), AnnouncementPriority::Polite)
            .await;

        let report = manager.generate_report().await;
        assert_eq!(report.wcag_level, WcagLevel::AA);
        assert_eq!(report.pending_announcements, 1);
        assert!(report.registered_shortcuts > 0);
    }

    #[test]
    fn test_color_contrast_calculation() {
        let checker = ColorContrastChecker::new();

        // Test exact WCAG examples
        let black = Color { r: 0, g: 0, b: 0 };
        let white = Color {
            r: 255,
            g: 255,
            b: 255,
        };
        let ratio = checker.calculate_contrast_ratio(black, white);

        // Black/white should be exactly 21:1
        assert!((ratio - 21.0).abs() < 0.1);
    }
}
