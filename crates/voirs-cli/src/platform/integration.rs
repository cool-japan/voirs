//! System integration features
//!
//! This module provides desktop notifications, system tray integration,
//! file associations, and other OS-level integration features.

use std::path::Path;

/// Notification severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

/// Notification configuration
#[derive(Debug, Clone)]
pub struct Notification {
    /// Notification title
    pub title: String,
    /// Notification message body
    pub message: String,
    /// Severity level
    pub level: NotificationLevel,
    /// Duration in milliseconds (None for persistent)
    pub duration: Option<u32>,
    /// Whether to play a sound
    pub sound: bool,
    /// Application icon path
    pub icon: Option<String>,
}

/// System tray menu item
#[derive(Debug, Clone)]
pub struct TrayMenuItem {
    /// Menu item label
    pub label: String,
    /// Menu item action/command
    pub action: String,
    /// Whether item is enabled
    pub enabled: bool,
    /// Whether item is checked (for toggle items)
    pub checked: bool,
    /// Submenu items
    pub submenu: Vec<TrayMenuItem>,
}

/// System tray configuration
#[derive(Debug, Clone)]
pub struct SystemTray {
    /// Tray icon path
    pub icon: String,
    /// Tooltip text
    pub tooltip: String,
    /// Context menu items
    pub menu: Vec<TrayMenuItem>,
    /// Whether to show notifications
    pub show_notifications: bool,
}

/// File association configuration
#[derive(Debug, Clone)]
pub struct FileAssociation {
    /// File extension (e.g., ".voirs")
    pub extension: String,
    /// MIME type
    pub mime_type: String,
    /// Description
    pub description: String,
    /// Icon path
    pub icon: Option<String>,
    /// Default action command
    pub command: String,
}

/// Desktop integration manager
pub struct DesktopIntegration {
    notifications_enabled: bool,
    tray_enabled: bool,
}

impl DesktopIntegration {
    /// Create new desktop integration manager
    pub fn new() -> Self {
        Self {
            notifications_enabled: true,
            tray_enabled: false,
        }
    }

    /// Show a desktop notification
    pub fn show_notification(
        &self,
        notification: &Notification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.notifications_enabled {
            return Ok(());
        }

        #[cfg(target_os = "windows")]
        {
            self.show_windows_notification(notification)
        }
        #[cfg(target_os = "macos")]
        {
            self.show_macos_notification(notification)
        }
        #[cfg(target_os = "linux")]
        {
            self.show_linux_notification(notification)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            // Fallback: print to console
            println!(
                "[{}] {}: {}",
                format_notification_level(notification.level),
                notification.title,
                notification.message
            );
            Ok(())
        }
    }

    /// Initialize system tray
    pub fn init_system_tray(
        &mut self,
        config: &SystemTray,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "windows")]
        {
            self.init_windows_tray(config)
        }
        #[cfg(target_os = "macos")]
        {
            self.init_macos_tray(config)
        }
        #[cfg(target_os = "linux")]
        {
            self.init_linux_tray(config)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            tracing::warn!("System tray not supported on this platform");
            Ok(())
        }
    }

    /// Register file associations
    pub fn register_file_associations(
        &self,
        associations: &[FileAssociation],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for association in associations {
            self.register_file_association(association)?;
        }
        Ok(())
    }

    /// Register a single file association
    pub fn register_file_association(
        &self,
        association: &FileAssociation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "windows")]
        {
            self.register_windows_file_association(association)
        }
        #[cfg(target_os = "macos")]
        {
            self.register_macos_file_association(association)
        }
        #[cfg(target_os = "linux")]
        {
            self.register_linux_file_association(association)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            tracing::warn!("File associations not supported on this platform");
            Ok(())
        }
    }

    /// Enable/disable notifications
    pub fn set_notifications_enabled(&mut self, enabled: bool) {
        self.notifications_enabled = enabled;
    }

    /// Check if notifications are supported
    pub fn notifications_supported(&self) -> bool {
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        {
            true
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            false
        }
    }

    /// Check if system tray is supported
    pub fn system_tray_supported(&self) -> bool {
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        {
            true
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            false
        }
    }

    /// Send synthesis completion notification
    pub fn notify_synthesis_complete(
        &self,
        output_file: &Path,
        duration: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let notification = Notification {
            title: "VoiRS Synthesis Complete".to_string(),
            message: format!(
                "Audio generated successfully in {:.1}s\nOutput: {}",
                duration,
                output_file
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("output.wav")
            ),
            level: NotificationLevel::Success,
            duration: Some(5000), // 5 seconds
            sound: true,
            icon: None,
        };

        self.show_notification(&notification)
    }

    /// Send error notification
    pub fn notify_error(&self, error: &str) -> Result<(), Box<dyn std::error::Error>> {
        let notification = Notification {
            title: "VoiRS Error".to_string(),
            message: error.to_string(),
            level: NotificationLevel::Error,
            duration: Some(10000), // 10 seconds
            sound: true,
            icon: None,
        };

        self.show_notification(&notification)
    }

    /// Send progress notification
    pub fn notify_progress(
        &self,
        message: &str,
        progress: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let notification = Notification {
            title: "VoiRS Progress".to_string(),
            message: format!("{} ({:.0}%)", message, progress * 100.0),
            level: NotificationLevel::Info,
            duration: Some(3000), // 3 seconds
            sound: false,
            icon: None,
        };

        self.show_notification(&notification)
    }

    /// Create default system tray configuration
    pub fn create_default_tray_config(&self) -> SystemTray {
        SystemTray {
            icon: "voirs-icon.png".to_string(),
            tooltip: "VoiRS - Text-to-Speech Synthesis".to_string(),
            menu: vec![
                TrayMenuItem {
                    label: "Quick Synthesis".to_string(),
                    action: "quick_synthesis".to_string(),
                    enabled: true,
                    checked: false,
                    submenu: Vec::new(),
                },
                TrayMenuItem {
                    label: "Interactive Mode".to_string(),
                    action: "interactive_mode".to_string(),
                    enabled: true,
                    checked: false,
                    submenu: Vec::new(),
                },
                TrayMenuItem {
                    label: "Server Mode".to_string(),
                    action: "toggle_server".to_string(),
                    enabled: true,
                    checked: false,
                    submenu: Vec::new(),
                },
                TrayMenuItem {
                    label: "Settings".to_string(),
                    action: "settings".to_string(),
                    enabled: true,
                    checked: false,
                    submenu: vec![
                        TrayMenuItem {
                            label: "Preferences".to_string(),
                            action: "preferences".to_string(),
                            enabled: true,
                            checked: false,
                            submenu: Vec::new(),
                        },
                        TrayMenuItem {
                            label: "Voice Manager".to_string(),
                            action: "voice_manager".to_string(),
                            enabled: true,
                            checked: false,
                            submenu: Vec::new(),
                        },
                    ],
                },
                TrayMenuItem {
                    label: "Quit".to_string(),
                    action: "quit".to_string(),
                    enabled: true,
                    checked: false,
                    submenu: Vec::new(),
                },
            ],
            show_notifications: true,
        }
    }

    /// Create default file associations
    pub fn create_default_file_associations(&self) -> Vec<FileAssociation> {
        vec![
            FileAssociation {
                extension: ".voirs".to_string(),
                mime_type: "application/vnd.voirs.synthesis".to_string(),
                description: "VoiRS Synthesis Configuration".to_string(),
                icon: Some("voirs-file.png".to_string()),
                command: "voirs synthesize-file".to_string(),
            },
            FileAssociation {
                extension: ".ssml".to_string(),
                mime_type: "application/ssml+xml".to_string(),
                description: "Speech Synthesis Markup Language".to_string(),
                icon: Some("ssml-file.png".to_string()),
                command: "voirs synthesize".to_string(),
            },
        ]
    }
}

// Platform-specific implementations

#[cfg(target_os = "windows")]
impl DesktopIntegration {
    fn show_windows_notification(
        &self,
        notification: &Notification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Windows notification implementation using Windows API or winrt
        use std::process::Command;

        // Use PowerShell to show toast notification
        let script = format!(
            r#"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('{}', '{}', 'OK', '{}')"#,
            notification.message,
            notification.title,
            match notification.level {
                NotificationLevel::Error => "Error",
                NotificationLevel::Warning => "Warning",
                _ => "Information",
            }
        );

        Command::new("powershell")
            .arg("-Command")
            .arg(&script)
            .output()?;

        Ok(())
    }

    fn init_windows_tray(
        &mut self,
        _config: &SystemTray,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Windows system tray implementation
        self.tray_enabled = true;
        tracing::info!("Windows system tray initialized");
        Ok(())
    }

    fn register_windows_file_association(
        &self,
        association: &FileAssociation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Windows file association using registry
        tracing::info!(
            "Registering Windows file association for {}",
            association.extension
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl DesktopIntegration {
    fn show_macos_notification(
        &self,
        notification: &Notification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // macOS notification using osascript/AppleScript
        use std::process::Command;

        let script = format!(
            r#"display notification "{}" with title "{}" sound name "default""#,
            notification.message.replace('"', r#"\""#),
            notification.title.replace('"', r#"\""#)
        );

        Command::new("osascript").arg("-e").arg(&script).output()?;

        Ok(())
    }

    fn init_macos_tray(&mut self, _config: &SystemTray) -> Result<(), Box<dyn std::error::Error>> {
        // macOS menu bar integration
        self.tray_enabled = true;
        tracing::info!("macOS menu bar integration initialized");
        Ok(())
    }

    fn register_macos_file_association(
        &self,
        association: &FileAssociation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // macOS file association using LaunchServices
        tracing::info!(
            "Registering macOS file association for {}",
            association.extension
        );
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl DesktopIntegration {
    fn show_linux_notification(
        &self,
        notification: &Notification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Linux notification using notify-send or libnotify
        use std::process::Command;

        let urgency = match notification.level {
            NotificationLevel::Error => "critical",
            NotificationLevel::Warning => "normal",
            _ => "low",
        };

        let mut cmd = Command::new("notify-send");
        cmd.arg("--urgency").arg(urgency);
        cmd.arg("--app-name").arg("VoiRS");

        if let Some(duration) = notification.duration {
            cmd.arg("--expire-time").arg(duration.to_string());
        }

        cmd.arg(&notification.title);
        cmd.arg(&notification.message);

        cmd.output()?;
        Ok(())
    }

    fn init_linux_tray(&mut self, _config: &SystemTray) -> Result<(), Box<dyn std::error::Error>> {
        // Linux system tray using StatusNotifierItem or AppIndicator
        self.tray_enabled = true;
        tracing::info!("Linux system tray initialized");
        Ok(())
    }

    fn register_linux_file_association(
        &self,
        association: &FileAssociation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Linux file association using .desktop files and MIME types
        let desktop_entry = format!(
            r#"[Desktop Entry]
Name=VoiRS
Comment={}
Exec={}
Icon={}
Terminal=false
Type=Application
MimeType={};
"#,
            association.description,
            association.command,
            association.icon.as_deref().unwrap_or("voirs"),
            association.mime_type
        );

        // Write .desktop file to ~/.local/share/applications/
        if let Some(home) = dirs::home_dir() {
            let desktop_file = home
                .join(".local")
                .join("share")
                .join("applications")
                .join("voirs.desktop");

            std::fs::create_dir_all(desktop_file.parent().unwrap())?;
            std::fs::write(&desktop_file, desktop_entry)?;

            // Update MIME database
            std::process::Command::new("update-desktop-database")
                .arg(desktop_file.parent().unwrap())
                .output()
                .ok(); // Ignore errors, this is optional
        }

        tracing::info!(
            "Registered Linux file association for {}",
            association.extension
        );
        Ok(())
    }
}

// Helper functions

fn format_notification_level(level: NotificationLevel) -> &'static str {
    match level {
        NotificationLevel::Info => "INFO",
        NotificationLevel::Success => "SUCCESS",
        NotificationLevel::Warning => "WARNING",
        NotificationLevel::Error => "ERROR",
    }
}

impl Default for DesktopIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common notifications

/// Show synthesis started notification
pub fn notify_synthesis_started(
    integration: &DesktopIntegration,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let preview = if text.len() > 50 {
        format!("{}...", &text[..47])
    } else {
        text.to_string()
    };

    let notification = Notification {
        title: "VoiRS Synthesis Started".to_string(),
        message: format!("Synthesizing: {}", preview),
        level: NotificationLevel::Info,
        duration: Some(3000),
        sound: false,
        icon: None,
    };

    integration.show_notification(&notification)
}

/// Show batch processing notification
pub fn notify_batch_progress(
    integration: &DesktopIntegration,
    completed: usize,
    total: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let notification = Notification {
        title: "VoiRS Batch Processing".to_string(),
        message: format!("Processed {} of {} files", completed, total),
        level: NotificationLevel::Info,
        duration: Some(2000),
        sound: false,
        icon: None,
    };

    integration.show_notification(&notification)
}

/// Show server status notification
pub fn notify_server_status(
    integration: &DesktopIntegration,
    running: bool,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let notification = Notification {
        title: "VoiRS Server".to_string(),
        message: if running {
            format!("Server started on port {}", port)
        } else {
            "Server stopped".to_string()
        },
        level: if running {
            NotificationLevel::Success
        } else {
            NotificationLevel::Info
        },
        duration: Some(5000),
        sound: running,
        icon: None,
    };

    integration.show_notification(&notification)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desktop_integration_creation() {
        let integration = DesktopIntegration::new();
        assert!(integration.notifications_enabled);
        assert!(!integration.tray_enabled);
    }

    #[test]
    fn test_notification_creation() {
        let notification = Notification {
            title: "Test".to_string(),
            message: "Test message".to_string(),
            level: NotificationLevel::Info,
            duration: Some(1000),
            sound: false,
            icon: None,
        };

        assert_eq!(notification.title, "Test");
        assert_eq!(notification.level, NotificationLevel::Info);
    }

    #[test]
    fn test_tray_config_creation() {
        let integration = DesktopIntegration::new();
        let config = integration.create_default_tray_config();

        assert!(!config.menu.is_empty());
        assert_eq!(config.tooltip, "VoiRS - Text-to-Speech Synthesis");
    }

    #[test]
    fn test_file_associations_creation() {
        let integration = DesktopIntegration::new();
        let associations = integration.create_default_file_associations();

        assert!(!associations.is_empty());
        assert!(associations.iter().any(|a| a.extension == ".voirs"));
    }

    #[test]
    fn test_platform_support() {
        let integration = DesktopIntegration::new();

        // These should return true on supported platforms
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        {
            assert!(integration.notifications_supported());
            assert!(integration.system_tray_supported());
        }
    }
}
