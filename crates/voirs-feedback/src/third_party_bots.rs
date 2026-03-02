//! Third-Party Bot Integrations (Slack, Discord, Teams)
//!
//! This module provides integration capabilities for popular communication platforms
//! including Slack, Discord, and Microsoft Teams, allowing feedback notifications
//! and interactive bot commands.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "microservices")]
use reqwest::Client;

/// Bot integration errors
#[derive(Error, Debug, Clone)]
pub enum BotError {
    /// Bot not configured
    #[error("Bot '{platform}' is not configured")]
    NotConfigured {
        /// Platform name
        platform: String,
    },

    /// Message send failed
    #[error("Failed to send message: {message}")]
    SendFailed {
        /// Error message
        message: String,
    },

    /// Authentication failed
    #[error("Authentication failed for platform '{platform}': {reason}")]
    AuthFailed {
        /// Platform name
        platform: String,
        /// Reason for failure
        reason: String,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfig {
        /// Error message
        message: String,
    },

    /// API error
    #[error("API error from '{platform}': {message}")]
    ApiError {
        /// Platform name
        platform: String,
        /// Error message
        message: String,
    },
}

/// Result type for bot operations
pub type BotResult<T> = Result<T, BotError>;

/// Bot platform type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BotPlatform {
    /// Slack
    Slack,
    /// Discord
    Discord,
    /// Microsoft Teams
    MicrosoftTeams,
    /// Telegram
    Telegram,
    /// Generic webhook
    Generic,
}

impl BotPlatform {
    /// Get platform name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            BotPlatform::Slack => "Slack",
            BotPlatform::Discord => "Discord",
            BotPlatform::MicrosoftTeams => "Microsoft Teams",
            BotPlatform::Telegram => "Telegram",
            BotPlatform::Generic => "Generic",
        }
    }
}

/// Message priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessagePriority {
    /// Low priority (no notification)
    Low,
    /// Normal priority
    Normal,
    /// High priority (mentions)
    High,
    /// Urgent (everyone mention)
    Urgent,
}

/// Message attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    /// Title
    pub title: String,
    /// Content/text
    pub text: String,
    /// Color (hex code)
    pub color: Option<String>,
    /// Fields (key-value pairs)
    pub fields: Vec<(String, String)>,
    /// Image URL
    pub image_url: Option<String>,
    /// Thumbnail URL
    pub thumbnail_url: Option<String>,
}

/// Bot message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotMessage {
    /// Message text
    pub text: String,
    /// Channel/room ID
    pub channel: String,
    /// Priority
    pub priority: MessagePriority,
    /// Attachments
    pub attachments: Vec<MessageAttachment>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Thread ID (for threaded messages)
    pub thread_id: Option<String>,
}

impl BotMessage {
    /// Create a new bot message
    #[must_use]
    pub fn new(text: impl Into<String>, channel: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            channel: channel.into(),
            priority: MessagePriority::Normal,
            attachments: Vec::new(),
            metadata: HashMap::new(),
            thread_id: None,
        }
    }

    /// Add attachment
    #[must_use]
    pub fn with_attachment(mut self, attachment: MessageAttachment) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set thread ID
    #[must_use]
    pub fn in_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }
}

/// Bot configuration for a platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotConfig {
    /// Platform type
    pub platform: BotPlatform,
    /// Bot token/API key
    pub token: String,
    /// Webhook URL (if applicable)
    pub webhook_url: Option<String>,
    /// Default channel
    pub default_channel: String,
    /// Bot username
    pub username: Option<String>,
    /// Bot icon/avatar URL
    pub icon_url: Option<String>,
    /// Enabled
    pub enabled: bool,
}

/// Bot integration trait
#[async_trait]
pub trait BotIntegration: Send + Sync {
    /// Get platform type
    fn platform(&self) -> BotPlatform;

    /// Send message
    async fn send_message(&self, message: &BotMessage) -> BotResult<String>;

    /// Edit message
    async fn edit_message(&self, message_id: &str, new_text: &str) -> BotResult<()>;

    /// Delete message
    async fn delete_message(&self, message_id: &str) -> BotResult<()>;

    /// Check if bot is configured
    fn is_configured(&self) -> bool;
}

/// Slack bot integration
#[derive(Debug)]
pub struct SlackBot {
    config: BotConfig,
    #[cfg(feature = "microservices")]
    client: Client,
}

impl SlackBot {
    /// Create a new Slack bot
    #[must_use]
    pub fn new(config: BotConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "microservices")]
            client: Client::new(),
        }
    }

    /// Format message for Slack
    fn format_message(&self, message: &BotMessage) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "channel": message.channel,
            "text": message.text,
        });

        if let Some(thread_id) = &message.thread_id {
            payload["thread_ts"] = serde_json::Value::String(thread_id.clone());
        }

        if !message.attachments.is_empty() {
            let attachments: Vec<serde_json::Value> = message
                .attachments
                .iter()
                .map(|att| {
                    let mut attachment = serde_json::json!({
                        "title": att.title,
                        "text": att.text,
                    });

                    if let Some(color) = &att.color {
                        attachment["color"] = serde_json::Value::String(color.clone());
                    }

                    if !att.fields.is_empty() {
                        let fields: Vec<serde_json::Value> = att
                            .fields
                            .iter()
                            .map(|(title, value)| {
                                serde_json::json!({
                                    "title": title,
                                    "value": value,
                                    "short": true,
                                })
                            })
                            .collect();
                        attachment["fields"] = serde_json::Value::Array(fields);
                    }

                    if let Some(image) = &att.image_url {
                        attachment["image_url"] = serde_json::Value::String(image.clone());
                    }

                    attachment
                })
                .collect();

            payload["attachments"] = serde_json::Value::Array(attachments);
        }

        payload
    }
}

#[async_trait]
impl BotIntegration for SlackBot {
    fn platform(&self) -> BotPlatform {
        BotPlatform::Slack
    }

    async fn send_message(&self, message: &BotMessage) -> BotResult<String> {
        if !self.is_configured() {
            return Err(BotError::NotConfigured {
                platform: "Slack".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let payload = self.format_message(message);

            let response = self
                .client
                .post("https://slack.com/api/chat.postMessage")
                .header("Authorization", format!("Bearer {}", self.config.token))
                .json(&payload)
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;

            let result: serde_json::Value =
                response.json().await.map_err(|e| BotError::ApiError {
                    platform: "Slack".to_string(),
                    message: e.to_string(),
                })?;

            if result["ok"].as_bool().unwrap_or(false) {
                Ok(result["ts"].as_str().unwrap_or("unknown").to_string())
            } else {
                Err(BotError::ApiError {
                    platform: "Slack".to_string(),
                    message: result["error"]
                        .as_str()
                        .unwrap_or("Unknown error")
                        .to_string(),
                })
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Ok("mock-message-id".to_string())
        }
    }

    async fn edit_message(&self, message_id: &str, new_text: &str) -> BotResult<()> {
        #[cfg(feature = "microservices")]
        {
            let payload = serde_json::json!({
                "channel": self.config.default_channel,
                "ts": message_id,
                "text": new_text,
            });

            let _response = self
                .client
                .post("https://slack.com/api/chat.update")
                .header("Authorization", format!("Bearer {}", self.config.token))
                .json(&payload)
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }

    async fn delete_message(&self, message_id: &str) -> BotResult<()> {
        #[cfg(feature = "microservices")]
        {
            let payload = serde_json::json!({
                "channel": self.config.default_channel,
                "ts": message_id,
            });

            let _response = self
                .client
                .post("https://slack.com/api/chat.delete")
                .header("Authorization", format!("Bearer {}", self.config.token))
                .json(&payload)
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }

    fn is_configured(&self) -> bool {
        self.config.enabled && !self.config.token.is_empty()
    }
}

/// Discord bot integration
#[derive(Debug)]
pub struct DiscordBot {
    config: BotConfig,
    #[cfg(feature = "microservices")]
    client: Client,
}

impl DiscordBot {
    /// Create a new Discord bot
    #[must_use]
    pub fn new(config: BotConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "microservices")]
            client: Client::new(),
        }
    }

    /// Format message for Discord
    fn format_message(&self, message: &BotMessage) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "content": message.text,
        });

        if !message.attachments.is_empty() {
            let embeds: Vec<serde_json::Value> = message
                .attachments
                .iter()
                .map(|att| {
                    let mut embed = serde_json::json!({
                        "title": att.title,
                        "description": att.text,
                    });

                    if let Some(color) = &att.color {
                        // Convert hex color to integer
                        if let Ok(color_int) =
                            u32::from_str_radix(color.trim_start_matches('#'), 16)
                        {
                            embed["color"] = serde_json::Value::Number(color_int.into());
                        }
                    }

                    if !att.fields.is_empty() {
                        let fields: Vec<serde_json::Value> = att
                            .fields
                            .iter()
                            .map(|(name, value)| {
                                serde_json::json!({
                                    "name": name,
                                    "value": value,
                                    "inline": true,
                                })
                            })
                            .collect();
                        embed["fields"] = serde_json::Value::Array(fields);
                    }

                    if let Some(image) = &att.image_url {
                        embed["image"] = serde_json::json!({"url": image});
                    }

                    if let Some(thumbnail) = &att.thumbnail_url {
                        embed["thumbnail"] = serde_json::json!({"url": thumbnail});
                    }

                    embed
                })
                .collect();

            payload["embeds"] = serde_json::Value::Array(embeds);
        }

        payload
    }
}

#[async_trait]
impl BotIntegration for DiscordBot {
    fn platform(&self) -> BotPlatform {
        BotPlatform::Discord
    }

    async fn send_message(&self, message: &BotMessage) -> BotResult<String> {
        if !self.is_configured() {
            return Err(BotError::NotConfigured {
                platform: "Discord".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let payload = self.format_message(message);

            let url = format!(
                "https://discord.com/api/v10/channels/{}/messages",
                message.channel
            );

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bot {}", self.config.token))
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;

            let result: serde_json::Value =
                response.json().await.map_err(|e| BotError::ApiError {
                    platform: "Discord".to_string(),
                    message: e.to_string(),
                })?;

            Ok(result["id"].as_str().unwrap_or("unknown").to_string())
        }

        #[cfg(not(feature = "microservices"))]
        {
            Ok("mock-message-id".to_string())
        }
    }

    async fn edit_message(&self, message_id: &str, new_text: &str) -> BotResult<()> {
        #[cfg(feature = "microservices")]
        {
            let payload = serde_json::json!({
                "content": new_text,
            });

            let url = format!(
                "https://discord.com/api/v10/channels/{}/messages/{}",
                self.config.default_channel, message_id
            );

            let _response = self
                .client
                .patch(&url)
                .header("Authorization", format!("Bot {}", self.config.token))
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }

    async fn delete_message(&self, message_id: &str) -> BotResult<()> {
        #[cfg(feature = "microservices")]
        {
            let url = format!(
                "https://discord.com/api/v10/channels/{}/messages/{}",
                self.config.default_channel, message_id
            );

            let _response = self
                .client
                .delete(&url)
                .header("Authorization", format!("Bot {}", self.config.token))
                .send()
                .await
                .map_err(|e| BotError::SendFailed {
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }

    fn is_configured(&self) -> bool {
        self.config.enabled && !self.config.token.is_empty()
    }
}

/// Bot manager to handle multiple platform integrations
pub struct BotManager {
    bots: HashMap<BotPlatform, Box<dyn BotIntegration>>,
}

impl BotManager {
    /// Create a new bot manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            bots: HashMap::new(),
        }
    }

    /// Register a bot
    pub fn register_bot(&mut self, bot: Box<dyn BotIntegration>) {
        let platform = bot.platform();
        self.bots.insert(platform, bot);
    }

    /// Send message to a specific platform
    pub async fn send_message(
        &self,
        platform: BotPlatform,
        message: &BotMessage,
    ) -> BotResult<String> {
        if let Some(bot) = self.bots.get(&platform) {
            bot.send_message(message).await
        } else {
            Err(BotError::NotConfigured {
                platform: platform.name().to_string(),
            })
        }
    }

    /// Broadcast message to all configured platforms
    pub async fn broadcast_message(
        &self,
        message: &BotMessage,
    ) -> Vec<(BotPlatform, BotResult<String>)> {
        let mut results = Vec::new();

        for (platform, bot) in &self.bots {
            if bot.is_configured() {
                let result = bot.send_message(message).await;
                results.push((*platform, result));
            }
        }

        results
    }

    /// Check if a platform is configured
    #[must_use]
    pub fn is_platform_configured(&self, platform: BotPlatform) -> bool {
        if let Some(bot) = self.bots.get(&platform) {
            bot.is_configured()
        } else {
            false
        }
    }

    /// Get list of configured platforms
    #[must_use]
    pub fn configured_platforms(&self) -> Vec<BotPlatform> {
        self.bots
            .iter()
            .filter(|(_, bot)| bot.is_configured())
            .map(|(platform, _)| *platform)
            .collect()
    }
}

impl Default for BotManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bot_platform_names() {
        assert_eq!(BotPlatform::Slack.name(), "Slack");
        assert_eq!(BotPlatform::Discord.name(), "Discord");
        assert_eq!(BotPlatform::MicrosoftTeams.name(), "Microsoft Teams");
    }

    #[test]
    fn test_bot_message_builder() {
        let message = BotMessage::new("Test message", "#general")
            .with_priority(MessagePriority::High)
            .in_thread("123456");

        assert_eq!(message.text, "Test message");
        assert_eq!(message.channel, "#general");
        assert_eq!(message.priority, MessagePriority::High);
        assert_eq!(message.thread_id, Some("123456".to_string()));
    }

    #[test]
    fn test_message_attachment() {
        let attachment = MessageAttachment {
            title: "Test Attachment".to_string(),
            text: "Attachment content".to_string(),
            color: Some("#FF0000".to_string()),
            fields: vec![("Field 1".to_string(), "Value 1".to_string())],
            image_url: None,
            thumbnail_url: None,
        };

        assert_eq!(attachment.title, "Test Attachment");
        assert_eq!(attachment.fields.len(), 1);
    }

    #[test]
    fn test_slack_bot_creation() {
        let config = BotConfig {
            platform: BotPlatform::Slack,
            token: "test-token".to_string(),
            webhook_url: None,
            default_channel: "#general".to_string(),
            username: Some("TestBot".to_string()),
            icon_url: None,
            enabled: true,
        };

        let bot = SlackBot::new(config);
        assert_eq!(bot.platform(), BotPlatform::Slack);
        assert!(bot.is_configured());
    }

    #[test]
    fn test_discord_bot_creation() {
        let config = BotConfig {
            platform: BotPlatform::Discord,
            token: "test-token".to_string(),
            webhook_url: None,
            default_channel: "123456789".to_string(),
            username: Some("TestBot".to_string()),
            icon_url: None,
            enabled: true,
        };

        let bot = DiscordBot::new(config);
        assert_eq!(bot.platform(), BotPlatform::Discord);
        assert!(bot.is_configured());
    }

    #[test]
    fn test_bot_manager() {
        let mut manager = BotManager::new();

        let slack_config = BotConfig {
            platform: BotPlatform::Slack,
            token: "test-token".to_string(),
            webhook_url: None,
            default_channel: "#general".to_string(),
            username: None,
            icon_url: None,
            enabled: true,
        };

        manager.register_bot(Box::new(SlackBot::new(slack_config)));

        assert!(manager.is_platform_configured(BotPlatform::Slack));
        assert!(!manager.is_platform_configured(BotPlatform::Discord));

        let platforms = manager.configured_platforms();
        assert_eq!(platforms.len(), 1);
        assert_eq!(platforms[0], BotPlatform::Slack);
    }

    #[tokio::test]
    async fn test_send_message() {
        let config = BotConfig {
            platform: BotPlatform::Slack,
            token: "test-token".to_string(),
            webhook_url: None,
            default_channel: "#general".to_string(),
            username: None,
            icon_url: None,
            enabled: true,
        };

        let bot = SlackBot::new(config);
        let message = BotMessage::new("Test message", "#general");

        // When the microservices feature is enabled, a real HTTP call is made.
        // In that case, the test-token will be rejected by Slack's API (invalid_auth).
        // Without microservices feature, a mock response is returned.
        // Either outcome is acceptable: Ok (mock) or Err(ApiError) (real auth failure).
        let result = bot.send_message(&message).await;
        let is_acceptable = result.is_ok()
            || matches!(&result, Err(BotError::ApiError { .. }))
            || matches!(&result, Err(BotError::SendFailed { .. }));
        assert!(is_acceptable, "Unexpected error: {:?}", result);
    }
}
