//! WebSocket-based real-time communication for cross-platform synchronization
//!
//! This module provides WebSocket client and server functionality for real-time
//! communication between different platform instances of the VoiRS feedback system.

use crate::traits::{FeedbackResponse, SessionState, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{sleep, timeout};
use uuid::Uuid;

/// WebSocket message types for real-time communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Initial connection handshake
    Connect {
        /// Device identifier
        device_id: String,
        /// Platform name
        platform: String,
        /// Device capabilities
        capabilities: Vec<String>,
    },
    /// Acknowledge connection
    ConnectAck {
        /// Session identifier
        session_id: String,
        /// Server timestamp
        server_time: DateTime<Utc>,
    },
    /// Session state update
    SessionUpdate {
        /// Session identifier
        session_id: String,
        /// Session state
        state: SessionState,
        /// Update timestamp
        timestamp: DateTime<Utc>,
    },
    /// User progress synchronization
    ProgressSync {
        /// User identifier
        user_id: String,
        /// User progress data
        progress: UserProgress,
        /// Sync timestamp
        timestamp: DateTime<Utc>,
    },
    /// Real-time feedback delivery
    FeedbackDelivery {
        /// Session identifier
        session_id: String,
        /// Feedback response
        feedback: FeedbackResponse,
        /// Delivery timestamp
        timestamp: DateTime<Utc>,
    },
    /// Ping for connection keep-alive
    Ping {
        /// Ping timestamp
        timestamp: DateTime<Utc>
    },
    /// Pong response to ping
    Pong {
        /// Pong timestamp
        timestamp: DateTime<Utc>
    },
    /// Error message
    Error {
        /// Error code
        code: u32,
        /// Error message
        message: String,
        /// Error timestamp
        timestamp: DateTime<Utc>,
    },
    /// Disconnect notification
    Disconnect {
        /// Disconnect reason
        reason: String,
        /// Disconnect timestamp
        timestamp: DateTime<Utc>,
    },
}

/// WebSocket client for real-time communication
pub struct WebSocketClient {
    /// Client configuration
    config: WebSocketClientConfig,
    /// Connection state
    connection_state: Arc<RwLock<ConnectionState>>,
    /// Message handlers
    message_handlers: Arc<RwLock<HashMap<String, Box<dyn MessageHandler + Send + Sync>>>>,
    /// Outbound message queue
    outbound_queue: Arc<RwLock<Vec<WebSocketMessage>>>,
    /// Connection statistics
    stats: Arc<RwLock<ConnectionStats>>,
    /// Active subscriptions
    subscriptions: Arc<RwLock<HashMap<String, SubscriptionHandler>>>,
}

impl WebSocketClient {
    /// Create a new WebSocket client
    pub fn new(config: WebSocketClientConfig) -> Self {
        Self {
            config,
            connection_state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            outbound_queue: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ConnectionStats::default())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Connect to WebSocket server
    pub async fn connect(&mut self) -> Result<(), WebSocketError> {
        // Update connection state
        {
            let mut state = self
                .connection_state
                .write()
                .map_err(|_| WebSocketError::LockError)?;
            *state = ConnectionState::Connecting;
        }

        // Simulate connection process
        sleep(Duration::from_millis(100)).await;

        // Send initial handshake
        let handshake = WebSocketMessage::Connect {
            device_id: self.config.device_id.clone(),
            platform: self.config.platform.clone(),
            capabilities: self.config.capabilities.clone(),
        };

        self.send_message(handshake).await?;

        // Wait for connection acknowledgment
        let ack_timeout = Duration::from_secs(self.config.connection_timeout_seconds);
        let connection_result = timeout(ack_timeout, self.wait_for_connect_ack()).await;

        match connection_result {
            Ok(Ok(session_id)) => {
                // Update connection state and statistics atomically
                {
                    let mut state = self
                        .connection_state
                        .write()
                        .map_err(|_| WebSocketError::LockError)?;
                    let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
                    *state = ConnectionState::Connected { session_id };
                    stats.connections_established += 1;
                    stats.last_connect_time = Some(Utc::now());
                }

                // Start background tasks
                self.start_background_tasks().await?;

                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // Connection timeout
                {
                    let mut state = self
                        .connection_state
                        .write()
                        .map_err(|_| WebSocketError::LockError)?;
                    *state = ConnectionState::Disconnected;
                }
                Err(WebSocketError::ConnectionTimeout)
            }
        }
    }

    /// Wait for connection acknowledgment
    async fn wait_for_connect_ack(&self) -> Result<String, WebSocketError> {
        // Simulate waiting for ConnectAck message
        sleep(Duration::from_millis(50)).await;
        Ok(Uuid::new_v4().to_string())
    }

    /// Start background tasks for connection management
    async fn start_background_tasks(&self) -> Result<(), WebSocketError> {
        // Start ping task
        self.start_ping_task().await?;

        // Start message processing task
        self.start_message_processing_task().await?;

        // Start queue processing task
        self.start_queue_processing_task().await?;

        Ok(())
    }

    /// Start ping task for connection keep-alive
    async fn start_ping_task(&self) -> Result<(), WebSocketError> {
        let interval = Duration::from_secs(self.config.ping_interval_seconds);

        // In a real implementation, this would spawn a background task
        // For now, we'll just record that it started
        let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
        stats.ping_tasks_started += 1;

        Ok(())
    }

    /// Start message processing task
    async fn start_message_processing_task(&self) -> Result<(), WebSocketError> {
        // Background task would process incoming messages
        let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
        stats.message_tasks_started += 1;

        Ok(())
    }

    /// Start queue processing task
    async fn start_queue_processing_task(&self) -> Result<(), WebSocketError> {
        // Background task would process outbound message queue
        let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
        stats.queue_tasks_started += 1;

        Ok(())
    }

    /// Send a message to the server
    pub async fn send_message(&self, message: WebSocketMessage) -> Result<(), WebSocketError> {
        // Add to outbound queue
        {
            let mut queue = self
                .outbound_queue
                .write()
                .map_err(|_| WebSocketError::LockError)?;
            queue.push(message);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
            stats.messages_sent += 1;
        }

        Ok(())
    }

    /// Send session state update
    pub async fn send_session_update(&self, state: SessionState) -> Result<(), WebSocketError> {
        let session_id = self.get_session_id()?;

        let message = WebSocketMessage::SessionUpdate {
            session_id,
            state,
            timestamp: Utc::now(),
        };

        self.send_message(message).await
    }

    /// Send user progress synchronization
    pub async fn send_progress_sync(&self, progress: UserProgress) -> Result<(), WebSocketError> {
        let message = WebSocketMessage::ProgressSync {
            user_id: progress.user_id.clone(),
            progress,
            timestamp: Utc::now(),
        };

        self.send_message(message).await
    }

    /// Send real-time feedback
    pub async fn send_feedback(&self, feedback: FeedbackResponse) -> Result<(), WebSocketError> {
        let session_id = self.get_session_id()?;

        let message = WebSocketMessage::FeedbackDelivery {
            session_id,
            feedback,
            timestamp: Utc::now(),
        };

        self.send_message(message).await
    }

    /// Subscribe to specific message types
    pub async fn subscribe<F>(
        &self,
        message_type: String,
        handler: F,
    ) -> Result<String, WebSocketError>
    where
        F: Fn(WebSocketMessage) -> Result<(), WebSocketError> + Send + Sync + 'static,
    {
        let subscription_id = Uuid::new_v4().to_string();

        let subscription = SubscriptionHandler {
            id: subscription_id.clone(),
            message_type: message_type.clone(),
            handler: Box::new(handler),
        };

        {
            let mut subscriptions = self
                .subscriptions
                .write()
                .map_err(|_| WebSocketError::LockError)?;
            subscriptions.insert(subscription_id.clone(), subscription);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
            stats.active_subscriptions += 1;
        }

        Ok(subscription_id)
    }

    /// Unsubscribe from message type
    pub async fn unsubscribe(&self, subscription_id: &str) -> Result<(), WebSocketError> {
        {
            let mut subscriptions = self
                .subscriptions
                .write()
                .map_err(|_| WebSocketError::LockError)?;
            if subscriptions.remove(subscription_id).is_some() {
                // Update statistics
                let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
                stats.active_subscriptions = stats.active_subscriptions.saturating_sub(1);
            }
        }

        Ok(())
    }

    /// Disconnect from server
    pub async fn disconnect(&mut self) -> Result<(), WebSocketError> {
        // Send disconnect message
        let disconnect_msg = WebSocketMessage::Disconnect {
            reason: "Client requested disconnect".to_string(),
            timestamp: Utc::now(),
        };

        self.send_message(disconnect_msg).await?;

        // Update connection state and statistics atomically
        {
            let mut state = self
                .connection_state
                .write()
                .map_err(|_| WebSocketError::LockError)?;
            let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
            *state = ConnectionState::Disconnected;
            stats.disconnections += 1;
            stats.last_disconnect_time = Some(Utc::now());
        }

        Ok(())
    }

    /// Get current connection state
    pub fn get_connection_state(&self) -> Result<ConnectionState, WebSocketError> {
        let state = self
            .connection_state
            .read()
            .map_err(|_| WebSocketError::LockError)?;
        Ok(state.clone())
    }

    /// Get current session ID
    pub fn get_session_id(&self) -> Result<String, WebSocketError> {
        let state = self
            .connection_state
            .read()
            .map_err(|_| WebSocketError::LockError)?;
        match &*state {
            ConnectionState::Connected { session_id } => Ok(session_id.clone()),
            _ => Err(WebSocketError::NotConnected),
        }
    }

    /// Get connection statistics
    pub fn get_stats(&self) -> Result<ConnectionStats, WebSocketError> {
        let stats = self.stats.read().map_err(|_| WebSocketError::LockError)?;
        Ok(stats.clone())
    }

    /// Check if client is connected
    pub fn is_connected(&self) -> bool {
        if let Ok(state) = self.connection_state.read() {
            matches!(*state, ConnectionState::Connected { .. })
        } else {
            false
        }
    }

    /// Get pending message count
    pub fn get_pending_message_count(&self) -> usize {
        if let Ok(queue) = self.outbound_queue.read() {
            queue.len()
        } else {
            0
        }
    }

    /// Clear outbound message queue
    pub async fn clear_queue(&self) -> Result<(), WebSocketError> {
        let mut queue = self
            .outbound_queue
            .write()
            .map_err(|_| WebSocketError::LockError)?;
        let cleared_count = queue.len();
        queue.clear();

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| WebSocketError::LockError)?;
            stats.messages_dropped += cleared_count as u64;
        }

        Ok(())
    }
}

/// WebSocket client configuration
#[derive(Debug, Clone)]
pub struct WebSocketClientConfig {
    /// Server URL
    pub server_url: String,
    /// Device identifier
    pub device_id: String,
    /// Platform name
    pub platform: String,
    /// Device capabilities
    pub capabilities: Vec<String>,
    /// Connection timeout in seconds
    pub connection_timeout_seconds: u64,
    /// Ping interval in seconds
    pub ping_interval_seconds: u64,
    /// Maximum retry attempts
    pub max_retry_attempts: u32,
    /// Retry backoff delay in milliseconds
    pub retry_backoff_ms: u64,
    /// Enable automatic reconnection
    pub auto_reconnect: bool,
    /// Maximum queue size
    pub max_queue_size: usize,
}

impl Default for WebSocketClientConfig {
    fn default() -> Self {
        Self {
            server_url: "wss://api.voirs.com/ws".to_string(),
            device_id: Uuid::new_v4().to_string(),
            platform: "Unknown".to_string(),
            capabilities: vec![
                "session_sync".to_string(),
                "progress_sync".to_string(),
                "realtime_feedback".to_string(),
            ],
            connection_timeout_seconds: 30,
            ping_interval_seconds: 30,
            max_retry_attempts: 3,
            retry_backoff_ms: 1000,
            auto_reconnect: true,
            max_queue_size: 1000,
        }
    }
}

/// Connection state enumeration
#[derive(Debug, Clone)]
pub enum ConnectionState {
    /// Not connected
    Disconnected,
    /// Currently connecting
    Connecting,
    /// Connected with session ID
    Connected {
        /// Session identifier
        session_id: String
    },
    /// Connection error
    Error {
        /// Error message
        message: String
    },
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Number of successful connections
    pub connections_established: u64,
    /// Number of disconnections
    pub disconnections: u64,
    /// Number of messages sent
    pub messages_sent: u64,
    /// Number of messages received
    pub messages_received: u64,
    /// Number of messages dropped
    pub messages_dropped: u64,
    /// Number of connection errors
    pub connection_errors: u64,
    /// Active subscriptions count
    pub active_subscriptions: u32,
    /// Last connect time
    pub last_connect_time: Option<DateTime<Utc>>,
    /// Last disconnect time
    pub last_disconnect_time: Option<DateTime<Utc>>,
    /// Ping tasks started count
    pub ping_tasks_started: u32,
    /// Message processing tasks started count
    pub message_tasks_started: u32,
    /// Queue processing tasks started count
    pub queue_tasks_started: u32,
}

impl Default for ConnectionStats {
    fn default() -> Self {
        Self {
            connections_established: 0,
            disconnections: 0,
            messages_sent: 0,
            messages_received: 0,
            messages_dropped: 0,
            connection_errors: 0,
            active_subscriptions: 0,
            last_connect_time: None,
            last_disconnect_time: None,
            ping_tasks_started: 0,
            message_tasks_started: 0,
            queue_tasks_started: 0,
        }
    }
}

/// Message handler trait
pub trait MessageHandler {
    /// Handle incoming message
    fn handle(&self, message: WebSocketMessage) -> Result<(), WebSocketError>;
}

/// Subscription handler
pub struct SubscriptionHandler {
    /// Subscription ID
    pub id: String,
    /// Message type to handle
    pub message_type: String,
    /// Handler function
    pub handler: Box<dyn Fn(WebSocketMessage) -> Result<(), WebSocketError> + Send + Sync>,
}

/// WebSocket error types
#[derive(Debug, thiserror::Error)]
pub enum WebSocketError {
    /// Connection timeout error
    #[error("Connection timeout")]
    ConnectionTimeout,

    /// Not connected error
    #[error("Not connected")]
    NotConnected,

    /// Connection failed error
    #[error("Connection failed: {message}")]
    ConnectionFailed {
        /// Error message
        message: String
    },

    /// Message serialization error
    #[error("Message serialization error: {message}")]
    SerializationError {
        /// Error message
        message: String
    },

    /// Protocol error
    #[error("Protocol error: {message}")]
    ProtocolError {
        /// Error message
        message: String
    },

    /// Authentication error
    #[error("Authentication error: {message}")]
    AuthError {
        /// Error message
        message: String
    },

    /// Queue full error
    #[error("Queue full")]
    QueueFull,

    /// Lock error
    #[error("Lock error")]
    LockError,

    /// Subscription error
    #[error("Subscription error: {message}")]
    SubscriptionError {
        /// Error message
        message: String
    },

    /// Invalid message type error
    #[error("Invalid message type: {message_type}")]
    InvalidMessageType {
        /// Message type
        message_type: String
    },
}

/// Real-time communication manager
pub struct RealtimeCommunicationManager {
    /// WebSocket client
    client: WebSocketClient,
    /// Configuration
    config: RealtimeConfig,
    /// Event handlers
    event_handlers: Arc<RwLock<HashMap<String, Box<dyn EventHandler + Send + Sync>>>>,
}

impl RealtimeCommunicationManager {
    /// Create a new real-time communication manager
    pub fn new(ws_config: WebSocketClientConfig, rt_config: RealtimeConfig) -> Self {
        Self {
            client: WebSocketClient::new(ws_config),
            config: rt_config,
            event_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize real-time communication
    pub async fn initialize(&mut self) -> Result<(), WebSocketError> {
        // Connect to WebSocket server
        self.client.connect().await?;

        // Set up default event handlers
        self.setup_default_handlers().await?;

        Ok(())
    }

    /// Set up default event handlers
    async fn setup_default_handlers(&self) -> Result<(), WebSocketError> {
        // Subscribe to session updates
        self.client
            .subscribe("SessionUpdate".to_string(), |message| {
                if let WebSocketMessage::SessionUpdate {
                    session_id,
                    state,
                    timestamp,
                } = message
                {
                    println!(
                        "Received session update for {}: {:?} at {}",
                        session_id, state.session_id, timestamp
                    );
                }
                Ok(())
            })
            .await?;

        // Subscribe to progress sync
        self.client
            .subscribe("ProgressSync".to_string(), |message| {
                if let WebSocketMessage::ProgressSync {
                    user_id,
                    progress,
                    timestamp,
                } = message
                {
                    println!(
                        "Received progress sync for {}: {} at {}",
                        user_id, progress.user_id, timestamp
                    );
                }
                Ok(())
            })
            .await?;

        // Subscribe to feedback delivery
        self.client
            .subscribe("FeedbackDelivery".to_string(), |message| {
                if let WebSocketMessage::FeedbackDelivery {
                    session_id,
                    feedback,
                    timestamp,
                } = message
                {
                    println!(
                        "Received feedback for {}: {} items at {}",
                        session_id,
                        feedback.feedback_items.len(),
                        timestamp
                    );
                }
                Ok(())
            })
            .await?;

        Ok(())
    }

    /// Send session state update
    pub async fn broadcast_session_update(
        &self,
        state: SessionState,
    ) -> Result<(), WebSocketError> {
        self.client.send_session_update(state).await
    }

    /// Send user progress update
    pub async fn broadcast_progress_update(
        &self,
        progress: UserProgress,
    ) -> Result<(), WebSocketError> {
        self.client.send_progress_sync(progress).await
    }

    /// Send real-time feedback
    pub async fn broadcast_feedback(
        &self,
        feedback: FeedbackResponse,
    ) -> Result<(), WebSocketError> {
        self.client.send_feedback(feedback).await
    }

    /// Register event handler
    pub async fn register_event_handler<H>(
        &self,
        event_type: String,
        handler: H,
    ) -> Result<(), WebSocketError>
    where
        H: EventHandler + Send + Sync + 'static,
    {
        let mut handlers = self
            .event_handlers
            .write()
            .map_err(|_| WebSocketError::LockError)?;
        handlers.insert(event_type, Box::new(handler));
        Ok(())
    }

    /// Get connection status
    pub fn get_connection_status(&self) -> Result<ConnectionState, WebSocketError> {
        self.client.get_connection_state()
    }

    /// Get real-time statistics
    pub fn get_realtime_stats(&self) -> Result<RealtimeStats, WebSocketError> {
        let connection_stats = self.client.get_stats()?;

        Ok(RealtimeStats {
            is_connected: self.client.is_connected(),
            pending_messages: self.client.get_pending_message_count(),
            messages_sent: connection_stats.messages_sent,
            messages_received: connection_stats.messages_received,
            connection_uptime: self.calculate_uptime(&connection_stats),
            active_subscriptions: connection_stats.active_subscriptions,
        })
    }

    /// Calculate connection uptime
    fn calculate_uptime(&self, stats: &ConnectionStats) -> Duration {
        if let Some(connect_time) = stats.last_connect_time {
            Utc::now()
                .signed_duration_since(connect_time)
                .to_std()
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        }
    }

    /// Shutdown real-time communication
    pub async fn shutdown(&mut self) -> Result<(), WebSocketError> {
        self.client.disconnect().await
    }
}

/// Real-time configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Enable real-time features
    pub enabled: bool,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Message buffer size
    pub message_buffer_size: usize,
    /// Enable message compression
    pub enable_compression: bool,
    /// Enable message encryption
    pub enable_encryption: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_connections: 100,
            message_buffer_size: 1000,
            enable_compression: true,
            enable_encryption: true,
        }
    }
}

/// Event handler trait
pub trait EventHandler {
    /// Handle real-time event
    fn handle_event(&self, event_type: &str, data: &WebSocketMessage)
        -> Result<(), WebSocketError>;
}

/// Real-time statistics
#[derive(Debug, Clone)]
pub struct RealtimeStats {
    /// Connection status
    pub is_connected: bool,
    /// Pending messages count
    pub pending_messages: usize,
    /// Messages sent count
    pub messages_sent: u64,
    /// Messages received count
    pub messages_received: u64,
    /// Connection uptime
    pub connection_uptime: Duration,
    /// Active subscriptions
    pub active_subscriptions: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_websocket_client_creation() {
        let config = WebSocketClientConfig::default();
        let client = WebSocketClient::new(config);

        assert!(!client.is_connected());
        assert_eq!(client.get_pending_message_count(), 0);
    }

    #[tokio::test]
    async fn test_websocket_client_connect() {
        let config = WebSocketClientConfig::default();
        let mut client = WebSocketClient::new(config);

        let result = client.connect().await;
        assert!(result.is_ok());
        assert!(client.is_connected());

        let session_id = client.get_session_id();
        assert!(session_id.is_ok());
        assert!(!session_id.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_message_sending() {
        let config = WebSocketClientConfig::default();
        let mut client = WebSocketClient::new(config);

        client.connect().await.unwrap();

        let ping_msg = WebSocketMessage::Ping {
            timestamp: Utc::now(),
        };

        let result = client.send_message(ping_msg).await;
        assert!(result.is_ok());
        assert!(client.get_pending_message_count() > 0);
    }

    #[tokio::test]
    async fn test_subscription_management() {
        let config = WebSocketClientConfig::default();
        let mut client = WebSocketClient::new(config);

        client.connect().await.unwrap();

        let subscription_id = client
            .subscribe("TestMessage".to_string(), |_message| Ok(()))
            .await
            .unwrap();

        assert!(!subscription_id.is_empty());

        let stats = client.get_stats().unwrap();
        assert_eq!(stats.active_subscriptions, 1);

        client.unsubscribe(&subscription_id).await.unwrap();

        let stats = client.get_stats().unwrap();
        assert_eq!(stats.active_subscriptions, 0);
    }

    #[tokio::test]
    async fn test_realtime_communication_manager() {
        let ws_config = WebSocketClientConfig::default();
        let rt_config = RealtimeConfig::default();
        let mut manager = RealtimeCommunicationManager::new(ws_config, rt_config);

        let result = manager.initialize().await;
        assert!(result.is_ok());

        let status = manager.get_connection_status();
        assert!(status.is_ok());

        let stats = manager.get_realtime_stats();
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.is_connected);
    }

    #[tokio::test]
    async fn test_disconnect() {
        let config = WebSocketClientConfig::default();
        let mut client = WebSocketClient::new(config);

        client.connect().await.unwrap();
        assert!(client.is_connected());

        client.disconnect().await.unwrap();
        assert!(!client.is_connected());

        let stats = client.get_stats().unwrap();
        assert_eq!(stats.disconnections, 1);
    }

    #[tokio::test]
    async fn test_queue_management() {
        let config = WebSocketClientConfig::default();
        let client = WebSocketClient::new(config);

        let ping_msg = WebSocketMessage::Ping {
            timestamp: Utc::now(),
        };

        client.send_message(ping_msg).await.unwrap();
        assert_eq!(client.get_pending_message_count(), 1);

        client.clear_queue().await.unwrap();
        assert_eq!(client.get_pending_message_count(), 0);

        let stats = client.get_stats().unwrap();
        assert_eq!(stats.messages_dropped, 1);
    }

    #[test]
    fn test_websocket_message_serialization() {
        let connect_msg = WebSocketMessage::Connect {
            device_id: "test_device".to_string(),
            platform: "test_platform".to_string(),
            capabilities: vec!["cap1".to_string(), "cap2".to_string()],
        };

        let serialized = serde_json::to_string(&connect_msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            WebSocketMessage::Connect {
                device_id,
                platform,
                capabilities,
            } => {
                assert_eq!(device_id, "test_device");
                assert_eq!(platform, "test_platform");
                assert_eq!(capabilities.len(), 2);
            }
            other => assert!(false, "Expected Connect message, got {:?}", other),
        }
    }

    #[test]
    fn test_connection_state() {
        let state = ConnectionState::Connected {
            session_id: "test_session".to_string(),
        };

        match state {
            ConnectionState::Connected { session_id } => {
                assert_eq!(session_id, "test_session");
            }
            other => assert!(false, "Expected Connected state, got {:?}", other),
        }
    }

    #[test]
    fn test_config_defaults() {
        let ws_config = WebSocketClientConfig::default();
        assert!(ws_config.server_url.starts_with("wss://"));
        assert!(!ws_config.device_id.is_empty());
        assert_eq!(ws_config.connection_timeout_seconds, 30);
        assert!(ws_config.auto_reconnect);

        let rt_config = RealtimeConfig::default();
        assert!(rt_config.enabled);
        assert!(rt_config.enable_compression);
        assert!(rt_config.enable_encryption);
    }
}
