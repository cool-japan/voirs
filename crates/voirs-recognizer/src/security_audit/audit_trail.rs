// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Audit trail logging and management

use super::{Result, SecurityAuditError};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

/// Audit event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication event
    Authentication,
    /// Authorization event
    Authorization,
    /// Data access
    DataAccess,
    /// Configuration change
    ConfigurationChange,
    /// Model access
    ModelAccess,
    /// Sensitive operation
    SensitiveOperation,
}

/// Audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: AuditEventType,
    /// User ID
    pub user_id: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Resource accessed
    pub resource: String,
    /// Action performed
    pub action: String,
    /// Result (success/failure)
    pub success: bool,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

/// Audit trail manager
pub struct AuditTrailManager {
    events: Arc<RwLock<Vec<AuditEvent>>>,
    max_events: usize,
}

impl AuditTrailManager {
    /// Create a new audit trail manager
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events,
        }
    }

    /// Log an audit event
    pub fn log_event(&self, event: AuditEvent) -> Result<()> {
        info!(
            "Audit event: {:?} on {} by {:?}",
            event.event_type, event.resource, event.user_id
        );

        let mut events = self.events.write();

        // Maintain max_events limit
        if events.len() >= self.max_events {
            events.remove(0);
        }

        events.push(event);

        Ok(())
    }

    /// Get all events
    #[must_use]
    pub fn get_events(&self) -> Vec<AuditEvent> {
        self.events.read().clone()
    }

    /// Get events by type
    #[must_use]
    pub fn get_events_by_type(&self, event_type: AuditEventType) -> Vec<AuditEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get events by user
    #[must_use]
    pub fn get_events_by_user(&self, user_id: &str) -> Vec<AuditEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect()
    }

    /// Clear all events
    pub fn clear(&self) {
        self.events.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event() -> AuditEvent {
        AuditEvent {
            id: "event-1".to_string(),
            event_type: AuditEventType::DataAccess,
            user_id: Some("user-123".to_string()),
            timestamp: Utc::now(),
            resource: "/api/data".to_string(),
            action: "read".to_string(),
            success: true,
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn test_audit_trail_basic() {
        let manager = AuditTrailManager::new(100);
        let event = create_test_event();

        manager.log_event(event.clone()).unwrap();

        let events = manager.get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event-1");
    }

    #[test]
    fn test_audit_trail_max_events() {
        let manager = AuditTrailManager::new(2);

        for i in 0..3 {
            let mut event = create_test_event();
            event.id = format!("event-{}", i);
            manager.log_event(event).unwrap();
        }

        let events = manager.get_events();
        assert_eq!(events.len(), 2); // Should only keep last 2
    }

    #[test]
    fn test_get_events_by_type() {
        let manager = AuditTrailManager::new(100);

        let mut event1 = create_test_event();
        event1.event_type = AuditEventType::Authentication;
        manager.log_event(event1).unwrap();

        let mut event2 = create_test_event();
        event2.event_type = AuditEventType::DataAccess;
        manager.log_event(event2).unwrap();

        let auth_events = manager.get_events_by_type(AuditEventType::Authentication);
        assert_eq!(auth_events.len(), 1);
    }
}
