//! LSP protocol message handling

use serde::{Deserialize, Serialize};

/// LSP request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMessage {
    /// Protocol version
    pub jsonrpc: String,

    /// Request ID
    pub id: i64,

    /// Method name
    pub method: String,

    /// Optional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// LSP response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    /// Protocol version
    pub jsonrpc: String,

    /// Request ID
    pub id: i64,

    /// Result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// Error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}

/// LSP notification message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationMessage {
    /// Protocol version
    pub jsonrpc: String,

    /// Method name
    pub method: String,

    /// Optional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// LSP response error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code
    pub code: i32,

    /// Error message
    pub message: String,

    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_message_serialization() {
        let request = RequestMessage {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: "initialize".to_string(),
            params: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"initialize\""));
    }

    #[test]
    fn test_response_message_serialization() {
        let response = ResponseMessage {
            jsonrpc: "2.0".to_string(),
            id: 1,
            result: Some(serde_json::json!({"status": "ok"})),
            error: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"result\""));
    }
}
