//! LSP server implementation

use super::{LspState, Position, Range};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

/// LSP server
pub struct LspServer {
    /// Server state
    state: Arc<LspState>,

    /// Server running flag
    running: Arc<RwLock<bool>>,
}

impl LspServer {
    /// Create a new LSP server
    pub fn new() -> Self {
        Self {
            state: Arc::new(LspState::new()),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the LSP server
    pub async fn start(&self) -> Result<(), LspError> {
        *self.running.write().await = true;

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);

        eprintln!("VoiRS LSP server started");

        while *self.running.read().await {
            let mut headers = String::new();

            // Read headers
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).await? == 0 {
                    return Ok(()); // EOF
                }

                if line == "\r\n" || line == "\n" {
                    break; // End of headers
                }

                headers.push_str(&line);
            }

            // Parse Content-Length header
            let content_length = self.parse_content_length(&headers)?;

            // Read content
            let mut content = vec![0u8; content_length];
            tokio::io::AsyncReadExt::read_exact(&mut reader, &mut content).await?;

            let content_str =
                String::from_utf8(content).map_err(|e| LspError::InvalidUtf8(e.to_string()))?;

            // Process request
            if let Some(response) = self.handle_request(&content_str).await? {
                // Send response
                let response_json = serde_json::to_string(&response)?;
                let response_msg = format!(
                    "Content-Length: {}\r\n\r\n{}",
                    response_json.len(),
                    response_json
                );

                stdout.write_all(response_msg.as_bytes()).await?;
                stdout.flush().await?;
            }
        }

        Ok(())
    }

    /// Stop the LSP server
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    /// Parse Content-Length header
    fn parse_content_length(&self, headers: &str) -> Result<usize, LspError> {
        for line in headers.lines() {
            if line.starts_with("Content-Length:") {
                let length_str = line.trim_start_matches("Content-Length:").trim();
                return length_str
                    .parse()
                    .map_err(|_| LspError::InvalidContentLength(line.to_string()));
            }
        }

        Err(LspError::MissingContentLength)
    }

    /// Handle an LSP request
    async fn handle_request(&self, content: &str) -> Result<Option<serde_json::Value>, LspError> {
        let request: serde_json::Value = serde_json::from_str(content)?;

        let method = request["method"].as_str().ok_or(LspError::MissingMethod)?;

        match method {
            "initialize" => Ok(Some(self.handle_initialize(&request).await?)),
            "initialized" => Ok(None), // Notification, no response
            "shutdown" => Ok(Some(self.handle_shutdown().await?)),
            "exit" => {
                self.stop().await;
                Ok(None)
            }
            "textDocument/didOpen" => {
                self.handle_did_open(&request).await?;
                Ok(None)
            }
            "textDocument/didChange" => {
                self.handle_did_change(&request).await?;
                Ok(None)
            }
            "textDocument/didClose" => {
                self.handle_did_close(&request).await?;
                Ok(None)
            }
            "textDocument/completion" => Ok(Some(self.handle_completion(&request).await?)),
            "textDocument/hover" => Ok(Some(self.handle_hover(&request).await?)),
            "textDocument/codeAction" => Ok(Some(self.handle_code_action(&request).await?)),
            "textDocument/formatting" => Ok(Some(self.handle_formatting(&request).await?)),
            "textDocument/rangeFormatting" => {
                Ok(Some(self.handle_range_formatting(&request).await?))
            }
            "textDocument/onTypeFormatting" => {
                Ok(Some(self.handle_on_type_formatting(&request).await?))
            }
            _ => {
                eprintln!("Unhandled method: {}", method);
                Ok(None)
            }
        }
    }

    /// Handle initialize request
    async fn handle_initialize(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "capabilities": {
                    "textDocumentSync": 1, // Full sync
                    "completionProvider": {
                        "triggerCharacters": ["<", " ", "=", "\""]
                    },
                    "hoverProvider": true,
                    "diagnosticProvider": true,
                    "codeActionProvider": {
                        "codeActionKinds": [
                            "quickfix",
                            "refactor",
                            "refactor.rewrite",
                            "source"
                        ]
                    },
                    "documentFormattingProvider": true,
                    "documentRangeFormattingProvider": true,
                    "documentOnTypeFormattingProvider": {
                        "firstTriggerCharacter": ">",
                        "moreTriggerCharacter": ["}","]"]
                    }
                },
                "serverInfo": {
                    "name": "voirs-lsp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        }))
    }

    /// Handle shutdown request
    async fn handle_shutdown(&self) -> Result<serde_json::Value, LspError> {
        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "result": null
        }))
    }

    /// Handle textDocument/didOpen notification
    async fn handle_did_open(&self, request: &serde_json::Value) -> Result<(), LspError> {
        let params = &request["params"];
        let text_document = &params["textDocument"];

        let uri = text_document["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?
            .to_string();

        let text = text_document["text"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("text".to_string()))?
            .to_string();

        let language_id = text_document["languageId"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("languageId".to_string()))?
            .to_string();

        self.state.open_document(uri, text, language_id).await;

        Ok(())
    }

    /// Handle textDocument/didChange notification
    async fn handle_did_change(&self, request: &serde_json::Value) -> Result<(), LspError> {
        let params = &request["params"];
        let text_document = &params["textDocument"];
        let content_changes = &params["contentChanges"];

        let uri = text_document["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        let version = text_document["version"]
            .as_i64()
            .ok_or_else(|| LspError::MissingField("version".to_string()))?
            as i32;

        if let Some(change) = content_changes.get(0) {
            if let Some(text) = change["text"].as_str() {
                self.state
                    .update_document(uri, text.to_string(), version)
                    .await;
            }
        }

        Ok(())
    }

    /// Handle textDocument/didClose notification
    async fn handle_did_close(&self, request: &serde_json::Value) -> Result<(), LspError> {
        let params = &request["params"];
        let text_document = &params["textDocument"];

        let uri = text_document["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        self.state.close_document(uri).await;

        Ok(())
    }

    /// Handle textDocument/completion request
    async fn handle_completion(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();

        // Get position
        let params = &request["params"];
        let position = &params["position"];

        let line = position["line"]
            .as_u64()
            .ok_or_else(|| LspError::MissingField("line".to_string()))? as u32;

        let character = position["character"]
            .as_u64()
            .ok_or_else(|| LspError::MissingField("character".to_string()))?
            as u32;

        let pos = Position::new(line, character);

        // Get document URI
        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        // Get completion items
        let items = self.get_completion_items(uri, pos).await;

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": items
        }))
    }

    /// Handle textDocument/hover request
    async fn handle_hover(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": "VoiRS SSML Element\n\nHover information will be provided here."
                }
            }
        }))
    }

    /// Get completion items for a position
    async fn get_completion_items(&self, _uri: &str, _pos: Position) -> Vec<serde_json::Value> {
        // Basic SSML tag completions
        vec![
            serde_json::json!({
                "label": "speak",
                "kind": 14, // Snippet
                "detail": "SSML root element",
                "insertText": "<speak>$1</speak>",
                "insertTextFormat": 2 // Snippet
            }),
            serde_json::json!({
                "label": "voice",
                "kind": 14,
                "detail": "Voice selection",
                "insertText": "<voice name=\"$1\">$2</voice>",
                "insertTextFormat": 2
            }),
            serde_json::json!({
                "label": "prosody",
                "kind": 14,
                "detail": "Prosody control",
                "insertText": "<prosody rate=\"$1\" pitch=\"$2\">$3</prosody>",
                "insertTextFormat": 2
            }),
            serde_json::json!({
                "label": "break",
                "kind": 14,
                "detail": "Insert pause",
                "insertText": "<break time=\"${1:500ms}\"/>",
                "insertTextFormat": 2
            }),
        ]
    }

    /// Handle textDocument/codeAction request
    async fn handle_code_action(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();
        let params = &request["params"];

        // Get document URI and range
        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        let range_json = &params["range"];
        let range = Range::new(
            Position::new(
                range_json["start"]["line"].as_u64().unwrap_or(0) as u32,
                range_json["start"]["character"].as_u64().unwrap_or(0) as u32,
            ),
            Position::new(
                range_json["end"]["line"].as_u64().unwrap_or(0) as u32,
                range_json["end"]["character"].as_u64().unwrap_or(0) as u32,
            ),
        );

        // Get document
        let doc = self.state.get_document(uri).await;
        let actions = if let Some(document) = doc {
            super::code_actions::get_code_actions(&document.text, range)
        } else {
            Vec::new()
        };

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": actions
        }))
    }

    /// Handle textDocument/formatting request
    async fn handle_formatting(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();
        let params = &request["params"];

        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        let doc = self.state.get_document(uri).await;
        let edits = if let Some(document) = doc {
            super::formatting::format_document(&document.text, &document.language_id)
                .map(|edits| edits.iter().map(|e| e.to_json()).collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": edits
        }))
    }

    /// Handle textDocument/rangeFormatting request
    async fn handle_range_formatting(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();
        let params = &request["params"];

        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        let range_json = &params["range"];
        let range = Range::new(
            Position::new(
                range_json["start"]["line"].as_u64().unwrap_or(0) as u32,
                range_json["start"]["character"].as_u64().unwrap_or(0) as u32,
            ),
            Position::new(
                range_json["end"]["line"].as_u64().unwrap_or(0) as u32,
                range_json["end"]["character"].as_u64().unwrap_or(0) as u32,
            ),
        );

        let doc = self.state.get_document(uri).await;
        let edits = if let Some(document) = doc {
            super::formatting::format_range(&document.text, range, &document.language_id)
                .map(|edits| edits.iter().map(|e| e.to_json()).collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": edits
        }))
    }

    /// Handle textDocument/onTypeFormatting request
    async fn handle_on_type_formatting(
        &self,
        request: &serde_json::Value,
    ) -> Result<serde_json::Value, LspError> {
        let id = request["id"].clone();
        let params = &request["params"];

        let uri = params["textDocument"]["uri"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("uri".to_string()))?;

        let position_json = &params["position"];
        let position = Position::new(
            position_json["line"].as_u64().unwrap_or(0) as u32,
            position_json["character"].as_u64().unwrap_or(0) as u32,
        );

        let ch_str = params["ch"]
            .as_str()
            .ok_or_else(|| LspError::MissingField("ch".to_string()))?;
        let ch = ch_str.chars().next().unwrap_or('>');

        let doc = self.state.get_document(uri).await;
        let edits = if let Some(document) = doc {
            super::formatting::format_on_type(&document.text, position, ch, &document.language_id)
                .map(|edits| edits.iter().map(|e| e.to_json()).collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": edits
        }))
    }

    /// Get server state (for testing)
    #[cfg(test)]
    pub fn state(&self) -> Arc<LspState> {
        Arc::clone(&self.state)
    }
}

impl Default for LspServer {
    fn default() -> Self {
        Self::new()
    }
}

/// LSP error types
#[derive(Debug, thiserror::Error)]
pub enum LspError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),

    #[error("Missing Content-Length header")]
    MissingContentLength,

    #[error("Invalid Content-Length: {0}")]
    InvalidContentLength(String),

    #[error("Missing method field")]
    MissingMethod,

    #[error("Missing field: {0}")]
    MissingField(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_server_creation() {
        let server = LspServer::new();
        assert!(server.state.documents.try_read().is_ok());
    }

    #[tokio::test]
    async fn test_parse_content_length() {
        let server = LspServer::new();
        let headers = "Content-Length: 123\r\nContent-Type: application/json\r\n";

        let length = server.parse_content_length(headers).unwrap();
        assert_eq!(length, 123);
    }

    #[tokio::test]
    async fn test_parse_content_length_error() {
        let server = LspServer::new();
        let headers = "Content-Type: application/json\r\n";

        let result = server.parse_content_length(headers);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_completion_items() {
        let server = LspServer::new();
        let items = server
            .get_completion_items("file:///test.ssml", Position::new(0, 0))
            .await;

        assert!(!items.is_empty());
        assert!(items.iter().any(|item| item["label"] == "speak"));
        assert!(items.iter().any(|item| item["label"] == "voice"));
    }
}
