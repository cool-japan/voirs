//! Language Server Protocol (LSP) implementation for VoiRS
//!
//! Provides IDE integration for SSML editing, voice management, and synthesis configuration.
//! Supports features like autocompletion, diagnostics, hover information, and code actions.

pub mod capabilities;
pub mod code_actions;
pub mod completion;
pub mod diagnostics;
pub mod formatting;
pub mod hover;
pub mod protocol;
pub mod server;
pub mod text_document;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub use server::LspServer;

/// LSP server state
pub struct LspState {
    /// Open text documents
    pub documents: Arc<RwLock<HashMap<String, TextDocument>>>,

    /// Available voices cache
    pub voices: Arc<RwLock<Vec<VoiceInfo>>>,

    /// Server capabilities
    pub capabilities: ServerCapabilities,
}

impl LspState {
    /// Create a new LSP state
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            voices: Arc::new(RwLock::new(Vec::new())),
            capabilities: ServerCapabilities::default(),
        }
    }

    /// Open a document
    pub async fn open_document(&self, uri: String, text: String, language_id: String) {
        let document = TextDocument::new(uri.clone(), text, language_id);
        self.documents.write().await.insert(uri, document);
    }

    /// Update a document
    pub async fn update_document(&self, uri: &str, text: String, version: i32) {
        if let Some(doc) = self.documents.write().await.get_mut(uri) {
            doc.update(text, version);
        }
    }

    /// Close a document
    pub async fn close_document(&self, uri: &str) {
        self.documents.write().await.remove(uri);
    }

    /// Get a document
    pub async fn get_document(&self, uri: &str) -> Option<TextDocument> {
        self.documents.read().await.get(uri).cloned()
    }
}

impl Default for LspState {
    fn default() -> Self {
        Self::new()
    }
}

/// Text document representation
#[derive(Debug, Clone)]
pub struct TextDocument {
    /// Document URI
    pub uri: String,

    /// Document text content
    pub text: String,

    /// Language identifier (ssml, voirs-config, etc.)
    pub language_id: String,

    /// Document version
    pub version: i32,
}

impl TextDocument {
    /// Create a new text document
    pub fn new(uri: String, text: String, language_id: String) -> Self {
        Self {
            uri,
            text,
            language_id,
            version: 1,
        }
    }

    /// Update document content
    pub fn update(&mut self, text: String, version: i32) {
        self.text = text;
        self.version = version;
    }

    /// Get line count
    pub fn line_count(&self) -> usize {
        self.text.lines().count()
    }

    /// Get line at position
    pub fn get_line(&self, line: usize) -> Option<&str> {
        self.text.lines().nth(line)
    }

    /// Get character at position
    pub fn get_char_at(&self, line: usize, character: usize) -> Option<char> {
        self.get_line(line)?.chars().nth(character)
    }
}

/// Voice information for completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Voice ID
    pub id: String,

    /// Voice name
    pub name: String,

    /// Language code
    pub language: String,

    /// Voice description
    pub description: String,

    /// Supported features
    pub features: Vec<String>,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Text document sync kind
    pub text_document_sync: TextDocumentSyncKind,

    /// Supports completion
    pub completion_provider: bool,

    /// Supports hover
    pub hover_provider: bool,

    /// Supports diagnostics
    pub diagnostic_provider: bool,

    /// Supports code actions
    pub code_action_provider: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            text_document_sync: TextDocumentSyncKind::Full,
            completion_provider: true,
            hover_provider: true,
            diagnostic_provider: true,
            code_action_provider: true,
        }
    }
}

/// Text document sync kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDocumentSyncKind {
    /// Documents should not be synced
    None,

    /// Documents are synced by sending full content
    Full,

    /// Documents are synced by sending incremental updates
    Incremental,
}

/// Position in a text document
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// Line number (0-indexed)
    pub line: u32,

    /// Character offset (0-indexed)
    pub character: u32,
}

impl Position {
    /// Create a new position
    pub fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }
}

/// Range in a text document
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    /// Start position
    pub start: Position,

    /// End position (exclusive)
    pub end: Position,
}

impl Range {
    /// Create a new range
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    /// Create a single-line range
    pub fn single_line(line: u32, start_char: u32, end_char: u32) -> Self {
        Self {
            start: Position::new(line, start_char),
            end: Position::new(line, end_char),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lsp_state_creation() {
        let state = LspState::new();
        assert!(state.documents.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_open_document() {
        let state = LspState::new();
        state
            .open_document(
                "file:///test.ssml".to_string(),
                "<speak>Hello</speak>".to_string(),
                "ssml".to_string(),
            )
            .await;

        let docs = state.documents.read().await;
        assert_eq!(docs.len(), 1);
        assert!(docs.contains_key("file:///test.ssml"));
    }

    #[tokio::test]
    async fn test_update_document() {
        let state = LspState::new();
        state
            .open_document(
                "file:///test.ssml".to_string(),
                "<speak>Hello</speak>".to_string(),
                "ssml".to_string(),
            )
            .await;

        state
            .update_document("file:///test.ssml", "<speak>World</speak>".to_string(), 2)
            .await;

        let doc = state.get_document("file:///test.ssml").await.unwrap();
        assert_eq!(doc.text, "<speak>World</speak>");
        assert_eq!(doc.version, 2);
    }

    #[tokio::test]
    async fn test_close_document() {
        let state = LspState::new();
        state
            .open_document(
                "file:///test.ssml".to_string(),
                "<speak>Hello</speak>".to_string(),
                "ssml".to_string(),
            )
            .await;

        state.close_document("file:///test.ssml").await;

        assert!(state.get_document("file:///test.ssml").await.is_none());
    }

    #[test]
    fn test_text_document_line_count() {
        let doc = TextDocument::new(
            "file:///test.txt".to_string(),
            "line 1\nline 2\nline 3".to_string(),
            "text".to_string(),
        );
        assert_eq!(doc.line_count(), 3);
    }

    #[test]
    fn test_position_creation() {
        let pos = Position::new(5, 10);
        assert_eq!(pos.line, 5);
        assert_eq!(pos.character, 10);
    }

    #[test]
    fn test_range_creation() {
        let range = Range::single_line(3, 5, 15);
        assert_eq!(range.start.line, 3);
        assert_eq!(range.start.character, 5);
        assert_eq!(range.end.line, 3);
        assert_eq!(range.end.character, 15);
    }

    #[test]
    fn test_server_capabilities_default() {
        let caps = ServerCapabilities::default();
        assert!(caps.completion_provider);
        assert!(caps.hover_provider);
        assert!(caps.diagnostic_provider);
        assert_eq!(caps.text_document_sync, TextDocumentSyncKind::Full);
    }
}
