//! LSP server capabilities

/// Get server capabilities for initialization
pub fn get_server_capabilities() -> serde_json::Value {
    serde_json::json!({
        "textDocumentSync": {
            "openClose": true,
            "change": 1, // Full document sync
            "save": {
                "includeText": false
            }
        },
        "completionProvider": {
            "triggerCharacters": ["<", " ", "=", "\"", "-"],
            "resolveProvider": false
        },
        "hoverProvider": true,
        "diagnosticProvider": {
            "interFileDependencies": false,
            "workspaceDiagnostics": false
        },
        "codeActionProvider": true,
        "documentFormattingProvider": true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_server_capabilities() {
        let caps = get_server_capabilities();
        assert!(caps["completionProvider"].is_object());
        assert!(caps["hoverProvider"].is_boolean());
        assert_eq!(caps["hoverProvider"], true);
    }
}
