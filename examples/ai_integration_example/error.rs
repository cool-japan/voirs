//! Error type shared across the AI integration example.

// Error types
#[derive(Debug)]
pub enum AIVoiceError {
    LLMError(String),
    VoiceSynthesisError(String),
    ConversationError(String),
    PersonalityError(String),
    StreamingError(String),
    ContentGenerationError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for AIVoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AIVoiceError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            AIVoiceError::VoiceSynthesisError(msg) => write!(f, "Voice synthesis error: {}", msg),
            AIVoiceError::ConversationError(msg) => write!(f, "Conversation error: {}", msg),
            AIVoiceError::PersonalityError(msg) => write!(f, "Personality error: {}", msg),
            AIVoiceError::StreamingError(msg) => write!(f, "Streaming error: {}", msg),
            AIVoiceError::ContentGenerationError(msg) => {
                write!(f, "Content generation error: {}", msg)
            }
            AIVoiceError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for AIVoiceError {}
