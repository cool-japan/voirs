//! Error types for the educational voice system.

// Error types
#[derive(Debug)]
pub enum EducationalVoiceError {
    LanguageLearningError(String),
    PronunciationAssessmentError(String),
    AdaptiveLearningError(String),
    SynthesisError(String),
    LessonManagementError(String),
    ProgressTrackingError(String),
    GamificationError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for EducationalVoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EducationalVoiceError::LanguageLearningError(msg) => {
                write!(f, "Language learning error: {}", msg)
            }
            EducationalVoiceError::PronunciationAssessmentError(msg) => {
                write!(f, "Pronunciation assessment error: {}", msg)
            }
            EducationalVoiceError::AdaptiveLearningError(msg) => {
                write!(f, "Adaptive learning error: {}", msg)
            }
            EducationalVoiceError::SynthesisError(msg) => write!(f, "Synthesis error: {}", msg),
            EducationalVoiceError::LessonManagementError(msg) => {
                write!(f, "Lesson management error: {}", msg)
            }
            EducationalVoiceError::ProgressTrackingError(msg) => {
                write!(f, "Progress tracking error: {}", msg)
            }
            EducationalVoiceError::GamificationError(msg) => {
                write!(f, "Gamification error: {}", msg)
            }
            EducationalVoiceError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for EducationalVoiceError {}
