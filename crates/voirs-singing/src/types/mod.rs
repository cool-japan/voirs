//! Type definitions for singing synthesis

pub mod core_types;
pub mod note_events;
pub mod request_response;
pub mod voice_types;

// Re-export all types for backwards compatibility
pub use core_types::{Articulation, BendCurve, BreathType, Dynamics, Expression, VoiceType};
pub use note_events::{BreathInfo, NoteEvent, PitchBend};
pub use request_response::{QualitySettings, SingingRequest, SingingResponse, SingingStats};
pub use voice_types::VoiceCharacteristics;
