//! Speaker characteristics and emotional analysis implementation
//!
//! This module provides speaker analysis capabilities including:
//! - Gender classification
//! - Age estimation
//! - Voice quality analysis
//! - Accent detection
//! - Emotional analysis

mod analyzer;
mod diarizer;

#[cfg(test)]
mod tests;

pub use analyzer::SpeakerAnalyzer;
pub use diarizer::{
    SpeakerChangePoint, SpeakerDiarizationResult, SpeakerDiarizer, SpeakerEmbedding, SpeakerSegment,
};
