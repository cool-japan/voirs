//! Core voice conversion functionality

// Module declarations
mod converter;
mod model_ops;
mod quality;
mod robustness;
mod signal_processing;
mod speaker_conversion;
#[cfg(test)]
mod tests;
mod type_conversions;
mod types;

// Re-export public API
pub use converter::VoiceConverter;
pub use types::{AudioFeatures, ConversionStats, QualityMetrics, VoiceConverterBuilder};
