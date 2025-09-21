//! Trait adapters for integrating VoiRS component crates with the SDK.
//!
//! This module provides adapter implementations that bridge the differences
//! between component crate traits and SDK traits, enabling seamless integration
//! of individual VoiRS components into the unified SDK pipeline.

pub mod acoustic;
pub mod g2p;
pub mod vocoder;

pub use acoustic::AcousticAdapter;
pub use g2p::G2pAdapter;
pub use vocoder::VocoderAdapter;
