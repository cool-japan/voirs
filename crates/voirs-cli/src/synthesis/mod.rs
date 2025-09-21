//! Advanced synthesis features for VoiRS CLI.
//!
//! This module provides high-level synthesis capabilities that extend beyond
//! basic text-to-speech, including voice cloning, emotion control, and
//! multimodal synthesis features. These features enable more natural and
//! expressive speech synthesis.
//!
//! ## Features
//!
//! - **Voice Cloning**: Create custom voices from reference audio samples
//! - **Emotion Control**: Add emotional expression to synthesized speech
//! - **Multimodal Synthesis**: Integrate text, audio, and visual information
//!
//! ## Modules
//!
//! - [`cloning`]: Voice cloning and adaptation functionality
//! - [`emotion`]: Emotion control and expression synthesis
//! - [`multimodal`]: Multimodal synthesis combining different input types
//!
//! ## Example
//!
//! ```rust,no_run
//! use voirs_cli::synthesis::emotion::{EmotionSynthesizer, EmotionType};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut synthesizer = EmotionSynthesizer::new();
//! synthesizer.set_current_emotion(EmotionType::Joy);
//! let emotion = synthesizer.analyze_text_emotion("Hello world!");
//!
//! // Use with synthesis pipeline...
//! # Ok(())
//! # }
//! ```

pub mod cloning;
pub mod emotion;
pub mod multimodal;

pub use cloning::*;
pub use emotion::*;
pub use multimodal::*;
