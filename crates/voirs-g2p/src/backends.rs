//! G2P backend implementations.
//!
//! This module provides various G2P backend implementations:
//! - Rule-based G2P (context-aware phonological rules) ✅
//! - Neural G2P (LSTM, attention mechanisms, beam search) ✅
//! - Phonetisaurus (FST-based pronunciation generation) ✅
//! - Hybrid approaches (multi-backend ensemble strategies) ✅
//! - Language-specific backends (OpenJTalk for Japanese) ✅
//! - Backend registry (dynamic backend management) ✅

pub mod chinese_pinyin;
pub mod hybrid;
pub mod japanese_dict;
pub mod neural;
pub mod openjtalk;
pub mod registry;
pub mod rule_based;

pub use chinese_pinyin::ChinesePinyinG2p;
pub use hybrid::{BackendConfig, HybridG2p, SelectionStrategy};
pub use japanese_dict::JapaneseDictG2p;
pub use neural::{LstmConfig, LstmTrainer, NeuralG2pBackend};
pub use openjtalk::OpenJTalkG2p;
pub use registry::{BackendInfo, BackendRegistry, RegistryG2p};
pub use rule_based::RuleBasedG2p;
