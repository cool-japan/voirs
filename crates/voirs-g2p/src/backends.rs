//! G2P backend implementations.

// TODO: Implement various G2P backends
// - Rule-based G2P (Festival, Flite-style)
// - Neural G2P (Transformer, seq2seq)
// - Hybrid approaches
// - Language-specific backends

pub mod rule_based;
pub mod neural;
pub mod hybrid;
pub mod registry;
pub mod openjtalk;

pub use rule_based::RuleBasedG2p;
pub use registry::{BackendRegistry, BackendInfo, RegistryG2p};
pub use neural::phonetisaurus::PhonetisaurusG2p;
pub use openjtalk::OpenJTalkG2p;
pub use hybrid::{HybridG2p, SelectionStrategy, BackendConfig};

// Placeholder for future implementations