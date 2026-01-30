//! Auto-generated module structure

// Re-export parent module items needed by submodules
pub use crate::phonology::{
    find_similar_phoneme, get_features, has_feature, PhonologicalFeature, ProcessType,
};
pub use crate::{G2pError, LanguageCode, Phoneme, Result};
pub use serde::{Deserialize, Serialize};

// Trait definition
pub trait PhonologicalProcess: Send + Sync {
    /// Apply the process to a phoneme sequence
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>>;
    /// Get the process type
    fn process_type(&self) -> ProcessType;
    /// Get the process name
    fn name(&self) -> &str {
        "Unknown"
    }
}

pub mod assimilationprocess_traits;
pub mod elisionprocess_traits;
pub mod finaldevoicingprocess_traits;
pub mod fortitionprocess_traits;
pub mod functions;
pub mod gdroppingprocess_traits;
pub mod hdroppingprocess_traits;
pub mod lenitionprocess_traits;
pub mod liaisonprocess_traits;
pub mod lvocalizationprocess_traits;
pub mod nasalizationprocess_traits;
pub mod palatalizationprocess_traits;
pub mod processconfig_traits;
pub mod rdroppingprocess_traits;
pub mod tflappingprocess_traits;
pub mod tglottalingprocess_traits;
pub mod thfrontingprocess_traits;
pub mod types;
pub mod voicingassimilationprocess_traits;
pub mod voweldevoicingprocess_traits;
pub mod vowelreductionprocess_traits;
pub mod yodcoalescenceprocess_traits;

// Re-export all types
pub use assimilationprocess_traits::*;
pub use elisionprocess_traits::*;
pub use finaldevoicingprocess_traits::*;
pub use fortitionprocess_traits::*;
pub use functions::*;
pub use gdroppingprocess_traits::*;
pub use hdroppingprocess_traits::*;
pub use lenitionprocess_traits::*;
pub use liaisonprocess_traits::*;
pub use lvocalizationprocess_traits::*;
pub use nasalizationprocess_traits::*;
pub use palatalizationprocess_traits::*;
pub use processconfig_traits::*;
pub use rdroppingprocess_traits::*;
pub use tflappingprocess_traits::*;
pub use tglottalingprocess_traits::*;
pub use thfrontingprocess_traits::*;
pub use types::*;
pub use voicingassimilationprocess_traits::*;
pub use voweldevoicingprocess_traits::*;
pub use vowelreductionprocess_traits::*;
pub use yodcoalescenceprocess_traits::*;
