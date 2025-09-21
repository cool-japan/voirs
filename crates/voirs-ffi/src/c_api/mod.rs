//! C API module organization.

pub mod allocator;
pub mod audio;
pub mod config;
pub mod convert;
pub mod core;
pub mod synthesis;
pub mod threading;
pub mod utils;
pub mod voice;
pub mod zero_copy;

// Re-export all public functions
pub use allocator::*;
pub use audio::*;
pub use config::*;
pub use convert::*;
pub use core::*;
pub use synthesis::*;
pub use threading::*;
pub use utils::*;
pub use voice::*;
pub use zero_copy::*;
