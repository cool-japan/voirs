#[cfg(feature = "c-api")]
mod core;

#[cfg(feature = "c-api")]
mod types;

#[cfg(feature = "c-api")]
mod recognition;

#[cfg(feature = "c-api")]
mod streaming;

#[cfg(feature = "c-api")]
mod error;

#[cfg(feature = "c-api")]
mod memory;

#[cfg(feature = "c-api")]
pub use core::*;

#[cfg(feature = "c-api")]
pub use types::*;

#[cfg(feature = "c-api")]
pub use recognition::*;

#[cfg(feature = "c-api")]
pub use streaming::*;

#[cfg(feature = "c-api")]
pub use error::*;

#[cfg(feature = "c-api")]
pub use memory::*;
