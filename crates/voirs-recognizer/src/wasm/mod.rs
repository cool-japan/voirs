#[cfg(feature = "wasm")]
mod recognizer;

#[cfg(feature = "wasm")]
mod worker;

#[cfg(feature = "wasm")]
mod streaming;

#[cfg(feature = "wasm")]
mod utils;

#[cfg(feature = "wasm")]
pub use recognizer::*;

#[cfg(feature = "wasm")]
pub use worker::*;

#[cfg(feature = "wasm")]
pub use streaming::*;

#[cfg(feature = "wasm")]
pub use utils::*;
