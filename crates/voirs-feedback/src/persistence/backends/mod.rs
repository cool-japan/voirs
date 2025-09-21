//! Persistence backend implementations

pub mod json_file;
pub mod memory;

// Database backends require the "persistence" feature
#[cfg(feature = "persistence")]
pub mod sqlite;

#[cfg(feature = "persistence")]
pub mod postgres;
