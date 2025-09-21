//! Metadata management and indexing for datasets
//!
//! This module provides efficient metadata handling, indexing, and caching
//! capabilities for large-scale dataset operations.

pub mod cache;
pub mod index;
pub mod manifest;
pub mod simple_manifest;

pub use cache::{CacheConfig, CacheStrategy, MetadataCache};
pub use index::{DatasetIndex, IndexBuilder, IndexConfig};
pub use manifest::{DatasetManifest, ManifestConfig, ManifestGenerator};
pub use simple_manifest::{
    SimpleManifest, SimpleManifestGenerator, SimpleSampleEntry, SimpleStatistics,
};
