//! Dataset implementations for various speech synthesis datasets
//!
//! This module provides implementations for popular speech synthesis datasets
//! including LJSpeech, VCTK, JVS, and others.

pub mod ljspeech;
pub mod vctk;
pub mod jvs;
pub mod custom;
pub mod dummy;

use crate::{DatasetSample, DatasetError, Result};
use std::path::Path;

/// Dataset loader trait
pub trait DatasetLoader {
    /// Load dataset from path
    fn load(&self, path: &Path) -> Result<Box<dyn crate::traits::Dataset<Sample = DatasetSample>>>;
    
    /// Get dataset name
    fn name(&self) -> &'static str;
    
    /// Get supported file extensions
    fn extensions(&self) -> &'static [&'static str];
}

/// Dataset registry for automatic dataset detection
pub struct DatasetRegistry {
    loaders: Vec<Box<dyn DatasetLoader>>,
}

impl DatasetRegistry {
    /// Create new dataset registry
    pub fn new() -> Self {
        Self {
            loaders: Vec::new(),
        }
    }
    
    /// Register dataset loader
    pub fn register<T: DatasetLoader + 'static>(&mut self, loader: T) {
        self.loaders.push(Box::new(loader));
    }
    
    /// Auto-detect and load dataset
    pub fn auto_load<P: AsRef<Path>>(&self, _path: P) -> Result<Box<dyn crate::traits::Dataset<Sample = DatasetSample>>> {
        // TODO: Implement auto-detection based on directory structure
        Err(DatasetError::LoadError("Auto-detection not implemented".to_string()))
    }
}

impl Default for DatasetRegistry {
    fn default() -> Self {
        Self::new()
    }
}