//! # NoCompressionCodec - Trait Implementations
//!
//! This module contains trait implementations for `NoCompressionCodec`.
//!
/// ## Implemented Traits
///
/// - `CompressionCodec`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::NoCompressionCodec;
use super::types::*;

impl CompressionCodec for NoCompressionCodec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(compressed_data.to_vec())
    }
    fn estimate_ratio(&self, _data: &[u8]) -> f32 {
        1.0
    }
    fn info(&self) -> CompressionCodecInfo {
        CompressionCodecInfo {
            name: "None".to_string(),
            speed: CompressionSpeed::VeryFast,
            typical_ratio: 1.0,
            lossless: true,
        }
    }
}
