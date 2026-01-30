//! # LZ4Codec - Trait Implementations
//!
//! This module contains trait implementations for `LZ4Codec`.
//!
/// ## Implemented Traits
///
/// - `CompressionCodec`
///
/// 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)
use super::types::LZ4Codec;
use super::types::*;

impl CompressionCodec for LZ4Codec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());
        let mut i = 0;
        while i < data.len() {
            let byte = data[i];
            let mut count = 1;
            while i + count < data.len() && data[i + count] == byte && count < 255 {
                count += 1;
            }
            compressed.push(count as u8);
            compressed.push(byte);
            i += count;
        }
        Ok(compressed)
    }
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if compressed_data.len() < 4 {
            return Err(CompressionError::InvalidData("Too short".to_string()));
        }
        let original_size = u32::from_le_bytes([
            compressed_data[0],
            compressed_data[1],
            compressed_data[2],
            compressed_data[3],
        ]) as usize;
        let mut decompressed = Vec::with_capacity(original_size);
        let mut i = 4;
        while i + 1 < compressed_data.len() {
            let count = compressed_data[i] as usize;
            let byte = compressed_data[i + 1];
            decompressed.extend(vec![byte; count]);
            i += 2;
        }
        Ok(decompressed)
    }
    fn estimate_ratio(&self, _data: &[u8]) -> f32 {
        0.7
    }
    fn info(&self) -> CompressionCodecInfo {
        CompressionCodecInfo {
            name: "LZ4".to_string(),
            speed: CompressionSpeed::Fast,
            typical_ratio: 0.7,
            lossless: true,
        }
    }
}
