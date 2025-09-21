//! Compression utilities for cache storage

use crate::RecognitionError;
use std::io::{Read, Write};

/// Compress data using zlib compression
pub fn compress_data(data: &[u8], level: u8) -> Result<Vec<u8>, RecognitionError> {
    use std::io::Cursor;
    
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level.into()));
    encoder.write_all(data).map_err(|e| {
        RecognitionError::ResourceError {
            message: format!("Compression failed: {}", e),
            source: Some(Box::new(e)),
        }
    })?;
    
    encoder.finish().map_err(|e| {
        RecognitionError::ResourceError {
            message: format!("Compression finalization failed: {}", e),
            source: Some(Box::new(e)),
        }
    })
}

/// Decompress zlib-compressed data
pub fn decompress_data(compressed_data: &[u8]) -> Result<Vec<u8>, RecognitionError> {
    use std::io::Cursor;
    
    let mut decoder = flate2::read::ZlibDecoder::new(Cursor::new(compressed_data));
    let mut decompressed = Vec::new();
    
    decoder.read_to_end(&mut decompressed).map_err(|e| {
        RecognitionError::ResourceError {
            message: format!("Decompression failed: {}", e),
            source: Some(Box::new(e)),
        }
    })?;
    
    Ok(decompressed)
}

/// Calculate compression ratio
pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
    if original_size == 0 {
        return 1.0;
    }
    compressed_size as f64 / original_size as f64
}

/// Adaptive compression that chooses the best level
pub fn adaptive_compress(data: &[u8]) -> Result<Vec<u8>, RecognitionError> {
    // For small data, use faster compression
    let level = if data.len() < 1024 {
        1 // Fast compression for small data
    } else if data.len() < 1024 * 1024 {
        6 // Balanced compression for medium data
    } else {
        9 // Maximum compression for large data
    };
    
    compress_data(data, level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_decompression() {
        let original_data = b"This is test data for compression";
        
        let compressed = compress_data(original_data, 6).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        
        assert_eq!(original_data, decompressed.as_slice());
        assert!(compressed.len() < original_data.len()); // Should be compressed
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(100, 50);
        assert_eq!(ratio, 0.5);
        
        let ratio_zero = compression_ratio(0, 10);
        assert_eq!(ratio_zero, 1.0);
    }

    #[test]
    fn test_adaptive_compression() {
        let small_data = b"small";
        let large_data = vec![b'x'; 2 * 1024 * 1024]; // 2MB
        
        let compressed_small = adaptive_compress(small_data).unwrap();
        let compressed_large = adaptive_compress(&large_data).unwrap();
        
        // Both should decompress correctly
        let decompressed_small = decompress_data(&compressed_small).unwrap();
        let decompressed_large = decompress_data(&compressed_large).unwrap();
        
        assert_eq!(small_data, decompressed_small.as_slice());
        assert_eq!(large_data, decompressed_large);
    }
}