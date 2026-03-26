//! Model format detection and unified loading utilities.

use std::path::Path;

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedFormat {
    /// ONNX model format (.onnx)
    Onnx,
    /// SafeTensors format (.safetensors)
    SafeTensors,
    /// PyTorch format (.pt, .pth, .bin)
    PyTorch,
    /// NumPy format (.npz, .npy)
    NumPy,
    /// Unknown format
    Unknown,
}

impl std::fmt::Display for DetectedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectedFormat::Onnx => write!(f, "ONNX"),
            DetectedFormat::SafeTensors => write!(f, "SafeTensors"),
            DetectedFormat::PyTorch => write!(f, "PyTorch"),
            DetectedFormat::NumPy => write!(f, "NumPy"),
            DetectedFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Utility for detecting model file formats.
pub struct ModelFormatDetector;

impl ModelFormatDetector {
    /// Detect model format from file extension.
    pub fn from_extension(path: &Path) -> DetectedFormat {
        match path.extension().and_then(|e| e.to_str()) {
            Some("onnx") => DetectedFormat::Onnx,
            Some("safetensors") => DetectedFormat::SafeTensors,
            Some("pt") | Some("pth") | Some("bin") => DetectedFormat::PyTorch,
            Some("npz") | Some("npy") => DetectedFormat::NumPy,
            _ => DetectedFormat::Unknown,
        }
    }

    /// Detect model format from file magic bytes.
    pub fn from_bytes(bytes: &[u8]) -> DetectedFormat {
        if bytes.len() < 8 {
            return DetectedFormat::Unknown;
        }

        // ONNX: protobuf with field 1 (ir_version) as first field
        // The typical ONNX file starts with 0x08 (field 1, varint)
        if bytes[0] == 0x08 {
            return DetectedFormat::Onnx;
        }

        // SafeTensors: starts with a JSON header length (little-endian u64)
        // followed by '{' character in the JSON header
        if bytes.len() >= 16 {
            let header_len = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            // Reasonable header size and starts with '{'
            if header_len > 0 && header_len < 100_000_000 && bytes.len() > 8 && bytes[8] == b'{' {
                return DetectedFormat::SafeTensors;
            }
        }

        // NumPy NPZ: ZIP file magic bytes
        if bytes[0] == 0x50 && bytes[1] == 0x4B {
            return DetectedFormat::NumPy;
        }

        // NumPy NPY: magic string "\x93NUMPY"
        if bytes.len() >= 6 && bytes[0] == 0x93 && &bytes[1..6] == b"NUMPY" {
            return DetectedFormat::NumPy;
        }

        DetectedFormat::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_by_extension() {
        assert_eq!(
            ModelFormatDetector::from_extension(Path::new("model.onnx")),
            DetectedFormat::Onnx
        );
        assert_eq!(
            ModelFormatDetector::from_extension(Path::new("model.safetensors")),
            DetectedFormat::SafeTensors
        );
        assert_eq!(
            ModelFormatDetector::from_extension(Path::new("model.pt")),
            DetectedFormat::PyTorch
        );
        assert_eq!(
            ModelFormatDetector::from_extension(Path::new("model.npz")),
            DetectedFormat::NumPy
        );
        assert_eq!(
            ModelFormatDetector::from_extension(Path::new("model.txt")),
            DetectedFormat::Unknown
        );
    }

    #[test]
    fn test_format_detection_by_bytes() {
        // ONNX-like start
        let onnx_bytes = [0x08, 0x07, 0x12, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            ModelFormatDetector::from_bytes(&onnx_bytes),
            DetectedFormat::Onnx
        );

        // SafeTensors-like start
        let mut st_bytes = vec![0; 16];
        st_bytes[0..8].copy_from_slice(&100u64.to_le_bytes());
        st_bytes[8] = b'{';
        assert_eq!(
            ModelFormatDetector::from_bytes(&st_bytes),
            DetectedFormat::SafeTensors
        );

        // NPZ (ZIP) magic
        let npz_bytes = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            ModelFormatDetector::from_bytes(&npz_bytes),
            DetectedFormat::NumPy
        );

        // Too short
        assert_eq!(
            ModelFormatDetector::from_bytes(&[0x00, 0x01]),
            DetectedFormat::Unknown
        );
    }

    #[test]
    fn test_detected_format_display() {
        assert_eq!(DetectedFormat::Onnx.to_string(), "ONNX");
        assert_eq!(DetectedFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(DetectedFormat::Unknown.to_string(), "Unknown");
    }
}
