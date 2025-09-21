//! Audio format detection utilities
//!
//! Provides functions to detect audio formats from file paths, extensions,
//! MIME types, and binary content headers.

use super::AudioFormat;
use crate::RecognitionError;
use std::path::Path;

/// Detect audio format from file path using extension
pub fn detect_format_from_path<P: AsRef<Path>>(path: P) -> Result<AudioFormat, RecognitionError> {
    let path = path.as_ref();

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return detect_format_from_extension(ext_str);
        }
    }

    Err(RecognitionError::UnsupportedFormat(format!(
        "Could not determine format from path: {path:?}"
    )))
}

/// Detect audio format from file extension
pub fn detect_format_from_extension(extension: &str) -> Result<AudioFormat, RecognitionError> {
    let ext = extension.to_lowercase();

    match ext.as_str() {
        "wav" | "wave" => Ok(AudioFormat::Wav),
        "flac" => Ok(AudioFormat::Flac),
        "mp3" => Ok(AudioFormat::Mp3),
        "ogg" | "oga" => Ok(AudioFormat::Ogg),
        "m4a" | "aac" | "mp4" => Ok(AudioFormat::M4a),
        _ => Ok(AudioFormat::Unknown),
    }
}

/// Detect audio format from MIME type
pub fn detect_format_from_mime(mime: &str) -> Result<AudioFormat, RecognitionError> {
    let mime = mime.to_lowercase();

    match mime.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => Ok(AudioFormat::Wav),
        "audio/flac" | "audio/x-flac" => Ok(AudioFormat::Flac),
        "audio/mpeg" | "audio/mp3" => Ok(AudioFormat::Mp3),
        "audio/ogg" | "application/ogg" => Ok(AudioFormat::Ogg),
        "audio/mp4" | "audio/aac" => Ok(AudioFormat::M4a),
        _ => Ok(AudioFormat::Unknown),
    }
}

/// Detect audio format from binary content by examining headers
pub fn detect_format_from_bytes(data: &[u8]) -> Result<AudioFormat, RecognitionError> {
    if data.len() < 4 {
        return Ok(AudioFormat::Unknown);
    }

    // Check WAV format (RIFF header)
    if data.starts_with(b"RIFF") && data.len() >= 12 && data[8..12] == *b"WAVE" {
        return Ok(AudioFormat::Wav);
    }

    // Check FLAC format
    if data.starts_with(b"fLaC") {
        return Ok(AudioFormat::Flac);
    }

    // Check MP3 format (ID3 tag or frame sync)
    if data.starts_with(b"ID3") {
        return Ok(AudioFormat::Mp3);
    }

    // Check for MP3 frame sync (11 bits set)
    if data.len() >= 2 && (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
        return Ok(AudioFormat::Mp3);
    }

    // Check OGG format
    if data.starts_with(b"OggS") {
        return Ok(AudioFormat::Ogg);
    }

    // Check M4A/MP4 format (ftyp box)
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        let brand = &data[8..12];
        if brand == b"M4A "
            || brand == b"mp41"
            || brand == b"mp42"
            || brand == b"isom"
            || brand == b"dash"
        {
            return Ok(AudioFormat::M4a);
        }
    }

    Ok(AudioFormat::Unknown)
}

/// Get all supported formats with their extensions
#[must_use]
pub fn get_supported_formats() -> Vec<(AudioFormat, &'static [&'static str])> {
    vec![
        (AudioFormat::Wav, AudioFormat::Wav.extensions()),
        (AudioFormat::Flac, AudioFormat::Flac.extensions()),
        (AudioFormat::Mp3, AudioFormat::Mp3.extensions()),
        (AudioFormat::Ogg, AudioFormat::Ogg.extensions()),
        (AudioFormat::M4a, AudioFormat::M4a.extensions()),
    ]
}

/// Check if format is supported for reading
#[must_use]
pub fn is_format_supported(format: AudioFormat) -> bool {
    match format {
        AudioFormat::Wav
        | AudioFormat::Flac
        | AudioFormat::Mp3
        | AudioFormat::Ogg
        | AudioFormat::M4a => true,
        AudioFormat::Unknown => false,
    }
}

/// Get format description string
#[must_use]
pub fn format_description(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "Waveform Audio File Format (WAV)",
        AudioFormat::Flac => "Free Lossless Audio Codec (FLAC)",
        AudioFormat::Mp3 => "MPEG-1 Audio Layer III (MP3)",
        AudioFormat::Ogg => "Ogg Vorbis",
        AudioFormat::M4a => "MPEG-4 Audio (M4A/AAC)",
        AudioFormat::Unknown => "Unknown format",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extension_detection() {
        assert_eq!(
            detect_format_from_extension("wav").unwrap(),
            AudioFormat::Wav
        );
        assert_eq!(
            detect_format_from_extension("WAV").unwrap(),
            AudioFormat::Wav
        );
        assert_eq!(
            detect_format_from_extension("flac").unwrap(),
            AudioFormat::Flac
        );
        assert_eq!(
            detect_format_from_extension("mp3").unwrap(),
            AudioFormat::Mp3
        );
        assert_eq!(
            detect_format_from_extension("ogg").unwrap(),
            AudioFormat::Ogg
        );
        assert_eq!(
            detect_format_from_extension("m4a").unwrap(),
            AudioFormat::M4a
        );
        assert_eq!(
            detect_format_from_extension("unknown").unwrap(),
            AudioFormat::Unknown
        );
    }

    #[test]
    fn test_path_detection() {
        let wav_path = PathBuf::from("test.wav");
        assert_eq!(
            detect_format_from_path(&wav_path).unwrap(),
            AudioFormat::Wav
        );

        let flac_path = PathBuf::from("/path/to/audio.flac");
        assert_eq!(
            detect_format_from_path(&flac_path).unwrap(),
            AudioFormat::Flac
        );
    }

    #[test]
    fn test_mime_detection() {
        assert_eq!(
            detect_format_from_mime("audio/wav").unwrap(),
            AudioFormat::Wav
        );
        assert_eq!(
            detect_format_from_mime("AUDIO/MPEG").unwrap(),
            AudioFormat::Mp3
        );
        assert_eq!(
            detect_format_from_mime("audio/flac").unwrap(),
            AudioFormat::Flac
        );
        assert_eq!(
            detect_format_from_mime("application/unknown").unwrap(),
            AudioFormat::Unknown
        );
    }

    #[test]
    fn test_bytes_detection() {
        // WAV header
        let wav_header = b"RIFF\x24\x08\x00\x00WAVEfmt ";
        assert_eq!(
            detect_format_from_bytes(wav_header).unwrap(),
            AudioFormat::Wav
        );

        // FLAC header
        let flac_header = b"fLaCAAAAAAAA";
        assert_eq!(
            detect_format_from_bytes(flac_header).unwrap(),
            AudioFormat::Flac
        );

        // MP3 ID3 header
        let mp3_id3_header = b"ID3\x03\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(
            detect_format_from_bytes(mp3_id3_header).unwrap(),
            AudioFormat::Mp3
        );

        // OGG header
        let ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        assert_eq!(
            detect_format_from_bytes(ogg_header).unwrap(),
            AudioFormat::Ogg
        );

        // Too short
        let short_data = b"AB";
        assert_eq!(
            detect_format_from_bytes(short_data).unwrap(),
            AudioFormat::Unknown
        );
    }

    #[test]
    fn test_supported_formats() {
        let formats = get_supported_formats();
        assert!(formats.len() >= 5); // At least WAV, FLAC, MP3, OGG, M4A

        assert!(is_format_supported(AudioFormat::Wav));
        assert!(is_format_supported(AudioFormat::Flac));
        assert!(is_format_supported(AudioFormat::Mp3));
        assert!(is_format_supported(AudioFormat::Ogg));
        assert!(is_format_supported(AudioFormat::M4a)); // Now implemented with Symphonia
        assert!(!is_format_supported(AudioFormat::Unknown));
    }

    #[test]
    fn test_format_descriptions() {
        assert!(format_description(AudioFormat::Wav).contains("WAV"));
        assert!(format_description(AudioFormat::Flac).contains("FLAC"));
        assert!(format_description(AudioFormat::Mp3).contains("MP3"));
        assert!(format_description(AudioFormat::Ogg).contains("Ogg"));
        assert!(format_description(AudioFormat::Unknown).contains("Unknown"));
    }
}
