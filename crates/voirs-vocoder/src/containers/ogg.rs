//! OGG container format implementation.

use crate::{AudioBuffer, Result, VocoderError};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::{CodecRegistry, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::ContainerConfig;

/// Write audio buffer to OGG container
pub fn write_ogg_container<P: AsRef<Path>>(
    audio: &AudioBuffer,
    path: P,
    config: &ContainerConfig,
) -> Result<()> {
    // Enhanced OGG container writing implementation
    // This creates an OGG container with Opus encoding for high-quality compression

    let path = path.as_ref();

    // Use Opus codec for OGG container (most modern and efficient)
    let codec_config =
        config
            .codec_config
            .as_ref()
            .cloned()
            .unwrap_or_else(|| crate::codecs::CodecConfig {
                sample_rate: audio.sample_rate(),
                channels: audio.channels() as u16, // Convert u32 to u16
                bit_rate: Some(128000),            // 128 kbps default for good quality
                quality: Some(0.8),
                compression_level: Some(10), // High quality Opus encoding
            });

    // Validate audio format for OGG/Opus
    if audio.channels() > 8 {
        return Err(VocoderError::ConfigError(
            "OGG/Opus does not support more than 8 channels".to_string(),
        ));
    }

    // For now, use Opus encoding directly and wrap in basic OGG structure
    // In a full implementation, this would use proper OGG page multiplexing
    let opus_data = crate::codecs::opus::encode_opus_bytes(audio, &codec_config)?;

    // Create OGG file with basic structure
    let file = File::create(path)
        .map_err(|e| VocoderError::InputError(format!("Failed to create OGG file: {e}")))?;

    let mut writer = BufWriter::new(file);

    // Write OGG page header (simplified structure)
    write_ogg_page_header(&mut writer, &opus_data, config)?;

    // Write Opus data
    writer
        .write_all(&opus_data)
        .map_err(|e| VocoderError::InputError(format!("Failed to write OGG data: {e}")))?;

    writer
        .flush()
        .map_err(|e| VocoderError::InputError(format!("Failed to flush OGG file: {e}")))?;

    Ok(())
}

/// Write OGG page header with metadata
fn write_ogg_page_header<W: Write>(
    writer: &mut W,
    opus_data: &[u8],
    config: &ContainerConfig,
) -> Result<()> {
    // OGG page structure (simplified for Opus)
    // This is a basic implementation - a full OGG muxer would be more complex

    // OGG page header signature
    writer
        .write_all(b"OggS")
        .map_err(|e| VocoderError::InputError(format!("Failed to write OGG signature: {e}")))?;

    // Version (0)
    writer
        .write_all(&[0])
        .map_err(|e| VocoderError::InputError(format!("Failed to write OGG version: {e}")))?;

    // Header type flags (0x02 = first page of stream)
    writer
        .write_all(&[0x02])
        .map_err(|e| VocoderError::InputError(format!("Failed to write OGG header flags: {e}")))?;

    // Granule position (0 for header page)
    writer
        .write_all(&[0; 8])
        .map_err(|e| VocoderError::InputError(format!("Failed to write granule position: {e}")))?;

    // Bitstream serial number (random)
    writer
        .write_all(&[0x12, 0x34, 0x56, 0x78])
        .map_err(|e| VocoderError::InputError(format!("Failed to write serial number: {e}")))?;

    // Page sequence number (0 for first page)
    writer
        .write_all(&[0; 4])
        .map_err(|e| VocoderError::InputError(format!("Failed to write page sequence: {e}")))?;

    // CRC checksum (simplified - would need proper calculation)
    writer
        .write_all(&[0; 4])
        .map_err(|e| VocoderError::InputError(format!("Failed to write CRC checksum: {e}")))?;

    // Number of page segments
    let segments = (opus_data.len() + 254) / 255; // Number of 255-byte segments needed
    writer
        .write_all(&[segments as u8])
        .map_err(|e| VocoderError::InputError(format!("Failed to write segment count: {e}")))?;

    // Segment table (lengths of each segment)
    for _ in 0..segments {
        let segment_size = std::cmp::min(255, opus_data.len());
        writer
            .write_all(&[segment_size as u8])
            .map_err(|e| VocoderError::InputError(format!("Failed to write segment size: {e}")))?;
    }

    // Write metadata if provided
    if let Some(metadata) = &config.metadata {
        write_vorbis_comment(writer, metadata)?;
    }

    Ok(())
}

/// Write Vorbis comment metadata
fn write_vorbis_comment<W: Write>(writer: &mut W, metadata: &super::AudioMetadata) -> Result<()> {
    // Vorbis comment format for metadata in OGG files
    let mut comment_data = Vec::new();

    // Vendor string
    let vendor = "VoiRS-Vocoder";
    comment_data.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    comment_data.extend_from_slice(vendor.as_bytes());

    // User comments
    let mut comments = Vec::new();

    if let Some(title) = &metadata.title {
        comments.push(format!("TITLE={title}"));
    }
    if let Some(artist) = &metadata.artist {
        comments.push(format!("ARTIST={artist}"));
    }
    if let Some(album) = &metadata.album {
        comments.push(format!("ALBUM={album}"));
    }
    if let Some(year) = metadata.year {
        comments.push(format!("DATE={year}"));
    }
    if let Some(genre) = &metadata.genre {
        comments.push(format!("GENRE={genre}"));
    }
    if let Some(track) = metadata.track {
        comments.push(format!("TRACKNUMBER={track}"));
    }

    // Write number of comments
    comment_data.extend_from_slice(&(comments.len() as u32).to_le_bytes());

    // Write each comment
    for comment in &comments {
        comment_data.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        comment_data.extend_from_slice(comment.as_bytes());
    }

    // Write the comment data
    writer
        .write_all(&comment_data)
        .map_err(|e| VocoderError::InputError(format!("Failed to write metadata: {e}")))?;

    Ok(())
}

/// Read audio from OGG container using Symphonia
pub fn read_ogg_container<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();

    // Open the file
    let file = File::open(path)
        .map_err(|e| VocoderError::InputError(format!("Failed to open OGG file: {e}")))?;

    let media_source = MediaSourceStream::new(Box::new(file), Default::default());

    // Create format hint for OGG
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(extension);
    }

    // Probe format
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let codec_registry = CodecRegistry::new();

    let probed = symphonia::default::get_probe()
        .format(&hint, media_source, &format_opts, &metadata_opts)
        .map_err(|e| VocoderError::InputError(format!("Failed to probe OGG format: {e}")))?;

    let mut format_reader = probed.format;

    // Get the default audio track
    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| VocoderError::InputError("No audio track found in OGG file".to_string()))?;

    let track_id = track.id;

    // Create decoder
    let decoder_opts = DecoderOptions::default();
    let mut decoder = codec_registry
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| VocoderError::InputError(format!("Failed to create decoder: {e}")))?;

    // Decode audio packets
    let mut samples = Vec::new();
    let mut sample_rate = 44100u32; // Default fallback
    let mut channels = 1u32; // Default fallback

    loop {
        let packet = match format_reader.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break, // End of stream
            Err(e) => {
                return Err(VocoderError::InputError(format!(
                    "Error reading packet: {e}"
                )))
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let audio_buf = decoder
            .decode(&packet)
            .map_err(|e| VocoderError::InputError(format!("Failed to decode packet: {e}")))?;

        // Extract audio information
        let spec = *audio_buf.spec();
        sample_rate = spec.rate;
        channels = spec.channels.count() as u32;

        // Convert to f32 samples
        match audio_buf {
            AudioBufferRef::F32(buf) => {
                // Interleave channels if needed
                if channels == 1 {
                    samples.extend_from_slice(buf.chan(0));
                } else {
                    // Interleave stereo channels
                    let chan0 = buf.chan(0);
                    let chan1 = buf.chan(1);
                    for i in 0..chan0.len() {
                        samples.push(chan0[i]);
                        if i < chan1.len() {
                            samples.push(chan1[i]);
                        }
                    }
                }
            }
            _ => {
                // Convert other formats to f32
                let mut sample_buf = SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(audio_buf);
                samples.extend_from_slice(sample_buf.samples());
            }
        }
    }

    if samples.is_empty() {
        return Err(VocoderError::InputError(
            "No audio data found in OGG file".to_string(),
        ));
    }

    Ok(AudioBuffer::new(samples, sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::{AudioMetadata, ContainerFormat};
    use std::fs;

    #[test]
    fn test_ogg_container_writing() {
        // Create longer audio data (1 second at 48kHz) for proper Opus encoding
        let duration_samples = 48000; // 1 second at 48kHz
        let audio_data: Vec<f32> = (0..duration_samples)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48000.0).sin() * 0.1)
            .collect();

        let audio = AudioBuffer::new(audio_data, 48000, 1);
        let config = crate::containers::ContainerConfig {
            format: ContainerFormat::Ogg,
            codec_config: None,
            metadata: None,
        };

        let test_path = "/tmp/test_audio_enhanced.ogg";
        let result = write_ogg_container(&audio, test_path, &config);

        // Should now succeed with enhanced implementation
        assert!(result.is_ok(), "OGG writing should succeed: {result:?}");

        // Verify file was created
        assert!(
            fs::metadata(test_path).is_ok(),
            "OGG file should be created"
        );

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_ogg_container_with_metadata() {
        // Create longer stereo audio data for proper Opus encoding
        let duration_samples = 24000; // 0.5 seconds at 48kHz per channel
        let audio_data: Vec<f32> = (0..duration_samples * 2) // Stereo = 2 channels
            .map(|i| {
                let channel = i % 2;
                let sample_idx = i / 2;
                let freq = if channel == 0 { 440.0 } else { 880.0 }; // Different freq per channel
                (sample_idx as f32 * freq * 2.0 * std::f32::consts::PI / 48000.0).sin() * 0.1
            })
            .collect();

        let audio = AudioBuffer::new(audio_data, 48000, 2);
        let metadata = AudioMetadata {
            title: Some("Test Track".to_string()),
            artist: Some("VoiRS".to_string()),
            album: Some("Test Album".to_string()),
            year: Some(2025),
            genre: Some("Electronic".to_string()),
            track: Some(1),
            duration: None,
        };

        let config = crate::containers::ContainerConfig {
            format: ContainerFormat::Ogg,
            codec_config: None,
            metadata: Some(metadata),
        };

        let test_path = "/tmp/test_audio_metadata.ogg";
        let result = write_ogg_container(&audio, test_path, &config);

        assert!(
            result.is_ok(),
            "OGG writing with metadata should succeed: {result:?}"
        );

        // Verify file was created
        assert!(
            fs::metadata(test_path).is_ok(),
            "OGG file with metadata should be created"
        );

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_ogg_invalid_channels() {
        // Test with too many channels
        let audio = AudioBuffer::new(vec![0.1; 100], 22050, 16); // 16 channels - too many for Opus
        let config = crate::containers::ContainerConfig {
            format: ContainerFormat::Ogg,
            codec_config: None,
            metadata: None,
        };

        let test_path = "/tmp/test_invalid.ogg";
        let result = write_ogg_container(&audio, test_path, &config);

        // Should fail with channel validation error
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not support more than 8 channels"));
    }

    #[test]
    fn test_ogg_read_nonexistent_file() {
        let result = read_ogg_container("/tmp/definitely_nonexistent_file.ogg");

        // Should fail with file not found error
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to open OGG file"));
    }

    #[test]
    fn test_vorbis_comment_generation() {
        let metadata = AudioMetadata {
            title: Some("Test".to_string()),
            artist: Some("Artist".to_string()),
            album: None,
            year: Some(2025),
            genre: None,
            track: Some(5),
            duration: None,
        };

        let mut buffer = Vec::new();
        let result = write_vorbis_comment(&mut buffer, &metadata);

        assert!(result.is_ok(), "Vorbis comment writing should succeed");
        assert!(
            !buffer.is_empty(),
            "Vorbis comment data should not be empty"
        );

        // Check that vendor string is present
        let buffer_str = String::from_utf8_lossy(&buffer);
        assert!(
            buffer_str.contains("VoiRS-Vocoder"),
            "Should contain vendor string"
        );
    }
}
