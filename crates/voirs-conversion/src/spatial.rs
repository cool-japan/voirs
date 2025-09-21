//! Spatial audio integration for voice conversion
//!
//! This module provides integration with the voirs-spatial crate to enable
//! 3D spatial audio processing during voice conversion.

#[cfg(feature = "spatial-integration")]
use voirs_spatial;

use crate::{Error, Result};

/// Spatial audio integration adapter for voice conversion
#[cfg(feature = "spatial-integration")]
#[derive(Debug, Clone)]
pub struct SpatialConversionAdapter {
    /// Spatial audio configuration
    config: Option<voirs_spatial::config::SpatialConfig>,
    /// Current listener position
    listener_position: SpatialPosition,
    /// Room acoustics model (placeholder)
    room_model: Option<String>,
}

#[cfg(feature = "spatial-integration")]
impl SpatialConversionAdapter {
    /// Create new spatial audio adapter
    pub fn new() -> Self {
        Self {
            config: None,
            listener_position: SpatialPosition::default(),
            room_model: None,
        }
    }

    /// Create adapter with spatial configuration
    pub fn with_config(config: voirs_spatial::config::SpatialConfig) -> Self {
        Self {
            config: Some(config),
            listener_position: SpatialPosition::default(),
            room_model: None,
        }
    }

    /// Set listener position in 3D space
    pub fn set_listener_position(&mut self, position: SpatialPosition) {
        self.listener_position = position;
    }

    /// Set room acoustics model
    pub fn set_room_model(&mut self, room_name: String) {
        self.room_model = Some(room_name);
    }

    /// Convert voice with spatial positioning
    pub async fn convert_with_spatial_position(
        &self,
        input_audio: &[f32],
        source_position: &SpatialPosition,
        target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<SpatialAudioOutput> {
        // Apply voice conversion first
        let converted_audio = self.apply_basic_conversion(input_audio, target_characteristics)?;

        // Apply spatial processing
        let spatial_audio = self.apply_spatial_processing(
            &converted_audio,
            source_position,
            &self.listener_position,
        )?;

        Ok(spatial_audio)
    }

    /// Convert voice for binaural output (stereo with HRTF)
    pub async fn convert_for_binaural_output(
        &self,
        input_audio: &[f32],
        source_position: &SpatialPosition,
        target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<BinauralAudioOutput> {
        let converted_audio = self.apply_basic_conversion(input_audio, target_characteristics)?;
        let binaural_audio = self.apply_hrtf_processing(&converted_audio, source_position)?;

        Ok(binaural_audio)
    }

    /// Convert voice for ambisonics output
    pub async fn convert_for_ambisonics(
        &self,
        input_audio: &[f32],
        source_position: &SpatialPosition,
        target_characteristics: &crate::types::VoiceCharacteristics,
        order: u8,
    ) -> Result<AmbisonicsOutput> {
        let converted_audio = self.apply_basic_conversion(input_audio, target_characteristics)?;
        let ambisonics_audio =
            self.encode_to_ambisonics(&converted_audio, source_position, order)?;

        Ok(ambisonics_audio)
    }

    /// Convert multiple voices with spatial positioning
    pub async fn convert_multiple_sources(
        &self,
        sources: &[SpatialVoiceSource],
    ) -> Result<SpatialAudioOutput> {
        let mut mixed_output = SpatialAudioOutput::new(2, 0); // Start with stereo

        for source in sources {
            let converted = self
                .convert_with_spatial_position(
                    &source.audio,
                    &source.position,
                    &source.target_characteristics,
                )
                .await?;

            mixed_output = self.mix_spatial_sources(&mixed_output, &converted)?;
        }

        Ok(mixed_output)
    }

    // Private helper methods
    fn apply_basic_conversion(
        &self,
        input_audio: &[f32],
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        // Placeholder for basic voice conversion
        Ok(input_audio.to_vec())
    }

    fn apply_spatial_processing(
        &self,
        audio: &[f32],
        source_position: &SpatialPosition,
        listener_position: &SpatialPosition,
    ) -> Result<SpatialAudioOutput> {
        // Calculate distance and direction
        let distance = self.calculate_distance(source_position, listener_position);
        let direction = self.calculate_direction(source_position, listener_position);

        // Apply distance attenuation
        let attenuated_audio = self.apply_distance_attenuation(audio, distance);

        // Apply directional filtering
        let directional_audio = self.apply_directional_processing(&attenuated_audio, &direction)?;

        // Apply room reverb if room model is available
        let final_audio = if let Some(ref room_name) = self.room_model {
            self.apply_room_reverb(&directional_audio, room_name)?
        } else {
            directional_audio
        };

        Ok(SpatialAudioOutput {
            channels: vec![final_audio.clone(), final_audio], // Simple stereo for now
            sample_rate: 44100,
            position_metadata: Some(source_position.clone()),
        })
    }

    fn apply_hrtf_processing(
        &self,
        audio: &[f32],
        source_position: &SpatialPosition,
    ) -> Result<BinauralAudioOutput> {
        // Placeholder HRTF processing
        let left_channel = audio.to_vec();
        let right_channel = audio.iter().map(|&x| x * 0.8).collect(); // Simple difference

        Ok(BinauralAudioOutput {
            left_channel,
            right_channel,
            sample_rate: 44100,
            hrtf_metadata: Some(HrtfMetadata {
                azimuth: source_position.azimuth,
                elevation: source_position.elevation,
                distance: source_position.distance,
            }),
        })
    }

    fn encode_to_ambisonics(
        &self,
        audio: &[f32],
        _source_position: &SpatialPosition,
        order: u8,
    ) -> Result<AmbisonicsOutput> {
        let num_channels = ((order + 1) * (order + 1)) as usize;
        let mut channels = Vec::with_capacity(num_channels);

        // Placeholder ambisonics encoding
        for i in 0..num_channels {
            let gain = 1.0 / (i + 1) as f32;
            let channel = audio.iter().map(|&x| x * gain).collect();
            channels.push(channel);
        }

        Ok(AmbisonicsOutput {
            channels,
            order,
            sample_rate: 44100,
        })
    }

    fn calculate_distance(&self, pos1: &SpatialPosition, pos2: &SpatialPosition) -> f32 {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn calculate_direction(
        &self,
        source: &SpatialPosition,
        listener: &SpatialPosition,
    ) -> SpatialDirection {
        let dx = source.x - listener.x;
        let dy = source.y - listener.y;
        let dz = source.z - listener.z;

        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        SpatialDirection {
            azimuth: dy.atan2(dx),
            elevation: (dz / distance).asin(),
            distance,
        }
    }

    fn apply_distance_attenuation(&self, audio: &[f32], distance: f32) -> Vec<f32> {
        let attenuation = 1.0 / (1.0 + distance * 0.1);
        audio.iter().map(|&x| x * attenuation).collect()
    }

    fn apply_directional_processing(
        &self,
        audio: &[f32],
        _direction: &SpatialDirection,
    ) -> Result<Vec<f32>> {
        // Placeholder directional processing
        Ok(audio.to_vec())
    }

    fn apply_room_reverb(&self, audio: &[f32], _room_name: &str) -> Result<Vec<f32>> {
        // Placeholder room reverb
        Ok(audio.to_vec())
    }

    fn mix_spatial_sources(
        &self,
        existing: &SpatialAudioOutput,
        new: &SpatialAudioOutput,
    ) -> Result<SpatialAudioOutput> {
        // Simple mixing - in reality would be more sophisticated
        let mut mixed_channels = Vec::new();

        for (i, existing_channel) in existing.channels.iter().enumerate() {
            if let Some(new_channel) = new.channels.get(i) {
                let mixed: Vec<f32> = existing_channel
                    .iter()
                    .zip(new_channel.iter())
                    .map(|(&a, &b)| (a + b) * 0.5)
                    .collect();
                mixed_channels.push(mixed);
            } else {
                mixed_channels.push(existing_channel.clone());
            }
        }

        Ok(SpatialAudioOutput {
            channels: mixed_channels,
            sample_rate: existing.sample_rate,
            position_metadata: None, // Mixed sources don't have single position
        })
    }
}

#[cfg(feature = "spatial-integration")]
impl Default for SpatialConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// 3D position in space
#[derive(Debug, Clone, Default)]
pub struct SpatialPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
}

/// Spatial direction information
#[derive(Debug, Clone)]
pub struct SpatialDirection {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
}

/// Voice source with spatial information
#[derive(Debug, Clone)]
pub struct SpatialVoiceSource {
    pub audio: Vec<f32>,
    pub position: SpatialPosition,
    pub target_characteristics: crate::types::VoiceCharacteristics,
}

/// Spatial audio output with multiple channels
#[derive(Debug, Clone)]
pub struct SpatialAudioOutput {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub position_metadata: Option<SpatialPosition>,
}

impl SpatialAudioOutput {
    pub fn new(num_channels: usize, length: usize) -> Self {
        Self {
            channels: vec![vec![0.0; length]; num_channels],
            sample_rate: 44100,
            position_metadata: None,
        }
    }
}

/// Binaural audio output (stereo with HRTF processing)
#[derive(Debug, Clone)]
pub struct BinauralAudioOutput {
    pub left_channel: Vec<f32>,
    pub right_channel: Vec<f32>,
    pub sample_rate: u32,
    pub hrtf_metadata: Option<HrtfMetadata>,
}

/// HRTF metadata for binaural processing
#[derive(Debug, Clone)]
pub struct HrtfMetadata {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
}

/// Ambisonics output for 360-degree audio
#[derive(Debug, Clone)]
pub struct AmbisonicsOutput {
    pub channels: Vec<Vec<f32>>,
    pub order: u8,
    pub sample_rate: u32,
}

// Stub implementation when spatial integration is disabled
#[cfg(not(feature = "spatial-integration"))]
#[derive(Debug, Clone)]
pub struct SpatialConversionAdapter;

#[cfg(not(feature = "spatial-integration"))]
impl SpatialConversionAdapter {
    pub fn new() -> Self {
        Self
    }

    pub async fn convert_with_spatial_position(
        &self,
        _input_audio: &[f32],
        _source_position: &SpatialPosition,
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<SpatialAudioOutput> {
        Err(Error::config(
            "Spatial integration not enabled. Enable with 'spatial-integration' feature."
                .to_string(),
        ))
    }

    pub async fn convert_for_binaural_output(
        &self,
        _input_audio: &[f32],
        _source_position: &SpatialPosition,
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<BinauralAudioOutput> {
        Err(Error::config(
            "Spatial integration not enabled. Enable with 'spatial-integration' feature."
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "spatial-integration"))]
impl Default for SpatialConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_position_default() {
        let pos = SpatialPosition::default();
        assert_eq!(pos.x, 0.0);
        assert_eq!(pos.y, 0.0);
        assert_eq!(pos.z, 0.0);
    }

    #[test]
    fn test_spatial_adapter_creation() {
        let adapter = SpatialConversionAdapter::new();
        assert!(matches!(adapter, SpatialConversionAdapter { .. }));
    }

    #[cfg(not(feature = "spatial-integration"))]
    #[tokio::test]
    async fn test_spatial_integration_disabled() {
        let adapter = SpatialConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let position = SpatialPosition::default();
        let characteristics = crate::types::VoiceCharacteristics::new();

        let result = adapter
            .convert_with_spatial_position(&audio, &position, &characteristics)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }

    #[test]
    fn test_spatial_audio_output_creation() {
        let output = SpatialAudioOutput::new(2, 1000);
        assert_eq!(output.channels.len(), 2);
        assert_eq!(output.channels[0].len(), 1000);
        assert_eq!(output.sample_rate, 44100);
    }
}
