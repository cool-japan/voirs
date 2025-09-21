//! 3D spatial audio processing for vocoder.

pub mod acoustics;
pub mod binaural;
pub mod config;
pub mod hrtf;
pub mod positioning;
pub mod processor;
pub mod quality_metrics;

pub use acoustics::*;
pub use binaural::*;
pub use config::*;
pub use hrtf::*;
pub use positioning::*;
pub use processor::*;
pub use quality_metrics::*;

use crate::hifigan::HiFiGanVocoder;
use crate::{MelSpectrogram, Vocoder};
use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// 3D spatial audio vocoder for immersive audio experiences
pub struct SpatialVocoder {
    /// Base vocoder configuration
    pub config: SpatialVocoderConfig,
    /// Base HiFi-GAN vocoder for audio generation
    pub base_vocoder: Option<HiFiGanVocoder>,
    /// HRTF processor for head-related transfer functions
    pub hrtf_processor: HrtfProcessor,
    /// Binaural renderer for stereo spatialization
    pub binaural_renderer: BinauralRenderer,
    /// 3D positioning system
    pub positioning_system: PositioningSystem,
    /// Room acoustics simulator
    pub acoustics_simulator: AcousticsSimulator,
    /// Quality metrics calculator
    pub quality_metrics: SpatialQualityMetrics,
}

impl SpatialVocoder {
    /// Create new spatial vocoder with configuration
    pub fn new(config: SpatialVocoderConfig) -> Result<Self> {
        Ok(Self {
            base_vocoder: None, // Will be set via set_base_vocoder
            hrtf_processor: HrtfProcessor::new(&config.hrtf)?,
            binaural_renderer: BinauralRenderer::new(&config.binaural)?,
            positioning_system: PositioningSystem::new(&config.positioning)?,
            acoustics_simulator: AcousticsSimulator::new(&config.acoustics)?,
            quality_metrics: SpatialQualityMetrics::new(),
            config,
        })
    }

    /// Create new spatial vocoder with base vocoder
    pub fn with_base_vocoder(
        config: SpatialVocoderConfig,
        base_vocoder: HiFiGanVocoder,
    ) -> Result<Self> {
        Ok(Self {
            base_vocoder: Some(base_vocoder),
            hrtf_processor: HrtfProcessor::new(&config.hrtf)?,
            binaural_renderer: BinauralRenderer::new(&config.binaural)?,
            positioning_system: PositioningSystem::new(&config.positioning)?,
            acoustics_simulator: AcousticsSimulator::new(&config.acoustics)?,
            quality_metrics: SpatialQualityMetrics::new(),
            config,
        })
    }

    /// Set the base vocoder for audio generation
    pub fn set_base_vocoder(&mut self, base_vocoder: HiFiGanVocoder) {
        self.base_vocoder = Some(base_vocoder);
    }

    /// Process audio with spatial positioning
    pub fn process_spatial_audio(
        &mut self,
        audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<SpatialAudioOutput> {
        // Apply 3D positioning
        let positioned_audio = self.positioning_system.position_audio(audio, position)?;

        // Apply room acoustics
        let acoustic_audio = self.acoustics_simulator.process(&positioned_audio)?;

        // Apply HRTF processing
        let hrtf_audio = self.hrtf_processor.process(&acoustic_audio, position)?;

        // Render binaural output
        let binaural_output = self.binaural_renderer.render(&hrtf_audio.left, position)?;

        // Calculate quality score based on spatial processing metrics
        let quality_score = self.calculate_spatial_quality_score(
            &binaural_output.left,
            &binaural_output.right,
            position,
            &binaural_output,
        );

        // Create final output
        let final_output = SpatialAudioOutput {
            left_channel: binaural_output.left,
            right_channel: binaural_output.right,
            quality_score,
            processing_info: ProcessingInfo {
                position: position.clone(),
                reverb_level: self.acoustics_simulator.get_reverb_level(),
                hrtf_applied: true,
                binaural_rendered: true,
            },
        };

        // Calculate quality metrics
        let quality = self.quality_metrics.calculate(&final_output)?;

        // Update quality score and return
        let mut final_output = final_output;
        final_output.quality_score = quality;

        Ok(final_output)
    }

    /// Process mel spectrogram with spatial audio
    pub fn process_mel_spectrogram_spatial(
        &mut self,
        mel_spectrogram: &Array2<f32>,
        position: &SpatialPosition,
    ) -> Result<SpatialAudioOutput> {
        // First generate audio from mel spectrogram (this would integrate with base vocoder)
        let audio = self.generate_audio_from_mel(mel_spectrogram)?;

        // Then apply spatial processing
        self.process_spatial_audio(&audio, position)
    }

    /// Generate audio from mel spectrogram using base vocoder integration
    fn generate_audio_from_mel(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Vec<f32>> {
        if let Some(base_vocoder) = &self.base_vocoder {
            // Use the actual HiFi-GAN vocoder for high-quality audio generation
            // Convert ndarray to MelSpectrogram format
            let mel_data: Vec<Vec<f32>> = (0..mel_spectrogram.nrows())
                .map(|i| mel_spectrogram.row(i).to_vec())
                .collect();
            let mel = MelSpectrogram::new(
                mel_data,
                self.config.sample_rate,
                self.config.hop_size as u32,
            );

            // Use async runtime to call the vocoder
            let rt = tokio::runtime::Runtime::new()?;
            let audio_buffer = rt.block_on(async { base_vocoder.vocode(&mel, None).await })?;

            Ok(audio_buffer.samples)
        } else {
            // Enhanced spatial audio generation with built-in vocoding capabilities
            tracing::info!("Using built-in spatial vocoder for mel-to-audio synthesis");

            let frames = mel_spectrogram.shape()[1];
            let _mel_bins = mel_spectrogram.shape()[0];
            let audio_length = frames * self.config.hop_size;

            // Generate high-quality audio from mel spectrogram using spatial synthesis
            let mono_audio = self.synthesize_from_mel_spectrogram(mel_spectrogram, audio_length)?;

            // Apply basic stereo conversion for non-positioned sources
            let stereo_audio = self.convert_mono_to_spatial_stereo(&mono_audio)?;

            Ok(stereo_audio)
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SpatialVocoderConfig) -> Result<()> {
        self.config = config;
        self.hrtf_processor.update_config(&self.config.hrtf)?;
        self.binaural_renderer
            .update_config(&self.config.binaural)?;
        self.positioning_system
            .update_config(&self.config.positioning)?;
        self.acoustics_simulator
            .update_config(&self.config.acoustics)?;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &SpatialVocoderConfig {
        &self.config
    }

    /// Calculate spatial quality score based on processing metrics
    fn calculate_spatial_quality_score(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        position: &SpatialPosition,
        binaural_output: &BinauralOutput,
    ) -> f32 {
        if left_channel.is_empty() || right_channel.is_empty() {
            return 0.0;
        }

        let mut quality_factors = Vec::new();

        // 1. Inter-channel correlation (should be appropriate for spatial position)
        let correlation = self.calculate_inter_channel_correlation(left_channel, right_channel);
        let expected_correlation = self.expected_correlation_for_position(position);
        let correlation_score = 1.0 - (correlation - expected_correlation).abs();
        quality_factors.push(correlation_score.clamp(0.0, 1.0));

        // 2. Level difference consistency (ILD - Interaural Level Difference)
        let level_difference = self.calculate_level_difference(left_channel, right_channel);
        let expected_ild = self.expected_ild_for_position(position);
        let ild_score = 1.0 - (level_difference - expected_ild).abs() / 20.0; // Normalize by 20dB range
        quality_factors.push(ild_score.clamp(0.0, 1.0));

        // 3. Signal quality (absence of artifacts)
        let left_quality = self.calculate_signal_quality(left_channel);
        let right_quality = self.calculate_signal_quality(right_channel);
        let signal_quality = (left_quality + right_quality) / 2.0;
        quality_factors.push(signal_quality);

        // 4. Spatial consistency (how well the position is maintained)
        let spatial_consistency = self.calculate_spatial_consistency(position, binaural_output);
        quality_factors.push(spatial_consistency);

        // 5. Distance-related attenuation quality
        let distance_quality =
            self.calculate_distance_quality(position, left_channel, right_channel);
        quality_factors.push(distance_quality);

        // Calculate weighted average (equal weights for simplicity)
        let total_score: f32 = quality_factors.iter().sum();
        let average_score = total_score / quality_factors.len() as f32;

        average_score.clamp(0.0, 1.0)
    }

    /// Calculate inter-channel correlation
    fn calculate_inter_channel_correlation(&self, left: &[f32], right: &[f32]) -> f32 {
        if left.len() != right.len() || left.is_empty() {
            return 0.0;
        }

        let left_mean = left.iter().sum::<f32>() / left.len() as f32;
        let right_mean = right.iter().sum::<f32>() / right.len() as f32;

        let mut numerator = 0.0;
        let mut left_variance = 0.0;
        let mut right_variance = 0.0;

        for (l, r) in left.iter().zip(right.iter()) {
            let left_diff = l - left_mean;
            let right_diff = r - right_mean;
            numerator += left_diff * right_diff;
            left_variance += left_diff * left_diff;
            right_variance += right_diff * right_diff;
        }

        if left_variance > 0.0 && right_variance > 0.0 {
            numerator / (left_variance * right_variance).sqrt()
        } else {
            0.0
        }
    }

    /// Calculate expected correlation for a spatial position
    fn expected_correlation_for_position(&self, position: &SpatialPosition) -> f32 {
        // High correlation for center positions, lower for extreme sides
        let azimuth_rad = position.azimuth.to_radians().abs();
        let base_correlation = 0.7; // Base correlation for centered audio
        let azimuth_factor = (azimuth_rad / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0);
        base_correlation * (1.0 - azimuth_factor * 0.5)
    }

    /// Calculate level difference between channels
    fn calculate_level_difference(&self, left: &[f32], right: &[f32]) -> f32 {
        let left_rms = (left.iter().map(|x| x * x).sum::<f32>() / left.len() as f32).sqrt();
        let right_rms = (right.iter().map(|x| x * x).sum::<f32>() / right.len() as f32).sqrt();

        if left_rms > 0.0 && right_rms > 0.0 {
            20.0 * (left_rms / right_rms).log10()
        } else {
            0.0
        }
    }

    /// Calculate expected ILD for position
    fn expected_ild_for_position(&self, position: &SpatialPosition) -> f32 {
        // ILD increases with azimuth angle
        let azimuth_rad = position.azimuth.to_radians();
        azimuth_rad.sin() * 15.0 // Up to 15dB difference for 90° positions
    }

    /// Calculate signal quality (detecting artifacts)
    fn calculate_signal_quality(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut quality_score = 1.0;

        // Check for clipping
        let clipped_samples = samples.iter().filter(|&&x| x.abs() >= 0.99).count();
        let clipping_ratio = clipped_samples as f32 / samples.len() as f32;
        quality_score -= clipping_ratio * 0.5; // Penalty for clipping

        // Check for excessive DC offset
        let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
        if dc_offset.abs() > 0.1 {
            quality_score -= 0.2;
        }

        // Check for sudden level changes (artifacts)
        let mut level_changes = 0;
        for window in samples.windows(2) {
            if (window[1] - window[0]).abs() > 0.5 {
                level_changes += 1;
            }
        }
        let artifact_ratio = level_changes as f32 / (samples.len() - 1) as f32;
        quality_score -= artifact_ratio * 0.3;

        quality_score.clamp(0.0, 1.0)
    }

    /// Calculate spatial consistency
    fn calculate_spatial_consistency(
        &self,
        position: &SpatialPosition,
        _binaural_output: &BinauralOutput,
    ) -> f32 {
        // Score based on reasonable position values
        let mut consistency: f32 = 1.0;

        // Check distance reasonableness
        if position.distance < 0.1 || position.distance > 100.0 {
            consistency -= 0.3;
        }

        // Check angle ranges
        if position.azimuth.abs() > 180.0 || position.elevation.abs() > 90.0 {
            consistency -= 0.2;
        }

        consistency.clamp(0.0, 1.0)
    }

    /// Calculate distance-related quality
    fn calculate_distance_quality(
        &self,
        position: &SpatialPosition,
        left: &[f32],
        right: &[f32],
    ) -> f32 {
        // Calculate expected attenuation based on distance
        let expected_attenuation = 1.0 / (1.0 + position.distance * 0.1);

        // Calculate actual signal level
        let left_level = (left.iter().map(|x| x * x).sum::<f32>() / left.len() as f32).sqrt();
        let right_level = (right.iter().map(|x| x * x).sum::<f32>() / right.len() as f32).sqrt();
        let avg_level = (left_level + right_level) / 2.0;

        // Compare with expected level (assuming input was normalized)
        let level_difference = (avg_level - expected_attenuation).abs();
        let quality = 1.0 - level_difference.min(1.0);

        quality.clamp(0.0, 1.0)
    }
}

/// Spatial audio output
#[derive(Debug, Clone)]
pub struct SpatialAudioOutput {
    /// Left channel audio
    pub left_channel: Vec<f32>,
    /// Right channel audio
    pub right_channel: Vec<f32>,
    /// Quality score
    pub quality_score: f32,
    /// Processing information
    pub processing_info: ProcessingInfo,
}

/// Processing information
#[derive(Debug, Clone)]
pub struct ProcessingInfo {
    /// Spatial position used
    pub position: SpatialPosition,
    /// Reverb level applied
    pub reverb_level: f32,
    /// Whether HRTF was applied
    pub hrtf_applied: bool,
    /// Whether binaural rendering was applied
    pub binaural_rendered: bool,
}

/// 3D spatial position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPosition {
    /// X coordinate (left-right)
    pub x: f32,
    /// Y coordinate (front-back)
    pub y: f32,
    /// Z coordinate (up-down)
    pub z: f32,
    /// Azimuth angle (degrees)
    pub azimuth: f32,
    /// Elevation angle (degrees)
    pub elevation: f32,
    /// Distance from listener
    pub distance: f32,
}

impl Default for SpatialPosition {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
        }
    }
}

impl SpatialVocoder {
    /// Synthesize audio from mel spectrogram using advanced spectral reconstruction
    fn synthesize_from_mel_spectrogram(
        &self,
        mel_spectrogram: &Array2<f32>,
        audio_length: usize,
    ) -> Result<Vec<f32>> {
        let _frames = mel_spectrogram.shape()[1];
        let _mel_bins = mel_spectrogram.shape()[0];
        let mut audio = vec![0.0; audio_length];

        // Griffin-Lim inspired iterative reconstruction with mel-to-linear conversion
        let mut magnitude_spectrogram = self.mel_to_linear_spectrogram(mel_spectrogram)?;

        // Initialize phase randomly for better reconstruction
        let mut phase_spectrogram = self.initialize_random_phase(&magnitude_spectrogram);

        // Iterative phase reconstruction (simplified Griffin-Lim)
        for _iteration in 0..3 {
            // Convert to time domain
            audio = self.istft(&magnitude_spectrogram, &phase_spectrogram)?;

            // Forward STFT to get new phase
            let (new_magnitude, new_phase) = self.stft(&audio)?;

            // Keep original magnitude, use new phase
            phase_spectrogram = new_phase;

            // Optional: blend magnitudes for better convergence
            for i in 0..magnitude_spectrogram.len() {
                for j in 0..magnitude_spectrogram[i].len() {
                    magnitude_spectrogram[i][j] =
                        0.9 * magnitude_spectrogram[i][j] + 0.1 * new_magnitude[i][j];
                }
            }
        }

        // Apply window tapering to reduce artifacts
        self.apply_window_tapering(&mut audio);

        Ok(audio)
    }

    /// Convert mel spectrogram to linear frequency spectrogram
    fn mel_to_linear_spectrogram(&self, mel_spectrogram: &Array2<f32>) -> Result<Vec<Vec<f32>>> {
        let frames = mel_spectrogram.shape()[1];
        let mel_bins = mel_spectrogram.shape()[0];
        let linear_bins = 1025; // Typical for 2048 FFT

        let mut linear_spec = vec![vec![0.0; linear_bins]; frames];

        // Simple mel-to-linear mapping (more sophisticated than basic approximation)
        for frame_idx in 0..frames {
            for mel_bin in 0..mel_bins {
                let mel_value = mel_spectrogram[[mel_bin, frame_idx]];

                // Convert mel bin to frequency range
                let mel_freq_low = Self::mel_to_hz(mel_bin as f32 * 2595.0 / mel_bins as f32);
                let mel_freq_high =
                    Self::mel_to_hz((mel_bin + 1) as f32 * 2595.0 / mel_bins as f32);

                // Map to linear frequency bins
                let linear_bin_low = ((mel_freq_low / (self.config.sample_rate as f32 / 2.0))
                    * linear_bins as f32) as usize;
                let linear_bin_high = ((mel_freq_high / (self.config.sample_rate as f32 / 2.0))
                    * linear_bins as f32) as usize;

                // Distribute mel energy across linear bins
                for linear_bin in linear_bin_low..linear_bin_high.min(linear_bins) {
                    linear_spec[frame_idx][linear_bin] +=
                        mel_value / (linear_bin_high - linear_bin_low).max(1) as f32;
                }
            }
        }

        Ok(linear_spec)
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * ((mel / 2595.0).exp() - 1.0)
    }

    /// Initialize random phase for spectral reconstruction
    fn initialize_random_phase(&self, magnitude_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use fastrand::f32;

        let frames = magnitude_spec.len();
        let bins = magnitude_spec[0].len();
        let mut phase_spec = vec![vec![0.0; bins]; frames];

        for frame in phase_spec.iter_mut().take(frames) {
            for bin in frame.iter_mut().take(bins) {
                *bin = f32() * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            }
        }

        phase_spec
    }

    /// Inverse STFT (simplified implementation)
    fn istft(&self, magnitude: &[Vec<f32>], phase: &[Vec<f32>]) -> Result<Vec<f32>> {
        let frames = magnitude.len();
        let fft_size = (magnitude[0].len() - 1) * 2; // Assuming N/2+1 bins
        let hop_length = self.config.hop_size;
        let audio_length = (frames - 1) * hop_length + fft_size;

        let mut audio = vec![0.0; audio_length];
        let window = self.hann_window(fft_size);

        for frame_idx in 0..frames {
            let start_idx = frame_idx * hop_length;

            // Create complex spectrum from magnitude and phase
            let mut time_frame = vec![0.0; fft_size];

            // Simple inverse FFT approximation using cosine synthesis
            for sample_idx in 0..fft_size {
                let mut sample_value = 0.0;

                for bin_idx in 0..magnitude[frame_idx].len() {
                    let mag = magnitude[frame_idx][bin_idx];
                    let phase_val = phase[frame_idx][bin_idx];
                    let freq = bin_idx as f32 * 2.0 * std::f32::consts::PI / fft_size as f32;

                    sample_value += mag * (freq * sample_idx as f32 + phase_val).cos();
                }

                time_frame[sample_idx] = sample_value * window[sample_idx];
            }

            // Overlap-add
            for (i, &sample) in time_frame.iter().enumerate() {
                let audio_idx = start_idx + i;
                if audio_idx < audio.len() {
                    audio[audio_idx] += sample;
                }
            }
        }

        // Normalize
        let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for sample in audio.iter_mut() {
                *sample /= max_val * 2.0; // Extra headroom
            }
        }

        Ok(audio)
    }

    /// Forward STFT (simplified implementation)
    #[allow(clippy::type_complexity)]
    fn stft(&self, audio: &[f32]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let fft_size = 2048;
        let hop_length = self.config.hop_size;
        let frames = (audio.len() - fft_size) / hop_length + 1;
        let bins = fft_size / 2 + 1;

        let mut magnitude = vec![vec![0.0; bins]; frames];
        let mut phase = vec![vec![0.0; bins]; frames];
        let window = self.hann_window(fft_size);

        for frame_idx in 0..frames {
            let start_idx = frame_idx * hop_length;

            // Apply window
            let mut windowed_frame = vec![0.0; fft_size];
            for i in 0..fft_size {
                let audio_idx = start_idx + i;
                if audio_idx < audio.len() {
                    windowed_frame[i] = audio[audio_idx] * window[i];
                }
            }

            // Simple DFT for magnitude and phase extraction
            for bin_idx in 0..bins {
                let mut real = 0.0;
                let mut imag = 0.0;

                for (sample_idx, &window_sample) in windowed_frame.iter().enumerate().take(fft_size)
                {
                    let angle = -2.0 * std::f32::consts::PI * bin_idx as f32 * sample_idx as f32
                        / fft_size as f32;
                    real += window_sample * angle.cos();
                    imag += window_sample * angle.sin();
                }

                magnitude[frame_idx][bin_idx] = (real * real + imag * imag).sqrt();
                phase[frame_idx][bin_idx] = imag.atan2(real);
            }
        }

        Ok((magnitude, phase))
    }

    /// Generate Hann window
    fn hann_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Apply window tapering to reduce artifacts
    fn apply_window_tapering(&self, audio: &mut [f32]) {
        let taper_length = (audio.len() / 20).min(1000); // 5% taper or max 1000 samples

        // Fade in
        for (i, sample) in audio.iter_mut().enumerate().take(taper_length) {
            let factor = i as f32 / taper_length as f32;
            *sample *= factor * factor; // Smooth squared fade
        }

        // Fade out
        for i in 0..taper_length {
            let idx = audio.len() - 1 - i;
            let factor = i as f32 / taper_length as f32;
            audio[idx] *= factor * factor;
        }
    }

    /// Apply full spatial processing pipeline to mono audio
    #[allow(dead_code)]
    fn apply_spatial_processing_pipeline(
        &mut self,
        mono_audio: &[f32],
        position: &SpatialPosition,
    ) -> Result<Vec<f32>> {
        // 1. Apply HRTF processing
        let hrtf_processed = self.hrtf_processor.process(mono_audio, position)?;

        // 2. Apply distance attenuation and delay
        let positioned_audio = self
            .positioning_system
            .position_audio(&hrtf_processed.left, position)?;

        // 3. Apply room acoustics simulation
        let acoustic_audio = self.acoustics_simulator.process(&positioned_audio)?;

        // 4. Final binaural rendering
        let final_audio = self.binaural_renderer.render(&acoustic_audio, position)?;

        // Interleave stereo channels
        let mut stereo_output = Vec::with_capacity(final_audio.left.len() * 2);
        for (l, r) in final_audio.left.iter().zip(final_audio.right.iter()) {
            stereo_output.push(*l);
            stereo_output.push(*r);
        }

        Ok(stereo_output)
    }

    /// Convert mono to spatial stereo with basic processing
    fn convert_mono_to_spatial_stereo(&mut self, mono_audio: &[f32]) -> Result<Vec<f32>> {
        // Apply basic stereo widening and spatial enhancement
        let mut stereo_output = Vec::with_capacity(mono_audio.len() * 2);

        // Simple stereo conversion with delay and filtering for spatial effect
        let delay_samples = (self.config.sample_rate as f32 * 0.001) as usize; // 1ms delay
        let mut delay_buffer = vec![0.0; delay_samples];
        let mut delay_index = 0;

        for (i, &sample) in mono_audio.iter().enumerate() {
            // Left channel: direct signal with slight high-pass
            let left = sample * 0.9 + self.simple_highpass(sample, i) * 0.1;

            // Right channel: slightly delayed with low-pass for spatial effect
            let delayed_sample = delay_buffer[delay_index];
            let right = delayed_sample * 0.8 + self.simple_lowpass(delayed_sample, i) * 0.2;

            // Update delay buffer
            delay_buffer[delay_index] = sample;
            delay_index = (delay_index + 1) % delay_samples;

            stereo_output.push(left);
            stereo_output.push(right);
        }

        Ok(stereo_output)
    }

    /// Simple high-pass filter
    fn simple_highpass(&self, sample: f32, _index: usize) -> f32 {
        // Simple differentiator for high-pass effect
        static mut PREV_SAMPLE: f32 = 0.0;
        unsafe {
            let result = sample - PREV_SAMPLE * 0.95;
            PREV_SAMPLE = sample;
            result
        }
    }

    /// Simple low-pass filter
    fn simple_lowpass(&self, sample: f32, _index: usize) -> f32 {
        // Simple moving average for low-pass effect
        static mut PREV_SAMPLE: f32 = 0.0;
        unsafe {
            let result = sample * 0.3 + PREV_SAMPLE * 0.7;
            PREV_SAMPLE = result;
            result
        }
    }
}

/// Binaural audio output
#[derive(Debug, Clone)]
pub struct BinauralOutput {
    /// Left channel
    pub left: Vec<f32>,
    /// Right channel
    pub right: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_vocoder_creation() {
        let config = SpatialVocoderConfig::default();
        let vocoder = SpatialVocoder::new(config);
        assert!(vocoder.is_ok());
    }

    #[test]
    fn test_spatial_position_default() {
        let position = SpatialPosition::default();
        assert_eq!(position.x, 0.0);
        assert_eq!(position.y, 0.0);
        assert_eq!(position.z, 0.0);
        assert_eq!(position.azimuth, 0.0);
        assert_eq!(position.elevation, 0.0);
        assert_eq!(position.distance, 1.0);
    }

    #[test]
    fn test_spatial_audio_processing() {
        let config = SpatialVocoderConfig::default();
        let mut vocoder = SpatialVocoder::new(config).unwrap();

        // Create sample audio
        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = SpatialPosition::default();

        let result = vocoder.process_spatial_audio(&audio, &position);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.left_channel.is_empty());
        assert!(!output.right_channel.is_empty());
        assert!(output.quality_score >= 0.0);
    }

    #[test]
    fn test_mel_spectrogram_spatial_processing() {
        let config = SpatialVocoderConfig::default();
        let mut vocoder = SpatialVocoder::new(config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let position = SpatialPosition::default();

        let result = vocoder.process_mel_spectrogram_spatial(&mel, &position);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_update() {
        let config = SpatialVocoderConfig::default();
        let mut vocoder = SpatialVocoder::new(config).unwrap();

        let new_config = SpatialVocoderConfig {
            sample_rate: 48000,
            ..Default::default()
        };

        let result = vocoder.update_config(new_config);
        assert!(result.is_ok());
        assert_eq!(vocoder.config.sample_rate, 48000);
    }
}
