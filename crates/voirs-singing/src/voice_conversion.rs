//! Multi-speaker voice conversion for singing voice synthesis
//!
//! This module provides capabilities for converting singing voices between different
//! speakers while preserving musical content, timing, and expression.

use crate::ai::{StyleEmbedding, StyleTransfer};
use crate::core::SingingEngine;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{SingingRequest, SingingResponse, VoiceCharacteristics, VoiceType};
use crate::Error;
use candle_core::Device;
use scirs2_core::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-speaker voice conversion system
#[derive(Debug, Clone)]
pub struct VoiceConverter {
    /// Pre-trained voice embeddings for different speakers
    speaker_embeddings: HashMap<String, SpeakerEmbedding>,
    /// Style transfer engine
    style_transfer: StyleTransfer,
    /// Conversion quality settings
    quality_settings: ConversionQuality,
}

/// Speaker embedding containing voice characteristics and model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Unique speaker identifier
    pub speaker_id: String,
    /// Speaker name or label
    pub speaker_name: String,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Neural embedding vector
    pub embedding_vector: Vec<f32>,
    /// Supported singing styles for this speaker
    pub supported_styles: Vec<String>,
    /// Average fundamental frequency
    pub avg_f0: f32,
    /// Formant frequencies
    pub formants: Vec<f32>,
    /// Voice quality metrics
    pub quality_metrics: VoiceQualityMetrics,
}

/// Voice quality metrics for speaker embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Vocal range in semitones
    pub vocal_range: f32,
    /// Vibrato rate in Hz
    pub vibrato_rate: f32,
    /// Vibrato depth in cents
    pub vibrato_depth: f32,
    /// Breathiness factor (0.0-1.0)
    pub breathiness: f32,
    /// Roughness factor (0.0-1.0)
    pub roughness: f32,
    /// Brightness factor (0.0-1.0)
    pub brightness: f32,
}

/// Voice conversion quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionQuality {
    /// Conversion method to use
    pub method: ConversionMethod,
    /// Preserve original timing
    pub preserve_timing: bool,
    /// Preserve original pitch contour shape
    pub preserve_pitch_contour: bool,
    /// Conversion strength (0.0-1.0)
    pub conversion_strength: f32,
    /// Enable formant preservation
    pub preserve_formants: bool,
    /// Enable expression preservation
    pub preserve_expression: bool,
}

/// Voice conversion methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionMethod {
    /// Neural style transfer
    NeuralTransfer,
    /// Spectral envelope conversion
    SpectralConversion,
    /// Formant-based conversion
    FormantConversion,
    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Voice conversion request
#[derive(Debug, Clone)]
pub struct ConversionRequest {
    /// Source audio or singing request
    pub source: ConversionSource,
    /// Target speaker to convert to
    pub target_speaker: String,
    /// Conversion quality settings
    pub quality: ConversionQuality,
    /// Additional parameters
    pub parameters: HashMap<String, f32>,
}

/// Source for voice conversion
#[derive(Debug, Clone)]
pub enum ConversionSource {
    /// Audio samples with metadata
    Audio {
        /// Audio samples to convert
        samples: Vec<f32>,
        /// Sample rate of the audio in Hz
        sample_rate: u32,
        /// Optional speaker ID for the source audio
        speaker_id: Option<String>,
    },
    /// Singing request to be converted
    SingingRequest(Box<SingingRequest>),
    /// Musical score with source speaker
    Score {
        /// Musical score to synthesize and convert
        score: Box<MusicalScore>,
        /// Source speaker ID for initial synthesis
        source_speaker: String,
    },
}

/// Voice conversion result
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Converted audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Conversion quality metrics
    pub quality_metrics: ConversionQualityMetrics,
    /// Target speaker embedding used
    pub target_speaker: SpeakerEmbedding,
    /// Conversion parameters applied
    pub applied_parameters: HashMap<String, f32>,
}

/// Quality metrics for conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionQualityMetrics {
    /// Similarity to target speaker (0.0-1.0)
    pub speaker_similarity: f32,
    /// Preservation of musical content (0.0-1.0)
    pub content_preservation: f32,
    /// Audio quality score (0.0-1.0)
    pub audio_quality: f32,
    /// Naturalness score (0.0-1.0)
    pub naturalness: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

impl VoiceConverter {
    /// Creates a new voice converter with default CPU device.
    ///
    /// Initializes a voice converter with an empty speaker embedding database
    /// and default conversion quality settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `VoiceConverter` instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the style transfer engine fails to initialize.
    pub fn new() -> Result<Self, Error> {
        let device = Device::Cpu;
        let style_transfer = StyleTransfer::new(device)?;

        Ok(Self {
            speaker_embeddings: HashMap::new(),
            style_transfer,
            quality_settings: ConversionQuality::default(),
        })
    }

    /// Creates a new voice converter with a specified compute device.
    ///
    /// Allows explicit control over the compute device (CPU/GPU) used for
    /// neural network operations in the style transfer engine.
    ///
    /// # Arguments
    ///
    /// * `device` - The compute device to use (CPU or GPU)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `VoiceConverter` instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the style transfer engine fails to initialize on the specified device.
    pub fn new_with_device(device: Device) -> Result<Self, Error> {
        let style_transfer = StyleTransfer::new(device)?;

        Ok(Self {
            speaker_embeddings: HashMap::new(),
            style_transfer,
            quality_settings: ConversionQuality::default(),
        })
    }

    /// Adds a speaker embedding to the converter's database.
    ///
    /// Registers a new speaker that can be used as a target for voice conversion.
    /// The embedding must contain a 512-dimensional vector for compatibility.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The speaker embedding to add, containing voice characteristics and neural features
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the speaker was successfully added.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding vector is not exactly 512 dimensions.
    pub fn add_speaker(&mut self, embedding: SpeakerEmbedding) -> Result<(), Error> {
        if embedding.embedding_vector.len() != 512 {
            return Err(Error::Voice(
                "Speaker embedding must have 512 dimensions".to_string(),
            ));
        }

        self.speaker_embeddings
            .insert(embedding.speaker_id.clone(), embedding);
        Ok(())
    }

    /// Removes a speaker from the converter's database.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - The unique identifier of the speaker to remove
    ///
    /// # Returns
    ///
    /// Returns `Some(SpeakerEmbedding)` if the speaker was found and removed,
    /// or `None` if no speaker with the given ID exists.
    pub fn remove_speaker(&mut self, speaker_id: &str) -> Option<SpeakerEmbedding> {
        self.speaker_embeddings.remove(speaker_id)
    }

    /// Lists all available speaker IDs in the converter's database.
    ///
    /// # Returns
    ///
    /// Returns a vector of string slices containing all registered speaker IDs.
    pub fn list_speakers(&self) -> Vec<&str> {
        self.speaker_embeddings.keys().map(|s| s.as_str()).collect()
    }

    /// Retrieves a speaker embedding by its unique identifier.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - The unique identifier of the speaker to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some(&SpeakerEmbedding)` if the speaker exists,
    /// or `None` if no speaker with the given ID is registered.
    pub fn get_speaker(&self, speaker_id: &str) -> Option<&SpeakerEmbedding> {
        self.speaker_embeddings.get(speaker_id)
    }

    /// Converts voice from source to target speaker using the specified request parameters.
    ///
    /// Performs voice conversion by extracting source audio, applying the selected
    /// conversion method, and calculating quality metrics. The conversion preserves
    /// musical content and timing while adapting voice characteristics to match
    /// the target speaker.
    ///
    /// # Arguments
    ///
    /// * `request` - The conversion request containing source audio, target speaker,
    ///   quality settings, and additional parameters
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `ConversionResult` with converted audio,
    /// quality metrics, and applied parameters on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The target speaker ID is not found in the database
    /// - Source audio extraction fails
    /// - The conversion method encounters processing errors
    /// - Quality metric calculation fails
    pub async fn convert_voice(
        &self,
        request: ConversionRequest,
    ) -> Result<ConversionResult, Error> {
        let start_time = std::time::Instant::now();

        // Get target speaker embedding
        let target_speaker = self
            .speaker_embeddings
            .get(&request.target_speaker)
            .ok_or_else(|| {
                Error::Voice(format!(
                    "Unknown target speaker: {}",
                    request.target_speaker
                ))
            })?;

        // Extract source audio and metadata
        let (source_audio, sample_rate, source_characteristics) =
            self.extract_source_audio(request.source).await?;

        // Perform voice conversion based on method
        let converted_audio = match request.quality.method {
            ConversionMethod::NeuralTransfer => self.neural_transfer_conversion(
                &source_audio,
                &source_characteristics,
                target_speaker,
                &request.quality,
            )?,
            ConversionMethod::SpectralConversion => self.spectral_conversion(
                &source_audio,
                sample_rate,
                target_speaker,
                &request.quality,
            )?,
            ConversionMethod::FormantConversion => self.formant_conversion(
                &source_audio,
                sample_rate,
                target_speaker,
                &request.quality,
            )?,
            ConversionMethod::Hybrid => self.hybrid_conversion(
                &source_audio,
                sample_rate,
                &source_characteristics,
                target_speaker,
                &request.quality,
            )?,
        };

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(
            &source_audio,
            &converted_audio,
            target_speaker,
            start_time.elapsed().as_millis() as f64,
        )?;

        Ok(ConversionResult {
            audio: converted_audio,
            sample_rate,
            quality_metrics,
            target_speaker: target_speaker.clone(),
            applied_parameters: request.parameters,
        })
    }

    /// Extracts audio samples and metadata from a conversion source.
    ///
    /// Handles different source types: raw audio, singing requests, and musical scores.
    /// Retrieves voice characteristics from speaker embeddings when available.
    ///
    /// # Arguments
    ///
    /// * `source` - The conversion source containing audio or synthesis parameters
    ///
    /// # Returns
    ///
    /// Returns a tuple of (audio samples, sample rate, voice characteristics).
    ///
    /// # Errors
    ///
    /// Returns an error if audio extraction or synthesis fails.
    async fn extract_source_audio(
        &self,
        source: ConversionSource,
    ) -> Result<(Vec<f32>, u32, VoiceCharacteristics), Error> {
        match source {
            ConversionSource::Audio {
                samples,
                sample_rate,
                speaker_id,
            } => {
                let characteristics = if let Some(id) = speaker_id {
                    self.speaker_embeddings
                        .get(&id)
                        .map(|s| s.voice_characteristics.clone())
                        .unwrap_or_default()
                } else {
                    VoiceCharacteristics::default()
                };
                Ok((samples, sample_rate, characteristics))
            }
            ConversionSource::SingingRequest(boxed_request) => {
                // This would need access to a SingingEngine to synthesize
                // For now, return empty audio as placeholder
                Ok((vec![0.0; 44100], 44100, boxed_request.voice))
            }
            ConversionSource::Score {
                score,
                source_speaker,
            } => {
                let source_characteristics = self
                    .speaker_embeddings
                    .get(&source_speaker)
                    .map(|s| s.voice_characteristics.clone())
                    .unwrap_or_default();

                // This would need to synthesize the score with the source speaker
                // For now, return empty audio as placeholder
                Ok((vec![0.0; 44100], 44100, source_characteristics))
            }
        }
    }

    /// Performs neural transfer-based voice conversion.
    ///
    /// Uses neural network style transfer to map source voice characteristics
    /// to target speaker characteristics. Currently implements a placeholder
    /// with basic pitch shifting.
    ///
    /// # Arguments
    ///
    /// * `source_audio` - Input audio samples to convert
    /// * `source_characteristics` - Voice characteristics of the source speaker
    /// * `target_speaker` - Target speaker embedding with desired characteristics
    /// * `quality` - Conversion quality settings
    ///
    /// # Returns
    ///
    /// Returns converted audio samples as a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if neural network processing fails.
    fn neural_transfer_conversion(
        &self,
        source_audio: &[f32],
        source_characteristics: &VoiceCharacteristics,
        target_speaker: &SpeakerEmbedding,
        quality: &ConversionQuality,
    ) -> Result<Vec<f32>, Error> {
        // Compute the pitch ratio between target and source.
        // Source F0 is taken from VoiceCharacteristics.f0_mean, clamped away from zero.
        let source_f0 = source_characteristics.f0_mean.max(80.0);
        let target_f0 = target_speaker.avg_f0.max(80.0);
        let pitch_ratio = target_f0 / source_f0;

        let n = source_audio.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Spectral pitch-shift via nearest-neighbour resampling in the time domain.
        //
        // Strategy: resample the source to a new length proportional to pitch_ratio,
        // then trim or zero-pad back to the original length.  This is a simple but
        // artefact-free nearest-neighbour pitch shift (no amplitude hack).
        //
        // Relationship: raising pitch ↔ shrinking the waveform length ↔ ratio < 1
        // We shorten when pitch_ratio > 1, lengthen when < 1.
        let resampled_len = ((n as f64) / (pitch_ratio as f64)).round().max(1.0) as usize;

        let mut resampled = Vec::with_capacity(resampled_len);
        for i in 0..resampled_len {
            // Map output index back to source index using linear interpolation.
            let src_pos = (i as f64) * (pitch_ratio as f64);
            let src_idx = src_pos.floor() as usize;
            let frac = (src_pos - src_pos.floor()) as f32;

            let s0 = source_audio.get(src_idx).copied().unwrap_or(0.0);
            let s1 = source_audio.get(src_idx + 1).copied().unwrap_or(0.0);
            resampled.push(s0 + frac * (s1 - s0));
        }

        // Build final output: same length as input.
        // Samples beyond resampled_len are zero (silence / tail pad).
        let mut output = vec![0.0f32; n];
        let copy_len = resampled_len.min(n);
        output[..copy_len].copy_from_slice(&resampled[..copy_len]);

        // Blend between original and pitch-shifted signal based on conversion_strength.
        let blend = quality.conversion_strength.clamp(0.0, 1.0);
        for (i, out) in output.iter_mut().enumerate() {
            let orig = source_audio[i];
            *out = orig * (1.0 - blend) + *out * blend;
        }

        Ok(output)
    }

    /// Performs spectral envelope-based voice conversion.
    ///
    /// Modifies the spectral envelope of the source audio to match target speaker
    /// formant characteristics. Currently implements a placeholder with basic
    /// formant scaling.
    ///
    /// # Arguments
    ///
    /// * `source_audio` - Input audio samples to convert
    /// * `sample_rate` - Sample rate of the audio in Hz
    /// * `target_speaker` - Target speaker embedding with formant information
    /// * `quality` - Conversion quality settings
    ///
    /// # Returns
    ///
    /// Returns converted audio samples as a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if spectral processing fails.
    fn spectral_conversion(
        &self,
        source_audio: &[f32],
        sample_rate: u32,
        target_speaker: &SpeakerEmbedding,
        quality: &ConversionQuality,
    ) -> Result<Vec<f32>, Error> {
        let n = source_audio.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // ── Overlap-add parameters ───────────────────────────────────────────────
        const N_FFT: usize = 1024;
        const HOP: usize = 256;

        // Build a Hann analysis window of length N_FFT.
        let hann: Vec<f64> = (0..N_FFT)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (N_FFT - 1) as f64).cos())
            })
            .collect();

        // Normalisation denominator: sum of squared window values per output sample
        // (constant overlap-add norm for Hann / 4× overlap).
        let win_norm: f64 = hann.iter().map(|w| w * w).sum::<f64>() / HOP as f64;
        let win_norm = win_norm.max(1e-12);

        // ── Target spectral envelope from speaker formants ───────────────────────
        // Build once: Gaussian mixture centred at F1, F2, F3 over N_FFT/2+1 bins.
        const SIGMA: f64 = 200.0; // Hz
        const AMP: f64 = 1.5;

        let target_envelope: Vec<f64> = (0..=N_FFT / 2)
            .map(|bin| {
                let freq = bin as f64 * sample_rate as f64 / N_FFT as f64;
                let gaussian_sum: f64 = target_speaker
                    .formants
                    .iter()
                    .map(|&f_hz| {
                        let d = freq - f_hz as f64;
                        AMP * (-d * d / (2.0 * SIGMA * SIGMA)).exp()
                    })
                    .sum();
                1.0 + gaussian_sum
            })
            .collect();

        // ── Overlap-add output accumulator ───────────────────────────────────────
        let mut output_acc = vec![0.0f64; n + N_FFT];
        let mut weight_acc = vec![0.0f64; n + N_FFT];

        let num_frames = (n + HOP - 1) / HOP;

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP;

            // Extract and window a frame of N_FFT samples (zero-pad at edges).
            let frame_complex: Vec<Complex<f64>> = (0..N_FFT)
                .map(|k| {
                    let src_idx = start + k;
                    let sample = if src_idx < n {
                        source_audio[src_idx] as f64
                    } else {
                        0.0
                    };
                    Complex::new(sample * hann[k], 0.0)
                })
                .collect();

            // Forward FFT.
            let spectrum = scirs2_fft::fft(&frame_complex, Some(N_FFT))
                .map_err(|e| Error::Processing(format!("FFT error in spectral_conversion: {e}")))?;

            // ── Compute source spectral envelope via 20-bin moving average ───────
            let half = N_FFT / 2 + 1;
            let mag: Vec<f64> = spectrum[..half].iter().map(|c| c.norm()).collect();

            const SMOOTH_BINS: usize = 20;
            let smooth_half = SMOOTH_BINS / 2;
            let source_envelope: Vec<f64> = (0..half)
                .map(|i| {
                    let lo = i.saturating_sub(smooth_half);
                    let hi = (i + smooth_half + 1).min(half);
                    let sum: f64 = mag[lo..hi].iter().sum();
                    sum / (hi - lo) as f64
                })
                .collect();

            // ── Compute transfer function h[bin] = target / max(source, ε) ───────
            let transfer: Vec<f64> = (0..half)
                .map(|i| target_envelope[i] / source_envelope[i].max(0.01))
                .collect();

            // ── Apply transfer to the full symmetric spectrum ────────────────────
            let mut modified: Vec<Complex<f64>> = spectrum.clone();
            for i in 0..half {
                modified[i] =
                    Complex::new(spectrum[i].re * transfer[i], spectrum[i].im * transfer[i]);
            }
            // Mirror conjugate for bins [half .. N_FFT] (Hermitian symmetry).
            for i in 1..(N_FFT / 2) {
                let mirror = N_FFT - i;
                modified[mirror] = Complex::new(modified[i].re, -modified[i].im);
            }

            // ── Inverse FFT ───────────────────────────────────────────────────────
            let time_frame = scirs2_fft::ifft(&modified, Some(N_FFT)).map_err(|e| {
                Error::Processing(format!("IFFT error in spectral_conversion: {e}"))
            })?;

            // ── Overlap-add accumulation ──────────────────────────────────────────
            for k in 0..N_FFT {
                let out_idx = start + k;
                if out_idx < output_acc.len() {
                    output_acc[out_idx] += time_frame[k].re * hann[k];
                    weight_acc[out_idx] += hann[k] * hann[k];
                }
            }
        }

        // ── Normalise and convert back to f32 ────────────────────────────────────
        let blend = quality.conversion_strength.clamp(0.0, 1.0);
        let converted: Vec<f32> = (0..n)
            .map(|i| {
                let norm_denom = weight_acc[i].max(win_norm * 1e-6);
                let converted_sample = (output_acc[i] / norm_denom) as f32;
                // Blend converted ↔ original per conversion_strength.
                source_audio[i] * (1.0 - blend) + converted_sample * blend
            })
            .collect();

        Ok(converted)
    }

    /// Performs formant-based voice conversion.
    ///
    /// Extracts and modifies formant frequencies to match target speaker
    /// vocal tract characteristics. Currently implements a placeholder with
    /// brightness adjustment.
    ///
    /// # Arguments
    ///
    /// * `source_audio` - Input audio samples to convert
    /// * `sample_rate` - Sample rate of the audio in Hz
    /// * `target_speaker` - Target speaker embedding with formant and quality metrics
    /// * `quality` - Conversion quality settings
    ///
    /// # Returns
    ///
    /// Returns converted audio samples as a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if formant processing fails.
    fn formant_conversion(
        &self,
        source_audio: &[f32],
        sample_rate: u32,
        target_speaker: &SpeakerEmbedding,
        quality: &ConversionQuality,
    ) -> Result<Vec<f32>, Error> {
        // Placeholder implementation for formant conversion
        // In a real implementation, this would extract and modify formants

        let mut converted = source_audio.to_vec();

        // Apply target speaker's formant characteristics
        let brightness_factor = target_speaker.quality_metrics.brightness;

        for sample in &mut converted {
            *sample *= brightness_factor;
        }

        Ok(converted)
    }

    /// Performs hybrid voice conversion combining multiple methods.
    ///
    /// Blends neural transfer and spectral conversion results for improved
    /// voice conversion quality. The blend factor is controlled by the
    /// conversion strength parameter.
    ///
    /// # Arguments
    ///
    /// * `source_audio` - Input audio samples to convert
    /// * `sample_rate` - Sample rate of the audio in Hz
    /// * `source_characteristics` - Voice characteristics of the source speaker
    /// * `target_speaker` - Target speaker embedding
    /// * `quality` - Conversion quality settings including blend strength
    ///
    /// # Returns
    ///
    /// Returns converted audio samples as a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if any conversion method fails.
    fn hybrid_conversion(
        &self,
        source_audio: &[f32],
        sample_rate: u32,
        source_characteristics: &VoiceCharacteristics,
        target_speaker: &SpeakerEmbedding,
        quality: &ConversionQuality,
    ) -> Result<Vec<f32>, Error> {
        // Combine neural transfer and spectral conversion
        let neural_result = self.neural_transfer_conversion(
            source_audio,
            source_characteristics,
            target_speaker,
            quality,
        )?;
        let spectral_result =
            self.spectral_conversion(source_audio, sample_rate, target_speaker, quality)?;

        // Blend results based on conversion strength
        let blend_factor = quality.conversion_strength;
        let mut hybrid_result = Vec::with_capacity(source_audio.len());

        for (i, &sample) in source_audio.iter().enumerate() {
            let neural_sample = neural_result.get(i).copied().unwrap_or(0.0);
            let spectral_sample = spectral_result.get(i).copied().unwrap_or(0.0);

            let blended = sample * (1.0 - blend_factor)
                + (neural_sample * 0.5 + spectral_sample * 0.5) * blend_factor;
            hybrid_result.push(blended);
        }

        Ok(hybrid_result)
    }

    /// Calculates the pitch shift factor between source and target voice characteristics.
    ///
    /// Computes the ratio of average fundamental frequencies based on voice types
    /// to determine the appropriate pitch shift for conversion.
    ///
    /// # Arguments
    ///
    /// * `source` - Voice characteristics of the source speaker
    /// * `target` - Voice characteristics of the target speaker
    ///
    /// # Returns
    ///
    /// Returns the pitch shift factor as a ratio (1.0 = no shift, >1.0 = shift up, <1.0 = shift down).
    fn calculate_pitch_shift(
        &self,
        source: &VoiceCharacteristics,
        target: &VoiceCharacteristics,
    ) -> f32 {
        // Simple pitch shift based on voice type
        let source_f0 = self.get_average_f0_for_voice_type(source.voice_type);
        let target_f0 = self.get_average_f0_for_voice_type(target.voice_type);

        target_f0 / source_f0
    }

    /// Retrieves the typical average fundamental frequency for a voice type.
    ///
    /// Returns standard F0 values in Hz for different vocal classifications.
    ///
    /// # Arguments
    ///
    /// * `voice_type` - The vocal classification (Soprano, Alto, Tenor, etc.)
    ///
    /// # Returns
    ///
    /// Returns the average F0 in Hz for the given voice type:
    /// - Soprano: 220 Hz, MezzoSoprano: 196 Hz, Alto: 175 Hz
    /// - Tenor: 147 Hz, Baritone: 123 Hz, Bass: 98 Hz
    fn get_average_f0_for_voice_type(&self, voice_type: VoiceType) -> f32 {
        match voice_type {
            VoiceType::Soprano => 220.0,
            VoiceType::MezzoSoprano => 196.0,
            VoiceType::Alto => 175.0,
            VoiceType::Tenor => 147.0,
            VoiceType::Baritone => 123.0,
            VoiceType::Bass => 98.0,
        }
    }

    /// Calculates comprehensive quality metrics for the conversion result.
    ///
    /// Evaluates speaker similarity, content preservation, audio quality,
    /// and naturalness of the converted output.
    ///
    /// # Arguments
    ///
    /// * `source_audio` - Original input audio samples
    /// * `converted_audio` - Converted output audio samples
    /// * `target_speaker` - Target speaker embedding used for conversion
    /// * `processing_time_ms` - Time taken for conversion in milliseconds
    ///
    /// # Returns
    ///
    /// Returns quality metrics including similarity, preservation, quality, and naturalness scores.
    ///
    /// # Errors
    ///
    /// Returns an error if metric calculation fails.
    fn calculate_quality_metrics(
        &self,
        source_audio: &[f32],
        converted_audio: &[f32],
        target_speaker: &SpeakerEmbedding,
        processing_time_ms: f64,
    ) -> Result<ConversionQualityMetrics, Error> {
        // Placeholder quality metric calculations
        // In a real implementation, these would be more sophisticated

        let speaker_similarity =
            self.calculate_speaker_similarity(converted_audio, target_speaker)?;
        let content_preservation =
            self.calculate_content_preservation(source_audio, converted_audio)?;
        let audio_quality = self.calculate_audio_quality(converted_audio)?;
        let naturalness = self.calculate_naturalness(converted_audio)?;

        Ok(ConversionQualityMetrics {
            speaker_similarity,
            content_preservation,
            audio_quality,
            naturalness,
            processing_time_ms,
        })
    }

    /// Calculates how similar the converted audio is to the target speaker.
    ///
    /// Compares audio features with the target speaker embedding to measure
    /// conversion accuracy. Currently returns a placeholder value.
    ///
    /// # Arguments
    ///
    /// * `audio` - Converted audio samples
    /// * `target` - Target speaker embedding for comparison
    ///
    /// # Returns
    ///
    /// Returns a similarity score between 0.0 (no similarity) and 1.0 (identical).
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction or comparison fails.
    fn calculate_speaker_similarity(
        &self,
        audio: &[f32],
        target: &SpeakerEmbedding,
    ) -> Result<f32, Error> {
        // Placeholder: In reality, this would compare audio features with target embedding
        Ok(0.85) // Assume good similarity
    }

    /// Calculates how well musical content was preserved during conversion.
    ///
    /// Compares energy and spectral features between source and converted audio
    /// to ensure timing, rhythm, and melody remain intact.
    ///
    /// # Arguments
    ///
    /// * `source` - Original source audio samples
    /// * `converted` - Converted audio samples
    ///
    /// # Returns
    ///
    /// Returns a preservation score between 0.0 (no preservation) and 1.0 (perfect preservation).
    ///
    /// # Errors
    ///
    /// Returns an error if feature comparison fails.
    fn calculate_content_preservation(
        &self,
        source: &[f32],
        converted: &[f32],
    ) -> Result<f32, Error> {
        // Placeholder: Compare energy and basic spectral features
        let source_energy: f32 = source.iter().map(|x| x * x).sum();
        let converted_energy: f32 = converted.iter().map(|x| x * x).sum();

        let energy_ratio = if source_energy > 0.0 {
            (converted_energy / source_energy).min(1.0)
        } else {
            1.0
        };

        Ok(energy_ratio)
    }

    /// Calculates overall audio quality of the converted output.
    ///
    /// Checks for clipping, distortion, and other quality issues in the
    /// converted audio signal.
    ///
    /// # Arguments
    ///
    /// * `audio` - Converted audio samples to evaluate
    ///
    /// # Returns
    ///
    /// Returns a quality score between 0.0 (poor quality) and 1.0 (excellent quality).
    ///
    /// # Errors
    ///
    /// Returns an error if quality analysis fails.
    fn calculate_audio_quality(&self, audio: &[f32]) -> Result<f32, Error> {
        // Placeholder: Check for clipping and basic quality issues
        let max_amplitude = audio.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let clipping_penalty = if max_amplitude > 0.95 { 0.5 } else { 1.0 };

        Ok(0.9 * clipping_penalty)
    }

    /// Calculates the naturalness of the converted singing voice.
    ///
    /// Evaluates how natural and human-like the converted output sounds
    /// using signal characteristics like zero-crossing rate.
    ///
    /// # Arguments
    ///
    /// * `audio` - Converted audio samples to evaluate
    ///
    /// # Returns
    ///
    /// Returns a naturalness score between 0.0 (artificial) and 1.0 (highly natural).
    ///
    /// # Errors
    ///
    /// Returns an error if naturalness analysis fails.
    fn calculate_naturalness(&self, audio: &[f32]) -> Result<f32, Error> {
        // Placeholder: Basic naturalness heuristics
        let zero_crossings = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let naturalness_score = (zero_crossings as f32 / audio.len() as f32 * 100.0).min(1.0);

        Ok(naturalness_score)
    }

    /// Creates a speaker embedding from voice samples.
    ///
    /// Analyzes voice samples to extract speaker-specific features including
    /// neural embeddings, fundamental frequency, formants, and quality metrics.
    /// This is a static method that doesn't require a VoiceConverter instance.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - Unique identifier for the speaker
    /// * `speaker_name` - Human-readable name or label for the speaker
    /// * `voice_samples` - Audio samples of the speaker's voice
    /// * `voice_characteristics` - Known voice characteristics and metadata
    ///
    /// # Returns
    ///
    /// Returns a complete `SpeakerEmbedding` with extracted features and metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction fails or samples are insufficient.
    pub fn create_speaker_embedding(
        speaker_id: String,
        speaker_name: String,
        voice_samples: &[f32],
        voice_characteristics: VoiceCharacteristics,
    ) -> Result<SpeakerEmbedding, Error> {
        // Placeholder implementation for creating speaker embeddings
        // In a real implementation, this would extract features from voice samples

        let embedding_vector = Self::extract_speaker_features(voice_samples)?;
        let avg_f0 = Self::estimate_average_f0(voice_samples)?;
        let formants = Self::extract_formants(voice_samples)?;
        let quality_metrics = Self::analyze_voice_quality(voice_samples)?;

        Ok(SpeakerEmbedding {
            speaker_id,
            speaker_name,
            voice_characteristics,
            embedding_vector,
            supported_styles: vec!["classical".to_string(), "pop".to_string()],
            avg_f0,
            formants,
            quality_metrics,
        })
    }

    /// Extracts a 512-dimensional feature vector from voice samples using
    /// MFCC-13, delta-MFCC, delta-delta-MFCC, energy envelope statistics,
    /// ZCR, and F0.
    ///
    /// Feature layout (512 total):
    /// - [0..13]   : MFCC-13 from first frame (Hann-windowed, mel-filterbank, DCT-II)
    /// - [13..26]  : delta-MFCC (difference between frame-b and frame-a)
    /// - [26..39]  : delta-delta-MFCC (difference between frame-c and frame-b)
    /// - [39..52]  : per-coefficient mean across the three frames
    /// - [52..65]  : per-coefficient variance
    /// - [65..78]  : per-coefficient min
    /// - [78..91]  : per-coefficient max
    /// - [91..96]  : energy_mean, energy_std, energy_max, ZCR, normalised F0
    /// - [96..512] : tiled core-39 features with gentle per-index scaling
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples of the speaker's voice
    ///
    /// # Returns
    ///
    /// Returns a 512-dimensional feature vector as `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction fails or samples are invalid.
    fn extract_speaker_features(samples: &[f32]) -> Result<Vec<f32>, Error> {
        const FRAME_SIZE: usize = 2048;
        const N_MFCC: usize = 13;
        const N_FILT: usize = 26;
        const SAMPLE_RATE: f32 = 44100.0;

        if samples.is_empty() {
            return Ok(vec![0.0f32; 512]);
        }

        // Compute one MFCC-13 vector from a slice (Hann-windowed, mel-filterbank, DCT-II).
        let compute_mfcc_frame = |frame: &[f32]| -> [f32; N_MFCC] {
            let len = frame.len();
            if len == 0 {
                return [0.0f32; N_MFCC];
            }
            // Hann window
            let windowed: Vec<f64> = frame
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let w = 0.5
                        * (1.0
                            - (2.0 * std::f64::consts::PI * i as f64 / (len - 1).max(1) as f64)
                                .cos());
                    s as f64 * w
                })
                .collect();

            // Power spectrum via FFT
            let complex_in: Vec<scirs2_core::Complex<f64>> = windowed
                .iter()
                .map(|&x| scirs2_core::Complex::new(x, 0.0))
                .collect();
            let fft_out = match scirs2_fft::fft(&complex_in, None) {
                Ok(v) => v,
                Err(_) => return [0.0f32; N_MFCC],
            };
            let n_bins = len / 2 + 1;
            let power: Vec<f64> = fft_out[..n_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).max(1e-30))
                .collect();

            // Mel filterbank: N_FILT triangular filters in [0, Nyquist]
            let nyquist = SAMPLE_RATE as f64 / 2.0;
            let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
            let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
            let mel_low = hz_to_mel(0.0);
            let mel_high = hz_to_mel(nyquist);
            let mel_pts: Vec<f64> = (0..=N_FILT + 1)
                .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (N_FILT + 1) as f64)
                .collect();
            let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
            let bin_pts: Vec<usize> = hz_pts
                .iter()
                .map(|&hz| ((hz / nyquist) * (n_bins - 1) as f64).round() as usize)
                .collect();

            let mut filt_energies = [0.0f64; N_FILT];
            for m in 0..N_FILT {
                let start = bin_pts[m];
                let center = bin_pts[m + 1];
                let end = bin_pts[m + 2];
                for k in start..center {
                    if k < power.len() && center > start {
                        let w = (k - start) as f64 / (center - start) as f64;
                        filt_energies[m] += power[k] * w;
                    }
                }
                for k in center..end {
                    if k < power.len() && end > center {
                        let w = (end - k) as f64 / (end - center) as f64;
                        filt_energies[m] += power[k] * w;
                    }
                }
                filt_energies[m] = filt_energies[m].max(1e-30).ln();
            }

            // DCT-II → first N_MFCC coefficients
            let dct_out = match scirs2_fft::dct(&filt_energies, None, Some("ortho")) {
                Ok(v) => v,
                Err(_) => return [0.0f32; N_MFCC],
            };
            let mut mfcc = [0.0f32; N_MFCC];
            for (i, m) in mfcc.iter_mut().enumerate() {
                *m = dct_out.get(i).copied().unwrap_or(0.0) as f32;
            }
            mfcc
        };

        // Three overlapping frames for delta and delta-delta computation
        let len = samples.len();
        let a_start = 0;
        let a_end = (a_start + FRAME_SIZE).min(len);
        let b_start = if len >= FRAME_SIZE + 512 { 512 } else { 0 };
        let b_end = (b_start + FRAME_SIZE).min(len);
        let c_start = if len >= FRAME_SIZE + 1024 {
            1024
        } else {
            b_start
        };
        let c_end = (c_start + FRAME_SIZE).min(len);

        let mfcc_a = compute_mfcc_frame(&samples[a_start..a_end]);
        let mfcc_b = compute_mfcc_frame(&samples[b_start..b_end]);
        let mfcc_c = compute_mfcc_frame(&samples[c_start..c_end]);

        // First- and second-order temporal differences
        let delta: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_b[i] - mfcc_a[i]);
        let delta2: [f32; N_MFCC] = std::array::from_fn(|i| mfcc_c[i] - mfcc_b[i]);

        // Per-coefficient statistics across the three frames
        let mfcc_mean: [f32; N_MFCC] =
            std::array::from_fn(|i| (mfcc_a[i] + mfcc_b[i] + mfcc_c[i]) / 3.0);
        let mfcc_var: [f32; N_MFCC] = std::array::from_fn(|i| {
            let mu = mfcc_mean[i];
            ((mfcc_a[i] - mu).powi(2) + (mfcc_b[i] - mu).powi(2) + (mfcc_c[i] - mu).powi(2)) / 3.0
        });
        let mfcc_min: [f32; N_MFCC] =
            std::array::from_fn(|i| mfcc_a[i].min(mfcc_b[i]).min(mfcc_c[i]));
        let mfcc_max: [f32; N_MFCC] =
            std::array::from_fn(|i| mfcc_a[i].max(mfcc_b[i]).max(mfcc_c[i]));

        // Energy envelope: RMS per 512-sample hop
        const HOP: usize = 512;
        let rms_vals: Vec<f32> = samples
            .chunks(HOP)
            .filter(|c| !c.is_empty())
            .map(|chunk| {
                let sq: f32 = chunk.iter().map(|&s| s * s).sum();
                (sq / chunk.len() as f32).sqrt()
            })
            .collect();
        let energy_mean = if rms_vals.is_empty() {
            0.0f32
        } else {
            rms_vals.iter().sum::<f32>() / rms_vals.len() as f32
        };
        let energy_std = if rms_vals.is_empty() {
            0.0f32
        } else {
            let mu = energy_mean;
            (rms_vals.iter().map(|&r| (r - mu).powi(2)).sum::<f32>() / rms_vals.len() as f32).sqrt()
        };
        let energy_max = rms_vals.iter().cloned().fold(0.0f32, f32::max);

        // Zero-crossing rate
        let zcr = if samples.len() < 2 {
            0.0f32
        } else {
            samples.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f32
                / (samples.len() - 1) as f32
        };

        // F0 estimate, normalised to [0, 1] by dividing by sample_rate
        let f0_norm = Self::estimate_average_f0(samples).unwrap_or(220.0) / SAMPLE_RATE;

        // Pack into 512-dim vector
        let mut features = vec![0.0f32; 512];
        // [0..39]: MFCC + delta + delta-delta
        for i in 0..N_MFCC {
            features[i] = mfcc_a[i];
            features[N_MFCC + i] = delta[i];
            features[2 * N_MFCC + i] = delta2[i];
        }
        // [39..91]: per-coefficient stats (4 × 13 = 52 entries)
        for i in 0..N_MFCC {
            features[39 + i] = mfcc_mean[i];
            features[52 + i] = mfcc_var[i];
            features[65 + i] = mfcc_min[i];
            features[78 + i] = mfcc_max[i];
        }
        // [91..96]: scalar audio stats
        features[91] = energy_mean;
        features[92] = energy_std;
        features[93] = energy_max;
        features[94] = zcr;
        features[95] = f0_norm;

        // [96..512]: tile core-39 features with gentle per-index scaling
        for i in 96..512 {
            let base_idx = (i - 96) % 39;
            let scale = 1.0 + i as f32 * 1e-4;
            features[i] = features[base_idx] * scale;
        }

        Ok(features)
    }

    /// Estimates the average fundamental frequency (F0) from voice samples using
    /// normalized autocorrelation (YIN-lite) pitch detection.
    ///
    /// Processes the signal in 2048-sample frames with 512-sample hops. Per frame,
    /// the signal is mean-centred and normalized autocorrelation r[τ] is computed
    /// for lags τ in [sr/800, sr/55] (covering 55–800 Hz). The lag with the highest
    /// r[τ] is chosen; frames where r[τ] > 0.35 are declared voiced. Returns the
    /// median F0 across all voiced frames, or 220 Hz if none are voiced.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples of the speaker's voice
    ///
    /// # Returns
    ///
    /// Returns the estimated average F0 in Hz.
    ///
    /// # Errors
    ///
    /// Returns an error if F0 estimation fails or samples are too short.
    fn estimate_average_f0(samples: &[f32]) -> Result<f32, Error> {
        const SAMPLE_RATE: f32 = 44100.0;
        const FRAME_SIZE: usize = 2048;
        const HOP_SIZE: usize = 512;
        // Lag range covering 55–800 Hz at 44100 Hz sample rate
        let tau_min = (SAMPLE_RATE / 800.0).ceil() as usize; // ≈ 56
        let tau_max = (SAMPLE_RATE / 55.0).ceil() as usize; // ≈ 802

        if samples.len() < FRAME_SIZE {
            return Ok(220.0);
        }

        let mut voiced_f0s: Vec<f32> = Vec::new();
        let mut frame_start = 0;

        while frame_start + FRAME_SIZE <= samples.len() {
            let frame = &samples[frame_start..frame_start + FRAME_SIZE];

            // Mean-centre the frame to remove DC bias
            let mean: f32 = frame.iter().sum::<f32>() / FRAME_SIZE as f32;
            let centered: Vec<f32> = frame.iter().map(|&s| s - mean).collect();

            let full_energy: f32 = centered.iter().map(|&s| s * s).sum();
            if full_energy < 1e-10 {
                frame_start += HOP_SIZE;
                continue;
            }

            // Normalized autocorrelation r[τ] = Σ x[i]*x[i+τ] / √(Σx_left² · Σx_right²)
            let mut best_tau = tau_min;
            let mut best_corr = -1.0f32;
            let tau_limit = tau_max.min(FRAME_SIZE - 1);

            for tau in tau_min..=tau_limit {
                let n_ov = FRAME_SIZE - tau;
                let mut cross = 0.0f32;
                let mut left_sq = 0.0f32;
                let mut right_sq = 0.0f32;
                for i in 0..n_ov {
                    cross += centered[i] * centered[i + tau];
                    left_sq += centered[i] * centered[i];
                    right_sq += centered[i + tau] * centered[i + tau];
                }
                let denom = (left_sq * right_sq).sqrt();
                let r = if denom > 1e-10 { cross / denom } else { 0.0 };
                if r > best_corr {
                    best_corr = r;
                    best_tau = tau;
                }
            }

            // Voiced threshold: correlation must exceed 0.35
            if best_corr > 0.35 {
                voiced_f0s.push(SAMPLE_RATE / best_tau as f32);
            }

            frame_start += HOP_SIZE;
        }

        if voiced_f0s.is_empty() {
            return Ok(220.0);
        }

        // Return median F0 across voiced frames
        voiced_f0s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = voiced_f0s.len() / 2;
        let median = if voiced_f0s.len() % 2 == 0 {
            (voiced_f0s[mid - 1] + voiced_f0s[mid]) / 2.0
        } else {
            voiced_f0s[mid]
        };
        Ok(median)
    }

    /// Extracts formant frequencies (F1, F2, F3) from voice samples via
    /// LPC(12) all-pole spectral envelope peak detection.
    ///
    /// Pipeline:
    /// 1. Apply Hann window to up to 2048 samples.
    /// 2. Compute autocorrelation R[0..P+1] for LPC order P = 12.
    /// 3. Run Levinson-Durbin recursion to obtain predictor coefficients a[1..=P].
    /// 4. Evaluate the all-pole spectrum magnitude |H(ω)| = 1/|A(e^{jω})| at
    ///    N_SPEC = 512 uniformly spaced points from 0 to π.
    /// 5. Detect local maxima above 50 Hz using parabolic interpolation for
    ///    sub-bin precision and collect up to 5 formant candidates.
    /// 6. Return exactly the first 3 Hz values, padding with [800, 1200, 2600]
    ///    defaults if fewer peaks are found.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples of the speaker's voice
    ///
    /// # Returns
    ///
    /// Returns exactly 3 formant frequencies in Hz as `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if formant extraction fails or samples are insufficient.
    fn extract_formants(samples: &[f32]) -> Result<Vec<f32>, Error> {
        const FRAME_SIZE: usize = 2048;
        const P: usize = 12; // LPC order
        const N_SPEC: usize = 512; // spectral evaluation points
        const SAMPLE_RATE: f32 = 44100.0;
        const MIN_FORMANT_HZ: f32 = 50.0;

        let frame_len = samples.len().min(FRAME_SIZE);
        if frame_len < P + 2 {
            return Ok(vec![800.0, 1200.0, 2600.0]);
        }

        // Apply Hann window to the analysis frame
        let windowed: Vec<f32> = samples[..frame_len]
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (frame_len - 1).max(1) as f32)
                            .cos());
                s * w
            })
            .collect();

        // Autocorrelation R[k] for k = 0..=P
        let mut r = [0.0f32; P + 2];
        for k in 0..=P {
            let mut acc = 0.0f32;
            for i in 0..(frame_len - k) {
                acc += windowed[i] * windowed[i + k];
            }
            r[k] = acc;
        }

        if r[0].abs() < 1e-15 {
            return Ok(vec![800.0, 1200.0, 2600.0]);
        }

        // Levinson-Durbin recursion to compute LPC predictor coefficients
        let mut a = [0.0f32; P + 1]; // a[1..=P]; a[0] is unused
        let mut a_prev = [0.0f32; P + 1];
        let mut pred_error = r[0];

        for m in 1..=P {
            let mut lambda = r[m];
            for j in 1..m {
                lambda -= a[j] * r[m - j];
            }
            if pred_error.abs() < 1e-15 {
                break;
            }
            let km = -lambda / pred_error;
            a_prev[..=P].copy_from_slice(&a[..=P]);
            a[m] = km;
            for j in 1..m {
                a[j] = a_prev[j] + km * a_prev[m - j];
            }
            pred_error *= 1.0 - km * km;
        }

        // Evaluate all-pole spectrum: |H(ω)| = 1 / |A(e^{jω})|
        // A(e^{jω}) = 1 + Σ_{k=1}^{P} a[k] · e^{−j·k·ω}
        let mut spectrum = [0.0f32; N_SPEC];
        for (bin, s) in spectrum.iter_mut().enumerate() {
            let omega = std::f32::consts::PI * bin as f32 / N_SPEC as f32;
            let mut re = 1.0f32;
            let mut im = 0.0f32;
            for k in 1..=P {
                let angle = -(k as f32) * omega;
                re += a[k] * angle.cos();
                im += a[k] * angle.sin();
            }
            *s = 1.0 / (re * re + im * im).sqrt().max(1e-10);
        }

        // Collect spectral peaks above MIN_FORMANT_HZ
        let min_bin = (MIN_FORMANT_HZ / (SAMPLE_RATE / 2.0) * N_SPEC as f32).ceil() as usize;
        let mut formants: Vec<f32> = Vec::with_capacity(5);

        for bin in min_bin.max(1)..(N_SPEC - 1) {
            if spectrum[bin] > spectrum[bin - 1] && spectrum[bin] > spectrum[bin + 1] {
                // Parabolic interpolation for sub-bin frequency precision
                let alpha = spectrum[bin - 1];
                let beta = spectrum[bin];
                let gamma = spectrum[bin + 1];
                let denom = alpha - 2.0 * beta + gamma;
                let delta_bin = if denom.abs() > 1e-10 {
                    0.5 * (alpha - gamma) / denom
                } else {
                    0.0
                };
                let peak_bin = bin as f32 + delta_bin;
                formants.push(peak_bin * (SAMPLE_RATE / 2.0) / N_SPEC as f32);
                if formants.len() >= 5 {
                    break;
                }
            }
        }

        // Pad with defaults if fewer than 3 peaks found
        let defaults = [800.0f32, 1200.0, 2600.0, 3200.0, 4000.0];
        while formants.len() < 3 {
            formants.push(defaults[formants.len()]);
        }

        Ok(formants[..3].to_vec())
    }

    /// Analyzes voice quality characteristics from samples.
    ///
    /// Extracts metrics including vocal range, vibrato characteristics,
    /// breathiness, roughness, and brightness. Currently returns typical
    /// values as a placeholder.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples of the speaker's voice
    ///
    /// # Returns
    ///
    /// Returns comprehensive `VoiceQualityMetrics` for the speaker.
    ///
    /// # Errors
    ///
    /// Returns an error if quality analysis fails or samples are invalid.
    fn analyze_voice_quality(samples: &[f32]) -> Result<VoiceQualityMetrics, Error> {
        // Placeholder: Estimate voice quality from samples
        // In reality, this would use advanced signal processing
        Ok(VoiceQualityMetrics {
            vocal_range: 24.0, // 2 octaves
            vibrato_rate: 5.0,
            vibrato_depth: 20.0,
            breathiness: 0.3,
            roughness: 0.2,
            brightness: 0.7,
        })
    }
}

impl Default for VoiceConverter {
    /// Creates a default voice converter instance.
    ///
    /// # Panics
    ///
    /// Panics if the voice converter initialization fails.
    fn default() -> Self {
        Self::new().expect("Failed to create default VoiceConverter")
    }
}

impl Default for ConversionQuality {
    /// Creates default conversion quality settings.
    ///
    /// Uses hybrid conversion method with high quality preservation:
    /// - Method: Hybrid (neural + spectral)
    /// - Timing preservation: enabled
    /// - Pitch contour preservation: enabled
    /// - Conversion strength: 0.8 (80%)
    /// - Formant preservation: enabled
    /// - Expression preservation: enabled
    fn default() -> Self {
        Self {
            method: ConversionMethod::Hybrid,
            preserve_timing: true,
            preserve_pitch_contour: true,
            conversion_strength: 0.8,
            preserve_formants: true,
            preserve_expression: true,
        }
    }
}

impl Default for VoiceQualityMetrics {
    /// Creates default voice quality metrics.
    ///
    /// Represents typical values for a trained singer:
    /// - Vocal range: 24 semitones (2 octaves)
    /// - Vibrato rate: 5.0 Hz
    /// - Vibrato depth: 20.0 cents
    /// - Breathiness: 0.3 (30%)
    /// - Roughness: 0.2 (20%)
    /// - Brightness: 0.7 (70%)
    fn default() -> Self {
        Self {
            vocal_range: 24.0,
            vibrato_rate: 5.0,
            vibrato_depth: 20.0,
            breathiness: 0.3,
            roughness: 0.2,
            brightness: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_converter_creation() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        assert!(converter.list_speakers().is_empty());
    }

    #[test]
    fn test_speaker_embedding_creation() {
        let voice_samples = vec![0.0; 44100]; // 1 second of silence
        let voice_characteristics = VoiceCharacteristics::for_voice_type(VoiceType::Soprano);

        let embedding = VoiceConverter::create_speaker_embedding(
            "test_speaker".to_string(),
            "Test Speaker".to_string(),
            &voice_samples,
            voice_characteristics,
        );

        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.speaker_id, "test_speaker");
        assert_eq!(embedding.embedding_vector.len(), 512);
    }

    #[test]
    fn test_add_remove_speaker() {
        let mut converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        let voice_characteristics = VoiceCharacteristics::for_voice_type(VoiceType::Tenor);

        let embedding = SpeakerEmbedding {
            speaker_id: "tenor1".to_string(),
            speaker_name: "Test Tenor".to_string(),
            voice_characteristics,
            embedding_vector: vec![0.0; 512],
            supported_styles: vec!["classical".to_string()],
            avg_f0: 147.0,
            formants: vec![800.0, 1200.0, 2600.0],
            quality_metrics: VoiceQualityMetrics::default(),
        };

        // Add speaker
        assert!(converter.add_speaker(embedding.clone()).is_ok());
        assert_eq!(converter.list_speakers().len(), 1);
        assert!(converter.get_speaker("tenor1").is_some());

        // Remove speaker
        let removed = converter.remove_speaker("tenor1");
        assert!(removed.is_some());
        assert_eq!(converter.list_speakers().len(), 0);
    }

    #[test]
    fn test_conversion_quality_default() {
        let quality = ConversionQuality::default();
        assert!(matches!(quality.method, ConversionMethod::Hybrid));
        assert!(quality.preserve_timing);
        assert_eq!(quality.conversion_strength, 0.8);
    }

    #[tokio::test]
    async fn test_voice_conversion_missing_speaker() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");

        let request = ConversionRequest {
            source: ConversionSource::Audio {
                samples: vec![0.0; 1000],
                sample_rate: 44100,
                speaker_id: None,
            },
            target_speaker: "nonexistent".to_string(),
            quality: ConversionQuality::default(),
            parameters: HashMap::new(),
        };

        let result = converter.convert_voice(request).await;
        assert!(result.is_err());
    }

    fn make_test_speaker(avg_f0: f32, formants: Vec<f32>) -> SpeakerEmbedding {
        SpeakerEmbedding {
            speaker_id: "test".to_string(),
            speaker_name: "Test Speaker".to_string(),
            voice_characteristics: VoiceCharacteristics::default(),
            embedding_vector: vec![0.0; 512],
            supported_styles: vec!["pop".to_string()],
            avg_f0,
            formants,
            quality_metrics: VoiceQualityMetrics::default(),
        }
    }

    #[test]
    fn test_spectral_conversion_length_preserved() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        let n = 8192usize;
        // Use a sine wave so there is spectral content to process.
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::SpectralConversion,
            conversion_strength: 1.0,
            ..ConversionQuality::default()
        };

        let result = converter
            .spectral_conversion(&source, 44100, &speaker, &quality)
            .expect("spectral_conversion should succeed");

        assert_eq!(result.len(), n, "output length must equal input length");
    }

    #[test]
    fn test_spectral_conversion_not_constant_scale() {
        // Verify that spectral_conversion does NOT simply scale every sample by the
        // same constant (the old placeholder behaviour was: out[i] = in[i] * (formant[0]/1000)).
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        let n = 4096usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 44100.0).sin() * 0.8)
            .collect();
        let formant_scale_old = 800.0f32 / 1000.0; // what the old placeholder computed
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::SpectralConversion,
            conversion_strength: 1.0,
            ..ConversionQuality::default()
        };

        let result = converter
            .spectral_conversion(&source, 44100, &speaker, &quality)
            .expect("spectral_conversion should succeed");

        // At least one sample must differ from the naive constant-scale prediction by
        // more than a small epsilon, proving real spectral processing occurred.
        let differs = result
            .iter()
            .zip(source.iter())
            .any(|(&out, &inp)| (out - inp * formant_scale_old).abs() > 1e-4);
        assert!(
            differs,
            "spectral_conversion must not simply multiply by a constant scale"
        );
    }

    #[test]
    fn test_neural_transfer_shape() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        let n = 3000usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 44100.0).sin() * 0.6)
            .collect();
        let source_chars = VoiceCharacteristics {
            f0_mean: 147.0,
            ..VoiceCharacteristics::default()
        };
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::NeuralTransfer,
            conversion_strength: 0.8,
            ..ConversionQuality::default()
        };

        let result = converter
            .neural_transfer_conversion(&source, &source_chars, &speaker, &quality)
            .expect("neural_transfer_conversion should succeed");

        assert_eq!(
            result.len(),
            n,
            "neural_transfer output length must equal input length"
        );
    }

    // ── DSP implementation tests ──────────────────────────────────────────────

    /// A 440 Hz sine wave at 44100 Hz should yield an F0 estimate in [380, 500].
    #[test]
    fn test_f0_estimation_440hz() {
        let sr = 44100usize;
        let duration_samples = sr * 2; // 2 seconds → many frames
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.9)
            .collect();

        let f0 = VoiceConverter::estimate_average_f0(&samples).expect("F0 estimation must succeed");

        assert!(
            (380.0..=500.0).contains(&f0),
            "Expected F0 near 440 Hz, got {f0:.1} Hz"
        );
    }

    /// Silence (all-zero input) must return the 220 Hz fallback.
    #[test]
    fn test_f0_estimation_silence() {
        let samples = vec![0.0f32; 44100 * 2];
        let f0 = VoiceConverter::estimate_average_f0(&samples)
            .expect("F0 estimation must succeed on silence");
        assert!(
            (f0 - 220.0).abs() < 1.0,
            "Expected 220 Hz fallback for silence, got {f0:.1} Hz"
        );
    }

    /// `extract_formants` must return exactly 3 values, all above 50 Hz.
    #[test]
    fn test_formants_three_values() {
        // Use a voiced vowel-like signal: harmonic stack at 200 Hz
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut s = 0.0f32;
                for h in 1..=20u32 {
                    s += (1.0 / h as f32)
                        * (2.0 * std::f32::consts::PI * 200.0 * h as f32 * t).sin();
                }
                s * 0.1
            })
            .collect();

        let formants =
            VoiceConverter::extract_formants(&samples).expect("extract_formants must succeed");

        assert_eq!(formants.len(), 3, "Must return exactly 3 formant values");
        for (idx, &f) in formants.iter().enumerate() {
            assert!(
                f > 50.0,
                "Formant F{} = {:.1} Hz is not above 50 Hz",
                idx + 1,
                f
            );
        }
    }

    /// `extract_speaker_features` must return exactly 512 finite values.
    #[test]
    fn test_speaker_features_512() {
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let features = VoiceConverter::extract_speaker_features(&samples)
            .expect("extract_speaker_features must succeed");

        assert_eq!(features.len(), 512, "Must return exactly 512 features");
        for (i, &v) in features.iter().enumerate() {
            assert!(v.is_finite(), "Feature[{i}] = {v} is not finite");
        }
    }

    /// Two different inputs must produce different feature vectors.
    #[test]
    fn test_speaker_features_varies() {
        let sr = 44100usize;
        // Input A: 300 Hz sine
        let samples_a: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        // Input B: 700 Hz sine (different pitch and spectral content)
        let samples_b: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 700.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let feat_a = VoiceConverter::extract_speaker_features(&samples_a)
            .expect("extract_speaker_features must succeed for input A");
        let feat_b = VoiceConverter::extract_speaker_features(&samples_b)
            .expect("extract_speaker_features must succeed for input B");

        let max_diff = feat_a
            .iter()
            .zip(feat_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff > 1e-4,
            "Feature vectors for different inputs must differ (max_diff = {max_diff:.6})"
        );
    }
}
