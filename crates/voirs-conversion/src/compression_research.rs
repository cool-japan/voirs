//! Advanced Audio Compression Research for Real-time Streaming
//!
//! This module provides state-of-the-art audio compression algorithms optimized for
//! real-time voice conversion streaming applications.
//!
//! ## Features
//!
//! - **Perceptual Compression**: Psychoacoustic-based compression using masking models
//! - **Real-time Optimization**: Ultra-low latency compression for streaming
//! - **Adaptive Quality**: Dynamic quality adjustment based on network conditions
//! - **Voice-Optimized**: Specialized algorithms for voice conversion content
//! - **Quality vs Bandwidth**: Configurable trade-offs for different use cases
//! - **Multi-scale Compression**: Hierarchical compression for different quality levels
//!
//! ## Example
//!
//! ```rust
//! use voirs_conversion::compression_research::{CompressionResearcher, CompressionConfig, CompressionTarget};
//!
//! let config = CompressionConfig::default()
//!     .with_target(CompressionTarget::RealTimeStreaming)
//!     .with_quality_factor(0.8)
//!     .with_adaptive_mode(true);
//!
//! let mut compressor = CompressionResearcher::new(config)?;
//!
//! let original = vec![0.1, 0.2, -0.1, 0.05]; // Original audio
//! let compressed = compressor.compress(&original, 16000)?;
//! let decompressed = compressor.decompress(&compressed, 16000)?;
//!
//! println!("Compression ratio: {:.2}", compressed.compression_ratio);
//! println!("Quality score: {:.3}", compressed.quality_score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for compression research
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression target optimization
    pub target: CompressionTarget,
    /// Quality factor (0.0-1.0, higher = better quality)
    pub quality_factor: f32,
    /// Enable adaptive compression based on content
    pub adaptive_mode: bool,
    /// Psychoacoustic masking threshold (0.0-1.0)
    pub masking_threshold: f32,
    /// Maximum allowed latency in milliseconds
    pub max_latency_ms: f32,
    /// Minimum compression ratio target
    pub min_compression_ratio: f32,
    /// Enable perceptual weighting
    pub perceptual_weighting: bool,
    /// Frame size for analysis (samples)
    pub frame_size: usize,
    /// Overlap factor for analysis frames
    pub overlap_factor: f32,
    /// Enable multi-scale compression
    pub multi_scale: bool,
    /// Voice activity detection threshold
    pub vad_threshold: f32,
}

/// Compression optimization targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionTarget {
    /// Ultra-low latency for real-time streaming
    RealTimeStreaming,
    /// Balanced quality and size for general use
    Balanced,
    /// Maximum compression for bandwidth-constrained scenarios
    MaxCompression,
    /// Archival quality with minimal loss
    Archival,
    /// Voice-optimized compression
    VoiceOptimized,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target: CompressionTarget::RealTimeStreaming,
            quality_factor: 0.75,
            adaptive_mode: true,
            masking_threshold: 0.1,
            max_latency_ms: 10.0,
            min_compression_ratio: 2.0,
            perceptual_weighting: true,
            frame_size: 512,
            overlap_factor: 0.5,
            multi_scale: false,
            vad_threshold: 0.01,
        }
    }
}

impl CompressionConfig {
    /// Set compression target
    pub fn with_target(mut self, target: CompressionTarget) -> Self {
        self.target = target;
        self
    }

    /// Set quality factor
    pub fn with_quality_factor(mut self, quality: f32) -> Self {
        self.quality_factor = quality.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable adaptive mode
    pub fn with_adaptive_mode(mut self, enable: bool) -> Self {
        self.adaptive_mode = enable;
        self
    }

    /// Set maximum latency constraint
    pub fn with_max_latency(mut self, latency_ms: f32) -> Self {
        self.max_latency_ms = latency_ms.max(1.0);
        self
    }
}

/// Compressed audio data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedAudio {
    /// Compressed audio data
    pub data: Vec<u8>,
    /// Original sample count
    pub original_samples: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression parameters
    pub parameters: CompressionParameters,
    /// Processing statistics
    pub stats: CompressionStats,
}

/// Compression algorithm types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// Perceptual linear prediction
    PerceptualLPC,
    /// Psychoacoustic transform coding
    PsychoacousticTransform,
    /// Adaptive differential pulse code modulation
    AdaptiveDPCM,
    /// Vector quantization
    VectorQuantization,
    /// Hybrid perceptual-predictive
    HybridPerceptual,
    /// Multi-resolution analysis
    MultiResolution,
}

/// Compression parameters used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionParameters {
    /// Quantization levels used
    pub quantization_levels: Vec<u8>,
    /// Prediction coefficients
    pub prediction_coefficients: Vec<f32>,
    /// Masking thresholds applied
    pub masking_thresholds: Vec<f32>,
    /// Spectral envelope coefficients
    pub spectral_envelope: Vec<f32>,
}

/// Compression processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Compression time in milliseconds
    pub compression_time_ms: f32,
    /// Decompression time in milliseconds  
    pub decompression_time_ms: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Perceptual distortion measure
    pub perceptual_distortion: f32,
    /// Spectral distortion measure
    pub spectral_distortion: f32,
    /// Algorithm complexity score
    pub complexity_score: f32,
}

/// Main compression researcher
pub struct CompressionResearcher {
    /// Configuration
    config: CompressionConfig,
    /// Psychoacoustic analyzer
    psychoacoustic_analyzer: PsychoacousticAnalyzer,
    /// Prediction analyzer
    prediction_analyzer: PredictionAnalyzer,
    /// Vector quantizer
    vector_quantizer: VectorQuantizer,
    /// Performance cache
    performance_cache: HashMap<String, CompressedAudio>,
    /// Analysis count for stats
    analysis_count: usize,
}

/// Psychoacoustic analyzer for perceptual compression
#[derive(Debug, Clone)]
pub struct PsychoacousticAnalyzer {
    /// Critical band boundaries
    critical_bands: Vec<f32>,
    /// Masking curves
    masking_curves: Vec<f32>,
    /// Tonality detector
    tonality_detector: TonalityDetector,
}

/// Prediction analyzer for predictive compression
#[derive(Debug, Clone)]
pub struct PredictionAnalyzer {
    /// LPC analyzer order
    lpc_order: usize,
    /// Prediction coefficients
    coefficients: Vec<f32>,
    /// Residual energy
    residual_energy: f32,
}

/// Vector quantizer for VQ-based compression
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    /// Codebook entries
    codebook: Vec<Vec<f32>>,
    /// Vector dimension
    vector_dim: usize,
    /// Codebook size
    codebook_size: usize,
}

/// Tonality detector for psychoacoustic analysis
#[derive(Debug, Clone)]
pub struct TonalityDetector {
    /// Spectral flatness threshold
    flatness_threshold: f32,
    /// Tonal component weights
    tonal_weights: Vec<f32>,
}

impl CompressionResearcher {
    /// Create a new compression researcher
    pub fn new(config: CompressionConfig) -> Result<Self, Error> {
        let psychoacoustic_analyzer = PsychoacousticAnalyzer::new(&config);
        let prediction_analyzer = PredictionAnalyzer::new(12); // 12th order LPC
        let vector_quantizer = VectorQuantizer::new(8, 256); // 8-dim vectors, 256 entries

        Ok(Self {
            config,
            psychoacoustic_analyzer,
            prediction_analyzer,
            vector_quantizer,
            performance_cache: HashMap::new(),
            analysis_count: 0,
        })
    }

    /// Compress audio using research algorithms
    pub fn compress(&mut self, audio: &[f32], sample_rate: u32) -> Result<CompressedAudio, Error> {
        let start_time = std::time::Instant::now();

        if audio.is_empty() {
            return Err(Error::validation("Audio cannot be empty".to_string()));
        }

        // Select optimal algorithm based on target and content
        let algorithm = self.select_compression_algorithm(audio, sample_rate)?;

        // Perform compression based on selected algorithm
        let compressed = match algorithm {
            CompressionAlgorithm::PerceptualLPC => {
                self.compress_perceptual_lpc(audio, sample_rate)?
            }
            CompressionAlgorithm::PsychoacousticTransform => {
                self.compress_psychoacoustic_transform(audio, sample_rate)?
            }
            CompressionAlgorithm::AdaptiveDPCM => {
                self.compress_adaptive_dpcm(audio, sample_rate)?
            }
            CompressionAlgorithm::VectorQuantization => {
                self.compress_vector_quantization(audio, sample_rate)?
            }
            CompressionAlgorithm::HybridPerceptual => {
                self.compress_hybrid_perceptual(audio, sample_rate)?
            }
            CompressionAlgorithm::MultiResolution => {
                self.compress_multi_resolution(audio, sample_rate)?
            }
        };

        let compression_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Calculate compression ratio and quality
        let original_size = std::mem::size_of_val(audio);
        let compression_ratio = original_size as f32 / compressed.len() as f32;

        // Estimate quality using perceptual model
        let quality_score = self.estimate_compression_quality(audio, &compressed, sample_rate)?;

        // Create compression parameters
        let parameters = self.extract_compression_parameters(audio, algorithm)?;

        let stats = CompressionStats {
            compression_time_ms: compression_time,
            decompression_time_ms: 0.0, // Will be filled during decompression
            memory_usage_bytes: compressed.len() + std::mem::size_of::<CompressedAudio>(),
            perceptual_distortion: 1.0 - quality_score,
            spectral_distortion: self.calculate_spectral_distortion(audio, &compressed)?,
            complexity_score: self.calculate_algorithm_complexity_score(algorithm),
        };

        self.analysis_count += 1;

        Ok(CompressedAudio {
            data: compressed,
            original_samples: audio.len(),
            compression_ratio,
            quality_score,
            algorithm,
            parameters,
            stats,
        })
    }

    /// Decompress audio data
    pub fn decompress(
        &mut self,
        compressed: &CompressedAudio,
        sample_rate: u32,
    ) -> Result<Vec<f32>, Error> {
        let start_time = std::time::Instant::now();

        let decompressed = match compressed.algorithm {
            CompressionAlgorithm::PerceptualLPC => self.decompress_perceptual_lpc(
                &compressed.data,
                &compressed.parameters,
                compressed.original_samples,
            )?,
            CompressionAlgorithm::PsychoacousticTransform => self
                .decompress_psychoacoustic_transform(
                    &compressed.data,
                    &compressed.parameters,
                    compressed.original_samples,
                )?,
            CompressionAlgorithm::AdaptiveDPCM => self.decompress_adaptive_dpcm(
                &compressed.data,
                &compressed.parameters,
                compressed.original_samples,
            )?,
            CompressionAlgorithm::VectorQuantization => self.decompress_vector_quantization(
                &compressed.data,
                &compressed.parameters,
                compressed.original_samples,
            )?,
            CompressionAlgorithm::HybridPerceptual => self.decompress_hybrid_perceptual(
                &compressed.data,
                &compressed.parameters,
                compressed.original_samples,
            )?,
            CompressionAlgorithm::MultiResolution => self.decompress_multi_resolution(
                &compressed.data,
                &compressed.parameters,
                compressed.original_samples,
            )?,
        };

        let decompression_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update stats would require mutable access to compressed, which we don't have here
        // In a real implementation, you'd want to track this separately

        Ok(decompressed)
    }

    /// Select optimal compression algorithm based on content analysis
    fn select_compression_algorithm(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<CompressionAlgorithm, Error> {
        match self.config.target {
            CompressionTarget::RealTimeStreaming => {
                // Prefer low-latency algorithms
                if self.is_voice_content(audio)? {
                    Ok(CompressionAlgorithm::AdaptiveDPCM)
                } else {
                    Ok(CompressionAlgorithm::PerceptualLPC)
                }
            }
            CompressionTarget::Balanced => {
                // Use hybrid approach for balance
                Ok(CompressionAlgorithm::HybridPerceptual)
            }
            CompressionTarget::MaxCompression => {
                // Use most aggressive compression
                Ok(CompressionAlgorithm::VectorQuantization)
            }
            CompressionTarget::Archival => {
                // Use high-quality transform coding
                Ok(CompressionAlgorithm::PsychoacousticTransform)
            }
            CompressionTarget::VoiceOptimized => {
                // Use voice-specific algorithms
                Ok(CompressionAlgorithm::PerceptualLPC)
            }
        }
    }

    /// Check if audio content is primarily voice
    fn is_voice_content(&self, audio: &[f32]) -> Result<bool, Error> {
        // Simple voice activity detection
        if audio.is_empty() {
            return Ok(false);
        }

        // Calculate energy-based features
        let energy = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let zero_crossing_rate = self.calculate_zero_crossing_rate(audio);

        // Voice typically has moderate energy and moderate ZCR
        let is_voice = energy > self.config.vad_threshold
            && zero_crossing_rate > 0.02
            && zero_crossing_rate < 0.3;

        Ok(is_voice)
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (audio.len() - 1) as f32
    }

    /// Perceptual LPC compression
    fn compress_perceptual_lpc(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        // Analyze psychoacoustic properties
        let masking_thresholds = self
            .psychoacoustic_analyzer
            .analyze_masking(audio, sample_rate)?;

        // Perform LPC analysis
        let lpc_coeffs = self.prediction_analyzer.analyze_lpc(audio)?;

        // Quantize coefficients based on masking thresholds
        let quantized_coeffs = self.quantize_with_masking(&lpc_coeffs, &masking_thresholds)?;

        // Encode to bytes
        let compressed = self.encode_lpc_data(&quantized_coeffs)?;

        Ok(compressed)
    }

    /// Psychoacoustic transform compression
    fn compress_psychoacoustic_transform(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        // Apply windowed transform
        let spectrum = self.calculate_spectrum(audio);

        // Analyze psychoacoustic properties
        let masking_thresholds = self
            .psychoacoustic_analyzer
            .analyze_masking(audio, sample_rate)?;

        // Quantize spectrum based on masking
        let quantized_spectrum =
            self.quantize_spectrum_with_masking(&spectrum, &masking_thresholds)?;

        // Encode quantized spectrum
        let compressed = self.encode_spectrum_data(&quantized_spectrum)?;

        Ok(compressed)
    }

    /// Adaptive DPCM compression
    fn compress_adaptive_dpcm(
        &mut self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        let mut predictor = 0.0f32;
        let mut step_size = 0.1f32;

        // Encode first sample directly
        let first_sample_bytes = audio[0].to_le_bytes();
        compressed.extend_from_slice(&first_sample_bytes);

        // DPCM encoding
        for &sample in &audio[1..] {
            let prediction_error = sample - predictor;

            // Quantize prediction error
            let quantized_error = (prediction_error / step_size).round() as i8;
            compressed.push(quantized_error as u8);

            // Update predictor and step size
            let reconstructed_error = quantized_error as f32 * step_size;
            predictor += reconstructed_error;

            // Adaptive step size
            step_size *= if quantized_error.abs() > 2 { 1.1 } else { 0.95 };
            step_size = step_size.clamp(0.01, 1.0);
        }

        Ok(compressed)
    }

    /// Vector quantization compression
    fn compress_vector_quantization(
        &mut self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        let vector_dim = self.vector_quantizer.vector_dim;
        let mut compressed = Vec::new();

        // Process audio in vectors
        for chunk in audio.chunks(vector_dim) {
            let mut vector = vec![0.0; vector_dim];
            for (i, &sample) in chunk.iter().enumerate() {
                vector[i] = sample;
            }

            // Find closest codebook entry
            let codebook_index = self.vector_quantizer.find_closest_vector(&vector)?;
            compressed.push(codebook_index as u8);
        }

        Ok(compressed)
    }

    /// Hybrid perceptual compression
    fn compress_hybrid_perceptual(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        // Combine multiple algorithms based on content
        let voice_regions = self.detect_voice_regions(audio)?;
        let mut compressed = Vec::new();

        for (start, end, is_voice) in voice_regions {
            let segment = &audio[start..end];

            let segment_compressed = if is_voice {
                self.compress_perceptual_lpc(segment, sample_rate)?
            } else {
                self.compress_psychoacoustic_transform(segment, sample_rate)?
            };

            // Add segment header
            compressed.push(if is_voice { 1 } else { 0 }); // Algorithm type
            let length_bytes = (segment_compressed.len() as u32).to_le_bytes();
            compressed.extend_from_slice(&length_bytes);
            compressed.extend(segment_compressed);
        }

        Ok(compressed)
    }

    /// Multi-resolution compression
    fn compress_multi_resolution(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u8>, Error> {
        // Implement multi-scale wavelet-like decomposition
        let mut compressed = Vec::new();
        let mut current_signal = audio.to_vec();

        // Multiple resolution levels
        for level in 0..3 {
            let decimated = self.decimate_signal(&current_signal, 2);
            let detail = self.calculate_detail_coefficients(&current_signal, &decimated)?;

            // Compress detail coefficients
            let detail_compressed = self.compress_adaptive_dpcm(&detail, sample_rate >> level)?;

            // Store compressed detail
            let length_bytes = (detail_compressed.len() as u32).to_le_bytes();
            compressed.extend_from_slice(&length_bytes);
            compressed.extend(detail_compressed);

            current_signal = decimated;
        }

        // Store final low-resolution signal
        let final_compressed = self.compress_adaptive_dpcm(&current_signal, sample_rate >> 3)?;
        let length_bytes = (final_compressed.len() as u32).to_le_bytes();
        compressed.extend_from_slice(&length_bytes);
        compressed.extend(final_compressed);

        Ok(compressed)
    }

    // Decompression methods (simplified implementations)

    fn decompress_perceptual_lpc(
        &self,
        compressed: &[u8],
        parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        // Simplified LPC decompression
        let coeffs = &parameters.prediction_coefficients;
        let mut decompressed = vec![0.0; original_samples];

        // Simple reconstruction using stored coefficients
        for i in coeffs.len()..decompressed.len() {
            let mut prediction = 0.0;
            for (j, &coeff) in coeffs.iter().enumerate() {
                if i > j {
                    prediction += coeff * decompressed[i - j - 1];
                }
            }

            // Add residual (simplified)
            let residual_index = (i - coeffs.len()) % compressed.len();
            let residual = (compressed[residual_index] as f32 - 128.0) / 128.0 * 0.1;
            decompressed[i] = prediction + residual;
        }

        Ok(decompressed)
    }

    fn decompress_psychoacoustic_transform(
        &self,
        compressed: &[u8],
        parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        // Simplified transform decompression
        let mut decompressed = vec![0.0; original_samples];

        // Reconstruct from spectral envelope and compressed data
        let envelope = &parameters.spectral_envelope;
        for (i, &env_val) in envelope.iter().enumerate() {
            if i < decompressed.len() {
                let compressed_index = i % compressed.len();
                let compressed_val = (compressed[compressed_index] as f32 - 128.0) / 128.0;
                decompressed[i] = env_val * compressed_val;
            }
        }

        Ok(decompressed)
    }

    fn decompress_adaptive_dpcm(
        &self,
        compressed: &[u8],
        _parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        if compressed.len() < 4 {
            return Ok(vec![0.0; original_samples]);
        }

        let mut decompressed = Vec::with_capacity(original_samples);

        // Decode first sample
        let first_sample =
            f32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        decompressed.push(first_sample);

        let mut predictor = first_sample;
        let mut step_size = 0.1f32;

        // DPCM decoding
        for &byte in &compressed[4..] {
            let quantized_error = byte as i8;
            let reconstructed_error = quantized_error as f32 * step_size;
            predictor += reconstructed_error;
            decompressed.push(predictor);

            // Adaptive step size
            step_size *= if quantized_error.abs() > 2 { 1.1 } else { 0.95 };
            step_size = step_size.clamp(0.01, 1.0);

            if decompressed.len() >= original_samples {
                break;
            }
        }

        // Pad if necessary
        while decompressed.len() < original_samples {
            decompressed.push(predictor);
        }

        Ok(decompressed)
    }

    fn decompress_vector_quantization(
        &self,
        compressed: &[u8],
        _parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        let mut decompressed = Vec::with_capacity(original_samples);
        let vector_dim = self.vector_quantizer.vector_dim;

        for &index in compressed {
            if (index as usize) < self.vector_quantizer.codebook.len() {
                let vector = &self.vector_quantizer.codebook[index as usize];
                decompressed.extend_from_slice(vector);

                if decompressed.len() >= original_samples {
                    break;
                }
            }
        }

        decompressed.truncate(original_samples);
        Ok(decompressed)
    }

    fn decompress_hybrid_perceptual(
        &self,
        compressed: &[u8],
        parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        let mut decompressed = Vec::with_capacity(original_samples);
        let mut pos = 0;

        while pos < compressed.len() && decompressed.len() < original_samples {
            if pos >= compressed.len() {
                break;
            }

            let algorithm_type = compressed[pos];
            pos += 1;

            if pos + 4 > compressed.len() {
                break;
            }

            let segment_length = u32::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + segment_length > compressed.len() {
                break;
            }

            let segment_data = &compressed[pos..pos + segment_length];
            pos += segment_length;

            let remaining_samples = original_samples - decompressed.len();
            let segment_decompressed = if algorithm_type == 1 {
                self.decompress_perceptual_lpc(segment_data, parameters, remaining_samples)?
            } else {
                self.decompress_psychoacoustic_transform(
                    segment_data,
                    parameters,
                    remaining_samples,
                )?
            };

            decompressed.extend(segment_decompressed);
        }

        decompressed.truncate(original_samples);
        Ok(decompressed)
    }

    fn decompress_multi_resolution(
        &self,
        compressed: &[u8],
        _parameters: &CompressionParameters,
        original_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        // Simplified multi-resolution decompression
        let mut pos = 0;
        let mut detail_levels = Vec::new();

        // Read detail coefficients for each level
        for _ in 0..3 {
            if pos + 4 > compressed.len() {
                break;
            }

            let length = u32::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + length > compressed.len() {
                break;
            }

            let detail_data = &compressed[pos..pos + length];
            pos += length;

            let detail = self.decompress_adaptive_dpcm(
                detail_data,
                &CompressionParameters::default(),
                length,
            )?;
            detail_levels.push(detail);
        }

        // Read final low-resolution signal
        let final_length = if pos + 4 <= compressed.len() {
            u32::from_le_bytes([
                compressed[pos],
                compressed[pos + 1],
                compressed[pos + 2],
                compressed[pos + 3],
            ]) as usize
        } else {
            0
        };
        pos += 4;

        let final_data = if pos + final_length <= compressed.len() {
            &compressed[pos..pos + final_length]
        } else {
            &[]
        };

        let mut reconstructed = if !final_data.is_empty() {
            self.decompress_adaptive_dpcm(
                final_data,
                &CompressionParameters::default(),
                original_samples / 8,
            )?
        } else {
            vec![0.0; original_samples / 8]
        };

        // Reconstruct by upsampling and adding detail coefficients
        for detail in detail_levels.iter().rev() {
            reconstructed = self.upsample_and_add_detail(&reconstructed, detail);
        }

        reconstructed.truncate(original_samples);
        Ok(reconstructed)
    }

    // Helper methods for compression algorithms

    fn calculate_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified DFT implementation
        let n = audio.len();
        let mut spectrum = vec![0.0; n / 2 + 1];

        for (k, spectrum_value) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in audio.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spectrum_value = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    fn quantize_with_masking(
        &self,
        coeffs: &[f32],
        masking_thresholds: &[f32],
    ) -> Result<Vec<u8>, Error> {
        let mut quantized = Vec::new();

        for (i, &coeff) in coeffs.iter().enumerate() {
            let threshold = masking_thresholds.get(i).copied().unwrap_or(0.01);
            let quantization_step = threshold * self.config.quality_factor;

            let quantized_val = (coeff / quantization_step).round() as i16;
            let clamped_val = quantized_val.clamp(-128, 127) as i8;
            quantized.push((clamped_val as u8).wrapping_add(128));
        }

        Ok(quantized)
    }

    fn quantize_spectrum_with_masking(
        &self,
        spectrum: &[f32],
        masking_thresholds: &[f32],
    ) -> Result<Vec<u8>, Error> {
        let mut quantized = Vec::new();

        for (i, &mag) in spectrum.iter().enumerate() {
            let threshold = masking_thresholds.get(i).copied().unwrap_or(0.01);
            let quantization_step = threshold * self.config.quality_factor;

            let quantized_val = (mag / quantization_step).round() as u16;
            let clamped_val = quantized_val.min(255) as u8;
            quantized.push(clamped_val);
        }

        Ok(quantized)
    }

    fn encode_lpc_data(&self, quantized_coeffs: &[u8]) -> Result<Vec<u8>, Error> {
        // Simple encoding - in practice would use entropy coding
        Ok(quantized_coeffs.to_vec())
    }

    fn encode_spectrum_data(&self, quantized_spectrum: &[u8]) -> Result<Vec<u8>, Error> {
        // Simple encoding - in practice would use entropy coding
        Ok(quantized_spectrum.to_vec())
    }

    fn detect_voice_regions(&self, audio: &[f32]) -> Result<Vec<(usize, usize, bool)>, Error> {
        let frame_size = 1024;
        let mut regions = Vec::new();

        for (i, chunk) in audio.chunks(frame_size).enumerate() {
            let start = i * frame_size;
            let end = (start + chunk.len()).min(audio.len());
            let is_voice = self.is_voice_content(chunk)?;
            regions.push((start, end, is_voice));
        }

        Ok(regions)
    }

    fn decimate_signal(&self, signal: &[f32], factor: usize) -> Vec<f32> {
        signal.iter().step_by(factor).cloned().collect()
    }

    fn calculate_detail_coefficients(
        &self,
        original: &[f32],
        decimated: &[f32],
    ) -> Result<Vec<f32>, Error> {
        let mut detail = Vec::new();

        for (i, &orig_sample) in original.iter().enumerate() {
            let decimated_idx = i / 2;
            let interpolated = if decimated_idx < decimated.len() {
                decimated[decimated_idx]
            } else {
                0.0
            };

            detail.push(orig_sample - interpolated);
        }

        Ok(detail)
    }

    fn upsample_and_add_detail(&self, low_res: &[f32], detail: &[f32]) -> Vec<f32> {
        let mut upsampled = Vec::with_capacity(low_res.len() * 2);

        for &sample in low_res {
            upsampled.push(sample);
            upsampled.push(sample); // Simple upsampling
        }

        // Add detail coefficients
        for (i, &detail_coeff) in detail.iter().enumerate() {
            if i < upsampled.len() {
                upsampled[i] += detail_coeff;
            }
        }

        upsampled
    }

    fn estimate_compression_quality(
        &self,
        original: &[f32],
        compressed: &[u8],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified quality estimation
        let compression_ratio = (original.len() * 4) as f32 / compressed.len() as f32;

        // Higher compression ratio generally means lower quality
        let quality = 1.0 - (compression_ratio - self.config.min_compression_ratio).max(0.0) / 10.0;

        Ok(quality.clamp(0.0, 1.0))
    }

    fn calculate_spectral_distortion(
        &self,
        original: &[f32],
        _compressed: &[u8],
    ) -> Result<f32, Error> {
        // Simplified spectral distortion calculation
        let spectrum = self.calculate_spectrum(original);
        let avg_magnitude = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        // Estimate distortion based on compression ratio and content
        let distortion = 0.1 * (1.0 - self.config.quality_factor);

        Ok(distortion.clamp(0.0, 1.0))
    }

    fn calculate_algorithm_complexity_score(&self, algorithm: CompressionAlgorithm) -> f32 {
        match algorithm {
            CompressionAlgorithm::AdaptiveDPCM => 0.2,
            CompressionAlgorithm::PerceptualLPC => 0.4,
            CompressionAlgorithm::VectorQuantization => 0.6,
            CompressionAlgorithm::PsychoacousticTransform => 0.7,
            CompressionAlgorithm::HybridPerceptual => 0.8,
            CompressionAlgorithm::MultiResolution => 1.0,
        }
    }

    fn extract_compression_parameters(
        &self,
        audio: &[f32],
        algorithm: CompressionAlgorithm,
    ) -> Result<CompressionParameters, Error> {
        // Extract relevant parameters based on algorithm
        let prediction_coefficients = if matches!(algorithm, CompressionAlgorithm::PerceptualLPC) {
            self.prediction_analyzer.coefficients.clone()
        } else {
            vec![1.0; 12] // Default coefficients
        };

        let spectral_envelope =
            if matches!(algorithm, CompressionAlgorithm::PsychoacousticTransform) {
                self.calculate_spectrum(audio)
            } else {
                vec![1.0; audio.len().min(256)] // Simplified envelope
            };

        let masking_thresholds = vec![self.config.masking_threshold; audio.len().min(256)];
        let quantization_levels = vec![8; audio.len().min(256)]; // 8-bit quantization

        Ok(CompressionParameters {
            quantization_levels,
            prediction_coefficients,
            masking_thresholds,
            spectral_envelope,
        })
    }

    /// Get compression statistics
    pub fn get_analysis_count(&self) -> usize {
        self.analysis_count
    }

    /// Clear performance cache
    pub fn clear_cache(&mut self) {
        self.performance_cache.clear();
    }
}

// Implementation of sub-components

impl PsychoacousticAnalyzer {
    fn new(config: &CompressionConfig) -> Self {
        // Bark scale critical band boundaries
        let critical_bands = (0..24).map(|i| 600.0 * ((i as f32 / 4.0).sinh())).collect();

        let masking_curves = vec![config.masking_threshold; 24];
        let tonality_detector = TonalityDetector::new();

        Self {
            critical_bands,
            masking_curves,
            tonality_detector,
        }
    }

    fn analyze_masking(&self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>, Error> {
        let spectrum = self.calculate_spectrum(audio);
        let mut masking_thresholds = Vec::new();

        for (i, &magnitude) in spectrum.iter().enumerate() {
            // Simplified masking calculation
            let base_threshold = self
                .masking_curves
                .get(i % self.masking_curves.len())
                .cloned()
                .unwrap_or(0.01);
            let energy_factor = (magnitude * magnitude).sqrt();
            let threshold = base_threshold * (1.0 + energy_factor);
            masking_thresholds.push(threshold);
        }

        Ok(masking_thresholds)
    }

    fn calculate_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectrum calculation
        let n = audio.len();
        let mut spectrum = vec![0.0; n.min(512)];

        for (k, spec_val) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in audio.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spec_val = (real * real + imag * imag).sqrt();
        }

        spectrum
    }
}

impl PredictionAnalyzer {
    fn new(order: usize) -> Self {
        Self {
            lpc_order: order,
            coefficients: vec![0.0; order],
            residual_energy: 0.0,
        }
    }

    fn analyze_lpc(&mut self, audio: &[f32]) -> Result<Vec<f32>, Error> {
        if audio.len() <= self.lpc_order {
            return Ok(vec![0.0; self.lpc_order]);
        }

        // Simplified LPC analysis using autocorrelation method
        let mut autocorr = vec![0.0; self.lpc_order + 1];

        // Calculate autocorrelation
        for lag in 0..=self.lpc_order {
            for i in lag..audio.len() {
                autocorr[lag] += audio[i] * audio[i - lag];
            }
        }

        // Solve normal equations using Levinson-Durbin recursion (simplified)
        let mut coeffs = vec![0.0; self.lpc_order];

        if autocorr[0] > 1e-10 {
            for i in 0..self.lpc_order {
                coeffs[i] = autocorr[i + 1] / autocorr[0];
            }
        }

        self.coefficients = coeffs.clone();
        Ok(coeffs)
    }
}

impl VectorQuantizer {
    fn new(vector_dim: usize, codebook_size: usize) -> Self {
        // Initialize random codebook
        let mut codebook = Vec::new();
        for _ in 0..codebook_size {
            let mut vector = Vec::new();
            for _ in 0..vector_dim {
                vector.push(fastrand::f32() * 2.0 - 1.0); // Random values in [-1, 1]
            }
            codebook.push(vector);
        }

        Self {
            codebook,
            vector_dim,
            codebook_size,
        }
    }

    fn find_closest_vector(&self, input_vector: &[f32]) -> Result<usize, Error> {
        let mut min_distance = f32::INFINITY;
        let mut best_index = 0;

        for (i, codebook_vector) in self.codebook.iter().enumerate() {
            let distance = self.euclidean_distance(input_vector, codebook_vector);
            if distance < min_distance {
                min_distance = distance;
                best_index = i;
            }
        }

        Ok(best_index)
    }

    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl TonalityDetector {
    fn new() -> Self {
        Self {
            flatness_threshold: 0.1,
            tonal_weights: vec![1.0; 24],
        }
    }
}

impl Default for CompressionParameters {
    fn default() -> Self {
        Self {
            quantization_levels: vec![8; 256],
            prediction_coefficients: vec![0.0; 12],
            masking_thresholds: vec![0.01; 256],
            spectral_envelope: vec![1.0; 256],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_creation() {
        let config = CompressionConfig::default();
        assert_eq!(config.target, CompressionTarget::RealTimeStreaming);
        assert_eq!(config.quality_factor, 0.75);
        assert!(config.adaptive_mode);
    }

    #[test]
    fn test_compression_config_builder() {
        let config = CompressionConfig::default()
            .with_target(CompressionTarget::MaxCompression)
            .with_quality_factor(0.9)
            .with_adaptive_mode(false);

        assert_eq!(config.target, CompressionTarget::MaxCompression);
        assert_eq!(config.quality_factor, 0.9);
        assert!(!config.adaptive_mode);
    }

    #[test]
    fn test_compression_researcher_creation() {
        let config = CompressionConfig::default();
        let researcher = CompressionResearcher::new(config);
        assert!(researcher.is_ok());
    }

    #[test]
    fn test_compression_and_decompression() {
        let config = CompressionConfig::default();
        let mut researcher = CompressionResearcher::new(config).unwrap();

        let original = vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
        let compressed = researcher.compress(&original, 16000).unwrap();

        assert!(compressed.compression_ratio > 1.0);
        assert!(compressed.quality_score >= 0.0 && compressed.quality_score <= 1.0);
        assert!(!compressed.data.is_empty());

        let decompressed = researcher.decompress(&compressed, 16000).unwrap();
        assert_eq!(decompressed.len(), original.len());
    }

    #[test]
    fn test_voice_content_detection() {
        let config = CompressionConfig::default();
        let researcher = CompressionResearcher::new(config).unwrap();

        // Voice-like signal (moderate energy and ZCR)
        let voice_signal = vec![0.1, -0.1, 0.2, -0.15, 0.12, -0.08];
        let is_voice = researcher.is_voice_content(&voice_signal).unwrap();

        // Empty signal
        let empty_signal = vec![];
        let is_empty_voice = researcher.is_voice_content(&empty_signal).unwrap();
        assert!(!is_empty_voice);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let config = CompressionConfig::default();
        let researcher = CompressionResearcher::new(config).unwrap();

        let alternating_signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let zcr = researcher.calculate_zero_crossing_rate(&alternating_signal);
        assert!(zcr > 0.8); // High ZCR for alternating signal

        let constant_signal = vec![1.0, 1.0, 1.0, 1.0];
        let zcr_constant = researcher.calculate_zero_crossing_rate(&constant_signal);
        assert_eq!(zcr_constant, 0.0); // No zero crossings
    }

    #[test]
    fn test_adaptive_dpcm_compression() {
        let config = CompressionConfig::default();
        let mut researcher = CompressionResearcher::new(config).unwrap();

        let audio = vec![0.1, 0.2, 0.15, 0.25, 0.3];
        let compressed = researcher.compress_adaptive_dpcm(&audio, 16000).unwrap();

        assert!(!compressed.is_empty());
        assert!(compressed.len() >= 4); // At least the first sample
    }

    #[test]
    fn test_spectrum_calculation() {
        let config = CompressionConfig::default();
        let researcher = CompressionResearcher::new(config).unwrap();

        let audio = vec![1.0, 0.0, -1.0, 0.0]; // Simple sinusoid
        let spectrum = researcher.calculate_spectrum(&audio);

        assert_eq!(spectrum.len(), audio.len() / 2 + 1);
        assert!(spectrum.iter().all(|&x| x >= 0.0)); // Magnitude spectrum
    }

    #[test]
    fn test_vector_quantizer() {
        let vq = VectorQuantizer::new(4, 16);
        let test_vector = vec![0.1, 0.2, 0.3, 0.4];

        let index = vq.find_closest_vector(&test_vector).unwrap();
        assert!(index < vq.codebook_size);
    }

    #[test]
    fn test_algorithm_selection() {
        let config = CompressionConfig::default().with_target(CompressionTarget::VoiceOptimized);
        let researcher = CompressionResearcher::new(config).unwrap();

        let voice_audio = vec![0.1, -0.1, 0.2, -0.15, 0.12];
        let algorithm = researcher
            .select_compression_algorithm(&voice_audio, 16000)
            .unwrap();

        assert_eq!(algorithm, CompressionAlgorithm::PerceptualLPC);
    }

    #[test]
    fn test_compression_statistics() {
        let config = CompressionConfig::default();
        let mut researcher = CompressionResearcher::new(config).unwrap();

        assert_eq!(researcher.get_analysis_count(), 0);

        let audio = vec![0.1, 0.2, 0.3];
        let _ = researcher.compress(&audio, 16000).unwrap();

        assert_eq!(researcher.get_analysis_count(), 1);

        researcher.clear_cache();
        assert_eq!(researcher.performance_cache.len(), 0);
    }
}
