//! Spatial Audio Compression for efficient transmission and storage
//!
//! This module provides compression algorithms specifically designed for spatial audio
//! that preserve spatial characteristics while reducing data size for network transmission
//! and storage applications. It supports various compression schemes including
//! perceptually-guided compression and ambisonics-aware compression.

use crate::types::Position3D;
use crate::{Error, Result};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::Complex32;
use scirs2_fft::{ComplexToReal, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Canonical Huffman entropy coder used by the entropy-coding stage.
mod huffman;

/// Spatial audio compression codec types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionCodec {
    /// Perceptual spatial audio codec
    PerceptualSpatial,
    /// Ambisonics-aware compression
    AmbisonicsOptimized,
    /// Position-based compression
    PositionalCompression,
    /// Hybrid compression (combines multiple methods)
    Hybrid,
    /// Lossless compression for archival
    Lossless,
}

/// Compression quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionQuality {
    /// Minimum quality, maximum compression
    Low,
    /// Balanced quality and compression
    Medium,
    /// High quality, moderate compression
    High,
    /// Maximum quality, minimum compression
    VeryHigh,
}

/// Spatial compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialCompressionConfig {
    /// Compression codec to use
    pub codec: CompressionCodec,
    /// Quality level
    pub quality: CompressionQuality,
    /// Target bitrate (bits per second)
    pub target_bitrate: u32,
    /// Sample rate
    pub sample_rate: f32,
    /// Number of spatial channels
    pub channel_count: usize,
    /// Perceptual masking parameters
    pub perceptual_params: PerceptualParams,
    /// Spatial parameters
    pub spatial_params: SpatialParams,
    /// Adaptive encoding parameters
    pub adaptive_params: AdaptiveParams,
}

/// Perceptual masking parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualParams {
    /// Enable perceptual masking
    pub masking_enabled: bool,
    /// Frequency resolution for masking (number of bands)
    pub frequency_bands: usize,
    /// Spatial masking threshold
    pub spatial_masking_threshold: f32,
    /// Temporal masking parameters
    pub temporal_masking: TemporalMasking,
    /// Loudness compensation
    pub loudness_compensation: bool,
}

/// Temporal masking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMasking {
    /// Enable temporal masking
    pub enabled: bool,
    /// Pre-masking duration (ms)
    pub pre_masking_ms: f32,
    /// Post-masking duration (ms)
    pub post_masking_ms: f32,
    /// Masking threshold
    pub threshold_db: f32,
}

/// Spatial compression parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialParams {
    /// Maximum spatial resolution (degrees)
    pub spatial_resolution: f32,
    /// Distance quantization levels
    pub distance_quantization: usize,
    /// Ambisonics order (if applicable)
    pub ambisonics_order: usize,
    /// Source clustering for position-based compression
    pub source_clustering: SourceClustering,
}

/// Source clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceClustering {
    /// Enable source clustering
    pub enabled: bool,
    /// Maximum cluster distance (meters)
    pub max_cluster_distance: f32,
    /// Maximum sources per cluster
    pub max_sources_per_cluster: usize,
    /// Cluster update interval (ms)
    pub update_interval_ms: f32,
}

/// Adaptive encoding parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Enable adaptive bitrate
    pub adaptive_bitrate: bool,
    /// Minimum bitrate (bits per second)
    pub min_bitrate: u32,
    /// Maximum bitrate (bits per second)
    pub max_bitrate: u32,
    /// Adaptation window (seconds)
    pub adaptation_window: f32,
    /// Quality threshold for adaptation
    pub quality_threshold: f32,
}

/// Compressed spatial audio frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedFrame {
    /// Compressed audio data
    pub audio_data: Vec<u8>,
    /// Spatial metadata
    pub spatial_metadata: SpatialMetadata,
    /// Compression statistics
    pub compression_stats: CompressionStats,
    /// Frame timestamp
    pub timestamp_ms: f64,
}

/// Spatial metadata for compressed frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMetadata {
    /// Source positions
    pub source_positions: Vec<Position3D>,
    /// Ambisonics coefficients (if applicable)
    pub ambisonics_coefficients: Vec<f32>,
    /// Spatial covariance matrix (compressed)
    pub spatial_covariance: Vec<f32>,
    /// Distance attenuation factors
    pub distance_factors: Vec<f32>,
    /// Listener orientation
    pub listener_orientation: (f32, f32, f32), // yaw, pitch, roll
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Original size (bytes)
    pub original_size: usize,
    /// Compressed size (bytes)
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Achieved bitrate
    pub achieved_bitrate: f32,
    /// Estimated quality loss (0.0 = lossless, 1.0 = maximum loss)
    pub quality_loss: f32,
    /// Processing time (ms)
    pub processing_time_ms: f32,
}

/// Spatial audio compressor
pub struct SpatialCompressor {
    /// Configuration
    config: SpatialCompressionConfig,
    /// Perceptual model
    perceptual_model: PerceptualModel,
    /// Spatial encoder
    spatial_encoder: SpatialEncoder,
    /// Adaptive controller
    adaptive_controller: AdaptiveController,
    /// Compression buffers
    input_buffer: Array2<f32>,
    output_buffer: Vec<u8>,
    /// Frame counter
    frame_count: u64,
}

/// Perceptual model for masking
#[derive(Debug)]
struct PerceptualModel {
    /// Frequency bands
    frequency_bands: Array1<f32>,
    /// Masking thresholds
    masking_thresholds: Array1<f32>,
    /// Bark scale coefficients
    bark_scale: Array1<f32>,
    /// Temporal masking state
    temporal_state: TemporalMaskingState,
}

/// Temporal masking state
#[derive(Debug)]
struct TemporalMaskingState {
    /// Previous frame energy
    prev_energy: Array1<f32>,
    /// Pre-masking buffer
    pre_masking_buffer: Array2<f32>,
    /// Post-masking buffer
    post_masking_buffer: Array2<f32>,
}

/// Spatial encoder for different compression methods
#[derive(Debug)]
struct SpatialEncoder {
    /// Current encoding method
    method: CompressionCodec,
    /// Quantization tables
    quantization_tables: HashMap<String, Array1<f32>>,
    /// Huffman coding tables
    huffman_tables: HashMap<String, Vec<(u8, Vec<bool>)>>,
    /// Source clusters
    source_clusters: Vec<SourceCluster>,
}

/// Source cluster for position-based compression
#[derive(Debug, Clone)]
struct SourceCluster {
    /// Cluster center
    center: Position3D,
    /// Source indices in this cluster
    source_indices: Vec<usize>,
    /// Representative audio signal
    representative_signal: Array1<f32>,
    /// Mixing weights for sources in cluster
    mixing_weights: Array1<f32>,
}

/// Adaptive bitrate controller
#[derive(Debug)]
struct AdaptiveController {
    /// Current target bitrate
    current_bitrate: u32,
    /// Quality history
    quality_history: Vec<f32>,
    /// Bitrate history
    bitrate_history: Vec<u32>,
    /// Adaptation window samples
    window_samples: usize,
}

impl Default for SpatialCompressionConfig {
    fn default() -> Self {
        Self {
            codec: CompressionCodec::PerceptualSpatial,
            quality: CompressionQuality::Medium,
            target_bitrate: 128000, // 128 kbps
            sample_rate: 48000.0,
            channel_count: 8,
            perceptual_params: PerceptualParams {
                masking_enabled: true,
                frequency_bands: 32,
                spatial_masking_threshold: -40.0,
                temporal_masking: TemporalMasking {
                    enabled: true,
                    pre_masking_ms: 2.0,
                    post_masking_ms: 100.0,
                    threshold_db: -20.0,
                },
                loudness_compensation: true,
            },
            spatial_params: SpatialParams {
                spatial_resolution: 5.0, // 5 degrees
                distance_quantization: 32,
                ambisonics_order: 3,
                source_clustering: SourceClustering {
                    enabled: true,
                    max_cluster_distance: 1.0,
                    max_sources_per_cluster: 4,
                    update_interval_ms: 100.0,
                },
            },
            adaptive_params: AdaptiveParams {
                adaptive_bitrate: true,
                min_bitrate: 64000,
                max_bitrate: 320000,
                adaptation_window: 5.0,
                quality_threshold: 0.85,
            },
        }
    }
}

impl SpatialCompressor {
    /// Create a new spatial compressor
    pub fn new(config: SpatialCompressionConfig) -> Result<Self> {
        let perceptual_model = PerceptualModel::new(&config.perceptual_params, config.sample_rate)?;
        let spatial_encoder = SpatialEncoder::new(&config)?;
        let adaptive_controller = AdaptiveController::new(&config.adaptive_params)?;

        let buffer_size = 1024; // Default frame size
        let input_buffer = Array2::zeros((config.channel_count, buffer_size));
        let output_buffer = Vec::with_capacity(buffer_size * config.channel_count);

        Ok(Self {
            config,
            perceptual_model,
            spatial_encoder,
            adaptive_controller,
            input_buffer,
            output_buffer,
            frame_count: 0,
        })
    }

    /// Compress a frame of spatial audio
    pub fn compress_frame(
        &mut self,
        audio_data: &Array2<f32>,
        spatial_metadata: &SpatialMetadata,
    ) -> Result<CompressedFrame> {
        let start_time = std::time::Instant::now();

        if audio_data.nrows() != self.config.channel_count {
            return Err(Error::LegacyProcessing(format!(
                "Expected {} channels, got {}",
                self.config.channel_count,
                audio_data.nrows()
            )));
        }

        // Update adaptive controller
        if self.config.adaptive_params.adaptive_bitrate {
            self.adaptive_controller.update(&self.config)?;
        }

        // Apply perceptual masking
        let masked_audio = self.apply_perceptual_masking(audio_data)?;

        // Compress spatial audio based on codec type
        let compressed_audio = match self.config.codec {
            CompressionCodec::PerceptualSpatial => {
                self.compress_perceptual_spatial(&masked_audio, spatial_metadata)?
            }
            CompressionCodec::AmbisonicsOptimized => {
                self.compress_ambisonics_optimized(&masked_audio, spatial_metadata)?
            }
            CompressionCodec::PositionalCompression => {
                self.compress_positional(&masked_audio, spatial_metadata)?
            }
            CompressionCodec::Hybrid => self.compress_hybrid(&masked_audio, spatial_metadata)?,
            CompressionCodec::Lossless => {
                self.compress_lossless(&masked_audio, spatial_metadata)?
            }
        };

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Calculate compression statistics
        let original_size = audio_data.len() * std::mem::size_of::<f32>();
        let compressed_size = compressed_audio.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let achieved_bitrate =
            (compressed_size as f32 * 8.0 * self.config.sample_rate) / audio_data.ncols() as f32;

        let compression_stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio,
            achieved_bitrate,
            quality_loss: self.estimate_quality_loss(&masked_audio, &compressed_audio)?,
            processing_time_ms: processing_time,
        };

        self.frame_count += 1;

        Ok(CompressedFrame {
            audio_data: compressed_audio,
            spatial_metadata: spatial_metadata.clone(),
            compression_stats,
            timestamp_ms: self.frame_count as f64 * 1000.0 * audio_data.ncols() as f64
                / self.config.sample_rate as f64,
        })
    }

    /// Apply perceptual masking to reduce data before compression
    fn apply_perceptual_masking(&mut self, audio_data: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.perceptual_params.masking_enabled {
            return Ok(audio_data.clone());
        }

        let mut masked_audio = audio_data.clone();

        // Apply frequency-domain masking
        for channel_idx in 0..audio_data.nrows() {
            let channel_data = audio_data.row(channel_idx).to_owned();
            let masked_channel = self.perceptual_model.apply_masking(&channel_data)?;
            masked_audio.row_mut(channel_idx).assign(&masked_channel);
        }

        // Apply temporal masking
        if self.config.perceptual_params.temporal_masking.enabled {
            self.perceptual_model
                .apply_temporal_masking(&mut masked_audio)?;
        }

        Ok(masked_audio)
    }

    /// Compress using perceptual spatial method
    fn compress_perceptual_spatial(
        &mut self,
        audio_data: &Array2<f32>,
        _spatial_metadata: &SpatialMetadata,
    ) -> Result<Vec<u8>> {
        // Simplified perceptual spatial compression
        let mut compressed = Vec::new();

        // Transform to frequency domain and quantize
        for channel in audio_data.rows() {
            let channel_owned = channel.to_owned();
            let quantized = self.quantize_channel(&channel_owned, self.config.quality)?;
            compressed.extend_from_slice(&quantized);
        }

        // Apply entropy coding
        self.apply_entropy_coding(&compressed)
    }

    /// Compress using ambisonics-optimized method
    fn compress_ambisonics_optimized(
        &mut self,
        audio_data: &Array2<f32>,
        spatial_metadata: &SpatialMetadata,
    ) -> Result<Vec<u8>> {
        // Convert to ambisonics representation if not already
        let ambisonics_data = self.convert_to_ambisonics(audio_data, spatial_metadata)?;

        // Apply hierarchical quantization (lower orders get more bits)
        let mut compressed = Vec::new();
        let order = self.config.spatial_params.ambisonics_order;

        for (idx, channel) in ambisonics_data.rows().into_iter().enumerate() {
            let channel_order = self.get_ambisonics_channel_order(idx);
            let quality_factor = if channel_order == 0 {
                1.0
            } else {
                0.7 / channel_order as f32
            };

            let adjusted_quality = match self.config.quality {
                CompressionQuality::Low => CompressionQuality::Low,
                CompressionQuality::Medium => {
                    if quality_factor > 0.5 {
                        CompressionQuality::Medium
                    } else {
                        CompressionQuality::Low
                    }
                }
                CompressionQuality::High => {
                    if quality_factor > 0.7 {
                        CompressionQuality::High
                    } else {
                        CompressionQuality::Medium
                    }
                }
                CompressionQuality::VeryHigh => CompressionQuality::High,
            };

            let channel_owned = channel.to_owned();
            let quantized = self.quantize_channel(&channel_owned, adjusted_quality)?;
            compressed.extend_from_slice(&quantized);
        }

        self.apply_entropy_coding(&compressed)
    }

    /// Compress using positional method
    fn compress_positional(
        &mut self,
        audio_data: &Array2<f32>,
        spatial_metadata: &SpatialMetadata,
    ) -> Result<Vec<u8>> {
        // Update source clusters
        self.spatial_encoder
            .update_clusters(&spatial_metadata.source_positions)?;

        let mut compressed = Vec::new();

        // Compress based on spatial clustering
        for cluster in &self.spatial_encoder.source_clusters {
            // Compress representative signal for cluster
            let quantized =
                self.quantize_channel(&cluster.representative_signal, self.config.quality)?;
            compressed.extend_from_slice(&quantized);

            // Compress mixing weights (these can be heavily quantized)
            let weight_bytes = self.quantize_weights(&cluster.mixing_weights)?;
            compressed.extend_from_slice(&weight_bytes);
        }

        // Compress cluster metadata
        let cluster_metadata = self.compress_cluster_metadata()?;
        compressed.extend_from_slice(&cluster_metadata);

        self.apply_entropy_coding(&compressed)
    }

    /// Compress using hybrid method
    fn compress_hybrid(
        &mut self,
        audio_data: &Array2<f32>,
        spatial_metadata: &SpatialMetadata,
    ) -> Result<Vec<u8>> {
        // Use different methods for different frequency ranges
        let mut compressed = Vec::new();

        // Low frequencies: use perceptual spatial
        let low_freq_data = self.filter_frequency_range(audio_data, 0.0, 1000.0)?;
        let low_compressed = self.compress_perceptual_spatial(&low_freq_data, spatial_metadata)?;
        compressed.extend_from_slice(&low_compressed);

        // Mid frequencies: use ambisonics optimized
        let mid_freq_data = self.filter_frequency_range(audio_data, 1000.0, 8000.0)?;
        let mid_compressed =
            self.compress_ambisonics_optimized(&mid_freq_data, spatial_metadata)?;
        compressed.extend_from_slice(&mid_compressed);

        // High frequencies: use positional
        let high_freq_data = self.filter_frequency_range(audio_data, 8000.0, 20000.0)?;
        let high_compressed = self.compress_positional(&high_freq_data, spatial_metadata)?;
        compressed.extend_from_slice(&high_compressed);

        Ok(compressed)
    }

    /// Lossless compression
    fn compress_lossless(
        &mut self,
        audio_data: &Array2<f32>,
        _spatial_metadata: &SpatialMetadata,
    ) -> Result<Vec<u8>> {
        // Convert to bytes and apply lossless compression (simplified)
        let mut data_bytes = Vec::new();
        for &sample in audio_data.iter() {
            data_bytes.extend_from_slice(&sample.to_le_bytes());
        }

        // Apply simple RLE or similar lossless compression
        self.apply_lossless_compression(&data_bytes)
    }

    /// Quantize audio channel based on quality level
    fn quantize_channel(
        &self,
        channel_data: &Array1<f32>,
        quality: CompressionQuality,
    ) -> Result<Vec<u8>> {
        let bit_depth = match quality {
            CompressionQuality::Low => 8,
            CompressionQuality::Medium => 12,
            CompressionQuality::High => 16,
            CompressionQuality::VeryHigh => 20,
        };

        let max_value = (1 << (bit_depth - 1)) - 1;
        let mut quantized = Vec::new();

        for &sample in channel_data.iter() {
            let quantized_sample = (sample * max_value as f32) as i32;
            let clamped_sample = quantized_sample.clamp(-max_value, max_value);

            // Pack into bytes based on bit depth
            match bit_depth {
                8 => quantized.push(clamped_sample as u8),
                12 => {
                    quantized.push((clamped_sample & 0xFF) as u8);
                    quantized.push(((clamped_sample >> 8) & 0x0F) as u8);
                }
                16 => quantized.extend_from_slice(&(clamped_sample as i16).to_le_bytes()),
                20 => {
                    quantized.extend_from_slice(&(clamped_sample & 0xFFFFFF).to_le_bytes()[..3]);
                }
                _ => return Err(Error::LegacyProcessing("Unsupported bit depth".to_string())),
            }
        }

        Ok(quantized)
    }

    /// Apply entropy coding to compressed data.
    ///
    /// Uses canonical Huffman coding over the byte stream (see the
    /// [`huffman`] module). The transform is exactly inverted by
    /// [`Self::decode_entropy_coding`] and is guaranteed never to inflate the
    /// data by more than a single format-marker byte (it falls back to a raw
    /// block for incompressible input).
    fn apply_entropy_coding(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(huffman::encode(data))
    }

    /// Reverse [`Self::apply_entropy_coding`], reconstructing the exact original
    /// bytes from a canonical-Huffman-coded buffer.
    fn decode_entropy_coding(&self, data: &[u8]) -> Result<Vec<u8>> {
        huffman::decode(data).map_err(Error::LegacyProcessing)
    }

    /// Convert audio data to ambisonics representation
    fn convert_to_ambisonics(
        &self,
        audio_data: &Array2<f32>,
        spatial_metadata: &SpatialMetadata,
    ) -> Result<Array2<f32>> {
        let order = self.config.spatial_params.ambisonics_order;
        let ambisonics_channels = (order + 1) * (order + 1);
        let mut ambisonics_data = Array2::zeros((ambisonics_channels, audio_data.ncols()));

        // Simplified conversion - in practice would use proper spherical harmonics
        for (source_idx, &position) in spatial_metadata.source_positions.iter().enumerate() {
            if source_idx >= audio_data.nrows() {
                break;
            }

            let azimuth = position.y.atan2(position.x);
            let elevation = position
                .z
                .atan2((position.x * position.x + position.y * position.y).sqrt());

            // W component (omnidirectional)
            ambisonics_data
                .row_mut(0)
                .scaled_add(1.0, &audio_data.row(source_idx));

            // X, Y, Z components (dipole)
            if ambisonics_channels > 1 {
                ambisonics_data
                    .row_mut(1)
                    .scaled_add(azimuth.cos() * elevation.cos(), &audio_data.row(source_idx));
            }
            if ambisonics_channels > 2 {
                ambisonics_data
                    .row_mut(2)
                    .scaled_add(azimuth.sin() * elevation.cos(), &audio_data.row(source_idx));
            }
            if ambisonics_channels > 3 {
                ambisonics_data
                    .row_mut(3)
                    .scaled_add(elevation.sin(), &audio_data.row(source_idx));
            }
        }

        Ok(ambisonics_data)
    }

    /// Get ambisonics order for channel index
    fn get_ambisonics_channel_order(&self, channel_idx: usize) -> usize {
        // Simplified mapping: channel 0 = order 0, channels 1-3 = order 1, etc.
        if channel_idx == 0 {
            0
        } else if channel_idx <= 3 {
            1
        } else if channel_idx <= 8 {
            2
        } else {
            3
        }
    }

    /// Apply a frequency-domain band-pass filter to every channel of `audio_data`.
    ///
    /// Each row of the input (one source/channel) is processed independently:
    ///
    /// 1. The samples are zero-padded to the next power of two and multiplied by
    ///    a periodic Hann window to suppress spectral leakage at the block edges.
    /// 2. A real forward FFT (`scirs2_fft::RealFftPlanner`) produces the
    ///    half-spectrum of `fft_len / 2 + 1` complex bins.  Bin `k` maps to the
    ///    physical frequency `k · sample_rate / fft_len` (Hz); using the real
    ///    FFT implicitly preserves the symmetric negative-frequency bins.
    /// 3. Every bin whose center frequency lies outside the inclusive
    ///    `[low_freq, high_freq]` pass-band is zeroed.
    /// 4. The inverse real FFT reconstructs the time-domain signal (the
    ///    `scirs2_fft` inverse already normalises by `1/N`), and the first `n`
    ///    samples are written back to the output row.
    ///
    /// The sample rate is taken from [`SpatialCompressionConfig::sample_rate`]
    /// (`self.config.sample_rate`).  This mirrors the real band-pass filters used
    /// in `core.rs` (`apply_air_absorption`) and the SDK audio enhancement path.
    fn filter_frequency_range(
        &self,
        audio_data: &Array2<f32>,
        low_freq: f32,
        high_freq: f32,
    ) -> Result<Array2<f32>> {
        let n = audio_data.ncols();
        let mut filtered = audio_data.clone();

        // Signals shorter than a minimal FFT block (or empty) cannot be
        // meaningfully band-limited; leave them untouched.
        if n < 4 {
            return Ok(filtered);
        }

        let sample_rate = self.config.sample_rate;
        if sample_rate <= 0.0 {
            return Err(Error::LegacyProcessing(
                "invalid sample rate for frequency filtering".to_string(),
            ));
        }

        let fft_len = n.next_power_of_two();
        let n_bins = fft_len / 2 + 1;
        let bin_hz = sample_rate / fft_len as f32;

        // Precompute the inclusive pass-band mask once for all channels.
        // Bin k corresponds to frequency k * sample_rate / fft_len.
        let pass_mask: Vec<bool> = (0..n_bins)
            .map(|k| {
                let freq_hz = k as f32 * bin_hz;
                freq_hz >= low_freq && freq_hz <= high_freq
            })
            .collect();

        // Precompute a periodic Hann window over the padded block.
        let window: Vec<f32> = (0..fft_len)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / fft_len as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();

        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(fft_len);
        let inv = planner.plan_fft_inverse(fft_len);

        let mut input = vec![0.0f32; fft_len];
        let mut spectrum = vec![Complex32::new(0.0, 0.0); n_bins];
        let mut output = vec![0.0f32; fft_len];

        for ch in 0..audio_data.nrows() {
            // Window the zero-padded channel signal.
            let row = audio_data.row(ch);
            for (slot, (&sample, &w)) in input.iter_mut().zip(row.iter().zip(window.iter())) {
                *slot = sample * w;
            }
            // Clear the zero-padding tail (samples beyond `n`).
            for slot in input.iter_mut().skip(n) {
                *slot = 0.0;
            }

            if fwd.process(&input, &mut spectrum).is_err() {
                return Err(Error::LegacyProcessing(
                    "forward FFT failed during frequency filtering".to_string(),
                ));
            }

            // Zero every bin outside the pass-band.
            for (bin, keep) in spectrum.iter_mut().zip(pass_mask.iter()) {
                if !*keep {
                    *bin = Complex32::new(0.0, 0.0);
                }
            }

            if inv.process(&spectrum, &mut output).is_err() {
                return Err(Error::LegacyProcessing(
                    "inverse FFT failed during frequency filtering".to_string(),
                ));
            }

            // The scirs2_fft inverse already applies the 1/N normalisation;
            // write back only the original `n` samples.
            let mut out_row = filtered.row_mut(ch);
            for (dst, &src) in out_row.iter_mut().zip(output.iter()) {
                *dst = src;
            }
        }

        Ok(filtered)
    }

    /// Quantize mixing weights with lower precision
    fn quantize_weights(&self, weights: &Array1<f32>) -> Result<Vec<u8>> {
        let mut quantized = Vec::new();
        for &weight in weights.iter() {
            let quantized_weight = (weight * 255.0) as u8;
            quantized.push(quantized_weight);
        }
        Ok(quantized)
    }

    /// Compress cluster metadata
    fn compress_cluster_metadata(&self) -> Result<Vec<u8>> {
        let mut metadata = Vec::new();

        // Number of clusters
        metadata.push(self.spatial_encoder.source_clusters.len() as u8);

        // For each cluster, store center position (quantized)
        for cluster in &self.spatial_encoder.source_clusters {
            let x_quantized = ((cluster.center.x + 10.0) * 25.5) as u8; // -10 to 10 meters
            let y_quantized = ((cluster.center.y + 10.0) * 25.5) as u8;
            let z_quantized = ((cluster.center.z + 10.0) * 25.5) as u8;

            metadata.extend_from_slice(&[x_quantized, y_quantized, z_quantized]);
            metadata.push(cluster.source_indices.len() as u8);
        }

        Ok(metadata)
    }

    /// Apply simple lossless compression
    fn apply_lossless_compression(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified LZ77-style compression
        let mut compressed = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let mut best_length = 0;
            let mut best_distance = 0;

            // Look for matches in previous data
            let search_start = i.saturating_sub(4096);
            for j in search_start..i {
                let mut length = 0;
                while i + length < data.len()
                    && j + length < i
                    && data[i + length] == data[j + length]
                    && length < 255
                {
                    length += 1;
                }

                if length > best_length && length >= 3 {
                    best_length = length;
                    best_distance = i - j;
                }
            }

            if best_length > 0 {
                // Encode match
                compressed.push(0xFF); // Escape
                compressed.push(0xFE); // Match marker
                compressed.extend_from_slice(&(best_distance as u16).to_le_bytes());
                compressed.push(best_length as u8);
                i += best_length;
            } else {
                // Literal byte
                compressed.push(data[i]);
                i += 1;
            }
        }

        Ok(compressed)
    }

    /// Estimate quality loss for compressed data
    fn estimate_quality_loss(&self, _original: &Array2<f32>, _compressed: &[u8]) -> Result<f32> {
        // Simplified quality estimation based on compression ratio
        let compression_ratio = _original.len() as f32 * 4.0 / _compressed.len() as f32;
        let quality_loss = (compression_ratio - 1.0) / 10.0;
        Ok(quality_loss.clamp(0.0, 1.0))
    }

    /// Get current configuration
    pub fn config(&self) -> &SpatialCompressionConfig {
        &self.config
    }

    /// Get compression statistics for the last frame
    pub fn get_stats(&self) -> Option<CompressionStats> {
        // In a full implementation, would track and return recent statistics
        None
    }
}

// Implementation of helper structures
impl PerceptualModel {
    fn new(params: &PerceptualParams, sample_rate: f32) -> Result<Self> {
        let frequency_bands = Array1::linspace(0.0, sample_rate / 2.0, params.frequency_bands);
        let masking_thresholds = Array1::zeros(params.frequency_bands);
        let bark_scale = Self::compute_bark_scale(&frequency_bands);

        let temporal_state = TemporalMaskingState {
            prev_energy: Array1::zeros(params.frequency_bands),
            pre_masking_buffer: Array2::zeros((params.frequency_bands, 10)),
            post_masking_buffer: Array2::zeros((params.frequency_bands, 100)),
        };

        Ok(Self {
            frequency_bands,
            masking_thresholds,
            bark_scale,
            temporal_state,
        })
    }

    fn compute_bark_scale(frequencies: &Array1<f32>) -> Array1<f32> {
        frequencies.mapv(|f| 13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan())
    }

    fn apply_masking(&mut self, channel_data: &Array1<f32>) -> Result<Array1<f32>> {
        // Simplified masking - just apply some attenuation
        Ok(channel_data.mapv(|x| x * 0.9))
    }

    fn apply_temporal_masking(&mut self, _audio_data: &mut Array2<f32>) -> Result<()> {
        // Placeholder for temporal masking
        Ok(())
    }
}

impl SpatialEncoder {
    fn new(config: &SpatialCompressionConfig) -> Result<Self> {
        let quantization_tables = HashMap::new();
        let huffman_tables = HashMap::new();
        let source_clusters = Vec::new();

        Ok(Self {
            method: config.codec,
            quantization_tables,
            huffman_tables,
            source_clusters,
        })
    }

    fn update_clusters(&mut self, _positions: &[Position3D]) -> Result<()> {
        // Placeholder for cluster update logic
        Ok(())
    }
}

impl AdaptiveController {
    fn new(params: &AdaptiveParams) -> Result<Self> {
        Ok(Self {
            current_bitrate: params.min_bitrate,
            quality_history: Vec::new(),
            bitrate_history: Vec::new(),
            window_samples: (params.adaptation_window * 48000.0) as usize, // Assuming 48kHz
        })
    }

    fn update(&mut self, _config: &SpatialCompressionConfig) -> Result<()> {
        // Placeholder for adaptive bitrate control
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = SpatialCompressionConfig::default();
        assert_eq!(config.codec, CompressionCodec::PerceptualSpatial);
        assert_eq!(config.quality, CompressionQuality::Medium);
        assert_eq!(config.target_bitrate, 128000);
    }

    #[test]
    fn test_compressor_creation() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config);
        assert!(compressor.is_ok());
    }

    #[test]
    fn test_frame_compression() {
        let config = SpatialCompressionConfig::default();
        let mut compressor = SpatialCompressor::new(config).unwrap();

        let audio_data = Array2::ones((8, 1024));
        let spatial_metadata = SpatialMetadata {
            source_positions: vec![Position3D {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }],
            ambisonics_coefficients: vec![],
            spatial_covariance: vec![],
            distance_factors: vec![1.0],
            listener_orientation: (0.0, 0.0, 0.0),
        };

        let result = compressor.compress_frame(&audio_data, &spatial_metadata);
        assert!(result.is_ok());

        let compressed_frame = result.unwrap();
        assert!(!compressed_frame.audio_data.is_empty());
        assert!(compressed_frame.compression_stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_quality_levels() {
        let qualities = [
            CompressionQuality::Low,
            CompressionQuality::Medium,
            CompressionQuality::High,
            CompressionQuality::VeryHigh,
        ];

        for quality in &qualities {
            let mut config = SpatialCompressionConfig::default();
            config.quality = *quality;
            let compressor = SpatialCompressor::new(config);
            assert!(compressor.is_ok());
        }
    }

    #[test]
    fn test_compression_codecs() {
        let codecs = [
            CompressionCodec::PerceptualSpatial,
            CompressionCodec::AmbisonicsOptimized,
            CompressionCodec::PositionalCompression,
            CompressionCodec::Hybrid,
            CompressionCodec::Lossless,
        ];

        for codec in &codecs {
            let mut config = SpatialCompressionConfig::default();
            config.codec = *codec;
            let compressor = SpatialCompressor::new(config);
            assert!(compressor.is_ok());
        }
    }

    #[test]
    fn test_perceptual_model() {
        let params = PerceptualParams {
            masking_enabled: true,
            frequency_bands: 32,
            spatial_masking_threshold: -40.0,
            temporal_masking: TemporalMasking {
                enabled: true,
                pre_masking_ms: 2.0,
                post_masking_ms: 100.0,
                threshold_db: -20.0,
            },
            loudness_compensation: true,
        };

        let model = PerceptualModel::new(&params, 48000.0);
        assert!(model.is_ok());
    }

    /// Deterministic LCG byte source (not the `rand` crate) for reproducible
    /// entropy-coding round-trip corpora.
    fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn test_entropy_coding_roundtrip_through_compressor() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config).unwrap();

        // Empty, single-symbol, highly repetitive, and high-entropy inputs.
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            vec![77u8; 1],
            vec![5u8; 4096],
            lcg_bytes(0xa5a5_5a5a_1234_5678, 3000),
        ];

        for data in &cases {
            let encoded = compressor.apply_entropy_coding(data).unwrap();
            let decoded = compressor.decode_entropy_coding(&encoded).unwrap();
            assert_eq!(&decoded, data, "entropy-coding round-trip must be lossless");
            // No inflation beyond the single format-marker byte.
            assert!(encoded.len() <= data.len() + 1);
        }
    }

    #[test]
    fn test_entropy_coding_compresses_repetitive() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config).unwrap();

        // Skewed/repetitive payload: mostly zeros with occasional spikes.
        let mut data = vec![0u8; 8000];
        for (i, byte) in data.iter_mut().enumerate() {
            if i % 37 == 0 {
                *byte = (i % 251) as u8;
            }
        }

        let encoded = compressor.apply_entropy_coding(&data).unwrap();
        assert!(
            encoded.len() < data.len(),
            "repetitive data should compress: {} -> {}",
            data.len(),
            encoded.len()
        );
        let decoded = compressor.decode_entropy_coding(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    /// Sum of squared samples (proxy for signal energy) over a single row.
    fn row_energy(data: &Array2<f32>, row: usize) -> f64 {
        data.row(row).iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    /// Build a single-channel `Array2` holding a pure sine tone at `freq_hz`.
    fn tone(freq_hz: f32, sample_rate: f32, len: usize) -> Array2<f32> {
        let samples: Vec<f32> = (0..len)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect();
        Array2::from_shape_vec((1, len), samples).unwrap()
    }

    #[test]
    fn test_filter_frequency_range_isolates_passband_tone() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config).unwrap();

        let sample_rate = 48_000.0f32;
        let len = 4096; // power of two => fft_len == len, ~11.7 Hz bin spacing
        let low_tone_hz = 1_000.0f32;
        let high_tone_hz = 8_000.0f32;

        // Mixed signal: low tone + high tone.
        let low = tone(low_tone_hz, sample_rate, len);
        let high = tone(high_tone_hz, sample_rate, len);
        let mixed = &low + &high;

        // Band-pass around ONLY the low tone (500 Hz .. 2 kHz).
        let filtered = compressor
            .filter_frequency_range(&mixed, 500.0, 2_000.0)
            .unwrap();

        // Energy of an isolated low tone (windowed reference) to compare against.
        let low_only = compressor
            .filter_frequency_range(&low, 500.0, 2_000.0)
            .unwrap();

        let filtered_energy = row_energy(&filtered, 0);
        let low_ref_energy = row_energy(&low_only, 0);

        // The pass-band tone's energy is largely retained ...
        assert!(
            filtered_energy > 0.7 * low_ref_energy,
            "pass-band tone energy should be retained: {filtered_energy} vs {low_ref_energy}"
        );

        // ... while the out-of-band (8 kHz) tone is strongly attenuated.
        // Reconstruct what the high tone contributed by removing the low part.
        let high_residual = &filtered - &low_only;
        let residual_energy = row_energy(&high_residual, 0);
        assert!(
            residual_energy < 0.05 * filtered_energy,
            "out-of-band tone must be strongly attenuated: residual {residual_energy} vs kept {filtered_energy}"
        );
    }

    #[test]
    fn test_filter_frequency_range_empty_band_silences() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config).unwrap();

        let sample_rate = 48_000.0f32;
        let len = 4096;

        // Signal energy lives only at 1 kHz and 8 kHz.
        let mixed = &tone(1_000.0, sample_rate, len) + &tone(8_000.0, sample_rate, len);
        let input_energy = row_energy(&mixed, 0);

        // A band containing neither tone => essentially no surviving energy.
        let filtered = compressor
            .filter_frequency_range(&mixed, 15_000.0, 16_000.0)
            .unwrap();
        let out_energy = row_energy(&filtered, 0);

        assert!(
            out_energy < 1e-3 * input_energy,
            "excluding all spectral content should yield near-silence: {out_energy} vs {input_energy}"
        );
    }

    #[test]
    fn test_filter_frequency_range_full_band_is_near_identity() {
        let config = SpatialCompressionConfig::default();
        let compressor = SpatialCompressor::new(config).unwrap();

        let sample_rate = 48_000.0f32;
        let len = 4096;
        let nyquist = sample_rate / 2.0;

        let mixed = &tone(1_000.0, sample_rate, len) + &tone(8_000.0, sample_rate, len);

        // Pass the entire spectrum [0, Nyquist]: window -> FFT -> (no zeroing) ->
        // inverse FFT is an identity on the Hann-windowed signal.
        let filtered = compressor
            .filter_frequency_range(&mixed, 0.0, nyquist)
            .unwrap();

        // Build the Hann-windowed reference the filter operates on.
        let window: Vec<f32> = (0..len)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / len as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        let reference: Vec<f32> = mixed
            .row(0)
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        let mut max_abs_err = 0.0f32;
        for (&got, &want) in filtered.row(0).iter().zip(reference.iter()) {
            max_abs_err = max_abs_err.max((got - want).abs());
        }
        assert!(
            max_abs_err < 1e-3,
            "full-band filtering must reproduce the windowed signal: max err {max_abs_err}"
        );
    }
}
