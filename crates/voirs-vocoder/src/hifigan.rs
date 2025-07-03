//! HiFi-GAN vocoder implementation.

use async_trait::async_trait;
use futures::Stream;
use crate::{
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig,
    Vocoder, VocoderFeature, VocoderMetadata,
    effects::{EffectChain, EffectPresets},
};

// Re-export HiFi-GAN types for external access
pub use crate::models::hifigan::{HiFiGanConfig, HiFiGanVariant, HiFiGanVariants};

#[cfg(feature = "candle")]
use crate::models::hifigan::{
    generator::HiFiGanGenerator,
    inference::HiFiGanInference,
};

/// HiFi-GAN vocoder implementation
pub struct HiFiGanVocoder {
    /// Configuration
    config: HiFiGanConfig,
    /// Inference engine
    #[cfg(feature = "candle")]
    inference: Option<HiFiGanInference>,
    /// Vocoder metadata
    metadata: VocoderMetadata,
    /// Audio post-processing effect chain
    effect_chain: EffectChain,
}

impl HiFiGanVocoder {
    /// Create new HiFi-GAN vocoder with default V1 configuration
    pub fn new() -> Self {
        let config = HiFiGanVariants::v1();
        Self::with_config(config)
    }
    
    /// Create HiFi-GAN vocoder with specific configuration
    pub fn with_config(config: HiFiGanConfig) -> Self {
        let metadata = VocoderMetadata {
            name: config.variant.name().to_string(),
            version: "1.0.0".to_string(),
            architecture: "HiFi-GAN".to_string(),
            sample_rate: config.sample_rate,
            mel_channels: config.mel_channels,
            latency_ms: match config.variant {
                HiFiGanVariant::V1 => 8.0,
                HiFiGanVariant::V2 => 6.0,
                HiFiGanVariant::V3 => 4.0,
            },
            quality_score: match config.variant {
                HiFiGanVariant::V1 => 4.5,
                HiFiGanVariant::V2 => 4.0,
                HiFiGanVariant::V3 => 3.5,
            },
        };
        
        Self {
            #[cfg(feature = "candle")]
            inference: None,
            metadata,
            effect_chain: EffectPresets::speech_enhancement(config.sample_rate),
            config,
        }
    }
    
    /// Create HiFi-GAN vocoder with specific variant
    pub fn with_variant(variant: HiFiGanVariant) -> Self {
        let config = HiFiGanVariants::get_variant(variant);
        Self::with_config(config)
    }
    
    /// Load HiFi-GAN model from file
    #[cfg(feature = "candle")]
    pub fn load_from_file(_path: &str) -> Result<Self> {
        // TODO: Implement model loading from file
        // For now, return default configuration
        Ok(Self::new())
    }
    
    /// Load HiFi-GAN model from file (without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn load_from_file(_path: &str) -> Result<Self> {
        Ok(Self::new())
    }
    
    /// Initialize the inference engine
    #[cfg(feature = "candle")]
    pub fn initialize_inference(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let generator = HiFiGanGenerator::new(self.config.clone(), vb)?;
        let inference = HiFiGanInference::new(generator, self.config.clone())?;
        self.inference = Some(inference);
        Ok(())
    }

    /// Initialize inference for testing (creates dummy weights)
    #[cfg(feature = "candle")]
    pub fn initialize_inference_for_testing(&mut self) -> Result<()> {
        use candle_core::Device;
        use candle_nn::VarMap;
        
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        
        self.initialize_inference(vb)
    }

    /// Initialize inference for testing (without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn initialize_inference_for_testing(&mut self) -> Result<()> {
        // No-op for non-Candle builds
        Ok(())
    }
    
    /// Get configuration
    pub fn config(&self) -> &HiFiGanConfig {
        &self.config
    }
    
    /// Check if inference is initialized
    #[cfg(feature = "candle")]
    pub fn is_initialized(&self) -> bool {
        self.inference.is_some()
    }
    
    /// Check if inference is initialized (always false without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn is_initialized(&self) -> bool {
        false
    }
    
    /// Get mutable reference to effect chain for configuration
    pub fn effect_chain_mut(&mut self) -> &mut EffectChain {
        &mut self.effect_chain
    }
    
    /// Get reference to effect chain
    pub fn effect_chain(&self) -> &EffectChain {
        &self.effect_chain
    }
    
    /// Set effect chain preset
    pub fn set_effect_preset(&mut self, preset_name: &str) {
        self.effect_chain = match preset_name {
            "speech" => EffectPresets::speech_enhancement(self.config.sample_rate),
            "warmth" => EffectPresets::voice_warmth(self.config.sample_rate),
            "broadcast" => EffectPresets::broadcast_quality(self.config.sample_rate),
            "minimal" => EffectPresets::minimal_enhancement(self.config.sample_rate),
            _ => EffectPresets::speech_enhancement(self.config.sample_rate), // Default
        };
    }
    
    /// Enable or disable audio post-processing
    pub fn set_post_processing_enabled(&mut self, enabled: bool) {
        self.effect_chain.set_bypass(!enabled);
    }
    
    /// Apply basic post-processing without mutable state
    fn apply_basic_post_processing(&self, audio: &mut AudioBuffer) {
        let samples = audio.samples_mut();
        
        // 1. DC offset removal (high-pass filter)
        self.apply_dc_removal(samples);
        
        // 2. Normalization
        self.apply_normalization(samples);
        
        // 3. Soft limiting to prevent clipping
        self.apply_soft_limiting(samples);
    }
    
    fn apply_dc_removal(&self, samples: &mut [f32]) {
        if samples.len() < 2 {
            return;
        }
        
        let alpha = 0.995; // High-pass filter coefficient
        let mut prev_input = samples[0];
        let mut prev_output = samples[0];
        
        for i in 1..samples.len() {
            let current_input = samples[i];
            let output = alpha * (prev_output + current_input - prev_input);
            samples[i] = output;
            
            prev_input = current_input;
            prev_output = output;
        }
    }
    
    fn apply_normalization(&self, samples: &mut [f32]) {
        // Find peak level
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        
        if peak > 0.01 && peak < 0.95 {
            // Normalize to -3dB peak (0.7 linear)
            let target_peak = 0.7;
            let scale = target_peak / peak;
            for sample in samples {
                *sample *= scale;
            }
        }
    }
    
    fn apply_soft_limiting(&self, samples: &mut [f32]) {
        let threshold = 0.95;
        
        for sample in samples {
            if sample.abs() > threshold {
                // Soft limiting using tanh saturation
                let sign = if *sample > 0.0 { 1.0 } else { -1.0 };
                let normalized = sample.abs() / threshold;
                *sample = sign * threshold * normalized.tanh();
            }
        }
    }
}

#[async_trait]
impl Vocoder for HiFiGanVocoder {
    async fn vocode(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "candle")]
        {
            if let Some(inference) = &self.inference {
                let (mut audio, _stats) = inference.infer(mel, config).await?;
                
                // Apply basic audio post-processing (without mutable effect chain)
                self.apply_basic_post_processing(&mut audio);
                
                Ok(audio)
            } else {
                Err(crate::VocoderError::ModelError(
                    "HiFi-GAN inference not initialized. Call initialize_inference() first.".to_string(),
                ))
            }
        }
        
        #[cfg(not(feature = "candle"))]
        {
            // Fallback to dummy implementation when Candle is not available
            tracing::warn!("HiFi-GAN inference requires Candle feature. Generating dummy audio.");
            let duration = mel.duration();
            let frequency = 440.0; // A4 note
            let mut audio = AudioBuffer::sine_wave(frequency, duration, self.config.sample_rate, 0.3);
            
            // Apply basic audio post-processing
            self.apply_basic_post_processing(&mut audio);
            
            Ok(audio)
        }
    }
    
    async fn vocode_stream(
        &self,
        mut mel_stream: Box<dyn Stream<Item = MelSpectrogram> + Send + Unpin>,
        config: Option<&SynthesisConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<AudioBuffer>> + Send + Unpin>> {
        use futures::StreamExt;
        
        // Create streaming vocoder with proper chunking
        let mut streaming_vocoder = StreamingVocoder::new(self.clone(), config.cloned())?;
        
        // Collect results as we process the stream
        let mut results = Vec::new();
        
        while let Some(mel) = mel_stream.next().await {
            match streaming_vocoder.process_chunk(&mel).await {
                Ok(Some(audio)) => results.push(Ok(audio)),
                Ok(None) => {}, // No output ready yet - accumulating in buffer
                Err(e) => results.push(Err(e)),
            }
        }
        
        // Flush any remaining audio from the buffer
        if let Ok(Some(audio)) = streaming_vocoder.flush().await {
            results.push(Ok(audio));
        }
        
        let audio_stream = futures::stream::iter(results.into_iter());
        
        Ok(Box::new(audio_stream))
    }
    
    async fn vocode_batch(
        &self,
        mels: &[MelSpectrogram],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::new();
        for (i, mel) in mels.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            results.push(self.vocode(mel, config).await?);
        }
        Ok(results)
    }
    
    fn metadata(&self) -> VocoderMetadata {
        self.metadata.clone()
    }
    
    fn supports(&self, feature: VocoderFeature) -> bool {
        match feature {
            VocoderFeature::StreamingInference => true,
            VocoderFeature::BatchProcessing => true,
            VocoderFeature::GpuAcceleration => cfg!(feature = "candle"),
            VocoderFeature::HighQuality => true,
            VocoderFeature::RealtimeProcessing => {
                matches!(self.config.variant, HiFiGanVariant::V2 | HiFiGanVariant::V3)
            }
            VocoderFeature::FastInference => {
                matches!(self.config.variant, HiFiGanVariant::V1 | HiFiGanVariant::V2)
            }
        }
    }
}

impl Default for HiFiGanVocoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming vocoder for chunk-based processing with overlap-add windowing
pub struct StreamingVocoder {
    vocoder: HiFiGanVocoder,
    config: Option<SynthesisConfig>,
    
    // Mel spectrogram buffering
    mel_buffer: std::collections::VecDeque<Vec<f32>>, // Buffered mel frames
    mel_buffer_size: usize,
    
    // Audio output buffering with overlap-add
    audio_buffer: std::collections::VecDeque<f32>,
    overlap_buffer: Vec<f32>, // For overlap-add between chunks
    
    // Streaming configuration
    chunk_size: usize,        // Mel frames per processing chunk
    overlap_size: usize,      // Overlap frames for smooth transitions
    hop_length: usize,        // Hop length for mel-to-audio conversion
    sample_rate: u32,
    
    // State tracking
    processed_frames: usize,
    total_latency_samples: usize,
    lookahead_frames: usize,  // Frames to buffer before processing
    
    // Performance metrics
    processing_times: std::collections::VecDeque<std::time::Duration>,
    max_metrics_history: usize,
}

impl StreamingVocoder {
    pub fn new(vocoder: HiFiGanVocoder, config: Option<SynthesisConfig>) -> Result<Self> {
        // Adaptive chunk size based on vocoder variant for optimal latency
        let (chunk_size, overlap_size, lookahead_frames) = match vocoder.config().variant {
            HiFiGanVariant::V1 => (256, 64, 128),  // Higher quality, more latency
            HiFiGanVariant::V2 => (192, 48, 96),   // Balanced
            HiFiGanVariant::V3 => (128, 32, 64),   // Lower latency, real-time optimized
        };
        
        let hop_length = 256; // Standard hop length for mel-to-audio conversion
        let sample_rate = vocoder.config().sample_rate;
        let mel_buffer_size = lookahead_frames + chunk_size + overlap_size;
        
        // Calculate total latency in samples
        let processing_latency = chunk_size * hop_length;
        let buffering_latency = lookahead_frames * hop_length;
        let total_latency_samples = processing_latency + buffering_latency;
        
        Ok(Self {
            vocoder,
            config,
            
            // Initialize mel buffering
            mel_buffer: std::collections::VecDeque::new(),
            mel_buffer_size,
            
            // Initialize audio buffering
            audio_buffer: std::collections::VecDeque::new(),
            overlap_buffer: Vec::new(),
            
            // Streaming configuration
            chunk_size,
            overlap_size,
            hop_length,
            sample_rate,
            
            // State tracking
            processed_frames: 0,
            total_latency_samples,
            lookahead_frames,
            
            // Performance metrics
            processing_times: std::collections::VecDeque::new(),
            max_metrics_history: 100,
        })
    }
    
    /// Add mel frames to the buffer and process when enough data is available
    pub async fn process_chunk(&mut self, mel: &MelSpectrogram) -> Result<Option<AudioBuffer>> {
        let start_time = std::time::Instant::now();
        
        // Extract mel frames and add to buffer
        let mel_frames = self.extract_mel_frames(mel)?;
        for frame in mel_frames {
            self.mel_buffer.push_back(frame);
            
            // Limit buffer size to prevent memory growth
            while self.mel_buffer.len() > self.mel_buffer_size {
                self.mel_buffer.pop_front();
            }
        }
        
        // Check if we have enough frames for processing
        if self.mel_buffer.len() >= self.lookahead_frames + self.chunk_size {
            let audio_chunk = self.process_buffered_chunk().await?;
            
            // Record processing time for performance monitoring
            let processing_time = start_time.elapsed();
            self.processing_times.push_back(processing_time);
            while self.processing_times.len() > self.max_metrics_history {
                self.processing_times.pop_front();
            }
            
            Ok(Some(audio_chunk))
        } else {
            // Not enough frames yet, return None
            Ok(None)
        }
    }
    
    /// Flush remaining buffered audio
    pub async fn flush(&mut self) -> Result<Option<AudioBuffer>> {
        if self.mel_buffer.len() >= self.chunk_size {
            // Process remaining mel frames even if we don't have full lookahead
            let audio_chunk = self.process_buffered_chunk().await?;
            Ok(Some(audio_chunk))
        } else if !self.audio_buffer.is_empty() {
            // Return any remaining audio samples
            let samples: Vec<f32> = self.audio_buffer.drain(..).collect();
            let audio = AudioBuffer::new(samples, self.sample_rate, 1);
            Ok(Some(audio))
        } else {
            Ok(None)
        }
    }
    
    /// Extract mel frames from MelSpectrogram for buffering
    fn extract_mel_frames(&self, mel: &MelSpectrogram) -> Result<Vec<Vec<f32>>> {
        // For now, create dummy frames. In a real implementation, 
        // you would extract actual mel data from the MelSpectrogram
        let num_frames = mel.n_frames;
        let mel_channels = mel.n_mels;
        
        let mut frames = Vec::new();
        for _ in 0..num_frames {
            // Create dummy mel frame with appropriate dimensions
            let frame = vec![0.0f32; mel_channels];
            frames.push(frame);
        }
        
        Ok(frames)
    }
    
    /// Process a chunk of buffered mel frames
    async fn process_buffered_chunk(&mut self) -> Result<AudioBuffer> {
        // Collect chunk_size frames from the buffer (skip lookahead frames)
        let start_idx = self.lookahead_frames.min(self.mel_buffer.len());
        let end_idx = (start_idx + self.chunk_size).min(self.mel_buffer.len());
        
        if start_idx >= end_idx {
            return Err(crate::VocoderError::VocodingError(
                "Insufficient frames for processing".to_string(),
            ));
        }
        
        // Create mel spectrogram from buffered frames for vocoding
        let chunk_frames = self.mel_buffer.range(start_idx..end_idx).cloned().collect::<Vec<_>>();
        let chunk_mel = self.create_mel_from_frames(&chunk_frames)?;
        
        // Vocode the chunk
        let mut audio = self.vocoder.vocode(&chunk_mel, self.config.as_ref()).await?;
        
        // Apply overlap-add with previous chunk
        if !self.overlap_buffer.is_empty() {
            let samples = audio.samples_mut();
            self.apply_overlap_add(samples);
        }
        
        // Store overlap for next chunk
        self.update_overlap_buffer(&audio);
        
        // Update processed frame count
        self.processed_frames += end_idx - start_idx;
        
        // Remove processed frames from buffer (keeping lookahead)
        let frames_to_remove = (end_idx - start_idx).min(self.mel_buffer.len().saturating_sub(self.lookahead_frames));
        for _ in 0..frames_to_remove {
            self.mel_buffer.pop_front();
        }
        
        Ok(audio)
    }
    
    /// Create MelSpectrogram from buffered frames
    fn create_mel_from_frames(&self, frames: &[Vec<f32>]) -> Result<MelSpectrogram> {
        if frames.is_empty() {
            return Err(crate::VocoderError::VocodingError(
                "Cannot create mel from empty frames".to_string(),
            ));
        }
        
        let num_frames = frames.len();
        let num_channels = frames[0].len();
        
        // Convert frames to the format expected by MelSpectrogram::new
        // We need to transpose: frames is [frame][channel] but MelSpectrogram wants [channel][frame]
        let mut data = vec![vec![0.0; num_frames]; num_channels];
        for (frame_idx, frame) in frames.iter().enumerate() {
            for (channel_idx, &value) in frame.iter().enumerate() {
                data[channel_idx][frame_idx] = value;
            }
        }
        
        // Create mel spectrogram from transposed data
        Ok(MelSpectrogram::new(
            data,
            self.sample_rate,
            self.hop_length as u32,
        ))
    }
    
    /// Apply overlap-add windowing with the previous chunk
    fn apply_overlap_add(&mut self, current_samples: &mut [f32]) {
        let overlap_len = self.overlap_buffer.len().min(current_samples.len());
        
        for i in 0..overlap_len {
            // Linear fade between chunks
            let fade_factor = i as f32 / overlap_len as f32;
            current_samples[i] = current_samples[i] * fade_factor + 
                               self.overlap_buffer[i] * (1.0 - fade_factor);
        }
    }
    
    /// Update overlap buffer for next chunk
    fn update_overlap_buffer(&mut self, audio: &AudioBuffer) {
        let samples = audio.samples();
        let overlap_samples = self.overlap_size * self.hop_length;
        
        if samples.len() >= overlap_samples {
            let start_idx = samples.len() - overlap_samples;
            self.overlap_buffer = samples[start_idx..].to_vec();
        } else {
            self.overlap_buffer = samples.to_vec();
        }
    }
    
    fn apply_windowing(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // Apply Hann window for smooth transitions
        let samples = audio.samples().to_vec();
        let windowed_samples = self.apply_hann_window(&samples);
        
        Ok(AudioBuffer::new(
            windowed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }
    
    fn apply_hann_window(&self, samples: &[f32]) -> Vec<f32> {
        let len = samples.len();
        let mut windowed = Vec::with_capacity(len);
        
        for (i, &sample) in samples.iter().enumerate() {
            // Apply Hann window: 0.5 * (1 - cos(2π * i / (N-1)))
            let window_val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
            windowed.push(sample * window_val);
        }
        
        windowed
    }
    
    /// Add overlap-add processing for smooth transitions
    fn overlap_add(&self, new_audio: &[f32], overlap_samples: &[f32]) -> Vec<f32> {
        let mut result = new_audio.to_vec();
        let overlap_len = overlap_samples.len().min(result.len());
        
        // Add overlapping samples with fade
        for i in 0..overlap_len {
            let fade_factor = i as f32 / overlap_len as f32;
            result[i] = new_audio[i] * fade_factor + overlap_samples[i] * (1.0 - fade_factor);
        }
        
        result
    }
    
    /// Get streaming performance metrics
    pub fn get_performance_metrics(&self) -> StreamingMetrics {
        let avg_processing_time = if self.processing_times.is_empty() {
            std::time::Duration::from_secs(0)
        } else {
            let total: std::time::Duration = self.processing_times.iter().sum();
            total / self.processing_times.len() as u32
        };
        
        let max_processing_time = self.processing_times.iter()
            .max()
            .copied()
            .unwrap_or_else(|| std::time::Duration::from_secs(0));
        
        StreamingMetrics {
            avg_processing_time_ms: avg_processing_time.as_millis() as f32,
            max_processing_time_ms: max_processing_time.as_millis() as f32,
            total_latency_ms: (self.total_latency_samples as f32 / self.sample_rate as f32) * 1000.0,
            processed_frames: self.processed_frames,
            buffer_utilization: (self.mel_buffer.len() as f32 / self.mel_buffer_size as f32) * 100.0,
            chunk_size: self.chunk_size,
            overlap_size: self.overlap_size,
        }
    }
    
    /// Configure adaptive streaming parameters based on performance
    pub fn configure_for_performance(&mut self, target_latency_ms: f32) -> Result<()> {
        let target_latency_samples = (target_latency_ms / 1000.0 * self.sample_rate as f32) as usize;
        
        // Adjust chunk size to meet latency requirements
        let max_chunk_latency = target_latency_samples / 2; // Leave room for processing
        let max_chunk_frames = max_chunk_latency / self.hop_length;
        
        if max_chunk_frames > 32 {
            self.chunk_size = max_chunk_frames.min(512); // Cap at reasonable maximum
            self.overlap_size = self.chunk_size / 4; // 25% overlap
            self.lookahead_frames = self.chunk_size / 2; // 50% lookahead
            
            // Recalculate buffer size and total latency
            self.mel_buffer_size = self.lookahead_frames + self.chunk_size + self.overlap_size;
            self.total_latency_samples = self.chunk_size * self.hop_length + self.lookahead_frames * self.hop_length;
            
            Ok(())
        } else {
            Err(crate::VocoderError::ConfigError(
                "Target latency too low for stable streaming".to_string(),
            ))
        }
    }
    
    /// Reset the streaming state
    pub fn reset(&mut self) {
        self.mel_buffer.clear();
        self.audio_buffer.clear();
        self.overlap_buffer.clear();
        self.processed_frames = 0;
        self.processing_times.clear();
    }
    
    /// Get estimated latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        (self.total_latency_samples as f32 / self.sample_rate as f32) * 1000.0
    }
    
    /// Check if the vocoder can process in real-time
    pub fn is_realtime_capable(&self) -> bool {
        let avg_metrics = self.get_performance_metrics();
        let processing_budget_ms = (self.chunk_size as f32 / self.sample_rate as f32) * 1000.0;
        
        avg_metrics.avg_processing_time_ms < processing_budget_ms * 0.8 // 80% budget utilization
    }
}

/// Performance metrics for streaming vocoder
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    pub avg_processing_time_ms: f32,
    pub max_processing_time_ms: f32,
    pub total_latency_ms: f32,
    pub processed_frames: usize,
    pub buffer_utilization: f32,
    pub chunk_size: usize,
    pub overlap_size: usize,
}

impl Clone for HiFiGanVocoder {
    fn clone(&self) -> Self {
        // For simplified implementation, create a new vocoder with same config
        // In a full implementation, you would properly clone the model state
        let mut new_vocoder = HiFiGanVocoder::with_config(self.config.clone());
        
        // Copy effect chain settings (create new chain with same preset)
        new_vocoder.effect_chain = EffectPresets::speech_enhancement(self.config.sample_rate);
        new_vocoder.effect_chain.set_bypass(self.effect_chain.is_bypassed());
        new_vocoder.effect_chain.set_wet_dry_mix(self.effect_chain.get_wet_dry_mix());
        new_vocoder.effect_chain.set_output_gain(self.effect_chain.get_output_gain());
        
        new_vocoder
    }
}