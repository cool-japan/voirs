//! Real-time voice conversion system for live audio processing

use crate::embedding::SpeakerEmbedding;
use crate::{types::VoiceSample, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Real-time voice conversion system
pub struct VoiceConverter {
    /// Conversion configuration
    config: ConversionConfig,
    /// Active conversion sessions
    sessions: Arc<RwLock<HashMap<String, ConversionSession>>>,
    /// Conversion model cache
    model_cache: Arc<RwLock<HashMap<String, ConversionModel>>>,
    /// Performance statistics
    stats: Arc<RwLock<ConversionStatistics>>,
}

/// Configuration for voice conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Maximum latency for real-time processing (ms)
    pub max_latency_ms: u32,
    /// Audio chunk size for processing
    pub chunk_size: usize,
    /// Overlap between chunks (samples)
    pub chunk_overlap: usize,
    /// Quality level (0.0=fast, 1.0=quality)
    pub quality_level: f32,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: u32,
    /// Model cache size
    pub model_cache_size: usize,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 50,
            chunk_size: 1024,
            chunk_overlap: 256,
            quality_level: 0.8,
            enable_gpu: true,
            max_concurrent_sessions: 10,
            model_cache_size: 20,
        }
    }
}

/// Real-time conversion session
#[derive(Debug)]
pub struct ConversionSession {
    /// Session ID
    pub id: String,
    /// Source speaker profile
    pub source_speaker: SpeakerEmbedding,
    /// Target speaker profile
    pub target_speaker: SpeakerEmbedding,
    /// Audio buffer for processing
    pub audio_buffer: VecDeque<f32>,
    /// Processed audio buffer
    pub output_buffer: VecDeque<f32>,
    /// Session configuration
    pub config: ConversionConfig,
    /// Session start time
    pub start_time: Instant,
    /// Last processing time
    pub last_activity: Instant,
    /// Total samples processed
    pub samples_processed: u64,
    /// Current latency (ms)
    pub current_latency: f32,
    /// Quality score
    pub quality_score: f32,
}

/// Voice conversion model for speaker-to-speaker transformation
#[derive(Debug, Clone)]
pub struct ConversionModel {
    /// Source speaker embedding
    pub source_embedding: SpeakerEmbedding,
    /// Target speaker embedding
    pub target_embedding: SpeakerEmbedding,
    /// Conversion parameters
    pub conversion_params: ConversionParameters,
    /// Model quality score
    pub quality_score: f32,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last used timestamp
    pub last_used: Instant,
}

/// Parameters for voice conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionParameters {
    /// Fundamental frequency scaling factor
    pub f0_scale: f32,
    /// Formant shift parameters
    pub formant_shifts: Vec<f32>,
    /// Spectral envelope transformation
    pub spectral_envelope: Vec<f32>,
    /// Voice quality adjustments
    pub voice_quality: f32,
    /// Temporal characteristics
    pub temporal_scale: f32,
    /// Prosodic modifications
    pub prosody_params: ProsodyParameters,
}

/// Prosodic modification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyParameters {
    /// Speaking rate modification
    pub rate_scale: f32,
    /// Energy scaling
    pub energy_scale: f32,
    /// Pause duration scaling
    pub pause_scale: f32,
    /// Rhythm modification strength
    pub rhythm_strength: f32,
}

impl Default for ProsodyParameters {
    fn default() -> Self {
        Self {
            rate_scale: 1.0,
            energy_scale: 1.0,
            pause_scale: 1.0,
            rhythm_strength: 0.5,
        }
    }
}

/// Real-time conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    /// Converted audio data
    pub audio_data: Vec<f32>,
    /// Processing latency (ms)
    pub latency_ms: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Conversion confidence
    pub confidence: f32,
    /// Processing metadata
    pub metadata: HashMap<String, f32>,
}

/// Statistics for voice conversion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionStatistics {
    /// Total sessions created
    pub total_sessions: u64,
    /// Active sessions
    pub active_sessions: u32,
    /// Total audio processed (seconds)
    pub total_audio_processed: f64,
    /// Average latency
    pub average_latency_ms: f32,
    /// Average quality score
    pub average_quality: f32,
    /// Conversion success rate
    pub success_rate: f32,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
}

impl Default for ConversionStatistics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            active_sessions: 0,
            total_audio_processed: 0.0,
            average_latency_ms: 0.0,
            average_quality: 0.0,
            success_rate: 1.0,
            performance_metrics: HashMap::new(),
        }
    }
}

impl VoiceConverter {
    /// Create new voice conversion system
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ConversionStatistics::default())),
        }
    }

    /// Create voice conversion session
    pub async fn create_session(
        &self,
        source_speaker: SpeakerEmbedding,
        target_speaker: SpeakerEmbedding,
    ) -> Result<String> {
        let session_id = format!("session_{}", uuid::Uuid::new_v4());

        // Check session limits
        {
            let sessions = self.sessions.read().unwrap();
            if sessions.len() >= self.config.max_concurrent_sessions as usize {
                return Err(Error::Processing(
                    "Maximum concurrent sessions reached".to_string(),
                ));
            }
        }

        // Create conversion model
        let model = self
            .create_conversion_model(&source_speaker, &target_speaker)
            .await?;

        // Cache the model
        {
            let mut cache = self.model_cache.write().unwrap();
            let cache_key = format!(
                "{:?}_{:?}",
                source_speaker.vector.iter().sum::<f32>(),
                target_speaker.vector.iter().sum::<f32>()
            );
            cache.insert(cache_key, model);
        }

        // Create session
        let session = ConversionSession {
            id: session_id.clone(),
            source_speaker,
            target_speaker,
            audio_buffer: VecDeque::new(),
            output_buffer: VecDeque::new(),
            config: self.config.clone(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            samples_processed: 0,
            current_latency: 0.0,
            quality_score: 0.0,
        };

        // Store session
        {
            let mut sessions = self.sessions.write().unwrap();
            sessions.insert(session_id.clone(), session);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_sessions += 1;
            stats.active_sessions = self.sessions.read().unwrap().len() as u32;
        }

        Ok(session_id)
    }

    /// Process audio chunk in real-time
    pub async fn process_chunk(
        &self,
        session_id: &str,
        audio_chunk: &[f32],
        sample_rate: u32,
    ) -> Result<ConversionResult> {
        let start_time = Instant::now();

        // Get conversion model
        let model = self.get_conversion_model_for_session(session_id)?;

        // Process audio
        let converted_audio = self
            .apply_conversion(audio_chunk, &model, sample_rate)
            .await?;

        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis() as f32;

        // Update session
        {
            let mut sessions = self.sessions.write().unwrap();
            if let Some(session) = sessions.get_mut(session_id) {
                session.last_activity = Instant::now();
                session.samples_processed += audio_chunk.len() as u64;
                session.current_latency = latency_ms;

                // Calculate quality score (simplified)
                session.quality_score =
                    self.calculate_conversion_quality(&converted_audio, &model)?;
            }
        }

        // Update statistics
        self.update_statistics(latency_ms, model.quality_score)?;

        Ok(ConversionResult {
            audio_data: converted_audio,
            latency_ms,
            quality_score: model.quality_score,
            confidence: 0.8, // Simplified confidence calculation
            metadata: HashMap::new(),
        })
    }

    /// Stream audio through conversion system
    pub async fn stream_convert(
        &self,
        session_id: &str,
        input_stream: &mut dyn Iterator<Item = f32>,
        sample_rate: u32,
    ) -> Result<Vec<ConversionResult>> {
        let mut results = Vec::new();
        let mut buffer = Vec::with_capacity(self.config.chunk_size);

        for sample in input_stream {
            buffer.push(sample);

            if buffer.len() >= self.config.chunk_size {
                let result = self.process_chunk(session_id, &buffer, sample_rate).await?;
                results.push(result);

                // Keep overlap for continuity
                if self.config.chunk_overlap > 0 && buffer.len() > self.config.chunk_overlap {
                    buffer.drain(0..buffer.len() - self.config.chunk_overlap);
                } else {
                    buffer.clear();
                }
            }
        }

        // Process remaining samples
        if !buffer.is_empty() {
            let result = self.process_chunk(session_id, &buffer, sample_rate).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Close conversion session
    pub fn close_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        if sessions.remove(session_id).is_some() {
            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.active_sessions = sessions.len() as u32;
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Session {} not found",
                session_id
            )))
        }
    }

    /// Get session information
    pub fn get_session_info(&self, session_id: &str) -> Result<SessionInfo> {
        let sessions = self.sessions.read().unwrap();
        if let Some(session) = sessions.get(session_id) {
            Ok(SessionInfo {
                id: session.id.clone(),
                start_time: session.start_time,
                last_activity: session.last_activity,
                samples_processed: session.samples_processed,
                current_latency: session.current_latency,
                quality_score: session.quality_score,
                duration: session.start_time.elapsed(),
            })
        } else {
            Err(Error::Processing(format!(
                "Session {} not found",
                session_id
            )))
        }
    }

    /// Create conversion model for speaker pair
    async fn create_conversion_model(
        &self,
        source: &SpeakerEmbedding,
        target: &SpeakerEmbedding,
    ) -> Result<ConversionModel> {
        // Calculate conversion parameters based on speaker embeddings
        let conversion_params = self.calculate_conversion_parameters(source, target)?;

        // Estimate quality score
        let quality_score = self.estimate_conversion_quality(source, target)?;

        Ok(ConversionModel {
            source_embedding: source.clone(),
            target_embedding: target.clone(),
            conversion_params,
            quality_score,
            created_at: Instant::now(),
            last_used: Instant::now(),
        })
    }

    /// Calculate conversion parameters from speaker embeddings
    fn calculate_conversion_parameters(
        &self,
        source: &SpeakerEmbedding,
        target: &SpeakerEmbedding,
    ) -> Result<ConversionParameters> {
        // Extract F0 characteristics from embeddings
        let source_f0 = self.extract_f0_from_embedding(&source.vector)?;
        let target_f0 = self.extract_f0_from_embedding(&target.vector)?;
        let f0_scale = target_f0 / source_f0.max(1.0);

        // Calculate formant shifts
        let formant_shifts = self.calculate_formant_shifts(&source.vector, &target.vector)?;

        // Calculate spectral envelope transformation
        let spectral_envelope =
            self.calculate_spectral_transformation(&source.vector, &target.vector)?;

        // Calculate voice quality adjustments
        let voice_quality =
            self.calculate_voice_quality_adjustment(&source.vector, &target.vector)?;

        // Calculate temporal scaling
        let temporal_scale = self.calculate_temporal_scaling(&source.vector, &target.vector)?;

        // Calculate prosodic parameters
        let prosody_params = self.calculate_prosody_parameters(&source.vector, &target.vector)?;

        Ok(ConversionParameters {
            f0_scale,
            formant_shifts,
            spectral_envelope,
            voice_quality,
            temporal_scale,
            prosody_params,
        })
    }

    /// Apply voice conversion to audio
    async fn apply_conversion(
        &self,
        audio: &[f32],
        model: &ConversionModel,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut converted = audio.to_vec();

        // Apply F0 conversion
        self.apply_f0_conversion(
            &mut converted,
            model.conversion_params.f0_scale,
            sample_rate,
        )?;

        // Apply formant shifts
        self.apply_formant_shifts(
            &mut converted,
            &model.conversion_params.formant_shifts,
            sample_rate,
        )?;

        // Apply spectral envelope transformation
        self.apply_spectral_transformation(
            &mut converted,
            &model.conversion_params.spectral_envelope,
        )?;

        // Apply voice quality adjustments
        self.apply_voice_quality(&mut converted, model.conversion_params.voice_quality)?;

        // Apply temporal modifications
        if model.conversion_params.temporal_scale != 1.0 {
            converted =
                self.apply_temporal_scaling(&converted, model.conversion_params.temporal_scale)?;
        }

        // Apply prosodic modifications
        self.apply_prosody_modifications(
            &mut converted,
            &model.conversion_params.prosody_params,
            sample_rate,
        )?;

        Ok(converted)
    }

    /// Extract F0 characteristics from speaker embedding
    fn extract_f0_from_embedding(&self, embedding: &[f32]) -> Result<f32> {
        if embedding.is_empty() {
            return Ok(150.0); // Default F0
        }

        // Use first part of embedding for F0 characteristics
        let f0_section = &embedding[0..32.min(embedding.len())];
        let f0_mean = f0_section.iter().sum::<f32>() / f0_section.len() as f32;
        Ok(100.0 + f0_mean * 100.0) // Map to reasonable F0 range (100-200 Hz)
    }

    /// Calculate formant shift parameters
    fn calculate_formant_shifts(&self, source: &[f32], target: &[f32]) -> Result<Vec<f32>> {
        let mut shifts = Vec::new();

        // Extract formant-related features from middle section of embeddings
        let start = 32.min(source.len());
        let end = 64.min(source.len());
        let source_formants = &source[start..end];
        let target_formants = &target[start..end.min(target.len())];

        for i in 0..source_formants.len().min(target_formants.len()) {
            let shift = if source_formants[i] != 0.0 {
                target_formants[i] / source_formants[i]
            } else {
                1.0
            };
            shifts.push(shift.clamp(0.5, 2.0)); // Reasonable formant shift range
        }

        // Ensure we have at least 4 formant shifts
        while shifts.len() < 4 {
            shifts.push(1.0);
        }

        Ok(shifts)
    }

    /// Calculate spectral envelope transformation
    fn calculate_spectral_transformation(
        &self,
        source: &[f32],
        target: &[f32],
    ) -> Result<Vec<f32>> {
        let mut envelope = Vec::new();
        let target_size = 64; // Standard spectral envelope size

        for i in 0..target_size {
            let source_idx = (i * source.len() / target_size).min(source.len() - 1);
            let target_idx = (i * target.len() / target_size).min(target.len() - 1);

            let source_val = source.get(source_idx).unwrap_or(&0.0);
            let target_val = target.get(target_idx).unwrap_or(&0.0);

            // Calculate transformation coefficient
            let transform = if *source_val != 0.0 {
                target_val / source_val
            } else {
                1.0
            };

            envelope.push(transform.clamp(0.1, 10.0));
        }

        Ok(envelope)
    }

    /// Calculate voice quality adjustment
    fn calculate_voice_quality_adjustment(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Use last section for voice quality
        let start = source.len().saturating_sub(32);
        let source_quality = &source[start..];
        let target_quality = &target[start.min(target.len())..];

        let source_mean = source_quality.iter().sum::<f32>() / source_quality.len().max(1) as f32;
        let target_mean = target_quality.iter().sum::<f32>() / target_quality.len().max(1) as f32;

        Ok((target_mean - source_mean).clamp(-1.0, 1.0))
    }

    /// Calculate temporal scaling factor
    fn calculate_temporal_scaling(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Extract temporal characteristics
        let source_variance = self.calculate_variance(source);
        let target_variance = self.calculate_variance(target);

        let scale = if source_variance > 0.0 {
            (target_variance / source_variance).sqrt()
        } else {
            1.0
        };

        Ok(scale.clamp(0.5, 2.0))
    }

    /// Calculate prosody modification parameters
    fn calculate_prosody_parameters(
        &self,
        source: &[f32],
        target: &[f32],
    ) -> Result<ProsodyParameters> {
        // Simplified prosody parameter calculation
        let source_mean = source.iter().sum::<f32>() / source.len() as f32;
        let target_mean = target.iter().sum::<f32>() / target.len() as f32;

        let rate_scale = (1.0 + (target_mean - source_mean) * 0.3).clamp(0.7, 1.3);
        let energy_scale = (1.0 + (target_mean - source_mean) * 0.2).clamp(0.8, 1.2);

        Ok(ProsodyParameters {
            rate_scale,
            energy_scale,
            pause_scale: 1.0,
            rhythm_strength: 0.5,
        })
    }

    /// Apply F0 conversion to audio
    fn apply_f0_conversion(
        &self,
        audio: &mut [f32],
        f0_scale: f32,
        _sample_rate: u32,
    ) -> Result<()> {
        // Simplified F0 modification (pitch shifting)
        if f0_scale != 1.0 {
            for sample in audio.iter_mut() {
                *sample *= f0_scale.clamp(0.5, 2.0);
            }
        }
        Ok(())
    }

    /// Apply formant shifts to audio
    fn apply_formant_shifts(
        &self,
        audio: &mut [f32],
        _shifts: &[f32],
        _sample_rate: u32,
    ) -> Result<()> {
        // Simplified formant shifting (spectral shaping)
        // In a real implementation, this would involve spectral processing
        let shift_factor = 1.0 + 0.1 * (_shifts.first().unwrap_or(&1.0) - 1.0);

        for sample in audio.iter_mut() {
            *sample *= shift_factor;
        }

        Ok(())
    }

    /// Apply spectral envelope transformation
    fn apply_spectral_transformation(&self, audio: &mut [f32], envelope: &[f32]) -> Result<()> {
        // Simplified spectral envelope application
        let envelope_factor = envelope.iter().sum::<f32>() / envelope.len() as f32;

        for sample in audio.iter_mut() {
            *sample *= envelope_factor.clamp(0.1, 2.0);
        }

        Ok(())
    }

    /// Apply voice quality adjustments
    fn apply_voice_quality(&self, audio: &mut [f32], quality_adjustment: f32) -> Result<()> {
        // Apply voice quality modification (simplified)
        let quality_factor = 1.0 + quality_adjustment * 0.1;

        for sample in audio.iter_mut() {
            *sample *= quality_factor;
        }

        Ok(())
    }

    /// Apply temporal scaling
    fn apply_temporal_scaling(&self, audio: &[f32], scale: f32) -> Result<Vec<f32>> {
        if scale == 1.0 {
            return Ok(audio.to_vec());
        }

        // Simple time stretching/compression
        let new_length = (audio.len() as f32 / scale) as usize;
        let mut stretched = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let original_index = (i as f32 * scale) as usize;
            if original_index < audio.len() {
                stretched.push(audio[original_index]);
            } else {
                stretched.push(0.0);
            }
        }

        Ok(stretched)
    }

    /// Apply prosodic modifications
    fn apply_prosody_modifications(
        &self,
        audio: &mut [f32],
        prosody: &ProsodyParameters,
        _sample_rate: u32,
    ) -> Result<()> {
        // Apply energy scaling
        for sample in audio.iter_mut() {
            *sample *= prosody.energy_scale;
        }

        Ok(())
    }

    /// Get conversion model for session
    fn get_conversion_model_for_session(&self, session_id: &str) -> Result<ConversionModel> {
        let sessions = self.sessions.read().unwrap();
        if let Some(session) = sessions.get(session_id) {
            let cache_key = format!(
                "{:?}_{:?}",
                session.source_speaker.vector.iter().sum::<f32>(),
                session.target_speaker.vector.iter().sum::<f32>()
            );

            let cache = self.model_cache.read().unwrap();
            if let Some(model) = cache.get(&cache_key) {
                let mut model = model.clone();
                model.last_used = Instant::now();
                Ok(model)
            } else {
                Err(Error::Processing(
                    "Conversion model not found in cache".to_string(),
                ))
            }
        } else {
            Err(Error::Processing(format!(
                "Session {} not found",
                session_id
            )))
        }
    }

    /// Calculate conversion quality
    fn calculate_conversion_quality(&self, _audio: &[f32], model: &ConversionModel) -> Result<f32> {
        // Simplified quality calculation
        Ok(model.quality_score)
    }

    /// Estimate conversion quality for speaker pair
    fn estimate_conversion_quality(
        &self,
        source: &SpeakerEmbedding,
        target: &SpeakerEmbedding,
    ) -> Result<f32> {
        // Calculate similarity between speakers
        let similarity = self.calculate_speaker_similarity(&source.vector, &target.vector);

        // Quality decreases with dissimilarity, but not linearly
        let quality = 0.5 + (1.0 - similarity) * 0.4; // Map to 0.5-0.9 range
        Ok(quality.clamp(0.1, 1.0))
    }

    /// Calculate speaker similarity
    fn calculate_speaker_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.is_empty() || embedding2.is_empty() {
            return 0.0;
        }

        let min_len = embedding1.len().min(embedding2.len());
        let dot_product = embedding1[..min_len]
            .iter()
            .zip(&embedding2[..min_len])
            .map(|(a, b)| a * b)
            .sum::<f32>();

        let norm1 = embedding1[..min_len]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let norm2 = embedding2[..min_len]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }

    /// Update conversion statistics
    fn update_statistics(&self, latency_ms: f32, quality_score: f32) -> Result<()> {
        let mut stats = self.stats.write().unwrap();

        // Update latency
        let total_samples = stats.total_sessions;
        if total_samples > 0 {
            stats.average_latency_ms = (stats.average_latency_ms * (total_samples - 1) as f32
                + latency_ms)
                / total_samples as f32;
            stats.average_quality = (stats.average_quality * (total_samples - 1) as f32
                + quality_score)
                / total_samples as f32;
        } else {
            stats.average_latency_ms = latency_ms;
            stats.average_quality = quality_score;
        }

        Ok(())
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> ConversionStatistics {
        self.stats.read().unwrap().clone()
    }

    /// Clean up inactive sessions
    pub fn cleanup_inactive_sessions(&self, timeout_duration: Duration) {
        let mut sessions = self.sessions.write().unwrap();
        let now = Instant::now();

        sessions.retain(|_, session| now.duration_since(session.last_activity) < timeout_duration);
    }
}

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub start_time: Instant,
    pub last_activity: Instant,
    pub samples_processed: u64,
    pub current_latency: f32,
    pub quality_score: f32,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_converter_creation() {
        let config = ConversionConfig::default();
        let converter = VoiceConverter::new(config);

        let stats = converter.get_statistics();
        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.active_sessions, 0);
    }

    #[tokio::test]
    async fn test_session_creation() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let source = SpeakerEmbedding::new(vec![0.1; 512]);
        let target = SpeakerEmbedding::new(vec![0.2; 512]);

        let session_id = converter.create_session(source, target).await;
        assert!(session_id.is_ok());

        let stats = converter.get_statistics();
        assert_eq!(stats.active_sessions, 1);
    }

    #[tokio::test]
    async fn test_audio_processing() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let source = SpeakerEmbedding::new(vec![0.1; 512]);
        let target = SpeakerEmbedding::new(vec![0.2; 512]);

        let session_id = converter.create_session(source, target).await.unwrap();

        let audio_chunk = vec![0.1; 1024];
        let result = converter
            .process_chunk(&session_id, &audio_chunk, 16000)
            .await;

        assert!(result.is_ok());
        let conversion_result = result.unwrap();
        assert!(!conversion_result.audio_data.is_empty());
        assert!(conversion_result.latency_ms >= 0.0);
        assert!(conversion_result.quality_score >= 0.0 && conversion_result.quality_score <= 1.0);
    }

    #[test]
    fn test_speaker_similarity_calculation() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![1.0, 0.0, 0.0];
        let embedding3 = vec![0.0, 1.0, 0.0];

        let similarity_same = converter.calculate_speaker_similarity(&embedding1, &embedding2);
        let similarity_different = converter.calculate_speaker_similarity(&embedding1, &embedding3);

        assert!((similarity_same - 1.0).abs() < 1e-6);
        assert!((similarity_different - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_f0_extraction() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let embedding = vec![0.5; 64]; // Should map to ~150 Hz

        let f0 = converter.extract_f0_from_embedding(&embedding).unwrap();
        assert!(f0 > 100.0 && f0 < 200.0);
    }

    #[test]
    fn test_conversion_parameters() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let source = vec![0.1; 128];
        let target = vec![0.2; 128];

        let params = converter.calculate_conversion_parameters(
            &SpeakerEmbedding::new(source),
            &SpeakerEmbedding::new(target),
        );

        assert!(params.is_ok());
        let parameters = params.unwrap();
        assert!(parameters.f0_scale > 0.0);
        assert!(!parameters.formant_shifts.is_empty());
        assert!(!parameters.spectral_envelope.is_empty());
    }

    #[tokio::test]
    async fn test_session_cleanup() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let source = SpeakerEmbedding::new(vec![0.1; 512]);
        let target = SpeakerEmbedding::new(vec![0.2; 512]);

        let session_id = converter.create_session(source, target).await.unwrap();
        assert_eq!(converter.get_statistics().active_sessions, 1);

        // Close session
        converter.close_session(&session_id).unwrap();
        assert_eq!(converter.get_statistics().active_sessions, 0);
    }

    #[test]
    fn test_temporal_scaling() {
        let converter = VoiceConverter::new(ConversionConfig::default());
        let audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Test time stretching (slower)
        let stretched = converter.apply_temporal_scaling(&audio, 0.5).unwrap();
        assert!(stretched.len() > audio.len());

        // Test time compression (faster)
        let compressed = converter.apply_temporal_scaling(&audio, 2.0).unwrap();
        assert!(compressed.len() < audio.len());
    }
}
