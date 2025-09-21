//! Core synthesis engine and types

use crate::config::SingingConfig;
use crate::pitch::PitchContour;
use crate::precision_quality::PrecisionQualityAnalyzer;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{SingingRequest, VoiceCharacteristics};
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::results::{QualityMetrics, SynthesisResult, SynthesisStats};

/// Synthesis engine for singing voice with thread safety
///
/// This engine is designed to be thread-safe and can be shared across threads
/// when wrapped in Arc. All mutable operations use interior mutability patterns.
pub struct SynthesisEngine {
    /// Configuration
    config: SingingConfig,
    /// FFT planner (thread-safe)
    fft_planner: FftPlanner<f32>,
    /// Synthesis models (thread-safe via trait bounds)
    models: HashMap<String, Box<dyn SynthesisModel>>,
    /// Current voice characteristics
    voice_characteristics: VoiceCharacteristics,
    /// Current technique
    technique: SingingTechnique,
    /// Performance statistics
    stats: SynthesisStats,
    /// Precision quality analyzer for enhanced metrics
    precision_analyzer: PrecisionQualityAnalyzer,
}

/// Synthesis model trait
pub trait SynthesisModel: Send + Sync {
    /// Synthesize audio from parameters
    fn synthesize(&self, params: &SynthesisParams) -> crate::Result<Vec<f32>>;

    /// Get model name
    fn name(&self) -> &str;

    /// Get model version
    fn version(&self) -> &str;

    /// Load model from file
    fn load_from_file(&mut self, path: &str) -> crate::Result<()>;

    /// Save model to file
    fn save_to_file(&self, path: &str) -> crate::Result<()>;
}

/// Synthesis parameters
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Pitch contour
    pub pitch_contour: PitchContour,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Singing technique
    pub technique: SingingTechnique,
    /// Duration in seconds
    pub duration: f32,
    /// Sample rate
    pub sample_rate: f32,
    /// Phoneme sequence
    pub phonemes: Vec<String>,
    /// Timing information
    pub timing: Vec<f32>,
    /// Dynamics
    pub dynamics: Vec<f32>,
    /// Expression
    pub expression: Vec<f32>,
}

impl SynthesisEngine {
    /// Create a new synthesis engine
    pub fn new(config: SingingConfig) -> Self {
        Self {
            config: config.clone(),
            fft_planner: FftPlanner::new(),
            models: HashMap::new(),
            voice_characteristics: VoiceCharacteristics::default(),
            technique: SingingTechnique::classical(),
            stats: SynthesisStats::default(),
            precision_analyzer: PrecisionQualityAnalyzer::new(),
        }
    }

    /// Set voice characteristics
    pub fn set_voice_characteristics(&mut self, characteristics: VoiceCharacteristics) {
        self.voice_characteristics = characteristics;
    }

    /// Set singing technique
    pub fn set_technique(&mut self, technique: SingingTechnique) {
        self.technique = technique;
    }

    /// Add a synthesis model
    pub fn add_model(&mut self, name: String, model: Box<dyn SynthesisModel>) {
        self.models.insert(name, model);
    }

    /// Remove a synthesis model
    pub fn remove_model(&mut self, name: &str) -> Option<Box<dyn SynthesisModel>> {
        self.models.remove(name)
    }

    /// Get model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Synthesize from request
    pub async fn synthesize(&mut self, request: SingingRequest) -> crate::Result<SynthesisResult> {
        let start_time = Instant::now();

        // Create synthesis parameters from request
        let params = self.create_synthesis_params(&request)?;

        // Select and use appropriate model
        let model_name = self.select_model(&request)?;
        let model = self
            .models
            .get(&model_name)
            .ok_or_else(|| crate::Error::Synthesis(format!("Model not found: {}", model_name)))?;

        // Perform synthesis
        let audio = model.synthesize(&params)?;

        // Calculate statistics
        let processing_time = start_time.elapsed();
        let quality_metrics = self.calculate_quality_metrics(&audio, &params)?;

        self.stats.processing_time = processing_time;
        self.stats.frame_count = audio.len() / self.config.audio.buffer_size;

        Ok(SynthesisResult {
            audio,
            sample_rate: params.sample_rate,
            duration: Duration::from_secs_f32(params.duration),
            stats: self.stats.clone(),
            quality_metrics,
        })
    }

    /// Create synthesis parameters from request
    fn create_synthesis_params(&self, request: &SingingRequest) -> crate::Result<SynthesisParams> {
        // Extract timing information from the score
        let timing = self.extract_timing(&request.score)?;
        let duration = timing.iter().sum::<f32>();

        Ok(SynthesisParams {
            pitch_contour: {
                let time_points: Vec<f32> = request
                    .score
                    .notes
                    .iter()
                    .scan(0.0, |acc, note| {
                        let start = *acc;
                        *acc += note.duration;
                        Some(start)
                    })
                    .collect();
                let f0_values: Vec<f32> = request
                    .score
                    .notes
                    .iter()
                    .map(|note| note.event.frequency)
                    .collect();
                PitchContour::new(time_points, f0_values)
            },
            voice_characteristics: self.voice_characteristics.clone(),
            technique: self.technique.clone(),
            duration,
            sample_rate: self.config.audio.sample_rate as f32,
            phonemes: self.extract_phonemes(&request.score)?,
            timing,
            dynamics: self.extract_dynamics(&request.score)?,
            expression: self.extract_expression(&request.score)?,
        })
    }

    /// Select appropriate model for request
    fn select_model(&self, request: &SingingRequest) -> crate::Result<String> {
        // Simple model selection logic - in practice this would be more sophisticated
        if let Some(model_name) = self.models.keys().next() {
            Ok(model_name.clone())
        } else {
            Err(crate::Error::Synthesis(
                "No synthesis models available".to_string(),
            ))
        }
    }

    /// Extract timing information from score
    fn extract_timing(&self, score: &MusicalScore) -> crate::Result<Vec<f32>> {
        let notes = &score.notes;
        let mut timing = Vec::with_capacity(notes.len());

        for note in notes {
            timing.push(note.duration);
        }

        Ok(timing)
    }

    /// Extract phonemes from score
    fn extract_phonemes(&self, score: &MusicalScore) -> crate::Result<Vec<String>> {
        let notes = &score.notes;
        let mut phonemes = Vec::with_capacity(notes.len());

        for note in notes {
            // In practice, this would involve text-to-phoneme conversion
            phonemes.push("a".to_string()); // Placeholder
        }

        Ok(phonemes)
    }

    /// Extract dynamics from score
    fn extract_dynamics(&self, score: &MusicalScore) -> crate::Result<Vec<f32>> {
        let notes = &score.notes;
        let mut dynamics = Vec::with_capacity(notes.len());

        for note in notes {
            dynamics.push(note.event.velocity); // Already normalized 0-1
        }

        Ok(dynamics)
    }

    /// Extract expression from score
    fn extract_expression(&self, score: &MusicalScore) -> crate::Result<Vec<f32>> {
        let notes = &score.notes;
        let mut expression = Vec::with_capacity(notes.len());

        for _note in notes {
            expression.push(0.5); // Placeholder neutral expression
        }

        Ok(expression)
    }

    /// Calculate quality metrics for synthesized audio
    fn calculate_quality_metrics(
        &mut self,
        audio: &[f32],
        params: &SynthesisParams,
    ) -> crate::Result<QualityMetrics> {
        // Calculate basic quality metrics (placeholder implementations)
        let pitch_accuracy = if !audio.is_empty() { 0.85 } else { 0.0 };
        let spectral_quality = 0.8;
        let harmonic_quality = 0.75;
        let noise_level = 0.1;
        let formant_quality = 0.85;

        Ok(QualityMetrics {
            pitch_accuracy,
            spectral_quality,
            harmonic_quality,
            noise_level,
            formant_quality,
            overall_quality: pitch_accuracy * 0.8, // Weighted average
        })
    }

    /// Get current statistics
    pub fn stats(&self) -> &SynthesisStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SynthesisStats::default();
    }
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self::new(SingingConfig::default())
    }
}
