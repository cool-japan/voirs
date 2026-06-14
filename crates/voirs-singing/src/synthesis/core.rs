//! Core synthesis engine and types

use crate::config::SingingConfig;
use crate::pitch::PitchContour;
use crate::precision_quality::PrecisionQualityAnalyzer;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{SingingRequest, VoiceCharacteristics};
use scirs2_fft::{FftPlanner, RealFftPlanner};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::quality_dsp;
use super::results::{QualityMetrics, SynthesisResult, SynthesisStats};

/// Cent deviation from the target pitch that maps the cents component of
/// pitch accuracy to `0.0` (one equal-tempered semitone).
const PITCH_CENTS_TOLERANCE: f32 = 100.0;

/// Synthesis engine for singing voice with thread safety
///
/// This engine is designed to be thread-safe and can be shared across threads
/// when wrapped in Arc. All mutable operations use interior mutability patterns.
pub struct SynthesisEngine {
    /// Configuration for singing synthesis
    config: SingingConfig,
    /// FFT planner for spectral processing (thread-safe)
    fft_planner: FftPlanner,
    /// Registered synthesis models mapped by name (thread-safe via trait bounds)
    models: HashMap<String, Box<dyn SynthesisModel>>,
    /// Current voice characteristics for synthesis
    voice_characteristics: VoiceCharacteristics,
    /// Current singing technique being applied
    technique: SingingTechnique,
    /// Performance statistics tracking
    stats: SynthesisStats,
    /// Precision quality analyzer for enhanced metrics analysis
    precision_analyzer: PrecisionQualityAnalyzer,
}

/// Synthesis model trait for audio generation
///
/// All synthesis models must be Send + Sync for thread-safe operation.
pub trait SynthesisModel: Send + Sync {
    /// Synthesize audio from parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Synthesis parameters including pitch, timing, and dynamics
    ///
    /// # Returns
    ///
    /// Vector of audio samples as f32 values in range [-1.0, 1.0]
    ///
    /// # Errors
    ///
    /// Returns an error if synthesis fails or parameters are invalid
    fn synthesize(&self, params: &SynthesisParams) -> crate::Result<Vec<f32>>;

    /// Get model name identifier
    ///
    /// # Returns
    ///
    /// String reference to the model name
    fn name(&self) -> &str;

    /// Get model version string
    ///
    /// # Returns
    ///
    /// String reference to the version (e.g., "1.0.0")
    fn version(&self) -> &str;

    /// Load model parameters from file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load model parameters from
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parameters are invalid
    fn load_from_file(&mut self, path: &str) -> crate::Result<()>;

    /// Save model parameters to file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save model parameters to
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    fn save_to_file(&self, path: &str) -> crate::Result<()>;
}

/// Synthesis parameters for audio generation
///
/// Contains all necessary information to synthesize singing voice audio.
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Pitch contour over time with F0 values in Hz
    pub pitch_contour: PitchContour,
    /// Voice characteristics including range, timbre, and vibrato
    pub voice_characteristics: VoiceCharacteristics,
    /// Singing technique to apply (classical, pop, jazz, etc.)
    pub technique: SingingTechnique,
    /// Total duration of synthesis in seconds
    pub duration: f32,
    /// Audio sample rate in Hz (e.g., 44100.0)
    pub sample_rate: f32,
    /// Phoneme sequence for linguistic control
    pub phonemes: Vec<String>,
    /// Timing information for each phoneme in seconds
    pub timing: Vec<f32>,
    /// Dynamic levels (0.0-1.0) for volume control over time
    pub dynamics: Vec<f32>,
    /// Expression parameters (0.0-1.0) for emotional control
    pub expression: Vec<f32>,
}

impl SynthesisEngine {
    /// Create a new synthesis engine
    ///
    /// # Arguments
    ///
    /// * `config` - Singing configuration for the engine
    ///
    /// # Returns
    ///
    /// A new SynthesisEngine instance with default settings
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

    /// Set voice characteristics for synthesis
    ///
    /// # Arguments
    ///
    /// * `characteristics` - Voice characteristics to apply to future synthesis
    pub fn set_voice_characteristics(&mut self, characteristics: VoiceCharacteristics) {
        self.voice_characteristics = characteristics;
    }

    /// Set singing technique
    ///
    /// # Arguments
    ///
    /// * `technique` - Singing technique to apply (classical, pop, jazz, etc.)
    pub fn set_technique(&mut self, technique: SingingTechnique) {
        self.technique = technique;
    }

    /// Add a synthesis model to the engine
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the model
    /// * `model` - Boxed synthesis model implementation
    pub fn add_model(&mut self, name: String, model: Box<dyn SynthesisModel>) {
        self.models.insert(name, model);
    }

    /// Remove a synthesis model from the engine
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the model to remove
    ///
    /// # Returns
    ///
    /// The removed model if it existed, None otherwise
    pub fn remove_model(&mut self, name: &str) -> Option<Box<dyn SynthesisModel>> {
        self.models.remove(name)
    }

    /// Get list of registered model names
    ///
    /// # Returns
    ///
    /// Vector of model name strings
    pub fn model_names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Synthesize audio from a singing request
    ///
    /// # Arguments
    ///
    /// * `request` - Singing request containing score and synthesis parameters
    ///
    /// # Returns
    ///
    /// SynthesisResult containing audio samples, quality metrics, and statistics
    ///
    /// # Errors
    ///
    /// Returns an error if no models are available or synthesis fails
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

    /// Calculate quality metrics for synthesized audio.
    ///
    /// Performs real signal analysis on the synthesized `audio` buffer (rather
    /// than returning fixed constants):
    ///
    /// * `pitch_accuracy` - autocorrelation F0 tracking compared, in cents,
    ///   against the intended note pitches in `params.pitch_contour`; when no
    ///   target pitches are present (or none can be matched) it falls back to raw
    ///   F0 stability.
    /// * `harmonic_quality` - harmonic-to-noise ratio (HNR) mapped to `[0, 1]`.
    /// * `noise_level` - aperiodic fraction (`1 - harmonic ratio`).
    /// * `spectral_quality` - spectral-envelope tonality and centroid
    ///   plausibility.
    /// * `formant_quality` - LPC all-pole envelope prominence / clarity.
    fn calculate_quality_metrics(
        &mut self,
        audio: &[f32],
        params: &SynthesisParams,
    ) -> crate::Result<QualityMetrics> {
        // Empty audio carries no measurable quality.
        if audio.is_empty() {
            return Ok(QualityMetrics::default());
        }

        let sample_rate = params.sample_rate;

        // Pitch accuracy: prefer target note pitches, fall back to F0 stability.
        let pitch_accuracy = if params.pitch_contour.f0_values.is_empty() {
            quality_dsp::f0_stability_score(audio, sample_rate)
        } else {
            let report = self.precision_analyzer.calculate_precision_pitch_accuracy(
                audio,
                &params.pitch_contour,
                sample_rate,
            )?;
            if report.total_notes == 0 {
                // No measured F0 matched a target pitch (e.g. unvoiced/noisy
                // output); score raw stability instead.
                quality_dsp::f0_stability_score(audio, sample_rate)
            } else {
                let cents_score =
                    1.0 - (report.mean_cent_deviation / PITCH_CENTS_TOLERANCE).clamp(0.0, 1.0);
                (0.6 * cents_score + 0.4 * report.pitch_stability).clamp(0.0, 1.0)
            }
        };

        // Harmonicity drives both harmonic quality (HNR in dB) and noise level
        // (the complementary aperiodic energy fraction).
        let harmonicity = quality_dsp::estimate_harmonicity(audio, sample_rate);
        let harmonic_quality = quality_dsp::harmonic_quality_from_hnr(harmonicity.hnr_db);
        let noise_level = (1.0 - harmonicity.harmonic_ratio).clamp(0.0, 1.0);

        // Spectral-envelope tonality / centroid plausibility.
        let spectral_quality = quality_dsp::spectral_quality_score(audio, sample_rate);

        // LPC-based formant prominence / clarity.
        let formant_quality = quality_dsp::formant_clarity_score(audio, sample_rate);

        // Weighted blend across all measured dimensions.
        let overall_quality = (0.30 * pitch_accuracy
            + 0.20 * harmonic_quality
            + 0.20 * spectral_quality
            + 0.15 * formant_quality
            + 0.15 * (1.0 - noise_level))
            .clamp(0.0, 1.0);

        Ok(QualityMetrics {
            pitch_accuracy,
            spectral_quality,
            harmonic_quality,
            noise_level,
            formant_quality,
            overall_quality,
        })
    }

    /// Get current performance statistics
    ///
    /// # Returns
    ///
    /// Reference to the current synthesis statistics
    pub fn stats(&self) -> &SynthesisStats {
        &self.stats
    }

    /// Reset performance statistics to default values
    pub fn reset_stats(&mut self) {
        self.stats = SynthesisStats::default();
    }
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self::new(SingingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 22_050.0;
    const LEN: usize = 11_025; // 0.5 s

    /// Build synthesis parameters carrying a constant target pitch contour.
    fn params_with_target(target_hz: f32) -> SynthesisParams {
        SynthesisParams {
            pitch_contour: PitchContour::new(vec![0.0; 10], vec![target_hz; 10]),
            voice_characteristics: VoiceCharacteristics::default(),
            technique: SingingTechnique::classical(),
            duration: LEN as f32 / SAMPLE_RATE,
            sample_rate: SAMPLE_RATE,
            phonemes: Vec::new(),
            timing: Vec::new(),
            dynamics: Vec::new(),
            expression: Vec::new(),
        }
    }

    /// Generate a pure sine tone.
    fn sine(freq: f32, amplitude: f32) -> Vec<f32> {
        (0..LEN)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Deterministic zero-mean pseudo-random noise (LCG, no `rand` crate).
    fn lcg_noise(seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..LEN)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32;
                0.8 * (2.0 * unit - 1.0)
            })
            .collect()
    }

    #[test]
    fn test_clean_tone_scores_high_pitch_and_harmonic_low_noise() {
        let mut engine = SynthesisEngine::default();
        let audio = sine(440.0, 1.0);
        let params = params_with_target(440.0);

        let metrics = engine
            .calculate_quality_metrics(&audio, &params)
            .expect("quality metrics");

        assert!(
            metrics.pitch_accuracy > 0.8,
            "pitch accuracy {}",
            metrics.pitch_accuracy
        );
        assert!(
            metrics.harmonic_quality > 0.7,
            "harmonic quality {}",
            metrics.harmonic_quality
        );
        assert!(
            metrics.noise_level < 0.2,
            "noise level {}",
            metrics.noise_level
        );
        assert!(
            metrics.spectral_quality > 0.5,
            "spectral quality {}",
            metrics.spectral_quality
        );
        assert!(metrics.overall_quality > 0.6);
        assert!((0.0..=1.0).contains(&metrics.formant_quality));
    }

    #[test]
    fn test_white_noise_scores_low_harmonic_high_noise() {
        let mut engine = SynthesisEngine::default();
        let audio = lcg_noise(0x1234_5678);
        let params = params_with_target(440.0);

        let metrics = engine
            .calculate_quality_metrics(&audio, &params)
            .expect("quality metrics");

        assert!(
            metrics.harmonic_quality < 0.2,
            "harmonic quality {}",
            metrics.harmonic_quality
        );
        assert!(
            metrics.noise_level > 0.5,
            "noise level {}",
            metrics.noise_level
        );
        // A clean tone must out-score noise on pitch accuracy.
        let tone_metrics = SynthesisEngine::default()
            .calculate_quality_metrics(&sine(440.0, 1.0), &params)
            .expect("tone metrics");
        assert!(
            tone_metrics.pitch_accuracy > metrics.pitch_accuracy,
            "tone {} should beat noise {}",
            tone_metrics.pitch_accuracy,
            metrics.pitch_accuracy
        );
    }

    #[test]
    fn test_empty_audio_returns_default_metrics() {
        let mut engine = SynthesisEngine::default();
        let params = params_with_target(440.0);
        let metrics = engine
            .calculate_quality_metrics(&[], &params)
            .expect("quality metrics");
        assert_eq!(metrics.overall_quality, 0.0);
        assert_eq!(metrics.pitch_accuracy, 0.0);
    }
}
