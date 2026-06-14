//! Singing voice synthesis integration for VoiRS SDK

use crate::{Result, VoirsError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Musical note representation
#[derive(Debug, Clone, PartialEq)]
pub struct MusicalNote {
    /// Note name (C, D, E, F, G, A, B)
    pub note: String,
    /// Octave (0-8)
    pub octave: u8,
    /// Pitch in Hz
    pub frequency: f32,
    /// Duration in beats
    pub duration: f32,
    /// Velocity (0.0-1.0)
    pub velocity: f32,
    /// Vibrato intensity (0.0-1.0)
    pub vibrato: f32,
}

/// Singing technique parameters
#[derive(Debug, Clone)]
pub struct SingingTechnique {
    /// Breath control intensity (0.0-1.0)
    pub breath_control: f32,
    /// Vocal fry amount (0.0-1.0)
    pub vocal_fry: f32,
    /// Head voice ratio (0.0-1.0, vs chest voice)
    pub head_voice_ratio: f32,
    /// Vibrato speed (Hz)
    pub vibrato_speed: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
    /// Pitch bend sensitivity (0.0-1.0)
    pub pitch_bend: f32,
    /// Legato vs staccato (0.0-1.0)
    pub legato: f32,
}

/// Musical score containing notes and timing
#[derive(Debug, Clone)]
pub struct MusicalScore {
    /// List of notes with timing
    pub notes: Vec<MusicalNote>,
    /// Beats per minute
    pub tempo: f32,
    /// Time signature numerator
    pub time_signature_num: u8,
    /// Time signature denominator
    pub time_signature_den: u8,
    /// Key signature
    pub key_signature: String,
}

/// Singing synthesis configuration
#[derive(Debug, Clone)]
pub struct SingingConfig {
    /// Enable singing mode
    pub enabled: bool,
    /// Voice type (soprano, alto, tenor, bass)
    pub voice_type: VoiceType,
    /// Default singing technique
    pub technique: SingingTechnique,
    /// Auto-detect notes from text
    pub auto_pitch_detection: bool,
    /// Cache musical scores
    pub cache_scores: bool,
}

/// Voice type for singing
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceType {
    Soprano,
    Alto,
    Tenor,
    Bass,
}

/// Singing synthesis result
#[derive(Debug, Clone)]
pub struct SingingResult {
    /// Synthesized singing audio
    pub audio: crate::audio::AudioBuffer,
    /// Applied musical score
    pub score: MusicalScore,
    /// Singing technique used
    pub technique: SingingTechnique,
    /// Performance statistics
    pub stats: SingingStats,
}

/// Performance statistics for singing
#[derive(Debug, Clone)]
pub struct SingingStats {
    /// Total notes sung
    pub total_notes: usize,
    /// Average pitch accuracy
    pub pitch_accuracy: f32,
    /// Vibrato consistency
    pub vibrato_consistency: f32,
    /// Breath control quality
    pub breath_quality: f32,
}

/// SDK-integrated singing controller
#[derive(Debug, Clone)]
pub struct SingingController {
    /// Internal singing processor (mock for now)
    config: Arc<RwLock<SingingConfig>>,
    /// Cached musical scores
    score_cache: Arc<RwLock<HashMap<String, MusicalScore>>>,
}

impl SingingController {
    /// Create new singing controller
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(SingingConfig::default())),
            score_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: SingingConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            score_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Set singing technique
    pub async fn set_technique(&self, technique: SingingTechnique) -> Result<()> {
        let mut config = self.config.write().await;
        config.technique = technique;
        Ok(())
    }

    /// Set voice type
    pub async fn set_voice_type(&self, voice_type: VoiceType) -> Result<()> {
        let mut config = self.config.write().await;
        config.voice_type = voice_type;
        Ok(())
    }

    /// Synthesize singing from musical score
    pub async fn synthesize_score(&self, score: MusicalScore, text: &str) -> Result<SingingResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing synthesis is disabled".to_string(),
            });
        }

        // Cache the score
        {
            let mut cache = self.score_cache.write().await;
            cache.insert(text.to_string(), score.clone());
        }

        // Additive DSP synthesis of the notated score.
        let audio = self
            .synthesize_notes(&score.notes, &config.technique)
            .await?;

        // Real, audio-derived quality statistics measured from the synthesized
        // waveform (see `compute_singing_stats` for the algorithms used).
        let stats = Self::compute_singing_stats(&audio, &score, &config.technique);

        Ok(SingingResult {
            audio,
            score: score.clone(),
            technique: config.technique.clone(),
            stats,
        })
    }

    /// Synthesize from text with automatic pitch detection
    pub async fn synthesize_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> Result<SingingResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "singing".to_string(),
                message: "Singing synthesis is disabled".to_string(),
            });
        }

        // Auto-generate musical score from text
        let score = self.generate_score_from_text(text, key, tempo).await?;
        self.synthesize_score(score, text).await
    }

    /// Apply singing preset
    pub async fn apply_preset(&self, preset_name: &str) -> Result<()> {
        let technique = match preset_name {
            "classical" => SingingTechnique {
                breath_control: 0.9,
                vocal_fry: 0.1,
                head_voice_ratio: 0.7,
                vibrato_speed: 6.0,
                vibrato_depth: 0.8,
                pitch_bend: 0.3,
                legato: 0.9,
            },
            "pop" => SingingTechnique {
                breath_control: 0.7,
                vocal_fry: 0.3,
                head_voice_ratio: 0.5,
                vibrato_speed: 4.5,
                vibrato_depth: 0.5,
                pitch_bend: 0.6,
                legato: 0.6,
            },
            "jazz" => SingingTechnique {
                breath_control: 0.8,
                vocal_fry: 0.4,
                head_voice_ratio: 0.6,
                vibrato_speed: 5.5,
                vibrato_depth: 0.7,
                pitch_bend: 0.8,
                legato: 0.5,
            },
            "opera" => SingingTechnique {
                breath_control: 1.0,
                vocal_fry: 0.0,
                head_voice_ratio: 0.8,
                vibrato_speed: 7.0,
                vibrato_depth: 0.9,
                pitch_bend: 0.2,
                legato: 1.0,
            },
            _ => {
                return Err(VoirsError::ConfigError {
                    field: "preset".to_string(),
                    message: format!("Unknown singing preset: {}", preset_name),
                })
            }
        };

        self.set_technique(technique).await
    }

    /// Get current singing configuration
    pub async fn get_config(&self) -> SingingConfig {
        self.config.read().await.clone()
    }

    /// Enable or disable singing synthesis
    pub async fn set_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if singing synthesis is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// List available singing presets
    pub fn list_presets(&self) -> Vec<String> {
        vec![
            "classical".to_string(),
            "pop".to_string(),
            "jazz".to_string(),
            "opera".to_string(),
        ]
    }

    /// Parse musical score from text format
    pub async fn parse_score(&self, score_text: &str) -> Result<MusicalScore> {
        // Mock implementation - in reality would parse formats like MusicXML, MIDI, etc.
        let lines: Vec<&str> = score_text.lines().collect();
        let mut notes = Vec::new();
        let mut tempo = 120.0;
        let mut key_signature = "C".to_string();

        for line in lines {
            if line.starts_with("TEMPO:") {
                if let Some(tempo_str) = line.split(':').nth(1) {
                    tempo = tempo_str.trim().parse().unwrap_or(120.0);
                }
            } else if line.starts_with("KEY:") {
                if let Some(key_str) = line.split(':').nth(1) {
                    key_signature = key_str.trim().to_string();
                }
            } else if line.starts_with("NOTE:") {
                // Parse note format: NOTE: C4 0.5 0.8 0.3
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let note_name = parts[1].chars().next().unwrap_or('C').to_string();
                    let octave = parts[1]
                        .chars()
                        .nth(1)
                        .and_then(|c| c.to_digit(10))
                        .unwrap_or(4) as u8;
                    let duration = parts[2].parse().unwrap_or(0.5);
                    let velocity = parts[3].parse().unwrap_or(0.8);
                    let vibrato = parts[4].parse().unwrap_or(0.3);

                    let frequency = self.note_to_frequency(&note_name, octave);
                    notes.push(MusicalNote {
                        note: note_name,
                        octave,
                        frequency,
                        duration,
                        velocity,
                        vibrato,
                    });
                }
            }
        }

        Ok(MusicalScore {
            notes,
            tempo,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature,
        })
    }

    /// Get cached musical score
    pub async fn get_cached_score(&self, text: &str) -> Option<MusicalScore> {
        let cache = self.score_cache.read().await;
        cache.get(text).cloned()
    }

    /// Clear score cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.score_cache.write().await;
        cache.clear();
        Ok(())
    }

    // Private helper methods

    /// Convert note name and octave to frequency
    fn note_to_frequency(&self, note: &str, octave: u8) -> f32 {
        let base_frequencies = HashMap::from([
            ("C", 261.63),
            ("D", 293.66),
            ("E", 329.63),
            ("F", 349.23),
            ("G", 392.00),
            ("A", 440.00),
            ("B", 493.88),
        ]);

        let base_freq = base_frequencies.get(note).copied().unwrap_or(440.0);
        base_freq * 2.0_f32.powi(octave as i32 - 4)
    }

    /// Generate musical score from text
    async fn generate_score_from_text(
        &self,
        text: &str,
        key: &str,
        tempo: f32,
    ) -> Result<MusicalScore> {
        // Mock implementation - in reality would use advanced text-to-melody algorithms
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut notes = Vec::new();
        let note_names = ["C", "D", "E", "F", "G", "A", "B"];

        for (i, word) in words.iter().enumerate() {
            let note_name = note_names[i % note_names.len()];
            let octave = 4 + (i / note_names.len()) as u8;
            let duration = 0.5 + (word.len() as f32 * 0.1);
            let frequency = self.note_to_frequency(note_name, octave);

            notes.push(MusicalNote {
                note: note_name.to_string(),
                octave,
                frequency,
                duration,
                velocity: 0.8,
                vibrato: 0.4,
            });
        }

        Ok(MusicalScore {
            notes,
            tempo,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature: key.to_string(),
        })
    }

    /// Synthesize notes into audio with realistic singing characteristics
    async fn synthesize_notes(
        &self,
        notes: &[MusicalNote],
        technique: &SingingTechnique,
    ) -> Result<crate::audio::AudioBuffer> {
        let sample_rate = 44100;
        let mut audio_samples = Vec::new();

        for note in notes {
            let note_duration = note.duration;
            let samples_per_note = (note_duration * sample_rate as f32) as usize;

            for i in 0..samples_per_note {
                let t = i as f32 / sample_rate as f32;
                let note_phase = i as f32 / samples_per_note as f32;

                // ADSR envelope for natural attack and release
                let envelope = self.calculate_adsr_envelope(note_phase, note_duration, technique);

                // Vibrato modulation
                let vibrato_freq = technique.vibrato_speed;
                let vibrato_depth = technique.vibrato_depth * note.vibrato;
                let vibrato_mod =
                    1.0 + vibrato_depth * (2.0 * std::f32::consts::PI * vibrato_freq * t).sin();

                let frequency = note.frequency * vibrato_mod;

                // Fundamental frequency + harmonics for richer sound
                let fundamental = (2.0 * std::f32::consts::PI * frequency * t).sin();

                // Add harmonics with decreasing amplitude (creates vocal timbre)
                let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * frequency * 2.0 * t).sin();
                let harmonic3 = 0.25 * (2.0 * std::f32::consts::PI * frequency * 3.0 * t).sin();
                let harmonic4 = 0.125 * (2.0 * std::f32::consts::PI * frequency * 4.0 * t).sin();

                // Apply head voice vs chest voice mixing
                let harmonic_mix = fundamental
                    + harmonic2 * (1.0 - technique.head_voice_ratio * 0.5)
                    + harmonic3 * (1.0 - technique.head_voice_ratio * 0.7)
                    + harmonic4 * (1.0 - technique.head_voice_ratio * 0.9);

                // Apply formant-like filtering for vowel characteristics
                let formant_enhanced = self.apply_formant_enhancement(
                    harmonic_mix,
                    frequency,
                    technique.head_voice_ratio,
                );

                // Apply velocity and envelope
                let mut sample = formant_enhanced * note.velocity * envelope;

                // Add breath noise for realism
                let breath_noise = self.generate_breath_noise(note_phase, technique);
                sample += breath_noise * (1.0 - technique.breath_control);

                // Apply vocal fry effect in lower register
                if frequency < 150.0 && technique.vocal_fry > 0.0 {
                    let fry_freq = frequency * 0.5;
                    let fry = technique.vocal_fry
                        * 0.1
                        * (2.0 * std::f32::consts::PI * fry_freq * t).sin();
                    sample += fry * envelope;
                }

                // Pitch bend effect for smooth transitions
                let bend_amount = technique.pitch_bend * 0.1 * note_phase.sin();
                sample *= 1.0 + bend_amount;

                // Apply breath control (reduces amplitude, increases breathiness)
                let processed_sample = sample * technique.breath_control;
                audio_samples.push(processed_sample.clamp(-1.0, 1.0));
            }

            // Add subtle pause between notes if not full legato
            if technique.legato < 1.0 {
                let pause_samples = ((1.0 - technique.legato) * sample_rate as f32 * 0.05) as usize;
                audio_samples.resize(audio_samples.len() + pause_samples, 0.0);
            }
        }

        Ok(crate::audio::AudioBuffer::mono(audio_samples, sample_rate))
    }

    /// Calculate ADSR envelope for natural note articulation
    fn calculate_adsr_envelope(
        &self,
        phase: f32,
        duration: f32,
        technique: &SingingTechnique,
    ) -> f32 {
        let attack_time = 0.05; // 50ms attack
        let decay_time = 0.1; // 100ms decay
        let sustain_level = 0.8;
        let release_start = 0.85; // Start release at 85% of note duration

        if phase < attack_time / duration {
            // Attack phase - smooth cubic curve
            let attack_phase = phase / (attack_time / duration);
            attack_phase * attack_phase * (3.0 - 2.0 * attack_phase)
        } else if phase < (attack_time + decay_time) / duration {
            // Decay phase
            let decay_phase = (phase - attack_time / duration) / (decay_time / duration);
            1.0 - (1.0 - sustain_level) * decay_phase
        } else if phase < release_start {
            // Sustain phase
            sustain_level
        } else {
            // Release phase - smooth exponential-like decay
            let release_phase = (phase - release_start) / (1.0 - release_start);
            sustain_level * (1.0 - release_phase).powf(2.0)
        }
    }

    /// Apply formant-like enhancement to simulate vowel characteristics
    fn apply_formant_enhancement(&self, signal: f32, frequency: f32, head_voice: f32) -> f32 {
        // Simple formant boost simulation
        // In reality, this would use proper formant filtering
        let formant_boost = if frequency > 200.0 && frequency < 800.0 {
            1.2 * (1.0 + head_voice * 0.3) // Boost mid frequencies for vowel clarity
        } else if frequency > 2000.0 {
            0.8 * (1.0 - head_voice * 0.2) // Reduce high frequencies in chest voice
        } else {
            1.0
        };

        signal * formant_boost
    }

    /// Generate breath noise for singing realism
    fn generate_breath_noise(&self, phase: f32, technique: &SingingTechnique) -> f32 {
        use fastrand;

        // More breath noise at note transitions (attack and release)
        let noise_intensity = if phase < 0.1 || phase > 0.9 {
            0.02
        } else {
            0.005
        };

        // Generate pink-ish noise (more natural than white noise)
        let white_noise = (fastrand::f32() * 2.0 - 1.0) * noise_intensity;

        // Simple low-pass filter for pink-ish characteristics
        white_noise * (1.0 - technique.breath_control * 0.5)
    }

    /// Compute real, audio-derived singing quality statistics from the
    /// synthesized waveform and the notated score.
    ///
    /// All three metrics are measured from the actual audio buffer (which is in
    /// scope here), not hardcoded:
    ///
    /// * `pitch_accuracy` — derived from a per-frame autocorrelation F0 track.
    ///   Combines (a) the *centering* error of each note (cents distance between
    ///   the median realized F0 and the notated pitch) and (b) *jitter* (fast
    ///   cycle-to-cycle F0 deviation, isolated via a local second-difference that
    ///   removes smooth vibrato). The combined cents error is mapped through a
    ///   tolerance to `[0, 1]`. Heavy F0 jitter therefore lowers the score while
    ///   a steady tone scores near 1.0.
    /// * `vibrato_consistency` — **audio-derived** from the regularity of the F0
    ///   modulation: the detrended, Hann-windowed voiced-F0 contour is run
    ///   through an FFT (`scirs2_fft`) and the fraction of spectral energy that
    ///   falls inside the 4–8 Hz vibrato band is taken as the consistency (a
    ///   sharp, clean modulation peak → high score). If notes are too short for a
    ///   reliable spectrum, it **falls back to a score-derived** estimate from the
    ///   regularity of the notated vibrato intensities (documented as such).
    /// * `breath_quality` — **audio-derived** from the periodicity
    ///   (harmonic-to-noise proxy) of the *low-energy* frames (note onsets/
    ///   offsets and transitions), which is where breath noise concentrates:
    ///   a well-supported voice stays periodic even when quiet, whereas
    ///   breathiness injects aperiodic noise and lowers the score.
    fn compute_singing_stats(
        audio: &crate::audio::AudioBuffer,
        score: &MusicalScore,
        technique: &SingingTechnique,
    ) -> SingingStats {
        let total_notes = score.notes.len();
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        if samples.is_empty() || total_notes == 0 {
            return SingingStats {
                total_notes,
                pitch_accuracy: 0.0,
                vibrato_consistency: 0.0,
                breath_quality: 0.0,
            };
        }

        let frame_rate = sample_rate as f32 / ANALYSIS_HOP_SIZE as f32;
        let segments = note_segments(&score.notes, technique.legato, sample_rate);

        let mut centering_cents: Vec<f32> = Vec::new();
        let mut jitter_cents: Vec<f32> = Vec::new();
        let mut vibrato_scores: Vec<f32> = Vec::new();
        let mut all_frames: Vec<FrameInfo> = Vec::new();

        for (idx, &(start, end)) in segments.iter().enumerate() {
            let start = start.min(samples.len());
            let end = end.min(samples.len());
            if end <= start {
                continue;
            }
            let note = &score.notes[idx];
            let frames = analyze_frames(&samples[start..end], sample_rate);
            let f0s: Vec<Option<f32>> = frames.iter().map(|f| f.f0).collect();

            if let Some((centering, jitter)) = pitch_accuracy_metrics(&f0s, note.frequency) {
                centering_cents.push(centering);
                jitter_cents.push(jitter);
            }
            if let Some(consistency) = vibrato_consistency_of(&f0s, frame_rate) {
                vibrato_scores.push(consistency);
            }
            all_frames.extend(frames);
        }

        // Pitch accuracy (audio-derived): mean centering error + mean jitter,
        // mapped through a cents tolerance into [0, 1].
        let pitch_accuracy = if centering_cents.is_empty() {
            0.0
        } else {
            let combined = mean_f32(&centering_cents) + mean_f32(&jitter_cents);
            (1.0 - combined / PITCH_TOLERANCE_CENTS).clamp(0.0, 1.0)
        };

        // Vibrato consistency: audio-derived when possible, score-derived fallback.
        let vibrato_consistency = if vibrato_scores.is_empty() {
            score_derived_vibrato_consistency(&score.notes)
        } else {
            mean_f32(&vibrato_scores).clamp(0.0, 1.0)
        };

        // Breath quality (audio-derived) from low-energy frame periodicity.
        let breath_quality = breath_quality_of(&all_frames);

        SingingStats {
            total_notes,
            pitch_accuracy,
            vibrato_consistency,
            breath_quality,
        }
    }
}

/// Short-time analysis frame size (samples) for F0 / energy tracking.
const ANALYSIS_FRAME_SIZE: usize = 2048;
/// Hop between successive analysis frames (samples).
const ANALYSIS_HOP_SIZE: usize = 1024;
/// Lowest fundamental frequency considered when estimating F0 (Hz).
const ANALYSIS_MIN_F0: f32 = 55.0;
/// Highest fundamental frequency considered when estimating F0 (Hz).
const ANALYSIS_MAX_F0: f32 = 1100.0;
/// Minimum normalised autocorrelation peak for a frame to be deemed voiced.
const VOICING_THRESHOLD: f32 = 0.25;
/// RMS below which a frame is treated as silence (no analysis performed).
const SILENCE_RMS_EPS: f32 = 1e-4;
/// Combined cents error (centering + jitter) that maps to zero pitch accuracy.
const PITCH_TOLERANCE_CENTS: f32 = 100.0;
/// Lower bound of the vibrato modulation band (Hz).
const VIBRATO_MIN_HZ: f32 = 4.0;
/// Upper bound of the vibrato modulation band (Hz).
const VIBRATO_MAX_HZ: f32 = 8.0;
/// Minimum voiced frames in a note before audio-derived vibrato analysis runs.
const MIN_VIBRATO_FRAMES: usize = 8;
/// Minimum F0 modulation depth (RMS, cents) for a note to count as having
/// vibrato. Below this the F0 is essentially steady (only sub-frame estimation
/// wobble), so there is no vibrato whose regularity could be assessed. This
/// gate is essential because the in-band energy *ratio* is scale-invariant and
/// would otherwise report a spurious value for an unmodulated tone.
const MIN_VIBRATO_DEPTH_CENTS: f32 = 20.0;

/// Per-frame short-time analysis used by the singing quality metrics.
#[derive(Debug, Clone, Copy)]
struct FrameInfo {
    /// RMS energy of the frame.
    rms: f32,
    /// Estimated fundamental frequency in Hz (`None` if unvoiced/silent).
    f0: Option<f32>,
    /// Normalised autocorrelation peak in `[0, 1]` (periodicity / HNR proxy).
    periodicity: f32,
}

/// Root-mean-square energy of a frame.
fn rms_of(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

/// Arithmetic mean of a slice (0.0 for an empty slice).
fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Median of a slice (0.0 for an empty slice).
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

/// Estimate the fundamental frequency of a frame via normalised autocorrelation.
///
/// Returns `(Some(f0_hz), periodicity)` for voiced frames and
/// `(None, periodicity)` otherwise. Uses parabolic interpolation around the
/// autocorrelation peak for sub-sample lag precision.
fn estimate_f0(frame: &[f32], sample_rate: u32) -> (Option<f32>, f32) {
    let n = frame.len();
    if n < 2 {
        return (None, 0.0);
    }
    // Remove DC offset.
    let mean = frame.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = frame.iter().map(|&x| x - mean).collect();
    let energy: f32 = centered.iter().map(|&x| x * x).sum();
    if energy <= 1e-12 {
        return (None, 0.0);
    }

    let min_lag = ((sample_rate as f32 / ANALYSIS_MAX_F0).floor() as usize).max(1);
    let max_lag = ((sample_rate as f32 / ANALYSIS_MIN_F0).ceil() as usize).min(n - 1);
    if max_lag <= min_lag {
        return (None, 0.0);
    }

    let mut corrs = vec![0.0f32; max_lag + 1];
    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut acc = 0.0f32;
        for i in 0..(n - lag) {
            acc += centered[i] * centered[i + lag];
        }
        let norm = acc / energy;
        corrs[lag] = norm;
        if norm > best_corr {
            best_corr = norm;
            best_lag = lag;
        }
    }

    let periodicity = best_corr.clamp(0.0, 1.0);
    if best_lag == 0 || best_corr < VOICING_THRESHOLD {
        return (None, periodicity);
    }

    // Parabolic interpolation around the peak for sub-sample lag precision.
    let refined_lag = if best_lag > min_lag && best_lag < max_lag {
        let a = corrs[best_lag - 1];
        let b = corrs[best_lag];
        let c = corrs[best_lag + 1];
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-9 {
            best_lag as f32 + 0.5 * (a - c) / denom
        } else {
            best_lag as f32
        }
    } else {
        best_lag as f32
    };

    let f0 = sample_rate as f32 / refined_lag;
    if f0.is_finite() && (ANALYSIS_MIN_F0..=ANALYSIS_MAX_F0).contains(&f0) {
        (Some(f0), periodicity)
    } else {
        (None, periodicity)
    }
}

/// Split a contiguous signal into overlapping analysis frames and analyze each.
fn analyze_frames(samples: &[f32], sample_rate: u32) -> Vec<FrameInfo> {
    let mut frames = Vec::new();
    if samples.is_empty() {
        return frames;
    }
    // Signals shorter than a full frame are analyzed as a single short frame.
    if samples.len() < ANALYSIS_FRAME_SIZE {
        let rms = rms_of(samples);
        let (f0, periodicity) = if rms > SILENCE_RMS_EPS {
            estimate_f0(samples, sample_rate)
        } else {
            (None, 0.0)
        };
        frames.push(FrameInfo {
            rms,
            f0,
            periodicity,
        });
        return frames;
    }
    let mut start = 0;
    while start + ANALYSIS_FRAME_SIZE <= samples.len() {
        let frame = &samples[start..start + ANALYSIS_FRAME_SIZE];
        let rms = rms_of(frame);
        let (f0, periodicity) = if rms > SILENCE_RMS_EPS {
            estimate_f0(frame, sample_rate)
        } else {
            (None, 0.0)
        };
        frames.push(FrameInfo {
            rms,
            f0,
            periodicity,
        });
        start += ANALYSIS_HOP_SIZE;
    }
    frames
}

/// Reconstruct the per-note voiced sample ranges produced by `synthesize_notes`.
///
/// Mirrors the exact sample layout of the synthesizer (including the silent
/// inter-note pauses inserted when `legato < 1.0`) so that each notated note can
/// be matched to its realized audio region.
fn note_segments(notes: &[MusicalNote], legato: f32, sample_rate: u32) -> Vec<(usize, usize)> {
    let mut segments = Vec::with_capacity(notes.len());
    let mut cursor = 0usize;
    for note in notes {
        let samples_per_note = (note.duration * sample_rate as f32) as usize;
        let start = cursor;
        let end = start + samples_per_note;
        segments.push((start, end));
        cursor = end;
        if legato < 1.0 {
            let pause = ((1.0 - legato) * sample_rate as f32 * 0.05) as usize;
            cursor += pause;
        }
    }
    segments
}

/// Compute the centering and jitter cents errors for one note's F0 track.
///
/// * centering — `|1200·log2(median_f0 / target)|`, how far the note's central
///   pitch sits from the notated pitch.
/// * jitter — mean local second-difference of the F0 track (in cents), which
///   removes smooth (locally linear) vibrato and isolates fast jitter.
///
/// Returns `None` when there are no voiced frames.
fn pitch_accuracy_metrics(f0s: &[Option<f32>], target_hz: f32) -> Option<(f32, f32)> {
    let voiced: Vec<f32> = f0s.iter().filter_map(|&x| x).collect();
    if voiced.is_empty() || target_hz <= 0.0 {
        return None;
    }

    let median = median_f32(&voiced);
    let centering_cents = (1200.0 * (median / target_hz).log2()).abs();

    let mut jitter_sum = 0.0f32;
    let mut jitter_count = 0usize;
    for i in 1..f0s.len().saturating_sub(1) {
        if let (Some(prev), Some(curr), Some(next)) = (f0s[i - 1], f0s[i], f0s[i + 1]) {
            let expected = 0.5 * (prev + next);
            if expected > 0.0 && curr > 0.0 {
                jitter_sum += (1200.0 * (curr / expected).log2()).abs();
                jitter_count += 1;
            }
        }
    }
    let jitter_cents = if jitter_count > 0 {
        jitter_sum / jitter_count as f32
    } else {
        0.0
    };

    Some((centering_cents, jitter_cents))
}

/// Audio-derived vibrato consistency for one note's F0 track.
///
/// The voiced-F0 contour is detrended, Hann-windowed and transformed with
/// `scirs2_fft::rfft`; the fraction of spectral energy inside the 4–8 Hz vibrato
/// band is returned as the consistency in `[0, 1]`. A clean, regular vibrato
/// concentrates its energy in that band (→ near 1.0), whereas an irregular
/// modulation smears energy across the spectrum (→ lower). Returns `None` when
/// the note has too few voiced frames for a reliable spectrum.
fn vibrato_consistency_of(f0s: &[Option<f32>], frame_rate: f32) -> Option<f32> {
    // Build a continuous contour, holding the last voiced value across gaps.
    let mut contour: Vec<f32> = Vec::with_capacity(f0s.len());
    let mut last: Option<f32> = None;
    let mut voiced_count = 0usize;
    for &f in f0s {
        match f {
            Some(v) => {
                last = Some(v);
                voiced_count += 1;
                contour.push(v);
            }
            None => {
                if let Some(v) = last {
                    contour.push(v);
                }
            }
        }
    }
    if voiced_count < MIN_VIBRATO_FRAMES || contour.len() < MIN_VIBRATO_FRAMES {
        return None;
    }

    let m = contour.len();
    let mean = mean_f32(&contour);
    if mean <= 0.0 {
        return Some(0.0);
    }

    // Express the F0 contour as a deviation in cents from its own mean. Cents are
    // pitch-perceptual and let us apply an absolute modulation-depth gate.
    let cents: Vec<f32> = contour
        .iter()
        .map(|&v| 1200.0 * (v / mean).log2())
        .collect();
    let depth_rms = (cents.iter().map(|&c| c * c).sum::<f32>() / m as f32).sqrt();
    if depth_rms < MIN_VIBRATO_DEPTH_CENTS {
        // No musically meaningful modulation: treat as "no vibrato" (score 0).
        return Some(0.0);
    }

    // Hann-window the cents deviation to reduce spectral leakage.
    let mut windowed: Vec<f64> = cents
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (m as f32 - 1.0)).cos();
            (c * w) as f64
        })
        .collect();

    // Zero-pad for finer bin spacing, then real FFT.
    let fft_len = m.next_power_of_two().max(64);
    windowed.resize(fft_len, 0.0);
    let spectrum = scirs2_fft::rfft(&windowed, None).ok()?;

    let bin_hz = frame_rate / fft_len as f32;
    let mut band_energy = 0.0f64;
    let mut total_energy = 0.0f64;
    // Skip the DC bin (k = 0).
    for (k, c) in spectrum.iter().enumerate().skip(1) {
        let power = c.norm() * c.norm();
        total_energy += power;
        let freq = k as f32 * bin_hz;
        if (VIBRATO_MIN_HZ..=VIBRATO_MAX_HZ).contains(&freq) {
            band_energy += power;
        }
    }
    if total_energy <= 0.0 {
        return Some(0.0);
    }
    Some(((band_energy / total_energy) as f32).clamp(0.0, 1.0))
}

/// Score-derived vibrato consistency fallback.
///
/// Used only when no note has enough voiced audio frames for the spectral
/// (audio-derived) analysis. Estimates regularity from the *notated* vibrato
/// intensities: uniform vibrato markings across notes → high consistency
/// (`1 − coefficient_of_variation`).
fn score_derived_vibrato_consistency(notes: &[MusicalNote]) -> f32 {
    let intensities: Vec<f32> = notes
        .iter()
        .map(|n| n.vibrato)
        .filter(|&v| v > 0.0)
        .collect();
    if intensities.len() < 2 {
        return 0.0;
    }
    let mean = mean_f32(&intensities);
    if mean <= 0.0 {
        return 0.0;
    }
    let var = intensities
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / intensities.len() as f32;
    let coefficient_of_variation = var.sqrt() / mean;
    (1.0 - coefficient_of_variation).clamp(0.0, 1.0)
}

/// Audio-derived breath quality from the periodicity of low-energy frames.
///
/// Pure-silence frames (the synthesizer fills inter-note pauses with zeros) are
/// excluded; among the remaining frames the lower-energy tier (note onsets,
/// offsets and transitions, where breath noise concentrates) is selected and its
/// mean periodicity returned. A clean, well-supported voice stays periodic even
/// when quiet (→ high quality); breathiness injects aperiodic noise (→ lower).
fn breath_quality_of(frames: &[FrameInfo]) -> f32 {
    let mut voiced: Vec<&FrameInfo> = frames.iter().filter(|f| f.rms > SILENCE_RMS_EPS).collect();
    if voiced.is_empty() {
        return 0.0;
    }
    voiced.sort_by(|a, b| {
        a.rms
            .partial_cmp(&b.rms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let tier = ((voiced.len() as f32 * 0.4).ceil() as usize).clamp(1, voiced.len());
    let mean_periodicity = voiced[..tier].iter().map(|f| f.periodicity).sum::<f32>() / tier as f32;
    mean_periodicity.clamp(0.0, 1.0)
}

/// Builder for singing controller configuration
#[derive(Debug, Clone)]
pub struct SingingControllerBuilder {
    config: SingingConfig,
}

impl SingingControllerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SingingConfig::default(),
        }
    }

    /// Enable or disable singing synthesis
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set voice type
    pub fn voice_type(mut self, voice_type: VoiceType) -> Self {
        self.config.voice_type = voice_type;
        self
    }

    /// Set singing technique
    pub fn technique(mut self, technique: SingingTechnique) -> Self {
        self.config.technique = technique;
        self
    }

    /// Enable auto pitch detection
    pub fn auto_pitch_detection(mut self, enabled: bool) -> Self {
        self.config.auto_pitch_detection = enabled;
        self
    }

    /// Enable score caching
    pub fn cache_scores(mut self, enabled: bool) -> Self {
        self.config.cache_scores = enabled;
        self
    }

    /// Build the singing controller
    pub async fn build(self) -> Result<SingingController> {
        let controller = SingingController::with_config(self.config).await?;
        Ok(controller)
    }
}

impl Default for SingingControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SingingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            voice_type: VoiceType::Alto,
            technique: SingingTechnique::default(),
            auto_pitch_detection: false,
            cache_scores: true,
        }
    }
}

impl Default for SingingTechnique {
    fn default() -> Self {
        Self {
            breath_control: 0.8,
            vocal_fry: 0.2,
            head_voice_ratio: 0.5,
            vibrato_speed: 5.0,
            vibrato_depth: 0.6,
            pitch_bend: 0.4,
            legato: 0.7,
        }
    }
}

impl MusicalNote {
    /// Create a new musical note
    pub fn new(note: String, octave: u8, duration: f32, velocity: f32) -> Self {
        let frequency = Self::calculate_frequency(&note, octave);
        Self {
            note,
            octave,
            frequency,
            duration,
            velocity,
            vibrato: 0.5,
        }
    }

    /// Calculate frequency from note name and octave
    fn calculate_frequency(note: &str, octave: u8) -> f32 {
        let base_frequencies = HashMap::from([
            ("C", 261.63),
            ("D", 293.66),
            ("E", 329.63),
            ("F", 349.23),
            ("G", 392.00),
            ("A", 440.00),
            ("B", 493.88),
        ]);

        let base_freq = base_frequencies.get(note).copied().unwrap_or(440.0);
        base_freq * 2.0_f32.powi(octave as i32 - 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_singing_controller_creation() {
        let controller = SingingController::new().await.unwrap();
        assert!(controller.is_enabled().await);
    }

    #[tokio::test]
    async fn test_singing_technique_setting() {
        let controller = SingingController::new().await.unwrap();
        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.1,
            head_voice_ratio: 0.8,
            vibrato_speed: 6.0,
            vibrato_depth: 0.7,
            pitch_bend: 0.3,
            legato: 0.9,
        };

        controller.set_technique(technique.clone()).await.unwrap();
        let config = controller.get_config().await;
        assert_eq!(config.technique.breath_control, 0.9);
    }

    #[tokio::test]
    async fn test_preset_application() {
        let controller = SingingController::new().await.unwrap();
        controller.apply_preset("classical").await.unwrap();

        let config = controller.get_config().await;
        assert_eq!(config.technique.breath_control, 0.9);
    }

    #[tokio::test]
    async fn test_singing_builder() {
        let controller = SingingControllerBuilder::new()
            .enabled(true)
            .voice_type(VoiceType::Soprano)
            .auto_pitch_detection(true)
            .build()
            .await
            .unwrap();

        assert!(controller.is_enabled().await);
        let config = controller.get_config().await;
        assert_eq!(config.voice_type, VoiceType::Soprano);
    }

    #[tokio::test]
    async fn test_musical_note_creation() {
        let note = MusicalNote::new("A".to_string(), 4, 0.5, 0.8);
        assert_eq!(note.note, "A");
        assert_eq!(note.octave, 4);
        assert!((note.frequency - 440.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_score_parsing() {
        let controller = SingingController::new().await.unwrap();
        let score_text = "TEMPO: 120\nKEY: C\nNOTE: C4 0.5 0.8 0.3\nNOTE: D4 0.5 0.8 0.3";

        let score = controller.parse_score(score_text).await.unwrap();
        assert_eq!(score.tempo, 120.0);
        assert_eq!(score.key_signature, "C");
        assert_eq!(score.notes.len(), 2);
    }

    #[tokio::test]
    async fn test_text_to_melody_generation() {
        let controller = SingingController::new().await.unwrap();
        let result = controller
            .synthesize_from_text("Hello world", "C", 120.0)
            .await
            .unwrap();

        assert_eq!(result.score.notes.len(), 2); // Two words
        assert!(result.audio.duration() > 0.0);
    }

    #[tokio::test]
    async fn test_preset_listing() {
        let controller = SingingController::new().await.unwrap();
        let presets = controller.list_presets();
        assert!(presets.contains(&"classical".to_string()));
        assert!(presets.contains(&"pop".to_string()));
        assert!(presets.contains(&"jazz".to_string()));
        assert!(presets.contains(&"opera".to_string()));
    }

    #[tokio::test]
    async fn test_harmonic_synthesis() {
        // Test that synthesis produces harmonics-rich audio
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.1,
            head_voice_ratio: 0.5,
            vibrato_speed: 5.0,
            vibrato_depth: 0.5,
            pitch_bend: 0.3,
            legato: 0.8,
        };

        controller.set_technique(technique).await.unwrap();

        let result = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // Audio should have content
        assert!(result.audio.samples().len() > 0);

        // Check that audio has reasonable amplitude (not all zeros)
        let max_amplitude = result
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(max_amplitude > 0.01, "Audio should have audible amplitude");
        assert!(max_amplitude <= 1.0, "Audio should be within bounds");
    }

    #[tokio::test]
    async fn test_adsr_envelope() {
        // Test that ADSR envelope is applied correctly
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 1.0, // Perfect breath control
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0, // No vibrato for cleaner test
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0, // Full legato
        };

        controller.set_technique(technique).await.unwrap();

        let result = controller
            .synthesize_from_text("A", "C", 60.0)
            .await
            .unwrap();

        let samples = result.audio.samples();

        // Attack phase: amplitude should increase at the start
        let attack_samples = &samples[0..100.min(samples.len())];
        if attack_samples.len() > 10 {
            let start_avg = attack_samples[0..5].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            let mid_avg = attack_samples[50..55].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            assert!(
                mid_avg >= start_avg * 0.8,
                "Envelope should have attack phase"
            );
        }

        // Release phase: amplitude should decrease at the end
        if samples.len() > 100 {
            let end_samples = &samples[samples.len() - 100..];
            let mid_end_avg = end_samples[0..5].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            let final_avg = end_samples[95..100].iter().map(|s| s.abs()).sum::<f32>() / 5.0;
            assert!(
                mid_end_avg >= final_avg,
                "Envelope should have release phase"
            );
        }
    }

    #[tokio::test]
    async fn test_breath_noise_modeling() {
        // Test with different breath control levels
        let controller = SingingController::new().await.unwrap();

        // Low breath control = more breath noise
        let technique_breathy = SingingTechnique {
            breath_control: 0.3,
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller
            .set_technique(technique_breathy.clone())
            .await
            .unwrap();
        let result_breathy = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // High breath control = less breath noise, cleaner sound
        let technique_clean = SingingTechnique {
            breath_control: 1.0,
            ..technique_breathy
        };

        controller.set_technique(technique_clean).await.unwrap();
        let result_clean = controller
            .synthesize_from_text("Test", "C", 120.0)
            .await
            .unwrap();

        // Both should produce valid audio
        assert!(result_breathy.audio.samples().len() > 0);
        assert!(result_clean.audio.samples().len() > 0);

        // Clean version should generally have higher average amplitude
        // (since breath noise is added, not replacing signal)
        let breathy_max = result_breathy
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let clean_max = result_clean
            .audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(clean_max > 0.01);
        assert!(breathy_max > 0.01);
    }

    #[tokio::test]
    async fn test_vocal_fry_effect() {
        // Test vocal fry in low register
        let controller = SingingController::new().await.unwrap();

        let technique = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.5,        // Moderate vocal fry
            head_voice_ratio: 0.3, // Chest voice dominant
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller.set_technique(technique).await.unwrap();

        // Low note should trigger vocal fry effect
        let score = MusicalScore {
            notes: vec![MusicalNote {
                note: "C".to_string(),
                octave: 2,       // Very low note
                frequency: 65.4, // C2
                duration: 0.5,
                velocity: 0.8,
                vibrato: 0.0,
            }],
            tempo: 120.0,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature: "C".to_string(),
        };

        let result = controller
            .synthesize_score(score, "Low note test")
            .await
            .unwrap();

        assert!(result.audio.samples().len() > 0);
        assert!(result.stats.total_notes == 1);
    }

    #[tokio::test]
    async fn test_legato_vs_staccato() {
        let controller = SingingController::new().await.unwrap();

        // Full legato - no pauses between notes
        let technique_legato = SingingTechnique {
            breath_control: 0.9,
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 0.0,
            vibrato_depth: 0.0,
            pitch_bend: 0.0,
            legato: 1.0,
        };

        controller
            .set_technique(technique_legato.clone())
            .await
            .unwrap();
        let result_legato = controller
            .synthesize_from_text("Hello World", "C", 120.0)
            .await
            .unwrap();

        // Staccato - pauses between notes
        let technique_staccato = SingingTechnique {
            legato: 0.3,
            ..technique_legato
        };

        controller.set_technique(technique_staccato).await.unwrap();
        let result_staccato = controller
            .synthesize_from_text("Hello World", "C", 120.0)
            .await
            .unwrap();

        // Staccato version should be longer due to pauses
        assert!(result_staccato.audio.duration() > result_legato.audio.duration() * 0.95);
    }

    // --- Real singing-statistics analysis tests ---------------------------------

    /// Deterministic LCG (Numerical Recipes constants) for reproducible test
    /// signals. The project policy forbids statistical RNGs in tests.
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use the high 24 bits as a uniform value in [0, 1).
        ((*state >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Deterministic uniform value in [-1, 1).
    fn lcg_signed(state: &mut u64) -> f32 {
        lcg_next(state) * 2.0 - 1.0
    }

    #[test]
    fn test_estimate_f0_detects_known_pitch() {
        let sample_rate = 44100u32;
        let freq = 220.0f32;
        let frame: Vec<f32> = (0..ANALYSIS_FRAME_SIZE)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let (f0, periodicity) = estimate_f0(&frame, sample_rate);
        let f0 = f0.expect("a steady sine must be detected as voiced");
        let cents = (1200.0 * (f0 / freq).log2()).abs();
        assert!(cents < 20.0, "f0 = {f0} Hz is {cents} cents off 220 Hz");
        assert!(periodicity > 0.8, "periodicity = {periodicity}");
    }

    #[test]
    fn test_high_jitter_lowers_pitch_accuracy() {
        let target = 220.0f32;
        let steady: Vec<Option<f32>> = (0..24).map(|_| Some(target)).collect();

        let mut state = 0x1234_5678_9abc_def0u64;
        let jittery: Vec<Option<f32>> = (0..24)
            .map(|_| Some(target * (1.0 + 0.05 * lcg_signed(&mut state))))
            .collect();

        let accuracy = |f0s: &[Option<f32>]| {
            let (centering, jitter) = pitch_accuracy_metrics(f0s, target).unwrap();
            (1.0 - (centering + jitter) / PITCH_TOLERANCE_CENTS).clamp(0.0, 1.0)
        };

        let steady_acc = accuracy(&steady);
        let jittery_acc = accuracy(&jittery);

        assert!(steady_acc > 0.95, "steady accuracy = {steady_acc}");
        assert!(
            steady_acc > jittery_acc,
            "steady {steady_acc} should exceed jittery {jittery_acc}"
        );
        assert!((0.0..=1.0).contains(&jittery_acc));
    }

    #[test]
    fn test_smooth_vibrato_not_counted_as_jitter() {
        // At equal modulation amplitude, smooth vibrato must yield far less
        // jitter than random perturbation (the second-difference removes the
        // locally-linear vibrato trend).
        let target = 220.0f32;
        let frame_rate = 44100.0 / ANALYSIS_HOP_SIZE as f32;
        let amp = 0.04f32;

        let smooth: Vec<Option<f32>> = (0..40)
            .map(|k| {
                let t = k as f32 / frame_rate;
                Some(target * (1.0 + amp * (2.0 * std::f32::consts::PI * 5.0 * t).sin()))
            })
            .collect();

        let mut state = 0x0f0f_0f0f_0f0f_0f0fu64;
        let random: Vec<Option<f32>> = (0..40)
            .map(|_| Some(target * (1.0 + amp * lcg_signed(&mut state))))
            .collect();

        let (_, jitter_smooth) = pitch_accuracy_metrics(&smooth, target).unwrap();
        let (_, jitter_random) = pitch_accuracy_metrics(&random, target).unwrap();
        assert!(
            jitter_smooth < jitter_random,
            "smooth jitter {jitter_smooth} should be < random jitter {jitter_random}"
        );
    }

    #[test]
    fn test_vibrato_consistency_band_concentration() {
        let frame_rate = 44100.0 / ANALYSIS_HOP_SIZE as f32; // ~43 Hz
        let clean: Vec<Option<f32>> = (0..48)
            .map(|k| {
                let t = k as f32 / frame_rate;
                Some(220.0 * (1.0 + 0.03 * (2.0 * std::f32::consts::PI * 5.0 * t).sin()))
            })
            .collect();

        let mut state = 0xdead_beef_cafe_babeu64;
        let noisy: Vec<Option<f32>> = (0..48)
            .map(|_| Some(220.0 * (1.0 + 0.03 * lcg_signed(&mut state))))
            .collect();

        let clean_vc = vibrato_consistency_of(&clean, frame_rate).unwrap();
        let noisy_vc = vibrato_consistency_of(&noisy, frame_rate).unwrap();

        assert!((0.0..=1.0).contains(&clean_vc), "clean = {clean_vc}");
        assert!((0.0..=1.0).contains(&noisy_vc), "noisy = {noisy_vc}");
        assert!(
            clean_vc > noisy_vc,
            "clean vibrato {clean_vc} should exceed irregular {noisy_vc}"
        );
        assert!(
            clean_vc > 0.4,
            "clean 5 Hz vibrato concentration = {clean_vc}"
        );
    }

    #[test]
    fn test_note_segments_match_synthesis_layout() {
        let notes = vec![
            MusicalNote::new("A".to_string(), 4, 0.5, 0.8),
            MusicalNote::new("C".to_string(), 4, 0.25, 0.8),
        ];
        // Full legato => no inter-note pause, segments are contiguous.
        let segs = note_segments(&notes, 1.0, 44100);
        assert_eq!(segs[0], (0, 22050));
        assert_eq!(segs[1], (22050, 22050 + 11025));

        // With legato < 1.0 a silent pause is inserted between notes.
        let segs = note_segments(&notes, 0.5, 44100);
        let pause = ((1.0 - 0.5) * 44100.0 * 0.05) as usize;
        assert_eq!(segs[0], (0, 22050));
        assert_eq!(segs[1].0, 22050 + pause);
    }

    fn deterministic_technique(vibrato_depth: f32) -> SingingTechnique {
        SingingTechnique {
            breath_control: 1.0, // disables stochastic breath noise -> deterministic audio
            vocal_fry: 0.0,
            head_voice_ratio: 0.5,
            vibrato_speed: 5.0,
            vibrato_depth,
            pitch_bend: 0.0,
            legato: 1.0,
        }
    }

    fn two_note_score(vibrato: f32) -> MusicalScore {
        MusicalScore {
            notes: vec![
                MusicalNote {
                    note: "A".to_string(),
                    octave: 3,
                    frequency: 220.0,
                    duration: 0.6,
                    velocity: 0.8,
                    vibrato,
                },
                MusicalNote {
                    note: "C".to_string(),
                    octave: 4,
                    frequency: 261.63,
                    duration: 0.6,
                    velocity: 0.8,
                    vibrato,
                },
            ],
            tempo: 120.0,
            time_signature_num: 4,
            time_signature_den: 4,
            key_signature: "C".to_string(),
        }
    }

    #[tokio::test]
    async fn test_singing_stats_deterministic_and_bounded() {
        let controller = SingingControllerBuilder::new()
            .enabled(true)
            .technique(deterministic_technique(0.0)) // no vibrato -> stable pitch
            .build()
            .await
            .unwrap();

        let score = two_note_score(0.0);
        let r1 = controller
            .synthesize_score(score.clone(), "la la")
            .await
            .unwrap();
        let r2 = controller
            .synthesize_score(score.clone(), "la la")
            .await
            .unwrap();

        // Deterministic audio (breath_control = 1.0) => identical stats.
        assert_eq!(r1.stats.pitch_accuracy, r2.stats.pitch_accuracy);
        assert_eq!(r1.stats.vibrato_consistency, r2.stats.vibrato_consistency);
        assert_eq!(r1.stats.breath_quality, r2.stats.breath_quality);

        // Every metric must lie within [0, 1].
        assert!(
            (0.0..=1.0).contains(&r1.stats.pitch_accuracy),
            "pitch_accuracy = {}",
            r1.stats.pitch_accuracy
        );
        assert!(
            (0.0..=1.0).contains(&r1.stats.vibrato_consistency),
            "vibrato_consistency = {}",
            r1.stats.vibrato_consistency
        );
        assert!(
            (0.0..=1.0).contains(&r1.stats.breath_quality),
            "breath_quality = {}",
            r1.stats.breath_quality
        );

        // Clean additive tones at the notated pitches should score well, and the
        // clean low-energy regions should look well-supported (high breath quality).
        assert!(
            r1.stats.pitch_accuracy > 0.7,
            "pitch_accuracy = {}",
            r1.stats.pitch_accuracy
        );
        assert!(
            r1.stats.breath_quality > 0.5,
            "breath_quality = {}",
            r1.stats.breath_quality
        );
        assert_eq!(r1.stats.total_notes, 2);
    }

    #[tokio::test]
    async fn test_audio_derived_vibrato_consistency_responds_to_vibrato() {
        // A score WITH vibrato should yield higher audio-derived vibrato
        // consistency than one without (which has no modulation to be consistent).
        let with_vibrato = SingingControllerBuilder::new()
            .enabled(true)
            .technique(deterministic_technique(0.3))
            .build()
            .await
            .unwrap();
        let without_vibrato = SingingControllerBuilder::new()
            .enabled(true)
            .technique(deterministic_technique(0.0))
            .build()
            .await
            .unwrap();

        // note.vibrato 0.2 x technique depth 0.3 => ~6% F0 modulation (well above
        // the depth gate, clean enough for a sharp 5 Hz band peak).
        let vib = with_vibrato
            .synthesize_score(two_note_score(0.2), "ah")
            .await
            .unwrap();
        let flat = without_vibrato
            .synthesize_score(two_note_score(0.0), "ah")
            .await
            .unwrap();

        assert!((0.0..=1.0).contains(&vib.stats.vibrato_consistency));
        assert!((0.0..=1.0).contains(&flat.stats.vibrato_consistency));
        // The unmodulated take has no vibrato, so the depth gate forces it to 0,
        // while the modulated take detects a regular 5 Hz vibrato.
        assert!(
            vib.stats.vibrato_consistency > flat.stats.vibrato_consistency,
            "with-vibrato {} should exceed flat {}",
            vib.stats.vibrato_consistency,
            flat.stats.vibrato_consistency
        );
        assert!(
            vib.stats.vibrato_consistency > 0.1,
            "audio-derived vibrato consistency too low: {}",
            vib.stats.vibrato_consistency
        );
    }
}
