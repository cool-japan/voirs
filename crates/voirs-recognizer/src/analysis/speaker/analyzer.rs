use crate::traits::{
    AccentInfo, AgeRange, Emotion, EmotionalAnalysis, Gender, SpeakerCharacteristics,
    VoiceCharacteristics, VoiceQuality,
};
use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Speaker characteristics analyzer
pub struct SpeakerAnalyzer {
    /// Sample rate for analysis
    pub(super) sample_rate: f32,
    /// Frame size for spectral analysis
    pub(super) frame_size: usize,
    /// Hop size for overlapping frames
    pub(super) hop_size: usize,
    /// Gender classification thresholds
    pub(super) gender_thresholds: GenderThresholds,
    /// Age classification ranges
    pub(super) age_ranges: AgeClassificationRanges,
}

/// Gender classification thresholds
#[derive(Debug, Clone)]
pub(super) struct GenderThresholds {
    /// F0 threshold between male and female (Hz)
    pub(super) f0_threshold: f32,
    /// Formant-based threshold
    pub(super) formant_threshold: f32,
}

/// Age classification ranges
#[derive(Debug, Clone)]
pub(super) struct AgeClassificationRanges {
    /// Child F0 range
    pub(super) child: (f32, f32),
    /// Teen F0 range
    pub(super) teen: (f32, f32),
    /// Adult F0 range
    pub(super) adult: (f32, f32),
    /// Senior F0 range
    pub(super) senior: (f32, f32),
}

#[derive(Debug, Clone)]
pub(super) struct F0Characteristics {
    pub(super) mean_f0: f32,
    pub(super) f0_range: (f32, f32),
    pub(super) f0_variation: f32,
    #[allow(dead_code)]
    pub(super) voiced_ratio: f32,
}

#[derive(Debug, Clone)]
pub(super) struct EnergyFeatures {
    pub(super) mean_energy: f32,
    pub(super) energy_variation: f32,
    #[allow(dead_code)]
    pub(super) energy_dynamics: f32,
}

#[derive(Debug, Clone)]
pub(super) struct SpectralFeatures {
    pub(super) centroid: f32,
    #[allow(dead_code)]
    pub(super) spread: f32,
    #[allow(dead_code)]
    pub(super) flux: f32,
}

impl SpeakerAnalyzer {
    /// Create a new speaker analyzer
    ///
    /// # Errors
    ///
    /// This function currently always succeeds, but returns a Result for
    /// consistency with the async interface and future extensibility.
    pub async fn new() -> Result<Self, RecognitionError> {
        let gender_thresholds = GenderThresholds {
            f0_threshold: 165.0,       // Typical boundary between male and female F0
            formant_threshold: 2800.0, // F1+F2+F3 average
        };

        let age_ranges = AgeClassificationRanges {
            child: (250.0, 450.0),
            teen: (200.0, 350.0),
            adult: (80.0, 300.0),
            senior: (85.0, 250.0),
        };

        Ok(Self {
            sample_rate: 16000.0,
            frame_size: 1024,
            hop_size: 512,
            gender_thresholds,
            age_ranges,
        })
    }

    /// Analyze speaker characteristics
    ///
    /// # Errors
    ///
    /// Returns an error if audio analysis fails, such as when the audio buffer
    /// is too short or contains invalid data.
    pub async fn analyze_speaker(
        &self,
        audio: &AudioBuffer,
    ) -> Result<SpeakerCharacteristics, RecognitionError> {
        // Extract fundamental frequency
        let f0_analysis = self.extract_f0_characteristics(audio).await?;

        // Extract formant frequencies
        let formants = self.extract_formants(audio).await?;

        // Analyze voice quality
        let voice_quality = self.analyze_voice_quality(audio).await?;

        // Classify gender
        let gender = self.classify_gender(&f0_analysis, &formants);

        // Estimate age
        let age_range = self.estimate_age(&f0_analysis, &voice_quality);

        // Detect accent (simplified)
        let accent = self.detect_accent(audio, &formants).await?;

        let voice_characteristics = VoiceCharacteristics {
            f0_range: f0_analysis.f0_range,
            formants,
            voice_quality,
        };

        Ok(SpeakerCharacteristics {
            gender,
            age_range,
            voice_characteristics,
            accent,
        })
    }

    /// Analyze emotional content
    /// Analyze emotional characteristics from audio
    ///
    /// # Errors
    ///
    /// Returns an error if emotional analysis fails, such as when the audio buffer
    /// is too short or spectral analysis cannot be performed.
    pub async fn analyze_emotion(
        &self,
        audio: &AudioBuffer,
    ) -> Result<EmotionalAnalysis, RecognitionError> {
        // Extract prosodic features for emotion
        let f0_features = self.extract_f0_characteristics(audio).await?;
        let energy_features = self.extract_energy_features(audio).await?;
        let spectral_features = self.extract_spectral_features(audio).await?;

        // Classify primary emotion
        let primary_emotion =
            Self::classify_primary_emotion(&f0_features, &energy_features, &spectral_features);

        // Calculate emotion scores for all emotions
        let emotion_scores =
            Self::calculate_emotion_scores(&f0_features, &energy_features, &spectral_features);

        // Calculate dimensional emotion values
        let (valence, arousal) = Self::calculate_emotion_dimensions(&emotion_scores);

        // Calculate intensity
        let intensity = Self::calculate_emotional_intensity(&f0_features, &energy_features);

        Ok(EmotionalAnalysis {
            primary_emotion,
            emotion_scores,
            intensity,
            valence,
            arousal,
        })
    }

    /// Extract F0 characteristics
    pub(super) async fn extract_f0_characteristics(
        &self,
        audio: &AudioBuffer,
    ) -> Result<F0Characteristics, RecognitionError> {
        let samples = audio.samples();

        // Extract pitch using autocorrelation (simplified)
        let pitch_contour = self.extract_pitch_simple(samples).await?;

        // Filter voiced segments
        let voiced_pitches: Vec<f32> = pitch_contour
            .iter()
            .filter(|&&p| p > 50.0 && p < 800.0)
            .copied()
            .collect();

        if voiced_pitches.is_empty() {
            return Ok(F0Characteristics {
                mean_f0: 0.0,
                f0_range: (0.0, 0.0),
                f0_variation: 0.0,
                voiced_ratio: 0.0,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let mean_f0 = voiced_pitches.iter().sum::<f32>() / voiced_pitches.len() as f32;
        let min_f0 = voiced_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = voiced_pitches
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let f0_variation = {
            #[allow(clippy::cast_precision_loss)]
            let variance = voiced_pitches
                .iter()
                .map(|&p| (p - mean_f0).powi(2))
                .sum::<f32>()
                / voiced_pitches.len() as f32;
            variance.sqrt()
        };

        #[allow(clippy::cast_precision_loss)]
        let voiced_ratio = voiced_pitches.len() as f32 / pitch_contour.len() as f32;

        Ok(F0Characteristics {
            mean_f0,
            f0_range: (min_f0, max_f0),
            f0_variation,
            voiced_ratio,
        })
    }

    /// Extract formant frequencies
    pub(super) async fn extract_formants(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<f32>, RecognitionError> {
        let samples = audio.samples();

        // Simplified formant extraction using spectral peaks
        let spectrum = self.compute_spectrum(samples).await?;

        // Find formant peaks (F1, F2, F3)
        let mut formants = Vec::new();

        // Expected formant ranges
        let formant_ranges = [
            (200.0, 1000.0),  // F1
            (800.0, 2500.0),  // F2
            (2000.0, 4000.0), // F3
        ];

        for &(min_freq, max_freq) in &formant_ranges {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let min_bin =
                (min_freq * spectrum.len() as f32 / (audio.sample_rate() as f32 / 2.0)) as usize;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let max_bin =
                (max_freq * spectrum.len() as f32 / (audio.sample_rate() as f32 / 2.0)) as usize;

            let range = min_bin..max_bin.min(spectrum.len());

            if let Some((peak_bin, _)) = spectrum[range]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                #[allow(clippy::cast_precision_loss)]
                let formant_freq = (min_bin + peak_bin) as f32 * audio.sample_rate() as f32
                    / 2.0
                    / spectrum.len() as f32;
                formants.push(formant_freq);
            }
        }

        // Fill with default values if needed
        while formants.len() < 3 {
            formants.push(0.0);
        }

        Ok(formants)
    }

    /// Analyze voice quality metrics
    pub(super) async fn analyze_voice_quality(
        &self,
        audio: &AudioBuffer,
    ) -> Result<VoiceQuality, RecognitionError> {
        let samples = audio.samples();

        // Calculate jitter (pitch period variation)
        let jitter = self.calculate_jitter(samples).await?;

        // Calculate shimmer (amplitude variation)
        let shimmer = self.calculate_shimmer(samples).await?;

        // Calculate harmonics-to-noise ratio
        let hnr = self.calculate_hnr(samples).await?;

        Ok(VoiceQuality {
            jitter,
            shimmer,
            hnr,
        })
    }

    /// Classify gender based on acoustic features
    pub(super) fn classify_gender(
        &self,
        f0_characteristics: &F0Characteristics,
        formants: &[f32],
    ) -> Option<Gender> {
        if f0_characteristics.mean_f0 == 0.0 {
            return None;
        }

        // Primary classification based on F0
        let f0_gender = if f0_characteristics.mean_f0 > self.gender_thresholds.f0_threshold {
            Gender::Female
        } else {
            Gender::Male
        };

        // Secondary classification based on formants
        let formant_sum = formants.iter().take(3).sum::<f32>();
        let formant_gender = if formant_sum > self.gender_thresholds.formant_threshold {
            Gender::Female
        } else {
            Gender::Male
        };

        // Combine classifications (F0 is more reliable)
        match (f0_gender, formant_gender) {
            (Gender::Male, Gender::Male) => Some(Gender::Male),
            (Gender::Female, Gender::Female) => Some(Gender::Female),
            (f0_class, _) => Some(f0_class), // Trust F0 more
        }
    }

    /// Estimate age range
    pub(super) fn estimate_age(
        &self,
        f0_characteristics: &F0Characteristics,
        voice_quality: &VoiceQuality,
    ) -> Option<AgeRange> {
        if f0_characteristics.mean_f0 == 0.0 {
            return None;
        }

        let f0 = f0_characteristics.mean_f0;

        // Age classification based on F0 and voice quality
        let age_from_f0 = if self.age_ranges.child.0 <= f0 && f0 <= self.age_ranges.child.1 {
            Some(AgeRange::Child)
        } else if self.age_ranges.teen.0 <= f0 && f0 <= self.age_ranges.teen.1 {
            Some(AgeRange::Teen)
        } else if self.age_ranges.adult.0 <= f0 && f0 <= self.age_ranges.adult.1 {
            Some(AgeRange::Adult)
        } else if self.age_ranges.senior.0 <= f0 && f0 <= self.age_ranges.senior.1 {
            Some(AgeRange::Senior)
        } else {
            Some(AgeRange::Adult) // Default
        };

        // Adjust based on voice quality (older speakers often have more jitter/shimmer)
        if voice_quality.jitter > 0.05 || voice_quality.shimmer > 0.1 {
            // Higher likelihood of senior
            match age_from_f0 {
                Some(AgeRange::Adult) => Some(AgeRange::Senior),
                other => other,
            }
        } else {
            age_from_f0
        }
    }

    /// Detect accent (simplified implementation)
    pub(super) async fn detect_accent(
        &self,
        _audio: &AudioBuffer,
        formants: &[f32],
    ) -> Result<Option<AccentInfo>, RecognitionError> {
        // Simplified accent detection based on formant patterns
        if formants.len() < 2 {
            return Ok(None);
        }

        let f1 = formants[0];
        let f2 = formants[1];

        // Very basic accent classification
        let accent_type = if f1 > 500.0 && f2 > 2200.0 {
            "American"
        } else if f1 < 400.0 && f2 < 2000.0 {
            "British"
        } else {
            "General"
        };

        Ok(Some(AccentInfo {
            accent_type: accent_type.to_string(),
            confidence: 0.6, // Low confidence for this simple method
            regional_indicators: vec!["formant-based".to_string()],
        }))
    }

    /// Extract energy features for emotion analysis
    pub(super) async fn extract_energy_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<EnergyFeatures, RecognitionError> {
        let samples = audio.samples();

        // Calculate frame-by-frame energy
        let mut energies = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            #[allow(clippy::cast_precision_loss)]
            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            energies.push(energy);
            pos += self.hop_size;
        }

        if energies.is_empty() {
            return Ok(EnergyFeatures {
                mean_energy: 0.0,
                energy_variation: 0.0,
                energy_dynamics: 0.0,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

        let energy_variation = {
            let variance = energies
                .iter()
                .map(|&e| (e - mean_energy).powi(2))
                .sum::<f32>();
            #[allow(clippy::cast_precision_loss)]
            let variance = variance / energies.len() as f32;
            variance.sqrt()
        };

        // Calculate energy dynamics (rate of change)
        let energy_dynamics = if energies.len() > 1 {
            let dynamics = energies
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f32>();
            #[allow(clippy::cast_precision_loss)]
            let dynamics = dynamics / (energies.len() - 1) as f32;
            dynamics
        } else {
            0.0
        };

        Ok(EnergyFeatures {
            mean_energy,
            energy_variation,
            energy_dynamics,
        })
    }

    /// Extract spectral features for emotion analysis.
    ///
    /// Returns centroid, spread (both in Hz), and spectral flux (normalised,
    /// approximately 0–1).
    pub(super) async fn extract_spectral_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<SpectralFeatures, RecognitionError> {
        let samples = audio.samples();

        // The one-sided rfft spectrum has `n_fft/2 + 1` bins; bin `i`
        // corresponds to frequency  `i * sample_rate / n_fft`.
        // spectrum.len() == n_fft/2 + 1  →  n_fft = (spectrum.len()-1)*2
        let spectrum = self.compute_spectrum(samples).await?;
        let n_fft = (spectrum.len() - 1) * 2;

        #[allow(clippy::cast_precision_loss)]
        let bin_hz = audio.sample_rate() as f32 / n_fft as f32;

        // Spectral centroid
        let mut weighted_sum = 0.0_f32;
        let mut magnitude_sum = 0.0_f32;
        for (i, &mag) in spectrum.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let frequency = i as f32 * bin_hz;
            weighted_sum += frequency * mag;
            magnitude_sum += mag;
        }
        let spectral_centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };

        // Spectral spread
        let spectral_spread = if magnitude_sum > 0.0 {
            let mut spread_sum = 0.0_f32;
            for (i, &mag) in spectrum.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let frequency = i as f32 * bin_hz;
                spread_sum += (frequency - spectral_centroid).powi(2) * mag;
            }
            (spread_sum / magnitude_sum).sqrt()
        } else {
            0.0
        };

        // Real spectral flux via half-wave rectified frame-to-frame difference.
        //
        // Frame the signal with hop = self.hop_size, compute per-frame magnitude
        // spectra, then accumulate the positive differences between consecutive
        // frames.  Normalise by the mean magnitude so the result lives roughly
        // in [0, 1].
        let spectral_flux = self.compute_spectral_flux(samples).await?;

        Ok(SpectralFeatures {
            centroid: spectral_centroid,
            spread: spectral_spread,
            flux: spectral_flux,
        })
    }

    /// Compute half-wave-rectified spectral flux across the signal.
    ///
    /// For each pair of consecutive frames `(t-1, t)` the per-bin flux is
    /// `max(0, |mag_t[i]| - |mag_{t-1}[i]|)`.  The frame-level flux is the
    /// mean of those per-bin values.  The overall flux is the mean across all
    /// frames, then normalised by the global mean magnitude so the result is
    /// roughly in [0, 1].  Returns 0.0 when fewer than two frames are
    /// available.
    pub(super) async fn compute_spectral_flux(
        &self,
        samples: &[f32],
    ) -> Result<f32, RecognitionError> {
        // Collect magnitude spectra for every frame
        let mut frame_spectra: Vec<Vec<f32>> = Vec::new();
        let mut pos = 0usize;
        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let mag = self.compute_spectrum(frame).await?;
            frame_spectra.push(mag);
            pos += self.hop_size;
        }
        // Handle a trailing partial frame so very short signals get at least 1
        if frame_spectra.is_empty() && !samples.is_empty() {
            // zero-pad to frame_size and compute once
            let mag = self.compute_spectrum(samples).await?;
            frame_spectra.push(mag);
        }

        if frame_spectra.len() < 2 {
            return Ok(0.0);
        }

        let n_bins = frame_spectra[0].len();
        let n_frames = frame_spectra.len();

        // Accumulate half-wave-rectified inter-frame differences
        let mut flux_sum = 0.0_f32;
        for t in 1..n_frames {
            let prev = &frame_spectra[t - 1];
            let curr = &frame_spectra[t];
            let mut frame_flux = 0.0_f32;
            for i in 0..n_bins {
                let diff = curr[i] - prev[i];
                if diff > 0.0 {
                    frame_flux += diff;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            {
                flux_sum += frame_flux / n_bins as f32;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let mean_flux = flux_sum / (n_frames - 1) as f32;

        // Normalise by the global mean magnitude so the output is ~[0, 1]
        #[allow(clippy::cast_precision_loss)]
        let global_mean_mag: f32 = frame_spectra
            .iter()
            .flat_map(|s| s.iter().copied())
            .sum::<f32>()
            / (n_frames * n_bins) as f32;

        let normalised_flux = if global_mean_mag > 0.0 {
            (mean_flux / global_mean_mag).min(1.0)
        } else {
            0.0
        };

        Ok(normalised_flux)
    }

    /// Classify primary emotion
    pub(super) fn classify_primary_emotion(
        f0_features: &F0Characteristics,
        energy_features: &EnergyFeatures,
        spectral_features: &SpectralFeatures,
    ) -> Emotion {
        // Simple rule-based emotion classification
        let high_arousal =
            f0_features.f0_variation > 30.0 || energy_features.energy_variation > 0.1;
        let high_valence = f0_features.mean_f0 > 200.0 && spectral_features.centroid > 2000.0;
        let high_energy = energy_features.mean_energy > 0.1;

        match (high_arousal, high_valence, high_energy) {
            (true, true, true) => Emotion::Joy,
            (true, false, true) => Emotion::Anger,
            (true, false, false) => Emotion::Fear,
            (false, false, false) => Emotion::Sadness,
            (true, true, false) => Emotion::Surprise,
            _ => Emotion::Neutral,
        }
    }

    /// Calculate emotion scores for all emotions
    pub(super) fn calculate_emotion_scores(
        f0_features: &F0Characteristics,
        energy_features: &EnergyFeatures,
        spectral_features: &SpectralFeatures,
    ) -> HashMap<Emotion, f32> {
        let mut scores = HashMap::new();

        // Normalize features
        let f0_norm = (f0_features.mean_f0 / 300.0).min(1.0);
        let f0_var_norm = (f0_features.f0_variation / 50.0).min(1.0);
        let energy_norm = (energy_features.mean_energy / 0.5).min(1.0);
        let energy_var_norm = (energy_features.energy_variation / 0.2).min(1.0);
        let centroid_norm = (spectral_features.centroid / 4000.0).min(1.0);

        // Calculate scores for each emotion
        scores.insert(Emotion::Joy, (f0_norm + energy_norm + centroid_norm) / 3.0);
        scores.insert(Emotion::Sadness, (1.0 - f0_norm + 1.0 - energy_norm) / 2.0);
        scores.insert(
            Emotion::Anger,
            (f0_var_norm + energy_var_norm + energy_norm) / 3.0,
        );
        scores.insert(Emotion::Fear, (f0_var_norm + centroid_norm) / 2.0);
        scores.insert(Emotion::Surprise, (f0_var_norm + energy_norm) / 2.0);
        scores.insert(Emotion::Disgust, centroid_norm * 0.5);
        scores.insert(Emotion::Neutral, 1.0 - f0_var_norm.max(energy_var_norm));

        // Normalize scores to sum to 1
        let total: f32 = scores.values().sum();
        if total > 0.0 {
            for score in scores.values_mut() {
                *score /= total;
            }
        }

        scores
    }

    /// Calculate emotion dimensions (valence and arousal)
    pub(super) fn calculate_emotion_dimensions(
        emotion_scores: &HashMap<Emotion, f32>,
    ) -> (f32, f32) {
        // Define emotion positions in valence-arousal space
        let emotion_positions = [
            (Emotion::Joy, (0.8, 0.8)),
            (Emotion::Sadness, (-0.7, -0.5)),
            (Emotion::Anger, (-0.6, 0.7)),
            (Emotion::Fear, (-0.5, 0.6)),
            (Emotion::Surprise, (0.2, 0.7)),
            (Emotion::Disgust, (-0.6, 0.3)),
            (Emotion::Neutral, (0.0, 0.0)),
        ];

        let mut valence = 0.0;
        let mut arousal = 0.0;

        for (emotion, (val, ar)) in &emotion_positions {
            if let Some(&score) = emotion_scores.get(emotion) {
                valence += val * score;
                arousal += ar * score;
            }
        }

        (valence.clamp(-1.0, 1.0), arousal.clamp(-1.0, 1.0))
    }

    /// Calculate emotional intensity
    pub(super) fn calculate_emotional_intensity(
        f0_features: &F0Characteristics,
        energy_features: &EnergyFeatures,
    ) -> f32 {
        let f0_intensity = f0_features.f0_variation / 50.0;
        let energy_intensity = energy_features.energy_variation / 0.2;

        ((f0_intensity + energy_intensity) / 2.0).min(1.0)
    }

    // Helper methods

    /// Extract pitch using simple autocorrelation
    async fn extract_pitch_simple(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut pitch_contour = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let pitch = self.autocorr_pitch(frame);
            pitch_contour.push(pitch);
            pos += self.hop_size;
        }

        Ok(pitch_contour)
    }

    /// Simple autocorrelation-based pitch detection
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn autocorr_pitch(&self, frame: &[f32]) -> f32 {
        let min_period = (self.sample_rate / 800.0) as usize; // Max F0
        let max_period = (self.sample_rate / 50.0) as usize; // Min F0

        let mut max_corr = 0.0;
        let mut best_period = 0;

        for period in min_period..max_period.min(frame.len() / 2) {
            let mut correlation = 0.0;
            for i in 0..frame.len() - period {
                correlation += frame[i] * frame[i + period];
            }

            if correlation > max_corr {
                max_corr = correlation;
                best_period = period;
            }
        }

        if best_period > 0 && max_corr > 0.1 {
            #[allow(clippy::cast_precision_loss)]
            {
                self.sample_rate / best_period as f32
            }
        } else {
            0.0
        }
    }

    /// Compute magnitude spectrum using a windowed FFT.
    ///
    /// Applies a Hann window to `samples` (zero-padded to `frame_size` if shorter),
    /// runs a real-valued FFT, and returns the one-sided magnitude spectrum of
    /// length `frame_size / 2 + 1`.  Bin `i` corresponds to frequency
    /// `i * sample_rate / frame_size` Hz.
    pub(super) async fn compute_spectrum(
        &self,
        samples: &[f32],
    ) -> Result<Vec<f32>, RecognitionError> {
        let n_fft = self.frame_size;
        let n_out = n_fft / 2 + 1;

        // Build Hann-windowed frame (zero-pad when samples is shorter than n_fft)
        let use_len = samples.len().min(n_fft);
        let mut windowed: Vec<f64> = Vec::with_capacity(n_fft);
        for i in 0..use_len {
            #[allow(clippy::cast_precision_loss)]
            let w =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n_fft as f64 - 1.0)).cos());
            windowed.push(f64::from(samples[i]) * w);
        }
        // Zero-pad remainder
        windowed.resize(n_fft, 0.0_f64);

        // Real-FFT: returns n_fft/2 + 1 complex bins
        let spectrum_complex = scirs2_fft::rfft(&windowed, Some(n_fft)).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("FFT computation failed: {e}"),
                source: None,
            }
        })?;

        debug_assert_eq!(spectrum_complex.len(), n_out);

        let magnitude: Vec<f32> = spectrum_complex
            .iter()
            .take(n_out)
            .map(|c| {
                #[allow(clippy::cast_precision_loss)]
                let mag = (c.re * c.re + c.im * c.im).sqrt() as f32;
                mag
            })
            .collect();

        Ok(magnitude)
    }

    /// Calculate jitter (pitch period variation)
    async fn calculate_jitter(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified jitter calculation
        let pitch_periods = self.extract_pitch_periods(samples).await?;

        if pitch_periods.len() < 2 {
            return Ok(0.0);
        }

        #[allow(clippy::cast_precision_loss)]
        let mean_period = pitch_periods.iter().sum::<f32>() / pitch_periods.len() as f32;

        let period_variations: Vec<f32> = pitch_periods
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let mean_variation = period_variations.iter().sum::<f32>() / period_variations.len() as f32;

        Ok(mean_variation / mean_period)
    }

    /// Calculate shimmer (amplitude variation)
    async fn calculate_shimmer(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Calculate frame-by-frame amplitudes
        let mut amplitudes = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let amplitude = frame
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            amplitudes.push(amplitude);
            pos += self.hop_size;
        }

        if amplitudes.len() < 2 {
            return Ok(0.0);
        }

        let amplitude_variations: Vec<f32> = amplitudes
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 {
                    (w[1] - w[0]).abs() / w[0]
                } else {
                    0.0
                }
            })
            .collect();

        #[allow(clippy::cast_precision_loss)]
        Ok(amplitude_variations.iter().sum::<f32>() / amplitude_variations.len() as f32)
    }

    /// Calculate harmonics-to-noise ratio
    async fn calculate_hnr(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified HNR calculation
        let spectrum = self.compute_spectrum(samples).await?;

        // Find harmonic peaks (simplified)
        let fundamental_bin = spectrum
            .iter()
            .enumerate()
            .skip(5) // Skip very low frequencies
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(10, |(i, _)| i);

        // Calculate harmonic energy
        let mut harmonic_energy = 0.0;
        for harmonic in 1..=5 {
            let bin = fundamental_bin * harmonic;
            if bin < spectrum.len() {
                harmonic_energy += spectrum[bin];
            }
        }

        // Calculate noise energy (simplified)
        let total_energy: f32 = spectrum.iter().sum();
        let noise_energy = total_energy - harmonic_energy;

        if noise_energy > 0.0 {
            Ok(10.0 * (harmonic_energy / noise_energy).log10())
        } else {
            Ok(30.0) // High HNR
        }
    }

    /// Extract pitch periods for jitter calculation
    async fn extract_pitch_periods(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simplified pitch period extraction
        let mut periods = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let pitch = self.autocorr_pitch(frame);

            if pitch > 0.0 {
                let period = self.sample_rate / pitch;
                periods.push(period);
            }

            pos += self.hop_size;
        }

        Ok(periods)
    }
}
