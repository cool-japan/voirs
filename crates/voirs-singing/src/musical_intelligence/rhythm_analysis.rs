//! Rhythm analysis and pattern detection

use super::types::{GrooveCharacteristics, RhythmResult};
use crate::score::MusicalScore;
use crate::Result;

/// Rhythm analysis system for tempo, groove, and swing detection
#[derive(Debug, Clone)]
pub struct RhythmAnalyzer {
    /// Minimum tempo for detection in BPM (beats per minute)
    min_tempo: f32,
    /// Maximum tempo for detection in BPM (beats per minute)
    max_tempo: f32,
    /// Analysis confidence threshold (0.0-1.0) for rhythm pattern recognition
    threshold: f32,
}

impl RhythmAnalyzer {
    /// Create a new rhythm analyzer
    pub fn new() -> Self {
        Self {
            min_tempo: 60.0,  // 60 BPM minimum
            max_tempo: 200.0, // 200 BPM maximum
            threshold: 0.6,
        }
    }

    /// Analyze rhythm from musical score
    ///
    /// # Arguments
    ///
    /// * `score` - Musical score with tempo and note timing information
    ///
    /// # Returns
    ///
    /// Rhythm analysis including tempo, time signature, groove characteristics, and swing ratio
    ///
    /// # Errors
    ///
    /// Returns error if rhythm analysis fails
    pub async fn analyze_rhythm(&self, score: &MusicalScore) -> Result<RhythmResult> {
        // Extract timing information from score
        let note_onsets = self.extract_note_onsets(score);

        // Detect tempo
        let tempo = self.detect_tempo(&note_onsets, score.tempo);

        // Detect time signature (use score's time signature as starting point)
        let time_signature = (
            score.time_signature.numerator,
            score.time_signature.denominator,
        );

        // Analyze groove characteristics
        let groove = self.analyze_groove(&note_onsets, tempo, time_signature);

        // Detect swing ratio
        let swing_ratio = self.detect_swing(&note_onsets, tempo);

        Ok(RhythmResult {
            tempo,
            time_signature,
            pattern_name: groove.groove_type.clone(),
            groove,
            confidence: 0.8, // Simplified confidence
            swing_ratio,
        })
    }

    /// Analyze rhythm from audio samples using onset detection
    ///
    /// # Arguments
    ///
    /// * `audio_samples` - Raw audio samples as f32 values
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Rhythm analysis extracted from audio onsets and energy patterns
    ///
    /// # Errors
    ///
    /// Returns error if audio rhythm analysis fails
    pub async fn analyze_audio_rhythm(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmResult> {
        // Detect onsets from audio
        let onsets = self.detect_onsets_from_audio(audio_samples, sample_rate);

        // Detect tempo from onsets
        let tempo = self.detect_tempo_from_onsets(&onsets);

        // Default time signature (could be enhanced with beat tracking)
        let time_signature = (4, 4);

        // Analyze groove
        let groove = self.analyze_groove_from_audio(&onsets, tempo);

        // Detect swing
        let swing_ratio = self.detect_swing_from_onsets(&onsets, tempo);

        Ok(RhythmResult {
            tempo,
            time_signature,
            pattern_name: groove.groove_type.clone(),
            groove,
            confidence: 0.7, // Lower confidence for audio analysis
            swing_ratio,
        })
    }

    /// Extract note onset times from musical score
    ///
    /// # Arguments
    ///
    /// * `score` - Musical score to extract onsets from
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds
    fn extract_note_onsets(&self, score: &MusicalScore) -> Vec<f32> {
        let mut onsets = Vec::new();
        let mut current_time = 0.0;

        for note in &score.notes {
            onsets.push(current_time);
            current_time += note.duration;
        }

        onsets
    }

    /// Detect tempo from note onsets using inter-onset intervals
    ///
    /// # Arguments
    ///
    /// * `onsets` - Note onset times in seconds
    /// * `score_tempo` - Score tempo as fallback value
    ///
    /// # Returns
    ///
    /// Detected tempo in BPM
    fn detect_tempo(&self, onsets: &[f32], score_tempo: f32) -> f32 {
        if onsets.len() < 2 {
            return score_tempo.clamp(self.min_tempo, self.max_tempo);
        }

        // Calculate inter-onset intervals
        let intervals: Vec<f32> = onsets.windows(2).map(|w| w[1] - w[0]).collect();

        if intervals.is_empty() {
            return score_tempo.clamp(self.min_tempo, self.max_tempo);
        }

        // Find most common interval (simplified approach)
        let median_interval = {
            let mut sorted_intervals = intervals.clone();
            sorted_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_intervals[sorted_intervals.len() / 2]
        };

        // Convert to BPM (assuming quarter note beat)
        let tempo = 60.0 / median_interval;
        tempo.clamp(self.min_tempo, self.max_tempo)
    }

    /// Analyze groove characteristics including microtiming and syncopation
    ///
    /// # Arguments
    ///
    /// * `onsets` - Note onset times in seconds
    /// * `tempo` - Detected tempo in BPM
    /// * `time_signature` - Time signature (numerator, denominator)
    ///
    /// # Returns
    ///
    /// Groove characteristics with timing variations and rhythmic patterns
    fn analyze_groove(
        &self,
        onsets: &[f32],
        tempo: f32,
        time_signature: (u8, u8),
    ) -> GrooveCharacteristics {
        let beat_duration = 60.0 / tempo;
        let measure_duration = beat_duration * time_signature.0 as f32;

        // Calculate microtiming variations
        let microtiming = self.calculate_microtiming(onsets, beat_duration);

        // Calculate dynamic patterns (simplified)
        let dynamics = vec![0.8; onsets.len()]; // Placeholder

        // Calculate rhythmic density
        let density = if measure_duration > 0.0 {
            onsets.len() as f32 / (onsets.last().unwrap_or(&0.0) / measure_duration)
        } else {
            1.0
        };

        // Calculate syncopation level (simplified)
        let syncopation = self.calculate_syncopation(onsets, beat_duration);

        GrooveCharacteristics {
            groove_type: self.classify_groove_type(syncopation, &microtiming),
            microtiming,
            dynamics,
            density: density.clamp(0.0, 10.0),
            syncopation: syncopation.clamp(0.0, 1.0),
        }
    }

    /// Calculate microtiming variations from expected beat positions
    ///
    /// # Arguments
    ///
    /// * `onsets` - Note onset times
    /// * `beat_duration` - Duration of one beat in seconds
    ///
    /// # Returns
    ///
    /// Vector of timing deviations from quantized grid in seconds
    fn calculate_microtiming(&self, onsets: &[f32], beat_duration: f32) -> Vec<f32> {
        onsets
            .iter()
            .map(|&onset| {
                let expected_beat = (onset / beat_duration).round() * beat_duration;
                onset - expected_beat
            })
            .collect()
    }

    /// Calculate syncopation level based on off-beat note placement
    ///
    /// # Arguments
    ///
    /// * `onsets` - Note onset times
    /// * `beat_duration` - Duration of one beat in seconds
    ///
    /// # Returns
    ///
    /// Syncopation level (0.0-1.0, higher means more syncopated)
    fn calculate_syncopation(&self, onsets: &[f32], beat_duration: f32) -> f32 {
        let mut syncopation_count = 0;
        let total_onsets = onsets.len();

        for &onset in onsets {
            let beat_position = (onset / beat_duration) % 1.0;

            // Consider off-beat positions as syncopated
            if beat_position > 0.3 && beat_position < 0.7 {
                syncopation_count += 1;
            }
        }

        if total_onsets > 0 {
            syncopation_count as f32 / total_onsets as f32
        } else {
            0.0
        }
    }

    /// Classify groove type based on syncopation and microtiming characteristics
    ///
    /// # Arguments
    ///
    /// * `syncopation` - Syncopation level (0.0-1.0)
    /// * `microtiming` - Microtiming variation values
    ///
    /// # Returns
    ///
    /// Groove type classification ("straight", "shuffled", or "syncopated")
    fn classify_groove_type(&self, syncopation: f32, microtiming: &[f32]) -> String {
        if syncopation > 0.4 {
            "syncopated".to_string()
        } else if microtiming.iter().any(|&mt| mt.abs() > 0.05) {
            "shuffled".to_string()
        } else {
            "straight".to_string()
        }
    }

    /// Detect swing ratio from eighth note timing patterns
    ///
    /// # Arguments
    ///
    /// * `onsets` - Note onset times
    /// * `tempo` - Detected tempo in BPM
    ///
    /// # Returns
    ///
    /// Swing ratio (typically 1.0-1.4) or None if no swing detected
    fn detect_swing(&self, onsets: &[f32], tempo: f32) -> Option<f32> {
        let beat_duration = 60.0 / tempo;
        let eighth_note_duration = beat_duration / 2.0;

        // Find pairs of eighth notes
        let mut swing_ratios = Vec::new();

        for i in 0..onsets.len().saturating_sub(1) {
            let interval = onsets[i + 1] - onsets[i];

            // Check if this could be a swing eighth note pair
            if interval > eighth_note_duration * 0.8 && interval < eighth_note_duration * 1.5 {
                let ratio = interval / eighth_note_duration;
                swing_ratios.push(ratio);
            }
        }

        if swing_ratios.len() >= 2 {
            let avg_ratio = swing_ratios.iter().sum::<f32>() / swing_ratios.len() as f32;
            if avg_ratio > 1.1 && avg_ratio < 1.4 {
                Some(avg_ratio)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Detect onsets from audio using energy-based onset detection
    ///
    /// # Arguments
    ///
    /// * `audio_samples` - Raw audio samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Vector of detected onset times in seconds
    fn detect_onsets_from_audio(&self, audio_samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut onsets = Vec::new();
        let window_size = sample_rate as usize / 100; // 10ms windows
        let hop_size = window_size / 2;

        let mut prev_energy = 0.0;
        let threshold = 0.1;

        for (i, chunk) in audio_samples.chunks(hop_size).enumerate() {
            if chunk.len() < hop_size {
                continue;
            }

            let energy: f32 = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;

            // Simple onset detection using energy increase
            if energy > prev_energy * (1.0 + threshold) && energy > 0.01 {
                let onset_time = i as f32 * hop_size as f32 / sample_rate as f32;
                onsets.push(onset_time);
            }

            prev_energy = energy;
        }

        onsets
    }

    /// Detect tempo from audio onsets using inter-onset interval analysis
    ///
    /// # Arguments
    ///
    /// * `onsets` - Detected onset times from audio
    ///
    /// # Returns
    ///
    /// Estimated tempo in BPM
    fn detect_tempo_from_onsets(&self, onsets: &[f32]) -> f32 {
        if onsets.len() < 4 {
            return 120.0; // Default tempo
        }

        // Calculate inter-onset intervals
        let intervals: Vec<f32> = onsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&interval| interval > 0.1 && interval < 2.0) // Filter reasonable intervals
            .collect();

        if intervals.is_empty() {
            return 120.0;
        }

        // Find most common interval using histogram approach
        let mut tempo_candidates = Vec::new();
        for &interval in &intervals {
            let tempo = 60.0 / interval;
            if tempo >= self.min_tempo && tempo <= self.max_tempo {
                tempo_candidates.push(tempo);
            }
        }

        if tempo_candidates.is_empty() {
            return 120.0;
        }

        // Simple median
        tempo_candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tempo_candidates[tempo_candidates.len() / 2]
    }

    /// Analyze groove from audio onsets
    ///
    /// # Arguments
    ///
    /// * `onsets` - Detected onset times from audio
    /// * `tempo` - Detected tempo in BPM
    ///
    /// # Returns
    ///
    /// Groove characteristics extracted from audio
    fn analyze_groove_from_audio(&self, onsets: &[f32], tempo: f32) -> GrooveCharacteristics {
        let beat_duration = 60.0 / tempo;

        // Simplified groove analysis for audio
        let microtiming = self.calculate_microtiming(onsets, beat_duration);
        let dynamics = vec![0.7; onsets.len()]; // Placeholder

        let density = if onsets.len() > 1 {
            let total_duration = onsets.last().unwrap() - onsets.first().unwrap();
            onsets.len() as f32 / total_duration.max(1.0)
        } else {
            1.0
        };

        let syncopation = self.calculate_syncopation(onsets, beat_duration);

        GrooveCharacteristics {
            groove_type: self.classify_groove_type(syncopation, &microtiming),
            microtiming,
            dynamics,
            density: density.clamp(0.0, 10.0),
            syncopation: syncopation.clamp(0.0, 1.0),
        }
    }

    /// Detect swing from audio onsets
    ///
    /// # Arguments
    ///
    /// * `onsets` - Detected onset times
    /// * `tempo` - Detected tempo in BPM
    ///
    /// # Returns
    ///
    /// Swing ratio or None if no swing detected
    fn detect_swing_from_onsets(&self, onsets: &[f32], tempo: f32) -> Option<f32> {
        self.detect_swing(onsets, tempo)
    }

    /// Set tempo detection range
    ///
    /// # Arguments
    ///
    /// * `min_tempo` - Minimum tempo in BPM (clamped to 30.0)
    /// * `max_tempo` - Maximum tempo in BPM (clamped to 300.0)
    pub fn set_tempo_range(&mut self, min_tempo: f32, max_tempo: f32) {
        self.min_tempo = min_tempo.max(30.0);
        self.max_tempo = max_tempo.min(300.0);
    }

    /// Set analysis threshold for rhythm pattern recognition
    ///
    /// # Arguments
    ///
    /// * `threshold` - Confidence threshold (0.0-1.0) for rhythm pattern detection
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for RhythmAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
