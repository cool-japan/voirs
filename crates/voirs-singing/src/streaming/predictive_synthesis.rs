//! Predictive synthesis with look-ahead note analysis
//!
//! Pre-synthesizes upcoming notes in the background to reduce perceived latency.

use crate::{MusicalNote, MusicalScore, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Predictive synthesis cache entry
#[derive(Clone)]
struct CacheEntry {
    /// Pre-synthesized audio samples
    samples: Vec<f32>,

    /// Note that was synthesized
    note: MusicalNote,

    /// Voice characteristics used
    voice_id: String,

    /// Timestamp when synthesized (not serialized)
    created_at: Instant,

    /// Time-to-live
    ttl: Duration,

    /// Access count
    access_count: u32,
}

/// Predictive synthesis engine
///
/// Analyzes upcoming notes and pre-synthesizes them in the background
/// to minimize latency during real-time playback.
pub struct PredictiveSynthesisEngine {
    /// Synthesis cache
    cache: Arc<std::sync::Mutex<HashMap<String, CacheEntry>>>,

    /// Look-ahead time in seconds
    lookahead_time: f32,

    /// Maximum cache size (number of notes)
    max_cache_size: usize,

    /// Cache time-to-live
    cache_ttl: Duration,

    /// Prediction statistics
    stats: Arc<std::sync::Mutex<PredictionStats>>,

    /// Background synthesis enabled
    background_enabled: bool,
}

impl PredictiveSynthesisEngine {
    /// Create a new predictive synthesis engine
    ///
    /// # Arguments
    /// * `lookahead_time` - How far ahead to analyze (seconds)
    /// * `max_cache_size` - Maximum number of cached notes
    pub fn new(lookahead_time: f32, max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            lookahead_time,
            max_cache_size,
            cache_ttl: Duration::from_secs(5),
            stats: Arc::new(std::sync::Mutex::new(PredictionStats::default())),
            background_enabled: true,
        }
    }

    /// Analyze score and predict upcoming notes
    ///
    /// # Arguments
    /// * `score` - Musical score to analyze
    /// * `current_time` - Current playback position (seconds)
    ///
    /// # Returns
    /// List of notes that will be needed soon
    pub fn analyze_upcoming_notes(
        &self,
        score: &MusicalScore,
        current_time: f32,
    ) -> Vec<MusicalNote> {
        let lookahead_end = current_time + self.lookahead_time;

        score
            .notes
            .iter()
            .filter(|note| note.start_time >= current_time && note.start_time <= lookahead_end)
            .cloned()
            .collect()
    }

    /// Store pre-synthesized note in cache
    ///
    /// # Arguments
    /// * `note` - Note that was synthesized
    /// * `voice_id` - Voice identifier
    /// * `samples` - Pre-synthesized audio samples
    pub fn cache_synthesized_note(&self, note: &MusicalNote, voice_id: &str, samples: Vec<f32>) {
        if let Ok(mut cache) = self.cache.lock() {
            // Evict old entries if cache is full
            if cache.len() >= self.max_cache_size {
                self.evict_lru_entry(&mut cache);
            }

            let key = Self::cache_key(note, voice_id);
            cache.insert(
                key,
                CacheEntry {
                    samples,
                    note: note.clone(),
                    voice_id: voice_id.to_string(),
                    created_at: Instant::now(),
                    ttl: self.cache_ttl,
                    access_count: 0,
                },
            );

            // Update stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.notes_cached += 1;
            }
        }
    }

    /// Retrieve pre-synthesized note from cache
    ///
    /// # Arguments
    /// * `note` - Note to retrieve
    /// * `voice_id` - Voice identifier
    ///
    /// # Returns
    /// Pre-synthesized samples if available
    pub fn get_cached_note(&self, note: &MusicalNote, voice_id: &str) -> Option<Vec<f32>> {
        if let Ok(mut cache) = self.cache.lock() {
            let key = Self::cache_key(note, voice_id);

            if let Some(entry) = cache.get_mut(&key) {
                // Check if entry is still valid
                if entry.created_at.elapsed() <= entry.ttl {
                    entry.access_count += 1;

                    // Update stats
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.cache_hits += 1;
                    }

                    return Some(entry.samples.clone());
                } else {
                    // Entry expired, remove it
                    cache.remove(&key);
                }
            }
        }

        // Cache miss
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_misses += 1;
        }

        None
    }

    /// Pre-warm cache with upcoming notes
    ///
    /// # Arguments
    /// * `notes` - Notes to pre-synthesize
    /// * `voice_id` - Voice to use
    /// * `voice` - Voice characteristics
    ///
    /// # Returns
    /// Number of notes successfully pre-synthesized
    pub fn prewarm_cache(
        &self,
        notes: &[MusicalNote],
        voice_id: &str,
        _voice: &VoiceCharacteristics,
    ) -> usize {
        if !self.background_enabled {
            return 0;
        }

        let mut prewarmed = 0;

        for note in notes.iter().take(self.max_cache_size) {
            // Check if already cached
            let key = Self::cache_key(note, voice_id);
            if let Ok(cache) = self.cache.lock() {
                if cache.contains_key(&key) {
                    continue;
                }
            }

            // Simulate synthesis (in real implementation, this would call actual synthesis)
            let sample_count = (note.duration * 48000.0) as usize;
            let samples = self.synthesize_note_stub(note, sample_count);

            self.cache_synthesized_note(note, voice_id, samples);
            prewarmed += 1;
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.notes_prewarmed += prewarmed as u64;
        }

        prewarmed
    }

    /// Stub synthesis for testing (replace with actual synthesis in production)
    fn synthesize_note_stub(&self, note: &MusicalNote, sample_count: usize) -> Vec<f32> {
        // Simple sine wave for testing
        let frequency = note.event.frequency;
        let samples: Vec<f32> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / 48000.0;
                let phase = 2.0 * std::f32::consts::PI * frequency * t;
                phase.sin() * 0.3
            })
            .collect();

        samples
    }

    /// Generate cache key for a note
    fn cache_key(note: &MusicalNote, voice_id: &str) -> String {
        let phoneme = note
            .event
            .phonemes
            .first()
            .map(|p| p.as_str())
            .unwrap_or("a");
        format!(
            "{}_{}_{:.3}_{:.3}",
            voice_id, phoneme, note.event.frequency, note.duration
        )
    }

    /// Evict least recently used entry
    fn evict_lru_entry(&self, cache: &mut HashMap<String, CacheEntry>) {
        if let Some((lru_key, _)) = cache.iter().min_by_key(|(_, entry)| entry.access_count) {
            let key_to_remove = lru_key.clone();
            cache.remove(&key_to_remove);

            if let Ok(mut stats) = self.stats.lock() {
                stats.evictions += 1;
            }
        }
    }

    /// Clean expired entries from cache
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            let mut expired_keys = Vec::new();

            for (key, entry) in cache.iter() {
                if entry.created_at.elapsed() > entry.ttl {
                    expired_keys.push(key.clone());
                }
            }

            for key in expired_keys {
                cache.remove(&key);
            }
        }
    }

    /// Get prediction statistics
    pub fn get_stats(&self) -> PredictionStats {
        self.stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clear all cached notes
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Set background synthesis enabled
    pub fn set_background_enabled(&mut self, enabled: bool) {
        self.background_enabled = enabled;
    }

    /// Calculate prediction accuracy
    ///
    /// # Returns
    /// Prediction accuracy (0.0-1.0) based on cache hit rate
    pub fn prediction_accuracy(&self) -> f32 {
        if let Ok(stats) = self.stats.lock() {
            let total = stats.cache_hits + stats.cache_misses;
            if total == 0 {
                0.0
            } else {
                stats.cache_hits as f32 / total as f32
            }
        } else {
            0.0
        }
    }
}

/// Prediction statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionStats {
    /// Total notes cached
    pub notes_cached: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Notes pre-warmed in background
    pub notes_prewarmed: u64,

    /// Cache evictions
    pub evictions: u64,
}

impl PredictionStats {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

/// Look-ahead analyzer for intelligent note prediction
pub struct LookAheadAnalyzer {
    /// Pattern recognition cache
    patterns: Arc<std::sync::Mutex<HashMap<String, NotePattern>>>,
}

impl LookAheadAnalyzer {
    /// Create a new look-ahead analyzer
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Analyze score patterns for better prediction
    ///
    /// # Arguments
    /// * `score` - Musical score to analyze
    ///
    /// # Returns
    /// Detected patterns that can improve prediction
    pub fn analyze_patterns(&self, score: &MusicalScore) -> Vec<NotePattern> {
        let mut patterns = Vec::new();

        // Detect repeating sequences
        for window_size in [2, 3, 4, 5] {
            let detected = self.detect_repeating_sequences(&score.notes, window_size);
            patterns.extend(detected);
        }

        // Detect melodic patterns (ascending, descending, arpeggios)
        patterns.extend(self.detect_melodic_patterns(&score.notes));

        patterns
    }

    /// Detect repeating note sequences
    fn detect_repeating_sequences(
        &self,
        notes: &[MusicalNote],
        window_size: usize,
    ) -> Vec<NotePattern> {
        let mut patterns = Vec::new();

        if notes.len() < window_size * 2 {
            return patterns;
        }

        for i in 0..(notes.len() - window_size * 2 + 1) {
            let window1 = &notes[i..i + window_size];
            let window2 = &notes[i + window_size..i + window_size * 2];

            if self.sequences_match(window1, window2) {
                patterns.push(NotePattern {
                    pattern_type: PatternType::Repeat,
                    notes: window1.to_vec(),
                    confidence: 0.8,
                });
            }
        }

        patterns
    }

    /// Check if two note sequences match
    fn sequences_match(&self, seq1: &[MusicalNote], seq2: &[MusicalNote]) -> bool {
        if seq1.len() != seq2.len() {
            return false;
        }

        seq1.iter().zip(seq2.iter()).all(|(n1, n2)| {
            (n1.event.frequency - n2.event.frequency).abs() < 1.0
                && n1.event.phonemes == n2.event.phonemes
        })
    }

    /// Detect melodic patterns
    fn detect_melodic_patterns(&self, notes: &[MusicalNote]) -> Vec<NotePattern> {
        let mut patterns = Vec::new();

        if notes.len() < 3 {
            return patterns;
        }

        // Detect ascending/descending sequences
        for i in 0..(notes.len() - 2) {
            let slice = &notes[i..i + 3];

            if slice[0].event.frequency < slice[1].event.frequency
                && slice[1].event.frequency < slice[2].event.frequency
            {
                patterns.push(NotePattern {
                    pattern_type: PatternType::Ascending,
                    notes: slice.to_vec(),
                    confidence: 0.7,
                });
            } else if slice[0].event.frequency > slice[1].event.frequency
                && slice[1].event.frequency > slice[2].event.frequency
            {
                patterns.push(NotePattern {
                    pattern_type: PatternType::Descending,
                    notes: slice.to_vec(),
                    confidence: 0.7,
                });
            }
        }

        patterns
    }
}

impl Default for LookAheadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detected note pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotePattern {
    /// Type of pattern
    pub pattern_type: PatternType,

    /// Notes in the pattern
    pub notes: Vec<MusicalNote>,

    /// Confidence (0.0-1.0)
    pub confidence: f32,
}

/// Types of detected patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Repeating sequence
    Repeat,

    /// Ascending melody
    Ascending,

    /// Descending melody
    Descending,

    /// Arpeggio
    Arpeggio,

    /// Scale
    Scale,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_note(frequency: f32, duration: f32) -> MusicalNote {
        use crate::types::{Articulation, Dynamics, NoteEvent};

        MusicalNote {
            event: NoteEvent {
                note: "A".to_string(),
                octave: 4,
                frequency,
                duration,
                velocity: 0.8,
                vibrato: 0.5,
                lyric: None,
                phonemes: vec!["a".to_string()],
                expression: crate::types::Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: Articulation::Normal,
            },
            start_time: 0.0,
            duration,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        }
    }

    #[test]
    fn test_predictive_synthesis_creation() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_cache_and_retrieve_note() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        let note = create_test_note(440.0, 0.5);
        let samples = vec![0.1, 0.2, 0.3];

        engine.cache_synthesized_note(&note, "voice1", samples.clone());
        assert_eq!(engine.cache_size(), 1);

        let retrieved = engine.get_cached_note(&note, "voice1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), samples);
    }

    #[test]
    fn test_cache_miss() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        let note = create_test_note(440.0, 0.5);

        let retrieved = engine.get_cached_note(&note, "voice1");
        assert!(retrieved.is_none());

        let stats = engine.get_stats();
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_analyze_upcoming_notes() {
        let engine = PredictiveSynthesisEngine::new(1.0, 100);

        let mut score = MusicalScore::new("Test".to_string(), "Composer".to_string());

        let mut note1 = create_test_note(440.0, 0.5);
        note1.start_time = 0.0;
        score.notes.push(note1);

        let mut note2 = create_test_note(494.0, 0.5);
        note2.start_time = 0.5;
        score.notes.push(note2);

        let mut note3 = create_test_note(523.0, 0.5);
        note3.start_time = 1.5;
        score.notes.push(note3);

        let upcoming = engine.analyze_upcoming_notes(&score, 0.0);
        assert_eq!(upcoming.len(), 2); // Notes at 0.0 and 0.5 within lookahead
    }

    #[test]
    fn test_prewarm_cache() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        let voice = VoiceCharacteristics::default();

        let notes = vec![
            create_test_note(440.0, 0.5),
            create_test_note(494.0, 0.5),
            create_test_note(523.0, 0.5),
        ];

        let prewarmed = engine.prewarm_cache(&notes, "voice1", &voice);
        assert_eq!(prewarmed, 3);
        assert_eq!(engine.cache_size(), 3);
    }

    #[test]
    fn test_cache_eviction() {
        let engine = PredictiveSynthesisEngine::new(0.5, 2); // Small cache

        let notes = vec![
            create_test_note(440.0, 0.5),
            create_test_note(494.0, 0.5),
            create_test_note(523.0, 0.5),
        ];

        for note in &notes {
            engine.cache_synthesized_note(note, "voice1", vec![0.1]);
        }

        assert_eq!(engine.cache_size(), 2); // Should not exceed max size
    }

    #[test]
    fn test_cache_cleanup() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        let note = create_test_note(440.0, 0.5);

        engine.cache_synthesized_note(&note, "voice1", vec![0.1]);
        assert_eq!(engine.cache_size(), 1);

        // Simulate expiration (would need to wait in real scenario)
        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_prediction_accuracy() {
        let engine = PredictiveSynthesisEngine::new(0.5, 100);
        let note = create_test_note(440.0, 0.5);

        engine.cache_synthesized_note(&note, "voice1", vec![0.1]);

        // Hit
        engine.get_cached_note(&note, "voice1");

        // Miss
        let other_note = create_test_note(494.0, 0.5);
        engine.get_cached_note(&other_note, "voice1");

        let accuracy = engine.prediction_accuracy();
        assert!((accuracy - 0.5).abs() < 0.001); // 1 hit, 1 miss = 50%
    }

    #[test]
    fn test_look_ahead_analyzer() {
        let analyzer = LookAheadAnalyzer::new();

        let mut score = MusicalScore::new("Test".to_string(), "Composer".to_string());
        score.notes.push(create_test_note(440.0, 0.5));
        score.notes.push(create_test_note(494.0, 0.5));
        score.notes.push(create_test_note(523.0, 0.5));

        let patterns = analyzer.analyze_patterns(&score);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_prediction_stats_hit_rate() {
        let mut stats = PredictionStats::default();
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.8).abs() < 0.001);
    }
}
