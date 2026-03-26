//! Comprehensive dataset profiling tools
//!
//! This module provides tools for analyzing and profiling speech synthesis datasets,
//! including audio quality metrics, text statistics, speaker distribution, and more.

use crate::traits::{Dataset, DatasetSample};
use crate::{AudioData, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Comprehensive dataset profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProfile {
    /// Basic statistics
    pub basic_stats: BasicStatistics,
    /// Audio characteristics
    pub audio_stats: AudioStatistics,
    /// Text characteristics
    pub text_stats: TextStatistics,
    /// Speaker distribution
    pub speaker_stats: SpeakerStatistics,
    /// Performance metrics
    pub performance_stats: PerformanceStatistics,
}

/// Basic dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicStatistics {
    /// Total number of samples
    pub total_samples: usize,
    /// Total duration in seconds
    pub total_duration_seconds: f64,
    /// Average sample duration
    pub avg_duration_seconds: f64,
    /// Minimum sample duration
    pub min_duration_seconds: f64,
    /// Maximum sample duration
    pub max_duration_seconds: f64,
    /// Standard deviation of durations
    pub duration_std_dev: f64,
    /// Total size in bytes
    pub total_size_bytes: usize,
}

/// Audio characteristics statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStatistics {
    /// Sample rate distribution
    pub sample_rate_distribution: HashMap<u32, usize>,
    /// Most common sample rate
    pub most_common_sample_rate: u32,
    /// Channel distribution
    pub channel_distribution: HashMap<u32, usize>,
    /// Average RMS level
    pub avg_rms: f32,
    /// Average peak level
    pub avg_peak: f32,
    /// Dynamic range (dB)
    pub avg_dynamic_range_db: f32,
    /// Average zero-crossing rate
    pub avg_zero_crossing_rate: f32,
    /// Silence ratio (percentage of samples below threshold)
    pub silence_ratio: f32,
}

/// Text characteristics statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Total number of words
    pub total_words: usize,
    /// Total number of characters
    pub total_characters: usize,
    /// Average text length (characters)
    pub avg_text_length: f64,
    /// Average word count per sample
    pub avg_word_count: f64,
    /// Unique words (vocabulary size)
    pub vocabulary_size: usize,
    /// Character set used
    pub character_set: Vec<char>,
    /// Most common words (top 10)
    pub common_words: Vec<(String, usize)>,
    /// Language distribution (if available)
    pub language_distribution: HashMap<String, usize>,
}

/// Speaker distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerStatistics {
    /// Number of unique speakers
    pub num_speakers: usize,
    /// Samples per speaker
    pub samples_per_speaker: HashMap<String, usize>,
    /// Duration per speaker (seconds)
    pub duration_per_speaker: HashMap<String, f64>,
    /// Most active speaker
    pub most_active_speaker: Option<String>,
    /// Least active speaker
    pub least_active_speaker: Option<String>,
    /// Speaker balance score (0.0 = unbalanced, 1.0 = perfectly balanced)
    pub balance_score: f64,
}

/// Performance profiling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Time to profile dataset (seconds)
    pub profiling_time_seconds: f64,
    /// Average sample load time (milliseconds)
    pub avg_sample_load_time_ms: f64,
    /// Peak memory usage estimate (bytes)
    pub peak_memory_bytes: usize,
    /// Samples processed per second
    pub samples_per_second: f64,
}

/// Dataset profiler for comprehensive analysis
pub struct DatasetProfiler;

impl DatasetProfiler {
    /// Profile a dataset comprehensively
    pub async fn profile<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
    ) -> Result<DatasetProfile> {
        let start_time = Instant::now();

        let basic_stats = Self::profile_basic_stats(dataset).await?;
        let audio_stats = Self::profile_audio_stats(dataset).await?;
        let text_stats = Self::profile_text_stats(dataset).await?;
        let speaker_stats = Self::profile_speaker_stats(dataset).await?;

        let profiling_time = start_time.elapsed().as_secs_f64();
        let performance_stats =
            Self::profile_performance_stats(dataset, profiling_time, &basic_stats).await?;

        Ok(DatasetProfile {
            basic_stats,
            audio_stats,
            text_stats,
            speaker_stats,
            performance_stats,
        })
    }

    /// Profile basic statistics
    async fn profile_basic_stats<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
    ) -> Result<BasicStatistics> {
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(BasicStatistics {
                total_samples: 0,
                total_duration_seconds: 0.0,
                avg_duration_seconds: 0.0,
                min_duration_seconds: 0.0,
                max_duration_seconds: 0.0,
                duration_std_dev: 0.0,
                total_size_bytes: 0,
            });
        }

        let mut durations = Vec::with_capacity(total_samples);
        let mut total_size = 0usize;

        for i in 0..total_samples {
            let sample = dataset.get(i).await?;
            let duration = sample.audio.duration() as f64;
            durations.push(duration);
            total_size += std::mem::size_of_val(sample.audio.samples());
        }

        let total_duration: f64 = durations.iter().sum();
        let avg_duration = total_duration / total_samples as f64;
        let min_duration = durations.iter().copied().fold(f64::INFINITY, f64::min);
        let max_duration = durations.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Calculate standard deviation
        let variance: f64 = durations
            .iter()
            .map(|&d| (d - avg_duration).powi(2))
            .sum::<f64>()
            / total_samples as f64;
        let duration_std_dev = variance.sqrt();

        Ok(BasicStatistics {
            total_samples,
            total_duration_seconds: total_duration,
            avg_duration_seconds: avg_duration,
            min_duration_seconds: min_duration,
            max_duration_seconds: max_duration,
            duration_std_dev,
            total_size_bytes: total_size,
        })
    }

    /// Profile audio characteristics
    async fn profile_audio_stats<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
    ) -> Result<AudioStatistics> {
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(AudioStatistics {
                sample_rate_distribution: HashMap::new(),
                most_common_sample_rate: 0,
                channel_distribution: HashMap::new(),
                avg_rms: 0.0,
                avg_peak: 0.0,
                avg_dynamic_range_db: 0.0,
                avg_zero_crossing_rate: 0.0,
                silence_ratio: 0.0,
            });
        }

        let mut sample_rates: HashMap<u32, usize> = HashMap::new();
        let mut channels: HashMap<u32, usize> = HashMap::new();
        let mut total_rms = 0.0f32;
        let mut total_peak = 0.0f32;
        let mut total_zcr = 0.0f32;
        let mut total_samples_count = 0usize;
        let mut silent_samples_count = 0usize;

        const SILENCE_THRESHOLD: f32 = 0.01;

        for i in 0..total_samples {
            let sample = dataset.get(i).await?;
            let audio = &sample.audio;

            // Sample rate distribution
            *sample_rates.entry(audio.sample_rate()).or_insert(0) += 1;

            // Channel distribution
            *channels.entry(audio.channels()).or_insert(0) += 1;

            // Audio metrics
            let samples = audio.samples();
            let rms = Self::calculate_rms(samples);
            let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            let zcr = Self::calculate_zero_crossing_rate(samples);

            total_rms += rms;
            total_peak += peak;
            total_zcr += zcr;

            // Silence detection
            total_samples_count += samples.len();
            silent_samples_count += samples
                .iter()
                .filter(|&&x| x.abs() < SILENCE_THRESHOLD)
                .count();
        }

        let most_common_sample_rate = sample_rates
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(&sr, _)| sr)
            .unwrap_or(0);

        let avg_rms = total_rms / total_samples as f32;
        let avg_peak = total_peak / total_samples as f32;
        let avg_zcr = total_zcr / total_samples as f32;
        let silence_ratio = if total_samples_count > 0 {
            (silent_samples_count as f32 / total_samples_count as f32) * 100.0
        } else {
            0.0
        };

        // Calculate average dynamic range in dB
        let avg_dynamic_range_db = if avg_rms > 1e-10 {
            20.0 * (avg_peak / avg_rms).log10()
        } else {
            0.0
        };

        Ok(AudioStatistics {
            sample_rate_distribution: sample_rates,
            most_common_sample_rate,
            channel_distribution: channels,
            avg_rms,
            avg_peak,
            avg_dynamic_range_db,
            avg_zero_crossing_rate: avg_zcr,
            silence_ratio,
        })
    }

    /// Profile text characteristics
    async fn profile_text_stats<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
    ) -> Result<TextStatistics> {
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(TextStatistics {
                total_words: 0,
                total_characters: 0,
                avg_text_length: 0.0,
                avg_word_count: 0.0,
                vocabulary_size: 0,
                character_set: vec![],
                common_words: vec![],
                language_distribution: HashMap::new(),
            });
        }

        let mut total_characters = 0usize;
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut character_set: HashSet<char> = HashSet::new();
        let mut language_dist: HashMap<String, usize> = HashMap::new();

        for i in 0..total_samples {
            let sample = dataset.get(i).await?;
            let text = &sample.text;

            total_characters += text.len();

            // Character set analysis
            for ch in text.chars() {
                character_set.insert(ch);
            }

            // Word counting
            let words: Vec<&str> = text.split_whitespace().collect();
            for word in &words {
                let word_lower = word.to_lowercase();
                *word_counts.entry(word_lower).or_insert(0) += 1;
            }

            // Language detection (if metadata available)
            if let Some(language_value) = sample.metadata.get("language") {
                if let Some(language_str) = language_value.as_str() {
                    *language_dist.entry(language_str.to_string()).or_insert(0) += 1;
                }
            }
        }

        let total_words: usize = word_counts.values().sum();
        let vocabulary_size = word_counts.len();
        let avg_text_length = total_characters as f64 / total_samples as f64;
        let avg_word_count = total_words as f64 / total_samples as f64;

        // Get top 10 most common words
        let mut word_vec: Vec<_> = word_counts.into_iter().collect();
        word_vec.sort_by_key(|b| std::cmp::Reverse(b.1));
        let common_words: Vec<_> = word_vec.into_iter().take(10).collect();

        let mut char_vec: Vec<char> = character_set.into_iter().collect();
        char_vec.sort();

        Ok(TextStatistics {
            total_words,
            total_characters,
            avg_text_length,
            avg_word_count,
            vocabulary_size,
            character_set: char_vec,
            common_words,
            language_distribution: language_dist,
        })
    }

    /// Profile speaker distribution
    async fn profile_speaker_stats<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
    ) -> Result<SpeakerStatistics> {
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(SpeakerStatistics {
                num_speakers: 0,
                samples_per_speaker: HashMap::new(),
                duration_per_speaker: HashMap::new(),
                most_active_speaker: None,
                least_active_speaker: None,
                balance_score: 1.0,
            });
        }

        let mut samples_per_speaker: HashMap<String, usize> = HashMap::new();
        let mut duration_per_speaker: HashMap<String, f64> = HashMap::new();

        for i in 0..total_samples {
            let sample = dataset.get(i).await?;
            let speaker = sample
                .speaker
                .as_ref()
                .map(|s| s.id.as_str())
                .unwrap_or("unknown");
            let duration = sample.audio.duration() as f64;

            *samples_per_speaker.entry(speaker.to_string()).or_insert(0) += 1;
            *duration_per_speaker
                .entry(speaker.to_string())
                .or_insert(0.0) += duration;
        }

        let num_speakers = samples_per_speaker.len();

        let most_active_speaker = samples_per_speaker
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(speaker, _)| speaker.clone());

        let least_active_speaker = samples_per_speaker
            .iter()
            .min_by_key(|&(_, count)| count)
            .map(|(speaker, _)| speaker.clone());

        // Calculate balance score (Gini coefficient based)
        let balance_score = Self::calculate_balance_score(&samples_per_speaker);

        Ok(SpeakerStatistics {
            num_speakers,
            samples_per_speaker,
            duration_per_speaker,
            most_active_speaker,
            least_active_speaker,
            balance_score,
        })
    }

    /// Profile performance characteristics
    async fn profile_performance_stats<D: Dataset<Sample = crate::DatasetSample>>(
        dataset: &D,
        profiling_time: f64,
        basic_stats: &BasicStatistics,
    ) -> Result<PerformanceStatistics> {
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(PerformanceStatistics {
                profiling_time_seconds: profiling_time,
                avg_sample_load_time_ms: 0.0,
                peak_memory_bytes: 0,
                samples_per_second: 0.0,
            });
        }

        // Measure average sample load time
        let mut total_load_time = 0.0;
        let samples_to_measure = total_samples.min(100); // Measure up to 100 samples

        for i in 0..samples_to_measure {
            let start = Instant::now();
            let _ = dataset.get(i).await?;
            total_load_time += start.elapsed().as_secs_f64();
        }

        let avg_sample_load_time_ms = (total_load_time / samples_to_measure as f64) * 1000.0;

        let samples_per_second = if profiling_time > 0.0 {
            total_samples as f64 / profiling_time
        } else {
            0.0
        };

        // Estimate peak memory usage
        let peak_memory_bytes = basic_stats.total_size_bytes;

        Ok(PerformanceStatistics {
            profiling_time_seconds: profiling_time,
            avg_sample_load_time_ms,
            peak_memory_bytes,
            samples_per_second,
        })
    }

    /// Calculate RMS of audio samples
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate zero-crossing rate
    fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 0..samples.len() - 1 {
            if (samples[i] >= 0.0 && samples[i + 1] < 0.0)
                || (samples[i] < 0.0 && samples[i + 1] >= 0.0)
            {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Calculate balance score (1.0 = perfectly balanced, 0.0 = completely unbalanced)
    fn calculate_balance_score(distribution: &HashMap<String, usize>) -> f64 {
        if distribution.is_empty() {
            return 1.0;
        }

        let total: usize = distribution.values().sum();
        if total == 0 {
            return 1.0;
        }

        let n = distribution.len() as f64;
        let expected_per_item = total as f64 / n;

        // Calculate normalized variance
        let variance: f64 = distribution
            .values()
            .map(|&count| {
                let diff = count as f64 - expected_per_item;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let max_variance = expected_per_item * expected_per_item * (n - 1.0);

        if max_variance < 1e-10 {
            return 1.0;
        }

        // Balance score: 1.0 - (actual_variance / max_variance)
        (1.0 - (variance / max_variance)).clamp(0.0, 1.0)
    }
}

impl DatasetProfile {
    /// Generate a human-readable report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Dataset Profile Report ===\n\n");

        // Basic Statistics
        report.push_str("Basic Statistics:\n");
        report.push_str(&format!(
            "  Total Samples: {}\n",
            self.basic_stats.total_samples
        ));
        report.push_str(&format!(
            "  Total Duration: {:.2} hours\n",
            self.basic_stats.total_duration_seconds / 3600.0
        ));
        report.push_str(&format!(
            "  Average Duration: {:.2} seconds\n",
            self.basic_stats.avg_duration_seconds
        ));
        report.push_str(&format!(
            "  Duration Range: {:.2}s - {:.2}s\n",
            self.basic_stats.min_duration_seconds, self.basic_stats.max_duration_seconds
        ));
        report.push_str(&format!(
            "  Total Size: {:.2} MB\n\n",
            self.basic_stats.total_size_bytes as f64 / 1_048_576.0
        ));

        // Audio Statistics
        report.push_str("Audio Characteristics:\n");
        report.push_str(&format!(
            "  Sample Rate: {} Hz\n",
            self.audio_stats.most_common_sample_rate
        ));
        report.push_str(&format!("  Average RMS: {:.4}\n", self.audio_stats.avg_rms));
        report.push_str(&format!(
            "  Average Peak: {:.4}\n",
            self.audio_stats.avg_peak
        ));
        report.push_str(&format!(
            "  Dynamic Range: {:.2} dB\n",
            self.audio_stats.avg_dynamic_range_db
        ));
        report.push_str(&format!(
            "  Zero-Crossing Rate: {:.4}\n",
            self.audio_stats.avg_zero_crossing_rate
        ));
        report.push_str(&format!(
            "  Silence Ratio: {:.2}%\n\n",
            self.audio_stats.silence_ratio
        ));

        // Text Statistics
        report.push_str("Text Characteristics:\n");
        report.push_str(&format!(
            "  Vocabulary Size: {}\n",
            self.text_stats.vocabulary_size
        ));
        report.push_str(&format!("  Total Words: {}\n", self.text_stats.total_words));
        report.push_str(&format!(
            "  Avg. Text Length: {:.1} characters\n",
            self.text_stats.avg_text_length
        ));
        report.push_str(&format!(
            "  Avg. Word Count: {:.1}\n",
            self.text_stats.avg_word_count
        ));
        report.push_str(&format!(
            "  Character Set Size: {}\n\n",
            self.text_stats.character_set.len()
        ));

        // Speaker Statistics
        report.push_str("Speaker Distribution:\n");
        report.push_str(&format!(
            "  Number of Speakers: {}\n",
            self.speaker_stats.num_speakers
        ));
        report.push_str(&format!(
            "  Balance Score: {:.3}\n",
            self.speaker_stats.balance_score
        ));
        if let Some(ref most_active) = self.speaker_stats.most_active_speaker {
            report.push_str(&format!("  Most Active Speaker: {}\n", most_active));
        }
        if let Some(ref least_active) = self.speaker_stats.least_active_speaker {
            report.push_str(&format!("  Least Active Speaker: {}\n\n", least_active));
        }

        // Performance Statistics
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!(
            "  Profiling Time: {:.2}s\n",
            self.performance_stats.profiling_time_seconds
        ));
        report.push_str(&format!(
            "  Avg. Sample Load Time: {:.2}ms\n",
            self.performance_stats.avg_sample_load_time_ms
        ));
        report.push_str(&format!(
            "  Processing Speed: {:.1} samples/second\n",
            self.performance_stats.samples_per_second
        ));
        report.push_str(&format!(
            "  Peak Memory: {:.2} MB\n",
            self.performance_stats.peak_memory_bytes as f64 / 1_048_576.0
        ));

        report
    }

    /// Save profile to JSON file
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            DatasetError::FormatError(format!("Failed to serialize profile: {}", e))
        })?;

        std::fs::write(path, json)?;

        Ok(())
    }

    /// Load profile from JSON file
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;

        serde_json::from_str(&json)
            .map_err(|e| DatasetError::FormatError(format!("Failed to deserialize profile: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_rms() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let rms = DatasetProfiler::calculate_rms(&samples);
        let expected = ((0.01 + 0.04 + 0.09 + 0.16) / 4.0f32).sqrt();
        assert!((rms - expected).abs() < 1e-6);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let zcr = DatasetProfiler::calculate_zero_crossing_rate(&samples);
        assert!((zcr - 1.0).abs() < 1e-6); // Every transition is a crossing
    }

    #[test]
    fn test_balance_score_perfect() {
        let mut dist = HashMap::new();
        dist.insert("A".to_string(), 10);
        dist.insert("B".to_string(), 10);
        dist.insert("C".to_string(), 10);

        let score = DatasetProfiler::calculate_balance_score(&dist);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_balance_score_unbalanced() {
        let mut dist = HashMap::new();
        dist.insert("A".to_string(), 100);
        dist.insert("B".to_string(), 1);

        let score = DatasetProfiler::calculate_balance_score(&dist);
        assert!(score < 0.5); // Should be quite unbalanced
    }

    #[test]
    fn test_balance_score_empty() {
        let dist = HashMap::new();
        let score = DatasetProfiler::calculate_balance_score(&dist);
        assert!((score - 1.0).abs() < 1e-6);
    }
}
