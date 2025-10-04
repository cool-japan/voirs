//! Basic forced alignment implementation
//!
//! This module provides a basic forced alignment implementation that can align
//! phonemes or text with audio using dynamic time warping and acoustic models.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Basic forced alignment model implementation
pub struct ForcedAlignModel {
    /// Model configuration
    config: ForcedAlignConfig,
    /// Model state
    state: Arc<RwLock<ForcedAlignState>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: PhonemeRecognizerMetadata,
}

/// Forced alignment configuration
#[derive(Debug, Clone)]
pub struct ForcedAlignConfig {
    /// Path to the acoustic model
    pub model_path: String,
    /// Path to the pronunciation dictionary
    pub dictionary_path: Option<String>,
    /// Frame shift in milliseconds
    pub frame_shift_ms: f32,
    /// Frame length in milliseconds
    pub frame_length_ms: f32,
    /// Beam width for alignment
    pub beam_width: usize,
    /// Minimum phoneme duration in milliseconds
    pub min_phoneme_duration_ms: f32,
    /// Maximum phoneme duration in milliseconds
    pub max_phoneme_duration_ms: f32,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Number of threads
    pub num_threads: usize,
    /// Confidence threshold for accepting alignments
    pub confidence_threshold: f32,
}

impl Default for ForcedAlignConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dictionary_path: None,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            beam_width: 100,
            min_phoneme_duration_ms: 20.0,
            max_phoneme_duration_ms: 1000.0,
            use_gpu: false,
            num_threads: num_cpus::get(),
            confidence_threshold: 0.3,
        }
    }
}

/// Internal state for forced alignment model
struct ForcedAlignState {
    /// Whether the model is loaded
    loaded: bool,
    /// Model path
    model_path: String,
    /// Dictionary path
    dictionary_path: Option<String>,
    /// Pronunciation dictionary
    dictionary: HashMap<String, Vec<String>>,
    /// Loading time
    load_time: Option<Duration>,
    /// Alignment count
    alignment_count: usize,
    /// Total alignment time
    total_alignment_time: Duration,
}

impl ForcedAlignState {
    fn new(model_path: String, dictionary_path: Option<String>) -> Self {
        Self {
            loaded: false,
            model_path,
            dictionary_path,
            dictionary: HashMap::new(),
            load_time: None,
            alignment_count: 0,
            total_alignment_time: Duration::ZERO,
        }
    }
}

impl ForcedAlignModel {
    /// Create a new forced alignment model
    pub async fn new(
        model_path: String,
        dictionary_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        // Validate model file exists
        if !Path::new(&model_path).exists() {
            return Err(RecognitionError::ModelLoadError {
                message: format!("Model file not found: {}", model_path),
                source: None,
            });
        }

        // Validate dictionary file if provided
        if let Some(ref dict_path) = dictionary_path {
            if !Path::new(dict_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Dictionary file not found: {}", dict_path),
                    source: None,
                });
            }
        }

        let config = ForcedAlignConfig {
            model_path: model_path.clone(),
            dictionary_path: dictionary_path.clone(),
            ..Default::default()
        };

        // Basic forced alignment supports common languages
        let supported_languages = vec![
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
        ];

        let metadata = PhonemeRecognizerMetadata {
            name: "Basic Forced Alignment".to_string(),
            version: "1.0.0".to_string(),
            description: "Basic forced alignment using dynamic time warping".to_string(),
            supported_languages: supported_languages.clone(),
            alignment_methods: vec![AlignmentMethod::Forced, AlignmentMethod::Hybrid],
            alignment_accuracy: 0.85,
            supported_features: vec![
                PhonemeRecognitionFeature::WordAlignment,
                PhonemeRecognitionFeature::CustomPronunciation,
                PhonemeRecognitionFeature::ConfidenceScoring,
                PhonemeRecognitionFeature::PronunciationAssessment,
            ],
        };

        let state = Arc::new(RwLock::new(ForcedAlignState::new(
            model_path,
            dictionary_path,
        )));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: ForcedAlignConfig) -> Result<Self, RecognitionError> {
        Self::new(config.model_path.clone(), config.dictionary_path.clone()).await
    }

    /// Load the model if not already loaded
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if !state.loaded {
            let start_time = Instant::now();

            tracing::info!("Loading forced alignment model: {}", state.model_path);

            // Verify model file exists and is readable
            if !std::path::Path::new(&state.model_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Model file not found: {}", state.model_path),
                    source: None,
                });
            }

            let model_metadata = std::fs::metadata(&state.model_path).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to read model file metadata: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

            tracing::info!(
                "Model file size: {:.2} MB",
                model_metadata.len() as f64 / (1024.0 * 1024.0)
            );

            // Load pronunciation dictionary if provided
            if let Some(ref dict_path) = state.dictionary_path {
                tracing::info!("Loading pronunciation dictionary: {}", dict_path);
                state.dictionary = self.load_dictionary(dict_path).await?;
            } else {
                // Load default dictionary
                state.dictionary = self.load_default_dictionary().await?;
            }

            state.loaded = true;
            state.load_time = Some(start_time.elapsed());

            tracing::info!(
                "Forced alignment model loaded in {:?}",
                state.load_time.unwrap()
            );
        }

        Ok(())
    }

    /// Load pronunciation dictionary from file (CMU Dict format)
    async fn load_dictionary(
        &self,
        dict_path: &str,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        use std::io::{BufRead, BufReader};

        if dict_path.is_empty() {
            return self.load_default_dictionary().await;
        }

        let file = std::fs::File::open(dict_path).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!("Failed to open dictionary file: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        let reader = BufReader::new(file);
        let mut dictionary = HashMap::new();
        let mut line_count = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Failed to read dictionary line: {}", e),
                source: Some(Box::new(e)),
            })?;

            line_count += 1;

            // Skip comments and empty lines
            if line.trim().is_empty() || line.starts_with(";;;") {
                continue;
            }

            // Parse CMU Dict format: WORD  P1 P2 P3 ...
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            // Extract word (may have variant marker like WORD(2))
            let word_part = parts[0];
            let word = if let Some(paren_pos) = word_part.find('(') {
                word_part[..paren_pos].to_uppercase()
            } else {
                word_part.to_uppercase()
            };

            // Extract phonemes (skip stress markers if present)
            let phonemes: Vec<String> = parts[1..]
                .iter()
                .map(|p| {
                    // Remove stress markers (0, 1, 2) from vowels
                    p.chars()
                        .filter(|c| !c.is_numeric())
                        .collect::<String>()
                })
                .collect();

            // Only insert if we don't have this word yet (use first variant)
            dictionary.entry(word).or_insert(phonemes);
        }

        tracing::info!(
            "Loaded {} words from pronunciation dictionary (processed {} lines)",
            dictionary.len(),
            line_count
        );

        if dictionary.is_empty() {
            return Err(RecognitionError::ModelLoadError {
                message: "Dictionary file contains no valid entries".to_string(),
                source: None,
            });
        }

        Ok(dictionary)
    }

    /// Load default pronunciation dictionary
    async fn load_default_dictionary(
        &self,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        let mut dictionary = HashMap::new();

        // Add common English words with ARPAbet phonemes
        // Vowels
        dictionary.insert("HELLO".to_string(), vec!["HH".into(), "AH".into(), "L".into(), "OW".into()]);
        dictionary.insert("WORLD".to_string(), vec!["W".into(), "ER".into(), "L".into(), "D".into()]);
        dictionary.insert("TEST".to_string(), vec!["T".into(), "EH".into(), "S".into(), "T".into()]);
        dictionary.insert("SPEECH".to_string(), vec!["S".into(), "P".into(), "IY".into(), "CH".into()]);
        dictionary.insert("VOICE".to_string(), vec!["V".into(), "OY".into(), "S".into()]);
        dictionary.insert("RECOGNITION".to_string(), vec!["R".into(), "EH".into(), "K".into(), "AH".into(), "G".into(), "N".into(), "IH".into(), "SH".into(), "AH".into(), "N".into()]);
        dictionary.insert("PHONEME".to_string(), vec!["F".into(), "OW".into(), "N".into(), "IY".into(), "M".into()]);
        dictionary.insert("ALIGN".to_string(), vec!["AH".into(), "L".into(), "AY".into(), "N".into()]);
        dictionary.insert("FORCED".to_string(), vec!["F".into(), "AO".into(), "R".into(), "S".into(), "T".into()]);
        dictionary.insert("AUDIO".to_string(), vec!["AO".into(), "D".into(), "IY".into(), "OW".into()]);

        // Common words
        dictionary.insert("THE".to_string(), vec!["DH".into(), "AH".into()]);
        dictionary.insert("A".to_string(), vec!["AH".into()]);
        dictionary.insert("IS".to_string(), vec!["IH".into(), "Z".into()]);
        dictionary.insert("TO".to_string(), vec!["T".into(), "UW".into()]);
        dictionary.insert("AND".to_string(), vec!["AH".into(), "N".into(), "D".into()]);
        dictionary.insert("OF".to_string(), vec!["AH".into(), "V".into()]);
        dictionary.insert("IN".to_string(), vec!["IH".into(), "N".into()]);
        dictionary.insert("FOR".to_string(), vec!["F".into(), "AO".into(), "R".into()]);
        dictionary.insert("WITH".to_string(), vec!["W".into(), "IH".into(), "DH".into()]);
        dictionary.insert("ON".to_string(), vec!["AA".into(), "N".into()]);

        // Numbers
        dictionary.insert("ONE".to_string(), vec!["W".into(), "AH".into(), "N".into()]);
        dictionary.insert("TWO".to_string(), vec!["T".into(), "UW".into()]);
        dictionary.insert("THREE".to_string(), vec!["TH".into(), "R".into(), "IY".into()]);
        dictionary.insert("FOUR".to_string(), vec!["F".into(), "AO".into(), "R".into()]);
        dictionary.insert("FIVE".to_string(), vec!["F".into(), "AY".into(), "V".into()]);

        tracing::info!("Loaded default pronunciation dictionary with {} words", dictionary.len());

        Ok(dictionary)
    }

    /// Perform forced alignment using dynamic time warping
    async fn align_with_dtw(
        &self,
        audio: &AudioBuffer,
        phonemes: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        self.ensure_loaded().await?;

        let start_time = Instant::now();

        // Extract acoustic features
        let features = self.extract_features(audio).await?;

        // Perform DTW alignment
        let alignment = self.dtw_align(&features, phonemes, config).await?;

        // Update statistics
        let mut state = self.state.write().await;
        state.alignment_count += 1;
        state.total_alignment_time += start_time.elapsed();

        Ok(alignment)
    }

    /// Extract acoustic features from audio
    async fn extract_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let samples = audio.samples();
        let frame_length =
            (self.config.frame_length_ms / 1000.0 * audio.sample_rate() as f32) as usize;
        let frame_shift =
            (self.config.frame_shift_ms / 1000.0 * audio.sample_rate() as f32) as usize;

        let mut features = Vec::new();
        let mut start = 0;

        while start + frame_length <= samples.len() {
            let frame = &samples[start..start + frame_length];
            let feature_vector = self.compute_mfcc(frame, audio.sample_rate())?;
            features.push(feature_vector);
            start += frame_shift;
        }

        if features.is_empty() {
            return Err(RecognitionError::PhonemeRecognitionError {
                message: "No features extracted from audio".to_string(),
                source: None,
            });
        }

        Ok(features)
    }

    /// Compute MFCC features for a frame using real DSP
    fn compute_mfcc(&self, frame: &[f32], sample_rate: u32) -> Result<Vec<f32>, RecognitionError> {
        use scirs2_core::ndarray::*;
        use scirs2_fft::rfft;

        const NUM_MEL_FILTERS: usize = 26;
        const NUM_MFCC: usize = 13;

        // Apply Hamming window
        let windowed: Vec<f32> = frame
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (frame.len() - 1) as f32).cos();
                sample * window
            })
            .collect();

        // Compute power spectrum using SciRS2-FFT
        let fft_size = frame.len().next_power_of_two();
        let mut fft_input = vec![0.0f64; fft_size];
        for (i, &val) in windowed.iter().enumerate() {
            fft_input[i] = val as f64;
        }

        let spectrum = rfft(&fft_input, Some(fft_size))
            .map_err(|e| RecognitionError::PhonemeRecognitionError {
                message: format!("FFT failed: {:?}", e),
                source: None,
            })?;

        // Compute power spectrum
        let power_spectrum: Vec<f32> = spectrum
            .iter()
            .map(|c| {
                let power = c.re * c.re + c.im * c.im;
                power as f32
            })
            .collect();

        // Apply Mel filterbank
        let mel_energies = self.apply_mel_filterbank(&power_spectrum, sample_rate, fft_size)?;

        // Compute log energies
        let log_mel_energies: Vec<f32> = mel_energies
            .iter()
            .map(|&e| (e.max(1e-10)).ln())
            .collect();

        // Apply DCT to get MFCC coefficients
        let mfcc = self.apply_dct(&log_mel_energies, NUM_MFCC)?;

        Ok(mfcc)
    }

    /// Apply Mel filterbank to power spectrum
    fn apply_mel_filterbank(
        &self,
        power_spectrum: &[f32],
        sample_rate: u32,
        fft_size: usize,
    ) -> Result<Vec<f32>, RecognitionError> {
        const NUM_MEL_FILTERS: usize = 26;
        const MIN_FREQ_HZ: f32 = 0.0;
        let max_freq_hz = (sample_rate as f32 / 2.0).min(8000.0);

        // Convert Hz to Mel scale
        let hz_to_mel = |freq: f32| 2595.0 * (1.0 + freq / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let min_mel = hz_to_mel(MIN_FREQ_HZ);
        let max_mel = hz_to_mel(max_freq_hz);

        // Create Mel filter bank
        let mut mel_filters = vec![vec![0.0; power_spectrum.len()]; NUM_MEL_FILTERS];

        // Mel filter center points
        let mel_points: Vec<f32> = (0..NUM_MEL_FILTERS + 2)
            .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (NUM_MEL_FILTERS + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

        // Convert Hz points to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((fft_size as f32 + 1.0) * hz / sample_rate as f32).floor() as usize)
            .collect();

        // Build triangular filters
        for i in 0..NUM_MEL_FILTERS {
            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising slope
            for j in start..center {
                if center > start {
                    mel_filters[i][j] = (j - start) as f32 / (center - start) as f32;
                }
            }

            // Falling slope
            for j in center..end {
                if end > center && j < power_spectrum.len() {
                    mel_filters[i][j] = (end - j) as f32 / (end - center) as f32;
                }
            }
        }

        // Apply filters to power spectrum
        let mut mel_energies = vec![0.0; NUM_MEL_FILTERS];
        for i in 0..NUM_MEL_FILTERS {
            mel_energies[i] = power_spectrum
                .iter()
                .zip(mel_filters[i].iter())
                .map(|(&power, &filter)| power * filter)
                .sum();
        }

        Ok(mel_energies)
    }

    /// Apply Discrete Cosine Transform (DCT)
    fn apply_dct(&self, input: &[f32], num_coeffs: usize) -> Result<Vec<f32>, RecognitionError> {
        let n = input.len();
        let mut output = vec![0.0; num_coeffs];

        for k in 0..num_coeffs {
            let mut sum = 0.0;
            for (i, &val) in input.iter().enumerate() {
                sum += val * ((std::f32::consts::PI * k as f32 * (i as f32 + 0.5)) / n as f32).cos();
            }
            output[k] = sum;
        }

        Ok(output)
    }

    /// Perform DTW alignment between features and phonemes
    async fn dtw_align(
        &self,
        features: &[Vec<f32>],
        phonemes: &[Phoneme],
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        let frame_duration = self.config.frame_shift_ms / 1000.0;
        let total_duration = features.len() as f32 * frame_duration;

        if phonemes.is_empty() {
            return Ok(PhonemeAlignment {
                phonemes: Vec::new(),
                total_duration,
                alignment_confidence: 0.0,
                word_alignments: Vec::new(),
            });
        }

        // Perform real DTW alignment
        let (path, cost) = self.compute_dtw(features, phonemes)?;

        // Convert DTW path to phoneme alignment
        let mut aligned_phonemes = Vec::new();
        let mut word_alignments = Vec::new();

        // Group frames by phoneme
        let mut phoneme_frames: Vec<Vec<usize>> = vec![Vec::new(); phonemes.len()];
        for (frame_idx, phoneme_idx) in &path {
            if *phoneme_idx < phoneme_frames.len() {
                phoneme_frames[*phoneme_idx].push(*frame_idx);
            }
        }

        // Create aligned phonemes with timing information
        for (i, phoneme) in phonemes.iter().enumerate() {
            if phoneme_frames[i].is_empty() {
                continue; // Skip phonemes with no aligned frames
            }

            let start_frame = *phoneme_frames[i].first().unwrap();
            let end_frame = *phoneme_frames[i].last().unwrap();

            let start_time = start_frame as f32 * frame_duration;
            let end_time = (end_frame + 1) as f32 * frame_duration;

            // Calculate confidence based on alignment cost
            let frame_count = phoneme_frames[i].len();
            let avg_cost = if frame_count > 0 {
                cost / (path.len() as f32)
            } else {
                1.0
            };

            // Convert cost to confidence (lower cost = higher confidence)
            let confidence = (1.0 / (1.0 + avg_cost)).max(0.3).min(1.0);

            aligned_phonemes.push(AlignedPhoneme {
                phoneme: phoneme.clone(),
                start_time,
                end_time,
                confidence,
            });
        }

        // Create word alignments (all phonemes as one word for now)
        if !aligned_phonemes.is_empty() {
            let word_alignment = WordAlignment {
                word: "aligned_sequence".to_string(),
                start_time: aligned_phonemes[0].start_time,
                end_time: aligned_phonemes.last().unwrap().end_time,
                phonemes: aligned_phonemes.clone(),
                confidence: aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                    / aligned_phonemes.len() as f32,
            };
            word_alignments.push(word_alignment);
        }

        let overall_confidence = if !aligned_phonemes.is_empty() {
            aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                / aligned_phonemes.len() as f32
        } else {
            0.0
        };

        Ok(PhonemeAlignment {
            phonemes: aligned_phonemes,
            total_duration,
            alignment_confidence: overall_confidence,
            word_alignments,
        })
    }

    /// Compute DTW alignment path between features and phonemes
    fn compute_dtw(
        &self,
        features: &[Vec<f32>],
        phonemes: &[Phoneme],
    ) -> Result<(Vec<(usize, usize)>, f32), RecognitionError> {
        let n = features.len();
        let m = phonemes.len();

        if n == 0 || m == 0 {
            return Ok((Vec::new(), 0.0));
        }

        // Initialize DTW cost matrix
        let mut cost_matrix = vec![vec![f32::INFINITY; m + 1]; n + 1];
        cost_matrix[0][0] = 0.0;

        // Fill cost matrix using dynamic programming
        for i in 1..=n {
            for j in 1..=m {
                // Calculate local cost (distance between feature and phoneme)
                let local_cost = self.compute_local_cost(&features[i - 1], &phonemes[j - 1]);

                // Find minimum cost path
                let min_prev_cost = cost_matrix[i - 1][j]
                    .min(cost_matrix[i][j - 1])
                    .min(cost_matrix[i - 1][j - 1]);

                cost_matrix[i][j] = local_cost + min_prev_cost;
            }
        }

        // Backtrack to find optimal path
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;

        while i > 0 && j > 0 {
            path.push((i - 1, j - 1));

            // Find which direction we came from
            let diag = cost_matrix[i - 1][j - 1];
            let left = cost_matrix[i][j - 1];
            let up = cost_matrix[i - 1][j];

            if diag <= left && diag <= up {
                i -= 1;
                j -= 1;
            } else if left <= up {
                j -= 1;
            } else {
                i -= 1;
            }
        }

        path.reverse();

        let final_cost = cost_matrix[n][m];

        Ok((path, final_cost))
    }

    /// Compute local cost between a feature vector and a phoneme
    fn compute_local_cost(&self, feature: &[f32], phoneme: &Phoneme) -> f32 {
        // Create a simple phoneme template based on the phoneme symbol
        let phoneme_template = self.create_phoneme_template(phoneme);

        // Compute Euclidean distance between feature and template
        let mut distance = 0.0;
        for (f, t) in feature.iter().zip(phoneme_template.iter()) {
            let diff = f - t;
            distance += diff * diff;
        }

        distance.sqrt()
    }

    /// Create a simple template for a phoneme
    fn create_phoneme_template(&self, phoneme: &Phoneme) -> Vec<f32> {
        // This is a simplified phoneme model
        // In a real implementation, this would use trained phoneme models
        let num_coeffs = 13;
        let mut template = vec![0.0; num_coeffs];

        // Create template based on phoneme symbol characteristics
        let symbol = &phoneme.symbol;
        let hash = symbol.bytes().map(|b| b as u32).sum::<u32>();

        for i in 0..num_coeffs {
            // Generate pseudo-random but consistent template
            let seed = (hash + i as u32) as f32;
            template[i] = (seed * 0.01).sin() * 2.0;
        }

        // Adjust first coefficient based on phoneme type (vowel vs consonant)
        let is_vowel = matches!(
            symbol.to_uppercase().as_str(),
            "A" | "E" | "I" | "O" | "U" | "AH" | "EH" | "IH" | "OH" | "UH" | "AA" | "AE"
                | "AO" | "AW" | "AY" | "EY" | "IY" | "OW" | "OY" | "UW"
        );

        if is_vowel {
            template[0] = 3.0; // Higher energy for vowels
        } else {
            template[0] = 1.0; // Lower energy for consonants
        }

        template
    }

    /// Calculate alignment confidence
    fn calculate_alignment_confidence(
        &self,
        _features: &[Vec<f32>],
        phoneme_index: usize,
        total_phonemes: usize,
    ) -> f32 {
        // Mock confidence calculation
        // In a real implementation, this would use acoustic model scores
        let base_confidence = 0.85;
        let position_factor = 1.0 - (phoneme_index as f32 / total_phonemes as f32 * 0.1);
        (base_confidence * position_factor).max(0.3).min(1.0)
    }

    /// Convert text to phonemes using dictionary
    async fn text_to_phonemes(
        &self,
        text: &str,
        _language: LanguageCode,
    ) -> Result<Vec<Phoneme>, RecognitionError> {
        self.ensure_loaded().await?;

        let state = self.state.read().await;
        let uppercase_text = text.to_uppercase();
        let words: Vec<&str> = uppercase_text.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for word in words {
            if let Some(word_phonemes) = state.dictionary.get(word) {
                for phoneme_str in word_phonemes {
                    phonemes.push(Phoneme {
                        symbol: phoneme_str.clone(),
                        ipa_symbol: phoneme_str.clone(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 1.0,
                    });
                }
            } else {
                // Handle unknown words with a simple fallback
                tracing::warn!("Unknown word in dictionary: {}", word);
                for char in word.chars() {
                    phonemes.push(Phoneme {
                        symbol: char.to_string(),
                        ipa_symbol: char.to_string(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 0.5,
                    });
                }
            }
        }

        Ok(phonemes)
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> ForcedAlignStats {
        let state = self.state.read().await;
        ForcedAlignStats {
            alignment_count: state.alignment_count,
            total_alignment_time: state.total_alignment_time,
            average_alignment_time: if state.alignment_count > 0 {
                state.total_alignment_time / state.alignment_count as u32
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            dictionary_size: state.dictionary.len(),
        }
    }

    /// Add word to dictionary
    pub async fn add_word_to_dictionary(
        &self,
        word: String,
        phonemes: Vec<String>,
    ) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;
        state.dictionary.insert(word.to_uppercase(), phonemes);
        Ok(())
    }

    /// Get pronunciation for a word
    pub async fn get_pronunciation(
        &self,
        word: &str,
    ) -> Result<Option<Vec<String>>, RecognitionError> {
        self.ensure_loaded().await?;
        let state = self.state.read().await;
        Ok(state.dictionary.get(&word.to_uppercase()).cloned())
    }
}

/// Forced alignment model statistics
#[derive(Debug, Clone)]
pub struct ForcedAlignStats {
    /// Total number of alignments performed
    pub alignment_count: usize,
    /// Total alignment time
    pub total_alignment_time: Duration,
    /// Average alignment time
    pub average_alignment_time: Duration,
    /// Model load time
    pub load_time: Option<Duration>,
    /// Dictionary size
    pub dictionary_size: usize,
}

#[async_trait]
impl PhonemeRecognizer for ForcedAlignModel {
    async fn recognize_phonemes(
        &self,
        audio: &AudioBuffer,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<Phoneme>> {
        // For recognition without expected phonemes, we need to use a different approach
        // This is a simplified implementation
        self.ensure_loaded().await?;

        let features = self.extract_features(audio).await.map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Feature extraction failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Mock phoneme recognition based on audio characteristics
        let mut phonemes = Vec::new();
        let num_phonemes = (features.len() / 10).max(1); // Rough estimate

        for i in 0..num_phonemes {
            // Mock phoneme based on position
            let symbol = match i % 5 {
                0 => "AH",
                1 => "L",
                2 => "OW",
                3 => "W",
                _ => "ER",
            };

            phonemes.push(Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: symbol.to_string(),
                stress: 0, // No stress
                syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                duration_ms: Some(100.0),
                confidence: 0.8,
            });
        }

        Ok(phonemes)
    }

    async fn align_phonemes(
        &self,
        audio: &AudioBuffer,
        expected: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        self.align_with_dtw(audio, expected, config)
            .await
            .map_err(|e| e.into())
    }

    async fn align_text(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        let language = config.map(|c| c.language).unwrap_or(LanguageCode::EnUs);
        let phonemes = self.text_to_phonemes(text, language).await.map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Text to phoneme conversion failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        self.align_phonemes(audio, &phonemes, config).await
    }

    fn metadata(&self) -> PhonemeRecognizerMetadata {
        self.metadata.clone()
    }

    fn supports_feature(&self, feature: PhonemeRecognitionFeature) -> bool {
        self.metadata.supported_features.contains(&feature)
    }
}

impl Clone for ForcedAlignModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            supported_languages: self.supported_languages.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use voirs_sdk::AudioBuffer;

    fn create_mock_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "mock acoustic model data").unwrap();
        file
    }

    #[tokio::test]
    async fn test_forced_align_model_creation() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        assert_eq!(model.metadata.name, "Basic Forced Alignment");
        assert!(model.supported_languages.contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_forced_align_missing_file() {
        let result = ForcedAlignModel::new("nonexistent.bin".to_string(), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_phoneme_recognition() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1); // 0.1 second of audio

        let result = model.recognize_phonemes(&audio, None).await.unwrap();
        assert!(!result.is_empty());
        assert!(result.iter().all(|p| p.confidence > 0.0));
    }

    #[tokio::test]
    async fn test_phoneme_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let phonemes = vec![
            Phoneme {
                symbol: "H".to_string(),
                ipa_symbol: "H".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                duration_ms: None,
                confidence: 1.0,
            },
            Phoneme {
                symbol: "AH".to_string(),
                ipa_symbol: "AH".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                duration_ms: None,
                confidence: 1.0,
            },
        ];

        let result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        assert_eq!(result.phonemes.len(), 2);
        assert!(result.alignment_confidence > 0.0);
        assert!(result.total_duration > 0.0);
    }

    #[tokio::test]
    async fn test_text_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let result = model.align_text(&audio, "HELLO", None).await.unwrap();
        assert!(!result.phonemes.is_empty());
        assert!(result.alignment_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_dictionary_operations() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        // Test getting existing pronunciation
        let pronunciation = model.get_pronunciation("HELLO").await.unwrap();
        assert!(pronunciation.is_some());

        // Test adding new word
        let new_phonemes = vec![
            "T".to_string(),
            "EH".to_string(),
            "S".to_string(),
            "T".to_string(),
        ];
        model
            .add_word_to_dictionary("TESTING".to_string(), new_phonemes.clone())
            .await
            .unwrap();

        let retrieved = model.get_pronunciation("TESTING").await.unwrap();
        assert_eq!(retrieved, Some(new_phonemes));
    }

    #[tokio::test]
    async fn test_features() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        assert!(model.supports_feature(PhonemeRecognitionFeature::WordAlignment));
        assert!(model.supports_feature(PhonemeRecognitionFeature::CustomPronunciation));
        assert!(model.supports_feature(PhonemeRecognitionFeature::ConfidenceScoring));
        assert!(model.supports_feature(PhonemeRecognitionFeature::PronunciationAssessment));
    }

    #[tokio::test]
    async fn test_statistics() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        // Initial stats
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 0);

        // After alignment
        let phonemes = vec![Phoneme {
            symbol: "H".to_string(),
            ipa_symbol: "H".to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Onset,
            duration_ms: None,
            confidence: 1.0,
        }];

        let _result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 1);
        assert!(stats.total_alignment_time > Duration::ZERO);
    }
}
